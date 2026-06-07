"""
Supertonic TTS — ComfyUI Custom Nodes
Supertonic's lightning-fast, on-device, multilingual Text-to-Speech system.

Expression Tags (type directly into text):
  <laugh> <breath> <surprise> <sigh> <scream>
  <throatclear> <sad> <angry> <cough> <yawn>
"""

import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio


def get_device() -> torch.device:
    if torch.version.hip:
        return torch.device("hip")
    return torch.device("cpu")


DEVICE = get_device()

try:
    from supertonic import TTS
    from supertonic import AVAILABLE_LANGUAGES as _SDK_LANGS
except ImportError:
    TTS = None
    _SDK_LANGS = []

_NODE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = _NODE_DIR / "models"


def ensure_models_dir():
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)


ensure_models_dir()
os.environ["HF_HOME"] = str(_MODELS_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(_MODELS_DIR / "hub")

SUPPORTED_LANGUAGES = _SDK_LANGS if _SDK_LANGS else [
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el",
    "es", "et", "fi", "fr", "hi", "hr", "hu", "id", "it",
    "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl",
    "sv", "tr", "uk", "vi", "na",
]

BUILTIN_VOICES = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]

# Expression tags supported by the Supertonic SDK
EXPRESSION_TAGS = [
    "<laugh>", "<breath>", "<surprise>", "<sigh>", "<scream>",
    "<throatclear>", "<sad>", "<angry>", "<cough>", "<yawn>",
]


def numpy_to_comfy_audio(wav: np.ndarray, sr: int) -> dict:
    wav_3d = wav[:, np.newaxis, :]
    tensor = torch.from_numpy(wav_3d.copy()).float()
    return {"waveform": tensor, "sample_rate": sr}


def comfy_audio_to_numpy(audio: dict) -> tuple:
    wav = audio["waveform"]
    sr = audio["sample_rate"]
    if isinstance(wav, torch.Tensor):
        wav = wav.cpu().numpy()
    while wav.ndim > 1:
        wav = wav.squeeze(0)
    wav = wav.astype(np.float32)
    peak = np.max(np.abs(wav))
    if peak > 1.0:
        wav = wav / 32768.0
    return wav, sr


def normalize_audio(wav: np.ndarray, peak_target: float = 0.95) -> np.ndarray:
    wav = np.ascontiguousarray(wav).astype(np.float32)
    peak = np.max(np.abs(wav))
    if peak > 1.0:
        wav = wav / 32768.0
        peak = np.max(np.abs(wav))
    if peak > 1e-8:
        wav = wav / peak * peak_target
    return np.clip(wav, -1.0, 1.0).astype(np.float32)


def apply_postprocessing(
    wav_np: np.ndarray,
    sr: int,
    trim_silence: bool,
    normalize_vol: bool,
    clarity_boost: bool,
    pitch_semitones: float,
    time_stretch: float,
    chorus_effect: bool,
) -> np.ndarray:
    """Apply librosa post-processing to the audio waveform (matching WebUI behavior)."""
    try:
        import librosa
    except ImportError:
        print("  [Warn] librosa not installed — skipping post-processing effects")
        return wav_np

    effects_applied = []

    # Trim silence
    if trim_silence:
        wav_np, _ = librosa.effects.trim(wav_np, top_db=25)
        effects_applied.append("trim")

    # Clarity boost (preemphasis)
    if clarity_boost:
        wav_np = librosa.effects.preemphasis(wav_np, coef=0.97)
        effects_applied.append("clarity")

    # Pitch shift
    if abs(pitch_semitones) >= 0.5:
        wav_np = librosa.effects.pitch_shift(y=wav_np, sr=sr, n_steps=float(pitch_semitones))
        effects_applied.append(f"pitch({pitch_semitones:+d}st)")

    # Time stretch
    if abs(time_stretch - 1.0) > 0.01:
        wav_np = librosa.effects.time_stretch(y=wav_np, rate=time_stretch)
        effects_applied.append(f"stretch({time_stretch:.2f}x)")

    # Chorus effect: pitch-shifted delayed mix
    if chorus_effect:
        wav_ps = librosa.effects.pitch_shift(y=wav_np, sr=sr, n_steps=-2)
        wav_delayed = np.pad(wav_ps, (int(sr * 0.03), 0), mode="constant")
        min_len = min(len(wav_np), len(wav_delayed))
        wav_np = (wav_np[:min_len] * 0.7) + (wav_delayed[:min_len] * 0.5)
        effects_applied.append("chorus")

    # Normalize volume
    if normalize_vol:
        wav_np = librosa.util.normalize(wav_np)
        effects_applied.append("normalize")

    if effects_applied:
        print(f"  Post-processing: {', '.join(effects_applied)}")

    return wav_np


_TTS_INSTANCE: Optional[TTS] = None


def get_tts(auto_download: bool = True) -> TTS:
    global _TTS_INSTANCE
    if _TTS_INSTANCE is None:
        if TTS is None:
            raise ImportError("supertonic package not found. Install: pip install supertonic")
        ensure_models_dir()
        os.environ["HF_HOME"] = str(_MODELS_DIR)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(_MODELS_DIR / "hub")
        print("[Supertonic] Initialising TTS engine ...")
        _TTS_INSTANCE = TTS(auto_download=auto_download)
        print(f"[Supertonic] Ready — {_TTS_INSTANCE.sample_rate} Hz")
    return _TTS_INSTANCE


class SupertonicLoader:
    CATEGORY = "Audio/Supertonic"
    RETURN_TYPES = ("SUPERTONIC_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def load(self) -> tuple:
        print("\n=== SupertonicLoader ===")
        tts = get_tts(auto_download=True)
        print(f"  Engine ready — {tts.sample_rate} Hz\n")
        return ({"tts": tts, "sample_rate": tts.sample_rate},)


class SupertonicTTS:
    CATEGORY = "Audio/Supertonic"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "synthesize"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SUPERTONIC_MODEL",),
                "text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Enter text with optional expression tags...\nTags: <laugh> <breath> <surprise> <sigh> <scream> <throatclear> <sad> <angry> <cough> <yawn>\nExample: Hello! <laugh> That was funny!",
                }),
                "language": (SUPPORTED_LANGUAGES, {"default": "en"}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05, "display": "slider"}),
                "quality": ("INT", {"default": 8, "min": 5, "max": 12, "step": 1, "display": "slider"}),
                "voice_style": (BUILTIN_VOICES, {"default": "M1"}),
            },
            "optional": {
                "custom_style_path": ("STRING", {"default": "", "multiline": False}),
                "trim_silence": ("BOOLEAN", {"default": True}),
                "normalize_volume": ("BOOLEAN", {"default": True}),
                "clarity_boost": ("BOOLEAN", {"default": False}),
                "pitch_semitones": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 1.0, "display": "slider"}),
                "time_stretch": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05, "display": "slider"}),
                "chorus_effect": ("BOOLEAN", {"default": False}),
            },
        }

    def synthesize(
        self,
        model: dict,
        text: str,
        language: str = "en",
        speed: float = 1.0,
        quality: int = 8,
        voice_style: str = "M1",
        custom_style_path: str = "",
        trim_silence: bool = True,
        normalize_volume: bool = True,
        clarity_boost: bool = False,
        pitch_semitones: float = 0.0,
        time_stretch: float = 1.0,
        chorus_effect: bool = False,
    ) -> tuple:
        if not text or not text.strip():
            print("[SupertonicTTS] Empty text — returning silence")
            return self._silence(model["sample_rate"])

        tts_obj = model["tts"]
        sr = tts_obj.sample_rate

        # Detect expression tags in text
        found_tags = [tag for tag in EXPRESSION_TAGS if tag in text]
        if found_tags:
            print(f"  Expression tags detected: {', '.join(found_tags)}")

        if custom_style_path and os.path.isfile(custom_style_path):
            style = tts_obj.get_voice_style_from_path(custom_style_path)
        else:
            style = tts_obj.get_voice_style(voice_style)

        print(f"\n=== SupertonicTTS ===")
        print(f"  Language: {language} | Speed: {speed} | Quality: {quality} | Voice: {voice_style}")
        if found_tags:
            print(f"  Tags: {', '.join(found_tags)}")
        print(f"  Text: {text[:120]}")

        try:
            wav_np, duration = tts_obj.synthesize(
                text=text, voice_style=style, lang=language,
                speed=speed, total_steps=quality, verbose=True,
            )
            dur_sec = float(duration[0]) if hasattr(duration, "__len__") else float(duration)
            print(f"  Generated {dur_sec:.2f}s ({wav_np.shape[-1]} samples @ {sr} Hz)")

            # Convert to 1D numpy for librosa processing
            if isinstance(wav_np, list):
                wav_np = np.array(wav_np, dtype=np.float32)
            if wav_np.ndim > 1:
                wav_np = wav_np.flatten()

            # Apply post-processing (expression tag effects from WebUI)
            wav_np = apply_postprocessing(
                wav_np, sr,
                trim_silence=trim_silence,
                normalize_vol=normalize_volume,
                clarity_boost=clarity_boost,
                pitch_semitones=pitch_semitones,
                time_stretch=time_stretch,
                chorus_effect=chorus_effect,
            )

            # Ensure 2D shape for ComfyUI [1, samples]
            if wav_np.ndim == 1:
                wav_np = wav_np[np.newaxis, :]

            return (numpy_to_comfy_audio(wav_np.astype(np.float32), sr),)

        except Exception as e:
            import traceback
            print(f"[SupertonicTTS] ERROR — {e}")
            traceback.print_exc()
            return self._silence(sr)

    @staticmethod
    def _silence(sr: int, dur: float = 1.0) -> tuple:
        return (numpy_to_comfy_audio(np.zeros((1, int(sr * dur)), dtype=np.float32), sr),)


NODE_CLASS_MAPPINGS = {
    "SupertonicLoader": SupertonicLoader,
    "SupertonicTTS":    SupertonicTTS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SupertonicLoader": "Supertonic Model Loader 🎤",
    "SupertonicTTS":    "Supertonic Text-to-Speech 🗣️",
}