"""
Supertonic TTS — ComfyUI Custom Nodes
Supertonic's lightning-fast, on-device, multilingual Text-to-Speech system.
"""

import os
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


def numpy_to_comfy_audio(wav: np.ndarray, sr: int) -> dict:
    wav_3d = wav[:, np.newaxis, :]
    tensor = torch.from_numpy(wav_3d.copy()).float()
    return {"waveform": tensor, "sample_rate": sr}


def apply_pitch_shift(wav_tensor: torch.Tensor, sample_rate: int, pitch_semitones: float) -> torch.Tensor:
    if abs(pitch_semitones) < 0.5:
        return wav_tensor
    if wav_tensor.ndim == 1:
        wav_tensor = wav_tensor.unsqueeze(0)
    try:
        return torchaudio.functional.pitch_shift(
            waveform=wav_tensor, sample_rate=sample_rate,
            n_steps=pitch_semitones, bins_per_octave=12, n_fft=2048,
        )
    except Exception as e:
        print(f"  [Warn] pitch_shift failed ({e}), returning original")
        return wav_tensor


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
                    "placeholder": "Enter text to speak...",
                }),
                "language": (SUPPORTED_LANGUAGES, {"default": "en"}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.7, "max": 2.0, "step": 0.05, "display": "slider"}),
                "quality": ("INT", {"default": 8, "min": 5, "max": 12, "step": 1, "display": "slider"}),
                "voice_style": (BUILTIN_VOICES, {"default": "M1"}),
            },
            "optional": {
                "custom_style_path": ("STRING", {"default": "", "multiline": False}),
                "pitch_semitones": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.5, "display": "slider"}),
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
        pitch_semitones: float = 0.0,
    ) -> tuple:
        if not text or not text.strip():
            print("[SupertonicTTS] Empty text — returning silence")
            return self._silence(model["sample_rate"])

        tts_obj = model["tts"]

        if custom_style_path and os.path.isfile(custom_style_path):
            style = tts_obj.get_voice_style_from_path(custom_style_path)
        else:
            style = tts_obj.get_voice_style(voice_style)

        print(f"\n=== SupertonicTTS ===")
        print(f"  Language: {language} | Speed: {speed} | Quality: {quality} | Voice: {voice_style}")
        print(f"  Text: {text[:120]}")

        try:
            wav_np, duration = tts_obj.synthesize(
                text=text, voice_style=style, lang=language,
                speed=speed, total_steps=quality, verbose=True,
            )
            dur_sec = float(duration[0]) if hasattr(duration, "__len__") else float(duration)
            print(f"  Generated {dur_sec:.2f}s ({wav_np.shape[-1]} samples)")

            wav_t = torch.from_numpy(wav_np.copy()).float()
            if abs(pitch_semitones) >= 0.5:
                wav_t = apply_pitch_shift(wav_t, tts_obj.sample_rate, pitch_semitones)

            wav_np = wav_t.cpu().numpy()
            if wav_np.ndim == 1:
                wav_np = wav_np[np.newaxis, :]

            return (numpy_to_comfy_audio(wav_np, tts_obj.sample_rate),)

        except Exception as e:
            import traceback
            print(f"[SupertonicTTS] ERROR — {e}")
            traceback.print_exc()
            return self._silence(tts_obj.sample_rate)

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