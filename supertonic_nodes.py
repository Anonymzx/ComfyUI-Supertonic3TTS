"""
Supertonic TTS — ComfyUI Custom Nodes
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
import soundfile as sf

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
RVC_TARGET_SR = 40000

def strip_path(s: str) -> str:
    return s.strip().strip("'\" \t").rstrip("\\/")

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
    return wav[np.newaxis, :], sr

def normalize_audio(wav: np.ndarray, peak_target: float = 0.95) -> np.ndarray:
    wav = np.ascontiguousarray(wav).astype(np.float32)
    peak = np.max(np.abs(wav))
    if peak > 1.0:
        wav = wav / 32768.0
        peak = np.max(np.abs(wav))
    if peak > 1e-8:
        wav = wav / peak * peak_target
    return np.clip(wav, -1.0, 1.0).astype(np.float32)

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


class RVCEngine:
    def __init__(self, model_path: str, device: torch.device = DEVICE):
        self.device = device
        self.model_path = model_path
        self.net_g = None
        self.hubert = None
        self.hubert_processor = None
        self.hubert_type = None
        self.tgt_sr = 40000
        self.version = "v2"
        self.if_f0 = 1
        self._rvc_impl_path = Path(__file__).parent
        self._load()

    def _load(self):
        print(f"\n=== RVC Engine ===")
        print(f"  Device: {self.device}")
        print(f"  Model:  {self.model_path}")

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"RVC model not found: {self.model_path}")

        ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
        self.version = ckpt.get("version", "v2")
        self.if_f0   = ckpt.get("f0", 1)
        self.tgt_sr  = ckpt.get("sr", 40000)
        cfg          = ckpt.get("config", None)
        weights      = ckpt.get("weight", None)
        print(f"  Version: {self.version} | sr: {self.tgt_sr} | f0: {self.if_f0}")
        print(f"  Config length: {len(cfg) if cfg is not None else 'None'}")
        print(f"  Config values: {cfg}")

        self._build_net_g(weights, cfg)
        self._load_hubert()
        print(f"  RVC Engine ready\n")

    def _build_net_g(self, weights, cfg):
        rvc_impl = str(self._rvc_impl_path)
        if rvc_impl not in sys.path:
            sys.path.insert(0, rvc_impl)

        try:
            from lib.infer_pack.models import (
                SynthesizerTrnMs768NSFsid,
                SynthesizerTrnMs256NSFsid,
                SynthesizerTrnMs768NSFsid_nono,
                SynthesizerTrnMs256NSFsid_nono,
            )

            if self.version == "v2":
                Cls = SynthesizerTrnMs768NSFsid if self.if_f0 else SynthesizerTrnMs768NSFsid_nono
            else:
                Cls = SynthesizerTrnMs256NSFsid if self.if_f0 else SynthesizerTrnMs256NSFsid_nono

            cfg_list = list(cfg)
            sr = self.tgt_sr
            variants = [
                cfg_list,
                cfg_list[:-1],
                cfg_list[:-2],
                cfg_list + [sr],
                cfg_list[:-1] + [sr],
            ]

            for args in variants:
                try:
                    net = Cls(*args, is_half=False)
                    net.eval()
                    net.load_state_dict(weights, strict=False)
                    self.net_g = net.float()
                    print(f"  SynthesizerTrn loaded OK ({Cls.__name__}) args_len={len(args)}")
                    return
                except TypeError as te:
                    print(f"  args_len={len(args)} failed: {te}")
                    continue
                except Exception as e:
                    print(f"  args_len={len(args)} error: {e}")
                    continue

            print(f"  [RVC] All cfg variants failed")
            self.net_g = None

        except Exception as e:
            import traceback
            print(f"  [RVC] net_g import failed: {e}")
            traceback.print_exc()
            self.net_g = None

    def _load_hubert(self):
        rvc_impl = str(self._rvc_impl_path)
        if rvc_impl not in sys.path:
            sys.path.insert(0, rvc_impl)

        search_dirs = [
            Path("F:/ComfyUI/custom_nodes/tts_audio_suite"),
            Path("F:/ComfyUI/models"),
            self._rvc_impl_path,
        ]
        hubert_pt = None
        for d in search_dirs:
            if not d.exists():
                continue
            hits = (
                list(d.rglob("hubert_base.pt")) +
                list(d.rglob("hubert-base.pt")) +
                list(d.rglob("checkpoint_best_legacy_500.pt"))
            )
            if hits:
                hubert_pt = str(hits[0])
                print(f"  Found local HuBERT: {Path(hubert_pt).name}")
                break

        if hubert_pt:
            tts_rvc_path = str(Path("F:/ComfyUI/custom_nodes/tts_audio_suite/engines/rvc"))
            if tts_rvc_path not in sys.path:
                sys.path.insert(0, tts_rvc_path)
            try:
                import importlib
                hm = importlib.import_module("hubert_models")
                loader_fn = None
                for fname in ["load_hubert_model", "load_model", "get_hubert_model", "load"]:
                    if hasattr(hm, fname):
                        loader_fn = getattr(hm, fname)
                        break
                if loader_fn:
                    self.hubert = loader_fn(hubert_pt)
                    self.hubert = self.hubert.float().eval()
                    self.hubert_type = "fairseq"
                    print(f"  HuBERT loaded via tts_audio_suite ({fname})")
                    return
                else:
                    print(f"  hubert_models.py: no loader found")
                    print(f"  Available: {[x for x in dir(hm) if not x.startswith('_')]}")
            except Exception as e:
                print(f"  [HuBERT/local] {e}")

        try:
            from transformers import HubertModel, Wav2Vec2FeatureExtractor
            cache = str(_MODELS_DIR / "hub")
            print(f"  Loading HuBERT via transformers (cache: {cache})...")
            self.hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained(
                "facebook/hubert-base-ls960",
                cache_dir=cache,
                local_files_only=False,
            )
            self.hubert = HubertModel.from_pretrained(
                "facebook/hubert-base-ls960",
                cache_dir=cache,
                local_files_only=False,
            ).float().eval()
            self.hubert_type = "transformers"
            print(f"  HuBERT loaded via transformers")
        except Exception as e:
            print(f"  [HuBERT/transformers] {e}")
            self.hubert = None
            self.hubert_type = None

    def infer(
        self,
        audio: np.ndarray,
        pitch_shift: int = 0,
        f0_method: str = "rmvpe",
        index_path: Optional[str] = None,
    ) -> np.ndarray:
        print(f"\n=== RVC Inference ===")
        print(f"  Input:  {audio.shape[-1]} samples @ 40kHz")
        wav_1d = audio[0].astype(np.float32)
        sr = 40000

        f0, f0_coarse = self._extract_f0(wav_1d, sr, f0_method, pitch_shift)
        feats = self._extract_hubert(audio, sr)

        if self.net_g is not None and feats is not None:
            out = self._run_net_g(feats, f0, f0_coarse, index_path)
            if out is not None:
                peak = np.max(np.abs(out))
                if peak > 1e-8:
                    out = out / peak * 0.95
                print(f"  Output: {out.shape[-1]} samples")
                return out[np.newaxis, :]

        print("  [RVC] net_g unavailable — using WORLD vocoder fallback")
        return self._world_fallback(wav_1d, sr, f0)

    def _extract_f0(self, wav_1d, sr, method, pitch_shift):
        import pyworld as pw
        wav_f64 = wav_1d.astype(np.float64)

        if method == "harvest":
            f0, t = pw.harvest(wav_f64, sr, f0_floor=50, f0_ceil=1100, frame_period=10)
        else:
            f0, t = pw.dio(wav_f64, sr, f0_floor=50, f0_ceil=1100, frame_period=10)
            f0 = pw.stonemask(wav_f64, f0, t, sr)

        if pitch_shift != 0:
            voiced = f0 > 0
            f0[voiced] *= 2.0 ** (pitch_shift / 12.0)
            f0 = np.where(f0 > 0, np.clip(f0, 50, 1100), 0)

        f0_mel_min = 1127 * np.log(1 + 50 / 700)
        f0_mel_max = 1127 * np.log(1 + 1100 / 700)
        f0_mel = 1127 * np.log(1 + np.where(f0 > 0, f0, 1e-5) / 700)
        f0_mel = np.where(f0 > 0, (f0_mel - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1, 0)
        f0_mel = np.clip(f0_mel, 0, 255)
        f0_coarse = np.round(f0_mel).astype(np.int32)

        return f0.astype(np.float32), f0_coarse

    def _extract_hubert(self, audio: np.ndarray, sr: int):
        if self.hubert is None:
            return None
        try:
            wav_t = torch.from_numpy(audio.astype(np.float32))
            wav_16k = torchaudio.functional.resample(wav_t, orig_freq=sr, new_freq=16000)

            if self.hubert_type == "transformers":
                wav_np = wav_16k.squeeze().numpy()
                inputs = self.hubert_processor(
                    wav_np, sampling_rate=16000, return_tensors="pt", padding=True,
                )
                with torch.no_grad():
                    out = self.hubert(inputs["input_values"].float())
                feats = out.last_hidden_state  # (1, T, 768)
            else:
                wav_in = wav_16k.float()
                padding_mask = torch.BoolTensor(wav_in.shape).fill_(False)
                with torch.no_grad():
                    logits = self.hubert.extract_features(
                        source=wav_in, padding_mask=padding_mask, output_layer=9,
                    )
                feats = logits[0]  # (1, T, 768)

            feats = torch.repeat_interleave(feats, 2, dim=1)  # (1, 2T, 768)
            print(f"  HuBERT feats: {feats.shape}")
            return feats

        except Exception as e:
            import traceback
            print(f"  [HuBERT extract] {e}")
            traceback.print_exc()
            return None

    def _run_net_g(self, feats, f0, f0_coarse, index_path):
        try:
            if index_path and os.path.isfile(index_path):
                feats = self._apply_index(feats, index_path)

            f0_t  = torch.FloatTensor(f0).unsqueeze(0)
            f0c_t = torch.LongTensor(f0_coarse).unsqueeze(0)

            T_feat = feats.shape[1]
            T_f0   = f0_t.shape[1]
            min_T  = min(T_feat, T_f0)
            feats  = feats[:, :min_T, :]
            f0_t   = f0_t[:, :min_T]
            f0c_t  = f0c_t[:, :min_T]
            feats_len = torch.LongTensor([min_T])

            with torch.no_grad():
                audio_out = self.net_g.infer(
                    feats, feats_len, f0c_t, f0_t,
                    torch.LongTensor([0]),
                )
            out = audio_out[0][0, 0].float().cpu().numpy()
            print(f"  net_g output: {len(out)} samples")
            return out

        except Exception as e:
            import traceback
            print(f"  [net_g] {e}")
            traceback.print_exc()
            return None

    def _apply_index(self, feats: torch.Tensor, index_path: str) -> torch.Tensor:
        try:
            import faiss
            index = faiss.read_index(index_path)
            npy = feats.squeeze(0).numpy().astype(np.float32)  # (T, 768)
            score, ix = index.search(npy, k=8)
            weight = np.square(1 / np.maximum(score, 1e-8))
            weight /= weight.sum(axis=1, keepdims=True)
            vecs = np.array([index.reconstruct(int(i)) for row in ix for i in row])
            vecs = vecs.reshape(len(npy), 8, -1)
            retrieved = np.sum(vecs * weight[:, :, np.newaxis], axis=1)
            blended = 0.75 * retrieved + 0.25 * npy
            feats = torch.FloatTensor(blended).unsqueeze(0)  # (1, T, 768)
            print(f"  Index retrieval applied")
        except Exception as e:
            print(f"  [Index] skipped: {e}")
        return feats

    def _world_fallback(self, wav_1d, sr, f0) -> np.ndarray:
        import pyworld as pw
        wav_f64 = wav_1d.astype(np.float64)
        _f0, t = pw.dio(wav_f64, sr, frame_period=10)
        _f0 = pw.stonemask(wav_f64, _f0, t, sr)
        sp = pw.cheaptrick(wav_f64, _f0, t, sr)
        ap = pw.d4c(wav_f64, _f0, t, sr)
        min_len = min(len(f0), len(_f0))
        f0_mod = np.zeros_like(_f0)
        f0_mod[:min_len] = f0[:min_len]
        out = pw.synthesize(f0_mod, sp, ap, sr, frame_period=10).astype(np.float32)
        peak = np.max(np.abs(out))
        if peak > 1e-8:
            out = out / peak * 0.95
        return out[np.newaxis, :]


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


class RVCVoiceConverter:
    CATEGORY = "Audio/Supertonic"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "convert"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "rvc_model_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Absolute path to RVC .pth model",
                }),
            },
            "optional": {
                "rvc_index_path": ("STRING", {"default": "", "multiline": False}),
                "pitch_shift": ("INT", {"default": 0, "min": -24, "max": 24, "step": 1, "display": "slider"}),
                "f0_method": (["rmvpe", "pm", "harvest"], {"default": "rmvpe"}),
            },
        }

    def convert(
        self,
        audio: dict,
        rvc_model_path: str,
        rvc_index_path: str = "",
        pitch_shift: int = 0,
        f0_method: str = "rmvpe",
    ) -> tuple:
        model_path = strip_path(rvc_model_path)
        raw_index = strip_path(rvc_index_path) if rvc_index_path else ""

        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError(f"RVC model not found at:\n  {model_path}")

        index_path: Optional[str] = raw_index if raw_index and os.path.isfile(raw_index) else None

        wav_np, orig_sr = comfy_audio_to_numpy(audio)
        print(f"\n=== RVCVoiceConverter ===")
        print(f"  Source: {wav_np.shape[-1]} samples @ {orig_sr} Hz")
        print(f"  Model:  {model_path}")
        print(f"  Index:  {os.path.basename(index_path) if index_path else '(none)'}")
        print(f"  Pitch:  {pitch_shift:+d} st | f0: {f0_method} | Device: {DEVICE}")

        if orig_sr != RVC_TARGET_SR:
            print(f"  Resampling {orig_sr} → {RVC_TARGET_SR} Hz ...")
            wav_t = torch.from_numpy(wav_np).float()
            wav_np = torchaudio.functional.resample(wav_t, orig_freq=orig_sr, new_freq=RVC_TARGET_SR).numpy()

        wav_np = normalize_audio(wav_np, peak_target=0.95)

        engine = RVCEngine(model_path, device=DEVICE)
        converted = engine.infer(
            audio=wav_np,
            pitch_shift=pitch_shift,
            f0_method=f0_method,
            index_path=index_path,
        )

        peak_out = np.max(np.abs(converted))
        if peak_out > 0:
            converted = converted / peak_out * 0.95

        print(f"  Converted: {converted.shape[-1]} samples @ {RVC_TARGET_SR} Hz")
        return (numpy_to_comfy_audio(converted.astype(np.float32), RVC_TARGET_SR),)


NODE_CLASS_MAPPINGS = {
    "SupertonicLoader":  SupertonicLoader,
    "SupertonicTTS":     SupertonicTTS,
    "RVCVoiceConverter": RVCVoiceConverter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SupertonicLoader":  "Supertonic Model Loader 🎤",
    "SupertonicTTS":     "Supertonic Text-to-Speech 🗣️",
    "RVCVoiceConverter": "RVC Voice Converter 🔊",
}