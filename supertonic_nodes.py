"""
Supertonic TTS — ComfyUI Custom Nodes
Supertonic's lightning-fast, on-device, multilingual Text-to-Speech system.

Expression Tags (type directly into text):
  <laugh> <breath> <surprise> <sigh> <scream>
  <throatclear> <sad> <angry> <cough> <yawn>
"""

import os
import logging
import shutil
import sys
import threading
import traceback
from pathlib import Path
from typing import Optional
import numpy as np
import torch

try:
    from supertonic import TTS
    from supertonic import AVAILABLE_LANGUAGES as _SDK_LANGS
except ImportError:
    TTS = None
    _SDK_LANGS = []

_NODE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = _NODE_DIR / "models"
_MODEL_DIR = _MODELS_DIR / "supertonic-3"  # SDK writes model here
_TEMP_DIR = _MODELS_DIR / ".supertonic-3.tmp"  # SDK's atomic download staging
_BRAILLE_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_BAR_WIDTH = 28


def ensure_models_dir():
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _spin(stop: threading.Event, label: str) -> None:
    """Background spinner: writes a Unicode Braille frame + label to stdout, overwriting."""
    i = 0
    while not stop.is_set():
        frame = _BRAILLE_FRAMES[i % len(_BRAILLE_FRAMES)]
        sys.stdout.write(f"\r  {frame} {label}")
        sys.stdout.flush()
        i += 1
        stop.wait(0.1)


# ponytail: contextmanager-ish stage spinner — checkmark on exit
class _Stage:
    def __init__(self, label: str):
        self.label = label
        self.stop = threading.Event()
        self.t = None

    def __enter__(self):
        self.t = threading.Thread(target=_spin, args=(self.stop, self.label), daemon=True)
        self.t.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.t.join()
        sys.stdout.write(f"\r  ✓ {self.label}\033[K\n")
        sys.stdout.flush()
        return False


def _dir_bytes(path: Path) -> int:
    """Sum byte size of all files under path. Fast on a few-hundred-MB tree."""
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _fetch_repo_total_bytes(repo_id: str, revision: str) -> int:
    """Sum Content-Length of every file in the HF repo. One HEAD per file."""
    from huggingface_hub import HfApi
    api = HfApi()
    files = api.list_repo_files(repo_id, revision=revision)
    total = 0
    for filename in files:
        try:
            meta = api.get_paths_info(repo_id, paths=[filename], revision=revision)
            if meta and meta[0].size is not None:
                total += meta[0].size
        except Exception:
            pass  # skip files we can't size; underestimate but don't block
    return total


def _render_download_slider(stop: threading.Event, total: int, label: str, sink) -> None:
    """Slider 0-100% driven by polling _TEMP_DIR size. Writes to `sink` (real stdout)
    so the bar is visible even while sys.stdout is swapped to /dev/null.
    """
    bar = "░" * _BAR_WIDTH
    sink.write(f"\r  {label} [{bar}]   0%   0.0/{total/1024/1024:6.1f} MB\033[K")
    sink.flush()
    while not stop.is_set():
        done = _dir_bytes(_TEMP_DIR)
        pct = min(100, int(done * 100 / total)) if total > 0 else 0
        filled = int(_BAR_WIDTH * pct / 100)
        bar = "█" * filled + "░" * (_BAR_WIDTH - filled)
        line = f"\r  {label} [{bar}] {pct:3d}%  {done/1024/1024:6.1f}/{total/1024/1024:6.1f} MB"
        sink.write(line + "\033[K")
        sink.flush()
        if pct >= 100:
            break
        stop.wait(0.15)
    bar = "█" * _BAR_WIDTH
    sink.write(f"\r  {label} [{bar}] 100%  done\033[K\n")
    sink.flush()


def _silence_noisy_loggers() -> dict:
    """Mute huggingface_hub / httpx / httpcore / supertonic to WARNING.
    Returns a dict of {logger_name: previous_level} so caller can restore.
    """
    names = ("huggingface_hub", "httpx", "httpcore", "http11", "filelock",
             "supertonic", "urllib3", "asyncio")
    saved = {}
    for n in names:
        lg = logging.getLogger(n)
        saved[n] = lg.level
        lg.setLevel(logging.WARNING)
    return saved


def _restore_loggers(saved: dict) -> None:
    for n, level in saved.items():
        logging.getLogger(n).setLevel(level)


def _silence_stdio():
    """Swap stdout/stderr to /dev/null. Returns handles to restore later.
    Caller MUST pass the real stdout to anything that needs to print (e.g. slider).
    """
    devnull_out = open(os.devnull, "w")
    devnull_err = open(os.devnull, "w")
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout = devnull_out
    sys.stderr = devnull_err
    return real_out, real_err, devnull_out, devnull_err


def _restore_stdio(real_out, real_err, d_out, d_err) -> None:
    sys.stdout = real_out
    sys.stderr = real_err
    d_out.close()
    d_err.close()


# ponytail: cache env vars set lazily inside get_tts() to keep imports side-effect free
# SUPERTONIC_CACHE_DIR points the SDK at our local models dir instead of ~/.cache/supertonic3
# HF_HOME keeps HuggingFace Hub lock/metadata files out of ~/.cache/huggingface
def _setup_cache() -> None:
    ensure_models_dir()
    os.environ["SUPERTONIC_CACHE_DIR"] = str(_MODEL_DIR)
    os.environ["HF_HOME"] = str(_MODELS_DIR / "hub")
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
    """Convert mono waveform to ComfyUI AUDIO spec.

    Contract:
      wav   : np.ndarray, shape [1, samples], float32, C-contiguous
      returns: {"waveform": torch.Tensor[1, 1, samples], "sample_rate": int}

    Note: real implementation lives in supertononic_utils — this wrapper keeps
    back-compat for any tests/imports that reach into supertonic_nodes directly.
    """
    # ponytail: works both as package member and standalone (tests use absolute import)
    try:
        from .supertonic_utils import numpy_to_comfy_audio as _impl
    except ImportError:
        from supertonic_utils import numpy_to_comfy_audio as _impl  # fallback for sys.path tests
    return _impl(wav, sr)


def apply_postprocessing(  # back-compat re-export
    wav_np: np.ndarray,
    sr: int,
    trim_silence: bool,
    normalize_vol: bool,
    clarity_boost: bool,
    pitch_semitones: float,
    time_stretch: float,
    chorus_effect: bool,
    verbose: bool = False,
) -> np.ndarray:
    try:
        from .supertonic_utils import apply_postprocessing as _impl
    except ImportError:
        from supertonic_utils import apply_postprocessing as _impl
    return _impl(
        wav_np, sr,
        trim_silence=trim_silence, normalize_vol=normalize_vol,
        clarity_boost=clarity_boost, pitch_semitones=pitch_semitones,
        time_stretch=time_stretch, chorus_effect=chorus_effect, verbose=verbose,
    )


_TTS_INSTANCE: Optional[TTS] = None


def get_tts(auto_download: bool = True) -> TTS:
    global _TTS_INSTANCE
    if _TTS_INSTANCE is None:
        if TTS is None:
            raise ImportError("supertonic package not found. Install: pip install supertonic")
        _setup_cache()
        print(f"[Supertonic] Loading from {_MODEL_DIR}")
        from supertonic.loader import has_all_onnx_modules
        needs_download = auto_download and not has_all_onnx_modules(_MODEL_DIR)
        if needs_download:
            _do_animated_download()
        with _Stage("Loading model ..."):
            _TTS_INSTANCE = TTS(model_dir=_MODEL_DIR, auto_download=auto_download)
        print(f"[Supertonic] Ready — {_TTS_INSTANCE.sample_rate} Hz")
    return _TTS_INSTANCE


def _do_animated_download() -> None:
    """Download model with slider, suppressing all HF Hub / httpx / supertonic log output."""
    from supertonic.loader import DEFAULT_MODEL, get_model_repo, get_model_revision
    repo_id = get_model_repo(DEFAULT_MODEL)
    revision = get_model_revision(DEFAULT_MODEL)
    if _TEMP_DIR.exists():
        shutil.rmtree(_TEMP_DIR)

    # Step 1: silence EVERYTHING (loggers + stdio) for the entire pre-fetch + download
    saved_levels = _silence_noisy_loggers()
    real_out, real_err, d_out, d_err = _silence_stdio()
    total = 0
    try:
        # Pre-fetch expected size (silenced — no HTTP logs leak)
        total = _fetch_repo_total_bytes(repo_id, revision)
        # Tell user where the model is going (write to REAL stdout before silencing further)
    finally:
        pass

    # Step 2: announce expected size (visible) and start slider
    msg = f"  Downloading model — {total/1024/1024:.1f} MB → {_MODEL_DIR}\n"
    real_out.write(msg)
    real_out.flush()

    stop = threading.Event()
    slider = threading.Thread(
        target=_render_download_slider,
        args=(stop, total, "Downloading model", real_out),  # slider writes to REAL stdout
        daemon=True,
    )
    slider.start()
    try:
        from supertonic.loader import download_model
        download_model(model_dir=_MODEL_DIR, model_name=DEFAULT_MODEL)
    finally:
        stop.set()
        slider.join()
        _restore_stdio(real_out, real_err, d_out, d_err)
        _restore_loggers(saved_levels)
    real_out.write("[Supertonic] Model ready\n")
    real_out.flush()


class SupertonicLoader:
    CATEGORY = "audio/Supertonic"
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
    CATEGORY = "audio/Supertonic"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "synthesize"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SUPERTONIC_MODEL",),
                "text": ("STRING", {
                    "default": "Hello! <laugh> This is Supertonic-3 speaking.",
                    "multiline": True,
                    "placeholder": "Enter text with optional expression tags...\nTags: <laugh> <breath> <surprise> <sigh> <scream> <throatclear> <sad> <angry> <cough> <yawn>\nExample: Hello! <laugh> That was funny!",
                }),
                "language": (SUPPORTED_LANGUAGES, {"default": "en"}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05, "display": "slider"}),
                "steps": ("INT", {
                    "default": 8, "min": 5, "max": 12, "step": 1, "display": "slider",
                    "tooltip": "Diffusion steps. Higher = smoother, slower. 8 is a good default; 5 is fast, 12 is max quality.",
                }),
                "voice_style": (BUILTIN_VOICES, {"default": "M1"}),
            },
            "optional": {
                "custom_style_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Optional. Absolute path to a Supertonic voice style .json (e.g. from Voice Builder). Leave empty to use the preset voice.",
                    "tooltip": "Absolute path to a custom voice style .json. If set, overrides the preset voice_style dropdown.",
                }),
                "verbose": ("BOOLEAN", {"default": False}),
            },
        }

    def synthesize(
        self,
        model: dict,
        text: str,
        language: str = "en",
        speed: float = 1.0,
        steps: int = 8,
        voice_style: str = "M1",
        custom_style_path: str = "",
        verbose: bool = False,
    ) -> tuple:
        if not text or not text.strip():
            if verbose:
                print("[SupertonicTTS] Empty text — returning silence")
            return self._silence(model["sample_rate"])

        tts_obj = model["tts"]
        sr = tts_obj.sample_rate

        found_tags = [tag for tag in EXPRESSION_TAGS if tag in text]

        style_path = Path(custom_style_path) if custom_style_path else None
        if style_path and style_path.is_file():
            style = tts_obj.get_voice_style_from_path(str(style_path))
        elif custom_style_path:
            print(f"  [Warn] custom_style_path not found: {custom_style_path} — falling back to preset '{voice_style}'")
            style = tts_obj.get_voice_style(voice_style)
        else:
            style = tts_obj.get_voice_style(voice_style)

        if verbose:
            print(f"\n=== SupertonicTTS ===")
            print(f"  Language: {language} | Speed: {speed} | Steps: {steps} | Voice: {voice_style}")
            if found_tags:
                print(f"  Expression tags detected: {', '.join(found_tags)}")
            print(f"  Text: {text[:120]}")

        try:
            wav_np, duration = tts_obj.synthesize(
                text=text, voice_style=style, lang=language,
                speed=speed, total_steps=steps, verbose=verbose,
            )
            if verbose:
                wav_repr = getattr(wav_np, "shape", type(wav_np).__name__)
                print(f"  SDK returned wav shape={wav_repr}, duration type={type(duration).__name__}")
            try:
                dur_sec = float(duration[0]) if hasattr(duration, "__len__") else float(duration)
            except (TypeError, ValueError, IndexError):
                dur_sec = float("nan")
            if verbose:
                samples = wav_np.shape[-1] if hasattr(wav_np, "shape") else "?"
                print(f"  Generated {dur_sec:.2f}s ({samples} samples @ {sr} Hz)")

            if isinstance(wav_np, list):
                wav_np = np.array(wav_np, dtype=np.float32)
            if wav_np.ndim > 1:
                wav_np = wav_np.flatten()
            if wav_np.ndim == 1:
                wav_np = wav_np[np.newaxis, :]

            return (numpy_to_comfy_audio(wav_np.astype(np.float32), sr),)

        except Exception as e:
            # ponytail: traceback always to stderr; verbose only gates friendly banner
            print(f"[SupertonicTTS] ERROR — {e}", file=sys.stderr)
            traceback.print_exc()
            return self._silence(sr)

    @staticmethod
    def _silence(sr: int, dur: float = 1.0) -> tuple:
        wav = np.zeros((1, int(sr * dur)), dtype=np.float32)
        return (numpy_to_comfy_audio(wav, sr),)


NODE_CLASS_MAPPINGS = {
    "SupertonicLoader": SupertonicLoader,
    "SupertonicTTS":    SupertonicTTS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SupertonicLoader": "Supertonic Model Loader 🎤",
    "SupertonicTTS":    "Supertonic Text-to-Speech 🗣️",
}