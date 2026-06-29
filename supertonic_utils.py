"""Internal audio utilities shared between Supertonic nodes.

Keep this module dependency-light: only numpy + torch (always ComfyUI deps).
Librosa is loaded lazily inside apply_postprocessing so the plugin still
imports cleanly when post-processing is not in use.
"""

import numpy as np
import torch


def numpy_to_comfy_audio(wav: np.ndarray, sr: int) -> dict:
    """Convert mono waveform to ComfyUI AUDIO spec.

    Contract:
      wav   : np.ndarray, shape [1, samples], float32, C-contiguous
      returns: {"waveform": torch.Tensor[1, 1, samples], "sample_rate": int}
    """
    assert wav.ndim == 2, f"expected 2D [1, samples], got shape {wav.shape}"
    wav_c = np.ascontiguousarray(wav, dtype=np.float32)
    tensor = torch.from_numpy(wav_c[None, :, :]).float()  # [1, 1, samples]
    return {"waveform": tensor, "sample_rate": sr}


def comfy_audio_to_numpy(payload: dict) -> tuple:
    """Reverse of numpy_to_comfy_audio. Returns (np.ndarray[1, samples], sample_rate)."""
    t = payload["waveform"]
    sr = int(payload["sample_rate"])
    if isinstance(t, torch.Tensor):
        arr = t.squeeze(0).cpu().numpy()
    else:
        arr = np.asarray(t).squeeze(0)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    return arr.astype(np.float32), sr


def apply_postprocessing(
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
    """Apply librosa post-processing to the audio waveform."""
    effects_requested = (
        trim_silence or clarity_boost or abs(pitch_semitones) >= 0.5
        or abs(time_stretch - 1.0) > 0.01 or chorus_effect or normalize_vol
    )

    try:
        import librosa
    except ImportError:
        if effects_requested:
            raise ImportError(
                "librosa is required for post-processing effects. "
                "Install: pip install librosa"
            ) from None
        return wav_np

    effects_applied = []

    if trim_silence:
        wav_np, _ = librosa.effects.trim(wav_np, top_db=25)
        effects_applied.append("trim")

    if clarity_boost:
        wav_np = librosa.effects.preemphasis(wav_np, coef=0.97)
        effects_applied.append("clarity")

    if abs(pitch_semitones) >= 0.5:
        wav_np = librosa.effects.pitch_shift(y=wav_np, sr=sr, n_steps=float(pitch_semitones))
        effects_applied.append(f"pitch({pitch_semitones:+.1f}st)")

    if abs(time_stretch - 1.0) > 0.01:
        # ponytail: librosa rate>1 = faster (shorter), rate<1 = slower (longer)
        wav_np = librosa.effects.time_stretch(y=wav_np, rate=time_stretch)
        effects_applied.append(f"stretch({time_stretch:.2f}x)")

    # ponytail: chorus delay assumes original sr; safe because this block runs after pitch/stretch
    if chorus_effect:
        wav_ps = librosa.effects.pitch_shift(y=wav_np, sr=sr, n_steps=-2)
        wav_delayed = np.pad(wav_ps, (int(sr * 0.03), 0), mode="constant")
        wav_delayed = wav_delayed[:len(wav_np)]
        wav_np = (wav_np * 0.7) + (wav_delayed * 0.5)
        effects_applied.append("chorus")

    if normalize_vol:
        wav_np = librosa.util.normalize(wav_np)
        effects_applied.append("normalize")

    if effects_applied and verbose:
        print(f"  Post-processing: {', '.join(effects_applied)}")

    return wav_np
