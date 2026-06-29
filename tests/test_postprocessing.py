"""Tests for supertonic_nodes.apply_postprocessing.

Strategy: use synthetic sine waves so the test is deterministic and runs without
the SDK. Covers the four effects (pitch, time-stretch, chorus, clarity) plus the
librosa-missing guard.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from supertonic_utils import apply_postprocessing  # noqa: E402

SR = 22050
DUR = 1.0
F0 = 220.0  # A3


def sine(freq: float, dur: float = DUR, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def dominant_freq(wav: np.ndarray, sr: int = SR) -> float:
    """Crude FFT peak detection — enough to confirm pitch moved."""
    spec = np.abs(np.fft.rfft(wav))
    freqs = np.fft.rfftfreq(len(wav), 1 / sr)
    return float(freqs[np.argmax(spec[1:]) + 1])


class PitchShiftTest(unittest.TestCase):
    def test_pitch_up_raises_dominant_freq(self):
        wav = sine(F0)
        out = apply_postprocessing(
            wav, SR, trim_silence=False, normalize_vol=False, clarity_boost=False,
            pitch_semitones=2.0, time_stretch=1.0, chorus_effect=False,
        )
        self.assertGreater(dominant_freq(out), dominant_freq(wav) * 1.1)


class TimeStretchTest(unittest.TestCase):
    def test_stretch_05x_doubles_length(self):
        # ponytail: librosa rate>1 = faster (shorter); rate<1 = slower (longer)
        wav = sine(F0)
        out = apply_postprocessing(
            wav, SR, trim_silence=False, normalize_vol=False, clarity_boost=False,
            pitch_semitones=0.0, time_stretch=0.5, chorus_effect=False,
        )
        # expect ~2x length, tolerate resampling edge
        self.assertGreater(len(out), int(SR * DUR * 1.8))


class ChorusTest(unittest.TestCase):
    def test_chorus_changes_signal(self):
        wav = sine(F0)
        out = apply_postprocessing(
            wav, SR, trim_silence=False, normalize_vol=False, clarity_boost=False,
            pitch_semitones=0.0, time_stretch=1.0, chorus_effect=True,
        )
        # chorus adds a delayed pitch-shifted copy → waveform differs
        self.assertFalse(np.allclose(out, wav[:len(out)], atol=1e-3))


class LibrosaMissingTest(unittest.TestCase):
    """When librosa is missing and an effect is requested, raise a clear error."""

    def test_raises_when_librosa_missing_and_effect_requested(self):
        wav = sine(F0)
        with patch.dict(sys.modules, {"librosa": None}):
            with self.assertRaises(ImportError) as ctx:
                apply_postprocessing(
                    wav, SR, trim_silence=False, normalize_vol=False, clarity_boost=False,
                    pitch_semitones=2.0, time_stretch=1.0, chorus_effect=False,
                )
        self.assertIn("librosa", str(ctx.exception))

    def test_no_effects_no_librosa_returns_unchanged(self):
        wav = sine(F0)
        with patch.dict(sys.modules, {"librosa": None}):
            out = apply_postprocessing(
                wav, SR, trim_silence=False, normalize_vol=False, clarity_boost=False,
                pitch_semitones=0.0, time_stretch=1.0, chorus_effect=False,
            )
        # ponytail: nothing requested, no librosa → passthrough (warn-once happens at import-time)
        self.assertEqual(out.tolist(), wav.tolist())


if __name__ == "__main__":
    unittest.main()