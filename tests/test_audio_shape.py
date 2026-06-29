"""Tests for supertonic_nodes audio shape contract.

ComfyUI's AUDIO spec is dict with key "waveform" (Tensor[B, C, samples]) and
"sample_rate" (int). Both numpy_to_comfy_audio and SupertonicTTS._silence must
produce that shape — these tests lock it in so a future refactor can't break it.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from supertonic_utils import numpy_to_comfy_audio  # noqa: E402
from supertonic_nodes import SupertonicTTS  # noqa: E402

SR = 22050


class NumpyToComfyAudioTest(unittest.TestCase):
    def test_2d_mono_input_produces_3d_tensor(self):
        wav = np.zeros((1, SR), dtype=np.float32)
        out = numpy_to_comfy_audio(wav, SR)
        self.assertEqual(out["sample_rate"], SR)
        self.assertIsInstance(out["waveform"], torch.Tensor)
        self.assertEqual(out["waveform"].shape, (1, 1, SR))
        self.assertEqual(out["waveform"].dtype, torch.float32)

    def test_1d_input_raises(self):
        wav = np.zeros(SR, dtype=np.float32)
        with self.assertRaises(AssertionError):
            numpy_to_comfy_audio(wav, SR)

    def test_3d_input_raises(self):
        wav = np.zeros((1, 1, SR), dtype=np.float32)
        with self.assertRaises(AssertionError):
            numpy_to_comfy_audio(wav, SR)


class SilenceTest(unittest.TestCase):
    def test_silence_shape(self):
        out = SupertonicTTS._silence(SR, dur=1.0)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 1)
        payload = out[0]
        self.assertEqual(payload["sample_rate"], SR)
        self.assertEqual(payload["waveform"].shape, (1, 1, SR))
        self.assertEqual(payload["waveform"].dtype, torch.float32)
        # all zeros (silence)
        self.assertEqual(payload["waveform"].abs().sum().item(), 0.0)

    def test_silence_duration(self):
        out = SupertonicTTS._silence(SR, dur=0.5)
        self.assertEqual(out[0]["waveform"].shape[-1], SR // 2)


if __name__ == "__main__":
    unittest.main()