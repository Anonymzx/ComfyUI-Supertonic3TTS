"""Supertonic Audio Effects — ComfyUI node.

Pisahkan post-processing dari TTS node supaya bisa dipakai ulang pada sumber
audio apapun (TTS, music gen, mic recording, dll.) tanpa terkunci ke TTS output.
"""

from .supertonic_utils import (
    apply_postprocessing,
    comfy_audio_to_numpy,
    numpy_to_comfy_audio,
)


class SupertonicEffects:
    """Chain audio post-processing effects. AUDIO → AUDIO."""

    CATEGORY = "audio/Supertonic"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "trim_silence": ("BOOLEAN", {"default": True}),
                "normalize_volume": ("BOOLEAN", {"default": True}),
                "clarity_boost": ("BOOLEAN", {"default": False}),
                "pitch_semitones": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.5, "display": "slider",
                    "tooltip": "Pitch shift in semitones. 0 = no change.",
                }),
                "time_stretch": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05, "display": "slider",
                    "tooltip": "Tempo change. >1 = faster, <1 = slower. Range 0.5–2.0.",
                }),
                "chorus_effect": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Adds a pitch-shifted delayed layer for a richer sound.",
                }),
            },
        }

    def apply(
        self,
        audio: dict,
        trim_silence: bool = True,
        normalize_volume: bool = True,
        clarity_boost: bool = False,
        pitch_semitones: float = 0.0,
        time_stretch: float = 1.0,
        chorus_effect: bool = False,
    ) -> tuple:
        wav_np, sr = comfy_audio_to_numpy(audio)
        wav_np = apply_postprocessing(
            wav_np, sr,
            trim_silence=trim_silence,
            normalize_vol=normalize_volume,
            clarity_boost=clarity_boost,
            pitch_semitones=pitch_semitones,
            time_stretch=time_stretch,
            chorus_effect=chorus_effect,
        )
        return (numpy_to_comfy_audio(wav_np, sr),)


NODE_CLASS_MAPPINGS = {
    "SupertonicEffects": SupertonicEffects,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SupertonicEffects": "Supertonic Effects ✨",
}
