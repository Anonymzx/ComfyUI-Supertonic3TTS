"""
ComfyUI-Supertonic3TTS

A suite of ComfyUI custom nodes integrating Supertonic's lightning-fast,
on-device, multilingual Text-to-Speech system.

Nodes (all under audio/Supertonic):
  - SupertonicLoader:    Load TTS engine → SUPERTONIC_MODEL
  - SupertonicTTS:       Synthesize speech → AUDIO
  - SupertonicEffects:   Post-process any AUDIO (trim/pitch/stretch/chorus) → AUDIO
  - SupertonicStyleLoader: Load custom .json voice style → SUPERSTYLE
"""

from .supertonic_nodes import NODE_CLASS_MAPPINGS as _NODES_BASE, NODE_DISPLAY_NAME_MAPPINGS as _DISPLAY_BASE, TTS
from .supertonic_effects import NODE_CLASS_MAPPINGS as _NODES_EFFECTS, NODE_DISPLAY_NAME_MAPPINGS as _DISPLAY_EFFECTS

NODE_CLASS_MAPPINGS = {**_NODES_BASE, **_NODES_EFFECTS}
NODE_DISPLAY_NAME_MAPPINGS = {**_DISPLAY_BASE, **_DISPLAY_EFFECTS}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# ponytail: banner only when SDK actually loaded — silent failure otherwise
if TTS is not None:
    print(f"\n{'='*60}")
    print(f"  ComfyUI-Supertonic3TTS loaded successfully!")
    print(f"  Nodes: {', '.join(NODE_CLASS_MAPPINGS.keys())}")
    print(f"  Engine: supertonic SDK (ONNX Runtime — auto CPU/GPU)")
    print(f"{'='*60}\n")