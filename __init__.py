"""
ComfyUI-Supertonic3TTS

A suite of ComfyUI custom nodes integrating Supertonic's lightning-fast,
on-device, multilingual Text-to-Speech system.

Nodes:
  - SupertonicLoader: Load TTS engine with voice style
  - SupertonicTTS:    Synthesize speech from text

Category: Audio/Supertonic
"""

from .supertonic_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print(f"\n{'='*60}")
print(f"  ComfyUI-Supertonic3TTS loaded successfully!")
print(f"  Nodes: {', '.join(NODE_CLASS_MAPPINGS.keys())}")
print(f"  Engine: supertonic SDK (ONNX Runtime — auto CPU/GPU)")
print(f"{'='*60}\n")