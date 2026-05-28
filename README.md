# ComfyUI-Supertonic3TTS

A suite of [ComfyUI](https://github.com/comfyanonymous/ComfyUI) custom nodes integrating **Supertone's Supertonic-3** — a lightning-fast, on-device, multilingual Text-to-Speech system running natively via ONNX Runtime.

Additionally includes an **RVC (Retrieval-Based Voice Conversion)** node for voice cloning/changing, though this is currently **non-functional** (see status below).

---

## 📦 Nodes

| Node | Type | Description |
|------|------|-------------|
| **Supertonic Model Loader** 🎤 | Loader | Initialises the Supertonic-3 TTS engine (auto-downloads ~400MB model from Hugging Face on first run) |
| **Supertonic Text-to-Speech** 🗣️ | Generate | Synthesises speech from text with 31 languages, 10 preset voices, and inline expression tags |
| **RVC Voice Converter** 🔊 | Convert | Attempts voice conversion via RVC .pth checkpoints — **currently non-functional** |

---

## ✨ Features (Working)

### Supertonic TTS

- **31 languages** — `en`, `ko`, `ja`, `id`, `ar`, `de`, `es`, `fr`, `hi`, `vi`, and more
- **10 built-in voices** — M1–M5 (male), F1–F5 (female)
- **Expression tags** — Place inline tags directly in your text:
  - `<laugh>`, `<breath>`, `<sigh>`, `<crying>`, `<whisper>`
  - `<shout>`, `<yawn>`, `<cough>`, `<clearthroat>`, `<mumble>`
- **Voice Builder JSON support** — Load custom voice profiles generated from [Supertone's Voice Builder](https://supertone-inc.github.io/supertonic-py/)
- **Custom style path** — Pass an external `.json` file path for zero-shot custom voices
- **Pitch shift** — Optional post-synthesis pitch shifting via torchaudio phase vocoder
- **Speed control** — Native SDK speed parameter (0.7x – 2.0x)
- **Quality control** — Denoising steps (5 – 12, higher = better quality)
- **CPU-friendly** — Runs entirely on CPU via ONNX Runtime, no GPU required
- **Local model cache** — HF models stored in `models/` inside this node's directory (not `~/.cache/`)

### Pipeline

```
Text → [SupertonicTTS] → AUDIO (44.1kHz float32)
```

---

## ❌ RVC Voice Converter — NOT WORKING

The **RVC Voice Converter** node is currently **non-functional**. It attempts to load an RVC .pth checkpoint and run voice conversion, but the output is pure static/buzzing or silence.

### Root Cause

The RVC pipeline requires:
- A **HuBERT** content encoder (for feature extraction)
- An **NSF-HiFiGAN** vocoder/generator (for waveform synthesis)
- Proper **feature dimension alignment** between HuBERT output and the generator input

While the `SynthesizerTrnMs768NSFsid` generator loads successfully from the checkpoint, the feature extraction pipeline produces incorrect tensor shapes or NaN values, resulting in unusable audio output.

### Known Issues

- `mat1 and mat2 shapes cannot be multiplied (768xT and 768xT)` — feature dimension mismatch between HuBERT embeddings and generator expectations
- Pure static/buzzing output — NaN explosion from FP16→CPU conversion or unnormalised intermediate tensors
- HuBERT model loading is fragile — depends on local files or transformers library, with inconsistent fallback behaviour
- WORLD vocoder fallback produces low-quality whispered output

### Status

| Component | Status |
|-----------|--------|
| RVC checkpoint loading | ✅ Works |
| SynthesizerTrn instantiation | ✅ Works |
| HuBERT model loading | ⚠️ Fragile |
| Feature extraction (HuBERT) | ❌ Shape errors |
| net_g.infer forward pass | ❌ Matrix dimension mismatch |
| Index file (.faiss) retrieval | ❌ Untested |
| Audio output | ❌ Static / buzzing |

### Recommendation

For RVC voice cloning, consider using a dedicated RVC application (e.g., [webui](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)) separately, then load the resulting audio into ComfyUI for further processing.

---

## 🚀 Installation

### Requirements

- ComfyUI (any recent version)
- Python 3.10+

### 1. Clone the repository

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/Anonymzx/ComfyUI-Supertonic3TTS.git
```

### 2. Install Python dependencies

**Activate your ComfyUI virtual environment first**, then:

```bash
# Windows (typical ComfyUI venv)
# cd ComfyUI
# .venv\Scripts\activate

pip install -r ComfyUI-Supertonic3TTS/requirements.txt
```

If you get dependency conflicts with `infer-rvc-python`, install it with `--no-deps`:

```bash
pip install infer-rvc-python soundfile ffmpeg-python --no-deps
```

### 3. Restart ComfyUI

The nodes will appear under **`Audio/Supertonic`** in the node menu.

---

## 🎮 Usage

### Basic TTS Pipeline

1. Add **Supertonic Model Loader** (no inputs needed)
2. Add **Supertonic Text-to-Speech**
3. Connect the model from step 1
4. Type your text, select language, voice, speed, and quality
5. Add **Preview Audio** or **Save Audio** to hear the result
6. Run the workflow

### "Golden Combo" Pipeline (RVC is broken)

```
Text → [SupertonicTTS] → AUDIO → [RVC Voice Converter] → ❌ (broken)

Planned:
Text → [SupertonicTTS] → AUDIO → [RVC Voice Converter] → Final AUDIO
```

---

## 🌍 Supported Languages

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| `en` | English | `ko` | Korean | `ja` | Japanese |
| `ar` | Arabic | `bg` | Bulgarian | `cs` | Czech |
| `da` | Danish | `de` | German | `el` | Greek |
| `es` | Spanish | `et` | Estonian | `fi` | Finnish |
| `fr` | French | `hi` | Hindi | `hr` | Croatian |
| `hu` | Hungarian | `id` | Indonesian | `it` | Italian |
| `lt` | Lithuanian | `lv` | Latvian | `nl` | Dutch |
| `pl` | Polish | `pt` | Portuguese | `ro` | Romanian |
| `ru` | Russian | `sk` | Slovak | `sl` | Slovenian |
| `sv` | Swedish | `tr` | Turkish | `uk` | Ukrainian |
| `vi` | Vietnamese | `na` | Unknown / fallback | |

---

## 🎭 Expression Tags

Place these directly in your text:

| Tag | Effect |
|-----|--------|
| `<laugh>` | Laughter effect |
| `<breath>` | Breath intake |
| `<sigh>` | Sigh |
| `<crying>` | Crying voice |
| `<whisper>` | Whispered speech |
| `<shout>` | Raised voice |
| `<yawn>` | Yawning |
| `<cough>` | Cough |
| `<clearthroat>` | Clearing throat |
| `<mumble>` | Mumbling |

Example:
```
<laugh> That's hilarious! <sigh> But I'm exhausted.
```

---

## 📁 File Structure

```
ComfyUI-Supertonic3TTS/
├── __init__.py              # Node registration
├── supertonic_nodes.py      # All node logic + RVC engine
├── rvc_model.py             # RVC model loader (stub)
├── hubert_models.py         # HuBERT model loader
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── models/                  # HF model cache (auto-created)
│   ├── hub/                 # Hugging Face Hub cache
│   └── rvc/                 # RVC model directory
├── styles/                  # Voice Builder JSON directory
└── lib/                     # RVC inference library
    ├── infer_pack/          # NSF-HiFiGAN SynthesizerTrn
    ├── rmvpe.py             # RMVPE pitch extractor
    └── ...
```

---

## 🤝 Contributing

Contributions are welcome, especially for fixing the RVC pipeline! Areas that need work:

- HuBERT feature extraction → generator shape alignment
- FP16/CPU NaN handling in intermediate tensors
- Proper index (.faiss) retrieval integration
- RMVPE pitch extraction integration

---

## 📄 License

Code: MIT License
Model: OpenRAIL-M License (Supertone)

Supertonic: Copyright (c) 2026 Supertone Inc.