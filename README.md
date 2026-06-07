# ComfyUI-Supertonic3TTS

A suite of [ComfyUI](https://github.com/comfyanonymous/ComfyUI) custom nodes integrating **Supertone's Supertonic-3** — a lightning-fast, on-device, multilingual Text-to-Speech system running natively via ONNX Runtime.

---

## 📦 Nodes

| Node | Type | Description |
|------|------|-------------|
| **Supertonic Model Loader** 🎤 | Loader | Initialises the Supertonic-3 TTS engine (auto-downloads ~400MB model from Hugging Face on first run) |
| **Supertonic Text-to-Speech** 🗣️ | Generate | Synthesises speech from text with 31 languages and 10 preset voices |

---

## ✨ Features

### Supertonic TTS

- **31 languages** — `en`, `ko`, `ja`, `id`, `ar`, `de`, `es`, `fr`, `hi`, `vi`, and more
- **10 built-in voices** — M1–M5 (male), F1–F5 (female)
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

## 🎭 Expression Tags — NOT SUPPORTED in Local SDK

The Supertonic-3 **documentation** advertises 10 expression tags (`<laugh>`, `<breath>`, `<sigh>`, etc.), but these tags **only work on the Supertone Cloud API / HTTP server endpoint** — **not** in the open-weight Python SDK running locally.

### Why they don't work

The `supertonic` Python package's text preprocessor (`core.py` → `_preprocess_text`) performs text normalization that strips or ignores angle-bracket tags. The tags are passed through as literal text (e.g., the word `<laugh>` is spoken as-is), not interpreted as vocal expressions.

| Tag | Expected | Actual (local SDK) |
|-----|----------|-------------------|
| `<laugh>` | Laughter effect | Spoken as "laugh" |
| `<whisper>` | Whispered speech | Spoken as "whisper" |
| `<sigh>` | Sigh effect | Spoken as "sigh" |

### Where they DO work

- ✅ **Supertone Cloud API** — managed endpoint at [api.supertone.ai](https://supertone-inc.github.io/supertonic-py/)
- ✅ **Supertone HTTP server** — `supertonic serve` mode (see [docs](https://supertone-inc.github.io/supertonic-py/))
- ❌ **Local Python SDK (TTS class)** — text preprocessor strips the tags

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

> **Note:** `torch` and `torchaudio` are **not** listed in requirements.txt — they are inherited from ComfyUI itself. Only `supertonic`, `numpy`, and `soundfile` are required on top of ComfyUI's base dependencies.

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

## 📁 File Structure

```
ComfyUI-Supertonic3TTS/
├── __init__.py              # Node registration
├── supertonic_nodes.py      # All node logic
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── LICENSE                  # MIT License
├── .gitignore
└── models/                  # HF model cache (auto-created)
    └── hub/                 # Hugging Face Hub cache
```

---

## 📄 License

Code: MIT License
Model: OpenRAIL-M License (Supertone)

Supertonic: Copyright (c) 2026 Supertone Inc.