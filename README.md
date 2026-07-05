# ComfyUI-Supertonic3TTS

A suite of [ComfyUI](https://github.com/comfyanonymous/ComfyUI) custom nodes integrating **Supertone's Supertonic-3** — a lightning-fast, on-device, multilingual Text-to-Speech system running natively via ONNX Runtime.

<div class align="center">
    
<img width="1081" height="607" alt="image" src="https://github.com/user-attachments/assets/9bc840c0-d108-4287-a98b-f54b63acce8c" />

</div>

---

## 📦 Nodes

| Node | Input | Output | Description |
|------|-------|--------|-------------|
| **Supertonic Model Loader** 🎤 | — | `SUPERTONIC_MODEL` | Initialises the TTS engine. Auto-downloads ~400MB model on first run, stored locally in `models/`. |
| **Supertonic Text-to-Speech** 🗣️ | `model`, text, lang, voice, speed, steps | `AUDIO` | Synthesises speech from text. 31 languages, 10 preset voices, expression tags, custom style `.json` via `custom_style_path`. |
| **Supertonic Effects** ✨ | `audio` | `AUDIO` | Optional post-processing (trim, normalize, pitch, stretch, chorus). Apply to any `AUDIO` source. |

---

## ✨ Features

### Supertonic TTS

- **31 languages** — `en`, `ko`, `ja`, `id`, `ar`, `de`, `es`, `fr`, `hi`, `vi`, and more
- **10 built-in voices** — M1–M5 (male), F1–F5 (female)
- **Custom voice styles** — Pass an absolute path to a Supertonic `.json` voice profile in the `custom_style_path` field (e.g. from [Supertone's Voice Builder](https://supertone-inc.github.io/supertonic-py/))
- **Expression tags** — Type tags like `<laugh>` or `<sigh>` directly into text for vocal expressions
- **Speed control** — Native SDK speed parameter (0.5x – 2.0x). For finer post-synthesis tempo tweaks, use the SupertonicEffects `time_stretch` slider.
- **Steps** — Diffusion steps (5–12, default 8). Higher = smoother, slower.
- **CPU-friendly** — Runs entirely on CPU via ONNX Runtime, no GPU required
- **Local model cache** — Model files stored in `models/supertonic-3/` inside this node's directory (not `~/.cache/`)

### Pipeline

```
┌──────────────┐    ┌────────────────────────────┐    ┌──────────────┐
│   text +     │ →  │     SupertonicTTS          │ →  │   AUDIO      │
│  language    │    │ (lang, voice, speed, steps)│    │              │
└──────────────┘    └─────────────┬──────────────┘    └──────┬───────┘
                                  │                            │
                                  │       ┌──────────────┐     │
                                  └─────→ │ SupertonicEffects│ ←─┘
                                          │ (trim/pitch/etc)│
                                          └──────┬────────┘
                                                 ▼
                                          Preview / Save Audio
```

---

## 🎭 Expression Tags

Type any of the following tags directly into your text to add vocal expressions:

| Tag | Effect |
|-----|--------|
| `<laugh>` | Laughter |
| `<breath>` | Breath intake |
| `<surprise>` | Surprise tone |
| `<sigh>` | Sigh |
| `<scream>` | Scream / shout |
| `<throatclear>` | Throat clear |
| `<sad>` | Sad tone |
| `<angry>` | Angry tone |
| `<cough>` | Cough |
| `<yawn>` | Yawn |

**Example:**
```
Halo! <laugh> Senang bertemu denganmu! <sigh> Tapi aku lelah.
```

The tags are passed to the Supertonic SDK which interprets them during synthesis.

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
pip install -r ComfyUI-Supertonic3TTS/requirements.txt
```

> **Note:** `torch` and `torchaudio` are **not** listed in requirements.txt — they are inherited from ComfyUI itself. Only `supertonic`, `numpy`, `soundfile`, and `librosa` are required on top of ComfyUI's base dependencies.

### 3. Restart ComfyUI

The nodes will appear under **`audio/Supertonic`** in the node menu.

> **First run only** — the Loader auto-downloads the ~400MB Supertonic-3 model into `models/supertonic-3/`. A clean 0–100% slider shows progress in the console. Subsequent loads skip the download.

---

## 🎮 Usage

### Basic TTS Pipeline

1. Add **Supertonic Model Loader** (no inputs needed)
2. Add **Supertonic Text-to-Speech**
3. Connect the model from step 1
4. Type your text (include expression tags like `<laugh>` if desired)
5. Configure: language, voice, speed, steps
6. Add **Preview Audio** or **Save Audio** to hear the result
7. Run the workflow

### Add post-processing

Wire the TTS `AUDIO` output into **Supertonic Effects**, then into Preview/Save. Adjust pitch / stretch / chorus as needed.

### Use a custom voice style

Set the optional `custom_style_path` field on the **Supertonic Text-to-Speech** node to an absolute path of a Supertonic voice style `.json` (e.g. generated from [Supertone's Voice Builder](https://supertone-inc.github.io/supertonic-py/)). When set, it overrides the preset `voice_style` dropdown.

### Speed vs Time Stretch

- **SupertonicTTS `speed`** changes tempo *during* synthesis (model-aware, cleanest).
- **SupertonicEffects `time_stretch`** uses phase vocoder after synthesis (any source, slight artifacts at extremes).
- Combine both: SDK `speed` first, then Effects `time_stretch` on the output. Effective tempo ≈ `speed × time_stretch`.

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

## 📄 License

Code: MIT License
Model: OpenRAIL-M License (Supertone)

Supertonic: Copyright (c) 2026 Supertone Inc.
