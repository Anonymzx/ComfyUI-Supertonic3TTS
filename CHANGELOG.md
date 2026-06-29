# Changelog

All notable changes to ComfyUI-Supertonic3TTS.

## [Unreleased] — 2026-06-30

### Added
- **`SupertonicEffects`** node — chainable post-processing (trim, normalize, clarity, pitch, time-stretch, chorus) as a standalone `AUDIO → AUDIO` node. Reusable on any audio source.
- **`custom_style_path` field** on `SupertonicTTS` — point at any Supertonic voice style `.json` (e.g. from Voice Builder) without needing a separate loader node.
- **First-run download slider** — animated 0–100% progress bar with byte-accurate counting; HF Hub HTTP logs and tqdm noise are silenced during the download.
- **Local model cache** — model files now download to `<node>/models/supertonic-3/` instead of `~/.cache/supertonic3/`.
- **CHANGELOG.md** — release history.
- **Unit tests** — `tests/test_audio_shape.py` (shape contract) + `tests/test_postprocessing.py` (effects behavior).

### Changed
- **`SupertonicTTS` simplified** — removed embedded post-processing params (`trim_silence`, `normalize_volume`, `clarity_boost`, `pitch_semitones`, `time_stretch`, `chorus_effect`). Use the new `SupertonicEffects` node instead.
- **`quality` → `steps`** — the diffusion-step parameter is now named `steps` for clarity.
- **Default text** is now a usable example (`Hello! <laugh> This is Supertonic-3 speaking.`) instead of empty.
- **CATEGORY** unified to `audio/Supertonic` across all nodes (was `Supertonic 3`).
- **Audio utilities** extracted into `supertonic_utils.py` and shared across nodes.

### Removed
- **`SupertonicStyleLoader` node** — custom voice styles are now loaded inline via the `custom_style_path` field on `SupertonicTTS`. One fewer node to wire.

## [1.0.0] — 2026-06-29

### Added
- Initial release.
- `SupertonicLoader` — initialises the TTS engine, auto-downloads the Supertonic-3 model.
- `SupertonicTTS` — synthesises speech with embedded post-processing effects.
- 31 languages, 10 preset voices, expression tags.