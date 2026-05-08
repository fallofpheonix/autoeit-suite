# AutoEIT System Architecture

## Overview
The AutoEIT transcription pipeline is designed as a modular solution for transcribing and evaluating Spanish Elicited Imitation Task (EIT) learner speech.

## System Components

### 1. Data Layer (`data/`)
- **Raw:** Original learner audio recordings and participant workbooks.
- **Processed:** Resampled, normalized, and segmented audio versions.
- **Metadata:** Task-specific data including target prompts and scoring rubrics.

### 2. Audio Processing Module (`src/audio/`)
- **Preprocessing:** Handles sample rate conversion (16kHz), loudness normalization, and denoising.
- **Segmentation:** Uses Voice Activity Detection (VAD) to split continuous recordings into item-level utterances based on expected prompt counts.

### 3. ASR Module (`src/asr/`)
- **Model Wrapper:** Interfaces with `faster-whisper`.
- **Decoding Engine:** Manages beam search settings, language identification, and learner-aware decoding parameters (disabling over-aggressive normalization).

### 4. Post-processing Module (`src/postprocess/`)
- **Transcript Cleanup:** Removes machine-generated artifacts and hallucinations (e.g., "Gracias por ver") while strictly preserving learner speech errors and disfluencies.

### 5. Evaluation Module (`src/eval/`)
- **Metrics:** Calculates Word Error Rate (WER) and Character Error Rate (CER) against human-verified gold standards.
- **Agreement Analysis:** Measures item-level agreement between the ASR output and human transcribers.

### 6. CLI & Configuration
- **CLI (`src/cli.py`):** Unified entry point for running the pipeline as a batch process.
- **Configs (`configs/`):** YAML-based parameter management for reproducible experiments.

## Data Flow
1. **Ingest:** Load participant ID and audio references from metadata.
2. **Preprocess:** Clean and normalize raw audio.
3. **Segment:** Identify the 30 EIT responses in the audio stream.
4. **Transcribe:** Run ASR for each segment.
5. **Clean:** Apply post-processing rules.
6. **Export:** Generate final reports in Excel/CSV/JSON format.
