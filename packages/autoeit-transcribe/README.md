# AutoEIT Task-I Transcription

Transcribes Spanish EIT learner audio and writes sentence-level output back into an Excel workbook.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 -m src.cli \
  --audio-dir data/raw \
  --prompt-xlsx "data/metadata/AutoEIT Sample Audio for Transcribing.xlsx" \
  --output-xlsx output/AutoEIT_Task1_Transcriptions_submission.xlsx \
  --model-size models/faster-whisper-tiny \
  --device cpu \
  --compute-type int8
```

## Tests

```bash
python3 -m pytest tests -q
```

## Project Layout

- `api/`: entrypoints (`api.cli`) and command wiring.
- `services/`: orchestration logic for transcription and submission checks.
- `core/`: domain models and shared errors.
- `config/`: runtime request assembly from CLI args.
- `utils/`: lightweight reusable helpers.
- `src/`: existing ASR/audio/postprocess modules kept for backwards compatibility.
- `tests/`: critical-path tests.

## Key Decisions

- Kept the ASR pipeline implementation in `src/` to avoid destabilizing tested behavior; the new layers call into it.
- `src.cli` remains as a compatibility shim so old invocation commands still work.
- Submission validation is intentionally minimal and checks only must-have artifacts.
