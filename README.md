# autoeit-suite

Integrated tooling for Spanish Elicited Imitation Task workflows.

## Scope

This repository combines two previously separate projects:

- `packages/autoeit-transcribe`: audio-to-workbook transcription pipeline.
- `packages/autoeit-score`: deterministic workbook scoring engine.

The intended pipeline is:

```text
audio files + prompt workbook
  -> transcription workbook
  -> deterministic STS scoring
  -> annotated workbook + CSV + audit log
```

## Repository Layout

```text
packages/
  autoeit-transcribe/   ASR, alignment, workbook transcription, submission audit
  autoeit-score/        Rule-based Spanish EIT scoring, workbook annotation
apps/                   Future review/UI surfaces
docs/                   Suite-level architecture and validation docs
```

## Run

Transcription:

```bash
cd packages/autoeit-transcribe
python3 -m pip install -r requirements.txt
python3 -m src.cli --help
python3 -m pytest tests -q
```

Scoring:

```bash
cd packages/autoeit-score
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
autoeit --help
pytest
```

## Artifact Policy

Large or generated artifacts are not tracked:

- raw audio
- ASR model binaries
- generated workbooks
- generated reports
- local databases
- Python caches

Use release assets, external object storage, or a dataset registry for reproducible data/model artifacts.
