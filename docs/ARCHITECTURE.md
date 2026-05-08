# Architecture

## Components

```text
autoeit-transcribe
  api/cli.py
  config/runtime.py
  services/transcription_service.py
  src/asr/
  src/audio/
  src/align/
  src/io/

autoeit-score
  autoeit/api/cli.py
  autoeit/core/{text,features,rubric}.py
  autoeit/services/{scoring,workbook}.py
```

## Data Flow

```text
Audio directory
  -> audio validation/segmentation
  -> ASR model
  -> sentence alignment
  -> transcription workbook
  -> rubric feature extraction
  -> deterministic scoring
  -> annotated workbook
  -> CSV scores
  -> downgrade/audit log
```

## Invariants

- Scoring is deterministic for identical workbook input.
- Transcription artifacts are generated outputs and must not be committed.
- Model binaries are external artifacts.
- Workbook output schemas must be versioned before downstream automation depends on them.
