# Contributing to AutoEIT

Thank you for your interest in contributing to the AutoEIT transcription pipeline!

## Development Principles
- **Accuracy First:** Focus on preserving learner production exactly as spoken.
- **Modularity:** Keep audio, ASR, and post-processing logic decoupled.
- **Reproducibility:** All pipeline runs should be deterministic and configurable via YAML.

## How to Contribute
1. **Report Bugs:** Open an issue with a detailed description and steps to reproduce.
2. **Feature Requests:** Propose new features by opening an issue first.
3. **Draft Pull Requests:** 
    - Ensure code passes linting and tests.
    - Include documentation for new modules.
    - Keep commits atomic and descriptive.

## Coding Standards
- Use Python 3.10+ features (like type hinting).
- Follow PEP 8 style guidelines.
- Add docstrings to all public functions and classes.

## Testing
Please add tests for any new logic in the `tests/` directory. Run tests using:
```bash
pytest tests/
```
