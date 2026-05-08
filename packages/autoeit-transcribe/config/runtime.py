from __future__ import annotations

from argparse import Namespace

from core.entities import BatchTranscriptionRequest


def request_from_cli_args(args: Namespace) -> BatchTranscriptionRequest:
    return BatchTranscriptionRequest(
        audio_dir=args.audio_dir,
        prompt_xlsx=args.prompt_xlsx,
        output_xlsx=args.output_xlsx,
        expected_count=args.expected_count,
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        only_sheet=args.sheet,
    )
