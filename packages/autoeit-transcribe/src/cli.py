"""Compatibility CLI module.

The production entrypoint lives in ``api.cli``.
"""

from api.cli import build_parser, main


if __name__ == "__main__":
    raise SystemExit(main())
