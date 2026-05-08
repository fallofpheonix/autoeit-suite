.PHONY: test test-transcribe test-score

test: test-transcribe test-score

test-transcribe:
	cd packages/autoeit-transcribe && python3 -m pytest tests -q

test-score:
	cd packages/autoeit-score && python3 -m pytest tests -q
