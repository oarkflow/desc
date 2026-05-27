# Image Feature Test Report

This file is the stable entry point for the real-model image feature matrix.

Run with:

```bash
make -C kyc image-matrix
```

The runner writes the complete per-run Markdown report to `test-results/image-matrix/report.md` and machine-readable results to `test-results/image-matrix/results.json`.

## Last Run Summary

Last attempted run: `2026-05-27T11:48:12+05:45`

Overall status: **fail**

The identity suite produced report artifacts in:

- `test-results/identity-suite/report.md`
- `test-results/identity-suite/results.json`
- `test-results/image-matrix/fixtures`
- `test-results/image-matrix/responses`

Key results:

- Unit tests passed: 53 run, 1 skipped.
- OCR smoke passed on `kyc/testdata/national-id.webp` with 8 extracted values.
- Anti-spoofing passed with the ONNX provider.
- Face check failed because MediaPipe did not return 478-point landmarks for the detected face in `citizenship.jpg`.
- Liveness check failed because no face was detected in the default document fixture.
- The image matrix produced partial artifacts, then was stopped after repeated PaddleOCR/PaddleX text detection exceptions and a CPU-bound stall before final matrix report generation.
