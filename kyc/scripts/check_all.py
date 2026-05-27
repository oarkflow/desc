import argparse
import subprocess
import sys
from pathlib import Path

from check_common import KYC_ROOT, print_json


SCRIPT_DIR = Path(__file__).resolve().parent


def run_check(name, args):
    command = [sys.executable, str(SCRIPT_DIR / f"check_{name}.py"), *args]
    completed = subprocess.run(command, cwd=str(KYC_ROOT.parent), text=True, capture_output=True)
    return {
        "name": name,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def main():
    parser = argparse.ArgumentParser(description="Run all local KYC/OCR capability checks.")
    parser.add_argument("--allow-missing-models", action="store_true", help="Do not fail the anti-spoofing check when its ONNX artifact is absent.")
    parser.add_argument("--skip-ocr", action="store_true")
    parser.add_argument("--skip-describe", action="store_true")
    parser.add_argument("--skip-face", action="store_true")
    parser.add_argument("--skip-liveness", action="store_true")
    parser.add_argument("--skip-anti-spoof", action="store_true")
    parser.add_argument("--skip-kyc-flow", action="store_true")
    args = parser.parse_args()

    checks = []
    if not args.skip_ocr:
        ocr_args = ["--allow-missing-model"] if args.allow_missing_models else []
        checks.append(("ocr", ocr_args))
    if not args.skip_describe:
        checks.append(("describe", ["--allow-empty-text"]))
    if not args.skip_face:
        checks.append(("face", []))
    if not args.skip_liveness:
        checks.append(("liveness", []))
    if not args.skip_anti_spoof:
        anti_spoof_args = ["--allow-missing-model"] if args.allow_missing_models else []
        checks.append(("anti_spoof", anti_spoof_args))
    if not args.skip_kyc_flow:
        checks.append(("kyc_flow", []))

    results = [run_check(name, check_args) for name, check_args in checks]
    failed = [result for result in results if result["returncode"] != 0]
    print_json(
        {
            "check": "all",
            "status": "fail" if failed else "pass",
            "results": results,
        }
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
