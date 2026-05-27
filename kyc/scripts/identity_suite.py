import argparse
from pathlib import Path
import runpy
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_MODULES = {
    "document": "kyc.scripts.check_document_identity",
    "ocr": "kyc.scripts.check_ocr",
    "describe": "kyc.scripts.check_describe",
    "anti-spoof": "kyc.scripts.check_anti_spoof",
    "liveness": "kyc.scripts.check_liveness",
    "kyc-flow": "kyc.scripts.check_kyc_flow",
    "matrix": "kyc.scripts.run_image_feature_matrix",
    "all-checks": "kyc.scripts.check_all",
}

FACE_COMMANDS = {
    "face-analyze": "analyze",
    "face-detect": "detect",
    "face-detect-all": "detect-all",
    "face-enroll": "enroll",
    "face-recognize": "recognize",
    "face-search": "search",
}


def dispatch_module(module, argv):
    sys.argv = [f"identity_suite.py {module}", *argv]
    runpy.run_module(module, run_name="__main__")


def dispatch_face(command, argv):
    sys.argv = ["identity_suite.py face", command, *argv]
    runpy.run_module("kyc.face.cli", run_name="__main__")


def main():
    parser = argparse.ArgumentParser(
        description="One CLI entry point for individual identity-suite features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/identity_suite.py document --image testdata/national-id.webp --json test-results/document.json
  python scripts/identity_suite.py document --image testdata/national-id.webp --match-face /tmp/selfie.jpg --match-landmarks
  python scripts/identity_suite.py face-detect testdata/citizenship.jpg --json test-results/face.json
  python scripts/identity_suite.py face-search /tmp/selfie.jpg ./people --verbose --json test-results/search.json
  python scripts/identity_suite.py anti-spoof --image /tmp/selfie.jpg
  python scripts/identity_suite.py liveness --image /tmp/selfie.jpg
""",
    )
    commands = sorted([*SCRIPT_MODULES.keys(), *FACE_COMMANDS.keys()])
    parser.add_argument("command", choices=commands)
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the selected command.")
    args = parser.parse_args()

    if args.command in FACE_COMMANDS:
        dispatch_face(FACE_COMMANDS[args.command], args.args)
        return
    dispatch_module(SCRIPT_MODULES[args.command], args.args)


if __name__ == "__main__":
    main()
