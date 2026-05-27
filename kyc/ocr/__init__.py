try:
    from kyc.ocr.service import app, run_cli
except ModuleNotFoundError:
    from ocr.service import app, run_cli

__all__ = ["app", "run_cli"]
