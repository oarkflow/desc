import sys

try:
    from kyc.ocr import service as _service
except ModuleNotFoundError:
    from ocr import service as _service

sys.modules[__name__] = _service

if __name__ == "__main__":
    _service.main()
