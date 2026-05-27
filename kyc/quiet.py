from __future__ import annotations

import contextlib
import os
import threading
import warnings


_LOCK = threading.RLock()


def configure_quiet_ml_runtime() -> None:
    """Reduce native ML framework startup chatter before models are imported."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("GLOG_logtostderr", "0")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("ABSL_LOGGING_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")
    os.environ.setdefault("FLAGS_minloglevel", "2")
    os.environ.setdefault("PADDLE_CPP_LOG_LEVEL", "ERROR")
    os.environ.setdefault("PADDLE_LOG_LEVEL", "ERROR")
    os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "kyc-matplotlib"))
    warnings.filterwarnings("ignore", message="No ccache found.*")
    warnings.filterwarnings("ignore", message="Error fetching version info.*")


def quiet_models_enabled() -> bool:
    return os.getenv("KYC_MODEL_VERBOSE", "false").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


@contextlib.contextmanager
def suppress_native_output():
    """Suppress stdout/stderr writes from native model libraries."""
    if not quiet_models_enabled():
        yield
        return

    with _LOCK:
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
        try:
            with open(os.devnull, "w") as devnull:
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    yield
        finally:
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)
