import hashlib
import re
import time
import uuid
from pathlib import Path

class LocalEvidenceStorage:
    def __init__(self, root="evidence"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_bytes(self, session_id, category, filename, data):
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename or "upload.bin")
        session_dir = self.root / session_id / category
        session_dir.mkdir(parents=True, exist_ok=True)
        unique_name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{safe_name}"
        path = session_dir / unique_name
        path.write_bytes(data)
        return {
            "path": str(path),
            "filename": safe_name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        }

    def delete_session(self, session_id):
        session_dir = self.root / session_id
        if not session_dir.exists():
            return
        for child in sorted(session_dir.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        session_dir.rmdir()


