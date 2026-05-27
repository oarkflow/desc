import base64
import hashlib
import time

import numpy as np

def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def row_to_dict(row):
    return dict(row) if row else None


def hash_secret(secret):
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def normalize_vector(vector):
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm == 0:
        return array.tolist()
    return (array / norm).astype(float).tolist()


def cosine_similarity(vector_a, vector_b):
    a = np.asarray(vector_a, dtype=np.float32)
    b = np.asarray(vector_b, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def decode_data_url(data_url):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return base64.b64decode(data_url)


