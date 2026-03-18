from __future__ import annotations

import gzip
import os
import shutil
import time
import urllib.request


MODEL_DIR = "data/models"
MODEL_PATH = os.path.join(MODEL_DIR, "trifecta120_model.joblib")
MODEL_GZ_PATH = os.path.join(MODEL_DIR, "trifecta120_model.joblib.gz")
LOCK_PATH = os.path.join(MODEL_DIR, ".model_extract.lock")


def ensure_model_ready() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        print(f"[MODEL] already exists: {MODEL_PATH}")
        return

    if os.path.exists(MODEL_GZ_PATH):
        _extract_with_lock(MODEL_GZ_PATH, MODEL_PATH)
        return

    model_url = os.getenv("MODEL_URL", "").strip()
    if not model_url:
        raise FileNotFoundError(
            "Model not found locally and MODEL_URL is empty. "
            "Set MODEL_URL in Render Environment Variables."
        )

    print(f"[MODEL] downloading: {model_url}")
    urllib.request.urlretrieve(model_url, MODEL_GZ_PATH)

    _extract_with_lock(MODEL_GZ_PATH, MODEL_PATH)


def _extract_with_lock(src_gz: str, dst_path: str) -> None:
    # 他workerが作成済みなら即終了
    if os.path.exists(dst_path):
        print(f"[MODEL] already exists after check: {dst_path}")
        return

    # 簡易ロック
    while True:
        try:
            fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if os.path.exists(dst_path):
                print(f"[MODEL] ready by another worker: {dst_path}")
                return
            time.sleep(0.5)

    try:
        if os.path.exists(dst_path):
            print(f"[MODEL] already exists in lock: {dst_path}")
            return

        tmp_path = dst_path + ".tmp"
        print(f"[MODEL] extracting downloaded gzip: {src_gz}")

        with gzip.open(src_gz, "rb") as f_in:
            with open(tmp_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.replace(tmp_path, dst_path)
        print(f"[MODEL] ready: {dst_path}")

    finally:
        if os.path.exists(LOCK_PATH):
            os.remove(LOCK_PATH)
