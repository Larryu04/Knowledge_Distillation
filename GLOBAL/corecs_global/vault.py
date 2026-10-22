"""
Encrypted, short-lived storage for lecturer credentials (LLM API key, optional HF token).

Keys are Fernet-encrypted with a per-installation key (data/secret.key, or GLOBAL_SECRET_KEY)
before they touch SQLite, are never returned by the API, and are deleted as soon as the
stages that need them are finished (the teacher key after dataset generation, everything
when the job reaches a terminal state).
"""

import os

from cryptography.fernet import Fernet, InvalidToken

from . import db
from .config import get_settings

TEACHER_KEY = "teacher_api_key"
HF_TOKEN = "hf_token"


def _fernet() -> Fernet:
    env_key = os.environ.get("GLOBAL_SECRET_KEY")
    if env_key:
        return Fernet(env_key.encode())
    path = get_settings().secret_key_path
    if not path.exists():
        key = Fernet.generate_key()
        # O_EXCL: two processes racing on first start must not each write a different key.
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(key)
        except FileExistsError:
            pass
    return Fernet(path.read_bytes().strip())


def store(job_id: str, name: str, value: str) -> None:
    db.put_secret(job_id, name, _fernet().encrypt(value.encode()))


def load(job_id: str, name: str) -> str | None:
    blob = db.get_secret(job_id, name)
    if blob is None:
        return None
    try:
        return _fernet().decrypt(blob).decode()
    except InvalidToken:
        return None


def forget(job_id: str, name: str | None = None) -> None:
    db.delete_secrets(job_id, name)
