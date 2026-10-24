"""
Publishing the finished model to the lecturer's Hugging Face account.

The lecturer's token is checked when the job is submitted (`check_token`), long before the
upload: a read-only or mistyped token should fail in seconds, not after hours of training.
That check also tells us the account name, so the repository id (`<user>/<slug>-tutor-gguf`)
is fixed up front and baked into the bundle (model card, DEPLOY.md, Space template). The
publish stage (stages/publish.py) then creates the repository if needed and pushes the GGUF,
its `.corecs.json` profile and the model card in one commit.

The repository is public: students' desktop apps download the model without a token.
"""

import re

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
TOKEN_HELP = ("Create one at https://huggingface.co/settings/tokens (Create new token, "
              "type \"Write\").")


class PublishError(RuntimeError):
    """A message meant for the lecturer."""


def repo_for(user: str, slug: str) -> str:
    return f"{user}/{slug}-tutor-gguf"


def _can_write(auth: dict, user: str) -> bool:
    token = auth.get("accessToken") or {}
    role = token.get("role")
    if role == "write":
        return True
    if role != "fineGrained":
        return False
    fine = token.get("fineGrained") or {}
    if "repo.write" in (fine.get("global") or []):
        return True
    return any((s.get("entity") or {}).get("type") == "user"
               and (s.get("entity") or {}).get("name") == user
               and "repo.write" in (s.get("permissions") or [])
               for s in fine.get("scoped") or [])


def check_token(token: str) -> str:
    """The token's account name, if the token may create and write model repositories."""
    try:
        info = HfApi(token=token).whoami()
    except HfHubHTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status == 401:
            raise PublishError(f"Hugging Face rejected this token. {TOKEN_HELP}") from e
        raise PublishError(f"Could not check the Hugging Face token (HTTP {status}). Try again.") from e
    except Exception as e:  # noqa: BLE001 - network trouble etc.
        raise PublishError(f"Could not reach Hugging Face to check the token: {type(e).__name__}.") from e
    user = info.get("name") or ""
    if info.get("type") != "user" or not user:
        raise PublishError(f"Use a personal Hugging Face access token. {TOKEN_HELP}")
    if not _can_write(info.get("auth") or {}, user):
        raise PublishError("This Hugging Face token is read-only, but the server needs to publish your "
                           f"model to your account. {TOKEN_HELP}")
    return user
