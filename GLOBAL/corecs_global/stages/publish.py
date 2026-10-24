"""
Stage 5 — Publish the model to the lecturer's Hugging Face account.

Creates `config.hf_repo` (public model repository) if it doesn't exist and pushes, in one
commit, the GGUF, its `.corecs.json` profile and the model card from the packaged bundle.
Students' desktop apps and the bundle's Space template download the model from there.

The repository id was fixed when the job was submitted (hf_publish.check_token), so the bundle
already links to it. Without one (demo mode without a token) the stage records that nothing
was published. Re-running is safe: files the Hub already has are not uploaded again.
"""

import os

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import HfHubHTTPError

from .base import StageContext, StageError, atomic_write_json

HUB = "https://huggingface.co"


def _bundle_files(ctx: StageContext) -> list[tuple[str, os.PathLike]]:
    bundle = ctx.ws.package_dir / "bundle"
    profiles = sorted((bundle / "model").glob("*.corecs.json"))
    if not profiles:
        raise StageError("No packaged model found; the packaging stage must run first.")
    profile = profiles[-1]
    gguf = profile.with_name(profile.name.removesuffix(".corecs.json") + ".gguf")
    if not gguf.exists():
        raise StageError(f"The packaged model {gguf.name} is missing; re-run packaging.")
    files = [(gguf.name, gguf), (profile.name, profile)]
    card = bundle / "huggingface" / "README.md"
    if card.exists():
        files.append(("README.md", card))
    return files


def _hub_error(e: HfHubHTTPError, repo: str) -> StageError:
    status = getattr(e.response, "status_code", None)
    if status in (401, 403):
        return StageError(f"Hugging Face refused to publish to {repo} (HTTP {status}). The token may have "
                          "been revoked or be read-only; retry the job with a token that has Write access.")
    return StageError(f"Publishing to Hugging Face failed (HTTP {status}): {e}. Retry the job to resume.")


def run(ctx: StageContext) -> None:
    repo = ctx.config.hf_repo
    if not repo:
        ctx.reporter.start("Skipping publishing")
        atomic_write_json(ctx.ws.publish, {"published": False,
                                           "reason": "No Hugging Face token was provided."})
        ctx.log("No Hugging Face token was provided, so the model was not published.")
        ctx.reporter.finish("Not published (no Hugging Face token)")
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise StageError("The Hugging Face token is no longer stored (it is deleted when a job fails or is "
                         "cancelled). Retry the job and enter your Write token again.")

    files = _bundle_files(ctx)
    size_gb = sum(os.path.getsize(p) for _, p in files) / 1e9
    api = HfApi(token=token)
    ctx.reporter.start(f"Publishing to {HUB}/{repo}")
    try:
        api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
        private = bool(getattr(api.repo_info(repo, repo_type="model"), "private", False))
        ctx.reporter.update(0, 1, f"Uploading {size_gb:.1f} GB to {repo}", force=True)
        commit = api.create_commit(
            repo, repo_type="model",
            operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for name, path in files],
            commit_message=f"Publish {ctx.config.display_name} (CoreCS Global job {ctx.job_id})",
        )
    except HfHubHTTPError as e:
        raise _hub_error(e, repo) from e

    result = {
        "published": True,
        "repo": repo,
        "url": f"{HUB}/{repo}",
        "commit": getattr(commit, "oid", None),
        "files": [name for name, _ in files],
        "private": private,
    }
    atomic_write_json(ctx.ws.publish, result)
    if private:
        ctx.log(f"{repo} already existed as a private repository. Students' apps can't download a private "
                "model: make it public in the repository settings on Hugging Face.")
    ctx.log(f"Published {', '.join(result['files'])} to {result['url']}")
    ctx.reporter.finish(f"Published to {result['url']}")
