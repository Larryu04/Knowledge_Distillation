"""
Stage 6 — Deployment instructions.

Builds a step-by-step guide filled in with this job's real file names and Hugging Face
repository: test locally, where the model was published (the publish stage uploads it to the
lecturer's account), share it with students, and optionally host an online demo on a Hugging
Face Space. The same guide ships inside the bundle as DEPLOY.md and is rendered by the web UI
from deploy.json. Jobs run without a token (demo mode) have no repository: the guide then
covers offline sharing only (and the Space template keeps a `{HF_USER}` placeholder).
"""

import hashlib

from .base import StageContext, StageError, atomic_write_json, read_json, slugify

HF_USER = "{HF_USER}"
SPACE_PORT = 7860


def repo_ids(hf_repo: str, slug: str) -> tuple[str, str]:
    """(model repo, Space repo). `hf_repo` is the job's published repo, or "" (placeholders)."""
    owner = hf_repo.split("/")[0] if hf_repo else HF_USER
    return hf_repo or f"{HF_USER}/{slug}-tutor-gguf", f"{owner}/{slug}-tutor-demo"


def build_guide(ctx: StageContext, profile: dict, gguf: str, bundle: str, slug: str,
                has_desktop: bool, has_bins: bool) -> dict:
    published = bool(ctx.config.hf_repo)
    model_repo, space_repo = repo_ids(ctx.config.hf_repo, slug)
    profile_file = gguf.removesuffix(".gguf") + ".corecs.json"
    link = f"https://huggingface.co/{model_repo}/resolve/main/{gguf}?download=true"
    root = f"{slug}-tutor"
    runtime_note = ("" if has_bins else
                    "\n\nThis bundle has no llama-server binary for your OS: install llama.cpp first "
                    "(`winget install llama.cpp` on Windows, `brew install llama.cpp` on macOS) and the "
                    "start script will use it.")
    desktop_text = (
        "**Option B: the CoreCS desktop app (Windows).** Download the **student app** ZIP from the "
        "CoreCS Global deploy page and share it. It holds the installer plus `corecs-model.json`, which "
        "names your Hugging Face repository: on first launch the app downloads your model by itself. "
        "Students must extract the ZIP completely before running the installer, or they get the "
        "default CoreCS model instead."
        if has_desktop and published else
        ("**Option B: the CoreCS desktop app (Windows).** Students install it from the bundle's "
         "`runtime/desktop` folder" if has_desktop else
         "**Option B: the CoreCS desktop app.** Build it from the `runtime/desktop` sources of the platform")
        + f", then copy `{gguf}` and `{profile_file}` into `%APPDATA%\\com.corecs.app\\models`."
    )
    sections = [
        {
            "id": "download",
            "title": "Download your bundle",
            "body": (f"Your bundle `{bundle}` contains everything needed to run the tutor offline:\n\n"
                     f"- `model/{gguf}`: the fine-tuned, quantized model\n"
                     f"- `model/{profile_file}`: the prompt preamble and sampling settings it was trained with\n"
                     "- `runtime/portable/`: a zero-install chat app (llama.cpp server + browser UI)\n"
                     + ("- `runtime/desktop/`: the CoreCS desktop installer\n" if has_desktop else "")
                     + ("- `adapter/`: the LoRA adapter (for further training or Hugging Face)\n" if ctx.config.include_adapter else "")
                     + "- `huggingface/`: a ready model card and a Space template\n"
                     "- `dataset/`: the generated training data and structured syllabi, for your review"),
            "commands": [],
        },
        {
            "id": "test",
            "title": "Test it on your own computer",
            "body": ("Unzip the bundle and start the portable runtime. A browser tab opens at "
                     "http://127.0.0.1:8088 once the model is loaded (10-60 seconds). Ask a few questions "
                     "from each course before sharing it with students." + runtime_note),
            "commands": [
                {"label": "Windows", "code": f"{root}\\runtime\\portable\\start-windows.bat"},
                {"label": "macOS / Linux", "code": f"sh {root}/runtime/portable/start-macos-linux.sh"},
            ],
        },
        {
            "id": "hub",
            "title": "Your model on Hugging Face",
            "body": (f"CoreCS Global published your model automatically to your Hugging Face account: "
                     f"**https://huggingface.co/{model_repo}**. The repository holds `{gguf}`, `{profile_file}` "
                     "and a model card. It is public, so students need no Hugging Face account. If you make "
                     "it private, the desktop app and the download link stop working for students."
                     if published else
                     "This job ran without a Hugging Face token, so the model was not published online. "
                     "Students can still use it offline from the bundle (see below). To publish it "
                     "automatically, run the job again with a Hugging Face token that has **Write** access."),
            "commands": [{"label": "Model page", "code": f"https://huggingface.co/{model_repo}"}] if published else [],
        },
        {
            "id": "share",
            "title": "Share it with your students",
            "body": (("Post these links on your course site (Moodle, Canvas, ...).\n\n"
                      f"- **Model page:** https://huggingface.co/{model_repo}\n"
                      f"- **Direct model download:** {link}\n\n"
                      "**Option A: the portable runtime (any OS).** Share the bundle's `runtime/portable` "
                      "folder (or the whole ZIP via your university drive). Students put the downloaded "
                      "`.gguf` and `.corecs.json` into a `model` folder inside `portable`, then run the "
                      "start script.\n\n"
                      if published else
                      "**Option A: the portable runtime (any OS).** Share the whole bundle ZIP (for example "
                      "via your university drive). Students unzip it and run the start script.\n\n")
                     + desktop_text + "\n\n"
                     "Everything runs on the student's computer: questions never leave it. A 7B model "
                     "needs about 6 GB of free RAM (or VRAM); a GPU makes it much faster."),
            "commands": [{"label": "Download link", "code": link}] if published else [],
        },
        *([{
            "id": "space",
            "title": "Optional: host an online demo on Hugging Face Spaces",
            "body": ("Useful for students on Chromebooks or tablets. Create a new Space at "
                     "https://huggingface.co/new-space with the **Docker** SDK (Blank template) named "
                     f"`{space_repo.split('/')[1]}`, then upload the template from the bundle's "
                     "`huggingface/space` folder (Files → Add file → Upload files, or the command below). "
                     "The Space downloads your published model when it starts.\n\n"
                     "Free CPU Spaces answer slowly with 7B models (a few tokens per second) and sleep "
                     "when idle; a GPU Space upgrade is fast but paid. Unlike the offline runtime, "
                     "questions are processed on Hugging Face's servers."),
            "commands": [
                {"label": "Upload the Space template (after `pip install -U huggingface_hub` and `hf auth login`)",
                 "code": f"cd {root}\nhf upload {space_repo} huggingface/space . --repo-type space"},
                {"label": "Demo URL", "code": f"https://huggingface.co/spaces/{space_repo}"},
            ],
        }] if published else []),
        {
            "id": "checklist",
            "title": "Before you share: checklist",
            "body": ("- Spot-check answers from every course (the `dataset/` folder shows what it learned).\n"
                     f"- The model inherits the license of its base model, **{profile['source']['base_model']}**; "
                     "check it allows your use (some models require attribution or acceptance of terms).\n"
                     "- Tell students the tutor can make mistakes and should not be used during exams "
                     "unless you allow it.\n"
                     "- Re-run the platform whenever the syllabus changes: the bundle is self-contained, "
                     "so a new version is just new files."),
            "commands": [],
        },
    ]
    return {
        "title": f"Deploy {profile['display_name']}",
        "display_name": profile["display_name"],
        "gguf": gguf,
        "profile": profile_file,
        "bundle": bundle,
        "published": published,
        "model_repo": model_repo if published else None,
        "space_repo": space_repo if published else None,
        "sections": sections,
    }


def render_markdown(guide: dict) -> str:
    out = [f"# {guide['title']}\n"]
    for i, sec in enumerate(guide["sections"], 1):
        out.append(f"## {i}. {sec['title']}\n\n{sec['body']}\n")
        for cmd in sec["commands"]:
            out.append(f"{cmd['label']}:\n\n```\n{cmd['code']}\n```\n")
    return "\n".join(out)


def bundle_readme(profile: dict, gguf: str) -> str:
    return f"""{profile['display_name']}
{'=' * len(profile['display_name'])}

An offline AI tutor for: {', '.join(profile['courses'])}

Quick start
-----------
Windows:        double-click runtime\\portable\\start-windows.bat
macOS / Linux:  sh runtime/portable/start-macos-linux.sh

A browser tab opens at http://127.0.0.1:8088 when the model is ready.
Everything runs on this computer; nothing is sent to the internet.

Model file: model/{gguf}
Publishing and sharing: see DEPLOY.md
"""


def space_files(profile: dict, gguf: str, model_repo: str) -> dict[str, str]:
    ctx_len = profile.get("context_length", 4096)
    dockerfile = f"""# Hugging Face Space (Docker SDK): llama.cpp server + the tutor's chat UI.
# MODEL_REPO can also be set as a Space variable (Settings -> Variables) instead of editing this.
FROM ghcr.io/ggml-org/llama.cpp:server

ENV MODEL_REPO={model_repo} \\
    MODEL_FILE={gguf} \\
    LLAMA_CACHE=/tmp/llama-cache
COPY webui /webui

EXPOSE {SPACE_PORT}
ENTRYPOINT ["/bin/sh", "-c"]
CMD ["/app/llama-server --hf-repo \\"$MODEL_REPO\\" --hf-file \\"$MODEL_FILE\\" --host 0.0.0.0 --port {SPACE_PORT} --ctx-size {ctx_len} --parallel 2 --jinja --path /webui"]
"""
    readme = f"""---
title: {profile['display_name']}
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: {SPACE_PORT}
pinned: false
---

Online demo of **{profile['display_name']}**, an AI tutor fine-tuned with CoreCS Global.
Model: https://huggingface.co/{model_repo}
"""
    return {"Dockerfile": dockerfile, "README.md": readme}


def run(ctx: StageContext) -> None:
    ctx.reporter.start("Writing deployment guide")
    bundle_dir = ctx.ws.package_dir / "bundle"
    profiles = sorted((bundle_dir / "model").glob("*.corecs.json"))
    if not profiles:
        raise StageError("No packaged model found; the packaging stage must run first.")
    profile = read_json(profiles[-1])
    gguf = profiles[-1].name.removesuffix(".corecs.json") + ".gguf"
    slug = slugify(ctx.config.program_name, fallback="tutor")
    has_bins = any((bundle_dir / "runtime" / "portable" / "bin").glob("llama-server*"))
    has_desktop = (bundle_dir / "runtime" / "desktop").is_dir()
    guide = build_guide(ctx, profile, gguf, f"{slug}-bundle.zip", slug, has_desktop, has_bins)
    # For the desktop app's model manifest (student_app.py): lets it verify the Hub download.
    gguf_path = bundle_dir / "model" / gguf
    if gguf_path.exists():
        ctx.reporter.update(0, 1, "Checksumming the model for the desktop app", force=True)
        digest = hashlib.sha256()
        with open(gguf_path, "rb") as f:
            while chunk := f.read(8 << 20):
                digest.update(chunk)
        guide["gguf_sha256"], guide["gguf_bytes"] = digest.hexdigest(), gguf_path.stat().st_size
    guide["samples"] = (read_json(ctx.ws.train_meta, default={}) or {}).get("samples", [])
    publish = read_json(ctx.ws.publish, default=None)
    if publish is not None:
        guide["publish"] = publish
        guide["published"] = bool(publish.get("published"))
    atomic_write_json(ctx.ws.deploy, guide)
    ctx.reporter.finish("Deployment guide ready")
