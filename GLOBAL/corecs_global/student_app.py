"""
The student desktop app download: one generic installer, told which model to pull.

The CoreCS desktop installer is the same build for every lecturer (no per-job rebuild, so no
compiler toolchain on the server and no re-signing). What makes it "this lecturer's app" is a
small `corecs-model.json` manifest shipped next to it in a ZIP:

    <slug>-tutor-app/
      CoreCS_x.y.z_x64-setup.exe     the generic installer
      corecs-model.json              {"repo": "<hf-user>/<slug>-tutor-gguf", "gguf": ..., ...}
      README.txt

The installer copies the manifest into the install folder, and on first launch the app pulls
that model from the Hugging Face Hub (runtime/desktop/src-tauri/src/provision.rs). Without a
manifest (the plain installer), the app pulls the default model, CoreCS v8.

The repository is the one the publish stage pushed the model to (stages/publish.py), so the
manifest is generated at download time from deploy.json; no lecturer input is needed.
"""

import io
import json
import re
import zipfile
from pathlib import Path

MANIFEST_FILE = "corecs-model.json"
MANIFEST_VERSION = 1
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")   # what the app accepts (provision.rs)


class ManifestError(ValueError):
    pass


def build_manifest(guide: dict, repo: str = "", revision: str = "main") -> dict:
    """The manifest for a finished job (`guide` = its deploy.json). `repo` defaults to the
    repository the publish stage pushed to; passing one covers older, manually published jobs."""
    repo = (repo or "").strip()
    if not repo:
        repo = (guide.get("model_repo") if guide.get("published") else "") or ""
        if not repo:
            raise ManifestError("This model was not published to Hugging Face, so the desktop app has "
                                "nothing to download. Run the job again with a Hugging Face Write token.")
    if not REPO_RE.match(repo):
        raise ManifestError("Invalid Hugging Face repository; use owner/name.")
    if not re.match(r"^[A-Za-z0-9_.-]+$", revision or ""):
        raise ManifestError("Invalid revision.")
    manifest = {
        "version": MANIFEST_VERSION,
        "source": "corecs-global",
        "display_name": guide["display_name"],
        "repo": repo,
        "revision": revision,
        "gguf": guide["gguf"],
        "profile": guide["profile"],
    }
    # Lets the app verify a 1-6 GB download: the publish stage pushed exactly this file.
    if guide.get("gguf_sha256"):
        manifest["sha256"] = guide["gguf_sha256"]
    if guide.get("gguf_bytes"):
        manifest["size_bytes"] = guide["gguf_bytes"]
    return manifest


def readme(manifest: dict, installer_name: str) -> str:
    title = f"{manifest['display_name']} (desktop app)"
    return f"""{title}
{'=' * len(title)}

!!! IMPORTANT: EXTRACT ALL FILES BEFORE RUNNING THE SETUP !!!

If you double-click {installer_name} while it is still inside the ZIP, Windows runs it
without {MANIFEST_FILE}. The app then installs the general CoreCS v8 model instead of
your course's model.

How to install
--------------
1. Right-click the ZIP file -> "Extract All..." -> "Extract".
2. Open the extracted folder. Check that {installer_name} and {MANIFEST_FILE} are
   next to each other in it.
3. Run {installer_name} from that extracted folder.
4. Open CoreCS. On first launch it downloads your course's model once (several GB, from
   https://huggingface.co/{manifest['repo']}). After that it works fully offline.

Already installed it from inside the ZIP and got the general model? Extract the ZIP and run
the setup again from the extracted folder. The app then downloads your course's model.

{MANIFEST_FILE} tells the app which model to download. Keep it next to the installer.
"""


def build_zip(installer: Path, manifest: dict, root: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(installer, f"{root}/{installer.name}")
        zf.writestr(f"{root}/{MANIFEST_FILE}", json.dumps(manifest, indent=2, ensure_ascii=False))
        zf.writestr(f"{root}/README.txt", readme(manifest, installer.name))
    return buf.getvalue()
