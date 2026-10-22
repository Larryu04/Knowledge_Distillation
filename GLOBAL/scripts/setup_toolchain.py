"""
setup_toolchain.py — install the llama.cpp toolchain CoreCS Global needs for packaging.

    python scripts/setup_toolchain.py                       # download the pinned build
    python scripts/setup_toolchain.py --from ../corecs/tools   # reuse an existing toolchain folder

Produces (default: GLOBAL/tools, or GLOBAL_TOOLCHAIN_DIR):
    tools/llama.cpp-src/   source tree: convert_hf_to_gguf.py + gguf-py (HF -> GGUF conversion)
    tools/llama.cpp-bin/   release binaries: llama-quantize (packaging) and llama-server + its
                           shared libraries (copied into every bundle's portable runtime)

Never `pip install -r tools/llama.cpp-src/requirements.txt` into the training venv: it pins
its own torch and would replace the CUDA build. The converter only needs its bundled gguf-py.
"""

import argparse
import json
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_BUILD = "b11269"      # same build as the desktop app's sidecar (runtime/desktop/scripts/fetch-llama.ps1)
REPO = "ggml-org/llama.cpp"
ROOT = Path(__file__).resolve().parent.parent


def asset_pattern() -> list[str]:
    system, machine = platform.system(), platform.machine().lower()
    if system == "Windows":
        return ["bin-win-vulkan-x64"]          # Vulkan: NVIDIA/AMD/Intel GPUs without the CUDA runtime
    if system == "Darwin":
        return ["bin-macos-arm64"] if machine in ("arm64", "aarch64") else ["bin-macos-x64"]
    return ["bin-ubuntu-vulkan-x64", "bin-ubuntu-x64"]


def download(url: str, dest: Path) -> None:
    print(f"[toolchain] downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "corecs-global-setup"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def extract(archive: Path, dest: Path) -> None:
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest, filter="data")


def flatten_bin(extracted: Path, dest: Path) -> None:
    """Release archives differ in layout (flat on Windows, build/bin/ elsewhere): copy the
    folder that actually contains llama-server."""
    hits = [p.parent for p in extracted.rglob("llama-server*") if p.is_file()]
    if not hits:
        raise SystemExit("[toolchain] llama-server not found in the release archive")
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(hits[0], dest)
    for f in dest.iterdir():
        if f.is_file() and not f.suffix:
            f.chmod(0o755)


def from_github(build: str, tools: Path) -> None:
    api = f"https://api.github.com/repos/{REPO}/releases/tags/{build}"
    with urllib.request.urlopen(urllib.request.Request(api, headers={"User-Agent": "corecs-global-setup"})) as r:
        assets = json.load(r)["assets"]
    asset = None
    for pattern in asset_pattern():
        asset = next((a for a in assets if pattern in a["name"] and a["name"].endswith((".zip", ".tar.gz"))), None)
        if asset:
            break
    if asset is None:
        raise SystemExit(f"[toolchain] no release asset for this platform in {build}: {asset_pattern()}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bin_archive = tmp / asset["name"]
        download(asset["browser_download_url"], bin_archive)
        extract(bin_archive, tmp / "bin")
        flatten_bin(tmp / "bin", tools / "llama.cpp-bin")

        src_archive = tmp / "src.zip"
        download(f"https://github.com/{REPO}/archive/refs/tags/{build}.zip", src_archive)
        extract(src_archive, tmp / "src")
        src_root = next((tmp / "src").iterdir())
        shutil.rmtree(tools / "llama.cpp-src", ignore_errors=True)
        shutil.copytree(src_root, tools / "llama.cpp-src")
    (tools / "BUILD.txt").write_text(f"{build} {asset['name']}\n", encoding="utf-8")


def from_existing(src: Path, tools: Path) -> None:
    for name in ("llama.cpp-src", "llama.cpp-bin"):
        if not (src / name).is_dir():
            raise SystemExit(f"[toolchain] {src / name} not found")
        print(f"[toolchain] copying {src / name}")
        shutil.rmtree(tools / name, ignore_errors=True)
        shutil.copytree(src / name, tools / name, ignore=shutil.ignore_patterns(".git", "build", "*.gguf"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", default=DEFAULT_BUILD, help=f"llama.cpp release tag (default {DEFAULT_BUILD})")
    ap.add_argument("--from", dest="src", type=Path, help="copy an existing tools folder instead of downloading")
    ap.add_argument("--dest", type=Path, default=Path(os.environ.get("GLOBAL_TOOLCHAIN_DIR", ROOT / "tools")))
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    if args.src:
        from_existing(args.src.resolve(), args.dest)
    else:
        from_github(args.build, args.dest)
    ok = (args.dest / "llama.cpp-src" / "convert_hf_to_gguf.py").exists() and any(
        (args.dest / "llama.cpp-bin").glob("llama-quantize*"))
    print(f"[toolchain] {'ready' if ok else 'INCOMPLETE'}: {args.dest}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
