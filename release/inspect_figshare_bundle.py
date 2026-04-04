#!/usr/bin/env python3
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path


FILES = [
    (
        "code.zip",
        "https://ndownloader.figshare.com/files/54861755",
        20229065,
        "2a93762a87db6a3e9065cbed16c87abf",
    ),
    (
        "datasets.zip",
        "https://ndownloader.figshare.com/files/55028555",
        2359373145,
        "774bae0908abb81a419adaa0cce83b2d",
    ),
    (
        "models.zip",
        "https://ndownloader.figshare.com/files/55032923",
        5419112931,
        "6b8702e8a52a3b30463a4101dfff28b5",
    ),
]


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_download(path: Path, url: str, expected_size: int) -> None:
    if path.exists() and path.stat().st_size == expected_size:
        return

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as response, path.open(
        "wb"
    ) as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)


def zip_preview(path: Path) -> dict:
    result = subprocess.run(
        ["unzip", "-l", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        lines = result.stdout.splitlines()
        return {
            "zip_ok": True,
            "zip_preview": lines[:80],
        }

    return {
        "zip_ok": False,
        "zip_error": result.stderr[:4000],
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: inspect_figshare_bundle.py <target-dir>", file=sys.stderr)
        return 2

    target_dir = Path(sys.argv[1]).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for name, url, expected_size, expected_md5 in FILES:
        target = target_dir / name
        print(f"DOWNLOADING {name}", flush=True)
        ensure_download(target, url, expected_size)

        entry = {
            "name": name,
            "path": str(target),
            "size": target.stat().st_size,
            "expected_size": expected_size,
            "md5": md5sum(target),
            "expected_md5": expected_md5,
        }
        entry.update(zip_preview(target))
        summary.append(entry)

    print("MIST_FIGSHARE_SUMMARY_START")
    print(json.dumps(summary, indent=2))
    print("MIST_FIGSHARE_SUMMARY_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
