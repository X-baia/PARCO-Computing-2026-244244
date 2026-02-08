#!/usr/bin/env python3
"""
Download SuiteSparse matrices, extract ONLY from the inner folder with the same
name as the matrix, keep the first .mtx found, rename outputs when requested,
and delete the .tar.gz archives and temporary folders.

Final outputs (same folder as this script):
- af_shell9.mtx
- webbase-1M.mtx
- thermal2.mtx
- ecology2.mtx
- amazon0312.mtx
- poisson.mtx        (from FEMLAB/poisson3Da)
"""

from __future__ import annotations

import shutil
import tarfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# (group, suitesparse_name, output_filename)
MATRICES = [
    ("Schenk_AFE", "af_shell9", "af_shell9.mtx"),
    ("Williams", "webbase-1M", "webbase-1M.mtx"),
    ("Schmid", "thermal2", "thermal2.mtx"),
    ("McRae", "ecology2", "ecology2.mtx"),
    ("SNAP", "amazon0312", "amazon0312.mtx"),
    ("FEMLAB", "poisson3Da", "poisson.mtx"),  # renamed output
]

PRIMARY_BASE = "https://suitesparse-collection-website.herokuapp.com/MM"
FALLBACK_BASES = [
    "https://sparse-files.engr.tamu.edu/MM",
    # Uncomment if non-HTTPS is acceptable:
    # "http://sparse-files.engr.tamu.edu/MM",
]

CHUNK_SIZE = 1024 * 1024  # 1 MiB


def download_file(url: str, dest: Path, retries: int = 3) -> None:
    last_err: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=180) as r:
                total = r.headers.get("Content-Length")
                total_bytes = int(total) if total and total.isdigit() else None

                tmp = dest.with_suffix(dest.suffix + ".part")
                written = 0
                with open(tmp, "wb") as f:
                    while True:
                        chunk = r.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        written += len(chunk)
                        if total_bytes:
                            pct = 100.0 * written / total_bytes
                            print(f"      {written}/{total_bytes} bytes ({pct:.1f}%)", end="\r")

                if total_bytes:
                    print(" " * 80, end="\r")

                tmp.replace(dest)
            return

        except (HTTPError, URLError, TimeoutError) as e:
            last_err = e
            print(f"[warn] attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(1.5 * attempt)

    raise RuntimeError(f"Failed to download after {retries} attempts: {url}") from last_err


def download_archive(group: str, name: str, out_dir: Path) -> Path:
    archive = out_dir / f"{group}__{name}.tar.gz"

    if archive.exists() and archive.stat().st_size > 0:
        print(f"[skip] archive exists: {archive.name}")
        return archive

    errors: list[tuple[str, str]] = []
    for base in [PRIMARY_BASE] + FALLBACK_BASES:
        url = f"{base}/{group}/{name}.tar.gz"
        try:
            print(f"[get ] {group}/{name}")
            print(f"      {url}")
            download_file(url, archive)
            print(f"[ok  ] downloaded {archive.name}")
            return archive
        except Exception as e:
            errors.append((url, str(e)))
            part = archive.with_suffix(archive.suffix + ".part")
            if part.exists():
                part.unlink(missing_ok=True)

    msg = "\n".join([f"  - {u}: {err}" for u, err in errors])
    raise RuntimeError(f"All mirrors failed for {group}/{name}:\n{msg}")


def extract_from_inner_folder(
    archive: Path,
    out_dir: Path,
    matrix_name: str,
    out_name: str,
) -> Path:
    """
    Enforces SuiteSparse layout:
      - extract archive
      - locate directory named exactly <matrix_name>
      - search ONLY inside that directory for *.mtx
      - keep the first one (sorted)
    """
    tmp_dir = out_dir / f"__extract__{matrix_name}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(path=tmp_dir)

        # Find inner folder named exactly matrix_name
        inner_dirs = sorted(p for p in tmp_dir.rglob(matrix_name) if p.is_dir())
        if not inner_dirs:
            raise RuntimeError(
                f"Inner folder '{matrix_name}' not found in {archive.name}"
            )

        inner = inner_dirs[0]

        # Only search inside that folder
        mtx_files = sorted(inner.rglob("*.mtx"))
        if not mtx_files:
            raise RuntimeError(
                f"No .mtx found inside inner folder '{inner}'"
            )

        chosen = mtx_files[0]
        dest = out_dir / out_name
        if dest.exists():
            dest.unlink()

        shutil.move(str(chosen), str(dest))
        return dest

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> int:
    out_dir = Path(__file__).resolve().parent
    print(f"Output folder: {out_dir}\n")

    for group, ss_name, out_name in MATRICES:
        print(f"=== {group}/{ss_name} ===")
        dest = out_dir / out_name

        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] exists: {dest.name}\n")
            continue

        archive = download_archive(group, ss_name, out_dir)

        extracted = extract_from_inner_folder(
            archive=archive,
            out_dir=out_dir,
            matrix_name=ss_name,
            out_name=out_name,
        )

        archive.unlink(missing_ok=True)

        print(f"[ok  ] saved: {extracted.name}")
        print(f"[ok  ] cleaned: {archive.name} + temp folder\n")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
