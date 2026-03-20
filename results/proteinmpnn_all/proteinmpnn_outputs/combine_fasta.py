"""
Combines all .fa files from ProteinMPNN output directories into a single FASTA file:
  - combined_ALL.fasta

Usage:
    python combine_fasta_all.py

Edit BASE_DIR below if needed.
"""

import os
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(r"D:\PhD_Thesis\MPNN\results\proteinmpnn_all\proteinmpnn_outputs")
OUTPUT_DIR = BASE_DIR  # change if you want the output file elsewhere
OUTPUT_FILE = OUTPUT_DIR / "combined_ALL.fasta"
# ──────────────────────────────────────────────────────────────────────────────


def main():
    print(f"\nScanning: {BASE_DIR}\n")

    all_fa_files: list[Path] = []

    for top_folder in sorted(BASE_DIR.iterdir()):
        if not top_folder.is_dir():
            continue
        fa_files = sorted(top_folder.rglob("*.fa"))
        if not fa_files:
            print(f"  [warn] No .fa files found under {top_folder.name}")
        else:
            print(f"  [ok]   {top_folder.name}  →  {len(fa_files)} .fa file(s)")
        all_fa_files.extend(fa_files)

    print(f"\nTotal .fa files found: {len(all_fa_files)}")

    seq_count = 0
    with OUTPUT_FILE.open("w") as out:
        for fa in all_fa_files:
            text = fa.read_text(errors="replace").strip()
            if not text:
                continue
            out.write(text)
            out.write("\n\n")
            seq_count += text.count(">")

    print(f"Total sequences written: {seq_count}")
    print(f"Output file: {OUTPUT_FILE}\n")
    print("Done!")


if __name__ == "__main__":
    main()