"""
combine_fasta_labeled.py

Rebuilds combined_HUMAN.fasta, combined_MOUSE.fasta, combined_ANCESTOR.fasta
but rewrites every sample sequence header to include:
  - species/level  (e.g. HUMAN_GPX6_level01_chainA)
  - PDB template   (e.g. 1GP1_A)
  - sample number  (e.g. sample=1)

New header format:
  >HUMAN_L01_1GP1_A_s1   (for human level01, PDB 1GP1_A, sample 1)
  >MOUSE_L03_2F8A_A_s2
  >ANCESTOR_1GP1_A_s1

This means every sequence label in the plots shows:
  species · level · PDB · sample

Usage:
    python combine_fasta_labeled.py
"""

import re
from pathlib import Path

BASE_DIR = Path(r"D:\PhD_Thesis\MPNN\results\proteinmpnn_all\proteinmpnn_outputs")

# Map each top-level folder to a category
CATEGORY_MAP = {
    "HUMAN":    "HUMAN",
    "MOUSE":    "MOUSE",
    "ANCESTOR": "ANCESTOR",
}

def category_of(folder_name):
    name_upper = folder_name.upper()
    for key in CATEGORY_MAP:
        if name_upper.startswith(key):
            return CATEGORY_MAP[key]
    return None

def extract_level(folder_name):
    """
    Extract level number from folder name like HUMAN_GPX6_level01_chainA
    Returns 'L01', 'L02', etc., or '' if not found.
    """
    m = re.search(r"level(\d+)", folder_name, re.IGNORECASE)
    if m:
        return f"L{int(m.group(1)):02d}"
    if "WT" in folder_name.upper():
        return "WT"
    return "L00"

def parse_fasta_blocks(text):
    """
    Yield (header_line, sequence_lines) for each record.
    header_line does NOT include the leading '>'.
    """
    current_header = None
    current_seq    = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_header is not None:
                yield current_header, "".join(current_seq)
            current_header = line[1:]
            current_seq    = []
        else:
            current_seq.append(line)
    if current_header is not None:
        yield current_header, "".join(current_seq)

def make_label(category, level, pdb, sample_num):
    """
    Build a short, readable label:
      HUMAN_L01_1GP1_A_s1
      MOUSE_WT_6VPD_A_s2
      ANCESTOR_L00_1GP1_A_s1
    """
    # Shorten category
    cat_short = {"HUMAN": "HUM", "MOUSE": "MOU", "ANCESTOR": "ANC"}[category]
    # Clean PDB name (remove trailing comma if present)
    pdb_clean = pdb.rstrip(",").strip()
    return f"{cat_short}_{level}_{pdb_clean}_s{sample_num}"

# ── Walk directory tree ────────────────────────────────────────────────────────

buckets = {"HUMAN": [], "MOUSE": [], "ANCESTOR": []}

for top_folder in sorted(BASE_DIR.iterdir()):
    if not top_folder.is_dir():
        continue
    cat = category_of(top_folder.name)
    if cat is None:
        continue
    level = extract_level(top_folder.name)

    # Each subfolder is a PDB code e.g. 1GP1_A
    for pdb_folder in sorted(top_folder.iterdir()):
        if not pdb_folder.is_dir():
            continue
        pdb_name = pdb_folder.name  # e.g. "1GP1_A"

        # Find .fa files inside
        fa_files = sorted(pdb_folder.rglob("*.fa"))
        for fa_path in fa_files:
            text = fa_path.read_text(errors="replace")
            current_pdb = pdb_name   # default to folder name
            sample_counter = 0

            for header, seq in parse_fasta_blocks(text):
                # Detect template line (no 'sample=' in header)
                if "sample=" not in header:
                    # This is the native/template sequence — extract PDB from header
                    # Header like: "1GP1_A, score=1.3948, ..."
                    current_pdb = header.split(",")[0].strip()
                    # Write template with a _t0 label
                    new_header  = make_label(cat, level, current_pdb, 0).replace("_s0", "_template")
                else:
                    # Sample sequence — extract sample number
                    m = re.search(r"sample=(\d+)", header)
                    sample_num = int(m.group(1)) if m else (sample_counter + 1)
                    sample_counter += 1
                    new_header = make_label(cat, level, current_pdb, sample_num)

                buckets[cat].append((new_header, seq))

# ── Write output files ─────────────────────────────────────────────────────────

for cat, records in buckets.items():
    out_path = BASE_DIR / f"combined_{cat}_labeled.fasta"
    seq_count = 0
    with out_path.open("w") as f:
        for header, seq in records:
            f.write(f">{header}\n{seq}\n\n")
            seq_count += 1
    print(f"[{cat}]  {seq_count} sequences  →  {out_path}")

print("\nDone.")
print("Use combined_HUMAN_labeled.fasta, combined_MOUSE_labeled.fasta,")
print("combined_ANCESTOR_labeled.fasta as inputs to mpnn_analysis.py")
print("\nThen in mpnn_analysis.py update the three FASTA paths:")
print('  FASTA_ANCESTOR = Path(r"...\\combined_ANCESTOR_labeled.fasta")')
print('  FASTA_HUMAN    = Path(r"...\\combined_HUMAN_labeled.fasta")')
print('  FASTA_MOUSE    = Path(r"...\\combined_MOUSE_labeled.fasta")')
print("\nAnd update is_sample() to match the new header format:")
print('  def is_sample(r): return "_s" in r.id and "_template" not in r.id')