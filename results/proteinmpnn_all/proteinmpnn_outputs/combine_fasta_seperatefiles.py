"""
combine_fasta_labeled_keep_scores.py

Rebuilds combined_HUMAN.fasta, combined_MOUSE.fasta, combined_ANCESTOR.fasta
while:

• Creating clean labels for sequences
• PRESERVING original ProteinMPNN score metadata

Final header format example:

>HUM_L01_1GP1_A_s1 | score=1.3948, global_score=1.5123, seq_recovery=0.45
"""

import re
from pathlib import Path

BASE_DIR = Path(r"D:\PhD_Thesis\MPNN\results\proteinmpnn_all\proteinmpnn_outputs")

CATEGORY_MAP = {
    "HUMAN": "HUMAN",
    "MOUSE": "MOUSE",
    "ANCESTOR": "ANCESTOR",
}


def category_of(folder_name):
    name_upper = folder_name.upper()
    for key in CATEGORY_MAP:
        if name_upper.startswith(key):
            return CATEGORY_MAP[key]
    return None


def extract_level(folder_name):
    m = re.search(r"level(\d+)", folder_name, re.IGNORECASE)
    if m:
        return f"L{int(m.group(1)):02d}"

    if "WT" in folder_name.upper():
        return "WT"

    return "L00"


def parse_fasta_blocks(text):
    current_header = None
    current_seq = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if line.startswith(">"):
            if current_header is not None:
                yield current_header, "".join(current_seq)

            current_header = line[1:]
            current_seq = []

        else:
            current_seq.append(line)

    if current_header is not None:
        yield current_header, "".join(current_seq)


def make_label(category, level, pdb, sample_num):

    cat_short = {
        "HUMAN": "HUM",
        "MOUSE": "MOU",
        "ANCESTOR": "ANC",
    }[category]

    pdb_clean = pdb.rstrip(",").strip()

    return f"{cat_short}_{level}_{pdb_clean}_s{sample_num}"


# ─────────────────────────────────────────────
# WALK DIRECTORY TREE
# ─────────────────────────────────────────────

buckets = {
    "HUMAN": [],
    "MOUSE": [],
    "ANCESTOR": [],
}

for top_folder in sorted(BASE_DIR.iterdir()):

    if not top_folder.is_dir():
        continue

    cat = category_of(top_folder.name)

    if cat is None:
        continue

    level = extract_level(top_folder.name)

    for pdb_folder in sorted(top_folder.iterdir()):

        if not pdb_folder.is_dir():
            continue

        pdb_name = pdb_folder.name

        fa_files = sorted(pdb_folder.rglob("*.fa"))

        for fa_path in fa_files:

            text = fa_path.read_text(errors="replace")

            current_pdb = pdb_name
            sample_counter = 0

            for header, seq in parse_fasta_blocks(text):

                # Split header into main + metadata
                parts = header.split(",", 1)

                header_main = parts[0].strip()
                header_meta = parts[1].strip() if len(parts) > 1 else ""

                # TEMPLATE SEQUENCE
                if "sample=" not in header:

                    current_pdb = header_main

                    label = make_label(cat, level, current_pdb, 0)
                    label = label.replace("_s0", "_template")

                    new_header = f"{label} | {header_meta}" if header_meta else label

                # SAMPLE SEQUENCE
                else:

                    m = re.search(r"sample=(\d+)", header)
                    sample_num = int(m.group(1)) if m else sample_counter + 1
                    sample_counter += 1

                    label = make_label(cat, level, current_pdb, sample_num)

                    new_header = f"{label} | {header_meta}" if header_meta else label

                buckets[cat].append((new_header, seq))


# ─────────────────────────────────────────────
# WRITE OUTPUT
# ─────────────────────────────────────────────

for cat, records in buckets.items():

    out_path = BASE_DIR / f"combined_{cat}_labeled.fasta"

    seq_count = 0

    with out_path.open("w") as f:

        for header, seq in records:

            f.write(f">{header}\n")
            f.write(seq + "\n\n")

            seq_count += 1

    print(f"[{cat}] {seq_count} sequences → {out_path}")

print("\nDone.")