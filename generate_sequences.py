"""
1_generate_sequences.py
================================================================
ProteinMPNN — Generate NEW sequences from each mutant backbone.

PURPOSE:
    Takes each mutant PDB and generates N new sequences that are
    compatible with that backbone. These are NOT the original
    sequences — ProteinMPNN samples what sequences would fold
    into that 3D shape.

OUTPUT:
    GPX6MPNN_generated/
    ├── L0/
    │   ├── E143S/seqs/E143S.fa    ← N generated sequences
    │   ├── G102S/seqs/G102S.fa
    │   └── ...
    └── all_generated_sequences.fasta   ← everything merged
    └── generated_sequences_summary.csv ← scores table

FASTA HEADER:
    >E143S, score=0.73, global_score=0.93, seq_recovery=0.57, T=0.1, sample=1
    score        = fit of generated sequence to backbone (lower = better)
    global_score = fit over all residues
    seq_recovery = % similarity to the original sequence
    sample       = sequence number (1 to NUM_SEQ)

USAGE:
    python 1_generate_sequences.py
================================================================
"""

import os
import subprocess
import glob
import re
import shutil
import tempfile
import pandas as pd
from pathlib import Path

# ================================================================
#  CONFIGURATION — edit these paths
# ================================================================
PROTEINMPNN = "/home/hp/nayanika/github/GPX6MPNN"
PDB_BASE    = "/home/hp/nayanika/github/GPX6MPNN/Mouse/RS_TS_mouse"
OUT_BASE    = "/home/hp/nayanika/github/GPX6MPNN/Mouse/GPX6MPNN_generated"

NUM_SEQ       = 10      # sequences to generate per PDB
TEMP          = "0.1"   # 0.1=conservative  0.2=moderate  0.3=diverse
SEED          = 37
USE_SOLUBLE   = True    # True = soluble model weights (good for cytosolic GPX6)
SKIP_SOLVATED = True    # True = skip *_solvated.pdb files
# ================================================================

# Residues to strip entirely (not protein)
STRIP_RESIDUES = {"HOH", "WAT", "SOL", "NA", "CL", "MG", "ZN", "CA"}

# GROMACS → standard PDB residue name mapping
# CYX = disulfide cysteine      → CYS
# HID = histidine (delta-H)     → HIS
# HIE = histidine (epsilon-H)   → HIS
# HIP = histidine (protonated)  → HIS
# PRX = selenocysteine (GPX6!)  → SEC  (ProteinMPNN knows SEC)
RENAME_RESIDUES = {
    "CYX": "CYS",
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",
    "PRX": "SEC",
}

def run(cmd):
    """Run a shell command, return True if successful."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    STDERR: {result.stderr.strip()}")
    return result.returncode == 0


def fix_pdb(src_pdb, dst_pdb, chain="A"):
    """
    Clean a GROMACS-output PDB for ProteinMPNN:
      1. Strip non-protein residues (water, ions)
      2. Rename non-standard residues (CYX→CYS, HID→HIS, PRX→SEC)
      3. Fix blank chain ID in column 22
      4. Force HETATM → ATOM for protein residues
    """
    with open(src_pdb) as f_in, open(dst_pdb, "w") as f_out:
        for line in f_in:
            if line.startswith(("ATOM  ", "HETATM")):
                resname = line[17:20].strip()

                # 1. Strip non-protein residues
                if resname in STRIP_RESIDUES:
                    continue

                # 2. Rename non-standard residues
                if resname in RENAME_RESIDUES:
                    new_resname = RENAME_RESIDUES[resname].ljust(3)
                    line = line[:17] + new_resname + line[20:]

                # 3. Fix blank chain ID (column 22, 0-indexed 21)
                if len(line) > 21 and line[21].strip() == "":
                    line = line[:21] + chain + line[22:]

                # 4. Force HETATM → ATOM for protein residues
                if line.startswith("HETATM"):
                    line = "ATOM  " + line[6:]

            f_out.write(line)


def main():
    # ── validate paths ──────────────────────────────────────────
    assert os.path.isdir(PROTEINMPNN), f"ProteinMPNN repo not found:\n  {PROTEINMPNN}"
    assert os.path.isdir(PDB_BASE),    f"PDB directory not found:\n  {PDB_BASE}"
    os.makedirs(OUT_BASE, exist_ok=True)

    soluble_flag = "--use_soluble_model" if USE_SOLUBLE else ""

    print("=" * 62)
    print("  ProteinMPNN — GENERATE NEW SEQUENCES  |  Mouse GPX6")
    print("=" * 62)
    print(f"  ProteinMPNN : {PROTEINMPNN}")
    print(f"  PDB input   : {PDB_BASE}")
    print(f"  Output      : {OUT_BASE}")
    print(f"  Seqs/PDB    : {NUM_SEQ}")
    print(f"  Temperature : {TEMP}")
    print(f"  Model       : {'soluble' if USE_SOLUBLE else 'vanilla'}")
    print(f"  Skip solv.  : {SKIP_SOLVATED}")
    print("=" * 62)

    lib_dirs   = sorted(glob.glob(os.path.join(PDB_BASE, "L*/")))
    records    = []
    failed     = []
    total_seq  = 0

    # ── main loop ───────────────────────────────────────────────
    for lib_dir in lib_dirs:
        lib_name = os.path.basename(lib_dir.rstrip("/"))
        lib_out  = os.path.join(OUT_BASE, lib_name)
        os.makedirs(lib_out, exist_ok=True)

        pdbs = sorted(glob.glob(os.path.join(lib_dir, "*.pdb")))
        if SKIP_SOLVATED:
            pdbs = [p for p in pdbs if "_solvated" not in p]

        print(f"\n  [{lib_name}]  {len(pdbs)} PDB(s)")
        print("  " + "-" * 50)

        for pdb in pdbs:
            variant     = Path(pdb).stem
            variant_out = os.path.join(lib_out, variant)
            jsonl_path  = os.path.join(variant_out, "parsed.jsonl")
            os.makedirs(variant_out, exist_ok=True)

            print(f"  -> {variant}", end="  ", flush=True)

            # STEP 1: clean PDB, then parse to JSONL
            fixed_pdb = os.path.join(variant_out, f"{variant}_clean.pdb")
            fix_pdb(pdb, fixed_pdb, chain="A")

            with tempfile.TemporaryDirectory() as tmpdir:
                shutil.copy(fixed_pdb, os.path.join(tmpdir, os.path.basename(pdb)))
                ok = run(
                    f"python {PROTEINMPNN}/helper_scripts/parse_multiple_chains.py "
                    f"--input_path {tmpdir} "
                    f"--output_path {jsonl_path}"
                )
            if not ok:
                print("FAILED (parse step)")
                failed.append(f"{lib_name}/{variant}")
                continue

            # STEP 2: generate new sequences using the cleaned PDB
            ok = run(
                f"python {PROTEINMPNN}/protein_mpnn_run.py "
                f"--pdb_path {fixed_pdb} "
                f"--jsonl_path {jsonl_path} "
                f"--out_folder {variant_out}/ "
                f"--num_seq_per_target {NUM_SEQ} "
                f"--sampling_temp {TEMP} "
                f"--seed {SEED} "
                f"--batch_size 1 "
                f"--save_score 1 "
                f"{soluble_flag}"
            )
            if not ok:
                print("FAILED (ProteinMPNN step)")
                failed.append(f"{lib_name}/{variant}")
                continue

            # STEP 3: parse scores from FASTA headers
            fa_files = glob.glob(os.path.join(variant_out, "seqs", "*.fa"))
            if fa_files:
                with open(fa_files[0]) as f:
                    content = f.read()
                headers = [l for l in content.splitlines() if l.startswith(">")]
                seqs    = [l for l in content.splitlines() if not l.startswith(">") and l.strip()]

                for i, (header, seq) in enumerate(zip(headers, seqs)):
                    s    = re.search(r'(?<![_a-z])score=([0-9.]+)',  header)
                    gs   = re.search(r'global_score=([0-9.]+)',       header)
                    rec  = re.search(r'seq_recovery=([0-9.]+)',        header)
                    samp = re.search(r'sample=([0-9]+)',               header)
                    records.append({
                        "library":      lib_name,
                        "variant":      variant,
                        "sample":       int(samp.group(1))  if samp else i + 1,
                        "score":        float(s.group(1))   if s    else None,
                        "global_score": float(gs.group(1))  if gs   else None,
                        "seq_recovery": float(rec.group(1)) if rec  else None,
                        "sequence":     seq
                    })

                print(f"OK — {len(headers)} sequences generated")
                total_seq += len(headers)
            else:
                print("WARNING: no output file found")
                failed.append(f"{lib_name}/{variant}")

    # ── merge all FASTAs ────────────────────────────────────────
    merged_fasta = os.path.join(OUT_BASE, "all_generated_sequences.fasta")
    with open(merged_fasta, "w") as out_f:
        for fa_file in sorted(glob.glob(os.path.join(OUT_BASE, "L*/*/seqs/*.fa"))):
            parts   = Path(fa_file).parts
            lib     = next((p for p in parts if p.startswith("L") and p[1:].isdigit()), "?")
            variant = parts[-3]
            with open(fa_file) as in_f:
                for line in in_f:
                    if line.startswith(">"):
                        out_f.write(line.rstrip() + f" | lib={lib} | variant={variant}\n")
                    else:
                        out_f.write(line)

    # ── save summary CSV ────────────────────────────────────────
    df = pd.DataFrame(records)
    summary_csv = os.path.join(OUT_BASE, "generated_sequences_summary.csv")
    df.sort_values(["library", "variant", "score"]).to_csv(summary_csv, index=False)

    # ── final report ────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  DONE")
    print("=" * 62)
    print(f"  Total sequences generated : {total_seq}")
    if failed:
        print(f"  Failed ({len(failed)})           : {failed}")
    print(f"\n  Per-variant FASTAs : {OUT_BASE}/L*/VARIANT/seqs/*.fa")
    print(f"  Merged FASTA       : {merged_fasta}")
    print(f"  Summary CSV        : {summary_csv}")
    print("\n  Preview — top 10 by score (best fit first):")
    print("-" * 62)
    if not df.empty:
        print(df.dropna(subset=["score"])
                .sort_values("score")
                [["library", "variant", "sample", "score", "global_score", "seq_recovery"]]
                .head(10)
                .to_string(index=False))
    print("=" * 62)


if __name__ == "__main__":
    main()
