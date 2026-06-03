"""
1_generate_sequences.py
================================================================
ProteinMPNN — Generate NEW sequences from each mutant backbone.
Supports BOTH Human and Mouse GPX6 datasets.

PURPOSE:
    Takes each mutant PDB and generates N new sequences that are
    compatible with that backbone. These are NOT the original
    sequences — ProteinMPNN samples what sequences would fold
    into that 3D shape.

OUTPUT:
    GPX6MPNN_generated/
    ├── Mouse/
    │   ├── L0/
    │   │   ├── E143S/seqs/E143S.fa
    │   │   └── ...
    │   ├── all_generated_sequences.fasta
    │   └── generated_sequences_summary.csv
    └── Human/
        ├── L0/
        │   ├── E143S/seqs/E143S.fa
        │   └── ...
        ├── all_generated_sequences.fasta
        └── generated_sequences_summary.csv

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

# Each entry: (species_label, pdb_input_dir, output_dir)
DATASETS = [
    (
        "Mouse",
        "/home/hp/nayanika/github/GPX6MPNN/Mouse/RS_TS_mouse",
        "/home/hp/nayanika/github/GPX6MPNN/Mouse/GPX6MPNN_generated",
    ),
    (
        "Human",
        "/home/hp/nayanika/github/GPX6MPNN/Human/RS_TS_human",
        "/home/hp/nayanika/github/GPX6MPNN/Human/GPX6MPNN_generated",
    ),
]

NUM_SEQ       = 10      # sequences to generate per PDB
TEMP          = "0.1"   # 0.1=conservative  0.2=moderate  0.3=diverse
SEED          = 37
USE_SOLUBLE   = True    # True = soluble model weights (good for cytosolic GPX6)
# ================================================================

# Residues to strip entirely (not protein)
STRIP_RESIDUES = {"HOH", "WAT", "SOL", "NA", "CL", "MG", "ZN", "CA"}

# GROMACS → standard PDB residue name mapping
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


def process_dataset(species, pdb_base, out_base, soluble_flag):
    """
    Run ProteinMPNN on all non-solvated PDBs under pdb_base.
    Returns (records_list, failed_list, total_seq_count).
    """
    print()
    print("=" * 62)
    print(f"  SPECIES : {species}")
    print("=" * 62)
    print(f"  PDB input : {pdb_base}")
    print(f"  Output    : {out_base}")
    print("=" * 62)

    os.makedirs(out_base, exist_ok=True)

    records   = []
    failed    = []
    total_seq = 0

    # Discover library sub-folders (L0, L1, …) OR use root directly
    lib_dirs = sorted(glob.glob(os.path.join(pdb_base, "L*/")))
    if not lib_dirs:
        # No L* sub-folders — treat the root itself as one "library"
        lib_dirs = [pdb_base + os.sep]

    for lib_dir in lib_dirs:
        lib_name = os.path.basename(lib_dir.rstrip("/"))
        lib_out  = os.path.join(out_base, lib_name)
        os.makedirs(lib_out, exist_ok=True)

        # Collect PDBs — only plain .pdb, never _solvated.pdb
        all_pdbs = sorted(glob.glob(os.path.join(lib_dir, "*.pdb")))
        pdbs = [
            p for p in all_pdbs
            if "_solvated" not in Path(p).stem   # skip *_solvated*
        ]

        skipped = len(all_pdbs) - len(pdbs)
        print(f"\n  [{lib_name}]  {len(pdbs)} PDB(s)  "
              f"(skipped {skipped} solvated)")
        print("  " + "-" * 50)

        if not pdbs:
            print("  No eligible PDBs found — skipping this library.")
            continue

        for pdb in pdbs:
            variant     = Path(pdb).stem
            variant_out = os.path.join(lib_out, variant)
            jsonl_path  = os.path.join(variant_out, "parsed.jsonl")
            os.makedirs(variant_out, exist_ok=True)

            print(f"  -> {variant}", end="  ", flush=True)

            # STEP 1: clean PDB
            fixed_pdb = os.path.join(variant_out, f"{variant}_clean.pdb")
            fix_pdb(pdb, fixed_pdb, chain="A")

            # STEP 2: parse to JSONL
            with tempfile.TemporaryDirectory() as tmpdir:
                shutil.copy(fixed_pdb,
                            os.path.join(tmpdir, os.path.basename(pdb)))
                ok = run(
                    f"python {PROTEINMPNN}/helper_scripts/parse_multiple_chains.py "
                    f"--input_path {tmpdir} "
                    f"--output_path {jsonl_path}"
                )
            if not ok:
                print("FAILED (parse step)")
                failed.append(f"{species}/{lib_name}/{variant}")
                continue

            # STEP 3: generate new sequences
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
                failed.append(f"{species}/{lib_name}/{variant}")
                continue

            # STEP 4: parse scores from FASTA headers
            fa_files = glob.glob(
                os.path.join(variant_out, "seqs", "*.fa"))
            if fa_files:
                with open(fa_files[0]) as f:
                    content = f.read()

                headers = [l for l in content.splitlines()
                           if l.startswith(">")]
                seqs    = [l for l in content.splitlines()
                           if not l.startswith(">") and l.strip()]

                for i, (header, seq) in enumerate(zip(headers, seqs)):
                    s    = re.search(r'(?<![_a-z])score=([0-9.]+)',  header)
                    gs   = re.search(r'global_score=([0-9.]+)',       header)
                    rec  = re.search(r'seq_recovery=([0-9.]+)',        header)
                    samp = re.search(r'sample=([0-9]+)',               header)
                    records.append({
                        "species":      species,
                        "library":      lib_name,
                        "variant":      variant,
                        "sample":       int(samp.group(1))  if samp else i + 1,
                        "score":        float(s.group(1))   if s    else None,
                        "global_score": float(gs.group(1))  if gs   else None,
                        "seq_recovery": float(rec.group(1)) if rec  else None,
                        "sequence":     seq,
                    })

                print(f"OK — {len(headers)} sequences generated")
                total_seq += len(headers)
            else:
                print("WARNING: no output .fa file found")
                failed.append(f"{species}/{lib_name}/{variant}")

    # ── merge all FASTAs for this species ──────────────────────
    merged_fasta = os.path.join(out_base, "all_generated_sequences.fasta")
    with open(merged_fasta, "w") as out_f:
        pattern = os.path.join(out_base, "L*/*/seqs/*.fa")
        fa_list = sorted(glob.glob(pattern))

        # Fallback: root-level library (no L* folders)
        if not fa_list:
            pattern = os.path.join(out_base, "*/seqs/*.fa")
            fa_list = sorted(glob.glob(pattern))

        for fa_file in fa_list:
            parts   = Path(fa_file).parts
            lib     = next(
                (p for p in parts if p.startswith("L") and p[1:].isdigit()),
                "root"
            )
            variant = parts[-3]
            with open(fa_file) as in_f:
                for line in in_f:
                    if line.startswith(">"):
                        out_f.write(
                            line.rstrip()
                            + f" | species={species}"
                            + f" | lib={lib}"
                            + f" | variant={variant}\n"
                        )
                    else:
                        out_f.write(line)

    # ── save per-species summary CSV ───────────────────────────
    df = pd.DataFrame(records)
    summary_csv = os.path.join(out_base, "generated_sequences_summary.csv")
    if not df.empty:
        df.sort_values(
            ["library", "variant", "score"]
        ).to_csv(summary_csv, index=False)

    return records, failed, total_seq


def main():
    # ── validate ProteinMPNN path ───────────────────────────────
    assert os.path.isdir(PROTEINMPNN), \
        f"ProteinMPNN repo not found:\n  {PROTEINMPNN}"

    soluble_flag = "--use_soluble_model" if USE_SOLUBLE else ""

    print("=" * 62)
    print("  ProteinMPNN — GENERATE NEW SEQUENCES")
    print("  Human + Mouse GPX6")
    print("=" * 62)
    print(f"  ProteinMPNN : {PROTEINMPNN}")
    print(f"  Seqs/PDB    : {NUM_SEQ}")
    print(f"  Temperature : {TEMP}")
    print(f"  Model       : {'soluble' if USE_SOLUBLE else 'vanilla'}")
    print(f"  Solvated    : SKIPPED (files with '_solvated' in name)")
    print("=" * 62)

    all_records  = []
    all_failed   = []
    grand_total  = 0

    # ── loop over datasets (Mouse, Human) ──────────────────────
    for species, pdb_base, out_base in DATASETS:
        assert os.path.isdir(pdb_base), \
            f"PDB directory not found for {species}:\n  {pdb_base}"

        records, failed, total_seq = process_dataset(
            species, pdb_base, out_base, soluble_flag
        )
        all_records.extend(records)
        all_failed.extend(failed)
        grand_total += total_seq

        # Per-species quick summary
        df_s = pd.DataFrame(records)
        print(f"\n  [{species}] {total_seq} sequences generated  "
              f"({len(failed)} failed)")
        if not df_s.empty:
            print(f"  Top 5 by score ({species}):")
            print(
                df_s.dropna(subset=["score"])
                    .sort_values("score")
                    [["variant", "sample", "score",
                      "global_score", "seq_recovery"]]
                    .head(5)
                    .to_string(index=False)
            )

    # ── combined summary across both species ───────────────────
    df_all = pd.DataFrame(all_records)

    print("\n" + "=" * 62)
    print("  FINAL SUMMARY — BOTH SPECIES")
    print("=" * 62)
    print(f"  Total sequences generated : {grand_total}")

    if all_failed:
        print(f"  Failed ({len(all_failed)}):")
        for f in all_failed:
            print(f"    - {f}")

    if not df_all.empty:
        print("\n  Top 10 overall by score (best fit first):")
        print("-" * 62)
        print(
            df_all.dropna(subset=["score"])
                  .sort_values("score")
                  [["species", "library", "variant",
                    "sample", "score", "global_score", "seq_recovery"]]
                  .head(10)
                  .to_string(index=False)
        )

    print("\n  Output locations:")
    for species, _, out_base in DATASETS:
        print(f"  [{species}]")
        print(f"    FASTAs  : {out_base}/L*/VARIANT/seqs/*.fa")
        print(f"    Merged  : {out_base}/all_generated_sequences.fasta")
        print(f"    CSV     : {out_base}/generated_sequences_summary.csv")
    print("=" * 62)


if __name__ == "__main__":
    main()