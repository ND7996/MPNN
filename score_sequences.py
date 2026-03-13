"""
2_score_existing_sequences.py
================================================================
ProteinMPNN — Score each mutant's OWN existing sequence.

PURPOSE:
    Takes each mutant PDB and scores its existing sequence
    against its backbone. No new sequences are generated.
    This tells you how well each mutation fits its structure.

    Lower score = sequence fits backbone well = likely more stable
    Higher score = poor fit = mutation may be destabilising

OUTPUT:
    GPX6MPNN_scores/
    ├── L0/
    │   ├── E143S/score_only/E143S_clean_pdb.npz  ← scores here
    │   └── ...
    ├── all_scores_summary.csv      ← all variants + scores
    └── ranked_by_score.csv         ← sorted low→high (best→worst)

    Use ranked_by_score.csv later to compare with your dG* values.

USAGE:
    python 2_score_existing_sequences.py
================================================================
"""

import os
import subprocess
import glob
import shutil
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

# ================================================================
#  CONFIGURATION — edit these paths
# ================================================================
PROTEINMPNN = "/home/hp/nayanika/github/GPX6MPNN"
PDB_BASE    = "/home/hp/nayanika/github/GPX6MPNN/Human/RS_TS_human"
OUT_BASE    = "/home/hp/nayanika/github/GPX6MPNN/Human/GPX6MPNN_scores"

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
      1. Strip non-protein residues (HOH, ions)
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


def read_score_npz(variant_out):
    """
    ProteinMPNN --score_only writes scores to:
        {variant_out}/score_only/{name}.npz
    The npz contains score.npy and global_score.npy as float32 arrays.
    Returns (score, global_score) floats, or (None, None) if not found.
    """
    npz_files = glob.glob(os.path.join(variant_out, "score_only", "*.npz"))
    if not npz_files:
        return None, None
    try:
        data         = np.load(npz_files[0])
        score        = float(data["score"].mean())
        global_score = float(data["global_score"].mean())
        return score, global_score
    except Exception as e:
        print(f"    WARNING: could not read npz: {e}")
        return None, None


def main():
    # ── validate paths ──────────────────────────────────────────
    assert os.path.isdir(PROTEINMPNN), f"ProteinMPNN repo not found:\n  {PROTEINMPNN}"
    assert os.path.isdir(PDB_BASE),    f"PDB directory not found:\n  {PDB_BASE}"
    os.makedirs(OUT_BASE, exist_ok=True)

    soluble_flag = "--use_soluble_model" if USE_SOLUBLE else ""

    print("=" * 62)
    print("  ProteinMPNN — SCORE EXISTING SEQUENCES  |  HumanGPX6")
    print("=" * 62)
    print(f"  ProteinMPNN : {PROTEINMPNN}")
    print(f"  PDB input   : {PDB_BASE}")
    print(f"  Output      : {OUT_BASE}")
    print(f"  Model       : {'soluble' if USE_SOLUBLE else 'vanilla'}")
    print(f"  Skip solv.  : {SKIP_SOLVATED}")
    print("=" * 62)

    lib_dirs = sorted(glob.glob(os.path.join(PDB_BASE, "L*/")))
    results  = []
    failed   = []

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
                results.append({"library": lib_name, "variant": variant,
                                 "pdb_file": os.path.basename(pdb),
                                 "score": None, "global_score": None})
                continue

            # STEP 2: score existing sequence using the cleaned PDB
            ok = run(
                f"python {PROTEINMPNN}/protein_mpnn_run.py "
                f"--pdb_path {fixed_pdb} "
                f"--jsonl_path {jsonl_path} "
                f"--out_folder {variant_out}/ "
                f"--score_only 1 "
                f"--save_score 1 "
                f"--batch_size 1 "
                f"{soluble_flag}"
            )
            if not ok:
                print("FAILED (ProteinMPNN step)")
                failed.append(f"{lib_name}/{variant}")
                results.append({"library": lib_name, "variant": variant,
                                 "pdb_file": os.path.basename(pdb),
                                 "score": None, "global_score": None})
                continue

            # STEP 3: read score and global_score from the .npz file
            score, global_score = read_score_npz(variant_out)

            print(f"score={score:.4f}   global_score={global_score:.4f}"
                  if score is not None else "WARNING: score not found")

            results.append({
                "library":      lib_name,
                "variant":      variant,
                "pdb_file":     os.path.basename(pdb),
                "score":        score,
                "global_score": global_score
            })

    # ── build and save tables ───────────────────────────────────
    df        = pd.DataFrame(results)
    df_ranked = df.dropna(subset=["score"]).sort_values("score").reset_index(drop=True)

    summary_csv = os.path.join(OUT_BASE, "all_scores_summary.csv")
    ranked_csv  = os.path.join(OUT_BASE, "ranked_by_score.csv")

    df.to_csv(summary_csv, index=False)
    df_ranked.to_csv(ranked_csv, index=False)

    # ── final report ────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  DONE — Top 20 variants (lowest score = best backbone fit)")
    print("=" * 62)
    print(df_ranked[["library", "variant", "score", "global_score"]]
          .head(20).to_string(index=False))
    print("\n" + "-" * 62)
    print(f"  Total scored  : {len(df_ranked)}")
    if failed:
        print(f"  Failed ({len(failed)})     : {failed}")
    print(f"\n  Full summary  -> {summary_csv}")
    print(f"  Ranked table  -> {ranked_csv}  <-- use this for dG* comparison")
    print("\n  Score guide:")
    print("    score        = fit of mutation's sequence to its backbone")
    print("    global_score = same averaged over all residues")
    print("    lower = better fit")
    print("=" * 62)


if __name__ == "__main__":
    main()
