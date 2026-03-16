# ProteinMPNN on homologous protein structures from the PDB
#
# Workflow:
#   1. Load a seed sequence from a FASTA file.
#   2. Query the RCSB PDB sequence-search API for homologous polymer entities.
#   3. Keep a small non-redundant set of PDB entries.
#   4. Download the corresponding PDB files.
#   5. Extract one sufficiently long protein chain from each structure.
#   6. Run ProteinMPNN on each extracted backbone.
#   7. Re-evaluate every generated sequence with ProteinMPNN --score_only over
#      100 stochastic passes to reduce score noise.
#   8. Collect the re-scored sequences into FASTA and CSV summaries.

# =============================================================================
# Setup
# =============================================================================

from pathlib import Path
import os
from collections import Counter
import csv
import json
import math
import re
import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import requests
from Bio import SeqIO
from Bio.PDB import PDBParser, PDBIO, Select

SEED_FASTA = Path(r'D:\PhD_Thesis\GPX6\analysis\alignment\all_sequences_for_selection.fasta')
SEED_RECORD_INDEX = 0
IDENTITY_CUTOFF = 0.95
EVALUE_CUTOFF = 1e-5
MAX_PDB_STRUCTURES = 10
CANDIDATE_POOL_SIZE = 40
MIN_CHAIN_LENGTH = 120
NUM_SEQ_PER_TARGET = 2
SAMPLING_TEMP = '0.1'
RANDOM_SEED = 37
REEVALUATION_RUNS = 100
REEVALUATION_BATCH_SIZE = 10

ROOT = Path('.')
PROTEINMPNN_DIR = Path(os.environ.get('PROTEINMPNN_HOME')).expanduser() if os.environ.get('PROTEINMPNN_HOME') else ROOT / 'ProteinMPNN'
WORK_DIR = ROOT / 'results' / 'proteinmpnn'
PDB_DIR = WORK_DIR / 'downloaded_pdbs'
CHAIN_DIR = WORK_DIR / 'single_chain_pdbs'
OUTPUT_DIR = WORK_DIR / 'proteinmpnn_outputs'
SCORE_ONLY_DIR = WORK_DIR / 'proteinmpnn_score_only'
SUMMARY_CSV = WORK_DIR / 'proteinmpnn_design_summary.csv'
DESIGN_FASTA = WORK_DIR / 'proteinmpnn_generated_sequences_design.fa'
REEVALUATED_FASTA = WORK_DIR / 'proteinmpnn_generated_sequences_reevaluated.fa'
FIGURES_DIR = Path('results/figures')
SUMMARY_PNG = FIGURES_DIR / 'sequenceTools_proteinmpnn_summary.png'
SUMMARY_PDF = FIGURES_DIR / 'sequenceTools_proteinmpnn_summary.pdf'

for directory in [WORK_DIR, PDB_DIR, CHAIN_DIR, OUTPUT_DIR, SCORE_ONLY_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# =============================================================================
# Helper Functions
# =============================================================================

def ensure_proteinmpnn_install(repo_dir: Path):
    repo_dir = repo_dir.expanduser().resolve()
    run_script = repo_dir / 'protein_mpnn_run.py'
    if run_script.exists():
        return repo_dir
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ['git', 'clone', '--depth', '1', 'https://github.com/dauparas/ProteinMPNN.git', str(repo_dir)],
        check=True,
    )
    if not run_script.exists():
        raise FileNotFoundError(f'ProteinMPNN clone finished, but {run_script} was not found.')
    return repo_dir


def load_seed_sequence(fasta_path: Path, record_index: int = 0):
    records = list(SeqIO.parse(fasta_path, 'fasta'))
    if not records:
        raise ValueError(f'No FASTA records found in {fasta_path}')
    if record_index >= len(records):
        raise IndexError(f'record_index={record_index} is out of range for {len(records)} FASTA records')
    return records[record_index]


def search_pdb_by_sequence(sequence: str, identity_cutoff: float, evalue_cutoff: float, rows: int = 50):
    query = {
        'query': {
            'type': 'terminal',
            'service': 'sequence',
            'parameters': {
                'evalue_cutoff': evalue_cutoff,
                'identity_cutoff': identity_cutoff,
                'target': 'pdb_protein_sequence',
                'value': sequence,
            },
        },
        'return_type': 'polymer_entity',
        'request_options': {'paginate': {'start': 0, 'rows': rows}},
    }
    response = requests.post('https://search.rcsb.org/rcsbsearch/v2/query', json=query, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data.get('result_set', [])


def get_entry_metadata(entry_id: str):
    response = requests.get(f'https://data.rcsb.org/rest/v1/core/entry/{entry_id}', timeout=60)
    response.raise_for_status()
    data = response.json()
    title = data.get('struct', {}).get('title', entry_id)
    exptl = data.get('exptl', [])
    method = exptl[0].get('method') if exptl else 'NA'
    resolution = None
    for key in ['refine', 'em_3d_reconstruction']:
        if key in data:
            candidate = data[key][0] if isinstance(data[key], list) else data[key]
            if isinstance(candidate, dict):
                resolution = candidate.get('ls_d_res_high') or candidate.get('resolution')
            if resolution is not None:
                break
    return {
        'entry_id': entry_id,
        'title': title,
        'method': method,
        'resolution': resolution,
    }


def choose_entries(result_set, max_structures: int):
    chosen = []
    seen = set()
    for item in result_set:
        polymer_entity_id = item['identifier']
        entry_id = polymer_entity_id.split('_')[0]
        if entry_id in seen:
            continue
        seen.add(entry_id)
        meta = get_entry_metadata(entry_id)
        meta['polymer_entity_id'] = polymer_entity_id
        meta['search_score'] = item.get('score')
        chosen.append(meta)
        if len(chosen) >= max_structures:
            break
    return chosen


def download_pdb(entry_id: str, out_dir: Path):
    out_path = out_dir / f'{entry_id}.pdb'
    if out_path.exists():
        return out_path
    try:
        response = requests.get(f'https://files.rcsb.org/download/{entry_id}.pdb', timeout=60)
        response.raise_for_status()
    except requests.HTTPError:
        return None
    out_path.write_text(response.text)
    return out_path


def amino_acid_residues(chain):
    return [residue for residue in chain if residue.id[0] == ' ']


def pick_longest_protein_chain(pdb_path: Path, min_chain_length: int):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    best = None
    for model in structure:
        for chain in model:
            residues = amino_acid_residues(chain)
            if len(residues) < min_chain_length:
                continue
            if best is None or len(residues) > best['length']:
                best = {'chain_id': chain.id, 'length': len(residues), 'structure': structure}
    return best


class SingleChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        return 1 if residue.id[0] == ' ' else 0


def extract_chain_pdb(structure, chain_id: str, out_path: Path):
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_path), SingleChainSelect(chain_id))
    return out_path


def parse_mpnn_fasta(fasta_path: Path):
    entries = []
    lines = [line.strip() for line in fasta_path.read_text().splitlines() if line.strip()]
    for i in range(0, len(lines), 2):
        header = lines[i]
        sequence = lines[i + 1]
        entries.append((header, sequence))
    return entries


def extract_float(header: str, key: str):
    match = re.search(rf'{re.escape(key)}=([-+]?[0-9]*\.?[0-9]+)', header)
    return float(match.group(1)) if match else None


def run_proteinmpnn(repo_dir: Path, pdb_path: Path, chain_id: str, out_root: Path, num_seq_per_target: int, sampling_temp: str, seed: int):
    target_out = out_root / pdb_path.stem
    if target_out.exists():
        shutil.rmtree(target_out)
    target_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        'python',
        str(repo_dir / 'protein_mpnn_run.py'),
        '--pdb_path', str(pdb_path),
        '--pdb_path_chains', chain_id,
        '--out_folder', str(target_out),
        '--num_seq_per_target', str(num_seq_per_target),
        '--sampling_temp', str(sampling_temp),
        '--seed', str(seed),
        '--batch_size', '1',
    ]
    subprocess.run(cmd, check=True)
    fasta_path = target_out / 'seqs' / f'{pdb_path.stem}.fa'
    return fasta_path


def run_proteinmpnn_score_only(repo_dir: Path, pdb_path: Path, chain_id: str, input_fasta: Path, out_root: Path, score_runs: int, batch_size: int, seed: int):
    if score_runs % batch_size != 0:
        raise ValueError('REEVALUATION_RUNS must be divisible by REEVALUATION_BATCH_SIZE')
    target_out = out_root / pdb_path.stem
    if target_out.exists():
        shutil.rmtree(target_out)
    target_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        'python',
        str(repo_dir / 'protein_mpnn_run.py'),
        '--pdb_path', str(pdb_path),
        '--pdb_path_chains', chain_id,
        '--out_folder', str(target_out),
        '--score_only', '1',
        '--path_to_fasta', str(input_fasta),
        '--num_seq_per_target', str(score_runs),
        '--batch_size', str(batch_size),
        '--seed', str(seed),
        '--suppress_print', '1',
    ]
    subprocess.run(cmd, check=True)
    return target_out / 'score_only'


def summarize_score_only(score_dir: Path, target_name: str, fasta_entries):
    summaries = []
    for index, (header, sequence) in enumerate(fasta_entries, start=1):
        score_file = score_dir / f'{target_name}_fasta_{index}.npz'
        if not score_file.exists():
            raise FileNotFoundError(f'Missing score_only output: {score_file}')
        with np.load(score_file, allow_pickle=True) as data:
            score_values = np.asarray(data['score'], dtype=float)
            global_values = np.asarray(data['global_score'], dtype=float)
        summaries.append({
            'header': header,
            'sequence': sequence,
            'score_mean': float(score_values.mean()),
            'score_std': float(score_values.std()),
            'global_score_mean': float(global_values.mean()),
            'global_score_std': float(global_values.std()),
            'score_runs': int(score_values.shape[0]),
        })
    return summaries


# =============================================================================
# Main Workflow
# =============================================================================

ensure_proteinmpnn_install(PROTEINMPNN_DIR)
seed_record = load_seed_sequence(SEED_FASTA, SEED_RECORD_INDEX)
seed_sequence = str(seed_record.seq)

hits = search_pdb_by_sequence(seed_sequence, IDENTITY_CUTOFF, EVALUE_CUTOFF, rows=80)
selected_entries = choose_entries(hits, CANDIDATE_POOL_SIZE)

design_rows = []
design_fasta_records = []
reevaluated_fasta_records = []
accepted_entries = []

for entry in selected_entries:
    if len(accepted_entries) >= MAX_PDB_STRUCTURES:
        break
    pdb_path = download_pdb(entry['entry_id'], PDB_DIR)
    if pdb_path is None:
        print(f"Skipping {entry['entry_id']}: PDB file could not be downloaded")
        continue
    chain_info = pick_longest_protein_chain(pdb_path, MIN_CHAIN_LENGTH)
    if chain_info is None:
        print(f"Skipping {entry['entry_id']}: no protein chain >= {MIN_CHAIN_LENGTH} residues")
        continue
    chain_id = chain_info['chain_id']
    single_chain_path = CHAIN_DIR / f"{entry['entry_id']}_{chain_id}.pdb"
    extract_chain_pdb(chain_info['structure'], chain_id, single_chain_path)
    fasta_path = run_proteinmpnn(
        PROTEINMPNN_DIR,
        single_chain_path,
        chain_id,
        OUTPUT_DIR,
        NUM_SEQ_PER_TARGET,
        SAMPLING_TEMP,
        RANDOM_SEED,
    )
    fasta_entries = parse_mpnn_fasta(fasta_path)
    native_header, native_sequence = fasta_entries[0]
    sample_pairs = list(enumerate(fasta_entries[1:], start=1))
    if 'X' in native_sequence or any('X' in sequence for _, (_, sequence) in sample_pairs):
        print(f"Skipping {entry['entry_id']}_{chain_id}: unresolved positions propagate as X in ProteinMPNN output")
        continue

    score_dir = run_proteinmpnn_score_only(
        PROTEINMPNN_DIR,
        single_chain_path,
        chain_id,
        fasta_path,
        SCORE_ONLY_DIR,
        REEVALUATION_RUNS,
        REEVALUATION_BATCH_SIZE,
        RANDOM_SEED,
    )
    rescored_entries = summarize_score_only(score_dir, single_chain_path.stem, fasta_entries)

    accepted_entries.append(entry['entry_id'])
    design_fasta_records.append((f">{entry['entry_id']}_{chain_id}_native {native_header.lstrip('>')}", native_sequence))
    native_summary = rescored_entries[0]
    native_fasta_header = (
        f">{entry['entry_id']}_{chain_id}_native "
        f"score={native_summary['score_mean']:.4f} score_std={native_summary['score_std']:.4f} "
        f"global_score={native_summary['global_score_mean']:.4f} global_score_std={native_summary['global_score_std']:.4f} "
        f"seq_recovery=1.0000 score_runs={native_summary['score_runs']} score_source=reevaluated"
    )
    reevaluated_fasta_records.append((native_fasta_header, native_sequence))

    for sample_index, ((header, sequence), summary) in enumerate(zip(fasta_entries[1:], rescored_entries[1:]), start=1):
        design_score_original = extract_float(header, 'score')
        global_score_original = extract_float(header, 'global_score')
        seq_recovery = extract_float(header, 'seq_recovery')
        design_header = f">{entry['entry_id']}_{chain_id}_sample_{sample_index} {header.lstrip('>')}"
        design_fasta_records.append((design_header, sequence))
        fasta_header = (
            f">{entry['entry_id']}_{chain_id}_sample_{sample_index} "
            f"score={summary['score_mean']:.4f} score_std={summary['score_std']:.4f} "
            f"global_score={summary['global_score_mean']:.4f} global_score_std={summary['global_score_std']:.4f} "
            f"seq_recovery={seq_recovery:.4f} original_score={design_score_original:.4f} "
            f"original_global_score={global_score_original:.4f} score_runs={summary['score_runs']} score_source=reevaluated"
        )
        reevaluated_fasta_records.append((fasta_header, sequence))
        design_rows.append({
            'entry_id': entry['entry_id'],
            'title': entry['title'],
            'method': entry['method'],
            'resolution': entry['resolution'],
            'chain_id': chain_id,
            'chain_length': chain_info['length'],
            'sample_index': sample_index,
            'design_score': summary['score_mean'],
            'design_score_std': summary['score_std'],
            'global_score': summary['global_score_mean'],
            'global_score_std': summary['global_score_std'],
            'original_design_score': design_score_original,
            'original_global_score': global_score_original,
            'seq_recovery': seq_recovery,
            'score_runs': summary['score_runs'],
        })

if not design_rows:
    raise RuntimeError('No ProteinMPNN designs were collected from the candidate PDB structures.')

with SUMMARY_CSV.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(design_rows[0].keys()))
    writer.writeheader()
    writer.writerows(design_rows)

with DESIGN_FASTA.open('w') as handle:
    for header, sequence in design_fasta_records:
        handle.write(header + '\n')
        handle.write(sequence + '\n')

with REEVALUATED_FASTA.open('w') as handle:
    for header, sequence in reevaluated_fasta_records:
        handle.write(header + '\n')
        handle.write(sequence + '\n')

print(f'Seed record: {seed_record.id}')
print(f'Accepted structures: {accepted_entries}')
print(f'Designed sequences: {len(design_rows)}')
print(f'Re-evaluation runs per sequence: {REEVALUATION_RUNS}')
print(f'Summary CSV: {SUMMARY_CSV}')
print(f'Design FASTA: {DESIGN_FASTA}')
print(f'Re-evaluated FASTA: {REEVALUATED_FASTA}')


# =============================================================================
# Summary Plot
# =============================================================================

grouped = {}
for row in design_rows:
    grouped.setdefault(row['entry_id'], []).append(row)

entry_ids = list(grouped)
mean_scores = [sum(item['design_score'] for item in grouped[e] if item['design_score'] is not None) / len(grouped[e]) for e in entry_ids]
mean_score_stds = [sum(item['design_score_std'] for item in grouped[e] if item['design_score_std'] is not None) / len(grouped[e]) for e in entry_ids]
mean_recovery = [sum(item['seq_recovery'] for item in grouped[e] if item['seq_recovery'] is not None) / len(grouped[e]) for e in entry_ids]
labels = [f"{entry_id}\n{grouped[entry_id][0]['chain_id']}" for entry_id in entry_ids]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=200)
axes[0].bar(labels, mean_scores, yerr=mean_score_stds, capsize=4, color='#2a6f97')
axes[0].set_title(f'Mean re-evaluated ProteinMPNN score ({REEVALUATION_RUNS} runs / sequence)')
axes[0].set_ylabel('score')
axes[0].tick_params(axis='x', rotation=0)

axes[1].bar(labels, mean_recovery, color='#bc4749')
axes[1].set_title('Mean sequence recovery')
axes[1].set_ylabel('seq_recovery')
axes[1].set_ylim(0, 1)
axes[1].tick_params(axis='x', rotation=0)

fig.suptitle('ProteinMPNN designs on homologous PDB backbones', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(SUMMARY_PNG, dpi=300, bbox_inches='tight')
fig.savefig(SUMMARY_PDF, bbox_inches='tight')
plt.show()

print(f'Saved figure: {SUMMARY_PNG}')
print(f'Saved figure: {SUMMARY_PDF}')