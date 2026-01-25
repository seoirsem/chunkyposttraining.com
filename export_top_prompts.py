"""Export top N violating prompts from each experiment for static hosting."""

import argparse
import json
import glob
import os
from collections import defaultdict
from pathlib import Path


# Experiments to include (v2 versions override v1 when available for same model)
EXPERIMENTS_INCLUDE = [
    'code_bleed',
    'cold_tone',
    'math_bleed',
    'rebuttal',
    'refusal',
    'story_bleed',
]

# V2 experiments that override their v1 counterparts when available
V2_OVERRIDES = {
    'code_bleed_v2': 'code_bleed',
    'rebuttal_v2': 'rebuttal',
    'refusal_v2': 'refusal',
}

# Model mapping (from top_n_scores.sh)
MODEL_MAP = {
    'vllm': 'Tulu',
    'tulu_sft': 'Tulu',
    'haiku_45': 'Haiku',
    'haiku_45_high': 'Haiku',
    'sonnet_45': 'Sonnet',
    'sonnet_4': 'Sonnet',
    'sonnet_4_high': 'Sonnet',
    'opus_45': 'Opus',
    'gpt_5_1': 'GPT-5.1',
    'gpt_5_mini': 'GPT-5-mini',
    'gemini_3': 'Gemini-3',
    'gemini_25': 'Gemini-2.5',
    'grok_41': 'Grok-4.1',
}

MODEL_ORDER = ['Tulu', 'Haiku', 'Sonnet', 'Opus', 'GPT-5.1', 'GPT-5-mini', 'Gemini-3', 'Gemini-2.5', 'Grok-4.1']


def normalize_experiment_name(exp_part: str) -> str:
    """Normalize experiment name by removing version/sysprompt suffixes.

    Returns (normalized_name, is_v2) tuple.
    V2 experiments are tracked separately so they can override v1.
    """
    exp_normalized = exp_part

    # Remove sysprompt suffixes first
    for suffix in ['_haiku_sysprompt', '_opus_sysprompt', '_sonnet_sysprompt', '_tulu_vllm']:
        exp_normalized = exp_normalized.replace(suffix, '')

    # Handle combined
    if exp_normalized == 'math_bleed_combined':
        exp_normalized = 'math_bleed'

    # Check if this is a v2 experiment that should override v1
    if exp_normalized in V2_OVERRIDES:
        return exp_normalized, True

    # Remove version suffixes for other experiments
    for suffix in ['_v2', '_v3', '_v1']:
        if exp_normalized.endswith(suffix):
            exp_normalized = exp_normalized[:-len(suffix)]

    return exp_normalized, False


def parse_filename(fname: str) -> tuple[str, str, bool] | None:
    """Parse filename to extract experiment and model.

    Returns (experiment_normalized, model_name, is_v2) or None if invalid.
    """
    if '_target_' not in fname:
        return None

    parts = fname.split('_target_')
    exp_part = parts[0]
    rest = parts[1]

    # Get model from rest (before _cheap_ or _query_)
    if '_cheap_' in rest:
        model_raw = rest.split('_cheap_')[0]
    elif '_query_' in rest:
        model_raw = rest.split('_query_')[0]
    else:
        model_raw = rest

    # Map model name
    model_name = MODEL_MAP.get(model_raw)
    if not model_name:
        return None

    # Normalize experiment
    exp_normalized, is_v2 = normalize_experiment_name(exp_part)

    # For v2 experiments, check if base experiment is in include list
    if is_v2:
        base_exp = V2_OVERRIDES.get(exp_normalized)
        if base_exp not in EXPERIMENTS_INCLUDE:
            return None
    else:
        # Skip experiments not in include list
        if exp_normalized not in EXPERIMENTS_INCLUDE:
            return None

    return exp_normalized, model_name, is_v2


def get_experiment_folder(jsonl_path: str) -> str:
    """Extract the experiment folder name from the path."""
    parts = jsonl_path.split('/')
    if 'jsons' in parts:
        jsons_idx = parts.index('jsons')
        if jsons_idx > 0:
            return parts[jsons_idx - 1]
    return os.path.dirname(jsonl_path)


def collect_all_entries(source_dir: str) -> dict:
    """Collect all entries from JSONL files, grouped by (experiment, model).

    V2 experiments override v1 for the same model when available.

    Returns: {(exp, model): [(score, entry_dict), ...]}
    """
    # Collect v1 and v2 data separately
    v1_data = defaultdict(list)
    v2_data = defaultdict(list)

    for jsonl_path in glob.glob(f"{source_dir}/**/jsons/*_simple.jsonl", recursive=True):
        fname = os.path.basename(jsonl_path).replace('_simple.jsonl', '')

        parsed = parse_filename(fname)
        if parsed is None:
            continue

        exp_normalized, model_name, is_v2 = parsed
        exp_folder = get_experiment_folder(jsonl_path)

        # Determine the base experiment name for grouping
        if is_v2:
            base_exp = V2_OVERRIDES.get(exp_normalized, exp_normalized)
        else:
            base_exp = exp_normalized

        try:
            with open(jsonl_path, 'r') as f:
                lines = f.readlines()

            if not lines:
                continue

            # Group by run_id and keep last iteration per run
            runs = {}
            for line in lines:
                d = json.loads(line)
                run_id = d.get('run_id')
                runs[run_id] = d

            # Extract entries from replay_buffer_entries of last iteration per run
            for run_id, last_iter in runs.items():
                entries = last_iter.get('replay_buffer_entries', [])
                for entry in entries:
                    # Use reward_score if available, fall back to opus_score
                    score = entry.get('reward_score') or entry.get('opus_score') or entry.get('filter_score') or 0
                    if score is None:
                        score = 0

                    entry_data = {
                        'query': entry.get('query', ''),
                        'response': entry.get('response', ''),
                        'score': score,
                        'model': model_name,
                        'experiment': base_exp,  # Use base experiment name
                        'experiment_folder': exp_folder,
                        'target': model_name,
                        'opus_score': entry.get('opus_score'),
                        'opus_reason': entry.get('opus_reason'),
                        'category': entry.get('category'),
                        'is_v2': is_v2,
                    }

                    if is_v2:
                        v2_data[(base_exp, model_name)].append((score, entry_data))
                    else:
                        v1_data[(base_exp, model_name)].append((score, entry_data))

        except Exception as e:
            print(f"Error processing {jsonl_path}: {e}")

    # Merge: prefer v2 over v1 for same (exp, model)
    all_data = {}
    all_keys = set(v1_data.keys()) | set(v2_data.keys())

    for key in all_keys:
        if key in v2_data and v2_data[key]:
            # Use v2 data if available
            all_data[key] = v2_data[key]
        elif key in v1_data:
            # Fall back to v1
            all_data[key] = v1_data[key]

    return all_data


def export_top_n(
    source_dir: str,
    output_path: str,
    top_n: int = 20,
    format: str = 'json',
):
    """Export top N violating prompts from each experiment/model combination."""

    print(f"Scanning {source_dir} for *_simple.jsonl files...")
    all_data = collect_all_entries(source_dir)

    print(f"Found data for {len(all_data)} experiment/model combinations")

    # Build export data
    export_data = {
        'metadata': {
            'top_n': top_n,
            'experiments': EXPERIMENTS_INCLUDE,
            'models': MODEL_ORDER,
        },
        'by_experiment': {},
        'all_entries': [],
    }

    # Process each experiment/model combination
    for (exp, model), entries in all_data.items():
        # Sort by score descending and take top N
        sorted_entries = sorted(entries, key=lambda x: x[0], reverse=True)[:top_n]

        if exp not in export_data['by_experiment']:
            export_data['by_experiment'][exp] = {}

        export_data['by_experiment'][exp][model] = [
            entry_data for score, entry_data in sorted_entries
        ]

        # Also add to flat list
        for score, entry_data in sorted_entries:
            export_data['all_entries'].append(entry_data)

    # Sort all_entries by score descending
    export_data['all_entries'].sort(key=lambda x: x.get('score', 0), reverse=True)

    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'json':
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"Exported to {output_path}")
    elif format == 'jsonl':
        with open(output_path, 'w') as f:
            for entry in export_data['all_entries']:
                f.write(json.dumps(entry) + '\n')
        print(f"Exported {len(export_data['all_entries'])} entries to {output_path}")

    # Print summary
    print("\nSummary by experiment:")
    for exp in EXPERIMENTS_INCLUDE:
        if exp in export_data['by_experiment']:
            models = export_data['by_experiment'][exp]
            total = sum(len(entries) for entries in models.values())
            model_counts = ', '.join(f"{m}: {len(entries)}" for m, entries in models.items())
            print(f"  {exp}: {total} entries ({model_counts})")
        else:
            print(f"  {exp}: 0 entries")

    print(f"\nTotal entries: {len(export_data['all_entries'])}")

    return export_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export top N violating prompts from each experiment"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="/workspace-vast/seoirsem/chunky/251209_model_chunky_pipeline",
        help="Source directory containing experiment results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="top_prompts_export.json",
        help="Output file path",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top entries to export per experiment/model (default: 20)",
    )
    parser.add_argument(
        "--format",
        choices=['json', 'jsonl'],
        default='json',
        help="Output format (default: json)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_top_n(
        source_dir=args.source_dir,
        output_path=args.output,
        top_n=args.top_n,
        format=args.format,
    )
