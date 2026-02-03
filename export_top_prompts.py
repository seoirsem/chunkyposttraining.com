"""Export marked prompts (good/great) from experiment viewer for static hosting."""

import argparse
import glob
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

# Default file locations
DEFAULT_MAPPING_FILE = "/workspace-vast/seoirsem/chunky/251209_model_chunky_pipeline/latest_experiments.json"
DEFAULT_MARKINGS_FILE = "/workspace-vast/seoirsem/chunky/251209_model_chunky_pipeline/sample_markings.json"
DEFAULT_TRANSLATION_CACHE_FILE = "/workspace-vast/seoirsem/chunky/251209_model_chunky_pipeline/translation_cache.json"

# V2 experiments that override their v1 counterparts when available
V2_OVERRIDES = {
    'code_bleed_v2': 'code_bleed',
    'rebuttal_v2': 'rebuttal',
    'refusal_v2': 'refusal',
}

# Map from filename pattern to config key (for sample_id matching)
FILENAME_TO_CONFIG_KEY = {
    'tulu': 'tulu',
    'vllm': 'tulu',
    'tulu_sft': 'tulu',
    'haiku': 'haiku',
    'haiku_45': 'haiku',
    'haiku_45_high': 'haiku',
    'sonnet': 'sonnet',
    'sonnet_45': 'sonnet',
    'sonnet_4': 'sonnet',
    'sonnet_4_high': 'sonnet',
    'opus': 'opus',
    'opus_45': 'opus',
    'gpt51': 'gpt51',
    'gpt_5_1': 'gpt51',
    'gpt_5_mini': 'gpt_5_mini',
    'gemini3': 'gemini3',
    'gemini_3': 'gemini3',
    'gemini_25': 'gemini_25',
    'grok41': 'grok41',
    'grok_41': 'grok41',
}

# Model config key to display name mapping
MODEL_DISPLAY = {
    'tulu': 'Tulu',
    'haiku': 'Haiku',
    'sonnet': 'Sonnet',
    'opus': 'Opus',
    'gpt51': 'GPT-5.1',
    'gpt_5_mini': 'GPT-5-mini',
    'gemini3': 'Gemini-3',
    'gemini_25': 'Gemini-2.5',
    'grok41': 'Grok-4.1',
}

MODEL_ORDER = ['Tulu', 'Haiku', 'Sonnet', 'Opus', 'GPT-5.1', 'GPT-5-mini', 'Gemini-3', 'Gemini-2.5', 'Grok-4.1']


def load_markings(markings_file: str) -> dict[str, str]:
    """Load sample markings from experiment viewer cache."""
    if os.path.exists(markings_file):
        with open(markings_file) as f:
            return json.load(f)
    print(f"Warning: Markings file {markings_file} not found")
    return {}


def load_translation_cache(translation_file: str) -> dict[str, str]:
    """Load translation cache (md5 hash -> translation)."""
    if os.path.exists(translation_file):
        with open(translation_file) as f:
            return json.load(f)
    print(f"Warning: Translation cache {translation_file} not found")
    return {}


def get_translation(text: str, translation_cache: dict) -> str | None:
    """Get translation for text if available in cache."""
    if not text:
        return None
    text_hash = hashlib.md5(text.encode()).hexdigest()
    return translation_cache.get(text_hash)


def get_reproducibility_data(base_path: str, experiment_path: str, threshold: float = 70.0) -> dict:
    """Load reproducibility data if available.

    Args:
        threshold: Threshold on 0-100 scale (will be converted to 0-200 for raw scores)
    """
    if experiment_path.endswith(".jsonl"):
        return {}

    # Handle both relative and absolute paths
    exp_path = Path(experiment_path)
    if exp_path.is_absolute():
        exp_dir = exp_path
    else:
        exp_dir = Path(base_path) / experiment_path
    repro_data = {}

    # Find all reproducibility folders and load from all of them
    repro_folders = sorted(exp_dir.glob("reproducibility_iter*_n*"))
    if not repro_folders:
        return {}

    # Convert threshold from 0-100 scale to 0-200 scale (raw opus scores)
    raw_threshold = threshold * 2

    # Load from ALL repro folders to maximize coverage
    for folder in repro_folders:
        json_files = list(folder.glob("*.json"))
        if not json_files:
            continue

        try:
            with open(json_files[0]) as f:
                data = json.load(f)

            is_comparison = data.get("metadata", {}).get("comparison_mode", False)

            for result in data.get("results", []):
                elo_id = result.get("_elo_id")
                if elo_id is None:
                    continue

                # Skip if we already have data for this elo_id
                if elo_id in repro_data:
                    continue

                # Handle comparison mode vs standard mode
                if is_comparison:
                    source = result.get("with_sysprompt", result)
                else:
                    source = result

                # Get all regeneration scores and compute fraction above threshold
                all_scores = source.get("all_scores", [])
                # Skip first score (it's the rescore of original), use rest for regen stats
                regen_scores = all_scores[1:] if len(all_scores) > 1 else []

                # Scores in repro data are on 0-200 scale (same as raw opus_score)
                fraction_above_threshold = 0.0
                if regen_scores:
                    above = sum(1 for s in regen_scores if s >= raw_threshold)
                    fraction_above_threshold = above / len(regen_scores)

                regen_stats = source.get("regeneration_scores", {})

                repro_data[elo_id] = {
                    "regen_mean": regen_stats.get("mean"),
                    "regen_std": regen_stats.get("std"),
                    "fraction_above_threshold": fraction_above_threshold,
                    "num_regenerations": len(regen_scores),
                }
        except Exception as e:
            print(f"Error loading {json_files[0]}: {e}")

    return repro_data


def load_experiment_mapping(mapping_file: str, include_story: bool = False) -> tuple[list[str], dict[str, dict]]:
    """Load experiments and paths from mapping file.

    Returns (experiments_list, full_mapping)
    """
    if os.path.exists(mapping_file):
        with open(mapping_file) as f:
            mapping = json.load(f)
        experiments = [k for k in mapping.keys() if k != 'base_path']
        if not include_story and 'story_bleed' in experiments:
            experiments.remove('story_bleed')
        return experiments, mapping
    else:
        # Fallback to hardcoded list if file doesn't exist
        print(f"Warning: Mapping file {mapping_file} not found, using defaults")
        default_exps = ['code_bleed', 'cold_tone', 'math_bleed', 'rebuttal', 'refusal']
        if include_story:
            default_exps.append('story_bleed')
        return default_exps, {}


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


def parse_filename(fname: str, experiments_include: list[str]) -> tuple[str, str, str, bool] | None:
    """Parse filename to extract experiment and model.

    Returns (experiment_normalized, model_config_key, model_display_name, is_v2) or None if invalid.
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

    # Map filename pattern to config key
    model_config_key = FILENAME_TO_CONFIG_KEY.get(model_raw)
    if not model_config_key:
        return None

    # Map config key to display name
    model_display = MODEL_DISPLAY.get(model_config_key)
    if not model_display:
        return None

    # Normalize experiment
    exp_normalized, is_v2 = normalize_experiment_name(exp_part)

    # For v2 experiments, check if base experiment is in include list
    if is_v2:
        base_exp = V2_OVERRIDES.get(exp_normalized)
        if base_exp not in experiments_include:
            return None
    else:
        # Skip experiments not in include list
        if exp_normalized not in experiments_include:
            return None

    return exp_normalized, model_config_key, model_display, is_v2


def get_experiment_folder(jsonl_path: str) -> str:
    """Extract the experiment folder name from the path."""
    parts = jsonl_path.split('/')
    if 'jsons' in parts:
        jsons_idx = parts.index('jsons')
        if jsons_idx > 0:
            return parts[jsons_idx - 1]
    return os.path.dirname(jsonl_path)


def collect_all_entries(source_dir: str, experiments_include: list[str], mapping: dict) -> dict:
    """Collect all entries from JSONL files, grouped by (experiment, model).

    V2 experiments override v1 for the same model when available.

    Returns: {(exp, model): [(sample_id, entry_dict, exp_path), ...]}
    """
    base_path = mapping.get('base_path', source_dir)

    # Collect v1 and v2 data separately
    v1_data = defaultdict(list)
    v2_data = defaultdict(list)

    for jsonl_path in glob.glob(f"{source_dir}/**/jsons/*_simple.jsonl", recursive=True):
        fname = os.path.basename(jsonl_path).replace('_simple.jsonl', '')

        parsed = parse_filename(fname, experiments_include)
        if parsed is None:
            continue

        exp_normalized, model_config_key, model_display, is_v2 = parsed
        exp_folder = get_experiment_folder(jsonl_path)

        # Determine the base experiment name for grouping
        if is_v2:
            base_exp = V2_OVERRIDES.get(exp_normalized, exp_normalized)
        else:
            base_exp = exp_normalized

        # Get the experiment path for reproducibility data
        try:
            exp_path = str(Path(jsonl_path).parent.parent.relative_to(base_path))
        except ValueError:
            # Path is not under base_path, use the absolute parent directory
            exp_path = str(Path(jsonl_path).parent.parent)

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
                for idx, entry in enumerate(entries):
                    # Use reward_score if available, fall back to opus_score
                    opus_score_raw = entry.get('opus_score') or 0
                    score = opus_score_raw / 2 if opus_score_raw else 0  # Convert to 0-100 scale

                    # Build sample_id matching experiment_viewer format
                    # Use config key (tulu, haiku, etc.) for sample_id
                    sample_id = f"{base_exp}_{model_config_key}_{run_id}_{idx}"

                    entry_data = {
                        'query': entry.get('query', ''),
                        'response': entry.get('response', ''),
                        'score': score,
                        'model': model_display,
                        'experiment': base_exp,
                        'experiment_folder': exp_folder,
                        'opus_score': entry.get('opus_score'),
                        'opus_reason': entry.get('opus_reason'),
                        'category': entry.get('category'),
                        '_elo_id': entry.get('_elo_id'),
                        'is_v2': is_v2,
                    }

                    if is_v2:
                        v2_data[(base_exp, model_display)].append((sample_id, entry_data, exp_path))
                    else:
                        v1_data[(base_exp, model_display)].append((sample_id, entry_data, exp_path))

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


def export_marked_entries(
    source_dir: str,
    output_path: str,
    format: str = 'json',
    mapping_file: str = DEFAULT_MAPPING_FILE,
    markings_file: str = DEFAULT_MARKINGS_FILE,
    translation_file: str = DEFAULT_TRANSLATION_CACHE_FILE,
    include_story: bool = False,
    repro_threshold: float = 70.0,
):
    """Export entries marked as 'good' or 'great' from experiment viewer."""

    # Load experiments from mapping file
    experiments_include, mapping = load_experiment_mapping(mapping_file, include_story)
    print(f"Loaded {len(experiments_include)} experiments from {mapping_file}")
    print(f"Experiments: {experiments_include}")

    # Load markings
    markings = load_markings(markings_file)
    good_great_ids = {k for k, v in markings.items() if v in ('good', 'great')}
    print(f"Loaded {len(markings)} markings, {len(good_great_ids)} marked as good/great")

    # Load translation cache
    translation_cache = load_translation_cache(translation_file)
    print(f"Loaded {len(translation_cache)} cached translations")

    base_path = mapping.get('base_path', source_dir)

    print(f"Scanning {source_dir} for *_simple.jsonl files...")
    all_data = collect_all_entries(source_dir, experiments_include, mapping)

    print(f"Found data for {len(all_data)} experiment/model combinations")

    # Cache for reproducibility data per experiment path
    repro_cache = {}

    # Build export data
    export_data = {
        'metadata': {
            'experiments': experiments_include,
            'models': MODEL_ORDER,
            'filter': 'good_and_great',
        },
        'by_experiment': {},
        'all_entries': [],
    }

    # Process each experiment/model combination
    for (exp, model), entries in all_data.items():
        matched_entries = []

        for sample_id, entry_data, exp_path in entries:
            # Only include entries marked as good or great
            if sample_id not in good_great_ids:
                continue

            # Only include entries with score >= 70
            if entry_data.get('score', 0) < 70:
                continue

            # Load reproducibility data for this experiment (cached)
            if exp_path not in repro_cache:
                repro_cache[exp_path] = get_reproducibility_data(base_path, exp_path, repro_threshold)
            repro_data = repro_cache[exp_path]

            # Add reproducibility info if available
            elo_id = entry_data.get('_elo_id')
            if elo_id and elo_id in repro_data:
                repro = repro_data[elo_id]
                entry_data['repro_fraction_above_threshold'] = repro.get('fraction_above_threshold')
                entry_data['repro_regen_mean'] = repro.get('regen_mean')
                entry_data['repro_num_regenerations'] = repro.get('num_regenerations')

            # Add translations if available
            query_translation = get_translation(entry_data.get('query', ''), translation_cache)
            response_translation = get_translation(entry_data.get('response', ''), translation_cache)
            if query_translation:
                entry_data['query_translation'] = query_translation
            if response_translation:
                entry_data['response_translation'] = response_translation

            # Add marking
            entry_data['marking'] = markings.get(sample_id)

            matched_entries.append(entry_data)

        if not matched_entries:
            continue

        if exp not in export_data['by_experiment']:
            export_data['by_experiment'][exp] = {}

        # Sort by score descending
        matched_entries.sort(key=lambda x: x.get('score', 0), reverse=True)

        export_data['by_experiment'][exp][model] = matched_entries

        # Also add to flat list
        export_data['all_entries'].extend(matched_entries)

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
    for exp in experiments_include:
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
        description="Export entries marked as good/great from experiment viewer"
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
        "--format",
        choices=['json', 'jsonl'],
        default='json',
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--mapping-file",
        type=str,
        default=DEFAULT_MAPPING_FILE,
        help="Path to latest_experiments.json mapping file",
    )
    parser.add_argument(
        "--markings-file",
        type=str,
        default=DEFAULT_MARKINGS_FILE,
        help="Path to sample_markings.json from experiment viewer",
    )
    parser.add_argument(
        "--translation-file",
        type=str,
        default=DEFAULT_TRANSLATION_CACHE_FILE,
        help="Path to translation_cache.json from experiment viewer",
    )
    parser.add_argument(
        "--include-story",
        action="store_true",
        help="Include story_bleed experiment (excluded by default)",
    )
    parser.add_argument(
        "--repro-threshold",
        type=float,
        default=70.0,
        help="Threshold for reproducibility fraction_above_threshold (0-100 scale)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_marked_entries(
        source_dir=args.source_dir,
        output_path=args.output,
        format=args.format,
        mapping_file=args.mapping_file,
        markings_file=args.markings_file,
        translation_file=args.translation_file,
        include_story=args.include_story,
        repro_threshold=args.repro_threshold,
    )
