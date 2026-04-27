#!/usr/bin/env python3
"""Phase 6: Knowledge fingerprinting and distillation detection.

Analyzes per-probe correctness patterns across models to detect:
  1. Knowledge overlap on rare probes (T5-T7) — fingerprint similarity
  2. Shared hallucinations — same wrong answers on failed probes
  3. Model provenance — which open-source model a target most resembles

This script operates on existing results from phases 2-4. It does NOT
require additional API calls — it analyzes the response data already
collected by 02_run_calibration.py and 04_run_targets.py.

Usage:
  python scripts/13_distillation_detection.py
  python scripts/13_distillation_detection.py --tiers T5 T6 T7
  python scripts/13_distillation_detection.py --include-targets
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent.parent / "data" / "distillation_detection.log"),
    ],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
FIG_DIR = PROJECT_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# Data Loading
# ================================================================

def load_all_responses():
    """Load all model response files. Returns dict: model_name -> probe_results."""
    responses_dir = PROJECT_ROOT / "data" / "raw_responses"
    all_responses = {}

    for f in sorted(responses_dir.glob("*_responses.json")):
        try:
            data = json.load(open(f))
            model_name = data["model_name"]
            all_responses[model_name] = data["probe_results"]
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load {f.name}: {e}")

    logger.info(f"Loaded responses for {len(all_responses)} models")
    return all_responses


def load_model_metadata():
    """Load model configs for family/architecture info."""
    config = json.load(open(PROJECT_ROOT / "configs" / "models.json"))
    metadata = {}
    for name, info in config.get("calibration_models", {}).items():
        metadata[name] = info
    for name, info in config.get("target_models", {}).items():
        metadata[name] = info
    return metadata


# ================================================================
# Knowledge Fingerprint Extraction
# ================================================================

def extract_fingerprints(all_responses, tiers=("T5", "T6", "T7")):
    """Extract binary knowledge fingerprint vectors for each model.

    Returns:
        fingerprints: dict model_name -> dict probe_id -> bool (correct)
        probe_ids: sorted list of probe IDs in the fingerprint
        wrong_answers: dict model_name -> dict probe_id -> str (actual response text)
    """
    # Find all probe IDs in target tiers across all models
    probe_ids_set = set()
    for model_name, results in all_responses.items():
        for r in results:
            pid = r["probe_id"]
            if any(pid.startswith(t) for t in tiers) and not r.get("excluded", False):
                probe_ids_set.add(pid)

    probe_ids = sorted(probe_ids_set)
    logger.info(f"Fingerprint probes: {len(probe_ids)} across tiers {tiers}")

    fingerprints = {}
    wrong_answers = {}

    for model_name, results in all_responses.items():
        fp = {}
        wa = {}
        result_map = {r["probe_id"]: r for r in results}

        for pid in probe_ids:
            if pid in result_map and not result_map[pid].get("excluded", False):
                fp[pid] = result_map[pid]["correct"]
                # Extract actual response text for hallucination analysis
                if not result_map[pid]["correct"]:
                    responses = result_map[pid].get("responses", {})
                    # Get the first non-refusal response text
                    for phrasing in ("direct", "fill_blank", "contextual"):
                        resp = responses.get(phrasing, {})
                        if resp and not resp.get("is_refusal", False):
                            wa[pid] = resp.get("text", "")
                            break

        fingerprints[model_name] = fp
        wrong_answers[model_name] = wa

    return fingerprints, probe_ids, wrong_answers


# ================================================================
# Fingerprint Similarity (Knowledge Overlap)
# ================================================================

def compute_pairwise_fps(fingerprints, probe_ids, tiers=("T5", "T6", "T7")):
    """Compute Fingerprint Similarity (FPS) for all model pairs.

    FPS(A, B) = Σ_t w_t × [observed_overlap_t / expected_overlap_t]

    Where expected overlap under independence = |K_A^t| × |K_B^t| / |probes_t|
    """
    tier_weights = {"T5": 1.0, "T6": 2.0, "T7": 4.0}

    # Group probes by tier
    probes_by_tier = defaultdict(list)
    for pid in probe_ids:
        tier = pid.split("_")[0]
        if tier in tiers:
            probes_by_tier[tier].append(pid)

    model_names = sorted(fingerprints.keys())
    n = len(model_names)

    fps_matrix = np.ones((n, n))  # diagonal = 1
    p_value_matrix = np.ones((n, n))
    detail_matrix = {}

    for i, j in combinations(range(n), 2):
        name_a, name_b = model_names[i], model_names[j]
        fp_a, fp_b = fingerprints[name_a], fingerprints[name_b]

        weighted_ratio_sum = 0
        total_weight = 0
        tier_details = {}
        combined_p = 1.0

        for tier, pids in probes_by_tier.items():
            w = tier_weights.get(tier, 1.0)

            # Count correct for each model on this tier
            correct_a = [pid for pid in pids if fp_a.get(pid, False)]
            correct_b = [pid for pid in pids if fp_b.get(pid, False)]

            n_total = len(pids)
            n_a = len(correct_a)
            n_b = len(correct_b)

            if n_a == 0 or n_b == 0 or n_total == 0:
                tier_details[tier] = {
                    "n_probes": n_total, "n_a": n_a, "n_b": n_b,
                    "overlap": 0, "expected": 0, "ratio": 1.0, "p_value": 1.0,
                }
                continue

            # Observed overlap
            overlap = len(set(correct_a) & set(correct_b))

            # Expected overlap under independence
            expected = n_a * n_b / n_total

            # Ratio (FPS contribution)
            ratio = overlap / expected if expected > 0 else 1.0

            # Hypergeometric test: P(X >= overlap | N=n_total, K=n_a, n=n_b)
            # This tests whether the overlap is significantly more than expected
            p_val = stats.hypergeom.sf(overlap - 1, n_total, n_a, n_b)

            tier_details[tier] = {
                "n_probes": n_total,
                "n_a": n_a,
                "n_b": n_b,
                "overlap": overlap,
                "expected": round(expected, 2),
                "ratio": round(ratio, 2),
                "p_value": p_val,
            }

            weighted_ratio_sum += w * ratio
            total_weight += w
            combined_p *= p_val  # Very rough combined p-value

        fps = weighted_ratio_sum / total_weight if total_weight > 0 else 1.0
        fps_matrix[i, j] = fps
        fps_matrix[j, i] = fps
        p_value_matrix[i, j] = combined_p
        p_value_matrix[j, i] = combined_p

        detail_matrix[(name_a, name_b)] = {
            "fps": round(fps, 3),
            "combined_p": combined_p,
            "tiers": tier_details,
        }

    return fps_matrix, p_value_matrix, detail_matrix, model_names


# ================================================================
# Shared Hallucination Analysis
# ================================================================

def compute_hallucination_similarity(fingerprints, wrong_answers, probe_ids):
    """Compute Hallucination Similarity Score (HSS) for all model pairs.

    HSS(A, B) = |{p : A(p) == B(p) != gold(p)}| / |{p : both wrong}|

    Two models giving the same wrong answer is a strong derivation signal.
    """
    model_names = sorted(fingerprints.keys())
    n = len(model_names)
    hss_matrix = np.zeros((n, n))
    hss_details = {}

    for i, j in combinations(range(n), 2):
        name_a, name_b = model_names[i], model_names[j]
        fp_a, fp_b = fingerprints[name_a], fingerprints[name_b]
        wa_a, wa_b = wrong_answers[name_a], wrong_answers[name_b]

        both_wrong = 0
        same_wrong = 0
        shared_wrong_examples = []

        for pid in probe_ids:
            a_correct = fp_a.get(pid, None)
            b_correct = fp_b.get(pid, None)

            # Both answered and both wrong
            if a_correct is False and b_correct is False:
                both_wrong += 1
                ans_a = wa_a.get(pid, "").strip().lower()
                ans_b = wa_b.get(pid, "").strip().lower()

                if ans_a and ans_b and ans_a == ans_b:
                    same_wrong += 1
                    if len(shared_wrong_examples) < 10:
                        shared_wrong_examples.append({
                            "probe_id": pid,
                            "shared_answer": wa_a.get(pid, ""),
                        })

        hss = same_wrong / both_wrong if both_wrong > 0 else 0
        hss_matrix[i, j] = hss
        hss_matrix[j, i] = hss

        hss_details[(name_a, name_b)] = {
            "hss": round(hss, 4),
            "both_wrong": both_wrong,
            "same_wrong": same_wrong,
            "examples": shared_wrong_examples,
        }

    return hss_matrix, hss_details, model_names


# ================================================================
# Combined Provenance Score
# ================================================================

def compute_provenance_scores(fps_matrix, hss_matrix, model_names, alpha=0.6, beta=0.4):
    """Combine FPS and HSS into a single provenance score."""
    # Normalize HSS to similar scale as FPS
    hss_max = hss_matrix.max() if hss_matrix.max() > 0 else 1.0
    hss_normalized = hss_matrix / hss_max * fps_matrix.max()

    provenance = alpha * fps_matrix + beta * hss_normalized

    return provenance


# ================================================================
# Known Pair Validation
# ================================================================

KNOWN_DERIVED_PAIRS = [
    # (student, teacher_or_base, relationship)
    # --- DeepSeek-R1 reasoning fine-tune ---
    ("deepseek-r1", "deepseek-v3", "reasoning fine-tune of same base"),
    # --- DeepSeek-R1 distilled into Qwen bases ---
    # These have TWO parents: R1 (teacher) and Qwen (base architecture)
    # Fingerprint should show overlap with BOTH, but more with R1 (knowledge source)
    ("deepseek-r1-distill-qwen-1.5b", "deepseek-r1", "distilled from R1 teacher"),
    ("deepseek-r1-distill-qwen-7b", "deepseek-r1", "distilled from R1 teacher"),
    ("deepseek-r1-distill-qwen-7b", "qwen-2.5-7b", "shares Qwen-2.5-7B base weights"),
    ("deepseek-r1-distill-qwen-14b", "deepseek-r1", "distilled from R1 teacher"),
    ("deepseek-r1-distill-qwen-32b", "deepseek-r1", "distilled from R1 teacher"),
    ("deepseek-r1-distill-qwen-32b", "qwq-32b", "both derive from Qwen-2.5-32B base"),
    # --- DeepSeek-R1 distilled into Llama bases ---
    ("deepseek-r1-distill-llama-8b", "deepseek-r1", "distilled from R1 teacher"),
    ("deepseek-r1-distill-llama-8b", "llama-3.1-8b", "shares Llama-3.1-8B base weights"),
    ("deepseek-r1-distill-llama-70b", "deepseek-r1", "distilled from R1 teacher"),
    ("deepseek-r1-distill-llama-70b", "llama-3.3-70b", "shares Llama-3.3-70B base weights"),
    # --- DeepSeek-R1-0528 distill ---
    ("deepseek-r1-0528-qwen3-8b", "qwen3-8b", "shares Qwen3-8B base weights"),
    # --- Nous Hermes fine-tunes of Llama-3.1-405B ---
    ("hermes-3-405b", "hermes-4-405b", "both fine-tunes of same Llama-3.1-405B base"),
    # --- NVIDIA Nemotron derivatives of Llama ---
    ("nemotron-70b", "llama-3.1-70b", "NVIDIA fine-tune of Llama-3.1-70B"),
    ("nemotron-super-49b", "llama-3.3-70b", "pruned + distilled from Llama-3.3-70B"),
    ("nemotron-super-49b", "llama-3.1-70b", "derivative of Llama family"),
    # --- Same family, different generation ---
    ("llama-3.3-70b", "llama-3.1-70b", "same family, newer generation"),
    ("gemma-3-27b", "gemma-2-27b", "same family, newer generation"),
    # --- Reasoning fine-tunes ---
    ("qwq-32b", "qwen-2.5-72b", "reasoning fine-tune (Qwen family)"),
]

KNOWN_INDEPENDENT_PAIRS = [
    # Models from different vendors/families, trained independently
    ("llama-3.1-70b", "qwen-2.5-72b", "different vendors (Meta vs Alibaba)"),
    ("deepseek-v3", "mistral-large", "different vendors (DeepSeek vs Mistral)"),
    ("gemma-2-27b", "phi-4", "different vendors (Google vs Microsoft)"),
    ("llama-3.1-8b", "qwen-2.5-7b", "different vendors, similar size"),
    ("mistral-small-24b", "qwen3-32b", "different vendors"),
    ("gemma-3-27b", "mistral-small-24b", "different vendors, similar size"),
    # Cross-vendor at same size — should show low FPS
    ("deepseek-r1-distill-llama-8b", "qwen-2.5-7b", "R1-distill-Llama vs independent Qwen"),
    ("deepseek-r1-distill-qwen-32b", "llama-3.1-70b", "R1-distill-Qwen vs independent Llama"),
]


def validate_known_pairs(detail_matrix, hss_details, model_names):
    """Check that known derived pairs have high FPS and known independent pairs have low FPS."""
    logger.info("\n=== Validation on Known Pairs ===")

    results = {"derived": [], "independent": []}

    logger.info("\n  Known derived pairs:")
    for student, teacher, relationship in KNOWN_DERIVED_PAIRS:
        key = tuple(sorted([student, teacher]))
        if key in detail_matrix:
            fps = detail_matrix[key]["fps"]
            hss = hss_details.get(key, {}).get("hss", 0)
            logger.info(f"    {student:25s} <- {teacher:25s}: FPS={fps:.2f}, HSS={hss:.3f} ({relationship})")
            results["derived"].append({
                "student": student, "teacher": teacher,
                "relationship": relationship, "fps": fps, "hss": hss,
            })
        else:
            logger.info(f"    {student:25s} <- {teacher:25s}: NOT IN DATA")

    logger.info("\n  Known independent pairs:")
    for model_a, model_b, relationship in KNOWN_INDEPENDENT_PAIRS:
        key = tuple(sorted([model_a, model_b]))
        if key in detail_matrix:
            fps = detail_matrix[key]["fps"]
            hss = hss_details.get(key, {}).get("hss", 0)
            logger.info(f"    {model_a:25s} vs {model_b:25s}: FPS={fps:.2f}, HSS={hss:.3f} ({relationship})")
            results["independent"].append({
                "model_a": model_a, "model_b": model_b,
                "relationship": relationship, "fps": fps, "hss": hss,
            })
        else:
            logger.info(f"    {model_a:25s} vs {model_b:25s}: NOT IN DATA")

    # Compute separation
    derived_fps = [r["fps"] for r in results["derived"]]
    independent_fps = [r["fps"] for r in results["independent"]]
    if derived_fps and independent_fps:
        logger.info(f"\n  Derived pairs: mean FPS = {np.mean(derived_fps):.2f}")
        logger.info(f"  Independent pairs: mean FPS = {np.mean(independent_fps):.2f}")
        logger.info(f"  Separation: {np.mean(derived_fps) / np.mean(independent_fps):.1f}x")

    return results


# ================================================================
# Target Model Provenance Analysis
# ================================================================

def analyze_target_provenance(fps_matrix, model_names, model_metadata):
    """For each target model, find which calibration model it most resembles."""
    logger.info("\n=== Target Model Provenance Analysis ===")

    config = json.load(open(PROJECT_ROOT / "configs" / "models.json"))
    calibration_names = set(config.get("calibration_models", {}).keys())
    target_names = set(config.get("target_models", {}).keys())

    results = {}

    for i, name in enumerate(model_names):
        if name not in target_names:
            continue

        # Find top-3 most similar calibration models
        similarities = []
        for j, other in enumerate(model_names):
            if other in calibration_names and i != j:
                similarities.append((other, fps_matrix[i, j]))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top3 = similarities[:3]

        logger.info(f"\n  {name}:")
        for cal_name, fps in top3:
            cal_info = model_metadata.get(cal_name, {})
            family = cal_info.get("family", "?")
            params = cal_info.get("params_billion", "?")
            logger.info(f"    -> {cal_name:25s} (FPS={fps:.2f}, family={family}, params={params}B)")

        results[name] = {
            "top_matches": [{"model": m, "fps": round(f, 3)} for m, f in top3],
            "max_fps": round(top3[0][1], 3) if top3 else 0,
        }

    return results


# ================================================================
# Visualization
# ================================================================

def plot_fps_heatmap(fps_matrix, model_names, filename="fig_fingerprint_heatmap"):
    """Plot pairwise fingerprint similarity heatmap."""
    n = len(model_names)
    if n < 2:
        logger.warning("Need at least 2 models for heatmap")
        return

    fig, ax = plt.subplots(figsize=(max(12, n * 0.5), max(10, n * 0.4)))

    # Use log scale for better visibility (FPS ranges from ~0.5 to potentially >10)
    im = ax.imshow(fps_matrix, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=min(fps_matrix.max(), 5.0))

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(model_names, rotation=90, fontsize=7)
    ax.set_yticklabels(model_names, fontsize=7)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Fingerprint Similarity Score (FPS)")

    ax.set_title("Knowledge Fingerprint Similarity\n(FPS > 3.0 suggests derivation)")

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / f"{filename}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info(f"Saved {filename}")


def plot_fps_distribution(detail_matrix, known_pairs_results, filename="fig_fps_distribution"):
    """Plot distribution of FPS scores, highlighting known derived vs independent pairs."""
    all_fps = [v["fps"] for v in detail_matrix.values()]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_fps, bins=30, alpha=0.5, color="gray", label="All pairs", density=True)

    # Overlay known pairs
    if known_pairs_results:
        derived_fps = [r["fps"] for r in known_pairs_results.get("derived", [])]
        independent_fps = [r["fps"] for r in known_pairs_results.get("independent", [])]

        if derived_fps:
            for f in derived_fps:
                ax.axvline(f, color="red", linestyle="--", alpha=0.7)
            ax.axvline(derived_fps[0], color="red", linestyle="--", alpha=0.7,
                       label="Known derived pairs")
        if independent_fps:
            for f in independent_fps:
                ax.axvline(f, color="blue", linestyle="--", alpha=0.7)
            ax.axvline(independent_fps[0], color="blue", linestyle="--", alpha=0.7,
                       label="Known independent pairs")

    ax.axvline(3.0, color="black", linestyle=":", alpha=0.5, label="FPS=3.0 threshold")
    ax.set_xlabel("Fingerprint Similarity Score (FPS)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of FPS Scores Across All Model Pairs")
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / f"{filename}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info(f"Saved {filename}")


def plot_per_tier_overlap(detail_matrix, model_names, filename="fig_per_tier_overlap"):
    """Plot per-tier overlap ratios for notable pairs."""
    # Select interesting pairs: known derived + highest FPS unknown pairs
    interesting = []

    for (a, b), details in detail_matrix.items():
        interesting.append((a, b, details["fps"], details["tiers"]))

    interesting.sort(key=lambda x: x[2], reverse=True)
    top_pairs = interesting[:10]

    if not top_pairs:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    tiers = ["T5", "T6", "T7"]
    x = np.arange(len(tiers))
    width = 0.08
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_pairs)))

    for idx, (a, b, fps, tier_data) in enumerate(top_pairs):
        ratios = [tier_data.get(t, {}).get("ratio", 1.0) for t in tiers]
        offset = (idx - len(top_pairs) / 2) * width
        label = f"{a[:12]}↔{b[:12]} (FPS={fps:.1f})"
        ax.bar(x + offset, ratios, width, label=label, color=colors[idx], alpha=0.8)

    ax.axhline(1.0, color="black", linestyle="--", alpha=0.3, label="Expected under independence")
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylabel("Overlap Ratio (observed / expected)")
    ax.set_title("Per-Tier Knowledge Overlap for Top Model Pairs")
    ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / f"{filename}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info(f"Saved {filename}")


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Knowledge fingerprinting and distillation detection analysis"
    )
    parser.add_argument(
        "--tiers", nargs="+", default=["T5", "T6", "T7"],
        help="Tiers to use for fingerprinting (default: T5 T6 T7)"
    )
    parser.add_argument(
        "--threshold", type=float, default=3.0,
        help="FPS threshold for flagging potential derivation (default: 3.0)"
    )
    parser.add_argument(
        "--include-targets", action="store_true",
        help="Include target (closed-source) models in analysis"
    )
    args = parser.parse_args()

    logger.info("=== Phase 6: Knowledge Fingerprinting & Distillation Detection ===")
    logger.info(f"Using tiers: {args.tiers}")

    # Load data
    all_responses = load_all_responses()
    model_metadata = load_model_metadata()

    if not all_responses:
        logger.error("No response data found. Run calibration/target experiments first.")
        sys.exit(1)

    # Filter to calibration models only (unless --include-targets)
    if not args.include_targets:
        config = json.load(open(PROJECT_ROOT / "configs" / "models.json"))
        cal_names = set(config.get("calibration_models", {}).keys())
        all_responses = {k: v for k, v in all_responses.items() if k in cal_names}
        logger.info(f"Filtered to {len(all_responses)} calibration models")

    # Extract fingerprints
    fingerprints, probe_ids, wrong_answers = extract_fingerprints(
        all_responses, tiers=tuple(args.tiers)
    )

    # Compute FPS
    logger.info("\n=== Computing Fingerprint Similarity (FPS) ===")
    fps_matrix, p_value_matrix, detail_matrix, model_names = compute_pairwise_fps(
        fingerprints, probe_ids, tiers=tuple(args.tiers)
    )

    # Compute HSS
    logger.info("\n=== Computing Hallucination Similarity Score (HSS) ===")
    hss_matrix, hss_details, _ = compute_hallucination_similarity(
        fingerprints, wrong_answers, probe_ids
    )

    # Combined provenance score
    provenance_matrix = compute_provenance_scores(fps_matrix, hss_matrix, model_names)

    # Report flagged pairs (FPS > threshold)
    logger.info(f"\n=== Flagged Pairs (FPS > {args.threshold}) ===")
    flagged = []
    for (a, b), details in detail_matrix.items():
        if details["fps"] > args.threshold:
            hss = hss_details.get((a, b), hss_details.get((b, a), {})).get("hss", 0)
            flagged.append((a, b, details["fps"], hss))
            logger.info(f"  {a:25s} <-> {b:25s}: FPS={details['fps']:.2f}, HSS={hss:.3f}")

    if not flagged:
        logger.info("  No pairs flagged above threshold")

    # Validate on known pairs
    known_results = validate_known_pairs(detail_matrix, hss_details, model_names)

    # Target provenance analysis
    if args.include_targets:
        target_results = analyze_target_provenance(
            fps_matrix, model_names, model_metadata
        )
    else:
        target_results = {}

    # Generate plots
    logger.info("\n=== Generating Figures ===")
    plot_fps_heatmap(fps_matrix, model_names)
    plot_fps_distribution(detail_matrix, known_results)
    plot_per_tier_overlap(detail_matrix, model_names)

    # Save full results
    results = {
        "config": {
            "tiers": args.tiers,
            "threshold": args.threshold,
            "n_models": len(model_names),
            "n_probes": len(probe_ids),
        },
        "model_names": model_names,
        "flagged_pairs": [
            {"model_a": a, "model_b": b, "fps": f, "hss": h}
            for a, b, f, h in flagged
        ],
        "known_pair_validation": known_results,
        "target_provenance": target_results,
        "pairwise_details": {
            f"{a}___{b}": details for (a, b), details in detail_matrix.items()
        },
        "hallucination_details": {
            f"{a}___{b}": details for (a, b), details in hss_details.items()
        },
    }

    output_file = RESULTS_DIR / "distillation_detection_results.json"
    json.dump(results, open(output_file, "w"), indent=2, default=str)
    logger.info(f"\nSaved full results to {output_file}")

    # Summary
    logger.info("\n=== Summary ===")
    logger.info(f"Models analyzed: {len(model_names)}")
    logger.info(f"Fingerprint probes: {len(probe_ids)}")
    logger.info(f"Total pairs: {len(detail_matrix)}")
    logger.info(f"Flagged pairs (FPS > {args.threshold}): {len(flagged)}")


if __name__ == "__main__":
    main()
