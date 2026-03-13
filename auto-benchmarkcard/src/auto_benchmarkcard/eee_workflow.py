"""EEE-to-BenchmarkCard workflow.

Alternative pipeline entry point that starts from Every Eval Ever (EEE)
evaluation data instead of UnitXT. Scans EEE evaluation JSONs, resolves
HuggingFace repos, then feeds into the standard composition pipeline.

Flow:
  EEE data → scan & aggregate → resolve HF repos
    → [HF Worker] → [Docling Worker] → [Composer Worker]
    → [Risk Worker] → [RAG Worker] → [FactReasoner Worker]
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from auto_benchmarkcard.config import Config
from auto_benchmarkcard.tools.eee.eee_tool import (
    scan_eee_folder,
    eee_to_pipeline_inputs,
    lookup_unitxt_paper,
)
from auto_benchmarkcard.workflow import (
    GraphState,
    OutputManager,
    build_workflow,
    sanitize_benchmark_name,
    setup_logging_suppression,
)

logger = logging.getLogger(__name__)


def build_eee_initial_state(
    benchmark_name: str,
    pipeline_inputs: Dict[str, Any],
    output_manager: OutputManager,
) -> Dict[str, Any]:
    """Build initial workflow state from EEE pipeline inputs.

    Pre-populates extracted_ids, hf_repo, and eee_metadata. The orchestrator
    detects eee_metadata and skips UnitXT + extractor steps automatically.

    Args:
        benchmark_name: Name of the benchmark.
        pipeline_inputs: Output from eee_to_pipeline_inputs().
        output_manager: Output manager for this benchmark.

    Returns:
        Initial state dict compatible with GraphState.
    """
    return {
        "query": benchmark_name,
        "catalog_path": None,
        "output_manager": output_manager,
        # EEE does not use UnitXT data — orchestrator skips unitxt/extractor
        # when eee_metadata is present
        "unitxt_json": None,
        "extracted_ids": pipeline_inputs["extracted_ids"],
        "hf_repo": pipeline_inputs["hf_repo"],
        # Rest starts empty — pipeline fills these
        "hf_json": None,
        "docling_output": None,
        "composed_card": None,
        "risk_enhanced_card": None,
        "completed": ["eee_scan done", "eee_resolve done"],
        "errors": [],
        "hf_extraction_attempted": False,
        "rag_results": None,
        "factuality_results": None,
        # EEE metadata is the primary data source
        "eee_metadata": pipeline_inputs["eee_metadata"],
    }


def _inject_evaluation_summary(final_card: Dict[str, Any], eee_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Inject the EEE evaluation summary into the final benchmark card.

    Adds the evaluation_summary field with aggregated model performance data.

    Args:
        final_card: The completed benchmark card dict.
        eee_metadata: EEE metadata containing evaluation_summary.

    Returns:
        Card with evaluation_summary added.
    """
    eval_summary = eee_metadata.get("evaluation_summary", {})
    if not eval_summary:
        return final_card

    card = final_card.get("benchmark_card", final_card)
    card["evaluation_summary"] = eval_summary

    # Also enrich baseline_results if it's empty or generic
    methodology = card.get("methodology", {})
    baseline = methodology.get("baseline_results", "")
    if not baseline or baseline.lower() in ("not specified", "not specified."):
        top = eval_summary.get("top_performers", [])
        stats = eval_summary.get("score_statistics", {})
        metric = eval_summary.get("primary_metric", "score")
        n_models = eval_summary.get("total_models_evaluated", 0)

        if top and stats:
            top_str = ", ".join(
                f"{p['model']} ({p['score']:.4f})" for p in top[:3]
            )
            methodology["baseline_results"] = (
                f"Based on {n_models} model evaluations from Every Eval Ever: "
                f"mean {metric} = {stats['mean']:.4f} (std = {stats['std_dev']:.4f}). "
                f"Top performers: {top_str}."
            )
            card["methodology"] = methodology

    if "benchmark_card" in final_card:
        final_card["benchmark_card"] = card
    else:
        final_card = card

    return final_card


def process_single_benchmark(
    benchmark_name: str,
    pipeline_inputs: Dict[str, Any],
    base_output_path: Optional[str] = None,
    debug: bool = False,
) -> Optional[Dict[str, Any]]:
    """Run the full pipeline for a single EEE benchmark.

    Args:
        benchmark_name: Name of the benchmark to process.
        pipeline_inputs: Output from eee_to_pipeline_inputs().
        base_output_path: Optional output directory.
        debug: Enable debug logging.

    Returns:
        Final benchmark card dict, or None on failure.
    """
    safe_name = sanitize_benchmark_name(benchmark_name)
    output_manager = OutputManager(safe_name, base_output_path)

    # Save EEE metadata as tool output
    eee_metadata = pipeline_inputs.get("eee_metadata", {})
    output_manager.save_tool_output(eee_metadata, "eee", f"{safe_name}.json")

    # Try to find paper URL via UnitXT catalog if not already set
    hf_repo = pipeline_inputs.get("hf_repo")
    extracted_ids = pipeline_inputs.get("extracted_ids", {})
    if not extracted_ids.get("paper_url") and hf_repo:
        unitxt_paper = lookup_unitxt_paper(hf_repo)
        if unitxt_paper:
            extracted_ids["paper_url"] = unitxt_paper

    # Build initial state
    initial_state = build_eee_initial_state(benchmark_name, pipeline_inputs, output_manager)

    # Run the standard workflow (skips unitxt + extractor automatically)
    workflow = build_workflow()

    logger.info("Processing benchmark: %s (hf_repo=%s)", benchmark_name, pipeline_inputs.get("hf_repo"))

    try:
        final_state = workflow.invoke(initial_state)

        # Inject evaluation summary into the final card
        final_card = final_state.get("final_card")
        if final_card and eee_metadata:
            final_card = _inject_evaluation_summary(final_card, eee_metadata)

            # Re-save the card with evaluation summary
            card_filename = f"benchmark_card_{safe_name}.json"
            output_manager.save_benchmark_card(final_card, card_filename)
            logger.info("Saved benchmark card with evaluation summary: %s", card_filename)

        # Log results
        completed = final_state.get("completed", [])
        errors = final_state.get("errors", [])
        logger.info("Completed steps: %s", completed)
        if errors:
            logger.warning("Errors: %s", errors)

        return final_card

    except Exception as e:
        logger.error("Failed to process %s: %s", benchmark_name, e, exc_info=debug)
        return None


def run_eee_pipeline(
    eee_path: str,
    output_path: Optional[str] = None,
    max_files_per_benchmark: int = 50,
    benchmarks_filter: Optional[List[str]] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """Run the full EEE-to-BenchmarkCard pipeline.

    Scans EEE data, discovers benchmarks, resolves sources, and generates
    benchmark cards for each discovered benchmark.

    Args:
        eee_path: Path to EEE data directory.
        output_path: Optional base output directory.
        max_files_per_benchmark: Max eval files to sample per benchmark folder.
        benchmarks_filter: If set, only process these benchmark names.
        debug: Enable debug logging.

    Returns:
        Summary dict with results per benchmark.
    """
    setup_logging_suppression(debug_mode=debug)

    logger.info("Models — composer: %s | light: %s | factreasoner: %s",
                Config.COMPOSER_MODEL, Config.LIGHT_MODEL, Config.FACTREASONER_MODEL)

    # Step 1: Scan EEE data
    logger.info("Scanning EEE data at: %s", eee_path)
    scan_result = scan_eee_folder(eee_path, max_files_per_benchmark)

    if scan_result.errors:
        for err in scan_result.errors:
            logger.warning("Scan error: %s", err)

    benchmarks = scan_result.benchmarks
    logger.info("Found %d unique benchmarks in %d files", len(benchmarks), scan_result.total_eval_files)

    # Apply filter if provided
    if benchmarks_filter:
        filter_set = {b.lower() for b in benchmarks_filter}
        benchmarks = {
            k: v for k, v in benchmarks.items()
            if k.lower() in filter_set
        }
        logger.info("Filtered to %d benchmarks: %s", len(benchmarks), list(benchmarks.keys()))

    # Step 2: Prepare pipeline inputs (resolves HF repos)
    logger.info("Resolving HuggingFace repos...")
    pipeline_inputs_map: Dict[str, Dict[str, Any]] = {}
    for name, bench in sorted(benchmarks.items()):
        inputs = eee_to_pipeline_inputs(bench)
        pipeline_inputs_map[name] = inputs
        hf = inputs.get("hf_repo", "None")
        logger.info("  %s -> hf_repo=%s (%d models)", name, hf, bench.num_models_evaluated)

    # Step 3: Process each benchmark
    summary = {
        "total_benchmarks": len(pipeline_inputs_map),
        "successful": [],
        "failed": [],
        "skipped": [],
    }

    for i, (name, inputs) in enumerate(sorted(pipeline_inputs_map.items()), 1):
        logger.info("\n=== [%d/%d] Processing: %s ===", i, len(pipeline_inputs_map), name)

        if not inputs.get("hf_repo"):
            logger.warning("Skipping %s: no HF repo resolved", name)
            summary["skipped"].append({"benchmark": name, "reason": "no_hf_repo"})
            continue

        card = process_single_benchmark(
            benchmark_name=name,
            pipeline_inputs=inputs,
            base_output_path=output_path,
            debug=debug,
        )

        if card:
            summary["successful"].append(name)
        else:
            summary["failed"].append(name)

    logger.info("\n=== EEE Pipeline Complete ===")
    logger.info("Success: %d | Failed: %d | Skipped: %d",
                len(summary["successful"]), len(summary["failed"]), len(summary["skipped"]))

    return summary
