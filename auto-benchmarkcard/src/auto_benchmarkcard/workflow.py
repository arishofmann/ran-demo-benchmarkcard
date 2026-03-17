"""Benchmark metadata extraction and fact verification pipeline.

Orchestrates the complete workflow from raw benchmark data to fact-checked
benchmark cards.
"""

from __future__ import annotations

import argparse
import json
import logging
import operator
import os
import sys
import warnings
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional, TypedDict


# Conditional logging suppression (will be overridden if --debug is used)
def setup_logging_suppression(debug_mode=False):
    """Setup logging suppression unless in debug mode.

    Args:
        debug_mode: If True, enable debug logging. If False, suppress most logging.
    """
    if not debug_mode:
        # Suppress third-party library logging - set to ERROR to completely silence INFO/WARNING
        os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # LiteLLM specific environment variables to disable verbose logging
        os.environ["LITELLM_LOG"] = "ERROR"
        os.environ["LITELLM_DROP_PARAMS"] = "true"  # Drop extra params instead of logging warnings

        # Suppress tqdm progress bars (they conflict with CLI spinners)
        os.environ["TQDM_DISABLE"] = "1"

        # Disable success callbacks and verbose info logging
        try:
            import litellm

            litellm.suppress_debug_info = True
            litellm.set_verbose = False
        except (ImportError, AttributeError):
            pass  # LiteLLM not installed or doesn't have these attrs

        # Suppress all noisy loggers
        noisy_loggers = [
            "vllm",
            "vllm.config",
            "vllm.utils.import_utils",
            "vllm.importing",
            "transformers",
            "torch",
            "faiss.loader",
            "faiss",
            "ai_atlas_nexus",
            "AIAtlasNexus",  # Risk Atlas library logger
            "LiteLLM",  # Main LiteLLM logger
            "litellm",  # Alternative spelling
            "litellm.llms",
            "litellm.utils",
            "litellm.cost_calculator",
            "httpx",
            "httpcore",
            "openai",
            "urllib3",
            "asyncio",
        ]

        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
            logging.getLogger(logger_name).propagate = False  # Prevent propagation

        # Suppress all tool loggers to prevent duplicate output with CLI
        tool_loggers = [
            "benchmarkcard.tools.unitxt.unitxt_tool",
            "benchmarkcard.tools.hf.hf_tool",
            "benchmarkcard.tools.docling.docling_tool",
            "benchmarkcard.tools.composer.composer_tool",
            "benchmarkcard.tools.rag.rag_retriever",
            "benchmarkcard.tools.ai_atlas_nexus.ai_atlas_nexus_tool",
            "benchmarkcard.tools.factreasoner.factreasoner_tool",
        ]

        for logger_name in tool_loggers:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
            logging.getLogger(logger_name).propagate = False

        # Set benchmarkcard logger to WARNING (quiet mode by default)
        logging.getLogger("benchmarkcard").setLevel(logging.WARNING)
    else:
        # Debug mode: show everything
        logging.getLogger("benchmarkcard").setLevel(logging.DEBUG)


# Apply initial suppression (will be modified later based on debug flag)
setup_logging_suppression(debug_mode=False)

# Suppress warnings that occur during imports
warnings.filterwarnings("ignore", message=".*Triton.*")
warnings.filterwarnings("ignore", message=".*not installed.*")
warnings.filterwarnings("ignore", message=".*dummy decorators.*")
warnings.filterwarnings("ignore", message=".*Failed to load GPU.*")
warnings.filterwarnings("ignore", message=".*platform.*")
warnings.filterwarnings("ignore", message=".*tokenizers.*parallelism.*")
warnings.filterwarnings("ignore", message=".*TOKENIZERS_PARALLELISM.*")

from langgraph.graph import END, START, StateGraph

from auto_benchmarkcard.config import Config
from auto_benchmarkcard.tools.composer.composer_tool import compose_benchmark_card
from auto_benchmarkcard.tools.docling.docling_tool import extract_paper_with_docling
from auto_benchmarkcard.tools.extractor.extractor_tool import extract_ids
from auto_benchmarkcard.tools.factreasoner.factreasoner_tool import (
    evaluate_factuality,
    flag_benchmark_card_fields,
)
from auto_benchmarkcard.tools.hf.hf_tool import hf_dataset_metadata
from auto_benchmarkcard.tools.rag.atomizer import atomize_benchmark_card
from auto_benchmarkcard.tools.rag.format_converter import (
    convert_rag_to_required_format,
    save_formatted_results,
)
from auto_benchmarkcard.tools.rag.indexer import MetadataIndexer
from auto_benchmarkcard.tools.rag.rag_retriever import RAGRetriever
from auto_benchmarkcard.tools.ai_atlas_nexus.ai_atlas_nexus_tool import identify_and_integrate_risks
from auto_benchmarkcard.tools.unitxt import unitxt_tool

# Configure logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class BenchmarkProcessingError(Exception):
    """Custom exception for benchmark processing errors."""

    def __init__(self, message: str, operation: str, original_error: Exception = None):
        """Initialize with error context.

        Args:
            message: Human-readable error description.
            operation: The operation that failed.
            original_error: The original exception that caused this error.
        """
        super().__init__(message)
        self.operation = operation
        self.original_error = original_error


class OutputManager:
    """Manages organized output structure for benchmark processing results.

    Creates a timestamped directory structure:
    output/
    └── <benchmark_name>_<timestamp>/
        ├── tool_output/        # All tool outputs (unitxt, hf, docling, rag, etc.)
        └── benchmarkcard/      # Final benchmark cards
    """

    def __init__(self, benchmark_name: str, base_path: Optional[str] = None):
        """Initialize output manager with timestamped directory structure.

        Args:
            benchmark_name: Name of the benchmark being processed.
            base_path: Optional base path to create output directory in.
        """
        self.benchmark_name = sanitize_benchmark_name(benchmark_name)
        self.timestamp = self._generate_timestamp()
        self.session_dir = f"{self.benchmark_name}_{self.timestamp}"

        # Set base output path
        if base_path:
            self.base_dir = os.path.join(base_path, Config.OUTPUT_DIR, self.session_dir)
        else:
            self.base_dir = os.path.join(Config.OUTPUT_DIR, self.session_dir)

        # Define standard subdirectories
        self.tool_output_dir = os.path.join(self.base_dir, Config.TOOL_OUTPUT_DIR)
        self.benchmarkcard_dir = os.path.join(self.base_dir, Config.BENCHMARK_CARD_DIR)

        # Create directory structure
        self._create_directories()

        logger.debug("Output session directory: %s", self.base_dir)

    def _generate_timestamp(self) -> str:
        """Generate human-readable timestamp.

        Returns:
            Timestamp string in configured format.
        """
        return datetime.now().strftime(Config.TIMESTAMP_FORMAT)

    def _create_directories(self) -> None:
        """Create the standard directory structure."""
        os.makedirs(self.tool_output_dir, exist_ok=True)
        os.makedirs(self.benchmarkcard_dir, exist_ok=True)

    def save_tool_output(self, data: Dict[str, Any], tool_name: str, filename: str) -> str:
        """Save output from any tool.

        Args:
            data: Data to save.
            tool_name: Name of the tool (unitxt, hf, docling, rag, factreasoner, etc.).
            filename: Output filename.

        Returns:
            Path to saved file.
        """
        tool_dir = os.path.join(self.tool_output_dir, tool_name)
        os.makedirs(tool_dir, exist_ok=True)

        output_file = os.path.join(tool_dir, filename)
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        return output_file

    def save_benchmark_card(self, data: Dict[str, Any], filename: str) -> str:
        """Save final benchmark card.

        Args:
            data: Benchmark card data.
            filename: Output filename.

        Returns:
            Path to saved file.
        """
        output_file = os.path.join(self.benchmarkcard_dir, filename)
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        return output_file

    def get_tool_output_path(self, tool_name: str, create_if_missing: bool = True) -> str:
        """Get path for tool output directory.

        Args:
            tool_name: Name of the tool.
            create_if_missing: Whether to create directory if it doesn't exist.

        Returns:
            Path to tool output directory.
        """
        tool_dir = os.path.join(self.tool_output_dir, tool_name)
        if create_if_missing:
            os.makedirs(tool_dir, exist_ok=True)
        return tool_dir

    def get_summary(self) -> Dict[str, str]:
        """Get summary of output locations.

        Returns:
            Dictionary containing paths to output directories and timestamp.
        """
        return {
            "session_directory": self.base_dir,
            "tool_output": self.tool_output_dir,
            "benchmark_cards": self.benchmarkcard_dir,
            "timestamp": self.timestamp,
        }


def sanitize_benchmark_name(name: str) -> str:
    """Convert benchmark name to filesystem-safe format.

    Args:
        name: Benchmark name to sanitize.

    Returns:
        Sanitized benchmark name safe for filesystem use.
    """
    return name.replace("/", "_").replace(" ", "_")


def extract_card(obj: dict) -> dict:
    """Extract benchmark card from wrapper object if needed.

    Args:
        obj: Dictionary potentially containing a 'benchmark_card' key.

    Returns:
        Extracted benchmark card or original object.
    """
    return obj.get("benchmark_card", obj) if isinstance(obj, dict) else obj


def is_not_specified(value: Any) -> bool:
    """Check if a value represents a 'Not specified' field.

    Args:
        value: Value to check.

    Returns:
        True if value represents 'Not specified', False otherwise.
    """
    _EMPTY_VALUES = {"not specified", "not specified.", "no information found", "no information found."}
    if isinstance(value, str) and value.strip().lower() in _EMPTY_VALUES:
        return True
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str) and value[0].strip().lower() in _EMPTY_VALUES:
        return True
    return False


_SKIP_SECTIONS = {"possible_risks", "flagged_fields", "missing_fields", "card_info", "evaluation_summary", "provenance"}


def extract_missing_fields(data: Any, prefix: str = "") -> List[str]:
    """Recursively extract fields with 'Not specified' values.

    Args:
        data: Data structure to analyze.
        prefix: Path prefix for nested fields.

    Returns:
        List of field paths that have 'Not specified' values.
    """
    missing_fields = []

    if isinstance(data, dict):
        for key, value in data.items():
            if key in _SKIP_SECTIONS:
                continue
            current_path = f"{prefix}.{key}" if prefix else key

            if is_not_specified(value):
                missing_fields.append(current_path)
            elif isinstance(value, (dict, list)):
                missing_fields.extend(extract_missing_fields(value, current_path))

    elif isinstance(data, list):
        for i, item in enumerate(data):
            current_path = f"{prefix}[{i}]" if prefix else f"[{i}]"

            # Only flag individual list items if the list has multiple elements
            if isinstance(item, str) and item == "Not specified" and len(data) > 1:
                missing_fields.append(current_path)
            elif isinstance(item, (dict, list)):
                missing_fields.extend(extract_missing_fields(item, current_path))

    return missing_fields


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def _backfill_from_provenance(
    card: Dict[str, Any], provenance: Dict[str, Any]
) -> Dict[str, Any]:
    """Fill 'Not specified' fields using provenance evidence when available."""
    for section_key, section_val in card.items():
        if not isinstance(section_val, dict):
            continue
        section_prov = provenance.get(section_key, {})
        if not section_prov:
            continue

        for field_key, field_val in section_val.items():
            field_prov = section_prov.get(field_key, {})
            if not field_prov or not field_prov.get("evidence"):
                continue

            evidence = field_prov["evidence"]
            # Skip evidence that is itself a "not specified" / "no information" variant
            if isinstance(evidence, str) and any(
                neg in evidence.lower() for neg in ["no information", "not specified", "not available", "none found"]
            ):
                continue
            if is_not_specified(field_val) and len(evidence) > 10:
                logger.debug(
                    "Backfilling %s.%s from provenance (%s)",
                    section_key, field_key, field_prov.get("source", "?"),
                )
                card[section_key][field_key] = evidence

    return card


def _normalize_not_specified(card: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize inconsistent 'Not specified' representations."""
    list_fields = {
        "domains", "languages", "similar_benchmarks", "resources",
        "audience", "tasks", "out_of_scope_uses", "methods", "metrics",
    }

    for section_key, section_val in card.items():
        if not isinstance(section_val, dict):
            continue
        for field_key, field_val in section_val.items():
            if field_key in ("provenance", "flagged_fields", "missing_fields", "card_info", "possible_risks"):
                continue

            is_list_field = field_key in list_fields

            if field_val is None or field_val == "" or field_val == []:
                card[section_key][field_key] = ["Not specified"] if is_list_field else "Not specified"
            elif isinstance(field_val, list) and all(
                isinstance(item, str) and item == "Not specified" for item in field_val
            ):
                card[section_key][field_key] = ["Not specified"]

    return card


# ---------------------------------------------------------------------------
# Deterministic HuggingFace tag extraction
# ---------------------------------------------------------------------------

_LANG_CODE_MAP = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
    "pt": "Portuguese", "ru": "Russian", "it": "Italian", "nl": "Dutch",
    "pl": "Polish", "sv": "Swedish", "da": "Danish", "no": "Norwegian",
    "fi": "Finnish", "tr": "Turkish", "hi": "Hindi", "bn": "Bengali",
    "vi": "Vietnamese", "th": "Thai", "he": "Hebrew", "cs": "Czech",
    "ro": "Romanian", "hu": "Hungarian", "uk": "Ukrainian", "id": "Indonesian",
    "ms": "Malay", "el": "Greek", "bg": "Bulgarian", "ca": "Catalan",
    "hr": "Croatian", "sk": "Slovak", "sl": "Slovenian", "lt": "Lithuanian",
    "lv": "Latvian", "et": "Estonian", "fa": "Persian", "ur": "Urdu",
    "ta": "Tamil", "te": "Telugu", "ml": "Malayalam", "mr": "Marathi",
    "sw": "Swahili", "af": "Afrikaans", "eu": "Basque", "gl": "Galician",
    "ka": "Georgian", "hy": "Armenian", "az": "Azerbaijani", "be": "Belarusian",
    "is": "Icelandic", "mk": "Macedonian", "sq": "Albanian", "sr": "Serbian",
    "cy": "Welsh", "ga": "Irish", "mt": "Maltese", "la": "Latin",
}

_LICENSE_MAP = {
    "mit": "MIT License",
    "apache-2.0": "Apache License 2.0",
    "cc-by-4.0": "Creative Commons Attribution 4.0",
    "cc-by-sa-4.0": "Creative Commons Attribution-ShareAlike 4.0",
    "cc-by-nc-4.0": "Creative Commons Attribution-NonCommercial 4.0",
    "cc-by-nc-sa-4.0": "Creative Commons Attribution-NonCommercial-ShareAlike 4.0",
    "cc-by-nc-nd-4.0": "Creative Commons Attribution-NonCommercial-NoDerivatives 4.0",
    "cc0-1.0": "Creative Commons Zero v1.0 Universal",
    "openrail": "Open RAIL License",
    "gpl-3.0": "GNU General Public License v3.0",
    "lgpl-3.0": "GNU Lesser General Public License v3.0",
    "bsd-3-clause": "BSD 3-Clause License",
    "bsd-2-clause": "BSD 2-Clause License",
    "odc-by": "Open Data Commons Attribution License",
    "odbl": "Open Data Commons Open Database License",
    "pddl": "Open Data Commons Public Domain Dedication and License",
}

_SIZE_CATEGORY_MAP = {
    "n<1K": "Less than 1K examples",
    "1K<n<10K": "1K to 10K examples",
    "10K<n<100K": "10K to 100K examples",
    "100K<n<1M": "100K to 1M examples",
    "1M<n<10M": "1M to 10M examples",
    "10M<n<100M": "10M to 100M examples",
    "100M<n<1B": "100M to 1B examples",
    "1B<n<10B": "1B to 10B examples",
    "10B<n<100B": "10B to 100B examples",
    "100B<n<1T": "100B to 1T examples",
}


def _extract_hf_tags(hf_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract deterministic field values from HuggingFace dataset tags.

    Returns a flat dict keyed by dotted field paths, e.g.:
        {"benchmark_details.languages": ["English"], "data.format": "parquet"}
    Only includes keys where a value was actually found.
    """
    if not hf_metadata:
        return {}

    tags = hf_metadata.get("tags")
    if tags is None:
        for v in hf_metadata.values():
            if isinstance(v, dict) and "tags" in v:
                tags = v["tags"]
                break
    if not tags or not isinstance(tags, list):
        return {}

    result: Dict[str, Any] = {}

    languages = []
    modalities = []
    task_labels = []
    fmt = None
    size = None
    license_val = None

    for tag in tags:
        if not isinstance(tag, str) or ":" not in tag:
            continue
        prefix, _, value = tag.partition(":")
        value = value.strip()
        if not value:
            continue

        if prefix == "language":
            lang_name = _LANG_CODE_MAP.get(value, value)
            languages.append(lang_name)

        elif prefix == "modality":
            modalities.append(value.capitalize())

        elif prefix in ("task_ids", "task_categories"):
            label = value.replace("-", " ").replace("_", " ")
            label = label[0].upper() + label[1:]
            task_labels.append(label)

        elif prefix == "format":
            fmt = value

        elif prefix == "size_categories":
            size = _SIZE_CATEGORY_MAP.get(value, value)

        elif prefix == "license":
            if value not in ("other", "unknown"):
                license_val = _LICENSE_MAP.get(value, value)

    if languages:
        result["benchmark_details.languages"] = sorted(set(languages))
    if modalities:
        result["benchmark_details.data_type"] = ", ".join(sorted(set(modalities))).lower()
    if task_labels:
        result["purpose_and_intended_users.tasks"] = sorted(set(task_labels))
    if fmt:
        result["data.format"] = fmt
    if size:
        result["data.size"] = size
    if license_val:
        result["ethical_and_legal_considerations.data_licensing"] = license_val

    return result


def _apply_deterministic_overrides(
    card: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply deterministic HF-extracted values to the benchmark card.

    Smart override: only replaces LLM-generated content when:
    - The LLM value is empty, "Not specified", or missing
    - The field is a "always-override" field (languages, license, data_type)

    Fields like data.size and methodology.metrics are only overridden when
    the LLM didn't produce a more specific value.
    """
    # Fields where deterministic tags ALWAYS win (factual identity)
    _ALWAYS_OVERRIDE = {
        "benchmark_details.languages",
        "benchmark_details.data_type",
        "ethical_and_legal_considerations.data_licensing",
    }

    _EMPTY = {"not specified", "not specified.", "no information found", ""}

    for dotted_key, override_val in overrides.items():
        section, _, field = dotted_key.partition(".")
        if not field or section not in card or not isinstance(card[section], dict):
            continue

        old = card[section].get(field)

        # Always override for identity fields
        if dotted_key in _ALWAYS_OVERRIDE:
            card[section][field] = override_val
            if old != override_val:
                logger.debug("HF override (always) %s: %r -> %r", dotted_key, old, override_val)
            continue

        # For other fields, only override if LLM value is empty/generic
        llm_is_empty = False
        if old is None:
            llm_is_empty = True
        elif isinstance(old, str) and old.strip().lower() in _EMPTY:
            llm_is_empty = True
        elif isinstance(old, list) and (
            not old
            or (len(old) == 1 and isinstance(old[0], str) and old[0].strip().lower() in _EMPTY)
        ):
            llm_is_empty = True

        if llm_is_empty:
            card[section][field] = override_val
            logger.debug("HF override (fill) %s: %r -> %r", dotted_key, old, override_val)
        else:
            logger.debug("HF override skipped %s: LLM has specific value %r", dotted_key, str(old)[:80])

    return card


def handle_error(error: Exception, operation: str, state: GraphState) -> Dict[str, Any]:
    """Consistent error handling for workflow steps.

    Args:
        error: The exception that occurred.
        operation: Name of the operation that failed.
        state: Current workflow state.

    Returns:
        Dictionary with updated error list and completion status.
    """
    error_msg = f"{operation} failed: {error}"
    logger.error(error_msg, exc_info=True)  # Include stack trace

    # Log original error if it's a wrapped exception
    if hasattr(error, "original_error") and error.original_error:
        logger.error("Original error: %s", error.original_error)

    errors = state.get("errors", [])
    errors.append(error_msg)

    return {"errors": errors, "completed": [f"{operation.lower()} failed"]}


class GraphState(TypedDict):
    query: str
    catalog_path: Optional[str]
    output_manager: OutputManager
    unitxt_json: Optional[Dict[str, Any]]
    extracted_ids: Optional[Dict[str, Any]]
    hf_repo: Optional[str]
    hf_json: Optional[Dict[str, Any]]
    docling_output: Optional[Dict[str, Any]]
    completed: Annotated[list, operator.add]
    errors: Optional[List[str]]
    composed_card: Optional[Dict[str, Any]]
    risk_enhanced_card: Optional[Dict[str, Any]]
    hf_extraction_attempted: Optional[bool]
    rag_results: Optional[Dict[str, Any]]
    factuality_results: Optional[Dict[str, Any]]
    final_card: Optional[Dict[str, Any]]
    eee_metadata: Optional[Dict[str, Any]]


def orchestrator(state: GraphState) -> Dict[str, str]:
    """Determine next workflow step based on current state.

    Args:
        state: Current workflow state containing all intermediate results.

    Returns:
        Dictionary with 'next' key indicating the next worker to run.
    """
    # EEE path: skip UnitXT and extractor when EEE metadata is present
    is_eee = state.get("eee_metadata") is not None
    if not is_eee:
        if state["unitxt_json"] is None:
            return {"next": "unitxt_worker"}
        if state["extracted_ids"] is None:
            return {"next": "extractor_worker"}

    # HuggingFace lookup if we have repo ID
    if state["hf_repo"] is not None and state["hf_json"] is None:
        return {"next": "hf_worker"}

    # Try extracting paper URL from HF data if not found in UnitXT
    current_paper_url = state.get("extracted_ids", {}).get("paper_url")
    has_hf_data = state.get("hf_json") is not None
    hf_extraction_attempted = state.get("hf_extraction_attempted", False)
    needs_hf_extraction = not current_paper_url and has_hf_data and not hf_extraction_attempted

    if needs_hf_extraction:
        return {"next": "hf_extractor_worker"}

    # Paper extraction if we have URL
    paper_url = state.get("extracted_ids", {}).get("paper_url")
    if paper_url and state["docling_output"] is None:
        return {"next": "docling_worker"}
    if state["composed_card"] is None:
        return {"next": "composer_worker"}
    if state["risk_enhanced_card"] is None:
        return {"next": "risk_worker"}
    if state["rag_results"] is None:
        return {"next": "rag_worker"}
    if state["factuality_results"] is None:
        return {"next": "factreasoner_worker"}
    return {"next": "END"}


def run_unitxt(state: GraphState) -> Dict[str, Any]:
    """Retrieve UnitXT metadata for benchmark.

    Args:
        state: Current workflow state containing query and catalog path.

    Returns:
        Dictionary containing unitxt_json and completion status.
    """
    try:
        result = unitxt_tool.unitxt_benchmark_lookup(
            state["query"], catalog_path=state.get("catalog_path")
        )
        unitxt_data = result.model_dump(mode="json")

        logger.info("UnitXT metadata retrieved")

        # Show snippet
        name = unitxt_data.get("name", "N/A")
        description = unitxt_data.get("description", "")
        if description:
            desc_preview = description[:60] + "..." if len(description) > 60 else description
            logger.info(f"Found: {name} - {desc_preview}")

        filename = f"{sanitize_benchmark_name(state['query'])}{Config.JSON_EXTENSION}"
        output_file = state["output_manager"].save_tool_output(unitxt_data, "unitxt", filename)
        logger.info("UnitXT output saved to: %s", output_file)

        return {
            "unitxt_json": unitxt_data,
            "completed": ["unitxt done"],
        }
    except Exception as e:
        return handle_error(e, "UnitXT lookup", state)


def run_extractor(state: GraphState) -> Dict[str, Any]:
    """Extract IDs and metadata from UnitXT data.

    Args:
        state: Current workflow state containing unitxt_json.

    Returns:
        Dictionary containing extracted_ids, hf_repo, and completion status.
    """
    logger.info("Starting ID and URL extraction")
    try:
        extracted = extract_ids.func(
            source=state["unitxt_json"], want=["hf_repo", "paper_url", "risk_tags"]
        )

        hf_repo = extracted.get("hf_repo")
        paper_url = extracted.get("paper_url")

        logger.info("ID extraction completed")

        # Show snippet
        hf_status = hf_repo if hf_repo else "None"
        paper_status = "Found" if paper_url else "None"
        logger.info(f"Extracted: HF={hf_status}, Paper={paper_status}")

        filename = f"{sanitize_benchmark_name(state['query'])}{Config.JSON_EXTENSION}"
        output_file = state["output_manager"].save_tool_output(extracted, "extractor", filename)
        logger.info("Extractor output saved to: %s", output_file)

        return {
            "extracted_ids": extracted,
            "hf_repo": hf_repo,
            "completed": [f"extract hf_repo={hf_repo}, paper_url={paper_url}"],
        }
    except Exception as e:
        return handle_error(e, "ID extraction", state)


def run_hf_extractor(state: GraphState):
    """Extract paper_url from HF metadata when not found in unitxt.

    Args:
        state: Current workflow state containing hf_json and extracted_ids.

    Returns:
        Dictionary containing updated extracted_ids with paper_url from HF metadata.
    """
    logger.info("Starting HuggingFace extraction")
    try:
        # Get existing extracted_ids
        current_extracted = state.get("extracted_ids", {})

        # Try to extract paper_url from HF metadata
        hf_data = state["hf_json"]
        paper_url = None

        # First try extracting from the top-level HF JSON (tags are at top level)
        hf_extracted = extract_ids.func(source=hf_data, want=["paper_url"])
        paper_url = hf_extracted.get("paper_url")
        if paper_url:
            logger.info("Found paper_url in HF top-level metadata: %s", paper_url)

        # Fallback: check nested dataset dicts (e.g. card_data, builder_metadata values)
        if not paper_url:
            for dataset_id, dataset_metadata in hf_data.items():
                if isinstance(dataset_metadata, dict):
                    hf_extracted = extract_ids.func(source=dataset_metadata, want=["paper_url"])
                    extracted_paper_url = hf_extracted.get("paper_url")
                    if extracted_paper_url:
                        paper_url = extracted_paper_url
                        logger.info("Found paper_url in HF dataset %s: %s", dataset_id, paper_url)
                        break

        if paper_url:
            # Merge the paper_url into existing extracted_ids
            updated_extracted = current_extracted.copy()
            updated_extracted["paper_url"] = paper_url
            updated_extracted["paper_url_from_hf"] = paper_url

            # Update the saved extractor output
            filename = f"{sanitize_benchmark_name(state['query'])}{Config.JSON_EXTENSION}"
            output_file = state["output_manager"].save_tool_output(
                updated_extracted, "extractor", filename
            )
            logger.info("HF extractor output saved to: %s", output_file)

            return {
                "extracted_ids": updated_extracted,
                "hf_extraction_attempted": True,
                "completed": [f"hf_extract paper_url={paper_url}"],
            }
        else:
            logger.info("No paper_url found in HF metadata")
            return {
                "hf_extraction_attempted": True,
                "completed": ["hf_extract no paper_url found"],
            }

    except Exception as e:
        result = handle_error(e, "HF extraction", state)
        result["hf_extraction_attempted"] = True
        return result


def run_docling(state: GraphState):
    """Extract paper content using docling.

    Args:
        state: Current workflow state containing extracted_ids with paper_url.

    Returns:
        Dictionary containing docling_output and completion status.
    """
    paper_url = state.get("extracted_ids", {}).get("paper_url")
    if paper_url:
        logger.info("Starting paper extraction")

    if not paper_url:
        # No paper URL available - skip docling
        return {
            "docling_output": None,
            "completed": ["docling skipped (no paper_url)"],
        }

    try:
        logger.info("Extracting paper from: %s", paper_url)

        # Call the underlying function directly, not the tool wrapper
        docling_result = extract_paper_with_docling.func(paper_url=paper_url)

        if docling_result.get("success"):
            logger.info("Docling extraction completed successfully")

            # Show snippet
            metadata = docling_result.get("metadata", {})
            title = metadata.get("title", "Unknown Paper")
            char_count = len(docling_result.get("filtered_text", ""))
            logger.info(f"Paper: {title} ({char_count:,} chars)")

            # Save docling output
            filename = f"{sanitize_benchmark_name(state['query'])}{Config.JSON_EXTENSION}"
            output_file = state["output_manager"].save_tool_output(
                docling_result, "docling", filename
            )
            logger.info("Docling output saved to: %s", output_file)

            return {
                "docling_output": docling_result,
                "completed": ["docling done"],
            }
        else:
            warning_msg = docling_result.get("warning")
            if warning_msg:
                # Handle warning case - continue without paper content
                logger.warning("Docling warning: %s", warning_msg)
                return {
                    "docling_output": None,
                    "completed": ["docling warning - continuing without paper"],
                }
            else:
                # Handle error case
                error_msg = (
                    f"Docling extraction failed: {docling_result.get('error', 'Unknown error')}"
                )
                fake_error = Exception(error_msg)
                result = handle_error(fake_error, "Docling extraction", state)
                result["docling_output"] = None
                return result

    except Exception as e:
        result = handle_error(e, "Docling extraction", state)
        result["docling_output"] = None
        return result


def run_hf(state: GraphState):
    """Get HuggingFace metadata for the dataset.

    Args:
        state: Current workflow state containing hf_repo identifier.

    Returns:
        Dictionary containing hf_json and completion status.
    """
    if not state["hf_repo"]:
        fake_error = Exception("No hf_repo available for HuggingFace lookup")
        return handle_error(fake_error, "HuggingFace lookup", state)

    try:
        # Fetch hf metadata - use .func to get the underlying function
        hf_data = hf_dataset_metadata.func(repo_id=state["hf_repo"])

        logger.info("HuggingFace metadata retrieved successfully")

        # Show snippet
        card_data = hf_data.get("card_data") or {}
        dataset_info = hf_data.get("dataset_info") or {}
        name = dataset_info.get("dataset_name", card_data.get("pretty_name", "Unknown"))
        task_cats = card_data.get("task_categories", [])
        if task_cats:
            task_preview = ", ".join(task_cats[:2])
            logger.info(f"Dataset: {name} ({task_preview})")

        # Save HuggingFace output
        filename = f"{sanitize_benchmark_name(state['query'])}{Config.JSON_EXTENSION}"
        output_file = state["output_manager"].save_tool_output(hf_data, "hf", filename)
        logger.info("HuggingFace output saved to: %s", output_file)

        return {
            "hf_json": hf_data,
            "completed": ["hf done"],
        }
    except Exception as e:
        return handle_error(e, "HuggingFace lookup", state)


def run_composer(state: GraphState):
    """Use LLM to create the final benchmark card.

    Args:
        state: Current workflow state containing all collected metadata.

    Returns:
        Dictionary containing composed_card and completion status.
    """
    logger.info("Starting benchmark card composition")

    try:
        # compose the card with all the metadata we collected
        # Use .func to get the underlying function
        # Extract just the benchmark name from catalog queries (after the last dot)
        query_for_composer = state["query"]
        if state.get("catalog_path") and "." in state["query"]:
            query_for_composer = state["query"].split(".")[-1]

        # In EEE mode, don't pass unitxt_metadata (it's None)
        unitxt_for_composer = state.get("unitxt_json") if not state.get("eee_metadata") else None

        result = compose_benchmark_card.func(
            unitxt_metadata=unitxt_for_composer,
            hf_metadata=state.get("hf_json"),
            extracted_ids=state.get("extracted_ids", {}),
            docling_output=state.get("docling_output"),
            query=query_for_composer,
            eee_metadata=state.get("eee_metadata"),
        )

        logger.info("Successfully composed benchmark card")

        # Save provenance separately to tool_output/composer/
        if result.get("provenance"):
            provenance_filename = f"provenance_{sanitize_benchmark_name(state['query'])}.json"
            provenance_output = state["output_manager"].save_tool_output(
                result["provenance"], "composer", provenance_filename
            )
            logger.info(f"Provenance tracking saved to: {provenance_output}")

        # Show snippet
        benchmark_card = result.get("benchmark_card", {})
        details = benchmark_card.get("benchmark_details", {})
        name = details.get("name", "N/A")
        domains = details.get("domains", [])
        languages = details.get("languages", [])

        domain_str = ", ".join(domains[:2]) if domains else "General"
        lang_str = ", ".join(languages[:2]) if languages else "Unknown"
        logger.info(f"Card: {name} | {domain_str} | {lang_str}")

        return {
            "composed_card": result,
            "completed": ["composer done"],
        }
    except Exception as e:
        return handle_error(e, "Composer", state)


def run_risk_identification(state: GraphState):
    """Identify risks using ai-atlas-nexus and integrate them into the benchmark card.

    Args:
        state: Current workflow state containing composed_card.

    Returns:
        Dictionary containing risk_enhanced_card and completion status.
    """
    logger.info("Starting risk identification")

    if not state.get("composed_card"):
        fake_error = Exception("No composed card available for risk identification")
        return handle_error(fake_error, "Risk identification", state)

    try:
        # Get the benchmark card from composed output
        benchmark_card = state["composed_card"]
        if "benchmark_card" in benchmark_card:
            benchmark_card = benchmark_card["benchmark_card"]

        # Identify and integrate risks
        risk_enhanced_card = identify_and_integrate_risks(benchmark_card)

        # Save the risk-enhanced card
        filename = f"{sanitize_benchmark_name(state['query'])}{Config.JSON_EXTENSION}"
        output_file = state["output_manager"].save_tool_output(
            {"benchmark_card": risk_enhanced_card}, "risk_enhanced", filename
        )
        logger.info("Risk-enhanced card saved to: %s", output_file)

        # Save risk analysis results to tool_output
        possible_risks = risk_enhanced_card.get("possible_risks", [])
        if possible_risks:
            risk_output = {
                "benchmark": state["query"],
                "risks_identified": len(possible_risks),
                "taxonomy": "ibm-risk-atlas",
                "risks": possible_risks,
            }
            risk_filename = f"risks_{sanitize_benchmark_name(state['query'])}.json"
            risk_output_file = state["output_manager"].save_tool_output(
                risk_output, "ai_atlas_nexus", risk_filename
            )
            logger.info("Risk identification results saved to: %s", risk_output_file)

        logger.info("Risk identification completed")

        # Show snippet
        possible_risks = risk_enhanced_card.get("possible_risks", [])

        if possible_risks:
            risk_count = len(possible_risks)
            # Risk structure uses "category" field for the risk name
            risk_names = [
                risk.get("category", risk.get("name", "Unknown")) for risk in possible_risks[:2]
            ]
            risk_preview = ", ".join(risk_names)
            if risk_count > 2:
                risk_preview += f" (+{risk_count-2} more)"
            logger.info(f"Risks: {risk_preview}")
        else:
            logger.info("Risks: None detected")

        return {
            "risk_enhanced_card": {"benchmark_card": risk_enhanced_card},
            "completed": ["risk identification done"],
        }

    except Exception as e:
        return handle_error(e, "Risk identification", state)


def run_rag(state: GraphState) -> Dict[str, Any]:
    """Process benchmark card through enhanced RAG system.

    Args:
        state: Current workflow state containing composed_card and metadata.

    Returns:
        Dictionary containing rag_results and completion status.
    """
    logger.info("Starting RAG processing")

    if not state.get("composed_card"):
        return handle_error(Exception("No composed card for RAG"), "RAG processing", state)

    try:
        benchmark_name = sanitize_benchmark_name(state["query"])

        # Load metadata from previous steps
        unitxt_data = state.get("unitxt_json") or {}
        hf_data = state.get("hf_json") or {}
        docling_data = state.get("docling_output")

        # Create searchable documents
        indexer = MetadataIndexer()
        documents = indexer.create_documents(unitxt_data, hf_data, state["query"], docling_data)

        # Initialize enhanced RAG retriever with lightweight model for reranking/reformulation
        from auto_benchmarkcard.config import get_light_llm_handler

        try:
            light_llm = get_light_llm_handler()
            retriever = RAGRetriever(
                embedding_model=Config.DEFAULT_EMBEDDING_MODEL,
                enable_llm_reranking=Config.ENABLE_LLM_RERANKING,
                enable_hybrid_search=Config.ENABLE_HYBRID_SEARCH,
                enable_query_expansion=Config.ENABLE_QUERY_EXPANSION,
                llm_handler=light_llm,
            )
        except Exception as e:
            logger.warning(f"Enhanced retriever failed: {e}")
            logger.info("Using basic retriever fallback")
            retriever = RAGRetriever(
                embedding_model=Config.DEFAULT_EMBEDDING_MODEL,
                enable_llm_reranking=False,
                enable_hybrid_search=False,
                enable_query_expansion=False,
            )

        retriever.index_documents(documents)

        # Get benchmark card (excluding risk sections for fact checking)
        benchmark_card = state["composed_card"]
        if "benchmark_card" in benchmark_card:
            benchmark_card = benchmark_card["benchmark_card"]

        # Break card into atomic statements (lightweight model suffices)
        statements = atomize_benchmark_card(
            benchmark_card, "all",
            engine_type=Config.LLM_ENGINE_TYPE,
            model_name=Config.LIGHT_MODEL,
        )

        # Extract statement texts for batch processing
        statement_texts = []
        for statement_obj in statements:
            if isinstance(statement_obj, str):
                statement_texts.append(statement_obj)
            else:
                statement_texts.append(statement_obj.get("text", ""))

        # Retrieve evidence for all statements using parallel batch processing
        logger.debug(f"Processing {len(statement_texts)} statements with parallel reranking")

        # Use parallel reranking if LLM reranking is enabled, otherwise fall back to sequential
        if retriever.enable_llm_reranking and retriever.llm_handler:
            import asyncio

            # Use nest_asyncio for compatibility with Jupyter and other event loops
            try:
                import nest_asyncio

                nest_asyncio.apply()
            except ImportError:
                logger.debug("nest_asyncio not available, using standard asyncio")

            batch_chunks = asyncio.run(
                retriever.retrieve_for_statements_batch_parallel(statement_texts)
            )
        else:
            batch_chunks = retriever.retrieve_for_statements_batch(statement_texts)

        # Combine results with original statement objects
        results = []
        for statement_obj, chunks in zip(statements, batch_chunks):
            results.append({"statement": statement_obj, "retrieved_chunks": chunks})


        # Format results
        raw_results = {
            "benchmark": state["query"],
            "num_statements": len(statements),
            "num_documents_indexed": len(documents),
            "results": results,
        }

        formatted_results = convert_rag_to_required_format(raw_results, "all", benchmark_card)

        # Save results using OutputManager
        rag_filename = f"formatted_rag_results_{benchmark_name}.jsonl"
        # For JSONL files, we need to save manually since save_tool_output expects JSON
        rag_tool_dir = state["output_manager"].get_tool_output_path("rag")
        output_path = os.path.join(rag_tool_dir, rag_filename)
        save_formatted_results(formatted_results, output_path)

        logger.info("RAG processing completed")

        # Show snippet
        atom_count = len(formatted_results.get("atoms", []))
        context_count = len(formatted_results.get("contexts", []))
        logger.info(f"RAG: {atom_count} claims, {context_count} evidence sources")
        logger.info("RAG results saved to: %s", output_path)

        return {
            "rag_results": formatted_results,
            "completed": ["rag done"],
        }

    except Exception as e:
        return handle_error(e, "RAG processing", state)


def run_factreasoner(state: GraphState):
    """Evaluate factuality of RAG results and merge with risk-enhanced card.

    Args:
        state: Current workflow state containing rag_results and risk_enhanced_card.

    Returns:
        Dictionary containing factuality_results, final_card, and completion status.
    """
    logger.info("Starting factuality evaluation")

    if not state.get("rag_results"):
        fake_error = Exception("No RAG results available for factuality evaluation")
        return handle_error(fake_error, "FactReasoner evaluation", state)

    try:
        benchmark_name = sanitize_benchmark_name(state["query"])
        rag_results = state["rag_results"]


        # Run factuality evaluation (uses its own FactReasoner-library LLM config)
        factuality_results = evaluate_factuality(
            formatted_rag_results=rag_results,
            model=Config.FACTREASONER_MODEL,
            cache_dir=Config.FACTREASONER_CACHE_DIR,
            merlin_path=str(Config.MERLIN_BIN),
            debug_mode=False,
            use_priors=False,
        )

        # Save factuality analysis results
        factuality_filename = f"factuality_results_{benchmark_name}.json"
        factuality_output = state["output_manager"].save_tool_output(
            factuality_results, "factreasoner", factuality_filename
        )

        # Get the risk-enhanced card (which has all the risks) as base for flagging
        risk_card_src = state.get("risk_enhanced_card") or state.get("composed_card") or {}
        clean_card = extract_card(risk_card_src)

        # Extract provenance from composed card to allow provenance-aware flagging
        composed_card_data = state.get("composed_card", {})
        provenance_data = composed_card_data.get("provenance") if isinstance(composed_card_data, dict) else None

        field_analysis = factuality_results.get("field_analysis", {})

        # Field-type-aware scoring: different thresholds for different field types
        # Identity fields (overview, goal, domains) are descriptive — relaxed threshold
        # Factual fields (annotation, size, baseline_results) need strict verification
        # Analytical fields (limitations, out_of_scope) are reasoned — skip flagging
        _IDENTITY_FIELDS = {
            "benchmark_details.overview", "benchmark_details.domains",
            "benchmark_details.data_type", "benchmark_details.similar_benchmarks",
            "purpose_and_intended_users.goal", "purpose_and_intended_users.audience",
            "purpose_and_intended_users.tasks",
        }
        _ANALYTICAL_FIELDS = {
            "purpose_and_intended_users.limitations",
            "purpose_and_intended_users.out_of_scope_uses",
            "methodology.interpretation",
        }

        # Use standard threshold for factual fields, relaxed for identity, skip analytical
        flagged_card = flag_benchmark_card_fields(
            benchmark_card=clean_card,
            field_analysis=field_analysis,
            threshold=Config.DEFAULT_FACTUALITY_THRESHOLD,
            provenance=provenance_data,
        )

        # Post-flagging: remove flags for identity and analytical fields
        # Identity fields get a relaxed standard — only flag if very low confidence
        # Analytical fields are reasoned, not extracted — never flag based on NLI
        if "flagged_fields" in flagged_card and isinstance(flagged_card["flagged_fields"], dict):
            fields_to_unflag = []
            for field_name, flag_reason in flagged_card["flagged_fields"].items():
                if field_name in _ANALYTICAL_FIELDS:
                    fields_to_unflag.append(field_name)
                    logger.debug("Unflagging analytical field %s (reasoned, not extracted)", field_name)
                elif field_name in _IDENTITY_FIELDS and "Possible Hallucination" in str(flag_reason):
                    # Only unflag identity fields if they were neutral (no evidence),
                    # not if they had low factuality scores (which indicates actual errors)
                    fields_to_unflag.append(field_name)
                    logger.debug("Unflagging identity field %s (neutral NLI expected for descriptive content)", field_name)
            for field_name in fields_to_unflag:
                del flagged_card["flagged_fields"][field_name]
            if fields_to_unflag:
                logger.info("Field-type-aware scoring: unflagged %d identity/analytical fields", len(fields_to_unflag))

        # Backfill: if a field says "Not specified" but provenance has evidence, use it
        if provenance_data:
            flagged_card = _backfill_from_provenance(flagged_card, provenance_data)

        # Normalize "Not specified" variants (empty lists, None, etc.)
        flagged_card = _normalize_not_specified(flagged_card)

        # Override with deterministic HF-extracted values (factual tags beat LLM guesses)
        hf_overrides = _extract_hf_tags(state.get("hf_json"))
        if hf_overrides:
            logger.info("Applying %d deterministic HF overrides: %s",
                        len(hf_overrides), list(hf_overrides.keys()))
            flagged_card = _apply_deterministic_overrides(flagged_card, hf_overrides)

            # Clear stale flags for fields we just overwrote with factual data
            if "flagged_fields" in flagged_card and isinstance(flagged_card["flagged_fields"], dict):
                for dotted_key in hf_overrides:
                    flag_key = dotted_key.split(".")[-1]
                    for k in (flag_key, dotted_key):
                        if k in flagged_card["flagged_fields"]:
                            del flagged_card["flagged_fields"][k]
                            logger.debug("Cleared stale flag for %s (deterministic override)", k)

        # Add missing_fields section
        flagged_card["missing_fields"] = extract_missing_fields(flagged_card)

        # Add card_info section
        from datetime import datetime

        flagged_card["card_info"] = {
            "created_at": datetime.now().isoformat(),
            "llm": Config.COMPOSER_MODEL,
        }

        # Save the single benchmark card with everything
        card_filename = f"benchmark_card_{benchmark_name}.json"
        output_path = state["output_manager"].save_benchmark_card(
            {"benchmark_card": flagged_card},
            card_filename,
        )
        state["final_card"] = {"benchmark_card": flagged_card}

        logger.info("FactReasoner evaluation complete")

        # Show snippet
        marginals = factuality_results.get("marginals", [])
        claims_evaluated = len(marginals)
        # Calculate flagged fields directly from marginals (p_true < 0.3 or exactly 0.5 indicating no evidence)
        flagged_fields = len(
            [m for m in marginals if m.get("p_true", 1.0) < 0.3 or m.get("p_true", 1.0) == 0.5]
        )
        logger.info(
            f"Factuality: {claims_evaluated} claims evaluated, {flagged_fields}/{claims_evaluated} fields flagged"
        )

        logger.info("Factuality results saved to: %s", factuality_output)
        logger.info("Final benchmark card saved to: %s", output_path)

        return {
            "factuality_results": factuality_results,
            "final_card": {"benchmark_card": flagged_card},
            "completed": ["factreasoner done"],
        }

    except Exception as e:
        return handle_error(e, "FactReasoner evaluation", state)


# build the workflow graph
def build_workflow():
    """Build the LangGraph workflow for metadata extraction.

    Returns:
        Compiled LangGraph workflow.
    """
    builder = StateGraph(GraphState)

    # Add workflow nodes
    builder.add_node("orchestrator", orchestrator)
    builder.add_node("unitxt_worker", run_unitxt)
    builder.add_node("extractor_worker", run_extractor)
    builder.add_node("hf_extractor_worker", run_hf_extractor)
    builder.add_node("docling_worker", run_docling)
    builder.add_node("hf_worker", run_hf)
    builder.add_node("composer_worker", run_composer)
    builder.add_node("risk_worker", run_risk_identification)
    builder.add_node("rag_worker", run_rag)
    builder.add_node("factreasoner_worker", run_factreasoner)

    # Connect workflow
    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges(
        "orchestrator",
        lambda s, *_: s["next"],
        {
            "unitxt_worker": "unitxt_worker",
            "extractor_worker": "extractor_worker",
            "hf_extractor_worker": "hf_extractor_worker",
            "docling_worker": "docling_worker",
            "hf_worker": "hf_worker",
            "composer_worker": "composer_worker",
            "risk_worker": "risk_worker",
            "rag_worker": "rag_worker",
            "factreasoner_worker": "factreasoner_worker",
            "END": END,
        },
    )
    builder.add_edge("unitxt_worker", "orchestrator")
    builder.add_edge("extractor_worker", "orchestrator")
    builder.add_edge("hf_extractor_worker", "orchestrator")
    builder.add_edge("docling_worker", "orchestrator")
    builder.add_edge("hf_worker", "orchestrator")
    builder.add_edge("composer_worker", "orchestrator")
    builder.add_edge("risk_worker", "orchestrator")
    builder.add_edge("rag_worker", "orchestrator")
    builder.add_edge("factreasoner_worker", END)

    return builder.compile()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Extract and fact-check benchmark metadata")
    parser.add_argument("query", help="Benchmark name to process")
    parser.add_argument("--catalog", "-c", help="Path to custom UnitXT catalog")
    parser.add_argument("--output", help="Output directory path for saving results")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode with full tool logging"
    )

    # RAG configuration
    parser.add_argument(
        "--enable-llm-reranking",
        action="store_true",
        default=True,
        help="Use LLM to rerank retrieved documents (default: enabled)",
    )
    parser.add_argument(
        "--disable-llm-reranking",
        dest="enable_llm_reranking",
        action="store_false",
        help="Disable LLM reranking for faster processing",
    )
    parser.add_argument(
        "--enable-hybrid-search",
        action="store_true",
        default=True,
        help="Combine vector search with BM25 (default: enabled)",
    )
    parser.add_argument(
        "--disable-hybrid-search",
        dest="enable_hybrid_search",
        action="store_false",
        help="Disable hybrid search, use vector search only",
    )
    parser.add_argument(
        "--enable-query-expansion",
        action="store_true",
        default=True,
        help="Use LLM to reformulate queries (default: enabled)",
    )
    parser.add_argument(
        "--disable-query-expansion",
        dest="enable_query_expansion",
        action="store_false",
        help="Disable query expansion",
    )

    # Chunking configuration
    parser.add_argument(
        "--parent-chunk-size",
        type=int,
        default=2048,
        help="Parent chunk size in characters (default: 2048)",
    )
    parser.add_argument(
        "--child-chunk-size",
        type=int,
        default=512,
        help="Child chunk size in characters (default: 512)",
    )

    # Processing configuration
    parser.add_argument(
        "--factuality-threshold",
        type=float,
        default=0.8,
        help="Threshold for flagging fields (0.0-1.0, higher=stricter, default=0.8)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of evidence chunks to retrieve per claim (default=3)",
    )

    return parser.parse_args()


def create_initial_state(args: argparse.Namespace, output_manager: OutputManager) -> Dict[str, Any]:
    """Create initial workflow state.

    Args:
        args: Parsed command line arguments.
        output_manager: Output manager instance.

    Returns:
        Initial state dictionary.
    """
    return {
        "query": args.query,
        "catalog_path": args.catalog,
        "output_manager": output_manager,
        "unitxt_json": None,
        "extracted_ids": None,
        "hf_repo": None,
        "hf_json": None,
        "docling_output": None,
        "composed_card": None,
        "risk_enhanced_card": None,
        "completed": [],
        "errors": [],
        "hf_extraction_attempted": False,
        "rag_results": None,
        "factuality_results": None,
        "eee_metadata": None,
    }


def log_execution_summary(state: Dict[str, Any], output_manager: OutputManager) -> None:
    """Log execution summary and results.

    Args:
        state: Final workflow state.
        output_manager: Output manager instance.
    """
    if state.get("errors"):
        logger.error("Errors encountered:")
        for error in state["errors"]:
            logger.error("  - %s", error)
    else:
        # Show final card location (always show this - it's the main output)
        final_card_path = output_manager.benchmarkcard_dir
        logger.info(f"Benchmark card saved to: {final_card_path}")

    # Debug mode: show detailed workflow steps
    logger.debug("Workflow completed")
    logger.debug("Steps: %s", " → ".join(state["completed"]))


def main() -> None:
    """Run the complete benchmark metadata extraction and fact verification pipeline."""
    try:
        # Validate configuration first
        Config.validate_config()

        # Parse arguments and setup
        args = parse_arguments()

        # Apply configuration from CLI arguments
        Config.ENABLE_LLM_RERANKING = args.enable_llm_reranking
        Config.ENABLE_HYBRID_SEARCH = args.enable_hybrid_search
        Config.ENABLE_QUERY_EXPANSION = args.enable_query_expansion
        Config.PARENT_CHUNK_SIZE = args.parent_chunk_size
        Config.CHILD_CHUNK_SIZE = args.child_chunk_size
        Config.DEFAULT_FACTUALITY_THRESHOLD = args.factuality_threshold
        Config.DEFAULT_TOP_K = args.top_k

        # Apply debug mode if flag is set
        if args.debug:
            setup_logging_suppression(debug_mode=True)  # Don't suppress in debug mode
            warnings.resetwarnings()  # Reset warning filters
            logging.basicConfig(level=logging.DEBUG, force=True)
            logger.info("Debug mode enabled - showing full logging output")
            logger.debug("Configuration: LLM rerank=%s, hybrid=%s, query expand=%s, chunks=%d/%d, threshold=%.2f, top-k=%d",
                        args.enable_llm_reranking, args.enable_hybrid_search, args.enable_query_expansion,
                        args.parent_chunk_size, args.child_chunk_size, args.factuality_threshold, args.top_k)

        logger.info("Models — composer: %s | light: %s | factreasoner: %s",
                    Config.COMPOSER_MODEL, Config.LIGHT_MODEL, Config.FACTREASONER_MODEL)
        logger.debug("Starting metadata extraction for: '%s'", args.query)
        if args.catalog:
            logger.debug("Using custom catalog: %s", args.catalog)

        # Create output manager for organized results
        output_manager = OutputManager(args.query, args.output)

        # Initialize workflow state
        initial_state = create_initial_state(args, output_manager)

        # Execute workflow
        workflow = build_workflow()
        state = workflow.invoke(initial_state)

        # Log execution summary and results
        log_execution_summary(state, output_manager)

    except KeyboardInterrupt:
        logger.error("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error in main pipeline: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
