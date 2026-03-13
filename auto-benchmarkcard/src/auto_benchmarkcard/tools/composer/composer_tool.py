"""Benchmark card composition tool using LLM-based synthesis.

This module provides functionality to compose structured benchmark cards
from heterogeneous metadata sources using large language models. It combines
data from UnitXT, HuggingFace, academic papers, and other sources into
standardized benchmark documentation.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Suppress noisy logging from external libraries
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from pydantic import BaseModel, Field

# use the shared llm instance
from auto_benchmarkcard.config import LLM, Config, get_light_llm_handler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section-specific extraction prompts — one per section
# ---------------------------------------------------------------------------
SECTION_EXTRACTION_PROMPTS: Dict[str, str] = {
    "benchmark_details": (
        "Extract key facts from the sources for these fields:\n"
        "- name: What is the official benchmark name? Include any acronym expansion.\n"
        "- overview: What does the benchmark measure? How many tasks/sub-datasets does it contain? "
        "What makes it distinctive? (2-3 key facts)\n"
        "- data_type: What is the primary data modality (text, image, audio, multimodal, tabular)?\n"
        "- domains: What RESEARCH domains does the benchmark target? "
        "(e.g., 'natural language inference', 'sentiment analysis' — NOT data sources like 'Wikipedia' or 'news')\n"
        "- languages: What languages are supported? Use full names (e.g., 'English' not 'en').\n"
        "- similar_benchmarks: What other benchmarks are explicitly compared to or cited as related? "
        "List each by name.\n"
        "- resources: What specific URLs are mentioned? (paper links, homepage, leaderboard, GitHub, HuggingFace)\n"
    ),
    "purpose_and_intended_users": (
        "Extract key facts from the sources for these fields:\n"
        "- goal: What is the primary research objective? What capability or behavior does the benchmark aim to measure?\n"
        "- audience: Who are the intended users? (e.g., NLP researchers, model developers, industry practitioners)\n"
        "- tasks: List EACH specific evaluation task or sub-task by name. "
        "For multi-task benchmarks, list every sub-task individually.\n"
        "- limitations: What limitations, biases, or constraints are explicitly mentioned? "
        "Include language restrictions, domain gaps, task format limitations.\n"
        "- out_of_scope_uses: What use cases does the benchmark explicitly NOT support? "
        "What should it NOT be used for?\n"
    ),
    "data": (
        "Extract key facts from the sources for these fields:\n"
        "- source: List EACH sub-dataset or data source separately with its origin. "
        "For multi-task benchmarks, name each task and where its data comes from "
        "(e.g., 'CoLA: acceptability judgments from linguistics publications', "
        "'SST-2: movie review sentences from Rotten Tomatoes'). "
        "How was each collected (crowdsourced, scraped, curated)?\n"
        "- size: How many total examples? Break down by sub-task if available. "
        "Include train/dev/test splits if mentioned. "
        "Prefer example counts over disk size.\n"
        "- format: What is the data structure? (e.g., 'sentence pairs with labels', "
        "'question-passage pairs'). What file format (JSON, CSV, parquet)?\n"
        "- annotation: How was labeling done for each task? Who annotated (crowdworkers, experts, automatic)? "
        "What quality control measures? Inter-annotator agreement numbers?\n"
    ),
    "methodology": (
        "Extract key facts from the sources for these fields:\n"
        "- methods: How are models evaluated? (zero-shot, few-shot, fine-tuning, submission-based). "
        "Is there a leaderboard? How are submissions handled?\n"
        "- metrics: List EACH metric by name (e.g., accuracy, F1, Matthews correlation, Spearman). "
        "Note which metric is used for which task if specified.\n"
        "- calculation: How is the overall score computed from individual task scores? "
        "Any weighting, averaging, or normalization?\n"
        "- interpretation: What score ranges are meaningful? What constitutes strong vs. weak performance? "
        "Do NOT mix in human baseline numbers here — those go in baseline_results.\n"
        "- baseline_results: What specific numerical results are reported for baselines or human performance? "
        "Include model names and their scores (e.g., 'BERT: 80.5 accuracy', 'Human: 87.1 F1'). "
        "Only include numbers explicitly stated in the sources.\n"
        "- validation: What quality assurance measures exist? (diagnostic sets, inter-annotator agreement, "
        "reproducibility checks)\n"
    ),
    "ethical_and_legal_considerations": (
        "Extract key facts from the sources for these fields:\n"
        "- privacy_and_anonymity: Does the data contain personal information? "
        "What anonymization was applied? Is data from public or private sources?\n"
        "- data_licensing: What specific license applies? (e.g., CC BY-SA 4.0, MIT, Apache 2.0). "
        "Are there usage restrictions?\n"
        "- consent_procedures: How were data subjects or annotators consented? "
        "Were crowdworkers compensated? What platform was used (MTurk, etc.)?\n"
        "- compliance_with_regulations: Any mention of IRB approval, GDPR compliance, "
        "ethical review board, or institutional oversight?\n"
    ),
}

_EXTRACTOR_SYSTEM = (
    "You are a fact extraction assistant. Your job is to read the provided sources "
    "about an AI benchmark and extract ONLY the key facts relevant to the requested fields.\n\n"
    "RULES:\n"
    "1. Return facts as short bullet points grouped by field name.\n"
    "2. Only include what the sources EXPLICITLY state. Do not infer or invent.\n"
    "3. If a source says nothing about a field, write: '- No information found'\n"
    "4. Keep each bullet point to ONE fact, ONE sentence.\n"
    "5. Prefer specific numbers, names, and quotes over vague summaries.\n"
    "6. Do NOT repeat the same fact under multiple fields.\n"
)


def extract_section_facts(
    section_name: str,
    paper_content: str,
    hf_metadata: Optional[Dict[str, Any]],
    unitxt_metadata: Optional[Dict[str, Any]],
    extracted_ids: Optional[Dict[str, Any]] = None,
    query: str = "",
    eee_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Use the light model to extract key facts for a section before composition.

    Args:
        section_name: Name of the benchmark card section.
        paper_content: Retrieved paper chunks (already filtered by RAG-lite).
        hf_metadata: HuggingFace metadata dict.
        unitxt_metadata: UnitXT catalog metadata dict.
        extracted_ids: Optional extracted identifiers.
        query: Benchmark name.
        eee_metadata: Optional EEE evaluation metadata (metrics, scores, etc.).

    Returns:
        Extracted facts as a formatted string of bullet points per field.
    """
    extraction_prompt = SECTION_EXTRACTION_PROMPTS.get(section_name)
    if not extraction_prompt:
        logger.warning("No extraction prompt for section %s, skipping extraction", section_name)
        return ""

    # Format sources compactly
    hf_text = "Not available"
    if hf_metadata:
        hf_compact = _compact_hf_metadata(hf_metadata) if isinstance(hf_metadata, dict) else {}
        hf_text = json.dumps(hf_compact, indent=2) if hf_compact else "Not available"

    ids_text = json.dumps(extracted_ids, indent=2) if extracted_ids else "Not available"

    # Build sources list dynamically — only include available sources
    source_parts = [f"1. PAPER CONTENT:\n{paper_content}"]
    source_parts.append(f"2. HuggingFace Dataset:\n{hf_text}")

    source_idx = 3
    if unitxt_metadata:
        unitxt_text = json.dumps(unitxt_metadata, indent=2)
        source_parts.append(f"{source_idx}. UnitXT Catalog:\n{unitxt_text}")
        source_idx += 1

    source_parts.append(f"{source_idx}. Extracted IDs:\n{ids_text}")
    source_idx += 1

    if eee_metadata:
        eee_compact = _compact_eee_metadata(eee_metadata)
        eee_text = json.dumps(eee_compact, indent=2) if eee_compact else "Not available"
        source_parts.append(f"{source_idx}. Every Eval Ever (EEE) Evaluation Data:\n{eee_text}")

    sources = "\n\n".join(source_parts)

    user_message = (
        f"Benchmark: {query}\n\n"
        f"SOURCES:\n\n"
        f"{sources}\n\n"
        f"---\n\n"
        f"{extraction_prompt}\n"
        f"Return facts as short bullet points per field. Only include what the sources explicitly state."
    )

    prompt = f"{_EXTRACTOR_SYSTEM}\n\n{user_message}"

    try:
        light_llm = get_light_llm_handler()
        facts = light_llm.generate(prompt)
        logger.debug("Extracted facts for %s (%d chars)", section_name, len(facts))
        return facts
    except Exception as e:
        logger.warning("Fact extraction failed for %s: %s — composer will use raw sources", section_name, e)
        return ""


def _compact_hf_metadata(hf_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only the fields useful for composition from HF metadata."""
    meta = hf_metadata
    if "tags" not in meta:
        for v in meta.values():
            if isinstance(v, dict) and "tags" in v:
                meta = v
                break

    compact: Dict[str, Any] = {}
    for key in ("id", "tags", "license", "downloads", "likes"):
        if key in meta:
            compact[key] = meta[key]

    if "card_data" in meta and meta["card_data"]:
        compact["card_data"] = meta["card_data"]

    if "readme_markdown" in meta and meta["readme_markdown"]:
        compact["readme_excerpt"] = meta["readme_markdown"][:1500]

    return compact


def _compact_eee_metadata(eee_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only the fields useful for composition from EEE metadata."""
    compact: Dict[str, Any] = {}

    if eee_metadata.get("benchmark_name"):
        compact["benchmark_name"] = eee_metadata["benchmark_name"]
    if eee_metadata.get("eval_library"):
        compact["eval_library"] = eee_metadata["eval_library"]
    if eee_metadata.get("source_urls"):
        compact["source_urls"] = eee_metadata["source_urls"][:5]

    # Include metrics info
    metrics = eee_metadata.get("metrics", {})
    if metrics:
        compact["metrics"] = {
            k: {
                "description": v.get("evaluation_description", ""),
                "lower_is_better": v.get("lower_is_better", False),
                "score_type": v.get("score_type", ""),
            }
            for k, v in list(metrics.items())[:10]
        }

    # Include evaluation summary (top performers, stats)
    eval_summary = eee_metadata.get("evaluation_summary", {})
    if eval_summary:
        compact["evaluation_summary"] = {
            "total_models": eval_summary.get("total_models_evaluated", 0),
            "primary_metric": eval_summary.get("primary_metric", ""),
            "score_statistics": eval_summary.get("score_statistics", {}),
            "top_performers": eval_summary.get("top_performers", [])[:5],
        }

    return compact


# schema for the benchmark card
class BenchmarkDetails(BaseModel):
    """Basic identifying information about a benchmark.

    Attributes:
        name: The official name of the benchmark as it appears in literature.
        overview: A comprehensive 2-3 sentence description explaining what the benchmark measures.
        data_type: The primary data modality (e.g., text, image, audio, multimodal, tabular).
        domains: Specific application domains or subject areas.
        languages: All languages supported in the dataset using full language names.
        similar_benchmarks: Names of closely related or comparable benchmarks.
        resources: URLs to official papers, datasets, leaderboards, and documentation.
    """

    name: str = Field(
        ...,
        description="The official name of the benchmark as it appears in literature",
    )
    overview: str = Field(
        ...,
        description="A comprehensive 2-3 sentence description explaining what the benchmark measures, its key characteristics, and its significance in the field",
    )
    data_type: str = Field(
        ...,
        description="The primary data modality (e.g., text, image, audio, multimodal, tabular)",
    )
    domains: List[str] = Field(
        ...,
        description="Specific application domains or subject areas (e.g., medical, legal, scientific, conversational AI)",
    )
    languages: List[str] = Field(
        ...,
        description="All languages supported in the dataset using full language names (e.g., 'English', 'Chinese', 'Spanish', 'Multilingual')",
    )
    similar_benchmarks: List[str] = Field(
        ...,
        description="Names of closely related or comparable benchmarks that measure similar capabilities",
    )
    resources: List[str] = Field(
        ...,
        description="URLs to official papers, datasets, leaderboards, and documentation",
    )
    provenance: Optional[Dict[str, Dict[str, str]]] = Field(
        default=None,
        description="Source evidence mapping: field_name -> {source, evidence}",
    )


class PurposeAndIntendedUsers(BaseModel):
    """Purpose, target users, and use case information.

    Attributes:
        goal: The primary objective and research question this benchmark addresses.
        audience: Target user groups for the benchmark.
        tasks: Specific evaluation tasks or subtasks the benchmark covers.
        limitations: Known limitations, biases, or constraints of the benchmark.
        out_of_scope_uses: Explicit examples of inappropriate or unsupported use cases.
    """

    goal: str = Field(
        ...,
        description="The primary objective and research question this benchmark addresses, including what capabilities or behaviors it aims to measure",
    )
    audience: List[str] = Field(
        ...,
        description="Target user groups (e.g., 'AI researchers', 'model developers', 'safety evaluators', 'industry practitioners')",
    )
    tasks: List[str] = Field(
        ...,
        description="Specific evaluation tasks or subtasks the benchmark covers (e.g., 'question answering', 'code generation', 'factual accuracy')",
    )
    limitations: str = Field(
        ...,
        description="Known limitations, biases, or constraints of the benchmark that users should be aware of",
    )
    out_of_scope_uses: List[str] = Field(
        ...,
        description="Explicit examples of inappropriate or unsupported use cases for this benchmark",
    )
    provenance: Optional[Dict[str, Dict[str, str]]] = Field(
        default=None,
        description="Source evidence mapping: field_name -> {source, evidence}",
    )


class DataInfo(BaseModel):
    """Information about dataset composition and collection.

    Attributes:
        source: Detailed information about data origins and collection methods.
        size: Dataset size with specific numbers.
        format: Data structure, file formats, and organization.
        annotation: Annotation methodology and quality control measures.
    """

    source: str = Field(
        ...,
        description="Detailed information about data origins, collection methods, and any preprocessing steps applied",
    )
    size: str = Field(
        ...,
        description="Dataset size. Prefer number of examples from paper (e.g., '817 questions'). "
        "If only disk size from HuggingFace is available, use that (e.g., '1.24 GB')",
    )
    format: str = Field(
        ...,
        description="The data format as described in the paper or README "
        "(e.g., 'JSON with question-answer pairs'). If only the HuggingFace hosting "
        "format is known, note it as such (e.g., 'parquet (HuggingFace hosting format)')",
    )
    annotation: str = Field(
        ...,
        description="Annotation methodology, quality control measures, inter-annotator agreement, and any human involvement in labeling",
    )
    provenance: Optional[Dict[str, Dict[str, str]]] = Field(
        default=None,
        description="Source evidence mapping: field_name -> {source, evidence}",
    )


class Methodology(BaseModel):
    """Evaluation methodology and metric specifications.

    Attributes:
        methods: Evaluation approaches and techniques applied.
        metrics: Specific quantitative metrics used.
        calculation: Detailed explanation of metric computation.
        interpretation: Guidelines for interpreting scores.
        baseline_results: Performance of established models or baselines.
        validation: Quality assurance measures and validation procedures.
    """

    methods: List[str] = Field(
        ...,
        description="Evaluation approaches and techniques applied within the benchmark (e.g., 'zero-shot evaluation', 'few-shot prompting', 'fine-tuning')",
    )
    metrics: List[str] = Field(
        ...,
        description="Specific quantitative metrics used (e.g., 'accuracy', 'F1-score', 'BLEU', 'exact match')",
    )
    calculation: str = Field(
        ...,
        description="Detailed explanation of how metrics are computed, including any normalization or aggregation methods",
    )
    interpretation: str = Field(
        ...,
        description="Guidelines for interpreting scores, including score ranges, what constitutes good performance, and any caveats",
    )
    baseline_results: str = Field(
        ...,
        description="Performance of established models or baselines, with specific numbers and context for comparison",
    )
    validation: str = Field(
        ...,
        description="Quality assurance measures, validation procedures, and steps taken to ensure reliable and reproducible evaluations",
    )
    provenance: Optional[Dict[str, Dict[str, str]]] = Field(
        default=None,
        description="Source evidence mapping: field_name -> {source, evidence}",
    )


class EthicalAndLegalConsiderations(BaseModel):
    """Ethical and legal aspects of the benchmark.

    Attributes:
        privacy_and_anonymity: Data protection and anonymization measures.
        data_licensing: License terms and usage restrictions.
        consent_procedures: Informed consent processes and participant rights.
        compliance_with_regulations: Adherence to relevant regulations and ethical reviews.
    """

    privacy_and_anonymity: str = Field(
        ...,
        description="Data protection measures, anonymization techniques, and handling of personally identifiable information",
    )
    data_licensing: str = Field(
        ...,
        description="Specific license terms, usage restrictions, and redistribution permissions",
    )
    consent_procedures: str = Field(
        ...,
        description="Details of informed consent processes, participant rights, and withdrawal procedures",
    )
    compliance_with_regulations: str = Field(
        ...,
        description="Adherence to relevant regulations (GDPR, IRB approval, etc.) and ethical review processes",
    )
    provenance: Optional[Dict[str, Dict[str, str]]] = Field(
        default=None,
        description="Source evidence mapping: field_name -> {source, evidence}",
    )


class BenchmarkCard(BaseModel):
    """Complete benchmark card structure.

    Attributes:
        benchmark_details: Basic identifying information.
        purpose_and_intended_users: Purpose and target user information.
        data: Dataset composition and collection details.
        methodology: Evaluation methodology and metrics.
        ethical_and_legal_considerations: Ethical and legal aspects.
    """

    benchmark_details: BenchmarkDetails
    purpose_and_intended_users: PurposeAndIntendedUsers
    data: DataInfo
    methodology: Methodology
    ethical_and_legal_considerations: EthicalAndLegalConsiderations


def extract_provenance(section_data: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Extract provenance from section data, returning clean data and provenance separately.

    Args:
        section_data: Section dictionary that may contain a 'provenance' field.

    Returns:
        Tuple of (clean_section_data without provenance, provenance_data).
    """
    # Make a copy to avoid mutating the original
    clean_data = dict(section_data)
    provenance = clean_data.pop("provenance", None) or {}
    return clean_data, provenance


@tool("compose_benchmark_card")
def compose_benchmark_card(
    unitxt_metadata: Optional[Dict[str, Any]] = None,
    hf_metadata: Optional[Dict[str, Any]] = None,
    extracted_ids: Optional[Dict[str, Any]] = None,
    docling_output: Optional[Dict[str, Any]] = None,
    query: str = "",
    eee_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compose a benchmark card from all the metadata we collected.

    Args:
        unitxt_metadata: Optional metadata from UnitXT catalog.
        hf_metadata: Optional metadata from HuggingFace.
        extracted_ids: Optional extracted identifier information.
        docling_output: Optional extracted paper content.
        query: Original query string for context.
        eee_metadata: Optional metadata from Every Eval Ever (EEE).

    Returns:
        Dictionary containing composed benchmark card and composition metadata.
    """

    logger.debug(f"Composing benchmark card for: {query}")

    # Log available data sources
    data_sources = []
    if unitxt_metadata:
        data_sources.append("UnitXT")
    if hf_metadata:
        data_sources.append("HuggingFace")
    if extracted_ids:
        data_sources.append("Extracted IDs")
    if docling_output and docling_output.get("success"):
        data_sources.append("Academic Paper")
    if eee_metadata:
        data_sources.append("EEE")

    logger.debug(f"Available data sources: {', '.join(data_sources)}")

    # Initialize paper retriever for RAG-lite (index once, retrieve per section)
    paper_retriever = None
    if docling_output and docling_output.get("success"):
        try:
            paper_text = docling_output.get("filtered_text", "")
            if paper_text:
                logger.debug("Initializing paper retriever for RAG-lite")
                # Initialize embeddings based on config
                if Config.DEFAULT_EMBEDDING_MODEL == "bge-large":
                    embeddings = HuggingFaceEmbeddings(
                        model_name="BAAI/bge-large-en-v1.5",
                        model_kwargs={"device": "cpu"},
                        encode_kwargs={"normalize_embeddings": True},
                    )
                elif Config.DEFAULT_EMBEDDING_MODEL == "e5-large":
                    embeddings = HuggingFaceEmbeddings(
                        model_name="intfloat/e5-large-v2",
                        model_kwargs={"device": "cpu"},
                        encode_kwargs={"normalize_embeddings": True},
                    )
                else:  # minilm fallback
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )

                # Chunk paper for retrieval (smaller chunks for better precision)
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", ". ", " "]
                )
                chunks = splitter.split_text(paper_text)

                # Create documents
                documents = [
                    Document(page_content=chunk, metadata={"chunk_idx": i})
                    for i, chunk in enumerate(chunks)
                ]

                # Create vectorstore and retriever
                paper_vectorstore = Chroma.from_documents(documents, embeddings)
                paper_retriever = paper_vectorstore.as_retriever(search_kwargs={"k": 5})
                logger.debug(f"Paper indexed: {len(chunks)} chunks ready for retrieval")
        except Exception as e:
            logger.warning(f"Failed to initialize paper retriever: {e}")
            paper_retriever = None

    # define the sections to generate
    sections = [
        ("benchmark_details", BenchmarkDetails),
        ("purpose_and_intended_users", PurposeAndIntendedUsers),
        ("data", DataInfo),
        ("methodology", Methodology),
        ("ethical_and_legal_considerations", EthicalAndLegalConsiderations),
    ]

    # Section-specific query templates for retrieval
    # Multiple queries per section to improve recall across different paper sections
    section_queries = {
        "benchmark_details": [
            "benchmark name overview introduction contribution",
            "related work similar benchmarks comparison",
            "resources homepage leaderboard repository URL",
        ],
        "data": [
            "dataset collection source corpus sub-task data origin",
            "dataset size examples training test split statistics",
            "annotation crowdsource label annotator agreement quality",
        ],
        "methodology": [
            "evaluation method metrics accuracy F1 score measurement",
            "baseline results performance human comparison model scores",
            "diagnostic analysis validation quality assurance",
        ],
        "purpose_and_intended_users": [
            "goal objective motivation purpose research question",
            "tasks sub-tasks evaluation individual task description",
            "limitations bias constraints scope out-of-scope",
        ],
        "ethical_and_legal_considerations": [
            "ethics privacy anonymity personal information",
            "license consent crowdworker compensation IRB",
        ],
    }

    # Load the gold example for format anchoring
    gold_example_path = Path(__file__).parent / "gold_example.json"
    gold_example: Dict[str, Any] = {}
    try:
        gold_example = json.loads(gold_example_path.read_text())
        logger.debug("Loaded gold example for format anchoring")
    except Exception as e:
        logger.warning("Could not load gold example: %s", e)

    generated_sections = {}
    all_provenance = {}

    for section_name, section_class in sections:
        logger.debug("Generating %s", section_name.replace("_", " ").title())

        # ── Step 0: Retrieve relevant paper chunks (RAG-lite) ──
        # Uses multiple queries per section for broader coverage, deduplicates by content
        paper_content = "Not available"
        if paper_retriever:
            try:
                queries = section_queries.get(section_name, [section_name.replace("_", " ")])
                # Collect unique chunks from all sub-queries
                seen_chunks = set()
                all_chunks = []
                for sq in queries:
                    for chunk in paper_retriever.get_relevant_documents(sq):
                        chunk_key = chunk.page_content[:200]
                        if chunk_key not in seen_chunks:
                            seen_chunks.add(chunk_key)
                            all_chunks.append(chunk)

                if all_chunks:
                    formatted_chunks = []
                    char_budget = 3000
                    chars_used = 0
                    for i, chunk in enumerate(all_chunks, 1):
                        text = chunk.page_content
                        if chars_used + len(text) > char_budget:
                            remaining = char_budget - chars_used
                            if remaining > 100:
                                formatted_chunks.append(f"[Paper Section {i}]\n{text[:remaining]}")
                            break
                        formatted_chunks.append(f"[Paper Section {i}]\n{text}")
                        chars_used += len(text)
                    paper_content = "\n\n".join(formatted_chunks)
                    logger.debug(f"Retrieved {len(all_chunks)} unique paper chunks for {section_name} (from {len(queries)} queries)")
                else:
                    logger.debug(f"No relevant chunks found for {section_name}, using fallback")
                    if docling_output and docling_output.get("filtered_text"):
                        paper_content = docling_output.get("filtered_text", "")[:3000]
            except Exception as e:
                logger.warning(f"Paper retrieval failed for {section_name}: {e}")
                if docling_output and docling_output.get("filtered_text"):
                    paper_content = docling_output.get("filtered_text", "")[:3000]
        elif docling_output and docling_output.get("success"):
            paper_content = docling_output.get("filtered_text", "Not available")[:3000]

        # ── Step 1: EXTRACT — light model extracts key facts ──
        extracted_facts = extract_section_facts(
            section_name=section_name,
            paper_content=paper_content,
            hf_metadata=hf_metadata,
            unitxt_metadata=unitxt_metadata,
            extracted_ids=extracted_ids,
            query=query,
            eee_metadata=eee_metadata,
        )

        # ── Step 2: COMPOSE — heavy model formats facts into schema ──
        # Build the gold example snippet for this section
        # Escape curly braces so ChatPromptTemplate doesn't treat them as variables
        gold_snippet = ""
        if gold_example and section_name in gold_example:
            gold_json = json.dumps(gold_example[section_name], indent=2)
            gold_json_escaped = gold_json.replace("{", "{{").replace("}", "}}")
            gold_snippet = (
                f"\n\nGOLD EXAMPLE (use this as a FORMAT reference — match the style, length, and level of detail):\n"
                f"```json\n{gold_json_escaped}\n```"
            )

        # Choose prompt based on whether extraction succeeded
        if extracted_facts:
            # Extraction succeeded → composer gets compressed facts
            section_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        f"""You are documenting an AI benchmark. Generate the '{section_name}' section.

You are given PRE-EXTRACTED FACTS (bullet points) that have already been filtered from the original sources. Your job is to FORMAT these facts into the required JSON schema — do NOT add information beyond what the facts state.

RULES:
1. Use ONLY the extracted facts below. If a field has no facts, write exactly "Not specified".
2. Write in third person. Describe the benchmark objectively.
3. Do not invent facts, URLs, numbers, or performance scores.
4. Be concise. Match the style and length of the gold example.
5. Each field value should be a clean, well-written summary of the relevant facts — not a dump of all bullets.
{gold_snippet}

PROVENANCE TRACKING (REQUIRED):
For every field you fill in (except "Not specified"), include a provenance entry:
{{{{
  "provenance": {{{{
    "field_name": {{{{
      "source": "paper|huggingface|unitxt|extracted_ids",
      "evidence": "the key fact that supports this field value"
    }}}}
  }}}}
}}}}""",
                    ),
                    (
                        "user",
                        f"""Benchmark: {{query}}

EXTRACTED FACTS:
{{extracted_facts}}

Generate the {section_name} section by formatting these facts into the required schema.""",
                    ),
                ]
            )

            chain = section_prompt | LLM.with_structured_output(section_class)

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    section_result = chain.invoke(
                        {
                            "query": query,
                            "extracted_facts": extracted_facts,
                        }
                    )
                    section_dict = section_result.model_dump()
                    clean_section, section_provenance = extract_provenance(section_dict)
                    generated_sections[section_name] = clean_section
                    if section_provenance:
                        all_provenance[section_name] = section_provenance
                    logger.debug("%s completed (extract→compose)", section_name.replace("_", " ").title())
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning("Failed to compose %s (attempt %d/%d): %s", section_name, attempt + 1, max_retries, e)
                    else:
                        logger.error("Failed to compose %s after %d attempts: %s", section_name, max_retries, e)
                        raise
        else:
            # Extraction failed → fallback to direct composition with raw sources
            logger.info("Extraction failed for %s, falling back to direct composition", section_name)

            hf_formatted = "Not available"
            if hf_metadata:
                if isinstance(hf_metadata, dict):
                    hf_compact = _compact_hf_metadata(hf_metadata)
                    hf_formatted = json.dumps(hf_compact, indent=2) if hf_compact else "Not available"
                else:
                    hf_formatted = str(hf_metadata)[:2000]

            extracted_formatted = json.dumps(extracted_ids, indent=2) if extracted_ids else "Not available"

            # Build fallback sources and invoke variables dynamically
            fallback_invoke_vars = {
                "query": query,
                "paper_content": paper_content,
                "hf_metadata": hf_formatted,
                "extracted_ids": extracted_formatted,
            }

            # Build numbered source list for the prompt
            fb_sources = [
                "1. PAPER CONTENT:\n{paper_content}",
                "2. HuggingFace Dataset:\n{hf_metadata}",
            ]
            fb_idx = 3
            if unitxt_metadata:
                unitxt_formatted = json.dumps(unitxt_metadata, indent=2)
                fb_sources.append(f"{fb_idx}. UnitXT Catalog:\n" + "{unitxt_metadata}")
                fallback_invoke_vars["unitxt_metadata"] = unitxt_formatted
                fb_idx += 1
            fb_sources.append(f"{fb_idx}. Extracted IDs:\n" + "{extracted_ids}")
            fb_idx += 1
            if eee_metadata:
                eee_compact = _compact_eee_metadata(eee_metadata)
                eee_formatted = json.dumps(eee_compact, indent=2)
                fb_sources.append(f"{fb_idx}. Every Eval Ever (EEE) Evaluation Data:\n" + "{eee_metadata}")
                fallback_invoke_vars["eee_metadata"] = eee_formatted

            fallback_sources_block = "\n\n".join(fb_sources)

            # Determine valid source names for provenance
            source_names = "paper|huggingface|extracted_ids"
            if unitxt_metadata:
                source_names += "|unitxt"
            if eee_metadata:
                source_names += "|eee"

            section_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are documenting an AI benchmark. Generate the '" + section_name + "' section.\n\n"
                        "RULES:\n"
                        "1. Use ONLY the provided metadata sources. If information is not found, write exactly \"Not specified\".\n"
                        "2. Write in third person. Describe the benchmark objectively.\n"
                        "3. Do not invent facts, URLs, numbers, or performance scores. Only include what the sources explicitly state.\n"
                        "4. Be concise. Match the style and length of the gold example.\n"
                        + gold_snippet + "\n\n"
                        "PROVENANCE TRACKING (REQUIRED):\n"
                        "For every field you fill in (except \"Not specified\"), include a provenance entry:\n"
                        "{{\n"
                        '  "provenance": {{\n'
                        '    "field_name": {{\n'
                        '      "source": "' + source_names + '",\n'
                        '      "evidence": "exact quote or description from the source"\n'
                        "    }}\n"
                        "  }}\n"
                        "}}",
                    ),
                    (
                        "user",
                        "Benchmark: {query}\n\n"
                        "METADATA SOURCES:\n\n"
                        + fallback_sources_block + "\n\n"
                        "Generate the " + section_name + " section using ONLY the sources above.",
                    ),
                ]
            )

            chain = section_prompt | LLM.with_structured_output(section_class)

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    section_result = chain.invoke(fallback_invoke_vars)
                    section_dict = section_result.model_dump()
                    clean_section, section_provenance = extract_provenance(section_dict)
                    generated_sections[section_name] = clean_section
                    if section_provenance:
                        all_provenance[section_name] = section_provenance
                    logger.debug("%s completed (direct)", section_name.replace("_", " ").title())
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning("Failed to generate %s (attempt %d/%d): %s", section_name, attempt + 1, max_retries, e)
                    else:
                        logger.error("Failed to generate %s after %d attempts: %s", section_name, max_retries, e)
                        raise

    # combine all sections into final benchmark card
    logger.debug("Combining all sections into final benchmark card")

    try:
        final_card = BenchmarkCard(
            benchmark_details=BenchmarkDetails(**generated_sections["benchmark_details"]),
            purpose_and_intended_users=PurposeAndIntendedUsers(
                **generated_sections["purpose_and_intended_users"]
            ),
            data=DataInfo(**generated_sections["data"]),
            methodology=Methodology(**generated_sections["methodology"]),
            ethical_and_legal_considerations=EthicalAndLegalConsiderations(
                **generated_sections["ethical_and_legal_considerations"]
            ),
        )

        logger.debug("Final benchmark card assembled successfully")

    except Exception as e:
        logger.error("Failed to assemble final benchmark card: %s", e)
        raise

    # add metadata about the composition process
    # Exclude provenance from benchmark_card output (it's saved separately)
    benchmark_card_dict = final_card.model_dump(exclude_none=True)
    # Double-check: remove any remaining provenance fields from nested sections
    for section_key in benchmark_card_dict:
        if isinstance(benchmark_card_dict[section_key], dict) and "provenance" in benchmark_card_dict[section_key]:
            del benchmark_card_dict[section_key]["provenance"]

    return {
        "benchmark_card": benchmark_card_dict,
        "provenance": all_provenance if all_provenance else None,
        "composition_metadata": {
            "sources_used": {
                "unitxt": bool(unitxt_metadata),
                "huggingface": bool(hf_metadata),
                "extracted_ids": bool(extracted_ids),
                "docling": bool(docling_output),
                "eee": bool(eee_metadata),
            },
            "query": query,
            "composition_timestamp": datetime.now().isoformat(),
            "generation_method": "extract_then_compose",
            "model_used": LLM.model_name,
        },
    }
