"""
ingest.py – Thesis-aware ingestion for Palash Mishra's Master's Thesis:
"A Generative Artificial Intelligence Approach for Enabling Automatic
Speech Recognition: Seamless Issue Logging, Accent Acquisition, Noise
Handling and Post-Processing"
Bauhaus-Universität Weimar, 2025

Place thesis PDF at: backend/data/thesis.pdf
Run standalone: python ingest.py  (for testing)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    chunk_id: str
    text: str
    page: int
    chapter: str          # e.g. "Chapter 3 Methodology"
    chapter_index: int
    section: str          # e.g. "3.2 Dataset Preprocessing"
    section_index: str    # e.g. "3.2"
    graph_nodes: list[str] = field(default_factory=list)
    embedding: Any = None


@dataclass
class AppState:
    """Singleton holding graph, chunks and vector store after ingestion."""
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    chunks: list[TextChunk] = field(default_factory=list)
    vector_store: Any = None          # VectorStore instance
    embedding_model: Any = None       # sentence_transformers model
    chapter_tree: list[dict] = field(default_factory=list)


APP_STATE = AppState()


# ─────────────────────────────────────────────────────────────────────────────
# Thesis-specific knowledge dictionaries
# Derived directly from reading the thesis ToC, figures, tables, equations.
# ─────────────────────────────────────────────────────────────────────────────

THESIS_CHAPTERS = {
    1: "Introduction",
    2: "Background",
    3: "Methodology",
    4: "Results",
    5: "Future Work",
    6: "Summary",
}

# All real sections from the thesis Table of Contents (page 9-11)
THESIS_SECTIONS: dict[str, str] = {
    "1.1": "Research Motivation and Questions",
    "1.2": "Contributions and Structure",
    "2.1": "Introduction to Speech Technologies",
    "2.1.1": "Overview of TTS and STT Systems",
    "2.1.2": "Historical Evolution and Advancements",
    "2.1.3": "Role of Speech Technologies in Domain-Specific Applications",
    "2.1.4": "Challenges in Prosody, Emotional Context, and Naturalness",
    "2.1.5": "Selection of TTS Models",
    "2.2": "Automatic Speech Recognition (ASR)",
    "2.2.1": "Core Components of ASR Systems",
    "2.2.2": "Evolution of ASR Models",
    "2.2.3": "Challenges in ASR",
    "2.2.4": "Phoneme-Level ASR with the Allosaurus Library",
    "2.3": "Natural Language Processing (NLP) for Speech Systems",
    "2.3.1": "Role of NLP in Enhancing ASR and TTS",
    "2.3.2": "Intent Recognition and Entity Extraction in Domain-Specific Contexts",
    "2.4": "Generative AI in Automatic Speech Recognition Transcription",
    "2.4.1": "Instruction-Tuned Language Models: The Case of Qwen Instruct",
    "2.5": "Domain-Specific Speech Transcription",
    "2.5.1": "Importance of Domain-Specific Datasets",
    "2.5.2": "Challenges in Collecting and Annotating Domain-Specific Data",
    "2.5.3": "Text Preprocessing for Speech Systems",
    "2.5.4": "Fine-Tuning ASR Models for Specific Domains",
    "2.6": "Error Handling in Real-Time Speech Transcription",
    "2.6.1": "Addressing Acoustic Noise and Interruptions",
    "2.6.2": "Error Correction and Filtering Mechanisms",
    "2.6.3": "Evaluation Metrics for Speech Systems",
    "3.1": "Overview",
    "3.2": "Dataset Preprocessing",
    "3.2.1": "Rule-Based Hardcoded Preprocessing",
    "3.2.2": "LLM-Based Preprocessing Methodology",
    "3.3": "Audio Synthesis: Text-to-Speech (TTS) Models",
    "3.3.1": "Input Prompt Processing and Model-Specific Strategies",
    "3.4": "Evaluation Techniques for TTS",
    "3.4.1": "Challenges in Human-Based Evaluation",
    "3.4.2": "Metric 1: Forced Alignment Evaluation",
    "3.4.3": "Metric 2: Levenshtein-Based Phoneme Similarity",
    "3.4.4": "Metric 3: LLM-Based Semantic Comparison",
    "3.5": "Fine-Tuning Process for Domain-Specific ASR using Whisper",
    "3.5.1": "Overview of Fine-Tuning Architecture",
    "3.5.2": "Alignment and Dataset Generation Workflow",
    "3.6": "Post-Processing Audio Transcriptions using Finetuned Whisper and LLMs",
    "3.6.1": "Overview of the Methodology",
    "3.6.2": "Preprocessing and Fine-Tuning for Domain Specificity",
    "3.6.3": "LLM-based Post-Processing Pipeline",
    "3.6.4": "Conceptual Insights and Implications",
    "3.7": "Summary",
    "4.1": "Preprocessing Results and Analysis",
    "4.1.1": "Overview",
    "4.1.2": "Evaluation Metric",
    "4.1.3": "Corpus-Level Similarity",
    "4.1.4": "Transformation Examples",
    "4.1.5": "Comparison and Evaluation",
    "4.2": "Audio Generation Results and Analysis",
    "4.2.1": "Human-Observed Characteristics of Synthesized Speech",
    "4.3": "Integrated Evaluation of Synthesized Speech Quality",
    "4.3.1": "Metric 1: Forced Alignment Analysis Results",
    "4.3.2": "Metric 2: Phonetic Similarity Evaluation Results",
    "4.3.3": "Metric 3: LLM-Based Verification Results",
    "4.3.4": "Model Comparison and Composite Scoring",
    "4.4": "Fine-Tuning Whisper ASR on Symbolic Technical Speech",
    "4.4.1": "Dataset Composition and Input Design",
    "4.4.2": "Training Configuration and Objective",
    "4.4.3": "Qualitative Improvements and Model Behavior",
    "4.5": "LLM-Assisted Refinement of Fine-Tuned ASR Output",
    "4.5.1": "Pipeline Architecture",
    "4.5.2": "Qualitative Examples and LLM Impact",
    "4.6": "Results of Finetuned Model on Unknown Test Data",
    "4.6.1": "Example Predictions",
    "4.6.2": "Metric Distributions",
    "4.7": "Summary",
    "5.1": "Preprocessing",
    "5.2": "Audio Generation",
    "5.3": "Integrated Evaluation of Synthesized Speech Quality",
    "5.4": "Fine-Tuning Whisper ASR on Symbolic Technical Speech",
    "5.5": "LLM-Assisted Refinement of Fine-Tuned ASR Output",
    "5.6": "Summary",
    "6": "Summary",
}

# ── Concept nodes ────────────────────────────────────────────────────────────
# Derived from reading the full thesis. Each entry:
#   keyword_variants → (node_id, label, text_snippet)
THESIS_CONCEPTS: list[tuple[list[str], str, str, str]] = [
    # (keyword_list, node_id, label, snippet)
    (["automatic speech recognition", "asr", "speech recognition"],
     "concept_asr", "ASR (Automatic Speech Recognition)",
     "ASR systems transcribe spoken language into text through interconnected components "
     "addressing noise, speaker variability, and linguistic complexity."),

    (["text-to-speech", "tts", "speech synthesis", "speech synthesiz"],
     "concept_tts", "TTS (Text-to-Speech)",
     "TTS systems convert written text into natural-sounding speech using deep learning "
     "architectures including transformers, autoregressive pipelines, and neural vocoders."),

    (["speech-to-text", "stt", "transcription"],
     "concept_stt", "STT (Speech-to-Text)",
     "STT systems translate spoken language into text, enabling hands-free control in "
     "customer service, industrial automation, and automotive manufacturing."),

    (["large language model", "llm", "language model", "generative ai", "generative artificial intelligence"],
     "concept_llm", "LLM (Large Language Model)",
     "Instruction-tuned LLMs such as Qwen2-72B-Instruct and Gemma-2-27B are used for "
     "context-sensitive preprocessing and post-processing of ASR transcriptions."),

    (["whisper", "openai whisper"],
     "concept_whisper", "Whisper (OpenAI ASR Model)",
     "OpenAI's Whisper is a transformer-based ASR model fine-tuned in this thesis on "
     "domain-specific symbolic technical speech from automotive manufacturing."),

    (["fine-tuning", "fine tuning", "finetuning", "fine-tuned", "finetuned"],
     "concept_finetuning", "Fine-Tuning",
     "Whisper is fine-tuned on a curated dataset of synthetic and real audio using "
     "LoRA-style parameter-efficient techniques with CTC and cross-entropy loss."),

    (["preprocessing", "pre-processing", "text normalization", "text preprocessing"],
     "concept_preprocessing", "Text Preprocessing",
     "Two preprocessing pipelines are employed: rule-based hardcoded normalization and "
     "LLM-based context-sensitive normalization for TTS-ready text."),

    (["rule-based preprocessing", "rule based preprocessing", "hardcoded preprocessing"],
     "concept_rule_based_preprocessing", "Rule-Based Preprocessing",
     "Deterministic transformation system applying symbol replacement, number expansion, "
     "CamelCase splitting, acronym handling, and whitespace normalization."),

    (["llm-based preprocessing", "llm based preprocessing", "llm preprocessing"],
     "concept_llm_preprocessing", "LLM-Based Preprocessing",
     "Instruction-tuned LLMs (Qwen2-72B-Instruct, Gemma-2-27B) perform context-aware "
     "text normalization via structured prompt schemas."),

    (["synthetic audio", "audio synthesis", "audio generation", "synthetic speech"],
     "concept_synthetic_audio", "Synthetic Audio Generation",
     "TTS models generate synthetic audio from preprocessed text to augment the training "
     "dataset for Whisper fine-tuning with diverse prosody and speaker variation."),

    (["noise", "acoustic noise", "noise robustness", "noisy", "noise handling"],
     "concept_noise", "Acoustic Noise and Robustness",
     "Background noise from machinery and speaker variability are addressed through "
     "spectral subtraction, beamforming, VAD, and noise-augmented training data."),

    (["nlp", "natural language processing"],
     "concept_nlp", "NLP (Natural Language Processing)",
     "NLP techniques including intent recognition and entity extraction enhance ASR and "
     "TTS performance in domain-specific automotive manufacturing contexts."),

    (["post-processing", "post processing", "llm post-processing", "refinement"],
     "concept_postprocessing", "LLM Post-Processing Pipeline",
     "Fine-tuned Whisper output is refined by an LLM (reward model Pφ) to produce "
     "semantically structured transcriptions r* from raw ASR output r'."),

    (["forced alignment", "ctc alignment", "alignment"],
     "concept_forced_alignment", "Forced Alignment Evaluation",
     "CTC-based forced alignment measures frame-level synchronization between audio and "
     "transcript; used as Metric 1 for TTS quality evaluation."),

    (["phoneme", "phoneme similarity", "levenshtein", "allosaurus", "g2p"],
     "concept_phoneme_similarity", "Phoneme Similarity (Levenshtein)",
     "Levenshtein distance over Allosaurus-generated phoneme sequences measures "
     "pronunciation similarity; used as Metric 2 for TTS quality evaluation."),

    (["lora", "low-rank adaptation", "parameter-efficient"],
     "concept_lora", "LoRA (Low-Rank Adaptation)",
     "LoRA reduces trainable parameters by introducing low-rank matrices A and B, "
     "enabling efficient domain-specific fine-tuning of large transformer models."),

    (["audiobot", "audio bot", "issue logging", "issue tracking"],
     "concept_audiobot", "Audiobot System",
     "An innovative audiobot system that accurately transcribes spoken narration in "
     "real-time and organizes output in a designated 'issue' directory."),

    (["automotive", "automotive manufacturing", "manufacturing", "prostech", "prostep"],
     "concept_automotive", "Automotive Manufacturing Domain",
     "The thesis targets automotive manufacturing for domain-specific ASR, with "
     "PROSTEP AG providing high-quality data and computational resources."),

    (["voice activity detection", "vad"],
     "concept_vad", "Voice Activity Detection (VAD)",
     "CNN-based noise-aware VAD models distinguish speech from non-speech segments, "
     "reducing error rates by 20% in high-noise factory scenarios."),

    (["dataset", "corpus", "training data", "augmentation"],
     "concept_dataset", "Dataset and Augmentation",
     "A curated corpus of symbolic technical speech is preprocessed, augmented with "
     "synthetic TTS audio and controlled noise for Whisper fine-tuning."),
]

# ── Method nodes ─────────────────────────────────────────────────────────────
THESIS_METHODS: list[tuple[list[str], str, str, str]] = [
    (["rule-based", "symbol replacement", "camelcase", "number expansion", "acronym"],
     "method_rule_based_preprocessing", "Rule-Based Preprocessing Method",
     "Deterministic pipeline: symbol replacement (symbols.json), number expansion, "
     "CamelCase splitting, acronym handling, whitespace normalization."),

    (["qwen", "qwen2", "gemma", "llm-based normaliz"],
     "method_llm_based_preprocessing", "LLM-Based Preprocessing Method",
     "Qwen2-72B-Instruct and Gemma-2-27B prompted with structured TTS preprocessing "
     "schema to perform context-sensitive text normalization."),

    (["whisperspeech", "whisper speech", "semantic token"],
     "method_whisperspeech", "WhisperSpeech TTS Model",
     "Combines Whisper encoder for semantic tokens, Transformer decoder for acoustic "
     "features, and Vocos vocoder for waveform reconstruction."),

    (["bark", "encodec", "vq-vae"],
     "method_bark", "Bark TTS Model",
     "Non-autoregressive GPT-style model using VQ-VAE tokenization and EnCodec decoder; "
     "supports zero-shot synthesis with accent and style transfer."),

    (["parler", "flan-t5", "dac vocoder", "g2p converter"],
     "method_parler", "Parler TTS Model",
     "Uses Flan-T5 encoder, autoregressive decoder with cross-attention, DAC vocoder; "
     "efficient multilingual synthesis with phoneme-aware pipeline."),

    (["kokoro", "styletts2", "istftnet", "emotion tag"],
     "method_kokoro", "Kokoro TTS Model",
     "Lightweight transformer on StyleTTS2 and ISTFTNet; emotion-conditioned speech "
     "via explicit emotion tags prepended to input text."),

    (["f5-tts", "f5 tts", "voice cloning", "diffwave", "diffusion", "speaker embedding"],
     "method_f5tts", "F5-TTS Voice Cloning Model",
     "Diffusion-based voice cloning using brief reference recordings; integrates speaker "
     "embeddings and text conditioning into a denoising pipeline."),

    (["ctc", "connectionist temporal classification", "viterbi"],
     "method_ctc_alignment", "CTC-Based Forced Alignment",
     "Viterbi dynamic-programming alignment of padded frame-level emissions to token "
     "sequences; computes per-token alignment confidence scores."),

    (["lcs", "longest common subsequence", "lcs-based similarity"],
     "method_lcs_similarity", "LCS-Based Similarity Metric",
     "Character-level similarity S(r,r') = 2·len(LCS(r,r')) / (len(r)+len(r')); "
     "used for preprocessing evaluation and post-processing fidelity."),

    (["cosine learning rate", "warmup", "cosine schedule"],
     "method_lr_schedule", "Cosine Learning Rate Schedule",
     "Cosine decay with warm-up phase Ew and total steps T; used during Whisper "
     "fine-tuning to stabilise training and avoid early overfitting."),

    (["reward model", "rlhf", "reinforcement"],
     "method_reward_model", "LLM Reward Model Post-Processing",
     "Reward model Pφ selects the most likely refined output r+ given raw ASR output r "
     "and policy π, refining transcriptions into semantically structured text r*."),

    (["spectral subtraction", "wiener filtering", "beamforming", "noise cancellation"],
     "method_noise_handling", "Acoustic Noise Handling Methods",
     "Spectral subtraction, Wiener filtering, adaptive beamforming, and SSL techniques "
     "(HuBERT, WavLM) improve ASR accuracy in noisy factory environments."),
]

# ── Metric nodes ─────────────────────────────────────────────────────────────
THESIS_METRICS: list[tuple[list[str], str, str, str]] = [
    (["word error rate", "wer"],
     "metric_wer", "WER (Word Error Rate)",
     "WER = (Substitutions + Deletions + Insertions) / Total Reference Words. "
     "Fine-tuned Whisper achieves WER = 0.31 on unknown test set."),

    (["character error rate", "cer"],
     "metric_cer", "CER (Character Error Rate)",
     "CER computed at character level; fine-tuned Whisper achieves CER = 0.05 on "
     "unknown test set, indicating strong character-level accuracy."),

    (["rouge-l", "rouge l", "rouge"],
     "metric_rouge", "ROUGE-L F1 Score",
     "Longest Common Subsequence-based F1 metric. Fine-tuned Whisper achieves "
     "ROUGE-L F1 = 0.90 on unknown test set."),

    (["bertscore", "bert score", "bert f1"],
     "metric_bertscore", "BERTScore F1",
     "Contextual embedding similarity via BERT; fine-tuned Whisper achieves "
     "BERTScore F1 = 0.87, indicating high semantic fidelity."),

    (["bleu", "bilingual evaluation understudy"],
     "metric_bleu", "BLEU Score",
     "N-gram precision metric with brevity penalty; used in back-transcription "
     "evaluation of TTS output quality."),

    (["mean opinion score", "mos"],
     "metric_mos", "MOS (Mean Opinion Score)",
     "Subjective perceptual quality metric on a 1-5 Likert scale; gold standard "
     "for TTS naturalness evaluation across human listeners."),

    (["forced alignment score", "alignment score", "normalized forced alignment"],
     "metric_forced_alignment_score", "Forced Alignment Score",
     "Normalized CTC alignment confidence scores across 11 TTS models; higher "
     "scores indicate clearer articulation and better prosodic structuring."),

    (["phoneme similarity", "phonetic similarity", "levenshtein distance"],
     "metric_phoneme_similarity", "Phoneme Similarity Score",
     "Levenshtein distance over Allosaurus phoneme sequences; used as Metric 2 "
     "to compare pronunciation of synthesized vs. reference speech."),

    (["llm verification", "llm-based semantic", "llm semantic score"],
     "metric_llm_verification", "LLM-Based Semantic Verification Score",
     "Instruction-tuned LLM verifies transcript presence/sequence in synthesized "
     "speech; used as Metric 3 for TTS quality evaluation."),

    (["composite score", "normalized evaluation score", "scaled performance"],
     "metric_composite", "Composite TTS Scoring",
     "Normalized evaluation score combining forced alignment, phoneme similarity, "
     "and LLM verification scaled to a 0-100 range for model comparison."),

    (["structural similarity", "mean structural similarity"],
     "metric_structural_similarity", "Structural Similarity (LCS-Based)",
     "S(r,r*) = 2·LCS(r,r*) / (len(r)+len(r*)); used to evaluate "
     "preprocessing output quality across corpus samples."),
]

# ── Result / Finding nodes ────────────────────────────────────────────────────
THESIS_RESULTS: list[tuple[list[str], str, str, str]] = [
    (["wer 0.31", "cer 0.05", "rouge 0.90", "bertscore 0.87", "aggregate performance"],
     "result_whisper_finetuning", "Fine-Tuned Whisper Performance on Test Set",
     "Fine-tuned Whisper achieves WER=0.31, CER=0.05, ROUGE-L F1=0.90, BERTScore F1=0.87 "
     "on unseen symbolic technical speech (114 utterances)."),

    (["rule-based vs llm", "preprocessing comparison", "corpus-level similarity"],
     "result_preprocessing", "Preprocessing Comparison Results",
     "LLM-based preprocessing shows higher semantic fluency; rule-based achieves stricter "
     "symbol-level fidelity. Mean character-level similarity evaluated across 100 utterances."),

    (["tts model comparison", "11 models", "composite scoring", "normalized score"],
     "result_tts_comparison", "TTS Model Comparison Results",
     "11 TTS models evaluated via forced alignment, phoneme similarity, and LLM verification. "
     "Composite normalized scores on 0-100 scale rank models for synthetic data generation."),

    (["llm post-processing impact", "r prime", "r star", "semantic structure"],
     "result_llm_postprocessing", "LLM Post-Processing Impact",
     "LLM post-processing transforms raw ASR output r' into semantically structured text r*, "
     "preserving meaning while correcting domain-specific terminology and structure."),

    (["audiobot interface", "interface evolution", "waveform editing", "defect logging"],
     "result_audiobot", "Audiobot Interface Evolution",
     "Three interface stages: initial ASR with symbol normalization, mid-stage LLM "
     "post-processing, final tool with waveform editing and defect logging."),
]


# ─────────────────────────────────────────────────────────────────────────────
# Regex patterns for heading detection
# ─────────────────────────────────────────────────────────────────────────────

# Matches: "1 Introduction", "2 Background", "3 Methodology" etc.
CHAPTER_HEADING_RE = re.compile(
    r"^(Chapter\s+)?(\d)\s+(Introduction|Background|Methodology|Results|Future\s*Work|Summary)",
    re.IGNORECASE,
)
# Matches section headings like "2.1", "3.2.1", "4.6.2" etc.
SECTION_HEADING_RE = re.compile(
    r"^(\d\.\d+(?:\.\d+)?)\s+([A-Z][^\n]{3,80})",
)
# Fallback: heading-looking line (title case, short, no period at end)
TITLE_LINE_RE = re.compile(r"^[A-Z][A-Za-z0-9\s,\-\(\)/:]{10,80}$")

# Pages to skip (front matter)
SKIP_PAGES_BEFORE = 17   # thesis content starts around page 18 (Chapter 1)


# ─────────────────────────────────────────────────────────────────────────────
# PDF extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_pages(pdf_path: Path) -> list[dict]:
    """Return list of {page_number, text} dicts from the PDF."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        pages.append({"page_number": page_num, "text": text})
    doc.close()
    logger.info(f"Extracted {len(pages)} pages from {pdf_path.name}")
    return pages


def clean_text(text: str) -> str:
    """Remove hyphenated line-breaks and normalise whitespace."""
    # Reconnect hyphenated words across line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse multiple spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    # Keep single newlines but remove triple+
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Structure detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_chapter_from_text(text: str) -> tuple[int | None, str | None]:
    """Detect chapter number and title from page text."""
    for line in text.split("\n")[:10]:
        line = line.strip()
        m = CHAPTER_HEADING_RE.match(line)
        if m:
            ch_num = int(m.group(2))
            ch_title = m.group(3).strip()
            return ch_num, f"Chapter {ch_num} {ch_title}"
        # Also match "Chapter3Methodology" style (no spaces, from PDF extraction)
        m2 = re.match(r"Chapter(\d)(Introduction|Background|Methodology|Results|FutureWork|Summary)",
                      line.replace(" ", ""), re.IGNORECASE)
        if m2:
            ch_num = int(m2.group(1))
            ch_name = THESIS_CHAPTERS.get(ch_num, m2.group(2))
            return ch_num, f"Chapter {ch_num} {ch_name}"
    return None, None


def detect_section_from_text(text: str) -> tuple[str | None, str | None]:
    """Return (section_number, section_title) if a section heading is found."""
    for line in text.split("\n")[:15]:
        line = line.strip()
        m = SECTION_HEADING_RE.match(line)
        if m:
            sec_num = m.group(1)
            sec_title = m.group(2).strip()
            # Cross-reference with known sections
            known = THESIS_SECTIONS.get(sec_num)
            return sec_num, known if known else sec_title
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_SIZE_CHARS = 1200    # ~350 tokens at 3.5 chars/token
CHUNK_OVERLAP_CHARS = 200


def chunk_text(
    text: str,
    page: int,
    chapter: str,
    chapter_index: int,
    section: str,
    section_index: str,
    chunk_id_prefix: str,
) -> list[TextChunk]:
    """Split text into overlapping chunks respecting paragraph boundaries."""
    # Split at paragraph boundaries first
    paragraphs = re.split(r"\n\n+", text)
    chunks: list[TextChunk] = []
    current = ""
    chunk_idx = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= CHUNK_SIZE_CHARS:
            current = current + "\n\n" + para if current else para
        else:
            if current:
                chunks.append(TextChunk(
                    chunk_id=f"{chunk_id_prefix}_{chunk_idx}",
                    text=current.strip(),
                    page=page,
                    chapter=chapter,
                    chapter_index=chapter_index,
                    section=section,
                    section_index=section_index,
                ))
                chunk_idx += 1
                # Keep overlap
                overlap_text = current[-CHUNK_OVERLAP_CHARS:] if len(current) > CHUNK_OVERLAP_CHARS else current
                current = overlap_text + "\n\n" + para
            else:
                current = para

    if current.strip():
        chunks.append(TextChunk(
            chunk_id=f"{chunk_id_prefix}_{chunk_idx}",
            text=current.strip(),
            page=page,
            chapter=chapter,
            chapter_index=chapter_index,
            section=section,
            section_index=section_index,
        ))

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_chapter_node_id(ch_idx: int) -> str:
    return f"chapter_{ch_idx}"


def build_section_node_id(sec_num: str) -> str:
    return f"section_{sec_num.replace('.', '_')}"


def ensure_chapter_node(g: nx.DiGraph, ch_idx: int) -> str:
    nid = build_chapter_node_id(ch_idx)
    if not g.has_node(nid):
        label = f"Chapter {ch_idx} {THESIS_CHAPTERS.get(ch_idx, '')}"
        g.add_node(nid, id=nid, label=label, type="Chapter",
                   chapter_index=ch_idx, section_index=None,
                   text_snippet=f"Chapter {ch_idx}: {THESIS_CHAPTERS.get(ch_idx, '')}")
    return nid


def ensure_section_node(g: nx.DiGraph, sec_num: str, ch_idx: int) -> str:
    nid = build_section_node_id(sec_num)
    if not g.has_node(nid):
        label = f"{sec_num} {THESIS_SECTIONS.get(sec_num, '')}"
        g.add_node(nid, id=nid, label=label, type="Section",
                   chapter_index=ch_idx, section_index=sec_num,
                   text_snippet=THESIS_SECTIONS.get(sec_num, label))
        # Connect to chapter
        parent_ch = int(sec_num.split(".")[0])
        ch_nid = ensure_chapter_node(g, parent_ch)
        g.add_edge(ch_nid, nid, rel="CONTAINS")
        # Connect subsection to parent section if applicable
        parts = sec_num.split(".")
        if len(parts) >= 3:
            parent_sec = ".".join(parts[:2])
            parent_nid = build_section_node_id(parent_sec)
            if g.has_node(parent_nid):
                g.add_edge(parent_nid, nid, rel="CONTAINS")
        elif len(parts) == 2 and int(parts[1]) > 1:
            # Add PRECEDES edge from previous section
            prev_sec = f"{parts[0]}.{int(parts[1])-1}"
            prev_nid = build_section_node_id(prev_sec)
            if g.has_node(prev_nid):
                g.add_edge(prev_nid, nid, rel="PRECEDES")
    return nid


def add_predefined_nodes(g: nx.DiGraph) -> None:
    """Add all concept, method, and metric nodes from the thesis dictionaries."""
    # Concepts
    for _, nid, label, snippet in THESIS_CONCEPTS:
        g.add_node(nid, id=nid, label=label, type="Concept",
                   chapter_index=None, section_index=None, text_snippet=snippet)

    # Methods
    for _, nid, label, snippet in THESIS_METHODS:
        g.add_node(nid, id=nid, label=label, type="Method",
                   chapter_index=None, section_index=None, text_snippet=snippet)

    # Metrics
    for _, nid, label, snippet in THESIS_METRICS:
        g.add_node(nid, id=nid, label=label, type="Metric",
                   chapter_index=None, section_index=None, text_snippet=snippet)

    # Results
    for _, nid, label, snippet in THESIS_RESULTS:
        g.add_node(nid, id=nid, label=label, type="Result",
                   chapter_index=None, section_index=None, text_snippet=snippet)


def add_predefined_edges(g: nx.DiGraph) -> None:
    """Add hand-crafted semantic edges derived from reading the thesis."""
    edges = [
        # Chapter-level thematic edges
        ("chapter_3", "concept_preprocessing", "EXPLAINS"),
        ("chapter_3", "concept_synthetic_audio", "EXPLAINS"),
        ("chapter_3", "concept_finetuning", "EXPLAINS"),
        ("chapter_3", "concept_postprocessing", "EXPLAINS"),
        ("chapter_3", "concept_forced_alignment", "EXPLAINS"),
        ("chapter_3", "concept_phoneme_similarity", "EXPLAINS"),
        ("chapter_2", "concept_asr", "EXPLAINS"),
        ("chapter_2", "concept_tts", "EXPLAINS"),
        ("chapter_2", "concept_whisper", "EXPLAINS"),
        ("chapter_2", "concept_llm", "EXPLAINS"),
        ("chapter_2", "concept_nlp", "EXPLAINS"),
        ("chapter_4", "metric_wer", "MENTIONS"),
        ("chapter_4", "metric_cer", "MENTIONS"),
        ("chapter_4", "metric_rouge", "MENTIONS"),
        ("chapter_4", "metric_bertscore", "MENTIONS"),
        ("chapter_4", "result_whisper_finetuning", "MENTIONS"),
        ("chapter_4", "result_preprocessing", "MENTIONS"),
        ("chapter_4", "result_tts_comparison", "MENTIONS"),
        ("chapter_4", "result_llm_postprocessing", "MENTIONS"),

        # Method → Metric uses
        ("method_ctc_alignment", "metric_forced_alignment_score", "USES"),
        ("method_lcs_similarity", "metric_structural_similarity", "USES"),
        ("method_rule_based_preprocessing", "metric_structural_similarity", "USES"),
        ("method_llm_based_preprocessing", "metric_structural_similarity", "USES"),
        ("method_reward_model", "metric_wer", "USES"),
        ("method_reward_model", "metric_cer", "USES"),
        ("method_reward_model", "metric_bertscore", "USES"),
        ("method_reward_model", "metric_rouge", "USES"),

        # Concept → Concept relationships
        ("concept_preprocessing", "concept_tts", "PRECEDES"),
        ("concept_tts", "concept_synthetic_audio", "USES"),
        ("concept_synthetic_audio", "concept_finetuning", "PRECEDES"),
        ("concept_finetuning", "concept_whisper", "USES"),
        ("concept_whisper", "concept_postprocessing", "PRECEDES"),
        ("concept_postprocessing", "concept_llm", "USES"),
        ("concept_asr", "concept_noise", "RELATED"),
        ("concept_asr", "concept_vad", "USES"),
        ("concept_nlp", "concept_asr", "ENHANCES"),
        ("concept_nlp", "concept_tts", "ENHANCES"),
        ("concept_lora", "concept_finetuning", "USES"),

        # Section → Concept mentions
        ("section_3_2", "concept_preprocessing", "EXPLAINS"),
        ("section_3_2_1", "concept_rule_based_preprocessing", "EXPLAINS"),
        ("section_3_2_2", "concept_llm_preprocessing", "EXPLAINS"),
        ("section_3_2_2", "concept_llm", "MENTIONS"),
        ("section_3_3", "concept_tts", "EXPLAINS"),
        ("section_3_3", "concept_synthetic_audio", "EXPLAINS"),
        ("section_3_4_2", "concept_forced_alignment", "EXPLAINS"),
        ("section_3_4_2", "method_ctc_alignment", "USES"),
        ("section_3_4_3", "concept_phoneme_similarity", "EXPLAINS"),
        ("section_3_4_3", "method_lcs_similarity", "USES"),
        ("section_3_5", "concept_finetuning", "EXPLAINS"),
        ("section_3_5", "concept_whisper", "USES"),
        ("section_3_5", "concept_lora", "MENTIONS"),
        ("section_3_6", "concept_postprocessing", "EXPLAINS"),
        ("section_3_6_3", "method_reward_model", "EXPLAINS"),
        ("section_2_2", "concept_asr", "EXPLAINS"),
        ("section_2_2_4", "concept_phoneme_similarity", "MENTIONS"),
        ("section_2_1_5", "concept_tts", "EXPLAINS"),
        ("section_2_6_3", "metric_wer", "EXPLAINS"),
        ("section_2_6_3", "metric_cer", "EXPLAINS"),
        ("section_2_6_3", "metric_rouge", "EXPLAINS"),
        ("section_2_6_3", "metric_bertscore", "EXPLAINS"),
        ("section_2_6_3", "metric_bleu", "EXPLAINS"),
        ("section_2_6_3", "metric_mos", "EXPLAINS"),
        ("section_4_4", "result_whisper_finetuning", "MENTIONS"),
        ("section_4_5", "result_llm_postprocessing", "MENTIONS"),
        ("section_4_6", "metric_wer", "MENTIONS"),
        ("section_4_6", "metric_cer", "MENTIONS"),
        ("section_4_6", "metric_rouge", "MENTIONS"),
        ("section_4_6", "metric_bertscore", "MENTIONS"),
        ("section_4_1", "result_preprocessing", "MENTIONS"),
        ("section_4_3", "result_tts_comparison", "MENTIONS"),

        # Method → Concept
        ("method_rule_based_preprocessing", "concept_rule_based_preprocessing", "IMPLEMENTS"),
        ("method_llm_based_preprocessing", "concept_llm_preprocessing", "IMPLEMENTS"),
        ("method_whisperspeech", "concept_whisper", "USES"),
        ("method_whisperspeech", "concept_tts", "IMPLEMENTS"),
        ("method_bark", "concept_tts", "IMPLEMENTS"),
        ("method_parler", "concept_tts", "IMPLEMENTS"),
        ("method_kokoro", "concept_tts", "IMPLEMENTS"),
        ("method_f5tts", "concept_tts", "IMPLEMENTS"),
        ("method_noise_handling", "concept_noise", "ADDRESSES"),
        ("method_noise_handling", "concept_vad", "USES"),

        # TTS model → metric links
        ("method_whisperspeech", "metric_forced_alignment_score", "EVALUATED_BY"),
        ("method_bark", "metric_forced_alignment_score", "EVALUATED_BY"),
        ("method_parler", "metric_forced_alignment_score", "EVALUATED_BY"),
        ("method_kokoro", "metric_forced_alignment_score", "EVALUATED_BY"),
        ("method_f5tts", "metric_forced_alignment_score", "EVALUATED_BY"),

        # Result → Metric
        ("result_whisper_finetuning", "metric_wer", "USES"),
        ("result_whisper_finetuning", "metric_cer", "USES"),
        ("result_whisper_finetuning", "metric_rouge", "USES"),
        ("result_whisper_finetuning", "metric_bertscore", "USES"),
        ("result_tts_comparison", "metric_composite", "USES"),
        ("result_preprocessing", "metric_structural_similarity", "USES"),
        ("result_llm_postprocessing", "metric_bertscore", "USES"),
        ("result_llm_postprocessing", "metric_rouge", "USES"),
    ]
    for src, tgt, rel in edges:
        if g.has_node(src) and g.has_node(tgt):
            g.add_edge(src, tgt, rel=rel)
        else:
            missing = [n for n in [src, tgt] if not g.has_node(n)]
            logger.debug(f"Skipping edge {src}→{tgt} ({rel}): missing nodes {missing}")


# ─────────────────────────────────────────────────────────────────────────────
# Keyword detection for chunk → graph node linking
# ─────────────────────────────────────────────────────────────────────────────

def detect_mentioned_nodes(text: str, g: nx.DiGraph) -> list[str]:
    """
    Return list of node IDs mentioned in a text chunk.
    Checks concepts, methods, metrics, and results against the text.
    """
    text_lower = text.lower()
    found: list[str] = []

    all_entity_lists = [THESIS_CONCEPTS, THESIS_METHODS, THESIS_METRICS, THESIS_RESULTS]
    for entity_list in all_entity_lists:
        for keywords, nid, _, _ in entity_list:
            if any(kw in text_lower for kw in keywords):
                if g.has_node(nid):
                    found.append(nid)

    return list(dict.fromkeys(found))   # deduplicate, preserve order


# ─────────────────────────────────────────────────────────────────────────────
# Chapter tree builder (for /chapters API)
# ─────────────────────────────────────────────────────────────────────────────

def build_chapter_tree(g: nx.DiGraph) -> list[dict]:
    """Build the chapter → section tree for the sidebar."""
    tree = []
    for ch_idx in sorted(THESIS_CHAPTERS.keys()):
        ch_nid = build_chapter_node_id(ch_idx)
        if not g.has_node(ch_nid):
            continue
        ch_data = dict(g.nodes[ch_nid])
        sections = []
        for sec_num, sec_title in THESIS_SECTIONS.items():
            if sec_num.startswith(str(ch_idx) + ".") and sec_num.count(".") == 1:
                sec_nid = build_section_node_id(sec_num)
                if g.has_node(sec_nid):
                    sec_data = dict(g.nodes[sec_nid])
                    # Subsections
                    subsections = []
                    for sub_num in THESIS_SECTIONS:
                        if sub_num.startswith(sec_num + "."):
                            sub_nid = build_section_node_id(sub_num)
                            if g.has_node(sub_nid):
                                subsections.append(dict(g.nodes[sub_nid]))
                    sec_data["sections"] = subsections
                    sections.append(sec_data)
        ch_data["sections"] = sections
        tree.append(ch_data)
    return tree


# ─────────────────────────────────────────────────────────────────────────────
# Main ingestion entry point
# ─────────────────────────────────────────────────────────────────────────────

def ingest(pdf_path: Path, embedding_model_name: str = "all-MiniLM-L6-v2") -> None:
    """
    Full ingestion pipeline:
      1. Extract text from thesis.pdf
      2. Build knowledge graph with thesis-specific nodes/edges
      3. Chunk text and attach to graph
      4. Compute embeddings and build FAISS index
    Results are stored in the module-level APP_STATE singleton.
    """
    from vector_store import VectorStore

    logger.info("=" * 60)
    logger.info("Starting GraphRAG ingestion for Palash Mishra's thesis")
    logger.info("=" * 60)

    if not pdf_path.exists():
        logger.error(f"PDF not found at {pdf_path}. "
                     "Place thesis.pdf in backend/data/ and restart.")
        return

    # ── 1. Extract PDF pages ─────────────────────────────────────────────────
    pages = extract_pages(pdf_path)
    content_pages = [p for p in pages if p["page_number"] >= SKIP_PAGES_BEFORE]
    logger.info(f"Processing {len(content_pages)} content pages "
                f"(skipping first {SKIP_PAGES_BEFORE-1} front-matter pages)")

    # ── 2. Initialise graph and add predefined nodes ─────────────────────────
    g = nx.DiGraph()
    add_predefined_nodes(g)

    # Ensure all chapter nodes exist
    for ch_idx in THESIS_CHAPTERS:
        ensure_chapter_node(g, ch_idx)

    # Ensure all section nodes exist (seeded from known ToC)
    for sec_num in THESIS_SECTIONS:
        ch_idx = int(sec_num.split(".")[0])
        ensure_section_node(g, sec_num, ch_idx)

    # PRECEDES edges between chapters
    ch_ids = sorted(THESIS_CHAPTERS.keys())
    for i in range(len(ch_ids) - 1):
        g.add_edge(build_chapter_node_id(ch_ids[i]),
                   build_chapter_node_id(ch_ids[i + 1]),
                   rel="PRECEDES")

    # Add all hand-crafted semantic edges
    add_predefined_edges(g)

    # ── 3. Parse pages → track current chapter/section ──────────────────────
    all_chunks: list[TextChunk] = []
    current_chapter_idx = 0
    current_chapter_label = "Preamble"
    current_section_num = ""
    current_section_label = ""

    for page_data in content_pages:
        pnum = page_data["page_number"]
        raw_text = page_data["text"]
        text = clean_text(raw_text)

        # Detect chapter boundary
        ch_idx, ch_label = detect_chapter_from_text(raw_text)
        if ch_idx is not None:
            current_chapter_idx = ch_idx
            current_chapter_label = ch_label

        # Detect section boundary
        sec_num, sec_label = detect_section_from_text(raw_text)
        if sec_num is not None:
            current_section_num = sec_num
            current_section_label = sec_label or THESIS_SECTIONS.get(sec_num, sec_num)

        if not text or len(text) < 60:
            continue

        # Build a chunk prefix based on location
        safe_ch = f"ch{current_chapter_idx}"
        safe_sec = current_section_num.replace(".", "_") if current_section_num else "nosec"
        chunk_prefix = f"chunk_{safe_ch}_{safe_sec}_p{pnum}"

        page_chunks = chunk_text(
            text=text,
            page=pnum,
            chapter=current_chapter_label,
            chapter_index=current_chapter_idx,
            section=current_section_label,
            section_index=current_section_num,
            chunk_id_prefix=chunk_prefix,
        )

        for chunk in page_chunks:
            # Attach chunk node to graph
            chunk_nid = f"chunk_{chunk.chunk_id}"
            g.add_node(chunk_nid, id=chunk_nid, label=f"Chunk p.{pnum}",
                       type="Chunk", chapter_index=current_chapter_idx,
                       section_index=current_section_num,
                       text_snippet=chunk.text[:150])

            # CONTAINS edges: section/chapter → chunk
            if current_section_num:
                sec_nid = build_section_node_id(current_section_num)
                if g.has_node(sec_nid):
                    g.add_edge(sec_nid, chunk_nid, rel="CONTAINS")
            elif current_chapter_idx > 0:
                ch_nid = build_chapter_node_id(current_chapter_idx)
                if g.has_node(ch_nid):
                    g.add_edge(ch_nid, chunk_nid, rel="CONTAINS")

            # MENTIONS edges: chunk → concepts/methods/metrics
            mentioned = detect_mentioned_nodes(chunk.text, g)
            chunk.graph_nodes = [current_section_num, current_chapter_label] + mentioned
            for mnid in mentioned:
                g.add_edge(chunk_nid, mnid, rel="MENTIONS")

            all_chunks.append(chunk)

    logger.info(f"Created {len(all_chunks)} text chunks")
    logger.info(f"Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    logger.info(f"  Chapters: {sum(1 for _,d in g.nodes(data=True) if d.get('type')=='Chapter')}")
    logger.info(f"  Sections: {sum(1 for _,d in g.nodes(data=True) if d.get('type')=='Section')}")
    logger.info(f"  Concepts: {sum(1 for _,d in g.nodes(data=True) if d.get('type')=='Concept')}")
    logger.info(f"  Methods:  {sum(1 for _,d in g.nodes(data=True) if d.get('type')=='Method')}")
    logger.info(f"  Metrics:  {sum(1 for _,d in g.nodes(data=True) if d.get('type')=='Metric')}")
    logger.info(f"  Results:  {sum(1 for _,d in g.nodes(data=True) if d.get('type')=='Result')}")
    logger.info(f"  Chunks:   {sum(1 for _,d in g.nodes(data=True) if d.get('type')=='Chunk')}")

    # ── 4. Load embedding model ───────────────────────────────────────────────
    embedding_model = None
    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {embedding_model_name}")
        embedding_model = SentenceTransformer(embedding_model_name)
        logger.info("Embedding model loaded.")
    except Exception as e:
        logger.warning(f"Could not load sentence-transformers: {e}. "
                       "Vector search will be disabled.")

    # ── 5. Compute embeddings ─────────────────────────────────────────────────
    if embedding_model is not None:
        logger.info(f"Computing embeddings for {len(all_chunks)} chunks…")
        texts = [c.text for c in all_chunks]
        embeddings = embedding_model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")
        for i, chunk in enumerate(all_chunks):
            chunk.embedding = embeddings[i]
        logger.info("Embeddings computed.")
    else:
        embeddings = None

    # ── 6. Build vector store ─────────────────────────────────────────────────
    vs = VectorStore()
    vs.build(all_chunks)
    logger.info("Vector store built.")

    # ── 7. Build chapter tree ─────────────────────────────────────────────────
    chapter_tree = build_chapter_tree(g)

    # ── 8. Update APP_STATE ───────────────────────────────────────────────────
    APP_STATE.graph = g
    APP_STATE.chunks = all_chunks
    APP_STATE.vector_store = vs
    APP_STATE.embedding_model = embedding_model
    APP_STATE.chapter_tree = chapter_tree

    logger.info("Ingestion complete. GraphRAG system ready.")
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/thesis.pdf")
    ingest(pdf)
    print(f"\nGraph nodes: {APP_STATE.graph.number_of_nodes()}")
    print(f"Graph edges: {APP_STATE.graph.number_of_edges()}")
    print(f"Chunks:      {len(APP_STATE.chunks)}")
    print(f"Chapter tree entries: {len(APP_STATE.chapter_tree)}")
