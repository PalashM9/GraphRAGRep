"""
llm.py – LLM abstraction for GraphRAG.

Providers (set LLM_PROVIDER in .env):
  groq    – Groq free tier, fast (default) → console.groq.com
  openai  – OpenAI API
  grok    – xAI Grok → console.x.ai
  puter   – Puter free endpoint
  mock    – offline placeholder

Get free Groq key at: https://console.groq.com
"""

import os
from dotenv import load_dotenv

load_dotenv()

PROVIDER      = os.getenv("LLM_PROVIDER",  "groq").lower()
GROQ_API_KEY  = os.getenv("GROQ_API_KEY",  "")
GROQ_MODEL    = os.getenv("GROQ_MODEL",    "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """You are an expert research assistant analysing a Master's thesis
titled "A Generative Artificial Intelligence Approach for Enabling Automatic Speech
Recognition: Seamless Issue Logging, Accent Acquisition, Noise Handling and
Post-Processing" by Palash Mishra, Bauhaus-Universität Weimar, 2025.

The thesis covers:
- Rule-based and LLM-based text preprocessing (Qwen2-72B-Instruct, Gemma-2-27B)
- TTS audio synthesis with WhisperSpeech, Bark, Parler, Kokoro, F5-TTS
- TTS evaluation: Forced Alignment (CTC), Phoneme Similarity (Levenshtein/Allosaurus),
  LLM-based Semantic Verification
- Whisper fine-tuning on symbolic technical speech (automotive manufacturing domain)
- LLM post-processing pipeline using a reward model Pφ to refine ASR output r' → r*
- Results: WER=0.31, CER=0.05, ROUGE-L F1=0.90, BERTScore F1=0.87 on unknown test set
- Dataset provided by PROSTEP AG

Be precise and structured. Always cite the chapter or section your answer comes from.
Keep answers concise but complete."""


def generate_answer(prompt: str) -> str:
    if PROVIDER == "groq":
        return _groq(prompt)
    elif PROVIDER == "grok":
        return _openai_compatible(
            prompt,
            base_url=os.getenv("GROK_BASE_URL", "https://api.x.ai/v1"),
            api_key=os.getenv("GROK_API_KEY", ""),
            model=os.getenv("GROK_MODEL", "grok-3-mini"),
        )
    elif PROVIDER == "openai":
        return _openai_compatible(
            prompt,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        )
    elif PROVIDER == "puter":
        return _openai_compatible(
            prompt,
            base_url="https://api.puter.com/puterai/openai/v1",
            api_key=os.getenv("PUTER_AUTH_TOKEN", ""),
            model=os.getenv("PUTER_MODEL", "gpt-4o-mini"),
        )
    elif PROVIDER == "flutter":
        return _openai_compatible(
            prompt,
            base_url=os.getenv("FLUTTER_BASE_URL", "https://api.flutter.io/v1"),
            api_key=os.getenv("FLUTTER_API_KEY", ""),
            model=os.getenv("FLUTTER_MODEL", "gpt-4o-mini"),
        )
    else:
        return _mock(prompt)


def _groq(prompt: str) -> str:
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.2,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"[Groq error] {e}")
        return _mock(prompt)


def _openai_compatible(prompt: str, base_url: str, api_key: str, model: str) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.2,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"[LLM error — {PROVIDER}] {e}")
        return _mock(prompt)


def _mock(prompt: str) -> str:
    lines = prompt.strip().split("\n")
    question_line = next((l for l in lines if l.startswith("QUESTION:")), "")
    question = question_line.replace("QUESTION:", "").strip() or "your question"
    return (
        f"[Demo mode — no LLM key configured]\n\n"
        f"Regarding: '{question}'\n\n"
        f"The thesis investigates LLM-based preprocessing pipelines and their "
        f"effect on ASR performance via WER/CER on Whisper fine-tuned models.\n\n"
        f"Set GROQ_API_KEY in .env for live answers (free at console.groq.com)."
    )