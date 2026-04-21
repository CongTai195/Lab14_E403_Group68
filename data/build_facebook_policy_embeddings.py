from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "facebook_policy_txt"
CHROMA_DIR = ROOT_DIR / "data" / "chroma_facebook_policy"
CHUNKS_JSONL = ROOT_DIR / "data" / "facebook_policy_chunks.jsonl"
MANIFEST_JSON = ROOT_DIR / "data" / "facebook_policy_embedding_manifest.json"

COLLECTION_NAME = "facebook_meta_policy_vi"
EMBEDDING_MODEL = "text-embedding-3-large"
LANGUAGE = "vi"

MIN_CHUNK_TOKENS = 650
PREFERRED_MAX_CHUNK_TOKENS = 900
HARD_MAX_CHUNK_TOKENS = 1200
PACKING_MAX_CHUNK_TOKENS = HARD_MAX_CHUNK_TOKENS - 40
OVERLAP_TOKENS = 100
SIMILARITY_THRESHOLD = 0.70

SOURCE_RE = re.compile(r"^===== SOURCE DOCUMENT\s+(\d+)\s*$")
DIVIDER_RE = re.compile(r"^={10,}$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
LIST_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[\.)]\s+)")
META_KEYS = {
    "TITLE",
    "SOURCE_TYPE",
    "CANONICAL_URL",
    "URL",
    "RETRIEVED_AT",
    "CONTENT_SHA256",
}


@dataclass
class SourceDocument:
    source_file: str
    group_id: str
    source_doc_id: str
    title: str
    source_type: str
    canonical_url: str
    source_sha256: str
    body_lines: list[str]


@dataclass
class SemanticUnit:
    text: str
    heading_path: str
    token_count: int
    embedding: list[float] | None = None


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


def get_encoding():
    try:
        return tiktoken.encoding_for_model(EMBEDDING_MODEL)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


ENCODING = get_encoding()


def token_count(text: str) -> int:
    return len(ENCODING.encode(text))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def discover_policy_files(data_dir: Path = DATA_DIR) -> list[Path]:
    files = []
    for path in sorted(data_dir.glob("0[1-7]_*.txt")):
        if path.name == "00_sources_manifest.txt":
            continue
        files.append(path)
    if not files:
        raise FileNotFoundError(f"No cleaned policy txt files found in {data_dir}")
    return files


def parse_header_value(lines: list[str], key: str, default: str = "") -> str:
    prefix = f"{key}:"
    for line in lines:
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return default


def parse_sources(path: Path) -> list[SourceDocument]:
    lines = path.read_text(encoding="utf-8").splitlines()
    group_id = path.stem

    docs: list[list[str]] = []
    current: list[str] | None = None
    for line in lines:
        if SOURCE_RE.match(line):
            if current:
                docs.append(current)
            current = [line]
        elif current is not None:
            current.append(line)
    if current:
        docs.append(current)

    parsed: list[SourceDocument] = []
    for doc_lines in docs:
        source_match = SOURCE_RE.match(doc_lines[0])
        if not source_match:
            continue
        source_doc_id = source_match.group(1)
        metadata: dict[str, str] = {}
        body_lines: list[str] = []
        in_body = False

        for line in doc_lines[1:]:
            if DIVIDER_RE.match(line):
                in_body = True
                continue
            if not in_body and ":" in line:
                key, value = line.split(":", 1)
                if key in META_KEYS:
                    metadata[key] = value.strip()
                    continue
            if in_body and line.strip():
                body_lines.append(line.rstrip())

        body_text = "\n".join(body_lines).strip()
        if not body_text:
            continue

        parsed.append(
            SourceDocument(
                source_file=path.name,
                group_id=group_id,
                source_doc_id=source_doc_id,
                title=metadata.get("TITLE", ""),
                source_type=metadata.get("SOURCE_TYPE", ""),
                canonical_url=metadata.get("CANONICAL_URL") or metadata.get("URL", ""),
                source_sha256=metadata.get("CONTENT_SHA256", sha256_text(body_text)),
                body_lines=body_lines,
            )
        )
    return parsed


def split_long_text(text: str, heading_path: str) -> list[SemanticUnit]:
    if token_count(text) <= HARD_MAX_CHUNK_TOKENS:
        return [SemanticUnit(text=text.strip(), heading_path=heading_path, token_count=token_count(text))]

    def token_window_units(value: str) -> list[SemanticUnit]:
        encoded = ENCODING.encode(value)
        units = []
        step = PACKING_MAX_CHUNK_TOKENS - OVERLAP_TOKENS
        for start in range(0, len(encoded), step):
            token_slice = encoded[start : start + PACKING_MAX_CHUNK_TOKENS]
            unit_text = ENCODING.decode(token_slice).strip()
            if unit_text:
                units.append(
                    SemanticUnit(
                        text=unit_text,
                        heading_path=heading_path,
                        token_count=token_count(unit_text),
                    )
                )
        return units

    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    parts = [part.strip() for part in parts if part.strip()]
    if len(parts) <= 1:
        return token_window_units(text)

    units: list[SemanticUnit] = []
    current: list[str] = []
    current_tokens = 0
    for part in parts:
        part_tokens = token_count(part)
        if part_tokens > HARD_MAX_CHUNK_TOKENS:
            if current:
                unit_text = " ".join(current).strip()
                units.append(SemanticUnit(unit_text, heading_path, token_count(unit_text)))
                current = []
                current_tokens = 0
            units.extend(token_window_units(part))
            continue
        if current and current_tokens + part_tokens > HARD_MAX_CHUNK_TOKENS:
            unit_text = " ".join(current).strip()
            units.append(SemanticUnit(unit_text, heading_path, token_count(unit_text)))
            current = []
            current_tokens = 0
        current.append(part)
        current_tokens += part_tokens
    if current:
        unit_text = " ".join(current).strip()
        units.append(SemanticUnit(unit_text, heading_path, token_count(unit_text)))
    return units


def force_split_by_tokens(text: str, heading_path: str) -> list[SemanticUnit]:
    encoded = ENCODING.encode(text)
    units: list[SemanticUnit] = []
    step = PACKING_MAX_CHUNK_TOKENS - OVERLAP_TOKENS
    for start in range(0, len(encoded), step):
        token_slice = encoded[start : start + PACKING_MAX_CHUNK_TOKENS]
        unit_text = ENCODING.decode(token_slice).strip()
        if unit_text:
            units.append(SemanticUnit(unit_text, heading_path, token_count(unit_text)))
    return units


def flush_block(block: list[str], heading_path: str, units: list[SemanticUnit]) -> None:
    if not block:
        return
    text = "\n".join(block).strip()
    if text:
        units.extend(split_long_text(text, heading_path))
    block.clear()


def make_semantic_units(source: SourceDocument) -> list[SemanticUnit]:
    units: list[SemanticUnit] = []
    heading_stack: list[str] = []
    block: list[str] = []
    block_kind: str | None = None

    def current_heading() -> str:
        return " > ".join(heading_stack) if heading_stack else source.title

    for raw_line in source.body_lines:
        line = raw_line.strip()
        if not line:
            flush_block(block, current_heading(), units)
            block_kind = None
            continue

        heading_match = HEADING_RE.match(line)
        if heading_match:
            flush_block(block, current_heading(), units)
            block_kind = None
            level = len(heading_match.group(1))
            heading = heading_match.group(2).strip()
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(heading)
            units.extend(split_long_text(line, current_heading()))
            continue

        kind = "table" if line.startswith("|") else "list" if LIST_RE.match(line) else "paragraph"
        if block and kind != block_kind:
            flush_block(block, current_heading(), units)
        block.append(line)
        block_kind = kind

    flush_block(block, current_heading(), units)
    return [unit for unit in units if unit.text.strip()]


def cosine(a: list[float], b: list[float]) -> float:
    av = np.array(a, dtype=np.float32)
    bv = np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom == 0.0:
        return 0.0
    return float(np.dot(av, bv) / denom)


def overlap_from_text(text: str, max_tokens: int = OVERLAP_TOKENS) -> str:
    encoded = ENCODING.encode(text)
    if len(encoded) <= max_tokens:
        return text
    return ENCODING.decode(encoded[-max_tokens:]).strip()


def embed_texts(client: OpenAI, texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    if not texts:
        return []

    all_embeddings: list[list[float]] = []
    batch: list[str] = []
    batch_tokens = 0
    max_batch_items = 64
    max_batch_tokens = 250_000

    def send_batch(items: list[str]) -> list[list[float]]:
        for attempt in range(6):
            try:
                response = client.embeddings.create(model=model, input=items)
                ordered = sorted(response.data, key=lambda item: item.index)
                return [item.embedding for item in ordered]
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(min(2**attempt, 30))
        raise RuntimeError("Embedding batch failed unexpectedly")

    for text in tqdm(texts, desc=f"Embedding with {model}", unit="text"):
        text_tokens = token_count(text)
        if batch and (len(batch) >= max_batch_items or batch_tokens + text_tokens > max_batch_tokens):
            all_embeddings.extend(send_batch(batch))
            batch = []
            batch_tokens = 0
        batch.append(text)
        batch_tokens += text_tokens

    if batch:
        all_embeddings.extend(send_batch(batch))

    return all_embeddings


def chunk_source(source: SourceDocument, units: list[SemanticUnit]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current_units: list[SemanticUnit] = []
    current_tokens = 0
    previous_overlap = ""
    previous_heading = ""

    def emit() -> None:
        nonlocal current_units, current_tokens, previous_overlap, previous_heading
        if not current_units:
            return
        chunk_text = "\n".join(unit.text for unit in current_units).strip()
        if not chunk_text:
            current_units = []
            current_tokens = 0
            return
        heading_path = current_units[-1].heading_path or source.title
        chunks.append(
            {
                "text": chunk_text,
                "heading_path": heading_path,
                "token_count": token_count(chunk_text),
            }
        )
        previous_overlap = overlap_from_text(chunk_text)
        previous_heading = heading_path
        current_units = []
        current_tokens = 0

    for unit in units:
        unit_tokens = unit.token_count
        if unit_tokens > HARD_MAX_CHUNK_TOKENS:
            emit()
            for split_unit in split_long_text(unit.text, unit.heading_path):
                chunks.append(
                    {
                        "text": split_unit.text,
                        "heading_path": split_unit.heading_path,
                        "token_count": split_unit.token_count,
                    }
                )
            previous_overlap = overlap_from_text(chunks[-1]["text"]) if chunks else ""
            previous_heading = unit.heading_path
            continue

        if unit_tokens > PACKING_MAX_CHUNK_TOKENS:
            emit()
            chunks.append(
                {
                    "text": unit.text,
                    "heading_path": unit.heading_path,
                    "token_count": unit.token_count,
                }
            )
            previous_overlap = overlap_from_text(unit.text)
            previous_heading = unit.heading_path
            continue

        if not current_units and previous_overlap:
            overlap_unit = SemanticUnit(
                text=previous_overlap,
                heading_path=previous_heading or unit.heading_path,
                token_count=token_count(previous_overlap),
            )
            if overlap_unit.token_count + unit_tokens <= PACKING_MAX_CHUNK_TOKENS:
                current_units.append(overlap_unit)
                current_tokens += overlap_unit.token_count

        should_emit = False
        if current_units and current_tokens + unit_tokens > PACKING_MAX_CHUNK_TOKENS:
            should_emit = True
        elif current_units and current_tokens >= MIN_CHUNK_TOKENS:
            last_embedding = current_units[-1].embedding
            next_embedding = unit.embedding
            similarity = cosine(last_embedding, next_embedding) if last_embedding and next_embedding else 1.0
            if current_tokens >= PREFERRED_MAX_CHUNK_TOKENS or similarity < SIMILARITY_THRESHOLD:
                should_emit = True

        if should_emit:
            emit()
            if previous_overlap:
                overlap_unit = SemanticUnit(
                    text=previous_overlap,
                    heading_path=previous_heading or unit.heading_path,
                    token_count=token_count(previous_overlap),
                )
                if overlap_unit.token_count + unit_tokens <= PACKING_MAX_CHUNK_TOKENS:
                    current_units.append(overlap_unit)
                    current_tokens += overlap_unit.token_count

        current_units.append(unit)
        current_tokens += unit_tokens

    emit()
    normalized_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        actual_tokens = token_count(chunk["text"])
        if actual_tokens <= HARD_MAX_CHUNK_TOKENS:
            chunk["token_count"] = actual_tokens
            normalized_chunks.append(chunk)
            continue
        for split_unit in force_split_by_tokens(chunk["text"], chunk["heading_path"]):
            normalized_chunks.append(
                {
                    "text": split_unit.text,
                    "heading_path": split_unit.heading_path,
                    "token_count": split_unit.token_count,
                }
            )
    return normalized_chunks


def build_chunks(client: OpenAI) -> list[Chunk]:
    policy_files = discover_policy_files()
    sources: list[SourceDocument] = []
    for path in policy_files:
        sources.extend(parse_sources(path))
    if not sources:
        raise RuntimeError("No source documents parsed from cleaned policy files.")

    source_units: list[tuple[SourceDocument, list[SemanticUnit]]] = []
    all_units: list[SemanticUnit] = []
    for source in sources:
        units = make_semantic_units(source)
        source_units.append((source, units))
        all_units.extend(units)

    unit_embeddings = embed_texts(client, [unit.text for unit in all_units])
    for unit, embedding in zip(all_units, unit_embeddings):
        unit.embedding = embedding

    chunks: list[Chunk] = []
    for source, units in source_units:
        raw_chunks = chunk_source(source, units)
        for idx, raw_chunk in enumerate(raw_chunks, start=1):
            text = raw_chunk["text"]
            chunk_id = f"fbmeta:{source.group_id}:source{source.source_doc_id}:chunk{idx:03d}"
            metadata = {
                "chunk_id": chunk_id,
                "source_file": source.source_file,
                "group_id": source.group_id,
                "source_doc_id": source.source_doc_id,
                "source_type": source.source_type,
                "title": source.title,
                "canonical_url": source.canonical_url,
                "heading_path": raw_chunk["heading_path"],
                "chunk_index": idx,
                "token_count": raw_chunk["token_count"],
                "language": LANGUAGE,
                "embedding_model": EMBEDDING_MODEL,
                "chunk_sha256": sha256_text(text),
                "source_sha256": source.source_sha256,
            }
            chunks.append(Chunk(chunk_id=chunk_id, text=text, metadata=metadata))

    return chunks


def reset_chroma_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_chunks_jsonl(chunks: list[Chunk]) -> None:
    with CHUNKS_JSONL.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(
                json.dumps(
                    {
                        "id": chunk.chunk_id,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_manifest(chunks: list[Chunk], elapsed_seconds: float) -> None:
    by_file: dict[str, int] = {}
    total_tokens = 0
    max_tokens = 0
    for chunk in chunks:
        source_file = str(chunk.metadata["source_file"])
        by_file[source_file] = by_file.get(source_file, 0) + 1
        tokens = int(chunk.metadata["token_count"])
        total_tokens += tokens
        max_tokens = max(max_tokens, tokens)

    manifest = {
        "collection_name": COLLECTION_NAME,
        "chroma_dir": str(CHROMA_DIR.relative_to(ROOT_DIR)),
        "chunks_jsonl": str(CHUNKS_JSONL.relative_to(ROOT_DIR)),
        "embedding_model": EMBEDDING_MODEL,
        "language": LANGUAGE,
        "chunk_count": len(chunks),
        "total_chunk_tokens": total_tokens,
        "max_chunk_tokens": max_tokens,
        "chunks_by_source_file": by_file,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "chunking": {
            "min_chunk_tokens": MIN_CHUNK_TOKENS,
            "preferred_max_chunk_tokens": PREFERRED_MAX_CHUNK_TOKENS,
            "hard_max_chunk_tokens": HARD_MAX_CHUNK_TOKENS,
            "overlap_tokens": OVERLAP_TOKENS,
            "similarity_threshold": SIMILARITY_THRESHOLD,
        },
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def persist_to_chroma(client: OpenAI, chunks: list[Chunk], rebuild: bool) -> None:
    if rebuild:
        reset_chroma_dir(CHROMA_DIR)
    else:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if rebuild:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    final_embeddings = embed_texts(client, [chunk.text for chunk in chunks])
    batch_size = 128
    for start in tqdm(range(0, len(chunks), batch_size), desc="Writing Chroma", unit="batch"):
        batch_chunks = chunks[start : start + batch_size]
        collection.add(
            ids=[chunk.chunk_id for chunk in batch_chunks],
            documents=[chunk.text for chunk in batch_chunks],
            metadatas=[chunk.metadata for chunk in batch_chunks],
            embeddings=final_embeddings[start : start + batch_size],
        )


def validate_chunks(chunks: list[Chunk]) -> None:
    if not chunks:
        raise RuntimeError("No chunks were generated.")
    for chunk in chunks:
        if not chunk.text.strip():
            raise RuntimeError(f"Empty chunk generated: {chunk.chunk_id}")
        tokens = int(chunk.metadata["token_count"])
        if tokens > HARD_MAX_CHUNK_TOKENS:
            raise RuntimeError(f"Chunk exceeds hard max tokens: {chunk.chunk_id} ({tokens})")
        for key in ("chunk_id", "title", "canonical_url"):
            if key not in chunk.metadata:
                raise RuntimeError(f"Missing metadata key {key} in {chunk.chunk_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build semantic chunks and Chroma embeddings for Facebook policy data.")
    parser.add_argument("--rebuild", action="store_true", help="Delete and rebuild the existing Chroma directory.")
    parser.add_argument("--dry-run", action="store_true", help="Parse and chunk only; do not call embeddings or write Chroma.")
    args = parser.parse_args()

    load_dotenv(ROOT_DIR / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env before building embeddings.")

    started = time.time()
    client = OpenAI()

    chunks = build_chunks(client)
    validate_chunks(chunks)
    write_chunks_jsonl(chunks)

    if not args.dry_run:
        persist_to_chroma(client, chunks, rebuild=args.rebuild)

    write_manifest(chunks, time.time() - started)
    print(f"Built {len(chunks)} chunks.")
    print(f"Wrote {CHUNKS_JSONL.relative_to(ROOT_DIR)}")
    print(f"Wrote {MANIFEST_JSON.relative_to(ROOT_DIR)}")
    if not args.dry_run:
        print(f"Wrote Chroma collection '{COLLECTION_NAME}' to {CHROMA_DIR.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
