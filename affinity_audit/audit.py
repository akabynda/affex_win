from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK_DIR = Path(__file__).resolve().parent / "work"
DEFAULT_MODEL = "google/gemini-2.5-flash"
RT_298 = 1.987204258e-3 * 298.0
USER_AGENT = "PCANN-affinity-audit/1.0 (literature evidence audit)"

DATASETS = (
    ("trainval", ROOT / "data/train/pcann-plus-trainval.csv"),
    ("testAB", ROOT / "data/test/testAB-clean.csv"),
    ("test_fabs", ROOT / "data/test/test-fabs.csv"),
)
PDB_DIR = ROOT / "data/raw/ppb-affinity/pdb"

UNIT_TO_M = {
    "m": 1.0,
    "mol/l": 1.0,
    "mm": 1e-3,
    "millimolar": 1e-3,
    "um": 1e-6,
    "μm": 1e-6,
    "µm": 1e-6,
    "micromolar": 1e-6,
    "nm": 1e-9,
    "nanomolar": 1e-9,
    "pm": 1e-12,
    "picomolar": 1e-12,
    "fm": 1e-15,
    "femtomolar": 1e-15,
}

AFFINITY_RE = re.compile(
    r"\b(?:k\s*[_-]?\s*d|dissociation constant|binding affinity|affinit|isothermal titration|"
    r"surface plasmon|biolayer interferometry|\bITC\b|\bSPR\b|\bBLI\b|nano\s*molar|micro\s*molar)",
    re.IGNORECASE,
)
CONCENTRATION_RE = re.compile(
    r"(?:[<>~≈]\s*)?\d+(?:\.\d+)?(?:\s*[x×]\s*10\s*\^?\s*[+-]?\d+)?\s*"
    r"(?:fM|pM|nM|[uµμ]M|mM|mol\s*/\s*[Ll]|M)\b",
    re.IGNORECASE,
)
SCIENTIFIC_CONCENTRATION_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?:[±+]\s*\d+(?:\.\d+)?)?\s*"
    r"(?:[x×]|9)\s*10\s*\^?\s*"
    r"(?P<sign>[+\-−–—À]?)\s*(?P<exponent>\d+)\s*"
    r"\(?(?P<unit>fM|pM|nM|[uµμ]M|mM|M)\)?"
    r"(?!\s*[\-−–—À]\s*1)",
    re.IGNORECASE,
)


def log(message: str) -> None:
    print(message, flush=True)


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safe_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def make_item_id(dataset: str, row_index: int, uid: str, receptor: str, ligand: str) -> str:
    readable = f"{dataset}-{row_index:04d}-{uid}-{receptor}-{ligand}"
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", readable)
    return cleaned[:180]


def build_manifest(work_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dataset_name, path in DATASETS:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row_index, row in enumerate(csv.DictReader(handle), start=1):
                uid = row["uid"].strip().lower()
                receptor = row["receptor_chains"].strip()
                ligand = row["ligand_chains"].strip()
                kd_m = float(row["KD"])
                dataset_dg = safe_float(row.get("dG"))
                computed_dg = RT_298 * math.log(kd_m)
                records.append(
                    {
                        "item_id": make_item_id(dataset_name, row_index, uid, receptor, ligand),
                        "dataset": dataset_name,
                        "dataset_subset": row.get("dataset", ""),
                        "row_index": row_index,
                        "uid": uid,
                        "receptor_chains": receptor,
                        "ligand_chains": ligand,
                        "dataset_kd_M": kd_m,
                        "dataset_dg_kcal_mol": dataset_dg,
                        "computed_dg_298_kcal_mol": computed_dg,
                        "dg_arithmetic_abs_error": (
                            abs(dataset_dg - computed_dg) if dataset_dg is not None else None
                        ),
                        "dataset_source": row.get("source", ""),
                        "dataset_method": row.get("method", ""),
                        "dataset_database": row.get("database", ""),
                    }
                )
    write_jsonl(work_dir / "manifest.jsonl", records)
    log(
        f"manifest: {len(records)} rows, "
        f"{len({row['uid'] for row in records})} unique PDB IDs -> {work_dir / 'manifest.jsonl'}"
    )
    return records


def continuation_value(lines: list[str], prefix: str, start: int) -> str:
    values: list[str] = []
    for line in lines:
        if not line.startswith(prefix):
            continue
        value = line[start:].strip()
        value = re.sub(r"^\d+\s+", "", value)
        values.append(value)
    return " ".join(values).strip()


def parse_pdb_metadata(uid: str) -> dict[str, Any]:
    path = PDB_DIR / f"{uid}.pdb"
    if not path.exists():
        return {"uid": uid, "error": f"missing local PDB file: {path}"}
    lines: list[str] = []
    with path.open(encoding="ascii", errors="replace") as handle:
        for line in handle:
            record = line[:6].strip()
            if record in {"HEADER", "TITLE", "COMPND", "JRNL", "EXPDTA"}:
                lines.append(line.rstrip("\r\n"))
            if record in {"ATOM", "HETATM"}:
                break

    journal: dict[str, list[str]] = {key: [] for key in ("AUTH", "TITL", "REF", "PMID", "DOI")}
    for line in lines:
        if not line.startswith("JRNL"):
            continue
        # Continuation records use a single sequence digit (for example ``TITL 2``).
        # PMID itself is numeric, so a generic ``\d+`` continuation pattern would
        # accidentally consume the identifier as if it were a sequence number.
        match = re.match(r"^JRNL\s+(AUTH|TITL|REF|PMID|DOI)(?=\s)\s*(?:[1-9]\s+)?(.*)$", line)
        if match:
            journal[match.group(1)].append(match.group(2).strip())

    header = next((line[10:].strip() for line in lines if line.startswith("HEADER")), "")
    return {
        "uid": uid,
        "pdb_url": f"https://www.rcsb.org/structure/{uid.upper()}",
        "header": header,
        "title": continuation_value(lines, "TITLE ", 10),
        "compound": continuation_value(lines, "COMPND", 10),
        "experimental_method": continuation_value(lines, "EXPDTA", 10),
        "article": {
            "authors": " ".join(journal["AUTH"]),
            "title": " ".join(journal["TITL"]),
            "reference": " ".join(journal["REF"]),
            "pmid": "".join(journal["PMID"]).strip(),
            "doi": "".join(journal["DOI"]).strip(),
        },
    }


def request_json(url: str, *, timeout: int = 45) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def request_text(url: str, *, timeout: int = 60) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/xml,text/xml"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def europe_pmc_record(article: dict[str, str]) -> dict[str, Any] | None:
    pmid = article.get("pmid", "").strip()
    doi = article.get("doi", "").strip()
    if pmid:
        query = f"EXT_ID:{pmid} AND SRC:MED"
    elif doi:
        query = f'DOI:"{doi}"'
    else:
        return None
    params = urllib.parse.urlencode({"query": query, "format": "json", "resultType": "core", "pageSize": 3})
    payload = request_json(f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?{params}")
    results = payload.get("resultList", {}).get("result", [])
    if not results:
        return None
    if doi:
        for result in results:
            if str(result.get("doi", "")).lower() == doi.lower():
                return result
    return results[0]


def xml_to_relevant_text(xml_text: str, max_chars: int = 50000) -> str:
    root = ET.fromstring(xml_text)
    text = " ".join(part.strip() for part in root.itertext() if part and part.strip())
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_chars:
        return text

    ranges: list[tuple[int, int]] = [(0, min(7000, len(text)))]
    for match in AFFINITY_RE.finditer(text):
        ranges.append((max(0, match.start() - 1400), min(len(text), match.end() + 2200)))
    ranges.sort()
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1] + 100:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    excerpts: list[str] = []
    used = 0
    for start, end in merged:
        excerpt = text[start:end]
        remaining = max_chars - used
        if remaining <= 0:
            break
        excerpts.append(excerpt[:remaining])
        used += min(len(excerpt), remaining)
    return "\n[... excerpt boundary ...]\n".join(excerpts)


def collect_one_source(uid: str, work_dir: Path, refresh: bool = False) -> tuple[str, str]:
    output_path = work_dir / "sources" / f"{uid}.json"
    if output_path.exists() and not refresh:
        return uid, "cached"
    metadata = parse_pdb_metadata(uid)
    source: dict[str, Any] = {
        "uid": uid,
        "pdb": metadata,
        "europe_pmc": None,
        "article_text": "",
        "article_text_kind": "none",
        "source_urls": [metadata.get("pdb_url", "")],
        "retrieval_error": None,
    }
    article = metadata.get("article", {})
    try:
        epmc = europe_pmc_record(article)
        if epmc:
            keep_fields = (
                "id",
                "source",
                "pmid",
                "pmcid",
                "doi",
                "title",
                "authorString",
                "journalTitle",
                "pubYear",
                "abstractText",
                "isOpenAccess",
                "inEPMC",
                "inPMC",
            )
            source["europe_pmc"] = {key: epmc.get(key) for key in keep_fields}
            pmid = str(epmc.get("pmid") or article.get("pmid") or "")
            if pmid:
                source["source_urls"].append(f"https://europepmc.org/article/MED/{pmid}")
            pmcid = str(epmc.get("pmcid") or "")
            if pmcid:
                fulltext_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
                try:
                    xml_text = request_text(fulltext_url)
                    source["article_text"] = xml_to_relevant_text(xml_text)
                    source["article_text_kind"] = "open_access_fulltext_relevant_excerpts"
                    source["source_urls"].append(f"https://europepmc.org/articles/{pmcid}")
                except (urllib.error.URLError, TimeoutError, ET.ParseError) as exc:
                    source["retrieval_error"] = f"full text: {type(exc).__name__}: {exc}"
            if not source["article_text"] and epmc.get("abstractText"):
                source["article_text"] = str(epmc["abstractText"])
                source["article_text_kind"] = "abstract"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        source["retrieval_error"] = f"Europe PMC: {type(exc).__name__}: {exc}"
    source["source_urls"] = [url for url in dict.fromkeys(source["source_urls"]) if url]
    write_json(output_path, source)
    return uid, source["article_text_kind"]


def collect_sources(
    records: list[dict[str, Any]], work_dir: Path, workers: int, refresh: bool = False
) -> None:
    uids = sorted({row["uid"] for row in records})
    counts: dict[str, int] = {}
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(collect_one_source, uid, work_dir, refresh): uid for uid in uids}
        for future in as_completed(futures):
            uid = futures[future]
            try:
                _, kind = future.result()
            except Exception as exc:  # Keep the bulk collection resumable.
                kind = "error"
                write_json(
                    work_dir / "sources" / f"{uid}.json",
                    {"uid": uid, "retrieval_error": f"{type(exc).__name__}: {exc}"},
                )
            counts[kind] = counts.get(kind, 0) + 1
            completed += 1
            if completed % 100 == 0 or completed == len(uids):
                log(f"sources: {completed}/{len(uids)} {counts}")


def local_affinity_contexts(source: dict[str, Any], max_contexts: int = 20) -> list[str]:
    """Find text windows containing both affinity language and a concentration.

    This is a high-recall, zero-cost pre-screen, not scientific extraction. It is used
    to avoid paying an LLM to read articles that contain no visible numeric candidate.
    """
    text = str(source.get("article_text") or "")
    if not text:
        return []
    contexts: list[str] = []
    for affinity_match in AFFINITY_RE.finditer(text):
        start = max(0, affinity_match.start() - 350)
        end = min(len(text), affinity_match.end() + 700)
        window = text[start:end]
        if not CONCENTRATION_RE.search(window):
            continue
        compact = re.sub(r"\s+", " ", window).strip()
        if compact not in contexts:
            contexts.append(compact)
        if len(contexts) >= max_contexts:
            break
    return contexts


def screen_sources(records: list[dict[str, Any]], work_dir: Path) -> dict[str, Any]:
    uid_rows: list[dict[str, Any]] = []
    positive_uids: set[str] = set()
    for uid in sorted({row["uid"] for row in records}):
        source_path = work_dir / "sources" / f"{uid}.json"
        if not source_path.exists():
            uid_rows.append({"uid": uid, "source_text_kind": "missing", "candidate_context_count": 0})
            continue
        source = read_json(source_path)
        contexts = local_affinity_contexts(source)
        if contexts:
            positive_uids.add(uid)
        uid_rows.append(
            {
                "uid": uid,
                "source_text_kind": source.get("article_text_kind", "none"),
                "candidate_context_count": len(contexts),
                "candidate_contexts": contexts,
            }
        )
    write_jsonl(work_dir / "local_screen.jsonl", uid_rows)
    kind_counts: dict[str, int] = {}
    for row in uid_rows:
        kind = str(row["source_text_kind"])
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    summary = {
        "unique_pdb_ids": len(uid_rows),
        "positive_pdb_ids": len(positive_uids),
        "dataset_rows_with_local_candidate": sum(row["uid"] in positive_uids for row in records),
        "source_text_kind_counts": kind_counts,
    }
    write_json(work_dir / "local_screen_summary.json", summary)
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def local_candidate_uids(work_dir: Path) -> set[str]:
    path = work_dir / "local_screen.jsonl"
    if not path.exists():
        return set()
    return {row["uid"] for row in read_jsonl(path) if int(row.get("candidate_context_count", 0)) > 0}


EXTRACTION_SCHEMA: dict[str, Any] = {
    "name": "protein_affinity_evidence",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "article_supports_target_interaction": {"type": "string", "enum": ["yes", "no", "uncertain"]},
            "measurements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "measurement_type": {
                            "type": "string",
                            "enum": ["Kd", "Ka", "Ki", "IC50", "EC50", "kon_koff", "other"],
                        },
                        "relation": {"type": "string", "enum": ["=", "<", ">", "~", "range"]},
                        "value": {"type": ["number", "null"]},
                        "lower_value": {"type": ["number", "null"]},
                        "upper_value": {"type": ["number", "null"]},
                        "unit": {"type": ["string", "null"]},
                        "temperature_K": {"type": ["number", "null"]},
                        "method": {"type": ["string", "null"]},
                        "molecule_1": {"type": ["string", "null"]},
                        "molecule_2": {"type": ["string", "null"]},
                        "construct_and_mutations": {"type": ["string", "null"]},
                        "relevance": {
                            "type": "string",
                            "enum": ["direct", "possible", "different_construct", "unrelated"],
                        },
                        "evidence_quote": {"type": ["string", "null"]},
                        "evidence_location": {"type": ["string", "null"]},
                        "source_url": {"type": ["string", "null"]},
                        "notes": {"type": ["string", "null"]},
                    },
                    "required": [
                        "measurement_type",
                        "relation",
                        "value",
                        "lower_value",
                        "upper_value",
                        "unit",
                        "temperature_K",
                        "method",
                        "molecule_1",
                        "molecule_2",
                        "construct_and_mutations",
                        "relevance",
                        "evidence_quote",
                        "evidence_location",
                        "source_url",
                        "notes",
                    ],
                    "additionalProperties": False,
                },
            },
            "search_summary": {"type": ["string", "null"]},
            "review_notes": {"type": ["string", "null"]},
        },
        "required": ["article_supports_target_interaction", "measurements", "search_summary", "review_notes"],
        "additionalProperties": False,
    },
}


SYSTEM_PROMPT = """You extract experimental protein-protein binding-affinity evidence for a scientific data audit.
Be conservative. Never infer or invent a numeric measurement. A PDB primary citation can describe a structure without
reporting affinity. Extract only numbers that are explicitly present in a source. Keep Kd, Ka, Ki, IC50 and EC50 distinct.
Do not convert units and do not calculate values. A measurement is direct only when it concerns the same interacting
proteins and apparently the same construct/mutation state as the target PDB complex. Include a short verbatim evidence
quote and a source URL. If the supplied text is insufficient and web search is available, search by PDB ID, DOI, article
title, protein names, and affinity terms. Return only the JSON object required by the response schema."""


def article_prompt(record: dict[str, Any], source: dict[str, Any]) -> str:
    pdb = source.get("pdb", {})
    article = pdb.get("article", {}) if isinstance(pdb, dict) else {}
    text = source.get("article_text", "")
    if len(text) > 50000:
        text = text[:50000]
    return f"""TARGET PDB COMPLEX
PDB ID: {record['uid'].upper()}
Receptor chains: {record['receptor_chains']}
Ligand chains: {record['ligand_chains']}
PDB title: {pdb.get('title', '')}
PDB compound description: {pdb.get('compound', '')}

PRIMARY CITATION FROM PDB
Title: {article.get('title', '')}
DOI: {article.get('doi', '')}
PMID: {article.get('pmid', '')}
Reference: {article.get('reference', '')}
Known source URLs: {json.dumps(source.get('source_urls', []), ensure_ascii=False)}

AVAILABLE ARTICLE MATERIAL ({source.get('article_text_kind', 'none')})
{text if text else '[No article text was retrievable locally. Use web search if available.]'}

TASK
Find every explicitly reported affinity measurement that may correspond to this target interaction. Do not use or guess
the value from PDBbind, SAbDab, SKEMPI, PCANN, benchmark datasets, or prediction pages as literature evidence. Prefer the
primary paper, its tables, figures, methods, and supplements. Return an empty measurements list if no explicit value is
available."""


def post_openrouter(api_key: str, payload: dict[str, Any], timeout: int = 180) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/",
            "X-Title": "PCANN literature affinity audit",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {body[:1000]}") from exc


def get_openrouter_credits(api_key: str) -> dict[str, Any]:
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/credits",
        headers={"Authorization": f"Bearer {api_key}", "User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter credits HTTP {exc.code}: {body[:500]}") from exc


def parse_model_json(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        text = "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)
    else:
        text = str(content or "")
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def normalize_unit(unit: Any) -> float | None:
    if unit is None:
        return None
    normalized = str(unit).strip().lower().replace(" ", "")
    normalized = normalized.replace("molar", "m") if normalized == "molar" else normalized
    return UNIT_TO_M.get(normalized)


def evidence_scientific_kd_candidates(quote: Any) -> list[float]:
    if not quote:
        return []
    candidates: list[float] = []
    for match in SCIENTIFIC_CONCENTRATION_RE.finditer(str(quote)):
        multiplier = normalize_unit(match.group("unit"))
        if multiplier is None:
            continue
        exponent = int(match.group("exponent"))
        if match.group("sign") in {"-", "−", "–", "—", "À"}:
            exponent = -exponent
        candidates.append(float(match.group("value")) * (10.0**exponent) * multiplier)
    return candidates


def apply_evidence_sanity(result: dict[str, Any]) -> dict[str, Any]:
    """Downgrade candidates when model JSON contradicts scientific notation in its quote."""
    selected = result.get("selected_measurement") or {}
    literature_kd = safe_float(result.get("literature_kd_M"))
    candidates = evidence_scientific_kd_candidates(selected.get("evidence_quote"))
    reviewed = dict(result)
    reviewed["original_status"] = result.get("status")
    reviewed["evidence_scientific_kd_candidates_M"] = candidates
    if not literature_kd or not candidates:
        reviewed["evidence_numeric_sanity"] = "not_applicable"
        return reviewed
    closest_delta = min(abs(math.log10(candidate / literature_kd)) for candidate in candidates if candidate > 0)
    if closest_delta > 0.15:
        reviewed["evidence_numeric_sanity"] = "inconsistent_with_scientific_notation_quote"
        reviewed["status"] = "needs_review"
    else:
        reviewed["evidence_numeric_sanity"] = "consistent"
    return reviewed


def select_measurement(extraction: dict[str, Any]) -> tuple[dict[str, Any] | None, float | None]:
    ranked: list[tuple[int, dict[str, Any], float]] = []
    relevance_rank = {"direct": 0, "possible": 1, "different_construct": 2, "unrelated": 3}
    for measurement in extraction.get("measurements", []):
        if measurement.get("measurement_type") != "Kd":
            continue
        value = safe_float(measurement.get("value"))
        multiplier = normalize_unit(measurement.get("unit"))
        if value is None or multiplier is None or value <= 0:
            continue
        kd_m = value * multiplier
        ranked.append((relevance_rank.get(measurement.get("relevance"), 9), measurement, kd_m))
    if not ranked:
        return None, None
    ranked.sort(key=lambda item: item[0])
    return ranked[0][1], ranked[0][2]


def classify_result(
    record: dict[str, Any], extraction: dict[str, Any]
) -> tuple[str, dict[str, Any] | None, float | None, float | None]:
    measurements = extraction.get("measurements", [])
    selected, literature_kd = select_measurement(extraction)
    if selected is None or literature_kd is None:
        if not measurements:
            return "not_found", None, None, None
        if any(item.get("relevance") in {"direct", "possible"} for item in measurements):
            return "needs_review", None, None, None
        return "not_comparable", None, None, None
    if selected.get("relevance") != "direct" or selected.get("relation") not in {"=", "~"}:
        return "needs_review", selected, literature_kd, None
    delta = abs(math.log10(literature_kd / float(record["dataset_kd_M"])))
    if delta <= 0.1:
        status = "match_candidate"
    elif delta <= 0.3:
        status = "approx_match_candidate"
    else:
        status = "conflict_candidate"
    return status, selected, literature_kd, delta


def result_path(work_dir: Path, item_id: str) -> Path:
    return work_dir / "results" / f"{item_id}.json"


def audit_one(
    record: dict[str, Any],
    work_dir: Path,
    api_key: str,
    model: str,
    web_search: bool,
    retry_errors: bool,
) -> tuple[str, str, float]:
    output_path = result_path(work_dir, record["item_id"])
    if output_path.exists():
        existing = read_json(output_path)
        if existing.get("status") != "error" or not retry_errors:
            return record["item_id"], "cached", 0.0
    source_path = work_dir / "sources" / f"{record['uid']}.json"
    if not source_path.exists():
        result = {**record, "status": "error", "error": f"missing source cache: {source_path}"}
        write_json(output_path, result)
        return record["item_id"], "error", 0.0
    source = read_json(source_path)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": article_prompt(record, source)},
        ],
        "temperature": 0,
        "max_tokens": 2500,
        "response_format": {"type": "json_schema", "json_schema": EXTRACTION_SCHEMA},
        "provider": {"require_parameters": True},
    }
    if web_search:
        # OpenRouter's server-tool endpoint currently returns 404 for some account/provider
        # combinations. The request-level web plugin is retained as a compatibility path;
        # raw responses and URL annotations are persisted for provenance.
        payload["plugins"] = [{"id": "web", "max_results": 5}]
    last_error: Exception | None = None
    started = time.time()
    for attempt in range(1, 4):
        try:
            response = post_openrouter(api_key, payload)
            raw_path = work_dir / "raw" / f"{record['item_id']}.json"
            write_json(raw_path, response)
            choice = response["choices"][0]
            message = choice["message"]
            extraction = parse_model_json(message.get("content"))
            status, selected, literature_kd, delta = classify_result(record, extraction)
            temperature = safe_float(selected.get("temperature_K")) if selected else None
            temperature = temperature or 298.0
            result = {
                **record,
                "status": status,
                "model": response.get("model", model),
                "requested_model": model,
                "web_search_enabled": web_search,
                "source_text_kind": source.get("article_text_kind", "none"),
                "pdb_article": source.get("pdb", {}).get("article", {}),
                "source_urls": source.get("source_urls", []),
                "extraction": extraction,
                "selected_measurement": selected,
                "literature_kd_M": literature_kd,
                "literature_dg_kcal_mol": (
                    1.987204258e-3 * temperature * math.log(literature_kd) if literature_kd else None
                ),
                "delta_log10_kd": delta,
                "kd_ratio_literature_over_dataset": (
                    literature_kd / float(record["dataset_kd_M"]) if literature_kd else None
                ),
                "annotations": message.get("annotations", []),
                "usage": response.get("usage", {}),
                "elapsed_seconds": time.time() - started,
            }
            write_json(output_path, result)
            return record["item_id"], status, safe_float(response.get("usage", {}).get("cost")) or 0.0
        except Exception as exc:  # Retry transient HTTP and provider/parse failures.
            last_error = exc
            if attempt < 3:
                time.sleep(2**attempt)
    result = {
        **record,
        "status": "error",
        "model": model,
        "web_search_enabled": web_search,
        "error": f"{type(last_error).__name__}: {last_error}",
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_path, result)
    return record["item_id"], "error", 0.0


def filter_records(records: list[dict[str, Any]], only_uid: str | None) -> list[dict[str, Any]]:
    selected = records
    if only_uid:
        selected = [row for row in selected if row["uid"] == only_uid.lower()]
    return selected


def run_audit(
    records: list[dict[str, Any]],
    work_dir: Path,
    workers: int,
    model: str,
    web_search: bool,
    retry_errors: bool,
    max_new_cost: float,
) -> None:
    load_env(ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing from the environment and repository .env")
    if max_new_cost == 0 and model != "openrouter/free" and not model.endswith(":free"):
        raise ValueError("--max-new-cost 0 is allowed only with openrouter/free or a :free model")
    counts: dict[str, int] = {}
    completed = 0
    new_cost = 0.0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for batch_start in range(0, len(records), workers):
            if max_new_cost > 0 and new_cost >= max_new_cost:
                log(
                    f"cost guard reached: new reported cost ${new_cost:.6f} >= "
                    f"limit ${max_new_cost:.6f}; {len(records) - completed} rows not submitted"
                )
                break
            batch = records[batch_start : batch_start + workers]
            futures = {
                executor.submit(audit_one, row, work_dir, api_key, model, web_search, retry_errors): row
                for row in batch
            }
            for future in as_completed(futures):
                try:
                    _, status, cost = future.result()
                    new_cost += cost
                except Exception as exc:
                    status = "worker_error"
                    log(f"worker error: {type(exc).__name__}: {exc}")
                counts[status] = counts.get(status, 0) + 1
                completed += 1
            if completed % 10 == 0 or completed == len(records):
                log(f"audit: {completed}/{len(records)} new_cost=${new_cost:.6f} {counts}")


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    article = result.get("pdb_article") or {}
    selected = result.get("selected_measurement") or {}
    extraction = result.get("extraction") or {}
    usage = result.get("usage") or {}
    return {
        "item_id": result.get("item_id"),
        "dataset": result.get("dataset"),
        "dataset_subset": result.get("dataset_subset"),
        "row_index": result.get("row_index"),
        "uid": result.get("uid"),
        "receptor_chains": result.get("receptor_chains"),
        "ligand_chains": result.get("ligand_chains"),
        "dataset_source": result.get("dataset_source"),
        "dataset_kd_M": result.get("dataset_kd_M"),
        "dataset_dg_kcal_mol": result.get("dataset_dg_kcal_mol"),
        "status": result.get("status"),
        "original_status": result.get("original_status", result.get("status")),
        "evidence_numeric_sanity": result.get("evidence_numeric_sanity"),
        "evidence_scientific_kd_candidates_M": json.dumps(
            result.get("evidence_scientific_kd_candidates_M", []), ensure_ascii=False
        ),
        "literature_kd_M": result.get("literature_kd_M"),
        "literature_dg_kcal_mol": result.get("literature_dg_kcal_mol"),
        "delta_log10_kd": result.get("delta_log10_kd"),
        "kd_ratio_literature_over_dataset": result.get("kd_ratio_literature_over_dataset"),
        "measurement_type": selected.get("measurement_type"),
        "reported_relation": selected.get("relation"),
        "reported_value": selected.get("value"),
        "reported_unit": selected.get("unit"),
        "temperature_K": selected.get("temperature_K"),
        "method": selected.get("method"),
        "construct_and_mutations": selected.get("construct_and_mutations"),
        "relevance": selected.get("relevance"),
        "evidence_quote": selected.get("evidence_quote"),
        "evidence_location": selected.get("evidence_location"),
        "evidence_url": selected.get("source_url"),
        "article_title": article.get("title"),
        "article_doi": article.get("doi"),
        "article_pmid": article.get("pmid"),
        "source_text_kind": result.get("source_text_kind"),
        "article_supports_target": extraction.get("article_supports_target_interaction"),
        "review_notes": extraction.get("review_notes"),
        "model": result.get("model"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cost": usage.get("cost"),
        "error": result.get("error"),
    }


def summarize(work_dir: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    screen_by_uid = {
        row["uid"]: row
        for row in read_jsonl(work_dir / "local_screen.jsonl")
    } if (work_dir / "local_screen.jsonl").exists() else {}
    for record in records:
        path = result_path(work_dir, record["item_id"])
        if path.exists():
            result = apply_evidence_sanity(read_json(path))
        else:
            screen = screen_by_uid.get(record["uid"], {})
            source_kind = screen.get("source_text_kind", "missing")
            if int(screen.get("candidate_context_count", 0)) > 0:
                status = "queued_local_candidate"
            elif source_kind in {"none", "missing"}:
                status = "no_retrievable_article_text"
            else:
                status = "no_numeric_candidate_in_retrieved_text"
            result = {**record, "status": status, "source_text_kind": source_kind}
        flat = flatten_result(result)
        rows.append(flat)
        status = str(flat["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        total_cost += safe_float(flat.get("cost")) or 0.0
        total_prompt_tokens += int(safe_float(flat.get("prompt_tokens")) or 0)
        total_completion_tokens += int(safe_float(flat.get("completion_tokens")) or 0)

    output_csv = work_dir / "affinity_audit.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "total_dataset_rows": len(records),
        "status_counts": status_counts,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "reported_openrouter_cost": total_cost,
        "output_csv": str(output_csv),
    }
    write_json(work_dir / "summary.json", summary)
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def get_records(work_dir: Path, rebuild: bool = False) -> list[dict[str, Any]]:
    manifest_path = work_dir / "manifest.jsonl"
    if rebuild or not manifest_path.exists():
        return build_manifest(work_dir)
    return read_jsonl(manifest_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit PCANN Kd values against literature linked from PDB entries")
    parser.add_argument("command", choices=("manifest", "sources", "screen", "run", "summarize", "credits", "all"))
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only-uid")
    parser.add_argument("--refresh-sources", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument(
        "--max-new-cost",
        type=float,
        default=1.0,
        help="hard guard on newly reported OpenRouter cost for this run (default: $1.00; one worker batch may overshoot)",
    )
    parser.add_argument(
        "--local-candidates-only",
        action="store_true",
        help="call the model only for rows whose retrieved article text passes the zero-cost numeric affinity screen",
    )
    state_group = parser.add_mutually_exclusive_group()
    state_group.add_argument(
        "--unprocessed-only",
        action="store_true",
        help="select only rows without an existing result JSON (useful with free-model daily limits)",
    )
    state_group.add_argument(
        "--errors-only",
        action="store_true",
        help="select only rows whose existing result has status=error",
    )
    search_group = parser.add_mutually_exclusive_group()
    search_group.add_argument("--web-search", dest="web_search", action="store_true")
    search_group.add_argument("--no-web-search", dest="web_search", action="store_false")
    parser.set_defaults(web_search=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.command == "credits":
        load_env(ROOT / ".env")
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing")
        payload = get_openrouter_credits(api_key)
        data = payload.get("data", payload)
        total_credits = safe_float(data.get("total_credits"))
        total_usage = safe_float(data.get("total_usage"))
        safe_output = {
            "total_credits": total_credits,
            "total_usage": total_usage,
            "remaining_credits": (
                total_credits - total_usage if total_credits is not None and total_usage is not None else None
            ),
        }
        log(json.dumps(safe_output, ensure_ascii=False, indent=2))
        return 0
    if args.command == "manifest":
        build_manifest(work_dir)
        return 0

    records = get_records(work_dir)
    selected = filter_records(records, args.only_uid)
    if args.command in {"sources", "all"}:
        collect_sources(selected, work_dir, args.workers, args.refresh_sources)
    if args.command in {"screen", "all"} or args.local_candidates_only:
        screen_sources(records, work_dir)
    if args.local_candidates_only:
        positive_uids = local_candidate_uids(work_dir)
        selected = [row for row in selected if row["uid"] in positive_uids]
        log(f"local candidate filter: {len(selected)} rows selected")
    if args.unprocessed_only:
        selected = [row for row in selected if not result_path(work_dir, row["item_id"]).exists()]
        log(f"unprocessed filter: {len(selected)} rows selected")
    if args.errors_only:
        selected = [
            row
            for row in selected
            if result_path(work_dir, row["item_id"]).exists()
            and read_json(result_path(work_dir, row["item_id"])).get("status") == "error"
        ]
        log(f"error filter: {len(selected)} rows selected")
    if args.limit is not None:
        selected = selected[: args.limit]
    if args.command in {"run", "all"}:
        missing_sources = [row["uid"] for row in selected if not (work_dir / "sources" / f"{row['uid']}.json").exists()]
        if missing_sources:
            log(f"collecting {len(set(missing_sources))} missing source records first")
            collect_sources(selected, work_dir, args.workers, False)
        log(f"model={args.model} rows={len(selected)} workers={args.workers} web_search={args.web_search}")
        run_audit(
            selected,
            work_dir,
            args.workers,
            args.model,
            args.web_search,
            args.retry_errors or args.errors_only,
            args.max_new_cost,
        )
    if args.command in {"summarize", "all"}:
        summarize(work_dir, records)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("interrupted; completed source/result files are cached and the command can be resumed")
        raise SystemExit(130)
