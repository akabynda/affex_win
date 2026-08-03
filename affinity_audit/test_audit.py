import importlib.util
import math
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("audit.py")
SPEC = importlib.util.spec_from_file_location("affinity_audit_script", MODULE_PATH)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_dataset_dg_formula() -> None:
    kd = 1e-9
    expected = audit.RT_298 * math.log(kd)
    assert -12.3 < expected < -12.2


def test_parse_pdb_primary_citation() -> None:
    metadata = audit.parse_pdb_metadata("1a22")
    assert metadata["article"]["pmid"] == "9571026"
    assert metadata["article"]["doi"].lower() == "10.1006/jmbi.1998.1669"
    assert "GROWTH HORMONE" in metadata["title"]


def test_classification_match_and_conflict() -> None:
    record = {"dataset_kd_M": 1e-9}
    extraction = {
        "measurements": [
            {
                "measurement_type": "Kd",
                "value": 1.05,
                "unit": "nM",
                "relation": "=",
                "relevance": "direct",
            }
        ]
    }
    status, _, kd, delta = audit.classify_result(record, extraction)
    assert status == "match_candidate"
    assert kd == 1.05e-9
    assert delta is not None and delta < 0.1

    extraction["measurements"][0]["value"] = 100.0
    status, _, _, _ = audit.classify_result(record, extraction)
    assert status == "conflict_candidate"


def test_non_kd_is_not_converted() -> None:
    record = {"dataset_kd_M": 1e-9}
    extraction = {
        "measurements": [
            {
                "measurement_type": "IC50",
                "value": 1.0,
                "unit": "nM",
                "relation": "=",
                "relevance": "direct",
            }
        ]
    }
    status, selected, kd, delta = audit.classify_result(record, extraction)
    assert status == "needs_review"
    assert selected is None and kd is None and delta is None


def test_local_screen_requires_affinity_and_concentration() -> None:
    positive = {"article_text": "Binding affinity was measured by SPR and the Kd was 12 nM for the complex."}
    assert audit.local_affinity_contexts(positive)
    assert not audit.local_affinity_contexts({"article_text": "The proteins bind with high affinity."})
    assert not audit.local_affinity_contexts({"article_text": "The buffer contained 12 nM salt."})


def test_scientific_notation_sanity_downgrades_bad_extraction() -> None:
    quote = "the dissociation constant K D was 5.2 9 10 À11 M"
    candidates = audit.evidence_scientific_kd_candidates(quote)
    assert candidates == [5.2e-11]
    result = {
        "status": "conflict_candidate",
        "literature_kd_M": 5.2,
        "selected_measurement": {"evidence_quote": quote},
    }
    reviewed = audit.apply_evidence_sanity(result)
    assert reviewed["status"] == "needs_review"
    assert reviewed["original_status"] == "conflict_candidate"

    kinetic_quote = "association kon=1.2 ×105 M−1 s−1 and Kd=90 nM"
    assert audit.evidence_scientific_kd_candidates(kinetic_quote) == []
