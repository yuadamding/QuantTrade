"""LSF-only regressions: issuer identity never becomes issue continuity."""

import gzip
import json

import pytest

from rl_quant.data_sources.massive import qt200_reference_supplement_v1 as supplement
from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.data_sources.massive.qt200_supplement_reconciliation_v1 import (
    compare_issue,
    economic_problems,
    reconcile_supplement,
)
from test_qt200_reference_supplement_v1 import _network, _plan


@pytest.mark.parametrize("observed,expected,classification", [
    ({"cik": "same", "name": "same"}, {"current_issuer_cik": "same"}, "unknown_issue_identity"),
    ({"composite_figi": "A", "share_class_figi": "B"},
     {"current_composite_figi": "A", "current_share_class_figi": "C"}, "conflicting_issue_identifiers"),
    ({"composite_figi": "A"}, {"current_composite_figi": "A"}, "exact_date_issue_identifiers_match_only"),
    ({"composite_figi": ""}, {"current_composite_figi": ""}, "unknown_issue_identity"),
])
def test_issue_comparison_is_not_continuity(observed, expected, classification):
    result = compare_issue(observed, expected)
    assert result["classification"] == classification
    assert result["continuous_interval_inferred"] is result["training_eligible"] is False


@pytest.mark.parametrize("value", [0, -1, True, None, "NaN", "Infinity", "bad"])
def test_unsupported_economic_quantity_flagged(value):
    assert "invalid_split_to" in economic_problems("splits", {"ticker": "AAPL", "split_from": 1, "split_to": value})


def test_currency_and_dates_not_fabricated():
    result = economic_problems("dividends", {"ticker": "AAPL", "cash_amount": 0.2})
    assert "non_usd_or_absent_currency" in result
    assert "missing_or_unclassified_pay_date" in result


def test_real_persistence_path_keeps_empty_inputs_unqualified(tmp_path, monkeypatch):
    root, evidence, evidence_sha, plan_sha, queries = _plan(tmp_path)
    calls = _network(monkeypatch, queries)
    supplement.capture_supplement(root=root, api_key="synthetic-test-only", plan_sha256=plan_sha,
                                  evidence=evidence, evidence_sha256=evidence_sha)
    complete_sha = transport.digest((root / "COMPLETE.json").read_bytes())
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    output = tmp_path / "reconciled"
    result = reconcile_supplement(root=root, plan_sha256=plan_sha, completion_sha256=complete_sha,
        evidence=evidence, evidence_sha256=evidence_sha, output=output)
    assert result["queries"] == result["empty_queries"] == len(calls) == 340
    assert result["training_ready"] is result["complete_issue_history_qualified"] is False
    assert result["alias_bar_rows"] == result["economic_rows"] == result["dated_reference_rows"] == 0
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    issues = json.loads(gzip.decompress((output / "coverage-and-issues.json.gz").read_bytes()))
    assert len(issues["empty_queries"]) == 340
    with pytest.raises(FileExistsError):
        reconcile_supplement(root=root, plan_sha256=plan_sha, completion_sha256=complete_sha,
            evidence=evidence, evidence_sha256=evidence_sha, output=output)
    path = root / queries[0].name / "page-0000.json.gz"
    path.chmod(0o600)
    path.write_bytes(b"corrupt")
    with pytest.raises(transport.ResearchCaptureError):
        reconcile_supplement(root=root, plan_sha256=plan_sha, completion_sha256=complete_sha,
            evidence=evidence, evidence_sha256=evidence_sha, output=tmp_path / "tampered")
    assert not (tmp_path / "tampered").exists()
