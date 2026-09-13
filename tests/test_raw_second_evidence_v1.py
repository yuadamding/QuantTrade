"""LSF-only source relocation regressions; all responses are synthetic."""

from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil

import pytest
import torch

from rl_quant.data_sources.massive import raw_second_direct_capture_v1 as direct
from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.data_sources.massive.raw_second_evidence_v1 import EvidenceRelocation, MAP_NAME
from test_raw_second_alias_capture_v1 import _route, _supplement
from test_raw_second_direct_capture_v1 import KEY, _body, _inventory, _network

pytestmark = pytest.mark.lsf_gpu
FLAGS = dict(training_ready=False, training_authorized=False, remote_native_replay_qualified=False,
             native_catalog_activated=False, original_references_rewritten=False)


@pytest.fixture(autouse=True, scope="module")
def assigned_gpu():
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1


def _fixture(tmp_path, monkeypatch, *, supplement=False, old_plan=False):
    original = tmp_path / "original"
    original.mkdir()
    route, query = _route(original)
    if supplement:
        route = _supplement(original, route)
    calls = _network(monkeypatch, [(query.url, 200, _body(query))])
    root = original / "capture"
    implementations = direct._implementations
    if old_plan:
        monkeypatch.setattr(direct, "_implementations", lambda: {
            k: v for k, v in implementations().items() if k != "evidence_implementation_sha256"})
    plan = direct.publish_direct_second_capture_plan(root=root, queries=(query,), alias_routes=(route,))
    direct.capture_direct_seconds(root=root, plan_sha256=plan, api_key=KEY)
    completion = sha256((root / "COMPLETE.json").read_bytes()).hexdigest()
    monkeypatch.setattr(direct, "_implementations", implementations)
    package = tmp_path / "package"
    shutil.copytree(root, package / "capture")
    inputs = [Path(route.event.path)]
    if supplement:
        inputs.extend(sorted(Path(route.identity.path).iterdir()))
    else:
        inputs.append(Path(route.identity.path))
    mapping = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            name = "capture/" + path.relative_to(root).as_posix()
            mapping.append(dict(original_path=str(path), published_path=name,
                                bytes=path.stat().st_size, sha256=sha256(path.read_bytes()).hexdigest()))
    for path in inputs:
        name = "evidence/" + path.relative_to(original).as_posix()
        destination = package / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        mapping.append(dict(original_path=str(path), published_path=name,
                            bytes=path.stat().st_size, sha256=sha256(path.read_bytes()).hexdigest()))
    resolver = _seal(package, mapping, plan, completion)
    # Remove original *locations*, preserving every original byte in another
    # test-owned directory. Neither replay may write or restore them.
    original.rename(tmp_path / "retained-original")
    monkeypatch.setattr(transport, "_fetch", lambda *_a, **_k: pytest.fail("Replay attempted HTTP"))
    return resolver, route, calls


def _seal(package, mapping, plan, completion):
    map_body = transport.canonical(dict(files=mapping, native_path_resolution_authorized=False, **FLAGS))
    path = package / MAP_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(map_body)
    rows = [dict(path=p.relative_to(package).as_posix(), bytes=p.stat().st_size,
                 sha256=sha256(p.read_bytes()).hexdigest()) for p in sorted(package.rglob("*"))
            if p.is_file() and p != package / "inventory.json"]
    inventory = transport.canonical(dict(schema="rl-quant.raw-second-evidence-package-v1", files=rows, **FLAGS))
    (package / "inventory.json").write_bytes(inventory)
    return EvidenceRelocation(str(package), sha256(inventory).hexdigest(), sha256(map_body).hexdigest(), plan, completion)


def _replay(resolver):
    return direct.replay_direct_second_capture(root=Path(resolver.package_root) / "capture",
        plan_sha256=resolver.plan_sha256, completion_sha256=resolver.completion_sha256,
        evidence_relocation=resolver)


@pytest.mark.parametrize("supplement", [False, True])
@pytest.mark.parametrize("old_plan", [False, True])
def test_explicit_relocation_replays_original_hashes_and_never_writes(tmp_path, monkeypatch, supplement, old_plan):
    resolver, route, calls = _fixture(tmp_path, monkeypatch, supplement=supplement, old_plan=old_plan)
    root = Path(resolver.package_root) / "capture"
    original_plan = (root / "plan.json").read_bytes()
    with pytest.raises(OSError):
        direct.replay_direct_second_capture(root=root, plan_sha256=resolver.plan_sha256,
                                            completion_sha256=resolver.completion_sha256)
    before = _inventory(tmp_path)
    result = _replay(resolver)
    assert result["plan_sha256"] == resolver.plan_sha256 and result["capture_sha256"] == resolver.completion_sha256
    assert result["query_count"] == 1 and len(calls) == 1 and result["read_only_replay_verified"]
    assert result["evidence_relocation"]["exact_bytes_verified"] and not result["training_ready"]
    assert json.loads(original_plan)["alias_routes"] == [asdict(route)]
    assert result["source_provenance"]["implementations"] == {
        k: v for k, v in json.loads(original_plan).items() if k.endswith("_implementation_sha256")}
    assert (root / "plan.json").read_bytes() == original_plan and _inventory(tmp_path) == before


@pytest.mark.parametrize("case", ["absolute", "traversal", "double-slash", "nul", "duplicate-original",
                                   "duplicate-destination", "missing", "wrong-hash", "wrong-size",
                                   "wrong-capture", "nested-source"])
def test_rehashed_mapping_cannot_redirect_or_omit_evidence(tmp_path, monkeypatch, case):
    resolver, route, _ = _fixture(tmp_path, monkeypatch, supplement=case == "nested-source")
    package = Path(resolver.package_root)
    mapping = json.loads((package / MAP_NAME).read_bytes())["files"]
    row = next(r for r in mapping if r["original_path"] == route.event.path)
    if case in ("absolute", "traversal", "double-slash", "nul"):
        row["published_path"] = {"absolute": "/outside", "traversal": "evidence/../outside",
                                  "double-slash": "evidence//outside", "nul": "evidence/\x00"}[case]
    elif case == "duplicate-original":
        mapping.append(dict(row))
    elif case == "duplicate-destination":
        mapping.append(dict(row, original_path="/unrelated/original.json"))
    elif case == "missing":
        mapping.remove(row)
    elif case == "wrong-hash":
        row["sha256"] = "0" * 64
    elif case == "wrong-size":
        row["bytes"] = True
    elif case == "wrong-capture":
        next(r for r in mapping if r["published_path"] == "capture/COMPLETE.json")["original_path"] = "/another/capture/COMPLETE.json"
    else:
        row["original_path"] = route.identity.path + "/nested/event.json"
    resolver = _seal(package, mapping, resolver.plan_sha256, resolver.completion_sha256)
    before = _inventory(tmp_path)
    with pytest.raises(ValueError):
        _replay(resolver)
    assert _inventory(tmp_path) == before


@pytest.mark.parametrize("case", ["extra-file", "extra-directory", "missing-file", "symlink", "hardlink",
                                   "changed-bytes", "map-tamper", "inventory-tamper", "wrong-plan", "wrong-completion"])
def test_physical_population_and_external_pins_fail_closed(tmp_path, monkeypatch, case):
    resolver, route, _ = _fixture(tmp_path, monkeypatch)
    package = Path(resolver.package_root)
    target = package / "evidence" / Path(route.event.path).name
    if case == "extra-file":
        (package / "unexpected.json").write_bytes(b"{}")
    elif case == "extra-directory":
        (package / "unexpected").mkdir()
    elif case in ("missing-file", "symlink", "hardlink"):
        target.unlink()
        if case == "symlink":
            target.symlink_to(tmp_path / "retained-original" / Path(route.event.path).name)
        elif case == "hardlink":
            os.link(tmp_path / "retained-original" / Path(route.event.path).name, target)
    elif case == "changed-bytes":
        target.write_bytes(b"{}")
    elif case in ("map-tamper", "inventory-tamper"):
        (package / (MAP_NAME if case == "map-tamper" else "inventory.json")).write_bytes(b"{}")
    elif case == "wrong-plan":
        resolver = replace(resolver, plan_sha256="0" * 64)
    else:
        resolver = replace(resolver, completion_sha256="0" * 64)
    with pytest.raises((ValueError, OSError)):
        _replay(resolver)


def test_acquisition_never_accepts_a_relocation_option(tmp_path, monkeypatch):
    resolver, _, calls = _fixture(tmp_path, monkeypatch)
    with pytest.raises(TypeError):
        direct.capture_direct_seconds(root=Path(resolver.package_root) / "capture",
            plan_sha256=resolver.plan_sha256, api_key=KEY, evidence_relocation=resolver)
    assert len(calls) == 1


@pytest.mark.parametrize("field", ["identity", "event"])
@pytest.mark.parametrize("component", ["./", "/"])
def test_original_reference_spelling_is_not_silently_normalized(tmp_path, monkeypatch, field, component):
    resolver, route, _ = _fixture(tmp_path, monkeypatch)
    ref = getattr(route, field)
    path = ref.path.rsplit("/", 1)
    changed = replace(route, **{field: replace(ref, path=path[0] + "/" + component + path[1])})
    query = direct.raw_seconds.SecondQuery(**json.loads(
        (Path(resolver.package_root) / "capture/plan.json").read_bytes())["queries"][0])
    with pytest.raises(ValueError, match="Noncanonical original"):
        changed.validate(query, evidence_relocation=resolver)


def test_relocation_does_not_repair_corrupt_original_reference_semantics(tmp_path, monkeypatch):
    resolver, route, _ = _fixture(tmp_path, monkeypatch)
    package = Path(resolver.package_root)
    target = package / "evidence" / Path(route.identity.path).name
    # Rehashing the map and inventory cannot replace the identity hash recorded
    # in the immutable original plan; no network or path fallback can fix it.
    target.write_bytes(b"{}")
    mapping = json.loads((package / MAP_NAME).read_bytes())["files"]
    row = next(r for r in mapping if r["original_path"] == route.identity.path)
    row.update(bytes=2, sha256=sha256(b"{}").hexdigest())
    resolver = _seal(package, mapping, resolver.plan_sha256, resolver.completion_sha256)
    with pytest.raises(ValueError, match="exact hash-bound"):
        _replay(resolver)
