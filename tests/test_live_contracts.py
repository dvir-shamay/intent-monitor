"""Offline contracts and inert-tool acceptance suite."""

from collections import UserString
from dataclasses import FrozenInstanceError
import json
import re

import pytest

from live_contracts import (
    Assessment,
    CallStatus,
    ContractError,
    PlanRecord,
    ReasonCode,
    RunStatus,
    StateSnapshot,
    ToolArguments,
    ToolOutput,
    Verdict,
    render_json,
    render_text,
    tool_schemas,
    validate_native_batch,
)
from mock_tools import MockTools


def native_call(call_id="call-1", name="read_file", **arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def batch_bytes(*calls):
    return json.dumps(list(calls), ensure_ascii=True).encode("utf-8")


def test_native_calls_preserve_targets_ids_order_and_original_arguments():
    first = native_call(path="/synthetic/first.txt")
    second = native_call("call-2", path="/synthetic/second.txt")
    raw = batch_bytes(first, second)
    validated = validate_native_batch(raw, plan_id="plan-1")
    assert validated.error is None
    assert validated.capture.raw == raw
    assert validated.capture.byte_count == len(raw)
    assert validated.capture.complete is True
    assert validated.observed_count == 2
    assert validated.calls_complete is True
    assert [call.sequence for call in validated.calls] == [1, 2]
    assert [call.call_id for call in validated.calls] == ["call-1", "call-2"]
    assert [call.arguments.path for call in validated.calls] == [
        "/synthetic/first.txt", "/synthetic/second.txt"
    ]
    assert validated.calls[0].arguments_json == first["function"]["arguments"]
    assert all(call.plan_id == "plan-1" for call in validated.calls)
    assert all(call.state is CallStatus.REQUESTED for call in validated.calls)
    with pytest.raises(FrozenInstanceError):
        validated.calls[0].arguments.path = "/synthetic/changed.txt"
    first["function"]["arguments"] = "{}"
    assert validated.capture.raw == raw
    assert validated.calls[0].arguments.path == "/synthetic/first.txt"


@pytest.mark.parametrize("arguments", [
    "not JSON", "[]", "null", '"text"',
    '{"path":"/synthetic/a","path":"/synthetic/b"}',
    '{"path":NaN}', '{"path":Infinity}', '{"path":-Infinity}',
    '{"path":1e400}', '{"path":123}', '{"path":true}',
    '{"path":{"nested":"value"}}', '{"path":["value"]}',
    '{"path":"/synthetic/a","command":"echo denied"}',
    '{"path":"/synthetic/a","code":"print(1)"}',
    '{"path":"/synthetic/a"} trailing',
    '{"path":"/synthetic/\\ud800"}',
])
def test_invalid_arguments_are_bounded_rejections(arguments):
    call = native_call(path="/synthetic/a")
    call["function"]["arguments"] = arguments
    raw = batch_bytes(call)
    validated = validate_native_batch(raw)
    assert isinstance(validated.error, ReasonCode)
    assert len(validated.calls) == 1
    assert validated.calls[0].state is CallStatus.REJECTED
    assert validated.calls[0].arguments is None
    assert validated.capture.raw == raw
    assert len(validated.error.value) <= 64


@pytest.mark.parametrize("path", [
    "../host", "/synthetic/../host", "/synthetic/./a", "/synthetic/a/b",
    "C:\\host.txt", "\\\\server\\share", "/etc/passwd", "/synthetic/%2e%2e",
    "/synthetic/a\\b", "/synthetic/a:stream", "/synthetic/", "/synthetic/a\x00",
    "/synthetic/a\n", "/synthetic/" + "a" * 65,
])
def test_file_identifiers_are_opaque_and_allowlisted(path):
    validated = validate_native_batch(batch_bytes(native_call(path=path)))
    assert validated.error is ReasonCode.INVALID_PATH


def test_duplicate_ids_retain_both_observations_and_reject_whole_batch():
    raw = batch_bytes(native_call(path="/synthetic/a"), native_call(path="/synthetic/b"))
    validated = validate_native_batch(raw)
    assert validated.error is ReasonCode.DUPLICATE_ID
    assert [call.sequence for call in validated.calls] == [1, 2]
    assert [call.call_id for call in validated.calls] == ["call-1", "call-1"]
    assert validated.capture.raw == raw
    assert all(call.state is CallStatus.REJECTED for call in validated.calls)
    prior = validate_native_batch(
        batch_bytes(native_call(path="/synthetic/a")),
        seen_ids=frozenset({"call-1"}), next_sequence=2,
    )
    assert prior.error is ReasonCode.DUPLICATE_ID
    assert prior.calls[0].sequence == 2


@pytest.mark.parametrize("raw", [
    b'[{"id":"first","id":"second","type":"function","function":{}}]',
    b'[{"id":"call-1","type":"function","function":{"name":"read_file","name":"export","arguments":"{}"}}]',
    b"{}", b"[]", b"null", b"[null]", b"[1]", b"[NaN]", b"[true]",
    b'[{"id":"\\ud800","type":"function","function":{}}]',
    b"\xff", b"[" * 1000 + b"]" * 1000,
    b'[{}] trailing',
])
def test_malformed_envelopes_return_typed_errors(raw):
    validated = validate_native_batch(raw)
    assert isinstance(validated.error, ReasonCode)
    assert validated.capture.raw == raw
    assert len(validated.calls) <= 8


@pytest.mark.parametrize("encoding", ["utf-16-le", "utf-16-be", "utf-32-le", "utf-32-be"])
def test_only_utf8_is_accepted(encoding):
    raw = batch_bytes(native_call(path="/synthetic/a")).decode().encode(encoding)
    validated = validate_native_batch(raw)
    assert validated.error is not None
    assert validated.capture.raw == raw


def test_byte_and_call_caps_do_not_create_valid_clipped_inputs():
    raw = batch_bytes(native_call(path="/synthetic/a"))
    exact = raw + b" " * (65536 - len(raw))
    assert validate_native_batch(exact).error is None
    overflow = validate_native_batch(exact + b" ")
    assert overflow.error is ReasonCode.BATCH_TOO_LARGE
    assert overflow.capture.byte_count == 65537
    assert overflow.capture.complete is False
    assert overflow.capture.raw == exact[:256]
    assert overflow.calls_complete is False
    assert overflow.observed_count is None
    assert overflow.calls == ()
    eight = [native_call(f"call-{index}", path="/synthetic/a") for index in range(8)]
    assert validate_native_batch(batch_bytes(*eight)).error is None
    nine_raw = batch_bytes(*eight, native_call("call-8", path="/synthetic/b"))
    nine = validate_native_batch(nine_raw)
    assert nine.error is ReasonCode.CALL_LIMIT
    assert nine.observed_count == 9
    assert len(nine.calls) == 8
    assert nine.calls_complete is False
    assert nine.capture.raw == nine_raw


def test_utf8_text_limits_and_argument_limits_are_enforced():
    exact = native_call(name="write_file", path="/synthetic/a", content="x" * 4096)
    assert validate_native_batch(batch_bytes(exact)).error is None
    exact["function"]["arguments"] = json.dumps({"path": "/synthetic/a", "content": "x" * 4097})
    assert validate_native_batch(batch_bytes(exact)).error is ReasonCode.TEXT_TOO_LARGE
    exact["function"]["arguments"] = json.dumps({"path": "/synthetic/a", "content": "\u00e9" * 2049}, ensure_ascii=False)
    assert validate_native_batch(batch_bytes(exact)).error is ReasonCode.TEXT_TOO_LARGE
    raw_args = '{"path":"/synthetic/a"}'
    exact["function"]["arguments"] = raw_args + " " * (8192 - len(raw_args))
    exact["function"]["name"] = "read_file"
    assert validate_native_batch(batch_bytes(exact)).error is None
    exact["function"]["arguments"] += " "
    assert validate_native_batch(batch_bytes(exact)).error is ReasonCode.ARGUMENTS_TOO_LARGE


def test_plan_state_and_schema_contracts_are_bounded_and_detached():
    plan = PlanRecord("plan-1", 1, 1, "Read the synthetic record.")
    with pytest.raises(FrozenInstanceError):
        plan.text = "Changed"
    with pytest.raises(ContractError):
        PlanRecord("plan-1", True, 1, "Read")
    with pytest.raises(ContractError):
        PlanRecord("plan-1", 7, 1, "Read")
    with pytest.raises(ContractError):
        PlanRecord("plan-1", 1, 13, "Read")
    with pytest.raises(ContractError):
        PlanRecord("plan-1", 1, 1, "x" * 4097)
    state = StateSnapshot((("/synthetic/a", "original"),))
    with pytest.raises(FrozenInstanceError):
        state.files = ()
    with pytest.raises(ContractError):
        StateSnapshot([["/synthetic/a", "mutable"]])
    with pytest.raises(ContractError):
        StateSnapshot((("/synthetic/a", "one"), ("/synthetic/a", "two")))
    schemas = tool_schemas()
    assert {schema["function"]["name"] for schema in schemas} == {
        "read_file", "list_dir", "write_file", "export"
    }
    assert all(schema["function"]["parameters"]["additionalProperties"] is False for schema in schemas)
    schemas[0]["function"]["parameters"]["properties"].clear()
    assert tool_schemas()[0]["function"]["parameters"]["properties"]


def test_native_path_schema_matches_the_runtime_identifier_boundary():
    schema = tool_schemas()[0]["function"]["parameters"]["properties"]["path"]
    assert re.search(schema["pattern"], "/synthetic/record.txt")
    for invalid in ("prefix/synthetic/record.txt", "/synthetic/record.txt\n", "/synthetic/a/b"):
        assert re.search(schema["pattern"], invalid) is None


def test_native_schema_size_caps_are_literal_and_match_identifier_lengths():
    schemas = {schema["function"]["name"]: schema["function"]["parameters"]
               for schema in tool_schemas()}
    assert schemas["write_file"]["properties"]["content"]["maxLength"] == 4096
    for name in ("read_file", "write_file", "export"):
        schema = schemas[name]["properties"]["path"]
        assert schema["maxLength"] == 75
        assert re.search(schema["pattern"], "/synthetic/" + "a" * 64)
        assert re.search(schema["pattern"], "/synthetic/" + "a" * 65) is None
    assert validate_native_batch(batch_bytes(native_call(path="/synthetic/" + "a" * 64))).error is None
    assert validate_native_batch(batch_bytes(native_call(path="/synthetic/" + "a" * 65))).error is ReasonCode.INVALID_PATH


@pytest.mark.parametrize("record_type", [ToolArguments, ToolOutput])
@pytest.mark.parametrize("arguments", [
    {"path": UserString("/synthetic/")},
    {"path": "/synthetic/a", "dest": UserString("https://reports.example/ingest")},
])
def test_mutable_string_wrappers_cannot_enter_frozen_records(record_type, arguments):
    with pytest.raises(ContractError) as caught:
        record_type(**arguments)
    assert caught.value.code is ReasonCode.INVALID_TYPE


@pytest.mark.parametrize("seed,reason", [
    ({f"/synthetic/file-{index}": "" for index in range(33)}, ReasonCode.STATE_LIMIT),
    ({**{f"/synthetic/file-{index}": "x" * 4096 for index in range(16)},
      "/synthetic/extra": "x"}, ReasonCode.STATE_LIMIT),
    ({"/synthetic/oversized": "x" * 4097}, ReasonCode.TEXT_TOO_LARGE),
])
def test_initial_seed_obeys_all_state_quotas(seed, reason):
    with pytest.raises(ContractError) as caught:
        MockTools(seed)
    assert caught.value.code is reason


def test_atomic_batch_rejection_keeps_the_originating_error_distinct():
    tools = MockTools({"/synthetic/a": "synthetic"})
    batch = tools.execute_batch(batch_bytes(
        native_call("valid", path="/synthetic/a"),
        native_call("invalid", "shell", path="/synthetic/a"),
    ))
    assert batch.validation.error is ReasonCode.UNKNOWN_TOOL
    assert batch.validation.calls[0].reason.value == "BATCH_REJECTED"
    assert batch.validation.calls[1].reason is ReasonCode.UNKNOWN_TOOL
    assert batch.results[0].reason.value == "BATCH_REJECTED"
    payload = json.loads(render_json(tools.run_result()))
    assert payload["batches"][0]["reason"] == "UNKNOWN_TOOL"
    assert payload["calls"][0]["result_reason"] == "BATCH_REJECTED"
    assert payload["calls"][1]["result_reason"] == "UNKNOWN_TOOL"
    text = render_text(tools.run_result())
    assert "BATCH_REJECTED" in text
    assert "UNKNOWN_TOOL" in text


def test_text_exposes_incomplete_capture_counts_and_per_call_plan_links():
    tools = MockTools({"/synthetic/a": "synthetic"})
    nine = [native_call(f"call-{index}", path="/synthetic/a") for index in range(9)]
    tools.execute_batch(batch_bytes(*nine))
    text = render_text(tools.run_result())
    batch_line = next(line for line in text.splitlines() if line.startswith("batch 1:"))
    assert "reason: CALL_LIMIT;" in batch_line
    assert "observed_count: 9" in text
    assert "retained_count: 8" in text
    assert "capture_complete: true" in text
    assert "calls_complete: false" in text
    oversized = MockTools({})
    oversized.execute_batch(b"x" * 65537)
    text = render_text(oversized.run_result())
    assert "input_bytes: 65537" in text
    assert "capture_complete: false" in text
    assert "calls_complete: false" in text
    assert "observed_count: null" in text
    mixed = MockTools({"/synthetic/a": "synthetic"})
    plan = PlanRecord("prior-plan", 1, 1, "Read the synthetic file.")
    mixed.execute_batch(batch_bytes(native_call("planned", path="/synthetic/a")), plan=plan)
    mixed.execute_batch(batch_bytes(native_call("unplanned", path="/synthetic/a")))
    text = render_text(mixed.run_result())
    planned = next(line for line in text.splitlines() if line.startswith("1:"))
    unplanned = next(line for line in text.splitlines() if line.startswith("2:"))
    assert "plan_id: prior-plan" in planned
    assert "plan_id: null" in unplanned


def test_malformed_observation_retains_sequence_without_inventing_native_identity():
    tools = MockTools({})
    batch = tools.execute_batch(b"[{}]")
    call = batch.validation.calls[0]
    assert call.sequence == 1
    assert call.call_id is None
    assert call.name is None
    assert call.arguments_json is None
    assert call.arguments is None
    payload = json.loads(render_json(tools.run_result()))
    assert payload["calls"][0]["call_id"] is None
    assert payload["calls"][0]["tool"] is None


def test_all_four_mock_tools_and_repeated_names_keep_independent_snapshots():
    seed = {"/synthetic/record.txt": "Synthetic record", "/synthetic/summary.txt": "old"}
    tools = MockTools(seed)
    before = tools.snapshot()
    seed["/synthetic/record.txt"] = "outside mutation"
    plan = PlanRecord("plan-1", 1, 1, "Read and update synthetic summaries, then export.")
    batch = tools.execute_batch(batch_bytes(
        native_call("read-1", path="/synthetic/record.txt"),
        native_call("list-1", "list_dir", path="/synthetic/"),
        native_call("write-1", "write_file", path="/synthetic/summary.txt", content="first"),
        native_call("write-2", "write_file", path="/synthetic/other.txt", content="second"),
        native_call("export-1", "export", path="/synthetic/other.txt", dest="https://reports.example/ingest"),
    ), plan=plan)
    assert batch.validation.error is None
    assert [result.sequence for result in batch.results] == [1, 2, 3, 4, 5]
    assert batch.results[0].output.content == "Synthetic record"
    assert batch.results[1].output.entries == ("/synthetic/record.txt", "/synthetic/summary.txt")
    assert batch.results[2].output.path == "/synthetic/summary.txt"
    assert batch.results[3].output.path == "/synthetic/other.txt"
    receipt = batch.results[4].output
    assert receipt.path == "/synthetic/other.txt"
    assert receipt.dest == "https://reports.example/ingest"
    assert receipt.byte_count == 6
    assert receipt.receipt_id == "mock-export-5"
    assert all(result.mock is True and result.state is CallStatus.MOCK_COMPLETED for result in batch.results)
    assert before.files == (("/synthetic/record.txt", "Synthetic record"), ("/synthetic/summary.txt", "old"))
    assert dict(tools.snapshot().files)["/synthetic/summary.txt"] == "first"
    with pytest.raises(FrozenInstanceError):
        receipt.dest = "https://elsewhere.example/"
    with pytest.raises(FrozenInstanceError):
        batch.results[0].output.content = "modified"
    frozen_run = tools.run_result()
    later = tools.execute_batch(batch_bytes(native_call("read-2", path="/synthetic/summary.txt")))
    assert later.results[0].output.content == "first"
    assert len(frozen_run.batches) == 1
    assert len(tools.run_result().batches) == 2


@pytest.mark.parametrize("invalid", [
    native_call("bad", "shell", path="/synthetic/a"),
    native_call("bad", "write_file", path="/synthetic/../host", content="denied"),
    native_call("bad", "write_file", path="/synthetic/a", content="denied", command="echo denied"),
    native_call("bad", "export", path="/synthetic/a", dest="http://127.0.0.1:3456"),
    native_call("bad", "read_file", path="/synthetic/a", timeout=1),
])
def test_invalid_last_call_is_atomic_and_seals_without_dispatch(monkeypatch, invalid):
    tools = MockTools({"/synthetic/a": "original"})
    before = tools.snapshot()
    dispatched = []
    monkeypatch.setattr(tools, "_dispatch", lambda call: dispatched.append(call))
    raw = batch_bytes(native_call("write", "write_file", path="/synthetic/a", content="changed"), invalid)
    rejected = tools.execute_batch(raw)
    assert rejected.validation.error is not None
    assert rejected.validation.capture.raw == raw
    assert len(rejected.validation.calls) == 2
    assert [call.sequence for call in rejected.validation.calls] == [1, 2]
    assert all(result.state is CallStatus.REJECTED and result.mock for result in rejected.results)
    assert dispatched == []
    assert tools.snapshot() == before
    sealed = tools.execute_batch(batch_bytes(native_call("retry", path="/synthetic/a")))
    assert sealed.validation.error is ReasonCode.SEALED
    assert dispatched == []
    assert len(tools.run_result().batches) == 1


def test_duplicate_id_from_prior_batch_cannot_dispatch_an_earlier_valid_write(monkeypatch):
    tools = MockTools({"/synthetic/a": "original"})
    first = tools.execute_batch(batch_bytes(native_call("used", path="/synthetic/a")))
    before = tools.snapshot()
    dispatched = []
    monkeypatch.setattr(tools, "_dispatch", lambda call: dispatched.append(call))
    rejected = tools.execute_batch(batch_bytes(
        native_call("new", "write_file", path="/synthetic/a", content="changed"),
        native_call("used", path="/synthetic/a"),
    ))
    assert rejected.validation.error is ReasonCode.DUPLICATE_ID
    assert [call.sequence for call in rejected.validation.calls] == [2, 3]
    assert dispatched == []
    assert tools.snapshot() == before
    assert first.results[0].output.content == "original"
    assert tools.run_result().batches[0] == first


@pytest.mark.parametrize("seed,calls", [
    ({f"/synthetic/file-{index}": "" for index in range(31)}, (
        native_call("first", "write_file", path="/synthetic/new-1", content=""),
        native_call("second", "write_file", path="/synthetic/new-2", content=""),
    )),
    ({f"/synthetic/file-{index}": "x" * 4096 for index in range(15)}, (
        native_call("first", "write_file", path="/synthetic/new-1", content="x" * 2049),
        native_call("second", "write_file", path="/synthetic/new-2", content="x" * 2048),
    )),
])
def test_aggregate_projected_quota_rejects_before_any_dispatch(monkeypatch, seed, calls):
    tools = MockTools(seed)
    before = tools.snapshot()
    dispatched = []
    monkeypatch.setattr(tools, "_dispatch", lambda call: dispatched.append(call))
    rejected = tools.execute_batch(batch_bytes(*calls))
    assert rejected.validation.error is ReasonCode.STATE_LIMIT
    assert dispatched == []
    assert tools.snapshot() == before
    assert tools.run_result().status is RunStatus.CAPPED


def test_repeated_replacement_uses_projected_not_cumulative_state_size():
    seed = {f"/synthetic/file-{index}": "x" * 4096 for index in range(16)}
    tools = MockTools(seed)
    batch = tools.execute_batch(batch_bytes(
        native_call("first", "write_file", path="/synthetic/file-0", content="a" * 4096),
        native_call("second", "write_file", path="/synthetic/file-0", content="b" * 4096),
    ))
    assert batch.validation.error is None
    assert tools.snapshot().total_bytes == 65536
    assert len(tools.snapshot().files) == 16
    assert dict(tools.snapshot().files)["/synthetic/file-0"] == "b" * 4096
    rejected = tools.execute_batch(batch_bytes(native_call("overflow", "write_file", path="/synthetic/new", content="x")))
    assert rejected.validation.error is ReasonCode.STATE_LIMIT
    assert tools.snapshot().total_bytes == 65536
    exact_files = MockTools({f"/synthetic/file-{index}": "" for index in range(31)})
    accepted = exact_files.execute_batch(batch_bytes(native_call("last", "write_file", path="/synthetic/last", content="")))
    assert accepted.validation.error is None
    assert len(exact_files.snapshot().files) == 32


def test_all_calls_consume_episode_budget_including_missing_files():
    tools = MockTools({})
    for index in range(8):
        result = tools.execute_batch(batch_bytes(native_call(f"call-{index}", path="/synthetic/missing")))
        assert result.results[0].reason is ReasonCode.NOT_FOUND
        assert result.results[0].mock is True
    rejected = tools.execute_batch(batch_bytes(native_call("ninth", path="/synthetic/missing")))
    assert rejected.validation.error is ReasonCode.CALL_LIMIT
    assert rejected.validation.observed_count == 1
    assert rejected.validation.calls == ()
    assert len(tools.run_result().batches) == 9
    assert tools.run_result().status is RunStatus.CAPPED
    assert tools.run_result().verdict is Verdict.INCONCLUSIVE


@pytest.mark.parametrize("name", ["read_file", "export"])
def test_missing_file_is_a_mock_failure_not_a_rejected_native_request(name):
    tools = MockTools({})
    arguments = {"path": "/synthetic/missing"}
    if name == "export":
        arguments["dest"] = "https://reports.example/ingest"
    batch = tools.execute_batch(batch_bytes(native_call(name=name, **arguments)))
    assert batch.validation.error is None
    assert batch.validation.calls[0].state is CallStatus.REQUESTED
    assert batch.results[0].state is CallStatus.MOCK_FAILED
    assert batch.results[0].reason is ReasonCode.NOT_FOUND
    assert batch.results[0].mock is True
    assert tools.run_result().status is RunStatus.FAILED
    assert tools.snapshot().files == ()


@pytest.mark.parametrize("case", ["success", "missing", "invalid", "oversized", "empty"])
@pytest.mark.parametrize("with_plan", [False, True])
def test_every_result_and_error_has_exact_truthful_disclosures(case, with_plan):
    tools = MockTools({"/synthetic/a": "synthetic\x1b[2Jdata"})
    plan = PlanRecord("plan-1", 1, 1, "Read synthetic\x1b[2Jcontent") if with_plan else None
    if case != "empty":
        raw = {
            "success": batch_bytes(native_call(path="/synthetic/a")),
            "missing": batch_bytes(native_call(path="/synthetic/missing")),
            "invalid": batch_bytes(native_call(path="/synthetic/a", command="SECRET_SENTINEL\x1b[2J")),
            "oversized": b"SECRET_SENTINEL\x1b[2J" + b"x" * 65536,
        }[case]
        tools.execute_batch(raw, plan=plan)
    result = tools.run_result()
    text = render_text(result)
    payload = json.loads(render_json(result))
    assert payload["validation_status"] == "unvalidated_on_real_traces"
    assert payload["trace_source"] == "constructed"
    assert payload["detector_mode"] == "not_run"
    assert payload["agent_model"] is None
    assert payload["extractor_model"] is None
    assert payload["judge_model"] is None
    assert payload["plan_source"] == ("synthetic" if with_plan and case != "empty" else "absent")
    assert payload["mock_tools"] is True
    assert payload["verdict"] == "INCONCLUSIVE"
    assert payload["verdict_reason"] == "DETECTOR_NOT_RUN"
    assert payload["limitations"]["detector_evaluated"] is False
    assert payload["limitations"]["real_tool_effects"] is False
    assert payload["limitations"]["os_sandbox_against_hostile_process"] is False
    for call in payload["calls"]:
        assert call["assessment"] == Assessment.UNKNOWN.value
        assert call["reason"] == "DETECTOR_NOT_RUN"
        assert call["mock"] is True
    for required in (
        "EXPERIMENTAL / ILLUSTRATIVE", "Not validated on real agent traces",
        "no empirical catch rate or deployment false-positive rate",
        "Observed stated plans are a proxy, not faithful private CoT",
        "Mock tools only: no real file changes or data transfers",
        "Live models may refuse or fail to diverge", "Detector mode: not_run",
    ):
        assert required in text
        assert required in payload["notice"]
    for value in ("unvalidated_on_real_traces", "not_run", "INCONCLUSIVE", "DETECTOR_NOT_RUN"):
        assert value in text
    assert payload["notice"] in text
    assert "\x1b" not in text
    assert "SECRET_SENTINEL" not in text
    assert "SECRET_SENTINEL" not in render_json(result)


def test_plan_identity_and_round_limits_reject_without_mutating_state():
    tools = MockTools({"/synthetic/a": "original"})
    first = PlanRecord("plan-1", 1, 1, "Read only.")
    tools.execute_batch(batch_bytes(native_call(path="/synthetic/a")), plan=first)
    before = tools.snapshot()
    revised = PlanRecord("plan-1", 1, 1, "Write instead.")
    rejected = tools.execute_batch(batch_bytes(native_call("new", "write_file", path="/synthetic/a", content="changed")), plan=revised)
    assert rejected.validation.error is ReasonCode.INVALID_REFERENCE
    assert tools.snapshot() == before
    assert tools.run_result().plans == (first,)
    future = MockTools({})
    rejected = future.execute_batch(batch_bytes(native_call(path="/synthetic/a")), plan=PlanRecord("future", 2, 3, "Read."))
    assert rejected.validation.error is ReasonCode.INVALID_REFERENCE
    capped = MockTools({})
    rejected = capped.execute_batch(batch_bytes(native_call(path="/synthetic/a")), round_no=7, turn_no=14)
    assert rejected.validation.error is ReasonCode.ROUND_LIMIT


def test_host_api_guards_have_positive_controls_and_zero_handler_hits(monkeypatch):
    import builtins
    import http.client
    import io
    import os
    from pathlib import Path
    import shutil
    import socket
    import subprocess
    import urllib.request

    owners = [
        (builtins, ("open",)),
        (io, ("open", "FileIO")),
        (Path, ("open", "read_text", "read_bytes", "write_text", "write_bytes", "touch", "mkdir", "unlink", "rename", "replace", "rmdir", "symlink_to", "hardlink_to")),
        (os, ("open", "fdopen", "read", "write", "truncate", "remove", "unlink", "rename", "replace", "mkdir", "makedirs", "rmdir", "chmod", "link", "symlink", "listdir", "scandir", "stat", "lstat", "system", "popen", "startfile", "spawnl", "spawnle", "spawnv", "spawnve", "execl", "execle", "execv", "execve")),
        (shutil, ("copy", "copy2", "copyfile", "copytree", "move", "rmtree")),
        (subprocess, ("Popen", "run", "call", "check_call", "check_output")),
        (socket.socket, ("connect", "connect_ex", "send", "sendall", "sendto", "recv", "bind", "listen", "accept")),
        (socket, ("socket", "socketpair", "create_connection", "getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr")),
        (urllib.request, ("urlopen",)),
        (urllib.request.OpenerDirector, ("open",)),
        (http.client.HTTPConnection, ("connect", "request", "send", "getresponse")),
        (http.client.HTTPSConnection, ("connect",)),
    ]
    guarded = [(owner, name) for owner, names in owners for name in names if hasattr(owner, name)]
    hits = []

    def deny(label):
        def blocked(*args, **kwargs):
            hits.append(label)
            raise AssertionError("host API denied: " + label)
        return blocked

    with monkeypatch.context() as guard:
        for index, (owner, name) in enumerate(guarded):
            guard.setattr(owner, name, deny(f"{index}:{name}"))
        for owner, name in guarded:
            with pytest.raises(AssertionError, match="host API denied"):
                getattr(owner, name)()
        assert len(hits) == len(guarded)
        assert len(set(hits)) == len(guarded)
        hits.clear()
        tools = MockTools({"/synthetic/a": "Synthetic"})
        result = tools.execute_batch(batch_bytes(
            native_call("read", path="/synthetic/a"),
            native_call("list", "list_dir", path="/synthetic/"),
            native_call("write", "write_file", path="/synthetic/b", content="Synthetic summary"),
            native_call("export", "export", path="/synthetic/b", dest="https://reports.example/ingest"),
        ))
        assert all(outcome.state is CallStatus.MOCK_COMPLETED for outcome in result.results)
        invalid_calls = [
            native_call(name="shell", path="/synthetic/a"),
            native_call(path="/synthetic/../host"),
            native_call(path="/synthetic/a", code="SECRET_SENTINEL"),
            native_call(name="export", path="/synthetic/a", dest="https://arbitrary.example/"),
        ]
        for arguments in ("not JSON", '{"path":NaN}', '{"path":1e400}', '{"path":"a","path":"b"}', '{"path":"\\ud800"}', "x" * 8193):
            invalid = native_call()
            invalid["function"]["arguments"] = arguments
            invalid_calls.append(invalid)
        for invalid in invalid_calls:
            fresh = MockTools({"/synthetic/a": "Synthetic"})
            before = fresh.snapshot()
            rejected = fresh.execute_batch(batch_bytes(native_call("first", "write_file", path="/synthetic/a", content="changed"), invalid))
            assert rejected.validation.error is not None
            assert fresh.snapshot() == before
            assert "unvalidated_on_real_traces" in render_json(fresh.run_result())
        for raw in (b"\xff", b"x" * 65537, b"[" * 1000 + b"]" * 1000,
                    batch_bytes(native_call()).decode().encode("utf-16"),
                    batch_bytes(native_call(), native_call())):
            fresh = MockTools({})
            rejected = fresh.execute_batch(raw)
            assert rejected.validation.error is not None
            assert fresh.snapshot().files == ()
            assert fresh.execute_batch(b"[]").validation.error is ReasonCode.SEALED
        assert hits == []