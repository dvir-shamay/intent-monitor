"""Offline contracts and inert-tool acceptance for LA-1 only."""

from dataclasses import FrozenInstanceError
import json

import pytest

from live_contracts import (
    CallStatus,
    ContractError,
    PlanRecord,
    ReasonCode,
    StateSnapshot,
    tool_schemas,
    validate_native_batch,
)


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