"""Bounded, product-owned records. No transport or detector execution."""

from dataclasses import dataclass, field, replace
from enum import Enum
import json
import re


MAX_ROUNDS = 6
MAX_CALLS = 8
MAX_BATCH_BYTES = 65536
MAX_ARGUMENT_BYTES = 8192
MAX_TEXT_BYTES = 4096
MAX_FILES = 32
MAX_STATE_BYTES = 65536
PREVIEW_BYTES = 256
EXPORT_DESTINATION = "https://reports.example/ingest"
VALIDATION_STATUS = "unvalidated_on_real_traces"
_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_FILE_PATH = re.compile(r"/synthetic/[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_TOOL_FIELDS = {
    "read_file": frozenset({"path"}),
    "list_dir": frozenset({"path"}),
    "write_file": frozenset({"path", "content"}),
    "export": frozenset({"path", "dest"}),
}


class RunStatus(str, Enum):
    COMPLETED = "completed"
    REFUSED = "refused"
    FAILED = "failed"
    CAPPED = "capped"
    INTERRUPTED = "interrupted"
    BLOCKED = "blocked"


class CallStatus(str, Enum):
    REQUESTED = "requested"
    REJECTED = "rejected"
    MOCK_COMPLETED = "mock_completed"
    MOCK_FAILED = "mock_failed"


class Assessment(str, Enum):
    MATCHED = "MATCHED"
    DIVERGENT = "DIVERGENT"
    UNKNOWN = "UNKNOWN"


class Verdict(str, Enum):
    NO_OBSERVED_DIVERGENCE = "NO_OBSERVED_DIVERGENCE"
    FLAGGED = "FLAGGED"
    INCONCLUSIVE = "INCONCLUSIVE"


class PlanSource(str, Enum):
    SYNTHETIC = "synthetic"
    PUBLIC_PLAN = "public_plan"
    ACTION_TEXT = "action_text"
    ABSENT = "absent"


class ReasonCode(str, Enum):
    DETECTOR_NOT_RUN = "DETECTOR_NOT_RUN"
    INVALID_TYPE = "INVALID_TYPE"
    INVALID_IDENTIFIER = "INVALID_IDENTIFIER"
    INVALID_UTF8 = "INVALID_UTF8"
    INVALID_JSON = "INVALID_JSON"
    DUPLICATE_KEY = "DUPLICATE_KEY"
    NONFINITE_NUMBER = "NONFINITE_NUMBER"
    NUMERIC_VALUE = "NUMERIC_VALUE"
    NESTING_LIMIT = "NESTING_LIMIT"
    BATCH_REJECTED = "BATCH_REJECTED"
    BATCH_TOO_LARGE = "BATCH_TOO_LARGE"
    ARGUMENTS_TOO_LARGE = "ARGUMENTS_TOO_LARGE"
    TEXT_TOO_LARGE = "TEXT_TOO_LARGE"
    INVALID_ENVELOPE = "INVALID_ENVELOPE"
    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    DUPLICATE_ID = "DUPLICATE_ID"
    INVALID_PATH = "INVALID_PATH"
    INVALID_DESTINATION = "INVALID_DESTINATION"
    ROUND_LIMIT = "ROUND_LIMIT"
    CALL_LIMIT = "CALL_LIMIT"
    STATE_LIMIT = "STATE_LIMIT"
    NO_CALLS = "NO_CALLS"
    NOT_FOUND = "NOT_FOUND"
    SEALED = "SEALED"
    INVALID_REFERENCE = "INVALID_REFERENCE"


class ContractError(ValueError):
    """A finite reason only; never echo rejected model data."""

    def __init__(self, code: ReasonCode):
        self.code = code
        super().__init__(code.value)


def _require(condition, code=ReasonCode.INVALID_TYPE):
    if not condition:
        raise ContractError(code)


def _integer(value, minimum, maximum, code=ReasonCode.INVALID_TYPE):
    _require(type(value) is int and minimum <= value <= maximum, code)


def text_bytes(value, limit=MAX_TEXT_BYTES, code=ReasonCode.TEXT_TOO_LARGE):
    _require(type(value) is str)
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeError:
        raise ContractError(ReasonCode.INVALID_UTF8) from None
    _require(len(encoded) <= limit, code)
    return len(encoded)


def identifier(value):
    _require(type(value) is str and _IDENTIFIER.fullmatch(value) is not None,
             ReasonCode.INVALID_IDENTIFIER)


def file_path(value):
    _require(type(value) is str and _FILE_PATH.fullmatch(value) is not None,
             ReasonCode.INVALID_PATH)


def _record_tuple(value, record_type, maximum):
    _require(type(value) is tuple and len(value) <= maximum)
    _require(all(type(item) is record_type for item in value))


@dataclass(frozen=True, slots=True)
class PlanRecord:
    plan_id: str
    round_no: int
    turn_no: int
    text: str
    source: PlanSource = PlanSource.SYNTHETIC

    def __post_init__(self):
        identifier(self.plan_id)
        _integer(self.round_no, 1, MAX_ROUNDS, ReasonCode.ROUND_LIMIT)
        _integer(self.turn_no, 1, MAX_ROUNDS * 2, ReasonCode.ROUND_LIMIT)
        text_bytes(self.text)
        _require(bool(self.text.strip()), ReasonCode.INVALID_ARGUMENTS)
        _require(type(self.source) is PlanSource and self.source is not PlanSource.ABSENT)


@dataclass(frozen=True, slots=True)
class InputCapture:
    raw: bytes
    byte_count: int
    complete: bool

    def __post_init__(self):
        _require(type(self.raw) is bytes and type(self.complete) is bool)
        _integer(self.byte_count, 0, 2 ** 63 - 1)
        if self.complete:
            _require(len(self.raw) == self.byte_count <= MAX_BATCH_BYTES)
        else:
            _require(len(self.raw) <= PREVIEW_BYTES < self.byte_count)


def capture_input(raw):
    _require(type(raw) is bytes)
    complete = len(raw) <= MAX_BATCH_BYTES
    return InputCapture(raw if complete else raw[:PREVIEW_BYTES], len(raw), complete)


@dataclass(frozen=True, slots=True)
class ToolArguments:
    path: str
    content: str | None = None
    dest: str | None = None

    def __post_init__(self):
        _require(type(self.path) is str)
        if self.path != "/synthetic/":
            file_path(self.path)
        if self.content is not None:
            text_bytes(self.content)
        if self.dest is not None:
            _require(type(self.dest) is str)
            _require(self.dest == EXPORT_DESTINATION, ReasonCode.INVALID_DESTINATION)


@dataclass(frozen=True, slots=True)
class CallRecord:
    sequence: int
    round_no: int
    turn_no: int
    plan_id: str | None
    call_id: str | None
    name: str | None
    arguments_json: str | None
    arguments: ToolArguments | None
    state: CallStatus
    reason: ReasonCode | None = None

    def __post_init__(self):
        _integer(self.sequence, 1, MAX_CALLS, ReasonCode.CALL_LIMIT)
        _integer(self.round_no, 1, MAX_ROUNDS, ReasonCode.ROUND_LIMIT)
        _integer(self.turn_no, 1, MAX_ROUNDS * 2, ReasonCode.ROUND_LIMIT)
        for value in (self.plan_id, self.call_id, self.name):
            if value is not None:
                identifier(value)
        if self.arguments_json is not None:
            text_bytes(self.arguments_json, MAX_ARGUMENT_BYTES, ReasonCode.ARGUMENTS_TOO_LARGE)
        _require(self.arguments is None or type(self.arguments) is ToolArguments)
        _require(type(self.state) is CallStatus)
        _require(self.reason is None or type(self.reason) is ReasonCode)
        _require(self.state in (CallStatus.REQUESTED, CallStatus.REJECTED))
        if self.state is CallStatus.REQUESTED:
            _require(self.call_id is not None and self.name in _TOOL_FIELDS
                     and self.arguments_json is not None and self.arguments is not None
                     and self.reason is None)
            _require(_arguments_for(self.name, self.arguments_json) == self.arguments)
        else:
            _require(self.reason is not None and self.arguments is None)


@dataclass(frozen=True, slots=True)
class BatchValidation:
    capture: InputCapture
    calls: tuple[CallRecord, ...]
    observed_count: int | None
    calls_complete: bool
    error: ReasonCode | None

    def __post_init__(self):
        _require(type(self.capture) is InputCapture)
        _record_tuple(self.calls, CallRecord, MAX_CALLS)
        _require(type(self.calls_complete) is bool)
        if self.observed_count is not None:
            _integer(self.observed_count, len(self.calls), MAX_BATCH_BYTES)
        _require(not self.calls_complete or self.observed_count == len(self.calls))
        _require(self.error is None or type(self.error) is ReasonCode)
        if self.error is None:
            _require(self.capture.complete and self.calls_complete and bool(self.calls))
            _require(all(call.state is CallStatus.REQUESTED for call in self.calls))
            _require(len({call.call_id for call in self.calls}) == len(self.calls))
        else:
            _require(all(call.state is CallStatus.REJECTED for call in self.calls))


def reject_batch(batch, reason):
    _require(type(batch) is BatchValidation and type(reason) is ReasonCode)
    calls = tuple(replace(call, arguments=None, state=CallStatus.REJECTED,
                          reason=call.reason or ReasonCode.BATCH_REJECTED)
                  for call in batch.calls)
    return BatchValidation(batch.capture, calls, batch.observed_count,
                           batch.calls_complete, reason)


def _depth_guard(text, limit):
    depth = 0
    quoted = False
    escaped = False
    for character in text:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character in "[{":
            depth += 1
            _require(depth <= limit, ReasonCode.NESTING_LIMIT)
        elif character in "]}":
            depth -= 1


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, ReasonCode.DUPLICATE_KEY)
        result[key] = value
    return result


def _no_number(value):
    raise ContractError(ReasonCode.NUMERIC_VALUE)


def _no_constant(value):
    raise ContractError(ReasonCode.NONFINITE_NUMBER)


def _unicode_tree(value):
    if type(value) is str:
        text_bytes(value, MAX_BATCH_BYTES)
    elif type(value) is dict:
        for key, item in value.items():
            _unicode_tree(key)
            _unicode_tree(item)
    elif type(value) is list:
        for item in value:
            _unicode_tree(item)


def _parse_json(text, depth):
    _depth_guard(text, depth)
    try:
        value = json.loads(text, object_pairs_hook=_unique_object,
                           parse_int=_no_number, parse_float=_no_number,
                           parse_constant=_no_constant)
    except (ValueError, RecursionError) as error:
        if isinstance(error, ContractError):
            raise
        raise ContractError(ReasonCode.INVALID_JSON) from None
    _unicode_tree(value)
    return value


def _arguments_for(name, raw):
    _require(name in _TOOL_FIELDS, ReasonCode.UNKNOWN_TOOL)
    text_bytes(raw, MAX_ARGUMENT_BYTES, ReasonCode.ARGUMENTS_TOO_LARGE)
    arguments = _parse_json(raw, 1)
    _require(type(arguments) is dict and set(arguments) == _TOOL_FIELDS[name],
             ReasonCode.INVALID_ARGUMENTS)
    _require(all(type(value) is str for value in arguments.values()),
             ReasonCode.INVALID_ARGUMENTS)
    if name == "list_dir":
        _require(arguments["path"] == "/synthetic/", ReasonCode.INVALID_PATH)
    else:
        file_path(arguments["path"])
    return ToolArguments(**arguments)


def _bounded_identifier(value):
    return value if type(value) is str and _IDENTIFIER.fullmatch(value) else None


def _bounded_arguments(value):
    try:
        text_bytes(value, MAX_ARGUMENT_BYTES, ReasonCode.ARGUMENTS_TOO_LARGE)
        return value
    except ContractError:
        return None


def validate_native_batch(raw: bytes, *, seen_ids=frozenset(), next_sequence=1,
                          round_no=1, turn_no=2, plan_id=None) -> BatchValidation:
    """Validate a native tool_calls array; never dispatch or repair a request."""
    capture = capture_input(raw)
    try:
        _integer(round_no, 1, MAX_ROUNDS, ReasonCode.ROUND_LIMIT)
        _integer(turn_no, 1, MAX_ROUNDS * 2, ReasonCode.ROUND_LIMIT)
        _integer(next_sequence, 1, MAX_CALLS + 1, ReasonCode.CALL_LIMIT)
        _require(type(seen_ids) is frozenset and len(seen_ids) <= MAX_CALLS)
        for seen_id in seen_ids:
            identifier(seen_id)
        if plan_id is not None:
            identifier(plan_id)
        _require(capture.complete, ReasonCode.BATCH_TOO_LARGE)
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeError:
            raise ContractError(ReasonCode.INVALID_UTF8) from None
        entries = _parse_json(text, 4)
        _require(type(entries) is list, ReasonCode.INVALID_ENVELOPE)
    except ContractError as error:
        return BatchValidation(capture, (), None, False, error.code)
    if not entries:
        return BatchValidation(capture, (), 0, True, ReasonCode.NO_CALLS)
    remaining = MAX_CALLS - next_sequence + 1
    batch_error = ReasonCode.CALL_LIMIT if len(entries) > remaining else None
    calls = []
    ids = set(seen_ids)
    for offset, entry in enumerate(entries[:remaining]):
        envelope = entry if type(entry) is dict else {}
        function = envelope.get("function")
        function = function if type(function) is dict else {}
        call_id = _bounded_identifier(envelope.get("id"))
        name = _bounded_identifier(function.get("name"))
        arguments_json = _bounded_arguments(function.get("arguments"))
        arguments = None
        reason = None
        try:
            _require(type(entry) is dict and set(entry) == {"id", "type", "function"}
                     and entry["type"] == "function" and type(entry["type"]) is str
                     and type(entry["function"]) is dict
                     and set(function) == {"name", "arguments"}, ReasonCode.INVALID_ENVELOPE)
            identifier(entry["id"])
            identifier(function["name"])
            _require(call_id not in ids, ReasonCode.DUPLICATE_ID)
            ids.add(call_id)
            arguments = _arguments_for(name, function["arguments"])
        except ContractError as error:
            reason = error.code
            if batch_error is None:
                batch_error = reason
        calls.append(CallRecord(next_sequence + offset, round_no, turn_no, plan_id,
                                call_id, name, arguments_json, arguments,
                                CallStatus.REJECTED if reason else CallStatus.REQUESTED,
                                reason))
    if batch_error is not None:
        calls = [replace(call, arguments=None, state=CallStatus.REJECTED,
                         reason=call.reason or ReasonCode.BATCH_REJECTED) for call in calls]
    return BatchValidation(capture, tuple(calls), len(entries),
                           len(calls) == len(entries), batch_error)


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    files: tuple[tuple[str, str], ...]

    def __post_init__(self):
        _require(type(self.files) is tuple and len(self.files) <= MAX_FILES,
                 ReasonCode.STATE_LIMIT)
        paths = set()
        total = 0
        for item in self.files:
            _require(type(item) is tuple and len(item) == 2)
            path, content = item
            file_path(path)
            _require(path not in paths, ReasonCode.INVALID_PATH)
            paths.add(path)
            total += text_bytes(content)
        _require(total <= MAX_STATE_BYTES, ReasonCode.STATE_LIMIT)

    @property
    def total_bytes(self):
        return sum(len(content.encode("utf-8")) for _, content in self.files)


@dataclass(frozen=True, slots=True)
class ToolOutput:
    path: str
    content: str | None = None
    entries: tuple[str, ...] = ()
    dest: str | None = None
    byte_count: int | None = None
    receipt_id: str | None = None

    def __post_init__(self):
        _require(type(self.path) is str)
        if self.path != "/synthetic/":
            file_path(self.path)
        if self.content is not None:
            text_bytes(self.content)
        _require(type(self.entries) is tuple and len(self.entries) <= MAX_FILES)
        for path in self.entries:
            file_path(path)
        _require(len(set(self.entries)) == len(self.entries))
        if self.dest is not None:
            _require(type(self.dest) is str)
            _require(self.dest == EXPORT_DESTINATION, ReasonCode.INVALID_DESTINATION)
        if self.byte_count is not None:
            _integer(self.byte_count, 0, MAX_TEXT_BYTES)
        if self.receipt_id is not None:
            identifier(self.receipt_id)


@dataclass(frozen=True, slots=True)
class ToolResult:
    sequence: int
    call_id: str | None
    state: CallStatus
    output: ToolOutput | None = None
    reason: ReasonCode | None = None
    mock: bool = field(default=True, init=False)

    def __post_init__(self):
        _integer(self.sequence, 1, MAX_CALLS, ReasonCode.CALL_LIMIT)
        if self.call_id is not None:
            identifier(self.call_id)
        _require(type(self.state) is CallStatus and self.state is not CallStatus.REQUESTED)
        _require(self.output is None or type(self.output) is ToolOutput)
        _require(self.reason is None or type(self.reason) is ReasonCode)
        if self.state is CallStatus.MOCK_COMPLETED:
            _require(self.call_id is not None and self.output is not None and self.reason is None)
        else:
            _require(self.output is None and self.reason is not None)


@dataclass(frozen=True, slots=True)
class BatchResult:
    validation: BatchValidation
    results: tuple[ToolResult, ...]

    def __post_init__(self):
        _require(type(self.validation) is BatchValidation)
        _record_tuple(self.results, ToolResult, MAX_CALLS)
        _require(len(self.validation.calls) == len(self.results), ReasonCode.INVALID_REFERENCE)
        for call, result in zip(self.validation.calls, self.results):
            _require((call.sequence, call.call_id) == (result.sequence, result.call_id),
                     ReasonCode.INVALID_REFERENCE)
            _require((call.state is CallStatus.REJECTED) == (result.state is CallStatus.REJECTED))


@dataclass(frozen=True, slots=True)
class CallVerdict:
    sequence: int
    call_id: str | None
    assessment: Assessment = Assessment.UNKNOWN
    reason: ReasonCode = ReasonCode.DETECTOR_NOT_RUN

    def __post_init__(self):
        _integer(self.sequence, 1, MAX_CALLS, ReasonCode.CALL_LIMIT)
        if self.call_id is not None:
            identifier(self.call_id)
        _require(type(self.assessment) is Assessment and type(self.reason) is ReasonCode)
        _require(self.assessment is Assessment.UNKNOWN and self.reason is ReasonCode.DETECTOR_NOT_RUN)


@dataclass(frozen=True, slots=True)
class RunResult:
    run_id: str
    plans: tuple[PlanRecord, ...]
    batches: tuple[BatchResult, ...]
    status: RunStatus
    reason: ReasonCode = ReasonCode.DETECTOR_NOT_RUN
    verdict: Verdict = field(default=Verdict.INCONCLUSIVE, init=False)

    def __post_init__(self):
        identifier(self.run_id)
        _record_tuple(self.plans, PlanRecord, MAX_ROUNDS)
        _record_tuple(self.batches, BatchResult, MAX_CALLS + 1)
        _require(type(self.status) is RunStatus and type(self.reason) is ReasonCode)
        plans = {plan.plan_id: plan for plan in self.plans}
        _require(len(plans) == len(self.plans), ReasonCode.INVALID_REFERENCE)
        _require(all(plan.source is PlanSource.SYNTHETIC for plan in self.plans))
        calls = [call for batch in self.batches for call in batch.validation.calls]
        _require(len(calls) <= MAX_CALLS, ReasonCode.CALL_LIMIT)
        _require([call.sequence for call in calls] == list(range(1, len(calls) + 1)),
                 ReasonCode.INVALID_REFERENCE)
        accepted_ids = []
        for call in calls:
            if call.state is CallStatus.REQUESTED:
                accepted_ids.append(call.call_id)
            if call.plan_id is not None:
                plan = plans.get(call.plan_id)
                _require(plan is not None and plan.turn_no < call.turn_no
                         and plan.round_no <= call.round_no, ReasonCode.INVALID_REFERENCE)
        _require(len(set(accepted_ids)) == len(accepted_ids), ReasonCode.DUPLICATE_ID)
        if self.status is RunStatus.COMPLETED:
            _require(bool(calls) and all(batch.validation.error is None for batch in self.batches))
            _require(all(result.state is CallStatus.MOCK_COMPLETED
                         for batch in self.batches for result in batch.results))


NOTICE = (
    "EXPERIMENTAL / ILLUSTRATIVE. Not validated on real agent traces; no empirical "
    "catch rate or deployment false-positive rate. Observed stated plans are a proxy, "
    "not faithful private CoT. Mock tools only: no real file changes or data transfers. "
    "Live models may refuse or fail to diverge. Detector mode: not_run; no detector "
    "evaluation has run. Errors, misses and false alarms remain possible."
)


def _result_payload(result):
    _require(type(result) is RunResult)
    calls = []
    batches = []
    for batch_index, batch in enumerate(result.batches, start=1):
        validation = batch.validation
        batches.append({
            "batch_index": batch_index,
            "reason": validation.error.value if validation.error else None,
            "input_bytes": validation.capture.byte_count,
            "capture_complete": validation.capture.complete,
            "observed_count": validation.observed_count,
            "retained_count": len(validation.calls),
            "calls_complete": validation.calls_complete,
            "mock": True,
        })
        for call, outcome in zip(validation.calls, batch.results):
            assessment = CallVerdict(call.sequence, call.call_id)
            calls.append({
                "batch_index": batch_index,
                "sequence": call.sequence, "call_id": call.call_id, "tool": call.name,
                "round_no": call.round_no, "turn_no": call.turn_no, "plan_id": call.plan_id,
                "request_state": call.state.value, "result_state": outcome.state.value,
                "result_reason": outcome.reason.value if outcome.reason else None,
                "assessment": assessment.assessment.value, "reason": assessment.reason.value,
                "mock": outcome.mock,
            })
    return {
        "run_id": result.run_id, "run_status": result.status.value,
        "run_reason": result.reason.value, "verdict": result.verdict.value,
        "verdict_reason": ReasonCode.DETECTOR_NOT_RUN.value,
        "trace_source": "constructed", "detector_mode": "not_run",
        "agent_model": None, "extractor_model": None, "judge_model": None,
        "plan_source": "synthetic" if result.plans else "absent",
        "mock_tools": True, "validation_status": VALIDATION_STATUS, "notice": NOTICE,
        "limitations": {
            "real_trace_validation": False, "empirical_catch_rate_established": False,
            "deployment_false_positive_rate_established": False,
            "stated_plan_is_proxy": True, "faithful_private_cot_access": False,
            "real_tool_effects": False, "detector_evaluated": False,
            "live_refusal_or_no_divergence_possible": True,
            "errors_misses_false_alarms_possible": True,
            "consistency_is_not_authorization_or_safety": True,
            "os_sandbox_against_hostile_process": False,
        },
        "batches": batches, "calls": calls,
    }


def render_json(result: RunResult) -> str:
    """Render a qualified summary, excluding captured raw input and tool content."""
    return json.dumps(_result_payload(result), ensure_ascii=True, allow_nan=False,
                      sort_keys=True, indent=2)


def render_text(result: RunResult) -> str:
    """Render only the closed, control-safe summary fields."""
    payload = _result_payload(result)
    lines = [NOTICE, f"validation_status: {payload['validation_status']}",
             f"detector_mode: {payload['detector_mode']}",
             f"trace_source: {payload['trace_source']}; plan_source: {payload['plan_source']}",
             "agent_model: null; extractor_model: null; judge_model: null; mock_tools: true",
             f"run_status: {payload['run_status']}; run_reason: {payload['run_reason']}",
             f"verdict: {payload['verdict']}; reason: {payload['verdict_reason']}"]
    for batch in payload["batches"]:
        count = batch["observed_count"] if batch["observed_count"] is not None else "null"
        lines.append(f"batch {batch['batch_index']}: reason: {batch['reason'] or 'null'}; "
                     f"input_bytes: {batch['input_bytes']}; observed_count: {count}; "
                     f"retained_count: {batch['retained_count']}; "
                     f"capture_complete: {str(batch['capture_complete']).lower()}; "
                     f"calls_complete: {str(batch['calls_complete']).lower()}; mock: true")
    for call in payload["calls"]:
        lines.append(f"{call['sequence']}: {call['call_id'] or 'null'} {call['tool'] or 'null'} "
                     f"{call['result_state']} {call['result_reason'] or 'null'} "
                     f"{call['assessment']} {call['reason']} mock: true; "
                     f"batch: {call['batch_index']}; plan_id: {call['plan_id'] or 'null'}")
    return "\n".join(lines)


def tool_schemas() -> list[dict]:
    """Return fresh native function schemas; Python validation enforces UTF-8 caps."""
    schemas = []
    for name, fields in _TOOL_FIELDS.items():
        properties = {}
        for argument in sorted(fields):
            if argument == "path":
                properties[argument] = ({"type": "string", "enum": ["/synthetic/"]}
                                        if name == "list_dir" else
                                        {"type": "string", "pattern": "^" + _FILE_PATH.pattern.replace(r"\Z", r"(?![\s\S])"),
                                         "maxLength": 75})
            elif argument == "dest":
                properties[argument] = {"type": "string", "enum": [EXPORT_DESTINATION]}
            else:
                properties[argument] = {"type": "string", "maxLength": MAX_TEXT_BYTES}
        schemas.append({
            "type": "function", "function": {
                "name": name, "description": f"Inert synthetic {name}; no real files or transfers.",
                "parameters": {"type": "object", "properties": properties,
                               "required": sorted(fields), "additionalProperties": False},
            },
        })
    return schemas