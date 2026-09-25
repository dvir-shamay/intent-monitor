# Live-agent: offline contracts and inert mock tools

Status: product-engineering substrate only. The full live-agent loop is not
implemented yet.

EXPERIMENTAL / ILLUSTRATIVE. Not validated on real agent traces; no empirical
catch rate or deployment false-positive rate. Observed stated plans are a proxy,
not faithful private CoT. Mock tools only: no real file changes or data transfers.
Live models may refuse or fail to diverge. Detector mode: not_run; no detector
evaluation has run. Errors, misses and false alarms remain possible.

## Ownership And Scope

- [live_contracts.py](../live_contracts.py) owns immutable records, native request
  validation, fresh tool schemas, and qualified summary rendering.
- [mock_tools.py](../mock_tools.py) owns copied synthetic state and four fixed
  dispatch branches. Neither module imports the existing monitor or research code.
- [tests/test_live_contracts.py](../tests/test_live_contracts.py) is the offline
  acceptance suite. Existing modules, fixtures, historical results and the 20
  existing monitor regressions remain unchanged.

Runtime dependencies are Python 3.12's standard library only. Pytest is a
development dependency. Imports and handler calls require no bridge, model,
credentials, research checkout, installation step, or host data.

No actor loop, chat transport, extractor, semantic judge, CLI, preset workflow,
trace persistence, packaging or research evaluation is included. Native requests
in this layer are constructed test input, not captured live-agent behavior.

## Native Input And Validation

`validate_native_batch(raw: bytes, *, seen_ids=frozenset(), next_sequence=1,
round_no=1, turn_no=2, plan_id=None) -> BatchValidation`

The input is a UTF-8 JSON **tool_calls array**, not a chat/HTTP envelope:

```json
[
  {
    "id": "read-1",
    "type": "function",
    "function": {
      "name": "read_file",
      "arguments": "{\"path\":\"/synthetic/record.txt\"}"
    }
  }
]
```

Envelope fields are exactly `id`, `type`, `function`; function fields are exactly
`name`, `arguments`. Arguments must be a JSON-encoded object with precisely the
required string fields. No coercion, extra fields, JSON repair or prose fallback
is performed. Unknown tools, duplicate object keys at either JSON layer, all
numeric tokens (including NaN/infinity), nested argument values, Booleans,
non-object arguments, malformed UTF-8, lone surrogates and trailing data reject.
The depth guard and parser consume the same strictly decoded UTF-8 text.
UTF-16/32 do not become an alternate native-input format.

`tool_schemas() -> list[dict]` returns detached OpenAI-style function definitions
with closed parameter objects. Schemas describe character bounds; Python
validation additionally enforces UTF-8 **byte** bounds. Mutating one returned
schema cannot alter validation or later schema output.

`BatchValidation` is not an execution capability. The public executor accepts
raw bytes and always performs validation itself; it does not accept a caller's
prevalidated batch or caller-selected handler.

## Records And Limits

All records are validated `frozen=True, slots=True` dataclasses. Nested snapshots
contain only other frozen records, tuples, bytes, strings, enums and scalar
values. Mutable mappings/lists are not retained. The caller owns every fresh
rendered JSON object; changing it cannot change recorded state.

| Record | Meaning |
| --- | --- |
| `PlanRecord` | ID, round, turn, original bounded text and plan source |
| `InputCapture` | Original bounded bytes, original byte count and completeness |
| `ToolArguments` | Typed path, optional write content and optional destination |
| `CallRecord` | Harness sequence, native ID, name, round/turn, prior plan ID, original bounded argument text, validated arguments or rejection |
| `BatchValidation` | Capture, ordered calls, observed count, call completeness and finite error |
| `ToolOutput` / `ToolResult` | Immutable synthetic output or typed failure, linked by sequence and native ID, always `mock: true` |
| `BatchResult` | Validation plus one linked result for each retained call |
| `StateSnapshot` | Detached virtual file/content tuples and `total_bytes` |
| `CallVerdict` | Per-call assessment; this layer permits only `UNKNOWN` / `DETECTOR_NOT_RUN` |
| `RunResult` | Run ID, plan snapshots, ordered batches, run status/reason; verdict fixed to `INCONCLUSIVE` |

| Surface | Bound |
| --- | --- |
| Rounds / plans / turns | 6 / 6 / 12 |
| Calls per episode | 8 attempted request slots, including mock failures and retained rejected calls |
| Native batch | 65,536 UTF-8 bytes; envelope nesting at most 4 |
| Argument JSON | 8,192 UTF-8 bytes; flat object only |
| Plan text / each file's content | 4,096 UTF-8 bytes |
| Native IDs / plan IDs / tool names | 1-64 ASCII letters, digits, underscores or hyphens |
| Virtual files / total content | 32 / 65,536 UTF-8 bytes |
| Oversized input preview | At most 256 bytes; original byte count retained |
| Retained batches | At most 8 request-bearing batches plus one terminal rejection |

Raw capture precedes semantic validation. Repeated tool names are never keys:
each retained call observation has a harness sequence. Native ID, name and
argument text can be null on rejected records when missing, malformed or too
large; the bounded raw capture remains the original evidence. No missing identity
is invented. Duplicate IDs within a batch or
against earlier dispatched calls reject the whole batch, retaining distinct
ordered observations. The in-bound original bytes remain available even when
malformed JSON prevents parsing individual calls.

The batch retains the originating error. Individually invalid calls retain their
own reasons; otherwise-valid calls prevented from dispatch by atomic rejection
use `BATCH_REJECTED`, not another call's error such as `UNKNOWN_TOOL`.

For a call-budget overflow, the raw in-bound array remains intact, the original
array count is recorded, and only remaining call slots are expanded into records.
`calls_complete` is false. For byte overflow, only the bounded preview is retained,
`capture.complete` and `calls_complete` are false, and `observed_count` is null
because the array was not parsed. Neither case dispatches a clipped prefix.

Rejection seals the instance. Later calls return an immutable `SEALED` rejection
with bounded capture but are **not appended** to the sealed run. The original
terminal error stays authoritative, and repeated attempts cannot grow memory.
Callers may inspect the returned rejection; reopening requires a new instance.

The public API requires actual `bytes` and typed caller metadata. Invalid seed
or record construction raises `ContractError` containing only a finite
`ReasonCode`, never raw input. Malformed native bytes and quota violations return
typed batch errors rather than parser exceptions.

## Inert Tool Semantics

`MockTools(seed: dict[str, str] | None = None, *, run_id="local")`

The default seed contains only an embedded synthetic record and empty summary.
Explicit seed dictionaries are validated and copied. No workspace is read.

`execute_batch(raw: bytes, *, round_no=1, turn_no=2, plan=None) -> BatchResult`

`snapshot() -> StateSnapshot`

`run_result() -> RunResult`

| Tool | Exact Arguments | Result / Effect |
| --- | --- | --- |
| `read_file` | `path` | Synthetic content or `NOT_FOUND`; no host reads |
| `list_dir` | `path` | Sorted tuple of current virtual file IDs |
| `write_file` | `path`, `content` | Replace/create virtual content; return byte count |
| `export` | `path`, `dest` | In-memory receipt identifying the existing virtual file, destination and byte count; no transfer |

File IDs match `/synthetic/[A-Za-z0-9][A-Za-z0-9._-]{0,63}` exactly. The only
directory is `/synthetic/`. The only destination is exactly
`https://reports.example/ingest`. These are opaque identifiers, never resolved,
normalized, decoded as paths, contacted, or passed to OS/HTTP APIs. Nested paths,
traversal, Windows drive/UNC paths, streams, URL-encoded traversal and arbitrary
command/code fields are rejected. Command-like file **content** remains inert data.

The entire batch, prior IDs, plan reference and projected state after every
write are checked **before any handler invocation**. Two writes that individually
fit but jointly exceed a quota reject atomically. Replacements count current
projected bytes, not a cumulative sum of old and new content.

Valid calls execute in array order, so a later read/export can observe an earlier
virtual write. Missing files are `mock_failed` / `NOT_FOUND`, not malformed native
calls; other valid calls may still execute. There is no rollback promise for a
valid batch with a mock outcome failure. Run status stays failed after such a
failure even if later mock calls succeed. Requested, rejected, mock-completed
and mock-failed states do not describe real-world file operations or transfers.

Plan snapshots in this offline executor must have `source=synthetic`, precede
the action turn, and not reuse an ID with changed content. Round/turn metadata
cannot move backward. Run records check call/result and plan references. This
does not implement an actor's temporal capture loop or semantic licensing.

## Qualified Results

`render_text(result: RunResult) -> str`

`render_json(result: RunResult) -> str`

Both consume the same closed summary schema, including error and empty-run
results. They always include:

- `validation_status: "unvalidated_on_real_traces"` and the notice above;
- `trace_source: "constructed"`, `detector_mode: "not_run"`;
- null `agent_model`, `extractor_model`, `judge_model`;
- actual `plan_source: "synthetic"` or `"absent"`, plus per-call plan IDs;
- `mock_tools: true`, per-result `mock: true`, and explicit limitations;
- separate `run_status`, `run_reason`, `verdict`, and `verdict_reason`.

Both formats expose per-batch original byte/count information, retained call
count, capture/call completeness and the originating error. Every call names its
batch and prior plan ID, with explicit null for an absent plan. This distinguishes
an incomplete captured batch from a fully observed rejection in text as well as
JSON.

The six run-status enums are completed/refused/failed/capped/interrupted/blocked.
This layer produces completed/failed/capped/blocked for mock execution only. It never
produces `NO_OBSERVED_DIVERGENCE` or `FLAGGED`: all run verdicts are `INCONCLUSIVE`
and all per-call assessments are `UNKNOWN`, because no detector ran. Enums for
future determinate verdicts do not enable a detector in this increment.

Summaries omit raw native bytes, plan text, argument strings and tool content.
Untrusted controls therefore cannot enter terminal text through those fields;
remaining identifiers and reasons are allowlisted. JSON uses `ensure_ascii=True`
and `allow_nan=False`. There is no write/export-to-disk facility. Raw captures
exist only in the caller's in-memory records and should be treated as untrusted
data. These controls are not an OS sandbox against a hostile local process.

## Offline Example

From the product directory, using only the new modules:

```python
>>> from live_contracts import PlanRecord, render_json
>>> from mock_tools import MockTools
>>> tools = MockTools({"/synthetic/record.txt": "Synthetic record"})
>>> plan = PlanRecord("plan-1", 1, 1, "Read the synthetic record.")
>>> raw = b'[{"id":"read-1","type":"function","function":{"name":"read_file","arguments":"{\\"path\\":\\"/synthetic/record.txt\\"}"}}]'
>>> batch = tools.execute_batch(raw, plan=plan)
>>> batch.results[0].output.content
'Synthetic record'
>>> batch.results[0].mock
True
>>> import json
>>> summary = json.loads(render_json(tools.run_result()))
>>> summary["run_status"], summary["verdict"], summary["detector_mode"]
('completed', 'INCONCLUSIVE', 'not_run')
>>> summary["validation_status"]
'unvalidated_on_real_traces'

```

## Verification And Review

From the product directory:

```powershell
python -B -m pytest tests/test_live_contracts.py -q -p no:cacheprovider
python -B -m pytest tests -q -p no:cacheprovider
python -B -S -m doctest docs/LIVE-CONTRACTS.md
```

Tests cover each handler, retained duplicate observations, strict decoding,
schema/size boundaries, seed/schema/snapshot aliasing, aggregate projected
quotas, sealed rejection, plan references and exact disclosure/error semantics.
The isolation test patches file reads/mutations, subprocess/shell creation, DNS,
socket and HTTP APIs around invocation. Every installed deny hook is first
exercised as a positive control; handler success and rejection paths must then
produce zero hits. No model or bridge is needed. These are engineering checks,
not detector accuracy evidence.

The offline suite passes 110 tests: 90 contract cases and all 20 existing monitor
regressions, plus 12 documentation doctest checks with site packages disabled.

The architect and eight-role design review closed with PASS. Design findings
LA1-SEC-01 and TW-1 through TW-4 are FIXED by strict common-text UTF-8 parsing,
surrogate rejection, atomicity/sealing assertions, retained-capture assertions,
aggregate-quota checks and exact truthful-rendering checks.

Implementation review dispositions:

| Findings | Disposition | Resolution |
| --- | --- | --- |
| LA1-CR-01, LA1-QA-01, DBG-01 | FIXED | Both public argument/output records require exact strings before directory/destination equality; four mutable-wrapper regressions |
| TW-5, TW-6 | FIXED | Literal native-schema limits and initial-state file/content/aggregate quota tests |
| LA1-QA-02 | FIXED | Current full offline suite includes all 20 unchanged existing regressions |
| LA1-QA-03, LA1-DOC-01, LA1-UX-03 | FIXED | Text and JSON identify every call's batch and plan ID or explicit null |
| LA1-DOC-02 | FIXED | Documented/tested nullable identity fields for malformed retained observations |
| LA1-UX-01 | FIXED | Both renderers expose byte counts, observed/retained counts and completeness |
| LA1-UX-02 | FIXED | Otherwise-valid blocked calls use BATCH_REJECTED, preserving the original batch/individual error |
| TW7 | FIXED | Batch-reason test targets the batch line, not an incidental matching header substring |