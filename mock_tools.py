"""Inert synthetic tools: owned memory only, without host I/O or callbacks."""

from live_contracts import (
    MAX_ROUNDS,
    BatchResult,
    BatchValidation,
    CallRecord,
    CallStatus,
    ContractError,
    PlanRecord,
    PlanSource,
    ReasonCode,
    RunResult,
    RunStatus,
    StateSnapshot,
    ToolOutput,
    ToolResult,
    capture_input,
    identifier,
    reject_batch,
    validate_native_batch,
)


class MockTools:
    """One bounded offline episode; rejected batches permanently seal dispatch."""

    def __init__(self, seed: dict[str, str] | None = None, *, run_id="la-1"):
        identifier(run_id)
        if seed is None:
            seed = {
                "/synthetic/record.txt": "Synthetic record: one sample item.",
                "/synthetic/summary.txt": "",
            }
        if type(seed) is not dict:
            raise ContractError(ReasonCode.INVALID_TYPE)
        snapshot = StateSnapshot(tuple(seed.items()))
        self._files = dict(snapshot.files)
        self._run_id = run_id
        self._plans = []
        self._batches = []
        self._seen_ids = frozenset()
        self._next_sequence = 1
        self._last_round = 1
        self._last_turn = 1
        self._sealed = False
        self._status = RunStatus.BLOCKED
        self._reason = ReasonCode.NO_CALLS

    def snapshot(self) -> StateSnapshot:
        return StateSnapshot(tuple(sorted(self._files.items())))

    def run_result(self) -> RunResult:
        return RunResult(self._run_id, tuple(self._plans), tuple(self._batches),
                         self._status, self._reason)

    def _plan_reference(self, plan, round_no, turn_no):
        if plan is None:
            return None
        if type(plan) is not PlanRecord or plan.source is not PlanSource.SYNTHETIC:
            raise ContractError(ReasonCode.INVALID_TYPE)
        if (type(round_no) is not int or type(turn_no) is not int
                or plan.round_no > round_no or plan.turn_no >= turn_no):
            raise ContractError(ReasonCode.INVALID_REFERENCE)
        existing = next((item for item in self._plans if item.plan_id == plan.plan_id), None)
        if existing is not None:
            if existing != plan:
                raise ContractError(ReasonCode.INVALID_REFERENCE)
        elif len(self._plans) >= MAX_ROUNDS:
            raise ContractError(ReasonCode.ROUND_LIMIT)
        else:
            self._plans.append(plan)
        return plan.plan_id

    def execute_batch(self, raw: bytes, *, round_no=1, turn_no=2,
                      plan: PlanRecord | None = None) -> BatchResult:
        """Record and validate a native array, then execute only fixed memory tools."""
        capture = capture_input(raw)
        if self._sealed:
            return BatchResult(BatchValidation(capture, (), None, False, ReasonCode.SEALED), ())
        plan_error = None
        plan_id = None
        try:
            plan_id = self._plan_reference(plan, round_no, turn_no)
        except ContractError as error:
            plan_error = error.code
        validation = validate_native_batch(
            raw, seen_ids=self._seen_ids, next_sequence=self._next_sequence,
            round_no=round_no, turn_no=turn_no, plan_id=plan_id,
        )
        if validation.error is None and plan_error is not None:
            validation = reject_batch(validation, plan_error)
        if validation.error is None and (round_no < self._last_round or turn_no < self._last_turn):
            validation = reject_batch(validation, ReasonCode.INVALID_REFERENCE)
        if validation.error is None:
            projected = self._files.copy()
            try:
                for call in validation.calls:
                    if call.name == "write_file":
                        projected[call.arguments.path] = call.arguments.content
                        StateSnapshot(tuple(projected.items()))
            except ContractError as error:
                validation = reject_batch(validation, error.code)
        if validation.error is not None:
            results = tuple(ToolResult(call.sequence, call.call_id, CallStatus.REJECTED,
                                       reason=call.reason) for call in validation.calls)
            self._sealed = True
            capped = {ReasonCode.CALL_LIMIT, ReasonCode.ROUND_LIMIT, ReasonCode.STATE_LIMIT,
                      ReasonCode.BATCH_TOO_LARGE, ReasonCode.ARGUMENTS_TOO_LARGE,
                      ReasonCode.TEXT_TOO_LARGE, ReasonCode.NESTING_LIMIT}
            self._status = RunStatus.CAPPED if validation.error in capped else RunStatus.BLOCKED
            self._reason = validation.error
        else:
            results = tuple(self._dispatch(call) for call in validation.calls)
            self._seen_ids = self._seen_ids | frozenset(call.call_id for call in validation.calls)
            self._last_round = round_no
            self._last_turn = turn_no
            failure = next((result.reason for result in results if result.reason is not None), None)
            if failure is not None:
                self._status = RunStatus.FAILED
                self._reason = failure
            elif self._status is not RunStatus.FAILED:
                self._status = RunStatus.COMPLETED
                self._reason = ReasonCode.DETECTOR_NOT_RUN
        batch = BatchResult(validation, results)
        self._batches.append(batch)
        self._next_sequence += len(validation.calls)
        return batch

    def _dispatch(self, call: CallRecord) -> ToolResult:
        if type(call) is not CallRecord or call.state is not CallStatus.REQUESTED:
            raise ContractError(ReasonCode.INVALID_TYPE)
        arguments = call.arguments
        if arguments is None:
            raise ContractError(ReasonCode.INVALID_ARGUMENTS)
        if call.name == "list_dir":
            output = ToolOutput(arguments.path, entries=tuple(sorted(self._files)))
        elif call.name == "write_file":
            if arguments.content is None:
                raise ContractError(ReasonCode.INVALID_ARGUMENTS)
            self._files[arguments.path] = arguments.content
            output = ToolOutput(arguments.path, byte_count=len(arguments.content.encode("utf-8")))
        elif call.name in ("read_file", "export"):
            if arguments.path not in self._files:
                return ToolResult(call.sequence, call.call_id, CallStatus.MOCK_FAILED,
                                  reason=ReasonCode.NOT_FOUND)
            content = self._files[arguments.path]
            if call.name == "read_file":
                output = ToolOutput(arguments.path, content=content)
            else:
                output = ToolOutput(arguments.path, dest=arguments.dest,
                                    byte_count=len(content.encode("utf-8")),
                                    receipt_id=f"mock-export-{call.sequence}")
        else:
            raise ContractError(ReasonCode.UNKNOWN_TOOL)
        return ToolResult(call.sequence, call.call_id, CallStatus.MOCK_COMPLETED, output=output)