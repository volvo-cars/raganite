#
# Copyright 2026 Volvo Car Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run hooks that total elapsed time and chat-token usage."""

from __future__ import annotations

import time
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from threading import Lock
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.tracers.context import register_configure_hook


@dataclass
class StageUsage:
    """Wall time, model time, and summed token usage for one stage."""

    elapsed_seconds: float = 0.0
    model_seconds: float = 0.0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_token_details: dict[str, int] = field(default_factory=dict)
    output_token_details: dict[str, int] = field(default_factory=dict)


@dataclass
class UsageReport:
    """Token and timing totals for one indexing, retrieval, or benchmark run."""

    name: str = ""
    elapsed_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_token_details: dict[str, int] = field(default_factory=dict)
    output_token_details: dict[str, int] = field(default_factory=dict)
    stages: dict[str, StageUsage] = field(default_factory=dict)

    def stage(self, name: str) -> StageUsage:
        return self.stages.setdefault(name, StageUsage())

    def rollup(self) -> None:
        stages = self.stages.values()
        self.input_tokens = sum(stage.input_tokens for stage in stages)
        self.output_tokens = sum(stage.output_tokens for stage in stages)
        self.total_tokens = sum(stage.total_tokens for stage in stages)
        self.input_token_details = _merge_details(*(stage.input_token_details for stage in stages))
        self.output_token_details = _merge_details(
            *(stage.output_token_details for stage in stages)
        )


HookFactory = Callable[[UsageReport], BaseCallbackHandler]

_factories: list[HookFactory] = []
_handler_var: ContextVar[BaseCallbackHandler | None] = ContextVar(
    "raganite_run_hooks", default=None
)
_run_var: ContextVar[UsageReport | None] = ContextVar("raganite_run", default=None)
_stage_var: ContextVar[str] = ContextVar("raganite_usage_stage", default="")
_installed = False


def _merge_details(*groups: Mapping[str, int]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for group in groups:
        for key, value in group.items():
            merged[key] = merged.get(key, 0) + int(value or 0)
    return merged


def _add_metadata(stage: StageUsage, metadata: Mapping[str, Any] | None) -> None:
    if not metadata:
        return
    stage.input_tokens += int(metadata.get("input_tokens") or 0)
    stage.output_tokens += int(metadata.get("output_tokens") or 0)
    stage.total_tokens += int(metadata.get("total_tokens") or 0)
    stage.input_token_details = _merge_details(
        stage.input_token_details, metadata.get("input_token_details") or {}
    )
    stage.output_token_details = _merge_details(
        stage.output_token_details, metadata.get("output_token_details") or {}
    )


def _usage_from(response: LLMResult) -> Mapping[str, Any] | None:
    try:
        generation = response.generations[0][0]
    except IndexError:
        return None
    if isinstance(generation, ChatGeneration) and isinstance(generation.message, AIMessage):
        return generation.message.usage_metadata
    return None


class UsageRunHook(BaseCallbackHandler):
    """Sum ``usage_metadata`` and model time onto the active run stage."""

    def __init__(self, report: UsageReport) -> None:
        super().__init__()
        self.report = report
        self._lock = Lock()
        self._calls: dict[UUID, float] = {}

    def _mark(self, run_id: UUID) -> None:
        with self._lock:
            self._calls.setdefault(run_id, time.perf_counter())

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._mark(run_id)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._mark(run_id)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        metadata = _usage_from(response)
        with self._lock:
            started = self._calls.pop(run_id, None)
            model_seconds = 0.0 if started is None else time.perf_counter() - started
            report = _run_var.get()
            if report is None:
                return
            name = _stage_var.get() or report.name or "run"
            stage = report.stage(name)
            stage.calls += 1
            stage.model_seconds += model_seconds
            _add_metadata(stage, metadata)
            report.rollup()

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            self._calls.pop(run_id, None)

    def add_stage_time(self, name: str, elapsed: float) -> None:
        with self._lock:
            report = _run_var.get()
            if report is not None:
                report.stage(name).elapsed_seconds += elapsed

    def absorb(self, parent: UsageReport, child: UsageReport) -> None:
        with self._lock:
            _absorb(parent, child)


class _Fanout(BaseCallbackHandler):
    """Forward one LangChain run to every hook registered for it."""

    def __init__(self, hooks: list[BaseCallbackHandler], usage: UsageRunHook) -> None:
        super().__init__()
        self.hooks = hooks
        self.usage = usage

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], *, run_id: UUID, **kwargs: Any
    ) -> None:
        for hook in self.hooks:
            hook.on_llm_start(serialized, prompts, run_id=run_id, **kwargs)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        for hook in self.hooks:
            hook.on_chat_model_start(serialized, messages, run_id=run_id, **kwargs)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        for hook in self.hooks:
            hook.on_llm_end(response, run_id=run_id, **kwargs)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        for hook in self.hooks:
            hook.on_llm_error(error, run_id=run_id, **kwargs)


def _install() -> None:
    global _installed
    if _installed:
        return
    register_configure_hook(_handler_var, inheritable=True)
    _installed = True


def register_run_hook(factory: HookFactory) -> None:
    """Register a hook factory. Each tracked run builds one callback from it.

    LangChain attaches that callback to every chat-model run inside the block,
    so the hook sees ``usage_metadata`` without each call site reading it.
    """
    _install()
    _factories.append(factory)


def unregister_run_hook(factory: HookFactory) -> None:
    """Remove a factory registered with ``register_run_hook``."""
    _factories.remove(factory)


def _usage_factory(report: UsageReport) -> UsageRunHook:
    return UsageRunHook(report)


def _absorb(parent: UsageReport, child: UsageReport) -> None:
    for name, stage in child.stages.items():
        target = parent.stage(name)
        target.elapsed_seconds += stage.elapsed_seconds
        target.model_seconds += stage.model_seconds
        target.calls += stage.calls
        target.input_tokens += stage.input_tokens
        target.output_tokens += stage.output_tokens
        target.total_tokens += stage.total_tokens
        target.input_token_details = _merge_details(
            target.input_token_details, stage.input_token_details
        )
        target.output_token_details = _merge_details(
            target.output_token_details, stage.output_token_details
        )
    parent.rollup()


@contextmanager
def track_run(name: str) -> Generator[UsageReport, None, None]:
    """Open a timed run and attach the registered hooks to model calls inside it.

    A run already in progress stays the hook owner. This block still gets its own
    report, and its stages are copied onto the outer run when the block ends.
    """
    parent = _run_var.get()
    report = UsageReport(name=name)
    handler_token = None
    if parent is None:
        hooks: list[BaseCallbackHandler] = [factory(report) for factory in list(_factories)]
        usage = next(hook for hook in hooks if isinstance(hook, UsageRunHook))
        handler_token = _handler_var.set(_Fanout(hooks, usage))
    run_token = _run_var.set(report)
    started = time.perf_counter()
    try:
        yield report
    finally:
        report.elapsed_seconds = time.perf_counter() - started
        if parent is not None:
            handler = _handler_var.get()
            if isinstance(handler, _Fanout):
                handler.usage.absorb(parent, report)
            else:
                _absorb(parent, report)
        _run_var.reset(run_token)
        if handler_token is not None:
            _handler_var.reset(handler_token)


@contextmanager
def usage_stage(name: str) -> Generator[None, None, None]:
    """Attribute model calls and wall time to one stage of the current run."""
    token = _stage_var.set(name)
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        handler = _handler_var.get()
        if isinstance(handler, _Fanout):
            handler.usage.add_stage_time(name, elapsed)
        _stage_var.reset(token)


def combine_usage(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum usage dictionaries, such as one benchmark row per question."""
    combined = UsageReport(name="combined")
    for payload in payloads:
        child = UsageReport(
            name=str(payload.get("name") or ""),
            elapsed_seconds=float(payload.get("elapsed_seconds") or 0.0),
            stages={
                name: StageUsage(**dict(stage))
                for name, stage in dict(payload.get("stages") or {}).items()
            },
        )
        child.rollup()
        _absorb(combined, child)
        combined.elapsed_seconds += child.elapsed_seconds
    return asdict(combined)


register_run_hook(_usage_factory)
_install()
