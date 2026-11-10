""" Bounded iteration (feedback) controller for finite streamz subgraphs.

Some data-cleaning style pipelines need to run the same set of rules over
a batch repeatedly until the output stops changing.  ``IterationController``
drives such a loop over a *finite* subgraph without stopping the rest of
the process: the caller declares an entry stream, one or more exit streams,
an initial batch and a convergence predicate.

Each round the controller emits the current batch into the entry stream,
one element at a time, collects whatever the declared exits produce, and
waits until every emit *and* every asynchronous branch (e.g. a ``sink``
whose function returns an awaitable) of that element has finished.  The
collected outputs - ordered by entry sequence number and, across branches,
by the declared exit order - form the batch of the next round.

Only operators that never delay emission on their own are allowed inside
the controlled subgraph: ``map``, ``filter``, ``flatten``, ``union`` and
plain ``sink``.  Windowing and queueing operators (``timed_window``,
``partition``, ``buffer``, ``collect``, ``zip``, ``map_async`` ...) hold
elements back by themselves, so a round could never be declared finished
reliably; they are rejected up front with a ``ValueError``.

Only one iteration batch may run at a time per controller, and external
emissions into the subgraph are not mixed into the collected outputs.
If a round fails or is cancelled, all metadata references are released
and no feedback task is left behind.

Example
-------
>>> source = Stream(asynchronous=True)                # doctest: +SKIP
>>> cleaned = source.map(normalize)                   # doctest: +SKIP
>>> controller = IterationController(                 # doctest: +SKIP
...     source, [cleaned],
...     converged=lambda previous, current: previous == current,
...     max_rounds=10)
>>> result = await controller.run(dirty_records)      # doctest: +SKIP
>>> result.converged, result.outputs                  # doctest: +SKIP
"""
from dataclasses import dataclass, field
from typing import Any, List

from . import core
from .core import Stream, RefCounter
from .sinks import sink as _sink

__all__ = ['IterationController', 'IterationResult', 'RoundRecord', 'iterate']

_TOKEN_KEY = 'iteration_token'

# Operators that pass elements on synchronously and therefore allow a round
# to terminate once the input batch has been fully emitted.
_ALLOWED_TYPES = (core.map, core.filter, core.flatten, core.union, _sink)


@dataclass
class RoundRecord:
    """ Inputs and collected outputs of a single iteration round. """
    index: int
    inputs: List[Any]
    outputs: List[Any]


@dataclass
class IterationResult:
    """ Outcome of an ``IterationController.run`` call.

    Attributes
    ----------
    converged: bool
        True when the run terminated normally, either because the
        convergence predicate accepted or because a round produced no
        output at all.  False when ``max_rounds`` was reached first.
    reason: str
        One of ``'converged'``, ``'empty'`` or ``'max_rounds'``.
    rounds: int
        Number of rounds that were executed.
    outputs: list
        Outputs collected during the final round.
    history: list of RoundRecord
        Per-round inputs and outputs, in execution order.
    """
    converged: bool
    reason: str
    rounds: int
    outputs: List[Any]
    history: List[RoundRecord] = field(default_factory=list)


class _ExitCollector(Stream):
    """ Internal stream attached to each declared exit.

    Records elements while the owning controller is running a round.
    Elements that do not belong to the controller (external emissions)
    are ignored: controller emissions carry a per-run token in their
    metadata, and untagged elements (e.g. intermediate ``flatten`` items,
    which streamz emits without metadata) are only accepted while the
    controller is synchronously emitting.
    """
    def __init__(self, upstream, controller, position):
        self._controller = controller
        self._position = position
        Stream.__init__(self, upstream,
                        stream_name='iteration-exit-%d' % position)

    def update(self, x, who=None, metadata=None):
        controller = self._controller
        if controller._accepts(metadata):
            controller._round_outputs[self._position].append(x)
        return []


class IterationController:
    """ Run a finite subgraph repeatedly until its output converges.

    Parameters
    ----------
    entry: Stream
        The stream into which batch elements are emitted.  Every node
        reachable from it (up to the exits) must be a ``map``, ``filter``,
        ``flatten``, ``union`` or plain ``sink`` node on the same IOLoop.
    exits: Stream or sequence of Stream
        Declared exit streams.  Their outputs are collected per round and,
        concatenated in the declared order, form the next batch.
    converged: callable
        Predicate called as ``converged(previous_outputs, current_outputs)``
        after each non-empty round (``previous_outputs`` is ``None`` on the
        first round).  It is only called once the round has fully finished,
        including pending asynchronous branches.  Return True to stop.
    max_rounds: int, optional
        Maximum number of rounds before giving up and returning a
        non-converged result.  Defaults to 100.
    """
    def __init__(self, entry, exits, converged, max_rounds=100):
        if isinstance(exits, Stream):
            exits = [exits]
        exits = list(exits)
        if not isinstance(entry, Stream):
            raise TypeError("entry must be a Stream, got %r" % (entry,))
        if not exits:
            raise ValueError("at least one exit stream is required")
        if not callable(converged):
            raise TypeError("converged must be callable")
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        self._entry = entry
        self._exits = exits
        self._converged = converged
        self._max_rounds = max_rounds
        self._running = False
        self._emitting = False
        self._token = None
        self._refs = []
        self._round_outputs = None
        self._validate()
        self._collectors = [
            _ExitCollector(exit, self, position)
            for position, exit in enumerate(exits)
        ]

    @property
    def entry(self):
        return self._entry

    @property
    def exits(self):
        return list(self._exits)

    @property
    def running(self):
        """ Whether an iteration batch is currently in flight. """
        return self._running

    def _validate(self):
        exit_ids = {id(exit) for exit in self._exits}
        seen = {}
        stack = [self._entry]
        while stack:
            node = stack.pop()
            if id(node) in seen:
                continue
            if isinstance(node, _ExitCollector):
                continue
            if not (type(node) is Stream or isinstance(node, _ALLOWED_TYPES)):
                raise ValueError(
                    "IterationController subgraphs may only contain map, "
                    "filter, flatten, union and plain sink nodes; got %r. "
                    "Windowing and queueing operators (timed_window, "
                    "sliding_window, partition, buffer, collect, zip, "
                    "map_async, ...) delay emission on their own, so a "
                    "round could never be declared finished."
                    % (node,))
            seen[id(node)] = node
            if id(node) in exit_ids:
                continue
            stack.extend(list(node.downstreams))
        for exit in self._exits:
            if id(exit) not in seen:
                raise ValueError(
                    "exit %r is not reachable from the entry stream"
                    % (exit,))
        loops = {id(node.loop) for node in seen.values()
                 if node.loop is not None}
        if len(loops) > 1:
            raise ValueError(
                "all nodes of the controlled subgraph must share one IOLoop")

    def _accepts(self, metadata):
        if not self._running:
            return False
        if metadata:
            return any(m.get(_TOKEN_KEY) is self._token for m in metadata)
        # Untagged element (e.g. an intermediate flatten item): only accept
        # it while this controller is synchronously emitting, so external
        # input can never mix into the controlled subgraph.
        return self._emitting

    async def run(self, initial_batch):
        """ Iterate over ``initial_batch`` until convergence.

        Returns an ``IterationResult``.  Raises ``RuntimeError`` if this
        controller is already running a batch.  Exceptions raised inside
        the subgraph propagate to the caller after metadata references
        have been released; cancellation is handled the same way.
        """
        if self._running:
            raise RuntimeError(
                "this controller is already running; only one iteration "
                "batch may run at a time")
        self._running = True
        self._token = object()
        self._refs = []
        history = []
        previous = None
        batch = list(initial_batch)
        round_index = 0
        try:
            while True:
                outputs = await self._run_round(batch)
                history.append(RoundRecord(index=round_index,
                                           inputs=list(batch),
                                           outputs=list(outputs)))
                round_index += 1
                if not outputs:
                    return IterationResult(converged=True, reason='empty',
                                           rounds=round_index, outputs=[],
                                           history=history)
                if self._converged(previous, outputs):
                    return IterationResult(converged=True,
                                           reason='converged',
                                           rounds=round_index,
                                           outputs=list(outputs),
                                           history=history)
                if round_index >= self._max_rounds:
                    return IterationResult(converged=False,
                                           reason='max_rounds',
                                           rounds=round_index,
                                           outputs=list(outputs),
                                           history=history)
                previous = outputs
                batch = outputs
        finally:
            # Release any metadata references still held (failure or
            # cancellation mid-round) so nothing is left dangling.
            for ref in self._refs:
                if ref.count > 0:
                    ref.release(ref.count)
            self._running = False
            self._emitting = False
            self._round_outputs = None

    async def _run_round(self, batch):
        self._round_outputs = [[] for _ in self._exits]
        for x in batch:
            # One reference per emitted element: the emitter holds one
            # count, the subgraph retains/releases the rest as the element
            # propagates.  A non-zero count after a failed round means the
            # graph leaked a reference, which we release below.
            ref = RefCounter(initial=1, loop=self._entry.loop)
            self._refs.append(ref)
            metadata = [{'ref': ref, _TOKEN_KEY: self._token}]
            try:
                self._emitting = True
                try:
                    future = self._entry.emit(x, asynchronous=True,
                                              metadata=metadata)
                finally:
                    self._emitting = False
                if future is not None:
                    # Wait until this emit and all of its asynchronous
                    # branches have finished before touching the next
                    # element or the convergence predicate.
                    await future
            except BaseException:
                if ref.count > 0:
                    ref.release(ref.count)
                raise
            ref.release()
        return [item
                for outputs in self._round_outputs
                for item in outputs]


async def iterate(entry, exits, initial_batch, converged, max_rounds=100):
    """ Convenience wrapper around ``IterationController``.

    Builds a controller for ``entry``/``exits`` and runs one iteration
    batch.  See ``IterationController`` for the semantics.
    """
    controller = IterationController(entry, exits, converged,
                                     max_rounds=max_rounds)
    return await controller.run(initial_batch)
