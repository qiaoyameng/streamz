"""Explicit checkpoint barriers for streamz graphs.

This module implements a first version of checkpointing for local streamz
graphs, supporting the ``map``/``starmap``/``filter``, ``accumulate``,
``partition`` and ``zip`` operators.

A :class:`CheckpointBarrier` with a given checkpoint id is injected at every
source of the graph.  Operators forward the barrier downstream once their own
state is consistent with it; multi-input operators such as ``zip`` align the
barrier across all of their inputs: an input that has received the barrier
first is not allowed to consume data past the barrier until the barrier has
arrived on every input.  Once all sources have emitted the barrier (emission
is synchronous), the state of every operator reflects exactly the data of the
finished epoch and is written to disk.

The checkpoint records the graph topology (node types, edges and the
registered names of user functions) together with a per-operator state
version.  Restoring into a structurally incompatible graph raises
:class:`CheckpointCompatibilityError`.  User functions are referenced by
application-registered names only (see :func:`register_function`); arbitrary
executable objects are never serialized into or deserialized from the
checkpoint.

Exactly-once delivery to external sinks is *not* covered by this mechanism.

Examples
--------
>>> from streamz import Stream
>>> from streamz.checkpoint import Checkpointer, register_function

>>> @register_function
... def inc(x):
...     return x + 1

>>> source = Stream()
>>> out = source.map(inc).sink_to_list()
>>> ck = Checkpointer(source, path="checkpoint.pkl")
>>> source.emit(1)
>>> ck.checkpoint(1)  # doctest: +SKIP
>>> # after a restart, rebuild the same graph, then:
>>> # Checkpointer(source, path="checkpoint.pkl").restore()
"""
import pickle
from collections import deque

FORMAT_VERSION = 1


class CheckpointError(Exception):
    """Base error for checkpointing failures."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised when a checkpoint does not match the graph being restored."""


class CheckpointBarrier:
    """Marker element injected at sources to trigger a checkpoint.

    Barriers flow through the graph like regular elements but are never
    passed to user functions.  All sources of a graph must emit a barrier
    with the same ``checkpoint_id``; multi-input operators block inputs that
    have already seen the barrier until every input has received it, so that
    data from before and after the barrier is never mixed.
    """

    __slots__ = ("checkpoint_id",)

    def __init__(self, checkpoint_id):
        self.checkpoint_id = checkpoint_id

    def __repr__(self):
        return "<CheckpointBarrier {}>".format(self.checkpoint_id)


_function_registry = {}
_function_ids = {}


def register_function(func=None, *, name=None):
    """Register a user function so checkpoints can reference it by name.

    Can be used as a plain decorator or with an explicit name::

        @register_function
        def inc(x):
            return x + 1

        @register_function(name="is-even")
        def is_even(x):
            return x % 2 == 0

    Only registered functions may be used in checkpointed operators; the
    checkpoint stores the registered name, never the function object itself.
    """
    if func is None:
        return lambda f: register_function(f, name=name)
    fname = name or getattr(func, "__name__", None)
    if not fname:
        raise CheckpointError(
            "Cannot register a checkpoint function without a name; "
            "pass name=... explicitly"
        )
    existing = _function_registry.get(fname)
    if existing is not None and existing is not func:
        raise CheckpointError(
            "A different function is already registered under the "
            "name {!r}".format(fname)
        )
    _function_registry[fname] = func
    _function_ids[id(func)] = fname
    return func


def registered_function_name(func):
    """Return the registered name of ``func``.

    Raises
    ------
    CheckpointError
        If the function was not registered with :func:`register_function`.
    """
    try:
        return _function_ids[id(func)]
    except KeyError:
        raise CheckpointError(
            "Function {!r} is used in the graph but was not registered. "
            "Register it with streamz.checkpoint.register_function so the "
            "checkpoint can reference it by name.".format(func)
        )


def get_registered_function(name):
    """Look up a function previously stored under ``name``."""
    try:
        return _function_registry[name]
    except KeyError:
        raise CheckpointError(
            "Checkpoint references function {!r}, but no function is "
            "registered under that name in this process.".format(name)
        )


def _traverse(sources):
    """Walk the connected graph from ``sources`` in deterministic order.

    Returns the list of nodes and a mapping of ``id(node) -> index``.  Both
    downstreams and upstreams are followed so that every node feeding into
    the graph (e.g. all parents of a ``zip``) is included.
    """
    order = []
    seen = {}
    queue = deque(sources)
    while queue:
        node = queue.popleft()
        if node is None or id(node) in seen:
            continue
        seen[id(node)] = len(order)
        order.append(node)
        queue.extend(d for d in node.downstreams if d is not None)
        queue.extend(u for u in node.upstreams if u is not None)
    return order, seen


def _describe_topology(order, seen):
    """Build a serializable fingerprint of the graph structure.

    Every entry records the node type, the indices of its upstream nodes and
    the registered names of its user functions.
    """
    topology = []
    for node in order:
        functions = {}
        for attr in getattr(node, "_checkpoint_func_attrs", ()):
            func = getattr(node, attr, None)
            if callable(func):
                functions[attr] = registered_function_name(func)
        topology.append({
            "type": type(node).__name__,
            "upstreams": [seen[id(u)] for u in node.upstreams if u is not None],
            "functions": functions,
        })
    return topology


class Checkpointer:
    """Inject checkpoint barriers and persist/restore operator state.

    Parameters
    ----------
    *sources: Stream
        Every source of the graph.  All sources must be provided so that
        barriers with the same id are injected on every input path and
        multi-input operators can align them.
    path: str
        Default file the checkpoint is written to / restored from.

    Examples
    --------
    >>> from streamz import Stream
    >>> source = Stream()
    >>> L = source.map(str).sink_to_list()
    >>> ck = Checkpointer(source, path="state.pkl")
    >>> source.emit(1)
    >>> ck.checkpoint(1)  # doctest: +SKIP
    >>> # in a fresh process, after rebuilding the same graph:
    >>> Checkpointer(source, path="state.pkl").restore()  # doctest: +SKIP
    """

    def __init__(self, *sources, path):
        if not sources:
            raise ValueError("At least one source stream is required")
        self.sources = sources
        self.path = path

    def checkpoint(self, checkpoint_id, path=None):
        """Emit a barrier from every source and write the aligned state.

        Emission is synchronous, so once this method returns every operator
        has seen the barrier and the checkpoint file has been written.
        """
        path = path if path is not None else self.path
        order, seen = _traverse(self.sources)
        barrier = CheckpointBarrier(checkpoint_id)
        for source in self.sources:
            source.emit(barrier)
        states = {}
        for index, node in enumerate(order):
            state = node._checkpoint_state()
            if state is not None:
                states[str(index)] = {
                    "state_version": node._state_version,
                    "state": state,
                }
        payload = {
            "format_version": FORMAT_VERSION,
            "checkpoint_id": checkpoint_id,
            "topology": _describe_topology(order, seen),
            "states": states,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        return path

    def restore(self, path=None):
        """Restore operator state from a checkpoint file.

        The graph reachable from ``self.sources`` must be structurally
        identical to the one the checkpoint was taken from (same node types,
        same edges, same registered function names and state versions),
        otherwise :class:`CheckpointCompatibilityError` is raised.

        Returns the checkpoint id stored in the file.
        """
        path = path if path is not None else self.path
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if payload.get("format_version") != FORMAT_VERSION:
            raise CheckpointCompatibilityError(
                "Unsupported checkpoint format version: {!r}".format(
                    payload.get("format_version")
                )
            )
        order, seen = _traverse(self.sources)
        topology = _describe_topology(order, seen)
        saved = payload["topology"]
        if len(saved) != len(topology):
            raise CheckpointCompatibilityError(
                "Checkpoint topology has {} nodes but the current graph "
                "has {}".format(len(saved), len(topology))
            )
        for index, (want, got) in enumerate(zip(saved, topology)):
            if want != got:
                raise CheckpointCompatibilityError(
                    "Graph is incompatible with checkpoint at node {}: "
                    "checkpoint has {!r}, current graph has {!r}".format(
                        index, want, got
                    )
                )
        # Every referenced function name must resolve in this process.
        for entry in saved:
            for name in entry["functions"].values():
                get_registered_function(name)
        for index, node in enumerate(order):
            entry = payload["states"].get(str(index))
            if entry is None:
                continue
            if entry["state_version"] != node._state_version:
                raise CheckpointCompatibilityError(
                    "State version mismatch at node {} ({}): checkpoint has "
                    "version {}, operator supports version {}".format(
                        index, entry["type"], entry["state_version"],
                        node._state_version,
                    )
                )
            node._restore_state(entry["state"])
        return payload["checkpoint_id"]
