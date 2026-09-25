""" Explicit checkpoint barriers for local streamz graphs.

This module provides a first version of fault-tolerant state checkpointing
for local (non-dask) streamz graphs built from ``map``, ``filter``,
``accumulate``, ``partition`` and ``zip`` (plus pass-through nodes and
sinks).

The model follows the classic checkpoint-barrier approach:

* :func:`save_checkpoint` injects a :class:`~streamz.core.CheckpointBarrier`
  with the same checkpoint id at every source of the graph.
* The barrier flows through the graph like a data element.  Single-input
  operators snapshot their state as soon as the barrier passes through.
  Multi-input operators (``zip``) align the barrier: an input that has
  already seen the barrier may not consume post-barrier data until all
  other inputs have seen it too, so that data from before and after the
  barrier never mix.
* Once the barrier has traversed the graph, the collected operator states
  together with a description of the graph topology are written to disk.

User functions are never serialized.  The checkpoint refers to them by the
name under which the application registered them with
:func:`register_function`; restoring a checkpoint resolves those names
against the registry of the current process.  Functions defined inside
``streamz`` itself are referenced by their qualified module path.

A checkpoint records the graph topology (node ids, node types, registered
function names, upstream connections) and a state-layout version per
operator.  :func:`restore_checkpoint` refuses to load a checkpoint into a
graph whose topology or state versions are incompatible by raising
:class:`CheckpointCompatibilityError`.

Exactly-once delivery to external sinks is *not* covered by this
mechanism; barriers simply terminate at sinks.

Examples
--------
>>> from streamz import Stream
>>> from streamz.checkpoint import (save_checkpoint, restore_checkpoint,
...                                 register_function)

>>> @register_function("add")
... def add(acc, x):
...     return acc + x

>>> source = Stream(stream_name="source")
>>> total = source.accumulate(add, start=0, stream_name="total")
>>> _ = total.sink(print)
>>> for i in range(3):
...     _ = source.emit(i)
0
1
3
>>> save_checkpoint(source, "state.chk")  # doctest: +SKIP

After a restart, rebuild the same graph, then:

>>> restore_checkpoint(source, "state.chk")  # doctest: +SKIP
"""

import pickle
import uuid
from collections import deque

from .core import (
    CheckpointBarrier,
    Stream,
    accumulate,
    filter as _filter,
    map as _map,
    partition,
    zip as _zip,
)
from .sinks import sink as _sink

__all__ = [
    "CheckpointError",
    "CheckpointCompatibilityError",
    "register_function",
    "registered_functions",
    "save_checkpoint",
    "restore_checkpoint",
]

#: Version of the checkpoint file format itself.
_FORMAT_VERSION = 1


class CheckpointError(Exception):
    """ An error occurred while creating a checkpoint. """


class CheckpointCompatibilityError(CheckpointError):
    """ A checkpoint cannot be restored into an incompatible graph. """


_function_registry = {}


def register_function(name, func=None):
    """ Register a user function under an application-chosen name.

    Checkpoints reference user functions by this name only, so any function
    used in a checkpointed graph must be registered.  Can be used as a
    decorator or called directly with a function.
    """
    if func is None:
        def decorator(f):
            _function_registry[name] = f
            return f
        return decorator
    _function_registry[name] = func
    return func


def registered_functions():
    """ Return a copy of the current function registry. """
    return dict(_function_registry)


# Operator classes with explicit checkpoint-barrier support.
_BARRIER_AWARE = (_map, _filter, accumulate, partition, _zip, _sink)


def _check_node_supported(node):
    if isinstance(node, _BARRIER_AWARE):
        return
    if type(node).update is Stream.update:
        # pass-through node (e.g. plain sources): forwards barriers as-is
        return
    raise CheckpointError(
        "Stream node %r of type %s does not support checkpoint barriers; "
        "supported operators are map, filter, accumulate, partition and zip"
        % (node, type(node).__name__))


def _find_sources(node):
    """ Walk upstreams from ``node`` and return all source nodes. """
    sources = []
    seen = set()
    queue = deque([node])
    while queue:
        n = queue.popleft()
        if id(n) in seen:
            continue
        seen.add(id(n))
        upstreams = [u for u in n.upstreams if u is not None]
        if not upstreams:
            sources.append(n)
        queue.extend(upstreams)
    return sources


def _walk_graph(sources):
    """ Deterministically order all nodes reachable from ``sources``. """
    order = []
    seen = set()
    queue = deque(sources)
    while queue:
        n = queue.popleft()
        if id(n) in seen:
            continue
        seen.add(id(n))
        order.append(n)
        queue.extend(d for d in n.downstreams if d is not None)
    return order


def _node_ids(order):
    ids = {}
    for i, n in enumerate(order):
        if n.name:
            ids[id(n)] = n.name
        else:
            ids[id(n)] = "%s-%d" % (type(n).__name__, i)
    return ids


def _function_ref(node):
    """ Return the name under which a node's user function is referenced.

    User functions must be registered with :func:`register_function`;
    functions defined in streamz itself are referenced by qualified name.
    """
    func = None
    for attr in ("func", "predicate"):
        candidate = getattr(node, attr, None)
        if candidate is not None:
            func = candidate
            break
    if func is None or not callable(func):
        return None
    for name, registered in _function_registry.items():
        if registered is func:
            return name
    module = getattr(func, "__module__", "") or ""
    qualname = getattr(func, "__qualname__", "") or ""
    if module.split(".")[0] == "streamz" and "<locals>" not in qualname:
        return "streamz:%s.%s" % (module, qualname)
    raise CheckpointError(
        "Function %r used by node %r is not registered; register it with "
        "streamz.checkpoint.register_function so the checkpoint can refer "
        "to it by name" % (func, node))


def _describe(node, ids):
    return {
        "id": ids[id(node)],
        "type": type(node).__name__,
        "func": _function_ref(node),
        "upstreams": sorted(ids[id(u)] for u in node.upstreams
                            if u is not None),
        "state_version": getattr(type(node), "_checkpoint_state_version", 1),
    }


def _graph_description(node):
    sources = _find_sources(node)
    if not sources:
        raise CheckpointError("No sources found upstream of %r" % (node,))
    order = _walk_graph(sources)
    for n in order:
        _check_node_supported(n)
    ids = _node_ids(order)
    topology = [_describe(n, ids) for n in order]
    return sources, ids, topology


def save_checkpoint(node, path, checkpoint_id=None):
    """ Checkpoint the state of the graph containing ``node`` to ``path``.

    Injects a checkpoint barrier at every source of the graph, waits for it
    to flow through the whole (synchronous) graph, and writes the collected
    operator states plus the graph topology to ``path``.

    Parameters
    ----------
    node: Stream
        Any node of the graph; typically a terminal node.
    path: str
        Filesystem path of the checkpoint file to write.
    checkpoint_id: hashable, optional
        Identifier for this checkpoint.  A random id is generated if not
        given.

    Returns
    -------
    The checkpoint id.
    """
    sources, ids, topology = _graph_description(node)
    if checkpoint_id is None:
        checkpoint_id = uuid.uuid4().hex
    barrier = CheckpointBarrier(checkpoint_id)
    barrier.node_ids = ids
    for source in sources:
        source._emit(barrier)
    payload = {
        "format_version": _FORMAT_VERSION,
        "checkpoint_id": checkpoint_id,
        "topology": topology,
        "states": barrier.states,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return checkpoint_id


def _validate_topology(saved, current):
    saved_by_id = {desc["id"]: desc for desc in saved}
    current_by_id = {desc["id"]: desc for desc in current}
    missing = sorted(set(saved_by_id) - set(current_by_id))
    extra = sorted(set(current_by_id) - set(saved_by_id))
    if missing or extra:
        raise CheckpointCompatibilityError(
            "Graph topology does not match checkpoint: "
            "missing nodes %r, unexpected nodes %r" % (missing, extra))
    for nid in sorted(saved_by_id):
        old, new = saved_by_id[nid], current_by_id[nid]
        for field in ("type", "func", "upstreams", "state_version"):
            if old[field] != new[field]:
                raise CheckpointCompatibilityError(
                    "Node %r is incompatible with checkpoint: field %r is "
                    "%r in the checkpoint but %r in the current graph"
                    % (nid, field, old[field], new[field]))
    for desc in saved:
        ref = desc["func"]
        if ref is None or ref.startswith("streamz:"):
            continue
        if ref not in _function_registry:
            raise CheckpointCompatibilityError(
                "Function %r referenced by the checkpoint is not registered "
                "in this process" % (ref,))


def restore_checkpoint(node, path):
    """ Restore a checkpoint written by :func:`save_checkpoint`.

    The graph containing ``node`` must have been rebuilt with the same
    topology, the same registered functions and compatible operator state
    versions; otherwise :class:`CheckpointCompatibilityError` is raised and
    no state is loaded.

    Returns
    -------
    The checkpoint id stored in the checkpoint file.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if payload.get("format_version") != _FORMAT_VERSION:
        raise CheckpointCompatibilityError(
            "Unsupported checkpoint format version %r (expected %r)"
            % (payload.get("format_version"), _FORMAT_VERSION))
    _, ids, topology = _graph_description(node)
    _validate_topology(payload["topology"], topology)
    nodes_by_id = {ids[id(n)]: n
                   for n in _walk_graph(_find_sources(node))}
    for nid, state in payload["states"].items():
        nodes_by_id[nid].load_checkpoint_state(state)
    return payload["checkpoint_id"]
