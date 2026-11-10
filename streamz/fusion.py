import asyncio
from collections import deque
from copy import copy
from inspect import isawaitable, iscoroutinefunction

from .core import (
    Stream,
    filter as Filter,
    flatten as Flatten,
    map as Map,
    no_default,
)


BOUNDARY_ASYNC = "async_function"
BOUNDARY_FORK = "fork"
BOUNDARY_CUSTOM_UPDATE = "custom_update_subclass"
BOUNDARY_MULTIPLE_UPSTREAMS = "multiple_upstreams"
BOUNDARY_NOT_MARKED = "not_marked_fusible"
BOUNDARY_SINK = "registered_sink"
BOUNDARY_STATEFUL = "stateful_operator"

STATEFUL_OPERATORS = {
    "accumulate",
    "collect",
    "latest",
    "partition",
    "partition_unique",
    "sliding_window",
    "timed_window",
    "timed_window_unique",
    "unique",
    "zip_latest",
}

ASYNC_OPERATORS = {"map_async"}


def _node_id(node):
    return id(node)


def _display_name(node):
    return str(node)


def _sink_boundary(node):
    for cls in type(node).__mro__:
        if cls is Stream:
            break
        if cls.__name__ in ("Sink", "sink"):
            return BOUNDARY_SINK
    return None


def _classified_boundary(node):
    for cls in type(node).__mro__:
        if cls is Stream:
            return None
        if cls.__name__ in ASYNC_OPERATORS:
            return BOUNDARY_ASYNC
        if cls.__name__ in STATEFUL_OPERATORS:
            return BOUNDARY_STATEFUL
    return None


def _boundary_reason(node):
    sink_reason = _sink_boundary(node)
    if sink_reason is not None:
        return sink_reason

    if type(node) not in (Map, Filter, Flatten):
        if len(getattr(node, "upstreams", ())) > 1:
            return BOUNDARY_MULTIPLE_UPSTREAMS
        classified_reason = _classified_boundary(node)
        if classified_reason is not None:
            return classified_reason
        if type(node) is not Stream:
            return BOUNDARY_CUSTOM_UPDATE
        return BOUNDARY_NOT_MARKED

    if not getattr(node, "fusible", False):
        return BOUNDARY_NOT_MARKED

    if type(node) is Map:
        if iscoroutinefunction(node.func) or isawaitable(node.func):
            return BOUNDARY_ASYNC
        if getattr(node.func, "_streamz_stateful", False):
            return BOUNDARY_STATEFUL

    return None


def _reachable_nodes(root):
    seen_ids = set()
    nodes = set()
    pending = deque([root])
    while pending:
        node = pending.popleft()
        node_id = _node_id(node)
        if node_id in seen_ids:
            continue
        seen_ids.add(node_id)
        nodes.add(node)
        neighbors = list(node.downstreams) + list(node.upstreams)
        for neighbor in neighbors:
            if _node_id(neighbor) not in seen_ids:
                pending.append(neighbor)
    return nodes


def _clone_node(node):
    if _sink_boundary(node) is not None:
        clone = object.__new__(type(node))
        clone.__dict__.update(node.__dict__)
    else:
        clone = copy(node)
    clone.upstreams = []
    clone.downstreams = type(node.downstreams)()

    for attribute in ("buffers",):
        value = getattr(clone, attribute, None)
        if isinstance(value, dict):
            setattr(clone, attribute, copy(value))

    for attribute in (
        "_buffer",
        "metadata_buffer",
        "_metadata_buffer",
        "lossless_buffer",
        "next",
    ):
        if hasattr(clone, attribute):
            value = getattr(clone, attribute)
            if isinstance(value, list):
                setattr(clone, attribute, value.copy())
            elif isinstance(value, deque):
                setattr(clone, attribute, value.copy())

    if hasattr(clone, "seen") and isinstance(clone.seen, list):
        clone.seen = clone.seen.copy()

    if type(node).__name__ == "accumulate":
        clone.state = no_default
    elif type(node).__name__ in ("partition",):
        clone._buffer = type(clone._buffer)()
        clone._metadata_buffer = type(clone._metadata_buffer)()
        clone._callbacks = {}
        clone.state = 0
    elif type(node).__name__ == "partition_unique":
        clone._buffer = {}
        clone._metadata_buffer = {}
    elif type(node).__name__ == "sliding_window":
        clone._buffer = deque(maxlen=clone._buffer.maxlen)
        clone.metadata_buffer = deque(maxlen=clone.metadata_buffer.maxlen)
    elif type(node).__name__ in ("timed_window", "timed_window_unique"):
        clone._buffer = []
        clone.metadata_buffer = []
    elif type(node).__name__ == "unique":
        if isinstance(clone.seen, list):
            clone.seen = []
        else:
            clone.seen = {}
    elif type(node).__name__ == "collect":
        clone.cache = deque()
        clone.metadata_cache = deque()
    elif type(node).__name__ == "zip_latest":
        clone.last = [None for _ in node.upstreams]
        clone.metadata = [None for _ in node.upstreams]
        clone.missing = set(node.upstreams)
        clone.lossless_buffer = deque()
    elif type(node).__name__ == "latest":
        clone.next = []
        clone.next_metadata = None
    elif type(node).__name__ == "map_async":
        clone.work_queue = asyncio.Queue(maxsize=node.work_queue.maxsize)
        clone.work_task = None

    return clone


def _operation(node):
    if type(node) is Map:
        return {
            "type": "map",
            "node": node,
            "func": node.func,
            "args": node.args,
            "kwargs": node.kwargs,
        }
    if type(node) is Filter:
        return {
            "type": "filter",
            "node": node,
            "predicate": node.predicate,
            "args": node.args,
            "kwargs": node.kwargs,
        }
    return {"type": "flatten", "node": node}


class FusedStream(Stream):
    """ Synchronous map/filter/flatten operations executed without node hops. """

    _graphviz_shape = "box"

    def __init__(self, upstream, operations, stream_name=None):
        self.operations = operations
        self._emitting = 0
        super().__init__(upstream, stream_name=stream_name)

    def _emit(self, x, metadata=None):
        if metadata is None:
            metadata = []
        if metadata:
            self._retain_refs(metadata, len(self.downstreams))

        result = []
        self._emitting += 1
        try:
            for downstream in list(self.downstreams):
                emitted = downstream.update(x, who=self, metadata=metadata)
                if type(emitted) is list:
                    result.extend(emitted)
                else:
                    result.append(emitted)
                self._release_refs(metadata)
        except Exception:
            self._release_refs(metadata)
            raise
        finally:
            self._emitting -= 1
        return [item for item in result if item is not None]

    def _run(self, value, metadata, index=0):
        result = []
        for position in range(index, len(self.operations)):
            operation = self.operations[position]
            operation_type = operation["type"]

            if operation_type == "map":
                value = operation["func"](
                    value, *operation["args"], **operation["kwargs"]
                )
            elif operation_type == "filter":
                accepted = operation["predicate"](
                    value, *operation["args"], **operation["kwargs"]
                )
                if not accepted:
                    return result
            else:
                items = iter(value)
                try:
                    item = next(items)
                except StopIteration:
                    return result

                for next_item in items:
                    result.extend(self._run(item, [], position + 1))
                    item = next_item
                return result + self._run(item, metadata, position + 1)

        return result + [
            item for item in self._emit(value, metadata=metadata)
            if item is not None
        ]

    def update(self, x, who=None, metadata=None):
        if metadata is None:
            metadata = []
        try:
            result = self._run(x, metadata)
        except Exception:
            if self._emitting:
                self._release_refs(metadata, len(self.downstreams))
            else:
                self._release_refs(metadata)
            raise

        return result


class FusionPlan:
    """ Description of the graph produced by :func:`fuse`. """

    def __init__(self, groups, boundaries, replacements, outputs):
        self.groups = groups
        self.boundaries = boundaries
        self.replacements = replacements
        self.outputs = outputs

    @property
    def fused_nodes(self):
        return [group["fused_node"] for group in self.groups]

    def to_dict(self):
        return {
            "groups": [
                {
                    "operations": [
                        {
                            key: value
                            for key, value in operation.items()
                            if key != "node"
                        }
                        for operation in group["operations"]
                    ],
                    "source_nodes": [
                        _display_name(node) for node in group["source_nodes"]
                    ],
                    "output_nodes": [
                        _display_name(node) for node in group["output_nodes"]
                    ],
                    "stop_reason": group.get("stop_reason"),
                    "fused_node": _display_name(group["fused_node"]),
                }
                for group in self.groups
            ],
            "boundaries": [
                {
                    "node": _display_name(node),
                    "reason": reason,
                }
                for node, reason in self.boundaries.items()
            ],
            "outputs": [_display_name(node) for node in self.outputs],
        }


class FusedGraph:
    """ An optimized graph and the plan used to construct it. """

    def __init__(self, root, plan):
        self.root = root
        self.plan = plan

    @property
    def outputs(self):
        return self.plan.outputs

    @property
    def fused_nodes(self):
        return self.plan.fused_nodes

    def emit(self, x, **kwargs):
        return self.root.emit(x, **kwargs)

    def to_dict(self):
        return self.plan.to_dict()


def _start_group(node, reachable, boundaries):
    if node not in reachable:
        return False

    reason = _boundary_reason(node)
    if reason is not None:
        boundaries[node] = reason
        return False

    eligible_upstreams = [
        upstream
        for upstream in node.upstreams
        if upstream in reachable
    ]
    if not eligible_upstreams:
        return True

    for upstream in eligible_upstreams:
        if upstream in boundaries:
            return True
        if len(upstream.downstreams) > 1:
            return True
    return False


def _collect_group(start, reachable, group_nodes):
    group = []
    node = start
    stop_reason = None
    while node in reachable and _boundary_reason(node) is None:
        if _node_id(node) in group_nodes:
            break
        group.append(node)
        group_nodes.add(_node_id(node))

        eligible_downstreams = [
            downstream
            for downstream in node.downstreams
            if downstream in reachable
        ]
        if len(eligible_downstreams) != 1:
            stop_reason = BOUNDARY_FORK if len(eligible_downstreams) > 1 else None
            break
        node = eligible_downstreams[0]

    return group, stop_reason


def _group_end_reason(group, reachable):
    end = group[-1]
    downstreams = [
        downstream
        for downstream in end.downstreams
        if downstream in reachable
    ]
    if len(downstreams) > 1:
        return BOUNDARY_FORK
    if not downstreams:
        return None
    return _boundary_reason(downstreams[0])


def fuse(root):
    """ Build a new graph containing explicit synchronous fusion nodes.

    The source graph is not modified. Every fusible chain remains a single
    shared node in the optimized graph, so branching before a shared chain does
    not duplicate its side effects.
    """
    reachable = _reachable_nodes(root)
    if _boundary_reason(root) is None:
        current = root
        while current.upstreams and _boundary_reason(current) is None:
            current = current.upstreams[0]
        root = current
    boundaries = {}
    groups = []
    group_nodes = set()

    pending = deque([root])
    seen = set()
    while pending:
        node = pending.popleft()
        node_id = _node_id(node)
        if node_id in seen:
            continue
        seen.add(node_id)

        if _start_group(node, reachable, boundaries):
            group, _ = _collect_group(node, reachable, group_nodes)
            if group:
                groups.append(group)

        for downstream in node.downstreams:
            if downstream in reachable and _node_id(downstream) not in seen:
                pending.append(downstream)

    reachable_list = list(_reachable_nodes(root))
    reachable_ids = {_node_id(node) for node in reachable_list}
    grouped_ids = set(group_nodes)
    replacements = {}
    for node in reachable_list:
        if _node_id(node) not in grouped_ids:
            replacements[node] = _clone_node(node)

    built_groups = []
    for original_group in groups:
        operations = [_operation(node) for node in original_group]
        source_nodes = []
        upstream_nodes = []

        first = original_group[0]
        for upstream in first.upstreams:
            if upstream not in reachable:
                continue
            source_nodes.append(upstream)
            upstream_nodes.append(replacements[upstream])

        if not upstream_nodes:
            fused_upstream = Stream(
                stream_name="fused-input:{}".format(_display_name(first))
            )
            replacements[first] = fused_upstream
            cloned_root = fused_upstream
        else:
            fused_upstream = upstream_nodes[0]

        fused_node = FusedStream(
            fused_upstream,
            operations,
            stream_name="fused:{}".format(
                ",".join(type(node).__name__ for node in original_group)
            ),
        )

        for node in original_group:
            replacements[node] = fused_node

        for downstream in original_group[-1].downstreams:
            replacement = replacements.get(downstream)
            if replacement is not None and replacement not in fused_node.downstreams:
                fused_node.connect(replacement)

        built_groups.append({
            "source_nodes": source_nodes,
            "output_nodes": list(original_group[-1].downstreams),
            "stop_reason": _group_end_reason(original_group, reachable),
            "operations": operations,
            "fused_node": fused_node,
        })

    for original in reachable_list:
        if _node_id(original) in grouped_ids:
            continue
        clone = replacements[original]
        for upstream in list(original.upstreams):
            if _node_id(upstream) in reachable_ids:
                upstream_clone = replacements[upstream]
                if upstream_clone not in clone.upstreams:
                    clone.upstreams.append(upstream_clone)
        for downstream in list(original.downstreams):
            if _node_id(downstream) in reachable_ids:
                downstream_clone = replacements[downstream]
                if downstream_clone not in clone.downstreams:
                    clone.downstreams.add(downstream_clone)

    cloned_root = replacements[root]
    outputs = [
        replacements[node]
        for node in reachable_list
        if not node.downstreams
    ]
    plan = FusionPlan(built_groups, boundaries, replacements, outputs)
    return FusedGraph(cloned_root, plan)
