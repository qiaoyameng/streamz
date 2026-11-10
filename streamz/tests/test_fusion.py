import pytest

from streamz import RefCounter, Stream
from streamz.core import (
    filter as Filter,
    flatten as Flatten,
    map as Map,
    no_default,
)
from streamz.fusion import (
    BOUNDARY_ASYNC,
    BOUNDARY_CUSTOM_UPDATE,
    BOUNDARY_FORK,
    BOUNDARY_MULTIPLE_UPSTREAMS,
    BOUNDARY_NOT_MARKED,
    BOUNDARY_SINK,
    BOUNDARY_STATEFUL,
    fuse,
)


def count_internal_emits(root):
    counts = []

    def attach(node, seen):
        if node in seen:
            return
        seen.add(node)
        original_emit = node._emit

        def counting_emit(x, metadata=None):
            counts.append(type(node).__name__)
            return original_emit(x, metadata=metadata)

        node._emit = counting_emit
        for downstream in list(node.downstreams):
            attach(downstream, seen)

    attach(root, set())
    return counts


class CustomStream(Stream):
    def update(self, x, who=None, metadata=None):
        return self._emit(x)


class MetadataProbe(Stream):
    def __init__(self, upstream, **kwargs):
        self.calls = []
        super().__init__(upstream, **kwargs)

    def update(self, x, who=None, metadata=None):
        self.calls.append((x, metadata))
        return []


async def _async_identity(value):
    return value


def test_fuse_preserves_callbacks_outputs_and_reduces_emit_hops():
    source = Stream()
    map_calls = []
    filter_calls = []

    def record_map(value):
        map_calls.append(value)
        return value + 1

    def record_filter(value):
        filter_calls.append(value)
        return value % 2 == 0

    stream = (
        source
        .map(record_map, fusible=True)
        .filter(record_filter, fusible=True)
        .map(lambda value: [value], fusible=True)
        .flatten(fusible=True)
        .map(lambda value: value * 10, fusible=True)
    )
    fused = fuse(source)
    fused_sink = []
    fused.fused_nodes[0].sink(fused_sink.append)
    original_sink = []
    stream.sink(original_sink.append)
    original_emit_events = count_internal_emits(source)
    fused_emit_events = count_internal_emits(fused.root)

    original_map_calls = []
    original_filter_calls = []
    original_map_calls.extend(map_calls)
    original_filter_calls.extend(filter_calls)
    map_calls.clear()
    filter_calls.clear()

    values = [0, 1, 2, 3]
    for value in values:
        source.emit(value)
    original_emits = len(original_emit_events)
    original_internal_emits = list(original_emit_events)
    original_emit_events.clear()

    fused_map_calls = []
    fused_filter_calls = []
    original_map_calls.extend(map_calls)
    original_filter_calls.extend(filter_calls)
    map_calls.clear()
    filter_calls.clear()

    for value in values:
        fused.emit(value)
    fused_emits = len(fused_emit_events)
    fused_map_calls.extend(map_calls)
    fused_filter_calls.extend(filter_calls)

    assert original_sink == [20, 40]
    assert fused_sink == [20, 40]
    assert fused_map_calls == original_map_calls
    assert fused_filter_calls == original_filter_calls
    assert fused_emits < original_emits
    assert sum(event == "map" for event in original_internal_emits) > 0
    assert "FusedStream" in fused_emit_events

    operations = [
        operation["type"]
        for operation in fused.plan.groups[0]["operations"]
    ]
    assert operations == ["map", "filter", "map", "flatten", "map"]


def test_fuse_flatten_processes_each_item_immediately():
    source = Stream()
    order = []
    stream = (
        source
        .flatten(fusible=True)
        .map(lambda value: order.append(("map", value)) or value, fusible=True)
    )
    stream.sink_to_list()

    fused = fuse(source)
    output = []
    fused.fused_nodes[0].sink(output.append)

    fused.emit([1, 2, 3])

    assert order == [("map", 1), ("map", 2), ("map", 3)]
    assert output == [1, 2, 3]


def test_fuse_releases_metadata_for_filter_empty_flatten_and_failures():
    source = Stream()
    stream = (
        source
        .filter(lambda value: value % 2 == 0, fusible=True)
        .flatten(fusible=True)
    )
    fused = fuse(source)
    stream.sink_to_list()

    filter_ref = RefCounter()
    fused.emit(1, metadata=[{"ref": filter_ref}])
    assert filter_ref.count == 0

    empty_source = Stream()
    empty_fused = fuse(empty_source)
    empty_ref = RefCounter()
    empty_fused.emit([], metadata=[{"ref": empty_ref}])
    assert empty_ref.count == 0

    failure_ref = RefCounter()
    with pytest.raises(TypeError):
        fused.emit(object(), metadata=[{"ref": failure_ref}])
    assert failure_ref.count == 0

    downstream_ref = RefCounter()

    def fail_sink(value):
        raise RuntimeError(value)

    downstream_source = Stream()
    downstream_stream = downstream_source.map(
        lambda value: value, fusible=True
    )
    downstream_fused = fuse(downstream_source)
    downstream_fused.fused_nodes[0].sink(fail_sink)
    with pytest.raises(RuntimeError):
        downstream_fused.emit(2, metadata=[{"ref": downstream_ref}])
    assert downstream_ref.count == 0


def test_fuse_flatten_matches_original_metadata_placement():
    source = Stream()
    stream = source.flatten(fusible=True)
    fused = fuse(source)

    original_probe = MetadataProbe(stream)
    fused_probe = MetadataProbe(fused.fused_nodes[0])
    metadata = [{"v": 1}]

    source.emit([1, 2], metadata=metadata)
    fused.emit([1, 2], metadata=metadata)

    assert original_probe.calls == fused_probe.calls
    assert fused_probe.calls == [(1, []), (2, metadata)]


def test_fuse_preserves_shared_subgraph_without_duplicate_side_effects():
    source = Stream()
    shared_calls = []
    shared = source.map(
        lambda value: shared_calls.append(value) or value + 1,
        fusible=True,
    )
    left = shared.map(lambda value: ("left", value), fusible=True)
    right = shared.map(lambda value: ("right", value), fusible=True)

    fused = fuse(source)
    output = []
    for node in fused.fused_nodes:
        if not node.downstreams:
            node.sink(output.append)

    fused.emit(10)

    assert shared_calls == [10]
    assert output == [("left", 11), ("right", 11)]
    assert len(fused.plan.groups) == 3
    assert fused.plan.groups[0]["stop_reason"] == BOUNDARY_FORK


def test_fuse_reports_boundaries_and_keeps_original_graph():
    source = Stream()
    unmarked = source.map(lambda value: value)
    stateful = unmarked.accumulate(lambda acc, value: acc + value, start=0)
    custom = CustomStream(stateful)
    async_node = custom.map_async(_async_identity)
    output = []
    async_node.sink(output.append)

    fused = fuse(source)
    boundaries = dict(fused.plan.boundaries)

    assert source in boundaries
    assert boundaries[source] == BOUNDARY_NOT_MARKED
    assert boundaries[unmarked] == BOUNDARY_NOT_MARKED
    assert boundaries[stateful] == BOUNDARY_STATEFUL
    assert boundaries[custom] == BOUNDARY_CUSTOM_UPDATE
    assert boundaries[async_node] == BOUNDARY_ASYNC
    assert any(reason == BOUNDARY_SINK for reason in boundaries.values())
    assert fused.plan.groups == []

    assert original_graph_keeps_state(source, stateful)


def test_fuse_from_middle_includes_upstream_and_preserves_state_boundary():
    source = Stream()
    stateful = source.accumulate(lambda acc, value: acc + value, start=0)
    terminal = stateful.map(lambda value: value, fusible=True)

    source.emit(10)
    fused = fuse(terminal)
    cloned_stateful = fused.plan.replacements[stateful]

    assert cloned_stateful is not stateful
    assert cloned_stateful.state == no_default
    assert fused.plan.boundaries[stateful] == BOUNDARY_STATEFUL

    source.emit(20)
    assert stateful.state == 30


def original_graph_keeps_state(source, stateful):
    source.emit(1)
    source.emit(2)
    return stateful.state == 3


def test_fuse_detects_async_map_boundary():
    async def async_map(value):
        return value

    source = Stream()
    node = source.map(async_map, fusible=True)
    fused = fuse(source)

    assert fused.plan.boundaries[node] == BOUNDARY_ASYNC
    assert fused.plan.groups == []


def test_fuse_detects_explicit_stateful_map_boundary():
    state = []

    def stateful_map(value):
        state.append(value)
        return len(state)

    stateful_map._streamz_stateful = True

    source = Stream()
    node = source.map(stateful_map, fusible=True)
    fused = fuse(source)

    assert fused.plan.boundaries[node] == BOUNDARY_STATEFUL
    assert fused.plan.groups == []


def test_fuse_treats_multi_upstream_node_as_boundary():
    left = Stream()
    right = Stream()
    joined = left.zip(right)
    terminal = joined.map(lambda value: value, fusible=True)

    fused = fuse(left)

    assert fused.plan.boundaries[joined] == BOUNDARY_MULTIPLE_UPSTREAMS
    assert len(fused.plan.groups) == 1
    assert fused.plan.groups[0]["operations"][0]["type"] == "map"
    assert fused.plan.groups[0]["source_nodes"] == [joined]


def test_fusion_markers_are_opt_in():
    source = Stream()
    marked = source.map(lambda value: value, fusible=True)
    unmarked = marked.filter(lambda value: True)

    fused = fuse(source)

    assert isinstance(marked, Map)
    assert isinstance(unmarked, Filter)
    assert len(fused.plan.groups) == 1
    assert fused.plan.groups[0]["operations"][0]["type"] == "map"
    assert fused.plan.boundaries[unmarked] == BOUNDARY_NOT_MARKED


def test_flatten_accepts_fusible_marker():
    source = Stream()
    flatten = source.flatten(fusible=True)

    assert isinstance(flatten, Flatten)
    assert flatten.fusible is True
