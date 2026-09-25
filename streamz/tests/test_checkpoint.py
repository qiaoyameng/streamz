import pytest

from streamz import Stream
from streamz.core import CheckpointBarrier
from streamz.checkpoint import (CheckpointCompatibilityError, CheckpointError,
                                register_function, registered_functions,
                                restore_checkpoint, save_checkpoint)


def emit(source, x):
    # Drive the graph synchronously.  Stream.emit would route through
    # Stream.loop (created by partition) and sync(), which relies on
    # cross-thread event-loop wakeup.
    source._emit(x)


@register_function("inc")
def inc(x):
    return x + 1


@register_function("is_even")
def is_even(x):
    return x % 2 == 0


@register_function("add")
def add(acc, x):
    return acc + x


@register_function("double")
def double(x):
    return 2 * x


class Collector:
    """ Module-level sink target so the sink function can be registered. """
    def __init__(self):
        self.items = []

    def collect(self, x):
        self.items.append(x)


def make_collector(name):
    collector = Collector()
    register_function(name, collector.collect)
    return collector


def build_chain():
    source = Stream(stream_name="source")
    s = source.map(inc, stream_name="inc").filter(is_even,
                                                  stream_name="even")
    acc = s.accumulate(add, start=0, stream_name="acc")
    part = s.partition(3, stream_name="part")
    acc_out = make_collector("acc_out")
    part_out = make_collector("part_out")
    acc.sink(acc_out.collect)
    part.sink(part_out.collect)
    return source, acc, acc_out, part_out


def run_full_chain(n):
    source, acc, acc_out, part_out = build_chain()
    for i in range(n):
        emit(source, i)
    return acc_out.items, part_out.items


def test_function_registry():
    funcs = registered_functions()
    assert funcs["inc"] is inc
    assert funcs["add"] is add


def test_checkpoint_restore_chain(tmp_path):
    expected_acc, expected_part = run_full_chain(10)

    source, acc, acc_out, part_out = build_chain()
    for i in range(5):
        emit(source, i)
    path = str(tmp_path / "state.chk")
    checkpoint_id = save_checkpoint(source, path, checkpoint_id="ck-1")
    assert checkpoint_id == "ck-1"
    assert acc.state == sum(x + 1 for x in range(5) if (x + 1) % 2 == 0)

    # simulate a restart: rebuild the graph from scratch
    source2, acc2, acc_out2, part_out2 = build_chain()
    assert acc2.state == 0
    assert restore_checkpoint(source2, path) == "ck-1"
    assert acc2.state == acc.state

    for i in range(5, 10):
        emit(source2, i)

    assert acc_out.items + acc_out2.items == expected_acc
    assert part_out.items + part_out2.items == expected_part


def test_checkpoint_restore_partition(tmp_path):
    expected_acc, expected_part = run_full_chain(8)

    source, acc, acc_out, part_out = build_chain()
    for i in range(5):
        emit(source, i)
    path = str(tmp_path / "state.chk")
    save_checkpoint(source, path)

    source2, acc2, acc_out2, part_out2 = build_chain()
    restore_checkpoint(source2, path)
    for i in range(5, 8):
        emit(source2, i)

    assert part_out.items + part_out2.items == expected_part


def build_zip_graph():
    left = Stream(stream_name="left")
    right = Stream(stream_name="right")
    z = left.zip(right, stream_name="zip")
    out = make_collector("zip_out")
    z.sink(out.collect)
    return left, right, z, out


def test_checkpoint_restore_zip(tmp_path):
    # full run for comparison
    left, right, z, out = build_zip_graph()
    for i in range(6):
        emit(left, i)
        emit(right, "r%d" % i)
    expected = out.items

    # split run
    left, right, z, out1 = build_zip_graph()
    for i in range(4):
        emit(left, i)
        emit(right, "r%d" % i)
    path = str(tmp_path / "state.chk")
    save_checkpoint(z, path)

    left2, right2, z2, out2 = build_zip_graph()
    restore_checkpoint(z2, path)
    for i in range(4, 6):
        emit(left2, i)
        emit(right2, "r%d" % i)

    assert out1.items + out2.items == expected


def test_zip_barrier_alignment_defers_post_barrier_data():
    left, right, z, out = build_zip_graph()
    emit(left, "a")
    emit(right, "x")
    assert out.items == [("a", "x")]

    barrier = CheckpointBarrier("ck-align")
    barrier.node_ids = {id(z): "zip"}
    left._emit(barrier)  # barrier arrives on the left input only
    # post-barrier data on the left input must not cross the barrier
    emit(left, "b")
    assert out.items == [("a", "x")]
    # pre-barrier data on the right input is still accepted
    emit(right, "y")
    assert out.items == [("a", "x")]
    # barrier aligns once the right input sees it; deferred data resumes
    right._emit(barrier)
    assert out.items == [("a", "x"), ("b", "y")]
    # the barrier snapshotted the pre-barrier leftover buffer ("y")
    assert barrier.states["zip"] == [[], [("y", [])]]


def test_save_checkpoint_unregistered_function(tmp_path):
    source = Stream()
    s = source.map(lambda x: x + 1)  # noqa: E731
    s.sink(make_collector("unregistered_out").collect)
    emit(source, 1)
    with pytest.raises(CheckpointError, match="not registered"):
        save_checkpoint(source, str(tmp_path / "state.chk"))


def test_save_checkpoint_unsupported_operator(tmp_path):
    source = Stream()
    s = source.buffer(2)  # no barrier support in this version
    s.sink(make_collector("buffer_out").collect)
    with pytest.raises(CheckpointError, match="does not support"):
        save_checkpoint(s, str(tmp_path / "state.chk"))


def test_restore_rejects_changed_function(tmp_path):
    source, acc, acc_out, part_out = build_chain()
    emit(source, 1)
    path = str(tmp_path / "state.chk")
    save_checkpoint(source, path)

    # rebuild the graph with a different function under another name
    source2 = Stream(stream_name="source")
    s2 = source2.map(double, stream_name="inc").filter(is_even,
                                                       stream_name="even")
    acc2 = s2.accumulate(add, start=0, stream_name="acc")
    part2 = s2.partition(3, stream_name="part")
    acc2.sink(make_collector("acc_out2").collect)
    part2.sink(make_collector("part_out2").collect)
    with pytest.raises(CheckpointCompatibilityError, match="func"):
        restore_checkpoint(source2, path)


def test_restore_rejects_changed_topology(tmp_path):
    source, acc, acc_out, part_out = build_chain()
    emit(source, 1)
    path = str(tmp_path / "state.chk")
    save_checkpoint(source, path)

    # rebuild with an extra map node
    source2 = Stream(stream_name="source")
    s2 = source2.map(inc, stream_name="inc").map(inc,
                                                 stream_name="inc-again")
    s2 = s2.filter(is_even, stream_name="even")
    acc2 = s2.accumulate(add, start=0, stream_name="acc")
    part2 = s2.partition(3, stream_name="part")
    acc2.sink(make_collector("acc_out3").collect)
    part2.sink(make_collector("part_out3").collect)
    with pytest.raises(CheckpointCompatibilityError, match="topology"):
        restore_checkpoint(source2, path)


def test_restore_rejects_incompatible_state_version(tmp_path, monkeypatch):
    source, acc, acc_out, part_out = build_chain()
    emit(source, 1)
    path = str(tmp_path / "state.chk")
    save_checkpoint(source, path)

    from streamz.core import accumulate
    monkeypatch.setattr(accumulate, "_checkpoint_state_version", 2)
    source2, acc2, acc_out2, part_out2 = build_chain()
    with pytest.raises(CheckpointCompatibilityError,
                       match="state_version"):
        restore_checkpoint(source2, path)


def test_restore_rejects_unregistered_function_name(tmp_path):
    source, acc, acc_out, part_out = build_chain()
    emit(source, 1)
    path = str(tmp_path / "state.chk")
    save_checkpoint(source, path)

    # simulate a process that forgot to register one of the functions
    import streamz.checkpoint as ck
    saved = ck._function_registry.pop("inc")
    try:
        source2, acc2, acc_out2, part_out2 = build_chain()
        with pytest.raises(CheckpointError):
            restore_checkpoint(source2, path)
    finally:
        ck._function_registry["inc"] = saved


def test_validate_topology_requires_registered_names():
    from streamz.checkpoint import _validate_topology
    desc = {"id": "n", "type": "map", "func": "missing-func",
            "upstreams": [], "state_version": 1}
    with pytest.raises(CheckpointCompatibilityError, match="not registered"):
        _validate_topology([desc], [dict(desc)])
