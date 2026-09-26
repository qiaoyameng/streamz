import pickle

import pytest

from streamz import Stream
from streamz.checkpoint import (
    CheckpointBarrier,
    CheckpointCompatibilityError,
    CheckpointError,
    Checkpointer,
    register_function,
)


@register_function
def inc(x):
    return x + 1


@register_function
def double(x):
    return 2 * x


@register_function
def is_even(x):
    return x % 2 == 0


@register_function
def add(acc, x):
    return acc + x


@register_function
def pair_sum(t):
    return t[0] + t[1]


@register_function
def inc2(x):
    return x + 2


def build_graph():
    source1 = Stream()
    source2 = Stream()
    left = source1.map(inc).filter(is_even)
    right = source2.map(double)
    zipped = left.zip(right)
    total = zipped.map(pair_sum).accumulate(add, start=0)
    parts = total.partition(3)
    out = parts.sink_to_list()
    nodes = {
        'left': left, 'right': right, 'zipped': zipped,
        'total': total, 'parts': parts,
    }
    return source1, source2, out, nodes


def emit_range(source1, source2, start, stop):
    for i in range(start, stop):
        source1.emit(i)
        source2.emit(i)


def test_checkpoint_restore_matches_full_run(tmp_path):
    path = str(tmp_path / "checkpoint.pkl")

    # Full run without checkpointing
    source1, source2, full_out, _ = build_graph()
    emit_range(source1, source2, 0, 10)

    # Run half, checkpoint, "restart" with a rebuilt graph, continue
    source1, source2, out, nodes = build_graph()
    ck = Checkpointer(source1, source2, path=path)
    emit_range(source1, source2, 0, 5)
    ck.checkpoint(1)

    source1, source2, restarted_out, nodes = build_graph()
    ck2 = Checkpointer(source1, source2, path=path)
    assert ck2.restore() == 1

    # Operator state was restored: accumulate total, partial partition
    # buffer and unpaired zip leftovers
    assert nodes['total'].state == 8
    assert dict(nodes['parts']._buffer) == {None: [2, 8]}
    right_upstream = nodes['zipped'].upstreams[1]
    assert [x for x, _ in nodes['zipped'].buffers[right_upstream]] == [4, 6, 8]

    emit_range(source1, source2, 5, 10)

    assert restarted_out == full_out
    assert restarted_out == [(2, 8, 18)]


def test_checkpoint_file_stores_function_names_only(tmp_path):
    path = str(tmp_path / "checkpoint.pkl")
    source1, source2, out, nodes = build_graph()
    ck = Checkpointer(source1, source2, path=path)
    emit_range(source1, source2, 0, 3)
    ck.checkpoint(1)

    with open(path, "rb") as f:
        payload = pickle.load(f)

    assert payload["checkpoint_id"] == 1
    assert payload["format_version"] == 1
    functions = [entry["functions"] for entry in payload["topology"]]
    assert {'func': 'inc'} in functions
    assert {'predicate': 'is_even'} in functions
    for entry in payload["topology"]:
        for name in entry["functions"].values():
            assert isinstance(name, str)
    # state versions are recorded per operator
    assert all("state_version" in s for s in payload["states"].values())


def test_restore_rejects_incompatible_topology(tmp_path):
    path = str(tmp_path / "checkpoint.pkl")
    source1, source2, out, nodes = build_graph()
    ck = Checkpointer(source1, source2, path=path)
    emit_range(source1, source2, 0, 5)
    ck.checkpoint(1)

    # An extra node in the graph
    source1 = Stream()
    source2 = Stream()
    left = source1.map(inc).map(inc).filter(is_even)
    right = source2.map(double)
    left.zip(right).map(pair_sum).accumulate(add, start=0).partition(3).sink_to_list()
    with pytest.raises(CheckpointCompatibilityError):
        Checkpointer(source1, source2, path=path).restore()

    # A different function under a different registered name
    source1 = Stream()
    source2 = Stream()
    left = source1.map(inc2).filter(is_even)
    right = source2.map(double)
    left.zip(right).map(pair_sum).accumulate(add, start=0).partition(3).sink_to_list()
    with pytest.raises(CheckpointCompatibilityError):
        Checkpointer(source1, source2, path=path).restore()


def test_checkpoint_requires_registered_functions(tmp_path):
    path = str(tmp_path / "checkpoint.pkl")
    source = Stream()
    source.map(lambda x: x + 1).sink_to_list()
    source.emit(1)
    with pytest.raises(CheckpointError, match="not registered"):
        Checkpointer(source, path=path).checkpoint(1)


def test_zip_barrier_blocks_input_until_aligned():
    source1 = Stream()
    source2 = Stream()
    out = source1.zip(source2).sink_to_list()

    source1.emit(CheckpointBarrier(1))
    # data behind the barrier on the first input must not cross it
    source1.emit('a1')
    source2.emit('b0')
    assert out == []

    # once the barrier arrives on the second input, the operators align
    # and buffered data flows again
    source2.emit(CheckpointBarrier(1))
    assert out == [('a1', 'b0')]


def test_zip_misaligned_barrier_ids_raise():
    source1 = Stream()
    source2 = Stream()
    source1.zip(source2).sink_to_list()

    source1.emit(CheckpointBarrier(1))
    with pytest.raises(CheckpointError, match="Misaligned"):
        source2.emit(CheckpointBarrier(2))


def test_barrier_passes_through_map_filter_accumulate():
    source = Stream()
    acc = source.map(inc).filter(is_even).accumulate(add, start=0)
    out = acc.sink_to_list()

    source.emit(1)
    source.emit(CheckpointBarrier(1))
    source.emit(3)

    assert out == [2, 6]
    assert acc.state == 6


def test_accumulate_without_start_roundtrip(tmp_path):
    path = str(tmp_path / "checkpoint.pkl")

    source = Stream()
    acc = source.accumulate(add)
    acc.sink_to_list()
    Checkpointer(source, path=path).checkpoint(1)

    source2 = Stream()
    acc2 = source2.accumulate(add)
    out2 = acc2.sink_to_list()
    Checkpointer(source2, path=path).restore()

    from streamz.core import no_default
    assert acc2.state is no_default
    source2.emit(5)
    source2.emit(6)
    assert out2 == [5, 11]


def test_multiple_checkpoints(tmp_path):
    path = str(tmp_path / "checkpoint.pkl")
    source1, source2, out, nodes = build_graph()
    ck = Checkpointer(source1, source2, path=path)

    emit_range(source1, source2, 0, 4)
    ck.checkpoint(1)
    emit_range(source1, source2, 4, 8)
    ck.checkpoint(2)

    source1, source2, restarted_out, nodes = build_graph()
    ck2 = Checkpointer(source1, source2, path=path)
    assert ck2.restore() == 2
    emit_range(source1, source2, 8, 10)

    source1, source2, full_out, _ = build_graph()
    emit_range(source1, source2, 0, 10)
    # output produced before the restart plus output after the restart
    # must match the uninterrupted run
    assert out + restarted_out == full_out
