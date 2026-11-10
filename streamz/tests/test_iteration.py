import asyncio

import pytest

from streamz import Stream
from streamz.iteration import IterationController, iterate
from streamz.utils_test import inc


def eq(previous, current):
    return previous == current


@pytest.mark.asyncio
async def test_converges_after_three_rounds():
    # min(x + 1, 3) saturates: 1 -> 2 -> 3 -> 3, so the third round
    # produces the same output as the second and the run converges.
    source = Stream(asynchronous=True)
    out = source.map(lambda x: min(x + 1, 3))

    result = await iterate(source, [out], [1], eq)

    assert result.converged
    assert result.reason == 'converged'
    assert result.rounds == 3
    assert result.outputs == [3]
    assert [(r.inputs, r.outputs) for r in result.history] == [
        ([1], [2]),
        ([2], [3]),
        ([3], [3]),
    ]


@pytest.mark.asyncio
async def test_never_converges_hits_max_rounds():
    source = Stream(asynchronous=True)
    out = source.map(inc)

    result = await iterate(source, [out], [0], lambda p, c: False,
                           max_rounds=3)

    assert not result.converged
    assert result.reason == 'max_rounds'
    assert result.rounds == 3
    assert result.outputs == [3]
    assert [(r.inputs, r.outputs) for r in result.history] == [
        ([0], [1]),
        ([1], [2]),
        ([2], [3]),
    ]


@pytest.mark.asyncio
async def test_empty_output_terminates_normally():
    source = Stream(asynchronous=True)
    out = source.filter(lambda x: x > 10)

    result = await iterate(source, [out], [1, 2, 3], eq)

    assert result.converged
    assert result.reason == 'empty'
    assert result.rounds == 1
    assert result.outputs == []
    assert [(r.inputs, r.outputs) for r in result.history] == [
        ([1, 2, 3], []),
    ]


@pytest.mark.asyncio
async def test_batch_order_follows_entry_sequence_and_branch_order():
    # Two branches merged with union: flatten emits x, x + 1 in order,
    # the second branch emits 10 * x.  The next batch is the concatenation
    # of the exit outputs in entry (emission) order.
    source = Stream(asynchronous=True)
    flat = source.map(lambda x: [x, x + 1]).flatten()
    times_ten = source.map(lambda x: 10 * x)
    both = flat.union(times_ten)

    result = await iterate(source, [both], [1], lambda p, c: False,
                           max_rounds=2)

    assert not result.converged
    assert [(r.inputs, r.outputs) for r in result.history] == [
        ([1], [1, 2, 10]),
        ([1, 2, 10], [1, 2, 10, 2, 3, 20, 10, 11, 100]),
    ]


@pytest.mark.asyncio
async def test_multiple_exits_concatenate_in_declared_order():
    source = Stream(asynchronous=True)
    plus_one = source.map(lambda x: x + 1)
    times_ten = source.map(lambda x: 10 * x)

    # exits declared as [times_ten, plus_one]: the times_ten outputs of
    # the whole round come before the plus_one outputs
    result = await iterate(source, [times_ten, plus_one], [1],
                           lambda p, c: False, max_rounds=2)

    assert [(r.inputs, r.outputs) for r in result.history] == [
        ([1], [10, 2]),
        ([10, 2], [100, 20, 11, 3]),
    ]


@pytest.mark.asyncio
async def test_manual_future_delays_convergence_check():
    # A sink branch is delayed with a manual future; the convergence
    # predicate must not be called before that branch completes.
    source = Stream(asynchronous=True)
    out = source.map(inc)
    gate = asyncio.get_running_loop().create_future()
    done = []

    async def waiter(x):
        await gate
        done.append(x)

    source.sink(waiter)

    calls = []

    def converged(previous, current):
        calls.append((previous, current))
        return True

    controller = IterationController(source, [out], converged)
    task = asyncio.ensure_future(controller.run([1]))
    await asyncio.sleep(0.05)

    # the round is stuck on the gated sink branch
    assert controller.running
    assert calls == []
    assert done == []

    gate.set_result(None)
    result = await task

    assert result.converged
    assert result.reason == 'converged'
    assert result.rounds == 1
    assert result.outputs == [2]
    assert calls == [(None, [2])]
    assert done == [1]


@pytest.mark.asyncio
async def test_external_input_does_not_mix_into_subgraph():
    source = Stream(asynchronous=True)
    out = source.map(inc)
    gate = asyncio.get_running_loop().create_future()

    async def waiter(x):
        await gate

    source.sink(waiter)

    controller = IterationController(source, [out], lambda p, c: True)
    task = asyncio.ensure_future(controller.run([1]))
    await asyncio.sleep(0.05)

    # external emission while the round is blocked on the gated branch
    external = asyncio.ensure_future(source.emit(999, asynchronous=True))
    await asyncio.sleep(0.05)
    assert controller._round_outputs == [[2]]

    gate.set_result(None)
    await external
    result = await task

    assert result.outputs == [2]
    assert [(r.inputs, r.outputs) for r in result.history] == [([1], [2])]


@pytest.mark.asyncio
async def test_only_one_batch_at_a_time():
    source = Stream(asynchronous=True)
    out = source.map(inc)
    gate = asyncio.get_running_loop().create_future()

    async def waiter(x):
        await gate

    source.sink(waiter)

    controller = IterationController(source, [out], lambda p, c: True)
    task = asyncio.ensure_future(controller.run([1]))
    await asyncio.sleep(0.05)

    with pytest.raises(RuntimeError):
        await controller.run([2])

    gate.set_result(None)
    await task


@pytest.mark.asyncio
async def test_exception_midway_releases_metadata():
    source = Stream(asynchronous=True)

    def boom(x):
        if x == 2:
            raise ValueError('boom')
        return x

    out = source.map(boom)
    controller = IterationController(source, [out], eq)

    with pytest.raises(ValueError, match='boom'):
        await controller.run([1, 2, 3])

    assert not controller.running
    assert controller._refs
    assert all(ref.count <= 0 for ref in controller._refs)


@pytest.mark.asyncio
async def test_cancellation_releases_metadata():
    source = Stream(asynchronous=True)
    out = source.map(inc)
    gate = asyncio.get_running_loop().create_future()

    async def waiter(x):
        await gate

    source.sink(waiter)

    controller = IterationController(source, [out], eq)
    task = asyncio.ensure_future(controller.run([1]))
    await asyncio.sleep(0.05)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not controller.running
    assert all(ref.count <= 0 for ref in controller._refs)

    # let the interrupted sink branch finish so no task is left behind
    gate.set_result(None)
    await asyncio.sleep(0.01)


def test_rejects_window_and_queue_operators():
    async def af(x):
        return x

    builders = [
        lambda s: s.partition(2),
        lambda s: s.buffer(2),
        lambda s: s.timed_window(0.01),
        lambda s: s.sliding_window(2),
        lambda s: s.map_async(af),
        lambda s: s.collect(),
        lambda s: s.delay(0.01),
    ]
    for build in builders:
        source = Stream()
        node = build(source)
        with pytest.raises(ValueError, match='delay emission'):
            IterationController(source, [node], eq)


def test_rejects_unreachable_exit():
    source = Stream()
    other = Stream().map(inc)
    with pytest.raises(ValueError, match='reachable'):
        IterationController(source, [other], eq)


def test_validates_arguments():
    source = Stream()
    out = source.map(inc)
    with pytest.raises(ValueError, match='exit'):
        IterationController(source, [], eq)
    with pytest.raises(ValueError, match='max_rounds'):
        IterationController(source, [out], eq, max_rounds=0)
    with pytest.raises(TypeError, match='callable'):
        IterationController(source, [out], 'not-callable')
