import pytest
from axon.utils import broadcast_shapes, BroadcastSemantics


@pytest.mark.parametrize("lhs, rhs, expected_result", [
    ((3, 4), (3, 4), (3, 4)),
    ((3, 1), (1, 4), (3, 4)),
    ((1, 1), (3, 4), (3, 4)),
    ((3, 4), (3, 1), (3, 4)),
    ((3, 4), (1, 1), (3, 4)),
    ((1,), (3, 4), (3, 4)),
    ((3, 4), (1,), (3, 4)),
])
def test_elementwise_broadcast_valid(lhs, rhs, expected_result):
    result = broadcast_shapes(lhs, rhs)
    assert result == expected_result


@pytest.mark.parametrize("lhs, rhs", [
    ((3, 4), (4, 4)),
    ((3, 4), (3, 5)),
    ((2, 3, 4), (2, 4, 5)),
])
def test_elementwise_broadcast_invalid(lhs, rhs):
    with pytest.raises(ValueError):
        broadcast_shapes(lhs, rhs)


@pytest.mark.parametrize("lhs, rhs, expected_result", [
    ((3, 4), (4, 5), (3, 5)),
    ((1, 3, 4), (2, 4, 5), (2, 3, 5)),
    ((2, 1, 3, 4), (2, 1, 4, 5), (2, 1, 3, 5)),
])
def test_matmul_broadcast_valid(lhs, rhs, expected_result):
    result = broadcast_shapes(lhs, rhs, semantics=BroadcastSemantics.MatMul)
    assert result == expected_result


@pytest.mark.parametrize("lhs, rhs", [
    ((3,), (4, 5)),
    ((3, 4), (5,)),
    ((2, 3, 4), (2, 5, 6)),
    ((2, 3, 4), (2, 3, 5)),
])
def test_matmul_broadcast_invalid(lhs, rhs):
    with pytest.raises(ValueError):
        broadcast_shapes(lhs, rhs, semantics=BroadcastSemantics.MatMul)
