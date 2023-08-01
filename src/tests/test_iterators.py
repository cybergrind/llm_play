from utils.iterators import flatten_subnested, split_with_overlap


A = ['a', 'b']
B = ['b', 'c', 'd']
C = ['d', 'e', 'f']


def flatten(lst):
    return list(flatten_subnested(lst))


def test_flatten():
    target = [A, B, C]
    assert flatten(target) == target
    assert flatten([[A], [B, C]]) == target
    assert flatten([[[A, B, C]]]) == target
    assert flatten([[[A, B, C]], [[[[]]]]]) == [*target, []]
    assert flatten([[A], [[[B]]], [[[C]]]]) == target


def test_split_with_overlap():
    def split_test(lst):
        return len(lst) <= 2

    assert split_with_overlap([1, 2, 3], 1, split_test) == [[1, 2], [2, 3]]
    assert split_with_overlap([1, 2, 3, 4], 1, split_test) == [[1, 2], [2, 3], [3, 4]]
    assert split_with_overlap([1, 2, 3, 4], 2, split_test) == [[1, 2], [2, 3], [2, 3], [3, 4]]

    def split_test_3(lst):
        return len(lst) <= 3

    assert split_with_overlap([1, 2, 3, 4], 2, split_test_3) == [[1, 2, 3], [2, 3, 4]]
    assert split_with_overlap([1, 2, 3, 4, 5], 2, split_test_3) == [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    assert split_with_overlap([1, 2, 3, 4, 5, 6], 2, split_test_3) == [
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
    ]
