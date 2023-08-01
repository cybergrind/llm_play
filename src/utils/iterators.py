import math
from typing import Callable


def flatten_subnested(lst):
    """
    flatten list of sublists
    A, B, C = list[str]

    [[A], [B, [C]]] => [A, B, C]
    """
    if not isinstance(lst, list):
        raise NotImplementedError(f'Requires list, got {type(lst)} / {lst}')

    if not lst:
        yield []

    if not isinstance(lst[0], list):
        yield lst

    for item in lst:
        if not item:  # empty list
            yield item
        elif not isinstance(item[0], list):  # list of strings
            yield item
        elif isinstance(item, list):  # list of lists
            for subitem in flatten_subnested(item):
                yield subitem
        else:
            raise NotImplementedError


def split_with_overlap(lst: list, overlap: int, func: Callable[[list], bool]):
    """
    split list into chunks with overlap
    overlap is number of elements that will be in both chunks
    use func to determine if chunk is good or need to be split further down
    use dichotomy
    """
    if not lst:
        return []
    if len(lst) == 1:
        return [lst]

    if func(lst):
        return [lst]

    if len(lst) == 2:
        assert all(func([item]) for item in lst)
        return [[lst[0]], [lst[1]]]

    mid = len(lst) // 2

    over_left, over_right = math.floor(overlap / 2), math.ceil(overlap / 2)

    if mid - over_left <= 0:
        over_left = 0

    left = lst[: mid + over_right]
    right = lst[mid - over_left :]
    print(f'{len(left)=} {len(right)=} {mid=}')

    left = split_with_overlap(left, overlap, func)
    right = split_with_overlap(right, overlap, func)

    return list(flatten_subnested([left, right]))
