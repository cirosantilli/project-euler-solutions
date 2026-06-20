#!/usr/bin/env python
from __future__ import annotations

import heapq
from pathlib import Path

INPUT_PATH = Path("1002_input.txt")


def parse_chords(values: list[int]) -> tuple[list[int], list[int]]:
    """Return chord endpoints ordered by increasing left endpoint."""
    if len(values) % 2:
        raise ValueError("the array length must be even")

    left: list[int] = []
    right: list[int] = []
    open_chord: dict[int, int] = {}

    for position, value in enumerate(values):
        chord = open_chord.pop(value, None)
        if chord is None:
            chord = len(left)
            open_chord[value] = chord
            left.append(position)
            right.append(-1)
        else:
            right[chord] = position

    if open_chord or len(left) * 2 != len(values):
        raise ValueError("every value must occur exactly twice")
    return left, right


class BipartiteCrossingSweep:
    """Build a parity spanning forest of an implicit bipartite circle graph."""

    def __init__(self, left: list[int], right: list[int]) -> None:
        chord_count = len(left)
        self.left = left
        self.right = right

        self.parent = list(range(chord_count))
        self.size = [1] * chord_count
        self.xor_to_parent = [0] * chord_count
        self.color_zero_count = [1] * chord_count
        self.color_one_count = [0] * chord_count

        self.heap_left = [-1] * chord_count
        self.heap_right = [-1] * chord_count
        self.heap_rank = [1] * chord_count
        self.component_heap = list(range(chord_count))

    def root(self, chord: int) -> int:
        parent = self.parent
        while parent[chord] != chord:
            chord = parent[chord]
        return chord

    def parity_to_root(self, chord: int) -> tuple[int, int]:
        parent = self.parent
        parity = 0
        while parent[chord] != chord:
            parity ^= self.xor_to_parent[chord]
            chord = parent[chord]
        return chord, parity

    def meld(self, first: int, second: int) -> int:
        if first < 0:
            return second
        if second < 0:
            return first
        if self.right[first] > self.right[second]:
            first, second = second, first

        self.heap_right[first] = self.meld(self.heap_right[first], second)
        left_child = self.heap_left[first]
        right_child = self.heap_right[first]
        left_rank = 0 if left_child < 0 else self.heap_rank[left_child]
        right_rank = 0 if right_child < 0 else self.heap_rank[right_child]
        if left_rank < right_rank:
            self.heap_left[first], self.heap_right[first] = right_child, left_child
            left_rank, right_rank = right_rank, left_rank
        self.heap_rank[first] = right_rank + 1
        return first

    def pop_heap(self, node: int) -> int:
        return self.meld(self.heap_left[node], self.heap_right[node])

    def union_opposite(self, first: int, second: int) -> int:
        """Merge components while requiring the two chords to have opposite colors."""
        first_root, first_parity = self.parity_to_root(first)
        second_root, second_parity = self.parity_to_root(second)
        if first_root == second_root:
            if first_parity ^ second_parity != 1:
                raise ValueError("the chord diagram is not bipartite-connectable")
            return first_root

        if self.size[first_root] < self.size[second_root]:
            first_root, second_root = second_root, first_root
            first_parity, second_parity = second_parity, first_parity

        shift = first_parity ^ second_parity ^ 1
        self.parent[second_root] = first_root
        self.xor_to_parent[second_root] = shift
        self.size[first_root] += self.size[second_root]

        if shift == 0:
            self.color_zero_count[first_root] += self.color_zero_count[second_root]
            self.color_one_count[first_root] += self.color_one_count[second_root]
        else:
            self.color_zero_count[first_root] += self.color_one_count[second_root]
            self.color_one_count[first_root] += self.color_zero_count[second_root]

        self.component_heap[first_root] = self.meld(
            self.component_heap[first_root], self.component_heap[second_root]
        )
        self.component_heap[second_root] = -1
        return first_root

    def solve(self) -> int:
        candidates: list[tuple[int, int, int]] = []

        for chord, (left_end, right_end) in enumerate(zip(self.left, self.right)):
            current = chord

            while candidates:
                candidate_right, node, stored_root = candidates[0]
                root = self.root(stored_root)

                if root != stored_root:
                    heapq.heappop(candidates)
                    continue
                heap_root = self.component_heap[root]
                if heap_root != node or self.right[node] != candidate_right:
                    heapq.heappop(candidates)
                    continue
                if root == self.root(current):
                    heapq.heappop(candidates)
                    continue

                if candidate_right <= left_end:
                    heapq.heappop(candidates)
                    while (
                        self.component_heap[root] >= 0
                        and self.right[self.component_heap[root]] <= left_end
                    ):
                        self.component_heap[root] = self.pop_heap(
                            self.component_heap[root]
                        )
                    heap_root = self.component_heap[root]
                    if heap_root >= 0:
                        heapq.heappush(
                            candidates, (self.right[heap_root], heap_root, root)
                        )
                    continue

                if candidate_right >= right_end:
                    break

                heapq.heappop(candidates)
                # Every consumed component is adjacent to the newly inserted
                # chord itself. The current root is only an aggregate and need
                # not have the same color as that chord.
                current = self.union_opposite(chord, node)

            current = self.root(current)
            heap_root = self.component_heap[current]
            heapq.heappush(
                candidates, (self.right[heap_root], heap_root, current)
            )

        answer = 0
        for chord in range(len(self.left)):
            if self.parent[chord] == chord:
                answer += max(
                    self.color_zero_count[chord], self.color_one_count[chord]
                )
        return answer


def maximal_above_connections(values: list[int]) -> int:
    left, right = parse_chords(values)
    return BipartiteCrossingSweep(left, right).solve()


def read_input(path: Path = INPUT_PATH) -> list[int]:
    return [int(value) for value in path.read_text(encoding="utf-8").strip().split(",")]


def main() -> None:
    assert maximal_above_connections([0, 1, 2, 1, 0, 2]) == 2
    print(maximal_above_connections(read_input()))


if __name__ == "__main__":
    main()
