#!/usr/bin/env python
from __future__ import annotations

import heapq
from pathlib import Path

MODULUS = 1_003_443_221
INPUT_PATH = Path("1001_input.txt")


def parse_chords(values: list[int]) -> tuple[list[int], list[int], list[int]]:
    """Return left ends, right ends, and the chord at every array position."""
    chord_count = len(values) // 2
    if len(values) != 2 * chord_count:
        raise ValueError("the array length must be even")

    left: list[int] = []
    right: list[int] = []
    chord_at = [-1] * len(values)
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
        chord_at[position] = chord

    if open_chord or len(left) != chord_count:
        raise ValueError("every value must occur exactly twice")
    return left, right, chord_at


class CrossingComponents:
    """Find components of a chord-intersection graph without listing its edges."""

    def __init__(self, left: list[int], right: list[int]) -> None:
        chord_count = len(left)
        self.left = left
        self.right = right

        self.parent = list(range(chord_count))
        self.size = [1] * chord_count

        # One leftist-heap node per chord. A component heap contains the right
        # endpoints of all of its intervals that have not expired yet.
        self.heap_left = [-1] * chord_count
        self.heap_right = [-1] * chord_count
        self.heap_rank = [1] * chord_count
        self.component_heap = list(range(chord_count))

    def find(self, chord: int) -> int:
        parent = self.parent
        root = chord
        while parent[root] != root:
            root = parent[root]
        while parent[chord] != chord:
            next_chord = parent[chord]
            parent[chord] = root
            chord = next_chord
        return root

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

    def union(self, first: int, second: int) -> int:
        first = self.find(first)
        second = self.find(second)
        if first == second:
            return first
        if self.size[first] < self.size[second]:
            first, second = second, first

        self.parent[second] = first
        self.size[first] += self.size[second]
        self.component_heap[first] = self.meld(
            self.component_heap[first], self.component_heap[second]
        )
        self.component_heap[second] = -1
        return first

    def roots(self) -> list[int]:
        """Run the sweep and return the final root of every chord."""
        candidates: list[tuple[int, int, int]] = []

        for chord, (left_end, right_end) in enumerate(zip(self.left, self.right)):
            current = chord

            while candidates:
                candidate_right, node, stored_root = candidates[0]
                root = self.find(stored_root)

                if root != stored_root:
                    heapq.heappop(candidates)
                    continue
                heap_root = self.component_heap[root]
                if heap_root != node or self.right[node] != candidate_right:
                    heapq.heappop(candidates)
                    continue
                if root == self.find(current):
                    # The growing component is intentionally absent from the
                    # candidate set until all other neighbours are consumed.
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

                # This component has an active interval ending inside the new
                # chord, hence at least one crossing edge joins the components.
                heapq.heappop(candidates)
                current = self.union(current, root)

            current = self.find(current)
            heap_root = self.component_heap[current]
            heapq.heappush(
                candidates, (self.right[heap_root], heap_root, current)
            )

        return [self.find(chord) for chord in range(len(self.left))]


def permutation_cut(word: list[int]) -> int | None:
    """Find a rotation whose first half contains every chord exactly once."""
    chord_count = len(word) // 2
    doubled = word + word
    frequencies: dict[int, int] = {}
    repeated = 0

    for chord in doubled[:chord_count]:
        old = frequencies.get(chord, 0)
        frequencies[chord] = old + 1
        if old == 1:
            repeated += 1

    for start in range(chord_count):
        if repeated == 0:
            return start

        outgoing = doubled[start]
        old = frequencies[outgoing]
        if old == 2:
            repeated -= 1
        if old == 1:
            del frequencies[outgoing]
        else:
            frequencies[outgoing] = old - 1

        incoming = doubled[start + chord_count]
        old = frequencies.get(incoming, 0)
        frequencies[incoming] = old + 1
        if old == 1:
            repeated += 1

    return None


def count_permutation_diagram(word: list[int], start: int, modulus: int) -> int:
    """Count non-crossing subsets when a line separates the two endpoint sets."""
    chord_count = len(word) // 2
    doubled = word + word
    first_half = doubled[start : start + chord_count]
    second_half = doubled[start + chord_count : start + 2 * chord_count]
    second_position = {chord: position for position, chord in enumerate(second_half)}

    # Chords selected in first-half order are non-crossing exactly when their
    # positions in the second half form a decreasing subsequence.
    fenwick = [0] * (chord_count + 1)
    subsequence_count = 0

    for chord in first_half:
        position = second_position[chord]

        prefix = 0
        index = position + 1
        while index:
            prefix += fenwick[index]
            index -= index & -index

        ending_here = (1 + subsequence_count - prefix) % modulus
        subsequence_count = (subsequence_count + ending_here) % modulus

        index = position + 1
        while index <= chord_count:
            fenwick[index] = (fenwick[index] + ending_here) % modulus
            index += index & -index

    return (subsequence_count + 1) % modulus


def count_interval_dp(word: list[int], modulus: int) -> int:
    """General O(k^2)-time, O(k)-space count for a k-chord component."""
    endpoint_count = len(word)
    mate = [-1] * endpoint_count
    first_position: dict[int, int] = {}
    for position, chord in enumerate(word):
        first = first_position.pop(chord, None)
        if first is None:
            first_position[chord] = position
        else:
            mate[first] = position
            mate[position] = first

    # inside[i] is the count strictly between the two ends of a chord whose
    # left endpoint is i. For one right boundary at a time, current[i] is the
    # count in [i, boundary]. Recomputing that row keeps storage linear.
    inside = [1] * endpoint_count
    current = [1] * (endpoint_count + 1)

    for boundary in range(endpoint_count):
        current[boundary + 1] = 1
        for position in range(boundary, -1, -1):
            other = mate[position]
            value = current[position + 1]
            if position < other <= boundary:
                value += inside[position] * current[other + 1]
            current[position] = value % modulus

        next_position = boundary + 1
        if next_position < endpoint_count and mate[next_position] < next_position:
            left_end = mate[next_position]
            inside[left_end] = current[left_end + 1]

    return current[0]


def count_component(word: list[int], modulus: int = MODULUS) -> int:
    """Count connectable subsets for one crossing-connected component."""
    start = permutation_cut(word)
    if start is not None:
        return count_permutation_diagram(word, start, modulus)
    return count_interval_dp(word, modulus)


def connectivity_number(values: list[int], modulus: int = MODULUS) -> int:
    """Return the number of connectable subarrays obtained by deleting pairs."""
    left, right, chord_at = parse_chords(values)
    roots = CrossingComponents(left, right).roots()

    words: dict[int, list[int]] = {}
    for chord in chord_at:
        root = roots[chord]
        words.setdefault(root, []).append(chord)

    answer = 1
    for word in words.values():
        answer = answer * count_component(word, modulus) % modulus
    return answer


def read_input(path: Path = INPUT_PATH) -> list[int]:
    return [int(value) for value in path.read_text(encoding="utf-8").strip().split(",")]


def main() -> None:
    assert connectivity_number([0, 1, 0, 1]) == 3
    assert (
        connectivity_number(
            [0, 1, 2, 3, 1, 4, 0, 5, 4, 2, 6, 7, 3, 8, 6, 5, 9, 8, 9, 7]
        )
        == 86
    )
    print(connectivity_number(read_input()))


if __name__ == "__main__":
    main()
