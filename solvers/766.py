#!/usr/bin/env python3
"""
Project Euler 766 — Sliding Block Puzzle

Counts the number of reachable configurations from the given initial layout.
Pieces slide any positive number of grid units orthogonally, without rotation.

Key detail: pieces of the same shape are indistinguishable, so a configuration
is identified by the set of occupied positions for each shape type (order-free).

No external libraries are used.
"""

from collections import deque


def _encode_segment(pos_list):
    """Encode a sorted list of anchor positions (each 0..31) into 5-bit packed int."""
    seg = 0
    shift = 0
    for p in pos_list:
        seg |= p << shift
        shift += 5
    return seg


def _decode_segment(seg_bits, k):
    """Decode k positions from a 5-bit packed int (returns a list length k)."""
    out = [0] * k
    for i in range(k):
        out[i] = (seg_bits >> (5 * i)) & 31
    return out


class Puzzle:
    """
    State encoding:
      - Every piece instance stores its anchor position (x + y*W) in 5 bits (0..31).
      - Within each piece-type, anchors are kept sorted to canonicalize indistinguishable pieces.
      - The whole state is a single Python int made by concatenating all type-segments.

    Precomputation:
      - For each type and each possible anchor position, store the occupied-cell bitmask
        and max steps to boundary in each direction (up, down, left, right).
    """

    def __init__(self, width, height, piece_types, initial_positions_by_type):
        """
        piece_types: list of dicts: { "offsets": [(dx,dy),...], "count": k }
        initial_positions_by_type: list of sorted (or unsorted) lists, one per type,
                                   containing anchor positions.
        """
        self.W = width
        self.H = height

        # Directions in order: up, down, left, right
        self._deltas = (-self.W, self.W, -1, 1)

        # Build per-type data including packed bit ranges.
        self.types = []
        start_piece_index = 0
        for t in piece_types:
            offs = t["offsets"]
            k = t["count"]

            mask_at = [0] * (self.W * self.H)
            limits_at = [None] * (self.W * self.H)

            for pos in range(self.W * self.H):
                x = pos % self.W
                y = pos // self.W

                # Build occupancy mask, checking bounds.
                m = 0
                ok = True
                for dx, dy in offs:
                    xx = x + dx
                    yy = y + dy
                    if xx < 0 or xx >= self.W or yy < 0 or yy >= self.H:
                        ok = False
                        break
                    m |= 1 << (yy * self.W + xx)
                if not ok:
                    continue

                mask_at[pos] = m

                # Max steps to boundary for each direction.
                max_up = min(y + dy for dx, dy in offs)
                max_down = min((self.H - 1) - (y + dy) for dx, dy in offs)
                max_left = min(x + dx for dx, dy in offs)
                max_right = min((self.W - 1) - (x + dx) for dx, dy in offs)
                limits_at[pos] = (max_up, max_down, max_left, max_right)

            shift0 = start_piece_index * 5
            seg_bits = 5 * k
            seg_mask = ((1 << seg_bits) - 1) << shift0

            self.types.append(
                {
                    "k": k,
                    "mask_at": mask_at,
                    "limits_at": limits_at,
                    "shift0": shift0,
                    "seg_mask": seg_mask,
                }
            )
            start_piece_index += k

        # Build initial packed state.
        if len(initial_positions_by_type) != len(self.types):
            raise ValueError("initial_positions_by_type must match piece_types length")

        state = 0
        for ti, tdata in enumerate(self.types):
            k = tdata["k"]
            shift0 = tdata["shift0"]
            pos_list = sorted(initial_positions_by_type[ti])
            if len(pos_list) != k:
                raise ValueError("Wrong number of pieces for a type")

            # Basic validity check: anchor must be valid (limits_at not None).
            limits_at = tdata["limits_at"]
            for p in pos_list:
                if p < 0 or p >= self.W * self.H or limits_at[p] is None:
                    raise ValueError("Invalid anchor position in initial state")

            seg = _encode_segment(pos_list)
            state |= seg << shift0

        self.initial_state = state

    def count_reachable(self):
        """
        BFS over canonical states. Returns number of reachable configurations.
        """
        seen = {self.initial_state}
        q = deque([self.initial_state])

        while q:
            s = q.popleft()

            # Decode all type segments for this state; compute total occupancy.
            occ = 0
            decoded_positions = []
            for tdata in self.types:
                seg = (s & tdata["seg_mask"]) >> tdata["shift0"]
                pos_list = _decode_segment(seg, tdata["k"])
                decoded_positions.append(pos_list)
                mask_at = tdata["mask_at"]
                for p in pos_list:
                    occ |= mask_at[p]

            # Generate moves type-by-type.
            for ti, tdata in enumerate(self.types):
                k = tdata["k"]
                pos_list = decoded_positions[ti]
                mask_at = tdata["mask_at"]
                limits_at = tdata["limits_at"]
                shift0 = tdata["shift0"]
                seg_mask = tdata["seg_mask"]

                for j in range(k):
                    pos = pos_list[j]
                    m_old = mask_at[pos]
                    occ_wo = occ ^ m_old

                    limits = limits_at[pos]  # (up, down, left, right)
                    for dir_idx, delta in enumerate(self._deltas):
                        limit = limits[dir_idx]
                        if limit <= 0:
                            continue

                        # Try steps in this direction; stop after first collision.
                        for step in range(1, limit + 1):
                            new_pos = pos + delta * step
                            new_mask = mask_at[new_pos]
                            if new_mask & occ_wo:
                                break

                            # Create updated, still-sorted positions for this type (single-element insertion).
                            new_positions = pos_list[:]  # small copy (k is small)
                            new_positions[j] = new_pos

                            idx = j
                            while (
                                idx > 0 and new_positions[idx] < new_positions[idx - 1]
                            ):
                                new_positions[idx], new_positions[idx - 1] = (
                                    new_positions[idx - 1],
                                    new_positions[idx],
                                )
                                idx -= 1
                            while (
                                idx < k - 1
                                and new_positions[idx] > new_positions[idx + 1]
                            ):
                                new_positions[idx], new_positions[idx + 1] = (
                                    new_positions[idx + 1],
                                    new_positions[idx],
                                )
                                idx += 1

                            new_seg = _encode_segment(new_positions)
                            new_state = (s & ~seg_mask) | (new_seg << shift0)

                            if new_state not in seen:
                                seen.add(new_state)
                                q.append(new_state)

        return len(seen)


def solve():
    # --- Example from the statement (reachable configurations = 208) ---
    # Board: 4x3
    # Pieces:
    #  - One L-triomino at top-left occupying (0,0),(1,0),(0,1) anchor at (0,0)
    #  - Seven indistinguishable 1x1 squares (red) at the remaining red cells
    # Empty cells (white) are just empty space.
    example = Puzzle(
        width=4,
        height=3,
        piece_types=[
            {"offsets": [(0, 0), (1, 0), (0, 1)], "count": 1},  # L triomino
            {"offsets": [(0, 0)], "count": 7},  # 1x1 squares
        ],
        initial_positions_by_type=[
            [0],  # L anchor at (0,0) => pos 0
            [2, 5, 6, 8, 9, 10, 11],  # red squares
        ],
    )
    assert example.count_reachable() == 208

    # --- Main puzzle ---
    # Board: 6x5
    # Empty cells are the white 1x2 area on the left (top two cells): not a piece.
    # Piece types:
    #  - Red L-triominoes (2): offsets [(0,0),(0,1),(1,0)] anchors at 1 and 4
    #  - Green L-triominoes (2): offsets [(0,1),(1,0),(1,1)] anchors at 2 and 22
    #  - Yellow vertical dominoes (2): offsets [(0,0),(0,1)] anchors at 11 and 16
    #  - Magenta 1x1 squares (6): anchors at 12,13,18,19,24,25
    #  - Blue 2x2 square (1): anchor at 14
    #  - Cyan horizontal domino (1): offsets [(0,0),(1,0)] anchor at 26
    main = Puzzle(
        width=6,
        height=5,
        piece_types=[
            {"offsets": [(0, 0), (0, 1), (1, 0)], "count": 2},  # red L
            {"offsets": [(0, 1), (1, 0), (1, 1)], "count": 2},  # green L
            {"offsets": [(0, 0), (0, 1)], "count": 2},  # vertical domino
            {"offsets": [(0, 0)], "count": 6},  # 1x1 squares
            {"offsets": [(0, 0), (1, 0), (0, 1), (1, 1)], "count": 1},  # 2x2
            {"offsets": [(0, 0), (1, 0)], "count": 1},  # horizontal domino
        ],
        initial_positions_by_type=[
            [1, 4],
            [2, 22],
            [11, 16],
            [12, 13, 18, 19, 24, 25],
            [14],
            [26],
        ],
    )

    print(main.count_reachable())


if __name__ == "__main__":
    solve()
