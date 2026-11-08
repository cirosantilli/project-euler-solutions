import ProjectEulerSolutions.Lib.SignedPowerComparison

namespace ProjectEulerSolutions.PowerMaximum
open PowerIntervals

abbrev Pair := Nat × Int

/-- One-based selection retains the earlier index when two exact powers agree. -/
def scan : List Pair → Nat → Nat → Pair → Nat
  | [], _, best, _ => best
  | current :: rest, next, best, incumbent =>
      if best = 0 ∨ 0 < signedComparison current.1 current.2 incumbent.1 incumbent.2 then
        scan rest (next + 1) next current
      else scan rest (next + 1) best incumbent

def maximumIndex (pairs : List Pair) : Nat := scan pairs 1 0 (1, 0)

theorem maximumIndex_nil : maximumIndex [] = 0 := rfl

theorem maximumIndex_cons (first : Pair) (rest : List Pair) :
    maximumIndex (first :: rest) = scan rest 2 1 first := by
  simp [maximumIndex, scan]

theorem scan_positive (pairs : List Pair) (next best : Nat) (incumbent : Pair)
    (positiveNext : 0 < next) (positiveBest : 0 < best) :
    0 < scan pairs next best incumbent := by
  induction pairs generalizing next best incumbent with
  | nil => exact positiveBest
  | cons current rest ih =>
      rw [scan]
      split
      · exact ih _ _ _ (by omega) positiveNext
      · exact ih _ _ _ (by omega) positiveBest

theorem maximumIndex_positive (first : Pair) (rest : List Pair) :
    0 < maximumIndex (first :: rest) := by
  rw [maximumIndex_cons]
  exact scan_positive rest 2 1 first (by decide) (by decide)

theorem scan_lt_end (pairs : List Pair) (next best : Nat) (incumbent : Pair) (earlier : best < next) :
    scan pairs next best incumbent < next + pairs.length := by
  induction pairs generalizing next best incumbent with
  | nil => simpa [scan] using earlier
  | cons current rest ih =>
      rw [scan]
      split
      · have := ih (next + 1) next current (by omega)
        simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using this
      · have := ih (next + 1) best incumbent (by omega)
        simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using this

theorem maximumIndex_le_length (pairs : List Pair) : maximumIndex pairs ≤ pairs.length := by
  have := scan_lt_end pairs 1 0 (1, 0) (by decide)
  change scan pairs 1 0 (1, 0) ≤ pairs.length
  omega

end ProjectEulerSolutions.PowerMaximum
