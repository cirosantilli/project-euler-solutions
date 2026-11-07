import ProjectEulerSolutions.P99
namespace ProjectEulerSolutions.P99
open ProjectEulerSolutions.PowerIntervals

private theorem compare_eq (a : Nat) (m : Int) (b : Nat) (n : Int) (ha : 0 < a) (hb : 0 < b) :
    comparePowers a m b n = signedComparison a m b n := by
  rw [comparePowers, dif_pos ha, dif_pos hb]
  have same : comparePositive a m b n ha hb = comparisonAlgorithm a m b n := by
    simp only [comparePositive, comparisonAlgorithm, normalizedComparison, refine_eq]
  rw [same, comparisonAlgorithm_eq a m b n ha hb]

private theorem scan_eq (pairs : List (Nat × Int)) (next best : Nat) (incumbent : Nat × Int)
    (valid : ∀ pair ∈ pairs, 0 < pair.1) (positive : 0 < incumbent.1) :
    scan pairs next best incumbent = ProjectEulerSolutions.PowerMaximum.scan pairs next best incumbent := by
  induction pairs generalizing next best incumbent with
  | nil => rfl
  | cons current rest ih =>
    have hc := valid current (by simp)
    have hr : ∀ pair ∈ rest, 0 < pair.1 := fun pair member => valid pair (by simp [member])
    simp only [scan, ProjectEulerSolutions.PowerMaximum.scan, compare_eq _ _ _ _ hc positive]
    split <;> apply ih <;> assumption

theorem equiv (pairs : List (Nat × Int)) (valid : ∀ pair ∈ pairs, 0 < pair.1) :
    ProjectEulerStatements.P99.naive pairs = solve pairs :=
  (scan_eq pairs 1 0 (1, 0) valid (by decide)).symm
end ProjectEulerSolutions.P99
