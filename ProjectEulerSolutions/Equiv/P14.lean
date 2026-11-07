import ProjectEulerSolutions.P14
namespace ProjectEulerSolutions.P14
open ProjectEulerStatements.P14 ProjectEulerStatements.ArgMaximum

theorem scan_correct (items : List Nat) (best : Nat) (cache : Cache) :
    scan items best cache = items.foldl (select collatzLen) best := by
  induction items generalizing best cache with
  | nil => rfl
  | cons head tail ih =>
    simp only [scan, List.foldl_cons, ih, select,
      (memoLength head cache).property,
      (memoLength best (memoLength head cache).val.2).property]

theorem equiv (n : Nat) : ProjectEulerStatements.P14.naive n = solve n := by
  rw [naive, argmax_range_succ]
  exact (scan_correct _ 0 initialCache).symm
end ProjectEulerSolutions.P14
