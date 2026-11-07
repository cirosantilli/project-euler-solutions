import ProjectEulerSolutions.P87
import ProjectEulerSolutions.Lib.ListEraseDups
namespace ProjectEulerSolutions.P87
open ProjectEulerSolutions.PrimePowerTriples

private theorem fold_triples (limit : Nat) (ps : List Nat) :
    ps.foldr (fun p acc => ps.foldr (fun q acc2 => ps.foldr (fun r acc3 =>
      if p^2 + q^3 + r^4 < limit then (p^2 + q^3 + r^4) :: acc3 else acc3) acc2) acc) [] =
    referenceValues limit ps := by
  conv_rhs => rw [← (List.foldr_cons_nil (l := referenceValues limit ps))]
  simp only [referenceValues, List.foldr_flatMap, List.foldr_filter, List.foldr_map,
    decide_eq_true_eq]

theorem equiv (n : Nat) : ProjectEulerStatements.P87.naive n = solve n := by
  rw [solve_eq]
  unfold ProjectEulerStatements.P87.naive ProjectEulerStatements.P87.countSums
  dsimp only
  rw [fold_triples]
  rw [ProjectEulerSolutions.ListEraseDups.length_eq_card]
  rw [referenceValues_complete]
  intro p
  simp [ProjectEulerStatements.P87.primesBelow, ProjectEulerStatements.P87.isPrime,
    primesThrough]
end ProjectEulerSolutions.P87
