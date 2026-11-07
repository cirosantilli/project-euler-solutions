import ProjectEulerSolutions.P93
namespace ProjectEulerSolutions.P93
open ProjectEulerSolutions.ArithmeticExpressions

private theorem mem_reduce (fuel : Nat) (xs : List Rat) (bound : xs.length ≤ fuel) (value : Rat) :
    value ∈ reduceFuel fuel xs ↔ Reduces xs value := by
  induction fuel generalizing xs with
  | zero =>
    have empty : xs = [] := List.length_eq_zero_iff.mp (by omega)
    subst xs
    rw [reduces_iff_mem_permuted]
    simp [reduceFuel, permuted, enumerate]
  | succ fuel ih =>
    match xs with
    | [] =>
      rw [reduces_iff_mem_permuted]
      simp [reduceFuel, permuted, enumerate]
    | [a] =>
      rw [reduces_iff_mem_permuted]
      simp [reduceFuel, permuted, enumerate]
    | a :: b :: tail =>
      rw [reduceFuel]
      constructor
      · intro member
        obtain ⟨order, ordered, member⟩ := List.mem_flatMap.mp member
        have permutation := List.mem_permutations.mp ordered
        have size := permutation.length_eq
        cases order with
        | nil => simp at member
        | cons x rest =>
          cases rest with
          | nil => simp at member
          | cons y rest =>
            obtain ⟨combined, operation, member⟩ := List.mem_flatMap.mp member
            have small : (combined :: rest).length ≤ fuel := by simp_all
            exact Reduces.step permutation.symm operation ((ih _ small).mp member)
      · intro derived
        cases derived with
        | @step xs rest x y combined value permutation operation continued =>
          apply List.mem_flatMap.mpr
          refine ⟨x :: y :: rest, List.mem_permutations.mpr permutation.symm, ?_⟩
          apply List.mem_flatMap.mpr
          have size := permutation.length_eq
          have small : (combined :: rest).length ≤ fuel := by simp_all
          exact ⟨combined, operation, (ih _ small).mpr continued⟩

private theorem scores_eq (digits : List Nat) : score digits = ProjectEulerStatements.P93.score digits := by
  apply ProjectEulerSolutions.PositiveRun.congr
  intro n _
  simp only [ProjectEulerStatements.P93.integers, List.mem_map, List.mem_filter]
  have same (value : Rat) : value ∈ reduceFuel digits.length (digits.map (fun n => (n : Rat))) ↔
      value ∈ permuted (digits.map (fun n => (n : Rat))) := by
    rw [mem_reduce _ _ (by simp), reduces_iff_mem_permuted]
  simp only [same]

theorem equiv (maxDigit : Int) : ProjectEulerStatements.P93.naive maxDigit = solve maxDigit := by
  simp only [ProjectEulerStatements.P93.naive, solve, show score = ProjectEulerStatements.P93.score from funext scores_eq]
end ProjectEulerSolutions.P93
