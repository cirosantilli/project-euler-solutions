import ProjectEulerSolutions.P68

set_option linter.style.header false

/-! The pruned search and exhaustive reference have the same result for every pair of inputs. -/
namespace ProjectEulerSolutions.P68
open ProjectEulerStatements.MagicRing

theorem equiv_at (n digits : Int) : ProjectEulerStatements.P68.naive n digits = solve n digits := by
  by_cases rejected : n < 3 ∨ digits < 3 * n
  · simp only [ProjectEulerStatements.P68.naive, referenceResult, solve, if_pos rejected]
  · have size : 2 ≤ n.toNat := by omega
    simp only [ProjectEulerStatements.P68.naive, referenceResult, solve, if_neg rejected]
    exact ((prunedSelectionScan_answer n.toNat digits.toNat).trans
      (answers_equal n.toNat digits.toNat size)).symm

theorem equiv : @ProjectEulerStatements.P68.naive = @solve :=
  funext (fun n => funext (equiv_at n))

end ProjectEulerSolutions.P68
