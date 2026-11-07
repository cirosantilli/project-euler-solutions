import ProjectEulerStatements.Lib.TruncatablePrimeBase
import ProjectEulerStatements.Lib.TruncatablePrimeDecimal
namespace ProjectEulerSolutions.P37
open ProjectEulerStatements.TruncatablePrimeBase

/-- Expand the right-truncatable frontier, accumulating only left-truncatable children. -/
def search (base : Nat) (success : Exhausts base) (depth : Nat) (frontier : List Nat) (total : Nat)
    (frontierCorrect : frontier = level base depth) (before : depth ≤ Nat.find success) : Nat :=
  if empty : frontier = [] then total
  else
    let next := frontier.flatMap (children base)
    search base success (depth+1) next (total + (next.filter (leftCheck base)).sum)
      (by simp [next, frontierCorrect, level]) (by
        have different : depth ≠ Nat.find success := by
          intro same
          exact empty (by rw [frontierCorrect, same]; exact Nat.find_spec success)
        omega)
termination_by Nat.find success - depth
decreasing_by
  have nonempty : frontier ≠ [] := by assumption
  have different : depth ≠ Nat.find success := by
    intro same
    exact nonempty (by rw [frontierCorrect, same]; exact Nat.find_spec success)
  omega

def solve (base : Nat) (success : Exhausts base) : Nat :=
  search base success 0 (roots base) 0 rfl (Nat.zero_le _)

theorem defaultExhausts : Exhausts 10 := ⟨8, Decimal.exhausted⟩
end ProjectEulerSolutions.P37
