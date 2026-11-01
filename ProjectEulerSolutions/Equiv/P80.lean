import ProjectEulerSolutions.P80
import ProjectEulerStatements.Lib.SquareRootDigits

namespace ProjectEulerSolutions.P80

theorem equiv (limit digits : Nat) : ProjectEulerStatements.P80.naive limit digits = solve limit digits := by
  rw [ProjectEulerStatements.SquareRootDigits.skip_zero]
  unfold solve
  congr 1
  funext total n
  by_cases square : Nat.sqrt n * Nat.sqrt n = n <;>
    simp [ProjectEulerStatements.P80.sqrtDigitsSum, ProjectEulerStatements.P80.isSquare,
      isPerfectSquare, sqrtFloor, digitSumDigitsOfSqrt, square]

end ProjectEulerSolutions.P80
