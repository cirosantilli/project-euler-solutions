import ProjectEulerSolutions.P56
import ProjectEulerSolutions.Lib.PowerDigitSumReference

namespace ProjectEulerSolutions.P56

theorem equiv (n : Nat) : ProjectEulerStatements.P56.naive n = solve n := by
  simp [solve, PowerDigitSum.maximum_eq_naive]

end ProjectEulerSolutions.P56
