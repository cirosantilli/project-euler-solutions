import ProjectEulerSolutions.P64

namespace ProjectEulerSolutions.P64

theorem equiv (n : Nat) : ProjectEulerStatements.P64.naive n = solve n :=
  (SqrtPeriodCounts.answer_eq_naive n).symm

end ProjectEulerSolutions.P64
