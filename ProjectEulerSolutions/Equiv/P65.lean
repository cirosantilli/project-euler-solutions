import ProjectEulerSolutions.P65

namespace ProjectEulerSolutions.P65

theorem equiv (n : Nat) : ProjectEulerStatements.P65.naive n = solve n :=
  (EConvergents.answer_eq_naive n).symm

end ProjectEulerSolutions.P65
