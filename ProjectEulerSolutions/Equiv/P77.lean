import ProjectEulerSolutions.P77

namespace ProjectEulerSolutions.P77

theorem equiv (target : Int) : ProjectEulerStatements.P77.naive target = solve target :=
  (solve_result target).symm

end ProjectEulerSolutions.P77
