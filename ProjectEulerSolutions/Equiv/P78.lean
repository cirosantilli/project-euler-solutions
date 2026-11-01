import ProjectEulerSolutions.P78

namespace ProjectEulerSolutions.P78

theorem equiv (modulus : Int) : ProjectEulerStatements.P78.naive modulus = solve modulus :=
  (solve_result modulus).symm

end ProjectEulerSolutions.P78
