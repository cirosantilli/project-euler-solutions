import ProjectEulerSolutions.P76

namespace ProjectEulerSolutions.P76

theorem equiv (n : Int) : ProjectEulerStatements.P76.naive n = solve n := (solve_result n).symm

end ProjectEulerSolutions.P76
