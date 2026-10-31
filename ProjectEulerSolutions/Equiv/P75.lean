import ProjectEulerSolutions.P75
namespace ProjectEulerSolutions.P75

theorem equiv (limit : Int) : ProjectEulerStatements.P75.naive limit = solve limit :=
  (solve_result limit).symm
end ProjectEulerSolutions.P75
