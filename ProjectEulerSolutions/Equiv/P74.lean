import ProjectEulerSolutions.P74
namespace ProjectEulerSolutions.P74

theorem equiv (limit target : Int) :
    ProjectEulerStatements.P74.naive limit target = solve limit target :=
  (solve_result limit target).symm
end ProjectEulerSolutions.P74
