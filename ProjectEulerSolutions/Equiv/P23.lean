import ProjectEulerSolutions.P23
namespace ProjectEulerSolutions.P23

theorem equiv (n : Nat) : ProjectEulerStatements.P23.naive n = solve n :=
  (ProjectEulerStatements.AbundantSums.unmarkedSum_values n).symm
end ProjectEulerSolutions.P23
