import ProjectEulerSolutions.P35
namespace ProjectEulerSolutions.P35

theorem equiv (n : Nat) : ProjectEulerStatements.P35.naive n = solve n :=
  (ProjectEulerStatements.CircularPrimes.algorithmFold_correct n).symm
end ProjectEulerSolutions.P35
