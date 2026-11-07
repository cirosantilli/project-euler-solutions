import ProjectEulerSolutions.P36
namespace ProjectEulerSolutions.P36

theorem equiv (n : Nat) : ProjectEulerStatements.P36.naive n = solve n :=
  (ProjectEulerStatements.DoublePalindromeSearch.scan_correct n).symm
end ProjectEulerSolutions.P36
