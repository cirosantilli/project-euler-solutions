import ProjectEulerStatements.Lib.AbundantSums
namespace ProjectEulerSolutions.P23

/-- Mark sums of abundant pairs, then sum the unmarked positive integers. -/
def solve (limit : Nat) : Nat :=
  let abundant := ProjectEulerStatements.AbundantSums.values limit
  let marked := ProjectEulerStatements.AbundantSums.pairs abundant
  ProjectEulerStatements.AbundantSums.unmarkedSum marked 1 limit
end ProjectEulerSolutions.P23
