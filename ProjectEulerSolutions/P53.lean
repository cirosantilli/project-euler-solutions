import ProjectEulerStatements.P53

namespace ProjectEulerSolutions.P53

/-- Once a binomial coefficient exceeds the threshold, symmetry counts the remaining row. -/
def countRow (n threshold r coefficient : Nat) : Nat :=
  if r > n / 2 then 0
  else
    let next := coefficient * (n - r + 1) / r
    if next > threshold then n - 2 * r + 1
    else countRow n threshold (r + 1) next
termination_by n / 2 + 1 - r

def countCombinatoricSelections (limitN threshold : Nat) : Nat :=
  ((List.range' 1 limitN).map fun n => countRow n threshold 1 1).sum

def solveCore (limitN threshold : Nat) : Nat := countCombinatoricSelections limitN threshold

def solve : Nat := solveCore 100 1000000

end ProjectEulerSolutions.P53
