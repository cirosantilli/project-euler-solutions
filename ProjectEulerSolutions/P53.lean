import ProjectEulerStatements.P53
import ProjectEulerSolutions.Termination.P53
namespace ProjectEulerSolutions.P53

def countCombinatoricSelections (limitN threshold : Nat) : Nat :=
  let rec loopN (n : Nat) (total : Nat) : Nat :=
    if n > limitN then
      total
    else
      let rec loopR (r : Nat) (c : Nat) (total : Nat) : Nat :=
        if r > n / 2 then
          loopN (n + 1) total
        else
          let c := c * (n - r + 1) / r
          if c > threshold then
            let total := total + (n - 2 * r + 1)
            loopN (n + 1) total
          else
            loopR (r + 1) c total
      termination_by 0
      decreasing_by all_goals exact Termination.decreases
      loopR 1 1 total
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loopN 1 0



termination_by 0
decreasing_by all_goals exact Termination.decreases
def solveCore (limitN threshold : Nat) :=
  countCombinatoricSelections limitN threshold

def solve : Nat :=
  solveCore 100 1000000
end ProjectEulerSolutions.P53
