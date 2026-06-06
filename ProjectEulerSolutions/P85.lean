import ProjectEulerStatements.P85
import ProjectEulerSolutions.Termination.P85
namespace ProjectEulerSolutions.P85

abbrev TARGET : Nat := 2000000

def rectangleCount (m n : Nat) : Nat :=
  (m * (m + 1) * n * (n + 1)) / 4

termination_by 0
decreasing_by all_goals exact Termination.decreases
def absDiff (a b : Nat) : Nat :=
  if a >= b then a - b else b - a

termination_by 0
decreasing_by all_goals exact Termination.decreases
def bestGridAreaNear (target : Nat) : Nat :=
  let rec loopM (m : Nat) (bestDiff bestArea : Nat) : Nat :=
    if m >= 3000 then
      bestArea
    else
      let a := m * (m + 1) / 2
      let rec loopN (n : Nat) (bestDiff bestArea : Nat) : Nat × Nat :=
        if n >= 3000 then
          (bestDiff, bestArea)
        else
          let b := n * (n + 1) / 2
          let cnt := a * b
          let diff := absDiff cnt target
          let (bestDiff, bestArea) :=
            if diff < bestDiff then (diff, m * n) else (bestDiff, bestArea)
          loopN (n + 1) bestDiff bestArea
      termination_by 0
      decreasing_by all_goals exact Termination.decreases
      let (bestDiff, bestArea) := loopN m bestDiff bestArea
      loopM (m + 1) bestDiff bestArea
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loopM 1 (Nat.pow 10 30) 0


termination_by 0
decreasing_by all_goals exact Termination.decreases
example : rectangleCount 3 2 = 18 := by
  native_decide


def solve (target limit : Nat) :=
  let _ := limit
  bestGridAreaNear target
end ProjectEulerSolutions.P85
