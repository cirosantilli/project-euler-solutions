import ProjectEulerStatements.P100
import ProjectEulerSolutions.Termination.P100
namespace ProjectEulerSolutions.P100

def nextSolution (u v : Nat) : Nat × Nat :=
  (3 * u + 4 * v, 2 * u + 3 * v)

termination_by 0
decreasing_by all_goals exact Termination.decreases
def blueRedForTotalLimit (limitN : Nat) : Nat × Nat × Nat :=
  let rec loop (u v : Nat) : Nat × Nat × Nat :=
    let n := (u + 1) / 2
    let b := (v + 1) / 2
    if n > limitN then
      (b, n - b, n)
    else
      let (u, v) := nextSolution u v
      loop u v
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loop 1 1


termination_by 0
decreasing_by all_goals exact Termination.decreases
example :
    let (u1, v1) := (1, 1)
    let (u2, v2) := nextSolution u1 v1
    let (u3, v3) := nextSolution u2 v2
    let (u4, v4) := nextSolution u3 v3
    let n3 := (u3 + 1) / 2
    let b3 := (v3 + 1) / 2
    let n4 := (u4 + 1) / 2
    let b4 := (v4 + 1) / 2
    (b3, n3) = (15, 21) && (b4, n4) = (85, 120) = true := by
  native_decide


def solve (limit : Nat) :=
  (blueRedForTotalLimit limit).1
end ProjectEulerSolutions.P100
