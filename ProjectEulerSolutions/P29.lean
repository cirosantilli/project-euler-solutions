import ProjectEulerStatements.P29
import ProjectEulerSolutions.Termination.P29
namespace ProjectEulerSolutions.P29

def solve (maxA maxB : Nat) : Nat :=
  let rec loopA (a : Nat) (vals : List Nat) : List Nat :=
    if a > maxA then
      vals
    else
      let rec loopB (b : Nat) (vals : List Nat) : List Nat :=
        if b > maxB then
          vals
        else
          let v := a ^ b
          let vals' := if vals.contains v then vals else v :: vals
          loopB (b + 1) vals'
      termination_by 0
      decreasing_by all_goals exact Termination.decreases
      loopA (a + 1) (loopB 2 vals)
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  (loopA 2 []).length


termination_by 0
decreasing_by all_goals exact Termination.decreases
example : solve 5 5 = 15 := by
  native_decide
end ProjectEulerSolutions.P29
