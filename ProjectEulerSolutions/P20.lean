import ProjectEulerStatements.P20
import ProjectEulerSolutions.Termination.P20
namespace ProjectEulerSolutions.P20

def digitSum (n : Nat) : Nat :=
  let s := toString n
  s.data.foldl (fun acc c => acc + (c.toNat - '0'.toNat)) 0

termination_by 0
decreasing_by all_goals exact Termination.decreases
def solve (n : Nat) : Nat :=
  let rec loop (k acc : Nat) : Nat :=
    if k > n then
      digitSum acc
    else
      loop (k + 1) (acc * k)
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loop 2 1

termination_by 0
decreasing_by all_goals exact Termination.decreases
example : solve 10 = 27 := by
  native_decide
end ProjectEulerSolutions.P20
