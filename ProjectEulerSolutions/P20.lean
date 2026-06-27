import ProjectEulerStatements.P20
import ProjectEulerSolutions.Termination.P20
namespace ProjectEulerSolutions.P20

def digitSum (n : Nat) : Nat :=
  let s := toString n
  s.data.foldl (fun acc c => acc + (c.toNat - '0'.toNat)) 0

def solve (n : Nat) : Nat :=
  let rec loop (k acc : Nat) : Nat :=
    if k > n then
      digitSum acc
    else
      loop (k + 1) (acc * k)
  termination_by n + 1 - k
  decreasing_by
    simp_wf
    exact Termination.loop_decreases (by assumption)
  loop 2 1

example : solve 10 = 27 := by
  native_decide
end ProjectEulerSolutions.P20
