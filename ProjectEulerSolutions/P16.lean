import ProjectEulerStatements.P16
import ProjectEulerSolutions.Termination.P16
namespace ProjectEulerSolutions.P16

def digitSum (n : Nat) : Nat :=
  let s := toString n
  s.data.foldl (fun acc c => acc + (c.toNat - '0'.toNat)) 0


termination_by 0
decreasing_by all_goals exact Termination.decreases
example : digitSum (Nat.pow 2 15) = 26 := by
  native_decide


def solve (n : Nat) :=
  digitSum (Nat.pow 2 n)
end ProjectEulerSolutions.P16
