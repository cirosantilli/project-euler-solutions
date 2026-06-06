import ProjectEulerStatements.P9
import Mathlib.Tactic
namespace ProjectEulerSolutions.P9

def loopN (total m n : Nat) : Nat :=
  if n >= m then
    0
  else
    let s := 2 * m * (m + n)
    if total % s != 0 then
      loopN total m (n + 1)
    else
      let k := total / s
      let a := k * (m * m - n * n)
      let b := k * (2 * m * n)
      let c := k * (m * m + n * n)
      if a > 0 && a * a + b * b == c * c && a + b + c == total then
        a * b * c
      else
        loopN total m (n + 1)
termination_by m - n
decreasing_by
  all_goals omega

def loopM (total m : Nat) : Nat :=
  if hstop : 2 * m * (m + 1) > total then
    0
  else
    let res := loopN total m 1
    if res != 0 then res else loopM total (m + 1)
termination_by total + 1 - m
decreasing_by
  have hm_le : m ≤ total := by
    have hmul : m ≤ 2 * m * (m + 1) := by nlinarith
    exact le_trans hmul (le_of_not_gt hstop)
  omega

def solve (total : Nat) : Nat :=
  loopM total 2

example : solve 12 = 60 := by
  native_decide
end ProjectEulerSolutions.P9
