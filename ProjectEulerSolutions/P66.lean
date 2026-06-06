import ProjectEulerStatements.P66
import ProjectEulerSolutions.Termination.P66
namespace ProjectEulerSolutions.P66

def sqrtFloor (n : Nat) : Nat :=
  let rec loop (lo hi : Nat) : Nat :=
    if lo > hi then
      hi
    else
      let mid := (lo + hi) / 2
      let sq := mid * mid
      if sq == n then
        mid
      else if sq < n then
        loop (mid + 1) hi
      else
        loop lo (mid - 1)
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loop 1 n

termination_by 0
decreasing_by all_goals exact Termination.decreases
def minimalPellX (d : Nat) : Option Nat :=
  let a0 := sqrtFloor d
  if a0 * a0 == d then
    none
  else
    let rec loop (m d0 a h_m2 h_m1 k_m2 k_m1 : Nat) : Nat :=
      let h := a * h_m1 + h_m2
      let k := a * k_m1 + k_m2
      if h * h == d * k * k + 1 then
        h
      else
        let m := d0 * a - m
        let d1 := (d - m * m) / d0
        let a := (a0 + m) / d1
        loop m d1 a h_m1 h k_m1 k
    termination_by 0
    decreasing_by all_goals exact Termination.decreases
    some (loop 0 1 a0 0 1 1 0)

termination_by 0
decreasing_by all_goals exact Termination.decreases
def solveCore (limit : Nat) : Nat :=
  let rec loop (d : Nat) (bestD bestX : Nat) : Nat :=
    if d > limit then
      bestD
    else
      match minimalPellX d with
      | none => loop (d + 1) bestD bestX
      | some x =>
          if x > bestX then
            loop (d + 1) d x
          else
            loop (d + 1) bestD bestX
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loop 2 0 0


termination_by 0
decreasing_by all_goals exact Termination.decreases
example : minimalPellX 13 = some 649 := by
  native_decide


def solve (limit : Nat) :=
  solveCore limit
end ProjectEulerSolutions.P66
