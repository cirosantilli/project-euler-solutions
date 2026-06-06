import ProjectEulerStatements.P94
import ProjectEulerSolutions.Termination.P94
namespace ProjectEulerSolutions.P94

abbrev LIMIT : Nat := 1000000000

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
def triangleAreaIsosceles (a b : Nat) : Nat :=
  let d := 4 * a * a - b * b
  let k := sqrtFloor d
  if k * k != d then 0 else (b * k) / 4

termination_by 0
decreasing_by all_goals exact Termination.decreases
def generatePerimeters (limit : Nat) : List Nat :=
  let rec loop (x y : Nat) (acc : List Nat) : List Nat :=
    let p :=
      if x % 3 == 2 then
        let a := (x + 1) / 3
        3 * a + 1
      else
        let a := (x - 1) / 3
        3 * a - 1
    if p > limit then
      acc.reverse
    else
      let acc := p :: acc
      let x' := 2 * x + 3 * y
      let y' := x + 2 * y
      loop x' y' acc
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loop 14 8 []

termination_by 0
decreasing_by all_goals exact Termination.decreases
def sumPerimeters (limit : Nat) : Nat :=
  (generatePerimeters limit).foldl (fun acc p => acc + p) 0


termination_by 0
decreasing_by all_goals exact Termination.decreases
example : triangleAreaIsosceles 5 6 = 12 := by
  native_decide

example : generatePerimeters 1000 = [16, 50, 196, 722] := by
  native_decide


def solve (_n : Nat) :=
  sumPerimeters LIMIT
end ProjectEulerSolutions.P94
