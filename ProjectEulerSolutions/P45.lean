import ProjectEulerStatements.P45
import ProjectEulerSolutions.Termination.P45
namespace ProjectEulerSolutions.P45

def triangle (n : Nat) : Nat :=
  n * (n + 1) / 2

termination_by 0
decreasing_by all_goals exact Termination.decreases
def pentagonal (n : Nat) : Nat :=
  n * (3 * n - 1) / 2

termination_by 0
decreasing_by all_goals exact Termination.decreases
def hexagonal (n : Nat) : Nat :=
  n * (2 * n - 1)

termination_by 0
decreasing_by all_goals exact Termination.decreases
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
def sqrtIfSquare (n : Nat) : Nat :=
  let s := sqrtFloor n
  if s * s == n then s else 0

termination_by 0
decreasing_by all_goals exact Termination.decreases
def isPentagonal (x : Nat) : Bool :=
  let d := 1 + 24 * x
  let s := sqrtIfSquare d
  s != 0 && (1 + s) % 6 == 0

termination_by 0
decreasing_by all_goals exact Termination.decreases
def isHexagonal (x : Nat) : Bool :=
  let d := 1 + 8 * x
  let s := sqrtIfSquare d
  s != 0 && (1 + s) % 4 == 0

termination_by 0
decreasing_by all_goals exact Termination.decreases
def solve (start limit : Nat) : Nat :=
  let _ := triangle 285
  let _ := pentagonal 165
  let _ := hexagonal 143
  let rec loop (i : Nat) : Nat :=
    if i > limit then
      0
    else
      let t := triangle (start + i)
      if isPentagonal t && isHexagonal t then t else loop (i + 1)
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loop 0


termination_by 0
decreasing_by all_goals exact Termination.decreases
example : triangle 285 = 40755 := by
  native_decide

example : pentagonal 165 = 40755 := by
  native_decide

example : hexagonal 143 = 40755 := by
  native_decide
end ProjectEulerSolutions.P45
