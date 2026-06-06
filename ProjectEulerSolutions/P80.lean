import ProjectEulerStatements.P80
import ProjectEulerSolutions.Termination.P80
namespace ProjectEulerSolutions.P80

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
def isPerfectSquare (n : Nat) : Bool :=
  let r := sqrtFloor n
  r * r == n

termination_by 0
decreasing_by all_goals exact Termination.decreases
def digitSumDigitsOfSqrt (n digits : Nat) : Nat :=
  if digits == 0 then
    0
  else
    let scale := Nat.pow 10 (2 * (digits - 1))
    let scaled := n * scale
    let rootScaled := sqrtFloor scaled
    let s := toString rootScaled
    s.data.foldl (fun acc c => acc + (c.toNat - '0'.toNat)) 0

termination_by 0
decreasing_by all_goals exact Termination.decreases
def solve (limit digits : Nat) : Nat :=
  let rec loop (n : Nat) (total : Nat) : Nat :=
    if n > limit then
      total
    else
      if isPerfectSquare n then
        loop (n + 1) total
      else
        loop (n + 1) (total + digitSumDigitsOfSqrt n digits)
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loop 1 0


termination_by 0
decreasing_by all_goals exact Termination.decreases
example : digitSumDigitsOfSqrt 2 100 = 475 := by
  native_decide
end ProjectEulerSolutions.P80
