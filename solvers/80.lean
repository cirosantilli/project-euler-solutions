import ProjectEulerStatements.P80
namespace ProjectEulerSolutions.P80

partial def sqrtFloor (n : Nat) : Nat :=
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
  loop 1 n

partial def isPerfectSquare (n : Nat) : Bool :=
  let r := sqrtFloor n
  r * r == n

partial def digitSumDigitsOfSqrt (n digits : Nat) : Nat :=
  if digits == 0 then
    0
  else
    let scale := Nat.pow 10 (2 * (digits - 1))
    let scaled := n * scale
    let rootScaled := sqrtFloor scaled
    let s := toString rootScaled
    s.data.foldl (fun acc c => acc + (c.toNat - '0'.toNat)) 0

partial def solve (limit digits : Nat) : Nat :=
  let rec loop (n : Nat) (total : Nat) : Nat :=
    if n > limit then
      total
    else
      if isPerfectSquare n then
        loop (n + 1) total
      else
        loop (n + 1) (total + digitSumDigitsOfSqrt n digits)
  loop 1 0


example : digitSumDigitsOfSqrt 2 100 = 475 := by
  native_decide


theorem equiv (n digits : Nat) : ProjectEulerStatements.P80.naive n digits = solve n digits := sorry
end ProjectEulerSolutions.P80
open ProjectEulerSolutions.P80

def main : IO Unit := do
  IO.println (solve 100 100)
