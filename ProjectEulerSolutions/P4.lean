import ProjectEulerStatements.P4
namespace ProjectEulerSolutions.P4

def isPalindrome (n : Nat) : Bool :=
  let s := toString n
  s = String.mk s.data.reverse

def loopB (a b lo best bestA bestB : Nat) : Nat × Nat × Nat :=
  if _hblo : b < lo then
    (best, bestA, bestB)
  else if _hb0 : b = 0 then
    (best, bestA, bestB)
  else
    let prod := a * b
    if hbest : prod <= best then
      (best, bestA, bestB)
    else if isPalindrome prod then
      (prod, a, b)
    else
      loopB a (b - 1) lo best bestA bestB
termination_by b
decreasing_by
  omega

def loopA (a lo hi best bestA bestB : Nat) : Nat × Nat × Nat :=
  if a < lo then
    (best, bestA, bestB)
  else if a * hi < best then
    (best, bestA, bestB)
  else
    let (best', bestA', bestB') := loopB a a lo best bestA bestB
    if a = 0 then
      (best', bestA', bestB')
    else
      loopA (a - 1) lo hi best' bestA' bestB'
termination_by a
decreasing_by
  omega

def largestPalindromeProduct (lo hi : Nat) : Nat × Nat × Nat :=
  loopA hi lo hi 0 0 0

example : (largestPalindromeProduct 10 99).1 = 9009 := by
  native_decide


def solve (digits : Nat) : Nat :=
  let lo :=
    match digits with
    | 0 => 0
    | d + 1 => Nat.pow 10 d
  let hi := Nat.pow 10 digits - 1
  (largestPalindromeProduct lo hi).1
end ProjectEulerSolutions.P4
