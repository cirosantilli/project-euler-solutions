import ProjectEulerStatements.P55
import ProjectEulerSolutions.Termination.P55
namespace ProjectEulerSolutions.P55

def reverseInt (n : Nat) : Nat :=
  let s := toString n
  let rs := String.mk s.data.reverse
  rs.toNat!

termination_by 0
decreasing_by all_goals exact Termination.decreases
def isPalindrome (n : Nat) : Bool :=
  let s := toString n
  s == String.mk s.data.reverse

termination_by 0
decreasing_by all_goals exact Termination.decreases
def isLychrelCandidate (n : Nat) (maxIters : Nat) : Bool :=
  let rec loop (x : Nat) (i : Nat) : Bool :=
    if i == maxIters then
      true
    else
      let x := x + reverseInt x
      if isPalindrome x then
        false
      else
        loop x (i + 1)
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loop n 0

termination_by 0
decreasing_by all_goals exact Termination.decreases
def solve (limit maxIters : Nat) : Nat :=
  let rec loop (n : Nat) (acc : Nat) : Nat :=
    if n >= limit then
      acc
    else
      let acc := if isLychrelCandidate n maxIters then acc + 1 else acc
      loop (n + 1) acc
  loop 1 0
termination_by 0
decreasing_by all_goals exact Termination.decreases
end ProjectEulerSolutions.P55
