import ProjectEulerStatements.P55
namespace ProjectEulerSolutions.P55

partial def reverseInt (n : Nat) : Nat :=
  let s := toString n
  let rs := String.mk s.data.reverse
  rs.toNat!

partial def isPalindrome (n : Nat) : Bool :=
  let s := toString n
  s == String.mk s.data.reverse

partial def isLychrelCandidate (n : Nat) (maxIters : Nat) : Bool :=
  let rec loop (x : Nat) (i : Nat) : Bool :=
    if i == maxIters then
      true
    else
      let x := x + reverseInt x
      if isPalindrome x then
        false
      else
        loop x (i + 1)
  loop n 0

partial def solveCore (limit maxIters : Nat) : Nat :=
  let rec loop (n : Nat) (acc : Nat) : Nat :=
    if n >= limit then
      acc
    else
      let acc := if isLychrelCandidate n maxIters then acc + 1 else acc
      loop (n + 1) acc
  loop 1 0



def solve (limit maxIters : Nat) :=
  solveCore limit maxIters

theorem equiv (limit maxIters : Nat) :
    ProjectEulerStatements.P55.naive limit maxIters = solve limit maxIters := sorry

end ProjectEulerSolutions.P55
