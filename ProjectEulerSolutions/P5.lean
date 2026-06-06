import ProjectEulerStatements.P5
namespace ProjectEulerSolutions.P5

def lcm (a b : Nat) : Nat :=
  a / Nat.gcd a b * b

def go (n x acc : Nat) : Nat :=
  if x > n then
    acc
  else
    go n (x + 1) (lcm acc x)
termination_by n + 1 - x
decreasing_by
  omega

def solve (n : Nat) : Nat :=
  go n 2 1

example : solve 10 = 2520 := by
  native_decide
end ProjectEulerSolutions.P5
