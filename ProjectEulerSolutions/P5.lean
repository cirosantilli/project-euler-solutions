import ProjectEulerStatements.P5
namespace ProjectEulerSolutions.P5

def lcm (a b : Nat) : Nat :=
  a / Nat.gcd a b * b

partial def solve (n : Nat) : Nat :=
  let rec go (x acc : Nat) : Nat :=
    if x > n then
      acc
    else
      go (x + 1) (lcm acc x)
  go 2 1

example : solve 10 = 2520 := by
  native_decide
end ProjectEulerSolutions.P5
