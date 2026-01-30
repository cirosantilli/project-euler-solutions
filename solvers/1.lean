import ProjectEulerStatements.P1

namespace ProjectEulerSolutions.P1

def sumOfMultiples (m n : Nat) : Nat :=
  let k := n / m
  (m * k * (k + 1)) / 2

def solve (n : Nat) : Nat :=
  let n' := n - 1
  sumOfMultiples 3 n' + sumOfMultiples 5 n' - sumOfMultiples 15 n'

example : solve 10 = 23 := rfl
theorem equiv (n : Nat) : ProjectEulerStatements.P1.naive n = solve n := sorry

end ProjectEulerSolutions.P1
open ProjectEulerSolutions.P1

def main : IO Unit := do
  IO.println (solve 1000)
