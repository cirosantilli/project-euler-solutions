import ProjectEulerStatements.P1

namespace ProjectEulerSolvers.P1

def sumOfMultiples (m n : Nat) : Nat :=
  let k := n / m
  (m * k * (k + 1)) / 2

def sol (n : Nat) : Nat :=
  let n' := n - 1
  sumOfMultiples 3 n' + sumOfMultiples 5 n' - sumOfMultiples 15 n'

example : sol 10 = 23 := rfl

theorem equiv (n : Nat) : ProjectEulerStatements.P1.naive n = sol n := sorry

end ProjectEulerSolvers.P1
open ProjectEulerSolvers.P1

def main : IO Unit := do
  IO.println (sol 1000)
