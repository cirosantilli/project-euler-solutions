import ProjectEulerStatements.P1

namespace ProjectEulerSolutions.P1

def sumOfMultiples (m n : Nat) : Nat :=
  let k := n / m
  (m * k * (k + 1)) / 2

def solve (n : Nat) : Nat :=
  let n' := n - 1
  sumOfMultiples 3 n' + sumOfMultiples 5 n' - sumOfMultiples 15 n'

example : solve 10 = 23 := rfl
end ProjectEulerSolutions.P1
