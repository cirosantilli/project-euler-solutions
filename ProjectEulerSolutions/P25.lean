import ProjectEulerStatements.P25
import ProjectEulerSolutions.Termination.P25
namespace ProjectEulerSolutions.P25

def fibPair (n : Nat) : Nat × Nat :=
  if n == 0 then
    (0, 1)
  else
    let (a, b) := fibPair (n / 2)
    let c := a * (2 * b - a)
    let d := a * a + b * b
    if n % 2 == 0 then
      (c, d)
    else
      (d, c + d)

termination_by 0
decreasing_by all_goals exact Termination.decreases
def fib (n : Nat) : Nat :=
  (fibPair n).1

termination_by 0
decreasing_by all_goals exact Termination.decreases
def solve (digits : Nat) : Nat :=
  if digits <= 1 then
    1
  else
    let threshold := Nat.pow 10 (digits - 1)
    let rec findUpper (hi : Nat) : Nat :=
      if fib hi >= threshold then hi else findUpper (hi * 2)
    termination_by 0
    decreasing_by all_goals exact Termination.decreases
    let rec binary (lo hi : Nat) : Nat :=
      if lo >= hi then
        lo
      else
        let mid := (lo + hi) / 2
        if fib mid >= threshold then
          binary lo mid
        else
          binary (mid + 1) hi
    termination_by 0
    decreasing_by all_goals exact Termination.decreases
    let hi := findUpper 1
    binary 1 hi

termination_by 0
decreasing_by all_goals exact Termination.decreases
example : solve 1 = 1 := by
  native_decide

example : solve 2 = 7 := by
  native_decide

example : solve 3 = 12 := by
  native_decide
end ProjectEulerSolutions.P25
