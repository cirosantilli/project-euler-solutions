import ProjectEulerStatements.P12
import ProjectEulerSolutions.Termination.P12
namespace ProjectEulerSolutions.P12

def stripFactor (x p exp : Nat) : Nat × Nat :=
  if x % p == 0 then
    stripFactor (x / p) p (exp + 1)
  else
    (x, exp)
termination_by 0
decreasing_by all_goals exact Termination.decreases

def divisorCount (n : Nat) : Nat :=
  if n <= 1 then
    1
  else
    let rec loop (x i total : Nat) : Nat :=
      if i * i > x then
        if x > 1 then total * 2 else total
      else if x % i == 0 then
        let (x', exp) := stripFactor x i 0
        loop x' (i + 1) (total * (exp + 1))
      else
        loop x (i + 1) total
    termination_by 0
    decreasing_by all_goals exact Termination.decreases
    loop n 2 1

def solve (k : Nat) : Nat :=
  let rec loop (n : Nat) : Nat :=
    let (a, b) :=
      if n % 2 == 0 then
        (n / 2, n + 1)
      else
        (n, (n + 1) / 2)
    let d := divisorCount a * divisorCount b
    if d > k then
      n * (n + 1) / 2
    else
      loop (n + 1)
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loop 1

example : solve 5 = 28 := by
  native_decide
end ProjectEulerSolutions.P12
