import ProjectEulerStatements.P2

namespace ProjectEulerSolvers.P2

partial def sol (limit : Nat) : Nat :=
  let rec go (a b total : Nat) : Nat :=
    if a ≤ limit then
      let total' := if a % 2 = 0 then total + a else total
      go b (a + b) total'
    else
      total
  go 1 2 0

theorem sol_4000000 : sol 4000000 = 4613732 := by native_decide
theorem equiv (n : Nat) : ProjectEulerStatements.P2.naive n = sol n := sorry

end ProjectEulerSolvers.P2
open ProjectEulerSolvers.P2

def main : IO Unit := do
  IO.println (sol 4000000)
