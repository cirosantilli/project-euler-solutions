import ProjectEulerStatements.P2

namespace ProjectEulerSolutions.P2

partial def solve (limit : Nat) : Nat :=
  let rec go (a b total : Nat) : Nat :=
    if a ≤ limit then
      let total' := if a % 2 = 0 then total + a else total
      go b (a + b) total'
    else
      total
  go 1 2 0

example : solve 4000000 = 4613732 := by native_decide
theorem equiv (n : Nat) : ProjectEulerStatements.P2.naive n = solve n := sorry

end ProjectEulerSolutions.P2
