import ProjectEulerStatements.P2

namespace ProjectEulerSolutions.P2

def solve (limit : Nat) : Nat :=
  let rec go (a b total : Nat) (ha : 0 < a) (hab : a < b) : Nat :=
    if a ≤ limit then
      let total' := if a % 2 = 0 then total + a else total
      go b (a + b) total' (by omega) (by omega)
    else
      total
  termination_by limit + 1 - a
  decreasing_by
    omega
  go 1 2 0 (by omega) (by omega)

example : solve 4000000 = 4613732 := by native_decide
end ProjectEulerSolutions.P2
