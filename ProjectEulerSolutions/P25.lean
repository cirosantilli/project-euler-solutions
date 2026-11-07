import ProjectEulerStatements.P25
namespace ProjectEulerSolutions.P25

/-- Carry consecutive Fibonacci numbers instead of recomputing them at each index. -/
def search (digits index a b : Nat) : Nat → Nat
  | 0 => index
  | fuel + 1 =>
      if digits ≤ ProjectEulerStatements.P25.numDigits a then index
      else search digits (index+1) b (a+b) fuel

def solve (digits : Nat) : Nat :=
  search digits 0 0 1 (ProjectEulerStatements.P25.fibBound digits + 1)
end ProjectEulerSolutions.P25
