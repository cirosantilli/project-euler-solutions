import ProjectEulerStatements.P63
import ProjectEulerSolutions.Lib.PowerfulDigitCounts

namespace ProjectEulerSolutions.P63

/-- Scan one seed's powers, stopping as soon as its digit count falls behind. -/
def countPowers (base seed exponent : Nat) : Nat → Nat
  | 0 => 0
  | remaining + 1 =>
      if base ^ (exponent - 1) ≤ seed ^ exponent then
        1 + countPowers base seed (exponent + 1) remaining
      else 0

def solve (base : Nat) : Nat :=
  ((PowerfulDigits.seeds base).map fun seed =>
    countPowers base seed 1 (PowerfulDigits.bound base - 1)).sum

end ProjectEulerSolutions.P63
