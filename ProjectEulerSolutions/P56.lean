import ProjectEulerSolutions.Lib.PowerDigitSum

namespace ProjectEulerSolutions.P56

/-- Both Python loops visit all bases and exponents below the limit, including zero. -/
def solve (n : Nat) : Nat := (PowerDigitSum.maximum n).toNat

end ProjectEulerSolutions.P56
