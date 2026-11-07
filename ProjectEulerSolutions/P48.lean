import ProjectEulerStatements.P48

namespace ProjectEulerSolutions.P48

/-- Modular powers and the finite accumulator used by the Python solution. -/
def powMod (a b modulus : Nat) : Nat := a ^ b % modulus

def selfPowersMod (n modulus : Nat) : Nat :=
  (List.range n).foldl (fun total i => (total + powMod (i + 1) (i + 1) modulus) % modulus) 0

def lastKDigitsSelfPowers (n k : Nat) : String :=
  let digits := toString (selfPowersMod n (10 ^ k))
  String.ofList (List.replicate (k - digits.length) '0') ++ digits

def exactSelfPowersSum (n : Nat) : Nat :=
  (List.range n).foldl (fun total i => total + (i + 1) ^ (i + 1)) 0

def solve (n : Nat) : Nat := selfPowersMod n (10 ^ 10)

end ProjectEulerSolutions.P48
