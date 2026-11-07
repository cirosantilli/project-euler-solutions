import lang.py.FormalSW.Calls

/-! Repeated modular multiplication and its agreement with exact integer powers. -/
namespace ProjectEulerSolutions.ModularPower

set_option Elab.async false

/-- Python's signed remainder, used only with a nonzero divisor. -/
def remainder (value modulus : Int) : Int :=
  if 0<modulus then value%modulus else value-modulus*((-value)/(-modulus))

theorem remainder_positive (value modulus : Int) (positive : 0<modulus) :
    remainder value modulus=value%modulus := by simp [remainder, positive]

def multiplyLoop (base modulus : Int) : Nat → Int → Int
  | 0, initial => initial
  | count+1, initial => multiplyLoop base modulus count (remainder (initial*base) modulus)

def power (base exponent modulus : Int) : Int :=
  multiplyLoop (remainder base modulus) modulus exponent.toNat 1

/-- After at least one multiplication, modular reduction agrees with the exact power. -/
theorem multiplyLoop_positive (base modulus : Int) (positiveModulus : 0<modulus)
    (count : Nat) (positive : 0<count) (initial : Int) :
    multiplyLoop base modulus count initial=(initial*base^count)%modulus := by
  induction count generalizing initial with
  | zero => omega
  | succ count ih =>
      rw [multiplyLoop]
      cases count with
      | zero => simp [multiplyLoop, remainder_positive _ _ positiveModulus, Int.pow_one]
      | succ count =>
          rw [ih (by omega), remainder_positive _ _ positiveModulus]
          rw [Int.mul_emod, Int.emod_emod, ← Int.mul_emod]
          simp only [Int.pow_succ', Int.mul_assoc]

/-- The reference's linear modular exponentiation agrees with builtin pow for positive exponents. -/
theorem power_positive (base : Int) (exponent : Nat) (positive : 0<exponent)
    (modulus : Int) (positiveModulus : 0<modulus) :
    power base exponent modulus=base^exponent%modulus := by
  simp only [power, Int.toNat_natCast, multiplyLoop_positive _ _ positiveModulus exponent positive,
    Int.one_mul, remainder_positive _ _ positiveModulus]
  have congruent (n : Nat) : (base % modulus)^n % modulus = base^n % modulus := by
    induction n with
    | zero => simp only [Int.pow_zero]
    | succ n ih =>
        simp only [Int.pow_succ]
        rw [Int.mul_emod, Int.emod_emod, ih, ← Int.mul_emod]
  exact congruent exponent

end ProjectEulerSolutions.ModularPower
