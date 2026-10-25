import lang.py.FormalSW.Calls
import Mathlib.Data.Int.ModEq
import Mathlib.Data.List.Range
import Mathlib.Algebra.BigOperators.Group.List.Basic
import Mathlib.Tactic.Ring
import Lean.Elab.Tactic.Omega

/-! Exact signed remainders, repeated multiplication, and the self-power series. -/
namespace ProjectEulerSolutions.SelfPowers

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
      | zero => simp [multiplyLoop, remainder_positive _ _ positiveModulus]
      | succ count =>
          rw [ih (by omega), remainder_positive _ _ positiveModulus]
          rw [Int.mul_emod, Int.emod_emod, ← Int.mul_emod]
          simp only [pow_succ', mul_assoc]

/-- The reference's linear modular exponentiation agrees with builtin pow for positive exponents. -/
theorem power_positive (base : Int) (exponent : Nat) (positive : 0<exponent)
    (modulus : Int) (positiveModulus : 0<modulus) :
    power base exponent modulus=base^exponent%modulus := by
  simp only [power, Int.toNat_natCast, multiplyLoop_positive _ _ positiveModulus exponent positive,
    one_mul, remainder_positive _ _ positiveModulus]
  have congruent : Int.ModEq modulus (base%modulus) base := by simp [Int.ModEq]
  exact congruent.pow exponent

def selfPower (index : Nat) (modulus : Int) : Int :=
  power ((index+1 : Nat) : Int) ((index+1 : Nat) : Int) modulus

theorem selfPower_exact (index : Nat) (modulus : Int) (positive : 0<modulus) :
    selfPower index modulus=(((index+1 : Nat) : Int)^(index+1))%modulus :=
  power_positive _ (index+1) (by omega) modulus positive

def series (first count : Nat) (modulus initial : Int) : Int :=
  (List.range' first count).foldl (fun total (i : Nat) =>
    remainder (total+selfPower i modulus) modulus) initial

theorem series_zero (first : Nat) (modulus initial : Int) : series first 0 modulus initial=initial := rfl

theorem series_succ (first count : Nat) (modulus initial : Int) :
    series first (count+1) modulus initial=
      series (first+1) count modulus (remainder (initial+selfPower first modulus) modulus) := by
  simp only [series, List.range'_succ, List.foldl_cons]

/-- A reduced initial accumulator and incremental modular additions give the exact self-power sum. -/
theorem series_exact (first count : Nat) (modulus initial : Int) (positive : 0<modulus)
    (reduced : initial%modulus=initial) :
    series first count modulus initial=
      (initial+((List.range' first count).map (fun i => ((i+1 : Nat) : Int)^(i+1))).sum)%modulus := by
  induction count generalizing first initial with
  | zero => simpa only [series_zero, List.range'_zero, List.map_nil, List.sum_nil, add_zero] using reduced.symm
  | succ count ih =>
      rw [series_succ, remainder_positive _ _ positive, selfPower_exact first modulus positive]
      rw [ih (first+1) _ (Int.emod_emod _ _)]
      rw [List.range'_succ, List.map_cons, List.sum_cons]
      have addRight (x y : Int) : (x+y%modulus)%modulus=(x+y)%modulus := by
        rw [Int.add_emod, Int.emod_emod, ← Int.add_emod]
      have addLeft (x y : Int) : (x%modulus+y)%modulus=(x+y)%modulus := by
        rw [Int.add_emod, Int.emod_emod, ← Int.add_emod]
      rw [addRight, addLeft]
      congr 1
      ring

end ProjectEulerSolutions.SelfPowers
