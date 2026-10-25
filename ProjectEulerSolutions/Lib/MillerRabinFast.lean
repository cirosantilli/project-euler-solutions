import ProjectEulerSolutions.Lib.MillerRabin
import Mathlib.Tactic.Ring

namespace ProjectEulerSolutions.MillerRabin.Fast

/-- An accumulator keeps the recursive modular-power result out of both
operands of a square during kernel reduction. -/
def powerAux (modulus : Nat) : Nat → Int → Nat → Int → Int
  | 0, _, _, result => result % (modulus : Int)
  | fuel + 1, base, exponent, result =>
      if exponent = 0 then result % (modulus : Int)
      else powerAux modulus fuel (base * base % (modulus : Int)) (exponent / 2)
        (if exponent % 2 = 0 then result else result * base % (modulus : Int))

private theorem power_remainder (base modulus : Int) (exponent : Nat) :
    (base % modulus) ^ exponent % modulus = base ^ exponent % modulus := by
  induction exponent with
  | zero => simp
  | succ exponent ih =>
    simp only [Int.pow_succ, Int.mul_emod, Int.emod_emod, ih]

theorem powerAux_correct (modulus fuel : Nat) (base : Int) (exponent : Nat) (result : Int)
    (bound : exponent ≤ fuel) :
    powerAux modulus fuel base exponent result = result * base ^ exponent % (modulus : Int) := by
  induction fuel generalizing base exponent result with
  | zero =>
    have zero : exponent = 0 := by omega
    subst exponent
    simp [powerAux]
  | succ fuel ih =>
    by_cases zero : exponent = 0
    · subst exponent
      simp [powerAux]
    · rw [powerAux, if_neg zero]
      have smaller : exponent / 2 ≤ fuel := by
        have := Nat.div_lt_self (Nat.pos_of_ne_zero zero) (by decide : 1 < 2)
        omega
      rw [ih _ _ _ smaller]
      have remainder := Nat.mod_lt exponent (by decide : 0 < 2)
      have division := Nat.mod_add_div exponent 2
      by_cases even : exponent % 2 = 0
      · rw [if_pos even]
        have decomposition : exponent = exponent / 2 + exponent / 2 := by omega
        have power : base ^ exponent = (base * base) ^ (exponent / 2) := by
          calc
            _ = base ^ (exponent / 2 + exponent / 2) := congrArg (base ^ ·) decomposition
            _ = _ := by rw [Int.pow_add, mul_pow]
        rw [Int.mul_emod, power_remainder, ← Int.mul_emod, power]
      · rw [if_neg even]
        have decomposition : exponent = exponent / 2 + exponent / 2 + 1 := by omega
        have power : base ^ exponent = base * (base * base) ^ (exponent / 2) := by
          calc
            _ = base ^ (exponent / 2 + exponent / 2 + 1) := congrArg (base ^ ·) decomposition
            _ = _ := by rw [Int.pow_succ, Int.pow_add, mul_pow]; ring
        rw [Int.mul_emod, Int.emod_emod, power_remainder, ← Int.mul_emod, power, mul_assoc]

def power (base : Int) (exponent modulus : Nat) : Int :=
  powerAux modulus exponent base exponent 1

theorem power_eq_source (base : Int) (exponent modulus : Nat) :
    power base exponent modulus = FormalSW.Py.Builtin.Int.powModBinary base exponent modulus := by
  rw [FormalSW.Py.Builtin.Int.powModBinary_correct, power, powerAux_correct _ _ _ _ _ (Nat.le_refl _)]
  simp

/-- The nonnegative arithmetic path avoids intermediate integer constructors. -/
def naturalPowerAux (modulus : Nat) : Nat → Nat → Nat → Nat → Nat
  | 0, _, _, result => result % modulus
  | fuel + 1, base, exponent, result =>
      if exponent = 0 then result % modulus
      else naturalPowerAux modulus fuel (base * base % modulus) (exponent / 2)
        (if exponent % 2 = 0 then result else result * base % modulus)

private theorem naturalPowerAux_cast (modulus fuel base exponent result : Nat) :
    (naturalPowerAux modulus fuel base exponent result : Int) =
      powerAux modulus fuel base exponent result := by
  induction fuel generalizing base exponent result with
  | zero => rfl
  | succ fuel ih =>
    rw [naturalPowerAux, powerAux]
    by_cases zero : exponent = 0
    · simp only [if_pos zero, Int.natCast_emod]
    · simp only [if_neg zero]
      by_cases even : exponent % 2 = 0
      · simpa only [if_pos even, Int.natCast_emod, Int.natCast_mul] using
          ih (base * base % modulus) (exponent / 2) result
      · simpa only [if_neg even, Int.natCast_emod, Int.natCast_mul] using
          ih (base * base % modulus) (exponent / 2) (result * base % modulus)

def naturalPower (base : Int) (exponent modulus : Nat) : Int :=
  if 0 ≤ base then naturalPowerAux modulus exponent base.toNat exponent 1
  else power base exponent modulus

theorem naturalPower_eq_source (base : Int) (exponent modulus : Nat) :
    naturalPower base exponent modulus = FormalSW.Py.Builtin.Int.powModBinary base exponent modulus := by
  by_cases nonnegative : 0 ≤ base
  · simp only [naturalPower, if_pos nonnegative, naturalPowerAux_cast, Int.toNat_of_nonneg nonnegative]
    exact power_eq_source base exponent modulus
  · simp only [naturalPower, if_neg nonnegative, power_eq_source]

/-- The same Miller–Rabin round, using the proved accumulator evaluator. -/
def check (a : Int) (n d s : Nat) : Bool :=
  if a % (n : Int) = 0 then true else
    let x := naturalPower a d n
    if x = 1 ∨ x = (n : Int) - 1 then true else squareSearch n (s - 1) x

theorem check_eq_source (a : Int) (n d s : Nat) : check a n d s = millerCheck a n d s := by
  simp only [check, millerCheck, naturalPower_eq_source]

def naturalScan (n : Nat) : List Nat → Option Bool
  | [] => none
  | p :: ps => if n = p then some true else if n % p = 0 then some false else naturalScan n ps

theorem naturalScan_eq_source (n : Nat) (ps : List Nat) : naturalScan n ps = smallScan n ps := by
  induction ps with
  | nil => rfl
  | cons p ps ih =>
    have zero : ((n % p : Nat) : Int) = 0 ↔ n % p = 0 := by omega
    simp only [naturalScan, smallScan, ← Int.natCast_emod, Int.natCast_inj, zero, ih]

/-- This remains an exact algorithmic model, without assuming primality. -/
def model (n : Nat) : Bool :=
  if n < 2 then false else
    match naturalScan n smallPrimes with
    | some answer => answer
    | none =>
        let ds := decompose (n - 1)
        bases.all (fun a => check a n ds.1 ds.2)

theorem model_eq_source (n : Nat) : model n = isPrimeModel n := by
  simp only [model, isPrimeModel, naturalScan_eq_source, check_eq_source]
  rfl

end ProjectEulerSolutions.MillerRabin.Fast
