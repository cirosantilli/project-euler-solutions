import Mathlib.NumberTheory.LucasPrimality
import Mathlib.Data.List.Prime
import lang.py.builtin.type.int

/-! A shared Lucas/Pratt certificate DAG interface.
The producer supplies only factors and residues; all primality follows from
kernel-checked factor proofs and the universal modular-power correctness theorem. -/
namespace ProjectEulerSolutions.PrattCertificates

open FormalSW.Py.Builtin.Int (powModBinary powModBinary_correct)

/-- Binary modular evaluation agrees with exponentiation in the quotient ring. -/
theorem binary_cast (base : Int) (exponent modulus : Nat) :
    ((powModBinary base exponent modulus : Int) : ZMod modulus) =
      (base : ZMod modulus) ^ exponent := by
  rw [powModBinary_correct, ZMod.intCast_mod]
  exact map_pow (Int.castRingHom (ZMod modulus)) base exponent

theorem binary_mod (base : Int) (exponent modulus : Nat) :
    powModBinary base exponent modulus % (modulus : Int) = powModBinary base exponent modulus := by
  rw [powModBinary_correct]
  exact Int.emod_emod _ _

theorem power_eq_one {base : Int} {exponent modulus : Nat} (computed : powModBinary base exponent modulus = 1) :
    (base : ZMod modulus) ^ exponent = 1 := by
  have cast : ((powModBinary base exponent modulus : Int) : ZMod modulus) = ((1 : Int) : ZMod modulus) :=
    congrArg (fun value : Int => (value : ZMod modulus)) computed
  rw [binary_cast] at cast
  simpa only [Int.cast_one] using cast

/-- Canonical residues and modulus>1 make inequality to 1 sound in ZMod. -/
theorem power_ne_one {base : Int} {exponent modulus : Nat} (modulusBound : 2 ≤ modulus)
    (computed : powModBinary base exponent modulus ≠ 1) :
    (base : ZMod modulus) ^ exponent ≠ 1 := by
  intro same
  have cast : ((powModBinary base exponent modulus : Int) : ZMod modulus) = 1 :=
    (binary_cast base exponent modulus).trans same
  have residues := (ZMod.intCast_eq_intCast_iff' (powModBinary base exponent modulus) 1 modulus).mp
    (by simpa only [Int.cast_one] using cast)
  have oneMod : (1 : Int) % (modulus : Int) = 1 :=
    Int.emod_eq_of_lt (by decide) (by omega)
  rw [binary_mod, oneMod] at residues
  exact computed residues

/-- A complete prime factorization supplies every prime divisor demanded by Lucas. -/
theorem prime_of_factorization {modulus : Nat} {base : Int} {factors : List Nat}
    (modulusBound : 2 ≤ modulus)
    (factorization : factors.prod = modulus - 1)
    (factorPrimes : ∀ factor ∈ factors, Nat.Prime factor)
    (fullPower : powModBinary base (modulus - 1) modulus = 1)
    (properPowers : ∀ factor ∈ factors,
      powModBinary base ((modulus - 1) / factor) modulus ≠ 1) :
    Nat.Prime modulus := by
  apply lucas_primality modulus (base : ZMod modulus)
  · exact power_eq_one fullPower
  · intro factor prime divides
    have inProduct : factor ∣ factors.prod := by
      rw [factorization]
      exact divides
    have member : factor ∈ factors := mem_list_primes_of_dvd_prod
      (Nat.prime_iff.mp prime) (fun value present => Nat.prime_iff.mp (factorPrimes value present)) inProduct
    exact power_ne_one modulusBound (properPowers factor member)

/-- No primality decision is executed here: factor primality is supplied separately.
Repeated factors remain in the product, while modular powers are checked only once. -/
def Checks (modulus : Nat) (base : Int) (factors : List Nat) : Prop :=
  2 ≤ modulus ∧ factors.prod = modulus - 1 ∧
    powModBinary base (modulus - 1) modulus = 1 ∧
    ∀ factor ∈ factors.eraseDups,
      powModBinary base ((modulus - 1) / factor) modulus ≠ 1

instance (modulus : Nat) (base : Int) (factors : List Nat) : Decidable (Checks modulus base factors) := by
  unfold Checks
  infer_instance

def checkNode (modulus : Nat) (base : Int) (factors : List Nat) : Bool :=
  decide (Checks modulus base factors)

theorem checkNode_sound {modulus : Nat} {base : Int} {factors : List Nat}
    (factorPrimes : ∀ factor ∈ factors, Nat.Prime factor)
    (checked : checkNode modulus base factors = true) : Nat.Prime modulus := by
  have checks : Checks modulus base factors := of_decide_eq_true checked
  obtain ⟨bound, factorization, fullPower, properPowers⟩ := checks
  apply prime_of_factorization bound factorization factorPrimes fullPower
  intro factor present
  exact properPowers factor (List.mem_eraseDups.mpr present)

/-- A previously certified DAG node. Referring to it never repeats a primality test. -/
structure CertifiedPrime where
  value : Nat
  prime : Nat.Prime value

def factorValues (factors : List CertifiedPrime) : List Nat := factors.map CertifiedPrime.value

theorem factorValues_prime (factors : List CertifiedPrime) :
    ∀ value ∈ factorValues factors, Nat.Prime value := by
  intro value present
  change value ∈ factors.map CertifiedPrime.value at present
  obtain ⟨factor, _, rfl⟩ := List.mem_map.mp present
  exact factor.prime

/-- The generator creates one named node per distinct prime, and factors point to
already named nodes. `by decide` or `rfl` proves only the small arithmetic checks. -/
def certify (modulus : Nat) (base : Int) (factors : List CertifiedPrime)
    (checked : checkNode modulus base (factorValues factors) = true) : CertifiedPrime :=
  ⟨modulus, checkNode_sound (factorValues_prime factors) checked⟩

/-- Convenient interface when the producer keeps prime theorems and data separate. -/
theorem certify_prime (modulus : Nat) (base : Int) (factors : List CertifiedPrime)
    (checked : checkNode modulus base (factorValues factors) = true) : Nat.Prime modulus :=
  (certify modulus base factors checked).prime

/-- Factor witnesses for composite rows need no trial-division certificate. -/
theorem composite_of_product {factor cofactor number : Nat} (product : factor * cofactor = number)
    (factorBound : 2 ≤ factor) (cofactorBound : 2 ≤ cofactor) : ¬ Nat.Prime number :=
  Nat.not_prime_of_mul_eq product (by omega) (by omega)

end ProjectEulerSolutions.PrattCertificates
