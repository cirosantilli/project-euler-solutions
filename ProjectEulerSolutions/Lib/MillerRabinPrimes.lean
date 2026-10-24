import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.PrattCertificates
import Mathlib.FieldTheory.Finite.Basic

namespace ProjectEulerSolutions.MillerRabin

private theorem residue_eq_of_cast (n : Nat) (x target : Int)
    (lower : 0 ≤ target) (upper : target < n) (same : (x : ZMod n) = (target : ZMod n)) :
    x % (n : Int) = target := by
  have residues := (ZMod.intCast_eq_intCast_iff' x target n).mp same
  simpa only [Int.emod_eq_of_lt lower upper] using residues

private theorem power_two_chain {R : Type*} [CommRing R] [NoZeroDivisors R]
    (x : R) (s : Nat) (power : x ^ (2 ^ s) = 1) :
    x = 1 ∨ ∃ i < s, x ^ (2 ^ i) = -1 := by
  induction s with
  | zero => exact Or.inl (by simpa using power)
  | succ s ih =>
    have squared : (x ^ (2 ^ s)) ^ 2 = 1 := by
      rw [← pow_mul, ← Nat.pow_succ]
      exact power
    rcases sq_eq_one_iff.mp squared with earlier | last
    · rcases ih earlier with one | ⟨i, inside, hit⟩
      · exact Or.inl one
      · exact Or.inr ⟨i, by omega, hit⟩
    · exact Or.inr ⟨s, by omega, last⟩

private theorem squareSearch_hit (n : Nat) (large : 2 ≤ n) (fuel : Nat) (x : Int)
    (i : Nat) (inside : i < fuel) (hit : (x : ZMod n) ^ (2 ^ (i + 1)) = -1) :
    squareSearch n fuel x = true := by
  induction fuel generalizing x i with
  | zero => omega
  | succ fuel ih =>
    rw [squareSearch]
    by_cases found : x * x % (n : Int) = (n : Int) - 1
    · simp only [found, ↓reduceIte]
    · simp only [found, ↓reduceIte]
      cases i with
      | zero =>
        have nextCast : ((x * x % (n : Int) : Int) : ZMod n) = (((n : Int) - 1 : Int) : ZMod n) := by
          simpa [ZMod.intCast_mod, pow_two] using hit
        have residue := residue_eq_of_cast n (x * x % (n : Int)) ((n : Int) - 1)
          (by omega) (by omega) nextCast
        have same : x * x % (n : Int) = (n : Int) - 1 := by
          simpa only [Int.emod_emod] using residue
        exact False.elim (found same)
      | succ i =>
        apply ih (x * x % (n : Int)) i (by omega)
        calc
          (((x * x % (n : Int) : Int) : ZMod n)) ^ (2 ^ (i + 1)) =
              ((x : ZMod n) ^ 2) ^ (2 ^ (i + 1)) := by simp [ZMod.intCast_mod, pow_two]
          _ = (x : ZMod n) ^ (2 ^ (i + 1 + 1)) := by
            rw [← pow_mul, Nat.pow_succ]
            congr 1
            omega
          _ = -1 := hit

/-- Every prime passes each strong probable-prime check. This has no
machine-word bound and makes no converse claim about composite inputs. -/
theorem millerCheck_of_prime (n d s : Nat) (a : Int) (prime : Nat.Prime n)
    (decomposition : d * 2 ^ s = n - 1) : millerCheck a n d s = true := by
  letI : Fact (Nat.Prime n) := ⟨prime⟩
  unfold millerCheck
  by_cases divides : a % (n : Int) = 0
  · simp only [divides, ↓reduceIte]
  · simp only [divides, ↓reduceIte]
    let x := FormalSW.Py.Builtin.Int.powModBinary a d n
    have canonical : x % (n : Int) = x := PrattCertificates.binary_mod a d n
    have nonzero : (a : ZMod n) ≠ 0 := by
      intro zero
      have residue := residue_eq_of_cast n a 0 (by omega) (by have := prime.two_le; omega)
        (by simpa using zero)
      exact divides residue
    have power : (x : ZMod n) ^ (2 ^ s) = 1 := by
      rw [PrattCertificates.binary_cast, ← pow_mul, decomposition]
      exact ZMod.pow_card_sub_one_eq_one nonzero
    have one_of_cast (same : (x : ZMod n) = 1) : x = 1 := by
      have residue := residue_eq_of_cast n x 1 (by omega) (by have := prime.two_le; omega)
        (by simpa using same)
      simpa only [canonical] using residue
    have last_of_cast (same : (x : ZMod n) = -1) : x = (n : Int) - 1 := by
      have residue := residue_eq_of_cast n x ((n : Int) - 1)
        (by have := prime.two_le; omega) (by omega) (by simpa using same)
      simpa only [canonical] using residue
    by_cases initial : x = 1 ∨ x = (n : Int) - 1
    · simp only [show FormalSW.Py.Builtin.Int.powModBinary a d n = x from rfl, initial, ↓reduceIte]
    · simp only [show FormalSW.Py.Builtin.Int.powModBinary a d n = x from rfl, initial, ↓reduceIte]
      rcases power_two_chain (x : ZMod n) s power with one | ⟨i, inside, hit⟩
      · exact False.elim (initial (Or.inl (one_of_cast one)))
      · cases i with
        | zero =>
          have minus : (x : ZMod n) = -1 := by simpa using hit
          exact False.elim (initial (Or.inr (last_of_cast minus)))
        | succ i => exact squareSearch_hit n prime.two_le (s - 1) x i (by omega) hit


private theorem smallScan_ne_false_of_prime (n : Nat) (prime : Nat.Prime n) (values : List Nat)
    (bounds : ∀ value ∈ values, 2 ≤ value) : smallScan n values ≠ some false := by
  induction values with
  | nil => simp [smallScan]
  | cons value rest ih =>
    by_cases same : (n : Int) = (value : Int)
    · simp [smallScan, same]
    · by_cases divides : (n : Int) % (value : Int) = 0
      · have divisor : value ∣ n := by exact_mod_cast Int.dvd_of_emod_eq_zero divides
        rcases (Nat.dvd_prime prime).mp divisor with one | equal
        · have bound := bounds value (by simp)
          omega
        · have : (n : Int) = (value : Int) := by omega
          exact False.elim (same this)
      · simpa only [smallScan, same, divides, ↓reduceIte] using
          ih (fun p member => bounds p (List.mem_cons_of_mem value member))

/-- The exact source model has no false negatives, for primes of any size.
Its possible false positives remain a separate obligation. -/
theorem isPrimeModel_of_prime (n : Nat) (prime : Nat.Prime n) : isPrimeModel n = true := by
  have large := prime.two_le
  have safe : smallScan n smallPrimes ≠ some false :=
    smallScan_ne_false_of_prime n prime smallPrimes (by
      intro p member
      simp only [smallPrimes, List.mem_cons, List.not_mem_nil, or_false] at member
      rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide)
  unfold isPrimeModel
  rw [if_neg (by omega : ¬ n < 2)]
  cases scan : smallScan n smallPrimes with
  | none =>
    apply List.all_eq_true.mpr
    intro a _
    exact millerCheck_of_prime n (decompose (n - 1)).1 (decompose (n - 1)).2 a prime
      (decompose_spec (n - 1) (by omega)).2.2
  | some answer =>
    cases answer with
    | false => exact False.elim (safe scan)
    | true => rfl

theorem not_prime_of_rejected (n : Nat) (rejected : isPrimeModel n = false) : ¬ Nat.Prime n := by
  intro prime
  have accepted := isPrimeModel_of_prime n prime
  rw [rejected] at accepted
  contradiction

end ProjectEulerSolutions.MillerRabin
