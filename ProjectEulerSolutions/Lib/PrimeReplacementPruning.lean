import ProjectEulerSolutions.Lib.PrimeReplacement
import Mathlib.Tactic.IntervalCases

/-! Necessary conditions for large prime-replacement families. -/
namespace ProjectEulerSolutions.PrimeReplacementPruning
open ProjectEulerStatements.P51

set_option Elab.async false

def candidate (n : Nat) (indices : List Nat) (digit : Nat) : Nat :=
  Nat.ofDigits 10 (replaceAt (Nat.digits 10 n) indices digit)

def eligible (n : Nat) (indices : List Nat) (digit : Nat) : Bool :=
  decide ((Nat.digits 10 (candidate n indices digit)).length = (Nat.digits 10 n).length) && isPrime (candidate n indices digit)

theorem familySize_eq_count (n : Nat) (indices : List Nat) :
    primeFamilySize n indices = ((List.range 10).filter (eligible n indices)).length := by
  have collect (digits : List Nat) :
      ((digits.filterMap (fun digit =>
        if (Nat.digits 10 (candidate n indices digit)).length = (Nat.digits 10 n).length
        then some (candidate n indices digit) else none)).filter isPrime).length =
      (digits.filter (eligible n indices)).length := by
    induction digits with
    | nil => rfl
    | cons digit tail ih =>
        cases prime : isPrime (candidate n indices digit) <;>
          by_cases width : (Nat.digits 10 (candidate n indices digit)).length = (Nat.digits 10 n).length <;>
          simp [eligible, width, prime, ih]
  exact collect (List.range 10)

private theorem filter_length_mono (left right : Nat → Bool) : ∀ values : List Nat,
    (∀ value ∈ values, left value = true → right value = true) →
      (values.filter left).length ≤ (values.filter right).length := by
  intro values
  induction values with
  | nil => simp
  | cons head tail ih =>
      intro included
      have rest := ih (fun value member => included value (by simp [member]))
      have current := included head (by simp)
      cases selected : left head <;> cases permitted : right head <;> simp_all
      omega

theorem familySize_le_count (n : Nat) (indices : List Nat) (keep : Nat → Bool)
    (included : ∀ digit < 10, eligible n indices digit = true → keep digit = true) :
    primeFamilySize n indices ≤ ((List.range 10).filter keep).length := by
  rw [familySize_eq_count]
  exact filter_length_mono (eligible n indices) keep (List.range 10)
    (fun digit member => included digit (List.mem_range.mp member))

/-- Replacing the units position with 0, 4, 6, or 8 cannot produce a prime. -/
theorem even_units_not_prime (n : Nat) (indices : List Nat) (digit : Nat) (units : 0 ∈ indices)
    (bad : digit ∈ [0, 4, 6, 8]) : ¬ Nat.Prime (candidate n indices digit) := by
  have composite : ∀ base : Nat, ¬ Nat.Prime (digit + 10 * base) := by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at bad
    rcases bad with rfl | rfl | rfl | rfl <;> intro base prime <;>
      have parity := prime.eq_two_or_odd <;> omega
  cases digits : Nat.digits 10 n with
  | nil => simpa only [candidate, digits, replaceAt, replaceAt.go] using Nat.not_prime_zero
  | cons head tail =>
      simpa only [candidate, digits, replaceAt, replaceAt.go, units, if_true, Nat.ofDigits, Nat.cast_id] using
        composite (Nat.ofDigits 10 (replaceAt.go indices digit tail 1))

/-- At least four replacement digits are composite, so a units-replacing family has at most six primes. -/
theorem familySize_units_le (n : Nat) (indices : List Nat) (units : 0 ∈ indices) : primeFamilySize n indices ≤ 6 := by
  have count := familySize_le_count n indices (fun digit => decide (digit ∉ [0, 4, 6, 8])) (by
    intro digit _ accepted
    simp only [eligible, Bool.and_eq_true, isPrime, decide_eq_true_eq] at accepted
    have excluded : digit ∉ [0, 4, 6, 8] := fun bad => even_units_not_prime n indices digit units bad accepted.2
    exact decide_eq_true excluded)
  have remaining : ((List.range 10).filter (fun digit => decide (digit ∉ [0, 4, 6, 8]))).length = 6 := by decide +kernel
  simpa only [remaining] using count

theorem large_family_excludes_units (n : Nat) (indices : List Nat) (large : 6 < primeFamilySize n indices) : 0 ∉ indices := by
  intro units
  have := familySize_units_le n indices units
  omega


private theorem powers_mod_three (positions : List Nat) :
    (positions.map (10 ^ ·)).sum % 3 = positions.length % 3 := by
  induction positions with
  | nil => rfl
  | cons head tail ih =>
      have power : 10 ^ head % 3 = 1 := by rw [Nat.pow_mod]; norm_num
      rw [List.map_cons, List.sum_cons, List.length_cons, Nat.add_mod, power, ih]
      omega

theorem weight_mod_three (indices : List Nat) (length : Nat) (distinct : indices.Nodup)
    (bounded : ∀ index ∈ indices, index < length) :
    PrimeReplacement.weight indices length % 3 = indices.length % 3 := by
  rw [PrimeReplacement.weight_eq_sum indices length distinct bounded]
  exact powers_mod_three indices

private theorem candidate_affine (n : Nat) (indices : List Nat) (original digit : Nat)
    (same : PrimeReplacement.SameDigit (Nat.digits 10 n) indices original) :
    candidate n indices digit = candidate n indices 0 + digit * PrimeReplacement.weight indices (Nat.digits 10 n).length := by
  have first := PrimeReplacement.shiftedValue_eq_replacement n indices original digit same
  have zero := PrimeReplacement.shiftedValue_eq_replacement n indices original 0 same
  dsimp only [PrimeReplacement.shiftedValue] at first zero
  simp only [Nat.cast_zero, zero_sub] at zero
  have linear : (candidate n indices digit : Int) = (candidate n indices 0 : Int) +
      (digit : Int) * PrimeReplacement.weight indices (Nat.digits 10 n).length := by
    dsimp only [candidate]
    nlinarith
  exact_mod_cast linear

private theorem zero_residue (offset step : Nat) (nonzero : step % 3 ≠ 0) :
    ∃ digit < 3, (offset + digit * step) % 3 = 0 := by
  have offsetCases : offset % 3 = 0 ∨ offset % 3 = 1 ∨ offset % 3 = 2 := by omega
  have stepCases : step % 3 = 1 ∨ step % 3 = 2 := by omega
  rcases offsetCases with zero | one | two
  · exact ⟨0, by decide, by simpa only [zero_mul, add_zero] using zero⟩
  · rcases stepCases with first | second
    · exact ⟨2, by decide, by omega⟩
    · exact ⟨1, by decide, by omega⟩
  · rcases stepCases with first | second
    · exact ⟨1, by decide, by omega⟩
    · exact ⟨2, by decide, by omega⟩

private theorem progression_mod (offset step digit : Nat) (zero : (offset + digit * step) % 3 = 0) :
    ∀ value ∈ [digit, digit + 3, digit + 6], (offset + value * step) % 3 = 0 := by
  intro value member
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · exact zero
  · have expanded : offset + (digit + 3) * step = (offset + digit * step) + step * 3 := by ring
    rw [expanded, Nat.add_mul_mod_self_right]
    exact zero
  · have expanded : offset + (digit + 6) * step = (offset + digit * step) + (2 * step) * 3 := by ring
    rw [expanded, Nat.add_mul_mod_self_right]
    exact zero

private theorem wide_multiple_three_not_prime (value : Nat) (wide : 2 ≤ (Nat.digits 10 value).length)
    (divisible : value % 3 = 0) : ¬ Nat.Prime value := by
  have lower : 10 ≤ value := (Nat.lt_digits_length_iff (b := 10) (k := 1) (by decide) value).mp (by omega)
  intro prime
  have divisor := (Nat.dvd_prime prime).mp (Nat.dvd_of_mod_eq_zero divisible)
  omega

/-- For multidigit candidates, a nonzero weight modulo three excludes at least three replacement digits from the prime family. -/
theorem familySize_nonzero_weight_le (n : Nat) (indices : List Nat) (original : Nat)
    (wide : 2 ≤ (Nat.digits 10 n).length)
    (same : PrimeReplacement.SameDigit (Nat.digits 10 n) indices original)
    (nonzero : PrimeReplacement.weight indices (Nat.digits 10 n).length % 3 ≠ 0) : primeFamilySize n indices ≤ 7 := by
  obtain ⟨digit, digitBound, zero⟩ := zero_residue (candidate n indices 0)
    (PrimeReplacement.weight indices (Nat.digits 10 n).length) nonzero
  have excluded : ∀ value ∈ [digit, digit + 3, digit + 6], eligible n indices value ≠ true := by
    intro value member accepted
    simp only [eligible, Bool.and_eq_true, isPrime, decide_eq_true_eq] at accepted
    have divisible : candidate n indices value % 3 = 0 := by
      rw [candidate_affine n indices original value same]
      exact progression_mod _ _ digit zero value member
    exact wide_multiple_three_not_prime (candidate n indices value) (by rw [accepted.1]; exact wide) divisible accepted.2
  have count := familySize_le_count n indices (fun value => decide (value ∉ [digit, digit + 3, digit + 6])) (by
    intro value _ accepted
    exact decide_eq_true (fun member => excluded value member accepted))
  have remaining : ((List.range 10).filter (fun value => decide (value ∉ [digit, digit + 3, digit + 6]))).length = 7 := by
    interval_cases digit <;> decide +kernel
  simpa only [remaining] using count

/-- A nonempty in-range selection that omits units requires at least two decimal digits. -/
theorem wide_of_no_units (n : Nat) (indices : List Nat) (nonempty : indices ≠ [])
    (bounded : ∀ index ∈ indices, index < (Nat.digits 10 n).length) (units : 0 ∉ indices) : 2 ≤ (Nat.digits 10 n).length := by
  obtain ⟨head, tail, rfl⟩ := List.exists_cons_of_ne_nil nonempty
  have inside := bounded head (by simp)
  have nonzero : head ≠ 0 := fun zero => units (by simp [zero])
  omega

/-- Every qualifying family of at least eight primes replaces a multiple of three positions. -/
theorem large_family_count_multiple_three (n : Nat) (indices : List Nat) (distinct : indices.Nodup)
    (nonempty : indices ≠ []) (bounded : ∀ index ∈ indices, index < (Nat.digits 10 n).length)
    (member : n ∈ familyNumbers n indices) (large : 7 < primeFamilySize n indices) : indices.length % 3 = 0 := by
  have units := large_family_excludes_units n indices (by omega)
  have wide := wide_of_no_units n indices nonempty bounded units
  obtain ⟨original, _, same⟩ := (PrimeReplacement.family_member_iff n indices).mp member
  by_contra nonzero
  have weight : PrimeReplacement.weight indices (Nat.digits 10 n).length % 3 ≠ 0 := by
    rwa [weight_mod_three indices _ distinct bounded]
  have upper := familySize_nonzero_weight_le n indices original wide same weight
  omega

theorem large_family_min_positions (n : Nat) (indices : List Nat) (distinct : indices.Nodup)
    (nonempty : indices ≠ []) (bounded : ∀ index ∈ indices, index < (Nat.digits 10 n).length)
    (member : n ∈ familyNumbers n indices) (large : 7 < primeFamilySize n indices) : 3 ≤ indices.length := by
  have multiple := large_family_count_multiple_three n indices distinct nonempty bounded member large
  have positive := List.length_pos_iff.mpr nonempty
  omega

end ProjectEulerSolutions.PrimeReplacementPruning
