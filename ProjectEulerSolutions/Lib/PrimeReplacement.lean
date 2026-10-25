import ProjectEulerStatements.P51
import Mathlib.Data.Nat.Digits.Lemmas
import Mathlib.Data.List.Range
import Mathlib.Data.List.Perm.Basic
import Mathlib.Data.List.Nodup
import Mathlib.Tactic.Ring

/-! Digit replacement and positional arithmetic shared by the problem 51 searches. -/
namespace ProjectEulerSolutions.PrimeReplacement
open ProjectEulerStatements.P51

set_option Elab.async false

/-- The selected positions within a digit list contribute these decimal place values. -/
def weight (indices : List Nat) (length : Nat) (start : Nat := 0) : Nat :=
  (((List.range' start length).filter (· ∈ indices)).map (10 ^ ·)).sum

@[simp] theorem weight_zero (indices : List Nat) (start : Nat) : weight indices 0 start = 0 := by
  simp [weight]

theorem weight_succ (indices : List Nat) (length start : Nat) :
    weight indices (length + 1) start =
      (if start ∈ indices then 10 ^ start else 0) + weight indices length (start + 1) := by
  by_cases member : start ∈ indices <;> simp [weight, List.range'_succ, member]

private theorem replace_length (items indices : List Nat) (digit start : Nat) :
    (replaceAt.go indices digit items start).length = items.length := by
  induction items generalizing start with
  | nil => rfl
  | cons head tail ih => simp [replaceAt.go, ih]

@[simp] theorem replaceAt_length (items indices : List Nat) (digit : Nat) :
    (replaceAt items indices digit).length = items.length := replace_length items indices digit 0

private theorem replace_get (items indices : List Nat) (digit start offset : Nat) :
    (replaceAt.go indices digit items start)[offset]? =
      items[offset]?.map (fun item => if start + offset ∈ indices then digit else item) := by
  induction offset generalizing items start with
  | zero => cases items <;> simp [replaceAt.go]
  | succ offset ih =>
      cases items with
      | nil => simp [replaceAt.go]
      | cons head tail => simpa [replaceAt.go, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using ih tail (start + 1)

/-- Replacement is a membership test on positions, so duplicate indices have no additional effect. -/
theorem replaceAt_get (items indices : List Nat) (digit offset : Nat) :
    (replaceAt items indices digit)[offset]? =
      items[offset]?.map (fun item => if offset ∈ indices then digit else item) := by
  simpa only [replaceAt, Nat.zero_add] using replace_get items indices digit 0 offset

/-- Every digit remains a decimal digit when the original digits and replacement are decimal. -/
theorem replaceAt_digits (items indices : List Nat) (digit : Nat) (digitBound : digit < 10)
    (bounded : ∀ item ∈ items, item < 10) : ∀ item ∈ replaceAt items indices digit, item < 10 := by
  intro item member
  obtain ⟨offset, selected⟩ := List.mem_iff_getElem?.mp member
  rw [replaceAt_get] at selected
  cases original : items[offset]? with
  | none => simp [original] at selected
  | some value =>
      have valueBound := bounded value (List.mem_of_getElem? original)
      simp only [original, Option.map_some, Option.some.injEq] at selected
      split_ifs at selected <;> omega

/-- All positions selected from this suffix originally contain the same digit. -/
def SameDigit (items indices : List Nat) (digit : Nat) (start : Nat := 0) : Prop :=
  ∀ offset value, items[offset]? = some value → start + offset ∈ indices → value = digit

private theorem same_tail (head : Nat) (tail indices : List Nat) (digit start : Nat)
    (same : SameDigit (head :: tail) indices digit start) : SameDigit tail indices digit (start + 1) := by
  intro offset value selected member
  exact same (offset + 1) value (by simpa using selected) (by simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using member)

private theorem replacement_difference (items indices : List Nat) (original digit start : Nat)
    (same : SameDigit items indices original start) :
    (10 : Int) ^ start * (((Nat.ofDigits 10 (replaceAt.go indices digit items start) : Nat) : Int) - ((Nat.ofDigits 10 items : Nat) : Int)) =
      ((digit : Int) - original) * (weight indices items.length start : Int) := by
  induction items generalizing start with
  | nil => simp [replaceAt.go, Nat.ofDigits]
  | cons head tail ih =>
      have rest := ih (start + 1) (same_tail head tail indices original start same)
      rw [List.length_cons, weight_succ]
      simp only [replaceAt.go, Nat.ofDigits, Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat, Nat.cast_id]
      by_cases member : start ∈ indices
      · have headSame : head = original := same 0 head rfl (by simpa using member)
        simp only [member, if_true, Nat.cast_pow, Nat.cast_ofNat]
        rw [headSame]
        rw [pow_succ] at rest
        nlinarith [rest]
      · simp only [member, if_false, Nat.cast_zero, zero_add]
        rw [pow_succ] at rest
        nlinarith [rest]

/-- The solver's signed shift formula is the reference's simultaneous digit replacement. -/
theorem replacement_value (items indices : List Nat) (original digit : Nat)
    (same : SameDigit items indices original) :
    ((Nat.ofDigits 10 (replaceAt items indices digit) : Nat) : Int) =
      ((Nat.ofDigits 10 items : Nat) : Int) + ((digit : Int) - original) * (weight indices items.length : Int) := by
  have difference := replacement_difference items indices original digit 0 same
  simp only [pow_zero, one_mul] at difference
  dsimp only [replaceAt]
  linarith


/-- Replacing selected positions changes nothing exactly when they already share that digit. -/
theorem replaceAt_eq_self_iff (items indices : List Nat) (digit : Nat) :
    replaceAt items indices digit = items ↔ SameDigit items indices digit := by
  constructor
  · intro equal offset value found member
    have selected := replaceAt_get items indices digit offset
    rw [equal, found] at selected
    simp only [Nat.zero_add] at member
    simpa only [member, if_true, Option.map_some, Option.some.injEq] using selected
  · intro same
    apply List.ext_getElem?
    intro offset
    rw [replaceAt_get]
    cases found : items[offset]? with
    | none => simp
    | some value =>
        by_cases member : offset ∈ indices
        · have equal := same offset value found (by simpa using member)
          simp [member, equal]
        · simp [member]

/-- Membership of the original number forces all replaced positions to have a common digit. -/
theorem family_member_iff (n : Nat) (indices : List Nat) :
    n ∈ familyNumbers n indices ↔ ∃ digit < 10, SameDigit (Nat.digits 10 n) indices digit := by
  have bounded : ∀ digit ∈ Nat.digits 10 n, digit < 10 := fun _ member => Nat.digits_lt_base (by decide) member
  simp only [familyNumbers, List.mem_filterMap, List.mem_range]
  constructor
  · rintro ⟨digit, digitBound, selected⟩
    split_ifs at selected with width
    have equal : Nat.ofDigits 10 (replaceAt (Nat.digits 10 n) indices digit) = n := Option.some.inj selected
    have sameDigits : replaceAt (Nat.digits 10 n) indices digit = Nat.digits 10 n :=
      Nat.ofDigits_inj_of_len_eq (by decide) (replaceAt_length _ _ _) (replaceAt_digits _ _ _ digitBound bounded) bounded
        (by simpa only [Nat.ofDigits_digits] using equal)
    exact ⟨digit, digitBound, (replaceAt_eq_self_iff _ _ _).mp sameDigits⟩
  · rintro ⟨digit, digitBound, same⟩
    refine ⟨digit, digitBound, ?_⟩
    rw [(replaceAt_eq_self_iff _ _ _).mpr same, Nat.ofDigits_digits]
    simp

/-- Only membership in the index list matters, independently of order or duplicate indices. -/
theorem replaceAt_congr_indices (items left right : List Nat) (digit : Nat)
    (same : ∀ position, position ∈ left ↔ position ∈ right) :
    replaceAt items left digit = replaceAt items right digit := by
  apply List.ext_getElem?
  intro offset
  simp only [replaceAt_get, same]

theorem family_congr_indices (n : Nat) (left right : List Nat)
    (same : ∀ position, position ∈ left ↔ position ∈ right) : familyNumbers n left = familyNumbers n right := by
  simp only [familyNumbers, replaceAt_congr_indices _ left right _ same]

theorem familySize_congr_indices (n : Nat) (left right : List Nat)
    (same : ∀ position, position ∈ left ↔ position ∈ right) : primeFamilySize n left = primeFamilySize n right := by
  simp only [primeFamilySize, family_congr_indices n left right same]


/-- For distinct in-range indices, the reference's selected weight is the solver's sum of powers. -/
theorem weight_eq_sum (indices : List Nat) (length : Nat) (distinct : indices.Nodup)
    (bounded : ∀ index ∈ indices, index < length) :
    weight indices length = (indices.map (10 ^ ·)).sum := by
  have unique : ((List.range' 0 length).filter (· ∈ indices)).Nodup := (List.nodup_range').filter _
  have represented (index : Nat) : index ∈ (List.range' 0 length).filter (· ∈ indices) ↔ index ∈ indices := by
    simp only [List.mem_filter, List.mem_range', decide_eq_true_eq]
    constructor
    · exact fun member => member.2
    · intro member
      exact ⟨⟨index, bounded index member, by simp⟩, member⟩
  have permuted := (List.perm_ext_iff_of_nodup unique distinct).mpr represented
  exact (permuted.map (10 ^ ·)).sum_eq

/-- Convert positions counted from the left to positions counted from the right. -/
def reflected (length : Nat) (indices : List Nat) : List Nat := indices.map (fun index => length - 1 - index)

theorem reflected_bounded (length : Nat) (indices : List Nat) (bounded : ∀ index ∈ indices, index < length) :
    ∀ index ∈ reflected length indices, index < length := by
  intro index member
  obtain ⟨original, present, rfl⟩ := List.mem_map.mp member
  have := bounded original present
  omega

theorem reflected_nodup (length : Nat) (indices : List Nat) (distinct : indices.Nodup)
    (bounded : ∀ index ∈ indices, index < length) : (reflected length indices).Nodup := by
  apply List.Nodup.map_on ?_ distinct
  intro left leftMember right rightMember equal
  have := bounded left leftMember
  have := bounded right rightMember
  omega

theorem reflected_twice (length : Nat) (indices : List Nat) (bounded : ∀ index ∈ indices, index < length) :
    reflected length (reflected length indices) = indices := by
  simp only [reflected, List.map_map]
  calc
    _ = indices.map id := by
      apply List.map_congr_left
      intro index member
      have := bounded index member
      dsimp
      omega
    _ = indices := List.map_id indices

/-- Big-endian positions carry exactly the decimal weights used in the Python shift sum. -/
theorem weight_reflected (indices : List Nat) (length : Nat) (distinct : indices.Nodup)
    (bounded : ∀ index ∈ indices, index < length) :
    weight (reflected length indices) length = (indices.map (fun index => 10 ^ (length - 1 - index))).sum := by
  rw [weight_eq_sum _ length (reflected_nodup length indices distinct bounded) (reflected_bounded length indices bounded)]
  simp only [reflected, List.map_map, Function.comp_def]


/-- Reading digits from the left and reflecting the selected positions gives the same shared-digit condition. -/
theorem sameDigit_reflected (items indices : List Nat) (digit : Nat)
    (bounded : ∀ index ∈ indices, index < items.length) :
    SameDigit items.reverse indices digit ↔ SameDigit items (reflected items.length indices) digit := by
  constructor
  · intro same offset value found member
    simp only [Nat.zero_add] at member
    obtain ⟨position, present, equal⟩ := List.mem_map.mp member
    have reverseFound : items.reverse[position]? = some value := by
      rw [List.getElem?_reverse (bounded position present), equal]
      exact found
    exact same position value reverseFound (by simpa using present)
  · intro same offset value found member
    have inside : offset < items.length := by
      simpa only [List.length_reverse] using (List.getElem?_eq_some_iff.mp found).1
    have originalFound : items[items.length - 1 - offset]? = some value := by
      rwa [List.getElem?_reverse inside] at found
    apply same _ value originalFound
    simp only [Nat.zero_add]
    exact List.mem_map.mpr ⟨offset, by simpa using member, rfl⟩

/-- This is the solver's big-endian decimal shift formula, with natural conversion made explicit. -/
theorem replacement_value_reflected (items indices : List Nat) (original digit : Nat)
    (distinct : indices.Nodup) (bounded : ∀ index ∈ indices, index < items.length)
    (same : SameDigit items.reverse indices original) :
    ((Nat.ofDigits 10 (replaceAt items (reflected items.length indices) digit) : Nat) : Int) =
      ((Nat.ofDigits 10 items : Nat) : Int) + ((digit : Int) - original) *
        ((indices.map (fun index => 10 ^ (items.length - 1 - index))).sum : Nat) := by
  have result := replacement_value items (reflected items.length indices) original digit ((sameDigit_reflected _ _ _ bounded).mp same)
  rwa [weight_reflected indices items.length distinct bounded] at result

/-- With valid decimal digits, keeping the full width is equivalent to a nonzero leading digit. -/
theorem decimal_width_iff (items : List Nat) (bounded : ∀ digit ∈ items, digit < 10) :
    (Nat.digits 10 (Nat.ofDigits 10 items)).length = items.length ↔
      ∀ nonempty : items ≠ [], items.getLast nonempty ≠ 0 := by
  constructor
  · intro width nonempty
    have equal : Nat.digits 10 (Nat.ofDigits 10 items) = items := Nat.ofDigits_inj_of_len_eq (by decide) width
      (fun _ member => Nat.digits_lt_base (by decide) member) bounded (Nat.ofDigits_digits 10 _)
    have nonzero : Nat.ofDigits 10 items ≠ 0 := by
      intro zero
      simp only [zero, Nat.digits_zero] at equal
      exact nonempty equal.symm
    simpa only [equal] using (Nat.getLast_digit_ne_zero 10 nonzero)
  · intro canonical
    rw [Nat.digits_ofDigits 10 (by decide) items bounded canonical]


/-- The leading digit is selected using the same position membership test as every other digit. -/
theorem replaceAt_getLast? (items indices : List Nat) (digit : Nat) :
    (replaceAt items indices digit).getLast? =
      items.getLast?.map (fun original => if items.length - 1 ∈ indices then digit else original) := by
  rw [List.getLast?_eq_getElem?, replaceAt_length, replaceAt_get, ← List.getLast?_eq_getElem?]

theorem decimal_width_iff_last (items : List Nat) (bounded : ∀ digit ∈ items, digit < 10) :
    (Nat.digits 10 (Nat.ofDigits 10 items)).length = items.length ↔ items.getLast? ≠ some 0 := by
  by_cases empty : items = []
  · subst items
    simp
  · have last : items.getLast? = some (items.getLast empty) := by
      rw [List.getLast?_eq_getElem?, List.getLast_eq_getElem]
      exact List.getElem?_eq_getElem (by have := List.length_pos_iff.mpr empty; omega)
    rw [decimal_width_iff items bounded, last]
    simp only [ne_eq, Option.some.injEq]
    constructor
    · intro canonical
      exact canonical empty
    · intro nonzero nonempty
      exact nonzero

/-- For a positive candidate, decimal width changes exactly when replacement zero hits the leading position. -/
theorem replacement_width (n : Nat) (positive : 0 < n) (indices : List Nat) (digit : Nat) (digitBound : digit < 10) :
    (Nat.digits 10 (Nat.ofDigits 10 (replaceAt (Nat.digits 10 n) indices digit))).length = (Nat.digits 10 n).length ↔
      digit ≠ 0 ∨ (Nat.digits 10 n).length - 1 ∉ indices := by
  let items := Nat.digits 10 n
  have nonempty : items ≠ [] := Nat.digits_ne_nil_iff_ne_zero.mpr (by omega)
  have leading : items.getLast nonempty ≠ 0 := Nat.getLast_digit_ne_zero 10 (by omega)
  have last : items.getLast? = some (items.getLast nonempty) := by
    rw [List.getLast?_eq_getElem?, List.getLast_eq_getElem]
    exact List.getElem?_eq_getElem (by have := List.length_pos_iff.mpr nonempty; omega)
  have bounded : ∀ value ∈ items, value < 10 := fun _ member => Nat.digits_lt_base (by decide) member
  have width := decimal_width_iff_last (replaceAt items indices digit) (replaceAt_digits items indices digit digitBound bounded)
  rw [replaceAt_length, replaceAt_getLast?, last] at width
  change _ = items.length ↔ digit ≠ 0 ∨ items.length - 1 ∉ indices
  rw [width]
  by_cases member : items.length - 1 ∈ indices <;> simp [member, leading]

/-- Every replacement remains a valid table index below the next power of ten. -/
theorem replacement_lt_pow (n : Nat) (indices : List Nat) (digit : Nat) (digitBound : digit < 10) :
    Nat.ofDigits 10 (replaceAt (Nat.digits 10 n) indices digit) < 10 ^ (Nat.digits 10 n).length := by
  have bounded : ∀ value ∈ Nat.digits 10 n, value < 10 := fun _ member => Nat.digits_lt_base (by decide) member
  simpa only [replaceAt_length] using Nat.ofDigits_lt_base_pow_length (by decide)
    (replaceAt_digits (Nat.digits 10 n) indices digit digitBound bounded)


/-- The integer expression evaluated for each replacement digit by the optimized search. -/
def shiftedValue (n : Nat) (indices : List Nat) (original digit : Nat) : Int :=
  (n : Int) + ((digit : Int) - original) * (weight indices (Nat.digits 10 n).length : Int)

theorem shiftedValue_eq_replacement (n : Nat) (indices : List Nat) (original digit : Nat)
    (same : SameDigit (Nat.digits 10 n) indices original) :
    shiftedValue n indices original digit = ((Nat.ofDigits 10 (replaceAt (Nat.digits 10 n) indices digit) : Nat) : Int) := by
  simpa only [shiftedValue, Nat.ofDigits_digits] using (replacement_value (Nat.digits 10 n) indices original digit same).symm

/-- All generated values fit the inclusive sieve used for the candidate's decimal width. -/
theorem shiftedValue_bound (n : Nat) (indices : List Nat) (original digit : Nat) (digitBound : digit < 10)
    (same : SameDigit (Nat.digits 10 n) indices original) :
    0 ≤ shiftedValue n indices original digit ∧ shiftedValue n indices original digit < (10 : Int) ^ (Nat.digits 10 n).length := by
  rw [shiftedValue_eq_replacement n indices original digit same]
  constructor
  · exact Int.natCast_nonneg _
  · exact_mod_cast replacement_lt_pow n indices digit digitBound

/-- The reference family is precisely the solver's signed shift values after excluding leading zero. -/
theorem family_values (n : Nat) (positive : 0 < n) (indices : List Nat) (original : Nat)
    (same : SameDigit (Nat.digits 10 n) indices original) :
    (familyNumbers n indices).map Int.ofNat =
      ((List.range 10).filter (fun digit => decide (digit ≠ 0 ∨ (Nat.digits 10 n).length - 1 ∉ indices))).map
        (shiftedValue n indices original) := by
  have collect (digits : List Nat) (bounded : ∀ digit ∈ digits, digit < 10) :
      (digits.filterMap (fun digit =>
        let value := Nat.ofDigits 10 (replaceAt (Nat.digits 10 n) indices digit)
        if (Nat.digits 10 value).length = (Nat.digits 10 n).length then some value else none)).map Int.ofNat =
      (digits.filter (fun digit => decide (digit ≠ 0 ∨ (Nat.digits 10 n).length - 1 ∉ indices))).map
        (shiftedValue n indices original) := by
    induction digits with
    | nil => rfl
    | cons digit tail ih =>
        have width := replacement_width n positive indices digit (bounded digit (by simp))
        have rest := ih (fun value member => bounded value (by simp [member]))
        by_cases accepted : digit ≠ 0 ∨ (Nat.digits 10 n).length - 1 ∉ indices <;>
          simp [width, accepted, rest, shiftedValue_eq_replacement n indices original digit same]
  exact collect (List.range 10) (fun _ member => List.mem_range.mp member)

/-- The leading position in the solver is zero; in the reference it is the last little-endian position. -/
theorem leading_reflected (length : Nat) (positive : 0 < length) (indices : List Nat)
    (bounded : ∀ index ∈ indices, index < length) : length - 1 ∈ reflected length indices ↔ 0 ∈ indices := by
  constructor
  · intro member
    obtain ⟨index, present, equal⟩ := List.mem_map.mp member
    have := bounded index present
    have zero : index = 0 := by omega
    simpa only [zero] using present
  · intro member
    exact List.mem_map.mpr ⟨0, member, by simp⟩

/-- The ordered replacement values, decimal weights, and leading-zero condition all match the Python search. -/
theorem family_values_reflected (n : Nat) (positive : 0 < n) (indices : List Nat) (original : Nat)
    (distinct : indices.Nodup) (bounded : ∀ index ∈ indices, index < (Nat.digits 10 n).length)
    (same : SameDigit (Nat.digits 10 n).reverse indices original) :
    (familyNumbers n (reflected (Nat.digits 10 n).length indices)).map Int.ofNat =
      ((List.range 10).filter (fun digit => decide (digit ≠ 0 ∨ 0 ∉ indices))).map
        (fun (digit : Nat) => (n : Int) + ((digit : Int) - original) *
          ((indices.map (fun index => 10 ^ ((Nat.digits 10 n).length - 1 - index))).sum : Nat)) := by
  have nonempty : Nat.digits 10 n ≠ [] := Nat.digits_ne_nil_iff_ne_zero.mpr (by omega)
  have lengthPositive : 0 < (Nat.digits 10 n).length := List.length_pos_iff.mpr nonempty
  have family := family_values n positive (reflected (Nat.digits 10 n).length indices) original
    ((sameDigit_reflected _ _ _ bounded).mp same)
  rw [family]
  simp only [leading_reflected _ lengthPositive indices bounded]
  apply List.map_congr_left
  intro digit _
  simp only [shiftedValue, weight_reflected indices _ distinct bounded]


/-- The shared-digit grouping guarantees that the original number belongs to its replacement family. -/
theorem family_member_of_sameDigit (n : Nat) (indices : List Nat) (original : Nat)
    (same : SameDigit (Nat.digits 10 n) indices original) : n ∈ familyNumbers n indices := by
  apply (family_member_iff n indices).mpr
  by_cases bounded : original < 10
  · exact ⟨original, bounded, same⟩
  · refine ⟨0, by decide, ?_⟩
    intro offset value found member
    have equal := same offset value found member
    have digitBound := Nat.digits_lt_base (by decide : 1 < 10) (List.mem_of_getElem? found)
    omega

/-- Counting prime shift values after the leading-zero exclusion gives exactly the reference's family size. -/
theorem familySize_reflected (n : Nat) (positive : 0 < n) (indices : List Nat) (original : Nat)
    (distinct : indices.Nodup) (bounded : ∀ index ∈ indices, index < (Nat.digits 10 n).length)
    (same : SameDigit (Nat.digits 10 n).reverse indices original) :
    primeFamilySize n (reflected (Nat.digits 10 n).length indices) =
      (((List.range 10).filter (fun digit => decide (digit ≠ 0 ∨ 0 ∉ indices))).filter
        (fun (digit : Nat) => isPrime (((n : Int) + ((digit : Int) - original) *
          ((indices.map (fun index => 10 ^ ((Nat.digits 10 n).length - 1 - index))).sum : Nat)).toNat))).length := by
  have values := family_values_reflected n positive indices original distinct bounded same
  have counted := congrArg (fun values : List Int => (values.filter (fun value => isPrime value.toNat)).length) values
  simpa only [List.filter_map, List.length_map, Function.comp_def, Int.toNat_natCast, primeFamilySize] using counted

end ProjectEulerSolutions.PrimeReplacement
