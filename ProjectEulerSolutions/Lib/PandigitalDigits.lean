import ProjectEulerStatements.Lib.PandigitalMask

/-! Bitwise duplicate detection for the target decimal digits one through a variable width. -/
namespace ProjectEulerSolutions.PandigitalDigits
open ProjectEulerStatements.PandigitalMask
set_option Elab.async false

/-- Track visited digits while checking their range and uniqueness. -/
def scan (width : Nat) : List Nat → Nat → Bool
  | [], _ => true
  | digit :: rest, mask =>
      if digit = 0 then false
      else if width < digit then false
      else if mask &&& 2 ^ digit ≠ 0 then false
      else scan width rest (mask ||| 2 ^ digit)

def check (width : Nat) (digits : List Nat) : Bool :=
  decide (digits.length = width) && scan width digits 0

def reference (width : Nat) (digits : List Nat) : Bool :=
  decide (digits.length = width) && digits.all (fun digit => decide (1 ≤ digit ∧ digit ≤ width)) &&
    decide (digits.eraseDups.length = width)

theorem scan_success (width : Nat) (items seen : List Nat) (distinct : seen.Nodup) :
    scan width items (encode seen) = true ↔
      (∀ digit ∈ items, 1 ≤ digit ∧ digit ≤ width) ∧ (items ++ seen).Nodup := by
  induction items generalizing seen with
  | nil => simp [scan, distinct]
  | cons head tail ih =>
      by_cases zero : head = 0
      · subst head
        simp [scan]
      · by_cases outside : width < head
        · simp only [scan, if_neg zero, if_pos outside, Bool.false_eq_true, false_iff]
          rintro ⟨bounded, _⟩
          have headBound := (bounded head (by simp)).2
          omega
        · by_cases duplicate : head ∈ seen
          · have overlap := (overlap_iff seen head).mpr duplicate
            simp only [scan, if_neg zero, if_neg outside, if_pos overlap, Bool.false_eq_true, false_iff]
            rintro ⟨_, nodup⟩
            exact (List.nodup_cons.mp nodup).1 (List.mem_append_right _ duplicate)
          · have noOverlap : ¬ encode seen &&& 2 ^ head ≠ 0 :=
              fun overlap => duplicate ((overlap_iff seen head).mp overlap)
            have next := ih (head :: seen) (List.nodup_cons.mpr ⟨duplicate, distinct⟩)
            change scan width tail (encode seen ||| 2 ^ head) = true ↔ _ at next
            rw [scan, if_neg zero, if_neg outside, if_neg noOverlap, next]
            have permutation : List.Perm (tail ++ head :: seen) ((head :: tail) ++ seen) := List.perm_middle
            constructor
            · rintro ⟨bounds, nodup⟩
              refine ⟨?_, permutation.nodup nodup⟩
              intro digit member
              rcases List.mem_cons.mp member with rfl | member
              · omega
              · exact bounds digit member
            · rintro ⟨bounds, nodup⟩
              exact ⟨fun digit member => bounds digit (List.mem_cons_of_mem _ member), permutation.symm.nodup nodup⟩

theorem check_success (width : Nat) (digits : List Nat) :
    check width digits = true ↔ digits.length = width ∧
      (∀ digit ∈ digits, 1 ≤ digit ∧ digit ≤ width) ∧ digits.Nodup := by
  simp only [check, Bool.and_eq_true, decide_eq_true_eq]
  have scanned := scan_success width digits [] (by simp)
  simpa only [encode, List.append_nil] using and_congr_right (fun _ => scanned)

theorem check_eq_reference (width : Nat) (digits : List Nat) : check width digits = reference width digits := by
  apply Bool.eq_iff_iff.mpr
  rw [check_success]
  simp only [reference, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true]
  constructor
  · rintro ⟨length, bounds, distinct⟩
    exact ⟨⟨length, bounds⟩, (eraseDups_length_eq_iff digits).mpr distinct |>.trans length⟩
  · rintro ⟨⟨length, bounds⟩, erased⟩
    exact ⟨length, bounds, (eraseDups_length_eq_iff digits).mp (erased.trans length.symm)⟩

/-- Length, range, and uniqueness require every target digit to occur exactly once. -/
theorem check_iff_perm (width : Nat) (digits : List Nat) :
    check width digits = true ↔ digits.Perm (List.range' 1 width) := by
  rw [check_success]
  constructor
  · rintro ⟨length, bounds, distinct⟩
    have subset : digits.toFinset ⊆ Finset.Icc 1 width := by
      intro digit member
      exact Finset.mem_Icc.mpr (bounds digit (List.mem_toFinset.mp member))
    have same : digits.toFinset = Finset.Icc 1 width := by
      apply Finset.eq_of_subset_of_card_le subset
      rw [List.toFinset_card_of_nodup distinct, length]
      simp
    apply List.perm_ext_iff_of_nodup distinct (List.nodup_range' (s := 1) (n := width)) |>.mpr
    intro digit
    rw [← List.mem_toFinset, same]
    simp only [Finset.mem_Icc, List.mem_range']
    constructor
    · rintro ⟨positive, upper⟩
      exact ⟨digit - 1, by omega, by omega⟩
    · rintro ⟨offset, inside, rfl⟩
      omega
  · intro permutation
    refine ⟨by simpa using permutation.length_eq, ?_, permutation.nodup_iff.mpr (List.nodup_range')⟩
    intro digit member
    obtain ⟨offset, inside, same⟩ := List.mem_range'.mp (permutation.mem_iff.mp member)
    omega

end ProjectEulerSolutions.PandigitalDigits
