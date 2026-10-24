import ProjectEulerSolutions.Lib.PandigitalDigits
import Mathlib.Data.Nat.Digits.Lemmas
import Mathlib.Data.Nat.Sqrt
import Mathlib.Data.List.Permutation
import Mathlib.Tactic.Linarith

/-! Complete factor bounds for pandigital multiplication with a variable target width. -/
namespace ProjectEulerSolutions.PandigitalProductSearch

set_option Elab.async false

def digits (n : Nat) : List Nat := (Nat.digits 10 n).reverse

def accepted (width a b : Nat) : Bool :=
  PandigitalDigits.check width (digits a ++ digits b ++ digits (a * b))

theorem accepted_spec (width a b : Nat) :
    accepted width a b = true ↔
      (digits a ++ digits b ++ digits (a * b)).Perm (List.range' 1 width) :=
  PandigitalDigits.check_iff_perm width _

theorem accepted_length (width a b : Nat) (yes : accepted width a b = true) :
    (Nat.digits 10 a).length + (Nat.digits 10 b).length +
      (Nat.digits 10 (a * b)).length = width := by
  have length := (PandigitalDigits.check_success width _).mp yes |>.1
  simpa only [List.length_append, digits, List.length_reverse] using length

theorem accepted_swap (width a b : Nat) : accepted width a b = accepted width b a := by
  apply Bool.eq_iff_iff.mpr
  rw [accepted_spec, accepted_spec, Nat.mul_comm b a]
  have swapped : (digits a ++ digits b ++ digits (a * b)).Perm
      (digits b ++ digits a ++ digits (a * b)) := (List.perm_append_comm).append_right _
  exact ⟨fun permuted => swapped.symm.trans permuted, fun permuted => swapped.trans permuted⟩

theorem digits_positive (n : Nat) (positive : 0 < n) : 0 < (Nat.digits 10 n).length := by
  simp only [List.length_pos_iff_ne_nil]
  exact Nat.digits_ne_nil_iff_ne_zero.mpr (by omega)

theorem digits_upper (n : Nat) : n < 10 ^ (Nat.digits 10 n).length :=
  (Nat.digits_length_le_iff (by decide : 1 < 10) n).mp (Nat.le_refl _)

theorem digits_lower (n : Nat) (positive : 0 < n) :
    10 ^ ((Nat.digits 10 n).length - 1) ≤ n := by
  apply (Nat.lt_digits_length_iff (by decide : 1 < 10) n).mp
  have := digits_positive n positive
  omega

/-- The product of positive a- and b-digit numbers has a+b-1 or a+b digits. -/
theorem product_length (a b : Nat) (aPositive : 0 < a) (bPositive : 0 < b) :
    (Nat.digits 10 a).length + (Nat.digits 10 b).length - 1 ≤
      (Nat.digits 10 (a * b)).length ∧
    (Nat.digits 10 (a * b)).length ≤
      (Nat.digits 10 a).length + (Nat.digits 10 b).length := by
  have aLength := digits_positive a aPositive
  have bLength := digits_positive b bPositive
  have productLower := Nat.mul_le_mul (digits_lower a aPositive) (digits_lower b bPositive)
  rw [← Nat.pow_add] at productLower
  have lower := (Nat.lt_digits_length_iff (by decide : 1 < 10) (a * b)).mpr productLower
  have leftUpper := Nat.mul_lt_mul_of_pos_right (digits_upper a) bPositive
  have rightUpper := Nat.mul_lt_mul_of_pos_left (digits_upper b)
    (Nat.pow_pos (by decide : 0 < 10) : 0 < 10 ^ (Nat.digits 10 a).length)
  have productUpper := leftUpper.trans rightUpper
  rw [← Nat.pow_add] at productUpper
  have upper := (Nat.digits_length_le_iff (by decide : 1 < 10) (a * b)).mpr productUpper
  constructor
  · omega
  · exact upper

/-- The identity's total width fixes the product width, independently of the digit values. -/
theorem product_width (width a b : Nat) (aPositive : 0 < a) (bPositive : 0 < b)
    (yes : accepted width a b = true) : (Nat.digits 10 (a * b)).length = width / 2 := by
  have total := accepted_length width a b yes
  have bounds := product_length a b aPositive bPositive
  have aLength := digits_positive a aPositive
  have bLength := digits_positive b bPositive
  omega

def lowerProduct (width : Nat) : Nat := 10 ^ (width / 2 - 1)
def upperProduct (width : Nat) : Nat := 10 ^ (width / 2) - 1
def firstB (width a : Nat) : Nat := max a ((lowerProduct width + a - 1) / a)
def lastB (width a : Nat) : Nat := upperProduct width / a

theorem product_bounds (width a b : Nat) (aPositive : 0 < a) (bPositive : 0 < b)
    (yes : accepted width a b = true) :
    lowerProduct width ≤ a * b ∧ a * b ≤ upperProduct width := by
  have positive : 0 < a * b := Nat.mul_pos aPositive bPositive
  have same := product_width width a b aPositive bPositive yes
  have lower := digits_lower (a * b) positive
  have upper := digits_upper (a * b)
  rw [same] at lower upper
  exact ⟨lower, by unfold upperProduct; omega⟩

theorem firstB_iff (width a b : Nat) (positive : 0 < a) :
    firstB width a ≤ b ↔ a ≤ b ∧ lowerProduct width ≤ a * b := by
  rw [firstB, Nat.max_le, Nat.div_le_iff_le_mul positive, Nat.mul_comm b a]
  have positiveLower : 0 < lowerProduct width := Nat.pow_pos (by decide)
  omega

theorem lastB_iff (width a b : Nat) (positive : 0 < a) :
    b ≤ lastB width a ↔ a * b ≤ upperProduct width := by
  rw [lastB, Nat.le_div_iff_mul_le positive, Nat.mul_comm b a]

def row (width a : Nat) : List Nat :=
  ((List.range' (firstB width a) (lastB width a + 1 - firstB width a)).filter
    (accepted width a)).map (a * ·)

def candidates (width : Nat) : List Nat :=
  (List.range' 1 (Nat.sqrt (upperProduct width))).flatMap (row width)

def Identity (width value : Nat) : Prop :=
  ∃ a b, 0 < a ∧ 0 < b ∧ value = a * b ∧ accepted width a b = true

theorem mem_row (width a value : Nat) (positive : 0 < a) :
    value ∈ row width a ↔ ∃ b, a ≤ b ∧ lowerProduct width ≤ a * b ∧
      a * b ≤ upperProduct width ∧ value = a * b ∧ accepted width a b = true := by
  simp only [row, List.mem_map, List.mem_filter, List.mem_range'_1]
  constructor
  · rintro ⟨b, ⟨⟨lower, upper⟩, yes⟩, equal⟩
    have low := (firstB_iff width a b positive).mp lower
    have high := (lastB_iff width a b positive).mp (by omega)
    exact ⟨b, low.1, low.2, high, equal.symm, yes⟩
  · rintro ⟨b, ordered, lower, upper, equal, yes⟩
    have low := (firstB_iff width a b positive).mpr ⟨ordered, lower⟩
    have high := (lastB_iff width a b positive).mpr upper
    exact ⟨b, ⟨⟨low, by omega⟩, yes⟩, equal.symm⟩

/-- Ordering the two factors removes symmetric work while retaining every product. -/
theorem mem_candidates (width value : Nat) : value ∈ candidates width ↔ Identity width value := by
  constructor
  · intro member
    obtain ⟨a, aMember, rowMember⟩ := List.mem_flatMap.mp member
    have bounds := List.mem_range'_1.mp aMember
    obtain ⟨b, ordered, _, _, equal, yes⟩ := (mem_row width a value (by omega)).mp rowMember
    exact ⟨a, b, by omega, by omega, equal, yes⟩
  · rintro ⟨a, b, aPositive, bPositive, equal, yes⟩
    have insert (a b : Nat) (aPositive : 0 < a) (bPositive : 0 < b) (ordered : a ≤ b)
        (equal : value = a * b) (yes : accepted width a b = true) : value ∈ candidates width := by
      obtain ⟨lower, upper⟩ := product_bounds width a b aPositive bPositive yes
      have square : a * a ≤ upperProduct width :=
        (Nat.mul_le_mul_left a ordered).trans upper
      have root : a ≤ Nat.sqrt (upperProduct width) := Nat.le_sqrt'.mpr (by simpa [Nat.pow_two]
        using square)
      exact List.mem_flatMap.mpr ⟨a, List.mem_range'_1.mpr ⟨aPositive, by omega⟩,
        (mem_row width a value aPositive).mpr ⟨b, ordered, lower, upper, equal, yes⟩⟩
    by_cases ordered : a ≤ b
    · exact insert a b aPositive bPositive ordered equal yes
    · exact insert b a bPositive aPositive (by omega)
        (by simpa only [Nat.mul_comm] using equal)
        (by simpa only [accepted_swap width a b] using yes)

theorem decimal_digits (n digit : Nat) (member : digit ∈ digits n) : digit < 10 :=
  Nat.digits_lt_base (by decide) (List.mem_reverse.mp member)

theorem accepted_large (width a b : Nat) (large : 9 < width) : accepted width a b = false := by
  apply Bool.eq_false_iff.mpr
  intro yes
  have permuted := (accepted_spec width a b).mp yes
  have target : 10 ∈ List.range' 1 width := List.mem_range'_1.mpr ⟨by decide, by omega⟩
  have member := permuted.mem_iff.mpr target
  simp only [List.mem_append] at member
  rcases member with (member | member) | member
  · have := decimal_digits a 10 member
    omega
  · have := decimal_digits b 10 member
    omega
  · have := decimal_digits (a * b) 10 member
    omega

theorem no_identity_small (width value : Nat) (small : width < 3) : ¬ Identity width value := by
  rintro ⟨a, b, aPositive, bPositive, _, yes⟩
  have total := accepted_length width a b yes
  have aLength := digits_positive a aPositive
  have bLength := digits_positive b bPositive
  have pLength := digits_positive (a * b) (Nat.mul_pos aPositive bPositive)
  omega

theorem no_identity_large (width value : Nat) (large : 9 < width) : ¬ Identity width value := by
  rintro ⟨a, b, _, _, _, yes⟩
  rw [accepted_large width a b large] at yes
  cases yes

def answer (width : Nat) : Nat := (candidates width).eraseDups.sum

theorem answer_small (width : Nat) (small : width < 3) : answer width = 0 := by
  have empty : candidates width = [] := List.eq_nil_iff_forall_not_mem.mpr
    (fun n member => no_identity_small width n small ((mem_candidates width n).mp member))
  simp [answer, empty]

theorem answer_large (width : Nat) (large : 9 < width) : answer width = 0 := by
  have empty : candidates width = [] := List.eq_nil_iff_forall_not_mem.mpr
    (fun n member => no_identity_large width n large ((mem_candidates width n).mp member))
  simp [answer, empty]

end ProjectEulerSolutions.PandigitalProductSearch
