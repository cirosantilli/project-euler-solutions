import ProjectEulerSolutions.Lib.PandigitalProductSearch
import lang.py.FormalSW.Permutations

/-! Splitting every target-digit permutation enumerates exactly all multiplication identities. -/
namespace ProjectEulerSolutions.PandigitalProductSearch
open FormalSW.PyAst

set_option Elab.async false

def value (items : List Nat) : Nat := Nat.ofDigits 10 items.reverse

theorem value_digits (n : Nat) : value (digits n) = n := by
  simp [value, digits, Nat.ofDigits_digits]

theorem digits_value (items : List Nat) (valid : ∀ digit ∈ items, 0 < digit ∧ digit < 10) :
    digits (value items) = items := by
  unfold value digits
  rw [Nat.digits_ofDigits 10 (by decide)]
  · simp
  · intro digit member
    exact (valid digit (List.mem_reverse.mp member)).2
  · intro nonempty
    have member := List.mem_reverse.mp (List.getLast_mem nonempty)
    have := (valid _ member).1
    omega

theorem value_positive (items : List Nat) (valid : ∀ digit ∈ items, 0 < digit ∧ digit < 10)
    (nonempty : items ≠ []) : 0 < value items := by
  by_contra zero
  have same : value items = 0 := by omega
  have restored := digits_value items valid
  rw [same] at restored
  simp only [digits, Nat.digits_zero, List.reverse_nil] at restored
  exact nonempty restored.symm

def splitProducts (width : Nat) (items : List Nat) : List Nat :=
  (List.range' 1 (width - 2)).flatMap fun first =>
    (List.range' (first + 1) (width - first - 1)).flatMap fun second =>
      if value (items.take first) * value ((items.drop first).take (second - first)) =
          value (items.drop second) then [value (items.drop second)] else []

theorem mem_splitProducts (width product : Nat) (items : List Nat) :
    product ∈ splitProducts width items ↔ ∃ first second,
      1 ≤ first ∧ first < 1 + (width - 2) ∧
      first + 1 ≤ second ∧ second < first + 1 + (width - first - 1) ∧
      value (items.take first) * value ((items.drop first).take (second - first)) =
        value (items.drop second) ∧ product = value (items.drop second) := by
  simp only [splitProducts, List.mem_flatMap, List.mem_range'_1]
  constructor
  · rintro ⟨first, firstBound, second, secondBound, member⟩
    split_ifs at member with equal
    · exact ⟨first, second, firstBound.1, firstBound.2, secondBound.1,
        secondBound.2, equal, by simpa using member⟩
    · simp at member
  · rintro ⟨first, second, firstLower, firstUpper, secondLower, secondUpper, equal, same⟩
    exact ⟨first, ⟨firstLower, firstUpper⟩, second, ⟨secondLower, secondUpper⟩,
      by simp [equal, same]⟩

def permutationProducts (width : Nat) : List Nat :=
  (Permutations.rows width (List.range' 1 width)).flatMap (splitProducts width)

theorem mem_permutationProducts (width product : Nat) (decimal : width ≤ 9) :
    product ∈ permutationProducts width ↔ Identity width product := by
  have rows (items : List Nat) :
      items ∈ Permutations.rows width (List.range' 1 width) ↔ items.Perm (List.range' 1 width) := by
    simpa only [List.length_range'] using (Permutations.mem_rows_full_iff (row := items)
      (items := List.range' 1 width))
  constructor
  · intro member
    obtain ⟨items, selected, member⟩ := List.mem_flatMap.mp member
    have permuted := (rows items).mp selected
    have length : items.length = width := by simpa using permuted.length_eq
    have valid : ∀ digit ∈ items, 0 < digit ∧ digit < 10 := by
      intro digit member
      have bound := List.mem_range'_1.mp (permuted.mem_iff.mp member)
      omega
    obtain ⟨first, second, firstLower, firstUpper, secondLower, secondUpper, equal, same⟩ :=
      (mem_splitProducts width product items).mp member
    have firstInside : first < width := by omega
    have secondInside : second < width := by omega
    have ordered : first ≤ second := by omega
    have firstValid : ∀ digit ∈ items.take first, 0 < digit ∧ digit < 10 :=
      fun digit member => valid digit (List.mem_of_mem_take member)
    have secondValid : ∀ digit ∈ (items.drop first).take (second - first),
        0 < digit ∧ digit < 10 :=
      fun digit member => valid digit (List.mem_of_mem_drop (List.mem_of_mem_take member))
    have productValid : ∀ digit ∈ items.drop second, 0 < digit ∧ digit < 10 :=
      fun digit member => valid digit (List.mem_of_mem_drop member)
    have firstNonempty : items.take first ≠ [] := by
      have size : 0 < (items.take first).length := by simp [List.length_take, length]; omega
      exact List.length_pos_iff_ne_nil.mp size
    have secondNonempty : (items.drop first).take (second - first) ≠ [] := by
      have size : 0 < ((items.drop first).take (second - first)).length := by
        simp [List.length_take, List.length_drop, length]
        omega
      exact List.length_pos_iff_ne_nil.mp size
    refine ⟨value (items.take first), value ((items.drop first).take (second - first)),
      value_positive _ firstValid firstNonempty, value_positive _ secondValid secondNonempty,
      same.trans equal.symm, ?_⟩
    apply (accepted_spec width _ _).mpr
    rw [equal, digits_value _ firstValid, digits_value _ secondValid, digits_value _ productValid]
    have combined : items.take first ++ (items.drop first).take (second - first) = items.take second
      := by
      rw [← List.take_add, Nat.add_sub_cancel' ordered]
    rw [combined, List.take_append_drop]
    exact permuted
  · rintro ⟨a, b, aPositive, bPositive, same, yes⟩
    let first := (digits a).length
    let second := first + (digits b).length
    let items := digits a ++ digits b ++ digits (a * b)
    have permuted := (accepted_spec width a b).mp yes
    have length : items.length = width := by simpa [items] using permuted.length_eq
    have aLength : 0 < (digits a).length := by simpa [digits] using digits_positive a aPositive
    have bLength : 0 < (digits b).length := by simpa [digits] using digits_positive b bPositive
    have pLength : 0 < (digits (a * b)).length := by
      simpa [digits] using digits_positive (a * b) (Nat.mul_pos aPositive bPositive)
    have total : first + (digits b).length + (digits (a * b)).length = width := by
      simpa [items, first, Nat.add_assoc] using length
    have firstPart : items.take first = digits a := by simp [items, first, List.append_assoc]
    have secondPart : (items.drop first).take (second - first) = digits b := by
      simp [items, first, second, List.append_assoc]
    have productPart : items.drop second = digits (a * b) := by
      simp only [items, second, first, ← List.length_append, List.drop_left]
    apply List.mem_flatMap.mpr
    refine ⟨items, (rows items).mpr permuted, (mem_splitProducts width product items).mpr ?_⟩
    refine ⟨first, second, by omega, by omega, by omega, by omega, ?_, ?_⟩
    · simp only [firstPart, secondPart, productPart, value_digits]
    · simpa only [productPart, value_digits] using same

private theorem distinct_eraseDups (items : List Nat) : items.eraseDups.Nodup := by
  induction measure : items.length using Nat.strong_induction_on generalizing items with
  | h size ih =>
      cases items with
      | nil => simp
      | cons head tail =>
          rw [List.eraseDups_cons, List.nodup_cons]
          constructor
          · intro present
            have filtered := List.mem_filter.mp (List.mem_eraseDups.mp present)
            simpa using filtered.2
          · exact ih _ (by
              have := List.length_filter_le (fun n => !(n == head)) tail
              simp only [List.length_cons] at measure
              omega) (tail.filter (fun n => !(n == head))) rfl

private theorem sum_perm {left right : List Nat} (permuted : left.Perm right) : left.sum = right.sum
  := by
  induction permuted with
  | nil => rfl
  | cons head permuted ih => simpa only [List.sum_cons] using congrArg (Nat.add head) ih
  | swap a b tail => simp only [List.sum_cons]; omega
  | trans _ _ ihFirst ihSecond => exact ihFirst.trans ihSecond

/-- Equal product membership suffices: deduplication removes differences in traversal order. -/
theorem permutation_answer (width : Nat) (decimal : width ≤ 9) :
    (permutationProducts width).eraseDups.sum = answer width := by
  apply sum_perm
  apply (List.perm_ext_iff_of_nodup (distinct_eraseDups _) (distinct_eraseDups _)).mpr
  intro product
  simp only [List.mem_eraseDups, mem_permutationProducts width product decimal,
    mem_candidates width product]

end ProjectEulerSolutions.PandigitalProductSearch
