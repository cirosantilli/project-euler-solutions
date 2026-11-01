import ProjectEulerSolutions.Lib.IndexedFilter
import Mathlib.Data.List.Perm.Basic

/-! Removing two operand occurrences decreases the size of a pair-reduction state. -/
namespace ProjectEulerSolutions.IndexedFilter
set_option Elab.async false

theorem length_add_one_le (keep : Nat → Bool) (first : Nat) (values : List α) (index : Nat)
    (lower : first ≤ index) (upper : index < first + values.length) (excluded : keep index = false) :
    (select keep first values).length + 1 ≤ values.length := by
  induction values generalizing first with
  | nil => simp at upper; omega
  | cons value rest ih =>
    by_cases equal : first = index
    · subst first
      simpa [select, excluded] using Nat.add_le_add_right (length_le keep (index + 1) rest) 1
    · have shorter := ih (first + 1) (by omega) (by simp only [List.length_cons] at upper; omega)
      cases selected : keep first <;> simp only [select, selected, Bool.false_eq_true, if_false, if_true, List.length_cons] <;> omega

theorem length_add_two_le (keep : Nat → Bool) (first : Nat) (values : List α) (i j : Nat)
    (lower : first ≤ i) (ordered : i < j) (upper : j < first + values.length)
    (excludeI : keep i = false) (excludeJ : keep j = false) :
    (select keep first values).length + 2 ≤ values.length := by
  induction values generalizing first with
  | nil => simp at upper; omega
  | cons value rest ih =>
    by_cases equal : first = i
    · subst first
      have shorter := length_add_one_le keep (i + 1) rest j (by omega) (by simp only [List.length_cons] at upper; omega) excludeJ
      simp only [select, excludeI, Bool.false_eq_true, if_false, List.length_cons]
      omega
    · have shorter := ih (first + 1) (by omega) (by simp only [List.length_cons] at upper; omega)
      cases selected : keep first <;> simp only [select, selected, Bool.false_eq_true, if_false, if_true, List.length_cons] <;> omega

theorem select_append (keep : Nat → Bool) (first : Nat) (left right : List α) :
    select keep first (left ++ right) = select keep first left ++ select keep (first + left.length) right := by
  induction left generalizing first with
  | nil => simp [select]
  | cons value rest ih =>
    simp only [List.cons_append, select, List.length_cons]
    split <;> simp [ih, Nat.add_comm, Nat.add_left_comm]

theorem select_all (keep : Nat → Bool) (first : Nat) (values : List α)
    (included : ∀ index, first ≤ index → index < first + values.length → keep index = true) :
    select keep first values = values := by
  induction values generalizing first with
  | nil => rfl
  | cons value rest ih =>
    have head := included first (by omega) (by simp)
    have tail := ih (first + 1) (by intro index lower upper; apply included index <;> simp_all only [List.length_cons] <;> omega)
    simp [select, head, tail]

def exceptTwo (i j index : Nat) : Bool := decide (index ≠ i ∧ index ≠ j)

theorem pair_replacement_shorter (values : List α) (i j : Nat) (ordered : i < j) (upper : j < values.length) :
    (select (exceptTwo i j) 0 values).length + 1 < values.length := by
  have shorter := length_add_two_le (exceptTwo i j) 0 values i j (by omega) ordered (by simpa using upper)
    (by simp [exceptTwo]) (by simp [exceptTwo])
  omega

/-- Positions distinguish the two chosen occurrences even when their values are equal. -/
theorem select_two_parts (before middle after : List α) (a b : α) :
    select (exceptTwo before.length (before.length + 1 + middle.length)) 0
      (before ++ a :: (middle ++ b :: after)) = before ++ middle ++ after := by
  let keep := exceptTwo before.length (before.length + 1 + middle.length)
  have leading : select keep 0 before = before := select_all keep 0 before (by
    intro index lower upper
    simp only [keep, exceptTwo, decide_eq_true_eq]
    omega)
  have between : select keep (before.length + 1) middle = middle := select_all keep _ middle (by
    intro index lower upper
    simp only [keep, exceptTwo, decide_eq_true_eq]
    omega)
  have suffix : select keep (before.length + 1 + middle.length + 1) after = after := select_all keep _ after (by
    intro index lower upper
    simp only [keep, exceptTwo, decide_eq_true_eq]
    omega)
  change select keep 0 (before ++ a :: (middle ++ b :: after)) = _
  rw [select_append, leading]
  simp only [Nat.zero_add, select, show keep before.length = false by simp [keep, exceptTwo],
    Bool.false_eq_true, if_false]
  rw [select_append, between]
  simp only [select, show keep (before.length + 1 + middle.length) = false by simp [keep, exceptTwo],
    Bool.false_eq_true, if_false, suffix, List.append_assoc]

/-- Split at two positions, without assuming that any element values are distinct. -/
theorem split_two (values : List α) (i j : Nat) (ordered : i < j) (upper : j < values.length) :
    ∃ before middle after,
      values = before ++ values[i]'(by omega) :: (middle ++ values[j] :: after) ∧
      before.length = i ∧ before.length + 1 + middle.length = j := by
  induction values generalizing i j with
  | nil => simp at upper
  | cons value rest ih =>
    cases j with
    | zero => omega
    | succ j =>
      have jInside : j < rest.length := by simpa using upper
      cases i with
      | zero =>
        refine ⟨[], rest.take j, rest.drop (j + 1), ?_, rfl, ?_⟩
        · have split : rest = rest.take j ++ rest[j] :: rest.drop (j + 1) := by
            calc
              rest = rest.take j ++ rest.drop j := (List.take_append_drop j rest).symm
              _ = _ := by rw [List.drop_eq_getElem_cons jInside]
          simpa only [List.nil_append, List.getElem_cons_zero, List.getElem_cons_succ] using congrArg (value :: ·) split
        · simp [List.length_take, Nat.min_eq_left (Nat.le_of_lt jInside), Nat.add_comm]
      | succ i =>
        obtain ⟨before, middle, after, split, first, second⟩ := ih i j (by omega) jInside
        refine ⟨value :: before, middle, after, ?_, by simp [first], ?_⟩
        · simpa only [List.cons_append, List.getElem_cons_succ] using congrArg (value :: ·) split
        · simp only [List.length_cons]
          omega

/-- Moving the selected occurrences to the front leaves exactly the source comprehension's remainder. -/
theorem select_two_perm (values : List α) (i j : Nat) (ordered : i < j) (upper : j < values.length) :
    values.Perm (values[i]'(by omega) :: values[j] :: select (exceptTwo i j) 0 values) := by
  obtain ⟨before, middle, after, split, first, second⟩ := split_two values i j ordered upper
  have selected := select_two_parts before middle after values[i] values[j]
  rw [← split] at selected
  rw [second, first] at selected
  rw [selected]
  conv_lhs => rw [split]
  apply List.perm_middle.trans
  simpa only [List.append_assoc] using
    (List.perm_middle (a := values[j]) (l₁ := before ++ middle) (l₂ := after)).cons values[i]

theorem select_two_length (values : List α) (i j : Nat) (ordered : i < j) (upper : j < values.length) :
    (select (exceptTwo i j) 0 values).length + 2 = values.length := by
  have size := (select_two_perm values i j ordered upper).length_eq
  simp only [List.length_cons] at size
  omega

end ProjectEulerSolutions.IndexedFilter
