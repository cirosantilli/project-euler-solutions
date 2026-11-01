import Mathlib.Data.List.Forall2

/-! Filtering by position preserves occurrence order and multiplicity. -/
namespace ProjectEulerSolutions.IndexedFilter
set_option Elab.async false

def select (keep : Nat → Bool) (first : Nat) : List α → List α
  | [] => []
  | value :: rest =>
      if keep first then value :: select keep (first + 1) rest else select keep (first + 1) rest

theorem length_le (keep : Nat → Bool) (first : Nat) (values : List α) :
    (select keep first values).length ≤ values.length := by
  induction values generalizing first with
  | nil => simp [select]
  | cons value rest ih =>
    simp only [select]
    split <;> simp only [List.length_cons] <;> have := ih (first + 1) <;> omega

theorem forall₂ {relation : α → β → Prop} {left : List α} {right : List β}
    (rows : List.Forall₂ relation left right) (keep : Nat → Bool) (first : Nat) :
    List.Forall₂ relation (select keep first left) (select keep first right) := by
  induction rows generalizing first with
  | nil => exact .nil
  | cons row rows ih =>
    simp only [select]
    split
    · exact .cons row (ih (first + 1))
    · exact ih (first + 1)

theorem select_map (keep : Nat → Bool) (first : Nat) (values : List α) (transform : α → β) :
    select keep first (values.map transform) = (select keep first values).map transform := by
  induction values generalizing first with
  | nil => rfl
  | cons value rest ih =>
    simp only [List.map_cons, select]
    split <;> simp [ih]

theorem sublist (keep : Nat → Bool) (first : Nat) (values : List α) :
    (select keep first values).Sublist values := by
  induction values generalizing first with
  | nil => exact .slnil
  | cons value rest ih =>
    simp only [select]
    split
    · exact .cons_cons value (ih (first + 1))
    · exact .cons value (ih (first + 1))

end ProjectEulerSolutions.IndexedFilter
