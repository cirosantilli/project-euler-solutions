import Mathlib.Data.Int.Order.Basic
import Mathlib.Data.List.GetD

/-! Parent order in an array-backed binary heap implies a globally minimal root. -/
namespace ProjectEulerSolutions.BinaryHeapOrder

variable {α : Type}

def Ordered (priority : α → Int) (values : List α) : Prop :=
  ∀ (i : Nat) (inside : i < values.length), 0 < i →
    priority (values[(i - 1) / 2]'(by have := Nat.div_le_self (i - 1) 2; omega)) ≤ priority values[i]

theorem root_le_index (priority : α → Int) (values : List α) (ordered : Ordered priority values)
    (nonempty : 0 < values.length) (i : Nat) (inside : i < values.length) :
    priority values[0] ≤ priority values[i] := by
  by_cases zero : i = 0
  · subst i
    exact le_refl _
  · have smaller : (i - 1) / 2 < i := by have := Nat.div_le_self (i - 1) 2; omega
    have parent := root_le_index priority values ordered nonempty ((i - 1) / 2) (by omega)
    exact Int.le_trans parent (ordered i inside (by omega))
termination_by i

theorem root_le_member (priority : α → Int) (root : α) (rest : List α)
    (ordered : Ordered priority (root :: rest)) (value : α) (member : value ∈ root :: rest) :
    priority root ≤ priority value := by
  obtain ⟨i, inside, equal⟩ := List.getElem_of_mem member
  have bound := root_le_index priority (root :: rest) ordered (by simp) i inside
  simpa only [List.getElem_cons_zero, equal] using bound

theorem ordered_nil (priority : α → Int) : Ordered priority [] := by
  intro i inside
  simp at inside

theorem ordered_singleton (priority : α → Int) (value : α) : Ordered priority [value] := by
  intro i inside positive
  simp only [List.length_singleton] at inside
  omega

/-- Removing the last entry leaves every surviving parent edge unchanged. -/
theorem ordered_prefix (priority : α → Int) (front suffix : List α)
    (ordered : Ordered priority (front ++ suffix)) : Ordered priority front := by
  intro i inside positive
  have parentInside : (i - 1) / 2 < front.length := by
    have := Nat.div_le_self (i - 1) 2
    omega
  have edge := ordered i (by simp only [List.length_append]; omega) positive
  simpa only [List.getElem_append_left inside, List.getElem_append_left parentInside] using edge

end ProjectEulerSolutions.BinaryHeapOrder
