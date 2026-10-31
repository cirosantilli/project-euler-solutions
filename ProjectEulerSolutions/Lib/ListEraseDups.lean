import Mathlib.Data.Finset.Card
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-! First-occurrence deduplication agrees with finite-set cardinality and summation. -/
namespace ProjectEulerSolutions.ListEraseDups
set_option Elab.async false

theorem nodup {α : Type*} [BEq α] [LawfulBEq α] (items : List α) : items.eraseDups.Nodup := by
  cases items with
  | nil => simp
  | cons head tail =>
      rw [List.eraseDups_cons, List.nodup_cons]
      exact ⟨by simp [List.mem_eraseDups], nodup (tail.filter (fun x => !(x == head)))⟩
termination_by items.length
decreasing_by
  have := List.length_filter_le (fun x => !(x == head)) tail
  simp only [List.length_cons]
  omega

theorem length_eq_card {α : Type*} [DecidableEq α] (items : List α) :
    items.eraseDups.length = items.toFinset.card := by
  have same : items.eraseDups.toFinset = items.toFinset := by
    ext value
    simp [List.mem_eraseDups]
  rw [← List.toFinset_card_of_nodup (nodup items), same]

theorem sum_eq (items : List Nat) :
    items.eraseDups.sum = items.toFinset.sum id := by
  have same : items.eraseDups.toFinset = items.toFinset := by
    ext value
    simp [List.mem_eraseDups]
  simpa only [same, List.map_id] using
    (List.sum_toFinset id (nodup items)).symm

end ProjectEulerSolutions.ListEraseDups
