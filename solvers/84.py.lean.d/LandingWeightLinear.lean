import solvers.«84.py.lean.d».LandingResult
import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise
import Mathlib.Algebra.Ring.Int.Defs

namespace solvers.«84».resolve_landing.weights
set_option Elab.async false

theorem total_linear (keys : Finset (Int × Bool)) (data : List CardData) (f : Int × Bool → Int)
    (supported : ∀ item ∈ data, item.1 ∈ keys) :
    (∑ landing ∈ keys, total data landing * f landing) = (data.map fun item => item.2 * f item.1).sum := by
  induction data with
  | nil => simp [total]
  | cons item rest ih =>
      have member := supported item (by simp)
      have tail := ih (fun current allowed => supported current (by simp [allowed]))
      change (∑ landing ∈ keys, ((if landing = item.1 then item.2 else 0) + total rest landing) * f landing) =
        item.2 * f item.1 + (rest.map fun current => current.2 * f current.1).sum
      simp_rw [Int.add_mul, ite_mul, zero_mul]
      rw [Finset.sum_add_distrib, tail]
      simp [member]

/-- Equal landing totals give equal weighted sums, even when lists aggregate duplicate keys differently. -/
theorem linear_equiv (left right : List CardData) (equal : ∀ landing, total left landing = total right landing)
    (f : Int × Bool → Int) :
    (left.map fun item => item.2 * f item.1).sum = (right.map fun item => item.2 * f item.1).sum := by
  let keys := ((left ++ right).map Prod.fst).toFinset
  have leftSupport : ∀ item ∈ left, item.1 ∈ keys := by
    intro item member
    exact List.mem_toFinset.mpr (List.mem_map.mpr ⟨item, List.mem_append_left _ member, rfl⟩)
  have rightSupport : ∀ item ∈ right, item.1 ∈ keys := by
    intro item member
    exact List.mem_toFinset.mpr (List.mem_map.mpr ⟨item, List.mem_append_right _ member, rfl⟩)
  rw [← total_linear keys left f leftSupport, ← total_linear keys right f rightSupport]
  apply Finset.sum_congr rfl
  intro landing _
  rw [equal]

theorem entries_linear (entries : List Entry) (data : List CardData) (distinct : Distinct entries)
    (counts : ∀ landing, (lookup entries landing).getD 0 = total data landing) (f : Int × Bool → Int) :
    (entries.map fun entry => entry.weight * f entry.landing).sum = (data.map fun item => item.2 * f item.1).sum := by
  have equal : ∀ landing, total (entries.map fun entry => (entry.landing, entry.weight)) landing = total data landing := by
    intro landing
    simpa only [total, List.map_map, Function.comp_def] using (lookup_sum entries landing distinct).symm.trans (counts landing)
  simpa only [List.map_map, Function.comp_def] using linear_equiv _ data equal f

end solvers.«84».resolve_landing.weights
