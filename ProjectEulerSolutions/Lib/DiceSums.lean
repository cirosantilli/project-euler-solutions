import Mathlib.Data.Finset.Card
import Mathlib.Data.Finset.Prod
import Mathlib.Order.Interval.Finset.Nat

/-! Exact counts of ordered dice pairs, separated by sum and doubles. -/
namespace ProjectEulerSolutions.DiceSums
set_option Elab.async false

def pairs (sides total : Nat) : Finset (Nat × Nat) :=
  ((Finset.Icc 1 sides) ×ˢ (Finset.Icc 1 sides)).filter fun pair => pair.1 + pair.2 = total

theorem pairs_eq_image (sides total : Nat) (lower : 2 ≤ total) :
    pairs sides total = (Finset.Icc (max 1 (total - sides)) (min sides (total - 1))).image
      (fun first => (first, total - first)) := by
  ext ⟨first, second⟩
  simp only [pairs, Finset.mem_filter, Finset.mem_product, Finset.mem_Icc, Finset.mem_image]
  constructor
  · rintro ⟨⟨⟨firstLower, firstUpper⟩, ⟨secondLower, secondUpper⟩⟩, sum⟩
    refine ⟨first, ⟨by omega, by omega⟩, ?_⟩
    exact Prod.ext rfl (by omega)
  · rintro ⟨value, ⟨lo, hi⟩, equal⟩
    have firstEqual := congrArg Prod.fst equal
    have secondEqual := congrArg Prod.snd equal
    dsimp at firstEqual secondEqual
    exact ⟨⟨⟨by omega, by omega⟩, ⟨by omega, by omega⟩⟩, by omega⟩

/-- The triangular dice-sum counts used by the optimized transition builder. -/
theorem pairs_card (sides total : Nat) (lower : 2 ≤ total) (upper : total ≤ 2 * sides) :
    (pairs sides total).card = min (total - 1) (2 * sides + 1 - total) := by
  rw [pairs_eq_image sides total lower,
    Finset.card_image_of_injective _ (fun _ _ equal => congrArg Prod.fst equal), Nat.card_Icc]
  omega

theorem doubles_eq (sides total : Nat) (lower : 2 ≤ total) (upper : total ≤ 2 * sides) :
    (pairs sides total).filter (fun pair => pair.1 = pair.2) =
      if total % 2 = 0 then {(total / 2, total / 2)} else ∅ := by
  ext ⟨first, second⟩
  by_cases even : total % 2 = 0
  · simp only [if_pos even, Finset.mem_filter, pairs, Finset.mem_product, Finset.mem_Icc,
      Finset.mem_singleton, Prod.mk.injEq]
    omega
  · simp only [if_neg even, Finset.mem_filter, pairs, Finset.mem_product, Finset.mem_Icc,
      Finset.notMem_empty, iff_false]
    omega

theorem doubles_card (sides total : Nat) (lower : 2 ≤ total) (upper : total ≤ 2 * sides) :
    ((pairs sides total).filter (fun pair => pair.1 = pair.2)).card =
      if total % 2 = 0 then 1 else 0 := by
  rw [doubles_eq sides total lower upper]
  split <;> simp

/-- Removing the unique even-sum double accounts for all ordinary rolls. -/
theorem non_doubles_card (sides total : Nat) (lower : 2 ≤ total) (upper : total ≤ 2 * sides) :
    ((pairs sides total).filter (fun pair => pair.1 ≠ pair.2)).card =
      min (total - 1) (2 * sides + 1 - total) - (if total % 2 = 0 then 1 else 0) := by
  have partition := Finset.card_filter_add_card_filter_not (s := pairs sides total) (p := fun pair => pair.1 = pair.2)
  rw [pairs_card sides total lower upper, doubles_card sides total lower upper] at partition
  rw [← partition, Nat.add_sub_cancel_left]

end ProjectEulerSolutions.DiceSums
