import solvers.«84.py.lean.d».OutcomeModel
import Mathlib.Algebra.BigOperators.Group.Finset.Sigma
import Mathlib.Algebra.BigOperators.Ring.List
import Mathlib.Algebra.Ring.Int.Defs

namespace solvers.«84».outcomes
open ProjectEulerSolutions
set_option Elab.async false

theorem per_move_sum (sides move : Nat) (lower : 2 ≤ move) (upper : move ≤ 2 * sides)
    (f : Nat → Bool → Int) :
    ((perMove sides move).map fun outcome => outcome.count * f outcome.move.toNat outcome.isDouble).sum =
      (((DiceSums.pairs sides move).filter fun pair => pair.1 ≠ pair.2).card : Int) * f move false +
      (((DiceSums.pairs sides move).filter fun pair => pair.1 = pair.2).card : Int) * f move true := by
  have split : (perMove sides move).map (fun outcome => outcome.count * f outcome.move.toNat outcome.isDouble) =
      (perMove sides move).map (fun outcome =>
        (if outcome.isDouble = false then outcome.count else 0) * f move false +
        (if outcome.isDouble = true then outcome.count else 0) * f move true) := by
    apply List.map_congr_left
    intro outcome member
    have same := (per_move_member sides move outcome member).1
    rw [same, Int.toNat_natCast]
    cases outcome.isDouble <;> simp
  rw [split, List.sum_map_add]
  rw [List.sum_map_mul_right (perMove sides move) (fun outcome => if outcome.isDouble = false then outcome.count else 0) (f move false),
    List.sum_map_mul_right (perMove sides move) (fun outcome => if outcome.isDouble = true then outcome.count else 0) (f move true)]
  rw [ordinary_weight sides move lower upper, double_weight sides move lower upper]

theorem pair_sum (sides move : Nat) (f : Nat → Bool → Int) :
    ∑ pair ∈ DiceSums.pairs sides move, f move (decide (pair.1 = pair.2)) =
      (((DiceSums.pairs sides move).filter fun pair => pair.1 ≠ pair.2).card : Int) * f move false +
      (((DiceSums.pairs sides move).filter fun pair => pair.1 = pair.2).card : Int) * f move true := by
  rw [← Finset.sum_filter_not_add_sum_filter (DiceSums.pairs sides move) (fun pair => pair.1 = pair.2)]
  have ordinary : (∑ pair ∈ (DiceSums.pairs sides move).filter (fun pair => pair.1 ≠ pair.2), f move (decide (pair.1 = pair.2))) =
      ∑ _pair ∈ (DiceSums.pairs sides move).filter (fun pair => pair.1 ≠ pair.2), f move false := by
    apply Finset.sum_congr rfl
    intro pair member
    simp [(Finset.mem_filter.mp member).2]
  have doubled : (∑ pair ∈ (DiceSums.pairs sides move).filter (fun pair => pair.1 = pair.2), f move (decide (pair.1 = pair.2))) =
      ∑ _pair ∈ (DiceSums.pairs sides move).filter (fun pair => pair.1 = pair.2), f move true := by
    apply Finset.sum_congr rfl
    intro pair member
    simp [(Finset.mem_filter.mp member).2]
  rw [ordinary, doubled]
  simp

theorem all_sum (sides : Nat) (positive : 0 < sides) (f : Nat → Bool → Int) :
    ((all sides).map fun outcome => outcome.count * f outcome.move.toNat outcome.isDouble).sum =
      ∑ pair ∈ (Finset.Icc 1 sides) ×ˢ (Finset.Icc 1 sides), f (pair.1 + pair.2) (decide (pair.1 = pair.2)) := by
  have interval : (List.range' 2 (2 * sides - 1)).toFinset = Finset.Icc 2 (2 * sides) := by
    ext move
    simp only [List.mem_toFinset, List.mem_range', Finset.mem_Icc, Nat.one_mul]
    constructor
    · rintro ⟨index, inside, rfl⟩
      omega
    · intro bounds
      exact ⟨move - 2, by omega, by omega⟩
  have grouped : ((all sides).map fun outcome => outcome.count * f outcome.move.toNat outcome.isDouble).sum =
      ∑ move ∈ Finset.Icc 2 (2 * sides), ∑ pair ∈ DiceSums.pairs sides move, f move (decide (pair.1 = pair.2)) := by
    unfold all
    rw [List.map_flatMap]
    simp only [List.flatMap_def, List.sum_flatten, List.map_map, Function.comp_def]
    rw [← List.sum_toFinset _ (List.nodup_range'), interval]
    apply Finset.sum_congr rfl
    intro move member
    rw [per_move_sum sides move (Finset.mem_Icc.mp member).1 (Finset.mem_Icc.mp member).2, pair_sum]
  rw [grouped]
  have fibers := Finset.sum_fiberwise_of_maps_to
    (s := (Finset.Icc 1 sides) ×ˢ (Finset.Icc 1 sides)) (t := Finset.Icc 2 (2 * sides))
    (g := fun pair : Nat × Nat => pair.1 + pair.2)
    (by intro pair member; simp only [Finset.mem_product, Finset.mem_Icc] at member ⊢; omega)
    (fun pair => f (pair.1 + pair.2) (decide (pair.1 = pair.2)))
  rw [← fibers]
  apply Finset.sum_congr rfl
  intro move _
  apply Finset.sum_congr rfl
  intro pair member
  rw [(Finset.mem_filter.mp member).2]

theorem faces_sum (sides : Nat) (f : Nat → Int) :
    ∑ face ∈ Finset.Icc 1 sides, f face = ((List.range sides).map fun face => f (face + 1)).sum := by
  rw [← List.toFinset_range'_1_1, List.sum_toFinset _ List.nodup_range', List.range'_eq_map_range, List.map_map]
  simp [Function.comp_def, Nat.add_comm]

theorem all_sum_list (sides : Nat) (positive : 0 < sides) (f : Nat → Bool → Int) :
    ((all sides).map fun outcome => outcome.count * f outcome.move.toNat outcome.isDouble).sum =
      ((List.range sides).map fun first => ((List.range sides).map fun second =>
        f ((first + 1) + (second + 1)) (decide (first + 1 = second + 1))).sum).sum := by
  rw [all_sum sides positive f, Finset.sum_product, faces_sum]
  simp_rw [faces_sum]

end solvers.«84».outcomes
