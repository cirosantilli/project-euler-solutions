import ProjectEulerSolutions.Lib.PolygonalCycle
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

/-! Complete four-digit polygonal families, with bounds derived from their growth. -/
namespace ProjectEulerSolutions.PolygonalFamilies
open ProjectEulerSolutions.PolygonalCycle (polygonal)

set_option Elab.async false

def accepted (value : Int) : Prop := 1000 ≤ value ∧ value < 10000
instance (value : Int) : Decidable (accepted value) := inferInstanceAs (Decidable (_ ∧ _))

def family (sides : Int) : List Int :=
  ((List.range' 1 140).map (fun (index : Nat) => polygonal sides index)).filter
    (fun value => decide (accepted value))

theorem step (sides index : Int) :
    polygonal sides (index + 1) = polygonal sides index + (sides - 2) * index + 1 := by
  unfold polygonal
  have numerator : (index + 1) * ((sides - 2) * (index + 1) - (sides - 4)) =
      index * ((sides - 2) * index - (sides - 4)) + ((sides - 2) * index + 1) * 2 := by ring
  rw [numerator, Int.add_mul_ediv_right _ _ (by decide : (2 : Int) ≠ 0)]
  ring

theorem increasing_step (sides index : Int) (valid : 3 ≤ sides) (positive : 1 ≤ index) :
    polygonal sides index < polygonal sides (index + 1) := by
  rw [step]
  have product : 0 ≤ (sides - 2) * index := mul_nonneg (by omega) (by omega)
  omega

theorem monotone_from (sides index : Int) (valid : 3 ≤ sides) (positive : 1 ≤ index) (offset : Nat) :
    polygonal sides index ≤ polygonal sides (index + offset) := by
  induction offset with
  | zero => simp
  | succ offset ih =>
      have next := increasing_step sides (index + offset) valid (by omega)
      exact ih.trans (by simpa [Nat.cast_add, Nat.cast_one, add_assoc] using next.le)

theorem above_four_digits (sides index : Int) (valid : 3 ≤ sides) (large : 141 ≤ index) :
    10000 ≤ polygonal sides index := by
  have extra : 0 ≤ (sides - 3) * (index * (index - 1)) :=
    mul_nonneg (by omega) (mul_nonneg (by omega) (by omega))
  have triangle : (20000 : Int) ≤ index * (index + 1) := by
    nlinarith [sq_nonneg (index - 141)]
  have numerator : (20000 : Int) ≤ index * ((sides - 2) * index - (sides - 4)) := by
    nlinarith [extra]
  unfold polygonal
  omega

/-- The index window omits no positive-index four-digit value in any polygonal family. -/
theorem mem_family (sides value : Int) (valid : 3 ≤ sides) :
    value ∈ family sides ↔ ∃ index : Nat, 1 ≤ index ∧ polygonal sides index = value ∧ accepted value := by
  simp only [family, List.mem_filter, decide_eq_true_eq, List.mem_map, List.mem_range'_1]
  constructor
  · rintro ⟨⟨index, bound, equal⟩, keep⟩
    exact ⟨index, bound.1, equal, keep⟩
  · rintro ⟨index, positive, equal, keep⟩
    have upper : index < 141 := by
      by_contra outside
      have above := above_four_digits sides index valid (by omega)
      rw [equal] at above
      exact (not_le_of_gt keep.2) above
    exact ⟨⟨index, ⟨positive, by omega⟩, equal⟩, keep⟩

def emitted (value : Int) : List Int := if accepted value then [value] else []

def collect : Nat → Int → Int → List Int
  | 0, _, _ => []
  | count + 1, sides, index =>
      let value := polygonal sides index
      if 10000 ≤ value then [] else emitted value ++ collect count sides (index + 1)

theorem collect_window (count : Nat) (sides index : Int) (valid : 3 ≤ sides) (positive : 1 ≤ index) :
    collect count sides index =
      ((List.range count).map (fun (offset : Nat) => polygonal sides (index + offset))).filter
        (fun value => decide (accepted value)) := by
  induction count generalizing index with
  | zero => rfl
  | succ count ih =>
      have mapped : (List.range (count + 1)).map (fun (offset : Nat) => polygonal sides (index + offset)) =
          polygonal sides index ::
            (List.range count).map (fun (offset : Nat) => polygonal sides (index + 1 + offset)) := by
        rw [List.range_succ_eq_map, List.map_cons, List.map_map]
        simp only [Nat.cast_zero, add_zero]
        congr 1
        apply List.map_congr_left
        intro offset _
        dsimp
        congr 1
        ring
      rw [mapped]
      by_cases done : 10000 ≤ polygonal sides index
      · have rejected (offset : Nat) : ¬ accepted (polygonal sides (index + offset)) := by
          have above := done.trans (monotone_from sides index valid positive offset)
          intro keep
          exact (not_le_of_gt keep.2) above
        have first : ¬ accepted (polygonal sides index) := by simpa using rejected 0
        have tail :
            ((List.range count).map (fun (offset : Nat) => polygonal sides (index + 1 + offset))).filter
              (fun value => decide (accepted value)) = [] := by
          apply List.filter_eq_nil_iff.mpr
          intro value member
          obtain ⟨offset, _, rfl⟩ := List.mem_map.mp member
          have absent : ¬ accepted (polygonal sides (index + 1 + offset)) := by
            simpa [Nat.cast_add, Nat.cast_one, add_assoc, add_comm, add_left_comm] using rejected (offset + 1)
          simpa using absent
        simp [collect, done, first, tail]
      · rw [collect, if_neg done, ih (index + 1) (by omega)]
        by_cases keep : accepted (polygonal sides index) <;> simp [emitted, keep]

theorem collect_family (sides : Int) (valid : 3 ≤ sides) : collect 140 sides 1 = family sides := by
  rw [collect_window 140 sides 1 valid (by decide)]
  unfold family
  rw [List.range'_eq_map_range, List.map_map]
  congr 1

end ProjectEulerSolutions.PolygonalFamilies
