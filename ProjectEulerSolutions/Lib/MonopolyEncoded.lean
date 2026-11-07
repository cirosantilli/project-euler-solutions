import ProjectEulerSolutions.Lib.MonopolyStationary
import Mathlib.Logic.Equiv.Fin.Basic

/-! The finite state model in the integer index order used by both Python solvers. -/
namespace ProjectEulerSolutions.Monopoly
open scoped BigOperators
set_option Elab.async false

def stateIndex : State ≃ Fin 120 := finProdFinEquiv

theorem stateIndex_val (state : State) : (stateIndex state).val = encode state.1.val state.2.val := by
  simp [stateIndex, finProdFinEquiv, encode, Nat.mul_comm, Nat.add_comm]

theorem stateIndex_decode (index : Fin 120) :
    stateValues (stateIndex.symm index) = (index.val / 3, index.val % 3) := rfl

def encodedWeight (sides : Nat) (before after : Fin 120) : Int :=
  ((turnWeights sides (before.val / 3, before.val % 3)).map fun entry =>
    if after.val = encode entry.1.1 entry.1.2 then (entry.2 : Int) else 0).sum

/-- Integer dictionary keys identify precisely the same destinations as finite state pairs. -/
theorem encodedWeight_eq (sides : Nat) (before after : Fin 120) :
    (encodedWeight sides before after : ℚ) = matrix sides (stateIndex.symm before) (stateIndex.symm after) := by
  unfold encodedWeight matrix
  rw [stateIndex_decode, stateIndex_decode]
  push_cast
  apply congrArg List.sum
  simp only [List.map_map, Function.comp_def]
  apply List.map_congr_left
  intro entry member
  have bounds := turnWeights_bound sides (before.val / 3, before.val % 3) (Nat.mod_lt _ (by decide)) entry member
  have selected : after.val = encode entry.1.1 entry.1.2 ↔ entry.1 = (after.val / 3, after.val % 3) := by
    constructor
    · intro same
      rw [same]
      exact (decode_encode entry.1.1 entry.1.2 bounds.2).symm
    · intro same
      rw [same]
      unfold encode
      omega
  simp [selected]

theorem encodedWeight_nonnegative (sides : Nat) (before after : Fin 120) : 0 ≤ encodedWeight sides before after := by
  exact_mod_cast (encodedWeight_eq sides before after).symm ▸ matrix_nonnegative sides (stateIndex.symm before) (stateIndex.symm after)

theorem encodedWeight_row_sum (sides : Nat) (before : Fin 120) :
    ∑ after, encodedWeight sides before after = 256 * (sides : Int) * sides := by
  have total := matrix_row_sum sides (stateIndex.symm before)
  have reorder := stateIndex.symm.sum_comp (fun after => matrix sides (stateIndex.symm before) after)
  rw [← reorder] at total
  simp only [← encodedWeight_eq] at total
  exact_mod_cast total

end ProjectEulerSolutions.Monopoly
