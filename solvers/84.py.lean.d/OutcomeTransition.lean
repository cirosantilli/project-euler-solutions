import solvers.«84.py.lean.d».OutcomeGrouping
import ProjectEulerSolutions.Lib.Monopoly

namespace solvers.«84».outcomes
open ProjectEulerSolutions
set_option Elab.async false

def categoryWeights (state : Nat × Nat) (move : Nat) (isDouble : Bool) : List ((Nat × Nat) × Nat) :=
  if isDouble ∧ state.2 = 2 then [((10, 0), 256)]
  else (Monopoly.landingWeights ((state.1 + move) % 40)).map fun (landing, weight) =>
    ((landing.1, if landing.2 then 0 else if isDouble then state.2 + 1 else 0), weight)

theorem category_roll (state : Nat × Nat) (first second : Nat) :
    categoryWeights state (first + second) (decide (first = second)) = Monopoly.rollWeights state first second := by
  by_cases same : first = second <;> simp [categoryWeights, Monopoly.rollWeights, same, Nat.add_assoc]

def contribution (state : Nat × Nat) (move : Nat) (isDouble : Bool) (target : Nat × Nat) : Int :=
  ((categoryWeights state move isDouble).map fun entry => if entry.1 = target then (entry.2 : Int) else 0).sum

/-- Grouping by sum and doubles preserves each transition entry, including third-double jail. -/
theorem transition_weight (sides : Nat) (positive : 0 < sides) (state target : Nat × Nat) :
    ((all sides).map fun outcome => outcome.count * contribution state outcome.move.toNat outcome.isDouble target).sum =
      ((Monopoly.turnWeights sides state).map fun entry => if entry.1 = target then (entry.2 : Int) else 0).sum := by
  rw [all_sum_list sides positive (fun move flag => contribution state move flag target)]
  simp only [contribution, category_roll, Monopoly.turnWeights, List.flatMap_def,
    List.map_flatten, List.sum_flatten, List.map_map, Function.comp_def]

theorem category_total (state : Nat × Nat) (move : Nat) (isDouble : Bool) :
    ((categoryWeights state move isDouble).map Prod.snd).sum = 256 := by
  unfold categoryWeights
  split
  · rfl
  · simpa only [List.map_map, Function.comp_def] using Monopoly.landingWeights_total ((state.1 + move) % 40)

end solvers.«84».outcomes
