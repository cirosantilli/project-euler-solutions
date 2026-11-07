import ProjectEulerSolutions.Lib.MonopolyCards
import Mathlib.Algebra.BigOperators.Group.List.Basic

/-! Exact aggregation of the reference's nested dice and card enumeration. -/
namespace ProjectEulerSolutions.Monopoly
set_option Elab.async false

theorem expandWeights_flatMap (values : List α) (weights : α → List (β × Nat)) :
    expandWeights (values.flatMap weights) = values.flatMap (fun value => expandWeights (weights value)) := by
  simp only [expandWeights, List.flatMap_assoc]

theorem sum_expandWeights (weights : List (α × Nat)) (contribution : α → Int) :
    ((expandWeights weights).map contribution).sum =
      (weights.map fun entry => (entry.2 : Int) * contribution entry.1).sum := by
  induction weights with
  | nil => rfl
  | cons first rest ih =>
      simp only [expandWeights, List.flatMap_cons, List.map_append, List.sum_append,
        List.map_replicate, List.sum_replicate, nsmul_eq_mul, List.map_cons, List.sum_cons]
      exact congrArg ((first.2 : Int) * contribution first.1 + ·) ih

theorem rollCards_sum (state : Nat × Nat) (first second : Nat) (contribution : (Nat × Nat) → Int) :
    ((rollCards state first second).map contribution).sum =
      ((rollWeights state first second).map fun entry => (entry.2 : Int) * contribution entry.1).sum :=
  ((rollCards_perm state first second).map contribution).sum_eq.trans (sum_expandWeights _ contribution)

def turnCards (sides : Nat) (state : Nat × Nat) : List (Nat × Nat) :=
  (List.range sides).flatMap fun first =>
    (List.range sides).flatMap fun second => rollCards state (first + 1) (second + 1)

/-- Every ordered dice pair and independent card pair appears with exactly its weighted multiplicity. -/
theorem turnCards_perm (sides : Nat) (state : Nat × Nat) :
    (turnCards sides state).Perm (expandWeights (turnWeights sides state)) := by
  unfold turnCards turnWeights
  rw [expandWeights_flatMap]
  apply List.Perm.flatMap_left
  intro first _
  rw [expandWeights_flatMap]
  exact List.Perm.flatMap_left _ (fun second _ => rollCards_perm state (first + 1) (second + 1))

/-- This holds for every contribution function, hence for every destination counter in a transition row. -/
theorem turnCards_sum (sides : Nat) (state : Nat × Nat) (contribution : (Nat × Nat) → Int) :
    ((turnCards sides state).map contribution).sum =
      ((turnWeights sides state).map fun entry => (entry.2 : Int) * contribution entry.1).sum :=
  ((turnCards_perm sides state).map contribution).sum_eq.trans (sum_expandWeights _ contribution)

theorem turnCards_bound (sides : Nat) (state after : Nat × Nat) (doublesBound : state.2 < 3)
    (member : after ∈ turnCards sides state) : after.1 < 40 ∧ after.2 < 3 := by
  obtain ⟨first, _, later⟩ := List.mem_flatMap.mp member
  obtain ⟨second, _, rolled⟩ := List.mem_flatMap.mp later
  rw [← rollCards_enumerated] at rolled
  obtain ⟨community, _, later⟩ := List.mem_flatMap.mp rolled
  obtain ⟨chance, chanceIn, rfl⟩ := List.mem_map.mp later
  exact advanceCards_bound state (first + 1) (second + 1) community chance doublesBound (List.mem_range.mp chanceIn)

end ProjectEulerSolutions.Monopoly
