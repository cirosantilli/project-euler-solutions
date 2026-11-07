import ProjectEulerSolutions.Lib.Monopoly
import ProjectEulerSolutions.Lib.StationarySystem
import Mathlib.Data.Fintype.Prod

/-! The finite transition matrix and uniqueness of the uniform-card stationary distribution. -/
namespace ProjectEulerSolutions.Monopoly
open scoped BigOperators
set_option Elab.async false

theorem moveWeights_bound (square : Nat) (bound : square < 40) :
    ∀ entry ∈ moveWeights square, entry.1.1 < 40 := by
  by_cases cc : square = 33 <;> simp [moveWeights, cc, bound]

theorem destinations_bound (square : Nat) (bound : square < 40) :
    ∀ destination ∈ destinations square, destination < 40 := by
  have railway := CyclicStops.nextStop_mem [5, 15, 25, 35] 5 square
  have utility := CyclicStops.nextStop_mem [12, 28] 12 square
  simp only [List.mem_cons, List.not_mem_nil, or_false] at railway utility
  have railwayBound : (CyclicStops.nextStop [5, 15, 25, 35] 5 square).toNat < 40 := by
    rcases railway with h | h | h | h | h <;> rw [h] <;> decide
  have utilityBound : (CyclicStops.nextStop [12, 28] 12 square).toNat < 40 := by
    rcases utility with h | h | h <;> rw [h] <;> decide
  simp [destinations, railwayBound, utilityBound, show square - 3 < 40 by omega]

theorem landingWeights_bound (square : Nat) (bound : square < 40) :
    ∀ entry ∈ landingWeights square, entry.1.1 < 40 := by
  unfold landingWeights
  split
  · simp
  · split
    · simpa using bound
    · split
      · intro entry member
        rcases List.mem_append.mp member with initial | moved
        · simp only [List.mem_cons, List.not_mem_nil, or_false] at initial
          rcases initial with rfl | rfl
          · exact bound
          · decide
        · obtain ⟨destination, member, moved⟩ := List.mem_flatMap.mp moved
          exact moveWeights_bound destination (destinations_bound square bound destination member) entry moved
      · simpa using bound

theorem rollWeights_bound (state : Nat × Nat) (doublesBound : state.2 < 3) (first second : Nat) :
    ∀ entry ∈ rollWeights state first second, entry.1.1 < 40 ∧ entry.1.2 < 3 := by
  unfold rollWeights
  split
  · simp
  · rename_i notThird
    intro entry member
    obtain ⟨⟨⟨square, jailed⟩, weight⟩, landingMember, rfl⟩ := List.mem_map.mp member
    dsimp
    refine ⟨landingWeights_bound _ (Nat.mod_lt _ (by decide)) ((square, jailed), weight) landingMember, ?_⟩
    split
    · decide
    · split <;> omega

theorem turnWeights_bound (sides : Nat) (state : Nat × Nat) (doublesBound : state.2 < 3) :
    ∀ entry ∈ turnWeights sides state, entry.1.1 < 40 ∧ entry.1.2 < 3 := by
  intro entry member
  obtain ⟨first, _, member⟩ := List.mem_flatMap.mp member
  obtain ⟨second, _, member⟩ := List.mem_flatMap.mp member
  exact rollWeights_bound state doublesBound (first + 1) (second + 1) entry member

abbrev State := Fin 40 × Fin 3

def stateValues (state : State) : Nat × Nat := (state.1.val, state.2.val)

theorem stateValues_injective : Function.Injective stateValues := by
  intro left right equal
  apply Prod.ext <;> apply Fin.ext
  · exact congrArg Prod.fst equal
  · exact congrArg Prod.snd equal

def matrix (sides : Nat) (before after : State) : ℚ :=
  ((turnWeights sides (stateValues before)).map fun entry =>
    if entry.1 = stateValues after then (entry.2 : ℚ) else 0).sum

theorem matrix_nonnegative (sides : Nat) (before after : State) : 0 ≤ matrix sides before after := by
  apply List.sum_nonneg
  intro value member
  obtain ⟨entry, _, rfl⟩ := List.mem_map.mp member
  split <;> positivity

theorem matrix_row_sum (sides : Nat) (before : State) :
    ∑ after, matrix sides before after = 256 * (sides : ℚ) * sides := by
  classical
  have total (entries : List ((Nat × Nat) × Nat))
      (valid : ∀ entry ∈ entries, entry.1.1 < 40 ∧ entry.1.2 < 3) :
      (∑ after : State, (entries.map fun entry =>
        if entry.1 = stateValues after then (entry.2 : ℚ) else 0).sum) =
        (entries.map fun entry => (entry.2 : ℚ)).sum := by
    induction entries with
    | nil => simp
    | cons entry rest ih =>
        have bound := valid entry (List.mem_cons_self ..)
        let target : State := (⟨entry.1.1, bound.1⟩, ⟨entry.1.2, bound.2⟩)
        have equal (after : State) : entry.1 = stateValues after ↔ target = after := by
          change stateValues target = stateValues after ↔ target = after
          exact stateValues_injective.eq_iff
        simp only [List.map_cons, List.sum_cons, Finset.sum_add_distrib, equal]
        rw [ih (fun item member => valid item (List.mem_cons_of_mem entry member))]
        simp
  unfold matrix
  rw [total _ (turnWeights_bound sides (stateValues before) before.2.isLt)]
  have castTotal := congrArg (fun n : Nat => (n : ℚ)) (turnWeights_total sides (stateValues before))
  simpa using castTotal

theorem matrix_positive_of_step (sides : Nat) (before after : State)
    (step : Step sides (stateValues before) (stateValues after)) : 0 < matrix sides before after := by
  obtain ⟨weight, positive, member⟩ := step
  have term : (weight : ℚ) ∈ ((turnWeights sides (stateValues before)).map fun entry =>
      if entry.1 = stateValues after then (entry.2 : ℚ) else 0) := by
    exact List.mem_map.mpr ⟨(stateValues after, weight), member, by simp⟩
  have bound : (weight : ℚ) ≤ matrix sides before after := by
    apply List.single_le_sum _ (weight : ℚ) term
    intro value member
    obtain ⟨entry, _, rfl⟩ := List.mem_map.mp member
    split <;> positivity
  exact lt_of_lt_of_le (by exact_mod_cast positive) bound

def doubleState (state : State) : State :=
  let next := doubleStep (stateValues state)
  let bound := doubleStep_bound (stateValues state) state.2.isLt
  (⟨next.1, bound.1⟩, ⟨next.2, bound.2⟩)

theorem doubleState_values (state : State) :
    stateValues (doubleState state) = doubleStep (stateValues state) := rfl

theorem matrix_reaches_jail (sides : Nat) (positive : 0 < sides) (state : State) :
    Relation.ReflTransGen (fun a b => 0 < matrix sides a b) state (10, 0) := by
  have step (before : State) : 0 < matrix sides before (doubleState before) :=
    matrix_positive_of_step sides before (doubleState before) (doubleStep_supported sides positive (stateValues before))
  have one : Relation.ReflTransGen (fun a b => 0 < matrix sides a b) state (doubleState state) :=
    Relation.ReflTransGen.single (step state)
  have two := one.tail (step (doubleState state))
  have three := two.tail (step (doubleState (doubleState state)))
  rcases jail_within_three state.1.val state.2.val state.2.isLt with first | second | third
  · have equal : doubleState state = (10, 0) := stateValues_injective first
    rwa [equal] at one
  · have equal : doubleState (doubleState state) = (10, 0) := stateValues_injective second
    rwa [equal] at two
  · have equal : doubleState (doubleState (doubleState state)) = (10, 0) := stateValues_injective third
    rwa [equal] at three

/-- The exact normalized stationary solution is unique for every positive number of die faces. -/
theorem stationary_unique (sides : Nat) (positive : 0 < sides) (left right : State → ℚ)
    (leftStationary : ∀ after, 256 * (sides : ℚ) * sides * left after =
      ∑ before, matrix sides before after * left before)
    (rightStationary : ∀ after, 256 * (sides : ℚ) * sides * right after =
      ∑ before, matrix sides before after * right before)
    (sameSum : ∑ state, left state = ∑ state, right state) : left = right :=
  StationaryWeights.unique (matrix sides) (256 * (sides : ℚ) * sides)
    (matrix_nonnegative sides) (matrix_row_sum sides) (10, 0) (matrix_reaches_jail sides positive)
    left right leftStationary rightStationary sameSum

/-- The model defines a single rational probability distribution, including one-sided dice. -/
theorem stationary_exists_unique (sides : Nat) (positive : 0 < sides) :
    ∃! value : State → ℚ, (∀ state, 0 ≤ value state) ∧ (∑ state, value state = 1) ∧
      ∀ after, 256 * (sides : ℚ) * sides * value after = ∑ before, matrix sides before after * value before := by
  obtain ⟨value, nonnegative, normalized, stationary⟩ :=
    StationaryWeights.exists_stationary (matrix sides) (256 * (sides : ℚ) * sides)
      (matrix_nonnegative sides) (matrix_row_sum sides) (10, 0) (matrix_reaches_jail sides positive)
  refine ⟨value, ⟨nonnegative, normalized, stationary⟩, ?_⟩
  intro other properties
  exact stationary_unique sides positive other value properties.2.2 stationary (properties.2.1.trans normalized.symm)

/-- Any one stationary equation may be replaced by normalization without introducing a singularity. -/
theorem stationary_system_bijective (sides : Nat) (positive : 0 < sides) (omitted : State) :
    Function.Bijective (StationaryWeights.normalizedSystem (matrix sides) (256 * (sides : ℚ) * sides) omitted) :=
  StationaryWeights.normalizedSystem_bijective (matrix sides) (256 * (sides : ℚ) * sides)
    (matrix_nonnegative sides) (matrix_row_sum sides) omitted (10, 0) (matrix_reaches_jail sides positive)

end ProjectEulerSolutions.Monopoly
