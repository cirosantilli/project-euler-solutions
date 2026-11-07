import ProjectEulerSolutions.Lib.CyclicStops

/-! Integer transition weights for the uniform-card statistical model of Monopoly. -/
namespace ProjectEulerSolutions.Monopoly
open CyclicStops
set_option Elab.async false

abbrev Landing := Nat × Bool
abbrev Weights := List (Landing × Nat)

def encode (square doubles : Nat) : Nat := square * 3 + doubles

theorem encode_bound (square doubles : Nat) (squareBound : square < 40) (doublesBound : doubles < 3) :
    encode square doubles < 120 := by unfold encode; omega

theorem decode_encode (square doubles : Nat) (doublesBound : doubles < 3) :
    (encode square doubles / 3, encode square doubles % 3) = (square, doubles) := by
  simp only [encode, Prod.mk.injEq]
  omega

theorem encode_injective (square doubles square' doubles' : Nat)
    (doublesBound : doubles < 3) (doublesBound' : doubles' < 3)
    (equal : encode square doubles = encode square' doubles') :
    (square, doubles) = (square', doubles') := by
  have first := decode_encode square doubles doublesBound
  have second := decode_encode square' doubles' doublesBound'
  rw [equal] at first
  exact first.symm.trans second

/-- A Chance movement has weight sixteen; landing on CC3 splits it into fourteen, one, one. -/
def moveWeights (destination : Nat) : Weights :=
  if destination = 33 then [((33, false), 14), ((0, false), 1), ((10, true), 1)]
  else [((destination, false), 16)]

def destinations (square : Nat) : List Nat :=
  [0, 11, 24, 39, 5, (nextStop [5, 15, 25, 35] 5 square).toNat,
    (nextStop [5, 15, 25, 35] 5 square).toNat, (nextStop [12, 28] 12 square).toNat, square - 3]

def landingWeights (square : Nat) : Weights :=
  if square = 30 then [((10, true), 256)]
  else if square ∈ [2, 17, 33] then [((square, false), 224), ((0, false), 16), ((10, true), 16)]
  else if square ∈ [7, 22, 36] then
    [((square, false), 96), ((10, true), 16)] ++ (destinations square).flatMap moveWeights
  else [((square, false), 256)]

theorem moveWeights_total (destination : Nat) : ((moveWeights destination).map Prod.snd).sum = 16 := by
  by_cases cc : destination = 33 <;> simp [moveWeights, cc]

theorem landingWeights_total (square : Nat) : ((landingWeights square).map Prod.snd).sum = 256 := by
  unfold landingWeights
  split
  · rfl
  · split
    · rfl
    · split
      · simp [destinations, moveWeights_total]
      · rfl

/-- Every non-G2J landing has a positive-weight choice that leaves the piece there. -/
theorem landingWeights_stay (square : Nat) (ordinary : square ≠ 30) :
    ∃ weight, 0 < weight ∧ ((square, false), weight) ∈ landingWeights square := by
  unfold landingWeights
  rw [if_neg ordinary]
  split
  · exact ⟨224, by decide, by simp⟩
  · split
    · exact ⟨96, by decide, List.mem_append_left _ (by simp)⟩
    · exact ⟨256, by decide, by simp⟩

/-- Choose a double-one roll and nonmoving cards, unless a jail rule intervenes. -/
def doubleStep (state : Nat × Nat) : Nat × Nat :=
  if state.2 = 2 ∨ (state.1 + 2) % 40 = 30 then (10, 0)
  else ((state.1 + 2) % 40, state.2 + 1)

theorem doubleStep_bound (state : Nat × Nat) (doublesBound : state.2 < 3) :
    (doubleStep state).1 < 40 ∧ (doubleStep state).2 < 3 := by
  unfold doubleStep
  split
  · decide
  · have := Nat.mod_lt (state.1 + 2) (by decide : 0 < 40)
    omega

/-- Jail is reachable from every state within three rolls, for every positive die size.
The selected double-one roll has positive probability, as do all selected nonmoving cards. -/
theorem jail_within_three (square doubles : Nat) (doublesBound : doubles < 3) :
    doubleStep (square, doubles) = (10, 0) ∨
      doubleStep (doubleStep (square, doubles)) = (10, 0) ∨
      doubleStep (doubleStep (doubleStep (square, doubles))) = (10, 0) := by
  have cases : doubles = 0 ∨ doubles = 1 ∨ doubles = 2 := by omega
  rcases cases with rfl | rfl | rfl
  · by_cases first : (square + 2) % 40 = 30
    · exact Or.inl (by simp [doubleStep, first])
    · by_cases second : (square + 2 + 2) % 40 = 30
      · exact Or.inr (Or.inl (by simp [doubleStep, first, second]))
      · exact Or.inr (Or.inr (by simp [doubleStep, first, second]))
  · by_cases first : (square + 2) % 40 = 30
    · exact Or.inl (by simp [doubleStep, first])
    · exact Or.inr (Or.inl (by simp [doubleStep, first]))
  · exact Or.inl (by simp [doubleStep])

/-- One ordered dice pair, retaining the common card denominator 256. -/
def rollWeights (state : Nat × Nat) (first second : Nat) : List ((Nat × Nat) × Nat) :=
  if first = second ∧ state.2 = 2 then [((10, 0), 256)]
  else (landingWeights ((state.1 + first + second) % 40)).map fun (landing, weight) =>
    ((landing.1, if landing.2 then 0 else if first = second then state.2 + 1 else 0), weight)

theorem rollWeights_total (state : Nat × Nat) (first second : Nat) :
    ((rollWeights state first second).map Prod.snd).sum = 256 := by
  unfold rollWeights
  split
  · rfl
  · simpa only [List.map_map, Function.comp_def] using landingWeights_total ((state.1 + first + second) % 40)

def turnWeights (sides : Nat) (state : Nat × Nat) : List ((Nat × Nat) × Nat) :=
  (List.range sides).flatMap fun first =>
    (List.range sides).flatMap fun second => rollWeights state (first + 1) (second + 1)

private theorem total_flatMap (items : List α) (f : α → List (β × Nat)) (total : Nat)
    (constant : ∀ item ∈ items, ((f item).map Prod.snd).sum = total) :
    ((items.flatMap f).map Prod.snd).sum = items.length * total := by
  induction items with
  | nil => simp
  | cons first rest ih =>
      have head := constant first (by simp)
      have tail := ih (fun item member => constant item (by simp [member]))
      simp [head, tail, Nat.add_mul, Nat.add_comm]

/-- All ordered dice/card combinations have total mass 256 times the number of dice pairs. -/
theorem turnWeights_total (sides : Nat) (state : Nat × Nat) :
    ((turnWeights sides state).map Prod.snd).sum = 256 * sides * sides := by
  unfold turnWeights
  rw [total_flatMap _ _ (sides * 256)]
  · simp [Nat.mul_comm]
  · intro first member
    simpa using total_flatMap (List.range sides) (fun second => rollWeights state (first + 1) (second + 1))
      256 (fun second _ => rollWeights_total state (first + 1) (second + 1))

/-- The all-double path used above is represented with strictly positive weight. -/
theorem doubleStep_mem (state : Nat × Nat) :
    ∃ weight, 0 < weight ∧ (doubleStep state, weight) ∈ rollWeights state 1 1 := by
  by_cases third : state.2 = 2
  · exact ⟨256, by decide, by simp [doubleStep, rollWeights, third]⟩
  · by_cases jail : (state.1 + 2) % 40 = 30
    · exact ⟨256, by decide, by simp [doubleStep, rollWeights, third, Nat.add_assoc, jail, landingWeights]⟩
    · obtain ⟨weight, positive, member⟩ := landingWeights_stay ((state.1 + 2) % 40) jail
      refine ⟨weight, positive, ?_⟩
      simpa [doubleStep, rollWeights, third, Nat.add_assoc, jail] using member

def Step (sides : Nat) (before after : Nat × Nat) : Prop :=
  ∃ weight, 0 < weight ∧ (after, weight) ∈ turnWeights sides before

theorem doubleStep_supported (sides : Nat) (positive : 0 < sides) (state : Nat × Nat) :
    Step sides state (doubleStep state) := by
  obtain ⟨weight, positiveWeight, member⟩ := doubleStep_mem state
  refine ⟨weight, positiveWeight, List.mem_flatMap.mpr ⟨0, by simpa, ?_⟩⟩
  exact List.mem_flatMap.mpr ⟨0, by simpa, member⟩

/-- All valid states can reach the same state, ruling out distinct closed communicating classes. -/
theorem reaches_jail (sides : Nat) (positive : 0 < sides) (state : Nat × Nat)
    (doublesBound : state.2 < 3) :
    Relation.TransGen (Step sides) state (10, 0) := by
  have one := Relation.TransGen.single (doubleStep_supported sides positive state)
  have two := one.trans (Relation.TransGen.single (doubleStep_supported sides positive (doubleStep state)))
  have three := two.trans (Relation.TransGen.single (doubleStep_supported sides positive (doubleStep (doubleStep state))))
  rcases jail_within_three state.1 state.2 doublesBound with first | second | third
  · simpa only [first] using one
  · simpa only [second] using two
  · simpa only [third] using three

end ProjectEulerSolutions.Monopoly
