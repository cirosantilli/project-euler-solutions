import ProjectEulerSolutions.Lib.Monopoly
import Mathlib.Tactic.IntervalCases

/-! Explicit independent card pairs agree with the weighted landing rules. -/
namespace ProjectEulerSolutions.Monopoly
open CyclicStops
set_option Elab.async false

def chanceCard (square card : Nat) : Landing :=
  if square ∈ [7, 22, 36] then
    if card < 6 then ([0, 10, 11, 24, 39, 5][card]?.getD 0, decide (card = 1))
    else if card ∈ [6, 7] then ((nextStop [5, 15, 25, 35] 5 square).toNat, false)
    else if card = 8 then ((nextStop [12, 28] 12 square).toNat, false)
    else if card = 9 then (square - 3, false)
    else (square, false)
  else (square, false)

def communityCard (landing : Landing) (card : Nat) : Landing :=
  if landing.1 ∈ [2, 17, 33] ∧ card < 2 then (if card = 0 then 0 else 10, decide (card = 1))
  else landing

def cardLanding (square community chance : Nat) : Landing :=
  if square = 30 then (10, true) else communityCard (chanceCard square chance) community

def landingCards (square : Nat) : List Landing :=
  (List.range 16).flatMap fun community => (List.range 16).map fun chance => cardLanding square community chance

def expandWeights (weights : List (α × Nat)) : List α :=
  weights.flatMap fun entry => List.replicate entry.2 entry.1

set_option maxRecDepth 4096 in
set_option maxHeartbeats 2000000 in
/-- All card faces are finite rules of the board; the comparison includes the Chance-to-CC3 cascade. -/
theorem landingCards_perm (square : Nat) (bound : square < 40) :
    (landingCards square).Perm (expandWeights (landingWeights square)) := by
  interval_cases square <;> decide

def advanceCards (state : Nat × Nat) (first second community chance : Nat) : Nat × Nat :=
  if first = second ∧ state.2 = 2 then (10, 0)
  else
    let landing := cardLanding ((state.1 + first + second) % 40) community chance
    (landing.1, if landing.2 then 0 else if first = second then state.2 + 1 else 0)

def rollCards (state : Nat × Nat) (first second : Nat) : List (Nat × Nat) :=
  if first = second ∧ state.2 = 2 then List.replicate 256 (10, 0)
  else (landingCards ((state.1 + first + second) % 40)).map fun landing =>
    (landing.1, if landing.2 then 0 else if first = second then state.2 + 1 else 0)

theorem expandWeights_map (weights : List (α × Nat)) (map : α → β) :
    expandWeights (weights.map fun entry => (map entry.1, entry.2)) = (expandWeights weights).map map := by
  simp [expandWeights, List.map_flatMap, List.map_replicate, List.flatMap_map]

/-- Enumerating independent cards and grouping their multiplicities give the same dice-pair outcomes. -/
theorem rollCards_perm (state : Nat × Nat) (first second : Nat) :
    (rollCards state first second).Perm (expandWeights (rollWeights state first second)) := by
  by_cases third : first = second ∧ state.2 = 2
  · simp only [rollCards, rollWeights, if_pos third, expandWeights, List.flatMap_cons, List.flatMap_nil, List.append_nil]
    exact List.Perm.refl _
  · let follow := fun landing : Landing =>
      (landing.1, if landing.2 then 0 else if first = second then state.2 + 1 else 0)
    simp only [rollCards, rollWeights, if_neg third]
    change ((landingCards ((state.1 + first + second) % 40)).map follow).Perm
      (expandWeights ((landingWeights ((state.1 + first + second) % 40)).map fun entry => (follow entry.1, entry.2)))
    rw [expandWeights_map]
    exact (landingCards_perm _ (Nat.mod_lt _ (by decide))).map _

theorem rollCards_enumerated (state : Nat × Nat) (first second : Nat) :
    (List.range 16).flatMap (fun community => (List.range 16).map fun chance =>
      advanceCards state first second community chance) = rollCards state first second := by
  by_cases third : first = second ∧ state.2 = 2
  · simp only [advanceCards, rollCards, if_pos third, List.map_const', List.length_range,
      List.flatMap_def, List.flatten_replicate_replicate]
  · simp only [advanceCards, rollCards, if_neg third, landingCards, List.map_flatMap, List.map_map, Function.comp_def]

theorem chanceCard_bound (square card : Nat) (inside : square < 40) (cardBound : card < 16) :
    (chanceCard square card).1 < 40 := by
  by_cases chance : square ∈ [7, 22, 36]
  · have choices : square = 7 ∨ square = 22 ∨ square = 36 := by simpa using chance
    rcases choices with rfl | rfl | rfl <;> interval_cases card <;> decide
  · simpa only [chanceCard, if_neg chance] using inside

theorem cardLanding_bound (square community chance : Nat) (inside : square < 40) (chanceBound : chance < 16) :
    (cardLanding square community chance).1 < 40 := by
  unfold cardLanding
  split
  · decide
  · unfold communityCard
    split
    · split <;> simp
    · exact chanceCard_bound square chance inside chanceBound

theorem advanceCards_bound (state : Nat × Nat) (first second community chance : Nat)
    (doublesBound : state.2 < 3) (chanceBound : chance < 16) :
    (advanceCards state first second community chance).1 < 40 ∧
      (advanceCards state first second community chance).2 < 3 := by
  unfold advanceCards
  split
  · decide
  · rename_i ordinary
    refine ⟨cardLanding_bound _ community chance (Nat.mod_lt _ (by decide)) chanceBound, ?_⟩
    dsimp only
    split_ifs <;> omega

end ProjectEulerSolutions.Monopoly
