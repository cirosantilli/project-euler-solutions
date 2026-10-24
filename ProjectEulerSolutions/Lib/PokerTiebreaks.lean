import lang.py.FormalSW.HeapSort
import Lean.Elab.Tactic.Omega

/-! Repeating fixed poker tie-break fields preserves their lexicographic order.
These are mathematical bridges; source execution is proved separately. -/
namespace ProjectEulerSolutions.PokerTiebreaks
open FormalSW.PyAst

set_option Elab.async false

/-- Expand each compressed field by its frequency, as the reference count key does. -/
def expand : List Nat → List Int → List Int
  | [], _ => []
  | _ :: _, [] => []
  | copies :: weights, value :: values => List.replicate copies value ++ expand weights values

theorem lexInt_prefix (initial left right : List Int) :
    HeapSort.lexInt (initial ++ left) (initial ++ right) = HeapSort.lexInt left right := by
  induction initial with
  | nil => rfl
  | cons value rest ih =>
      simpa only [List.cons_append, HeapSort.lexInt, if_pos rfl] using ih

theorem lexInt_replicate_append (copies : Nat) (positive : 0 < copies)
    (x y : Int) (left right : List Int) :
    HeapSort.lexInt (List.replicate copies x ++ left) (List.replicate copies y ++ right) =
      if x = y then HeapSort.lexInt left right else decide (x < y) := by
  cases copies with
  | zero => omega
  | succ copies =>
      simp only [List.replicate_succ, List.cons_append, HeapSort.lexInt]
      by_cases same : x = y
      · subst y
        simpa only [if_pos rfl] using lexInt_prefix (List.replicate copies x) left right
      · simp only [if_neg same]

/-- Matching frequency patterns give expanded keys of matching lengths. -/
theorem expand_length_eq (weights : List Nat) (left right : List Int)
    (leftLength : left.length = weights.length) (rightLength : right.length = weights.length) :
    (expand weights left).length = (expand weights right).length := by
  induction weights generalizing left right with
  | nil => rfl
  | cons copies weights ih =>
      cases left with
      | nil => simp at leftLength
      | cons x left =>
          cases right with
          | nil => simp at rightLength
          | cons y right =>
              simp only [expand, List.length_append, List.length_replicate]
              rw [ih left right (Nat.succ.inj leftLength) (Nat.succ.inj rightLength)]

/-- Positive fixed multiplicities preserve every lexicographic comparison. -/
theorem expand_lexInt (weights : List Nat) (left right : List Int)
    (leftLength : left.length = weights.length) (rightLength : right.length = weights.length)
    (positive : ∀ copies ∈ weights, 0 < copies) :
    HeapSort.lexInt (expand weights left) (expand weights right) = HeapSort.lexInt left right := by
  induction weights generalizing left right with
  | nil =>
      have leftEmpty : left = [] := List.length_eq_zero_iff.mp leftLength
      have rightEmpty : right = [] := List.length_eq_zero_iff.mp rightLength
      subst left
      subst right
      rfl
  | cons copies weights ih =>
      cases left with
      | nil => simp at leftLength
      | cons x left =>
          cases right with
          | nil => simp at rightLength
          | cons y right =>
              have firstPositive := positive copies List.mem_cons_self
              have restPositive : ∀ count ∈ weights, 0 < count := by
                intro count member
                exact positive count (List.mem_cons_of_mem copies member)
              change HeapSort.lexInt (List.replicate copies x ++ expand weights left)
                (List.replicate copies y ++ expand weights right) =
                  if x = y then HeapSort.lexInt left right else decide (x < y)
              rw [lexInt_replicate_append copies firstPositive x y,
                ih left right (Nat.succ.inj leftLength) (Nat.succ.inj rightLength) restPositive]

/-- The reference's pure recurrence treats either empty key as a false comparison. -/
def referenceGreater : List Int → List Int → Bool
  | [], _ => false
  | _ :: _, [] => false
  | x :: left, y :: right => if x = y then referenceGreater left right else decide (y < x)

/-- The reference empty-key convention agrees with tuple ordering for equal key lengths. -/
theorem referenceGreater_eq_lexInt (left right : List Int) (sameLength : left.length = right.length) :
    referenceGreater left right = HeapSort.lexInt right left := by
  induction left generalizing right with
  | nil => cases right <;> rfl
  | cons x left ih =>
      cases right with
      | nil => simp at sameLength
      | cons y right =>
          by_cases same : x = y
          · subst y
            simpa only [referenceGreater, HeapSort.lexInt, if_pos rfl] using
              ih right (Nat.succ.inj sameLength)
          · simp only [referenceGreater, HeapSort.lexInt, if_neg same, if_neg (Ne.symm same)]

/-- The expanded reference key and compressed solver key choose the same winner. -/
theorem expanded_referenceGreater (weights : List Nat) (left right : List Int)
    (leftLength : left.length = weights.length) (rightLength : right.length = weights.length)
    (positive : ∀ copies ∈ weights, 0 < copies) :
    referenceGreater (expand weights left) (expand weights right) = HeapSort.lexInt right left := by
  rw [referenceGreater_eq_lexInt _ _ (expand_length_eq weights left right leftLength rightLength)]
  exact expand_lexInt weights right left rightLength leftLength positive

theorem four_of_a_kind (four kicker otherFour otherKicker : Int) :
    referenceGreater [four, four, four, four, kicker] [otherFour, otherFour, otherFour, otherFour, otherKicker] =
      HeapSort.lexInt [otherFour, otherKicker] [four, kicker] := by
  simpa [expand, List.replicate] using
    expanded_referenceGreater [4, 1] [four, kicker] [otherFour, otherKicker] rfl rfl
      (by intro copies member; simp at member; omega)

theorem full_house (trip pair otherTrip otherPair : Int) :
    referenceGreater [trip, trip, trip, pair, pair] [otherTrip, otherTrip, otherTrip, otherPair, otherPair] =
      HeapSort.lexInt [otherTrip, otherPair] [trip, pair] := by
  simpa [expand, List.replicate] using
    expanded_referenceGreater [3, 2] [trip, pair] [otherTrip, otherPair] rfl rfl
      (by intro copies member; simp at member; omega)

theorem three_of_a_kind (trip first second otherTrip otherFirst otherSecond : Int) :
    referenceGreater [trip, trip, trip, first, second] [otherTrip, otherTrip, otherTrip, otherFirst, otherSecond] =
      HeapSort.lexInt [otherTrip, otherFirst, otherSecond] [trip, first, second] := by
  simpa [expand, List.replicate] using
    expanded_referenceGreater [3, 1, 1] [trip, first, second] [otherTrip, otherFirst, otherSecond] rfl rfl
      (by intro copies member; simp at member; omega)

theorem two_pairs (high low kicker otherHigh otherLow otherKicker : Int) :
    referenceGreater [high, high, low, low, kicker] [otherHigh, otherHigh, otherLow, otherLow, otherKicker] =
      HeapSort.lexInt [otherHigh, otherLow, otherKicker] [high, low, kicker] := by
  simpa [expand, List.replicate] using
    expanded_referenceGreater [2, 2, 1] [high, low, kicker] [otherHigh, otherLow, otherKicker] rfl rfl
      (by intro copies member; simp at member; omega)

theorem one_pair (pair first second third otherPair otherFirst otherSecond otherThird : Int) :
    referenceGreater [pair, pair, first, second, third] [otherPair, otherPair, otherFirst, otherSecond, otherThird] =
      HeapSort.lexInt [otherPair, otherFirst, otherSecond, otherThird] [pair, first, second, third] := by
  simpa [expand, List.replicate] using
    expanded_referenceGreater [2, 1, 1, 1] [pair, first, second, third] [otherPair, otherFirst, otherSecond, otherThird] rfl rfl
      (by intro copies member; simp at member; omega)

end ProjectEulerSolutions.PokerTiebreaks
