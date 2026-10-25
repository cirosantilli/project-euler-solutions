import Init.Data.List.Range
import Init.Data.List.Nat.Pairwise
import Lean.Elab.Tactic.Omega

/-! Endpoint and indexed-consecutive straight tests agree on five distinct sorted ranks. -/
namespace ProjectEulerSolutions.PokerStraights

set_option Elab.async false

/-- The reference's indexed-consecutive test, before its nonempty and wheel cases. -/
def consecutive (ranks : List Int) : Bool :=
  ranks.zipIdx.all fun pair => decide (pair.1 = ranks.headD 0 + (pair.2 : Int))

theorem consecutive_get (ranks : List Int) (formed : consecutive ranks = true)
    (index : Nat) (inside : index < ranks.length) : ranks[index] = ranks.headD 0 + (index : Int) := by
  have bounds : index < ranks.zipIdx.length := by simpa using inside
  have member : (ranks[index], index) ∈ ranks.zipIdx := by
    simpa only [List.getElem_zipIdx, Nat.zero_add] using
      List.getElem_mem (l := ranks.zipIdx) bounds
  have tested := (List.all_eq_true.mp formed) (ranks[index], index) member
  simpa using tested

/-- A consecutive list is strictly increasing, independently of any assumed rank bounds. -/
theorem consecutive_strict (ranks : List Int) (formed : consecutive ranks = true) :
    ranks.Pairwise (fun left right : Int => left < right) := by
  apply List.pairwise_iff_getElem.mpr
  intro left right leftInside rightInside ordered
  rw [consecutive_get ranks formed left leftInside, consecutive_get ranks formed right rightInside]
  omega

theorem consecutive_nodup (ranks : List Int) (formed : consecutive ranks = true) : ranks.Nodup := by
  apply (consecutive_strict ranks formed).imp
  intro left right smaller
  omega

/-- Four strictly positive integer gaps occupy width four exactly when every gap is one. -/
theorem endpoint_width_iff (a b c d e : Int) (ab : a < b) (bc : b < c) (cd : c < d) (de : d < e) :
    e-a = 4 ↔ b = a+1 ∧ c = a+2 ∧ d = a+3 ∧ e = a+4 := by
  omega

theorem consecutive_five (a b c d e : Int) :
    consecutive [a, b, c, d, e] = decide (b = a+1 ∧ c = a+2 ∧ d = a+3 ∧ e = a+4) := by
  simp [consecutive]

theorem consecutive_five_width (a b c d e : Int) (ab : a < b) (bc : b < c) (cd : c < d) (de : d < e) :
    consecutive [a, b, c, d, e] = decide (e-a = 4) := by
  apply Bool.eq_iff_iff.mpr
  simp only [consecutive_five, decide_eq_true_eq]
  exact (endpoint_width_iff a b c d e ab bc cd de).symm

def wheel : List Int := [2, 3, 4, 5, 14]

def referenceStraight (ranks : List Int) : Bool :=
  decide (ranks = wheel) || (!ranks.isEmpty && consecutive ranks)

def straightHigh (ranks : List Int) : Int :=
  if ranks = wheel then 5 else ranks.getLast?.getD 0

theorem wheel_nodup : wheel.Nodup := by decide

theorem wheel_not_consecutive : consecutive wheel = false := by decide

theorem wheel_high : straightHigh wheel = 5 := by simp [straightHigh]

theorem width_not_wheel (a b c d e : Int) (width : e-a = 4) : [a, b, c, d, e] ≠ wheel := by
  intro same
  have first : a = 2 := by simpa [wheel] using congrArg (fun ranks : List Int => ranks.headD 0) same
  have last : e = 14 := by simpa [wheel] using congrArg (fun ranks : List Int => ranks.getLast?.getD 0) same
  omega

/-- Repeated ranks cannot satisfy the reference straight test, even with its wheel exception. -/
theorem straight_nodup (ranks : List Int) (formed : referenceStraight ranks = true) : ranks.Nodup := by
  by_cases special : ranks = wheel
  · subst ranks
    exact wheel_nodup
  · have tested : ranks.isEmpty = false ∧ consecutive ranks = true := by
      simpa [referenceStraight, special] using formed
    exact consecutive_nodup ranks tested.2

theorem not_straight_of_duplicates (ranks : List Int) (repeated : ¬ ranks.Nodup) : referenceStraight ranks = false := by
  apply Bool.eq_false_iff.mpr
  intro formed
  exact repeated (straight_nodup ranks formed)

/-- The root's width-or-wheel branch is exactly the indexed reference branch on distinct sorted ranks. -/
theorem five_straight (a b c d e : Int) (ab : a < b) (bc : b < c) (cd : c < d) (de : d < e) :
    referenceStraight [a, b, c, d, e] =
      (decide (e-a = 4) || decide ([a, b, c, d, e] = wheel)) := by
  change (decide ([a, b, c, d, e] = wheel) || consecutive [a, b, c, d, e]) = _
  rw [consecutive_five_width a b c d e ab bc cd de]
  exact Bool.or_comm _ _

theorem high_five_of_not_wheel (a b c d e : Int) (ordinary : [a, b, c, d, e] ≠ wheel) :
    straightHigh [a, b, c, d, e] = e := by
  rw [straightHigh, if_neg ordinary]
  rfl

/-- Width-four lists use their endpoint, while the disjoint wheel uses five. -/
theorem width_high (a b c d e : Int) (width : e-a = 4) : straightHigh [a, b, c, d, e] = e :=
  high_five_of_not_wheel a b c d e (width_not_wheel a b c d e width)

/-- A five-term consecutive sequence ending at ace is exactly the royal-flush rank sequence. -/
theorem royal_of_consecutive (a b c d e : Int) (formed : consecutive [a, b, c, d, e] = true)
    (highest : e = 14) : [a, b, c, d, e] = [10, 11, 12, 13, 14] := by
  have increments : b = a+1 ∧ c = a+2 ∧ d = a+3 ∧ e = a+4 := by
    simpa only [consecutive_five, decide_eq_true_eq] using formed
  have first : a = 10 := by omega
  have second : b = 11 := by omega
  have third : c = 12 := by omega
  have fourth : d = 13 := by omega
  subst a
  subst b
  subst c
  subst d
  subst e
  rfl

theorem royal_high : straightHigh [10, 11, 12, 13, 14] = 14 := by decide

/-- The upper rank bound is needed here: a nonroyal straight cannot outrank a royal flush. -/
theorem nonroyal_high_lt_fourteen (a b c d e : Int) (formed : referenceStraight [a, b, c, d, e] = true)
    (bounded : e ≤ 14) (nonroyal : [a, b, c, d, e] ≠ [10, 11, 12, 13, 14]) :
    straightHigh [a, b, c, d, e] < 14 := by
  by_cases special : [a, b, c, d, e] = wheel
  · simp [straightHigh, special]
  · have tested : consecutive [a, b, c, d, e] = true := by
      simpa [referenceStraight, special] using formed
    have notHighest : e ≠ 14 := fun highest => nonroyal (royal_of_consecutive a b c d e tested highest)
    rw [high_five_of_not_wheel a b c d e special]
    omega

end ProjectEulerSolutions.PokerStraights
