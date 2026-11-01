import Mathlib.Data.Nat.Digits.Lemmas
import Mathlib.Tactic.Linarith
import ProjectEulerStatements.Lib.BoundedOrbit

/-! Complete decimal digit-square orbits, with a proved finite invariant rather than a step cutoff. -/
namespace ProjectEulerSolutions.DigitSquareChains
open ProjectEulerStatements.BoundedOrbit
set_option Elab.async false

def nextTerm (n : Nat) : Nat := ((Nat.digits 10 n).map (fun d => d * d)).sum

@[simp] theorem nextTerm_zero : nextTerm 0 = 0 := by simp [nextTerm]

theorem nextTerm_step (n : Nat) :
    nextTerm n = (n % 10) * (n % 10) + nextTerm (n / 10) := by
  by_cases zero : n = 0
  · subst n; simp
  · simp only [nextTerm, Nat.digits_def' (by omega : 1 < 10) (by omega : 0 < n),
      List.map_cons, List.sum_cons]

/-- Squaring any decimal digit multiplies it by at most nine. -/
theorem nextTerm_linear (n : Nat) : nextTerm n ≤ 9 * n := by
  induction n using Nat.strong_induction_on with
  | h n ih =>
      by_cases zero : n = 0
      · subst n; simp
      · have recursiveBound := ih (n / 10) (Nat.div_lt_self (by omega) (by omega))
        have digit : n % 10 ≤ 9 := by omega
        have square := Nat.mul_le_mul_left (n % 10) digit
        rw [nextTerm_step]
        have decomposition := Nat.mod_add_div n 10
        nlinarith

/-- Separate the last two digits before applying the coarse linear bound. -/
theorem nextTerm_bound (n : Nat) : nextTerm n ≤ 162 + 9 * (n / 100) := by
  have low : n % 10 ≤ 9 := by omega
  have high : (n / 10) % 10 ≤ 9 := by omega
  have lowSquare := Nat.mul_self_le_mul_self low
  have highSquare := Nat.mul_self_le_mul_self high
  have tail := nextTerm_linear (n / 100)
  rw [nextTerm_step n, nextTerm_step (n / 10)]
  have division : n / 10 / 10 = n / 100 := by rw [Nat.div_div_eq_div_mul]
  rw [division]
  omega

/-- Above the invariant interval from zero through 199, the digit-square transform strictly decreases. -/
theorem nextTerm_lt (n : Nat) (large : 200 ≤ n) : nextTerm n < n := by
  have bounded := nextTerm_bound n
  have decomposition := Nat.mod_add_div n 100
  have quotient : 2 ≤ n / 100 := by omega
  omega

def bound (start : Nat) : Nat := max start 199

theorem nextTerm_bounded (start n : Nat) (inside : n ≤ bound start) :
    nextTerm n ≤ bound start := by
  by_cases large : 200 ≤ n
  · exact (Nat.le_of_lt (nextTerm_lt n large)).trans inside
  · have bounded := nextTerm_bound n
    have quotient : n / 100 ≤ 1 := by omega
    have floor : 199 ≤ bound start := Nat.le_max_right _ _
    omega

def orbit (start : Nat) : List Nat :=
  walk nextTerm (bound start) (nextTerm_bounded start) start []
    (Nat.le_max_left _ _) (by simp) (by simp [Within])

theorem orbit_trace (start : Nat) : Trace nextTerm start [] (orbit start) :=
  walk_trace nextTerm (bound start) (nextTerm_bounded start) start []
    (Nat.le_max_left _ _) (by simp) (by simp [Within])

/-- Membership in the finite first-repeat trace decides reachability along the unbounded orbit. -/
theorem mem_orbit (start target : Nat) :
    target ∈ orbit start ↔ ∃ steps : Nat, nextTerm^[steps] start = target :=
  (orbit_trace start).mem_iff

def endsAt89 (start : Nat) : Bool := decide (89 ∈ orbit start)

theorem endsAt89_iff (start : Nat) :
    endsAt89 start = true ↔ ∃ steps : Nat, nextTerm^[steps] start = 89 := by
  simp only [endsAt89, decide_eq_true_eq, mem_orbit]

end ProjectEulerSolutions.DigitSquareChains
