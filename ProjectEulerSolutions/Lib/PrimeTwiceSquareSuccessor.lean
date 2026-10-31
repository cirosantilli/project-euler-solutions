import ProjectEulerSolutions.Lib.PrimeTwiceSquare
import Mathlib.Data.Nat.Find

/-! Least counterexamples above a shared threshold, with no assumption of unbounded existence. -/
namespace ProjectEulerSolutions.PrimeTwiceSquareSuccessor
open PrimeTwiceSquare
set_option Elab.async false

def value (index : Nat) : Nat := 9 + 2 * index

def accepts (after : Int) (n : Nat) : Bool := decide (after < (n : Int)) && counterexample n

theorem accepts_of (after : Int) (n : Nat) (later : after < (n : Int)) (accepted : counterexample n = true) :
    accepts after n = true := by
  exact Bool.and_eq_true_iff.mpr ⟨decide_eq_true later, accepted⟩

abbrev Success (after : Int) : Prop := ∃ index, accepts after (value index) = true

noncomputable def winnerIndex (after : Int) (success : Success after) : Nat := Nat.find success
noncomputable def answer (after : Int) (success : Success after) : Nat := value (winnerIndex after success)

theorem winner_accepted (after : Int) (success : Success after) : accepts after (answer after success) = true :=
  Nat.find_spec success

theorem winner_minimal (after : Int) (success : Success after) (index : Nat)
    (accepted : accepts after (value index) = true) : winnerIndex after success ≤ index :=
  Nat.find_min' success accepted

theorem earlier_rejected (after : Int) (success : Success after) (index : Nat)
    (before : index < winnerIndex after success) : accepts after (value index) = false := by
  apply Bool.eq_false_iff.mpr
  intro accepted
  have least := winner_minimal after success index accepted
  omega

theorem no_success (after : Int) (absent : ¬ Success after) (index : Nat) :
    accepts after (value index) = false := by
  apply Bool.eq_false_iff.mpr
  exact fun accepted => absent ⟨index, accepted⟩

/-- Every odd composite already lies in the source's positive odd enumeration. -/
theorem odd_composite_index (n : Nat) (composite : oddComposite n = true) : ∃ index, n = value index := by
  have facts : n % 2 = 1 ∧ ¬ Nat.Prime n ∧ 1 < n := by simpa only [oddComposite, decide_eq_true_eq] using composite
  have lower : 9 ≤ n := by
    by_contra small
    have cases : n = 3 ∨ n = 5 ∨ n = 7 := by omega
    rcases cases with rfl | rfl | rfl <;> exact facts.2.1 (by decide)
  exact ⟨(n - 9) / 2, by unfold value; omega⟩

theorem answer_spec (after : Int) (success : Success after) :
    after < (answer after success : Int) ∧ oddComposite (answer after success) = true ∧
      ¬ Written (answer after success) := by
  have accepted := winner_accepted after success
  simp only [accepts, counterexample, Bool.and_eq_true, decide_eq_true_eq, Bool.not_eq_true'] at accepted
  exact ⟨accepted.1, accepted.2.1, fun written => by
    have yes := (fast_iff _).mpr written
    rw [accepted.2.2] at yes
    contradiction⟩

theorem answer_le_counterexample (after : Int) (success : Success after) (n : Nat)
    (accepted : accepts after n = true) : answer after success ≤ n := by
  have composite : oddComposite n = true := by
    simp only [accepts, counterexample, Bool.and_eq_true] at accepted
    exact accepted.2.1
  obtain ⟨index, rfl⟩ := odd_composite_index n composite
  have bound := winner_minimal after success index accepted
  unfold answer value
  omega

noncomputable def tailWritten (n first : Nat) : Bool := by
  classical
  exact decide (∃ k : Nat, first ≤ k ∧ 2 * k * k < n ∧ Nat.Prime (n - 2 * k * k))

theorem tail_iff (n first : Nat) : tailWritten n first = true ↔
    ∃ k : Nat, first ≤ k ∧ 2 * k * k < n ∧ Nat.Prime (n - 2 * k * k) := by
  classical
  simp only [tailWritten, decide_eq_true_eq]

theorem tail_stopped (n first : Nat) (stopped : n ≤ 2 * first * first) : tailWritten n first = false := by
  apply Bool.eq_false_iff.mpr
  intro accepted
  obtain ⟨k, bound, small, _⟩ := (tail_iff n first).mp accepted
  have monotone := Nat.mul_le_mul (Nat.mul_le_mul_left 2 bound) bound
  omega

theorem tail_current (n first : Nat) (small : 2 * first * first < n)
    (prime : Nat.Prime (n - 2 * first * first)) : tailWritten n first = true :=
  (tail_iff n first).mpr ⟨first, Nat.le_refl _, small, prime⟩

theorem tail_next (n first : Nat) (rejected : ¬ Nat.Prime (n - 2 * first * first)) :
    tailWritten n first = tailWritten n (first + 1) := by
  apply Bool.eq_iff_iff.mpr
  rw [tail_iff, tail_iff]
  constructor
  · rintro ⟨k, bound, small, prime⟩
    have different : first ≠ k := by intro same; subst k; exact rejected prime
    exact ⟨k, by omega, small, prime⟩
  · rintro ⟨k, bound, small, prime⟩
    exact ⟨k, by omega, small, prime⟩

theorem tail_start (n : Nat) : tailWritten n 1 = fast n := by
  apply Bool.eq_iff_iff.mpr
  rw [tail_iff, fast_iff]
  constructor
  · rintro ⟨k, positive, small, prime⟩
    refine ⟨n - 2 * k * k, k, prime, positive, ?_⟩
    rw [show 2 * k ^ 2 = 2 * k * k by simp [Nat.pow_two, Nat.mul_assoc]]
    exact (Nat.sub_add_cancel small.le).symm
  · rintro ⟨p, k, prime, positive, equation⟩
    have lower := prime.two_le
    have square : 2 * k ^ 2 = 2 * k * k := by simp [Nat.pow_two, Nat.mul_assoc]
    exact ⟨k, positive, by omega, by simpa only [show n - 2 * k * k = p by omega] using prime⟩

end ProjectEulerSolutions.PrimeTwiceSquareSuccessor
