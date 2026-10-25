import ProjectEulerSolutions.Lib.DigitHistogram
import ProjectEulerSolutions.Lib.PrimePermutations
import Mathlib.Data.Nat.Find
import Mathlib.Tactic.IntervalCases

/-! Digit-permutation predicates and a safe finite witness for the positive multiple search. -/
namespace ProjectEulerSolutions.PermutedMultiples
open DigitHistogram PrimePermutations

set_option Elab.async false

def same (a b : Nat) : Bool := decide (signature a = signature b)

def accepts (limit : Int) (x : Nat) : Bool :=
  decide (2 ≤ limit) && (List.range (limit-1).toNat).all (fun i => same x ((i+2)*x))

/-- Sorting and counting express the same equality of decimal digit multisets. -/
theorem sorted_same (a b : Nat) : decide (sortedDigits a = sortedDigits b) = same a b := by
  apply Bool.eq_iff_iff.mpr
  simp only [same, decide_eq_true_eq, sortedDigits_eq_iff, signature_eq_iff]

theorem same_permutation (a b : Nat) : same a b = true ↔ (Nat.digits 10 a).Perm (Nat.digits 10 b) := by
  simp only [same, decide_eq_true_eq, signature_eq_iff]

/-- The small cyclic-number witness covers every supported target without an existence axiom. -/
theorem witness (limit : Int) (supported : 2 ≤ limit ∧ limit ≤ 6) : accepts limit 142857 = true := by
  obtain ⟨lower, upper⟩ := supported
  interval_cases limit <;> decide +kernel

private theorem exists_positive (limit : Int) (supported : 2 ≤ limit ∧ limit ≤ 6) :
    ∃ x : Nat, 0 < x ∧ accepts limit x = true := ⟨142857, by decide, witness limit supported⟩

/-- Only the supported interval uses the least positive witness. -/
noncomputable def answer (limit : Int) : Nat :=
  if supported : 2 ≤ limit ∧ limit ≤ 6 then Nat.find (exists_positive limit supported) else 0

theorem answer_positive (limit : Int) (supported : 2 ≤ limit ∧ limit ≤ 6) : 0 < answer limit := by
  simp only [answer, dif_pos supported]
  exact (Nat.find_spec (exists_positive limit supported)).1

theorem answer_accepted (limit : Int) (supported : 2 ≤ limit ∧ limit ≤ 6) : accepts limit (answer limit) = true := by
  simp only [answer, dif_pos supported]
  exact (Nat.find_spec (exists_positive limit supported)).2

theorem answer_minimal (limit : Int) (supported : 2 ≤ limit ∧ limit ≤ 6) (x : Nat)
    (positive : 0 < x) (accepted : accepts limit x = true) : answer limit ≤ x := by
  simp only [answer, dif_pos supported]
  exact Nat.find_min' (exists_positive limit supported) ⟨positive, accepted⟩

theorem answer_le_witness (limit : Int) (supported : 2 ≤ limit ∧ limit ≤ 6) : answer limit ≤ 142857 :=
  answer_minimal limit supported 142857 (by decide) (witness limit supported)

/-- Acceptance includes the final requested multiple, so it has the candidate's digit width. -/
theorem last_permutation (limit : Nat) (x : Nat) (supported : 2 ≤ limit) (accepted : accepts limit x = true) :
    (Nat.digits 10 x).Perm (Nat.digits 10 (limit*x)) := by
  simp only [accepts, Bool.and_eq_true, decide_eq_true_eq] at accepted
  have accepted := accepted.2
  have present : limit-2 ∈ List.range ((limit : Int)-1).toNat := by
    simp only [List.mem_range]
    omega
  have final := List.all_eq_true.mp accepted (limit-2) present
  have same : limit-2+2 = limit := by omega
  exact (same_permutation x _).mp (by simpa only [same] using final)

def lower (digits : Nat) : Nat := 10 ^ (digits-1)
def upper (limit digits : Nat) : Nat := (10 ^ digits-1)/limit
noncomputable def answerWidth (limit : Int) : Nat := (Nat.digits 10 (answer limit)).length

/-- The finite witness bounds the decimal width of the least positive answer. -/
theorem width_bounds (limit : Int) (supported : 2 ≤ limit ∧ limit ≤ 6) : 1 ≤ answerWidth limit ∧ answerWidth limit ≤ 6 := by
  have positive := answer_positive limit supported
  have nonempty := (Nat.digits_ne_nil_iff_ne_zero (b := 10)).mpr (by omega : answer limit ≠ 0)
  refine ⟨by simpa only [answerWidth] using List.length_pos_iff.mpr nonempty, ?_⟩
  apply (Nat.digits_length_le_iff (b := 10) (k := 6) (by decide) (answer limit)).mpr
  have maximum := answer_le_witness limit supported
  norm_num
  omega

/-- Earlier width windows start no later than the reference minimum. -/
theorem lower_before (limit : Int) (supported : 2 ≤ limit ∧ limit ≤ 6) (digits : Nat) (before : digits ≤ answerWidth limit) :
    lower digits ≤ answer limit := by
  have widths := width_bounds limit supported
  have leading : 10 ^ (answerWidth limit-1) ≤ answer limit :=
    (Nat.lt_digits_length_iff (b := 10) (k := answerWidth limit-1) (by decide) (answer limit)).mp
      (by change answerWidth limit-1 < answerWidth limit; omega)
  exact (Nat.pow_le_pow_right (by decide : 1 ≤ 10) (show digits-1 ≤ answerWidth limit-1 by omega)).trans leading

/-- The last multiple's equal digit width justifies the optimized window's division by the target. -/
theorem inside_width (limit : Nat) (supported : 2 ≤ (limit : Int) ∧ (limit : Int) ≤ 6) :
    answer limit ≤ upper limit (answerWidth limit) := by
  have accepted := answer_accepted limit supported
  have same := (last_permutation limit (answer limit) (by omega) accepted).length_eq
  have bound : limit * answer limit < 10 ^ answerWidth limit :=
    (Nat.digits_length_le_iff (b := 10) (k := answerWidth limit) (by decide) (limit * answer limit)).mp (by rw [← same]; exact Nat.le_refl _)
  have product : answer limit * limit ≤ 10 ^ answerWidth limit-1 := by
    rw [Nat.mul_comm]
    omega
  exact (Nat.le_div_iff_mul_le (by omega : 0 < limit)).mpr product

private theorem find_range_minimum (predicate : Nat → Bool) (winner : Nat) (accepted : predicate winner = true)
    (least : ∀ n, 0 < n → predicate n = true → winner ≤ n) (first count : Nat)
    (positive : 0 < first) (lower : first ≤ winner) (upper : winner < first+count) :
    (List.range' first count).find? predicate = some winner := by
  induction count generalizing first with
  | zero => omega
  | succ count ih =>
      cases current : predicate first with
      | true =>
          have minimum := least first positive current
          have same : first = winner := by omega
          simp only [List.range'_succ, List.find?_cons, current]
          exact congrArg some same
      | false =>
          have different : first ≠ winner := by intro same; rw [same, accepted] at current; contradiction
          have rest := ih (first+1) (by omega) (by omega) (by omega)
          simpa only [List.range'_succ, List.find?_cons, current] using rest

/-- Each positive width scan first finds the global minimum when its upper bound reaches it. -/
theorem window (limit digits : Nat) (supported : 2 ≤ (limit : Int) ∧ (limit : Int) ≤ 6) (before : digits ≤ answerWidth limit) :
    (List.range' (lower digits) (upper limit digits+1-lower digits)).find? (fun x => accepts limit x) =
      if answer limit ≤ upper limit digits then some (answer limit) else none := by
  have minimum := lower_before limit supported digits before
  have positive : 0 < lower digits := Nat.pow_pos (by decide)
  by_cases inside : answer limit ≤ upper limit digits
  · rw [if_pos inside]
    exact find_range_minimum (accepts limit) (answer limit) (answer_accepted limit supported)
      (answer_minimal limit supported) (lower digits) (upper limit digits+1-lower digits) positive minimum (by omega)
  · rw [if_neg inside]
    apply List.find?_eq_none.mpr
    intro x member accepted
    obtain ⟨i, bound, same⟩ := List.mem_range'.mp member
    have lower : 0 < x := by omega
    have upper : x ≤ upper limit digits := by omega
    have least := answer_minimal limit supported x lower accepted
    omega

/-- Every supported decimal window contains at least its smallest number. -/
theorem window_nonempty (limit digits : Nat) (supported : 2 ≤ limit ∧ limit ≤ 6) (positive : 1 ≤ digits) :
    lower digits ≤ upper limit digits := by
  cases digits with
  | zero => omega
  | succ digits =>
      have powerPositive : 0 < 10 ^ digits := Nat.pow_pos (by decide)
      have scaled : 10 ^ digits * limit ≤ 10 ^ digits * 6 := Nat.mul_le_mul_left _ supported.2
      apply (Nat.le_div_iff_mul_le (by omega : 0 < limit)).mpr
      simp only [lower, Nat.add_sub_cancel, Nat.pow_succ]
      omega

end ProjectEulerSolutions.PermutedMultiples
