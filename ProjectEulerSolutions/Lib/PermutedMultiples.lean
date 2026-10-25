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

end ProjectEulerSolutions.PermutedMultiples
