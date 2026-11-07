import ProjectEulerSolutions.P49
import ProjectEulerSolutions.Lib.PrimePermutations

namespace ProjectEulerSolutions.P49

theorem sameDigits_eq (a b : Nat) : sameDigits a b = ProjectEulerStatements.P49.isPermutation a b := by
  rw [← PrimePermutations.permutation_statement]
  apply Bool.eq_iff_iff.mpr
  simp only [sameDigits, decide_eq_true_eq, DigitHistogram.signature_eq_iff,
    PrimePermutations.sortedDigits_eq_iff]

theorem allPerm_eq (values : List Nat) : allPerm values = ProjectEulerStatements.P49.allPerm values := by
  cases values with
  | nil => rfl
  | cons first rest =>
      apply congrArg (fun predicate => rest.all predicate)
      funext value
      exact sameDigits_eq first value

theorem generic_equiv (n seqlen : Nat) : ProjectEulerStatements.P49.naive n seqlen = generic n seqlen := by
  simp only [ProjectEulerStatements.P49.naive, generic, allPerm_eq]
  split <;> rfl

theorem equiv (n seqlen : Nat) : ProjectEulerStatements.P49.naive n seqlen = solve n seqlen := by
  unfold solve
  split
  · rename_i canonical
    obtain ⟨rfl, rfl⟩ := canonical
    exact P49Buckets.rows_correct.symm
  · exact generic_equiv n seqlen

end ProjectEulerSolutions.P49
