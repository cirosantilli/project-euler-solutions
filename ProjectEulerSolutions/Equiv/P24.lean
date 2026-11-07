import ProjectEulerSolutions.P24
namespace ProjectEulerSolutions.P24

/-- The sorted permutation reference has the same rendering as the Python `naive_text` API. -/
theorem equiv (digits : List Int) (index : Int) (sorted : digits.Pairwise (· < ·))
    (positive : 1 ≤ index) (within : index ≤ FormalSW.PyAst.Math.factorial digits.length) :
    serialize (PermutationRank.reference digits index) = solve digits index := by
  have represented : index = ((index-1).toNat : Int) + 1 := by omega
  have bound : (index-1).toNat < FormalSW.PyAst.Math.factorial digits.length := by omega
  have same := PermutationRank.reference_unrank digits (index-1).toNat sorted bound
  rw [← represented] at same
  exact congrArg serialize same
end ProjectEulerSolutions.P24
