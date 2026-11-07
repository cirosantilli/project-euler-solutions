import ProjectEulerSolutions.P21
namespace ProjectEulerSolutions.P21
open ProjectEulerStatements.P21

theorem sigmaFast_correct (n : Nat) : sigmaFast n = ProjectEulerStatements.DivisorSumFactors.sigma n := by
  rw [sigmaFast]
  split
  · rename_i small
    interval_cases n <;> simp [ProjectEulerStatements.DivisorSumFactors.sigma]
  · rename_i large
    dsimp only
    rw [sigmaFast_correct]
    exact (ProjectEulerStatements.DivisorSumFactors.strip_sigma n n.minFac (by omega)
      (Nat.minFac_prime (by omega))).symm
termination_by n
decreasing_by exact strip_smaller n (by omega)


theorem cachedSum_correct (maximum n : Nat) : cachedSum (P21DivisorSieve.compute maximum) n = d n := by
  rw [P21DivisorSieve.compute_correct]
  unfold cachedSum
  split
  · rename_i inside
    simpa [P21DivisorSieve.encode] using
      (show (P21DivisorSieve.encode (maximum+1) d)[n]! = d n by
        simp [P21DivisorSieve.encode, show n < maximum+1 by simpa using inside])
  · rw [sigmaFast_correct]; exact (ProjectEulerStatements.DivisorSumFactors.proper_sum_eq n).symm

theorem equiv (limit : Nat) : naive limit = solve limit := by
  simp only [solve, cachedSum_correct]
  let items := (List.range' 1 (limit-1)).filter (fun n => decide (isAmicable n))
  have same : items.toFinset = (Finset.Icc 1 (limit-1)).filter isAmicable := by
    ext n
    simp only [items, List.mem_toFinset, List.mem_filter, List.mem_range',
      Finset.mem_filter, Finset.mem_Icc, decide_eq_true_eq]
    constructor
    · rintro ⟨⟨k, bound, rfl⟩, accepted⟩
      exact ⟨⟨by omega, by omega⟩, accepted⟩
    · rintro ⟨⟨lower, upper⟩, accepted⟩
      exact ⟨⟨n-1, by omega, by omega⟩, accepted⟩
  change ((Finset.Icc 1 (limit-1)).filter isAmicable).sum id = items.sum
  rw [← same, List.sum_toFinset]
  · simp
  · exact (List.nodup_range' (s := 1) (n := limit-1)).filter _
end ProjectEulerSolutions.P21
