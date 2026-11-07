import ProjectEulerSolutions.P47
namespace ProjectEulerSolutions.P47
open ConsecutiveFactors

theorem search_correct (k maximum : Nat) (positive : 0 < k) (nonzero : 0 < maximum) :
    search k maximum positive nonzero = least k := by
  rw [search]
  split
  · rename_i answer found
    have spec := scan_initial k maximum positive
    rw [found] at spec
    split at spec
    · exact Option.some.inj spec
    · contradiction
  · exact search_correct k (2*maximum) positive (by omega)
termination_by lastIndex k + 1 - maximum
decreasing_by
  have missing : scan k maximum 2 0 (P47FactorSieve.compute maximum) = none := by assumption
  have := missing_bound k maximum positive missing
  omega

/-- Changing the initial sieve size does not change the first qualifying run. -/
theorem equiv (k startLimit : Nat) (positive : 0 < k) (nonzero : 0 < startLimit) :
    ProjectEulerStatements.P47.naive k = solve k startLimit positive nonzero :=
  (search_correct k startLimit positive nonzero).symm
end ProjectEulerSolutions.P47
