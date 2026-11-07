import ProjectEulerSolutions.P60

namespace ProjectEulerSolutions.P60

theorem equiv (limit : Nat) : ProjectEulerStatements.P60.naive limit = solve limit := by
  apply CliqueEnumeration.exhaustive_eq_search
  exact (List.pairwise_lt_range (n := limit + 1)).filter _

end ProjectEulerSolutions.P60
