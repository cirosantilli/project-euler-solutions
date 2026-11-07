import ProjectEulerStatements.P60

namespace ProjectEulerSolutions.P60

/-- Ordered clique search with compatibility filtering and sum-bound pruning. -/
def solve (limit : Nat) : CliqueSearch.Best :=
  CliqueSearch.search ProjectEulerStatements.P60.goodPair 5 []
    (ProjectEulerStatements.P60.primes limit) 0 ⟨10^30, []⟩

end ProjectEulerSolutions.P60
