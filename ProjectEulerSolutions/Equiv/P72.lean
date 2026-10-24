import ProjectEulerSolutions.P72

set_option linter.style.header false

/-! Exhaustive residue counts and the indexed totient sieve agree for every signed bound. -/
namespace ProjectEulerSolutions.P72

theorem equiv (limit : Int) : ProjectEulerStatements.P72.naive limit = solve limit :=
  (ProjectEulerStatements.P72.naive_eq_result limit).trans (solve_eq_result limit).symm

end ProjectEulerSolutions.P72
