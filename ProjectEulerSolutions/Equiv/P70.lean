import ProjectEulerSolutions.P70

set_option linter.style.header false

/-! The array solver and exhaustive reference share the same least ratio minimizer. -/
namespace ProjectEulerSolutions.P70

theorem equiv (limit : Int) : ProjectEulerStatements.P70.naive limit = solve limit :=
  (ProjectEulerStatements.P70.naive_eq_reference limit).trans (solve_eq_reference limit).symm

end ProjectEulerSolutions.P70
