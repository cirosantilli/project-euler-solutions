import ProjectEulerSolutions.P71

set_option linter.style.header false

/-! The reduced ceiling scan and exhaustive search select the same unique fraction. -/
namespace ProjectEulerSolutions.P71

theorem equiv (limit num den : Int) (positive : 0 < num) (proper : num < den) :
    ProjectEulerStatements.P71.naive limit num den = solve limit num den :=
  (solve_eq_reference limit num den positive proper).symm

end ProjectEulerSolutions.P71
