import ProjectEulerSolutions.P69

set_option linter.style.header false

/-! The greedy product is the least maximizer of the reference's totient ratio. -/
namespace ProjectEulerSolutions.P69

theorem equiv (limit : Int) : ProjectEulerStatements.P69.naive limit = solve limit :=
  (ProjectEulerStatements.P69.naive_eq_reference limit).trans
    (ProjectEulerStatements.TotientMaximum.optimizedResult_eq limit).symm

end ProjectEulerSolutions.P69
