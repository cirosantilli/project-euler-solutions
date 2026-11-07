import ProjectEulerSolutions.P95
namespace ProjectEulerSolutions.P95

theorem equiv (limit : Int) : ProjectEulerStatements.P95.naive limit = solve limit := by
  apply congrArg Prod.snd
  apply ProjectEulerSolutions.BoundedCycleReference.scan_eq_optimized
  simp [ProjectEulerStatements.P21.d, ProjectEulerStatements.P21.properDivisors]

end ProjectEulerSolutions.P95
