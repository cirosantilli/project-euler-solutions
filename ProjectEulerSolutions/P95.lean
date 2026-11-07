import ProjectEulerStatements.P95

namespace ProjectEulerSolutions.P95

/-- Cache processed nodes so each bounded divisor orbit is traversed only once. -/
def solve (limit : Int) : Nat :=
  (ProjectEulerSolutions.BoundedCycleWalk.solve ProjectEulerStatements.P21.d limit.toNat).2

end ProjectEulerSolutions.P95
