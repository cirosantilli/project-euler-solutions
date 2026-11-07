import ProjectEulerStatements.P62

namespace ProjectEulerSolutions.P62

/-- Incremental count/minimum buckets, cleared only after a complete digit width. -/
def solve (target : Int) : Option Int :=
  ProjectEulerStatements.UnboundedSearch.run (CubicSearch.step target) CubicSearch.start

end ProjectEulerSolutions.P62
