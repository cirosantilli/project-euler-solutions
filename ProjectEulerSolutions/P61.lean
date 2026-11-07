import ProjectEulerStatements.P61
import ProjectEulerSolutions.Lib.PolygonalCycleSearch

namespace ProjectEulerSolutions.P61

/-- Index the polygonal families by prefix before the finite depth-first search. -/
def solve (maxSides : Nat) : Nat :=
  (PolygonalCycleSearch.optimized PolygonalFamilies.family
    (PolygonalCycleSearch.indexed PolygonalFamilies.family) maxSides).toNat

end ProjectEulerSolutions.P61
