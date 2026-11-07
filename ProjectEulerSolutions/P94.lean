import ProjectEulerStatements.P94

namespace ProjectEulerSolutions.P94

/-- Enumerate the increasing Pell family, stopping at the first excessive perimeter. -/
def solve (limit : Int) : Int :=
  (ProjectEulerSolutions.AlmostEquilateral.perimetersFrom limit 2 (by decide)).sum

end ProjectEulerSolutions.P94
