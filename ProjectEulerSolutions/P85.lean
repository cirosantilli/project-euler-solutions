import ProjectEulerStatements.P85
namespace ProjectEulerSolutions.P85

/-- Invert the triangular count and check only the two nearest columns in each row. -/
def solve (target : Int) : Nat :=
  let grid := ProjectEulerSolutions.RectangleSearch.optimizedGrid target
  grid.1 * grid.2
end ProjectEulerSolutions.P85
