import ProjectEulerSolutions.P85
namespace ProjectEulerSolutions.P85

theorem equiv (target : Int) : ProjectEulerStatements.P85.naive target = solve target := by
  unfold ProjectEulerStatements.P85.naive solve
  rw [ProjectEulerSolutions.RectangleSearch.grids_equal]
end ProjectEulerSolutions.P85
