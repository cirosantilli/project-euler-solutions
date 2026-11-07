import ProjectEulerSolutions.P61

namespace ProjectEulerSolutions.P61

theorem equiv (maxSides : Nat) : ProjectEulerStatements.P61.naive maxSides = solve maxSides := by
  unfold ProjectEulerStatements.P61.naive solve
  rw [PolygonalCycleSearch.models_equal _ _ (fun _ _ => rfl)]

end ProjectEulerSolutions.P61
