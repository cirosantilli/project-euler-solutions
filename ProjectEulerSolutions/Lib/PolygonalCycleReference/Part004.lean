import ProjectEulerSolutions.Lib.PolygonalCycleReference.Data

/-! Independent reference roots keep the kernel check within its memory cap. -/
namespace ProjectEulerSolutions.PolygonalCycle.Reference.Part004
def starts : List Int := [1891, 1953, 2016, 2080]
def results : List Int := [0, 0, 0, 0]
set_option maxRecDepth 4096 in
set_option maxHeartbeats 2000000 in
-- Four exhaustive reference roots are evaluated by the kernel together.
theorem checked : starts.map visit = results := by decide
end ProjectEulerSolutions.PolygonalCycle.Reference.Part004
