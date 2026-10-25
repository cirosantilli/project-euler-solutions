import ProjectEulerSolutions.Lib.PolygonalCycleReference.Data

/-! Independent reference roots keep the kernel check within its memory cap. -/
namespace ProjectEulerSolutions.PolygonalCycle.Reference.Part020
def starts : List Int := [7875, 8001, 8128, 8256]
def results : List Int := [0, 0, 0, 28684]
set_option maxRecDepth 4096 in
set_option maxHeartbeats 2000000 in
-- Four exhaustive reference roots are evaluated by the kernel together.
theorem checked : starts.map visit = results := by decide
end ProjectEulerSolutions.PolygonalCycle.Reference.Part020
