import ProjectEulerSolutions.Lib.PolygonalCycle

namespace ProjectEulerSolutions.PolygonalCycle
set_option maxRecDepth 4096 in
set_option maxHeartbeats 2000000 in
-- The kernel evaluates the complete ordered DFS, including unsuccessful starts.
theorem cycle_checked :
    cycle = some [(1281, 8), (8128, 6), (2882, 5), (8256, 3), (5625, 4), (2512, 7)] := by decide

theorem answer_checked : answer = 28684 := by
  rw [answer, cycle_checked]
  decide
end ProjectEulerSolutions.PolygonalCycle
