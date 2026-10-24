import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch183
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch184

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan139 : scan edge (search edge 4) 5 [] 0 tail139 best = best := by
  have cons : tail139 = 821 :: tail140 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates139_checked]
  change scan edge (search edge 4) 5 [] 0 tail140
    (if 4 ≤ rootCandidates139.length then search edge 4 [821] rootCandidates139 821 best else best) = best
  rw [if_pos (by decide), rootBranch139]
  exact outerScan140

set_option maxRecDepth 4096 in
theorem outerScan138 : scan edge (search edge 4) 5 [] 0 tail138 best = best := by
  have cons : tail138 = 811 :: tail139 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates138_checked]
  change scan edge (search edge 4) 5 [] 0 tail139
    (if 4 ≤ rootCandidates138.length then search edge 4 [811] rootCandidates138 811 best else best) = best
  rw [if_pos (by decide), rootBranch138]
  exact outerScan139

end ProjectEulerSolutions.PrimePairUpper.Search
