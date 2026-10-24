import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch148
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch149

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan229 : scan edge (search edge 4) 5 [] 0 tail229 best = best := by
  have cons : tail229 = 1459 :: tail230 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates229_checked]
  change scan edge (search edge 4) 5 [] 0 tail230
    (if 4 ≤ rootCandidates229.length then search edge 4 [1459] rootCandidates229 1459 best else best) = best
  rw [if_pos (by decide), rootBranch229]
  exact outerScan230

set_option maxRecDepth 4096 in
theorem outerScan228 : scan edge (search edge 4) 5 [] 0 tail228 best = best := by
  have cons : tail228 = 1453 :: tail229 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates228_checked]
  change scan edge (search edge 4) 5 [] 0 tail229
    (if 4 ≤ rootCandidates228.length then search edge 4 [1453] rootCandidates228 1453 best else best) = best
  rw [if_pos (by decide), rootBranch228]
  exact outerScan229

set_option maxRecDepth 4096 in
theorem outerScan227 : scan edge (search edge 4) 5 [] 0 tail227 best = best := by
  have cons : tail227 = 1451 :: tail228 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates227_checked]
  change scan edge (search edge 4) 5 [] 0 tail228
    (if 4 ≤ rootCandidates227.length then search edge 4 [1451] rootCandidates227 1451 best else best) = best
  rw [if_pos (by decide), rootBranch227]
  exact outerScan228

end ProjectEulerSolutions.PrimePairUpper.Search
