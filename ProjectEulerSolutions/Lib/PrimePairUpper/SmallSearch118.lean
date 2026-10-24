import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch117
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch118

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan328 : scan edge (search edge 4) 5 [] 0 tail328 best = best := by
  have cons : tail328 = 2221 :: tail329 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates328_checked]
  change scan edge (search edge 4) 5 [] 0 tail329
    (if 4 ≤ rootCandidates328.length then search edge 4 [2221] rootCandidates328 2221 best else best) = best
  rw [if_pos (by decide), rootBranch328]
  exact outerScan329

set_option maxRecDepth 4096 in
theorem outerScan327 : scan edge (search edge 4) 5 [] 0 tail327 best = best := by
  have cons : tail327 = 2213 :: tail328 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates327_checked]
  change scan edge (search edge 4) 5 [] 0 tail328
    (if 4 ≤ rootCandidates327.length then search edge 4 [2213] rootCandidates327 2213 best else best) = best
  rw [if_pos (by decide), rootBranch327]
  exact outerScan328

set_option maxRecDepth 4096 in
theorem outerScan326 : scan edge (search edge 4) 5 [] 0 tail326 best = best := by
  have cons : tail326 = 2207 :: tail327 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates326_checked]
  change scan edge (search edge 4) 5 [] 0 tail327
    (if 4 ≤ rootCandidates326.length then search edge 4 [2207] rootCandidates326 2207 best else best) = best
  rw [if_pos (by decide), rootBranch326]
  exact outerScan327

set_option maxRecDepth 4096 in
theorem outerScan325 : scan edge (search edge 4) 5 [] 0 tail325 best = best := by
  have cons : tail325 = 2203 :: tail326 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates325_checked]
  change scan edge (search edge 4) 5 [] 0 tail326
    (if 4 ≤ rootCandidates325.length then search edge 4 [2203] rootCandidates325 2203 best else best) = best
  rw [if_pos (by decide), rootBranch325]
  exact outerScan326

end ProjectEulerSolutions.PrimePairUpper.Search
