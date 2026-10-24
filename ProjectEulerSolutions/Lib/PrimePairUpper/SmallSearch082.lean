import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch081
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch082

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan469 : scan edge (search edge 4) 5 [] 0 tail469 best = best := by
  have cons : tail469 = 3347 :: tail470 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates469_checked]
  change scan edge (search edge 4) 5 [] 0 tail470
    (if 4 ≤ rootCandidates469.length then search edge 4 [3347] rootCandidates469 3347 best else best) = best
  rw [if_pos (by decide), rootBranch469]
  exact outerScan470

set_option maxRecDepth 4096 in
theorem outerScan468 : scan edge (search edge 4) 5 [] 0 tail468 best = best := by
  have cons : tail468 = 3343 :: tail469 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates468_checked]
  change scan edge (search edge 4) 5 [] 0 tail469
    (if 4 ≤ rootCandidates468.length then search edge 4 [3343] rootCandidates468 3343 best else best) = best
  rw [if_pos (by decide), rootBranch468]
  exact outerScan469

set_option maxRecDepth 4096 in
theorem outerScan467 : scan edge (search edge 4) 5 [] 0 tail467 best = best := by
  have cons : tail467 = 3331 :: tail468 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates467_checked]
  change scan edge (search edge 4) 5 [] 0 tail468
    (if 4 ≤ rootCandidates467.length then search edge 4 [3331] rootCandidates467 3331 best else best) = best
  rw [if_pos (by decide), rootBranch467]
  exact outerScan468

set_option maxRecDepth 4096 in
theorem outerScan466 : scan edge (search edge 4) 5 [] 0 tail466 best = best := by
  have cons : tail466 = 3329 :: tail467 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates466_checked]
  change scan edge (search edge 4) 5 [] 0 tail467
    (if 4 ≤ rootCandidates466.length then search edge 4 [3329] rootCandidates466 3329 best else best) = best
  rw [if_pos (by decide), rootBranch466]
  exact outerScan467

end ProjectEulerSolutions.PrimePairUpper.Search
