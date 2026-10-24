import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch099
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch100

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan395 : scan edge (search edge 4) 5 [] 0 tail395 best = best := by
  have cons : tail395 = 2729 :: tail396 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates395_checked]
  change scan edge (search edge 4) 5 [] 0 tail396
    (if 4 ≤ rootCandidates395.length then search edge 4 [2729] rootCandidates395 2729 best else best) = best
  rw [if_pos (by decide), rootBranch395]
  exact outerScan396

set_option maxRecDepth 4096 in
theorem outerScan394 : scan edge (search edge 4) 5 [] 0 tail394 best = best := by
  have cons : tail394 = 2719 :: tail395 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates394_checked]
  change scan edge (search edge 4) 5 [] 0 tail395
    (if 4 ≤ rootCandidates394.length then search edge 4 [2719] rootCandidates394 2719 best else best) = best
  rw [if_pos (by decide), rootBranch394]
  exact outerScan395

set_option maxRecDepth 4096 in
theorem outerScan393 : scan edge (search edge 4) 5 [] 0 tail393 best = best := by
  have cons : tail393 = 2713 :: tail394 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates393_checked]
  change scan edge (search edge 4) 5 [] 0 tail394
    (if 4 ≤ rootCandidates393.length then search edge 4 [2713] rootCandidates393 2713 best else best) = best
  rw [if_pos (by decide), rootBranch393]
  exact outerScan394

set_option maxRecDepth 4096 in
theorem outerScan392 : scan edge (search edge 4) 5 [] 0 tail392 best = best := by
  have cons : tail392 = 2711 :: tail393 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates392_checked]
  change scan edge (search edge 4) 5 [] 0 tail393
    (if 4 ≤ rootCandidates392.length then search edge 4 [2711] rootCandidates392 2711 best else best) = best
  rw [if_pos (by decide), rootBranch392]
  exact outerScan393

set_option maxRecDepth 4096 in
theorem outerScan391 : scan edge (search edge 4) 5 [] 0 tail391 best = best := by
  have cons : tail391 = 2707 :: tail392 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates391_checked]
  change scan edge (search edge 4) 5 [] 0 tail392
    (if 4 ≤ rootCandidates391.length then search edge 4 [2707] rootCandidates391 2707 best else best) = best
  rw [if_pos (by decide), rootBranch391]
  exact outerScan392

end ProjectEulerSolutions.PrimePairUpper.Search
