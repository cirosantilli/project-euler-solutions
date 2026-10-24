import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch101
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch102

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan386 : scan edge (search edge 4) 5 [] 0 tail386 best = best := by
  have cons : tail386 = 2683 :: tail387 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates386_checked]
  change scan edge (search edge 4) 5 [] 0 tail387
    (if 4 ≤ rootCandidates386.length then search edge 4 [2683] rootCandidates386 2683 best else best) = best
  rw [if_pos (by decide), rootBranch386]
  exact outerScan387

set_option maxRecDepth 4096 in
theorem outerScan385 : scan edge (search edge 4) 5 [] 0 tail385 best = best := by
  have cons : tail385 = 2677 :: tail386 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates385_checked]
  change scan edge (search edge 4) 5 [] 0 tail386
    (if 4 ≤ rootCandidates385.length then search edge 4 [2677] rootCandidates385 2677 best else best) = best
  rw [if_pos (by decide), rootBranch385]
  exact outerScan386

set_option maxRecDepth 4096 in
theorem outerScan384 : scan edge (search edge 4) 5 [] 0 tail384 best = best := by
  have cons : tail384 = 2671 :: tail385 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates384_checked]
  change scan edge (search edge 4) 5 [] 0 tail385
    (if 4 ≤ rootCandidates384.length then search edge 4 [2671] rootCandidates384 2671 best else best) = best
  rw [if_pos (by decide), rootBranch384]
  exact outerScan385

set_option maxRecDepth 4096 in
theorem outerScan383 : scan edge (search edge 4) 5 [] 0 tail383 best = best := by
  have cons : tail383 = 2663 :: tail384 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates383_checked]
  change scan edge (search edge 4) 5 [] 0 tail384
    (if 4 ≤ rootCandidates383.length then search edge 4 [2663] rootCandidates383 2663 best else best) = best
  rw [if_pos (by decide), rootBranch383]
  exact outerScan384

end ProjectEulerSolutions.PrimePairUpper.Search
