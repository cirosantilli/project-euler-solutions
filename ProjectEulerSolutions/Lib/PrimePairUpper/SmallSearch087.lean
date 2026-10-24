import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch086
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch087

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan447 : scan edge (search edge 4) 5 [] 0 tail447 best = best := by
  have cons : tail447 = 3181 :: tail448 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates447_checked]
  change scan edge (search edge 4) 5 [] 0 tail448
    (if 4 ≤ rootCandidates447.length then search edge 4 [3181] rootCandidates447 3181 best else best) = best
  rw [if_pos (by decide), rootBranch447]
  exact outerScan448

set_option maxRecDepth 4096 in
theorem outerScan446 : scan edge (search edge 4) 5 [] 0 tail446 best = best := by
  have cons : tail446 = 3169 :: tail447 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates446_checked]
  change scan edge (search edge 4) 5 [] 0 tail447
    (if 4 ≤ rootCandidates446.length then search edge 4 [3169] rootCandidates446 3169 best else best) = best
  rw [if_pos (by decide), rootBranch446]
  exact outerScan447

set_option maxRecDepth 4096 in
theorem outerScan445 : scan edge (search edge 4) 5 [] 0 tail445 best = best := by
  have cons : tail445 = 3167 :: tail446 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates445_checked]
  change scan edge (search edge 4) 5 [] 0 tail446
    (if 4 ≤ rootCandidates445.length then search edge 4 [3167] rootCandidates445 3167 best else best) = best
  rw [if_pos (by decide), rootBranch445]
  exact outerScan446

set_option maxRecDepth 4096 in
theorem outerScan444 : scan edge (search edge 4) 5 [] 0 tail444 best = best := by
  have cons : tail444 = 3163 :: tail445 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates444_checked]
  change scan edge (search edge 4) 5 [] 0 tail445
    (if 4 ≤ rootCandidates444.length then search edge 4 [3163] rootCandidates444 3163 best else best) = best
  rw [if_pos (by decide), rootBranch444]
  exact outerScan445

set_option maxRecDepth 4096 in
theorem outerScan443 : scan edge (search edge 4) 5 [] 0 tail443 best = best := by
  have cons : tail443 = 3137 :: tail444 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates443_checked]
  change scan edge (search edge 4) 5 [] 0 tail444
    (if 4 ≤ rootCandidates443.length then search edge 4 [3137] rootCandidates443 3137 best else best) = best
  rw [if_pos (by decide), rootBranch443]
  exact outerScan444

end ProjectEulerSolutions.PrimePairUpper.Search
