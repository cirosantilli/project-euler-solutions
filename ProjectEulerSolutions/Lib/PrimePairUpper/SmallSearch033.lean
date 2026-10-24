import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch032
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch033

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan783 : scan edge (search edge 4) 5 [] 0 tail783 best = best := by
  have cons : tail783 = 6029 :: tail784 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates783_checked]
  change scan edge (search edge 4) 5 [] 0 tail784
    (if 4 ≤ rootCandidates783.length then search edge 4 [6029] rootCandidates783 6029 best else best) = best
  rw [if_pos (by decide), rootBranch783]
  exact outerScan784

set_option maxRecDepth 4096 in
theorem outerScan782 : scan edge (search edge 4) 5 [] 0 tail782 best = best := by
  have cons : tail782 = 6011 :: tail783 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates782_checked]
  change scan edge (search edge 4) 5 [] 0 tail783
    (if 4 ≤ rootCandidates782.length then search edge 4 [6011] rootCandidates782 6011 best else best) = best
  rw [if_pos (by decide), rootBranch782]
  exact outerScan783

set_option maxRecDepth 4096 in
theorem outerScan781 : scan edge (search edge 4) 5 [] 0 tail781 best = best := by
  have cons : tail781 = 6007 :: tail782 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates781_checked]
  change scan edge (search edge 4) 5 [] 0 tail782
    (if 4 ≤ rootCandidates781.length then search edge 4 [6007] rootCandidates781 6007 best else best) = best
  rw [if_pos (by decide), rootBranch781]
  exact outerScan782

set_option maxRecDepth 4096 in
theorem outerScan780 : scan edge (search edge 4) 5 [] 0 tail780 best = best := by
  have cons : tail780 = 5987 :: tail781 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates780_checked]
  change scan edge (search edge 4) 5 [] 0 tail781
    (if 4 ≤ rootCandidates780.length then search edge 4 [5987] rootCandidates780 5987 best else best) = best
  rw [if_pos (by decide), rootBranch780]
  exact outerScan781

set_option maxRecDepth 4096 in
theorem outerScan779 : scan edge (search edge 4) 5 [] 0 tail779 best = best := by
  have cons : tail779 = 5981 :: tail780 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates779_checked]
  change scan edge (search edge 4) 5 [] 0 tail780
    (if 4 ≤ rootCandidates779.length then search edge 4 [5981] rootCandidates779 5981 best else best) = best
  rw [if_pos (by decide), rootBranch779]
  exact outerScan780

set_option maxRecDepth 4096 in
theorem outerScan778 : scan edge (search edge 4) 5 [] 0 tail778 best = best := by
  have cons : tail778 = 5953 :: tail779 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates778_checked]
  change scan edge (search edge 4) 5 [] 0 tail779
    (if 4 ≤ rootCandidates778.length then search edge 4 [5953] rootCandidates778 5953 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan779

set_option maxRecDepth 4096 in
theorem outerScan777 : scan edge (search edge 4) 5 [] 0 tail777 best = best := by
  have cons : tail777 = 5939 :: tail778 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates777_checked]
  change scan edge (search edge 4) 5 [] 0 tail778
    (if 4 ≤ rootCandidates777.length then search edge 4 [5939] rootCandidates777 5939 best else best) = best
  rw [if_pos (by decide), rootBranch777]
  exact outerScan778

set_option maxRecDepth 4096 in
theorem outerScan776 : scan edge (search edge 4) 5 [] 0 tail776 best = best := by
  have cons : tail776 = 5927 :: tail777 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates776_checked]
  change scan edge (search edge 4) 5 [] 0 tail777
    (if 4 ≤ rootCandidates776.length then search edge 4 [5927] rootCandidates776 5927 best else best) = best
  rw [if_pos (by decide), rootBranch776]
  exact outerScan777

end ProjectEulerSolutions.PrimePairUpper.Search
