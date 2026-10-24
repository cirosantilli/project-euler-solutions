import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch046
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch047

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan674 : scan edge (search edge 4) 5 [] 0 tail674 best = best := by
  have cons : tail674 = 5059 :: tail675 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates674_checked]
  change scan edge (search edge 4) 5 [] 0 tail675
    (if 4 ≤ rootCandidates674.length then search edge 4 [5059] rootCandidates674 5059 best else best) = best
  rw [if_pos (by decide), rootBranch674]
  exact outerScan675

set_option maxRecDepth 4096 in
theorem outerScan673 : scan edge (search edge 4) 5 [] 0 tail673 best = best := by
  have cons : tail673 = 5051 :: tail674 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates673_checked]
  change scan edge (search edge 4) 5 [] 0 tail674
    (if 4 ≤ rootCandidates673.length then search edge 4 [5051] rootCandidates673 5051 best else best) = best
  rw [if_pos (by decide), rootBranch673]
  exact outerScan674

set_option maxRecDepth 4096 in
theorem outerScan672 : scan edge (search edge 4) 5 [] 0 tail672 best = best := by
  have cons : tail672 = 5039 :: tail673 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates672_checked]
  change scan edge (search edge 4) 5 [] 0 tail673
    (if 4 ≤ rootCandidates672.length then search edge 4 [5039] rootCandidates672 5039 best else best) = best
  rw [if_pos (by decide), rootBranch672]
  exact outerScan673

set_option maxRecDepth 4096 in
theorem outerScan671 : scan edge (search edge 4) 5 [] 0 tail671 best = best := by
  have cons : tail671 = 5023 :: tail672 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates671_checked]
  change scan edge (search edge 4) 5 [] 0 tail672
    (if 4 ≤ rootCandidates671.length then search edge 4 [5023] rootCandidates671 5023 best else best) = best
  rw [if_pos (by decide), rootBranch671]
  exact outerScan672

set_option maxRecDepth 4096 in
theorem outerScan670 : scan edge (search edge 4) 5 [] 0 tail670 best = best := by
  have cons : tail670 = 5021 :: tail671 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates670_checked]
  change scan edge (search edge 4) 5 [] 0 tail671
    (if 4 ≤ rootCandidates670.length then search edge 4 [5021] rootCandidates670 5021 best else best) = best
  rw [if_pos (by decide), rootBranch670]
  exact outerScan671

set_option maxRecDepth 4096 in
theorem outerScan669 : scan edge (search edge 4) 5 [] 0 tail669 best = best := by
  have cons : tail669 = 5011 :: tail670 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates669_checked]
  change scan edge (search edge 4) 5 [] 0 tail670
    (if 4 ≤ rootCandidates669.length then search edge 4 [5011] rootCandidates669 5011 best else best) = best
  rw [if_pos (by decide), rootBranch669]
  exact outerScan670

set_option maxRecDepth 4096 in
theorem outerScan668 : scan edge (search edge 4) 5 [] 0 tail668 best = best := by
  have cons : tail668 = 5009 :: tail669 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates668_checked]
  change scan edge (search edge 4) 5 [] 0 tail669
    (if 4 ≤ rootCandidates668.length then search edge 4 [5009] rootCandidates668 5009 best else best) = best
  rw [if_pos (by decide), rootBranch668]
  exact outerScan669

end ProjectEulerSolutions.PrimePairUpper.Search
