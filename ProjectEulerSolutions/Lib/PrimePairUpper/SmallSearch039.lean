import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch038
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch039

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan736 : scan edge (search edge 4) 5 [] 0 tail736 best = best := by
  have cons : tail736 = 5623 :: tail737 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates736_checked]
  change scan edge (search edge 4) 5 [] 0 tail737
    (if 4 ≤ rootCandidates736.length then search edge 4 [5623] rootCandidates736 5623 best else best) = best
  rw [if_pos (by decide), rootBranch736]
  exact outerScan737

set_option maxRecDepth 4096 in
theorem outerScan735 : scan edge (search edge 4) 5 [] 0 tail735 best = best := by
  have cons : tail735 = 5591 :: tail736 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates735_checked]
  change scan edge (search edge 4) 5 [] 0 tail736
    (if 4 ≤ rootCandidates735.length then search edge 4 [5591] rootCandidates735 5591 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan736

set_option maxRecDepth 4096 in
theorem outerScan734 : scan edge (search edge 4) 5 [] 0 tail734 best = best := by
  have cons : tail734 = 5581 :: tail735 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates734_checked]
  change scan edge (search edge 4) 5 [] 0 tail735
    (if 4 ≤ rootCandidates734.length then search edge 4 [5581] rootCandidates734 5581 best else best) = best
  rw [if_pos (by decide), rootBranch734]
  exact outerScan735

set_option maxRecDepth 4096 in
theorem outerScan733 : scan edge (search edge 4) 5 [] 0 tail733 best = best := by
  have cons : tail733 = 5573 :: tail734 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates733_checked]
  change scan edge (search edge 4) 5 [] 0 tail734
    (if 4 ≤ rootCandidates733.length then search edge 4 [5573] rootCandidates733 5573 best else best) = best
  rw [if_pos (by decide), rootBranch733]
  exact outerScan734

set_option maxRecDepth 4096 in
theorem outerScan732 : scan edge (search edge 4) 5 [] 0 tail732 best = best := by
  have cons : tail732 = 5569 :: tail733 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates732_checked]
  change scan edge (search edge 4) 5 [] 0 tail733
    (if 4 ≤ rootCandidates732.length then search edge 4 [5569] rootCandidates732 5569 best else best) = best
  rw [if_pos (by decide), rootBranch732]
  exact outerScan733

set_option maxRecDepth 4096 in
theorem outerScan731 : scan edge (search edge 4) 5 [] 0 tail731 best = best := by
  have cons : tail731 = 5563 :: tail732 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates731_checked]
  change scan edge (search edge 4) 5 [] 0 tail732
    (if 4 ≤ rootCandidates731.length then search edge 4 [5563] rootCandidates731 5563 best else best) = best
  rw [if_pos (by decide), rootBranch731]
  exact outerScan732

set_option maxRecDepth 4096 in
theorem outerScan730 : scan edge (search edge 4) 5 [] 0 tail730 best = best := by
  have cons : tail730 = 5557 :: tail731 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates730_checked]
  change scan edge (search edge 4) 5 [] 0 tail731
    (if 4 ≤ rootCandidates730.length then search edge 4 [5557] rootCandidates730 5557 best else best) = best
  rw [if_pos (by decide), rootBranch730]
  exact outerScan731

set_option maxRecDepth 4096 in
theorem outerScan729 : scan edge (search edge 4) 5 [] 0 tail729 best = best := by
  have cons : tail729 = 5531 :: tail730 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates729_checked]
  change scan edge (search edge 4) 5 [] 0 tail730
    (if 4 ≤ rootCandidates729.length then search edge 4 [5531] rootCandidates729 5531 best else best) = best
  rw [if_pos (by decide), rootBranch729]
  exact outerScan730

end ProjectEulerSolutions.PrimePairUpper.Search
