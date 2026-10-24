import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch057
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch058

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan600 : scan edge (search edge 4) 5 [] 0 tail600 best = best := by
  have cons : tail600 = 4441 :: tail601 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates600_checked]
  change scan edge (search edge 4) 5 [] 0 tail601
    (if 4 ≤ rootCandidates600.length then search edge 4 [4441] rootCandidates600 4441 best else best) = best
  rw [if_pos (by decide), rootBranch600]
  exact outerScan601

set_option maxRecDepth 4096 in
theorem outerScan599 : scan edge (search edge 4) 5 [] 0 tail599 best = best := by
  have cons : tail599 = 4423 :: tail600 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates599_checked]
  change scan edge (search edge 4) 5 [] 0 tail600
    (if 4 ≤ rootCandidates599.length then search edge 4 [4423] rootCandidates599 4423 best else best) = best
  rw [if_pos (by decide), rootBranch599]
  exact outerScan600

set_option maxRecDepth 4096 in
theorem outerScan598 : scan edge (search edge 4) 5 [] 0 tail598 best = best := by
  have cons : tail598 = 4421 :: tail599 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates598_checked]
  change scan edge (search edge 4) 5 [] 0 tail599
    (if 4 ≤ rootCandidates598.length then search edge 4 [4421] rootCandidates598 4421 best else best) = best
  rw [if_pos (by decide), rootBranch598]
  exact outerScan599

set_option maxRecDepth 4096 in
theorem outerScan597 : scan edge (search edge 4) 5 [] 0 tail597 best = best := by
  have cons : tail597 = 4409 :: tail598 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates597_checked]
  change scan edge (search edge 4) 5 [] 0 tail598
    (if 4 ≤ rootCandidates597.length then search edge 4 [4409] rootCandidates597 4409 best else best) = best
  rw [if_pos (by decide), rootBranch597]
  exact outerScan598

set_option maxRecDepth 4096 in
theorem outerScan596 : scan edge (search edge 4) 5 [] 0 tail596 best = best := by
  have cons : tail596 = 4397 :: tail597 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates596_checked]
  change scan edge (search edge 4) 5 [] 0 tail597
    (if 4 ≤ rootCandidates596.length then search edge 4 [4397] rootCandidates596 4397 best else best) = best
  rw [if_pos (by decide), rootBranch596]
  exact outerScan597

set_option maxRecDepth 4096 in
theorem outerScan595 : scan edge (search edge 4) 5 [] 0 tail595 best = best := by
  have cons : tail595 = 4391 :: tail596 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates595_checked]
  change scan edge (search edge 4) 5 [] 0 tail596
    (if 4 ≤ rootCandidates595.length then search edge 4 [4391] rootCandidates595 4391 best else best) = best
  rw [if_pos (by decide), rootBranch595]
  exact outerScan596

set_option maxRecDepth 4096 in
theorem outerScan594 : scan edge (search edge 4) 5 [] 0 tail594 best = best := by
  have cons : tail594 = 4373 :: tail595 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates594_checked]
  change scan edge (search edge 4) 5 [] 0 tail595
    (if 4 ≤ rootCandidates594.length then search edge 4 [4373] rootCandidates594 4373 best else best) = best
  rw [if_pos (by decide), rootBranch594]
  exact outerScan595

end ProjectEulerSolutions.PrimePairUpper.Search
