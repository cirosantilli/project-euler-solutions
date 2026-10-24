import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch041
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch042

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan712 : scan edge (search edge 4) 5 [] 0 tail712 best = best := by
  have cons : tail712 = 5417 :: tail713 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates712_checked]
  change scan edge (search edge 4) 5 [] 0 tail713
    (if 4 ≤ rootCandidates712.length then search edge 4 [5417] rootCandidates712 5417 best else best) = best
  rw [if_pos (by decide), rootBranch712]
  exact outerScan713

set_option maxRecDepth 4096 in
theorem outerScan711 : scan edge (search edge 4) 5 [] 0 tail711 best = best := by
  have cons : tail711 = 5413 :: tail712 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates711_checked]
  change scan edge (search edge 4) 5 [] 0 tail712
    (if 4 ≤ rootCandidates711.length then search edge 4 [5413] rootCandidates711 5413 best else best) = best
  rw [if_pos (by decide), rootBranch711]
  exact outerScan712

set_option maxRecDepth 4096 in
theorem outerScan710 : scan edge (search edge 4) 5 [] 0 tail710 best = best := by
  have cons : tail710 = 5407 :: tail711 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates710_checked]
  change scan edge (search edge 4) 5 [] 0 tail711
    (if 4 ≤ rootCandidates710.length then search edge 4 [5407] rootCandidates710 5407 best else best) = best
  rw [if_pos (by decide), rootBranch710]
  exact outerScan711

set_option maxRecDepth 4096 in
theorem outerScan709 : scan edge (search edge 4) 5 [] 0 tail709 best = best := by
  have cons : tail709 = 5399 :: tail710 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates709_checked]
  change scan edge (search edge 4) 5 [] 0 tail710
    (if 4 ≤ rootCandidates709.length then search edge 4 [5399] rootCandidates709 5399 best else best) = best
  rw [if_pos (by decide), rootBranch709]
  exact outerScan710

set_option maxRecDepth 4096 in
theorem outerScan708 : scan edge (search edge 4) 5 [] 0 tail708 best = best := by
  have cons : tail708 = 5393 :: tail709 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates708_checked]
  change scan edge (search edge 4) 5 [] 0 tail709
    (if 4 ≤ rootCandidates708.length then search edge 4 [5393] rootCandidates708 5393 best else best) = best
  rw [if_pos (by decide), rootBranch708]
  exact outerScan709

set_option maxRecDepth 4096 in
theorem outerScan707 : scan edge (search edge 4) 5 [] 0 tail707 best = best := by
  have cons : tail707 = 5387 :: tail708 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates707_checked]
  change scan edge (search edge 4) 5 [] 0 tail708
    (if 4 ≤ rootCandidates707.length then search edge 4 [5387] rootCandidates707 5387 best else best) = best
  rw [if_pos (by decide), rootBranch707]
  exact outerScan708

set_option maxRecDepth 4096 in
theorem outerScan706 : scan edge (search edge 4) 5 [] 0 tail706 best = best := by
  have cons : tail706 = 5381 :: tail707 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates706_checked]
  change scan edge (search edge 4) 5 [] 0 tail707
    (if 4 ≤ rootCandidates706.length then search edge 4 [5381] rootCandidates706 5381 best else best) = best
  rw [if_pos (by decide), rootBranch706]
  exact outerScan707

set_option maxRecDepth 4096 in
theorem outerScan705 : scan edge (search edge 4) 5 [] 0 tail705 best = best := by
  have cons : tail705 = 5351 :: tail706 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates705_checked]
  change scan edge (search edge 4) 5 [] 0 tail706
    (if 4 ≤ rootCandidates705.length then search edge 4 [5351] rootCandidates705 5351 best else best) = best
  rw [if_pos (by decide), rootBranch705]
  exact outerScan706

end ProjectEulerSolutions.PrimePairUpper.Search
