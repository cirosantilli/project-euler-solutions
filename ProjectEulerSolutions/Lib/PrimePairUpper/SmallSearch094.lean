import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch093
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch094

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan416 : scan edge (search edge 4) 5 [] 0 tail416 best = best := by
  have cons : tail416 = 2897 :: tail417 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates416_checked]
  change scan edge (search edge 4) 5 [] 0 tail417
    (if 4 ≤ rootCandidates416.length then search edge 4 [2897] rootCandidates416 2897 best else best) = best
  rw [if_pos (by decide), rootBranch416]
  exact outerScan417

set_option maxRecDepth 4096 in
theorem outerScan415 : scan edge (search edge 4) 5 [] 0 tail415 best = best := by
  have cons : tail415 = 2887 :: tail416 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates415_checked]
  change scan edge (search edge 4) 5 [] 0 tail416
    (if 4 ≤ rootCandidates415.length then search edge 4 [2887] rootCandidates415 2887 best else best) = best
  rw [if_pos (by decide), rootBranch415]
  exact outerScan416

set_option maxRecDepth 4096 in
theorem outerScan414 : scan edge (search edge 4) 5 [] 0 tail414 best = best := by
  have cons : tail414 = 2879 :: tail415 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates414_checked]
  change scan edge (search edge 4) 5 [] 0 tail415
    (if 4 ≤ rootCandidates414.length then search edge 4 [2879] rootCandidates414 2879 best else best) = best
  rw [if_pos (by decide), rootBranch414]
  exact outerScan415

set_option maxRecDepth 4096 in
theorem outerScan413 : scan edge (search edge 4) 5 [] 0 tail413 best = best := by
  have cons : tail413 = 2861 :: tail414 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates413_checked]
  change scan edge (search edge 4) 5 [] 0 tail414
    (if 4 ≤ rootCandidates413.length then search edge 4 [2861] rootCandidates413 2861 best else best) = best
  rw [if_pos (by decide), rootBranch413]
  exact outerScan414

end ProjectEulerSolutions.PrimePairUpper.Search
