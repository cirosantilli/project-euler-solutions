import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch023
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch024

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan855 : scan edge (search edge 4) 5 [] 0 tail855 best = best := by
  have cons : tail855 = 6659 :: tail856 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates855_checked]
  change scan edge (search edge 4) 5 [] 0 tail856
    (if 4 ≤ rootCandidates855.length then search edge 4 [6659] rootCandidates855 6659 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan856

set_option maxRecDepth 4096 in
theorem outerScan854 : scan edge (search edge 4) 5 [] 0 tail854 best = best := by
  have cons : tail854 = 6653 :: tail855 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates854_checked]
  change scan edge (search edge 4) 5 [] 0 tail855
    (if 4 ≤ rootCandidates854.length then search edge 4 [6653] rootCandidates854 6653 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan855

set_option maxRecDepth 4096 in
theorem outerScan853 : scan edge (search edge 4) 5 [] 0 tail853 best = best := by
  have cons : tail853 = 6637 :: tail854 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates853_checked]
  change scan edge (search edge 4) 5 [] 0 tail854
    (if 4 ≤ rootCandidates853.length then search edge 4 [6637] rootCandidates853 6637 best else best) = best
  rw [if_pos (by decide), rootBranch853]
  exact outerScan854

set_option maxRecDepth 4096 in
theorem outerScan852 : scan edge (search edge 4) 5 [] 0 tail852 best = best := by
  have cons : tail852 = 6619 :: tail853 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates852_checked]
  change scan edge (search edge 4) 5 [] 0 tail853
    (if 4 ≤ rootCandidates852.length then search edge 4 [6619] rootCandidates852 6619 best else best) = best
  rw [if_pos (by decide), rootBranch852]
  exact outerScan853

set_option maxRecDepth 4096 in
theorem outerScan851 : scan edge (search edge 4) 5 [] 0 tail851 best = best := by
  have cons : tail851 = 6607 :: tail852 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates851_checked]
  change scan edge (search edge 4) 5 [] 0 tail852
    (if 4 ≤ rootCandidates851.length then search edge 4 [6607] rootCandidates851 6607 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan852

set_option maxRecDepth 4096 in
theorem outerScan850 : scan edge (search edge 4) 5 [] 0 tail850 best = best := by
  have cons : tail850 = 6599 :: tail851 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates850_checked]
  change scan edge (search edge 4) 5 [] 0 tail851
    (if 4 ≤ rootCandidates850.length then search edge 4 [6599] rootCandidates850 6599 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan851

set_option maxRecDepth 4096 in
theorem outerScan849 : scan edge (search edge 4) 5 [] 0 tail849 best = best := by
  have cons : tail849 = 6581 :: tail850 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates849_checked]
  change scan edge (search edge 4) 5 [] 0 tail850
    (if 4 ≤ rootCandidates849.length then search edge 4 [6581] rootCandidates849 6581 best else best) = best
  rw [if_pos (by decide), rootBranch849]
  exact outerScan850

set_option maxRecDepth 4096 in
theorem outerScan848 : scan edge (search edge 4) 5 [] 0 tail848 best = best := by
  have cons : tail848 = 6577 :: tail849 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates848_checked]
  change scan edge (search edge 4) 5 [] 0 tail849
    (if 4 ≤ rootCandidates848.length then search edge 4 [6577] rootCandidates848 6577 best else best) = best
  rw [if_pos (by decide), rootBranch848]
  exact outerScan849

end ProjectEulerSolutions.PrimePairUpper.Search
