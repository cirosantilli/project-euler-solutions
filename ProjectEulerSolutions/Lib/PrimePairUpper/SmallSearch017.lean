import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch016
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch017

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan911 : scan edge (search edge 4) 5 [] 0 tail911 best = best := by
  have cons : tail911 = 7129 :: tail912 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates911_checked]
  change scan edge (search edge 4) 5 [] 0 tail912
    (if 4 ≤ rootCandidates911.length then search edge 4 [7129] rootCandidates911 7129 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan912

set_option maxRecDepth 4096 in
theorem outerScan910 : scan edge (search edge 4) 5 [] 0 tail910 best = best := by
  have cons : tail910 = 7127 :: tail911 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates910_checked]
  change scan edge (search edge 4) 5 [] 0 tail911
    (if 4 ≤ rootCandidates910.length then search edge 4 [7127] rootCandidates910 7127 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan911

set_option maxRecDepth 4096 in
theorem outerScan909 : scan edge (search edge 4) 5 [] 0 tail909 best = best := by
  have cons : tail909 = 7121 :: tail910 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates909_checked]
  change scan edge (search edge 4) 5 [] 0 tail910
    (if 4 ≤ rootCandidates909.length then search edge 4 [7121] rootCandidates909 7121 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan910

set_option maxRecDepth 4096 in
theorem outerScan908 : scan edge (search edge 4) 5 [] 0 tail908 best = best := by
  have cons : tail908 = 7109 :: tail909 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates908_checked]
  change scan edge (search edge 4) 5 [] 0 tail909
    (if 4 ≤ rootCandidates908.length then search edge 4 [7109] rootCandidates908 7109 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan909

set_option maxRecDepth 4096 in
theorem outerScan907 : scan edge (search edge 4) 5 [] 0 tail907 best = best := by
  have cons : tail907 = 7103 :: tail908 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates907_checked]
  change scan edge (search edge 4) 5 [] 0 tail908
    (if 4 ≤ rootCandidates907.length then search edge 4 [7103] rootCandidates907 7103 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan908

set_option maxRecDepth 4096 in
theorem outerScan906 : scan edge (search edge 4) 5 [] 0 tail906 best = best := by
  have cons : tail906 = 7079 :: tail907 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates906_checked]
  change scan edge (search edge 4) 5 [] 0 tail907
    (if 4 ≤ rootCandidates906.length then search edge 4 [7079] rootCandidates906 7079 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan907

set_option maxRecDepth 4096 in
theorem outerScan905 : scan edge (search edge 4) 5 [] 0 tail905 best = best := by
  have cons : tail905 = 7069 :: tail906 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates905_checked]
  change scan edge (search edge 4) 5 [] 0 tail906
    (if 4 ≤ rootCandidates905.length then search edge 4 [7069] rootCandidates905 7069 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan906

set_option maxRecDepth 4096 in
theorem outerScan904 : scan edge (search edge 4) 5 [] 0 tail904 best = best := by
  have cons : tail904 = 7057 :: tail905 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates904_checked]
  change scan edge (search edge 4) 5 [] 0 tail905
    (if 4 ≤ rootCandidates904.length then search edge 4 [7057] rootCandidates904 7057 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan905

end ProjectEulerSolutions.PrimePairUpper.Search
