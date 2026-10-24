import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch018
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch019

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan895 : scan edge (search edge 4) 5 [] 0 tail895 best = best := by
  have cons : tail895 = 6983 :: tail896 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates895_checked]
  change scan edge (search edge 4) 5 [] 0 tail896
    (if 4 ≤ rootCandidates895.length then search edge 4 [6983] rootCandidates895 6983 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan896

set_option maxRecDepth 4096 in
theorem outerScan894 : scan edge (search edge 4) 5 [] 0 tail894 best = best := by
  have cons : tail894 = 6977 :: tail895 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates894_checked]
  change scan edge (search edge 4) 5 [] 0 tail895
    (if 4 ≤ rootCandidates894.length then search edge 4 [6977] rootCandidates894 6977 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan895

set_option maxRecDepth 4096 in
theorem outerScan893 : scan edge (search edge 4) 5 [] 0 tail893 best = best := by
  have cons : tail893 = 6971 :: tail894 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates893_checked]
  change scan edge (search edge 4) 5 [] 0 tail894
    (if 4 ≤ rootCandidates893.length then search edge 4 [6971] rootCandidates893 6971 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan894

set_option maxRecDepth 4096 in
theorem outerScan892 : scan edge (search edge 4) 5 [] 0 tail892 best = best := by
  have cons : tail892 = 6967 :: tail893 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates892_checked]
  change scan edge (search edge 4) 5 [] 0 tail893
    (if 4 ≤ rootCandidates892.length then search edge 4 [6967] rootCandidates892 6967 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan893

set_option maxRecDepth 4096 in
theorem outerScan891 : scan edge (search edge 4) 5 [] 0 tail891 best = best := by
  have cons : tail891 = 6961 :: tail892 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates891_checked]
  change scan edge (search edge 4) 5 [] 0 tail892
    (if 4 ≤ rootCandidates891.length then search edge 4 [6961] rootCandidates891 6961 best else best) = best
  rw [if_pos (by decide), rootBranch891]
  exact outerScan892

set_option maxRecDepth 4096 in
theorem outerScan890 : scan edge (search edge 4) 5 [] 0 tail890 best = best := by
  have cons : tail890 = 6959 :: tail891 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates890_checked]
  change scan edge (search edge 4) 5 [] 0 tail891
    (if 4 ≤ rootCandidates890.length then search edge 4 [6959] rootCandidates890 6959 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan891

set_option maxRecDepth 4096 in
theorem outerScan889 : scan edge (search edge 4) 5 [] 0 tail889 best = best := by
  have cons : tail889 = 6949 :: tail890 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates889_checked]
  change scan edge (search edge 4) 5 [] 0 tail890
    (if 4 ≤ rootCandidates889.length then search edge 4 [6949] rootCandidates889 6949 best else best) = best
  rw [if_pos (by decide), rootBranch889]
  exact outerScan890

set_option maxRecDepth 4096 in
theorem outerScan888 : scan edge (search edge 4) 5 [] 0 tail888 best = best := by
  have cons : tail888 = 6947 :: tail889 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates888_checked]
  change scan edge (search edge 4) 5 [] 0 tail889
    (if 4 ≤ rootCandidates888.length then search edge 4 [6947] rootCandidates888 6947 best else best) = best
  rw [if_pos (by decide), rootBranch888]
  exact outerScan889

end ProjectEulerSolutions.PrimePairUpper.Search
