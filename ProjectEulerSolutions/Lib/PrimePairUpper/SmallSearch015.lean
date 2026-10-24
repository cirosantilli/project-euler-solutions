import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch014
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch015

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan927 : scan edge (search edge 4) 5 [] 0 tail927 best = best := by
  have cons : tail927 = 7297 :: tail928 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates927_checked]
  change scan edge (search edge 4) 5 [] 0 tail928
    (if 4 ≤ rootCandidates927.length then search edge 4 [7297] rootCandidates927 7297 best else best) = best
  rw [if_pos (by decide), rootBranch927]
  exact outerScan928

set_option maxRecDepth 4096 in
theorem outerScan926 : scan edge (search edge 4) 5 [] 0 tail926 best = best := by
  have cons : tail926 = 7283 :: tail927 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates926_checked]
  change scan edge (search edge 4) 5 [] 0 tail927
    (if 4 ≤ rootCandidates926.length then search edge 4 [7283] rootCandidates926 7283 best else best) = best
  rw [if_pos (by decide), rootBranch926]
  exact outerScan927

set_option maxRecDepth 4096 in
theorem outerScan925 : scan edge (search edge 4) 5 [] 0 tail925 best = best := by
  have cons : tail925 = 7253 :: tail926 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates925_checked]
  change scan edge (search edge 4) 5 [] 0 tail926
    (if 4 ≤ rootCandidates925.length then search edge 4 [7253] rootCandidates925 7253 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan926

set_option maxRecDepth 4096 in
theorem outerScan924 : scan edge (search edge 4) 5 [] 0 tail924 best = best := by
  have cons : tail924 = 7247 :: tail925 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates924_checked]
  change scan edge (search edge 4) 5 [] 0 tail925
    (if 4 ≤ rootCandidates924.length then search edge 4 [7247] rootCandidates924 7247 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan925

set_option maxRecDepth 4096 in
theorem outerScan923 : scan edge (search edge 4) 5 [] 0 tail923 best = best := by
  have cons : tail923 = 7243 :: tail924 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates923_checked]
  change scan edge (search edge 4) 5 [] 0 tail924
    (if 4 ≤ rootCandidates923.length then search edge 4 [7243] rootCandidates923 7243 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan924

set_option maxRecDepth 4096 in
theorem outerScan922 : scan edge (search edge 4) 5 [] 0 tail922 best = best := by
  have cons : tail922 = 7237 :: tail923 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates922_checked]
  change scan edge (search edge 4) 5 [] 0 tail923
    (if 4 ≤ rootCandidates922.length then search edge 4 [7237] rootCandidates922 7237 best else best) = best
  rw [if_pos (by decide), rootBranch922]
  exact outerScan923

set_option maxRecDepth 4096 in
theorem outerScan921 : scan edge (search edge 4) 5 [] 0 tail921 best = best := by
  have cons : tail921 = 7229 :: tail922 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates921_checked]
  change scan edge (search edge 4) 5 [] 0 tail922
    (if 4 ≤ rootCandidates921.length then search edge 4 [7229] rootCandidates921 7229 best else best) = best
  rw [if_pos (by decide), rootBranch921]
  exact outerScan922

set_option maxRecDepth 4096 in
theorem outerScan920 : scan edge (search edge 4) 5 [] 0 tail920 best = best := by
  have cons : tail920 = 7219 :: tail921 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates920_checked]
  change scan edge (search edge 4) 5 [] 0 tail921
    (if 4 ≤ rootCandidates920.length then search edge 4 [7219] rootCandidates920 7219 best else best) = best
  rw [if_pos (by decide), rootBranch920]
  exact outerScan921

end ProjectEulerSolutions.PrimePairUpper.Search
