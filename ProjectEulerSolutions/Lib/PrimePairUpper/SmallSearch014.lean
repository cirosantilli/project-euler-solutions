import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch013
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch014

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan935 : scan edge (search edge 4) 5 [] 0 tail935 best = best := by
  have cons : tail935 = 7369 :: tail936 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates935_checked]
  change scan edge (search edge 4) 5 [] 0 tail936
    (if 4 ≤ rootCandidates935.length then search edge 4 [7369] rootCandidates935 7369 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan936

set_option maxRecDepth 4096 in
theorem outerScan934 : scan edge (search edge 4) 5 [] 0 tail934 best = best := by
  have cons : tail934 = 7351 :: tail935 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates934_checked]
  change scan edge (search edge 4) 5 [] 0 tail935
    (if 4 ≤ rootCandidates934.length then search edge 4 [7351] rootCandidates934 7351 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan935

set_option maxRecDepth 4096 in
theorem outerScan933 : scan edge (search edge 4) 5 [] 0 tail933 best = best := by
  have cons : tail933 = 7349 :: tail934 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates933_checked]
  change scan edge (search edge 4) 5 [] 0 tail934
    (if 4 ≤ rootCandidates933.length then search edge 4 [7349] rootCandidates933 7349 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan934

set_option maxRecDepth 4096 in
theorem outerScan932 : scan edge (search edge 4) 5 [] 0 tail932 best = best := by
  have cons : tail932 = 7333 :: tail933 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates932_checked]
  change scan edge (search edge 4) 5 [] 0 tail933
    (if 4 ≤ rootCandidates932.length then search edge 4 [7333] rootCandidates932 7333 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan933

set_option maxRecDepth 4096 in
theorem outerScan931 : scan edge (search edge 4) 5 [] 0 tail931 best = best := by
  have cons : tail931 = 7331 :: tail932 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates931_checked]
  change scan edge (search edge 4) 5 [] 0 tail932
    (if 4 ≤ rootCandidates931.length then search edge 4 [7331] rootCandidates931 7331 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan932

set_option maxRecDepth 4096 in
theorem outerScan930 : scan edge (search edge 4) 5 [] 0 tail930 best = best := by
  have cons : tail930 = 7321 :: tail931 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates930_checked]
  change scan edge (search edge 4) 5 [] 0 tail931
    (if 4 ≤ rootCandidates930.length then search edge 4 [7321] rootCandidates930 7321 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan931

set_option maxRecDepth 4096 in
theorem outerScan929 : scan edge (search edge 4) 5 [] 0 tail929 best = best := by
  have cons : tail929 = 7309 :: tail930 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates929_checked]
  change scan edge (search edge 4) 5 [] 0 tail930
    (if 4 ≤ rootCandidates929.length then search edge 4 [7309] rootCandidates929 7309 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan930

set_option maxRecDepth 4096 in
theorem outerScan928 : scan edge (search edge 4) 5 [] 0 tail928 best = best := by
  have cons : tail928 = 7307 :: tail929 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates928_checked]
  change scan edge (search edge 4) 5 [] 0 tail929
    (if 4 ≤ rootCandidates928.length then search edge 4 [7307] rootCandidates928 7307 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan929

end ProjectEulerSolutions.PrimePairUpper.Search
