import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch010
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch011

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan959 : scan edge (search edge 4) 5 [] 0 tail959 best = best := by
  have cons : tail959 = 7577 :: tail960 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates959_checked]
  change scan edge (search edge 4) 5 [] 0 tail960
    (if 4 ≤ rootCandidates959.length then search edge 4 [7577] rootCandidates959 7577 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan960

set_option maxRecDepth 4096 in
theorem outerScan958 : scan edge (search edge 4) 5 [] 0 tail958 best = best := by
  have cons : tail958 = 7573 :: tail959 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates958_checked]
  change scan edge (search edge 4) 5 [] 0 tail959
    (if 4 ≤ rootCandidates958.length then search edge 4 [7573] rootCandidates958 7573 best else best) = best
  rw [if_pos (by decide), rootBranch958]
  exact outerScan959

set_option maxRecDepth 4096 in
theorem outerScan957 : scan edge (search edge 4) 5 [] 0 tail957 best = best := by
  have cons : tail957 = 7561 :: tail958 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates957_checked]
  change scan edge (search edge 4) 5 [] 0 tail958
    (if 4 ≤ rootCandidates957.length then search edge 4 [7561] rootCandidates957 7561 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan958

set_option maxRecDepth 4096 in
theorem outerScan956 : scan edge (search edge 4) 5 [] 0 tail956 best = best := by
  have cons : tail956 = 7559 :: tail957 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates956_checked]
  change scan edge (search edge 4) 5 [] 0 tail957
    (if 4 ≤ rootCandidates956.length then search edge 4 [7559] rootCandidates956 7559 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan957

set_option maxRecDepth 4096 in
theorem outerScan955 : scan edge (search edge 4) 5 [] 0 tail955 best = best := by
  have cons : tail955 = 7549 :: tail956 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates955_checked]
  change scan edge (search edge 4) 5 [] 0 tail956
    (if 4 ≤ rootCandidates955.length then search edge 4 [7549] rootCandidates955 7549 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan956

set_option maxRecDepth 4096 in
theorem outerScan954 : scan edge (search edge 4) 5 [] 0 tail954 best = best := by
  have cons : tail954 = 7547 :: tail955 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates954_checked]
  change scan edge (search edge 4) 5 [] 0 tail955
    (if 4 ≤ rootCandidates954.length then search edge 4 [7547] rootCandidates954 7547 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan955

set_option maxRecDepth 4096 in
theorem outerScan953 : scan edge (search edge 4) 5 [] 0 tail953 best = best := by
  have cons : tail953 = 7541 :: tail954 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates953_checked]
  change scan edge (search edge 4) 5 [] 0 tail954
    (if 4 ≤ rootCandidates953.length then search edge 4 [7541] rootCandidates953 7541 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan954

set_option maxRecDepth 4096 in
theorem outerScan952 : scan edge (search edge 4) 5 [] 0 tail952 best = best := by
  have cons : tail952 = 7537 :: tail953 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates952_checked]
  change scan edge (search edge 4) 5 [] 0 tail953
    (if 4 ≤ rootCandidates952.length then search edge 4 [7537] rootCandidates952 7537 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan953

end ProjectEulerSolutions.PrimePairUpper.Search
