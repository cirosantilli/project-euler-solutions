import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch005
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch006

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan999 : scan edge (search edge 4) 5 [] 0 tail999 best = best := by
  have cons : tail999 = 7933 :: tail1000 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates999_checked]
  change scan edge (search edge 4) 5 [] 0 tail1000
    (if 4 ≤ rootCandidates999.length then search edge 4 [7933] rootCandidates999 7933 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1000

set_option maxRecDepth 4096 in
theorem outerScan998 : scan edge (search edge 4) 5 [] 0 tail998 best = best := by
  have cons : tail998 = 7927 :: tail999 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates998_checked]
  change scan edge (search edge 4) 5 [] 0 tail999
    (if 4 ≤ rootCandidates998.length then search edge 4 [7927] rootCandidates998 7927 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan999

set_option maxRecDepth 4096 in
theorem outerScan997 : scan edge (search edge 4) 5 [] 0 tail997 best = best := by
  have cons : tail997 = 7919 :: tail998 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates997_checked]
  change scan edge (search edge 4) 5 [] 0 tail998
    (if 4 ≤ rootCandidates997.length then search edge 4 [7919] rootCandidates997 7919 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan998

set_option maxRecDepth 4096 in
theorem outerScan996 : scan edge (search edge 4) 5 [] 0 tail996 best = best := by
  have cons : tail996 = 7907 :: tail997 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates996_checked]
  change scan edge (search edge 4) 5 [] 0 tail997
    (if 4 ≤ rootCandidates996.length then search edge 4 [7907] rootCandidates996 7907 best else best) = best
  rw [if_pos (by decide), rootBranch996]
  exact outerScan997

set_option maxRecDepth 4096 in
theorem outerScan995 : scan edge (search edge 4) 5 [] 0 tail995 best = best := by
  have cons : tail995 = 7901 :: tail996 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates995_checked]
  change scan edge (search edge 4) 5 [] 0 tail996
    (if 4 ≤ rootCandidates995.length then search edge 4 [7901] rootCandidates995 7901 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan996

set_option maxRecDepth 4096 in
theorem outerScan994 : scan edge (search edge 4) 5 [] 0 tail994 best = best := by
  have cons : tail994 = 7883 :: tail995 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates994_checked]
  change scan edge (search edge 4) 5 [] 0 tail995
    (if 4 ≤ rootCandidates994.length then search edge 4 [7883] rootCandidates994 7883 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan995

set_option maxRecDepth 4096 in
theorem outerScan993 : scan edge (search edge 4) 5 [] 0 tail993 best = best := by
  have cons : tail993 = 7879 :: tail994 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates993_checked]
  change scan edge (search edge 4) 5 [] 0 tail994
    (if 4 ≤ rootCandidates993.length then search edge 4 [7879] rootCandidates993 7879 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan994

set_option maxRecDepth 4096 in
theorem outerScan992 : scan edge (search edge 4) 5 [] 0 tail992 best = best := by
  have cons : tail992 = 7877 :: tail993 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates992_checked]
  change scan edge (search edge 4) 5 [] 0 tail993
    (if 4 ≤ rootCandidates992.length then search edge 4 [7877] rootCandidates992 7877 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan993

end ProjectEulerSolutions.PrimePairUpper.Search
