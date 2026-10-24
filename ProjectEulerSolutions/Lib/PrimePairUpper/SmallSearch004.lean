import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch003
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch004

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan1015 : scan edge (search edge 4) 5 [] 0 tail1015 best = best := by
  have cons : tail1015 = 8093 :: tail1016 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1015_checked]
  change scan edge (search edge 4) 5 [] 0 tail1016
    (if 4 ≤ rootCandidates1015.length then search edge 4 [8093] rootCandidates1015 8093 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1016

set_option maxRecDepth 4096 in
theorem outerScan1014 : scan edge (search edge 4) 5 [] 0 tail1014 best = best := by
  have cons : tail1014 = 8089 :: tail1015 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1014_checked]
  change scan edge (search edge 4) 5 [] 0 tail1015
    (if 4 ≤ rootCandidates1014.length then search edge 4 [8089] rootCandidates1014 8089 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1015

set_option maxRecDepth 4096 in
theorem outerScan1013 : scan edge (search edge 4) 5 [] 0 tail1013 best = best := by
  have cons : tail1013 = 8087 :: tail1014 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1013_checked]
  change scan edge (search edge 4) 5 [] 0 tail1014
    (if 4 ≤ rootCandidates1013.length then search edge 4 [8087] rootCandidates1013 8087 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1014

set_option maxRecDepth 4096 in
theorem outerScan1012 : scan edge (search edge 4) 5 [] 0 tail1012 best = best := by
  have cons : tail1012 = 8081 :: tail1013 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1012_checked]
  change scan edge (search edge 4) 5 [] 0 tail1013
    (if 4 ≤ rootCandidates1012.length then search edge 4 [8081] rootCandidates1012 8081 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1013

set_option maxRecDepth 4096 in
theorem outerScan1011 : scan edge (search edge 4) 5 [] 0 tail1011 best = best := by
  have cons : tail1011 = 8069 :: tail1012 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1011_checked]
  change scan edge (search edge 4) 5 [] 0 tail1012
    (if 4 ≤ rootCandidates1011.length then search edge 4 [8069] rootCandidates1011 8069 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1012

set_option maxRecDepth 4096 in
theorem outerScan1010 : scan edge (search edge 4) 5 [] 0 tail1010 best = best := by
  have cons : tail1010 = 8059 :: tail1011 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1010_checked]
  change scan edge (search edge 4) 5 [] 0 tail1011
    (if 4 ≤ rootCandidates1010.length then search edge 4 [8059] rootCandidates1010 8059 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1011

set_option maxRecDepth 4096 in
theorem outerScan1009 : scan edge (search edge 4) 5 [] 0 tail1009 best = best := by
  have cons : tail1009 = 8053 :: tail1010 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1009_checked]
  change scan edge (search edge 4) 5 [] 0 tail1010
    (if 4 ≤ rootCandidates1009.length then search edge 4 [8053] rootCandidates1009 8053 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1010

set_option maxRecDepth 4096 in
theorem outerScan1008 : scan edge (search edge 4) 5 [] 0 tail1008 best = best := by
  have cons : tail1008 = 8039 :: tail1009 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates1008_checked]
  change scan edge (search edge 4) 5 [] 0 tail1009
    (if 4 ≤ rootCandidates1008.length then search edge 4 [8039] rootCandidates1008 8039 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan1009

end ProjectEulerSolutions.PrimePairUpper.Search
