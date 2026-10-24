import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch009
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch010

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan967 : scan edge (search edge 4) 5 [] 0 tail967 best = best := by
  have cons : tail967 = 7643 :: tail968 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates967_checked]
  change scan edge (search edge 4) 5 [] 0 tail968
    (if 4 ≤ rootCandidates967.length then search edge 4 [7643] rootCandidates967 7643 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan968

set_option maxRecDepth 4096 in
theorem outerScan966 : scan edge (search edge 4) 5 [] 0 tail966 best = best := by
  have cons : tail966 = 7639 :: tail967 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates966_checked]
  change scan edge (search edge 4) 5 [] 0 tail967
    (if 4 ≤ rootCandidates966.length then search edge 4 [7639] rootCandidates966 7639 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan967

set_option maxRecDepth 4096 in
theorem outerScan965 : scan edge (search edge 4) 5 [] 0 tail965 best = best := by
  have cons : tail965 = 7621 :: tail966 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates965_checked]
  change scan edge (search edge 4) 5 [] 0 tail966
    (if 4 ≤ rootCandidates965.length then search edge 4 [7621] rootCandidates965 7621 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan966

set_option maxRecDepth 4096 in
theorem outerScan964 : scan edge (search edge 4) 5 [] 0 tail964 best = best := by
  have cons : tail964 = 7607 :: tail965 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates964_checked]
  change scan edge (search edge 4) 5 [] 0 tail965
    (if 4 ≤ rootCandidates964.length then search edge 4 [7607] rootCandidates964 7607 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan965

set_option maxRecDepth 4096 in
theorem outerScan963 : scan edge (search edge 4) 5 [] 0 tail963 best = best := by
  have cons : tail963 = 7603 :: tail964 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates963_checked]
  change scan edge (search edge 4) 5 [] 0 tail964
    (if 4 ≤ rootCandidates963.length then search edge 4 [7603] rootCandidates963 7603 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan964

set_option maxRecDepth 4096 in
theorem outerScan962 : scan edge (search edge 4) 5 [] 0 tail962 best = best := by
  have cons : tail962 = 7591 :: tail963 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates962_checked]
  change scan edge (search edge 4) 5 [] 0 tail963
    (if 4 ≤ rootCandidates962.length then search edge 4 [7591] rootCandidates962 7591 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan963

set_option maxRecDepth 4096 in
theorem outerScan961 : scan edge (search edge 4) 5 [] 0 tail961 best = best := by
  have cons : tail961 = 7589 :: tail962 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates961_checked]
  change scan edge (search edge 4) 5 [] 0 tail962
    (if 4 ≤ rootCandidates961.length then search edge 4 [7589] rootCandidates961 7589 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan962

set_option maxRecDepth 4096 in
theorem outerScan960 : scan edge (search edge 4) 5 [] 0 tail960 best = best := by
  have cons : tail960 = 7583 :: tail961 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates960_checked]
  change scan edge (search edge 4) 5 [] 0 tail961
    (if 4 ≤ rootCandidates960.length then search edge 4 [7583] rootCandidates960 7583 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan961

end ProjectEulerSolutions.PrimePairUpper.Search
