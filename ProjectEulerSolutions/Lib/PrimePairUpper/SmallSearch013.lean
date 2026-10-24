import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch012
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch013

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan943 : scan edge (search edge 4) 5 [] 0 tail943 best = best := by
  have cons : tail943 = 7477 :: tail944 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates943_checked]
  change scan edge (search edge 4) 5 [] 0 tail944
    (if 4 ≤ rootCandidates943.length then search edge 4 [7477] rootCandidates943 7477 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan944

set_option maxRecDepth 4096 in
theorem outerScan942 : scan edge (search edge 4) 5 [] 0 tail942 best = best := by
  have cons : tail942 = 7459 :: tail943 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates942_checked]
  change scan edge (search edge 4) 5 [] 0 tail943
    (if 4 ≤ rootCandidates942.length then search edge 4 [7459] rootCandidates942 7459 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan943

set_option maxRecDepth 4096 in
theorem outerScan941 : scan edge (search edge 4) 5 [] 0 tail941 best = best := by
  have cons : tail941 = 7457 :: tail942 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates941_checked]
  change scan edge (search edge 4) 5 [] 0 tail942
    (if 4 ≤ rootCandidates941.length then search edge 4 [7457] rootCandidates941 7457 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan942

set_option maxRecDepth 4096 in
theorem outerScan940 : scan edge (search edge 4) 5 [] 0 tail940 best = best := by
  have cons : tail940 = 7451 :: tail941 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates940_checked]
  change scan edge (search edge 4) 5 [] 0 tail941
    (if 4 ≤ rootCandidates940.length then search edge 4 [7451] rootCandidates940 7451 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan941

set_option maxRecDepth 4096 in
theorem outerScan939 : scan edge (search edge 4) 5 [] 0 tail939 best = best := by
  have cons : tail939 = 7433 :: tail940 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates939_checked]
  change scan edge (search edge 4) 5 [] 0 tail940
    (if 4 ≤ rootCandidates939.length then search edge 4 [7433] rootCandidates939 7433 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan940

set_option maxRecDepth 4096 in
theorem outerScan938 : scan edge (search edge 4) 5 [] 0 tail938 best = best := by
  have cons : tail938 = 7417 :: tail939 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates938_checked]
  change scan edge (search edge 4) 5 [] 0 tail939
    (if 4 ≤ rootCandidates938.length then search edge 4 [7417] rootCandidates938 7417 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan939

set_option maxRecDepth 4096 in
theorem outerScan937 : scan edge (search edge 4) 5 [] 0 tail937 best = best := by
  have cons : tail937 = 7411 :: tail938 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates937_checked]
  change scan edge (search edge 4) 5 [] 0 tail938
    (if 4 ≤ rootCandidates937.length then search edge 4 [7411] rootCandidates937 7411 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan938

set_option maxRecDepth 4096 in
theorem outerScan936 : scan edge (search edge 4) 5 [] 0 tail936 best = best := by
  have cons : tail936 = 7393 :: tail937 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates936_checked]
  change scan edge (search edge 4) 5 [] 0 tail937
    (if 4 ≤ rootCandidates936.length then search edge 4 [7393] rootCandidates936 7393 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan937

end ProjectEulerSolutions.PrimePairUpper.Search
