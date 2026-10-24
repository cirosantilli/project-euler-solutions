import ProjectEulerSolutions.Lib.PrimePairLarge.Search083
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot010
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot011
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot012
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot013
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot014
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot015
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot016
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot017
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan17 : scan edge (search edge 4) 5 [] 0 tail17 winner = winner := by
  rw [tail17, scan, if_neg (by decide), root17Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail18
    (if 4 ≤ root17Candidates.length then search edge 4 [71] root17Candidates 71 winner else winner) = winner
  rw [if_pos (by decide), root17Branch]
  exact outerScan18
set_option maxRecDepth 4096 in
theorem outerScan16 : scan edge (search edge 4) 5 [] 0 tail16 winner = winner := by
  rw [tail16, scan, if_neg (by decide), root16Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail17
    (if 4 ≤ root16Candidates.length then search edge 4 [67] root16Candidates 67 winner else winner) = winner
  rw [if_pos (by decide), root16Branch]
  exact outerScan17
set_option maxRecDepth 4096 in
theorem outerScan15 : scan edge (search edge 4) 5 [] 0 tail15 winner = winner := by
  rw [tail15, scan, if_neg (by decide), root15Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail16
    (if 4 ≤ root15Candidates.length then search edge 4 [61] root15Candidates 61 winner else winner) = winner
  rw [if_pos (by decide), root15Branch]
  exact outerScan16
set_option maxRecDepth 4096 in
theorem outerScan14 : scan edge (search edge 4) 5 [] 0 tail14 winner = winner := by
  rw [tail14, scan, if_neg (by decide), root14Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail15
    (if 4 ≤ root14Candidates.length then search edge 4 [59] root14Candidates 59 winner else winner) = winner
  rw [if_pos (by decide), root14Branch]
  exact outerScan15
set_option maxRecDepth 4096 in
theorem outerScan13 : scan edge (search edge 4) 5 [] 0 tail13 winner = winner := by
  rw [tail13, scan, if_neg (by decide), root13Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail14
    (if 4 ≤ root13Candidates.length then search edge 4 [53] root13Candidates 53 winner else winner) = winner
  rw [if_pos (by decide), root13Branch]
  exact outerScan14
set_option maxRecDepth 4096 in
theorem outerScan12 : scan edge (search edge 4) 5 [] 0 tail12 winner = winner := by
  rw [tail12, scan, if_neg (by decide), root12Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail13
    (if 4 ≤ root12Candidates.length then search edge 4 [47] root12Candidates 47 winner else winner) = winner
  rw [if_pos (by decide), root12Branch]
  exact outerScan13
set_option maxRecDepth 4096 in
theorem outerScan11 : scan edge (search edge 4) 5 [] 0 tail11 winner = winner := by
  rw [tail11, scan, if_neg (by decide), root11Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail12
    (if 4 ≤ root11Candidates.length then search edge 4 [43] root11Candidates 43 winner else winner) = winner
  rw [if_pos (by decide), root11Branch]
  exact outerScan12
set_option maxRecDepth 4096 in
theorem outerScan10 : scan edge (search edge 4) 5 [] 0 tail10 winner = winner := by
  rw [tail10, scan, if_neg (by decide), root10Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail11
    (if 4 ≤ root10Candidates.length then search edge 4 [41] root10Candidates 41 winner else winner) = winner
  rw [if_pos (by decide), root10Branch]
  exact outerScan11
end ProjectEulerSolutions.PrimePairLarge.Search
