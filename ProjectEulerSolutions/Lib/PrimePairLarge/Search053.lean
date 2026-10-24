import ProjectEulerSolutions.Lib.PrimePairLarge.Search052
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot258
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot259
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot260
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot261
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot262
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot263
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot264
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot265
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan265 : scan edge (search edge 4) 5 [] 0 tail265 winner = winner := by
  rw [tail265, scan, if_neg (by decide), root265Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail266
    (if 4 ≤ root265Candidates.length then search edge 4 [1721] root265Candidates 1721 winner else winner) = winner
  rw [if_pos (by decide), root265Branch]
  exact outerScan266
set_option maxRecDepth 4096 in
theorem outerScan264 : scan edge (search edge 4) 5 [] 0 tail264 winner = winner := by
  rw [tail264, scan, if_neg (by decide), root264Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail265
    (if 4 ≤ root264Candidates.length then search edge 4 [1709] root264Candidates 1709 winner else winner) = winner
  rw [if_pos (by decide), root264Branch]
  exact outerScan265
set_option maxRecDepth 4096 in
theorem outerScan263 : scan edge (search edge 4) 5 [] 0 tail263 winner = winner := by
  rw [tail263, scan, if_neg (by decide), root263Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail264
    (if 4 ≤ root263Candidates.length then search edge 4 [1699] root263Candidates 1699 winner else winner) = winner
  rw [if_pos (by decide), root263Branch]
  exact outerScan264
set_option maxRecDepth 4096 in
theorem outerScan262 : scan edge (search edge 4) 5 [] 0 tail262 winner = winner := by
  rw [tail262, scan, if_neg (by decide), root262Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail263
    (if 4 ≤ root262Candidates.length then search edge 4 [1697] root262Candidates 1697 winner else winner) = winner
  rw [if_pos (by decide), root262Branch]
  exact outerScan263
set_option maxRecDepth 4096 in
theorem outerScan261 : scan edge (search edge 4) 5 [] 0 tail261 winner = winner := by
  rw [tail261, scan, if_neg (by decide), root261Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail262
    (if 4 ≤ root261Candidates.length then search edge 4 [1693] root261Candidates 1693 winner else winner) = winner
  rw [if_pos (by decide), root261Branch]
  exact outerScan262
set_option maxRecDepth 4096 in
theorem outerScan260 : scan edge (search edge 4) 5 [] 0 tail260 winner = winner := by
  rw [tail260, scan, if_neg (by decide), root260Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail261
    (if 4 ≤ root260Candidates.length then search edge 4 [1669] root260Candidates 1669 winner else winner) = winner
  rw [if_pos (by decide), root260Branch]
  exact outerScan261
set_option maxRecDepth 4096 in
theorem outerScan259 : scan edge (search edge 4) 5 [] 0 tail259 winner = winner := by
  rw [tail259, scan, if_neg (by decide), root259Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail260
    (if 4 ≤ root259Candidates.length then search edge 4 [1667] root259Candidates 1667 winner else winner) = winner
  rw [if_pos (by decide), root259Branch]
  exact outerScan260
set_option maxRecDepth 4096 in
theorem outerScan258 : scan edge (search edge 4) 5 [] 0 tail258 winner = winner := by
  rw [tail258, scan, if_neg (by decide), root258Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail259
    (if 4 ≤ root258Candidates.length then search edge 4 [1663] root258Candidates 1663 winner else winner) = winner
  rw [if_pos (by decide), root258Branch]
  exact outerScan259
end ProjectEulerSolutions.PrimePairLarge.Search
