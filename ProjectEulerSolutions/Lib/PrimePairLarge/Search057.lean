import ProjectEulerSolutions.Lib.PrimePairLarge.Search056
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot226
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot227
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot228
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot229
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot230
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot231
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot232
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot233
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan233 : scan edge (search edge 4) 5 [] 0 tail233 winner = winner := by
  rw [tail233, scan, if_neg (by decide), root233Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail234
    (if 4 ≤ root233Candidates.length then search edge 4 [1487] root233Candidates 1487 winner else winner) = winner
  rw [if_pos (by decide), root233Branch]
  exact outerScan234
set_option maxRecDepth 4096 in
theorem outerScan232 : scan edge (search edge 4) 5 [] 0 tail232 winner = winner := by
  rw [tail232, scan, if_neg (by decide), root232Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail233
    (if 4 ≤ root232Candidates.length then search edge 4 [1483] root232Candidates 1483 winner else winner) = winner
  rw [if_pos (by decide), root232Branch]
  exact outerScan233
set_option maxRecDepth 4096 in
theorem outerScan231 : scan edge (search edge 4) 5 [] 0 tail231 winner = winner := by
  rw [tail231, scan, if_neg (by decide), root231Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail232
    (if 4 ≤ root231Candidates.length then search edge 4 [1481] root231Candidates 1481 winner else winner) = winner
  rw [if_pos (by decide), root231Branch]
  exact outerScan232
set_option maxRecDepth 4096 in
theorem outerScan230 : scan edge (search edge 4) 5 [] 0 tail230 winner = winner := by
  rw [tail230, scan, if_neg (by decide), root230Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail231
    (if 4 ≤ root230Candidates.length then search edge 4 [1471] root230Candidates 1471 winner else winner) = winner
  rw [if_pos (by decide), root230Branch]
  exact outerScan231
set_option maxRecDepth 4096 in
theorem outerScan229 : scan edge (search edge 4) 5 [] 0 tail229 winner = winner := by
  rw [tail229, scan, if_neg (by decide), root229Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail230
    (if 4 ≤ root229Candidates.length then search edge 4 [1459] root229Candidates 1459 winner else winner) = winner
  rw [if_pos (by decide), root229Branch]
  exact outerScan230
set_option maxRecDepth 4096 in
theorem outerScan228 : scan edge (search edge 4) 5 [] 0 tail228 winner = winner := by
  rw [tail228, scan, if_neg (by decide), root228Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail229
    (if 4 ≤ root228Candidates.length then search edge 4 [1453] root228Candidates 1453 winner else winner) = winner
  rw [if_pos (by decide), root228Branch]
  exact outerScan229
set_option maxRecDepth 4096 in
theorem outerScan227 : scan edge (search edge 4) 5 [] 0 tail227 winner = winner := by
  rw [tail227, scan, if_neg (by decide), root227Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail228
    (if 4 ≤ root227Candidates.length then search edge 4 [1451] root227Candidates 1451 winner else winner) = winner
  rw [if_pos (by decide), root227Branch]
  exact outerScan228
set_option maxRecDepth 4096 in
theorem outerScan226 : scan edge (search edge 4) 5 [] 0 tail226 winner = winner := by
  rw [tail226, scan, if_neg (by decide), root226Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail227
    (if 4 ≤ root226Candidates.length then search edge 4 [1447] root226Candidates 1447 winner else winner) = winner
  rw [if_pos (by decide), root226Branch]
  exact outerScan227
end ProjectEulerSolutions.PrimePairLarge.Search
