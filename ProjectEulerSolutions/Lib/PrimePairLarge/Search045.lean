import ProjectEulerSolutions.Lib.PrimePairLarge.Search044
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot322
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot323
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot324
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot325
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot326
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot327
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot328
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot329
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan329 : scan edge (search edge 4) 5 [] 0 tail329 winner = winner := by
  rw [tail329, scan, if_neg (by decide), root329Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail330
    (if 4 ≤ root329Candidates.length then search edge 4 [2237] root329Candidates 2237 winner else winner) = winner
  rw [if_pos (by decide), root329Branch]
  exact outerScan330
set_option maxRecDepth 4096 in
theorem outerScan328 : scan edge (search edge 4) 5 [] 0 tail328 winner = winner := by
  rw [tail328, scan, if_neg (by decide), root328Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail329
    (if 4 ≤ root328Candidates.length then search edge 4 [2221] root328Candidates 2221 winner else winner) = winner
  rw [if_pos (by decide), root328Branch]
  exact outerScan329
set_option maxRecDepth 4096 in
theorem outerScan327 : scan edge (search edge 4) 5 [] 0 tail327 winner = winner := by
  rw [tail327, scan, if_neg (by decide), root327Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail328
    (if 4 ≤ root327Candidates.length then search edge 4 [2213] root327Candidates 2213 winner else winner) = winner
  rw [if_pos (by decide), root327Branch]
  exact outerScan328
set_option maxRecDepth 4096 in
theorem outerScan326 : scan edge (search edge 4) 5 [] 0 tail326 winner = winner := by
  rw [tail326, scan, if_neg (by decide), root326Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail327
    (if 4 ≤ root326Candidates.length then search edge 4 [2207] root326Candidates 2207 winner else winner) = winner
  rw [if_pos (by decide), root326Branch]
  exact outerScan327
set_option maxRecDepth 4096 in
theorem outerScan325 : scan edge (search edge 4) 5 [] 0 tail325 winner = winner := by
  rw [tail325, scan, if_neg (by decide), root325Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail326
    (if 4 ≤ root325Candidates.length then search edge 4 [2203] root325Candidates 2203 winner else winner) = winner
  rw [if_pos (by decide), root325Branch]
  exact outerScan326
set_option maxRecDepth 4096 in
theorem outerScan324 : scan edge (search edge 4) 5 [] 0 tail324 winner = winner := by
  rw [tail324, scan, if_neg (by decide), root324Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail325
    (if 4 ≤ root324Candidates.length then search edge 4 [2179] root324Candidates 2179 winner else winner) = winner
  rw [if_pos (by decide), root324Branch]
  exact outerScan325
set_option maxRecDepth 4096 in
theorem outerScan323 : scan edge (search edge 4) 5 [] 0 tail323 winner = winner := by
  rw [tail323, scan, if_neg (by decide), root323Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail324
    (if 4 ≤ root323Candidates.length then search edge 4 [2161] root323Candidates 2161 winner else winner) = winner
  rw [if_pos (by decide), root323Branch]
  exact outerScan324
set_option maxRecDepth 4096 in
theorem outerScan322 : scan edge (search edge 4) 5 [] 0 tail322 winner = winner := by
  rw [tail322, scan, if_neg (by decide), root322Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail323
    (if 4 ≤ root322Candidates.length then search edge 4 [2153] root322Candidates 2153 winner else winner) = winner
  rw [if_pos (by decide), root322Branch]
  exact outerScan323
end ProjectEulerSolutions.PrimePairLarge.Search
