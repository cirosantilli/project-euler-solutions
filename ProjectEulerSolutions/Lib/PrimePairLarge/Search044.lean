import ProjectEulerSolutions.Lib.PrimePairLarge.Search043
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot330
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot331
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot332
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot333
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot334
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot335
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot336
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot337
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan337 : scan edge (search edge 4) 5 [] 0 tail337 winner = winner := by
  rw [tail337, scan, if_neg (by decide), root337Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail338
    (if 4 ≤ root337Candidates.length then search edge 4 [2287] root337Candidates 2287 winner else winner) = winner
  rw [if_pos (by decide), root337Branch]
  exact outerScan338
set_option maxRecDepth 4096 in
theorem outerScan336 : scan edge (search edge 4) 5 [] 0 tail336 winner = winner := by
  rw [tail336, scan, if_neg (by decide), root336Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail337
    (if 4 ≤ root336Candidates.length then search edge 4 [2281] root336Candidates 2281 winner else winner) = winner
  rw [if_pos (by decide), root336Branch]
  exact outerScan337
set_option maxRecDepth 4096 in
theorem outerScan335 : scan edge (search edge 4) 5 [] 0 tail335 winner = winner := by
  rw [tail335, scan, if_neg (by decide), root335Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail336
    (if 4 ≤ root335Candidates.length then search edge 4 [2273] root335Candidates 2273 winner else winner) = winner
  rw [if_pos (by decide), root335Branch]
  exact outerScan336
set_option maxRecDepth 4096 in
theorem outerScan334 : scan edge (search edge 4) 5 [] 0 tail334 winner = winner := by
  rw [tail334, scan, if_neg (by decide), root334Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail335
    (if 4 ≤ root334Candidates.length then search edge 4 [2269] root334Candidates 2269 winner else winner) = winner
  rw [if_pos (by decide), root334Branch]
  exact outerScan335
set_option maxRecDepth 4096 in
theorem outerScan333 : scan edge (search edge 4) 5 [] 0 tail333 winner = winner := by
  rw [tail333, scan, if_neg (by decide), root333Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail334
    (if 4 ≤ root333Candidates.length then search edge 4 [2267] root333Candidates 2267 winner else winner) = winner
  rw [if_pos (by decide), root333Branch]
  exact outerScan334
set_option maxRecDepth 4096 in
theorem outerScan332 : scan edge (search edge 4) 5 [] 0 tail332 winner = winner := by
  rw [tail332, scan, if_neg (by decide), root332Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail333
    (if 4 ≤ root332Candidates.length then search edge 4 [2251] root332Candidates 2251 winner else winner) = winner
  rw [if_pos (by decide), root332Branch]
  exact outerScan333
set_option maxRecDepth 4096 in
theorem outerScan331 : scan edge (search edge 4) 5 [] 0 tail331 winner = winner := by
  rw [tail331, scan, if_neg (by decide), root331Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail332
    (if 4 ≤ root331Candidates.length then search edge 4 [2243] root331Candidates 2243 winner else winner) = winner
  rw [if_pos (by decide), root331Branch]
  exact outerScan332
set_option maxRecDepth 4096 in
theorem outerScan330 : scan edge (search edge 4) 5 [] 0 tail330 winner = winner := by
  rw [tail330, scan, if_neg (by decide), root330Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail331
    (if 4 ≤ root330Candidates.length then search edge 4 [2239] root330Candidates 2239 winner else winner) = winner
  rw [if_pos (by decide), root330Branch]
  exact outerScan331
end ProjectEulerSolutions.PrimePairLarge.Search
