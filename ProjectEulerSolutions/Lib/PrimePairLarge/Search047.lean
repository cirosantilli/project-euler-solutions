import ProjectEulerSolutions.Lib.PrimePairLarge.Search046
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot306
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot307
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot308
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot309
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot310
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot311
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot312
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot313
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan313 : scan edge (search edge 4) 5 [] 0 tail313 winner = winner := by
  rw [tail313, scan, if_neg (by decide), root313Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail314
    (if 4 ≤ root313Candidates.length then search edge 4 [2089] root313Candidates 2089 winner else winner) = winner
  rw [if_pos (by decide), root313Branch]
  exact outerScan314
set_option maxRecDepth 4096 in
theorem outerScan312 : scan edge (search edge 4) 5 [] 0 tail312 winner = winner := by
  rw [tail312, scan, if_neg (by decide), root312Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail313
    (if 4 ≤ root312Candidates.length then search edge 4 [2087] root312Candidates 2087 winner else winner) = winner
  rw [if_pos (by decide), root312Branch]
  exact outerScan313
set_option maxRecDepth 4096 in
theorem outerScan311 : scan edge (search edge 4) 5 [] 0 tail311 winner = winner := by
  rw [tail311, scan, if_neg (by decide), root311Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail312
    (if 4 ≤ root311Candidates.length then search edge 4 [2083] root311Candidates 2083 winner else winner) = winner
  rw [if_pos (by decide), root311Branch]
  exact outerScan312
set_option maxRecDepth 4096 in
theorem outerScan310 : scan edge (search edge 4) 5 [] 0 tail310 winner = winner := by
  rw [tail310, scan, if_neg (by decide), root310Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail311
    (if 4 ≤ root310Candidates.length then search edge 4 [2081] root310Candidates 2081 winner else winner) = winner
  rw [if_pos (by decide), root310Branch]
  exact outerScan311
set_option maxRecDepth 4096 in
theorem outerScan309 : scan edge (search edge 4) 5 [] 0 tail309 winner = winner := by
  rw [tail309, scan, if_neg (by decide), root309Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail310
    (if 4 ≤ root309Candidates.length then search edge 4 [2069] root309Candidates 2069 winner else winner) = winner
  rw [if_pos (by decide), root309Branch]
  exact outerScan310
set_option maxRecDepth 4096 in
theorem outerScan308 : scan edge (search edge 4) 5 [] 0 tail308 winner = winner := by
  rw [tail308, scan, if_neg (by decide), root308Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail309
    (if 4 ≤ root308Candidates.length then search edge 4 [2063] root308Candidates 2063 winner else winner) = winner
  rw [if_pos (by decide), root308Branch]
  exact outerScan309
set_option maxRecDepth 4096 in
theorem outerScan307 : scan edge (search edge 4) 5 [] 0 tail307 winner = winner := by
  rw [tail307, scan, if_neg (by decide), root307Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail308
    (if 4 ≤ root307Candidates.length then search edge 4 [2053] root307Candidates 2053 winner else winner) = winner
  rw [if_pos (by decide), root307Branch]
  exact outerScan308
set_option maxRecDepth 4096 in
theorem outerScan306 : scan edge (search edge 4) 5 [] 0 tail306 winner = winner := by
  rw [tail306, scan, if_neg (by decide), root306Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail307
    (if 4 ≤ root306Candidates.length then search edge 4 [2039] root306Candidates 2039 winner else winner) = winner
  rw [if_pos (by decide), root306Branch]
  exact outerScan307
end ProjectEulerSolutions.PrimePairLarge.Search
