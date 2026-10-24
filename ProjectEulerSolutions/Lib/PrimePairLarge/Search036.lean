import ProjectEulerSolutions.Lib.PrimePairLarge.Search035
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot394
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot395
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot396
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot397
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot398
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot399
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot400
import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot401
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
set_option maxRecDepth 4096 in
theorem outerScan401 : scan edge (search edge 4) 5 [] 0 tail401 winner = winner := by
  rw [tail401, scan, if_neg (by decide), root401Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail402
    (if 4 ≤ root401Candidates.length then search edge 4 [2777] root401Candidates 2777 winner else winner) = winner
  rw [if_pos (by decide), root401Branch]
  exact outerScan402
set_option maxRecDepth 4096 in
theorem outerScan400 : scan edge (search edge 4) 5 [] 0 tail400 winner = winner := by
  rw [tail400, scan, if_neg (by decide), root400Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail401
    (if 4 ≤ root400Candidates.length then search edge 4 [2767] root400Candidates 2767 winner else winner) = winner
  rw [if_pos (by decide), root400Branch]
  exact outerScan401
set_option maxRecDepth 4096 in
theorem outerScan399 : scan edge (search edge 4) 5 [] 0 tail399 winner = winner := by
  rw [tail399, scan, if_neg (by decide), root399Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail400
    (if 4 ≤ root399Candidates.length then search edge 4 [2753] root399Candidates 2753 winner else winner) = winner
  rw [if_pos (by decide), root399Branch]
  exact outerScan400
set_option maxRecDepth 4096 in
theorem outerScan398 : scan edge (search edge 4) 5 [] 0 tail398 winner = winner := by
  rw [tail398, scan, if_neg (by decide), root398Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail399
    (if 4 ≤ root398Candidates.length then search edge 4 [2749] root398Candidates 2749 winner else winner) = winner
  rw [if_pos (by decide), root398Branch]
  exact outerScan399
set_option maxRecDepth 4096 in
theorem outerScan397 : scan edge (search edge 4) 5 [] 0 tail397 winner = winner := by
  rw [tail397, scan, if_neg (by decide), root397Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail398
    (if 4 ≤ root397Candidates.length then search edge 4 [2741] root397Candidates 2741 winner else winner) = winner
  rw [if_pos (by decide), root397Branch]
  exact outerScan398
set_option maxRecDepth 4096 in
theorem outerScan396 : scan edge (search edge 4) 5 [] 0 tail396 winner = winner := by
  rw [tail396, scan, if_neg (by decide), root396Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail397
    (if 4 ≤ root396Candidates.length then search edge 4 [2731] root396Candidates 2731 winner else winner) = winner
  rw [if_pos (by decide), root396Branch]
  exact outerScan397
set_option maxRecDepth 4096 in
theorem outerScan395 : scan edge (search edge 4) 5 [] 0 tail395 winner = winner := by
  rw [tail395, scan, if_neg (by decide), root395Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail396
    (if 4 ≤ root395Candidates.length then search edge 4 [2729] root395Candidates 2729 winner else winner) = winner
  rw [if_pos (by decide), root395Branch]
  exact outerScan396
set_option maxRecDepth 4096 in
theorem outerScan394 : scan edge (search edge 4) 5 [] 0 tail394 winner = winner := by
  rw [tail394, scan, if_neg (by decide), root394Candidates_checked]
  change scan edge (search edge 4) 5 [] 0 tail395
    (if 4 ≤ root394Candidates.length then search edge 4 [2719] root394Candidates 2719 winner else winner) = winner
  rw [if_pos (by decide), root394Branch]
  exact outerScan395
end ProjectEulerSolutions.PrimePairLarge.Search
