import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearch007
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallBranch008

namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

set_option maxRecDepth 4096 in
theorem outerScan983 : scan edge (search edge 4) 5 [] 0 tail983 best = best := by
  have cons : tail983 = 7789 :: tail984 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates983_checked]
  change scan edge (search edge 4) 5 [] 0 tail984
    (if 4 ≤ rootCandidates983.length then search edge 4 [7789] rootCandidates983 7789 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan984

set_option maxRecDepth 4096 in
theorem outerScan982 : scan edge (search edge 4) 5 [] 0 tail982 best = best := by
  have cons : tail982 = 7759 :: tail983 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates982_checked]
  change scan edge (search edge 4) 5 [] 0 tail983
    (if 4 ≤ rootCandidates982.length then search edge 4 [7759] rootCandidates982 7759 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan983

set_option maxRecDepth 4096 in
theorem outerScan981 : scan edge (search edge 4) 5 [] 0 tail981 best = best := by
  have cons : tail981 = 7757 :: tail982 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates981_checked]
  change scan edge (search edge 4) 5 [] 0 tail982
    (if 4 ≤ rootCandidates981.length then search edge 4 [7757] rootCandidates981 7757 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan982

set_option maxRecDepth 4096 in
theorem outerScan980 : scan edge (search edge 4) 5 [] 0 tail980 best = best := by
  have cons : tail980 = 7753 :: tail981 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates980_checked]
  change scan edge (search edge 4) 5 [] 0 tail981
    (if 4 ≤ rootCandidates980.length then search edge 4 [7753] rootCandidates980 7753 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan981

set_option maxRecDepth 4096 in
theorem outerScan979 : scan edge (search edge 4) 5 [] 0 tail979 best = best := by
  have cons : tail979 = 7741 :: tail980 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates979_checked]
  change scan edge (search edge 4) 5 [] 0 tail980
    (if 4 ≤ rootCandidates979.length then search edge 4 [7741] rootCandidates979 7741 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan980

set_option maxRecDepth 4096 in
theorem outerScan978 : scan edge (search edge 4) 5 [] 0 tail978 best = best := by
  have cons : tail978 = 7727 :: tail979 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates978_checked]
  change scan edge (search edge 4) 5 [] 0 tail979
    (if 4 ≤ rootCandidates978.length then search edge 4 [7727] rootCandidates978 7727 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan979

set_option maxRecDepth 4096 in
theorem outerScan977 : scan edge (search edge 4) 5 [] 0 tail977 best = best := by
  have cons : tail977 = 7723 :: tail978 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates977_checked]
  change scan edge (search edge 4) 5 [] 0 tail978
    (if 4 ≤ rootCandidates977.length then search edge 4 [7723] rootCandidates977 7723 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan978

set_option maxRecDepth 4096 in
theorem outerScan976 : scan edge (search edge 4) 5 [] 0 tail976 best = best := by
  have cons : tail976 = 7717 :: tail977 := rfl
  rw [cons, scan, if_neg (by decide), rootCandidates976_checked]
  change scan edge (search edge 4) 5 [] 0 tail977
    (if 4 ≤ rootCandidates976.length then search edge 4 [7717] rootCandidates976 7717 best else best) = best
  rw [if_neg (by decide)]
  exact outerScan977

end ProjectEulerSolutions.PrimePairUpper.Search
