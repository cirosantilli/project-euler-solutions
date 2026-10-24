import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot021Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root21Candidates32 : List Nat := [19913]
set_option maxRecDepth 8192 in
theorem root21Candidates32_checked : candidates edge 4 89 4973 root21Tail33 winner.bound = root21Candidates32 := by
  unfold candidates
  have row : edge 4973 = (SortedNatTree.contains FastRows.row4973) := by
    change Data.row4973.contains = (SortedNatTree.contains FastRows.row4973)
    exact FastRows.row4973_checked.symm
  rw [row]
  decide
theorem root21Step32 : scan edge (search edge 3) 4 [89] 89 root21Tail32 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail33 winner := by
  rw [root21Tail32, scan, if_neg (by decide), root21Candidates32_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail33
    (if 3 ≤ root21Candidates32.length then search edge 3 [89, 4973] root21Candidates32 5062 winner else winner) = _
  rw [if_neg (by decide)]
def root21Candidates33 : List Nat := [5087, 8147]
set_option maxRecDepth 8192 in
theorem root21Candidates33_checked : candidates edge 4 89 5003 root21Tail34 winner.bound = root21Candidates33 := by
  unfold candidates
  have row : edge 5003 = (SortedNatTree.contains FastRows.row5003) := by
    change Data.row5003.contains = (SortedNatTree.contains FastRows.row5003)
    exact FastRows.row5003_checked.symm
  rw [row]
  decide
theorem root21Step33 : scan edge (search edge 3) 4 [89] 89 root21Tail33 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail34 winner := by
  rw [root21Tail33, scan, if_neg (by decide), root21Candidates33_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail34
    (if 3 ≤ root21Candidates33.length then search edge 3 [89, 5003] root21Candidates33 5092 winner else winner) = _
  rw [if_neg (by decide)]
def root21Candidates34 : List Nat := [6299, 7727, 18521]
set_option maxRecDepth 8192 in
theorem root21Candidates34_checked : candidates edge 4 89 5039 root21Tail35 winner.bound = root21Candidates34 := by
  unfold candidates
  have row : edge 5039 = (SortedNatTree.contains FastRows.row5039) := by
    change Data.row5039.contains = (SortedNatTree.contains FastRows.row5039)
    exact FastRows.row5039_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root21Child34 : search edge 3 [89, 5039] root21Candidates34 5128 winner = winner := by decide
theorem root21Step34 : scan edge (search edge 3) 4 [89] 89 root21Tail34 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail35 winner := by
  rw [root21Tail34, scan, if_neg (by decide), root21Candidates34_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail35
    (if 3 ≤ root21Candidates34.length then search edge 3 [89, 5039] root21Candidates34 5128 winner else winner) = _
  rw [if_pos (by decide), root21Child34]
def root21Candidates35 : List Nat := [5861, 8147, 10463]
set_option maxRecDepth 8192 in
theorem root21Candidates35_checked : candidates edge 4 89 5087 root21Tail36 winner.bound = root21Candidates35 := by
  unfold candidates
  have row : edge 5087 = (SortedNatTree.contains FastRows.row5087) := by
    change Data.row5087.contains = (SortedNatTree.contains FastRows.row5087)
    exact FastRows.row5087_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root21Child35 : search edge 3 [89, 5087] root21Candidates35 5176 winner = winner := by decide
theorem root21Step35 : scan edge (search edge 3) 4 [89] 89 root21Tail35 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail36 winner := by
  rw [root21Tail35, scan, if_neg (by decide), root21Candidates35_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail36
    (if 3 ≤ root21Candidates35.length then search edge 3 [89, 5087] root21Candidates35 5176 winner else winner) = _
  rw [if_pos (by decide), root21Child35]
def root21Candidates36 : List Nat := [5393, 18521]
set_option maxRecDepth 8192 in
theorem root21Candidates36_checked : candidates edge 4 89 5171 root21Tail37 winner.bound = root21Candidates36 := by
  unfold candidates
  have row : edge 5171 = (SortedNatTree.contains FastRows.row5171) := by
    change Data.row5171.contains = (SortedNatTree.contains FastRows.row5171)
    exact FastRows.row5171_checked.symm
  rw [row]
  decide
theorem root21Step36 : scan edge (search edge 3) 4 [89] 89 root21Tail36 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail37 winner := by
  rw [root21Tail36, scan, if_neg (by decide), root21Candidates36_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail37
    (if 3 ≤ root21Candidates36.length then search edge 3 [89, 5171] root21Candidates36 5260 winner else winner) = _
  rw [if_neg (by decide)]
def root21Candidates37 : List Nat := [15461]
set_option maxRecDepth 8192 in
theorem root21Candidates37_checked : candidates edge 4 89 5189 root21Tail38 winner.bound = root21Candidates37 := by
  unfold candidates
  have row : edge 5189 = (SortedNatTree.contains FastRows.row5189) := by
    change Data.row5189.contains = (SortedNatTree.contains FastRows.row5189)
    exact FastRows.row5189_checked.symm
  rw [row]
  decide
theorem root21Step37 : scan edge (search edge 3) 4 [89] 89 root21Tail37 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail38 winner := by
  rw [root21Tail37, scan, if_neg (by decide), root21Candidates37_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail38
    (if 3 ≤ root21Candidates37.length then search edge 3 [89, 5189] root21Candidates37 5278 winner else winner) = _
  rw [if_neg (by decide)]
def root21Candidates38 : List Nat := [8147, 10463, 15551]
set_option maxRecDepth 8192 in
theorem root21Candidates38_checked : candidates edge 4 89 5309 root21Tail39 winner.bound = root21Candidates38 := by
  unfold candidates
  have row : edge 5309 = (SortedNatTree.contains FastRows.row5309) := by
    change Data.row5309.contains = (SortedNatTree.contains FastRows.row5309)
    exact FastRows.row5309_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root21Child38 : search edge 3 [89, 5309] root21Candidates38 5398 winner = winner := by decide
theorem root21Step38 : scan edge (search edge 3) 4 [89] 89 root21Tail38 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail39 winner := by
  rw [root21Tail38, scan, if_neg (by decide), root21Candidates38_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail39
    (if 3 ≤ root21Candidates38.length then search edge 3 [89, 5309] root21Candidates38 5398 winner else winner) = _
  rw [if_pos (by decide), root21Child38]
def root21Candidates39 : List Nat := [7649]
set_option maxRecDepth 8192 in
theorem root21Candidates39_checked : candidates edge 4 89 5333 root21Tail40 winner.bound = root21Candidates39 := by
  unfold candidates
  have row : edge 5333 = (SortedNatTree.contains FastRows.row5333) := by
    change Data.row5333.contains = (SortedNatTree.contains FastRows.row5333)
    exact FastRows.row5333_checked.symm
  rw [row]
  decide
theorem root21Step39 : scan edge (search edge 3) 4 [89] 89 root21Tail39 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail40 winner := by
  rw [root21Tail39, scan, if_neg (by decide), root21Candidates39_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail40
    (if 3 ≤ root21Candidates39.length then search edge 3 [89, 5333] root21Candidates39 5422 winner else winner) = _
  rw [if_neg (by decide)]
def root21Candidates40 : List Nat := [6353, 10463, 15083]
set_option maxRecDepth 8192 in
theorem root21Candidates40_checked : candidates edge 4 89 5387 root21Tail41 winner.bound = root21Candidates40 := by
  unfold candidates
  have row : edge 5387 = (SortedNatTree.contains FastRows.row5387) := by
    change Data.row5387.contains = (SortedNatTree.contains FastRows.row5387)
    exact FastRows.row5387_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root21Child40 : search edge 3 [89, 5387] root21Candidates40 5476 winner = winner := by decide
theorem root21Step40 : scan edge (search edge 3) 4 [89] 89 root21Tail40 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail41 winner := by
  rw [root21Tail40, scan, if_neg (by decide), root21Candidates40_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail41
    (if 3 ≤ root21Candidates40.length then search edge 3 [89, 5387] root21Candidates40 5476 winner else winner) = _
  rw [if_pos (by decide), root21Child40]
def root21Candidates41 : List Nat := [6521]
set_option maxRecDepth 8192 in
theorem root21Candidates41_checked : candidates edge 4 89 5393 root21Tail42 winner.bound = root21Candidates41 := by
  unfold candidates
  have row : edge 5393 = (SortedNatTree.contains FastRows.row5393) := by
    change Data.row5393.contains = (SortedNatTree.contains FastRows.row5393)
    exact FastRows.row5393_checked.symm
  rw [row]
  decide
theorem root21Step41 : scan edge (search edge 3) 4 [89] 89 root21Tail41 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail42 winner := by
  rw [root21Tail41, scan, if_neg (by decide), root21Candidates41_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail42
    (if 3 ≤ root21Candidates41.length then search edge 3 [89, 5393] root21Candidates41 5482 winner else winner) = _
  rw [if_neg (by decide)]
def root21Candidates42 : List Nat := [16487]
set_option maxRecDepth 8192 in
theorem root21Candidates42_checked : candidates edge 4 89 5507 root21Tail43 winner.bound = root21Candidates42 := by
  unfold candidates
  have row : edge 5507 = (SortedNatTree.contains FastRows.row5507) := by
    change Data.row5507.contains = (SortedNatTree.contains FastRows.row5507)
    exact FastRows.row5507_checked.symm
  rw [row]
  decide
theorem root21Step42 : scan edge (search edge 3) 4 [89] 89 root21Tail42 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail43 winner := by
  rw [root21Tail42, scan, if_neg (by decide), root21Candidates42_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail43
    (if 3 ≤ root21Candidates42.length then search edge 3 [89, 5507] root21Candidates42 5596 winner else winner) = _
  rw [if_neg (by decide)]
def root21Candidates43 : List Nat := [8663]
set_option maxRecDepth 8192 in
theorem root21Candidates43_checked : candidates edge 4 89 5651 root21Tail44 winner.bound = root21Candidates43 := by
  unfold candidates
  have row : edge 5651 = (SortedNatTree.contains FastRows.row5651) := by
    change Data.row5651.contains = (SortedNatTree.contains FastRows.row5651)
    exact FastRows.row5651_checked.symm
  rw [row]
  decide
theorem root21Step43 : scan edge (search edge 3) 4 [89] 89 root21Tail43 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail44 winner := by
  rw [root21Tail43, scan, if_neg (by decide), root21Candidates43_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail44
    (if 3 ≤ root21Candidates43.length then search edge 3 [89, 5651] root21Candidates43 5740 winner else winner) = _
  rw [if_neg (by decide)]
def root21Candidates44 : List Nat := [5987]
set_option maxRecDepth 8192 in
theorem root21Candidates44_checked : candidates edge 4 89 5861 root21Tail45 winner.bound = root21Candidates44 := by
  unfold candidates
  have row : edge 5861 = (SortedNatTree.contains FastRows.row5861) := by
    change Data.row5861.contains = (SortedNatTree.contains FastRows.row5861)
    exact FastRows.row5861_checked.symm
  rw [row]
  decide
theorem root21Step44 : scan edge (search edge 3) 4 [89] 89 root21Tail44 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail45 winner := by
  rw [root21Tail44, scan, if_neg (by decide), root21Candidates44_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail45
    (if 3 ≤ root21Candidates44.length then search edge 3 [89, 5861] root21Candidates44 5950 winner else winner) = _
  rw [if_neg (by decide)]
def root21Candidates45 : List Nat := [6263, 10667, 15299, 16673, 17021]
set_option maxRecDepth 8192 in
theorem root21Candidates45_checked : candidates edge 4 89 5879 root21Tail46 winner.bound = root21Candidates45 := by
  unfold candidates
  have row : edge 5879 = (SortedNatTree.contains FastRows.row5879) := by
    change Data.row5879.contains = (SortedNatTree.contains FastRows.row5879)
    exact FastRows.row5879_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root21Child45 : search edge 3 [89, 5879] root21Candidates45 5968 winner = winner := by decide
theorem root21Step45 : scan edge (search edge 3) 4 [89] 89 root21Tail45 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail46 winner := by
  rw [root21Tail45, scan, if_neg (by decide), root21Candidates45_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail46
    (if 3 ≤ root21Candidates45.length then search edge 3 [89, 5879] root21Candidates45 5968 winner else winner) = _
  rw [if_pos (by decide), root21Child45]
def root21Candidates46 : List Nat := [11489]
set_option maxRecDepth 8192 in
theorem root21Candidates46_checked : candidates edge 4 89 5903 root21Tail47 winner.bound = root21Candidates46 := by
  unfold candidates
  have row : edge 5903 = (SortedNatTree.contains FastRows.row5903) := by
    change Data.row5903.contains = (SortedNatTree.contains FastRows.row5903)
    exact FastRows.row5903_checked.symm
  rw [row]
  decide
theorem root21Step46 : scan edge (search edge 3) 4 [89] 89 root21Tail46 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail47 winner := by
  rw [root21Tail46, scan, if_neg (by decide), root21Candidates46_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail47
    (if 3 ≤ root21Candidates46.length then search edge 3 [89, 5903] root21Candidates46 5992 winner else winner) = _
  rw [if_neg (by decide)]
def root21Candidates47 : List Nat := [14621]
set_option maxRecDepth 8192 in
theorem root21Candidates47_checked : candidates edge 4 89 5987 root21Tail48 winner.bound = root21Candidates47 := by
  unfold candidates
  have row : edge 5987 = (SortedNatTree.contains FastRows.row5987) := by
    change Data.row5987.contains = (SortedNatTree.contains FastRows.row5987)
    exact FastRows.row5987_checked.symm
  rw [row]
  decide
theorem root21Step47 : scan edge (search edge 3) 4 [89] 89 root21Tail47 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail48 winner := by
  rw [root21Tail47, scan, if_neg (by decide), root21Candidates47_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail48
    (if 3 ≤ root21Candidates47.length then search edge 3 [89, 5987] root21Candidates47 6076 winner else winner) = _
  rw [if_neg (by decide)]
def root21Candidates48 : List Nat := [6353, 8663, 11471, 14621]
set_option maxRecDepth 8192 in
theorem root21Candidates48_checked : candidates edge 4 89 6113 root21Tail49 winner.bound = root21Candidates48 := by
  unfold candidates
  have row : edge 6113 = (SortedNatTree.contains FastRows.row6113) := by
    change Data.row6113.contains = (SortedNatTree.contains FastRows.row6113)
    exact FastRows.row6113_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root21Child48 : search edge 3 [89, 6113] root21Candidates48 6202 winner = winner := by decide
theorem root21Step48 : scan edge (search edge 3) 4 [89] 89 root21Tail48 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail49 winner := by
  rw [root21Tail48, scan, if_neg (by decide), root21Candidates48_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail49
    (if 3 ≤ root21Candidates48.length then search edge 3 [89, 6113] root21Candidates48 6202 winner else winner) = _
  rw [if_pos (by decide), root21Child48]
def root21Candidates49 : List Nat := [9791, 11489]
set_option maxRecDepth 8192 in
theorem root21Candidates49_checked : candidates edge 4 89 6263 root21Tail50 winner.bound = root21Candidates49 := by
  unfold candidates
  have row : edge 6263 = (SortedNatTree.contains FastRows.row6263) := by
    change Data.row6263.contains = (SortedNatTree.contains FastRows.row6263)
    exact FastRows.row6263_checked.symm
  rw [row]
  decide
theorem root21Step49 : scan edge (search edge 3) 4 [89] 89 root21Tail49 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail50 winner := by
  rw [root21Tail49, scan, if_neg (by decide), root21Candidates49_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail50
    (if 3 ≤ root21Candidates49.length then search edge 3 [89, 6263] root21Candidates49 6352 winner else winner) = _
  rw [if_neg (by decide)]
def root21Candidates50 : List Nat := [8147, 12347]
set_option maxRecDepth 8192 in
theorem root21Candidates50_checked : candidates edge 4 89 6299 root21Tail51 winner.bound = root21Candidates50 := by
  unfold candidates
  have row : edge 6299 = (SortedNatTree.contains FastRows.row6299) := by
    change Data.row6299.contains = (SortedNatTree.contains FastRows.row6299)
    exact FastRows.row6299_checked.symm
  rw [row]
  decide
theorem root21Step50 : scan edge (search edge 3) 4 [89] 89 root21Tail50 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail51 winner := by
  rw [root21Tail50, scan, if_neg (by decide), root21Candidates50_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail51
    (if 3 ≤ root21Candidates50.length then search edge 3 [89, 6299] root21Candidates50 6388 winner else winner) = _
  rw [if_neg (by decide)]
def root21Candidates51 : List Nat := [10271, 12347]
set_option maxRecDepth 8192 in
theorem root21Candidates51_checked : candidates edge 4 89 6323 root21Tail52 winner.bound = root21Candidates51 := by
  unfold candidates
  have row : edge 6323 = (SortedNatTree.contains FastRows.row6323) := by
    change Data.row6323.contains = (SortedNatTree.contains FastRows.row6323)
    exact FastRows.row6323_checked.symm
  rw [row]
  decide
theorem root21Step51 : scan edge (search edge 3) 4 [89] 89 root21Tail51 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail52 winner := by
  rw [root21Tail51, scan, if_neg (by decide), root21Candidates51_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail52
    (if 3 ≤ root21Candidates51.length then search edge 3 [89, 6323] root21Candidates51 6412 winner else winner) = _
  rw [if_neg (by decide)]
def root21Candidates52 : List Nat := [15299, 18797]
set_option maxRecDepth 8192 in
theorem root21Candidates52_checked : candidates edge 4 89 6353 root21Tail53 winner.bound = root21Candidates52 := by
  unfold candidates
  have row : edge 6353 = (SortedNatTree.contains FastRows.row6353) := by
    change Data.row6353.contains = (SortedNatTree.contains FastRows.row6353)
    exact FastRows.row6353_checked.symm
  rw [row]
  decide
theorem root21Step52 : scan edge (search edge 3) 4 [89] 89 root21Tail52 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail53 winner := by
  rw [root21Tail52, scan, if_neg (by decide), root21Candidates52_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail53
    (if 3 ≤ root21Candidates52.length then search edge 3 [89, 6353] root21Candidates52 6442 winner else winner) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
