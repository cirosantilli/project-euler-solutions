import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot008Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root8Candidates32 : List Nat := [4723, 5851, 9049, 9511, 18637]
set_option maxRecDepth 8192 in
theorem root8Candidates32_checked : candidates edge 4 31 4003 root8Tail33 winner.bound = root8Candidates32 := by
  unfold candidates
  have row : edge 4003 = (SortedNatTree.contains FastRows.row4003) := by
    change Data.row4003.contains = (SortedNatTree.contains FastRows.row4003)
    exact FastRows.row4003_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child32 : search edge 3 [31, 4003] root8Candidates32 4034 winner = winner := by decide
theorem root8Step32 : scan edge (search edge 3) 4 [31] 31 root8Tail32 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail33 winner := by
  rw [root8Tail32, scan, if_neg (by decide), root8Candidates32_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail33
    (if 3 ≤ root8Candidates32.length then search edge 3 [31, 4003] root8Candidates32 4034 winner else winner) = _
  rw [if_pos (by decide), root8Child32]
def root8Candidates33 : List Nat := [4441, 14143, 17923, 20857]
set_option maxRecDepth 8192 in
theorem root8Candidates33_checked : candidates edge 4 31 4021 root8Tail34 winner.bound = root8Candidates33 := by
  unfold candidates
  have row : edge 4021 = (SortedNatTree.contains FastRows.row4021) := by
    change Data.row4021.contains = (SortedNatTree.contains FastRows.row4021)
    exact FastRows.row4021_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child33 : search edge 3 [31, 4021] root8Candidates33 4052 winner = winner := by decide
theorem root8Step33 : scan edge (search edge 3) 4 [31] 31 root8Tail33 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail34 winner := by
  rw [root8Tail33, scan, if_neg (by decide), root8Candidates33_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail34
    (if 3 ≤ root8Candidates33.length then search edge 3 [31, 4021] root8Candidates33 4052 winner else winner) = _
  rw [if_pos (by decide), root8Child33]
def root8Candidates34 : List Nat := [5011, 5779, 11677, 17209]
set_option maxRecDepth 8192 in
theorem root8Candidates34_checked : candidates edge 4 31 4159 root8Tail35 winner.bound = root8Candidates34 := by
  unfold candidates
  have row : edge 4159 = (SortedNatTree.contains FastRows.row4159) := by
    change Data.row4159.contains = (SortedNatTree.contains FastRows.row4159)
    exact FastRows.row4159_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child34 : search edge 3 [31, 4159] root8Candidates34 4190 winner = winner := by decide
theorem root8Step34 : scan edge (search edge 3) 4 [31] 31 root8Tail34 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail35 winner := by
  rw [root8Tail34, scan, if_neg (by decide), root8Candidates34_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail35
    (if 3 ≤ root8Candidates34.length then search edge 3 [31, 4159] root8Candidates34 4190 winner else winner) = _
  rw [if_pos (by decide), root8Child34]
def root8Candidates35 : List Nat := [4339, 5527, 11491, 17209, 19309, 20857]
set_option maxRecDepth 8192 in
theorem root8Candidates35_checked : candidates edge 4 31 4243 root8Tail36 winner.bound = root8Candidates35 := by
  unfold candidates
  have row : edge 4243 = (SortedNatTree.contains FastRows.row4243) := by
    change Data.row4243.contains = (SortedNatTree.contains FastRows.row4243)
    exact FastRows.row4243_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child35 : search edge 3 [31, 4243] root8Candidates35 4274 winner = winner := by decide
theorem root8Step35 : scan edge (search edge 3) 4 [31] 31 root8Tail35 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail36 winner := by
  rw [root8Tail35, scan, if_neg (by decide), root8Candidates35_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail36
    (if 3 ≤ root8Candidates35.length then search edge 3 [31, 4243] root8Candidates35 4274 winner else winner) = _
  rw [if_pos (by decide), root8Child35]
def root8Candidates36 : List Nat := [5011, 5281, 9817, 18541]
set_option maxRecDepth 8192 in
theorem root8Candidates36_checked : candidates edge 4 31 4261 root8Tail37 winner.bound = root8Candidates36 := by
  unfold candidates
  have row : edge 4261 = (SortedNatTree.contains FastRows.row4261) := by
    change Data.row4261.contains = (SortedNatTree.contains FastRows.row4261)
    exact FastRows.row4261_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child36 : search edge 3 [31, 4261] root8Candidates36 4292 winner = winner := by decide
theorem root8Step36 : scan edge (search edge 3) 4 [31] 31 root8Tail36 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail37 winner := by
  rw [root8Tail36, scan, if_neg (by decide), root8Candidates36_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail37
    (if 3 ≤ root8Candidates36.length then search edge 3 [31, 4261] root8Candidates36 4292 winner else winner) = _
  rw [if_pos (by decide), root8Child36]
def root8Candidates37 : List Nat := [5701, 9679, 10651, 19309]
set_option maxRecDepth 8192 in
theorem root8Candidates37_checked : candidates edge 4 31 4339 root8Tail38 winner.bound = root8Candidates37 := by
  unfold candidates
  have row : edge 4339 = (SortedNatTree.contains FastRows.row4339) := by
    change Data.row4339.contains = (SortedNatTree.contains FastRows.row4339)
    exact FastRows.row4339_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child37 : search edge 3 [31, 4339] root8Candidates37 4370 winner = winner := by decide
theorem root8Step37 : scan edge (search edge 3) 4 [31] 31 root8Tail37 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail38 winner := by
  rw [root8Tail37, scan, if_neg (by decide), root8Candidates37_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail38
    (if 3 ≤ root8Candidates37.length then search edge 3 [31, 4339] root8Candidates37 4370 winner else winner) = _
  rw [if_pos (by decide), root8Child37]
def root8Candidates38 : List Nat := [7411, 11491, 16693]
set_option maxRecDepth 8192 in
theorem root8Candidates38_checked : candidates edge 4 31 4357 root8Tail39 winner.bound = root8Candidates38 := by
  unfold candidates
  have row : edge 4357 = (SortedNatTree.contains FastRows.row4357) := by
    change Data.row4357.contains = (SortedNatTree.contains FastRows.row4357)
    exact FastRows.row4357_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child38 : search edge 3 [31, 4357] root8Candidates38 4388 winner = winner := by decide
theorem root8Step38 : scan edge (search edge 3) 4 [31] 31 root8Tail38 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail39 winner := by
  rw [root8Tail38, scan, if_neg (by decide), root8Candidates38_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail39
    (if 3 ≤ root8Candidates38.length then search edge 3 [31, 4357] root8Candidates38 4388 winner else winner) = _
  rw [if_pos (by decide), root8Child38]
def root8Candidates39 : List Nat := [6271, 17497]
set_option maxRecDepth 8192 in
theorem root8Candidates39_checked : candidates edge 4 31 4441 root8Tail40 winner.bound = root8Candidates39 := by
  unfold candidates
  have row : edge 4441 = (SortedNatTree.contains FastRows.row4441) := by
    change Data.row4441.contains = (SortedNatTree.contains FastRows.row4441)
    exact FastRows.row4441_checked.symm
  rw [row]
  decide
theorem root8Step39 : scan edge (search edge 3) 4 [31] 31 root8Tail39 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail40 winner := by
  rw [root8Tail39, scan, if_neg (by decide), root8Candidates39_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail40
    (if 3 ≤ root8Candidates39.length then search edge 3 [31, 4441] root8Candidates39 4472 winner else winner) = _
  rw [if_neg (by decide)]
def root8Candidates40 : List Nat := [8287, 9049]
set_option maxRecDepth 8192 in
theorem root8Candidates40_checked : candidates edge 4 31 4513 root8Tail41 winner.bound = root8Candidates40 := by
  unfold candidates
  have row : edge 4513 = (SortedNatTree.contains FastRows.row4513) := by
    change Data.row4513.contains = (SortedNatTree.contains FastRows.row4513)
    exact FastRows.row4513_checked.symm
  rw [row]
  decide
theorem root8Step40 : scan edge (search edge 3) 4 [31] 31 root8Tail40 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail41 winner := by
  rw [root8Tail40, scan, if_neg (by decide), root8Candidates40_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail41
    (if 3 ≤ root8Candidates40.length then search edge 3 [31, 4513] root8Candidates40 4544 winner else winner) = _
  rw [if_neg (by decide)]
def root8Candidates41 : List Nat := [5851, 6343, 6571, 8287, 9133, 10273, 15667]
set_option maxRecDepth 8192 in
theorem root8Candidates41_checked : candidates edge 4 31 4549 root8Tail42 winner.bound = root8Candidates41 := by
  unfold candidates
  have row : edge 4549 = (SortedNatTree.contains FastRows.row4549) := by
    change Data.row4549.contains = (SortedNatTree.contains FastRows.row4549)
    exact FastRows.row4549_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child41 : search edge 3 [31, 4549] root8Candidates41 4580 winner = winner := by decide
theorem root8Step41 : scan edge (search edge 3) 4 [31] 31 root8Tail41 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail42 winner := by
  rw [root8Tail41, scan, if_neg (by decide), root8Candidates41_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail42
    (if 3 ≤ root8Candidates41.length then search edge 3 [31, 4549] root8Candidates41 4580 winner else winner) = _
  rw [if_pos (by decide), root8Child41]
def root8Candidates42 : List Nat := []
set_option maxRecDepth 8192 in
theorem root8Candidates42_checked : candidates edge 4 31 4723 root8Tail43 winner.bound = root8Candidates42 := by
  unfold candidates
  have row : edge 4723 = (SortedNatTree.contains FastRows.row4723) := by
    change Data.row4723.contains = (SortedNatTree.contains FastRows.row4723)
    exact FastRows.row4723_checked.symm
  rw [row]
  decide
theorem root8Step42 : scan edge (search edge 3) 4 [31] 31 root8Tail42 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail43 winner := by
  rw [root8Tail42, scan, if_neg (by decide), root8Candidates42_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail43
    (if 3 ≤ root8Candidates42.length then search edge 3 [31, 4723] root8Candidates42 4754 winner else winner) = _
  rw [if_neg (by decide)]
def root8Candidates43 : List Nat := [6271, 8191, 9049, 13009, 17497, 20857]
set_option maxRecDepth 8192 in
theorem root8Candidates43_checked : candidates edge 4 31 5011 root8Tail44 winner.bound = root8Candidates43 := by
  unfold candidates
  have row : edge 5011 = (SortedNatTree.contains FastRows.row5011) := by
    change Data.row5011.contains = (SortedNatTree.contains FastRows.row5011)
    exact FastRows.row5011_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child43 : search edge 3 [31, 5011] root8Candidates43 5042 winner = winner := by decide
theorem root8Step43 : scan edge (search edge 3) 4 [31] 31 root8Tail43 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail44 winner := by
  rw [root8Tail43, scan, if_neg (by decide), root8Candidates43_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail44
    (if 3 ≤ root8Candidates43.length then search edge 3 [31, 5011] root8Candidates43 5042 winner else winner) = _
  rw [if_pos (by decide), root8Child43]
def root8Candidates44 : List Nat := [5851, 9439, 11287, 11491]
set_option maxRecDepth 8192 in
theorem root8Candidates44_checked : candidates edge 4 31 5179 root8Tail45 winner.bound = root8Candidates44 := by
  unfold candidates
  have row : edge 5179 = (SortedNatTree.contains FastRows.row5179) := by
    change Data.row5179.contains = (SortedNatTree.contains FastRows.row5179)
    exact FastRows.row5179_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child44 : search edge 3 [31, 5179] root8Candidates44 5210 winner = winner := by decide
theorem root8Step44 : scan edge (search edge 3) 4 [31] 31 root8Tail44 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail45 winner := by
  rw [root8Tail44, scan, if_neg (by decide), root8Candidates44_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail45
    (if 3 ≤ root8Candidates44.length then search edge 3 [31, 5179] root8Candidates44 5210 winner else winner) = _
  rw [if_pos (by decide), root8Child44]
def root8Candidates45 : List Nat := [8287, 20443]
set_option maxRecDepth 8192 in
theorem root8Candidates45_checked : candidates edge 4 31 5281 root8Tail46 winner.bound = root8Candidates45 := by
  unfold candidates
  have row : edge 5281 = (SortedNatTree.contains FastRows.row5281) := by
    change Data.row5281.contains = (SortedNatTree.contains FastRows.row5281)
    exact FastRows.row5281_checked.symm
  rw [row]
  decide
theorem root8Step45 : scan edge (search edge 3) 4 [31] 31 root8Tail45 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail46 winner := by
  rw [root8Tail45, scan, if_neg (by decide), root8Candidates45_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail46
    (if 3 ≤ root8Candidates45.length then search edge 3 [31, 5281] root8Candidates45 5312 winner else winner) = _
  rw [if_neg (by decide)]
def root8Candidates46 : List Nat := [6343, 10867]
set_option maxRecDepth 8192 in
theorem root8Candidates46_checked : candidates edge 4 31 5527 root8Tail47 winner.bound = root8Candidates46 := by
  unfold candidates
  have row : edge 5527 = (SortedNatTree.contains FastRows.row5527) := by
    change Data.row5527.contains = (SortedNatTree.contains FastRows.row5527)
    exact FastRows.row5527_checked.symm
  rw [row]
  decide
theorem root8Step46 : scan edge (search edge 3) 4 [31] 31 root8Tail46 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail47 winner := by
  rw [root8Tail46, scan, if_neg (by decide), root8Candidates46_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail47
    (if 3 ≤ root8Candidates46.length then search edge 3 [31, 5527] root8Candidates46 5558 winner else winner) = _
  rw [if_neg (by decide)]
def root8Candidates47 : List Nat := [16033]
set_option maxRecDepth 8192 in
theorem root8Candidates47_checked : candidates edge 4 31 5569 root8Tail48 winner.bound = root8Candidates47 := by
  unfold candidates
  have row : edge 5569 = (SortedNatTree.contains FastRows.row5569) := by
    change Data.row5569.contains = (SortedNatTree.contains FastRows.row5569)
    exact FastRows.row5569_checked.symm
  rw [row]
  decide
theorem root8Step47 : scan edge (search edge 3) 4 [31] 31 root8Tail47 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail48 winner := by
  rw [root8Tail47, scan, if_neg (by decide), root8Candidates47_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail48
    (if 3 ≤ root8Candidates47.length then search edge 3 [31, 5569] root8Candidates47 5600 winner else winner) = _
  rw [if_neg (by decide)]
def root8Candidates48 : List Nat := [7459, 9439, 15187]
set_option maxRecDepth 8192 in
theorem root8Candidates48_checked : candidates edge 4 31 5701 root8Tail49 winner.bound = root8Candidates48 := by
  unfold candidates
  have row : edge 5701 = (SortedNatTree.contains FastRows.row5701) := by
    change Data.row5701.contains = (SortedNatTree.contains FastRows.row5701)
    exact FastRows.row5701_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child48 : search edge 3 [31, 5701] root8Candidates48 5732 winner = winner := by decide
theorem root8Step48 : scan edge (search edge 3) 4 [31] 31 root8Tail48 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail49 winner := by
  rw [root8Tail48, scan, if_neg (by decide), root8Candidates48_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail49
    (if 3 ≤ root8Candidates48.length then search edge 3 [31, 5701] root8Candidates48 5732 winner else winner) = _
  rw [if_pos (by decide), root8Child48]
def root8Candidates49 : List Nat := [7591]
set_option maxRecDepth 8192 in
theorem root8Candidates49_checked : candidates edge 4 31 5779 root8Tail50 winner.bound = root8Candidates49 := by
  unfold candidates
  have row : edge 5779 = (SortedNatTree.contains FastRows.row5779) := by
    change Data.row5779.contains = (SortedNatTree.contains FastRows.row5779)
    exact FastRows.row5779_checked.symm
  rw [row]
  decide
theorem root8Step49 : scan edge (search edge 3) 4 [31] 31 root8Tail49 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail50 winner := by
  rw [root8Tail49, scan, if_neg (by decide), root8Candidates49_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail50
    (if 3 ≤ root8Candidates49.length then search edge 3 [31, 5779] root8Candidates49 5810 winner else winner) = _
  rw [if_neg (by decide)]
def root8Candidates50 : List Nat := [8377, 9733, 18637]
set_option maxRecDepth 8192 in
theorem root8Candidates50_checked : candidates edge 4 31 5851 root8Tail51 winner.bound = root8Candidates50 := by
  unfold candidates
  have row : edge 5851 = (SortedNatTree.contains FastRows.row5851) := by
    change Data.row5851.contains = (SortedNatTree.contains FastRows.row5851)
    exact FastRows.row5851_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child50 : search edge 3 [31, 5851] root8Candidates50 5882 winner = winner := by decide
theorem root8Step50 : scan edge (search edge 3) 4 [31] 31 root8Tail50 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail51 winner := by
  rw [root8Tail50, scan, if_neg (by decide), root8Candidates50_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail51
    (if 3 ≤ root8Candidates50.length then search edge 3 [31, 5851] root8Candidates50 5882 winner else winner) = _
  rw [if_pos (by decide), root8Child50]
def root8Candidates51 : List Nat := [10867, 15991]
set_option maxRecDepth 8192 in
theorem root8Candidates51_checked : candidates edge 4 31 5881 root8Tail52 winner.bound = root8Candidates51 := by
  unfold candidates
  have row : edge 5881 = (SortedNatTree.contains FastRows.row5881) := by
    change Data.row5881.contains = (SortedNatTree.contains FastRows.row5881)
    exact FastRows.row5881_checked.symm
  rw [row]
  decide
theorem root8Step51 : scan edge (search edge 3) 4 [31] 31 root8Tail51 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail52 winner := by
  rw [root8Tail51, scan, if_neg (by decide), root8Candidates51_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail52
    (if 3 ≤ root8Candidates51.length then search edge 3 [31, 5881] root8Candidates51 5912 winner else winner) = _
  rw [if_neg (by decide)]
def root8Candidates52 : List Nat := []
set_option maxRecDepth 8192 in
theorem root8Candidates52_checked : candidates edge 4 31 6037 root8Tail53 winner.bound = root8Candidates52 := by
  unfold candidates
  have row : edge 6037 = (SortedNatTree.contains FastRows.row6037) := by
    change Data.row6037.contains = (SortedNatTree.contains FastRows.row6037)
    exact FastRows.row6037_checked.symm
  rw [row]
  decide
theorem root8Step52 : scan edge (search edge 3) 4 [31] 31 root8Tail52 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail53 winner := by
  rw [root8Tail52, scan, if_neg (by decide), root8Candidates52_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail53
    (if 3 ≤ root8Candidates52.length then search edge 3 [31, 6037] root8Candidates52 6068 winner else winner) = _
  rw [if_neg (by decide)]
def root8Candidates53 : List Nat := []
set_option maxRecDepth 8192 in
theorem root8Candidates53_checked : candidates edge 4 31 6067 root8Tail54 winner.bound = root8Candidates53 := by
  unfold candidates
  have row : edge 6067 = (SortedNatTree.contains FastRows.row6067) := by
    change Data.row6067.contains = (SortedNatTree.contains FastRows.row6067)
    exact FastRows.row6067_checked.symm
  rw [row]
  decide
theorem root8Step53 : scan edge (search edge 3) 4 [31] 31 root8Tail53 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail54 winner := by
  rw [root8Tail53, scan, if_neg (by decide), root8Candidates53_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail54
    (if 3 ≤ root8Candidates53.length then search edge 3 [31, 6067] root8Candidates53 6098 winner else winner) = _
  rw [if_neg (by decide)]
def root8Candidates54 : List Nat := []
set_option maxRecDepth 8192 in
theorem root8Candidates54_checked : candidates edge 4 31 6073 root8Tail55 winner.bound = root8Candidates54 := by
  unfold candidates
  have row : edge 6073 = (SortedNatTree.contains FastRows.row6073) := by
    change Data.row6073.contains = (SortedNatTree.contains FastRows.row6073)
    exact FastRows.row6073_checked.symm
  rw [row]
  decide
theorem root8Step54 : scan edge (search edge 3) 4 [31] 31 root8Tail54 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail55 winner := by
  rw [root8Tail54, scan, if_neg (by decide), root8Candidates54_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail55
    (if 3 ≤ root8Candidates54.length then search edge 3 [31, 6073] root8Candidates54 6104 winner else winner) = _
  rw [if_neg (by decide)]
def root8Candidates55 : List Nat := [8641, 11743, 13723, 14143, 17209, 18637, 19309]
set_option maxRecDepth 8192 in
theorem root8Candidates55_checked : candidates edge 4 31 6271 root8Tail56 winner.bound = root8Candidates55 := by
  unfold candidates
  have row : edge 6271 = (SortedNatTree.contains FastRows.row6271) := by
    change Data.row6271.contains = (SortedNatTree.contains FastRows.row6271)
    exact FastRows.row6271_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child55 : search edge 3 [31, 6271] root8Candidates55 6302 winner = winner := by decide
theorem root8Step55 : scan edge (search edge 3) 4 [31] 31 root8Tail55 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail56 winner := by
  rw [root8Tail55, scan, if_neg (by decide), root8Candidates55_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail56
    (if 3 ≤ root8Candidates55.length then search edge 3 [31, 6271] root8Candidates55 6302 winner else winner) = _
  rw [if_pos (by decide), root8Child55]
def root8Candidates56 : List Nat := [10321, 14143]
set_option maxRecDepth 8192 in
theorem root8Candidates56_checked : candidates edge 4 31 6343 root8Tail57 winner.bound = root8Candidates56 := by
  unfold candidates
  have row : edge 6343 = (SortedNatTree.contains FastRows.row6343) := by
    change Data.row6343.contains = (SortedNatTree.contains FastRows.row6343)
    exact FastRows.row6343_checked.symm
  rw [row]
  decide
theorem root8Step56 : scan edge (search edge 3) 4 [31] 31 root8Tail56 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail57 winner := by
  rw [root8Tail56, scan, if_neg (by decide), root8Candidates56_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail57
    (if 3 ≤ root8Candidates56.length then search edge 3 [31, 6343] root8Candidates56 6374 winner else winner) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
