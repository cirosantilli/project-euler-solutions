import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot007Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root7Candidates32 : List Nat := [7757, 15161]
set_option maxRecDepth 8192 in
theorem root7Candidates32_checked : candidates edge 4 29 4241 root7Tail33 winner.bound = root7Candidates32 := by
  unfold candidates
  have row : edge 4241 = (SortedNatTree.contains FastRows.row4241) := by
    change Data.row4241.contains = (SortedNatTree.contains FastRows.row4241)
    exact FastRows.row4241_checked.symm
  rw [row]
  decide
theorem root7Step32 : scan edge (search edge 3) 4 [29] 29 root7Tail32 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail33 winner := by
  rw [root7Tail32, scan, if_neg (by decide), root7Candidates32_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail33
    (if 3 ≤ root7Candidates32.length then search edge 3 [29, 4241] root7Candidates32 4270 winner else winner) = _
  rw [if_neg (by decide)]
def root7Candidates33 : List Nat := [4523, 5237]
set_option maxRecDepth 8192 in
theorem root7Candidates33_checked : candidates edge 4 29 4337 root7Tail34 winner.bound = root7Candidates33 := by
  unfold candidates
  have row : edge 4337 = (SortedNatTree.contains FastRows.row4337) := by
    change Data.row4337.contains = (SortedNatTree.contains FastRows.row4337)
    exact FastRows.row4337_checked.symm
  rw [row]
  decide
theorem root7Step33 : scan edge (search edge 3) 4 [29] 29 root7Tail33 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail34 winner := by
  rw [root7Tail33, scan, if_neg (by decide), root7Candidates33_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail34
    (if 3 ≤ root7Candidates33.length then search edge 3 [29, 4337] root7Candidates33 4366 winner else winner) = _
  rw [if_neg (by decide)]
def root7Candidates34 : List Nat := [6563, 7841, 8861, 12473]
set_option maxRecDepth 8192 in
theorem root7Candidates34_checked : candidates edge 4 29 4397 root7Tail35 winner.bound = root7Candidates34 := by
  unfold candidates
  have row : edge 4397 = (SortedNatTree.contains FastRows.row4397) := by
    change Data.row4397.contains = (SortedNatTree.contains FastRows.row4397)
    exact FastRows.row4397_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child34 : search edge 3 [29, 4397] root7Candidates34 4426 winner = winner := by decide
theorem root7Step34 : scan edge (search edge 3) 4 [29] 29 root7Tail34 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail35 winner := by
  rw [root7Tail34, scan, if_neg (by decide), root7Candidates34_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail35
    (if 3 ≤ root7Candidates34.length then search edge 3 [29, 4397] root7Candidates34 4426 winner else winner) = _
  rw [if_pos (by decide), root7Child34]
def root7Candidates35 : List Nat := [5039, 5081, 6911, 9743, 18593]
set_option maxRecDepth 8192 in
theorem root7Candidates35_checked : candidates edge 4 29 4523 root7Tail36 winner.bound = root7Candidates35 := by
  unfold candidates
  have row : edge 4523 = (SortedNatTree.contains FastRows.row4523) := by
    change Data.row4523.contains = (SortedNatTree.contains FastRows.row4523)
    exact FastRows.row4523_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child35 : search edge 3 [29, 4523] root7Candidates35 4552 winner = winner := by decide
theorem root7Step35 : scan edge (search edge 3) 4 [29] 29 root7Tail35 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail36 winner := by
  rw [root7Tail35, scan, if_neg (by decide), root7Candidates35_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail36
    (if 3 ≤ root7Candidates35.length then search edge 3 [29, 4523] root7Candidates35 4552 winner else winner) = _
  rw [if_pos (by decide), root7Child35]
def root7Candidates36 : List Nat := []
set_option maxRecDepth 8192 in
theorem root7Candidates36_checked : candidates edge 4 29 4673 root7Tail37 winner.bound = root7Candidates36 := by
  unfold candidates
  have row : edge 4673 = (SortedNatTree.contains FastRows.row4673) := by
    change Data.row4673.contains = (SortedNatTree.contains FastRows.row4673)
    exact FastRows.row4673_checked.symm
  rw [row]
  decide
theorem root7Step36 : scan edge (search edge 3) 4 [29] 29 root7Tail36 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail37 winner := by
  rw [root7Tail36, scan, if_neg (by decide), root7Candidates36_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail37
    (if 3 ≤ root7Candidates36.length then search edge 3 [29, 4673] root7Candidates36 4702 winner else winner) = _
  rw [if_neg (by decide)]
def root7Candidates37 : List Nat := [5237, 6047, 12893, 13553, 18593]
set_option maxRecDepth 8192 in
theorem root7Candidates37_checked : candidates edge 4 29 4787 root7Tail38 winner.bound = root7Candidates37 := by
  unfold candidates
  have row : edge 4787 = (SortedNatTree.contains FastRows.row4787) := by
    change Data.row4787.contains = (SortedNatTree.contains FastRows.row4787)
    exact FastRows.row4787_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child37 : search edge 3 [29, 4787] root7Candidates37 4816 winner = winner := by decide
theorem root7Step37 : scan edge (search edge 3) 4 [29] 29 root7Tail37 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail38 winner := by
  rw [root7Tail37, scan, if_neg (by decide), root7Candidates37_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail38
    (if 3 ≤ root7Candidates37.length then search edge 3 [29, 4787] root7Candidates37 4816 winner else winner) = _
  rw [if_pos (by decide), root7Child37]
def root7Candidates38 : List Nat := [6299, 9539]
set_option maxRecDepth 8192 in
theorem root7Candidates38_checked : candidates edge 4 29 5039 root7Tail39 winner.bound = root7Candidates38 := by
  unfold candidates
  have row : edge 5039 = (SortedNatTree.contains FastRows.row5039) := by
    change Data.row5039.contains = (SortedNatTree.contains FastRows.row5039)
    exact FastRows.row5039_checked.symm
  rw [row]
  decide
theorem root7Step38 : scan edge (search edge 3) 4 [29] 29 root7Tail38 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail39 winner := by
  rw [root7Tail38, scan, if_neg (by decide), root7Candidates38_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail39
    (if 3 ≤ root7Candidates38.length then search edge 3 [29, 5039] root7Candidates38 5068 winner else winner) = _
  rw [if_neg (by decide)]
def root7Candidates39 : List Nat := [11621, 20399, 20903]
set_option maxRecDepth 8192 in
theorem root7Candidates39_checked : candidates edge 4 29 5081 root7Tail40 winner.bound = root7Candidates39 := by
  unfold candidates
  have row : edge 5081 = (SortedNatTree.contains FastRows.row5081) := by
    change Data.row5081.contains = (SortedNatTree.contains FastRows.row5081)
    exact FastRows.row5081_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child39 : search edge 3 [29, 5081] root7Candidates39 5110 winner = winner := by decide
theorem root7Step39 : scan edge (search edge 3) 4 [29] 29 root7Tail39 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail40 winner := by
  rw [root7Tail39, scan, if_neg (by decide), root7Candidates39_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail40
    (if 3 ≤ root7Candidates39.length then search edge 3 [29, 5081] root7Candidates39 5110 winner else winner) = _
  rw [if_pos (by decide), root7Child39]
def root7Candidates40 : List Nat := [6563, 7487, 10133, 15161, 18719]
set_option maxRecDepth 8192 in
theorem root7Candidates40_checked : candidates edge 4 29 5237 root7Tail41 winner.bound = root7Candidates40 := by
  unfold candidates
  have row : edge 5237 = (SortedNatTree.contains FastRows.row5237) := by
    change Data.row5237.contains = (SortedNatTree.contains FastRows.row5237)
    exact FastRows.row5237_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child40 : search edge 3 [29, 5237] root7Candidates40 5266 winner = winner := by decide
theorem root7Step40 : scan edge (search edge 3) 4 [29] 29 root7Tail40 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail41 winner := by
  rw [root7Tail40, scan, if_neg (by decide), root7Candidates40_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail41
    (if 3 ≤ root7Candidates40.length then search edge 3 [29, 5237] root7Candidates40 5266 winner else winner) = _
  rw [if_pos (by decide), root7Child40]
def root7Candidates41 : List Nat := [19013]
set_option maxRecDepth 8192 in
theorem root7Candidates41_checked : candidates edge 4 29 5441 root7Tail42 winner.bound = root7Candidates41 := by
  unfold candidates
  have row : edge 5441 = (SortedNatTree.contains FastRows.row5441) := by
    change Data.row5441.contains = (SortedNatTree.contains FastRows.row5441)
    exact FastRows.row5441_checked.symm
  rw [row]
  decide
theorem root7Step41 : scan edge (search edge 3) 4 [29] 29 root7Tail41 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail42 winner := by
  rw [root7Tail41, scan, if_neg (by decide), root7Candidates41_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail42
    (if 3 ≤ root7Candidates41.length then search edge 3 [29, 5441] root7Candidates41 5470 winner else winner) = _
  rw [if_neg (by decide)]
def root7Candidates42 : List Nat := [6827, 10169, 11621]
set_option maxRecDepth 8192 in
theorem root7Candidates42_checked : candidates edge 4 29 5861 root7Tail43 winner.bound = root7Candidates42 := by
  unfold candidates
  have row : edge 5861 = (SortedNatTree.contains FastRows.row5861) := by
    change Data.row5861.contains = (SortedNatTree.contains FastRows.row5861)
    exact FastRows.row5861_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child42 : search edge 3 [29, 5861] root7Candidates42 5890 winner = winner := by decide
theorem root7Step42 : scan edge (search edge 3) 4 [29] 29 root7Tail42 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail43 winner := by
  rw [root7Tail42, scan, if_neg (by decide), root7Candidates42_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail43
    (if 3 ≤ root7Candidates42.length then search edge 3 [29, 5861] root7Candidates42 5890 winner else winner) = _
  rw [if_pos (by decide), root7Child42]
def root7Candidates43 : List Nat := [9311]
set_option maxRecDepth 8192 in
theorem root7Candidates43_checked : candidates edge 4 29 6047 root7Tail44 winner.bound = root7Candidates43 := by
  unfold candidates
  have row : edge 6047 = (SortedNatTree.contains FastRows.row6047) := by
    change Data.row6047.contains = (SortedNatTree.contains FastRows.row6047)
    exact FastRows.row6047_checked.symm
  rw [row]
  decide
theorem root7Step43 : scan edge (search edge 3) 4 [29] 29 root7Tail43 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail44 winner := by
  rw [root7Tail43, scan, if_neg (by decide), root7Candidates43_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail44
    (if 3 ≤ root7Candidates43.length then search edge 3 [29, 6047] root7Candidates43 6076 winner else winner) = _
  rw [if_neg (by decide)]
def root7Candidates44 : List Nat := [14489, 18593]
set_option maxRecDepth 8192 in
theorem root7Candidates44_checked : candidates edge 4 29 6221 root7Tail45 winner.bound = root7Candidates44 := by
  unfold candidates
  have row : edge 6221 = (SortedNatTree.contains FastRows.row6221) := by
    change Data.row6221.contains = (SortedNatTree.contains FastRows.row6221)
    exact FastRows.row6221_checked.symm
  rw [row]
  decide
theorem root7Step44 : scan edge (search edge 3) 4 [29] 29 root7Tail44 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail45 winner := by
  rw [root7Tail44, scan, if_neg (by decide), root7Candidates44_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail45
    (if 3 ≤ root7Candidates44.length then search edge 3 [29, 6221] root7Candidates44 6250 winner else winner) = _
  rw [if_neg (by decide)]
def root7Candidates45 : List Nat := [10169]
set_option maxRecDepth 8192 in
theorem root7Candidates45_checked : candidates edge 4 29 6269 root7Tail46 winner.bound = root7Candidates45 := by
  unfold candidates
  have row : edge 6269 = (SortedNatTree.contains FastRows.row6269) := by
    change Data.row6269.contains = (SortedNatTree.contains FastRows.row6269)
    exact FastRows.row6269_checked.symm
  rw [row]
  decide
theorem root7Step45 : scan edge (search edge 3) 4 [29] 29 root7Tail45 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail46 winner := by
  rw [root7Tail45, scan, if_neg (by decide), root7Candidates45_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail46
    (if 3 ≤ root7Candidates45.length then search edge 3 [29, 6269] root7Candidates45 6298 winner else winner) = _
  rw [if_neg (by decide)]
def root7Candidates46 : List Nat := [7589, 9857, 10133, 12641, 13451]
set_option maxRecDepth 8192 in
theorem root7Candidates46_checked : candidates edge 4 29 6299 root7Tail47 winner.bound = root7Candidates46 := by
  unfold candidates
  have row : edge 6299 = (SortedNatTree.contains FastRows.row6299) := by
    change Data.row6299.contains = (SortedNatTree.contains FastRows.row6299)
    exact FastRows.row6299_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child46 : search edge 3 [29, 6299] root7Candidates46 6328 winner = winner := by decide
theorem root7Step46 : scan edge (search edge 3) 4 [29] 29 root7Tail46 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail47 winner := by
  rw [root7Tail46, scan, if_neg (by decide), root7Candidates46_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail47
    (if 3 ≤ root7Candidates46.length then search edge 3 [29, 6299] root7Candidates46 6328 winner else winner) = _
  rw [if_pos (by decide), root7Child46]
end ProjectEulerSolutions.PrimePairLarge.Search
