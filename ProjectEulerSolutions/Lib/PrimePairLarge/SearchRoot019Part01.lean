import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot019Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root19Candidates32 : List Nat := [18637]
set_option maxRecDepth 8192 in
theorem root19Candidates32_checked : candidates edge 4 79 4231 root19Tail33 winner.bound = root19Candidates32 := by
  unfold candidates
  have row : edge 4231 = (SortedNatTree.contains FastRows.row4231) := by
    change Data.row4231.contains = (SortedNatTree.contains FastRows.row4231)
    exact FastRows.row4231_checked.symm
  rw [row]
  decide
theorem root19Step32 : scan edge (search edge 3) 4 [79] 79 root19Tail32 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail33 winner := by
  rw [root19Tail32, scan, if_neg (by decide), root19Candidates32_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail33
    (if 3 ≤ root19Candidates32.length then search edge 3 [79, 4231] root19Candidates32 4310 winner else winner) = _
  rw [if_neg (by decide)]
def root19Candidates33 : List Nat := []
set_option maxRecDepth 8192 in
theorem root19Candidates33_checked : candidates edge 4 79 4483 root19Tail34 winner.bound = root19Candidates33 := by
  unfold candidates
  have row : edge 4483 = (SortedNatTree.contains FastRows.row4483) := by
    change Data.row4483.contains = (SortedNatTree.contains FastRows.row4483)
    exact FastRows.row4483_checked.symm
  rw [row]
  decide
theorem root19Step33 : scan edge (search edge 3) 4 [79] 79 root19Tail33 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail34 winner := by
  rw [root19Tail33, scan, if_neg (by decide), root19Candidates33_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail34
    (if 3 ≤ root19Candidates33.length then search edge 3 [79, 4483] root19Candidates33 4562 winner else winner) = _
  rw [if_neg (by decide)]
def root19Candidates34 : List Nat := [6451, 6553, 19993]
set_option maxRecDepth 8192 in
theorem root19Candidates34_checked : candidates edge 4 79 4831 root19Tail35 winner.bound = root19Candidates34 := by
  unfold candidates
  have row : edge 4831 = (SortedNatTree.contains FastRows.row4831) := by
    change Data.row4831.contains = (SortedNatTree.contains FastRows.row4831)
    exact FastRows.row4831_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root19Child34 : search edge 3 [79, 4831] root19Candidates34 4910 winner = winner := by decide
theorem root19Step34 : scan edge (search edge 3) 4 [79] 79 root19Tail34 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail35 winner := by
  rw [root19Tail34, scan, if_neg (by decide), root19Candidates34_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail35
    (if 3 ≤ root19Candidates34.length then search edge 3 [79, 4831] root19Candidates34 4910 winner else winner) = _
  rw [if_pos (by decide), root19Child34]
def root19Candidates35 : List Nat := [5791]
set_option maxRecDepth 8192 in
theorem root19Candidates35_checked : candidates edge 4 79 4999 root19Tail36 winner.bound = root19Candidates35 := by
  unfold candidates
  have row : edge 4999 = (SortedNatTree.contains FastRows.row4999) := by
    change Data.row4999.contains = (SortedNatTree.contains FastRows.row4999)
    exact FastRows.row4999_checked.symm
  rw [row]
  decide
theorem root19Step35 : scan edge (search edge 3) 4 [79] 79 root19Tail35 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail36 winner := by
  rw [root19Tail35, scan, if_neg (by decide), root19Candidates35_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail36
    (if 3 ≤ root19Candidates35.length then search edge 3 [79, 4999] root19Candidates35 5078 winner else winner) = _
  rw [if_neg (by decide)]
def root19Candidates36 : List Nat := [6091, 11551]
set_option maxRecDepth 8192 in
theorem root19Candidates36_checked : candidates edge 4 79 5077 root19Tail37 winner.bound = root19Candidates36 := by
  unfold candidates
  have row : edge 5077 = (SortedNatTree.contains FastRows.row5077) := by
    change Data.row5077.contains = (SortedNatTree.contains FastRows.row5077)
    exact FastRows.row5077_checked.symm
  rw [row]
  decide
theorem root19Step36 : scan edge (search edge 3) 4 [79] 79 root19Tail36 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail37 winner := by
  rw [root19Tail36, scan, if_neg (by decide), root19Candidates36_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail37
    (if 3 ≤ root19Candidates36.length then search edge 3 [79, 5077] root19Candidates36 5156 winner else winner) = _
  rw [if_neg (by decide)]
def root19Candidates37 : List Nat := [8263, 17137]
set_option maxRecDepth 8192 in
theorem root19Candidates37_checked : candidates edge 4 79 5101 root19Tail38 winner.bound = root19Candidates37 := by
  unfold candidates
  have row : edge 5101 = (SortedNatTree.contains FastRows.row5101) := by
    change Data.row5101.contains = (SortedNatTree.contains FastRows.row5101)
    exact FastRows.row5101_checked.symm
  rw [row]
  decide
theorem root19Step37 : scan edge (search edge 3) 4 [79] 79 root19Tail37 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail38 winner := by
  rw [root19Tail37, scan, if_neg (by decide), root19Candidates37_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail38
    (if 3 ≤ root19Candidates37.length then search edge 3 [79, 5101] root19Candidates37 5180 winner else winner) = _
  rw [if_neg (by decide)]
def root19Candidates38 : List Nat := [5659]
set_option maxRecDepth 8192 in
theorem root19Candidates38_checked : candidates edge 4 79 5323 root19Tail39 winner.bound = root19Candidates38 := by
  unfold candidates
  have row : edge 5323 = (SortedNatTree.contains FastRows.row5323) := by
    change Data.row5323.contains = (SortedNatTree.contains FastRows.row5323)
    exact FastRows.row5323_checked.symm
  rw [row]
  decide
theorem root19Step38 : scan edge (search edge 3) 4 [79] 79 root19Tail38 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail39 winner := by
  rw [root19Tail38, scan, if_neg (by decide), root19Candidates38_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail39
    (if 3 ≤ root19Candidates38.length then search edge 3 [79, 5323] root19Candidates38 5402 winner else winner) = _
  rw [if_neg (by decide)]
def root19Candidates39 : List Nat := [7549, 15643]
set_option maxRecDepth 8192 in
theorem root19Candidates39_checked : candidates edge 4 79 5449 root19Tail40 winner.bound = root19Candidates39 := by
  unfold candidates
  have row : edge 5449 = (SortedNatTree.contains FastRows.row5449) := by
    change Data.row5449.contains = (SortedNatTree.contains FastRows.row5449)
    exact FastRows.row5449_checked.symm
  rw [row]
  decide
theorem root19Step39 : scan edge (search edge 3) 4 [79] 79 root19Tail39 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail40 winner := by
  rw [root19Tail39, scan, if_neg (by decide), root19Candidates39_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail40
    (if 3 ≤ root19Candidates39.length then search edge 3 [79, 5449] root19Candidates39 5528 winner else winner) = _
  rw [if_neg (by decide)]
def root19Candidates40 : List Nat := [8521, 8887]
set_option maxRecDepth 8192 in
theorem root19Candidates40_checked : candidates edge 4 79 5503 root19Tail41 winner.bound = root19Candidates40 := by
  unfold candidates
  have row : edge 5503 = (SortedNatTree.contains FastRows.row5503) := by
    change Data.row5503.contains = (SortedNatTree.contains FastRows.row5503)
    exact FastRows.row5503_checked.symm
  rw [row]
  decide
theorem root19Step40 : scan edge (search edge 3) 4 [79] 79 root19Tail40 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail41 winner := by
  rw [root19Tail40, scan, if_neg (by decide), root19Candidates40_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail41
    (if 3 ≤ root19Candidates40.length then search edge 3 [79, 5503] root19Candidates40 5582 winner else winner) = _
  rw [if_neg (by decide)]
def root19Candidates41 : List Nat := [5653, 7561, 11551, 17467, 18793]
set_option maxRecDepth 8192 in
theorem root19Candidates41_checked : candidates edge 4 79 5581 root19Tail42 winner.bound = root19Candidates41 := by
  unfold candidates
  have row : edge 5581 = (SortedNatTree.contains FastRows.row5581) := by
    change Data.row5581.contains = (SortedNatTree.contains FastRows.row5581)
    exact FastRows.row5581_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root19Child41 : search edge 3 [79, 5581] root19Candidates41 5660 winner = winner := by decide
theorem root19Step41 : scan edge (search edge 3) 4 [79] 79 root19Tail41 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail42 winner := by
  rw [root19Tail41, scan, if_neg (by decide), root19Candidates41_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail42
    (if 3 ≤ root19Candidates41.length then search edge 3 [79, 5581] root19Candidates41 5660 winner else winner) = _
  rw [if_pos (by decide), root19Child41]
def root19Candidates42 : List Nat := [5659, 6373, 14779, 17467]
set_option maxRecDepth 8192 in
theorem root19Candidates42_checked : candidates edge 4 79 5647 root19Tail43 winner.bound = root19Candidates42 := by
  unfold candidates
  have row : edge 5647 = (SortedNatTree.contains FastRows.row5647) := by
    change Data.row5647.contains = (SortedNatTree.contains FastRows.row5647)
    exact FastRows.row5647_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root19Child42 : search edge 3 [79, 5647] root19Candidates42 5726 winner = winner := by decide
theorem root19Step42 : scan edge (search edge 3) 4 [79] 79 root19Tail42 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail43 winner := by
  rw [root19Tail42, scan, if_neg (by decide), root19Candidates42_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail43
    (if 3 ≤ root19Candidates42.length then search edge 3 [79, 5647] root19Candidates42 5726 winner else winner) = _
  rw [if_pos (by decide), root19Child42]
def root19Candidates43 : List Nat := [7549, 20149]
set_option maxRecDepth 8192 in
theorem root19Candidates43_checked : candidates edge 4 79 5653 root19Tail44 winner.bound = root19Candidates43 := by
  unfold candidates
  have row : edge 5653 = (SortedNatTree.contains FastRows.row5653) := by
    change Data.row5653.contains = (SortedNatTree.contains FastRows.row5653)
    exact FastRows.row5653_checked.symm
  rw [row]
  decide
theorem root19Step43 : scan edge (search edge 3) 4 [79] 79 root19Tail43 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail44 winner := by
  rw [root19Tail43, scan, if_neg (by decide), root19Candidates43_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail44
    (if 3 ≤ root19Candidates43.length then search edge 3 [79, 5653] root19Candidates43 5732 winner else winner) = _
  rw [if_neg (by decide)]
def root19Candidates44 : List Nat := [8461, 17467, 19609]
set_option maxRecDepth 8192 in
theorem root19Candidates44_checked : candidates edge 4 79 5659 root19Tail45 winner.bound = root19Candidates44 := by
  unfold candidates
  have row : edge 5659 = (SortedNatTree.contains FastRows.row5659) := by
    change Data.row5659.contains = (SortedNatTree.contains FastRows.row5659)
    exact FastRows.row5659_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root19Child44 : search edge 3 [79, 5659] root19Candidates44 5738 winner = winner := by decide
theorem root19Step44 : scan edge (search edge 3) 4 [79] 79 root19Tail44 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail45 winner := by
  rw [root19Tail44, scan, if_neg (by decide), root19Candidates44_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail45
    (if 3 ≤ root19Candidates44.length then search edge 3 [79, 5659] root19Candidates44 5738 winner else winner) = _
  rw [if_pos (by decide), root19Child44]
def root19Candidates45 : List Nat := [11551, 18517]
set_option maxRecDepth 8192 in
theorem root19Candidates45_checked : candidates edge 4 79 5791 root19Tail46 winner.bound = root19Candidates45 := by
  unfold candidates
  have row : edge 5791 = (SortedNatTree.contains FastRows.row5791) := by
    change Data.row5791.contains = (SortedNatTree.contains FastRows.row5791)
    exact FastRows.row5791_checked.symm
  rw [row]
  decide
theorem root19Step45 : scan edge (search edge 3) 4 [79] 79 root19Tail45 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail46 winner := by
  rw [root19Tail45, scan, if_neg (by decide), root19Candidates45_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail46
    (if 3 ≤ root19Candidates45.length then search edge 3 [79, 5791] root19Candidates45 5870 winner else winner) = _
  rw [if_neg (by decide)]
def root19Candidates46 : List Nat := []
set_option maxRecDepth 8192 in
theorem root19Candidates46_checked : candidates edge 4 79 6091 root19Tail47 winner.bound = root19Candidates46 := by
  unfold candidates
  have row : edge 6091 = (SortedNatTree.contains FastRows.row6091) := by
    change Data.row6091.contains = (SortedNatTree.contains FastRows.row6091)
    exact FastRows.row6091_checked.symm
  rw [row]
  decide
theorem root19Step46 : scan edge (search edge 3) 4 [79] 79 root19Tail46 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail47 winner := by
  rw [root19Tail46, scan, if_neg (by decide), root19Candidates46_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail47
    (if 3 ≤ root19Candidates46.length then search edge 3 [79, 6091] root19Candidates46 6170 winner else winner) = _
  rw [if_neg (by decide)]
def root19Candidates47 : List Nat := [7933, 18637]
set_option maxRecDepth 8192 in
theorem root19Candidates47_checked : candidates edge 4 79 6217 root19Tail48 winner.bound = root19Candidates47 := by
  unfold candidates
  have row : edge 6217 = (SortedNatTree.contains FastRows.row6217) := by
    change Data.row6217.contains = (SortedNatTree.contains FastRows.row6217)
    exact FastRows.row6217_checked.symm
  rw [row]
  decide
theorem root19Step47 : scan edge (search edge 3) 4 [79] 79 root19Tail47 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail48 winner := by
  rw [root19Tail47, scan, if_neg (by decide), root19Candidates47_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail48
    (if 3 ≤ root19Candidates47.length then search edge 3 [79, 6217] root19Candidates47 6296 winner else winner) = _
  rw [if_neg (by decide)]
def root19Candidates48 : List Nat := [6781]
set_option maxRecDepth 8192 in
theorem root19Candidates48_checked : candidates edge 4 79 6373 root19Tail49 winner.bound = root19Candidates48 := by
  unfold candidates
  have row : edge 6373 = (SortedNatTree.contains FastRows.row6373) := by
    change Data.row6373.contains = (SortedNatTree.contains FastRows.row6373)
    exact FastRows.row6373_checked.symm
  rw [row]
  decide
theorem root19Step48 : scan edge (search edge 3) 4 [79] 79 root19Tail48 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail49 winner := by
  rw [root19Tail48, scan, if_neg (by decide), root19Candidates48_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail49
    (if 3 ≤ root19Candidates48.length then search edge 3 [79, 6373] root19Candidates48 6452 winner else winner) = _
  rw [if_neg (by decide)]
def root19Candidates49 : List Nat := [7561]
set_option maxRecDepth 8192 in
theorem root19Candidates49_checked : candidates edge 4 79 6451 root19Tail50 winner.bound = root19Candidates49 := by
  unfold candidates
  have row : edge 6451 = (SortedNatTree.contains FastRows.row6451) := by
    change Data.row6451.contains = (SortedNatTree.contains FastRows.row6451)
    exact FastRows.row6451_checked.symm
  rw [row]
  decide
theorem root19Step49 : scan edge (search edge 3) 4 [79] 79 root19Tail49 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail50 winner := by
  rw [root19Tail49, scan, if_neg (by decide), root19Candidates49_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail50
    (if 3 ≤ root19Candidates49.length then search edge 3 [79, 6451] root19Candidates49 6530 winner else winner) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
