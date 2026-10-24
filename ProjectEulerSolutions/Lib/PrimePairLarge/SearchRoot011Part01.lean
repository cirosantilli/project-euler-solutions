import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot011Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root11Candidates32 : List Nat := [5857, 9349, 12073, 12157, 14551]
set_option maxRecDepth 8192 in
theorem root11Candidates32_checked : candidates edge 4 43 4111 root11Tail33 winner.bound = root11Candidates32 := by
  unfold candidates
  have row : edge 4111 = (SortedNatTree.contains FastRows.row4111) := by
    change Data.row4111.contains = (SortedNatTree.contains FastRows.row4111)
    exact FastRows.row4111_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root11Child32 : search edge 3 [43, 4111] root11Candidates32 4154 winner = winner := by decide
theorem root11Step32 : scan edge (search edge 3) 4 [43] 43 root11Tail32 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail33 winner := by
  rw [root11Tail32, scan, if_neg (by decide), root11Candidates32_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail33
    (if 3 ≤ root11Candidates32.length then search edge 3 [43, 4111] root11Candidates32 4154 winner else winner) = _
  rw [if_pos (by decide), root11Child32]
def root11Candidates33 : List Nat := []
set_option maxRecDepth 8192 in
theorem root11Candidates33_checked : candidates edge 4 43 4243 root11Tail34 winner.bound = root11Candidates33 := by
  unfold candidates
  have row : edge 4243 = (SortedNatTree.contains FastRows.row4243) := by
    change Data.row4243.contains = (SortedNatTree.contains FastRows.row4243)
    exact FastRows.row4243_checked.symm
  rw [row]
  decide
theorem root11Step33 : scan edge (search edge 3) 4 [43] 43 root11Tail33 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail34 winner := by
  rw [root11Tail33, scan, if_neg (by decide), root11Candidates33_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail34
    (if 3 ≤ root11Candidates33.length then search edge 3 [43, 4243] root11Candidates33 4286 winner else winner) = _
  rw [if_neg (by decide)]
def root11Candidates34 : List Nat := [4813, 5653]
set_option maxRecDepth 8192 in
theorem root11Candidates34_checked : candidates edge 4 43 4363 root11Tail35 winner.bound = root11Candidates34 := by
  unfold candidates
  have row : edge 4363 = (SortedNatTree.contains FastRows.row4363) := by
    change Data.row4363.contains = (SortedNatTree.contains FastRows.row4363)
    exact FastRows.row4363_checked.symm
  rw [row]
  decide
theorem root11Step34 : scan edge (search edge 3) 4 [43] 43 root11Tail34 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail35 winner := by
  rw [root11Tail34, scan, if_neg (by decide), root11Candidates34_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail35
    (if 3 ≤ root11Candidates34.length then search edge 3 [43, 4363] root11Candidates34 4406 winner else winner) = _
  rw [if_neg (by decide)]
def root11Candidates35 : List Nat := [9787, 18661]
set_option maxRecDepth 8192 in
theorem root11Candidates35_checked : candidates edge 4 43 4783 root11Tail36 winner.bound = root11Candidates35 := by
  unfold candidates
  have row : edge 4783 = (SortedNatTree.contains FastRows.row4783) := by
    change Data.row4783.contains = (SortedNatTree.contains FastRows.row4783)
    exact FastRows.row4783_checked.symm
  rw [row]
  decide
theorem root11Step35 : scan edge (search edge 3) 4 [43] 43 root11Tail35 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail36 winner := by
  rw [root11Tail35, scan, if_neg (by decide), root11Candidates35_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail36
    (if 3 ≤ root11Candidates35.length then search edge 3 [43, 4783] root11Candidates35 4826 winner else winner) = _
  rw [if_neg (by decide)]
def root11Candidates36 : List Nat := [7351]
set_option maxRecDepth 8192 in
theorem root11Candidates36_checked : candidates edge 4 43 4813 root11Tail37 winner.bound = root11Candidates36 := by
  unfold candidates
  have row : edge 4813 = (SortedNatTree.contains FastRows.row4813) := by
    change Data.row4813.contains = (SortedNatTree.contains FastRows.row4813)
    exact FastRows.row4813_checked.symm
  rw [row]
  decide
theorem root11Step36 : scan edge (search edge 3) 4 [43] 43 root11Tail36 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail37 winner := by
  rw [root11Tail36, scan, if_neg (by decide), root11Candidates36_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail37
    (if 3 ≤ root11Candidates36.length then search edge 3 [43, 4813] root11Candidates36 4856 winner else winner) = _
  rw [if_neg (by decide)]
def root11Candidates37 : List Nat := [8623, 12157, 12703]
set_option maxRecDepth 8192 in
theorem root11Candidates37_checked : candidates edge 4 43 5563 root11Tail38 winner.bound = root11Candidates37 := by
  unfold candidates
  have row : edge 5563 = (SortedNatTree.contains FastRows.row5563) := by
    change Data.row5563.contains = (SortedNatTree.contains FastRows.row5563)
    exact FastRows.row5563_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root11Child37 : search edge 3 [43, 5563] root11Candidates37 5606 winner = winner := by decide
theorem root11Step37 : scan edge (search edge 3) 4 [43] 43 root11Tail37 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail38 winner := by
  rw [root11Tail37, scan, if_neg (by decide), root11Candidates37_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail38
    (if 3 ≤ root11Candidates37.length then search edge 3 [43, 5563] root11Candidates37 5606 winner else winner) = _
  rw [if_pos (by decide), root11Child37]
def root11Candidates38 : List Nat := [5923, 8521]
set_option maxRecDepth 8192 in
theorem root11Candidates38_checked : candidates edge 4 43 5569 root11Tail39 winner.bound = root11Candidates38 := by
  unfold candidates
  have row : edge 5569 = (SortedNatTree.contains FastRows.row5569) := by
    change Data.row5569.contains = (SortedNatTree.contains FastRows.row5569)
    exact FastRows.row5569_checked.symm
  rw [row]
  decide
theorem root11Step38 : scan edge (search edge 3) 4 [43] 43 root11Tail38 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail39 winner := by
  rw [root11Tail38, scan, if_neg (by decide), root11Candidates38_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail39
    (if 3 ≤ root11Candidates38.length then search edge 3 [43, 5569] root11Candidates38 5612 winner else winner) = _
  rw [if_neg (by decide)]
def root11Candidates39 : List Nat := [5923, 13921, 15187]
set_option maxRecDepth 8192 in
theorem root11Candidates39_checked : candidates edge 4 43 5653 root11Tail40 winner.bound = root11Candidates39 := by
  unfold candidates
  have row : edge 5653 = (SortedNatTree.contains FastRows.row5653) := by
    change Data.row5653.contains = (SortedNatTree.contains FastRows.row5653)
    exact FastRows.row5653_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root11Child39 : search edge 3 [43, 5653] root11Candidates39 5696 winner = winner := by decide
theorem root11Step39 : scan edge (search edge 3) 4 [43] 43 root11Tail39 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail40 winner := by
  rw [root11Tail39, scan, if_neg (by decide), root11Candidates39_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail40
    (if 3 ≤ root11Candidates39.length then search edge 3 [43, 5653] root11Candidates39 5696 winner else winner) = _
  rw [if_pos (by decide), root11Child39]
def root11Candidates40 : List Nat := [9007]
set_option maxRecDepth 8192 in
theorem root11Candidates40_checked : candidates edge 4 43 5857 root11Tail41 winner.bound = root11Candidates40 := by
  unfold candidates
  have row : edge 5857 = (SortedNatTree.contains FastRows.row5857) := by
    change Data.row5857.contains = (SortedNatTree.contains FastRows.row5857)
    exact FastRows.row5857_checked.symm
  rw [row]
  decide
theorem root11Step40 : scan edge (search edge 3) 4 [43] 43 root11Tail40 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail41 winner := by
  rw [root11Tail40, scan, if_neg (by decide), root11Candidates40_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail41
    (if 3 ≤ root11Candidates40.length then search edge 3 [43, 5857] root11Candidates40 5900 winner else winner) = _
  rw [if_neg (by decide)]
def root11Candidates41 : List Nat := [8623]
set_option maxRecDepth 8192 in
theorem root11Candidates41_checked : candidates edge 4 43 5923 root11Tail42 winner.bound = root11Candidates41 := by
  unfold candidates
  have row : edge 5923 = (SortedNatTree.contains FastRows.row5923) := by
    change Data.row5923.contains = (SortedNatTree.contains FastRows.row5923)
    exact FastRows.row5923_checked.symm
  rw [row]
  decide
theorem root11Step41 : scan edge (search edge 3) 4 [43] 43 root11Tail41 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail42 winner := by
  rw [root11Tail41, scan, if_neg (by decide), root11Candidates41_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail42
    (if 3 ≤ root11Candidates41.length then search edge 3 [43, 5923] root11Candidates41 5966 winner else winner) = _
  rw [if_neg (by decide)]
def root11Candidates42 : List Nat := []
set_option maxRecDepth 8192 in
theorem root11Candidates42_checked : candidates edge 4 43 6091 root11Tail43 winner.bound = root11Candidates42 := by
  unfold candidates
  have row : edge 6091 = (SortedNatTree.contains FastRows.row6091) := by
    change Data.row6091.contains = (SortedNatTree.contains FastRows.row6091)
    exact FastRows.row6091_checked.symm
  rw [row]
  decide
theorem root11Step42 : scan edge (search edge 3) 4 [43] 43 root11Tail42 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail43 winner := by
  rw [root11Tail42, scan, if_neg (by decide), root11Candidates42_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail43
    (if 3 ≤ root11Candidates42.length then search edge 3 [43, 6091] root11Candidates42 6134 winner else winner) = _
  rw [if_neg (by decide)]
def root11Candidates43 : List Nat := [9697]
set_option maxRecDepth 8192 in
theorem root11Candidates43_checked : candidates edge 4 43 6343 root11Tail44 winner.bound = root11Candidates43 := by
  unfold candidates
  have row : edge 6343 = (SortedNatTree.contains FastRows.row6343) := by
    change Data.row6343.contains = (SortedNatTree.contains FastRows.row6343)
    exact FastRows.row6343_checked.symm
  rw [row]
  decide
theorem root11Step43 : scan edge (search edge 3) 4 [43] 43 root11Tail43 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail44 winner := by
  rw [root11Tail43, scan, if_neg (by decide), root11Candidates43_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail44
    (if 3 ≤ root11Candidates43.length then search edge 3 [43, 6343] root11Candidates43 6386 winner else winner) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
