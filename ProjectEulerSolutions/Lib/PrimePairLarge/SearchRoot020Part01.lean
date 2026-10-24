import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot020Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root20Candidates32 : List Nat := [9437, 14897, 19391, 21503]
set_option maxRecDepth 8192 in
theorem root20Candidates32_checked : candidates edge 4 83 3557 root20Tail33 winner.bound = root20Candidates32 := by
  unfold candidates
  have row : edge 3557 = (SortedNatTree.contains FastRows.row3557) := by
    change Data.row3557.contains = (SortedNatTree.contains FastRows.row3557)
    exact FastRows.row3557_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root20Child32 : search edge 3 [83, 3557] root20Candidates32 3640 winner = winner := by decide
theorem root20Step32 : scan edge (search edge 3) 4 [83] 83 root20Tail32 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail33 winner := by
  rw [root20Tail32, scan, if_neg (by decide), root20Candidates32_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail33
    (if 3 ≤ root20Candidates32.length then search edge 3 [83, 3557] root20Candidates32 3640 winner else winner) = _
  rw [if_pos (by decide), root20Child32]
def root20Candidates33 : List Nat := [4643, 9473, 14897]
set_option maxRecDepth 8192 in
theorem root20Candidates33_checked : candidates edge 4 83 3659 root20Tail34 winner.bound = root20Candidates33 := by
  unfold candidates
  have row : edge 3659 = (SortedNatTree.contains FastRows.row3659) := by
    change Data.row3659.contains = (SortedNatTree.contains FastRows.row3659)
    exact FastRows.row3659_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root20Child33 : search edge 3 [83, 3659] root20Candidates33 3742 winner = winner := by decide
theorem root20Step33 : scan edge (search edge 3) 4 [83] 83 root20Tail33 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail34 winner := by
  rw [root20Tail33, scan, if_neg (by decide), root20Candidates33_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail34
    (if 3 ≤ root20Candidates33.length then search edge 3 [83, 3659] root20Candidates33 3742 winner else winner) = _
  rw [if_pos (by decide), root20Child33]
def root20Candidates34 : List Nat := []
set_option maxRecDepth 8192 in
theorem root20Candidates34_checked : candidates edge 4 83 3923 root20Tail35 winner.bound = root20Candidates34 := by
  unfold candidates
  have row : edge 3923 = (SortedNatTree.contains FastRows.row3923) := by
    change Data.row3923.contains = (SortedNatTree.contains FastRows.row3923)
    exact FastRows.row3923_checked.symm
  rw [row]
  decide
theorem root20Step34 : scan edge (search edge 3) 4 [83] 83 root20Tail34 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail35 winner := by
  rw [root20Tail34, scan, if_neg (by decide), root20Candidates34_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail35
    (if 3 ≤ root20Candidates34.length then search edge 3 [83, 3923] root20Candidates34 4006 winner else winner) = _
  rw [if_neg (by decide)]
def root20Candidates35 : List Nat := [5717, 9161, 11909]
set_option maxRecDepth 8192 in
theorem root20Candidates35_checked : candidates edge 4 83 4643 root20Tail36 winner.bound = root20Candidates35 := by
  unfold candidates
  have row : edge 4643 = (SortedNatTree.contains FastRows.row4643) := by
    change Data.row4643.contains = (SortedNatTree.contains FastRows.row4643)
    exact FastRows.row4643_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root20Child35 : search edge 3 [83, 4643] root20Candidates35 4726 winner = winner := by decide
theorem root20Step35 : scan edge (search edge 3) 4 [83] 83 root20Tail35 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail36 winner := by
  rw [root20Tail35, scan, if_neg (by decide), root20Candidates35_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail36
    (if 3 ≤ root20Candidates35.length then search edge 3 [83, 4643] root20Candidates35 4726 winner else winner) = _
  rw [if_pos (by decide), root20Child35]
def root20Candidates36 : List Nat := [8093, 9539, 14897, 20177]
set_option maxRecDepth 8192 in
theorem root20Candidates36_checked : candidates edge 4 83 5039 root20Tail37 winner.bound = root20Candidates36 := by
  unfold candidates
  have row : edge 5039 = (SortedNatTree.contains FastRows.row5039) := by
    change Data.row5039.contains = (SortedNatTree.contains FastRows.row5039)
    exact FastRows.row5039_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root20Child36 : search edge 3 [83, 5039] root20Candidates36 5122 winner = winner := by decide
theorem root20Step36 : scan edge (search edge 3) 4 [83] 83 root20Tail36 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail37 winner := by
  rw [root20Tail36, scan, if_neg (by decide), root20Candidates36_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail37
    (if 3 ≤ root20Candidates36.length then search edge 3 [83, 5039] root20Candidates36 5122 winner else winner) = _
  rw [if_pos (by decide), root20Child36]
def root20Candidates37 : List Nat := [5717]
set_option maxRecDepth 8192 in
theorem root20Candidates37_checked : candidates edge 4 83 5441 root20Tail38 winner.bound = root20Candidates37 := by
  unfold candidates
  have row : edge 5441 = (SortedNatTree.contains FastRows.row5441) := by
    change Data.row5441.contains = (SortedNatTree.contains FastRows.row5441)
    exact FastRows.row5441_checked.symm
  rw [row]
  decide
theorem root20Step37 : scan edge (search edge 3) 4 [83] 83 root20Tail37 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail38 winner := by
  rw [root20Tail37, scan, if_neg (by decide), root20Candidates37_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail38
    (if 3 ≤ root20Candidates37.length then search edge 3 [83, 5441] root20Candidates37 5524 winner else winner) = _
  rw [if_neg (by decide)]
def root20Candidates38 : List Nat := [8969, 9887, 11279, 15731]
set_option maxRecDepth 8192 in
theorem root20Candidates38_checked : candidates edge 4 83 5591 root20Tail39 winner.bound = root20Candidates38 := by
  unfold candidates
  have row : edge 5591 = (SortedNatTree.contains FastRows.row5591) := by
    change Data.row5591.contains = (SortedNatTree.contains FastRows.row5591)
    exact FastRows.row5591_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root20Child38 : search edge 3 [83, 5591] root20Candidates38 5674 winner = winner := by decide
theorem root20Step38 : scan edge (search edge 3) 4 [83] 83 root20Tail38 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail39 winner := by
  rw [root20Tail38, scan, if_neg (by decide), root20Candidates38_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail39
    (if 3 ≤ root20Candidates38.length then search edge 3 [83, 5591] root20Candidates38 5674 winner else winner) = _
  rw [if_pos (by decide), root20Child38]
def root20Candidates39 : List Nat := [7079, 8171, 9887]
set_option maxRecDepth 8192 in
theorem root20Candidates39_checked : candidates edge 4 83 5717 root20Tail40 winner.bound = root20Candidates39 := by
  unfold candidates
  have row : edge 5717 = (SortedNatTree.contains FastRows.row5717) := by
    change Data.row5717.contains = (SortedNatTree.contains FastRows.row5717)
    exact FastRows.row5717_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root20Child39 : search edge 3 [83, 5717] root20Candidates39 5800 winner = winner := by decide
theorem root20Step39 : scan edge (search edge 3) 4 [83] 83 root20Tail39 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail40 winner := by
  rw [root20Tail39, scan, if_neg (by decide), root20Candidates39_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail40
    (if 3 ≤ root20Candidates39.length then search edge 3 [83, 5717] root20Candidates39 5800 winner else winner) = _
  rw [if_pos (by decide), root20Child39]
def root20Candidates40 : List Nat := [11279, 12197, 19841]
set_option maxRecDepth 8192 in
theorem root20Candidates40_checked : candidates edge 4 83 5897 root20Tail41 winner.bound = root20Candidates40 := by
  unfold candidates
  have row : edge 5897 = (SortedNatTree.contains FastRows.row5897) := by
    change Data.row5897.contains = (SortedNatTree.contains FastRows.row5897)
    exact FastRows.row5897_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root20Child40 : search edge 3 [83, 5897] root20Candidates40 5980 winner = winner := by decide
theorem root20Step40 : scan edge (search edge 3) 4 [83] 83 root20Tail40 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail41 winner := by
  rw [root20Tail40, scan, if_neg (by decide), root20Candidates40_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail41
    (if 3 ≤ root20Candidates40.length then search edge 3 [83, 5897] root20Candidates40 5980 winner else winner) = _
  rw [if_pos (by decide), root20Child40]
def root20Candidates41 : List Nat := [6491, 11447]
set_option maxRecDepth 8192 in
theorem root20Candidates41_checked : candidates edge 4 83 5987 root20Tail42 winner.bound = root20Candidates41 := by
  unfold candidates
  have row : edge 5987 = (SortedNatTree.contains FastRows.row5987) := by
    change Data.row5987.contains = (SortedNatTree.contains FastRows.row5987)
    exact FastRows.row5987_checked.symm
  rw [row]
  decide
theorem root20Step41 : scan edge (search edge 3) 4 [83] 83 root20Tail41 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail42 winner := by
  rw [root20Tail41, scan, if_neg (by decide), root20Candidates41_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail42
    (if 3 ≤ root20Candidates41.length then search edge 3 [83, 5987] root20Candidates41 6070 winner else winner) = _
  rw [if_neg (by decide)]
def root20Candidates42 : List Nat := [9473, 18329]
set_option maxRecDepth 8192 in
theorem root20Candidates42_checked : candidates edge 4 83 6203 root20Tail43 winner.bound = root20Candidates42 := by
  unfold candidates
  have row : edge 6203 = (SortedNatTree.contains FastRows.row6203) := by
    change Data.row6203.contains = (SortedNatTree.contains FastRows.row6203)
    exact FastRows.row6203_checked.symm
  rw [row]
  decide
theorem root20Step42 : scan edge (search edge 3) 4 [83] 83 root20Tail42 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail43 winner := by
  rw [root20Tail42, scan, if_neg (by decide), root20Candidates42_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail43
    (if 3 ≤ root20Candidates42.length then search edge 3 [83, 6203] root20Candidates42 6286 winner else winner) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
