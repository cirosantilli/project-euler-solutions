import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root246Candidates : List Nat := Data.row1579
set_option maxRecDepth 8192 in
theorem root246Candidates_checked : candidates edge 5 0 1579 tail247 winner.bound = root246Candidates := by
  apply candidates_eq_row edge 5 0 1579 tail247 root246Candidates winner.bound tail247_ordered
  · decide
  · decide
  · funext q
    rfl
def root246Tail36 : List Nat := []
def root246Tail35 : List Nat := 24379 :: root246Tail36
def root246Tail34 : List Nat := 23977 :: root246Tail35
def root246Tail33 : List Nat := 23131 :: root246Tail34
def root246Tail32 : List Nat := 22123 :: root246Tail33
def root246Tail31 : List Nat := 21817 :: root246Tail32
def root246Tail30 : List Nat := 21649 :: root246Tail31
def root246Tail29 : List Nat := 21523 :: root246Tail30
def root246Tail28 : List Nat := 20743 :: root246Tail29
def root246Tail27 : List Nat := 20011 :: root246Tail28
def root246Tail26 : List Nat := 19213 :: root246Tail27
def root246Tail25 : List Nat := 18121 :: root246Tail26
def root246Tail24 : List Nat := 17167 :: root246Tail25
def root246Tail23 : List Nat := 16987 :: root246Tail24
def root246Tail22 : List Nat := 16069 :: root246Tail23
def root246Tail21 : List Nat := 15913 :: root246Tail22
def root246Tail20 : List Nat := 13177 :: root246Tail21
def root246Tail19 : List Nat := 12619 :: root246Tail20
def root246Tail18 : List Nat := 11833 :: root246Tail19
def root246Tail17 : List Nat := 11701 :: root246Tail18
def root246Tail16 : List Nat := 11617 :: root246Tail17
def root246Tail15 : List Nat := 11287 :: root246Tail16
def root246Tail14 : List Nat := 11251 :: root246Tail15
def root246Tail13 : List Nat := 10069 :: root246Tail14
def root246Tail12 : List Nat := 9859 :: root246Tail13
def root246Tail11 : List Nat := 9349 :: root246Tail12
def root246Tail10 : List Nat := 9109 :: root246Tail11
def root246Tail9 : List Nat := 8059 :: root246Tail10
def root246Tail8 : List Nat := 7723 :: root246Tail9
def root246Tail7 : List Nat := 7477 :: root246Tail8
def root246Tail6 : List Nat := 6301 :: root246Tail7
def root246Tail5 : List Nat := 6073 :: root246Tail6
def root246Tail4 : List Nat := 4783 :: root246Tail5
def root246Tail3 : List Nat := 3907 :: root246Tail4
def root246Tail2 : List Nat := 3301 :: root246Tail3
def root246Tail1 : List Nat := 1669 :: root246Tail2
def root246Tail0 : List Nat := 1627 :: root246Tail1
def root246Candidates0 : List Nat := [11833]
set_option maxRecDepth 8192 in
theorem root246Candidates0_checked : candidates edge 4 1579 1627 root246Tail1 winner.bound = root246Candidates0 := by
  unfold candidates
  have row : edge 1627 = (SortedNatTree.contains FastRows.row1627) := by
    change Data.row1627.contains = (SortedNatTree.contains FastRows.row1627)
    exact FastRows.row1627_checked.symm
  rw [row]
  decide
theorem root246Step0 : scan edge (search edge 3) 4 [1579] 1579 root246Tail0 winner =
    scan edge (search edge 3) 4 [1579] 1579 root246Tail1 winner := by
  rw [root246Tail0, scan, if_neg (by decide), root246Candidates0_checked]
  change scan edge (search edge 3) 4 [1579] 1579 root246Tail1
    (if 3 ≤ root246Candidates0.length then search edge 3 [1579, 1627] root246Candidates0 3206 winner else winner) = _
  rw [if_neg (by decide)]
def root246Candidates1 : List Nat := [3301, 19213]
set_option maxRecDepth 8192 in
theorem root246Candidates1_checked : candidates edge 4 1579 1669 root246Tail2 winner.bound = root246Candidates1 := by
  unfold candidates
  have row : edge 1669 = (SortedNatTree.contains FastRows.row1669) := by
    change Data.row1669.contains = (SortedNatTree.contains FastRows.row1669)
    exact FastRows.row1669_checked.symm
  rw [row]
  decide
theorem root246Step1 : scan edge (search edge 3) 4 [1579] 1579 root246Tail1 winner =
    scan edge (search edge 3) 4 [1579] 1579 root246Tail2 winner := by
  rw [root246Tail1, scan, if_neg (by decide), root246Candidates1_checked]
  change scan edge (search edge 3) 4 [1579] 1579 root246Tail2
    (if 3 ≤ root246Candidates1.length then search edge 3 [1579, 1669] root246Candidates1 3248 winner else winner) = _
  rw [if_neg (by decide)]
def root246Candidates2 : List Nat := [20743]
set_option maxRecDepth 8192 in
theorem root246Candidates2_checked : candidates edge 4 1579 3301 root246Tail3 winner.bound = root246Candidates2 := by
  unfold candidates
  have row : edge 3301 = (SortedNatTree.contains FastRows.row3301) := by
    change Data.row3301.contains = (SortedNatTree.contains FastRows.row3301)
    exact FastRows.row3301_checked.symm
  rw [row]
  decide
theorem root246Step2 : scan edge (search edge 3) 4 [1579] 1579 root246Tail2 winner =
    scan edge (search edge 3) 4 [1579] 1579 root246Tail3 winner := by
  rw [root246Tail2, scan, if_neg (by decide), root246Candidates2_checked]
  change scan edge (search edge 3) 4 [1579] 1579 root246Tail3
    (if 3 ≤ root246Candidates2.length then search edge 3 [1579, 3301] root246Candidates2 4880 winner else winner) = _
  rw [if_neg (by decide)]
def root246Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root246Candidates3_checked : candidates edge 4 1579 3907 root246Tail4 winner.bound = root246Candidates3 := by
  unfold candidates
  have row : edge 3907 = (SortedNatTree.contains FastRows.row3907) := by
    change Data.row3907.contains = (SortedNatTree.contains FastRows.row3907)
    exact FastRows.row3907_checked.symm
  rw [row]
  decide
theorem root246Step3 : scan edge (search edge 3) 4 [1579] 1579 root246Tail3 winner =
    scan edge (search edge 3) 4 [1579] 1579 root246Tail4 winner := by
  rw [root246Tail3, scan, if_neg (by decide), root246Candidates3_checked]
  change scan edge (search edge 3) 4 [1579] 1579 root246Tail4
    (if 3 ≤ root246Candidates3.length then search edge 3 [1579, 3907] root246Candidates3 5486 winner else winner) = _
  rw [if_neg (by decide)]
def root246Candidates4 : List Nat := [6073]
set_option maxRecDepth 8192 in
theorem root246Candidates4_checked : candidates edge 4 1579 4783 root246Tail5 winner.bound = root246Candidates4 := by
  unfold candidates
  have row : edge 4783 = (SortedNatTree.contains FastRows.row4783) := by
    change Data.row4783.contains = (SortedNatTree.contains FastRows.row4783)
    exact FastRows.row4783_checked.symm
  rw [row]
  decide
theorem root246Step4 : scan edge (search edge 3) 4 [1579] 1579 root246Tail4 winner =
    scan edge (search edge 3) 4 [1579] 1579 root246Tail5 winner := by
  rw [root246Tail4, scan, if_neg (by decide), root246Candidates4_checked]
  change scan edge (search edge 3) 4 [1579] 1579 root246Tail5
    (if 3 ≤ root246Candidates4.length then search edge 3 [1579, 4783] root246Candidates4 6362 winner else winner) = _
  rw [if_neg (by decide)]
def root246Candidates5 : List Nat := [7723]
set_option maxRecDepth 8192 in
theorem root246Candidates5_checked : candidates edge 4 1579 6073 root246Tail6 winner.bound = root246Candidates5 := by
  unfold candidates
  have row : edge 6073 = (SortedNatTree.contains FastRows.row6073) := by
    change Data.row6073.contains = (SortedNatTree.contains FastRows.row6073)
    exact FastRows.row6073_checked.symm
  rw [row]
  decide
theorem root246Step5 : scan edge (search edge 3) 4 [1579] 1579 root246Tail5 winner =
    scan edge (search edge 3) 4 [1579] 1579 root246Tail6 winner := by
  rw [root246Tail5, scan, if_neg (by decide), root246Candidates5_checked]
  change scan edge (search edge 3) 4 [1579] 1579 root246Tail6
    (if 3 ≤ root246Candidates5.length then search edge 3 [1579, 6073] root246Candidates5 7652 winner else winner) = _
  rw [if_neg (by decide)]
theorem root246Scan6 : scan edge (search edge 3) 4 [1579] 1579 root246Tail6 winner = winner := by rw [root246Tail6, scan, if_pos (by decide)]
theorem root246Scan5 : scan edge (search edge 3) 4 [1579] 1579 root246Tail5 winner = winner :=
  (root246Step5).trans root246Scan6
theorem root246Scan4 : scan edge (search edge 3) 4 [1579] 1579 root246Tail4 winner = winner :=
  (root246Step4).trans root246Scan5
theorem root246Scan3 : scan edge (search edge 3) 4 [1579] 1579 root246Tail3 winner = winner :=
  (root246Step3).trans root246Scan4
theorem root246Scan2 : scan edge (search edge 3) 4 [1579] 1579 root246Tail2 winner = winner :=
  (root246Step2).trans root246Scan3
theorem root246Scan1 : scan edge (search edge 3) 4 [1579] 1579 root246Tail1 winner = winner :=
  (root246Step1).trans root246Scan2
theorem root246Scan0 : scan edge (search edge 3) 4 [1579] 1579 root246Tail0 winner = winner :=
  (root246Step0).trans root246Scan1
set_option maxRecDepth 4096 in
theorem root246Branch : search edge 4 [1579] root246Candidates 1579 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1579] 1579 root246Tail0 winner = winner
  exact root246Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
