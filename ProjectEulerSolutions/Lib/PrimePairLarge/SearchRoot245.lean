import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root245Candidates : List Nat := Data.row1571
set_option maxRecDepth 8192 in
theorem root245Candidates_checked : candidates edge 5 0 1571 tail246 winner.bound = root245Candidates := by
  apply candidates_eq_row edge 5 0 1571 tail246 root245Candidates winner.bound tail246_ordered
  · decide
  · decide
  · funext q
    rfl
def root245Tail35 : List Nat := []
def root245Tail34 : List Nat := 23753 :: root245Tail35
def root245Tail33 : List Nat := 23567 :: root245Tail34
def root245Tail32 : List Nat := 23459 :: root245Tail33
def root245Tail31 : List Nat := 22481 :: root245Tail32
def root245Tail30 : List Nat := 20807 :: root245Tail31
def root245Tail29 : List Nat := 20393 :: root245Tail30
def root245Tail28 : List Nat := 19889 :: root245Tail29
def root245Tail27 : List Nat := 15641 :: root245Tail28
def root245Tail26 : List Nat := 15131 :: root245Tail27
def root245Tail25 : List Nat := 14891 :: root245Tail26
def root245Tail24 : List Nat := 14879 :: root245Tail25
def root245Tail23 : List Nat := 14369 :: root245Tail24
def root245Tail22 : List Nat := 13709 :: root245Tail23
def root245Tail21 : List Nat := 13127 :: root245Tail22
def root245Tail20 : List Nat := 12983 :: root245Tail21
def root245Tail19 : List Nat := 10271 :: root245Tail20
def root245Tail18 : List Nat := 10169 :: root245Tail19
def root245Tail17 : List Nat := 9887 :: root245Tail18
def root245Tail16 : List Nat := 9257 :: root245Tail17
def root245Tail15 : List Nat := 8831 :: root245Tail16
def root245Tail14 : List Nat := 8501 :: root245Tail15
def root245Tail13 : List Nat := 7907 :: root245Tail14
def root245Tail12 : List Nat := 7529 :: root245Tail13
def root245Tail11 : List Nat := 6983 :: root245Tail12
def root245Tail10 : List Nat := 6563 :: root245Tail11
def root245Tail9 : List Nat := 5981 :: root245Tail10
def root245Tail8 : List Nat := 5879 :: root245Tail9
def root245Tail7 : List Nat := 5333 :: root245Tail8
def root245Tail6 : List Nat := 4547 :: root245Tail7
def root245Tail5 : List Nat := 4157 :: root245Tail6
def root245Tail4 : List Nat := 4133 :: root245Tail5
def root245Tail3 : List Nat := 2819 :: root245Tail4
def root245Tail2 : List Nat := 2399 :: root245Tail3
def root245Tail1 : List Nat := 1907 :: root245Tail2
def root245Tail0 : List Nat := 1607 :: root245Tail1
def root245Candidates0 : List Nat := [13127]
set_option maxRecDepth 8192 in
theorem root245Candidates0_checked : candidates edge 4 1571 1607 root245Tail1 winner.bound = root245Candidates0 := by
  unfold candidates
  have row : edge 1607 = (SortedNatTree.contains FastRows.row1607) := by
    change Data.row1607.contains = (SortedNatTree.contains FastRows.row1607)
    exact FastRows.row1607_checked.symm
  rw [row]
  decide
theorem root245Step0 : scan edge (search edge 3) 4 [1571] 1571 root245Tail0 winner =
    scan edge (search edge 3) 4 [1571] 1571 root245Tail1 winner := by
  rw [root245Tail0, scan, if_neg (by decide), root245Candidates0_checked]
  change scan edge (search edge 3) 4 [1571] 1571 root245Tail1
    (if 3 ≤ root245Candidates0.length then search edge 3 [1571, 1607] root245Candidates0 3178 winner else winner) = _
  rw [if_neg (by decide)]
def root245Candidates1 : List Nat := [14369]
set_option maxRecDepth 8192 in
theorem root245Candidates1_checked : candidates edge 4 1571 1907 root245Tail2 winner.bound = root245Candidates1 := by
  unfold candidates
  have row : edge 1907 = (SortedNatTree.contains FastRows.row1907) := by
    change Data.row1907.contains = (SortedNatTree.contains FastRows.row1907)
    exact FastRows.row1907_checked.symm
  rw [row]
  decide
theorem root245Step1 : scan edge (search edge 3) 4 [1571] 1571 root245Tail1 winner =
    scan edge (search edge 3) 4 [1571] 1571 root245Tail2 winner := by
  rw [root245Tail1, scan, if_neg (by decide), root245Candidates1_checked]
  change scan edge (search edge 3) 4 [1571] 1571 root245Tail2
    (if 3 ≤ root245Candidates1.length then search edge 3 [1571, 1907] root245Candidates1 3478 winner else winner) = _
  rw [if_neg (by decide)]
def root245Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root245Candidates2_checked : candidates edge 4 1571 2399 root245Tail3 winner.bound = root245Candidates2 := by
  unfold candidates
  have row : edge 2399 = (SortedNatTree.contains FastRows.row2399) := by
    change Data.row2399.contains = (SortedNatTree.contains FastRows.row2399)
    exact FastRows.row2399_checked.symm
  rw [row]
  decide
theorem root245Step2 : scan edge (search edge 3) 4 [1571] 1571 root245Tail2 winner =
    scan edge (search edge 3) 4 [1571] 1571 root245Tail3 winner := by
  rw [root245Tail2, scan, if_neg (by decide), root245Candidates2_checked]
  change scan edge (search edge 3) 4 [1571] 1571 root245Tail3
    (if 3 ≤ root245Candidates2.length then search edge 3 [1571, 2399] root245Candidates2 3970 winner else winner) = _
  rw [if_neg (by decide)]
def root245Candidates3 : List Nat := [5879]
set_option maxRecDepth 8192 in
theorem root245Candidates3_checked : candidates edge 4 1571 2819 root245Tail4 winner.bound = root245Candidates3 := by
  unfold candidates
  have row : edge 2819 = (SortedNatTree.contains FastRows.row2819) := by
    change Data.row2819.contains = (SortedNatTree.contains FastRows.row2819)
    exact FastRows.row2819_checked.symm
  rw [row]
  decide
theorem root245Step3 : scan edge (search edge 3) 4 [1571] 1571 root245Tail3 winner =
    scan edge (search edge 3) 4 [1571] 1571 root245Tail4 winner := by
  rw [root245Tail3, scan, if_neg (by decide), root245Candidates3_checked]
  change scan edge (search edge 3) 4 [1571] 1571 root245Tail4
    (if 3 ≤ root245Candidates3.length then search edge 3 [1571, 2819] root245Candidates3 4390 winner else winner) = _
  rw [if_neg (by decide)]
def root245Candidates4 : List Nat := [10169]
set_option maxRecDepth 8192 in
theorem root245Candidates4_checked : candidates edge 4 1571 4133 root245Tail5 winner.bound = root245Candidates4 := by
  unfold candidates
  have row : edge 4133 = (SortedNatTree.contains FastRows.row4133) := by
    change Data.row4133.contains = (SortedNatTree.contains FastRows.row4133)
    exact FastRows.row4133_checked.symm
  rw [row]
  decide
theorem root245Step4 : scan edge (search edge 3) 4 [1571] 1571 root245Tail4 winner =
    scan edge (search edge 3) 4 [1571] 1571 root245Tail5 winner := by
  rw [root245Tail4, scan, if_neg (by decide), root245Candidates4_checked]
  change scan edge (search edge 3) 4 [1571] 1571 root245Tail5
    (if 3 ≤ root245Candidates4.length then search edge 3 [1571, 4133] root245Candidates4 5704 winner else winner) = _
  rw [if_neg (by decide)]
def root245Candidates5 : List Nat := [5333]
set_option maxRecDepth 8192 in
theorem root245Candidates5_checked : candidates edge 4 1571 4157 root245Tail6 winner.bound = root245Candidates5 := by
  unfold candidates
  have row : edge 4157 = (SortedNatTree.contains FastRows.row4157) := by
    change Data.row4157.contains = (SortedNatTree.contains FastRows.row4157)
    exact FastRows.row4157_checked.symm
  rw [row]
  decide
theorem root245Step5 : scan edge (search edge 3) 4 [1571] 1571 root245Tail5 winner =
    scan edge (search edge 3) 4 [1571] 1571 root245Tail6 winner := by
  rw [root245Tail5, scan, if_neg (by decide), root245Candidates5_checked]
  change scan edge (search edge 3) 4 [1571] 1571 root245Tail6
    (if 3 ≤ root245Candidates5.length then search edge 3 [1571, 4157] root245Candidates5 5728 winner else winner) = _
  rw [if_neg (by decide)]
def root245Candidates6 : List Nat := [8831]
set_option maxRecDepth 8192 in
theorem root245Candidates6_checked : candidates edge 4 1571 4547 root245Tail7 winner.bound = root245Candidates6 := by
  unfold candidates
  have row : edge 4547 = (SortedNatTree.contains FastRows.row4547) := by
    change Data.row4547.contains = (SortedNatTree.contains FastRows.row4547)
    exact FastRows.row4547_checked.symm
  rw [row]
  decide
theorem root245Step6 : scan edge (search edge 3) 4 [1571] 1571 root245Tail6 winner =
    scan edge (search edge 3) 4 [1571] 1571 root245Tail7 winner := by
  rw [root245Tail6, scan, if_neg (by decide), root245Candidates6_checked]
  change scan edge (search edge 3) 4 [1571] 1571 root245Tail7
    (if 3 ≤ root245Candidates6.length then search edge 3 [1571, 4547] root245Candidates6 6118 winner else winner) = _
  rw [if_neg (by decide)]
def root245Candidates7 : List Nat := [9887]
set_option maxRecDepth 8192 in
theorem root245Candidates7_checked : candidates edge 4 1571 5333 root245Tail8 winner.bound = root245Candidates7 := by
  unfold candidates
  have row : edge 5333 = (SortedNatTree.contains FastRows.row5333) := by
    change Data.row5333.contains = (SortedNatTree.contains FastRows.row5333)
    exact FastRows.row5333_checked.symm
  rw [row]
  decide
theorem root245Step7 : scan edge (search edge 3) 4 [1571] 1571 root245Tail7 winner =
    scan edge (search edge 3) 4 [1571] 1571 root245Tail8 winner := by
  rw [root245Tail7, scan, if_neg (by decide), root245Candidates7_checked]
  change scan edge (search edge 3) 4 [1571] 1571 root245Tail8
    (if 3 ≤ root245Candidates7.length then search edge 3 [1571, 5333] root245Candidates7 6904 winner else winner) = _
  rw [if_neg (by decide)]
def root245Candidates8 : List Nat := [9887]
set_option maxRecDepth 8192 in
theorem root245Candidates8_checked : candidates edge 4 1571 5879 root245Tail9 winner.bound = root245Candidates8 := by
  unfold candidates
  have row : edge 5879 = (SortedNatTree.contains FastRows.row5879) := by
    change Data.row5879.contains = (SortedNatTree.contains FastRows.row5879)
    exact FastRows.row5879_checked.symm
  rw [row]
  decide
theorem root245Step8 : scan edge (search edge 3) 4 [1571] 1571 root245Tail8 winner =
    scan edge (search edge 3) 4 [1571] 1571 root245Tail9 winner := by
  rw [root245Tail8, scan, if_neg (by decide), root245Candidates8_checked]
  change scan edge (search edge 3) 4 [1571] 1571 root245Tail9
    (if 3 ≤ root245Candidates8.length then search edge 3 [1571, 5879] root245Candidates8 7450 winner else winner) = _
  rw [if_neg (by decide)]
def root245Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root245Candidates9_checked : candidates edge 4 1571 5981 root245Tail10 winner.bound = root245Candidates9 := by
  unfold candidates
  have row : edge 5981 = (SortedNatTree.contains FastRows.row5981) := by
    change Data.row5981.contains = (SortedNatTree.contains FastRows.row5981)
    exact FastRows.row5981_checked.symm
  rw [row]
  decide
theorem root245Step9 : scan edge (search edge 3) 4 [1571] 1571 root245Tail9 winner =
    scan edge (search edge 3) 4 [1571] 1571 root245Tail10 winner := by
  rw [root245Tail9, scan, if_neg (by decide), root245Candidates9_checked]
  change scan edge (search edge 3) 4 [1571] 1571 root245Tail10
    (if 3 ≤ root245Candidates9.length then search edge 3 [1571, 5981] root245Candidates9 7552 winner else winner) = _
  rw [if_neg (by decide)]
theorem root245Scan10 : scan edge (search edge 3) 4 [1571] 1571 root245Tail10 winner = winner := by rw [root245Tail10, scan, if_pos (by decide)]
theorem root245Scan9 : scan edge (search edge 3) 4 [1571] 1571 root245Tail9 winner = winner :=
  (root245Step9).trans root245Scan10
theorem root245Scan8 : scan edge (search edge 3) 4 [1571] 1571 root245Tail8 winner = winner :=
  (root245Step8).trans root245Scan9
theorem root245Scan7 : scan edge (search edge 3) 4 [1571] 1571 root245Tail7 winner = winner :=
  (root245Step7).trans root245Scan8
theorem root245Scan6 : scan edge (search edge 3) 4 [1571] 1571 root245Tail6 winner = winner :=
  (root245Step6).trans root245Scan7
theorem root245Scan5 : scan edge (search edge 3) 4 [1571] 1571 root245Tail5 winner = winner :=
  (root245Step5).trans root245Scan6
theorem root245Scan4 : scan edge (search edge 3) 4 [1571] 1571 root245Tail4 winner = winner :=
  (root245Step4).trans root245Scan5
theorem root245Scan3 : scan edge (search edge 3) 4 [1571] 1571 root245Tail3 winner = winner :=
  (root245Step3).trans root245Scan4
theorem root245Scan2 : scan edge (search edge 3) 4 [1571] 1571 root245Tail2 winner = winner :=
  (root245Step2).trans root245Scan3
theorem root245Scan1 : scan edge (search edge 3) 4 [1571] 1571 root245Tail1 winner = winner :=
  (root245Step1).trans root245Scan2
theorem root245Scan0 : scan edge (search edge 3) 4 [1571] 1571 root245Tail0 winner = winner :=
  (root245Step0).trans root245Scan1
set_option maxRecDepth 4096 in
theorem root245Branch : search edge 4 [1571] root245Candidates 1571 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1571] 1571 root245Tail0 winner = winner
  exact root245Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
