import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root266Candidates : List Nat := Data.row1723
set_option maxRecDepth 8192 in
theorem root266Candidates_checked : candidates edge 5 0 1723 tail267 winner.bound = root266Candidates := by
  apply candidates_eq_row edge 5 0 1723 tail267 root266Candidates winner.bound tail267_ordered
  · decide
  · decide
  · funext q
    rfl
def root266Tail52 : List Nat := []
def root266Tail51 : List Nat := 24049 :: root266Tail52
def root266Tail50 : List Nat := 22483 :: root266Tail51
def root266Tail49 : List Nat := 22441 :: root266Tail50
def root266Tail48 : List Nat := 21997 :: root266Tail49
def root266Tail47 : List Nat := 21991 :: root266Tail48
def root266Tail46 : List Nat := 20173 :: root266Tail47
def root266Tail45 : List Nat := 19501 :: root266Tail46
def root266Tail44 : List Nat := 18637 :: root266Tail45
def root266Tail43 : List Nat := 18301 :: root266Tail44
def root266Tail42 : List Nat := 17377 :: root266Tail43
def root266Tail41 : List Nat := 16843 :: root266Tail42
def root266Tail40 : List Nat := 16741 :: root266Tail41
def root266Tail39 : List Nat := 16699 :: root266Tail40
def root266Tail38 : List Nat := 15607 :: root266Tail39
def root266Tail37 : List Nat := 15289 :: root266Tail38
def root266Tail36 : List Nat := 15271 :: root266Tail37
def root266Tail35 : List Nat := 14767 :: root266Tail36
def root266Tail34 : List Nat := 13591 :: root266Tail35
def root266Tail33 : List Nat := 12433 :: root266Tail34
def root266Tail32 : List Nat := 12211 :: root266Tail33
def root266Tail31 : List Nat := 11467 :: root266Tail32
def root266Tail30 : List Nat := 10711 :: root266Tail31
def root266Tail29 : List Nat := 10627 :: root266Tail30
def root266Tail28 : List Nat := 10111 :: root266Tail29
def root266Tail27 : List Nat := 9643 :: root266Tail28
def root266Tail26 : List Nat := 9337 :: root266Tail27
def root266Tail25 : List Nat := 9109 :: root266Tail26
def root266Tail24 : List Nat := 9067 :: root266Tail25
def root266Tail23 : List Nat := 8863 :: root266Tail24
def root266Tail22 : List Nat := 8623 :: root266Tail23
def root266Tail21 : List Nat := 8563 :: root266Tail22
def root266Tail20 : List Nat := 8467 :: root266Tail21
def root266Tail19 : List Nat := 8329 :: root266Tail20
def root266Tail18 : List Nat := 8233 :: root266Tail19
def root266Tail17 : List Nat := 7993 :: root266Tail18
def root266Tail16 : List Nat := 7951 :: root266Tail17
def root266Tail15 : List Nat := 6547 :: root266Tail16
def root266Tail14 : List Nat := 6397 :: root266Tail15
def root266Tail13 : List Nat := 5431 :: root266Tail14
def root266Tail12 : List Nat := 5419 :: root266Tail13
def root266Tail11 : List Nat := 4999 :: root266Tail12
def root266Tail10 : List Nat := 4993 :: root266Tail11
def root266Tail9 : List Nat := 4813 :: root266Tail10
def root266Tail8 : List Nat := 4783 :: root266Tail9
def root266Tail7 : List Nat := 4621 :: root266Tail8
def root266Tail6 : List Nat := 4423 :: root266Tail7
def root266Tail5 : List Nat := 4363 :: root266Tail6
def root266Tail4 : List Nat := 4057 :: root266Tail5
def root266Tail3 : List Nat := 3457 :: root266Tail4
def root266Tail2 : List Nat := 3319 :: root266Tail3
def root266Tail1 : List Nat := 2179 :: root266Tail2
def root266Tail0 : List Nat := 1831 :: root266Tail1
def root266Candidates0 : List Nat := [8233]
set_option maxRecDepth 8192 in
theorem root266Candidates0_checked : candidates edge 4 1723 1831 root266Tail1 winner.bound = root266Candidates0 := by
  unfold candidates
  have row : edge 1831 = (SortedNatTree.contains FastRows.row1831) := by
    change Data.row1831.contains = (SortedNatTree.contains FastRows.row1831)
    exact FastRows.row1831_checked.symm
  rw [row]
  decide
theorem root266Step0 : scan edge (search edge 3) 4 [1723] 1723 root266Tail0 winner =
    scan edge (search edge 3) 4 [1723] 1723 root266Tail1 winner := by
  rw [root266Tail0, scan, if_neg (by decide), root266Candidates0_checked]
  change scan edge (search edge 3) 4 [1723] 1723 root266Tail1
    (if 3 ≤ root266Candidates0.length then search edge 3 [1723, 1831] root266Candidates0 3554 winner else winner) = _
  rw [if_neg (by decide)]
def root266Candidates1 : List Nat := [8467, 9109, 16843]
set_option maxRecDepth 8192 in
theorem root266Candidates1_checked : candidates edge 4 1723 2179 root266Tail2 winner.bound = root266Candidates1 := by
  unfold candidates
  have row : edge 2179 = (SortedNatTree.contains FastRows.row2179) := by
    change Data.row2179.contains = (SortedNatTree.contains FastRows.row2179)
    exact FastRows.row2179_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root266Child1 : search edge 3 [1723, 2179] root266Candidates1 3902 winner = winner := by decide
theorem root266Step1 : scan edge (search edge 3) 4 [1723] 1723 root266Tail1 winner =
    scan edge (search edge 3) 4 [1723] 1723 root266Tail2 winner := by
  rw [root266Tail1, scan, if_neg (by decide), root266Candidates1_checked]
  change scan edge (search edge 3) 4 [1723] 1723 root266Tail2
    (if 3 ≤ root266Candidates1.length then search edge 3 [1723, 2179] root266Candidates1 3902 winner else winner) = _
  rw [if_pos (by decide), root266Child1]
def root266Candidates2 : List Nat := [4363, 5419, 7951]
set_option maxRecDepth 8192 in
theorem root266Candidates2_checked : candidates edge 4 1723 3319 root266Tail3 winner.bound = root266Candidates2 := by
  unfold candidates
  have row : edge 3319 = (SortedNatTree.contains FastRows.row3319) := by
    change Data.row3319.contains = (SortedNatTree.contains FastRows.row3319)
    exact FastRows.row3319_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root266Child2 : search edge 3 [1723, 3319] root266Candidates2 5042 winner = winner := by decide
theorem root266Step2 : scan edge (search edge 3) 4 [1723] 1723 root266Tail2 winner =
    scan edge (search edge 3) 4 [1723] 1723 root266Tail3 winner := by
  rw [root266Tail2, scan, if_neg (by decide), root266Candidates2_checked]
  change scan edge (search edge 3) 4 [1723] 1723 root266Tail3
    (if 3 ≤ root266Candidates2.length then search edge 3 [1723, 3319] root266Candidates2 5042 winner else winner) = _
  rw [if_pos (by decide), root266Child2]
def root266Candidates3 : List Nat := [5419, 15607]
set_option maxRecDepth 8192 in
theorem root266Candidates3_checked : candidates edge 4 1723 3457 root266Tail4 winner.bound = root266Candidates3 := by
  unfold candidates
  have row : edge 3457 = (SortedNatTree.contains FastRows.row3457) := by
    change Data.row3457.contains = (SortedNatTree.contains FastRows.row3457)
    exact FastRows.row3457_checked.symm
  rw [row]
  decide
theorem root266Step3 : scan edge (search edge 3) 4 [1723] 1723 root266Tail3 winner =
    scan edge (search edge 3) 4 [1723] 1723 root266Tail4 winner := by
  rw [root266Tail3, scan, if_neg (by decide), root266Candidates3_checked]
  change scan edge (search edge 3) 4 [1723] 1723 root266Tail4
    (if 3 ≤ root266Candidates3.length then search edge 3 [1723, 3457] root266Candidates3 5180 winner else winner) = _
  rw [if_neg (by decide)]
def root266Candidates4 : List Nat := [5419, 9109]
set_option maxRecDepth 8192 in
theorem root266Candidates4_checked : candidates edge 4 1723 4057 root266Tail5 winner.bound = root266Candidates4 := by
  unfold candidates
  have row : edge 4057 = (SortedNatTree.contains FastRows.row4057) := by
    change Data.row4057.contains = (SortedNatTree.contains FastRows.row4057)
    exact FastRows.row4057_checked.symm
  rw [row]
  decide
theorem root266Step4 : scan edge (search edge 3) 4 [1723] 1723 root266Tail4 winner =
    scan edge (search edge 3) 4 [1723] 1723 root266Tail5 winner := by
  rw [root266Tail4, scan, if_neg (by decide), root266Candidates4_checked]
  change scan edge (search edge 3) 4 [1723] 1723 root266Tail5
    (if 3 ≤ root266Candidates4.length then search edge 3 [1723, 4057] root266Candidates4 5780 winner else winner) = _
  rw [if_neg (by decide)]
def root266Candidates5 : List Nat := [4813]
set_option maxRecDepth 8192 in
theorem root266Candidates5_checked : candidates edge 4 1723 4363 root266Tail6 winner.bound = root266Candidates5 := by
  unfold candidates
  have row : edge 4363 = (SortedNatTree.contains FastRows.row4363) := by
    change Data.row4363.contains = (SortedNatTree.contains FastRows.row4363)
    exact FastRows.row4363_checked.symm
  rw [row]
  decide
theorem root266Step5 : scan edge (search edge 3) 4 [1723] 1723 root266Tail5 winner =
    scan edge (search edge 3) 4 [1723] 1723 root266Tail6 winner := by
  rw [root266Tail5, scan, if_neg (by decide), root266Candidates5_checked]
  change scan edge (search edge 3) 4 [1723] 1723 root266Tail6
    (if 3 ≤ root266Candidates5.length then search edge 3 [1723, 4363] root266Candidates5 6086 winner else winner) = _
  rw [if_neg (by decide)]
def root266Candidates6 : List Nat := [8863, 9337]
set_option maxRecDepth 8192 in
theorem root266Candidates6_checked : candidates edge 4 1723 4423 root266Tail7 winner.bound = root266Candidates6 := by
  unfold candidates
  have row : edge 4423 = (SortedNatTree.contains FastRows.row4423) := by
    change Data.row4423.contains = (SortedNatTree.contains FastRows.row4423)
    exact FastRows.row4423_checked.symm
  rw [row]
  decide
theorem root266Step6 : scan edge (search edge 3) 4 [1723] 1723 root266Tail6 winner =
    scan edge (search edge 3) 4 [1723] 1723 root266Tail7 winner := by
  rw [root266Tail6, scan, if_neg (by decide), root266Candidates6_checked]
  change scan edge (search edge 3) 4 [1723] 1723 root266Tail7
    (if 3 ≤ root266Candidates6.length then search edge 3 [1723, 4423] root266Candidates6 6146 winner else winner) = _
  rw [if_neg (by decide)]
def root266Candidates7 : List Nat := [9337, 17377]
set_option maxRecDepth 8192 in
theorem root266Candidates7_checked : candidates edge 4 1723 4621 root266Tail8 winner.bound = root266Candidates7 := by
  unfold candidates
  have row : edge 4621 = (SortedNatTree.contains FastRows.row4621) := by
    change Data.row4621.contains = (SortedNatTree.contains FastRows.row4621)
    exact FastRows.row4621_checked.symm
  rw [row]
  decide
theorem root266Step7 : scan edge (search edge 3) 4 [1723] 1723 root266Tail7 winner =
    scan edge (search edge 3) 4 [1723] 1723 root266Tail8 winner := by
  rw [root266Tail7, scan, if_neg (by decide), root266Candidates7_checked]
  change scan edge (search edge 3) 4 [1723] 1723 root266Tail8
    (if 3 ≤ root266Candidates7.length then search edge 3 [1723, 4621] root266Candidates7 6344 winner else winner) = _
  rw [if_neg (by decide)]
def root266Candidates8 : List Nat := [4999]
set_option maxRecDepth 8192 in
theorem root266Candidates8_checked : candidates edge 4 1723 4783 root266Tail9 winner.bound = root266Candidates8 := by
  unfold candidates
  have row : edge 4783 = (SortedNatTree.contains FastRows.row4783) := by
    change Data.row4783.contains = (SortedNatTree.contains FastRows.row4783)
    exact FastRows.row4783_checked.symm
  rw [row]
  decide
theorem root266Step8 : scan edge (search edge 3) 4 [1723] 1723 root266Tail8 winner =
    scan edge (search edge 3) 4 [1723] 1723 root266Tail9 winner := by
  rw [root266Tail8, scan, if_neg (by decide), root266Candidates8_checked]
  change scan edge (search edge 3) 4 [1723] 1723 root266Tail9
    (if 3 ≤ root266Candidates8.length then search edge 3 [1723, 4783] root266Candidates8 6506 winner else winner) = _
  rw [if_neg (by decide)]
def root266Candidates9 : List Nat := [8863, 9643]
set_option maxRecDepth 8192 in
theorem root266Candidates9_checked : candidates edge 4 1723 4813 root266Tail10 winner.bound = root266Candidates9 := by
  unfold candidates
  have row : edge 4813 = (SortedNatTree.contains FastRows.row4813) := by
    change Data.row4813.contains = (SortedNatTree.contains FastRows.row4813)
    exact FastRows.row4813_checked.symm
  rw [row]
  decide
theorem root266Step9 : scan edge (search edge 3) 4 [1723] 1723 root266Tail9 winner =
    scan edge (search edge 3) 4 [1723] 1723 root266Tail10 winner := by
  rw [root266Tail9, scan, if_neg (by decide), root266Candidates9_checked]
  change scan edge (search edge 3) 4 [1723] 1723 root266Tail10
    (if 3 ≤ root266Candidates9.length then search edge 3 [1723, 4813] root266Candidates9 6536 winner else winner) = _
  rw [if_neg (by decide)]
def root266Candidates10 : List Nat := [4999, 8563]
set_option maxRecDepth 8192 in
theorem root266Candidates10_checked : candidates edge 4 1723 4993 root266Tail11 winner.bound = root266Candidates10 := by
  unfold candidates
  have row : edge 4993 = (SortedNatTree.contains FastRows.row4993) := by
    change Data.row4993.contains = (SortedNatTree.contains FastRows.row4993)
    exact FastRows.row4993_checked.symm
  rw [row]
  decide
theorem root266Step10 : scan edge (search edge 3) 4 [1723] 1723 root266Tail10 winner =
    scan edge (search edge 3) 4 [1723] 1723 root266Tail11 winner := by
  rw [root266Tail10, scan, if_neg (by decide), root266Candidates10_checked]
  change scan edge (search edge 3) 4 [1723] 1723 root266Tail11
    (if 3 ≤ root266Candidates10.length then search edge 3 [1723, 4993] root266Candidates10 6716 winner else winner) = _
  rw [if_neg (by decide)]
def root266Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root266Candidates11_checked : candidates edge 4 1723 4999 root266Tail12 winner.bound = root266Candidates11 := by
  unfold candidates
  have row : edge 4999 = (SortedNatTree.contains FastRows.row4999) := by
    change Data.row4999.contains = (SortedNatTree.contains FastRows.row4999)
    exact FastRows.row4999_checked.symm
  rw [row]
  decide
theorem root266Step11 : scan edge (search edge 3) 4 [1723] 1723 root266Tail11 winner =
    scan edge (search edge 3) 4 [1723] 1723 root266Tail12 winner := by
  rw [root266Tail11, scan, if_neg (by decide), root266Candidates11_checked]
  change scan edge (search edge 3) 4 [1723] 1723 root266Tail12
    (if 3 ≤ root266Candidates11.length then search edge 3 [1723, 4999] root266Candidates11 6722 winner else winner) = _
  rw [if_neg (by decide)]
def root266Candidates12 : List Nat := [10627, 11467, 18301]
set_option maxRecDepth 8192 in
theorem root266Candidates12_checked : candidates edge 4 1723 5419 root266Tail13 winner.bound = root266Candidates12 := by
  unfold candidates
  have row : edge 5419 = (SortedNatTree.contains FastRows.row5419) := by
    change Data.row5419.contains = (SortedNatTree.contains FastRows.row5419)
    exact FastRows.row5419_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root266Child12 : search edge 3 [1723, 5419] root266Candidates12 7142 winner = winner := by decide
theorem root266Step12 : scan edge (search edge 3) 4 [1723] 1723 root266Tail12 winner =
    scan edge (search edge 3) 4 [1723] 1723 root266Tail13 winner := by
  rw [root266Tail12, scan, if_neg (by decide), root266Candidates12_checked]
  change scan edge (search edge 3) 4 [1723] 1723 root266Tail13
    (if 3 ≤ root266Candidates12.length then search edge 3 [1723, 5419] root266Candidates12 7142 winner else winner) = _
  rw [if_pos (by decide), root266Child12]
def root266Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root266Candidates13_checked : candidates edge 4 1723 5431 root266Tail14 winner.bound = root266Candidates13 := by
  unfold candidates
  have row : edge 5431 = (SortedNatTree.contains FastRows.row5431) := by
    change Data.row5431.contains = (SortedNatTree.contains FastRows.row5431)
    exact FastRows.row5431_checked.symm
  rw [row]
  decide
theorem root266Step13 : scan edge (search edge 3) 4 [1723] 1723 root266Tail13 winner =
    scan edge (search edge 3) 4 [1723] 1723 root266Tail14 winner := by
  rw [root266Tail13, scan, if_neg (by decide), root266Candidates13_checked]
  change scan edge (search edge 3) 4 [1723] 1723 root266Tail14
    (if 3 ≤ root266Candidates13.length then search edge 3 [1723, 5431] root266Candidates13 7154 winner else winner) = _
  rw [if_neg (by decide)]
theorem root266Scan14 : scan edge (search edge 3) 4 [1723] 1723 root266Tail14 winner = winner := by rw [root266Tail14, scan, if_pos (by decide)]
theorem root266Scan13 : scan edge (search edge 3) 4 [1723] 1723 root266Tail13 winner = winner :=
  (root266Step13).trans root266Scan14
theorem root266Scan12 : scan edge (search edge 3) 4 [1723] 1723 root266Tail12 winner = winner :=
  (root266Step12).trans root266Scan13
theorem root266Scan11 : scan edge (search edge 3) 4 [1723] 1723 root266Tail11 winner = winner :=
  (root266Step11).trans root266Scan12
theorem root266Scan10 : scan edge (search edge 3) 4 [1723] 1723 root266Tail10 winner = winner :=
  (root266Step10).trans root266Scan11
theorem root266Scan9 : scan edge (search edge 3) 4 [1723] 1723 root266Tail9 winner = winner :=
  (root266Step9).trans root266Scan10
theorem root266Scan8 : scan edge (search edge 3) 4 [1723] 1723 root266Tail8 winner = winner :=
  (root266Step8).trans root266Scan9
theorem root266Scan7 : scan edge (search edge 3) 4 [1723] 1723 root266Tail7 winner = winner :=
  (root266Step7).trans root266Scan8
theorem root266Scan6 : scan edge (search edge 3) 4 [1723] 1723 root266Tail6 winner = winner :=
  (root266Step6).trans root266Scan7
theorem root266Scan5 : scan edge (search edge 3) 4 [1723] 1723 root266Tail5 winner = winner :=
  (root266Step5).trans root266Scan6
theorem root266Scan4 : scan edge (search edge 3) 4 [1723] 1723 root266Tail4 winner = winner :=
  (root266Step4).trans root266Scan5
theorem root266Scan3 : scan edge (search edge 3) 4 [1723] 1723 root266Tail3 winner = winner :=
  (root266Step3).trans root266Scan4
theorem root266Scan2 : scan edge (search edge 3) 4 [1723] 1723 root266Tail2 winner = winner :=
  (root266Step2).trans root266Scan3
theorem root266Scan1 : scan edge (search edge 3) 4 [1723] 1723 root266Tail1 winner = winner :=
  (root266Step1).trans root266Scan2
theorem root266Scan0 : scan edge (search edge 3) 4 [1723] 1723 root266Tail0 winner = winner :=
  (root266Step0).trans root266Scan1
set_option maxRecDepth 4096 in
theorem root266Branch : search edge 4 [1723] root266Candidates 1723 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1723] 1723 root266Tail0 winner = winner
  exact root266Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
