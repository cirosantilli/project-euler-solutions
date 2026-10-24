import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root231Candidates : List Nat := Data.row1481
set_option maxRecDepth 8192 in
theorem root231Candidates_checked : candidates edge 5 0 1481 tail232 winner.bound = root231Candidates := by
  apply candidates_eq_row edge 5 0 1481 tail232 root231Candidates winner.bound tail232_ordered
  · decide
  · decide
  · funext q
    rfl
def root231Tail53 : List Nat := []
def root231Tail52 : List Nat := 23831 :: root231Tail53
def root231Tail51 : List Nat := 23741 :: root231Tail52
def root231Tail50 : List Nat := 23117 :: root231Tail51
def root231Tail49 : List Nat := 22901 :: root231Tail50
def root231Tail48 : List Nat := 22571 :: root231Tail49
def root231Tail47 : List Nat := 22349 :: root231Tail48
def root231Tail46 : List Nat := 21227 :: root231Tail47
def root231Tail45 : List Nat := 21023 :: root231Tail46
def root231Tail44 : List Nat := 19709 :: root231Tail45
def root231Tail43 : List Nat := 19553 :: root231Tail44
def root231Tail42 : List Nat := 19427 :: root231Tail43
def root231Tail41 : List Nat := 18701 :: root231Tail42
def root231Tail40 : List Nat := 18371 :: root231Tail41
def root231Tail39 : List Nat := 18329 :: root231Tail40
def root231Tail38 : List Nat := 17393 :: root231Tail39
def root231Tail37 : List Nat := 16301 :: root231Tail38
def root231Tail36 : List Nat := 16061 :: root231Tail37
def root231Tail35 : List Nat := 15629 :: root231Tail36
def root231Tail34 : List Nat := 15269 :: root231Tail35
def root231Tail33 : List Nat := 14717 :: root231Tail34
def root231Tail32 : List Nat := 14327 :: root231Tail33
def root231Tail31 : List Nat := 13907 :: root231Tail32
def root231Tail30 : List Nat := 13799 :: root231Tail31
def root231Tail29 : List Nat := 13619 :: root231Tail30
def root231Tail28 : List Nat := 13499 :: root231Tail29
def root231Tail27 : List Nat := 13337 :: root231Tail28
def root231Tail26 : List Nat := 12647 :: root231Tail27
def root231Tail25 : List Nat := 11987 :: root231Tail26
def root231Tail24 : List Nat := 11681 :: root231Tail25
def root231Tail23 : List Nat := 10427 :: root231Tail24
def root231Tail22 : List Nat := 10007 :: root231Tail23
def root231Tail21 : List Nat := 9923 :: root231Tail22
def root231Tail20 : List Nat := 8963 :: root231Tail21
def root231Tail19 : List Nat := 8807 :: root231Tail20
def root231Tail18 : List Nat := 8597 :: root231Tail19
def root231Tail17 : List Nat := 8501 :: root231Tail18
def root231Tail16 : List Nat := 7823 :: root231Tail17
def root231Tail15 : List Nat := 7703 :: root231Tail16
def root231Tail14 : List Nat := 7529 :: root231Tail15
def root231Tail13 : List Nat := 6869 :: root231Tail14
def root231Tail12 : List Nat := 6569 :: root231Tail13
def root231Tail11 : List Nat := 5981 :: root231Tail12
def root231Tail10 : List Nat := 5639 :: root231Tail11
def root231Tail9 : List Nat := 5189 :: root231Tail10
def root231Tail8 : List Nat := 4211 :: root231Tail9
def root231Tail7 : List Nat := 4127 :: root231Tail8
def root231Tail6 : List Nat := 3881 :: root231Tail7
def root231Tail5 : List Nat := 3803 :: root231Tail6
def root231Tail4 : List Nat := 3167 :: root231Tail5
def root231Tail3 : List Nat := 3089 :: root231Tail4
def root231Tail2 : List Nat := 2879 :: root231Tail3
def root231Tail1 : List Nat := 2663 :: root231Tail2
def root231Tail0 : List Nat := 1733 :: root231Tail1
def root231Candidates0 : List Nat := [2663, 8807, 17393, 18329]
set_option maxRecDepth 8192 in
theorem root231Candidates0_checked : candidates edge 4 1481 1733 root231Tail1 winner.bound = root231Candidates0 := by
  unfold candidates
  have row : edge 1733 = (SortedNatTree.contains FastRows.row1733) := by
    change Data.row1733.contains = (SortedNatTree.contains FastRows.row1733)
    exact FastRows.row1733_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root231Child0 : search edge 3 [1481, 1733] root231Candidates0 3214 winner = winner := by decide
theorem root231Step0 : scan edge (search edge 3) 4 [1481] 1481 root231Tail0 winner =
    scan edge (search edge 3) 4 [1481] 1481 root231Tail1 winner := by
  rw [root231Tail0, scan, if_neg (by decide), root231Candidates0_checked]
  change scan edge (search edge 3) 4 [1481] 1481 root231Tail1
    (if 3 ≤ root231Candidates0.length then search edge 3 [1481, 1733] root231Candidates0 3214 winner else winner) = _
  rw [if_pos (by decide), root231Child0]
def root231Candidates1 : List Nat := [10007, 13337, 14327]
set_option maxRecDepth 8192 in
theorem root231Candidates1_checked : candidates edge 4 1481 2663 root231Tail2 winner.bound = root231Candidates1 := by
  unfold candidates
  have row : edge 2663 = (SortedNatTree.contains FastRows.row2663) := by
    change Data.row2663.contains = (SortedNatTree.contains FastRows.row2663)
    exact FastRows.row2663_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root231Child1 : search edge 3 [1481, 2663] root231Candidates1 4144 winner = winner := by decide
theorem root231Step1 : scan edge (search edge 3) 4 [1481] 1481 root231Tail1 winner =
    scan edge (search edge 3) 4 [1481] 1481 root231Tail2 winner := by
  rw [root231Tail1, scan, if_neg (by decide), root231Candidates1_checked]
  change scan edge (search edge 3) 4 [1481] 1481 root231Tail2
    (if 3 ≤ root231Candidates1.length then search edge 3 [1481, 2663] root231Candidates1 4144 winner else winner) = _
  rw [if_pos (by decide), root231Child1]
def root231Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root231Candidates2_checked : candidates edge 4 1481 2879 root231Tail3 winner.bound = root231Candidates2 := by
  unfold candidates
  have row : edge 2879 = (SortedNatTree.contains FastRows.row2879) := by
    change Data.row2879.contains = (SortedNatTree.contains FastRows.row2879)
    exact FastRows.row2879_checked.symm
  rw [row]
  decide
theorem root231Step2 : scan edge (search edge 3) 4 [1481] 1481 root231Tail2 winner =
    scan edge (search edge 3) 4 [1481] 1481 root231Tail3 winner := by
  rw [root231Tail2, scan, if_neg (by decide), root231Candidates2_checked]
  change scan edge (search edge 3) 4 [1481] 1481 root231Tail3
    (if 3 ≤ root231Candidates2.length then search edge 3 [1481, 2879] root231Candidates2 4360 winner else winner) = _
  rw [if_neg (by decide)]
def root231Candidates3 : List Nat := [5639]
set_option maxRecDepth 8192 in
theorem root231Candidates3_checked : candidates edge 4 1481 3089 root231Tail4 winner.bound = root231Candidates3 := by
  unfold candidates
  have row : edge 3089 = (SortedNatTree.contains FastRows.row3089) := by
    change Data.row3089.contains = (SortedNatTree.contains FastRows.row3089)
    exact FastRows.row3089_checked.symm
  rw [row]
  decide
theorem root231Step3 : scan edge (search edge 3) 4 [1481] 1481 root231Tail3 winner =
    scan edge (search edge 3) 4 [1481] 1481 root231Tail4 winner := by
  rw [root231Tail3, scan, if_neg (by decide), root231Candidates3_checked]
  change scan edge (search edge 3) 4 [1481] 1481 root231Tail4
    (if 3 ≤ root231Candidates3.length then search edge 3 [1481, 3089] root231Candidates3 4570 winner else winner) = _
  rw [if_neg (by decide)]
def root231Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root231Candidates4_checked : candidates edge 4 1481 3167 root231Tail5 winner.bound = root231Candidates4 := by
  unfold candidates
  have row : edge 3167 = (SortedNatTree.contains FastRows.row3167) := by
    change Data.row3167.contains = (SortedNatTree.contains FastRows.row3167)
    exact FastRows.row3167_checked.symm
  rw [row]
  decide
theorem root231Step4 : scan edge (search edge 3) 4 [1481] 1481 root231Tail4 winner =
    scan edge (search edge 3) 4 [1481] 1481 root231Tail5 winner := by
  rw [root231Tail4, scan, if_neg (by decide), root231Candidates4_checked]
  change scan edge (search edge 3) 4 [1481] 1481 root231Tail5
    (if 3 ≤ root231Candidates4.length then search edge 3 [1481, 3167] root231Candidates4 4648 winner else winner) = _
  rw [if_neg (by decide)]
def root231Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root231Candidates5_checked : candidates edge 4 1481 3803 root231Tail6 winner.bound = root231Candidates5 := by
  unfold candidates
  have row : edge 3803 = (SortedNatTree.contains FastRows.row3803) := by
    change Data.row3803.contains = (SortedNatTree.contains FastRows.row3803)
    exact FastRows.row3803_checked.symm
  rw [row]
  decide
theorem root231Step5 : scan edge (search edge 3) 4 [1481] 1481 root231Tail5 winner =
    scan edge (search edge 3) 4 [1481] 1481 root231Tail6 winner := by
  rw [root231Tail5, scan, if_neg (by decide), root231Candidates5_checked]
  change scan edge (search edge 3) 4 [1481] 1481 root231Tail6
    (if 3 ≤ root231Candidates5.length then search edge 3 [1481, 3803] root231Candidates5 5284 winner else winner) = _
  rw [if_neg (by decide)]
def root231Candidates6 : List Nat := [18701]
set_option maxRecDepth 8192 in
theorem root231Candidates6_checked : candidates edge 4 1481 3881 root231Tail7 winner.bound = root231Candidates6 := by
  unfold candidates
  have row : edge 3881 = (SortedNatTree.contains FastRows.row3881) := by
    change Data.row3881.contains = (SortedNatTree.contains FastRows.row3881)
    exact FastRows.row3881_checked.symm
  rw [row]
  decide
theorem root231Step6 : scan edge (search edge 3) 4 [1481] 1481 root231Tail6 winner =
    scan edge (search edge 3) 4 [1481] 1481 root231Tail7 winner := by
  rw [root231Tail6, scan, if_neg (by decide), root231Candidates6_checked]
  change scan edge (search edge 3) 4 [1481] 1481 root231Tail7
    (if 3 ≤ root231Candidates6.length then search edge 3 [1481, 3881] root231Candidates6 5362 winner else winner) = _
  rw [if_neg (by decide)]
def root231Candidates7 : List Nat := [11681]
set_option maxRecDepth 8192 in
theorem root231Candidates7_checked : candidates edge 4 1481 4127 root231Tail8 winner.bound = root231Candidates7 := by
  unfold candidates
  have row : edge 4127 = (SortedNatTree.contains FastRows.row4127) := by
    change Data.row4127.contains = (SortedNatTree.contains FastRows.row4127)
    exact FastRows.row4127_checked.symm
  rw [row]
  decide
theorem root231Step7 : scan edge (search edge 3) 4 [1481] 1481 root231Tail7 winner =
    scan edge (search edge 3) 4 [1481] 1481 root231Tail8 winner := by
  rw [root231Tail7, scan, if_neg (by decide), root231Candidates7_checked]
  change scan edge (search edge 3) 4 [1481] 1481 root231Tail8
    (if 3 ≤ root231Candidates7.length then search edge 3 [1481, 4127] root231Candidates7 5608 winner else winner) = _
  rw [if_neg (by decide)]
def root231Candidates8 : List Nat := [7703, 8963, 13619, 14717]
set_option maxRecDepth 8192 in
theorem root231Candidates8_checked : candidates edge 4 1481 4211 root231Tail9 winner.bound = root231Candidates8 := by
  unfold candidates
  have row : edge 4211 = (SortedNatTree.contains FastRows.row4211) := by
    change Data.row4211.contains = (SortedNatTree.contains FastRows.row4211)
    exact FastRows.row4211_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root231Child8 : search edge 3 [1481, 4211] root231Candidates8 5692 winner = winner := by decide
theorem root231Step8 : scan edge (search edge 3) 4 [1481] 1481 root231Tail8 winner =
    scan edge (search edge 3) 4 [1481] 1481 root231Tail9 winner := by
  rw [root231Tail8, scan, if_neg (by decide), root231Candidates8_checked]
  change scan edge (search edge 3) 4 [1481] 1481 root231Tail9
    (if 3 ≤ root231Candidates8.length then search edge 3 [1481, 4211] root231Candidates8 5692 winner else winner) = _
  rw [if_pos (by decide), root231Child8]
def root231Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root231Candidates9_checked : candidates edge 4 1481 5189 root231Tail10 winner.bound = root231Candidates9 := by
  unfold candidates
  have row : edge 5189 = (SortedNatTree.contains FastRows.row5189) := by
    change Data.row5189.contains = (SortedNatTree.contains FastRows.row5189)
    exact FastRows.row5189_checked.symm
  rw [row]
  decide
theorem root231Step9 : scan edge (search edge 3) 4 [1481] 1481 root231Tail9 winner =
    scan edge (search edge 3) 4 [1481] 1481 root231Tail10 winner := by
  rw [root231Tail9, scan, if_neg (by decide), root231Candidates9_checked]
  change scan edge (search edge 3) 4 [1481] 1481 root231Tail10
    (if 3 ≤ root231Candidates9.length then search edge 3 [1481, 5189] root231Candidates9 6670 winner else winner) = _
  rw [if_neg (by decide)]
def root231Candidates10 : List Nat := [8963, 13619]
set_option maxRecDepth 8192 in
theorem root231Candidates10_checked : candidates edge 4 1481 5639 root231Tail11 winner.bound = root231Candidates10 := by
  unfold candidates
  have row : edge 5639 = (SortedNatTree.contains FastRows.row5639) := by
    change Data.row5639.contains = (SortedNatTree.contains FastRows.row5639)
    exact FastRows.row5639_checked.symm
  rw [row]
  decide
theorem root231Step10 : scan edge (search edge 3) 4 [1481] 1481 root231Tail10 winner =
    scan edge (search edge 3) 4 [1481] 1481 root231Tail11 winner := by
  rw [root231Tail10, scan, if_neg (by decide), root231Candidates10_checked]
  change scan edge (search edge 3) 4 [1481] 1481 root231Tail11
    (if 3 ≤ root231Candidates10.length then search edge 3 [1481, 5639] root231Candidates10 7120 winner else winner) = _
  rw [if_neg (by decide)]
def root231Candidates11 : List Nat := [16061]
set_option maxRecDepth 8192 in
theorem root231Candidates11_checked : candidates edge 4 1481 5981 root231Tail12 winner.bound = root231Candidates11 := by
  unfold candidates
  have row : edge 5981 = (SortedNatTree.contains FastRows.row5981) := by
    change Data.row5981.contains = (SortedNatTree.contains FastRows.row5981)
    exact FastRows.row5981_checked.symm
  rw [row]
  decide
theorem root231Step11 : scan edge (search edge 3) 4 [1481] 1481 root231Tail11 winner =
    scan edge (search edge 3) 4 [1481] 1481 root231Tail12 winner := by
  rw [root231Tail11, scan, if_neg (by decide), root231Candidates11_checked]
  change scan edge (search edge 3) 4 [1481] 1481 root231Tail12
    (if 3 ≤ root231Candidates11.length then search edge 3 [1481, 5981] root231Candidates11 7462 winner else winner) = _
  rw [if_neg (by decide)]
theorem root231Scan12 : scan edge (search edge 3) 4 [1481] 1481 root231Tail12 winner = winner := by rw [root231Tail12, scan, if_pos (by decide)]
theorem root231Scan11 : scan edge (search edge 3) 4 [1481] 1481 root231Tail11 winner = winner :=
  (root231Step11).trans root231Scan12
theorem root231Scan10 : scan edge (search edge 3) 4 [1481] 1481 root231Tail10 winner = winner :=
  (root231Step10).trans root231Scan11
theorem root231Scan9 : scan edge (search edge 3) 4 [1481] 1481 root231Tail9 winner = winner :=
  (root231Step9).trans root231Scan10
theorem root231Scan8 : scan edge (search edge 3) 4 [1481] 1481 root231Tail8 winner = winner :=
  (root231Step8).trans root231Scan9
theorem root231Scan7 : scan edge (search edge 3) 4 [1481] 1481 root231Tail7 winner = winner :=
  (root231Step7).trans root231Scan8
theorem root231Scan6 : scan edge (search edge 3) 4 [1481] 1481 root231Tail6 winner = winner :=
  (root231Step6).trans root231Scan7
theorem root231Scan5 : scan edge (search edge 3) 4 [1481] 1481 root231Tail5 winner = winner :=
  (root231Step5).trans root231Scan6
theorem root231Scan4 : scan edge (search edge 3) 4 [1481] 1481 root231Tail4 winner = winner :=
  (root231Step4).trans root231Scan5
theorem root231Scan3 : scan edge (search edge 3) 4 [1481] 1481 root231Tail3 winner = winner :=
  (root231Step3).trans root231Scan4
theorem root231Scan2 : scan edge (search edge 3) 4 [1481] 1481 root231Tail2 winner = winner :=
  (root231Step2).trans root231Scan3
theorem root231Scan1 : scan edge (search edge 3) 4 [1481] 1481 root231Tail1 winner = winner :=
  (root231Step1).trans root231Scan2
theorem root231Scan0 : scan edge (search edge 3) 4 [1481] 1481 root231Tail0 winner = winner :=
  (root231Step0).trans root231Scan1
set_option maxRecDepth 4096 in
theorem root231Branch : search edge 4 [1481] root231Candidates 1481 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1481] 1481 root231Tail0 winner = winner
  exact root231Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
