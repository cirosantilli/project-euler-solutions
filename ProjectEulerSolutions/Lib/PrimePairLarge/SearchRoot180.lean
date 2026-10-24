import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root180Candidates : List Nat := Data.row1093
set_option maxRecDepth 8192 in
theorem root180Candidates_checked : candidates edge 5 0 1093 tail181 winner.bound = root180Candidates := by
  apply candidates_eq_row edge 5 0 1093 tail181 root180Candidates winner.bound tail181_ordered
  · decide
  · decide
  · funext q
    rfl
def root180Tail50 : List Nat := []
def root180Tail49 : List Nat := 24049 :: root180Tail50
def root180Tail48 : List Nat := 23971 :: root180Tail49
def root180Tail47 : List Nat := 23593 :: root180Tail48
def root180Tail46 : List Nat := 22483 :: root180Tail47
def root180Tail45 : List Nat := 22453 :: root180Tail46
def root180Tail44 : List Nat := 21529 :: root180Tail45
def root180Tail43 : List Nat := 20611 :: root180Tail44
def root180Tail42 : List Nat := 20359 :: root180Tail43
def root180Tail41 : List Nat := 19993 :: root180Tail42
def root180Tail40 : List Nat := 19447 :: root180Tail41
def root180Tail39 : List Nat := 18523 :: root180Tail40
def root180Tail38 : List Nat := 17851 :: root180Tail39
def root180Tail37 : List Nat := 17539 :: root180Tail38
def root180Tail36 : List Nat := 17389 :: root180Tail37
def root180Tail35 : List Nat := 17077 :: root180Tail36
def root180Tail34 : List Nat := 17041 :: root180Tail35
def root180Tail33 : List Nat := 16183 :: root180Tail34
def root180Tail32 : List Nat := 15919 :: root180Tail33
def root180Tail31 : List Nat := 15511 :: root180Tail32
def root180Tail30 : List Nat := 14431 :: root180Tail31
def root180Tail29 : List Nat := 13759 :: root180Tail30
def root180Tail28 : List Nat := 13063 :: root180Tail29
def root180Tail27 : List Nat := 13033 :: root180Tail28
def root180Tail26 : List Nat := 12823 :: root180Tail27
def root180Tail25 : List Nat := 12391 :: root180Tail26
def root180Tail24 : List Nat := 10093 :: root180Tail25
def root180Tail23 : List Nat := 9883 :: root180Tail24
def root180Tail22 : List Nat := 9043 :: root180Tail23
def root180Tail21 : List Nat := 8689 :: root180Tail22
def root180Tail20 : List Nat := 7741 :: root180Tail21
def root180Tail19 : List Nat := 7309 :: root180Tail20
def root180Tail18 : List Nat := 7057 :: root180Tail19
def root180Tail17 : List Nat := 6619 :: root180Tail18
def root180Tail16 : List Nat := 6553 :: root180Tail17
def root180Tail15 : List Nat := 6481 :: root180Tail16
def root180Tail14 : List Nat := 6427 :: root180Tail15
def root180Tail13 : List Nat := 5779 :: root180Tail14
def root180Tail12 : List Nat := 5569 :: root180Tail13
def root180Tail11 : List Nat := 5413 :: root180Tail12
def root180Tail10 : List Nat := 4813 :: root180Tail11
def root180Tail9 : List Nat := 4603 :: root180Tail10
def root180Tail8 : List Nat := 4447 :: root180Tail9
def root180Tail7 : List Nat := 4243 :: root180Tail8
def root180Tail6 : List Nat := 4201 :: root180Tail7
def root180Tail5 : List Nat := 3331 :: root180Tail6
def root180Tail4 : List Nat := 3061 :: root180Tail5
def root180Tail3 : List Nat := 2437 :: root180Tail4
def root180Tail2 : List Nat := 1861 :: root180Tail3
def root180Tail1 : List Nat := 1549 :: root180Tail2
def root180Tail0 : List Nat := 1471 :: root180Tail1
def root180Candidates0 : List Nat := [14431, 21529]
set_option maxRecDepth 8192 in
theorem root180Candidates0_checked : candidates edge 4 1093 1471 root180Tail1 winner.bound = root180Candidates0 := by
  unfold candidates
  have row : edge 1471 = (SortedNatTree.contains FastRows.row1471) := by
    change Data.row1471.contains = (SortedNatTree.contains FastRows.row1471)
    exact FastRows.row1471_checked.symm
  rw [row]
  decide
theorem root180Step0 : scan edge (search edge 3) 4 [1093] 1093 root180Tail0 winner =
    scan edge (search edge 3) 4 [1093] 1093 root180Tail1 winner := by
  rw [root180Tail0, scan, if_neg (by decide), root180Candidates0_checked]
  change scan edge (search edge 3) 4 [1093] 1093 root180Tail1
    (if 3 ≤ root180Candidates0.length then search edge 3 [1093, 1471] root180Candidates0 2564 winner else winner) = _
  rw [if_neg (by decide)]
def root180Candidates1 : List Nat := [8689]
set_option maxRecDepth 8192 in
theorem root180Candidates1_checked : candidates edge 4 1093 1549 root180Tail2 winner.bound = root180Candidates1 := by
  unfold candidates
  have row : edge 1549 = (SortedNatTree.contains FastRows.row1549) := by
    change Data.row1549.contains = (SortedNatTree.contains FastRows.row1549)
    exact FastRows.row1549_checked.symm
  rw [row]
  decide
theorem root180Step1 : scan edge (search edge 3) 4 [1093] 1093 root180Tail1 winner =
    scan edge (search edge 3) 4 [1093] 1093 root180Tail2 winner := by
  rw [root180Tail1, scan, if_neg (by decide), root180Candidates1_checked]
  change scan edge (search edge 3) 4 [1093] 1093 root180Tail2
    (if 3 ≤ root180Candidates1.length then search edge 3 [1093, 1549] root180Candidates1 2642 winner else winner) = _
  rw [if_neg (by decide)]
def root180Candidates2 : List Nat := [19993]
set_option maxRecDepth 8192 in
theorem root180Candidates2_checked : candidates edge 4 1093 1861 root180Tail3 winner.bound = root180Candidates2 := by
  unfold candidates
  have row : edge 1861 = (SortedNatTree.contains FastRows.row1861) := by
    change Data.row1861.contains = (SortedNatTree.contains FastRows.row1861)
    exact FastRows.row1861_checked.symm
  rw [row]
  decide
theorem root180Step2 : scan edge (search edge 3) 4 [1093] 1093 root180Tail2 winner =
    scan edge (search edge 3) 4 [1093] 1093 root180Tail3 winner := by
  rw [root180Tail2, scan, if_neg (by decide), root180Candidates2_checked]
  change scan edge (search edge 3) 4 [1093] 1093 root180Tail3
    (if 3 ≤ root180Candidates2.length then search edge 3 [1093, 1861] root180Candidates2 2954 winner else winner) = _
  rw [if_neg (by decide)]
def root180Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root180Candidates3_checked : candidates edge 4 1093 2437 root180Tail4 winner.bound = root180Candidates3 := by
  unfold candidates
  have row : edge 2437 = (SortedNatTree.contains FastRows.row2437) := by
    change Data.row2437.contains = (SortedNatTree.contains FastRows.row2437)
    exact FastRows.row2437_checked.symm
  rw [row]
  decide
theorem root180Step3 : scan edge (search edge 3) 4 [1093] 1093 root180Tail3 winner =
    scan edge (search edge 3) 4 [1093] 1093 root180Tail4 winner := by
  rw [root180Tail3, scan, if_neg (by decide), root180Candidates3_checked]
  change scan edge (search edge 3) 4 [1093] 1093 root180Tail4
    (if 3 ≤ root180Candidates3.length then search edge 3 [1093, 2437] root180Candidates3 3530 winner else winner) = _
  rw [if_neg (by decide)]
def root180Candidates4 : List Nat := [15919]
set_option maxRecDepth 8192 in
theorem root180Candidates4_checked : candidates edge 4 1093 3061 root180Tail5 winner.bound = root180Candidates4 := by
  unfold candidates
  have row : edge 3061 = (SortedNatTree.contains FastRows.row3061) := by
    change Data.row3061.contains = (SortedNatTree.contains FastRows.row3061)
    exact FastRows.row3061_checked.symm
  rw [row]
  decide
theorem root180Step4 : scan edge (search edge 3) 4 [1093] 1093 root180Tail4 winner =
    scan edge (search edge 3) 4 [1093] 1093 root180Tail5 winner := by
  rw [root180Tail4, scan, if_neg (by decide), root180Candidates4_checked]
  change scan edge (search edge 3) 4 [1093] 1093 root180Tail5
    (if 3 ≤ root180Candidates4.length then search edge 3 [1093, 3061] root180Candidates4 4154 winner else winner) = _
  rw [if_neg (by decide)]
def root180Candidates5 : List Nat := [18523]
set_option maxRecDepth 8192 in
theorem root180Candidates5_checked : candidates edge 4 1093 3331 root180Tail6 winner.bound = root180Candidates5 := by
  unfold candidates
  have row : edge 3331 = (SortedNatTree.contains FastRows.row3331) := by
    change Data.row3331.contains = (SortedNatTree.contains FastRows.row3331)
    exact FastRows.row3331_checked.symm
  rw [row]
  decide
theorem root180Step5 : scan edge (search edge 3) 4 [1093] 1093 root180Tail5 winner =
    scan edge (search edge 3) 4 [1093] 1093 root180Tail6 winner := by
  rw [root180Tail5, scan, if_neg (by decide), root180Candidates5_checked]
  change scan edge (search edge 3) 4 [1093] 1093 root180Tail6
    (if 3 ≤ root180Candidates5.length then search edge 3 [1093, 3331] root180Candidates5 4424 winner else winner) = _
  rw [if_neg (by decide)]
def root180Candidates6 : List Nat := [17041]
set_option maxRecDepth 8192 in
theorem root180Candidates6_checked : candidates edge 4 1093 4201 root180Tail7 winner.bound = root180Candidates6 := by
  unfold candidates
  have row : edge 4201 = (SortedNatTree.contains FastRows.row4201) := by
    change Data.row4201.contains = (SortedNatTree.contains FastRows.row4201)
    exact FastRows.row4201_checked.symm
  rw [row]
  decide
theorem root180Step6 : scan edge (search edge 3) 4 [1093] 1093 root180Tail6 winner =
    scan edge (search edge 3) 4 [1093] 1093 root180Tail7 winner := by
  rw [root180Tail6, scan, if_neg (by decide), root180Candidates6_checked]
  change scan edge (search edge 3) 4 [1093] 1093 root180Tail7
    (if 3 ≤ root180Candidates6.length then search edge 3 [1093, 4201] root180Candidates6 5294 winner else winner) = _
  rw [if_neg (by decide)]
def root180Candidates7 : List Nat := [6553, 12391, 13063]
set_option maxRecDepth 8192 in
theorem root180Candidates7_checked : candidates edge 4 1093 4243 root180Tail8 winner.bound = root180Candidates7 := by
  unfold candidates
  have row : edge 4243 = (SortedNatTree.contains FastRows.row4243) := by
    change Data.row4243.contains = (SortedNatTree.contains FastRows.row4243)
    exact FastRows.row4243_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root180Child7 : search edge 3 [1093, 4243] root180Candidates7 5336 winner = winner := by decide
theorem root180Step7 : scan edge (search edge 3) 4 [1093] 1093 root180Tail7 winner =
    scan edge (search edge 3) 4 [1093] 1093 root180Tail8 winner := by
  rw [root180Tail7, scan, if_neg (by decide), root180Candidates7_checked]
  change scan edge (search edge 3) 4 [1093] 1093 root180Tail8
    (if 3 ≤ root180Candidates7.length then search edge 3 [1093, 4243] root180Candidates7 5336 winner else winner) = _
  rw [if_pos (by decide), root180Child7]
def root180Candidates8 : List Nat := [6427, 7057]
set_option maxRecDepth 8192 in
theorem root180Candidates8_checked : candidates edge 4 1093 4447 root180Tail9 winner.bound = root180Candidates8 := by
  unfold candidates
  have row : edge 4447 = (SortedNatTree.contains FastRows.row4447) := by
    change Data.row4447.contains = (SortedNatTree.contains FastRows.row4447)
    exact FastRows.row4447_checked.symm
  rw [row]
  decide
theorem root180Step8 : scan edge (search edge 3) 4 [1093] 1093 root180Tail8 winner =
    scan edge (search edge 3) 4 [1093] 1093 root180Tail9 winner := by
  rw [root180Tail8, scan, if_neg (by decide), root180Candidates8_checked]
  change scan edge (search edge 3) 4 [1093] 1093 root180Tail9
    (if 3 ≤ root180Candidates8.length then search edge 3 [1093, 4447] root180Candidates8 5540 winner else winner) = _
  rw [if_neg (by decide)]
def root180Candidates9 : List Nat := [5569]
set_option maxRecDepth 8192 in
theorem root180Candidates9_checked : candidates edge 4 1093 4603 root180Tail10 winner.bound = root180Candidates9 := by
  unfold candidates
  have row : edge 4603 = (SortedNatTree.contains FastRows.row4603) := by
    change Data.row4603.contains = (SortedNatTree.contains FastRows.row4603)
    exact FastRows.row4603_checked.symm
  rw [row]
  decide
theorem root180Step9 : scan edge (search edge 3) 4 [1093] 1093 root180Tail9 winner =
    scan edge (search edge 3) 4 [1093] 1093 root180Tail10 winner := by
  rw [root180Tail9, scan, if_neg (by decide), root180Candidates9_checked]
  change scan edge (search edge 3) 4 [1093] 1093 root180Tail10
    (if 3 ≤ root180Candidates9.length then search edge 3 [1093, 4603] root180Candidates9 5696 winner else winner) = _
  rw [if_neg (by decide)]
def root180Candidates10 : List Nat := [6427, 7057]
set_option maxRecDepth 8192 in
theorem root180Candidates10_checked : candidates edge 4 1093 4813 root180Tail11 winner.bound = root180Candidates10 := by
  unfold candidates
  have row : edge 4813 = (SortedNatTree.contains FastRows.row4813) := by
    change Data.row4813.contains = (SortedNatTree.contains FastRows.row4813)
    exact FastRows.row4813_checked.symm
  rw [row]
  decide
theorem root180Step10 : scan edge (search edge 3) 4 [1093] 1093 root180Tail10 winner =
    scan edge (search edge 3) 4 [1093] 1093 root180Tail11 winner := by
  rw [root180Tail10, scan, if_neg (by decide), root180Candidates10_checked]
  change scan edge (search edge 3) 4 [1093] 1093 root180Tail11
    (if 3 ≤ root180Candidates10.length then search edge 3 [1093, 4813] root180Candidates10 5906 winner else winner) = _
  rw [if_neg (by decide)]
def root180Candidates11 : List Nat := [5569]
set_option maxRecDepth 8192 in
theorem root180Candidates11_checked : candidates edge 4 1093 5413 root180Tail12 winner.bound = root180Candidates11 := by
  unfold candidates
  have row : edge 5413 = (SortedNatTree.contains FastRows.row5413) := by
    change Data.row5413.contains = (SortedNatTree.contains FastRows.row5413)
    exact FastRows.row5413_checked.symm
  rw [row]
  decide
theorem root180Step11 : scan edge (search edge 3) 4 [1093] 1093 root180Tail11 winner =
    scan edge (search edge 3) 4 [1093] 1093 root180Tail12 winner := by
  rw [root180Tail11, scan, if_neg (by decide), root180Candidates11_checked]
  change scan edge (search edge 3) 4 [1093] 1093 root180Tail12
    (if 3 ≤ root180Candidates11.length then search edge 3 [1093, 5413] root180Candidates11 6506 winner else winner) = _
  rw [if_neg (by decide)]
def root180Candidates12 : List Nat := [7309]
set_option maxRecDepth 8192 in
theorem root180Candidates12_checked : candidates edge 4 1093 5569 root180Tail13 winner.bound = root180Candidates12 := by
  unfold candidates
  have row : edge 5569 = (SortedNatTree.contains FastRows.row5569) := by
    change Data.row5569.contains = (SortedNatTree.contains FastRows.row5569)
    exact FastRows.row5569_checked.symm
  rw [row]
  decide
theorem root180Step12 : scan edge (search edge 3) 4 [1093] 1093 root180Tail12 winner =
    scan edge (search edge 3) 4 [1093] 1093 root180Tail13 winner := by
  rw [root180Tail12, scan, if_neg (by decide), root180Candidates12_checked]
  change scan edge (search edge 3) 4 [1093] 1093 root180Tail13
    (if 3 ≤ root180Candidates12.length then search edge 3 [1093, 5569] root180Candidates12 6662 winner else winner) = _
  rw [if_neg (by decide)]
def root180Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root180Candidates13_checked : candidates edge 4 1093 5779 root180Tail14 winner.bound = root180Candidates13 := by
  unfold candidates
  have row : edge 5779 = (SortedNatTree.contains FastRows.row5779) := by
    change Data.row5779.contains = (SortedNatTree.contains FastRows.row5779)
    exact FastRows.row5779_checked.symm
  rw [row]
  decide
theorem root180Step13 : scan edge (search edge 3) 4 [1093] 1093 root180Tail13 winner =
    scan edge (search edge 3) 4 [1093] 1093 root180Tail14 winner := by
  rw [root180Tail13, scan, if_neg (by decide), root180Candidates13_checked]
  change scan edge (search edge 3) 4 [1093] 1093 root180Tail14
    (if 3 ≤ root180Candidates13.length then search edge 3 [1093, 5779] root180Candidates13 6872 winner else winner) = _
  rw [if_neg (by decide)]
theorem root180Scan14 : scan edge (search edge 3) 4 [1093] 1093 root180Tail14 winner = winner := by rw [root180Tail14, scan, if_pos (by decide)]
theorem root180Scan13 : scan edge (search edge 3) 4 [1093] 1093 root180Tail13 winner = winner :=
  (root180Step13).trans root180Scan14
theorem root180Scan12 : scan edge (search edge 3) 4 [1093] 1093 root180Tail12 winner = winner :=
  (root180Step12).trans root180Scan13
theorem root180Scan11 : scan edge (search edge 3) 4 [1093] 1093 root180Tail11 winner = winner :=
  (root180Step11).trans root180Scan12
theorem root180Scan10 : scan edge (search edge 3) 4 [1093] 1093 root180Tail10 winner = winner :=
  (root180Step10).trans root180Scan11
theorem root180Scan9 : scan edge (search edge 3) 4 [1093] 1093 root180Tail9 winner = winner :=
  (root180Step9).trans root180Scan10
theorem root180Scan8 : scan edge (search edge 3) 4 [1093] 1093 root180Tail8 winner = winner :=
  (root180Step8).trans root180Scan9
theorem root180Scan7 : scan edge (search edge 3) 4 [1093] 1093 root180Tail7 winner = winner :=
  (root180Step7).trans root180Scan8
theorem root180Scan6 : scan edge (search edge 3) 4 [1093] 1093 root180Tail6 winner = winner :=
  (root180Step6).trans root180Scan7
theorem root180Scan5 : scan edge (search edge 3) 4 [1093] 1093 root180Tail5 winner = winner :=
  (root180Step5).trans root180Scan6
theorem root180Scan4 : scan edge (search edge 3) 4 [1093] 1093 root180Tail4 winner = winner :=
  (root180Step4).trans root180Scan5
theorem root180Scan3 : scan edge (search edge 3) 4 [1093] 1093 root180Tail3 winner = winner :=
  (root180Step3).trans root180Scan4
theorem root180Scan2 : scan edge (search edge 3) 4 [1093] 1093 root180Tail2 winner = winner :=
  (root180Step2).trans root180Scan3
theorem root180Scan1 : scan edge (search edge 3) 4 [1093] 1093 root180Tail1 winner = winner :=
  (root180Step1).trans root180Scan2
theorem root180Scan0 : scan edge (search edge 3) 4 [1093] 1093 root180Tail0 winner = winner :=
  (root180Step0).trans root180Scan1
set_option maxRecDepth 4096 in
theorem root180Branch : search edge 4 [1093] root180Candidates 1093 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1093] 1093 root180Tail0 winner = winner
  exact root180Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
