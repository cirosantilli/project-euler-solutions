import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root219Candidates : List Nat := Data.row1399
set_option maxRecDepth 8192 in
theorem root219Candidates_checked : candidates edge 5 0 1399 tail220 winner.bound = root219Candidates := by
  apply candidates_eq_row edge 5 0 1399 tail220 root219Candidates winner.bound tail220_ordered
  · decide
  · decide
  · funext q
    rfl
def root219Tail42 : List Nat := []
def root219Tail41 : List Nat := 24469 :: root219Tail42
def root219Tail40 : List Nat := 23827 :: root219Tail41
def root219Tail39 : List Nat := 23167 :: root219Tail40
def root219Tail38 : List Nat := 23071 :: root219Tail39
def root219Tail37 : List Nat := 19069 :: root219Tail38
def root219Tail36 : List Nat := 18553 :: root219Tail37
def root219Tail35 : List Nat := 18217 :: root219Tail36
def root219Tail34 : List Nat := 17317 :: root219Tail35
def root219Tail33 : List Nat := 17209 :: root219Tail34
def root219Tail32 : List Nat := 16759 :: root219Tail33
def root219Tail31 : List Nat := 16141 :: root219Tail32
def root219Tail30 : List Nat := 16111 :: root219Tail31
def root219Tail29 : List Nat := 15733 :: root219Tail30
def root219Tail28 : List Nat := 15289 :: root219Tail29
def root219Tail27 : List Nat := 15091 :: root219Tail28
def root219Tail26 : List Nat := 14923 :: root219Tail27
def root219Tail25 : List Nat := 14431 :: root219Tail26
def root219Tail24 : List Nat := 13327 :: root219Tail25
def root219Tail23 : List Nat := 13171 :: root219Tail24
def root219Tail22 : List Nat := 12853 :: root219Tail23
def root219Tail21 : List Nat := 12613 :: root219Tail22
def root219Tail20 : List Nat := 11593 :: root219Tail21
def root219Tail19 : List Nat := 10891 :: root219Tail20
def root219Tail18 : List Nat := 9871 :: root219Tail19
def root219Tail17 : List Nat := 9463 :: root219Tail18
def root219Tail16 : List Nat := 8923 :: root219Tail17
def root219Tail15 : List Nat := 8461 :: root219Tail16
def root219Tail14 : List Nat := 7759 :: root219Tail15
def root219Tail13 : List Nat := 6793 :: root219Tail14
def root219Tail12 : List Nat := 6529 :: root219Tail13
def root219Tail11 : List Nat := 6481 :: root219Tail12
def root219Tail10 : List Nat := 6133 :: root219Tail11
def root219Tail9 : List Nat := 5881 :: root219Tail10
def root219Tail8 : List Nat := 4987 :: root219Tail9
def root219Tail7 : List Nat := 4801 :: root219Tail8
def root219Tail6 : List Nat := 4339 :: root219Tail7
def root219Tail5 : List Nat := 4261 :: root219Tail6
def root219Tail4 : List Nat := 3433 :: root219Tail5
def root219Tail3 : List Nat := 2551 :: root219Tail4
def root219Tail2 : List Nat := 2131 :: root219Tail3
def root219Tail1 : List Nat := 1741 :: root219Tail2
def root219Tail0 : List Nat := 1489 :: root219Tail1
def root219Candidates0 : List Nat := [4801]
set_option maxRecDepth 8192 in
theorem root219Candidates0_checked : candidates edge 4 1399 1489 root219Tail1 winner.bound = root219Candidates0 := by
  unfold candidates
  have row : edge 1489 = (SortedNatTree.contains FastRows.row1489) := by
    change Data.row1489.contains = (SortedNatTree.contains FastRows.row1489)
    exact FastRows.row1489_checked.symm
  rw [row]
  decide
theorem root219Step0 : scan edge (search edge 3) 4 [1399] 1399 root219Tail0 winner =
    scan edge (search edge 3) 4 [1399] 1399 root219Tail1 winner := by
  rw [root219Tail0, scan, if_neg (by decide), root219Candidates0_checked]
  change scan edge (search edge 3) 4 [1399] 1399 root219Tail1
    (if 3 ≤ root219Candidates0.length then search edge 3 [1399, 1489] root219Candidates0 2888 winner else winner) = _
  rw [if_neg (by decide)]
def root219Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root219Candidates1_checked : candidates edge 4 1399 1741 root219Tail2 winner.bound = root219Candidates1 := by
  unfold candidates
  have row : edge 1741 = (SortedNatTree.contains FastRows.row1741) := by
    change Data.row1741.contains = (SortedNatTree.contains FastRows.row1741)
    exact FastRows.row1741_checked.symm
  rw [row]
  decide
theorem root219Step1 : scan edge (search edge 3) 4 [1399] 1399 root219Tail1 winner =
    scan edge (search edge 3) 4 [1399] 1399 root219Tail2 winner := by
  rw [root219Tail1, scan, if_neg (by decide), root219Candidates1_checked]
  change scan edge (search edge 3) 4 [1399] 1399 root219Tail2
    (if 3 ≤ root219Candidates1.length then search edge 3 [1399, 1741] root219Candidates1 3140 winner else winner) = _
  rw [if_neg (by decide)]
def root219Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root219Candidates2_checked : candidates edge 4 1399 2131 root219Tail3 winner.bound = root219Candidates2 := by
  unfold candidates
  have row : edge 2131 = (SortedNatTree.contains FastRows.row2131) := by
    change Data.row2131.contains = (SortedNatTree.contains FastRows.row2131)
    exact FastRows.row2131_checked.symm
  rw [row]
  decide
theorem root219Step2 : scan edge (search edge 3) 4 [1399] 1399 root219Tail2 winner =
    scan edge (search edge 3) 4 [1399] 1399 root219Tail3 winner := by
  rw [root219Tail2, scan, if_neg (by decide), root219Candidates2_checked]
  change scan edge (search edge 3) 4 [1399] 1399 root219Tail3
    (if 3 ≤ root219Candidates2.length then search edge 3 [1399, 2131] root219Candidates2 3530 winner else winner) = _
  rw [if_neg (by decide)]
def root219Candidates3 : List Nat := [6793]
set_option maxRecDepth 8192 in
theorem root219Candidates3_checked : candidates edge 4 1399 2551 root219Tail4 winner.bound = root219Candidates3 := by
  unfold candidates
  have row : edge 2551 = (SortedNatTree.contains FastRows.row2551) := by
    change Data.row2551.contains = (SortedNatTree.contains FastRows.row2551)
    exact FastRows.row2551_checked.symm
  rw [row]
  decide
theorem root219Step3 : scan edge (search edge 3) 4 [1399] 1399 root219Tail3 winner =
    scan edge (search edge 3) 4 [1399] 1399 root219Tail4 winner := by
  rw [root219Tail3, scan, if_neg (by decide), root219Candidates3_checked]
  change scan edge (search edge 3) 4 [1399] 1399 root219Tail4
    (if 3 ≤ root219Candidates3.length then search edge 3 [1399, 2551] root219Candidates3 3950 winner else winner) = _
  rw [if_neg (by decide)]
def root219Candidates4 : List Nat := [4987, 8461]
set_option maxRecDepth 8192 in
theorem root219Candidates4_checked : candidates edge 4 1399 3433 root219Tail5 winner.bound = root219Candidates4 := by
  unfold candidates
  have row : edge 3433 = (SortedNatTree.contains FastRows.row3433) := by
    change Data.row3433.contains = (SortedNatTree.contains FastRows.row3433)
    exact FastRows.row3433_checked.symm
  rw [row]
  decide
theorem root219Step4 : scan edge (search edge 3) 4 [1399] 1399 root219Tail4 winner =
    scan edge (search edge 3) 4 [1399] 1399 root219Tail5 winner := by
  rw [root219Tail4, scan, if_neg (by decide), root219Candidates4_checked]
  change scan edge (search edge 3) 4 [1399] 1399 root219Tail5
    (if 3 ≤ root219Candidates4.length then search edge 3 [1399, 3433] root219Candidates4 4832 winner else winner) = _
  rw [if_neg (by decide)]
def root219Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root219Candidates5_checked : candidates edge 4 1399 4261 root219Tail6 winner.bound = root219Candidates5 := by
  unfold candidates
  have row : edge 4261 = (SortedNatTree.contains FastRows.row4261) := by
    change Data.row4261.contains = (SortedNatTree.contains FastRows.row4261)
    exact FastRows.row4261_checked.symm
  rw [row]
  decide
theorem root219Step5 : scan edge (search edge 3) 4 [1399] 1399 root219Tail5 winner =
    scan edge (search edge 3) 4 [1399] 1399 root219Tail6 winner := by
  rw [root219Tail5, scan, if_neg (by decide), root219Candidates5_checked]
  change scan edge (search edge 3) 4 [1399] 1399 root219Tail6
    (if 3 ≤ root219Candidates5.length then search edge 3 [1399, 4261] root219Candidates5 5660 winner else winner) = _
  rw [if_neg (by decide)]
def root219Candidates6 : List Nat := [8461]
set_option maxRecDepth 8192 in
theorem root219Candidates6_checked : candidates edge 4 1399 4339 root219Tail7 winner.bound = root219Candidates6 := by
  unfold candidates
  have row : edge 4339 = (SortedNatTree.contains FastRows.row4339) := by
    change Data.row4339.contains = (SortedNatTree.contains FastRows.row4339)
    exact FastRows.row4339_checked.symm
  rw [row]
  decide
theorem root219Step6 : scan edge (search edge 3) 4 [1399] 1399 root219Tail6 winner =
    scan edge (search edge 3) 4 [1399] 1399 root219Tail7 winner := by
  rw [root219Tail6, scan, if_neg (by decide), root219Candidates6_checked]
  change scan edge (search edge 3) 4 [1399] 1399 root219Tail7
    (if 3 ≤ root219Candidates6.length then search edge 3 [1399, 4339] root219Candidates6 5738 winner else winner) = _
  rw [if_neg (by decide)]
def root219Candidates7 : List Nat := [8923]
set_option maxRecDepth 8192 in
theorem root219Candidates7_checked : candidates edge 4 1399 4801 root219Tail8 winner.bound = root219Candidates7 := by
  unfold candidates
  have row : edge 4801 = (SortedNatTree.contains FastRows.row4801) := by
    change Data.row4801.contains = (SortedNatTree.contains FastRows.row4801)
    exact FastRows.row4801_checked.symm
  rw [row]
  decide
theorem root219Step7 : scan edge (search edge 3) 4 [1399] 1399 root219Tail7 winner =
    scan edge (search edge 3) 4 [1399] 1399 root219Tail8 winner := by
  rw [root219Tail7, scan, if_neg (by decide), root219Candidates7_checked]
  change scan edge (search edge 3) 4 [1399] 1399 root219Tail8
    (if 3 ≤ root219Candidates7.length then search edge 3 [1399, 4801] root219Candidates7 6200 winner else winner) = _
  rw [if_neg (by decide)]
def root219Candidates8 : List Nat := [8461, 18217]
set_option maxRecDepth 8192 in
theorem root219Candidates8_checked : candidates edge 4 1399 4987 root219Tail9 winner.bound = root219Candidates8 := by
  unfold candidates
  have row : edge 4987 = (SortedNatTree.contains FastRows.row4987) := by
    change Data.row4987.contains = (SortedNatTree.contains FastRows.row4987)
    exact FastRows.row4987_checked.symm
  rw [row]
  decide
theorem root219Step8 : scan edge (search edge 3) 4 [1399] 1399 root219Tail8 winner =
    scan edge (search edge 3) 4 [1399] 1399 root219Tail9 winner := by
  rw [root219Tail8, scan, if_neg (by decide), root219Candidates8_checked]
  change scan edge (search edge 3) 4 [1399] 1399 root219Tail9
    (if 3 ≤ root219Candidates8.length then search edge 3 [1399, 4987] root219Candidates8 6386 winner else winner) = _
  rw [if_neg (by decide)]
def root219Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root219Candidates9_checked : candidates edge 4 1399 5881 root219Tail10 winner.bound = root219Candidates9 := by
  unfold candidates
  have row : edge 5881 = (SortedNatTree.contains FastRows.row5881) := by
    change Data.row5881.contains = (SortedNatTree.contains FastRows.row5881)
    exact FastRows.row5881_checked.symm
  rw [row]
  decide
theorem root219Step9 : scan edge (search edge 3) 4 [1399] 1399 root219Tail9 winner =
    scan edge (search edge 3) 4 [1399] 1399 root219Tail10 winner := by
  rw [root219Tail9, scan, if_neg (by decide), root219Candidates9_checked]
  change scan edge (search edge 3) 4 [1399] 1399 root219Tail10
    (if 3 ≤ root219Candidates9.length then search edge 3 [1399, 5881] root219Candidates9 7280 winner else winner) = _
  rw [if_neg (by decide)]
def root219Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root219Candidates10_checked : candidates edge 4 1399 6133 root219Tail11 winner.bound = root219Candidates10 := by
  unfold candidates
  have row : edge 6133 = (SortedNatTree.contains FastRows.row6133) := by
    change Data.row6133.contains = (SortedNatTree.contains FastRows.row6133)
    exact FastRows.row6133_checked.symm
  rw [row]
  decide
theorem root219Step10 : scan edge (search edge 3) 4 [1399] 1399 root219Tail10 winner =
    scan edge (search edge 3) 4 [1399] 1399 root219Tail11 winner := by
  rw [root219Tail10, scan, if_neg (by decide), root219Candidates10_checked]
  change scan edge (search edge 3) 4 [1399] 1399 root219Tail11
    (if 3 ≤ root219Candidates10.length then search edge 3 [1399, 6133] root219Candidates10 7532 winner else winner) = _
  rw [if_neg (by decide)]
theorem root219Scan11 : scan edge (search edge 3) 4 [1399] 1399 root219Tail11 winner = winner := by rw [root219Tail11, scan, if_pos (by decide)]
theorem root219Scan10 : scan edge (search edge 3) 4 [1399] 1399 root219Tail10 winner = winner :=
  (root219Step10).trans root219Scan11
theorem root219Scan9 : scan edge (search edge 3) 4 [1399] 1399 root219Tail9 winner = winner :=
  (root219Step9).trans root219Scan10
theorem root219Scan8 : scan edge (search edge 3) 4 [1399] 1399 root219Tail8 winner = winner :=
  (root219Step8).trans root219Scan9
theorem root219Scan7 : scan edge (search edge 3) 4 [1399] 1399 root219Tail7 winner = winner :=
  (root219Step7).trans root219Scan8
theorem root219Scan6 : scan edge (search edge 3) 4 [1399] 1399 root219Tail6 winner = winner :=
  (root219Step6).trans root219Scan7
theorem root219Scan5 : scan edge (search edge 3) 4 [1399] 1399 root219Tail5 winner = winner :=
  (root219Step5).trans root219Scan6
theorem root219Scan4 : scan edge (search edge 3) 4 [1399] 1399 root219Tail4 winner = winner :=
  (root219Step4).trans root219Scan5
theorem root219Scan3 : scan edge (search edge 3) 4 [1399] 1399 root219Tail3 winner = winner :=
  (root219Step3).trans root219Scan4
theorem root219Scan2 : scan edge (search edge 3) 4 [1399] 1399 root219Tail2 winner = winner :=
  (root219Step2).trans root219Scan3
theorem root219Scan1 : scan edge (search edge 3) 4 [1399] 1399 root219Tail1 winner = winner :=
  (root219Step1).trans root219Scan2
theorem root219Scan0 : scan edge (search edge 3) 4 [1399] 1399 root219Tail0 winner = winner :=
  (root219Step0).trans root219Scan1
set_option maxRecDepth 4096 in
theorem root219Branch : search edge 4 [1399] root219Candidates 1399 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1399] 1399 root219Tail0 winner = winner
  exact root219Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
