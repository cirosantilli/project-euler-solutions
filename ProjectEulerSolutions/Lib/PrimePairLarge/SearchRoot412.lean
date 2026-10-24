import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root412Candidates : List Nat := Data.row2857
set_option maxRecDepth 8192 in
theorem root412Candidates_checked : candidates edge 5 0 2857 tail413 winner.bound = root412Candidates := by
  apply candidates_eq_row edge 5 0 2857 tail413 root412Candidates winner.bound tail413_ordered
  · decide
  · decide
  · funext q
    rfl
def root412Tail44 : List Nat := []
def root412Tail43 : List Nat := 22369 :: root412Tail44
def root412Tail42 : List Nat := 22051 :: root412Tail43
def root412Tail41 : List Nat := 21991 :: root412Tail42
def root412Tail40 : List Nat := 21727 :: root412Tail41
def root412Tail39 : List Nat := 20857 :: root412Tail40
def root412Tail38 : List Nat := 19543 :: root412Tail39
def root412Tail37 : List Nat := 19009 :: root412Tail38
def root412Tail36 : List Nat := 18787 :: root412Tail37
def root412Tail35 : List Nat := 16447 :: root412Tail36
def root412Tail34 : List Nat := 16369 :: root412Tail35
def root412Tail33 : List Nat := 15919 :: root412Tail34
def root412Tail32 : List Nat := 15907 :: root412Tail33
def root412Tail31 : List Nat := 15427 :: root412Tail32
def root412Tail30 : List Nat := 14983 :: root412Tail31
def root412Tail29 : List Nat := 14389 :: root412Tail30
def root412Tail28 : List Nat := 14071 :: root412Tail29
def root412Tail27 : List Nat := 13759 :: root412Tail28
def root412Tail26 : List Nat := 13693 :: root412Tail27
def root412Tail25 : List Nat := 12841 :: root412Tail26
def root412Tail24 : List Nat := 12781 :: root412Tail25
def root412Tail23 : List Nat := 12109 :: root412Tail24
def root412Tail22 : List Nat := 11677 :: root412Tail23
def root412Tail21 : List Nat := 10861 :: root412Tail22
def root412Tail20 : List Nat := 10111 :: root412Tail21
def root412Tail19 : List Nat := 9043 :: root412Tail20
def root412Tail18 : List Nat := 8941 :: root412Tail19
def root412Tail17 : List Nat := 7933 :: root412Tail18
def root412Tail16 : List Nat := 7699 :: root412Tail17
def root412Tail15 : List Nat := 7573 :: root412Tail16
def root412Tail14 : List Nat := 6091 :: root412Tail15
def root412Tail13 : List Nat := 5881 :: root412Tail14
def root412Tail12 : List Nat := 5641 :: root412Tail13
def root412Tail11 : List Nat := 4957 :: root412Tail12
def root412Tail10 : List Nat := 4783 :: root412Tail11
def root412Tail9 : List Nat := 4561 :: root412Tail10
def root412Tail8 : List Nat := 4549 :: root412Tail9
def root412Tail7 : List Nat := 4519 :: root412Tail8
def root412Tail6 : List Nat := 3907 :: root412Tail7
def root412Tail5 : List Nat := 3709 :: root412Tail6
def root412Tail4 : List Nat := 3499 :: root412Tail5
def root412Tail3 : List Nat := 3469 :: root412Tail4
def root412Tail2 : List Nat := 3229 :: root412Tail3
def root412Tail1 : List Nat := 3163 :: root412Tail2
def root412Tail0 : List Nat := 3049 :: root412Tail1
def root412Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root412Candidates0_checked : candidates edge 4 2857 3049 root412Tail1 winner.bound = root412Candidates0 := by
  unfold candidates
  have row : edge 3049 = (SortedNatTree.contains FastRows.row3049) := by
    change Data.row3049.contains = (SortedNatTree.contains FastRows.row3049)
    exact FastRows.row3049_checked.symm
  rw [row]
  decide
theorem root412Step0 : scan edge (search edge 3) 4 [2857] 2857 root412Tail0 winner =
    scan edge (search edge 3) 4 [2857] 2857 root412Tail1 winner := by
  rw [root412Tail0, scan, if_neg (by decide), root412Candidates0_checked]
  change scan edge (search edge 3) 4 [2857] 2857 root412Tail1
    (if 3 ≤ root412Candidates0.length then search edge 3 [2857, 3049] root412Candidates0 5906 winner else winner) = _
  rw [if_neg (by decide)]
def root412Candidates1 : List Nat := [9043, 10861]
set_option maxRecDepth 8192 in
theorem root412Candidates1_checked : candidates edge 4 2857 3163 root412Tail2 winner.bound = root412Candidates1 := by
  unfold candidates
  have row : edge 3163 = (SortedNatTree.contains FastRows.row3163) := by
    change Data.row3163.contains = (SortedNatTree.contains FastRows.row3163)
    exact FastRows.row3163_checked.symm
  rw [row]
  decide
theorem root412Step1 : scan edge (search edge 3) 4 [2857] 2857 root412Tail1 winner =
    scan edge (search edge 3) 4 [2857] 2857 root412Tail2 winner := by
  rw [root412Tail1, scan, if_neg (by decide), root412Candidates1_checked]
  change scan edge (search edge 3) 4 [2857] 2857 root412Tail2
    (if 3 ≤ root412Candidates1.length then search edge 3 [2857, 3163] root412Candidates1 6020 winner else winner) = _
  rw [if_neg (by decide)]
def root412Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root412Candidates2_checked : candidates edge 4 2857 3229 root412Tail3 winner.bound = root412Candidates2 := by
  unfold candidates
  have row : edge 3229 = (SortedNatTree.contains FastRows.row3229) := by
    change Data.row3229.contains = (SortedNatTree.contains FastRows.row3229)
    exact FastRows.row3229_checked.symm
  rw [row]
  decide
theorem root412Step2 : scan edge (search edge 3) 4 [2857] 2857 root412Tail2 winner =
    scan edge (search edge 3) 4 [2857] 2857 root412Tail3 winner := by
  rw [root412Tail2, scan, if_neg (by decide), root412Candidates2_checked]
  change scan edge (search edge 3) 4 [2857] 2857 root412Tail3
    (if 3 ≤ root412Candidates2.length then search edge 3 [2857, 3229] root412Candidates2 6086 winner else winner) = _
  rw [if_neg (by decide)]
def root412Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root412Candidates3_checked : candidates edge 4 2857 3469 root412Tail4 winner.bound = root412Candidates3 := by
  unfold candidates
  have row : edge 3469 = (SortedNatTree.contains FastRows.row3469) := by
    change Data.row3469.contains = (SortedNatTree.contains FastRows.row3469)
    exact FastRows.row3469_checked.symm
  rw [row]
  decide
theorem root412Step3 : scan edge (search edge 3) 4 [2857] 2857 root412Tail3 winner =
    scan edge (search edge 3) 4 [2857] 2857 root412Tail4 winner := by
  rw [root412Tail3, scan, if_neg (by decide), root412Candidates3_checked]
  change scan edge (search edge 3) 4 [2857] 2857 root412Tail4
    (if 3 ≤ root412Candidates3.length then search edge 3 [2857, 3469] root412Candidates3 6326 winner else winner) = _
  rw [if_neg (by decide)]
def root412Candidates4 : List Nat := [5881]
set_option maxRecDepth 8192 in
theorem root412Candidates4_checked : candidates edge 4 2857 3499 root412Tail5 winner.bound = root412Candidates4 := by
  unfold candidates
  have row : edge 3499 = (SortedNatTree.contains FastRows.row3499) := by
    change Data.row3499.contains = (SortedNatTree.contains FastRows.row3499)
    exact FastRows.row3499_checked.symm
  rw [row]
  decide
theorem root412Step4 : scan edge (search edge 3) 4 [2857] 2857 root412Tail4 winner =
    scan edge (search edge 3) 4 [2857] 2857 root412Tail5 winner := by
  rw [root412Tail4, scan, if_neg (by decide), root412Candidates4_checked]
  change scan edge (search edge 3) 4 [2857] 2857 root412Tail5
    (if 3 ≤ root412Candidates4.length then search edge 3 [2857, 3499] root412Candidates4 6356 winner else winner) = _
  rw [if_neg (by decide)]
def root412Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root412Candidates5_checked : candidates edge 4 2857 3709 root412Tail6 winner.bound = root412Candidates5 := by
  unfold candidates
  have row : edge 3709 = (SortedNatTree.contains FastRows.row3709) := by
    change Data.row3709.contains = (SortedNatTree.contains FastRows.row3709)
    exact FastRows.row3709_checked.symm
  rw [row]
  decide
theorem root412Step5 : scan edge (search edge 3) 4 [2857] 2857 root412Tail5 winner =
    scan edge (search edge 3) 4 [2857] 2857 root412Tail6 winner := by
  rw [root412Tail5, scan, if_neg (by decide), root412Candidates5_checked]
  change scan edge (search edge 3) 4 [2857] 2857 root412Tail6
    (if 3 ≤ root412Candidates5.length then search edge 3 [2857, 3709] root412Candidates5 6566 winner else winner) = _
  rw [if_neg (by decide)]
def root412Candidates6 : List Nat := [7573, 15427]
set_option maxRecDepth 8192 in
theorem root412Candidates6_checked : candidates edge 4 2857 3907 root412Tail7 winner.bound = root412Candidates6 := by
  unfold candidates
  have row : edge 3907 = (SortedNatTree.contains FastRows.row3907) := by
    change Data.row3907.contains = (SortedNatTree.contains FastRows.row3907)
    exact FastRows.row3907_checked.symm
  rw [row]
  decide
theorem root412Step6 : scan edge (search edge 3) 4 [2857] 2857 root412Tail6 winner =
    scan edge (search edge 3) 4 [2857] 2857 root412Tail7 winner := by
  rw [root412Tail6, scan, if_neg (by decide), root412Candidates6_checked]
  change scan edge (search edge 3) 4 [2857] 2857 root412Tail7
    (if 3 ≤ root412Candidates6.length then search edge 3 [2857, 3907] root412Candidates6 6764 winner else winner) = _
  rw [if_neg (by decide)]
def root412Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root412Candidates7_checked : candidates edge 4 2857 4519 root412Tail8 winner.bound = root412Candidates7 := by
  unfold candidates
  have row : edge 4519 = (SortedNatTree.contains FastRows.row4519) := by
    change Data.row4519.contains = (SortedNatTree.contains FastRows.row4519)
    exact FastRows.row4519_checked.symm
  rw [row]
  decide
theorem root412Step7 : scan edge (search edge 3) 4 [2857] 2857 root412Tail7 winner =
    scan edge (search edge 3) 4 [2857] 2857 root412Tail8 winner := by
  rw [root412Tail7, scan, if_neg (by decide), root412Candidates7_checked]
  change scan edge (search edge 3) 4 [2857] 2857 root412Tail8
    (if 3 ≤ root412Candidates7.length then search edge 3 [2857, 4519] root412Candidates7 7376 winner else winner) = _
  rw [if_neg (by decide)]
def root412Candidates8 : List Nat := [12109]
set_option maxRecDepth 8192 in
theorem root412Candidates8_checked : candidates edge 4 2857 4549 root412Tail9 winner.bound = root412Candidates8 := by
  unfold candidates
  have row : edge 4549 = (SortedNatTree.contains FastRows.row4549) := by
    change Data.row4549.contains = (SortedNatTree.contains FastRows.row4549)
    exact FastRows.row4549_checked.symm
  rw [row]
  decide
theorem root412Step8 : scan edge (search edge 3) 4 [2857] 2857 root412Tail8 winner =
    scan edge (search edge 3) 4 [2857] 2857 root412Tail9 winner := by
  rw [root412Tail8, scan, if_neg (by decide), root412Candidates8_checked]
  change scan edge (search edge 3) 4 [2857] 2857 root412Tail9
    (if 3 ≤ root412Candidates8.length then search edge 3 [2857, 4549] root412Candidates8 7406 winner else winner) = _
  rw [if_neg (by decide)]
def root412Candidates9 : List Nat := [4783]
set_option maxRecDepth 8192 in
theorem root412Candidates9_checked : candidates edge 4 2857 4561 root412Tail10 winner.bound = root412Candidates9 := by
  unfold candidates
  have row : edge 4561 = (SortedNatTree.contains FastRows.row4561) := by
    change Data.row4561.contains = (SortedNatTree.contains FastRows.row4561)
    exact FastRows.row4561_checked.symm
  rw [row]
  decide
theorem root412Step9 : scan edge (search edge 3) 4 [2857] 2857 root412Tail9 winner =
    scan edge (search edge 3) 4 [2857] 2857 root412Tail10 winner := by
  rw [root412Tail9, scan, if_neg (by decide), root412Candidates9_checked]
  change scan edge (search edge 3) 4 [2857] 2857 root412Tail10
    (if 3 ≤ root412Candidates9.length then search edge 3 [2857, 4561] root412Candidates9 7418 winner else winner) = _
  rw [if_neg (by decide)]
def root412Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root412Candidates10_checked : candidates edge 4 2857 4783 root412Tail11 winner.bound = root412Candidates10 := by
  unfold candidates
  have row : edge 4783 = (SortedNatTree.contains FastRows.row4783) := by
    change Data.row4783.contains = (SortedNatTree.contains FastRows.row4783)
    exact FastRows.row4783_checked.symm
  rw [row]
  decide
theorem root412Step10 : scan edge (search edge 3) 4 [2857] 2857 root412Tail10 winner =
    scan edge (search edge 3) 4 [2857] 2857 root412Tail11 winner := by
  rw [root412Tail10, scan, if_neg (by decide), root412Candidates10_checked]
  change scan edge (search edge 3) 4 [2857] 2857 root412Tail11
    (if 3 ≤ root412Candidates10.length then search edge 3 [2857, 4783] root412Candidates10 7640 winner else winner) = _
  rw [if_neg (by decide)]
def root412Candidates11 : List Nat := [15919]
set_option maxRecDepth 8192 in
theorem root412Candidates11_checked : candidates edge 4 2857 4957 root412Tail12 winner.bound = root412Candidates11 := by
  unfold candidates
  have row : edge 4957 = (SortedNatTree.contains FastRows.row4957) := by
    change Data.row4957.contains = (SortedNatTree.contains FastRows.row4957)
    exact FastRows.row4957_checked.symm
  rw [row]
  decide
theorem root412Step11 : scan edge (search edge 3) 4 [2857] 2857 root412Tail11 winner =
    scan edge (search edge 3) 4 [2857] 2857 root412Tail12 winner := by
  rw [root412Tail11, scan, if_neg (by decide), root412Candidates11_checked]
  change scan edge (search edge 3) 4 [2857] 2857 root412Tail12
    (if 3 ≤ root412Candidates11.length then search edge 3 [2857, 4957] root412Candidates11 7814 winner else winner) = _
  rw [if_neg (by decide)]
def root412Candidates12 : List Nat := [15907]
set_option maxRecDepth 8192 in
theorem root412Candidates12_checked : candidates edge 4 2857 5641 root412Tail13 winner.bound = root412Candidates12 := by
  unfold candidates
  have row : edge 5641 = (SortedNatTree.contains FastRows.row5641) := by
    change Data.row5641.contains = (SortedNatTree.contains FastRows.row5641)
    exact FastRows.row5641_checked.symm
  rw [row]
  decide
theorem root412Step12 : scan edge (search edge 3) 4 [2857] 2857 root412Tail12 winner =
    scan edge (search edge 3) 4 [2857] 2857 root412Tail13 winner := by
  rw [root412Tail12, scan, if_neg (by decide), root412Candidates12_checked]
  change scan edge (search edge 3) 4 [2857] 2857 root412Tail13
    (if 3 ≤ root412Candidates12.length then search edge 3 [2857, 5641] root412Candidates12 8498 winner else winner) = _
  rw [if_neg (by decide)]
theorem root412Scan13 : scan edge (search edge 3) 4 [2857] 2857 root412Tail13 winner = winner := by rw [root412Tail13, scan, if_pos (by decide)]
theorem root412Scan12 : scan edge (search edge 3) 4 [2857] 2857 root412Tail12 winner = winner :=
  (root412Step12).trans root412Scan13
theorem root412Scan11 : scan edge (search edge 3) 4 [2857] 2857 root412Tail11 winner = winner :=
  (root412Step11).trans root412Scan12
theorem root412Scan10 : scan edge (search edge 3) 4 [2857] 2857 root412Tail10 winner = winner :=
  (root412Step10).trans root412Scan11
theorem root412Scan9 : scan edge (search edge 3) 4 [2857] 2857 root412Tail9 winner = winner :=
  (root412Step9).trans root412Scan10
theorem root412Scan8 : scan edge (search edge 3) 4 [2857] 2857 root412Tail8 winner = winner :=
  (root412Step8).trans root412Scan9
theorem root412Scan7 : scan edge (search edge 3) 4 [2857] 2857 root412Tail7 winner = winner :=
  (root412Step7).trans root412Scan8
theorem root412Scan6 : scan edge (search edge 3) 4 [2857] 2857 root412Tail6 winner = winner :=
  (root412Step6).trans root412Scan7
theorem root412Scan5 : scan edge (search edge 3) 4 [2857] 2857 root412Tail5 winner = winner :=
  (root412Step5).trans root412Scan6
theorem root412Scan4 : scan edge (search edge 3) 4 [2857] 2857 root412Tail4 winner = winner :=
  (root412Step4).trans root412Scan5
theorem root412Scan3 : scan edge (search edge 3) 4 [2857] 2857 root412Tail3 winner = winner :=
  (root412Step3).trans root412Scan4
theorem root412Scan2 : scan edge (search edge 3) 4 [2857] 2857 root412Tail2 winner = winner :=
  (root412Step2).trans root412Scan3
theorem root412Scan1 : scan edge (search edge 3) 4 [2857] 2857 root412Tail1 winner = winner :=
  (root412Step1).trans root412Scan2
theorem root412Scan0 : scan edge (search edge 3) 4 [2857] 2857 root412Tail0 winner = winner :=
  (root412Step0).trans root412Scan1
set_option maxRecDepth 4096 in
theorem root412Branch : search edge 4 [2857] root412Candidates 2857 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2857] 2857 root412Tail0 winner = winner
  exact root412Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
