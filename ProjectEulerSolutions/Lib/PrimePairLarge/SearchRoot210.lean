import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root210Candidates : List Nat := Data.row1303
set_option maxRecDepth 8192 in
theorem root210Candidates_checked : candidates edge 5 0 1303 tail211 winner.bound = root210Candidates := by
  apply candidates_eq_row edge 5 0 1303 tail211 root210Candidates winner.bound tail211_ordered
  · decide
  · decide
  · funext q
    rfl
def root210Tail45 : List Nat := []
def root210Tail44 : List Nat := 23563 :: root210Tail45
def root210Tail43 : List Nat := 22651 :: root210Tail44
def root210Tail42 : List Nat := 20899 :: root210Tail43
def root210Tail41 : List Nat := 20479 :: root210Tail42
def root210Tail40 : List Nat := 20233 :: root210Tail41
def root210Tail39 : List Nat := 20101 :: root210Tail40
def root210Tail38 : List Nat := 19249 :: root210Tail39
def root210Tail37 : List Nat := 19207 :: root210Tail38
def root210Tail36 : List Nat := 19183 :: root210Tail37
def root210Tail35 : List Nat := 18427 :: root210Tail36
def root210Tail34 : List Nat := 17569 :: root210Tail35
def root210Tail33 : List Nat := 17203 :: root210Tail34
def root210Tail32 : List Nat := 16603 :: root210Tail33
def root210Tail31 : List Nat := 15511 :: root210Tail32
def root210Tail30 : List Nat := 14683 :: root210Tail31
def root210Tail29 : List Nat := 13537 :: root210Tail30
def root210Tail28 : List Nat := 13171 :: root210Tail29
def root210Tail27 : List Nat := 12823 :: root210Tail28
def root210Tail26 : List Nat := 12757 :: root210Tail27
def root210Tail25 : List Nat := 10429 :: root210Tail26
def root210Tail24 : List Nat := 10039 :: root210Tail25
def root210Tail23 : List Nat := 9109 :: root210Tail24
def root210Tail22 : List Nat := 8863 :: root210Tail23
def root210Tail21 : List Nat := 8647 :: root210Tail22
def root210Tail20 : List Nat := 8101 :: root210Tail21
def root210Tail19 : List Nat := 7333 :: root210Tail20
def root210Tail18 : List Nat := 7309 :: root210Tail19
def root210Tail17 : List Nat := 7207 :: root210Tail18
def root210Tail16 : List Nat := 7027 :: root210Tail17
def root210Tail15 : List Nat := 6871 :: root210Tail16
def root210Tail14 : List Nat := 6043 :: root210Tail15
def root210Tail13 : List Nat := 5443 :: root210Tail14
def root210Tail12 : List Nat := 5119 :: root210Tail13
def root210Tail11 : List Nat := 5023 :: root210Tail12
def root210Tail10 : List Nat := 4993 :: root210Tail11
def root210Tail9 : List Nat := 4759 :: root210Tail10
def root210Tail8 : List Nat := 4561 :: root210Tail9
def root210Tail7 : List Nat := 4519 :: root210Tail8
def root210Tail6 : List Nat := 4363 :: root210Tail7
def root210Tail5 : List Nat := 3469 :: root210Tail6
def root210Tail4 : List Nat := 3109 :: root210Tail5
def root210Tail3 : List Nat := 2953 :: root210Tail4
def root210Tail2 : List Nat := 2857 :: root210Tail3
def root210Tail1 : List Nat := 2143 :: root210Tail2
def root210Tail0 : List Nat := 1723 :: root210Tail1
def root210Candidates0 : List Nat := [4363, 4993, 8863, 9109]
set_option maxRecDepth 8192 in
theorem root210Candidates0_checked : candidates edge 4 1303 1723 root210Tail1 winner.bound = root210Candidates0 := by
  unfold candidates
  have row : edge 1723 = (SortedNatTree.contains FastRows.row1723) := by
    change Data.row1723.contains = (SortedNatTree.contains FastRows.row1723)
    exact FastRows.row1723_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root210Child0 : search edge 3 [1303, 1723] root210Candidates0 3026 winner = winner := by decide
theorem root210Step0 : scan edge (search edge 3) 4 [1303] 1303 root210Tail0 winner =
    scan edge (search edge 3) 4 [1303] 1303 root210Tail1 winner := by
  rw [root210Tail0, scan, if_neg (by decide), root210Candidates0_checked]
  change scan edge (search edge 3) 4 [1303] 1303 root210Tail1
    (if 3 ≤ root210Candidates0.length then search edge 3 [1303, 1723] root210Candidates0 3026 winner else winner) = _
  rw [if_pos (by decide), root210Child0]
def root210Candidates1 : List Nat := [6871, 8863, 9109]
set_option maxRecDepth 8192 in
theorem root210Candidates1_checked : candidates edge 4 1303 2143 root210Tail2 winner.bound = root210Candidates1 := by
  unfold candidates
  have row : edge 2143 = (SortedNatTree.contains FastRows.row2143) := by
    change Data.row2143.contains = (SortedNatTree.contains FastRows.row2143)
    exact FastRows.row2143_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root210Child1 : search edge 3 [1303, 2143] root210Candidates1 3446 winner = winner := by decide
theorem root210Step1 : scan edge (search edge 3) 4 [1303] 1303 root210Tail1 winner =
    scan edge (search edge 3) 4 [1303] 1303 root210Tail2 winner := by
  rw [root210Tail1, scan, if_neg (by decide), root210Candidates1_checked]
  change scan edge (search edge 3) 4 [1303] 1303 root210Tail2
    (if 3 ≤ root210Candidates1.length then search edge 3 [1303, 2143] root210Candidates1 3446 winner else winner) = _
  rw [if_pos (by decide), root210Child1]
def root210Candidates2 : List Nat := [3469, 4519, 4561]
set_option maxRecDepth 8192 in
theorem root210Candidates2_checked : candidates edge 4 1303 2857 root210Tail3 winner.bound = root210Candidates2 := by
  unfold candidates
  have row : edge 2857 = (SortedNatTree.contains FastRows.row2857) := by
    change Data.row2857.contains = (SortedNatTree.contains FastRows.row2857)
    exact FastRows.row2857_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root210Child2 : search edge 3 [1303, 2857] root210Candidates2 4160 winner = winner := by decide
theorem root210Step2 : scan edge (search edge 3) 4 [1303] 1303 root210Tail2 winner =
    scan edge (search edge 3) 4 [1303] 1303 root210Tail3 winner := by
  rw [root210Tail2, scan, if_neg (by decide), root210Candidates2_checked]
  change scan edge (search edge 3) 4 [1303] 1303 root210Tail3
    (if 3 ≤ root210Candidates2.length then search edge 3 [1303, 2857] root210Candidates2 4160 winner else winner) = _
  rw [if_pos (by decide), root210Child2]
def root210Candidates3 : List Nat := [7309, 14683, 20101]
set_option maxRecDepth 8192 in
theorem root210Candidates3_checked : candidates edge 4 1303 2953 root210Tail4 winner.bound = root210Candidates3 := by
  unfold candidates
  have row : edge 2953 = (SortedNatTree.contains FastRows.row2953) := by
    change Data.row2953.contains = (SortedNatTree.contains FastRows.row2953)
    exact FastRows.row2953_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root210Child3 : search edge 3 [1303, 2953] root210Candidates3 4256 winner = winner := by decide
theorem root210Step3 : scan edge (search edge 3) 4 [1303] 1303 root210Tail3 winner =
    scan edge (search edge 3) 4 [1303] 1303 root210Tail4 winner := by
  rw [root210Tail3, scan, if_neg (by decide), root210Candidates3_checked]
  change scan edge (search edge 3) 4 [1303] 1303 root210Tail4
    (if 3 ≤ root210Candidates3.length then search edge 3 [1303, 2953] root210Candidates3 4256 winner else winner) = _
  rw [if_pos (by decide), root210Child3]
def root210Candidates4 : List Nat := [4759, 15511]
set_option maxRecDepth 8192 in
theorem root210Candidates4_checked : candidates edge 4 1303 3109 root210Tail5 winner.bound = root210Candidates4 := by
  unfold candidates
  have row : edge 3109 = (SortedNatTree.contains FastRows.row3109) := by
    change Data.row3109.contains = (SortedNatTree.contains FastRows.row3109)
    exact FastRows.row3109_checked.symm
  rw [row]
  decide
theorem root210Step4 : scan edge (search edge 3) 4 [1303] 1303 root210Tail4 winner =
    scan edge (search edge 3) 4 [1303] 1303 root210Tail5 winner := by
  rw [root210Tail4, scan, if_neg (by decide), root210Candidates4_checked]
  change scan edge (search edge 3) 4 [1303] 1303 root210Tail5
    (if 3 ≤ root210Candidates4.length then search edge 3 [1303, 3109] root210Candidates4 4412 winner else winner) = _
  rw [if_neg (by decide)]
def root210Candidates5 : List Nat := [7309]
set_option maxRecDepth 8192 in
theorem root210Candidates5_checked : candidates edge 4 1303 3469 root210Tail6 winner.bound = root210Candidates5 := by
  unfold candidates
  have row : edge 3469 = (SortedNatTree.contains FastRows.row3469) := by
    change Data.row3469.contains = (SortedNatTree.contains FastRows.row3469)
    exact FastRows.row3469_checked.symm
  rw [row]
  decide
theorem root210Step5 : scan edge (search edge 3) 4 [1303] 1303 root210Tail5 winner =
    scan edge (search edge 3) 4 [1303] 1303 root210Tail6 winner := by
  rw [root210Tail5, scan, if_neg (by decide), root210Candidates5_checked]
  change scan edge (search edge 3) 4 [1303] 1303 root210Tail6
    (if 3 ≤ root210Candidates5.length then search edge 3 [1303, 3469] root210Candidates5 4772 winner else winner) = _
  rw [if_neg (by decide)]
def root210Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root210Candidates6_checked : candidates edge 4 1303 4363 root210Tail7 winner.bound = root210Candidates6 := by
  unfold candidates
  have row : edge 4363 = (SortedNatTree.contains FastRows.row4363) := by
    change Data.row4363.contains = (SortedNatTree.contains FastRows.row4363)
    exact FastRows.row4363_checked.symm
  rw [row]
  decide
theorem root210Step6 : scan edge (search edge 3) 4 [1303] 1303 root210Tail6 winner =
    scan edge (search edge 3) 4 [1303] 1303 root210Tail7 winner := by
  rw [root210Tail6, scan, if_neg (by decide), root210Candidates6_checked]
  change scan edge (search edge 3) 4 [1303] 1303 root210Tail7
    (if 3 ≤ root210Candidates6.length then search edge 3 [1303, 4363] root210Candidates6 5666 winner else winner) = _
  rw [if_neg (by decide)]
def root210Candidates7 : List Nat := [6043, 19183]
set_option maxRecDepth 8192 in
theorem root210Candidates7_checked : candidates edge 4 1303 4519 root210Tail8 winner.bound = root210Candidates7 := by
  unfold candidates
  have row : edge 4519 = (SortedNatTree.contains FastRows.row4519) := by
    change Data.row4519.contains = (SortedNatTree.contains FastRows.row4519)
    exact FastRows.row4519_checked.symm
  rw [row]
  decide
theorem root210Step7 : scan edge (search edge 3) 4 [1303] 1303 root210Tail7 winner =
    scan edge (search edge 3) 4 [1303] 1303 root210Tail8 winner := by
  rw [root210Tail7, scan, if_neg (by decide), root210Candidates7_checked]
  change scan edge (search edge 3) 4 [1303] 1303 root210Tail8
    (if 3 ≤ root210Candidates7.length then search edge 3 [1303, 4519] root210Candidates7 5822 winner else winner) = _
  rw [if_neg (by decide)]
def root210Candidates8 : List Nat := [8101, 8863]
set_option maxRecDepth 8192 in
theorem root210Candidates8_checked : candidates edge 4 1303 4561 root210Tail9 winner.bound = root210Candidates8 := by
  unfold candidates
  have row : edge 4561 = (SortedNatTree.contains FastRows.row4561) := by
    change Data.row4561.contains = (SortedNatTree.contains FastRows.row4561)
    exact FastRows.row4561_checked.symm
  rw [row]
  decide
theorem root210Step8 : scan edge (search edge 3) 4 [1303] 1303 root210Tail8 winner =
    scan edge (search edge 3) 4 [1303] 1303 root210Tail9 winner := by
  rw [root210Tail8, scan, if_neg (by decide), root210Candidates8_checked]
  change scan edge (search edge 3) 4 [1303] 1303 root210Tail9
    (if 3 ≤ root210Candidates8.length then search edge 3 [1303, 4561] root210Candidates8 5864 winner else winner) = _
  rw [if_neg (by decide)]
def root210Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root210Candidates9_checked : candidates edge 4 1303 4759 root210Tail10 winner.bound = root210Candidates9 := by
  unfold candidates
  have row : edge 4759 = (SortedNatTree.contains FastRows.row4759) := by
    change Data.row4759.contains = (SortedNatTree.contains FastRows.row4759)
    exact FastRows.row4759_checked.symm
  rw [row]
  decide
theorem root210Step9 : scan edge (search edge 3) 4 [1303] 1303 root210Tail9 winner =
    scan edge (search edge 3) 4 [1303] 1303 root210Tail10 winner := by
  rw [root210Tail9, scan, if_neg (by decide), root210Candidates9_checked]
  change scan edge (search edge 3) 4 [1303] 1303 root210Tail10
    (if 3 ≤ root210Candidates9.length then search edge 3 [1303, 4759] root210Candidates9 6062 winner else winner) = _
  rw [if_neg (by decide)]
def root210Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root210Candidates10_checked : candidates edge 4 1303 4993 root210Tail11 winner.bound = root210Candidates10 := by
  unfold candidates
  have row : edge 4993 = (SortedNatTree.contains FastRows.row4993) := by
    change Data.row4993.contains = (SortedNatTree.contains FastRows.row4993)
    exact FastRows.row4993_checked.symm
  rw [row]
  decide
theorem root210Step10 : scan edge (search edge 3) 4 [1303] 1303 root210Tail10 winner =
    scan edge (search edge 3) 4 [1303] 1303 root210Tail11 winner := by
  rw [root210Tail10, scan, if_neg (by decide), root210Candidates10_checked]
  change scan edge (search edge 3) 4 [1303] 1303 root210Tail11
    (if 3 ≤ root210Candidates10.length then search edge 3 [1303, 4993] root210Candidates10 6296 winner else winner) = _
  rw [if_neg (by decide)]
def root210Candidates11 : List Nat := [5443]
set_option maxRecDepth 8192 in
theorem root210Candidates11_checked : candidates edge 4 1303 5023 root210Tail12 winner.bound = root210Candidates11 := by
  unfold candidates
  have row : edge 5023 = (SortedNatTree.contains FastRows.row5023) := by
    change Data.row5023.contains = (SortedNatTree.contains FastRows.row5023)
    exact FastRows.row5023_checked.symm
  rw [row]
  decide
theorem root210Step11 : scan edge (search edge 3) 4 [1303] 1303 root210Tail11 winner =
    scan edge (search edge 3) 4 [1303] 1303 root210Tail12 winner := by
  rw [root210Tail11, scan, if_neg (by decide), root210Candidates11_checked]
  change scan edge (search edge 3) 4 [1303] 1303 root210Tail12
    (if 3 ≤ root210Candidates11.length then search edge 3 [1303, 5023] root210Candidates11 6326 winner else winner) = _
  rw [if_neg (by decide)]
def root210Candidates12 : List Nat := [7207, 8647]
set_option maxRecDepth 8192 in
theorem root210Candidates12_checked : candidates edge 4 1303 5119 root210Tail13 winner.bound = root210Candidates12 := by
  unfold candidates
  have row : edge 5119 = (SortedNatTree.contains FastRows.row5119) := by
    change Data.row5119.contains = (SortedNatTree.contains FastRows.row5119)
    exact FastRows.row5119_checked.symm
  rw [row]
  decide
theorem root210Step12 : scan edge (search edge 3) 4 [1303] 1303 root210Tail12 winner =
    scan edge (search edge 3) 4 [1303] 1303 root210Tail13 winner := by
  rw [root210Tail12, scan, if_neg (by decide), root210Candidates12_checked]
  change scan edge (search edge 3) 4 [1303] 1303 root210Tail13
    (if 3 ≤ root210Candidates12.length then search edge 3 [1303, 5119] root210Candidates12 6422 winner else winner) = _
  rw [if_neg (by decide)]
def root210Candidates13 : List Nat := [8101]
set_option maxRecDepth 8192 in
theorem root210Candidates13_checked : candidates edge 4 1303 5443 root210Tail14 winner.bound = root210Candidates13 := by
  unfold candidates
  have row : edge 5443 = (SortedNatTree.contains FastRows.row5443) := by
    change Data.row5443.contains = (SortedNatTree.contains FastRows.row5443)
    exact FastRows.row5443_checked.symm
  rw [row]
  decide
theorem root210Step13 : scan edge (search edge 3) 4 [1303] 1303 root210Tail13 winner =
    scan edge (search edge 3) 4 [1303] 1303 root210Tail14 winner := by
  rw [root210Tail13, scan, if_neg (by decide), root210Candidates13_checked]
  change scan edge (search edge 3) 4 [1303] 1303 root210Tail14
    (if 3 ≤ root210Candidates13.length then search edge 3 [1303, 5443] root210Candidates13 6746 winner else winner) = _
  rw [if_neg (by decide)]
def root210Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root210Candidates14_checked : candidates edge 4 1303 6043 root210Tail15 winner.bound = root210Candidates14 := by
  unfold candidates
  have row : edge 6043 = (SortedNatTree.contains FastRows.row6043) := by
    change Data.row6043.contains = (SortedNatTree.contains FastRows.row6043)
    exact FastRows.row6043_checked.symm
  rw [row]
  decide
theorem root210Step14 : scan edge (search edge 3) 4 [1303] 1303 root210Tail14 winner =
    scan edge (search edge 3) 4 [1303] 1303 root210Tail15 winner := by
  rw [root210Tail14, scan, if_neg (by decide), root210Candidates14_checked]
  change scan edge (search edge 3) 4 [1303] 1303 root210Tail15
    (if 3 ≤ root210Candidates14.length then search edge 3 [1303, 6043] root210Candidates14 7346 winner else winner) = _
  rw [if_neg (by decide)]
theorem root210Scan15 : scan edge (search edge 3) 4 [1303] 1303 root210Tail15 winner = winner := by rw [root210Tail15, scan, if_pos (by decide)]
theorem root210Scan14 : scan edge (search edge 3) 4 [1303] 1303 root210Tail14 winner = winner :=
  (root210Step14).trans root210Scan15
theorem root210Scan13 : scan edge (search edge 3) 4 [1303] 1303 root210Tail13 winner = winner :=
  (root210Step13).trans root210Scan14
theorem root210Scan12 : scan edge (search edge 3) 4 [1303] 1303 root210Tail12 winner = winner :=
  (root210Step12).trans root210Scan13
theorem root210Scan11 : scan edge (search edge 3) 4 [1303] 1303 root210Tail11 winner = winner :=
  (root210Step11).trans root210Scan12
theorem root210Scan10 : scan edge (search edge 3) 4 [1303] 1303 root210Tail10 winner = winner :=
  (root210Step10).trans root210Scan11
theorem root210Scan9 : scan edge (search edge 3) 4 [1303] 1303 root210Tail9 winner = winner :=
  (root210Step9).trans root210Scan10
theorem root210Scan8 : scan edge (search edge 3) 4 [1303] 1303 root210Tail8 winner = winner :=
  (root210Step8).trans root210Scan9
theorem root210Scan7 : scan edge (search edge 3) 4 [1303] 1303 root210Tail7 winner = winner :=
  (root210Step7).trans root210Scan8
theorem root210Scan6 : scan edge (search edge 3) 4 [1303] 1303 root210Tail6 winner = winner :=
  (root210Step6).trans root210Scan7
theorem root210Scan5 : scan edge (search edge 3) 4 [1303] 1303 root210Tail5 winner = winner :=
  (root210Step5).trans root210Scan6
theorem root210Scan4 : scan edge (search edge 3) 4 [1303] 1303 root210Tail4 winner = winner :=
  (root210Step4).trans root210Scan5
theorem root210Scan3 : scan edge (search edge 3) 4 [1303] 1303 root210Tail3 winner = winner :=
  (root210Step3).trans root210Scan4
theorem root210Scan2 : scan edge (search edge 3) 4 [1303] 1303 root210Tail2 winner = winner :=
  (root210Step2).trans root210Scan3
theorem root210Scan1 : scan edge (search edge 3) 4 [1303] 1303 root210Tail1 winner = winner :=
  (root210Step1).trans root210Scan2
theorem root210Scan0 : scan edge (search edge 3) 4 [1303] 1303 root210Tail0 winner = winner :=
  (root210Step0).trans root210Scan1
set_option maxRecDepth 4096 in
theorem root210Branch : search edge 4 [1303] root210Candidates 1303 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1303] 1303 root210Tail0 winner = winner
  exact root210Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
