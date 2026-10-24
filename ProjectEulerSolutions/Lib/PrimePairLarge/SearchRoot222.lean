import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root222Candidates : List Nat := Data.row1427
set_option maxRecDepth 8192 in
theorem root222Candidates_checked : candidates edge 5 0 1427 tail223 winner.bound = root222Candidates := by
  apply candidates_eq_row edge 5 0 1427 tail223 root222Candidates winner.bound tail223_ordered
  · decide
  · decide
  · funext q
    rfl
def root222Tail49 : List Nat := []
def root222Tail48 : List Nat := 22469 :: root222Tail49
def root222Tail47 : List Nat := 22397 :: root222Tail48
def root222Tail46 : List Nat := 22247 :: root222Tail47
def root222Tail45 : List Nat := 22193 :: root222Tail46
def root222Tail44 : List Nat := 22133 :: root222Tail45
def root222Tail43 : List Nat := 22037 :: root222Tail44
def root222Tail42 : List Nat := 21881 :: root222Tail43
def root222Tail41 : List Nat := 19961 :: root222Tail42
def root222Tail40 : List Nat := 19919 :: root222Tail41
def root222Tail39 : List Nat := 19793 :: root222Tail40
def root222Tail38 : List Nat := 19559 :: root222Tail39
def root222Tail37 : List Nat := 18413 :: root222Tail38
def root222Tail36 : List Nat := 18119 :: root222Tail37
def root222Tail35 : List Nat := 17909 :: root222Tail36
def root222Tail34 : List Nat := 17321 :: root222Tail35
def root222Tail33 : List Nat := 14753 :: root222Tail34
def root222Tail32 : List Nat := 14519 :: root222Tail33
def root222Tail31 : List Nat := 13913 :: root222Tail32
def root222Tail30 : List Nat := 13829 :: root222Tail31
def root222Tail29 : List Nat := 13721 :: root222Tail30
def root222Tail28 : List Nat := 13679 :: root222Tail29
def root222Tail27 : List Nat := 12659 :: root222Tail28
def root222Tail26 : List Nat := 12197 :: root222Tail27
def root222Tail25 : List Nat := 11903 :: root222Tail26
def root222Tail24 : List Nat := 11213 :: root222Tail25
def root222Tail23 : List Nat := 10151 :: root222Tail24
def root222Tail22 : List Nat := 10091 :: root222Tail23
def root222Tail21 : List Nat := 9479 :: root222Tail22
def root222Tail20 : List Nat := 9341 :: root222Tail21
def root222Tail19 : List Nat := 9323 :: root222Tail20
def root222Tail18 : List Nat := 8807 :: root222Tail19
def root222Tail17 : List Nat := 8273 :: root222Tail18
def root222Tail16 : List Nat := 8111 :: root222Tail17
def root222Tail15 : List Nat := 7937 :: root222Tail16
def root222Tail14 : List Nat := 7451 :: root222Tail15
def root222Tail13 : List Nat := 5843 :: root222Tail14
def root222Tail12 : List Nat := 5501 :: root222Tail13
def root222Tail11 : List Nat := 5477 :: root222Tail12
def root222Tail10 : List Nat := 5297 :: root222Tail11
def root222Tail9 : List Nat := 4451 :: root222Tail10
def root222Tail8 : List Nat := 4157 :: root222Tail9
def root222Tail7 : List Nat := 4019 :: root222Tail8
def root222Tail6 : List Nat := 3989 :: root222Tail7
def root222Tail5 : List Nat := 3767 :: root222Tail6
def root222Tail4 : List Nat := 3359 :: root222Tail5
def root222Tail3 : List Nat := 3011 :: root222Tail4
def root222Tail2 : List Nat := 2777 :: root222Tail3
def root222Tail1 : List Nat := 2339 :: root222Tail2
def root222Tail0 : List Nat := 1931 :: root222Tail1
def root222Candidates0 : List Nat := [5297, 5477, 11213]
set_option maxRecDepth 8192 in
theorem root222Candidates0_checked : candidates edge 4 1427 1931 root222Tail1 winner.bound = root222Candidates0 := by
  unfold candidates
  have row : edge 1931 = (SortedNatTree.contains FastRows.row1931) := by
    change Data.row1931.contains = (SortedNatTree.contains FastRows.row1931)
    exact FastRows.row1931_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root222Child0 : search edge 3 [1427, 1931] root222Candidates0 3358 winner = winner := by decide
theorem root222Step0 : scan edge (search edge 3) 4 [1427] 1427 root222Tail0 winner =
    scan edge (search edge 3) 4 [1427] 1427 root222Tail1 winner := by
  rw [root222Tail0, scan, if_neg (by decide), root222Candidates0_checked]
  change scan edge (search edge 3) 4 [1427] 1427 root222Tail1
    (if 3 ≤ root222Candidates0.length then search edge 3 [1427, 1931] root222Candidates0 3358 winner else winner) = _
  rw [if_pos (by decide), root222Child0]
def root222Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root222Candidates1_checked : candidates edge 4 1427 2339 root222Tail2 winner.bound = root222Candidates1 := by
  unfold candidates
  have row : edge 2339 = (SortedNatTree.contains FastRows.row2339) := by
    change Data.row2339.contains = (SortedNatTree.contains FastRows.row2339)
    exact FastRows.row2339_checked.symm
  rw [row]
  decide
theorem root222Step1 : scan edge (search edge 3) 4 [1427] 1427 root222Tail1 winner =
    scan edge (search edge 3) 4 [1427] 1427 root222Tail2 winner := by
  rw [root222Tail1, scan, if_neg (by decide), root222Candidates1_checked]
  change scan edge (search edge 3) 4 [1427] 1427 root222Tail2
    (if 3 ≤ root222Candidates1.length then search edge 3 [1427, 2339] root222Candidates1 3766 winner else winner) = _
  rw [if_neg (by decide)]
def root222Candidates2 : List Nat := [5477]
set_option maxRecDepth 8192 in
theorem root222Candidates2_checked : candidates edge 4 1427 2777 root222Tail3 winner.bound = root222Candidates2 := by
  unfold candidates
  have row : edge 2777 = (SortedNatTree.contains FastRows.row2777) := by
    change Data.row2777.contains = (SortedNatTree.contains FastRows.row2777)
    exact FastRows.row2777_checked.symm
  rw [row]
  decide
theorem root222Step2 : scan edge (search edge 3) 4 [1427] 1427 root222Tail2 winner =
    scan edge (search edge 3) 4 [1427] 1427 root222Tail3 winner := by
  rw [root222Tail2, scan, if_neg (by decide), root222Candidates2_checked]
  change scan edge (search edge 3) 4 [1427] 1427 root222Tail3
    (if 3 ≤ root222Candidates2.length then search edge 3 [1427, 2777] root222Candidates2 4204 winner else winner) = _
  rw [if_neg (by decide)]
def root222Candidates3 : List Nat := [7937, 10151]
set_option maxRecDepth 8192 in
theorem root222Candidates3_checked : candidates edge 4 1427 3011 root222Tail4 winner.bound = root222Candidates3 := by
  unfold candidates
  have row : edge 3011 = (SortedNatTree.contains FastRows.row3011) := by
    change Data.row3011.contains = (SortedNatTree.contains FastRows.row3011)
    exact FastRows.row3011_checked.symm
  rw [row]
  decide
theorem root222Step3 : scan edge (search edge 3) 4 [1427] 1427 root222Tail3 winner =
    scan edge (search edge 3) 4 [1427] 1427 root222Tail4 winner := by
  rw [root222Tail3, scan, if_neg (by decide), root222Candidates3_checked]
  change scan edge (search edge 3) 4 [1427] 1427 root222Tail4
    (if 3 ≤ root222Candidates3.length then search edge 3 [1427, 3011] root222Candidates3 4438 winner else winner) = _
  rw [if_neg (by decide)]
def root222Candidates4 : List Nat := [12659]
set_option maxRecDepth 8192 in
theorem root222Candidates4_checked : candidates edge 4 1427 3359 root222Tail5 winner.bound = root222Candidates4 := by
  unfold candidates
  have row : edge 3359 = (SortedNatTree.contains FastRows.row3359) := by
    change Data.row3359.contains = (SortedNatTree.contains FastRows.row3359)
    exact FastRows.row3359_checked.symm
  rw [row]
  decide
theorem root222Step4 : scan edge (search edge 3) 4 [1427] 1427 root222Tail4 winner =
    scan edge (search edge 3) 4 [1427] 1427 root222Tail5 winner := by
  rw [root222Tail4, scan, if_neg (by decide), root222Candidates4_checked]
  change scan edge (search edge 3) 4 [1427] 1427 root222Tail5
    (if 3 ≤ root222Candidates4.length then search edge 3 [1427, 3359] root222Candidates4 4786 winner else winner) = _
  rw [if_neg (by decide)]
def root222Candidates5 : List Nat := [19793]
set_option maxRecDepth 8192 in
theorem root222Candidates5_checked : candidates edge 4 1427 3767 root222Tail6 winner.bound = root222Candidates5 := by
  unfold candidates
  have row : edge 3767 = (SortedNatTree.contains FastRows.row3767) := by
    change Data.row3767.contains = (SortedNatTree.contains FastRows.row3767)
    exact FastRows.row3767_checked.symm
  rw [row]
  decide
theorem root222Step5 : scan edge (search edge 3) 4 [1427] 1427 root222Tail5 winner =
    scan edge (search edge 3) 4 [1427] 1427 root222Tail6 winner := by
  rw [root222Tail5, scan, if_neg (by decide), root222Candidates5_checked]
  change scan edge (search edge 3) 4 [1427] 1427 root222Tail6
    (if 3 ≤ root222Candidates5.length then search edge 3 [1427, 3767] root222Candidates5 5194 winner else winner) = _
  rw [if_neg (by decide)]
def root222Candidates6 : List Nat := [4157]
set_option maxRecDepth 8192 in
theorem root222Candidates6_checked : candidates edge 4 1427 3989 root222Tail7 winner.bound = root222Candidates6 := by
  unfold candidates
  have row : edge 3989 = (SortedNatTree.contains FastRows.row3989) := by
    change Data.row3989.contains = (SortedNatTree.contains FastRows.row3989)
    exact FastRows.row3989_checked.symm
  rw [row]
  decide
theorem root222Step6 : scan edge (search edge 3) 4 [1427] 1427 root222Tail6 winner =
    scan edge (search edge 3) 4 [1427] 1427 root222Tail7 winner := by
  rw [root222Tail6, scan, if_neg (by decide), root222Candidates6_checked]
  change scan edge (search edge 3) 4 [1427] 1427 root222Tail7
    (if 3 ≤ root222Candidates6.length then search edge 3 [1427, 3989] root222Candidates6 5416 winner else winner) = _
  rw [if_neg (by decide)]
def root222Candidates7 : List Nat := [9479]
set_option maxRecDepth 8192 in
theorem root222Candidates7_checked : candidates edge 4 1427 4019 root222Tail8 winner.bound = root222Candidates7 := by
  unfold candidates
  have row : edge 4019 = (SortedNatTree.contains FastRows.row4019) := by
    change Data.row4019.contains = (SortedNatTree.contains FastRows.row4019)
    exact FastRows.row4019_checked.symm
  rw [row]
  decide
theorem root222Step7 : scan edge (search edge 3) 4 [1427] 1427 root222Tail7 winner =
    scan edge (search edge 3) 4 [1427] 1427 root222Tail8 winner := by
  rw [root222Tail7, scan, if_neg (by decide), root222Candidates7_checked]
  change scan edge (search edge 3) 4 [1427] 1427 root222Tail8
    (if 3 ≤ root222Candidates7.length then search edge 3 [1427, 4019] root222Candidates7 5446 winner else winner) = _
  rw [if_neg (by decide)]
def root222Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root222Candidates8_checked : candidates edge 4 1427 4157 root222Tail9 winner.bound = root222Candidates8 := by
  unfold candidates
  have row : edge 4157 = (SortedNatTree.contains FastRows.row4157) := by
    change Data.row4157.contains = (SortedNatTree.contains FastRows.row4157)
    exact FastRows.row4157_checked.symm
  rw [row]
  decide
theorem root222Step8 : scan edge (search edge 3) 4 [1427] 1427 root222Tail8 winner =
    scan edge (search edge 3) 4 [1427] 1427 root222Tail9 winner := by
  rw [root222Tail8, scan, if_neg (by decide), root222Candidates8_checked]
  change scan edge (search edge 3) 4 [1427] 1427 root222Tail9
    (if 3 ≤ root222Candidates8.length then search edge 3 [1427, 4157] root222Candidates8 5584 winner else winner) = _
  rw [if_neg (by decide)]
def root222Candidates9 : List Nat := [14753, 17909]
set_option maxRecDepth 8192 in
theorem root222Candidates9_checked : candidates edge 4 1427 4451 root222Tail10 winner.bound = root222Candidates9 := by
  unfold candidates
  have row : edge 4451 = (SortedNatTree.contains FastRows.row4451) := by
    change Data.row4451.contains = (SortedNatTree.contains FastRows.row4451)
    exact FastRows.row4451_checked.symm
  rw [row]
  decide
theorem root222Step9 : scan edge (search edge 3) 4 [1427] 1427 root222Tail9 winner =
    scan edge (search edge 3) 4 [1427] 1427 root222Tail10 winner := by
  rw [root222Tail9, scan, if_neg (by decide), root222Candidates9_checked]
  change scan edge (search edge 3) 4 [1427] 1427 root222Tail10
    (if 3 ≤ root222Candidates9.length then search edge 3 [1427, 4451] root222Candidates9 5878 winner else winner) = _
  rw [if_neg (by decide)]
def root222Candidates10 : List Nat := [5843]
set_option maxRecDepth 8192 in
theorem root222Candidates10_checked : candidates edge 4 1427 5297 root222Tail11 winner.bound = root222Candidates10 := by
  unfold candidates
  have row : edge 5297 = (SortedNatTree.contains FastRows.row5297) := by
    change Data.row5297.contains = (SortedNatTree.contains FastRows.row5297)
    exact FastRows.row5297_checked.symm
  rw [row]
  decide
theorem root222Step10 : scan edge (search edge 3) 4 [1427] 1427 root222Tail10 winner =
    scan edge (search edge 3) 4 [1427] 1427 root222Tail11 winner := by
  rw [root222Tail10, scan, if_neg (by decide), root222Candidates10_checked]
  change scan edge (search edge 3) 4 [1427] 1427 root222Tail11
    (if 3 ≤ root222Candidates10.length then search edge 3 [1427, 5297] root222Candidates10 6724 winner else winner) = _
  rw [if_neg (by decide)]
def root222Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root222Candidates11_checked : candidates edge 4 1427 5477 root222Tail12 winner.bound = root222Candidates11 := by
  unfold candidates
  have row : edge 5477 = (SortedNatTree.contains FastRows.row5477) := by
    change Data.row5477.contains = (SortedNatTree.contains FastRows.row5477)
    exact FastRows.row5477_checked.symm
  rw [row]
  decide
theorem root222Step11 : scan edge (search edge 3) 4 [1427] 1427 root222Tail11 winner =
    scan edge (search edge 3) 4 [1427] 1427 root222Tail12 winner := by
  rw [root222Tail11, scan, if_neg (by decide), root222Candidates11_checked]
  change scan edge (search edge 3) 4 [1427] 1427 root222Tail12
    (if 3 ≤ root222Candidates11.length then search edge 3 [1427, 5477] root222Candidates11 6904 winner else winner) = _
  rw [if_neg (by decide)]
def root222Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root222Candidates12_checked : candidates edge 4 1427 5501 root222Tail13 winner.bound = root222Candidates12 := by
  unfold candidates
  have row : edge 5501 = (SortedNatTree.contains FastRows.row5501) := by
    change Data.row5501.contains = (SortedNatTree.contains FastRows.row5501)
    exact FastRows.row5501_checked.symm
  rw [row]
  decide
theorem root222Step12 : scan edge (search edge 3) 4 [1427] 1427 root222Tail12 winner =
    scan edge (search edge 3) 4 [1427] 1427 root222Tail13 winner := by
  rw [root222Tail12, scan, if_neg (by decide), root222Candidates12_checked]
  change scan edge (search edge 3) 4 [1427] 1427 root222Tail13
    (if 3 ≤ root222Candidates12.length then search edge 3 [1427, 5501] root222Candidates12 6928 winner else winner) = _
  rw [if_neg (by decide)]
def root222Candidates13 : List Nat := [7937]
set_option maxRecDepth 8192 in
theorem root222Candidates13_checked : candidates edge 4 1427 5843 root222Tail14 winner.bound = root222Candidates13 := by
  unfold candidates
  have row : edge 5843 = (SortedNatTree.contains FastRows.row5843) := by
    change Data.row5843.contains = (SortedNatTree.contains FastRows.row5843)
    exact FastRows.row5843_checked.symm
  rw [row]
  decide
theorem root222Step13 : scan edge (search edge 3) 4 [1427] 1427 root222Tail13 winner =
    scan edge (search edge 3) 4 [1427] 1427 root222Tail14 winner := by
  rw [root222Tail13, scan, if_neg (by decide), root222Candidates13_checked]
  change scan edge (search edge 3) 4 [1427] 1427 root222Tail14
    (if 3 ≤ root222Candidates13.length then search edge 3 [1427, 5843] root222Candidates13 7270 winner else winner) = _
  rw [if_neg (by decide)]
theorem root222Scan14 : scan edge (search edge 3) 4 [1427] 1427 root222Tail14 winner = winner := by rw [root222Tail14, scan, if_pos (by decide)]
theorem root222Scan13 : scan edge (search edge 3) 4 [1427] 1427 root222Tail13 winner = winner :=
  (root222Step13).trans root222Scan14
theorem root222Scan12 : scan edge (search edge 3) 4 [1427] 1427 root222Tail12 winner = winner :=
  (root222Step12).trans root222Scan13
theorem root222Scan11 : scan edge (search edge 3) 4 [1427] 1427 root222Tail11 winner = winner :=
  (root222Step11).trans root222Scan12
theorem root222Scan10 : scan edge (search edge 3) 4 [1427] 1427 root222Tail10 winner = winner :=
  (root222Step10).trans root222Scan11
theorem root222Scan9 : scan edge (search edge 3) 4 [1427] 1427 root222Tail9 winner = winner :=
  (root222Step9).trans root222Scan10
theorem root222Scan8 : scan edge (search edge 3) 4 [1427] 1427 root222Tail8 winner = winner :=
  (root222Step8).trans root222Scan9
theorem root222Scan7 : scan edge (search edge 3) 4 [1427] 1427 root222Tail7 winner = winner :=
  (root222Step7).trans root222Scan8
theorem root222Scan6 : scan edge (search edge 3) 4 [1427] 1427 root222Tail6 winner = winner :=
  (root222Step6).trans root222Scan7
theorem root222Scan5 : scan edge (search edge 3) 4 [1427] 1427 root222Tail5 winner = winner :=
  (root222Step5).trans root222Scan6
theorem root222Scan4 : scan edge (search edge 3) 4 [1427] 1427 root222Tail4 winner = winner :=
  (root222Step4).trans root222Scan5
theorem root222Scan3 : scan edge (search edge 3) 4 [1427] 1427 root222Tail3 winner = winner :=
  (root222Step3).trans root222Scan4
theorem root222Scan2 : scan edge (search edge 3) 4 [1427] 1427 root222Tail2 winner = winner :=
  (root222Step2).trans root222Scan3
theorem root222Scan1 : scan edge (search edge 3) 4 [1427] 1427 root222Tail1 winner = winner :=
  (root222Step1).trans root222Scan2
theorem root222Scan0 : scan edge (search edge 3) 4 [1427] 1427 root222Tail0 winner = winner :=
  (root222Step0).trans root222Scan1
set_option maxRecDepth 4096 in
theorem root222Branch : search edge 4 [1427] root222Candidates 1427 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1427] 1427 root222Tail0 winner = winner
  exact root222Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
