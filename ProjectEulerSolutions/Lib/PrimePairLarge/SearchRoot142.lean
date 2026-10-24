import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root142Candidates : List Nat := Data.row829
set_option maxRecDepth 8192 in
theorem root142Candidates_checked : candidates edge 5 0 829 tail143 winner.bound = root142Candidates := by
  apply candidates_eq_row edge 5 0 829 tail143 root142Candidates winner.bound tail143_ordered
  · decide
  · decide
  · funext q
    rfl
def root142Tail62 : List Nat := []
def root142Tail61 : List Nat := 24841 :: root142Tail62
def root142Tail60 : List Nat := 24631 :: root142Tail61
def root142Tail59 : List Nat := 24469 :: root142Tail60
def root142Tail58 : List Nat := 24043 :: root142Tail59
def root142Tail57 : List Nat := 24007 :: root142Tail58
def root142Tail56 : List Nat := 23833 :: root142Tail57
def root142Tail55 : List Nat := 23431 :: root142Tail56
def root142Tail54 : List Nat := 23041 :: root142Tail55
def root142Tail53 : List Nat := 23017 :: root142Tail54
def root142Tail52 : List Nat := 22573 :: root142Tail53
def root142Tail51 : List Nat := 22027 :: root142Tail52
def root142Tail50 : List Nat := 21961 :: root142Tail51
def root142Tail49 : List Nat := 21799 :: root142Tail50
def root142Tail48 : List Nat := 20857 :: root142Tail49
def root142Tail47 : List Nat := 19219 :: root142Tail48
def root142Tail46 : List Nat := 18439 :: root142Tail47
def root142Tail45 : List Nat := 18181 :: root142Tail46
def root142Tail44 : List Nat := 18169 :: root142Tail45
def root142Tail43 : List Nat := 18127 :: root142Tail44
def root142Tail42 : List Nat := 17929 :: root142Tail43
def root142Tail41 : List Nat := 17683 :: root142Tail42
def root142Tail40 : List Nat := 17431 :: root142Tail41
def root142Tail39 : List Nat := 17389 :: root142Tail40
def root142Tail38 : List Nat := 17383 :: root142Tail39
def root142Tail37 : List Nat := 15643 :: root142Tail38
def root142Tail36 : List Nat := 15121 :: root142Tail37
def root142Tail35 : List Nat := 14827 :: root142Tail36
def root142Tail34 : List Nat := 14071 :: root142Tail35
def root142Tail33 : List Nat := 13903 :: root142Tail34
def root142Tail32 : List Nat := 13591 :: root142Tail33
def root142Tail31 : List Nat := 13477 :: root142Tail32
def root142Tail30 : List Nat := 13297 :: root142Tail31
def root142Tail29 : List Nat := 12517 :: root142Tail30
def root142Tail28 : List Nat := 12163 :: root142Tail29
def root142Tail27 : List Nat := 11467 :: root142Tail28
def root142Tail26 : List Nat := 11299 :: root142Tail27
def root142Tail25 : List Nat := 10939 :: root142Tail26
def root142Tail24 : List Nat := 10903 :: root142Tail25
def root142Tail23 : List Nat := 10687 :: root142Tail24
def root142Tail22 : List Nat := 10273 :: root142Tail23
def root142Tail21 : List Nat := 9091 :: root142Tail22
def root142Tail20 : List Nat := 9001 :: root142Tail21
def root142Tail19 : List Nat := 8923 :: root142Tail20
def root142Tail18 : List Nat := 8803 :: root142Tail19
def root142Tail17 : List Nat := 8539 :: root142Tail18
def root142Tail16 : List Nat := 8209 :: root142Tail17
def root142Tail15 : List Nat := 7573 :: root142Tail16
def root142Tail14 : List Nat := 7351 :: root142Tail15
def root142Tail13 : List Nat := 7321 :: root142Tail14
def root142Tail12 : List Nat := 6823 :: root142Tail13
def root142Tail11 : List Nat := 5857 :: root142Tail12
def root142Tail10 : List Nat := 5839 :: root142Tail11
def root142Tail9 : List Nat := 5563 :: root142Tail10
def root142Tail8 : List Nat := 5527 :: root142Tail9
def root142Tail7 : List Nat := 5107 :: root142Tail8
def root142Tail6 : List Nat := 3847 :: root142Tail7
def root142Tail5 : List Nat := 3361 :: root142Tail6
def root142Tail4 : List Nat := 3319 :: root142Tail5
def root142Tail3 : List Nat := 2671 :: root142Tail4
def root142Tail2 : List Nat := 2293 :: root142Tail3
def root142Tail1 : List Nat := 1237 :: root142Tail2
def root142Tail0 : List Nat := 1033 :: root142Tail1
def root142Candidates0 : List Nat := [10903]
set_option maxRecDepth 8192 in
theorem root142Candidates0_checked : candidates edge 4 829 1033 root142Tail1 winner.bound = root142Candidates0 := by
  unfold candidates
  have row : edge 1033 = (SortedNatTree.contains FastRows.row1033) := by
    change Data.row1033.contains = (SortedNatTree.contains FastRows.row1033)
    exact FastRows.row1033_checked.symm
  rw [row]
  decide
theorem root142Step0 : scan edge (search edge 3) 4 [829] 829 root142Tail0 winner =
    scan edge (search edge 3) 4 [829] 829 root142Tail1 winner := by
  rw [root142Tail0, scan, if_neg (by decide), root142Candidates0_checked]
  change scan edge (search edge 3) 4 [829] 829 root142Tail1
    (if 3 ≤ root142Candidates0.length then search edge 3 [829, 1033] root142Candidates0 1862 winner else winner) = _
  rw [if_neg (by decide)]
def root142Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root142Candidates1_checked : candidates edge 4 829 1237 root142Tail2 winner.bound = root142Candidates1 := by
  unfold candidates
  have row : edge 1237 = (SortedNatTree.contains FastRows.row1237) := by
    change Data.row1237.contains = (SortedNatTree.contains FastRows.row1237)
    exact FastRows.row1237_checked.symm
  rw [row]
  decide
theorem root142Step1 : scan edge (search edge 3) 4 [829] 829 root142Tail1 winner =
    scan edge (search edge 3) 4 [829] 829 root142Tail2 winner := by
  rw [root142Tail1, scan, if_neg (by decide), root142Candidates1_checked]
  change scan edge (search edge 3) 4 [829] 829 root142Tail2
    (if 3 ≤ root142Candidates1.length then search edge 3 [829, 1237] root142Candidates1 2066 winner else winner) = _
  rw [if_neg (by decide)]
def root142Candidates2 : List Nat := [2671]
set_option maxRecDepth 8192 in
theorem root142Candidates2_checked : candidates edge 4 829 2293 root142Tail3 winner.bound = root142Candidates2 := by
  unfold candidates
  have row : edge 2293 = (SortedNatTree.contains FastRows.row2293) := by
    change Data.row2293.contains = (SortedNatTree.contains FastRows.row2293)
    exact FastRows.row2293_checked.symm
  rw [row]
  decide
theorem root142Step2 : scan edge (search edge 3) 4 [829] 829 root142Tail2 winner =
    scan edge (search edge 3) 4 [829] 829 root142Tail3 winner := by
  rw [root142Tail2, scan, if_neg (by decide), root142Candidates2_checked]
  change scan edge (search edge 3) 4 [829] 829 root142Tail3
    (if 3 ≤ root142Candidates2.length then search edge 3 [829, 2293] root142Candidates2 3122 winner else winner) = _
  rw [if_neg (by decide)]
def root142Candidates3 : List Nat := [3361]
set_option maxRecDepth 8192 in
theorem root142Candidates3_checked : candidates edge 4 829 2671 root142Tail4 winner.bound = root142Candidates3 := by
  unfold candidates
  have row : edge 2671 = (SortedNatTree.contains FastRows.row2671) := by
    change Data.row2671.contains = (SortedNatTree.contains FastRows.row2671)
    exact FastRows.row2671_checked.symm
  rw [row]
  decide
theorem root142Step3 : scan edge (search edge 3) 4 [829] 829 root142Tail3 winner =
    scan edge (search edge 3) 4 [829] 829 root142Tail4 winner := by
  rw [root142Tail3, scan, if_neg (by decide), root142Candidates3_checked]
  change scan edge (search edge 3) 4 [829] 829 root142Tail4
    (if 3 ≤ root142Candidates3.length then search edge 3 [829, 2671] root142Candidates3 3500 winner else winner) = _
  rw [if_neg (by decide)]
def root142Candidates4 : List Nat := [7321, 10903, 13297]
set_option maxRecDepth 8192 in
theorem root142Candidates4_checked : candidates edge 4 829 3319 root142Tail5 winner.bound = root142Candidates4 := by
  unfold candidates
  have row : edge 3319 = (SortedNatTree.contains FastRows.row3319) := by
    change Data.row3319.contains = (SortedNatTree.contains FastRows.row3319)
    exact FastRows.row3319_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root142Child4 : search edge 3 [829, 3319] root142Candidates4 4148 winner = winner := by decide
theorem root142Step4 : scan edge (search edge 3) 4 [829] 829 root142Tail4 winner =
    scan edge (search edge 3) 4 [829] 829 root142Tail5 winner := by
  rw [root142Tail4, scan, if_neg (by decide), root142Candidates4_checked]
  change scan edge (search edge 3) 4 [829] 829 root142Tail5
    (if 3 ≤ root142Candidates4.length then search edge 3 [829, 3319] root142Candidates4 4148 winner else winner) = _
  rw [if_pos (by decide), root142Child4]
def root142Candidates5 : List Nat := [17389]
set_option maxRecDepth 8192 in
theorem root142Candidates5_checked : candidates edge 4 829 3361 root142Tail6 winner.bound = root142Candidates5 := by
  unfold candidates
  have row : edge 3361 = (SortedNatTree.contains FastRows.row3361) := by
    change Data.row3361.contains = (SortedNatTree.contains FastRows.row3361)
    exact FastRows.row3361_checked.symm
  rw [row]
  decide
theorem root142Step5 : scan edge (search edge 3) 4 [829] 829 root142Tail5 winner =
    scan edge (search edge 3) 4 [829] 829 root142Tail6 winner := by
  rw [root142Tail5, scan, if_neg (by decide), root142Candidates5_checked]
  change scan edge (search edge 3) 4 [829] 829 root142Tail6
    (if 3 ≤ root142Candidates5.length then search edge 3 [829, 3361] root142Candidates5 4190 winner else winner) = _
  rw [if_neg (by decide)]
def root142Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root142Candidates6_checked : candidates edge 4 829 3847 root142Tail7 winner.bound = root142Candidates6 := by
  unfold candidates
  have row : edge 3847 = (SortedNatTree.contains FastRows.row3847) := by
    change Data.row3847.contains = (SortedNatTree.contains FastRows.row3847)
    exact FastRows.row3847_checked.symm
  rw [row]
  decide
theorem root142Step6 : scan edge (search edge 3) 4 [829] 829 root142Tail6 winner =
    scan edge (search edge 3) 4 [829] 829 root142Tail7 winner := by
  rw [root142Tail6, scan, if_neg (by decide), root142Candidates6_checked]
  change scan edge (search edge 3) 4 [829] 829 root142Tail7
    (if 3 ≤ root142Candidates6.length then search edge 3 [829, 3847] root142Candidates6 4676 winner else winner) = _
  rw [if_neg (by decide)]
def root142Candidates7 : List Nat := [5839, 13297]
set_option maxRecDepth 8192 in
theorem root142Candidates7_checked : candidates edge 4 829 5107 root142Tail8 winner.bound = root142Candidates7 := by
  unfold candidates
  have row : edge 5107 = (SortedNatTree.contains FastRows.row5107) := by
    change Data.row5107.contains = (SortedNatTree.contains FastRows.row5107)
    exact FastRows.row5107_checked.symm
  rw [row]
  decide
theorem root142Step7 : scan edge (search edge 3) 4 [829] 829 root142Tail7 winner =
    scan edge (search edge 3) 4 [829] 829 root142Tail8 winner := by
  rw [root142Tail7, scan, if_neg (by decide), root142Candidates7_checked]
  change scan edge (search edge 3) 4 [829] 829 root142Tail8
    (if 3 ≤ root142Candidates7.length then search edge 3 [829, 5107] root142Candidates7 5936 winner else winner) = _
  rw [if_neg (by decide)]
def root142Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root142Candidates8_checked : candidates edge 4 829 5527 root142Tail9 winner.bound = root142Candidates8 := by
  unfold candidates
  have row : edge 5527 = (SortedNatTree.contains FastRows.row5527) := by
    change Data.row5527.contains = (SortedNatTree.contains FastRows.row5527)
    exact FastRows.row5527_checked.symm
  rw [row]
  decide
theorem root142Step8 : scan edge (search edge 3) 4 [829] 829 root142Tail8 winner =
    scan edge (search edge 3) 4 [829] 829 root142Tail9 winner := by
  rw [root142Tail8, scan, if_neg (by decide), root142Candidates8_checked]
  change scan edge (search edge 3) 4 [829] 829 root142Tail9
    (if 3 ≤ root142Candidates8.length then search edge 3 [829, 5527] root142Candidates8 6356 winner else winner) = _
  rw [if_neg (by decide)]
def root142Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root142Candidates9_checked : candidates edge 4 829 5563 root142Tail10 winner.bound = root142Candidates9 := by
  unfold candidates
  have row : edge 5563 = (SortedNatTree.contains FastRows.row5563) := by
    change Data.row5563.contains = (SortedNatTree.contains FastRows.row5563)
    exact FastRows.row5563_checked.symm
  rw [row]
  decide
theorem root142Step9 : scan edge (search edge 3) 4 [829] 829 root142Tail9 winner =
    scan edge (search edge 3) 4 [829] 829 root142Tail10 winner := by
  rw [root142Tail9, scan, if_neg (by decide), root142Candidates9_checked]
  change scan edge (search edge 3) 4 [829] 829 root142Tail10
    (if 3 ≤ root142Candidates9.length then search edge 3 [829, 5563] root142Candidates9 6392 winner else winner) = _
  rw [if_neg (by decide)]
def root142Candidates10 : List Nat := [8539, 15121]
set_option maxRecDepth 8192 in
theorem root142Candidates10_checked : candidates edge 4 829 5839 root142Tail11 winner.bound = root142Candidates10 := by
  unfold candidates
  have row : edge 5839 = (SortedNatTree.contains FastRows.row5839) := by
    change Data.row5839.contains = (SortedNatTree.contains FastRows.row5839)
    exact FastRows.row5839_checked.symm
  rw [row]
  decide
theorem root142Step10 : scan edge (search edge 3) 4 [829] 829 root142Tail10 winner =
    scan edge (search edge 3) 4 [829] 829 root142Tail11 winner := by
  rw [root142Tail10, scan, if_neg (by decide), root142Candidates10_checked]
  change scan edge (search edge 3) 4 [829] 829 root142Tail11
    (if 3 ≤ root142Candidates10.length then search edge 3 [829, 5839] root142Candidates10 6668 winner else winner) = _
  rw [if_neg (by decide)]
def root142Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root142Candidates11_checked : candidates edge 4 829 5857 root142Tail12 winner.bound = root142Candidates11 := by
  unfold candidates
  have row : edge 5857 = (SortedNatTree.contains FastRows.row5857) := by
    change Data.row5857.contains = (SortedNatTree.contains FastRows.row5857)
    exact FastRows.row5857_checked.symm
  rw [row]
  decide
theorem root142Step11 : scan edge (search edge 3) 4 [829] 829 root142Tail11 winner =
    scan edge (search edge 3) 4 [829] 829 root142Tail12 winner := by
  rw [root142Tail11, scan, if_neg (by decide), root142Candidates11_checked]
  change scan edge (search edge 3) 4 [829] 829 root142Tail12
    (if 3 ≤ root142Candidates11.length then search edge 3 [829, 5857] root142Candidates11 6686 winner else winner) = _
  rw [if_neg (by decide)]
theorem root142Scan12 : scan edge (search edge 3) 4 [829] 829 root142Tail12 winner = winner := by rw [root142Tail12, scan, if_pos (by decide)]
theorem root142Scan11 : scan edge (search edge 3) 4 [829] 829 root142Tail11 winner = winner :=
  (root142Step11).trans root142Scan12
theorem root142Scan10 : scan edge (search edge 3) 4 [829] 829 root142Tail10 winner = winner :=
  (root142Step10).trans root142Scan11
theorem root142Scan9 : scan edge (search edge 3) 4 [829] 829 root142Tail9 winner = winner :=
  (root142Step9).trans root142Scan10
theorem root142Scan8 : scan edge (search edge 3) 4 [829] 829 root142Tail8 winner = winner :=
  (root142Step8).trans root142Scan9
theorem root142Scan7 : scan edge (search edge 3) 4 [829] 829 root142Tail7 winner = winner :=
  (root142Step7).trans root142Scan8
theorem root142Scan6 : scan edge (search edge 3) 4 [829] 829 root142Tail6 winner = winner :=
  (root142Step6).trans root142Scan7
theorem root142Scan5 : scan edge (search edge 3) 4 [829] 829 root142Tail5 winner = winner :=
  (root142Step5).trans root142Scan6
theorem root142Scan4 : scan edge (search edge 3) 4 [829] 829 root142Tail4 winner = winner :=
  (root142Step4).trans root142Scan5
theorem root142Scan3 : scan edge (search edge 3) 4 [829] 829 root142Tail3 winner = winner :=
  (root142Step3).trans root142Scan4
theorem root142Scan2 : scan edge (search edge 3) 4 [829] 829 root142Tail2 winner = winner :=
  (root142Step2).trans root142Scan3
theorem root142Scan1 : scan edge (search edge 3) 4 [829] 829 root142Tail1 winner = winner :=
  (root142Step1).trans root142Scan2
theorem root142Scan0 : scan edge (search edge 3) 4 [829] 829 root142Tail0 winner = winner :=
  (root142Step0).trans root142Scan1
set_option maxRecDepth 4096 in
theorem root142Branch : search edge 4 [829] root142Candidates 829 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [829] 829 root142Tail0 winner = winner
  exact root142Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
