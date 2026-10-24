import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root214Candidates : List Nat := Data.row1327
set_option maxRecDepth 8192 in
theorem root214Candidates_checked : candidates edge 5 0 1327 tail215 winner.bound = root214Candidates := by
  apply candidates_eq_row edge 5 0 1327 tail215 root214Candidates winner.bound tail215_ordered
  · decide
  · decide
  · funext q
    rfl
def root214Tail52 : List Nat := []
def root214Tail51 : List Nat := 24337 :: root214Tail52
def root214Tail50 : List Nat := 24169 :: root214Tail51
def root214Tail49 : List Nat := 24133 :: root214Tail50
def root214Tail48 : List Nat := 23497 :: root214Tail49
def root214Tail47 : List Nat := 22921 :: root214Tail48
def root214Tail46 : List Nat := 22573 :: root214Tail47
def root214Tail45 : List Nat := 20521 :: root214Tail46
def root214Tail44 : List Nat := 19603 :: root214Tail45
def root214Tail43 : List Nat := 19267 :: root214Tail44
def root214Tail42 : List Nat := 19141 :: root214Tail43
def root214Tail41 : List Nat := 18679 :: root214Tail42
def root214Tail40 : List Nat := 18457 :: root214Tail41
def root214Tail39 : List Nat := 18379 :: root214Tail40
def root214Tail38 : List Nat := 17881 :: root214Tail39
def root214Tail37 : List Nat := 17209 :: root214Tail38
def root214Tail36 : List Nat := 16249 :: root214Tail37
def root214Tail35 : List Nat := 14929 :: root214Tail36
def root214Tail34 : List Nat := 13807 :: root214Tail35
def root214Tail33 : List Nat := 13297 :: root214Tail34
def root214Tail32 : List Nat := 13267 :: root214Tail33
def root214Tail31 : List Nat := 12919 :: root214Tail32
def root214Tail30 : List Nat := 11443 :: root214Tail31
def root214Tail29 : List Nat := 11113 :: root214Tail30
def root214Tail28 : List Nat := 10273 :: root214Tail29
def root214Tail27 : List Nat := 9391 :: root214Tail28
def root214Tail26 : List Nat := 9241 :: root214Tail27
def root214Tail25 : List Nat := 9181 :: root214Tail26
def root214Tail24 : List Nat := 8971 :: root214Tail25
def root214Tail23 : List Nat := 8737 :: root214Tail24
def root214Tail22 : List Nat := 8563 :: root214Tail23
def root214Tail21 : List Nat := 8263 :: root214Tail22
def root214Tail20 : List Nat := 7963 :: root214Tail21
def root214Tail19 : List Nat := 7717 :: root214Tail20
def root214Tail18 : List Nat := 7351 :: root214Tail19
def root214Tail17 : List Nat := 7129 :: root214Tail18
def root214Tail16 : List Nat := 6619 :: root214Tail17
def root214Tail15 : List Nat := 6301 :: root214Tail16
def root214Tail14 : List Nat := 6217 :: root214Tail15
def root214Tail13 : List Nat := 6163 :: root214Tail14
def root214Tail12 : List Nat := 5701 :: root214Tail13
def root214Tail11 : List Nat := 4621 :: root214Tail12
def root214Tail10 : List Nat := 4357 :: root214Tail11
def root214Tail9 : List Nat := 3853 :: root214Tail10
def root214Tail8 : List Nat := 3769 :: root214Tail9
def root214Tail7 : List Nat := 3739 :: root214Tail8
def root214Tail6 : List Nat := 3361 :: root214Tail7
def root214Tail5 : List Nat := 3109 :: root214Tail6
def root214Tail4 : List Nat := 2551 :: root214Tail5
def root214Tail3 : List Nat := 2293 :: root214Tail4
def root214Tail2 : List Nat := 1723 :: root214Tail3
def root214Tail1 : List Nat := 1471 :: root214Tail2
def root214Tail0 : List Nat := 1429 :: root214Tail1
def root214Candidates0 : List Nat := [1723, 3739, 6619, 7351, 7963, 8737, 11113, 19267]
set_option maxRecDepth 8192 in
theorem root214Candidates0_checked : candidates edge 4 1327 1429 root214Tail1 winner.bound = root214Candidates0 := by
  unfold candidates
  have row : edge 1429 = (SortedNatTree.contains FastRows.row1429) := by
    change Data.row1429.contains = (SortedNatTree.contains FastRows.row1429)
    exact FastRows.row1429_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root214Child0 : search edge 3 [1327, 1429] root214Candidates0 2756 winner = winner := by decide
theorem root214Step0 : scan edge (search edge 3) 4 [1327] 1327 root214Tail0 winner =
    scan edge (search edge 3) 4 [1327] 1327 root214Tail1 winner := by
  rw [root214Tail0, scan, if_neg (by decide), root214Candidates0_checked]
  change scan edge (search edge 3) 4 [1327] 1327 root214Tail1
    (if 3 ≤ root214Candidates0.length then search edge 3 [1327, 1429] root214Candidates0 2756 winner else winner) = _
  rw [if_pos (by decide), root214Child0]
def root214Candidates1 : List Nat := [6301, 9181]
set_option maxRecDepth 8192 in
theorem root214Candidates1_checked : candidates edge 4 1327 1471 root214Tail2 winner.bound = root214Candidates1 := by
  unfold candidates
  have row : edge 1471 = (SortedNatTree.contains FastRows.row1471) := by
    change Data.row1471.contains = (SortedNatTree.contains FastRows.row1471)
    exact FastRows.row1471_checked.symm
  rw [row]
  decide
theorem root214Step1 : scan edge (search edge 3) 4 [1327] 1327 root214Tail1 winner =
    scan edge (search edge 3) 4 [1327] 1327 root214Tail2 winner := by
  rw [root214Tail1, scan, if_neg (by decide), root214Candidates1_checked]
  change scan edge (search edge 3) 4 [1327] 1327 root214Tail2
    (if 3 ≤ root214Candidates1.length then search edge 3 [1327, 1471] root214Candidates1 2798 winner else winner) = _
  rw [if_neg (by decide)]
def root214Candidates2 : List Nat := [4621, 8563]
set_option maxRecDepth 8192 in
theorem root214Candidates2_checked : candidates edge 4 1327 1723 root214Tail3 winner.bound = root214Candidates2 := by
  unfold candidates
  have row : edge 1723 = (SortedNatTree.contains FastRows.row1723) := by
    change Data.row1723.contains = (SortedNatTree.contains FastRows.row1723)
    exact FastRows.row1723_checked.symm
  rw [row]
  decide
theorem root214Step2 : scan edge (search edge 3) 4 [1327] 1327 root214Tail2 winner =
    scan edge (search edge 3) 4 [1327] 1327 root214Tail3 winner := by
  rw [root214Tail2, scan, if_neg (by decide), root214Candidates2_checked]
  change scan edge (search edge 3) 4 [1327] 1327 root214Tail3
    (if 3 ≤ root214Candidates2.length then search edge 3 [1327, 1723] root214Candidates2 3050 winner else winner) = _
  rw [if_neg (by decide)]
def root214Candidates3 : List Nat := [3853]
set_option maxRecDepth 8192 in
theorem root214Candidates3_checked : candidates edge 4 1327 2293 root214Tail4 winner.bound = root214Candidates3 := by
  unfold candidates
  have row : edge 2293 = (SortedNatTree.contains FastRows.row2293) := by
    change Data.row2293.contains = (SortedNatTree.contains FastRows.row2293)
    exact FastRows.row2293_checked.symm
  rw [row]
  decide
theorem root214Step3 : scan edge (search edge 3) 4 [1327] 1327 root214Tail3 winner =
    scan edge (search edge 3) 4 [1327] 1327 root214Tail4 winner := by
  rw [root214Tail3, scan, if_neg (by decide), root214Candidates3_checked]
  change scan edge (search edge 3) 4 [1327] 1327 root214Tail4
    (if 3 ≤ root214Candidates3.length then search edge 3 [1327, 2293] root214Candidates3 3620 winner else winner) = _
  rw [if_neg (by decide)]
def root214Candidates4 : List Nat := [6163, 9181]
set_option maxRecDepth 8192 in
theorem root214Candidates4_checked : candidates edge 4 1327 2551 root214Tail5 winner.bound = root214Candidates4 := by
  unfold candidates
  have row : edge 2551 = (SortedNatTree.contains FastRows.row2551) := by
    change Data.row2551.contains = (SortedNatTree.contains FastRows.row2551)
    exact FastRows.row2551_checked.symm
  rw [row]
  decide
theorem root214Step4 : scan edge (search edge 3) 4 [1327] 1327 root214Tail4 winner =
    scan edge (search edge 3) 4 [1327] 1327 root214Tail5 winner := by
  rw [root214Tail4, scan, if_neg (by decide), root214Candidates4_checked]
  change scan edge (search edge 3) 4 [1327] 1327 root214Tail5
    (if 3 ≤ root214Candidates4.length then search edge 3 [1327, 2551] root214Candidates4 3878 winner else winner) = _
  rw [if_neg (by decide)]
def root214Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root214Candidates5_checked : candidates edge 4 1327 3109 root214Tail6 winner.bound = root214Candidates5 := by
  unfold candidates
  have row : edge 3109 = (SortedNatTree.contains FastRows.row3109) := by
    change Data.row3109.contains = (SortedNatTree.contains FastRows.row3109)
    exact FastRows.row3109_checked.symm
  rw [row]
  decide
theorem root214Step5 : scan edge (search edge 3) 4 [1327] 1327 root214Tail5 winner =
    scan edge (search edge 3) 4 [1327] 1327 root214Tail6 winner := by
  rw [root214Tail5, scan, if_neg (by decide), root214Candidates5_checked]
  change scan edge (search edge 3) 4 [1327] 1327 root214Tail6
    (if 3 ≤ root214Candidates5.length then search edge 3 [1327, 3109] root214Candidates5 4436 winner else winner) = _
  rw [if_neg (by decide)]
def root214Candidates6 : List Nat := [3739, 8737, 8971]
set_option maxRecDepth 8192 in
theorem root214Candidates6_checked : candidates edge 4 1327 3361 root214Tail7 winner.bound = root214Candidates6 := by
  unfold candidates
  have row : edge 3361 = (SortedNatTree.contains FastRows.row3361) := by
    change Data.row3361.contains = (SortedNatTree.contains FastRows.row3361)
    exact FastRows.row3361_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root214Child6 : search edge 3 [1327, 3361] root214Candidates6 4688 winner = winner := by decide
theorem root214Step6 : scan edge (search edge 3) 4 [1327] 1327 root214Tail6 winner =
    scan edge (search edge 3) 4 [1327] 1327 root214Tail7 winner := by
  rw [root214Tail6, scan, if_neg (by decide), root214Candidates6_checked]
  change scan edge (search edge 3) 4 [1327] 1327 root214Tail7
    (if 3 ≤ root214Candidates6.length then search edge 3 [1327, 3361] root214Candidates6 4688 winner else winner) = _
  rw [if_pos (by decide), root214Child6]
def root214Candidates7 : List Nat := [13807]
set_option maxRecDepth 8192 in
theorem root214Candidates7_checked : candidates edge 4 1327 3739 root214Tail8 winner.bound = root214Candidates7 := by
  unfold candidates
  have row : edge 3739 = (SortedNatTree.contains FastRows.row3739) := by
    change Data.row3739.contains = (SortedNatTree.contains FastRows.row3739)
    exact FastRows.row3739_checked.symm
  rw [row]
  decide
theorem root214Step7 : scan edge (search edge 3) 4 [1327] 1327 root214Tail7 winner =
    scan edge (search edge 3) 4 [1327] 1327 root214Tail8 winner := by
  rw [root214Tail7, scan, if_neg (by decide), root214Candidates7_checked]
  change scan edge (search edge 3) 4 [1327] 1327 root214Tail8
    (if 3 ≤ root214Candidates7.length then search edge 3 [1327, 3739] root214Candidates7 5066 winner else winner) = _
  rw [if_neg (by decide)]
def root214Candidates8 : List Nat := [6619, 8971, 18379]
set_option maxRecDepth 8192 in
theorem root214Candidates8_checked : candidates edge 4 1327 3769 root214Tail9 winner.bound = root214Candidates8 := by
  unfold candidates
  have row : edge 3769 = (SortedNatTree.contains FastRows.row3769) := by
    change Data.row3769.contains = (SortedNatTree.contains FastRows.row3769)
    exact FastRows.row3769_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root214Child8 : search edge 3 [1327, 3769] root214Candidates8 5096 winner = winner := by decide
theorem root214Step8 : scan edge (search edge 3) 4 [1327] 1327 root214Tail8 winner =
    scan edge (search edge 3) 4 [1327] 1327 root214Tail9 winner := by
  rw [root214Tail8, scan, if_neg (by decide), root214Candidates8_checked]
  change scan edge (search edge 3) 4 [1327] 1327 root214Tail9
    (if 3 ≤ root214Candidates8.length then search edge 3 [1327, 3769] root214Candidates8 5096 winner else winner) = _
  rw [if_pos (by decide), root214Child8]
def root214Candidates9 : List Nat := [10273, 13297, 18457]
set_option maxRecDepth 8192 in
theorem root214Candidates9_checked : candidates edge 4 1327 3853 root214Tail10 winner.bound = root214Candidates9 := by
  unfold candidates
  have row : edge 3853 = (SortedNatTree.contains FastRows.row3853) := by
    change Data.row3853.contains = (SortedNatTree.contains FastRows.row3853)
    exact FastRows.row3853_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root214Child9 : search edge 3 [1327, 3853] root214Candidates9 5180 winner = winner := by decide
theorem root214Step9 : scan edge (search edge 3) 4 [1327] 1327 root214Tail9 winner =
    scan edge (search edge 3) 4 [1327] 1327 root214Tail10 winner := by
  rw [root214Tail9, scan, if_neg (by decide), root214Candidates9_checked]
  change scan edge (search edge 3) 4 [1327] 1327 root214Tail10
    (if 3 ≤ root214Candidates9.length then search edge 3 [1327, 3853] root214Candidates9 5180 winner else winner) = _
  rw [if_pos (by decide), root214Child9]
def root214Candidates10 : List Nat := [6619]
set_option maxRecDepth 8192 in
theorem root214Candidates10_checked : candidates edge 4 1327 4357 root214Tail11 winner.bound = root214Candidates10 := by
  unfold candidates
  have row : edge 4357 = (SortedNatTree.contains FastRows.row4357) := by
    change Data.row4357.contains = (SortedNatTree.contains FastRows.row4357)
    exact FastRows.row4357_checked.symm
  rw [row]
  decide
theorem root214Step10 : scan edge (search edge 3) 4 [1327] 1327 root214Tail10 winner =
    scan edge (search edge 3) 4 [1327] 1327 root214Tail11 winner := by
  rw [root214Tail10, scan, if_neg (by decide), root214Candidates10_checked]
  change scan edge (search edge 3) 4 [1327] 1327 root214Tail11
    (if 3 ≤ root214Candidates10.length then search edge 3 [1327, 4357] root214Candidates10 5684 winner else winner) = _
  rw [if_neg (by decide)]
def root214Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root214Candidates11_checked : candidates edge 4 1327 4621 root214Tail12 winner.bound = root214Candidates11 := by
  unfold candidates
  have row : edge 4621 = (SortedNatTree.contains FastRows.row4621) := by
    change Data.row4621.contains = (SortedNatTree.contains FastRows.row4621)
    exact FastRows.row4621_checked.symm
  rw [row]
  decide
theorem root214Step11 : scan edge (search edge 3) 4 [1327] 1327 root214Tail11 winner =
    scan edge (search edge 3) 4 [1327] 1327 root214Tail12 winner := by
  rw [root214Tail11, scan, if_neg (by decide), root214Candidates11_checked]
  change scan edge (search edge 3) 4 [1327] 1327 root214Tail12
    (if 3 ≤ root214Candidates11.length then search edge 3 [1327, 4621] root214Candidates11 5948 winner else winner) = _
  rw [if_neg (by decide)]
def root214Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root214Candidates12_checked : candidates edge 4 1327 5701 root214Tail13 winner.bound = root214Candidates12 := by
  unfold candidates
  have row : edge 5701 = (SortedNatTree.contains FastRows.row5701) := by
    change Data.row5701.contains = (SortedNatTree.contains FastRows.row5701)
    exact FastRows.row5701_checked.symm
  rw [row]
  decide
theorem root214Step12 : scan edge (search edge 3) 4 [1327] 1327 root214Tail12 winner =
    scan edge (search edge 3) 4 [1327] 1327 root214Tail13 winner := by
  rw [root214Tail12, scan, if_neg (by decide), root214Candidates12_checked]
  change scan edge (search edge 3) 4 [1327] 1327 root214Tail13
    (if 3 ≤ root214Candidates12.length then search edge 3 [1327, 5701] root214Candidates12 7028 winner else winner) = _
  rw [if_neg (by decide)]
def root214Candidates13 : List Nat := [8971]
set_option maxRecDepth 8192 in
theorem root214Candidates13_checked : candidates edge 4 1327 6163 root214Tail14 winner.bound = root214Candidates13 := by
  unfold candidates
  have row : edge 6163 = (SortedNatTree.contains FastRows.row6163) := by
    change Data.row6163.contains = (SortedNatTree.contains FastRows.row6163)
    exact FastRows.row6163_checked.symm
  rw [row]
  decide
theorem root214Step13 : scan edge (search edge 3) 4 [1327] 1327 root214Tail13 winner =
    scan edge (search edge 3) 4 [1327] 1327 root214Tail14 winner := by
  rw [root214Tail13, scan, if_neg (by decide), root214Candidates13_checked]
  change scan edge (search edge 3) 4 [1327] 1327 root214Tail14
    (if 3 ≤ root214Candidates13.length then search edge 3 [1327, 6163] root214Candidates13 7490 winner else winner) = _
  rw [if_neg (by decide)]
theorem root214Scan14 : scan edge (search edge 3) 4 [1327] 1327 root214Tail14 winner = winner := by rw [root214Tail14, scan, if_pos (by decide)]
theorem root214Scan13 : scan edge (search edge 3) 4 [1327] 1327 root214Tail13 winner = winner :=
  (root214Step13).trans root214Scan14
theorem root214Scan12 : scan edge (search edge 3) 4 [1327] 1327 root214Tail12 winner = winner :=
  (root214Step12).trans root214Scan13
theorem root214Scan11 : scan edge (search edge 3) 4 [1327] 1327 root214Tail11 winner = winner :=
  (root214Step11).trans root214Scan12
theorem root214Scan10 : scan edge (search edge 3) 4 [1327] 1327 root214Tail10 winner = winner :=
  (root214Step10).trans root214Scan11
theorem root214Scan9 : scan edge (search edge 3) 4 [1327] 1327 root214Tail9 winner = winner :=
  (root214Step9).trans root214Scan10
theorem root214Scan8 : scan edge (search edge 3) 4 [1327] 1327 root214Tail8 winner = winner :=
  (root214Step8).trans root214Scan9
theorem root214Scan7 : scan edge (search edge 3) 4 [1327] 1327 root214Tail7 winner = winner :=
  (root214Step7).trans root214Scan8
theorem root214Scan6 : scan edge (search edge 3) 4 [1327] 1327 root214Tail6 winner = winner :=
  (root214Step6).trans root214Scan7
theorem root214Scan5 : scan edge (search edge 3) 4 [1327] 1327 root214Tail5 winner = winner :=
  (root214Step5).trans root214Scan6
theorem root214Scan4 : scan edge (search edge 3) 4 [1327] 1327 root214Tail4 winner = winner :=
  (root214Step4).trans root214Scan5
theorem root214Scan3 : scan edge (search edge 3) 4 [1327] 1327 root214Tail3 winner = winner :=
  (root214Step3).trans root214Scan4
theorem root214Scan2 : scan edge (search edge 3) 4 [1327] 1327 root214Tail2 winner = winner :=
  (root214Step2).trans root214Scan3
theorem root214Scan1 : scan edge (search edge 3) 4 [1327] 1327 root214Tail1 winner = winner :=
  (root214Step1).trans root214Scan2
theorem root214Scan0 : scan edge (search edge 3) 4 [1327] 1327 root214Tail0 winner = winner :=
  (root214Step0).trans root214Scan1
set_option maxRecDepth 4096 in
theorem root214Branch : search edge 4 [1327] root214Candidates 1327 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1327] 1327 root214Tail0 winner = winner
  exact root214Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
