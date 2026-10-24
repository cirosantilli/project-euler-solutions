import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root135Candidates : List Nat := Data.row787
set_option maxRecDepth 8192 in
theorem root135Candidates_checked : candidates edge 5 0 787 tail136 winner.bound = root135Candidates := by
  apply candidates_eq_row edge 5 0 787 tail136 root135Candidates winner.bound tail136_ordered
  · decide
  · decide
  · funext q
    rfl
def root135Tail58 : List Nat := []
def root135Tail57 : List Nat := 25183 :: root135Tail58
def root135Tail56 : List Nat := 24103 :: root135Tail57
def root135Tail55 : List Nat := 24043 :: root135Tail56
def root135Tail54 : List Nat := 23773 :: root135Tail55
def root135Tail53 : List Nat := 22279 :: root135Tail54
def root135Tail52 : List Nat := 22129 :: root135Tail53
def root135Tail51 : List Nat := 21751 :: root135Tail52
def root135Tail50 : List Nat := 21673 :: root135Tail51
def root135Tail49 : List Nat := 21529 :: root135Tail50
def root135Tail48 : List Nat := 21499 :: root135Tail49
def root135Tail47 : List Nat := 21481 :: root135Tail48
def root135Tail46 : List Nat := 20959 :: root135Tail47
def root135Tail45 : List Nat := 20431 :: root135Tail46
def root135Tail44 : List Nat := 20407 :: root135Tail45
def root135Tail43 : List Nat := 20011 :: root135Tail44
def root135Tail42 : List Nat := 19483 :: root135Tail43
def root135Tail41 : List Nat := 19423 :: root135Tail42
def root135Tail40 : List Nat := 19237 :: root135Tail41
def root135Tail39 : List Nat := 18859 :: root135Tail40
def root135Tail38 : List Nat := 18253 :: root135Tail39
def root135Tail37 : List Nat := 17911 :: root135Tail38
def root135Tail36 : List Nat := 17341 :: root135Tail37
def root135Tail35 : List Nat := 16633 :: root135Tail36
def root135Tail34 : List Nat := 16111 :: root135Tail35
def root135Tail33 : List Nat := 15583 :: root135Tail34
def root135Tail32 : List Nat := 15199 :: root135Tail33
def root135Tail31 : List Nat := 15031 :: root135Tail32
def root135Tail30 : List Nat := 12619 :: root135Tail31
def root135Tail29 : List Nat := 12577 :: root135Tail30
def root135Tail28 : List Nat := 12511 :: root135Tail29
def root135Tail27 : List Nat := 12373 :: root135Tail28
def root135Tail26 : List Nat := 12241 :: root135Tail27
def root135Tail25 : List Nat := 10501 :: root135Tail26
def root135Tail24 : List Nat := 10039 :: root135Tail25
def root135Tail23 : List Nat := 9931 :: root135Tail24
def root135Tail22 : List Nat := 9343 :: root135Tail23
def root135Tail21 : List Nat := 9181 :: root135Tail22
def root135Tail20 : List Nat := 8461 :: root135Tail21
def root135Tail19 : List Nat := 8317 :: root135Tail20
def root135Tail18 : List Nat := 7993 :: root135Tail19
def root135Tail17 : List Nat := 7873 :: root135Tail18
def root135Tail16 : List Nat := 7159 :: root135Tail17
def root135Tail15 : List Nat := 6733 :: root135Tail16
def root135Tail14 : List Nat := 6367 :: root135Tail15
def root135Tail13 : List Nat := 6007 :: root135Tail14
def root135Tail12 : List Nat := 5653 :: root135Tail13
def root135Tail11 : List Nat := 5347 :: root135Tail12
def root135Tail10 : List Nat := 4507 :: root135Tail11
def root135Tail9 : List Nat := 4177 :: root135Tail10
def root135Tail8 : List Nat := 4129 :: root135Tail9
def root135Tail7 : List Nat := 3631 :: root135Tail8
def root135Tail6 : List Nat := 3457 :: root135Tail7
def root135Tail5 : List Nat := 3301 :: root135Tail6
def root135Tail4 : List Nat := 2731 :: root135Tail5
def root135Tail3 : List Nat := 2671 :: root135Tail4
def root135Tail2 : List Nat := 2017 :: root135Tail3
def root135Tail1 : List Nat := 1483 :: root135Tail2
def root135Tail0 : List Nat := 823 :: root135Tail1
def root135Candidates0 : List Nat := [2017, 6733, 12619]
set_option maxRecDepth 8192 in
theorem root135Candidates0_checked : candidates edge 4 787 823 root135Tail1 winner.bound = root135Candidates0 := by
  unfold candidates
  have row : edge 823 = (SortedNatTree.contains FastRows.row823) := by
    change Data.row823.contains = (SortedNatTree.contains FastRows.row823)
    exact FastRows.row823_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root135Child0 : search edge 3 [787, 823] root135Candidates0 1610 winner = winner := by decide
theorem root135Step0 : scan edge (search edge 3) 4 [787] 787 root135Tail0 winner =
    scan edge (search edge 3) 4 [787] 787 root135Tail1 winner := by
  rw [root135Tail0, scan, if_neg (by decide), root135Candidates0_checked]
  change scan edge (search edge 3) 4 [787] 787 root135Tail1
    (if 3 ≤ root135Candidates0.length then search edge 3 [787, 823] root135Candidates0 1610 winner else winner) = _
  rw [if_pos (by decide), root135Child0]
def root135Candidates1 : List Nat := [3457, 7159, 7873, 10501]
set_option maxRecDepth 8192 in
theorem root135Candidates1_checked : candidates edge 4 787 1483 root135Tail2 winner.bound = root135Candidates1 := by
  unfold candidates
  have row : edge 1483 = (SortedNatTree.contains FastRows.row1483) := by
    change Data.row1483.contains = (SortedNatTree.contains FastRows.row1483)
    exact FastRows.row1483_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root135Child1 : search edge 3 [787, 1483] root135Candidates1 2270 winner = winner := by decide
theorem root135Step1 : scan edge (search edge 3) 4 [787] 787 root135Tail1 winner =
    scan edge (search edge 3) 4 [787] 787 root135Tail2 winner := by
  rw [root135Tail1, scan, if_neg (by decide), root135Candidates1_checked]
  change scan edge (search edge 3) 4 [787] 787 root135Tail2
    (if 3 ≤ root135Candidates1.length then search edge 3 [787, 1483] root135Candidates1 2270 winner else winner) = _
  rw [if_pos (by decide), root135Child1]
def root135Candidates2 : List Nat := [5653, 8317]
set_option maxRecDepth 8192 in
theorem root135Candidates2_checked : candidates edge 4 787 2017 root135Tail3 winner.bound = root135Candidates2 := by
  unfold candidates
  have row : edge 2017 = (SortedNatTree.contains FastRows.row2017) := by
    change Data.row2017.contains = (SortedNatTree.contains FastRows.row2017)
    exact FastRows.row2017_checked.symm
  rw [row]
  decide
theorem root135Step2 : scan edge (search edge 3) 4 [787] 787 root135Tail2 winner =
    scan edge (search edge 3) 4 [787] 787 root135Tail3 winner := by
  rw [root135Tail2, scan, if_neg (by decide), root135Candidates2_checked]
  change scan edge (search edge 3) 4 [787] 787 root135Tail3
    (if 3 ≤ root135Candidates2.length then search edge 3 [787, 2017] root135Candidates2 2804 winner else winner) = _
  rw [if_neg (by decide)]
def root135Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root135Candidates3_checked : candidates edge 4 787 2671 root135Tail4 winner.bound = root135Candidates3 := by
  unfold candidates
  have row : edge 2671 = (SortedNatTree.contains FastRows.row2671) := by
    change Data.row2671.contains = (SortedNatTree.contains FastRows.row2671)
    exact FastRows.row2671_checked.symm
  rw [row]
  decide
theorem root135Step3 : scan edge (search edge 3) 4 [787] 787 root135Tail3 winner =
    scan edge (search edge 3) 4 [787] 787 root135Tail4 winner := by
  rw [root135Tail3, scan, if_neg (by decide), root135Candidates3_checked]
  change scan edge (search edge 3) 4 [787] 787 root135Tail4
    (if 3 ≤ root135Candidates3.length then search edge 3 [787, 2671] root135Candidates3 3458 winner else winner) = _
  rw [if_neg (by decide)]
def root135Candidates4 : List Nat := [3301, 4129, 7993]
set_option maxRecDepth 8192 in
theorem root135Candidates4_checked : candidates edge 4 787 2731 root135Tail5 winner.bound = root135Candidates4 := by
  unfold candidates
  have row : edge 2731 = (SortedNatTree.contains FastRows.row2731) := by
    change Data.row2731.contains = (SortedNatTree.contains FastRows.row2731)
    exact FastRows.row2731_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root135Child4 : search edge 3 [787, 2731] root135Candidates4 3518 winner = winner := by decide
theorem root135Step4 : scan edge (search edge 3) 4 [787] 787 root135Tail4 winner =
    scan edge (search edge 3) 4 [787] 787 root135Tail5 winner := by
  rw [root135Tail4, scan, if_neg (by decide), root135Candidates4_checked]
  change scan edge (search edge 3) 4 [787] 787 root135Tail5
    (if 3 ≤ root135Candidates4.length then search edge 3 [787, 2731] root135Candidates4 3518 winner else winner) = _
  rw [if_pos (by decide), root135Child4]
def root135Candidates5 : List Nat := [15031]
set_option maxRecDepth 8192 in
theorem root135Candidates5_checked : candidates edge 4 787 3301 root135Tail6 winner.bound = root135Candidates5 := by
  unfold candidates
  have row : edge 3301 = (SortedNatTree.contains FastRows.row3301) := by
    change Data.row3301.contains = (SortedNatTree.contains FastRows.row3301)
    exact FastRows.row3301_checked.symm
  rw [row]
  decide
theorem root135Step5 : scan edge (search edge 3) 4 [787] 787 root135Tail5 winner =
    scan edge (search edge 3) 4 [787] 787 root135Tail6 winner := by
  rw [root135Tail5, scan, if_neg (by decide), root135Candidates5_checked]
  change scan edge (search edge 3) 4 [787] 787 root135Tail6
    (if 3 ≤ root135Candidates5.length then search edge 3 [787, 3301] root135Candidates5 4088 winner else winner) = _
  rw [if_neg (by decide)]
def root135Candidates6 : List Nat := [9931]
set_option maxRecDepth 8192 in
theorem root135Candidates6_checked : candidates edge 4 787 3457 root135Tail7 winner.bound = root135Candidates6 := by
  unfold candidates
  have row : edge 3457 = (SortedNatTree.contains FastRows.row3457) := by
    change Data.row3457.contains = (SortedNatTree.contains FastRows.row3457)
    exact FastRows.row3457_checked.symm
  rw [row]
  decide
theorem root135Step6 : scan edge (search edge 3) 4 [787] 787 root135Tail6 winner =
    scan edge (search edge 3) 4 [787] 787 root135Tail7 winner := by
  rw [root135Tail6, scan, if_neg (by decide), root135Candidates6_checked]
  change scan edge (search edge 3) 4 [787] 787 root135Tail7
    (if 3 ≤ root135Candidates6.length then search edge 3 [787, 3457] root135Candidates6 4244 winner else winner) = _
  rw [if_neg (by decide)]
def root135Candidates7 : List Nat := [5347, 12577]
set_option maxRecDepth 8192 in
theorem root135Candidates7_checked : candidates edge 4 787 3631 root135Tail8 winner.bound = root135Candidates7 := by
  unfold candidates
  have row : edge 3631 = (SortedNatTree.contains FastRows.row3631) := by
    change Data.row3631.contains = (SortedNatTree.contains FastRows.row3631)
    exact FastRows.row3631_checked.symm
  rw [row]
  decide
theorem root135Step7 : scan edge (search edge 3) 4 [787] 787 root135Tail7 winner =
    scan edge (search edge 3) 4 [787] 787 root135Tail8 winner := by
  rw [root135Tail7, scan, if_neg (by decide), root135Candidates7_checked]
  change scan edge (search edge 3) 4 [787] 787 root135Tail8
    (if 3 ≤ root135Candidates7.length then search edge 3 [787, 3631] root135Candidates7 4418 winner else winner) = _
  rw [if_neg (by decide)]
def root135Candidates8 : List Nat := [7159, 9343, 12373]
set_option maxRecDepth 8192 in
theorem root135Candidates8_checked : candidates edge 4 787 4129 root135Tail9 winner.bound = root135Candidates8 := by
  unfold candidates
  have row : edge 4129 = (SortedNatTree.contains FastRows.row4129) := by
    change Data.row4129.contains = (SortedNatTree.contains FastRows.row4129)
    exact FastRows.row4129_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root135Child8 : search edge 3 [787, 4129] root135Candidates8 4916 winner = winner := by decide
theorem root135Step8 : scan edge (search edge 3) 4 [787] 787 root135Tail8 winner =
    scan edge (search edge 3) 4 [787] 787 root135Tail9 winner := by
  rw [root135Tail8, scan, if_neg (by decide), root135Candidates8_checked]
  change scan edge (search edge 3) 4 [787] 787 root135Tail9
    (if 3 ≤ root135Candidates8.length then search edge 3 [787, 4129] root135Candidates8 4916 winner else winner) = _
  rw [if_pos (by decide), root135Child8]
def root135Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root135Candidates9_checked : candidates edge 4 787 4177 root135Tail10 winner.bound = root135Candidates9 := by
  unfold candidates
  have row : edge 4177 = (SortedNatTree.contains FastRows.row4177) := by
    change Data.row4177.contains = (SortedNatTree.contains FastRows.row4177)
    exact FastRows.row4177_checked.symm
  rw [row]
  decide
theorem root135Step9 : scan edge (search edge 3) 4 [787] 787 root135Tail9 winner =
    scan edge (search edge 3) 4 [787] 787 root135Tail10 winner := by
  rw [root135Tail9, scan, if_neg (by decide), root135Candidates9_checked]
  change scan edge (search edge 3) 4 [787] 787 root135Tail10
    (if 3 ≤ root135Candidates9.length then search edge 3 [787, 4177] root135Candidates9 4964 winner else winner) = _
  rw [if_neg (by decide)]
def root135Candidates10 : List Nat := [7993]
set_option maxRecDepth 8192 in
theorem root135Candidates10_checked : candidates edge 4 787 4507 root135Tail11 winner.bound = root135Candidates10 := by
  unfold candidates
  have row : edge 4507 = (SortedNatTree.contains FastRows.row4507) := by
    change Data.row4507.contains = (SortedNatTree.contains FastRows.row4507)
    exact FastRows.row4507_checked.symm
  rw [row]
  decide
theorem root135Step10 : scan edge (search edge 3) 4 [787] 787 root135Tail10 winner =
    scan edge (search edge 3) 4 [787] 787 root135Tail11 winner := by
  rw [root135Tail10, scan, if_neg (by decide), root135Candidates10_checked]
  change scan edge (search edge 3) 4 [787] 787 root135Tail11
    (if 3 ≤ root135Candidates10.length then search edge 3 [787, 4507] root135Candidates10 5294 winner else winner) = _
  rw [if_neg (by decide)]
def root135Candidates11 : List Nat := [8461, 16111]
set_option maxRecDepth 8192 in
theorem root135Candidates11_checked : candidates edge 4 787 5347 root135Tail12 winner.bound = root135Candidates11 := by
  unfold candidates
  have row : edge 5347 = (SortedNatTree.contains FastRows.row5347) := by
    change Data.row5347.contains = (SortedNatTree.contains FastRows.row5347)
    exact FastRows.row5347_checked.symm
  rw [row]
  decide
theorem root135Step11 : scan edge (search edge 3) 4 [787] 787 root135Tail11 winner =
    scan edge (search edge 3) 4 [787] 787 root135Tail12 winner := by
  rw [root135Tail11, scan, if_neg (by decide), root135Candidates11_checked]
  change scan edge (search edge 3) 4 [787] 787 root135Tail12
    (if 3 ≤ root135Candidates11.length then search edge 3 [787, 5347] root135Candidates11 6134 winner else winner) = _
  rw [if_neg (by decide)]
def root135Candidates12 : List Nat := [6367]
set_option maxRecDepth 8192 in
theorem root135Candidates12_checked : candidates edge 4 787 5653 root135Tail13 winner.bound = root135Candidates12 := by
  unfold candidates
  have row : edge 5653 = (SortedNatTree.contains FastRows.row5653) := by
    change Data.row5653.contains = (SortedNatTree.contains FastRows.row5653)
    exact FastRows.row5653_checked.symm
  rw [row]
  decide
theorem root135Step12 : scan edge (search edge 3) 4 [787] 787 root135Tail12 winner =
    scan edge (search edge 3) 4 [787] 787 root135Tail13 winner := by
  rw [root135Tail12, scan, if_neg (by decide), root135Candidates12_checked]
  change scan edge (search edge 3) 4 [787] 787 root135Tail13
    (if 3 ≤ root135Candidates12.length then search edge 3 [787, 5653] root135Candidates12 6440 winner else winner) = _
  rw [if_neg (by decide)]
def root135Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root135Candidates13_checked : candidates edge 4 787 6007 root135Tail14 winner.bound = root135Candidates13 := by
  unfold candidates
  have row : edge 6007 = (SortedNatTree.contains FastRows.row6007) := by
    change Data.row6007.contains = (SortedNatTree.contains FastRows.row6007)
    exact FastRows.row6007_checked.symm
  rw [row]
  decide
theorem root135Step13 : scan edge (search edge 3) 4 [787] 787 root135Tail13 winner =
    scan edge (search edge 3) 4 [787] 787 root135Tail14 winner := by
  rw [root135Tail13, scan, if_neg (by decide), root135Candidates13_checked]
  change scan edge (search edge 3) 4 [787] 787 root135Tail14
    (if 3 ≤ root135Candidates13.length then search edge 3 [787, 6007] root135Candidates13 6794 winner else winner) = _
  rw [if_neg (by decide)]
theorem root135Scan14 : scan edge (search edge 3) 4 [787] 787 root135Tail14 winner = winner := by rw [root135Tail14, scan, if_pos (by decide)]
theorem root135Scan13 : scan edge (search edge 3) 4 [787] 787 root135Tail13 winner = winner :=
  (root135Step13).trans root135Scan14
theorem root135Scan12 : scan edge (search edge 3) 4 [787] 787 root135Tail12 winner = winner :=
  (root135Step12).trans root135Scan13
theorem root135Scan11 : scan edge (search edge 3) 4 [787] 787 root135Tail11 winner = winner :=
  (root135Step11).trans root135Scan12
theorem root135Scan10 : scan edge (search edge 3) 4 [787] 787 root135Tail10 winner = winner :=
  (root135Step10).trans root135Scan11
theorem root135Scan9 : scan edge (search edge 3) 4 [787] 787 root135Tail9 winner = winner :=
  (root135Step9).trans root135Scan10
theorem root135Scan8 : scan edge (search edge 3) 4 [787] 787 root135Tail8 winner = winner :=
  (root135Step8).trans root135Scan9
theorem root135Scan7 : scan edge (search edge 3) 4 [787] 787 root135Tail7 winner = winner :=
  (root135Step7).trans root135Scan8
theorem root135Scan6 : scan edge (search edge 3) 4 [787] 787 root135Tail6 winner = winner :=
  (root135Step6).trans root135Scan7
theorem root135Scan5 : scan edge (search edge 3) 4 [787] 787 root135Tail5 winner = winner :=
  (root135Step5).trans root135Scan6
theorem root135Scan4 : scan edge (search edge 3) 4 [787] 787 root135Tail4 winner = winner :=
  (root135Step4).trans root135Scan5
theorem root135Scan3 : scan edge (search edge 3) 4 [787] 787 root135Tail3 winner = winner :=
  (root135Step3).trans root135Scan4
theorem root135Scan2 : scan edge (search edge 3) 4 [787] 787 root135Tail2 winner = winner :=
  (root135Step2).trans root135Scan3
theorem root135Scan1 : scan edge (search edge 3) 4 [787] 787 root135Tail1 winner = winner :=
  (root135Step1).trans root135Scan2
theorem root135Scan0 : scan edge (search edge 3) 4 [787] 787 root135Tail0 winner = winner :=
  (root135Step0).trans root135Scan1
set_option maxRecDepth 4096 in
theorem root135Branch : search edge 4 [787] root135Candidates 787 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [787] 787 root135Tail0 winner = winner
  exact root135Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
