import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root111Candidates : List Nat := Data.row619
set_option maxRecDepth 8192 in
theorem root111Candidates_checked : candidates edge 5 0 619 tail112 winner.bound = root111Candidates := by
  apply candidates_eq_row edge 5 0 619 tail112 root111Candidates winner.bound tail112_ordered
  · decide
  · decide
  · funext q
    rfl
def root111Tail58 : List Nat := []
def root111Tail57 : List Nat := 24481 :: root111Tail58
def root111Tail56 : List Nat := 24151 :: root111Tail57
def root111Tail55 : List Nat := 24133 :: root111Tail56
def root111Tail54 : List Nat := 22741 :: root111Tail55
def root111Tail53 : List Nat := 22639 :: root111Tail54
def root111Tail52 : List Nat := 21499 :: root111Tail53
def root111Tail51 : List Nat := 21187 :: root111Tail52
def root111Tail50 : List Nat := 21169 :: root111Tail51
def root111Tail49 : List Nat := 21121 :: root111Tail50
def root111Tail48 : List Nat := 20389 :: root111Tail49
def root111Tail47 : List Nat := 20161 :: root111Tail48
def root111Tail46 : List Nat := 20143 :: root111Tail47
def root111Tail45 : List Nat := 19801 :: root111Tail46
def root111Tail44 : List Nat := 19777 :: root111Tail45
def root111Tail43 : List Nat := 19009 :: root111Tail44
def root111Tail42 : List Nat := 18289 :: root111Tail43
def root111Tail41 : List Nat := 18229 :: root111Tail42
def root111Tail40 : List Nat := 17551 :: root111Tail41
def root111Tail39 : List Nat := 17431 :: root111Tail40
def root111Tail38 : List Nat := 16339 :: root111Tail39
def root111Tail37 : List Nat := 16111 :: root111Tail38
def root111Tail36 : List Nat := 15901 :: root111Tail37
def root111Tail35 : List Nat := 15331 :: root111Tail36
def root111Tail34 : List Nat := 15031 :: root111Tail35
def root111Tail33 : List Nat := 14887 :: root111Tail34
def root111Tail32 : List Nat := 14779 :: root111Tail33
def root111Tail31 : List Nat := 14683 :: root111Tail32
def root111Tail30 : List Nat := 13339 :: root111Tail31
def root111Tail29 : List Nat := 12637 :: root111Tail30
def root111Tail28 : List Nat := 12511 :: root111Tail29
def root111Tail27 : List Nat := 12451 :: root111Tail28
def root111Tail26 : List Nat := 11677 :: root111Tail27
def root111Tail25 : List Nat := 10831 :: root111Tail26
def root111Tail24 : List Nat := 10729 :: root111Tail25
def root111Tail23 : List Nat := 10567 :: root111Tail24
def root111Tail22 : List Nat := 10399 :: root111Tail23
def root111Tail21 : List Nat := 10357 :: root111Tail22
def root111Tail20 : List Nat := 9601 :: root111Tail21
def root111Tail19 : List Nat := 9337 :: root111Tail20
def root111Tail18 : List Nat := 9007 :: root111Tail19
def root111Tail17 : List Nat := 8377 :: root111Tail18
def root111Tail16 : List Nat := 7753 :: root111Tail17
def root111Tail15 : List Nat := 7687 :: root111Tail16
def root111Tail14 : List Nat := 7459 :: root111Tail15
def root111Tail13 : List Nat := 7309 :: root111Tail14
def root111Tail12 : List Nat := 7207 :: root111Tail13
def root111Tail11 : List Nat := 6991 :: root111Tail12
def root111Tail10 : List Nat := 5419 :: root111Tail11
def root111Tail9 : List Nat := 4621 :: root111Tail10
def root111Tail8 : List Nat := 3793 :: root111Tail9
def root111Tail7 : List Nat := 3571 :: root111Tail8
def root111Tail6 : List Nat := 2707 :: root111Tail7
def root111Tail5 : List Nat := 2671 :: root111Tail6
def root111Tail4 : List Nat := 1747 :: root111Tail5
def root111Tail3 : List Nat := 1597 :: root111Tail4
def root111Tail2 : List Nat := 1579 :: root111Tail3
def root111Tail1 : List Nat := 1051 :: root111Tail2
def root111Tail0 : List Nat := 811 :: root111Tail1
def root111Candidates0 : List Nat := [10567, 13339, 15901, 17551, 18229, 19009, 24481]
set_option maxRecDepth 8192 in
theorem root111Candidates0_checked : candidates edge 4 619 811 root111Tail1 winner.bound = root111Candidates0 := by
  unfold candidates
  have row : edge 811 = (SortedNatTree.contains FastRows.row811) := by
    change Data.row811.contains = (SortedNatTree.contains FastRows.row811)
    exact FastRows.row811_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root111Child0 : search edge 3 [619, 811] root111Candidates0 1430 winner = winner := by decide
theorem root111Step0 : scan edge (search edge 3) 4 [619] 619 root111Tail0 winner =
    scan edge (search edge 3) 4 [619] 619 root111Tail1 winner := by
  rw [root111Tail0, scan, if_neg (by decide), root111Candidates0_checked]
  change scan edge (search edge 3) 4 [619] 619 root111Tail1
    (if 3 ≤ root111Candidates0.length then search edge 3 [619, 811] root111Candidates0 1430 winner else winner) = _
  rw [if_pos (by decide), root111Child0]
def root111Candidates1 : List Nat := [9601, 14683]
set_option maxRecDepth 8192 in
theorem root111Candidates1_checked : candidates edge 4 619 1051 root111Tail2 winner.bound = root111Candidates1 := by
  unfold candidates
  have row : edge 1051 = (SortedNatTree.contains FastRows.row1051) := by
    change Data.row1051.contains = (SortedNatTree.contains FastRows.row1051)
    exact FastRows.row1051_checked.symm
  rw [row]
  decide
theorem root111Step1 : scan edge (search edge 3) 4 [619] 619 root111Tail1 winner =
    scan edge (search edge 3) 4 [619] 619 root111Tail2 winner := by
  rw [root111Tail1, scan, if_neg (by decide), root111Candidates1_checked]
  change scan edge (search edge 3) 4 [619] 619 root111Tail2
    (if 3 ≤ root111Candidates1.length then search edge 3 [619, 1051] root111Candidates1 1670 winner else winner) = _
  rw [if_neg (by decide)]
def root111Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root111Candidates2_checked : candidates edge 4 619 1579 root111Tail3 winner.bound = root111Candidates2 := by
  unfold candidates
  have row : edge 1579 = (SortedNatTree.contains FastRows.row1579) := by
    change Data.row1579.contains = (SortedNatTree.contains FastRows.row1579)
    exact FastRows.row1579_checked.symm
  rw [row]
  decide
theorem root111Step2 : scan edge (search edge 3) 4 [619] 619 root111Tail2 winner =
    scan edge (search edge 3) 4 [619] 619 root111Tail3 winner := by
  rw [root111Tail2, scan, if_neg (by decide), root111Candidates2_checked]
  change scan edge (search edge 3) 4 [619] 619 root111Tail3
    (if 3 ≤ root111Candidates2.length then search edge 3 [619, 1579] root111Candidates2 2198 winner else winner) = _
  rw [if_neg (by decide)]
def root111Candidates3 : List Nat := [11677, 16111]
set_option maxRecDepth 8192 in
theorem root111Candidates3_checked : candidates edge 4 619 1597 root111Tail4 winner.bound = root111Candidates3 := by
  unfold candidates
  have row : edge 1597 = (SortedNatTree.contains FastRows.row1597) := by
    change Data.row1597.contains = (SortedNatTree.contains FastRows.row1597)
    exact FastRows.row1597_checked.symm
  rw [row]
  decide
theorem root111Step3 : scan edge (search edge 3) 4 [619] 619 root111Tail3 winner =
    scan edge (search edge 3) 4 [619] 619 root111Tail4 winner := by
  rw [root111Tail3, scan, if_neg (by decide), root111Candidates3_checked]
  change scan edge (search edge 3) 4 [619] 619 root111Tail4
    (if 3 ≤ root111Candidates3.length then search edge 3 [619, 1597] root111Candidates3 2216 winner else winner) = _
  rw [if_neg (by decide)]
def root111Candidates4 : List Nat := [2671]
set_option maxRecDepth 8192 in
theorem root111Candidates4_checked : candidates edge 4 619 1747 root111Tail5 winner.bound = root111Candidates4 := by
  unfold candidates
  have row : edge 1747 = (SortedNatTree.contains FastRows.row1747) := by
    change Data.row1747.contains = (SortedNatTree.contains FastRows.row1747)
    exact FastRows.row1747_checked.symm
  rw [row]
  decide
theorem root111Step4 : scan edge (search edge 3) 4 [619] 619 root111Tail4 winner =
    scan edge (search edge 3) 4 [619] 619 root111Tail5 winner := by
  rw [root111Tail4, scan, if_neg (by decide), root111Candidates4_checked]
  change scan edge (search edge 3) 4 [619] 619 root111Tail5
    (if 3 ≤ root111Candidates4.length then search edge 3 [619, 1747] root111Candidates4 2366 winner else winner) = _
  rw [if_neg (by decide)]
def root111Candidates5 : List Nat := [3571, 21121]
set_option maxRecDepth 8192 in
theorem root111Candidates5_checked : candidates edge 4 619 2671 root111Tail6 winner.bound = root111Candidates5 := by
  unfold candidates
  have row : edge 2671 = (SortedNatTree.contains FastRows.row2671) := by
    change Data.row2671.contains = (SortedNatTree.contains FastRows.row2671)
    exact FastRows.row2671_checked.symm
  rw [row]
  decide
theorem root111Step5 : scan edge (search edge 3) 4 [619] 619 root111Tail5 winner =
    scan edge (search edge 3) 4 [619] 619 root111Tail6 winner := by
  rw [root111Tail5, scan, if_neg (by decide), root111Candidates5_checked]
  change scan edge (search edge 3) 4 [619] 619 root111Tail6
    (if 3 ≤ root111Candidates5.length then search edge 3 [619, 2671] root111Candidates5 3290 winner else winner) = _
  rw [if_neg (by decide)]
def root111Candidates6 : List Nat := [3793, 10729]
set_option maxRecDepth 8192 in
theorem root111Candidates6_checked : candidates edge 4 619 2707 root111Tail7 winner.bound = root111Candidates6 := by
  unfold candidates
  have row : edge 2707 = (SortedNatTree.contains FastRows.row2707) := by
    change Data.row2707.contains = (SortedNatTree.contains FastRows.row2707)
    exact FastRows.row2707_checked.symm
  rw [row]
  decide
theorem root111Step6 : scan edge (search edge 3) 4 [619] 619 root111Tail6 winner =
    scan edge (search edge 3) 4 [619] 619 root111Tail7 winner := by
  rw [root111Tail6, scan, if_neg (by decide), root111Candidates6_checked]
  change scan edge (search edge 3) 4 [619] 619 root111Tail7
    (if 3 ≤ root111Candidates6.length then search edge 3 [619, 2707] root111Candidates6 3326 winner else winner) = _
  rw [if_neg (by decide)]
def root111Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root111Candidates7_checked : candidates edge 4 619 3571 root111Tail8 winner.bound = root111Candidates7 := by
  unfold candidates
  have row : edge 3571 = (SortedNatTree.contains FastRows.row3571) := by
    change Data.row3571.contains = (SortedNatTree.contains FastRows.row3571)
    exact FastRows.row3571_checked.symm
  rw [row]
  decide
theorem root111Step7 : scan edge (search edge 3) 4 [619] 619 root111Tail7 winner =
    scan edge (search edge 3) 4 [619] 619 root111Tail8 winner := by
  rw [root111Tail7, scan, if_neg (by decide), root111Candidates7_checked]
  change scan edge (search edge 3) 4 [619] 619 root111Tail8
    (if 3 ≤ root111Candidates7.length then search edge 3 [619, 3571] root111Candidates7 4190 winner else winner) = _
  rw [if_neg (by decide)]
def root111Candidates8 : List Nat := [9007, 21169]
set_option maxRecDepth 8192 in
theorem root111Candidates8_checked : candidates edge 4 619 3793 root111Tail9 winner.bound = root111Candidates8 := by
  unfold candidates
  have row : edge 3793 = (SortedNatTree.contains FastRows.row3793) := by
    change Data.row3793.contains = (SortedNatTree.contains FastRows.row3793)
    exact FastRows.row3793_checked.symm
  rw [row]
  decide
theorem root111Step8 : scan edge (search edge 3) 4 [619] 619 root111Tail8 winner =
    scan edge (search edge 3) 4 [619] 619 root111Tail9 winner := by
  rw [root111Tail8, scan, if_neg (by decide), root111Candidates8_checked]
  change scan edge (search edge 3) 4 [619] 619 root111Tail9
    (if 3 ≤ root111Candidates8.length then search edge 3 [619, 3793] root111Candidates8 4412 winner else winner) = _
  rw [if_neg (by decide)]
def root111Candidates9 : List Nat := [9337]
set_option maxRecDepth 8192 in
theorem root111Candidates9_checked : candidates edge 4 619 4621 root111Tail10 winner.bound = root111Candidates9 := by
  unfold candidates
  have row : edge 4621 = (SortedNatTree.contains FastRows.row4621) := by
    change Data.row4621.contains = (SortedNatTree.contains FastRows.row4621)
    exact FastRows.row4621_checked.symm
  rw [row]
  decide
theorem root111Step9 : scan edge (search edge 3) 4 [619] 619 root111Tail9 winner =
    scan edge (search edge 3) 4 [619] 619 root111Tail10 winner := by
  rw [root111Tail9, scan, if_neg (by decide), root111Candidates9_checked]
  change scan edge (search edge 3) 4 [619] 619 root111Tail10
    (if 3 ≤ root111Candidates9.length then search edge 3 [619, 4621] root111Candidates9 5240 winner else winner) = _
  rw [if_neg (by decide)]
def root111Candidates10 : List Nat := [7687, 14683, 15331]
set_option maxRecDepth 8192 in
theorem root111Candidates10_checked : candidates edge 4 619 5419 root111Tail11 winner.bound = root111Candidates10 := by
  unfold candidates
  have row : edge 5419 = (SortedNatTree.contains FastRows.row5419) := by
    change Data.row5419.contains = (SortedNatTree.contains FastRows.row5419)
    exact FastRows.row5419_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root111Child10 : search edge 3 [619, 5419] root111Candidates10 6038 winner = winner := by decide
theorem root111Step10 : scan edge (search edge 3) 4 [619] 619 root111Tail10 winner =
    scan edge (search edge 3) 4 [619] 619 root111Tail11 winner := by
  rw [root111Tail10, scan, if_neg (by decide), root111Candidates10_checked]
  change scan edge (search edge 3) 4 [619] 619 root111Tail11
    (if 3 ≤ root111Candidates10.length then search edge 3 [619, 5419] root111Candidates10 6038 winner else winner) = _
  rw [if_pos (by decide), root111Child10]
theorem root111Scan11 : scan edge (search edge 3) 4 [619] 619 root111Tail11 winner = winner := by rw [root111Tail11, scan, if_pos (by decide)]
theorem root111Scan10 : scan edge (search edge 3) 4 [619] 619 root111Tail10 winner = winner :=
  (root111Step10).trans root111Scan11
theorem root111Scan9 : scan edge (search edge 3) 4 [619] 619 root111Tail9 winner = winner :=
  (root111Step9).trans root111Scan10
theorem root111Scan8 : scan edge (search edge 3) 4 [619] 619 root111Tail8 winner = winner :=
  (root111Step8).trans root111Scan9
theorem root111Scan7 : scan edge (search edge 3) 4 [619] 619 root111Tail7 winner = winner :=
  (root111Step7).trans root111Scan8
theorem root111Scan6 : scan edge (search edge 3) 4 [619] 619 root111Tail6 winner = winner :=
  (root111Step6).trans root111Scan7
theorem root111Scan5 : scan edge (search edge 3) 4 [619] 619 root111Tail5 winner = winner :=
  (root111Step5).trans root111Scan6
theorem root111Scan4 : scan edge (search edge 3) 4 [619] 619 root111Tail4 winner = winner :=
  (root111Step4).trans root111Scan5
theorem root111Scan3 : scan edge (search edge 3) 4 [619] 619 root111Tail3 winner = winner :=
  (root111Step3).trans root111Scan4
theorem root111Scan2 : scan edge (search edge 3) 4 [619] 619 root111Tail2 winner = winner :=
  (root111Step2).trans root111Scan3
theorem root111Scan1 : scan edge (search edge 3) 4 [619] 619 root111Tail1 winner = winner :=
  (root111Step1).trans root111Scan2
theorem root111Scan0 : scan edge (search edge 3) 4 [619] 619 root111Tail0 winner = winner :=
  (root111Step0).trans root111Scan1
set_option maxRecDepth 4096 in
theorem root111Branch : search edge 4 [619] root111Candidates 619 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [619] 619 root111Tail0 winner = winner
  exact root111Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
