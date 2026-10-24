import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root252Candidates : List Nat := Data.row1613
set_option maxRecDepth 8192 in
theorem root252Candidates_checked : candidates edge 5 0 1613 tail253 winner.bound = root252Candidates := by
  apply candidates_eq_row edge 5 0 1613 tail253 root252Candidates winner.bound tail253_ordered
  · decide
  · decide
  · funext q
    rfl
def root252Tail53 : List Nat := []
def root252Tail52 : List Nat := 23993 :: root252Tail53
def root252Tail51 : List Nat := 23957 :: root252Tail52
def root252Tail50 : List Nat := 23567 :: root252Tail51
def root252Tail49 : List Nat := 22907 :: root252Tail50
def root252Tail48 : List Nat := 22691 :: root252Tail49
def root252Tail47 : List Nat := 21683 :: root252Tail48
def root252Tail46 : List Nat := 21401 :: root252Tail47
def root252Tail45 : List Nat := 19997 :: root252Tail46
def root252Tail44 : List Nat := 19841 :: root252Tail45
def root252Tail43 : List Nat := 19073 :: root252Tail44
def root252Tail42 : List Nat := 18959 :: root252Tail43
def root252Tail41 : List Nat := 18713 :: root252Tail42
def root252Tail40 : List Nat := 18041 :: root252Tail41
def root252Tail39 : List Nat := 17909 :: root252Tail40
def root252Tail38 : List Nat := 17789 :: root252Tail39
def root252Tail37 : List Nat := 17393 :: root252Tail38
def root252Tail36 : List Nat := 16811 :: root252Tail37
def root252Tail35 : List Nat := 16481 :: root252Tail36
def root252Tail34 : List Nat := 15647 :: root252Tail35
def root252Tail33 : List Nat := 15083 :: root252Tail34
def root252Tail32 : List Nat := 14957 :: root252Tail33
def root252Tail31 : List Nat := 13913 :: root252Tail32
def root252Tail30 : List Nat := 13151 :: root252Tail31
def root252Tail29 : List Nat := 12791 :: root252Tail30
def root252Tail28 : List Nat := 12149 :: root252Tail29
def root252Tail27 : List Nat := 12101 :: root252Tail28
def root252Tail26 : List Nat := 11057 :: root252Tail27
def root252Tail25 : List Nat := 11027 :: root252Tail26
def root252Tail24 : List Nat := 10979 :: root252Tail25
def root252Tail23 : List Nat := 10427 :: root252Tail24
def root252Tail22 : List Nat := 9743 :: root252Tail23
def root252Tail21 : List Nat := 9029 :: root252Tail22
def root252Tail20 : List Nat := 8861 :: root252Tail21
def root252Tail19 : List Nat := 8783 :: root252Tail20
def root252Tail18 : List Nat := 7853 :: root252Tail19
def root252Tail17 : List Nat := 7487 :: root252Tail18
def root252Tail16 : List Nat := 7481 :: root252Tail17
def root252Tail15 : List Nat := 7229 :: root252Tail16
def root252Tail14 : List Nat := 6971 :: root252Tail15
def root252Tail13 : List Nat := 6899 :: root252Tail14
def root252Tail12 : List Nat := 6311 :: root252Tail13
def root252Tail11 : List Nat := 5813 :: root252Tail12
def root252Tail10 : List Nat := 5297 :: root252Tail11
def root252Tail9 : List Nat := 4967 :: root252Tail10
def root252Tail8 : List Nat := 4673 :: root252Tail9
def root252Tail7 : List Nat := 4133 :: root252Tail8
def root252Tail6 : List Nat := 3779 :: root252Tail7
def root252Tail5 : List Nat := 3023 :: root252Tail6
def root252Tail4 : List Nat := 2693 :: root252Tail5
def root252Tail3 : List Nat := 2477 :: root252Tail4
def root252Tail2 : List Nat := 1979 :: root252Tail3
def root252Tail1 : List Nat := 1949 :: root252Tail2
def root252Tail0 : List Nat := 1637 :: root252Tail1
def root252Candidates0 : List Nat := [16481]
set_option maxRecDepth 8192 in
theorem root252Candidates0_checked : candidates edge 4 1613 1637 root252Tail1 winner.bound = root252Candidates0 := by
  unfold candidates
  have row : edge 1637 = (SortedNatTree.contains FastRows.row1637) := by
    change Data.row1637.contains = (SortedNatTree.contains FastRows.row1637)
    exact FastRows.row1637_checked.symm
  rw [row]
  decide
theorem root252Step0 : scan edge (search edge 3) 4 [1613] 1613 root252Tail0 winner =
    scan edge (search edge 3) 4 [1613] 1613 root252Tail1 winner := by
  rw [root252Tail0, scan, if_neg (by decide), root252Candidates0_checked]
  change scan edge (search edge 3) 4 [1613] 1613 root252Tail1
    (if 3 ≤ root252Candidates0.length then search edge 3 [1613, 1637] root252Candidates0 3250 winner else winner) = _
  rw [if_neg (by decide)]
def root252Candidates1 : List Nat := [19997]
set_option maxRecDepth 8192 in
theorem root252Candidates1_checked : candidates edge 4 1613 1949 root252Tail2 winner.bound = root252Candidates1 := by
  unfold candidates
  have row : edge 1949 = (SortedNatTree.contains FastRows.row1949) := by
    change Data.row1949.contains = (SortedNatTree.contains FastRows.row1949)
    exact FastRows.row1949_checked.symm
  rw [row]
  decide
theorem root252Step1 : scan edge (search edge 3) 4 [1613] 1613 root252Tail1 winner =
    scan edge (search edge 3) 4 [1613] 1613 root252Tail2 winner := by
  rw [root252Tail1, scan, if_neg (by decide), root252Candidates1_checked]
  change scan edge (search edge 3) 4 [1613] 1613 root252Tail2
    (if 3 ≤ root252Candidates1.length then search edge 3 [1613, 1949] root252Candidates1 3562 winner else winner) = _
  rw [if_neg (by decide)]
def root252Candidates2 : List Nat := [2693, 14957, 15083]
set_option maxRecDepth 8192 in
theorem root252Candidates2_checked : candidates edge 4 1613 1979 root252Tail3 winner.bound = root252Candidates2 := by
  unfold candidates
  have row : edge 1979 = (SortedNatTree.contains FastRows.row1979) := by
    change Data.row1979.contains = (SortedNatTree.contains FastRows.row1979)
    exact FastRows.row1979_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root252Child2 : search edge 3 [1613, 1979] root252Candidates2 3592 winner = winner := by decide
theorem root252Step2 : scan edge (search edge 3) 4 [1613] 1613 root252Tail2 winner =
    scan edge (search edge 3) 4 [1613] 1613 root252Tail3 winner := by
  rw [root252Tail2, scan, if_neg (by decide), root252Candidates2_checked]
  change scan edge (search edge 3) 4 [1613] 1613 root252Tail3
    (if 3 ≤ root252Candidates2.length then search edge 3 [1613, 1979] root252Candidates2 3592 winner else winner) = _
  rw [if_pos (by decide), root252Child2]
def root252Candidates3 : List Nat := [4133, 7853, 8783]
set_option maxRecDepth 8192 in
theorem root252Candidates3_checked : candidates edge 4 1613 2477 root252Tail4 winner.bound = root252Candidates3 := by
  unfold candidates
  have row : edge 2477 = (SortedNatTree.contains FastRows.row2477) := by
    change Data.row2477.contains = (SortedNatTree.contains FastRows.row2477)
    exact FastRows.row2477_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root252Child3 : search edge 3 [1613, 2477] root252Candidates3 4090 winner = winner := by decide
theorem root252Step3 : scan edge (search edge 3) 4 [1613] 1613 root252Tail3 winner =
    scan edge (search edge 3) 4 [1613] 1613 root252Tail4 winner := by
  rw [root252Tail3, scan, if_neg (by decide), root252Candidates3_checked]
  change scan edge (search edge 3) 4 [1613] 1613 root252Tail4
    (if 3 ≤ root252Candidates3.length then search edge 3 [1613, 2477] root252Candidates3 4090 winner else winner) = _
  rw [if_pos (by decide), root252Child3]
def root252Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root252Candidates4_checked : candidates edge 4 1613 2693 root252Tail5 winner.bound = root252Candidates4 := by
  unfold candidates
  have row : edge 2693 = (SortedNatTree.contains FastRows.row2693) := by
    change Data.row2693.contains = (SortedNatTree.contains FastRows.row2693)
    exact FastRows.row2693_checked.symm
  rw [row]
  decide
theorem root252Step4 : scan edge (search edge 3) 4 [1613] 1613 root252Tail4 winner =
    scan edge (search edge 3) 4 [1613] 1613 root252Tail5 winner := by
  rw [root252Tail4, scan, if_neg (by decide), root252Candidates4_checked]
  change scan edge (search edge 3) 4 [1613] 1613 root252Tail5
    (if 3 ≤ root252Candidates4.length then search edge 3 [1613, 2693] root252Candidates4 4306 winner else winner) = _
  rw [if_neg (by decide)]
def root252Candidates5 : List Nat := [8783, 17909]
set_option maxRecDepth 8192 in
theorem root252Candidates5_checked : candidates edge 4 1613 3023 root252Tail6 winner.bound = root252Candidates5 := by
  unfold candidates
  have row : edge 3023 = (SortedNatTree.contains FastRows.row3023) := by
    change Data.row3023.contains = (SortedNatTree.contains FastRows.row3023)
    exact FastRows.row3023_checked.symm
  rw [row]
  decide
theorem root252Step5 : scan edge (search edge 3) 4 [1613] 1613 root252Tail5 winner =
    scan edge (search edge 3) 4 [1613] 1613 root252Tail6 winner := by
  rw [root252Tail5, scan, if_neg (by decide), root252Candidates5_checked]
  change scan edge (search edge 3) 4 [1613] 1613 root252Tail6
    (if 3 ≤ root252Candidates5.length then search edge 3 [1613, 3023] root252Candidates5 4636 winner else winner) = _
  rw [if_neg (by decide)]
def root252Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root252Candidates6_checked : candidates edge 4 1613 3779 root252Tail7 winner.bound = root252Candidates6 := by
  unfold candidates
  have row : edge 3779 = (SortedNatTree.contains FastRows.row3779) := by
    change Data.row3779.contains = (SortedNatTree.contains FastRows.row3779)
    exact FastRows.row3779_checked.symm
  rw [row]
  decide
theorem root252Step6 : scan edge (search edge 3) 4 [1613] 1613 root252Tail6 winner =
    scan edge (search edge 3) 4 [1613] 1613 root252Tail7 winner := by
  rw [root252Tail6, scan, if_neg (by decide), root252Candidates6_checked]
  change scan edge (search edge 3) 4 [1613] 1613 root252Tail7
    (if 3 ≤ root252Candidates6.length then search edge 3 [1613, 3779] root252Candidates6 5392 winner else winner) = _
  rw [if_neg (by decide)]
def root252Candidates7 : List Nat := [15647]
set_option maxRecDepth 8192 in
theorem root252Candidates7_checked : candidates edge 4 1613 4133 root252Tail8 winner.bound = root252Candidates7 := by
  unfold candidates
  have row : edge 4133 = (SortedNatTree.contains FastRows.row4133) := by
    change Data.row4133.contains = (SortedNatTree.contains FastRows.row4133)
    exact FastRows.row4133_checked.symm
  rw [row]
  decide
theorem root252Step7 : scan edge (search edge 3) 4 [1613] 1613 root252Tail7 winner =
    scan edge (search edge 3) 4 [1613] 1613 root252Tail8 winner := by
  rw [root252Tail7, scan, if_neg (by decide), root252Candidates7_checked]
  change scan edge (search edge 3) 4 [1613] 1613 root252Tail8
    (if 3 ≤ root252Candidates7.length then search edge 3 [1613, 4133] root252Candidates7 5746 winner else winner) = _
  rw [if_neg (by decide)]
def root252Candidates8 : List Nat := [14957, 16481]
set_option maxRecDepth 8192 in
theorem root252Candidates8_checked : candidates edge 4 1613 4673 root252Tail9 winner.bound = root252Candidates8 := by
  unfold candidates
  have row : edge 4673 = (SortedNatTree.contains FastRows.row4673) := by
    change Data.row4673.contains = (SortedNatTree.contains FastRows.row4673)
    exact FastRows.row4673_checked.symm
  rw [row]
  decide
theorem root252Step8 : scan edge (search edge 3) 4 [1613] 1613 root252Tail8 winner =
    scan edge (search edge 3) 4 [1613] 1613 root252Tail9 winner := by
  rw [root252Tail8, scan, if_neg (by decide), root252Candidates8_checked]
  change scan edge (search edge 3) 4 [1613] 1613 root252Tail9
    (if 3 ≤ root252Candidates8.length then search edge 3 [1613, 4673] root252Candidates8 6286 winner else winner) = _
  rw [if_neg (by decide)]
def root252Candidates9 : List Nat := [11057, 18713]
set_option maxRecDepth 8192 in
theorem root252Candidates9_checked : candidates edge 4 1613 4967 root252Tail10 winner.bound = root252Candidates9 := by
  unfold candidates
  have row : edge 4967 = (SortedNatTree.contains FastRows.row4967) := by
    change Data.row4967.contains = (SortedNatTree.contains FastRows.row4967)
    exact FastRows.row4967_checked.symm
  rw [row]
  decide
theorem root252Step9 : scan edge (search edge 3) 4 [1613] 1613 root252Tail9 winner =
    scan edge (search edge 3) 4 [1613] 1613 root252Tail10 winner := by
  rw [root252Tail9, scan, if_neg (by decide), root252Candidates9_checked]
  change scan edge (search edge 3) 4 [1613] 1613 root252Tail10
    (if 3 ≤ root252Candidates9.length then search edge 3 [1613, 4967] root252Candidates9 6580 winner else winner) = _
  rw [if_neg (by decide)]
def root252Candidates10 : List Nat := [18713]
set_option maxRecDepth 8192 in
theorem root252Candidates10_checked : candidates edge 4 1613 5297 root252Tail11 winner.bound = root252Candidates10 := by
  unfold candidates
  have row : edge 5297 = (SortedNatTree.contains FastRows.row5297) := by
    change Data.row5297.contains = (SortedNatTree.contains FastRows.row5297)
    exact FastRows.row5297_checked.symm
  rw [row]
  decide
theorem root252Step10 : scan edge (search edge 3) 4 [1613] 1613 root252Tail10 winner =
    scan edge (search edge 3) 4 [1613] 1613 root252Tail11 winner := by
  rw [root252Tail10, scan, if_neg (by decide), root252Candidates10_checked]
  change scan edge (search edge 3) 4 [1613] 1613 root252Tail11
    (if 3 ≤ root252Candidates10.length then search edge 3 [1613, 5297] root252Candidates10 6910 winner else winner) = _
  rw [if_neg (by decide)]
def root252Candidates11 : List Nat := [14957, 16481]
set_option maxRecDepth 8192 in
theorem root252Candidates11_checked : candidates edge 4 1613 5813 root252Tail12 winner.bound = root252Candidates11 := by
  unfold candidates
  have row : edge 5813 = (SortedNatTree.contains FastRows.row5813) := by
    change Data.row5813.contains = (SortedNatTree.contains FastRows.row5813)
    exact FastRows.row5813_checked.symm
  rw [row]
  decide
theorem root252Step11 : scan edge (search edge 3) 4 [1613] 1613 root252Tail11 winner =
    scan edge (search edge 3) 4 [1613] 1613 root252Tail12 winner := by
  rw [root252Tail11, scan, if_neg (by decide), root252Candidates11_checked]
  change scan edge (search edge 3) 4 [1613] 1613 root252Tail12
    (if 3 ≤ root252Candidates11.length then search edge 3 [1613, 5813] root252Candidates11 7426 winner else winner) = _
  rw [if_neg (by decide)]
theorem root252Scan12 : scan edge (search edge 3) 4 [1613] 1613 root252Tail12 winner = winner := by rw [root252Tail12, scan, if_pos (by decide)]
theorem root252Scan11 : scan edge (search edge 3) 4 [1613] 1613 root252Tail11 winner = winner :=
  (root252Step11).trans root252Scan12
theorem root252Scan10 : scan edge (search edge 3) 4 [1613] 1613 root252Tail10 winner = winner :=
  (root252Step10).trans root252Scan11
theorem root252Scan9 : scan edge (search edge 3) 4 [1613] 1613 root252Tail9 winner = winner :=
  (root252Step9).trans root252Scan10
theorem root252Scan8 : scan edge (search edge 3) 4 [1613] 1613 root252Tail8 winner = winner :=
  (root252Step8).trans root252Scan9
theorem root252Scan7 : scan edge (search edge 3) 4 [1613] 1613 root252Tail7 winner = winner :=
  (root252Step7).trans root252Scan8
theorem root252Scan6 : scan edge (search edge 3) 4 [1613] 1613 root252Tail6 winner = winner :=
  (root252Step6).trans root252Scan7
theorem root252Scan5 : scan edge (search edge 3) 4 [1613] 1613 root252Tail5 winner = winner :=
  (root252Step5).trans root252Scan6
theorem root252Scan4 : scan edge (search edge 3) 4 [1613] 1613 root252Tail4 winner = winner :=
  (root252Step4).trans root252Scan5
theorem root252Scan3 : scan edge (search edge 3) 4 [1613] 1613 root252Tail3 winner = winner :=
  (root252Step3).trans root252Scan4
theorem root252Scan2 : scan edge (search edge 3) 4 [1613] 1613 root252Tail2 winner = winner :=
  (root252Step2).trans root252Scan3
theorem root252Scan1 : scan edge (search edge 3) 4 [1613] 1613 root252Tail1 winner = winner :=
  (root252Step1).trans root252Scan2
theorem root252Scan0 : scan edge (search edge 3) 4 [1613] 1613 root252Tail0 winner = winner :=
  (root252Step0).trans root252Scan1
set_option maxRecDepth 4096 in
theorem root252Branch : search edge 4 [1613] root252Candidates 1613 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1613] 1613 root252Tail0 winner = winner
  exact root252Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
