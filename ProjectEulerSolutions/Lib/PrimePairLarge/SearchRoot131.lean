import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root131Candidates : List Nat := Data.row757
set_option maxRecDepth 8192 in
theorem root131Candidates_checked : candidates edge 5 0 757 tail132 winner.bound = root131Candidates := by
  apply candidates_eq_row edge 5 0 757 tail132 root131Candidates winner.bound tail132_ordered
  · decide
  · decide
  · funext q
    rfl
def root131Tail59 : List Nat := []
def root131Tail58 : List Nat := 24799 :: root131Tail59
def root131Tail57 : List Nat := 23929 :: root131Tail58
def root131Tail56 : List Nat := 23677 :: root131Tail57
def root131Tail55 : List Nat := 23671 :: root131Tail56
def root131Tail54 : List Nat := 23041 :: root131Tail55
def root131Tail53 : List Nat := 22963 :: root131Tail54
def root131Tail52 : List Nat := 22921 :: root131Tail53
def root131Tail51 : List Nat := 22453 :: root131Tail52
def root131Tail50 : List Nat := 21991 :: root131Tail51
def root131Tail49 : List Nat := 21391 :: root131Tail50
def root131Tail48 : List Nat := 21157 :: root131Tail49
def root131Tail47 : List Nat := 21067 :: root131Tail48
def root131Tail46 : List Nat := 20173 :: root131Tail47
def root131Tail45 : List Nat := 20089 :: root131Tail46
def root131Tail44 : List Nat := 19843 :: root131Tail45
def root131Tail43 : List Nat := 19183 :: root131Tail44
def root131Tail42 : List Nat := 18451 :: root131Tail43
def root131Tail41 : List Nat := 18427 :: root131Tail42
def root131Tail40 : List Nat := 18199 :: root131Tail41
def root131Tail39 : List Nat := 17209 :: root131Tail40
def root131Tail38 : List Nat := 17107 :: root131Tail39
def root131Tail37 : List Nat := 17041 :: root131Tail38
def root131Tail36 : List Nat := 16519 :: root131Tail37
def root131Tail35 : List Nat := 15139 :: root131Tail36
def root131Tail34 : List Nat := 14461 :: root131Tail35
def root131Tail33 : List Nat := 14437 :: root131Tail34
def root131Tail32 : List Nat := 13633 :: root131Tail33
def root131Tail31 : List Nat := 12829 :: root131Tail32
def root131Tail30 : List Nat := 12421 :: root131Tail31
def root131Tail29 : List Nat := 11959 :: root131Tail30
def root131Tail28 : List Nat := 11497 :: root131Tail29
def root131Tail27 : List Nat := 11437 :: root131Tail28
def root131Tail26 : List Nat := 10771 :: root131Tail27
def root131Tail25 : List Nat := 10651 :: root131Tail26
def root131Tail24 : List Nat := 10567 :: root131Tail25
def root131Tail23 : List Nat := 9769 :: root131Tail24
def root131Tail22 : List Nat := 9109 :: root131Tail23
def root131Tail21 : List Nat := 8971 :: root131Tail22
def root131Tail20 : List Nat := 8731 :: root131Tail21
def root131Tail19 : List Nat := 8269 :: root131Tail20
def root131Tail18 : List Nat := 7321 :: root131Tail19
def root131Tail17 : List Nat := 6661 :: root131Tail18
def root131Tail16 : List Nat := 6619 :: root131Tail17
def root131Tail15 : List Nat := 6211 :: root131Tail16
def root131Tail14 : List Nat := 6043 :: root131Tail15
def root131Tail13 : List Nat := 5791 :: root131Tail14
def root131Tail12 : List Nat := 5653 :: root131Tail13
def root131Tail11 : List Nat := 5569 :: root131Tail12
def root131Tail10 : List Nat := 5077 :: root131Tail11
def root131Tail9 : List Nat := 4993 :: root131Tail10
def root131Tail8 : List Nat := 4549 :: root131Tail9
def root131Tail7 : List Nat := 3931 :: root131Tail8
def root131Tail6 : List Nat := 3733 :: root131Tail7
def root131Tail5 : List Nat := 3217 :: root131Tail6
def root131Tail4 : List Nat := 3037 :: root131Tail5
def root131Tail3 : List Nat := 1423 :: root131Tail4
def root131Tail2 : List Nat := 1231 :: root131Tail3
def root131Tail1 : List Nat := 829 :: root131Tail2
def root131Tail0 : List Nat := 811 :: root131Tail1
def root131Candidates0 : List Nat := [5791, 10567]
set_option maxRecDepth 8192 in
theorem root131Candidates0_checked : candidates edge 4 757 811 root131Tail1 winner.bound = root131Candidates0 := by
  unfold candidates
  have row : edge 811 = (SortedNatTree.contains FastRows.row811) := by
    change Data.row811.contains = (SortedNatTree.contains FastRows.row811)
    exact FastRows.row811_checked.symm
  rw [row]
  decide
theorem root131Step0 : scan edge (search edge 3) 4 [757] 757 root131Tail0 winner =
    scan edge (search edge 3) 4 [757] 757 root131Tail1 winner := by
  rw [root131Tail0, scan, if_neg (by decide), root131Candidates0_checked]
  change scan edge (search edge 3) 4 [757] 757 root131Tail1
    (if 3 ≤ root131Candidates0.length then search edge 3 [757, 811] root131Candidates0 1568 winner else winner) = _
  rw [if_neg (by decide)]
def root131Candidates1 : List Nat := [7321, 23041]
set_option maxRecDepth 8192 in
theorem root131Candidates1_checked : candidates edge 4 757 829 root131Tail2 winner.bound = root131Candidates1 := by
  unfold candidates
  have row : edge 829 = (SortedNatTree.contains FastRows.row829) := by
    change Data.row829.contains = (SortedNatTree.contains FastRows.row829)
    exact FastRows.row829_checked.symm
  rw [row]
  decide
theorem root131Step1 : scan edge (search edge 3) 4 [757] 757 root131Tail1 winner =
    scan edge (search edge 3) 4 [757] 757 root131Tail2 winner := by
  rw [root131Tail1, scan, if_neg (by decide), root131Candidates1_checked]
  change scan edge (search edge 3) 4 [757] 757 root131Tail2
    (if 3 ≤ root131Candidates1.length then search edge 3 [757, 829] root131Candidates1 1586 winner else winner) = _
  rw [if_neg (by decide)]
def root131Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root131Candidates2_checked : candidates edge 4 757 1231 root131Tail3 winner.bound = root131Candidates2 := by
  unfold candidates
  have row : edge 1231 = (SortedNatTree.contains FastRows.row1231) := by
    change Data.row1231.contains = (SortedNatTree.contains FastRows.row1231)
    exact FastRows.row1231_checked.symm
  rw [row]
  decide
theorem root131Step2 : scan edge (search edge 3) 4 [757] 757 root131Tail2 winner =
    scan edge (search edge 3) 4 [757] 757 root131Tail3 winner := by
  rw [root131Tail2, scan, if_neg (by decide), root131Candidates2_checked]
  change scan edge (search edge 3) 4 [757] 757 root131Tail3
    (if 3 ≤ root131Candidates2.length then search edge 3 [757, 1231] root131Candidates2 1988 winner else winner) = _
  rw [if_neg (by decide)]
def root131Candidates3 : List Nat := [5791, 9769]
set_option maxRecDepth 8192 in
theorem root131Candidates3_checked : candidates edge 4 757 1423 root131Tail4 winner.bound = root131Candidates3 := by
  unfold candidates
  have row : edge 1423 = (SortedNatTree.contains FastRows.row1423) := by
    change Data.row1423.contains = (SortedNatTree.contains FastRows.row1423)
    exact FastRows.row1423_checked.symm
  rw [row]
  decide
theorem root131Step3 : scan edge (search edge 3) 4 [757] 757 root131Tail3 winner =
    scan edge (search edge 3) 4 [757] 757 root131Tail4 winner := by
  rw [root131Tail3, scan, if_neg (by decide), root131Candidates3_checked]
  change scan edge (search edge 3) 4 [757] 757 root131Tail4
    (if 3 ≤ root131Candidates3.length then search edge 3 [757, 1423] root131Candidates3 2180 winner else winner) = _
  rw [if_neg (by decide)]
def root131Candidates4 : List Nat := [11437, 17209, 20089]
set_option maxRecDepth 8192 in
theorem root131Candidates4_checked : candidates edge 4 757 3037 root131Tail5 winner.bound = root131Candidates4 := by
  unfold candidates
  have row : edge 3037 = (SortedNatTree.contains FastRows.row3037) := by
    change Data.row3037.contains = (SortedNatTree.contains FastRows.row3037)
    exact FastRows.row3037_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root131Child4 : search edge 3 [757, 3037] root131Candidates4 3794 winner = winner := by decide
theorem root131Step4 : scan edge (search edge 3) 4 [757] 757 root131Tail4 winner =
    scan edge (search edge 3) 4 [757] 757 root131Tail5 winner := by
  rw [root131Tail4, scan, if_neg (by decide), root131Candidates4_checked]
  change scan edge (search edge 3) 4 [757] 757 root131Tail5
    (if 3 ≤ root131Candidates4.length then search edge 3 [757, 3037] root131Candidates4 3794 winner else winner) = _
  rw [if_pos (by decide), root131Child4]
def root131Candidates5 : List Nat := [3931, 6211, 6619, 12421]
set_option maxRecDepth 8192 in
theorem root131Candidates5_checked : candidates edge 4 757 3217 root131Tail6 winner.bound = root131Candidates5 := by
  unfold candidates
  have row : edge 3217 = (SortedNatTree.contains FastRows.row3217) := by
    change Data.row3217.contains = (SortedNatTree.contains FastRows.row3217)
    exact FastRows.row3217_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root131Child5 : search edge 3 [757, 3217] root131Candidates5 3974 winner = winner := by decide
theorem root131Step5 : scan edge (search edge 3) 4 [757] 757 root131Tail5 winner =
    scan edge (search edge 3) 4 [757] 757 root131Tail6 winner := by
  rw [root131Tail5, scan, if_neg (by decide), root131Candidates5_checked]
  change scan edge (search edge 3) 4 [757] 757 root131Tail6
    (if 3 ≤ root131Candidates5.length then search edge 3 [757, 3217] root131Candidates5 3974 winner else winner) = _
  rw [if_pos (by decide), root131Child5]
def root131Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root131Candidates6_checked : candidates edge 4 757 3733 root131Tail7 winner.bound = root131Candidates6 := by
  unfold candidates
  have row : edge 3733 = (SortedNatTree.contains FastRows.row3733) := by
    change Data.row3733.contains = (SortedNatTree.contains FastRows.row3733)
    exact FastRows.row3733_checked.symm
  rw [row]
  decide
theorem root131Step6 : scan edge (search edge 3) 4 [757] 757 root131Tail6 winner =
    scan edge (search edge 3) 4 [757] 757 root131Tail7 winner := by
  rw [root131Tail6, scan, if_neg (by decide), root131Candidates6_checked]
  change scan edge (search edge 3) 4 [757] 757 root131Tail7
    (if 3 ≤ root131Candidates6.length then search edge 3 [757, 3733] root131Candidates6 4490 winner else winner) = _
  rw [if_neg (by decide)]
def root131Candidates7 : List Nat := [6661]
set_option maxRecDepth 8192 in
theorem root131Candidates7_checked : candidates edge 4 757 3931 root131Tail8 winner.bound = root131Candidates7 := by
  unfold candidates
  have row : edge 3931 = (SortedNatTree.contains FastRows.row3931) := by
    change Data.row3931.contains = (SortedNatTree.contains FastRows.row3931)
    exact FastRows.row3931_checked.symm
  rw [row]
  decide
theorem root131Step7 : scan edge (search edge 3) 4 [757] 757 root131Tail7 winner =
    scan edge (search edge 3) 4 [757] 757 root131Tail8 winner := by
  rw [root131Tail7, scan, if_neg (by decide), root131Candidates7_checked]
  change scan edge (search edge 3) 4 [757] 757 root131Tail8
    (if 3 ≤ root131Candidates7.length then search edge 3 [757, 3931] root131Candidates7 4688 winner else winner) = _
  rw [if_neg (by decide)]
def root131Candidates8 : List Nat := [14461, 17041]
set_option maxRecDepth 8192 in
theorem root131Candidates8_checked : candidates edge 4 757 4549 root131Tail9 winner.bound = root131Candidates8 := by
  unfold candidates
  have row : edge 4549 = (SortedNatTree.contains FastRows.row4549) := by
    change Data.row4549.contains = (SortedNatTree.contains FastRows.row4549)
    exact FastRows.row4549_checked.symm
  rw [row]
  decide
theorem root131Step8 : scan edge (search edge 3) 4 [757] 757 root131Tail8 winner =
    scan edge (search edge 3) 4 [757] 757 root131Tail9 winner := by
  rw [root131Tail8, scan, if_neg (by decide), root131Candidates8_checked]
  change scan edge (search edge 3) 4 [757] 757 root131Tail9
    (if 3 ≤ root131Candidates8.length then search edge 3 [757, 4549] root131Candidates8 5306 winner else winner) = _
  rw [if_neg (by decide)]
def root131Candidates9 : List Nat := [8269, 8971, 14461]
set_option maxRecDepth 8192 in
theorem root131Candidates9_checked : candidates edge 4 757 4993 root131Tail10 winner.bound = root131Candidates9 := by
  unfold candidates
  have row : edge 4993 = (SortedNatTree.contains FastRows.row4993) := by
    change Data.row4993.contains = (SortedNatTree.contains FastRows.row4993)
    exact FastRows.row4993_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root131Child9 : search edge 3 [757, 4993] root131Candidates9 5750 winner = winner := by decide
theorem root131Step9 : scan edge (search edge 3) 4 [757] 757 root131Tail9 winner =
    scan edge (search edge 3) 4 [757] 757 root131Tail10 winner := by
  rw [root131Tail9, scan, if_neg (by decide), root131Candidates9_checked]
  change scan edge (search edge 3) 4 [757] 757 root131Tail10
    (if 3 ≤ root131Candidates9.length then search edge 3 [757, 4993] root131Candidates9 5750 winner else winner) = _
  rw [if_pos (by decide), root131Child9]
def root131Candidates10 : List Nat := [6043]
set_option maxRecDepth 8192 in
theorem root131Candidates10_checked : candidates edge 4 757 5077 root131Tail11 winner.bound = root131Candidates10 := by
  unfold candidates
  have row : edge 5077 = (SortedNatTree.contains FastRows.row5077) := by
    change Data.row5077.contains = (SortedNatTree.contains FastRows.row5077)
    exact FastRows.row5077_checked.symm
  rw [row]
  decide
theorem root131Step10 : scan edge (search edge 3) 4 [757] 757 root131Tail10 winner =
    scan edge (search edge 3) 4 [757] 757 root131Tail11 winner := by
  rw [root131Tail10, scan, if_neg (by decide), root131Candidates10_checked]
  change scan edge (search edge 3) 4 [757] 757 root131Tail11
    (if 3 ≤ root131Candidates10.length then search edge 3 [757, 5077] root131Candidates10 5834 winner else winner) = _
  rw [if_neg (by decide)]
def root131Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root131Candidates11_checked : candidates edge 4 757 5569 root131Tail12 winner.bound = root131Candidates11 := by
  unfold candidates
  have row : edge 5569 = (SortedNatTree.contains FastRows.row5569) := by
    change Data.row5569.contains = (SortedNatTree.contains FastRows.row5569)
    exact FastRows.row5569_checked.symm
  rw [row]
  decide
theorem root131Step11 : scan edge (search edge 3) 4 [757] 757 root131Tail11 winner =
    scan edge (search edge 3) 4 [757] 757 root131Tail12 winner := by
  rw [root131Tail11, scan, if_neg (by decide), root131Candidates11_checked]
  change scan edge (search edge 3) 4 [757] 757 root131Tail12
    (if 3 ≤ root131Candidates11.length then search edge 3 [757, 5569] root131Candidates11 6326 winner else winner) = _
  rw [if_neg (by decide)]
def root131Candidates12 : List Nat := [6211, 6619, 8269, 8731]
set_option maxRecDepth 8192 in
theorem root131Candidates12_checked : candidates edge 4 757 5653 root131Tail13 winner.bound = root131Candidates12 := by
  unfold candidates
  have row : edge 5653 = (SortedNatTree.contains FastRows.row5653) := by
    change Data.row5653.contains = (SortedNatTree.contains FastRows.row5653)
    exact FastRows.row5653_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root131Child12 : search edge 3 [757, 5653] root131Candidates12 6410 winner = winner := by decide
theorem root131Step12 : scan edge (search edge 3) 4 [757] 757 root131Tail12 winner =
    scan edge (search edge 3) 4 [757] 757 root131Tail13 winner := by
  rw [root131Tail12, scan, if_neg (by decide), root131Candidates12_checked]
  change scan edge (search edge 3) 4 [757] 757 root131Tail13
    (if 3 ≤ root131Candidates12.length then search edge 3 [757, 5653] root131Candidates12 6410 winner else winner) = _
  rw [if_pos (by decide), root131Child12]
def root131Candidates13 : List Nat := [14461]
set_option maxRecDepth 8192 in
theorem root131Candidates13_checked : candidates edge 4 757 5791 root131Tail14 winner.bound = root131Candidates13 := by
  unfold candidates
  have row : edge 5791 = (SortedNatTree.contains FastRows.row5791) := by
    change Data.row5791.contains = (SortedNatTree.contains FastRows.row5791)
    exact FastRows.row5791_checked.symm
  rw [row]
  decide
theorem root131Step13 : scan edge (search edge 3) 4 [757] 757 root131Tail13 winner =
    scan edge (search edge 3) 4 [757] 757 root131Tail14 winner := by
  rw [root131Tail13, scan, if_neg (by decide), root131Candidates13_checked]
  change scan edge (search edge 3) 4 [757] 757 root131Tail14
    (if 3 ≤ root131Candidates13.length then search edge 3 [757, 5791] root131Candidates13 6548 winner else winner) = _
  rw [if_neg (by decide)]
def root131Candidates14 : List Nat := [6619, 15139]
set_option maxRecDepth 8192 in
theorem root131Candidates14_checked : candidates edge 4 757 6043 root131Tail15 winner.bound = root131Candidates14 := by
  unfold candidates
  have row : edge 6043 = (SortedNatTree.contains FastRows.row6043) := by
    change Data.row6043.contains = (SortedNatTree.contains FastRows.row6043)
    exact FastRows.row6043_checked.symm
  rw [row]
  decide
theorem root131Step14 : scan edge (search edge 3) 4 [757] 757 root131Tail14 winner =
    scan edge (search edge 3) 4 [757] 757 root131Tail15 winner := by
  rw [root131Tail14, scan, if_neg (by decide), root131Candidates14_checked]
  change scan edge (search edge 3) 4 [757] 757 root131Tail15
    (if 3 ≤ root131Candidates14.length then search edge 3 [757, 6043] root131Candidates14 6800 winner else winner) = _
  rw [if_neg (by decide)]
def root131Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root131Candidates15_checked : candidates edge 4 757 6211 root131Tail16 winner.bound = root131Candidates15 := by
  unfold candidates
  have row : edge 6211 = (SortedNatTree.contains FastRows.row6211) := by
    change Data.row6211.contains = (SortedNatTree.contains FastRows.row6211)
    exact FastRows.row6211_checked.symm
  rw [row]
  decide
theorem root131Step15 : scan edge (search edge 3) 4 [757] 757 root131Tail15 winner =
    scan edge (search edge 3) 4 [757] 757 root131Tail16 winner := by
  rw [root131Tail15, scan, if_neg (by decide), root131Candidates15_checked]
  change scan edge (search edge 3) 4 [757] 757 root131Tail16
    (if 3 ≤ root131Candidates15.length then search edge 3 [757, 6211] root131Candidates15 6968 winner else winner) = _
  rw [if_neg (by decide)]
theorem root131Scan16 : scan edge (search edge 3) 4 [757] 757 root131Tail16 winner = winner := by rw [root131Tail16, scan, if_pos (by decide)]
theorem root131Scan15 : scan edge (search edge 3) 4 [757] 757 root131Tail15 winner = winner :=
  (root131Step15).trans root131Scan16
theorem root131Scan14 : scan edge (search edge 3) 4 [757] 757 root131Tail14 winner = winner :=
  (root131Step14).trans root131Scan15
theorem root131Scan13 : scan edge (search edge 3) 4 [757] 757 root131Tail13 winner = winner :=
  (root131Step13).trans root131Scan14
theorem root131Scan12 : scan edge (search edge 3) 4 [757] 757 root131Tail12 winner = winner :=
  (root131Step12).trans root131Scan13
theorem root131Scan11 : scan edge (search edge 3) 4 [757] 757 root131Tail11 winner = winner :=
  (root131Step11).trans root131Scan12
theorem root131Scan10 : scan edge (search edge 3) 4 [757] 757 root131Tail10 winner = winner :=
  (root131Step10).trans root131Scan11
theorem root131Scan9 : scan edge (search edge 3) 4 [757] 757 root131Tail9 winner = winner :=
  (root131Step9).trans root131Scan10
theorem root131Scan8 : scan edge (search edge 3) 4 [757] 757 root131Tail8 winner = winner :=
  (root131Step8).trans root131Scan9
theorem root131Scan7 : scan edge (search edge 3) 4 [757] 757 root131Tail7 winner = winner :=
  (root131Step7).trans root131Scan8
theorem root131Scan6 : scan edge (search edge 3) 4 [757] 757 root131Tail6 winner = winner :=
  (root131Step6).trans root131Scan7
theorem root131Scan5 : scan edge (search edge 3) 4 [757] 757 root131Tail5 winner = winner :=
  (root131Step5).trans root131Scan6
theorem root131Scan4 : scan edge (search edge 3) 4 [757] 757 root131Tail4 winner = winner :=
  (root131Step4).trans root131Scan5
theorem root131Scan3 : scan edge (search edge 3) 4 [757] 757 root131Tail3 winner = winner :=
  (root131Step3).trans root131Scan4
theorem root131Scan2 : scan edge (search edge 3) 4 [757] 757 root131Tail2 winner = winner :=
  (root131Step2).trans root131Scan3
theorem root131Scan1 : scan edge (search edge 3) 4 [757] 757 root131Tail1 winner = winner :=
  (root131Step1).trans root131Scan2
theorem root131Scan0 : scan edge (search edge 3) 4 [757] 757 root131Tail0 winner = winner :=
  (root131Step0).trans root131Scan1
set_option maxRecDepth 4096 in
theorem root131Branch : search edge 4 [757] root131Candidates 757 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [757] 757 root131Tail0 winner = winner
  exact root131Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
