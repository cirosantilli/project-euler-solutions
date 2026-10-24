import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root269Candidates : List Nat := Data.row1747
set_option maxRecDepth 8192 in
theorem root269Candidates_checked : candidates edge 5 0 1747 tail270 winner.bound = root269Candidates := by
  apply candidates_eq_row edge 5 0 1747 tail270 root269Candidates winner.bound tail270_ordered
  · decide
  · decide
  · funext q
    rfl
def root269Tail54 : List Nat := []
def root269Tail53 : List Nat := 24061 :: root269Tail54
def root269Tail52 : List Nat := 24049 :: root269Tail53
def root269Tail51 : List Nat := 23623 :: root269Tail52
def root269Tail50 : List Nat := 22501 :: root269Tail51
def root269Tail49 : List Nat := 20323 :: root269Tail50
def root269Tail48 : List Nat := 20269 :: root269Tail49
def root269Tail47 : List Nat := 19603 :: root269Tail48
def root269Tail46 : List Nat := 18211 :: root269Tail47
def root269Tail45 : List Nat := 18169 :: root269Tail46
def root269Tail44 : List Nat := 18121 :: root269Tail45
def root269Tail43 : List Nat := 17827 :: root269Tail44
def root269Tail42 : List Nat := 17707 :: root269Tail43
def root269Tail41 : List Nat := 17497 :: root269Tail42
def root269Tail40 : List Nat := 16573 :: root269Tail41
def root269Tail39 : List Nat := 16453 :: root269Tail40
def root269Tail38 : List Nat := 15643 :: root269Tail39
def root269Tail37 : List Nat := 14347 :: root269Tail38
def root269Tail36 : List Nat := 13291 :: root269Tail37
def root269Tail35 : List Nat := 13249 :: root269Tail36
def root269Tail34 : List Nat := 13093 :: root269Tail35
def root269Tail33 : List Nat := 12973 :: root269Tail34
def root269Tail32 : List Nat := 12763 :: root269Tail33
def root269Tail31 : List Nat := 12457 :: root269Tail32
def root269Tail30 : List Nat := 12007 :: root269Tail31
def root269Tail29 : List Nat := 9811 :: root269Tail30
def root269Tail28 : List Nat := 9643 :: root269Tail29
def root269Tail27 : List Nat := 9403 :: root269Tail28
def root269Tail26 : List Nat := 9157 :: root269Tail27
def root269Tail25 : List Nat := 9151 :: root269Tail26
def root269Tail24 : List Nat := 8731 :: root269Tail25
def root269Tail23 : List Nat := 8599 :: root269Tail24
def root269Tail22 : List Nat := 8563 :: root269Tail23
def root269Tail21 : List Nat := 8011 :: root269Tail22
def root269Tail20 : List Nat := 7639 :: root269Tail21
def root269Tail19 : List Nat := 7477 :: root269Tail20
def root269Tail18 : List Nat := 7177 :: root269Tail19
def root269Tail17 : List Nat := 7129 :: root269Tail18
def root269Tail16 : List Nat := 6967 :: root269Tail17
def root269Tail15 : List Nat := 6421 :: root269Tail16
def root269Tail14 : List Nat := 6367 :: root269Tail15
def root269Tail13 : List Nat := 6037 :: root269Tail14
def root269Tail12 : List Nat := 5737 :: root269Tail13
def root269Tail11 : List Nat := 5569 :: root269Tail12
def root269Tail10 : List Nat := 4993 :: root269Tail11
def root269Tail9 : List Nat := 4231 :: root269Tail10
def root269Tail8 : List Nat := 4099 :: root269Tail9
def root269Tail7 : List Nat := 3889 :: root269Tail8
def root269Tail6 : List Nat := 3769 :: root269Tail7
def root269Tail5 : List Nat := 3691 :: root269Tail6
def root269Tail4 : List Nat := 2671 :: root269Tail5
def root269Tail3 : List Nat := 2179 :: root269Tail4
def root269Tail2 : List Nat := 2053 :: root269Tail3
def root269Tail1 : List Nat := 1933 :: root269Tail2
def root269Tail0 : List Nat := 1873 :: root269Tail1
def root269Candidates0 : List Nat := [5569, 8011]
set_option maxRecDepth 8192 in
theorem root269Candidates0_checked : candidates edge 4 1747 1873 root269Tail1 winner.bound = root269Candidates0 := by
  unfold candidates
  have row : edge 1873 = (SortedNatTree.contains FastRows.row1873) := by
    change Data.row1873.contains = (SortedNatTree.contains FastRows.row1873)
    exact FastRows.row1873_checked.symm
  rw [row]
  decide
theorem root269Step0 : scan edge (search edge 3) 4 [1747] 1747 root269Tail0 winner =
    scan edge (search edge 3) 4 [1747] 1747 root269Tail1 winner := by
  rw [root269Tail0, scan, if_neg (by decide), root269Candidates0_checked]
  change scan edge (search edge 3) 4 [1747] 1747 root269Tail1
    (if 3 ≤ root269Candidates0.length then search edge 3 [1747, 1873] root269Candidates0 3620 winner else winner) = _
  rw [if_neg (by decide)]
def root269Candidates1 : List Nat := [7639]
set_option maxRecDepth 8192 in
theorem root269Candidates1_checked : candidates edge 4 1747 1933 root269Tail2 winner.bound = root269Candidates1 := by
  unfold candidates
  have row : edge 1933 = (SortedNatTree.contains FastRows.row1933) := by
    change Data.row1933.contains = (SortedNatTree.contains FastRows.row1933)
    exact FastRows.row1933_checked.symm
  rw [row]
  decide
theorem root269Step1 : scan edge (search edge 3) 4 [1747] 1747 root269Tail1 winner =
    scan edge (search edge 3) 4 [1747] 1747 root269Tail2 winner := by
  rw [root269Tail1, scan, if_neg (by decide), root269Candidates1_checked]
  change scan edge (search edge 3) 4 [1747] 1747 root269Tail2
    (if 3 ≤ root269Candidates1.length then search edge 3 [1747, 1933] root269Candidates1 3680 winner else winner) = _
  rw [if_neg (by decide)]
def root269Candidates2 : List Nat := [6421]
set_option maxRecDepth 8192 in
theorem root269Candidates2_checked : candidates edge 4 1747 2053 root269Tail3 winner.bound = root269Candidates2 := by
  unfold candidates
  have row : edge 2053 = (SortedNatTree.contains FastRows.row2053) := by
    change Data.row2053.contains = (SortedNatTree.contains FastRows.row2053)
    exact FastRows.row2053_checked.symm
  rw [row]
  decide
theorem root269Step2 : scan edge (search edge 3) 4 [1747] 1747 root269Tail2 winner =
    scan edge (search edge 3) 4 [1747] 1747 root269Tail3 winner := by
  rw [root269Tail2, scan, if_neg (by decide), root269Candidates2_checked]
  change scan edge (search edge 3) 4 [1747] 1747 root269Tail3
    (if 3 ≤ root269Candidates2.length then search edge 3 [1747, 2053] root269Candidates2 3800 winner else winner) = _
  rw [if_neg (by decide)]
def root269Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root269Candidates3_checked : candidates edge 4 1747 2179 root269Tail4 winner.bound = root269Candidates3 := by
  unfold candidates
  have row : edge 2179 = (SortedNatTree.contains FastRows.row2179) := by
    change Data.row2179.contains = (SortedNatTree.contains FastRows.row2179)
    exact FastRows.row2179_checked.symm
  rw [row]
  decide
theorem root269Step3 : scan edge (search edge 3) 4 [1747] 1747 root269Tail3 winner =
    scan edge (search edge 3) 4 [1747] 1747 root269Tail4 winner := by
  rw [root269Tail3, scan, if_neg (by decide), root269Candidates3_checked]
  change scan edge (search edge 3) 4 [1747] 1747 root269Tail4
    (if 3 ≤ root269Candidates3.length then search edge 3 [1747, 2179] root269Candidates3 3926 winner else winner) = _
  rw [if_neg (by decide)]
def root269Candidates4 : List Nat := [3769, 4231, 7129, 9811]
set_option maxRecDepth 8192 in
theorem root269Candidates4_checked : candidates edge 4 1747 2671 root269Tail5 winner.bound = root269Candidates4 := by
  unfold candidates
  have row : edge 2671 = (SortedNatTree.contains FastRows.row2671) := by
    change Data.row2671.contains = (SortedNatTree.contains FastRows.row2671)
    exact FastRows.row2671_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root269Child4 : search edge 3 [1747, 2671] root269Candidates4 4418 winner = winner := by decide
theorem root269Step4 : scan edge (search edge 3) 4 [1747] 1747 root269Tail4 winner =
    scan edge (search edge 3) 4 [1747] 1747 root269Tail5 winner := by
  rw [root269Tail4, scan, if_neg (by decide), root269Candidates4_checked]
  change scan edge (search edge 3) 4 [1747] 1747 root269Tail5
    (if 3 ≤ root269Candidates4.length then search edge 3 [1747, 2671] root269Candidates4 4418 winner else winner) = _
  rw [if_pos (by decide), root269Child4]
def root269Candidates5 : List Nat := [7477, 8731, 13291]
set_option maxRecDepth 8192 in
theorem root269Candidates5_checked : candidates edge 4 1747 3691 root269Tail6 winner.bound = root269Candidates5 := by
  unfold candidates
  have row : edge 3691 = (SortedNatTree.contains FastRows.row3691) := by
    change Data.row3691.contains = (SortedNatTree.contains FastRows.row3691)
    exact FastRows.row3691_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root269Child5 : search edge 3 [1747, 3691] root269Candidates5 5438 winner = winner := by decide
theorem root269Step5 : scan edge (search edge 3) 4 [1747] 1747 root269Tail5 winner =
    scan edge (search edge 3) 4 [1747] 1747 root269Tail6 winner := by
  rw [root269Tail5, scan, if_neg (by decide), root269Candidates5_checked]
  change scan edge (search edge 3) 4 [1747] 1747 root269Tail6
    (if 3 ≤ root269Candidates5.length then search edge 3 [1747, 3691] root269Candidates5 5438 winner else winner) = _
  rw [if_pos (by decide), root269Child5]
def root269Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root269Candidates6_checked : candidates edge 4 1747 3769 root269Tail7 winner.bound = root269Candidates6 := by
  unfold candidates
  have row : edge 3769 = (SortedNatTree.contains FastRows.row3769) := by
    change Data.row3769.contains = (SortedNatTree.contains FastRows.row3769)
    exact FastRows.row3769_checked.symm
  rw [row]
  decide
theorem root269Step6 : scan edge (search edge 3) 4 [1747] 1747 root269Tail6 winner =
    scan edge (search edge 3) 4 [1747] 1747 root269Tail7 winner := by
  rw [root269Tail6, scan, if_neg (by decide), root269Candidates6_checked]
  change scan edge (search edge 3) 4 [1747] 1747 root269Tail7
    (if 3 ≤ root269Candidates6.length then search edge 3 [1747, 3769] root269Candidates6 5516 winner else winner) = _
  rw [if_neg (by decide)]
def root269Candidates7 : List Nat := [7477, 12973, 17827, 19603]
set_option maxRecDepth 8192 in
theorem root269Candidates7_checked : candidates edge 4 1747 3889 root269Tail8 winner.bound = root269Candidates7 := by
  unfold candidates
  have row : edge 3889 = (SortedNatTree.contains FastRows.row3889) := by
    change Data.row3889.contains = (SortedNatTree.contains FastRows.row3889)
    exact FastRows.row3889_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root269Child7 : search edge 3 [1747, 3889] root269Candidates7 5636 winner = winner := by decide
theorem root269Step7 : scan edge (search edge 3) 4 [1747] 1747 root269Tail7 winner =
    scan edge (search edge 3) 4 [1747] 1747 root269Tail8 winner := by
  rw [root269Tail7, scan, if_neg (by decide), root269Candidates7_checked]
  change scan edge (search edge 3) 4 [1747] 1747 root269Tail8
    (if 3 ≤ root269Candidates7.length then search edge 3 [1747, 3889] root269Candidates7 5636 winner else winner) = _
  rw [if_pos (by decide), root269Child7]
def root269Candidates8 : List Nat := [18211]
set_option maxRecDepth 8192 in
theorem root269Candidates8_checked : candidates edge 4 1747 4099 root269Tail9 winner.bound = root269Candidates8 := by
  unfold candidates
  have row : edge 4099 = (SortedNatTree.contains FastRows.row4099) := by
    change Data.row4099.contains = (SortedNatTree.contains FastRows.row4099)
    exact FastRows.row4099_checked.symm
  rw [row]
  decide
theorem root269Step8 : scan edge (search edge 3) 4 [1747] 1747 root269Tail8 winner =
    scan edge (search edge 3) 4 [1747] 1747 root269Tail9 winner := by
  rw [root269Tail8, scan, if_neg (by decide), root269Candidates8_checked]
  change scan edge (search edge 3) 4 [1747] 1747 root269Tail9
    (if 3 ≤ root269Candidates8.length then search edge 3 [1747, 4099] root269Candidates8 5846 winner else winner) = _
  rw [if_neg (by decide)]
def root269Candidates9 : List Nat := [6037, 12763, 18169]
set_option maxRecDepth 8192 in
theorem root269Candidates9_checked : candidates edge 4 1747 4231 root269Tail10 winner.bound = root269Candidates9 := by
  unfold candidates
  have row : edge 4231 = (SortedNatTree.contains FastRows.row4231) := by
    change Data.row4231.contains = (SortedNatTree.contains FastRows.row4231)
    exact FastRows.row4231_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root269Child9 : search edge 3 [1747, 4231] root269Candidates9 5978 winner = winner := by decide
theorem root269Step9 : scan edge (search edge 3) 4 [1747] 1747 root269Tail9 winner =
    scan edge (search edge 3) 4 [1747] 1747 root269Tail10 winner := by
  rw [root269Tail9, scan, if_neg (by decide), root269Candidates9_checked]
  change scan edge (search edge 3) 4 [1747] 1747 root269Tail10
    (if 3 ≤ root269Candidates9.length then search edge 3 [1747, 4231] root269Candidates9 5978 winner else winner) = _
  rw [if_pos (by decide), root269Child9]
def root269Candidates10 : List Nat := [8563]
set_option maxRecDepth 8192 in
theorem root269Candidates10_checked : candidates edge 4 1747 4993 root269Tail11 winner.bound = root269Candidates10 := by
  unfold candidates
  have row : edge 4993 = (SortedNatTree.contains FastRows.row4993) := by
    change Data.row4993.contains = (SortedNatTree.contains FastRows.row4993)
    exact FastRows.row4993_checked.symm
  rw [row]
  decide
theorem root269Step10 : scan edge (search edge 3) 4 [1747] 1747 root269Tail10 winner =
    scan edge (search edge 3) 4 [1747] 1747 root269Tail11 winner := by
  rw [root269Tail10, scan, if_neg (by decide), root269Candidates10_checked]
  change scan edge (search edge 3) 4 [1747] 1747 root269Tail11
    (if 3 ≤ root269Candidates10.length then search edge 3 [1747, 4993] root269Candidates10 6740 winner else winner) = _
  rw [if_neg (by decide)]
def root269Candidates11 : List Nat := [12007]
set_option maxRecDepth 8192 in
theorem root269Candidates11_checked : candidates edge 4 1747 5569 root269Tail12 winner.bound = root269Candidates11 := by
  unfold candidates
  have row : edge 5569 = (SortedNatTree.contains FastRows.row5569) := by
    change Data.row5569.contains = (SortedNatTree.contains FastRows.row5569)
    exact FastRows.row5569_checked.symm
  rw [row]
  decide
theorem root269Step11 : scan edge (search edge 3) 4 [1747] 1747 root269Tail11 winner =
    scan edge (search edge 3) 4 [1747] 1747 root269Tail12 winner := by
  rw [root269Tail11, scan, if_neg (by decide), root269Candidates11_checked]
  change scan edge (search edge 3) 4 [1747] 1747 root269Tail12
    (if 3 ≤ root269Candidates11.length then search edge 3 [1747, 5569] root269Candidates11 7316 winner else winner) = _
  rw [if_neg (by decide)]
def root269Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root269Candidates12_checked : candidates edge 4 1747 5737 root269Tail13 winner.bound = root269Candidates12 := by
  unfold candidates
  have row : edge 5737 = (SortedNatTree.contains FastRows.row5737) := by
    change Data.row5737.contains = (SortedNatTree.contains FastRows.row5737)
    exact FastRows.row5737_checked.symm
  rw [row]
  decide
theorem root269Step12 : scan edge (search edge 3) 4 [1747] 1747 root269Tail12 winner =
    scan edge (search edge 3) 4 [1747] 1747 root269Tail13 winner := by
  rw [root269Tail12, scan, if_neg (by decide), root269Candidates12_checked]
  change scan edge (search edge 3) 4 [1747] 1747 root269Tail13
    (if 3 ≤ root269Candidates12.length then search edge 3 [1747, 5737] root269Candidates12 7484 winner else winner) = _
  rw [if_neg (by decide)]
def root269Candidates13 : List Nat := [9643, 12457, 13093, 13291, 16453, 17827, 18169]
set_option maxRecDepth 8192 in
theorem root269Candidates13_checked : candidates edge 4 1747 6037 root269Tail14 winner.bound = root269Candidates13 := by
  unfold candidates
  have row : edge 6037 = (SortedNatTree.contains FastRows.row6037) := by
    change Data.row6037.contains = (SortedNatTree.contains FastRows.row6037)
    exact FastRows.row6037_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root269Child13 : search edge 3 [1747, 6037] root269Candidates13 7784 winner = winner := by decide
theorem root269Step13 : scan edge (search edge 3) 4 [1747] 1747 root269Tail13 winner =
    scan edge (search edge 3) 4 [1747] 1747 root269Tail14 winner := by
  rw [root269Tail13, scan, if_neg (by decide), root269Candidates13_checked]
  change scan edge (search edge 3) 4 [1747] 1747 root269Tail14
    (if 3 ≤ root269Candidates13.length then search edge 3 [1747, 6037] root269Candidates13 7784 winner else winner) = _
  rw [if_pos (by decide), root269Child13]
theorem root269Scan14 : scan edge (search edge 3) 4 [1747] 1747 root269Tail14 winner = winner := by rw [root269Tail14, scan, if_pos (by decide)]
theorem root269Scan13 : scan edge (search edge 3) 4 [1747] 1747 root269Tail13 winner = winner :=
  (root269Step13).trans root269Scan14
theorem root269Scan12 : scan edge (search edge 3) 4 [1747] 1747 root269Tail12 winner = winner :=
  (root269Step12).trans root269Scan13
theorem root269Scan11 : scan edge (search edge 3) 4 [1747] 1747 root269Tail11 winner = winner :=
  (root269Step11).trans root269Scan12
theorem root269Scan10 : scan edge (search edge 3) 4 [1747] 1747 root269Tail10 winner = winner :=
  (root269Step10).trans root269Scan11
theorem root269Scan9 : scan edge (search edge 3) 4 [1747] 1747 root269Tail9 winner = winner :=
  (root269Step9).trans root269Scan10
theorem root269Scan8 : scan edge (search edge 3) 4 [1747] 1747 root269Tail8 winner = winner :=
  (root269Step8).trans root269Scan9
theorem root269Scan7 : scan edge (search edge 3) 4 [1747] 1747 root269Tail7 winner = winner :=
  (root269Step7).trans root269Scan8
theorem root269Scan6 : scan edge (search edge 3) 4 [1747] 1747 root269Tail6 winner = winner :=
  (root269Step6).trans root269Scan7
theorem root269Scan5 : scan edge (search edge 3) 4 [1747] 1747 root269Tail5 winner = winner :=
  (root269Step5).trans root269Scan6
theorem root269Scan4 : scan edge (search edge 3) 4 [1747] 1747 root269Tail4 winner = winner :=
  (root269Step4).trans root269Scan5
theorem root269Scan3 : scan edge (search edge 3) 4 [1747] 1747 root269Tail3 winner = winner :=
  (root269Step3).trans root269Scan4
theorem root269Scan2 : scan edge (search edge 3) 4 [1747] 1747 root269Tail2 winner = winner :=
  (root269Step2).trans root269Scan3
theorem root269Scan1 : scan edge (search edge 3) 4 [1747] 1747 root269Tail1 winner = winner :=
  (root269Step1).trans root269Scan2
theorem root269Scan0 : scan edge (search edge 3) 4 [1747] 1747 root269Tail0 winner = winner :=
  (root269Step0).trans root269Scan1
set_option maxRecDepth 4096 in
theorem root269Branch : search edge 4 [1747] root269Candidates 1747 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1747] 1747 root269Tail0 winner = winner
  exact root269Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
