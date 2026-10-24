import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root275Candidates : List Nat := Data.row1789
set_option maxRecDepth 8192 in
theorem root275Candidates_checked : candidates edge 5 0 1789 tail276 winner.bound = root275Candidates := by
  apply candidates_eq_row edge 5 0 1789 tail276 root275Candidates winner.bound tail276_ordered
  · decide
  · decide
  · funext q
    rfl
def root275Tail52 : List Nat := []
def root275Tail51 : List Nat := 23761 :: root275Tail52
def root275Tail50 : List Nat := 23539 :: root275Tail51
def root275Tail49 : List Nat := 23131 :: root275Tail50
def root275Tail48 : List Nat := 20857 :: root275Tail49
def root275Tail47 : List Nat := 20479 :: root275Tail48
def root275Tail46 : List Nat := 19381 :: root275Tail47
def root275Tail45 : List Nat := 18919 :: root275Tail46
def root275Tail44 : List Nat := 18757 :: root275Tail45
def root275Tail43 : List Nat := 17971 :: root275Tail44
def root275Tail42 : List Nat := 17449 :: root275Tail43
def root275Tail41 : List Nat := 16069 :: root275Tail42
def root275Tail40 : List Nat := 15391 :: root275Tail41
def root275Tail39 : List Nat := 15277 :: root275Tail40
def root275Tail38 : List Nat := 14479 :: root275Tail39
def root275Tail37 : List Nat := 14389 :: root275Tail38
def root275Tail36 : List Nat := 13003 :: root275Tail37
def root275Tail35 : List Nat := 12763 :: root275Tail36
def root275Tail34 : List Nat := 12703 :: root275Tail35
def root275Tail33 : List Nat := 12583 :: root275Tail34
def root275Tail32 : List Nat := 12241 :: root275Tail33
def root275Tail31 : List Nat := 12211 :: root275Tail32
def root275Tail30 : List Nat := 12157 :: root275Tail31
def root275Tail29 : List Nat := 11497 :: root275Tail30
def root275Tail28 : List Nat := 11197 :: root275Tail29
def root275Tail27 : List Nat := 10789 :: root275Tail28
def root275Tail26 : List Nat := 9907 :: root275Tail27
def root275Tail25 : List Nat := 9811 :: root275Tail26
def root275Tail24 : List Nat := 9619 :: root275Tail25
def root275Tail23 : List Nat := 9439 :: root275Tail24
def root275Tail22 : List Nat := 8641 :: root275Tail23
def root275Tail21 : List Nat := 8011 :: root275Tail22
def root275Tail20 : List Nat := 7393 :: root275Tail21
def root275Tail19 : List Nat := 6661 :: root275Tail20
def root275Tail18 : List Nat := 6547 :: root275Tail19
def root275Tail17 : List Nat := 6421 :: root275Tail18
def root275Tail16 : List Nat := 6343 :: root275Tail17
def root275Tail15 : List Nat := 6133 :: root275Tail16
def root275Tail14 : List Nat := 5953 :: root275Tail15
def root275Tail13 : List Nat := 5023 :: root275Tail14
def root275Tail12 : List Nat := 4831 :: root275Tail13
def root275Tail11 : List Nat := 4813 :: root275Tail12
def root275Tail10 : List Nat := 4243 :: root275Tail11
def root275Tail9 : List Nat := 4111 :: root275Tail10
def root275Tail8 : List Nat := 3907 :: root275Tail9
def root275Tail7 : List Nat := 3853 :: root275Tail8
def root275Tail6 : List Nat := 2887 :: root275Tail7
def root275Tail5 : List Nat := 2683 :: root275Tail6
def root275Tail4 : List Nat := 2521 :: root275Tail5
def root275Tail3 : List Nat := 2503 :: root275Tail4
def root275Tail2 : List Nat := 2467 :: root275Tail3
def root275Tail1 : List Nat := 2377 :: root275Tail2
def root275Tail0 : List Nat := 2089 :: root275Tail1
def root275Candidates0 : List Nat := [15277]
set_option maxRecDepth 8192 in
theorem root275Candidates0_checked : candidates edge 4 1789 2089 root275Tail1 winner.bound = root275Candidates0 := by
  unfold candidates
  have row : edge 2089 = (SortedNatTree.contains FastRows.row2089) := by
    change Data.row2089.contains = (SortedNatTree.contains FastRows.row2089)
    exact FastRows.row2089_checked.symm
  rw [row]
  decide
theorem root275Step0 : scan edge (search edge 3) 4 [1789] 1789 root275Tail0 winner =
    scan edge (search edge 3) 4 [1789] 1789 root275Tail1 winner := by
  rw [root275Tail0, scan, if_neg (by decide), root275Candidates0_checked]
  change scan edge (search edge 3) 4 [1789] 1789 root275Tail1
    (if 3 ≤ root275Candidates0.length then search edge 3 [1789, 2089] root275Candidates0 3878 winner else winner) = _
  rw [if_neg (by decide)]
def root275Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root275Candidates1_checked : candidates edge 4 1789 2377 root275Tail2 winner.bound = root275Candidates1 := by
  unfold candidates
  have row : edge 2377 = (SortedNatTree.contains FastRows.row2377) := by
    change Data.row2377.contains = (SortedNatTree.contains FastRows.row2377)
    exact FastRows.row2377_checked.symm
  rw [row]
  decide
theorem root275Step1 : scan edge (search edge 3) 4 [1789] 1789 root275Tail1 winner =
    scan edge (search edge 3) 4 [1789] 1789 root275Tail2 winner := by
  rw [root275Tail1, scan, if_neg (by decide), root275Candidates1_checked]
  change scan edge (search edge 3) 4 [1789] 1789 root275Tail2
    (if 3 ≤ root275Candidates1.length then search edge 3 [1789, 2377] root275Candidates1 4166 winner else winner) = _
  rw [if_neg (by decide)]
def root275Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root275Candidates2_checked : candidates edge 4 1789 2467 root275Tail3 winner.bound = root275Candidates2 := by
  unfold candidates
  have row : edge 2467 = (SortedNatTree.contains FastRows.row2467) := by
    change Data.row2467.contains = (SortedNatTree.contains FastRows.row2467)
    exact FastRows.row2467_checked.symm
  rw [row]
  decide
theorem root275Step2 : scan edge (search edge 3) 4 [1789] 1789 root275Tail2 winner =
    scan edge (search edge 3) 4 [1789] 1789 root275Tail3 winner := by
  rw [root275Tail2, scan, if_neg (by decide), root275Candidates2_checked]
  change scan edge (search edge 3) 4 [1789] 1789 root275Tail3
    (if 3 ≤ root275Candidates2.length then search edge 3 [1789, 2467] root275Candidates2 4256 winner else winner) = _
  rw [if_neg (by decide)]
def root275Candidates3 : List Nat := [6421, 7393, 9439]
set_option maxRecDepth 8192 in
theorem root275Candidates3_checked : candidates edge 4 1789 2503 root275Tail4 winner.bound = root275Candidates3 := by
  unfold candidates
  have row : edge 2503 = (SortedNatTree.contains FastRows.row2503) := by
    change Data.row2503.contains = (SortedNatTree.contains FastRows.row2503)
    exact FastRows.row2503_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root275Child3 : search edge 3 [1789, 2503] root275Candidates3 4292 winner = winner := by decide
theorem root275Step3 : scan edge (search edge 3) 4 [1789] 1789 root275Tail3 winner =
    scan edge (search edge 3) 4 [1789] 1789 root275Tail4 winner := by
  rw [root275Tail3, scan, if_neg (by decide), root275Candidates3_checked]
  change scan edge (search edge 3) 4 [1789] 1789 root275Tail4
    (if 3 ≤ root275Candidates3.length then search edge 3 [1789, 2503] root275Candidates3 4292 winner else winner) = _
  rw [if_pos (by decide), root275Child3]
def root275Candidates4 : List Nat := [6343, 9907]
set_option maxRecDepth 8192 in
theorem root275Candidates4_checked : candidates edge 4 1789 2521 root275Tail5 winner.bound = root275Candidates4 := by
  unfold candidates
  have row : edge 2521 = (SortedNatTree.contains FastRows.row2521) := by
    change Data.row2521.contains = (SortedNatTree.contains FastRows.row2521)
    exact FastRows.row2521_checked.symm
  rw [row]
  decide
theorem root275Step4 : scan edge (search edge 3) 4 [1789] 1789 root275Tail4 winner =
    scan edge (search edge 3) 4 [1789] 1789 root275Tail5 winner := by
  rw [root275Tail4, scan, if_neg (by decide), root275Candidates4_checked]
  change scan edge (search edge 3) 4 [1789] 1789 root275Tail5
    (if 3 ≤ root275Candidates4.length then search edge 3 [1789, 2521] root275Candidates4 4310 winner else winner) = _
  rw [if_neg (by decide)]
def root275Candidates5 : List Nat := [5023]
set_option maxRecDepth 8192 in
theorem root275Candidates5_checked : candidates edge 4 1789 2683 root275Tail6 winner.bound = root275Candidates5 := by
  unfold candidates
  have row : edge 2683 = (SortedNatTree.contains FastRows.row2683) := by
    change Data.row2683.contains = (SortedNatTree.contains FastRows.row2683)
    exact FastRows.row2683_checked.symm
  rw [row]
  decide
theorem root275Step5 : scan edge (search edge 3) 4 [1789] 1789 root275Tail5 winner =
    scan edge (search edge 3) 4 [1789] 1789 root275Tail6 winner := by
  rw [root275Tail5, scan, if_neg (by decide), root275Candidates5_checked]
  change scan edge (search edge 3) 4 [1789] 1789 root275Tail6
    (if 3 ≤ root275Candidates5.length then search edge 3 [1789, 2683] root275Candidates5 4472 winner else winner) = _
  rw [if_neg (by decide)]
def root275Candidates6 : List Nat := [9439, 12763, 15277]
set_option maxRecDepth 8192 in
theorem root275Candidates6_checked : candidates edge 4 1789 2887 root275Tail7 winner.bound = root275Candidates6 := by
  unfold candidates
  have row : edge 2887 = (SortedNatTree.contains FastRows.row2887) := by
    change Data.row2887.contains = (SortedNatTree.contains FastRows.row2887)
    exact FastRows.row2887_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root275Child6 : search edge 3 [1789, 2887] root275Candidates6 4676 winner = winner := by decide
theorem root275Step6 : scan edge (search edge 3) 4 [1789] 1789 root275Tail6 winner =
    scan edge (search edge 3) 4 [1789] 1789 root275Tail7 winner := by
  rw [root275Tail6, scan, if_neg (by decide), root275Candidates6_checked]
  change scan edge (search edge 3) 4 [1789] 1789 root275Tail7
    (if 3 ≤ root275Candidates6.length then search edge 3 [1789, 2887] root275Candidates6 4676 winner else winner) = _
  rw [if_pos (by decide), root275Child6]
def root275Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root275Candidates7_checked : candidates edge 4 1789 3853 root275Tail8 winner.bound = root275Candidates7 := by
  unfold candidates
  have row : edge 3853 = (SortedNatTree.contains FastRows.row3853) := by
    change Data.row3853.contains = (SortedNatTree.contains FastRows.row3853)
    exact FastRows.row3853_checked.symm
  rw [row]
  decide
theorem root275Step7 : scan edge (search edge 3) 4 [1789] 1789 root275Tail7 winner =
    scan edge (search edge 3) 4 [1789] 1789 root275Tail8 winner := by
  rw [root275Tail7, scan, if_neg (by decide), root275Candidates7_checked]
  change scan edge (search edge 3) 4 [1789] 1789 root275Tail8
    (if 3 ≤ root275Candidates7.length then search edge 3 [1789, 3853] root275Candidates7 5642 winner else winner) = _
  rw [if_neg (by decide)]
def root275Candidates8 : List Nat := [9619, 13003]
set_option maxRecDepth 8192 in
theorem root275Candidates8_checked : candidates edge 4 1789 3907 root275Tail9 winner.bound = root275Candidates8 := by
  unfold candidates
  have row : edge 3907 = (SortedNatTree.contains FastRows.row3907) := by
    change Data.row3907.contains = (SortedNatTree.contains FastRows.row3907)
    exact FastRows.row3907_checked.symm
  rw [row]
  decide
theorem root275Step8 : scan edge (search edge 3) 4 [1789] 1789 root275Tail8 winner =
    scan edge (search edge 3) 4 [1789] 1789 root275Tail9 winner := by
  rw [root275Tail8, scan, if_neg (by decide), root275Candidates8_checked]
  change scan edge (search edge 3) 4 [1789] 1789 root275Tail9
    (if 3 ≤ root275Candidates8.length then search edge 3 [1789, 3907] root275Candidates8 5696 winner else winner) = _
  rw [if_neg (by decide)]
def root275Candidates9 : List Nat := [6661, 9811, 12157, 18919]
set_option maxRecDepth 8192 in
theorem root275Candidates9_checked : candidates edge 4 1789 4111 root275Tail10 winner.bound = root275Candidates9 := by
  unfold candidates
  have row : edge 4111 = (SortedNatTree.contains FastRows.row4111) := by
    change Data.row4111.contains = (SortedNatTree.contains FastRows.row4111)
    exact FastRows.row4111_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root275Child9 : search edge 3 [1789, 4111] root275Candidates9 5900 winner = winner := by decide
theorem root275Step9 : scan edge (search edge 3) 4 [1789] 1789 root275Tail9 winner =
    scan edge (search edge 3) 4 [1789] 1789 root275Tail10 winner := by
  rw [root275Tail9, scan, if_neg (by decide), root275Candidates9_checked]
  change scan edge (search edge 3) 4 [1789] 1789 root275Tail10
    (if 3 ≤ root275Candidates9.length then search edge 3 [1789, 4111] root275Candidates9 5900 winner else winner) = _
  rw [if_pos (by decide), root275Child9]
def root275Candidates10 : List Nat := [4831, 11197]
set_option maxRecDepth 8192 in
theorem root275Candidates10_checked : candidates edge 4 1789 4243 root275Tail11 winner.bound = root275Candidates10 := by
  unfold candidates
  have row : edge 4243 = (SortedNatTree.contains FastRows.row4243) := by
    change Data.row4243.contains = (SortedNatTree.contains FastRows.row4243)
    exact FastRows.row4243_checked.symm
  rw [row]
  decide
theorem root275Step10 : scan edge (search edge 3) 4 [1789] 1789 root275Tail10 winner =
    scan edge (search edge 3) 4 [1789] 1789 root275Tail11 winner := by
  rw [root275Tail10, scan, if_neg (by decide), root275Candidates10_checked]
  change scan edge (search edge 3) 4 [1789] 1789 root275Tail11
    (if 3 ≤ root275Candidates10.length then search edge 3 [1789, 4243] root275Candidates10 6032 winner else winner) = _
  rw [if_neg (by decide)]
def root275Candidates11 : List Nat := [6661, 12241, 18919]
set_option maxRecDepth 8192 in
theorem root275Candidates11_checked : candidates edge 4 1789 4813 root275Tail12 winner.bound = root275Candidates11 := by
  unfold candidates
  have row : edge 4813 = (SortedNatTree.contains FastRows.row4813) := by
    change Data.row4813.contains = (SortedNatTree.contains FastRows.row4813)
    exact FastRows.row4813_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root275Child11 : search edge 3 [1789, 4813] root275Candidates11 6602 winner = winner := by decide
theorem root275Step11 : scan edge (search edge 3) 4 [1789] 1789 root275Tail11 winner =
    scan edge (search edge 3) 4 [1789] 1789 root275Tail12 winner := by
  rw [root275Tail11, scan, if_neg (by decide), root275Candidates11_checked]
  change scan edge (search edge 3) 4 [1789] 1789 root275Tail12
    (if 3 ≤ root275Candidates11.length then search edge 3 [1789, 4813] root275Candidates11 6602 winner else winner) = _
  rw [if_pos (by decide), root275Child11]
def root275Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root275Candidates12_checked : candidates edge 4 1789 4831 root275Tail13 winner.bound = root275Candidates12 := by
  unfold candidates
  have row : edge 4831 = (SortedNatTree.contains FastRows.row4831) := by
    change Data.row4831.contains = (SortedNatTree.contains FastRows.row4831)
    exact FastRows.row4831_checked.symm
  rw [row]
  decide
theorem root275Step12 : scan edge (search edge 3) 4 [1789] 1789 root275Tail12 winner =
    scan edge (search edge 3) 4 [1789] 1789 root275Tail13 winner := by
  rw [root275Tail12, scan, if_neg (by decide), root275Candidates12_checked]
  change scan edge (search edge 3) 4 [1789] 1789 root275Tail13
    (if 3 ≤ root275Candidates12.length then search edge 3 [1789, 4831] root275Candidates12 6620 winner else winner) = _
  rw [if_neg (by decide)]
def root275Candidates13 : List Nat := [9907, 15277]
set_option maxRecDepth 8192 in
theorem root275Candidates13_checked : candidates edge 4 1789 5023 root275Tail14 winner.bound = root275Candidates13 := by
  unfold candidates
  have row : edge 5023 = (SortedNatTree.contains FastRows.row5023) := by
    change Data.row5023.contains = (SortedNatTree.contains FastRows.row5023)
    exact FastRows.row5023_checked.symm
  rw [row]
  decide
theorem root275Step13 : scan edge (search edge 3) 4 [1789] 1789 root275Tail13 winner =
    scan edge (search edge 3) 4 [1789] 1789 root275Tail14 winner := by
  rw [root275Tail13, scan, if_neg (by decide), root275Candidates13_checked]
  change scan edge (search edge 3) 4 [1789] 1789 root275Tail14
    (if 3 ≤ root275Candidates13.length then search edge 3 [1789, 5023] root275Candidates13 6812 winner else winner) = _
  rw [if_neg (by decide)]
def root275Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root275Candidates14_checked : candidates edge 4 1789 5953 root275Tail15 winner.bound = root275Candidates14 := by
  unfold candidates
  have row : edge 5953 = (SortedNatTree.contains FastRows.row5953) := by
    change Data.row5953.contains = (SortedNatTree.contains FastRows.row5953)
    exact FastRows.row5953_checked.symm
  rw [row]
  decide
theorem root275Step14 : scan edge (search edge 3) 4 [1789] 1789 root275Tail14 winner =
    scan edge (search edge 3) 4 [1789] 1789 root275Tail15 winner := by
  rw [root275Tail14, scan, if_neg (by decide), root275Candidates14_checked]
  change scan edge (search edge 3) 4 [1789] 1789 root275Tail15
    (if 3 ≤ root275Candidates14.length then search edge 3 [1789, 5953] root275Candidates14 7742 winner else winner) = _
  rw [if_neg (by decide)]
theorem root275Scan15 : scan edge (search edge 3) 4 [1789] 1789 root275Tail15 winner = winner := by rw [root275Tail15, scan, if_pos (by decide)]
theorem root275Scan14 : scan edge (search edge 3) 4 [1789] 1789 root275Tail14 winner = winner :=
  (root275Step14).trans root275Scan15
theorem root275Scan13 : scan edge (search edge 3) 4 [1789] 1789 root275Tail13 winner = winner :=
  (root275Step13).trans root275Scan14
theorem root275Scan12 : scan edge (search edge 3) 4 [1789] 1789 root275Tail12 winner = winner :=
  (root275Step12).trans root275Scan13
theorem root275Scan11 : scan edge (search edge 3) 4 [1789] 1789 root275Tail11 winner = winner :=
  (root275Step11).trans root275Scan12
theorem root275Scan10 : scan edge (search edge 3) 4 [1789] 1789 root275Tail10 winner = winner :=
  (root275Step10).trans root275Scan11
theorem root275Scan9 : scan edge (search edge 3) 4 [1789] 1789 root275Tail9 winner = winner :=
  (root275Step9).trans root275Scan10
theorem root275Scan8 : scan edge (search edge 3) 4 [1789] 1789 root275Tail8 winner = winner :=
  (root275Step8).trans root275Scan9
theorem root275Scan7 : scan edge (search edge 3) 4 [1789] 1789 root275Tail7 winner = winner :=
  (root275Step7).trans root275Scan8
theorem root275Scan6 : scan edge (search edge 3) 4 [1789] 1789 root275Tail6 winner = winner :=
  (root275Step6).trans root275Scan7
theorem root275Scan5 : scan edge (search edge 3) 4 [1789] 1789 root275Tail5 winner = winner :=
  (root275Step5).trans root275Scan6
theorem root275Scan4 : scan edge (search edge 3) 4 [1789] 1789 root275Tail4 winner = winner :=
  (root275Step4).trans root275Scan5
theorem root275Scan3 : scan edge (search edge 3) 4 [1789] 1789 root275Tail3 winner = winner :=
  (root275Step3).trans root275Scan4
theorem root275Scan2 : scan edge (search edge 3) 4 [1789] 1789 root275Tail2 winner = winner :=
  (root275Step2).trans root275Scan3
theorem root275Scan1 : scan edge (search edge 3) 4 [1789] 1789 root275Tail1 winner = winner :=
  (root275Step1).trans root275Scan2
theorem root275Scan0 : scan edge (search edge 3) 4 [1789] 1789 root275Tail0 winner = winner :=
  (root275Step0).trans root275Scan1
set_option maxRecDepth 4096 in
theorem root275Branch : search edge 4 [1789] root275Candidates 1789 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1789] 1789 root275Tail0 winner = winner
  exact root275Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
