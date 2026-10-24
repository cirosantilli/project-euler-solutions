import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root310Candidates : List Nat := Data.row2081
set_option maxRecDepth 8192 in
theorem root310Candidates_checked : candidates edge 5 0 2081 tail311 winner.bound = root310Candidates := by
  apply candidates_eq_row edge 5 0 2081 tail311 root310Candidates winner.bound tail311_ordered
  · decide
  · decide
  · funext q
    rfl
def root310Tail49 : List Nat := []
def root310Tail48 : List Nat := 23291 :: root310Tail49
def root310Tail47 : List Nat := 22973 :: root310Tail48
def root310Tail46 : List Nat := 22877 :: root310Tail47
def root310Tail45 : List Nat := 22643 :: root310Tail46
def root310Tail44 : List Nat := 21713 :: root310Tail45
def root310Tail43 : List Nat := 20981 :: root310Tail44
def root310Tail42 : List Nat := 20441 :: root310Tail43
def root310Tail41 : List Nat := 20333 :: root310Tail42
def root310Tail40 : List Nat := 19991 :: root310Tail41
def root310Tail39 : List Nat := 19553 :: root310Tail40
def root310Tail38 : List Nat := 18671 :: root310Tail39
def root310Tail37 : List Nat := 18461 :: root310Tail38
def root310Tail36 : List Nat := 18341 :: root310Tail37
def root310Tail35 : List Nat := 18149 :: root310Tail36
def root310Tail34 : List Nat := 18041 :: root310Tail35
def root310Tail33 : List Nat := 17207 :: root310Tail34
def root310Tail32 : List Nat := 15329 :: root310Tail33
def root310Tail31 : List Nat := 14957 :: root310Tail32
def root310Tail30 : List Nat := 14891 :: root310Tail31
def root310Tail29 : List Nat := 13799 :: root310Tail30
def root310Tail28 : List Nat := 13523 :: root310Tail29
def root310Tail27 : List Nat := 13487 :: root310Tail28
def root310Tail26 : List Nat := 12713 :: root310Tail27
def root310Tail25 : List Nat := 12473 :: root310Tail26
def root310Tail24 : List Nat := 11633 :: root310Tail25
def root310Tail23 : List Nat := 11489 :: root310Tail24
def root310Tail22 : List Nat := 10733 :: root310Tail23
def root310Tail21 : List Nat := 10691 :: root310Tail22
def root310Tail20 : List Nat := 10289 :: root310Tail21
def root310Tail19 : List Nat := 9857 :: root310Tail20
def root310Tail18 : List Nat := 9629 :: root310Tail19
def root310Tail17 : List Nat := 9371 :: root310Tail18
def root310Tail16 : List Nat := 8747 :: root310Tail17
def root310Tail15 : List Nat := 7793 :: root310Tail16
def root310Tail14 : List Nat := 7229 :: root310Tail15
def root310Tail13 : List Nat := 7109 :: root310Tail14
def root310Tail12 : List Nat := 6791 :: root310Tail13
def root310Tail11 : List Nat := 6689 :: root310Tail12
def root310Tail10 : List Nat := 5807 :: root310Tail11
def root310Tail9 : List Nat := 5801 :: root310Tail10
def root310Tail8 : List Nat := 5657 :: root310Tail9
def root310Tail7 : List Nat := 5441 :: root310Tail8
def root310Tail6 : List Nat := 5153 :: root310Tail7
def root310Tail5 : List Nat := 4943 :: root310Tail6
def root310Tail4 : List Nat := 4691 :: root310Tail5
def root310Tail3 : List Nat := 4349 :: root310Tail4
def root310Tail2 : List Nat := 4127 :: root310Tail3
def root310Tail1 : List Nat := 3323 :: root310Tail2
def root310Tail0 : List Nat := 2699 :: root310Tail1
def root310Candidates0 : List Nat := [10691]
set_option maxRecDepth 8192 in
theorem root310Candidates0_checked : candidates edge 4 2081 2699 root310Tail1 winner.bound = root310Candidates0 := by
  unfold candidates
  have row : edge 2699 = (SortedNatTree.contains FastRows.row2699) := by
    change Data.row2699.contains = (SortedNatTree.contains FastRows.row2699)
    exact FastRows.row2699_checked.symm
  rw [row]
  decide
theorem root310Step0 : scan edge (search edge 3) 4 [2081] 2081 root310Tail0 winner =
    scan edge (search edge 3) 4 [2081] 2081 root310Tail1 winner := by
  rw [root310Tail0, scan, if_neg (by decide), root310Candidates0_checked]
  change scan edge (search edge 3) 4 [2081] 2081 root310Tail1
    (if 3 ≤ root310Candidates0.length then search edge 3 [2081, 2699] root310Candidates0 4780 winner else winner) = _
  rw [if_neg (by decide)]
def root310Candidates1 : List Nat := [4349, 7229, 10733, 12713]
set_option maxRecDepth 8192 in
theorem root310Candidates1_checked : candidates edge 4 2081 3323 root310Tail2 winner.bound = root310Candidates1 := by
  unfold candidates
  have row : edge 3323 = (SortedNatTree.contains FastRows.row3323) := by
    change Data.row3323.contains = (SortedNatTree.contains FastRows.row3323)
    exact FastRows.row3323_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root310Child1 : search edge 3 [2081, 3323] root310Candidates1 5404 winner = winner := by decide
theorem root310Step1 : scan edge (search edge 3) 4 [2081] 2081 root310Tail1 winner =
    scan edge (search edge 3) 4 [2081] 2081 root310Tail2 winner := by
  rw [root310Tail1, scan, if_neg (by decide), root310Candidates1_checked]
  change scan edge (search edge 3) 4 [2081] 2081 root310Tail2
    (if 3 ≤ root310Candidates1.length then search edge 3 [2081, 3323] root310Candidates1 5404 winner else winner) = _
  rw [if_pos (by decide), root310Child1]
def root310Candidates2 : List Nat := [14957]
set_option maxRecDepth 8192 in
theorem root310Candidates2_checked : candidates edge 4 2081 4127 root310Tail3 winner.bound = root310Candidates2 := by
  unfold candidates
  have row : edge 4127 = (SortedNatTree.contains FastRows.row4127) := by
    change Data.row4127.contains = (SortedNatTree.contains FastRows.row4127)
    exact FastRows.row4127_checked.symm
  rw [row]
  decide
theorem root310Step2 : scan edge (search edge 3) 4 [2081] 2081 root310Tail2 winner =
    scan edge (search edge 3) 4 [2081] 2081 root310Tail3 winner := by
  rw [root310Tail2, scan, if_neg (by decide), root310Candidates2_checked]
  change scan edge (search edge 3) 4 [2081] 2081 root310Tail3
    (if 3 ≤ root310Candidates2.length then search edge 3 [2081, 4127] root310Candidates2 6208 winner else winner) = _
  rw [if_neg (by decide)]
def root310Candidates3 : List Nat := [7793, 9629, 12713, 17207]
set_option maxRecDepth 8192 in
theorem root310Candidates3_checked : candidates edge 4 2081 4349 root310Tail4 winner.bound = root310Candidates3 := by
  unfold candidates
  have row : edge 4349 = (SortedNatTree.contains FastRows.row4349) := by
    change Data.row4349.contains = (SortedNatTree.contains FastRows.row4349)
    exact FastRows.row4349_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root310Child3 : search edge 3 [2081, 4349] root310Candidates3 6430 winner = winner := by decide
theorem root310Step3 : scan edge (search edge 3) 4 [2081] 2081 root310Tail3 winner =
    scan edge (search edge 3) 4 [2081] 2081 root310Tail4 winner := by
  rw [root310Tail3, scan, if_neg (by decide), root310Candidates3_checked]
  change scan edge (search edge 3) 4 [2081] 2081 root310Tail4
    (if 3 ≤ root310Candidates3.length then search edge 3 [2081, 4349] root310Candidates3 6430 winner else winner) = _
  rw [if_pos (by decide), root310Child3]
def root310Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root310Candidates4_checked : candidates edge 4 2081 4691 root310Tail5 winner.bound = root310Candidates4 := by
  unfold candidates
  have row : edge 4691 = (SortedNatTree.contains FastRows.row4691) := by
    change Data.row4691.contains = (SortedNatTree.contains FastRows.row4691)
    exact FastRows.row4691_checked.symm
  rw [row]
  decide
theorem root310Step4 : scan edge (search edge 3) 4 [2081] 2081 root310Tail4 winner =
    scan edge (search edge 3) 4 [2081] 2081 root310Tail5 winner := by
  rw [root310Tail4, scan, if_neg (by decide), root310Candidates4_checked]
  change scan edge (search edge 3) 4 [2081] 2081 root310Tail5
    (if 3 ≤ root310Candidates4.length then search edge 3 [2081, 4691] root310Candidates4 6772 winner else winner) = _
  rw [if_neg (by decide)]
def root310Candidates5 : List Nat := [15329]
set_option maxRecDepth 8192 in
theorem root310Candidates5_checked : candidates edge 4 2081 4943 root310Tail6 winner.bound = root310Candidates5 := by
  unfold candidates
  have row : edge 4943 = (SortedNatTree.contains FastRows.row4943) := by
    change Data.row4943.contains = (SortedNatTree.contains FastRows.row4943)
    exact FastRows.row4943_checked.symm
  rw [row]
  decide
theorem root310Step5 : scan edge (search edge 3) 4 [2081] 2081 root310Tail5 winner =
    scan edge (search edge 3) 4 [2081] 2081 root310Tail6 winner := by
  rw [root310Tail5, scan, if_neg (by decide), root310Candidates5_checked]
  change scan edge (search edge 3) 4 [2081] 2081 root310Tail6
    (if 3 ≤ root310Candidates5.length then search edge 3 [2081, 4943] root310Candidates5 7024 winner else winner) = _
  rw [if_neg (by decide)]
def root310Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root310Candidates6_checked : candidates edge 4 2081 5153 root310Tail7 winner.bound = root310Candidates6 := by
  unfold candidates
  have row : edge 5153 = (SortedNatTree.contains FastRows.row5153) := by
    change Data.row5153.contains = (SortedNatTree.contains FastRows.row5153)
    exact FastRows.row5153_checked.symm
  rw [row]
  decide
theorem root310Step6 : scan edge (search edge 3) 4 [2081] 2081 root310Tail6 winner =
    scan edge (search edge 3) 4 [2081] 2081 root310Tail7 winner := by
  rw [root310Tail6, scan, if_neg (by decide), root310Candidates6_checked]
  change scan edge (search edge 3) 4 [2081] 2081 root310Tail7
    (if 3 ≤ root310Candidates6.length then search edge 3 [2081, 5153] root310Candidates6 7234 winner else winner) = _
  rw [if_neg (by decide)]
def root310Candidates7 : List Nat := [13799]
set_option maxRecDepth 8192 in
theorem root310Candidates7_checked : candidates edge 4 2081 5441 root310Tail8 winner.bound = root310Candidates7 := by
  unfold candidates
  have row : edge 5441 = (SortedNatTree.contains FastRows.row5441) := by
    change Data.row5441.contains = (SortedNatTree.contains FastRows.row5441)
    exact FastRows.row5441_checked.symm
  rw [row]
  decide
theorem root310Step7 : scan edge (search edge 3) 4 [2081] 2081 root310Tail7 winner =
    scan edge (search edge 3) 4 [2081] 2081 root310Tail8 winner := by
  rw [root310Tail7, scan, if_neg (by decide), root310Candidates7_checked]
  change scan edge (search edge 3) 4 [2081] 2081 root310Tail8
    (if 3 ≤ root310Candidates7.length then search edge 3 [2081, 5441] root310Candidates7 7522 winner else winner) = _
  rw [if_neg (by decide)]
def root310Candidates8 : List Nat := [6791]
set_option maxRecDepth 8192 in
theorem root310Candidates8_checked : candidates edge 4 2081 5657 root310Tail9 winner.bound = root310Candidates8 := by
  unfold candidates
  have row : edge 5657 = (SortedNatTree.contains FastRows.row5657) := by
    change Data.row5657.contains = (SortedNatTree.contains FastRows.row5657)
    exact FastRows.row5657_checked.symm
  rw [row]
  decide
theorem root310Step8 : scan edge (search edge 3) 4 [2081] 2081 root310Tail8 winner =
    scan edge (search edge 3) 4 [2081] 2081 root310Tail9 winner := by
  rw [root310Tail8, scan, if_neg (by decide), root310Candidates8_checked]
  change scan edge (search edge 3) 4 [2081] 2081 root310Tail9
    (if 3 ≤ root310Candidates8.length then search edge 3 [2081, 5657] root310Candidates8 7738 winner else winner) = _
  rw [if_neg (by decide)]
def root310Candidates9 : List Nat := [9371]
set_option maxRecDepth 8192 in
theorem root310Candidates9_checked : candidates edge 4 2081 5801 root310Tail10 winner.bound = root310Candidates9 := by
  unfold candidates
  have row : edge 5801 = (SortedNatTree.contains FastRows.row5801) := by
    change Data.row5801.contains = (SortedNatTree.contains FastRows.row5801)
    exact FastRows.row5801_checked.symm
  rw [row]
  decide
theorem root310Step9 : scan edge (search edge 3) 4 [2081] 2081 root310Tail9 winner =
    scan edge (search edge 3) 4 [2081] 2081 root310Tail10 winner := by
  rw [root310Tail9, scan, if_neg (by decide), root310Candidates9_checked]
  change scan edge (search edge 3) 4 [2081] 2081 root310Tail10
    (if 3 ≤ root310Candidates9.length then search edge 3 [2081, 5801] root310Candidates9 7882 winner else winner) = _
  rw [if_neg (by decide)]
def root310Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root310Candidates10_checked : candidates edge 4 2081 5807 root310Tail11 winner.bound = root310Candidates10 := by
  unfold candidates
  have row : edge 5807 = (SortedNatTree.contains FastRows.row5807) := by
    change Data.row5807.contains = (SortedNatTree.contains FastRows.row5807)
    exact FastRows.row5807_checked.symm
  rw [row]
  decide
theorem root310Step10 : scan edge (search edge 3) 4 [2081] 2081 root310Tail10 winner =
    scan edge (search edge 3) 4 [2081] 2081 root310Tail11 winner := by
  rw [root310Tail10, scan, if_neg (by decide), root310Candidates10_checked]
  change scan edge (search edge 3) 4 [2081] 2081 root310Tail11
    (if 3 ≤ root310Candidates10.length then search edge 3 [2081, 5807] root310Candidates10 7888 winner else winner) = _
  rw [if_neg (by decide)]
theorem root310Scan11 : scan edge (search edge 3) 4 [2081] 2081 root310Tail11 winner = winner := by rw [root310Tail11, scan, if_pos (by decide)]
theorem root310Scan10 : scan edge (search edge 3) 4 [2081] 2081 root310Tail10 winner = winner :=
  (root310Step10).trans root310Scan11
theorem root310Scan9 : scan edge (search edge 3) 4 [2081] 2081 root310Tail9 winner = winner :=
  (root310Step9).trans root310Scan10
theorem root310Scan8 : scan edge (search edge 3) 4 [2081] 2081 root310Tail8 winner = winner :=
  (root310Step8).trans root310Scan9
theorem root310Scan7 : scan edge (search edge 3) 4 [2081] 2081 root310Tail7 winner = winner :=
  (root310Step7).trans root310Scan8
theorem root310Scan6 : scan edge (search edge 3) 4 [2081] 2081 root310Tail6 winner = winner :=
  (root310Step6).trans root310Scan7
theorem root310Scan5 : scan edge (search edge 3) 4 [2081] 2081 root310Tail5 winner = winner :=
  (root310Step5).trans root310Scan6
theorem root310Scan4 : scan edge (search edge 3) 4 [2081] 2081 root310Tail4 winner = winner :=
  (root310Step4).trans root310Scan5
theorem root310Scan3 : scan edge (search edge 3) 4 [2081] 2081 root310Tail3 winner = winner :=
  (root310Step3).trans root310Scan4
theorem root310Scan2 : scan edge (search edge 3) 4 [2081] 2081 root310Tail2 winner = winner :=
  (root310Step2).trans root310Scan3
theorem root310Scan1 : scan edge (search edge 3) 4 [2081] 2081 root310Tail1 winner = winner :=
  (root310Step1).trans root310Scan2
theorem root310Scan0 : scan edge (search edge 3) 4 [2081] 2081 root310Tail0 winner = winner :=
  (root310Step0).trans root310Scan1
set_option maxRecDepth 4096 in
theorem root310Branch : search edge 4 [2081] root310Candidates 2081 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2081] 2081 root310Tail0 winner = winner
  exact root310Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
