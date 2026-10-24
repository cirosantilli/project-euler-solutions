import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root408Candidates : List Nat := Data.row2833
set_option maxRecDepth 8192 in
theorem root408Candidates_checked : candidates edge 5 0 2833 tail409 winner.bound = root408Candidates := by
  apply candidates_eq_row edge 5 0 2833 tail409 root408Candidates winner.bound tail409_ordered
  · decide
  · decide
  · funext q
    rfl
def root408Tail39 : List Nat := []
def root408Tail38 : List Nat := 22651 :: root408Tail39
def root408Tail37 : List Nat := 22549 :: root408Tail38
def root408Tail36 : List Nat := 20959 :: root408Tail37
def root408Tail35 : List Nat := 20407 :: root408Tail36
def root408Tail34 : List Nat := 20161 :: root408Tail35
def root408Tail33 : List Nat := 19867 :: root408Tail34
def root408Tail32 : List Nat := 18181 :: root408Tail33
def root408Tail31 : List Nat := 18097 :: root408Tail32
def root408Tail30 : List Nat := 17449 :: root408Tail31
def root408Tail29 : List Nat := 17389 :: root408Tail30
def root408Tail28 : List Nat := 17137 :: root408Tail29
def root408Tail27 : List Nat := 16567 :: root408Tail28
def root408Tail26 : List Nat := 16249 :: root408Tail27
def root408Tail25 : List Nat := 15787 :: root408Tail26
def root408Tail24 : List Nat := 15601 :: root408Tail25
def root408Tail23 : List Nat := 14071 :: root408Tail24
def root408Tail22 : List Nat := 13873 :: root408Tail23
def root408Tail21 : List Nat := 13291 :: root408Tail22
def root408Tail20 : List Nat := 12517 :: root408Tail21
def root408Tail19 : List Nat := 12511 :: root408Tail20
def root408Tail18 : List Nat := 11719 :: root408Tail19
def root408Tail17 : List Nat := 10177 :: root408Tail18
def root408Tail16 : List Nat := 10099 :: root408Tail17
def root408Tail15 : List Nat := 9631 :: root408Tail16
def root408Tail14 : List Nat := 9613 :: root408Tail15
def root408Tail13 : List Nat := 8581 :: root408Tail14
def root408Tail12 : List Nat := 8389 :: root408Tail13
def root408Tail11 : List Nat := 7723 :: root408Tail12
def root408Tail10 : List Nat := 7297 :: root408Tail11
def root408Tail9 : List Nat := 6361 :: root408Tail10
def root408Tail8 : List Nat := 5869 :: root408Tail9
def root408Tail7 : List Nat := 5851 :: root408Tail8
def root408Tail6 : List Nat := 5683 :: root408Tail7
def root408Tail5 : List Nat := 5113 :: root408Tail6
def root408Tail4 : List Nat := 4759 :: root408Tail5
def root408Tail3 : List Nat := 4591 :: root408Tail4
def root408Tail2 : List Nat := 4021 :: root408Tail3
def root408Tail1 : List Nat := 3229 :: root408Tail2
def root408Tail0 : List Nat := 3037 :: root408Tail1
def root408Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root408Candidates0_checked : candidates edge 4 2833 3037 root408Tail1 winner.bound = root408Candidates0 := by
  unfold candidates
  have row : edge 3037 = (SortedNatTree.contains FastRows.row3037) := by
    change Data.row3037.contains = (SortedNatTree.contains FastRows.row3037)
    exact FastRows.row3037_checked.symm
  rw [row]
  decide
theorem root408Step0 : scan edge (search edge 3) 4 [2833] 2833 root408Tail0 winner =
    scan edge (search edge 3) 4 [2833] 2833 root408Tail1 winner := by
  rw [root408Tail0, scan, if_neg (by decide), root408Candidates0_checked]
  change scan edge (search edge 3) 4 [2833] 2833 root408Tail1
    (if 3 ≤ root408Candidates0.length then search edge 3 [2833, 3037] root408Candidates0 5870 winner else winner) = _
  rw [if_neg (by decide)]
def root408Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root408Candidates1_checked : candidates edge 4 2833 3229 root408Tail2 winner.bound = root408Candidates1 := by
  unfold candidates
  have row : edge 3229 = (SortedNatTree.contains FastRows.row3229) := by
    change Data.row3229.contains = (SortedNatTree.contains FastRows.row3229)
    exact FastRows.row3229_checked.symm
  rw [row]
  decide
theorem root408Step1 : scan edge (search edge 3) 4 [2833] 2833 root408Tail1 winner =
    scan edge (search edge 3) 4 [2833] 2833 root408Tail2 winner := by
  rw [root408Tail1, scan, if_neg (by decide), root408Candidates1_checked]
  change scan edge (search edge 3) 4 [2833] 2833 root408Tail2
    (if 3 ≤ root408Candidates1.length then search edge 3 [2833, 3229] root408Candidates1 6062 winner else winner) = _
  rw [if_neg (by decide)]
def root408Candidates2 : List Nat := [18097]
set_option maxRecDepth 8192 in
theorem root408Candidates2_checked : candidates edge 4 2833 4021 root408Tail3 winner.bound = root408Candidates2 := by
  unfold candidates
  have row : edge 4021 = (SortedNatTree.contains FastRows.row4021) := by
    change Data.row4021.contains = (SortedNatTree.contains FastRows.row4021)
    exact FastRows.row4021_checked.symm
  rw [row]
  decide
theorem root408Step2 : scan edge (search edge 3) 4 [2833] 2833 root408Tail2 winner =
    scan edge (search edge 3) 4 [2833] 2833 root408Tail3 winner := by
  rw [root408Tail2, scan, if_neg (by decide), root408Candidates2_checked]
  change scan edge (search edge 3) 4 [2833] 2833 root408Tail3
    (if 3 ≤ root408Candidates2.length then search edge 3 [2833, 4021] root408Candidates2 6854 winner else winner) = _
  rw [if_neg (by decide)]
def root408Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root408Candidates3_checked : candidates edge 4 2833 4591 root408Tail4 winner.bound = root408Candidates3 := by
  unfold candidates
  have row : edge 4591 = (SortedNatTree.contains FastRows.row4591) := by
    change Data.row4591.contains = (SortedNatTree.contains FastRows.row4591)
    exact FastRows.row4591_checked.symm
  rw [row]
  decide
theorem root408Step3 : scan edge (search edge 3) 4 [2833] 2833 root408Tail3 winner =
    scan edge (search edge 3) 4 [2833] 2833 root408Tail4 winner := by
  rw [root408Tail3, scan, if_neg (by decide), root408Candidates3_checked]
  change scan edge (search edge 3) 4 [2833] 2833 root408Tail4
    (if 3 ≤ root408Candidates3.length then search edge 3 [2833, 4591] root408Candidates3 7424 winner else winner) = _
  rw [if_neg (by decide)]
def root408Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root408Candidates4_checked : candidates edge 4 2833 4759 root408Tail5 winner.bound = root408Candidates4 := by
  unfold candidates
  have row : edge 4759 = (SortedNatTree.contains FastRows.row4759) := by
    change Data.row4759.contains = (SortedNatTree.contains FastRows.row4759)
    exact FastRows.row4759_checked.symm
  rw [row]
  decide
theorem root408Step4 : scan edge (search edge 3) 4 [2833] 2833 root408Tail4 winner =
    scan edge (search edge 3) 4 [2833] 2833 root408Tail5 winner := by
  rw [root408Tail4, scan, if_neg (by decide), root408Candidates4_checked]
  change scan edge (search edge 3) 4 [2833] 2833 root408Tail5
    (if 3 ≤ root408Candidates4.length then search edge 3 [2833, 4759] root408Candidates4 7592 winner else winner) = _
  rw [if_neg (by decide)]
def root408Candidates5 : List Nat := [15601]
set_option maxRecDepth 8192 in
theorem root408Candidates5_checked : candidates edge 4 2833 5113 root408Tail6 winner.bound = root408Candidates5 := by
  unfold candidates
  have row : edge 5113 = (SortedNatTree.contains FastRows.row5113) := by
    change Data.row5113.contains = (SortedNatTree.contains FastRows.row5113)
    exact FastRows.row5113_checked.symm
  rw [row]
  decide
theorem root408Step5 : scan edge (search edge 3) 4 [2833] 2833 root408Tail5 winner =
    scan edge (search edge 3) 4 [2833] 2833 root408Tail6 winner := by
  rw [root408Tail5, scan, if_neg (by decide), root408Candidates5_checked]
  change scan edge (search edge 3) 4 [2833] 2833 root408Tail6
    (if 3 ≤ root408Candidates5.length then search edge 3 [2833, 5113] root408Candidates5 7946 winner else winner) = _
  rw [if_neg (by decide)]
def root408Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root408Candidates6_checked : candidates edge 4 2833 5683 root408Tail7 winner.bound = root408Candidates6 := by
  unfold candidates
  have row : edge 5683 = (SortedNatTree.contains FastRows.row5683) := by
    change Data.row5683.contains = (SortedNatTree.contains FastRows.row5683)
    exact FastRows.row5683_checked.symm
  rw [row]
  decide
theorem root408Step6 : scan edge (search edge 3) 4 [2833] 2833 root408Tail6 winner =
    scan edge (search edge 3) 4 [2833] 2833 root408Tail7 winner := by
  rw [root408Tail6, scan, if_neg (by decide), root408Candidates6_checked]
  change scan edge (search edge 3) 4 [2833] 2833 root408Tail7
    (if 3 ≤ root408Candidates6.length then search edge 3 [2833, 5683] root408Candidates6 8516 winner else winner) = _
  rw [if_neg (by decide)]
theorem root408Scan7 : scan edge (search edge 3) 4 [2833] 2833 root408Tail7 winner = winner := by rw [root408Tail7, scan, if_pos (by decide)]
theorem root408Scan6 : scan edge (search edge 3) 4 [2833] 2833 root408Tail6 winner = winner :=
  (root408Step6).trans root408Scan7
theorem root408Scan5 : scan edge (search edge 3) 4 [2833] 2833 root408Tail5 winner = winner :=
  (root408Step5).trans root408Scan6
theorem root408Scan4 : scan edge (search edge 3) 4 [2833] 2833 root408Tail4 winner = winner :=
  (root408Step4).trans root408Scan5
theorem root408Scan3 : scan edge (search edge 3) 4 [2833] 2833 root408Tail3 winner = winner :=
  (root408Step3).trans root408Scan4
theorem root408Scan2 : scan edge (search edge 3) 4 [2833] 2833 root408Tail2 winner = winner :=
  (root408Step2).trans root408Scan3
theorem root408Scan1 : scan edge (search edge 3) 4 [2833] 2833 root408Tail1 winner = winner :=
  (root408Step1).trans root408Scan2
theorem root408Scan0 : scan edge (search edge 3) 4 [2833] 2833 root408Tail0 winner = winner :=
  (root408Step0).trans root408Scan1
set_option maxRecDepth 4096 in
theorem root408Branch : search edge 4 [2833] root408Candidates 2833 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2833] 2833 root408Tail0 winner = winner
  exact root408Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
