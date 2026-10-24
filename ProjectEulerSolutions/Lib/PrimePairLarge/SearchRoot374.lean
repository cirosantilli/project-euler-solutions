import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root374Candidates : List Nat := Data.row2591
set_option maxRecDepth 8192 in
theorem root374Candidates_checked : candidates edge 5 0 2591 tail375 winner.bound = root374Candidates := by
  apply candidates_eq_row edge 5 0 2591 tail375 root374Candidates winner.bound tail375_ordered
  · decide
  · decide
  · funext q
    rfl
def root374Tail44 : List Nat := []
def root374Tail43 : List Nat := 23297 :: root374Tail44
def root374Tail42 : List Nat := 22613 :: root374Tail43
def root374Tail41 : List Nat := 22541 :: root374Tail42
def root374Tail40 : List Nat := 21977 :: root374Tail41
def root374Tail39 : List Nat := 21911 :: root374Tail40
def root374Tail38 : List Nat := 21893 :: root374Tail39
def root374Tail37 : List Nat := 21683 :: root374Tail38
def root374Tail36 : List Nat := 21179 :: root374Tail37
def root374Tail35 : List Nat := 20759 :: root374Tail36
def root374Tail34 : List Nat := 19571 :: root374Tail35
def root374Tail33 : List Nat := 19433 :: root374Tail34
def root374Tail32 : List Nat := 17597 :: root374Tail33
def root374Tail31 : List Nat := 16829 :: root374Tail32
def root374Tail30 : List Nat := 16223 :: root374Tail31
def root374Tail29 : List Nat := 16061 :: root374Tail30
def root374Tail28 : List Nat := 15959 :: root374Tail29
def root374Tail27 : List Nat := 15359 :: root374Tail28
def root374Tail26 : List Nat := 14249 :: root374Tail27
def root374Tail25 : List Nat := 13457 :: root374Tail26
def root374Tail24 : List Nat := 13121 :: root374Tail25
def root374Tail23 : List Nat := 13037 :: root374Tail24
def root374Tail22 : List Nat := 11939 :: root374Tail23
def root374Tail21 : List Nat := 11927 :: root374Tail22
def root374Tail20 : List Nat := 11519 :: root374Tail21
def root374Tail19 : List Nat := 10151 :: root374Tail20
def root374Tail18 : List Nat := 9941 :: root374Tail19
def root374Tail17 : List Nat := 9767 :: root374Tail18
def root374Tail16 : List Nat := 8423 :: root374Tail17
def root374Tail15 : List Nat := 8231 :: root374Tail16
def root374Tail14 : List Nat := 7937 :: root374Tail15
def root374Tail13 : List Nat := 7499 :: root374Tail14
def root374Tail12 : List Nat := 7193 :: root374Tail13
def root374Tail11 : List Nat := 6299 :: root374Tail12
def root374Tail10 : List Nat := 6197 :: root374Tail11
def root374Tail9 : List Nat := 5849 :: root374Tail10
def root374Tail8 : List Nat := 5807 :: root374Tail9
def root374Tail7 : List Nat := 5741 :: root374Tail8
def root374Tail6 : List Nat := 4937 :: root374Tail7
def root374Tail5 : List Nat := 4283 :: root374Tail6
def root374Tail4 : List Nat := 4211 :: root374Tail5
def root374Tail3 : List Nat := 4073 :: root374Tail4
def root374Tail2 : List Nat := 3527 :: root374Tail3
def root374Tail1 : List Nat := 3257 :: root374Tail2
def root374Tail0 : List Nat := 2633 :: root374Tail1
def root374Candidates0 : List Nat := [5807, 7193, 15959, 16223]
set_option maxRecDepth 8192 in
theorem root374Candidates0_checked : candidates edge 4 2591 2633 root374Tail1 winner.bound = root374Candidates0 := by
  unfold candidates
  have row : edge 2633 = (SortedNatTree.contains FastRows.row2633) := by
    change Data.row2633.contains = (SortedNatTree.contains FastRows.row2633)
    exact FastRows.row2633_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root374Child0 : search edge 3 [2591, 2633] root374Candidates0 5224 winner = winner := by decide
theorem root374Step0 : scan edge (search edge 3) 4 [2591] 2591 root374Tail0 winner =
    scan edge (search edge 3) 4 [2591] 2591 root374Tail1 winner := by
  rw [root374Tail0, scan, if_neg (by decide), root374Candidates0_checked]
  change scan edge (search edge 3) 4 [2591] 2591 root374Tail1
    (if 3 ≤ root374Candidates0.length then search edge 3 [2591, 2633] root374Candidates0 5224 winner else winner) = _
  rw [if_pos (by decide), root374Child0]
def root374Candidates1 : List Nat := [4937]
set_option maxRecDepth 8192 in
theorem root374Candidates1_checked : candidates edge 4 2591 3257 root374Tail2 winner.bound = root374Candidates1 := by
  unfold candidates
  have row : edge 3257 = (SortedNatTree.contains FastRows.row3257) := by
    change Data.row3257.contains = (SortedNatTree.contains FastRows.row3257)
    exact FastRows.row3257_checked.symm
  rw [row]
  decide
theorem root374Step1 : scan edge (search edge 3) 4 [2591] 2591 root374Tail1 winner =
    scan edge (search edge 3) 4 [2591] 2591 root374Tail2 winner := by
  rw [root374Tail1, scan, if_neg (by decide), root374Candidates1_checked]
  change scan edge (search edge 3) 4 [2591] 2591 root374Tail2
    (if 3 ≤ root374Candidates1.length then search edge 3 [2591, 3257] root374Candidates1 5848 winner else winner) = _
  rw [if_neg (by decide)]
def root374Candidates2 : List Nat := [11519, 19571]
set_option maxRecDepth 8192 in
theorem root374Candidates2_checked : candidates edge 4 2591 3527 root374Tail3 winner.bound = root374Candidates2 := by
  unfold candidates
  have row : edge 3527 = (SortedNatTree.contains FastRows.row3527) := by
    change Data.row3527.contains = (SortedNatTree.contains FastRows.row3527)
    exact FastRows.row3527_checked.symm
  rw [row]
  decide
theorem root374Step2 : scan edge (search edge 3) 4 [2591] 2591 root374Tail2 winner =
    scan edge (search edge 3) 4 [2591] 2591 root374Tail3 winner := by
  rw [root374Tail2, scan, if_neg (by decide), root374Candidates2_checked]
  change scan edge (search edge 3) 4 [2591] 2591 root374Tail3
    (if 3 ≤ root374Candidates2.length then search edge 3 [2591, 3527] root374Candidates2 6118 winner else winner) = _
  rw [if_neg (by decide)]
def root374Candidates3 : List Nat := [13037, 16829]
set_option maxRecDepth 8192 in
theorem root374Candidates3_checked : candidates edge 4 2591 4073 root374Tail4 winner.bound = root374Candidates3 := by
  unfold candidates
  have row : edge 4073 = (SortedNatTree.contains FastRows.row4073) := by
    change Data.row4073.contains = (SortedNatTree.contains FastRows.row4073)
    exact FastRows.row4073_checked.symm
  rw [row]
  decide
theorem root374Step3 : scan edge (search edge 3) 4 [2591] 2591 root374Tail3 winner =
    scan edge (search edge 3) 4 [2591] 2591 root374Tail4 winner := by
  rw [root374Tail3, scan, if_neg (by decide), root374Candidates3_checked]
  change scan edge (search edge 3) 4 [2591] 2591 root374Tail4
    (if 3 ≤ root374Candidates3.length then search edge 3 [2591, 4073] root374Candidates3 6664 winner else winner) = _
  rw [if_neg (by decide)]
def root374Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root374Candidates4_checked : candidates edge 4 2591 4211 root374Tail5 winner.bound = root374Candidates4 := by
  unfold candidates
  have row : edge 4211 = (SortedNatTree.contains FastRows.row4211) := by
    change Data.row4211.contains = (SortedNatTree.contains FastRows.row4211)
    exact FastRows.row4211_checked.symm
  rw [row]
  decide
theorem root374Step4 : scan edge (search edge 3) 4 [2591] 2591 root374Tail4 winner =
    scan edge (search edge 3) 4 [2591] 2591 root374Tail5 winner := by
  rw [root374Tail4, scan, if_neg (by decide), root374Candidates4_checked]
  change scan edge (search edge 3) 4 [2591] 2591 root374Tail5
    (if 3 ≤ root374Candidates4.length then search edge 3 [2591, 4211] root374Candidates4 6802 winner else winner) = _
  rw [if_neg (by decide)]
def root374Candidates5 : List Nat := [15359]
set_option maxRecDepth 8192 in
theorem root374Candidates5_checked : candidates edge 4 2591 4283 root374Tail6 winner.bound = root374Candidates5 := by
  unfold candidates
  have row : edge 4283 = (SortedNatTree.contains FastRows.row4283) := by
    change Data.row4283.contains = (SortedNatTree.contains FastRows.row4283)
    exact FastRows.row4283_checked.symm
  rw [row]
  decide
theorem root374Step5 : scan edge (search edge 3) 4 [2591] 2591 root374Tail5 winner =
    scan edge (search edge 3) 4 [2591] 2591 root374Tail6 winner := by
  rw [root374Tail5, scan, if_neg (by decide), root374Candidates5_checked]
  change scan edge (search edge 3) 4 [2591] 2591 root374Tail6
    (if 3 ≤ root374Candidates5.length then search edge 3 [2591, 4283] root374Candidates5 6874 winner else winner) = _
  rw [if_neg (by decide)]
def root374Candidates6 : List Nat := [11927, 17597]
set_option maxRecDepth 8192 in
theorem root374Candidates6_checked : candidates edge 4 2591 4937 root374Tail7 winner.bound = root374Candidates6 := by
  unfold candidates
  have row : edge 4937 = (SortedNatTree.contains FastRows.row4937) := by
    change Data.row4937.contains = (SortedNatTree.contains FastRows.row4937)
    exact FastRows.row4937_checked.symm
  rw [row]
  decide
theorem root374Step6 : scan edge (search edge 3) 4 [2591] 2591 root374Tail6 winner =
    scan edge (search edge 3) 4 [2591] 2591 root374Tail7 winner := by
  rw [root374Tail6, scan, if_neg (by decide), root374Candidates6_checked]
  change scan edge (search edge 3) 4 [2591] 2591 root374Tail7
    (if 3 ≤ root374Candidates6.length then search edge 3 [2591, 4937] root374Candidates6 7528 winner else winner) = _
  rw [if_neg (by decide)]
def root374Candidates7 : List Nat := [6299, 11519]
set_option maxRecDepth 8192 in
theorem root374Candidates7_checked : candidates edge 4 2591 5741 root374Tail8 winner.bound = root374Candidates7 := by
  unfold candidates
  have row : edge 5741 = (SortedNatTree.contains FastRows.row5741) := by
    change Data.row5741.contains = (SortedNatTree.contains FastRows.row5741)
    exact FastRows.row5741_checked.symm
  rw [row]
  decide
theorem root374Step7 : scan edge (search edge 3) 4 [2591] 2591 root374Tail7 winner =
    scan edge (search edge 3) 4 [2591] 2591 root374Tail8 winner := by
  rw [root374Tail7, scan, if_neg (by decide), root374Candidates7_checked]
  change scan edge (search edge 3) 4 [2591] 2591 root374Tail8
    (if 3 ≤ root374Candidates7.length then search edge 3 [2591, 5741] root374Candidates7 8332 winner else winner) = _
  rw [if_neg (by decide)]
def root374Candidates8 : List Nat := [7499]
set_option maxRecDepth 8192 in
theorem root374Candidates8_checked : candidates edge 4 2591 5807 root374Tail9 winner.bound = root374Candidates8 := by
  unfold candidates
  have row : edge 5807 = (SortedNatTree.contains FastRows.row5807) := by
    change Data.row5807.contains = (SortedNatTree.contains FastRows.row5807)
    exact FastRows.row5807_checked.symm
  rw [row]
  decide
theorem root374Step8 : scan edge (search edge 3) 4 [2591] 2591 root374Tail8 winner =
    scan edge (search edge 3) 4 [2591] 2591 root374Tail9 winner := by
  rw [root374Tail8, scan, if_neg (by decide), root374Candidates8_checked]
  change scan edge (search edge 3) 4 [2591] 2591 root374Tail9
    (if 3 ≤ root374Candidates8.length then search edge 3 [2591, 5807] root374Candidates8 8398 winner else winner) = _
  rw [if_neg (by decide)]
def root374Candidates9 : List Nat := [13457]
set_option maxRecDepth 8192 in
theorem root374Candidates9_checked : candidates edge 4 2591 5849 root374Tail10 winner.bound = root374Candidates9 := by
  unfold candidates
  have row : edge 5849 = (SortedNatTree.contains FastRows.row5849) := by
    change Data.row5849.contains = (SortedNatTree.contains FastRows.row5849)
    exact FastRows.row5849_checked.symm
  rw [row]
  decide
theorem root374Step9 : scan edge (search edge 3) 4 [2591] 2591 root374Tail9 winner =
    scan edge (search edge 3) 4 [2591] 2591 root374Tail10 winner := by
  rw [root374Tail9, scan, if_neg (by decide), root374Candidates9_checked]
  change scan edge (search edge 3) 4 [2591] 2591 root374Tail10
    (if 3 ≤ root374Candidates9.length then search edge 3 [2591, 5849] root374Candidates9 8440 winner else winner) = _
  rw [if_neg (by decide)]
theorem root374Scan10 : scan edge (search edge 3) 4 [2591] 2591 root374Tail10 winner = winner := by rw [root374Tail10, scan, if_pos (by decide)]
theorem root374Scan9 : scan edge (search edge 3) 4 [2591] 2591 root374Tail9 winner = winner :=
  (root374Step9).trans root374Scan10
theorem root374Scan8 : scan edge (search edge 3) 4 [2591] 2591 root374Tail8 winner = winner :=
  (root374Step8).trans root374Scan9
theorem root374Scan7 : scan edge (search edge 3) 4 [2591] 2591 root374Tail7 winner = winner :=
  (root374Step7).trans root374Scan8
theorem root374Scan6 : scan edge (search edge 3) 4 [2591] 2591 root374Tail6 winner = winner :=
  (root374Step6).trans root374Scan7
theorem root374Scan5 : scan edge (search edge 3) 4 [2591] 2591 root374Tail5 winner = winner :=
  (root374Step5).trans root374Scan6
theorem root374Scan4 : scan edge (search edge 3) 4 [2591] 2591 root374Tail4 winner = winner :=
  (root374Step4).trans root374Scan5
theorem root374Scan3 : scan edge (search edge 3) 4 [2591] 2591 root374Tail3 winner = winner :=
  (root374Step3).trans root374Scan4
theorem root374Scan2 : scan edge (search edge 3) 4 [2591] 2591 root374Tail2 winner = winner :=
  (root374Step2).trans root374Scan3
theorem root374Scan1 : scan edge (search edge 3) 4 [2591] 2591 root374Tail1 winner = winner :=
  (root374Step1).trans root374Scan2
theorem root374Scan0 : scan edge (search edge 3) 4 [2591] 2591 root374Tail0 winner = winner :=
  (root374Step0).trans root374Scan1
set_option maxRecDepth 4096 in
theorem root374Branch : search edge 4 [2591] root374Candidates 2591 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2591] 2591 root374Tail0 winner = winner
  exact root374Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
