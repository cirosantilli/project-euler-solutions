import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root474Candidates : List Nat := Data.row3389
set_option maxRecDepth 8192 in
theorem root474Candidates_checked : candidates edge 5 0 3389 tail475 winner.bound = root474Candidates := by
  apply candidates_eq_row edge 5 0 3389 tail475 root474Candidates winner.bound tail475_ordered
  · decide
  · decide
  · funext q
    rfl
def root474Tail34 : List Nat := []
def root474Tail33 : List Nat := 21773 :: root474Tail34
def root474Tail32 : List Nat := 20759 :: root474Tail33
def root474Tail31 : List Nat := 20051 :: root474Tail32
def root474Tail30 : List Nat := 19709 :: root474Tail31
def root474Tail29 : List Nat := 18803 :: root474Tail30
def root474Tail28 : List Nat := 18521 :: root474Tail29
def root474Tail27 : List Nat := 18329 :: root474Tail28
def root474Tail26 : List Nat := 17609 :: root474Tail27
def root474Tail25 : List Nat := 17123 :: root474Tail26
def root474Tail24 : List Nat := 16943 :: root474Tail25
def root474Tail23 : List Nat := 16901 :: root474Tail24
def root474Tail22 : List Nat := 16349 :: root474Tail23
def root474Tail21 : List Nat := 15263 :: root474Tail22
def root474Tail20 : List Nat := 15173 :: root474Tail21
def root474Tail19 : List Nat := 14549 :: root474Tail20
def root474Tail18 : List Nat := 13931 :: root474Tail19
def root474Tail17 : List Nat := 13313 :: root474Tail18
def root474Tail16 : List Nat := 12653 :: root474Tail17
def root474Tail15 : List Nat := 12251 :: root474Tail16
def root474Tail14 : List Nat := 11789 :: root474Tail15
def root474Tail13 : List Nat := 11369 :: root474Tail14
def root474Tail12 : List Nat := 10601 :: root474Tail13
def root474Tail11 : List Nat := 7937 :: root474Tail12
def root474Tail10 : List Nat := 7841 :: root474Tail11
def root474Tail9 : List Nat := 7727 :: root474Tail10
def root474Tail8 : List Nat := 7547 :: root474Tail9
def root474Tail7 : List Nat := 6449 :: root474Tail8
def root474Tail6 : List Nat := 5879 :: root474Tail7
def root474Tail5 : List Nat := 5387 :: root474Tail6
def root474Tail4 : List Nat := 5153 :: root474Tail5
def root474Tail3 : List Nat := 4799 :: root474Tail4
def root474Tail2 : List Nat := 4409 :: root474Tail3
def root474Tail1 : List Nat := 4139 :: root474Tail2
def root474Tail0 : List Nat := 3803 :: root474Tail1
def root474Candidates0 : List Nat := [13931]
set_option maxRecDepth 8192 in
theorem root474Candidates0_checked : candidates edge 4 3389 3803 root474Tail1 winner.bound = root474Candidates0 := by
  unfold candidates
  have row : edge 3803 = (SortedNatTree.contains FastRows.row3803) := by
    change Data.row3803.contains = (SortedNatTree.contains FastRows.row3803)
    exact FastRows.row3803_checked.symm
  rw [row]
  decide
theorem root474Step0 : scan edge (search edge 3) 4 [3389] 3389 root474Tail0 winner =
    scan edge (search edge 3) 4 [3389] 3389 root474Tail1 winner := by
  rw [root474Tail0, scan, if_neg (by decide), root474Candidates0_checked]
  change scan edge (search edge 3) 4 [3389] 3389 root474Tail1
    (if 3 ≤ root474Candidates0.length then search edge 3 [3389, 3803] root474Candidates0 7192 winner else winner) = _
  rw [if_neg (by decide)]
def root474Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root474Candidates1_checked : candidates edge 4 3389 4139 root474Tail2 winner.bound = root474Candidates1 := by
  unfold candidates
  have row : edge 4139 = (SortedNatTree.contains FastRows.row4139) := by
    change Data.row4139.contains = (SortedNatTree.contains FastRows.row4139)
    exact FastRows.row4139_checked.symm
  rw [row]
  decide
theorem root474Step1 : scan edge (search edge 3) 4 [3389] 3389 root474Tail1 winner =
    scan edge (search edge 3) 4 [3389] 3389 root474Tail2 winner := by
  rw [root474Tail1, scan, if_neg (by decide), root474Candidates1_checked]
  change scan edge (search edge 3) 4 [3389] 3389 root474Tail2
    (if 3 ≤ root474Candidates1.length then search edge 3 [3389, 4139] root474Candidates1 7528 winner else winner) = _
  rw [if_neg (by decide)]
def root474Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root474Candidates2_checked : candidates edge 4 3389 4409 root474Tail3 winner.bound = root474Candidates2 := by
  unfold candidates
  have row : edge 4409 = (SortedNatTree.contains FastRows.row4409) := by
    change Data.row4409.contains = (SortedNatTree.contains FastRows.row4409)
    exact FastRows.row4409_checked.symm
  rw [row]
  decide
theorem root474Step2 : scan edge (search edge 3) 4 [3389] 3389 root474Tail2 winner =
    scan edge (search edge 3) 4 [3389] 3389 root474Tail3 winner := by
  rw [root474Tail2, scan, if_neg (by decide), root474Candidates2_checked]
  change scan edge (search edge 3) 4 [3389] 3389 root474Tail3
    (if 3 ≤ root474Candidates2.length then search edge 3 [3389, 4409] root474Candidates2 7798 winner else winner) = _
  rw [if_neg (by decide)]
def root474Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root474Candidates3_checked : candidates edge 4 3389 4799 root474Tail4 winner.bound = root474Candidates3 := by
  unfold candidates
  have row : edge 4799 = (SortedNatTree.contains FastRows.row4799) := by
    change Data.row4799.contains = (SortedNatTree.contains FastRows.row4799)
    exact FastRows.row4799_checked.symm
  rw [row]
  decide
theorem root474Step3 : scan edge (search edge 3) 4 [3389] 3389 root474Tail3 winner =
    scan edge (search edge 3) 4 [3389] 3389 root474Tail4 winner := by
  rw [root474Tail3, scan, if_neg (by decide), root474Candidates3_checked]
  change scan edge (search edge 3) 4 [3389] 3389 root474Tail4
    (if 3 ≤ root474Candidates3.length then search edge 3 [3389, 4799] root474Candidates3 8188 winner else winner) = _
  rw [if_neg (by decide)]
def root474Candidates4 : List Nat := [12251]
set_option maxRecDepth 8192 in
theorem root474Candidates4_checked : candidates edge 4 3389 5153 root474Tail5 winner.bound = root474Candidates4 := by
  unfold candidates
  have row : edge 5153 = (SortedNatTree.contains FastRows.row5153) := by
    change Data.row5153.contains = (SortedNatTree.contains FastRows.row5153)
    exact FastRows.row5153_checked.symm
  rw [row]
  decide
theorem root474Step4 : scan edge (search edge 3) 4 [3389] 3389 root474Tail4 winner =
    scan edge (search edge 3) 4 [3389] 3389 root474Tail5 winner := by
  rw [root474Tail4, scan, if_neg (by decide), root474Candidates4_checked]
  change scan edge (search edge 3) 4 [3389] 3389 root474Tail5
    (if 3 ≤ root474Candidates4.length then search edge 3 [3389, 5153] root474Candidates4 8542 winner else winner) = _
  rw [if_neg (by decide)]
def root474Candidates5 : List Nat := [16901]
set_option maxRecDepth 8192 in
theorem root474Candidates5_checked : candidates edge 4 3389 5387 root474Tail6 winner.bound = root474Candidates5 := by
  unfold candidates
  have row : edge 5387 = (SortedNatTree.contains FastRows.row5387) := by
    change Data.row5387.contains = (SortedNatTree.contains FastRows.row5387)
    exact FastRows.row5387_checked.symm
  rw [row]
  decide
theorem root474Step5 : scan edge (search edge 3) 4 [3389] 3389 root474Tail5 winner =
    scan edge (search edge 3) 4 [3389] 3389 root474Tail6 winner := by
  rw [root474Tail5, scan, if_neg (by decide), root474Candidates5_checked]
  change scan edge (search edge 3) 4 [3389] 3389 root474Tail6
    (if 3 ≤ root474Candidates5.length then search edge 3 [3389, 5387] root474Candidates5 8776 winner else winner) = _
  rw [if_neg (by decide)]
theorem root474Scan6 : scan edge (search edge 3) 4 [3389] 3389 root474Tail6 winner = winner := by rw [root474Tail6, scan, if_pos (by decide)]
theorem root474Scan5 : scan edge (search edge 3) 4 [3389] 3389 root474Tail5 winner = winner :=
  (root474Step5).trans root474Scan6
theorem root474Scan4 : scan edge (search edge 3) 4 [3389] 3389 root474Tail4 winner = winner :=
  (root474Step4).trans root474Scan5
theorem root474Scan3 : scan edge (search edge 3) 4 [3389] 3389 root474Tail3 winner = winner :=
  (root474Step3).trans root474Scan4
theorem root474Scan2 : scan edge (search edge 3) 4 [3389] 3389 root474Tail2 winner = winner :=
  (root474Step2).trans root474Scan3
theorem root474Scan1 : scan edge (search edge 3) 4 [3389] 3389 root474Tail1 winner = winner :=
  (root474Step1).trans root474Scan2
theorem root474Scan0 : scan edge (search edge 3) 4 [3389] 3389 root474Tail0 winner = winner :=
  (root474Step0).trans root474Scan1
set_option maxRecDepth 4096 in
theorem root474Branch : search edge 4 [3389] root474Candidates 3389 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3389] 3389 root474Tail0 winner = winner
  exact root474Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
