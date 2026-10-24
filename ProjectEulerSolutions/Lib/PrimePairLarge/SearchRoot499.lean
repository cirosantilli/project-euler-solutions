import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root499Candidates : List Nat := Data.row3583
set_option maxRecDepth 8192 in
theorem root499Candidates_checked : candidates edge 5 0 3583 tail500 winner.bound = root499Candidates := by
  apply candidates_eq_row edge 5 0 3583 tail500 root499Candidates winner.bound tail500_ordered
  · decide
  · decide
  · funext q
    rfl
def root499Tail40 : List Nat := []
def root499Tail39 : List Nat := 22147 :: root499Tail40
def root499Tail38 : List Nat := 22051 :: root499Tail39
def root499Tail37 : List Nat := 21661 :: root499Tail38
def root499Tail36 : List Nat := 20731 :: root499Tail37
def root499Tail35 : List Nat := 19237 :: root499Tail36
def root499Tail34 : List Nat := 19141 :: root499Tail35
def root499Tail33 : List Nat := 18757 :: root499Tail34
def root499Tail32 : List Nat := 16741 :: root499Tail33
def root499Tail31 : List Nat := 16267 :: root499Tail32
def root499Tail30 : List Nat := 15649 :: root499Tail31
def root499Tail29 : List Nat := 15361 :: root499Tail30
def root499Tail28 : List Nat := 14851 :: root499Tail29
def root499Tail27 : List Nat := 14713 :: root499Tail28
def root499Tail26 : List Nat := 13933 :: root499Tail27
def root499Tail25 : List Nat := 13831 :: root499Tail26
def root499Tail24 : List Nat := 13693 :: root499Tail25
def root499Tail23 : List Nat := 12979 :: root499Tail24
def root499Tail22 : List Nat := 12613 :: root499Tail23
def root499Tail21 : List Nat := 12517 :: root499Tail22
def root499Tail20 : List Nat := 12457 :: root499Tail21
def root499Tail19 : List Nat := 11887 :: root499Tail20
def root499Tail18 : List Nat := 10837 :: root499Tail19
def root499Tail17 : List Nat := 10723 :: root499Tail18
def root499Tail16 : List Nat := 9787 :: root499Tail17
def root499Tail15 : List Nat := 9631 :: root499Tail16
def root499Tail14 : List Nat := 8779 :: root499Tail15
def root499Tail13 : List Nat := 8527 :: root499Tail14
def root499Tail12 : List Nat := 8191 :: root499Tail13
def root499Tail11 : List Nat := 7951 :: root499Tail12
def root499Tail10 : List Nat := 7759 :: root499Tail11
def root499Tail9 : List Nat := 7057 :: root499Tail10
def root499Tail8 : List Nat := 6247 :: root499Tail9
def root499Tail7 : List Nat := 6067 :: root499Tail8
def root499Tail6 : List Nat := 5953 :: root499Tail7
def root499Tail5 : List Nat := 5857 :: root499Tail6
def root499Tail4 : List Nat := 5689 :: root499Tail5
def root499Tail3 : List Nat := 5179 :: root499Tail4
def root499Tail2 : List Nat := 4651 :: root499Tail3
def root499Tail1 : List Nat := 4423 :: root499Tail2
def root499Tail0 : List Nat := 4177 :: root499Tail1
def root499Candidates0 : List Nat := [13693]
set_option maxRecDepth 8192 in
theorem root499Candidates0_checked : candidates edge 4 3583 4177 root499Tail1 winner.bound = root499Candidates0 := by
  unfold candidates
  have row : edge 4177 = (SortedNatTree.contains FastRows.row4177) := by
    change Data.row4177.contains = (SortedNatTree.contains FastRows.row4177)
    exact FastRows.row4177_checked.symm
  rw [row]
  decide
theorem root499Step0 : scan edge (search edge 3) 4 [3583] 3583 root499Tail0 winner =
    scan edge (search edge 3) 4 [3583] 3583 root499Tail1 winner := by
  rw [root499Tail0, scan, if_neg (by decide), root499Candidates0_checked]
  change scan edge (search edge 3) 4 [3583] 3583 root499Tail1
    (if 3 ≤ root499Candidates0.length then search edge 3 [3583, 4177] root499Candidates0 7760 winner else winner) = _
  rw [if_neg (by decide)]
def root499Candidates1 : List Nat := [5689, 13693]
set_option maxRecDepth 8192 in
theorem root499Candidates1_checked : candidates edge 4 3583 4423 root499Tail2 winner.bound = root499Candidates1 := by
  unfold candidates
  have row : edge 4423 = (SortedNatTree.contains FastRows.row4423) := by
    change Data.row4423.contains = (SortedNatTree.contains FastRows.row4423)
    exact FastRows.row4423_checked.symm
  rw [row]
  decide
theorem root499Step1 : scan edge (search edge 3) 4 [3583] 3583 root499Tail1 winner =
    scan edge (search edge 3) 4 [3583] 3583 root499Tail2 winner := by
  rw [root499Tail1, scan, if_neg (by decide), root499Candidates1_checked]
  change scan edge (search edge 3) 4 [3583] 3583 root499Tail2
    (if 3 ≤ root499Candidates1.length then search edge 3 [3583, 4423] root499Candidates1 8006 winner else winner) = _
  rw [if_neg (by decide)]
def root499Candidates2 : List Nat := [9631]
set_option maxRecDepth 8192 in
theorem root499Candidates2_checked : candidates edge 4 3583 4651 root499Tail3 winner.bound = root499Candidates2 := by
  unfold candidates
  have row : edge 4651 = (SortedNatTree.contains FastRows.row4651) := by
    change Data.row4651.contains = (SortedNatTree.contains FastRows.row4651)
    exact FastRows.row4651_checked.symm
  rw [row]
  decide
theorem root499Step2 : scan edge (search edge 3) 4 [3583] 3583 root499Tail2 winner =
    scan edge (search edge 3) 4 [3583] 3583 root499Tail3 winner := by
  rw [root499Tail2, scan, if_neg (by decide), root499Candidates2_checked]
  change scan edge (search edge 3) 4 [3583] 3583 root499Tail3
    (if 3 ≤ root499Candidates2.length then search edge 3 [3583, 4651] root499Candidates2 8234 winner else winner) = _
  rw [if_neg (by decide)]
def root499Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root499Candidates3_checked : candidates edge 4 3583 5179 root499Tail4 winner.bound = root499Candidates3 := by
  unfold candidates
  have row : edge 5179 = (SortedNatTree.contains FastRows.row5179) := by
    change Data.row5179.contains = (SortedNatTree.contains FastRows.row5179)
    exact FastRows.row5179_checked.symm
  rw [row]
  decide
theorem root499Step3 : scan edge (search edge 3) 4 [3583] 3583 root499Tail3 winner =
    scan edge (search edge 3) 4 [3583] 3583 root499Tail4 winner := by
  rw [root499Tail3, scan, if_neg (by decide), root499Candidates3_checked]
  change scan edge (search edge 3) 4 [3583] 3583 root499Tail4
    (if 3 ≤ root499Candidates3.length then search edge 3 [3583, 5179] root499Candidates3 8762 winner else winner) = _
  rw [if_neg (by decide)]
theorem root499Scan4 : scan edge (search edge 3) 4 [3583] 3583 root499Tail4 winner = winner := by rw [root499Tail4, scan, if_pos (by decide)]
theorem root499Scan3 : scan edge (search edge 3) 4 [3583] 3583 root499Tail3 winner = winner :=
  (root499Step3).trans root499Scan4
theorem root499Scan2 : scan edge (search edge 3) 4 [3583] 3583 root499Tail2 winner = winner :=
  (root499Step2).trans root499Scan3
theorem root499Scan1 : scan edge (search edge 3) 4 [3583] 3583 root499Tail1 winner = winner :=
  (root499Step1).trans root499Scan2
theorem root499Scan0 : scan edge (search edge 3) 4 [3583] 3583 root499Tail0 winner = winner :=
  (root499Step0).trans root499Scan1
set_option maxRecDepth 4096 in
theorem root499Branch : search edge 4 [3583] root499Candidates 3583 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3583] 3583 root499Tail0 winner = winner
  exact root499Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
