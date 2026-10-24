import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root395Candidates : List Nat := Data.row2729
set_option maxRecDepth 8192 in
theorem root395Candidates_checked : candidates edge 5 0 2729 tail396 winner.bound = root395Candidates := by
  apply candidates_eq_row edge 5 0 2729 tail396 root395Candidates winner.bound tail396_ordered
  · decide
  · decide
  · funext q
    rfl
def root395Tail34 : List Nat := []
def root395Tail33 : List Nat := 22961 :: root395Tail34
def root395Tail32 : List Nat := 22193 :: root395Tail33
def root395Tail31 : List Nat := 21599 :: root395Tail32
def root395Tail30 : List Nat := 20357 :: root395Tail31
def root395Tail29 : List Nat := 19979 :: root395Tail30
def root395Tail28 : List Nat := 19181 :: root395Tail29
def root395Tail27 : List Nat := 18341 :: root395Tail28
def root395Tail26 : List Nat := 18089 :: root395Tail27
def root395Tail25 : List Nat := 17123 :: root395Tail26
def root395Tail24 : List Nat := 16223 :: root395Tail25
def root395Tail23 : List Nat := 15809 :: root395Tail24
def root395Tail22 : List Nat := 15767 :: root395Tail23
def root395Tail21 : List Nat := 15581 :: root395Tail22
def root395Tail20 : List Nat := 15173 :: root395Tail21
def root395Tail19 : List Nat := 14897 :: root395Tail20
def root395Tail18 : List Nat := 13829 :: root395Tail19
def root395Tail17 : List Nat := 13691 :: root395Tail18
def root395Tail16 : List Nat := 13649 :: root395Tail17
def root395Tail15 : List Nat := 13457 :: root395Tail16
def root395Tail14 : List Nat := 13037 :: root395Tail15
def root395Tail13 : List Nat := 12659 :: root395Tail14
def root395Tail12 : List Nat := 11789 :: root395Tail13
def root395Tail11 : List Nat := 11213 :: root395Tail12
def root395Tail10 : List Nat := 10973 :: root395Tail11
def root395Tail9 : List Nat := 10193 :: root395Tail10
def root395Tail8 : List Nat := 9719 :: root395Tail9
def root395Tail7 : List Nat := 8081 :: root395Tail8
def root395Tail6 : List Nat := 6551 :: root395Tail7
def root395Tail5 : List Nat := 6473 :: root395Tail6
def root395Tail4 : List Nat := 6131 :: root395Tail5
def root395Tail3 : List Nat := 5813 :: root395Tail4
def root395Tail2 : List Nat := 4409 :: root395Tail3
def root395Tail1 : List Nat := 3191 :: root395Tail2
def root395Tail0 : List Nat := 3023 :: root395Tail1
def root395Candidates0 : List Nat := [6551]
set_option maxRecDepth 8192 in
theorem root395Candidates0_checked : candidates edge 4 2729 3023 root395Tail1 winner.bound = root395Candidates0 := by
  unfold candidates
  have row : edge 3023 = (SortedNatTree.contains FastRows.row3023) := by
    change Data.row3023.contains = (SortedNatTree.contains FastRows.row3023)
    exact FastRows.row3023_checked.symm
  rw [row]
  decide
theorem root395Step0 : scan edge (search edge 3) 4 [2729] 2729 root395Tail0 winner =
    scan edge (search edge 3) 4 [2729] 2729 root395Tail1 winner := by
  rw [root395Tail0, scan, if_neg (by decide), root395Candidates0_checked]
  change scan edge (search edge 3) 4 [2729] 2729 root395Tail1
    (if 3 ≤ root395Candidates0.length then search edge 3 [2729, 3023] root395Candidates0 5752 winner else winner) = _
  rw [if_neg (by decide)]
def root395Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root395Candidates1_checked : candidates edge 4 2729 3191 root395Tail2 winner.bound = root395Candidates1 := by
  unfold candidates
  have row : edge 3191 = (SortedNatTree.contains FastRows.row3191) := by
    change Data.row3191.contains = (SortedNatTree.contains FastRows.row3191)
    exact FastRows.row3191_checked.symm
  rw [row]
  decide
theorem root395Step1 : scan edge (search edge 3) 4 [2729] 2729 root395Tail1 winner =
    scan edge (search edge 3) 4 [2729] 2729 root395Tail2 winner := by
  rw [root395Tail1, scan, if_neg (by decide), root395Candidates1_checked]
  change scan edge (search edge 3) 4 [2729] 2729 root395Tail2
    (if 3 ≤ root395Candidates1.length then search edge 3 [2729, 3191] root395Candidates1 5920 winner else winner) = _
  rw [if_neg (by decide)]
def root395Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root395Candidates2_checked : candidates edge 4 2729 4409 root395Tail3 winner.bound = root395Candidates2 := by
  unfold candidates
  have row : edge 4409 = (SortedNatTree.contains FastRows.row4409) := by
    change Data.row4409.contains = (SortedNatTree.contains FastRows.row4409)
    exact FastRows.row4409_checked.symm
  rw [row]
  decide
theorem root395Step2 : scan edge (search edge 3) 4 [2729] 2729 root395Tail2 winner =
    scan edge (search edge 3) 4 [2729] 2729 root395Tail3 winner := by
  rw [root395Tail2, scan, if_neg (by decide), root395Candidates2_checked]
  change scan edge (search edge 3) 4 [2729] 2729 root395Tail3
    (if 3 ≤ root395Candidates2.length then search edge 3 [2729, 4409] root395Candidates2 7138 winner else winner) = _
  rw [if_neg (by decide)]
def root395Candidates3 : List Nat := [12659]
set_option maxRecDepth 8192 in
theorem root395Candidates3_checked : candidates edge 4 2729 5813 root395Tail4 winner.bound = root395Candidates3 := by
  unfold candidates
  have row : edge 5813 = (SortedNatTree.contains FastRows.row5813) := by
    change Data.row5813.contains = (SortedNatTree.contains FastRows.row5813)
    exact FastRows.row5813_checked.symm
  rw [row]
  decide
theorem root395Step3 : scan edge (search edge 3) 4 [2729] 2729 root395Tail3 winner =
    scan edge (search edge 3) 4 [2729] 2729 root395Tail4 winner := by
  rw [root395Tail3, scan, if_neg (by decide), root395Candidates3_checked]
  change scan edge (search edge 3) 4 [2729] 2729 root395Tail4
    (if 3 ≤ root395Candidates3.length then search edge 3 [2729, 5813] root395Candidates3 8542 winner else winner) = _
  rw [if_neg (by decide)]
theorem root395Scan4 : scan edge (search edge 3) 4 [2729] 2729 root395Tail4 winner = winner := by rw [root395Tail4, scan, if_pos (by decide)]
theorem root395Scan3 : scan edge (search edge 3) 4 [2729] 2729 root395Tail3 winner = winner :=
  (root395Step3).trans root395Scan4
theorem root395Scan2 : scan edge (search edge 3) 4 [2729] 2729 root395Tail2 winner = winner :=
  (root395Step2).trans root395Scan3
theorem root395Scan1 : scan edge (search edge 3) 4 [2729] 2729 root395Tail1 winner = winner :=
  (root395Step1).trans root395Scan2
theorem root395Scan0 : scan edge (search edge 3) 4 [2729] 2729 root395Tail0 winner = winner :=
  (root395Step0).trans root395Scan1
set_option maxRecDepth 4096 in
theorem root395Branch : search edge 4 [2729] root395Candidates 2729 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2729] 2729 root395Tail0 winner = winner
  exact root395Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
