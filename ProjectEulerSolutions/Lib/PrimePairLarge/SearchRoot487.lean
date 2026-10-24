import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root487Candidates : List Nat := Data.row3511
set_option maxRecDepth 8192 in
theorem root487Candidates_checked : candidates edge 5 0 3511 tail488 winner.bound = root487Candidates := by
  apply candidates_eq_row edge 5 0 3511 tail488 root487Candidates winner.bound tail488_ordered
  · decide
  · decide
  · funext q
    rfl
def root487Tail37 : List Nat := []
def root487Tail36 : List Nat := 21997 :: root487Tail37
def root487Tail35 : List Nat := 20899 :: root487Tail36
def root487Tail34 : List Nat := 20857 :: root487Tail35
def root487Tail33 : List Nat := 20233 :: root487Tail34
def root487Tail32 : List Nat := 19861 :: root487Tail33
def root487Tail31 : List Nat := 19429 :: root487Tail32
def root487Tail30 : List Nat := 19009 :: root487Tail31
def root487Tail29 : List Nat := 18637 :: root487Tail30
def root487Tail28 : List Nat := 18433 :: root487Tail29
def root487Tail27 : List Nat := 17707 :: root487Tail28
def root487Tail26 : List Nat := 16693 :: root487Tail27
def root487Tail25 : List Nat := 16231 :: root487Tail26
def root487Tail24 : List Nat := 15601 :: root487Tail25
def root487Tail23 : List Nat := 14563 :: root487Tail24
def root487Tail22 : List Nat := 14173 :: root487Tail23
def root487Tail21 : List Nat := 13591 :: root487Tail22
def root487Tail20 : List Nat := 13417 :: root487Tail21
def root487Tail19 : List Nat := 12919 :: root487Tail20
def root487Tail18 : List Nat := 12451 :: root487Tail19
def root487Tail17 : List Nat := 12007 :: root487Tail18
def root487Tail16 : List Nat := 11161 :: root487Tail17
def root487Tail15 : List Nat := 11113 :: root487Tail16
def root487Tail14 : List Nat := 11083 :: root487Tail15
def root487Tail13 : List Nat := 11071 :: root487Tail14
def root487Tail12 : List Nat := 9439 :: root487Tail13
def root487Tail11 : List Nat := 9283 :: root487Tail12
def root487Tail10 : List Nat := 9181 :: root487Tail11
def root487Tail9 : List Nat := 8527 :: root487Tail10
def root487Tail8 : List Nat := 8089 :: root487Tail9
def root487Tail7 : List Nat := 8011 :: root487Tail8
def root487Tail6 : List Nat := 7963 :: root487Tail7
def root487Tail5 : List Nat := 7879 :: root487Tail6
def root487Tail4 : List Nat := 7561 :: root487Tail5
def root487Tail3 : List Nat := 6967 :: root487Tail4
def root487Tail2 : List Nat := 5233 :: root487Tail3
def root487Tail1 : List Nat := 4153 :: root487Tail2
def root487Tail0 : List Nat := 3889 :: root487Tail1
def root487Candidates0 : List Nat := [9439]
set_option maxRecDepth 8192 in
theorem root487Candidates0_checked : candidates edge 4 3511 3889 root487Tail1 winner.bound = root487Candidates0 := by
  unfold candidates
  have row : edge 3889 = (SortedNatTree.contains FastRows.row3889) := by
    change Data.row3889.contains = (SortedNatTree.contains FastRows.row3889)
    exact FastRows.row3889_checked.symm
  rw [row]
  decide
theorem root487Step0 : scan edge (search edge 3) 4 [3511] 3511 root487Tail0 winner =
    scan edge (search edge 3) 4 [3511] 3511 root487Tail1 winner := by
  rw [root487Tail0, scan, if_neg (by decide), root487Candidates0_checked]
  change scan edge (search edge 3) 4 [3511] 3511 root487Tail1
    (if 3 ≤ root487Candidates0.length then search edge 3 [3511, 3889] root487Candidates0 7400 winner else winner) = _
  rw [if_neg (by decide)]
def root487Candidates1 : List Nat := [9283]
set_option maxRecDepth 8192 in
theorem root487Candidates1_checked : candidates edge 4 3511 4153 root487Tail2 winner.bound = root487Candidates1 := by
  unfold candidates
  have row : edge 4153 = (SortedNatTree.contains FastRows.row4153) := by
    change Data.row4153.contains = (SortedNatTree.contains FastRows.row4153)
    exact FastRows.row4153_checked.symm
  rw [row]
  decide
theorem root487Step1 : scan edge (search edge 3) 4 [3511] 3511 root487Tail1 winner =
    scan edge (search edge 3) 4 [3511] 3511 root487Tail2 winner := by
  rw [root487Tail1, scan, if_neg (by decide), root487Candidates1_checked]
  change scan edge (search edge 3) 4 [3511] 3511 root487Tail2
    (if 3 ≤ root487Candidates1.length then search edge 3 [3511, 4153] root487Candidates1 7664 winner else winner) = _
  rw [if_neg (by decide)]
def root487Candidates2 : List Nat := [8089, 13591]
set_option maxRecDepth 8192 in
theorem root487Candidates2_checked : candidates edge 4 3511 5233 root487Tail3 winner.bound = root487Candidates2 := by
  unfold candidates
  have row : edge 5233 = (SortedNatTree.contains FastRows.row5233) := by
    change Data.row5233.contains = (SortedNatTree.contains FastRows.row5233)
    exact FastRows.row5233_checked.symm
  rw [row]
  decide
theorem root487Step2 : scan edge (search edge 3) 4 [3511] 3511 root487Tail2 winner =
    scan edge (search edge 3) 4 [3511] 3511 root487Tail3 winner := by
  rw [root487Tail2, scan, if_neg (by decide), root487Candidates2_checked]
  change scan edge (search edge 3) 4 [3511] 3511 root487Tail3
    (if 3 ≤ root487Candidates2.length then search edge 3 [3511, 5233] root487Candidates2 8744 winner else winner) = _
  rw [if_neg (by decide)]
theorem root487Scan3 : scan edge (search edge 3) 4 [3511] 3511 root487Tail3 winner = winner := by rw [root487Tail3, scan, if_pos (by decide)]
theorem root487Scan2 : scan edge (search edge 3) 4 [3511] 3511 root487Tail2 winner = winner :=
  (root487Step2).trans root487Scan3
theorem root487Scan1 : scan edge (search edge 3) 4 [3511] 3511 root487Tail1 winner = winner :=
  (root487Step1).trans root487Scan2
theorem root487Scan0 : scan edge (search edge 3) 4 [3511] 3511 root487Tail0 winner = winner :=
  (root487Step0).trans root487Scan1
set_option maxRecDepth 4096 in
theorem root487Branch : search edge 4 [3511] root487Candidates 3511 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3511] 3511 root487Tail0 winner = winner
  exact root487Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
