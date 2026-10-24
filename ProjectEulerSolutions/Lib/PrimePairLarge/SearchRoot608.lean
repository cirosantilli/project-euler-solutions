import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root608Candidates : List Nat := Data.row4507
set_option maxRecDepth 8192 in
theorem root608Candidates_checked : candidates edge 5 0 4507 tail609 winner.bound = root608Candidates := by
  apply candidates_eq_row edge 5 0 4507 tail609 root608Candidates winner.bound tail609_ordered
  · decide
  · decide
  · funext q
    rfl
def root608Tail41 : List Nat := []
def root608Tail40 : List Nat := 21433 :: root608Tail41
def root608Tail39 : List Nat := 21319 :: root608Tail40
def root608Tail38 : List Nat := 21283 :: root608Tail39
def root608Tail37 : List Nat := 19309 :: root608Tail38
def root608Tail36 : List Nat := 18451 :: root608Tail37
def root608Tail35 : List Nat := 18211 :: root608Tail36
def root608Tail34 : List Nat := 18061 :: root608Tail35
def root608Tail33 : List Nat := 17011 :: root608Tail34
def root608Tail32 : List Nat := 16759 :: root608Tail33
def root608Tail31 : List Nat := 16267 :: root608Tail32
def root608Tail30 : List Nat := 15919 :: root608Tail31
def root608Tail29 : List Nat := 15439 :: root608Tail30
def root608Tail28 : List Nat := 15277 :: root608Tail29
def root608Tail27 : List Nat := 15217 :: root608Tail28
def root608Tail26 : List Nat := 14533 :: root608Tail27
def root608Tail25 : List Nat := 13807 :: root608Tail26
def root608Tail24 : List Nat := 13633 :: root608Tail25
def root608Tail23 : List Nat := 13327 :: root608Tail24
def root608Tail22 : List Nat := 12907 :: root608Tail23
def root608Tail21 : List Nat := 12739 :: root608Tail22
def root608Tail20 : List Nat := 12421 :: root608Tail21
def root608Tail19 : List Nat := 12277 :: root608Tail20
def root608Tail18 : List Nat := 12253 :: root608Tail19
def root608Tail17 : List Nat := 11257 :: root608Tail18
def root608Tail16 : List Nat := 9967 :: root608Tail17
def root608Tail15 : List Nat := 9901 :: root608Tail16
def root608Tail14 : List Nat := 9187 :: root608Tail15
def root608Tail13 : List Nat := 8893 :: root608Tail14
def root608Tail12 : List Nat := 8377 :: root608Tail13
def root608Tail11 : List Nat := 7993 :: root608Tail12
def root608Tail10 : List Nat := 7759 :: root608Tail11
def root608Tail9 : List Nat := 7603 :: root608Tail10
def root608Tail8 : List Nat := 7129 :: root608Tail9
def root608Tail7 : List Nat := 6709 :: root608Tail8
def root608Tail6 : List Nat := 6217 :: root608Tail7
def root608Tail5 : List Nat := 6091 :: root608Tail6
def root608Tail4 : List Nat := 5881 :: root608Tail5
def root608Tail3 : List Nat := 5419 :: root608Tail4
def root608Tail2 : List Nat := 4999 :: root608Tail3
def root608Tail1 : List Nat := 4801 :: root608Tail2
def root608Tail0 : List Nat := 4789 :: root608Tail1
def root608Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root608Candidates0_checked : candidates edge 4 4507 4789 root608Tail1 winner.bound = root608Candidates0 := by
  unfold candidates
  have row : edge 4789 = (SortedNatTree.contains FastRows.row4789) := by
    change Data.row4789.contains = (SortedNatTree.contains FastRows.row4789)
    exact FastRows.row4789_checked.symm
  rw [row]
  decide
theorem root608Step0 : scan edge (search edge 3) 4 [4507] 4507 root608Tail0 winner =
    scan edge (search edge 3) 4 [4507] 4507 root608Tail1 winner := by
  rw [root608Tail0, scan, if_neg (by decide), root608Candidates0_checked]
  change scan edge (search edge 3) 4 [4507] 4507 root608Tail1
    (if 3 ≤ root608Candidates0.length then search edge 3 [4507, 4789] root608Candidates0 9296 winner else winner) = _
  rw [if_neg (by decide)]
def root608Candidates1 : List Nat := [5419, 7993, 12739, 15919]
set_option maxRecDepth 8192 in
theorem root608Candidates1_checked : candidates edge 4 4507 4801 root608Tail2 winner.bound = root608Candidates1 := by
  unfold candidates
  have row : edge 4801 = (SortedNatTree.contains FastRows.row4801) := by
    change Data.row4801.contains = (SortedNatTree.contains FastRows.row4801)
    exact FastRows.row4801_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root608Child1 : search edge 3 [4507, 4801] root608Candidates1 9308 winner = winner := by decide
theorem root608Step1 : scan edge (search edge 3) 4 [4507] 4507 root608Tail1 winner =
    scan edge (search edge 3) 4 [4507] 4507 root608Tail2 winner := by
  rw [root608Tail1, scan, if_neg (by decide), root608Candidates1_checked]
  change scan edge (search edge 3) 4 [4507] 4507 root608Tail2
    (if 3 ≤ root608Candidates1.length then search edge 3 [4507, 4801] root608Candidates1 9308 winner else winner) = _
  rw [if_pos (by decide), root608Child1]
def root608Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root608Candidates2_checked : candidates edge 4 4507 4999 root608Tail3 winner.bound = root608Candidates2 := by
  unfold candidates
  have row : edge 4999 = (SortedNatTree.contains FastRows.row4999) := by
    change Data.row4999.contains = (SortedNatTree.contains FastRows.row4999)
    exact FastRows.row4999_checked.symm
  rw [row]
  decide
theorem root608Step2 : scan edge (search edge 3) 4 [4507] 4507 root608Tail2 winner =
    scan edge (search edge 3) 4 [4507] 4507 root608Tail3 winner := by
  rw [root608Tail2, scan, if_neg (by decide), root608Candidates2_checked]
  change scan edge (search edge 3) 4 [4507] 4507 root608Tail3
    (if 3 ≤ root608Candidates2.length then search edge 3 [4507, 4999] root608Candidates2 9506 winner else winner) = _
  rw [if_neg (by decide)]
theorem root608Scan3 : scan edge (search edge 3) 4 [4507] 4507 root608Tail3 winner = winner := by rw [root608Tail3, scan, if_pos (by decide)]
theorem root608Scan2 : scan edge (search edge 3) 4 [4507] 4507 root608Tail2 winner = winner :=
  (root608Step2).trans root608Scan3
theorem root608Scan1 : scan edge (search edge 3) 4 [4507] 4507 root608Tail1 winner = winner :=
  (root608Step1).trans root608Scan2
theorem root608Scan0 : scan edge (search edge 3) 4 [4507] 4507 root608Tail0 winner = winner :=
  (root608Step0).trans root608Scan1
set_option maxRecDepth 4096 in
theorem root608Branch : search edge 4 [4507] root608Candidates 4507 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4507] 4507 root608Tail0 winner = winner
  exact root608Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
