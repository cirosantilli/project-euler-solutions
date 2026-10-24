import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root417Candidates : List Nat := Data.row2903
set_option maxRecDepth 8192 in
theorem root417Candidates_checked : candidates edge 5 0 2903 tail418 winner.bound = root417Candidates := by
  apply candidates_eq_row edge 5 0 2903 tail418 root417Candidates winner.bound tail418_ordered
  · decide
  · decide
  · funext q
    rfl
def root417Tail39 : List Nat := []
def root417Tail38 : List Nat := 22937 :: root417Tail39
def root417Tail37 : List Nat := 22811 :: root417Tail38
def root417Tail36 : List Nat := 21839 :: root417Tail37
def root417Tail35 : List Nat := 20201 :: root417Tail36
def root417Tail34 : List Nat := 19937 :: root417Tail35
def root417Tail33 : List Nat := 19853 :: root417Tail34
def root417Tail32 : List Nat := 18773 :: root417Tail33
def root417Tail31 : List Nat := 18341 :: root417Tail32
def root417Tail30 : List Nat := 17747 :: root417Tail31
def root417Tail29 : List Nat := 17471 :: root417Tail30
def root417Tail28 : List Nat := 16001 :: root417Tail29
def root417Tail27 : List Nat := 15101 :: root417Tail28
def root417Tail26 : List Nat := 14051 :: root417Tail27
def root417Tail25 : List Nat := 13841 :: root417Tail26
def root417Tail24 : List Nat := 13799 :: root417Tail25
def root417Tail23 : List Nat := 13469 :: root417Tail24
def root417Tail22 : List Nat := 13007 :: root417Tail23
def root417Tail21 : List Nat := 11909 :: root417Tail22
def root417Tail20 : List Nat := 11003 :: root417Tail21
def root417Tail19 : List Nat := 10667 :: root417Tail20
def root417Tail18 : List Nat := 10607 :: root417Tail19
def root417Tail17 : List Nat := 9851 :: root417Tail18
def root417Tail16 : List Nat := 9539 :: root417Tail17
def root417Tail15 : List Nat := 9173 :: root417Tail16
def root417Tail14 : List Nat := 8963 :: root417Tail15
def root417Tail13 : List Nat := 8123 :: root417Tail14
def root417Tail12 : List Nat := 7457 :: root417Tail13
def root417Tail11 : List Nat := 7433 :: root417Tail12
def root417Tail10 : List Nat := 7121 :: root417Tail11
def root417Tail9 : List Nat := 7103 :: root417Tail10
def root417Tail8 : List Nat := 6779 :: root417Tail9
def root417Tail7 : List Nat := 6701 :: root417Tail8
def root417Tail6 : List Nat := 6131 :: root417Tail7
def root417Tail5 : List Nat := 5927 :: root417Tail6
def root417Tail4 : List Nat := 4721 :: root417Tail5
def root417Tail3 : List Nat := 4637 :: root417Tail4
def root417Tail2 : List Nat := 4409 :: root417Tail3
def root417Tail1 : List Nat := 4373 :: root417Tail2
def root417Tail0 : List Nat := 2963 :: root417Tail1
def root417Candidates0 : List Nat := [9851]
set_option maxRecDepth 8192 in
theorem root417Candidates0_checked : candidates edge 4 2903 2963 root417Tail1 winner.bound = root417Candidates0 := by
  unfold candidates
  have row : edge 2963 = (SortedNatTree.contains FastRows.row2963) := by
    change Data.row2963.contains = (SortedNatTree.contains FastRows.row2963)
    exact FastRows.row2963_checked.symm
  rw [row]
  decide
theorem root417Step0 : scan edge (search edge 3) 4 [2903] 2903 root417Tail0 winner =
    scan edge (search edge 3) 4 [2903] 2903 root417Tail1 winner := by
  rw [root417Tail0, scan, if_neg (by decide), root417Candidates0_checked]
  change scan edge (search edge 3) 4 [2903] 2903 root417Tail1
    (if 3 ≤ root417Candidates0.length then search edge 3 [2903, 2963] root417Candidates0 5866 winner else winner) = _
  rw [if_neg (by decide)]
def root417Candidates1 : List Nat := [5927, 7121]
set_option maxRecDepth 8192 in
theorem root417Candidates1_checked : candidates edge 4 2903 4373 root417Tail2 winner.bound = root417Candidates1 := by
  unfold candidates
  have row : edge 4373 = (SortedNatTree.contains FastRows.row4373) := by
    change Data.row4373.contains = (SortedNatTree.contains FastRows.row4373)
    exact FastRows.row4373_checked.symm
  rw [row]
  decide
theorem root417Step1 : scan edge (search edge 3) 4 [2903] 2903 root417Tail1 winner =
    scan edge (search edge 3) 4 [2903] 2903 root417Tail2 winner := by
  rw [root417Tail1, scan, if_neg (by decide), root417Candidates1_checked]
  change scan edge (search edge 3) 4 [2903] 2903 root417Tail2
    (if 3 ≤ root417Candidates1.length then search edge 3 [2903, 4373] root417Candidates1 7276 winner else winner) = _
  rw [if_neg (by decide)]
def root417Candidates2 : List Nat := [13469]
set_option maxRecDepth 8192 in
theorem root417Candidates2_checked : candidates edge 4 2903 4409 root417Tail3 winner.bound = root417Candidates2 := by
  unfold candidates
  have row : edge 4409 = (SortedNatTree.contains FastRows.row4409) := by
    change Data.row4409.contains = (SortedNatTree.contains FastRows.row4409)
    exact FastRows.row4409_checked.symm
  rw [row]
  decide
theorem root417Step2 : scan edge (search edge 3) 4 [2903] 2903 root417Tail2 winner =
    scan edge (search edge 3) 4 [2903] 2903 root417Tail3 winner := by
  rw [root417Tail2, scan, if_neg (by decide), root417Candidates2_checked]
  change scan edge (search edge 3) 4 [2903] 2903 root417Tail3
    (if 3 ≤ root417Candidates2.length then search edge 3 [2903, 4409] root417Candidates2 7312 winner else winner) = _
  rw [if_neg (by decide)]
def root417Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root417Candidates3_checked : candidates edge 4 2903 4637 root417Tail4 winner.bound = root417Candidates3 := by
  unfold candidates
  have row : edge 4637 = (SortedNatTree.contains FastRows.row4637) := by
    change Data.row4637.contains = (SortedNatTree.contains FastRows.row4637)
    exact FastRows.row4637_checked.symm
  rw [row]
  decide
theorem root417Step3 : scan edge (search edge 3) 4 [2903] 2903 root417Tail3 winner =
    scan edge (search edge 3) 4 [2903] 2903 root417Tail4 winner := by
  rw [root417Tail3, scan, if_neg (by decide), root417Candidates3_checked]
  change scan edge (search edge 3) 4 [2903] 2903 root417Tail4
    (if 3 ≤ root417Candidates3.length then search edge 3 [2903, 4637] root417Candidates3 7540 winner else winner) = _
  rw [if_neg (by decide)]
def root417Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root417Candidates4_checked : candidates edge 4 2903 4721 root417Tail5 winner.bound = root417Candidates4 := by
  unfold candidates
  have row : edge 4721 = (SortedNatTree.contains FastRows.row4721) := by
    change Data.row4721.contains = (SortedNatTree.contains FastRows.row4721)
    exact FastRows.row4721_checked.symm
  rw [row]
  decide
theorem root417Step4 : scan edge (search edge 3) 4 [2903] 2903 root417Tail4 winner =
    scan edge (search edge 3) 4 [2903] 2903 root417Tail5 winner := by
  rw [root417Tail4, scan, if_neg (by decide), root417Candidates4_checked]
  change scan edge (search edge 3) 4 [2903] 2903 root417Tail5
    (if 3 ≤ root417Candidates4.length then search edge 3 [2903, 4721] root417Candidates4 7624 winner else winner) = _
  rw [if_neg (by decide)]
theorem root417Scan5 : scan edge (search edge 3) 4 [2903] 2903 root417Tail5 winner = winner := by rw [root417Tail5, scan, if_pos (by decide)]
theorem root417Scan4 : scan edge (search edge 3) 4 [2903] 2903 root417Tail4 winner = winner :=
  (root417Step4).trans root417Scan5
theorem root417Scan3 : scan edge (search edge 3) 4 [2903] 2903 root417Tail3 winner = winner :=
  (root417Step3).trans root417Scan4
theorem root417Scan2 : scan edge (search edge 3) 4 [2903] 2903 root417Tail2 winner = winner :=
  (root417Step2).trans root417Scan3
theorem root417Scan1 : scan edge (search edge 3) 4 [2903] 2903 root417Tail1 winner = winner :=
  (root417Step1).trans root417Scan2
theorem root417Scan0 : scan edge (search edge 3) 4 [2903] 2903 root417Tail0 winner = winner :=
  (root417Step0).trans root417Scan1
set_option maxRecDepth 4096 in
theorem root417Branch : search edge 4 [2903] root417Candidates 2903 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2903] 2903 root417Tail0 winner = winner
  exact root417Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
