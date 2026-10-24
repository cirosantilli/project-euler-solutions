import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root533Candidates : List Nat := Data.row3863
set_option maxRecDepth 8192 in
theorem root533Candidates_checked : candidates edge 5 0 3863 tail534 winner.bound = root533Candidates := by
  apply candidates_eq_row edge 5 0 3863 tail534 root533Candidates winner.bound tail534_ordered
  · decide
  · decide
  · funext q
    rfl
def root533Tail33 : List Nat := []
def root533Tail32 : List Nat := 22037 :: root533Tail33
def root533Tail31 : List Nat := 21977 :: root533Tail32
def root533Tail30 : List Nat := 21323 :: root533Tail31
def root533Tail29 : List Nat := 21017 :: root533Tail30
def root533Tail28 : List Nat := 20261 :: root533Tail29
def root533Tail27 : List Nat := 20201 :: root533Tail28
def root533Tail26 : List Nat := 19463 :: root533Tail27
def root533Tail25 : List Nat := 19289 :: root533Tail26
def root533Tail24 : List Nat := 19013 :: root533Tail25
def root533Tail23 : List Nat := 18719 :: root533Tail24
def root533Tail22 : List Nat := 18329 :: root533Tail23
def root533Tail21 : List Nat := 17471 :: root533Tail22
def root533Tail20 : List Nat := 17093 :: root533Tail21
def root533Tail19 : List Nat := 16979 :: root533Tail20
def root533Tail18 : List Nat := 14879 :: root533Tail19
def root533Tail17 : List Nat := 14717 :: root533Tail18
def root533Tail16 : List Nat := 12323 :: root533Tail17
def root533Tail15 : List Nat := 12041 :: root533Tail16
def root533Tail14 : List Nat := 11789 :: root533Tail15
def root533Tail13 : List Nat := 11777 :: root533Tail14
def root533Tail12 : List Nat := 10739 :: root533Tail13
def root533Tail11 : List Nat := 9203 :: root533Tail12
def root533Tail10 : List Nat := 8867 :: root533Tail11
def root533Tail9 : List Nat := 8849 :: root533Tail10
def root533Tail8 : List Nat := 8543 :: root533Tail9
def root533Tail7 : List Nat := 7757 :: root533Tail8
def root533Tail6 : List Nat := 7673 :: root533Tail7
def root533Tail5 : List Nat := 7187 :: root533Tail6
def root533Tail4 : List Nat := 6779 :: root533Tail5
def root533Tail3 : List Nat := 6569 :: root533Tail4
def root533Tail2 : List Nat := 5297 :: root533Tail3
def root533Tail1 : List Nat := 4451 :: root533Tail2
def root533Tail0 : List Nat := 4007 :: root533Tail1
def root533Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root533Candidates0_checked : candidates edge 4 3863 4007 root533Tail1 winner.bound = root533Candidates0 := by
  unfold candidates
  have row : edge 4007 = (SortedNatTree.contains FastRows.row4007) := by
    change Data.row4007.contains = (SortedNatTree.contains FastRows.row4007)
    exact FastRows.row4007_checked.symm
  rw [row]
  decide
theorem root533Step0 : scan edge (search edge 3) 4 [3863] 3863 root533Tail0 winner =
    scan edge (search edge 3) 4 [3863] 3863 root533Tail1 winner := by
  rw [root533Tail0, scan, if_neg (by decide), root533Candidates0_checked]
  change scan edge (search edge 3) 4 [3863] 3863 root533Tail1
    (if 3 ≤ root533Candidates0.length then search edge 3 [3863, 4007] root533Candidates0 7870 winner else winner) = _
  rw [if_neg (by decide)]
def root533Candidates1 : List Nat := [11789, 17471]
set_option maxRecDepth 8192 in
theorem root533Candidates1_checked : candidates edge 4 3863 4451 root533Tail2 winner.bound = root533Candidates1 := by
  unfold candidates
  have row : edge 4451 = (SortedNatTree.contains FastRows.row4451) := by
    change Data.row4451.contains = (SortedNatTree.contains FastRows.row4451)
    exact FastRows.row4451_checked.symm
  rw [row]
  decide
theorem root533Step1 : scan edge (search edge 3) 4 [3863] 3863 root533Tail1 winner =
    scan edge (search edge 3) 4 [3863] 3863 root533Tail2 winner := by
  rw [root533Tail1, scan, if_neg (by decide), root533Candidates1_checked]
  change scan edge (search edge 3) 4 [3863] 3863 root533Tail2
    (if 3 ≤ root533Candidates1.length then search edge 3 [3863, 4451] root533Candidates1 8314 winner else winner) = _
  rw [if_neg (by decide)]
def root533Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root533Candidates2_checked : candidates edge 4 3863 5297 root533Tail3 winner.bound = root533Candidates2 := by
  unfold candidates
  have row : edge 5297 = (SortedNatTree.contains FastRows.row5297) := by
    change Data.row5297.contains = (SortedNatTree.contains FastRows.row5297)
    exact FastRows.row5297_checked.symm
  rw [row]
  decide
theorem root533Step2 : scan edge (search edge 3) 4 [3863] 3863 root533Tail2 winner =
    scan edge (search edge 3) 4 [3863] 3863 root533Tail3 winner := by
  rw [root533Tail2, scan, if_neg (by decide), root533Candidates2_checked]
  change scan edge (search edge 3) 4 [3863] 3863 root533Tail3
    (if 3 ≤ root533Candidates2.length then search edge 3 [3863, 5297] root533Candidates2 9160 winner else winner) = _
  rw [if_neg (by decide)]
theorem root533Scan3 : scan edge (search edge 3) 4 [3863] 3863 root533Tail3 winner = winner := by rw [root533Tail3, scan, if_pos (by decide)]
theorem root533Scan2 : scan edge (search edge 3) 4 [3863] 3863 root533Tail2 winner = winner :=
  (root533Step2).trans root533Scan3
theorem root533Scan1 : scan edge (search edge 3) 4 [3863] 3863 root533Tail1 winner = winner :=
  (root533Step1).trans root533Scan2
theorem root533Scan0 : scan edge (search edge 3) 4 [3863] 3863 root533Tail0 winner = winner :=
  (root533Step0).trans root533Scan1
set_option maxRecDepth 4096 in
theorem root533Branch : search edge 4 [3863] root533Candidates 3863 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3863] 3863 root533Tail0 winner = winner
  exact root533Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
