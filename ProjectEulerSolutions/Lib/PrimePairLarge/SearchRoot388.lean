import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root388Candidates : List Nat := Data.row2689
set_option maxRecDepth 8192 in
theorem root388Candidates_checked : candidates edge 5 0 2689 tail389 winner.bound = root388Candidates := by
  apply candidates_eq_row edge 5 0 2689 tail389 root388Candidates winner.bound tail389_ordered
  · decide
  · decide
  · funext q
    rfl
def root388Tail38 : List Nat := []
def root388Tail37 : List Nat := 22921 :: root388Tail38
def root388Tail36 : List Nat := 22273 :: root388Tail37
def root388Tail35 : List Nat := 22189 :: root388Tail36
def root388Tail34 : List Nat := 21787 :: root388Tail35
def root388Tail33 : List Nat := 21391 :: root388Tail34
def root388Tail32 : List Nat := 21319 :: root388Tail33
def root388Tail31 : List Nat := 20593 :: root388Tail32
def root388Tail30 : List Nat := 19309 :: root388Tail31
def root388Tail29 : List Nat := 19009 :: root388Tail30
def root388Tail28 : List Nat := 18379 :: root388Tail29
def root388Tail27 : List Nat := 17989 :: root388Tail28
def root388Tail26 : List Nat := 17041 :: root388Tail27
def root388Tail25 : List Nat := 16381 :: root388Tail26
def root388Tail24 : List Nat := 15919 :: root388Tail25
def root388Tail23 : List Nat := 14731 :: root388Tail24
def root388Tail22 : List Nat := 12907 :: root388Tail23
def root388Tail21 : List Nat := 12697 :: root388Tail22
def root388Tail20 : List Nat := 10891 :: root388Tail21
def root388Tail19 : List Nat := 10009 :: root388Tail20
def root388Tail18 : List Nat := 9643 :: root388Tail19
def root388Tail17 : List Nat := 9013 :: root388Tail18
def root388Tail16 : List Nat := 8761 :: root388Tail17
def root388Tail15 : List Nat := 8527 :: root388Tail16
def root388Tail14 : List Nat := 8317 :: root388Tail15
def root388Tail13 : List Nat := 8191 :: root388Tail14
def root388Tail12 : List Nat := 8161 :: root388Tail13
def root388Tail11 : List Nat := 8089 :: root388Tail12
def root388Tail10 : List Nat := 7477 :: root388Tail11
def root388Tail9 : List Nat := 6763 :: root388Tail10
def root388Tail8 : List Nat := 6679 :: root388Tail9
def root388Tail7 : List Nat := 6379 :: root388Tail8
def root388Tail6 : List Nat := 6367 :: root388Tail7
def root388Tail5 : List Nat := 6043 :: root388Tail6
def root388Tail4 : List Nat := 5347 :: root388Tail5
def root388Tail3 : List Nat := 4507 :: root388Tail4
def root388Tail2 : List Nat := 4423 :: root388Tail3
def root388Tail1 : List Nat := 3343 :: root388Tail2
def root388Tail0 : List Nat := 3271 :: root388Tail1
def root388Candidates0 : List Nat := [9013]
set_option maxRecDepth 8192 in
theorem root388Candidates0_checked : candidates edge 4 2689 3271 root388Tail1 winner.bound = root388Candidates0 := by
  unfold candidates
  have row : edge 3271 = (SortedNatTree.contains FastRows.row3271) := by
    change Data.row3271.contains = (SortedNatTree.contains FastRows.row3271)
    exact FastRows.row3271_checked.symm
  rw [row]
  decide
theorem root388Step0 : scan edge (search edge 3) 4 [2689] 2689 root388Tail0 winner =
    scan edge (search edge 3) 4 [2689] 2689 root388Tail1 winner := by
  rw [root388Tail0, scan, if_neg (by decide), root388Candidates0_checked]
  change scan edge (search edge 3) 4 [2689] 2689 root388Tail1
    (if 3 ≤ root388Candidates0.length then search edge 3 [2689, 3271] root388Candidates0 5960 winner else winner) = _
  rw [if_neg (by decide)]
def root388Candidates1 : List Nat := [6379]
set_option maxRecDepth 8192 in
theorem root388Candidates1_checked : candidates edge 4 2689 3343 root388Tail2 winner.bound = root388Candidates1 := by
  unfold candidates
  have row : edge 3343 = (SortedNatTree.contains FastRows.row3343) := by
    change Data.row3343.contains = (SortedNatTree.contains FastRows.row3343)
    exact FastRows.row3343_checked.symm
  rw [row]
  decide
theorem root388Step1 : scan edge (search edge 3) 4 [2689] 2689 root388Tail1 winner =
    scan edge (search edge 3) 4 [2689] 2689 root388Tail2 winner := by
  rw [root388Tail1, scan, if_neg (by decide), root388Candidates1_checked]
  change scan edge (search edge 3) 4 [2689] 2689 root388Tail2
    (if 3 ≤ root388Candidates1.length then search edge 3 [2689, 3343] root388Candidates1 6032 winner else winner) = _
  rw [if_neg (by decide)]
def root388Candidates2 : List Nat := [8317]
set_option maxRecDepth 8192 in
theorem root388Candidates2_checked : candidates edge 4 2689 4423 root388Tail3 winner.bound = root388Candidates2 := by
  unfold candidates
  have row : edge 4423 = (SortedNatTree.contains FastRows.row4423) := by
    change Data.row4423.contains = (SortedNatTree.contains FastRows.row4423)
    exact FastRows.row4423_checked.symm
  rw [row]
  decide
theorem root388Step2 : scan edge (search edge 3) 4 [2689] 2689 root388Tail2 winner =
    scan edge (search edge 3) 4 [2689] 2689 root388Tail3 winner := by
  rw [root388Tail2, scan, if_neg (by decide), root388Candidates2_checked]
  change scan edge (search edge 3) 4 [2689] 2689 root388Tail3
    (if 3 ≤ root388Candidates2.length then search edge 3 [2689, 4423] root388Candidates2 7112 winner else winner) = _
  rw [if_neg (by decide)]
def root388Candidates3 : List Nat := [12907, 15919]
set_option maxRecDepth 8192 in
theorem root388Candidates3_checked : candidates edge 4 2689 4507 root388Tail4 winner.bound = root388Candidates3 := by
  unfold candidates
  have row : edge 4507 = (SortedNatTree.contains FastRows.row4507) := by
    change Data.row4507.contains = (SortedNatTree.contains FastRows.row4507)
    exact FastRows.row4507_checked.symm
  rw [row]
  decide
theorem root388Step3 : scan edge (search edge 3) 4 [2689] 2689 root388Tail3 winner =
    scan edge (search edge 3) 4 [2689] 2689 root388Tail4 winner := by
  rw [root388Tail3, scan, if_neg (by decide), root388Candidates3_checked]
  change scan edge (search edge 3) 4 [2689] 2689 root388Tail4
    (if 3 ≤ root388Candidates3.length then search edge 3 [2689, 4507] root388Candidates3 7196 winner else winner) = _
  rw [if_neg (by decide)]
def root388Candidates4 : List Nat := [16381]
set_option maxRecDepth 8192 in
theorem root388Candidates4_checked : candidates edge 4 2689 5347 root388Tail5 winner.bound = root388Candidates4 := by
  unfold candidates
  have row : edge 5347 = (SortedNatTree.contains FastRows.row5347) := by
    change Data.row5347.contains = (SortedNatTree.contains FastRows.row5347)
    exact FastRows.row5347_checked.symm
  rw [row]
  decide
theorem root388Step4 : scan edge (search edge 3) 4 [2689] 2689 root388Tail4 winner =
    scan edge (search edge 3) 4 [2689] 2689 root388Tail5 winner := by
  rw [root388Tail4, scan, if_neg (by decide), root388Candidates4_checked]
  change scan edge (search edge 3) 4 [2689] 2689 root388Tail5
    (if 3 ≤ root388Candidates4.length then search edge 3 [2689, 5347] root388Candidates4 8036 winner else winner) = _
  rw [if_neg (by decide)]
theorem root388Scan5 : scan edge (search edge 3) 4 [2689] 2689 root388Tail5 winner = winner := by rw [root388Tail5, scan, if_pos (by decide)]
theorem root388Scan4 : scan edge (search edge 3) 4 [2689] 2689 root388Tail4 winner = winner :=
  (root388Step4).trans root388Scan5
theorem root388Scan3 : scan edge (search edge 3) 4 [2689] 2689 root388Tail3 winner = winner :=
  (root388Step3).trans root388Scan4
theorem root388Scan2 : scan edge (search edge 3) 4 [2689] 2689 root388Tail2 winner = winner :=
  (root388Step2).trans root388Scan3
theorem root388Scan1 : scan edge (search edge 3) 4 [2689] 2689 root388Tail1 winner = winner :=
  (root388Step1).trans root388Scan2
theorem root388Scan0 : scan edge (search edge 3) 4 [2689] 2689 root388Tail0 winner = winner :=
  (root388Step0).trans root388Scan1
set_option maxRecDepth 4096 in
theorem root388Branch : search edge 4 [2689] root388Candidates 2689 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2689] 2689 root388Tail0 winner = winner
  exact root388Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
