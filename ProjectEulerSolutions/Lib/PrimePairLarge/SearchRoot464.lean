import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root464Candidates : List Nat := Data.row3319
set_option maxRecDepth 8192 in
theorem root464Candidates_checked : candidates edge 5 0 3319 tail465 winner.bound = root464Candidates := by
  apply candidates_eq_row edge 5 0 3319 tail465 root464Candidates winner.bound tail465_ordered
  · decide
  · decide
  · funext q
    rfl
def root464Tail38 : List Nat := []
def root464Tail37 : List Nat := 21727 :: root464Tail38
def root464Tail36 : List Nat := 21529 :: root464Tail37
def root464Tail35 : List Nat := 21157 :: root464Tail36
def root464Tail34 : List Nat := 20509 :: root464Tail35
def root464Tail33 : List Nat := 19069 :: root464Tail34
def root464Tail32 : List Nat := 18973 :: root464Tail33
def root464Tail31 : List Nat := 17791 :: root464Tail32
def root464Tail30 : List Nat := 17107 :: root464Tail31
def root464Tail29 : List Nat := 16561 :: root464Tail30
def root464Tail28 : List Nat := 15373 :: root464Tail29
def root464Tail27 : List Nat := 15217 :: root464Tail28
def root464Tail26 : List Nat := 14683 :: root464Tail27
def root464Tail25 : List Nat := 13723 :: root464Tail26
def root464Tail24 : List Nat := 13399 :: root464Tail25
def root464Tail23 : List Nat := 13297 :: root464Tail24
def root464Tail22 : List Nat := 12799 :: root464Tail23
def root464Tail21 : List Nat := 12289 :: root464Tail22
def root464Tail20 : List Nat := 11731 :: root464Tail21
def root464Tail19 : List Nat := 11617 :: root464Tail20
def root464Tail18 : List Nat := 10903 :: root464Tail19
def root464Tail17 : List Nat := 9871 :: root464Tail18
def root464Tail16 : List Nat := 9391 :: root464Tail17
def root464Tail15 : List Nat := 9151 :: root464Tail16
def root464Tail14 : List Nat := 8647 :: root464Tail15
def root464Tail13 : List Nat := 8101 :: root464Tail14
def root464Tail12 : List Nat := 7951 :: root464Tail13
def root464Tail11 : List Nat := 7669 :: root464Tail12
def root464Tail10 : List Nat := 7321 :: root464Tail11
def root464Tail9 : List Nat := 6577 :: root464Tail10
def root464Tail8 : List Nat := 6481 :: root464Tail9
def root464Tail7 : List Nat := 6367 :: root464Tail8
def root464Tail6 : List Nat := 5419 :: root464Tail7
def root464Tail5 : List Nat := 5209 :: root464Tail6
def root464Tail4 : List Nat := 4507 :: root464Tail5
def root464Tail3 : List Nat := 4363 :: root464Tail4
def root464Tail2 : List Nat := 4339 :: root464Tail3
def root464Tail1 : List Nat := 3613 :: root464Tail2
def root464Tail0 : List Nat := 3373 :: root464Tail1
def root464Candidates0 : List Nat := [4339]
set_option maxRecDepth 8192 in
theorem root464Candidates0_checked : candidates edge 4 3319 3373 root464Tail1 winner.bound = root464Candidates0 := by
  unfold candidates
  have row : edge 3373 = (SortedNatTree.contains FastRows.row3373) := by
    change Data.row3373.contains = (SortedNatTree.contains FastRows.row3373)
    exact FastRows.row3373_checked.symm
  rw [row]
  decide
theorem root464Step0 : scan edge (search edge 3) 4 [3319] 3319 root464Tail0 winner =
    scan edge (search edge 3) 4 [3319] 3319 root464Tail1 winner := by
  rw [root464Tail0, scan, if_neg (by decide), root464Candidates0_checked]
  change scan edge (search edge 3) 4 [3319] 3319 root464Tail1
    (if 3 ≤ root464Candidates0.length then search edge 3 [3319, 3373] root464Candidates0 6692 winner else winner) = _
  rw [if_neg (by decide)]
def root464Candidates1 : List Nat := [9151, 17791, 18973]
set_option maxRecDepth 8192 in
theorem root464Candidates1_checked : candidates edge 4 3319 3613 root464Tail2 winner.bound = root464Candidates1 := by
  unfold candidates
  have row : edge 3613 = (SortedNatTree.contains FastRows.row3613) := by
    change Data.row3613.contains = (SortedNatTree.contains FastRows.row3613)
    exact FastRows.row3613_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root464Child1 : search edge 3 [3319, 3613] root464Candidates1 6932 winner = winner := by decide
theorem root464Step1 : scan edge (search edge 3) 4 [3319] 3319 root464Tail1 winner =
    scan edge (search edge 3) 4 [3319] 3319 root464Tail2 winner := by
  rw [root464Tail1, scan, if_neg (by decide), root464Candidates1_checked]
  change scan edge (search edge 3) 4 [3319] 3319 root464Tail2
    (if 3 ≤ root464Candidates1.length then search edge 3 [3319, 3613] root464Candidates1 6932 winner else winner) = _
  rw [if_pos (by decide), root464Child1]
def root464Candidates2 : List Nat := [10903]
set_option maxRecDepth 8192 in
theorem root464Candidates2_checked : candidates edge 4 3319 4339 root464Tail3 winner.bound = root464Candidates2 := by
  unfold candidates
  have row : edge 4339 = (SortedNatTree.contains FastRows.row4339) := by
    change Data.row4339.contains = (SortedNatTree.contains FastRows.row4339)
    exact FastRows.row4339_checked.symm
  rw [row]
  decide
theorem root464Step2 : scan edge (search edge 3) 4 [3319] 3319 root464Tail2 winner =
    scan edge (search edge 3) 4 [3319] 3319 root464Tail3 winner := by
  rw [root464Tail2, scan, if_neg (by decide), root464Candidates2_checked]
  change scan edge (search edge 3) 4 [3319] 3319 root464Tail3
    (if 3 ≤ root464Candidates2.length then search edge 3 [3319, 4339] root464Candidates2 7658 winner else winner) = _
  rw [if_neg (by decide)]
def root464Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root464Candidates3_checked : candidates edge 4 3319 4363 root464Tail4 winner.bound = root464Candidates3 := by
  unfold candidates
  have row : edge 4363 = (SortedNatTree.contains FastRows.row4363) := by
    change Data.row4363.contains = (SortedNatTree.contains FastRows.row4363)
    exact FastRows.row4363_checked.symm
  rw [row]
  decide
theorem root464Step3 : scan edge (search edge 3) 4 [3319] 3319 root464Tail3 winner =
    scan edge (search edge 3) 4 [3319] 3319 root464Tail4 winner := by
  rw [root464Tail3, scan, if_neg (by decide), root464Candidates3_checked]
  change scan edge (search edge 3) 4 [3319] 3319 root464Tail4
    (if 3 ≤ root464Candidates3.length then search edge 3 [3319, 4363] root464Candidates3 7682 winner else winner) = _
  rw [if_neg (by decide)]
def root464Candidates4 : List Nat := [5419, 15217]
set_option maxRecDepth 8192 in
theorem root464Candidates4_checked : candidates edge 4 3319 4507 root464Tail5 winner.bound = root464Candidates4 := by
  unfold candidates
  have row : edge 4507 = (SortedNatTree.contains FastRows.row4507) := by
    change Data.row4507.contains = (SortedNatTree.contains FastRows.row4507)
    exact FastRows.row4507_checked.symm
  rw [row]
  decide
theorem root464Step4 : scan edge (search edge 3) 4 [3319] 3319 root464Tail4 winner =
    scan edge (search edge 3) 4 [3319] 3319 root464Tail5 winner := by
  rw [root464Tail4, scan, if_neg (by decide), root464Candidates4_checked]
  change scan edge (search edge 3) 4 [3319] 3319 root464Tail5
    (if 3 ≤ root464Candidates4.length then search edge 3 [3319, 4507] root464Candidates4 7826 winner else winner) = _
  rw [if_neg (by decide)]
def root464Candidates5 : List Nat := [5419]
set_option maxRecDepth 8192 in
theorem root464Candidates5_checked : candidates edge 4 3319 5209 root464Tail6 winner.bound = root464Candidates5 := by
  unfold candidates
  have row : edge 5209 = (SortedNatTree.contains FastRows.row5209) := by
    change Data.row5209.contains = (SortedNatTree.contains FastRows.row5209)
    exact FastRows.row5209_checked.symm
  rw [row]
  decide
theorem root464Step5 : scan edge (search edge 3) 4 [3319] 3319 root464Tail5 winner =
    scan edge (search edge 3) 4 [3319] 3319 root464Tail6 winner := by
  rw [root464Tail5, scan, if_neg (by decide), root464Candidates5_checked]
  change scan edge (search edge 3) 4 [3319] 3319 root464Tail6
    (if 3 ≤ root464Candidates5.length then search edge 3 [3319, 5209] root464Candidates5 8528 winner else winner) = _
  rw [if_neg (by decide)]
def root464Candidates6 : List Nat := [13399, 14683]
set_option maxRecDepth 8192 in
theorem root464Candidates6_checked : candidates edge 4 3319 5419 root464Tail7 winner.bound = root464Candidates6 := by
  unfold candidates
  have row : edge 5419 = (SortedNatTree.contains FastRows.row5419) := by
    change Data.row5419.contains = (SortedNatTree.contains FastRows.row5419)
    exact FastRows.row5419_checked.symm
  rw [row]
  decide
theorem root464Step6 : scan edge (search edge 3) 4 [3319] 3319 root464Tail6 winner =
    scan edge (search edge 3) 4 [3319] 3319 root464Tail7 winner := by
  rw [root464Tail6, scan, if_neg (by decide), root464Candidates6_checked]
  change scan edge (search edge 3) 4 [3319] 3319 root464Tail7
    (if 3 ≤ root464Candidates6.length then search edge 3 [3319, 5419] root464Candidates6 8738 winner else winner) = _
  rw [if_neg (by decide)]
theorem root464Scan7 : scan edge (search edge 3) 4 [3319] 3319 root464Tail7 winner = winner := by rw [root464Tail7, scan, if_pos (by decide)]
theorem root464Scan6 : scan edge (search edge 3) 4 [3319] 3319 root464Tail6 winner = winner :=
  (root464Step6).trans root464Scan7
theorem root464Scan5 : scan edge (search edge 3) 4 [3319] 3319 root464Tail5 winner = winner :=
  (root464Step5).trans root464Scan6
theorem root464Scan4 : scan edge (search edge 3) 4 [3319] 3319 root464Tail4 winner = winner :=
  (root464Step4).trans root464Scan5
theorem root464Scan3 : scan edge (search edge 3) 4 [3319] 3319 root464Tail3 winner = winner :=
  (root464Step3).trans root464Scan4
theorem root464Scan2 : scan edge (search edge 3) 4 [3319] 3319 root464Tail2 winner = winner :=
  (root464Step2).trans root464Scan3
theorem root464Scan1 : scan edge (search edge 3) 4 [3319] 3319 root464Tail1 winner = winner :=
  (root464Step1).trans root464Scan2
theorem root464Scan0 : scan edge (search edge 3) 4 [3319] 3319 root464Tail0 winner = winner :=
  (root464Step0).trans root464Scan1
set_option maxRecDepth 4096 in
theorem root464Branch : search edge 4 [3319] root464Candidates 3319 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3319] 3319 root464Tail0 winner = winner
  exact root464Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
