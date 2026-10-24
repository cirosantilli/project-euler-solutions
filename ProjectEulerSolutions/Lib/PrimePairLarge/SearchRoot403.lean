import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root403Candidates : List Nat := Data.row2791
set_option maxRecDepth 8192 in
theorem root403Candidates_checked : candidates edge 5 0 2791 tail404 winner.bound = root403Candidates := by
  apply candidates_eq_row edge 5 0 2791 tail404 root403Candidates winner.bound tail404_ordered
  · decide
  · decide
  · funext q
    rfl
def root403Tail37 : List Nat := []
def root403Tail36 : List Nat := 21961 :: root403Tail37
def root403Tail35 : List Nat := 21943 :: root403Tail36
def root403Tail34 : List Nat := 20047 :: root403Tail35
def root403Tail33 : List Nat := 19207 :: root403Tail34
def root403Tail32 : List Nat := 18523 :: root403Tail33
def root403Tail31 : List Nat := 18313 :: root403Tail32
def root403Tail30 : List Nat := 18097 :: root403Tail31
def root403Tail29 : List Nat := 17863 :: root403Tail30
def root403Tail28 : List Nat := 16843 :: root403Tail29
def root403Tail27 : List Nat := 16759 :: root403Tail28
def root403Tail26 : List Nat := 15619 :: root403Tail27
def root403Tail25 : List Nat := 15541 :: root403Tail26
def root403Tail24 : List Nat := 12391 :: root403Tail25
def root403Tail23 : List Nat := 12073 :: root403Tail24
def root403Tail22 : List Nat := 11467 :: root403Tail23
def root403Tail21 : List Nat := 11353 :: root403Tail22
def root403Tail20 : List Nat := 10789 :: root403Tail21
def root403Tail19 : List Nat := 10333 :: root403Tail20
def root403Tail18 : List Nat := 10177 :: root403Tail19
def root403Tail17 : List Nat := 9649 :: root403Tail18
def root403Tail16 : List Nat := 9511 :: root403Tail17
def root403Tail15 : List Nat := 8647 :: root403Tail16
def root403Tail14 : List Nat := 8287 :: root403Tail15
def root403Tail13 : List Nat := 8179 :: root403Tail14
def root403Tail12 : List Nat := 7489 :: root403Tail13
def root403Tail11 : List Nat := 7369 :: root403Tail12
def root403Tail10 : List Nat := 6961 :: root403Tail11
def root403Tail9 : List Nat := 6883 :: root403Tail10
def root403Tail8 : List Nat := 6607 :: root403Tail9
def root403Tail7 : List Nat := 6379 :: root403Tail8
def root403Tail6 : List Nat := 5281 :: root403Tail7
def root403Tail5 : List Nat := 5179 :: root403Tail6
def root403Tail4 : List Nat := 5113 :: root403Tail5
def root403Tail3 : List Nat := 4297 :: root403Tail4
def root403Tail2 : List Nat := 3967 :: root403Tail3
def root403Tail1 : List Nat := 3499 :: root403Tail2
def root403Tail0 : List Nat := 3001 :: root403Tail1
def root403Candidates0 : List Nat := [3499, 3967, 6607, 8647]
set_option maxRecDepth 8192 in
theorem root403Candidates0_checked : candidates edge 4 2791 3001 root403Tail1 winner.bound = root403Candidates0 := by
  unfold candidates
  have row : edge 3001 = (SortedNatTree.contains FastRows.row3001) := by
    change Data.row3001.contains = (SortedNatTree.contains FastRows.row3001)
    exact FastRows.row3001_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root403Child0 : search edge 3 [2791, 3001] root403Candidates0 5792 winner = winner := by decide
theorem root403Step0 : scan edge (search edge 3) 4 [2791] 2791 root403Tail0 winner =
    scan edge (search edge 3) 4 [2791] 2791 root403Tail1 winner := by
  rw [root403Tail0, scan, if_neg (by decide), root403Candidates0_checked]
  change scan edge (search edge 3) 4 [2791] 2791 root403Tail1
    (if 3 ≤ root403Candidates0.length then search edge 3 [2791, 3001] root403Candidates0 5792 winner else winner) = _
  rw [if_pos (by decide), root403Child0]
def root403Candidates1 : List Nat := [10333]
set_option maxRecDepth 8192 in
theorem root403Candidates1_checked : candidates edge 4 2791 3499 root403Tail2 winner.bound = root403Candidates1 := by
  unfold candidates
  have row : edge 3499 = (SortedNatTree.contains FastRows.row3499) := by
    change Data.row3499.contains = (SortedNatTree.contains FastRows.row3499)
    exact FastRows.row3499_checked.symm
  rw [row]
  decide
theorem root403Step1 : scan edge (search edge 3) 4 [2791] 2791 root403Tail1 winner =
    scan edge (search edge 3) 4 [2791] 2791 root403Tail2 winner := by
  rw [root403Tail1, scan, if_neg (by decide), root403Candidates1_checked]
  change scan edge (search edge 3) 4 [2791] 2791 root403Tail2
    (if 3 ≤ root403Candidates1.length then search edge 3 [2791, 3499] root403Candidates1 6290 winner else winner) = _
  rw [if_neg (by decide)]
def root403Candidates2 : List Nat := [11467, 18313]
set_option maxRecDepth 8192 in
theorem root403Candidates2_checked : candidates edge 4 2791 3967 root403Tail3 winner.bound = root403Candidates2 := by
  unfold candidates
  have row : edge 3967 = (SortedNatTree.contains FastRows.row3967) := by
    change Data.row3967.contains = (SortedNatTree.contains FastRows.row3967)
    exact FastRows.row3967_checked.symm
  rw [row]
  decide
theorem root403Step2 : scan edge (search edge 3) 4 [2791] 2791 root403Tail2 winner =
    scan edge (search edge 3) 4 [2791] 2791 root403Tail3 winner := by
  rw [root403Tail2, scan, if_neg (by decide), root403Candidates2_checked]
  change scan edge (search edge 3) 4 [2791] 2791 root403Tail3
    (if 3 ≤ root403Candidates2.length then search edge 3 [2791, 3967] root403Candidates2 6758 winner else winner) = _
  rw [if_neg (by decide)]
def root403Candidates3 : List Nat := [6607]
set_option maxRecDepth 8192 in
theorem root403Candidates3_checked : candidates edge 4 2791 4297 root403Tail4 winner.bound = root403Candidates3 := by
  unfold candidates
  have row : edge 4297 = (SortedNatTree.contains FastRows.row4297) := by
    change Data.row4297.contains = (SortedNatTree.contains FastRows.row4297)
    exact FastRows.row4297_checked.symm
  rw [row]
  decide
theorem root403Step3 : scan edge (search edge 3) 4 [2791] 2791 root403Tail3 winner =
    scan edge (search edge 3) 4 [2791] 2791 root403Tail4 winner := by
  rw [root403Tail3, scan, if_neg (by decide), root403Candidates3_checked]
  change scan edge (search edge 3) 4 [2791] 2791 root403Tail4
    (if 3 ≤ root403Candidates3.length then search edge 3 [2791, 4297] root403Candidates3 7088 winner else winner) = _
  rw [if_neg (by decide)]
def root403Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root403Candidates4_checked : candidates edge 4 2791 5113 root403Tail5 winner.bound = root403Candidates4 := by
  unfold candidates
  have row : edge 5113 = (SortedNatTree.contains FastRows.row5113) := by
    change Data.row5113.contains = (SortedNatTree.contains FastRows.row5113)
    exact FastRows.row5113_checked.symm
  rw [row]
  decide
theorem root403Step4 : scan edge (search edge 3) 4 [2791] 2791 root403Tail4 winner =
    scan edge (search edge 3) 4 [2791] 2791 root403Tail5 winner := by
  rw [root403Tail4, scan, if_neg (by decide), root403Candidates4_checked]
  change scan edge (search edge 3) 4 [2791] 2791 root403Tail5
    (if 3 ≤ root403Candidates4.length then search edge 3 [2791, 5113] root403Candidates4 7904 winner else winner) = _
  rw [if_neg (by decide)]
def root403Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root403Candidates5_checked : candidates edge 4 2791 5179 root403Tail6 winner.bound = root403Candidates5 := by
  unfold candidates
  have row : edge 5179 = (SortedNatTree.contains FastRows.row5179) := by
    change Data.row5179.contains = (SortedNatTree.contains FastRows.row5179)
    exact FastRows.row5179_checked.symm
  rw [row]
  decide
theorem root403Step5 : scan edge (search edge 3) 4 [2791] 2791 root403Tail5 winner =
    scan edge (search edge 3) 4 [2791] 2791 root403Tail6 winner := by
  rw [root403Tail5, scan, if_neg (by decide), root403Candidates5_checked]
  change scan edge (search edge 3) 4 [2791] 2791 root403Tail6
    (if 3 ≤ root403Candidates5.length then search edge 3 [2791, 5179] root403Candidates5 7970 winner else winner) = _
  rw [if_neg (by decide)]
def root403Candidates6 : List Nat := [7489, 8287]
set_option maxRecDepth 8192 in
theorem root403Candidates6_checked : candidates edge 4 2791 5281 root403Tail7 winner.bound = root403Candidates6 := by
  unfold candidates
  have row : edge 5281 = (SortedNatTree.contains FastRows.row5281) := by
    change Data.row5281.contains = (SortedNatTree.contains FastRows.row5281)
    exact FastRows.row5281_checked.symm
  rw [row]
  decide
theorem root403Step6 : scan edge (search edge 3) 4 [2791] 2791 root403Tail6 winner =
    scan edge (search edge 3) 4 [2791] 2791 root403Tail7 winner := by
  rw [root403Tail6, scan, if_neg (by decide), root403Candidates6_checked]
  change scan edge (search edge 3) 4 [2791] 2791 root403Tail7
    (if 3 ≤ root403Candidates6.length then search edge 3 [2791, 5281] root403Candidates6 8072 winner else winner) = _
  rw [if_neg (by decide)]
theorem root403Scan7 : scan edge (search edge 3) 4 [2791] 2791 root403Tail7 winner = winner := by rw [root403Tail7, scan, if_pos (by decide)]
theorem root403Scan6 : scan edge (search edge 3) 4 [2791] 2791 root403Tail6 winner = winner :=
  (root403Step6).trans root403Scan7
theorem root403Scan5 : scan edge (search edge 3) 4 [2791] 2791 root403Tail5 winner = winner :=
  (root403Step5).trans root403Scan6
theorem root403Scan4 : scan edge (search edge 3) 4 [2791] 2791 root403Tail4 winner = winner :=
  (root403Step4).trans root403Scan5
theorem root403Scan3 : scan edge (search edge 3) 4 [2791] 2791 root403Tail3 winner = winner :=
  (root403Step3).trans root403Scan4
theorem root403Scan2 : scan edge (search edge 3) 4 [2791] 2791 root403Tail2 winner = winner :=
  (root403Step2).trans root403Scan3
theorem root403Scan1 : scan edge (search edge 3) 4 [2791] 2791 root403Tail1 winner = winner :=
  (root403Step1).trans root403Scan2
theorem root403Scan0 : scan edge (search edge 3) 4 [2791] 2791 root403Tail0 winner = winner :=
  (root403Step0).trans root403Scan1
set_option maxRecDepth 4096 in
theorem root403Branch : search edge 4 [2791] root403Candidates 2791 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2791] 2791 root403Tail0 winner = winner
  exact root403Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
