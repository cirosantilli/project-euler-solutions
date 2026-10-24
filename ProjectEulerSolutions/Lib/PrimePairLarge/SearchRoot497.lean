import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root497Candidates : List Nat := Data.row3571
set_option maxRecDepth 8192 in
theorem root497Candidates_checked : candidates edge 5 0 3571 tail498 winner.bound = root497Candidates := by
  apply candidates_eq_row edge 5 0 3571 tail498 root497Candidates winner.bound tail498_ordered
  · decide
  · decide
  · funext q
    rfl
def root497Tail39 : List Nat := []
def root497Tail38 : List Nat := 22159 :: root497Tail39
def root497Tail37 : List Nat := 22063 :: root497Tail38
def root497Tail36 : List Nat := 21871 :: root497Tail37
def root497Tail35 : List Nat := 21517 :: root497Tail36
def root497Tail34 : List Nat := 21211 :: root497Tail35
def root497Tail33 : List Nat := 20773 :: root497Tail34
def root497Tail32 : List Nat := 20359 :: root497Tail33
def root497Tail31 : List Nat := 20107 :: root497Tail32
def root497Tail30 : List Nat := 19753 :: root497Tail31
def root497Tail29 : List Nat := 19699 :: root497Tail30
def root497Tail28 : List Nat := 19501 :: root497Tail29
def root497Tail27 : List Nat := 19387 :: root497Tail28
def root497Tail26 : List Nat := 18127 :: root497Tail27
def root497Tail25 : List Nat := 17581 :: root497Tail26
def root497Tail24 : List Nat := 17203 :: root497Tail25
def root497Tail23 : List Nat := 15907 :: root497Tail24
def root497Tail22 : List Nat := 14503 :: root497Tail23
def root497Tail21 : List Nat := 14449 :: root497Tail22
def root497Tail20 : List Nat := 14143 :: root497Tail21
def root497Tail19 : List Nat := 13327 :: root497Tail20
def root497Tail18 : List Nat := 13147 :: root497Tail19
def root497Tail17 : List Nat := 13063 :: root497Tail18
def root497Tail16 : List Nat := 12289 :: root497Tail17
def root497Tail15 : List Nat := 11959 :: root497Tail16
def root497Tail14 : List Nat := 11329 :: root497Tail15
def root497Tail13 : List Nat := 9181 :: root497Tail14
def root497Tail12 : List Nat := 9067 :: root497Tail13
def root497Tail11 : List Nat := 9001 :: root497Tail12
def root497Tail10 : List Nat := 7963 :: root497Tail11
def root497Tail9 : List Nat := 7699 :: root497Tail10
def root497Tail8 : List Nat := 6883 :: root497Tail9
def root497Tail7 : List Nat := 6451 :: root497Tail8
def root497Tail6 : List Nat := 6007 :: root497Tail7
def root497Tail5 : List Nat := 5023 :: root497Tail6
def root497Tail4 : List Nat := 4969 :: root497Tail5
def root497Tail3 : List Nat := 4957 :: root497Tail4
def root497Tail2 : List Nat := 4801 :: root497Tail3
def root497Tail1 : List Nat := 4363 :: root497Tail2
def root497Tail0 : List Nat := 4111 :: root497Tail1
def root497Candidates0 : List Nat := [9181]
set_option maxRecDepth 8192 in
theorem root497Candidates0_checked : candidates edge 4 3571 4111 root497Tail1 winner.bound = root497Candidates0 := by
  unfold candidates
  have row : edge 4111 = (SortedNatTree.contains FastRows.row4111) := by
    change Data.row4111.contains = (SortedNatTree.contains FastRows.row4111)
    exact FastRows.row4111_checked.symm
  rw [row]
  decide
theorem root497Step0 : scan edge (search edge 3) 4 [3571] 3571 root497Tail0 winner =
    scan edge (search edge 3) 4 [3571] 3571 root497Tail1 winner := by
  rw [root497Tail0, scan, if_neg (by decide), root497Candidates0_checked]
  change scan edge (search edge 3) 4 [3571] 3571 root497Tail1
    (if 3 ≤ root497Candidates0.length then search edge 3 [3571, 4111] root497Candidates0 7682 winner else winner) = _
  rw [if_neg (by decide)]
def root497Candidates1 : List Nat := [13063, 13147]
set_option maxRecDepth 8192 in
theorem root497Candidates1_checked : candidates edge 4 3571 4363 root497Tail2 winner.bound = root497Candidates1 := by
  unfold candidates
  have row : edge 4363 = (SortedNatTree.contains FastRows.row4363) := by
    change Data.row4363.contains = (SortedNatTree.contains FastRows.row4363)
    exact FastRows.row4363_checked.symm
  rw [row]
  decide
theorem root497Step1 : scan edge (search edge 3) 4 [3571] 3571 root497Tail1 winner =
    scan edge (search edge 3) 4 [3571] 3571 root497Tail2 winner := by
  rw [root497Tail1, scan, if_neg (by decide), root497Candidates1_checked]
  change scan edge (search edge 3) 4 [3571] 3571 root497Tail2
    (if 3 ≤ root497Candidates1.length then search edge 3 [3571, 4363] root497Candidates1 7934 winner else winner) = _
  rw [if_neg (by decide)]
def root497Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root497Candidates2_checked : candidates edge 4 3571 4801 root497Tail3 winner.bound = root497Candidates2 := by
  unfold candidates
  have row : edge 4801 = (SortedNatTree.contains FastRows.row4801) := by
    change Data.row4801.contains = (SortedNatTree.contains FastRows.row4801)
    exact FastRows.row4801_checked.symm
  rw [row]
  decide
theorem root497Step2 : scan edge (search edge 3) 4 [3571] 3571 root497Tail2 winner =
    scan edge (search edge 3) 4 [3571] 3571 root497Tail3 winner := by
  rw [root497Tail2, scan, if_neg (by decide), root497Candidates2_checked]
  change scan edge (search edge 3) 4 [3571] 3571 root497Tail3
    (if 3 ≤ root497Candidates2.length then search edge 3 [3571, 4801] root497Candidates2 8372 winner else winner) = _
  rw [if_neg (by decide)]
def root497Candidates3 : List Nat := [5023, 17203]
set_option maxRecDepth 8192 in
theorem root497Candidates3_checked : candidates edge 4 3571 4957 root497Tail4 winner.bound = root497Candidates3 := by
  unfold candidates
  have row : edge 4957 = (SortedNatTree.contains FastRows.row4957) := by
    change Data.row4957.contains = (SortedNatTree.contains FastRows.row4957)
    exact FastRows.row4957_checked.symm
  rw [row]
  decide
theorem root497Step3 : scan edge (search edge 3) 4 [3571] 3571 root497Tail3 winner =
    scan edge (search edge 3) 4 [3571] 3571 root497Tail4 winner := by
  rw [root497Tail3, scan, if_neg (by decide), root497Candidates3_checked]
  change scan edge (search edge 3) 4 [3571] 3571 root497Tail4
    (if 3 ≤ root497Candidates3.length then search edge 3 [3571, 4957] root497Candidates3 8528 winner else winner) = _
  rw [if_neg (by decide)]
def root497Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root497Candidates4_checked : candidates edge 4 3571 4969 root497Tail5 winner.bound = root497Candidates4 := by
  unfold candidates
  have row : edge 4969 = (SortedNatTree.contains FastRows.row4969) := by
    change Data.row4969.contains = (SortedNatTree.contains FastRows.row4969)
    exact FastRows.row4969_checked.symm
  rw [row]
  decide
theorem root497Step4 : scan edge (search edge 3) 4 [3571] 3571 root497Tail4 winner =
    scan edge (search edge 3) 4 [3571] 3571 root497Tail5 winner := by
  rw [root497Tail4, scan, if_neg (by decide), root497Candidates4_checked]
  change scan edge (search edge 3) 4 [3571] 3571 root497Tail5
    (if 3 ≤ root497Candidates4.length then search edge 3 [3571, 4969] root497Candidates4 8540 winner else winner) = _
  rw [if_neg (by decide)]
def root497Candidates5 : List Nat := [11959]
set_option maxRecDepth 8192 in
theorem root497Candidates5_checked : candidates edge 4 3571 5023 root497Tail6 winner.bound = root497Candidates5 := by
  unfold candidates
  have row : edge 5023 = (SortedNatTree.contains FastRows.row5023) := by
    change Data.row5023.contains = (SortedNatTree.contains FastRows.row5023)
    exact FastRows.row5023_checked.symm
  rw [row]
  decide
theorem root497Step5 : scan edge (search edge 3) 4 [3571] 3571 root497Tail5 winner =
    scan edge (search edge 3) 4 [3571] 3571 root497Tail6 winner := by
  rw [root497Tail5, scan, if_neg (by decide), root497Candidates5_checked]
  change scan edge (search edge 3) 4 [3571] 3571 root497Tail6
    (if 3 ≤ root497Candidates5.length then search edge 3 [3571, 5023] root497Candidates5 8594 winner else winner) = _
  rw [if_neg (by decide)]
theorem root497Scan6 : scan edge (search edge 3) 4 [3571] 3571 root497Tail6 winner = winner := by rw [root497Tail6, scan, if_pos (by decide)]
theorem root497Scan5 : scan edge (search edge 3) 4 [3571] 3571 root497Tail5 winner = winner :=
  (root497Step5).trans root497Scan6
theorem root497Scan4 : scan edge (search edge 3) 4 [3571] 3571 root497Tail4 winner = winner :=
  (root497Step4).trans root497Scan5
theorem root497Scan3 : scan edge (search edge 3) 4 [3571] 3571 root497Tail3 winner = winner :=
  (root497Step3).trans root497Scan4
theorem root497Scan2 : scan edge (search edge 3) 4 [3571] 3571 root497Tail2 winner = winner :=
  (root497Step2).trans root497Scan3
theorem root497Scan1 : scan edge (search edge 3) 4 [3571] 3571 root497Tail1 winner = winner :=
  (root497Step1).trans root497Scan2
theorem root497Scan0 : scan edge (search edge 3) 4 [3571] 3571 root497Tail0 winner = winner :=
  (root497Step0).trans root497Scan1
set_option maxRecDepth 4096 in
theorem root497Branch : search edge 4 [3571] root497Candidates 3571 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3571] 3571 root497Tail0 winner = winner
  exact root497Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
