import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root436Candidates : List Nat := Data.row3067
set_option maxRecDepth 8192 in
theorem root436Candidates_checked : candidates edge 5 0 3067 tail437 winner.bound = root436Candidates := by
  apply candidates_eq_row edge 5 0 3067 tail437 root436Candidates winner.bound tail437_ordered
  · decide
  · decide
  · funext q
    rfl
def root436Tail39 : List Nat := []
def root436Tail38 : List Nat := 22441 :: root436Tail39
def root436Tail37 : List Nat := 22303 :: root436Tail38
def root436Tail36 : List Nat := 21787 :: root436Tail37
def root436Tail35 : List Nat := 21757 :: root436Tail36
def root436Tail34 : List Nat := 21517 :: root436Tail35
def root436Tail33 : List Nat := 21391 :: root436Tail34
def root436Tail32 : List Nat := 20551 :: root436Tail33
def root436Tail31 : List Nat := 20509 :: root436Tail32
def root436Tail30 : List Nat := 17683 :: root436Tail31
def root436Tail29 : List Nat := 17053 :: root436Tail30
def root436Tail28 : List Nat := 15901 :: root436Tail29
def root436Tail27 : List Nat := 15667 :: root436Tail28
def root436Tail26 : List Nat := 15439 :: root436Tail27
def root436Tail25 : List Nat := 12823 :: root436Tail26
def root436Tail24 : List Nat := 12547 :: root436Tail25
def root436Tail23 : List Nat := 11953 :: root436Tail24
def root436Tail22 : List Nat := 11941 :: root436Tail23
def root436Tail21 : List Nat := 11491 :: root436Tail22
def root436Tail20 : List Nat := 10909 :: root436Tail21
def root436Tail19 : List Nat := 10567 :: root436Tail20
def root436Tail18 : List Nat := 10399 :: root436Tail19
def root436Tail17 : List Nat := 10069 :: root436Tail18
def root436Tail16 : List Nat := 9829 :: root436Tail17
def root436Tail15 : List Nat := 9787 :: root436Tail16
def root436Tail14 : List Nat := 9127 :: root436Tail15
def root436Tail13 : List Nat := 9001 :: root436Tail14
def root436Tail12 : List Nat := 8929 :: root436Tail13
def root436Tail11 : List Nat := 8269 :: root436Tail12
def root436Tail10 : List Nat := 7753 :: root436Tail11
def root436Tail9 : List Nat := 7687 :: root436Tail10
def root436Tail8 : List Nat := 7489 :: root436Tail9
def root436Tail7 : List Nat := 7459 :: root436Tail8
def root436Tail6 : List Nat := 6661 :: root436Tail7
def root436Tail5 : List Nat := 5737 :: root436Tail6
def root436Tail4 : List Nat := 5419 :: root436Tail5
def root436Tail3 : List Nat := 5413 :: root436Tail4
def root436Tail2 : List Nat := 5167 :: root436Tail3
def root436Tail1 : List Nat := 4909 :: root436Tail2
def root436Tail0 : List Nat := 3373 :: root436Tail1
def root436Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root436Candidates0_checked : candidates edge 4 3067 3373 root436Tail1 winner.bound = root436Candidates0 := by
  unfold candidates
  have row : edge 3373 = (SortedNatTree.contains FastRows.row3373) := by
    change Data.row3373.contains = (SortedNatTree.contains FastRows.row3373)
    exact FastRows.row3373_checked.symm
  rw [row]
  decide
theorem root436Step0 : scan edge (search edge 3) 4 [3067] 3067 root436Tail0 winner =
    scan edge (search edge 3) 4 [3067] 3067 root436Tail1 winner := by
  rw [root436Tail0, scan, if_neg (by decide), root436Candidates0_checked]
  change scan edge (search edge 3) 4 [3067] 3067 root436Tail1
    (if 3 ≤ root436Candidates0.length then search edge 3 [3067, 3373] root436Candidates0 6440 winner else winner) = _
  rw [if_neg (by decide)]
def root436Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root436Candidates1_checked : candidates edge 4 3067 4909 root436Tail2 winner.bound = root436Candidates1 := by
  unfold candidates
  have row : edge 4909 = (SortedNatTree.contains FastRows.row4909) := by
    change Data.row4909.contains = (SortedNatTree.contains FastRows.row4909)
    exact FastRows.row4909_checked.symm
  rw [row]
  decide
theorem root436Step1 : scan edge (search edge 3) 4 [3067] 3067 root436Tail1 winner =
    scan edge (search edge 3) 4 [3067] 3067 root436Tail2 winner := by
  rw [root436Tail1, scan, if_neg (by decide), root436Candidates1_checked]
  change scan edge (search edge 3) 4 [3067] 3067 root436Tail2
    (if 3 ≤ root436Candidates1.length then search edge 3 [3067, 4909] root436Candidates1 7976 winner else winner) = _
  rw [if_neg (by decide)]
def root436Candidates2 : List Nat := [5737, 6661]
set_option maxRecDepth 8192 in
theorem root436Candidates2_checked : candidates edge 4 3067 5167 root436Tail3 winner.bound = root436Candidates2 := by
  unfold candidates
  have row : edge 5167 = (SortedNatTree.contains FastRows.row5167) := by
    change Data.row5167.contains = (SortedNatTree.contains FastRows.row5167)
    exact FastRows.row5167_checked.symm
  rw [row]
  decide
theorem root436Step2 : scan edge (search edge 3) 4 [3067] 3067 root436Tail2 winner =
    scan edge (search edge 3) 4 [3067] 3067 root436Tail3 winner := by
  rw [root436Tail2, scan, if_neg (by decide), root436Candidates2_checked]
  change scan edge (search edge 3) 4 [3067] 3067 root436Tail3
    (if 3 ≤ root436Candidates2.length then search edge 3 [3067, 5167] root436Candidates2 8234 winner else winner) = _
  rw [if_neg (by decide)]
def root436Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root436Candidates3_checked : candidates edge 4 3067 5413 root436Tail4 winner.bound = root436Candidates3 := by
  unfold candidates
  have row : edge 5413 = (SortedNatTree.contains FastRows.row5413) := by
    change Data.row5413.contains = (SortedNatTree.contains FastRows.row5413)
    exact FastRows.row5413_checked.symm
  rw [row]
  decide
theorem root436Step3 : scan edge (search edge 3) 4 [3067] 3067 root436Tail3 winner =
    scan edge (search edge 3) 4 [3067] 3067 root436Tail4 winner := by
  rw [root436Tail3, scan, if_neg (by decide), root436Candidates3_checked]
  change scan edge (search edge 3) 4 [3067] 3067 root436Tail4
    (if 3 ≤ root436Candidates3.length then search edge 3 [3067, 5413] root436Candidates3 8480 winner else winner) = _
  rw [if_neg (by decide)]
def root436Candidates4 : List Nat := [7687]
set_option maxRecDepth 8192 in
theorem root436Candidates4_checked : candidates edge 4 3067 5419 root436Tail5 winner.bound = root436Candidates4 := by
  unfold candidates
  have row : edge 5419 = (SortedNatTree.contains FastRows.row5419) := by
    change Data.row5419.contains = (SortedNatTree.contains FastRows.row5419)
    exact FastRows.row5419_checked.symm
  rw [row]
  decide
theorem root436Step4 : scan edge (search edge 3) 4 [3067] 3067 root436Tail4 winner =
    scan edge (search edge 3) 4 [3067] 3067 root436Tail5 winner := by
  rw [root436Tail4, scan, if_neg (by decide), root436Candidates4_checked]
  change scan edge (search edge 3) 4 [3067] 3067 root436Tail5
    (if 3 ≤ root436Candidates4.length then search edge 3 [3067, 5419] root436Candidates4 8486 winner else winner) = _
  rw [if_neg (by decide)]
def root436Candidates5 : List Nat := [7687, 8929, 10069]
set_option maxRecDepth 8192 in
theorem root436Candidates5_checked : candidates edge 4 3067 5737 root436Tail6 winner.bound = root436Candidates5 := by
  unfold candidates
  have row : edge 5737 = (SortedNatTree.contains FastRows.row5737) := by
    change Data.row5737.contains = (SortedNatTree.contains FastRows.row5737)
    exact FastRows.row5737_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root436Child5 : search edge 3 [3067, 5737] root436Candidates5 8804 winner = winner := by decide
theorem root436Step5 : scan edge (search edge 3) 4 [3067] 3067 root436Tail5 winner =
    scan edge (search edge 3) 4 [3067] 3067 root436Tail6 winner := by
  rw [root436Tail5, scan, if_neg (by decide), root436Candidates5_checked]
  change scan edge (search edge 3) 4 [3067] 3067 root436Tail6
    (if 3 ≤ root436Candidates5.length then search edge 3 [3067, 5737] root436Candidates5 8804 winner else winner) = _
  rw [if_pos (by decide), root436Child5]
theorem root436Scan6 : scan edge (search edge 3) 4 [3067] 3067 root436Tail6 winner = winner := by rw [root436Tail6, scan, if_pos (by decide)]
theorem root436Scan5 : scan edge (search edge 3) 4 [3067] 3067 root436Tail5 winner = winner :=
  (root436Step5).trans root436Scan6
theorem root436Scan4 : scan edge (search edge 3) 4 [3067] 3067 root436Tail4 winner = winner :=
  (root436Step4).trans root436Scan5
theorem root436Scan3 : scan edge (search edge 3) 4 [3067] 3067 root436Tail3 winner = winner :=
  (root436Step3).trans root436Scan4
theorem root436Scan2 : scan edge (search edge 3) 4 [3067] 3067 root436Tail2 winner = winner :=
  (root436Step2).trans root436Scan3
theorem root436Scan1 : scan edge (search edge 3) 4 [3067] 3067 root436Tail1 winner = winner :=
  (root436Step1).trans root436Scan2
theorem root436Scan0 : scan edge (search edge 3) 4 [3067] 3067 root436Tail0 winner = winner :=
  (root436Step0).trans root436Scan1
set_option maxRecDepth 4096 in
theorem root436Branch : search edge 4 [3067] root436Candidates 3067 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3067] 3067 root436Tail0 winner = winner
  exact root436Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
