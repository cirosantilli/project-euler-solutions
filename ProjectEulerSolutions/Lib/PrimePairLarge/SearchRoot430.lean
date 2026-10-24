import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root430Candidates : List Nat := Data.row3019
set_option maxRecDepth 8192 in
theorem root430Candidates_checked : candidates edge 5 0 3019 tail431 winner.bound = root430Candidates := by
  apply candidates_eq_row edge 5 0 3019 tail431 root430Candidates winner.bound tail431_ordered
  · decide
  · decide
  · funext q
    rfl
def root430Tail39 : List Nat := []
def root430Tail38 : List Nat := 22441 :: root430Tail39
def root430Tail37 : List Nat := 22063 :: root430Tail38
def root430Tail36 : List Nat := 20599 :: root430Tail37
def root430Tail35 : List Nat := 19843 :: root430Tail36
def root430Tail34 : List Nat := 19753 :: root430Tail35
def root430Tail33 : List Nat := 19483 :: root430Tail34
def root430Tail32 : List Nat := 16063 :: root430Tail33
def root430Tail31 : List Nat := 15919 :: root430Tail32
def root430Tail30 : List Nat := 15139 :: root430Tail31
def root430Tail29 : List Nat := 13399 :: root430Tail30
def root430Tail28 : List Nat := 13099 :: root430Tail29
def root430Tail27 : List Nat := 12487 :: root430Tail28
def root430Tail26 : List Nat := 12049 :: root430Tail27
def root430Tail25 : List Nat := 11887 :: root430Tail26
def root430Tail24 : List Nat := 11839 :: root430Tail25
def root430Tail23 : List Nat := 11047 :: root430Tail24
def root430Tail22 : List Nat := 10177 :: root430Tail23
def root430Tail21 : List Nat := 9613 :: root430Tail22
def root430Tail20 : List Nat := 9277 :: root430Tail21
def root430Tail19 : List Nat := 9181 :: root430Tail20
def root430Tail18 : List Nat := 8941 :: root430Tail19
def root430Tail17 : List Nat := 8629 :: root430Tail18
def root430Tail16 : List Nat := 8317 :: root430Tail17
def root430Tail15 : List Nat := 8269 :: root430Tail16
def root430Tail14 : List Nat := 7603 :: root430Tail15
def root430Tail13 : List Nat := 7417 :: root430Tail14
def root430Tail12 : List Nat := 7351 :: root430Tail13
def root430Tail11 : List Nat := 6841 :: root430Tail12
def root430Tail10 : List Nat := 6571 :: root430Tail11
def root430Tail9 : List Nat := 6553 :: root430Tail10
def root430Tail8 : List Nat := 5839 :: root430Tail9
def root430Tail7 : List Nat := 5779 :: root430Tail8
def root430Tail6 : List Nat := 5419 :: root430Tail7
def root430Tail5 : List Nat := 5413 :: root430Tail6
def root430Tail4 : List Nat := 4621 :: root430Tail5
def root430Tail3 : List Nat := 4603 :: root430Tail4
def root430Tail2 : List Nat := 4519 :: root430Tail3
def root430Tail1 : List Nat := 3571 :: root430Tail2
def root430Tail0 : List Nat := 3217 :: root430Tail1
def root430Candidates0 : List Nat := [16063, 19483]
set_option maxRecDepth 8192 in
theorem root430Candidates0_checked : candidates edge 4 3019 3217 root430Tail1 winner.bound = root430Candidates0 := by
  unfold candidates
  have row : edge 3217 = (SortedNatTree.contains FastRows.row3217) := by
    change Data.row3217.contains = (SortedNatTree.contains FastRows.row3217)
    exact FastRows.row3217_checked.symm
  rw [row]
  decide
theorem root430Step0 : scan edge (search edge 3) 4 [3019] 3019 root430Tail0 winner =
    scan edge (search edge 3) 4 [3019] 3019 root430Tail1 winner := by
  rw [root430Tail0, scan, if_neg (by decide), root430Candidates0_checked]
  change scan edge (search edge 3) 4 [3019] 3019 root430Tail1
    (if 3 ≤ root430Candidates0.length then search edge 3 [3019, 3217] root430Candidates0 6236 winner else winner) = _
  rw [if_neg (by decide)]
def root430Candidates1 : List Nat := [9181]
set_option maxRecDepth 8192 in
theorem root430Candidates1_checked : candidates edge 4 3019 3571 root430Tail2 winner.bound = root430Candidates1 := by
  unfold candidates
  have row : edge 3571 = (SortedNatTree.contains FastRows.row3571) := by
    change Data.row3571.contains = (SortedNatTree.contains FastRows.row3571)
    exact FastRows.row3571_checked.symm
  rw [row]
  decide
theorem root430Step1 : scan edge (search edge 3) 4 [3019] 3019 root430Tail1 winner =
    scan edge (search edge 3) 4 [3019] 3019 root430Tail2 winner := by
  rw [root430Tail1, scan, if_neg (by decide), root430Candidates1_checked]
  change scan edge (search edge 3) 4 [3019] 3019 root430Tail2
    (if 3 ≤ root430Candidates1.length then search edge 3 [3019, 3571] root430Candidates1 6590 winner else winner) = _
  rw [if_neg (by decide)]
def root430Candidates2 : List Nat := [5419]
set_option maxRecDepth 8192 in
theorem root430Candidates2_checked : candidates edge 4 3019 4519 root430Tail3 winner.bound = root430Candidates2 := by
  unfold candidates
  have row : edge 4519 = (SortedNatTree.contains FastRows.row4519) := by
    change Data.row4519.contains = (SortedNatTree.contains FastRows.row4519)
    exact FastRows.row4519_checked.symm
  rw [row]
  decide
theorem root430Step2 : scan edge (search edge 3) 4 [3019] 3019 root430Tail2 winner =
    scan edge (search edge 3) 4 [3019] 3019 root430Tail3 winner := by
  rw [root430Tail2, scan, if_neg (by decide), root430Candidates2_checked]
  change scan edge (search edge 3) 4 [3019] 3019 root430Tail3
    (if 3 ≤ root430Candidates2.length then search edge 3 [3019, 4519] root430Candidates2 7538 winner else winner) = _
  rw [if_neg (by decide)]
def root430Candidates3 : List Nat := [7603]
set_option maxRecDepth 8192 in
theorem root430Candidates3_checked : candidates edge 4 3019 4603 root430Tail4 winner.bound = root430Candidates3 := by
  unfold candidates
  have row : edge 4603 = (SortedNatTree.contains FastRows.row4603) := by
    change Data.row4603.contains = (SortedNatTree.contains FastRows.row4603)
    exact FastRows.row4603_checked.symm
  rw [row]
  decide
theorem root430Step3 : scan edge (search edge 3) 4 [3019] 3019 root430Tail3 winner =
    scan edge (search edge 3) 4 [3019] 3019 root430Tail4 winner := by
  rw [root430Tail3, scan, if_neg (by decide), root430Candidates3_checked]
  change scan edge (search edge 3) 4 [3019] 3019 root430Tail4
    (if 3 ≤ root430Candidates3.length then search edge 3 [3019, 4603] root430Candidates3 7622 winner else winner) = _
  rw [if_neg (by decide)]
def root430Candidates4 : List Nat := [5779]
set_option maxRecDepth 8192 in
theorem root430Candidates4_checked : candidates edge 4 3019 4621 root430Tail5 winner.bound = root430Candidates4 := by
  unfold candidates
  have row : edge 4621 = (SortedNatTree.contains FastRows.row4621) := by
    change Data.row4621.contains = (SortedNatTree.contains FastRows.row4621)
    exact FastRows.row4621_checked.symm
  rw [row]
  decide
theorem root430Step4 : scan edge (search edge 3) 4 [3019] 3019 root430Tail4 winner =
    scan edge (search edge 3) 4 [3019] 3019 root430Tail5 winner := by
  rw [root430Tail4, scan, if_neg (by decide), root430Candidates4_checked]
  change scan edge (search edge 3) 4 [3019] 3019 root430Tail5
    (if 3 ≤ root430Candidates4.length then search edge 3 [3019, 4621] root430Candidates4 7640 winner else winner) = _
  rw [if_neg (by decide)]
def root430Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root430Candidates5_checked : candidates edge 4 3019 5413 root430Tail6 winner.bound = root430Candidates5 := by
  unfold candidates
  have row : edge 5413 = (SortedNatTree.contains FastRows.row5413) := by
    change Data.row5413.contains = (SortedNatTree.contains FastRows.row5413)
    exact FastRows.row5413_checked.symm
  rw [row]
  decide
theorem root430Step5 : scan edge (search edge 3) 4 [3019] 3019 root430Tail5 winner =
    scan edge (search edge 3) 4 [3019] 3019 root430Tail6 winner := by
  rw [root430Tail5, scan, if_neg (by decide), root430Candidates5_checked]
  change scan edge (search edge 3) 4 [3019] 3019 root430Tail6
    (if 3 ≤ root430Candidates5.length then search edge 3 [3019, 5413] root430Candidates5 8432 winner else winner) = _
  rw [if_neg (by decide)]
def root430Candidates6 : List Nat := [13399]
set_option maxRecDepth 8192 in
theorem root430Candidates6_checked : candidates edge 4 3019 5419 root430Tail7 winner.bound = root430Candidates6 := by
  unfold candidates
  have row : edge 5419 = (SortedNatTree.contains FastRows.row5419) := by
    change Data.row5419.contains = (SortedNatTree.contains FastRows.row5419)
    exact FastRows.row5419_checked.symm
  rw [row]
  decide
theorem root430Step6 : scan edge (search edge 3) 4 [3019] 3019 root430Tail6 winner =
    scan edge (search edge 3) 4 [3019] 3019 root430Tail7 winner := by
  rw [root430Tail6, scan, if_neg (by decide), root430Candidates6_checked]
  change scan edge (search edge 3) 4 [3019] 3019 root430Tail7
    (if 3 ≤ root430Candidates6.length then search edge 3 [3019, 5419] root430Candidates6 8438 winner else winner) = _
  rw [if_neg (by decide)]
theorem root430Scan7 : scan edge (search edge 3) 4 [3019] 3019 root430Tail7 winner = winner := by rw [root430Tail7, scan, if_pos (by decide)]
theorem root430Scan6 : scan edge (search edge 3) 4 [3019] 3019 root430Tail6 winner = winner :=
  (root430Step6).trans root430Scan7
theorem root430Scan5 : scan edge (search edge 3) 4 [3019] 3019 root430Tail5 winner = winner :=
  (root430Step5).trans root430Scan6
theorem root430Scan4 : scan edge (search edge 3) 4 [3019] 3019 root430Tail4 winner = winner :=
  (root430Step4).trans root430Scan5
theorem root430Scan3 : scan edge (search edge 3) 4 [3019] 3019 root430Tail3 winner = winner :=
  (root430Step3).trans root430Scan4
theorem root430Scan2 : scan edge (search edge 3) 4 [3019] 3019 root430Tail2 winner = winner :=
  (root430Step2).trans root430Scan3
theorem root430Scan1 : scan edge (search edge 3) 4 [3019] 3019 root430Tail1 winner = winner :=
  (root430Step1).trans root430Scan2
theorem root430Scan0 : scan edge (search edge 3) 4 [3019] 3019 root430Tail0 winner = winner :=
  (root430Step0).trans root430Scan1
set_option maxRecDepth 4096 in
theorem root430Branch : search edge 4 [3019] root430Candidates 3019 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3019] 3019 root430Tail0 winner = winner
  exact root430Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
