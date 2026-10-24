import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root490Candidates : List Nat := Data.row3529
set_option maxRecDepth 8192 in
theorem root490Candidates_checked : candidates edge 5 0 3529 tail491 winner.bound = root490Candidates := by
  apply candidates_eq_row edge 5 0 3529 tail491 root490Candidates winner.bound tail491_ordered
  · decide
  · decide
  · funext q
    rfl
def root490Tail40 : List Nat := []
def root490Tail39 : List Nat := 22303 :: root490Tail40
def root490Tail38 : List Nat := 21589 :: root490Tail39
def root490Tail37 : List Nat := 21433 :: root490Tail38
def root490Tail36 : List Nat := 21013 :: root490Tail37
def root490Tail35 : List Nat := 20983 :: root490Tail36
def root490Tail34 : List Nat := 20857 :: root490Tail35
def root490Tail33 : List Nat := 20719 :: root490Tail34
def root490Tail32 : List Nat := 20521 :: root490Tail33
def root490Tail31 : List Nat := 19867 :: root490Tail32
def root490Tail30 : List Nat := 16981 :: root490Tail31
def root490Tail29 : List Nat := 16699 :: root490Tail30
def root490Tail28 : List Nat := 16633 :: root490Tail29
def root490Tail27 : List Nat := 15217 :: root490Tail28
def root490Tail26 : List Nat := 14143 :: root490Tail27
def root490Tail25 : List Nat := 13567 :: root490Tail26
def root490Tail24 : List Nat := 13339 :: root490Tail25
def root490Tail23 : List Nat := 12211 :: root490Tail24
def root490Tail22 : List Nat := 11437 :: root490Tail23
def root490Tail21 : List Nat := 11257 :: root490Tail22
def root490Tail20 : List Nat := 11173 :: root490Tail21
def root490Tail19 : List Nat := 11059 :: root490Tail20
def root490Tail18 : List Nat := 10069 :: root490Tail19
def root490Tail17 : List Nat := 9619 :: root490Tail18
def root490Tail16 : List Nat := 9403 :: root490Tail17
def root490Tail15 : List Nat := 8707 :: root490Tail16
def root490Tail14 : List Nat := 8521 :: root490Tail15
def root490Tail13 : List Nat := 8059 :: root490Tail14
def root490Tail12 : List Nat := 7393 :: root490Tail13
def root490Tail11 : List Nat := 6733 :: root490Tail12
def root490Tail10 : List Nat := 6553 :: root490Tail11
def root490Tail9 : List Nat := 6427 :: root490Tail10
def root490Tail8 : List Nat := 6397 :: root490Tail9
def root490Tail7 : List Nat := 5851 :: root490Tail8
def root490Tail6 : List Nat := 5569 :: root490Tail7
def root490Tail5 : List Nat := 5527 :: root490Tail6
def root490Tail4 : List Nat := 5011 :: root490Tail5
def root490Tail3 : List Nat := 4969 :: root490Tail4
def root490Tail2 : List Nat := 4657 :: root490Tail3
def root490Tail1 : List Nat := 4423 :: root490Tail2
def root490Tail0 : List Nat := 3943 :: root490Tail1
def root490Candidates0 : List Nat := [5569, 8059]
set_option maxRecDepth 8192 in
theorem root490Candidates0_checked : candidates edge 4 3529 3943 root490Tail1 winner.bound = root490Candidates0 := by
  unfold candidates
  have row : edge 3943 = (SortedNatTree.contains FastRows.row3943) := by
    change Data.row3943.contains = (SortedNatTree.contains FastRows.row3943)
    exact FastRows.row3943_checked.symm
  rw [row]
  decide
theorem root490Step0 : scan edge (search edge 3) 4 [3529] 3529 root490Tail0 winner =
    scan edge (search edge 3) 4 [3529] 3529 root490Tail1 winner := by
  rw [root490Tail0, scan, if_neg (by decide), root490Candidates0_checked]
  change scan edge (search edge 3) 4 [3529] 3529 root490Tail1
    (if 3 ≤ root490Candidates0.length then search edge 3 [3529, 3943] root490Candidates0 7472 winner else winner) = _
  rw [if_neg (by decide)]
def root490Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root490Candidates1_checked : candidates edge 4 3529 4423 root490Tail2 winner.bound = root490Candidates1 := by
  unfold candidates
  have row : edge 4423 = (SortedNatTree.contains FastRows.row4423) := by
    change Data.row4423.contains = (SortedNatTree.contains FastRows.row4423)
    exact FastRows.row4423_checked.symm
  rw [row]
  decide
theorem root490Step1 : scan edge (search edge 3) 4 [3529] 3529 root490Tail1 winner =
    scan edge (search edge 3) 4 [3529] 3529 root490Tail2 winner := by
  rw [root490Tail1, scan, if_neg (by decide), root490Candidates1_checked]
  change scan edge (search edge 3) 4 [3529] 3529 root490Tail2
    (if 3 ≤ root490Candidates1.length then search edge 3 [3529, 4423] root490Candidates1 7952 winner else winner) = _
  rw [if_neg (by decide)]
def root490Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root490Candidates2_checked : candidates edge 4 3529 4657 root490Tail3 winner.bound = root490Candidates2 := by
  unfold candidates
  have row : edge 4657 = (SortedNatTree.contains FastRows.row4657) := by
    change Data.row4657.contains = (SortedNatTree.contains FastRows.row4657)
    exact FastRows.row4657_checked.symm
  rw [row]
  decide
theorem root490Step2 : scan edge (search edge 3) 4 [3529] 3529 root490Tail2 winner =
    scan edge (search edge 3) 4 [3529] 3529 root490Tail3 winner := by
  rw [root490Tail2, scan, if_neg (by decide), root490Candidates2_checked]
  change scan edge (search edge 3) 4 [3529] 3529 root490Tail3
    (if 3 ≤ root490Candidates2.length then search edge 3 [3529, 4657] root490Candidates2 8186 winner else winner) = _
  rw [if_neg (by decide)]
def root490Candidates3 : List Nat := [6397]
set_option maxRecDepth 8192 in
theorem root490Candidates3_checked : candidates edge 4 3529 4969 root490Tail4 winner.bound = root490Candidates3 := by
  unfold candidates
  have row : edge 4969 = (SortedNatTree.contains FastRows.row4969) := by
    change Data.row4969.contains = (SortedNatTree.contains FastRows.row4969)
    exact FastRows.row4969_checked.symm
  rw [row]
  decide
theorem root490Step3 : scan edge (search edge 3) 4 [3529] 3529 root490Tail3 winner =
    scan edge (search edge 3) 4 [3529] 3529 root490Tail4 winner := by
  rw [root490Tail3, scan, if_neg (by decide), root490Candidates3_checked]
  change scan edge (search edge 3) 4 [3529] 3529 root490Tail4
    (if 3 ≤ root490Candidates3.length then search edge 3 [3529, 4969] root490Candidates3 8498 winner else winner) = _
  rw [if_neg (by decide)]
def root490Candidates4 : List Nat := [6397, 7393]
set_option maxRecDepth 8192 in
theorem root490Candidates4_checked : candidates edge 4 3529 5011 root490Tail5 winner.bound = root490Candidates4 := by
  unfold candidates
  have row : edge 5011 = (SortedNatTree.contains FastRows.row5011) := by
    change Data.row5011.contains = (SortedNatTree.contains FastRows.row5011)
    exact FastRows.row5011_checked.symm
  rw [row]
  decide
theorem root490Step4 : scan edge (search edge 3) 4 [3529] 3529 root490Tail4 winner =
    scan edge (search edge 3) 4 [3529] 3529 root490Tail5 winner := by
  rw [root490Tail4, scan, if_neg (by decide), root490Candidates4_checked]
  change scan edge (search edge 3) 4 [3529] 3529 root490Tail5
    (if 3 ≤ root490Candidates4.length then search edge 3 [3529, 5011] root490Candidates4 8540 winner else winner) = _
  rw [if_neg (by decide)]
def root490Candidates5 : List Nat := [9403]
set_option maxRecDepth 8192 in
theorem root490Candidates5_checked : candidates edge 4 3529 5527 root490Tail6 winner.bound = root490Candidates5 := by
  unfold candidates
  have row : edge 5527 = (SortedNatTree.contains FastRows.row5527) := by
    change Data.row5527.contains = (SortedNatTree.contains FastRows.row5527)
    exact FastRows.row5527_checked.symm
  rw [row]
  decide
theorem root490Step5 : scan edge (search edge 3) 4 [3529] 3529 root490Tail5 winner =
    scan edge (search edge 3) 4 [3529] 3529 root490Tail6 winner := by
  rw [root490Tail5, scan, if_neg (by decide), root490Candidates5_checked]
  change scan edge (search edge 3) 4 [3529] 3529 root490Tail6
    (if 3 ≤ root490Candidates5.length then search edge 3 [3529, 5527] root490Candidates5 9056 winner else winner) = _
  rw [if_neg (by decide)]
def root490Candidates6 : List Nat := [8521]
set_option maxRecDepth 8192 in
theorem root490Candidates6_checked : candidates edge 4 3529 5569 root490Tail7 winner.bound = root490Candidates6 := by
  unfold candidates
  have row : edge 5569 = (SortedNatTree.contains FastRows.row5569) := by
    change Data.row5569.contains = (SortedNatTree.contains FastRows.row5569)
    exact FastRows.row5569_checked.symm
  rw [row]
  decide
theorem root490Step6 : scan edge (search edge 3) 4 [3529] 3529 root490Tail6 winner =
    scan edge (search edge 3) 4 [3529] 3529 root490Tail7 winner := by
  rw [root490Tail6, scan, if_neg (by decide), root490Candidates6_checked]
  change scan edge (search edge 3) 4 [3529] 3529 root490Tail7
    (if 3 ≤ root490Candidates6.length then search edge 3 [3529, 5569] root490Candidates6 9098 winner else winner) = _
  rw [if_neg (by decide)]
theorem root490Scan7 : scan edge (search edge 3) 4 [3529] 3529 root490Tail7 winner = winner := by rw [root490Tail7, scan, if_pos (by decide)]
theorem root490Scan6 : scan edge (search edge 3) 4 [3529] 3529 root490Tail6 winner = winner :=
  (root490Step6).trans root490Scan7
theorem root490Scan5 : scan edge (search edge 3) 4 [3529] 3529 root490Tail5 winner = winner :=
  (root490Step5).trans root490Scan6
theorem root490Scan4 : scan edge (search edge 3) 4 [3529] 3529 root490Tail4 winner = winner :=
  (root490Step4).trans root490Scan5
theorem root490Scan3 : scan edge (search edge 3) 4 [3529] 3529 root490Tail3 winner = winner :=
  (root490Step3).trans root490Scan4
theorem root490Scan2 : scan edge (search edge 3) 4 [3529] 3529 root490Tail2 winner = winner :=
  (root490Step2).trans root490Scan3
theorem root490Scan1 : scan edge (search edge 3) 4 [3529] 3529 root490Tail1 winner = winner :=
  (root490Step1).trans root490Scan2
theorem root490Scan0 : scan edge (search edge 3) 4 [3529] 3529 root490Tail0 winner = winner :=
  (root490Step0).trans root490Scan1
set_option maxRecDepth 4096 in
theorem root490Branch : search edge 4 [3529] root490Candidates 3529 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3529] 3529 root490Tail0 winner = winner
  exact root490Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
