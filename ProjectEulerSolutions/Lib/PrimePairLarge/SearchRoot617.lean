import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root617Candidates : List Nat := Data.row4583
set_option maxRecDepth 8192 in
theorem root617Candidates_checked : candidates edge 5 0 4583 tail618 winner.bound = root617Candidates := by
  apply candidates_eq_row edge 5 0 4583 tail618 root617Candidates winner.bound tail618_ordered
  · decide
  · decide
  · funext q
    rfl
def root617Tail36 : List Nat := []
def root617Tail35 : List Nat := 21317 :: root617Tail36
def root617Tail34 : List Nat := 20939 :: root617Tail35
def root617Tail33 : List Nat := 20873 :: root617Tail34
def root617Tail32 : List Nat := 20249 :: root617Tail33
def root617Tail31 : List Nat := 20231 :: root617Tail32
def root617Tail30 : List Nat := 19739 :: root617Tail31
def root617Tail29 : List Nat := 19571 :: root617Tail30
def root617Tail28 : List Nat := 19181 :: root617Tail29
def root617Tail27 : List Nat := 18947 :: root617Tail28
def root617Tail26 : List Nat := 18899 :: root617Tail27
def root617Tail25 : List Nat := 18713 :: root617Tail26
def root617Tail24 : List Nat := 18503 :: root617Tail25
def root617Tail23 : List Nat := 17729 :: root617Tail24
def root617Tail22 : List Nat := 17477 :: root617Tail23
def root617Tail21 : List Nat := 17117 :: root617Tail22
def root617Tail20 : List Nat := 16871 :: root617Tail21
def root617Tail19 : List Nat := 15959 :: root617Tail20
def root617Tail18 : List Nat := 15773 :: root617Tail19
def root617Tail17 : List Nat := 14009 :: root617Tail18
def root617Tail16 : List Nat := 13463 :: root617Tail17
def root617Tail15 : List Nat := 13007 :: root617Tail16
def root617Tail14 : List Nat := 11093 :: root617Tail15
def root617Tail13 : List Nat := 10691 :: root617Tail14
def root617Tail12 : List Nat := 10613 :: root617Tail13
def root617Tail11 : List Nat := 10607 :: root617Tail12
def root617Tail10 : List Nat := 9551 :: root617Tail11
def root617Tail9 : List Nat := 9137 :: root617Tail10
def root617Tail8 : List Nat := 8627 :: root617Tail9
def root617Tail7 : List Nat := 7577 :: root617Tail8
def root617Tail6 : List Nat := 7523 :: root617Tail7
def root617Tail5 : List Nat := 5651 :: root617Tail6
def root617Tail4 : List Nat := 5039 :: root617Tail5
def root617Tail3 : List Nat := 5003 :: root617Tail4
def root617Tail2 : List Nat := 4871 :: root617Tail3
def root617Tail1 : List Nat := 4787 :: root617Tail2
def root617Tail0 : List Nat := 4637 :: root617Tail1
def root617Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root617Candidates0_checked : candidates edge 4 4583 4637 root617Tail1 winner.bound = root617Candidates0 := by
  unfold candidates
  have row : edge 4637 = (SortedNatTree.contains FastRows.row4637) := by
    change Data.row4637.contains = (SortedNatTree.contains FastRows.row4637)
    exact FastRows.row4637_checked.symm
  rw [row]
  decide
theorem root617Step0 : scan edge (search edge 3) 4 [4583] 4583 root617Tail0 winner =
    scan edge (search edge 3) 4 [4583] 4583 root617Tail1 winner := by
  rw [root617Tail0, scan, if_neg (by decide), root617Candidates0_checked]
  change scan edge (search edge 3) 4 [4583] 4583 root617Tail1
    (if 3 ≤ root617Candidates0.length then search edge 3 [4583, 4637] root617Candidates0 9220 winner else winner) = _
  rw [if_neg (by decide)]
def root617Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root617Candidates1_checked : candidates edge 4 4583 4787 root617Tail2 winner.bound = root617Candidates1 := by
  unfold candidates
  have row : edge 4787 = (SortedNatTree.contains FastRows.row4787) := by
    change Data.row4787.contains = (SortedNatTree.contains FastRows.row4787)
    exact FastRows.row4787_checked.symm
  rw [row]
  decide
theorem root617Step1 : scan edge (search edge 3) 4 [4583] 4583 root617Tail1 winner =
    scan edge (search edge 3) 4 [4583] 4583 root617Tail2 winner := by
  rw [root617Tail1, scan, if_neg (by decide), root617Candidates1_checked]
  change scan edge (search edge 3) 4 [4583] 4583 root617Tail2
    (if 3 ≤ root617Candidates1.length then search edge 3 [4583, 4787] root617Candidates1 9370 winner else winner) = _
  rw [if_neg (by decide)]
def root617Candidates2 : List Nat := [8627]
set_option maxRecDepth 8192 in
theorem root617Candidates2_checked : candidates edge 4 4583 4871 root617Tail3 winner.bound = root617Candidates2 := by
  unfold candidates
  have row : edge 4871 = (SortedNatTree.contains FastRows.row4871) := by
    change Data.row4871.contains = (SortedNatTree.contains FastRows.row4871)
    exact FastRows.row4871_checked.symm
  rw [row]
  decide
theorem root617Step2 : scan edge (search edge 3) 4 [4583] 4583 root617Tail2 winner =
    scan edge (search edge 3) 4 [4583] 4583 root617Tail3 winner := by
  rw [root617Tail2, scan, if_neg (by decide), root617Candidates2_checked]
  change scan edge (search edge 3) 4 [4583] 4583 root617Tail3
    (if 3 ≤ root617Candidates2.length then search edge 3 [4583, 4871] root617Candidates2 9454 winner else winner) = _
  rw [if_neg (by decide)]
def root617Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root617Candidates3_checked : candidates edge 4 4583 5003 root617Tail4 winner.bound = root617Candidates3 := by
  unfold candidates
  have row : edge 5003 = (SortedNatTree.contains FastRows.row5003) := by
    change Data.row5003.contains = (SortedNatTree.contains FastRows.row5003)
    exact FastRows.row5003_checked.symm
  rw [row]
  decide
theorem root617Step3 : scan edge (search edge 3) 4 [4583] 4583 root617Tail3 winner =
    scan edge (search edge 3) 4 [4583] 4583 root617Tail4 winner := by
  rw [root617Tail3, scan, if_neg (by decide), root617Candidates3_checked]
  change scan edge (search edge 3) 4 [4583] 4583 root617Tail4
    (if 3 ≤ root617Candidates3.length then search edge 3 [4583, 5003] root617Candidates3 9586 winner else winner) = _
  rw [if_neg (by decide)]
def root617Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root617Candidates4_checked : candidates edge 4 4583 5039 root617Tail5 winner.bound = root617Candidates4 := by
  unfold candidates
  have row : edge 5039 = (SortedNatTree.contains FastRows.row5039) := by
    change Data.row5039.contains = (SortedNatTree.contains FastRows.row5039)
    exact FastRows.row5039_checked.symm
  rw [row]
  decide
theorem root617Step4 : scan edge (search edge 3) 4 [4583] 4583 root617Tail4 winner =
    scan edge (search edge 3) 4 [4583] 4583 root617Tail5 winner := by
  rw [root617Tail4, scan, if_neg (by decide), root617Candidates4_checked]
  change scan edge (search edge 3) 4 [4583] 4583 root617Tail5
    (if 3 ≤ root617Candidates4.length then search edge 3 [4583, 5039] root617Candidates4 9622 winner else winner) = _
  rw [if_neg (by decide)]
theorem root617Scan5 : scan edge (search edge 3) 4 [4583] 4583 root617Tail5 winner = winner := by rw [root617Tail5, scan, if_pos (by decide)]
theorem root617Scan4 : scan edge (search edge 3) 4 [4583] 4583 root617Tail4 winner = winner :=
  (root617Step4).trans root617Scan5
theorem root617Scan3 : scan edge (search edge 3) 4 [4583] 4583 root617Tail3 winner = winner :=
  (root617Step3).trans root617Scan4
theorem root617Scan2 : scan edge (search edge 3) 4 [4583] 4583 root617Tail2 winner = winner :=
  (root617Step2).trans root617Scan3
theorem root617Scan1 : scan edge (search edge 3) 4 [4583] 4583 root617Tail1 winner = winner :=
  (root617Step1).trans root617Scan2
theorem root617Scan0 : scan edge (search edge 3) 4 [4583] 4583 root617Tail0 winner = winner :=
  (root617Step0).trans root617Scan1
set_option maxRecDepth 4096 in
theorem root617Branch : search edge 4 [4583] root617Candidates 4583 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4583] 4583 root617Tail0 winner = winner
  exact root617Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
