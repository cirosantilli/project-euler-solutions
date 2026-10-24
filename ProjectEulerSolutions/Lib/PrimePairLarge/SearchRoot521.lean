import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root521Candidates : List Nat := Data.row3767
set_option maxRecDepth 8192 in
theorem root521Candidates_checked : candidates edge 5 0 3767 tail522 winner.bound = root521Candidates := by
  apply candidates_eq_row edge 5 0 3767 tail522 root521Candidates winner.bound tail522_ordered
  · decide
  · decide
  · funext q
    rfl
def root521Tail38 : List Nat := []
def root521Tail37 : List Nat := 21701 :: root521Tail38
def root521Tail36 : List Nat := 19793 :: root521Tail37
def root521Tail35 : List Nat := 19727 :: root521Tail36
def root521Tail34 : List Nat := 19697 :: root521Tail35
def root521Tail33 : List Nat := 19583 :: root521Tail34
def root521Tail32 : List Nat := 19373 :: root521Tail33
def root521Tail31 : List Nat := 19121 :: root521Tail32
def root521Tail30 : List Nat := 18341 :: root521Tail31
def root521Tail29 : List Nat := 17597 :: root521Tail30
def root521Tail28 : List Nat := 17417 :: root521Tail29
def root521Tail27 : List Nat := 17333 :: root521Tail28
def root521Tail26 : List Nat := 15803 :: root521Tail27
def root521Tail25 : List Nat := 14771 :: root521Tail26
def root521Tail24 : List Nat := 14087 :: root521Tail25
def root521Tail23 : List Nat := 13763 :: root521Tail24
def root521Tail22 : List Nat := 13751 :: root521Tail23
def root521Tail21 : List Nat := 13499 :: root521Tail22
def root521Tail20 : List Nat := 12113 :: root521Tail21
def root521Tail19 : List Nat := 12041 :: root521Tail20
def root521Tail18 : List Nat := 11717 :: root521Tail19
def root521Tail17 : List Nat := 9437 :: root521Tail18
def root521Tail16 : List Nat := 9377 :: root521Tail17
def root521Tail15 : List Nat := 8933 :: root521Tail16
def root521Tail14 : List Nat := 7793 :: root521Tail15
def root521Tail13 : List Nat := 7547 :: root521Tail14
def root521Tail12 : List Nat := 7487 :: root521Tail13
def root521Tail11 : List Nat := 7193 :: root521Tail12
def root521Tail10 : List Nat := 7001 :: root521Tail11
def root521Tail9 : List Nat := 6911 :: root521Tail10
def root521Tail8 : List Nat := 6719 :: root521Tail9
def root521Tail7 : List Nat := 6701 :: root521Tail8
def root521Tail6 : List Nat := 6521 :: root521Tail7
def root521Tail5 : List Nat := 5987 :: root521Tail6
def root521Tail4 : List Nat := 4937 :: root521Tail5
def root521Tail3 : List Nat := 4523 :: root521Tail4
def root521Tail2 : List Nat := 4127 :: root521Tail3
def root521Tail1 : List Nat := 3947 :: root521Tail2
def root521Tail0 : List Nat := 3821 :: root521Tail1
def root521Candidates0 : List Nat := [8933]
set_option maxRecDepth 8192 in
theorem root521Candidates0_checked : candidates edge 4 3767 3821 root521Tail1 winner.bound = root521Candidates0 := by
  unfold candidates
  have row : edge 3821 = (SortedNatTree.contains FastRows.row3821) := by
    change Data.row3821.contains = (SortedNatTree.contains FastRows.row3821)
    exact FastRows.row3821_checked.symm
  rw [row]
  decide
theorem root521Step0 : scan edge (search edge 3) 4 [3767] 3767 root521Tail0 winner =
    scan edge (search edge 3) 4 [3767] 3767 root521Tail1 winner := by
  rw [root521Tail0, scan, if_neg (by decide), root521Candidates0_checked]
  change scan edge (search edge 3) 4 [3767] 3767 root521Tail1
    (if 3 ≤ root521Candidates0.length then search edge 3 [3767, 3821] root521Candidates0 7588 winner else winner) = _
  rw [if_neg (by decide)]
def root521Candidates1 : List Nat := [7001, 15803, 17333]
set_option maxRecDepth 8192 in
theorem root521Candidates1_checked : candidates edge 4 3767 3947 root521Tail2 winner.bound = root521Candidates1 := by
  unfold candidates
  have row : edge 3947 = (SortedNatTree.contains FastRows.row3947) := by
    change Data.row3947.contains = (SortedNatTree.contains FastRows.row3947)
    exact FastRows.row3947_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root521Child1 : search edge 3 [3767, 3947] root521Candidates1 7714 winner = winner := by decide
theorem root521Step1 : scan edge (search edge 3) 4 [3767] 3767 root521Tail1 winner =
    scan edge (search edge 3) 4 [3767] 3767 root521Tail2 winner := by
  rw [root521Tail1, scan, if_neg (by decide), root521Candidates1_checked]
  change scan edge (search edge 3) 4 [3767] 3767 root521Tail2
    (if 3 ≤ root521Candidates1.length then search edge 3 [3767, 3947] root521Candidates1 7714 winner else winner) = _
  rw [if_pos (by decide), root521Child1]
def root521Candidates2 : List Nat := [8933, 13751]
set_option maxRecDepth 8192 in
theorem root521Candidates2_checked : candidates edge 4 3767 4127 root521Tail3 winner.bound = root521Candidates2 := by
  unfold candidates
  have row : edge 4127 = (SortedNatTree.contains FastRows.row4127) := by
    change Data.row4127.contains = (SortedNatTree.contains FastRows.row4127)
    exact FastRows.row4127_checked.symm
  rw [row]
  decide
theorem root521Step2 : scan edge (search edge 3) 4 [3767] 3767 root521Tail2 winner =
    scan edge (search edge 3) 4 [3767] 3767 root521Tail3 winner := by
  rw [root521Tail2, scan, if_neg (by decide), root521Candidates2_checked]
  change scan edge (search edge 3) 4 [3767] 3767 root521Tail3
    (if 3 ≤ root521Candidates2.length then search edge 3 [3767, 4127] root521Candidates2 7894 winner else winner) = _
  rw [if_neg (by decide)]
def root521Candidates3 : List Nat := [6911]
set_option maxRecDepth 8192 in
theorem root521Candidates3_checked : candidates edge 4 3767 4523 root521Tail4 winner.bound = root521Candidates3 := by
  unfold candidates
  have row : edge 4523 = (SortedNatTree.contains FastRows.row4523) := by
    change Data.row4523.contains = (SortedNatTree.contains FastRows.row4523)
    exact FastRows.row4523_checked.symm
  rw [row]
  decide
theorem root521Step3 : scan edge (search edge 3) 4 [3767] 3767 root521Tail3 winner =
    scan edge (search edge 3) 4 [3767] 3767 root521Tail4 winner := by
  rw [root521Tail3, scan, if_neg (by decide), root521Candidates3_checked]
  change scan edge (search edge 3) 4 [3767] 3767 root521Tail4
    (if 3 ≤ root521Candidates3.length then search edge 3 [3767, 4523] root521Candidates3 8290 winner else winner) = _
  rw [if_neg (by decide)]
def root521Candidates4 : List Nat := [7001]
set_option maxRecDepth 8192 in
theorem root521Candidates4_checked : candidates edge 4 3767 4937 root521Tail5 winner.bound = root521Candidates4 := by
  unfold candidates
  have row : edge 4937 = (SortedNatTree.contains FastRows.row4937) := by
    change Data.row4937.contains = (SortedNatTree.contains FastRows.row4937)
    exact FastRows.row4937_checked.symm
  rw [row]
  decide
theorem root521Step4 : scan edge (search edge 3) 4 [3767] 3767 root521Tail4 winner =
    scan edge (search edge 3) 4 [3767] 3767 root521Tail5 winner := by
  rw [root521Tail4, scan, if_neg (by decide), root521Candidates4_checked]
  change scan edge (search edge 3) 4 [3767] 3767 root521Tail5
    (if 3 ≤ root521Candidates4.length then search edge 3 [3767, 4937] root521Candidates4 8704 winner else winner) = _
  rw [if_neg (by decide)]
theorem root521Scan5 : scan edge (search edge 3) 4 [3767] 3767 root521Tail5 winner = winner := by rw [root521Tail5, scan, if_pos (by decide)]
theorem root521Scan4 : scan edge (search edge 3) 4 [3767] 3767 root521Tail4 winner = winner :=
  (root521Step4).trans root521Scan5
theorem root521Scan3 : scan edge (search edge 3) 4 [3767] 3767 root521Tail3 winner = winner :=
  (root521Step3).trans root521Scan4
theorem root521Scan2 : scan edge (search edge 3) 4 [3767] 3767 root521Tail2 winner = winner :=
  (root521Step2).trans root521Scan3
theorem root521Scan1 : scan edge (search edge 3) 4 [3767] 3767 root521Tail1 winner = winner :=
  (root521Step1).trans root521Scan2
theorem root521Scan0 : scan edge (search edge 3) 4 [3767] 3767 root521Tail0 winner = winner :=
  (root521Step0).trans root521Scan1
set_option maxRecDepth 4096 in
theorem root521Branch : search edge 4 [3767] root521Candidates 3767 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3767] 3767 root521Tail0 winner = winner
  exact root521Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
