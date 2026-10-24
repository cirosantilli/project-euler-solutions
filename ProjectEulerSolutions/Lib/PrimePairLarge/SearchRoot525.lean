import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root525Candidates : List Nat := Data.row3797
set_option maxRecDepth 8192 in
theorem root525Candidates_checked : candidates edge 5 0 3797 tail526 winner.bound = root525Candidates := by
  apply candidates_eq_row edge 5 0 3797 tail526 root525Candidates winner.bound tail526_ordered
  · decide
  · decide
  · funext q
    rfl
def root525Tail43 : List Nat := []
def root525Tail42 : List Nat := 21851 :: root525Tail43
def root525Tail41 : List Nat := 20807 :: root525Tail42
def root525Tail40 : List Nat := 20261 :: root525Tail41
def root525Tail39 : List Nat := 20123 :: root525Tail40
def root525Tail38 : List Nat := 19763 :: root525Tail39
def root525Tail37 : List Nat := 19709 :: root525Tail38
def root525Tail36 : List Nat := 19073 :: root525Tail37
def root525Tail35 : List Nat := 18587 :: root525Tail36
def root525Tail34 : List Nat := 18371 :: root525Tail35
def root525Tail33 : List Nat := 17981 :: root525Tail34
def root525Tail32 : List Nat := 16229 :: root525Tail33
def root525Tail31 : List Nat := 16193 :: root525Tail32
def root525Tail30 : List Nat := 15923 :: root525Tail31
def root525Tail29 : List Nat := 15641 :: root525Tail30
def root525Tail28 : List Nat := 15263 :: root525Tail29
def root525Tail27 : List Nat := 14891 :: root525Tail28
def root525Tail26 : List Nat := 14033 :: root525Tail27
def root525Tail25 : List Nat := 13967 :: root525Tail26
def root525Tail24 : List Nat := 13883 :: root525Tail25
def root525Tail23 : List Nat := 13487 :: root525Tail24
def root525Tail22 : List Nat := 13331 :: root525Tail23
def root525Tail21 : List Nat := 12917 :: root525Tail22
def root525Tail20 : List Nat := 12413 :: root525Tail21
def root525Tail19 : List Nat := 11399 :: root525Tail20
def root525Tail18 : List Nat := 10889 :: root525Tail19
def root525Tail17 : List Nat := 10733 :: root525Tail18
def root525Tail16 : List Nat := 10271 :: root525Tail17
def root525Tail15 : List Nat := 9839 :: root525Tail16
def root525Tail14 : List Nat := 9791 :: root525Tail15
def root525Tail13 : List Nat := 8861 :: root525Tail14
def root525Tail12 : List Nat := 8537 :: root525Tail13
def root525Tail11 : List Nat := 8273 :: root525Tail12
def root525Tail10 : List Nat := 7937 :: root525Tail11
def root525Tail9 : List Nat := 7529 :: root525Tail10
def root525Tail8 : List Nat := 7109 :: root525Tail9
def root525Tail7 : List Nat := 7019 :: root525Tail8
def root525Tail6 : List Nat := 7013 :: root525Tail7
def root525Tail5 : List Nat := 6089 :: root525Tail6
def root525Tail4 : List Nat := 5393 :: root525Tail5
def root525Tail3 : List Nat := 5303 :: root525Tail4
def root525Tail2 : List Nat := 5261 :: root525Tail3
def root525Tail1 : List Nat := 4049 :: root525Tail2
def root525Tail0 : List Nat := 3833 :: root525Tail1
def root525Candidates0 : List Nat := [14891, 18371]
set_option maxRecDepth 8192 in
theorem root525Candidates0_checked : candidates edge 4 3797 3833 root525Tail1 winner.bound = root525Candidates0 := by
  unfold candidates
  have row : edge 3833 = (SortedNatTree.contains FastRows.row3833) := by
    change Data.row3833.contains = (SortedNatTree.contains FastRows.row3833)
    exact FastRows.row3833_checked.symm
  rw [row]
  decide
theorem root525Step0 : scan edge (search edge 3) 4 [3797] 3797 root525Tail0 winner =
    scan edge (search edge 3) 4 [3797] 3797 root525Tail1 winner := by
  rw [root525Tail0, scan, if_neg (by decide), root525Candidates0_checked]
  change scan edge (search edge 3) 4 [3797] 3797 root525Tail1
    (if 3 ≤ root525Candidates0.length then search edge 3 [3797, 3833] root525Candidates0 7630 winner else winner) = _
  rw [if_neg (by decide)]
def root525Candidates1 : List Nat := [5261, 13967]
set_option maxRecDepth 8192 in
theorem root525Candidates1_checked : candidates edge 4 3797 4049 root525Tail2 winner.bound = root525Candidates1 := by
  unfold candidates
  have row : edge 4049 = (SortedNatTree.contains FastRows.row4049) := by
    change Data.row4049.contains = (SortedNatTree.contains FastRows.row4049)
    exact FastRows.row4049_checked.symm
  rw [row]
  decide
theorem root525Step1 : scan edge (search edge 3) 4 [3797] 3797 root525Tail1 winner =
    scan edge (search edge 3) 4 [3797] 3797 root525Tail2 winner := by
  rw [root525Tail1, scan, if_neg (by decide), root525Candidates1_checked]
  change scan edge (search edge 3) 4 [3797] 3797 root525Tail2
    (if 3 ≤ root525Candidates1.length then search edge 3 [3797, 4049] root525Candidates1 7846 winner else winner) = _
  rw [if_neg (by decide)]
def root525Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root525Candidates2_checked : candidates edge 4 3797 5261 root525Tail3 winner.bound = root525Candidates2 := by
  unfold candidates
  have row : edge 5261 = (SortedNatTree.contains FastRows.row5261) := by
    change Data.row5261.contains = (SortedNatTree.contains FastRows.row5261)
    exact FastRows.row5261_checked.symm
  rw [row]
  decide
theorem root525Step2 : scan edge (search edge 3) 4 [3797] 3797 root525Tail2 winner =
    scan edge (search edge 3) 4 [3797] 3797 root525Tail3 winner := by
  rw [root525Tail2, scan, if_neg (by decide), root525Candidates2_checked]
  change scan edge (search edge 3) 4 [3797] 3797 root525Tail3
    (if 3 ≤ root525Candidates2.length then search edge 3 [3797, 5261] root525Candidates2 9058 winner else winner) = _
  rw [if_neg (by decide)]
def root525Candidates3 : List Nat := [12917, 14891]
set_option maxRecDepth 8192 in
theorem root525Candidates3_checked : candidates edge 4 3797 5303 root525Tail4 winner.bound = root525Candidates3 := by
  unfold candidates
  have row : edge 5303 = (SortedNatTree.contains FastRows.row5303) := by
    change Data.row5303.contains = (SortedNatTree.contains FastRows.row5303)
    exact FastRows.row5303_checked.symm
  rw [row]
  decide
theorem root525Step3 : scan edge (search edge 3) 4 [3797] 3797 root525Tail3 winner =
    scan edge (search edge 3) 4 [3797] 3797 root525Tail4 winner := by
  rw [root525Tail3, scan, if_neg (by decide), root525Candidates3_checked]
  change scan edge (search edge 3) 4 [3797] 3797 root525Tail4
    (if 3 ≤ root525Candidates3.length then search edge 3 [3797, 5303] root525Candidates3 9100 winner else winner) = _
  rw [if_neg (by decide)]
def root525Candidates4 : List Nat := [7937]
set_option maxRecDepth 8192 in
theorem root525Candidates4_checked : candidates edge 4 3797 5393 root525Tail5 winner.bound = root525Candidates4 := by
  unfold candidates
  have row : edge 5393 = (SortedNatTree.contains FastRows.row5393) := by
    change Data.row5393.contains = (SortedNatTree.contains FastRows.row5393)
    exact FastRows.row5393_checked.symm
  rw [row]
  decide
theorem root525Step4 : scan edge (search edge 3) 4 [3797] 3797 root525Tail4 winner =
    scan edge (search edge 3) 4 [3797] 3797 root525Tail5 winner := by
  rw [root525Tail4, scan, if_neg (by decide), root525Candidates4_checked]
  change scan edge (search edge 3) 4 [3797] 3797 root525Tail5
    (if 3 ≤ root525Candidates4.length then search edge 3 [3797, 5393] root525Candidates4 9190 winner else winner) = _
  rw [if_neg (by decide)]
theorem root525Scan5 : scan edge (search edge 3) 4 [3797] 3797 root525Tail5 winner = winner := by rw [root525Tail5, scan, if_pos (by decide)]
theorem root525Scan4 : scan edge (search edge 3) 4 [3797] 3797 root525Tail4 winner = winner :=
  (root525Step4).trans root525Scan5
theorem root525Scan3 : scan edge (search edge 3) 4 [3797] 3797 root525Tail3 winner = winner :=
  (root525Step3).trans root525Scan4
theorem root525Scan2 : scan edge (search edge 3) 4 [3797] 3797 root525Tail2 winner = winner :=
  (root525Step2).trans root525Scan3
theorem root525Scan1 : scan edge (search edge 3) 4 [3797] 3797 root525Tail1 winner = winner :=
  (root525Step1).trans root525Scan2
theorem root525Scan0 : scan edge (search edge 3) 4 [3797] 3797 root525Tail0 winner = winner :=
  (root525Step0).trans root525Scan1
set_option maxRecDepth 4096 in
theorem root525Branch : search edge 4 [3797] root525Candidates 3797 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3797] 3797 root525Tail0 winner = winner
  exact root525Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
