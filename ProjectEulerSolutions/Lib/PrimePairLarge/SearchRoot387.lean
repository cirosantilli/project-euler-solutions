import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root387Candidates : List Nat := Data.row2687
set_option maxRecDepth 8192 in
theorem root387Candidates_checked : candidates edge 5 0 2687 tail388 winner.bound = root387Candidates := by
  apply candidates_eq_row edge 5 0 2687 tail388 root387Candidates winner.bound tail388_ordered
  · decide
  · decide
  · funext q
    rfl
def root387Tail42 : List Nat := []
def root387Tail41 : List Nat := 23201 :: root387Tail42
def root387Tail40 : List Nat := 22907 :: root387Tail41
def root387Tail39 : List Nat := 22697 :: root387Tail40
def root387Tail38 : List Nat := 21773 :: root387Tail39
def root387Tail37 : List Nat := 21767 :: root387Tail38
def root387Tail36 : List Nat := 20261 :: root387Tail37
def root387Tail35 : List Nat := 20117 :: root387Tail36
def root387Tail34 : List Nat := 19961 :: root387Tail35
def root387Tail33 : List Nat := 19421 :: root387Tail34
def root387Tail32 : List Nat := 19001 :: root387Tail33
def root387Tail31 : List Nat := 17783 :: root387Tail32
def root387Tail30 : List Nat := 17291 :: root387Tail31
def root387Tail29 : List Nat := 16883 :: root387Tail30
def root387Tail28 : List Nat := 16493 :: root387Tail29
def root387Tail27 : List Nat := 16223 :: root387Tail28
def root387Tail26 : List Nat := 15971 :: root387Tail27
def root387Tail25 : List Nat := 14813 :: root387Tail26
def root387Tail24 : List Nat := 14447 :: root387Tail25
def root387Tail23 : List Nat := 13469 :: root387Tail24
def root387Tail22 : List Nat := 13121 :: root387Tail23
def root387Tail21 : List Nat := 11213 :: root387Tail22
def root387Tail20 : List Nat := 11087 :: root387Tail21
def root387Tail19 : List Nat := 10457 :: root387Tail20
def root387Tail18 : List Nat := 10193 :: root387Tail19
def root387Tail17 : List Nat := 9551 :: root387Tail18
def root387Tail16 : List Nat := 8693 :: root387Tail17
def root387Tail15 : List Nat := 8627 :: root387Tail16
def root387Tail14 : List Nat := 8573 :: root387Tail15
def root387Tail13 : List Nat := 8543 :: root387Tail14
def root387Tail12 : List Nat := 8363 :: root387Tail13
def root387Tail11 : List Nat := 8231 :: root387Tail12
def root387Tail10 : List Nat := 8069 :: root387Tail11
def root387Tail9 : List Nat := 8039 :: root387Tail10
def root387Tail8 : List Nat := 7703 :: root387Tail9
def root387Tail7 : List Nat := 6737 :: root387Tail8
def root387Tail6 : List Nat := 6599 :: root387Tail7
def root387Tail5 : List Nat := 6221 :: root387Tail6
def root387Tail4 : List Nat := 5897 :: root387Tail5
def root387Tail3 : List Nat := 5237 :: root387Tail4
def root387Tail2 : List Nat := 4481 :: root387Tail3
def root387Tail1 : List Nat := 4007 :: root387Tail2
def root387Tail0 : List Nat := 3023 :: root387Tail1
def root387Candidates0 : List Nat := [8069, 14447, 19001, 19961]
set_option maxRecDepth 8192 in
theorem root387Candidates0_checked : candidates edge 4 2687 3023 root387Tail1 winner.bound = root387Candidates0 := by
  unfold candidates
  have row : edge 3023 = (SortedNatTree.contains FastRows.row3023) := by
    change Data.row3023.contains = (SortedNatTree.contains FastRows.row3023)
    exact FastRows.row3023_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root387Child0 : search edge 3 [2687, 3023] root387Candidates0 5710 winner = winner := by decide
theorem root387Step0 : scan edge (search edge 3) 4 [2687] 2687 root387Tail0 winner =
    scan edge (search edge 3) 4 [2687] 2687 root387Tail1 winner := by
  rw [root387Tail0, scan, if_neg (by decide), root387Candidates0_checked]
  change scan edge (search edge 3) 4 [2687] 2687 root387Tail1
    (if 3 ≤ root387Candidates0.length then search edge 3 [2687, 3023] root387Candidates0 5710 winner else winner) = _
  rw [if_pos (by decide), root387Child0]
def root387Candidates1 : List Nat := [8693]
set_option maxRecDepth 8192 in
theorem root387Candidates1_checked : candidates edge 4 2687 4007 root387Tail2 winner.bound = root387Candidates1 := by
  unfold candidates
  have row : edge 4007 = (SortedNatTree.contains FastRows.row4007) := by
    change Data.row4007.contains = (SortedNatTree.contains FastRows.row4007)
    exact FastRows.row4007_checked.symm
  rw [row]
  decide
theorem root387Step1 : scan edge (search edge 3) 4 [2687] 2687 root387Tail1 winner =
    scan edge (search edge 3) 4 [2687] 2687 root387Tail2 winner := by
  rw [root387Tail1, scan, if_neg (by decide), root387Candidates1_checked]
  change scan edge (search edge 3) 4 [2687] 2687 root387Tail2
    (if 3 ≤ root387Candidates1.length then search edge 3 [2687, 4007] root387Candidates1 6694 winner else winner) = _
  rw [if_neg (by decide)]
def root387Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root387Candidates2_checked : candidates edge 4 2687 4481 root387Tail3 winner.bound = root387Candidates2 := by
  unfold candidates
  have row : edge 4481 = (SortedNatTree.contains FastRows.row4481) := by
    change Data.row4481.contains = (SortedNatTree.contains FastRows.row4481)
    exact FastRows.row4481_checked.symm
  rw [row]
  decide
theorem root387Step2 : scan edge (search edge 3) 4 [2687] 2687 root387Tail2 winner =
    scan edge (search edge 3) 4 [2687] 2687 root387Tail3 winner := by
  rw [root387Tail2, scan, if_neg (by decide), root387Candidates2_checked]
  change scan edge (search edge 3) 4 [2687] 2687 root387Tail3
    (if 3 ≤ root387Candidates2.length then search edge 3 [2687, 4481] root387Candidates2 7168 winner else winner) = _
  rw [if_neg (by decide)]
def root387Candidates3 : List Nat := [14447]
set_option maxRecDepth 8192 in
theorem root387Candidates3_checked : candidates edge 4 2687 5237 root387Tail4 winner.bound = root387Candidates3 := by
  unfold candidates
  have row : edge 5237 = (SortedNatTree.contains FastRows.row5237) := by
    change Data.row5237.contains = (SortedNatTree.contains FastRows.row5237)
    exact FastRows.row5237_checked.symm
  rw [row]
  decide
theorem root387Step3 : scan edge (search edge 3) 4 [2687] 2687 root387Tail3 winner =
    scan edge (search edge 3) 4 [2687] 2687 root387Tail4 winner := by
  rw [root387Tail3, scan, if_neg (by decide), root387Candidates3_checked]
  change scan edge (search edge 3) 4 [2687] 2687 root387Tail4
    (if 3 ≤ root387Candidates3.length then search edge 3 [2687, 5237] root387Candidates3 7924 winner else winner) = _
  rw [if_neg (by decide)]
theorem root387Scan4 : scan edge (search edge 3) 4 [2687] 2687 root387Tail4 winner = winner := by rw [root387Tail4, scan, if_pos (by decide)]
theorem root387Scan3 : scan edge (search edge 3) 4 [2687] 2687 root387Tail3 winner = winner :=
  (root387Step3).trans root387Scan4
theorem root387Scan2 : scan edge (search edge 3) 4 [2687] 2687 root387Tail2 winner = winner :=
  (root387Step2).trans root387Scan3
theorem root387Scan1 : scan edge (search edge 3) 4 [2687] 2687 root387Tail1 winner = winner :=
  (root387Step1).trans root387Scan2
theorem root387Scan0 : scan edge (search edge 3) 4 [2687] 2687 root387Tail0 winner = winner :=
  (root387Step0).trans root387Scan1
set_option maxRecDepth 4096 in
theorem root387Branch : search edge 4 [2687] root387Candidates 2687 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2687] 2687 root387Tail0 winner = winner
  exact root387Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
