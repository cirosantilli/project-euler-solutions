import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root558Candidates : List Nat := Data.row4073
set_option maxRecDepth 8192 in
theorem root558Candidates_checked : candidates edge 5 0 4073 tail559 winner.bound = root558Candidates := by
  apply candidates_eq_row edge 5 0 4073 tail559 root558Candidates winner.bound tail559_ordered
  · decide
  · decide
  · funext q
    rfl
def root558Tail35 : List Nat := []
def root558Tail34 : List Nat := 21017 :: root558Tail35
def root558Tail33 : List Nat := 20261 :: root558Tail34
def root558Tail32 : List Nat := 20063 :: root558Tail33
def root558Tail31 : List Nat := 19697 :: root558Tail32
def root558Tail30 : List Nat := 19079 :: root558Tail31
def root558Tail29 : List Nat := 18731 :: root558Tail30
def root558Tail28 : List Nat := 17837 :: root558Tail29
def root558Tail27 : List Nat := 17627 :: root558Tail28
def root558Tail26 : List Nat := 17123 :: root558Tail27
def root558Tail25 : List Nat := 16901 :: root558Tail26
def root558Tail24 : List Nat := 16829 :: root558Tail25
def root558Tail23 : List Nat := 16493 :: root558Tail24
def root558Tail22 : List Nat := 15761 :: root558Tail23
def root558Tail21 : List Nat := 15383 :: root558Tail22
def root558Tail20 : List Nat := 14969 :: root558Tail21
def root558Tail19 : List Nat := 14939 :: root558Tail20
def root558Tail18 : List Nat := 13763 :: root558Tail19
def root558Tail17 : List Nat := 13553 :: root558Tail18
def root558Tail16 : List Nat := 13037 :: root558Tail17
def root558Tail15 : List Nat := 13007 :: root558Tail16
def root558Tail14 : List Nat := 10301 :: root558Tail15
def root558Tail13 : List Nat := 9539 :: root558Tail14
def root558Tail12 : List Nat := 9341 :: root558Tail13
def root558Tail11 : List Nat := 9029 :: root558Tail12
def root558Tail10 : List Nat := 8861 :: root558Tail11
def root558Tail9 : List Nat := 8627 :: root558Tail10
def root558Tail8 : List Nat := 8573 :: root558Tail9
def root558Tail7 : List Nat := 7901 :: root558Tail8
def root558Tail6 : List Nat := 6977 :: root558Tail7
def root558Tail5 : List Nat := 6389 :: root558Tail6
def root558Tail4 : List Nat := 5393 :: root558Tail5
def root558Tail3 : List Nat := 5087 :: root558Tail4
def root558Tail2 : List Nat := 5003 :: root558Tail3
def root558Tail1 : List Nat := 4877 :: root558Tail2
def root558Tail0 : List Nat := 4481 :: root558Tail1
def root558Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root558Candidates0_checked : candidates edge 4 4073 4481 root558Tail1 winner.bound = root558Candidates0 := by
  unfold candidates
  have row : edge 4481 = (SortedNatTree.contains FastRows.row4481) := by
    change Data.row4481.contains = (SortedNatTree.contains FastRows.row4481)
    exact FastRows.row4481_checked.symm
  rw [row]
  decide
theorem root558Step0 : scan edge (search edge 3) 4 [4073] 4073 root558Tail0 winner =
    scan edge (search edge 3) 4 [4073] 4073 root558Tail1 winner := by
  rw [root558Tail0, scan, if_neg (by decide), root558Candidates0_checked]
  change scan edge (search edge 3) 4 [4073] 4073 root558Tail1
    (if 3 ≤ root558Candidates0.length then search edge 3 [4073, 4481] root558Candidates0 8554 winner else winner) = _
  rw [if_neg (by decide)]
def root558Candidates1 : List Nat := [6977, 8573, 14939]
set_option maxRecDepth 8192 in
theorem root558Candidates1_checked : candidates edge 4 4073 4877 root558Tail2 winner.bound = root558Candidates1 := by
  unfold candidates
  have row : edge 4877 = (SortedNatTree.contains FastRows.row4877) := by
    change Data.row4877.contains = (SortedNatTree.contains FastRows.row4877)
    exact FastRows.row4877_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root558Child1 : search edge 3 [4073, 4877] root558Candidates1 8950 winner = winner := by decide
theorem root558Step1 : scan edge (search edge 3) 4 [4073] 4073 root558Tail1 winner =
    scan edge (search edge 3) 4 [4073] 4073 root558Tail2 winner := by
  rw [root558Tail1, scan, if_neg (by decide), root558Candidates1_checked]
  change scan edge (search edge 3) 4 [4073] 4073 root558Tail2
    (if 3 ≤ root558Candidates1.length then search edge 3 [4073, 4877] root558Candidates1 8950 winner else winner) = _
  rw [if_pos (by decide), root558Child1]
def root558Candidates2 : List Nat := [5087, 9029, 14939]
set_option maxRecDepth 8192 in
theorem root558Candidates2_checked : candidates edge 4 4073 5003 root558Tail3 winner.bound = root558Candidates2 := by
  unfold candidates
  have row : edge 5003 = (SortedNatTree.contains FastRows.row5003) := by
    change Data.row5003.contains = (SortedNatTree.contains FastRows.row5003)
    exact FastRows.row5003_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root558Child2 : search edge 3 [4073, 5003] root558Candidates2 9076 winner = winner := by decide
theorem root558Step2 : scan edge (search edge 3) 4 [4073] 4073 root558Tail2 winner =
    scan edge (search edge 3) 4 [4073] 4073 root558Tail3 winner := by
  rw [root558Tail2, scan, if_neg (by decide), root558Candidates2_checked]
  change scan edge (search edge 3) 4 [4073] 4073 root558Tail3
    (if 3 ≤ root558Candidates2.length then search edge 3 [4073, 5003] root558Candidates2 9076 winner else winner) = _
  rw [if_pos (by decide), root558Child2]
def root558Candidates3 : List Nat := [9539]
set_option maxRecDepth 8192 in
theorem root558Candidates3_checked : candidates edge 4 4073 5087 root558Tail4 winner.bound = root558Candidates3 := by
  unfold candidates
  have row : edge 5087 = (SortedNatTree.contains FastRows.row5087) := by
    change Data.row5087.contains = (SortedNatTree.contains FastRows.row5087)
    exact FastRows.row5087_checked.symm
  rw [row]
  decide
theorem root558Step3 : scan edge (search edge 3) 4 [4073] 4073 root558Tail3 winner =
    scan edge (search edge 3) 4 [4073] 4073 root558Tail4 winner := by
  rw [root558Tail3, scan, if_neg (by decide), root558Candidates3_checked]
  change scan edge (search edge 3) 4 [4073] 4073 root558Tail4
    (if 3 ≤ root558Candidates3.length then search edge 3 [4073, 5087] root558Candidates3 9160 winner else winner) = _
  rw [if_neg (by decide)]
def root558Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root558Candidates4_checked : candidates edge 4 4073 5393 root558Tail5 winner.bound = root558Candidates4 := by
  unfold candidates
  have row : edge 5393 = (SortedNatTree.contains FastRows.row5393) := by
    change Data.row5393.contains = (SortedNatTree.contains FastRows.row5393)
    exact FastRows.row5393_checked.symm
  rw [row]
  decide
theorem root558Step4 : scan edge (search edge 3) 4 [4073] 4073 root558Tail4 winner =
    scan edge (search edge 3) 4 [4073] 4073 root558Tail5 winner := by
  rw [root558Tail4, scan, if_neg (by decide), root558Candidates4_checked]
  change scan edge (search edge 3) 4 [4073] 4073 root558Tail5
    (if 3 ≤ root558Candidates4.length then search edge 3 [4073, 5393] root558Candidates4 9466 winner else winner) = _
  rw [if_neg (by decide)]
theorem root558Scan5 : scan edge (search edge 3) 4 [4073] 4073 root558Tail5 winner = winner := by rw [root558Tail5, scan, if_pos (by decide)]
theorem root558Scan4 : scan edge (search edge 3) 4 [4073] 4073 root558Tail4 winner = winner :=
  (root558Step4).trans root558Scan5
theorem root558Scan3 : scan edge (search edge 3) 4 [4073] 4073 root558Tail3 winner = winner :=
  (root558Step3).trans root558Scan4
theorem root558Scan2 : scan edge (search edge 3) 4 [4073] 4073 root558Tail2 winner = winner :=
  (root558Step2).trans root558Scan3
theorem root558Scan1 : scan edge (search edge 3) 4 [4073] 4073 root558Tail1 winner = winner :=
  (root558Step1).trans root558Scan2
theorem root558Scan0 : scan edge (search edge 3) 4 [4073] 4073 root558Tail0 winner = winner :=
  (root558Step0).trans root558Scan1
set_option maxRecDepth 4096 in
theorem root558Branch : search edge 4 [4073] root558Candidates 4073 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4073] 4073 root558Tail0 winner = winner
  exact root558Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
