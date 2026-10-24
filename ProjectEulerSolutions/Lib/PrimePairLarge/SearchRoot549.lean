import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root549Candidates : List Nat := Data.row4003
set_option maxRecDepth 8192 in
theorem root549Candidates_checked : candidates edge 5 0 4003 tail550 winner.bound = root549Candidates := by
  apply candidates_eq_row edge 5 0 4003 tail550 root549Candidates winner.bound tail550_ordered
  · decide
  · decide
  · funext q
    rfl
def root549Tail33 : List Nat := []
def root549Tail32 : List Nat := 19753 :: root549Tail33
def root549Tail31 : List Nat := 19681 :: root549Tail32
def root549Tail30 : List Nat := 19477 :: root549Tail31
def root549Tail29 : List Nat := 18973 :: root549Tail30
def root549Tail28 : List Nat := 18637 :: root549Tail29
def root549Tail27 : List Nat := 18439 :: root549Tail28
def root549Tail26 : List Nat := 18229 :: root549Tail27
def root549Tail25 : List Nat := 18199 :: root549Tail26
def root549Tail24 : List Nat := 16057 :: root549Tail25
def root549Tail23 : List Nat := 15373 :: root549Tail24
def root549Tail22 : List Nat := 15259 :: root549Tail23
def root549Tail21 : List Nat := 14503 :: root549Tail22
def root549Tail20 : List Nat := 13063 :: root549Tail21
def root549Tail19 : List Nat := 12967 :: root549Tail20
def root549Tail18 : List Nat := 12547 :: root549Tail19
def root549Tail17 : List Nat := 12097 :: root549Tail18
def root549Tail16 : List Nat := 11953 :: root549Tail17
def root549Tail15 : List Nat := 11161 :: root549Tail16
def root549Tail14 : List Nat := 10303 :: root549Tail15
def root549Tail13 : List Nat := 9787 :: root549Tail14
def root549Tail12 : List Nat := 9511 :: root549Tail13
def root549Tail11 : List Nat := 9049 :: root549Tail12
def root549Tail10 : List Nat := 8839 :: root549Tail11
def root549Tail9 : List Nat := 8629 :: root549Tail10
def root549Tail8 : List Nat := 8539 :: root549Tail9
def root549Tail7 : List Nat := 8527 :: root549Tail8
def root549Tail6 : List Nat := 6163 :: root549Tail7
def root549Tail5 : List Nat := 5851 :: root549Tail6
def root549Tail4 : List Nat := 5503 :: root549Tail5
def root549Tail3 : List Nat := 5431 :: root549Tail4
def root549Tail2 : List Nat := 4723 :: root549Tail3
def root549Tail1 : List Nat := 4597 :: root549Tail2
def root549Tail0 : List Nat := 4567 :: root549Tail1
def root549Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root549Candidates0_checked : candidates edge 4 4003 4567 root549Tail1 winner.bound = root549Candidates0 := by
  unfold candidates
  have row : edge 4567 = (SortedNatTree.contains FastRows.row4567) := by
    change Data.row4567.contains = (SortedNatTree.contains FastRows.row4567)
    exact FastRows.row4567_checked.symm
  rw [row]
  decide
theorem root549Step0 : scan edge (search edge 3) 4 [4003] 4003 root549Tail0 winner =
    scan edge (search edge 3) 4 [4003] 4003 root549Tail1 winner := by
  rw [root549Tail0, scan, if_neg (by decide), root549Candidates0_checked]
  change scan edge (search edge 3) 4 [4003] 4003 root549Tail1
    (if 3 ≤ root549Candidates0.length then search edge 3 [4003, 4567] root549Candidates0 8570 winner else winner) = _
  rw [if_neg (by decide)]
def root549Candidates1 : List Nat := [15259]
set_option maxRecDepth 8192 in
theorem root549Candidates1_checked : candidates edge 4 4003 4597 root549Tail2 winner.bound = root549Candidates1 := by
  unfold candidates
  have row : edge 4597 = (SortedNatTree.contains FastRows.row4597) := by
    change Data.row4597.contains = (SortedNatTree.contains FastRows.row4597)
    exact FastRows.row4597_checked.symm
  rw [row]
  decide
theorem root549Step1 : scan edge (search edge 3) 4 [4003] 4003 root549Tail1 winner =
    scan edge (search edge 3) 4 [4003] 4003 root549Tail2 winner := by
  rw [root549Tail1, scan, if_neg (by decide), root549Candidates1_checked]
  change scan edge (search edge 3) 4 [4003] 4003 root549Tail2
    (if 3 ≤ root549Candidates1.length then search edge 3 [4003, 4597] root549Candidates1 8600 winner else winner) = _
  rw [if_neg (by decide)]
def root549Candidates2 : List Nat := [8839]
set_option maxRecDepth 8192 in
theorem root549Candidates2_checked : candidates edge 4 4003 4723 root549Tail3 winner.bound = root549Candidates2 := by
  unfold candidates
  have row : edge 4723 = (SortedNatTree.contains FastRows.row4723) := by
    change Data.row4723.contains = (SortedNatTree.contains FastRows.row4723)
    exact FastRows.row4723_checked.symm
  rw [row]
  decide
theorem root549Step2 : scan edge (search edge 3) 4 [4003] 4003 root549Tail2 winner =
    scan edge (search edge 3) 4 [4003] 4003 root549Tail3 winner := by
  rw [root549Tail2, scan, if_neg (by decide), root549Candidates2_checked]
  change scan edge (search edge 3) 4 [4003] 4003 root549Tail3
    (if 3 ≤ root549Candidates2.length then search edge 3 [4003, 4723] root549Candidates2 8726 winner else winner) = _
  rw [if_neg (by decide)]
def root549Candidates3 : List Nat := [15259]
set_option maxRecDepth 8192 in
theorem root549Candidates3_checked : candidates edge 4 4003 5431 root549Tail4 winner.bound = root549Candidates3 := by
  unfold candidates
  have row : edge 5431 = (SortedNatTree.contains FastRows.row5431) := by
    change Data.row5431.contains = (SortedNatTree.contains FastRows.row5431)
    exact FastRows.row5431_checked.symm
  rw [row]
  decide
theorem root549Step3 : scan edge (search edge 3) 4 [4003] 4003 root549Tail3 winner =
    scan edge (search edge 3) 4 [4003] 4003 root549Tail4 winner := by
  rw [root549Tail3, scan, if_neg (by decide), root549Candidates3_checked]
  change scan edge (search edge 3) 4 [4003] 4003 root549Tail4
    (if 3 ≤ root549Candidates3.length then search edge 3 [4003, 5431] root549Candidates3 9434 winner else winner) = _
  rw [if_neg (by decide)]
def root549Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root549Candidates4_checked : candidates edge 4 4003 5503 root549Tail5 winner.bound = root549Candidates4 := by
  unfold candidates
  have row : edge 5503 = (SortedNatTree.contains FastRows.row5503) := by
    change Data.row5503.contains = (SortedNatTree.contains FastRows.row5503)
    exact FastRows.row5503_checked.symm
  rw [row]
  decide
theorem root549Step4 : scan edge (search edge 3) 4 [4003] 4003 root549Tail4 winner =
    scan edge (search edge 3) 4 [4003] 4003 root549Tail5 winner := by
  rw [root549Tail4, scan, if_neg (by decide), root549Candidates4_checked]
  change scan edge (search edge 3) 4 [4003] 4003 root549Tail5
    (if 3 ≤ root549Candidates4.length then search edge 3 [4003, 5503] root549Candidates4 9506 winner else winner) = _
  rw [if_neg (by decide)]
theorem root549Scan5 : scan edge (search edge 3) 4 [4003] 4003 root549Tail5 winner = winner := by rw [root549Tail5, scan, if_pos (by decide)]
theorem root549Scan4 : scan edge (search edge 3) 4 [4003] 4003 root549Tail4 winner = winner :=
  (root549Step4).trans root549Scan5
theorem root549Scan3 : scan edge (search edge 3) 4 [4003] 4003 root549Tail3 winner = winner :=
  (root549Step3).trans root549Scan4
theorem root549Scan2 : scan edge (search edge 3) 4 [4003] 4003 root549Tail2 winner = winner :=
  (root549Step2).trans root549Scan3
theorem root549Scan1 : scan edge (search edge 3) 4 [4003] 4003 root549Tail1 winner = winner :=
  (root549Step1).trans root549Scan2
theorem root549Scan0 : scan edge (search edge 3) 4 [4003] 4003 root549Tail0 winner = winner :=
  (root549Step0).trans root549Scan1
set_option maxRecDepth 4096 in
theorem root549Branch : search edge 4 [4003] root549Candidates 4003 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4003] 4003 root549Tail0 winner = winner
  exact root549Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
