import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root344Candidates : List Nat := Data.row2341
set_option maxRecDepth 8192 in
theorem root344Candidates_checked : candidates edge 5 0 2341 tail345 winner.bound = root344Candidates := by
  apply candidates_eq_row edge 5 0 2341 tail345 root344Candidates winner.bound tail345_ordered
  · decide
  · decide
  · funext q
    rfl
def root344Tail36 : List Nat := []
def root344Tail35 : List Nat := 23557 :: root344Tail36
def root344Tail34 : List Nat := 22549 :: root344Tail35
def root344Tail33 : List Nat := 21313 :: root344Tail34
def root344Tail32 : List Nat := 21187 :: root344Tail33
def root344Tail31 : List Nat := 20353 :: root344Tail32
def root344Tail30 : List Nat := 19441 :: root344Tail31
def root344Tail29 : List Nat := 19051 :: root344Tail30
def root344Tail28 : List Nat := 18433 :: root344Tail29
def root344Tail27 : List Nat := 15907 :: root344Tail28
def root344Tail26 : List Nat := 15901 :: root344Tail27
def root344Tail25 : List Nat := 15727 :: root344Tail26
def root344Tail24 : List Nat := 15439 :: root344Tail25
def root344Tail23 : List Nat := 14563 :: root344Tail24
def root344Tail22 : List Nat := 14479 :: root344Tail23
def root344Tail21 : List Nat := 13309 :: root344Tail22
def root344Tail20 : List Nat := 13177 :: root344Tail21
def root344Tail19 : List Nat := 12409 :: root344Tail20
def root344Tail18 : List Nat := 12379 :: root344Tail19
def root344Tail17 : List Nat := 11953 :: root344Tail18
def root344Tail16 : List Nat := 11701 :: root344Tail17
def root344Tail15 : List Nat := 11083 :: root344Tail16
def root344Tail14 : List Nat := 10477 :: root344Tail15
def root344Tail13 : List Nat := 10099 :: root344Tail14
def root344Tail12 : List Nat := 9859 :: root344Tail13
def root344Tail11 : List Nat := 9679 :: root344Tail12
def root344Tail10 : List Nat := 9511 :: root344Tail11
def root344Tail9 : List Nat := 8377 :: root344Tail10
def root344Tail8 : List Nat := 8179 :: root344Tail9
def root344Tail7 : List Nat := 7879 :: root344Tail8
def root344Tail6 : List Nat := 7489 :: root344Tail7
def root344Tail5 : List Nat := 6571 :: root344Tail6
def root344Tail4 : List Nat := 5107 :: root344Tail5
def root344Tail3 : List Nat := 3967 :: root344Tail4
def root344Tail2 : List Nat := 3373 :: root344Tail3
def root344Tail1 : List Nat := 3259 :: root344Tail2
def root344Tail0 : List Nat := 2833 :: root344Tail1
def root344Candidates0 : List Nat := [10099]
set_option maxRecDepth 8192 in
theorem root344Candidates0_checked : candidates edge 4 2341 2833 root344Tail1 winner.bound = root344Candidates0 := by
  unfold candidates
  have row : edge 2833 = (SortedNatTree.contains FastRows.row2833) := by
    change Data.row2833.contains = (SortedNatTree.contains FastRows.row2833)
    exact FastRows.row2833_checked.symm
  rw [row]
  decide
theorem root344Step0 : scan edge (search edge 3) 4 [2341] 2341 root344Tail0 winner =
    scan edge (search edge 3) 4 [2341] 2341 root344Tail1 winner := by
  rw [root344Tail0, scan, if_neg (by decide), root344Candidates0_checked]
  change scan edge (search edge 3) 4 [2341] 2341 root344Tail1
    (if 3 ≤ root344Candidates0.length then search edge 3 [2341, 2833] root344Candidates0 5174 winner else winner) = _
  rw [if_neg (by decide)]
def root344Candidates1 : List Nat := [5107, 11953]
set_option maxRecDepth 8192 in
theorem root344Candidates1_checked : candidates edge 4 2341 3259 root344Tail2 winner.bound = root344Candidates1 := by
  unfold candidates
  have row : edge 3259 = (SortedNatTree.contains FastRows.row3259) := by
    change Data.row3259.contains = (SortedNatTree.contains FastRows.row3259)
    exact FastRows.row3259_checked.symm
  rw [row]
  decide
theorem root344Step1 : scan edge (search edge 3) 4 [2341] 2341 root344Tail1 winner =
    scan edge (search edge 3) 4 [2341] 2341 root344Tail2 winner := by
  rw [root344Tail1, scan, if_neg (by decide), root344Candidates1_checked]
  change scan edge (search edge 3) 4 [2341] 2341 root344Tail2
    (if 3 ≤ root344Candidates1.length then search edge 3 [2341, 3259] root344Candidates1 5600 winner else winner) = _
  rw [if_neg (by decide)]
def root344Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root344Candidates2_checked : candidates edge 4 2341 3373 root344Tail3 winner.bound = root344Candidates2 := by
  unfold candidates
  have row : edge 3373 = (SortedNatTree.contains FastRows.row3373) := by
    change Data.row3373.contains = (SortedNatTree.contains FastRows.row3373)
    exact FastRows.row3373_checked.symm
  rw [row]
  decide
theorem root344Step2 : scan edge (search edge 3) 4 [2341] 2341 root344Tail2 winner =
    scan edge (search edge 3) 4 [2341] 2341 root344Tail3 winner := by
  rw [root344Tail2, scan, if_neg (by decide), root344Candidates2_checked]
  change scan edge (search edge 3) 4 [2341] 2341 root344Tail3
    (if 3 ≤ root344Candidates2.length then search edge 3 [2341, 3373] root344Candidates2 5714 winner else winner) = _
  rw [if_neg (by decide)]
def root344Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root344Candidates3_checked : candidates edge 4 2341 3967 root344Tail4 winner.bound = root344Candidates3 := by
  unfold candidates
  have row : edge 3967 = (SortedNatTree.contains FastRows.row3967) := by
    change Data.row3967.contains = (SortedNatTree.contains FastRows.row3967)
    exact FastRows.row3967_checked.symm
  rw [row]
  decide
theorem root344Step3 : scan edge (search edge 3) 4 [2341] 2341 root344Tail3 winner =
    scan edge (search edge 3) 4 [2341] 2341 root344Tail4 winner := by
  rw [root344Tail3, scan, if_neg (by decide), root344Candidates3_checked]
  change scan edge (search edge 3) 4 [2341] 2341 root344Tail4
    (if 3 ≤ root344Candidates3.length then search edge 3 [2341, 3967] root344Candidates3 6308 winner else winner) = _
  rw [if_neg (by decide)]
def root344Candidates4 : List Nat := [7879, 10099]
set_option maxRecDepth 8192 in
theorem root344Candidates4_checked : candidates edge 4 2341 5107 root344Tail5 winner.bound = root344Candidates4 := by
  unfold candidates
  have row : edge 5107 = (SortedNatTree.contains FastRows.row5107) := by
    change Data.row5107.contains = (SortedNatTree.contains FastRows.row5107)
    exact FastRows.row5107_checked.symm
  rw [row]
  decide
theorem root344Step4 : scan edge (search edge 3) 4 [2341] 2341 root344Tail4 winner =
    scan edge (search edge 3) 4 [2341] 2341 root344Tail5 winner := by
  rw [root344Tail4, scan, if_neg (by decide), root344Candidates4_checked]
  change scan edge (search edge 3) 4 [2341] 2341 root344Tail5
    (if 3 ≤ root344Candidates4.length then search edge 3 [2341, 5107] root344Candidates4 7448 winner else winner) = _
  rw [if_neg (by decide)]
theorem root344Scan5 : scan edge (search edge 3) 4 [2341] 2341 root344Tail5 winner = winner := by rw [root344Tail5, scan, if_pos (by decide)]
theorem root344Scan4 : scan edge (search edge 3) 4 [2341] 2341 root344Tail4 winner = winner :=
  (root344Step4).trans root344Scan5
theorem root344Scan3 : scan edge (search edge 3) 4 [2341] 2341 root344Tail3 winner = winner :=
  (root344Step3).trans root344Scan4
theorem root344Scan2 : scan edge (search edge 3) 4 [2341] 2341 root344Tail2 winner = winner :=
  (root344Step2).trans root344Scan3
theorem root344Scan1 : scan edge (search edge 3) 4 [2341] 2341 root344Tail1 winner = winner :=
  (root344Step1).trans root344Scan2
theorem root344Scan0 : scan edge (search edge 3) 4 [2341] 2341 root344Tail0 winner = winner :=
  (root344Step0).trans root344Scan1
set_option maxRecDepth 4096 in
theorem root344Branch : search edge 4 [2341] root344Candidates 2341 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2341] 2341 root344Tail0 winner = winner
  exact root344Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
