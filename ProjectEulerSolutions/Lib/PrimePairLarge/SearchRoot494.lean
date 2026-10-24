import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root494Candidates : List Nat := Data.row3547
set_option maxRecDepth 8192 in
theorem root494Candidates_checked : candidates edge 5 0 3547 tail495 winner.bound = root494Candidates := by
  apply candidates_eq_row edge 5 0 3547 tail495 root494Candidates winner.bound tail495_ordered
  · decide
  · decide
  · funext q
    rfl
def root494Tail34 : List Nat := []
def root494Tail33 : List Nat := 22483 :: root494Tail34
def root494Tail32 : List Nat := 22129 :: root494Tail33
def root494Tail31 : List Nat := 19423 :: root494Tail32
def root494Tail30 : List Nat := 18979 :: root494Tail31
def root494Tail29 : List Nat := 18457 :: root494Tail30
def root494Tail28 : List Nat := 18451 :: root494Tail29
def root494Tail27 : List Nat := 18433 :: root494Tail28
def root494Tail26 : List Nat := 18061 :: root494Tail27
def root494Tail25 : List Nat := 16633 :: root494Tail26
def root494Tail24 : List Nat := 15937 :: root494Tail25
def root494Tail23 : List Nat := 15289 :: root494Tail24
def root494Tail22 : List Nat := 13537 :: root494Tail23
def root494Tail21 : List Nat := 12907 :: root494Tail22
def root494Tail20 : List Nat := 12703 :: root494Tail21
def root494Tail19 : List Nat := 12637 :: root494Tail20
def root494Tail18 : List Nat := 12343 :: root494Tail19
def root494Tail17 : List Nat := 11839 :: root494Tail18
def root494Tail16 : List Nat := 10333 :: root494Tail17
def root494Tail15 : List Nat := 9817 :: root494Tail16
def root494Tail14 : List Nat := 9733 :: root494Tail15
def root494Tail13 : List Nat := 9319 :: root494Tail14
def root494Tail12 : List Nat := 9043 :: root494Tail13
def root494Tail11 : List Nat := 8461 :: root494Tail12
def root494Tail10 : List Nat := 8431 :: root494Tail11
def root494Tail9 : List Nat := 7993 :: root494Tail10
def root494Tail8 : List Nat := 7243 :: root494Tail9
def root494Tail7 : List Nat := 7213 :: root494Tail8
def root494Tail6 : List Nat := 6709 :: root494Tail7
def root494Tail5 : List Nat := 6481 :: root494Tail6
def root494Tail4 : List Nat := 6151 :: root494Tail5
def root494Tail3 : List Nat := 5827 :: root494Tail4
def root494Tail2 : List Nat := 5449 :: root494Tail3
def root494Tail1 : List Nat := 4357 :: root494Tail2
def root494Tail0 : List Nat := 3643 :: root494Tail1
def root494Candidates0 : List Nat := [5449, 9817]
set_option maxRecDepth 8192 in
theorem root494Candidates0_checked : candidates edge 4 3547 3643 root494Tail1 winner.bound = root494Candidates0 := by
  unfold candidates
  have row : edge 3643 = (SortedNatTree.contains FastRows.row3643) := by
    change Data.row3643.contains = (SortedNatTree.contains FastRows.row3643)
    exact FastRows.row3643_checked.symm
  rw [row]
  decide
theorem root494Step0 : scan edge (search edge 3) 4 [3547] 3547 root494Tail0 winner =
    scan edge (search edge 3) 4 [3547] 3547 root494Tail1 winner := by
  rw [root494Tail0, scan, if_neg (by decide), root494Candidates0_checked]
  change scan edge (search edge 3) 4 [3547] 3547 root494Tail1
    (if 3 ≤ root494Candidates0.length then search edge 3 [3547, 3643] root494Candidates0 7190 winner else winner) = _
  rw [if_neg (by decide)]
def root494Candidates1 : List Nat := [5449]
set_option maxRecDepth 8192 in
theorem root494Candidates1_checked : candidates edge 4 3547 4357 root494Tail2 winner.bound = root494Candidates1 := by
  unfold candidates
  have row : edge 4357 = (SortedNatTree.contains FastRows.row4357) := by
    change Data.row4357.contains = (SortedNatTree.contains FastRows.row4357)
    exact FastRows.row4357_checked.symm
  rw [row]
  decide
theorem root494Step1 : scan edge (search edge 3) 4 [3547] 3547 root494Tail1 winner =
    scan edge (search edge 3) 4 [3547] 3547 root494Tail2 winner := by
  rw [root494Tail1, scan, if_neg (by decide), root494Candidates1_checked]
  change scan edge (search edge 3) 4 [3547] 3547 root494Tail2
    (if 3 ≤ root494Candidates1.length then search edge 3 [3547, 4357] root494Candidates1 7904 winner else winner) = _
  rw [if_neg (by decide)]
def root494Candidates2 : List Nat := [7243, 9817]
set_option maxRecDepth 8192 in
theorem root494Candidates2_checked : candidates edge 4 3547 5449 root494Tail3 winner.bound = root494Candidates2 := by
  unfold candidates
  have row : edge 5449 = (SortedNatTree.contains FastRows.row5449) := by
    change Data.row5449.contains = (SortedNatTree.contains FastRows.row5449)
    exact FastRows.row5449_checked.symm
  rw [row]
  decide
theorem root494Step2 : scan edge (search edge 3) 4 [3547] 3547 root494Tail2 winner =
    scan edge (search edge 3) 4 [3547] 3547 root494Tail3 winner := by
  rw [root494Tail2, scan, if_neg (by decide), root494Candidates2_checked]
  change scan edge (search edge 3) 4 [3547] 3547 root494Tail3
    (if 3 ≤ root494Candidates2.length then search edge 3 [3547, 5449] root494Candidates2 8996 winner else winner) = _
  rw [if_neg (by decide)]
theorem root494Scan3 : scan edge (search edge 3) 4 [3547] 3547 root494Tail3 winner = winner := by rw [root494Tail3, scan, if_pos (by decide)]
theorem root494Scan2 : scan edge (search edge 3) 4 [3547] 3547 root494Tail2 winner = winner :=
  (root494Step2).trans root494Scan3
theorem root494Scan1 : scan edge (search edge 3) 4 [3547] 3547 root494Tail1 winner = winner :=
  (root494Step1).trans root494Scan2
theorem root494Scan0 : scan edge (search edge 3) 4 [3547] 3547 root494Tail0 winner = winner :=
  (root494Step0).trans root494Scan1
set_option maxRecDepth 4096 in
theorem root494Branch : search edge 4 [3547] root494Candidates 3547 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3547] 3547 root494Tail0 winner = winner
  exact root494Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
