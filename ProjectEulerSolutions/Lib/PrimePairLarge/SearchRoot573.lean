import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root573Candidates : List Nat := Data.row4211
set_option maxRecDepth 8192 in
theorem root573Candidates_checked : candidates edge 5 0 4211 tail574 winner.bound = root573Candidates := by
  apply candidates_eq_row edge 5 0 4211 tail574 root573Candidates winner.bound tail574_ordered
  · decide
  · decide
  · funext q
    rfl
def root573Tail28 : List Nat := []
def root573Tail27 : List Nat := 21737 :: root573Tail28
def root573Tail26 : List Nat := 21647 :: root573Tail27
def root573Tail25 : List Nat := 21227 :: root573Tail26
def root573Tail24 : List Nat := 19973 :: root573Tail25
def root573Tail23 : List Nat := 19763 :: root573Tail24
def root573Tail22 : List Nat := 19289 :: root573Tail23
def root573Tail21 : List Nat := 18671 :: root573Tail22
def root573Tail20 : List Nat := 18041 :: root573Tail21
def root573Tail19 : List Nat := 16811 :: root573Tail20
def root573Tail18 : List Nat := 15803 :: root573Tail19
def root573Tail17 : List Nat := 14717 :: root573Tail18
def root573Tail16 : List Nat := 14669 :: root573Tail17
def root573Tail15 : List Nat := 14033 :: root573Tail16
def root573Tail14 : List Nat := 13619 :: root573Tail15
def root573Tail13 : List Nat := 12227 :: root573Tail14
def root573Tail12 : List Nat := 10973 :: root573Tail13
def root573Tail11 : List Nat := 10607 :: root573Tail12
def root573Tail10 : List Nat := 10601 :: root573Tail11
def root573Tail9 : List Nat := 10181 :: root573Tail10
def root573Tail8 : List Nat := 9887 :: root573Tail9
def root573Tail7 : List Nat := 9221 :: root573Tail8
def root573Tail6 : List Nat := 8963 :: root573Tail7
def root573Tail5 : List Nat := 7703 :: root573Tail6
def root573Tail4 : List Nat := 6911 :: root573Tail5
def root573Tail3 : List Nat := 5693 :: root573Tail4
def root573Tail2 : List Nat := 5483 :: root573Tail3
def root573Tail1 : List Nat := 5351 :: root573Tail2
def root573Tail0 : List Nat := 5237 :: root573Tail1
def root573Candidates0 : List Nat := [5693, 9221]
set_option maxRecDepth 8192 in
theorem root573Candidates0_checked : candidates edge 4 4211 5237 root573Tail1 winner.bound = root573Candidates0 := by
  unfold candidates
  have row : edge 5237 = (SortedNatTree.contains FastRows.row5237) := by
    change Data.row5237.contains = (SortedNatTree.contains FastRows.row5237)
    exact FastRows.row5237_checked.symm
  rw [row]
  decide
theorem root573Step0 : scan edge (search edge 3) 4 [4211] 4211 root573Tail0 winner =
    scan edge (search edge 3) 4 [4211] 4211 root573Tail1 winner := by
  rw [root573Tail0, scan, if_neg (by decide), root573Candidates0_checked]
  change scan edge (search edge 3) 4 [4211] 4211 root573Tail1
    (if 3 ≤ root573Candidates0.length then search edge 3 [4211, 5237] root573Candidates0 9448 winner else winner) = _
  rw [if_neg (by decide)]
def root573Candidates1 : List Nat := [15803]
set_option maxRecDepth 8192 in
theorem root573Candidates1_checked : candidates edge 4 4211 5351 root573Tail2 winner.bound = root573Candidates1 := by
  unfold candidates
  have row : edge 5351 = (SortedNatTree.contains FastRows.row5351) := by
    change Data.row5351.contains = (SortedNatTree.contains FastRows.row5351)
    exact FastRows.row5351_checked.symm
  rw [row]
  decide
theorem root573Step1 : scan edge (search edge 3) 4 [4211] 4211 root573Tail1 winner =
    scan edge (search edge 3) 4 [4211] 4211 root573Tail2 winner := by
  rw [root573Tail1, scan, if_neg (by decide), root573Candidates1_checked]
  change scan edge (search edge 3) 4 [4211] 4211 root573Tail2
    (if 3 ≤ root573Candidates1.length then search edge 3 [4211, 5351] root573Candidates1 9562 winner else winner) = _
  rw [if_neg (by decide)]
theorem root573Scan2 : scan edge (search edge 3) 4 [4211] 4211 root573Tail2 winner = winner := by rw [root573Tail2, scan, if_pos (by decide)]
theorem root573Scan1 : scan edge (search edge 3) 4 [4211] 4211 root573Tail1 winner = winner :=
  (root573Step1).trans root573Scan2
theorem root573Scan0 : scan edge (search edge 3) 4 [4211] 4211 root573Tail0 winner = winner :=
  (root573Step0).trans root573Scan1
set_option maxRecDepth 4096 in
theorem root573Branch : search edge 4 [4211] root573Candidates 4211 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4211] 4211 root573Tail0 winner = winner
  exact root573Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
