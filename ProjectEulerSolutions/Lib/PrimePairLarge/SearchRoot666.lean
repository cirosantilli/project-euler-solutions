import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root666Candidates : List Nat := Data.row4999
set_option maxRecDepth 8192 in
theorem root666Candidates_checked : candidates edge 5 0 4999 tail667 winner.bound = root666Candidates := by
  apply candidates_eq_row edge 5 0 4999 tail667 root666Candidates winner.bound tail667_ordered
  · decide
  · decide
  · funext q
    rfl
def root666Tail27 : List Nat := []
def root666Tail26 : List Nat := 20143 :: root666Tail27
def root666Tail25 : List Nat := 19447 :: root666Tail26
def root666Tail24 : List Nat := 19249 :: root666Tail25
def root666Tail23 : List Nat := 19219 :: root666Tail24
def root666Tail22 : List Nat := 18427 :: root666Tail23
def root666Tail21 : List Nat := 16927 :: root666Tail22
def root666Tail20 : List Nat := 15907 :: root666Tail21
def root666Tail19 : List Nat := 15817 :: root666Tail20
def root666Tail18 : List Nat := 14143 :: root666Tail19
def root666Tail17 : List Nat := 13537 :: root666Tail18
def root666Tail16 : List Nat := 13159 :: root666Tail17
def root666Tail15 : List Nat := 11701 :: root666Tail16
def root666Tail14 : List Nat := 11491 :: root666Tail15
def root666Tail13 : List Nat := 11047 :: root666Tail14
def root666Tail12 : List Nat := 10861 :: root666Tail13
def root666Tail11 : List Nat := 10837 :: root666Tail12
def root666Tail10 : List Nat := 10429 :: root666Tail11
def root666Tail9 : List Nat := 9883 :: root666Tail10
def root666Tail8 : List Nat := 8779 :: root666Tail9
def root666Tail7 : List Nat := 7309 :: root666Tail8
def root666Tail6 : List Nat := 7069 :: root666Tail7
def root666Tail5 : List Nat := 6829 :: root666Tail6
def root666Tail4 : List Nat := 6763 :: root666Tail5
def root666Tail3 : List Nat := 6367 :: root666Tail4
def root666Tail2 : List Nat := 5791 :: root666Tail3
def root666Tail1 : List Nat := 5119 :: root666Tail2
def root666Tail0 : List Nat := 5107 :: root666Tail1
def root666Candidates0 : List Nat := [6763]
set_option maxRecDepth 8192 in
theorem root666Candidates0_checked : candidates edge 4 4999 5107 root666Tail1 winner.bound = root666Candidates0 := by
  unfold candidates
  have row : edge 5107 = (SortedNatTree.contains FastRows.row5107) := by
    change Data.row5107.contains = (SortedNatTree.contains FastRows.row5107)
    exact FastRows.row5107_checked.symm
  rw [row]
  decide
theorem root666Step0 : scan edge (search edge 3) 4 [4999] 4999 root666Tail0 winner =
    scan edge (search edge 3) 4 [4999] 4999 root666Tail1 winner := by
  rw [root666Tail0, scan, if_neg (by decide), root666Candidates0_checked]
  change scan edge (search edge 3) 4 [4999] 4999 root666Tail1
    (if 3 ≤ root666Candidates0.length then search edge 3 [4999, 5107] root666Candidates0 10106 winner else winner) = _
  rw [if_neg (by decide)]
def root666Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root666Candidates1_checked : candidates edge 4 4999 5119 root666Tail2 winner.bound = root666Candidates1 := by
  unfold candidates
  have row : edge 5119 = (SortedNatTree.contains FastRows.row5119) := by
    change Data.row5119.contains = (SortedNatTree.contains FastRows.row5119)
    exact FastRows.row5119_checked.symm
  rw [row]
  decide
theorem root666Step1 : scan edge (search edge 3) 4 [4999] 4999 root666Tail1 winner =
    scan edge (search edge 3) 4 [4999] 4999 root666Tail2 winner := by
  rw [root666Tail1, scan, if_neg (by decide), root666Candidates1_checked]
  change scan edge (search edge 3) 4 [4999] 4999 root666Tail2
    (if 3 ≤ root666Candidates1.length then search edge 3 [4999, 5119] root666Candidates1 10118 winner else winner) = _
  rw [if_neg (by decide)]
theorem root666Scan2 : scan edge (search edge 3) 4 [4999] 4999 root666Tail2 winner = winner := by rw [root666Tail2, scan, if_pos (by decide)]
theorem root666Scan1 : scan edge (search edge 3) 4 [4999] 4999 root666Tail1 winner = winner :=
  (root666Step1).trans root666Scan2
theorem root666Scan0 : scan edge (search edge 3) 4 [4999] 4999 root666Tail0 winner = winner :=
  (root666Step0).trans root666Scan1
set_option maxRecDepth 4096 in
theorem root666Branch : search edge 4 [4999] root666Candidates 4999 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4999] 4999 root666Tail0 winner = winner
  exact root666Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
