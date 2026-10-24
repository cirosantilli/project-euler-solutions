import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root452Candidates : List Nat := Data.row3217
set_option maxRecDepth 8192 in
theorem root452Candidates_checked : candidates edge 5 0 3217 tail453 winner.bound = root452Candidates := by
  apply candidates_eq_row edge 5 0 3217 tail453 root452Candidates winner.bound tail453_ordered
  · decide
  · decide
  · funext q
    rfl
def root452Tail33 : List Nat := []
def root452Tail32 : List Nat := 22669 :: root452Tail33
def root452Tail31 : List Nat := 20743 :: root452Tail32
def root452Tail30 : List Nat := 20731 :: root452Tail31
def root452Tail29 : List Nat := 20011 :: root452Tail30
def root452Tail28 : List Nat := 19681 :: root452Tail29
def root452Tail27 : List Nat := 19483 :: root452Tail28
def root452Tail26 : List Nat := 17761 :: root452Tail27
def root452Tail25 : List Nat := 17581 :: root452Tail26
def root452Tail24 : List Nat := 17377 :: root452Tail25
def root452Tail23 : List Nat := 16831 :: root452Tail24
def root452Tail22 : List Nat := 16063 :: root452Tail23
def root452Tail21 : List Nat := 15907 :: root452Tail22
def root452Tail20 : List Nat := 15739 :: root452Tail21
def root452Tail19 : List Nat := 15649 :: root452Tail20
def root452Tail18 : List Nat := 15601 :: root452Tail19
def root452Tail17 : List Nat := 14737 :: root452Tail18
def root452Tail16 : List Nat := 13759 :: root452Tail17
def root452Tail15 : List Nat := 13417 :: root452Tail16
def root452Tail14 : List Nat := 12763 :: root452Tail15
def root452Tail13 : List Nat := 12421 :: root452Tail14
def root452Tail12 : List Nat := 11197 :: root452Tail13
def root452Tail11 : List Nat := 10861 :: root452Tail12
def root452Tail10 : List Nat := 10273 :: root452Tail11
def root452Tail9 : List Nat := 9907 :: root452Tail10
def root452Tail8 : List Nat := 7177 :: root452Tail9
def root452Tail7 : List Nat := 6619 :: root452Tail8
def root452Tail6 : List Nat := 6301 :: root452Tail7
def root452Tail5 : List Nat := 6211 :: root452Tail6
def root452Tail4 : List Nat := 6073 :: root452Tail5
def root452Tail3 : List Nat := 5821 :: root452Tail4
def root452Tail2 : List Nat := 4789 :: root452Tail3
def root452Tail1 : List Nat := 3931 :: root452Tail2
def root452Tail0 : List Nat := 3511 :: root452Tail1
def root452Candidates0 : List Nat := [13417, 15601]
set_option maxRecDepth 8192 in
theorem root452Candidates0_checked : candidates edge 4 3217 3511 root452Tail1 winner.bound = root452Candidates0 := by
  unfold candidates
  have row : edge 3511 = (SortedNatTree.contains FastRows.row3511) := by
    change Data.row3511.contains = (SortedNatTree.contains FastRows.row3511)
    exact FastRows.row3511_checked.symm
  rw [row]
  decide
theorem root452Step0 : scan edge (search edge 3) 4 [3217] 3217 root452Tail0 winner =
    scan edge (search edge 3) 4 [3217] 3217 root452Tail1 winner := by
  rw [root452Tail0, scan, if_neg (by decide), root452Candidates0_checked]
  change scan edge (search edge 3) 4 [3217] 3217 root452Tail1
    (if 3 ≤ root452Candidates0.length then search edge 3 [3217, 3511] root452Candidates0 6728 winner else winner) = _
  rw [if_neg (by decide)]
def root452Candidates1 : List Nat := [15601]
set_option maxRecDepth 8192 in
theorem root452Candidates1_checked : candidates edge 4 3217 3931 root452Tail2 winner.bound = root452Candidates1 := by
  unfold candidates
  have row : edge 3931 = (SortedNatTree.contains FastRows.row3931) := by
    change Data.row3931.contains = (SortedNatTree.contains FastRows.row3931)
    exact FastRows.row3931_checked.symm
  rw [row]
  decide
theorem root452Step1 : scan edge (search edge 3) 4 [3217] 3217 root452Tail1 winner =
    scan edge (search edge 3) 4 [3217] 3217 root452Tail2 winner := by
  rw [root452Tail1, scan, if_neg (by decide), root452Candidates1_checked]
  change scan edge (search edge 3) 4 [3217] 3217 root452Tail2
    (if 3 ≤ root452Candidates1.length then search edge 3 [3217, 3931] root452Candidates1 7148 winner else winner) = _
  rw [if_neg (by decide)]
def root452Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root452Candidates2_checked : candidates edge 4 3217 4789 root452Tail3 winner.bound = root452Candidates2 := by
  unfold candidates
  have row : edge 4789 = (SortedNatTree.contains FastRows.row4789) := by
    change Data.row4789.contains = (SortedNatTree.contains FastRows.row4789)
    exact FastRows.row4789_checked.symm
  rw [row]
  decide
theorem root452Step2 : scan edge (search edge 3) 4 [3217] 3217 root452Tail2 winner =
    scan edge (search edge 3) 4 [3217] 3217 root452Tail3 winner := by
  rw [root452Tail2, scan, if_neg (by decide), root452Candidates2_checked]
  change scan edge (search edge 3) 4 [3217] 3217 root452Tail3
    (if 3 ≤ root452Candidates2.length then search edge 3 [3217, 4789] root452Candidates2 8006 winner else winner) = _
  rw [if_neg (by decide)]
theorem root452Scan3 : scan edge (search edge 3) 4 [3217] 3217 root452Tail3 winner = winner := by rw [root452Tail3, scan, if_pos (by decide)]
theorem root452Scan2 : scan edge (search edge 3) 4 [3217] 3217 root452Tail2 winner = winner :=
  (root452Step2).trans root452Scan3
theorem root452Scan1 : scan edge (search edge 3) 4 [3217] 3217 root452Tail1 winner = winner :=
  (root452Step1).trans root452Scan2
theorem root452Scan0 : scan edge (search edge 3) 4 [3217] 3217 root452Tail0 winner = winner :=
  (root452Step0).trans root452Scan1
set_option maxRecDepth 4096 in
theorem root452Branch : search edge 4 [3217] root452Candidates 3217 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3217] 3217 root452Tail0 winner = winner
  exact root452Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
