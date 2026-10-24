import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root506Candidates : List Nat := Data.row3637
set_option maxRecDepth 8192 in
theorem root506Candidates_checked : candidates edge 5 0 3637 tail507 winner.bound = root506Candidates := by
  apply candidates_eq_row edge 5 0 3637 tail507 root506Candidates winner.bound tail507_ordered
  · decide
  · decide
  · funext q
    rfl
def root506Tail30 : List Nat := []
def root506Tail29 : List Nat := 22129 :: root506Tail30
def root506Tail28 : List Nat := 22003 :: root506Tail29
def root506Tail27 : List Nat := 18793 :: root506Tail28
def root506Tail26 : List Nat := 18211 :: root506Tail27
def root506Tail25 : List Nat := 18013 :: root506Tail26
def root506Tail24 : List Nat := 17383 :: root506Tail25
def root506Tail23 : List Nat := 15139 :: root506Tail24
def root506Tail22 : List Nat := 14851 :: root506Tail23
def root506Tail21 : List Nat := 14011 :: root506Tail22
def root506Tail20 : List Nat := 13309 :: root506Tail21
def root506Tail19 : List Nat := 13003 :: root506Tail20
def root506Tail18 : List Nat := 12451 :: root506Tail19
def root506Tail17 : List Nat := 11863 :: root506Tail18
def root506Tail16 : List Nat := 11827 :: root506Tail17
def root506Tail15 : List Nat := 11503 :: root506Tail16
def root506Tail14 : List Nat := 10771 :: root506Tail15
def root506Tail13 : List Nat := 9697 :: root506Tail14
def root506Tail12 : List Nat := 9649 :: root506Tail13
def root506Tail11 : List Nat := 8803 :: root506Tail12
def root506Tail10 : List Nat := 8677 :: root506Tail11
def root506Tail9 : List Nat := 8059 :: root506Tail10
def root506Tail8 : List Nat := 7639 :: root506Tail9
def root506Tail7 : List Nat := 6871 :: root506Tail8
def root506Tail6 : List Nat := 6841 :: root506Tail7
def root506Tail5 : List Nat := 6343 :: root506Tail6
def root506Tail4 : List Nat := 5839 :: root506Tail5
def root506Tail3 : List Nat := 5659 :: root506Tail4
def root506Tail2 : List Nat := 5503 :: root506Tail3
def root506Tail1 : List Nat := 5407 :: root506Tail2
def root506Tail0 : List Nat := 3643 :: root506Tail1
def root506Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root506Candidates0_checked : candidates edge 4 3637 3643 root506Tail1 winner.bound = root506Candidates0 := by
  unfold candidates
  have row : edge 3643 = (SortedNatTree.contains FastRows.row3643) := by
    change Data.row3643.contains = (SortedNatTree.contains FastRows.row3643)
    exact FastRows.row3643_checked.symm
  rw [row]
  decide
theorem root506Step0 : scan edge (search edge 3) 4 [3637] 3637 root506Tail0 winner =
    scan edge (search edge 3) 4 [3637] 3637 root506Tail1 winner := by
  rw [root506Tail0, scan, if_neg (by decide), root506Candidates0_checked]
  change scan edge (search edge 3) 4 [3637] 3637 root506Tail1
    (if 3 ≤ root506Candidates0.length then search edge 3 [3637, 3643] root506Candidates0 7280 winner else winner) = _
  rw [if_neg (by decide)]
def root506Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root506Candidates1_checked : candidates edge 4 3637 5407 root506Tail2 winner.bound = root506Candidates1 := by
  unfold candidates
  have row : edge 5407 = (SortedNatTree.contains FastRows.row5407) := by
    change Data.row5407.contains = (SortedNatTree.contains FastRows.row5407)
    exact FastRows.row5407_checked.symm
  rw [row]
  decide
theorem root506Step1 : scan edge (search edge 3) 4 [3637] 3637 root506Tail1 winner =
    scan edge (search edge 3) 4 [3637] 3637 root506Tail2 winner := by
  rw [root506Tail1, scan, if_neg (by decide), root506Candidates1_checked]
  change scan edge (search edge 3) 4 [3637] 3637 root506Tail2
    (if 3 ≤ root506Candidates1.length then search edge 3 [3637, 5407] root506Candidates1 9044 winner else winner) = _
  rw [if_neg (by decide)]
def root506Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root506Candidates2_checked : candidates edge 4 3637 5503 root506Tail3 winner.bound = root506Candidates2 := by
  unfold candidates
  have row : edge 5503 = (SortedNatTree.contains FastRows.row5503) := by
    change Data.row5503.contains = (SortedNatTree.contains FastRows.row5503)
    exact FastRows.row5503_checked.symm
  rw [row]
  decide
theorem root506Step2 : scan edge (search edge 3) 4 [3637] 3637 root506Tail2 winner =
    scan edge (search edge 3) 4 [3637] 3637 root506Tail3 winner := by
  rw [root506Tail2, scan, if_neg (by decide), root506Candidates2_checked]
  change scan edge (search edge 3) 4 [3637] 3637 root506Tail3
    (if 3 ≤ root506Candidates2.length then search edge 3 [3637, 5503] root506Candidates2 9140 winner else winner) = _
  rw [if_neg (by decide)]
theorem root506Scan3 : scan edge (search edge 3) 4 [3637] 3637 root506Tail3 winner = winner := by rw [root506Tail3, scan, if_pos (by decide)]
theorem root506Scan2 : scan edge (search edge 3) 4 [3637] 3637 root506Tail2 winner = winner :=
  (root506Step2).trans root506Scan3
theorem root506Scan1 : scan edge (search edge 3) 4 [3637] 3637 root506Tail1 winner = winner :=
  (root506Step1).trans root506Scan2
theorem root506Scan0 : scan edge (search edge 3) 4 [3637] 3637 root506Tail0 winner = winner :=
  (root506Step0).trans root506Scan1
set_option maxRecDepth 4096 in
theorem root506Branch : search edge 4 [3637] root506Candidates 3637 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3637] 3637 root506Tail0 winner = winner
  exact root506Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
