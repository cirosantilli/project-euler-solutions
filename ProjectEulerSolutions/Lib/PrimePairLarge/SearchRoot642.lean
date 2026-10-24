import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root642Candidates : List Nat := Data.row4793
set_option maxRecDepth 8192 in
theorem root642Candidates_checked : candidates edge 5 0 4793 tail643 winner.bound = root642Candidates := by
  apply candidates_eq_row edge 5 0 4793 tail643 root642Candidates winner.bound tail643_ordered
  · decide
  · decide
  · funext q
    rfl
def root642Tail28 : List Nat := []
def root642Tail27 : List Nat := 20333 :: root642Tail28
def root642Tail26 : List Nat := 20327 :: root642Tail27
def root642Tail25 : List Nat := 19697 :: root642Tail26
def root642Tail24 : List Nat := 19073 :: root642Tail25
def root642Tail23 : List Nat := 18251 :: root642Tail24
def root642Tail22 : List Nat := 17393 :: root642Tail23
def root642Tail21 : List Nat := 16889 :: root642Tail22
def root642Tail20 : List Nat := 16631 :: root642Tail21
def root642Tail19 : List Nat := 16427 :: root642Tail20
def root642Tail18 : List Nat := 16361 :: root642Tail19
def root642Tail17 : List Nat := 13697 :: root642Tail18
def root642Tail16 : List Nat := 13229 :: root642Tail17
def root642Tail15 : List Nat := 13151 :: root642Tail16
def root642Tail14 : List Nat := 13007 :: root642Tail15
def root642Tail13 : List Nat := 11621 :: root642Tail14
def root642Tail12 : List Nat := 9791 :: root642Tail13
def root642Tail11 : List Nat := 8783 :: root642Tail12
def root642Tail10 : List Nat := 8669 :: root642Tail11
def root642Tail9 : List Nat := 8123 :: root642Tail10
def root642Tail8 : List Nat := 7481 :: root642Tail9
def root642Tail7 : List Nat := 7013 :: root642Tail8
def root642Tail6 : List Nat := 6869 :: root642Tail7
def root642Tail5 : List Nat := 6389 :: root642Tail6
def root642Tail4 : List Nat := 5501 :: root642Tail5
def root642Tail3 : List Nat := 5399 :: root642Tail4
def root642Tail2 : List Nat := 5099 :: root642Tail3
def root642Tail1 : List Nat := 5003 :: root642Tail2
def root642Tail0 : List Nat := 4871 :: root642Tail1
def root642Candidates0 : List Nat := [6389, 8123, 9791]
set_option maxRecDepth 8192 in
theorem root642Candidates0_checked : candidates edge 4 4793 4871 root642Tail1 winner.bound = root642Candidates0 := by
  unfold candidates
  have row : edge 4871 = (SortedNatTree.contains FastRows.row4871) := by
    change Data.row4871.contains = (SortedNatTree.contains FastRows.row4871)
    exact FastRows.row4871_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root642Child0 : search edge 3 [4793, 4871] root642Candidates0 9664 winner = winner := by decide
theorem root642Step0 : scan edge (search edge 3) 4 [4793] 4793 root642Tail0 winner =
    scan edge (search edge 3) 4 [4793] 4793 root642Tail1 winner := by
  rw [root642Tail0, scan, if_neg (by decide), root642Candidates0_checked]
  change scan edge (search edge 3) 4 [4793] 4793 root642Tail1
    (if 3 ≤ root642Candidates0.length then search edge 3 [4793, 4871] root642Candidates0 9664 winner else winner) = _
  rw [if_pos (by decide), root642Child0]
def root642Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root642Candidates1_checked : candidates edge 4 4793 5003 root642Tail2 winner.bound = root642Candidates1 := by
  unfold candidates
  have row : edge 5003 = (SortedNatTree.contains FastRows.row5003) := by
    change Data.row5003.contains = (SortedNatTree.contains FastRows.row5003)
    exact FastRows.row5003_checked.symm
  rw [row]
  decide
theorem root642Step1 : scan edge (search edge 3) 4 [4793] 4793 root642Tail1 winner =
    scan edge (search edge 3) 4 [4793] 4793 root642Tail2 winner := by
  rw [root642Tail1, scan, if_neg (by decide), root642Candidates1_checked]
  change scan edge (search edge 3) 4 [4793] 4793 root642Tail2
    (if 3 ≤ root642Candidates1.length then search edge 3 [4793, 5003] root642Candidates1 9796 winner else winner) = _
  rw [if_neg (by decide)]
def root642Candidates2 : List Nat := [5501]
set_option maxRecDepth 8192 in
theorem root642Candidates2_checked : candidates edge 4 4793 5099 root642Tail3 winner.bound = root642Candidates2 := by
  unfold candidates
  have row : edge 5099 = (SortedNatTree.contains FastRows.row5099) := by
    change Data.row5099.contains = (SortedNatTree.contains FastRows.row5099)
    exact FastRows.row5099_checked.symm
  rw [row]
  decide
theorem root642Step2 : scan edge (search edge 3) 4 [4793] 4793 root642Tail2 winner =
    scan edge (search edge 3) 4 [4793] 4793 root642Tail3 winner := by
  rw [root642Tail2, scan, if_neg (by decide), root642Candidates2_checked]
  change scan edge (search edge 3) 4 [4793] 4793 root642Tail3
    (if 3 ≤ root642Candidates2.length then search edge 3 [4793, 5099] root642Candidates2 9892 winner else winner) = _
  rw [if_neg (by decide)]
theorem root642Scan3 : scan edge (search edge 3) 4 [4793] 4793 root642Tail3 winner = winner := by rw [root642Tail3, scan, if_pos (by decide)]
theorem root642Scan2 : scan edge (search edge 3) 4 [4793] 4793 root642Tail2 winner = winner :=
  (root642Step2).trans root642Scan3
theorem root642Scan1 : scan edge (search edge 3) 4 [4793] 4793 root642Tail1 winner = winner :=
  (root642Step1).trans root642Scan2
theorem root642Scan0 : scan edge (search edge 3) 4 [4793] 4793 root642Tail0 winner = winner :=
  (root642Step0).trans root642Scan1
set_option maxRecDepth 4096 in
theorem root642Branch : search edge 4 [4793] root642Candidates 4793 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4793] 4793 root642Tail0 winner = winner
  exact root642Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
