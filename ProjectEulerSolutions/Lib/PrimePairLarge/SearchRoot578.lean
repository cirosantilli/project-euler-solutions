import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root578Candidates : List Nat := Data.row4241
set_option maxRecDepth 8192 in
theorem root578Candidates_checked : candidates edge 5 0 4241 tail579 winner.bound = root578Candidates := by
  apply candidates_eq_row edge 5 0 4241 tail579 root578Candidates winner.bound tail579_ordered
  · decide
  · decide
  · funext q
    rfl
def root578Tail31 : List Nat := []
def root578Tail30 : List Nat := 21701 :: root578Tail31
def root578Tail29 : List Nat := 21011 :: root578Tail30
def root578Tail28 : List Nat := 20327 :: root578Tail29
def root578Tail27 : List Nat := 20231 :: root578Tail28
def root578Tail26 : List Nat := 19433 :: root578Tail27
def root578Tail25 : List Nat := 18743 :: root578Tail26
def root578Tail24 : List Nat := 18521 :: root578Tail25
def root578Tail23 : List Nat := 16937 :: root578Tail24
def root578Tail22 : List Nat := 15233 :: root578Tail23
def root578Tail21 : List Nat := 15173 :: root578Tail22
def root578Tail20 : List Nat := 15161 :: root578Tail21
def root578Tail19 : List Nat := 13709 :: root578Tail20
def root578Tail18 : List Nat := 13457 :: root578Tail19
def root578Tail17 : List Nat := 11351 :: root578Tail18
def root578Tail16 : List Nat := 10949 :: root578Tail17
def root578Tail15 : List Nat := 10289 :: root578Tail16
def root578Tail14 : List Nat := 10259 :: root578Tail15
def root578Tail13 : List Nat := 9413 :: root578Tail14
def root578Tail12 : List Nat := 9323 :: root578Tail13
def root578Tail11 : List Nat := 8147 :: root578Tail12
def root578Tail10 : List Nat := 8093 :: root578Tail11
def root578Tail9 : List Nat := 7757 :: root578Tail10
def root578Tail8 : List Nat := 7673 :: root578Tail9
def root578Tail7 : List Nat := 7559 :: root578Tail8
def root578Tail6 : List Nat := 7103 :: root578Tail7
def root578Tail5 : List Nat := 6977 :: root578Tail6
def root578Tail4 : List Nat := 6389 :: root578Tail5
def root578Tail3 : List Nat := 6053 :: root578Tail4
def root578Tail2 : List Nat := 5981 :: root578Tail3
def root578Tail1 : List Nat := 4793 :: root578Tail2
def root578Tail0 : List Nat := 4751 :: root578Tail1
def root578Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root578Candidates0_checked : candidates edge 4 4241 4751 root578Tail1 winner.bound = root578Candidates0 := by
  unfold candidates
  have row : edge 4751 = (SortedNatTree.contains FastRows.row4751) := by
    change Data.row4751.contains = (SortedNatTree.contains FastRows.row4751)
    exact FastRows.row4751_checked.symm
  rw [row]
  decide
theorem root578Step0 : scan edge (search edge 3) 4 [4241] 4241 root578Tail0 winner =
    scan edge (search edge 3) 4 [4241] 4241 root578Tail1 winner := by
  rw [root578Tail0, scan, if_neg (by decide), root578Candidates0_checked]
  change scan edge (search edge 3) 4 [4241] 4241 root578Tail1
    (if 3 ≤ root578Candidates0.length then search edge 3 [4241, 4751] root578Candidates0 8992 winner else winner) = _
  rw [if_neg (by decide)]
def root578Candidates1 : List Nat := [6389]
set_option maxRecDepth 8192 in
theorem root578Candidates1_checked : candidates edge 4 4241 4793 root578Tail2 winner.bound = root578Candidates1 := by
  unfold candidates
  have row : edge 4793 = (SortedNatTree.contains FastRows.row4793) := by
    change Data.row4793.contains = (SortedNatTree.contains FastRows.row4793)
    exact FastRows.row4793_checked.symm
  rw [row]
  decide
theorem root578Step1 : scan edge (search edge 3) 4 [4241] 4241 root578Tail1 winner =
    scan edge (search edge 3) 4 [4241] 4241 root578Tail2 winner := by
  rw [root578Tail1, scan, if_neg (by decide), root578Candidates1_checked]
  change scan edge (search edge 3) 4 [4241] 4241 root578Tail2
    (if 3 ≤ root578Candidates1.length then search edge 3 [4241, 4793] root578Candidates1 9034 winner else winner) = _
  rw [if_neg (by decide)]
theorem root578Scan2 : scan edge (search edge 3) 4 [4241] 4241 root578Tail2 winner = winner := by rw [root578Tail2, scan, if_pos (by decide)]
theorem root578Scan1 : scan edge (search edge 3) 4 [4241] 4241 root578Tail1 winner = winner :=
  (root578Step1).trans root578Scan2
theorem root578Scan0 : scan edge (search edge 3) 4 [4241] 4241 root578Tail0 winner = winner :=
  (root578Step0).trans root578Scan1
set_option maxRecDepth 4096 in
theorem root578Branch : search edge 4 [4241] root578Candidates 4241 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4241] 4241 root578Tail0 winner = winner
  exact root578Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
