import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root524Candidates : List Nat := Data.row3793
set_option maxRecDepth 8192 in
theorem root524Candidates_checked : candidates edge 5 0 3793 tail525 winner.bound = root524Candidates := by
  apply candidates_eq_row edge 5 0 3793 tail525 root524Candidates winner.bound tail525_ordered
  · decide
  · decide
  · funext q
    rfl
def root524Tail26 : List Nat := []
def root524Tail25 : List Nat := 21277 :: root524Tail26
def root524Tail24 : List Nat := 21169 :: root524Tail25
def root524Tail23 : List Nat := 20611 :: root524Tail24
def root524Tail22 : List Nat := 20353 :: root524Tail23
def root524Tail21 : List Nat := 18913 :: root524Tail22
def root524Tail20 : List Nat := 18859 :: root524Tail21
def root524Tail19 : List Nat := 17293 :: root524Tail20
def root524Tail18 : List Nat := 16981 :: root524Tail19
def root524Tail17 : List Nat := 16363 :: root524Tail18
def root524Tail16 : List Nat := 15439 :: root524Tail17
def root524Tail15 : List Nat := 14011 :: root524Tail16
def root524Tail14 : List Nat := 13219 :: root524Tail15
def root524Tail13 : List Nat := 11731 :: root524Tail14
def root524Tail12 : List Nat := 11689 :: root524Tail13
def root524Tail11 : List Nat := 10753 :: root524Tail12
def root524Tail10 : List Nat := 10093 :: root524Tail11
def root524Tail9 : List Nat := 9007 :: root524Tail10
def root524Tail8 : List Nat := 8179 :: root524Tail9
def root524Tail7 : List Nat := 8167 :: root524Tail8
def root524Tail6 : List Nat := 7873 :: root524Tail7
def root524Tail5 : List Nat := 7213 :: root524Tail6
def root524Tail4 : List Nat := 6949 :: root524Tail5
def root524Tail3 : List Nat := 6907 :: root524Tail4
def root524Tail2 : List Nat := 6733 :: root524Tail3
def root524Tail1 : List Nat := 5659 :: root524Tail2
def root524Tail0 : List Nat := 5179 :: root524Tail1
def root524Candidates0 : List Nat := [7873]
set_option maxRecDepth 8192 in
theorem root524Candidates0_checked : candidates edge 4 3793 5179 root524Tail1 winner.bound = root524Candidates0 := by
  unfold candidates
  have row : edge 5179 = (SortedNatTree.contains FastRows.row5179) := by
    change Data.row5179.contains = (SortedNatTree.contains FastRows.row5179)
    exact FastRows.row5179_checked.symm
  rw [row]
  decide
theorem root524Step0 : scan edge (search edge 3) 4 [3793] 3793 root524Tail0 winner =
    scan edge (search edge 3) 4 [3793] 3793 root524Tail1 winner := by
  rw [root524Tail0, scan, if_neg (by decide), root524Candidates0_checked]
  change scan edge (search edge 3) 4 [3793] 3793 root524Tail1
    (if 3 ≤ root524Candidates0.length then search edge 3 [3793, 5179] root524Candidates0 8972 winner else winner) = _
  rw [if_neg (by decide)]
theorem root524Scan1 : scan edge (search edge 3) 4 [3793] 3793 root524Tail1 winner = winner := by rw [root524Tail1, scan, if_pos (by decide)]
theorem root524Scan0 : scan edge (search edge 3) 4 [3793] 3793 root524Tail0 winner = winner :=
  (root524Step0).trans root524Scan1
set_option maxRecDepth 4096 in
theorem root524Branch : search edge 4 [3793] root524Candidates 3793 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3793] 3793 root524Tail0 winner = winner
  exact root524Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
