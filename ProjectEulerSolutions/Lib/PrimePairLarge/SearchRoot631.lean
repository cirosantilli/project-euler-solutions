import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root631Candidates : List Nat := Data.row4691
set_option maxRecDepth 8192 in
theorem root631Candidates_checked : candidates edge 5 0 4691 tail632 winner.bound = root631Candidates := by
  apply candidates_eq_row edge 5 0 4691 tail632 root631Candidates winner.bound tail632_ordered
  · decide
  · decide
  · funext q
    rfl
def root631Tail28 : List Nat := []
def root631Tail27 : List Nat := 20369 :: root631Tail28
def root631Tail26 : List Nat := 19841 :: root631Tail27
def root631Tail25 : List Nat := 19571 :: root631Tail26
def root631Tail24 : List Nat := 19181 :: root631Tail25
def root631Tail23 : List Nat := 19163 :: root631Tail24
def root631Tail22 : List Nat := 18719 :: root631Tail23
def root631Tail21 : List Nat := 18311 :: root631Tail22
def root631Tail20 : List Nat := 17921 :: root631Tail21
def root631Tail19 : List Nat := 17609 :: root631Tail20
def root631Tail18 : List Nat := 17471 :: root631Tail19
def root631Tail17 : List Nat := 17027 :: root631Tail18
def root631Tail16 : List Nat := 16481 :: root631Tail17
def root631Tail15 : List Nat := 16127 :: root631Tail16
def root631Tail14 : List Nat := 16001 :: root631Tail15
def root631Tail13 : List Nat := 15887 :: root631Tail14
def root631Tail12 : List Nat := 15359 :: root631Tail13
def root631Tail11 : List Nat := 14699 :: root631Tail12
def root631Tail10 : List Nat := 14519 :: root631Tail11
def root631Tail9 : List Nat := 14249 :: root631Tail10
def root631Tail8 : List Nat := 13709 :: root631Tail9
def root631Tail7 : List Nat := 13457 :: root631Tail8
def root631Tail6 : List Nat := 12239 :: root631Tail7
def root631Tail5 : List Nat := 7457 :: root631Tail6
def root631Tail4 : List Nat := 6911 :: root631Tail5
def root631Tail3 : List Nat := 6581 :: root631Tail4
def root631Tail2 : List Nat := 6521 :: root631Tail3
def root631Tail1 : List Nat := 5987 :: root631Tail2
def root631Tail0 : List Nat := 4817 :: root631Tail1
def root631Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root631Candidates0_checked : candidates edge 4 4691 4817 root631Tail1 winner.bound = root631Candidates0 := by
  unfold candidates
  have row : edge 4817 = (SortedNatTree.contains FastRows.row4817) := by
    change Data.row4817.contains = (SortedNatTree.contains FastRows.row4817)
    exact FastRows.row4817_checked.symm
  rw [row]
  decide
theorem root631Step0 : scan edge (search edge 3) 4 [4691] 4691 root631Tail0 winner =
    scan edge (search edge 3) 4 [4691] 4691 root631Tail1 winner := by
  rw [root631Tail0, scan, if_neg (by decide), root631Candidates0_checked]
  change scan edge (search edge 3) 4 [4691] 4691 root631Tail1
    (if 3 ≤ root631Candidates0.length then search edge 3 [4691, 4817] root631Candidates0 9508 winner else winner) = _
  rw [if_neg (by decide)]
theorem root631Scan1 : scan edge (search edge 3) 4 [4691] 4691 root631Tail1 winner = winner := by rw [root631Tail1, scan, if_pos (by decide)]
theorem root631Scan0 : scan edge (search edge 3) 4 [4691] 4691 root631Tail0 winner = winner :=
  (root631Step0).trans root631Scan1
set_option maxRecDepth 4096 in
theorem root631Branch : search edge 4 [4691] root631Candidates 4691 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4691] 4691 root631Tail0 winner = winner
  exact root631Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
