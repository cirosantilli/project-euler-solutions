import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root593Candidates : List Nat := Data.row4363
set_option maxRecDepth 8192 in
theorem root593Candidates_checked : candidates edge 5 0 4363 tail594 winner.bound = root593Candidates := by
  apply candidates_eq_row edge 5 0 4363 tail594 root593Candidates winner.bound tail594_ordered
  · decide
  · decide
  · funext q
    rfl
def root593Tail37 : List Nat := []
def root593Tail36 : List Nat := 20743 :: root593Tail37
def root593Tail35 : List Nat := 20719 :: root593Tail36
def root593Tail34 : List Nat := 19801 :: root593Tail35
def root593Tail33 : List Nat := 19423 :: root593Tail34
def root593Tail32 : List Nat := 18367 :: root593Tail33
def root593Tail31 : List Nat := 17137 :: root593Tail32
def root593Tail30 : List Nat := 17029 :: root593Tail31
def root593Tail29 : List Nat := 16963 :: root593Tail30
def root593Tail28 : List Nat := 16729 :: root593Tail29
def root593Tail27 : List Nat := 16267 :: root593Tail28
def root593Tail26 : List Nat := 16063 :: root593Tail27
def root593Tail25 : List Nat := 15889 :: root593Tail26
def root593Tail24 : List Nat := 14929 :: root593Tail25
def root593Tail23 : List Nat := 14869 :: root593Tail24
def root593Tail22 : List Nat := 14779 :: root593Tail23
def root593Tail21 : List Nat := 13873 :: root593Tail22
def root593Tail20 : List Nat := 13417 :: root593Tail21
def root593Tail19 : List Nat := 13291 :: root593Tail20
def root593Tail18 : List Nat := 13147 :: root593Tail19
def root593Tail17 : List Nat := 13063 :: root593Tail18
def root593Tail16 : List Nat := 11779 :: root593Tail17
def root593Tail15 : List Nat := 11527 :: root593Tail16
def root593Tail14 : List Nat := 10453 :: root593Tail15
def root593Tail13 : List Nat := 10009 :: root593Tail14
def root593Tail12 : List Nat := 9319 :: root593Tail13
def root593Tail11 : List Nat := 9199 :: root593Tail12
def root593Tail10 : List Nat := 9091 :: root593Tail11
def root593Tail9 : List Nat := 8761 :: root593Tail10
def root593Tail8 : List Nat := 8713 :: root593Tail9
def root593Tail7 : List Nat := 8461 :: root593Tail8
def root593Tail6 : List Nat := 8311 :: root593Tail7
def root593Tail5 : List Nat := 7687 :: root593Tail6
def root593Tail4 : List Nat := 7621 :: root593Tail5
def root593Tail3 : List Nat := 7459 :: root593Tail4
def root593Tail2 : List Nat := 6133 :: root593Tail3
def root593Tail1 : List Nat := 5653 :: root593Tail2
def root593Tail0 : List Nat := 4813 :: root593Tail1
def root593Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root593Candidates0_checked : candidates edge 4 4363 4813 root593Tail1 winner.bound = root593Candidates0 := by
  unfold candidates
  have row : edge 4813 = (SortedNatTree.contains FastRows.row4813) := by
    change Data.row4813.contains = (SortedNatTree.contains FastRows.row4813)
    exact FastRows.row4813_checked.symm
  rw [row]
  decide
theorem root593Step0 : scan edge (search edge 3) 4 [4363] 4363 root593Tail0 winner =
    scan edge (search edge 3) 4 [4363] 4363 root593Tail1 winner := by
  rw [root593Tail0, scan, if_neg (by decide), root593Candidates0_checked]
  change scan edge (search edge 3) 4 [4363] 4363 root593Tail1
    (if 3 ≤ root593Candidates0.length then search edge 3 [4363, 4813] root593Candidates0 9176 winner else winner) = _
  rw [if_neg (by decide)]
theorem root593Scan1 : scan edge (search edge 3) 4 [4363] 4363 root593Tail1 winner = winner := by rw [root593Tail1, scan, if_pos (by decide)]
theorem root593Scan0 : scan edge (search edge 3) 4 [4363] 4363 root593Tail0 winner = winner :=
  (root593Step0).trans root593Scan1
set_option maxRecDepth 4096 in
theorem root593Branch : search edge 4 [4363] root593Candidates 4363 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4363] 4363 root593Tail0 winner = winner
  exact root593Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
