import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root582Candidates : List Nat := Data.row4261
set_option maxRecDepth 8192 in
theorem root582Candidates_checked : candidates edge 5 0 4261 tail583 winner.bound = root582Candidates := by
  apply candidates_eq_row edge 5 0 4261 tail583 root582Candidates winner.bound tail583_ordered
  · decide
  · decide
  · funext q
    rfl
def root582Tail25 : List Nat := []
def root582Tail24 : List Nat := 21019 :: root582Tail25
def root582Tail23 : List Nat := 20743 :: root582Tail24
def root582Tail22 : List Nat := 20719 :: root582Tail23
def root582Tail21 : List Nat := 19867 :: root582Tail22
def root582Tail20 : List Nat := 18541 :: root582Tail21
def root582Tail19 : List Nat := 17737 :: root582Tail20
def root582Tail18 : List Nat := 16843 :: root582Tail19
def root582Tail17 : List Nat := 15787 :: root582Tail18
def root582Tail16 : List Nat := 15373 :: root582Tail17
def root582Tail15 : List Nat := 14869 :: root582Tail16
def root582Tail14 : List Nat := 13267 :: root582Tail15
def root582Tail13 : List Nat := 12451 :: root582Tail14
def root582Tail12 : List Nat := 12433 :: root582Tail13
def root582Tail11 : List Nat := 12073 :: root582Tail12
def root582Tail10 : List Nat := 11971 :: root582Tail11
def root582Tail9 : List Nat := 11437 :: root582Tail10
def root582Tail8 : List Nat := 9817 :: root582Tail9
def root582Tail7 : List Nat := 9397 :: root582Tail8
def root582Tail6 : List Nat := 9319 :: root582Tail7
def root582Tail5 : List Nat := 8179 :: root582Tail6
def root582Tail4 : List Nat := 7951 :: root582Tail5
def root582Tail3 : List Nat := 7621 :: root582Tail4
def root582Tail2 : List Nat := 5827 :: root582Tail3
def root582Tail1 : List Nat := 5281 :: root582Tail2
def root582Tail0 : List Nat := 5011 :: root582Tail1
def root582Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root582Candidates0_checked : candidates edge 4 4261 5011 root582Tail1 winner.bound = root582Candidates0 := by
  unfold candidates
  have row : edge 5011 = (SortedNatTree.contains FastRows.row5011) := by
    change Data.row5011.contains = (SortedNatTree.contains FastRows.row5011)
    exact FastRows.row5011_checked.symm
  rw [row]
  decide
theorem root582Step0 : scan edge (search edge 3) 4 [4261] 4261 root582Tail0 winner =
    scan edge (search edge 3) 4 [4261] 4261 root582Tail1 winner := by
  rw [root582Tail0, scan, if_neg (by decide), root582Candidates0_checked]
  change scan edge (search edge 3) 4 [4261] 4261 root582Tail1
    (if 3 ≤ root582Candidates0.length then search edge 3 [4261, 5011] root582Candidates0 9272 winner else winner) = _
  rw [if_neg (by decide)]
def root582Candidates1 : List Nat := [7951]
set_option maxRecDepth 8192 in
theorem root582Candidates1_checked : candidates edge 4 4261 5281 root582Tail2 winner.bound = root582Candidates1 := by
  unfold candidates
  have row : edge 5281 = (SortedNatTree.contains FastRows.row5281) := by
    change Data.row5281.contains = (SortedNatTree.contains FastRows.row5281)
    exact FastRows.row5281_checked.symm
  rw [row]
  decide
theorem root582Step1 : scan edge (search edge 3) 4 [4261] 4261 root582Tail1 winner =
    scan edge (search edge 3) 4 [4261] 4261 root582Tail2 winner := by
  rw [root582Tail1, scan, if_neg (by decide), root582Candidates1_checked]
  change scan edge (search edge 3) 4 [4261] 4261 root582Tail2
    (if 3 ≤ root582Candidates1.length then search edge 3 [4261, 5281] root582Candidates1 9542 winner else winner) = _
  rw [if_neg (by decide)]
theorem root582Scan2 : scan edge (search edge 3) 4 [4261] 4261 root582Tail2 winner = winner := by rw [root582Tail2, scan, if_pos (by decide)]
theorem root582Scan1 : scan edge (search edge 3) 4 [4261] 4261 root582Tail1 winner = winner :=
  (root582Step1).trans root582Scan2
theorem root582Scan0 : scan edge (search edge 3) 4 [4261] 4261 root582Tail0 winner = winner :=
  (root582Step0).trans root582Scan1
set_option maxRecDepth 4096 in
theorem root582Branch : search edge 4 [4261] root582Candidates 4261 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4261] 4261 root582Tail0 winner = winner
  exact root582Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
