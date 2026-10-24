import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root669Candidates : List Nat := Data.row5011
set_option maxRecDepth 8192 in
theorem root669Candidates_checked : candidates edge 5 0 5011 tail670 winner.bound = root669Candidates := by
  apply candidates_eq_row edge 5 0 5011 tail670 root669Candidates winner.bound tail670_ordered
  · decide
  · decide
  · funext q
    rfl
def root669Tail25 : List Nat := []
def root669Tail24 : List Nat := 20947 :: root669Tail25
def root669Tail23 : List Nat := 20857 :: root669Tail24
def root669Tail22 : List Nat := 19963 :: root669Tail23
def root669Tail21 : List Nat := 17497 :: root669Tail22
def root669Tail20 : List Nat := 17377 :: root669Tail21
def root669Tail19 : List Nat := 17107 :: root669Tail20
def root669Tail18 : List Nat := 15649 :: root669Tail19
def root669Tail17 : List Nat := 14827 :: root669Tail18
def root669Tail16 : List Nat := 14533 :: root669Tail17
def root669Tail15 : List Nat := 14479 :: root669Tail16
def root669Tail14 : List Nat := 13009 :: root669Tail15
def root669Tail13 : List Nat := 12823 :: root669Tail14
def root669Tail12 : List Nat := 11383 :: root669Tail13
def root669Tail11 : List Nat := 10531 :: root669Tail12
def root669Tail10 : List Nat := 9649 :: root669Tail11
def root669Tail9 : List Nat := 9049 :: root669Tail10
def root669Tail8 : List Nat := 8191 :: root669Tail9
def root669Tail7 : List Nat := 8053 :: root669Tail8
def root669Tail6 : List Nat := 7393 :: root669Tail7
def root669Tail5 : List Nat := 6961 :: root669Tail6
def root669Tail4 : List Nat := 6949 :: root669Tail5
def root669Tail3 : List Nat := 6397 :: root669Tail4
def root669Tail2 : List Nat := 6271 :: root669Tail3
def root669Tail1 : List Nat := 6163 :: root669Tail2
def root669Tail0 : List Nat := 5209 :: root669Tail1
def root669Candidates0 : List Nat := [6397, 8191, 12823, 14533]
set_option maxRecDepth 8192 in
theorem root669Candidates0_checked : candidates edge 4 5011 5209 root669Tail1 winner.bound = root669Candidates0 := by
  unfold candidates
  have row : edge 5209 = (SortedNatTree.contains FastRows.row5209) := by
    change Data.row5209.contains = (SortedNatTree.contains FastRows.row5209)
    exact FastRows.row5209_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root669Child0 : search edge 3 [5011, 5209] root669Candidates0 10220 winner = winner := by decide
theorem root669Step0 : scan edge (search edge 3) 4 [5011] 5011 root669Tail0 winner =
    scan edge (search edge 3) 4 [5011] 5011 root669Tail1 winner := by
  rw [root669Tail0, scan, if_neg (by decide), root669Candidates0_checked]
  change scan edge (search edge 3) 4 [5011] 5011 root669Tail1
    (if 3 ≤ root669Candidates0.length then search edge 3 [5011, 5209] root669Candidates0 10220 winner else winner) = _
  rw [if_pos (by decide), root669Child0]
theorem root669Scan1 : scan edge (search edge 3) 4 [5011] 5011 root669Tail1 winner = winner := by rw [root669Tail1, scan, if_pos (by decide)]
theorem root669Scan0 : scan edge (search edge 3) 4 [5011] 5011 root669Tail0 winner = winner :=
  (root669Step0).trans root669Scan1
set_option maxRecDepth 4096 in
theorem root669Branch : search edge 4 [5011] root669Candidates 5011 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5011] 5011 root669Tail0 winner = winner
  exact root669Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
