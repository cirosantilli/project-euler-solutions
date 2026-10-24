import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root543Candidates : List Nat := Data.row3931
set_option maxRecDepth 8192 in
theorem root543Candidates_checked : candidates edge 5 0 3931 tail544 winner.bound = root543Candidates := by
  apply candidates_eq_row edge 5 0 3931 tail544 root543Candidates winner.bound tail544_ordered
  · decide
  · decide
  · funext q
    rfl
def root543Tail32 : List Nat := []
def root543Tail31 : List Nat := 21397 :: root543Tail32
def root543Tail30 : List Nat := 21031 :: root543Tail31
def root543Tail29 : List Nat := 20149 :: root543Tail30
def root543Tail28 : List Nat := 20011 :: root543Tail29
def root543Tail27 : List Nat := 19681 :: root543Tail28
def root543Tail26 : List Nat := 19471 :: root543Tail27
def root543Tail25 : List Nat := 19087 :: root543Tail26
def root543Tail24 : List Nat := 17971 :: root543Tail25
def root543Tail23 : List Nat := 17419 :: root543Tail24
def root543Tail22 : List Nat := 17029 :: root543Tail23
def root543Tail21 : List Nat := 16963 :: root543Tail22
def root543Tail20 : List Nat := 16747 :: root543Tail21
def root543Tail19 : List Nat := 16333 :: root543Tail20
def root543Tail18 : List Nat := 16189 :: root543Tail19
def root543Tail17 : List Nat := 16033 :: root543Tail18
def root543Tail16 : List Nat := 15601 :: root543Tail17
def root543Tail15 : List Nat := 13681 :: root543Tail16
def root543Tail14 : List Nat := 12589 :: root543Tail15
def root543Tail13 : List Nat := 9619 :: root543Tail14
def root543Tail12 : List Nat := 9403 :: root543Tail13
def root543Tail11 : List Nat := 8803 :: root543Tail12
def root543Tail10 : List Nat := 8233 :: root543Tail11
def root543Tail9 : List Nat := 8089 :: root543Tail10
def root543Tail8 : List Nat := 7639 :: root543Tail9
def root543Tail7 : List Nat := 7561 :: root543Tail8
def root543Tail6 : List Nat := 7507 :: root543Tail7
def root543Tail5 : List Nat := 7459 :: root543Tail6
def root543Tail4 : List Nat := 6661 :: root543Tail5
def root543Tail3 : List Nat := 6553 :: root543Tail4
def root543Tail2 : List Nat := 5527 :: root543Tail3
def root543Tail1 : List Nat := 5449 :: root543Tail2
def root543Tail0 : List Nat := 5413 :: root543Tail1
def root543Candidates0 : List Nat := [5527, 9619, 16333]
set_option maxRecDepth 8192 in
theorem root543Candidates0_checked : candidates edge 4 3931 5413 root543Tail1 winner.bound = root543Candidates0 := by
  unfold candidates
  have row : edge 5413 = (SortedNatTree.contains FastRows.row5413) := by
    change Data.row5413.contains = (SortedNatTree.contains FastRows.row5413)
    exact FastRows.row5413_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root543Child0 : search edge 3 [3931, 5413] root543Candidates0 9344 winner = winner := by decide
theorem root543Step0 : scan edge (search edge 3) 4 [3931] 3931 root543Tail0 winner =
    scan edge (search edge 3) 4 [3931] 3931 root543Tail1 winner := by
  rw [root543Tail0, scan, if_neg (by decide), root543Candidates0_checked]
  change scan edge (search edge 3) 4 [3931] 3931 root543Tail1
    (if 3 ≤ root543Candidates0.length then search edge 3 [3931, 5413] root543Candidates0 9344 winner else winner) = _
  rw [if_pos (by decide), root543Child0]
def root543Candidates1 : List Nat := [7507, 8089]
set_option maxRecDepth 8192 in
theorem root543Candidates1_checked : candidates edge 4 3931 5449 root543Tail2 winner.bound = root543Candidates1 := by
  unfold candidates
  have row : edge 5449 = (SortedNatTree.contains FastRows.row5449) := by
    change Data.row5449.contains = (SortedNatTree.contains FastRows.row5449)
    exact FastRows.row5449_checked.symm
  rw [row]
  decide
theorem root543Step1 : scan edge (search edge 3) 4 [3931] 3931 root543Tail1 winner =
    scan edge (search edge 3) 4 [3931] 3931 root543Tail2 winner := by
  rw [root543Tail1, scan, if_neg (by decide), root543Candidates1_checked]
  change scan edge (search edge 3) 4 [3931] 3931 root543Tail2
    (if 3 ≤ root543Candidates1.length then search edge 3 [3931, 5449] root543Candidates1 9380 winner else winner) = _
  rw [if_neg (by decide)]
theorem root543Scan2 : scan edge (search edge 3) 4 [3931] 3931 root543Tail2 winner = winner := by rw [root543Tail2, scan, if_pos (by decide)]
theorem root543Scan1 : scan edge (search edge 3) 4 [3931] 3931 root543Tail1 winner = winner :=
  (root543Step1).trans root543Scan2
theorem root543Scan0 : scan edge (search edge 3) 4 [3931] 3931 root543Tail0 winner = winner :=
  (root543Step0).trans root543Scan1
set_option maxRecDepth 4096 in
theorem root543Branch : search edge 4 [3931] root543Candidates 3931 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3931] 3931 root543Tail0 winner = winner
  exact root543Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
