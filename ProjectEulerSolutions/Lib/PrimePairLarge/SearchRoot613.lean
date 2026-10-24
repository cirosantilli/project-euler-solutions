import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root613Candidates : List Nat := Data.row4547
set_option maxRecDepth 8192 in
theorem root613Candidates_checked : candidates edge 5 0 4547 tail614 winner.bound = root613Candidates := by
  apply candidates_eq_row edge 5 0 4547 tail614 root613Candidates winner.bound tail614_ordered
  · decide
  · decide
  · funext q
    rfl
def root613Tail30 : List Nat := []
def root613Tail29 : List Nat := 19073 :: root613Tail30
def root613Tail28 : List Nat := 18587 :: root613Tail29
def root613Tail27 : List Nat := 18287 :: root613Tail28
def root613Tail26 : List Nat := 17609 :: root613Tail27
def root613Tail25 : List Nat := 17033 :: root613Tail26
def root613Tail24 : List Nat := 16763 :: root613Tail25
def root613Tail23 : List Nat := 16319 :: root613Tail24
def root613Tail22 : List Nat := 16139 :: root613Tail23
def root613Tail21 : List Nat := 16103 :: root613Tail22
def root613Tail20 : List Nat := 15299 :: root613Tail21
def root613Tail19 : List Nat := 15227 :: root613Tail20
def root613Tail18 : List Nat := 12149 :: root613Tail19
def root613Tail17 : List Nat := 12101 :: root613Tail18
def root613Tail16 : List Nat := 11321 :: root613Tail17
def root613Tail15 : List Nat := 11177 :: root613Tail16
def root613Tail14 : List Nat := 10889 :: root613Tail15
def root613Tail13 : List Nat := 9851 :: root613Tail14
def root613Tail12 : List Nat := 8969 :: root613Tail13
def root613Tail11 : List Nat := 8831 :: root613Tail12
def root613Tail10 : List Nat := 8597 :: root613Tail11
def root613Tail9 : List Nat := 7919 :: root613Tail10
def root613Tail8 : List Nat := 7547 :: root613Tail9
def root613Tail7 : List Nat := 7283 :: root613Tail8
def root613Tail6 : List Nat := 6863 :: root613Tail7
def root613Tail5 : List Nat := 6791 :: root613Tail6
def root613Tail4 : List Nat := 6689 :: root613Tail5
def root613Tail3 : List Nat := 6521 :: root613Tail4
def root613Tail2 : List Nat := 6311 :: root613Tail3
def root613Tail1 : List Nat := 4973 :: root613Tail2
def root613Tail0 : List Nat := 4943 :: root613Tail1
def root613Candidates0 : List Nat := [7919]
set_option maxRecDepth 8192 in
theorem root613Candidates0_checked : candidates edge 4 4547 4943 root613Tail1 winner.bound = root613Candidates0 := by
  unfold candidates
  have row : edge 4943 = (SortedNatTree.contains FastRows.row4943) := by
    change Data.row4943.contains = (SortedNatTree.contains FastRows.row4943)
    exact FastRows.row4943_checked.symm
  rw [row]
  decide
theorem root613Step0 : scan edge (search edge 3) 4 [4547] 4547 root613Tail0 winner =
    scan edge (search edge 3) 4 [4547] 4547 root613Tail1 winner := by
  rw [root613Tail0, scan, if_neg (by decide), root613Candidates0_checked]
  change scan edge (search edge 3) 4 [4547] 4547 root613Tail1
    (if 3 ≤ root613Candidates0.length then search edge 3 [4547, 4943] root613Candidates0 9490 winner else winner) = _
  rw [if_neg (by decide)]
def root613Candidates1 : List Nat := [16319]
set_option maxRecDepth 8192 in
theorem root613Candidates1_checked : candidates edge 4 4547 4973 root613Tail2 winner.bound = root613Candidates1 := by
  unfold candidates
  have row : edge 4973 = (SortedNatTree.contains FastRows.row4973) := by
    change Data.row4973.contains = (SortedNatTree.contains FastRows.row4973)
    exact FastRows.row4973_checked.symm
  rw [row]
  decide
theorem root613Step1 : scan edge (search edge 3) 4 [4547] 4547 root613Tail1 winner =
    scan edge (search edge 3) 4 [4547] 4547 root613Tail2 winner := by
  rw [root613Tail1, scan, if_neg (by decide), root613Candidates1_checked]
  change scan edge (search edge 3) 4 [4547] 4547 root613Tail2
    (if 3 ≤ root613Candidates1.length then search edge 3 [4547, 4973] root613Candidates1 9520 winner else winner) = _
  rw [if_neg (by decide)]
theorem root613Scan2 : scan edge (search edge 3) 4 [4547] 4547 root613Tail2 winner = winner := by rw [root613Tail2, scan, if_pos (by decide)]
theorem root613Scan1 : scan edge (search edge 3) 4 [4547] 4547 root613Tail1 winner = winner :=
  (root613Step1).trans root613Scan2
theorem root613Scan0 : scan edge (search edge 3) 4 [4547] 4547 root613Tail0 winner = winner :=
  (root613Step0).trans root613Scan1
set_option maxRecDepth 4096 in
theorem root613Branch : search edge 4 [4547] root613Candidates 4547 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4547] 4547 root613Tail0 winner = winner
  exact root613Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
