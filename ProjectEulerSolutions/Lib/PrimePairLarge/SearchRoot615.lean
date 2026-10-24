import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root615Candidates : List Nat := Data.row4561
set_option maxRecDepth 8192 in
theorem root615Candidates_checked : candidates edge 5 0 4561 tail616 winner.bound = root615Candidates := by
  apply candidates_eq_row edge 5 0 4561 tail616 root615Candidates winner.bound tail616_ordered
  · decide
  · decide
  · funext q
    rfl
def root615Tail34 : List Nat := []
def root615Tail33 : List Nat := 21319 :: root615Tail34
def root615Tail32 : List Nat := 21031 :: root615Tail33
def root615Tail31 : List Nat := 19429 :: root615Tail32
def root615Tail30 : List Nat := 17491 :: root615Tail31
def root615Tail29 : List Nat := 15817 :: root615Tail30
def root615Tail28 : List Nat := 15271 :: root615Tail29
def root615Tail27 : List Nat := 14107 :: root615Tail28
def root615Tail26 : List Nat := 13729 :: root615Tail27
def root615Tail25 : List Nat := 13381 :: root615Tail26
def root615Tail24 : List Nat := 13309 :: root615Tail25
def root615Tail23 : List Nat := 13249 :: root615Tail24
def root615Tail22 : List Nat := 12967 :: root615Tail23
def root615Tail21 : List Nat := 12037 :: root615Tail22
def root615Tail20 : List Nat := 11833 :: root615Tail21
def root615Tail19 : List Nat := 11527 :: root615Tail20
def root615Tail18 : List Nat := 10957 :: root615Tail19
def root615Tail17 : List Nat := 10477 :: root615Tail18
def root615Tail16 : List Nat := 10321 :: root615Tail17
def root615Tail15 : List Nat := 9907 :: root615Tail16
def root615Tail14 : List Nat := 9811 :: root615Tail15
def root615Tail13 : List Nat := 9283 :: root615Tail14
def root615Tail12 : List Nat := 8863 :: root615Tail13
def root615Tail11 : List Nat := 8779 :: root615Tail12
def root615Tail10 : List Nat := 8737 :: root615Tail11
def root615Tail9 : List Nat := 8389 :: root615Tail10
def root615Tail8 : List Nat := 8101 :: root615Tail9
def root615Tail7 : List Nat := 8017 :: root615Tail8
def root615Tail6 : List Nat := 7723 :: root615Tail7
def root615Tail5 : List Nat := 6379 :: root615Tail6
def root615Tail4 : List Nat := 6073 :: root615Tail5
def root615Tail3 : List Nat := 5527 :: root615Tail4
def root615Tail2 : List Nat := 5419 :: root615Tail3
def root615Tail1 : List Nat := 4909 :: root615Tail2
def root615Tail0 : List Nat := 4783 :: root615Tail1
def root615Candidates0 : List Nat := [6073]
set_option maxRecDepth 8192 in
theorem root615Candidates0_checked : candidates edge 4 4561 4783 root615Tail1 winner.bound = root615Candidates0 := by
  unfold candidates
  have row : edge 4783 = (SortedNatTree.contains FastRows.row4783) := by
    change Data.row4783.contains = (SortedNatTree.contains FastRows.row4783)
    exact FastRows.row4783_checked.symm
  rw [row]
  decide
theorem root615Step0 : scan edge (search edge 3) 4 [4561] 4561 root615Tail0 winner =
    scan edge (search edge 3) 4 [4561] 4561 root615Tail1 winner := by
  rw [root615Tail0, scan, if_neg (by decide), root615Candidates0_checked]
  change scan edge (search edge 3) 4 [4561] 4561 root615Tail1
    (if 3 ≤ root615Candidates0.length then search edge 3 [4561, 4783] root615Candidates0 9344 winner else winner) = _
  rw [if_neg (by decide)]
def root615Candidates1 : List Nat := [7723, 9907, 10477, 10957]
set_option maxRecDepth 8192 in
theorem root615Candidates1_checked : candidates edge 4 4561 4909 root615Tail2 winner.bound = root615Candidates1 := by
  unfold candidates
  have row : edge 4909 = (SortedNatTree.contains FastRows.row4909) := by
    change Data.row4909.contains = (SortedNatTree.contains FastRows.row4909)
    exact FastRows.row4909_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root615Child1 : search edge 3 [4561, 4909] root615Candidates1 9470 winner = winner := by decide
theorem root615Step1 : scan edge (search edge 3) 4 [4561] 4561 root615Tail1 winner =
    scan edge (search edge 3) 4 [4561] 4561 root615Tail2 winner := by
  rw [root615Tail1, scan, if_neg (by decide), root615Candidates1_checked]
  change scan edge (search edge 3) 4 [4561] 4561 root615Tail2
    (if 3 ≤ root615Candidates1.length then search edge 3 [4561, 4909] root615Candidates1 9470 winner else winner) = _
  rw [if_pos (by decide), root615Child1]
theorem root615Scan2 : scan edge (search edge 3) 4 [4561] 4561 root615Tail2 winner = winner := by rw [root615Tail2, scan, if_pos (by decide)]
theorem root615Scan1 : scan edge (search edge 3) 4 [4561] 4561 root615Tail1 winner = winner :=
  (root615Step1).trans root615Scan2
theorem root615Scan0 : scan edge (search edge 3) 4 [4561] 4561 root615Tail0 winner = winner :=
  (root615Step0).trans root615Scan1
set_option maxRecDepth 4096 in
theorem root615Branch : search edge 4 [4561] root615Candidates 4561 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4561] 4561 root615Tail0 winner = winner
  exact root615Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
