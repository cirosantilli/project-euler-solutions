import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root527Candidates : List Nat := Data.row3821
set_option maxRecDepth 8192 in
theorem root527Candidates_checked : candidates edge 5 0 3821 tail528 winner.bound = root527Candidates := by
  apply candidates_eq_row edge 5 0 3821 tail528 root527Candidates winner.bound tail528_ordered
  · decide
  · decide
  · funext q
    rfl
def root527Tail33 : List Nat := []
def root527Tail32 : List Nat := 21101 :: root527Tail33
def root527Tail31 : List Nat := 20117 :: root527Tail32
def root527Tail30 : List Nat := 19403 :: root527Tail31
def root527Tail29 : List Nat := 18917 :: root527Tail30
def root527Tail28 : List Nat := 18371 :: root527Tail29
def root527Tail27 : List Nat := 17627 :: root527Tail28
def root527Tail26 : List Nat := 17609 :: root527Tail27
def root527Tail25 : List Nat := 16937 :: root527Tail26
def root527Tail24 : List Nat := 16451 :: root527Tail25
def root527Tail23 : List Nat := 15497 :: root527Tail24
def root527Tail22 : List Nat := 14867 :: root527Tail23
def root527Tail21 : List Nat := 14669 :: root527Tail22
def root527Tail20 : List Nat := 14657 :: root527Tail21
def root527Tail19 : List Nat := 14633 :: root527Tail20
def root527Tail18 : List Nat := 14591 :: root527Tail19
def root527Tail17 : List Nat := 13679 :: root527Tail18
def root527Tail16 : List Nat := 13241 :: root527Tail17
def root527Tail15 : List Nat := 13187 :: root527Tail16
def root527Tail14 : List Nat := 12821 :: root527Tail15
def root527Tail13 : List Nat := 11399 :: root527Tail14
def root527Tail12 : List Nat := 10067 :: root527Tail13
def root527Tail11 : List Nat := 9803 :: root527Tail12
def root527Tail10 : List Nat := 9539 :: root527Tail11
def root527Tail9 : List Nat := 8951 :: root527Tail10
def root527Tail8 : List Nat := 8933 :: root527Tail9
def root527Tail7 : List Nat := 7307 :: root527Tail8
def root527Tail6 : List Nat := 7019 :: root527Tail7
def root527Tail5 : List Nat := 6917 :: root527Tail6
def root527Tail4 : List Nat := 6863 :: root527Tail5
def root527Tail3 : List Nat := 6473 :: root527Tail4
def root527Tail2 : List Nat := 6263 :: root527Tail3
def root527Tail1 : List Nat := 6047 :: root527Tail2
def root527Tail0 : List Nat := 4703 :: root527Tail1
def root527Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root527Candidates0_checked : candidates edge 4 3821 4703 root527Tail1 winner.bound = root527Candidates0 := by
  unfold candidates
  have row : edge 4703 = (SortedNatTree.contains FastRows.row4703) := by
    change Data.row4703.contains = (SortedNatTree.contains FastRows.row4703)
    exact FastRows.row4703_checked.symm
  rw [row]
  decide
theorem root527Step0 : scan edge (search edge 3) 4 [3821] 3821 root527Tail0 winner =
    scan edge (search edge 3) 4 [3821] 3821 root527Tail1 winner := by
  rw [root527Tail0, scan, if_neg (by decide), root527Candidates0_checked]
  change scan edge (search edge 3) 4 [3821] 3821 root527Tail1
    (if 3 ≤ root527Candidates0.length then search edge 3 [3821, 4703] root527Candidates0 8524 winner else winner) = _
  rw [if_neg (by decide)]
theorem root527Scan1 : scan edge (search edge 3) 4 [3821] 3821 root527Tail1 winner = winner := by rw [root527Tail1, scan, if_pos (by decide)]
theorem root527Scan0 : scan edge (search edge 3) 4 [3821] 3821 root527Tail0 winner = winner :=
  (root527Step0).trans root527Scan1
set_option maxRecDepth 4096 in
theorem root527Branch : search edge 4 [3821] root527Candidates 3821 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3821] 3821 root527Tail0 winner = winner
  exact root527Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
