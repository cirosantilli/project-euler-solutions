import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root628Candidates : List Nat := Data.row4663
set_option maxRecDepth 8192 in
theorem root628Candidates_checked : candidates edge 5 0 4663 tail629 winner.bound = root628Candidates := by
  apply candidates_eq_row edge 5 0 4663 tail629 root628Candidates winner.bound tail629_ordered
  · decide
  · decide
  · funext q
    rfl
def root628Tail34 : List Nat := []
def root628Tail33 : List Nat := 20773 :: root628Tail34
def root628Tail32 : List Nat := 20479 :: root628Tail33
def root628Tail31 : List Nat := 19753 :: root628Tail32
def root628Tail30 : List Nat := 19543 :: root628Tail31
def root628Tail29 : List Nat := 18439 :: root628Tail30
def root628Tail28 : List Nat := 18367 :: root628Tail29
def root628Tail27 : List Nat := 17851 :: root628Tail28
def root628Tail26 : List Nat := 17749 :: root628Tail27
def root628Tail25 : List Nat := 17569 :: root628Tail26
def root628Tail24 : List Nat := 17107 :: root628Tail25
def root628Tail23 : List Nat := 17077 :: root628Tail24
def root628Tail22 : List Nat := 16057 :: root628Tail23
def root628Tail21 : List Nat := 15061 :: root628Tail22
def root628Tail20 : List Nat := 14923 :: root628Tail21
def root628Tail19 : List Nat := 14731 :: root628Tail20
def root628Tail18 : List Nat := 14533 :: root628Tail19
def root628Tail17 : List Nat := 12799 :: root628Tail18
def root628Tail16 : List Nat := 11113 :: root628Tail17
def root628Tail15 : List Nat := 10399 :: root628Tail16
def root628Tail14 : List Nat := 9949 :: root628Tail15
def root628Tail13 : List Nat := 9811 :: root628Tail14
def root628Tail12 : List Nat := 9787 :: root628Tail13
def root628Tail11 : List Nat := 9601 :: root628Tail12
def root628Tail10 : List Nat := 9199 :: root628Tail11
def root628Tail9 : List Nat := 8233 :: root628Tail10
def root628Tail8 : List Nat := 8017 :: root628Tail9
def root628Tail7 : List Nat := 7879 :: root628Tail8
def root628Tail6 : List Nat := 7753 :: root628Tail7
def root628Tail5 : List Nat := 7603 :: root628Tail6
def root628Tail4 : List Nat := 6637 :: root628Tail5
def root628Tail3 : List Nat := 6619 :: root628Tail4
def root628Tail2 : List Nat := 5839 :: root628Tail3
def root628Tail1 : List Nat := 5011 :: root628Tail2
def root628Tail0 : List Nat := 4993 :: root628Tail1
def root628Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root628Candidates0_checked : candidates edge 4 4663 4993 root628Tail1 winner.bound = root628Candidates0 := by
  unfold candidates
  have row : edge 4993 = (SortedNatTree.contains FastRows.row4993) := by
    change Data.row4993.contains = (SortedNatTree.contains FastRows.row4993)
    exact FastRows.row4993_checked.symm
  rw [row]
  decide
theorem root628Step0 : scan edge (search edge 3) 4 [4663] 4663 root628Tail0 winner =
    scan edge (search edge 3) 4 [4663] 4663 root628Tail1 winner := by
  rw [root628Tail0, scan, if_neg (by decide), root628Candidates0_checked]
  change scan edge (search edge 3) 4 [4663] 4663 root628Tail1
    (if 3 ≤ root628Candidates0.length then search edge 3 [4663, 4993] root628Candidates0 9656 winner else winner) = _
  rw [if_neg (by decide)]
def root628Candidates1 : List Nat := [14533]
set_option maxRecDepth 8192 in
theorem root628Candidates1_checked : candidates edge 4 4663 5011 root628Tail2 winner.bound = root628Candidates1 := by
  unfold candidates
  have row : edge 5011 = (SortedNatTree.contains FastRows.row5011) := by
    change Data.row5011.contains = (SortedNatTree.contains FastRows.row5011)
    exact FastRows.row5011_checked.symm
  rw [row]
  decide
theorem root628Step1 : scan edge (search edge 3) 4 [4663] 4663 root628Tail1 winner =
    scan edge (search edge 3) 4 [4663] 4663 root628Tail2 winner := by
  rw [root628Tail1, scan, if_neg (by decide), root628Candidates1_checked]
  change scan edge (search edge 3) 4 [4663] 4663 root628Tail2
    (if 3 ≤ root628Candidates1.length then search edge 3 [4663, 5011] root628Candidates1 9674 winner else winner) = _
  rw [if_neg (by decide)]
theorem root628Scan2 : scan edge (search edge 3) 4 [4663] 4663 root628Tail2 winner = winner := by rw [root628Tail2, scan, if_pos (by decide)]
theorem root628Scan1 : scan edge (search edge 3) 4 [4663] 4663 root628Tail1 winner = winner :=
  (root628Step1).trans root628Scan2
theorem root628Scan0 : scan edge (search edge 3) 4 [4663] 4663 root628Tail0 winner = winner :=
  (root628Step0).trans root628Scan1
set_option maxRecDepth 4096 in
theorem root628Branch : search edge 4 [4663] root628Candidates 4663 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4663] 4663 root628Tail0 winner = winner
  exact root628Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
