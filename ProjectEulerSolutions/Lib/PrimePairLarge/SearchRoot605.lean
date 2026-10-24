import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root605Candidates : List Nat := Data.row4481
set_option maxRecDepth 8192 in
theorem root605Candidates_checked : candidates edge 5 0 4481 tail606 winner.bound = root605Candidates := by
  apply candidates_eq_row edge 5 0 4481 tail606 root605Candidates winner.bound tail606_ordered
  · decide
  · decide
  · funext q
    rfl
def root605Tail31 : List Nat := []
def root605Tail30 : List Nat := 21491 :: root605Tail31
def root605Tail29 : List Nat := 21269 :: root605Tail30
def root605Tail28 : List Nat := 21227 :: root605Tail29
def root605Tail27 : List Nat := 20297 :: root605Tail28
def root605Tail26 : List Nat := 19583 :: root605Tail27
def root605Tail25 : List Nat := 19391 :: root605Tail26
def root605Tail24 : List Nat := 18329 :: root605Tail25
def root605Tail23 : List Nat := 17231 :: root605Tail24
def root605Tail22 : List Nat := 16979 :: root605Tail23
def root605Tail21 : List Nat := 15809 :: root605Tail22
def root605Tail20 : List Nat := 15749 :: root605Tail21
def root605Tail19 : List Nat := 15401 :: root605Tail20
def root605Tail18 : List Nat := 15329 :: root605Tail19
def root605Tail17 : List Nat := 14657 :: root605Tail18
def root605Tail16 : List Nat := 13397 :: root605Tail17
def root605Tail15 : List Nat := 12641 :: root605Tail16
def root605Tail14 : List Nat := 11423 :: root605Tail15
def root605Tail13 : List Nat := 11003 :: root605Tail14
def root605Tail12 : List Nat := 10847 :: root605Tail13
def root605Tail11 : List Nat := 10139 :: root605Tail12
def root605Tail10 : List Nat := 9743 :: root605Tail11
def root605Tail9 : List Nat := 8831 :: root605Tail10
def root605Tail8 : List Nat := 8117 :: root605Tail9
def root605Tail7 : List Nat := 7907 :: root605Tail8
def root605Tail6 : List Nat := 7583 :: root605Tail7
def root605Tail5 : List Nat := 7529 :: root605Tail6
def root605Tail4 : List Nat := 7193 :: root605Tail5
def root605Tail3 : List Nat := 6791 :: root605Tail4
def root605Tail2 : List Nat := 6491 :: root605Tail3
def root605Tail1 : List Nat := 6203 :: root605Tail2
def root605Tail0 : List Nat := 5189 :: root605Tail1
def root605Candidates0 : List Nat := [6491]
set_option maxRecDepth 8192 in
theorem root605Candidates0_checked : candidates edge 4 4481 5189 root605Tail1 winner.bound = root605Candidates0 := by
  unfold candidates
  have row : edge 5189 = (SortedNatTree.contains FastRows.row5189) := by
    change Data.row5189.contains = (SortedNatTree.contains FastRows.row5189)
    exact FastRows.row5189_checked.symm
  rw [row]
  decide
theorem root605Step0 : scan edge (search edge 3) 4 [4481] 4481 root605Tail0 winner =
    scan edge (search edge 3) 4 [4481] 4481 root605Tail1 winner := by
  rw [root605Tail0, scan, if_neg (by decide), root605Candidates0_checked]
  change scan edge (search edge 3) 4 [4481] 4481 root605Tail1
    (if 3 ≤ root605Candidates0.length then search edge 3 [4481, 5189] root605Candidates0 9670 winner else winner) = _
  rw [if_neg (by decide)]
theorem root605Scan1 : scan edge (search edge 3) 4 [4481] 4481 root605Tail1 winner = winner := by rw [root605Tail1, scan, if_pos (by decide)]
theorem root605Scan0 : scan edge (search edge 3) 4 [4481] 4481 root605Tail0 winner = winner :=
  (root605Step0).trans root605Scan1
set_option maxRecDepth 4096 in
theorem root605Branch : search edge 4 [4481] root605Candidates 4481 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4481] 4481 root605Tail0 winner = winner
  exact root605Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
