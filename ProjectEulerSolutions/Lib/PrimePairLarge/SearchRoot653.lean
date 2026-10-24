import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root653Candidates : List Nat := Data.row4909
set_option maxRecDepth 8192 in
theorem root653Candidates_checked : candidates edge 5 0 4909 tail654 winner.bound = root653Candidates := by
  apply candidates_eq_row edge 5 0 4909 tail654 root653Candidates winner.bound tail654_ordered
  · decide
  · decide
  · funext q
    rfl
def root653Tail28 : List Nat := []
def root653Tail27 : List Nat := 20719 :: root653Tail28
def root653Tail26 : List Nat := 20071 :: root653Tail27
def root653Tail25 : List Nat := 19543 :: root653Tail26
def root653Tail24 : List Nat := 19237 :: root653Tail25
def root653Tail23 : List Nat := 18859 :: root653Tail24
def root653Tail22 : List Nat := 17989 :: root653Tail23
def root653Tail21 : List Nat := 14947 :: root653Tail22
def root653Tail20 : List Nat := 14821 :: root653Tail21
def root653Tail19 : List Nat := 14029 :: root653Tail20
def root653Tail18 : List Nat := 13627 :: root653Tail19
def root653Tail17 : List Nat := 13441 :: root653Tail18
def root653Tail16 : List Nat := 10957 :: root653Tail17
def root653Tail15 : List Nat := 10837 :: root653Tail16
def root653Tail14 : List Nat := 10477 :: root653Tail15
def root653Tail13 : List Nat := 10369 :: root653Tail14
def root653Tail12 : List Nat := 9907 :: root653Tail13
def root653Tail11 : List Nat := 9157 :: root653Tail12
def root653Tail10 : List Nat := 9013 :: root653Tail11
def root653Tail9 : List Nat := 8713 :: root653Tail10
def root653Tail8 : List Nat := 8293 :: root653Tail9
def root653Tail7 : List Nat := 7933 :: root653Tail8
def root653Tail6 : List Nat := 7723 :: root653Tail7
def root653Tail5 : List Nat := 7177 :: root653Tail6
def root653Tail4 : List Nat := 5857 :: root653Tail5
def root653Tail3 : List Nat := 5647 :: root653Tail4
def root653Tail2 : List Nat := 5503 :: root653Tail3
def root653Tail1 : List Nat := 5443 :: root653Tail2
def root653Tail0 : List Nat := 5023 :: root653Tail1
def root653Candidates0 : List Nat := [5443, 9907]
set_option maxRecDepth 8192 in
theorem root653Candidates0_checked : candidates edge 4 4909 5023 root653Tail1 winner.bound = root653Candidates0 := by
  unfold candidates
  have row : edge 5023 = (SortedNatTree.contains FastRows.row5023) := by
    change Data.row5023.contains = (SortedNatTree.contains FastRows.row5023)
    exact FastRows.row5023_checked.symm
  rw [row]
  decide
theorem root653Step0 : scan edge (search edge 3) 4 [4909] 4909 root653Tail0 winner =
    scan edge (search edge 3) 4 [4909] 4909 root653Tail1 winner := by
  rw [root653Tail0, scan, if_neg (by decide), root653Candidates0_checked]
  change scan edge (search edge 3) 4 [4909] 4909 root653Tail1
    (if 3 ≤ root653Candidates0.length then search edge 3 [4909, 5023] root653Candidates0 9932 winner else winner) = _
  rw [if_neg (by decide)]
theorem root653Scan1 : scan edge (search edge 3) 4 [4909] 4909 root653Tail1 winner = winner := by rw [root653Tail1, scan, if_pos (by decide)]
theorem root653Scan0 : scan edge (search edge 3) 4 [4909] 4909 root653Tail0 winner = winner :=
  (root653Step0).trans root653Scan1
set_option maxRecDepth 4096 in
theorem root653Branch : search edge 4 [4909] root653Candidates 4909 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4909] 4909 root653Tail0 winner = winner
  exact root653Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
