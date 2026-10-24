import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root488Candidates : List Nat := Data.row3517
set_option maxRecDepth 8192 in
theorem root488Candidates_checked : candidates edge 5 0 3517 tail489 winner.bound = root488Candidates := by
  apply candidates_eq_row edge 5 0 3517 tail489 root488Candidates winner.bound tail489_ordered
  · decide
  · decide
  · funext q
    rfl
def root488Tail33 : List Nat := []
def root488Tail32 : List Nat := 22039 :: root488Tail33
def root488Tail31 : List Nat := 21817 :: root488Tail32
def root488Tail30 : List Nat := 21529 :: root488Tail31
def root488Tail29 : List Nat := 21499 :: root488Tail30
def root488Tail28 : List Nat := 21481 :: root488Tail29
def root488Tail27 : List Nat := 21397 :: root488Tail28
def root488Tail26 : List Nat := 20731 :: root488Tail27
def root488Tail25 : List Nat := 19813 :: root488Tail26
def root488Tail24 : List Nat := 15817 :: root488Tail25
def root488Tail23 : List Nat := 15349 :: root488Tail24
def root488Tail22 : List Nat := 15319 :: root488Tail23
def root488Tail21 : List Nat := 14401 :: root488Tail22
def root488Tail20 : List Nat := 13759 :: root488Tail21
def root488Tail19 : List Nat := 13597 :: root488Tail20
def root488Tail18 : List Nat := 13177 :: root488Tail19
def root488Tail17 : List Nat := 11149 :: root488Tail18
def root488Tail16 : List Nat := 10663 :: root488Tail17
def root488Tail15 : List Nat := 9883 :: root488Tail16
def root488Tail14 : List Nat := 9769 :: root488Tail15
def root488Tail13 : List Nat := 9433 :: root488Tail14
def root488Tail12 : List Nat := 8599 :: root488Tail13
def root488Tail11 : List Nat := 8263 :: root488Tail12
def root488Tail10 : List Nat := 8161 :: root488Tail11
def root488Tail9 : List Nat := 7717 :: root488Tail10
def root488Tail8 : List Nat := 7669 :: root488Tail9
def root488Tail7 : List Nat := 7621 :: root488Tail8
def root488Tail6 : List Nat := 7507 :: root488Tail7
def root488Tail5 : List Nat := 7159 :: root488Tail6
def root488Tail4 : List Nat := 6907 :: root488Tail5
def root488Tail3 : List Nat := 6829 :: root488Tail4
def root488Tail2 : List Nat := 5659 :: root488Tail3
def root488Tail1 : List Nat := 4177 :: root488Tail2
def root488Tail0 : List Nat := 4099 :: root488Tail1
def root488Candidates0 : List Nat := [13597, 15817]
set_option maxRecDepth 8192 in
theorem root488Candidates0_checked : candidates edge 4 3517 4099 root488Tail1 winner.bound = root488Candidates0 := by
  unfold candidates
  have row : edge 4099 = (SortedNatTree.contains FastRows.row4099) := by
    change Data.row4099.contains = (SortedNatTree.contains FastRows.row4099)
    exact FastRows.row4099_checked.symm
  rw [row]
  decide
theorem root488Step0 : scan edge (search edge 3) 4 [3517] 3517 root488Tail0 winner =
    scan edge (search edge 3) 4 [3517] 3517 root488Tail1 winner := by
  rw [root488Tail0, scan, if_neg (by decide), root488Candidates0_checked]
  change scan edge (search edge 3) 4 [3517] 3517 root488Tail1
    (if 3 ≤ root488Candidates0.length then search edge 3 [3517, 4099] root488Candidates0 7616 winner else winner) = _
  rw [if_neg (by decide)]
def root488Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root488Candidates1_checked : candidates edge 4 3517 4177 root488Tail2 winner.bound = root488Candidates1 := by
  unfold candidates
  have row : edge 4177 = (SortedNatTree.contains FastRows.row4177) := by
    change Data.row4177.contains = (SortedNatTree.contains FastRows.row4177)
    exact FastRows.row4177_checked.symm
  rw [row]
  decide
theorem root488Step1 : scan edge (search edge 3) 4 [3517] 3517 root488Tail1 winner =
    scan edge (search edge 3) 4 [3517] 3517 root488Tail2 winner := by
  rw [root488Tail1, scan, if_neg (by decide), root488Candidates1_checked]
  change scan edge (search edge 3) 4 [3517] 3517 root488Tail2
    (if 3 ≤ root488Candidates1.length then search edge 3 [3517, 4177] root488Candidates1 7694 winner else winner) = _
  rw [if_neg (by decide)]
theorem root488Scan2 : scan edge (search edge 3) 4 [3517] 3517 root488Tail2 winner = winner := by rw [root488Tail2, scan, if_pos (by decide)]
theorem root488Scan1 : scan edge (search edge 3) 4 [3517] 3517 root488Tail1 winner = winner :=
  (root488Step1).trans root488Scan2
theorem root488Scan0 : scan edge (search edge 3) 4 [3517] 3517 root488Tail0 winner = winner :=
  (root488Step0).trans root488Scan1
set_option maxRecDepth 4096 in
theorem root488Branch : search edge 4 [3517] root488Candidates 3517 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3517] 3517 root488Tail0 winner = winner
  exact root488Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
