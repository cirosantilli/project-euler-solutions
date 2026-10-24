import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root592Candidates : List Nat := Data.row4357
set_option maxRecDepth 8192 in
theorem root592Candidates_checked : candidates edge 5 0 4357 tail593 winner.bound = root592Candidates := by
  apply candidates_eq_row edge 5 0 4357 tail593 root592Candidates winner.bound tail593_ordered
  · decide
  · decide
  · funext q
    rfl
def root592Tail43 : List Nat := []
def root592Tail42 : List Nat := 21613 :: root592Tail43
def root592Tail41 : List Nat := 21487 :: root592Tail42
def root592Tail40 : List Nat := 21397 :: root592Tail41
def root592Tail39 : List Nat := 19861 :: root592Tail40
def root592Tail38 : List Nat := 19813 :: root592Tail39
def root592Tail37 : List Nat := 18553 :: root592Tail38
def root592Tail36 : List Nat := 18307 :: root592Tail37
def root592Tail35 : List Nat := 18289 :: root592Tail36
def root592Tail34 : List Nat := 18133 :: root592Tail35
def root592Tail33 : List Nat := 18121 :: root592Tail34
def root592Tail32 : List Nat := 18043 :: root592Tail33
def root592Tail31 : List Nat := 17959 :: root592Tail32
def root592Tail30 : List Nat := 17749 :: root592Tail31
def root592Tail29 : List Nat := 17383 :: root592Tail30
def root592Tail28 : List Nat := 17167 :: root592Tail29
def root592Tail27 : List Nat := 16693 :: root592Tail28
def root592Tail26 : List Nat := 16573 :: root592Tail27
def root592Tail25 : List Nat := 15901 :: root592Tail26
def root592Tail24 : List Nat := 15739 :: root592Tail25
def root592Tail23 : List Nat := 15451 :: root592Tail24
def root592Tail22 : List Nat := 14551 :: root592Tail23
def root592Tail21 : List Nat := 14341 :: root592Tail22
def root592Tail20 : List Nat := 13183 :: root592Tail21
def root592Tail19 : List Nat := 12973 :: root592Tail20
def root592Tail18 : List Nat := 12799 :: root592Tail19
def root592Tail17 : List Nat := 11491 :: root592Tail18
def root592Tail16 : List Nat := 9859 :: root592Tail17
def root592Tail15 : List Nat := 9001 :: root592Tail16
def root592Tail14 : List Nat := 8893 :: root592Tail15
def root592Tail13 : List Nat := 8209 :: root592Tail14
def root592Tail12 : List Nat := 8053 :: root592Tail13
def root592Tail11 : List Nat := 7411 :: root592Tail12
def root592Tail10 : List Nat := 7297 :: root592Tail11
def root592Tail9 : List Nat := 6829 :: root592Tail10
def root592Tail8 : List Nat := 6691 :: root592Tail9
def root592Tail7 : List Nat := 6619 :: root592Tail8
def root592Tail6 : List Nat := 5563 :: root592Tail7
def root592Tail5 : List Nat := 5479 :: root592Tail6
def root592Tail4 : List Nat := 5449 :: root592Tail5
def root592Tail3 : List Nat := 5431 :: root592Tail4
def root592Tail2 : List Nat := 5107 :: root592Tail3
def root592Tail1 : List Nat := 5101 :: root592Tail2
def root592Tail0 : List Nat := 4813 :: root592Tail1
def root592Candidates0 : List Nat := [16573]
set_option maxRecDepth 8192 in
theorem root592Candidates0_checked : candidates edge 4 4357 4813 root592Tail1 winner.bound = root592Candidates0 := by
  unfold candidates
  have row : edge 4813 = (SortedNatTree.contains FastRows.row4813) := by
    change Data.row4813.contains = (SortedNatTree.contains FastRows.row4813)
    exact FastRows.row4813_checked.symm
  rw [row]
  decide
theorem root592Step0 : scan edge (search edge 3) 4 [4357] 4357 root592Tail0 winner =
    scan edge (search edge 3) 4 [4357] 4357 root592Tail1 winner := by
  rw [root592Tail0, scan, if_neg (by decide), root592Candidates0_checked]
  change scan edge (search edge 3) 4 [4357] 4357 root592Tail1
    (if 3 ≤ root592Candidates0.length then search edge 3 [4357, 4813] root592Candidates0 9170 winner else winner) = _
  rw [if_neg (by decide)]
def root592Candidates1 : List Nat := [14551]
set_option maxRecDepth 8192 in
theorem root592Candidates1_checked : candidates edge 4 4357 5101 root592Tail2 winner.bound = root592Candidates1 := by
  unfold candidates
  have row : edge 5101 = (SortedNatTree.contains FastRows.row5101) := by
    change Data.row5101.contains = (SortedNatTree.contains FastRows.row5101)
    exact FastRows.row5101_checked.symm
  rw [row]
  decide
theorem root592Step1 : scan edge (search edge 3) 4 [4357] 4357 root592Tail1 winner =
    scan edge (search edge 3) 4 [4357] 4357 root592Tail2 winner := by
  rw [root592Tail1, scan, if_neg (by decide), root592Candidates1_checked]
  change scan edge (search edge 3) 4 [4357] 4357 root592Tail2
    (if 3 ≤ root592Candidates1.length then search edge 3 [4357, 5101] root592Candidates1 9458 winner else winner) = _
  rw [if_neg (by decide)]
def root592Candidates2 : List Nat := [14551]
set_option maxRecDepth 8192 in
theorem root592Candidates2_checked : candidates edge 4 4357 5107 root592Tail3 winner.bound = root592Candidates2 := by
  unfold candidates
  have row : edge 5107 = (SortedNatTree.contains FastRows.row5107) := by
    change Data.row5107.contains = (SortedNatTree.contains FastRows.row5107)
    exact FastRows.row5107_checked.symm
  rw [row]
  decide
theorem root592Step2 : scan edge (search edge 3) 4 [4357] 4357 root592Tail2 winner =
    scan edge (search edge 3) 4 [4357] 4357 root592Tail3 winner := by
  rw [root592Tail2, scan, if_neg (by decide), root592Candidates2_checked]
  change scan edge (search edge 3) 4 [4357] 4357 root592Tail3
    (if 3 ≤ root592Candidates2.length then search edge 3 [4357, 5107] root592Candidates2 9464 winner else winner) = _
  rw [if_neg (by decide)]
theorem root592Scan3 : scan edge (search edge 3) 4 [4357] 4357 root592Tail3 winner = winner := by rw [root592Tail3, scan, if_pos (by decide)]
theorem root592Scan2 : scan edge (search edge 3) 4 [4357] 4357 root592Tail2 winner = winner :=
  (root592Step2).trans root592Scan3
theorem root592Scan1 : scan edge (search edge 3) 4 [4357] 4357 root592Tail1 winner = winner :=
  (root592Step1).trans root592Scan2
theorem root592Scan0 : scan edge (search edge 3) 4 [4357] 4357 root592Tail0 winner = winner :=
  (root592Step0).trans root592Scan1
set_option maxRecDepth 4096 in
theorem root592Branch : search edge 4 [4357] root592Candidates 4357 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4357] 4357 root592Tail0 winner = winner
  exact root592Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
