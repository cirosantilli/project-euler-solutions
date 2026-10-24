import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root446Candidates : List Nat := Data.row3169
set_option maxRecDepth 8192 in
theorem root446Candidates_checked : candidates edge 5 0 3169 tail447 winner.bound = root446Candidates := by
  apply candidates_eq_row edge 5 0 3169 tail447 root446Candidates winner.bound tail447_ordered
  · decide
  · decide
  · funext q
    rfl
def root446Tail28 : List Nat := []
def root446Tail27 : List Nat := 22651 :: root446Tail28
def root446Tail26 : List Nat := 22051 :: root446Tail27
def root446Tail25 : List Nat := 19717 :: root446Tail26
def root446Tail24 : List Nat := 18493 :: root446Tail25
def root446Tail23 : List Nat := 17683 :: root446Tail24
def root446Tail22 : List Nat := 15559 :: root446Tail23
def root446Tail21 : List Nat := 15541 :: root446Tail22
def root446Tail20 : List Nat := 15493 :: root446Tail21
def root446Tail19 : List Nat := 14503 :: root446Tail20
def root446Tail18 : List Nat := 13873 :: root446Tail19
def root446Tail17 : List Nat := 12823 :: root446Tail18
def root446Tail16 : List Nat := 12697 :: root446Tail17
def root446Tail15 : List Nat := 12049 :: root446Tail16
def root446Tail14 : List Nat := 11149 :: root446Tail15
def root446Tail13 : List Nat := 10639 :: root446Tail14
def root446Tail12 : List Nat := 10369 :: root446Tail13
def root446Tail11 : List Nat := 8419 :: root446Tail12
def root446Tail10 : List Nat := 8377 :: root446Tail11
def root446Tail9 : List Nat := 8179 :: root446Tail10
def root446Tail8 : List Nat := 7213 :: root446Tail9
def root446Tail7 : List Nat := 6361 :: root446Tail8
def root446Tail6 : List Nat := 6037 :: root446Tail7
def root446Tail5 : List Nat := 5953 :: root446Tail6
def root446Tail4 : List Nat := 5119 :: root446Tail5
def root446Tail3 : List Nat := 5101 :: root446Tail4
def root446Tail2 : List Nat := 3733 :: root446Tail3
def root446Tail1 : List Nat := 3391 :: root446Tail2
def root446Tail0 : List Nat := 3229 :: root446Tail1
def root446Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root446Candidates0_checked : candidates edge 4 3169 3229 root446Tail1 winner.bound = root446Candidates0 := by
  unfold candidates
  have row : edge 3229 = (SortedNatTree.contains FastRows.row3229) := by
    change Data.row3229.contains = (SortedNatTree.contains FastRows.row3229)
    exact FastRows.row3229_checked.symm
  rw [row]
  decide
theorem root446Step0 : scan edge (search edge 3) 4 [3169] 3169 root446Tail0 winner =
    scan edge (search edge 3) 4 [3169] 3169 root446Tail1 winner := by
  rw [root446Tail0, scan, if_neg (by decide), root446Candidates0_checked]
  change scan edge (search edge 3) 4 [3169] 3169 root446Tail1
    (if 3 ≤ root446Candidates0.length then search edge 3 [3169, 3229] root446Candidates0 6398 winner else winner) = _
  rw [if_neg (by decide)]
def root446Candidates1 : List Nat := [6361, 12049, 15559]
set_option maxRecDepth 8192 in
theorem root446Candidates1_checked : candidates edge 4 3169 3391 root446Tail2 winner.bound = root446Candidates1 := by
  unfold candidates
  have row : edge 3391 = (SortedNatTree.contains FastRows.row3391) := by
    change Data.row3391.contains = (SortedNatTree.contains FastRows.row3391)
    exact FastRows.row3391_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root446Child1 : search edge 3 [3169, 3391] root446Candidates1 6560 winner = winner := by decide
theorem root446Step1 : scan edge (search edge 3) 4 [3169] 3169 root446Tail1 winner =
    scan edge (search edge 3) 4 [3169] 3169 root446Tail2 winner := by
  rw [root446Tail1, scan, if_neg (by decide), root446Candidates1_checked]
  change scan edge (search edge 3) 4 [3169] 3169 root446Tail2
    (if 3 ≤ root446Candidates1.length then search edge 3 [3169, 3391] root446Candidates1 6560 winner else winner) = _
  rw [if_pos (by decide), root446Child1]
def root446Candidates2 : List Nat := [12697, 15541]
set_option maxRecDepth 8192 in
theorem root446Candidates2_checked : candidates edge 4 3169 3733 root446Tail3 winner.bound = root446Candidates2 := by
  unfold candidates
  have row : edge 3733 = (SortedNatTree.contains FastRows.row3733) := by
    change Data.row3733.contains = (SortedNatTree.contains FastRows.row3733)
    exact FastRows.row3733_checked.symm
  rw [row]
  decide
theorem root446Step2 : scan edge (search edge 3) 4 [3169] 3169 root446Tail2 winner =
    scan edge (search edge 3) 4 [3169] 3169 root446Tail3 winner := by
  rw [root446Tail2, scan, if_neg (by decide), root446Candidates2_checked]
  change scan edge (search edge 3) 4 [3169] 3169 root446Tail3
    (if 3 ≤ root446Candidates2.length then search edge 3 [3169, 3733] root446Candidates2 6902 winner else winner) = _
  rw [if_neg (by decide)]
def root446Candidates3 : List Nat := [5953, 10369]
set_option maxRecDepth 8192 in
theorem root446Candidates3_checked : candidates edge 4 3169 5101 root446Tail4 winner.bound = root446Candidates3 := by
  unfold candidates
  have row : edge 5101 = (SortedNatTree.contains FastRows.row5101) := by
    change Data.row5101.contains = (SortedNatTree.contains FastRows.row5101)
    exact FastRows.row5101_checked.symm
  rw [row]
  decide
theorem root446Step3 : scan edge (search edge 3) 4 [3169] 3169 root446Tail3 winner =
    scan edge (search edge 3) 4 [3169] 3169 root446Tail4 winner := by
  rw [root446Tail3, scan, if_neg (by decide), root446Candidates3_checked]
  change scan edge (search edge 3) 4 [3169] 3169 root446Tail4
    (if 3 ≤ root446Candidates3.length then search edge 3 [3169, 5101] root446Candidates3 8270 winner else winner) = _
  rw [if_neg (by decide)]
def root446Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root446Candidates4_checked : candidates edge 4 3169 5119 root446Tail5 winner.bound = root446Candidates4 := by
  unfold candidates
  have row : edge 5119 = (SortedNatTree.contains FastRows.row5119) := by
    change Data.row5119.contains = (SortedNatTree.contains FastRows.row5119)
    exact FastRows.row5119_checked.symm
  rw [row]
  decide
theorem root446Step4 : scan edge (search edge 3) 4 [3169] 3169 root446Tail4 winner =
    scan edge (search edge 3) 4 [3169] 3169 root446Tail5 winner := by
  rw [root446Tail4, scan, if_neg (by decide), root446Candidates4_checked]
  change scan edge (search edge 3) 4 [3169] 3169 root446Tail5
    (if 3 ≤ root446Candidates4.length then search edge 3 [3169, 5119] root446Candidates4 8288 winner else winner) = _
  rw [if_neg (by decide)]
theorem root446Scan5 : scan edge (search edge 3) 4 [3169] 3169 root446Tail5 winner = winner := by rw [root446Tail5, scan, if_pos (by decide)]
theorem root446Scan4 : scan edge (search edge 3) 4 [3169] 3169 root446Tail4 winner = winner :=
  (root446Step4).trans root446Scan5
theorem root446Scan3 : scan edge (search edge 3) 4 [3169] 3169 root446Tail3 winner = winner :=
  (root446Step3).trans root446Scan4
theorem root446Scan2 : scan edge (search edge 3) 4 [3169] 3169 root446Tail2 winner = winner :=
  (root446Step2).trans root446Scan3
theorem root446Scan1 : scan edge (search edge 3) 4 [3169] 3169 root446Tail1 winner = winner :=
  (root446Step1).trans root446Scan2
theorem root446Scan0 : scan edge (search edge 3) 4 [3169] 3169 root446Tail0 winner = winner :=
  (root446Step0).trans root446Scan1
set_option maxRecDepth 4096 in
theorem root446Branch : search edge 4 [3169] root446Candidates 3169 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3169] 3169 root446Tail0 winner = winner
  exact root446Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
