import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root534Candidates : List Nat := Data.row3877
set_option maxRecDepth 8192 in
theorem root534Candidates_checked : candidates edge 5 0 3877 tail535 winner.bound = root534Candidates := by
  apply candidates_eq_row edge 5 0 3877 tail535 root534Candidates winner.bound tail535_ordered
  · decide
  · decide
  · funext q
    rfl
def root534Tail27 : List Nat := []
def root534Tail26 : List Nat := 21799 :: root534Tail27
def root534Tail25 : List Nat := 21487 :: root534Tail26
def root534Tail24 : List Nat := 21013 :: root534Tail25
def root534Tail23 : List Nat := 20731 :: root534Tail24
def root534Tail22 : List Nat := 19543 :: root534Tail23
def root534Tail21 : List Nat := 19471 :: root534Tail22
def root534Tail20 : List Nat := 19219 :: root534Tail21
def root534Tail19 : List Nat := 18679 :: root534Tail20
def root534Tail18 : List Nat := 18553 :: root534Tail19
def root534Tail17 : List Nat := 18049 :: root534Tail18
def root534Tail16 : List Nat := 17431 :: root534Tail17
def root534Tail15 : List Nat := 15217 :: root534Tail16
def root534Tail14 : List Nat := 12781 :: root534Tail15
def root534Tail13 : List Nat := 10501 :: root534Tail14
def root534Tail12 : List Nat := 9049 :: root534Tail13
def root534Tail11 : List Nat := 8263 :: root534Tail12
def root534Tail10 : List Nat := 7159 :: root534Tail11
def root534Tail9 : List Nat := 6607 :: root534Tail10
def root534Tail8 : List Nat := 6469 :: root534Tail9
def root534Tail7 : List Nat := 6373 :: root534Tail8
def root534Tail6 : List Nat := 6163 :: root534Tail7
def root534Tail5 : List Nat := 6151 :: root534Tail6
def root534Tail4 : List Nat := 5683 :: root534Tail5
def root534Tail3 : List Nat := 5479 :: root534Tail4
def root534Tail2 : List Nat := 5227 :: root534Tail3
def root534Tail1 : List Nat := 5167 :: root534Tail2
def root534Tail0 : List Nat := 4507 :: root534Tail1
def root534Candidates0 : List Nat := [15217]
set_option maxRecDepth 8192 in
theorem root534Candidates0_checked : candidates edge 4 3877 4507 root534Tail1 winner.bound = root534Candidates0 := by
  unfold candidates
  have row : edge 4507 = (SortedNatTree.contains FastRows.row4507) := by
    change Data.row4507.contains = (SortedNatTree.contains FastRows.row4507)
    exact FastRows.row4507_checked.symm
  rw [row]
  decide
theorem root534Step0 : scan edge (search edge 3) 4 [3877] 3877 root534Tail0 winner =
    scan edge (search edge 3) 4 [3877] 3877 root534Tail1 winner := by
  rw [root534Tail0, scan, if_neg (by decide), root534Candidates0_checked]
  change scan edge (search edge 3) 4 [3877] 3877 root534Tail1
    (if 3 ≤ root534Candidates0.length then search edge 3 [3877, 4507] root534Candidates0 8384 winner else winner) = _
  rw [if_neg (by decide)]
def root534Candidates1 : List Nat := [10501]
set_option maxRecDepth 8192 in
theorem root534Candidates1_checked : candidates edge 4 3877 5167 root534Tail2 winner.bound = root534Candidates1 := by
  unfold candidates
  have row : edge 5167 = (SortedNatTree.contains FastRows.row5167) := by
    change Data.row5167.contains = (SortedNatTree.contains FastRows.row5167)
    exact FastRows.row5167_checked.symm
  rw [row]
  decide
theorem root534Step1 : scan edge (search edge 3) 4 [3877] 3877 root534Tail1 winner =
    scan edge (search edge 3) 4 [3877] 3877 root534Tail2 winner := by
  rw [root534Tail1, scan, if_neg (by decide), root534Candidates1_checked]
  change scan edge (search edge 3) 4 [3877] 3877 root534Tail2
    (if 3 ≤ root534Candidates1.length then search edge 3 [3877, 5167] root534Candidates1 9044 winner else winner) = _
  rw [if_neg (by decide)]
def root534Candidates2 : List Nat := [5683]
set_option maxRecDepth 8192 in
theorem root534Candidates2_checked : candidates edge 4 3877 5227 root534Tail3 winner.bound = root534Candidates2 := by
  unfold candidates
  have row : edge 5227 = (SortedNatTree.contains FastRows.row5227) := by
    change Data.row5227.contains = (SortedNatTree.contains FastRows.row5227)
    exact FastRows.row5227_checked.symm
  rw [row]
  decide
theorem root534Step2 : scan edge (search edge 3) 4 [3877] 3877 root534Tail2 winner =
    scan edge (search edge 3) 4 [3877] 3877 root534Tail3 winner := by
  rw [root534Tail2, scan, if_neg (by decide), root534Candidates2_checked]
  change scan edge (search edge 3) 4 [3877] 3877 root534Tail3
    (if 3 ≤ root534Candidates2.length then search edge 3 [3877, 5227] root534Candidates2 9104 winner else winner) = _
  rw [if_neg (by decide)]
def root534Candidates3 : List Nat := [6607, 10501]
set_option maxRecDepth 8192 in
theorem root534Candidates3_checked : candidates edge 4 3877 5479 root534Tail4 winner.bound = root534Candidates3 := by
  unfold candidates
  have row : edge 5479 = (SortedNatTree.contains FastRows.row5479) := by
    change Data.row5479.contains = (SortedNatTree.contains FastRows.row5479)
    exact FastRows.row5479_checked.symm
  rw [row]
  decide
theorem root534Step3 : scan edge (search edge 3) 4 [3877] 3877 root534Tail3 winner =
    scan edge (search edge 3) 4 [3877] 3877 root534Tail4 winner := by
  rw [root534Tail3, scan, if_neg (by decide), root534Candidates3_checked]
  change scan edge (search edge 3) 4 [3877] 3877 root534Tail4
    (if 3 ≤ root534Candidates3.length then search edge 3 [3877, 5479] root534Candidates3 9356 winner else winner) = _
  rw [if_neg (by decide)]
theorem root534Scan4 : scan edge (search edge 3) 4 [3877] 3877 root534Tail4 winner = winner := by rw [root534Tail4, scan, if_pos (by decide)]
theorem root534Scan3 : scan edge (search edge 3) 4 [3877] 3877 root534Tail3 winner = winner :=
  (root534Step3).trans root534Scan4
theorem root534Scan2 : scan edge (search edge 3) 4 [3877] 3877 root534Tail2 winner = winner :=
  (root534Step2).trans root534Scan3
theorem root534Scan1 : scan edge (search edge 3) 4 [3877] 3877 root534Tail1 winner = winner :=
  (root534Step1).trans root534Scan2
theorem root534Scan0 : scan edge (search edge 3) 4 [3877] 3877 root534Tail0 winner = winner :=
  (root534Step0).trans root534Scan1
set_option maxRecDepth 4096 in
theorem root534Branch : search edge 4 [3877] root534Candidates 3877 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3877] 3877 root534Tail0 winner = winner
  exact root534Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
