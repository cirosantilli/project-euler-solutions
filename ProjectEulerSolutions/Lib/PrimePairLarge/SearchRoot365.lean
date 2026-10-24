import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root365Candidates : List Nat := Data.row2503
set_option maxRecDepth 8192 in
theorem root365Candidates_checked : candidates edge 5 0 2503 tail366 winner.bound = root365Candidates := by
  apply candidates_eq_row edge 5 0 2503 tail366 root365Candidates winner.bound tail366_ordered
  · decide
  · decide
  · funext q
    rfl
def root365Tail45 : List Nat := []
def root365Tail44 : List Nat := 23251 :: root365Tail45
def root365Tail43 : List Nat := 23131 :: root365Tail44
def root365Tail42 : List Nat := 22717 :: root365Tail43
def root365Tail41 : List Nat := 22543 :: root365Tail42
def root365Tail40 : List Nat := 22381 :: root365Tail41
def root365Tail39 : List Nat := 22129 :: root365Tail40
def root365Tail38 : List Nat := 21529 :: root365Tail39
def root365Tail37 : List Nat := 20533 :: root365Tail38
def root365Tail36 : List Nat := 20143 :: root365Tail37
def root365Tail35 : List Nat := 19777 :: root365Tail36
def root365Tail34 : List Nat := 19231 :: root365Tail35
def root365Tail33 : List Nat := 18979 :: root365Tail34
def root365Tail32 : List Nat := 18379 :: root365Tail33
def root365Tail31 : List Nat := 18181 :: root365Tail32
def root365Tail30 : List Nat := 17923 :: root365Tail31
def root365Tail29 : List Nat := 17581 :: root365Tail30
def root365Tail28 : List Nat := 17167 :: root365Tail29
def root365Tail27 : List Nat := 15913 :: root365Tail28
def root365Tail26 : List Nat := 15817 :: root365Tail27
def root365Tail25 : List Nat := 14149 :: root365Tail26
def root365Tail24 : List Nat := 13291 :: root365Tail25
def root365Tail23 : List Nat := 12637 :: root365Tail24
def root365Tail22 : List Nat := 12619 :: root365Tail23
def root365Tail21 : List Nat := 11827 :: root365Tail22
def root365Tail20 : List Nat := 10993 :: root365Tail21
def root365Tail19 : List Nat := 9613 :: root365Tail20
def root365Tail18 : List Nat := 9601 :: root365Tail19
def root365Tail17 : List Nat := 9439 :: root365Tail18
def root365Tail16 : List Nat := 9319 :: root365Tail17
def root365Tail15 : List Nat := 8719 :: root365Tail16
def root365Tail14 : List Nat := 7591 :: root365Tail15
def root365Tail13 : List Nat := 7393 :: root365Tail14
def root365Tail12 : List Nat := 6967 :: root365Tail13
def root365Tail11 : List Nat := 6763 :: root365Tail12
def root365Tail10 : List Nat := 6421 :: root365Tail11
def root365Tail9 : List Nat := 6163 :: root365Tail10
def root365Tail8 : List Nat := 6079 :: root365Tail9
def root365Tail7 : List Nat := 5869 :: root365Tail8
def root365Tail6 : List Nat := 5623 :: root365Tail7
def root365Tail5 : List Nat := 5581 :: root365Tail6
def root365Tail4 : List Nat := 5281 :: root365Tail5
def root365Tail3 : List Nat := 4447 :: root365Tail4
def root365Tail2 : List Nat := 3559 :: root365Tail3
def root365Tail1 : List Nat := 3517 :: root365Tail2
def root365Tail0 : List Nat := 3391 :: root365Tail1
def root365Candidates0 : List Nat := [6079, 9439, 15817]
set_option maxRecDepth 8192 in
theorem root365Candidates0_checked : candidates edge 4 2503 3391 root365Tail1 winner.bound = root365Candidates0 := by
  unfold candidates
  have row : edge 3391 = (SortedNatTree.contains FastRows.row3391) := by
    change Data.row3391.contains = (SortedNatTree.contains FastRows.row3391)
    exact FastRows.row3391_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root365Child0 : search edge 3 [2503, 3391] root365Candidates0 5894 winner = winner := by decide
theorem root365Step0 : scan edge (search edge 3) 4 [2503] 2503 root365Tail0 winner =
    scan edge (search edge 3) 4 [2503] 2503 root365Tail1 winner := by
  rw [root365Tail0, scan, if_neg (by decide), root365Candidates0_checked]
  change scan edge (search edge 3) 4 [2503] 2503 root365Tail1
    (if 3 ≤ root365Candidates0.length then search edge 3 [2503, 3391] root365Candidates0 5894 winner else winner) = _
  rw [if_pos (by decide), root365Child0]
def root365Candidates1 : List Nat := [15817]
set_option maxRecDepth 8192 in
theorem root365Candidates1_checked : candidates edge 4 2503 3517 root365Tail2 winner.bound = root365Candidates1 := by
  unfold candidates
  have row : edge 3517 = (SortedNatTree.contains FastRows.row3517) := by
    change Data.row3517.contains = (SortedNatTree.contains FastRows.row3517)
    exact FastRows.row3517_checked.symm
  rw [row]
  decide
theorem root365Step1 : scan edge (search edge 3) 4 [2503] 2503 root365Tail1 winner =
    scan edge (search edge 3) 4 [2503] 2503 root365Tail2 winner := by
  rw [root365Tail1, scan, if_neg (by decide), root365Candidates1_checked]
  change scan edge (search edge 3) 4 [2503] 2503 root365Tail2
    (if 3 ≤ root365Candidates1.length then search edge 3 [2503, 3517] root365Candidates1 6020 winner else winner) = _
  rw [if_neg (by decide)]
def root365Candidates2 : List Nat := [5869]
set_option maxRecDepth 8192 in
theorem root365Candidates2_checked : candidates edge 4 2503 3559 root365Tail3 winner.bound = root365Candidates2 := by
  unfold candidates
  have row : edge 3559 = (SortedNatTree.contains FastRows.row3559) := by
    change Data.row3559.contains = (SortedNatTree.contains FastRows.row3559)
    exact FastRows.row3559_checked.symm
  rw [row]
  decide
theorem root365Step2 : scan edge (search edge 3) 4 [2503] 2503 root365Tail2 winner =
    scan edge (search edge 3) 4 [2503] 2503 root365Tail3 winner := by
  rw [root365Tail2, scan, if_neg (by decide), root365Candidates2_checked]
  change scan edge (search edge 3) 4 [2503] 2503 root365Tail3
    (if 3 ≤ root365Candidates2.length then search edge 3 [2503, 3559] root365Candidates2 6062 winner else winner) = _
  rw [if_neg (by decide)]
def root365Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root365Candidates3_checked : candidates edge 4 2503 4447 root365Tail4 winner.bound = root365Candidates3 := by
  unfold candidates
  have row : edge 4447 = (SortedNatTree.contains FastRows.row4447) := by
    change Data.row4447.contains = (SortedNatTree.contains FastRows.row4447)
    exact FastRows.row4447_checked.symm
  rw [row]
  decide
theorem root365Step3 : scan edge (search edge 3) 4 [2503] 2503 root365Tail3 winner =
    scan edge (search edge 3) 4 [2503] 2503 root365Tail4 winner := by
  rw [root365Tail3, scan, if_neg (by decide), root365Candidates3_checked]
  change scan edge (search edge 3) 4 [2503] 2503 root365Tail4
    (if 3 ≤ root365Candidates3.length then search edge 3 [2503, 4447] root365Candidates3 6950 winner else winner) = _
  rw [if_neg (by decide)]
def root365Candidates4 : List Nat := [5869]
set_option maxRecDepth 8192 in
theorem root365Candidates4_checked : candidates edge 4 2503 5281 root365Tail5 winner.bound = root365Candidates4 := by
  unfold candidates
  have row : edge 5281 = (SortedNatTree.contains FastRows.row5281) := by
    change Data.row5281.contains = (SortedNatTree.contains FastRows.row5281)
    exact FastRows.row5281_checked.symm
  rw [row]
  decide
theorem root365Step4 : scan edge (search edge 3) 4 [2503] 2503 root365Tail4 winner =
    scan edge (search edge 3) 4 [2503] 2503 root365Tail5 winner := by
  rw [root365Tail4, scan, if_neg (by decide), root365Candidates4_checked]
  change scan edge (search edge 3) 4 [2503] 2503 root365Tail5
    (if 3 ≤ root365Candidates4.length then search edge 3 [2503, 5281] root365Candidates4 7784 winner else winner) = _
  rw [if_neg (by decide)]
def root365Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root365Candidates5_checked : candidates edge 4 2503 5581 root365Tail6 winner.bound = root365Candidates5 := by
  unfold candidates
  have row : edge 5581 = (SortedNatTree.contains FastRows.row5581) := by
    change Data.row5581.contains = (SortedNatTree.contains FastRows.row5581)
    exact FastRows.row5581_checked.symm
  rw [row]
  decide
theorem root365Step5 : scan edge (search edge 3) 4 [2503] 2503 root365Tail5 winner =
    scan edge (search edge 3) 4 [2503] 2503 root365Tail6 winner := by
  rw [root365Tail5, scan, if_neg (by decide), root365Candidates5_checked]
  change scan edge (search edge 3) 4 [2503] 2503 root365Tail6
    (if 3 ≤ root365Candidates5.length then search edge 3 [2503, 5581] root365Candidates5 8084 winner else winner) = _
  rw [if_neg (by decide)]
def root365Candidates6 : List Nat := [6421, 9319]
set_option maxRecDepth 8192 in
theorem root365Candidates6_checked : candidates edge 4 2503 5623 root365Tail7 winner.bound = root365Candidates6 := by
  unfold candidates
  have row : edge 5623 = (SortedNatTree.contains FastRows.row5623) := by
    change Data.row5623.contains = (SortedNatTree.contains FastRows.row5623)
    exact FastRows.row5623_checked.symm
  rw [row]
  decide
theorem root365Step6 : scan edge (search edge 3) 4 [2503] 2503 root365Tail6 winner =
    scan edge (search edge 3) 4 [2503] 2503 root365Tail7 winner := by
  rw [root365Tail6, scan, if_neg (by decide), root365Candidates6_checked]
  change scan edge (search edge 3) 4 [2503] 2503 root365Tail7
    (if 3 ≤ root365Candidates6.length then search edge 3 [2503, 5623] root365Candidates6 8126 winner else winner) = _
  rw [if_neg (by decide)]
def root365Candidates7 : List Nat := [6163, 9439, 10993]
set_option maxRecDepth 8192 in
theorem root365Candidates7_checked : candidates edge 4 2503 5869 root365Tail8 winner.bound = root365Candidates7 := by
  unfold candidates
  have row : edge 5869 = (SortedNatTree.contains FastRows.row5869) := by
    change Data.row5869.contains = (SortedNatTree.contains FastRows.row5869)
    exact FastRows.row5869_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root365Child7 : search edge 3 [2503, 5869] root365Candidates7 8372 winner = winner := by decide
theorem root365Step7 : scan edge (search edge 3) 4 [2503] 2503 root365Tail7 winner =
    scan edge (search edge 3) 4 [2503] 2503 root365Tail8 winner := by
  rw [root365Tail7, scan, if_neg (by decide), root365Candidates7_checked]
  change scan edge (search edge 3) 4 [2503] 2503 root365Tail8
    (if 3 ≤ root365Candidates7.length then search edge 3 [2503, 5869] root365Candidates7 8372 winner else winner) = _
  rw [if_pos (by decide), root365Child7]
theorem root365Scan8 : scan edge (search edge 3) 4 [2503] 2503 root365Tail8 winner = winner := by rw [root365Tail8, scan, if_pos (by decide)]
theorem root365Scan7 : scan edge (search edge 3) 4 [2503] 2503 root365Tail7 winner = winner :=
  (root365Step7).trans root365Scan8
theorem root365Scan6 : scan edge (search edge 3) 4 [2503] 2503 root365Tail6 winner = winner :=
  (root365Step6).trans root365Scan7
theorem root365Scan5 : scan edge (search edge 3) 4 [2503] 2503 root365Tail5 winner = winner :=
  (root365Step5).trans root365Scan6
theorem root365Scan4 : scan edge (search edge 3) 4 [2503] 2503 root365Tail4 winner = winner :=
  (root365Step4).trans root365Scan5
theorem root365Scan3 : scan edge (search edge 3) 4 [2503] 2503 root365Tail3 winner = winner :=
  (root365Step3).trans root365Scan4
theorem root365Scan2 : scan edge (search edge 3) 4 [2503] 2503 root365Tail2 winner = winner :=
  (root365Step2).trans root365Scan3
theorem root365Scan1 : scan edge (search edge 3) 4 [2503] 2503 root365Tail1 winner = winner :=
  (root365Step1).trans root365Scan2
theorem root365Scan0 : scan edge (search edge 3) 4 [2503] 2503 root365Tail0 winner = winner :=
  (root365Step0).trans root365Scan1
set_option maxRecDepth 4096 in
theorem root365Branch : search edge 4 [2503] root365Candidates 2503 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2503] 2503 root365Tail0 winner = winner
  exact root365Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
