import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root300Candidates : List Nat := Data.row1999
set_option maxRecDepth 8192 in
theorem root300Candidates_checked : candidates edge 5 0 1999 tail301 winner.bound = root300Candidates := by
  apply candidates_eq_row edge 5 0 1999 tail301 root300Candidates winner.bound tail301_ordered
  · decide
  · decide
  · funext q
    rfl
def root300Tail41 : List Nat := []
def root300Tail40 : List Nat := 23671 :: root300Tail41
def root300Tail39 : List Nat := 23599 :: root300Tail40
def root300Tail38 : List Nat := 22501 :: root300Tail39
def root300Tail37 : List Nat := 22369 :: root300Tail38
def root300Tail36 : List Nat := 22291 :: root300Tail37
def root300Tail35 : List Nat := 21193 :: root300Tail36
def root300Tail34 : List Nat := 21031 :: root300Tail35
def root300Tail33 : List Nat := 19483 :: root300Tail34
def root300Tail32 : List Nat := 19441 :: root300Tail33
def root300Tail31 : List Nat := 19309 :: root300Tail32
def root300Tail30 : List Nat := 18679 :: root300Tail31
def root300Tail29 : List Nat := 17299 :: root300Tail30
def root300Tail28 : List Nat := 16747 :: root300Tail29
def root300Tail27 : List Nat := 16567 :: root300Tail28
def root300Tail26 : List Nat := 16231 :: root300Tail27
def root300Tail25 : List Nat := 16033 :: root300Tail26
def root300Tail24 : List Nat := 15787 :: root300Tail25
def root300Tail23 : List Nat := 14563 :: root300Tail24
def root300Tail22 : List Nat := 14557 :: root300Tail23
def root300Tail21 : List Nat := 14437 :: root300Tail22
def root300Tail20 : List Nat := 13267 :: root300Tail21
def root300Tail19 : List Nat := 12541 :: root300Tail20
def root300Tail18 : List Nat := 12421 :: root300Tail19
def root300Tail17 : List Nat := 12379 :: root300Tail18
def root300Tail16 : List Nat := 11149 :: root300Tail17
def root300Tail15 : List Nat := 10657 :: root300Tail16
def root300Tail14 : List Nat := 10567 :: root300Tail15
def root300Tail13 : List Nat := 9739 :: root300Tail14
def root300Tail12 : List Nat := 8779 :: root300Tail13
def root300Tail11 : List Nat := 8263 :: root300Tail12
def root300Tail10 : List Nat := 7717 :: root300Tail11
def root300Tail9 : List Nat := 6961 :: root300Tail10
def root300Tail8 : List Nat := 6469 :: root300Tail9
def root300Tail7 : List Nat := 6121 :: root300Tail8
def root300Tail6 : List Nat := 5869 :: root300Tail7
def root300Tail5 : List Nat := 3733 :: root300Tail6
def root300Tail4 : List Nat := 3607 :: root300Tail5
def root300Tail3 : List Nat := 3469 :: root300Tail4
def root300Tail2 : List Nat := 3361 :: root300Tail3
def root300Tail1 : List Nat := 2521 :: root300Tail2
def root300Tail0 : List Nat := 2347 :: root300Tail1
def root300Candidates0 : List Nat := [6469]
set_option maxRecDepth 8192 in
theorem root300Candidates0_checked : candidates edge 4 1999 2347 root300Tail1 winner.bound = root300Candidates0 := by
  unfold candidates
  have row : edge 2347 = (SortedNatTree.contains FastRows.row2347) := by
    change Data.row2347.contains = (SortedNatTree.contains FastRows.row2347)
    exact FastRows.row2347_checked.symm
  rw [row]
  decide
theorem root300Step0 : scan edge (search edge 3) 4 [1999] 1999 root300Tail0 winner =
    scan edge (search edge 3) 4 [1999] 1999 root300Tail1 winner := by
  rw [root300Tail0, scan, if_neg (by decide), root300Candidates0_checked]
  change scan edge (search edge 3) 4 [1999] 1999 root300Tail1
    (if 3 ≤ root300Candidates0.length then search edge 3 [1999, 2347] root300Candidates0 4346 winner else winner) = _
  rw [if_neg (by decide)]
def root300Candidates1 : List Nat := [16747]
set_option maxRecDepth 8192 in
theorem root300Candidates1_checked : candidates edge 4 1999 2521 root300Tail2 winner.bound = root300Candidates1 := by
  unfold candidates
  have row : edge 2521 = (SortedNatTree.contains FastRows.row2521) := by
    change Data.row2521.contains = (SortedNatTree.contains FastRows.row2521)
    exact FastRows.row2521_checked.symm
  rw [row]
  decide
theorem root300Step1 : scan edge (search edge 3) 4 [1999] 1999 root300Tail1 winner =
    scan edge (search edge 3) 4 [1999] 1999 root300Tail2 winner := by
  rw [root300Tail1, scan, if_neg (by decide), root300Candidates1_checked]
  change scan edge (search edge 3) 4 [1999] 1999 root300Tail2
    (if 3 ≤ root300Candidates1.length then search edge 3 [1999, 2521] root300Candidates1 4520 winner else winner) = _
  rw [if_neg (by decide)]
def root300Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root300Candidates2_checked : candidates edge 4 1999 3361 root300Tail3 winner.bound = root300Candidates2 := by
  unfold candidates
  have row : edge 3361 = (SortedNatTree.contains FastRows.row3361) := by
    change Data.row3361.contains = (SortedNatTree.contains FastRows.row3361)
    exact FastRows.row3361_checked.symm
  rw [row]
  decide
theorem root300Step2 : scan edge (search edge 3) 4 [1999] 1999 root300Tail2 winner =
    scan edge (search edge 3) 4 [1999] 1999 root300Tail3 winner := by
  rw [root300Tail2, scan, if_neg (by decide), root300Candidates2_checked]
  change scan edge (search edge 3) 4 [1999] 1999 root300Tail3
    (if 3 ≤ root300Candidates2.length then search edge 3 [1999, 3361] root300Candidates2 5360 winner else winner) = _
  rw [if_neg (by decide)]
def root300Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root300Candidates3_checked : candidates edge 4 1999 3469 root300Tail4 winner.bound = root300Candidates3 := by
  unfold candidates
  have row : edge 3469 = (SortedNatTree.contains FastRows.row3469) := by
    change Data.row3469.contains = (SortedNatTree.contains FastRows.row3469)
    exact FastRows.row3469_checked.symm
  rw [row]
  decide
theorem root300Step3 : scan edge (search edge 3) 4 [1999] 1999 root300Tail3 winner =
    scan edge (search edge 3) 4 [1999] 1999 root300Tail4 winner := by
  rw [root300Tail3, scan, if_neg (by decide), root300Candidates3_checked]
  change scan edge (search edge 3) 4 [1999] 1999 root300Tail4
    (if 3 ≤ root300Candidates3.length then search edge 3 [1999, 3469] root300Candidates3 5468 winner else winner) = _
  rw [if_neg (by decide)]
def root300Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root300Candidates4_checked : candidates edge 4 1999 3607 root300Tail5 winner.bound = root300Candidates4 := by
  unfold candidates
  have row : edge 3607 = (SortedNatTree.contains FastRows.row3607) := by
    change Data.row3607.contains = (SortedNatTree.contains FastRows.row3607)
    exact FastRows.row3607_checked.symm
  rw [row]
  decide
theorem root300Step4 : scan edge (search edge 3) 4 [1999] 1999 root300Tail4 winner =
    scan edge (search edge 3) 4 [1999] 1999 root300Tail5 winner := by
  rw [root300Tail4, scan, if_neg (by decide), root300Candidates4_checked]
  change scan edge (search edge 3) 4 [1999] 1999 root300Tail5
    (if 3 ≤ root300Candidates4.length then search edge 3 [1999, 3607] root300Candidates4 5606 winner else winner) = _
  rw [if_neg (by decide)]
def root300Candidates5 : List Nat := [16231]
set_option maxRecDepth 8192 in
theorem root300Candidates5_checked : candidates edge 4 1999 3733 root300Tail6 winner.bound = root300Candidates5 := by
  unfold candidates
  have row : edge 3733 = (SortedNatTree.contains FastRows.row3733) := by
    change Data.row3733.contains = (SortedNatTree.contains FastRows.row3733)
    exact FastRows.row3733_checked.symm
  rw [row]
  decide
theorem root300Step5 : scan edge (search edge 3) 4 [1999] 1999 root300Tail5 winner =
    scan edge (search edge 3) 4 [1999] 1999 root300Tail6 winner := by
  rw [root300Tail5, scan, if_neg (by decide), root300Candidates5_checked]
  change scan edge (search edge 3) 4 [1999] 1999 root300Tail6
    (if 3 ≤ root300Candidates5.length then search edge 3 [1999, 3733] root300Candidates5 5732 winner else winner) = _
  rw [if_neg (by decide)]
def root300Candidates6 : List Nat := [6961, 10657]
set_option maxRecDepth 8192 in
theorem root300Candidates6_checked : candidates edge 4 1999 5869 root300Tail7 winner.bound = root300Candidates6 := by
  unfold candidates
  have row : edge 5869 = (SortedNatTree.contains FastRows.row5869) := by
    change Data.row5869.contains = (SortedNatTree.contains FastRows.row5869)
    exact FastRows.row5869_checked.symm
  rw [row]
  decide
theorem root300Step6 : scan edge (search edge 3) 4 [1999] 1999 root300Tail6 winner =
    scan edge (search edge 3) 4 [1999] 1999 root300Tail7 winner := by
  rw [root300Tail6, scan, if_neg (by decide), root300Candidates6_checked]
  change scan edge (search edge 3) 4 [1999] 1999 root300Tail7
    (if 3 ≤ root300Candidates6.length then search edge 3 [1999, 5869] root300Candidates6 7868 winner else winner) = _
  rw [if_neg (by decide)]
theorem root300Scan7 : scan edge (search edge 3) 4 [1999] 1999 root300Tail7 winner = winner := by rw [root300Tail7, scan, if_pos (by decide)]
theorem root300Scan6 : scan edge (search edge 3) 4 [1999] 1999 root300Tail6 winner = winner :=
  (root300Step6).trans root300Scan7
theorem root300Scan5 : scan edge (search edge 3) 4 [1999] 1999 root300Tail5 winner = winner :=
  (root300Step5).trans root300Scan6
theorem root300Scan4 : scan edge (search edge 3) 4 [1999] 1999 root300Tail4 winner = winner :=
  (root300Step4).trans root300Scan5
theorem root300Scan3 : scan edge (search edge 3) 4 [1999] 1999 root300Tail3 winner = winner :=
  (root300Step3).trans root300Scan4
theorem root300Scan2 : scan edge (search edge 3) 4 [1999] 1999 root300Tail2 winner = winner :=
  (root300Step2).trans root300Scan3
theorem root300Scan1 : scan edge (search edge 3) 4 [1999] 1999 root300Tail1 winner = winner :=
  (root300Step1).trans root300Scan2
theorem root300Scan0 : scan edge (search edge 3) 4 [1999] 1999 root300Tail0 winner = winner :=
  (root300Step0).trans root300Scan1
set_option maxRecDepth 4096 in
theorem root300Branch : search edge 4 [1999] root300Candidates 1999 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1999] 1999 root300Tail0 winner = winner
  exact root300Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
