import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root491Candidates : List Nat := Data.row3533
set_option maxRecDepth 8192 in
theorem root491Candidates_checked : candidates edge 5 0 3533 tail492 winner.bound = root491Candidates := by
  apply candidates_eq_row edge 5 0 3533 tail492 root491Candidates winner.bound tail492_ordered
  · decide
  · decide
  · funext q
    rfl
def root491Tail37 : List Nat := []
def root491Tail36 : List Nat := 22349 :: root491Tail37
def root491Tail35 : List Nat := 22247 :: root491Tail36
def root491Tail34 : List Nat := 20771 :: root491Tail35
def root491Tail33 : List Nat := 20483 :: root491Tail34
def root491Tail32 : List Nat := 20399 :: root491Tail33
def root491Tail31 : List Nat := 19937 :: root491Tail32
def root491Tail30 : List Nat := 18947 :: root491Tail31
def root491Tail29 : List Nat := 18839 :: root491Tail30
def root491Tail28 : List Nat := 15227 :: root491Tail29
def root491Tail27 : List Nat := 13967 :: root491Tail28
def root491Tail26 : List Nat := 13613 :: root491Tail27
def root491Tail25 : List Nat := 11171 :: root491Tail26
def root491Tail24 : List Nat := 11087 :: root491Tail25
def root491Tail23 : List Nat := 10709 :: root491Tail24
def root491Tail22 : List Nat := 10691 :: root491Tail23
def root491Tail21 : List Nat := 10313 :: root491Tail22
def root491Tail20 : List Nat := 10289 :: root491Tail21
def root491Tail19 : List Nat := 10169 :: root491Tail20
def root491Tail18 : List Nat := 10163 :: root491Tail19
def root491Tail17 : List Nat := 9539 :: root491Tail18
def root491Tail16 : List Nat := 9341 :: root491Tail17
def root491Tail15 : List Nat := 9323 :: root491Tail16
def root491Tail14 : List Nat := 9173 :: root491Tail15
def root491Tail13 : List Nat := 8951 :: root491Tail14
def root491Tail12 : List Nat := 8861 :: root491Tail13
def root491Tail11 : List Nat := 8273 :: root491Tail12
def root491Tail10 : List Nat := 7607 :: root491Tail11
def root491Tail9 : List Nat := 7079 :: root491Tail10
def root491Tail8 : List Nat := 7019 :: root491Tail9
def root491Tail7 : List Nat := 6113 :: root491Tail8
def root491Tail6 : List Nat := 5483 :: root491Tail7
def root491Tail5 : List Nat := 5099 :: root491Tail6
def root491Tail4 : List Nat := 4703 :: root491Tail5
def root491Tail3 : List Nat := 4679 :: root491Tail4
def root491Tail2 : List Nat := 4583 :: root491Tail3
def root491Tail1 : List Nat := 4133 :: root491Tail2
def root491Tail0 : List Nat := 3593 :: root491Tail1
def root491Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root491Candidates0_checked : candidates edge 4 3533 3593 root491Tail1 winner.bound = root491Candidates0 := by
  unfold candidates
  have row : edge 3593 = (SortedNatTree.contains FastRows.row3593) := by
    change Data.row3593.contains = (SortedNatTree.contains FastRows.row3593)
    exact FastRows.row3593_checked.symm
  rw [row]
  decide
theorem root491Step0 : scan edge (search edge 3) 4 [3533] 3533 root491Tail0 winner =
    scan edge (search edge 3) 4 [3533] 3533 root491Tail1 winner := by
  rw [root491Tail0, scan, if_neg (by decide), root491Candidates0_checked]
  change scan edge (search edge 3) 4 [3533] 3533 root491Tail1
    (if 3 ≤ root491Candidates0.length then search edge 3 [3533, 3593] root491Candidates0 7126 winner else winner) = _
  rw [if_neg (by decide)]
def root491Candidates1 : List Nat := [9173, 10169]
set_option maxRecDepth 8192 in
theorem root491Candidates1_checked : candidates edge 4 3533 4133 root491Tail2 winner.bound = root491Candidates1 := by
  unfold candidates
  have row : edge 4133 = (SortedNatTree.contains FastRows.row4133) := by
    change Data.row4133.contains = (SortedNatTree.contains FastRows.row4133)
    exact FastRows.row4133_checked.symm
  rw [row]
  decide
theorem root491Step1 : scan edge (search edge 3) 4 [3533] 3533 root491Tail1 winner =
    scan edge (search edge 3) 4 [3533] 3533 root491Tail2 winner := by
  rw [root491Tail1, scan, if_neg (by decide), root491Candidates1_checked]
  change scan edge (search edge 3) 4 [3533] 3533 root491Tail2
    (if 3 ≤ root491Candidates1.length then search edge 3 [3533, 4133] root491Candidates1 7666 winner else winner) = _
  rw [if_neg (by decide)]
def root491Candidates2 : List Nat := [10691]
set_option maxRecDepth 8192 in
theorem root491Candidates2_checked : candidates edge 4 3533 4583 root491Tail3 winner.bound = root491Candidates2 := by
  unfold candidates
  have row : edge 4583 = (SortedNatTree.contains FastRows.row4583) := by
    change Data.row4583.contains = (SortedNatTree.contains FastRows.row4583)
    exact FastRows.row4583_checked.symm
  rw [row]
  decide
theorem root491Step2 : scan edge (search edge 3) 4 [3533] 3533 root491Tail2 winner =
    scan edge (search edge 3) 4 [3533] 3533 root491Tail3 winner := by
  rw [root491Tail2, scan, if_neg (by decide), root491Candidates2_checked]
  change scan edge (search edge 3) 4 [3533] 3533 root491Tail3
    (if 3 ≤ root491Candidates2.length then search edge 3 [3533, 4583] root491Candidates2 8116 winner else winner) = _
  rw [if_neg (by decide)]
def root491Candidates3 : List Nat := [10691]
set_option maxRecDepth 8192 in
theorem root491Candidates3_checked : candidates edge 4 3533 4679 root491Tail4 winner.bound = root491Candidates3 := by
  unfold candidates
  have row : edge 4679 = (SortedNatTree.contains FastRows.row4679) := by
    change Data.row4679.contains = (SortedNatTree.contains FastRows.row4679)
    exact FastRows.row4679_checked.symm
  rw [row]
  decide
theorem root491Step3 : scan edge (search edge 3) 4 [3533] 3533 root491Tail3 winner =
    scan edge (search edge 3) 4 [3533] 3533 root491Tail4 winner := by
  rw [root491Tail3, scan, if_neg (by decide), root491Candidates3_checked]
  change scan edge (search edge 3) 4 [3533] 3533 root491Tail4
    (if 3 ≤ root491Candidates3.length then search edge 3 [3533, 4679] root491Candidates3 8212 winner else winner) = _
  rw [if_neg (by decide)]
def root491Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root491Candidates4_checked : candidates edge 4 3533 4703 root491Tail5 winner.bound = root491Candidates4 := by
  unfold candidates
  have row : edge 4703 = (SortedNatTree.contains FastRows.row4703) := by
    change Data.row4703.contains = (SortedNatTree.contains FastRows.row4703)
    exact FastRows.row4703_checked.symm
  rw [row]
  decide
theorem root491Step4 : scan edge (search edge 3) 4 [3533] 3533 root491Tail4 winner =
    scan edge (search edge 3) 4 [3533] 3533 root491Tail5 winner := by
  rw [root491Tail4, scan, if_neg (by decide), root491Candidates4_checked]
  change scan edge (search edge 3) 4 [3533] 3533 root491Tail5
    (if 3 ≤ root491Candidates4.length then search edge 3 [3533, 4703] root491Candidates4 8236 winner else winner) = _
  rw [if_neg (by decide)]
def root491Candidates5 : List Nat := [8273, 9323]
set_option maxRecDepth 8192 in
theorem root491Candidates5_checked : candidates edge 4 3533 5099 root491Tail6 winner.bound = root491Candidates5 := by
  unfold candidates
  have row : edge 5099 = (SortedNatTree.contains FastRows.row5099) := by
    change Data.row5099.contains = (SortedNatTree.contains FastRows.row5099)
    exact FastRows.row5099_checked.symm
  rw [row]
  decide
theorem root491Step5 : scan edge (search edge 3) 4 [3533] 3533 root491Tail5 winner =
    scan edge (search edge 3) 4 [3533] 3533 root491Tail6 winner := by
  rw [root491Tail5, scan, if_neg (by decide), root491Candidates5_checked]
  change scan edge (search edge 3) 4 [3533] 3533 root491Tail6
    (if 3 ≤ root491Candidates5.length then search edge 3 [3533, 5099] root491Candidates5 8632 winner else winner) = _
  rw [if_neg (by decide)]
def root491Candidates6 : List Nat := [15227]
set_option maxRecDepth 8192 in
theorem root491Candidates6_checked : candidates edge 4 3533 5483 root491Tail7 winner.bound = root491Candidates6 := by
  unfold candidates
  have row : edge 5483 = (SortedNatTree.contains FastRows.row5483) := by
    change Data.row5483.contains = (SortedNatTree.contains FastRows.row5483)
    exact FastRows.row5483_checked.symm
  rw [row]
  decide
theorem root491Step6 : scan edge (search edge 3) 4 [3533] 3533 root491Tail6 winner =
    scan edge (search edge 3) 4 [3533] 3533 root491Tail7 winner := by
  rw [root491Tail6, scan, if_neg (by decide), root491Candidates6_checked]
  change scan edge (search edge 3) 4 [3533] 3533 root491Tail7
    (if 3 ≤ root491Candidates6.length then search edge 3 [3533, 5483] root491Candidates6 9016 winner else winner) = _
  rw [if_neg (by decide)]
theorem root491Scan7 : scan edge (search edge 3) 4 [3533] 3533 root491Tail7 winner = winner := by rw [root491Tail7, scan, if_pos (by decide)]
theorem root491Scan6 : scan edge (search edge 3) 4 [3533] 3533 root491Tail6 winner = winner :=
  (root491Step6).trans root491Scan7
theorem root491Scan5 : scan edge (search edge 3) 4 [3533] 3533 root491Tail5 winner = winner :=
  (root491Step5).trans root491Scan6
theorem root491Scan4 : scan edge (search edge 3) 4 [3533] 3533 root491Tail4 winner = winner :=
  (root491Step4).trans root491Scan5
theorem root491Scan3 : scan edge (search edge 3) 4 [3533] 3533 root491Tail3 winner = winner :=
  (root491Step3).trans root491Scan4
theorem root491Scan2 : scan edge (search edge 3) 4 [3533] 3533 root491Tail2 winner = winner :=
  (root491Step2).trans root491Scan3
theorem root491Scan1 : scan edge (search edge 3) 4 [3533] 3533 root491Tail1 winner = winner :=
  (root491Step1).trans root491Scan2
theorem root491Scan0 : scan edge (search edge 3) 4 [3533] 3533 root491Tail0 winner = winner :=
  (root491Step0).trans root491Scan1
set_option maxRecDepth 4096 in
theorem root491Branch : search edge 4 [3533] root491Candidates 3533 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3533] 3533 root491Tail0 winner = winner
  exact root491Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
