import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root540Candidates : List Nat := Data.row3919
set_option maxRecDepth 8192 in
theorem root540Candidates_checked : candidates edge 5 0 3919 tail541 winner.bound = root540Candidates := by
  apply candidates_eq_row edge 5 0 3919 tail541 root540Candidates winner.bound tail541_ordered
  · decide
  · decide
  · funext q
    rfl
def root540Tail36 : List Nat := []
def root540Tail35 : List Nat := 22039 :: root540Tail36
def root540Tail34 : List Nat := 20773 :: root540Tail35
def root540Tail33 : List Nat := 20719 :: root540Tail34
def root540Tail32 : List Nat := 20173 :: root540Tail33
def root540Tail31 : List Nat := 20143 :: root540Tail32
def root540Tail30 : List Nat := 20101 :: root540Tail31
def root540Tail29 : List Nat := 19993 :: root540Tail30
def root540Tail28 : List Nat := 17749 :: root540Tail29
def root540Tail27 : List Nat := 17599 :: root540Tail28
def root540Tail26 : List Nat := 16141 :: root540Tail27
def root540Tail25 : List Nat := 14251 :: root540Tail26
def root540Tail24 : List Nat := 13831 :: root540Tail25
def root540Tail23 : List Nat := 13723 :: root540Tail24
def root540Tail22 : List Nat := 13339 :: root540Tail23
def root540Tail21 : List Nat := 11677 :: root540Tail22
def root540Tail20 : List Nat := 9547 :: root540Tail21
def root540Tail19 : List Nat := 9133 :: root540Tail20
def root540Tail18 : List Nat := 9103 :: root540Tail19
def root540Tail17 : List Nat := 8317 :: root540Tail18
def root540Tail16 : List Nat := 8011 :: root540Tail17
def root540Tail15 : List Nat := 7789 :: root540Tail16
def root540Tail14 : List Nat := 7489 :: root540Tail15
def root540Tail13 : List Nat := 7309 :: root540Tail14
def root540Tail12 : List Nat := 6961 :: root540Tail13
def root540Tail11 : List Nat := 6529 :: root540Tail12
def root540Tail10 : List Nat := 6397 :: root540Tail11
def root540Tail9 : List Nat := 6151 :: root540Tail10
def root540Tail8 : List Nat := 6007 :: root540Tail9
def root540Tail7 : List Nat := 5743 :: root540Tail8
def root540Tail6 : List Nat := 5503 :: root540Tail7
def root540Tail5 : List Nat := 5281 :: root540Tail6
def root540Tail4 : List Nat := 5179 :: root540Tail5
def root540Tail3 : List Nat := 4957 :: root540Tail4
def root540Tail2 : List Nat := 4591 :: root540Tail3
def root540Tail1 : List Nat := 4567 :: root540Tail2
def root540Tail0 : List Nat := 4297 :: root540Tail1
def root540Candidates0 : List Nat := [8317, 9133]
set_option maxRecDepth 8192 in
theorem root540Candidates0_checked : candidates edge 4 3919 4297 root540Tail1 winner.bound = root540Candidates0 := by
  unfold candidates
  have row : edge 4297 = (SortedNatTree.contains FastRows.row4297) := by
    change Data.row4297.contains = (SortedNatTree.contains FastRows.row4297)
    exact FastRows.row4297_checked.symm
  rw [row]
  decide
theorem root540Step0 : scan edge (search edge 3) 4 [3919] 3919 root540Tail0 winner =
    scan edge (search edge 3) 4 [3919] 3919 root540Tail1 winner := by
  rw [root540Tail0, scan, if_neg (by decide), root540Candidates0_checked]
  change scan edge (search edge 3) 4 [3919] 3919 root540Tail1
    (if 3 ≤ root540Candidates0.length then search edge 3 [3919, 4297] root540Candidates0 8216 winner else winner) = _
  rw [if_neg (by decide)]
def root540Candidates1 : List Nat := [7789]
set_option maxRecDepth 8192 in
theorem root540Candidates1_checked : candidates edge 4 3919 4567 root540Tail2 winner.bound = root540Candidates1 := by
  unfold candidates
  have row : edge 4567 = (SortedNatTree.contains FastRows.row4567) := by
    change Data.row4567.contains = (SortedNatTree.contains FastRows.row4567)
    exact FastRows.row4567_checked.symm
  rw [row]
  decide
theorem root540Step1 : scan edge (search edge 3) 4 [3919] 3919 root540Tail1 winner =
    scan edge (search edge 3) 4 [3919] 3919 root540Tail2 winner := by
  rw [root540Tail1, scan, if_neg (by decide), root540Candidates1_checked]
  change scan edge (search edge 3) 4 [3919] 3919 root540Tail2
    (if 3 ≤ root540Candidates1.length then search edge 3 [3919, 4567] root540Candidates1 8486 winner else winner) = _
  rw [if_neg (by decide)]
def root540Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root540Candidates2_checked : candidates edge 4 3919 4591 root540Tail3 winner.bound = root540Candidates2 := by
  unfold candidates
  have row : edge 4591 = (SortedNatTree.contains FastRows.row4591) := by
    change Data.row4591.contains = (SortedNatTree.contains FastRows.row4591)
    exact FastRows.row4591_checked.symm
  rw [row]
  decide
theorem root540Step2 : scan edge (search edge 3) 4 [3919] 3919 root540Tail2 winner =
    scan edge (search edge 3) 4 [3919] 3919 root540Tail3 winner := by
  rw [root540Tail2, scan, if_neg (by decide), root540Candidates2_checked]
  change scan edge (search edge 3) 4 [3919] 3919 root540Tail3
    (if 3 ≤ root540Candidates2.length then search edge 3 [3919, 4591] root540Candidates2 8510 winner else winner) = _
  rw [if_neg (by decide)]
def root540Candidates3 : List Nat := [13723]
set_option maxRecDepth 8192 in
theorem root540Candidates3_checked : candidates edge 4 3919 4957 root540Tail4 winner.bound = root540Candidates3 := by
  unfold candidates
  have row : edge 4957 = (SortedNatTree.contains FastRows.row4957) := by
    change Data.row4957.contains = (SortedNatTree.contains FastRows.row4957)
    exact FastRows.row4957_checked.symm
  rw [row]
  decide
theorem root540Step3 : scan edge (search edge 3) 4 [3919] 3919 root540Tail3 winner =
    scan edge (search edge 3) 4 [3919] 3919 root540Tail4 winner := by
  rw [root540Tail3, scan, if_neg (by decide), root540Candidates3_checked]
  change scan edge (search edge 3) 4 [3919] 3919 root540Tail4
    (if 3 ≤ root540Candidates3.length then search edge 3 [3919, 4957] root540Candidates3 8876 winner else winner) = _
  rw [if_neg (by decide)]
def root540Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root540Candidates4_checked : candidates edge 4 3919 5179 root540Tail5 winner.bound = root540Candidates4 := by
  unfold candidates
  have row : edge 5179 = (SortedNatTree.contains FastRows.row5179) := by
    change Data.row5179.contains = (SortedNatTree.contains FastRows.row5179)
    exact FastRows.row5179_checked.symm
  rw [row]
  decide
theorem root540Step4 : scan edge (search edge 3) 4 [3919] 3919 root540Tail4 winner =
    scan edge (search edge 3) 4 [3919] 3919 root540Tail5 winner := by
  rw [root540Tail4, scan, if_neg (by decide), root540Candidates4_checked]
  change scan edge (search edge 3) 4 [3919] 3919 root540Tail5
    (if 3 ≤ root540Candidates4.length then search edge 3 [3919, 5179] root540Candidates4 9098 winner else winner) = _
  rw [if_neg (by decide)]
def root540Candidates5 : List Nat := [7489]
set_option maxRecDepth 8192 in
theorem root540Candidates5_checked : candidates edge 4 3919 5281 root540Tail6 winner.bound = root540Candidates5 := by
  unfold candidates
  have row : edge 5281 = (SortedNatTree.contains FastRows.row5281) := by
    change Data.row5281.contains = (SortedNatTree.contains FastRows.row5281)
    exact FastRows.row5281_checked.symm
  rw [row]
  decide
theorem root540Step5 : scan edge (search edge 3) 4 [3919] 3919 root540Tail5 winner =
    scan edge (search edge 3) 4 [3919] 3919 root540Tail6 winner := by
  rw [root540Tail5, scan, if_neg (by decide), root540Candidates5_checked]
  change scan edge (search edge 3) 4 [3919] 3919 root540Tail6
    (if 3 ≤ root540Candidates5.length then search edge 3 [3919, 5281] root540Candidates5 9200 winner else winner) = _
  rw [if_neg (by decide)]
def root540Candidates6 : List Nat := [16141]
set_option maxRecDepth 8192 in
theorem root540Candidates6_checked : candidates edge 4 3919 5503 root540Tail7 winner.bound = root540Candidates6 := by
  unfold candidates
  have row : edge 5503 = (SortedNatTree.contains FastRows.row5503) := by
    change Data.row5503.contains = (SortedNatTree.contains FastRows.row5503)
    exact FastRows.row5503_checked.symm
  rw [row]
  decide
theorem root540Step6 : scan edge (search edge 3) 4 [3919] 3919 root540Tail6 winner =
    scan edge (search edge 3) 4 [3919] 3919 root540Tail7 winner := by
  rw [root540Tail6, scan, if_neg (by decide), root540Candidates6_checked]
  change scan edge (search edge 3) 4 [3919] 3919 root540Tail7
    (if 3 ≤ root540Candidates6.length then search edge 3 [3919, 5503] root540Candidates6 9422 winner else winner) = _
  rw [if_neg (by decide)]
theorem root540Scan7 : scan edge (search edge 3) 4 [3919] 3919 root540Tail7 winner = winner := by rw [root540Tail7, scan, if_pos (by decide)]
theorem root540Scan6 : scan edge (search edge 3) 4 [3919] 3919 root540Tail6 winner = winner :=
  (root540Step6).trans root540Scan7
theorem root540Scan5 : scan edge (search edge 3) 4 [3919] 3919 root540Tail5 winner = winner :=
  (root540Step5).trans root540Scan6
theorem root540Scan4 : scan edge (search edge 3) 4 [3919] 3919 root540Tail4 winner = winner :=
  (root540Step4).trans root540Scan5
theorem root540Scan3 : scan edge (search edge 3) 4 [3919] 3919 root540Tail3 winner = winner :=
  (root540Step3).trans root540Scan4
theorem root540Scan2 : scan edge (search edge 3) 4 [3919] 3919 root540Tail2 winner = winner :=
  (root540Step2).trans root540Scan3
theorem root540Scan1 : scan edge (search edge 3) 4 [3919] 3919 root540Tail1 winner = winner :=
  (root540Step1).trans root540Scan2
theorem root540Scan0 : scan edge (search edge 3) 4 [3919] 3919 root540Tail0 winner = winner :=
  (root540Step0).trans root540Scan1
set_option maxRecDepth 4096 in
theorem root540Branch : search edge 4 [3919] root540Candidates 3919 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3919] 3919 root540Tail0 winner = winner
  exact root540Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
