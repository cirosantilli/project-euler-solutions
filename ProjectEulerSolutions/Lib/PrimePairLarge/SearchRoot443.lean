import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root443Candidates : List Nat := Data.row3137
set_option maxRecDepth 8192 in
theorem root443Candidates_checked : candidates edge 5 0 3137 tail444 winner.bound = root443Candidates := by
  apply candidates_eq_row edge 5 0 3137 tail444 root443Candidates winner.bound tail444_ordered
  · decide
  · decide
  · funext q
    rfl
def root443Tail35 : List Nat := []
def root443Tail34 : List Nat := 22469 :: root443Tail35
def root443Tail33 : List Nat := 22067 :: root443Tail34
def root443Tail32 : List Nat := 20681 :: root443Tail33
def root443Tail31 : List Nat := 20663 :: root443Tail32
def root443Tail30 : List Nat := 18803 :: root443Tail31
def root443Tail29 : List Nat := 18371 :: root443Tail30
def root443Tail28 : List Nat := 17387 :: root443Tail29
def root443Tail27 : List Nat := 17333 :: root443Tail28
def root443Tail26 : List Nat := 14867 :: root443Tail27
def root443Tail25 : List Nat := 14081 :: root443Tail26
def root443Tail24 : List Nat := 13931 :: root443Tail25
def root443Tail23 : List Nat := 13217 :: root443Tail24
def root443Tail22 : List Nat := 12107 :: root443Tail23
def root443Tail21 : List Nat := 10937 :: root443Tail22
def root443Tail20 : List Nat := 10889 :: root443Tail21
def root443Tail19 : List Nat := 10781 :: root443Tail20
def root443Tail18 : List Nat := 10667 :: root443Tail19
def root443Tail17 : List Nat := 10601 :: root443Tail18
def root443Tail16 : List Nat := 10427 :: root443Tail17
def root443Tail15 : List Nat := 10079 :: root443Tail16
def root443Tail14 : List Nat := 9923 :: root443Tail15
def root443Tail13 : List Nat := 9767 :: root443Tail14
def root443Tail12 : List Nat := 8513 :: root443Tail13
def root443Tail11 : List Nat := 8387 :: root443Tail12
def root443Tail10 : List Nat := 7919 :: root443Tail11
def root443Tail9 : List Nat := 7331 :: root443Tail10
def root443Tail8 : List Nat := 6803 :: root443Tail9
def root443Tail7 : List Nat := 6311 :: root443Tail8
def root443Tail6 : List Nat := 5501 :: root443Tail7
def root443Tail5 : List Nat := 5189 :: root443Tail6
def root443Tail4 : List Nat := 5009 :: root443Tail5
def root443Tail3 : List Nat := 4349 :: root443Tail4
def root443Tail2 : List Nat := 3821 :: root443Tail3
def root443Tail1 : List Nat := 3593 :: root443Tail2
def root443Tail0 : List Nat := 3371 :: root443Tail1
def root443Candidates0 : List Nat := [5009]
set_option maxRecDepth 8192 in
theorem root443Candidates0_checked : candidates edge 4 3137 3371 root443Tail1 winner.bound = root443Candidates0 := by
  unfold candidates
  have row : edge 3371 = (SortedNatTree.contains FastRows.row3371) := by
    change Data.row3371.contains = (SortedNatTree.contains FastRows.row3371)
    exact FastRows.row3371_checked.symm
  rw [row]
  decide
theorem root443Step0 : scan edge (search edge 3) 4 [3137] 3137 root443Tail0 winner =
    scan edge (search edge 3) 4 [3137] 3137 root443Tail1 winner := by
  rw [root443Tail0, scan, if_neg (by decide), root443Candidates0_checked]
  change scan edge (search edge 3) 4 [3137] 3137 root443Tail1
    (if 3 ≤ root443Candidates0.length then search edge 3 [3137, 3371] root443Candidates0 6508 winner else winner) = _
  rw [if_neg (by decide)]
def root443Candidates1 : List Nat := [6311]
set_option maxRecDepth 8192 in
theorem root443Candidates1_checked : candidates edge 4 3137 3593 root443Tail2 winner.bound = root443Candidates1 := by
  unfold candidates
  have row : edge 3593 = (SortedNatTree.contains FastRows.row3593) := by
    change Data.row3593.contains = (SortedNatTree.contains FastRows.row3593)
    exact FastRows.row3593_checked.symm
  rw [row]
  decide
theorem root443Step1 : scan edge (search edge 3) 4 [3137] 3137 root443Tail1 winner =
    scan edge (search edge 3) 4 [3137] 3137 root443Tail2 winner := by
  rw [root443Tail1, scan, if_neg (by decide), root443Candidates1_checked]
  change scan edge (search edge 3) 4 [3137] 3137 root443Tail2
    (if 3 ≤ root443Candidates1.length then search edge 3 [3137, 3593] root443Candidates1 6730 winner else winner) = _
  rw [if_neg (by decide)]
def root443Candidates2 : List Nat := [14867, 18371]
set_option maxRecDepth 8192 in
theorem root443Candidates2_checked : candidates edge 4 3137 3821 root443Tail3 winner.bound = root443Candidates2 := by
  unfold candidates
  have row : edge 3821 = (SortedNatTree.contains FastRows.row3821) := by
    change Data.row3821.contains = (SortedNatTree.contains FastRows.row3821)
    exact FastRows.row3821_checked.symm
  rw [row]
  decide
theorem root443Step2 : scan edge (search edge 3) 4 [3137] 3137 root443Tail2 winner =
    scan edge (search edge 3) 4 [3137] 3137 root443Tail3 winner := by
  rw [root443Tail2, scan, if_neg (by decide), root443Candidates2_checked]
  change scan edge (search edge 3) 4 [3137] 3137 root443Tail3
    (if 3 ≤ root443Candidates2.length then search edge 3 [3137, 3821] root443Candidates2 6958 winner else winner) = _
  rw [if_neg (by decide)]
def root443Candidates3 : List Nat := [13217]
set_option maxRecDepth 8192 in
theorem root443Candidates3_checked : candidates edge 4 3137 4349 root443Tail4 winner.bound = root443Candidates3 := by
  unfold candidates
  have row : edge 4349 = (SortedNatTree.contains FastRows.row4349) := by
    change Data.row4349.contains = (SortedNatTree.contains FastRows.row4349)
    exact FastRows.row4349_checked.symm
  rw [row]
  decide
theorem root443Step3 : scan edge (search edge 3) 4 [3137] 3137 root443Tail3 winner =
    scan edge (search edge 3) 4 [3137] 3137 root443Tail4 winner := by
  rw [root443Tail3, scan, if_neg (by decide), root443Candidates3_checked]
  change scan edge (search edge 3) 4 [3137] 3137 root443Tail4
    (if 3 ≤ root443Candidates3.length then search edge 3 [3137, 4349] root443Candidates3 7486 winner else winner) = _
  rw [if_neg (by decide)]
def root443Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root443Candidates4_checked : candidates edge 4 3137 5009 root443Tail5 winner.bound = root443Candidates4 := by
  unfold candidates
  have row : edge 5009 = (SortedNatTree.contains FastRows.row5009) := by
    change Data.row5009.contains = (SortedNatTree.contains FastRows.row5009)
    exact FastRows.row5009_checked.symm
  rw [row]
  decide
theorem root443Step4 : scan edge (search edge 3) 4 [3137] 3137 root443Tail4 winner =
    scan edge (search edge 3) 4 [3137] 3137 root443Tail5 winner := by
  rw [root443Tail4, scan, if_neg (by decide), root443Candidates4_checked]
  change scan edge (search edge 3) 4 [3137] 3137 root443Tail5
    (if 3 ≤ root443Candidates4.length then search edge 3 [3137, 5009] root443Candidates4 8146 winner else winner) = _
  rw [if_neg (by decide)]
def root443Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root443Candidates5_checked : candidates edge 4 3137 5189 root443Tail6 winner.bound = root443Candidates5 := by
  unfold candidates
  have row : edge 5189 = (SortedNatTree.contains FastRows.row5189) := by
    change Data.row5189.contains = (SortedNatTree.contains FastRows.row5189)
    exact FastRows.row5189_checked.symm
  rw [row]
  decide
theorem root443Step5 : scan edge (search edge 3) 4 [3137] 3137 root443Tail5 winner =
    scan edge (search edge 3) 4 [3137] 3137 root443Tail6 winner := by
  rw [root443Tail5, scan, if_neg (by decide), root443Candidates5_checked]
  change scan edge (search edge 3) 4 [3137] 3137 root443Tail6
    (if 3 ≤ root443Candidates5.length then search edge 3 [3137, 5189] root443Candidates5 8326 winner else winner) = _
  rw [if_neg (by decide)]
def root443Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root443Candidates6_checked : candidates edge 4 3137 5501 root443Tail7 winner.bound = root443Candidates6 := by
  unfold candidates
  have row : edge 5501 = (SortedNatTree.contains FastRows.row5501) := by
    change Data.row5501.contains = (SortedNatTree.contains FastRows.row5501)
    exact FastRows.row5501_checked.symm
  rw [row]
  decide
theorem root443Step6 : scan edge (search edge 3) 4 [3137] 3137 root443Tail6 winner =
    scan edge (search edge 3) 4 [3137] 3137 root443Tail7 winner := by
  rw [root443Tail6, scan, if_neg (by decide), root443Candidates6_checked]
  change scan edge (search edge 3) 4 [3137] 3137 root443Tail7
    (if 3 ≤ root443Candidates6.length then search edge 3 [3137, 5501] root443Candidates6 8638 winner else winner) = _
  rw [if_neg (by decide)]
theorem root443Scan7 : scan edge (search edge 3) 4 [3137] 3137 root443Tail7 winner = winner := by rw [root443Tail7, scan, if_pos (by decide)]
theorem root443Scan6 : scan edge (search edge 3) 4 [3137] 3137 root443Tail6 winner = winner :=
  (root443Step6).trans root443Scan7
theorem root443Scan5 : scan edge (search edge 3) 4 [3137] 3137 root443Tail5 winner = winner :=
  (root443Step5).trans root443Scan6
theorem root443Scan4 : scan edge (search edge 3) 4 [3137] 3137 root443Tail4 winner = winner :=
  (root443Step4).trans root443Scan5
theorem root443Scan3 : scan edge (search edge 3) 4 [3137] 3137 root443Tail3 winner = winner :=
  (root443Step3).trans root443Scan4
theorem root443Scan2 : scan edge (search edge 3) 4 [3137] 3137 root443Tail2 winner = winner :=
  (root443Step2).trans root443Scan3
theorem root443Scan1 : scan edge (search edge 3) 4 [3137] 3137 root443Tail1 winner = winner :=
  (root443Step1).trans root443Scan2
theorem root443Scan0 : scan edge (search edge 3) 4 [3137] 3137 root443Tail0 winner = winner :=
  (root443Step0).trans root443Scan1
set_option maxRecDepth 4096 in
theorem root443Branch : search edge 4 [3137] root443Candidates 3137 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3137] 3137 root443Tail0 winner = winner
  exact root443Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
