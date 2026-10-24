import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root347Candidates : List Nat := Data.row2357
set_option maxRecDepth 8192 in
theorem root347Candidates_checked : candidates edge 5 0 2357 tail348 winner.bound = root347Candidates := by
  apply candidates_eq_row edge 5 0 2357 tail348 root347Candidates winner.bound tail348_ordered
  · decide
  · decide
  · funext q
    rfl
def root347Tail39 : List Nat := []
def root347Tail38 : List Nat := 22787 :: root347Tail39
def root347Tail37 : List Nat := 22031 :: root347Tail38
def root347Tail36 : List Nat := 20981 :: root347Tail37
def root347Tail35 : List Nat := 20201 :: root347Tail36
def root347Tail34 : List Nat := 19013 :: root347Tail35
def root347Tail33 : List Nat := 16661 :: root347Tail34
def root347Tail32 : List Nat := 15959 :: root347Tail33
def root347Tail31 : List Nat := 15923 :: root347Tail32
def root347Tail30 : List Nat := 14657 :: root347Tail31
def root347Tail29 : List Nat := 13691 :: root347Tail30
def root347Tail28 : List Nat := 12329 :: root347Tail29
def root347Tail27 : List Nat := 12227 :: root347Tail28
def root347Tail26 : List Nat := 11633 :: root347Tail27
def root347Tail25 : List Nat := 11369 :: root347Tail26
def root347Tail24 : List Nat := 10613 :: root347Tail25
def root347Tail23 : List Nat := 9851 :: root347Tail24
def root347Tail22 : List Nat := 9431 :: root347Tail23
def root347Tail21 : List Nat := 8699 :: root347Tail22
def root347Tail20 : List Nat := 8447 :: root347Tail21
def root347Tail19 : List Nat := 8237 :: root347Tail20
def root347Tail18 : List Nat := 7937 :: root347Tail19
def root347Tail17 : List Nat := 7919 :: root347Tail18
def root347Tail16 : List Nat := 7901 :: root347Tail17
def root347Tail15 : List Nat := 7457 :: root347Tail16
def root347Tail14 : List Nat := 7331 :: root347Tail15
def root347Tail13 : List Nat := 7187 :: root347Tail14
def root347Tail12 : List Nat := 6869 :: root347Tail13
def root347Tail11 : List Nat := 6551 :: root347Tail12
def root347Tail10 : List Nat := 6323 :: root347Tail11
def root347Tail9 : List Nat := 5711 :: root347Tail10
def root347Tail8 : List Nat := 5693 :: root347Tail9
def root347Tail7 : List Nat := 5087 :: root347Tail8
def root347Tail6 : List Nat := 4289 :: root347Tail7
def root347Tail5 : List Nat := 4091 :: root347Tail6
def root347Tail4 : List Nat := 3821 :: root347Tail5
def root347Tail3 : List Nat := 3527 :: root347Tail4
def root347Tail2 : List Nat := 3467 :: root347Tail3
def root347Tail1 : List Nat := 2903 :: root347Tail2
def root347Tail0 : List Nat := 2579 :: root347Tail1
def root347Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root347Candidates0_checked : candidates edge 4 2357 2579 root347Tail1 winner.bound = root347Candidates0 := by
  unfold candidates
  have row : edge 2579 = (SortedNatTree.contains FastRows.row2579) := by
    change Data.row2579.contains = (SortedNatTree.contains FastRows.row2579)
    exact FastRows.row2579_checked.symm
  rw [row]
  decide
theorem root347Step0 : scan edge (search edge 3) 4 [2357] 2357 root347Tail0 winner =
    scan edge (search edge 3) 4 [2357] 2357 root347Tail1 winner := by
  rw [root347Tail0, scan, if_neg (by decide), root347Candidates0_checked]
  change scan edge (search edge 3) 4 [2357] 2357 root347Tail1
    (if 3 ≤ root347Candidates0.length then search edge 3 [2357, 2579] root347Candidates0 4936 winner else winner) = _
  rw [if_neg (by decide)]
def root347Candidates1 : List Nat := [7457, 9851, 20201]
set_option maxRecDepth 8192 in
theorem root347Candidates1_checked : candidates edge 4 2357 2903 root347Tail2 winner.bound = root347Candidates1 := by
  unfold candidates
  have row : edge 2903 = (SortedNatTree.contains FastRows.row2903) := by
    change Data.row2903.contains = (SortedNatTree.contains FastRows.row2903)
    exact FastRows.row2903_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root347Child1 : search edge 3 [2357, 2903] root347Candidates1 5260 winner = winner := by decide
theorem root347Step1 : scan edge (search edge 3) 4 [2357] 2357 root347Tail1 winner =
    scan edge (search edge 3) 4 [2357] 2357 root347Tail2 winner := by
  rw [root347Tail1, scan, if_neg (by decide), root347Candidates1_checked]
  change scan edge (search edge 3) 4 [2357] 2357 root347Tail2
    (if 3 ≤ root347Candidates1.length then search edge 3 [2357, 2903] root347Candidates1 5260 winner else winner) = _
  rw [if_pos (by decide), root347Child1]
def root347Candidates2 : List Nat := [16661]
set_option maxRecDepth 8192 in
theorem root347Candidates2_checked : candidates edge 4 2357 3467 root347Tail3 winner.bound = root347Candidates2 := by
  unfold candidates
  have row : edge 3467 = (SortedNatTree.contains FastRows.row3467) := by
    change Data.row3467.contains = (SortedNatTree.contains FastRows.row3467)
    exact FastRows.row3467_checked.symm
  rw [row]
  decide
theorem root347Step2 : scan edge (search edge 3) 4 [2357] 2357 root347Tail2 winner =
    scan edge (search edge 3) 4 [2357] 2357 root347Tail3 winner := by
  rw [root347Tail2, scan, if_neg (by decide), root347Candidates2_checked]
  change scan edge (search edge 3) 4 [2357] 2357 root347Tail3
    (if 3 ≤ root347Candidates2.length then search edge 3 [2357, 3467] root347Candidates2 5824 winner else winner) = _
  rw [if_neg (by decide)]
def root347Candidates3 : List Nat := [6551, 7901]
set_option maxRecDepth 8192 in
theorem root347Candidates3_checked : candidates edge 4 2357 3527 root347Tail4 winner.bound = root347Candidates3 := by
  unfold candidates
  have row : edge 3527 = (SortedNatTree.contains FastRows.row3527) := by
    change Data.row3527.contains = (SortedNatTree.contains FastRows.row3527)
    exact FastRows.row3527_checked.symm
  rw [row]
  decide
theorem root347Step3 : scan edge (search edge 3) 4 [2357] 2357 root347Tail3 winner =
    scan edge (search edge 3) 4 [2357] 2357 root347Tail4 winner := by
  rw [root347Tail3, scan, if_neg (by decide), root347Candidates3_checked]
  change scan edge (search edge 3) 4 [2357] 2357 root347Tail4
    (if 3 ≤ root347Candidates3.length then search edge 3 [2357, 3527] root347Candidates3 5884 winner else winner) = _
  rw [if_neg (by decide)]
def root347Candidates4 : List Nat := [14657]
set_option maxRecDepth 8192 in
theorem root347Candidates4_checked : candidates edge 4 2357 3821 root347Tail5 winner.bound = root347Candidates4 := by
  unfold candidates
  have row : edge 3821 = (SortedNatTree.contains FastRows.row3821) := by
    change Data.row3821.contains = (SortedNatTree.contains FastRows.row3821)
    exact FastRows.row3821_checked.symm
  rw [row]
  decide
theorem root347Step4 : scan edge (search edge 3) 4 [2357] 2357 root347Tail4 winner =
    scan edge (search edge 3) 4 [2357] 2357 root347Tail5 winner := by
  rw [root347Tail4, scan, if_neg (by decide), root347Candidates4_checked]
  change scan edge (search edge 3) 4 [2357] 2357 root347Tail5
    (if 3 ≤ root347Candidates4.length then search edge 3 [2357, 3821] root347Candidates4 6178 winner else winner) = _
  rw [if_neg (by decide)]
def root347Candidates5 : List Nat := [7937, 8237]
set_option maxRecDepth 8192 in
theorem root347Candidates5_checked : candidates edge 4 2357 4091 root347Tail6 winner.bound = root347Candidates5 := by
  unfold candidates
  have row : edge 4091 = (SortedNatTree.contains FastRows.row4091) := by
    change Data.row4091.contains = (SortedNatTree.contains FastRows.row4091)
    exact FastRows.row4091_checked.symm
  rw [row]
  decide
theorem root347Step5 : scan edge (search edge 3) 4 [2357] 2357 root347Tail5 winner =
    scan edge (search edge 3) 4 [2357] 2357 root347Tail6 winner := by
  rw [root347Tail5, scan, if_neg (by decide), root347Candidates5_checked]
  change scan edge (search edge 3) 4 [2357] 2357 root347Tail6
    (if 3 ≤ root347Candidates5.length then search edge 3 [2357, 4091] root347Candidates5 6448 winner else winner) = _
  rw [if_neg (by decide)]
def root347Candidates6 : List Nat := [6869, 7331, 7937]
set_option maxRecDepth 8192 in
theorem root347Candidates6_checked : candidates edge 4 2357 4289 root347Tail7 winner.bound = root347Candidates6 := by
  unfold candidates
  have row : edge 4289 = (SortedNatTree.contains FastRows.row4289) := by
    change Data.row4289.contains = (SortedNatTree.contains FastRows.row4289)
    exact FastRows.row4289_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root347Child6 : search edge 3 [2357, 4289] root347Candidates6 6646 winner = winner := by decide
theorem root347Step6 : scan edge (search edge 3) 4 [2357] 2357 root347Tail6 winner =
    scan edge (search edge 3) 4 [2357] 2357 root347Tail7 winner := by
  rw [root347Tail6, scan, if_neg (by decide), root347Candidates6_checked]
  change scan edge (search edge 3) 4 [2357] 2357 root347Tail7
    (if 3 ≤ root347Candidates6.length then search edge 3 [2357, 4289] root347Candidates6 6646 winner else winner) = _
  rw [if_pos (by decide), root347Child6]
def root347Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root347Candidates7_checked : candidates edge 4 2357 5087 root347Tail8 winner.bound = root347Candidates7 := by
  unfold candidates
  have row : edge 5087 = (SortedNatTree.contains FastRows.row5087) := by
    change Data.row5087.contains = (SortedNatTree.contains FastRows.row5087)
    exact FastRows.row5087_checked.symm
  rw [row]
  decide
theorem root347Step7 : scan edge (search edge 3) 4 [2357] 2357 root347Tail7 winner =
    scan edge (search edge 3) 4 [2357] 2357 root347Tail8 winner := by
  rw [root347Tail7, scan, if_neg (by decide), root347Candidates7_checked]
  change scan edge (search edge 3) 4 [2357] 2357 root347Tail8
    (if 3 ≤ root347Candidates7.length then search edge 3 [2357, 5087] root347Candidates7 7444 winner else winner) = _
  rw [if_neg (by decide)]
def root347Candidates8 : List Nat := [7187, 8699]
set_option maxRecDepth 8192 in
theorem root347Candidates8_checked : candidates edge 4 2357 5693 root347Tail9 winner.bound = root347Candidates8 := by
  unfold candidates
  have row : edge 5693 = (SortedNatTree.contains FastRows.row5693) := by
    change Data.row5693.contains = (SortedNatTree.contains FastRows.row5693)
    exact FastRows.row5693_checked.symm
  rw [row]
  decide
theorem root347Step8 : scan edge (search edge 3) 4 [2357] 2357 root347Tail8 winner =
    scan edge (search edge 3) 4 [2357] 2357 root347Tail9 winner := by
  rw [root347Tail8, scan, if_neg (by decide), root347Candidates8_checked]
  change scan edge (search edge 3) 4 [2357] 2357 root347Tail9
    (if 3 ≤ root347Candidates8.length then search edge 3 [2357, 5693] root347Candidates8 8050 winner else winner) = _
  rw [if_neg (by decide)]
def root347Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root347Candidates9_checked : candidates edge 4 2357 5711 root347Tail10 winner.bound = root347Candidates9 := by
  unfold candidates
  have row : edge 5711 = (SortedNatTree.contains FastRows.row5711) := by
    change Data.row5711.contains = (SortedNatTree.contains FastRows.row5711)
    exact FastRows.row5711_checked.symm
  rw [row]
  decide
theorem root347Step9 : scan edge (search edge 3) 4 [2357] 2357 root347Tail9 winner =
    scan edge (search edge 3) 4 [2357] 2357 root347Tail10 winner := by
  rw [root347Tail9, scan, if_neg (by decide), root347Candidates9_checked]
  change scan edge (search edge 3) 4 [2357] 2357 root347Tail10
    (if 3 ≤ root347Candidates9.length then search edge 3 [2357, 5711] root347Candidates9 8068 winner else winner) = _
  rw [if_neg (by decide)]
theorem root347Scan10 : scan edge (search edge 3) 4 [2357] 2357 root347Tail10 winner = winner := by rw [root347Tail10, scan, if_pos (by decide)]
theorem root347Scan9 : scan edge (search edge 3) 4 [2357] 2357 root347Tail9 winner = winner :=
  (root347Step9).trans root347Scan10
theorem root347Scan8 : scan edge (search edge 3) 4 [2357] 2357 root347Tail8 winner = winner :=
  (root347Step8).trans root347Scan9
theorem root347Scan7 : scan edge (search edge 3) 4 [2357] 2357 root347Tail7 winner = winner :=
  (root347Step7).trans root347Scan8
theorem root347Scan6 : scan edge (search edge 3) 4 [2357] 2357 root347Tail6 winner = winner :=
  (root347Step6).trans root347Scan7
theorem root347Scan5 : scan edge (search edge 3) 4 [2357] 2357 root347Tail5 winner = winner :=
  (root347Step5).trans root347Scan6
theorem root347Scan4 : scan edge (search edge 3) 4 [2357] 2357 root347Tail4 winner = winner :=
  (root347Step4).trans root347Scan5
theorem root347Scan3 : scan edge (search edge 3) 4 [2357] 2357 root347Tail3 winner = winner :=
  (root347Step3).trans root347Scan4
theorem root347Scan2 : scan edge (search edge 3) 4 [2357] 2357 root347Tail2 winner = winner :=
  (root347Step2).trans root347Scan3
theorem root347Scan1 : scan edge (search edge 3) 4 [2357] 2357 root347Tail1 winner = winner :=
  (root347Step1).trans root347Scan2
theorem root347Scan0 : scan edge (search edge 3) 4 [2357] 2357 root347Tail0 winner = winner :=
  (root347Step0).trans root347Scan1
set_option maxRecDepth 4096 in
theorem root347Branch : search edge 4 [2357] root347Candidates 2357 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2357] 2357 root347Tail0 winner = winner
  exact root347Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
