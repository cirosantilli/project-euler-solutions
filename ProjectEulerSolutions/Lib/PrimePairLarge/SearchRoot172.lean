import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root172Candidates : List Nat := Data.row1039
set_option maxRecDepth 8192 in
theorem root172Candidates_checked : candidates edge 5 0 1039 tail173 winner.bound = root172Candidates := by
  apply candidates_eq_row edge 5 0 1039 tail173 root172Candidates winner.bound tail173_ordered
  · decide
  · decide
  · funext q
    rfl
def root172Tail45 : List Nat := []
def root172Tail44 : List Nat := 24841 :: root172Tail45
def root172Tail43 : List Nat := 24061 :: root172Tail44
def root172Tail42 : List Nat := 23593 :: root172Tail43
def root172Tail41 : List Nat := 23497 :: root172Tail42
def root172Tail40 : List Nat := 22279 :: root172Tail41
def root172Tail39 : List Nat := 22129 :: root172Tail40
def root172Tail38 : List Nat := 22111 :: root172Tail39
def root172Tail37 : List Nat := 21481 :: root172Tail38
def root172Tail36 : List Nat := 20743 :: root172Tail37
def root172Tail35 : List Nat := 20611 :: root172Tail36
def root172Tail34 : List Nat := 19477 :: root172Tail35
def root172Tail33 : List Nat := 18517 :: root172Tail34
def root172Tail32 : List Nat := 18253 :: root172Tail33
def root172Tail31 : List Nat := 18043 :: root172Tail32
def root172Tail30 : List Nat := 17041 :: root172Tail31
def root172Tail29 : List Nat := 16069 :: root172Tail30
def root172Tail28 : List Nat := 14143 :: root172Tail29
def root172Tail27 : List Nat := 13309 :: root172Tail28
def root172Tail26 : List Nat := 13291 :: root172Tail27
def root172Tail25 : List Nat := 12703 :: root172Tail26
def root172Tail24 : List Nat := 12583 :: root172Tail25
def root172Tail23 : List Nat := 12421 :: root172Tail24
def root172Tail22 : List Nat := 12241 :: root172Tail23
def root172Tail21 : List Nat := 11617 :: root172Tail22
def root172Tail20 : List Nat := 9901 :: root172Tail21
def root172Tail19 : List Nat := 8677 :: root172Tail20
def root172Tail18 : List Nat := 8431 :: root172Tail19
def root172Tail17 : List Nat := 8161 :: root172Tail18
def root172Tail16 : List Nat := 7993 :: root172Tail17
def root172Tail15 : List Nat := 6709 :: root172Tail16
def root172Tail14 : List Nat := 6073 :: root172Tail15
def root172Tail13 : List Nat := 5851 :: root172Tail14
def root172Tail12 : List Nat := 5821 :: root172Tail13
def root172Tail11 : List Nat := 5233 :: root172Tail12
def root172Tail10 : List Nat := 4723 :: root172Tail11
def root172Tail9 : List Nat := 4297 :: root172Tail10
def root172Tail8 : List Nat := 3967 :: root172Tail9
def root172Tail7 : List Nat := 3637 :: root172Tail8
def root172Tail6 : List Nat := 3457 :: root172Tail7
def root172Tail5 : List Nat := 2917 :: root172Tail6
def root172Tail4 : List Nat := 2659 :: root172Tail5
def root172Tail3 : List Nat := 2467 :: root172Tail4
def root172Tail2 : List Nat := 1999 :: root172Tail3
def root172Tail1 : List Nat := 1663 :: root172Tail2
def root172Tail0 : List Nat := 1279 :: root172Tail1
def root172Candidates0 : List Nat := [22279]
set_option maxRecDepth 8192 in
theorem root172Candidates0_checked : candidates edge 4 1039 1279 root172Tail1 winner.bound = root172Candidates0 := by
  unfold candidates
  have row : edge 1279 = (SortedNatTree.contains FastRows.row1279) := by
    change Data.row1279.contains = (SortedNatTree.contains FastRows.row1279)
    exact FastRows.row1279_checked.symm
  rw [row]
  decide
theorem root172Step0 : scan edge (search edge 3) 4 [1039] 1039 root172Tail0 winner =
    scan edge (search edge 3) 4 [1039] 1039 root172Tail1 winner := by
  rw [root172Tail0, scan, if_neg (by decide), root172Candidates0_checked]
  change scan edge (search edge 3) 4 [1039] 1039 root172Tail1
    (if 3 ≤ root172Candidates0.length then search edge 3 [1039, 1279] root172Candidates0 2318 winner else winner) = _
  rw [if_neg (by decide)]
def root172Candidates1 : List Nat := [12583, 14143]
set_option maxRecDepth 8192 in
theorem root172Candidates1_checked : candidates edge 4 1039 1663 root172Tail2 winner.bound = root172Candidates1 := by
  unfold candidates
  have row : edge 1663 = (SortedNatTree.contains FastRows.row1663) := by
    change Data.row1663.contains = (SortedNatTree.contains FastRows.row1663)
    exact FastRows.row1663_checked.symm
  rw [row]
  decide
theorem root172Step1 : scan edge (search edge 3) 4 [1039] 1039 root172Tail1 winner =
    scan edge (search edge 3) 4 [1039] 1039 root172Tail2 winner := by
  rw [root172Tail1, scan, if_neg (by decide), root172Candidates1_checked]
  change scan edge (search edge 3) 4 [1039] 1039 root172Tail2
    (if 3 ≤ root172Candidates1.length then search edge 3 [1039, 1663] root172Candidates1 2702 winner else winner) = _
  rw [if_neg (by decide)]
def root172Candidates2 : List Nat := [12421]
set_option maxRecDepth 8192 in
theorem root172Candidates2_checked : candidates edge 4 1039 1999 root172Tail3 winner.bound = root172Candidates2 := by
  unfold candidates
  have row : edge 1999 = (SortedNatTree.contains FastRows.row1999) := by
    change Data.row1999.contains = (SortedNatTree.contains FastRows.row1999)
    exact FastRows.row1999_checked.symm
  rw [row]
  decide
theorem root172Step2 : scan edge (search edge 3) 4 [1039] 1039 root172Tail2 winner =
    scan edge (search edge 3) 4 [1039] 1039 root172Tail3 winner := by
  rw [root172Tail2, scan, if_neg (by decide), root172Candidates2_checked]
  change scan edge (search edge 3) 4 [1039] 1039 root172Tail3
    (if 3 ≤ root172Candidates2.length then search edge 3 [1039, 1999] root172Candidates2 3038 winner else winner) = _
  rw [if_neg (by decide)]
def root172Candidates3 : List Nat := [4723]
set_option maxRecDepth 8192 in
theorem root172Candidates3_checked : candidates edge 4 1039 2467 root172Tail4 winner.bound = root172Candidates3 := by
  unfold candidates
  have row : edge 2467 = (SortedNatTree.contains FastRows.row2467) := by
    change Data.row2467.contains = (SortedNatTree.contains FastRows.row2467)
    exact FastRows.row2467_checked.symm
  rw [row]
  decide
theorem root172Step3 : scan edge (search edge 3) 4 [1039] 1039 root172Tail3 winner =
    scan edge (search edge 3) 4 [1039] 1039 root172Tail4 winner := by
  rw [root172Tail3, scan, if_neg (by decide), root172Candidates3_checked]
  change scan edge (search edge 3) 4 [1039] 1039 root172Tail4
    (if 3 ≤ root172Candidates3.length then search edge 3 [1039, 2467] root172Candidates3 3506 winner else winner) = _
  rw [if_neg (by decide)]
def root172Candidates4 : List Nat := [4723]
set_option maxRecDepth 8192 in
theorem root172Candidates4_checked : candidates edge 4 1039 2659 root172Tail5 winner.bound = root172Candidates4 := by
  unfold candidates
  have row : edge 2659 = (SortedNatTree.contains FastRows.row2659) := by
    change Data.row2659.contains = (SortedNatTree.contains FastRows.row2659)
    exact FastRows.row2659_checked.symm
  rw [row]
  decide
theorem root172Step4 : scan edge (search edge 3) 4 [1039] 1039 root172Tail4 winner =
    scan edge (search edge 3) 4 [1039] 1039 root172Tail5 winner := by
  rw [root172Tail4, scan, if_neg (by decide), root172Candidates4_checked]
  change scan edge (search edge 3) 4 [1039] 1039 root172Tail5
    (if 3 ≤ root172Candidates4.length then search edge 3 [1039, 2659] root172Candidates4 3698 winner else winner) = _
  rw [if_neg (by decide)]
def root172Candidates5 : List Nat := [5851]
set_option maxRecDepth 8192 in
theorem root172Candidates5_checked : candidates edge 4 1039 2917 root172Tail6 winner.bound = root172Candidates5 := by
  unfold candidates
  have row : edge 2917 = (SortedNatTree.contains FastRows.row2917) := by
    change Data.row2917.contains = (SortedNatTree.contains FastRows.row2917)
    exact FastRows.row2917_checked.symm
  rw [row]
  decide
theorem root172Step5 : scan edge (search edge 3) 4 [1039] 1039 root172Tail5 winner =
    scan edge (search edge 3) 4 [1039] 1039 root172Tail6 winner := by
  rw [root172Tail5, scan, if_neg (by decide), root172Candidates5_checked]
  change scan edge (search edge 3) 4 [1039] 1039 root172Tail6
    (if 3 ≤ root172Candidates5.length then search edge 3 [1039, 2917] root172Candidates5 3956 winner else winner) = _
  rw [if_neg (by decide)]
def root172Candidates6 : List Nat := [4297, 11617]
set_option maxRecDepth 8192 in
theorem root172Candidates6_checked : candidates edge 4 1039 3457 root172Tail7 winner.bound = root172Candidates6 := by
  unfold candidates
  have row : edge 3457 = (SortedNatTree.contains FastRows.row3457) := by
    change Data.row3457.contains = (SortedNatTree.contains FastRows.row3457)
    exact FastRows.row3457_checked.symm
  rw [row]
  decide
theorem root172Step6 : scan edge (search edge 3) 4 [1039] 1039 root172Tail6 winner =
    scan edge (search edge 3) 4 [1039] 1039 root172Tail7 winner := by
  rw [root172Tail6, scan, if_neg (by decide), root172Candidates6_checked]
  change scan edge (search edge 3) 4 [1039] 1039 root172Tail7
    (if 3 ≤ root172Candidates6.length then search edge 3 [1039, 3457] root172Candidates6 4496 winner else winner) = _
  rw [if_neg (by decide)]
def root172Candidates7 : List Nat := [8677, 13309]
set_option maxRecDepth 8192 in
theorem root172Candidates7_checked : candidates edge 4 1039 3637 root172Tail8 winner.bound = root172Candidates7 := by
  unfold candidates
  have row : edge 3637 = (SortedNatTree.contains FastRows.row3637) := by
    change Data.row3637.contains = (SortedNatTree.contains FastRows.row3637)
    exact FastRows.row3637_checked.symm
  rw [row]
  decide
theorem root172Step7 : scan edge (search edge 3) 4 [1039] 1039 root172Tail7 winner =
    scan edge (search edge 3) 4 [1039] 1039 root172Tail8 winner := by
  rw [root172Tail7, scan, if_neg (by decide), root172Candidates7_checked]
  change scan edge (search edge 3) 4 [1039] 1039 root172Tail8
    (if 3 ≤ root172Candidates7.length then search edge 3 [1039, 3637] root172Candidates7 4676 winner else winner) = _
  rw [if_neg (by decide)]
def root172Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root172Candidates8_checked : candidates edge 4 1039 3967 root172Tail9 winner.bound = root172Candidates8 := by
  unfold candidates
  have row : edge 3967 = (SortedNatTree.contains FastRows.row3967) := by
    change Data.row3967.contains = (SortedNatTree.contains FastRows.row3967)
    exact FastRows.row3967_checked.symm
  rw [row]
  decide
theorem root172Step8 : scan edge (search edge 3) 4 [1039] 1039 root172Tail8 winner =
    scan edge (search edge 3) 4 [1039] 1039 root172Tail9 winner := by
  rw [root172Tail8, scan, if_neg (by decide), root172Candidates8_checked]
  change scan edge (search edge 3) 4 [1039] 1039 root172Tail9
    (if 3 ≤ root172Candidates8.length then search edge 3 [1039, 3967] root172Candidates8 5006 winner else winner) = _
  rw [if_neg (by decide)]
def root172Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root172Candidates9_checked : candidates edge 4 1039 4297 root172Tail10 winner.bound = root172Candidates9 := by
  unfold candidates
  have row : edge 4297 = (SortedNatTree.contains FastRows.row4297) := by
    change Data.row4297.contains = (SortedNatTree.contains FastRows.row4297)
    exact FastRows.row4297_checked.symm
  rw [row]
  decide
theorem root172Step9 : scan edge (search edge 3) 4 [1039] 1039 root172Tail9 winner =
    scan edge (search edge 3) 4 [1039] 1039 root172Tail10 winner := by
  rw [root172Tail9, scan, if_neg (by decide), root172Candidates9_checked]
  change scan edge (search edge 3) 4 [1039] 1039 root172Tail10
    (if 3 ≤ root172Candidates9.length then search edge 3 [1039, 4297] root172Candidates9 5336 winner else winner) = _
  rw [if_neg (by decide)]
def root172Candidates10 : List Nat := [9901]
set_option maxRecDepth 8192 in
theorem root172Candidates10_checked : candidates edge 4 1039 4723 root172Tail11 winner.bound = root172Candidates10 := by
  unfold candidates
  have row : edge 4723 = (SortedNatTree.contains FastRows.row4723) := by
    change Data.row4723.contains = (SortedNatTree.contains FastRows.row4723)
    exact FastRows.row4723_checked.symm
  rw [row]
  decide
theorem root172Step10 : scan edge (search edge 3) 4 [1039] 1039 root172Tail10 winner =
    scan edge (search edge 3) 4 [1039] 1039 root172Tail11 winner := by
  rw [root172Tail10, scan, if_neg (by decide), root172Candidates10_checked]
  change scan edge (search edge 3) 4 [1039] 1039 root172Tail11
    (if 3 ≤ root172Candidates10.length then search edge 3 [1039, 4723] root172Candidates10 5762 winner else winner) = _
  rw [if_neg (by decide)]
def root172Candidates11 : List Nat := [8677]
set_option maxRecDepth 8192 in
theorem root172Candidates11_checked : candidates edge 4 1039 5233 root172Tail12 winner.bound = root172Candidates11 := by
  unfold candidates
  have row : edge 5233 = (SortedNatTree.contains FastRows.row5233) := by
    change Data.row5233.contains = (SortedNatTree.contains FastRows.row5233)
    exact FastRows.row5233_checked.symm
  rw [row]
  decide
theorem root172Step11 : scan edge (search edge 3) 4 [1039] 1039 root172Tail11 winner =
    scan edge (search edge 3) 4 [1039] 1039 root172Tail12 winner := by
  rw [root172Tail11, scan, if_neg (by decide), root172Candidates11_checked]
  change scan edge (search edge 3) 4 [1039] 1039 root172Tail12
    (if 3 ≤ root172Candidates11.length then search edge 3 [1039, 5233] root172Candidates11 6272 winner else winner) = _
  rw [if_neg (by decide)]
def root172Candidates12 : List Nat := [13291]
set_option maxRecDepth 8192 in
theorem root172Candidates12_checked : candidates edge 4 1039 5821 root172Tail13 winner.bound = root172Candidates12 := by
  unfold candidates
  have row : edge 5821 = (SortedNatTree.contains FastRows.row5821) := by
    change Data.row5821.contains = (SortedNatTree.contains FastRows.row5821)
    exact FastRows.row5821_checked.symm
  rw [row]
  decide
theorem root172Step12 : scan edge (search edge 3) 4 [1039] 1039 root172Tail12 winner =
    scan edge (search edge 3) 4 [1039] 1039 root172Tail13 winner := by
  rw [root172Tail12, scan, if_neg (by decide), root172Candidates12_checked]
  change scan edge (search edge 3) 4 [1039] 1039 root172Tail13
    (if 3 ≤ root172Candidates12.length then search edge 3 [1039, 5821] root172Candidates12 6860 winner else winner) = _
  rw [if_neg (by decide)]
def root172Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root172Candidates13_checked : candidates edge 4 1039 5851 root172Tail14 winner.bound = root172Candidates13 := by
  unfold candidates
  have row : edge 5851 = (SortedNatTree.contains FastRows.row5851) := by
    change Data.row5851.contains = (SortedNatTree.contains FastRows.row5851)
    exact FastRows.row5851_checked.symm
  rw [row]
  decide
theorem root172Step13 : scan edge (search edge 3) 4 [1039] 1039 root172Tail13 winner =
    scan edge (search edge 3) 4 [1039] 1039 root172Tail14 winner := by
  rw [root172Tail13, scan, if_neg (by decide), root172Candidates13_checked]
  change scan edge (search edge 3) 4 [1039] 1039 root172Tail14
    (if 3 ≤ root172Candidates13.length then search edge 3 [1039, 5851] root172Candidates13 6890 winner else winner) = _
  rw [if_neg (by decide)]
def root172Candidates14 : List Nat := [8677]
set_option maxRecDepth 8192 in
theorem root172Candidates14_checked : candidates edge 4 1039 6073 root172Tail15 winner.bound = root172Candidates14 := by
  unfold candidates
  have row : edge 6073 = (SortedNatTree.contains FastRows.row6073) := by
    change Data.row6073.contains = (SortedNatTree.contains FastRows.row6073)
    exact FastRows.row6073_checked.symm
  rw [row]
  decide
theorem root172Step14 : scan edge (search edge 3) 4 [1039] 1039 root172Tail14 winner =
    scan edge (search edge 3) 4 [1039] 1039 root172Tail15 winner := by
  rw [root172Tail14, scan, if_neg (by decide), root172Candidates14_checked]
  change scan edge (search edge 3) 4 [1039] 1039 root172Tail15
    (if 3 ≤ root172Candidates14.length then search edge 3 [1039, 6073] root172Candidates14 7112 winner else winner) = _
  rw [if_neg (by decide)]
theorem root172Scan15 : scan edge (search edge 3) 4 [1039] 1039 root172Tail15 winner = winner := by rw [root172Tail15, scan, if_pos (by decide)]
theorem root172Scan14 : scan edge (search edge 3) 4 [1039] 1039 root172Tail14 winner = winner :=
  (root172Step14).trans root172Scan15
theorem root172Scan13 : scan edge (search edge 3) 4 [1039] 1039 root172Tail13 winner = winner :=
  (root172Step13).trans root172Scan14
theorem root172Scan12 : scan edge (search edge 3) 4 [1039] 1039 root172Tail12 winner = winner :=
  (root172Step12).trans root172Scan13
theorem root172Scan11 : scan edge (search edge 3) 4 [1039] 1039 root172Tail11 winner = winner :=
  (root172Step11).trans root172Scan12
theorem root172Scan10 : scan edge (search edge 3) 4 [1039] 1039 root172Tail10 winner = winner :=
  (root172Step10).trans root172Scan11
theorem root172Scan9 : scan edge (search edge 3) 4 [1039] 1039 root172Tail9 winner = winner :=
  (root172Step9).trans root172Scan10
theorem root172Scan8 : scan edge (search edge 3) 4 [1039] 1039 root172Tail8 winner = winner :=
  (root172Step8).trans root172Scan9
theorem root172Scan7 : scan edge (search edge 3) 4 [1039] 1039 root172Tail7 winner = winner :=
  (root172Step7).trans root172Scan8
theorem root172Scan6 : scan edge (search edge 3) 4 [1039] 1039 root172Tail6 winner = winner :=
  (root172Step6).trans root172Scan7
theorem root172Scan5 : scan edge (search edge 3) 4 [1039] 1039 root172Tail5 winner = winner :=
  (root172Step5).trans root172Scan6
theorem root172Scan4 : scan edge (search edge 3) 4 [1039] 1039 root172Tail4 winner = winner :=
  (root172Step4).trans root172Scan5
theorem root172Scan3 : scan edge (search edge 3) 4 [1039] 1039 root172Tail3 winner = winner :=
  (root172Step3).trans root172Scan4
theorem root172Scan2 : scan edge (search edge 3) 4 [1039] 1039 root172Tail2 winner = winner :=
  (root172Step2).trans root172Scan3
theorem root172Scan1 : scan edge (search edge 3) 4 [1039] 1039 root172Tail1 winner = winner :=
  (root172Step1).trans root172Scan2
theorem root172Scan0 : scan edge (search edge 3) 4 [1039] 1039 root172Tail0 winner = winner :=
  (root172Step0).trans root172Scan1
set_option maxRecDepth 4096 in
theorem root172Branch : search edge 4 [1039] root172Candidates 1039 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1039] 1039 root172Tail0 winner = winner
  exact root172Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
