import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root257Candidates : List Nat := Data.row1657
set_option maxRecDepth 8192 in
theorem root257Candidates_checked : candidates edge 5 0 1657 tail258 winner.bound = root257Candidates := by
  apply candidates_eq_row edge 5 0 1657 tail258 root257Candidates winner.bound tail258_ordered
  · decide
  · decide
  · funext q
    rfl
def root257Tail51 : List Nat := []
def root257Tail50 : List Nat := 23977 :: root257Tail51
def root257Tail49 : List Nat := 23563 :: root257Tail50
def root257Tail48 : List Nat := 23473 :: root257Tail49
def root257Tail47 : List Nat := 22699 :: root257Tail48
def root257Tail46 : List Nat := 22549 :: root257Tail47
def root257Tail45 : List Nat := 21433 :: root257Tail46
def root257Tail44 : List Nat := 20809 :: root257Tail45
def root257Tail43 : List Nat := 20173 :: root257Tail44
def root257Tail42 : List Nat := 18793 :: root257Tail43
def root257Tail41 : List Nat := 18481 :: root257Tail42
def root257Tail40 : List Nat := 18199 :: root257Tail41
def root257Tail39 : List Nat := 17239 :: root257Tail40
def root257Tail38 : List Nat := 17029 :: root257Tail39
def root257Tail37 : List Nat := 16693 :: root257Tail38
def root257Tail36 : List Nat := 15787 :: root257Tail37
def root257Tail35 : List Nat := 14947 :: root257Tail36
def root257Tail34 : List Nat := 14449 :: root257Tail35
def root257Tail33 : List Nat := 13903 :: root257Tail34
def root257Tail32 : List Nat := 12739 :: root257Tail33
def root257Tail31 : List Nat := 12301 :: root257Tail32
def root257Tail30 : List Nat := 11047 :: root257Tail31
def root257Tail29 : List Nat := 10789 :: root257Tail30
def root257Tail28 : List Nat := 10639 :: root257Tail29
def root257Tail27 : List Nat := 10627 :: root257Tail28
def root257Tail26 : List Nat := 10513 :: root257Tail27
def root257Tail25 : List Nat := 9931 :: root257Tail26
def root257Tail24 : List Nat := 9817 :: root257Tail25
def root257Tail23 : List Nat := 8647 :: root257Tail24
def root257Tail22 : List Nat := 8623 :: root257Tail23
def root257Tail21 : List Nat := 8269 :: root257Tail22
def root257Tail20 : List Nat := 8167 :: root257Tail21
def root257Tail19 : List Nat := 8017 :: root257Tail20
def root257Tail18 : List Nat := 7237 :: root257Tail19
def root257Tail17 : List Nat := 6907 :: root257Tail18
def root257Tail16 : List Nat := 6733 :: root257Tail17
def root257Tail15 : List Nat := 6397 :: root257Tail16
def root257Tail14 : List Nat := 5101 :: root257Tail15
def root257Tail13 : List Nat := 5059 :: root257Tail14
def root257Tail12 : List Nat := 4969 :: root257Tail13
def root257Tail11 : List Nat := 4549 :: root257Tail12
def root257Tail10 : List Nat := 4273 :: root257Tail11
def root257Tail9 : List Nat := 4111 :: root257Tail10
def root257Tail8 : List Nat := 3967 :: root257Tail9
def root257Tail7 : List Nat := 3463 :: root257Tail8
def root257Tail6 : List Nat := 2917 :: root257Tail7
def root257Tail5 : List Nat := 2887 :: root257Tail6
def root257Tail4 : List Nat := 2791 :: root257Tail5
def root257Tail3 : List Nat := 2683 :: root257Tail4
def root257Tail2 : List Nat := 2203 :: root257Tail3
def root257Tail1 : List Nat := 1879 :: root257Tail2
def root257Tail0 : List Nat := 1699 :: root257Tail1
def root257Candidates0 : List Nat := [2203, 10789, 18481]
set_option maxRecDepth 8192 in
theorem root257Candidates0_checked : candidates edge 4 1657 1699 root257Tail1 winner.bound = root257Candidates0 := by
  unfold candidates
  have row : edge 1699 = (SortedNatTree.contains FastRows.row1699) := by
    change Data.row1699.contains = (SortedNatTree.contains FastRows.row1699)
    exact FastRows.row1699_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root257Child0 : search edge 3 [1657, 1699] root257Candidates0 3356 winner = winner := by decide
theorem root257Step0 : scan edge (search edge 3) 4 [1657] 1657 root257Tail0 winner =
    scan edge (search edge 3) 4 [1657] 1657 root257Tail1 winner := by
  rw [root257Tail0, scan, if_neg (by decide), root257Candidates0_checked]
  change scan edge (search edge 3) 4 [1657] 1657 root257Tail1
    (if 3 ≤ root257Candidates0.length then search edge 3 [1657, 1699] root257Candidates0 3356 winner else winner) = _
  rw [if_pos (by decide), root257Child0]
def root257Candidates1 : List Nat := [12301]
set_option maxRecDepth 8192 in
theorem root257Candidates1_checked : candidates edge 4 1657 1879 root257Tail2 winner.bound = root257Candidates1 := by
  unfold candidates
  have row : edge 1879 = (SortedNatTree.contains FastRows.row1879) := by
    change Data.row1879.contains = (SortedNatTree.contains FastRows.row1879)
    exact FastRows.row1879_checked.symm
  rw [row]
  decide
theorem root257Step1 : scan edge (search edge 3) 4 [1657] 1657 root257Tail1 winner =
    scan edge (search edge 3) 4 [1657] 1657 root257Tail2 winner := by
  rw [root257Tail1, scan, if_neg (by decide), root257Candidates1_checked]
  change scan edge (search edge 3) 4 [1657] 1657 root257Tail2
    (if 3 ≤ root257Candidates1.length then search edge 3 [1657, 1879] root257Candidates1 3536 winner else winner) = _
  rw [if_neg (by decide)]
def root257Candidates2 : List Nat := [2887, 14449]
set_option maxRecDepth 8192 in
theorem root257Candidates2_checked : candidates edge 4 1657 2203 root257Tail3 winner.bound = root257Candidates2 := by
  unfold candidates
  have row : edge 2203 = (SortedNatTree.contains FastRows.row2203) := by
    change Data.row2203.contains = (SortedNatTree.contains FastRows.row2203)
    exact FastRows.row2203_checked.symm
  rw [row]
  decide
theorem root257Step2 : scan edge (search edge 3) 4 [1657] 1657 root257Tail2 winner =
    scan edge (search edge 3) 4 [1657] 1657 root257Tail3 winner := by
  rw [root257Tail2, scan, if_neg (by decide), root257Candidates2_checked]
  change scan edge (search edge 3) 4 [1657] 1657 root257Tail3
    (if 3 ≤ root257Candidates2.length then search edge 3 [1657, 2203] root257Candidates2 3860 winner else winner) = _
  rw [if_neg (by decide)]
def root257Candidates3 : List Nat := [2791, 8167, 11047]
set_option maxRecDepth 8192 in
theorem root257Candidates3_checked : candidates edge 4 1657 2683 root257Tail4 winner.bound = root257Candidates3 := by
  unfold candidates
  have row : edge 2683 = (SortedNatTree.contains FastRows.row2683) := by
    change Data.row2683.contains = (SortedNatTree.contains FastRows.row2683)
    exact FastRows.row2683_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root257Child3 : search edge 3 [1657, 2683] root257Candidates3 4340 winner = winner := by decide
theorem root257Step3 : scan edge (search edge 3) 4 [1657] 1657 root257Tail3 winner =
    scan edge (search edge 3) 4 [1657] 1657 root257Tail4 winner := by
  rw [root257Tail3, scan, if_neg (by decide), root257Candidates3_checked]
  change scan edge (search edge 3) 4 [1657] 1657 root257Tail4
    (if 3 ≤ root257Candidates3.length then search edge 3 [1657, 2683] root257Candidates3 4340 winner else winner) = _
  rw [if_pos (by decide), root257Child3]
def root257Candidates4 : List Nat := [3967, 8647, 10789]
set_option maxRecDepth 8192 in
theorem root257Candidates4_checked : candidates edge 4 1657 2791 root257Tail5 winner.bound = root257Candidates4 := by
  unfold candidates
  have row : edge 2791 = (SortedNatTree.contains FastRows.row2791) := by
    change Data.row2791.contains = (SortedNatTree.contains FastRows.row2791)
    exact FastRows.row2791_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root257Child4 : search edge 3 [1657, 2791] root257Candidates4 4448 winner = winner := by decide
theorem root257Step4 : scan edge (search edge 3) 4 [1657] 1657 root257Tail4 winner =
    scan edge (search edge 3) 4 [1657] 1657 root257Tail5 winner := by
  rw [root257Tail4, scan, if_neg (by decide), root257Candidates4_checked]
  change scan edge (search edge 3) 4 [1657] 1657 root257Tail5
    (if 3 ≤ root257Candidates4.length then search edge 3 [1657, 2791] root257Candidates4 4448 winner else winner) = _
  rw [if_pos (by decide), root257Child4]
def root257Candidates5 : List Nat := [2917, 4273, 6733]
set_option maxRecDepth 8192 in
theorem root257Candidates5_checked : candidates edge 4 1657 2887 root257Tail6 winner.bound = root257Candidates5 := by
  unfold candidates
  have row : edge 2887 = (SortedNatTree.contains FastRows.row2887) := by
    change Data.row2887.contains = (SortedNatTree.contains FastRows.row2887)
    exact FastRows.row2887_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root257Child5 : search edge 3 [1657, 2887] root257Candidates5 4544 winner = winner := by decide
theorem root257Step5 : scan edge (search edge 3) 4 [1657] 1657 root257Tail5 winner =
    scan edge (search edge 3) 4 [1657] 1657 root257Tail6 winner := by
  rw [root257Tail5, scan, if_neg (by decide), root257Candidates5_checked]
  change scan edge (search edge 3) 4 [1657] 1657 root257Tail6
    (if 3 ≤ root257Candidates5.length then search edge 3 [1657, 2887] root257Candidates5 4544 winner else winner) = _
  rw [if_pos (by decide), root257Child5]
def root257Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root257Candidates6_checked : candidates edge 4 1657 2917 root257Tail7 winner.bound = root257Candidates6 := by
  unfold candidates
  have row : edge 2917 = (SortedNatTree.contains FastRows.row2917) := by
    change Data.row2917.contains = (SortedNatTree.contains FastRows.row2917)
    exact FastRows.row2917_checked.symm
  rw [row]
  decide
theorem root257Step6 : scan edge (search edge 3) 4 [1657] 1657 root257Tail6 winner =
    scan edge (search edge 3) 4 [1657] 1657 root257Tail7 winner := by
  rw [root257Tail6, scan, if_neg (by decide), root257Candidates6_checked]
  change scan edge (search edge 3) 4 [1657] 1657 root257Tail7
    (if 3 ≤ root257Candidates6.length then search edge 3 [1657, 2917] root257Candidates6 4574 winner else winner) = _
  rw [if_neg (by decide)]
def root257Candidates7 : List Nat := [4969, 17029]
set_option maxRecDepth 8192 in
theorem root257Candidates7_checked : candidates edge 4 1657 3463 root257Tail8 winner.bound = root257Candidates7 := by
  unfold candidates
  have row : edge 3463 = (SortedNatTree.contains FastRows.row3463) := by
    change Data.row3463.contains = (SortedNatTree.contains FastRows.row3463)
    exact FastRows.row3463_checked.symm
  rw [row]
  decide
theorem root257Step7 : scan edge (search edge 3) 4 [1657] 1657 root257Tail7 winner =
    scan edge (search edge 3) 4 [1657] 1657 root257Tail8 winner := by
  rw [root257Tail7, scan, if_neg (by decide), root257Candidates7_checked]
  change scan edge (search edge 3) 4 [1657] 1657 root257Tail8
    (if 3 ≤ root257Candidates7.length then search edge 3 [1657, 3463] root257Candidates7 5120 winner else winner) = _
  rw [if_neg (by decide)]
def root257Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root257Candidates8_checked : candidates edge 4 1657 3967 root257Tail9 winner.bound = root257Candidates8 := by
  unfold candidates
  have row : edge 3967 = (SortedNatTree.contains FastRows.row3967) := by
    change Data.row3967.contains = (SortedNatTree.contains FastRows.row3967)
    exact FastRows.row3967_checked.symm
  rw [row]
  decide
theorem root257Step8 : scan edge (search edge 3) 4 [1657] 1657 root257Tail8 winner =
    scan edge (search edge 3) 4 [1657] 1657 root257Tail9 winner := by
  rw [root257Tail8, scan, if_neg (by decide), root257Candidates8_checked]
  change scan edge (search edge 3) 4 [1657] 1657 root257Tail9
    (if 3 ≤ root257Candidates8.length then search edge 3 [1657, 3967] root257Candidates8 5624 winner else winner) = _
  rw [if_neg (by decide)]
def root257Candidates9 : List Nat := [8167, 10627, 10639]
set_option maxRecDepth 8192 in
theorem root257Candidates9_checked : candidates edge 4 1657 4111 root257Tail10 winner.bound = root257Candidates9 := by
  unfold candidates
  have row : edge 4111 = (SortedNatTree.contains FastRows.row4111) := by
    change Data.row4111.contains = (SortedNatTree.contains FastRows.row4111)
    exact FastRows.row4111_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root257Child9 : search edge 3 [1657, 4111] root257Candidates9 5768 winner = winner := by decide
theorem root257Step9 : scan edge (search edge 3) 4 [1657] 1657 root257Tail9 winner =
    scan edge (search edge 3) 4 [1657] 1657 root257Tail10 winner := by
  rw [root257Tail9, scan, if_neg (by decide), root257Candidates9_checked]
  change scan edge (search edge 3) 4 [1657] 1657 root257Tail10
    (if 3 ≤ root257Candidates9.length then search edge 3 [1657, 4111] root257Candidates9 5768 winner else winner) = _
  rw [if_pos (by decide), root257Child9]
def root257Candidates10 : List Nat := [4969]
set_option maxRecDepth 8192 in
theorem root257Candidates10_checked : candidates edge 4 1657 4273 root257Tail11 winner.bound = root257Candidates10 := by
  unfold candidates
  have row : edge 4273 = (SortedNatTree.contains FastRows.row4273) := by
    change Data.row4273.contains = (SortedNatTree.contains FastRows.row4273)
    exact FastRows.row4273_checked.symm
  rw [row]
  decide
theorem root257Step10 : scan edge (search edge 3) 4 [1657] 1657 root257Tail10 winner =
    scan edge (search edge 3) 4 [1657] 1657 root257Tail11 winner := by
  rw [root257Tail10, scan, if_neg (by decide), root257Candidates10_checked]
  change scan edge (search edge 3) 4 [1657] 1657 root257Tail11
    (if 3 ≤ root257Candidates10.length then search edge 3 [1657, 4273] root257Candidates10 5930 winner else winner) = _
  rw [if_neg (by decide)]
def root257Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root257Candidates11_checked : candidates edge 4 1657 4549 root257Tail12 winner.bound = root257Candidates11 := by
  unfold candidates
  have row : edge 4549 = (SortedNatTree.contains FastRows.row4549) := by
    change Data.row4549.contains = (SortedNatTree.contains FastRows.row4549)
    exact FastRows.row4549_checked.symm
  rw [row]
  decide
theorem root257Step11 : scan edge (search edge 3) 4 [1657] 1657 root257Tail11 winner =
    scan edge (search edge 3) 4 [1657] 1657 root257Tail12 winner := by
  rw [root257Tail11, scan, if_neg (by decide), root257Candidates11_checked]
  change scan edge (search edge 3) 4 [1657] 1657 root257Tail12
    (if 3 ≤ root257Candidates11.length then search edge 3 [1657, 4549] root257Candidates11 6206 winner else winner) = _
  rw [if_neg (by decide)]
def root257Candidates12 : List Nat := [6397, 9817]
set_option maxRecDepth 8192 in
theorem root257Candidates12_checked : candidates edge 4 1657 4969 root257Tail13 winner.bound = root257Candidates12 := by
  unfold candidates
  have row : edge 4969 = (SortedNatTree.contains FastRows.row4969) := by
    change Data.row4969.contains = (SortedNatTree.contains FastRows.row4969)
    exact FastRows.row4969_checked.symm
  rw [row]
  decide
theorem root257Step12 : scan edge (search edge 3) 4 [1657] 1657 root257Tail12 winner =
    scan edge (search edge 3) 4 [1657] 1657 root257Tail13 winner := by
  rw [root257Tail12, scan, if_neg (by decide), root257Candidates12_checked]
  change scan edge (search edge 3) 4 [1657] 1657 root257Tail13
    (if 3 ≤ root257Candidates12.length then search edge 3 [1657, 4969] root257Candidates12 6626 winner else winner) = _
  rw [if_neg (by decide)]
def root257Candidates13 : List Nat := [8269, 10789, 16693]
set_option maxRecDepth 8192 in
theorem root257Candidates13_checked : candidates edge 4 1657 5059 root257Tail14 winner.bound = root257Candidates13 := by
  unfold candidates
  have row : edge 5059 = (SortedNatTree.contains FastRows.row5059) := by
    change Data.row5059.contains = (SortedNatTree.contains FastRows.row5059)
    exact FastRows.row5059_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root257Child13 : search edge 3 [1657, 5059] root257Candidates13 6716 winner = winner := by decide
theorem root257Step13 : scan edge (search edge 3) 4 [1657] 1657 root257Tail13 winner =
    scan edge (search edge 3) 4 [1657] 1657 root257Tail14 winner := by
  rw [root257Tail13, scan, if_neg (by decide), root257Candidates13_checked]
  change scan edge (search edge 3) 4 [1657] 1657 root257Tail14
    (if 3 ≤ root257Candidates13.length then search edge 3 [1657, 5059] root257Candidates13 6716 winner else winner) = _
  rw [if_pos (by decide), root257Child13]
def root257Candidates14 : List Nat := [6397, 7237, 9931]
set_option maxRecDepth 8192 in
theorem root257Candidates14_checked : candidates edge 4 1657 5101 root257Tail15 winner.bound = root257Candidates14 := by
  unfold candidates
  have row : edge 5101 = (SortedNatTree.contains FastRows.row5101) := by
    change Data.row5101.contains = (SortedNatTree.contains FastRows.row5101)
    exact FastRows.row5101_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root257Child14 : search edge 3 [1657, 5101] root257Candidates14 6758 winner = winner := by decide
theorem root257Step14 : scan edge (search edge 3) 4 [1657] 1657 root257Tail14 winner =
    scan edge (search edge 3) 4 [1657] 1657 root257Tail15 winner := by
  rw [root257Tail14, scan, if_neg (by decide), root257Candidates14_checked]
  change scan edge (search edge 3) 4 [1657] 1657 root257Tail15
    (if 3 ≤ root257Candidates14.length then search edge 3 [1657, 5101] root257Candidates14 6758 winner else winner) = _
  rw [if_pos (by decide), root257Child14]
theorem root257Scan15 : scan edge (search edge 3) 4 [1657] 1657 root257Tail15 winner = winner := by rw [root257Tail15, scan, if_pos (by decide)]
theorem root257Scan14 : scan edge (search edge 3) 4 [1657] 1657 root257Tail14 winner = winner :=
  (root257Step14).trans root257Scan15
theorem root257Scan13 : scan edge (search edge 3) 4 [1657] 1657 root257Tail13 winner = winner :=
  (root257Step13).trans root257Scan14
theorem root257Scan12 : scan edge (search edge 3) 4 [1657] 1657 root257Tail12 winner = winner :=
  (root257Step12).trans root257Scan13
theorem root257Scan11 : scan edge (search edge 3) 4 [1657] 1657 root257Tail11 winner = winner :=
  (root257Step11).trans root257Scan12
theorem root257Scan10 : scan edge (search edge 3) 4 [1657] 1657 root257Tail10 winner = winner :=
  (root257Step10).trans root257Scan11
theorem root257Scan9 : scan edge (search edge 3) 4 [1657] 1657 root257Tail9 winner = winner :=
  (root257Step9).trans root257Scan10
theorem root257Scan8 : scan edge (search edge 3) 4 [1657] 1657 root257Tail8 winner = winner :=
  (root257Step8).trans root257Scan9
theorem root257Scan7 : scan edge (search edge 3) 4 [1657] 1657 root257Tail7 winner = winner :=
  (root257Step7).trans root257Scan8
theorem root257Scan6 : scan edge (search edge 3) 4 [1657] 1657 root257Tail6 winner = winner :=
  (root257Step6).trans root257Scan7
theorem root257Scan5 : scan edge (search edge 3) 4 [1657] 1657 root257Tail5 winner = winner :=
  (root257Step5).trans root257Scan6
theorem root257Scan4 : scan edge (search edge 3) 4 [1657] 1657 root257Tail4 winner = winner :=
  (root257Step4).trans root257Scan5
theorem root257Scan3 : scan edge (search edge 3) 4 [1657] 1657 root257Tail3 winner = winner :=
  (root257Step3).trans root257Scan4
theorem root257Scan2 : scan edge (search edge 3) 4 [1657] 1657 root257Tail2 winner = winner :=
  (root257Step2).trans root257Scan3
theorem root257Scan1 : scan edge (search edge 3) 4 [1657] 1657 root257Tail1 winner = winner :=
  (root257Step1).trans root257Scan2
theorem root257Scan0 : scan edge (search edge 3) 4 [1657] 1657 root257Tail0 winner = winner :=
  (root257Step0).trans root257Scan1
set_option maxRecDepth 4096 in
theorem root257Branch : search edge 4 [1657] root257Candidates 1657 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1657] 1657 root257Tail0 winner = winner
  exact root257Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
