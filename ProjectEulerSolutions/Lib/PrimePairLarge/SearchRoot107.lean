import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root107Candidates : List Nat := Data.row601
set_option maxRecDepth 8192 in
theorem root107Candidates_checked : candidates edge 5 0 601 tail108 winner.bound = root107Candidates := by
  apply candidates_eq_row edge 5 0 601 tail108 root107Candidates winner.bound tail108_ordered
  · decide
  · decide
  · funext q
    rfl
def root107Tail57 : List Nat := []
def root107Tail56 : List Nat := 24517 :: root107Tail57
def root107Tail55 : List Nat := 24061 :: root107Tail56
def root107Tail54 : List Nat := 23929 :: root107Tail55
def root107Tail53 : List Nat := 23269 :: root107Tail54
def root107Tail52 : List Nat := 23131 :: root107Tail53
def root107Tail51 : List Nat := 22291 :: root107Tail52
def root107Tail50 : List Nat := 22159 :: root107Tail51
def root107Tail49 : List Nat := 21961 :: root107Tail50
def root107Tail48 : List Nat := 21751 :: root107Tail49
def root107Tail47 : List Nat := 21157 :: root107Tail48
def root107Tail46 : List Nat := 20323 :: root107Tail47
def root107Tail45 : List Nat := 19681 :: root107Tail46
def root107Tail44 : List Nat := 19219 :: root107Tail45
def root107Tail43 : List Nat := 18691 :: root107Tail44
def root107Tail42 : List Nat := 18523 :: root107Tail43
def root107Tail41 : List Nat := 18313 :: root107Tail42
def root107Tail40 : List Nat := 18181 :: root107Tail41
def root107Tail39 : List Nat := 17749 :: root107Tail40
def root107Tail38 : List Nat := 17623 :: root107Tail39
def root107Tail37 : List Nat := 17581 :: root107Tail38
def root107Tail36 : List Nat := 17041 :: root107Tail37
def root107Tail35 : List Nat := 16087 :: root107Tail36
def root107Tail34 : List Nat := 15877 :: root107Tail35
def root107Tail33 : List Nat := 15331 :: root107Tail34
def root107Tail32 : List Nat := 15241 :: root107Tail33
def root107Tail31 : List Nat := 13597 :: root107Tail32
def root107Tail30 : List Nat := 13219 :: root107Tail31
def root107Tail29 : List Nat := 12841 :: root107Tail30
def root107Tail28 : List Nat := 12583 :: root107Tail29
def root107Tail27 : List Nat := 11827 :: root107Tail28
def root107Tail26 : List Nat := 11503 :: root107Tail27
def root107Tail25 : List Nat := 11197 :: root107Tail26
def root107Tail24 : List Nat := 11071 :: root107Tail25
def root107Tail23 : List Nat := 10909 :: root107Tail24
def root107Tail22 : List Nat := 10111 :: root107Tail23
def root107Tail21 : List Nat := 9817 :: root107Tail22
def root107Tail20 : List Nat := 8377 :: root107Tail21
def root107Tail19 : List Nat := 8317 :: root107Tail20
def root107Tail18 : List Nat := 7639 :: root107Tail19
def root107Tail17 : List Nat := 7603 :: root107Tail18
def root107Tail16 : List Nat := 7411 :: root107Tail17
def root107Tail15 : List Nat := 7213 :: root107Tail16
def root107Tail14 : List Nat := 5197 :: root107Tail15
def root107Tail13 : List Nat := 4909 :: root107Tail14
def root107Tail12 : List Nat := 4663 :: root107Tail13
def root107Tail11 : List Nat := 4159 :: root107Tail12
def root107Tail10 : List Nat := 3187 :: root107Tail11
def root107Tail9 : List Nat := 2719 :: root107Tail10
def root107Tail8 : List Nat := 2689 :: root107Tail9
def root107Tail7 : List Nat := 2011 :: root107Tail8
def root107Tail6 : List Nat := 1879 :: root107Tail7
def root107Tail5 : List Nat := 1867 :: root107Tail6
def root107Tail4 : List Nat := 1627 :: root107Tail5
def root107Tail3 : List Nat := 1597 :: root107Tail4
def root107Tail2 : List Nat := 1321 :: root107Tail3
def root107Tail1 : List Nat := 1171 :: root107Tail2
def root107Tail0 : List Nat := 823 :: root107Tail1
def root107Candidates0 : List Nat := [4909, 10909]
set_option maxRecDepth 8192 in
theorem root107Candidates0_checked : candidates edge 4 601 823 root107Tail1 winner.bound = root107Candidates0 := by
  unfold candidates
  have row : edge 823 = (SortedNatTree.contains FastRows.row823) := by
    change Data.row823.contains = (SortedNatTree.contains FastRows.row823)
    exact FastRows.row823_checked.symm
  rw [row]
  decide
theorem root107Step0 : scan edge (search edge 3) 4 [601] 601 root107Tail0 winner =
    scan edge (search edge 3) 4 [601] 601 root107Tail1 winner := by
  rw [root107Tail0, scan, if_neg (by decide), root107Candidates0_checked]
  change scan edge (search edge 3) 4 [601] 601 root107Tail1
    (if 3 ≤ root107Candidates0.length then search edge 3 [601, 823] root107Candidates0 1424 winner else winner) = _
  rw [if_neg (by decide)]
def root107Candidates1 : List Nat := [2011, 4663]
set_option maxRecDepth 8192 in
theorem root107Candidates1_checked : candidates edge 4 601 1171 root107Tail2 winner.bound = root107Candidates1 := by
  unfold candidates
  have row : edge 1171 = (SortedNatTree.contains FastRows.row1171) := by
    change Data.row1171.contains = (SortedNatTree.contains FastRows.row1171)
    exact FastRows.row1171_checked.symm
  rw [row]
  decide
theorem root107Step1 : scan edge (search edge 3) 4 [601] 601 root107Tail1 winner =
    scan edge (search edge 3) 4 [601] 601 root107Tail2 winner := by
  rw [root107Tail1, scan, if_neg (by decide), root107Candidates1_checked]
  change scan edge (search edge 3) 4 [601] 601 root107Tail2
    (if 3 ≤ root107Candidates1.length then search edge 3 [601, 1171] root107Candidates1 1772 winner else winner) = _
  rw [if_neg (by decide)]
def root107Candidates2 : List Nat := [24061]
set_option maxRecDepth 8192 in
theorem root107Candidates2_checked : candidates edge 4 601 1321 root107Tail3 winner.bound = root107Candidates2 := by
  unfold candidates
  have row : edge 1321 = (SortedNatTree.contains FastRows.row1321) := by
    change Data.row1321.contains = (SortedNatTree.contains FastRows.row1321)
    exact FastRows.row1321_checked.symm
  rw [row]
  decide
theorem root107Step2 : scan edge (search edge 3) 4 [601] 601 root107Tail2 winner =
    scan edge (search edge 3) 4 [601] 601 root107Tail3 winner := by
  rw [root107Tail2, scan, if_neg (by decide), root107Candidates2_checked]
  change scan edge (search edge 3) 4 [601] 601 root107Tail3
    (if 3 ≤ root107Candidates2.length then search edge 3 [601, 1321] root107Candidates2 1922 winner else winner) = _
  rw [if_neg (by decide)]
def root107Candidates3 : List Nat := [18523]
set_option maxRecDepth 8192 in
theorem root107Candidates3_checked : candidates edge 4 601 1597 root107Tail4 winner.bound = root107Candidates3 := by
  unfold candidates
  have row : edge 1597 = (SortedNatTree.contains FastRows.row1597) := by
    change Data.row1597.contains = (SortedNatTree.contains FastRows.row1597)
    exact FastRows.row1597_checked.symm
  rw [row]
  decide
theorem root107Step3 : scan edge (search edge 3) 4 [601] 601 root107Tail3 winner =
    scan edge (search edge 3) 4 [601] 601 root107Tail4 winner := by
  rw [root107Tail3, scan, if_neg (by decide), root107Candidates3_checked]
  change scan edge (search edge 3) 4 [601] 601 root107Tail4
    (if 3 ≤ root107Candidates3.length then search edge 3 [601, 1597] root107Candidates3 2198 winner else winner) = _
  rw [if_neg (by decide)]
def root107Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root107Candidates4_checked : candidates edge 4 601 1627 root107Tail5 winner.bound = root107Candidates4 := by
  unfold candidates
  have row : edge 1627 = (SortedNatTree.contains FastRows.row1627) := by
    change Data.row1627.contains = (SortedNatTree.contains FastRows.row1627)
    exact FastRows.row1627_checked.symm
  rw [row]
  decide
theorem root107Step4 : scan edge (search edge 3) 4 [601] 601 root107Tail4 winner =
    scan edge (search edge 3) 4 [601] 601 root107Tail5 winner := by
  rw [root107Tail4, scan, if_neg (by decide), root107Candidates4_checked]
  change scan edge (search edge 3) 4 [601] 601 root107Tail5
    (if 3 ≤ root107Candidates4.length then search edge 3 [601, 1627] root107Candidates4 2228 winner else winner) = _
  rw [if_neg (by decide)]
def root107Candidates5 : List Nat := [7639, 15241, 18313]
set_option maxRecDepth 8192 in
theorem root107Candidates5_checked : candidates edge 4 601 1867 root107Tail6 winner.bound = root107Candidates5 := by
  unfold candidates
  have row : edge 1867 = (SortedNatTree.contains FastRows.row1867) := by
    change Data.row1867.contains = (SortedNatTree.contains FastRows.row1867)
    exact FastRows.row1867_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root107Child5 : search edge 3 [601, 1867] root107Candidates5 2468 winner = winner := by decide
theorem root107Step5 : scan edge (search edge 3) 4 [601] 601 root107Tail5 winner =
    scan edge (search edge 3) 4 [601] 601 root107Tail6 winner := by
  rw [root107Tail5, scan, if_neg (by decide), root107Candidates5_checked]
  change scan edge (search edge 3) 4 [601] 601 root107Tail6
    (if 3 ≤ root107Candidates5.length then search edge 3 [601, 1867] root107Candidates5 2468 winner else winner) = _
  rw [if_pos (by decide), root107Child5]
def root107Candidates6 : List Nat := [7411]
set_option maxRecDepth 8192 in
theorem root107Candidates6_checked : candidates edge 4 601 1879 root107Tail7 winner.bound = root107Candidates6 := by
  unfold candidates
  have row : edge 1879 = (SortedNatTree.contains FastRows.row1879) := by
    change Data.row1879.contains = (SortedNatTree.contains FastRows.row1879)
    exact FastRows.row1879_checked.symm
  rw [row]
  decide
theorem root107Step6 : scan edge (search edge 3) 4 [601] 601 root107Tail6 winner =
    scan edge (search edge 3) 4 [601] 601 root107Tail7 winner := by
  rw [root107Tail6, scan, if_neg (by decide), root107Candidates6_checked]
  change scan edge (search edge 3) 4 [601] 601 root107Tail7
    (if 3 ≤ root107Candidates6.length then search edge 3 [601, 1879] root107Candidates6 2480 winner else winner) = _
  rw [if_neg (by decide)]
def root107Candidates7 : List Nat := [15331, 15877, 20323, 23269]
set_option maxRecDepth 8192 in
theorem root107Candidates7_checked : candidates edge 4 601 2011 root107Tail8 winner.bound = root107Candidates7 := by
  unfold candidates
  have row : edge 2011 = (SortedNatTree.contains FastRows.row2011) := by
    change Data.row2011.contains = (SortedNatTree.contains FastRows.row2011)
    exact FastRows.row2011_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root107Child7 : search edge 3 [601, 2011] root107Candidates7 2612 winner = winner := by decide
theorem root107Step7 : scan edge (search edge 3) 4 [601] 601 root107Tail7 winner =
    scan edge (search edge 3) 4 [601] 601 root107Tail8 winner := by
  rw [root107Tail7, scan, if_neg (by decide), root107Candidates7_checked]
  change scan edge (search edge 3) 4 [601] 601 root107Tail8
    (if 3 ≤ root107Candidates7.length then search edge 3 [601, 2011] root107Candidates7 2612 winner else winner) = _
  rw [if_pos (by decide), root107Child7]
def root107Candidates8 : List Nat := [8317, 17041]
set_option maxRecDepth 8192 in
theorem root107Candidates8_checked : candidates edge 4 601 2689 root107Tail9 winner.bound = root107Candidates8 := by
  unfold candidates
  have row : edge 2689 = (SortedNatTree.contains FastRows.row2689) := by
    change Data.row2689.contains = (SortedNatTree.contains FastRows.row2689)
    exact FastRows.row2689_checked.symm
  rw [row]
  decide
theorem root107Step8 : scan edge (search edge 3) 4 [601] 601 root107Tail8 winner =
    scan edge (search edge 3) 4 [601] 601 root107Tail9 winner := by
  rw [root107Tail8, scan, if_neg (by decide), root107Candidates8_checked]
  change scan edge (search edge 3) 4 [601] 601 root107Tail9
    (if 3 ≤ root107Candidates8.length then search edge 3 [601, 2689] root107Candidates8 3290 winner else winner) = _
  rw [if_neg (by decide)]
def root107Candidates9 : List Nat := [9817, 17749, 21961]
set_option maxRecDepth 8192 in
theorem root107Candidates9_checked : candidates edge 4 601 2719 root107Tail10 winner.bound = root107Candidates9 := by
  unfold candidates
  have row : edge 2719 = (SortedNatTree.contains FastRows.row2719) := by
    change Data.row2719.contains = (SortedNatTree.contains FastRows.row2719)
    exact FastRows.row2719_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root107Child9 : search edge 3 [601, 2719] root107Candidates9 3320 winner = winner := by decide
theorem root107Step9 : scan edge (search edge 3) 4 [601] 601 root107Tail9 winner =
    scan edge (search edge 3) 4 [601] 601 root107Tail10 winner := by
  rw [root107Tail9, scan, if_neg (by decide), root107Candidates9_checked]
  change scan edge (search edge 3) 4 [601] 601 root107Tail10
    (if 3 ≤ root107Candidates9.length then search edge 3 [601, 2719] root107Candidates9 3320 winner else winner) = _
  rw [if_pos (by decide), root107Child9]
def root107Candidates10 : List Nat := [7411, 18181]
set_option maxRecDepth 8192 in
theorem root107Candidates10_checked : candidates edge 4 601 3187 root107Tail11 winner.bound = root107Candidates10 := by
  unfold candidates
  have row : edge 3187 = (SortedNatTree.contains FastRows.row3187) := by
    change Data.row3187.contains = (SortedNatTree.contains FastRows.row3187)
    exact FastRows.row3187_checked.symm
  rw [row]
  decide
theorem root107Step10 : scan edge (search edge 3) 4 [601] 601 root107Tail10 winner =
    scan edge (search edge 3) 4 [601] 601 root107Tail11 winner := by
  rw [root107Tail10, scan, if_neg (by decide), root107Candidates10_checked]
  change scan edge (search edge 3) 4 [601] 601 root107Tail11
    (if 3 ≤ root107Candidates10.length then search edge 3 [601, 3187] root107Candidates10 3788 winner else winner) = _
  rw [if_neg (by decide)]
def root107Candidates11 : List Nat := [19681]
set_option maxRecDepth 8192 in
theorem root107Candidates11_checked : candidates edge 4 601 4159 root107Tail12 winner.bound = root107Candidates11 := by
  unfold candidates
  have row : edge 4159 = (SortedNatTree.contains FastRows.row4159) := by
    change Data.row4159.contains = (SortedNatTree.contains FastRows.row4159)
    exact FastRows.row4159_checked.symm
  rw [row]
  decide
theorem root107Step11 : scan edge (search edge 3) 4 [601] 601 root107Tail11 winner =
    scan edge (search edge 3) 4 [601] 601 root107Tail12 winner := by
  rw [root107Tail11, scan, if_neg (by decide), root107Candidates11_checked]
  change scan edge (search edge 3) 4 [601] 601 root107Tail12
    (if 3 ≤ root107Candidates11.length then search edge 3 [601, 4159] root107Candidates11 4760 winner else winner) = _
  rw [if_neg (by decide)]
def root107Candidates12 : List Nat := [7603, 17749]
set_option maxRecDepth 8192 in
theorem root107Candidates12_checked : candidates edge 4 601 4663 root107Tail13 winner.bound = root107Candidates12 := by
  unfold candidates
  have row : edge 4663 = (SortedNatTree.contains FastRows.row4663) := by
    change Data.row4663.contains = (SortedNatTree.contains FastRows.row4663)
    exact FastRows.row4663_checked.symm
  rw [row]
  decide
theorem root107Step12 : scan edge (search edge 3) 4 [601] 601 root107Tail12 winner =
    scan edge (search edge 3) 4 [601] 601 root107Tail13 winner := by
  rw [root107Tail12, scan, if_neg (by decide), root107Candidates12_checked]
  change scan edge (search edge 3) 4 [601] 601 root107Tail13
    (if 3 ≤ root107Candidates12.length then search edge 3 [601, 4663] root107Candidates12 5264 winner else winner) = _
  rw [if_neg (by decide)]
def root107Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root107Candidates13_checked : candidates edge 4 601 4909 root107Tail14 winner.bound = root107Candidates13 := by
  unfold candidates
  have row : edge 4909 = (SortedNatTree.contains FastRows.row4909) := by
    change Data.row4909.contains = (SortedNatTree.contains FastRows.row4909)
    exact FastRows.row4909_checked.symm
  rw [row]
  decide
theorem root107Step13 : scan edge (search edge 3) 4 [601] 601 root107Tail13 winner =
    scan edge (search edge 3) 4 [601] 601 root107Tail14 winner := by
  rw [root107Tail13, scan, if_neg (by decide), root107Candidates13_checked]
  change scan edge (search edge 3) 4 [601] 601 root107Tail14
    (if 3 ≤ root107Candidates13.length then search edge 3 [601, 4909] root107Candidates13 5510 winner else winner) = _
  rw [if_neg (by decide)]
def root107Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root107Candidates14_checked : candidates edge 4 601 5197 root107Tail15 winner.bound = root107Candidates14 := by
  unfold candidates
  have row : edge 5197 = (SortedNatTree.contains FastRows.row5197) := by
    change Data.row5197.contains = (SortedNatTree.contains FastRows.row5197)
    exact FastRows.row5197_checked.symm
  rw [row]
  decide
theorem root107Step14 : scan edge (search edge 3) 4 [601] 601 root107Tail14 winner =
    scan edge (search edge 3) 4 [601] 601 root107Tail15 winner := by
  rw [root107Tail14, scan, if_neg (by decide), root107Candidates14_checked]
  change scan edge (search edge 3) 4 [601] 601 root107Tail15
    (if 3 ≤ root107Candidates14.length then search edge 3 [601, 5197] root107Candidates14 5798 winner else winner) = _
  rw [if_neg (by decide)]
theorem root107Scan15 : scan edge (search edge 3) 4 [601] 601 root107Tail15 winner = winner := by rw [root107Tail15, scan, if_pos (by decide)]
theorem root107Scan14 : scan edge (search edge 3) 4 [601] 601 root107Tail14 winner = winner :=
  (root107Step14).trans root107Scan15
theorem root107Scan13 : scan edge (search edge 3) 4 [601] 601 root107Tail13 winner = winner :=
  (root107Step13).trans root107Scan14
theorem root107Scan12 : scan edge (search edge 3) 4 [601] 601 root107Tail12 winner = winner :=
  (root107Step12).trans root107Scan13
theorem root107Scan11 : scan edge (search edge 3) 4 [601] 601 root107Tail11 winner = winner :=
  (root107Step11).trans root107Scan12
theorem root107Scan10 : scan edge (search edge 3) 4 [601] 601 root107Tail10 winner = winner :=
  (root107Step10).trans root107Scan11
theorem root107Scan9 : scan edge (search edge 3) 4 [601] 601 root107Tail9 winner = winner :=
  (root107Step9).trans root107Scan10
theorem root107Scan8 : scan edge (search edge 3) 4 [601] 601 root107Tail8 winner = winner :=
  (root107Step8).trans root107Scan9
theorem root107Scan7 : scan edge (search edge 3) 4 [601] 601 root107Tail7 winner = winner :=
  (root107Step7).trans root107Scan8
theorem root107Scan6 : scan edge (search edge 3) 4 [601] 601 root107Tail6 winner = winner :=
  (root107Step6).trans root107Scan7
theorem root107Scan5 : scan edge (search edge 3) 4 [601] 601 root107Tail5 winner = winner :=
  (root107Step5).trans root107Scan6
theorem root107Scan4 : scan edge (search edge 3) 4 [601] 601 root107Tail4 winner = winner :=
  (root107Step4).trans root107Scan5
theorem root107Scan3 : scan edge (search edge 3) 4 [601] 601 root107Tail3 winner = winner :=
  (root107Step3).trans root107Scan4
theorem root107Scan2 : scan edge (search edge 3) 4 [601] 601 root107Tail2 winner = winner :=
  (root107Step2).trans root107Scan3
theorem root107Scan1 : scan edge (search edge 3) 4 [601] 601 root107Tail1 winner = winner :=
  (root107Step1).trans root107Scan2
theorem root107Scan0 : scan edge (search edge 3) 4 [601] 601 root107Tail0 winner = winner :=
  (root107Step0).trans root107Scan1
set_option maxRecDepth 4096 in
theorem root107Branch : search edge 4 [601] root107Candidates 601 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [601] 601 root107Tail0 winner = winner
  exact root107Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
