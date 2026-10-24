import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root177Candidates : List Nat := Data.row1069
set_option maxRecDepth 8192 in
theorem root177Candidates_checked : candidates edge 5 0 1069 tail178 winner.bound = root177Candidates := by
  apply candidates_eq_row edge 5 0 1069 tail178 root177Candidates winner.bound tail178_ordered
  · decide
  · decide
  · funext q
    rfl
def root177Tail54 : List Nat := []
def root177Tail53 : List Nat := 24877 :: root177Tail54
def root177Tail52 : List Nat := 23473 :: root177Tail53
def root177Tail51 : List Nat := 23017 :: root177Tail52
def root177Tail50 : List Nat := 21817 :: root177Tail51
def root177Tail49 : List Nat := 21799 :: root177Tail50
def root177Tail48 : List Nat := 21673 :: root177Tail49
def root177Tail47 : List Nat := 21433 :: root177Tail48
def root177Tail46 : List Nat := 21013 :: root177Tail47
def root177Tail45 : List Nat := 20719 :: root177Tail46
def root177Tail44 : List Nat := 19717 :: root177Tail45
def root177Tail43 : List Nat := 19531 :: root177Tail44
def root177Tail42 : List Nat := 19207 :: root177Tail43
def root177Tail41 : List Nat := 18397 :: root177Tail42
def root177Tail40 : List Nat := 17467 :: root177Tail41
def root177Tail39 : List Nat := 17401 :: root177Tail40
def root177Tail38 : List Nat := 17011 :: root177Tail39
def root177Tail37 : List Nat := 16633 :: root177Tail38
def root177Tail36 : List Nat := 15031 :: root177Tail37
def root177Tail35 : List Nat := 14779 :: root177Tail36
def root177Tail34 : List Nat := 14449 :: root177Tail35
def root177Tail33 : List Nat := 13309 :: root177Tail34
def root177Tail32 : List Nat := 12853 :: root177Tail33
def root177Tail31 : List Nat := 12343 :: root177Tail32
def root177Tail30 : List Nat := 12049 :: root177Tail31
def root177Tail29 : List Nat := 11551 :: root177Tail30
def root177Tail28 : List Nat := 11443 :: root177Tail29
def root177Tail27 : List Nat := 11353 :: root177Tail28
def root177Tail26 : List Nat := 11173 :: root177Tail27
def root177Tail25 : List Nat := 10729 :: root177Tail26
def root177Tail24 : List Nat := 9133 :: root177Tail25
def root177Tail23 : List Nat := 8629 :: root177Tail24
def root177Tail22 : List Nat := 8461 :: root177Tail23
def root177Tail21 : List Nat := 8329 :: root177Tail22
def root177Tail20 : List Nat := 8221 :: root177Tail21
def root177Tail19 : List Nat := 8161 :: root177Tail20
def root177Tail18 : List Nat := 7993 :: root177Tail19
def root177Tail17 : List Nat := 7069 :: root177Tail18
def root177Tail16 : List Nat := 6883 :: root177Tail17
def root177Tail15 : List Nat := 6841 :: root177Tail16
def root177Tail14 : List Nat := 6529 :: root177Tail15
def root177Tail13 : List Nat := 6337 :: root177Tail14
def root177Tail12 : List Nat := 5683 :: root177Tail13
def root177Tail11 : List Nat := 5059 :: root177Tail12
def root177Tail10 : List Nat := 4909 :: root177Tail11
def root177Tail9 : List Nat := 4261 :: root177Tail10
def root177Tail8 : List Nat := 4219 :: root177Tail9
def root177Tail7 : List Nat := 3313 :: root177Tail8
def root177Tail6 : List Nat := 3061 :: root177Tail7
def root177Tail5 : List Nat := 2659 :: root177Tail6
def root177Tail4 : List Nat := 2281 :: root177Tail5
def root177Tail3 : List Nat := 2203 :: root177Tail4
def root177Tail2 : List Nat := 2113 :: root177Tail3
def root177Tail1 : List Nat := 1609 :: root177Tail2
def root177Tail0 : List Nat := 1321 :: root177Tail1
def root177Candidates0 : List Nat := [4261, 14779]
set_option maxRecDepth 8192 in
theorem root177Candidates0_checked : candidates edge 4 1069 1321 root177Tail1 winner.bound = root177Candidates0 := by
  unfold candidates
  have row : edge 1321 = (SortedNatTree.contains FastRows.row1321) := by
    change Data.row1321.contains = (SortedNatTree.contains FastRows.row1321)
    exact FastRows.row1321_checked.symm
  rw [row]
  decide
theorem root177Step0 : scan edge (search edge 3) 4 [1069] 1069 root177Tail0 winner =
    scan edge (search edge 3) 4 [1069] 1069 root177Tail1 winner := by
  rw [root177Tail0, scan, if_neg (by decide), root177Candidates0_checked]
  change scan edge (search edge 3) 4 [1069] 1069 root177Tail1
    (if 3 ≤ root177Candidates0.length then search edge 3 [1069, 1321] root177Candidates0 2390 winner else winner) = _
  rw [if_neg (by decide)]
def root177Candidates1 : List Nat := [18397]
set_option maxRecDepth 8192 in
theorem root177Candidates1_checked : candidates edge 4 1069 1609 root177Tail2 winner.bound = root177Candidates1 := by
  unfold candidates
  have row : edge 1609 = (SortedNatTree.contains FastRows.row1609) := by
    change Data.row1609.contains = (SortedNatTree.contains FastRows.row1609)
    exact FastRows.row1609_checked.symm
  rw [row]
  decide
theorem root177Step1 : scan edge (search edge 3) 4 [1069] 1069 root177Tail1 winner =
    scan edge (search edge 3) 4 [1069] 1069 root177Tail2 winner := by
  rw [root177Tail1, scan, if_neg (by decide), root177Candidates1_checked]
  change scan edge (search edge 3) 4 [1069] 1069 root177Tail2
    (if 3 ≤ root177Candidates1.length then search edge 3 [1069, 1609] root177Candidates1 2678 winner else winner) = _
  rw [if_neg (by decide)]
def root177Candidates2 : List Nat := [2659]
set_option maxRecDepth 8192 in
theorem root177Candidates2_checked : candidates edge 4 1069 2113 root177Tail3 winner.bound = root177Candidates2 := by
  unfold candidates
  have row : edge 2113 = (SortedNatTree.contains FastRows.row2113) := by
    change Data.row2113.contains = (SortedNatTree.contains FastRows.row2113)
    exact FastRows.row2113_checked.symm
  rw [row]
  decide
theorem root177Step2 : scan edge (search edge 3) 4 [1069] 1069 root177Tail2 winner =
    scan edge (search edge 3) 4 [1069] 1069 root177Tail3 winner := by
  rw [root177Tail2, scan, if_neg (by decide), root177Candidates2_checked]
  change scan edge (search edge 3) 4 [1069] 1069 root177Tail3
    (if 3 ≤ root177Candidates2.length then search edge 3 [1069, 2113] root177Candidates2 3182 winner else winner) = _
  rw [if_neg (by decide)]
def root177Candidates3 : List Nat := [3061, 6529, 8161, 9133, 11353, 14449, 19207]
set_option maxRecDepth 8192 in
theorem root177Candidates3_checked : candidates edge 4 1069 2203 root177Tail4 winner.bound = root177Candidates3 := by
  unfold candidates
  have row : edge 2203 = (SortedNatTree.contains FastRows.row2203) := by
    change Data.row2203.contains = (SortedNatTree.contains FastRows.row2203)
    exact FastRows.row2203_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root177Child3 : search edge 3 [1069, 2203] root177Candidates3 3272 winner = winner := by decide
theorem root177Step3 : scan edge (search edge 3) 4 [1069] 1069 root177Tail3 winner =
    scan edge (search edge 3) 4 [1069] 1069 root177Tail4 winner := by
  rw [root177Tail3, scan, if_neg (by decide), root177Candidates3_checked]
  change scan edge (search edge 3) 4 [1069] 1069 root177Tail4
    (if 3 ≤ root177Candidates3.length then search edge 3 [1069, 2203] root177Candidates3 3272 winner else winner) = _
  rw [if_pos (by decide), root177Child3]
def root177Candidates4 : List Nat := [12853]
set_option maxRecDepth 8192 in
theorem root177Candidates4_checked : candidates edge 4 1069 2281 root177Tail5 winner.bound = root177Candidates4 := by
  unfold candidates
  have row : edge 2281 = (SortedNatTree.contains FastRows.row2281) := by
    change Data.row2281.contains = (SortedNatTree.contains FastRows.row2281)
    exact FastRows.row2281_checked.symm
  rw [row]
  decide
theorem root177Step4 : scan edge (search edge 3) 4 [1069] 1069 root177Tail4 winner =
    scan edge (search edge 3) 4 [1069] 1069 root177Tail5 winner := by
  rw [root177Tail4, scan, if_neg (by decide), root177Candidates4_checked]
  change scan edge (search edge 3) 4 [1069] 1069 root177Tail5
    (if 3 ≤ root177Candidates4.length then search edge 3 [1069, 2281] root177Candidates4 3350 winner else winner) = _
  rw [if_neg (by decide)]
def root177Candidates5 : List Nat := [12853]
set_option maxRecDepth 8192 in
theorem root177Candidates5_checked : candidates edge 4 1069 2659 root177Tail6 winner.bound = root177Candidates5 := by
  unfold candidates
  have row : edge 2659 = (SortedNatTree.contains FastRows.row2659) := by
    change Data.row2659.contains = (SortedNatTree.contains FastRows.row2659)
    exact FastRows.row2659_checked.symm
  rw [row]
  decide
theorem root177Step5 : scan edge (search edge 3) 4 [1069] 1069 root177Tail5 winner =
    scan edge (search edge 3) 4 [1069] 1069 root177Tail6 winner := by
  rw [root177Tail5, scan, if_neg (by decide), root177Candidates5_checked]
  change scan edge (search edge 3) 4 [1069] 1069 root177Tail6
    (if 3 ≤ root177Candidates5.length then search edge 3 [1069, 2659] root177Candidates5 3728 winner else winner) = _
  rw [if_neg (by decide)]
def root177Candidates6 : List Nat := [12853, 21433]
set_option maxRecDepth 8192 in
theorem root177Candidates6_checked : candidates edge 4 1069 3061 root177Tail7 winner.bound = root177Candidates6 := by
  unfold candidates
  have row : edge 3061 = (SortedNatTree.contains FastRows.row3061) := by
    change Data.row3061.contains = (SortedNatTree.contains FastRows.row3061)
    exact FastRows.row3061_checked.symm
  rw [row]
  decide
theorem root177Step6 : scan edge (search edge 3) 4 [1069] 1069 root177Tail6 winner =
    scan edge (search edge 3) 4 [1069] 1069 root177Tail7 winner := by
  rw [root177Tail6, scan, if_neg (by decide), root177Candidates6_checked]
  change scan edge (search edge 3) 4 [1069] 1069 root177Tail7
    (if 3 ≤ root177Candidates6.length then search edge 3 [1069, 3061] root177Candidates6 4130 winner else winner) = _
  rw [if_neg (by decide)]
def root177Candidates7 : List Nat := [8629, 14449]
set_option maxRecDepth 8192 in
theorem root177Candidates7_checked : candidates edge 4 1069 3313 root177Tail8 winner.bound = root177Candidates7 := by
  unfold candidates
  have row : edge 3313 = (SortedNatTree.contains FastRows.row3313) := by
    change Data.row3313.contains = (SortedNatTree.contains FastRows.row3313)
    exact FastRows.row3313_checked.symm
  rw [row]
  decide
theorem root177Step7 : scan edge (search edge 3) 4 [1069] 1069 root177Tail7 winner =
    scan edge (search edge 3) 4 [1069] 1069 root177Tail8 winner := by
  rw [root177Tail7, scan, if_neg (by decide), root177Candidates7_checked]
  change scan edge (search edge 3) 4 [1069] 1069 root177Tail8
    (if 3 ≤ root177Candidates7.length then search edge 3 [1069, 3313] root177Candidates7 4382 winner else winner) = _
  rw [if_neg (by decide)]
def root177Candidates8 : List Nat := [6337, 6529, 6883]
set_option maxRecDepth 8192 in
theorem root177Candidates8_checked : candidates edge 4 1069 4219 root177Tail9 winner.bound = root177Candidates8 := by
  unfold candidates
  have row : edge 4219 = (SortedNatTree.contains FastRows.row4219) := by
    change Data.row4219.contains = (SortedNatTree.contains FastRows.row4219)
    exact FastRows.row4219_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root177Child8 : search edge 3 [1069, 4219] root177Candidates8 5288 winner = winner := by decide
theorem root177Step8 : scan edge (search edge 3) 4 [1069] 1069 root177Tail8 winner =
    scan edge (search edge 3) 4 [1069] 1069 root177Tail9 winner := by
  rw [root177Tail8, scan, if_neg (by decide), root177Candidates8_checked]
  change scan edge (search edge 3) 4 [1069] 1069 root177Tail9
    (if 3 ≤ root177Candidates8.length then search edge 3 [1069, 4219] root177Candidates8 5288 winner else winner) = _
  rw [if_pos (by decide), root177Child8]
def root177Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root177Candidates9_checked : candidates edge 4 1069 4261 root177Tail10 winner.bound = root177Candidates9 := by
  unfold candidates
  have row : edge 4261 = (SortedNatTree.contains FastRows.row4261) := by
    change Data.row4261.contains = (SortedNatTree.contains FastRows.row4261)
    exact FastRows.row4261_checked.symm
  rw [row]
  decide
theorem root177Step9 : scan edge (search edge 3) 4 [1069] 1069 root177Tail9 winner =
    scan edge (search edge 3) 4 [1069] 1069 root177Tail10 winner := by
  rw [root177Tail9, scan, if_neg (by decide), root177Candidates9_checked]
  change scan edge (search edge 3) 4 [1069] 1069 root177Tail10
    (if 3 ≤ root177Candidates9.length then search edge 3 [1069, 4261] root177Candidates9 5330 winner else winner) = _
  rw [if_neg (by decide)]
def root177Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root177Candidates10_checked : candidates edge 4 1069 4909 root177Tail11 winner.bound = root177Candidates10 := by
  unfold candidates
  have row : edge 4909 = (SortedNatTree.contains FastRows.row4909) := by
    change Data.row4909.contains = (SortedNatTree.contains FastRows.row4909)
    exact FastRows.row4909_checked.symm
  rw [row]
  decide
theorem root177Step10 : scan edge (search edge 3) 4 [1069] 1069 root177Tail10 winner =
    scan edge (search edge 3) 4 [1069] 1069 root177Tail11 winner := by
  rw [root177Tail10, scan, if_neg (by decide), root177Candidates10_checked]
  change scan edge (search edge 3) 4 [1069] 1069 root177Tail11
    (if 3 ≤ root177Candidates10.length then search edge 3 [1069, 4909] root177Candidates10 5978 winner else winner) = _
  rw [if_neg (by decide)]
def root177Candidates11 : List Nat := [10729]
set_option maxRecDepth 8192 in
theorem root177Candidates11_checked : candidates edge 4 1069 5059 root177Tail12 winner.bound = root177Candidates11 := by
  unfold candidates
  have row : edge 5059 = (SortedNatTree.contains FastRows.row5059) := by
    change Data.row5059.contains = (SortedNatTree.contains FastRows.row5059)
    exact FastRows.row5059_checked.symm
  rw [row]
  decide
theorem root177Step11 : scan edge (search edge 3) 4 [1069] 1069 root177Tail11 winner =
    scan edge (search edge 3) 4 [1069] 1069 root177Tail12 winner := by
  rw [root177Tail11, scan, if_neg (by decide), root177Candidates11_checked]
  change scan edge (search edge 3) 4 [1069] 1069 root177Tail12
    (if 3 ≤ root177Candidates11.length then search edge 3 [1069, 5059] root177Candidates11 6128 winner else winner) = _
  rw [if_neg (by decide)]
def root177Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root177Candidates12_checked : candidates edge 4 1069 5683 root177Tail13 winner.bound = root177Candidates12 := by
  unfold candidates
  have row : edge 5683 = (SortedNatTree.contains FastRows.row5683) := by
    change Data.row5683.contains = (SortedNatTree.contains FastRows.row5683)
    exact FastRows.row5683_checked.symm
  rw [row]
  decide
theorem root177Step12 : scan edge (search edge 3) 4 [1069] 1069 root177Tail12 winner =
    scan edge (search edge 3) 4 [1069] 1069 root177Tail13 winner := by
  rw [root177Tail12, scan, if_neg (by decide), root177Candidates12_checked]
  change scan edge (search edge 3) 4 [1069] 1069 root177Tail13
    (if 3 ≤ root177Candidates12.length then search edge 3 [1069, 5683] root177Candidates12 6752 winner else winner) = _
  rw [if_neg (by decide)]
theorem root177Scan13 : scan edge (search edge 3) 4 [1069] 1069 root177Tail13 winner = winner := by rw [root177Tail13, scan, if_pos (by decide)]
theorem root177Scan12 : scan edge (search edge 3) 4 [1069] 1069 root177Tail12 winner = winner :=
  (root177Step12).trans root177Scan13
theorem root177Scan11 : scan edge (search edge 3) 4 [1069] 1069 root177Tail11 winner = winner :=
  (root177Step11).trans root177Scan12
theorem root177Scan10 : scan edge (search edge 3) 4 [1069] 1069 root177Tail10 winner = winner :=
  (root177Step10).trans root177Scan11
theorem root177Scan9 : scan edge (search edge 3) 4 [1069] 1069 root177Tail9 winner = winner :=
  (root177Step9).trans root177Scan10
theorem root177Scan8 : scan edge (search edge 3) 4 [1069] 1069 root177Tail8 winner = winner :=
  (root177Step8).trans root177Scan9
theorem root177Scan7 : scan edge (search edge 3) 4 [1069] 1069 root177Tail7 winner = winner :=
  (root177Step7).trans root177Scan8
theorem root177Scan6 : scan edge (search edge 3) 4 [1069] 1069 root177Tail6 winner = winner :=
  (root177Step6).trans root177Scan7
theorem root177Scan5 : scan edge (search edge 3) 4 [1069] 1069 root177Tail5 winner = winner :=
  (root177Step5).trans root177Scan6
theorem root177Scan4 : scan edge (search edge 3) 4 [1069] 1069 root177Tail4 winner = winner :=
  (root177Step4).trans root177Scan5
theorem root177Scan3 : scan edge (search edge 3) 4 [1069] 1069 root177Tail3 winner = winner :=
  (root177Step3).trans root177Scan4
theorem root177Scan2 : scan edge (search edge 3) 4 [1069] 1069 root177Tail2 winner = winner :=
  (root177Step2).trans root177Scan3
theorem root177Scan1 : scan edge (search edge 3) 4 [1069] 1069 root177Tail1 winner = winner :=
  (root177Step1).trans root177Scan2
theorem root177Scan0 : scan edge (search edge 3) 4 [1069] 1069 root177Tail0 winner = winner :=
  (root177Step0).trans root177Scan1
set_option maxRecDepth 4096 in
theorem root177Branch : search edge 4 [1069] root177Candidates 1069 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1069] 1069 root177Tail0 winner = winner
  exact root177Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
