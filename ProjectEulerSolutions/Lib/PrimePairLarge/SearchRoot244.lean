import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root244Candidates : List Nat := Data.row1567
set_option maxRecDepth 8192 in
theorem root244Candidates_checked : candidates edge 5 0 1567 tail245 winner.bound = root244Candidates := by
  apply candidates_eq_row edge 5 0 1567 tail245 root244Candidates winner.bound tail245_ordered
  · decide
  · decide
  · funext q
    rfl
def root244Tail51 : List Nat := []
def root244Tail50 : List Nat := 24133 :: root244Tail51
def root244Tail49 : List Nat := 24049 :: root244Tail50
def root244Tail48 : List Nat := 23509 :: root244Tail49
def root244Tail47 : List Nat := 23269 :: root244Tail48
def root244Tail46 : List Nat := 22273 :: root244Tail47
def root244Tail45 : List Nat := 20749 :: root244Tail46
def root244Tail44 : List Nat := 20233 :: root244Tail45
def root244Tail43 : List Nat := 19603 :: root244Tail44
def root244Tail42 : List Nat := 19501 :: root244Tail43
def root244Tail41 : List Nat := 19237 :: root244Tail42
def root244Tail40 : List Nat := 19051 :: root244Tail41
def root244Tail39 : List Nat := 18439 :: root244Tail40
def root244Tail38 : List Nat := 18301 :: root244Tail39
def root244Tail37 : List Nat := 18169 :: root244Tail38
def root244Tail36 : List Nat := 17137 :: root244Tail37
def root244Tail35 : List Nat := 14431 :: root244Tail36
def root244Tail34 : List Nat := 14323 :: root244Tail35
def root244Tail33 : List Nat := 14281 :: root244Tail34
def root244Tail32 : List Nat := 13903 :: root244Tail33
def root244Tail31 : List Nat := 13807 :: root244Tail32
def root244Tail30 : List Nat := 13693 :: root244Tail31
def root244Tail29 : List Nat := 12421 :: root244Tail30
def root244Tail28 : List Nat := 12289 :: root244Tail29
def root244Tail27 : List Nat := 11497 :: root244Tail28
def root244Tail26 : List Nat := 11119 :: root244Tail27
def root244Tail25 : List Nat := 10657 :: root244Tail26
def root244Tail24 : List Nat := 10111 :: root244Tail25
def root244Tail23 : List Nat := 9733 :: root244Tail24
def root244Tail22 : List Nat := 9511 :: root244Tail23
def root244Tail21 : List Nat := 9439 :: root244Tail22
def root244Tail20 : List Nat := 9187 :: root244Tail21
def root244Tail19 : List Nat := 9091 :: root244Tail20
def root244Tail18 : List Nat := 8779 :: root244Tail19
def root244Tail17 : List Nat := 8293 :: root244Tail18
def root244Tail16 : List Nat := 7759 :: root244Tail17
def root244Tail15 : List Nat := 7027 :: root244Tail16
def root244Tail14 : List Nat := 6481 :: root244Tail15
def root244Tail13 : List Nat := 6277 :: root244Tail14
def root244Tail12 : List Nat := 5641 :: root244Tail13
def root244Tail11 : List Nat := 5431 :: root244Tail12
def root244Tail10 : List Nat := 4597 :: root244Tail11
def root244Tail9 : List Nat := 4567 :: root244Tail10
def root244Tail8 : List Nat := 4507 :: root244Tail9
def root244Tail7 : List Nat := 4441 :: root244Tail8
def root244Tail6 : List Nat := 3919 :: root244Tail7
def root244Tail5 : List Nat := 2593 :: root244Tail6
def root244Tail4 : List Nat := 2551 :: root244Tail5
def root244Tail3 : List Nat := 2539 :: root244Tail4
def root244Tail2 : List Nat := 2311 :: root244Tail3
def root244Tail1 : List Nat := 2161 :: root244Tail2
def root244Tail0 : List Nat := 1879 :: root244Tail1
def root244Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root244Candidates0_checked : candidates edge 4 1567 1879 root244Tail1 winner.bound = root244Candidates0 := by
  unfold candidates
  have row : edge 1879 = (SortedNatTree.contains FastRows.row1879) := by
    change Data.row1879.contains = (SortedNatTree.contains FastRows.row1879)
    exact FastRows.row1879_checked.symm
  rw [row]
  decide
theorem root244Step0 : scan edge (search edge 3) 4 [1567] 1567 root244Tail0 winner =
    scan edge (search edge 3) 4 [1567] 1567 root244Tail1 winner := by
  rw [root244Tail0, scan, if_neg (by decide), root244Candidates0_checked]
  change scan edge (search edge 3) 4 [1567] 1567 root244Tail1
    (if 3 ≤ root244Candidates0.length then search edge 3 [1567, 1879] root244Candidates0 3446 winner else winner) = _
  rw [if_neg (by decide)]
def root244Candidates1 : List Nat := [4597, 7027, 7759]
set_option maxRecDepth 8192 in
theorem root244Candidates1_checked : candidates edge 4 1567 2161 root244Tail2 winner.bound = root244Candidates1 := by
  unfold candidates
  have row : edge 2161 = (SortedNatTree.contains FastRows.row2161) := by
    change Data.row2161.contains = (SortedNatTree.contains FastRows.row2161)
    exact FastRows.row2161_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root244Child1 : search edge 3 [1567, 2161] root244Candidates1 3728 winner = winner := by decide
theorem root244Step1 : scan edge (search edge 3) 4 [1567] 1567 root244Tail1 winner =
    scan edge (search edge 3) 4 [1567] 1567 root244Tail2 winner := by
  rw [root244Tail1, scan, if_neg (by decide), root244Candidates1_checked]
  change scan edge (search edge 3) 4 [1567] 1567 root244Tail2
    (if 3 ≤ root244Candidates1.length then search edge 3 [1567, 2161] root244Candidates1 3728 winner else winner) = _
  rw [if_pos (by decide), root244Child1]
def root244Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root244Candidates2_checked : candidates edge 4 1567 2311 root244Tail3 winner.bound = root244Candidates2 := by
  unfold candidates
  have row : edge 2311 = (SortedNatTree.contains FastRows.row2311) := by
    change Data.row2311.contains = (SortedNatTree.contains FastRows.row2311)
    exact FastRows.row2311_checked.symm
  rw [row]
  decide
theorem root244Step2 : scan edge (search edge 3) 4 [1567] 1567 root244Tail2 winner =
    scan edge (search edge 3) 4 [1567] 1567 root244Tail3 winner := by
  rw [root244Tail2, scan, if_neg (by decide), root244Candidates2_checked]
  change scan edge (search edge 3) 4 [1567] 1567 root244Tail3
    (if 3 ≤ root244Candidates2.length then search edge 3 [1567, 2311] root244Candidates2 3878 winner else winner) = _
  rw [if_neg (by decide)]
def root244Candidates3 : List Nat := [4441, 9511]
set_option maxRecDepth 8192 in
theorem root244Candidates3_checked : candidates edge 4 1567 2539 root244Tail4 winner.bound = root244Candidates3 := by
  unfold candidates
  have row : edge 2539 = (SortedNatTree.contains FastRows.row2539) := by
    change Data.row2539.contains = (SortedNatTree.contains FastRows.row2539)
    exact FastRows.row2539_checked.symm
  rw [row]
  decide
theorem root244Step3 : scan edge (search edge 3) 4 [1567] 1567 root244Tail3 winner =
    scan edge (search edge 3) 4 [1567] 1567 root244Tail4 winner := by
  rw [root244Tail3, scan, if_neg (by decide), root244Candidates3_checked]
  change scan edge (search edge 3) 4 [1567] 1567 root244Tail4
    (if 3 ≤ root244Candidates3.length then search edge 3 [1567, 2539] root244Candidates3 4106 winner else winner) = _
  rw [if_neg (by decide)]
def root244Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root244Candidates4_checked : candidates edge 4 1567 2551 root244Tail5 winner.bound = root244Candidates4 := by
  unfold candidates
  have row : edge 2551 = (SortedNatTree.contains FastRows.row2551) := by
    change Data.row2551.contains = (SortedNatTree.contains FastRows.row2551)
    exact FastRows.row2551_checked.symm
  rw [row]
  decide
theorem root244Step4 : scan edge (search edge 3) 4 [1567] 1567 root244Tail4 winner =
    scan edge (search edge 3) 4 [1567] 1567 root244Tail5 winner := by
  rw [root244Tail4, scan, if_neg (by decide), root244Candidates4_checked]
  change scan edge (search edge 3) 4 [1567] 1567 root244Tail5
    (if 3 ≤ root244Candidates4.length then search edge 3 [1567, 2551] root244Candidates4 4118 winner else winner) = _
  rw [if_neg (by decide)]
def root244Candidates5 : List Nat := [7027, 8293, 10111, 12421]
set_option maxRecDepth 8192 in
theorem root244Candidates5_checked : candidates edge 4 1567 2593 root244Tail6 winner.bound = root244Candidates5 := by
  unfold candidates
  have row : edge 2593 = (SortedNatTree.contains FastRows.row2593) := by
    change Data.row2593.contains = (SortedNatTree.contains FastRows.row2593)
    exact FastRows.row2593_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root244Child5 : search edge 3 [1567, 2593] root244Candidates5 4160 winner = winner := by decide
theorem root244Step5 : scan edge (search edge 3) 4 [1567] 1567 root244Tail5 winner =
    scan edge (search edge 3) 4 [1567] 1567 root244Tail6 winner := by
  rw [root244Tail5, scan, if_neg (by decide), root244Candidates5_checked]
  change scan edge (search edge 3) 4 [1567] 1567 root244Tail6
    (if 3 ≤ root244Candidates5.length then search edge 3 [1567, 2593] root244Candidates5 4160 winner else winner) = _
  rw [if_pos (by decide), root244Child5]
def root244Candidates6 : List Nat := [4567]
set_option maxRecDepth 8192 in
theorem root244Candidates6_checked : candidates edge 4 1567 3919 root244Tail7 winner.bound = root244Candidates6 := by
  unfold candidates
  have row : edge 3919 = (SortedNatTree.contains FastRows.row3919) := by
    change Data.row3919.contains = (SortedNatTree.contains FastRows.row3919)
    exact FastRows.row3919_checked.symm
  rw [row]
  decide
theorem root244Step6 : scan edge (search edge 3) 4 [1567] 1567 root244Tail6 winner =
    scan edge (search edge 3) 4 [1567] 1567 root244Tail7 winner := by
  rw [root244Tail6, scan, if_neg (by decide), root244Candidates6_checked]
  change scan edge (search edge 3) 4 [1567] 1567 root244Tail7
    (if 3 ≤ root244Candidates6.length then search edge 3 [1567, 3919] root244Candidates6 5486 winner else winner) = _
  rw [if_neg (by decide)]
def root244Candidates7 : List Nat := [6481, 7759]
set_option maxRecDepth 8192 in
theorem root244Candidates7_checked : candidates edge 4 1567 4441 root244Tail8 winner.bound = root244Candidates7 := by
  unfold candidates
  have row : edge 4441 = (SortedNatTree.contains FastRows.row4441) := by
    change Data.row4441.contains = (SortedNatTree.contains FastRows.row4441)
    exact FastRows.row4441_checked.symm
  rw [row]
  decide
theorem root244Step7 : scan edge (search edge 3) 4 [1567] 1567 root244Tail7 winner =
    scan edge (search edge 3) 4 [1567] 1567 root244Tail8 winner := by
  rw [root244Tail7, scan, if_neg (by decide), root244Candidates7_checked]
  change scan edge (search edge 3) 4 [1567] 1567 root244Tail8
    (if 3 ≤ root244Candidates7.length then search edge 3 [1567, 4441] root244Candidates7 6008 winner else winner) = _
  rw [if_neg (by decide)]
def root244Candidates8 : List Nat := [7759, 9187, 12421, 13807]
set_option maxRecDepth 8192 in
theorem root244Candidates8_checked : candidates edge 4 1567 4507 root244Tail9 winner.bound = root244Candidates8 := by
  unfold candidates
  have row : edge 4507 = (SortedNatTree.contains FastRows.row4507) := by
    change Data.row4507.contains = (SortedNatTree.contains FastRows.row4507)
    exact FastRows.row4507_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root244Child8 : search edge 3 [1567, 4507] root244Candidates8 6074 winner = winner := by decide
theorem root244Step8 : scan edge (search edge 3) 4 [1567] 1567 root244Tail8 winner =
    scan edge (search edge 3) 4 [1567] 1567 root244Tail9 winner := by
  rw [root244Tail8, scan, if_neg (by decide), root244Candidates8_checked]
  change scan edge (search edge 3) 4 [1567] 1567 root244Tail9
    (if 3 ≤ root244Candidates8.length then search edge 3 [1567, 4507] root244Candidates8 6074 winner else winner) = _
  rw [if_pos (by decide), root244Child8]
def root244Candidates9 : List Nat := [9091]
set_option maxRecDepth 8192 in
theorem root244Candidates9_checked : candidates edge 4 1567 4567 root244Tail10 winner.bound = root244Candidates9 := by
  unfold candidates
  have row : edge 4567 = (SortedNatTree.contains FastRows.row4567) := by
    change Data.row4567.contains = (SortedNatTree.contains FastRows.row4567)
    exact FastRows.row4567_checked.symm
  rw [row]
  decide
theorem root244Step9 : scan edge (search edge 3) 4 [1567] 1567 root244Tail9 winner =
    scan edge (search edge 3) 4 [1567] 1567 root244Tail10 winner := by
  rw [root244Tail9, scan, if_neg (by decide), root244Candidates9_checked]
  change scan edge (search edge 3) 4 [1567] 1567 root244Tail10
    (if 3 ≤ root244Candidates9.length then search edge 3 [1567, 4567] root244Candidates9 6134 winner else winner) = _
  rw [if_neg (by decide)]
def root244Candidates10 : List Nat := [8293]
set_option maxRecDepth 8192 in
theorem root244Candidates10_checked : candidates edge 4 1567 4597 root244Tail11 winner.bound = root244Candidates10 := by
  unfold candidates
  have row : edge 4597 = (SortedNatTree.contains FastRows.row4597) := by
    change Data.row4597.contains = (SortedNatTree.contains FastRows.row4597)
    exact FastRows.row4597_checked.symm
  rw [row]
  decide
theorem root244Step10 : scan edge (search edge 3) 4 [1567] 1567 root244Tail10 winner =
    scan edge (search edge 3) 4 [1567] 1567 root244Tail11 winner := by
  rw [root244Tail10, scan, if_neg (by decide), root244Candidates10_checked]
  change scan edge (search edge 3) 4 [1567] 1567 root244Tail11
    (if 3 ≤ root244Candidates10.length then search edge 3 [1567, 4597] root244Candidates10 6164 winner else winner) = _
  rw [if_neg (by decide)]
def root244Candidates11 : List Nat := [5641]
set_option maxRecDepth 8192 in
theorem root244Candidates11_checked : candidates edge 4 1567 5431 root244Tail12 winner.bound = root244Candidates11 := by
  unfold candidates
  have row : edge 5431 = (SortedNatTree.contains FastRows.row5431) := by
    change Data.row5431.contains = (SortedNatTree.contains FastRows.row5431)
    exact FastRows.row5431_checked.symm
  rw [row]
  decide
theorem root244Step11 : scan edge (search edge 3) 4 [1567] 1567 root244Tail11 winner =
    scan edge (search edge 3) 4 [1567] 1567 root244Tail12 winner := by
  rw [root244Tail11, scan, if_neg (by decide), root244Candidates11_checked]
  change scan edge (search edge 3) 4 [1567] 1567 root244Tail12
    (if 3 ≤ root244Candidates11.length then search edge 3 [1567, 5431] root244Candidates11 6998 winner else winner) = _
  rw [if_neg (by decide)]
def root244Candidates12 : List Nat := [7027, 7759, 9091]
set_option maxRecDepth 8192 in
theorem root244Candidates12_checked : candidates edge 4 1567 5641 root244Tail13 winner.bound = root244Candidates12 := by
  unfold candidates
  have row : edge 5641 = (SortedNatTree.contains FastRows.row5641) := by
    change Data.row5641.contains = (SortedNatTree.contains FastRows.row5641)
    exact FastRows.row5641_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root244Child12 : search edge 3 [1567, 5641] root244Candidates12 7208 winner = winner := by decide
theorem root244Step12 : scan edge (search edge 3) 4 [1567] 1567 root244Tail12 winner =
    scan edge (search edge 3) 4 [1567] 1567 root244Tail13 winner := by
  rw [root244Tail12, scan, if_neg (by decide), root244Candidates12_checked]
  change scan edge (search edge 3) 4 [1567] 1567 root244Tail13
    (if 3 ≤ root244Candidates12.length then search edge 3 [1567, 5641] root244Candidates12 7208 winner else winner) = _
  rw [if_pos (by decide), root244Child12]
theorem root244Scan13 : scan edge (search edge 3) 4 [1567] 1567 root244Tail13 winner = winner := by rw [root244Tail13, scan, if_pos (by decide)]
theorem root244Scan12 : scan edge (search edge 3) 4 [1567] 1567 root244Tail12 winner = winner :=
  (root244Step12).trans root244Scan13
theorem root244Scan11 : scan edge (search edge 3) 4 [1567] 1567 root244Tail11 winner = winner :=
  (root244Step11).trans root244Scan12
theorem root244Scan10 : scan edge (search edge 3) 4 [1567] 1567 root244Tail10 winner = winner :=
  (root244Step10).trans root244Scan11
theorem root244Scan9 : scan edge (search edge 3) 4 [1567] 1567 root244Tail9 winner = winner :=
  (root244Step9).trans root244Scan10
theorem root244Scan8 : scan edge (search edge 3) 4 [1567] 1567 root244Tail8 winner = winner :=
  (root244Step8).trans root244Scan9
theorem root244Scan7 : scan edge (search edge 3) 4 [1567] 1567 root244Tail7 winner = winner :=
  (root244Step7).trans root244Scan8
theorem root244Scan6 : scan edge (search edge 3) 4 [1567] 1567 root244Tail6 winner = winner :=
  (root244Step6).trans root244Scan7
theorem root244Scan5 : scan edge (search edge 3) 4 [1567] 1567 root244Tail5 winner = winner :=
  (root244Step5).trans root244Scan6
theorem root244Scan4 : scan edge (search edge 3) 4 [1567] 1567 root244Tail4 winner = winner :=
  (root244Step4).trans root244Scan5
theorem root244Scan3 : scan edge (search edge 3) 4 [1567] 1567 root244Tail3 winner = winner :=
  (root244Step3).trans root244Scan4
theorem root244Scan2 : scan edge (search edge 3) 4 [1567] 1567 root244Tail2 winner = winner :=
  (root244Step2).trans root244Scan3
theorem root244Scan1 : scan edge (search edge 3) 4 [1567] 1567 root244Tail1 winner = winner :=
  (root244Step1).trans root244Scan2
theorem root244Scan0 : scan edge (search edge 3) 4 [1567] 1567 root244Tail0 winner = winner :=
  (root244Step0).trans root244Scan1
set_option maxRecDepth 4096 in
theorem root244Branch : search edge 4 [1567] root244Candidates 1567 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1567] 1567 root244Tail0 winner = winner
  exact root244Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
