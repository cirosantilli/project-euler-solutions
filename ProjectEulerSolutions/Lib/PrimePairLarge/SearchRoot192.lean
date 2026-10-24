import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root192Candidates : List Nat := Data.row1187
set_option maxRecDepth 8192 in
theorem root192Candidates_checked : candidates edge 5 0 1187 tail193 winner.bound = root192Candidates := by
  apply candidates_eq_row edge 5 0 1187 tail193 root192Candidates winner.bound tail193_ordered
  · decide
  · decide
  · funext q
    rfl
def root192Tail50 : List Nat := []
def root192Tail49 : List Nat := 24623 :: root192Tail50
def root192Tail48 : List Nat := 24197 :: root192Tail49
def root192Tail47 : List Nat := 23747 :: root192Tail48
def root192Tail46 : List Nat := 22907 :: root192Tail47
def root192Tail45 : List Nat := 22367 :: root192Tail46
def root192Tail44 : List Nat := 22349 :: root192Tail45
def root192Tail43 : List Nat := 21683 :: root192Tail44
def root192Tail42 : List Nat := 18959 :: root192Tail43
def root192Tail41 : List Nat := 18911 :: root192Tail42
def root192Tail40 : List Nat := 18401 :: root192Tail41
def root192Tail39 : List Nat := 16931 :: root192Tail40
def root192Tail38 : List Nat := 16889 :: root192Tail39
def root192Tail37 : List Nat := 15809 :: root192Tail38
def root192Tail36 : List Nat := 15803 :: root192Tail37
def root192Tail35 : List Nat := 15383 :: root192Tail36
def root192Tail34 : List Nat := 15263 :: root192Tail35
def root192Tail33 : List Nat := 14969 :: root192Tail34
def root192Tail32 : List Nat := 14243 :: root192Tail33
def root192Tail31 : List Nat := 14051 :: root192Tail32
def root192Tail30 : List Nat := 13781 :: root192Tail31
def root192Tail29 : List Nat := 12611 :: root192Tail30
def root192Tail28 : List Nat := 12569 :: root192Tail29
def root192Tail27 : List Nat := 12413 :: root192Tail28
def root192Tail26 : List Nat := 12239 :: root192Tail27
def root192Tail25 : List Nat := 11909 :: root192Tail26
def root192Tail24 : List Nat := 10733 :: root192Tail25
def root192Tail23 : List Nat := 9839 :: root192Tail24
def root192Tail22 : List Nat := 9341 :: root192Tail23
def root192Tail21 : List Nat := 9221 :: root192Tail22
def root192Tail20 : List Nat := 9173 :: root192Tail21
def root192Tail19 : List Nat := 8963 :: root192Tail20
def root192Tail18 : List Nat := 8117 :: root192Tail19
def root192Tail17 : List Nat := 8093 :: root192Tail18
def root192Tail16 : List Nat := 7253 :: root192Tail17
def root192Tail15 : List Nat := 7079 :: root192Tail16
def root192Tail14 : List Nat := 6983 :: root192Tail15
def root192Tail13 : List Nat := 6857 :: root192Tail14
def root192Tail12 : List Nat := 6833 :: root192Tail13
def root192Tail11 : List Nat := 6701 :: root192Tail12
def root192Tail10 : List Nat := 6569 :: root192Tail11
def root192Tail9 : List Nat := 6203 :: root192Tail10
def root192Tail8 : List Nat := 5861 :: root192Tail9
def root192Tail7 : List Nat := 5477 :: root192Tail8
def root192Tail6 : List Nat := 4937 :: root192Tail7
def root192Tail5 : List Nat := 4649 :: root192Tail6
def root192Tail4 : List Nat := 4523 :: root192Tail5
def root192Tail3 : List Nat := 4001 :: root192Tail4
def root192Tail2 : List Nat := 2879 :: root192Tail3
def root192Tail1 : List Nat := 1451 :: root192Tail2
def root192Tail0 : List Nat := 1361 :: root192Tail1
def root192Candidates0 : List Nat := [9341, 14243, 14969, 18911]
set_option maxRecDepth 8192 in
theorem root192Candidates0_checked : candidates edge 4 1187 1361 root192Tail1 winner.bound = root192Candidates0 := by
  unfold candidates
  have row : edge 1361 = (SortedNatTree.contains FastRows.row1361) := by
    change Data.row1361.contains = (SortedNatTree.contains FastRows.row1361)
    exact FastRows.row1361_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root192Child0 : search edge 3 [1187, 1361] root192Candidates0 2548 winner = winner := by decide
theorem root192Step0 : scan edge (search edge 3) 4 [1187] 1187 root192Tail0 winner =
    scan edge (search edge 3) 4 [1187] 1187 root192Tail1 winner := by
  rw [root192Tail0, scan, if_neg (by decide), root192Candidates0_checked]
  change scan edge (search edge 3) 4 [1187] 1187 root192Tail1
    (if 3 ≤ root192Candidates0.length then search edge 3 [1187, 1361] root192Candidates0 2548 winner else winner) = _
  rw [if_pos (by decide), root192Child0]
def root192Candidates1 : List Nat := [14243]
set_option maxRecDepth 8192 in
theorem root192Candidates1_checked : candidates edge 4 1187 1451 root192Tail2 winner.bound = root192Candidates1 := by
  unfold candidates
  have row : edge 1451 = (SortedNatTree.contains FastRows.row1451) := by
    change Data.row1451.contains = (SortedNatTree.contains FastRows.row1451)
    exact FastRows.row1451_checked.symm
  rw [row]
  decide
theorem root192Step1 : scan edge (search edge 3) 4 [1187] 1187 root192Tail1 winner =
    scan edge (search edge 3) 4 [1187] 1187 root192Tail2 winner := by
  rw [root192Tail1, scan, if_neg (by decide), root192Candidates1_checked]
  change scan edge (search edge 3) 4 [1187] 1187 root192Tail2
    (if 3 ≤ root192Candidates1.length then search edge 3 [1187, 1451] root192Candidates1 2638 winner else winner) = _
  rw [if_neg (by decide)]
def root192Candidates2 : List Nat := [6701]
set_option maxRecDepth 8192 in
theorem root192Candidates2_checked : candidates edge 4 1187 2879 root192Tail3 winner.bound = root192Candidates2 := by
  unfold candidates
  have row : edge 2879 = (SortedNatTree.contains FastRows.row2879) := by
    change Data.row2879.contains = (SortedNatTree.contains FastRows.row2879)
    exact FastRows.row2879_checked.symm
  rw [row]
  decide
theorem root192Step2 : scan edge (search edge 3) 4 [1187] 1187 root192Tail2 winner =
    scan edge (search edge 3) 4 [1187] 1187 root192Tail3 winner := by
  rw [root192Tail2, scan, if_neg (by decide), root192Candidates2_checked]
  change scan edge (search edge 3) 4 [1187] 1187 root192Tail3
    (if 3 ≤ root192Candidates2.length then search edge 3 [1187, 2879] root192Candidates2 4066 winner else winner) = _
  rw [if_neg (by decide)]
def root192Candidates3 : List Nat := [12569, 16889]
set_option maxRecDepth 8192 in
theorem root192Candidates3_checked : candidates edge 4 1187 4001 root192Tail4 winner.bound = root192Candidates3 := by
  unfold candidates
  have row : edge 4001 = (SortedNatTree.contains FastRows.row4001) := by
    change Data.row4001.contains = (SortedNatTree.contains FastRows.row4001)
    exact FastRows.row4001_checked.symm
  rw [row]
  decide
theorem root192Step3 : scan edge (search edge 3) 4 [1187] 1187 root192Tail3 winner =
    scan edge (search edge 3) 4 [1187] 1187 root192Tail4 winner := by
  rw [root192Tail3, scan, if_neg (by decide), root192Candidates3_checked]
  change scan edge (search edge 3) 4 [1187] 1187 root192Tail4
    (if 3 ≤ root192Candidates3.length then search edge 3 [1187, 4001] root192Candidates3 5188 winner else winner) = _
  rw [if_neg (by decide)]
def root192Candidates4 : List Nat := [8093, 9221, 12569, 12611]
set_option maxRecDepth 8192 in
theorem root192Candidates4_checked : candidates edge 4 1187 4523 root192Tail5 winner.bound = root192Candidates4 := by
  unfold candidates
  have row : edge 4523 = (SortedNatTree.contains FastRows.row4523) := by
    change Data.row4523.contains = (SortedNatTree.contains FastRows.row4523)
    exact FastRows.row4523_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root192Child4 : search edge 3 [1187, 4523] root192Candidates4 5710 winner = winner := by decide
theorem root192Step4 : scan edge (search edge 3) 4 [1187] 1187 root192Tail4 winner =
    scan edge (search edge 3) 4 [1187] 1187 root192Tail5 winner := by
  rw [root192Tail4, scan, if_neg (by decide), root192Candidates4_checked]
  change scan edge (search edge 3) 4 [1187] 1187 root192Tail5
    (if 3 ≤ root192Candidates4.length then search edge 3 [1187, 4523] root192Candidates4 5710 winner else winner) = _
  rw [if_pos (by decide), root192Child4]
def root192Candidates5 : List Nat := [8117, 9221]
set_option maxRecDepth 8192 in
theorem root192Candidates5_checked : candidates edge 4 1187 4649 root192Tail6 winner.bound = root192Candidates5 := by
  unfold candidates
  have row : edge 4649 = (SortedNatTree.contains FastRows.row4649) := by
    change Data.row4649.contains = (SortedNatTree.contains FastRows.row4649)
    exact FastRows.row4649_checked.symm
  rw [row]
  decide
theorem root192Step5 : scan edge (search edge 3) 4 [1187] 1187 root192Tail5 winner =
    scan edge (search edge 3) 4 [1187] 1187 root192Tail6 winner := by
  rw [root192Tail5, scan, if_neg (by decide), root192Candidates5_checked]
  change scan edge (search edge 3) 4 [1187] 1187 root192Tail6
    (if 3 ≤ root192Candidates5.length then search edge 3 [1187, 4649] root192Candidates5 5836 winner else winner) = _
  rw [if_neg (by decide)]
def root192Candidates6 : List Nat := [9221]
set_option maxRecDepth 8192 in
theorem root192Candidates6_checked : candidates edge 4 1187 4937 root192Tail7 winner.bound = root192Candidates6 := by
  unfold candidates
  have row : edge 4937 = (SortedNatTree.contains FastRows.row4937) := by
    change Data.row4937.contains = (SortedNatTree.contains FastRows.row4937)
    exact FastRows.row4937_checked.symm
  rw [row]
  decide
theorem root192Step6 : scan edge (search edge 3) 4 [1187] 1187 root192Tail6 winner =
    scan edge (search edge 3) 4 [1187] 1187 root192Tail7 winner := by
  rw [root192Tail6, scan, if_neg (by decide), root192Candidates6_checked]
  change scan edge (search edge 3) 4 [1187] 1187 root192Tail7
    (if 3 ≤ root192Candidates6.length then search edge 3 [1187, 4937] root192Candidates6 6124 winner else winner) = _
  rw [if_neg (by decide)]
def root192Candidates7 : List Nat := [9839, 15263, 15803]
set_option maxRecDepth 8192 in
theorem root192Candidates7_checked : candidates edge 4 1187 5477 root192Tail8 winner.bound = root192Candidates7 := by
  unfold candidates
  have row : edge 5477 = (SortedNatTree.contains FastRows.row5477) := by
    change Data.row5477.contains = (SortedNatTree.contains FastRows.row5477)
    exact FastRows.row5477_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root192Child7 : search edge 3 [1187, 5477] root192Candidates7 6664 winner = winner := by decide
theorem root192Step7 : scan edge (search edge 3) 4 [1187] 1187 root192Tail7 winner =
    scan edge (search edge 3) 4 [1187] 1187 root192Tail8 winner := by
  rw [root192Tail7, scan, if_neg (by decide), root192Candidates7_checked]
  change scan edge (search edge 3) 4 [1187] 1187 root192Tail8
    (if 3 ≤ root192Candidates7.length then search edge 3 [1187, 5477] root192Candidates7 6664 winner else winner) = _
  rw [if_pos (by decide), root192Child7]
def root192Candidates8 : List Nat := [9221]
set_option maxRecDepth 8192 in
theorem root192Candidates8_checked : candidates edge 4 1187 5861 root192Tail9 winner.bound = root192Candidates8 := by
  unfold candidates
  have row : edge 5861 = (SortedNatTree.contains FastRows.row5861) := by
    change Data.row5861.contains = (SortedNatTree.contains FastRows.row5861)
    exact FastRows.row5861_checked.symm
  rw [row]
  decide
theorem root192Step8 : scan edge (search edge 3) 4 [1187] 1187 root192Tail8 winner =
    scan edge (search edge 3) 4 [1187] 1187 root192Tail9 winner := by
  rw [root192Tail8, scan, if_neg (by decide), root192Candidates8_checked]
  change scan edge (search edge 3) 4 [1187] 1187 root192Tail9
    (if 3 ≤ root192Candidates8.length then search edge 3 [1187, 5861] root192Candidates8 7048 winner else winner) = _
  rw [if_neg (by decide)]
def root192Candidates9 : List Nat := [7253]
set_option maxRecDepth 8192 in
theorem root192Candidates9_checked : candidates edge 4 1187 6203 root192Tail10 winner.bound = root192Candidates9 := by
  unfold candidates
  have row : edge 6203 = (SortedNatTree.contains FastRows.row6203) := by
    change Data.row6203.contains = (SortedNatTree.contains FastRows.row6203)
    exact FastRows.row6203_checked.symm
  rw [row]
  decide
theorem root192Step9 : scan edge (search edge 3) 4 [1187] 1187 root192Tail9 winner =
    scan edge (search edge 3) 4 [1187] 1187 root192Tail10 winner := by
  rw [root192Tail9, scan, if_neg (by decide), root192Candidates9_checked]
  change scan edge (search edge 3) 4 [1187] 1187 root192Tail10
    (if 3 ≤ root192Candidates9.length then search edge 3 [1187, 6203] root192Candidates9 7390 winner else winner) = _
  rw [if_neg (by decide)]
theorem root192Scan10 : scan edge (search edge 3) 4 [1187] 1187 root192Tail10 winner = winner := by rw [root192Tail10, scan, if_pos (by decide)]
theorem root192Scan9 : scan edge (search edge 3) 4 [1187] 1187 root192Tail9 winner = winner :=
  (root192Step9).trans root192Scan10
theorem root192Scan8 : scan edge (search edge 3) 4 [1187] 1187 root192Tail8 winner = winner :=
  (root192Step8).trans root192Scan9
theorem root192Scan7 : scan edge (search edge 3) 4 [1187] 1187 root192Tail7 winner = winner :=
  (root192Step7).trans root192Scan8
theorem root192Scan6 : scan edge (search edge 3) 4 [1187] 1187 root192Tail6 winner = winner :=
  (root192Step6).trans root192Scan7
theorem root192Scan5 : scan edge (search edge 3) 4 [1187] 1187 root192Tail5 winner = winner :=
  (root192Step5).trans root192Scan6
theorem root192Scan4 : scan edge (search edge 3) 4 [1187] 1187 root192Tail4 winner = winner :=
  (root192Step4).trans root192Scan5
theorem root192Scan3 : scan edge (search edge 3) 4 [1187] 1187 root192Tail3 winner = winner :=
  (root192Step3).trans root192Scan4
theorem root192Scan2 : scan edge (search edge 3) 4 [1187] 1187 root192Tail2 winner = winner :=
  (root192Step2).trans root192Scan3
theorem root192Scan1 : scan edge (search edge 3) 4 [1187] 1187 root192Tail1 winner = winner :=
  (root192Step1).trans root192Scan2
theorem root192Scan0 : scan edge (search edge 3) 4 [1187] 1187 root192Tail0 winner = winner :=
  (root192Step0).trans root192Scan1
set_option maxRecDepth 4096 in
theorem root192Branch : search edge 4 [1187] root192Candidates 1187 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1187] 1187 root192Tail0 winner = winner
  exact root192Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
