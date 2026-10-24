import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root251Candidates : List Nat := Data.row1609
set_option maxRecDepth 8192 in
theorem root251Candidates_checked : candidates edge 5 0 1609 tail252 winner.bound = root251Candidates := by
  apply candidates_eq_row edge 5 0 1609 tail252 root251Candidates winner.bound tail252_ordered
  · decide
  · decide
  · funext q
    rfl
def root251Tail53 : List Nat := []
def root251Tail52 : List Nat := 23773 :: root251Tail53
def root251Tail51 : List Nat := 23761 :: root251Tail52
def root251Tail50 : List Nat := 23041 :: root251Tail51
def root251Tail49 : List Nat := 21067 :: root251Tail50
def root251Tail48 : List Nat := 20887 :: root251Tail49
def root251Tail47 : List Nat := 20173 :: root251Tail48
def root251Tail46 : List Nat := 19597 :: root251Tail47
def root251Tail45 : List Nat := 19309 :: root251Tail46
def root251Tail44 : List Nat := 19009 :: root251Tail45
def root251Tail43 : List Nat := 18757 :: root251Tail44
def root251Tail42 : List Nat := 18397 :: root251Tail43
def root251Tail41 : List Nat := 18211 :: root251Tail42
def root251Tail40 : List Nat := 17881 :: root251Tail41
def root251Tail39 : List Nat := 17077 :: root251Tail40
def root251Tail38 : List Nat := 16843 :: root251Tail39
def root251Tail37 : List Nat := 16603 :: root251Tail38
def root251Tail36 : List Nat := 15901 :: root251Tail37
def root251Tail35 : List Nat := 15817 :: root251Tail36
def root251Tail34 : List Nat := 14437 :: root251Tail35
def root251Tail33 : List Nat := 14293 :: root251Tail34
def root251Tail32 : List Nat := 13693 :: root251Tail33
def root251Tail31 : List Nat := 13399 :: root251Tail32
def root251Tail30 : List Nat := 12421 :: root251Tail31
def root251Tail29 : List Nat := 12043 :: root251Tail30
def root251Tail28 : List Nat := 11677 :: root251Tail29
def root251Tail27 : List Nat := 10861 :: root251Tail28
def root251Tail26 : List Nat := 9883 :: root251Tail27
def root251Tail25 : List Nat := 9679 :: root251Tail26
def root251Tail24 : List Nat := 9001 :: root251Tail25
def root251Tail23 : List Nat := 8839 :: root251Tail24
def root251Tail22 : List Nat := 8287 :: root251Tail23
def root251Tail21 : List Nat := 7603 :: root251Tail22
def root251Tail20 : List Nat := 7243 :: root251Tail21
def root251Tail19 : List Nat := 6991 :: root251Tail20
def root251Tail18 : List Nat := 5647 :: root251Tail19
def root251Tail17 : List Nat := 5179 :: root251Tail18
def root251Tail16 : List Nat := 5011 :: root251Tail17
def root251Tail15 : List Nat := 4723 :: root251Tail16
def root251Tail14 : List Nat := 4621 :: root251Tail15
def root251Tail13 : List Nat := 4201 :: root251Tail14
def root251Tail12 : List Nat := 4129 :: root251Tail13
def root251Tail11 : List Nat := 3967 :: root251Tail12
def root251Tail10 : List Nat := 3793 :: root251Tail11
def root251Tail9 : List Nat := 3769 :: root251Tail10
def root251Tail8 : List Nat := 3697 :: root251Tail9
def root251Tail7 : List Nat := 3673 :: root251Tail8
def root251Tail6 : List Nat := 3613 :: root251Tail7
def root251Tail5 : List Nat := 2833 :: root251Tail6
def root251Tail4 : List Nat := 2803 :: root251Tail5
def root251Tail3 : List Nat := 2467 :: root251Tail4
def root251Tail2 : List Nat := 2161 :: root251Tail3
def root251Tail1 : List Nat := 2017 :: root251Tail2
def root251Tail0 : List Nat := 1783 :: root251Tail1
def root251Candidates0 : List Nat := [3769, 5011, 14293]
set_option maxRecDepth 8192 in
theorem root251Candidates0_checked : candidates edge 4 1609 1783 root251Tail1 winner.bound = root251Candidates0 := by
  unfold candidates
  have row : edge 1783 = (SortedNatTree.contains FastRows.row1783) := by
    change Data.row1783.contains = (SortedNatTree.contains FastRows.row1783)
    exact FastRows.row1783_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root251Child0 : search edge 3 [1609, 1783] root251Candidates0 3392 winner = winner := by decide
theorem root251Step0 : scan edge (search edge 3) 4 [1609] 1609 root251Tail0 winner =
    scan edge (search edge 3) 4 [1609] 1609 root251Tail1 winner := by
  rw [root251Tail0, scan, if_neg (by decide), root251Candidates0_checked]
  change scan edge (search edge 3) 4 [1609] 1609 root251Tail1
    (if 3 ≤ root251Candidates0.length then search edge 3 [1609, 1783] root251Candidates0 3392 winner else winner) = _
  rw [if_pos (by decide), root251Child0]
def root251Candidates1 : List Nat := [4621, 7603, 9001]
set_option maxRecDepth 8192 in
theorem root251Candidates1_checked : candidates edge 4 1609 2017 root251Tail2 winner.bound = root251Candidates1 := by
  unfold candidates
  have row : edge 2017 = (SortedNatTree.contains FastRows.row2017) := by
    change Data.row2017.contains = (SortedNatTree.contains FastRows.row2017)
    exact FastRows.row2017_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root251Child1 : search edge 3 [1609, 2017] root251Candidates1 3626 winner = winner := by decide
theorem root251Step1 : scan edge (search edge 3) 4 [1609] 1609 root251Tail1 winner =
    scan edge (search edge 3) 4 [1609] 1609 root251Tail2 winner := by
  rw [root251Tail1, scan, if_neg (by decide), root251Candidates1_checked]
  change scan edge (search edge 3) 4 [1609] 1609 root251Tail2
    (if 3 ≤ root251Candidates1.length then search edge 3 [1609, 2017] root251Candidates1 3626 winner else winner) = _
  rw [if_pos (by decide), root251Child1]
def root251Candidates2 : List Nat := [9883]
set_option maxRecDepth 8192 in
theorem root251Candidates2_checked : candidates edge 4 1609 2161 root251Tail3 winner.bound = root251Candidates2 := by
  unfold candidates
  have row : edge 2161 = (SortedNatTree.contains FastRows.row2161) := by
    change Data.row2161.contains = (SortedNatTree.contains FastRows.row2161)
    exact FastRows.row2161_checked.symm
  rw [row]
  decide
theorem root251Step2 : scan edge (search edge 3) 4 [1609] 1609 root251Tail2 winner =
    scan edge (search edge 3) 4 [1609] 1609 root251Tail3 winner := by
  rw [root251Tail2, scan, if_neg (by decide), root251Candidates2_checked]
  change scan edge (search edge 3) 4 [1609] 1609 root251Tail3
    (if 3 ≤ root251Candidates2.length then search edge 3 [1609, 2161] root251Candidates2 3770 winner else winner) = _
  rw [if_neg (by decide)]
def root251Candidates3 : List Nat := [4723, 10861, 14437]
set_option maxRecDepth 8192 in
theorem root251Candidates3_checked : candidates edge 4 1609 2467 root251Tail4 winner.bound = root251Candidates3 := by
  unfold candidates
  have row : edge 2467 = (SortedNatTree.contains FastRows.row2467) := by
    change Data.row2467.contains = (SortedNatTree.contains FastRows.row2467)
    exact FastRows.row2467_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root251Child3 : search edge 3 [1609, 2467] root251Candidates3 4076 winner = winner := by decide
theorem root251Step3 : scan edge (search edge 3) 4 [1609] 1609 root251Tail3 winner =
    scan edge (search edge 3) 4 [1609] 1609 root251Tail4 winner := by
  rw [root251Tail3, scan, if_neg (by decide), root251Candidates3_checked]
  change scan edge (search edge 3) 4 [1609] 1609 root251Tail4
    (if 3 ≤ root251Candidates3.length then search edge 3 [1609, 2467] root251Candidates3 4076 winner else winner) = _
  rw [if_pos (by decide), root251Child3]
def root251Candidates4 : List Nat := [3769, 9883]
set_option maxRecDepth 8192 in
theorem root251Candidates4_checked : candidates edge 4 1609 2803 root251Tail5 winner.bound = root251Candidates4 := by
  unfold candidates
  have row : edge 2803 = (SortedNatTree.contains FastRows.row2803) := by
    change Data.row2803.contains = (SortedNatTree.contains FastRows.row2803)
    exact FastRows.row2803_checked.symm
  rw [row]
  decide
theorem root251Step4 : scan edge (search edge 3) 4 [1609] 1609 root251Tail4 winner =
    scan edge (search edge 3) 4 [1609] 1609 root251Tail5 winner := by
  rw [root251Tail4, scan, if_neg (by decide), root251Candidates4_checked]
  change scan edge (search edge 3) 4 [1609] 1609 root251Tail5
    (if 3 ≤ root251Candidates4.length then search edge 3 [1609, 2803] root251Candidates4 4412 winner else winner) = _
  rw [if_neg (by decide)]
def root251Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root251Candidates5_checked : candidates edge 4 1609 2833 root251Tail6 winner.bound = root251Candidates5 := by
  unfold candidates
  have row : edge 2833 = (SortedNatTree.contains FastRows.row2833) := by
    change Data.row2833.contains = (SortedNatTree.contains FastRows.row2833)
    exact FastRows.row2833_checked.symm
  rw [row]
  decide
theorem root251Step5 : scan edge (search edge 3) 4 [1609] 1609 root251Tail5 winner =
    scan edge (search edge 3) 4 [1609] 1609 root251Tail6 winner := by
  rw [root251Tail5, scan, if_neg (by decide), root251Candidates5_checked]
  change scan edge (search edge 3) 4 [1609] 1609 root251Tail6
    (if 3 ≤ root251Candidates5.length then search edge 3 [1609, 2833] root251Candidates5 4442 winner else winner) = _
  rw [if_neg (by decide)]
def root251Candidates6 : List Nat := [3697, 14293]
set_option maxRecDepth 8192 in
theorem root251Candidates6_checked : candidates edge 4 1609 3613 root251Tail7 winner.bound = root251Candidates6 := by
  unfold candidates
  have row : edge 3613 = (SortedNatTree.contains FastRows.row3613) := by
    change Data.row3613.contains = (SortedNatTree.contains FastRows.row3613)
    exact FastRows.row3613_checked.symm
  rw [row]
  decide
theorem root251Step6 : scan edge (search edge 3) 4 [1609] 1609 root251Tail6 winner =
    scan edge (search edge 3) 4 [1609] 1609 root251Tail7 winner := by
  rw [root251Tail6, scan, if_neg (by decide), root251Candidates6_checked]
  change scan edge (search edge 3) 4 [1609] 1609 root251Tail7
    (if 3 ≤ root251Candidates6.length then search edge 3 [1609, 3613] root251Candidates6 5222 winner else winner) = _
  rw [if_neg (by decide)]
def root251Candidates7 : List Nat := [4129]
set_option maxRecDepth 8192 in
theorem root251Candidates7_checked : candidates edge 4 1609 3673 root251Tail8 winner.bound = root251Candidates7 := by
  unfold candidates
  have row : edge 3673 = (SortedNatTree.contains FastRows.row3673) := by
    change Data.row3673.contains = (SortedNatTree.contains FastRows.row3673)
    exact FastRows.row3673_checked.symm
  rw [row]
  decide
theorem root251Step7 : scan edge (search edge 3) 4 [1609] 1609 root251Tail7 winner =
    scan edge (search edge 3) 4 [1609] 1609 root251Tail8 winner := by
  rw [root251Tail7, scan, if_neg (by decide), root251Candidates7_checked]
  change scan edge (search edge 3) 4 [1609] 1609 root251Tail8
    (if 3 ≤ root251Candidates7.length then search edge 3 [1609, 3673] root251Candidates7 5282 winner else winner) = _
  rw [if_neg (by decide)]
def root251Candidates8 : List Nat := [16843, 18211]
set_option maxRecDepth 8192 in
theorem root251Candidates8_checked : candidates edge 4 1609 3697 root251Tail9 winner.bound = root251Candidates8 := by
  unfold candidates
  have row : edge 3697 = (SortedNatTree.contains FastRows.row3697) := by
    change Data.row3697.contains = (SortedNatTree.contains FastRows.row3697)
    exact FastRows.row3697_checked.symm
  rw [row]
  decide
theorem root251Step8 : scan edge (search edge 3) 4 [1609] 1609 root251Tail8 winner =
    scan edge (search edge 3) 4 [1609] 1609 root251Tail9 winner := by
  rw [root251Tail8, scan, if_neg (by decide), root251Candidates8_checked]
  change scan edge (search edge 3) 4 [1609] 1609 root251Tail9
    (if 3 ≤ root251Candidates8.length then search edge 3 [1609, 3697] root251Candidates8 5306 winner else winner) = _
  rw [if_neg (by decide)]
def root251Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root251Candidates9_checked : candidates edge 4 1609 3769 root251Tail10 winner.bound = root251Candidates9 := by
  unfold candidates
  have row : edge 3769 = (SortedNatTree.contains FastRows.row3769) := by
    change Data.row3769.contains = (SortedNatTree.contains FastRows.row3769)
    exact FastRows.row3769_checked.symm
  rw [row]
  decide
theorem root251Step9 : scan edge (search edge 3) 4 [1609] 1609 root251Tail9 winner =
    scan edge (search edge 3) 4 [1609] 1609 root251Tail10 winner := by
  rw [root251Tail9, scan, if_neg (by decide), root251Candidates9_checked]
  change scan edge (search edge 3) 4 [1609] 1609 root251Tail10
    (if 3 ≤ root251Candidates9.length then search edge 3 [1609, 3769] root251Candidates9 5378 winner else winner) = _
  rw [if_neg (by decide)]
def root251Candidates10 : List Nat := [5179]
set_option maxRecDepth 8192 in
theorem root251Candidates10_checked : candidates edge 4 1609 3793 root251Tail11 winner.bound = root251Candidates10 := by
  unfold candidates
  have row : edge 3793 = (SortedNatTree.contains FastRows.row3793) := by
    change Data.row3793.contains = (SortedNatTree.contains FastRows.row3793)
    exact FastRows.row3793_checked.symm
  rw [row]
  decide
theorem root251Step10 : scan edge (search edge 3) 4 [1609] 1609 root251Tail10 winner =
    scan edge (search edge 3) 4 [1609] 1609 root251Tail11 winner := by
  rw [root251Tail10, scan, if_neg (by decide), root251Candidates10_checked]
  change scan edge (search edge 3) 4 [1609] 1609 root251Tail11
    (if 3 ≤ root251Candidates10.length then search edge 3 [1609, 3793] root251Candidates10 5402 winner else winner) = _
  rw [if_neg (by decide)]
def root251Candidates11 : List Nat := [5011]
set_option maxRecDepth 8192 in
theorem root251Candidates11_checked : candidates edge 4 1609 3967 root251Tail12 winner.bound = root251Candidates11 := by
  unfold candidates
  have row : edge 3967 = (SortedNatTree.contains FastRows.row3967) := by
    change Data.row3967.contains = (SortedNatTree.contains FastRows.row3967)
    exact FastRows.row3967_checked.symm
  rw [row]
  decide
theorem root251Step11 : scan edge (search edge 3) 4 [1609] 1609 root251Tail11 winner =
    scan edge (search edge 3) 4 [1609] 1609 root251Tail12 winner := by
  rw [root251Tail11, scan, if_neg (by decide), root251Candidates11_checked]
  change scan edge (search edge 3) 4 [1609] 1609 root251Tail12
    (if 3 ≤ root251Candidates11.length then search edge 3 [1609, 3967] root251Candidates11 5576 winner else winner) = _
  rw [if_neg (by decide)]
def root251Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root251Candidates12_checked : candidates edge 4 1609 4129 root251Tail13 winner.bound = root251Candidates12 := by
  unfold candidates
  have row : edge 4129 = (SortedNatTree.contains FastRows.row4129) := by
    change Data.row4129.contains = (SortedNatTree.contains FastRows.row4129)
    exact FastRows.row4129_checked.symm
  rw [row]
  decide
theorem root251Step12 : scan edge (search edge 3) 4 [1609] 1609 root251Tail12 winner =
    scan edge (search edge 3) 4 [1609] 1609 root251Tail13 winner := by
  rw [root251Tail12, scan, if_neg (by decide), root251Candidates12_checked]
  change scan edge (search edge 3) 4 [1609] 1609 root251Tail13
    (if 3 ≤ root251Candidates12.length then search edge 3 [1609, 4129] root251Candidates12 5738 winner else winner) = _
  rw [if_neg (by decide)]
def root251Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root251Candidates13_checked : candidates edge 4 1609 4201 root251Tail14 winner.bound = root251Candidates13 := by
  unfold candidates
  have row : edge 4201 = (SortedNatTree.contains FastRows.row4201) := by
    change Data.row4201.contains = (SortedNatTree.contains FastRows.row4201)
    exact FastRows.row4201_checked.symm
  rw [row]
  decide
theorem root251Step13 : scan edge (search edge 3) 4 [1609] 1609 root251Tail13 winner =
    scan edge (search edge 3) 4 [1609] 1609 root251Tail14 winner := by
  rw [root251Tail13, scan, if_neg (by decide), root251Candidates13_checked]
  change scan edge (search edge 3) 4 [1609] 1609 root251Tail14
    (if 3 ≤ root251Candidates13.length then search edge 3 [1609, 4201] root251Candidates13 5810 winner else winner) = _
  rw [if_neg (by decide)]
def root251Candidates14 : List Nat := [8287, 12421]
set_option maxRecDepth 8192 in
theorem root251Candidates14_checked : candidates edge 4 1609 4621 root251Tail15 winner.bound = root251Candidates14 := by
  unfold candidates
  have row : edge 4621 = (SortedNatTree.contains FastRows.row4621) := by
    change Data.row4621.contains = (SortedNatTree.contains FastRows.row4621)
    exact FastRows.row4621_checked.symm
  rw [row]
  decide
theorem root251Step14 : scan edge (search edge 3) 4 [1609] 1609 root251Tail14 winner =
    scan edge (search edge 3) 4 [1609] 1609 root251Tail15 winner := by
  rw [root251Tail14, scan, if_neg (by decide), root251Candidates14_checked]
  change scan edge (search edge 3) 4 [1609] 1609 root251Tail15
    (if 3 ≤ root251Candidates14.length then search edge 3 [1609, 4621] root251Candidates14 6230 winner else winner) = _
  rw [if_neg (by decide)]
def root251Candidates15 : List Nat := [7243, 8839]
set_option maxRecDepth 8192 in
theorem root251Candidates15_checked : candidates edge 4 1609 4723 root251Tail16 winner.bound = root251Candidates15 := by
  unfold candidates
  have row : edge 4723 = (SortedNatTree.contains FastRows.row4723) := by
    change Data.row4723.contains = (SortedNatTree.contains FastRows.row4723)
    exact FastRows.row4723_checked.symm
  rw [row]
  decide
theorem root251Step15 : scan edge (search edge 3) 4 [1609] 1609 root251Tail15 winner =
    scan edge (search edge 3) 4 [1609] 1609 root251Tail16 winner := by
  rw [root251Tail15, scan, if_neg (by decide), root251Candidates15_checked]
  change scan edge (search edge 3) 4 [1609] 1609 root251Tail16
    (if 3 ≤ root251Candidates15.length then search edge 3 [1609, 4723] root251Candidates15 6332 winner else winner) = _
  rw [if_neg (by decide)]
def root251Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root251Candidates16_checked : candidates edge 4 1609 5011 root251Tail17 winner.bound = root251Candidates16 := by
  unfold candidates
  have row : edge 5011 = (SortedNatTree.contains FastRows.row5011) := by
    change Data.row5011.contains = (SortedNatTree.contains FastRows.row5011)
    exact FastRows.row5011_checked.symm
  rw [row]
  decide
theorem root251Step16 : scan edge (search edge 3) 4 [1609] 1609 root251Tail16 winner =
    scan edge (search edge 3) 4 [1609] 1609 root251Tail17 winner := by
  rw [root251Tail16, scan, if_neg (by decide), root251Candidates16_checked]
  change scan edge (search edge 3) 4 [1609] 1609 root251Tail17
    (if 3 ≤ root251Candidates16.length then search edge 3 [1609, 5011] root251Candidates16 6620 winner else winner) = _
  rw [if_neg (by decide)]
def root251Candidates17 : List Nat := [19009]
set_option maxRecDepth 8192 in
theorem root251Candidates17_checked : candidates edge 4 1609 5179 root251Tail18 winner.bound = root251Candidates17 := by
  unfold candidates
  have row : edge 5179 = (SortedNatTree.contains FastRows.row5179) := by
    change Data.row5179.contains = (SortedNatTree.contains FastRows.row5179)
    exact FastRows.row5179_checked.symm
  rw [row]
  decide
theorem root251Step17 : scan edge (search edge 3) 4 [1609] 1609 root251Tail17 winner =
    scan edge (search edge 3) 4 [1609] 1609 root251Tail18 winner := by
  rw [root251Tail17, scan, if_neg (by decide), root251Candidates17_checked]
  change scan edge (search edge 3) 4 [1609] 1609 root251Tail18
    (if 3 ≤ root251Candidates17.length then search edge 3 [1609, 5179] root251Candidates17 6788 winner else winner) = _
  rw [if_neg (by decide)]
def root251Candidates18 : List Nat := [18397]
set_option maxRecDepth 8192 in
theorem root251Candidates18_checked : candidates edge 4 1609 5647 root251Tail19 winner.bound = root251Candidates18 := by
  unfold candidates
  have row : edge 5647 = (SortedNatTree.contains FastRows.row5647) := by
    change Data.row5647.contains = (SortedNatTree.contains FastRows.row5647)
    exact FastRows.row5647_checked.symm
  rw [row]
  decide
theorem root251Step18 : scan edge (search edge 3) 4 [1609] 1609 root251Tail18 winner =
    scan edge (search edge 3) 4 [1609] 1609 root251Tail19 winner := by
  rw [root251Tail18, scan, if_neg (by decide), root251Candidates18_checked]
  change scan edge (search edge 3) 4 [1609] 1609 root251Tail19
    (if 3 ≤ root251Candidates18.length then search edge 3 [1609, 5647] root251Candidates18 7256 winner else winner) = _
  rw [if_neg (by decide)]
theorem root251Scan19 : scan edge (search edge 3) 4 [1609] 1609 root251Tail19 winner = winner := by rw [root251Tail19, scan, if_pos (by decide)]
theorem root251Scan18 : scan edge (search edge 3) 4 [1609] 1609 root251Tail18 winner = winner :=
  (root251Step18).trans root251Scan19
theorem root251Scan17 : scan edge (search edge 3) 4 [1609] 1609 root251Tail17 winner = winner :=
  (root251Step17).trans root251Scan18
theorem root251Scan16 : scan edge (search edge 3) 4 [1609] 1609 root251Tail16 winner = winner :=
  (root251Step16).trans root251Scan17
theorem root251Scan15 : scan edge (search edge 3) 4 [1609] 1609 root251Tail15 winner = winner :=
  (root251Step15).trans root251Scan16
theorem root251Scan14 : scan edge (search edge 3) 4 [1609] 1609 root251Tail14 winner = winner :=
  (root251Step14).trans root251Scan15
theorem root251Scan13 : scan edge (search edge 3) 4 [1609] 1609 root251Tail13 winner = winner :=
  (root251Step13).trans root251Scan14
theorem root251Scan12 : scan edge (search edge 3) 4 [1609] 1609 root251Tail12 winner = winner :=
  (root251Step12).trans root251Scan13
theorem root251Scan11 : scan edge (search edge 3) 4 [1609] 1609 root251Tail11 winner = winner :=
  (root251Step11).trans root251Scan12
theorem root251Scan10 : scan edge (search edge 3) 4 [1609] 1609 root251Tail10 winner = winner :=
  (root251Step10).trans root251Scan11
theorem root251Scan9 : scan edge (search edge 3) 4 [1609] 1609 root251Tail9 winner = winner :=
  (root251Step9).trans root251Scan10
theorem root251Scan8 : scan edge (search edge 3) 4 [1609] 1609 root251Tail8 winner = winner :=
  (root251Step8).trans root251Scan9
theorem root251Scan7 : scan edge (search edge 3) 4 [1609] 1609 root251Tail7 winner = winner :=
  (root251Step7).trans root251Scan8
theorem root251Scan6 : scan edge (search edge 3) 4 [1609] 1609 root251Tail6 winner = winner :=
  (root251Step6).trans root251Scan7
theorem root251Scan5 : scan edge (search edge 3) 4 [1609] 1609 root251Tail5 winner = winner :=
  (root251Step5).trans root251Scan6
theorem root251Scan4 : scan edge (search edge 3) 4 [1609] 1609 root251Tail4 winner = winner :=
  (root251Step4).trans root251Scan5
theorem root251Scan3 : scan edge (search edge 3) 4 [1609] 1609 root251Tail3 winner = winner :=
  (root251Step3).trans root251Scan4
theorem root251Scan2 : scan edge (search edge 3) 4 [1609] 1609 root251Tail2 winner = winner :=
  (root251Step2).trans root251Scan3
theorem root251Scan1 : scan edge (search edge 3) 4 [1609] 1609 root251Tail1 winner = winner :=
  (root251Step1).trans root251Scan2
theorem root251Scan0 : scan edge (search edge 3) 4 [1609] 1609 root251Tail0 winner = winner :=
  (root251Step0).trans root251Scan1
set_option maxRecDepth 4096 in
theorem root251Branch : search edge 4 [1609] root251Candidates 1609 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1609] 1609 root251Tail0 winner = winner
  exact root251Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
