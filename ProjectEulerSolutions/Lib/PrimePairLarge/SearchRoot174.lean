import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root174Candidates : List Nat := Data.row1051
set_option maxRecDepth 8192 in
theorem root174Candidates_checked : candidates edge 5 0 1051 tail175 winner.bound = root174Candidates := by
  apply candidates_eq_row edge 5 0 1051 tail175 root174Candidates winner.bound tail175_ordered
  · decide
  · decide
  · funext q
    rfl
def root174Tail57 : List Nat := []
def root174Tail56 : List Nat := 24979 :: root174Tail57
def root174Tail55 : List Nat := 24781 :: root174Tail56
def root174Tail54 : List Nat := 24763 :: root174Tail55
def root174Tail53 : List Nat := 23773 :: root174Tail54
def root174Tail52 : List Nat := 23311 :: root174Tail53
def root174Tail51 : List Nat := 22669 :: root174Tail52
def root174Tail50 : List Nat := 22453 :: root174Tail51
def root174Tail49 : List Nat := 21787 :: root174Tail50
def root174Tail48 : List Nat := 21529 :: root174Tail49
def root174Tail47 : List Nat := 20887 :: root174Tail48
def root174Tail46 : List Nat := 19813 :: root174Tail47
def root174Tail45 : List Nat := 19699 :: root174Tail46
def root174Tail44 : List Nat := 19237 :: root174Tail45
def root174Tail43 : List Nat := 18757 :: root174Tail44
def root174Tail42 : List Nat := 18493 :: root174Tail43
def root174Tail41 : List Nat := 17863 :: root174Tail42
def root174Tail40 : List Nat := 17791 :: root174Tail41
def root174Tail39 : List Nat := 16447 :: root174Tail40
def root174Tail38 : List Nat := 16183 :: root174Tail39
def root174Tail37 : List Nat := 16069 :: root174Tail38
def root174Tail36 : List Nat := 15817 :: root174Tail37
def root174Tail35 : List Nat := 14683 :: root174Tail36
def root174Tail34 : List Nat := 13807 :: root174Tail35
def root174Tail33 : List Nat := 13297 :: root174Tail34
def root174Tail32 : List Nat := 13003 :: root174Tail33
def root174Tail31 : List Nat := 12373 :: root174Tail32
def root174Tail30 : List Nat := 11299 :: root174Tail31
def root174Tail29 : List Nat := 11239 :: root174Tail30
def root174Tail28 : List Nat := 11161 :: root174Tail29
def root174Tail27 : List Nat := 9697 :: root174Tail28
def root174Tail26 : List Nat := 9601 :: root174Tail27
def root174Tail25 : List Nat := 8623 :: root174Tail26
def root174Tail24 : List Nat := 8353 :: root174Tail25
def root174Tail23 : List Nat := 8311 :: root174Tail24
def root174Tail22 : List Nat := 7867 :: root174Tail23
def root174Tail21 : List Nat := 6607 :: root174Tail22
def root174Tail20 : List Nat := 6301 :: root174Tail21
def root174Tail19 : List Nat := 5839 :: root174Tail20
def root174Tail18 : List Nat := 5443 :: root174Tail19
def root174Tail17 : List Nat := 4909 :: root174Tail18
def root174Tail16 : List Nat := 4813 :: root174Tail17
def root174Tail15 : List Nat := 4729 :: root174Tail16
def root174Tail14 : List Nat := 4507 :: root174Tail15
def root174Tail13 : List Nat := 4057 :: root174Tail14
def root174Tail12 : List Nat := 4003 :: root174Tail13
def root174Tail11 : List Nat := 3739 :: root174Tail12
def root174Tail10 : List Nat := 3709 :: root174Tail11
def root174Tail9 : List Nat := 3583 :: root174Tail10
def root174Tail8 : List Nat := 3331 :: root174Tail9
def root174Tail7 : List Nat := 3271 :: root174Tail8
def root174Tail6 : List Nat := 3067 :: root174Tail7
def root174Tail5 : List Nat := 2713 :: root174Tail6
def root174Tail4 : List Nat := 2617 :: root174Tail5
def root174Tail3 : List Nat := 2389 :: root174Tail4
def root174Tail2 : List Nat := 2347 :: root174Tail3
def root174Tail1 : List Nat := 2221 :: root174Tail2
def root174Tail0 : List Nat := 2017 :: root174Tail1
def root174Candidates0 : List Nat := [2221, 2617, 8311, 11299]
set_option maxRecDepth 8192 in
theorem root174Candidates0_checked : candidates edge 4 1051 2017 root174Tail1 winner.bound = root174Candidates0 := by
  unfold candidates
  have row : edge 2017 = (SortedNatTree.contains FastRows.row2017) := by
    change Data.row2017.contains = (SortedNatTree.contains FastRows.row2017)
    exact FastRows.row2017_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root174Child0 : search edge 3 [1051, 2017] root174Candidates0 3068 winner = winner := by decide
theorem root174Step0 : scan edge (search edge 3) 4 [1051] 1051 root174Tail0 winner =
    scan edge (search edge 3) 4 [1051] 1051 root174Tail1 winner := by
  rw [root174Tail0, scan, if_neg (by decide), root174Candidates0_checked]
  change scan edge (search edge 3) 4 [1051] 1051 root174Tail1
    (if 3 ≤ root174Candidates0.length then search edge 3 [1051, 2017] root174Candidates0 3068 winner else winner) = _
  rw [if_pos (by decide), root174Child0]
def root174Candidates1 : List Nat := [2389, 4909, 5839, 19699]
set_option maxRecDepth 8192 in
theorem root174Candidates1_checked : candidates edge 4 1051 2221 root174Tail2 winner.bound = root174Candidates1 := by
  unfold candidates
  have row : edge 2221 = (SortedNatTree.contains FastRows.row2221) := by
    change Data.row2221.contains = (SortedNatTree.contains FastRows.row2221)
    exact FastRows.row2221_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root174Child1 : search edge 3 [1051, 2221] root174Candidates1 3272 winner = winner := by decide
theorem root174Step1 : scan edge (search edge 3) 4 [1051] 1051 root174Tail1 winner =
    scan edge (search edge 3) 4 [1051] 1051 root174Tail2 winner := by
  rw [root174Tail1, scan, if_neg (by decide), root174Candidates1_checked]
  change scan edge (search edge 3) 4 [1051] 1051 root174Tail2
    (if 3 ≤ root174Candidates1.length then search edge 3 [1051, 2221] root174Candidates1 3272 winner else winner) = _
  rw [if_pos (by decide), root174Child1]
def root174Candidates2 : List Nat := [3067]
set_option maxRecDepth 8192 in
theorem root174Candidates2_checked : candidates edge 4 1051 2347 root174Tail3 winner.bound = root174Candidates2 := by
  unfold candidates
  have row : edge 2347 = (SortedNatTree.contains FastRows.row2347) := by
    change Data.row2347.contains = (SortedNatTree.contains FastRows.row2347)
    exact FastRows.row2347_checked.symm
  rw [row]
  decide
theorem root174Step2 : scan edge (search edge 3) 4 [1051] 1051 root174Tail2 winner =
    scan edge (search edge 3) 4 [1051] 1051 root174Tail3 winner := by
  rw [root174Tail2, scan, if_neg (by decide), root174Candidates2_checked]
  change scan edge (search edge 3) 4 [1051] 1051 root174Tail3
    (if 3 ≤ root174Candidates2.length then search edge 3 [1051, 2347] root174Candidates2 3398 winner else winner) = _
  rw [if_neg (by decide)]
def root174Candidates3 : List Nat := [9697]
set_option maxRecDepth 8192 in
theorem root174Candidates3_checked : candidates edge 4 1051 2389 root174Tail4 winner.bound = root174Candidates3 := by
  unfold candidates
  have row : edge 2389 = (SortedNatTree.contains FastRows.row2389) := by
    change Data.row2389.contains = (SortedNatTree.contains FastRows.row2389)
    exact FastRows.row2389_checked.symm
  rw [row]
  decide
theorem root174Step3 : scan edge (search edge 3) 4 [1051] 1051 root174Tail3 winner =
    scan edge (search edge 3) 4 [1051] 1051 root174Tail4 winner := by
  rw [root174Tail3, scan, if_neg (by decide), root174Candidates3_checked]
  change scan edge (search edge 3) 4 [1051] 1051 root174Tail4
    (if 3 ≤ root174Candidates3.length then search edge 3 [1051, 2389] root174Candidates3 3440 winner else winner) = _
  rw [if_neg (by decide)]
def root174Candidates4 : List Nat := [5839, 17791]
set_option maxRecDepth 8192 in
theorem root174Candidates4_checked : candidates edge 4 1051 2617 root174Tail5 winner.bound = root174Candidates4 := by
  unfold candidates
  have row : edge 2617 = (SortedNatTree.contains FastRows.row2617) := by
    change Data.row2617.contains = (SortedNatTree.contains FastRows.row2617)
    exact FastRows.row2617_checked.symm
  rw [row]
  decide
theorem root174Step4 : scan edge (search edge 3) 4 [1051] 1051 root174Tail4 winner =
    scan edge (search edge 3) 4 [1051] 1051 root174Tail5 winner := by
  rw [root174Tail4, scan, if_neg (by decide), root174Candidates4_checked]
  change scan edge (search edge 3) 4 [1051] 1051 root174Tail5
    (if 3 ≤ root174Candidates4.length then search edge 3 [1051, 2617] root174Candidates4 3668 winner else winner) = _
  rw [if_neg (by decide)]
def root174Candidates5 : List Nat := [3739, 5443, 13807, 15817]
set_option maxRecDepth 8192 in
theorem root174Candidates5_checked : candidates edge 4 1051 2713 root174Tail6 winner.bound = root174Candidates5 := by
  unfold candidates
  have row : edge 2713 = (SortedNatTree.contains FastRows.row2713) := by
    change Data.row2713.contains = (SortedNatTree.contains FastRows.row2713)
    exact FastRows.row2713_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root174Child5 : search edge 3 [1051, 2713] root174Candidates5 3764 winner = winner := by decide
theorem root174Step5 : scan edge (search edge 3) 4 [1051] 1051 root174Tail5 winner =
    scan edge (search edge 3) 4 [1051] 1051 root174Tail6 winner := by
  rw [root174Tail5, scan, if_neg (by decide), root174Candidates5_checked]
  change scan edge (search edge 3) 4 [1051] 1051 root174Tail6
    (if 3 ≤ root174Candidates5.length then search edge 3 [1051, 2713] root174Candidates5 3764 winner else winner) = _
  rw [if_pos (by decide), root174Child5]
def root174Candidates6 : List Nat := [4909, 21787]
set_option maxRecDepth 8192 in
theorem root174Candidates6_checked : candidates edge 4 1051 3067 root174Tail7 winner.bound = root174Candidates6 := by
  unfold candidates
  have row : edge 3067 = (SortedNatTree.contains FastRows.row3067) := by
    change Data.row3067.contains = (SortedNatTree.contains FastRows.row3067)
    exact FastRows.row3067_checked.symm
  rw [row]
  decide
theorem root174Step6 : scan edge (search edge 3) 4 [1051] 1051 root174Tail6 winner =
    scan edge (search edge 3) 4 [1051] 1051 root174Tail7 winner := by
  rw [root174Tail6, scan, if_neg (by decide), root174Candidates6_checked]
  change scan edge (search edge 3) 4 [1051] 1051 root174Tail7
    (if 3 ≤ root174Candidates6.length then search edge 3 [1051, 3067] root174Candidates6 4118 winner else winner) = _
  rw [if_neg (by decide)]
def root174Candidates7 : List Nat := [9697]
set_option maxRecDepth 8192 in
theorem root174Candidates7_checked : candidates edge 4 1051 3271 root174Tail8 winner.bound = root174Candidates7 := by
  unfold candidates
  have row : edge 3271 = (SortedNatTree.contains FastRows.row3271) := by
    change Data.row3271.contains = (SortedNatTree.contains FastRows.row3271)
    exact FastRows.row3271_checked.symm
  rw [row]
  decide
theorem root174Step7 : scan edge (search edge 3) 4 [1051] 1051 root174Tail7 winner =
    scan edge (search edge 3) 4 [1051] 1051 root174Tail8 winner := by
  rw [root174Tail7, scan, if_neg (by decide), root174Candidates7_checked]
  change scan edge (search edge 3) 4 [1051] 1051 root174Tail8
    (if 3 ≤ root174Candidates7.length then search edge 3 [1051, 3271] root174Candidates7 4322 winner else winner) = _
  rw [if_neg (by decide)]
def root174Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root174Candidates8_checked : candidates edge 4 1051 3331 root174Tail9 winner.bound = root174Candidates8 := by
  unfold candidates
  have row : edge 3331 = (SortedNatTree.contains FastRows.row3331) := by
    change Data.row3331.contains = (SortedNatTree.contains FastRows.row3331)
    exact FastRows.row3331_checked.symm
  rw [row]
  decide
theorem root174Step8 : scan edge (search edge 3) 4 [1051] 1051 root174Tail8 winner =
    scan edge (search edge 3) 4 [1051] 1051 root174Tail9 winner := by
  rw [root174Tail8, scan, if_neg (by decide), root174Candidates8_checked]
  change scan edge (search edge 3) 4 [1051] 1051 root174Tail9
    (if 3 ≤ root174Candidates8.length then search edge 3 [1051, 3331] root174Candidates8 4382 winner else winner) = _
  rw [if_neg (by decide)]
def root174Candidates9 : List Nat := [18757, 19237]
set_option maxRecDepth 8192 in
theorem root174Candidates9_checked : candidates edge 4 1051 3583 root174Tail10 winner.bound = root174Candidates9 := by
  unfold candidates
  have row : edge 3583 = (SortedNatTree.contains FastRows.row3583) := by
    change Data.row3583.contains = (SortedNatTree.contains FastRows.row3583)
    exact FastRows.row3583_checked.symm
  rw [row]
  decide
theorem root174Step9 : scan edge (search edge 3) 4 [1051] 1051 root174Tail9 winner =
    scan edge (search edge 3) 4 [1051] 1051 root174Tail10 winner := by
  rw [root174Tail9, scan, if_neg (by decide), root174Candidates9_checked]
  change scan edge (search edge 3) 4 [1051] 1051 root174Tail10
    (if 3 ≤ root174Candidates9.length then search edge 3 [1051, 3583] root174Candidates9 4634 winner else winner) = _
  rw [if_neg (by decide)]
def root174Candidates10 : List Nat := [15817]
set_option maxRecDepth 8192 in
theorem root174Candidates10_checked : candidates edge 4 1051 3709 root174Tail11 winner.bound = root174Candidates10 := by
  unfold candidates
  have row : edge 3709 = (SortedNatTree.contains FastRows.row3709) := by
    change Data.row3709.contains = (SortedNatTree.contains FastRows.row3709)
    exact FastRows.row3709_checked.symm
  rw [row]
  decide
theorem root174Step10 : scan edge (search edge 3) 4 [1051] 1051 root174Tail10 winner =
    scan edge (search edge 3) 4 [1051] 1051 root174Tail11 winner := by
  rw [root174Tail10, scan, if_neg (by decide), root174Candidates10_checked]
  change scan edge (search edge 3) 4 [1051] 1051 root174Tail11
    (if 3 ≤ root174Candidates10.length then search edge 3 [1051, 3709] root174Candidates10 4760 winner else winner) = _
  rw [if_neg (by decide)]
def root174Candidates11 : List Nat := [8353, 13807, 19237]
set_option maxRecDepth 8192 in
theorem root174Candidates11_checked : candidates edge 4 1051 3739 root174Tail12 winner.bound = root174Candidates11 := by
  unfold candidates
  have row : edge 3739 = (SortedNatTree.contains FastRows.row3739) := by
    change Data.row3739.contains = (SortedNatTree.contains FastRows.row3739)
    exact FastRows.row3739_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root174Child11 : search edge 3 [1051, 3739] root174Candidates11 4790 winner = winner := by decide
theorem root174Step11 : scan edge (search edge 3) 4 [1051] 1051 root174Tail11 winner =
    scan edge (search edge 3) 4 [1051] 1051 root174Tail12 winner := by
  rw [root174Tail11, scan, if_neg (by decide), root174Candidates11_checked]
  change scan edge (search edge 3) 4 [1051] 1051 root174Tail12
    (if 3 ≤ root174Candidates11.length then search edge 3 [1051, 3739] root174Candidates11 4790 winner else winner) = _
  rw [if_pos (by decide), root174Child11]
def root174Candidates12 : List Nat := [11161]
set_option maxRecDepth 8192 in
theorem root174Candidates12_checked : candidates edge 4 1051 4003 root174Tail13 winner.bound = root174Candidates12 := by
  unfold candidates
  have row : edge 4003 = (SortedNatTree.contains FastRows.row4003) := by
    change Data.row4003.contains = (SortedNatTree.contains FastRows.row4003)
    exact FastRows.row4003_checked.symm
  rw [row]
  decide
theorem root174Step12 : scan edge (search edge 3) 4 [1051] 1051 root174Tail12 winner =
    scan edge (search edge 3) 4 [1051] 1051 root174Tail13 winner := by
  rw [root174Tail12, scan, if_neg (by decide), root174Candidates12_checked]
  change scan edge (search edge 3) 4 [1051] 1051 root174Tail13
    (if 3 ≤ root174Candidates12.length then search edge 3 [1051, 4003] root174Candidates12 5054 winner else winner) = _
  rw [if_neg (by decide)]
def root174Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root174Candidates13_checked : candidates edge 4 1051 4057 root174Tail14 winner.bound = root174Candidates13 := by
  unfold candidates
  have row : edge 4057 = (SortedNatTree.contains FastRows.row4057) := by
    change Data.row4057.contains = (SortedNatTree.contains FastRows.row4057)
    exact FastRows.row4057_checked.symm
  rw [row]
  decide
theorem root174Step13 : scan edge (search edge 3) 4 [1051] 1051 root174Tail13 winner =
    scan edge (search edge 3) 4 [1051] 1051 root174Tail14 winner := by
  rw [root174Tail13, scan, if_neg (by decide), root174Candidates13_checked]
  change scan edge (search edge 3) 4 [1051] 1051 root174Tail14
    (if 3 ≤ root174Candidates13.length then search edge 3 [1051, 4057] root174Candidates13 5108 winner else winner) = _
  rw [if_neg (by decide)]
def root174Candidates14 : List Nat := [13807]
set_option maxRecDepth 8192 in
theorem root174Candidates14_checked : candidates edge 4 1051 4507 root174Tail15 winner.bound = root174Candidates14 := by
  unfold candidates
  have row : edge 4507 = (SortedNatTree.contains FastRows.row4507) := by
    change Data.row4507.contains = (SortedNatTree.contains FastRows.row4507)
    exact FastRows.row4507_checked.symm
  rw [row]
  decide
theorem root174Step14 : scan edge (search edge 3) 4 [1051] 1051 root174Tail14 winner =
    scan edge (search edge 3) 4 [1051] 1051 root174Tail15 winner := by
  rw [root174Tail14, scan, if_neg (by decide), root174Candidates14_checked]
  change scan edge (search edge 3) 4 [1051] 1051 root174Tail15
    (if 3 ≤ root174Candidates14.length then search edge 3 [1051, 4507] root174Candidates14 5558 winner else winner) = _
  rw [if_neg (by decide)]
def root174Candidates15 : List Nat := [4909, 9601, 11239]
set_option maxRecDepth 8192 in
theorem root174Candidates15_checked : candidates edge 4 1051 4729 root174Tail16 winner.bound = root174Candidates15 := by
  unfold candidates
  have row : edge 4729 = (SortedNatTree.contains FastRows.row4729) := by
    change Data.row4729.contains = (SortedNatTree.contains FastRows.row4729)
    exact FastRows.row4729_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root174Child15 : search edge 3 [1051, 4729] root174Candidates15 5780 winner = winner := by decide
theorem root174Step15 : scan edge (search edge 3) 4 [1051] 1051 root174Tail15 winner =
    scan edge (search edge 3) 4 [1051] 1051 root174Tail16 winner := by
  rw [root174Tail15, scan, if_neg (by decide), root174Candidates15_checked]
  change scan edge (search edge 3) 4 [1051] 1051 root174Tail16
    (if 3 ≤ root174Candidates15.length then search edge 3 [1051, 4729] root174Candidates15 5780 winner else winner) = _
  rw [if_pos (by decide), root174Child15]
def root174Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root174Candidates16_checked : candidates edge 4 1051 4813 root174Tail17 winner.bound = root174Candidates16 := by
  unfold candidates
  have row : edge 4813 = (SortedNatTree.contains FastRows.row4813) := by
    change Data.row4813.contains = (SortedNatTree.contains FastRows.row4813)
    exact FastRows.row4813_checked.symm
  rw [row]
  decide
theorem root174Step16 : scan edge (search edge 3) 4 [1051] 1051 root174Tail16 winner =
    scan edge (search edge 3) 4 [1051] 1051 root174Tail17 winner := by
  rw [root174Tail16, scan, if_neg (by decide), root174Candidates16_checked]
  change scan edge (search edge 3) 4 [1051] 1051 root174Tail17
    (if 3 ≤ root174Candidates16.length then search edge 3 [1051, 4813] root174Candidates16 5864 winner else winner) = _
  rw [if_neg (by decide)]
def root174Candidates17 : List Nat := [5443, 19237]
set_option maxRecDepth 8192 in
theorem root174Candidates17_checked : candidates edge 4 1051 4909 root174Tail18 winner.bound = root174Candidates17 := by
  unfold candidates
  have row : edge 4909 = (SortedNatTree.contains FastRows.row4909) := by
    change Data.row4909.contains = (SortedNatTree.contains FastRows.row4909)
    exact FastRows.row4909_checked.symm
  rw [row]
  decide
theorem root174Step17 : scan edge (search edge 3) 4 [1051] 1051 root174Tail17 winner =
    scan edge (search edge 3) 4 [1051] 1051 root174Tail18 winner := by
  rw [root174Tail17, scan, if_neg (by decide), root174Candidates17_checked]
  change scan edge (search edge 3) 4 [1051] 1051 root174Tail18
    (if 3 ≤ root174Candidates17.length then search edge 3 [1051, 4909] root174Candidates17 5960 winner else winner) = _
  rw [if_neg (by decide)]
def root174Candidates18 : List Nat := []
set_option maxRecDepth 8192 in
theorem root174Candidates18_checked : candidates edge 4 1051 5443 root174Tail19 winner.bound = root174Candidates18 := by
  unfold candidates
  have row : edge 5443 = (SortedNatTree.contains FastRows.row5443) := by
    change Data.row5443.contains = (SortedNatTree.contains FastRows.row5443)
    exact FastRows.row5443_checked.symm
  rw [row]
  decide
theorem root174Step18 : scan edge (search edge 3) 4 [1051] 1051 root174Tail18 winner =
    scan edge (search edge 3) 4 [1051] 1051 root174Tail19 winner := by
  rw [root174Tail18, scan, if_neg (by decide), root174Candidates18_checked]
  change scan edge (search edge 3) 4 [1051] 1051 root174Tail19
    (if 3 ≤ root174Candidates18.length then search edge 3 [1051, 5443] root174Candidates18 6494 winner else winner) = _
  rw [if_neg (by decide)]
def root174Candidates19 : List Nat := []
set_option maxRecDepth 8192 in
theorem root174Candidates19_checked : candidates edge 4 1051 5839 root174Tail20 winner.bound = root174Candidates19 := by
  unfold candidates
  have row : edge 5839 = (SortedNatTree.contains FastRows.row5839) := by
    change Data.row5839.contains = (SortedNatTree.contains FastRows.row5839)
    exact FastRows.row5839_checked.symm
  rw [row]
  decide
theorem root174Step19 : scan edge (search edge 3) 4 [1051] 1051 root174Tail19 winner =
    scan edge (search edge 3) 4 [1051] 1051 root174Tail20 winner := by
  rw [root174Tail19, scan, if_neg (by decide), root174Candidates19_checked]
  change scan edge (search edge 3) 4 [1051] 1051 root174Tail20
    (if 3 ≤ root174Candidates19.length then search edge 3 [1051, 5839] root174Candidates19 6890 winner else winner) = _
  rw [if_neg (by decide)]
theorem root174Scan20 : scan edge (search edge 3) 4 [1051] 1051 root174Tail20 winner = winner := by rw [root174Tail20, scan, if_pos (by decide)]
theorem root174Scan19 : scan edge (search edge 3) 4 [1051] 1051 root174Tail19 winner = winner :=
  (root174Step19).trans root174Scan20
theorem root174Scan18 : scan edge (search edge 3) 4 [1051] 1051 root174Tail18 winner = winner :=
  (root174Step18).trans root174Scan19
theorem root174Scan17 : scan edge (search edge 3) 4 [1051] 1051 root174Tail17 winner = winner :=
  (root174Step17).trans root174Scan18
theorem root174Scan16 : scan edge (search edge 3) 4 [1051] 1051 root174Tail16 winner = winner :=
  (root174Step16).trans root174Scan17
theorem root174Scan15 : scan edge (search edge 3) 4 [1051] 1051 root174Tail15 winner = winner :=
  (root174Step15).trans root174Scan16
theorem root174Scan14 : scan edge (search edge 3) 4 [1051] 1051 root174Tail14 winner = winner :=
  (root174Step14).trans root174Scan15
theorem root174Scan13 : scan edge (search edge 3) 4 [1051] 1051 root174Tail13 winner = winner :=
  (root174Step13).trans root174Scan14
theorem root174Scan12 : scan edge (search edge 3) 4 [1051] 1051 root174Tail12 winner = winner :=
  (root174Step12).trans root174Scan13
theorem root174Scan11 : scan edge (search edge 3) 4 [1051] 1051 root174Tail11 winner = winner :=
  (root174Step11).trans root174Scan12
theorem root174Scan10 : scan edge (search edge 3) 4 [1051] 1051 root174Tail10 winner = winner :=
  (root174Step10).trans root174Scan11
theorem root174Scan9 : scan edge (search edge 3) 4 [1051] 1051 root174Tail9 winner = winner :=
  (root174Step9).trans root174Scan10
theorem root174Scan8 : scan edge (search edge 3) 4 [1051] 1051 root174Tail8 winner = winner :=
  (root174Step8).trans root174Scan9
theorem root174Scan7 : scan edge (search edge 3) 4 [1051] 1051 root174Tail7 winner = winner :=
  (root174Step7).trans root174Scan8
theorem root174Scan6 : scan edge (search edge 3) 4 [1051] 1051 root174Tail6 winner = winner :=
  (root174Step6).trans root174Scan7
theorem root174Scan5 : scan edge (search edge 3) 4 [1051] 1051 root174Tail5 winner = winner :=
  (root174Step5).trans root174Scan6
theorem root174Scan4 : scan edge (search edge 3) 4 [1051] 1051 root174Tail4 winner = winner :=
  (root174Step4).trans root174Scan5
theorem root174Scan3 : scan edge (search edge 3) 4 [1051] 1051 root174Tail3 winner = winner :=
  (root174Step3).trans root174Scan4
theorem root174Scan2 : scan edge (search edge 3) 4 [1051] 1051 root174Tail2 winner = winner :=
  (root174Step2).trans root174Scan3
theorem root174Scan1 : scan edge (search edge 3) 4 [1051] 1051 root174Tail1 winner = winner :=
  (root174Step1).trans root174Scan2
theorem root174Scan0 : scan edge (search edge 3) 4 [1051] 1051 root174Tail0 winner = winner :=
  (root174Step0).trans root174Scan1
set_option maxRecDepth 4096 in
theorem root174Branch : search edge 4 [1051] root174Candidates 1051 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1051] 1051 root174Tail0 winner = winner
  exact root174Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
