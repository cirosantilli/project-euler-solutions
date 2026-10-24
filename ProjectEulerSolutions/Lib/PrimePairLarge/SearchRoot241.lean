import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root241Candidates : List Nat := Data.row1549
set_option maxRecDepth 8192 in
theorem root241Candidates_checked : candidates edge 5 0 1549 tail242 winner.bound = root241Candidates := by
  apply candidates_eq_row edge 5 0 1549 tail242 root241Candidates winner.bound tail242_ordered
  · decide
  · decide
  · funext q
    rfl
def root241Tail54 : List Nat := []
def root241Tail53 : List Nat := 23623 :: root241Tail54
def root241Tail52 : List Nat := 23059 :: root241Tail53
def root241Tail51 : List Nat := 22777 :: root241Tail52
def root241Tail50 : List Nat := 22111 :: root241Tail51
def root241Tail49 : List Nat := 21817 :: root241Tail50
def root241Tail48 : List Nat := 21379 :: root241Tail49
def root241Tail47 : List Nat := 20551 :: root241Tail48
def root241Tail46 : List Nat := 20533 :: root241Tail47
def root241Tail45 : List Nat := 20173 :: root241Tail46
def root241Tail44 : List Nat := 19819 :: root241Tail45
def root241Tail43 : List Nat := 19753 :: root241Tail44
def root241Tail42 : List Nat := 19717 :: root241Tail43
def root241Tail41 : List Nat := 19207 :: root241Tail42
def root241Tail40 : List Nat := 18583 :: root241Tail41
def root241Tail39 : List Nat := 16987 :: root241Tail40
def root241Tail38 : List Nat := 15727 :: root241Tail39
def root241Tail37 : List Nat := 15601 :: root241Tail38
def root241Tail36 : List Nat := 15289 :: root241Tail37
def root241Tail35 : List Nat := 15031 :: root241Tail36
def root241Tail34 : List Nat := 14323 :: root241Tail35
def root241Tail33 : List Nat := 12343 :: root241Tail34
def root241Tail32 : List Nat := 11821 :: root241Tail33
def root241Tail31 : List Nat := 11719 :: root241Tail32
def root241Tail30 : List Nat := 9829 :: root241Tail31
def root241Tail29 : List Nat := 9391 :: root241Tail30
def root241Tail28 : List Nat := 9007 :: root241Tail29
def root241Tail27 : List Nat := 8689 :: root241Tail28
def root241Tail26 : List Nat := 8521 :: root241Tail27
def root241Tail25 : List Nat := 7393 :: root241Tail26
def root241Tail24 : List Nat := 7333 :: root241Tail25
def root241Tail23 : List Nat := 6997 :: root241Tail24
def root241Tail22 : List Nat := 6577 :: root241Tail23
def root241Tail21 : List Nat := 6361 :: root241Tail22
def root241Tail20 : List Nat := 5881 :: root241Tail21
def root241Tail19 : List Nat := 5479 :: root241Tail20
def root241Tail18 : List Nat := 5209 :: root241Tail19
def root241Tail17 : List Nat := 5059 :: root241Tail18
def root241Tail16 : List Nat := 4789 :: root241Tail17
def root241Tail15 : List Nat := 4783 :: root241Tail16
def root241Tail14 : List Nat := 4561 :: root241Tail15
def root241Tail13 : List Nat := 4057 :: root241Tail14
def root241Tail12 : List Nat := 3823 :: root241Tail13
def root241Tail11 : List Nat := 3733 :: root241Tail12
def root241Tail10 : List Nat := 3637 :: root241Tail11
def root241Tail9 : List Nat := 3529 :: root241Tail10
def root241Tail8 : List Nat := 3463 :: root241Tail9
def root241Tail7 : List Nat := 3253 :: root241Tail8
def root241Tail6 : List Nat := 3019 :: root241Tail7
def root241Tail5 : List Nat := 2833 :: root241Tail6
def root241Tail4 : List Nat := 2707 :: root241Tail5
def root241Tail3 : List Nat := 2311 :: root241Tail4
def root241Tail2 : List Nat := 2221 :: root241Tail3
def root241Tail1 : List Nat := 2143 :: root241Tail2
def root241Tail0 : List Nat := 1933 :: root241Tail1
def root241Candidates0 : List Nat := [3733, 7333, 8521, 9391]
set_option maxRecDepth 8192 in
theorem root241Candidates0_checked : candidates edge 4 1549 1933 root241Tail1 winner.bound = root241Candidates0 := by
  unfold candidates
  have row : edge 1933 = (SortedNatTree.contains FastRows.row1933) := by
    change Data.row1933.contains = (SortedNatTree.contains FastRows.row1933)
    exact FastRows.row1933_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root241Child0 : search edge 3 [1549, 1933] root241Candidates0 3482 winner = winner := by decide
theorem root241Step0 : scan edge (search edge 3) 4 [1549] 1549 root241Tail0 winner =
    scan edge (search edge 3) 4 [1549] 1549 root241Tail1 winner := by
  rw [root241Tail0, scan, if_neg (by decide), root241Candidates0_checked]
  change scan edge (search edge 3) 4 [1549] 1549 root241Tail1
    (if 3 ≤ root241Candidates0.length then search edge 3 [1549, 1933] root241Candidates0 3482 winner else winner) = _
  rw [if_pos (by decide), root241Child0]
def root241Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root241Candidates1_checked : candidates edge 4 1549 2143 root241Tail2 winner.bound = root241Candidates1 := by
  unfold candidates
  have row : edge 2143 = (SortedNatTree.contains FastRows.row2143) := by
    change Data.row2143.contains = (SortedNatTree.contains FastRows.row2143)
    exact FastRows.row2143_checked.symm
  rw [row]
  decide
theorem root241Step1 : scan edge (search edge 3) 4 [1549] 1549 root241Tail1 winner =
    scan edge (search edge 3) 4 [1549] 1549 root241Tail2 winner := by
  rw [root241Tail1, scan, if_neg (by decide), root241Candidates1_checked]
  change scan edge (search edge 3) 4 [1549] 1549 root241Tail2
    (if 3 ≤ root241Candidates1.length then search edge 3 [1549, 2143] root241Candidates1 3692 winner else winner) = _
  rw [if_neg (by decide)]
def root241Candidates2 : List Nat := [15601]
set_option maxRecDepth 8192 in
theorem root241Candidates2_checked : candidates edge 4 1549 2221 root241Tail3 winner.bound = root241Candidates2 := by
  unfold candidates
  have row : edge 2221 = (SortedNatTree.contains FastRows.row2221) := by
    change Data.row2221.contains = (SortedNatTree.contains FastRows.row2221)
    exact FastRows.row2221_checked.symm
  rw [row]
  decide
theorem root241Step2 : scan edge (search edge 3) 4 [1549] 1549 root241Tail2 winner =
    scan edge (search edge 3) 4 [1549] 1549 root241Tail3 winner := by
  rw [root241Tail2, scan, if_neg (by decide), root241Candidates2_checked]
  change scan edge (search edge 3) 4 [1549] 1549 root241Tail3
    (if 3 ≤ root241Candidates2.length then search edge 3 [1549, 2221] root241Candidates2 3770 winner else winner) = _
  rw [if_neg (by decide)]
def root241Candidates3 : List Nat := [3529, 9391]
set_option maxRecDepth 8192 in
theorem root241Candidates3_checked : candidates edge 4 1549 2311 root241Tail4 winner.bound = root241Candidates3 := by
  unfold candidates
  have row : edge 2311 = (SortedNatTree.contains FastRows.row2311) := by
    change Data.row2311.contains = (SortedNatTree.contains FastRows.row2311)
    exact FastRows.row2311_checked.symm
  rw [row]
  decide
theorem root241Step3 : scan edge (search edge 3) 4 [1549] 1549 root241Tail3 winner =
    scan edge (search edge 3) 4 [1549] 1549 root241Tail4 winner := by
  rw [root241Tail3, scan, if_neg (by decide), root241Candidates3_checked]
  change scan edge (search edge 3) 4 [1549] 1549 root241Tail4
    (if 3 ≤ root241Candidates3.length then search edge 3 [1549, 2311] root241Candidates3 3860 winner else winner) = _
  rw [if_neg (by decide)]
def root241Candidates4 : List Nat := [3253, 3463, 19819]
set_option maxRecDepth 8192 in
theorem root241Candidates4_checked : candidates edge 4 1549 2707 root241Tail5 winner.bound = root241Candidates4 := by
  unfold candidates
  have row : edge 2707 = (SortedNatTree.contains FastRows.row2707) := by
    change Data.row2707.contains = (SortedNatTree.contains FastRows.row2707)
    exact FastRows.row2707_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root241Child4 : search edge 3 [1549, 2707] root241Candidates4 4256 winner = winner := by decide
theorem root241Step4 : scan edge (search edge 3) 4 [1549] 1549 root241Tail4 winner =
    scan edge (search edge 3) 4 [1549] 1549 root241Tail5 winner := by
  rw [root241Tail4, scan, if_neg (by decide), root241Candidates4_checked]
  change scan edge (search edge 3) 4 [1549] 1549 root241Tail5
    (if 3 ≤ root241Candidates4.length then search edge 3 [1549, 2707] root241Candidates4 4256 winner else winner) = _
  rw [if_pos (by decide), root241Child4]
def root241Candidates5 : List Nat := [6361, 11719, 15601]
set_option maxRecDepth 8192 in
theorem root241Candidates5_checked : candidates edge 4 1549 2833 root241Tail6 winner.bound = root241Candidates5 := by
  unfold candidates
  have row : edge 2833 = (SortedNatTree.contains FastRows.row2833) := by
    change Data.row2833.contains = (SortedNatTree.contains FastRows.row2833)
    exact FastRows.row2833_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root241Child5 : search edge 3 [1549, 2833] root241Candidates5 4382 winner = winner := by decide
theorem root241Step5 : scan edge (search edge 3) 4 [1549] 1549 root241Tail5 winner =
    scan edge (search edge 3) 4 [1549] 1549 root241Tail6 winner := by
  rw [root241Tail5, scan, if_neg (by decide), root241Candidates5_checked]
  change scan edge (search edge 3) 4 [1549] 1549 root241Tail6
    (if 3 ≤ root241Candidates5.length then search edge 3 [1549, 2833] root241Candidates5 4382 winner else winner) = _
  rw [if_pos (by decide), root241Child5]
def root241Candidates6 : List Nat := [19753]
set_option maxRecDepth 8192 in
theorem root241Candidates6_checked : candidates edge 4 1549 3019 root241Tail7 winner.bound = root241Candidates6 := by
  unfold candidates
  have row : edge 3019 = (SortedNatTree.contains FastRows.row3019) := by
    change Data.row3019.contains = (SortedNatTree.contains FastRows.row3019)
    exact FastRows.row3019_checked.symm
  rw [row]
  decide
theorem root241Step6 : scan edge (search edge 3) 4 [1549] 1549 root241Tail6 winner =
    scan edge (search edge 3) 4 [1549] 1549 root241Tail7 winner := by
  rw [root241Tail6, scan, if_neg (by decide), root241Candidates6_checked]
  change scan edge (search edge 3) 4 [1549] 1549 root241Tail7
    (if 3 ≤ root241Candidates6.length then search edge 3 [1549, 3019] root241Candidates6 4568 winner else winner) = _
  rw [if_neg (by decide)]
def root241Candidates7 : List Nat := [5479, 19717]
set_option maxRecDepth 8192 in
theorem root241Candidates7_checked : candidates edge 4 1549 3253 root241Tail8 winner.bound = root241Candidates7 := by
  unfold candidates
  have row : edge 3253 = (SortedNatTree.contains FastRows.row3253) := by
    change Data.row3253.contains = (SortedNatTree.contains FastRows.row3253)
    exact FastRows.row3253_checked.symm
  rw [row]
  decide
theorem root241Step7 : scan edge (search edge 3) 4 [1549] 1549 root241Tail7 winner =
    scan edge (search edge 3) 4 [1549] 1549 root241Tail8 winner := by
  rw [root241Tail7, scan, if_neg (by decide), root241Candidates7_checked]
  change scan edge (search edge 3) 4 [1549] 1549 root241Tail8
    (if 3 ≤ root241Candidates7.length then search edge 3 [1549, 3253] root241Candidates7 4802 winner else winner) = _
  rw [if_neg (by decide)]
def root241Candidates8 : List Nat := [5479]
set_option maxRecDepth 8192 in
theorem root241Candidates8_checked : candidates edge 4 1549 3463 root241Tail9 winner.bound = root241Candidates8 := by
  unfold candidates
  have row : edge 3463 = (SortedNatTree.contains FastRows.row3463) := by
    change Data.row3463.contains = (SortedNatTree.contains FastRows.row3463)
    exact FastRows.row3463_checked.symm
  rw [row]
  decide
theorem root241Step8 : scan edge (search edge 3) 4 [1549] 1549 root241Tail8 winner =
    scan edge (search edge 3) 4 [1549] 1549 root241Tail9 winner := by
  rw [root241Tail8, scan, if_neg (by decide), root241Candidates8_checked]
  change scan edge (search edge 3) 4 [1549] 1549 root241Tail9
    (if 3 ≤ root241Candidates8.length then search edge 3 [1549, 3463] root241Candidates8 5012 winner else winner) = _
  rw [if_neg (by decide)]
def root241Candidates9 : List Nat := [7393, 8521]
set_option maxRecDepth 8192 in
theorem root241Candidates9_checked : candidates edge 4 1549 3529 root241Tail10 winner.bound = root241Candidates9 := by
  unfold candidates
  have row : edge 3529 = (SortedNatTree.contains FastRows.row3529) := by
    change Data.row3529.contains = (SortedNatTree.contains FastRows.row3529)
    exact FastRows.row3529_checked.symm
  rw [row]
  decide
theorem root241Step9 : scan edge (search edge 3) 4 [1549] 1549 root241Tail9 winner =
    scan edge (search edge 3) 4 [1549] 1549 root241Tail10 winner := by
  rw [root241Tail9, scan, if_neg (by decide), root241Candidates9_checked]
  change scan edge (search edge 3) 4 [1549] 1549 root241Tail10
    (if 3 ≤ root241Candidates9.length then search edge 3 [1549, 3529] root241Candidates9 5078 winner else winner) = _
  rw [if_neg (by decide)]
def root241Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root241Candidates10_checked : candidates edge 4 1549 3637 root241Tail11 winner.bound = root241Candidates10 := by
  unfold candidates
  have row : edge 3637 = (SortedNatTree.contains FastRows.row3637) := by
    change Data.row3637.contains = (SortedNatTree.contains FastRows.row3637)
    exact FastRows.row3637_checked.symm
  rw [row]
  decide
theorem root241Step10 : scan edge (search edge 3) 4 [1549] 1549 root241Tail10 winner =
    scan edge (search edge 3) 4 [1549] 1549 root241Tail11 winner := by
  rw [root241Tail10, scan, if_neg (by decide), root241Candidates10_checked]
  change scan edge (search edge 3) 4 [1549] 1549 root241Tail11
    (if 3 ≤ root241Candidates10.length then search edge 3 [1549, 3637] root241Candidates10 5186 winner else winner) = _
  rw [if_neg (by decide)]
def root241Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root241Candidates11_checked : candidates edge 4 1549 3733 root241Tail12 winner.bound = root241Candidates11 := by
  unfold candidates
  have row : edge 3733 = (SortedNatTree.contains FastRows.row3733) := by
    change Data.row3733.contains = (SortedNatTree.contains FastRows.row3733)
    exact FastRows.row3733_checked.symm
  rw [row]
  decide
theorem root241Step11 : scan edge (search edge 3) 4 [1549] 1549 root241Tail11 winner =
    scan edge (search edge 3) 4 [1549] 1549 root241Tail12 winner := by
  rw [root241Tail11, scan, if_neg (by decide), root241Candidates11_checked]
  change scan edge (search edge 3) 4 [1549] 1549 root241Tail12
    (if 3 ≤ root241Candidates11.length then search edge 3 [1549, 3733] root241Candidates11 5282 winner else winner) = _
  rw [if_neg (by decide)]
def root241Candidates12 : List Nat := [5209, 7393, 8689]
set_option maxRecDepth 8192 in
theorem root241Candidates12_checked : candidates edge 4 1549 3823 root241Tail13 winner.bound = root241Candidates12 := by
  unfold candidates
  have row : edge 3823 = (SortedNatTree.contains FastRows.row3823) := by
    change Data.row3823.contains = (SortedNatTree.contains FastRows.row3823)
    exact FastRows.row3823_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root241Child12 : search edge 3 [1549, 3823] root241Candidates12 5372 winner = winner := by decide
theorem root241Step12 : scan edge (search edge 3) 4 [1549] 1549 root241Tail12 winner =
    scan edge (search edge 3) 4 [1549] 1549 root241Tail13 winner := by
  rw [root241Tail12, scan, if_neg (by decide), root241Candidates12_checked]
  change scan edge (search edge 3) 4 [1549] 1549 root241Tail13
    (if 3 ≤ root241Candidates12.length then search edge 3 [1549, 3823] root241Candidates12 5372 winner else winner) = _
  rw [if_pos (by decide), root241Child12]
def root241Candidates13 : List Nat := [7393, 9829]
set_option maxRecDepth 8192 in
theorem root241Candidates13_checked : candidates edge 4 1549 4057 root241Tail14 winner.bound = root241Candidates13 := by
  unfold candidates
  have row : edge 4057 = (SortedNatTree.contains FastRows.row4057) := by
    change Data.row4057.contains = (SortedNatTree.contains FastRows.row4057)
    exact FastRows.row4057_checked.symm
  rw [row]
  decide
theorem root241Step13 : scan edge (search edge 3) 4 [1549] 1549 root241Tail13 winner =
    scan edge (search edge 3) 4 [1549] 1549 root241Tail14 winner := by
  rw [root241Tail13, scan, if_neg (by decide), root241Candidates13_checked]
  change scan edge (search edge 3) 4 [1549] 1549 root241Tail14
    (if 3 ≤ root241Candidates13.length then search edge 3 [1549, 4057] root241Candidates13 5606 winner else winner) = _
  rw [if_neg (by decide)]
def root241Candidates14 : List Nat := [4783]
set_option maxRecDepth 8192 in
theorem root241Candidates14_checked : candidates edge 4 1549 4561 root241Tail15 winner.bound = root241Candidates14 := by
  unfold candidates
  have row : edge 4561 = (SortedNatTree.contains FastRows.row4561) := by
    change Data.row4561.contains = (SortedNatTree.contains FastRows.row4561)
    exact FastRows.row4561_checked.symm
  rw [row]
  decide
theorem root241Step14 : scan edge (search edge 3) 4 [1549] 1549 root241Tail14 winner =
    scan edge (search edge 3) 4 [1549] 1549 root241Tail15 winner := by
  rw [root241Tail14, scan, if_neg (by decide), root241Candidates14_checked]
  change scan edge (search edge 3) 4 [1549] 1549 root241Tail15
    (if 3 ≤ root241Candidates14.length then search edge 3 [1549, 4561] root241Candidates14 6110 winner else winner) = _
  rw [if_neg (by decide)]
def root241Candidates15 : List Nat := [15601, 18583]
set_option maxRecDepth 8192 in
theorem root241Candidates15_checked : candidates edge 4 1549 4783 root241Tail16 winner.bound = root241Candidates15 := by
  unfold candidates
  have row : edge 4783 = (SortedNatTree.contains FastRows.row4783) := by
    change Data.row4783.contains = (SortedNatTree.contains FastRows.row4783)
    exact FastRows.row4783_checked.symm
  rw [row]
  decide
theorem root241Step15 : scan edge (search edge 3) 4 [1549] 1549 root241Tail15 winner =
    scan edge (search edge 3) 4 [1549] 1549 root241Tail16 winner := by
  rw [root241Tail15, scan, if_neg (by decide), root241Candidates15_checked]
  change scan edge (search edge 3) 4 [1549] 1549 root241Tail16
    (if 3 ≤ root241Candidates15.length then search edge 3 [1549, 4783] root241Candidates15 6332 winner else winner) = _
  rw [if_neg (by decide)]
def root241Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root241Candidates16_checked : candidates edge 4 1549 4789 root241Tail17 winner.bound = root241Candidates16 := by
  unfold candidates
  have row : edge 4789 = (SortedNatTree.contains FastRows.row4789) := by
    change Data.row4789.contains = (SortedNatTree.contains FastRows.row4789)
    exact FastRows.row4789_checked.symm
  rw [row]
  decide
theorem root241Step16 : scan edge (search edge 3) 4 [1549] 1549 root241Tail16 winner =
    scan edge (search edge 3) 4 [1549] 1549 root241Tail17 winner := by
  rw [root241Tail16, scan, if_neg (by decide), root241Candidates16_checked]
  change scan edge (search edge 3) 4 [1549] 1549 root241Tail17
    (if 3 ≤ root241Candidates16.length then search edge 3 [1549, 4789] root241Candidates16 6338 winner else winner) = _
  rw [if_neg (by decide)]
def root241Candidates17 : List Nat := [9007, 15601]
set_option maxRecDepth 8192 in
theorem root241Candidates17_checked : candidates edge 4 1549 5059 root241Tail18 winner.bound = root241Candidates17 := by
  unfold candidates
  have row : edge 5059 = (SortedNatTree.contains FastRows.row5059) := by
    change Data.row5059.contains = (SortedNatTree.contains FastRows.row5059)
    exact FastRows.row5059_checked.symm
  rw [row]
  decide
theorem root241Step17 : scan edge (search edge 3) 4 [1549] 1549 root241Tail17 winner =
    scan edge (search edge 3) 4 [1549] 1549 root241Tail18 winner := by
  rw [root241Tail17, scan, if_neg (by decide), root241Candidates17_checked]
  change scan edge (search edge 3) 4 [1549] 1549 root241Tail18
    (if 3 ≤ root241Candidates17.length then search edge 3 [1549, 5059] root241Candidates17 6608 winner else winner) = _
  rw [if_neg (by decide)]
def root241Candidates18 : List Nat := []
set_option maxRecDepth 8192 in
theorem root241Candidates18_checked : candidates edge 4 1549 5209 root241Tail19 winner.bound = root241Candidates18 := by
  unfold candidates
  have row : edge 5209 = (SortedNatTree.contains FastRows.row5209) := by
    change Data.row5209.contains = (SortedNatTree.contains FastRows.row5209)
    exact FastRows.row5209_checked.symm
  rw [row]
  decide
theorem root241Step18 : scan edge (search edge 3) 4 [1549] 1549 root241Tail18 winner =
    scan edge (search edge 3) 4 [1549] 1549 root241Tail19 winner := by
  rw [root241Tail18, scan, if_neg (by decide), root241Candidates18_checked]
  change scan edge (search edge 3) 4 [1549] 1549 root241Tail19
    (if 3 ≤ root241Candidates18.length then search edge 3 [1549, 5209] root241Candidates18 6758 winner else winner) = _
  rw [if_neg (by decide)]
def root241Candidates19 : List Nat := []
set_option maxRecDepth 8192 in
theorem root241Candidates19_checked : candidates edge 4 1549 5479 root241Tail20 winner.bound = root241Candidates19 := by
  unfold candidates
  have row : edge 5479 = (SortedNatTree.contains FastRows.row5479) := by
    change Data.row5479.contains = (SortedNatTree.contains FastRows.row5479)
    exact FastRows.row5479_checked.symm
  rw [row]
  decide
theorem root241Step19 : scan edge (search edge 3) 4 [1549] 1549 root241Tail19 winner =
    scan edge (search edge 3) 4 [1549] 1549 root241Tail20 winner := by
  rw [root241Tail19, scan, if_neg (by decide), root241Candidates19_checked]
  change scan edge (search edge 3) 4 [1549] 1549 root241Tail20
    (if 3 ≤ root241Candidates19.length then search edge 3 [1549, 5479] root241Candidates19 7028 winner else winner) = _
  rw [if_neg (by decide)]
def root241Candidates20 : List Nat := []
set_option maxRecDepth 8192 in
theorem root241Candidates20_checked : candidates edge 4 1549 5881 root241Tail21 winner.bound = root241Candidates20 := by
  unfold candidates
  have row : edge 5881 = (SortedNatTree.contains FastRows.row5881) := by
    change Data.row5881.contains = (SortedNatTree.contains FastRows.row5881)
    exact FastRows.row5881_checked.symm
  rw [row]
  decide
theorem root241Step20 : scan edge (search edge 3) 4 [1549] 1549 root241Tail20 winner =
    scan edge (search edge 3) 4 [1549] 1549 root241Tail21 winner := by
  rw [root241Tail20, scan, if_neg (by decide), root241Candidates20_checked]
  change scan edge (search edge 3) 4 [1549] 1549 root241Tail21
    (if 3 ≤ root241Candidates20.length then search edge 3 [1549, 5881] root241Candidates20 7430 winner else winner) = _
  rw [if_neg (by decide)]
theorem root241Scan21 : scan edge (search edge 3) 4 [1549] 1549 root241Tail21 winner = winner := by rw [root241Tail21, scan, if_pos (by decide)]
theorem root241Scan20 : scan edge (search edge 3) 4 [1549] 1549 root241Tail20 winner = winner :=
  (root241Step20).trans root241Scan21
theorem root241Scan19 : scan edge (search edge 3) 4 [1549] 1549 root241Tail19 winner = winner :=
  (root241Step19).trans root241Scan20
theorem root241Scan18 : scan edge (search edge 3) 4 [1549] 1549 root241Tail18 winner = winner :=
  (root241Step18).trans root241Scan19
theorem root241Scan17 : scan edge (search edge 3) 4 [1549] 1549 root241Tail17 winner = winner :=
  (root241Step17).trans root241Scan18
theorem root241Scan16 : scan edge (search edge 3) 4 [1549] 1549 root241Tail16 winner = winner :=
  (root241Step16).trans root241Scan17
theorem root241Scan15 : scan edge (search edge 3) 4 [1549] 1549 root241Tail15 winner = winner :=
  (root241Step15).trans root241Scan16
theorem root241Scan14 : scan edge (search edge 3) 4 [1549] 1549 root241Tail14 winner = winner :=
  (root241Step14).trans root241Scan15
theorem root241Scan13 : scan edge (search edge 3) 4 [1549] 1549 root241Tail13 winner = winner :=
  (root241Step13).trans root241Scan14
theorem root241Scan12 : scan edge (search edge 3) 4 [1549] 1549 root241Tail12 winner = winner :=
  (root241Step12).trans root241Scan13
theorem root241Scan11 : scan edge (search edge 3) 4 [1549] 1549 root241Tail11 winner = winner :=
  (root241Step11).trans root241Scan12
theorem root241Scan10 : scan edge (search edge 3) 4 [1549] 1549 root241Tail10 winner = winner :=
  (root241Step10).trans root241Scan11
theorem root241Scan9 : scan edge (search edge 3) 4 [1549] 1549 root241Tail9 winner = winner :=
  (root241Step9).trans root241Scan10
theorem root241Scan8 : scan edge (search edge 3) 4 [1549] 1549 root241Tail8 winner = winner :=
  (root241Step8).trans root241Scan9
theorem root241Scan7 : scan edge (search edge 3) 4 [1549] 1549 root241Tail7 winner = winner :=
  (root241Step7).trans root241Scan8
theorem root241Scan6 : scan edge (search edge 3) 4 [1549] 1549 root241Tail6 winner = winner :=
  (root241Step6).trans root241Scan7
theorem root241Scan5 : scan edge (search edge 3) 4 [1549] 1549 root241Tail5 winner = winner :=
  (root241Step5).trans root241Scan6
theorem root241Scan4 : scan edge (search edge 3) 4 [1549] 1549 root241Tail4 winner = winner :=
  (root241Step4).trans root241Scan5
theorem root241Scan3 : scan edge (search edge 3) 4 [1549] 1549 root241Tail3 winner = winner :=
  (root241Step3).trans root241Scan4
theorem root241Scan2 : scan edge (search edge 3) 4 [1549] 1549 root241Tail2 winner = winner :=
  (root241Step2).trans root241Scan3
theorem root241Scan1 : scan edge (search edge 3) 4 [1549] 1549 root241Tail1 winner = winner :=
  (root241Step1).trans root241Scan2
theorem root241Scan0 : scan edge (search edge 3) 4 [1549] 1549 root241Tail0 winner = winner :=
  (root241Step0).trans root241Scan1
set_option maxRecDepth 4096 in
theorem root241Branch : search edge 4 [1549] root241Candidates 1549 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1549] 1549 root241Tail0 winner = winner
  exact root241Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
