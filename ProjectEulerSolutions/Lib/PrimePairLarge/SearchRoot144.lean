import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root144Candidates : List Nat := Data.row853
set_option maxRecDepth 8192 in
theorem root144Candidates_checked : candidates edge 5 0 853 tail145 winner.bound = root144Candidates := by
  apply candidates_eq_row edge 5 0 853 tail145 root144Candidates winner.bound tail145_ordered
  · decide
  · decide
  · funext q
    rfl
def root144Tail75 : List Nat := []
def root144Tail74 : List Nat := 24979 :: root144Tail75
def root144Tail73 : List Nat := 24439 :: root144Tail74
def root144Tail72 : List Nat := 24133 :: root144Tail73
def root144Tail71 : List Nat := 24091 :: root144Tail72
def root144Tail70 : List Nat := 23929 :: root144Tail71
def root144Tail69 : List Nat := 23857 :: root144Tail70
def root144Tail68 : List Nat := 23599 :: root144Tail69
def root144Tail67 : List Nat := 23593 :: root144Tail68
def root144Tail66 : List Nat := 23059 :: root144Tail67
def root144Tail65 : List Nat := 22129 :: root144Tail66
def root144Tail64 : List Nat := 21787 :: root144Tail65
def root144Tail63 : List Nat := 21661 :: root144Tail64
def root144Tail62 : List Nat := 21499 :: root144Tail63
def root144Tail61 : List Nat := 21193 :: root144Tail62
def root144Tail60 : List Nat := 19477 :: root144Tail61
def root144Tail59 : List Nat := 19219 :: root144Tail60
def root144Tail58 : List Nat := 18517 :: root144Tail59
def root144Tail57 : List Nat := 18049 :: root144Tail58
def root144Tail56 : List Nat := 17929 :: root144Tail57
def root144Tail55 : List Nat := 17497 :: root144Tail56
def root144Tail54 : List Nat := 17257 :: root144Tail55
def root144Tail53 : List Nat := 17209 :: root144Tail54
def root144Tail52 : List Nat := 16381 :: root144Tail53
def root144Tail51 : List Nat := 16111 :: root144Tail52
def root144Tail50 : List Nat := 15739 :: root144Tail51
def root144Tail49 : List Nat := 15733 :: root144Tail50
def root144Tail48 : List Nat := 15277 :: root144Tail49
def root144Tail47 : List Nat := 14947 :: root144Tail48
def root144Tail46 : List Nat := 14557 :: root144Tail47
def root144Tail45 : List Nat := 14479 :: root144Tail46
def root144Tail44 : List Nat := 14197 :: root144Tail45
def root144Tail43 : List Nat := 13687 :: root144Tail44
def root144Tail42 : List Nat := 13381 :: root144Tail43
def root144Tail41 : List Nat := 13309 :: root144Tail42
def root144Tail40 : List Nat := 13171 :: root144Tail41
def root144Tail39 : List Nat := 12973 :: root144Tail40
def root144Tail38 : List Nat := 12967 :: root144Tail39
def root144Tail37 : List Nat := 12889 :: root144Tail38
def root144Tail36 : List Nat := 12583 :: root144Tail37
def root144Tail35 : List Nat := 12421 :: root144Tail36
def root144Tail34 : List Nat := 12211 :: root144Tail35
def root144Tail33 : List Nat := 12049 :: root144Tail34
def root144Tail32 : List Nat := 11839 :: root144Tail33
def root144Tail31 : List Nat := 11329 :: root144Tail32
def root144Tail30 : List Nat := 10909 :: root144Tail31
def root144Tail29 : List Nat := 10651 :: root144Tail30
def root144Tail28 : List Nat := 9781 :: root144Tail29
def root144Tail27 : List Nat := 9679 :: root144Tail28
def root144Tail26 : List Nat := 9397 :: root144Tail27
def root144Tail25 : List Nat := 8779 :: root144Tail26
def root144Tail24 : List Nat := 8689 :: root144Tail25
def root144Tail23 : List Nat := 8017 :: root144Tail24
def root144Tail22 : List Nat := 7759 :: root144Tail23
def root144Tail21 : List Nat := 7603 :: root144Tail22
def root144Tail20 : List Nat := 6961 :: root144Tail21
def root144Tail19 : List Nat := 6637 :: root144Tail20
def root144Tail18 : List Nat := 6379 :: root144Tail19
def root144Tail17 : List Nat := 5413 :: root144Tail18
def root144Tail16 : List Nat := 4789 :: root144Tail17
def root144Tail15 : List Nat := 4657 :: root144Tail16
def root144Tail14 : List Nat := 4639 :: root144Tail15
def root144Tail13 : List Nat := 4621 :: root144Tail14
def root144Tail12 : List Nat := 4447 :: root144Tail13
def root144Tail11 : List Nat := 4261 :: root144Tail12
def root144Tail10 : List Nat := 3853 :: root144Tail11
def root144Tail9 : List Nat := 3697 :: root144Tail10
def root144Tail8 : List Nat := 3001 :: root144Tail9
def root144Tail7 : List Nat := 2917 :: root144Tail8
def root144Tail6 : List Nat := 2707 :: root144Tail7
def root144Tail5 : List Nat := 2593 :: root144Tail6
def root144Tail4 : List Nat := 2269 :: root144Tail5
def root144Tail3 : List Nat := 1801 :: root144Tail4
def root144Tail2 : List Nat := 1279 :: root144Tail3
def root144Tail1 : List Nat := 1153 :: root144Tail2
def root144Tail0 : List Nat := 1129 :: root144Tail1
def root144Candidates0 : List Nat := [2269, 3001, 7603, 13309]
set_option maxRecDepth 8192 in
theorem root144Candidates0_checked : candidates edge 4 853 1129 root144Tail1 winner.bound = root144Candidates0 := by
  unfold candidates
  have row : edge 1129 = (SortedNatTree.contains FastRows.row1129) := by
    change Data.row1129.contains = (SortedNatTree.contains FastRows.row1129)
    exact FastRows.row1129_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root144Child0 : search edge 3 [853, 1129] root144Candidates0 1982 winner = winner := by decide
theorem root144Step0 : scan edge (search edge 3) 4 [853] 853 root144Tail0 winner =
    scan edge (search edge 3) 4 [853] 853 root144Tail1 winner := by
  rw [root144Tail0, scan, if_neg (by decide), root144Candidates0_checked]
  change scan edge (search edge 3) 4 [853] 853 root144Tail1
    (if 3 ≤ root144Candidates0.length then search edge 3 [853, 1129] root144Candidates0 1982 winner else winner) = _
  rw [if_pos (by decide), root144Child0]
def root144Candidates1 : List Nat := [1801, 3853, 8017]
set_option maxRecDepth 8192 in
theorem root144Candidates1_checked : candidates edge 4 853 1153 root144Tail2 winner.bound = root144Candidates1 := by
  unfold candidates
  have row : edge 1153 = (SortedNatTree.contains FastRows.row1153) := by
    change Data.row1153.contains = (SortedNatTree.contains FastRows.row1153)
    exact FastRows.row1153_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root144Child1 : search edge 3 [853, 1153] root144Candidates1 2006 winner = winner := by decide
theorem root144Step1 : scan edge (search edge 3) 4 [853] 853 root144Tail1 winner =
    scan edge (search edge 3) 4 [853] 853 root144Tail2 winner := by
  rw [root144Tail1, scan, if_neg (by decide), root144Candidates1_checked]
  change scan edge (search edge 3) 4 [853] 853 root144Tail2
    (if 3 ≤ root144Candidates1.length then search edge 3 [853, 1153] root144Candidates1 2006 winner else winner) = _
  rw [if_pos (by decide), root144Child1]
def root144Candidates2 : List Nat := [9781, 12049, 12973]
set_option maxRecDepth 8192 in
theorem root144Candidates2_checked : candidates edge 4 853 1279 root144Tail3 winner.bound = root144Candidates2 := by
  unfold candidates
  have row : edge 1279 = (SortedNatTree.contains FastRows.row1279) := by
    change Data.row1279.contains = (SortedNatTree.contains FastRows.row1279)
    exact FastRows.row1279_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root144Child2 : search edge 3 [853, 1279] root144Candidates2 2132 winner = winner := by decide
theorem root144Step2 : scan edge (search edge 3) 4 [853] 853 root144Tail2 winner =
    scan edge (search edge 3) 4 [853] 853 root144Tail3 winner := by
  rw [root144Tail2, scan, if_neg (by decide), root144Candidates2_checked]
  change scan edge (search edge 3) 4 [853] 853 root144Tail3
    (if 3 ≤ root144Candidates2.length then search edge 3 [853, 1279] root144Candidates2 2132 winner else winner) = _
  rw [if_pos (by decide), root144Child2]
def root144Candidates3 : List Nat := [6637]
set_option maxRecDepth 8192 in
theorem root144Candidates3_checked : candidates edge 4 853 1801 root144Tail4 winner.bound = root144Candidates3 := by
  unfold candidates
  have row : edge 1801 = (SortedNatTree.contains FastRows.row1801) := by
    change Data.row1801.contains = (SortedNatTree.contains FastRows.row1801)
    exact FastRows.row1801_checked.symm
  rw [row]
  decide
theorem root144Step3 : scan edge (search edge 3) 4 [853] 853 root144Tail3 winner =
    scan edge (search edge 3) 4 [853] 853 root144Tail4 winner := by
  rw [root144Tail3, scan, if_neg (by decide), root144Candidates3_checked]
  change scan edge (search edge 3) 4 [853] 853 root144Tail4
    (if 3 ≤ root144Candidates3.length then search edge 3 [853, 1801] root144Candidates3 2654 winner else winner) = _
  rw [if_neg (by decide)]
def root144Candidates4 : List Nat := [4621, 8779, 16111]
set_option maxRecDepth 8192 in
theorem root144Candidates4_checked : candidates edge 4 853 2269 root144Tail5 winner.bound = root144Candidates4 := by
  unfold candidates
  have row : edge 2269 = (SortedNatTree.contains FastRows.row2269) := by
    change Data.row2269.contains = (SortedNatTree.contains FastRows.row2269)
    exact FastRows.row2269_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root144Child4 : search edge 3 [853, 2269] root144Candidates4 3122 winner = winner := by decide
theorem root144Step4 : scan edge (search edge 3) 4 [853] 853 root144Tail4 winner =
    scan edge (search edge 3) 4 [853] 853 root144Tail5 winner := by
  rw [root144Tail4, scan, if_neg (by decide), root144Candidates4_checked]
  change scan edge (search edge 3) 4 [853] 853 root144Tail5
    (if 3 ≤ root144Candidates4.length then search edge 3 [853, 2269] root144Candidates4 3122 winner else winner) = _
  rw [if_pos (by decide), root144Child4]
def root144Candidates5 : List Nat := [12421, 12967, 17209]
set_option maxRecDepth 8192 in
theorem root144Candidates5_checked : candidates edge 4 853 2593 root144Tail6 winner.bound = root144Candidates5 := by
  unfold candidates
  have row : edge 2593 = (SortedNatTree.contains FastRows.row2593) := by
    change Data.row2593.contains = (SortedNatTree.contains FastRows.row2593)
    exact FastRows.row2593_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root144Child5 : search edge 3 [853, 2593] root144Candidates5 3446 winner = winner := by decide
theorem root144Step5 : scan edge (search edge 3) 4 [853] 853 root144Tail5 winner =
    scan edge (search edge 3) 4 [853] 853 root144Tail6 winner := by
  rw [root144Tail5, scan, if_neg (by decide), root144Candidates5_checked]
  change scan edge (search edge 3) 4 [853] 853 root144Tail6
    (if 3 ≤ root144Candidates5.length then search edge 3 [853, 2593] root144Candidates5 3446 winner else winner) = _
  rw [if_pos (by decide), root144Child5]
def root144Candidates6 : List Nat := [4261, 5413, 6961, 7759]
set_option maxRecDepth 8192 in
theorem root144Candidates6_checked : candidates edge 4 853 2707 root144Tail7 winner.bound = root144Candidates6 := by
  unfold candidates
  have row : edge 2707 = (SortedNatTree.contains FastRows.row2707) := by
    change Data.row2707.contains = (SortedNatTree.contains FastRows.row2707)
    exact FastRows.row2707_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root144Child6 : search edge 3 [853, 2707] root144Candidates6 3560 winner = winner := by decide
theorem root144Step6 : scan edge (search edge 3) 4 [853] 853 root144Tail6 winner =
    scan edge (search edge 3) 4 [853] 853 root144Tail7 winner := by
  rw [root144Tail6, scan, if_neg (by decide), root144Candidates6_checked]
  change scan edge (search edge 3) 4 [853] 853 root144Tail7
    (if 3 ≤ root144Candidates6.length then search edge 3 [853, 2707] root144Candidates6 3560 winner else winner) = _
  rw [if_pos (by decide), root144Child6]
def root144Candidates7 : List Nat := [21499, 22129]
set_option maxRecDepth 8192 in
theorem root144Candidates7_checked : candidates edge 4 853 2917 root144Tail8 winner.bound = root144Candidates7 := by
  unfold candidates
  have row : edge 2917 = (SortedNatTree.contains FastRows.row2917) := by
    change Data.row2917.contains = (SortedNatTree.contains FastRows.row2917)
    exact FastRows.row2917_checked.symm
  rw [row]
  decide
theorem root144Step7 : scan edge (search edge 3) 4 [853] 853 root144Tail7 winner =
    scan edge (search edge 3) 4 [853] 853 root144Tail8 winner := by
  rw [root144Tail7, scan, if_neg (by decide), root144Candidates7_checked]
  change scan edge (search edge 3) 4 [853] 853 root144Tail8
    (if 3 ≤ root144Candidates7.length then search edge 3 [853, 2917] root144Candidates7 3770 winner else winner) = _
  rw [if_neg (by decide)]
def root144Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root144Candidates8_checked : candidates edge 4 853 3001 root144Tail9 winner.bound = root144Candidates8 := by
  unfold candidates
  have row : edge 3001 = (SortedNatTree.contains FastRows.row3001) := by
    change Data.row3001.contains = (SortedNatTree.contains FastRows.row3001)
    exact FastRows.row3001_checked.symm
  rw [row]
  decide
theorem root144Step8 : scan edge (search edge 3) 4 [853] 853 root144Tail8 winner =
    scan edge (search edge 3) 4 [853] 853 root144Tail9 winner := by
  rw [root144Tail8, scan, if_neg (by decide), root144Candidates8_checked]
  change scan edge (search edge 3) 4 [853] 853 root144Tail9
    (if 3 ≤ root144Candidates8.length then search edge 3 [853, 3001] root144Candidates8 3854 winner else winner) = _
  rw [if_neg (by decide)]
def root144Candidates9 : List Nat := [10651, 14479, 15739, 19219]
set_option maxRecDepth 8192 in
theorem root144Candidates9_checked : candidates edge 4 853 3697 root144Tail10 winner.bound = root144Candidates9 := by
  unfold candidates
  have row : edge 3697 = (SortedNatTree.contains FastRows.row3697) := by
    change Data.row3697.contains = (SortedNatTree.contains FastRows.row3697)
    exact FastRows.row3697_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root144Child9 : search edge 3 [853, 3697] root144Candidates9 4550 winner = winner := by decide
theorem root144Step9 : scan edge (search edge 3) 4 [853] 853 root144Tail9 winner =
    scan edge (search edge 3) 4 [853] 853 root144Tail10 winner := by
  rw [root144Tail9, scan, if_neg (by decide), root144Candidates9_checked]
  change scan edge (search edge 3) 4 [853] 853 root144Tail10
    (if 3 ≤ root144Candidates9.length then search edge 3 [853, 3697] root144Candidates9 4550 winner else winner) = _
  rw [if_pos (by decide), root144Child9]
def root144Candidates10 : List Nat := [19219]
set_option maxRecDepth 8192 in
theorem root144Candidates10_checked : candidates edge 4 853 3853 root144Tail11 winner.bound = root144Candidates10 := by
  unfold candidates
  have row : edge 3853 = (SortedNatTree.contains FastRows.row3853) := by
    change Data.row3853.contains = (SortedNatTree.contains FastRows.row3853)
    exact FastRows.row3853_checked.symm
  rw [row]
  decide
theorem root144Step10 : scan edge (search edge 3) 4 [853] 853 root144Tail10 winner =
    scan edge (search edge 3) 4 [853] 853 root144Tail11 winner := by
  rw [root144Tail10, scan, if_neg (by decide), root144Candidates10_checked]
  change scan edge (search edge 3) 4 [853] 853 root144Tail11
    (if 3 ≤ root144Candidates10.length then search edge 3 [853, 3853] root144Candidates10 4706 winner else winner) = _
  rw [if_neg (by decide)]
def root144Candidates11 : List Nat := [9397]
set_option maxRecDepth 8192 in
theorem root144Candidates11_checked : candidates edge 4 853 4261 root144Tail12 winner.bound = root144Candidates11 := by
  unfold candidates
  have row : edge 4261 = (SortedNatTree.contains FastRows.row4261) := by
    change Data.row4261.contains = (SortedNatTree.contains FastRows.row4261)
    exact FastRows.row4261_checked.symm
  rw [row]
  decide
theorem root144Step11 : scan edge (search edge 3) 4 [853] 853 root144Tail11 winner =
    scan edge (search edge 3) 4 [853] 853 root144Tail12 winner := by
  rw [root144Tail11, scan, if_neg (by decide), root144Candidates11_checked]
  change scan edge (search edge 3) 4 [853] 853 root144Tail12
    (if 3 ≤ root144Candidates11.length then search edge 3 [853, 4261] root144Candidates11 5114 winner else winner) = _
  rw [if_neg (by decide)]
def root144Candidates12 : List Nat := [4789, 18517]
set_option maxRecDepth 8192 in
theorem root144Candidates12_checked : candidates edge 4 853 4447 root144Tail13 winner.bound = root144Candidates12 := by
  unfold candidates
  have row : edge 4447 = (SortedNatTree.contains FastRows.row4447) := by
    change Data.row4447.contains = (SortedNatTree.contains FastRows.row4447)
    exact FastRows.row4447_checked.symm
  rw [row]
  decide
theorem root144Step12 : scan edge (search edge 3) 4 [853] 853 root144Tail12 winner =
    scan edge (search edge 3) 4 [853] 853 root144Tail13 winner := by
  rw [root144Tail12, scan, if_neg (by decide), root144Candidates12_checked]
  change scan edge (search edge 3) 4 [853] 853 root144Tail13
    (if 3 ≤ root144Candidates12.length then search edge 3 [853, 4447] root144Candidates12 5300 winner else winner) = _
  rw [if_neg (by decide)]
def root144Candidates13 : List Nat := [10651, 12421, 14197, 15733]
set_option maxRecDepth 8192 in
theorem root144Candidates13_checked : candidates edge 4 853 4621 root144Tail14 winner.bound = root144Candidates13 := by
  unfold candidates
  have row : edge 4621 = (SortedNatTree.contains FastRows.row4621) := by
    change Data.row4621.contains = (SortedNatTree.contains FastRows.row4621)
    exact FastRows.row4621_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root144Child13 : search edge 3 [853, 4621] root144Candidates13 5474 winner = winner := by decide
theorem root144Step13 : scan edge (search edge 3) 4 [853] 853 root144Tail13 winner =
    scan edge (search edge 3) 4 [853] 853 root144Tail14 winner := by
  rw [root144Tail13, scan, if_neg (by decide), root144Candidates13_checked]
  change scan edge (search edge 3) 4 [853] 853 root144Tail14
    (if 3 ≤ root144Candidates13.length then search edge 3 [853, 4621] root144Candidates13 5474 winner else winner) = _
  rw [if_pos (by decide), root144Child13]
def root144Candidates14 : List Nat := [9679, 17257]
set_option maxRecDepth 8192 in
theorem root144Candidates14_checked : candidates edge 4 853 4639 root144Tail15 winner.bound = root144Candidates14 := by
  unfold candidates
  have row : edge 4639 = (SortedNatTree.contains FastRows.row4639) := by
    change Data.row4639.contains = (SortedNatTree.contains FastRows.row4639)
    exact FastRows.row4639_checked.symm
  rw [row]
  decide
theorem root144Step14 : scan edge (search edge 3) 4 [853] 853 root144Tail14 winner =
    scan edge (search edge 3) 4 [853] 853 root144Tail15 winner := by
  rw [root144Tail14, scan, if_neg (by decide), root144Candidates14_checked]
  change scan edge (search edge 3) 4 [853] 853 root144Tail15
    (if 3 ≤ root144Candidates14.length then search edge 3 [853, 4639] root144Candidates14 5492 winner else winner) = _
  rw [if_neg (by decide)]
def root144Candidates15 : List Nat := [4789]
set_option maxRecDepth 8192 in
theorem root144Candidates15_checked : candidates edge 4 853 4657 root144Tail16 winner.bound = root144Candidates15 := by
  unfold candidates
  have row : edge 4657 = (SortedNatTree.contains FastRows.row4657) := by
    change Data.row4657.contains = (SortedNatTree.contains FastRows.row4657)
    exact FastRows.row4657_checked.symm
  rw [row]
  decide
theorem root144Step15 : scan edge (search edge 3) 4 [853] 853 root144Tail15 winner =
    scan edge (search edge 3) 4 [853] 853 root144Tail16 winner := by
  rw [root144Tail15, scan, if_neg (by decide), root144Candidates15_checked]
  change scan edge (search edge 3) 4 [853] 853 root144Tail16
    (if 3 ≤ root144Candidates15.length then search edge 3 [853, 4657] root144Candidates15 5510 winner else winner) = _
  rw [if_neg (by decide)]
def root144Candidates16 : List Nat := [17209]
set_option maxRecDepth 8192 in
theorem root144Candidates16_checked : candidates edge 4 853 4789 root144Tail17 winner.bound = root144Candidates16 := by
  unfold candidates
  have row : edge 4789 = (SortedNatTree.contains FastRows.row4789) := by
    change Data.row4789.contains = (SortedNatTree.contains FastRows.row4789)
    exact FastRows.row4789_checked.symm
  rw [row]
  decide
theorem root144Step16 : scan edge (search edge 3) 4 [853] 853 root144Tail16 winner =
    scan edge (search edge 3) 4 [853] 853 root144Tail17 winner := by
  rw [root144Tail16, scan, if_neg (by decide), root144Candidates16_checked]
  change scan edge (search edge 3) 4 [853] 853 root144Tail17
    (if 3 ≤ root144Candidates16.length then search edge 3 [853, 4789] root144Candidates16 5642 winner else winner) = _
  rw [if_neg (by decide)]
def root144Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root144Candidates17_checked : candidates edge 4 853 5413 root144Tail18 winner.bound = root144Candidates17 := by
  unfold candidates
  have row : edge 5413 = (SortedNatTree.contains FastRows.row5413) := by
    change Data.row5413.contains = (SortedNatTree.contains FastRows.row5413)
    exact FastRows.row5413_checked.symm
  rw [row]
  decide
theorem root144Step17 : scan edge (search edge 3) 4 [853] 853 root144Tail17 winner =
    scan edge (search edge 3) 4 [853] 853 root144Tail18 winner := by
  rw [root144Tail17, scan, if_neg (by decide), root144Candidates17_checked]
  change scan edge (search edge 3) 4 [853] 853 root144Tail18
    (if 3 ≤ root144Candidates17.length then search edge 3 [853, 5413] root144Candidates17 6266 winner else winner) = _
  rw [if_neg (by decide)]
theorem root144Scan18 : scan edge (search edge 3) 4 [853] 853 root144Tail18 winner = winner := by rw [root144Tail18, scan, if_pos (by decide)]
theorem root144Scan17 : scan edge (search edge 3) 4 [853] 853 root144Tail17 winner = winner :=
  (root144Step17).trans root144Scan18
theorem root144Scan16 : scan edge (search edge 3) 4 [853] 853 root144Tail16 winner = winner :=
  (root144Step16).trans root144Scan17
theorem root144Scan15 : scan edge (search edge 3) 4 [853] 853 root144Tail15 winner = winner :=
  (root144Step15).trans root144Scan16
theorem root144Scan14 : scan edge (search edge 3) 4 [853] 853 root144Tail14 winner = winner :=
  (root144Step14).trans root144Scan15
theorem root144Scan13 : scan edge (search edge 3) 4 [853] 853 root144Tail13 winner = winner :=
  (root144Step13).trans root144Scan14
theorem root144Scan12 : scan edge (search edge 3) 4 [853] 853 root144Tail12 winner = winner :=
  (root144Step12).trans root144Scan13
theorem root144Scan11 : scan edge (search edge 3) 4 [853] 853 root144Tail11 winner = winner :=
  (root144Step11).trans root144Scan12
theorem root144Scan10 : scan edge (search edge 3) 4 [853] 853 root144Tail10 winner = winner :=
  (root144Step10).trans root144Scan11
theorem root144Scan9 : scan edge (search edge 3) 4 [853] 853 root144Tail9 winner = winner :=
  (root144Step9).trans root144Scan10
theorem root144Scan8 : scan edge (search edge 3) 4 [853] 853 root144Tail8 winner = winner :=
  (root144Step8).trans root144Scan9
theorem root144Scan7 : scan edge (search edge 3) 4 [853] 853 root144Tail7 winner = winner :=
  (root144Step7).trans root144Scan8
theorem root144Scan6 : scan edge (search edge 3) 4 [853] 853 root144Tail6 winner = winner :=
  (root144Step6).trans root144Scan7
theorem root144Scan5 : scan edge (search edge 3) 4 [853] 853 root144Tail5 winner = winner :=
  (root144Step5).trans root144Scan6
theorem root144Scan4 : scan edge (search edge 3) 4 [853] 853 root144Tail4 winner = winner :=
  (root144Step4).trans root144Scan5
theorem root144Scan3 : scan edge (search edge 3) 4 [853] 853 root144Tail3 winner = winner :=
  (root144Step3).trans root144Scan4
theorem root144Scan2 : scan edge (search edge 3) 4 [853] 853 root144Tail2 winner = winner :=
  (root144Step2).trans root144Scan3
theorem root144Scan1 : scan edge (search edge 3) 4 [853] 853 root144Tail1 winner = winner :=
  (root144Step1).trans root144Scan2
theorem root144Scan0 : scan edge (search edge 3) 4 [853] 853 root144Tail0 winner = winner :=
  (root144Step0).trans root144Scan1
set_option maxRecDepth 4096 in
theorem root144Branch : search edge 4 [853] root144Candidates 853 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [853] 853 root144Tail0 winner = winner
  exact root144Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
