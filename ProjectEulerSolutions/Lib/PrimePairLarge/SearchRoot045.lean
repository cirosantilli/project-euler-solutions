import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root45Candidates : List Nat := Data.row223
set_option maxRecDepth 8192 in
theorem root45Candidates_checked : candidates edge 5 0 223 tail46 winner.bound = root45Candidates := by
  apply candidates_eq_row edge 5 0 223 tail46 root45Candidates winner.bound tail46_ordered
  · decide
  · decide
  · funext q
    rfl
def root45Tail68 : List Nat := []
def root45Tail67 : List Nat := 25153 :: root45Tail68
def root45Tail66 : List Nat := 24979 :: root45Tail67
def root45Tail65 : List Nat := 24763 :: root45Tail66
def root45Tail64 : List Nat := 24571 :: root45Tail65
def root45Tail63 : List Nat := 24097 :: root45Tail64
def root45Tail62 : List Nat := 23893 :: root45Tail63
def root45Tail61 : List Nat := 23563 :: root45Tail62
def root45Tail60 : List Nat := 22501 :: root45Tail61
def root45Tail59 : List Nat := 22303 :: root45Tail60
def root45Tail58 : List Nat := 22171 :: root45Tail59
def root45Tail57 : List Nat := 21577 :: root45Tail58
def root45Tail56 : List Nat := 21319 :: root45Tail57
def root45Tail55 : List Nat := 19993 :: root45Tail56
def root45Tail54 : List Nat := 18757 :: root45Tail55
def root45Tail53 : List Nat := 18691 :: root45Tail54
def root45Tail52 : List Nat := 18523 :: root45Tail53
def root45Tail51 : List Nat := 18427 :: root45Tail52
def root45Tail50 : List Nat := 17377 :: root45Tail51
def root45Tail49 : List Nat := 17299 :: root45Tail50
def root45Tail48 : List Nat := 17257 :: root45Tail49
def root45Tail47 : List Nat := 17203 :: root45Tail48
def root45Tail46 : List Nat := 16843 :: root45Tail47
def root45Tail45 : List Nat := 16633 :: root45Tail46
def root45Tail44 : List Nat := 15919 :: root45Tail45
def root45Tail43 : List Nat := 15859 :: root45Tail44
def root45Tail42 : List Nat := 15739 :: root45Tail43
def root45Tail41 : List Nat := 15331 :: root45Tail42
def root45Tail40 : List Nat := 14869 :: root45Tail41
def root45Tail39 : List Nat := 14779 :: root45Tail40
def root45Tail38 : List Nat := 14533 :: root45Tail39
def root45Tail37 : List Nat := 13729 :: root45Tail38
def root45Tail36 : List Nat := 13339 :: root45Tail37
def root45Tail35 : List Nat := 13009 :: root45Tail36
def root45Tail34 : List Nat := 12853 :: root45Tail35
def root45Tail33 : List Nat := 12799 :: root45Tail34
def root45Tail32 : List Nat := 12421 :: root45Tail33
def root45Tail31 : List Nat := 11743 :: root45Tail32
def root45Tail30 : List Nat := 11593 :: root45Tail31
def root45Tail29 : List Nat := 11083 :: root45Tail30
def root45Tail28 : List Nat := 10837 :: root45Tail29
def root45Tail27 : List Nat := 9649 :: root45Tail28
def root45Tail26 : List Nat := 9157 :: root45Tail27
def root45Tail25 : List Nat := 9007 :: root45Tail26
def root45Tail24 : List Nat := 7927 :: root45Tail25
def root45Tail23 : List Nat := 7537 :: root45Tail24
def root45Tail22 : List Nat := 7309 :: root45Tail23
def root45Tail21 : List Nat := 7159 :: root45Tail22
def root45Tail20 : List Nat := 6907 :: root45Tail21
def root45Tail19 : List Nat := 6781 :: root45Tail20
def root45Tail18 : List Nat := 6373 :: root45Tail19
def root45Tail17 : List Nat := 6133 :: root45Tail18
def root45Tail16 : List Nat := 6121 :: root45Tail17
def root45Tail15 : List Nat := 6007 :: root45Tail16
def root45Tail14 : List Nat := 5647 :: root45Tail15
def root45Tail13 : List Nat := 4261 :: root45Tail14
def root45Tail12 : List Nat := 4159 :: root45Tail13
def root45Tail11 : List Nat := 3547 :: root45Tail12
def root45Tail10 : List Nat := 2551 :: root45Tail11
def root45Tail9 : List Nat := 2437 :: root45Tail10
def root45Tail8 : List Nat := 2179 :: root45Tail9
def root45Tail7 : List Nat := 1447 :: root45Tail8
def root45Tail6 : List Nat := 1093 :: root45Tail7
def root45Tail5 : List Nat := 919 :: root45Tail6
def root45Tail4 : List Nat := 829 :: root45Tail5
def root45Tail3 : List Nat := 547 :: root45Tail4
def root45Tail2 : List Nat := 337 :: root45Tail3
def root45Tail1 : List Nat := 277 :: root45Tail2
def root45Tail0 : List Nat := 229 :: root45Tail1
def root45Candidates0 : List Nat := [547, 15859, 16633, 17299]
set_option maxRecDepth 8192 in
theorem root45Candidates0_checked : candidates edge 4 223 229 root45Tail1 winner.bound = root45Candidates0 := by
  unfold candidates
  have row : edge 229 = (SortedNatTree.contains FastRows.row229) := by
    change Data.row229.contains = (SortedNatTree.contains FastRows.row229)
    exact FastRows.row229_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root45Child0 : search edge 3 [223, 229] root45Candidates0 452 winner = winner := by decide
theorem root45Step0 : scan edge (search edge 3) 4 [223] 223 root45Tail0 winner =
    scan edge (search edge 3) 4 [223] 223 root45Tail1 winner := by
  rw [root45Tail0, scan, if_neg (by decide), root45Candidates0_checked]
  change scan edge (search edge 3) 4 [223] 223 root45Tail1
    (if 3 ≤ root45Candidates0.length then search edge 3 [223, 229] root45Candidates0 452 winner else winner) = _
  rw [if_pos (by decide), root45Child0]
def root45Candidates1 : List Nat := [2437, 17299, 23893]
set_option maxRecDepth 8192 in
theorem root45Candidates1_checked : candidates edge 4 223 277 root45Tail2 winner.bound = root45Candidates1 := by
  unfold candidates
  have row : edge 277 = (SortedNatTree.contains FastRows.row277) := by
    change Data.row277.contains = (SortedNatTree.contains FastRows.row277)
    exact FastRows.row277_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root45Child1 : search edge 3 [223, 277] root45Candidates1 500 winner = winner := by decide
theorem root45Step1 : scan edge (search edge 3) 4 [223] 223 root45Tail1 winner =
    scan edge (search edge 3) 4 [223] 223 root45Tail2 winner := by
  rw [root45Tail1, scan, if_neg (by decide), root45Candidates1_checked]
  change scan edge (search edge 3) 4 [223] 223 root45Tail2
    (if 3 ≤ root45Candidates1.length then search edge 3 [223, 277] root45Candidates1 500 winner else winner) = _
  rw [if_pos (by decide), root45Child1]
def root45Candidates2 : List Nat := [919, 6781, 18757]
set_option maxRecDepth 8192 in
theorem root45Candidates2_checked : candidates edge 4 223 337 root45Tail3 winner.bound = root45Candidates2 := by
  unfold candidates
  have row : edge 337 = (SortedNatTree.contains FastRows.row337) := by
    change Data.row337.contains = (SortedNatTree.contains FastRows.row337)
    exact FastRows.row337_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root45Child2 : search edge 3 [223, 337] root45Candidates2 560 winner = winner := by decide
theorem root45Step2 : scan edge (search edge 3) 4 [223] 223 root45Tail2 winner =
    scan edge (search edge 3) 4 [223] 223 root45Tail3 winner := by
  rw [root45Tail2, scan, if_neg (by decide), root45Candidates2_checked]
  change scan edge (search edge 3) 4 [223] 223 root45Tail3
    (if 3 ≤ root45Candidates2.length then search edge 3 [223, 337] root45Candidates2 560 winner else winner) = _
  rw [if_pos (by decide), root45Child2]
def root45Candidates3 : List Nat := [18757]
set_option maxRecDepth 8192 in
theorem root45Candidates3_checked : candidates edge 4 223 547 root45Tail4 winner.bound = root45Candidates3 := by
  unfold candidates
  have row : edge 547 = (SortedNatTree.contains FastRows.row547) := by
    change Data.row547.contains = (SortedNatTree.contains FastRows.row547)
    exact FastRows.row547_checked.symm
  rw [row]
  decide
theorem root45Step3 : scan edge (search edge 3) 4 [223] 223 root45Tail3 winner =
    scan edge (search edge 3) 4 [223] 223 root45Tail4 winner := by
  rw [root45Tail3, scan, if_neg (by decide), root45Candidates3_checked]
  change scan edge (search edge 3) 4 [223] 223 root45Tail4
    (if 3 ≤ root45Candidates3.length then search edge 3 [223, 547] root45Candidates3 770 winner else winner) = _
  rw [if_neg (by decide)]
def root45Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root45Candidates4_checked : candidates edge 4 223 829 root45Tail5 winner.bound = root45Candidates4 := by
  unfold candidates
  have row : edge 829 = (SortedNatTree.contains FastRows.row829) := by
    change Data.row829.contains = (SortedNatTree.contains FastRows.row829)
    exact FastRows.row829_checked.symm
  rw [row]
  decide
theorem root45Step4 : scan edge (search edge 3) 4 [223] 223 root45Tail4 winner =
    scan edge (search edge 3) 4 [223] 223 root45Tail5 winner := by
  rw [root45Tail4, scan, if_neg (by decide), root45Candidates4_checked]
  change scan edge (search edge 3) 4 [223] 223 root45Tail5
    (if 3 ≤ root45Candidates4.length then search edge 3 [223, 829] root45Candidates4 1052 winner else winner) = _
  rw [if_neg (by decide)]
def root45Candidates5 : List Nat := [9649]
set_option maxRecDepth 8192 in
theorem root45Candidates5_checked : candidates edge 4 223 919 root45Tail6 winner.bound = root45Candidates5 := by
  unfold candidates
  have row : edge 919 = (SortedNatTree.contains FastRows.row919) := by
    change Data.row919.contains = (SortedNatTree.contains FastRows.row919)
    exact FastRows.row919_checked.symm
  rw [row]
  decide
theorem root45Step5 : scan edge (search edge 3) 4 [223] 223 root45Tail5 winner =
    scan edge (search edge 3) 4 [223] 223 root45Tail6 winner := by
  rw [root45Tail5, scan, if_neg (by decide), root45Candidates5_checked]
  change scan edge (search edge 3) 4 [223] 223 root45Tail6
    (if 3 ≤ root45Candidates5.length then search edge 3 [223, 919] root45Candidates5 1142 winner else winner) = _
  rw [if_neg (by decide)]
def root45Candidates6 : List Nat := [2437, 7309, 15919, 18523, 19993]
set_option maxRecDepth 8192 in
theorem root45Candidates6_checked : candidates edge 4 223 1093 root45Tail7 winner.bound = root45Candidates6 := by
  unfold candidates
  have row : edge 1093 = (SortedNatTree.contains FastRows.row1093) := by
    change Data.row1093.contains = (SortedNatTree.contains FastRows.row1093)
    exact FastRows.row1093_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root45Child6 : search edge 3 [223, 1093] root45Candidates6 1316 winner = winner := by decide
theorem root45Step6 : scan edge (search edge 3) 4 [223] 223 root45Tail6 winner =
    scan edge (search edge 3) 4 [223] 223 root45Tail7 winner := by
  rw [root45Tail6, scan, if_neg (by decide), root45Candidates6_checked]
  change scan edge (search edge 3) 4 [223] 223 root45Tail7
    (if 3 ≤ root45Candidates6.length then search edge 3 [223, 1093] root45Candidates6 1316 winner else winner) = _
  rw [if_pos (by decide), root45Child6]
def root45Candidates7 : List Nat := [15331, 16633, 18523]
set_option maxRecDepth 8192 in
theorem root45Candidates7_checked : candidates edge 4 223 1447 root45Tail8 winner.bound = root45Candidates7 := by
  unfold candidates
  have row : edge 1447 = (SortedNatTree.contains FastRows.row1447) := by
    change Data.row1447.contains = (SortedNatTree.contains FastRows.row1447)
    exact FastRows.row1447_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root45Child7 : search edge 3 [223, 1447] root45Candidates7 1670 winner = winner := by decide
theorem root45Step7 : scan edge (search edge 3) 4 [223] 223 root45Tail7 winner =
    scan edge (search edge 3) 4 [223] 223 root45Tail8 winner := by
  rw [root45Tail7, scan, if_neg (by decide), root45Candidates7_checked]
  change scan edge (search edge 3) 4 [223] 223 root45Tail8
    (if 3 ≤ root45Candidates7.length then search edge 3 [223, 1447] root45Candidates7 1670 winner else winner) = _
  rw [if_pos (by decide), root45Child7]
def root45Candidates8 : List Nat := [4261, 5647, 16843]
set_option maxRecDepth 8192 in
theorem root45Candidates8_checked : candidates edge 4 223 2179 root45Tail9 winner.bound = root45Candidates8 := by
  unfold candidates
  have row : edge 2179 = (SortedNatTree.contains FastRows.row2179) := by
    change Data.row2179.contains = (SortedNatTree.contains FastRows.row2179)
    exact FastRows.row2179_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root45Child8 : search edge 3 [223, 2179] root45Candidates8 2402 winner = winner := by decide
theorem root45Step8 : scan edge (search edge 3) 4 [223] 223 root45Tail8 winner =
    scan edge (search edge 3) 4 [223] 223 root45Tail9 winner := by
  rw [root45Tail8, scan, if_neg (by decide), root45Candidates8_checked]
  change scan edge (search edge 3) 4 [223] 223 root45Tail9
    (if 3 ≤ root45Candidates8.length then search edge 3 [223, 2179] root45Candidates8 2402 winner else winner) = _
  rw [if_pos (by decide), root45Child8]
def root45Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root45Candidates9_checked : candidates edge 4 223 2437 root45Tail10 winner.bound = root45Candidates9 := by
  unfold candidates
  have row : edge 2437 = (SortedNatTree.contains FastRows.row2437) := by
    change Data.row2437.contains = (SortedNatTree.contains FastRows.row2437)
    exact FastRows.row2437_checked.symm
  rw [row]
  decide
theorem root45Step9 : scan edge (search edge 3) 4 [223] 223 root45Tail9 winner =
    scan edge (search edge 3) 4 [223] 223 root45Tail10 winner := by
  rw [root45Tail9, scan, if_neg (by decide), root45Candidates9_checked]
  change scan edge (search edge 3) 4 [223] 223 root45Tail10
    (if 3 ≤ root45Candidates9.length then search edge 3 [223, 2437] root45Candidates9 2660 winner else winner) = _
  rw [if_neg (by decide)]
def root45Candidates10 : List Nat := [7537, 11743, 12799, 17377, 22171]
set_option maxRecDepth 8192 in
theorem root45Candidates10_checked : candidates edge 4 223 2551 root45Tail11 winner.bound = root45Candidates10 := by
  unfold candidates
  have row : edge 2551 = (SortedNatTree.contains FastRows.row2551) := by
    change Data.row2551.contains = (SortedNatTree.contains FastRows.row2551)
    exact FastRows.row2551_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root45Child10 : search edge 3 [223, 2551] root45Candidates10 2774 winner = winner := by decide
theorem root45Step10 : scan edge (search edge 3) 4 [223] 223 root45Tail10 winner =
    scan edge (search edge 3) 4 [223] 223 root45Tail11 winner := by
  rw [root45Tail10, scan, if_neg (by decide), root45Candidates10_checked]
  change scan edge (search edge 3) 4 [223] 223 root45Tail11
    (if 3 ≤ root45Candidates10.length then search edge 3 [223, 2551] root45Candidates10 2774 winner else winner) = _
  rw [if_pos (by decide), root45Child10]
def root45Candidates11 : List Nat := [16633]
set_option maxRecDepth 8192 in
theorem root45Candidates11_checked : candidates edge 4 223 3547 root45Tail12 winner.bound = root45Candidates11 := by
  unfold candidates
  have row : edge 3547 = (SortedNatTree.contains FastRows.row3547) := by
    change Data.row3547.contains = (SortedNatTree.contains FastRows.row3547)
    exact FastRows.row3547_checked.symm
  rw [row]
  decide
theorem root45Step11 : scan edge (search edge 3) 4 [223] 223 root45Tail11 winner =
    scan edge (search edge 3) 4 [223] 223 root45Tail12 winner := by
  rw [root45Tail11, scan, if_neg (by decide), root45Candidates11_checked]
  change scan edge (search edge 3) 4 [223] 223 root45Tail12
    (if 3 ≤ root45Candidates11.length then search edge 3 [223, 3547] root45Candidates11 3770 winner else winner) = _
  rw [if_neg (by decide)]
def root45Candidates12 : List Nat := [18757]
set_option maxRecDepth 8192 in
theorem root45Candidates12_checked : candidates edge 4 223 4159 root45Tail13 winner.bound = root45Candidates12 := by
  unfold candidates
  have row : edge 4159 = (SortedNatTree.contains FastRows.row4159) := by
    change Data.row4159.contains = (SortedNatTree.contains FastRows.row4159)
    exact FastRows.row4159_checked.symm
  rw [row]
  decide
theorem root45Step12 : scan edge (search edge 3) 4 [223] 223 root45Tail12 winner =
    scan edge (search edge 3) 4 [223] 223 root45Tail13 winner := by
  rw [root45Tail12, scan, if_neg (by decide), root45Candidates12_checked]
  change scan edge (search edge 3) 4 [223] 223 root45Tail13
    (if 3 ≤ root45Candidates12.length then search edge 3 [223, 4159] root45Candidates12 4382 winner else winner) = _
  rw [if_neg (by decide)]
def root45Candidates13 : List Nat := [14869, 16843]
set_option maxRecDepth 8192 in
theorem root45Candidates13_checked : candidates edge 4 223 4261 root45Tail14 winner.bound = root45Candidates13 := by
  unfold candidates
  have row : edge 4261 = (SortedNatTree.contains FastRows.row4261) := by
    change Data.row4261.contains = (SortedNatTree.contains FastRows.row4261)
    exact FastRows.row4261_checked.symm
  rw [row]
  decide
theorem root45Step13 : scan edge (search edge 3) 4 [223] 223 root45Tail13 winner =
    scan edge (search edge 3) 4 [223] 223 root45Tail14 winner := by
  rw [root45Tail13, scan, if_neg (by decide), root45Candidates13_checked]
  change scan edge (search edge 3) 4 [223] 223 root45Tail14
    (if 3 ≤ root45Candidates13.length then search edge 3 [223, 4261] root45Candidates13 4484 winner else winner) = _
  rw [if_neg (by decide)]
def root45Candidates14 : List Nat := [6373, 13729, 14779, 15331, 18523]
set_option maxRecDepth 8192 in
theorem root45Candidates14_checked : candidates edge 4 223 5647 root45Tail15 winner.bound = root45Candidates14 := by
  unfold candidates
  have row : edge 5647 = (SortedNatTree.contains FastRows.row5647) := by
    change Data.row5647.contains = (SortedNatTree.contains FastRows.row5647)
    exact FastRows.row5647_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root45Child14 : search edge 3 [223, 5647] root45Candidates14 5870 winner = winner := by decide
theorem root45Step14 : scan edge (search edge 3) 4 [223] 223 root45Tail14 winner =
    scan edge (search edge 3) 4 [223] 223 root45Tail15 winner := by
  rw [root45Tail14, scan, if_neg (by decide), root45Candidates14_checked]
  change scan edge (search edge 3) 4 [223] 223 root45Tail15
    (if 3 ≤ root45Candidates14.length then search edge 3 [223, 5647] root45Candidates14 5870 winner else winner) = _
  rw [if_pos (by decide), root45Child14]
def root45Candidates15 : List Nat := [10837]
set_option maxRecDepth 8192 in
theorem root45Candidates15_checked : candidates edge 4 223 6007 root45Tail16 winner.bound = root45Candidates15 := by
  unfold candidates
  have row : edge 6007 = (SortedNatTree.contains FastRows.row6007) := by
    change Data.row6007.contains = (SortedNatTree.contains FastRows.row6007)
    exact FastRows.row6007_checked.symm
  rw [row]
  decide
theorem root45Step15 : scan edge (search edge 3) 4 [223] 223 root45Tail15 winner =
    scan edge (search edge 3) 4 [223] 223 root45Tail16 winner := by
  rw [root45Tail15, scan, if_neg (by decide), root45Candidates15_checked]
  change scan edge (search edge 3) 4 [223] 223 root45Tail16
    (if 3 ≤ root45Candidates15.length then search edge 3 [223, 6007] root45Candidates15 6230 winner else winner) = _
  rw [if_neg (by decide)]
def root45Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root45Candidates16_checked : candidates edge 4 223 6121 root45Tail17 winner.bound = root45Candidates16 := by
  unfold candidates
  have row : edge 6121 = (SortedNatTree.contains FastRows.row6121) := by
    change Data.row6121.contains = (SortedNatTree.contains FastRows.row6121)
    exact FastRows.row6121_checked.symm
  rw [row]
  decide
theorem root45Step16 : scan edge (search edge 3) 4 [223] 223 root45Tail16 winner =
    scan edge (search edge 3) 4 [223] 223 root45Tail17 winner := by
  rw [root45Tail16, scan, if_neg (by decide), root45Candidates16_checked]
  change scan edge (search edge 3) 4 [223] 223 root45Tail17
    (if 3 ≤ root45Candidates16.length then search edge 3 [223, 6121] root45Candidates16 6344 winner else winner) = _
  rw [if_neg (by decide)]
def root45Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root45Candidates17_checked : candidates edge 4 223 6133 root45Tail18 winner.bound = root45Candidates17 := by
  unfold candidates
  have row : edge 6133 = (SortedNatTree.contains FastRows.row6133) := by
    change Data.row6133.contains = (SortedNatTree.contains FastRows.row6133)
    exact FastRows.row6133_checked.symm
  rw [row]
  decide
theorem root45Step17 : scan edge (search edge 3) 4 [223] 223 root45Tail17 winner =
    scan edge (search edge 3) 4 [223] 223 root45Tail18 winner := by
  rw [root45Tail17, scan, if_neg (by decide), root45Candidates17_checked]
  change scan edge (search edge 3) 4 [223] 223 root45Tail18
    (if 3 ≤ root45Candidates17.length then search edge 3 [223, 6133] root45Candidates17 6356 winner else winner) = _
  rw [if_neg (by decide)]
def root45Candidates18 : List Nat := [6781, 11083]
set_option maxRecDepth 8192 in
theorem root45Candidates18_checked : candidates edge 4 223 6373 root45Tail19 winner.bound = root45Candidates18 := by
  unfold candidates
  have row : edge 6373 = (SortedNatTree.contains FastRows.row6373) := by
    change Data.row6373.contains = (SortedNatTree.contains FastRows.row6373)
    exact FastRows.row6373_checked.symm
  rw [row]
  decide
theorem root45Step18 : scan edge (search edge 3) 4 [223] 223 root45Tail18 winner =
    scan edge (search edge 3) 4 [223] 223 root45Tail19 winner := by
  rw [root45Tail18, scan, if_neg (by decide), root45Candidates18_checked]
  change scan edge (search edge 3) 4 [223] 223 root45Tail19
    (if 3 ≤ root45Candidates18.length then search edge 3 [223, 6373] root45Candidates18 6596 winner else winner) = _
  rw [if_neg (by decide)]
theorem root45Scan19 : scan edge (search edge 3) 4 [223] 223 root45Tail19 winner = winner := by rw [root45Tail19, scan, if_pos (by decide)]
theorem root45Scan18 : scan edge (search edge 3) 4 [223] 223 root45Tail18 winner = winner :=
  (root45Step18).trans root45Scan19
theorem root45Scan17 : scan edge (search edge 3) 4 [223] 223 root45Tail17 winner = winner :=
  (root45Step17).trans root45Scan18
theorem root45Scan16 : scan edge (search edge 3) 4 [223] 223 root45Tail16 winner = winner :=
  (root45Step16).trans root45Scan17
theorem root45Scan15 : scan edge (search edge 3) 4 [223] 223 root45Tail15 winner = winner :=
  (root45Step15).trans root45Scan16
theorem root45Scan14 : scan edge (search edge 3) 4 [223] 223 root45Tail14 winner = winner :=
  (root45Step14).trans root45Scan15
theorem root45Scan13 : scan edge (search edge 3) 4 [223] 223 root45Tail13 winner = winner :=
  (root45Step13).trans root45Scan14
theorem root45Scan12 : scan edge (search edge 3) 4 [223] 223 root45Tail12 winner = winner :=
  (root45Step12).trans root45Scan13
theorem root45Scan11 : scan edge (search edge 3) 4 [223] 223 root45Tail11 winner = winner :=
  (root45Step11).trans root45Scan12
theorem root45Scan10 : scan edge (search edge 3) 4 [223] 223 root45Tail10 winner = winner :=
  (root45Step10).trans root45Scan11
theorem root45Scan9 : scan edge (search edge 3) 4 [223] 223 root45Tail9 winner = winner :=
  (root45Step9).trans root45Scan10
theorem root45Scan8 : scan edge (search edge 3) 4 [223] 223 root45Tail8 winner = winner :=
  (root45Step8).trans root45Scan9
theorem root45Scan7 : scan edge (search edge 3) 4 [223] 223 root45Tail7 winner = winner :=
  (root45Step7).trans root45Scan8
theorem root45Scan6 : scan edge (search edge 3) 4 [223] 223 root45Tail6 winner = winner :=
  (root45Step6).trans root45Scan7
theorem root45Scan5 : scan edge (search edge 3) 4 [223] 223 root45Tail5 winner = winner :=
  (root45Step5).trans root45Scan6
theorem root45Scan4 : scan edge (search edge 3) 4 [223] 223 root45Tail4 winner = winner :=
  (root45Step4).trans root45Scan5
theorem root45Scan3 : scan edge (search edge 3) 4 [223] 223 root45Tail3 winner = winner :=
  (root45Step3).trans root45Scan4
theorem root45Scan2 : scan edge (search edge 3) 4 [223] 223 root45Tail2 winner = winner :=
  (root45Step2).trans root45Scan3
theorem root45Scan1 : scan edge (search edge 3) 4 [223] 223 root45Tail1 winner = winner :=
  (root45Step1).trans root45Scan2
theorem root45Scan0 : scan edge (search edge 3) 4 [223] 223 root45Tail0 winner = winner :=
  (root45Step0).trans root45Scan1
set_option maxRecDepth 4096 in
theorem root45Branch : search edge 4 [223] root45Candidates 223 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [223] 223 root45Tail0 winner = winner
  exact root45Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
