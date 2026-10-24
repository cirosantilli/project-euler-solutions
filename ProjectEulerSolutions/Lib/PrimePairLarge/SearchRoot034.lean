import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root34Candidates : List Nat := Data.row157
set_option maxRecDepth 8192 in
theorem root34Candidates_checked : candidates edge 5 0 157 tail35 winner.bound = root34Candidates := by
  apply candidates_eq_row edge 5 0 157 tail35 root34Candidates winner.bound tail35_ordered
  · decide
  · decide
  · funext q
    rfl
def root34Tail78 : List Nat := []
def root34Tail77 : List Nat := 25657 :: root34Tail78
def root34Tail76 : List Nat := 25639 :: root34Tail77
def root34Tail75 : List Nat := 25243 :: root34Tail76
def root34Tail74 : List Nat := 25219 :: root34Tail75
def root34Tail73 : List Nat := 24481 :: root34Tail74
def root34Tail72 : List Nat := 24169 :: root34Tail73
def root34Tail71 : List Nat := 23269 :: root34Tail72
def root34Tail70 : List Nat := 22159 :: root34Tail71
def root34Tail69 : List Nat := 21859 :: root34Tail70
def root34Tail68 : List Nat := 21757 :: root34Tail69
def root34Tail67 : List Nat := 21613 :: root34Tail68
def root34Tail66 : List Nat := 21193 :: root34Tail67
def root34Tail65 : List Nat := 21169 :: root34Tail66
def root34Tail64 : List Nat := 20959 :: root34Tail65
def root34Tail63 : List Nat := 20521 :: root34Tail64
def root34Tail62 : List Nat := 20389 :: root34Tail63
def root34Tail61 : List Nat := 20269 :: root34Tail62
def root34Tail60 : List Nat := 19993 :: root34Tail61
def root34Tail59 : List Nat := 19861 :: root34Tail60
def root34Tail58 : List Nat := 19843 :: root34Tail59
def root34Tail57 : List Nat := 19531 :: root34Tail58
def root34Tail56 : List Nat := 19087 :: root34Tail57
def root34Tail55 : List Nat := 18517 :: root34Tail56
def root34Tail54 : List Nat := 17851 :: root34Tail55
def root34Tail53 : List Nat := 17827 :: root34Tail54
def root34Tail52 : List Nat := 17431 :: root34Tail53
def root34Tail51 : List Nat := 16741 :: root34Tail52
def root34Tail50 : List Nat := 16447 :: root34Tail51
def root34Tail49 : List Nat := 16273 :: root34Tail50
def root34Tail48 : List Nat := 16087 :: root34Tail49
def root34Tail47 : List Nat := 15859 :: root34Tail48
def root34Tail46 : List Nat := 15727 :: root34Tail47
def root34Tail45 : List Nat := 15373 :: root34Tail46
def root34Tail44 : List Nat := 15313 :: root34Tail45
def root34Tail43 : List Nat := 15031 :: root34Tail44
def root34Tail42 : List Nat := 14071 :: root34Tail43
def root34Tail41 : List Nat := 13591 :: root34Tail42
def root34Tail40 : List Nat := 12583 :: root34Tail41
def root34Tail39 : List Nat := 12517 :: root34Tail40
def root34Tail38 : List Nat := 11923 :: root34Tail39
def root34Tail37 : List Nat := 11887 :: root34Tail38
def root34Tail36 : List Nat := 11719 :: root34Tail37
def root34Tail35 : List Nat := 11071 :: root34Tail36
def root34Tail34 : List Nat := 10627 :: root34Tail35
def root34Tail33 : List Nat := 10477 :: root34Tail34
def root34Tail32 : List Nat := 9883 :: root34Tail33
def root34Tail31 : List Nat := 9769 :: root34Tail32
def root34Tail30 : List Nat := 9421 :: root34Tail31
def root34Tail29 : List Nat := 9343 :: root34Tail30
def root34Tail28 : List Nat := 9091 :: root34Tail29
def root34Tail27 : List Nat := 8713 :: root34Tail28
def root34Tail26 : List Nat := 7699 :: root34Tail27
def root34Tail25 : List Nat := 7561 :: root34Tail26
def root34Tail24 : List Nat := 7309 :: root34Tail25
def root34Tail23 : List Nat := 6997 :: root34Tail24
def root34Tail22 : List Nat := 6907 :: root34Tail23
def root34Tail21 : List Nat := 6661 :: root34Tail22
def root34Tail20 : List Nat := 6073 :: root34Tail21
def root34Tail19 : List Nat := 5653 :: root34Tail20
def root34Tail18 : List Nat := 4129 :: root34Tail19
def root34Tail17 : List Nat := 3709 :: root34Tail18
def root34Tail16 : List Nat := 3613 :: root34Tail17
def root34Tail15 : List Nat := 3217 :: root34Tail16
def root34Tail14 : List Nat := 3109 :: root34Tail15
def root34Tail13 : List Nat := 3079 :: root34Tail14
def root34Tail12 : List Nat := 2749 :: root34Tail13
def root34Tail11 : List Nat := 2713 :: root34Tail12
def root34Tail10 : List Nat := 2647 :: root34Tail11
def root34Tail9 : List Nat := 2539 :: root34Tail10
def root34Tail8 : List Nat := 2377 :: root34Tail9
def root34Tail7 : List Nat := 2239 :: root34Tail8
def root34Tail6 : List Nat := 1663 :: root34Tail7
def root34Tail5 : List Nat := 1657 :: root34Tail6
def root34Tail4 : List Nat := 733 :: root34Tail5
def root34Tail3 : List Nat := 571 :: root34Tail4
def root34Tail2 : List Nat := 277 :: root34Tail3
def root34Tail1 : List Nat := 229 :: root34Tail2
def root34Tail0 : List Nat := 181 :: root34Tail1
def root34Candidates0 : List Nat := [1657, 11923]
set_option maxRecDepth 8192 in
theorem root34Candidates0_checked : candidates edge 4 157 181 root34Tail1 winner.bound = root34Candidates0 := by
  unfold candidates
  have row : edge 181 = (SortedNatTree.contains FastRows.row181) := by
    change Data.row181.contains = (SortedNatTree.contains FastRows.row181)
    exact FastRows.row181_checked.symm
  rw [row]
  decide
theorem root34Step0 : scan edge (search edge 3) 4 [157] 157 root34Tail0 winner =
    scan edge (search edge 3) 4 [157] 157 root34Tail1 winner := by
  rw [root34Tail0, scan, if_neg (by decide), root34Candidates0_checked]
  change scan edge (search edge 3) 4 [157] 157 root34Tail1
    (if 3 ≤ root34Candidates0.length then search edge 3 [157, 181] root34Candidates0 338 winner else winner) = _
  rw [if_neg (by decide)]
def root34Candidates1 : List Nat := [2239, 15859, 24481]
set_option maxRecDepth 8192 in
theorem root34Candidates1_checked : candidates edge 4 157 229 root34Tail2 winner.bound = root34Candidates1 := by
  unfold candidates
  have row : edge 229 = (SortedNatTree.contains FastRows.row229) := by
    change Data.row229.contains = (SortedNatTree.contains FastRows.row229)
    exact FastRows.row229_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root34Child1 : search edge 3 [157, 229] root34Candidates1 386 winner = winner := by decide
theorem root34Step1 : scan edge (search edge 3) 4 [157] 157 root34Tail1 winner =
    scan edge (search edge 3) 4 [157] 157 root34Tail2 winner := by
  rw [root34Tail1, scan, if_neg (by decide), root34Candidates1_checked]
  change scan edge (search edge 3) 4 [157] 157 root34Tail2
    (if 3 ≤ root34Candidates1.length then search edge 3 [157, 229] root34Candidates1 386 winner else winner) = _
  rw [if_pos (by decide), root34Child1]
def root34Candidates2 : List Nat := [3079]
set_option maxRecDepth 8192 in
theorem root34Candidates2_checked : candidates edge 4 157 277 root34Tail3 winner.bound = root34Candidates2 := by
  unfold candidates
  have row : edge 277 = (SortedNatTree.contains FastRows.row277) := by
    change Data.row277.contains = (SortedNatTree.contains FastRows.row277)
    exact FastRows.row277_checked.symm
  rw [row]
  decide
theorem root34Step2 : scan edge (search edge 3) 4 [157] 157 root34Tail2 winner =
    scan edge (search edge 3) 4 [157] 157 root34Tail3 winner := by
  rw [root34Tail2, scan, if_neg (by decide), root34Candidates2_checked]
  change scan edge (search edge 3) 4 [157] 157 root34Tail3
    (if 3 ≤ root34Candidates2.length then search edge 3 [157, 277] root34Candidates2 434 winner else winner) = _
  rw [if_neg (by decide)]
def root34Candidates3 : List Nat := [2647, 15031, 16273]
set_option maxRecDepth 8192 in
theorem root34Candidates3_checked : candidates edge 4 157 571 root34Tail4 winner.bound = root34Candidates3 := by
  unfold candidates
  have row : edge 571 = (SortedNatTree.contains FastRows.row571) := by
    change Data.row571.contains = (SortedNatTree.contains FastRows.row571)
    exact FastRows.row571_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root34Child3 : search edge 3 [157, 571] root34Candidates3 728 winner = winner := by decide
theorem root34Step3 : scan edge (search edge 3) 4 [157] 157 root34Tail3 winner =
    scan edge (search edge 3) 4 [157] 157 root34Tail4 winner := by
  rw [root34Tail3, scan, if_neg (by decide), root34Candidates3_checked]
  change scan edge (search edge 3) 4 [157] 157 root34Tail4
    (if 3 ≤ root34Candidates3.length then search edge 3 [157, 571] root34Candidates3 728 winner else winner) = _
  rw [if_pos (by decide), root34Child3]
def root34Candidates4 : List Nat := [3709, 9421, 16447, 21757]
set_option maxRecDepth 8192 in
theorem root34Candidates4_checked : candidates edge 4 157 733 root34Tail5 winner.bound = root34Candidates4 := by
  unfold candidates
  have row : edge 733 = (SortedNatTree.contains FastRows.row733) := by
    change Data.row733.contains = (SortedNatTree.contains FastRows.row733)
    exact FastRows.row733_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root34Child4 : search edge 3 [157, 733] root34Candidates4 890 winner = winner := by decide
theorem root34Step4 : scan edge (search edge 3) 4 [157] 157 root34Tail4 winner =
    scan edge (search edge 3) 4 [157] 157 root34Tail5 winner := by
  rw [root34Tail4, scan, if_neg (by decide), root34Candidates4_checked]
  change scan edge (search edge 3) 4 [157] 157 root34Tail5
    (if 3 ≤ root34Candidates4.length then search edge 3 [157, 733] root34Candidates4 890 winner else winner) = _
  rw [if_pos (by decide), root34Child4]
def root34Candidates5 : List Nat := [6907, 10627]
set_option maxRecDepth 8192 in
theorem root34Candidates5_checked : candidates edge 4 157 1657 root34Tail6 winner.bound = root34Candidates5 := by
  unfold candidates
  have row : edge 1657 = (SortedNatTree.contains FastRows.row1657) := by
    change Data.row1657.contains = (SortedNatTree.contains FastRows.row1657)
    exact FastRows.row1657_checked.symm
  rw [row]
  decide
theorem root34Step5 : scan edge (search edge 3) 4 [157] 157 root34Tail5 winner =
    scan edge (search edge 3) 4 [157] 157 root34Tail6 winner := by
  rw [root34Tail5, scan, if_neg (by decide), root34Candidates5_checked]
  change scan edge (search edge 3) 4 [157] 157 root34Tail6
    (if 3 ≤ root34Candidates5.length then search edge 3 [157, 1657] root34Candidates5 1814 winner else winner) = _
  rw [if_neg (by decide)]
def root34Candidates6 : List Nat := [3613, 5653, 9769, 11923, 12583, 15859, 16273]
set_option maxRecDepth 8192 in
theorem root34Candidates6_checked : candidates edge 4 157 1663 root34Tail7 winner.bound = root34Candidates6 := by
  unfold candidates
  have row : edge 1663 = (SortedNatTree.contains FastRows.row1663) := by
    change Data.row1663.contains = (SortedNatTree.contains FastRows.row1663)
    exact FastRows.row1663_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root34Child6 : search edge 3 [157, 1663] root34Candidates6 1820 winner = winner := by decide
theorem root34Step6 : scan edge (search edge 3) 4 [157] 157 root34Tail6 winner =
    scan edge (search edge 3) 4 [157] 157 root34Tail7 winner := by
  rw [root34Tail6, scan, if_neg (by decide), root34Candidates6_checked]
  change scan edge (search edge 3) 4 [157] 157 root34Tail7
    (if 3 ≤ root34Candidates6.length then search edge 3 [157, 1663] root34Candidates6 1820 winner else winner) = _
  rw [if_pos (by decide), root34Child6]
def root34Candidates7 : List Nat := [4129, 11719]
set_option maxRecDepth 8192 in
theorem root34Candidates7_checked : candidates edge 4 157 2239 root34Tail8 winner.bound = root34Candidates7 := by
  unfold candidates
  have row : edge 2239 = (SortedNatTree.contains FastRows.row2239) := by
    change Data.row2239.contains = (SortedNatTree.contains FastRows.row2239)
    exact FastRows.row2239_checked.symm
  rw [row]
  decide
theorem root34Step7 : scan edge (search edge 3) 4 [157] 157 root34Tail7 winner =
    scan edge (search edge 3) 4 [157] 157 root34Tail8 winner := by
  rw [root34Tail7, scan, if_neg (by decide), root34Candidates7_checked]
  change scan edge (search edge 3) 4 [157] 157 root34Tail8
    (if 3 ≤ root34Candidates7.length then search edge 3 [157, 2239] root34Candidates7 2396 winner else winner) = _
  rw [if_neg (by decide)]
def root34Candidates8 : List Nat := [6073, 20521]
set_option maxRecDepth 8192 in
theorem root34Candidates8_checked : candidates edge 4 157 2377 root34Tail9 winner.bound = root34Candidates8 := by
  unfold candidates
  have row : edge 2377 = (SortedNatTree.contains FastRows.row2377) := by
    change Data.row2377.contains = (SortedNatTree.contains FastRows.row2377)
    exact FastRows.row2377_checked.symm
  rw [row]
  decide
theorem root34Step8 : scan edge (search edge 3) 4 [157] 157 root34Tail8 winner =
    scan edge (search edge 3) 4 [157] 157 root34Tail9 winner := by
  rw [root34Tail8, scan, if_neg (by decide), root34Candidates8_checked]
  change scan edge (search edge 3) 4 [157] 157 root34Tail9
    (if 3 ≤ root34Candidates8.length then search edge 3 [157, 2377] root34Candidates8 2534 winner else winner) = _
  rw [if_neg (by decide)]
def root34Candidates9 : List Nat := [19843, 23269]
set_option maxRecDepth 8192 in
theorem root34Candidates9_checked : candidates edge 4 157 2539 root34Tail10 winner.bound = root34Candidates9 := by
  unfold candidates
  have row : edge 2539 = (SortedNatTree.contains FastRows.row2539) := by
    change Data.row2539.contains = (SortedNatTree.contains FastRows.row2539)
    exact FastRows.row2539_checked.symm
  rw [row]
  decide
theorem root34Step9 : scan edge (search edge 3) 4 [157] 157 root34Tail9 winner =
    scan edge (search edge 3) 4 [157] 157 root34Tail10 winner := by
  rw [root34Tail9, scan, if_neg (by decide), root34Candidates9_checked]
  change scan edge (search edge 3) 4 [157] 157 root34Tail10
    (if 3 ≤ root34Candidates9.length then search edge 3 [157, 2539] root34Candidates9 2696 winner else winner) = _
  rw [if_neg (by decide)]
def root34Candidates10 : List Nat := [12583, 14071]
set_option maxRecDepth 8192 in
theorem root34Candidates10_checked : candidates edge 4 157 2647 root34Tail11 winner.bound = root34Candidates10 := by
  unfold candidates
  have row : edge 2647 = (SortedNatTree.contains FastRows.row2647) := by
    change Data.row2647.contains = (SortedNatTree.contains FastRows.row2647)
    exact FastRows.row2647_checked.symm
  rw [row]
  decide
theorem root34Step10 : scan edge (search edge 3) 4 [157] 157 root34Tail10 winner =
    scan edge (search edge 3) 4 [157] 157 root34Tail11 winner := by
  rw [root34Tail10, scan, if_neg (by decide), root34Candidates10_checked]
  change scan edge (search edge 3) 4 [157] 157 root34Tail11
    (if 3 ≤ root34Candidates10.length then search edge 3 [157, 2647] root34Candidates10 2804 winner else winner) = _
  rw [if_neg (by decide)]
def root34Candidates11 : List Nat := [12583, 15313]
set_option maxRecDepth 8192 in
theorem root34Candidates11_checked : candidates edge 4 157 2713 root34Tail12 winner.bound = root34Candidates11 := by
  unfold candidates
  have row : edge 2713 = (SortedNatTree.contains FastRows.row2713) := by
    change Data.row2713.contains = (SortedNatTree.contains FastRows.row2713)
    exact FastRows.row2713_checked.symm
  rw [row]
  decide
theorem root34Step11 : scan edge (search edge 3) 4 [157] 157 root34Tail11 winner =
    scan edge (search edge 3) 4 [157] 157 root34Tail12 winner := by
  rw [root34Tail11, scan, if_neg (by decide), root34Candidates11_checked]
  change scan edge (search edge 3) 4 [157] 157 root34Tail12
    (if 3 ≤ root34Candidates11.length then search edge 3 [157, 2713] root34Candidates11 2870 winner else winner) = _
  rw [if_neg (by decide)]
def root34Candidates12 : List Nat := [10477]
set_option maxRecDepth 8192 in
theorem root34Candidates12_checked : candidates edge 4 157 2749 root34Tail13 winner.bound = root34Candidates12 := by
  unfold candidates
  have row : edge 2749 = (SortedNatTree.contains FastRows.row2749) := by
    change Data.row2749.contains = (SortedNatTree.contains FastRows.row2749)
    exact FastRows.row2749_checked.symm
  rw [row]
  decide
theorem root34Step12 : scan edge (search edge 3) 4 [157] 157 root34Tail12 winner =
    scan edge (search edge 3) 4 [157] 157 root34Tail13 winner := by
  rw [root34Tail12, scan, if_neg (by decide), root34Candidates12_checked]
  change scan edge (search edge 3) 4 [157] 157 root34Tail13
    (if 3 ≤ root34Candidates12.length then search edge 3 [157, 2749] root34Candidates12 2906 winner else winner) = _
  rw [if_neg (by decide)]
def root34Candidates13 : List Nat := [3709, 7561, 7699]
set_option maxRecDepth 8192 in
theorem root34Candidates13_checked : candidates edge 4 157 3079 root34Tail14 winner.bound = root34Candidates13 := by
  unfold candidates
  have row : edge 3079 = (SortedNatTree.contains FastRows.row3079) := by
    change Data.row3079.contains = (SortedNatTree.contains FastRows.row3079)
    exact FastRows.row3079_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root34Child13 : search edge 3 [157, 3079] root34Candidates13 3236 winner = winner := by decide
theorem root34Step13 : scan edge (search edge 3) 4 [157] 157 root34Tail13 winner =
    scan edge (search edge 3) 4 [157] 157 root34Tail14 winner := by
  rw [root34Tail13, scan, if_neg (by decide), root34Candidates13_checked]
  change scan edge (search edge 3) 4 [157] 157 root34Tail14
    (if 3 ≤ root34Candidates13.length then search edge 3 [157, 3079] root34Candidates13 3236 winner else winner) = _
  rw [if_pos (by decide), root34Child13]
def root34Candidates14 : List Nat := [6073, 6997, 9883]
set_option maxRecDepth 8192 in
theorem root34Candidates14_checked : candidates edge 4 157 3109 root34Tail15 winner.bound = root34Candidates14 := by
  unfold candidates
  have row : edge 3109 = (SortedNatTree.contains FastRows.row3109) := by
    change Data.row3109.contains = (SortedNatTree.contains FastRows.row3109)
    exact FastRows.row3109_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root34Child14 : search edge 3 [157, 3109] root34Candidates14 3266 winner = winner := by decide
theorem root34Step14 : scan edge (search edge 3) 4 [157] 157 root34Tail14 winner =
    scan edge (search edge 3) 4 [157] 157 root34Tail15 winner := by
  rw [root34Tail14, scan, if_neg (by decide), root34Candidates14_checked]
  change scan edge (search edge 3) 4 [157] 157 root34Tail15
    (if 3 ≤ root34Candidates14.length then search edge 3 [157, 3109] root34Candidates14 3266 winner else winner) = _
  rw [if_pos (by decide), root34Child14]
def root34Candidates15 : List Nat := [6073]
set_option maxRecDepth 8192 in
theorem root34Candidates15_checked : candidates edge 4 157 3217 root34Tail16 winner.bound = root34Candidates15 := by
  unfold candidates
  have row : edge 3217 = (SortedNatTree.contains FastRows.row3217) := by
    change Data.row3217.contains = (SortedNatTree.contains FastRows.row3217)
    exact FastRows.row3217_checked.symm
  rw [row]
  decide
theorem root34Step15 : scan edge (search edge 3) 4 [157] 157 root34Tail15 winner =
    scan edge (search edge 3) 4 [157] 157 root34Tail16 winner := by
  rw [root34Tail15, scan, if_neg (by decide), root34Candidates15_checked]
  change scan edge (search edge 3) 4 [157] 157 root34Tail16
    (if 3 ≤ root34Candidates15.length then search edge 3 [157, 3217] root34Candidates15 3374 winner else winner) = _
  rw [if_neg (by decide)]
def root34Candidates16 : List Nat := [14071, 15313, 20269]
set_option maxRecDepth 8192 in
theorem root34Candidates16_checked : candidates edge 4 157 3613 root34Tail17 winner.bound = root34Candidates16 := by
  unfold candidates
  have row : edge 3613 = (SortedNatTree.contains FastRows.row3613) := by
    change Data.row3613.contains = (SortedNatTree.contains FastRows.row3613)
    exact FastRows.row3613_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root34Child16 : search edge 3 [157, 3613] root34Candidates16 3770 winner = winner := by decide
theorem root34Step16 : scan edge (search edge 3) 4 [157] 157 root34Tail16 winner =
    scan edge (search edge 3) 4 [157] 157 root34Tail17 winner := by
  rw [root34Tail16, scan, if_neg (by decide), root34Candidates16_checked]
  change scan edge (search edge 3) 4 [157] 157 root34Tail17
    (if 3 ≤ root34Candidates16.length then search edge 3 [157, 3613] root34Candidates16 3770 winner else winner) = _
  rw [if_pos (by decide), root34Child16]
def root34Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root34Candidates17_checked : candidates edge 4 157 3709 root34Tail18 winner.bound = root34Candidates17 := by
  unfold candidates
  have row : edge 3709 = (SortedNatTree.contains FastRows.row3709) := by
    change Data.row3709.contains = (SortedNatTree.contains FastRows.row3709)
    exact FastRows.row3709_checked.symm
  rw [row]
  decide
theorem root34Step17 : scan edge (search edge 3) 4 [157] 157 root34Tail17 winner =
    scan edge (search edge 3) 4 [157] 157 root34Tail18 winner := by
  rw [root34Tail17, scan, if_neg (by decide), root34Candidates17_checked]
  change scan edge (search edge 3) 4 [157] 157 root34Tail18
    (if 3 ≤ root34Candidates17.length then search edge 3 [157, 3709] root34Candidates17 3866 winner else winner) = _
  rw [if_neg (by decide)]
def root34Candidates18 : List Nat := [9091, 9343, 11071, 15859]
set_option maxRecDepth 8192 in
theorem root34Candidates18_checked : candidates edge 4 157 4129 root34Tail19 winner.bound = root34Candidates18 := by
  unfold candidates
  have row : edge 4129 = (SortedNatTree.contains FastRows.row4129) := by
    change Data.row4129.contains = (SortedNatTree.contains FastRows.row4129)
    exact FastRows.row4129_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root34Child18 : search edge 3 [157, 4129] root34Candidates18 4286 winner = winner := by decide
theorem root34Step18 : scan edge (search edge 3) 4 [157] 157 root34Tail18 winner =
    scan edge (search edge 3) 4 [157] 157 root34Tail19 winner := by
  rw [root34Tail18, scan, if_neg (by decide), root34Candidates18_checked]
  change scan edge (search edge 3) 4 [157] 157 root34Tail19
    (if 3 ≤ root34Candidates18.length then search edge 3 [157, 4129] root34Candidates18 4286 winner else winner) = _
  rw [if_pos (by decide), root34Child18]
def root34Candidates19 : List Nat := [13591]
set_option maxRecDepth 8192 in
theorem root34Candidates19_checked : candidates edge 4 157 5653 root34Tail20 winner.bound = root34Candidates19 := by
  unfold candidates
  have row : edge 5653 = (SortedNatTree.contains FastRows.row5653) := by
    change Data.row5653.contains = (SortedNatTree.contains FastRows.row5653)
    exact FastRows.row5653_checked.symm
  rw [row]
  decide
theorem root34Step19 : scan edge (search edge 3) 4 [157] 157 root34Tail19 winner =
    scan edge (search edge 3) 4 [157] 157 root34Tail20 winner := by
  rw [root34Tail19, scan, if_neg (by decide), root34Candidates19_checked]
  change scan edge (search edge 3) 4 [157] 157 root34Tail20
    (if 3 ≤ root34Candidates19.length then search edge 3 [157, 5653] root34Candidates19 5810 winner else winner) = _
  rw [if_neg (by decide)]
def root34Candidates20 : List Nat := [9769]
set_option maxRecDepth 8192 in
theorem root34Candidates20_checked : candidates edge 4 157 6073 root34Tail21 winner.bound = root34Candidates20 := by
  unfold candidates
  have row : edge 6073 = (SortedNatTree.contains FastRows.row6073) := by
    change Data.row6073.contains = (SortedNatTree.contains FastRows.row6073)
    exact FastRows.row6073_checked.symm
  rw [row]
  decide
theorem root34Step20 : scan edge (search edge 3) 4 [157] 157 root34Tail20 winner =
    scan edge (search edge 3) 4 [157] 157 root34Tail21 winner := by
  rw [root34Tail20, scan, if_neg (by decide), root34Candidates20_checked]
  change scan edge (search edge 3) 4 [157] 157 root34Tail21
    (if 3 ≤ root34Candidates20.length then search edge 3 [157, 6073] root34Candidates20 6230 winner else winner) = _
  rw [if_neg (by decide)]
theorem root34Scan21 : scan edge (search edge 3) 4 [157] 157 root34Tail21 winner = winner := by rw [root34Tail21, scan, if_pos (by decide)]
theorem root34Scan20 : scan edge (search edge 3) 4 [157] 157 root34Tail20 winner = winner :=
  (root34Step20).trans root34Scan21
theorem root34Scan19 : scan edge (search edge 3) 4 [157] 157 root34Tail19 winner = winner :=
  (root34Step19).trans root34Scan20
theorem root34Scan18 : scan edge (search edge 3) 4 [157] 157 root34Tail18 winner = winner :=
  (root34Step18).trans root34Scan19
theorem root34Scan17 : scan edge (search edge 3) 4 [157] 157 root34Tail17 winner = winner :=
  (root34Step17).trans root34Scan18
theorem root34Scan16 : scan edge (search edge 3) 4 [157] 157 root34Tail16 winner = winner :=
  (root34Step16).trans root34Scan17
theorem root34Scan15 : scan edge (search edge 3) 4 [157] 157 root34Tail15 winner = winner :=
  (root34Step15).trans root34Scan16
theorem root34Scan14 : scan edge (search edge 3) 4 [157] 157 root34Tail14 winner = winner :=
  (root34Step14).trans root34Scan15
theorem root34Scan13 : scan edge (search edge 3) 4 [157] 157 root34Tail13 winner = winner :=
  (root34Step13).trans root34Scan14
theorem root34Scan12 : scan edge (search edge 3) 4 [157] 157 root34Tail12 winner = winner :=
  (root34Step12).trans root34Scan13
theorem root34Scan11 : scan edge (search edge 3) 4 [157] 157 root34Tail11 winner = winner :=
  (root34Step11).trans root34Scan12
theorem root34Scan10 : scan edge (search edge 3) 4 [157] 157 root34Tail10 winner = winner :=
  (root34Step10).trans root34Scan11
theorem root34Scan9 : scan edge (search edge 3) 4 [157] 157 root34Tail9 winner = winner :=
  (root34Step9).trans root34Scan10
theorem root34Scan8 : scan edge (search edge 3) 4 [157] 157 root34Tail8 winner = winner :=
  (root34Step8).trans root34Scan9
theorem root34Scan7 : scan edge (search edge 3) 4 [157] 157 root34Tail7 winner = winner :=
  (root34Step7).trans root34Scan8
theorem root34Scan6 : scan edge (search edge 3) 4 [157] 157 root34Tail6 winner = winner :=
  (root34Step6).trans root34Scan7
theorem root34Scan5 : scan edge (search edge 3) 4 [157] 157 root34Tail5 winner = winner :=
  (root34Step5).trans root34Scan6
theorem root34Scan4 : scan edge (search edge 3) 4 [157] 157 root34Tail4 winner = winner :=
  (root34Step4).trans root34Scan5
theorem root34Scan3 : scan edge (search edge 3) 4 [157] 157 root34Tail3 winner = winner :=
  (root34Step3).trans root34Scan4
theorem root34Scan2 : scan edge (search edge 3) 4 [157] 157 root34Tail2 winner = winner :=
  (root34Step2).trans root34Scan3
theorem root34Scan1 : scan edge (search edge 3) 4 [157] 157 root34Tail1 winner = winner :=
  (root34Step1).trans root34Scan2
theorem root34Scan0 : scan edge (search edge 3) 4 [157] 157 root34Tail0 winner = winner :=
  (root34Step0).trans root34Scan1
set_option maxRecDepth 4096 in
theorem root34Branch : search edge 4 [157] root34Candidates 157 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [157] 157 root34Tail0 winner = winner
  exact root34Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
