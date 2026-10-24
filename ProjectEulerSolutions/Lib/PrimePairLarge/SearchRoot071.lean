import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root71Candidates : List Nat := Data.row373
set_option maxRecDepth 8192 in
theorem root71Candidates_checked : candidates edge 5 0 373 tail72 winner.bound = root71Candidates := by
  apply candidates_eq_row edge 5 0 373 tail72 root71Candidates winner.bound tail72_ordered
  · decide
  · decide
  · funext q
    rfl
def root71Tail66 : List Nat := []
def root71Tail65 : List Nat := 25423 :: root71Tail66
def root71Tail64 : List Nat := 24841 :: root71Tail65
def root71Tail63 : List Nat := 24799 :: root71Tail64
def root71Tail62 : List Nat := 23497 :: root71Tail63
def root71Tail61 : List Nat := 23371 :: root71Tail62
def root71Tail60 : List Nat := 22189 :: root71Tail61
def root71Tail59 : List Nat := 21841 :: root71Tail60
def root71Tail58 : List Nat := 21517 :: root71Tail59
def root71Tail57 : List Nat := 20929 :: root71Tail58
def root71Tail56 : List Nat := 20599 :: root71Tail57
def root71Tail55 : List Nat := 20011 :: root71Tail56
def root71Tail54 : List Nat := 19993 :: root71Tail55
def root71Tail53 : List Nat := 19423 :: root71Tail54
def root71Tail52 : List Nat := 18973 :: root71Tail53
def root71Tail51 : List Nat := 18913 :: root71Tail52
def root71Tail50 : List Nat := 18427 :: root71Tail51
def root71Tail49 : List Nat := 18289 :: root71Tail50
def root71Tail48 : List Nat := 18217 :: root71Tail49
def root71Tail47 : List Nat := 17659 :: root71Tail48
def root71Tail46 : List Nat := 17491 :: root71Tail47
def root71Tail45 : List Nat := 17443 :: root71Tail46
def root71Tail44 : List Nat := 16693 :: root71Tail45
def root71Tail43 : List Nat := 16633 :: root71Tail44
def root71Tail42 : List Nat := 15679 :: root71Tail43
def root71Tail41 : List Nat := 15559 :: root71Tail42
def root71Tail40 : List Nat := 14923 :: root71Tail41
def root71Tail39 : List Nat := 14419 :: root71Tail40
def root71Tail38 : List Nat := 14341 :: root71Tail39
def root71Tail37 : List Nat := 14323 :: root71Tail38
def root71Tail36 : List Nat := 14071 :: root71Tail37
def root71Tail35 : List Nat := 13513 :: root71Tail36
def root71Tail34 : List Nat := 13063 :: root71Tail35
def root71Tail33 : List Nat := 12097 :: root71Tail34
def root71Tail32 : List Nat := 11971 :: root71Tail33
def root71Tail31 : List Nat := 11887 :: root71Tail32
def root71Tail30 : List Nat := 10657 :: root71Tail31
def root71Tail29 : List Nat := 10459 :: root71Tail30
def root71Tail28 : List Nat := 10099 :: root71Tail29
def root71Tail27 : List Nat := 9859 :: root71Tail28
def root71Tail26 : List Nat := 9661 :: root71Tail27
def root71Tail25 : List Nat := 9391 :: root71Tail26
def root71Tail24 : List Nat := 7417 :: root71Tail25
def root71Tail23 : List Nat := 7351 :: root71Tail24
def root71Tail22 : List Nat := 7057 :: root71Tail23
def root71Tail21 : List Nat := 6823 :: root71Tail22
def root71Tail20 : List Nat := 6793 :: root71Tail21
def root71Tail19 : List Nat := 6163 :: root71Tail20
def root71Tail18 : List Nat := 6133 :: root71Tail19
def root71Tail17 : List Nat := 5881 :: root71Tail18
def root71Tail16 : List Nat := 5449 :: root71Tail17
def root71Tail15 : List Nat := 5101 :: root71Tail16
def root71Tail14 : List Nat := 4957 :: root71Tail15
def root71Tail13 : List Nat := 4903 :: root71Tail14
def root71Tail12 : List Nat := 4519 :: root71Tail13
def root71Tail11 : List Nat := 3967 :: root71Tail12
def root71Tail10 : List Nat := 2467 :: root71Tail11
def root71Tail9 : List Nat := 2161 :: root71Tail10
def root71Tail8 : List Nat := 1867 :: root71Tail9
def root71Tail7 : List Nat := 1531 :: root71Tail8
def root71Tail6 : List Nat := 1447 :: root71Tail7
def root71Tail5 : List Nat := 1237 :: root71Tail6
def root71Tail4 : List Nat := 1009 :: root71Tail5
def root71Tail3 : List Nat := 937 :: root71Tail4
def root71Tail2 : List Nat := 859 :: root71Tail3
def root71Tail1 : List Nat := 823 :: root71Tail2
def root71Tail0 : List Nat := 661 :: root71Tail1
def root71Candidates0 : List Nat := [1237, 1447, 20011, 21841]
set_option maxRecDepth 8192 in
theorem root71Candidates0_checked : candidates edge 4 373 661 root71Tail1 winner.bound = root71Candidates0 := by
  unfold candidates
  have row : edge 661 = (SortedNatTree.contains FastRows.row661) := by
    change Data.row661.contains = (SortedNatTree.contains FastRows.row661)
    exact FastRows.row661_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root71Child0 : search edge 3 [373, 661] root71Candidates0 1034 winner = winner := by decide
theorem root71Step0 : scan edge (search edge 3) 4 [373] 373 root71Tail0 winner =
    scan edge (search edge 3) 4 [373] 373 root71Tail1 winner := by
  rw [root71Tail0, scan, if_neg (by decide), root71Candidates0_checked]
  change scan edge (search edge 3) 4 [373] 373 root71Tail1
    (if 3 ≤ root71Candidates0.length then search edge 3 [373, 661] root71Candidates0 1034 winner else winner) = _
  rw [if_pos (by decide), root71Child0]
def root71Candidates1 : List Nat := [5881, 10099, 15559, 17659, 18427, 20599]
set_option maxRecDepth 8192 in
theorem root71Candidates1_checked : candidates edge 4 373 823 root71Tail2 winner.bound = root71Candidates1 := by
  unfold candidates
  have row : edge 823 = (SortedNatTree.contains FastRows.row823) := by
    change Data.row823.contains = (SortedNatTree.contains FastRows.row823)
    exact FastRows.row823_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root71Child1 : search edge 3 [373, 823] root71Candidates1 1196 winner = winner := by decide
theorem root71Step1 : scan edge (search edge 3) 4 [373] 373 root71Tail1 winner =
    scan edge (search edge 3) 4 [373] 373 root71Tail2 winner := by
  rw [root71Tail1, scan, if_neg (by decide), root71Candidates1_checked]
  change scan edge (search edge 3) 4 [373] 373 root71Tail2
    (if 3 ≤ root71Candidates1.length then search edge 3 [373, 823] root71Candidates1 1196 winner else winner) = _
  rw [if_pos (by decide), root71Child1]
def root71Candidates2 : List Nat := [7417, 10459, 22189]
set_option maxRecDepth 8192 in
theorem root71Candidates2_checked : candidates edge 4 373 859 root71Tail3 winner.bound = root71Candidates2 := by
  unfold candidates
  have row : edge 859 = (SortedNatTree.contains FastRows.row859) := by
    change Data.row859.contains = (SortedNatTree.contains FastRows.row859)
    exact FastRows.row859_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root71Child2 : search edge 3 [373, 859] root71Candidates2 1232 winner = winner := by decide
theorem root71Step2 : scan edge (search edge 3) 4 [373] 373 root71Tail2 winner =
    scan edge (search edge 3) 4 [373] 373 root71Tail3 winner := by
  rw [root71Tail2, scan, if_neg (by decide), root71Candidates2_checked]
  change scan edge (search edge 3) 4 [373] 373 root71Tail3
    (if 3 ≤ root71Candidates2.length then search edge 3 [373, 859] root71Candidates2 1232 winner else winner) = _
  rw [if_pos (by decide), root71Child2]
def root71Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root71Candidates3_checked : candidates edge 4 373 937 root71Tail4 winner.bound = root71Candidates3 := by
  unfold candidates
  have row : edge 937 = (SortedNatTree.contains FastRows.row937) := by
    change Data.row937.contains = (SortedNatTree.contains FastRows.row937)
    exact FastRows.row937_checked.symm
  rw [row]
  decide
theorem root71Step3 : scan edge (search edge 3) 4 [373] 373 root71Tail3 winner =
    scan edge (search edge 3) 4 [373] 373 root71Tail4 winner := by
  rw [root71Tail3, scan, if_neg (by decide), root71Candidates3_checked]
  change scan edge (search edge 3) 4 [373] 373 root71Tail4
    (if 3 ≤ root71Candidates3.length then search edge 3 [373, 937] root71Candidates3 1310 winner else winner) = _
  rw [if_neg (by decide)]
def root71Candidates4 : List Nat := [21517]
set_option maxRecDepth 8192 in
theorem root71Candidates4_checked : candidates edge 4 373 1009 root71Tail5 winner.bound = root71Candidates4 := by
  unfold candidates
  have row : edge 1009 = (SortedNatTree.contains FastRows.row1009) := by
    change Data.row1009.contains = (SortedNatTree.contains FastRows.row1009)
    exact FastRows.row1009_checked.symm
  rw [row]
  decide
theorem root71Step4 : scan edge (search edge 3) 4 [373] 373 root71Tail4 winner =
    scan edge (search edge 3) 4 [373] 373 root71Tail5 winner := by
  rw [root71Tail4, scan, if_neg (by decide), root71Candidates4_checked]
  change scan edge (search edge 3) 4 [373] 373 root71Tail5
    (if 3 ≤ root71Candidates4.length then search edge 3 [373, 1009] root71Candidates4 1382 winner else winner) = _
  rw [if_neg (by decide)]
def root71Candidates5 : List Nat := [20011]
set_option maxRecDepth 8192 in
theorem root71Candidates5_checked : candidates edge 4 373 1237 root71Tail6 winner.bound = root71Candidates5 := by
  unfold candidates
  have row : edge 1237 = (SortedNatTree.contains FastRows.row1237) := by
    change Data.row1237.contains = (SortedNatTree.contains FastRows.row1237)
    exact FastRows.row1237_checked.symm
  rw [row]
  decide
theorem root71Step5 : scan edge (search edge 3) 4 [373] 373 root71Tail5 winner =
    scan edge (search edge 3) 4 [373] 373 root71Tail6 winner := by
  rw [root71Tail5, scan, if_neg (by decide), root71Candidates5_checked]
  change scan edge (search edge 3) 4 [373] 373 root71Tail6
    (if 3 ≤ root71Candidates5.length then search edge 3 [373, 1237] root71Candidates5 1610 winner else winner) = _
  rw [if_neg (by decide)]
def root71Candidates6 : List Nat := [11887, 16633, 20599, 22189]
set_option maxRecDepth 8192 in
theorem root71Candidates6_checked : candidates edge 4 373 1447 root71Tail7 winner.bound = root71Candidates6 := by
  unfold candidates
  have row : edge 1447 = (SortedNatTree.contains FastRows.row1447) := by
    change Data.row1447.contains = (SortedNatTree.contains FastRows.row1447)
    exact FastRows.row1447_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root71Child6 : search edge 3 [373, 1447] root71Candidates6 1820 winner = winner := by decide
theorem root71Step6 : scan edge (search edge 3) 4 [373] 373 root71Tail6 winner =
    scan edge (search edge 3) 4 [373] 373 root71Tail7 winner := by
  rw [root71Tail6, scan, if_neg (by decide), root71Candidates6_checked]
  change scan edge (search edge 3) 4 [373] 373 root71Tail7
    (if 3 ≤ root71Candidates6.length then search edge 3 [373, 1447] root71Candidates6 1820 winner else winner) = _
  rw [if_pos (by decide), root71Child6]
def root71Candidates7 : List Nat := [2467, 17443, 21517]
set_option maxRecDepth 8192 in
theorem root71Candidates7_checked : candidates edge 4 373 1531 root71Tail8 winner.bound = root71Candidates7 := by
  unfold candidates
  have row : edge 1531 = (SortedNatTree.contains FastRows.row1531) := by
    change Data.row1531.contains = (SortedNatTree.contains FastRows.row1531)
    exact FastRows.row1531_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root71Child7 : search edge 3 [373, 1531] root71Candidates7 1904 winner = winner := by decide
theorem root71Step7 : scan edge (search edge 3) 4 [373] 373 root71Tail7 winner =
    scan edge (search edge 3) 4 [373] 373 root71Tail8 winner := by
  rw [root71Tail7, scan, if_neg (by decide), root71Candidates7_checked]
  change scan edge (search edge 3) 4 [373] 373 root71Tail8
    (if 3 ≤ root71Candidates7.length then search edge 3 [373, 1531] root71Candidates7 1904 winner else winner) = _
  rw [if_pos (by decide), root71Child7]
def root71Candidates8 : List Nat := [11887, 14923, 16693, 18913, 23371]
set_option maxRecDepth 8192 in
theorem root71Candidates8_checked : candidates edge 4 373 1867 root71Tail9 winner.bound = root71Candidates8 := by
  unfold candidates
  have row : edge 1867 = (SortedNatTree.contains FastRows.row1867) := by
    change Data.row1867.contains = (SortedNatTree.contains FastRows.row1867)
    exact FastRows.row1867_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root71Child8 : search edge 3 [373, 1867] root71Candidates8 2240 winner = winner := by decide
theorem root71Step8 : scan edge (search edge 3) 4 [373] 373 root71Tail8 winner =
    scan edge (search edge 3) 4 [373] 373 root71Tail9 winner := by
  rw [root71Tail8, scan, if_neg (by decide), root71Candidates8_checked]
  change scan edge (search edge 3) 4 [373] 373 root71Tail9
    (if 3 ≤ root71Candidates8.length then search edge 3 [373, 1867] root71Candidates8 2240 winner else winner) = _
  rw [if_pos (by decide), root71Child8]
def root71Candidates9 : List Nat := [18289]
set_option maxRecDepth 8192 in
theorem root71Candidates9_checked : candidates edge 4 373 2161 root71Tail10 winner.bound = root71Candidates9 := by
  unfold candidates
  have row : edge 2161 = (SortedNatTree.contains FastRows.row2161) := by
    change Data.row2161.contains = (SortedNatTree.contains FastRows.row2161)
    exact FastRows.row2161_checked.symm
  rw [row]
  decide
theorem root71Step9 : scan edge (search edge 3) 4 [373] 373 root71Tail9 winner =
    scan edge (search edge 3) 4 [373] 373 root71Tail10 winner := by
  rw [root71Tail9, scan, if_neg (by decide), root71Candidates9_checked]
  change scan edge (search edge 3) 4 [373] 373 root71Tail10
    (if 3 ≤ root71Candidates9.length then search edge 3 [373, 2161] root71Candidates9 2534 winner else winner) = _
  rw [if_neg (by decide)]
def root71Candidates10 : List Nat := [17659]
set_option maxRecDepth 8192 in
theorem root71Candidates10_checked : candidates edge 4 373 2467 root71Tail11 winner.bound = root71Candidates10 := by
  unfold candidates
  have row : edge 2467 = (SortedNatTree.contains FastRows.row2467) := by
    change Data.row2467.contains = (SortedNatTree.contains FastRows.row2467)
    exact FastRows.row2467_checked.symm
  rw [row]
  decide
theorem root71Step10 : scan edge (search edge 3) 4 [373] 373 root71Tail10 winner =
    scan edge (search edge 3) 4 [373] 373 root71Tail11 winner := by
  rw [root71Tail10, scan, if_neg (by decide), root71Candidates10_checked]
  change scan edge (search edge 3) 4 [373] 373 root71Tail11
    (if 3 ≤ root71Candidates10.length then search edge 3 [373, 2467] root71Candidates10 2840 winner else winner) = _
  rw [if_neg (by decide)]
def root71Candidates11 : List Nat := [11971, 20011]
set_option maxRecDepth 8192 in
theorem root71Candidates11_checked : candidates edge 4 373 3967 root71Tail12 winner.bound = root71Candidates11 := by
  unfold candidates
  have row : edge 3967 = (SortedNatTree.contains FastRows.row3967) := by
    change Data.row3967.contains = (SortedNatTree.contains FastRows.row3967)
    exact FastRows.row3967_checked.symm
  rw [row]
  decide
theorem root71Step11 : scan edge (search edge 3) 4 [373] 373 root71Tail11 winner =
    scan edge (search edge 3) 4 [373] 373 root71Tail12 winner := by
  rw [root71Tail11, scan, if_neg (by decide), root71Candidates11_checked]
  change scan edge (search edge 3) 4 [373] 373 root71Tail12
    (if 3 ≤ root71Candidates11.length then search edge 3 [373, 3967] root71Candidates11 4340 winner else winner) = _
  rw [if_neg (by decide)]
def root71Candidates12 : List Nat := [6133, 18289]
set_option maxRecDepth 8192 in
theorem root71Candidates12_checked : candidates edge 4 373 4519 root71Tail13 winner.bound = root71Candidates12 := by
  unfold candidates
  have row : edge 4519 = (SortedNatTree.contains FastRows.row4519) := by
    change Data.row4519.contains = (SortedNatTree.contains FastRows.row4519)
    exact FastRows.row4519_checked.symm
  rw [row]
  decide
theorem root71Step12 : scan edge (search edge 3) 4 [373] 373 root71Tail12 winner =
    scan edge (search edge 3) 4 [373] 373 root71Tail13 winner := by
  rw [root71Tail12, scan, if_neg (by decide), root71Candidates12_checked]
  change scan edge (search edge 3) 4 [373] 373 root71Tail13
    (if 3 ≤ root71Candidates12.length then search edge 3 [373, 4519] root71Candidates12 4892 winner else winner) = _
  rw [if_neg (by decide)]
def root71Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root71Candidates13_checked : candidates edge 4 373 4903 root71Tail14 winner.bound = root71Candidates13 := by
  unfold candidates
  have row : edge 4903 = (SortedNatTree.contains FastRows.row4903) := by
    change Data.row4903.contains = (SortedNatTree.contains FastRows.row4903)
    exact FastRows.row4903_checked.symm
  rw [row]
  decide
theorem root71Step13 : scan edge (search edge 3) 4 [373] 373 root71Tail13 winner =
    scan edge (search edge 3) 4 [373] 373 root71Tail14 winner := by
  rw [root71Tail13, scan, if_neg (by decide), root71Candidates13_checked]
  change scan edge (search edge 3) 4 [373] 373 root71Tail14
    (if 3 ≤ root71Candidates13.length then search edge 3 [373, 4903] root71Candidates13 5276 winner else winner) = _
  rw [if_neg (by decide)]
def root71Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root71Candidates14_checked : candidates edge 4 373 4957 root71Tail15 winner.bound = root71Candidates14 := by
  unfold candidates
  have row : edge 4957 = (SortedNatTree.contains FastRows.row4957) := by
    change Data.row4957.contains = (SortedNatTree.contains FastRows.row4957)
    exact FastRows.row4957_checked.symm
  rw [row]
  decide
theorem root71Step14 : scan edge (search edge 3) 4 [373] 373 root71Tail14 winner =
    scan edge (search edge 3) 4 [373] 373 root71Tail15 winner := by
  rw [root71Tail14, scan, if_neg (by decide), root71Candidates14_checked]
  change scan edge (search edge 3) 4 [373] 373 root71Tail15
    (if 3 ≤ root71Candidates14.length then search edge 3 [373, 4957] root71Candidates14 5330 winner else winner) = _
  rw [if_neg (by decide)]
def root71Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root71Candidates15_checked : candidates edge 4 373 5101 root71Tail16 winner.bound = root71Candidates15 := by
  unfold candidates
  have row : edge 5101 = (SortedNatTree.contains FastRows.row5101) := by
    change Data.row5101.contains = (SortedNatTree.contains FastRows.row5101)
    exact FastRows.row5101_checked.symm
  rw [row]
  decide
theorem root71Step15 : scan edge (search edge 3) 4 [373] 373 root71Tail15 winner =
    scan edge (search edge 3) 4 [373] 373 root71Tail16 winner := by
  rw [root71Tail15, scan, if_neg (by decide), root71Candidates15_checked]
  change scan edge (search edge 3) 4 [373] 373 root71Tail16
    (if 3 ≤ root71Candidates15.length then search edge 3 [373, 5101] root71Candidates15 5474 winner else winner) = _
  rw [if_neg (by decide)]
def root71Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root71Candidates16_checked : candidates edge 4 373 5449 root71Tail17 winner.bound = root71Candidates16 := by
  unfold candidates
  have row : edge 5449 = (SortedNatTree.contains FastRows.row5449) := by
    change Data.row5449.contains = (SortedNatTree.contains FastRows.row5449)
    exact FastRows.row5449_checked.symm
  rw [row]
  decide
theorem root71Step16 : scan edge (search edge 3) 4 [373] 373 root71Tail16 winner =
    scan edge (search edge 3) 4 [373] 373 root71Tail17 winner := by
  rw [root71Tail16, scan, if_neg (by decide), root71Candidates16_checked]
  change scan edge (search edge 3) 4 [373] 373 root71Tail17
    (if 3 ≤ root71Candidates16.length then search edge 3 [373, 5449] root71Candidates16 5822 winner else winner) = _
  rw [if_neg (by decide)]
def root71Candidates17 : List Nat := [11971]
set_option maxRecDepth 8192 in
theorem root71Candidates17_checked : candidates edge 4 373 5881 root71Tail18 winner.bound = root71Candidates17 := by
  unfold candidates
  have row : edge 5881 = (SortedNatTree.contains FastRows.row5881) := by
    change Data.row5881.contains = (SortedNatTree.contains FastRows.row5881)
    exact FastRows.row5881_checked.symm
  rw [row]
  decide
theorem root71Step17 : scan edge (search edge 3) 4 [373] 373 root71Tail17 winner =
    scan edge (search edge 3) 4 [373] 373 root71Tail18 winner := by
  rw [root71Tail17, scan, if_neg (by decide), root71Candidates17_checked]
  change scan edge (search edge 3) 4 [373] 373 root71Tail18
    (if 3 ≤ root71Candidates17.length then search edge 3 [373, 5881] root71Candidates17 6254 winner else winner) = _
  rw [if_neg (by decide)]
def root71Candidates18 : List Nat := [9661, 14071]
set_option maxRecDepth 8192 in
theorem root71Candidates18_checked : candidates edge 4 373 6133 root71Tail19 winner.bound = root71Candidates18 := by
  unfold candidates
  have row : edge 6133 = (SortedNatTree.contains FastRows.row6133) := by
    change Data.row6133.contains = (SortedNatTree.contains FastRows.row6133)
    exact FastRows.row6133_checked.symm
  rw [row]
  decide
theorem root71Step18 : scan edge (search edge 3) 4 [373] 373 root71Tail18 winner =
    scan edge (search edge 3) 4 [373] 373 root71Tail19 winner := by
  rw [root71Tail18, scan, if_neg (by decide), root71Candidates18_checked]
  change scan edge (search edge 3) 4 [373] 373 root71Tail19
    (if 3 ≤ root71Candidates18.length then search edge 3 [373, 6133] root71Candidates18 6506 winner else winner) = _
  rw [if_neg (by decide)]
def root71Candidates19 : List Nat := [6793]
set_option maxRecDepth 8192 in
theorem root71Candidates19_checked : candidates edge 4 373 6163 root71Tail20 winner.bound = root71Candidates19 := by
  unfold candidates
  have row : edge 6163 = (SortedNatTree.contains FastRows.row6163) := by
    change Data.row6163.contains = (SortedNatTree.contains FastRows.row6163)
    exact FastRows.row6163_checked.symm
  rw [row]
  decide
theorem root71Step19 : scan edge (search edge 3) 4 [373] 373 root71Tail19 winner =
    scan edge (search edge 3) 4 [373] 373 root71Tail20 winner := by
  rw [root71Tail19, scan, if_neg (by decide), root71Candidates19_checked]
  change scan edge (search edge 3) 4 [373] 373 root71Tail20
    (if 3 ≤ root71Candidates19.length then search edge 3 [373, 6163] root71Candidates19 6536 winner else winner) = _
  rw [if_neg (by decide)]
theorem root71Scan20 : scan edge (search edge 3) 4 [373] 373 root71Tail20 winner = winner := by rw [root71Tail20, scan, if_pos (by decide)]
theorem root71Scan19 : scan edge (search edge 3) 4 [373] 373 root71Tail19 winner = winner :=
  (root71Step19).trans root71Scan20
theorem root71Scan18 : scan edge (search edge 3) 4 [373] 373 root71Tail18 winner = winner :=
  (root71Step18).trans root71Scan19
theorem root71Scan17 : scan edge (search edge 3) 4 [373] 373 root71Tail17 winner = winner :=
  (root71Step17).trans root71Scan18
theorem root71Scan16 : scan edge (search edge 3) 4 [373] 373 root71Tail16 winner = winner :=
  (root71Step16).trans root71Scan17
theorem root71Scan15 : scan edge (search edge 3) 4 [373] 373 root71Tail15 winner = winner :=
  (root71Step15).trans root71Scan16
theorem root71Scan14 : scan edge (search edge 3) 4 [373] 373 root71Tail14 winner = winner :=
  (root71Step14).trans root71Scan15
theorem root71Scan13 : scan edge (search edge 3) 4 [373] 373 root71Tail13 winner = winner :=
  (root71Step13).trans root71Scan14
theorem root71Scan12 : scan edge (search edge 3) 4 [373] 373 root71Tail12 winner = winner :=
  (root71Step12).trans root71Scan13
theorem root71Scan11 : scan edge (search edge 3) 4 [373] 373 root71Tail11 winner = winner :=
  (root71Step11).trans root71Scan12
theorem root71Scan10 : scan edge (search edge 3) 4 [373] 373 root71Tail10 winner = winner :=
  (root71Step10).trans root71Scan11
theorem root71Scan9 : scan edge (search edge 3) 4 [373] 373 root71Tail9 winner = winner :=
  (root71Step9).trans root71Scan10
theorem root71Scan8 : scan edge (search edge 3) 4 [373] 373 root71Tail8 winner = winner :=
  (root71Step8).trans root71Scan9
theorem root71Scan7 : scan edge (search edge 3) 4 [373] 373 root71Tail7 winner = winner :=
  (root71Step7).trans root71Scan8
theorem root71Scan6 : scan edge (search edge 3) 4 [373] 373 root71Tail6 winner = winner :=
  (root71Step6).trans root71Scan7
theorem root71Scan5 : scan edge (search edge 3) 4 [373] 373 root71Tail5 winner = winner :=
  (root71Step5).trans root71Scan6
theorem root71Scan4 : scan edge (search edge 3) 4 [373] 373 root71Tail4 winner = winner :=
  (root71Step4).trans root71Scan5
theorem root71Scan3 : scan edge (search edge 3) 4 [373] 373 root71Tail3 winner = winner :=
  (root71Step3).trans root71Scan4
theorem root71Scan2 : scan edge (search edge 3) 4 [373] 373 root71Tail2 winner = winner :=
  (root71Step2).trans root71Scan3
theorem root71Scan1 : scan edge (search edge 3) 4 [373] 373 root71Tail1 winner = winner :=
  (root71Step1).trans root71Scan2
theorem root71Scan0 : scan edge (search edge 3) 4 [373] 373 root71Tail0 winner = winner :=
  (root71Step0).trans root71Scan1
set_option maxRecDepth 4096 in
theorem root71Branch : search edge 4 [373] root71Candidates 373 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [373] 373 root71Tail0 winner = winner
  exact root71Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
