import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root43Candidates : List Nat := Data.row199
set_option maxRecDepth 8192 in
theorem root43Candidates_checked : candidates edge 5 0 199 tail44 winner.bound = root43Candidates := by
  apply candidates_eq_row edge 5 0 199 tail44 root43Candidates winner.bound tail44_ordered
  · decide
  · decide
  · funext q
    rfl
def root43Tail61 : List Nat := []
def root43Tail60 : List Nat := 24337 :: root43Tail61
def root43Tail59 : List Nat := 23977 :: root43Tail60
def root43Tail58 : List Nat := 23041 :: root43Tail59
def root43Tail57 : List Nat := 22369 :: root43Tail58
def root43Tail56 : List Nat := 22171 :: root43Tail57
def root43Tail55 : List Nat := 21991 :: root43Tail56
def root43Tail54 : List Nat := 21757 :: root43Tail55
def root43Tail53 : List Nat := 21529 :: root43Tail54
def root43Tail52 : List Nat := 20287 :: root43Tail53
def root43Tail51 : List Nat := 19531 :: root43Tail52
def root43Tail50 : List Nat := 18541 :: root43Tail51
def root43Tail49 : List Nat := 18181 :: root43Tail50
def root43Tail48 : List Nat := 18013 :: root43Tail49
def root43Tail47 : List Nat := 17851 :: root43Tail48
def root43Tail46 : List Nat := 17683 :: root43Tail47
def root43Tail45 : List Nat := 17383 :: root43Tail46
def root43Tail44 : List Nat := 17341 :: root43Tail45
def root43Tail43 : List Nat := 16993 :: root43Tail44
def root43Tail42 : List Nat := 16573 :: root43Tail43
def root43Tail41 : List Nat := 15271 :: root43Tail42
def root43Tail40 : List Nat := 14737 :: root43Tail41
def root43Tail39 : List Nat := 14449 :: root43Tail40
def root43Tail38 : List Nat := 14149 :: root43Tail39
def root43Tail37 : List Nat := 13183 :: root43Tail38
def root43Tail36 : List Nat := 13099 :: root43Tail37
def root43Tail35 : List Nat := 12517 :: root43Tail36
def root43Tail34 : List Nat := 12157 :: root43Tail35
def root43Tail33 : List Nat := 11971 :: root43Tail34
def root43Tail32 : List Nat := 10903 :: root43Tail33
def root43Tail31 : List Nat := 10141 :: root43Tail32
def root43Tail30 : List Nat := 10039 :: root43Tail31
def root43Tail29 : List Nat := 9721 :: root43Tail30
def root43Tail28 : List Nat := 9511 :: root43Tail29
def root43Tail27 : List Nat := 9343 :: root43Tail28
def root43Tail26 : List Nat := 8443 :: root43Tail27
def root43Tail25 : List Nat := 6949 :: root43Tail26
def root43Tail24 : List Nat := 6301 :: root43Tail25
def root43Tail23 : List Nat := 5107 :: root43Tail24
def root43Tail22 : List Nat := 5023 :: root43Tail23
def root43Tail21 : List Nat := 5011 :: root43Tail22
def root43Tail20 : List Nat := 4327 :: root43Tail21
def root43Tail19 : List Nat := 3529 :: root43Tail20
def root43Tail18 : List Nat := 3259 :: root43Tail19
def root43Tail17 : List Nat := 3109 :: root43Tail18
def root43Tail16 : List Nat := 3067 :: root43Tail17
def root43Tail15 : List Nat := 2953 :: root43Tail16
def root43Tail14 : List Nat := 2917 :: root43Tail15
def root43Tail13 : List Nat := 2437 :: root43Tail14
def root43Tail12 : List Nat := 2251 :: root43Tail13
def root43Tail11 : List Nat := 1933 :: root43Tail12
def root43Tail10 : List Nat := 1609 :: root43Tail11
def root43Tail9 : List Nat := 1489 :: root43Tail10
def root43Tail8 : List Nat := 1153 :: root43Tail9
def root43Tail7 : List Nat := 877 :: root43Tail8
def root43Tail6 : List Nat := 811 :: root43Tail7
def root43Tail5 : List Nat := 751 :: root43Tail6
def root43Tail4 : List Nat := 739 :: root43Tail5
def root43Tail3 : List Nat := 673 :: root43Tail4
def root43Tail2 : List Nat := 379 :: root43Tail3
def root43Tail1 : List Nat := 373 :: root43Tail2
def root43Tail0 : List Nat := 211 :: root43Tail1
def root43Candidates0 : List Nat := [373]
set_option maxRecDepth 8192 in
theorem root43Candidates0_checked : candidates edge 4 199 211 root43Tail1 winner.bound = root43Candidates0 := by
  unfold candidates
  have row : edge 211 = (SortedNatTree.contains FastRows.row211) := by
    change Data.row211.contains = (SortedNatTree.contains FastRows.row211)
    exact FastRows.row211_checked.symm
  rw [row]
  decide
theorem root43Step0 : scan edge (search edge 3) 4 [199] 199 root43Tail0 winner =
    scan edge (search edge 3) 4 [199] 199 root43Tail1 winner := by
  rw [root43Tail0, scan, if_neg (by decide), root43Candidates0_checked]
  change scan edge (search edge 3) 4 [199] 199 root43Tail1
    (if 3 ≤ root43Candidates0.length then search edge 3 [199, 211] root43Candidates0 410 winner else winner) = _
  rw [if_neg (by decide)]
def root43Candidates1 : List Nat := [11971]
set_option maxRecDepth 8192 in
theorem root43Candidates1_checked : candidates edge 4 199 373 root43Tail2 winner.bound = root43Candidates1 := by
  unfold candidates
  have row : edge 373 = (SortedNatTree.contains FastRows.row373) := by
    change Data.row373.contains = (SortedNatTree.contains FastRows.row373)
    exact FastRows.row373_checked.symm
  rw [row]
  decide
theorem root43Step1 : scan edge (search edge 3) 4 [199] 199 root43Tail1 winner =
    scan edge (search edge 3) 4 [199] 199 root43Tail2 winner := by
  rw [root43Tail1, scan, if_neg (by decide), root43Candidates1_checked]
  change scan edge (search edge 3) 4 [199] 199 root43Tail2
    (if 3 ≤ root43Candidates1.length then search edge 3 [199, 373] root43Candidates1 572 winner else winner) = _
  rw [if_neg (by decide)]
def root43Candidates2 : List Nat := [751, 811, 877, 3259]
set_option maxRecDepth 8192 in
theorem root43Candidates2_checked : candidates edge 4 199 379 root43Tail3 winner.bound = root43Candidates2 := by
  unfold candidates
  have row : edge 379 = (SortedNatTree.contains FastRows.row379) := by
    change Data.row379.contains = (SortedNatTree.contains FastRows.row379)
    exact FastRows.row379_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root43Child2 : search edge 3 [199, 379] root43Candidates2 578 winner = winner := by decide
theorem root43Step2 : scan edge (search edge 3) 4 [199] 199 root43Tail2 winner =
    scan edge (search edge 3) 4 [199] 199 root43Tail3 winner := by
  rw [root43Tail2, scan, if_neg (by decide), root43Candidates2_checked]
  change scan edge (search edge 3) 4 [199] 199 root43Tail3
    (if 3 ≤ root43Candidates2.length then search edge 3 [199, 379] root43Candidates2 578 winner else winner) = _
  rw [if_pos (by decide), root43Child2]
def root43Candidates3 : List Nat := [1489]
set_option maxRecDepth 8192 in
theorem root43Candidates3_checked : candidates edge 4 199 673 root43Tail4 winner.bound = root43Candidates3 := by
  unfold candidates
  have row : edge 673 = (SortedNatTree.contains FastRows.row673) := by
    change Data.row673.contains = (SortedNatTree.contains FastRows.row673)
    exact FastRows.row673_checked.symm
  rw [row]
  decide
theorem root43Step3 : scan edge (search edge 3) 4 [199] 199 root43Tail3 winner =
    scan edge (search edge 3) 4 [199] 199 root43Tail4 winner := by
  rw [root43Tail3, scan, if_neg (by decide), root43Candidates3_checked]
  change scan edge (search edge 3) 4 [199] 199 root43Tail4
    (if 3 ≤ root43Candidates3.length then search edge 3 [199, 673] root43Candidates3 872 winner else winner) = _
  rw [if_neg (by decide)]
def root43Candidates4 : List Nat := [751, 14149, 18181]
set_option maxRecDepth 8192 in
theorem root43Candidates4_checked : candidates edge 4 199 739 root43Tail5 winner.bound = root43Candidates4 := by
  unfold candidates
  have row : edge 739 = (SortedNatTree.contains FastRows.row739) := by
    change Data.row739.contains = (SortedNatTree.contains FastRows.row739)
    exact FastRows.row739_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root43Child4 : search edge 3 [199, 739] root43Candidates4 938 winner = winner := by decide
theorem root43Step4 : scan edge (search edge 3) 4 [199] 199 root43Tail4 winner =
    scan edge (search edge 3) 4 [199] 199 root43Tail5 winner := by
  rw [root43Tail4, scan, if_neg (by decide), root43Candidates4_checked]
  change scan edge (search edge 3) 4 [199] 199 root43Tail5
    (if 3 ≤ root43Candidates4.length then search edge 3 [199, 739] root43Candidates4 938 winner else winner) = _
  rw [if_pos (by decide), root43Child4]
def root43Candidates5 : List Nat := [2437, 10141]
set_option maxRecDepth 8192 in
theorem root43Candidates5_checked : candidates edge 4 199 751 root43Tail6 winner.bound = root43Candidates5 := by
  unfold candidates
  have row : edge 751 = (SortedNatTree.contains FastRows.row751) := by
    change Data.row751.contains = (SortedNatTree.contains FastRows.row751)
    exact FastRows.row751_checked.symm
  rw [row]
  decide
theorem root43Step5 : scan edge (search edge 3) 4 [199] 199 root43Tail5 winner =
    scan edge (search edge 3) 4 [199] 199 root43Tail6 winner := by
  rw [root43Tail5, scan, if_neg (by decide), root43Candidates5_checked]
  change scan edge (search edge 3) 4 [199] 199 root43Tail6
    (if 3 ≤ root43Candidates5.length then search edge 3 [199, 751] root43Candidates5 950 winner else winner) = _
  rw [if_neg (by decide)]
def root43Candidates6 : List Nat := [3109, 15271, 17383]
set_option maxRecDepth 8192 in
theorem root43Candidates6_checked : candidates edge 4 199 811 root43Tail7 winner.bound = root43Candidates6 := by
  unfold candidates
  have row : edge 811 = (SortedNatTree.contains FastRows.row811) := by
    change Data.row811.contains = (SortedNatTree.contains FastRows.row811)
    exact FastRows.row811_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root43Child6 : search edge 3 [199, 811] root43Candidates6 1010 winner = winner := by decide
theorem root43Step6 : scan edge (search edge 3) 4 [199] 199 root43Tail6 winner =
    scan edge (search edge 3) 4 [199] 199 root43Tail7 winner := by
  rw [root43Tail6, scan, if_neg (by decide), root43Candidates6_checked]
  change scan edge (search edge 3) 4 [199] 199 root43Tail7
    (if 3 ≤ root43Candidates6.length then search edge 3 [199, 811] root43Candidates6 1010 winner else winner) = _
  rw [if_pos (by decide), root43Child6]
def root43Candidates7 : List Nat := [2251, 2917, 17683]
set_option maxRecDepth 8192 in
theorem root43Candidates7_checked : candidates edge 4 199 877 root43Tail8 winner.bound = root43Candidates7 := by
  unfold candidates
  have row : edge 877 = (SortedNatTree.contains FastRows.row877) := by
    change Data.row877.contains = (SortedNatTree.contains FastRows.row877)
    exact FastRows.row877_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root43Child7 : search edge 3 [199, 877] root43Candidates7 1076 winner = winner := by decide
theorem root43Step7 : scan edge (search edge 3) 4 [199] 199 root43Tail7 winner =
    scan edge (search edge 3) 4 [199] 199 root43Tail8 winner := by
  rw [root43Tail7, scan, if_neg (by decide), root43Candidates7_checked]
  change scan edge (search edge 3) 4 [199] 199 root43Tail8
    (if 3 ≤ root43Candidates7.length then search edge 3 [199, 877] root43Candidates7 1076 winner else winner) = _
  rw [if_pos (by decide), root43Child7]
def root43Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root43Candidates8_checked : candidates edge 4 199 1153 root43Tail9 winner.bound = root43Candidates8 := by
  unfold candidates
  have row : edge 1153 = (SortedNatTree.contains FastRows.row1153) := by
    change Data.row1153.contains = (SortedNatTree.contains FastRows.row1153)
    exact FastRows.row1153_checked.symm
  rw [row]
  decide
theorem root43Step8 : scan edge (search edge 3) 4 [199] 199 root43Tail8 winner =
    scan edge (search edge 3) 4 [199] 199 root43Tail9 winner := by
  rw [root43Tail8, scan, if_neg (by decide), root43Candidates8_checked]
  change scan edge (search edge 3) 4 [199] 199 root43Tail9
    (if 3 ≤ root43Candidates8.length then search edge 3 [199, 1153] root43Candidates8 1352 winner else winner) = _
  rw [if_neg (by decide)]
def root43Candidates9 : List Nat := [14149]
set_option maxRecDepth 8192 in
theorem root43Candidates9_checked : candidates edge 4 199 1489 root43Tail10 winner.bound = root43Candidates9 := by
  unfold candidates
  have row : edge 1489 = (SortedNatTree.contains FastRows.row1489) := by
    change Data.row1489.contains = (SortedNatTree.contains FastRows.row1489)
    exact FastRows.row1489_checked.symm
  rw [row]
  decide
theorem root43Step9 : scan edge (search edge 3) 4 [199] 199 root43Tail9 winner =
    scan edge (search edge 3) 4 [199] 199 root43Tail10 winner := by
  rw [root43Tail9, scan, if_neg (by decide), root43Candidates9_checked]
  change scan edge (search edge 3) 4 [199] 199 root43Tail10
    (if 3 ≤ root43Candidates9.length then search edge 3 [199, 1489] root43Candidates9 1688 winner else winner) = _
  rw [if_neg (by decide)]
def root43Candidates10 : List Nat := [5011, 23041]
set_option maxRecDepth 8192 in
theorem root43Candidates10_checked : candidates edge 4 199 1609 root43Tail11 winner.bound = root43Candidates10 := by
  unfold candidates
  have row : edge 1609 = (SortedNatTree.contains FastRows.row1609) := by
    change Data.row1609.contains = (SortedNatTree.contains FastRows.row1609)
    exact FastRows.row1609_checked.symm
  rw [row]
  decide
theorem root43Step10 : scan edge (search edge 3) 4 [199] 199 root43Tail10 winner =
    scan edge (search edge 3) 4 [199] 199 root43Tail11 winner := by
  rw [root43Tail10, scan, if_neg (by decide), root43Candidates10_checked]
  change scan edge (search edge 3) 4 [199] 199 root43Tail11
    (if 3 ≤ root43Candidates10.length then search edge 3 [199, 1609] root43Candidates10 1808 winner else winner) = _
  rw [if_neg (by decide)]
def root43Candidates11 : List Nat := [5011, 5023]
set_option maxRecDepth 8192 in
theorem root43Candidates11_checked : candidates edge 4 199 1933 root43Tail12 winner.bound = root43Candidates11 := by
  unfold candidates
  have row : edge 1933 = (SortedNatTree.contains FastRows.row1933) := by
    change Data.row1933.contains = (SortedNatTree.contains FastRows.row1933)
    exact FastRows.row1933_checked.symm
  rw [row]
  decide
theorem root43Step11 : scan edge (search edge 3) 4 [199] 199 root43Tail11 winner =
    scan edge (search edge 3) 4 [199] 199 root43Tail12 winner := by
  rw [root43Tail11, scan, if_neg (by decide), root43Candidates11_checked]
  change scan edge (search edge 3) 4 [199] 199 root43Tail12
    (if 3 ≤ root43Candidates11.length then search edge 3 [199, 1933] root43Candidates11 2132 winner else winner) = _
  rw [if_neg (by decide)]
def root43Candidates12 : List Nat := [12157, 15271]
set_option maxRecDepth 8192 in
theorem root43Candidates12_checked : candidates edge 4 199 2251 root43Tail13 winner.bound = root43Candidates12 := by
  unfold candidates
  have row : edge 2251 = (SortedNatTree.contains FastRows.row2251) := by
    change Data.row2251.contains = (SortedNatTree.contains FastRows.row2251)
    exact FastRows.row2251_checked.symm
  rw [row]
  decide
theorem root43Step12 : scan edge (search edge 3) 4 [199] 199 root43Tail12 winner =
    scan edge (search edge 3) 4 [199] 199 root43Tail13 winner := by
  rw [root43Tail12, scan, if_neg (by decide), root43Candidates12_checked]
  change scan edge (search edge 3) 4 [199] 199 root43Tail13
    (if 3 ≤ root43Candidates12.length then search edge 3 [199, 2251] root43Candidates12 2450 winner else winner) = _
  rw [if_neg (by decide)]
def root43Candidates13 : List Nat := [6301]
set_option maxRecDepth 8192 in
theorem root43Candidates13_checked : candidates edge 4 199 2437 root43Tail14 winner.bound = root43Candidates13 := by
  unfold candidates
  have row : edge 2437 = (SortedNatTree.contains FastRows.row2437) := by
    change Data.row2437.contains = (SortedNatTree.contains FastRows.row2437)
    exact FastRows.row2437_checked.symm
  rw [row]
  decide
theorem root43Step13 : scan edge (search edge 3) 4 [199] 199 root43Tail13 winner =
    scan edge (search edge 3) 4 [199] 199 root43Tail14 winner := by
  rw [root43Tail13, scan, if_neg (by decide), root43Candidates13_checked]
  change scan edge (search edge 3) 4 [199] 199 root43Tail14
    (if 3 ≤ root43Candidates13.length then search edge 3 [199, 2437] root43Candidates13 2636 winner else winner) = _
  rw [if_neg (by decide)]
def root43Candidates14 : List Nat := [5011, 9721]
set_option maxRecDepth 8192 in
theorem root43Candidates14_checked : candidates edge 4 199 2917 root43Tail15 winner.bound = root43Candidates14 := by
  unfold candidates
  have row : edge 2917 = (SortedNatTree.contains FastRows.row2917) := by
    change Data.row2917.contains = (SortedNatTree.contains FastRows.row2917)
    exact FastRows.row2917_checked.symm
  rw [row]
  decide
theorem root43Step14 : scan edge (search edge 3) 4 [199] 199 root43Tail14 winner =
    scan edge (search edge 3) 4 [199] 199 root43Tail15 winner := by
  rw [root43Tail14, scan, if_neg (by decide), root43Candidates14_checked]
  change scan edge (search edge 3) 4 [199] 199 root43Tail15
    (if 3 ≤ root43Candidates14.length then search edge 3 [199, 2917] root43Candidates14 3116 winner else winner) = _
  rw [if_neg (by decide)]
def root43Candidates15 : List Nat := [9343, 12517]
set_option maxRecDepth 8192 in
theorem root43Candidates15_checked : candidates edge 4 199 2953 root43Tail16 winner.bound = root43Candidates15 := by
  unfold candidates
  have row : edge 2953 = (SortedNatTree.contains FastRows.row2953) := by
    change Data.row2953.contains = (SortedNatTree.contains FastRows.row2953)
    exact FastRows.row2953_checked.symm
  rw [row]
  decide
theorem root43Step15 : scan edge (search edge 3) 4 [199] 199 root43Tail15 winner =
    scan edge (search edge 3) 4 [199] 199 root43Tail16 winner := by
  rw [root43Tail15, scan, if_neg (by decide), root43Candidates15_checked]
  change scan edge (search edge 3) 4 [199] 199 root43Tail16
    (if 3 ≤ root43Candidates15.length then search edge 3 [199, 2953] root43Candidates15 3152 winner else winner) = _
  rw [if_neg (by decide)]
def root43Candidates16 : List Nat := [17683, 21757]
set_option maxRecDepth 8192 in
theorem root43Candidates16_checked : candidates edge 4 199 3067 root43Tail17 winner.bound = root43Candidates16 := by
  unfold candidates
  have row : edge 3067 = (SortedNatTree.contains FastRows.row3067) := by
    change Data.row3067.contains = (SortedNatTree.contains FastRows.row3067)
    exact FastRows.row3067_checked.symm
  rw [row]
  decide
theorem root43Step16 : scan edge (search edge 3) 4 [199] 199 root43Tail16 winner =
    scan edge (search edge 3) 4 [199] 199 root43Tail17 winner := by
  rw [root43Tail16, scan, if_neg (by decide), root43Candidates16_checked]
  change scan edge (search edge 3) 4 [199] 199 root43Tail17
    (if 3 ≤ root43Candidates16.length then search edge 3 [199, 3067] root43Candidates16 3266 winner else winner) = _
  rw [if_neg (by decide)]
def root43Candidates17 : List Nat := [10903, 15271, 16993]
set_option maxRecDepth 8192 in
theorem root43Candidates17_checked : candidates edge 4 199 3109 root43Tail18 winner.bound = root43Candidates17 := by
  unfold candidates
  have row : edge 3109 = (SortedNatTree.contains FastRows.row3109) := by
    change Data.row3109.contains = (SortedNatTree.contains FastRows.row3109)
    exact FastRows.row3109_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root43Child17 : search edge 3 [199, 3109] root43Candidates17 3308 winner = winner := by decide
theorem root43Step17 : scan edge (search edge 3) 4 [199] 199 root43Tail17 winner =
    scan edge (search edge 3) 4 [199] 199 root43Tail18 winner := by
  rw [root43Tail17, scan, if_neg (by decide), root43Candidates17_checked]
  change scan edge (search edge 3) 4 [199] 199 root43Tail18
    (if 3 ≤ root43Candidates17.length then search edge 3 [199, 3109] root43Candidates17 3308 winner else winner) = _
  rw [if_pos (by decide), root43Child17]
def root43Candidates18 : List Nat := [4327, 5107, 18013]
set_option maxRecDepth 8192 in
theorem root43Candidates18_checked : candidates edge 4 199 3259 root43Tail19 winner.bound = root43Candidates18 := by
  unfold candidates
  have row : edge 3259 = (SortedNatTree.contains FastRows.row3259) := by
    change Data.row3259.contains = (SortedNatTree.contains FastRows.row3259)
    exact FastRows.row3259_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root43Child18 : search edge 3 [199, 3259] root43Candidates18 3458 winner = winner := by decide
theorem root43Step18 : scan edge (search edge 3) 4 [199] 199 root43Tail18 winner =
    scan edge (search edge 3) 4 [199] 199 root43Tail19 winner := by
  rw [root43Tail18, scan, if_neg (by decide), root43Candidates18_checked]
  change scan edge (search edge 3) 4 [199] 199 root43Tail19
    (if 3 ≤ root43Candidates18.length then search edge 3 [199, 3259] root43Candidates18 3458 winner else winner) = _
  rw [if_pos (by decide), root43Child18]
def root43Candidates19 : List Nat := [5011]
set_option maxRecDepth 8192 in
theorem root43Candidates19_checked : candidates edge 4 199 3529 root43Tail20 winner.bound = root43Candidates19 := by
  unfold candidates
  have row : edge 3529 = (SortedNatTree.contains FastRows.row3529) := by
    change Data.row3529.contains = (SortedNatTree.contains FastRows.row3529)
    exact FastRows.row3529_checked.symm
  rw [row]
  decide
theorem root43Step19 : scan edge (search edge 3) 4 [199] 199 root43Tail19 winner =
    scan edge (search edge 3) 4 [199] 199 root43Tail20 winner := by
  rw [root43Tail19, scan, if_neg (by decide), root43Candidates19_checked]
  change scan edge (search edge 3) 4 [199] 199 root43Tail20
    (if 3 ≤ root43Candidates19.length then search edge 3 [199, 3529] root43Candidates19 3728 winner else winner) = _
  rw [if_neg (by decide)]
def root43Candidates20 : List Nat := [5107]
set_option maxRecDepth 8192 in
theorem root43Candidates20_checked : candidates edge 4 199 4327 root43Tail21 winner.bound = root43Candidates20 := by
  unfold candidates
  have row : edge 4327 = (SortedNatTree.contains FastRows.row4327) := by
    change Data.row4327.contains = (SortedNatTree.contains FastRows.row4327)
    exact FastRows.row4327_checked.symm
  rw [row]
  decide
theorem root43Step20 : scan edge (search edge 3) 4 [199] 199 root43Tail20 winner =
    scan edge (search edge 3) 4 [199] 199 root43Tail21 winner := by
  rw [root43Tail20, scan, if_neg (by decide), root43Candidates20_checked]
  change scan edge (search edge 3) 4 [199] 199 root43Tail21
    (if 3 ≤ root43Candidates20.length then search edge 3 [199, 4327] root43Candidates20 4526 winner else winner) = _
  rw [if_neg (by decide)]
def root43Candidates21 : List Nat := [6949]
set_option maxRecDepth 8192 in
theorem root43Candidates21_checked : candidates edge 4 199 5011 root43Tail22 winner.bound = root43Candidates21 := by
  unfold candidates
  have row : edge 5011 = (SortedNatTree.contains FastRows.row5011) := by
    change Data.row5011.contains = (SortedNatTree.contains FastRows.row5011)
    exact FastRows.row5011_checked.symm
  rw [row]
  decide
theorem root43Step21 : scan edge (search edge 3) 4 [199] 199 root43Tail21 winner =
    scan edge (search edge 3) 4 [199] 199 root43Tail22 winner := by
  rw [root43Tail21, scan, if_neg (by decide), root43Candidates21_checked]
  change scan edge (search edge 3) 4 [199] 199 root43Tail22
    (if 3 ≤ root43Candidates21.length then search edge 3 [199, 5011] root43Candidates21 5210 winner else winner) = _
  rw [if_neg (by decide)]
def root43Candidates22 : List Nat := [10903, 16993]
set_option maxRecDepth 8192 in
theorem root43Candidates22_checked : candidates edge 4 199 5023 root43Tail23 winner.bound = root43Candidates22 := by
  unfold candidates
  have row : edge 5023 = (SortedNatTree.contains FastRows.row5023) := by
    change Data.row5023.contains = (SortedNatTree.contains FastRows.row5023)
    exact FastRows.row5023_checked.symm
  rw [row]
  decide
theorem root43Step22 : scan edge (search edge 3) 4 [199] 199 root43Tail22 winner =
    scan edge (search edge 3) 4 [199] 199 root43Tail23 winner := by
  rw [root43Tail22, scan, if_neg (by decide), root43Candidates22_checked]
  change scan edge (search edge 3) 4 [199] 199 root43Tail23
    (if 3 ≤ root43Candidates22.length then search edge 3 [199, 5023] root43Candidates22 5222 winner else winner) = _
  rw [if_neg (by decide)]
def root43Candidates23 : List Nat := [17341]
set_option maxRecDepth 8192 in
theorem root43Candidates23_checked : candidates edge 4 199 5107 root43Tail24 winner.bound = root43Candidates23 := by
  unfold candidates
  have row : edge 5107 = (SortedNatTree.contains FastRows.row5107) := by
    change Data.row5107.contains = (SortedNatTree.contains FastRows.row5107)
    exact FastRows.row5107_checked.symm
  rw [row]
  decide
theorem root43Step23 : scan edge (search edge 3) 4 [199] 199 root43Tail23 winner =
    scan edge (search edge 3) 4 [199] 199 root43Tail24 winner := by
  rw [root43Tail23, scan, if_neg (by decide), root43Candidates23_checked]
  change scan edge (search edge 3) 4 [199] 199 root43Tail24
    (if 3 ≤ root43Candidates23.length then search edge 3 [199, 5107] root43Candidates23 5306 winner else winner) = _
  rw [if_neg (by decide)]
def root43Candidates24 : List Nat := []
set_option maxRecDepth 8192 in
theorem root43Candidates24_checked : candidates edge 4 199 6301 root43Tail25 winner.bound = root43Candidates24 := by
  unfold candidates
  have row : edge 6301 = (SortedNatTree.contains FastRows.row6301) := by
    change Data.row6301.contains = (SortedNatTree.contains FastRows.row6301)
    exact FastRows.row6301_checked.symm
  rw [row]
  decide
theorem root43Step24 : scan edge (search edge 3) 4 [199] 199 root43Tail24 winner =
    scan edge (search edge 3) 4 [199] 199 root43Tail25 winner := by
  rw [root43Tail24, scan, if_neg (by decide), root43Candidates24_checked]
  change scan edge (search edge 3) 4 [199] 199 root43Tail25
    (if 3 ≤ root43Candidates24.length then search edge 3 [199, 6301] root43Candidates24 6500 winner else winner) = _
  rw [if_neg (by decide)]
theorem root43Scan25 : scan edge (search edge 3) 4 [199] 199 root43Tail25 winner = winner := by rw [root43Tail25, scan, if_pos (by decide)]
theorem root43Scan24 : scan edge (search edge 3) 4 [199] 199 root43Tail24 winner = winner :=
  (root43Step24).trans root43Scan25
theorem root43Scan23 : scan edge (search edge 3) 4 [199] 199 root43Tail23 winner = winner :=
  (root43Step23).trans root43Scan24
theorem root43Scan22 : scan edge (search edge 3) 4 [199] 199 root43Tail22 winner = winner :=
  (root43Step22).trans root43Scan23
theorem root43Scan21 : scan edge (search edge 3) 4 [199] 199 root43Tail21 winner = winner :=
  (root43Step21).trans root43Scan22
theorem root43Scan20 : scan edge (search edge 3) 4 [199] 199 root43Tail20 winner = winner :=
  (root43Step20).trans root43Scan21
theorem root43Scan19 : scan edge (search edge 3) 4 [199] 199 root43Tail19 winner = winner :=
  (root43Step19).trans root43Scan20
theorem root43Scan18 : scan edge (search edge 3) 4 [199] 199 root43Tail18 winner = winner :=
  (root43Step18).trans root43Scan19
theorem root43Scan17 : scan edge (search edge 3) 4 [199] 199 root43Tail17 winner = winner :=
  (root43Step17).trans root43Scan18
theorem root43Scan16 : scan edge (search edge 3) 4 [199] 199 root43Tail16 winner = winner :=
  (root43Step16).trans root43Scan17
theorem root43Scan15 : scan edge (search edge 3) 4 [199] 199 root43Tail15 winner = winner :=
  (root43Step15).trans root43Scan16
theorem root43Scan14 : scan edge (search edge 3) 4 [199] 199 root43Tail14 winner = winner :=
  (root43Step14).trans root43Scan15
theorem root43Scan13 : scan edge (search edge 3) 4 [199] 199 root43Tail13 winner = winner :=
  (root43Step13).trans root43Scan14
theorem root43Scan12 : scan edge (search edge 3) 4 [199] 199 root43Tail12 winner = winner :=
  (root43Step12).trans root43Scan13
theorem root43Scan11 : scan edge (search edge 3) 4 [199] 199 root43Tail11 winner = winner :=
  (root43Step11).trans root43Scan12
theorem root43Scan10 : scan edge (search edge 3) 4 [199] 199 root43Tail10 winner = winner :=
  (root43Step10).trans root43Scan11
theorem root43Scan9 : scan edge (search edge 3) 4 [199] 199 root43Tail9 winner = winner :=
  (root43Step9).trans root43Scan10
theorem root43Scan8 : scan edge (search edge 3) 4 [199] 199 root43Tail8 winner = winner :=
  (root43Step8).trans root43Scan9
theorem root43Scan7 : scan edge (search edge 3) 4 [199] 199 root43Tail7 winner = winner :=
  (root43Step7).trans root43Scan8
theorem root43Scan6 : scan edge (search edge 3) 4 [199] 199 root43Tail6 winner = winner :=
  (root43Step6).trans root43Scan7
theorem root43Scan5 : scan edge (search edge 3) 4 [199] 199 root43Tail5 winner = winner :=
  (root43Step5).trans root43Scan6
theorem root43Scan4 : scan edge (search edge 3) 4 [199] 199 root43Tail4 winner = winner :=
  (root43Step4).trans root43Scan5
theorem root43Scan3 : scan edge (search edge 3) 4 [199] 199 root43Tail3 winner = winner :=
  (root43Step3).trans root43Scan4
theorem root43Scan2 : scan edge (search edge 3) 4 [199] 199 root43Tail2 winner = winner :=
  (root43Step2).trans root43Scan3
theorem root43Scan1 : scan edge (search edge 3) 4 [199] 199 root43Tail1 winner = winner :=
  (root43Step1).trans root43Scan2
theorem root43Scan0 : scan edge (search edge 3) 4 [199] 199 root43Tail0 winner = winner :=
  (root43Step0).trans root43Scan1
set_option maxRecDepth 4096 in
theorem root43Branch : search edge 4 [199] root43Candidates 199 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [199] 199 root43Tail0 winner = winner
  exact root43Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
