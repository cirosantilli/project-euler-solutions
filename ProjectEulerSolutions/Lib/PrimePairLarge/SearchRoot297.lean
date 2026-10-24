import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root297Candidates : List Nat := Data.row1987
set_option maxRecDepth 8192 in
theorem root297Candidates_checked : candidates edge 5 0 1987 tail298 winner.bound = root297Candidates := by
  apply candidates_eq_row edge 5 0 1987 tail298 root297Candidates winner.bound tail298_ordered
  · decide
  · decide
  · funext q
    rfl
def root297Tail57 : List Nat := []
def root297Tail56 : List Nat := 23773 :: root297Tail57
def root297Tail55 : List Nat := 23719 :: root297Tail56
def root297Tail54 : List Nat := 23017 :: root297Tail55
def root297Tail53 : List Nat := 21673 :: root297Tail54
def root297Tail52 : List Nat := 19891 :: root297Tail53
def root297Tail51 : List Nat := 19447 :: root297Tail52
def root297Tail50 : List Nat := 19429 :: root297Tail51
def root297Tail49 : List Nat := 18973 :: root297Tail50
def root297Tail48 : List Nat := 17977 :: root297Tail49
def root297Tail47 : List Nat := 17683 :: root297Tail48
def root297Tail46 : List Nat := 17293 :: root297Tail47
def root297Tail45 : List Nat := 16831 :: root297Tail46
def root297Tail44 : List Nat := 16747 :: root297Tail45
def root297Tail43 : List Nat := 16741 :: root297Tail44
def root297Tail42 : List Nat := 15817 :: root297Tail43
def root297Tail41 : List Nat := 14143 :: root297Tail42
def root297Tail40 : List Nat := 13093 :: root297Tail41
def root297Tail39 : List Nat := 12601 :: root297Tail40
def root297Tail38 : List Nat := 12373 :: root297Tail39
def root297Tail37 : List Nat := 11353 :: root297Tail38
def root297Tail36 : List Nat := 11287 :: root297Tail37
def root297Tail35 : List Nat := 11131 :: root297Tail36
def root297Tail34 : List Nat := 10903 :: root297Tail35
def root297Tail33 : List Nat := 10627 :: root297Tail34
def root297Tail32 : List Nat := 10513 :: root297Tail33
def root297Tail31 : List Nat := 9439 :: root297Tail32
def root297Tail30 : List Nat := 9421 :: root297Tail31
def root297Tail29 : List Nat := 8893 :: root297Tail30
def root297Tail28 : List Nat := 8821 :: root297Tail29
def root297Tail27 : List Nat := 8539 :: root297Tail28
def root297Tail26 : List Nat := 8263 :: root297Tail27
def root297Tail25 : List Nat := 8191 :: root297Tail26
def root297Tail24 : List Nat := 7549 :: root297Tail25
def root297Tail23 : List Nat := 7237 :: root297Tail24
def root297Tail22 : List Nat := 7027 :: root297Tail23
def root297Tail21 : List Nat := 6481 :: root297Tail22
def root297Tail20 : List Nat := 6133 :: root297Tail21
def root297Tail19 : List Nat := 6079 :: root297Tail20
def root297Tail18 : List Nat := 5659 :: root297Tail19
def root297Tail17 : List Nat := 5563 :: root297Tail18
def root297Tail16 : List Nat := 5521 :: root297Tail17
def root297Tail15 : List Nat := 4801 :: root297Tail16
def root297Tail14 : List Nat := 4639 :: root297Tail15
def root297Tail13 : List Nat := 4423 :: root297Tail14
def root297Tail12 : List Nat := 3877 :: root297Tail13
def root297Tail11 : List Nat := 3709 :: root297Tail12
def root297Tail10 : List Nat := 3517 :: root297Tail11
def root297Tail9 : List Nat := 3499 :: root297Tail10
def root297Tail8 : List Nat := 3361 :: root297Tail9
def root297Tail7 : List Nat := 3331 :: root297Tail8
def root297Tail6 : List Nat := 3079 :: root297Tail7
def root297Tail5 : List Nat := 2719 :: root297Tail6
def root297Tail4 : List Nat := 2707 :: root297Tail5
def root297Tail3 : List Nat := 2551 :: root297Tail4
def root297Tail2 : List Nat := 2467 :: root297Tail3
def root297Tail1 : List Nat := 2383 :: root297Tail2
def root297Tail0 : List Nat := 2131 :: root297Tail1
def root297Candidates0 : List Nat := [5563, 7027, 12373]
set_option maxRecDepth 8192 in
theorem root297Candidates0_checked : candidates edge 4 1987 2131 root297Tail1 winner.bound = root297Candidates0 := by
  unfold candidates
  have row : edge 2131 = (SortedNatTree.contains FastRows.row2131) := by
    change Data.row2131.contains = (SortedNatTree.contains FastRows.row2131)
    exact FastRows.row2131_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root297Child0 : search edge 3 [1987, 2131] root297Candidates0 4118 winner = winner := by decide
theorem root297Step0 : scan edge (search edge 3) 4 [1987] 1987 root297Tail0 winner =
    scan edge (search edge 3) 4 [1987] 1987 root297Tail1 winner := by
  rw [root297Tail0, scan, if_neg (by decide), root297Candidates0_checked]
  change scan edge (search edge 3) 4 [1987] 1987 root297Tail1
    (if 3 ≤ root297Candidates0.length then search edge 3 [1987, 2131] root297Candidates0 4118 winner else winner) = _
  rw [if_pos (by decide), root297Child0]
def root297Candidates1 : List Nat := [4639, 8263]
set_option maxRecDepth 8192 in
theorem root297Candidates1_checked : candidates edge 4 1987 2383 root297Tail2 winner.bound = root297Candidates1 := by
  unfold candidates
  have row : edge 2383 = (SortedNatTree.contains FastRows.row2383) := by
    change Data.row2383.contains = (SortedNatTree.contains FastRows.row2383)
    exact FastRows.row2383_checked.symm
  rw [row]
  decide
theorem root297Step1 : scan edge (search edge 3) 4 [1987] 1987 root297Tail1 winner =
    scan edge (search edge 3) 4 [1987] 1987 root297Tail2 winner := by
  rw [root297Tail1, scan, if_neg (by decide), root297Candidates1_checked]
  change scan edge (search edge 3) 4 [1987] 1987 root297Tail2
    (if 3 ≤ root297Candidates1.length then search edge 3 [1987, 2383] root297Candidates1 4370 winner else winner) = _
  rw [if_neg (by decide)]
def root297Candidates2 : List Nat := [2551, 16741]
set_option maxRecDepth 8192 in
theorem root297Candidates2_checked : candidates edge 4 1987 2467 root297Tail3 winner.bound = root297Candidates2 := by
  unfold candidates
  have row : edge 2467 = (SortedNatTree.contains FastRows.row2467) := by
    change Data.row2467.contains = (SortedNatTree.contains FastRows.row2467)
    exact FastRows.row2467_checked.symm
  rw [row]
  decide
theorem root297Step2 : scan edge (search edge 3) 4 [1987] 1987 root297Tail2 winner =
    scan edge (search edge 3) 4 [1987] 1987 root297Tail3 winner := by
  rw [root297Tail2, scan, if_neg (by decide), root297Candidates2_checked]
  change scan edge (search edge 3) 4 [1987] 1987 root297Tail3
    (if 3 ≤ root297Candidates2.length then search edge 3 [1987, 2467] root297Candidates2 4454 winner else winner) = _
  rw [if_neg (by decide)]
def root297Candidates3 : List Nat := [3331]
set_option maxRecDepth 8192 in
theorem root297Candidates3_checked : candidates edge 4 1987 2551 root297Tail4 winner.bound = root297Candidates3 := by
  unfold candidates
  have row : edge 2551 = (SortedNatTree.contains FastRows.row2551) := by
    change Data.row2551.contains = (SortedNatTree.contains FastRows.row2551)
    exact FastRows.row2551_checked.symm
  rw [row]
  decide
theorem root297Step3 : scan edge (search edge 3) 4 [1987] 1987 root297Tail3 winner =
    scan edge (search edge 3) 4 [1987] 1987 root297Tail4 winner := by
  rw [root297Tail3, scan, if_neg (by decide), root297Candidates3_checked]
  change scan edge (search edge 3) 4 [1987] 1987 root297Tail4
    (if 3 ≤ root297Candidates3.length then search edge 3 [1987, 2551] root297Candidates3 4538 winner else winner) = _
  rw [if_neg (by decide)]
def root297Candidates4 : List Nat := [3517, 6079]
set_option maxRecDepth 8192 in
theorem root297Candidates4_checked : candidates edge 4 1987 2707 root297Tail5 winner.bound = root297Candidates4 := by
  unfold candidates
  have row : edge 2707 = (SortedNatTree.contains FastRows.row2707) := by
    change Data.row2707.contains = (SortedNatTree.contains FastRows.row2707)
    exact FastRows.row2707_checked.symm
  rw [row]
  decide
theorem root297Step4 : scan edge (search edge 3) 4 [1987] 1987 root297Tail4 winner =
    scan edge (search edge 3) 4 [1987] 1987 root297Tail5 winner := by
  rw [root297Tail4, scan, if_neg (by decide), root297Candidates4_checked]
  change scan edge (search edge 3) 4 [1987] 1987 root297Tail5
    (if 3 ≤ root297Candidates4.length then search edge 3 [1987, 2707] root297Candidates4 4694 winner else winner) = _
  rw [if_neg (by decide)]
def root297Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root297Candidates5_checked : candidates edge 4 1987 2719 root297Tail6 winner.bound = root297Candidates5 := by
  unfold candidates
  have row : edge 2719 = (SortedNatTree.contains FastRows.row2719) := by
    change Data.row2719.contains = (SortedNatTree.contains FastRows.row2719)
    exact FastRows.row2719_checked.symm
  rw [row]
  decide
theorem root297Step5 : scan edge (search edge 3) 4 [1987] 1987 root297Tail5 winner =
    scan edge (search edge 3) 4 [1987] 1987 root297Tail6 winner := by
  rw [root297Tail5, scan, if_neg (by decide), root297Candidates5_checked]
  change scan edge (search edge 3) 4 [1987] 1987 root297Tail6
    (if 3 ≤ root297Candidates5.length then search edge 3 [1987, 2719] root297Candidates5 4706 winner else winner) = _
  rw [if_neg (by decide)]
def root297Candidates6 : List Nat := [3709, 3877, 8821]
set_option maxRecDepth 8192 in
theorem root297Candidates6_checked : candidates edge 4 1987 3079 root297Tail7 winner.bound = root297Candidates6 := by
  unfold candidates
  have row : edge 3079 = (SortedNatTree.contains FastRows.row3079) := by
    change Data.row3079.contains = (SortedNatTree.contains FastRows.row3079)
    exact FastRows.row3079_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root297Child6 : search edge 3 [1987, 3079] root297Candidates6 5066 winner = winner := by decide
theorem root297Step6 : scan edge (search edge 3) 4 [1987] 1987 root297Tail6 winner =
    scan edge (search edge 3) 4 [1987] 1987 root297Tail7 winner := by
  rw [root297Tail6, scan, if_neg (by decide), root297Candidates6_checked]
  change scan edge (search edge 3) 4 [1987] 1987 root297Tail7
    (if 3 ≤ root297Candidates6.length then search edge 3 [1987, 3079] root297Candidates6 5066 winner else winner) = _
  rw [if_pos (by decide), root297Child6]
def root297Candidates7 : List Nat := [9421, 16747]
set_option maxRecDepth 8192 in
theorem root297Candidates7_checked : candidates edge 4 1987 3331 root297Tail8 winner.bound = root297Candidates7 := by
  unfold candidates
  have row : edge 3331 = (SortedNatTree.contains FastRows.row3331) := by
    change Data.row3331.contains = (SortedNatTree.contains FastRows.row3331)
    exact FastRows.row3331_checked.symm
  rw [row]
  decide
theorem root297Step7 : scan edge (search edge 3) 4 [1987] 1987 root297Tail7 winner =
    scan edge (search edge 3) 4 [1987] 1987 root297Tail8 winner := by
  rw [root297Tail7, scan, if_neg (by decide), root297Candidates7_checked]
  change scan edge (search edge 3) 4 [1987] 1987 root297Tail8
    (if 3 ≤ root297Candidates7.length then search edge 3 [1987, 3331] root297Candidates7 5318 winner else winner) = _
  rw [if_neg (by decide)]
def root297Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root297Candidates8_checked : candidates edge 4 1987 3361 root297Tail9 winner.bound = root297Candidates8 := by
  unfold candidates
  have row : edge 3361 = (SortedNatTree.contains FastRows.row3361) := by
    change Data.row3361.contains = (SortedNatTree.contains FastRows.row3361)
    exact FastRows.row3361_checked.symm
  rw [row]
  decide
theorem root297Step8 : scan edge (search edge 3) 4 [1987] 1987 root297Tail8 winner =
    scan edge (search edge 3) 4 [1987] 1987 root297Tail9 winner := by
  rw [root297Tail8, scan, if_neg (by decide), root297Candidates8_checked]
  change scan edge (search edge 3) 4 [1987] 1987 root297Tail9
    (if 3 ≤ root297Candidates8.length then search edge 3 [1987, 3361] root297Candidates8 5348 winner else winner) = _
  rw [if_neg (by decide)]
def root297Candidates9 : List Nat := [8191, 12373, 17977]
set_option maxRecDepth 8192 in
theorem root297Candidates9_checked : candidates edge 4 1987 3499 root297Tail10 winner.bound = root297Candidates9 := by
  unfold candidates
  have row : edge 3499 = (SortedNatTree.contains FastRows.row3499) := by
    change Data.row3499.contains = (SortedNatTree.contains FastRows.row3499)
    exact FastRows.row3499_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root297Child9 : search edge 3 [1987, 3499] root297Candidates9 5486 winner = winner := by decide
theorem root297Step9 : scan edge (search edge 3) 4 [1987] 1987 root297Tail9 winner =
    scan edge (search edge 3) 4 [1987] 1987 root297Tail10 winner := by
  rw [root297Tail9, scan, if_neg (by decide), root297Candidates9_checked]
  change scan edge (search edge 3) 4 [1987] 1987 root297Tail10
    (if 3 ≤ root297Candidates9.length then search edge 3 [1987, 3499] root297Candidates9 5486 winner else winner) = _
  rw [if_pos (by decide), root297Child9]
def root297Candidates10 : List Nat := [5659, 8263, 15817]
set_option maxRecDepth 8192 in
theorem root297Candidates10_checked : candidates edge 4 1987 3517 root297Tail11 winner.bound = root297Candidates10 := by
  unfold candidates
  have row : edge 3517 = (SortedNatTree.contains FastRows.row3517) := by
    change Data.row3517.contains = (SortedNatTree.contains FastRows.row3517)
    exact FastRows.row3517_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root297Child10 : search edge 3 [1987, 3517] root297Candidates10 5504 winner = winner := by decide
theorem root297Step10 : scan edge (search edge 3) 4 [1987] 1987 root297Tail10 winner =
    scan edge (search edge 3) 4 [1987] 1987 root297Tail11 winner := by
  rw [root297Tail10, scan, if_neg (by decide), root297Candidates10_checked]
  change scan edge (search edge 3) 4 [1987] 1987 root297Tail11
    (if 3 ≤ root297Candidates10.length then search edge 3 [1987, 3517] root297Candidates10 5504 winner else winner) = _
  rw [if_pos (by decide), root297Child10]
def root297Candidates11 : List Nat := [12601, 15817]
set_option maxRecDepth 8192 in
theorem root297Candidates11_checked : candidates edge 4 1987 3709 root297Tail12 winner.bound = root297Candidates11 := by
  unfold candidates
  have row : edge 3709 = (SortedNatTree.contains FastRows.row3709) := by
    change Data.row3709.contains = (SortedNatTree.contains FastRows.row3709)
    exact FastRows.row3709_checked.symm
  rw [row]
  decide
theorem root297Step11 : scan edge (search edge 3) 4 [1987] 1987 root297Tail11 winner =
    scan edge (search edge 3) 4 [1987] 1987 root297Tail12 winner := by
  rw [root297Tail11, scan, if_neg (by decide), root297Candidates11_checked]
  change scan edge (search edge 3) 4 [1987] 1987 root297Tail12
    (if 3 ≤ root297Candidates11.length then search edge 3 [1987, 3709] root297Candidates11 5696 winner else winner) = _
  rw [if_neg (by decide)]
def root297Candidates12 : List Nat := [8263]
set_option maxRecDepth 8192 in
theorem root297Candidates12_checked : candidates edge 4 1987 3877 root297Tail13 winner.bound = root297Candidates12 := by
  unfold candidates
  have row : edge 3877 = (SortedNatTree.contains FastRows.row3877) := by
    change Data.row3877.contains = (SortedNatTree.contains FastRows.row3877)
    exact FastRows.row3877_checked.symm
  rw [row]
  decide
theorem root297Step12 : scan edge (search edge 3) 4 [1987] 1987 root297Tail12 winner =
    scan edge (search edge 3) 4 [1987] 1987 root297Tail13 winner := by
  rw [root297Tail12, scan, if_neg (by decide), root297Candidates12_checked]
  change scan edge (search edge 3) 4 [1987] 1987 root297Tail13
    (if 3 ≤ root297Candidates12.length then search edge 3 [1987, 3877] root297Candidates12 5864 winner else winner) = _
  rw [if_neg (by decide)]
def root297Candidates13 : List Nat := [6133, 9421]
set_option maxRecDepth 8192 in
theorem root297Candidates13_checked : candidates edge 4 1987 4423 root297Tail14 winner.bound = root297Candidates13 := by
  unfold candidates
  have row : edge 4423 = (SortedNatTree.contains FastRows.row4423) := by
    change Data.row4423.contains = (SortedNatTree.contains FastRows.row4423)
    exact FastRows.row4423_checked.symm
  rw [row]
  decide
theorem root297Step13 : scan edge (search edge 3) 4 [1987] 1987 root297Tail13 winner =
    scan edge (search edge 3) 4 [1987] 1987 root297Tail14 winner := by
  rw [root297Tail13, scan, if_neg (by decide), root297Candidates13_checked]
  change scan edge (search edge 3) 4 [1987] 1987 root297Tail14
    (if 3 ≤ root297Candidates13.length then search edge 3 [1987, 4423] root297Candidates13 6410 winner else winner) = _
  rw [if_neg (by decide)]
def root297Candidates14 : List Nat := [7237]
set_option maxRecDepth 8192 in
theorem root297Candidates14_checked : candidates edge 4 1987 4639 root297Tail15 winner.bound = root297Candidates14 := by
  unfold candidates
  have row : edge 4639 = (SortedNatTree.contains FastRows.row4639) := by
    change Data.row4639.contains = (SortedNatTree.contains FastRows.row4639)
    exact FastRows.row4639_checked.symm
  rw [row]
  decide
theorem root297Step14 : scan edge (search edge 3) 4 [1987] 1987 root297Tail14 winner =
    scan edge (search edge 3) 4 [1987] 1987 root297Tail15 winner := by
  rw [root297Tail14, scan, if_neg (by decide), root297Candidates14_checked]
  change scan edge (search edge 3) 4 [1987] 1987 root297Tail15
    (if 3 ≤ root297Candidates14.length then search edge 3 [1987, 4639] root297Candidates14 6626 winner else winner) = _
  rw [if_neg (by decide)]
def root297Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root297Candidates15_checked : candidates edge 4 1987 4801 root297Tail16 winner.bound = root297Candidates15 := by
  unfold candidates
  have row : edge 4801 = (SortedNatTree.contains FastRows.row4801) := by
    change Data.row4801.contains = (SortedNatTree.contains FastRows.row4801)
    exact FastRows.row4801_checked.symm
  rw [row]
  decide
theorem root297Step15 : scan edge (search edge 3) 4 [1987] 1987 root297Tail15 winner =
    scan edge (search edge 3) 4 [1987] 1987 root297Tail16 winner := by
  rw [root297Tail15, scan, if_neg (by decide), root297Candidates15_checked]
  change scan edge (search edge 3) 4 [1987] 1987 root297Tail16
    (if 3 ≤ root297Candidates15.length then search edge 3 [1987, 4801] root297Candidates15 6788 winner else winner) = _
  rw [if_neg (by decide)]
def root297Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root297Candidates16_checked : candidates edge 4 1987 5521 root297Tail17 winner.bound = root297Candidates16 := by
  unfold candidates
  have row : edge 5521 = (SortedNatTree.contains FastRows.row5521) := by
    change Data.row5521.contains = (SortedNatTree.contains FastRows.row5521)
    exact FastRows.row5521_checked.symm
  rw [row]
  decide
theorem root297Step16 : scan edge (search edge 3) 4 [1987] 1987 root297Tail16 winner =
    scan edge (search edge 3) 4 [1987] 1987 root297Tail17 winner := by
  rw [root297Tail16, scan, if_neg (by decide), root297Candidates16_checked]
  change scan edge (search edge 3) 4 [1987] 1987 root297Tail17
    (if 3 ≤ root297Candidates16.length then search edge 3 [1987, 5521] root297Candidates16 7508 winner else winner) = _
  rw [if_neg (by decide)]
def root297Candidates17 : List Nat := [6079]
set_option maxRecDepth 8192 in
theorem root297Candidates17_checked : candidates edge 4 1987 5563 root297Tail18 winner.bound = root297Candidates17 := by
  unfold candidates
  have row : edge 5563 = (SortedNatTree.contains FastRows.row5563) := by
    change Data.row5563.contains = (SortedNatTree.contains FastRows.row5563)
    exact FastRows.row5563_checked.symm
  rw [row]
  decide
theorem root297Step17 : scan edge (search edge 3) 4 [1987] 1987 root297Tail17 winner =
    scan edge (search edge 3) 4 [1987] 1987 root297Tail18 winner := by
  rw [root297Tail17, scan, if_neg (by decide), root297Candidates17_checked]
  change scan edge (search edge 3) 4 [1987] 1987 root297Tail18
    (if 3 ≤ root297Candidates17.length then search edge 3 [1987, 5563] root297Candidates17 7550 winner else winner) = _
  rw [if_neg (by decide)]
def root297Candidates18 : List Nat := [13093]
set_option maxRecDepth 8192 in
theorem root297Candidates18_checked : candidates edge 4 1987 5659 root297Tail19 winner.bound = root297Candidates18 := by
  unfold candidates
  have row : edge 5659 = (SortedNatTree.contains FastRows.row5659) := by
    change Data.row5659.contains = (SortedNatTree.contains FastRows.row5659)
    exact FastRows.row5659_checked.symm
  rw [row]
  decide
theorem root297Step18 : scan edge (search edge 3) 4 [1987] 1987 root297Tail18 winner =
    scan edge (search edge 3) 4 [1987] 1987 root297Tail19 winner := by
  rw [root297Tail18, scan, if_neg (by decide), root297Candidates18_checked]
  change scan edge (search edge 3) 4 [1987] 1987 root297Tail19
    (if 3 ≤ root297Candidates18.length then search edge 3 [1987, 5659] root297Candidates18 7646 winner else winner) = _
  rw [if_neg (by decide)]
theorem root297Scan19 : scan edge (search edge 3) 4 [1987] 1987 root297Tail19 winner = winner := by rw [root297Tail19, scan, if_pos (by decide)]
theorem root297Scan18 : scan edge (search edge 3) 4 [1987] 1987 root297Tail18 winner = winner :=
  (root297Step18).trans root297Scan19
theorem root297Scan17 : scan edge (search edge 3) 4 [1987] 1987 root297Tail17 winner = winner :=
  (root297Step17).trans root297Scan18
theorem root297Scan16 : scan edge (search edge 3) 4 [1987] 1987 root297Tail16 winner = winner :=
  (root297Step16).trans root297Scan17
theorem root297Scan15 : scan edge (search edge 3) 4 [1987] 1987 root297Tail15 winner = winner :=
  (root297Step15).trans root297Scan16
theorem root297Scan14 : scan edge (search edge 3) 4 [1987] 1987 root297Tail14 winner = winner :=
  (root297Step14).trans root297Scan15
theorem root297Scan13 : scan edge (search edge 3) 4 [1987] 1987 root297Tail13 winner = winner :=
  (root297Step13).trans root297Scan14
theorem root297Scan12 : scan edge (search edge 3) 4 [1987] 1987 root297Tail12 winner = winner :=
  (root297Step12).trans root297Scan13
theorem root297Scan11 : scan edge (search edge 3) 4 [1987] 1987 root297Tail11 winner = winner :=
  (root297Step11).trans root297Scan12
theorem root297Scan10 : scan edge (search edge 3) 4 [1987] 1987 root297Tail10 winner = winner :=
  (root297Step10).trans root297Scan11
theorem root297Scan9 : scan edge (search edge 3) 4 [1987] 1987 root297Tail9 winner = winner :=
  (root297Step9).trans root297Scan10
theorem root297Scan8 : scan edge (search edge 3) 4 [1987] 1987 root297Tail8 winner = winner :=
  (root297Step8).trans root297Scan9
theorem root297Scan7 : scan edge (search edge 3) 4 [1987] 1987 root297Tail7 winner = winner :=
  (root297Step7).trans root297Scan8
theorem root297Scan6 : scan edge (search edge 3) 4 [1987] 1987 root297Tail6 winner = winner :=
  (root297Step6).trans root297Scan7
theorem root297Scan5 : scan edge (search edge 3) 4 [1987] 1987 root297Tail5 winner = winner :=
  (root297Step5).trans root297Scan6
theorem root297Scan4 : scan edge (search edge 3) 4 [1987] 1987 root297Tail4 winner = winner :=
  (root297Step4).trans root297Scan5
theorem root297Scan3 : scan edge (search edge 3) 4 [1987] 1987 root297Tail3 winner = winner :=
  (root297Step3).trans root297Scan4
theorem root297Scan2 : scan edge (search edge 3) 4 [1987] 1987 root297Tail2 winner = winner :=
  (root297Step2).trans root297Scan3
theorem root297Scan1 : scan edge (search edge 3) 4 [1987] 1987 root297Tail1 winner = winner :=
  (root297Step1).trans root297Scan2
theorem root297Scan0 : scan edge (search edge 3) 4 [1987] 1987 root297Tail0 winner = winner :=
  (root297Step0).trans root297Scan1
set_option maxRecDepth 4096 in
theorem root297Branch : search edge 4 [1987] root297Candidates 1987 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1987] 1987 root297Tail0 winner = winner
  exact root297Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
