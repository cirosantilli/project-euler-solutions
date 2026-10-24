import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root118Candidates : List Nat := Data.row661
set_option maxRecDepth 8192 in
theorem root118Candidates_checked : candidates edge 5 0 661 tail119 winner.bound = root118Candidates := by
  apply candidates_eq_row edge 5 0 661 tail119 root118Candidates winner.bound tail119_ordered
  · decide
  · decide
  · funext q
    rfl
def root118Tail71 : List Nat := []
def root118Tail70 : List Nat := 25183 :: root118Tail71
def root118Tail69 : List Nat := 25117 :: root118Tail70
def root118Tail68 : List Nat := 24697 :: root118Tail69
def root118Tail67 : List Nat := 24229 :: root118Tail68
def root118Tail66 : List Nat := 24049 :: root118Tail67
def root118Tail65 : List Nat := 23917 :: root118Tail66
def root118Tail64 : List Nat := 23599 :: root118Tail65
def root118Tail63 : List Nat := 23209 :: root118Tail64
def root118Tail62 : List Nat := 23017 :: root118Tail63
def root118Tail61 : List Nat := 22369 :: root118Tail62
def root118Tail60 : List Nat := 22153 :: root118Tail61
def root118Tail59 : List Nat := 21841 :: root118Tail60
def root118Tail58 : List Nat := 21739 :: root118Tail59
def root118Tail57 : List Nat := 20749 :: root118Tail58
def root118Tail56 : List Nat := 20641 :: root118Tail57
def root118Tail55 : List Nat := 20563 :: root118Tail56
def root118Tail54 : List Nat := 20011 :: root118Tail55
def root118Tail53 : List Nat := 19891 :: root118Tail54
def root118Tail52 : List Nat := 19819 :: root118Tail53
def root118Tail51 : List Nat := 19441 :: root118Tail52
def root118Tail50 : List Nat := 18379 :: root118Tail51
def root118Tail49 : List Nat := 17239 :: root118Tail50
def root118Tail48 : List Nat := 16987 :: root118Tail49
def root118Tail47 : List Nat := 16381 :: root118Tail48
def root118Tail46 : List Nat := 14827 :: root118Tail47
def root118Tail45 : List Nat := 14821 :: root118Tail46
def root118Tail44 : List Nat := 14767 :: root118Tail45
def root118Tail43 : List Nat := 14533 :: root118Tail44
def root118Tail42 : List Nat := 14221 :: root118Tail43
def root118Tail41 : List Nat := 13963 :: root118Tail42
def root118Tail40 : List Nat := 13759 :: root118Tail41
def root118Tail39 : List Nat := 13147 :: root118Tail40
def root118Tail38 : List Nat := 11719 :: root118Tail39
def root118Tail37 : List Nat := 11587 :: root118Tail38
def root118Tail36 : List Nat := 11491 :: root118Tail37
def root118Tail35 : List Nat := 11239 :: root118Tail36
def root118Tail34 : List Nat := 11197 :: root118Tail35
def root118Tail33 : List Nat := 11173 :: root118Tail34
def root118Tail32 : List Nat := 11083 :: root118Tail33
def root118Tail31 : List Nat := 10243 :: root118Tail32
def root118Tail30 : List Nat := 9871 :: root118Tail31
def root118Tail29 : List Nat := 9643 :: root118Tail30
def root118Tail28 : List Nat := 9343 :: root118Tail29
def root118Tail27 : List Nat := 9157 :: root118Tail28
def root118Tail26 : List Nat := 8419 :: root118Tail27
def root118Tail25 : List Nat := 7873 :: root118Tail26
def root118Tail24 : List Nat := 7159 :: root118Tail25
def root118Tail23 : List Nat := 6871 :: root118Tail24
def root118Tail22 : List Nat := 6781 :: root118Tail23
def root118Tail21 : List Nat := 6529 :: root118Tail22
def root118Tail20 : List Nat := 6151 :: root118Tail21
def root118Tail19 : List Nat := 5641 :: root118Tail20
def root118Tail18 : List Nat := 5503 :: root118Tail19
def root118Tail17 : List Nat := 4759 :: root118Tail18
def root118Tail16 : List Nat := 3931 :: root118Tail17
def root118Tail15 : List Nat := 3823 :: root118Tail16
def root118Tail14 : List Nat := 3361 :: root118Tail15
def root118Tail13 : List Nat := 3331 :: root118Tail14
def root118Tail12 : List Nat := 2953 :: root118Tail13
def root118Tail11 : List Nat := 2857 :: root118Tail12
def root118Tail10 : List Nat := 2791 :: root118Tail11
def root118Tail9 : List Nat := 2539 :: root118Tail10
def root118Tail8 : List Nat := 1993 :: root118Tail9
def root118Tail7 : List Nat := 1933 :: root118Tail8
def root118Tail6 : List Nat := 1777 :: root118Tail7
def root118Tail5 : List Nat := 1489 :: root118Tail6
def root118Tail4 : List Nat := 1447 :: root118Tail5
def root118Tail3 : List Nat := 1237 :: root118Tail4
def root118Tail2 : List Nat := 883 :: root118Tail3
def root118Tail1 : List Nat := 877 :: root118Tail2
def root118Tail0 : List Nat := 769 :: root118Tail1
def root118Candidates0 : List Nat := [5503, 9157, 10243, 11173, 14221]
set_option maxRecDepth 8192 in
theorem root118Candidates0_checked : candidates edge 4 661 769 root118Tail1 winner.bound = root118Candidates0 := by
  unfold candidates
  have row : edge 769 = (SortedNatTree.contains FastRows.row769) := by
    change Data.row769.contains = (SortedNatTree.contains FastRows.row769)
    exact FastRows.row769_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root118Child0 : search edge 3 [661, 769] root118Candidates0 1430 winner = winner := by decide
theorem root118Step0 : scan edge (search edge 3) 4 [661] 661 root118Tail0 winner =
    scan edge (search edge 3) 4 [661] 661 root118Tail1 winner := by
  rw [root118Tail0, scan, if_neg (by decide), root118Candidates0_checked]
  change scan edge (search edge 3) 4 [661] 661 root118Tail1
    (if 3 ≤ root118Candidates0.length then search edge 3 [661, 769] root118Candidates0 1430 winner else winner) = _
  rw [if_pos (by decide), root118Child0]
def root118Candidates1 : List Nat := [883, 3823, 6529]
set_option maxRecDepth 8192 in
theorem root118Candidates1_checked : candidates edge 4 661 877 root118Tail2 winner.bound = root118Candidates1 := by
  unfold candidates
  have row : edge 877 = (SortedNatTree.contains FastRows.row877) := by
    change Data.row877.contains = (SortedNatTree.contains FastRows.row877)
    exact FastRows.row877_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root118Child1 : search edge 3 [661, 877] root118Candidates1 1538 winner = winner := by decide
theorem root118Step1 : scan edge (search edge 3) 4 [661] 661 root118Tail1 winner =
    scan edge (search edge 3) 4 [661] 661 root118Tail2 winner := by
  rw [root118Tail1, scan, if_neg (by decide), root118Candidates1_checked]
  change scan edge (search edge 3) 4 [661] 661 root118Tail2
    (if 3 ≤ root118Candidates1.length then search edge 3 [661, 877] root118Candidates1 1538 winner else winner) = _
  rw [if_pos (by decide), root118Child1]
def root118Candidates2 : List Nat := [6151]
set_option maxRecDepth 8192 in
theorem root118Candidates2_checked : candidates edge 4 661 883 root118Tail3 winner.bound = root118Candidates2 := by
  unfold candidates
  have row : edge 883 = (SortedNatTree.contains FastRows.row883) := by
    change Data.row883.contains = (SortedNatTree.contains FastRows.row883)
    exact FastRows.row883_checked.symm
  rw [row]
  decide
theorem root118Step2 : scan edge (search edge 3) 4 [661] 661 root118Tail2 winner =
    scan edge (search edge 3) 4 [661] 661 root118Tail3 winner := by
  rw [root118Tail2, scan, if_neg (by decide), root118Candidates2_checked]
  change scan edge (search edge 3) 4 [661] 661 root118Tail3
    (if 3 ≤ root118Candidates2.length then search edge 3 [661, 883] root118Candidates2 1544 winner else winner) = _
  rw [if_neg (by decide)]
def root118Candidates3 : List Nat := [9343, 14821, 19819, 20011]
set_option maxRecDepth 8192 in
theorem root118Candidates3_checked : candidates edge 4 661 1237 root118Tail4 winner.bound = root118Candidates3 := by
  unfold candidates
  have row : edge 1237 = (SortedNatTree.contains FastRows.row1237) := by
    change Data.row1237.contains = (SortedNatTree.contains FastRows.row1237)
    exact FastRows.row1237_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root118Child3 : search edge 3 [661, 1237] root118Candidates3 1898 winner = winner := by decide
theorem root118Step3 : scan edge (search edge 3) 4 [661] 661 root118Tail3 winner =
    scan edge (search edge 3) 4 [661] 661 root118Tail4 winner := by
  rw [root118Tail3, scan, if_neg (by decide), root118Candidates3_checked]
  change scan edge (search edge 3) 4 [661] 661 root118Tail4
    (if 3 ≤ root118Candidates3.length then search edge 3 [661, 1237] root118Candidates3 1898 winner else winner) = _
  rw [if_pos (by decide), root118Child3]
def root118Candidates4 : List Nat := [1993, 5641, 13147]
set_option maxRecDepth 8192 in
theorem root118Candidates4_checked : candidates edge 4 661 1447 root118Tail5 winner.bound = root118Candidates4 := by
  unfold candidates
  have row : edge 1447 = (SortedNatTree.contains FastRows.row1447) := by
    change Data.row1447.contains = (SortedNatTree.contains FastRows.row1447)
    exact FastRows.row1447_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root118Child4 : search edge 3 [661, 1447] root118Candidates4 2108 winner = winner := by decide
theorem root118Step4 : scan edge (search edge 3) 4 [661] 661 root118Tail4 winner =
    scan edge (search edge 3) 4 [661] 661 root118Tail5 winner := by
  rw [root118Tail4, scan, if_neg (by decide), root118Candidates4_checked]
  change scan edge (search edge 3) 4 [661] 661 root118Tail5
    (if 3 ≤ root118Candidates4.length then search edge 3 [661, 1447] root118Candidates4 2108 winner else winner) = _
  rw [if_pos (by decide), root118Child4]
def root118Candidates5 : List Nat := [20749]
set_option maxRecDepth 8192 in
theorem root118Candidates5_checked : candidates edge 4 661 1489 root118Tail6 winner.bound = root118Candidates5 := by
  unfold candidates
  have row : edge 1489 = (SortedNatTree.contains FastRows.row1489) := by
    change Data.row1489.contains = (SortedNatTree.contains FastRows.row1489)
    exact FastRows.row1489_checked.symm
  rw [row]
  decide
theorem root118Step5 : scan edge (search edge 3) 4 [661] 661 root118Tail5 winner =
    scan edge (search edge 3) 4 [661] 661 root118Tail6 winner := by
  rw [root118Tail5, scan, if_neg (by decide), root118Candidates5_checked]
  change scan edge (search edge 3) 4 [661] 661 root118Tail6
    (if 3 ≤ root118Candidates5.length then search edge 3 [661, 1489] root118Candidates5 2150 winner else winner) = _
  rw [if_neg (by decide)]
def root118Candidates6 : List Nat := [2539, 6151, 19891, 23209]
set_option maxRecDepth 8192 in
theorem root118Candidates6_checked : candidates edge 4 661 1777 root118Tail7 winner.bound = root118Candidates6 := by
  unfold candidates
  have row : edge 1777 = (SortedNatTree.contains FastRows.row1777) := by
    change Data.row1777.contains = (SortedNatTree.contains FastRows.row1777)
    exact FastRows.row1777_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root118Child6 : search edge 3 [661, 1777] root118Candidates6 2438 winner = winner := by decide
theorem root118Step6 : scan edge (search edge 3) 4 [661] 661 root118Tail6 winner =
    scan edge (search edge 3) 4 [661] 661 root118Tail7 winner := by
  rw [root118Tail6, scan, if_neg (by decide), root118Candidates6_checked]
  change scan edge (search edge 3) 4 [661] 661 root118Tail7
    (if 3 ≤ root118Candidates6.length then search edge 3 [661, 1777] root118Candidates6 2438 winner else winner) = _
  rw [if_pos (by decide), root118Child6]
def root118Candidates7 : List Nat := [14533]
set_option maxRecDepth 8192 in
theorem root118Candidates7_checked : candidates edge 4 661 1933 root118Tail8 winner.bound = root118Candidates7 := by
  unfold candidates
  have row : edge 1933 = (SortedNatTree.contains FastRows.row1933) := by
    change Data.row1933.contains = (SortedNatTree.contains FastRows.row1933)
    exact FastRows.row1933_checked.symm
  rw [row]
  decide
theorem root118Step7 : scan edge (search edge 3) 4 [661] 661 root118Tail7 winner =
    scan edge (search edge 3) 4 [661] 661 root118Tail8 winner := by
  rw [root118Tail7, scan, if_neg (by decide), root118Candidates7_checked]
  change scan edge (search edge 3) 4 [661] 661 root118Tail8
    (if 3 ≤ root118Candidates7.length then search edge 3 [661, 1933] root118Candidates7 2594 winner else winner) = _
  rw [if_neg (by decide)]
def root118Candidates8 : List Nat := [6151, 6781]
set_option maxRecDepth 8192 in
theorem root118Candidates8_checked : candidates edge 4 661 1993 root118Tail9 winner.bound = root118Candidates8 := by
  unfold candidates
  have row : edge 1993 = (SortedNatTree.contains FastRows.row1993) := by
    change Data.row1993.contains = (SortedNatTree.contains FastRows.row1993)
    exact FastRows.row1993_checked.symm
  rw [row]
  decide
theorem root118Step8 : scan edge (search edge 3) 4 [661] 661 root118Tail8 winner =
    scan edge (search edge 3) 4 [661] 661 root118Tail9 winner := by
  rw [root118Tail8, scan, if_neg (by decide), root118Candidates8_checked]
  change scan edge (search edge 3) 4 [661] 661 root118Tail9
    (if 3 ≤ root118Candidates8.length then search edge 3 [661, 1993] root118Candidates8 2654 winner else winner) = _
  rw [if_neg (by decide)]
def root118Candidates9 : List Nat := [7159]
set_option maxRecDepth 8192 in
theorem root118Candidates9_checked : candidates edge 4 661 2539 root118Tail10 winner.bound = root118Candidates9 := by
  unfold candidates
  have row : edge 2539 = (SortedNatTree.contains FastRows.row2539) := by
    change Data.row2539.contains = (SortedNatTree.contains FastRows.row2539)
    exact FastRows.row2539_checked.symm
  rw [row]
  decide
theorem root118Step9 : scan edge (search edge 3) 4 [661] 661 root118Tail9 winner =
    scan edge (search edge 3) 4 [661] 661 root118Tail10 winner := by
  rw [root118Tail9, scan, if_neg (by decide), root118Candidates9_checked]
  change scan edge (search edge 3) 4 [661] 661 root118Tail10
    (if 3 ≤ root118Candidates9.length then search edge 3 [661, 2539] root118Candidates9 3200 winner else winner) = _
  rw [if_neg (by decide)]
def root118Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root118Candidates10_checked : candidates edge 4 661 2791 root118Tail11 winner.bound = root118Candidates10 := by
  unfold candidates
  have row : edge 2791 = (SortedNatTree.contains FastRows.row2791) := by
    change Data.row2791.contains = (SortedNatTree.contains FastRows.row2791)
    exact FastRows.row2791_checked.symm
  rw [row]
  decide
theorem root118Step10 : scan edge (search edge 3) 4 [661] 661 root118Tail10 winner =
    scan edge (search edge 3) 4 [661] 661 root118Tail11 winner := by
  rw [root118Tail10, scan, if_neg (by decide), root118Candidates10_checked]
  change scan edge (search edge 3) 4 [661] 661 root118Tail11
    (if 3 ≤ root118Candidates10.length then search edge 3 [661, 2791] root118Candidates10 3452 winner else winner) = _
  rw [if_neg (by decide)]
def root118Candidates11 : List Nat := [5641, 13759, 22369]
set_option maxRecDepth 8192 in
theorem root118Candidates11_checked : candidates edge 4 661 2857 root118Tail12 winner.bound = root118Candidates11 := by
  unfold candidates
  have row : edge 2857 = (SortedNatTree.contains FastRows.row2857) := by
    change Data.row2857.contains = (SortedNatTree.contains FastRows.row2857)
    exact FastRows.row2857_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root118Child11 : search edge 3 [661, 2857] root118Candidates11 3518 winner = winner := by decide
theorem root118Step11 : scan edge (search edge 3) 4 [661] 661 root118Tail11 winner =
    scan edge (search edge 3) 4 [661] 661 root118Tail12 winner := by
  rw [root118Tail11, scan, if_neg (by decide), root118Candidates11_checked]
  change scan edge (search edge 3) 4 [661] 661 root118Tail12
    (if 3 ≤ root118Candidates11.length then search edge 3 [661, 2857] root118Candidates11 3518 winner else winner) = _
  rw [if_pos (by decide), root118Child11]
def root118Candidates12 : List Nat := [9343]
set_option maxRecDepth 8192 in
theorem root118Candidates12_checked : candidates edge 4 661 2953 root118Tail13 winner.bound = root118Candidates12 := by
  unfold candidates
  have row : edge 2953 = (SortedNatTree.contains FastRows.row2953) := by
    change Data.row2953.contains = (SortedNatTree.contains FastRows.row2953)
    exact FastRows.row2953_checked.symm
  rw [row]
  decide
theorem root118Step12 : scan edge (search edge 3) 4 [661] 661 root118Tail12 winner =
    scan edge (search edge 3) 4 [661] 661 root118Tail13 winner := by
  rw [root118Tail12, scan, if_neg (by decide), root118Candidates12_checked]
  change scan edge (search edge 3) 4 [661] 661 root118Tail13
    (if 3 ≤ root118Candidates12.length then search edge 3 [661, 2953] root118Candidates12 3614 winner else winner) = _
  rw [if_neg (by decide)]
def root118Candidates13 : List Nat := [7159]
set_option maxRecDepth 8192 in
theorem root118Candidates13_checked : candidates edge 4 661 3331 root118Tail14 winner.bound = root118Candidates13 := by
  unfold candidates
  have row : edge 3331 = (SortedNatTree.contains FastRows.row3331) := by
    change Data.row3331.contains = (SortedNatTree.contains FastRows.row3331)
    exact FastRows.row3331_checked.symm
  rw [row]
  decide
theorem root118Step13 : scan edge (search edge 3) 4 [661] 661 root118Tail13 winner =
    scan edge (search edge 3) 4 [661] 661 root118Tail14 winner := by
  rw [root118Tail13, scan, if_neg (by decide), root118Candidates13_checked]
  change scan edge (search edge 3) 4 [661] 661 root118Tail14
    (if 3 ≤ root118Candidates13.length then search edge 3 [661, 3331] root118Candidates13 3992 winner else winner) = _
  rw [if_neg (by decide)]
def root118Candidates14 : List Nat := [14533, 20563]
set_option maxRecDepth 8192 in
theorem root118Candidates14_checked : candidates edge 4 661 3361 root118Tail15 winner.bound = root118Candidates14 := by
  unfold candidates
  have row : edge 3361 = (SortedNatTree.contains FastRows.row3361) := by
    change Data.row3361.contains = (SortedNatTree.contains FastRows.row3361)
    exact FastRows.row3361_checked.symm
  rw [row]
  decide
theorem root118Step14 : scan edge (search edge 3) 4 [661] 661 root118Tail14 winner =
    scan edge (search edge 3) 4 [661] 661 root118Tail15 winner := by
  rw [root118Tail14, scan, if_neg (by decide), root118Candidates14_checked]
  change scan edge (search edge 3) 4 [661] 661 root118Tail15
    (if 3 ≤ root118Candidates14.length then search edge 3 [661, 3361] root118Candidates14 4022 winner else winner) = _
  rw [if_neg (by decide)]
def root118Candidates15 : List Nat := [4759]
set_option maxRecDepth 8192 in
theorem root118Candidates15_checked : candidates edge 4 661 3823 root118Tail16 winner.bound = root118Candidates15 := by
  unfold candidates
  have row : edge 3823 = (SortedNatTree.contains FastRows.row3823) := by
    change Data.row3823.contains = (SortedNatTree.contains FastRows.row3823)
    exact FastRows.row3823_checked.symm
  rw [row]
  decide
theorem root118Step15 : scan edge (search edge 3) 4 [661] 661 root118Tail15 winner =
    scan edge (search edge 3) 4 [661] 661 root118Tail16 winner := by
  rw [root118Tail15, scan, if_neg (by decide), root118Candidates15_checked]
  change scan edge (search edge 3) 4 [661] 661 root118Tail16
    (if 3 ≤ root118Candidates15.length then search edge 3 [661, 3823] root118Candidates15 4484 winner else winner) = _
  rw [if_neg (by decide)]
def root118Candidates16 : List Nat := [20011]
set_option maxRecDepth 8192 in
theorem root118Candidates16_checked : candidates edge 4 661 3931 root118Tail17 winner.bound = root118Candidates16 := by
  unfold candidates
  have row : edge 3931 = (SortedNatTree.contains FastRows.row3931) := by
    change Data.row3931.contains = (SortedNatTree.contains FastRows.row3931)
    exact FastRows.row3931_checked.symm
  rw [row]
  decide
theorem root118Step16 : scan edge (search edge 3) 4 [661] 661 root118Tail16 winner =
    scan edge (search edge 3) 4 [661] 661 root118Tail17 winner := by
  rw [root118Tail16, scan, if_neg (by decide), root118Candidates16_checked]
  change scan edge (search edge 3) 4 [661] 661 root118Tail17
    (if 3 ≤ root118Candidates16.length then search edge 3 [661, 3931] root118Candidates16 4592 winner else winner) = _
  rw [if_neg (by decide)]
def root118Candidates17 : List Nat := [6151]
set_option maxRecDepth 8192 in
theorem root118Candidates17_checked : candidates edge 4 661 4759 root118Tail18 winner.bound = root118Candidates17 := by
  unfold candidates
  have row : edge 4759 = (SortedNatTree.contains FastRows.row4759) := by
    change Data.row4759.contains = (SortedNatTree.contains FastRows.row4759)
    exact FastRows.row4759_checked.symm
  rw [row]
  decide
theorem root118Step17 : scan edge (search edge 3) 4 [661] 661 root118Tail17 winner =
    scan edge (search edge 3) 4 [661] 661 root118Tail18 winner := by
  rw [root118Tail17, scan, if_neg (by decide), root118Candidates17_checked]
  change scan edge (search edge 3) 4 [661] 661 root118Tail18
    (if 3 ≤ root118Candidates17.length then search edge 3 [661, 4759] root118Candidates17 5420 winner else winner) = _
  rw [if_neg (by decide)]
def root118Candidates18 : List Nat := []
set_option maxRecDepth 8192 in
theorem root118Candidates18_checked : candidates edge 4 661 5503 root118Tail19 winner.bound = root118Candidates18 := by
  unfold candidates
  have row : edge 5503 = (SortedNatTree.contains FastRows.row5503) := by
    change Data.row5503.contains = (SortedNatTree.contains FastRows.row5503)
    exact FastRows.row5503_checked.symm
  rw [row]
  decide
theorem root118Step18 : scan edge (search edge 3) 4 [661] 661 root118Tail18 winner =
    scan edge (search edge 3) 4 [661] 661 root118Tail19 winner := by
  rw [root118Tail18, scan, if_neg (by decide), root118Candidates18_checked]
  change scan edge (search edge 3) 4 [661] 661 root118Tail19
    (if 3 ≤ root118Candidates18.length then search edge 3 [661, 5503] root118Candidates18 6164 winner else winner) = _
  rw [if_neg (by decide)]
def root118Candidates19 : List Nat := [11173]
set_option maxRecDepth 8192 in
theorem root118Candidates19_checked : candidates edge 4 661 5641 root118Tail20 winner.bound = root118Candidates19 := by
  unfold candidates
  have row : edge 5641 = (SortedNatTree.contains FastRows.row5641) := by
    change Data.row5641.contains = (SortedNatTree.contains FastRows.row5641)
    exact FastRows.row5641_checked.symm
  rw [row]
  decide
theorem root118Step19 : scan edge (search edge 3) 4 [661] 661 root118Tail19 winner =
    scan edge (search edge 3) 4 [661] 661 root118Tail20 winner := by
  rw [root118Tail19, scan, if_neg (by decide), root118Candidates19_checked]
  change scan edge (search edge 3) 4 [661] 661 root118Tail20
    (if 3 ≤ root118Candidates19.length then search edge 3 [661, 5641] root118Candidates19 6302 winner else winner) = _
  rw [if_neg (by decide)]
def root118Candidates20 : List Nat := [14827]
set_option maxRecDepth 8192 in
theorem root118Candidates20_checked : candidates edge 4 661 6151 root118Tail21 winner.bound = root118Candidates20 := by
  unfold candidates
  have row : edge 6151 = (SortedNatTree.contains FastRows.row6151) := by
    change Data.row6151.contains = (SortedNatTree.contains FastRows.row6151)
    exact FastRows.row6151_checked.symm
  rw [row]
  decide
theorem root118Step20 : scan edge (search edge 3) 4 [661] 661 root118Tail20 winner =
    scan edge (search edge 3) 4 [661] 661 root118Tail21 winner := by
  rw [root118Tail20, scan, if_neg (by decide), root118Candidates20_checked]
  change scan edge (search edge 3) 4 [661] 661 root118Tail21
    (if 3 ≤ root118Candidates20.length then search edge 3 [661, 6151] root118Candidates20 6812 winner else winner) = _
  rw [if_neg (by decide)]
theorem root118Scan21 : scan edge (search edge 3) 4 [661] 661 root118Tail21 winner = winner := by rw [root118Tail21, scan, if_pos (by decide)]
theorem root118Scan20 : scan edge (search edge 3) 4 [661] 661 root118Tail20 winner = winner :=
  (root118Step20).trans root118Scan21
theorem root118Scan19 : scan edge (search edge 3) 4 [661] 661 root118Tail19 winner = winner :=
  (root118Step19).trans root118Scan20
theorem root118Scan18 : scan edge (search edge 3) 4 [661] 661 root118Tail18 winner = winner :=
  (root118Step18).trans root118Scan19
theorem root118Scan17 : scan edge (search edge 3) 4 [661] 661 root118Tail17 winner = winner :=
  (root118Step17).trans root118Scan18
theorem root118Scan16 : scan edge (search edge 3) 4 [661] 661 root118Tail16 winner = winner :=
  (root118Step16).trans root118Scan17
theorem root118Scan15 : scan edge (search edge 3) 4 [661] 661 root118Tail15 winner = winner :=
  (root118Step15).trans root118Scan16
theorem root118Scan14 : scan edge (search edge 3) 4 [661] 661 root118Tail14 winner = winner :=
  (root118Step14).trans root118Scan15
theorem root118Scan13 : scan edge (search edge 3) 4 [661] 661 root118Tail13 winner = winner :=
  (root118Step13).trans root118Scan14
theorem root118Scan12 : scan edge (search edge 3) 4 [661] 661 root118Tail12 winner = winner :=
  (root118Step12).trans root118Scan13
theorem root118Scan11 : scan edge (search edge 3) 4 [661] 661 root118Tail11 winner = winner :=
  (root118Step11).trans root118Scan12
theorem root118Scan10 : scan edge (search edge 3) 4 [661] 661 root118Tail10 winner = winner :=
  (root118Step10).trans root118Scan11
theorem root118Scan9 : scan edge (search edge 3) 4 [661] 661 root118Tail9 winner = winner :=
  (root118Step9).trans root118Scan10
theorem root118Scan8 : scan edge (search edge 3) 4 [661] 661 root118Tail8 winner = winner :=
  (root118Step8).trans root118Scan9
theorem root118Scan7 : scan edge (search edge 3) 4 [661] 661 root118Tail7 winner = winner :=
  (root118Step7).trans root118Scan8
theorem root118Scan6 : scan edge (search edge 3) 4 [661] 661 root118Tail6 winner = winner :=
  (root118Step6).trans root118Scan7
theorem root118Scan5 : scan edge (search edge 3) 4 [661] 661 root118Tail5 winner = winner :=
  (root118Step5).trans root118Scan6
theorem root118Scan4 : scan edge (search edge 3) 4 [661] 661 root118Tail4 winner = winner :=
  (root118Step4).trans root118Scan5
theorem root118Scan3 : scan edge (search edge 3) 4 [661] 661 root118Tail3 winner = winner :=
  (root118Step3).trans root118Scan4
theorem root118Scan2 : scan edge (search edge 3) 4 [661] 661 root118Tail2 winner = winner :=
  (root118Step2).trans root118Scan3
theorem root118Scan1 : scan edge (search edge 3) 4 [661] 661 root118Tail1 winner = winner :=
  (root118Step1).trans root118Scan2
theorem root118Scan0 : scan edge (search edge 3) 4 [661] 661 root118Tail0 winner = winner :=
  (root118Step0).trans root118Scan1
set_option maxRecDepth 4096 in
theorem root118Branch : search edge 4 [661] root118Candidates 661 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [661] 661 root118Tail0 winner = winner
  exact root118Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
