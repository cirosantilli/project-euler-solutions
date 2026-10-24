import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root121Candidates : List Nat := Data.row683
set_option maxRecDepth 8192 in
theorem root121Candidates_checked : candidates edge 5 0 683 tail122 winner.bound = root121Candidates := by
  apply candidates_eq_row edge 5 0 683 tail122 root121Candidates winner.bound tail122_ordered
  · decide
  · decide
  · funext q
    rfl
def root121Tail62 : List Nat := []
def root121Tail61 : List Nat := 23747 :: root121Tail62
def root121Tail60 : List Nat := 23687 :: root121Tail61
def root121Tail59 : List Nat := 22937 :: root121Tail60
def root121Tail58 : List Nat := 22697 :: root121Tail59
def root121Tail57 : List Nat := 21713 :: root121Tail58
def root121Tail56 : List Nat := 21101 :: root121Tail57
def root121Tail55 : List Nat := 21089 :: root121Tail56
def root121Tail54 : List Nat := 20753 :: root121Tail55
def root121Tail53 : List Nat := 20411 :: root121Tail54
def root121Tail52 : List Nat := 19553 :: root121Tail53
def root121Tail51 : List Nat := 19301 :: root121Tail52
def root121Tail50 : List Nat := 18773 :: root121Tail51
def root121Tail49 : List Nat := 18413 :: root121Tail50
def root121Tail48 : List Nat := 18329 :: root121Tail49
def root121Tail47 : List Nat := 18143 :: root121Tail48
def root121Tail46 : List Nat := 16487 :: root121Tail47
def root121Tail45 : List Nat := 16067 :: root121Tail46
def root121Tail44 : List Nat := 14549 :: root121Tail45
def root121Tail43 : List Nat := 14537 :: root121Tail44
def root121Tail42 : List Nat := 14423 :: root121Tail43
def root121Tail41 : List Nat := 13781 :: root121Tail42
def root121Tail40 : List Nat := 13337 :: root121Tail41
def root121Tail39 : List Nat := 13121 :: root121Tail40
def root121Tail38 : List Nat := 13103 :: root121Tail39
def root121Tail37 : List Nat := 12689 :: root121Tail38
def root121Tail36 : List Nat := 12641 :: root121Tail37
def root121Tail35 : List Nat := 11093 :: root121Tail36
def root121Tail34 : List Nat := 10859 :: root121Tail35
def root121Tail33 : List Nat := 10499 :: root121Tail34
def root121Tail32 : List Nat := 10331 :: root121Tail33
def root121Tail31 : List Nat := 10247 :: root121Tail32
def root121Tail30 : List Nat := 9467 :: root121Tail31
def root121Tail29 : List Nat := 9323 :: root121Tail30
def root121Tail28 : List Nat := 9311 :: root121Tail29
def root121Tail27 : List Nat := 8963 :: root121Tail28
def root121Tail26 : List Nat := 8861 :: root121Tail27
def root121Tail25 : List Nat := 8837 :: root121Tail26
def root121Tail24 : List Nat := 7643 :: root121Tail25
def root121Tail23 : List Nat := 7499 :: root121Tail24
def root121Tail22 : List Nat := 6659 :: root121Tail23
def root121Tail21 : List Nat := 6653 :: root121Tail22
def root121Tail20 : List Nat := 6317 :: root121Tail21
def root121Tail19 : List Nat := 6131 :: root121Tail20
def root121Tail18 : List Nat := 6047 :: root121Tail19
def root121Tail17 : List Nat := 5657 :: root121Tail18
def root121Tail16 : List Nat := 4937 :: root121Tail17
def root121Tail15 : List Nat := 4679 :: root121Tail16
def root121Tail14 : List Nat := 4649 :: root121Tail15
def root121Tail13 : List Nat := 4637 :: root121Tail14
def root121Tail12 : List Nat := 4073 :: root121Tail13
def root121Tail11 : List Nat := 3989 :: root121Tail12
def root121Tail10 : List Nat := 3929 :: root121Tail11
def root121Tail9 : List Nat := 3797 :: root121Tail10
def root121Tail8 : List Nat := 3209 :: root121Tail9
def root121Tail7 : List Nat := 3083 :: root121Tail8
def root121Tail6 : List Nat := 2957 :: root121Tail7
def root121Tail5 : List Nat := 2939 :: root121Tail6
def root121Tail4 : List Nat := 2843 :: root121Tail5
def root121Tail3 : List Nat := 2417 :: root121Tail4
def root121Tail2 : List Nat := 1499 :: root121Tail3
def root121Tail1 : List Nat := 911 :: root121Tail2
def root121Tail0 : List Nat := 719 :: root121Tail1
def root121Candidates0 : List Nat := [911, 6317, 8963, 10331, 19553]
set_option maxRecDepth 8192 in
theorem root121Candidates0_checked : candidates edge 4 683 719 root121Tail1 winner.bound = root121Candidates0 := by
  unfold candidates
  have row : edge 719 = (SortedNatTree.contains FastRows.row719) := by
    change Data.row719.contains = (SortedNatTree.contains FastRows.row719)
    exact FastRows.row719_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root121Child0 : search edge 3 [683, 719] root121Candidates0 1402 winner = winner := by decide
theorem root121Step0 : scan edge (search edge 3) 4 [683] 683 root121Tail0 winner =
    scan edge (search edge 3) 4 [683] 683 root121Tail1 winner := by
  rw [root121Tail0, scan, if_neg (by decide), root121Candidates0_checked]
  change scan edge (search edge 3) 4 [683] 683 root121Tail1
    (if 3 ≤ root121Candidates0.length then search edge 3 [683, 719] root121Candidates0 1402 winner else winner) = _
  rw [if_pos (by decide), root121Child0]
def root121Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root121Candidates1_checked : candidates edge 4 683 911 root121Tail2 winner.bound = root121Candidates1 := by
  unfold candidates
  have row : edge 911 = (SortedNatTree.contains FastRows.row911) := by
    change Data.row911.contains = (SortedNatTree.contains FastRows.row911)
    exact FastRows.row911_checked.symm
  rw [row]
  decide
theorem root121Step1 : scan edge (search edge 3) 4 [683] 683 root121Tail1 winner =
    scan edge (search edge 3) 4 [683] 683 root121Tail2 winner := by
  rw [root121Tail1, scan, if_neg (by decide), root121Candidates1_checked]
  change scan edge (search edge 3) 4 [683] 683 root121Tail2
    (if 3 ≤ root121Candidates1.length then search edge 3 [683, 911] root121Candidates1 1594 winner else winner) = _
  rw [if_neg (by decide)]
def root121Candidates2 : List Nat := [22697]
set_option maxRecDepth 8192 in
theorem root121Candidates2_checked : candidates edge 4 683 1499 root121Tail3 winner.bound = root121Candidates2 := by
  unfold candidates
  have row : edge 1499 = (SortedNatTree.contains FastRows.row1499) := by
    change Data.row1499.contains = (SortedNatTree.contains FastRows.row1499)
    exact FastRows.row1499_checked.symm
  rw [row]
  decide
theorem root121Step2 : scan edge (search edge 3) 4 [683] 683 root121Tail2 winner =
    scan edge (search edge 3) 4 [683] 683 root121Tail3 winner := by
  rw [root121Tail2, scan, if_neg (by decide), root121Candidates2_checked]
  change scan edge (search edge 3) 4 [683] 683 root121Tail3
    (if 3 ≤ root121Candidates2.length then search edge 3 [683, 1499] root121Candidates2 2182 winner else winner) = _
  rw [if_neg (by decide)]
def root121Candidates3 : List Nat := [18143, 20753]
set_option maxRecDepth 8192 in
theorem root121Candidates3_checked : candidates edge 4 683 2417 root121Tail4 winner.bound = root121Candidates3 := by
  unfold candidates
  have row : edge 2417 = (SortedNatTree.contains FastRows.row2417) := by
    change Data.row2417.contains = (SortedNatTree.contains FastRows.row2417)
    exact FastRows.row2417_checked.symm
  rw [row]
  decide
theorem root121Step3 : scan edge (search edge 3) 4 [683] 683 root121Tail3 winner =
    scan edge (search edge 3) 4 [683] 683 root121Tail4 winner := by
  rw [root121Tail3, scan, if_neg (by decide), root121Candidates3_checked]
  change scan edge (search edge 3) 4 [683] 683 root121Tail4
    (if 3 ≤ root121Candidates3.length then search edge 3 [683, 2417] root121Candidates3 3100 winner else winner) = _
  rw [if_neg (by decide)]
def root121Candidates4 : List Nat := [3989, 5657, 6047]
set_option maxRecDepth 8192 in
theorem root121Candidates4_checked : candidates edge 4 683 2843 root121Tail5 winner.bound = root121Candidates4 := by
  unfold candidates
  have row : edge 2843 = (SortedNatTree.contains FastRows.row2843) := by
    change Data.row2843.contains = (SortedNatTree.contains FastRows.row2843)
    exact FastRows.row2843_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root121Child4 : search edge 3 [683, 2843] root121Candidates4 3526 winner = winner := by decide
theorem root121Step4 : scan edge (search edge 3) 4 [683] 683 root121Tail4 winner =
    scan edge (search edge 3) 4 [683] 683 root121Tail5 winner := by
  rw [root121Tail4, scan, if_neg (by decide), root121Candidates4_checked]
  change scan edge (search edge 3) 4 [683] 683 root121Tail5
    (if 3 ≤ root121Candidates4.length then search edge 3 [683, 2843] root121Candidates4 3526 winner else winner) = _
  rw [if_pos (by decide), root121Child4]
def root121Candidates5 : List Nat := [4679]
set_option maxRecDepth 8192 in
theorem root121Candidates5_checked : candidates edge 4 683 2939 root121Tail6 winner.bound = root121Candidates5 := by
  unfold candidates
  have row : edge 2939 = (SortedNatTree.contains FastRows.row2939) := by
    change Data.row2939.contains = (SortedNatTree.contains FastRows.row2939)
    exact FastRows.row2939_checked.symm
  rw [row]
  decide
theorem root121Step5 : scan edge (search edge 3) 4 [683] 683 root121Tail5 winner =
    scan edge (search edge 3) 4 [683] 683 root121Tail6 winner := by
  rw [root121Tail5, scan, if_neg (by decide), root121Candidates5_checked]
  change scan edge (search edge 3) 4 [683] 683 root121Tail6
    (if 3 ≤ root121Candidates5.length then search edge 3 [683, 2939] root121Candidates5 3622 winner else winner) = _
  rw [if_neg (by decide)]
def root121Candidates6 : List Nat := [9323]
set_option maxRecDepth 8192 in
theorem root121Candidates6_checked : candidates edge 4 683 2957 root121Tail7 winner.bound = root121Candidates6 := by
  unfold candidates
  have row : edge 2957 = (SortedNatTree.contains FastRows.row2957) := by
    change Data.row2957.contains = (SortedNatTree.contains FastRows.row2957)
    exact FastRows.row2957_checked.symm
  rw [row]
  decide
theorem root121Step6 : scan edge (search edge 3) 4 [683] 683 root121Tail6 winner =
    scan edge (search edge 3) 4 [683] 683 root121Tail7 winner := by
  rw [root121Tail6, scan, if_neg (by decide), root121Candidates6_checked]
  change scan edge (search edge 3) 4 [683] 683 root121Tail7
    (if 3 ≤ root121Candidates6.length then search edge 3 [683, 2957] root121Candidates6 3640 winner else winner) = _
  rw [if_neg (by decide)]
def root121Candidates7 : List Nat := [3797, 8963]
set_option maxRecDepth 8192 in
theorem root121Candidates7_checked : candidates edge 4 683 3083 root121Tail8 winner.bound = root121Candidates7 := by
  unfold candidates
  have row : edge 3083 = (SortedNatTree.contains FastRows.row3083) := by
    change Data.row3083.contains = (SortedNatTree.contains FastRows.row3083)
    exact FastRows.row3083_checked.symm
  rw [row]
  decide
theorem root121Step7 : scan edge (search edge 3) 4 [683] 683 root121Tail7 winner =
    scan edge (search edge 3) 4 [683] 683 root121Tail8 winner := by
  rw [root121Tail7, scan, if_neg (by decide), root121Candidates7_checked]
  change scan edge (search edge 3) 4 [683] 683 root121Tail8
    (if 3 ≤ root121Candidates7.length then search edge 3 [683, 3083] root121Candidates7 3766 winner else winner) = _
  rw [if_neg (by decide)]
def root121Candidates8 : List Nat := [6653, 8861, 16067, 16487, 19301]
set_option maxRecDepth 8192 in
theorem root121Candidates8_checked : candidates edge 4 683 3209 root121Tail9 winner.bound = root121Candidates8 := by
  unfold candidates
  have row : edge 3209 = (SortedNatTree.contains FastRows.row3209) := by
    change Data.row3209.contains = (SortedNatTree.contains FastRows.row3209)
    exact FastRows.row3209_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root121Child8 : search edge 3 [683, 3209] root121Candidates8 3892 winner = winner := by decide
theorem root121Step8 : scan edge (search edge 3) 4 [683] 683 root121Tail8 winner =
    scan edge (search edge 3) 4 [683] 683 root121Tail9 winner := by
  rw [root121Tail8, scan, if_neg (by decide), root121Candidates8_checked]
  change scan edge (search edge 3) 4 [683] 683 root121Tail9
    (if 3 ≤ root121Candidates8.length then search edge 3 [683, 3209] root121Candidates8 3892 winner else winner) = _
  rw [if_pos (by decide), root121Child8]
def root121Candidates9 : List Nat := [8861]
set_option maxRecDepth 8192 in
theorem root121Candidates9_checked : candidates edge 4 683 3797 root121Tail10 winner.bound = root121Candidates9 := by
  unfold candidates
  have row : edge 3797 = (SortedNatTree.contains FastRows.row3797) := by
    change Data.row3797.contains = (SortedNatTree.contains FastRows.row3797)
    exact FastRows.row3797_checked.symm
  rw [row]
  decide
theorem root121Step9 : scan edge (search edge 3) 4 [683] 683 root121Tail9 winner =
    scan edge (search edge 3) 4 [683] 683 root121Tail10 winner := by
  rw [root121Tail9, scan, if_neg (by decide), root121Candidates9_checked]
  change scan edge (search edge 3) 4 [683] 683 root121Tail10
    (if 3 ≤ root121Candidates9.length then search edge 3 [683, 3797] root121Candidates9 4480 winner else winner) = _
  rw [if_neg (by decide)]
def root121Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root121Candidates10_checked : candidates edge 4 683 3929 root121Tail11 winner.bound = root121Candidates10 := by
  unfold candidates
  have row : edge 3929 = (SortedNatTree.contains FastRows.row3929) := by
    change Data.row3929.contains = (SortedNatTree.contains FastRows.row3929)
    exact FastRows.row3929_checked.symm
  rw [row]
  decide
theorem root121Step10 : scan edge (search edge 3) 4 [683] 683 root121Tail10 winner =
    scan edge (search edge 3) 4 [683] 683 root121Tail11 winner := by
  rw [root121Tail10, scan, if_neg (by decide), root121Candidates10_checked]
  change scan edge (search edge 3) 4 [683] 683 root121Tail11
    (if 3 ≤ root121Candidates10.length then search edge 3 [683, 3929] root121Candidates10 4612 winner else winner) = _
  rw [if_neg (by decide)]
def root121Candidates11 : List Nat := [8861, 9311, 20411]
set_option maxRecDepth 8192 in
theorem root121Candidates11_checked : candidates edge 4 683 3989 root121Tail12 winner.bound = root121Candidates11 := by
  unfold candidates
  have row : edge 3989 = (SortedNatTree.contains FastRows.row3989) := by
    change Data.row3989.contains = (SortedNatTree.contains FastRows.row3989)
    exact FastRows.row3989_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root121Child11 : search edge 3 [683, 3989] root121Candidates11 4672 winner = winner := by decide
theorem root121Step11 : scan edge (search edge 3) 4 [683] 683 root121Tail11 winner =
    scan edge (search edge 3) 4 [683] 683 root121Tail12 winner := by
  rw [root121Tail11, scan, if_neg (by decide), root121Candidates11_checked]
  change scan edge (search edge 3) 4 [683] 683 root121Tail12
    (if 3 ≤ root121Candidates11.length then search edge 3 [683, 3989] root121Candidates11 4672 winner else winner) = _
  rw [if_pos (by decide), root121Child11]
def root121Candidates12 : List Nat := [8861]
set_option maxRecDepth 8192 in
theorem root121Candidates12_checked : candidates edge 4 683 4073 root121Tail13 winner.bound = root121Candidates12 := by
  unfold candidates
  have row : edge 4073 = (SortedNatTree.contains FastRows.row4073) := by
    change Data.row4073.contains = (SortedNatTree.contains FastRows.row4073)
    exact FastRows.row4073_checked.symm
  rw [row]
  decide
theorem root121Step12 : scan edge (search edge 3) 4 [683] 683 root121Tail12 winner =
    scan edge (search edge 3) 4 [683] 683 root121Tail13 winner := by
  rw [root121Tail12, scan, if_neg (by decide), root121Candidates12_checked]
  change scan edge (search edge 3) 4 [683] 683 root121Tail13
    (if 3 ≤ root121Candidates12.length then search edge 3 [683, 4073] root121Candidates12 4756 winner else winner) = _
  rw [if_neg (by decide)]
def root121Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root121Candidates13_checked : candidates edge 4 683 4637 root121Tail14 winner.bound = root121Candidates13 := by
  unfold candidates
  have row : edge 4637 = (SortedNatTree.contains FastRows.row4637) := by
    change Data.row4637.contains = (SortedNatTree.contains FastRows.row4637)
    exact FastRows.row4637_checked.symm
  rw [row]
  decide
theorem root121Step13 : scan edge (search edge 3) 4 [683] 683 root121Tail13 winner =
    scan edge (search edge 3) 4 [683] 683 root121Tail14 winner := by
  rw [root121Tail13, scan, if_neg (by decide), root121Candidates13_checked]
  change scan edge (search edge 3) 4 [683] 683 root121Tail14
    (if 3 ≤ root121Candidates13.length then search edge 3 [683, 4637] root121Candidates13 5320 winner else winner) = _
  rw [if_neg (by decide)]
def root121Candidates14 : List Nat := [12641, 20411]
set_option maxRecDepth 8192 in
theorem root121Candidates14_checked : candidates edge 4 683 4649 root121Tail15 winner.bound = root121Candidates14 := by
  unfold candidates
  have row : edge 4649 = (SortedNatTree.contains FastRows.row4649) := by
    change Data.row4649.contains = (SortedNatTree.contains FastRows.row4649)
    exact FastRows.row4649_checked.symm
  rw [row]
  decide
theorem root121Step14 : scan edge (search edge 3) 4 [683] 683 root121Tail14 winner =
    scan edge (search edge 3) 4 [683] 683 root121Tail15 winner := by
  rw [root121Tail14, scan, if_neg (by decide), root121Candidates14_checked]
  change scan edge (search edge 3) 4 [683] 683 root121Tail15
    (if 3 ≤ root121Candidates14.length then search edge 3 [683, 4649] root121Candidates14 5332 winner else winner) = _
  rw [if_neg (by decide)]
def root121Candidates15 : List Nat := [6653, 18413]
set_option maxRecDepth 8192 in
theorem root121Candidates15_checked : candidates edge 4 683 4679 root121Tail16 winner.bound = root121Candidates15 := by
  unfold candidates
  have row : edge 4679 = (SortedNatTree.contains FastRows.row4679) := by
    change Data.row4679.contains = (SortedNatTree.contains FastRows.row4679)
    exact FastRows.row4679_checked.symm
  rw [row]
  decide
theorem root121Step15 : scan edge (search edge 3) 4 [683] 683 root121Tail15 winner =
    scan edge (search edge 3) 4 [683] 683 root121Tail16 winner := by
  rw [root121Tail15, scan, if_neg (by decide), root121Candidates15_checked]
  change scan edge (search edge 3) 4 [683] 683 root121Tail16
    (if 3 ≤ root121Candidates15.length then search edge 3 [683, 4679] root121Candidates15 5362 winner else winner) = _
  rw [if_neg (by decide)]
def root121Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root121Candidates16_checked : candidates edge 4 683 4937 root121Tail17 winner.bound = root121Candidates16 := by
  unfold candidates
  have row : edge 4937 = (SortedNatTree.contains FastRows.row4937) := by
    change Data.row4937.contains = (SortedNatTree.contains FastRows.row4937)
    exact FastRows.row4937_checked.symm
  rw [row]
  decide
theorem root121Step16 : scan edge (search edge 3) 4 [683] 683 root121Tail16 winner =
    scan edge (search edge 3) 4 [683] 683 root121Tail17 winner := by
  rw [root121Tail16, scan, if_neg (by decide), root121Candidates16_checked]
  change scan edge (search edge 3) 4 [683] 683 root121Tail17
    (if 3 ≤ root121Candidates16.length then search edge 3 [683, 4937] root121Candidates16 5620 winner else winner) = _
  rw [if_neg (by decide)]
def root121Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root121Candidates17_checked : candidates edge 4 683 5657 root121Tail18 winner.bound = root121Candidates17 := by
  unfold candidates
  have row : edge 5657 = (SortedNatTree.contains FastRows.row5657) := by
    change Data.row5657.contains = (SortedNatTree.contains FastRows.row5657)
    exact FastRows.row5657_checked.symm
  rw [row]
  decide
theorem root121Step17 : scan edge (search edge 3) 4 [683] 683 root121Tail17 winner =
    scan edge (search edge 3) 4 [683] 683 root121Tail18 winner := by
  rw [root121Tail17, scan, if_neg (by decide), root121Candidates17_checked]
  change scan edge (search edge 3) 4 [683] 683 root121Tail18
    (if 3 ≤ root121Candidates17.length then search edge 3 [683, 5657] root121Candidates17 6340 winner else winner) = _
  rw [if_neg (by decide)]
def root121Candidates18 : List Nat := [7643, 8963, 9311, 9467]
set_option maxRecDepth 8192 in
theorem root121Candidates18_checked : candidates edge 4 683 6047 root121Tail19 winner.bound = root121Candidates18 := by
  unfold candidates
  have row : edge 6047 = (SortedNatTree.contains FastRows.row6047) := by
    change Data.row6047.contains = (SortedNatTree.contains FastRows.row6047)
    exact FastRows.row6047_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root121Child18 : search edge 3 [683, 6047] root121Candidates18 6730 winner = winner := by decide
theorem root121Step18 : scan edge (search edge 3) 4 [683] 683 root121Tail18 winner =
    scan edge (search edge 3) 4 [683] 683 root121Tail19 winner := by
  rw [root121Tail18, scan, if_neg (by decide), root121Candidates18_checked]
  change scan edge (search edge 3) 4 [683] 683 root121Tail19
    (if 3 ≤ root121Candidates18.length then search edge 3 [683, 6047] root121Candidates18 6730 winner else winner) = _
  rw [if_pos (by decide), root121Child18]
def root121Candidates19 : List Nat := []
set_option maxRecDepth 8192 in
theorem root121Candidates19_checked : candidates edge 4 683 6131 root121Tail20 winner.bound = root121Candidates19 := by
  unfold candidates
  have row : edge 6131 = (SortedNatTree.contains FastRows.row6131) := by
    change Data.row6131.contains = (SortedNatTree.contains FastRows.row6131)
    exact FastRows.row6131_checked.symm
  rw [row]
  decide
theorem root121Step19 : scan edge (search edge 3) 4 [683] 683 root121Tail19 winner =
    scan edge (search edge 3) 4 [683] 683 root121Tail20 winner := by
  rw [root121Tail19, scan, if_neg (by decide), root121Candidates19_checked]
  change scan edge (search edge 3) 4 [683] 683 root121Tail20
    (if 3 ≤ root121Candidates19.length then search edge 3 [683, 6131] root121Candidates19 6814 winner else winner) = _
  rw [if_neg (by decide)]
def root121Candidates20 : List Nat := [9323, 14423, 14549, 18413]
set_option maxRecDepth 8192 in
theorem root121Candidates20_checked : candidates edge 4 683 6317 root121Tail21 winner.bound = root121Candidates20 := by
  unfold candidates
  have row : edge 6317 = (SortedNatTree.contains FastRows.row6317) := by
    change Data.row6317.contains = (SortedNatTree.contains FastRows.row6317)
    exact FastRows.row6317_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root121Child20 : search edge 3 [683, 6317] root121Candidates20 7000 winner = winner := by decide
theorem root121Step20 : scan edge (search edge 3) 4 [683] 683 root121Tail20 winner =
    scan edge (search edge 3) 4 [683] 683 root121Tail21 winner := by
  rw [root121Tail20, scan, if_neg (by decide), root121Candidates20_checked]
  change scan edge (search edge 3) 4 [683] 683 root121Tail21
    (if 3 ≤ root121Candidates20.length then search edge 3 [683, 6317] root121Candidates20 7000 winner else winner) = _
  rw [if_pos (by decide), root121Child20]
theorem root121Scan21 : scan edge (search edge 3) 4 [683] 683 root121Tail21 winner = winner := by rw [root121Tail21, scan, if_pos (by decide)]
theorem root121Scan20 : scan edge (search edge 3) 4 [683] 683 root121Tail20 winner = winner :=
  (root121Step20).trans root121Scan21
theorem root121Scan19 : scan edge (search edge 3) 4 [683] 683 root121Tail19 winner = winner :=
  (root121Step19).trans root121Scan20
theorem root121Scan18 : scan edge (search edge 3) 4 [683] 683 root121Tail18 winner = winner :=
  (root121Step18).trans root121Scan19
theorem root121Scan17 : scan edge (search edge 3) 4 [683] 683 root121Tail17 winner = winner :=
  (root121Step17).trans root121Scan18
theorem root121Scan16 : scan edge (search edge 3) 4 [683] 683 root121Tail16 winner = winner :=
  (root121Step16).trans root121Scan17
theorem root121Scan15 : scan edge (search edge 3) 4 [683] 683 root121Tail15 winner = winner :=
  (root121Step15).trans root121Scan16
theorem root121Scan14 : scan edge (search edge 3) 4 [683] 683 root121Tail14 winner = winner :=
  (root121Step14).trans root121Scan15
theorem root121Scan13 : scan edge (search edge 3) 4 [683] 683 root121Tail13 winner = winner :=
  (root121Step13).trans root121Scan14
theorem root121Scan12 : scan edge (search edge 3) 4 [683] 683 root121Tail12 winner = winner :=
  (root121Step12).trans root121Scan13
theorem root121Scan11 : scan edge (search edge 3) 4 [683] 683 root121Tail11 winner = winner :=
  (root121Step11).trans root121Scan12
theorem root121Scan10 : scan edge (search edge 3) 4 [683] 683 root121Tail10 winner = winner :=
  (root121Step10).trans root121Scan11
theorem root121Scan9 : scan edge (search edge 3) 4 [683] 683 root121Tail9 winner = winner :=
  (root121Step9).trans root121Scan10
theorem root121Scan8 : scan edge (search edge 3) 4 [683] 683 root121Tail8 winner = winner :=
  (root121Step8).trans root121Scan9
theorem root121Scan7 : scan edge (search edge 3) 4 [683] 683 root121Tail7 winner = winner :=
  (root121Step7).trans root121Scan8
theorem root121Scan6 : scan edge (search edge 3) 4 [683] 683 root121Tail6 winner = winner :=
  (root121Step6).trans root121Scan7
theorem root121Scan5 : scan edge (search edge 3) 4 [683] 683 root121Tail5 winner = winner :=
  (root121Step5).trans root121Scan6
theorem root121Scan4 : scan edge (search edge 3) 4 [683] 683 root121Tail4 winner = winner :=
  (root121Step4).trans root121Scan5
theorem root121Scan3 : scan edge (search edge 3) 4 [683] 683 root121Tail3 winner = winner :=
  (root121Step3).trans root121Scan4
theorem root121Scan2 : scan edge (search edge 3) 4 [683] 683 root121Tail2 winner = winner :=
  (root121Step2).trans root121Scan3
theorem root121Scan1 : scan edge (search edge 3) 4 [683] 683 root121Tail1 winner = winner :=
  (root121Step1).trans root121Scan2
theorem root121Scan0 : scan edge (search edge 3) 4 [683] 683 root121Tail0 winner = winner :=
  (root121Step0).trans root121Scan1
set_option maxRecDepth 4096 in
theorem root121Branch : search edge 4 [683] root121Candidates 683 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [683] 683 root121Tail0 winner = winner
  exact root121Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
