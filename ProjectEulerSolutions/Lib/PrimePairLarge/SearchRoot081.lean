import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root81Candidates : List Nat := Data.row433
set_option maxRecDepth 8192 in
theorem root81Candidates_checked : candidates edge 5 0 433 tail82 winner.bound = root81Candidates := by
  apply candidates_eq_row edge 5 0 433 tail82 root81Candidates winner.bound tail82_ordered
  · decide
  · decide
  · funext q
    rfl
def root81Tail76 : List Nat := []
def root81Tail75 : List Nat := 24889 :: root81Tail76
def root81Tail74 : List Nat := 24481 :: root81Tail75
def root81Tail73 : List Nat := 23887 :: root81Tail74
def root81Tail72 : List Nat := 23773 :: root81Tail73
def root81Tail71 : List Nat := 23671 :: root81Tail72
def root81Tail70 : List Nat := 23557 :: root81Tail71
def root81Tail69 : List Nat := 23431 :: root81Tail70
def root81Tail68 : List Nat := 21739 :: root81Tail69
def root81Tail67 : List Nat := 21613 :: root81Tail68
def root81Tail66 : List Nat := 21211 :: root81Tail67
def root81Tail65 : List Nat := 20773 :: root81Tail66
def root81Tail64 : List Nat := 19681 :: root81Tail65
def root81Tail63 : List Nat := 19531 :: root81Tail64
def root81Tail62 : List Nat := 19219 :: root81Tail63
def root81Tail61 : List Nat := 18973 :: root81Tail62
def root81Tail60 : List Nat := 18523 :: root81Tail61
def root81Tail59 : List Nat := 18433 :: root81Tail60
def root81Tail58 : List Nat := 17971 :: root81Tail59
def root81Tail57 : List Nat := 17959 :: root81Tail58
def root81Tail56 : List Nat := 17707 :: root81Tail57
def root81Tail55 : List Nat := 17551 :: root81Tail56
def root81Tail54 : List Nat := 17539 :: root81Tail55
def root81Tail53 : List Nat := 17497 :: root81Tail54
def root81Tail52 : List Nat := 17431 :: root81Tail53
def root81Tail51 : List Nat := 17419 :: root81Tail52
def root81Tail50 : List Nat := 17293 :: root81Tail51
def root81Tail49 : List Nat := 16879 :: root81Tail50
def root81Tail48 : List Nat := 16831 :: root81Tail49
def root81Tail47 : List Nat := 15901 :: root81Tail48
def root81Tail46 : List Nat := 14779 :: root81Tail47
def root81Tail45 : List Nat := 14653 :: root81Tail46
def root81Tail44 : List Nat := 13681 :: root81Tail45
def root81Tail43 : List Nat := 13441 :: root81Tail44
def root81Tail42 : List Nat := 13219 :: root81Tail43
def root81Tail41 : List Nat := 13177 :: root81Tail42
def root81Tail40 : List Nat := 12547 :: root81Tail41
def root81Tail39 : List Nat := 11839 :: root81Tail40
def root81Tail38 : List Nat := 11833 :: root81Tail39
def root81Tail37 : List Nat := 11677 :: root81Tail38
def root81Tail36 : List Nat := 11503 :: root81Tail37
def root81Tail35 : List Nat := 11173 :: root81Tail36
def root81Tail34 : List Nat := 11113 :: root81Tail35
def root81Tail33 : List Nat := 10039 :: root81Tail34
def root81Tail32 : List Nat := 9901 :: root81Tail33
def root81Tail31 : List Nat := 9787 :: root81Tail32
def root81Tail30 : List Nat := 9613 :: root81Tail31
def root81Tail29 : List Nat := 8629 :: root81Tail30
def root81Tail28 : List Nat := 8167 :: root81Tail29
def root81Tail27 : List Nat := 7717 :: root81Tail28
def root81Tail26 : List Nat := 7393 :: root81Tail27
def root81Tail25 : List Nat := 6781 :: root81Tail26
def root81Tail24 : List Nat := 6379 :: root81Tail25
def root81Tail23 : List Nat := 6247 :: root81Tail24
def root81Tail22 : List Nat := 6091 :: root81Tail23
def root81Tail21 : List Nat := 5407 :: root81Tail22
def root81Tail20 : List Nat := 5281 :: root81Tail21
def root81Tail19 : List Nat := 4903 :: root81Tail20
def root81Tail18 : List Nat := 4663 :: root81Tail19
def root81Tail17 : List Nat := 4483 :: root81Tail18
def root81Tail16 : List Nat := 4441 :: root81Tail17
def root81Tail15 : List Nat := 3739 :: root81Tail16
def root81Tail14 : List Nat := 3697 :: root81Tail15
def root81Tail13 : List Nat := 3613 :: root81Tail14
def root81Tail12 : List Nat := 3529 :: root81Tail13
def root81Tail11 : List Nat := 3391 :: root81Tail12
def root81Tail10 : List Nat := 2857 :: root81Tail11
def root81Tail9 : List Nat := 2719 :: root81Tail10
def root81Tail8 : List Nat := 2203 :: root81Tail9
def root81Tail7 : List Nat := 1801 :: root81Tail8
def root81Tail6 : List Nat := 1471 :: root81Tail7
def root81Tail5 : List Nat := 1087 :: root81Tail6
def root81Tail4 : List Nat := 1009 :: root81Tail5
def root81Tail3 : List Nat := 883 :: root81Tail4
def root81Tail2 : List Nat := 859 :: root81Tail3
def root81Tail1 : List Nat := 787 :: root81Tail2
def root81Tail0 : List Nat := 571 :: root81Tail1
def root81Candidates0 : List Nat := [4483, 11503, 11839, 13219]
set_option maxRecDepth 8192 in
theorem root81Candidates0_checked : candidates edge 4 433 571 root81Tail1 winner.bound = root81Candidates0 := by
  unfold candidates
  have row : edge 571 = (SortedNatTree.contains FastRows.row571) := by
    change Data.row571.contains = (SortedNatTree.contains FastRows.row571)
    exact FastRows.row571_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root81Child0 : search edge 3 [433, 571] root81Candidates0 1004 winner = winner := by decide
theorem root81Step0 : scan edge (search edge 3) 4 [433] 433 root81Tail0 winner =
    scan edge (search edge 3) 4 [433] 433 root81Tail1 winner := by
  rw [root81Tail0, scan, if_neg (by decide), root81Candidates0_checked]
  change scan edge (search edge 3) 4 [433] 433 root81Tail1
    (if 3 ≤ root81Candidates0.length then search edge 3 [433, 571] root81Candidates0 1004 winner else winner) = _
  rw [if_pos (by decide), root81Child0]
def root81Candidates1 : List Nat := [10039, 23773]
set_option maxRecDepth 8192 in
theorem root81Candidates1_checked : candidates edge 4 433 787 root81Tail2 winner.bound = root81Candidates1 := by
  unfold candidates
  have row : edge 787 = (SortedNatTree.contains FastRows.row787) := by
    change Data.row787.contains = (SortedNatTree.contains FastRows.row787)
    exact FastRows.row787_checked.symm
  rw [row]
  decide
theorem root81Step1 : scan edge (search edge 3) 4 [433] 433 root81Tail1 winner =
    scan edge (search edge 3) 4 [433] 433 root81Tail2 winner := by
  rw [root81Tail1, scan, if_neg (by decide), root81Candidates1_checked]
  change scan edge (search edge 3) 4 [433] 433 root81Tail2
    (if 3 ≤ root81Candidates1.length then search edge 3 [433, 787] root81Candidates1 1220 winner else winner) = _
  rw [if_neg (by decide)]
def root81Candidates2 : List Nat := [3391, 18433]
set_option maxRecDepth 8192 in
theorem root81Candidates2_checked : candidates edge 4 433 859 root81Tail3 winner.bound = root81Candidates2 := by
  unfold candidates
  have row : edge 859 = (SortedNatTree.contains FastRows.row859) := by
    change Data.row859.contains = (SortedNatTree.contains FastRows.row859)
    exact FastRows.row859_checked.symm
  rw [row]
  decide
theorem root81Step2 : scan edge (search edge 3) 4 [433] 433 root81Tail2 winner =
    scan edge (search edge 3) 4 [433] 433 root81Tail3 winner := by
  rw [root81Tail2, scan, if_neg (by decide), root81Candidates2_checked]
  change scan edge (search edge 3) 4 [433] 433 root81Tail3
    (if 3 ≤ root81Candidates2.length then search edge 3 [433, 859] root81Candidates2 1292 winner else winner) = _
  rw [if_neg (by decide)]
def root81Candidates3 : List Nat := [17293, 19219]
set_option maxRecDepth 8192 in
theorem root81Candidates3_checked : candidates edge 4 433 883 root81Tail4 winner.bound = root81Candidates3 := by
  unfold candidates
  have row : edge 883 = (SortedNatTree.contains FastRows.row883) := by
    change Data.row883.contains = (SortedNatTree.contains FastRows.row883)
    exact FastRows.row883_checked.symm
  rw [row]
  decide
theorem root81Step3 : scan edge (search edge 3) 4 [433] 433 root81Tail3 winner =
    scan edge (search edge 3) 4 [433] 433 root81Tail4 winner := by
  rw [root81Tail3, scan, if_neg (by decide), root81Candidates3_checked]
  change scan edge (search edge 3) 4 [433] 433 root81Tail4
    (if 3 ≤ root81Candidates3.length then search edge 3 [433, 883] root81Candidates3 1316 winner else winner) = _
  rw [if_neg (by decide)]
def root81Candidates4 : List Nat := [3613, 7393, 16831, 18523]
set_option maxRecDepth 8192 in
theorem root81Candidates4_checked : candidates edge 4 433 1009 root81Tail5 winner.bound = root81Candidates4 := by
  unfold candidates
  have row : edge 1009 = (SortedNatTree.contains FastRows.row1009) := by
    change Data.row1009.contains = (SortedNatTree.contains FastRows.row1009)
    exact FastRows.row1009_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root81Child4 : search edge 3 [433, 1009] root81Candidates4 1442 winner = winner := by decide
theorem root81Step4 : scan edge (search edge 3) 4 [433] 433 root81Tail4 winner =
    scan edge (search edge 3) 4 [433] 433 root81Tail5 winner := by
  rw [root81Tail4, scan, if_neg (by decide), root81Candidates4_checked]
  change scan edge (search edge 3) 4 [433] 433 root81Tail5
    (if 3 ≤ root81Candidates4.length then search edge 3 [433, 1009] root81Candidates4 1442 winner else winner) = _
  rw [if_pos (by decide), root81Child4]
def root81Candidates5 : List Nat := [6091, 17971, 24481]
set_option maxRecDepth 8192 in
theorem root81Candidates5_checked : candidates edge 4 433 1087 root81Tail6 winner.bound = root81Candidates5 := by
  unfold candidates
  have row : edge 1087 = (SortedNatTree.contains FastRows.row1087) := by
    change Data.row1087.contains = (SortedNatTree.contains FastRows.row1087)
    exact FastRows.row1087_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root81Child5 : search edge 3 [433, 1087] root81Candidates5 1520 winner = winner := by decide
theorem root81Step5 : scan edge (search edge 3) 4 [433] 433 root81Tail5 winner =
    scan edge (search edge 3) 4 [433] 433 root81Tail6 winner := by
  rw [root81Tail5, scan, if_neg (by decide), root81Candidates5_checked]
  change scan edge (search edge 3) 4 [433] 433 root81Tail6
    (if 3 ≤ root81Candidates5.length then search edge 3 [433, 1087] root81Candidates5 1520 winner else winner) = _
  rw [if_pos (by decide), root81Child5]
def root81Candidates6 : List Nat := [3613, 9787, 16831, 17431]
set_option maxRecDepth 8192 in
theorem root81Candidates6_checked : candidates edge 4 433 1471 root81Tail7 winner.bound = root81Candidates6 := by
  unfold candidates
  have row : edge 1471 = (SortedNatTree.contains FastRows.row1471) := by
    change Data.row1471.contains = (SortedNatTree.contains FastRows.row1471)
    exact FastRows.row1471_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root81Child6 : search edge 3 [433, 1471] root81Candidates6 1904 winner = winner := by decide
theorem root81Step6 : scan edge (search edge 3) 4 [433] 433 root81Tail6 winner =
    scan edge (search edge 3) 4 [433] 433 root81Tail7 winner := by
  rw [root81Tail6, scan, if_neg (by decide), root81Candidates6_checked]
  change scan edge (search edge 3) 4 [433] 433 root81Tail7
    (if 3 ≤ root81Candidates6.length then search edge 3 [433, 1471] root81Candidates6 1904 winner else winner) = _
  rw [if_pos (by decide), root81Child6]
def root81Candidates7 : List Nat := [3613]
set_option maxRecDepth 8192 in
theorem root81Candidates7_checked : candidates edge 4 433 1801 root81Tail8 winner.bound = root81Candidates7 := by
  unfold candidates
  have row : edge 1801 = (SortedNatTree.contains FastRows.row1801) := by
    change Data.row1801.contains = (SortedNatTree.contains FastRows.row1801)
    exact FastRows.row1801_checked.symm
  rw [row]
  decide
theorem root81Step7 : scan edge (search edge 3) 4 [433] 433 root81Tail7 winner =
    scan edge (search edge 3) 4 [433] 433 root81Tail8 winner := by
  rw [root81Tail7, scan, if_neg (by decide), root81Candidates7_checked]
  change scan edge (search edge 3) 4 [433] 433 root81Tail8
    (if 3 ≤ root81Candidates7.length then search edge 3 [433, 1801] root81Candidates7 2234 winner else winner) = _
  rw [if_neg (by decide)]
def root81Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root81Candidates8_checked : candidates edge 4 433 2203 root81Tail9 winner.bound = root81Candidates8 := by
  unfold candidates
  have row : edge 2203 = (SortedNatTree.contains FastRows.row2203) := by
    change Data.row2203.contains = (SortedNatTree.contains FastRows.row2203)
    exact FastRows.row2203_checked.symm
  rw [row]
  decide
theorem root81Step8 : scan edge (search edge 3) 4 [433] 433 root81Tail8 winner =
    scan edge (search edge 3) 4 [433] 433 root81Tail9 winner := by
  rw [root81Tail8, scan, if_neg (by decide), root81Candidates8_checked]
  change scan edge (search edge 3) 4 [433] 433 root81Tail9
    (if 3 ≤ root81Candidates8.length then search edge 3 [433, 2203] root81Candidates8 2636 winner else winner) = _
  rw [if_neg (by decide)]
def root81Candidates9 : List Nat := [3391]
set_option maxRecDepth 8192 in
theorem root81Candidates9_checked : candidates edge 4 433 2719 root81Tail10 winner.bound = root81Candidates9 := by
  unfold candidates
  have row : edge 2719 = (SortedNatTree.contains FastRows.row2719) := by
    change Data.row2719.contains = (SortedNatTree.contains FastRows.row2719)
    exact FastRows.row2719_checked.symm
  rw [row]
  decide
theorem root81Step9 : scan edge (search edge 3) 4 [433] 433 root81Tail9 winner =
    scan edge (search edge 3) 4 [433] 433 root81Tail10 winner := by
  rw [root81Tail9, scan, if_neg (by decide), root81Candidates9_checked]
  change scan edge (search edge 3) 4 [433] 433 root81Tail10
    (if 3 ≤ root81Candidates9.length then search edge 3 [433, 2719] root81Candidates9 3152 winner else winner) = _
  rw [if_neg (by decide)]
def root81Candidates10 : List Nat := [6091, 11677]
set_option maxRecDepth 8192 in
theorem root81Candidates10_checked : candidates edge 4 433 2857 root81Tail11 winner.bound = root81Candidates10 := by
  unfold candidates
  have row : edge 2857 = (SortedNatTree.contains FastRows.row2857) := by
    change Data.row2857.contains = (SortedNatTree.contains FastRows.row2857)
    exact FastRows.row2857_checked.symm
  rw [row]
  decide
theorem root81Step10 : scan edge (search edge 3) 4 [433] 433 root81Tail10 winner =
    scan edge (search edge 3) 4 [433] 433 root81Tail11 winner := by
  rw [root81Tail10, scan, if_neg (by decide), root81Candidates10_checked]
  change scan edge (search edge 3) 4 [433] 433 root81Tail11
    (if 3 ≤ root81Candidates10.length then search edge 3 [433, 2857] root81Candidates10 3290 winner else winner) = _
  rw [if_neg (by decide)]
def root81Candidates11 : List Nat := [12547]
set_option maxRecDepth 8192 in
theorem root81Candidates11_checked : candidates edge 4 433 3391 root81Tail12 winner.bound = root81Candidates11 := by
  unfold candidates
  have row : edge 3391 = (SortedNatTree.contains FastRows.row3391) := by
    change Data.row3391.contains = (SortedNatTree.contains FastRows.row3391)
    exact FastRows.row3391_checked.symm
  rw [row]
  decide
theorem root81Step11 : scan edge (search edge 3) 4 [433] 433 root81Tail11 winner =
    scan edge (search edge 3) 4 [433] 433 root81Tail12 winner := by
  rw [root81Tail11, scan, if_neg (by decide), root81Candidates11_checked]
  change scan edge (search edge 3) 4 [433] 433 root81Tail12
    (if 3 ≤ root81Candidates11.length then search edge 3 [433, 3391] root81Candidates11 3824 winner else winner) = _
  rw [if_neg (by decide)]
def root81Candidates12 : List Nat := [7393, 11173]
set_option maxRecDepth 8192 in
theorem root81Candidates12_checked : candidates edge 4 433 3529 root81Tail13 winner.bound = root81Candidates12 := by
  unfold candidates
  have row : edge 3529 = (SortedNatTree.contains FastRows.row3529) := by
    change Data.row3529.contains = (SortedNatTree.contains FastRows.row3529)
    exact FastRows.row3529_checked.symm
  rw [row]
  decide
theorem root81Step12 : scan edge (search edge 3) 4 [433] 433 root81Tail12 winner =
    scan edge (search edge 3) 4 [433] 433 root81Tail13 winner := by
  rw [root81Tail12, scan, if_neg (by decide), root81Candidates12_checked]
  change scan edge (search edge 3) 4 [433] 433 root81Tail13
    (if 3 ≤ root81Candidates12.length then search edge 3 [433, 3529] root81Candidates12 3962 winner else winner) = _
  rw [if_neg (by decide)]
def root81Candidates13 : List Nat := [3697, 9613, 18973, 21739]
set_option maxRecDepth 8192 in
theorem root81Candidates13_checked : candidates edge 4 433 3613 root81Tail14 winner.bound = root81Candidates13 := by
  unfold candidates
  have row : edge 3613 = (SortedNatTree.contains FastRows.row3613) := by
    change Data.row3613.contains = (SortedNatTree.contains FastRows.row3613)
    exact FastRows.row3613_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root81Child13 : search edge 3 [433, 3613] root81Candidates13 4046 winner = winner := by decide
theorem root81Step13 : scan edge (search edge 3) 4 [433] 433 root81Tail13 winner =
    scan edge (search edge 3) 4 [433] 433 root81Tail14 winner := by
  rw [root81Tail13, scan, if_neg (by decide), root81Candidates13_checked]
  change scan edge (search edge 3) 4 [433] 433 root81Tail14
    (if 3 ≤ root81Candidates13.length then search edge 3 [433, 3613] root81Candidates13 4046 winner else winner) = _
  rw [if_pos (by decide), root81Child13]
def root81Candidates14 : List Nat := [19219]
set_option maxRecDepth 8192 in
theorem root81Candidates14_checked : candidates edge 4 433 3697 root81Tail15 winner.bound = root81Candidates14 := by
  unfold candidates
  have row : edge 3697 = (SortedNatTree.contains FastRows.row3697) := by
    change Data.row3697.contains = (SortedNatTree.contains FastRows.row3697)
    exact FastRows.row3697_checked.symm
  rw [row]
  decide
theorem root81Step14 : scan edge (search edge 3) 4 [433] 433 root81Tail14 winner =
    scan edge (search edge 3) 4 [433] 433 root81Tail15 winner := by
  rw [root81Tail14, scan, if_neg (by decide), root81Candidates14_checked]
  change scan edge (search edge 3) 4 [433] 433 root81Tail15
    (if 3 ≤ root81Candidates14.length then search edge 3 [433, 3697] root81Candidates14 4130 winner else winner) = _
  rw [if_neg (by decide)]
def root81Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root81Candidates15_checked : candidates edge 4 433 3739 root81Tail16 winner.bound = root81Candidates15 := by
  unfold candidates
  have row : edge 3739 = (SortedNatTree.contains FastRows.row3739) := by
    change Data.row3739.contains = (SortedNatTree.contains FastRows.row3739)
    exact FastRows.row3739_checked.symm
  rw [row]
  decide
theorem root81Step15 : scan edge (search edge 3) 4 [433] 433 root81Tail15 winner =
    scan edge (search edge 3) 4 [433] 433 root81Tail16 winner := by
  rw [root81Tail15, scan, if_neg (by decide), root81Candidates15_checked]
  change scan edge (search edge 3) 4 [433] 433 root81Tail16
    (if 3 ≤ root81Candidates15.length then search edge 3 [433, 3739] root81Candidates15 4172 winner else winner) = _
  rw [if_neg (by decide)]
def root81Candidates16 : List Nat := [6247, 17497]
set_option maxRecDepth 8192 in
theorem root81Candidates16_checked : candidates edge 4 433 4441 root81Tail17 winner.bound = root81Candidates16 := by
  unfold candidates
  have row : edge 4441 = (SortedNatTree.contains FastRows.row4441) := by
    change Data.row4441.contains = (SortedNatTree.contains FastRows.row4441)
    exact FastRows.row4441_checked.symm
  rw [row]
  decide
theorem root81Step16 : scan edge (search edge 3) 4 [433] 433 root81Tail16 winner =
    scan edge (search edge 3) 4 [433] 433 root81Tail17 winner := by
  rw [root81Tail16, scan, if_neg (by decide), root81Candidates16_checked]
  change scan edge (search edge 3) 4 [433] 433 root81Tail17
    (if 3 ≤ root81Candidates16.length then search edge 3 [433, 4441] root81Candidates16 4874 winner else winner) = _
  rw [if_neg (by decide)]
def root81Candidates17 : List Nat := [7717, 17959]
set_option maxRecDepth 8192 in
theorem root81Candidates17_checked : candidates edge 4 433 4483 root81Tail18 winner.bound = root81Candidates17 := by
  unfold candidates
  have row : edge 4483 = (SortedNatTree.contains FastRows.row4483) := by
    change Data.row4483.contains = (SortedNatTree.contains FastRows.row4483)
    exact FastRows.row4483_checked.symm
  rw [row]
  decide
theorem root81Step17 : scan edge (search edge 3) 4 [433] 433 root81Tail17 winner =
    scan edge (search edge 3) 4 [433] 433 root81Tail18 winner := by
  rw [root81Tail17, scan, if_neg (by decide), root81Candidates17_checked]
  change scan edge (search edge 3) 4 [433] 433 root81Tail18
    (if 3 ≤ root81Candidates17.length then search edge 3 [433, 4483] root81Candidates17 4916 winner else winner) = _
  rw [if_neg (by decide)]
def root81Candidates18 : List Nat := [9787, 11113, 20773]
set_option maxRecDepth 8192 in
theorem root81Candidates18_checked : candidates edge 4 433 4663 root81Tail19 winner.bound = root81Candidates18 := by
  unfold candidates
  have row : edge 4663 = (SortedNatTree.contains FastRows.row4663) := by
    change Data.row4663.contains = (SortedNatTree.contains FastRows.row4663)
    exact FastRows.row4663_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root81Child18 : search edge 3 [433, 4663] root81Candidates18 5096 winner = winner := by decide
theorem root81Step18 : scan edge (search edge 3) 4 [433] 433 root81Tail18 winner =
    scan edge (search edge 3) 4 [433] 433 root81Tail19 winner := by
  rw [root81Tail18, scan, if_neg (by decide), root81Candidates18_checked]
  change scan edge (search edge 3) 4 [433] 433 root81Tail19
    (if 3 ≤ root81Candidates18.length then search edge 3 [433, 4663] root81Candidates18 5096 winner else winner) = _
  rw [if_pos (by decide), root81Child18]
def root81Candidates19 : List Nat := []
set_option maxRecDepth 8192 in
theorem root81Candidates19_checked : candidates edge 4 433 4903 root81Tail20 winner.bound = root81Candidates19 := by
  unfold candidates
  have row : edge 4903 = (SortedNatTree.contains FastRows.row4903) := by
    change Data.row4903.contains = (SortedNatTree.contains FastRows.row4903)
    exact FastRows.row4903_checked.symm
  rw [row]
  decide
theorem root81Step19 : scan edge (search edge 3) 4 [433] 433 root81Tail19 winner =
    scan edge (search edge 3) 4 [433] 433 root81Tail20 winner := by
  rw [root81Tail19, scan, if_neg (by decide), root81Candidates19_checked]
  change scan edge (search edge 3) 4 [433] 433 root81Tail20
    (if 3 ≤ root81Candidates19.length then search edge 3 [433, 4903] root81Candidates19 5336 winner else winner) = _
  rw [if_neg (by decide)]
def root81Candidates20 : List Nat := [19219]
set_option maxRecDepth 8192 in
theorem root81Candidates20_checked : candidates edge 4 433 5281 root81Tail21 winner.bound = root81Candidates20 := by
  unfold candidates
  have row : edge 5281 = (SortedNatTree.contains FastRows.row5281) := by
    change Data.row5281.contains = (SortedNatTree.contains FastRows.row5281)
    exact FastRows.row5281_checked.symm
  rw [row]
  decide
theorem root81Step20 : scan edge (search edge 3) 4 [433] 433 root81Tail20 winner =
    scan edge (search edge 3) 4 [433] 433 root81Tail21 winner := by
  rw [root81Tail20, scan, if_neg (by decide), root81Candidates20_checked]
  change scan edge (search edge 3) 4 [433] 433 root81Tail21
    (if 3 ≤ root81Candidates20.length then search edge 3 [433, 5281] root81Candidates20 5714 winner else winner) = _
  rw [if_neg (by decide)]
def root81Candidates21 : List Nat := [11113]
set_option maxRecDepth 8192 in
theorem root81Candidates21_checked : candidates edge 4 433 5407 root81Tail22 winner.bound = root81Candidates21 := by
  unfold candidates
  have row : edge 5407 = (SortedNatTree.contains FastRows.row5407) := by
    change Data.row5407.contains = (SortedNatTree.contains FastRows.row5407)
    exact FastRows.row5407_checked.symm
  rw [row]
  decide
theorem root81Step21 : scan edge (search edge 3) 4 [433] 433 root81Tail21 winner =
    scan edge (search edge 3) 4 [433] 433 root81Tail22 winner := by
  rw [root81Tail21, scan, if_neg (by decide), root81Candidates21_checked]
  change scan edge (search edge 3) 4 [433] 433 root81Tail22
    (if 3 ≤ root81Candidates21.length then search edge 3 [433, 5407] root81Candidates21 5840 winner else winner) = _
  rw [if_neg (by decide)]
def root81Candidates22 : List Nat := [17959]
set_option maxRecDepth 8192 in
theorem root81Candidates22_checked : candidates edge 4 433 6091 root81Tail23 winner.bound = root81Candidates22 := by
  unfold candidates
  have row : edge 6091 = (SortedNatTree.contains FastRows.row6091) := by
    change Data.row6091.contains = (SortedNatTree.contains FastRows.row6091)
    exact FastRows.row6091_checked.symm
  rw [row]
  decide
theorem root81Step22 : scan edge (search edge 3) 4 [433] 433 root81Tail22 winner =
    scan edge (search edge 3) 4 [433] 433 root81Tail23 winner := by
  rw [root81Tail22, scan, if_neg (by decide), root81Candidates22_checked]
  change scan edge (search edge 3) 4 [433] 433 root81Tail23
    (if 3 ≤ root81Candidates22.length then search edge 3 [433, 6091] root81Candidates22 6524 winner else winner) = _
  rw [if_neg (by decide)]
def root81Candidates23 : List Nat := []
set_option maxRecDepth 8192 in
theorem root81Candidates23_checked : candidates edge 4 433 6247 root81Tail24 winner.bound = root81Candidates23 := by
  unfold candidates
  have row : edge 6247 = (SortedNatTree.contains FastRows.row6247) := by
    change Data.row6247.contains = (SortedNatTree.contains FastRows.row6247)
    exact FastRows.row6247_checked.symm
  rw [row]
  decide
theorem root81Step23 : scan edge (search edge 3) 4 [433] 433 root81Tail23 winner =
    scan edge (search edge 3) 4 [433] 433 root81Tail24 winner := by
  rw [root81Tail23, scan, if_neg (by decide), root81Candidates23_checked]
  change scan edge (search edge 3) 4 [433] 433 root81Tail24
    (if 3 ≤ root81Candidates23.length then search edge 3 [433, 6247] root81Candidates23 6680 winner else winner) = _
  rw [if_neg (by decide)]
def root81Candidates24 : List Nat := [9787, 11503, 14779, 17551]
set_option maxRecDepth 8192 in
theorem root81Candidates24_checked : candidates edge 4 433 6379 root81Tail25 winner.bound = root81Candidates24 := by
  unfold candidates
  have row : edge 6379 = (SortedNatTree.contains FastRows.row6379) := by
    change Data.row6379.contains = (SortedNatTree.contains FastRows.row6379)
    exact FastRows.row6379_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root81Child24 : search edge 3 [433, 6379] root81Candidates24 6812 winner = winner := by decide
theorem root81Step24 : scan edge (search edge 3) 4 [433] 433 root81Tail24 winner =
    scan edge (search edge 3) 4 [433] 433 root81Tail25 winner := by
  rw [root81Tail24, scan, if_neg (by decide), root81Candidates24_checked]
  change scan edge (search edge 3) 4 [433] 433 root81Tail25
    (if 3 ≤ root81Candidates24.length then search edge 3 [433, 6379] root81Candidates24 6812 winner else winner) = _
  rw [if_pos (by decide), root81Child24]
theorem root81Scan25 : scan edge (search edge 3) 4 [433] 433 root81Tail25 winner = winner := by rw [root81Tail25, scan, if_pos (by decide)]
theorem root81Scan24 : scan edge (search edge 3) 4 [433] 433 root81Tail24 winner = winner :=
  (root81Step24).trans root81Scan25
theorem root81Scan23 : scan edge (search edge 3) 4 [433] 433 root81Tail23 winner = winner :=
  (root81Step23).trans root81Scan24
theorem root81Scan22 : scan edge (search edge 3) 4 [433] 433 root81Tail22 winner = winner :=
  (root81Step22).trans root81Scan23
theorem root81Scan21 : scan edge (search edge 3) 4 [433] 433 root81Tail21 winner = winner :=
  (root81Step21).trans root81Scan22
theorem root81Scan20 : scan edge (search edge 3) 4 [433] 433 root81Tail20 winner = winner :=
  (root81Step20).trans root81Scan21
theorem root81Scan19 : scan edge (search edge 3) 4 [433] 433 root81Tail19 winner = winner :=
  (root81Step19).trans root81Scan20
theorem root81Scan18 : scan edge (search edge 3) 4 [433] 433 root81Tail18 winner = winner :=
  (root81Step18).trans root81Scan19
theorem root81Scan17 : scan edge (search edge 3) 4 [433] 433 root81Tail17 winner = winner :=
  (root81Step17).trans root81Scan18
theorem root81Scan16 : scan edge (search edge 3) 4 [433] 433 root81Tail16 winner = winner :=
  (root81Step16).trans root81Scan17
theorem root81Scan15 : scan edge (search edge 3) 4 [433] 433 root81Tail15 winner = winner :=
  (root81Step15).trans root81Scan16
theorem root81Scan14 : scan edge (search edge 3) 4 [433] 433 root81Tail14 winner = winner :=
  (root81Step14).trans root81Scan15
theorem root81Scan13 : scan edge (search edge 3) 4 [433] 433 root81Tail13 winner = winner :=
  (root81Step13).trans root81Scan14
theorem root81Scan12 : scan edge (search edge 3) 4 [433] 433 root81Tail12 winner = winner :=
  (root81Step12).trans root81Scan13
theorem root81Scan11 : scan edge (search edge 3) 4 [433] 433 root81Tail11 winner = winner :=
  (root81Step11).trans root81Scan12
theorem root81Scan10 : scan edge (search edge 3) 4 [433] 433 root81Tail10 winner = winner :=
  (root81Step10).trans root81Scan11
theorem root81Scan9 : scan edge (search edge 3) 4 [433] 433 root81Tail9 winner = winner :=
  (root81Step9).trans root81Scan10
theorem root81Scan8 : scan edge (search edge 3) 4 [433] 433 root81Tail8 winner = winner :=
  (root81Step8).trans root81Scan9
theorem root81Scan7 : scan edge (search edge 3) 4 [433] 433 root81Tail7 winner = winner :=
  (root81Step7).trans root81Scan8
theorem root81Scan6 : scan edge (search edge 3) 4 [433] 433 root81Tail6 winner = winner :=
  (root81Step6).trans root81Scan7
theorem root81Scan5 : scan edge (search edge 3) 4 [433] 433 root81Tail5 winner = winner :=
  (root81Step5).trans root81Scan6
theorem root81Scan4 : scan edge (search edge 3) 4 [433] 433 root81Tail4 winner = winner :=
  (root81Step4).trans root81Scan5
theorem root81Scan3 : scan edge (search edge 3) 4 [433] 433 root81Tail3 winner = winner :=
  (root81Step3).trans root81Scan4
theorem root81Scan2 : scan edge (search edge 3) 4 [433] 433 root81Tail2 winner = winner :=
  (root81Step2).trans root81Scan3
theorem root81Scan1 : scan edge (search edge 3) 4 [433] 433 root81Tail1 winner = winner :=
  (root81Step1).trans root81Scan2
theorem root81Scan0 : scan edge (search edge 3) 4 [433] 433 root81Tail0 winner = winner :=
  (root81Step0).trans root81Scan1
set_option maxRecDepth 4096 in
theorem root81Branch : search edge 4 [433] root81Candidates 433 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [433] 433 root81Tail0 winner = winner
  exact root81Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
