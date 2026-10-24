import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root88Candidates : List Nat := Data.row467
set_option maxRecDepth 8192 in
theorem root88Candidates_checked : candidates edge 5 0 467 tail89 winner.bound = root88Candidates := by
  apply candidates_eq_row edge 5 0 467 tail89 root88Candidates winner.bound tail89_ordered
  · decide
  · decide
  · funext q
    rfl
def root88Tail78 : List Nat := []
def root88Tail77 : List Nat := 25253 :: root88Tail78
def root88Tail76 : List Nat := 25073 :: root88Tail77
def root88Tail75 : List Nat := 23561 :: root88Tail76
def root88Tail74 : List Nat := 23447 :: root88Tail75
def root88Tail73 : List Nat := 23333 :: root88Tail74
def root88Tail72 : List Nat := 21737 :: root88Tail73
def root88Tail71 : List Nat := 21503 :: root88Tail72
def root88Tail70 : List Nat := 21347 :: root88Tail71
def root88Tail69 : List Nat := 21221 :: root88Tail70
def root88Tail68 : List Nat := 19991 :: root88Tail69
def root88Tail67 : List Nat := 19793 :: root88Tail68
def root88Tail66 : List Nat := 19763 :: root88Tail67
def root88Tail65 : List Nat := 19391 :: root88Tail66
def root88Tail64 : List Nat := 19373 :: root88Tail65
def root88Tail63 : List Nat := 18593 :: root88Tail64
def root88Tail62 : List Nat := 18521 :: root88Tail63
def root88Tail61 : List Nat := 17987 :: root88Tail62
def root88Tail60 : List Nat := 17597 :: root88Tail61
def root88Tail59 : List Nat := 16937 :: root88Tail60
def root88Tail58 : List Nat := 16811 :: root88Tail59
def root88Tail57 : List Nat := 16661 :: root88Tail58
def root88Tail56 : List Nat := 16001 :: root88Tail57
def root88Tail55 : List Nat := 15761 :: root88Tail56
def root88Tail54 : List Nat := 15749 :: root88Tail55
def root88Tail53 : List Nat := 15731 :: root88Tail54
def root88Tail52 : List Nat := 14669 :: root88Tail53
def root88Tail51 : List Nat := 14627 :: root88Tail52
def root88Tail50 : List Nat := 14543 :: root88Tail51
def root88Tail49 : List Nat := 13691 :: root88Tail50
def root88Tail48 : List Nat := 13679 :: root88Tail49
def root88Tail47 : List Nat := 13553 :: root88Tail48
def root88Tail46 : List Nat := 13463 :: root88Tail47
def root88Tail45 : List Nat := 13217 :: root88Tail46
def root88Tail44 : List Nat := 12893 :: root88Tail45
def root88Tail43 : List Nat := 12527 :: root88Tail44
def root88Tail42 : List Nat := 11789 :: root88Tail43
def root88Tail41 : List Nat := 11243 :: root88Tail42
def root88Tail40 : List Nat := 11117 :: root88Tail41
def root88Tail39 : List Nat := 10691 :: root88Tail40
def root88Tail38 : List Nat := 10667 :: root88Tail39
def root88Tail37 : List Nat := 10289 :: root88Tail38
def root88Tail36 : List Nat := 10259 :: root88Tail37
def root88Tail35 : List Nat := 10133 :: root88Tail36
def root88Tail34 : List Nat := 9923 :: root88Tail35
def root88Tail33 : List Nat := 9767 :: root88Tail34
def root88Tail32 : List Nat := 9473 :: root88Tail33
def root88Tail31 : List Nat := 9293 :: root88Tail32
def root88Tail30 : List Nat := 9221 :: root88Tail31
def root88Tail29 : List Nat := 8537 :: root88Tail30
def root88Tail28 : List Nat := 8231 :: root88Tail29
def root88Tail27 : List Nat := 6827 :: root88Tail28
def root88Tail26 : List Nat := 6779 :: root88Tail27
def root88Tail25 : List Nat := 6323 :: root88Tail26
def root88Tail24 : List Nat := 6101 :: root88Tail25
def root88Tail23 : List Nat := 6029 :: root88Tail24
def root88Tail22 : List Nat := 5861 :: root88Tail23
def root88Tail21 : List Nat := 5477 :: root88Tail22
def root88Tail20 : List Nat := 4463 :: root88Tail21
def root88Tail19 : List Nat := 4421 :: root88Tail20
def root88Tail18 : List Nat := 4253 :: root88Tail19
def root88Tail17 : List Nat := 4127 :: root88Tail18
def root88Tail16 : List Nat := 4049 :: root88Tail17
def root88Tail15 : List Nat := 3989 :: root88Tail16
def root88Tail14 : List Nat := 3917 :: root88Tail15
def root88Tail13 : List Nat := 2687 :: root88Tail14
def root88Tail12 : List Nat := 2621 :: root88Tail13
def root88Tail11 : List Nat := 2333 :: root88Tail12
def root88Tail10 : List Nat := 2141 :: root88Tail11
def root88Tail9 : List Nat := 2099 :: root88Tail10
def root88Tail8 : List Nat := 2039 :: root88Tail9
def root88Tail7 : List Nat := 1973 :: root88Tail8
def root88Tail6 : List Nat := 1913 :: root88Tail7
def root88Tail5 : List Nat := 1571 :: root88Tail6
def root88Tail4 : List Nat := 1301 :: root88Tail5
def root88Tail3 : List Nat := 941 :: root88Tail4
def root88Tail2 : List Nat := 641 :: root88Tail3
def root88Tail1 : List Nat := 617 :: root88Tail2
def root88Tail0 : List Nat := 587 :: root88Tail1
def root88Candidates0 : List Nat := [617, 1571, 6323, 9473, 16937, 23333]
set_option maxRecDepth 8192 in
theorem root88Candidates0_checked : candidates edge 4 467 587 root88Tail1 winner.bound = root88Candidates0 := by
  unfold candidates
  have row : edge 587 = (SortedNatTree.contains FastRows.row587) := by
    change Data.row587.contains = (SortedNatTree.contains FastRows.row587)
    exact FastRows.row587_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root88Child0 : search edge 3 [467, 587] root88Candidates0 1054 winner = winner := by decide
theorem root88Step0 : scan edge (search edge 3) 4 [467] 467 root88Tail0 winner =
    scan edge (search edge 3) 4 [467] 467 root88Tail1 winner := by
  rw [root88Tail0, scan, if_neg (by decide), root88Candidates0_checked]
  change scan edge (search edge 3) 4 [467] 467 root88Tail1
    (if 3 ≤ root88Candidates0.length then search edge 3 [467, 587] root88Candidates0 1054 winner else winner) = _
  rw [if_pos (by decide), root88Child0]
def root88Candidates1 : List Nat := [1301, 4253, 6323, 9221, 9473, 14669]
set_option maxRecDepth 8192 in
theorem root88Candidates1_checked : candidates edge 4 467 617 root88Tail2 winner.bound = root88Candidates1 := by
  unfold candidates
  have row : edge 617 = (SortedNatTree.contains FastRows.row617) := by
    change Data.row617.contains = (SortedNatTree.contains FastRows.row617)
    exact FastRows.row617_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root88Child1 : search edge 3 [467, 617] root88Candidates1 1084 winner = winner := by decide
theorem root88Step1 : scan edge (search edge 3) 4 [467] 467 root88Tail1 winner =
    scan edge (search edge 3) 4 [467] 467 root88Tail2 winner := by
  rw [root88Tail1, scan, if_neg (by decide), root88Candidates1_checked]
  change scan edge (search edge 3) 4 [467] 467 root88Tail2
    (if 3 ≤ root88Candidates1.length then search edge 3 [467, 617] root88Candidates1 1084 winner else winner) = _
  rw [if_pos (by decide), root88Child1]
def root88Candidates2 : List Nat := [10667]
set_option maxRecDepth 8192 in
theorem root88Candidates2_checked : candidates edge 4 467 641 root88Tail3 winner.bound = root88Candidates2 := by
  unfold candidates
  have row : edge 641 = (SortedNatTree.contains FastRows.row641) := by
    change Data.row641.contains = (SortedNatTree.contains FastRows.row641)
    exact FastRows.row641_checked.symm
  rw [row]
  decide
theorem root88Step2 : scan edge (search edge 3) 4 [467] 467 root88Tail2 winner =
    scan edge (search edge 3) 4 [467] 467 root88Tail3 winner := by
  rw [root88Tail2, scan, if_neg (by decide), root88Candidates2_checked]
  change scan edge (search edge 3) 4 [467] 467 root88Tail3
    (if 3 ≤ root88Candidates2.length then search edge 3 [467, 641] root88Candidates2 1108 winner else winner) = _
  rw [if_neg (by decide)]
def root88Candidates3 : List Nat := [2099, 9923, 18593, 19793, 23333]
set_option maxRecDepth 8192 in
theorem root88Candidates3_checked : candidates edge 4 467 941 root88Tail4 winner.bound = root88Candidates3 := by
  unfold candidates
  have row : edge 941 = (SortedNatTree.contains FastRows.row941) := by
    change Data.row941.contains = (SortedNatTree.contains FastRows.row941)
    exact FastRows.row941_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root88Child3 : search edge 3 [467, 941] root88Candidates3 1408 winner = winner := by decide
theorem root88Step3 : scan edge (search edge 3) 4 [467] 467 root88Tail3 winner =
    scan edge (search edge 3) 4 [467] 467 root88Tail4 winner := by
  rw [root88Tail3, scan, if_neg (by decide), root88Candidates3_checked]
  change scan edge (search edge 3) 4 [467] 467 root88Tail4
    (if 3 ≤ root88Candidates3.length then search edge 3 [467, 941] root88Candidates3 1408 winner else winner) = _
  rw [if_pos (by decide), root88Child3]
def root88Candidates4 : List Nat := [2141, 11243, 14669, 19373]
set_option maxRecDepth 8192 in
theorem root88Candidates4_checked : candidates edge 4 467 1301 root88Tail5 winner.bound = root88Candidates4 := by
  unfold candidates
  have row : edge 1301 = (SortedNatTree.contains FastRows.row1301) := by
    change Data.row1301.contains = (SortedNatTree.contains FastRows.row1301)
    exact FastRows.row1301_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root88Child4 : search edge 3 [467, 1301] root88Candidates4 1768 winner = winner := by decide
theorem root88Step4 : scan edge (search edge 3) 4 [467] 467 root88Tail4 winner =
    scan edge (search edge 3) 4 [467] 467 root88Tail5 winner := by
  rw [root88Tail4, scan, if_neg (by decide), root88Candidates4_checked]
  change scan edge (search edge 3) 4 [467] 467 root88Tail5
    (if 3 ≤ root88Candidates4.length then search edge 3 [467, 1301] root88Candidates4 1768 winner else winner) = _
  rw [if_pos (by decide), root88Child4]
def root88Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root88Candidates5_checked : candidates edge 4 467 1571 root88Tail6 winner.bound = root88Candidates5 := by
  unfold candidates
  have row : edge 1571 = (SortedNatTree.contains FastRows.row1571) := by
    change Data.row1571.contains = (SortedNatTree.contains FastRows.row1571)
    exact FastRows.row1571_checked.symm
  rw [row]
  decide
theorem root88Step5 : scan edge (search edge 3) 4 [467] 467 root88Tail5 winner =
    scan edge (search edge 3) 4 [467] 467 root88Tail6 winner := by
  rw [root88Tail5, scan, if_neg (by decide), root88Candidates5_checked]
  change scan edge (search edge 3) 4 [467] 467 root88Tail6
    (if 3 ≤ root88Candidates5.length then search edge 3 [467, 1571] root88Candidates5 2038 winner else winner) = _
  rw [if_neg (by decide)]
def root88Candidates6 : List Nat := [5861, 11789, 18593, 21503]
set_option maxRecDepth 8192 in
theorem root88Candidates6_checked : candidates edge 4 467 1913 root88Tail7 winner.bound = root88Candidates6 := by
  unfold candidates
  have row : edge 1913 = (SortedNatTree.contains FastRows.row1913) := by
    change Data.row1913.contains = (SortedNatTree.contains FastRows.row1913)
    exact FastRows.row1913_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root88Child6 : search edge 3 [467, 1913] root88Candidates6 2380 winner = winner := by decide
theorem root88Step6 : scan edge (search edge 3) 4 [467] 467 root88Tail6 winner =
    scan edge (search edge 3) 4 [467] 467 root88Tail7 winner := by
  rw [root88Tail6, scan, if_neg (by decide), root88Candidates6_checked]
  change scan edge (search edge 3) 4 [467] 467 root88Tail7
    (if 3 ≤ root88Candidates6.length then search edge 3 [467, 1913] root88Candidates6 2380 winner else winner) = _
  rw [if_pos (by decide), root88Child6]
def root88Candidates7 : List Nat := [17987]
set_option maxRecDepth 8192 in
theorem root88Candidates7_checked : candidates edge 4 467 1973 root88Tail8 winner.bound = root88Candidates7 := by
  unfold candidates
  have row : edge 1973 = (SortedNatTree.contains FastRows.row1973) := by
    change Data.row1973.contains = (SortedNatTree.contains FastRows.row1973)
    exact FastRows.row1973_checked.symm
  rw [row]
  decide
theorem root88Step7 : scan edge (search edge 3) 4 [467] 467 root88Tail7 winner =
    scan edge (search edge 3) 4 [467] 467 root88Tail8 winner := by
  rw [root88Tail7, scan, if_neg (by decide), root88Candidates7_checked]
  change scan edge (search edge 3) 4 [467] 467 root88Tail8
    (if 3 ≤ root88Candidates7.length then search edge 3 [467, 1973] root88Candidates7 2440 winner else winner) = _
  rw [if_neg (by decide)]
def root88Candidates8 : List Nat := [3917, 15749, 18593]
set_option maxRecDepth 8192 in
theorem root88Candidates8_checked : candidates edge 4 467 2039 root88Tail9 winner.bound = root88Candidates8 := by
  unfold candidates
  have row : edge 2039 = (SortedNatTree.contains FastRows.row2039) := by
    change Data.row2039.contains = (SortedNatTree.contains FastRows.row2039)
    exact FastRows.row2039_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root88Child8 : search edge 3 [467, 2039] root88Candidates8 2506 winner = winner := by decide
theorem root88Step8 : scan edge (search edge 3) 4 [467] 467 root88Tail8 winner =
    scan edge (search edge 3) 4 [467] 467 root88Tail9 winner := by
  rw [root88Tail8, scan, if_neg (by decide), root88Candidates8_checked]
  change scan edge (search edge 3) 4 [467] 467 root88Tail9
    (if 3 ≤ root88Candidates8.length then search edge 3 [467, 2039] root88Candidates8 2506 winner else winner) = _
  rw [if_pos (by decide), root88Child8]
def root88Candidates9 : List Nat := [17597, 19793]
set_option maxRecDepth 8192 in
theorem root88Candidates9_checked : candidates edge 4 467 2099 root88Tail10 winner.bound = root88Candidates9 := by
  unfold candidates
  have row : edge 2099 = (SortedNatTree.contains FastRows.row2099) := by
    change Data.row2099.contains = (SortedNatTree.contains FastRows.row2099)
    exact FastRows.row2099_checked.symm
  rw [row]
  decide
theorem root88Step9 : scan edge (search edge 3) 4 [467] 467 root88Tail9 winner =
    scan edge (search edge 3) 4 [467] 467 root88Tail10 winner := by
  rw [root88Tail9, scan, if_neg (by decide), root88Candidates9_checked]
  change scan edge (search edge 3) 4 [467] 467 root88Tail10
    (if 3 ≤ root88Candidates9.length then search edge 3 [467, 2099] root88Candidates9 2566 winner else winner) = _
  rw [if_neg (by decide)]
def root88Candidates10 : List Nat := [2621, 4049]
set_option maxRecDepth 8192 in
theorem root88Candidates10_checked : candidates edge 4 467 2141 root88Tail11 winner.bound = root88Candidates10 := by
  unfold candidates
  have row : edge 2141 = (SortedNatTree.contains FastRows.row2141) := by
    change Data.row2141.contains = (SortedNatTree.contains FastRows.row2141)
    exact FastRows.row2141_checked.symm
  rw [row]
  decide
theorem root88Step10 : scan edge (search edge 3) 4 [467] 467 root88Tail10 winner =
    scan edge (search edge 3) 4 [467] 467 root88Tail11 winner := by
  rw [root88Tail10, scan, if_neg (by decide), root88Candidates10_checked]
  change scan edge (search edge 3) 4 [467] 467 root88Tail11
    (if 3 ≤ root88Candidates10.length then search edge 3 [467, 2141] root88Candidates10 2608 winner else winner) = _
  rw [if_neg (by decide)]
def root88Candidates11 : List Nat := [9473]
set_option maxRecDepth 8192 in
theorem root88Candidates11_checked : candidates edge 4 467 2333 root88Tail12 winner.bound = root88Candidates11 := by
  unfold candidates
  have row : edge 2333 = (SortedNatTree.contains FastRows.row2333) := by
    change Data.row2333.contains = (SortedNatTree.contains FastRows.row2333)
    exact FastRows.row2333_checked.symm
  rw [row]
  decide
theorem root88Step11 : scan edge (search edge 3) 4 [467] 467 root88Tail11 winner =
    scan edge (search edge 3) 4 [467] 467 root88Tail12 winner := by
  rw [root88Tail11, scan, if_neg (by decide), root88Candidates11_checked]
  change scan edge (search edge 3) 4 [467] 467 root88Tail12
    (if 3 ≤ root88Candidates11.length then search edge 3 [467, 2333] root88Candidates11 2800 winner else winner) = _
  rw [if_neg (by decide)]
def root88Candidates12 : List Nat := [5477, 6101, 17987, 19373]
set_option maxRecDepth 8192 in
theorem root88Candidates12_checked : candidates edge 4 467 2621 root88Tail13 winner.bound = root88Candidates12 := by
  unfold candidates
  have row : edge 2621 = (SortedNatTree.contains FastRows.row2621) := by
    change Data.row2621.contains = (SortedNatTree.contains FastRows.row2621)
    exact FastRows.row2621_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root88Child12 : search edge 3 [467, 2621] root88Candidates12 3088 winner = winner := by decide
theorem root88Step12 : scan edge (search edge 3) 4 [467] 467 root88Tail12 winner =
    scan edge (search edge 3) 4 [467] 467 root88Tail13 winner := by
  rw [root88Tail12, scan, if_neg (by decide), root88Candidates12_checked]
  change scan edge (search edge 3) 4 [467] 467 root88Tail13
    (if 3 ≤ root88Candidates12.length then search edge 3 [467, 2621] root88Candidates12 3088 winner else winner) = _
  rw [if_pos (by decide), root88Child12]
def root88Candidates13 : List Nat := [8231]
set_option maxRecDepth 8192 in
theorem root88Candidates13_checked : candidates edge 4 467 2687 root88Tail14 winner.bound = root88Candidates13 := by
  unfold candidates
  have row : edge 2687 = (SortedNatTree.contains FastRows.row2687) := by
    change Data.row2687.contains = (SortedNatTree.contains FastRows.row2687)
    exact FastRows.row2687_checked.symm
  rw [row]
  decide
theorem root88Step13 : scan edge (search edge 3) 4 [467] 467 root88Tail13 winner =
    scan edge (search edge 3) 4 [467] 467 root88Tail14 winner := by
  rw [root88Tail13, scan, if_neg (by decide), root88Candidates13_checked]
  change scan edge (search edge 3) 4 [467] 467 root88Tail14
    (if 3 ≤ root88Candidates13.length then search edge 3 [467, 2687] root88Candidates13 3154 winner else winner) = _
  rw [if_neg (by decide)]
def root88Candidates14 : List Nat := [6323, 6779]
set_option maxRecDepth 8192 in
theorem root88Candidates14_checked : candidates edge 4 467 3917 root88Tail15 winner.bound = root88Candidates14 := by
  unfold candidates
  have row : edge 3917 = (SortedNatTree.contains FastRows.row3917) := by
    change Data.row3917.contains = (SortedNatTree.contains FastRows.row3917)
    exact FastRows.row3917_checked.symm
  rw [row]
  decide
theorem root88Step14 : scan edge (search edge 3) 4 [467] 467 root88Tail14 winner =
    scan edge (search edge 3) 4 [467] 467 root88Tail15 winner := by
  rw [root88Tail14, scan, if_neg (by decide), root88Candidates14_checked]
  change scan edge (search edge 3) 4 [467] 467 root88Tail15
    (if 3 ≤ root88Candidates14.length then search edge 3 [467, 3917] root88Candidates14 4384 winner else winner) = _
  rw [if_neg (by decide)]
def root88Candidates15 : List Nat := [10133]
set_option maxRecDepth 8192 in
theorem root88Candidates15_checked : candidates edge 4 467 3989 root88Tail16 winner.bound = root88Candidates15 := by
  unfold candidates
  have row : edge 3989 = (SortedNatTree.contains FastRows.row3989) := by
    change Data.row3989.contains = (SortedNatTree.contains FastRows.row3989)
    exact FastRows.row3989_checked.symm
  rw [row]
  decide
theorem root88Step15 : scan edge (search edge 3) 4 [467] 467 root88Tail15 winner =
    scan edge (search edge 3) 4 [467] 467 root88Tail16 winner := by
  rw [root88Tail15, scan, if_neg (by decide), root88Candidates15_checked]
  change scan edge (search edge 3) 4 [467] 467 root88Tail16
    (if 3 ≤ root88Candidates15.length then search edge 3 [467, 3989] root88Candidates15 4456 winner else winner) = _
  rw [if_neg (by decide)]
def root88Candidates16 : List Nat := [19763]
set_option maxRecDepth 8192 in
theorem root88Candidates16_checked : candidates edge 4 467 4049 root88Tail17 winner.bound = root88Candidates16 := by
  unfold candidates
  have row : edge 4049 = (SortedNatTree.contains FastRows.row4049) := by
    change Data.row4049.contains = (SortedNatTree.contains FastRows.row4049)
    exact FastRows.row4049_checked.symm
  rw [row]
  decide
theorem root88Step16 : scan edge (search edge 3) 4 [467] 467 root88Tail16 winner =
    scan edge (search edge 3) 4 [467] 467 root88Tail17 winner := by
  rw [root88Tail16, scan, if_neg (by decide), root88Candidates16_checked]
  change scan edge (search edge 3) 4 [467] 467 root88Tail17
    (if 3 ≤ root88Candidates16.length then search edge 3 [467, 4049] root88Candidates16 4516 winner else winner) = _
  rw [if_neg (by decide)]
def root88Candidates17 : List Nat := [9767]
set_option maxRecDepth 8192 in
theorem root88Candidates17_checked : candidates edge 4 467 4127 root88Tail18 winner.bound = root88Candidates17 := by
  unfold candidates
  have row : edge 4127 = (SortedNatTree.contains FastRows.row4127) := by
    change Data.row4127.contains = (SortedNatTree.contains FastRows.row4127)
    exact FastRows.row4127_checked.symm
  rw [row]
  decide
theorem root88Step17 : scan edge (search edge 3) 4 [467] 467 root88Tail17 winner =
    scan edge (search edge 3) 4 [467] 467 root88Tail18 winner := by
  rw [root88Tail17, scan, if_neg (by decide), root88Candidates17_checked]
  change scan edge (search edge 3) 4 [467] 467 root88Tail18
    (if 3 ≤ root88Candidates17.length then search edge 3 [467, 4127] root88Candidates17 4594 winner else winner) = _
  rw [if_neg (by decide)]
def root88Candidates18 : List Nat := [4463, 9293, 16937]
set_option maxRecDepth 8192 in
theorem root88Candidates18_checked : candidates edge 4 467 4253 root88Tail19 winner.bound = root88Candidates18 := by
  unfold candidates
  have row : edge 4253 = (SortedNatTree.contains FastRows.row4253) := by
    change Data.row4253.contains = (SortedNatTree.contains FastRows.row4253)
    exact FastRows.row4253_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root88Child18 : search edge 3 [467, 4253] root88Candidates18 4720 winner = winner := by decide
theorem root88Step18 : scan edge (search edge 3) 4 [467] 467 root88Tail18 winner =
    scan edge (search edge 3) 4 [467] 467 root88Tail19 winner := by
  rw [root88Tail18, scan, if_neg (by decide), root88Candidates18_checked]
  change scan edge (search edge 3) 4 [467] 467 root88Tail19
    (if 3 ≤ root88Candidates18.length then search edge 3 [467, 4253] root88Candidates18 4720 winner else winner) = _
  rw [if_pos (by decide), root88Child18]
def root88Candidates19 : List Nat := [19763]
set_option maxRecDepth 8192 in
theorem root88Candidates19_checked : candidates edge 4 467 4421 root88Tail20 winner.bound = root88Candidates19 := by
  unfold candidates
  have row : edge 4421 = (SortedNatTree.contains FastRows.row4421) := by
    change Data.row4421.contains = (SortedNatTree.contains FastRows.row4421)
    exact FastRows.row4421_checked.symm
  rw [row]
  decide
theorem root88Step19 : scan edge (search edge 3) 4 [467] 467 root88Tail19 winner =
    scan edge (search edge 3) 4 [467] 467 root88Tail20 winner := by
  rw [root88Tail19, scan, if_neg (by decide), root88Candidates19_checked]
  change scan edge (search edge 3) 4 [467] 467 root88Tail20
    (if 3 ≤ root88Candidates19.length then search edge 3 [467, 4421] root88Candidates19 4888 winner else winner) = _
  rw [if_neg (by decide)]
def root88Candidates20 : List Nat := [5861]
set_option maxRecDepth 8192 in
theorem root88Candidates20_checked : candidates edge 4 467 4463 root88Tail21 winner.bound = root88Candidates20 := by
  unfold candidates
  have row : edge 4463 = (SortedNatTree.contains FastRows.row4463) := by
    change Data.row4463.contains = (SortedNatTree.contains FastRows.row4463)
    exact FastRows.row4463_checked.symm
  rw [row]
  decide
theorem root88Step20 : scan edge (search edge 3) 4 [467] 467 root88Tail20 winner =
    scan edge (search edge 3) 4 [467] 467 root88Tail21 winner := by
  rw [root88Tail20, scan, if_neg (by decide), root88Candidates20_checked]
  change scan edge (search edge 3) 4 [467] 467 root88Tail21
    (if 3 ≤ root88Candidates20.length then search edge 3 [467, 4463] root88Candidates20 4930 winner else winner) = _
  rw [if_neg (by decide)]
def root88Candidates21 : List Nat := []
set_option maxRecDepth 8192 in
theorem root88Candidates21_checked : candidates edge 4 467 5477 root88Tail22 winner.bound = root88Candidates21 := by
  unfold candidates
  have row : edge 5477 = (SortedNatTree.contains FastRows.row5477) := by
    change Data.row5477.contains = (SortedNatTree.contains FastRows.row5477)
    exact FastRows.row5477_checked.symm
  rw [row]
  decide
theorem root88Step21 : scan edge (search edge 3) 4 [467] 467 root88Tail21 winner =
    scan edge (search edge 3) 4 [467] 467 root88Tail22 winner := by
  rw [root88Tail21, scan, if_neg (by decide), root88Candidates21_checked]
  change scan edge (search edge 3) 4 [467] 467 root88Tail22
    (if 3 ≤ root88Candidates21.length then search edge 3 [467, 5477] root88Candidates21 5944 winner else winner) = _
  rw [if_neg (by decide)]
def root88Candidates22 : List Nat := [6827, 9221, 13679]
set_option maxRecDepth 8192 in
theorem root88Candidates22_checked : candidates edge 4 467 5861 root88Tail23 winner.bound = root88Candidates22 := by
  unfold candidates
  have row : edge 5861 = (SortedNatTree.contains FastRows.row5861) := by
    change Data.row5861.contains = (SortedNatTree.contains FastRows.row5861)
    exact FastRows.row5861_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root88Child22 : search edge 3 [467, 5861] root88Candidates22 6328 winner = winner := by decide
theorem root88Step22 : scan edge (search edge 3) 4 [467] 467 root88Tail22 winner =
    scan edge (search edge 3) 4 [467] 467 root88Tail23 winner := by
  rw [root88Tail22, scan, if_neg (by decide), root88Candidates22_checked]
  change scan edge (search edge 3) 4 [467] 467 root88Tail23
    (if 3 ≤ root88Candidates22.length then search edge 3 [467, 5861] root88Candidates22 6328 winner else winner) = _
  rw [if_pos (by decide), root88Child22]
def root88Candidates23 : List Nat := [9923, 16661]
set_option maxRecDepth 8192 in
theorem root88Candidates23_checked : candidates edge 4 467 6029 root88Tail24 winner.bound = root88Candidates23 := by
  unfold candidates
  have row : edge 6029 = (SortedNatTree.contains FastRows.row6029) := by
    change Data.row6029.contains = (SortedNatTree.contains FastRows.row6029)
    exact FastRows.row6029_checked.symm
  rw [row]
  decide
theorem root88Step23 : scan edge (search edge 3) 4 [467] 467 root88Tail23 winner =
    scan edge (search edge 3) 4 [467] 467 root88Tail24 winner := by
  rw [root88Tail23, scan, if_neg (by decide), root88Candidates23_checked]
  change scan edge (search edge 3) 4 [467] 467 root88Tail24
    (if 3 ≤ root88Candidates23.length then search edge 3 [467, 6029] root88Candidates23 6496 winner else winner) = _
  rw [if_neg (by decide)]
def root88Candidates24 : List Nat := [9293]
set_option maxRecDepth 8192 in
theorem root88Candidates24_checked : candidates edge 4 467 6101 root88Tail25 winner.bound = root88Candidates24 := by
  unfold candidates
  have row : edge 6101 = (SortedNatTree.contains FastRows.row6101) := by
    change Data.row6101.contains = (SortedNatTree.contains FastRows.row6101)
    exact FastRows.row6101_checked.symm
  rw [row]
  decide
theorem root88Step24 : scan edge (search edge 3) 4 [467] 467 root88Tail24 winner =
    scan edge (search edge 3) 4 [467] 467 root88Tail25 winner := by
  rw [root88Tail24, scan, if_neg (by decide), root88Candidates24_checked]
  change scan edge (search edge 3) 4 [467] 467 root88Tail25
    (if 3 ≤ root88Candidates24.length then search edge 3 [467, 6101] root88Candidates24 6568 winner else winner) = _
  rw [if_neg (by decide)]
def root88Candidates25 : List Nat := []
set_option maxRecDepth 8192 in
theorem root88Candidates25_checked : candidates edge 4 467 6323 root88Tail26 winner.bound = root88Candidates25 := by
  unfold candidates
  have row : edge 6323 = (SortedNatTree.contains FastRows.row6323) := by
    change Data.row6323.contains = (SortedNatTree.contains FastRows.row6323)
    exact FastRows.row6323_checked.symm
  rw [row]
  decide
theorem root88Step25 : scan edge (search edge 3) 4 [467] 467 root88Tail25 winner =
    scan edge (search edge 3) 4 [467] 467 root88Tail26 winner := by
  rw [root88Tail25, scan, if_neg (by decide), root88Candidates25_checked]
  change scan edge (search edge 3) 4 [467] 467 root88Tail26
    (if 3 ≤ root88Candidates25.length then search edge 3 [467, 6323] root88Candidates25 6790 winner else winner) = _
  rw [if_neg (by decide)]
theorem root88Scan26 : scan edge (search edge 3) 4 [467] 467 root88Tail26 winner = winner := by rw [root88Tail26, scan, if_pos (by decide)]
theorem root88Scan25 : scan edge (search edge 3) 4 [467] 467 root88Tail25 winner = winner :=
  (root88Step25).trans root88Scan26
theorem root88Scan24 : scan edge (search edge 3) 4 [467] 467 root88Tail24 winner = winner :=
  (root88Step24).trans root88Scan25
theorem root88Scan23 : scan edge (search edge 3) 4 [467] 467 root88Tail23 winner = winner :=
  (root88Step23).trans root88Scan24
theorem root88Scan22 : scan edge (search edge 3) 4 [467] 467 root88Tail22 winner = winner :=
  (root88Step22).trans root88Scan23
theorem root88Scan21 : scan edge (search edge 3) 4 [467] 467 root88Tail21 winner = winner :=
  (root88Step21).trans root88Scan22
theorem root88Scan20 : scan edge (search edge 3) 4 [467] 467 root88Tail20 winner = winner :=
  (root88Step20).trans root88Scan21
theorem root88Scan19 : scan edge (search edge 3) 4 [467] 467 root88Tail19 winner = winner :=
  (root88Step19).trans root88Scan20
theorem root88Scan18 : scan edge (search edge 3) 4 [467] 467 root88Tail18 winner = winner :=
  (root88Step18).trans root88Scan19
theorem root88Scan17 : scan edge (search edge 3) 4 [467] 467 root88Tail17 winner = winner :=
  (root88Step17).trans root88Scan18
theorem root88Scan16 : scan edge (search edge 3) 4 [467] 467 root88Tail16 winner = winner :=
  (root88Step16).trans root88Scan17
theorem root88Scan15 : scan edge (search edge 3) 4 [467] 467 root88Tail15 winner = winner :=
  (root88Step15).trans root88Scan16
theorem root88Scan14 : scan edge (search edge 3) 4 [467] 467 root88Tail14 winner = winner :=
  (root88Step14).trans root88Scan15
theorem root88Scan13 : scan edge (search edge 3) 4 [467] 467 root88Tail13 winner = winner :=
  (root88Step13).trans root88Scan14
theorem root88Scan12 : scan edge (search edge 3) 4 [467] 467 root88Tail12 winner = winner :=
  (root88Step12).trans root88Scan13
theorem root88Scan11 : scan edge (search edge 3) 4 [467] 467 root88Tail11 winner = winner :=
  (root88Step11).trans root88Scan12
theorem root88Scan10 : scan edge (search edge 3) 4 [467] 467 root88Tail10 winner = winner :=
  (root88Step10).trans root88Scan11
theorem root88Scan9 : scan edge (search edge 3) 4 [467] 467 root88Tail9 winner = winner :=
  (root88Step9).trans root88Scan10
theorem root88Scan8 : scan edge (search edge 3) 4 [467] 467 root88Tail8 winner = winner :=
  (root88Step8).trans root88Scan9
theorem root88Scan7 : scan edge (search edge 3) 4 [467] 467 root88Tail7 winner = winner :=
  (root88Step7).trans root88Scan8
theorem root88Scan6 : scan edge (search edge 3) 4 [467] 467 root88Tail6 winner = winner :=
  (root88Step6).trans root88Scan7
theorem root88Scan5 : scan edge (search edge 3) 4 [467] 467 root88Tail5 winner = winner :=
  (root88Step5).trans root88Scan6
theorem root88Scan4 : scan edge (search edge 3) 4 [467] 467 root88Tail4 winner = winner :=
  (root88Step4).trans root88Scan5
theorem root88Scan3 : scan edge (search edge 3) 4 [467] 467 root88Tail3 winner = winner :=
  (root88Step3).trans root88Scan4
theorem root88Scan2 : scan edge (search edge 3) 4 [467] 467 root88Tail2 winner = winner :=
  (root88Step2).trans root88Scan3
theorem root88Scan1 : scan edge (search edge 3) 4 [467] 467 root88Tail1 winner = winner :=
  (root88Step1).trans root88Scan2
theorem root88Scan0 : scan edge (search edge 3) 4 [467] 467 root88Tail0 winner = winner :=
  (root88Step0).trans root88Scan1
set_option maxRecDepth 4096 in
theorem root88Branch : search edge 4 [467] root88Candidates 467 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [467] 467 root88Tail0 winner = winner
  exact root88Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
