import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root53Candidates : List Nat := Data.row263
set_option maxRecDepth 8192 in
theorem root53Candidates_checked : candidates edge 5 0 263 tail54 winner.bound = root53Candidates := by
  apply candidates_eq_row edge 5 0 263 tail54 root53Candidates winner.bound tail54_ordered
  · decide
  · decide
  · funext q
    rfl
def root53Tail72 : List Nat := []
def root53Tail71 : List Nat := 25679 :: root53Tail72
def root53Tail70 : List Nat := 23537 :: root53Tail71
def root53Tail69 : List Nat := 23459 :: root53Tail70
def root53Tail68 : List Nat := 23357 :: root53Tail69
def root53Tail67 : List Nat := 23333 :: root53Tail68
def root53Tail66 : List Nat := 23081 :: root53Tail67
def root53Tail65 : List Nat := 23057 :: root53Tail66
def root53Tail64 : List Nat := 22739 :: root53Tail65
def root53Tail63 : List Nat := 21143 :: root53Tail64
def root53Tail62 : List Nat := 21101 :: root53Tail63
def root53Tail61 : List Nat := 21059 :: root53Tail62
def root53Tail60 : List Nat := 20939 :: root53Tail61
def root53Tail59 : List Nat := 20747 :: root53Tail60
def root53Tail58 : List Nat := 20249 :: root53Tail59
def root53Tail57 : List Nat := 20051 :: root53Tail58
def root53Tail56 : List Nat := 19973 :: root53Tail57
def root53Tail55 : List Nat := 19379 :: root53Tail56
def root53Tail54 : List Nat := 19259 :: root53Tail55
def root53Tail53 : List Nat := 18917 :: root53Tail54
def root53Tail52 : List Nat := 18353 :: root53Tail53
def root53Tail51 : List Nat := 17807 :: root53Tail52
def root53Tail50 : List Nat := 16943 :: root53Tail51
def root53Tail49 : List Nat := 15581 :: root53Tail50
def root53Tail48 : List Nat := 15413 :: root53Tail49
def root53Tail47 : List Nat := 14633 :: root53Tail48
def root53Tail46 : List Nat := 14489 :: root53Tail47
def root53Tail45 : List Nat := 14159 :: root53Tail46
def root53Tail44 : List Nat := 14153 :: root53Tail45
def root53Tail43 : List Nat := 13187 :: root53Tail44
def root53Tail42 : List Nat := 12911 :: root53Tail43
def root53Tail41 : List Nat := 12641 :: root53Tail42
def root53Tail40 : List Nat := 12323 :: root53Tail41
def root53Tail39 : List Nat := 11783 :: root53Tail40
def root53Tail38 : List Nat := 11321 :: root53Tail39
def root53Tail37 : List Nat := 11279 :: root53Tail38
def root53Tail36 : List Nat := 10457 :: root53Tail37
def root53Tail35 : List Nat := 10331 :: root53Tail36
def root53Tail34 : List Nat := 10181 :: root53Tail35
def root53Tail33 : List Nat := 9521 :: root53Tail34
def root53Tail32 : List Nat := 9323 :: root53Tail33
def root53Tail31 : List Nat := 8693 :: root53Tail32
def root53Tail30 : List Nat := 8087 :: root53Tail31
def root53Tail29 : List Nat := 7937 :: root53Tail30
def root53Tail28 : List Nat := 7451 :: root53Tail29
def root53Tail27 : List Nat := 7253 :: root53Tail28
def root53Tail26 : List Nat := 7043 :: root53Tail27
def root53Tail25 : List Nat := 7013 :: root53Tail26
def root53Tail24 : List Nat := 6701 :: root53Tail25
def root53Tail23 : List Nat := 6197 :: root53Tail24
def root53Tail22 : List Nat := 5987 :: root53Tail23
def root53Tail21 : List Nat := 5693 :: root53Tail22
def root53Tail20 : List Nat := 5189 :: root53Tail21
def root53Tail19 : List Nat := 4523 :: root53Tail20
def root53Tail18 : List Nat := 3923 :: root53Tail19
def root53Tail17 : List Nat := 3557 :: root53Tail18
def root53Tail16 : List Nat := 3467 :: root53Tail17
def root53Tail15 : List Nat := 3359 :: root53Tail16
def root53Tail14 : List Nat := 2999 :: root53Tail15
def root53Tail13 : List Nat := 2789 :: root53Tail14
def root53Tail12 : List Nat := 2633 :: root53Tail13
def root53Tail11 : List Nat := 2477 :: root53Tail12
def root53Tail10 : List Nat := 2339 :: root53Tail11
def root53Tail9 : List Nat := 2129 :: root53Tail10
def root53Tail8 : List Nat := 2099 :: root53Tail9
def root53Tail7 : List Nat := 1091 :: root53Tail8
def root53Tail6 : List Nat := 1049 :: root53Tail7
def root53Tail5 : List Nat := 941 :: root53Tail6
def root53Tail4 : List Nat := 821 :: root53Tail5
def root53Tail3 : List Nat := 761 :: root53Tail4
def root53Tail2 : List Nat := 647 :: root53Tail3
def root53Tail1 : List Nat := 443 :: root53Tail2
def root53Tail0 : List Nat := 293 :: root53Tail1
def root53Candidates0 : List Nat := [5189, 19973, 21101]
set_option maxRecDepth 8192 in
theorem root53Candidates0_checked : candidates edge 4 263 293 root53Tail1 winner.bound = root53Candidates0 := by
  unfold candidates
  have row : edge 293 = (SortedNatTree.contains FastRows.row293) := by
    change Data.row293.contains = (SortedNatTree.contains FastRows.row293)
    exact FastRows.row293_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root53Child0 : search edge 3 [263, 293] root53Candidates0 556 winner = winner := by decide
theorem root53Step0 : scan edge (search edge 3) 4 [263] 263 root53Tail0 winner =
    scan edge (search edge 3) 4 [263] 263 root53Tail1 winner := by
  rw [root53Tail0, scan, if_neg (by decide), root53Candidates0_checked]
  change scan edge (search edge 3) 4 [263] 263 root53Tail1
    (if 3 ≤ root53Candidates0.length then search edge 3 [263, 293] root53Candidates0 556 winner else winner) = _
  rw [if_pos (by decide), root53Child0]
def root53Candidates1 : List Nat := [761, 10181, 10457, 20249, 23537]
set_option maxRecDepth 8192 in
theorem root53Candidates1_checked : candidates edge 4 263 443 root53Tail2 winner.bound = root53Candidates1 := by
  unfold candidates
  have row : edge 443 = (SortedNatTree.contains FastRows.row443) := by
    change Data.row443.contains = (SortedNatTree.contains FastRows.row443)
    exact FastRows.row443_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root53Child1 : search edge 3 [263, 443] root53Candidates1 706 winner = winner := by decide
theorem root53Step1 : scan edge (search edge 3) 4 [263] 263 root53Tail1 winner =
    scan edge (search edge 3) 4 [263] 263 root53Tail2 winner := by
  rw [root53Tail1, scan, if_neg (by decide), root53Candidates1_checked]
  change scan edge (search edge 3) 4 [263] 263 root53Tail2
    (if 3 ≤ root53Candidates1.length then search edge 3 [263, 443] root53Candidates1 706 winner else winner) = _
  rw [if_pos (by decide), root53Child1]
def root53Candidates2 : List Nat := [821, 7253, 9521]
set_option maxRecDepth 8192 in
theorem root53Candidates2_checked : candidates edge 4 263 647 root53Tail3 winner.bound = root53Candidates2 := by
  unfold candidates
  have row : edge 647 = (SortedNatTree.contains FastRows.row647) := by
    change Data.row647.contains = (SortedNatTree.contains FastRows.row647)
    exact FastRows.row647_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root53Child2 : search edge 3 [263, 647] root53Candidates2 910 winner = winner := by decide
theorem root53Step2 : scan edge (search edge 3) 4 [263] 263 root53Tail2 winner =
    scan edge (search edge 3) 4 [263] 263 root53Tail3 winner := by
  rw [root53Tail2, scan, if_neg (by decide), root53Candidates2_checked]
  change scan edge (search edge 3) 4 [263] 263 root53Tail3
    (if 3 ≤ root53Candidates2.length then search edge 3 [263, 647] root53Candidates2 910 winner else winner) = _
  rw [if_pos (by decide), root53Child2]
def root53Candidates3 : List Nat := [3923, 20939, 23081]
set_option maxRecDepth 8192 in
theorem root53Candidates3_checked : candidates edge 4 263 761 root53Tail4 winner.bound = root53Candidates3 := by
  unfold candidates
  have row : edge 761 = (SortedNatTree.contains FastRows.row761) := by
    change Data.row761.contains = (SortedNatTree.contains FastRows.row761)
    exact FastRows.row761_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root53Child3 : search edge 3 [263, 761] root53Candidates3 1024 winner = winner := by decide
theorem root53Step3 : scan edge (search edge 3) 4 [263] 263 root53Tail3 winner =
    scan edge (search edge 3) 4 [263] 263 root53Tail4 winner := by
  rw [root53Tail3, scan, if_neg (by decide), root53Candidates3_checked]
  change scan edge (search edge 3) 4 [263] 263 root53Tail4
    (if 3 ≤ root53Candidates3.length then search edge 3 [263, 761] root53Candidates3 1024 winner else winner) = _
  rw [if_pos (by decide), root53Child3]
def root53Candidates4 : List Nat := [15581, 17807, 23057]
set_option maxRecDepth 8192 in
theorem root53Candidates4_checked : candidates edge 4 263 821 root53Tail5 winner.bound = root53Candidates4 := by
  unfold candidates
  have row : edge 821 = (SortedNatTree.contains FastRows.row821) := by
    change Data.row821.contains = (SortedNatTree.contains FastRows.row821)
    exact FastRows.row821_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root53Child4 : search edge 3 [263, 821] root53Candidates4 1084 winner = winner := by decide
theorem root53Step4 : scan edge (search edge 3) 4 [263] 263 root53Tail4 winner =
    scan edge (search edge 3) 4 [263] 263 root53Tail5 winner := by
  rw [root53Tail4, scan, if_neg (by decide), root53Candidates4_checked]
  change scan edge (search edge 3) 4 [263] 263 root53Tail5
    (if 3 ≤ root53Candidates4.length then search edge 3 [263, 821] root53Candidates4 1084 winner else winner) = _
  rw [if_pos (by decide), root53Child4]
def root53Candidates5 : List Nat := [1049, 2099, 8693, 23333]
set_option maxRecDepth 8192 in
theorem root53Candidates5_checked : candidates edge 4 263 941 root53Tail6 winner.bound = root53Candidates5 := by
  unfold candidates
  have row : edge 941 = (SortedNatTree.contains FastRows.row941) := by
    change Data.row941.contains = (SortedNatTree.contains FastRows.row941)
    exact FastRows.row941_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root53Child5 : search edge 3 [263, 941] root53Candidates5 1204 winner = winner := by decide
theorem root53Step5 : scan edge (search edge 3) 4 [263] 263 root53Tail5 winner =
    scan edge (search edge 3) 4 [263] 263 root53Tail6 winner := by
  rw [root53Tail5, scan, if_neg (by decide), root53Candidates5_checked]
  change scan edge (search edge 3) 4 [263] 263 root53Tail6
    (if 3 ≤ root53Candidates5.length then search edge 3 [263, 941] root53Candidates5 1204 winner else winner) = _
  rw [if_pos (by decide), root53Child5]
def root53Candidates6 : List Nat := [2633, 7013, 9323]
set_option maxRecDepth 8192 in
theorem root53Candidates6_checked : candidates edge 4 263 1049 root53Tail7 winner.bound = root53Candidates6 := by
  unfold candidates
  have row : edge 1049 = (SortedNatTree.contains FastRows.row1049) := by
    change Data.row1049.contains = (SortedNatTree.contains FastRows.row1049)
    exact FastRows.row1049_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root53Child6 : search edge 3 [263, 1049] root53Candidates6 1312 winner = winner := by decide
theorem root53Step6 : scan edge (search edge 3) 4 [263] 263 root53Tail6 winner =
    scan edge (search edge 3) 4 [263] 263 root53Tail7 winner := by
  rw [root53Tail6, scan, if_neg (by decide), root53Candidates6_checked]
  change scan edge (search edge 3) 4 [263] 263 root53Tail7
    (if 3 ≤ root53Candidates6.length then search edge 3 [263, 1049] root53Candidates6 1312 winner else winner) = _
  rw [if_pos (by decide), root53Child6]
def root53Candidates7 : List Nat := [2633, 10181, 15581]
set_option maxRecDepth 8192 in
theorem root53Candidates7_checked : candidates edge 4 263 1091 root53Tail8 winner.bound = root53Candidates7 := by
  unfold candidates
  have row : edge 1091 = (SortedNatTree.contains FastRows.row1091) := by
    change Data.row1091.contains = (SortedNatTree.contains FastRows.row1091)
    exact FastRows.row1091_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root53Child7 : search edge 3 [263, 1091] root53Candidates7 1354 winner = winner := by decide
theorem root53Step7 : scan edge (search edge 3) 4 [263] 263 root53Tail7 winner =
    scan edge (search edge 3) 4 [263] 263 root53Tail8 winner := by
  rw [root53Tail7, scan, if_neg (by decide), root53Candidates7_checked]
  change scan edge (search edge 3) 4 [263] 263 root53Tail8
    (if 3 ≤ root53Candidates7.length then search edge 3 [263, 1091] root53Candidates7 1354 winner else winner) = _
  rw [if_pos (by decide), root53Child7]
def root53Candidates8 : List Nat := [2129, 21059, 21143]
set_option maxRecDepth 8192 in
theorem root53Candidates8_checked : candidates edge 4 263 2099 root53Tail9 winner.bound = root53Candidates8 := by
  unfold candidates
  have row : edge 2099 = (SortedNatTree.contains FastRows.row2099) := by
    change Data.row2099.contains = (SortedNatTree.contains FastRows.row2099)
    exact FastRows.row2099_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root53Child8 : search edge 3 [263, 2099] root53Candidates8 2362 winner = winner := by decide
theorem root53Step8 : scan edge (search edge 3) 4 [263] 263 root53Tail8 winner =
    scan edge (search edge 3) 4 [263] 263 root53Tail9 winner := by
  rw [root53Tail8, scan, if_neg (by decide), root53Candidates8_checked]
  change scan edge (search edge 3) 4 [263] 263 root53Tail9
    (if 3 ≤ root53Candidates8.length then search edge 3 [263, 2099] root53Candidates8 2362 winner else winner) = _
  rw [if_pos (by decide), root53Child8]
def root53Candidates9 : List Nat := [7937, 10457, 20051]
set_option maxRecDepth 8192 in
theorem root53Candidates9_checked : candidates edge 4 263 2129 root53Tail10 winner.bound = root53Candidates9 := by
  unfold candidates
  have row : edge 2129 = (SortedNatTree.contains FastRows.row2129) := by
    change Data.row2129.contains = (SortedNatTree.contains FastRows.row2129)
    exact FastRows.row2129_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root53Child9 : search edge 3 [263, 2129] root53Candidates9 2392 winner = winner := by decide
theorem root53Step9 : scan edge (search edge 3) 4 [263] 263 root53Tail9 winner =
    scan edge (search edge 3) 4 [263] 263 root53Tail10 winner := by
  rw [root53Tail9, scan, if_neg (by decide), root53Candidates9_checked]
  change scan edge (search edge 3) 4 [263] 263 root53Tail10
    (if 3 ≤ root53Candidates9.length then search edge 3 [263, 2129] root53Candidates9 2392 winner else winner) = _
  rw [if_pos (by decide), root53Child9]
def root53Candidates10 : List Nat := [7013, 18917]
set_option maxRecDepth 8192 in
theorem root53Candidates10_checked : candidates edge 4 263 2339 root53Tail11 winner.bound = root53Candidates10 := by
  unfold candidates
  have row : edge 2339 = (SortedNatTree.contains FastRows.row2339) := by
    change Data.row2339.contains = (SortedNatTree.contains FastRows.row2339)
    exact FastRows.row2339_checked.symm
  rw [row]
  decide
theorem root53Step10 : scan edge (search edge 3) 4 [263] 263 root53Tail10 winner =
    scan edge (search edge 3) 4 [263] 263 root53Tail11 winner := by
  rw [root53Tail10, scan, if_neg (by decide), root53Candidates10_checked]
  change scan edge (search edge 3) 4 [263] 263 root53Tail11
    (if 3 ≤ root53Candidates10.length then search edge 3 [263, 2339] root53Candidates10 2602 winner else winner) = _
  rw [if_neg (by decide)]
def root53Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root53Candidates11_checked : candidates edge 4 263 2477 root53Tail12 winner.bound = root53Candidates11 := by
  unfold candidates
  have row : edge 2477 = (SortedNatTree.contains FastRows.row2477) := by
    change Data.row2477.contains = (SortedNatTree.contains FastRows.row2477)
    exact FastRows.row2477_checked.symm
  rw [row]
  decide
theorem root53Step11 : scan edge (search edge 3) 4 [263] 263 root53Tail11 winner =
    scan edge (search edge 3) 4 [263] 263 root53Tail12 winner := by
  rw [root53Tail11, scan, if_neg (by decide), root53Candidates11_checked]
  change scan edge (search edge 3) 4 [263] 263 root53Tail12
    (if 3 ≤ root53Candidates11.length then search edge 3 [263, 2477] root53Candidates11 2740 winner else winner) = _
  rw [if_neg (by decide)]
def root53Candidates12 : List Nat := [5693, 20051]
set_option maxRecDepth 8192 in
theorem root53Candidates12_checked : candidates edge 4 263 2633 root53Tail13 winner.bound = root53Candidates12 := by
  unfold candidates
  have row : edge 2633 = (SortedNatTree.contains FastRows.row2633) := by
    change Data.row2633.contains = (SortedNatTree.contains FastRows.row2633)
    exact FastRows.row2633_checked.symm
  rw [row]
  decide
theorem root53Step12 : scan edge (search edge 3) 4 [263] 263 root53Tail12 winner =
    scan edge (search edge 3) 4 [263] 263 root53Tail13 winner := by
  rw [root53Tail12, scan, if_neg (by decide), root53Candidates12_checked]
  change scan edge (search edge 3) 4 [263] 263 root53Tail13
    (if 3 ≤ root53Candidates12.length then search edge 3 [263, 2633] root53Candidates12 2896 winner else winner) = _
  rw [if_neg (by decide)]
def root53Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root53Candidates13_checked : candidates edge 4 263 2789 root53Tail14 winner.bound = root53Candidates13 := by
  unfold candidates
  have row : edge 2789 = (SortedNatTree.contains FastRows.row2789) := by
    change Data.row2789.contains = (SortedNatTree.contains FastRows.row2789)
    exact FastRows.row2789_checked.symm
  rw [row]
  decide
theorem root53Step13 : scan edge (search edge 3) 4 [263] 263 root53Tail13 winner =
    scan edge (search edge 3) 4 [263] 263 root53Tail14 winner := by
  rw [root53Tail13, scan, if_neg (by decide), root53Candidates13_checked]
  change scan edge (search edge 3) 4 [263] 263 root53Tail14
    (if 3 ≤ root53Candidates13.length then search edge 3 [263, 2789] root53Candidates13 3052 winner else winner) = _
  rw [if_neg (by decide)]
def root53Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root53Candidates14_checked : candidates edge 4 263 2999 root53Tail15 winner.bound = root53Candidates14 := by
  unfold candidates
  have row : edge 2999 = (SortedNatTree.contains FastRows.row2999) := by
    change Data.row2999.contains = (SortedNatTree.contains FastRows.row2999)
    exact FastRows.row2999_checked.symm
  rw [row]
  decide
theorem root53Step14 : scan edge (search edge 3) 4 [263] 263 root53Tail14 winner =
    scan edge (search edge 3) 4 [263] 263 root53Tail15 winner := by
  rw [root53Tail14, scan, if_neg (by decide), root53Candidates14_checked]
  change scan edge (search edge 3) 4 [263] 263 root53Tail15
    (if 3 ≤ root53Candidates14.length then search edge 3 [263, 2999] root53Candidates14 3262 winner else winner) = _
  rw [if_neg (by decide)]
def root53Candidates15 : List Nat := [9521, 21143]
set_option maxRecDepth 8192 in
theorem root53Candidates15_checked : candidates edge 4 263 3359 root53Tail16 winner.bound = root53Candidates15 := by
  unfold candidates
  have row : edge 3359 = (SortedNatTree.contains FastRows.row3359) := by
    change Data.row3359.contains = (SortedNatTree.contains FastRows.row3359)
    exact FastRows.row3359_checked.symm
  rw [row]
  decide
theorem root53Step15 : scan edge (search edge 3) 4 [263] 263 root53Tail15 winner =
    scan edge (search edge 3) 4 [263] 263 root53Tail16 winner := by
  rw [root53Tail15, scan, if_neg (by decide), root53Candidates15_checked]
  change scan edge (search edge 3) 4 [263] 263 root53Tail16
    (if 3 ≤ root53Candidates15.length then search edge 3 [263, 3359] root53Candidates15 3622 winner else winner) = _
  rw [if_neg (by decide)]
def root53Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root53Candidates16_checked : candidates edge 4 263 3467 root53Tail17 winner.bound = root53Candidates16 := by
  unfold candidates
  have row : edge 3467 = (SortedNatTree.contains FastRows.row3467) := by
    change Data.row3467.contains = (SortedNatTree.contains FastRows.row3467)
    exact FastRows.row3467_checked.symm
  rw [row]
  decide
theorem root53Step16 : scan edge (search edge 3) 4 [263] 263 root53Tail16 winner =
    scan edge (search edge 3) 4 [263] 263 root53Tail17 winner := by
  rw [root53Tail16, scan, if_neg (by decide), root53Candidates16_checked]
  change scan edge (search edge 3) 4 [263] 263 root53Tail17
    (if 3 ≤ root53Candidates16.length then search edge 3 [263, 3467] root53Candidates16 3730 winner else winner) = _
  rw [if_neg (by decide)]
def root53Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root53Candidates17_checked : candidates edge 4 263 3557 root53Tail18 winner.bound = root53Candidates17 := by
  unfold candidates
  have row : edge 3557 = (SortedNatTree.contains FastRows.row3557) := by
    change Data.row3557.contains = (SortedNatTree.contains FastRows.row3557)
    exact FastRows.row3557_checked.symm
  rw [row]
  decide
theorem root53Step17 : scan edge (search edge 3) 4 [263] 263 root53Tail17 winner =
    scan edge (search edge 3) 4 [263] 263 root53Tail18 winner := by
  rw [root53Tail17, scan, if_neg (by decide), root53Candidates17_checked]
  change scan edge (search edge 3) 4 [263] 263 root53Tail18
    (if 3 ≤ root53Candidates17.length then search edge 3 [263, 3557] root53Candidates17 3820 winner else winner) = _
  rw [if_neg (by decide)]
def root53Candidates18 : List Nat := [11783]
set_option maxRecDepth 8192 in
theorem root53Candidates18_checked : candidates edge 4 263 3923 root53Tail19 winner.bound = root53Candidates18 := by
  unfold candidates
  have row : edge 3923 = (SortedNatTree.contains FastRows.row3923) := by
    change Data.row3923.contains = (SortedNatTree.contains FastRows.row3923)
    exact FastRows.row3923_checked.symm
  rw [row]
  decide
theorem root53Step18 : scan edge (search edge 3) 4 [263] 263 root53Tail18 winner =
    scan edge (search edge 3) 4 [263] 263 root53Tail19 winner := by
  rw [root53Tail18, scan, if_neg (by decide), root53Candidates18_checked]
  change scan edge (search edge 3) 4 [263] 263 root53Tail19
    (if 3 ≤ root53Candidates18.length then search edge 3 [263, 3923] root53Candidates18 4186 winner else winner) = _
  rw [if_neg (by decide)]
def root53Candidates19 : List Nat := [7937, 10457, 18353]
set_option maxRecDepth 8192 in
theorem root53Candidates19_checked : candidates edge 4 263 4523 root53Tail20 winner.bound = root53Candidates19 := by
  unfold candidates
  have row : edge 4523 = (SortedNatTree.contains FastRows.row4523) := by
    change Data.row4523.contains = (SortedNatTree.contains FastRows.row4523)
    exact FastRows.row4523_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root53Child19 : search edge 3 [263, 4523] root53Candidates19 4786 winner = winner := by decide
theorem root53Step19 : scan edge (search edge 3) 4 [263] 263 root53Tail19 winner =
    scan edge (search edge 3) 4 [263] 263 root53Tail20 winner := by
  rw [root53Tail19, scan, if_neg (by decide), root53Candidates19_checked]
  change scan edge (search edge 3) 4 [263] 263 root53Tail20
    (if 3 ≤ root53Candidates19.length then search edge 3 [263, 4523] root53Candidates19 4786 winner else winner) = _
  rw [if_pos (by decide), root53Child19]
def root53Candidates20 : List Nat := [10331]
set_option maxRecDepth 8192 in
theorem root53Candidates20_checked : candidates edge 4 263 5189 root53Tail21 winner.bound = root53Candidates20 := by
  unfold candidates
  have row : edge 5189 = (SortedNatTree.contains FastRows.row5189) := by
    change Data.row5189.contains = (SortedNatTree.contains FastRows.row5189)
    exact FastRows.row5189_checked.symm
  rw [row]
  decide
theorem root53Step20 : scan edge (search edge 3) 4 [263] 263 root53Tail20 winner =
    scan edge (search edge 3) 4 [263] 263 root53Tail21 winner := by
  rw [root53Tail20, scan, if_neg (by decide), root53Candidates20_checked]
  change scan edge (search edge 3) 4 [263] 263 root53Tail21
    (if 3 ≤ root53Candidates20.length then search edge 3 [263, 5189] root53Candidates20 5452 winner else winner) = _
  rw [if_neg (by decide)]
def root53Candidates21 : List Nat := []
set_option maxRecDepth 8192 in
theorem root53Candidates21_checked : candidates edge 4 263 5693 root53Tail22 winner.bound = root53Candidates21 := by
  unfold candidates
  have row : edge 5693 = (SortedNatTree.contains FastRows.row5693) := by
    change Data.row5693.contains = (SortedNatTree.contains FastRows.row5693)
    exact FastRows.row5693_checked.symm
  rw [row]
  decide
theorem root53Step21 : scan edge (search edge 3) 4 [263] 263 root53Tail21 winner =
    scan edge (search edge 3) 4 [263] 263 root53Tail22 winner := by
  rw [root53Tail21, scan, if_neg (by decide), root53Candidates21_checked]
  change scan edge (search edge 3) 4 [263] 263 root53Tail22
    (if 3 ≤ root53Candidates21.length then search edge 3 [263, 5693] root53Candidates21 5956 winner else winner) = _
  rw [if_neg (by decide)]
def root53Candidates22 : List Nat := [14489, 19259]
set_option maxRecDepth 8192 in
theorem root53Candidates22_checked : candidates edge 4 263 5987 root53Tail23 winner.bound = root53Candidates22 := by
  unfold candidates
  have row : edge 5987 = (SortedNatTree.contains FastRows.row5987) := by
    change Data.row5987.contains = (SortedNatTree.contains FastRows.row5987)
    exact FastRows.row5987_checked.symm
  rw [row]
  decide
theorem root53Step22 : scan edge (search edge 3) 4 [263] 263 root53Tail22 winner =
    scan edge (search edge 3) 4 [263] 263 root53Tail23 winner := by
  rw [root53Tail22, scan, if_neg (by decide), root53Candidates22_checked]
  change scan edge (search edge 3) 4 [263] 263 root53Tail23
    (if 3 ≤ root53Candidates22.length then search edge 3 [263, 5987] root53Candidates22 6250 winner else winner) = _
  rw [if_neg (by decide)]
def root53Candidates23 : List Nat := [19259]
set_option maxRecDepth 8192 in
theorem root53Candidates23_checked : candidates edge 4 263 6197 root53Tail24 winner.bound = root53Candidates23 := by
  unfold candidates
  have row : edge 6197 = (SortedNatTree.contains FastRows.row6197) := by
    change Data.row6197.contains = (SortedNatTree.contains FastRows.row6197)
    exact FastRows.row6197_checked.symm
  rw [row]
  decide
theorem root53Step23 : scan edge (search edge 3) 4 [263] 263 root53Tail23 winner =
    scan edge (search edge 3) 4 [263] 263 root53Tail24 winner := by
  rw [root53Tail23, scan, if_neg (by decide), root53Candidates23_checked]
  change scan edge (search edge 3) 4 [263] 263 root53Tail24
    (if 3 ≤ root53Candidates23.length then search edge 3 [263, 6197] root53Candidates23 6460 winner else winner) = _
  rw [if_neg (by decide)]
theorem root53Scan24 : scan edge (search edge 3) 4 [263] 263 root53Tail24 winner = winner := by rw [root53Tail24, scan, if_pos (by decide)]
theorem root53Scan23 : scan edge (search edge 3) 4 [263] 263 root53Tail23 winner = winner :=
  (root53Step23).trans root53Scan24
theorem root53Scan22 : scan edge (search edge 3) 4 [263] 263 root53Tail22 winner = winner :=
  (root53Step22).trans root53Scan23
theorem root53Scan21 : scan edge (search edge 3) 4 [263] 263 root53Tail21 winner = winner :=
  (root53Step21).trans root53Scan22
theorem root53Scan20 : scan edge (search edge 3) 4 [263] 263 root53Tail20 winner = winner :=
  (root53Step20).trans root53Scan21
theorem root53Scan19 : scan edge (search edge 3) 4 [263] 263 root53Tail19 winner = winner :=
  (root53Step19).trans root53Scan20
theorem root53Scan18 : scan edge (search edge 3) 4 [263] 263 root53Tail18 winner = winner :=
  (root53Step18).trans root53Scan19
theorem root53Scan17 : scan edge (search edge 3) 4 [263] 263 root53Tail17 winner = winner :=
  (root53Step17).trans root53Scan18
theorem root53Scan16 : scan edge (search edge 3) 4 [263] 263 root53Tail16 winner = winner :=
  (root53Step16).trans root53Scan17
theorem root53Scan15 : scan edge (search edge 3) 4 [263] 263 root53Tail15 winner = winner :=
  (root53Step15).trans root53Scan16
theorem root53Scan14 : scan edge (search edge 3) 4 [263] 263 root53Tail14 winner = winner :=
  (root53Step14).trans root53Scan15
theorem root53Scan13 : scan edge (search edge 3) 4 [263] 263 root53Tail13 winner = winner :=
  (root53Step13).trans root53Scan14
theorem root53Scan12 : scan edge (search edge 3) 4 [263] 263 root53Tail12 winner = winner :=
  (root53Step12).trans root53Scan13
theorem root53Scan11 : scan edge (search edge 3) 4 [263] 263 root53Tail11 winner = winner :=
  (root53Step11).trans root53Scan12
theorem root53Scan10 : scan edge (search edge 3) 4 [263] 263 root53Tail10 winner = winner :=
  (root53Step10).trans root53Scan11
theorem root53Scan9 : scan edge (search edge 3) 4 [263] 263 root53Tail9 winner = winner :=
  (root53Step9).trans root53Scan10
theorem root53Scan8 : scan edge (search edge 3) 4 [263] 263 root53Tail8 winner = winner :=
  (root53Step8).trans root53Scan9
theorem root53Scan7 : scan edge (search edge 3) 4 [263] 263 root53Tail7 winner = winner :=
  (root53Step7).trans root53Scan8
theorem root53Scan6 : scan edge (search edge 3) 4 [263] 263 root53Tail6 winner = winner :=
  (root53Step6).trans root53Scan7
theorem root53Scan5 : scan edge (search edge 3) 4 [263] 263 root53Tail5 winner = winner :=
  (root53Step5).trans root53Scan6
theorem root53Scan4 : scan edge (search edge 3) 4 [263] 263 root53Tail4 winner = winner :=
  (root53Step4).trans root53Scan5
theorem root53Scan3 : scan edge (search edge 3) 4 [263] 263 root53Tail3 winner = winner :=
  (root53Step3).trans root53Scan4
theorem root53Scan2 : scan edge (search edge 3) 4 [263] 263 root53Tail2 winner = winner :=
  (root53Step2).trans root53Scan3
theorem root53Scan1 : scan edge (search edge 3) 4 [263] 263 root53Tail1 winner = winner :=
  (root53Step1).trans root53Scan2
theorem root53Scan0 : scan edge (search edge 3) 4 [263] 263 root53Tail0 winner = winner :=
  (root53Step0).trans root53Scan1
set_option maxRecDepth 4096 in
theorem root53Branch : search edge 4 [263] root53Candidates 263 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [263] 263 root53Tail0 winner = winner
  exact root53Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
