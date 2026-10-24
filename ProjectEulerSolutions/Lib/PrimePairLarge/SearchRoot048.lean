import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root48Candidates : List Nat := Data.row233
set_option maxRecDepth 8192 in
theorem root48Candidates_checked : candidates edge 5 0 233 tail49 winner.bound = root48Candidates := by
  apply candidates_eq_row edge 5 0 233 tail49 root48Candidates winner.bound tail49_ordered
  · decide
  · decide
  · funext q
    rfl
def root48Tail76 : List Nat := []
def root48Tail75 : List Nat := 24683 :: root48Tail76
def root48Tail74 : List Nat := 24083 :: root48Tail75
def root48Tail73 : List Nat := 24029 :: root48Tail74
def root48Tail72 : List Nat := 23831 :: root48Tail73
def root48Tail71 : List Nat := 23567 :: root48Tail72
def root48Tail70 : List Nat := 23189 :: root48Tail71
def root48Tail69 : List Nat := 21377 :: root48Tail70
def root48Tail68 : List Nat := 20681 :: root48Tail69
def root48Tail67 : List Nat := 20327 :: root48Tail68
def root48Tail66 : List Nat := 19949 :: root48Tail67
def root48Tail65 : List Nat := 19913 :: root48Tail66
def root48Tail64 : List Nat := 19391 :: root48Tail65
def root48Tail63 : List Nat := 19319 :: root48Tail64
def root48Tail62 : List Nat := 19013 :: root48Tail63
def root48Tail61 : List Nat := 18899 :: root48Tail62
def root48Tail60 : List Nat := 18749 :: root48Tail61
def root48Tail59 : List Nat := 18593 :: root48Tail60
def root48Tail58 : List Nat := 17669 :: root48Tail59
def root48Tail57 : List Nat := 17027 :: root48Tail58
def root48Tail56 : List Nat := 16901 :: root48Tail57
def root48Tail55 : List Nat := 16829 :: root48Tail56
def root48Tail54 : List Nat := 16481 :: root48Tail55
def root48Tail53 : List Nat := 16427 :: root48Tail54
def root48Tail52 : List Nat := 15473 :: root48Tail53
def root48Tail51 : List Nat := 14831 :: root48Tail52
def root48Tail50 : List Nat := 14747 :: root48Tail51
def root48Tail49 : List Nat := 14561 :: root48Tail50
def root48Tail48 : List Nat := 14321 :: root48Tail49
def root48Tail47 : List Nat := 13907 :: root48Tail48
def root48Tail46 : List Nat := 13781 :: root48Tail47
def root48Tail45 : List Nat := 13751 :: root48Tail46
def root48Tail44 : List Nat := 12071 :: root48Tail45
def root48Tail43 : List Nat := 11801 :: root48Tail44
def root48Tail42 : List Nat := 11789 :: root48Tail43
def root48Tail41 : List Nat := 11261 :: root48Tail42
def root48Tail40 : List Nat := 11159 :: root48Tail41
def root48Tail39 : List Nat := 10253 :: root48Tail40
def root48Tail38 : List Nat := 9929 :: root48Tail39
def root48Tail37 : List Nat := 9257 :: root48Tail38
def root48Tail36 : List Nat := 8933 :: root48Tail37
def root48Tail35 : List Nat := 8867 :: root48Tail36
def root48Tail34 : List Nat := 8291 :: root48Tail35
def root48Tail33 : List Nat := 8123 :: root48Tail34
def root48Tail32 : List Nat := 7949 :: root48Tail33
def root48Tail31 : List Nat := 7607 :: root48Tail32
def root48Tail30 : List Nat := 7547 :: root48Tail31
def root48Tail29 : List Nat := 7187 :: root48Tail30
def root48Tail28 : List Nat := 7043 :: root48Tail29
def root48Tail27 : List Nat := 7001 :: root48Tail28
def root48Tail26 : List Nat := 6947 :: root48Tail27
def root48Tail25 : List Nat := 6359 :: root48Tail26
def root48Tail24 : List Nat := 6263 :: root48Tail25
def root48Tail23 : List Nat := 6011 :: root48Tail24
def root48Tail22 : List Nat := 5813 :: root48Tail23
def root48Tail21 : List Nat := 5807 :: root48Tail22
def root48Tail20 : List Nat := 5591 :: root48Tail21
def root48Tail19 : List Nat := 5297 :: root48Tail20
def root48Tail18 : List Nat := 5009 :: root48Tail19
def root48Tail17 : List Nat := 4793 :: root48Tail18
def root48Tail16 : List Nat := 4337 :: root48Tail17
def root48Tail15 : List Nat := 3389 :: root48Tail16
def root48Tail14 : List Nat := 3329 :: root48Tail15
def root48Tail13 : List Nat := 3167 :: root48Tail14
def root48Tail12 : List Nat := 3083 :: root48Tail13
def root48Tail11 : List Nat := 2969 :: root48Tail12
def root48Tail10 : List Nat := 1583 :: root48Tail11
def root48Tail9 : List Nat := 1187 :: root48Tail10
def root48Tail8 : List Nat := 1163 :: root48Tail9
def root48Tail7 : List Nat := 1151 :: root48Tail8
def root48Tail6 : List Nat := 983 :: root48Tail7
def root48Tail5 : List Nat := 911 :: root48Tail6
def root48Tail4 : List Nat := 881 :: root48Tail5
def root48Tail3 : List Nat := 617 :: root48Tail4
def root48Tail2 : List Nat := 347 :: root48Tail3
def root48Tail1 : List Nat := 251 :: root48Tail2
def root48Tail0 : List Nat := 239 :: root48Tail1
def root48Candidates0 : List Nat := [347, 5807, 19319]
set_option maxRecDepth 8192 in
theorem root48Candidates0_checked : candidates edge 4 233 239 root48Tail1 winner.bound = root48Candidates0 := by
  unfold candidates
  have row : edge 239 = (SortedNatTree.contains FastRows.row239) := by
    change Data.row239.contains = (SortedNatTree.contains FastRows.row239)
    exact FastRows.row239_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root48Child0 : search edge 3 [233, 239] root48Candidates0 472 winner = winner := by decide
theorem root48Step0 : scan edge (search edge 3) 4 [233] 233 root48Tail0 winner =
    scan edge (search edge 3) 4 [233] 233 root48Tail1 winner := by
  rw [root48Tail0, scan, if_neg (by decide), root48Candidates0_checked]
  change scan edge (search edge 3) 4 [233] 233 root48Tail1
    (if 3 ≤ root48Candidates0.length then search edge 3 [233, 239] root48Candidates0 472 winner else winner) = _
  rw [if_pos (by decide), root48Child0]
def root48Candidates1 : List Nat := [347, 1151, 8933, 24683]
set_option maxRecDepth 8192 in
theorem root48Candidates1_checked : candidates edge 4 233 251 root48Tail2 winner.bound = root48Candidates1 := by
  unfold candidates
  have row : edge 251 = (SortedNatTree.contains FastRows.row251) := by
    change Data.row251.contains = (SortedNatTree.contains FastRows.row251)
    exact FastRows.row251_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root48Child1 : search edge 3 [233, 251] root48Candidates1 484 winner = winner := by decide
theorem root48Step1 : scan edge (search edge 3) 4 [233] 233 root48Tail1 winner =
    scan edge (search edge 3) 4 [233] 233 root48Tail2 winner := by
  rw [root48Tail1, scan, if_neg (by decide), root48Candidates1_checked]
  change scan edge (search edge 3) 4 [233] 233 root48Tail2
    (if 3 ≤ root48Candidates1.length then search edge 3 [233, 251] root48Candidates1 484 winner else winner) = _
  rw [if_pos (by decide), root48Child1]
def root48Candidates2 : List Nat := [983, 17027]
set_option maxRecDepth 8192 in
theorem root48Candidates2_checked : candidates edge 4 233 347 root48Tail3 winner.bound = root48Candidates2 := by
  unfold candidates
  have row : edge 347 = (SortedNatTree.contains FastRows.row347) := by
    change Data.row347.contains = (SortedNatTree.contains FastRows.row347)
    exact FastRows.row347_checked.symm
  rw [row]
  decide
theorem root48Step2 : scan edge (search edge 3) 4 [233] 233 root48Tail2 winner =
    scan edge (search edge 3) 4 [233] 233 root48Tail3 winner := by
  rw [root48Tail2, scan, if_neg (by decide), root48Candidates2_checked]
  change scan edge (search edge 3) 4 [233] 233 root48Tail3
    (if 3 ≤ root48Candidates2.length then search edge 3 [233, 347] root48Candidates2 580 winner else winner) = _
  rw [if_neg (by decide)]
def root48Candidates3 : List Nat := [983, 6011]
set_option maxRecDepth 8192 in
theorem root48Candidates3_checked : candidates edge 4 233 617 root48Tail4 winner.bound = root48Candidates3 := by
  unfold candidates
  have row : edge 617 = (SortedNatTree.contains FastRows.row617) := by
    change Data.row617.contains = (SortedNatTree.contains FastRows.row617)
    exact FastRows.row617_checked.symm
  rw [row]
  decide
theorem root48Step3 : scan edge (search edge 3) 4 [233] 233 root48Tail3 winner =
    scan edge (search edge 3) 4 [233] 233 root48Tail4 winner := by
  rw [root48Tail3, scan, if_neg (by decide), root48Candidates3_checked]
  change scan edge (search edge 3) 4 [233] 233 root48Tail4
    (if 3 ≤ root48Candidates3.length then search edge 3 [233, 617] root48Candidates3 850 winner else winner) = _
  rw [if_neg (by decide)]
def root48Candidates4 : List Nat := [983, 5591, 7607]
set_option maxRecDepth 8192 in
theorem root48Candidates4_checked : candidates edge 4 233 881 root48Tail5 winner.bound = root48Candidates4 := by
  unfold candidates
  have row : edge 881 = (SortedNatTree.contains FastRows.row881) := by
    change Data.row881.contains = (SortedNatTree.contains FastRows.row881)
    exact FastRows.row881_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root48Child4 : search edge 3 [233, 881] root48Candidates4 1114 winner = winner := by decide
theorem root48Step4 : scan edge (search edge 3) 4 [233] 233 root48Tail4 winner =
    scan edge (search edge 3) 4 [233] 233 root48Tail5 winner := by
  rw [root48Tail4, scan, if_neg (by decide), root48Candidates4_checked]
  change scan edge (search edge 3) 4 [233] 233 root48Tail5
    (if 3 ≤ root48Candidates4.length then search edge 3 [233, 881] root48Candidates4 1114 winner else winner) = _
  rw [if_pos (by decide), root48Child4]
def root48Candidates5 : List Nat := [4337, 11261, 13751, 23831]
set_option maxRecDepth 8192 in
theorem root48Candidates5_checked : candidates edge 4 233 911 root48Tail6 winner.bound = root48Candidates5 := by
  unfold candidates
  have row : edge 911 = (SortedNatTree.contains FastRows.row911) := by
    change Data.row911.contains = (SortedNatTree.contains FastRows.row911)
    exact FastRows.row911_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root48Child5 : search edge 3 [233, 911] root48Candidates5 1144 winner = winner := by decide
theorem root48Step5 : scan edge (search edge 3) 4 [233] 233 root48Tail5 winner =
    scan edge (search edge 3) 4 [233] 233 root48Tail6 winner := by
  rw [root48Tail5, scan, if_neg (by decide), root48Candidates5_checked]
  change scan edge (search edge 3) 4 [233] 233 root48Tail6
    (if 3 ≤ root48Candidates5.length then search edge 3 [233, 911] root48Candidates5 1144 winner else winner) = _
  rw [if_pos (by decide), root48Child5]
def root48Candidates6 : List Nat := [11789, 14747]
set_option maxRecDepth 8192 in
theorem root48Candidates6_checked : candidates edge 4 233 983 root48Tail7 winner.bound = root48Candidates6 := by
  unfold candidates
  have row : edge 983 = (SortedNatTree.contains FastRows.row983) := by
    change Data.row983.contains = (SortedNatTree.contains FastRows.row983)
    exact FastRows.row983_checked.symm
  rw [row]
  decide
theorem root48Step6 : scan edge (search edge 3) 4 [233] 233 root48Tail6 winner =
    scan edge (search edge 3) 4 [233] 233 root48Tail7 winner := by
  rw [root48Tail6, scan, if_neg (by decide), root48Candidates6_checked]
  change scan edge (search edge 3) 4 [233] 233 root48Tail7
    (if 3 ≤ root48Candidates6.length then search edge 3 [233, 983] root48Candidates6 1216 winner else winner) = _
  rw [if_neg (by decide)]
def root48Candidates7 : List Nat := [4337]
set_option maxRecDepth 8192 in
theorem root48Candidates7_checked : candidates edge 4 233 1151 root48Tail8 winner.bound = root48Candidates7 := by
  unfold candidates
  have row : edge 1151 = (SortedNatTree.contains FastRows.row1151) := by
    change Data.row1151.contains = (SortedNatTree.contains FastRows.row1151)
    exact FastRows.row1151_checked.symm
  rw [row]
  decide
theorem root48Step7 : scan edge (search edge 3) 4 [233] 233 root48Tail7 winner =
    scan edge (search edge 3) 4 [233] 233 root48Tail8 winner := by
  rw [root48Tail7, scan, if_neg (by decide), root48Candidates7_checked]
  change scan edge (search edge 3) 4 [233] 233 root48Tail8
    (if 3 ≤ root48Candidates7.length then search edge 3 [233, 1151] root48Candidates7 1384 winner else winner) = _
  rw [if_neg (by decide)]
def root48Candidates8 : List Nat := [17027]
set_option maxRecDepth 8192 in
theorem root48Candidates8_checked : candidates edge 4 233 1163 root48Tail9 winner.bound = root48Candidates8 := by
  unfold candidates
  have row : edge 1163 = (SortedNatTree.contains FastRows.row1163) := by
    change Data.row1163.contains = (SortedNatTree.contains FastRows.row1163)
    exact FastRows.row1163_checked.symm
  rw [row]
  decide
theorem root48Step8 : scan edge (search edge 3) 4 [233] 233 root48Tail8 winner =
    scan edge (search edge 3) 4 [233] 233 root48Tail9 winner := by
  rw [root48Tail8, scan, if_neg (by decide), root48Candidates8_checked]
  change scan edge (search edge 3) 4 [233] 233 root48Tail9
    (if 3 ≤ root48Candidates8.length then search edge 3 [233, 1163] root48Candidates8 1396 winner else winner) = _
  rw [if_neg (by decide)]
def root48Candidates9 : List Nat := [13781]
set_option maxRecDepth 8192 in
theorem root48Candidates9_checked : candidates edge 4 233 1187 root48Tail10 winner.bound = root48Candidates9 := by
  unfold candidates
  have row : edge 1187 = (SortedNatTree.contains FastRows.row1187) := by
    change Data.row1187.contains = (SortedNatTree.contains FastRows.row1187)
    exact FastRows.row1187_checked.symm
  rw [row]
  decide
theorem root48Step9 : scan edge (search edge 3) 4 [233] 233 root48Tail9 winner =
    scan edge (search edge 3) 4 [233] 233 root48Tail10 winner := by
  rw [root48Tail9, scan, if_neg (by decide), root48Candidates9_checked]
  change scan edge (search edge 3) 4 [233] 233 root48Tail10
    (if 3 ≤ root48Candidates9.length then search edge 3 [233, 1187] root48Candidates9 1420 winner else winner) = _
  rw [if_neg (by decide)]
def root48Candidates10 : List Nat := [5009, 7043, 7949, 23831]
set_option maxRecDepth 8192 in
theorem root48Candidates10_checked : candidates edge 4 233 1583 root48Tail11 winner.bound = root48Candidates10 := by
  unfold candidates
  have row : edge 1583 = (SortedNatTree.contains FastRows.row1583) := by
    change Data.row1583.contains = (SortedNatTree.contains FastRows.row1583)
    exact FastRows.row1583_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root48Child10 : search edge 3 [233, 1583] root48Candidates10 1816 winner = winner := by decide
theorem root48Step10 : scan edge (search edge 3) 4 [233] 233 root48Tail10 winner =
    scan edge (search edge 3) 4 [233] 233 root48Tail11 winner := by
  rw [root48Tail10, scan, if_neg (by decide), root48Candidates10_checked]
  change scan edge (search edge 3) 4 [233] 233 root48Tail11
    (if 3 ≤ root48Candidates10.length then search edge 3 [233, 1583] root48Candidates10 1816 winner else winner) = _
  rw [if_pos (by decide), root48Child10]
def root48Candidates11 : List Nat := [5009, 14561, 21377]
set_option maxRecDepth 8192 in
theorem root48Candidates11_checked : candidates edge 4 233 2969 root48Tail12 winner.bound = root48Candidates11 := by
  unfold candidates
  have row : edge 2969 = (SortedNatTree.contains FastRows.row2969) := by
    change Data.row2969.contains = (SortedNatTree.contains FastRows.row2969)
    exact FastRows.row2969_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root48Child11 : search edge 3 [233, 2969] root48Candidates11 3202 winner = winner := by decide
theorem root48Step11 : scan edge (search edge 3) 4 [233] 233 root48Tail11 winner =
    scan edge (search edge 3) 4 [233] 233 root48Tail12 winner := by
  rw [root48Tail11, scan, if_neg (by decide), root48Candidates11_checked]
  change scan edge (search edge 3) 4 [233] 233 root48Tail12
    (if 3 ≤ root48Candidates11.length then search edge 3 [233, 2969] root48Candidates11 3202 winner else winner) = _
  rw [if_pos (by decide), root48Child11]
def root48Candidates12 : List Nat := [3329, 7187, 9929]
set_option maxRecDepth 8192 in
theorem root48Candidates12_checked : candidates edge 4 233 3083 root48Tail13 winner.bound = root48Candidates12 := by
  unfold candidates
  have row : edge 3083 = (SortedNatTree.contains FastRows.row3083) := by
    change Data.row3083.contains = (SortedNatTree.contains FastRows.row3083)
    exact FastRows.row3083_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root48Child12 : search edge 3 [233, 3083] root48Candidates12 3316 winner = winner := by decide
theorem root48Step12 : scan edge (search edge 3) 4 [233] 233 root48Tail12 winner =
    scan edge (search edge 3) 4 [233] 233 root48Tail13 winner := by
  rw [root48Tail12, scan, if_neg (by decide), root48Candidates12_checked]
  change scan edge (search edge 3) 4 [233] 233 root48Tail13
    (if 3 ≤ root48Candidates12.length then search edge 3 [233, 3083] root48Candidates12 3316 winner else winner) = _
  rw [if_pos (by decide), root48Child12]
def root48Candidates13 : List Nat := [14831]
set_option maxRecDepth 8192 in
theorem root48Candidates13_checked : candidates edge 4 233 3167 root48Tail14 winner.bound = root48Candidates13 := by
  unfold candidates
  have row : edge 3167 = (SortedNatTree.contains FastRows.row3167) := by
    change Data.row3167.contains = (SortedNatTree.contains FastRows.row3167)
    exact FastRows.row3167_checked.symm
  rw [row]
  decide
theorem root48Step13 : scan edge (search edge 3) 4 [233] 233 root48Tail13 winner =
    scan edge (search edge 3) 4 [233] 233 root48Tail14 winner := by
  rw [root48Tail13, scan, if_neg (by decide), root48Candidates13_checked]
  change scan edge (search edge 3) 4 [233] 233 root48Tail14
    (if 3 ≤ root48Candidates13.length then search edge 3 [233, 3167] root48Candidates13 3400 winner else winner) = _
  rw [if_neg (by decide)]
def root48Candidates14 : List Nat := [6359, 7001]
set_option maxRecDepth 8192 in
theorem root48Candidates14_checked : candidates edge 4 233 3329 root48Tail15 winner.bound = root48Candidates14 := by
  unfold candidates
  have row : edge 3329 = (SortedNatTree.contains FastRows.row3329) := by
    change Data.row3329.contains = (SortedNatTree.contains FastRows.row3329)
    exact FastRows.row3329_checked.symm
  rw [row]
  decide
theorem root48Step14 : scan edge (search edge 3) 4 [233] 233 root48Tail14 winner =
    scan edge (search edge 3) 4 [233] 233 root48Tail15 winner := by
  rw [root48Tail14, scan, if_neg (by decide), root48Candidates14_checked]
  change scan edge (search edge 3) 4 [233] 233 root48Tail15
    (if 3 ≤ root48Candidates14.length then search edge 3 [233, 3329] root48Candidates14 3562 winner else winner) = _
  rw [if_neg (by decide)]
def root48Candidates15 : List Nat := [7547, 11789, 16901]
set_option maxRecDepth 8192 in
theorem root48Candidates15_checked : candidates edge 4 233 3389 root48Tail16 winner.bound = root48Candidates15 := by
  unfold candidates
  have row : edge 3389 = (SortedNatTree.contains FastRows.row3389) := by
    change Data.row3389.contains = (SortedNatTree.contains FastRows.row3389)
    exact FastRows.row3389_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root48Child15 : search edge 3 [233, 3389] root48Candidates15 3622 winner = winner := by decide
theorem root48Step15 : scan edge (search edge 3) 4 [233] 233 root48Tail15 winner =
    scan edge (search edge 3) 4 [233] 233 root48Tail16 winner := by
  rw [root48Tail15, scan, if_neg (by decide), root48Candidates15_checked]
  change scan edge (search edge 3) 4 [233] 233 root48Tail16
    (if 3 ≤ root48Candidates15.length then search edge 3 [233, 3389] root48Candidates15 3622 winner else winner) = _
  rw [if_pos (by decide), root48Child15]
def root48Candidates16 : List Nat := [7043, 7547, 15473]
set_option maxRecDepth 8192 in
theorem root48Candidates16_checked : candidates edge 4 233 4337 root48Tail17 winner.bound = root48Candidates16 := by
  unfold candidates
  have row : edge 4337 = (SortedNatTree.contains FastRows.row4337) := by
    change Data.row4337.contains = (SortedNatTree.contains FastRows.row4337)
    exact FastRows.row4337_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root48Child16 : search edge 3 [233, 4337] root48Candidates16 4570 winner = winner := by decide
theorem root48Step16 : scan edge (search edge 3) 4 [233] 233 root48Tail16 winner =
    scan edge (search edge 3) 4 [233] 233 root48Tail17 winner := by
  rw [root48Tail16, scan, if_neg (by decide), root48Candidates16_checked]
  change scan edge (search edge 3) 4 [233] 233 root48Tail17
    (if 3 ≤ root48Candidates16.length then search edge 3 [233, 4337] root48Candidates16 4570 winner else winner) = _
  rw [if_pos (by decide), root48Child16]
def root48Candidates17 : List Nat := [8123, 16427, 20327]
set_option maxRecDepth 8192 in
theorem root48Candidates17_checked : candidates edge 4 233 4793 root48Tail18 winner.bound = root48Candidates17 := by
  unfold candidates
  have row : edge 4793 = (SortedNatTree.contains FastRows.row4793) := by
    change Data.row4793.contains = (SortedNatTree.contains FastRows.row4793)
    exact FastRows.row4793_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root48Child17 : search edge 3 [233, 4793] root48Candidates17 5026 winner = winner := by decide
theorem root48Step17 : scan edge (search edge 3) 4 [233] 233 root48Tail17 winner =
    scan edge (search edge 3) 4 [233] 233 root48Tail18 winner := by
  rw [root48Tail17, scan, if_neg (by decide), root48Candidates17_checked]
  change scan edge (search edge 3) 4 [233] 233 root48Tail18
    (if 3 ≤ root48Candidates17.length then search edge 3 [233, 4793] root48Candidates17 5026 winner else winner) = _
  rw [if_pos (by decide), root48Child17]
def root48Candidates18 : List Nat := [6947]
set_option maxRecDepth 8192 in
theorem root48Candidates18_checked : candidates edge 4 233 5009 root48Tail19 winner.bound = root48Candidates18 := by
  unfold candidates
  have row : edge 5009 = (SortedNatTree.contains FastRows.row5009) := by
    change Data.row5009.contains = (SortedNatTree.contains FastRows.row5009)
    exact FastRows.row5009_checked.symm
  rw [row]
  decide
theorem root48Step18 : scan edge (search edge 3) 4 [233] 233 root48Tail18 winner =
    scan edge (search edge 3) 4 [233] 233 root48Tail19 winner := by
  rw [root48Tail18, scan, if_neg (by decide), root48Candidates18_checked]
  change scan edge (search edge 3) 4 [233] 233 root48Tail19
    (if 3 ≤ root48Candidates18.length then search edge 3 [233, 5009] root48Candidates18 5242 winner else winner) = _
  rw [if_neg (by decide)]
def root48Candidates19 : List Nat := [6263]
set_option maxRecDepth 8192 in
theorem root48Candidates19_checked : candidates edge 4 233 5297 root48Tail20 winner.bound = root48Candidates19 := by
  unfold candidates
  have row : edge 5297 = (SortedNatTree.contains FastRows.row5297) := by
    change Data.row5297.contains = (SortedNatTree.contains FastRows.row5297)
    exact FastRows.row5297_checked.symm
  rw [row]
  decide
theorem root48Step19 : scan edge (search edge 3) 4 [233] 233 root48Tail19 winner =
    scan edge (search edge 3) 4 [233] 233 root48Tail20 winner := by
  rw [root48Tail19, scan, if_neg (by decide), root48Candidates19_checked]
  change scan edge (search edge 3) 4 [233] 233 root48Tail20
    (if 3 ≤ root48Candidates19.length then search edge 3 [233, 5297] root48Candidates19 5530 winner else winner) = _
  rw [if_neg (by decide)]
def root48Candidates20 : List Nat := [16829, 19013]
set_option maxRecDepth 8192 in
theorem root48Candidates20_checked : candidates edge 4 233 5591 root48Tail21 winner.bound = root48Candidates20 := by
  unfold candidates
  have row : edge 5591 = (SortedNatTree.contains FastRows.row5591) := by
    change Data.row5591.contains = (SortedNatTree.contains FastRows.row5591)
    exact FastRows.row5591_checked.symm
  rw [row]
  decide
theorem root48Step20 : scan edge (search edge 3) 4 [233] 233 root48Tail20 winner =
    scan edge (search edge 3) 4 [233] 233 root48Tail21 winner := by
  rw [root48Tail20, scan, if_neg (by decide), root48Candidates20_checked]
  change scan edge (search edge 3) 4 [233] 233 root48Tail21
    (if 3 ≤ root48Candidates20.length then search edge 3 [233, 5591] root48Candidates20 5824 winner else winner) = _
  rw [if_neg (by decide)]
def root48Candidates21 : List Nat := [7949, 13781]
set_option maxRecDepth 8192 in
theorem root48Candidates21_checked : candidates edge 4 233 5807 root48Tail22 winner.bound = root48Candidates21 := by
  unfold candidates
  have row : edge 5807 = (SortedNatTree.contains FastRows.row5807) := by
    change Data.row5807.contains = (SortedNatTree.contains FastRows.row5807)
    exact FastRows.row5807_checked.symm
  rw [row]
  decide
theorem root48Step21 : scan edge (search edge 3) 4 [233] 233 root48Tail21 winner =
    scan edge (search edge 3) 4 [233] 233 root48Tail22 winner := by
  rw [root48Tail21, scan, if_neg (by decide), root48Candidates21_checked]
  change scan edge (search edge 3) 4 [233] 233 root48Tail22
    (if 3 ≤ root48Candidates21.length then search edge 3 [233, 5807] root48Candidates21 6040 winner else winner) = _
  rw [if_neg (by decide)]
def root48Candidates22 : List Nat := [8867, 13751, 16481]
set_option maxRecDepth 8192 in
theorem root48Candidates22_checked : candidates edge 4 233 5813 root48Tail23 winner.bound = root48Candidates22 := by
  unfold candidates
  have row : edge 5813 = (SortedNatTree.contains FastRows.row5813) := by
    change Data.row5813.contains = (SortedNatTree.contains FastRows.row5813)
    exact FastRows.row5813_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root48Child22 : search edge 3 [233, 5813] root48Candidates22 6046 winner = winner := by decide
theorem root48Step22 : scan edge (search edge 3) 4 [233] 233 root48Tail22 winner =
    scan edge (search edge 3) 4 [233] 233 root48Tail23 winner := by
  rw [root48Tail22, scan, if_neg (by decide), root48Candidates22_checked]
  change scan edge (search edge 3) 4 [233] 233 root48Tail23
    (if 3 ≤ root48Candidates22.length then search edge 3 [233, 5813] root48Candidates22 6046 winner else winner) = _
  rw [if_pos (by decide), root48Child22]
def root48Candidates23 : List Nat := []
set_option maxRecDepth 8192 in
theorem root48Candidates23_checked : candidates edge 4 233 6011 root48Tail24 winner.bound = root48Candidates23 := by
  unfold candidates
  have row : edge 6011 = (SortedNatTree.contains FastRows.row6011) := by
    change Data.row6011.contains = (SortedNatTree.contains FastRows.row6011)
    exact FastRows.row6011_checked.symm
  rw [row]
  decide
theorem root48Step23 : scan edge (search edge 3) 4 [233] 233 root48Tail23 winner =
    scan edge (search edge 3) 4 [233] 233 root48Tail24 winner := by
  rw [root48Tail23, scan, if_neg (by decide), root48Candidates23_checked]
  change scan edge (search edge 3) 4 [233] 233 root48Tail24
    (if 3 ≤ root48Candidates23.length then search edge 3 [233, 6011] root48Candidates23 6244 winner else winner) = _
  rw [if_neg (by decide)]
def root48Candidates24 : List Nat := [6359, 11261, 14321, 14831]
set_option maxRecDepth 8192 in
theorem root48Candidates24_checked : candidates edge 4 233 6263 root48Tail25 winner.bound = root48Candidates24 := by
  unfold candidates
  have row : edge 6263 = (SortedNatTree.contains FastRows.row6263) := by
    change Data.row6263.contains = (SortedNatTree.contains FastRows.row6263)
    exact FastRows.row6263_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root48Child24 : search edge 3 [233, 6263] root48Candidates24 6496 winner = winner := by decide
theorem root48Step24 : scan edge (search edge 3) 4 [233] 233 root48Tail24 winner =
    scan edge (search edge 3) 4 [233] 233 root48Tail25 winner := by
  rw [root48Tail24, scan, if_neg (by decide), root48Candidates24_checked]
  change scan edge (search edge 3) 4 [233] 233 root48Tail25
    (if 3 ≤ root48Candidates24.length then search edge 3 [233, 6263] root48Candidates24 6496 winner else winner) = _
  rw [if_pos (by decide), root48Child24]
def root48Candidates25 : List Nat := [8123]
set_option maxRecDepth 8192 in
theorem root48Candidates25_checked : candidates edge 4 233 6359 root48Tail26 winner.bound = root48Candidates25 := by
  unfold candidates
  have row : edge 6359 = (SortedNatTree.contains FastRows.row6359) := by
    change Data.row6359.contains = (SortedNatTree.contains FastRows.row6359)
    exact FastRows.row6359_checked.symm
  rw [row]
  decide
theorem root48Step25 : scan edge (search edge 3) 4 [233] 233 root48Tail25 winner =
    scan edge (search edge 3) 4 [233] 233 root48Tail26 winner := by
  rw [root48Tail25, scan, if_neg (by decide), root48Candidates25_checked]
  change scan edge (search edge 3) 4 [233] 233 root48Tail26
    (if 3 ≤ root48Candidates25.length then search edge 3 [233, 6359] root48Candidates25 6592 winner else winner) = _
  rw [if_neg (by decide)]
theorem root48Scan26 : scan edge (search edge 3) 4 [233] 233 root48Tail26 winner = winner := by rw [root48Tail26, scan, if_pos (by decide)]
theorem root48Scan25 : scan edge (search edge 3) 4 [233] 233 root48Tail25 winner = winner :=
  (root48Step25).trans root48Scan26
theorem root48Scan24 : scan edge (search edge 3) 4 [233] 233 root48Tail24 winner = winner :=
  (root48Step24).trans root48Scan25
theorem root48Scan23 : scan edge (search edge 3) 4 [233] 233 root48Tail23 winner = winner :=
  (root48Step23).trans root48Scan24
theorem root48Scan22 : scan edge (search edge 3) 4 [233] 233 root48Tail22 winner = winner :=
  (root48Step22).trans root48Scan23
theorem root48Scan21 : scan edge (search edge 3) 4 [233] 233 root48Tail21 winner = winner :=
  (root48Step21).trans root48Scan22
theorem root48Scan20 : scan edge (search edge 3) 4 [233] 233 root48Tail20 winner = winner :=
  (root48Step20).trans root48Scan21
theorem root48Scan19 : scan edge (search edge 3) 4 [233] 233 root48Tail19 winner = winner :=
  (root48Step19).trans root48Scan20
theorem root48Scan18 : scan edge (search edge 3) 4 [233] 233 root48Tail18 winner = winner :=
  (root48Step18).trans root48Scan19
theorem root48Scan17 : scan edge (search edge 3) 4 [233] 233 root48Tail17 winner = winner :=
  (root48Step17).trans root48Scan18
theorem root48Scan16 : scan edge (search edge 3) 4 [233] 233 root48Tail16 winner = winner :=
  (root48Step16).trans root48Scan17
theorem root48Scan15 : scan edge (search edge 3) 4 [233] 233 root48Tail15 winner = winner :=
  (root48Step15).trans root48Scan16
theorem root48Scan14 : scan edge (search edge 3) 4 [233] 233 root48Tail14 winner = winner :=
  (root48Step14).trans root48Scan15
theorem root48Scan13 : scan edge (search edge 3) 4 [233] 233 root48Tail13 winner = winner :=
  (root48Step13).trans root48Scan14
theorem root48Scan12 : scan edge (search edge 3) 4 [233] 233 root48Tail12 winner = winner :=
  (root48Step12).trans root48Scan13
theorem root48Scan11 : scan edge (search edge 3) 4 [233] 233 root48Tail11 winner = winner :=
  (root48Step11).trans root48Scan12
theorem root48Scan10 : scan edge (search edge 3) 4 [233] 233 root48Tail10 winner = winner :=
  (root48Step10).trans root48Scan11
theorem root48Scan9 : scan edge (search edge 3) 4 [233] 233 root48Tail9 winner = winner :=
  (root48Step9).trans root48Scan10
theorem root48Scan8 : scan edge (search edge 3) 4 [233] 233 root48Tail8 winner = winner :=
  (root48Step8).trans root48Scan9
theorem root48Scan7 : scan edge (search edge 3) 4 [233] 233 root48Tail7 winner = winner :=
  (root48Step7).trans root48Scan8
theorem root48Scan6 : scan edge (search edge 3) 4 [233] 233 root48Tail6 winner = winner :=
  (root48Step6).trans root48Scan7
theorem root48Scan5 : scan edge (search edge 3) 4 [233] 233 root48Tail5 winner = winner :=
  (root48Step5).trans root48Scan6
theorem root48Scan4 : scan edge (search edge 3) 4 [233] 233 root48Tail4 winner = winner :=
  (root48Step4).trans root48Scan5
theorem root48Scan3 : scan edge (search edge 3) 4 [233] 233 root48Tail3 winner = winner :=
  (root48Step3).trans root48Scan4
theorem root48Scan2 : scan edge (search edge 3) 4 [233] 233 root48Tail2 winner = winner :=
  (root48Step2).trans root48Scan3
theorem root48Scan1 : scan edge (search edge 3) 4 [233] 233 root48Tail1 winner = winner :=
  (root48Step1).trans root48Scan2
theorem root48Scan0 : scan edge (search edge 3) 4 [233] 233 root48Tail0 winner = winner :=
  (root48Step0).trans root48Scan1
set_option maxRecDepth 4096 in
theorem root48Branch : search edge 4 [233] root48Candidates 233 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [233] 233 root48Tail0 winner = winner
  exact root48Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
