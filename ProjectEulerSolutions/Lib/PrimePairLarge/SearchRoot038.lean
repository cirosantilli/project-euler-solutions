import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root38Candidates : List Nat := Data.row179
set_option maxRecDepth 8192 in
theorem root38Candidates_checked : candidates edge 5 0 179 tail39 winner.bound = root38Candidates := by
  apply candidates_eq_row edge 5 0 179 tail39 root38Candidates winner.bound tail39_ordered
  · decide
  · decide
  · funext q
    rfl
def root38Tail76 : List Nat := []
def root38Tail75 : List Nat := 25679 :: root38Tail76
def root38Tail74 : List Nat := 25343 :: root38Tail75
def root38Tail73 : List Nat := 25307 :: root38Tail74
def root38Tail72 : List Nat := 24677 :: root38Tail73
def root38Tail71 : List Nat := 24509 :: root38Tail72
def root38Tail70 : List Nat := 24251 :: root38Tail71
def root38Tail69 : List Nat := 23753 :: root38Tail70
def root38Tail68 : List Nat := 23663 :: root38Tail69
def root38Tail67 : List Nat := 23567 :: root38Tail68
def root38Tail66 : List Nat := 23159 :: root38Tail67
def root38Tail65 : List Nat := 22571 :: root38Tail66
def root38Tail64 : List Nat := 22343 :: root38Tail65
def root38Tail63 : List Nat := 21803 :: root38Tail64
def root38Tail62 : List Nat := 21713 :: root38Tail63
def root38Tail61 : List Nat := 21647 :: root38Tail62
def root38Tail60 : List Nat := 20117 :: root38Tail61
def root38Tail59 : List Nat := 20051 :: root38Tail60
def root38Tail58 : List Nat := 19763 :: root38Tail59
def root38Tail57 : List Nat := 18521 :: root38Tail58
def root38Tail56 : List Nat := 18269 :: root38Tail57
def root38Tail55 : List Nat := 17117 :: root38Tail56
def root38Tail54 : List Nat := 16889 :: root38Tail55
def root38Tail53 : List Nat := 16649 :: root38Tail54
def root38Tail52 : List Nat := 16553 :: root38Tail53
def root38Tail51 : List Nat := 16421 :: root38Tail52
def root38Tail50 : List Nat := 16319 :: root38Tail51
def root38Tail49 : List Nat := 16193 :: root38Tail50
def root38Tail48 : List Nat := 15797 :: root38Tail49
def root38Tail47 : List Nat := 15767 :: root38Tail48
def root38Tail46 : List Nat := 14087 :: root38Tail47
def root38Tail45 : List Nat := 13613 :: root38Tail46
def root38Tail44 : List Nat := 12641 :: root38Tail45
def root38Tail43 : List Nat := 11867 :: root38Tail44
def root38Tail42 : List Nat := 10733 :: root38Tail43
def root38Tail41 : List Nat := 10463 :: root38Tail42
def root38Tail40 : List Nat := 10457 :: root38Tail41
def root38Tail39 : List Nat := 10223 :: root38Tail40
def root38Tail38 : List Nat := 9887 :: root38Tail39
def root38Tail37 : List Nat := 9803 :: root38Tail38
def root38Tail36 : List Nat := 9533 :: root38Tail37
def root38Tail35 : List Nat := 9311 :: root38Tail36
def root38Tail34 : List Nat := 9137 :: root38Tail35
def root38Tail33 : List Nat := 8009 :: root38Tail34
def root38Tail32 : List Nat := 7877 :: root38Tail33
def root38Tail31 : List Nat := 7673 :: root38Tail32
def root38Tail30 : List Nat := 7547 :: root38Tail31
def root38Tail29 : List Nat := 7541 :: root38Tail30
def root38Tail28 : List Nat := 7331 :: root38Tail29
def root38Tail27 : List Nat := 5867 :: root38Tail28
def root38Tail26 : List Nat := 5279 :: root38Tail27
def root38Tail25 : List Nat := 5153 :: root38Tail26
def root38Tail24 : List Nat := 4973 :: root38Tail25
def root38Tail23 : List Nat := 4817 :: root38Tail24
def root38Tail22 : List Nat := 4787 :: root38Tail23
def root38Tail21 : List Nat := 4649 :: root38Tail22
def root38Tail20 : List Nat := 4517 :: root38Tail21
def root38Tail19 : List Nat := 4349 :: root38Tail20
def root38Tail18 : List Nat := 3719 :: root38Tail19
def root38Tail17 : List Nat := 3137 :: root38Tail18
def root38Tail16 : List Nat := 2579 :: root38Tail17
def root38Tail15 : List Nat := 2381 :: root38Tail16
def root38Tail14 : List Nat := 2339 :: root38Tail15
def root38Tail13 : List Nat := 2309 :: root38Tail14
def root38Tail12 : List Nat := 2039 :: root38Tail13
def root38Tail11 : List Nat := 1847 :: root38Tail12
def root38Tail10 : List Nat := 1787 :: root38Tail11
def root38Tail9 : List Nat := 1487 :: root38Tail10
def root38Tail8 : List Nat := 1451 :: root38Tail9
def root38Tail7 : List Nat := 1289 :: root38Tail8
def root38Tail6 : List Nat := 953 :: root38Tail7
def root38Tail5 : List Nat := 743 :: root38Tail6
def root38Tail4 : List Nat := 719 :: root38Tail5
def root38Tail3 : List Nat := 593 :: root38Tail4
def root38Tail2 : List Nat := 383 :: root38Tail3
def root38Tail1 : List Nat := 317 :: root38Tail2
def root38Tail0 : List Nat := 269 :: root38Tail1
def root38Candidates0 : List Nat := [317, 16193]
set_option maxRecDepth 8192 in
theorem root38Candidates0_checked : candidates edge 4 179 269 root38Tail1 winner.bound = root38Candidates0 := by
  unfold candidates
  have row : edge 269 = (SortedNatTree.contains FastRows.row269) := by
    change Data.row269.contains = (SortedNatTree.contains FastRows.row269)
    exact FastRows.row269_checked.symm
  rw [row]
  decide
theorem root38Step0 : scan edge (search edge 3) 4 [179] 179 root38Tail0 winner =
    scan edge (search edge 3) 4 [179] 179 root38Tail1 winner := by
  rw [root38Tail0, scan, if_neg (by decide), root38Candidates0_checked]
  change scan edge (search edge 3) 4 [179] 179 root38Tail1
    (if 3 ≤ root38Candidates0.length then search edge 3 [179, 269] root38Candidates0 448 winner else winner) = _
  rw [if_neg (by decide)]
def root38Candidates1 : List Nat := [4973, 16649, 23663]
set_option maxRecDepth 8192 in
theorem root38Candidates1_checked : candidates edge 4 179 317 root38Tail2 winner.bound = root38Candidates1 := by
  unfold candidates
  have row : edge 317 = (SortedNatTree.contains FastRows.row317) := by
    change Data.row317.contains = (SortedNatTree.contains FastRows.row317)
    exact FastRows.row317_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root38Child1 : search edge 3 [179, 317] root38Candidates1 496 winner = winner := by decide
theorem root38Step1 : scan edge (search edge 3) 4 [179] 179 root38Tail1 winner =
    scan edge (search edge 3) 4 [179] 179 root38Tail2 winner := by
  rw [root38Tail1, scan, if_neg (by decide), root38Candidates1_checked]
  change scan edge (search edge 3) 4 [179] 179 root38Tail2
    (if 3 ≤ root38Candidates1.length then search edge 3 [179, 317] root38Candidates1 496 winner else winner) = _
  rw [if_pos (by decide), root38Child1]
def root38Candidates2 : List Nat := [7331, 7877]
set_option maxRecDepth 8192 in
theorem root38Candidates2_checked : candidates edge 4 179 383 root38Tail3 winner.bound = root38Candidates2 := by
  unfold candidates
  have row : edge 383 = (SortedNatTree.contains FastRows.row383) := by
    change Data.row383.contains = (SortedNatTree.contains FastRows.row383)
    exact FastRows.row383_checked.symm
  rw [row]
  decide
theorem root38Step2 : scan edge (search edge 3) 4 [179] 179 root38Tail2 winner =
    scan edge (search edge 3) 4 [179] 179 root38Tail3 winner := by
  rw [root38Tail2, scan, if_neg (by decide), root38Candidates2_checked]
  change scan edge (search edge 3) 4 [179] 179 root38Tail3
    (if 3 ≤ root38Candidates2.length then search edge 3 [179, 383] root38Candidates2 562 winner else winner) = _
  rw [if_neg (by decide)]
def root38Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root38Candidates3_checked : candidates edge 4 179 593 root38Tail4 winner.bound = root38Candidates3 := by
  unfold candidates
  have row : edge 593 = (SortedNatTree.contains FastRows.row593) := by
    change Data.row593.contains = (SortedNatTree.contains FastRows.row593)
    exact FastRows.row593_checked.symm
  rw [row]
  decide
theorem root38Step3 : scan edge (search edge 3) 4 [179] 179 root38Tail3 winner =
    scan edge (search edge 3) 4 [179] 179 root38Tail4 winner := by
  rw [root38Tail3, scan, if_neg (by decide), root38Candidates3_checked]
  change scan edge (search edge 3) 4 [179] 179 root38Tail4
    (if 3 ≤ root38Candidates3.length then search edge 3 [179, 593] root38Candidates3 772 winner else winner) = _
  rw [if_neg (by decide)]
def root38Candidates4 : List Nat := [2579, 4973, 19763, 20117]
set_option maxRecDepth 8192 in
theorem root38Candidates4_checked : candidates edge 4 179 719 root38Tail5 winner.bound = root38Candidates4 := by
  unfold candidates
  have row : edge 719 = (SortedNatTree.contains FastRows.row719) := by
    change Data.row719.contains = (SortedNatTree.contains FastRows.row719)
    exact FastRows.row719_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root38Child4 : search edge 3 [179, 719] root38Candidates4 898 winner = winner := by decide
theorem root38Step4 : scan edge (search edge 3) 4 [179] 179 root38Tail4 winner =
    scan edge (search edge 3) 4 [179] 179 root38Tail5 winner := by
  rw [root38Tail4, scan, if_neg (by decide), root38Candidates4_checked]
  change scan edge (search edge 3) 4 [179] 179 root38Tail5
    (if 3 ≤ root38Candidates4.length then search edge 3 [179, 719] root38Candidates4 898 winner else winner) = _
  rw [if_pos (by decide), root38Child4]
def root38Candidates5 : List Nat := [2309, 14087, 22571, 23663]
set_option maxRecDepth 8192 in
theorem root38Candidates5_checked : candidates edge 4 179 743 root38Tail6 winner.bound = root38Candidates5 := by
  unfold candidates
  have row : edge 743 = (SortedNatTree.contains FastRows.row743) := by
    change Data.row743.contains = (SortedNatTree.contains FastRows.row743)
    exact FastRows.row743_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root38Child5 : search edge 3 [179, 743] root38Candidates5 922 winner = winner := by decide
theorem root38Step5 : scan edge (search edge 3) 4 [179] 179 root38Tail5 winner =
    scan edge (search edge 3) 4 [179] 179 root38Tail6 winner := by
  rw [root38Tail5, scan, if_neg (by decide), root38Candidates5_checked]
  change scan edge (search edge 3) 4 [179] 179 root38Tail6
    (if 3 ≤ root38Candidates5.length then search edge 3 [179, 743] root38Candidates5 922 winner else winner) = _
  rw [if_pos (by decide), root38Child5]
def root38Candidates6 : List Nat := [10463, 16889, 20051]
set_option maxRecDepth 8192 in
theorem root38Candidates6_checked : candidates edge 4 179 953 root38Tail7 winner.bound = root38Candidates6 := by
  unfold candidates
  have row : edge 953 = (SortedNatTree.contains FastRows.row953) := by
    change Data.row953.contains = (SortedNatTree.contains FastRows.row953)
    exact FastRows.row953_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root38Child6 : search edge 3 [179, 953] root38Candidates6 1132 winner = winner := by decide
theorem root38Step6 : scan edge (search edge 3) 4 [179] 179 root38Tail6 winner =
    scan edge (search edge 3) 4 [179] 179 root38Tail7 winner := by
  rw [root38Tail6, scan, if_neg (by decide), root38Candidates6_checked]
  change scan edge (search edge 3) 4 [179] 179 root38Tail7
    (if 3 ≤ root38Candidates6.length then search edge 3 [179, 953] root38Candidates6 1132 winner else winner) = _
  rw [if_pos (by decide), root38Child6]
def root38Candidates7 : List Nat := [2039, 7541, 22571]
set_option maxRecDepth 8192 in
theorem root38Candidates7_checked : candidates edge 4 179 1289 root38Tail8 winner.bound = root38Candidates7 := by
  unfold candidates
  have row : edge 1289 = (SortedNatTree.contains FastRows.row1289) := by
    change Data.row1289.contains = (SortedNatTree.contains FastRows.row1289)
    exact FastRows.row1289_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root38Child7 : search edge 3 [179, 1289] root38Candidates7 1468 winner = winner := by decide
theorem root38Step7 : scan edge (search edge 3) 4 [179] 179 root38Tail7 winner =
    scan edge (search edge 3) 4 [179] 179 root38Tail8 winner := by
  rw [root38Tail7, scan, if_neg (by decide), root38Candidates7_checked]
  change scan edge (search edge 3) 4 [179] 179 root38Tail8
    (if 3 ≤ root38Candidates7.length then search edge 3 [179, 1289] root38Candidates7 1468 winner else winner) = _
  rw [if_pos (by decide), root38Child7]
def root38Candidates8 : List Nat := [2039, 10463, 18521]
set_option maxRecDepth 8192 in
theorem root38Candidates8_checked : candidates edge 4 179 1451 root38Tail9 winner.bound = root38Candidates8 := by
  unfold candidates
  have row : edge 1451 = (SortedNatTree.contains FastRows.row1451) := by
    change Data.row1451.contains = (SortedNatTree.contains FastRows.row1451)
    exact FastRows.row1451_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root38Child8 : search edge 3 [179, 1451] root38Candidates8 1630 winner = winner := by decide
theorem root38Step8 : scan edge (search edge 3) 4 [179] 179 root38Tail8 winner =
    scan edge (search edge 3) 4 [179] 179 root38Tail9 winner := by
  rw [root38Tail8, scan, if_neg (by decide), root38Candidates8_checked]
  change scan edge (search edge 3) 4 [179] 179 root38Tail9
    (if 3 ≤ root38Candidates8.length then search edge 3 [179, 1451] root38Candidates8 1630 winner else winner) = _
  rw [if_pos (by decide), root38Child8]
def root38Candidates9 : List Nat := [16193, 16649]
set_option maxRecDepth 8192 in
theorem root38Candidates9_checked : candidates edge 4 179 1487 root38Tail10 winner.bound = root38Candidates9 := by
  unfold candidates
  have row : edge 1487 = (SortedNatTree.contains FastRows.row1487) := by
    change Data.row1487.contains = (SortedNatTree.contains FastRows.row1487)
    exact FastRows.row1487_checked.symm
  rw [row]
  decide
theorem root38Step9 : scan edge (search edge 3) 4 [179] 179 root38Tail9 winner =
    scan edge (search edge 3) 4 [179] 179 root38Tail10 winner := by
  rw [root38Tail9, scan, if_neg (by decide), root38Candidates9_checked]
  change scan edge (search edge 3) 4 [179] 179 root38Tail10
    (if 3 ≤ root38Candidates9.length then search edge 3 [179, 1487] root38Candidates9 1666 winner else winner) = _
  rw [if_neg (by decide)]
def root38Candidates10 : List Nat := [4349, 7547, 10457, 10733]
set_option maxRecDepth 8192 in
theorem root38Candidates10_checked : candidates edge 4 179 1787 root38Tail11 winner.bound = root38Candidates10 := by
  unfold candidates
  have row : edge 1787 = (SortedNatTree.contains FastRows.row1787) := by
    change Data.row1787.contains = (SortedNatTree.contains FastRows.row1787)
    exact FastRows.row1787_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root38Child10 : search edge 3 [179, 1787] root38Candidates10 1966 winner = winner := by decide
theorem root38Step10 : scan edge (search edge 3) 4 [179] 179 root38Tail10 winner =
    scan edge (search edge 3) 4 [179] 179 root38Tail11 winner := by
  rw [root38Tail10, scan, if_neg (by decide), root38Candidates10_checked]
  change scan edge (search edge 3) 4 [179] 179 root38Tail11
    (if 3 ≤ root38Candidates10.length then search edge 3 [179, 1787] root38Candidates10 1966 winner else winner) = _
  rw [if_pos (by decide), root38Child10]
def root38Candidates11 : List Nat := [9311, 15767, 20051, 23663]
set_option maxRecDepth 8192 in
theorem root38Candidates11_checked : candidates edge 4 179 1847 root38Tail12 winner.bound = root38Candidates11 := by
  unfold candidates
  have row : edge 1847 = (SortedNatTree.contains FastRows.row1847) := by
    change Data.row1847.contains = (SortedNatTree.contains FastRows.row1847)
    exact FastRows.row1847_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root38Child11 : search edge 3 [179, 1847] root38Candidates11 2026 winner = winner := by decide
theorem root38Step11 : scan edge (search edge 3) 4 [179] 179 root38Tail11 winner =
    scan edge (search edge 3) 4 [179] 179 root38Tail12 winner := by
  rw [root38Tail11, scan, if_neg (by decide), root38Candidates11_checked]
  change scan edge (search edge 3) 4 [179] 179 root38Tail12
    (if 3 ≤ root38Candidates11.length then search edge 3 [179, 1847] root38Candidates11 2026 winner else winner) = _
  rw [if_pos (by decide), root38Child11]
def root38Candidates12 : List Nat := [7673, 10463, 16553, 18269]
set_option maxRecDepth 8192 in
theorem root38Candidates12_checked : candidates edge 4 179 2039 root38Tail13 winner.bound = root38Candidates12 := by
  unfold candidates
  have row : edge 2039 = (SortedNatTree.contains FastRows.row2039) := by
    change Data.row2039.contains = (SortedNatTree.contains FastRows.row2039)
    exact FastRows.row2039_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root38Child12 : search edge 3 [179, 2039] root38Candidates12 2218 winner = winner := by decide
theorem root38Step12 : scan edge (search edge 3) 4 [179] 179 root38Tail12 winner =
    scan edge (search edge 3) 4 [179] 179 root38Tail13 winner := by
  rw [root38Tail12, scan, if_neg (by decide), root38Candidates12_checked]
  change scan edge (search edge 3) 4 [179] 179 root38Tail13
    (if 3 ≤ root38Candidates12.length then search edge 3 [179, 2039] root38Candidates12 2218 winner else winner) = _
  rw [if_pos (by decide), root38Child12]
def root38Candidates13 : List Nat := [9887]
set_option maxRecDepth 8192 in
theorem root38Candidates13_checked : candidates edge 4 179 2309 root38Tail14 winner.bound = root38Candidates13 := by
  unfold candidates
  have row : edge 2309 = (SortedNatTree.contains FastRows.row2309) := by
    change Data.row2309.contains = (SortedNatTree.contains FastRows.row2309)
    exact FastRows.row2309_checked.symm
  rw [row]
  decide
theorem root38Step13 : scan edge (search edge 3) 4 [179] 179 root38Tail13 winner =
    scan edge (search edge 3) 4 [179] 179 root38Tail14 winner := by
  rw [root38Tail13, scan, if_neg (by decide), root38Candidates13_checked]
  change scan edge (search edge 3) 4 [179] 179 root38Tail14
    (if 3 ≤ root38Candidates13.length then search edge 3 [179, 2309] root38Candidates13 2488 winner else winner) = _
  rw [if_neg (by decide)]
def root38Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root38Candidates14_checked : candidates edge 4 179 2339 root38Tail15 winner.bound = root38Candidates14 := by
  unfold candidates
  have row : edge 2339 = (SortedNatTree.contains FastRows.row2339) := by
    change Data.row2339.contains = (SortedNatTree.contains FastRows.row2339)
    exact FastRows.row2339_checked.symm
  rw [row]
  decide
theorem root38Step14 : scan edge (search edge 3) 4 [179] 179 root38Tail14 winner =
    scan edge (search edge 3) 4 [179] 179 root38Tail15 winner := by
  rw [root38Tail14, scan, if_neg (by decide), root38Candidates14_checked]
  change scan edge (search edge 3) 4 [179] 179 root38Tail15
    (if 3 ≤ root38Candidates14.length then search edge 3 [179, 2339] root38Candidates14 2518 winner else winner) = _
  rw [if_neg (by decide)]
def root38Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root38Candidates15_checked : candidates edge 4 179 2381 root38Tail16 winner.bound = root38Candidates15 := by
  unfold candidates
  have row : edge 2381 = (SortedNatTree.contains FastRows.row2381) := by
    change Data.row2381.contains = (SortedNatTree.contains FastRows.row2381)
    exact FastRows.row2381_checked.symm
  rw [row]
  decide
theorem root38Step15 : scan edge (search edge 3) 4 [179] 179 root38Tail15 winner =
    scan edge (search edge 3) 4 [179] 179 root38Tail16 winner := by
  rw [root38Tail15, scan, if_neg (by decide), root38Candidates15_checked]
  change scan edge (search edge 3) 4 [179] 179 root38Tail16
    (if 3 ≤ root38Candidates15.length then search edge 3 [179, 2381] root38Candidates15 2560 winner else winner) = _
  rw [if_neg (by decide)]
def root38Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root38Candidates16_checked : candidates edge 4 179 2579 root38Tail17 winner.bound = root38Candidates16 := by
  unfold candidates
  have row : edge 2579 = (SortedNatTree.contains FastRows.row2579) := by
    change Data.row2579.contains = (SortedNatTree.contains FastRows.row2579)
    exact FastRows.row2579_checked.symm
  rw [row]
  decide
theorem root38Step16 : scan edge (search edge 3) 4 [179] 179 root38Tail16 winner =
    scan edge (search edge 3) 4 [179] 179 root38Tail17 winner := by
  rw [root38Tail16, scan, if_neg (by decide), root38Candidates16_checked]
  change scan edge (search edge 3) 4 [179] 179 root38Tail17
    (if 3 ≤ root38Candidates16.length then search edge 3 [179, 2579] root38Candidates16 2758 winner else winner) = _
  rw [if_neg (by decide)]
def root38Candidates17 : List Nat := [4349, 7331]
set_option maxRecDepth 8192 in
theorem root38Candidates17_checked : candidates edge 4 179 3137 root38Tail18 winner.bound = root38Candidates17 := by
  unfold candidates
  have row : edge 3137 = (SortedNatTree.contains FastRows.row3137) := by
    change Data.row3137.contains = (SortedNatTree.contains FastRows.row3137)
    exact FastRows.row3137_checked.symm
  rw [row]
  decide
theorem root38Step17 : scan edge (search edge 3) 4 [179] 179 root38Tail17 winner =
    scan edge (search edge 3) 4 [179] 179 root38Tail18 winner := by
  rw [root38Tail17, scan, if_neg (by decide), root38Candidates17_checked]
  change scan edge (search edge 3) 4 [179] 179 root38Tail18
    (if 3 ≤ root38Candidates17.length then search edge 3 [179, 3137] root38Candidates17 3316 winner else winner) = _
  rw [if_neg (by decide)]
def root38Candidates18 : List Nat := []
set_option maxRecDepth 8192 in
theorem root38Candidates18_checked : candidates edge 4 179 3719 root38Tail19 winner.bound = root38Candidates18 := by
  unfold candidates
  have row : edge 3719 = (SortedNatTree.contains FastRows.row3719) := by
    change Data.row3719.contains = (SortedNatTree.contains FastRows.row3719)
    exact FastRows.row3719_checked.symm
  rw [row]
  decide
theorem root38Step18 : scan edge (search edge 3) 4 [179] 179 root38Tail18 winner =
    scan edge (search edge 3) 4 [179] 179 root38Tail19 winner := by
  rw [root38Tail18, scan, if_neg (by decide), root38Candidates18_checked]
  change scan edge (search edge 3) 4 [179] 179 root38Tail19
    (if 3 ≤ root38Candidates18.length then search edge 3 [179, 3719] root38Candidates18 3898 winner else winner) = _
  rw [if_neg (by decide)]
def root38Candidates19 : List Nat := []
set_option maxRecDepth 8192 in
theorem root38Candidates19_checked : candidates edge 4 179 4349 root38Tail20 winner.bound = root38Candidates19 := by
  unfold candidates
  have row : edge 4349 = (SortedNatTree.contains FastRows.row4349) := by
    change Data.row4349.contains = (SortedNatTree.contains FastRows.row4349)
    exact FastRows.row4349_checked.symm
  rw [row]
  decide
theorem root38Step19 : scan edge (search edge 3) 4 [179] 179 root38Tail19 winner =
    scan edge (search edge 3) 4 [179] 179 root38Tail20 winner := by
  rw [root38Tail19, scan, if_neg (by decide), root38Candidates19_checked]
  change scan edge (search edge 3) 4 [179] 179 root38Tail20
    (if 3 ≤ root38Candidates19.length then search edge 3 [179, 4349] root38Candidates19 4528 winner else winner) = _
  rw [if_neg (by decide)]
def root38Candidates20 : List Nat := [5279]
set_option maxRecDepth 8192 in
theorem root38Candidates20_checked : candidates edge 4 179 4517 root38Tail21 winner.bound = root38Candidates20 := by
  unfold candidates
  have row : edge 4517 = (SortedNatTree.contains FastRows.row4517) := by
    change Data.row4517.contains = (SortedNatTree.contains FastRows.row4517)
    exact FastRows.row4517_checked.symm
  rw [row]
  decide
theorem root38Step20 : scan edge (search edge 3) 4 [179] 179 root38Tail20 winner =
    scan edge (search edge 3) 4 [179] 179 root38Tail21 winner := by
  rw [root38Tail20, scan, if_neg (by decide), root38Candidates20_checked]
  change scan edge (search edge 3) 4 [179] 179 root38Tail21
    (if 3 ≤ root38Candidates20.length then search edge 3 [179, 4517] root38Candidates20 4696 winner else winner) = _
  rw [if_neg (by decide)]
def root38Candidates21 : List Nat := [12641]
set_option maxRecDepth 8192 in
theorem root38Candidates21_checked : candidates edge 4 179 4649 root38Tail22 winner.bound = root38Candidates21 := by
  unfold candidates
  have row : edge 4649 = (SortedNatTree.contains FastRows.row4649) := by
    change Data.row4649.contains = (SortedNatTree.contains FastRows.row4649)
    exact FastRows.row4649_checked.symm
  rw [row]
  decide
theorem root38Step21 : scan edge (search edge 3) 4 [179] 179 root38Tail21 winner =
    scan edge (search edge 3) 4 [179] 179 root38Tail22 winner := by
  rw [root38Tail21, scan, if_neg (by decide), root38Candidates21_checked]
  change scan edge (search edge 3) 4 [179] 179 root38Tail22
    (if 3 ≤ root38Candidates21.length then search edge 3 [179, 4649] root38Candidates21 4828 winner else winner) = _
  rw [if_neg (by decide)]
def root38Candidates22 : List Nat := [8009, 10457, 14087]
set_option maxRecDepth 8192 in
theorem root38Candidates22_checked : candidates edge 4 179 4787 root38Tail23 winner.bound = root38Candidates22 := by
  unfold candidates
  have row : edge 4787 = (SortedNatTree.contains FastRows.row4787) := by
    change Data.row4787.contains = (SortedNatTree.contains FastRows.row4787)
    exact FastRows.row4787_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root38Child22 : search edge 3 [179, 4787] root38Candidates22 4966 winner = winner := by decide
theorem root38Step22 : scan edge (search edge 3) 4 [179] 179 root38Tail22 winner =
    scan edge (search edge 3) 4 [179] 179 root38Tail23 winner := by
  rw [root38Tail22, scan, if_neg (by decide), root38Candidates22_checked]
  change scan edge (search edge 3) 4 [179] 179 root38Tail23
    (if 3 ≤ root38Candidates22.length then search edge 3 [179, 4787] root38Candidates22 4966 winner else winner) = _
  rw [if_pos (by decide), root38Child22]
def root38Candidates23 : List Nat := [18521]
set_option maxRecDepth 8192 in
theorem root38Candidates23_checked : candidates edge 4 179 4817 root38Tail24 winner.bound = root38Candidates23 := by
  unfold candidates
  have row : edge 4817 = (SortedNatTree.contains FastRows.row4817) := by
    change Data.row4817.contains = (SortedNatTree.contains FastRows.row4817)
    exact FastRows.row4817_checked.symm
  rw [row]
  decide
theorem root38Step23 : scan edge (search edge 3) 4 [179] 179 root38Tail23 winner =
    scan edge (search edge 3) 4 [179] 179 root38Tail24 winner := by
  rw [root38Tail23, scan, if_neg (by decide), root38Candidates23_checked]
  change scan edge (search edge 3) 4 [179] 179 root38Tail24
    (if 3 ≤ root38Candidates23.length then search edge 3 [179, 4817] root38Candidates23 4996 winner else winner) = _
  rw [if_neg (by decide)]
def root38Candidates24 : List Nat := [10223, 14087, 16319, 20051]
set_option maxRecDepth 8192 in
theorem root38Candidates24_checked : candidates edge 4 179 4973 root38Tail25 winner.bound = root38Candidates24 := by
  unfold candidates
  have row : edge 4973 = (SortedNatTree.contains FastRows.row4973) := by
    change Data.row4973.contains = (SortedNatTree.contains FastRows.row4973)
    exact FastRows.row4973_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root38Child24 : search edge 3 [179, 4973] root38Candidates24 5152 winner = winner := by decide
theorem root38Step24 : scan edge (search edge 3) 4 [179] 179 root38Tail24 winner =
    scan edge (search edge 3) 4 [179] 179 root38Tail25 winner := by
  rw [root38Tail24, scan, if_neg (by decide), root38Candidates24_checked]
  change scan edge (search edge 3) 4 [179] 179 root38Tail25
    (if 3 ≤ root38Candidates24.length then search edge 3 [179, 4973] root38Candidates24 5152 winner else winner) = _
  rw [if_pos (by decide), root38Child24]
def root38Candidates25 : List Nat := [5867, 9311]
set_option maxRecDepth 8192 in
theorem root38Candidates25_checked : candidates edge 4 179 5153 root38Tail26 winner.bound = root38Candidates25 := by
  unfold candidates
  have row : edge 5153 = (SortedNatTree.contains FastRows.row5153) := by
    change Data.row5153.contains = (SortedNatTree.contains FastRows.row5153)
    exact FastRows.row5153_checked.symm
  rw [row]
  decide
theorem root38Step25 : scan edge (search edge 3) 4 [179] 179 root38Tail25 winner =
    scan edge (search edge 3) 4 [179] 179 root38Tail26 winner := by
  rw [root38Tail25, scan, if_neg (by decide), root38Candidates25_checked]
  change scan edge (search edge 3) 4 [179] 179 root38Tail26
    (if 3 ≤ root38Candidates25.length then search edge 3 [179, 5153] root38Candidates25 5332 winner else winner) = _
  rw [if_neg (by decide)]
def root38Candidates26 : List Nat := []
set_option maxRecDepth 8192 in
theorem root38Candidates26_checked : candidates edge 4 179 5279 root38Tail27 winner.bound = root38Candidates26 := by
  unfold candidates
  have row : edge 5279 = (SortedNatTree.contains FastRows.row5279) := by
    change Data.row5279.contains = (SortedNatTree.contains FastRows.row5279)
    exact FastRows.row5279_checked.symm
  rw [row]
  decide
theorem root38Step26 : scan edge (search edge 3) 4 [179] 179 root38Tail26 winner =
    scan edge (search edge 3) 4 [179] 179 root38Tail27 winner := by
  rw [root38Tail26, scan, if_neg (by decide), root38Candidates26_checked]
  change scan edge (search edge 3) 4 [179] 179 root38Tail27
    (if 3 ≤ root38Candidates26.length then search edge 3 [179, 5279] root38Candidates26 5458 winner else winner) = _
  rw [if_neg (by decide)]
def root38Candidates27 : List Nat := [18269]
set_option maxRecDepth 8192 in
theorem root38Candidates27_checked : candidates edge 4 179 5867 root38Tail28 winner.bound = root38Candidates27 := by
  unfold candidates
  have row : edge 5867 = (SortedNatTree.contains FastRows.row5867) := by
    change Data.row5867.contains = (SortedNatTree.contains FastRows.row5867)
    exact FastRows.row5867_checked.symm
  rw [row]
  decide
theorem root38Step27 : scan edge (search edge 3) 4 [179] 179 root38Tail27 winner =
    scan edge (search edge 3) 4 [179] 179 root38Tail28 winner := by
  rw [root38Tail27, scan, if_neg (by decide), root38Candidates27_checked]
  change scan edge (search edge 3) 4 [179] 179 root38Tail28
    (if 3 ≤ root38Candidates27.length then search edge 3 [179, 5867] root38Candidates27 6046 winner else winner) = _
  rw [if_neg (by decide)]
theorem root38Scan28 : scan edge (search edge 3) 4 [179] 179 root38Tail28 winner = winner := by rw [root38Tail28, scan, if_pos (by decide)]
theorem root38Scan27 : scan edge (search edge 3) 4 [179] 179 root38Tail27 winner = winner :=
  (root38Step27).trans root38Scan28
theorem root38Scan26 : scan edge (search edge 3) 4 [179] 179 root38Tail26 winner = winner :=
  (root38Step26).trans root38Scan27
theorem root38Scan25 : scan edge (search edge 3) 4 [179] 179 root38Tail25 winner = winner :=
  (root38Step25).trans root38Scan26
theorem root38Scan24 : scan edge (search edge 3) 4 [179] 179 root38Tail24 winner = winner :=
  (root38Step24).trans root38Scan25
theorem root38Scan23 : scan edge (search edge 3) 4 [179] 179 root38Tail23 winner = winner :=
  (root38Step23).trans root38Scan24
theorem root38Scan22 : scan edge (search edge 3) 4 [179] 179 root38Tail22 winner = winner :=
  (root38Step22).trans root38Scan23
theorem root38Scan21 : scan edge (search edge 3) 4 [179] 179 root38Tail21 winner = winner :=
  (root38Step21).trans root38Scan22
theorem root38Scan20 : scan edge (search edge 3) 4 [179] 179 root38Tail20 winner = winner :=
  (root38Step20).trans root38Scan21
theorem root38Scan19 : scan edge (search edge 3) 4 [179] 179 root38Tail19 winner = winner :=
  (root38Step19).trans root38Scan20
theorem root38Scan18 : scan edge (search edge 3) 4 [179] 179 root38Tail18 winner = winner :=
  (root38Step18).trans root38Scan19
theorem root38Scan17 : scan edge (search edge 3) 4 [179] 179 root38Tail17 winner = winner :=
  (root38Step17).trans root38Scan18
theorem root38Scan16 : scan edge (search edge 3) 4 [179] 179 root38Tail16 winner = winner :=
  (root38Step16).trans root38Scan17
theorem root38Scan15 : scan edge (search edge 3) 4 [179] 179 root38Tail15 winner = winner :=
  (root38Step15).trans root38Scan16
theorem root38Scan14 : scan edge (search edge 3) 4 [179] 179 root38Tail14 winner = winner :=
  (root38Step14).trans root38Scan15
theorem root38Scan13 : scan edge (search edge 3) 4 [179] 179 root38Tail13 winner = winner :=
  (root38Step13).trans root38Scan14
theorem root38Scan12 : scan edge (search edge 3) 4 [179] 179 root38Tail12 winner = winner :=
  (root38Step12).trans root38Scan13
theorem root38Scan11 : scan edge (search edge 3) 4 [179] 179 root38Tail11 winner = winner :=
  (root38Step11).trans root38Scan12
theorem root38Scan10 : scan edge (search edge 3) 4 [179] 179 root38Tail10 winner = winner :=
  (root38Step10).trans root38Scan11
theorem root38Scan9 : scan edge (search edge 3) 4 [179] 179 root38Tail9 winner = winner :=
  (root38Step9).trans root38Scan10
theorem root38Scan8 : scan edge (search edge 3) 4 [179] 179 root38Tail8 winner = winner :=
  (root38Step8).trans root38Scan9
theorem root38Scan7 : scan edge (search edge 3) 4 [179] 179 root38Tail7 winner = winner :=
  (root38Step7).trans root38Scan8
theorem root38Scan6 : scan edge (search edge 3) 4 [179] 179 root38Tail6 winner = winner :=
  (root38Step6).trans root38Scan7
theorem root38Scan5 : scan edge (search edge 3) 4 [179] 179 root38Tail5 winner = winner :=
  (root38Step5).trans root38Scan6
theorem root38Scan4 : scan edge (search edge 3) 4 [179] 179 root38Tail4 winner = winner :=
  (root38Step4).trans root38Scan5
theorem root38Scan3 : scan edge (search edge 3) 4 [179] 179 root38Tail3 winner = winner :=
  (root38Step3).trans root38Scan4
theorem root38Scan2 : scan edge (search edge 3) 4 [179] 179 root38Tail2 winner = winner :=
  (root38Step2).trans root38Scan3
theorem root38Scan1 : scan edge (search edge 3) 4 [179] 179 root38Tail1 winner = winner :=
  (root38Step1).trans root38Scan2
theorem root38Scan0 : scan edge (search edge 3) 4 [179] 179 root38Tail0 winner = winner :=
  (root38Step0).trans root38Scan1
set_option maxRecDepth 4096 in
theorem root38Branch : search edge 4 [179] root38Candidates 179 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [179] 179 root38Tail0 winner = winner
  exact root38Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
