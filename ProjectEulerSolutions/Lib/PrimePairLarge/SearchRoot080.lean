import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root80Candidates : List Nat := Data.row431
set_option maxRecDepth 8192 in
theorem root80Candidates_checked : candidates edge 5 0 431 tail81 winner.bound = root80Candidates := by
  apply candidates_eq_row edge 5 0 431 tail81 root80Candidates winner.bound tail81_ordered
  · decide
  · decide
  · funext q
    rfl
def root80Tail73 : List Nat := []
def root80Tail72 : List Nat := 25031 :: root80Tail73
def root80Tail71 : List Nat := 25013 :: root80Tail72
def root80Tail70 : List Nat := 24821 :: root80Tail71
def root80Tail69 : List Nat := 24659 :: root80Tail70
def root80Tail68 : List Nat := 23687 :: root80Tail69
def root80Tail67 : List Nat := 23567 :: root80Tail68
def root80Tail66 : List Nat := 23189 :: root80Tail67
def root80Tail65 : List Nat := 23099 :: root80Tail66
def root80Tail64 : List Nat := 23039 :: root80Tail65
def root80Tail63 : List Nat := 22637 :: root80Tail64
def root80Tail62 : List Nat := 22619 :: root80Tail63
def root80Tail61 : List Nat := 22613 :: root80Tail62
def root80Tail60 : List Nat := 21839 :: root80Tail61
def root80Tail59 : List Nat := 21467 :: root80Tail60
def root80Tail58 : List Nat := 20771 :: root80Tail59
def root80Tail57 : List Nat := 20393 :: root80Tail58
def root80Tail56 : List Nat := 20123 :: root80Tail57
def root80Tail55 : List Nat := 20117 :: root80Tail56
def root80Tail54 : List Nat := 19403 :: root80Tail55
def root80Tail53 : List Nat := 19301 :: root80Tail54
def root80Tail52 : List Nat := 18287 :: root80Tail53
def root80Tail51 : List Nat := 18191 :: root80Tail52
def root80Tail50 : List Nat := 17909 :: root80Tail51
def root80Tail49 : List Nat := 15749 :: root80Tail50
def root80Tail48 : List Nat := 15671 :: root80Tail49
def root80Tail47 : List Nat := 15473 :: root80Tail48
def root80Tail46 : List Nat := 14867 :: root80Tail47
def root80Tail45 : List Nat := 14717 :: root80Tail46
def root80Tail44 : List Nat := 14633 :: root80Tail45
def root80Tail43 : List Nat := 13859 :: root80Tail44
def root80Tail42 : List Nat := 13331 :: root80Tail43
def root80Tail41 : List Nat := 13043 :: root80Tail42
def root80Tail40 : List Nat := 12743 :: root80Tail41
def root80Tail39 : List Nat := 12203 :: root80Tail40
def root80Tail38 : List Nat := 11909 :: root80Tail39
def root80Tail37 : List Nat := 11903 :: root80Tail38
def root80Tail36 : List Nat := 11807 :: root80Tail37
def root80Tail35 : List Nat := 11597 :: root80Tail36
def root80Tail34 : List Nat := 11489 :: root80Tail35
def root80Tail33 : List Nat := 11093 :: root80Tail34
def root80Tail32 : List Nat := 10607 :: root80Tail33
def root80Tail31 : List Nat := 10247 :: root80Tail32
def root80Tail30 : List Nat := 9941 :: root80Tail31
def root80Tail29 : List Nat := 9767 :: root80Tail30
def root80Tail28 : List Nat := 9431 :: root80Tail29
def root80Tail27 : List Nat := 9311 :: root80Tail28
def root80Tail26 : List Nat := 8513 :: root80Tail27
def root80Tail25 : List Nat := 7517 :: root80Tail26
def root80Tail24 : List Nat := 7283 :: root80Tail25
def root80Tail23 : List Nat := 7127 :: root80Tail24
def root80Tail22 : List Nat := 6803 :: root80Tail23
def root80Tail21 : List Nat := 6329 :: root80Tail22
def root80Tail20 : List Nat := 6197 :: root80Tail21
def root80Tail19 : List Nat := 5981 :: root80Tail20
def root80Tail18 : List Nat := 5903 :: root80Tail19
def root80Tail17 : List Nat := 4943 :: root80Tail18
def root80Tail16 : List Nat := 4217 :: root80Tail17
def root80Tail15 : List Nat := 3851 :: root80Tail16
def root80Tail14 : List Nat := 3671 :: root80Tail15
def root80Tail13 : List Nat := 2837 :: root80Tail14
def root80Tail12 : List Nat := 2549 :: root80Tail13
def root80Tail11 : List Nat := 2417 :: root80Tail12
def root80Tail10 : List Nat := 2393 :: root80Tail11
def root80Tail9 : List Nat := 1823 :: root80Tail10
def root80Tail8 : List Nat := 1361 :: root80Tail9
def root80Tail7 : List Nat := 1193 :: root80Tail8
def root80Tail6 : List Nat := 1163 :: root80Tail7
def root80Tail5 : List Nat := 1031 :: root80Tail6
def root80Tail4 : List Nat := 983 :: root80Tail5
def root80Tail3 : List Nat := 947 :: root80Tail4
def root80Tail2 : List Nat := 929 :: root80Tail3
def root80Tail1 : List Nat := 857 :: root80Tail2
def root80Tail0 : List Nat := 479 :: root80Tail1
def root80Candidates0 : List Nat := [7517, 9941, 11909, 12203, 20123]
set_option maxRecDepth 8192 in
theorem root80Candidates0_checked : candidates edge 4 431 479 root80Tail1 winner.bound = root80Candidates0 := by
  unfold candidates
  have row : edge 479 = (SortedNatTree.contains FastRows.row479) := by
    change Data.row479.contains = (SortedNatTree.contains FastRows.row479)
    exact FastRows.row479_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root80Child0 : search edge 3 [431, 479] root80Candidates0 910 winner = winner := by decide
theorem root80Step0 : scan edge (search edge 3) 4 [431] 431 root80Tail0 winner =
    scan edge (search edge 3) 4 [431] 431 root80Tail1 winner := by
  rw [root80Tail0, scan, if_neg (by decide), root80Candidates0_checked]
  change scan edge (search edge 3) 4 [431] 431 root80Tail1
    (if 3 ≤ root80Candidates0.length then search edge 3 [431, 479] root80Candidates0 910 winner else winner) = _
  rw [if_pos (by decide), root80Child0]
def root80Candidates1 : List Nat := [6329]
set_option maxRecDepth 8192 in
theorem root80Candidates1_checked : candidates edge 4 431 857 root80Tail2 winner.bound = root80Candidates1 := by
  unfold candidates
  have row : edge 857 = (SortedNatTree.contains FastRows.row857) := by
    change Data.row857.contains = (SortedNatTree.contains FastRows.row857)
    exact FastRows.row857_checked.symm
  rw [row]
  decide
theorem root80Step1 : scan edge (search edge 3) 4 [431] 431 root80Tail1 winner =
    scan edge (search edge 3) 4 [431] 431 root80Tail2 winner := by
  rw [root80Tail1, scan, if_neg (by decide), root80Candidates1_checked]
  change scan edge (search edge 3) 4 [431] 431 root80Tail2
    (if 3 ≤ root80Candidates1.length then search edge 3 [431, 857] root80Candidates1 1288 winner else winner) = _
  rw [if_neg (by decide)]
def root80Candidates2 : List Nat := [983, 14867, 20771]
set_option maxRecDepth 8192 in
theorem root80Candidates2_checked : candidates edge 4 431 929 root80Tail3 winner.bound = root80Candidates2 := by
  unfold candidates
  have row : edge 929 = (SortedNatTree.contains FastRows.row929) := by
    change Data.row929.contains = (SortedNatTree.contains FastRows.row929)
    exact FastRows.row929_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root80Child2 : search edge 3 [431, 929] root80Candidates2 1360 winner = winner := by decide
theorem root80Step2 : scan edge (search edge 3) 4 [431] 431 root80Tail2 winner =
    scan edge (search edge 3) 4 [431] 431 root80Tail3 winner := by
  rw [root80Tail2, scan, if_neg (by decide), root80Candidates2_checked]
  change scan edge (search edge 3) 4 [431] 431 root80Tail3
    (if 3 ≤ root80Candidates2.length then search edge 3 [431, 929] root80Candidates2 1360 winner else winner) = _
  rw [if_pos (by decide), root80Child2]
def root80Candidates3 : List Nat := [5981, 20117, 22637]
set_option maxRecDepth 8192 in
theorem root80Candidates3_checked : candidates edge 4 431 947 root80Tail4 winner.bound = root80Candidates3 := by
  unfold candidates
  have row : edge 947 = (SortedNatTree.contains FastRows.row947) := by
    change Data.row947.contains = (SortedNatTree.contains FastRows.row947)
    exact FastRows.row947_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root80Child3 : search edge 3 [431, 947] root80Candidates3 1378 winner = winner := by decide
theorem root80Step3 : scan edge (search edge 3) 4 [431] 431 root80Tail3 winner =
    scan edge (search edge 3) 4 [431] 431 root80Tail4 winner := by
  rw [root80Tail3, scan, if_neg (by decide), root80Candidates3_checked]
  change scan edge (search edge 3) 4 [431] 431 root80Tail4
    (if 3 ≤ root80Candidates3.length then search edge 3 [431, 947] root80Candidates3 1378 winner else winner) = _
  rw [if_pos (by decide), root80Child3]
def root80Candidates4 : List Nat := [11489, 13043, 19301, 21467]
set_option maxRecDepth 8192 in
theorem root80Candidates4_checked : candidates edge 4 431 983 root80Tail5 winner.bound = root80Candidates4 := by
  unfold candidates
  have row : edge 983 = (SortedNatTree.contains FastRows.row983) := by
    change Data.row983.contains = (SortedNatTree.contains FastRows.row983)
    exact FastRows.row983_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root80Child4 : search edge 3 [431, 983] root80Candidates4 1414 winner = winner := by decide
theorem root80Step4 : scan edge (search edge 3) 4 [431] 431 root80Tail4 winner =
    scan edge (search edge 3) 4 [431] 431 root80Tail5 winner := by
  rw [root80Tail4, scan, if_neg (by decide), root80Candidates4_checked]
  change scan edge (search edge 3) 4 [431] 431 root80Tail5
    (if 3 ≤ root80Candidates4.length then search edge 3 [431, 983] root80Candidates4 1414 winner else winner) = _
  rw [if_pos (by decide), root80Child4]
def root80Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root80Candidates5_checked : candidates edge 4 431 1031 root80Tail6 winner.bound = root80Candidates5 := by
  unfold candidates
  have row : edge 1031 = (SortedNatTree.contains FastRows.row1031) := by
    change Data.row1031.contains = (SortedNatTree.contains FastRows.row1031)
    exact FastRows.row1031_checked.symm
  rw [row]
  decide
theorem root80Step5 : scan edge (search edge 3) 4 [431] 431 root80Tail5 winner =
    scan edge (search edge 3) 4 [431] 431 root80Tail6 winner := by
  rw [root80Tail5, scan, if_neg (by decide), root80Candidates5_checked]
  change scan edge (search edge 3) 4 [431] 431 root80Tail6
    (if 3 ≤ root80Candidates5.length then search edge 3 [431, 1031] root80Candidates5 1462 winner else winner) = _
  rw [if_neg (by decide)]
def root80Candidates6 : List Nat := [2417, 6197, 6803, 12743]
set_option maxRecDepth 8192 in
theorem root80Candidates6_checked : candidates edge 4 431 1163 root80Tail7 winner.bound = root80Candidates6 := by
  unfold candidates
  have row : edge 1163 = (SortedNatTree.contains FastRows.row1163) := by
    change Data.row1163.contains = (SortedNatTree.contains FastRows.row1163)
    exact FastRows.row1163_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root80Child6 : search edge 3 [431, 1163] root80Candidates6 1594 winner = winner := by decide
theorem root80Step6 : scan edge (search edge 3) 4 [431] 431 root80Tail6 winner =
    scan edge (search edge 3) 4 [431] 431 root80Tail7 winner := by
  rw [root80Tail6, scan, if_neg (by decide), root80Candidates6_checked]
  change scan edge (search edge 3) 4 [431] 431 root80Tail7
    (if 3 ≤ root80Candidates6.length then search edge 3 [431, 1163] root80Candidates6 1594 winner else winner) = _
  rw [if_pos (by decide), root80Child6]
def root80Candidates7 : List Nat := [3671, 13043, 23189]
set_option maxRecDepth 8192 in
theorem root80Candidates7_checked : candidates edge 4 431 1193 root80Tail8 winner.bound = root80Candidates7 := by
  unfold candidates
  have row : edge 1193 = (SortedNatTree.contains FastRows.row1193) := by
    change Data.row1193.contains = (SortedNatTree.contains FastRows.row1193)
    exact FastRows.row1193_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root80Child7 : search edge 3 [431, 1193] root80Candidates7 1624 winner = winner := by decide
theorem root80Step7 : scan edge (search edge 3) 4 [431] 431 root80Tail7 winner =
    scan edge (search edge 3) 4 [431] 431 root80Tail8 winner := by
  rw [root80Tail7, scan, if_neg (by decide), root80Candidates7_checked]
  change scan edge (search edge 3) 4 [431] 431 root80Tail8
    (if 3 ≤ root80Candidates7.length then search edge 3 [431, 1193] root80Candidates7 1624 winner else winner) = _
  rw [if_pos (by decide), root80Child7]
def root80Candidates8 : List Nat := [13331]
set_option maxRecDepth 8192 in
theorem root80Candidates8_checked : candidates edge 4 431 1361 root80Tail9 winner.bound = root80Candidates8 := by
  unfold candidates
  have row : edge 1361 = (SortedNatTree.contains FastRows.row1361) := by
    change Data.row1361.contains = (SortedNatTree.contains FastRows.row1361)
    exact FastRows.row1361_checked.symm
  rw [row]
  decide
theorem root80Step8 : scan edge (search edge 3) 4 [431] 431 root80Tail8 winner =
    scan edge (search edge 3) 4 [431] 431 root80Tail9 winner := by
  rw [root80Tail8, scan, if_neg (by decide), root80Candidates8_checked]
  change scan edge (search edge 3) 4 [431] 431 root80Tail9
    (if 3 ≤ root80Candidates8.length then search edge 3 [431, 1361] root80Candidates8 1792 winner else winner) = _
  rw [if_neg (by decide)]
def root80Candidates9 : List Nat := [11903, 12203]
set_option maxRecDepth 8192 in
theorem root80Candidates9_checked : candidates edge 4 431 1823 root80Tail10 winner.bound = root80Candidates9 := by
  unfold candidates
  have row : edge 1823 = (SortedNatTree.contains FastRows.row1823) := by
    change Data.row1823.contains = (SortedNatTree.contains FastRows.row1823)
    exact FastRows.row1823_checked.symm
  rw [row]
  decide
theorem root80Step9 : scan edge (search edge 3) 4 [431] 431 root80Tail9 winner =
    scan edge (search edge 3) 4 [431] 431 root80Tail10 winner := by
  rw [root80Tail9, scan, if_neg (by decide), root80Candidates9_checked]
  change scan edge (search edge 3) 4 [431] 431 root80Tail10
    (if 3 ≤ root80Candidates9.length then search edge 3 [431, 1823] root80Candidates9 2254 winner else winner) = _
  rw [if_neg (by decide)]
def root80Candidates10 : List Nat := [3851, 5981, 6329, 7127]
set_option maxRecDepth 8192 in
theorem root80Candidates10_checked : candidates edge 4 431 2393 root80Tail11 winner.bound = root80Candidates10 := by
  unfold candidates
  have row : edge 2393 = (SortedNatTree.contains FastRows.row2393) := by
    change Data.row2393.contains = (SortedNatTree.contains FastRows.row2393)
    exact FastRows.row2393_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root80Child10 : search edge 3 [431, 2393] root80Candidates10 2824 winner = winner := by decide
theorem root80Step10 : scan edge (search edge 3) 4 [431] 431 root80Tail10 winner =
    scan edge (search edge 3) 4 [431] 431 root80Tail11 winner := by
  rw [root80Tail10, scan, if_neg (by decide), root80Candidates10_checked]
  change scan edge (search edge 3) 4 [431] 431 root80Tail11
    (if 3 ≤ root80Candidates10.length then search edge 3 [431, 2393] root80Candidates10 2824 winner else winner) = _
  rw [if_pos (by decide), root80Child10]
def root80Candidates11 : List Nat := [5903, 23099]
set_option maxRecDepth 8192 in
theorem root80Candidates11_checked : candidates edge 4 431 2417 root80Tail12 winner.bound = root80Candidates11 := by
  unfold candidates
  have row : edge 2417 = (SortedNatTree.contains FastRows.row2417) := by
    change Data.row2417.contains = (SortedNatTree.contains FastRows.row2417)
    exact FastRows.row2417_checked.symm
  rw [row]
  decide
theorem root80Step11 : scan edge (search edge 3) 4 [431] 431 root80Tail11 winner =
    scan edge (search edge 3) 4 [431] 431 root80Tail12 winner := by
  rw [root80Tail11, scan, if_neg (by decide), root80Candidates11_checked]
  change scan edge (search edge 3) 4 [431] 431 root80Tail12
    (if 3 ≤ root80Candidates11.length then search edge 3 [431, 2417] root80Candidates11 2848 winner else winner) = _
  rw [if_neg (by decide)]
def root80Candidates12 : List Nat := [14717]
set_option maxRecDepth 8192 in
theorem root80Candidates12_checked : candidates edge 4 431 2549 root80Tail13 winner.bound = root80Candidates12 := by
  unfold candidates
  have row : edge 2549 = (SortedNatTree.contains FastRows.row2549) := by
    change Data.row2549.contains = (SortedNatTree.contains FastRows.row2549)
    exact FastRows.row2549_checked.symm
  rw [row]
  decide
theorem root80Step12 : scan edge (search edge 3) 4 [431] 431 root80Tail12 winner =
    scan edge (search edge 3) 4 [431] 431 root80Tail13 winner := by
  rw [root80Tail12, scan, if_neg (by decide), root80Candidates12_checked]
  change scan edge (search edge 3) 4 [431] 431 root80Tail13
    (if 3 ≤ root80Candidates12.length then search edge 3 [431, 2549] root80Candidates12 2980 winner else winner) = _
  rw [if_neg (by decide)]
def root80Candidates13 : List Nat := [9311, 9431, 12203, 15749, 20117, 20771]
set_option maxRecDepth 8192 in
theorem root80Candidates13_checked : candidates edge 4 431 2837 root80Tail14 winner.bound = root80Candidates13 := by
  unfold candidates
  have row : edge 2837 = (SortedNatTree.contains FastRows.row2837) := by
    change Data.row2837.contains = (SortedNatTree.contains FastRows.row2837)
    exact FastRows.row2837_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root80Child13 : search edge 3 [431, 2837] root80Candidates13 3268 winner = winner := by decide
theorem root80Step13 : scan edge (search edge 3) 4 [431] 431 root80Tail13 winner =
    scan edge (search edge 3) 4 [431] 431 root80Tail14 winner := by
  rw [root80Tail13, scan, if_neg (by decide), root80Candidates13_checked]
  change scan edge (search edge 3) 4 [431] 431 root80Tail14
    (if 3 ≤ root80Candidates13.length then search edge 3 [431, 2837] root80Candidates13 3268 winner else winner) = _
  rw [if_pos (by decide), root80Child13]
def root80Candidates14 : List Nat := [18287, 20123]
set_option maxRecDepth 8192 in
theorem root80Candidates14_checked : candidates edge 4 431 3671 root80Tail15 winner.bound = root80Candidates14 := by
  unfold candidates
  have row : edge 3671 = (SortedNatTree.contains FastRows.row3671) := by
    change Data.row3671.contains = (SortedNatTree.contains FastRows.row3671)
    exact FastRows.row3671_checked.symm
  rw [row]
  decide
theorem root80Step14 : scan edge (search edge 3) 4 [431] 431 root80Tail14 winner =
    scan edge (search edge 3) 4 [431] 431 root80Tail15 winner := by
  rw [root80Tail14, scan, if_neg (by decide), root80Candidates14_checked]
  change scan edge (search edge 3) 4 [431] 431 root80Tail15
    (if 3 ≤ root80Candidates14.length then search edge 3 [431, 3671] root80Candidates14 4102 winner else winner) = _
  rw [if_neg (by decide)]
def root80Candidates15 : List Nat := [5903, 12743]
set_option maxRecDepth 8192 in
theorem root80Candidates15_checked : candidates edge 4 431 3851 root80Tail16 winner.bound = root80Candidates15 := by
  unfold candidates
  have row : edge 3851 = (SortedNatTree.contains FastRows.row3851) := by
    change Data.row3851.contains = (SortedNatTree.contains FastRows.row3851)
    exact FastRows.row3851_checked.symm
  rw [row]
  decide
theorem root80Step15 : scan edge (search edge 3) 4 [431] 431 root80Tail15 winner =
    scan edge (search edge 3) 4 [431] 431 root80Tail16 winner := by
  rw [root80Tail15, scan, if_neg (by decide), root80Candidates15_checked]
  change scan edge (search edge 3) 4 [431] 431 root80Tail16
    (if 3 ≤ root80Candidates15.length then search edge 3 [431, 3851] root80Candidates15 4282 winner else winner) = _
  rw [if_neg (by decide)]
def root80Candidates16 : List Nat := [5981, 11489, 11597, 17909, 20771]
set_option maxRecDepth 8192 in
theorem root80Candidates16_checked : candidates edge 4 431 4217 root80Tail17 winner.bound = root80Candidates16 := by
  unfold candidates
  have row : edge 4217 = (SortedNatTree.contains FastRows.row4217) := by
    change Data.row4217.contains = (SortedNatTree.contains FastRows.row4217)
    exact FastRows.row4217_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root80Child16 : search edge 3 [431, 4217] root80Candidates16 4648 winner = winner := by decide
theorem root80Step16 : scan edge (search edge 3) 4 [431] 431 root80Tail16 winner =
    scan edge (search edge 3) 4 [431] 431 root80Tail17 winner := by
  rw [root80Tail16, scan, if_neg (by decide), root80Candidates16_checked]
  change scan edge (search edge 3) 4 [431] 431 root80Tail17
    (if 3 ≤ root80Candidates16.length then search edge 3 [431, 4217] root80Candidates16 4648 winner else winner) = _
  rw [if_pos (by decide), root80Child16]
def root80Candidates17 : List Nat := [7517]
set_option maxRecDepth 8192 in
theorem root80Candidates17_checked : candidates edge 4 431 4943 root80Tail18 winner.bound = root80Candidates17 := by
  unfold candidates
  have row : edge 4943 = (SortedNatTree.contains FastRows.row4943) := by
    change Data.row4943.contains = (SortedNatTree.contains FastRows.row4943)
    exact FastRows.row4943_checked.symm
  rw [row]
  decide
theorem root80Step17 : scan edge (search edge 3) 4 [431] 431 root80Tail17 winner =
    scan edge (search edge 3) 4 [431] 431 root80Tail18 winner := by
  rw [root80Tail17, scan, if_neg (by decide), root80Candidates17_checked]
  change scan edge (search edge 3) 4 [431] 431 root80Tail18
    (if 3 ≤ root80Candidates17.length then search edge 3 [431, 4943] root80Candidates17 5374 winner else winner) = _
  rw [if_neg (by decide)]
def root80Candidates18 : List Nat := [11489, 14867]
set_option maxRecDepth 8192 in
theorem root80Candidates18_checked : candidates edge 4 431 5903 root80Tail19 winner.bound = root80Candidates18 := by
  unfold candidates
  have row : edge 5903 = (SortedNatTree.contains FastRows.row5903) := by
    change Data.row5903.contains = (SortedNatTree.contains FastRows.row5903)
    exact FastRows.row5903_checked.symm
  rw [row]
  decide
theorem root80Step18 : scan edge (search edge 3) 4 [431] 431 root80Tail18 winner =
    scan edge (search edge 3) 4 [431] 431 root80Tail19 winner := by
  rw [root80Tail18, scan, if_neg (by decide), root80Candidates18_checked]
  change scan edge (search edge 3) 4 [431] 431 root80Tail19
    (if 3 ≤ root80Candidates18.length then search edge 3 [431, 5903] root80Candidates18 6334 winner else winner) = _
  rw [if_neg (by decide)]
def root80Candidates19 : List Nat := [7283]
set_option maxRecDepth 8192 in
theorem root80Candidates19_checked : candidates edge 4 431 5981 root80Tail20 winner.bound = root80Candidates19 := by
  unfold candidates
  have row : edge 5981 = (SortedNatTree.contains FastRows.row5981) := by
    change Data.row5981.contains = (SortedNatTree.contains FastRows.row5981)
    exact FastRows.row5981_checked.symm
  rw [row]
  decide
theorem root80Step19 : scan edge (search edge 3) 4 [431] 431 root80Tail19 winner =
    scan edge (search edge 3) 4 [431] 431 root80Tail20 winner := by
  rw [root80Tail19, scan, if_neg (by decide), root80Candidates19_checked]
  change scan edge (search edge 3) 4 [431] 431 root80Tail20
    (if 3 ≤ root80Candidates19.length then search edge 3 [431, 5981] root80Candidates19 6412 winner else winner) = _
  rw [if_neg (by decide)]
def root80Candidates20 : List Nat := []
set_option maxRecDepth 8192 in
theorem root80Candidates20_checked : candidates edge 4 431 6197 root80Tail21 winner.bound = root80Candidates20 := by
  unfold candidates
  have row : edge 6197 = (SortedNatTree.contains FastRows.row6197) := by
    change Data.row6197.contains = (SortedNatTree.contains FastRows.row6197)
    exact FastRows.row6197_checked.symm
  rw [row]
  decide
theorem root80Step20 : scan edge (search edge 3) 4 [431] 431 root80Tail20 winner =
    scan edge (search edge 3) 4 [431] 431 root80Tail21 winner := by
  rw [root80Tail20, scan, if_neg (by decide), root80Candidates20_checked]
  change scan edge (search edge 3) 4 [431] 431 root80Tail21
    (if 3 ≤ root80Candidates20.length then search edge 3 [431, 6197] root80Candidates20 6628 winner else winner) = _
  rw [if_neg (by decide)]
def root80Candidates21 : List Nat := [18287]
set_option maxRecDepth 8192 in
theorem root80Candidates21_checked : candidates edge 4 431 6329 root80Tail22 winner.bound = root80Candidates21 := by
  unfold candidates
  have row : edge 6329 = (SortedNatTree.contains FastRows.row6329) := by
    change Data.row6329.contains = (SortedNatTree.contains FastRows.row6329)
    exact FastRows.row6329_checked.symm
  rw [row]
  decide
theorem root80Step21 : scan edge (search edge 3) 4 [431] 431 root80Tail21 winner =
    scan edge (search edge 3) 4 [431] 431 root80Tail22 winner := by
  rw [root80Tail21, scan, if_neg (by decide), root80Candidates21_checked]
  change scan edge (search edge 3) 4 [431] 431 root80Tail22
    (if 3 ≤ root80Candidates21.length then search edge 3 [431, 6329] root80Candidates21 6760 winner else winner) = _
  rw [if_neg (by decide)]
theorem root80Scan22 : scan edge (search edge 3) 4 [431] 431 root80Tail22 winner = winner := by rw [root80Tail22, scan, if_pos (by decide)]
theorem root80Scan21 : scan edge (search edge 3) 4 [431] 431 root80Tail21 winner = winner :=
  (root80Step21).trans root80Scan22
theorem root80Scan20 : scan edge (search edge 3) 4 [431] 431 root80Tail20 winner = winner :=
  (root80Step20).trans root80Scan21
theorem root80Scan19 : scan edge (search edge 3) 4 [431] 431 root80Tail19 winner = winner :=
  (root80Step19).trans root80Scan20
theorem root80Scan18 : scan edge (search edge 3) 4 [431] 431 root80Tail18 winner = winner :=
  (root80Step18).trans root80Scan19
theorem root80Scan17 : scan edge (search edge 3) 4 [431] 431 root80Tail17 winner = winner :=
  (root80Step17).trans root80Scan18
theorem root80Scan16 : scan edge (search edge 3) 4 [431] 431 root80Tail16 winner = winner :=
  (root80Step16).trans root80Scan17
theorem root80Scan15 : scan edge (search edge 3) 4 [431] 431 root80Tail15 winner = winner :=
  (root80Step15).trans root80Scan16
theorem root80Scan14 : scan edge (search edge 3) 4 [431] 431 root80Tail14 winner = winner :=
  (root80Step14).trans root80Scan15
theorem root80Scan13 : scan edge (search edge 3) 4 [431] 431 root80Tail13 winner = winner :=
  (root80Step13).trans root80Scan14
theorem root80Scan12 : scan edge (search edge 3) 4 [431] 431 root80Tail12 winner = winner :=
  (root80Step12).trans root80Scan13
theorem root80Scan11 : scan edge (search edge 3) 4 [431] 431 root80Tail11 winner = winner :=
  (root80Step11).trans root80Scan12
theorem root80Scan10 : scan edge (search edge 3) 4 [431] 431 root80Tail10 winner = winner :=
  (root80Step10).trans root80Scan11
theorem root80Scan9 : scan edge (search edge 3) 4 [431] 431 root80Tail9 winner = winner :=
  (root80Step9).trans root80Scan10
theorem root80Scan8 : scan edge (search edge 3) 4 [431] 431 root80Tail8 winner = winner :=
  (root80Step8).trans root80Scan9
theorem root80Scan7 : scan edge (search edge 3) 4 [431] 431 root80Tail7 winner = winner :=
  (root80Step7).trans root80Scan8
theorem root80Scan6 : scan edge (search edge 3) 4 [431] 431 root80Tail6 winner = winner :=
  (root80Step6).trans root80Scan7
theorem root80Scan5 : scan edge (search edge 3) 4 [431] 431 root80Tail5 winner = winner :=
  (root80Step5).trans root80Scan6
theorem root80Scan4 : scan edge (search edge 3) 4 [431] 431 root80Tail4 winner = winner :=
  (root80Step4).trans root80Scan5
theorem root80Scan3 : scan edge (search edge 3) 4 [431] 431 root80Tail3 winner = winner :=
  (root80Step3).trans root80Scan4
theorem root80Scan2 : scan edge (search edge 3) 4 [431] 431 root80Tail2 winner = winner :=
  (root80Step2).trans root80Scan3
theorem root80Scan1 : scan edge (search edge 3) 4 [431] 431 root80Tail1 winner = winner :=
  (root80Step1).trans root80Scan2
theorem root80Scan0 : scan edge (search edge 3) 4 [431] 431 root80Tail0 winner = winner :=
  (root80Step0).trans root80Scan1
set_option maxRecDepth 4096 in
theorem root80Branch : search edge 4 [431] root80Candidates 431 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [431] 431 root80Tail0 winner = winner
  exact root80Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
