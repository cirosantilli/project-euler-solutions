import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root58Candidates : List Nat := Data.row283
set_option maxRecDepth 8192 in
theorem root58Candidates_checked : candidates edge 5 0 283 tail59 winner.bound = root58Candidates := by
  apply candidates_eq_row edge 5 0 283 tail59 root58Candidates winner.bound tail59_ordered
  · decide
  · decide
  · funext q
    rfl
def root58Tail72 : List Nat := []
def root58Tail71 : List Nat := 25747 :: root58Tail72
def root58Tail70 : List Nat := 24847 :: root58Tail71
def root58Tail69 : List Nat := 23689 :: root58Tail70
def root58Tail68 : List Nat := 23629 :: root58Tail69
def root58Tail67 : List Nat := 23011 :: root58Tail68
def root58Tail66 : List Nat := 22639 :: root58Tail67
def root58Tail65 : List Nat := 22621 :: root58Tail66
def root58Tail64 : List Nat := 21523 :: root58Tail65
def root58Tail63 : List Nat := 21379 :: root58Tail64
def root58Tail62 : List Nat := 21193 :: root58Tail63
def root58Tail61 : List Nat := 20899 :: root58Tail62
def root58Tail60 : List Nat := 20749 :: root58Tail61
def root58Tail59 : List Nat := 20731 :: root58Tail60
def root58Tail58 : List Nat := 19891 :: root58Tail59
def root58Tail57 : List Nat := 19609 :: root58Tail58
def root58Tail56 : List Nat := 18793 :: root58Tail57
def root58Tail55 : List Nat := 18211 :: root58Tail56
def root58Tail54 : List Nat := 17383 :: root58Tail55
def root58Tail53 : List Nat := 17203 :: root58Tail54
def root58Tail52 : List Nat := 16987 :: root58Tail53
def root58Tail51 : List Nat := 16699 :: root58Tail52
def root58Tail50 : List Nat := 16333 :: root58Tail51
def root58Tail49 : List Nat := 16087 :: root58Tail50
def root58Tail48 : List Nat := 16069 :: root58Tail49
def root58Tail47 : List Nat := 15913 :: root58Tail48
def root58Tail46 : List Nat := 15787 :: root58Tail47
def root58Tail45 : List Nat := 15289 :: root58Tail46
def root58Tail44 : List Nat := 15241 :: root58Tail45
def root58Tail43 : List Nat := 14929 :: root58Tail44
def root58Tail42 : List Nat := 14653 :: root58Tail43
def root58Tail41 : List Nat := 14347 :: root58Tail42
def root58Tail40 : List Nat := 13963 :: root58Tail41
def root58Tail39 : List Nat := 13339 :: root58Tail40
def root58Tail38 : List Nat := 13099 :: root58Tail39
def root58Tail37 : List Nat := 13063 :: root58Tail38
def root58Tail36 : List Nat := 11887 :: root58Tail37
def root58Tail35 : List Nat := 11821 :: root58Tail36
def root58Tail34 : List Nat := 10837 :: root58Tail35
def root58Tail33 : List Nat := 10273 :: root58Tail34
def root58Tail32 : List Nat := 9931 :: root58Tail33
def root58Tail31 : List Nat := 9349 :: root58Tail32
def root58Tail30 : List Nat := 9049 :: root58Tail31
def root58Tail29 : List Nat := 8293 :: root58Tail30
def root58Tail28 : List Nat := 8287 :: root58Tail29
def root58Tail27 : List Nat := 8191 :: root58Tail28
def root58Tail26 : List Nat := 7057 :: root58Tail27
def root58Tail25 : List Nat := 7039 :: root58Tail26
def root58Tail24 : List Nat := 6763 :: root58Tail25
def root58Tail23 : List Nat := 6733 :: root58Tail24
def root58Tail22 : List Nat := 6607 :: root58Tail23
def root58Tail21 : List Nat := 6397 :: root58Tail22
def root58Tail20 : List Nat := 5857 :: root58Tail21
def root58Tail19 : List Nat := 5851 :: root58Tail20
def root58Tail18 : List Nat := 5737 :: root58Tail19
def root58Tail17 : List Nat := 5653 :: root58Tail18
def root58Tail16 : List Nat := 4813 :: root58Tail17
def root58Tail15 : List Nat := 3529 :: root58Tail16
def root58Tail14 : List Nat := 3001 :: root58Tail15
def root58Tail13 : List Nat := 2503 :: root58Tail14
def root58Tail12 : List Nat := 2287 :: root58Tail13
def root58Tail11 : List Nat := 2113 :: root58Tail12
def root58Tail10 : List Nat := 1657 :: root58Tail11
def root58Tail9 : List Nat := 1489 :: root58Tail10
def root58Tail8 : List Nat := 1429 :: root58Tail9
def root58Tail7 : List Nat := 1321 :: root58Tail8
def root58Tail6 : List Nat := 1237 :: root58Tail7
def root58Tail5 : List Nat := 1051 :: root58Tail6
def root58Tail4 : List Nat := 601 :: root58Tail5
def root58Tail3 : List Nat := 541 :: root58Tail4
def root58Tail2 : List Nat := 487 :: root58Tail3
def root58Tail1 : List Nat := 463 :: root58Tail2
def root58Tail0 : List Nat := 397 :: root58Tail1
def root58Candidates0 : List Nat := [6397, 16069, 20731]
set_option maxRecDepth 8192 in
theorem root58Candidates0_checked : candidates edge 4 283 397 root58Tail1 winner.bound = root58Candidates0 := by
  unfold candidates
  have row : edge 397 = (SortedNatTree.contains FastRows.row397) := by
    change Data.row397.contains = (SortedNatTree.contains FastRows.row397)
    exact FastRows.row397_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root58Child0 : search edge 3 [283, 397] root58Candidates0 680 winner = winner := by decide
theorem root58Step0 : scan edge (search edge 3) 4 [283] 283 root58Tail0 winner =
    scan edge (search edge 3) 4 [283] 283 root58Tail1 winner := by
  rw [root58Tail0, scan, if_neg (by decide), root58Candidates0_checked]
  change scan edge (search edge 3) 4 [283] 283 root58Tail1
    (if 3 ≤ root58Candidates0.length then search edge 3 [283, 397] root58Candidates0 680 winner else winner) = _
  rw [if_pos (by decide), root58Child0]
def root58Candidates1 : List Nat := [1489, 3001, 9931]
set_option maxRecDepth 8192 in
theorem root58Candidates1_checked : candidates edge 4 283 463 root58Tail2 winner.bound = root58Candidates1 := by
  unfold candidates
  have row : edge 463 = (SortedNatTree.contains FastRows.row463) := by
    change Data.row463.contains = (SortedNatTree.contains FastRows.row463)
    exact FastRows.row463_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root58Child1 : search edge 3 [283, 463] root58Candidates1 746 winner = winner := by decide
theorem root58Step1 : scan edge (search edge 3) 4 [283] 283 root58Tail1 winner =
    scan edge (search edge 3) 4 [283] 283 root58Tail2 winner := by
  rw [root58Tail1, scan, if_neg (by decide), root58Candidates1_checked]
  change scan edge (search edge 3) 4 [283] 283 root58Tail2
    (if 3 ≤ root58Candidates1.length then search edge 3 [283, 463] root58Candidates1 746 winner else winner) = _
  rw [if_pos (by decide), root58Child1]
def root58Candidates2 : List Nat := [601, 6607]
set_option maxRecDepth 8192 in
theorem root58Candidates2_checked : candidates edge 4 283 487 root58Tail3 winner.bound = root58Candidates2 := by
  unfold candidates
  have row : edge 487 = (SortedNatTree.contains FastRows.row487) := by
    change Data.row487.contains = (SortedNatTree.contains FastRows.row487)
    exact FastRows.row487_checked.symm
  rw [row]
  decide
theorem root58Step2 : scan edge (search edge 3) 4 [283] 283 root58Tail2 winner =
    scan edge (search edge 3) 4 [283] 283 root58Tail3 winner := by
  rw [root58Tail2, scan, if_neg (by decide), root58Candidates2_checked]
  change scan edge (search edge 3) 4 [283] 283 root58Tail3
    (if 3 ≤ root58Candidates2.length then search edge 3 [283, 487] root58Candidates2 770 winner else winner) = _
  rw [if_neg (by decide)]
def root58Candidates3 : List Nat := [9049]
set_option maxRecDepth 8192 in
theorem root58Candidates3_checked : candidates edge 4 283 541 root58Tail4 winner.bound = root58Candidates3 := by
  unfold candidates
  have row : edge 541 = (SortedNatTree.contains FastRows.row541) := by
    change Data.row541.contains = (SortedNatTree.contains FastRows.row541)
    exact FastRows.row541_checked.symm
  rw [row]
  decide
theorem root58Step3 : scan edge (search edge 3) 4 [283] 283 root58Tail3 winner =
    scan edge (search edge 3) 4 [283] 283 root58Tail4 winner := by
  rw [root58Tail3, scan, if_neg (by decide), root58Candidates3_checked]
  change scan edge (search edge 3) 4 [283] 283 root58Tail4
    (if 3 ≤ root58Candidates3.length then search edge 3 [283, 541] root58Candidates3 824 winner else winner) = _
  rw [if_neg (by decide)]
def root58Candidates4 : List Nat := [1321, 15241, 16087]
set_option maxRecDepth 8192 in
theorem root58Candidates4_checked : candidates edge 4 283 601 root58Tail5 winner.bound = root58Candidates4 := by
  unfold candidates
  have row : edge 601 = (SortedNatTree.contains FastRows.row601) := by
    change Data.row601.contains = (SortedNatTree.contains FastRows.row601)
    exact FastRows.row601_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root58Child4 : search edge 3 [283, 601] root58Candidates4 884 winner = winner := by decide
theorem root58Step4 : scan edge (search edge 3) 4 [283] 283 root58Tail4 winner =
    scan edge (search edge 3) 4 [283] 283 root58Tail5 winner := by
  rw [root58Tail4, scan, if_neg (by decide), root58Candidates4_checked]
  change scan edge (search edge 3) 4 [283] 283 root58Tail5
    (if 3 ≤ root58Candidates4.length then search edge 3 [283, 601] root58Candidates4 884 winner else winner) = _
  rw [if_pos (by decide), root58Child4]
def root58Candidates5 : List Nat := [4813, 6607, 16069]
set_option maxRecDepth 8192 in
theorem root58Candidates5_checked : candidates edge 4 283 1051 root58Tail6 winner.bound = root58Candidates5 := by
  unfold candidates
  have row : edge 1051 = (SortedNatTree.contains FastRows.row1051) := by
    change Data.row1051.contains = (SortedNatTree.contains FastRows.row1051)
    exact FastRows.row1051_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root58Child5 : search edge 3 [283, 1051] root58Candidates5 1334 winner = winner := by decide
theorem root58Step5 : scan edge (search edge 3) 4 [283] 283 root58Tail5 winner =
    scan edge (search edge 3) 4 [283] 283 root58Tail6 winner := by
  rw [root58Tail5, scan, if_neg (by decide), root58Candidates5_checked]
  change scan edge (search edge 3) 4 [283] 283 root58Tail6
    (if 3 ≤ root58Candidates5.length then search edge 3 [283, 1051] root58Candidates5 1334 winner else winner) = _
  rw [if_pos (by decide), root58Child5]
def root58Candidates6 : List Nat := [1657, 9931, 14653, 22639, 23011]
set_option maxRecDepth 8192 in
theorem root58Candidates6_checked : candidates edge 4 283 1237 root58Tail7 winner.bound = root58Candidates6 := by
  unfold candidates
  have row : edge 1237 = (SortedNatTree.contains FastRows.row1237) := by
    change Data.row1237.contains = (SortedNatTree.contains FastRows.row1237)
    exact FastRows.row1237_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root58Child6 : search edge 3 [283, 1237] root58Candidates6 1520 winner = winner := by decide
theorem root58Step6 : scan edge (search edge 3) 4 [283] 283 root58Tail6 winner =
    scan edge (search edge 3) 4 [283] 283 root58Tail7 winner := by
  rw [root58Tail6, scan, if_neg (by decide), root58Candidates6_checked]
  change scan edge (search edge 3) 4 [283] 283 root58Tail7
    (if 3 ≤ root58Candidates6.length then search edge 3 [283, 1237] root58Candidates6 1520 winner else winner) = _
  rw [if_pos (by decide), root58Child6]
def root58Candidates7 : List Nat := [3001, 6607, 11821]
set_option maxRecDepth 8192 in
theorem root58Candidates7_checked : candidates edge 4 283 1321 root58Tail8 winner.bound = root58Candidates7 := by
  unfold candidates
  have row : edge 1321 = (SortedNatTree.contains FastRows.row1321) := by
    change Data.row1321.contains = (SortedNatTree.contains FastRows.row1321)
    exact FastRows.row1321_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root58Child7 : search edge 3 [283, 1321] root58Candidates7 1604 winner = winner := by decide
theorem root58Step7 : scan edge (search edge 3) 4 [283] 283 root58Tail7 winner =
    scan edge (search edge 3) 4 [283] 283 root58Tail8 winner := by
  rw [root58Tail7, scan, if_neg (by decide), root58Candidates7_checked]
  change scan edge (search edge 3) 4 [283] 283 root58Tail8
    (if 3 ≤ root58Candidates7.length then search edge 3 [283, 1321] root58Candidates7 1604 winner else winner) = _
  rw [if_pos (by decide), root58Child7]
def root58Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root58Candidates8_checked : candidates edge 4 283 1429 root58Tail9 winner.bound = root58Candidates8 := by
  unfold candidates
  have row : edge 1429 = (SortedNatTree.contains FastRows.row1429) := by
    change Data.row1429.contains = (SortedNatTree.contains FastRows.row1429)
    exact FastRows.row1429_checked.symm
  rw [row]
  decide
theorem root58Step8 : scan edge (search edge 3) 4 [283] 283 root58Tail8 winner =
    scan edge (search edge 3) 4 [283] 283 root58Tail9 winner := by
  rw [root58Tail8, scan, if_neg (by decide), root58Candidates8_checked]
  change scan edge (search edge 3) 4 [283] 283 root58Tail9
    (if 3 ≤ root58Candidates8.length then search edge 3 [283, 1429] root58Candidates8 1712 winner else winner) = _
  rw [if_neg (by decide)]
def root58Candidates9 : List Nat := [5857, 14929, 20749, 21379]
set_option maxRecDepth 8192 in
theorem root58Candidates9_checked : candidates edge 4 283 1489 root58Tail10 winner.bound = root58Candidates9 := by
  unfold candidates
  have row : edge 1489 = (SortedNatTree.contains FastRows.row1489) := by
    change Data.row1489.contains = (SortedNatTree.contains FastRows.row1489)
    exact FastRows.row1489_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root58Child9 : search edge 3 [283, 1489] root58Candidates9 1772 winner = winner := by decide
theorem root58Step9 : scan edge (search edge 3) 4 [283] 283 root58Tail9 winner =
    scan edge (search edge 3) 4 [283] 283 root58Tail10 winner := by
  rw [root58Tail9, scan, if_neg (by decide), root58Candidates9_checked]
  change scan edge (search edge 3) 4 [283] 283 root58Tail10
    (if 3 ≤ root58Candidates9.length then search edge 3 [283, 1489] root58Candidates9 1772 winner else winner) = _
  rw [if_pos (by decide), root58Child9]
def root58Candidates10 : List Nat := [6397, 6733, 9931, 15787, 18793]
set_option maxRecDepth 8192 in
theorem root58Candidates10_checked : candidates edge 4 283 1657 root58Tail11 winner.bound = root58Candidates10 := by
  unfold candidates
  have row : edge 1657 = (SortedNatTree.contains FastRows.row1657) := by
    change Data.row1657.contains = (SortedNatTree.contains FastRows.row1657)
    exact FastRows.row1657_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root58Child10 : search edge 3 [283, 1657] root58Candidates10 1940 winner = winner := by decide
theorem root58Step10 : scan edge (search edge 3) 4 [283] 283 root58Tail10 winner =
    scan edge (search edge 3) 4 [283] 283 root58Tail11 winner := by
  rw [root58Tail10, scan, if_neg (by decide), root58Candidates10_checked]
  change scan edge (search edge 3) 4 [283] 283 root58Tail11
    (if 3 ≤ root58Candidates10.length then search edge 3 [283, 1657] root58Candidates10 1940 winner else winner) = _
  rw [if_pos (by decide), root58Child10]
def root58Candidates11 : List Nat := [21379, 22621]
set_option maxRecDepth 8192 in
theorem root58Candidates11_checked : candidates edge 4 283 2113 root58Tail12 winner.bound = root58Candidates11 := by
  unfold candidates
  have row : edge 2113 = (SortedNatTree.contains FastRows.row2113) := by
    change Data.row2113.contains = (SortedNatTree.contains FastRows.row2113)
    exact FastRows.row2113_checked.symm
  rw [row]
  decide
theorem root58Step11 : scan edge (search edge 3) 4 [283] 283 root58Tail11 winner =
    scan edge (search edge 3) 4 [283] 283 root58Tail12 winner := by
  rw [root58Tail11, scan, if_neg (by decide), root58Candidates11_checked]
  change scan edge (search edge 3) 4 [283] 283 root58Tail12
    (if 3 ≤ root58Candidates11.length then search edge 3 [283, 2113] root58Candidates11 2396 winner else winner) = _
  rw [if_neg (by decide)]
def root58Candidates12 : List Nat := [5857, 9049]
set_option maxRecDepth 8192 in
theorem root58Candidates12_checked : candidates edge 4 283 2287 root58Tail13 winner.bound = root58Candidates12 := by
  unfold candidates
  have row : edge 2287 = (SortedNatTree.contains FastRows.row2287) := by
    change Data.row2287.contains = (SortedNatTree.contains FastRows.row2287)
    exact FastRows.row2287_checked.symm
  rw [row]
  decide
theorem root58Step12 : scan edge (search edge 3) 4 [283] 283 root58Tail12 winner =
    scan edge (search edge 3) 4 [283] 283 root58Tail13 winner := by
  rw [root58Tail12, scan, if_neg (by decide), root58Candidates12_checked]
  change scan edge (search edge 3) 4 [283] 283 root58Tail13
    (if 3 ≤ root58Candidates12.length then search edge 3 [283, 2287] root58Candidates12 2570 winner else winner) = _
  rw [if_neg (by decide)]
def root58Candidates13 : List Nat := [6763, 15913]
set_option maxRecDepth 8192 in
theorem root58Candidates13_checked : candidates edge 4 283 2503 root58Tail14 winner.bound = root58Candidates13 := by
  unfold candidates
  have row : edge 2503 = (SortedNatTree.contains FastRows.row2503) := by
    change Data.row2503.contains = (SortedNatTree.contains FastRows.row2503)
    exact FastRows.row2503_checked.symm
  rw [row]
  decide
theorem root58Step13 : scan edge (search edge 3) 4 [283] 283 root58Tail13 winner =
    scan edge (search edge 3) 4 [283] 283 root58Tail14 winner := by
  rw [root58Tail13, scan, if_neg (by decide), root58Candidates13_checked]
  change scan edge (search edge 3) 4 [283] 283 root58Tail14
    (if 3 ≤ root58Candidates13.length then search edge 3 [283, 2503] root58Candidates13 2786 winner else winner) = _
  rw [if_neg (by decide)]
def root58Candidates14 : List Nat := [6607, 6733, 9931, 15289]
set_option maxRecDepth 8192 in
theorem root58Candidates14_checked : candidates edge 4 283 3001 root58Tail15 winner.bound = root58Candidates14 := by
  unfold candidates
  have row : edge 3001 = (SortedNatTree.contains FastRows.row3001) := by
    change Data.row3001.contains = (SortedNatTree.contains FastRows.row3001)
    exact FastRows.row3001_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root58Child14 : search edge 3 [283, 3001] root58Candidates14 3284 winner = winner := by decide
theorem root58Step14 : scan edge (search edge 3) 4 [283] 283 root58Tail14 winner =
    scan edge (search edge 3) 4 [283] 283 root58Tail15 winner := by
  rw [root58Tail14, scan, if_neg (by decide), root58Candidates14_checked]
  change scan edge (search edge 3) 4 [283] 283 root58Tail15
    (if 3 ≤ root58Candidates14.length then search edge 3 [283, 3001] root58Candidates14 3284 winner else winner) = _
  rw [if_pos (by decide), root58Child14]
def root58Candidates15 : List Nat := [5851, 6397, 6733, 13339, 16699]
set_option maxRecDepth 8192 in
theorem root58Candidates15_checked : candidates edge 4 283 3529 root58Tail16 winner.bound = root58Candidates15 := by
  unfold candidates
  have row : edge 3529 = (SortedNatTree.contains FastRows.row3529) := by
    change Data.row3529.contains = (SortedNatTree.contains FastRows.row3529)
    exact FastRows.row3529_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root58Child15 : search edge 3 [283, 3529] root58Candidates15 3812 winner = winner := by decide
theorem root58Step15 : scan edge (search edge 3) 4 [283] 283 root58Tail15 winner =
    scan edge (search edge 3) 4 [283] 283 root58Tail16 winner := by
  rw [root58Tail15, scan, if_neg (by decide), root58Candidates15_checked]
  change scan edge (search edge 3) 4 [283] 283 root58Tail16
    (if 3 ≤ root58Candidates15.length then search edge 3 [283, 3529] root58Candidates15 3812 winner else winner) = _
  rw [if_pos (by decide), root58Child15]
def root58Candidates16 : List Nat := [7057]
set_option maxRecDepth 8192 in
theorem root58Candidates16_checked : candidates edge 4 283 4813 root58Tail17 winner.bound = root58Candidates16 := by
  unfold candidates
  have row : edge 4813 = (SortedNatTree.contains FastRows.row4813) := by
    change Data.row4813.contains = (SortedNatTree.contains FastRows.row4813)
    exact FastRows.row4813_checked.symm
  rw [row]
  decide
theorem root58Step16 : scan edge (search edge 3) 4 [283] 283 root58Tail16 winner =
    scan edge (search edge 3) 4 [283] 283 root58Tail17 winner := by
  rw [root58Tail16, scan, if_neg (by decide), root58Candidates16_checked]
  change scan edge (search edge 3) 4 [283] 283 root58Tail17
    (if 3 ≤ root58Candidates16.length then search edge 3 [283, 4813] root58Candidates16 5096 winner else winner) = _
  rw [if_neg (by decide)]
def root58Candidates17 : List Nat := [18211]
set_option maxRecDepth 8192 in
theorem root58Candidates17_checked : candidates edge 4 283 5653 root58Tail18 winner.bound = root58Candidates17 := by
  unfold candidates
  have row : edge 5653 = (SortedNatTree.contains FastRows.row5653) := by
    change Data.row5653.contains = (SortedNatTree.contains FastRows.row5653)
    exact FastRows.row5653_checked.symm
  rw [row]
  decide
theorem root58Step17 : scan edge (search edge 3) 4 [283] 283 root58Tail17 winner =
    scan edge (search edge 3) 4 [283] 283 root58Tail18 winner := by
  rw [root58Tail17, scan, if_neg (by decide), root58Candidates17_checked]
  change scan edge (search edge 3) 4 [283] 283 root58Tail18
    (if 3 ≤ root58Candidates17.length then search edge 3 [283, 5653] root58Candidates17 5936 winner else winner) = _
  rw [if_neg (by decide)]
def root58Candidates18 : List Nat := [16699]
set_option maxRecDepth 8192 in
theorem root58Candidates18_checked : candidates edge 4 283 5737 root58Tail19 winner.bound = root58Candidates18 := by
  unfold candidates
  have row : edge 5737 = (SortedNatTree.contains FastRows.row5737) := by
    change Data.row5737.contains = (SortedNatTree.contains FastRows.row5737)
    exact FastRows.row5737_checked.symm
  rw [row]
  decide
theorem root58Step18 : scan edge (search edge 3) 4 [283] 283 root58Tail18 winner =
    scan edge (search edge 3) 4 [283] 283 root58Tail19 winner := by
  rw [root58Tail18, scan, if_neg (by decide), root58Candidates18_checked]
  change scan edge (search edge 3) 4 [283] 283 root58Tail19
    (if 3 ≤ root58Candidates18.length then search edge 3 [283, 5737] root58Candidates18 6020 winner else winner) = _
  rw [if_neg (by decide)]
def root58Candidates19 : List Nat := [6607, 10837, 13063, 19891]
set_option maxRecDepth 8192 in
theorem root58Candidates19_checked : candidates edge 4 283 5851 root58Tail20 winner.bound = root58Candidates19 := by
  unfold candidates
  have row : edge 5851 = (SortedNatTree.contains FastRows.row5851) := by
    change Data.row5851.contains = (SortedNatTree.contains FastRows.row5851)
    exact FastRows.row5851_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root58Child19 : search edge 3 [283, 5851] root58Candidates19 6134 winner = winner := by decide
theorem root58Step19 : scan edge (search edge 3) 4 [283] 283 root58Tail19 winner =
    scan edge (search edge 3) 4 [283] 283 root58Tail20 winner := by
  rw [root58Tail19, scan, if_neg (by decide), root58Candidates19_checked]
  change scan edge (search edge 3) 4 [283] 283 root58Tail20
    (if 3 ≤ root58Candidates19.length then search edge 3 [283, 5851] root58Candidates19 6134 winner else winner) = _
  rw [if_pos (by decide), root58Child19]
def root58Candidates20 : List Nat := [8293, 16987]
set_option maxRecDepth 8192 in
theorem root58Candidates20_checked : candidates edge 4 283 5857 root58Tail21 winner.bound = root58Candidates20 := by
  unfold candidates
  have row : edge 5857 = (SortedNatTree.contains FastRows.row5857) := by
    change Data.row5857.contains = (SortedNatTree.contains FastRows.row5857)
    exact FastRows.row5857_checked.symm
  rw [row]
  decide
theorem root58Step20 : scan edge (search edge 3) 4 [283] 283 root58Tail20 winner =
    scan edge (search edge 3) 4 [283] 283 root58Tail21 winner := by
  rw [root58Tail20, scan, if_neg (by decide), root58Candidates20_checked]
  change scan edge (search edge 3) 4 [283] 283 root58Tail21
    (if 3 ≤ root58Candidates20.length then search edge 3 [283, 5857] root58Candidates20 6140 winner else winner) = _
  rw [if_neg (by decide)]
def root58Candidates21 : List Nat := []
set_option maxRecDepth 8192 in
theorem root58Candidates21_checked : candidates edge 4 283 6397 root58Tail22 winner.bound = root58Candidates21 := by
  unfold candidates
  have row : edge 6397 = (SortedNatTree.contains FastRows.row6397) := by
    change Data.row6397.contains = (SortedNatTree.contains FastRows.row6397)
    exact FastRows.row6397_checked.symm
  rw [row]
  decide
theorem root58Step21 : scan edge (search edge 3) 4 [283] 283 root58Tail21 winner =
    scan edge (search edge 3) 4 [283] 283 root58Tail22 winner := by
  rw [root58Tail21, scan, if_neg (by decide), root58Candidates21_checked]
  change scan edge (search edge 3) 4 [283] 283 root58Tail22
    (if 3 ≤ root58Candidates21.length then search edge 3 [283, 6397] root58Candidates21 6680 winner else winner) = _
  rw [if_neg (by decide)]
theorem root58Scan22 : scan edge (search edge 3) 4 [283] 283 root58Tail22 winner = winner := by rw [root58Tail22, scan, if_pos (by decide)]
theorem root58Scan21 : scan edge (search edge 3) 4 [283] 283 root58Tail21 winner = winner :=
  (root58Step21).trans root58Scan22
theorem root58Scan20 : scan edge (search edge 3) 4 [283] 283 root58Tail20 winner = winner :=
  (root58Step20).trans root58Scan21
theorem root58Scan19 : scan edge (search edge 3) 4 [283] 283 root58Tail19 winner = winner :=
  (root58Step19).trans root58Scan20
theorem root58Scan18 : scan edge (search edge 3) 4 [283] 283 root58Tail18 winner = winner :=
  (root58Step18).trans root58Scan19
theorem root58Scan17 : scan edge (search edge 3) 4 [283] 283 root58Tail17 winner = winner :=
  (root58Step17).trans root58Scan18
theorem root58Scan16 : scan edge (search edge 3) 4 [283] 283 root58Tail16 winner = winner :=
  (root58Step16).trans root58Scan17
theorem root58Scan15 : scan edge (search edge 3) 4 [283] 283 root58Tail15 winner = winner :=
  (root58Step15).trans root58Scan16
theorem root58Scan14 : scan edge (search edge 3) 4 [283] 283 root58Tail14 winner = winner :=
  (root58Step14).trans root58Scan15
theorem root58Scan13 : scan edge (search edge 3) 4 [283] 283 root58Tail13 winner = winner :=
  (root58Step13).trans root58Scan14
theorem root58Scan12 : scan edge (search edge 3) 4 [283] 283 root58Tail12 winner = winner :=
  (root58Step12).trans root58Scan13
theorem root58Scan11 : scan edge (search edge 3) 4 [283] 283 root58Tail11 winner = winner :=
  (root58Step11).trans root58Scan12
theorem root58Scan10 : scan edge (search edge 3) 4 [283] 283 root58Tail10 winner = winner :=
  (root58Step10).trans root58Scan11
theorem root58Scan9 : scan edge (search edge 3) 4 [283] 283 root58Tail9 winner = winner :=
  (root58Step9).trans root58Scan10
theorem root58Scan8 : scan edge (search edge 3) 4 [283] 283 root58Tail8 winner = winner :=
  (root58Step8).trans root58Scan9
theorem root58Scan7 : scan edge (search edge 3) 4 [283] 283 root58Tail7 winner = winner :=
  (root58Step7).trans root58Scan8
theorem root58Scan6 : scan edge (search edge 3) 4 [283] 283 root58Tail6 winner = winner :=
  (root58Step6).trans root58Scan7
theorem root58Scan5 : scan edge (search edge 3) 4 [283] 283 root58Tail5 winner = winner :=
  (root58Step5).trans root58Scan6
theorem root58Scan4 : scan edge (search edge 3) 4 [283] 283 root58Tail4 winner = winner :=
  (root58Step4).trans root58Scan5
theorem root58Scan3 : scan edge (search edge 3) 4 [283] 283 root58Tail3 winner = winner :=
  (root58Step3).trans root58Scan4
theorem root58Scan2 : scan edge (search edge 3) 4 [283] 283 root58Tail2 winner = winner :=
  (root58Step2).trans root58Scan3
theorem root58Scan1 : scan edge (search edge 3) 4 [283] 283 root58Tail1 winner = winner :=
  (root58Step1).trans root58Scan2
theorem root58Scan0 : scan edge (search edge 3) 4 [283] 283 root58Tail0 winner = winner :=
  (root58Step0).trans root58Scan1
set_option maxRecDepth 4096 in
theorem root58Branch : search edge 4 [283] root58Candidates 283 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [283] 283 root58Tail0 winner = winner
  exact root58Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
