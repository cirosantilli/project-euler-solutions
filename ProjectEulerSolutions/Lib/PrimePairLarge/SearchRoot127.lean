import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root127Candidates : List Nat := Data.row733
set_option maxRecDepth 8192 in
theorem root127Candidates_checked : candidates edge 5 0 733 tail128 winner.bound = root127Candidates := by
  apply candidates_eq_row edge 5 0 733 tail128 root127Candidates winner.bound tail128_ordered
  · decide
  · decide
  · funext q
    rfl
def root127Tail69 : List Nat := []
def root127Tail68 : List Nat := 25183 :: root127Tail69
def root127Tail67 : List Nat := 25171 :: root127Tail68
def root127Tail66 : List Nat := 25033 :: root127Tail67
def root127Tail65 : List Nat := 24907 :: root127Tail66
def root127Tail64 : List Nat := 24103 :: root127Tail65
def root127Tail63 : List Nat := 23893 :: root127Tail64
def root127Tail62 : List Nat := 23311 :: root127Tail63
def root127Tail61 : List Nat := 23053 :: root127Tail62
def root127Tail60 : List Nat := 22369 :: root127Tail61
def root127Tail59 : List Nat := 21991 :: root127Tail60
def root127Tail58 : List Nat := 21841 :: root127Tail59
def root127Tail57 : List Nat := 21757 :: root127Tail58
def root127Tail56 : List Nat := 21529 :: root127Tail57
def root127Tail55 : List Nat := 20899 :: root127Tail56
def root127Tail54 : List Nat := 20887 :: root127Tail55
def root127Tail53 : List Nat := 20551 :: root127Tail54
def root127Tail52 : List Nat := 20509 :: root127Tail53
def root127Tail51 : List Nat := 19819 :: root127Tail52
def root127Tail50 : List Nat := 18493 :: root127Tail51
def root127Tail49 : List Nat := 18307 :: root127Tail50
def root127Tail48 : List Nat := 17053 :: root127Tail49
def root127Tail47 : List Nat := 16963 :: root127Tail48
def root127Tail46 : List Nat := 16921 :: root127Tail47
def root127Tail45 : List Nat := 16603 :: root127Tail46
def root127Tail44 : List Nat := 16447 :: root127Tail45
def root127Tail43 : List Nat := 15919 :: root127Tail44
def root127Tail42 : List Nat := 15439 :: root127Tail43
def root127Tail41 : List Nat := 15331 :: root127Tail42
def root127Tail40 : List Nat := 15199 :: root127Tail41
def root127Tail39 : List Nat := 15061 :: root127Tail40
def root127Tail38 : List Nat := 14767 :: root127Tail39
def root127Tail37 : List Nat := 14419 :: root127Tail38
def root127Tail36 : List Nat := 14401 :: root127Tail37
def root127Tail35 : List Nat := 14389 :: root127Tail36
def root127Tail34 : List Nat := 14347 :: root127Tail35
def root127Tail33 : List Nat := 13693 :: root127Tail34
def root127Tail32 : List Nat := 13441 :: root127Tail33
def root127Tail31 : List Nat := 13063 :: root127Tail32
def root127Tail30 : List Nat := 12049 :: root127Tail31
def root127Tail29 : List Nat := 11503 :: root127Tail30
def root127Tail28 : List Nat := 11083 :: root127Tail29
def root127Tail27 : List Nat := 11047 :: root127Tail28
def root127Tail26 : List Nat := 9421 :: root127Tail27
def root127Tail25 : List Nat := 9403 :: root127Tail26
def root127Tail24 : List Nat := 9043 :: root127Tail25
def root127Tail23 : List Nat := 8641 :: root127Tail24
def root127Tail22 : List Nat := 8431 :: root127Tail23
def root127Tail21 : List Nat := 8263 :: root127Tail22
def root127Tail20 : List Nat := 7879 :: root127Tail21
def root127Tail19 : List Nat := 7723 :: root127Tail20
def root127Tail18 : List Nat := 7717 :: root127Tail19
def root127Tail17 : List Nat := 5851 :: root127Tail18
def root127Tail16 : List Nat := 5743 :: root127Tail17
def root127Tail15 : List Nat := 5701 :: root127Tail16
def root127Tail14 : List Nat := 4903 :: root127Tail15
def root127Tail13 : List Nat := 4549 :: root127Tail14
def root127Tail12 : List Nat := 4441 :: root127Tail13
def root127Tail11 : List Nat := 4357 :: root127Tail12
def root127Tail10 : List Nat := 4027 :: root127Tail11
def root127Tail9 : List Nat := 3709 :: root127Tail10
def root127Tail8 : List Nat := 3331 :: root127Tail9
def root127Tail7 : List Nat := 2551 :: root127Tail8
def root127Tail6 : List Nat := 2113 :: root127Tail7
def root127Tail5 : List Nat := 1999 :: root127Tail6
def root127Tail4 : List Nat := 1249 :: root127Tail5
def root127Tail3 : List Nat := 1039 :: root127Tail4
def root127Tail2 : List Nat := 991 :: root127Tail3
def root127Tail1 : List Nat := 883 :: root127Tail2
def root127Tail0 : List Nat := 829 :: root127Tail1
def root127Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root127Candidates0_checked : candidates edge 4 733 829 root127Tail1 winner.bound = root127Candidates0 := by
  unfold candidates
  have row : edge 829 = (SortedNatTree.contains FastRows.row829) := by
    change Data.row829.contains = (SortedNatTree.contains FastRows.row829)
    exact FastRows.row829_checked.symm
  rw [row]
  decide
theorem root127Step0 : scan edge (search edge 3) 4 [733] 733 root127Tail0 winner =
    scan edge (search edge 3) 4 [733] 733 root127Tail1 winner := by
  rw [root127Tail0, scan, if_neg (by decide), root127Candidates0_checked]
  change scan edge (search edge 3) 4 [733] 733 root127Tail1
    (if 3 ≤ root127Candidates0.length then search edge 3 [733, 829] root127Candidates0 1562 winner else winner) = _
  rw [if_neg (by decide)]
def root127Candidates1 : List Nat := [991, 9043, 9421, 14419, 18493]
set_option maxRecDepth 8192 in
theorem root127Candidates1_checked : candidates edge 4 733 883 root127Tail2 winner.bound = root127Candidates1 := by
  unfold candidates
  have row : edge 883 = (SortedNatTree.contains FastRows.row883) := by
    change Data.row883.contains = (SortedNatTree.contains FastRows.row883)
    exact FastRows.row883_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root127Child1 : search edge 3 [733, 883] root127Candidates1 1616 winner = winner := by decide
theorem root127Step1 : scan edge (search edge 3) 4 [733] 733 root127Tail1 winner =
    scan edge (search edge 3) 4 [733] 733 root127Tail2 winner := by
  rw [root127Tail1, scan, if_neg (by decide), root127Candidates1_checked]
  change scan edge (search edge 3) 4 [733] 733 root127Tail2
    (if 3 ≤ root127Candidates1.length then search edge 3 [733, 883] root127Candidates1 1616 winner else winner) = _
  rw [if_pos (by decide), root127Child1]
def root127Candidates2 : List Nat := [8641, 18493, 23053]
set_option maxRecDepth 8192 in
theorem root127Candidates2_checked : candidates edge 4 733 991 root127Tail3 winner.bound = root127Candidates2 := by
  unfold candidates
  have row : edge 991 = (SortedNatTree.contains FastRows.row991) := by
    change Data.row991.contains = (SortedNatTree.contains FastRows.row991)
    exact FastRows.row991_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root127Child2 : search edge 3 [733, 991] root127Candidates2 1724 winner = winner := by decide
theorem root127Step2 : scan edge (search edge 3) 4 [733] 733 root127Tail2 winner =
    scan edge (search edge 3) 4 [733] 733 root127Tail3 winner := by
  rw [root127Tail2, scan, if_neg (by decide), root127Candidates2_checked]
  change scan edge (search edge 3) 4 [733] 733 root127Tail3
    (if 3 ≤ root127Candidates2.length then search edge 3 [733, 991] root127Candidates2 1724 winner else winner) = _
  rw [if_pos (by decide), root127Child2]
def root127Candidates3 : List Nat := [1999, 5851, 8431]
set_option maxRecDepth 8192 in
theorem root127Candidates3_checked : candidates edge 4 733 1039 root127Tail4 winner.bound = root127Candidates3 := by
  unfold candidates
  have row : edge 1039 = (SortedNatTree.contains FastRows.row1039) := by
    change Data.row1039.contains = (SortedNatTree.contains FastRows.row1039)
    exact FastRows.row1039_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root127Child3 : search edge 3 [733, 1039] root127Candidates3 1772 winner = winner := by decide
theorem root127Step3 : scan edge (search edge 3) 4 [733] 733 root127Tail3 winner =
    scan edge (search edge 3) 4 [733] 733 root127Tail4 winner := by
  rw [root127Tail3, scan, if_neg (by decide), root127Candidates3_checked]
  change scan edge (search edge 3) 4 [733] 733 root127Tail4
    (if 3 ≤ root127Candidates3.length then search edge 3 [733, 1039] root127Candidates3 1772 winner else winner) = _
  rw [if_pos (by decide), root127Child3]
def root127Candidates4 : List Nat := [2551, 4441, 5851, 8641, 11503, 15061, 20899]
set_option maxRecDepth 8192 in
theorem root127Candidates4_checked : candidates edge 4 733 1249 root127Tail5 winner.bound = root127Candidates4 := by
  unfold candidates
  have row : edge 1249 = (SortedNatTree.contains FastRows.row1249) := by
    change Data.row1249.contains = (SortedNatTree.contains FastRows.row1249)
    exact FastRows.row1249_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root127Child4 : search edge 3 [733, 1249] root127Candidates4 1982 winner = winner := by decide
theorem root127Step4 : scan edge (search edge 3) 4 [733] 733 root127Tail4 winner =
    scan edge (search edge 3) 4 [733] 733 root127Tail5 winner := by
  rw [root127Tail4, scan, if_neg (by decide), root127Candidates4_checked]
  change scan edge (search edge 3) 4 [733] 733 root127Tail5
    (if 3 ≤ root127Candidates4.length then search edge 3 [733, 1249] root127Candidates4 1982 winner else winner) = _
  rw [if_pos (by decide), root127Child4]
def root127Candidates5 : List Nat := [7717, 8263, 22369]
set_option maxRecDepth 8192 in
theorem root127Candidates5_checked : candidates edge 4 733 1999 root127Tail6 winner.bound = root127Candidates5 := by
  unfold candidates
  have row : edge 1999 = (SortedNatTree.contains FastRows.row1999) := by
    change Data.row1999.contains = (SortedNatTree.contains FastRows.row1999)
    exact FastRows.row1999_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root127Child5 : search edge 3 [733, 1999] root127Candidates5 2732 winner = winner := by decide
theorem root127Step5 : scan edge (search edge 3) 4 [733] 733 root127Tail5 winner =
    scan edge (search edge 3) 4 [733] 733 root127Tail6 winner := by
  rw [root127Tail5, scan, if_neg (by decide), root127Candidates5_checked]
  change scan edge (search edge 3) 4 [733] 733 root127Tail6
    (if 3 ≤ root127Candidates5.length then search edge 3 [733, 1999] root127Candidates5 2732 winner else winner) = _
  rw [if_pos (by decide), root127Child5]
def root127Candidates6 : List Nat := [22369]
set_option maxRecDepth 8192 in
theorem root127Candidates6_checked : candidates edge 4 733 2113 root127Tail7 winner.bound = root127Candidates6 := by
  unfold candidates
  have row : edge 2113 = (SortedNatTree.contains FastRows.row2113) := by
    change Data.row2113.contains = (SortedNatTree.contains FastRows.row2113)
    exact FastRows.row2113_checked.symm
  rw [row]
  decide
theorem root127Step6 : scan edge (search edge 3) 4 [733] 733 root127Tail6 winner =
    scan edge (search edge 3) 4 [733] 733 root127Tail7 winner := by
  rw [root127Tail6, scan, if_neg (by decide), root127Candidates6_checked]
  change scan edge (search edge 3) 4 [733] 733 root127Tail7
    (if 3 ≤ root127Candidates6.length then search edge 3 [733, 2113] root127Candidates6 2846 winner else winner) = _
  rw [if_neg (by decide)]
def root127Candidates7 : List Nat := [3331, 16921, 21991]
set_option maxRecDepth 8192 in
theorem root127Candidates7_checked : candidates edge 4 733 2551 root127Tail8 winner.bound = root127Candidates7 := by
  unfold candidates
  have row : edge 2551 = (SortedNatTree.contains FastRows.row2551) := by
    change Data.row2551.contains = (SortedNatTree.contains FastRows.row2551)
    exact FastRows.row2551_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root127Child7 : search edge 3 [733, 2551] root127Candidates7 3284 winner = winner := by decide
theorem root127Step7 : scan edge (search edge 3) 4 [733] 733 root127Tail7 winner =
    scan edge (search edge 3) 4 [733] 733 root127Tail8 winner := by
  rw [root127Tail7, scan, if_neg (by decide), root127Candidates7_checked]
  change scan edge (search edge 3) 4 [733] 733 root127Tail8
    (if 3 ≤ root127Candidates7.length then search edge 3 [733, 2551] root127Candidates7 3284 winner else winner) = _
  rw [if_pos (by decide), root127Child7]
def root127Candidates8 : List Nat := [4903, 5701, 7717, 9421]
set_option maxRecDepth 8192 in
theorem root127Candidates8_checked : candidates edge 4 733 3331 root127Tail9 winner.bound = root127Candidates8 := by
  unfold candidates
  have row : edge 3331 = (SortedNatTree.contains FastRows.row3331) := by
    change Data.row3331.contains = (SortedNatTree.contains FastRows.row3331)
    exact FastRows.row3331_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root127Child8 : search edge 3 [733, 3331] root127Candidates8 4064 winner = winner := by decide
theorem root127Step8 : scan edge (search edge 3) 4 [733] 733 root127Tail8 winner =
    scan edge (search edge 3) 4 [733] 733 root127Tail9 winner := by
  rw [root127Tail8, scan, if_neg (by decide), root127Candidates8_checked]
  change scan edge (search edge 3) 4 [733] 733 root127Tail9
    (if 3 ≤ root127Candidates8.length then search edge 3 [733, 3331] root127Candidates8 4064 winner else winner) = _
  rw [if_pos (by decide), root127Child8]
def root127Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root127Candidates9_checked : candidates edge 4 733 3709 root127Tail10 winner.bound = root127Candidates9 := by
  unfold candidates
  have row : edge 3709 = (SortedNatTree.contains FastRows.row3709) := by
    change Data.row3709.contains = (SortedNatTree.contains FastRows.row3709)
    exact FastRows.row3709_checked.symm
  rw [row]
  decide
theorem root127Step9 : scan edge (search edge 3) 4 [733] 733 root127Tail9 winner =
    scan edge (search edge 3) 4 [733] 733 root127Tail10 winner := by
  rw [root127Tail9, scan, if_neg (by decide), root127Candidates9_checked]
  change scan edge (search edge 3) 4 [733] 733 root127Tail10
    (if 3 ≤ root127Candidates9.length then search edge 3 [733, 3709] root127Candidates9 4442 winner else winner) = _
  rw [if_neg (by decide)]
def root127Candidates10 : List Nat := [14419]
set_option maxRecDepth 8192 in
theorem root127Candidates10_checked : candidates edge 4 733 4027 root127Tail11 winner.bound = root127Candidates10 := by
  unfold candidates
  have row : edge 4027 = (SortedNatTree.contains FastRows.row4027) := by
    change Data.row4027.contains = (SortedNatTree.contains FastRows.row4027)
    exact FastRows.row4027_checked.symm
  rw [row]
  decide
theorem root127Step10 : scan edge (search edge 3) 4 [733] 733 root127Tail10 winner =
    scan edge (search edge 3) 4 [733] 733 root127Tail11 winner := by
  rw [root127Tail10, scan, if_neg (by decide), root127Candidates10_checked]
  change scan edge (search edge 3) 4 [733] 733 root127Tail11
    (if 3 ≤ root127Candidates10.length then search edge 3 [733, 4027] root127Candidates10 4760 winner else winner) = _
  rw [if_neg (by decide)]
def root127Candidates11 : List Nat := [18307]
set_option maxRecDepth 8192 in
theorem root127Candidates11_checked : candidates edge 4 733 4357 root127Tail12 winner.bound = root127Candidates11 := by
  unfold candidates
  have row : edge 4357 = (SortedNatTree.contains FastRows.row4357) := by
    change Data.row4357.contains = (SortedNatTree.contains FastRows.row4357)
    exact FastRows.row4357_checked.symm
  rw [row]
  decide
theorem root127Step11 : scan edge (search edge 3) 4 [733] 733 root127Tail11 winner =
    scan edge (search edge 3) 4 [733] 733 root127Tail12 winner := by
  rw [root127Tail11, scan, if_neg (by decide), root127Candidates11_checked]
  change scan edge (search edge 3) 4 [733] 733 root127Tail12
    (if 3 ≤ root127Candidates11.length then search edge 3 [733, 4357] root127Candidates11 5090 winner else winner) = _
  rw [if_neg (by decide)]
def root127Candidates12 : List Nat := [5743]
set_option maxRecDepth 8192 in
theorem root127Candidates12_checked : candidates edge 4 733 4441 root127Tail13 winner.bound = root127Candidates12 := by
  unfold candidates
  have row : edge 4441 = (SortedNatTree.contains FastRows.row4441) := by
    change Data.row4441.contains = (SortedNatTree.contains FastRows.row4441)
    exact FastRows.row4441_checked.symm
  rw [row]
  decide
theorem root127Step12 : scan edge (search edge 3) 4 [733] 733 root127Tail12 winner =
    scan edge (search edge 3) 4 [733] 733 root127Tail13 winner := by
  rw [root127Tail12, scan, if_neg (by decide), root127Candidates12_checked]
  change scan edge (search edge 3) 4 [733] 733 root127Tail13
    (if 3 ≤ root127Candidates12.length then search edge 3 [733, 4441] root127Candidates12 5174 winner else winner) = _
  rw [if_neg (by decide)]
def root127Candidates13 : List Nat := [5851, 7717]
set_option maxRecDepth 8192 in
theorem root127Candidates13_checked : candidates edge 4 733 4549 root127Tail14 winner.bound = root127Candidates13 := by
  unfold candidates
  have row : edge 4549 = (SortedNatTree.contains FastRows.row4549) := by
    change Data.row4549.contains = (SortedNatTree.contains FastRows.row4549)
    exact FastRows.row4549_checked.symm
  rw [row]
  decide
theorem root127Step13 : scan edge (search edge 3) 4 [733] 733 root127Tail13 winner =
    scan edge (search edge 3) 4 [733] 733 root127Tail14 winner := by
  rw [root127Tail13, scan, if_neg (by decide), root127Candidates13_checked]
  change scan edge (search edge 3) 4 [733] 733 root127Tail14
    (if 3 ≤ root127Candidates13.length then search edge 3 [733, 4549] root127Candidates13 5282 winner else winner) = _
  rw [if_neg (by decide)]
def root127Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root127Candidates14_checked : candidates edge 4 733 4903 root127Tail15 winner.bound = root127Candidates14 := by
  unfold candidates
  have row : edge 4903 = (SortedNatTree.contains FastRows.row4903) := by
    change Data.row4903.contains = (SortedNatTree.contains FastRows.row4903)
    exact FastRows.row4903_checked.symm
  rw [row]
  decide
theorem root127Step14 : scan edge (search edge 3) 4 [733] 733 root127Tail14 winner =
    scan edge (search edge 3) 4 [733] 733 root127Tail15 winner := by
  rw [root127Tail14, scan, if_neg (by decide), root127Candidates14_checked]
  change scan edge (search edge 3) 4 [733] 733 root127Tail15
    (if 3 ≤ root127Candidates14.length then search edge 3 [733, 4903] root127Candidates14 5636 winner else winner) = _
  rw [if_neg (by decide)]
def root127Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root127Candidates15_checked : candidates edge 4 733 5701 root127Tail16 winner.bound = root127Candidates15 := by
  unfold candidates
  have row : edge 5701 = (SortedNatTree.contains FastRows.row5701) := by
    change Data.row5701.contains = (SortedNatTree.contains FastRows.row5701)
    exact FastRows.row5701_checked.symm
  rw [row]
  decide
theorem root127Step15 : scan edge (search edge 3) 4 [733] 733 root127Tail15 winner =
    scan edge (search edge 3) 4 [733] 733 root127Tail16 winner := by
  rw [root127Tail15, scan, if_neg (by decide), root127Candidates15_checked]
  change scan edge (search edge 3) 4 [733] 733 root127Tail16
    (if 3 ≤ root127Candidates15.length then search edge 3 [733, 5701] root127Candidates15 6434 winner else winner) = _
  rw [if_neg (by decide)]
def root127Candidates16 : List Nat := [7717, 9043, 12049]
set_option maxRecDepth 8192 in
theorem root127Candidates16_checked : candidates edge 4 733 5743 root127Tail17 winner.bound = root127Candidates16 := by
  unfold candidates
  have row : edge 5743 = (SortedNatTree.contains FastRows.row5743) := by
    change Data.row5743.contains = (SortedNatTree.contains FastRows.row5743)
    exact FastRows.row5743_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root127Child16 : search edge 3 [733, 5743] root127Candidates16 6476 winner = winner := by decide
theorem root127Step16 : scan edge (search edge 3) 4 [733] 733 root127Tail16 winner =
    scan edge (search edge 3) 4 [733] 733 root127Tail17 winner := by
  rw [root127Tail16, scan, if_neg (by decide), root127Candidates16_checked]
  change scan edge (search edge 3) 4 [733] 733 root127Tail17
    (if 3 ≤ root127Candidates16.length then search edge 3 [733, 5743] root127Candidates16 6476 winner else winner) = _
  rw [if_pos (by decide), root127Child16]
def root127Candidates17 : List Nat := [13063, 14419, 15439, 18493]
set_option maxRecDepth 8192 in
theorem root127Candidates17_checked : candidates edge 4 733 5851 root127Tail18 winner.bound = root127Candidates17 := by
  unfold candidates
  have row : edge 5851 = (SortedNatTree.contains FastRows.row5851) := by
    change Data.row5851.contains = (SortedNatTree.contains FastRows.row5851)
    exact FastRows.row5851_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root127Child17 : search edge 3 [733, 5851] root127Candidates17 6584 winner = winner := by decide
theorem root127Step17 : scan edge (search edge 3) 4 [733] 733 root127Tail17 winner =
    scan edge (search edge 3) 4 [733] 733 root127Tail18 winner := by
  rw [root127Tail17, scan, if_neg (by decide), root127Candidates17_checked]
  change scan edge (search edge 3) 4 [733] 733 root127Tail18
    (if 3 ≤ root127Candidates17.length then search edge 3 [733, 5851] root127Candidates17 6584 winner else winner) = _
  rw [if_pos (by decide), root127Child17]
theorem root127Scan18 : scan edge (search edge 3) 4 [733] 733 root127Tail18 winner = winner := by rw [root127Tail18, scan, if_pos (by decide)]
theorem root127Scan17 : scan edge (search edge 3) 4 [733] 733 root127Tail17 winner = winner :=
  (root127Step17).trans root127Scan18
theorem root127Scan16 : scan edge (search edge 3) 4 [733] 733 root127Tail16 winner = winner :=
  (root127Step16).trans root127Scan17
theorem root127Scan15 : scan edge (search edge 3) 4 [733] 733 root127Tail15 winner = winner :=
  (root127Step15).trans root127Scan16
theorem root127Scan14 : scan edge (search edge 3) 4 [733] 733 root127Tail14 winner = winner :=
  (root127Step14).trans root127Scan15
theorem root127Scan13 : scan edge (search edge 3) 4 [733] 733 root127Tail13 winner = winner :=
  (root127Step13).trans root127Scan14
theorem root127Scan12 : scan edge (search edge 3) 4 [733] 733 root127Tail12 winner = winner :=
  (root127Step12).trans root127Scan13
theorem root127Scan11 : scan edge (search edge 3) 4 [733] 733 root127Tail11 winner = winner :=
  (root127Step11).trans root127Scan12
theorem root127Scan10 : scan edge (search edge 3) 4 [733] 733 root127Tail10 winner = winner :=
  (root127Step10).trans root127Scan11
theorem root127Scan9 : scan edge (search edge 3) 4 [733] 733 root127Tail9 winner = winner :=
  (root127Step9).trans root127Scan10
theorem root127Scan8 : scan edge (search edge 3) 4 [733] 733 root127Tail8 winner = winner :=
  (root127Step8).trans root127Scan9
theorem root127Scan7 : scan edge (search edge 3) 4 [733] 733 root127Tail7 winner = winner :=
  (root127Step7).trans root127Scan8
theorem root127Scan6 : scan edge (search edge 3) 4 [733] 733 root127Tail6 winner = winner :=
  (root127Step6).trans root127Scan7
theorem root127Scan5 : scan edge (search edge 3) 4 [733] 733 root127Tail5 winner = winner :=
  (root127Step5).trans root127Scan6
theorem root127Scan4 : scan edge (search edge 3) 4 [733] 733 root127Tail4 winner = winner :=
  (root127Step4).trans root127Scan5
theorem root127Scan3 : scan edge (search edge 3) 4 [733] 733 root127Tail3 winner = winner :=
  (root127Step3).trans root127Scan4
theorem root127Scan2 : scan edge (search edge 3) 4 [733] 733 root127Tail2 winner = winner :=
  (root127Step2).trans root127Scan3
theorem root127Scan1 : scan edge (search edge 3) 4 [733] 733 root127Tail1 winner = winner :=
  (root127Step1).trans root127Scan2
theorem root127Scan0 : scan edge (search edge 3) 4 [733] 733 root127Tail0 winner = winner :=
  (root127Step0).trans root127Scan1
set_option maxRecDepth 4096 in
theorem root127Branch : search edge 4 [733] root127Candidates 733 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [733] 733 root127Tail0 winner = winner
  exact root127Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
