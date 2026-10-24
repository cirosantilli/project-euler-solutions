import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root55Candidates : List Nat := Data.row271
set_option maxRecDepth 8192 in
theorem root55Candidates_checked : candidates edge 5 0 271 tail56 winner.bound = root55Candidates := by
  apply candidates_eq_row edge 5 0 271 tail56 root55Candidates winner.bound tail56_ordered
  · decide
  · decide
  · funext q
    rfl
def root55Tail73 : List Nat := []
def root55Tail72 : List Nat := 25633 :: root55Tail73
def root55Tail71 : List Nat := 25321 :: root55Tail72
def root55Tail70 : List Nat := 25243 :: root55Tail71
def root55Tail69 : List Nat := 23293 :: root55Tail70
def root55Tail68 : List Nat := 22567 :: root55Tail69
def root55Tail67 : List Nat := 22549 :: root55Tail68
def root55Tail66 : List Nat := 21067 :: root55Tail67
def root55Tail65 : List Nat := 20983 :: root55Tail66
def root55Tail64 : List Nat := 20719 :: root55Tail65
def root55Tail63 : List Nat := 19237 :: root55Tail64
def root55Tail62 : List Nat := 18661 :: root55Tail63
def root55Tail61 : List Nat := 17863 :: root55Tail62
def root55Tail60 : List Nat := 17749 :: root55Tail61
def root55Tail59 : List Nat := 17203 :: root55Tail60
def root55Tail58 : List Nat := 17137 :: root55Tail59
def root55Tail57 : List Nat := 16759 :: root55Tail58
def root55Tail56 : List Nat := 16561 :: root55Tail57
def root55Tail55 : List Nat := 16267 :: root55Tail56
def root55Tail54 : List Nat := 16087 :: root55Tail55
def root55Tail53 : List Nat := 15511 :: root55Tail54
def root55Tail52 : List Nat := 15259 :: root55Tail53
def root55Tail51 : List Nat := 15199 :: root55Tail52
def root55Tail50 : List Nat := 15031 :: root55Tail51
def root55Tail49 : List Nat := 14713 :: root55Tail50
def root55Tail48 : List Nat := 14293 :: root55Tail49
def root55Tail47 : List Nat := 14071 :: root55Tail48
def root55Tail46 : List Nat := 13903 :: root55Tail47
def root55Tail45 : List Nat := 13873 :: root55Tail46
def root55Tail44 : List Nat := 13477 :: root55Tail45
def root55Tail43 : List Nat := 13183 :: root55Tail44
def root55Tail42 : List Nat := 12889 :: root55Tail43
def root55Tail41 : List Nat := 12781 :: root55Tail42
def root55Tail40 : List Nat := 12511 :: root55Tail41
def root55Tail39 : List Nat := 12343 :: root55Tail40
def root55Tail38 : List Nat := 12109 :: root55Tail39
def root55Tail37 : List Nat := 12037 :: root55Tail38
def root55Tail36 : List Nat := 11677 :: root55Tail37
def root55Tail35 : List Nat := 11353 :: root55Tail36
def root55Tail34 : List Nat := 11239 :: root55Tail35
def root55Tail33 : List Nat := 11131 :: root55Tail34
def root55Tail32 : List Nat := 11059 :: root55Tail33
def root55Tail31 : List Nat := 10513 :: root55Tail32
def root55Tail30 : List Nat := 9433 :: root55Tail31
def root55Tail29 : List Nat := 9421 :: root55Tail30
def root55Tail28 : List Nat := 9043 :: root55Tail29
def root55Tail27 : List Nat := 8971 :: root55Tail28
def root55Tail26 : List Nat := 8941 :: root55Tail27
def root55Tail25 : List Nat := 8887 :: root55Tail26
def root55Tail24 : List Nat := 8269 :: root55Tail25
def root55Tail23 : List Nat := 8101 :: root55Tail24
def root55Tail22 : List Nat := 7639 :: root55Tail23
def root55Tail21 : List Nat := 7549 :: root55Tail22
def root55Tail20 : List Nat := 7177 :: root55Tail21
def root55Tail19 : List Nat := 6733 :: root55Tail20
def root55Tail18 : List Nat := 6451 :: root55Tail19
def root55Tail17 : List Nat := 6163 :: root55Tail18
def root55Tail16 : List Nat := 5737 :: root55Tail17
def root55Tail15 : List Nat := 5701 :: root55Tail16
def root55Tail14 : List Nat := 4111 :: root55Tail15
def root55Tail13 : List Nat := 3541 :: root55Tail14
def root55Tail12 : List Nat := 3457 :: root55Tail13
def root55Tail11 : List Nat := 2971 :: root55Tail12
def root55Tail10 : List Nat := 2179 :: root55Tail11
def root55Tail9 : List Nat := 2131 :: root55Tail10
def root55Tail8 : List Nat := 2089 :: root55Tail9
def root55Tail7 : List Nat := 1873 :: root55Tail8
def root55Tail6 : List Nat := 1747 :: root55Tail7
def root55Tail5 : List Nat := 1567 :: root55Tail6
def root55Tail4 : List Nat := 1549 :: root55Tail5
def root55Tail3 : List Nat := 1399 :: root55Tail4
def root55Tail2 : List Nat := 1213 :: root55Tail3
def root55Tail1 : List Nat := 1087 :: root55Tail2
def root55Tail0 : List Nat := 409 :: root55Tail1
def root55Candidates0 : List Nat := [1873, 13183]
set_option maxRecDepth 8192 in
theorem root55Candidates0_checked : candidates edge 4 271 409 root55Tail1 winner.bound = root55Candidates0 := by
  unfold candidates
  have row : edge 409 = (SortedNatTree.contains FastRows.row409) := by
    change Data.row409.contains = (SortedNatTree.contains FastRows.row409)
    exact FastRows.row409_checked.symm
  rw [row]
  decide
theorem root55Step0 : scan edge (search edge 3) 4 [271] 271 root55Tail0 winner =
    scan edge (search edge 3) 4 [271] 271 root55Tail1 winner := by
  rw [root55Tail0, scan, if_neg (by decide), root55Candidates0_checked]
  change scan edge (search edge 3) 4 [271] 271 root55Tail1
    (if 3 ≤ root55Candidates0.length then search edge 3 [271, 409] root55Candidates0 680 winner else winner) = _
  rw [if_neg (by decide)]
def root55Candidates1 : List Nat := [2179, 6451, 15199]
set_option maxRecDepth 8192 in
theorem root55Candidates1_checked : candidates edge 4 271 1087 root55Tail2 winner.bound = root55Candidates1 := by
  unfold candidates
  have row : edge 1087 = (SortedNatTree.contains FastRows.row1087) := by
    change Data.row1087.contains = (SortedNatTree.contains FastRows.row1087)
    exact FastRows.row1087_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root55Child1 : search edge 3 [271, 1087] root55Candidates1 1358 winner = winner := by decide
theorem root55Step1 : scan edge (search edge 3) 4 [271] 271 root55Tail1 winner =
    scan edge (search edge 3) 4 [271] 271 root55Tail2 winner := by
  rw [root55Tail1, scan, if_neg (by decide), root55Candidates1_checked]
  change scan edge (search edge 3) 4 [271] 271 root55Tail2
    (if 3 ≤ root55Candidates1.length then search edge 3 [271, 1087] root55Candidates1 1358 winner else winner) = _
  rw [if_pos (by decide), root55Child1]
def root55Candidates2 : List Nat := [12109, 14071]
set_option maxRecDepth 8192 in
theorem root55Candidates2_checked : candidates edge 4 271 1213 root55Tail3 winner.bound = root55Candidates2 := by
  unfold candidates
  have row : edge 1213 = (SortedNatTree.contains FastRows.row1213) := by
    change Data.row1213.contains = (SortedNatTree.contains FastRows.row1213)
    exact FastRows.row1213_checked.symm
  rw [row]
  decide
theorem root55Step2 : scan edge (search edge 3) 4 [271] 271 root55Tail2 winner =
    scan edge (search edge 3) 4 [271] 271 root55Tail3 winner := by
  rw [root55Tail2, scan, if_neg (by decide), root55Candidates2_checked]
  change scan edge (search edge 3) 4 [271] 271 root55Tail3
    (if 3 ≤ root55Candidates2.length then search edge 3 [271, 1213] root55Candidates2 1484 winner else winner) = _
  rw [if_neg (by decide)]
def root55Candidates3 : List Nat := [2131, 16759]
set_option maxRecDepth 8192 in
theorem root55Candidates3_checked : candidates edge 4 271 1399 root55Tail4 winner.bound = root55Candidates3 := by
  unfold candidates
  have row : edge 1399 = (SortedNatTree.contains FastRows.row1399) := by
    change Data.row1399.contains = (SortedNatTree.contains FastRows.row1399)
    exact FastRows.row1399_checked.symm
  rw [row]
  decide
theorem root55Step3 : scan edge (search edge 3) 4 [271] 271 root55Tail3 winner =
    scan edge (search edge 3) 4 [271] 271 root55Tail4 winner := by
  rw [root55Tail3, scan, if_neg (by decide), root55Candidates3_checked]
  change scan edge (search edge 3) 4 [271] 271 root55Tail4
    (if 3 ≤ root55Candidates3.length then search edge 3 [271, 1399] root55Candidates3 1670 winner else winner) = _
  rw [if_neg (by decide)]
def root55Candidates4 : List Nat := [12343, 15031]
set_option maxRecDepth 8192 in
theorem root55Candidates4_checked : candidates edge 4 271 1549 root55Tail5 winner.bound = root55Candidates4 := by
  unfold candidates
  have row : edge 1549 = (SortedNatTree.contains FastRows.row1549) := by
    change Data.row1549.contains = (SortedNatTree.contains FastRows.row1549)
    exact FastRows.row1549_checked.symm
  rw [row]
  decide
theorem root55Step4 : scan edge (search edge 3) 4 [271] 271 root55Tail4 winner =
    scan edge (search edge 3) 4 [271] 271 root55Tail5 winner := by
  rw [root55Tail4, scan, if_neg (by decide), root55Candidates4_checked]
  change scan edge (search edge 3) 4 [271] 271 root55Tail5
    (if 3 ≤ root55Candidates4.length then search edge 3 [271, 1549] root55Candidates4 1820 winner else winner) = _
  rw [if_neg (by decide)]
def root55Candidates5 : List Nat := [13903, 17137, 19237]
set_option maxRecDepth 8192 in
theorem root55Candidates5_checked : candidates edge 4 271 1567 root55Tail6 winner.bound = root55Candidates5 := by
  unfold candidates
  have row : edge 1567 = (SortedNatTree.contains FastRows.row1567) := by
    change Data.row1567.contains = (SortedNatTree.contains FastRows.row1567)
    exact FastRows.row1567_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root55Child5 : search edge 3 [271, 1567] root55Candidates5 1838 winner = winner := by decide
theorem root55Step5 : scan edge (search edge 3) 4 [271] 271 root55Tail5 winner =
    scan edge (search edge 3) 4 [271] 271 root55Tail6 winner := by
  rw [root55Tail5, scan, if_neg (by decide), root55Candidates5_checked]
  change scan edge (search edge 3) 4 [271] 271 root55Tail6
    (if 3 ≤ root55Candidates5.length then search edge 3 [271, 1567] root55Candidates5 1838 winner else winner) = _
  rw [if_pos (by decide), root55Child5]
def root55Candidates6 : List Nat := [1873, 2179, 5737, 7177, 7639]
set_option maxRecDepth 8192 in
theorem root55Candidates6_checked : candidates edge 4 271 1747 root55Tail7 winner.bound = root55Candidates6 := by
  unfold candidates
  have row : edge 1747 = (SortedNatTree.contains FastRows.row1747) := by
    change Data.row1747.contains = (SortedNatTree.contains FastRows.row1747)
    exact FastRows.row1747_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root55Child6 : search edge 3 [271, 1747] root55Candidates6 2018 winner = winner := by decide
theorem root55Step6 : scan edge (search edge 3) 4 [271] 271 root55Tail6 winner =
    scan edge (search edge 3) 4 [271] 271 root55Tail7 winner := by
  rw [root55Tail6, scan, if_neg (by decide), root55Candidates6_checked]
  change scan edge (search edge 3) 4 [271] 271 root55Tail7
    (if 3 ≤ root55Candidates6.length then search edge 3 [271, 1747] root55Candidates6 2018 winner else winner) = _
  rw [if_pos (by decide), root55Child6]
def root55Candidates7 : List Nat := [9433]
set_option maxRecDepth 8192 in
theorem root55Candidates7_checked : candidates edge 4 271 1873 root55Tail8 winner.bound = root55Candidates7 := by
  unfold candidates
  have row : edge 1873 = (SortedNatTree.contains FastRows.row1873) := by
    change Data.row1873.contains = (SortedNatTree.contains FastRows.row1873)
    exact FastRows.row1873_checked.symm
  rw [row]
  decide
theorem root55Step7 : scan edge (search edge 3) 4 [271] 271 root55Tail7 winner =
    scan edge (search edge 3) 4 [271] 271 root55Tail8 winner := by
  rw [root55Tail7, scan, if_neg (by decide), root55Candidates7_checked]
  change scan edge (search edge 3) 4 [271] 271 root55Tail8
    (if 3 ≤ root55Candidates7.length then search edge 3 [271, 1873] root55Candidates7 2144 winner else winner) = _
  rw [if_neg (by decide)]
def root55Candidates8 : List Nat := [12037]
set_option maxRecDepth 8192 in
theorem root55Candidates8_checked : candidates edge 4 271 2089 root55Tail9 winner.bound = root55Candidates8 := by
  unfold candidates
  have row : edge 2089 = (SortedNatTree.contains FastRows.row2089) := by
    change Data.row2089.contains = (SortedNatTree.contains FastRows.row2089)
    exact FastRows.row2089_checked.symm
  rw [row]
  decide
theorem root55Step8 : scan edge (search edge 3) 4 [271] 271 root55Tail8 winner =
    scan edge (search edge 3) 4 [271] 271 root55Tail9 winner := by
  rw [root55Tail8, scan, if_neg (by decide), root55Candidates8_checked]
  change scan edge (search edge 3) 4 [271] 271 root55Tail9
    (if 3 ≤ root55Candidates8.length then search edge 3 [271, 2089] root55Candidates8 2360 winner else winner) = _
  rw [if_neg (by decide)]
def root55Candidates9 : List Nat := [3541]
set_option maxRecDepth 8192 in
theorem root55Candidates9_checked : candidates edge 4 271 2131 root55Tail10 winner.bound = root55Candidates9 := by
  unfold candidates
  have row : edge 2131 = (SortedNatTree.contains FastRows.row2131) := by
    change Data.row2131.contains = (SortedNatTree.contains FastRows.row2131)
    exact FastRows.row2131_checked.symm
  rw [row]
  decide
theorem root55Step9 : scan edge (search edge 3) 4 [271] 271 root55Tail9 winner =
    scan edge (search edge 3) 4 [271] 271 root55Tail10 winner := by
  rw [root55Tail9, scan, if_neg (by decide), root55Candidates9_checked]
  change scan edge (search edge 3) 4 [271] 271 root55Tail10
    (if 3 ≤ root55Candidates9.length then search edge 3 [271, 2131] root55Candidates9 2402 winner else winner) = _
  rw [if_neg (by decide)]
def root55Candidates10 : List Nat := [20719]
set_option maxRecDepth 8192 in
theorem root55Candidates10_checked : candidates edge 4 271 2179 root55Tail11 winner.bound = root55Candidates10 := by
  unfold candidates
  have row : edge 2179 = (SortedNatTree.contains FastRows.row2179) := by
    change Data.row2179.contains = (SortedNatTree.contains FastRows.row2179)
    exact FastRows.row2179_checked.symm
  rw [row]
  decide
theorem root55Step10 : scan edge (search edge 3) 4 [271] 271 root55Tail10 winner =
    scan edge (search edge 3) 4 [271] 271 root55Tail11 winner := by
  rw [root55Tail10, scan, if_neg (by decide), root55Candidates10_checked]
  change scan edge (search edge 3) 4 [271] 271 root55Tail11
    (if 3 ≤ root55Candidates10.length then search edge 3 [271, 2179] root55Candidates10 2450 winner else winner) = _
  rw [if_neg (by decide)]
def root55Candidates11 : List Nat := [13477]
set_option maxRecDepth 8192 in
theorem root55Candidates11_checked : candidates edge 4 271 2971 root55Tail12 winner.bound = root55Candidates11 := by
  unfold candidates
  have row : edge 2971 = (SortedNatTree.contains FastRows.row2971) := by
    change Data.row2971.contains = (SortedNatTree.contains FastRows.row2971)
    exact FastRows.row2971_checked.symm
  rw [row]
  decide
theorem root55Step11 : scan edge (search edge 3) 4 [271] 271 root55Tail11 winner =
    scan edge (search edge 3) 4 [271] 271 root55Tail12 winner := by
  rw [root55Tail11, scan, if_neg (by decide), root55Candidates11_checked]
  change scan edge (search edge 3) 4 [271] 271 root55Tail12
    (if 3 ≤ root55Candidates11.length then search edge 3 [271, 2971] root55Candidates11 3242 winner else winner) = _
  rw [if_neg (by decide)]
def root55Candidates12 : List Nat := [12109, 20983]
set_option maxRecDepth 8192 in
theorem root55Candidates12_checked : candidates edge 4 271 3457 root55Tail13 winner.bound = root55Candidates12 := by
  unfold candidates
  have row : edge 3457 = (SortedNatTree.contains FastRows.row3457) := by
    change Data.row3457.contains = (SortedNatTree.contains FastRows.row3457)
    exact FastRows.row3457_checked.symm
  rw [row]
  decide
theorem root55Step12 : scan edge (search edge 3) 4 [271] 271 root55Tail12 winner =
    scan edge (search edge 3) 4 [271] 271 root55Tail13 winner := by
  rw [root55Tail12, scan, if_neg (by decide), root55Candidates12_checked]
  change scan edge (search edge 3) 4 [271] 271 root55Tail13
    (if 3 ≤ root55Candidates12.length then search edge 3 [271, 3457] root55Candidates12 3728 winner else winner) = _
  rw [if_neg (by decide)]
def root55Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root55Candidates13_checked : candidates edge 4 271 3541 root55Tail14 winner.bound = root55Candidates13 := by
  unfold candidates
  have row : edge 3541 = (SortedNatTree.contains FastRows.row3541) := by
    change Data.row3541.contains = (SortedNatTree.contains FastRows.row3541)
    exact FastRows.row3541_checked.symm
  rw [row]
  decide
theorem root55Step13 : scan edge (search edge 3) 4 [271] 271 root55Tail13 winner =
    scan edge (search edge 3) 4 [271] 271 root55Tail14 winner := by
  rw [root55Tail13, scan, if_neg (by decide), root55Candidates13_checked]
  change scan edge (search edge 3) 4 [271] 271 root55Tail14
    (if 3 ≤ root55Candidates13.length then search edge 3 [271, 3541] root55Candidates13 3812 winner else winner) = _
  rw [if_neg (by decide)]
def root55Candidates14 : List Nat := [11353]
set_option maxRecDepth 8192 in
theorem root55Candidates14_checked : candidates edge 4 271 4111 root55Tail15 winner.bound = root55Candidates14 := by
  unfold candidates
  have row : edge 4111 = (SortedNatTree.contains FastRows.row4111) := by
    change Data.row4111.contains = (SortedNatTree.contains FastRows.row4111)
    exact FastRows.row4111_checked.symm
  rw [row]
  decide
theorem root55Step14 : scan edge (search edge 3) 4 [271] 271 root55Tail14 winner =
    scan edge (search edge 3) 4 [271] 271 root55Tail15 winner := by
  rw [root55Tail14, scan, if_neg (by decide), root55Candidates14_checked]
  change scan edge (search edge 3) 4 [271] 271 root55Tail15
    (if 3 ≤ root55Candidates14.length then search edge 3 [271, 4111] root55Candidates14 4382 winner else winner) = _
  rw [if_neg (by decide)]
def root55Candidates15 : List Nat := [6733, 13477, 17203]
set_option maxRecDepth 8192 in
theorem root55Candidates15_checked : candidates edge 4 271 5701 root55Tail16 winner.bound = root55Candidates15 := by
  unfold candidates
  have row : edge 5701 = (SortedNatTree.contains FastRows.row5701) := by
    change Data.row5701.contains = (SortedNatTree.contains FastRows.row5701)
    exact FastRows.row5701_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root55Child15 : search edge 3 [271, 5701] root55Candidates15 5972 winner = winner := by decide
theorem root55Step15 : scan edge (search edge 3) 4 [271] 271 root55Tail15 winner =
    scan edge (search edge 3) 4 [271] 271 root55Tail16 winner := by
  rw [root55Tail15, scan, if_neg (by decide), root55Candidates15_checked]
  change scan edge (search edge 3) 4 [271] 271 root55Tail16
    (if 3 ≤ root55Candidates15.length then search edge 3 [271, 5701] root55Candidates15 5972 winner else winner) = _
  rw [if_pos (by decide), root55Child15]
def root55Candidates16 : List Nat := [8941, 12511]
set_option maxRecDepth 8192 in
theorem root55Candidates16_checked : candidates edge 4 271 5737 root55Tail17 winner.bound = root55Candidates16 := by
  unfold candidates
  have row : edge 5737 = (SortedNatTree.contains FastRows.row5737) := by
    change Data.row5737.contains = (SortedNatTree.contains FastRows.row5737)
    exact FastRows.row5737_checked.symm
  rw [row]
  decide
theorem root55Step16 : scan edge (search edge 3) 4 [271] 271 root55Tail16 winner =
    scan edge (search edge 3) 4 [271] 271 root55Tail17 winner := by
  rw [root55Tail16, scan, if_neg (by decide), root55Candidates16_checked]
  change scan edge (search edge 3) 4 [271] 271 root55Tail17
    (if 3 ≤ root55Candidates16.length then search edge 3 [271, 5737] root55Candidates16 6008 winner else winner) = _
  rw [if_neg (by decide)]
def root55Candidates17 : List Nat := [6451, 8971, 13183]
set_option maxRecDepth 8192 in
theorem root55Candidates17_checked : candidates edge 4 271 6163 root55Tail18 winner.bound = root55Candidates17 := by
  unfold candidates
  have row : edge 6163 = (SortedNatTree.contains FastRows.row6163) := by
    change Data.row6163.contains = (SortedNatTree.contains FastRows.row6163)
    exact FastRows.row6163_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root55Child17 : search edge 3 [271, 6163] root55Candidates17 6434 winner = winner := by decide
theorem root55Step17 : scan edge (search edge 3) 4 [271] 271 root55Tail17 winner =
    scan edge (search edge 3) 4 [271] 271 root55Tail18 winner := by
  rw [root55Tail17, scan, if_neg (by decide), root55Candidates17_checked]
  change scan edge (search edge 3) 4 [271] 271 root55Tail18
    (if 3 ≤ root55Candidates17.length then search edge 3 [271, 6163] root55Candidates17 6434 winner else winner) = _
  rw [if_pos (by decide), root55Child17]
theorem root55Scan18 : scan edge (search edge 3) 4 [271] 271 root55Tail18 winner = winner := by rw [root55Tail18, scan, if_pos (by decide)]
theorem root55Scan17 : scan edge (search edge 3) 4 [271] 271 root55Tail17 winner = winner :=
  (root55Step17).trans root55Scan18
theorem root55Scan16 : scan edge (search edge 3) 4 [271] 271 root55Tail16 winner = winner :=
  (root55Step16).trans root55Scan17
theorem root55Scan15 : scan edge (search edge 3) 4 [271] 271 root55Tail15 winner = winner :=
  (root55Step15).trans root55Scan16
theorem root55Scan14 : scan edge (search edge 3) 4 [271] 271 root55Tail14 winner = winner :=
  (root55Step14).trans root55Scan15
theorem root55Scan13 : scan edge (search edge 3) 4 [271] 271 root55Tail13 winner = winner :=
  (root55Step13).trans root55Scan14
theorem root55Scan12 : scan edge (search edge 3) 4 [271] 271 root55Tail12 winner = winner :=
  (root55Step12).trans root55Scan13
theorem root55Scan11 : scan edge (search edge 3) 4 [271] 271 root55Tail11 winner = winner :=
  (root55Step11).trans root55Scan12
theorem root55Scan10 : scan edge (search edge 3) 4 [271] 271 root55Tail10 winner = winner :=
  (root55Step10).trans root55Scan11
theorem root55Scan9 : scan edge (search edge 3) 4 [271] 271 root55Tail9 winner = winner :=
  (root55Step9).trans root55Scan10
theorem root55Scan8 : scan edge (search edge 3) 4 [271] 271 root55Tail8 winner = winner :=
  (root55Step8).trans root55Scan9
theorem root55Scan7 : scan edge (search edge 3) 4 [271] 271 root55Tail7 winner = winner :=
  (root55Step7).trans root55Scan8
theorem root55Scan6 : scan edge (search edge 3) 4 [271] 271 root55Tail6 winner = winner :=
  (root55Step6).trans root55Scan7
theorem root55Scan5 : scan edge (search edge 3) 4 [271] 271 root55Tail5 winner = winner :=
  (root55Step5).trans root55Scan6
theorem root55Scan4 : scan edge (search edge 3) 4 [271] 271 root55Tail4 winner = winner :=
  (root55Step4).trans root55Scan5
theorem root55Scan3 : scan edge (search edge 3) 4 [271] 271 root55Tail3 winner = winner :=
  (root55Step3).trans root55Scan4
theorem root55Scan2 : scan edge (search edge 3) 4 [271] 271 root55Tail2 winner = winner :=
  (root55Step2).trans root55Scan3
theorem root55Scan1 : scan edge (search edge 3) 4 [271] 271 root55Tail1 winner = winner :=
  (root55Step1).trans root55Scan2
theorem root55Scan0 : scan edge (search edge 3) 4 [271] 271 root55Tail0 winner = winner :=
  (root55Step0).trans root55Scan1
set_option maxRecDepth 4096 in
theorem root55Branch : search edge 4 [271] root55Candidates 271 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [271] 271 root55Tail0 winner = winner
  exact root55Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
