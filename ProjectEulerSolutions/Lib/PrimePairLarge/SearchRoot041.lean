import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root41Candidates : List Nat := Data.row193
set_option maxRecDepth 8192 in
theorem root41Candidates_checked : candidates edge 5 0 193 tail42 winner.bound = root41Candidates := by
  apply candidates_eq_row edge 5 0 193 tail42 root41Candidates winner.bound tail42_ordered
  · decide
  · decide
  · funext q
    rfl
def root41Tail78 : List Nat := []
def root41Tail77 : List Nat := 25693 :: root41Tail78
def root41Tail76 : List Nat := 25171 :: root41Tail77
def root41Tail75 : List Nat := 25087 :: root41Tail76
def root41Tail74 : List Nat := 23761 :: root41Tail75
def root41Tail73 : List Nat := 23677 :: root41Tail74
def root41Tail72 : List Nat := 23371 :: root41Tail73
def root41Tail71 : List Nat := 22993 :: root41Tail72
def root41Tail70 : List Nat := 22549 :: root41Tail71
def root41Tail69 : List Nat := 21787 :: root41Tail70
def root41Tail68 : List Nat := 21493 :: root41Tail69
def root41Tail67 : List Nat := 21391 :: root41Tail68
def root41Tail66 : List Nat := 20233 :: root41Tail67
def root41Tail65 : List Nat := 20089 :: root41Tail66
def root41Tail64 : List Nat := 19819 :: root41Tail65
def root41Tail63 : List Nat := 18661 :: root41Tail64
def root41Tail62 : List Nat := 17911 :: root41Tail63
def root41Tail61 : List Nat := 17377 :: root41Tail62
def root41Tail60 : List Nat := 16567 :: root41Tail61
def root41Tail59 : List Nat := 16519 :: root41Tail60
def root41Tail58 : List Nat := 16057 :: root41Tail59
def root41Tail57 : List Nat := 15991 :: root41Tail58
def root41Tail56 : List Nat := 15679 :: root41Tail57
def root41Tail55 : List Nat := 15661 :: root41Tail56
def root41Tail54 : List Nat := 15619 :: root41Tail55
def root41Tail53 : List Nat := 14797 :: root41Tail54
def root41Tail52 : List Nat := 14437 :: root41Tail53
def root41Tail51 : List Nat := 14107 :: root41Tail52
def root41Tail50 : List Nat := 14083 :: root41Tail51
def root41Tail49 : List Nat := 13183 :: root41Tail50
def root41Tail48 : List Nat := 13033 :: root41Tail49
def root41Tail47 : List Nat := 12841 :: root41Tail48
def root41Tail46 : List Nat := 12823 :: root41Tail47
def root41Tail45 : List Nat := 12487 :: root41Tail46
def root41Tail44 : List Nat := 12211 :: root41Tail45
def root41Tail43 : List Nat := 12043 :: root41Tail44
def root41Tail42 : List Nat := 11833 :: root41Tail43
def root41Tail41 : List Nat := 11497 :: root41Tail42
def root41Tail40 : List Nat := 11119 :: root41Tail41
def root41Tail39 : List Nat := 10867 :: root41Tail40
def root41Tail38 : List Nat := 10267 :: root41Tail39
def root41Tail37 : List Nat := 10243 :: root41Tail38
def root41Tail36 : List Nat := 9439 :: root41Tail37
def root41Tail35 : List Nat := 9283 :: root41Tail36
def root41Tail34 : List Nat := 9151 :: root41Tail35
def root41Tail33 : List Nat := 8821 :: root41Tail34
def root41Tail32 : List Nat := 8623 :: root41Tail33
def root41Tail31 : List Nat := 8191 :: root41Tail32
def root41Tail30 : List Nat := 8161 :: root41Tail31
def root41Tail29 : List Nat := 7237 :: root41Tail30
def root41Tail28 : List Nat := 6733 :: root41Tail29
def root41Tail27 : List Nat := 6427 :: root41Tail28
def root41Tail26 : List Nat := 5827 :: root41Tail27
def root41Tail25 : List Nat := 4969 :: root41Tail26
def root41Tail24 : List Nat := 4657 :: root41Tail25
def root41Tail23 : List Nat := 3727 :: root41Tail24
def root41Tail22 : List Nat := 3499 :: root41Tail23
def root41Tail21 : List Nat := 3457 :: root41Tail22
def root41Tail20 : List Nat := 3037 :: root41Tail21
def root41Tail19 : List Nat := 2803 :: root41Tail20
def root41Tail18 : List Nat := 2719 :: root41Tail19
def root41Tail17 : List Nat := 2131 :: root41Tail18
def root41Tail16 : List Nat := 2011 :: root41Tail17
def root41Tail15 : List Nat := 1723 :: root41Tail16
def root41Tail14 : List Nat := 1669 :: root41Tail15
def root41Tail13 : List Nat := 1627 :: root41Tail14
def root41Tail12 : List Nat := 1399 :: root41Tail13
def root41Tail11 : List Nat := 1297 :: root41Tail12
def root41Tail10 : List Nat := 1291 :: root41Tail11
def root41Tail9 : List Nat := 1009 :: root41Tail10
def root41Tail8 : List Nat := 883 :: root41Tail9
def root41Tail7 : List Nat := 811 :: root41Tail8
def root41Tail6 : List Nat := 751 :: root41Tail7
def root41Tail5 : List Nat := 601 :: root41Tail6
def root41Tail4 : List Nat := 577 :: root41Tail5
def root41Tail3 : List Nat := 541 :: root41Tail4
def root41Tail2 : List Nat := 433 :: root41Tail3
def root41Tail1 : List Nat := 373 :: root41Tail2
def root41Tail0 : List Nat := 283 :: root41Tail1
def root41Candidates0 : List Nat := [541, 601, 6733, 8191]
set_option maxRecDepth 8192 in
theorem root41Candidates0_checked : candidates edge 4 193 283 root41Tail1 winner.bound = root41Candidates0 := by
  unfold candidates
  have row : edge 283 = (SortedNatTree.contains FastRows.row283) := by
    change Data.row283.contains = (SortedNatTree.contains FastRows.row283)
    exact FastRows.row283_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root41Child0 : search edge 3 [193, 283] root41Candidates0 476 winner = winner := by decide
theorem root41Step0 : scan edge (search edge 3) 4 [193] 193 root41Tail0 winner =
    scan edge (search edge 3) 4 [193] 193 root41Tail1 winner := by
  rw [root41Tail0, scan, if_neg (by decide), root41Candidates0_checked]
  change scan edge (search edge 3) 4 [193] 193 root41Tail1
    (if 3 ≤ root41Candidates0.length then search edge 3 [193, 283] root41Candidates0 476 winner else winner) = _
  rw [if_pos (by decide), root41Child0]
def root41Candidates1 : List Nat := [1009, 15679, 23371]
set_option maxRecDepth 8192 in
theorem root41Candidates1_checked : candidates edge 4 193 373 root41Tail2 winner.bound = root41Candidates1 := by
  unfold candidates
  have row : edge 373 = (SortedNatTree.contains FastRows.row373) := by
    change Data.row373.contains = (SortedNatTree.contains FastRows.row373)
    exact FastRows.row373_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root41Child1 : search edge 3 [193, 373] root41Candidates1 566 winner = winner := by decide
theorem root41Step1 : scan edge (search edge 3) 4 [193] 193 root41Tail1 winner =
    scan edge (search edge 3) 4 [193] 193 root41Tail2 winner := by
  rw [root41Tail1, scan, if_neg (by decide), root41Candidates1_checked]
  change scan edge (search edge 3) 4 [193] 193 root41Tail2
    (if 3 ≤ root41Candidates1.length then search edge 3 [193, 373] root41Candidates1 566 winner else winner) = _
  rw [if_pos (by decide), root41Child1]
def root41Candidates2 : List Nat := [883, 1009, 2719, 11833]
set_option maxRecDepth 8192 in
theorem root41Candidates2_checked : candidates edge 4 193 433 root41Tail3 winner.bound = root41Candidates2 := by
  unfold candidates
  have row : edge 433 = (SortedNatTree.contains FastRows.row433) := by
    change Data.row433.contains = (SortedNatTree.contains FastRows.row433)
    exact FastRows.row433_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root41Child2 : search edge 3 [193, 433] root41Candidates2 626 winner = winner := by decide
theorem root41Step2 : scan edge (search edge 3) 4 [193] 193 root41Tail2 winner =
    scan edge (search edge 3) 4 [193] 193 root41Tail3 winner := by
  rw [root41Tail2, scan, if_neg (by decide), root41Candidates2_checked]
  change scan edge (search edge 3) 4 [193] 193 root41Tail3
    (if 3 ≤ root41Candidates2.length then search edge 3 [193, 433] root41Candidates2 626 winner else winner) = _
  rw [if_pos (by decide), root41Child2]
def root41Candidates3 : List Nat := [9439, 12043, 16519, 17911]
set_option maxRecDepth 8192 in
theorem root41Candidates3_checked : candidates edge 4 193 541 root41Tail4 winner.bound = root41Candidates3 := by
  unfold candidates
  have row : edge 541 = (SortedNatTree.contains FastRows.row541) := by
    change Data.row541.contains = (SortedNatTree.contains FastRows.row541)
    exact FastRows.row541_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root41Child3 : search edge 3 [193, 541] root41Candidates3 734 winner = winner := by decide
theorem root41Step3 : scan edge (search edge 3) 4 [193] 193 root41Tail3 winner =
    scan edge (search edge 3) 4 [193] 193 root41Tail4 winner := by
  rw [root41Tail3, scan, if_neg (by decide), root41Candidates3_checked]
  change scan edge (search edge 3) 4 [193] 193 root41Tail4
    (if 3 ≤ root41Candidates3.length then search edge 3 [193, 541] root41Candidates3 734 winner else winner) = _
  rw [if_pos (by decide), root41Child3]
def root41Candidates4 : List Nat := [1723, 10267, 15661, 19819]
set_option maxRecDepth 8192 in
theorem root41Candidates4_checked : candidates edge 4 193 577 root41Tail5 winner.bound = root41Candidates4 := by
  unfold candidates
  have row : edge 577 = (SortedNatTree.contains FastRows.row577) := by
    change Data.row577.contains = (SortedNatTree.contains FastRows.row577)
    exact FastRows.row577_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root41Child4 : search edge 3 [193, 577] root41Candidates4 770 winner = winner := by decide
theorem root41Step4 : scan edge (search edge 3) 4 [193] 193 root41Tail4 winner =
    scan edge (search edge 3) 4 [193] 193 root41Tail5 winner := by
  rw [root41Tail4, scan, if_neg (by decide), root41Candidates4_checked]
  change scan edge (search edge 3) 4 [193] 193 root41Tail5
    (if 3 ≤ root41Candidates4.length then search edge 3 [193, 577] root41Candidates4 770 winner else winner) = _
  rw [if_pos (by decide), root41Child4]
def root41Candidates5 : List Nat := [1627, 2011, 2719, 12841]
set_option maxRecDepth 8192 in
theorem root41Candidates5_checked : candidates edge 4 193 601 root41Tail6 winner.bound = root41Candidates5 := by
  unfold candidates
  have row : edge 601 = (SortedNatTree.contains FastRows.row601) := by
    change Data.row601.contains = (SortedNatTree.contains FastRows.row601)
    exact FastRows.row601_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root41Child5 : search edge 3 [193, 601] root41Candidates5 794 winner = winner := by decide
theorem root41Step5 : scan edge (search edge 3) 4 [193] 193 root41Tail5 winner =
    scan edge (search edge 3) 4 [193] 193 root41Tail6 winner := by
  rw [root41Tail5, scan, if_neg (by decide), root41Candidates5_checked]
  change scan edge (search edge 3) 4 [193] 193 root41Tail6
    (if 3 ≤ root41Candidates5.length then search edge 3 [193, 601] root41Candidates5 794 winner else winner) = _
  rw [if_pos (by decide), root41Child5]
def root41Candidates6 : List Nat := [16057]
set_option maxRecDepth 8192 in
theorem root41Candidates6_checked : candidates edge 4 193 751 root41Tail7 winner.bound = root41Candidates6 := by
  unfold candidates
  have row : edge 751 = (SortedNatTree.contains FastRows.row751) := by
    change Data.row751.contains = (SortedNatTree.contains FastRows.row751)
    exact FastRows.row751_checked.symm
  rw [row]
  decide
theorem root41Step6 : scan edge (search edge 3) 4 [193] 193 root41Tail6 winner =
    scan edge (search edge 3) 4 [193] 193 root41Tail7 winner := by
  rw [root41Tail6, scan, if_neg (by decide), root41Candidates6_checked]
  change scan edge (search edge 3) 4 [193] 193 root41Tail7
    (if 3 ≤ root41Candidates6.length then search edge 3 [193, 751] root41Candidates6 944 winner else winner) = _
  rw [if_neg (by decide)]
def root41Candidates7 : List Nat := [1723, 2011, 8191, 12841]
set_option maxRecDepth 8192 in
theorem root41Candidates7_checked : candidates edge 4 193 811 root41Tail8 winner.bound = root41Candidates7 := by
  unfold candidates
  have row : edge 811 = (SortedNatTree.contains FastRows.row811) := by
    change Data.row811.contains = (SortedNatTree.contains FastRows.row811)
    exact FastRows.row811_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root41Child7 : search edge 3 [193, 811] root41Candidates7 1004 winner = winner := by decide
theorem root41Step7 : scan edge (search edge 3) 4 [193] 193 root41Tail7 winner =
    scan edge (search edge 3) 4 [193] 193 root41Tail8 winner := by
  rw [root41Tail7, scan, if_neg (by decide), root41Candidates7_checked]
  change scan edge (search edge 3) 4 [193] 193 root41Tail8
    (if 3 ≤ root41Candidates7.length then search edge 3 [193, 811] root41Candidates7 1004 winner else winner) = _
  rw [if_pos (by decide), root41Child7]
def root41Candidates8 : List Nat := [14797, 16567]
set_option maxRecDepth 8192 in
theorem root41Candidates8_checked : candidates edge 4 193 883 root41Tail9 winner.bound = root41Candidates8 := by
  unfold candidates
  have row : edge 883 = (SortedNatTree.contains FastRows.row883) := by
    change Data.row883.contains = (SortedNatTree.contains FastRows.row883)
    exact FastRows.row883_checked.symm
  rw [row]
  decide
theorem root41Step8 : scan edge (search edge 3) 4 [193] 193 root41Tail8 winner =
    scan edge (search edge 3) 4 [193] 193 root41Tail9 winner := by
  rw [root41Tail8, scan, if_neg (by decide), root41Candidates8_checked]
  change scan edge (search edge 3) 4 [193] 193 root41Tail9
    (if 3 ≤ root41Candidates8.length then search edge 3 [193, 883] root41Candidates8 1076 winner else winner) = _
  rw [if_neg (by decide)]
def root41Candidates9 : List Nat := [4657, 8821, 12841]
set_option maxRecDepth 8192 in
theorem root41Candidates9_checked : candidates edge 4 193 1009 root41Tail10 winner.bound = root41Candidates9 := by
  unfold candidates
  have row : edge 1009 = (SortedNatTree.contains FastRows.row1009) := by
    change Data.row1009.contains = (SortedNatTree.contains FastRows.row1009)
    exact FastRows.row1009_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root41Child9 : search edge 3 [193, 1009] root41Candidates9 1202 winner = winner := by decide
theorem root41Step9 : scan edge (search edge 3) 4 [193] 193 root41Tail9 winner =
    scan edge (search edge 3) 4 [193] 193 root41Tail10 winner := by
  rw [root41Tail9, scan, if_neg (by decide), root41Candidates9_checked]
  change scan edge (search edge 3) 4 [193] 193 root41Tail10
    (if 3 ≤ root41Candidates9.length then search edge 3 [193, 1009] root41Candidates9 1202 winner else winner) = _
  rw [if_pos (by decide), root41Child9]
def root41Candidates10 : List Nat := [3457, 6733, 10243, 15661, 22549]
set_option maxRecDepth 8192 in
theorem root41Candidates10_checked : candidates edge 4 193 1291 root41Tail11 winner.bound = root41Candidates10 := by
  unfold candidates
  have row : edge 1291 = (SortedNatTree.contains FastRows.row1291) := by
    change Data.row1291.contains = (SortedNatTree.contains FastRows.row1291)
    exact FastRows.row1291_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root41Child10 : search edge 3 [193, 1291] root41Candidates10 1484 winner = winner := by decide
theorem root41Step10 : scan edge (search edge 3) 4 [193] 193 root41Tail10 winner =
    scan edge (search edge 3) 4 [193] 193 root41Tail11 winner := by
  rw [root41Tail10, scan, if_neg (by decide), root41Candidates10_checked]
  change scan edge (search edge 3) 4 [193] 193 root41Tail11
    (if 3 ≤ root41Candidates10.length then search edge 3 [193, 1291] root41Candidates10 1484 winner else winner) = _
  rw [if_pos (by decide), root41Child10]
def root41Candidates11 : List Nat := [18661, 22993]
set_option maxRecDepth 8192 in
theorem root41Candidates11_checked : candidates edge 4 193 1297 root41Tail12 winner.bound = root41Candidates11 := by
  unfold candidates
  have row : edge 1297 = (SortedNatTree.contains FastRows.row1297) := by
    change Data.row1297.contains = (SortedNatTree.contains FastRows.row1297)
    exact FastRows.row1297_checked.symm
  rw [row]
  decide
theorem root41Step11 : scan edge (search edge 3) 4 [193] 193 root41Tail11 winner =
    scan edge (search edge 3) 4 [193] 193 root41Tail12 winner := by
  rw [root41Tail11, scan, if_neg (by decide), root41Candidates11_checked]
  change scan edge (search edge 3) 4 [193] 193 root41Tail12
    (if 3 ≤ root41Candidates11.length then search edge 3 [193, 1297] root41Candidates11 1490 winner else winner) = _
  rw [if_neg (by decide)]
def root41Candidates12 : List Nat := [2131]
set_option maxRecDepth 8192 in
theorem root41Candidates12_checked : candidates edge 4 193 1399 root41Tail13 winner.bound = root41Candidates12 := by
  unfold candidates
  have row : edge 1399 = (SortedNatTree.contains FastRows.row1399) := by
    change Data.row1399.contains = (SortedNatTree.contains FastRows.row1399)
    exact FastRows.row1399_checked.symm
  rw [row]
  decide
theorem root41Step12 : scan edge (search edge 3) 4 [193] 193 root41Tail12 winner =
    scan edge (search edge 3) 4 [193] 193 root41Tail13 winner := by
  rw [root41Tail12, scan, if_neg (by decide), root41Candidates12_checked]
  change scan edge (search edge 3) 4 [193] 193 root41Tail13
    (if 3 ≤ root41Candidates12.length then search edge 3 [193, 1399] root41Candidates12 1592 winner else winner) = _
  rw [if_neg (by decide)]
def root41Candidates13 : List Nat := [10267, 11833]
set_option maxRecDepth 8192 in
theorem root41Candidates13_checked : candidates edge 4 193 1627 root41Tail14 winner.bound = root41Candidates13 := by
  unfold candidates
  have row : edge 1627 = (SortedNatTree.contains FastRows.row1627) := by
    change Data.row1627.contains = (SortedNatTree.contains FastRows.row1627)
    exact FastRows.row1627_checked.symm
  rw [row]
  decide
theorem root41Step13 : scan edge (search edge 3) 4 [193] 193 root41Tail13 winner =
    scan edge (search edge 3) 4 [193] 193 root41Tail14 winner := by
  rw [root41Tail13, scan, if_neg (by decide), root41Candidates13_checked]
  change scan edge (search edge 3) 4 [193] 193 root41Tail14
    (if 3 ≤ root41Candidates13.length then search edge 3 [193, 1627] root41Candidates13 1820 winner else winner) = _
  rw [if_neg (by decide)]
def root41Candidates14 : List Nat := [3499, 6733, 11497]
set_option maxRecDepth 8192 in
theorem root41Candidates14_checked : candidates edge 4 193 1669 root41Tail15 winner.bound = root41Candidates14 := by
  unfold candidates
  have row : edge 1669 = (SortedNatTree.contains FastRows.row1669) := by
    change Data.row1669.contains = (SortedNatTree.contains FastRows.row1669)
    exact FastRows.row1669_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root41Child14 : search edge 3 [193, 1669] root41Candidates14 1862 winner = winner := by decide
theorem root41Step14 : scan edge (search edge 3) 4 [193] 193 root41Tail14 winner =
    scan edge (search edge 3) 4 [193] 193 root41Tail15 winner := by
  rw [root41Tail14, scan, if_neg (by decide), root41Candidates14_checked]
  change scan edge (search edge 3) 4 [193] 193 root41Tail15
    (if 3 ≤ root41Candidates14.length then search edge 3 [193, 1669] root41Candidates14 1862 winner else winner) = _
  rw [if_pos (by decide), root41Child14]
def root41Candidates15 : List Nat := [3457, 8623, 12211, 17377]
set_option maxRecDepth 8192 in
theorem root41Candidates15_checked : candidates edge 4 193 1723 root41Tail16 winner.bound = root41Candidates15 := by
  unfold candidates
  have row : edge 1723 = (SortedNatTree.contains FastRows.row1723) := by
    change Data.row1723.contains = (SortedNatTree.contains FastRows.row1723)
    exact FastRows.row1723_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root41Child15 : search edge 3 [193, 1723] root41Candidates15 1916 winner = winner := by decide
theorem root41Step15 : scan edge (search edge 3) 4 [193] 193 root41Tail15 winner =
    scan edge (search edge 3) 4 [193] 193 root41Tail16 winner := by
  rw [root41Tail15, scan, if_neg (by decide), root41Candidates15_checked]
  change scan edge (search edge 3) 4 [193] 193 root41Tail16
    (if 3 ≤ root41Candidates15.length then search edge 3 [193, 1723] root41Candidates15 1916 winner else winner) = _
  rw [if_pos (by decide), root41Child15]
def root41Candidates16 : List Nat := [13183, 18661]
set_option maxRecDepth 8192 in
theorem root41Candidates16_checked : candidates edge 4 193 2011 root41Tail17 winner.bound = root41Candidates16 := by
  unfold candidates
  have row : edge 2011 = (SortedNatTree.contains FastRows.row2011) := by
    change Data.row2011.contains = (SortedNatTree.contains FastRows.row2011)
    exact FastRows.row2011_checked.symm
  rw [row]
  decide
theorem root41Step16 : scan edge (search edge 3) 4 [193] 193 root41Tail16 winner =
    scan edge (search edge 3) 4 [193] 193 root41Tail17 winner := by
  rw [root41Tail16, scan, if_neg (by decide), root41Candidates16_checked]
  change scan edge (search edge 3) 4 [193] 193 root41Tail17
    (if 3 ≤ root41Candidates16.length then search edge 3 [193, 2011] root41Candidates16 2204 winner else winner) = _
  rw [if_neg (by decide)]
def root41Candidates17 : List Nat := [3037, 15619, 16567]
set_option maxRecDepth 8192 in
theorem root41Candidates17_checked : candidates edge 4 193 2131 root41Tail18 winner.bound = root41Candidates17 := by
  unfold candidates
  have row : edge 2131 = (SortedNatTree.contains FastRows.row2131) := by
    change Data.row2131.contains = (SortedNatTree.contains FastRows.row2131)
    exact FastRows.row2131_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root41Child17 : search edge 3 [193, 2131] root41Candidates17 2324 winner = winner := by decide
theorem root41Step17 : scan edge (search edge 3) 4 [193] 193 root41Tail17 winner =
    scan edge (search edge 3) 4 [193] 193 root41Tail18 winner := by
  rw [root41Tail17, scan, if_neg (by decide), root41Candidates17_checked]
  change scan edge (search edge 3) 4 [193] 193 root41Tail18
    (if 3 ≤ root41Candidates17.length then search edge 3 [193, 2131] root41Candidates17 2324 winner else winner) = _
  rw [if_pos (by decide), root41Child17]
def root41Candidates18 : List Nat := []
set_option maxRecDepth 8192 in
theorem root41Candidates18_checked : candidates edge 4 193 2719 root41Tail19 winner.bound = root41Candidates18 := by
  unfold candidates
  have row : edge 2719 = (SortedNatTree.contains FastRows.row2719) := by
    change Data.row2719.contains = (SortedNatTree.contains FastRows.row2719)
    exact FastRows.row2719_checked.symm
  rw [row]
  decide
theorem root41Step18 : scan edge (search edge 3) 4 [193] 193 root41Tail18 winner =
    scan edge (search edge 3) 4 [193] 193 root41Tail19 winner := by
  rw [root41Tail18, scan, if_neg (by decide), root41Candidates18_checked]
  change scan edge (search edge 3) 4 [193] 193 root41Tail19
    (if 3 ≤ root41Candidates18.length then search edge 3 [193, 2719] root41Candidates18 2912 winner else winner) = _
  rw [if_neg (by decide)]
def root41Candidates19 : List Nat := [4969, 6733]
set_option maxRecDepth 8192 in
theorem root41Candidates19_checked : candidates edge 4 193 2803 root41Tail20 winner.bound = root41Candidates19 := by
  unfold candidates
  have row : edge 2803 = (SortedNatTree.contains FastRows.row2803) := by
    change Data.row2803.contains = (SortedNatTree.contains FastRows.row2803)
    exact FastRows.row2803_checked.symm
  rw [row]
  decide
theorem root41Step19 : scan edge (search edge 3) 4 [193] 193 root41Tail19 winner =
    scan edge (search edge 3) 4 [193] 193 root41Tail20 winner := by
  rw [root41Tail19, scan, if_neg (by decide), root41Candidates19_checked]
  change scan edge (search edge 3) 4 [193] 193 root41Tail20
    (if 3 ≤ root41Candidates19.length then search edge 3 [193, 2803] root41Candidates19 2996 winner else winner) = _
  rw [if_neg (by decide)]
def root41Candidates20 : List Nat := [11119, 14797, 16057, 20089]
set_option maxRecDepth 8192 in
theorem root41Candidates20_checked : candidates edge 4 193 3037 root41Tail21 winner.bound = root41Candidates20 := by
  unfold candidates
  have row : edge 3037 = (SortedNatTree.contains FastRows.row3037) := by
    change Data.row3037.contains = (SortedNatTree.contains FastRows.row3037)
    exact FastRows.row3037_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root41Child20 : search edge 3 [193, 3037] root41Candidates20 3230 winner = winner := by decide
theorem root41Step20 : scan edge (search edge 3) 4 [193] 193 root41Tail20 winner =
    scan edge (search edge 3) 4 [193] 193 root41Tail21 winner := by
  rw [root41Tail20, scan, if_neg (by decide), root41Candidates20_checked]
  change scan edge (search edge 3) 4 [193] 193 root41Tail21
    (if 3 ≤ root41Candidates20.length then search edge 3 [193, 3037] root41Candidates20 3230 winner else winner) = _
  rw [if_pos (by decide), root41Child20]
def root41Candidates21 : List Nat := [3499, 13033]
set_option maxRecDepth 8192 in
theorem root41Candidates21_checked : candidates edge 4 193 3457 root41Tail22 winner.bound = root41Candidates21 := by
  unfold candidates
  have row : edge 3457 = (SortedNatTree.contains FastRows.row3457) := by
    change Data.row3457.contains = (SortedNatTree.contains FastRows.row3457)
    exact FastRows.row3457_checked.symm
  rw [row]
  decide
theorem root41Step21 : scan edge (search edge 3) 4 [193] 193 root41Tail21 winner =
    scan edge (search edge 3) 4 [193] 193 root41Tail22 winner := by
  rw [root41Tail21, scan, if_neg (by decide), root41Candidates21_checked]
  change scan edge (search edge 3) 4 [193] 193 root41Tail22
    (if 3 ≤ root41Candidates21.length then search edge 3 [193, 3457] root41Candidates21 3650 winner else winner) = _
  rw [if_neg (by decide)]
def root41Candidates22 : List Nat := [8191, 12487, 14437, 14797]
set_option maxRecDepth 8192 in
theorem root41Candidates22_checked : candidates edge 4 193 3499 root41Tail23 winner.bound = root41Candidates22 := by
  unfold candidates
  have row : edge 3499 = (SortedNatTree.contains FastRows.row3499) := by
    change Data.row3499.contains = (SortedNatTree.contains FastRows.row3499)
    exact FastRows.row3499_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root41Child22 : search edge 3 [193, 3499] root41Candidates22 3692 winner = winner := by decide
theorem root41Step22 : scan edge (search edge 3) 4 [193] 193 root41Tail22 winner =
    scan edge (search edge 3) 4 [193] 193 root41Tail23 winner := by
  rw [root41Tail22, scan, if_neg (by decide), root41Candidates22_checked]
  change scan edge (search edge 3) 4 [193] 193 root41Tail23
    (if 3 ≤ root41Candidates22.length then search edge 3 [193, 3499] root41Candidates22 3692 winner else winner) = _
  rw [if_pos (by decide), root41Child22]
def root41Candidates23 : List Nat := [8623]
set_option maxRecDepth 8192 in
theorem root41Candidates23_checked : candidates edge 4 193 3727 root41Tail24 winner.bound = root41Candidates23 := by
  unfold candidates
  have row : edge 3727 = (SortedNatTree.contains FastRows.row3727) := by
    change Data.row3727.contains = (SortedNatTree.contains FastRows.row3727)
    exact FastRows.row3727_checked.symm
  rw [row]
  decide
theorem root41Step23 : scan edge (search edge 3) 4 [193] 193 root41Tail23 winner =
    scan edge (search edge 3) 4 [193] 193 root41Tail24 winner := by
  rw [root41Tail23, scan, if_neg (by decide), root41Candidates23_checked]
  change scan edge (search edge 3) 4 [193] 193 root41Tail24
    (if 3 ≤ root41Candidates23.length then search edge 3 [193, 3727] root41Candidates23 3920 winner else winner) = _
  rw [if_neg (by decide)]
def root41Candidates24 : List Nat := [9283, 14797, 16057]
set_option maxRecDepth 8192 in
theorem root41Candidates24_checked : candidates edge 4 193 4657 root41Tail25 winner.bound = root41Candidates24 := by
  unfold candidates
  have row : edge 4657 = (SortedNatTree.contains FastRows.row4657) := by
    change Data.row4657.contains = (SortedNatTree.contains FastRows.row4657)
    exact FastRows.row4657_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root41Child24 : search edge 3 [193, 4657] root41Candidates24 4850 winner = winner := by decide
theorem root41Step24 : scan edge (search edge 3) 4 [193] 193 root41Tail24 winner =
    scan edge (search edge 3) 4 [193] 193 root41Tail25 winner := by
  rw [root41Tail24, scan, if_neg (by decide), root41Candidates24_checked]
  change scan edge (search edge 3) 4 [193] 193 root41Tail25
    (if 3 ≤ root41Candidates24.length then search edge 3 [193, 4657] root41Candidates24 4850 winner else winner) = _
  rw [if_pos (by decide), root41Child24]
def root41Candidates25 : List Nat := [5827, 11497, 12487]
set_option maxRecDepth 8192 in
theorem root41Candidates25_checked : candidates edge 4 193 4969 root41Tail26 winner.bound = root41Candidates25 := by
  unfold candidates
  have row : edge 4969 = (SortedNatTree.contains FastRows.row4969) := by
    change Data.row4969.contains = (SortedNatTree.contains FastRows.row4969)
    exact FastRows.row4969_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root41Child25 : search edge 3 [193, 4969] root41Candidates25 5162 winner = winner := by decide
theorem root41Step25 : scan edge (search edge 3) 4 [193] 193 root41Tail25 winner =
    scan edge (search edge 3) 4 [193] 193 root41Tail26 winner := by
  rw [root41Tail25, scan, if_neg (by decide), root41Candidates25_checked]
  change scan edge (search edge 3) 4 [193] 193 root41Tail26
    (if 3 ≤ root41Candidates25.length then search edge 3 [193, 4969] root41Candidates25 5162 winner else winner) = _
  rw [if_pos (by decide), root41Child25]
def root41Candidates26 : List Nat := [14437]
set_option maxRecDepth 8192 in
theorem root41Candidates26_checked : candidates edge 4 193 5827 root41Tail27 winner.bound = root41Candidates26 := by
  unfold candidates
  have row : edge 5827 = (SortedNatTree.contains FastRows.row5827) := by
    change Data.row5827.contains = (SortedNatTree.contains FastRows.row5827)
    exact FastRows.row5827_checked.symm
  rw [row]
  decide
theorem root41Step26 : scan edge (search edge 3) 4 [193] 193 root41Tail26 winner =
    scan edge (search edge 3) 4 [193] 193 root41Tail27 winner := by
  rw [root41Tail26, scan, if_neg (by decide), root41Candidates26_checked]
  change scan edge (search edge 3) 4 [193] 193 root41Tail27
    (if 3 ≤ root41Candidates26.length then search edge 3 [193, 5827] root41Candidates26 6020 winner else winner) = _
  rw [if_neg (by decide)]
def root41Candidates27 : List Nat := [12211]
set_option maxRecDepth 8192 in
theorem root41Candidates27_checked : candidates edge 4 193 6427 root41Tail28 winner.bound = root41Candidates27 := by
  unfold candidates
  have row : edge 6427 = (SortedNatTree.contains FastRows.row6427) := by
    change Data.row6427.contains = (SortedNatTree.contains FastRows.row6427)
    exact FastRows.row6427_checked.symm
  rw [row]
  decide
theorem root41Step27 : scan edge (search edge 3) 4 [193] 193 root41Tail27 winner =
    scan edge (search edge 3) 4 [193] 193 root41Tail28 winner := by
  rw [root41Tail27, scan, if_neg (by decide), root41Candidates27_checked]
  change scan edge (search edge 3) 4 [193] 193 root41Tail28
    (if 3 ≤ root41Candidates27.length then search edge 3 [193, 6427] root41Candidates27 6620 winner else winner) = _
  rw [if_neg (by decide)]
theorem root41Scan28 : scan edge (search edge 3) 4 [193] 193 root41Tail28 winner = winner := by rw [root41Tail28, scan, if_pos (by decide)]
theorem root41Scan27 : scan edge (search edge 3) 4 [193] 193 root41Tail27 winner = winner :=
  (root41Step27).trans root41Scan28
theorem root41Scan26 : scan edge (search edge 3) 4 [193] 193 root41Tail26 winner = winner :=
  (root41Step26).trans root41Scan27
theorem root41Scan25 : scan edge (search edge 3) 4 [193] 193 root41Tail25 winner = winner :=
  (root41Step25).trans root41Scan26
theorem root41Scan24 : scan edge (search edge 3) 4 [193] 193 root41Tail24 winner = winner :=
  (root41Step24).trans root41Scan25
theorem root41Scan23 : scan edge (search edge 3) 4 [193] 193 root41Tail23 winner = winner :=
  (root41Step23).trans root41Scan24
theorem root41Scan22 : scan edge (search edge 3) 4 [193] 193 root41Tail22 winner = winner :=
  (root41Step22).trans root41Scan23
theorem root41Scan21 : scan edge (search edge 3) 4 [193] 193 root41Tail21 winner = winner :=
  (root41Step21).trans root41Scan22
theorem root41Scan20 : scan edge (search edge 3) 4 [193] 193 root41Tail20 winner = winner :=
  (root41Step20).trans root41Scan21
theorem root41Scan19 : scan edge (search edge 3) 4 [193] 193 root41Tail19 winner = winner :=
  (root41Step19).trans root41Scan20
theorem root41Scan18 : scan edge (search edge 3) 4 [193] 193 root41Tail18 winner = winner :=
  (root41Step18).trans root41Scan19
theorem root41Scan17 : scan edge (search edge 3) 4 [193] 193 root41Tail17 winner = winner :=
  (root41Step17).trans root41Scan18
theorem root41Scan16 : scan edge (search edge 3) 4 [193] 193 root41Tail16 winner = winner :=
  (root41Step16).trans root41Scan17
theorem root41Scan15 : scan edge (search edge 3) 4 [193] 193 root41Tail15 winner = winner :=
  (root41Step15).trans root41Scan16
theorem root41Scan14 : scan edge (search edge 3) 4 [193] 193 root41Tail14 winner = winner :=
  (root41Step14).trans root41Scan15
theorem root41Scan13 : scan edge (search edge 3) 4 [193] 193 root41Tail13 winner = winner :=
  (root41Step13).trans root41Scan14
theorem root41Scan12 : scan edge (search edge 3) 4 [193] 193 root41Tail12 winner = winner :=
  (root41Step12).trans root41Scan13
theorem root41Scan11 : scan edge (search edge 3) 4 [193] 193 root41Tail11 winner = winner :=
  (root41Step11).trans root41Scan12
theorem root41Scan10 : scan edge (search edge 3) 4 [193] 193 root41Tail10 winner = winner :=
  (root41Step10).trans root41Scan11
theorem root41Scan9 : scan edge (search edge 3) 4 [193] 193 root41Tail9 winner = winner :=
  (root41Step9).trans root41Scan10
theorem root41Scan8 : scan edge (search edge 3) 4 [193] 193 root41Tail8 winner = winner :=
  (root41Step8).trans root41Scan9
theorem root41Scan7 : scan edge (search edge 3) 4 [193] 193 root41Tail7 winner = winner :=
  (root41Step7).trans root41Scan8
theorem root41Scan6 : scan edge (search edge 3) 4 [193] 193 root41Tail6 winner = winner :=
  (root41Step6).trans root41Scan7
theorem root41Scan5 : scan edge (search edge 3) 4 [193] 193 root41Tail5 winner = winner :=
  (root41Step5).trans root41Scan6
theorem root41Scan4 : scan edge (search edge 3) 4 [193] 193 root41Tail4 winner = winner :=
  (root41Step4).trans root41Scan5
theorem root41Scan3 : scan edge (search edge 3) 4 [193] 193 root41Tail3 winner = winner :=
  (root41Step3).trans root41Scan4
theorem root41Scan2 : scan edge (search edge 3) 4 [193] 193 root41Tail2 winner = winner :=
  (root41Step2).trans root41Scan3
theorem root41Scan1 : scan edge (search edge 3) 4 [193] 193 root41Tail1 winner = winner :=
  (root41Step1).trans root41Scan2
theorem root41Scan0 : scan edge (search edge 3) 4 [193] 193 root41Tail0 winner = winner :=
  (root41Step0).trans root41Scan1
set_option maxRecDepth 4096 in
theorem root41Branch : search edge 4 [193] root41Candidates 193 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [193] 193 root41Tail0 winner = winner
  exact root41Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
