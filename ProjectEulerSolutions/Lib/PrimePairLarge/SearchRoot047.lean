import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root47Candidates : List Nat := Data.row229
set_option maxRecDepth 8192 in
theorem root47Candidates_checked : candidates edge 5 0 229 tail48 winner.bound = root47Candidates := by
  apply candidates_eq_row edge 5 0 229 tail48 root47Candidates winner.bound tail48_ordered
  · decide
  · decide
  · funext q
    rfl
def root47Tail70 : List Nat := []
def root47Tail69 : List Nat := 25339 :: root47Tail70
def root47Tail68 : List Nat := 24481 :: root47Tail69
def root47Tail67 : List Nat := 24247 :: root47Tail68
def root47Tail66 : List Nat := 24061 :: root47Tail67
def root47Tail65 : List Nat := 23011 :: root47Tail66
def root47Tail64 : List Nat := 22639 :: root47Tail65
def root47Tail63 : List Nat := 21673 :: root47Tail64
def root47Tail62 : List Nat := 20983 :: root47Tail63
def root47Tail61 : List Nat := 20551 :: root47Tail62
def root47Tail60 : List Nat := 20089 :: root47Tail61
def root47Tail59 : List Nat := 19777 :: root47Tail60
def root47Tail58 : List Nat := 18433 :: root47Tail59
def root47Tail57 : List Nat := 17929 :: root47Tail58
def root47Tail56 : List Nat := 17749 :: root47Tail57
def root47Tail55 : List Nat := 17443 :: root47Tail56
def root47Tail54 : List Nat := 17299 :: root47Tail55
def root47Tail53 : List Nat := 17107 :: root47Tail54
def root47Tail52 : List Nat := 16633 :: root47Tail53
def root47Tail51 : List Nat := 16519 :: root47Tail52
def root47Tail50 : List Nat := 16477 :: root47Tail51
def root47Tail49 : List Nat := 16339 :: root47Tail50
def root47Tail48 : List Nat := 15859 :: root47Tail49
def root47Tail47 : List Nat := 15817 :: root47Tail48
def root47Tail46 : List Nat := 15787 :: root47Tail47
def root47Tail45 : List Nat := 15733 :: root47Tail46
def root47Tail44 : List Nat := 15511 :: root47Tail45
def root47Tail43 : List Nat := 15289 :: root47Tail44
def root47Tail42 : List Nat := 14431 :: root47Tail43
def root47Tail41 : List Nat := 14149 :: root47Tail42
def root47Tail40 : List Nat := 14083 :: root47Tail41
def root47Tail39 : List Nat := 13873 :: root47Tail40
def root47Tail38 : List Nat := 12511 :: root47Tail39
def root47Tail37 : List Nat := 12037 :: root47Tail38
def root47Tail36 : List Nat := 11491 :: root47Tail37
def root47Tail35 : List Nat := 10453 :: root47Tail36
def root47Tail34 : List Nat := 10243 :: root47Tail35
def root47Tail33 : List Nat := 9907 :: root47Tail34
def root47Tail32 : List Nat := 9697 :: root47Tail33
def root47Tail31 : List Nat := 9463 :: root47Tail32
def root47Tail30 : List Nat := 9001 :: root47Tail31
def root47Tail29 : List Nat := 7759 :: root47Tail30
def root47Tail28 : List Nat := 7591 :: root47Tail29
def root47Tail27 : List Nat := 7213 :: root47Tail28
def root47Tail26 : List Nat := 6733 :: root47Tail27
def root47Tail25 : List Nat := 6247 :: root47Tail26
def root47Tail24 : List Nat := 5869 :: root47Tail25
def root47Tail23 : List Nat := 5659 :: root47Tail24
def root47Tail22 : List Nat := 5569 :: root47Tail23
def root47Tail21 : List Nat := 5077 :: root47Tail22
def root47Tail20 : List Nat := 4021 :: root47Tail21
def root47Tail19 : List Nat := 3919 :: root47Tail20
def root47Tail18 : List Nat := 3559 :: root47Tail19
def root47Tail17 : List Nat := 3541 :: root47Tail18
def root47Tail16 : List Nat := 3391 :: root47Tail17
def root47Tail15 : List Nat := 3301 :: root47Tail16
def root47Tail14 : List Nat := 3121 :: root47Tail15
def root47Tail13 : List Nat := 2767 :: root47Tail14
def root47Tail12 : List Nat := 2347 :: root47Tail13
def root47Tail11 : List Nat := 2239 :: root47Tail12
def root47Tail10 : List Nat := 2221 :: root47Tail11
def root47Tail9 : List Nat := 2083 :: root47Tail10
def root47Tail8 : List Nat := 1801 :: root47Tail9
def root47Tail7 : List Nat := 1231 :: root47Tail8
def root47Tail6 : List Nat := 937 :: root47Tail7
def root47Tail5 : List Nat := 631 :: root47Tail6
def root47Tail4 : List Nat := 613 :: root47Tail5
def root47Tail3 : List Nat := 547 :: root47Tail4
def root47Tail2 : List Nat := 499 :: root47Tail3
def root47Tail1 : List Nat := 433 :: root47Tail2
def root47Tail0 : List Nat := 373 :: root47Tail1
def root47Candidates0 : List Nat := [937, 16633, 17443]
set_option maxRecDepth 8192 in
theorem root47Candidates0_checked : candidates edge 4 229 373 root47Tail1 winner.bound = root47Candidates0 := by
  unfold candidates
  have row : edge 373 = (SortedNatTree.contains FastRows.row373) := by
    change Data.row373.contains = (SortedNatTree.contains FastRows.row373)
    exact FastRows.row373_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root47Child0 : search edge 3 [229, 373] root47Candidates0 602 winner = winner := by decide
theorem root47Step0 : scan edge (search edge 3) 4 [229] 229 root47Tail0 winner =
    scan edge (search edge 3) 4 [229] 229 root47Tail1 winner := by
  rw [root47Tail0, scan, if_neg (by decide), root47Candidates0_checked]
  change scan edge (search edge 3) 4 [229] 229 root47Tail1
    (if 3 ≤ root47Candidates0.length then search edge 3 [229, 373] root47Candidates0 602 winner else winner) = _
  rw [if_pos (by decide), root47Child0]
def root47Candidates1 : List Nat := [1801, 3391, 6247, 18433, 24481]
set_option maxRecDepth 8192 in
theorem root47Candidates1_checked : candidates edge 4 229 433 root47Tail2 winner.bound = root47Candidates1 := by
  unfold candidates
  have row : edge 433 = (SortedNatTree.contains FastRows.row433) := by
    change Data.row433.contains = (SortedNatTree.contains FastRows.row433)
    exact FastRows.row433_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root47Child1 : search edge 3 [229, 433] root47Candidates1 662 winner = winner := by decide
theorem root47Step1 : scan edge (search edge 3) 4 [229] 229 root47Tail1 winner =
    scan edge (search edge 3) 4 [229] 229 root47Tail2 winner := by
  rw [root47Tail1, scan, if_neg (by decide), root47Candidates1_checked]
  change scan edge (search edge 3) 4 [229] 229 root47Tail2
    (if 3 ≤ root47Candidates1.length then search edge 3 [229, 433] root47Candidates1 662 winner else winner) = _
  rw [if_pos (by decide), root47Child1]
def root47Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root47Candidates2_checked : candidates edge 4 229 499 root47Tail3 winner.bound = root47Candidates2 := by
  unfold candidates
  have row : edge 499 = (SortedNatTree.contains FastRows.row499) := by
    change Data.row499.contains = (SortedNatTree.contains FastRows.row499)
    exact FastRows.row499_checked.symm
  rw [row]
  decide
theorem root47Step2 : scan edge (search edge 3) 4 [229] 229 root47Tail2 winner =
    scan edge (search edge 3) 4 [229] 229 root47Tail3 winner := by
  rw [root47Tail2, scan, if_neg (by decide), root47Candidates2_checked]
  change scan edge (search edge 3) 4 [229] 229 root47Tail3
    (if 3 ≤ root47Candidates2.length then search edge 3 [229, 499] root47Candidates2 728 winner else winner) = _
  rw [if_neg (by decide)]
def root47Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root47Candidates3_checked : candidates edge 4 229 547 root47Tail4 winner.bound = root47Candidates3 := by
  unfold candidates
  have row : edge 547 = (SortedNatTree.contains FastRows.row547) := by
    change Data.row547.contains = (SortedNatTree.contains FastRows.row547)
    exact FastRows.row547_checked.symm
  rw [row]
  decide
theorem root47Step3 : scan edge (search edge 3) 4 [229] 229 root47Tail3 winner =
    scan edge (search edge 3) 4 [229] 229 root47Tail4 winner := by
  rw [root47Tail3, scan, if_neg (by decide), root47Candidates3_checked]
  change scan edge (search edge 3) 4 [229] 229 root47Tail4
    (if 3 ≤ root47Candidates3.length then search edge 3 [229, 547] root47Candidates3 776 winner else winner) = _
  rw [if_neg (by decide)]
def root47Candidates4 : List Nat := [4021, 6733, 20089, 23011]
set_option maxRecDepth 8192 in
theorem root47Candidates4_checked : candidates edge 4 229 613 root47Tail5 winner.bound = root47Candidates4 := by
  unfold candidates
  have row : edge 613 = (SortedNatTree.contains FastRows.row613) := by
    change Data.row613.contains = (SortedNatTree.contains FastRows.row613)
    exact FastRows.row613_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root47Child4 : search edge 3 [229, 613] root47Candidates4 842 winner = winner := by decide
theorem root47Step4 : scan edge (search edge 3) 4 [229] 229 root47Tail4 winner =
    scan edge (search edge 3) 4 [229] 229 root47Tail5 winner := by
  rw [root47Tail4, scan, if_neg (by decide), root47Candidates4_checked]
  change scan edge (search edge 3) 4 [229] 229 root47Tail5
    (if 3 ≤ root47Candidates4.length then search edge 3 [229, 613] root47Candidates4 842 winner else winner) = _
  rw [if_pos (by decide), root47Child4]
def root47Candidates5 : List Nat := [2221, 9463, 12037]
set_option maxRecDepth 8192 in
theorem root47Candidates5_checked : candidates edge 4 229 631 root47Tail6 winner.bound = root47Candidates5 := by
  unfold candidates
  have row : edge 631 = (SortedNatTree.contains FastRows.row631) := by
    change Data.row631.contains = (SortedNatTree.contains FastRows.row631)
    exact FastRows.row631_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root47Child5 : search edge 3 [229, 631] root47Candidates5 860 winner = winner := by decide
theorem root47Step5 : scan edge (search edge 3) 4 [229] 229 root47Tail5 winner =
    scan edge (search edge 3) 4 [229] 229 root47Tail6 winner := by
  rw [root47Tail5, scan, if_neg (by decide), root47Candidates5_checked]
  change scan edge (search edge 3) 4 [229] 229 root47Tail6
    (if 3 ≤ root47Candidates5.length then search edge 3 [229, 631] root47Candidates5 860 winner else winner) = _
  rw [if_pos (by decide), root47Child5]
def root47Candidates6 : List Nat := [7213, 16339]
set_option maxRecDepth 8192 in
theorem root47Candidates6_checked : candidates edge 4 229 937 root47Tail7 winner.bound = root47Candidates6 := by
  unfold candidates
  have row : edge 937 = (SortedNatTree.contains FastRows.row937) := by
    change Data.row937.contains = (SortedNatTree.contains FastRows.row937)
    exact FastRows.row937_checked.symm
  rw [row]
  decide
theorem root47Step6 : scan edge (search edge 3) 4 [229] 229 root47Tail6 winner =
    scan edge (search edge 3) 4 [229] 229 root47Tail7 winner := by
  rw [root47Tail6, scan, if_neg (by decide), root47Candidates6_checked]
  change scan edge (search edge 3) 4 [229] 229 root47Tail7
    (if 3 ≤ root47Candidates6.length then search edge 3 [229, 937] root47Candidates6 1166 winner else winner) = _
  rw [if_neg (by decide)]
def root47Candidates7 : List Nat := [5659]
set_option maxRecDepth 8192 in
theorem root47Candidates7_checked : candidates edge 4 229 1231 root47Tail8 winner.bound = root47Candidates7 := by
  unfold candidates
  have row : edge 1231 = (SortedNatTree.contains FastRows.row1231) := by
    change Data.row1231.contains = (SortedNatTree.contains FastRows.row1231)
    exact FastRows.row1231_checked.symm
  rw [row]
  decide
theorem root47Step7 : scan edge (search edge 3) 4 [229] 229 root47Tail7 winner =
    scan edge (search edge 3) 4 [229] 229 root47Tail8 winner := by
  rw [root47Tail7, scan, if_neg (by decide), root47Candidates7_checked]
  change scan edge (search edge 3) 4 [229] 229 root47Tail8
    (if 3 ≤ root47Candidates7.length then search edge 3 [229, 1231] root47Candidates7 1460 winner else winner) = _
  rw [if_neg (by decide)]
def root47Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root47Candidates8_checked : candidates edge 4 229 1801 root47Tail9 winner.bound = root47Candidates8 := by
  unfold candidates
  have row : edge 1801 = (SortedNatTree.contains FastRows.row1801) := by
    change Data.row1801.contains = (SortedNatTree.contains FastRows.row1801)
    exact FastRows.row1801_checked.symm
  rw [row]
  decide
theorem root47Step8 : scan edge (search edge 3) 4 [229] 229 root47Tail8 winner =
    scan edge (search edge 3) 4 [229] 229 root47Tail9 winner := by
  rw [root47Tail8, scan, if_neg (by decide), root47Candidates8_checked]
  change scan edge (search edge 3) 4 [229] 229 root47Tail9
    (if 3 ≤ root47Candidates8.length then search edge 3 [229, 1801] root47Candidates8 2030 winner else winner) = _
  rw [if_neg (by decide)]
def root47Candidates9 : List Nat := [14149]
set_option maxRecDepth 8192 in
theorem root47Candidates9_checked : candidates edge 4 229 2083 root47Tail10 winner.bound = root47Candidates9 := by
  unfold candidates
  have row : edge 2083 = (SortedNatTree.contains FastRows.row2083) := by
    change Data.row2083.contains = (SortedNatTree.contains FastRows.row2083)
    exact FastRows.row2083_checked.symm
  rw [row]
  decide
theorem root47Step9 : scan edge (search edge 3) 4 [229] 229 root47Tail9 winner =
    scan edge (search edge 3) 4 [229] 229 root47Tail10 winner := by
  rw [root47Tail9, scan, if_neg (by decide), root47Candidates9_checked]
  change scan edge (search edge 3) 4 [229] 229 root47Tail10
    (if 3 ≤ root47Candidates9.length then search edge 3 [229, 2083] root47Candidates9 2312 winner else winner) = _
  rw [if_neg (by decide)]
def root47Candidates10 : List Nat := [9907, 14083]
set_option maxRecDepth 8192 in
theorem root47Candidates10_checked : candidates edge 4 229 2221 root47Tail11 winner.bound = root47Candidates10 := by
  unfold candidates
  have row : edge 2221 = (SortedNatTree.contains FastRows.row2221) := by
    change Data.row2221.contains = (SortedNatTree.contains FastRows.row2221)
    exact FastRows.row2221_checked.symm
  rw [row]
  decide
theorem root47Step10 : scan edge (search edge 3) 4 [229] 229 root47Tail10 winner =
    scan edge (search edge 3) 4 [229] 229 root47Tail11 winner := by
  rw [root47Tail10, scan, if_neg (by decide), root47Candidates10_checked]
  change scan edge (search edge 3) 4 [229] 229 root47Tail11
    (if 3 ≤ root47Candidates10.length then search edge 3 [229, 2221] root47Candidates10 2450 winner else winner) = _
  rw [if_neg (by decide)]
def root47Candidates11 : List Nat := [3919]
set_option maxRecDepth 8192 in
theorem root47Candidates11_checked : candidates edge 4 229 2239 root47Tail12 winner.bound = root47Candidates11 := by
  unfold candidates
  have row : edge 2239 = (SortedNatTree.contains FastRows.row2239) := by
    change Data.row2239.contains = (SortedNatTree.contains FastRows.row2239)
    exact FastRows.row2239_checked.symm
  rw [row]
  decide
theorem root47Step11 : scan edge (search edge 3) 4 [229] 229 root47Tail11 winner =
    scan edge (search edge 3) 4 [229] 229 root47Tail12 winner := by
  rw [root47Tail11, scan, if_neg (by decide), root47Candidates11_checked]
  change scan edge (search edge 3) 4 [229] 229 root47Tail12
    (if 3 ≤ root47Candidates11.length then search edge 3 [229, 2239] root47Candidates11 2468 winner else winner) = _
  rw [if_neg (by decide)]
def root47Candidates12 : List Nat := [15289]
set_option maxRecDepth 8192 in
theorem root47Candidates12_checked : candidates edge 4 229 2347 root47Tail13 winner.bound = root47Candidates12 := by
  unfold candidates
  have row : edge 2347 = (SortedNatTree.contains FastRows.row2347) := by
    change Data.row2347.contains = (SortedNatTree.contains FastRows.row2347)
    exact FastRows.row2347_checked.symm
  rw [row]
  decide
theorem root47Step12 : scan edge (search edge 3) 4 [229] 229 root47Tail12 winner =
    scan edge (search edge 3) 4 [229] 229 root47Tail13 winner := by
  rw [root47Tail12, scan, if_neg (by decide), root47Candidates12_checked]
  change scan edge (search edge 3) 4 [229] 229 root47Tail13
    (if 3 ≤ root47Candidates12.length then search edge 3 [229, 2347] root47Candidates12 2576 winner else winner) = _
  rw [if_neg (by decide)]
def root47Candidates13 : List Nat := [16477, 18433, 20551]
set_option maxRecDepth 8192 in
theorem root47Candidates13_checked : candidates edge 4 229 2767 root47Tail14 winner.bound = root47Candidates13 := by
  unfold candidates
  have row : edge 2767 = (SortedNatTree.contains FastRows.row2767) := by
    change Data.row2767.contains = (SortedNatTree.contains FastRows.row2767)
    exact FastRows.row2767_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root47Child13 : search edge 3 [229, 2767] root47Candidates13 2996 winner = winner := by decide
theorem root47Step13 : scan edge (search edge 3) 4 [229] 229 root47Tail13 winner =
    scan edge (search edge 3) 4 [229] 229 root47Tail14 winner := by
  rw [root47Tail13, scan, if_neg (by decide), root47Candidates13_checked]
  change scan edge (search edge 3) 4 [229] 229 root47Tail14
    (if 3 ≤ root47Candidates13.length then search edge 3 [229, 2767] root47Candidates13 2996 winner else winner) = _
  rw [if_pos (by decide), root47Child13]
def root47Candidates14 : List Nat := [7759]
set_option maxRecDepth 8192 in
theorem root47Candidates14_checked : candidates edge 4 229 3121 root47Tail15 winner.bound = root47Candidates14 := by
  unfold candidates
  have row : edge 3121 = (SortedNatTree.contains FastRows.row3121) := by
    change Data.row3121.contains = (SortedNatTree.contains FastRows.row3121)
    exact FastRows.row3121_checked.symm
  rw [row]
  decide
theorem root47Step14 : scan edge (search edge 3) 4 [229] 229 root47Tail14 winner =
    scan edge (search edge 3) 4 [229] 229 root47Tail15 winner := by
  rw [root47Tail14, scan, if_neg (by decide), root47Candidates14_checked]
  change scan edge (search edge 3) 4 [229] 229 root47Tail15
    (if 3 ≤ root47Candidates14.length then search edge 3 [229, 3121] root47Candidates14 3350 winner else winner) = _
  rw [if_neg (by decide)]
def root47Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root47Candidates15_checked : candidates edge 4 229 3301 root47Tail16 winner.bound = root47Candidates15 := by
  unfold candidates
  have row : edge 3301 = (SortedNatTree.contains FastRows.row3301) := by
    change Data.row3301.contains = (SortedNatTree.contains FastRows.row3301)
    exact FastRows.row3301_checked.symm
  rw [row]
  decide
theorem root47Step15 : scan edge (search edge 3) 4 [229] 229 root47Tail15 winner =
    scan edge (search edge 3) 4 [229] 229 root47Tail16 winner := by
  rw [root47Tail15, scan, if_neg (by decide), root47Candidates15_checked]
  change scan edge (search edge 3) 4 [229] 229 root47Tail16
    (if 3 ≤ root47Candidates15.length then search edge 3 [229, 3301] root47Candidates15 3530 winner else winner) = _
  rw [if_neg (by decide)]
def root47Candidates16 : List Nat := [6733, 9463, 11491, 15817]
set_option maxRecDepth 8192 in
theorem root47Candidates16_checked : candidates edge 4 229 3391 root47Tail17 winner.bound = root47Candidates16 := by
  unfold candidates
  have row : edge 3391 = (SortedNatTree.contains FastRows.row3391) := by
    change Data.row3391.contains = (SortedNatTree.contains FastRows.row3391)
    exact FastRows.row3391_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root47Child16 : search edge 3 [229, 3391] root47Candidates16 3620 winner = winner := by decide
theorem root47Step16 : scan edge (search edge 3) 4 [229] 229 root47Tail16 winner =
    scan edge (search edge 3) 4 [229] 229 root47Tail17 winner := by
  rw [root47Tail16, scan, if_neg (by decide), root47Candidates16_checked]
  change scan edge (search edge 3) 4 [229] 229 root47Tail17
    (if 3 ≤ root47Candidates16.length then search edge 3 [229, 3391] root47Candidates16 3620 winner else winner) = _
  rw [if_pos (by decide), root47Child16]
def root47Candidates17 : List Nat := [9001, 20089]
set_option maxRecDepth 8192 in
theorem root47Candidates17_checked : candidates edge 4 229 3541 root47Tail18 winner.bound = root47Candidates17 := by
  unfold candidates
  have row : edge 3541 = (SortedNatTree.contains FastRows.row3541) := by
    change Data.row3541.contains = (SortedNatTree.contains FastRows.row3541)
    exact FastRows.row3541_checked.symm
  rw [row]
  decide
theorem root47Step17 : scan edge (search edge 3) 4 [229] 229 root47Tail17 winner =
    scan edge (search edge 3) 4 [229] 229 root47Tail18 winner := by
  rw [root47Tail17, scan, if_neg (by decide), root47Candidates17_checked]
  change scan edge (search edge 3) 4 [229] 229 root47Tail18
    (if 3 ≤ root47Candidates17.length then search edge 3 [229, 3541] root47Candidates17 3770 winner else winner) = _
  rw [if_neg (by decide)]
def root47Candidates18 : List Nat := [4021, 5869]
set_option maxRecDepth 8192 in
theorem root47Candidates18_checked : candidates edge 4 229 3559 root47Tail19 winner.bound = root47Candidates18 := by
  unfold candidates
  have row : edge 3559 = (SortedNatTree.contains FastRows.row3559) := by
    change Data.row3559.contains = (SortedNatTree.contains FastRows.row3559)
    exact FastRows.row3559_checked.symm
  rw [row]
  decide
theorem root47Step18 : scan edge (search edge 3) 4 [229] 229 root47Tail18 winner =
    scan edge (search edge 3) 4 [229] 229 root47Tail19 winner := by
  rw [root47Tail18, scan, if_neg (by decide), root47Candidates18_checked]
  change scan edge (search edge 3) 4 [229] 229 root47Tail19
    (if 3 ≤ root47Candidates18.length then search edge 3 [229, 3559] root47Candidates18 3788 winner else winner) = _
  rw [if_neg (by decide)]
def root47Candidates19 : List Nat := [17749]
set_option maxRecDepth 8192 in
theorem root47Candidates19_checked : candidates edge 4 229 3919 root47Tail20 winner.bound = root47Candidates19 := by
  unfold candidates
  have row : edge 3919 = (SortedNatTree.contains FastRows.row3919) := by
    change Data.row3919.contains = (SortedNatTree.contains FastRows.row3919)
    exact FastRows.row3919_checked.symm
  rw [row]
  decide
theorem root47Step19 : scan edge (search edge 3) 4 [229] 229 root47Tail19 winner =
    scan edge (search edge 3) 4 [229] 229 root47Tail20 winner := by
  rw [root47Tail19, scan, if_neg (by decide), root47Candidates19_checked]
  change scan edge (search edge 3) 4 [229] 229 root47Tail20
    (if 3 ≤ root47Candidates19.length then search edge 3 [229, 3919] root47Candidates19 4148 winner else winner) = _
  rw [if_neg (by decide)]
def root47Candidates20 : List Nat := [6733, 10453, 12037, 17929]
set_option maxRecDepth 8192 in
theorem root47Candidates20_checked : candidates edge 4 229 4021 root47Tail21 winner.bound = root47Candidates20 := by
  unfold candidates
  have row : edge 4021 = (SortedNatTree.contains FastRows.row4021) := by
    change Data.row4021.contains = (SortedNatTree.contains FastRows.row4021)
    exact FastRows.row4021_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root47Child20 : search edge 3 [229, 4021] root47Candidates20 4250 winner = winner := by decide
theorem root47Step20 : scan edge (search edge 3) 4 [229] 229 root47Tail20 winner =
    scan edge (search edge 3) 4 [229] 229 root47Tail21 winner := by
  rw [root47Tail20, scan, if_neg (by decide), root47Candidates20_checked]
  change scan edge (search edge 3) 4 [229] 229 root47Tail21
    (if 3 ≤ root47Candidates20.length then search edge 3 [229, 4021] root47Candidates20 4250 winner else winner) = _
  rw [if_pos (by decide), root47Child20]
def root47Candidates21 : List Nat := [9907, 16633]
set_option maxRecDepth 8192 in
theorem root47Candidates21_checked : candidates edge 4 229 5077 root47Tail22 winner.bound = root47Candidates21 := by
  unfold candidates
  have row : edge 5077 = (SortedNatTree.contains FastRows.row5077) := by
    change Data.row5077.contains = (SortedNatTree.contains FastRows.row5077)
    exact FastRows.row5077_checked.symm
  rw [row]
  decide
theorem root47Step21 : scan edge (search edge 3) 4 [229] 229 root47Tail21 winner =
    scan edge (search edge 3) 4 [229] 229 root47Tail22 winner := by
  rw [root47Tail21, scan, if_neg (by decide), root47Candidates21_checked]
  change scan edge (search edge 3) 4 [229] 229 root47Tail22
    (if 3 ≤ root47Candidates21.length then search edge 3 [229, 5077] root47Candidates21 5306 winner else winner) = _
  rw [if_neg (by decide)]
def root47Candidates22 : List Nat := [12511]
set_option maxRecDepth 8192 in
theorem root47Candidates22_checked : candidates edge 4 229 5569 root47Tail23 winner.bound = root47Candidates22 := by
  unfold candidates
  have row : edge 5569 = (SortedNatTree.contains FastRows.row5569) := by
    change Data.row5569.contains = (SortedNatTree.contains FastRows.row5569)
    exact FastRows.row5569_checked.symm
  rw [row]
  decide
theorem root47Step22 : scan edge (search edge 3) 4 [229] 229 root47Tail22 winner =
    scan edge (search edge 3) 4 [229] 229 root47Tail23 winner := by
  rw [root47Tail22, scan, if_neg (by decide), root47Candidates22_checked]
  change scan edge (search edge 3) 4 [229] 229 root47Tail23
    (if 3 ≤ root47Candidates22.length then search edge 3 [229, 5569] root47Candidates22 5798 winner else winner) = _
  rw [if_neg (by decide)]
def root47Candidates23 : List Nat := [7759]
set_option maxRecDepth 8192 in
theorem root47Candidates23_checked : candidates edge 4 229 5659 root47Tail24 winner.bound = root47Candidates23 := by
  unfold candidates
  have row : edge 5659 = (SortedNatTree.contains FastRows.row5659) := by
    change Data.row5659.contains = (SortedNatTree.contains FastRows.row5659)
    exact FastRows.row5659_checked.symm
  rw [row]
  decide
theorem root47Step23 : scan edge (search edge 3) 4 [229] 229 root47Tail23 winner =
    scan edge (search edge 3) 4 [229] 229 root47Tail24 winner := by
  rw [root47Tail23, scan, if_neg (by decide), root47Candidates23_checked]
  change scan edge (search edge 3) 4 [229] 229 root47Tail24
    (if 3 ≤ root47Candidates23.length then search edge 3 [229, 5659] root47Candidates23 5888 winner else winner) = _
  rw [if_neg (by decide)]
def root47Candidates24 : List Nat := []
set_option maxRecDepth 8192 in
theorem root47Candidates24_checked : candidates edge 4 229 5869 root47Tail25 winner.bound = root47Candidates24 := by
  unfold candidates
  have row : edge 5869 = (SortedNatTree.contains FastRows.row5869) := by
    change Data.row5869.contains = (SortedNatTree.contains FastRows.row5869)
    exact FastRows.row5869_checked.symm
  rw [row]
  decide
theorem root47Step24 : scan edge (search edge 3) 4 [229] 229 root47Tail24 winner =
    scan edge (search edge 3) 4 [229] 229 root47Tail25 winner := by
  rw [root47Tail24, scan, if_neg (by decide), root47Candidates24_checked]
  change scan edge (search edge 3) 4 [229] 229 root47Tail25
    (if 3 ≤ root47Candidates24.length then search edge 3 [229, 5869] root47Candidates24 6098 winner else winner) = _
  rw [if_neg (by decide)]
def root47Candidates25 : List Nat := [12511, 15787, 17299]
set_option maxRecDepth 8192 in
theorem root47Candidates25_checked : candidates edge 4 229 6247 root47Tail26 winner.bound = root47Candidates25 := by
  unfold candidates
  have row : edge 6247 = (SortedNatTree.contains FastRows.row6247) := by
    change Data.row6247.contains = (SortedNatTree.contains FastRows.row6247)
    exact FastRows.row6247_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root47Child25 : search edge 3 [229, 6247] root47Candidates25 6476 winner = winner := by decide
theorem root47Step25 : scan edge (search edge 3) 4 [229] 229 root47Tail25 winner =
    scan edge (search edge 3) 4 [229] 229 root47Tail26 winner := by
  rw [root47Tail25, scan, if_neg (by decide), root47Candidates25_checked]
  change scan edge (search edge 3) 4 [229] 229 root47Tail26
    (if 3 ≤ root47Candidates25.length then search edge 3 [229, 6247] root47Candidates25 6476 winner else winner) = _
  rw [if_pos (by decide), root47Child25]
theorem root47Scan26 : scan edge (search edge 3) 4 [229] 229 root47Tail26 winner = winner := by rw [root47Tail26, scan, if_pos (by decide)]
theorem root47Scan25 : scan edge (search edge 3) 4 [229] 229 root47Tail25 winner = winner :=
  (root47Step25).trans root47Scan26
theorem root47Scan24 : scan edge (search edge 3) 4 [229] 229 root47Tail24 winner = winner :=
  (root47Step24).trans root47Scan25
theorem root47Scan23 : scan edge (search edge 3) 4 [229] 229 root47Tail23 winner = winner :=
  (root47Step23).trans root47Scan24
theorem root47Scan22 : scan edge (search edge 3) 4 [229] 229 root47Tail22 winner = winner :=
  (root47Step22).trans root47Scan23
theorem root47Scan21 : scan edge (search edge 3) 4 [229] 229 root47Tail21 winner = winner :=
  (root47Step21).trans root47Scan22
theorem root47Scan20 : scan edge (search edge 3) 4 [229] 229 root47Tail20 winner = winner :=
  (root47Step20).trans root47Scan21
theorem root47Scan19 : scan edge (search edge 3) 4 [229] 229 root47Tail19 winner = winner :=
  (root47Step19).trans root47Scan20
theorem root47Scan18 : scan edge (search edge 3) 4 [229] 229 root47Tail18 winner = winner :=
  (root47Step18).trans root47Scan19
theorem root47Scan17 : scan edge (search edge 3) 4 [229] 229 root47Tail17 winner = winner :=
  (root47Step17).trans root47Scan18
theorem root47Scan16 : scan edge (search edge 3) 4 [229] 229 root47Tail16 winner = winner :=
  (root47Step16).trans root47Scan17
theorem root47Scan15 : scan edge (search edge 3) 4 [229] 229 root47Tail15 winner = winner :=
  (root47Step15).trans root47Scan16
theorem root47Scan14 : scan edge (search edge 3) 4 [229] 229 root47Tail14 winner = winner :=
  (root47Step14).trans root47Scan15
theorem root47Scan13 : scan edge (search edge 3) 4 [229] 229 root47Tail13 winner = winner :=
  (root47Step13).trans root47Scan14
theorem root47Scan12 : scan edge (search edge 3) 4 [229] 229 root47Tail12 winner = winner :=
  (root47Step12).trans root47Scan13
theorem root47Scan11 : scan edge (search edge 3) 4 [229] 229 root47Tail11 winner = winner :=
  (root47Step11).trans root47Scan12
theorem root47Scan10 : scan edge (search edge 3) 4 [229] 229 root47Tail10 winner = winner :=
  (root47Step10).trans root47Scan11
theorem root47Scan9 : scan edge (search edge 3) 4 [229] 229 root47Tail9 winner = winner :=
  (root47Step9).trans root47Scan10
theorem root47Scan8 : scan edge (search edge 3) 4 [229] 229 root47Tail8 winner = winner :=
  (root47Step8).trans root47Scan9
theorem root47Scan7 : scan edge (search edge 3) 4 [229] 229 root47Tail7 winner = winner :=
  (root47Step7).trans root47Scan8
theorem root47Scan6 : scan edge (search edge 3) 4 [229] 229 root47Tail6 winner = winner :=
  (root47Step6).trans root47Scan7
theorem root47Scan5 : scan edge (search edge 3) 4 [229] 229 root47Tail5 winner = winner :=
  (root47Step5).trans root47Scan6
theorem root47Scan4 : scan edge (search edge 3) 4 [229] 229 root47Tail4 winner = winner :=
  (root47Step4).trans root47Scan5
theorem root47Scan3 : scan edge (search edge 3) 4 [229] 229 root47Tail3 winner = winner :=
  (root47Step3).trans root47Scan4
theorem root47Scan2 : scan edge (search edge 3) 4 [229] 229 root47Tail2 winner = winner :=
  (root47Step2).trans root47Scan3
theorem root47Scan1 : scan edge (search edge 3) 4 [229] 229 root47Tail1 winner = winner :=
  (root47Step1).trans root47Scan2
theorem root47Scan0 : scan edge (search edge 3) 4 [229] 229 root47Tail0 winner = winner :=
  (root47Step0).trans root47Scan1
set_option maxRecDepth 4096 in
theorem root47Branch : search edge 4 [229] root47Candidates 229 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [229] 229 root47Tail0 winner = winner
  exact root47Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
