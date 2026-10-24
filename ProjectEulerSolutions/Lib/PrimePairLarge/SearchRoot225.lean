import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root225Candidates : List Nat := Data.row1439
set_option maxRecDepth 8192 in
theorem root225Candidates_checked : candidates edge 5 0 1439 tail226 winner.bound = root225Candidates := by
  apply candidates_eq_row edge 5 0 1439 tail226 root225Candidates winner.bound tail226_ordered
  · decide
  · decide
  · funext q
    rfl
def root225Tail66 : List Nat := []
def root225Tail65 : List Nat := 23957 :: root225Tail66
def root225Tail64 : List Nat := 23831 :: root225Tail65
def root225Tail63 : List Nat := 23789 :: root225Tail64
def root225Tail62 : List Nat := 22787 :: root225Tail63
def root225Tail61 : List Nat := 22307 :: root225Tail62
def root225Tail60 : List Nat := 21773 :: root225Tail61
def root225Tail59 : List Nat := 21383 :: root225Tail60
def root225Tail58 : List Nat := 20807 :: root225Tail59
def root225Tail57 : List Nat := 20477 :: root225Tail58
def root225Tail56 : List Nat := 19583 :: root225Tail57
def root225Tail55 : List Nat := 19463 :: root225Tail56
def root225Tail54 : List Nat := 17957 :: root225Tail55
def root225Tail53 : List Nat := 16607 :: root225Tail54
def root225Tail52 : List Nat := 16349 :: root225Tail53
def root225Tail51 : List Nat := 15881 :: root225Tail52
def root225Tail50 : List Nat := 15227 :: root225Tail51
def root225Tail49 : List Nat := 15173 :: root225Tail50
def root225Tail48 : List Nat := 14891 :: root225Tail49
def root225Tail47 : List Nat := 14723 :: root225Tail48
def root225Tail46 : List Nat := 14543 :: root225Tail47
def root225Tail45 : List Nat := 13799 :: root225Tail46
def root225Tail44 : List Nat := 13241 :: root225Tail45
def root225Tail43 : List Nat := 12917 :: root225Tail44
def root225Tail42 : List Nat := 12281 :: root225Tail43
def root225Tail41 : List Nat := 12149 :: root225Tail42
def root225Tail40 : List Nat := 12071 :: root225Tail41
def root225Tail39 : List Nat := 11939 :: root225Tail40
def root225Tail38 : List Nat := 10733 :: root225Tail39
def root225Tail37 : List Nat := 10589 :: root225Tail38
def root225Tail36 : List Nat := 10301 :: root225Tail37
def root225Tail35 : List Nat := 10139 :: root225Tail36
def root225Tail34 : List Nat := 9521 :: root225Tail35
def root225Tail33 : List Nat := 9293 :: root225Tail34
def root225Tail32 : List Nat := 9137 :: root225Tail33
def root225Tail31 : List Nat := 9059 :: root225Tail32
def root225Tail30 : List Nat := 9041 :: root225Tail31
def root225Tail29 : List Nat := 8933 :: root225Tail30
def root225Tail28 : List Nat := 8543 :: root225Tail29
def root225Tail27 : List Nat := 8081 :: root225Tail28
def root225Tail26 : List Nat := 8039 :: root225Tail27
def root225Tail25 : List Nat := 7583 :: root225Tail26
def root225Tail24 : List Nat := 7127 :: root225Tail25
def root225Tail23 : List Nat := 6983 :: root225Tail24
def root225Tail22 : List Nat := 6869 :: root225Tail23
def root225Tail21 : List Nat := 6659 :: root225Tail22
def root225Tail20 : List Nat := 6581 :: root225Tail21
def root225Tail19 : List Nat := 5783 :: root225Tail20
def root225Tail18 : List Nat := 5657 :: root225Tail19
def root225Tail17 : List Nat := 5279 :: root225Tail18
def root225Tail16 : List Nat := 5273 :: root225Tail17
def root225Tail15 : List Nat := 5261 :: root225Tail16
def root225Tail14 : List Nat := 5231 :: root225Tail15
def root225Tail13 : List Nat := 5021 :: root225Tail14
def root225Tail12 : List Nat := 4931 :: root225Tail13
def root225Tail11 : List Nat := 4463 :: root225Tail12
def root225Tail10 : List Nat := 4337 :: root225Tail11
def root225Tail9 : List Nat := 4271 :: root225Tail10
def root225Tail8 : List Nat := 4091 :: root225Tail9
def root225Tail7 : List Nat := 4007 :: root225Tail8
def root225Tail6 : List Nat := 3833 :: root225Tail7
def root225Tail5 : List Nat := 3719 :: root225Tail6
def root225Tail4 : List Nat := 3413 :: root225Tail5
def root225Tail3 : List Nat := 2837 :: root225Tail4
def root225Tail2 : List Nat := 2579 :: root225Tail3
def root225Tail1 : List Nat := 2243 :: root225Tail2
def root225Tail0 : List Nat := 2003 :: root225Tail1
def root225Candidates0 : List Nat := [3413, 4271, 5657, 6983, 16607, 19583]
set_option maxRecDepth 8192 in
theorem root225Candidates0_checked : candidates edge 4 1439 2003 root225Tail1 winner.bound = root225Candidates0 := by
  unfold candidates
  have row : edge 2003 = (SortedNatTree.contains FastRows.row2003) := by
    change Data.row2003.contains = (SortedNatTree.contains FastRows.row2003)
    exact FastRows.row2003_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root225Child0 : search edge 3 [1439, 2003] root225Candidates0 3442 winner = winner := by decide
theorem root225Step0 : scan edge (search edge 3) 4 [1439] 1439 root225Tail0 winner =
    scan edge (search edge 3) 4 [1439] 1439 root225Tail1 winner := by
  rw [root225Tail0, scan, if_neg (by decide), root225Candidates0_checked]
  change scan edge (search edge 3) 4 [1439] 1439 root225Tail1
    (if 3 ≤ root225Candidates0.length then search edge 3 [1439, 2003] root225Candidates0 3442 winner else winner) = _
  rw [if_pos (by decide), root225Child0]
def root225Candidates1 : List Nat := [8081, 16349]
set_option maxRecDepth 8192 in
theorem root225Candidates1_checked : candidates edge 4 1439 2243 root225Tail2 winner.bound = root225Candidates1 := by
  unfold candidates
  have row : edge 2243 = (SortedNatTree.contains FastRows.row2243) := by
    change Data.row2243.contains = (SortedNatTree.contains FastRows.row2243)
    exact FastRows.row2243_checked.symm
  rw [row]
  decide
theorem root225Step1 : scan edge (search edge 3) 4 [1439] 1439 root225Tail1 winner =
    scan edge (search edge 3) 4 [1439] 1439 root225Tail2 winner := by
  rw [root225Tail1, scan, if_neg (by decide), root225Candidates1_checked]
  change scan edge (search edge 3) 4 [1439] 1439 root225Tail2
    (if 3 ≤ root225Candidates1.length then search edge 3 [1439, 2243] root225Candidates1 3682 winner else winner) = _
  rw [if_neg (by decide)]
def root225Candidates2 : List Nat := [4007, 4931, 15173, 19463]
set_option maxRecDepth 8192 in
theorem root225Candidates2_checked : candidates edge 4 1439 2579 root225Tail3 winner.bound = root225Candidates2 := by
  unfold candidates
  have row : edge 2579 = (SortedNatTree.contains FastRows.row2579) := by
    change Data.row2579.contains = (SortedNatTree.contains FastRows.row2579)
    exact FastRows.row2579_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root225Child2 : search edge 3 [1439, 2579] root225Candidates2 4018 winner = winner := by decide
theorem root225Step2 : scan edge (search edge 3) 4 [1439] 1439 root225Tail2 winner =
    scan edge (search edge 3) 4 [1439] 1439 root225Tail3 winner := by
  rw [root225Tail2, scan, if_neg (by decide), root225Candidates2_checked]
  change scan edge (search edge 3) 4 [1439] 1439 root225Tail3
    (if 3 ≤ root225Candidates2.length then search edge 3 [1439, 2579] root225Candidates2 4018 winner else winner) = _
  rw [if_pos (by decide), root225Child2]
def root225Candidates3 : List Nat := [10733, 14891]
set_option maxRecDepth 8192 in
theorem root225Candidates3_checked : candidates edge 4 1439 2837 root225Tail4 winner.bound = root225Candidates3 := by
  unfold candidates
  have row : edge 2837 = (SortedNatTree.contains FastRows.row2837) := by
    change Data.row2837.contains = (SortedNatTree.contains FastRows.row2837)
    exact FastRows.row2837_checked.symm
  rw [row]
  decide
theorem root225Step3 : scan edge (search edge 3) 4 [1439] 1439 root225Tail3 winner =
    scan edge (search edge 3) 4 [1439] 1439 root225Tail4 winner := by
  rw [root225Tail3, scan, if_neg (by decide), root225Candidates3_checked]
  change scan edge (search edge 3) 4 [1439] 1439 root225Tail4
    (if 3 ≤ root225Candidates3.length then search edge 3 [1439, 2837] root225Candidates3 4276 winner else winner) = _
  rw [if_neg (by decide)]
def root225Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root225Candidates4_checked : candidates edge 4 1439 3413 root225Tail5 winner.bound = root225Candidates4 := by
  unfold candidates
  have row : edge 3413 = (SortedNatTree.contains FastRows.row3413) := by
    change Data.row3413.contains = (SortedNatTree.contains FastRows.row3413)
    exact FastRows.row3413_checked.symm
  rw [row]
  decide
theorem root225Step4 : scan edge (search edge 3) 4 [1439] 1439 root225Tail4 winner =
    scan edge (search edge 3) 4 [1439] 1439 root225Tail5 winner := by
  rw [root225Tail4, scan, if_neg (by decide), root225Candidates4_checked]
  change scan edge (search edge 3) 4 [1439] 1439 root225Tail5
    (if 3 ≤ root225Candidates4.length then search edge 3 [1439, 3413] root225Candidates4 4852 winner else winner) = _
  rw [if_neg (by decide)]
def root225Candidates5 : List Nat := [4271, 5273]
set_option maxRecDepth 8192 in
theorem root225Candidates5_checked : candidates edge 4 1439 3719 root225Tail6 winner.bound = root225Candidates5 := by
  unfold candidates
  have row : edge 3719 = (SortedNatTree.contains FastRows.row3719) := by
    change Data.row3719.contains = (SortedNatTree.contains FastRows.row3719)
    exact FastRows.row3719_checked.symm
  rw [row]
  decide
theorem root225Step5 : scan edge (search edge 3) 4 [1439] 1439 root225Tail5 winner =
    scan edge (search edge 3) 4 [1439] 1439 root225Tail6 winner := by
  rw [root225Tail5, scan, if_neg (by decide), root225Candidates5_checked]
  change scan edge (search edge 3) 4 [1439] 1439 root225Tail6
    (if 3 ≤ root225Candidates5.length then search edge 3 [1439, 3719] root225Candidates5 5158 winner else winner) = _
  rw [if_neg (by decide)]
def root225Candidates6 : List Nat := [5657, 6983, 14891]
set_option maxRecDepth 8192 in
theorem root225Candidates6_checked : candidates edge 4 1439 3833 root225Tail7 winner.bound = root225Candidates6 := by
  unfold candidates
  have row : edge 3833 = (SortedNatTree.contains FastRows.row3833) := by
    change Data.row3833.contains = (SortedNatTree.contains FastRows.row3833)
    exact FastRows.row3833_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root225Child6 : search edge 3 [1439, 3833] root225Candidates6 5272 winner = winner := by decide
theorem root225Step6 : scan edge (search edge 3) 4 [1439] 1439 root225Tail6 winner =
    scan edge (search edge 3) 4 [1439] 1439 root225Tail7 winner := by
  rw [root225Tail6, scan, if_neg (by decide), root225Candidates6_checked]
  change scan edge (search edge 3) 4 [1439] 1439 root225Tail7
    (if 3 ≤ root225Candidates6.length then search edge 3 [1439, 3833] root225Candidates6 5272 winner else winner) = _
  rw [if_pos (by decide), root225Child6]
def root225Candidates7 : List Nat := [4091, 4337, 8081, 12149, 15173]
set_option maxRecDepth 8192 in
theorem root225Candidates7_checked : candidates edge 4 1439 4007 root225Tail8 winner.bound = root225Candidates7 := by
  unfold candidates
  have row : edge 4007 = (SortedNatTree.contains FastRows.row4007) := by
    change Data.row4007.contains = (SortedNatTree.contains FastRows.row4007)
    exact FastRows.row4007_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root225Child7 : search edge 3 [1439, 4007] root225Candidates7 5446 winner = winner := by decide
theorem root225Step7 : scan edge (search edge 3) 4 [1439] 1439 root225Tail7 winner =
    scan edge (search edge 3) 4 [1439] 1439 root225Tail8 winner := by
  rw [root225Tail7, scan, if_neg (by decide), root225Candidates7_checked]
  change scan edge (search edge 3) 4 [1439] 1439 root225Tail8
    (if 3 ≤ root225Candidates7.length then search edge 3 [1439, 4007] root225Candidates7 5446 winner else winner) = _
  rw [if_pos (by decide), root225Child7]
def root225Candidates8 : List Nat := [13241]
set_option maxRecDepth 8192 in
theorem root225Candidates8_checked : candidates edge 4 1439 4091 root225Tail9 winner.bound = root225Candidates8 := by
  unfold candidates
  have row : edge 4091 = (SortedNatTree.contains FastRows.row4091) := by
    change Data.row4091.contains = (SortedNatTree.contains FastRows.row4091)
    exact FastRows.row4091_checked.symm
  rw [row]
  decide
theorem root225Step8 : scan edge (search edge 3) 4 [1439] 1439 root225Tail8 winner =
    scan edge (search edge 3) 4 [1439] 1439 root225Tail9 winner := by
  rw [root225Tail8, scan, if_neg (by decide), root225Candidates8_checked]
  change scan edge (search edge 3) 4 [1439] 1439 root225Tail9
    (if 3 ≤ root225Candidates8.length then search edge 3 [1439, 4091] root225Candidates8 5530 winner else winner) = _
  rw [if_neg (by decide)]
def root225Candidates9 : List Nat := [9137]
set_option maxRecDepth 8192 in
theorem root225Candidates9_checked : candidates edge 4 1439 4271 root225Tail10 winner.bound = root225Candidates9 := by
  unfold candidates
  have row : edge 4271 = (SortedNatTree.contains FastRows.row4271) := by
    change Data.row4271.contains = (SortedNatTree.contains FastRows.row4271)
    exact FastRows.row4271_checked.symm
  rw [row]
  decide
theorem root225Step9 : scan edge (search edge 3) 4 [1439] 1439 root225Tail9 winner =
    scan edge (search edge 3) 4 [1439] 1439 root225Tail10 winner := by
  rw [root225Tail9, scan, if_neg (by decide), root225Candidates9_checked]
  change scan edge (search edge 3) 4 [1439] 1439 root225Tail10
    (if 3 ≤ root225Candidates9.length then search edge 3 [1439, 4271] root225Candidates9 5710 winner else winner) = _
  rw [if_neg (by decide)]
def root225Candidates10 : List Nat := [9293]
set_option maxRecDepth 8192 in
theorem root225Candidates10_checked : candidates edge 4 1439 4337 root225Tail11 winner.bound = root225Candidates10 := by
  unfold candidates
  have row : edge 4337 = (SortedNatTree.contains FastRows.row4337) := by
    change Data.row4337.contains = (SortedNatTree.contains FastRows.row4337)
    exact FastRows.row4337_checked.symm
  rw [row]
  decide
theorem root225Step10 : scan edge (search edge 3) 4 [1439] 1439 root225Tail10 winner =
    scan edge (search edge 3) 4 [1439] 1439 root225Tail11 winner := by
  rw [root225Tail10, scan, if_neg (by decide), root225Candidates10_checked]
  change scan edge (search edge 3) 4 [1439] 1439 root225Tail11
    (if 3 ≤ root225Candidates10.length then search edge 3 [1439, 4337] root225Candidates10 5776 winner else winner) = _
  rw [if_neg (by decide)]
def root225Candidates11 : List Nat := [5273, 10301, 12281, 17957]
set_option maxRecDepth 8192 in
theorem root225Candidates11_checked : candidates edge 4 1439 4463 root225Tail12 winner.bound = root225Candidates11 := by
  unfold candidates
  have row : edge 4463 = (SortedNatTree.contains FastRows.row4463) := by
    change Data.row4463.contains = (SortedNatTree.contains FastRows.row4463)
    exact FastRows.row4463_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root225Child11 : search edge 3 [1439, 4463] root225Candidates11 5902 winner = winner := by decide
theorem root225Step11 : scan edge (search edge 3) 4 [1439] 1439 root225Tail11 winner =
    scan edge (search edge 3) 4 [1439] 1439 root225Tail12 winner := by
  rw [root225Tail11, scan, if_neg (by decide), root225Candidates11_checked]
  change scan edge (search edge 3) 4 [1439] 1439 root225Tail12
    (if 3 ≤ root225Candidates11.length then search edge 3 [1439, 4463] root225Candidates11 5902 winner else winner) = _
  rw [if_pos (by decide), root225Child11]
def root225Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root225Candidates12_checked : candidates edge 4 1439 4931 root225Tail13 winner.bound = root225Candidates12 := by
  unfold candidates
  have row : edge 4931 = (SortedNatTree.contains FastRows.row4931) := by
    change Data.row4931.contains = (SortedNatTree.contains FastRows.row4931)
    exact FastRows.row4931_checked.symm
  rw [row]
  decide
theorem root225Step12 : scan edge (search edge 3) 4 [1439] 1439 root225Tail12 winner =
    scan edge (search edge 3) 4 [1439] 1439 root225Tail13 winner := by
  rw [root225Tail12, scan, if_neg (by decide), root225Candidates12_checked]
  change scan edge (search edge 3) 4 [1439] 1439 root225Tail13
    (if 3 ≤ root225Candidates12.length then search edge 3 [1439, 4931] root225Candidates12 6370 winner else winner) = _
  rw [if_neg (by decide)]
def root225Candidates13 : List Nat := [7127, 10733]
set_option maxRecDepth 8192 in
theorem root225Candidates13_checked : candidates edge 4 1439 5021 root225Tail14 winner.bound = root225Candidates13 := by
  unfold candidates
  have row : edge 5021 = (SortedNatTree.contains FastRows.row5021) := by
    change Data.row5021.contains = (SortedNatTree.contains FastRows.row5021)
    exact FastRows.row5021_checked.symm
  rw [row]
  decide
theorem root225Step13 : scan edge (search edge 3) 4 [1439] 1439 root225Tail13 winner =
    scan edge (search edge 3) 4 [1439] 1439 root225Tail14 winner := by
  rw [root225Tail13, scan, if_neg (by decide), root225Candidates13_checked]
  change scan edge (search edge 3) 4 [1439] 1439 root225Tail14
    (if 3 ≤ root225Candidates13.length then search edge 3 [1439, 5021] root225Candidates13 6460 winner else winner) = _
  rw [if_neg (by decide)]
def root225Candidates14 : List Nat := [6581]
set_option maxRecDepth 8192 in
theorem root225Candidates14_checked : candidates edge 4 1439 5231 root225Tail15 winner.bound = root225Candidates14 := by
  unfold candidates
  have row : edge 5231 = (SortedNatTree.contains FastRows.row5231) := by
    change Data.row5231.contains = (SortedNatTree.contains FastRows.row5231)
    exact FastRows.row5231_checked.symm
  rw [row]
  decide
theorem root225Step14 : scan edge (search edge 3) 4 [1439] 1439 root225Tail14 winner =
    scan edge (search edge 3) 4 [1439] 1439 root225Tail15 winner := by
  rw [root225Tail14, scan, if_neg (by decide), root225Candidates14_checked]
  change scan edge (search edge 3) 4 [1439] 1439 root225Tail15
    (if 3 ≤ root225Candidates14.length then search edge 3 [1439, 5231] root225Candidates14 6670 winner else winner) = _
  rw [if_neg (by decide)]
def root225Candidates15 : List Nat := [6983, 12149]
set_option maxRecDepth 8192 in
theorem root225Candidates15_checked : candidates edge 4 1439 5261 root225Tail16 winner.bound = root225Candidates15 := by
  unfold candidates
  have row : edge 5261 = (SortedNatTree.contains FastRows.row5261) := by
    change Data.row5261.contains = (SortedNatTree.contains FastRows.row5261)
    exact FastRows.row5261_checked.symm
  rw [row]
  decide
theorem root225Step15 : scan edge (search edge 3) 4 [1439] 1439 root225Tail15 winner =
    scan edge (search edge 3) 4 [1439] 1439 root225Tail16 winner := by
  rw [root225Tail15, scan, if_neg (by decide), root225Candidates15_checked]
  change scan edge (search edge 3) 4 [1439] 1439 root225Tail16
    (if 3 ≤ root225Candidates15.length then search edge 3 [1439, 5261] root225Candidates15 6700 winner else winner) = _
  rw [if_neg (by decide)]
def root225Candidates16 : List Nat := [7127]
set_option maxRecDepth 8192 in
theorem root225Candidates16_checked : candidates edge 4 1439 5273 root225Tail17 winner.bound = root225Candidates16 := by
  unfold candidates
  have row : edge 5273 = (SortedNatTree.contains FastRows.row5273) := by
    change Data.row5273.contains = (SortedNatTree.contains FastRows.row5273)
    exact FastRows.row5273_checked.symm
  rw [row]
  decide
theorem root225Step16 : scan edge (search edge 3) 4 [1439] 1439 root225Tail16 winner =
    scan edge (search edge 3) 4 [1439] 1439 root225Tail17 winner := by
  rw [root225Tail16, scan, if_neg (by decide), root225Candidates16_checked]
  change scan edge (search edge 3) 4 [1439] 1439 root225Tail17
    (if 3 ≤ root225Candidates16.length then search edge 3 [1439, 5273] root225Candidates16 6712 winner else winner) = _
  rw [if_neg (by decide)]
def root225Candidates17 : List Nat := [5783]
set_option maxRecDepth 8192 in
theorem root225Candidates17_checked : candidates edge 4 1439 5279 root225Tail18 winner.bound = root225Candidates17 := by
  unfold candidates
  have row : edge 5279 = (SortedNatTree.contains FastRows.row5279) := by
    change Data.row5279.contains = (SortedNatTree.contains FastRows.row5279)
    exact FastRows.row5279_checked.symm
  rw [row]
  decide
theorem root225Step17 : scan edge (search edge 3) 4 [1439] 1439 root225Tail17 winner =
    scan edge (search edge 3) 4 [1439] 1439 root225Tail18 winner := by
  rw [root225Tail17, scan, if_neg (by decide), root225Candidates17_checked]
  change scan edge (search edge 3) 4 [1439] 1439 root225Tail18
    (if 3 ≤ root225Candidates17.length then search edge 3 [1439, 5279] root225Candidates17 6718 winner else winner) = _
  rw [if_neg (by decide)]
def root225Candidates18 : List Nat := [6983]
set_option maxRecDepth 8192 in
theorem root225Candidates18_checked : candidates edge 4 1439 5657 root225Tail19 winner.bound = root225Candidates18 := by
  unfold candidates
  have row : edge 5657 = (SortedNatTree.contains FastRows.row5657) := by
    change Data.row5657.contains = (SortedNatTree.contains FastRows.row5657)
    exact FastRows.row5657_checked.symm
  rw [row]
  decide
theorem root225Step18 : scan edge (search edge 3) 4 [1439] 1439 root225Tail18 winner =
    scan edge (search edge 3) 4 [1439] 1439 root225Tail19 winner := by
  rw [root225Tail18, scan, if_neg (by decide), root225Candidates18_checked]
  change scan edge (search edge 3) 4 [1439] 1439 root225Tail19
    (if 3 ≤ root225Candidates18.length then search edge 3 [1439, 5657] root225Candidates18 7096 winner else winner) = _
  rw [if_neg (by decide)]
def root225Candidates19 : List Nat := [6983]
set_option maxRecDepth 8192 in
theorem root225Candidates19_checked : candidates edge 4 1439 5783 root225Tail20 winner.bound = root225Candidates19 := by
  unfold candidates
  have row : edge 5783 = (SortedNatTree.contains FastRows.row5783) := by
    change Data.row5783.contains = (SortedNatTree.contains FastRows.row5783)
    exact FastRows.row5783_checked.symm
  rw [row]
  decide
theorem root225Step19 : scan edge (search edge 3) 4 [1439] 1439 root225Tail19 winner =
    scan edge (search edge 3) 4 [1439] 1439 root225Tail20 winner := by
  rw [root225Tail19, scan, if_neg (by decide), root225Candidates19_checked]
  change scan edge (search edge 3) 4 [1439] 1439 root225Tail20
    (if 3 ≤ root225Candidates19.length then search edge 3 [1439, 5783] root225Candidates19 7222 winner else winner) = _
  rw [if_neg (by decide)]
theorem root225Scan20 : scan edge (search edge 3) 4 [1439] 1439 root225Tail20 winner = winner := by rw [root225Tail20, scan, if_pos (by decide)]
theorem root225Scan19 : scan edge (search edge 3) 4 [1439] 1439 root225Tail19 winner = winner :=
  (root225Step19).trans root225Scan20
theorem root225Scan18 : scan edge (search edge 3) 4 [1439] 1439 root225Tail18 winner = winner :=
  (root225Step18).trans root225Scan19
theorem root225Scan17 : scan edge (search edge 3) 4 [1439] 1439 root225Tail17 winner = winner :=
  (root225Step17).trans root225Scan18
theorem root225Scan16 : scan edge (search edge 3) 4 [1439] 1439 root225Tail16 winner = winner :=
  (root225Step16).trans root225Scan17
theorem root225Scan15 : scan edge (search edge 3) 4 [1439] 1439 root225Tail15 winner = winner :=
  (root225Step15).trans root225Scan16
theorem root225Scan14 : scan edge (search edge 3) 4 [1439] 1439 root225Tail14 winner = winner :=
  (root225Step14).trans root225Scan15
theorem root225Scan13 : scan edge (search edge 3) 4 [1439] 1439 root225Tail13 winner = winner :=
  (root225Step13).trans root225Scan14
theorem root225Scan12 : scan edge (search edge 3) 4 [1439] 1439 root225Tail12 winner = winner :=
  (root225Step12).trans root225Scan13
theorem root225Scan11 : scan edge (search edge 3) 4 [1439] 1439 root225Tail11 winner = winner :=
  (root225Step11).trans root225Scan12
theorem root225Scan10 : scan edge (search edge 3) 4 [1439] 1439 root225Tail10 winner = winner :=
  (root225Step10).trans root225Scan11
theorem root225Scan9 : scan edge (search edge 3) 4 [1439] 1439 root225Tail9 winner = winner :=
  (root225Step9).trans root225Scan10
theorem root225Scan8 : scan edge (search edge 3) 4 [1439] 1439 root225Tail8 winner = winner :=
  (root225Step8).trans root225Scan9
theorem root225Scan7 : scan edge (search edge 3) 4 [1439] 1439 root225Tail7 winner = winner :=
  (root225Step7).trans root225Scan8
theorem root225Scan6 : scan edge (search edge 3) 4 [1439] 1439 root225Tail6 winner = winner :=
  (root225Step6).trans root225Scan7
theorem root225Scan5 : scan edge (search edge 3) 4 [1439] 1439 root225Tail5 winner = winner :=
  (root225Step5).trans root225Scan6
theorem root225Scan4 : scan edge (search edge 3) 4 [1439] 1439 root225Tail4 winner = winner :=
  (root225Step4).trans root225Scan5
theorem root225Scan3 : scan edge (search edge 3) 4 [1439] 1439 root225Tail3 winner = winner :=
  (root225Step3).trans root225Scan4
theorem root225Scan2 : scan edge (search edge 3) 4 [1439] 1439 root225Tail2 winner = winner :=
  (root225Step2).trans root225Scan3
theorem root225Scan1 : scan edge (search edge 3) 4 [1439] 1439 root225Tail1 winner = winner :=
  (root225Step1).trans root225Scan2
theorem root225Scan0 : scan edge (search edge 3) 4 [1439] 1439 root225Tail0 winner = winner :=
  (root225Step0).trans root225Scan1
set_option maxRecDepth 4096 in
theorem root225Branch : search edge 4 [1439] root225Candidates 1439 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1439] 1439 root225Tail0 winner = winner
  exact root225Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
