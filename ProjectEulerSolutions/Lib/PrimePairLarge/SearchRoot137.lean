import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root137Candidates : List Nat := Data.row809
set_option maxRecDepth 8192 in
theorem root137Candidates_checked : candidates edge 5 0 809 tail138 winner.bound = root137Candidates := by
  apply candidates_eq_row edge 5 0 809 tail138 root137Candidates winner.bound tail138_ordered
  · decide
  · decide
  · funext q
    rfl
def root137Tail55 : List Nat := []
def root137Tail54 : List Nat := 25121 :: root137Tail55
def root137Tail53 : List Nat := 24317 :: root137Tail54
def root137Tail52 : List Nat := 23879 :: root137Tail53
def root137Tail51 : List Nat := 23789 :: root137Tail52
def root137Tail50 : List Nat := 23459 :: root137Tail51
def root137Tail49 : List Nat := 22697 :: root137Tail50
def root137Tail48 : List Nat := 22343 :: root137Tail49
def root137Tail47 : List Nat := 22013 :: root137Tail48
def root137Tail46 : List Nat := 21017 :: root137Tail47
def root137Tail45 : List Nat := 19973 :: root137Tail46
def root137Tail44 : List Nat := 19379 :: root137Tail45
def root137Tail43 : List Nat := 18731 :: root137Tail44
def root137Tail42 : List Nat := 18329 :: root137Tail43
def root137Tail41 : List Nat := 17387 :: root137Tail42
def root137Tail40 : List Nat := 16427 :: root137Tail41
def root137Tail39 : List Nat := 15287 :: root137Tail40
def root137Tail38 : List Nat := 14423 :: root137Tail39
def root137Tail37 : List Nat := 13691 :: root137Tail38
def root137Tail36 : List Nat := 12959 :: root137Tail37
def root137Tail35 : List Nat := 12893 :: root137Tail36
def root137Tail34 : List Nat := 12479 :: root137Tail35
def root137Tail33 : List Nat := 12323 :: root137Tail34
def root137Tail32 : List Nat := 12101 :: root137Tail33
def root137Tail31 : List Nat := 11903 :: root137Tail32
def root137Tail30 : List Nat := 11003 :: root137Tail31
def root137Tail29 : List Nat := 10853 :: root137Tail30
def root137Tail28 : List Nat := 10691 :: root137Tail29
def root137Tail27 : List Nat := 10253 :: root137Tail28
def root137Tail26 : List Nat := 10211 :: root137Tail27
def root137Tail25 : List Nat := 10139 :: root137Tail26
def root137Tail24 : List Nat := 9281 :: root137Tail25
def root137Tail23 : List Nat := 9227 :: root137Tail24
def root137Tail22 : List Nat := 6719 :: root137Tail23
def root137Tail21 : List Nat := 5531 :: root137Tail22
def root137Tail20 : List Nat := 5393 :: root137Tail21
def root137Tail19 : List Nat := 5237 :: root137Tail20
def root137Tail18 : List Nat := 4679 :: root137Tail19
def root137Tail17 : List Nat := 4271 :: root137Tail18
def root137Tail16 : List Nat := 3947 :: root137Tail17
def root137Tail15 : List Nat := 3209 :: root137Tail16
def root137Tail14 : List Nat := 3011 :: root137Tail15
def root137Tail13 : List Nat := 2963 :: root137Tail14
def root137Tail12 : List Nat := 2927 :: root137Tail13
def root137Tail11 : List Nat := 2243 :: root137Tail12
def root137Tail10 : List Nat := 2141 :: root137Tail11
def root137Tail9 : List Nat := 1583 :: root137Tail10
def root137Tail8 : List Nat := 1553 :: root137Tail9
def root137Tail7 : List Nat := 1427 :: root137Tail8
def root137Tail6 : List Nat := 1361 :: root137Tail7
def root137Tail5 : List Nat := 1091 :: root137Tail6
def root137Tail4 : List Nat := 983 :: root137Tail5
def root137Tail3 : List Nat := 929 :: root137Tail4
def root137Tail2 : List Nat := 839 :: root137Tail3
def root137Tail1 : List Nat := 827 :: root137Tail2
def root137Tail0 : List Nat := 821 :: root137Tail1
def root137Candidates0 : List Nat := [827, 12101]
set_option maxRecDepth 8192 in
theorem root137Candidates0_checked : candidates edge 4 809 821 root137Tail1 winner.bound = root137Candidates0 := by
  unfold candidates
  have row : edge 821 = (SortedNatTree.contains FastRows.row821) := by
    change Data.row821.contains = (SortedNatTree.contains FastRows.row821)
    exact FastRows.row821_checked.symm
  rw [row]
  decide
theorem root137Step0 : scan edge (search edge 3) 4 [809] 809 root137Tail0 winner =
    scan edge (search edge 3) 4 [809] 809 root137Tail1 winner := by
  rw [root137Tail0, scan, if_neg (by decide), root137Candidates0_checked]
  change scan edge (search edge 3) 4 [809] 809 root137Tail1
    (if 3 ≤ root137Candidates0.length then search edge 3 [809, 821] root137Candidates0 1630 winner else winner) = _
  rw [if_neg (by decide)]
def root137Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root137Candidates1_checked : candidates edge 4 809 827 root137Tail2 winner.bound = root137Candidates1 := by
  unfold candidates
  have row : edge 827 = (SortedNatTree.contains FastRows.row827) := by
    change Data.row827.contains = (SortedNatTree.contains FastRows.row827)
    exact FastRows.row827_checked.symm
  rw [row]
  decide
theorem root137Step1 : scan edge (search edge 3) 4 [809] 809 root137Tail1 winner =
    scan edge (search edge 3) 4 [809] 809 root137Tail2 winner := by
  rw [root137Tail1, scan, if_neg (by decide), root137Candidates1_checked]
  change scan edge (search edge 3) 4 [809] 809 root137Tail2
    (if 3 ≤ root137Candidates1.length then search edge 3 [809, 827] root137Candidates1 1636 winner else winner) = _
  rw [if_neg (by decide)]
def root137Candidates2 : List Nat := [4679, 5237]
set_option maxRecDepth 8192 in
theorem root137Candidates2_checked : candidates edge 4 809 839 root137Tail3 winner.bound = root137Candidates2 := by
  unfold candidates
  have row : edge 839 = (SortedNatTree.contains FastRows.row839) := by
    change Data.row839.contains = (SortedNatTree.contains FastRows.row839)
    exact FastRows.row839_checked.symm
  rw [row]
  decide
theorem root137Step2 : scan edge (search edge 3) 4 [809] 809 root137Tail2 winner =
    scan edge (search edge 3) 4 [809] 809 root137Tail3 winner := by
  rw [root137Tail2, scan, if_neg (by decide), root137Candidates2_checked]
  change scan edge (search edge 3) 4 [809] 809 root137Tail3
    (if 3 ≤ root137Candidates2.length then search edge 3 [809, 839] root137Candidates2 1648 winner else winner) = _
  rw [if_neg (by decide)]
def root137Candidates3 : List Nat := [983, 3209, 22343]
set_option maxRecDepth 8192 in
theorem root137Candidates3_checked : candidates edge 4 809 929 root137Tail4 winner.bound = root137Candidates3 := by
  unfold candidates
  have row : edge 929 = (SortedNatTree.contains FastRows.row929) := by
    change Data.row929.contains = (SortedNatTree.contains FastRows.row929)
    exact FastRows.row929_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root137Child3 : search edge 3 [809, 929] root137Candidates3 1738 winner = winner := by decide
theorem root137Step3 : scan edge (search edge 3) 4 [809] 809 root137Tail3 winner =
    scan edge (search edge 3) 4 [809] 809 root137Tail4 winner := by
  rw [root137Tail3, scan, if_neg (by decide), root137Candidates3_checked]
  change scan edge (search edge 3) 4 [809] 809 root137Tail4
    (if 3 ≤ root137Candidates3.length then search edge 3 [809, 929] root137Candidates3 1738 winner else winner) = _
  rw [if_pos (by decide), root137Child3]
def root137Candidates4 : List Nat := [1091, 6719, 9227]
set_option maxRecDepth 8192 in
theorem root137Candidates4_checked : candidates edge 4 809 983 root137Tail5 winner.bound = root137Candidates4 := by
  unfold candidates
  have row : edge 983 = (SortedNatTree.contains FastRows.row983) := by
    change Data.row983.contains = (SortedNatTree.contains FastRows.row983)
    exact FastRows.row983_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root137Child4 : search edge 3 [809, 983] root137Candidates4 1792 winner = winner := by decide
theorem root137Step4 : scan edge (search edge 3) 4 [809] 809 root137Tail4 winner =
    scan edge (search edge 3) 4 [809] 809 root137Tail5 winner := by
  rw [root137Tail4, scan, if_neg (by decide), root137Candidates4_checked]
  change scan edge (search edge 3) 4 [809] 809 root137Tail5
    (if 3 ≤ root137Candidates4.length then search edge 3 [809, 983] root137Candidates4 1792 winner else winner) = _
  rw [if_pos (by decide), root137Child4]
def root137Candidates5 : List Nat := [11903, 17387]
set_option maxRecDepth 8192 in
theorem root137Candidates5_checked : candidates edge 4 809 1091 root137Tail6 winner.bound = root137Candidates5 := by
  unfold candidates
  have row : edge 1091 = (SortedNatTree.contains FastRows.row1091) := by
    change Data.row1091.contains = (SortedNatTree.contains FastRows.row1091)
    exact FastRows.row1091_checked.symm
  rw [row]
  decide
theorem root137Step5 : scan edge (search edge 3) 4 [809] 809 root137Tail5 winner =
    scan edge (search edge 3) 4 [809] 809 root137Tail6 winner := by
  rw [root137Tail5, scan, if_neg (by decide), root137Candidates5_checked]
  change scan edge (search edge 3) 4 [809] 809 root137Tail6
    (if 3 ≤ root137Candidates5.length then search edge 3 [809, 1091] root137Candidates5 1900 winner else winner) = _
  rw [if_neg (by decide)]
def root137Candidates6 : List Nat := [2141, 3947, 12479, 21017]
set_option maxRecDepth 8192 in
theorem root137Candidates6_checked : candidates edge 4 809 1361 root137Tail7 winner.bound = root137Candidates6 := by
  unfold candidates
  have row : edge 1361 = (SortedNatTree.contains FastRows.row1361) := by
    change Data.row1361.contains = (SortedNatTree.contains FastRows.row1361)
    exact FastRows.row1361_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root137Child6 : search edge 3 [809, 1361] root137Candidates6 2170 winner = winner := by decide
theorem root137Step6 : scan edge (search edge 3) 4 [809] 809 root137Tail6 winner =
    scan edge (search edge 3) 4 [809] 809 root137Tail7 winner := by
  rw [root137Tail6, scan, if_neg (by decide), root137Candidates6_checked]
  change scan edge (search edge 3) 4 [809] 809 root137Tail7
    (if 3 ≤ root137Candidates6.length then search edge 3 [809, 1361] root137Candidates6 2170 winner else winner) = _
  rw [if_pos (by decide), root137Child6]
def root137Candidates7 : List Nat := [3011, 11903]
set_option maxRecDepth 8192 in
theorem root137Candidates7_checked : candidates edge 4 809 1427 root137Tail8 winner.bound = root137Candidates7 := by
  unfold candidates
  have row : edge 1427 = (SortedNatTree.contains FastRows.row1427) := by
    change Data.row1427.contains = (SortedNatTree.contains FastRows.row1427)
    exact FastRows.row1427_checked.symm
  rw [row]
  decide
theorem root137Step7 : scan edge (search edge 3) 4 [809] 809 root137Tail7 winner =
    scan edge (search edge 3) 4 [809] 809 root137Tail8 winner := by
  rw [root137Tail7, scan, if_neg (by decide), root137Candidates7_checked]
  change scan edge (search edge 3) 4 [809] 809 root137Tail8
    (if 3 ≤ root137Candidates7.length then search edge 3 [809, 1427] root137Candidates7 2236 winner else winner) = _
  rw [if_neg (by decide)]
def root137Candidates8 : List Nat := [1583, 15287, 19379]
set_option maxRecDepth 8192 in
theorem root137Candidates8_checked : candidates edge 4 809 1553 root137Tail9 winner.bound = root137Candidates8 := by
  unfold candidates
  have row : edge 1553 = (SortedNatTree.contains FastRows.row1553) := by
    change Data.row1553.contains = (SortedNatTree.contains FastRows.row1553)
    exact FastRows.row1553_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root137Child8 : search edge 3 [809, 1553] root137Candidates8 2362 winner = winner := by decide
theorem root137Step8 : scan edge (search edge 3) 4 [809] 809 root137Tail8 winner =
    scan edge (search edge 3) 4 [809] 809 root137Tail9 winner := by
  rw [root137Tail8, scan, if_neg (by decide), root137Candidates8_checked]
  change scan edge (search edge 3) 4 [809] 809 root137Tail9
    (if 3 ≤ root137Candidates8.length then search edge 3 [809, 1553] root137Candidates8 2362 winner else winner) = _
  rw [if_pos (by decide), root137Child8]
def root137Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root137Candidates9_checked : candidates edge 4 809 1583 root137Tail10 winner.bound = root137Candidates9 := by
  unfold candidates
  have row : edge 1583 = (SortedNatTree.contains FastRows.row1583) := by
    change Data.row1583.contains = (SortedNatTree.contains FastRows.row1583)
    exact FastRows.row1583_checked.symm
  rw [row]
  decide
theorem root137Step9 : scan edge (search edge 3) 4 [809] 809 root137Tail9 winner =
    scan edge (search edge 3) 4 [809] 809 root137Tail10 winner := by
  rw [root137Tail9, scan, if_neg (by decide), root137Candidates9_checked]
  change scan edge (search edge 3) 4 [809] 809 root137Tail10
    (if 3 ≤ root137Candidates9.length then search edge 3 [809, 1583] root137Candidates9 2392 winner else winner) = _
  rw [if_neg (by decide)]
def root137Candidates10 : List Nat := [3947, 10139, 11903]
set_option maxRecDepth 8192 in
theorem root137Candidates10_checked : candidates edge 4 809 2141 root137Tail11 winner.bound = root137Candidates10 := by
  unfold candidates
  have row : edge 2141 = (SortedNatTree.contains FastRows.row2141) := by
    change Data.row2141.contains = (SortedNatTree.contains FastRows.row2141)
    exact FastRows.row2141_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root137Child10 : search edge 3 [809, 2141] root137Candidates10 2950 winner = winner := by decide
theorem root137Step10 : scan edge (search edge 3) 4 [809] 809 root137Tail10 winner =
    scan edge (search edge 3) 4 [809] 809 root137Tail11 winner := by
  rw [root137Tail10, scan, if_neg (by decide), root137Candidates10_checked]
  change scan edge (search edge 3) 4 [809] 809 root137Tail11
    (if 3 ≤ root137Candidates10.length then search edge 3 [809, 2141] root137Candidates10 2950 winner else winner) = _
  rw [if_pos (by decide), root137Child10]
def root137Candidates11 : List Nat := [16427]
set_option maxRecDepth 8192 in
theorem root137Candidates11_checked : candidates edge 4 809 2243 root137Tail12 winner.bound = root137Candidates11 := by
  unfold candidates
  have row : edge 2243 = (SortedNatTree.contains FastRows.row2243) := by
    change Data.row2243.contains = (SortedNatTree.contains FastRows.row2243)
    exact FastRows.row2243_checked.symm
  rw [row]
  decide
theorem root137Step11 : scan edge (search edge 3) 4 [809] 809 root137Tail11 winner =
    scan edge (search edge 3) 4 [809] 809 root137Tail12 winner := by
  rw [root137Tail11, scan, if_neg (by decide), root137Candidates11_checked]
  change scan edge (search edge 3) 4 [809] 809 root137Tail12
    (if 3 ≤ root137Candidates11.length then search edge 3 [809, 2243] root137Candidates11 3052 winner else winner) = _
  rw [if_neg (by decide)]
def root137Candidates12 : List Nat := [5531, 9227, 10853]
set_option maxRecDepth 8192 in
theorem root137Candidates12_checked : candidates edge 4 809 2927 root137Tail13 winner.bound = root137Candidates12 := by
  unfold candidates
  have row : edge 2927 = (SortedNatTree.contains FastRows.row2927) := by
    change Data.row2927.contains = (SortedNatTree.contains FastRows.row2927)
    exact FastRows.row2927_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root137Child12 : search edge 3 [809, 2927] root137Candidates12 3736 winner = winner := by decide
theorem root137Step12 : scan edge (search edge 3) 4 [809] 809 root137Tail12 winner =
    scan edge (search edge 3) 4 [809] 809 root137Tail13 winner := by
  rw [root137Tail12, scan, if_neg (by decide), root137Candidates12_checked]
  change scan edge (search edge 3) 4 [809] 809 root137Tail13
    (if 3 ≤ root137Candidates12.length then search edge 3 [809, 2927] root137Candidates12 3736 winner else winner) = _
  rw [if_pos (by decide), root137Child12]
def root137Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root137Candidates13_checked : candidates edge 4 809 2963 root137Tail14 winner.bound = root137Candidates13 := by
  unfold candidates
  have row : edge 2963 = (SortedNatTree.contains FastRows.row2963) := by
    change Data.row2963.contains = (SortedNatTree.contains FastRows.row2963)
    exact FastRows.row2963_checked.symm
  rw [row]
  decide
theorem root137Step13 : scan edge (search edge 3) 4 [809] 809 root137Tail13 winner =
    scan edge (search edge 3) 4 [809] 809 root137Tail14 winner := by
  rw [root137Tail13, scan, if_neg (by decide), root137Candidates13_checked]
  change scan edge (search edge 3) 4 [809] 809 root137Tail14
    (if 3 ≤ root137Candidates13.length then search edge 3 [809, 2963] root137Candidates13 3772 winner else winner) = _
  rw [if_neg (by decide)]
def root137Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root137Candidates14_checked : candidates edge 4 809 3011 root137Tail15 winner.bound = root137Candidates14 := by
  unfold candidates
  have row : edge 3011 = (SortedNatTree.contains FastRows.row3011) := by
    change Data.row3011.contains = (SortedNatTree.contains FastRows.row3011)
    exact FastRows.row3011_checked.symm
  rw [row]
  decide
theorem root137Step14 : scan edge (search edge 3) 4 [809] 809 root137Tail14 winner =
    scan edge (search edge 3) 4 [809] 809 root137Tail15 winner := by
  rw [root137Tail14, scan, if_neg (by decide), root137Candidates14_checked]
  change scan edge (search edge 3) 4 [809] 809 root137Tail15
    (if 3 ≤ root137Candidates14.length then search edge 3 [809, 3011] root137Candidates14 3820 winner else winner) = _
  rw [if_neg (by decide)]
def root137Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root137Candidates15_checked : candidates edge 4 809 3209 root137Tail16 winner.bound = root137Candidates15 := by
  unfold candidates
  have row : edge 3209 = (SortedNatTree.contains FastRows.row3209) := by
    change Data.row3209.contains = (SortedNatTree.contains FastRows.row3209)
    exact FastRows.row3209_checked.symm
  rw [row]
  decide
theorem root137Step15 : scan edge (search edge 3) 4 [809] 809 root137Tail15 winner =
    scan edge (search edge 3) 4 [809] 809 root137Tail16 winner := by
  rw [root137Tail15, scan, if_neg (by decide), root137Candidates15_checked]
  change scan edge (search edge 3) 4 [809] 809 root137Tail16
    (if 3 ≤ root137Candidates15.length then search edge 3 [809, 3209] root137Candidates15 4018 winner else winner) = _
  rw [if_neg (by decide)]
def root137Candidates16 : List Nat := [4679, 5393]
set_option maxRecDepth 8192 in
theorem root137Candidates16_checked : candidates edge 4 809 3947 root137Tail17 winner.bound = root137Candidates16 := by
  unfold candidates
  have row : edge 3947 = (SortedNatTree.contains FastRows.row3947) := by
    change Data.row3947.contains = (SortedNatTree.contains FastRows.row3947)
    exact FastRows.row3947_checked.symm
  rw [row]
  decide
theorem root137Step16 : scan edge (search edge 3) 4 [809] 809 root137Tail16 winner =
    scan edge (search edge 3) 4 [809] 809 root137Tail17 winner := by
  rw [root137Tail16, scan, if_neg (by decide), root137Candidates16_checked]
  change scan edge (search edge 3) 4 [809] 809 root137Tail17
    (if 3 ≤ root137Candidates16.length then search edge 3 [809, 3947] root137Candidates16 4756 winner else winner) = _
  rw [if_neg (by decide)]
def root137Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root137Candidates17_checked : candidates edge 4 809 4271 root137Tail18 winner.bound = root137Candidates17 := by
  unfold candidates
  have row : edge 4271 = (SortedNatTree.contains FastRows.row4271) := by
    change Data.row4271.contains = (SortedNatTree.contains FastRows.row4271)
    exact FastRows.row4271_checked.symm
  rw [row]
  decide
theorem root137Step17 : scan edge (search edge 3) 4 [809] 809 root137Tail17 winner =
    scan edge (search edge 3) 4 [809] 809 root137Tail18 winner := by
  rw [root137Tail17, scan, if_neg (by decide), root137Candidates17_checked]
  change scan edge (search edge 3) 4 [809] 809 root137Tail18
    (if 3 ≤ root137Candidates17.length then search edge 3 [809, 4271] root137Candidates17 5080 winner else winner) = _
  rw [if_neg (by decide)]
def root137Candidates18 : List Nat := [5393, 10691]
set_option maxRecDepth 8192 in
theorem root137Candidates18_checked : candidates edge 4 809 4679 root137Tail19 winner.bound = root137Candidates18 := by
  unfold candidates
  have row : edge 4679 = (SortedNatTree.contains FastRows.row4679) := by
    change Data.row4679.contains = (SortedNatTree.contains FastRows.row4679)
    exact FastRows.row4679_checked.symm
  rw [row]
  decide
theorem root137Step18 : scan edge (search edge 3) 4 [809] 809 root137Tail18 winner =
    scan edge (search edge 3) 4 [809] 809 root137Tail19 winner := by
  rw [root137Tail18, scan, if_neg (by decide), root137Candidates18_checked]
  change scan edge (search edge 3) 4 [809] 809 root137Tail19
    (if 3 ≤ root137Candidates18.length then search edge 3 [809, 4679] root137Candidates18 5488 winner else winner) = _
  rw [if_neg (by decide)]
def root137Candidates19 : List Nat := []
set_option maxRecDepth 8192 in
theorem root137Candidates19_checked : candidates edge 4 809 5237 root137Tail20 winner.bound = root137Candidates19 := by
  unfold candidates
  have row : edge 5237 = (SortedNatTree.contains FastRows.row5237) := by
    change Data.row5237.contains = (SortedNatTree.contains FastRows.row5237)
    exact FastRows.row5237_checked.symm
  rw [row]
  decide
theorem root137Step19 : scan edge (search edge 3) 4 [809] 809 root137Tail19 winner =
    scan edge (search edge 3) 4 [809] 809 root137Tail20 winner := by
  rw [root137Tail19, scan, if_neg (by decide), root137Candidates19_checked]
  change scan edge (search edge 3) 4 [809] 809 root137Tail20
    (if 3 ≤ root137Candidates19.length then search edge 3 [809, 5237] root137Candidates19 6046 winner else winner) = _
  rw [if_neg (by decide)]
def root137Candidates20 : List Nat := []
set_option maxRecDepth 8192 in
theorem root137Candidates20_checked : candidates edge 4 809 5393 root137Tail21 winner.bound = root137Candidates20 := by
  unfold candidates
  have row : edge 5393 = (SortedNatTree.contains FastRows.row5393) := by
    change Data.row5393.contains = (SortedNatTree.contains FastRows.row5393)
    exact FastRows.row5393_checked.symm
  rw [row]
  decide
theorem root137Step20 : scan edge (search edge 3) 4 [809] 809 root137Tail20 winner =
    scan edge (search edge 3) 4 [809] 809 root137Tail21 winner := by
  rw [root137Tail20, scan, if_neg (by decide), root137Candidates20_checked]
  change scan edge (search edge 3) 4 [809] 809 root137Tail21
    (if 3 ≤ root137Candidates20.length then search edge 3 [809, 5393] root137Candidates20 6202 winner else winner) = _
  rw [if_neg (by decide)]
def root137Candidates21 : List Nat := []
set_option maxRecDepth 8192 in
theorem root137Candidates21_checked : candidates edge 4 809 5531 root137Tail22 winner.bound = root137Candidates21 := by
  unfold candidates
  have row : edge 5531 = (SortedNatTree.contains FastRows.row5531) := by
    change Data.row5531.contains = (SortedNatTree.contains FastRows.row5531)
    exact FastRows.row5531_checked.symm
  rw [row]
  decide
theorem root137Step21 : scan edge (search edge 3) 4 [809] 809 root137Tail21 winner =
    scan edge (search edge 3) 4 [809] 809 root137Tail22 winner := by
  rw [root137Tail21, scan, if_neg (by decide), root137Candidates21_checked]
  change scan edge (search edge 3) 4 [809] 809 root137Tail22
    (if 3 ≤ root137Candidates21.length then search edge 3 [809, 5531] root137Candidates21 6340 winner else winner) = _
  rw [if_neg (by decide)]
theorem root137Scan22 : scan edge (search edge 3) 4 [809] 809 root137Tail22 winner = winner := by rw [root137Tail22, scan, if_pos (by decide)]
theorem root137Scan21 : scan edge (search edge 3) 4 [809] 809 root137Tail21 winner = winner :=
  (root137Step21).trans root137Scan22
theorem root137Scan20 : scan edge (search edge 3) 4 [809] 809 root137Tail20 winner = winner :=
  (root137Step20).trans root137Scan21
theorem root137Scan19 : scan edge (search edge 3) 4 [809] 809 root137Tail19 winner = winner :=
  (root137Step19).trans root137Scan20
theorem root137Scan18 : scan edge (search edge 3) 4 [809] 809 root137Tail18 winner = winner :=
  (root137Step18).trans root137Scan19
theorem root137Scan17 : scan edge (search edge 3) 4 [809] 809 root137Tail17 winner = winner :=
  (root137Step17).trans root137Scan18
theorem root137Scan16 : scan edge (search edge 3) 4 [809] 809 root137Tail16 winner = winner :=
  (root137Step16).trans root137Scan17
theorem root137Scan15 : scan edge (search edge 3) 4 [809] 809 root137Tail15 winner = winner :=
  (root137Step15).trans root137Scan16
theorem root137Scan14 : scan edge (search edge 3) 4 [809] 809 root137Tail14 winner = winner :=
  (root137Step14).trans root137Scan15
theorem root137Scan13 : scan edge (search edge 3) 4 [809] 809 root137Tail13 winner = winner :=
  (root137Step13).trans root137Scan14
theorem root137Scan12 : scan edge (search edge 3) 4 [809] 809 root137Tail12 winner = winner :=
  (root137Step12).trans root137Scan13
theorem root137Scan11 : scan edge (search edge 3) 4 [809] 809 root137Tail11 winner = winner :=
  (root137Step11).trans root137Scan12
theorem root137Scan10 : scan edge (search edge 3) 4 [809] 809 root137Tail10 winner = winner :=
  (root137Step10).trans root137Scan11
theorem root137Scan9 : scan edge (search edge 3) 4 [809] 809 root137Tail9 winner = winner :=
  (root137Step9).trans root137Scan10
theorem root137Scan8 : scan edge (search edge 3) 4 [809] 809 root137Tail8 winner = winner :=
  (root137Step8).trans root137Scan9
theorem root137Scan7 : scan edge (search edge 3) 4 [809] 809 root137Tail7 winner = winner :=
  (root137Step7).trans root137Scan8
theorem root137Scan6 : scan edge (search edge 3) 4 [809] 809 root137Tail6 winner = winner :=
  (root137Step6).trans root137Scan7
theorem root137Scan5 : scan edge (search edge 3) 4 [809] 809 root137Tail5 winner = winner :=
  (root137Step5).trans root137Scan6
theorem root137Scan4 : scan edge (search edge 3) 4 [809] 809 root137Tail4 winner = winner :=
  (root137Step4).trans root137Scan5
theorem root137Scan3 : scan edge (search edge 3) 4 [809] 809 root137Tail3 winner = winner :=
  (root137Step3).trans root137Scan4
theorem root137Scan2 : scan edge (search edge 3) 4 [809] 809 root137Tail2 winner = winner :=
  (root137Step2).trans root137Scan3
theorem root137Scan1 : scan edge (search edge 3) 4 [809] 809 root137Tail1 winner = winner :=
  (root137Step1).trans root137Scan2
theorem root137Scan0 : scan edge (search edge 3) 4 [809] 809 root137Tail0 winner = winner :=
  (root137Step0).trans root137Scan1
set_option maxRecDepth 4096 in
theorem root137Branch : search edge 4 [809] root137Candidates 809 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [809] 809 root137Tail0 winner = winner
  exact root137Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
