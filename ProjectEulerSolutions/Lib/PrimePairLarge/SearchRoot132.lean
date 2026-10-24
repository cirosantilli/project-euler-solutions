import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root132Candidates : List Nat := Data.row761
set_option maxRecDepth 8192 in
theorem root132Candidates_checked : candidates edge 5 0 761 tail133 winner.bound = root132Candidates := by
  apply candidates_eq_row edge 5 0 761 tail133 root132Candidates winner.bound tail133_ordered
  · decide
  · decide
  · funext q
    rfl
def root132Tail64 : List Nat := []
def root132Tail63 : List Nat := 24611 :: root132Tail64
def root132Tail62 : List Nat := 24527 :: root132Tail63
def root132Tail61 : List Nat := 23561 :: root132Tail62
def root132Tail60 : List Nat := 23339 :: root132Tail61
def root132Tail59 : List Nat := 23279 :: root132Tail60
def root132Tail58 : List Nat := 23081 :: root132Tail59
def root132Tail57 : List Nat := 22877 :: root132Tail58
def root132Tail56 : List Nat := 20939 :: root132Tail57
def root132Tail55 : List Nat := 20789 :: root132Tail56
def root132Tail54 : List Nat := 20663 :: root132Tail55
def root132Tail53 : List Nat := 20357 :: root132Tail54
def root132Tail52 : List Nat := 20147 :: root132Tail53
def root132Tail51 : List Nat := 19697 :: root132Tail52
def root132Tail50 : List Nat := 19583 :: root132Tail51
def root132Tail49 : List Nat := 19181 :: root132Tail50
def root132Tail48 : List Nat := 18899 :: root132Tail49
def root132Tail47 : List Nat := 18869 :: root132Tail48
def root132Tail46 : List Nat := 17291 :: root132Tail47
def root132Tail45 : List Nat := 17117 :: root132Tail46
def root132Tail44 : List Nat := 16487 :: root132Tail45
def root132Tail43 : List Nat := 16193 :: root132Tail44
def root132Tail42 : List Nat := 15527 :: root132Tail43
def root132Tail41 : List Nat := 15101 :: root132Tail42
def root132Tail40 : List Nat := 14813 :: root132Tail41
def root132Tail39 : List Nat := 13931 :: root132Tail40
def root132Tail38 : List Nat := 13799 :: root132Tail39
def root132Tail37 : List Nat := 13721 :: root132Tail38
def root132Tail36 : List Nat := 13619 :: root132Tail37
def root132Tail35 : List Nat := 13469 :: root132Tail36
def root132Tail34 : List Nat := 13001 :: root132Tail35
def root132Tail33 : List Nat := 12527 :: root132Tail34
def root132Tail32 : List Nat := 12251 :: root132Tail33
def root132Tail31 : List Nat := 11969 :: root132Tail32
def root132Tail30 : List Nat := 11831 :: root132Tail31
def root132Tail29 : List Nat := 11801 :: root132Tail30
def root132Tail28 : List Nat := 11171 :: root132Tail29
def root132Tail27 : List Nat := 10949 :: root132Tail28
def root132Tail26 : List Nat := 9839 :: root132Tail27
def root132Tail25 : List Nat := 9767 :: root132Tail26
def root132Tail24 : List Nat := 8243 :: root132Tail25
def root132Tail23 : List Nat := 7529 :: root132Tail24
def root132Tail22 : List Nat := 7283 :: root132Tail23
def root132Tail21 : List Nat := 6911 :: root132Tail22
def root132Tail20 : List Nat := 5879 :: root132Tail21
def root132Tail19 : List Nat := 5417 :: root132Tail20
def root132Tail18 : List Nat := 5351 :: root132Tail19
def root132Tail17 : List Nat := 5261 :: root132Tail18
def root132Tail16 : List Nat := 5009 :: root132Tail17
def root132Tail15 : List Nat := 4967 :: root132Tail16
def root132Tail14 : List Nat := 4463 :: root132Tail15
def root132Tail13 : List Nat := 3923 :: root132Tail14
def root132Tail12 : List Nat := 3779 :: root132Tail13
def root132Tail11 : List Nat := 3539 :: root132Tail12
def root132Tail10 : List Nat := 3329 :: root132Tail11
def root132Tail9 : List Nat := 3251 :: root132Tail10
def root132Tail8 : List Nat := 3209 :: root132Tail9
def root132Tail7 : List Nat := 2663 :: root132Tail8
def root132Tail6 : List Nat := 2621 :: root132Tail7
def root132Tail5 : List Nat := 2531 :: root132Tail6
def root132Tail4 : List Nat := 2351 :: root132Tail5
def root132Tail3 : List Nat := 1889 :: root132Tail4
def root132Tail2 : List Nat := 1613 :: root132Tail3
def root132Tail1 : List Nat := 1187 :: root132Tail2
def root132Tail0 : List Nat := 977 :: root132Tail1
def root132Candidates0 : List Nat := [18899]
set_option maxRecDepth 8192 in
theorem root132Candidates0_checked : candidates edge 4 761 977 root132Tail1 winner.bound = root132Candidates0 := by
  unfold candidates
  have row : edge 977 = (SortedNatTree.contains FastRows.row977) := by
    change Data.row977.contains = (SortedNatTree.contains FastRows.row977)
    exact FastRows.row977_checked.symm
  rw [row]
  decide
theorem root132Step0 : scan edge (search edge 3) 4 [761] 761 root132Tail0 winner =
    scan edge (search edge 3) 4 [761] 761 root132Tail1 winner := by
  rw [root132Tail0, scan, if_neg (by decide), root132Candidates0_checked]
  change scan edge (search edge 3) 4 [761] 761 root132Tail1
    (if 3 ≤ root132Candidates0.length then search edge 3 [761, 977] root132Candidates0 1738 winner else winner) = _
  rw [if_neg (by decide)]
def root132Candidates1 : List Nat := [9839]
set_option maxRecDepth 8192 in
theorem root132Candidates1_checked : candidates edge 4 761 1187 root132Tail2 winner.bound = root132Candidates1 := by
  unfold candidates
  have row : edge 1187 = (SortedNatTree.contains FastRows.row1187) := by
    change Data.row1187.contains = (SortedNatTree.contains FastRows.row1187)
    exact FastRows.row1187_checked.symm
  rw [row]
  decide
theorem root132Step1 : scan edge (search edge 3) 4 [761] 761 root132Tail1 winner =
    scan edge (search edge 3) 4 [761] 761 root132Tail2 winner := by
  rw [root132Tail1, scan, if_neg (by decide), root132Candidates1_checked]
  change scan edge (search edge 3) 4 [761] 761 root132Tail2
    (if 3 ≤ root132Candidates1.length then search edge 3 [761, 1187] root132Candidates1 1948 winner else winner) = _
  rw [if_neg (by decide)]
def root132Candidates2 : List Nat := [3779, 4967]
set_option maxRecDepth 8192 in
theorem root132Candidates2_checked : candidates edge 4 761 1613 root132Tail3 winner.bound = root132Candidates2 := by
  unfold candidates
  have row : edge 1613 = (SortedNatTree.contains FastRows.row1613) := by
    change Data.row1613.contains = (SortedNatTree.contains FastRows.row1613)
    exact FastRows.row1613_checked.symm
  rw [row]
  decide
theorem root132Step2 : scan edge (search edge 3) 4 [761] 761 root132Tail2 winner =
    scan edge (search edge 3) 4 [761] 761 root132Tail3 winner := by
  rw [root132Tail2, scan, if_neg (by decide), root132Candidates2_checked]
  change scan edge (search edge 3) 4 [761] 761 root132Tail3
    (if 3 ≤ root132Candidates2.length then search edge 3 [761, 1613] root132Candidates2 2374 winner else winner) = _
  rw [if_neg (by decide)]
def root132Candidates3 : List Nat := [5417, 12527, 13619, 17291, 19697]
set_option maxRecDepth 8192 in
theorem root132Candidates3_checked : candidates edge 4 761 1889 root132Tail4 winner.bound = root132Candidates3 := by
  unfold candidates
  have row : edge 1889 = (SortedNatTree.contains FastRows.row1889) := by
    change Data.row1889.contains = (SortedNatTree.contains FastRows.row1889)
    exact FastRows.row1889_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root132Child3 : search edge 3 [761, 1889] root132Candidates3 2650 winner = winner := by decide
theorem root132Step3 : scan edge (search edge 3) 4 [761] 761 root132Tail3 winner =
    scan edge (search edge 3) 4 [761] 761 root132Tail4 winner := by
  rw [root132Tail3, scan, if_neg (by decide), root132Candidates3_checked]
  change scan edge (search edge 3) 4 [761] 761 root132Tail4
    (if 3 ≤ root132Candidates3.length then search edge 3 [761, 1889] root132Candidates3 2650 winner else winner) = _
  rw [if_pos (by decide), root132Child3]
def root132Candidates4 : List Nat := [11171, 20147]
set_option maxRecDepth 8192 in
theorem root132Candidates4_checked : candidates edge 4 761 2351 root132Tail5 winner.bound = root132Candidates4 := by
  unfold candidates
  have row : edge 2351 = (SortedNatTree.contains FastRows.row2351) := by
    change Data.row2351.contains = (SortedNatTree.contains FastRows.row2351)
    exact FastRows.row2351_checked.symm
  rw [row]
  decide
theorem root132Step4 : scan edge (search edge 3) 4 [761] 761 root132Tail4 winner =
    scan edge (search edge 3) 4 [761] 761 root132Tail5 winner := by
  rw [root132Tail4, scan, if_neg (by decide), root132Candidates4_checked]
  change scan edge (search edge 3) 4 [761] 761 root132Tail5
    (if 3 ≤ root132Candidates4.length then search edge 3 [761, 2351] root132Candidates4 3112 winner else winner) = _
  rw [if_neg (by decide)]
def root132Candidates5 : List Nat := [3539, 4967]
set_option maxRecDepth 8192 in
theorem root132Candidates5_checked : candidates edge 4 761 2531 root132Tail6 winner.bound = root132Candidates5 := by
  unfold candidates
  have row : edge 2531 = (SortedNatTree.contains FastRows.row2531) := by
    change Data.row2531.contains = (SortedNatTree.contains FastRows.row2531)
    exact FastRows.row2531_checked.symm
  rw [row]
  decide
theorem root132Step5 : scan edge (search edge 3) 4 [761] 761 root132Tail5 winner =
    scan edge (search edge 3) 4 [761] 761 root132Tail6 winner := by
  rw [root132Tail5, scan, if_neg (by decide), root132Candidates5_checked]
  change scan edge (search edge 3) 4 [761] 761 root132Tail6
    (if 3 ≤ root132Candidates5.length then search edge 3 [761, 2531] root132Candidates5 3292 winner else winner) = _
  rw [if_neg (by decide)]
def root132Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root132Candidates6_checked : candidates edge 4 761 2621 root132Tail7 winner.bound = root132Candidates6 := by
  unfold candidates
  have row : edge 2621 = (SortedNatTree.contains FastRows.row2621) := by
    change Data.row2621.contains = (SortedNatTree.contains FastRows.row2621)
    exact FastRows.row2621_checked.symm
  rw [row]
  decide
theorem root132Step6 : scan edge (search edge 3) 4 [761] 761 root132Tail6 winner =
    scan edge (search edge 3) 4 [761] 761 root132Tail7 winner := by
  rw [root132Tail6, scan, if_neg (by decide), root132Candidates6_checked]
  change scan edge (search edge 3) 4 [761] 761 root132Tail7
    (if 3 ≤ root132Candidates6.length then search edge 3 [761, 2621] root132Candidates6 3382 winner else winner) = _
  rw [if_neg (by decide)]
def root132Candidates7 : List Nat := [3251, 3779]
set_option maxRecDepth 8192 in
theorem root132Candidates7_checked : candidates edge 4 761 2663 root132Tail8 winner.bound = root132Candidates7 := by
  unfold candidates
  have row : edge 2663 = (SortedNatTree.contains FastRows.row2663) := by
    change Data.row2663.contains = (SortedNatTree.contains FastRows.row2663)
    exact FastRows.row2663_checked.symm
  rw [row]
  decide
theorem root132Step7 : scan edge (search edge 3) 4 [761] 761 root132Tail7 winner =
    scan edge (search edge 3) 4 [761] 761 root132Tail8 winner := by
  rw [root132Tail7, scan, if_neg (by decide), root132Candidates7_checked]
  change scan edge (search edge 3) 4 [761] 761 root132Tail8
    (if 3 ≤ root132Candidates7.length then search edge 3 [761, 2663] root132Candidates7 3424 winner else winner) = _
  rw [if_neg (by decide)]
def root132Candidates8 : List Nat := [5417, 16487]
set_option maxRecDepth 8192 in
theorem root132Candidates8_checked : candidates edge 4 761 3209 root132Tail9 winner.bound = root132Candidates8 := by
  unfold candidates
  have row : edge 3209 = (SortedNatTree.contains FastRows.row3209) := by
    change Data.row3209.contains = (SortedNatTree.contains FastRows.row3209)
    exact FastRows.row3209_checked.symm
  rw [row]
  decide
theorem root132Step8 : scan edge (search edge 3) 4 [761] 761 root132Tail8 winner =
    scan edge (search edge 3) 4 [761] 761 root132Tail9 winner := by
  rw [root132Tail8, scan, if_neg (by decide), root132Candidates8_checked]
  change scan edge (search edge 3) 4 [761] 761 root132Tail9
    (if 3 ≤ root132Candidates8.length then search edge 3 [761, 3209] root132Candidates8 3970 winner else winner) = _
  rw [if_neg (by decide)]
def root132Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root132Candidates9_checked : candidates edge 4 761 3251 root132Tail10 winner.bound = root132Candidates9 := by
  unfold candidates
  have row : edge 3251 = (SortedNatTree.contains FastRows.row3251) := by
    change Data.row3251.contains = (SortedNatTree.contains FastRows.row3251)
    exact FastRows.row3251_checked.symm
  rw [row]
  decide
theorem root132Step9 : scan edge (search edge 3) 4 [761] 761 root132Tail9 winner =
    scan edge (search edge 3) 4 [761] 761 root132Tail10 winner := by
  rw [root132Tail9, scan, if_neg (by decide), root132Candidates9_checked]
  change scan edge (search edge 3) 4 [761] 761 root132Tail10
    (if 3 ≤ root132Candidates9.length then search edge 3 [761, 3251] root132Candidates9 4012 winner else winner) = _
  rw [if_neg (by decide)]
def root132Candidates10 : List Nat := [4967, 16487]
set_option maxRecDepth 8192 in
theorem root132Candidates10_checked : candidates edge 4 761 3329 root132Tail11 winner.bound = root132Candidates10 := by
  unfold candidates
  have row : edge 3329 = (SortedNatTree.contains FastRows.row3329) := by
    change Data.row3329.contains = (SortedNatTree.contains FastRows.row3329)
    exact FastRows.row3329_checked.symm
  rw [row]
  decide
theorem root132Step10 : scan edge (search edge 3) 4 [761] 761 root132Tail10 winner =
    scan edge (search edge 3) 4 [761] 761 root132Tail11 winner := by
  rw [root132Tail10, scan, if_neg (by decide), root132Candidates10_checked]
  change scan edge (search edge 3) 4 [761] 761 root132Tail11
    (if 3 ≤ root132Candidates10.length then search edge 3 [761, 3329] root132Candidates10 4090 winner else winner) = _
  rw [if_neg (by decide)]
def root132Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root132Candidates11_checked : candidates edge 4 761 3539 root132Tail12 winner.bound = root132Candidates11 := by
  unfold candidates
  have row : edge 3539 = (SortedNatTree.contains FastRows.row3539) := by
    change Data.row3539.contains = (SortedNatTree.contains FastRows.row3539)
    exact FastRows.row3539_checked.symm
  rw [row]
  decide
theorem root132Step11 : scan edge (search edge 3) 4 [761] 761 root132Tail11 winner =
    scan edge (search edge 3) 4 [761] 761 root132Tail12 winner := by
  rw [root132Tail11, scan, if_neg (by decide), root132Candidates11_checked]
  change scan edge (search edge 3) 4 [761] 761 root132Tail12
    (if 3 ≤ root132Candidates11.length then search edge 3 [761, 3539] root132Candidates11 4300 winner else winner) = _
  rw [if_neg (by decide)]
def root132Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root132Candidates12_checked : candidates edge 4 761 3779 root132Tail13 winner.bound = root132Candidates12 := by
  unfold candidates
  have row : edge 3779 = (SortedNatTree.contains FastRows.row3779) := by
    change Data.row3779.contains = (SortedNatTree.contains FastRows.row3779)
    exact FastRows.row3779_checked.symm
  rw [row]
  decide
theorem root132Step12 : scan edge (search edge 3) 4 [761] 761 root132Tail12 winner =
    scan edge (search edge 3) 4 [761] 761 root132Tail13 winner := by
  rw [root132Tail12, scan, if_neg (by decide), root132Candidates12_checked]
  change scan edge (search edge 3) 4 [761] 761 root132Tail13
    (if 3 ≤ root132Candidates12.length then search edge 3 [761, 3779] root132Candidates12 4540 winner else winner) = _
  rw [if_neg (by decide)]
def root132Candidates13 : List Nat := [19583]
set_option maxRecDepth 8192 in
theorem root132Candidates13_checked : candidates edge 4 761 3923 root132Tail14 winner.bound = root132Candidates13 := by
  unfold candidates
  have row : edge 3923 = (SortedNatTree.contains FastRows.row3923) := by
    change Data.row3923.contains = (SortedNatTree.contains FastRows.row3923)
    exact FastRows.row3923_checked.symm
  rw [row]
  decide
theorem root132Step13 : scan edge (search edge 3) 4 [761] 761 root132Tail13 winner =
    scan edge (search edge 3) 4 [761] 761 root132Tail14 winner := by
  rw [root132Tail13, scan, if_neg (by decide), root132Candidates13_checked]
  change scan edge (search edge 3) 4 [761] 761 root132Tail14
    (if 3 ≤ root132Candidates13.length then search edge 3 [761, 3923] root132Candidates13 4684 winner else winner) = _
  rw [if_neg (by decide)]
def root132Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root132Candidates14_checked : candidates edge 4 761 4463 root132Tail15 winner.bound = root132Candidates14 := by
  unfold candidates
  have row : edge 4463 = (SortedNatTree.contains FastRows.row4463) := by
    change Data.row4463.contains = (SortedNatTree.contains FastRows.row4463)
    exact FastRows.row4463_checked.symm
  rw [row]
  decide
theorem root132Step14 : scan edge (search edge 3) 4 [761] 761 root132Tail14 winner =
    scan edge (search edge 3) 4 [761] 761 root132Tail15 winner := by
  rw [root132Tail14, scan, if_neg (by decide), root132Candidates14_checked]
  change scan edge (search edge 3) 4 [761] 761 root132Tail15
    (if 3 ≤ root132Candidates14.length then search edge 3 [761, 4463] root132Candidates14 5224 winner else winner) = _
  rw [if_neg (by decide)]
def root132Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root132Candidates15_checked : candidates edge 4 761 4967 root132Tail16 winner.bound = root132Candidates15 := by
  unfold candidates
  have row : edge 4967 = (SortedNatTree.contains FastRows.row4967) := by
    change Data.row4967.contains = (SortedNatTree.contains FastRows.row4967)
    exact FastRows.row4967_checked.symm
  rw [row]
  decide
theorem root132Step15 : scan edge (search edge 3) 4 [761] 761 root132Tail15 winner =
    scan edge (search edge 3) 4 [761] 761 root132Tail16 winner := by
  rw [root132Tail15, scan, if_neg (by decide), root132Candidates15_checked]
  change scan edge (search edge 3) 4 [761] 761 root132Tail16
    (if 3 ≤ root132Candidates15.length then search edge 3 [761, 4967] root132Candidates15 5728 winner else winner) = _
  rw [if_neg (by decide)]
def root132Candidates16 : List Nat := [5351, 9839]
set_option maxRecDepth 8192 in
theorem root132Candidates16_checked : candidates edge 4 761 5009 root132Tail17 winner.bound = root132Candidates16 := by
  unfold candidates
  have row : edge 5009 = (SortedNatTree.contains FastRows.row5009) := by
    change Data.row5009.contains = (SortedNatTree.contains FastRows.row5009)
    exact FastRows.row5009_checked.symm
  rw [row]
  decide
theorem root132Step16 : scan edge (search edge 3) 4 [761] 761 root132Tail16 winner =
    scan edge (search edge 3) 4 [761] 761 root132Tail17 winner := by
  rw [root132Tail16, scan, if_neg (by decide), root132Candidates16_checked]
  change scan edge (search edge 3) 4 [761] 761 root132Tail17
    (if 3 ≤ root132Candidates16.length then search edge 3 [761, 5009] root132Candidates16 5770 winner else winner) = _
  rw [if_neg (by decide)]
def root132Candidates17 : List Nat := [18869]
set_option maxRecDepth 8192 in
theorem root132Candidates17_checked : candidates edge 4 761 5261 root132Tail18 winner.bound = root132Candidates17 := by
  unfold candidates
  have row : edge 5261 = (SortedNatTree.contains FastRows.row5261) := by
    change Data.row5261.contains = (SortedNatTree.contains FastRows.row5261)
    exact FastRows.row5261_checked.symm
  rw [row]
  decide
theorem root132Step17 : scan edge (search edge 3) 4 [761] 761 root132Tail17 winner =
    scan edge (search edge 3) 4 [761] 761 root132Tail18 winner := by
  rw [root132Tail17, scan, if_neg (by decide), root132Candidates17_checked]
  change scan edge (search edge 3) 4 [761] 761 root132Tail18
    (if 3 ≤ root132Candidates17.length then search edge 3 [761, 5261] root132Candidates17 6022 winner else winner) = _
  rw [if_neg (by decide)]
def root132Candidates18 : List Nat := [5879, 15101]
set_option maxRecDepth 8192 in
theorem root132Candidates18_checked : candidates edge 4 761 5351 root132Tail19 winner.bound = root132Candidates18 := by
  unfold candidates
  have row : edge 5351 = (SortedNatTree.contains FastRows.row5351) := by
    change Data.row5351.contains = (SortedNatTree.contains FastRows.row5351)
    exact FastRows.row5351_checked.symm
  rw [row]
  decide
theorem root132Step18 : scan edge (search edge 3) 4 [761] 761 root132Tail18 winner =
    scan edge (search edge 3) 4 [761] 761 root132Tail19 winner := by
  rw [root132Tail18, scan, if_neg (by decide), root132Candidates18_checked]
  change scan edge (search edge 3) 4 [761] 761 root132Tail19
    (if 3 ≤ root132Candidates18.length then search edge 3 [761, 5351] root132Candidates18 6112 winner else winner) = _
  rw [if_neg (by decide)]
def root132Candidates19 : List Nat := []
set_option maxRecDepth 8192 in
theorem root132Candidates19_checked : candidates edge 4 761 5417 root132Tail20 winner.bound = root132Candidates19 := by
  unfold candidates
  have row : edge 5417 = (SortedNatTree.contains FastRows.row5417) := by
    change Data.row5417.contains = (SortedNatTree.contains FastRows.row5417)
    exact FastRows.row5417_checked.symm
  rw [row]
  decide
theorem root132Step19 : scan edge (search edge 3) 4 [761] 761 root132Tail19 winner =
    scan edge (search edge 3) 4 [761] 761 root132Tail20 winner := by
  rw [root132Tail19, scan, if_neg (by decide), root132Candidates19_checked]
  change scan edge (search edge 3) 4 [761] 761 root132Tail20
    (if 3 ≤ root132Candidates19.length then search edge 3 [761, 5417] root132Candidates19 6178 winner else winner) = _
  rw [if_neg (by decide)]
def root132Candidates20 : List Nat := [12251]
set_option maxRecDepth 8192 in
theorem root132Candidates20_checked : candidates edge 4 761 5879 root132Tail21 winner.bound = root132Candidates20 := by
  unfold candidates
  have row : edge 5879 = (SortedNatTree.contains FastRows.row5879) := by
    change Data.row5879.contains = (SortedNatTree.contains FastRows.row5879)
    exact FastRows.row5879_checked.symm
  rw [row]
  decide
theorem root132Step20 : scan edge (search edge 3) 4 [761] 761 root132Tail20 winner =
    scan edge (search edge 3) 4 [761] 761 root132Tail21 winner := by
  rw [root132Tail20, scan, if_neg (by decide), root132Candidates20_checked]
  change scan edge (search edge 3) 4 [761] 761 root132Tail21
    (if 3 ≤ root132Candidates20.length then search edge 3 [761, 5879] root132Candidates20 6640 winner else winner) = _
  rw [if_neg (by decide)]
theorem root132Scan21 : scan edge (search edge 3) 4 [761] 761 root132Tail21 winner = winner := by rw [root132Tail21, scan, if_pos (by decide)]
theorem root132Scan20 : scan edge (search edge 3) 4 [761] 761 root132Tail20 winner = winner :=
  (root132Step20).trans root132Scan21
theorem root132Scan19 : scan edge (search edge 3) 4 [761] 761 root132Tail19 winner = winner :=
  (root132Step19).trans root132Scan20
theorem root132Scan18 : scan edge (search edge 3) 4 [761] 761 root132Tail18 winner = winner :=
  (root132Step18).trans root132Scan19
theorem root132Scan17 : scan edge (search edge 3) 4 [761] 761 root132Tail17 winner = winner :=
  (root132Step17).trans root132Scan18
theorem root132Scan16 : scan edge (search edge 3) 4 [761] 761 root132Tail16 winner = winner :=
  (root132Step16).trans root132Scan17
theorem root132Scan15 : scan edge (search edge 3) 4 [761] 761 root132Tail15 winner = winner :=
  (root132Step15).trans root132Scan16
theorem root132Scan14 : scan edge (search edge 3) 4 [761] 761 root132Tail14 winner = winner :=
  (root132Step14).trans root132Scan15
theorem root132Scan13 : scan edge (search edge 3) 4 [761] 761 root132Tail13 winner = winner :=
  (root132Step13).trans root132Scan14
theorem root132Scan12 : scan edge (search edge 3) 4 [761] 761 root132Tail12 winner = winner :=
  (root132Step12).trans root132Scan13
theorem root132Scan11 : scan edge (search edge 3) 4 [761] 761 root132Tail11 winner = winner :=
  (root132Step11).trans root132Scan12
theorem root132Scan10 : scan edge (search edge 3) 4 [761] 761 root132Tail10 winner = winner :=
  (root132Step10).trans root132Scan11
theorem root132Scan9 : scan edge (search edge 3) 4 [761] 761 root132Tail9 winner = winner :=
  (root132Step9).trans root132Scan10
theorem root132Scan8 : scan edge (search edge 3) 4 [761] 761 root132Tail8 winner = winner :=
  (root132Step8).trans root132Scan9
theorem root132Scan7 : scan edge (search edge 3) 4 [761] 761 root132Tail7 winner = winner :=
  (root132Step7).trans root132Scan8
theorem root132Scan6 : scan edge (search edge 3) 4 [761] 761 root132Tail6 winner = winner :=
  (root132Step6).trans root132Scan7
theorem root132Scan5 : scan edge (search edge 3) 4 [761] 761 root132Tail5 winner = winner :=
  (root132Step5).trans root132Scan6
theorem root132Scan4 : scan edge (search edge 3) 4 [761] 761 root132Tail4 winner = winner :=
  (root132Step4).trans root132Scan5
theorem root132Scan3 : scan edge (search edge 3) 4 [761] 761 root132Tail3 winner = winner :=
  (root132Step3).trans root132Scan4
theorem root132Scan2 : scan edge (search edge 3) 4 [761] 761 root132Tail2 winner = winner :=
  (root132Step2).trans root132Scan3
theorem root132Scan1 : scan edge (search edge 3) 4 [761] 761 root132Tail1 winner = winner :=
  (root132Step1).trans root132Scan2
theorem root132Scan0 : scan edge (search edge 3) 4 [761] 761 root132Tail0 winner = winner :=
  (root132Step0).trans root132Scan1
set_option maxRecDepth 4096 in
theorem root132Branch : search edge 4 [761] root132Candidates 761 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [761] 761 root132Tail0 winner = winner
  exact root132Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
