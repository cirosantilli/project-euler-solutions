import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root75Candidates : List Nat := Data.row397
set_option maxRecDepth 8192 in
theorem root75Candidates_checked : candidates edge 5 0 397 tail76 winner.bound = root75Candidates := by
  apply candidates_eq_row edge 5 0 397 tail76 root75Candidates winner.bound tail76_ordered
  · decide
  · decide
  · funext q
    rfl
def root75Tail75 : List Nat := []
def root75Tail74 : List Nat := 25453 :: root75Tail75
def root75Tail73 : List Nat := 25321 :: root75Tail74
def root75Tail72 : List Nat := 25237 :: root75Tail73
def root75Tail71 : List Nat := 24247 :: root75Tail72
def root75Tail70 : List Nat := 23767 :: root75Tail71
def root75Tail69 : List Nat := 23743 :: root75Tail70
def root75Tail68 : List Nat := 23029 :: root75Tail69
def root75Tail67 : List Nat := 22381 :: root75Tail68
def root75Tail66 : List Nat := 21601 :: root75Tail67
def root75Tail65 : List Nat := 21139 :: root75Tail66
def root75Tail64 : List Nat := 20731 :: root75Tail65
def root75Tail63 : List Nat := 20407 :: root75Tail64
def root75Tail62 : List Nat := 20341 :: root75Tail63
def root75Tail61 : List Nat := 18691 :: root75Tail62
def root75Tail60 : List Nat := 18433 :: root75Tail61
def root75Tail59 : List Nat := 17863 :: root75Tail60
def root75Tail58 : List Nat := 17707 :: root75Tail59
def root75Tail57 : List Nat := 17569 :: root75Tail58
def root75Tail56 : List Nat := 17137 :: root75Tail57
def root75Tail55 : List Nat := 16843 :: root75Tail56
def root75Tail54 : List Nat := 16069 :: root75Tail55
def root75Tail53 : List Nat := 15817 :: root75Tail54
def root75Tail52 : List Nat := 15427 :: root75Tail53
def root75Tail51 : List Nat := 14797 :: root75Tail52
def root75Tail50 : List Nat := 14557 :: root75Tail51
def root75Tail49 : List Nat := 14083 :: root75Tail50
def root75Tail48 : List Nat := 14071 :: root75Tail49
def root75Tail47 : List Nat := 13831 :: root75Tail48
def root75Tail46 : List Nat := 13693 :: root75Tail47
def root75Tail45 : List Nat := 13147 :: root75Tail46
def root75Tail44 : List Nat := 12583 :: root75Tail45
def root75Tail43 : List Nat := 12373 :: root75Tail44
def root75Tail42 : List Nat := 11839 :: root75Tail43
def root75Tail41 : List Nat := 11491 :: root75Tail42
def root75Tail40 : List Nat := 10663 :: root75Tail41
def root75Tail39 : List Nat := 10357 :: root75Tail40
def root75Tail38 : List Nat := 10243 :: root75Tail39
def root75Tail37 : List Nat := 8971 :: root75Tail38
def root75Tail36 : List Nat := 8329 :: root75Tail37
def root75Tail35 : List Nat := 8101 :: root75Tail36
def root75Tail34 : List Nat := 7933 :: root75Tail35
def root75Tail33 : List Nat := 7927 :: root75Tail34
def root75Tail32 : List Nat := 7741 :: root75Tail33
def root75Tail31 : List Nat := 7639 :: root75Tail32
def root75Tail30 : List Nat := 7177 :: root75Tail31
def root75Tail29 : List Nat := 6829 :: root75Tail30
def root75Tail28 : List Nat := 6703 :: root75Tail29
def root75Tail27 : List Nat := 6397 :: root75Tail28
def root75Tail26 : List Nat := 6121 :: root75Tail27
def root75Tail25 : List Nat := 5791 :: root75Tail26
def root75Tail24 : List Nat := 5749 :: root75Tail25
def root75Tail23 : List Nat := 5197 :: root75Tail24
def root75Tail22 : List Nat := 4987 :: root75Tail23
def root75Tail21 : List Nat := 4759 :: root75Tail22
def root75Tail20 : List Nat := 4657 :: root75Tail21
def root75Tail19 : List Nat := 4567 :: root75Tail20
def root75Tail18 : List Nat := 4483 :: root75Tail19
def root75Tail17 : List Nat := 4441 :: root75Tail18
def root75Tail16 : List Nat := 4057 :: root75Tail17
def root75Tail15 : List Nat := 3943 :: root75Tail16
def root75Tail14 : List Nat := 3919 :: root75Tail15
def root75Tail13 : List Nat := 3877 :: root75Tail14
def root75Tail12 : List Nat := 3847 :: root75Tail13
def root75Tail11 : List Nat := 3727 :: root75Tail12
def root75Tail10 : List Nat := 3511 :: root75Tail11
def root75Tail9 : List Nat := 2851 :: root75Tail10
def root75Tail8 : List Nat := 2593 :: root75Tail9
def root75Tail7 : List Nat := 2251 :: root75Tail8
def root75Tail6 : List Nat := 1861 :: root75Tail7
def root75Tail5 : List Nat := 1621 :: root75Tail6
def root75Tail4 : List Nat := 1483 :: root75Tail5
def root75Tail3 : List Nat := 1063 :: root75Tail4
def root75Tail2 : List Nat := 907 :: root75Tail3
def root75Tail1 : List Nat := 673 :: root75Tail2
def root75Tail0 : List Nat := 547 :: root75Tail1
def root75Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root75Candidates0_checked : candidates edge 4 397 547 root75Tail1 winner.bound = root75Candidates0 := by
  unfold candidates
  have row : edge 547 = (SortedNatTree.contains FastRows.row547) := by
    change Data.row547.contains = (SortedNatTree.contains FastRows.row547)
    exact FastRows.row547_checked.symm
  rw [row]
  decide
theorem root75Step0 : scan edge (search edge 3) 4 [397] 397 root75Tail0 winner =
    scan edge (search edge 3) 4 [397] 397 root75Tail1 winner := by
  rw [root75Tail0, scan, if_neg (by decide), root75Candidates0_checked]
  change scan edge (search edge 3) 4 [397] 397 root75Tail1
    (if 3 ≤ root75Candidates0.length then search edge 3 [397, 547] root75Candidates0 944 winner else winner) = _
  rw [if_neg (by decide)]
def root75Candidates1 : List Nat := [7177]
set_option maxRecDepth 8192 in
theorem root75Candidates1_checked : candidates edge 4 397 673 root75Tail2 winner.bound = root75Candidates1 := by
  unfold candidates
  have row : edge 673 = (SortedNatTree.contains FastRows.row673) := by
    change Data.row673.contains = (SortedNatTree.contains FastRows.row673)
    exact FastRows.row673_checked.symm
  rw [row]
  decide
theorem root75Step1 : scan edge (search edge 3) 4 [397] 397 root75Tail1 winner =
    scan edge (search edge 3) 4 [397] 397 root75Tail2 winner := by
  rw [root75Tail1, scan, if_neg (by decide), root75Candidates1_checked]
  change scan edge (search edge 3) 4 [397] 397 root75Tail2
    (if 3 ≤ root75Candidates1.length then search edge 3 [397, 673] root75Candidates1 1070 winner else winner) = _
  rw [if_neg (by decide)]
def root75Candidates2 : List Nat := [3919, 4567, 4987]
set_option maxRecDepth 8192 in
theorem root75Candidates2_checked : candidates edge 4 397 907 root75Tail3 winner.bound = root75Candidates2 := by
  unfold candidates
  have row : edge 907 = (SortedNatTree.contains FastRows.row907) := by
    change Data.row907.contains = (SortedNatTree.contains FastRows.row907)
    exact FastRows.row907_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root75Child2 : search edge 3 [397, 907] root75Candidates2 1304 winner = winner := by decide
theorem root75Step2 : scan edge (search edge 3) 4 [397] 397 root75Tail2 winner =
    scan edge (search edge 3) 4 [397] 397 root75Tail3 winner := by
  rw [root75Tail2, scan, if_neg (by decide), root75Candidates2_checked]
  change scan edge (search edge 3) 4 [397] 397 root75Tail3
    (if 3 ≤ root75Candidates2.length then search edge 3 [397, 907] root75Candidates2 1304 winner else winner) = _
  rw [if_pos (by decide), root75Child2]
def root75Candidates3 : List Nat := [3919, 5197, 16843]
set_option maxRecDepth 8192 in
theorem root75Candidates3_checked : candidates edge 4 397 1063 root75Tail4 winner.bound = root75Candidates3 := by
  unfold candidates
  have row : edge 1063 = (SortedNatTree.contains FastRows.row1063) := by
    change Data.row1063.contains = (SortedNatTree.contains FastRows.row1063)
    exact FastRows.row1063_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root75Child3 : search edge 3 [397, 1063] root75Candidates3 1460 winner = winner := by decide
theorem root75Step3 : scan edge (search edge 3) 4 [397] 397 root75Tail3 winner =
    scan edge (search edge 3) 4 [397] 397 root75Tail4 winner := by
  rw [root75Tail3, scan, if_neg (by decide), root75Candidates3_checked]
  change scan edge (search edge 3) 4 [397] 397 root75Tail4
    (if 3 ≤ root75Candidates3.length then search edge 3 [397, 1063] root75Candidates3 1460 winner else winner) = _
  rw [if_pos (by decide), root75Child3]
def root75Candidates4 : List Nat := [2593, 17137]
set_option maxRecDepth 8192 in
theorem root75Candidates4_checked : candidates edge 4 397 1483 root75Tail5 winner.bound = root75Candidates4 := by
  unfold candidates
  have row : edge 1483 = (SortedNatTree.contains FastRows.row1483) := by
    change Data.row1483.contains = (SortedNatTree.contains FastRows.row1483)
    exact FastRows.row1483_checked.symm
  rw [row]
  decide
theorem root75Step4 : scan edge (search edge 3) 4 [397] 397 root75Tail4 winner =
    scan edge (search edge 3) 4 [397] 397 root75Tail5 winner := by
  rw [root75Tail4, scan, if_neg (by decide), root75Candidates4_checked]
  change scan edge (search edge 3) 4 [397] 397 root75Tail5
    (if 3 ≤ root75Candidates4.length then search edge 3 [397, 1483] root75Candidates4 1880 winner else winner) = _
  rw [if_neg (by decide)]
def root75Candidates5 : List Nat := [2851, 3943, 4441, 7639, 7933, 8971]
set_option maxRecDepth 8192 in
theorem root75Candidates5_checked : candidates edge 4 397 1621 root75Tail6 winner.bound = root75Candidates5 := by
  unfold candidates
  have row : edge 1621 = (SortedNatTree.contains FastRows.row1621) := by
    change Data.row1621.contains = (SortedNatTree.contains FastRows.row1621)
    exact FastRows.row1621_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root75Child5 : search edge 3 [397, 1621] root75Candidates5 2018 winner = winner := by decide
theorem root75Step5 : scan edge (search edge 3) 4 [397] 397 root75Tail5 winner =
    scan edge (search edge 3) 4 [397] 397 root75Tail6 winner := by
  rw [root75Tail5, scan, if_neg (by decide), root75Candidates5_checked]
  change scan edge (search edge 3) 4 [397] 397 root75Tail6
    (if 3 ≤ root75Candidates5.length then search edge 3 [397, 1621] root75Candidates5 2018 winner else winner) = _
  rw [if_pos (by decide), root75Child5]
def root75Candidates6 : List Nat := [6121, 17137]
set_option maxRecDepth 8192 in
theorem root75Candidates6_checked : candidates edge 4 397 1861 root75Tail7 winner.bound = root75Candidates6 := by
  unfold candidates
  have row : edge 1861 = (SortedNatTree.contains FastRows.row1861) := by
    change Data.row1861.contains = (SortedNatTree.contains FastRows.row1861)
    exact FastRows.row1861_checked.symm
  rw [row]
  decide
theorem root75Step6 : scan edge (search edge 3) 4 [397] 397 root75Tail6 winner =
    scan edge (search edge 3) 4 [397] 397 root75Tail7 winner := by
  rw [root75Tail6, scan, if_neg (by decide), root75Candidates6_checked]
  change scan edge (search edge 3) 4 [397] 397 root75Tail7
    (if 3 ≤ root75Candidates6.length then search edge 3 [397, 1861] root75Candidates6 2258 winner else winner) = _
  rw [if_neg (by decide)]
def root75Candidates7 : List Nat := [2593, 7927, 11839]
set_option maxRecDepth 8192 in
theorem root75Candidates7_checked : candidates edge 4 397 2251 root75Tail8 winner.bound = root75Candidates7 := by
  unfold candidates
  have row : edge 2251 = (SortedNatTree.contains FastRows.row2251) := by
    change Data.row2251.contains = (SortedNatTree.contains FastRows.row2251)
    exact FastRows.row2251_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root75Child7 : search edge 3 [397, 2251] root75Candidates7 2648 winner = winner := by decide
theorem root75Step7 : scan edge (search edge 3) 4 [397] 397 root75Tail7 winner =
    scan edge (search edge 3) 4 [397] 397 root75Tail8 winner := by
  rw [root75Tail7, scan, if_neg (by decide), root75Candidates7_checked]
  change scan edge (search edge 3) 4 [397] 397 root75Tail8
    (if 3 ≤ root75Candidates7.length then search edge 3 [397, 2251] root75Candidates7 2648 winner else winner) = _
  rw [if_pos (by decide), root75Child7]
def root75Candidates8 : List Nat := [3847, 7927]
set_option maxRecDepth 8192 in
theorem root75Candidates8_checked : candidates edge 4 397 2593 root75Tail9 winner.bound = root75Candidates8 := by
  unfold candidates
  have row : edge 2593 = (SortedNatTree.contains FastRows.row2593) := by
    change Data.row2593.contains = (SortedNatTree.contains FastRows.row2593)
    exact FastRows.row2593_checked.symm
  rw [row]
  decide
theorem root75Step8 : scan edge (search edge 3) 4 [397] 397 root75Tail8 winner =
    scan edge (search edge 3) 4 [397] 397 root75Tail9 winner := by
  rw [root75Tail8, scan, if_neg (by decide), root75Candidates8_checked]
  change scan edge (search edge 3) 4 [397] 397 root75Tail9
    (if 3 ≤ root75Candidates8.length then search edge 3 [397, 2593] root75Candidates8 2990 winner else winner) = _
  rw [if_neg (by decide)]
def root75Candidates9 : List Nat := [11839, 15427]
set_option maxRecDepth 8192 in
theorem root75Candidates9_checked : candidates edge 4 397 2851 root75Tail10 winner.bound = root75Candidates9 := by
  unfold candidates
  have row : edge 2851 = (SortedNatTree.contains FastRows.row2851) := by
    change Data.row2851.contains = (SortedNatTree.contains FastRows.row2851)
    exact FastRows.row2851_checked.symm
  rw [row]
  decide
theorem root75Step9 : scan edge (search edge 3) 4 [397] 397 root75Tail9 winner =
    scan edge (search edge 3) 4 [397] 397 root75Tail10 winner := by
  rw [root75Tail9, scan, if_neg (by decide), root75Candidates9_checked]
  change scan edge (search edge 3) 4 [397] 397 root75Tail10
    (if 3 ≤ root75Candidates9.length then search edge 3 [397, 2851] root75Candidates9 3248 winner else winner) = _
  rw [if_neg (by decide)]
def root75Candidates10 : List Nat := [17707, 18433]
set_option maxRecDepth 8192 in
theorem root75Candidates10_checked : candidates edge 4 397 3511 root75Tail11 winner.bound = root75Candidates10 := by
  unfold candidates
  have row : edge 3511 = (SortedNatTree.contains FastRows.row3511) := by
    change Data.row3511.contains = (SortedNatTree.contains FastRows.row3511)
    exact FastRows.row3511_checked.symm
  rw [row]
  decide
theorem root75Step10 : scan edge (search edge 3) 4 [397] 397 root75Tail10 winner =
    scan edge (search edge 3) 4 [397] 397 root75Tail11 winner := by
  rw [root75Tail10, scan, if_neg (by decide), root75Candidates10_checked]
  change scan edge (search edge 3) 4 [397] 397 root75Tail11
    (if 3 ≤ root75Candidates10.length then search edge 3 [397, 3511] root75Candidates10 3908 winner else winner) = _
  rw [if_neg (by decide)]
def root75Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root75Candidates11_checked : candidates edge 4 397 3727 root75Tail12 winner.bound = root75Candidates11 := by
  unfold candidates
  have row : edge 3727 = (SortedNatTree.contains FastRows.row3727) := by
    change Data.row3727.contains = (SortedNatTree.contains FastRows.row3727)
    exact FastRows.row3727_checked.symm
  rw [row]
  decide
theorem root75Step11 : scan edge (search edge 3) 4 [397] 397 root75Tail11 winner =
    scan edge (search edge 3) 4 [397] 397 root75Tail12 winner := by
  rw [root75Tail11, scan, if_neg (by decide), root75Candidates11_checked]
  change scan edge (search edge 3) 4 [397] 397 root75Tail12
    (if 3 ≤ root75Candidates11.length then search edge 3 [397, 3727] root75Candidates11 4124 winner else winner) = _
  rw [if_neg (by decide)]
def root75Candidates12 : List Nat := [5197]
set_option maxRecDepth 8192 in
theorem root75Candidates12_checked : candidates edge 4 397 3847 root75Tail13 winner.bound = root75Candidates12 := by
  unfold candidates
  have row : edge 3847 = (SortedNatTree.contains FastRows.row3847) := by
    change Data.row3847.contains = (SortedNatTree.contains FastRows.row3847)
    exact FastRows.row3847_checked.symm
  rw [row]
  decide
theorem root75Step12 : scan edge (search edge 3) 4 [397] 397 root75Tail12 winner =
    scan edge (search edge 3) 4 [397] 397 root75Tail13 winner := by
  rw [root75Tail12, scan, if_neg (by decide), root75Candidates12_checked]
  change scan edge (search edge 3) 4 [397] 397 root75Tail13
    (if 3 ≤ root75Candidates12.length then search edge 3 [397, 3847] root75Candidates12 4244 winner else winner) = _
  rw [if_neg (by decide)]
def root75Candidates13 : List Nat := [20731]
set_option maxRecDepth 8192 in
theorem root75Candidates13_checked : candidates edge 4 397 3877 root75Tail14 winner.bound = root75Candidates13 := by
  unfold candidates
  have row : edge 3877 = (SortedNatTree.contains FastRows.row3877) := by
    change Data.row3877.contains = (SortedNatTree.contains FastRows.row3877)
    exact FastRows.row3877_checked.symm
  rw [row]
  decide
theorem root75Step13 : scan edge (search edge 3) 4 [397] 397 root75Tail13 winner =
    scan edge (search edge 3) 4 [397] 397 root75Tail14 winner := by
  rw [root75Tail13, scan, if_neg (by decide), root75Candidates13_checked]
  change scan edge (search edge 3) 4 [397] 397 root75Tail14
    (if 3 ≤ root75Candidates13.length then search edge 3 [397, 3877] root75Candidates13 4274 winner else winner) = _
  rw [if_neg (by decide)]
def root75Candidates14 : List Nat := [4567, 6397, 13831]
set_option maxRecDepth 8192 in
theorem root75Candidates14_checked : candidates edge 4 397 3919 root75Tail15 winner.bound = root75Candidates14 := by
  unfold candidates
  have row : edge 3919 = (SortedNatTree.contains FastRows.row3919) := by
    change Data.row3919.contains = (SortedNatTree.contains FastRows.row3919)
    exact FastRows.row3919_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root75Child14 : search edge 3 [397, 3919] root75Candidates14 4316 winner = winner := by decide
theorem root75Step14 : scan edge (search edge 3) 4 [397] 397 root75Tail14 winner =
    scan edge (search edge 3) 4 [397] 397 root75Tail15 winner := by
  rw [root75Tail14, scan, if_neg (by decide), root75Candidates14_checked]
  change scan edge (search edge 3) 4 [397] 397 root75Tail15
    (if 3 ≤ root75Candidates14.length then search edge 3 [397, 3919] root75Candidates14 4316 winner else winner) = _
  rw [if_pos (by decide), root75Child14]
def root75Candidates15 : List Nat := [6829, 14083]
set_option maxRecDepth 8192 in
theorem root75Candidates15_checked : candidates edge 4 397 3943 root75Tail16 winner.bound = root75Candidates15 := by
  unfold candidates
  have row : edge 3943 = (SortedNatTree.contains FastRows.row3943) := by
    change Data.row3943.contains = (SortedNatTree.contains FastRows.row3943)
    exact FastRows.row3943_checked.symm
  rw [row]
  decide
theorem root75Step15 : scan edge (search edge 3) 4 [397] 397 root75Tail15 winner =
    scan edge (search edge 3) 4 [397] 397 root75Tail16 winner := by
  rw [root75Tail15, scan, if_neg (by decide), root75Candidates15_checked]
  change scan edge (search edge 3) 4 [397] 397 root75Tail16
    (if 3 ≤ root75Candidates15.length then search edge 3 [397, 3943] root75Candidates15 4340 winner else winner) = _
  rw [if_neg (by decide)]
def root75Candidates16 : List Nat := [4483, 12583]
set_option maxRecDepth 8192 in
theorem root75Candidates16_checked : candidates edge 4 397 4057 root75Tail17 winner.bound = root75Candidates16 := by
  unfold candidates
  have row : edge 4057 = (SortedNatTree.contains FastRows.row4057) := by
    change Data.row4057.contains = (SortedNatTree.contains FastRows.row4057)
    exact FastRows.row4057_checked.symm
  rw [row]
  decide
theorem root75Step16 : scan edge (search edge 3) 4 [397] 397 root75Tail16 winner =
    scan edge (search edge 3) 4 [397] 397 root75Tail17 winner := by
  rw [root75Tail16, scan, if_neg (by decide), root75Candidates16_checked]
  change scan edge (search edge 3) 4 [397] 397 root75Tail17
    (if 3 ≤ root75Candidates16.length then search edge 3 [397, 4057] root75Candidates16 4454 winner else winner) = _
  rw [if_neg (by decide)]
def root75Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root75Candidates17_checked : candidates edge 4 397 4441 root75Tail18 winner.bound = root75Candidates17 := by
  unfold candidates
  have row : edge 4441 = (SortedNatTree.contains FastRows.row4441) := by
    change Data.row4441.contains = (SortedNatTree.contains FastRows.row4441)
    exact FastRows.row4441_checked.symm
  rw [row]
  decide
theorem root75Step17 : scan edge (search edge 3) 4 [397] 397 root75Tail17 winner =
    scan edge (search edge 3) 4 [397] 397 root75Tail18 winner := by
  rw [root75Tail17, scan, if_neg (by decide), root75Candidates17_checked]
  change scan edge (search edge 3) 4 [397] 397 root75Tail18
    (if 3 ≤ root75Candidates17.length then search edge 3 [397, 4441] root75Candidates17 4838 winner else winner) = _
  rw [if_neg (by decide)]
def root75Candidates18 : List Nat := [20407]
set_option maxRecDepth 8192 in
theorem root75Candidates18_checked : candidates edge 4 397 4483 root75Tail19 winner.bound = root75Candidates18 := by
  unfold candidates
  have row : edge 4483 = (SortedNatTree.contains FastRows.row4483) := by
    change Data.row4483.contains = (SortedNatTree.contains FastRows.row4483)
    exact FastRows.row4483_checked.symm
  rw [row]
  decide
theorem root75Step18 : scan edge (search edge 3) 4 [397] 397 root75Tail18 winner =
    scan edge (search edge 3) 4 [397] 397 root75Tail19 winner := by
  rw [root75Tail18, scan, if_neg (by decide), root75Candidates18_checked]
  change scan edge (search edge 3) 4 [397] 397 root75Tail19
    (if 3 ≤ root75Candidates18.length then search edge 3 [397, 4483] root75Candidates18 4880 winner else winner) = _
  rw [if_neg (by decide)]
def root75Candidates19 : List Nat := [18433]
set_option maxRecDepth 8192 in
theorem root75Candidates19_checked : candidates edge 4 397 4567 root75Tail20 winner.bound = root75Candidates19 := by
  unfold candidates
  have row : edge 4567 = (SortedNatTree.contains FastRows.row4567) := by
    change Data.row4567.contains = (SortedNatTree.contains FastRows.row4567)
    exact FastRows.row4567_checked.symm
  rw [row]
  decide
theorem root75Step19 : scan edge (search edge 3) 4 [397] 397 root75Tail19 winner =
    scan edge (search edge 3) 4 [397] 397 root75Tail20 winner := by
  rw [root75Tail19, scan, if_neg (by decide), root75Candidates19_checked]
  change scan edge (search edge 3) 4 [397] 397 root75Tail20
    (if 3 ≤ root75Candidates19.length then search edge 3 [397, 4567] root75Candidates19 4964 winner else winner) = _
  rw [if_neg (by decide)]
def root75Candidates20 : List Nat := [7177, 7933, 8971, 14797]
set_option maxRecDepth 8192 in
theorem root75Candidates20_checked : candidates edge 4 397 4657 root75Tail21 winner.bound = root75Candidates20 := by
  unfold candidates
  have row : edge 4657 = (SortedNatTree.contains FastRows.row4657) := by
    change Data.row4657.contains = (SortedNatTree.contains FastRows.row4657)
    exact FastRows.row4657_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root75Child20 : search edge 3 [397, 4657] root75Candidates20 5054 winner = winner := by decide
theorem root75Step20 : scan edge (search edge 3) 4 [397] 397 root75Tail20 winner =
    scan edge (search edge 3) 4 [397] 397 root75Tail21 winner := by
  rw [root75Tail20, scan, if_neg (by decide), root75Candidates20_checked]
  change scan edge (search edge 3) 4 [397] 397 root75Tail21
    (if 3 ≤ root75Candidates20.length then search edge 3 [397, 4657] root75Candidates20 5054 winner else winner) = _
  rw [if_pos (by decide), root75Child20]
def root75Candidates21 : List Nat := [14797]
set_option maxRecDepth 8192 in
theorem root75Candidates21_checked : candidates edge 4 397 4759 root75Tail22 winner.bound = root75Candidates21 := by
  unfold candidates
  have row : edge 4759 = (SortedNatTree.contains FastRows.row4759) := by
    change Data.row4759.contains = (SortedNatTree.contains FastRows.row4759)
    exact FastRows.row4759_checked.symm
  rw [row]
  decide
theorem root75Step21 : scan edge (search edge 3) 4 [397] 397 root75Tail21 winner =
    scan edge (search edge 3) 4 [397] 397 root75Tail22 winner := by
  rw [root75Tail21, scan, if_neg (by decide), root75Candidates21_checked]
  change scan edge (search edge 3) 4 [397] 397 root75Tail22
    (if 3 ≤ root75Candidates21.length then search edge 3 [397, 4759] root75Candidates21 5156 winner else winner) = _
  rw [if_neg (by decide)]
def root75Candidates22 : List Nat := [5197, 12583]
set_option maxRecDepth 8192 in
theorem root75Candidates22_checked : candidates edge 4 397 4987 root75Tail23 winner.bound = root75Candidates22 := by
  unfold candidates
  have row : edge 4987 = (SortedNatTree.contains FastRows.row4987) := by
    change Data.row4987.contains = (SortedNatTree.contains FastRows.row4987)
    exact FastRows.row4987_checked.symm
  rw [row]
  decide
theorem root75Step22 : scan edge (search edge 3) 4 [397] 397 root75Tail22 winner =
    scan edge (search edge 3) 4 [397] 397 root75Tail23 winner := by
  rw [root75Tail22, scan, if_neg (by decide), root75Candidates22_checked]
  change scan edge (search edge 3) 4 [397] 397 root75Tail23
    (if 3 ≤ root75Candidates22.length then search edge 3 [397, 4987] root75Candidates22 5384 winner else winner) = _
  rw [if_neg (by decide)]
def root75Candidates23 : List Nat := []
set_option maxRecDepth 8192 in
theorem root75Candidates23_checked : candidates edge 4 397 5197 root75Tail24 winner.bound = root75Candidates23 := by
  unfold candidates
  have row : edge 5197 = (SortedNatTree.contains FastRows.row5197) := by
    change Data.row5197.contains = (SortedNatTree.contains FastRows.row5197)
    exact FastRows.row5197_checked.symm
  rw [row]
  decide
theorem root75Step23 : scan edge (search edge 3) 4 [397] 397 root75Tail23 winner =
    scan edge (search edge 3) 4 [397] 397 root75Tail24 winner := by
  rw [root75Tail23, scan, if_neg (by decide), root75Candidates23_checked]
  change scan edge (search edge 3) 4 [397] 397 root75Tail24
    (if 3 ≤ root75Candidates23.length then search edge 3 [397, 5197] root75Candidates23 5594 winner else winner) = _
  rw [if_neg (by decide)]
def root75Candidates24 : List Nat := [8101]
set_option maxRecDepth 8192 in
theorem root75Candidates24_checked : candidates edge 4 397 5749 root75Tail25 winner.bound = root75Candidates24 := by
  unfold candidates
  have row : edge 5749 = (SortedNatTree.contains FastRows.row5749) := by
    change Data.row5749.contains = (SortedNatTree.contains FastRows.row5749)
    exact FastRows.row5749_checked.symm
  rw [row]
  decide
theorem root75Step24 : scan edge (search edge 3) 4 [397] 397 root75Tail24 winner =
    scan edge (search edge 3) 4 [397] 397 root75Tail25 winner := by
  rw [root75Tail24, scan, if_neg (by decide), root75Candidates24_checked]
  change scan edge (search edge 3) 4 [397] 397 root75Tail25
    (if 3 ≤ root75Candidates24.length then search edge 3 [397, 5749] root75Candidates24 6146 winner else winner) = _
  rw [if_neg (by decide)]
def root75Candidates25 : List Nat := [10243, 13693, 16843, 17569]
set_option maxRecDepth 8192 in
theorem root75Candidates25_checked : candidates edge 4 397 5791 root75Tail26 winner.bound = root75Candidates25 := by
  unfold candidates
  have row : edge 5791 = (SortedNatTree.contains FastRows.row5791) := by
    change Data.row5791.contains = (SortedNatTree.contains FastRows.row5791)
    exact FastRows.row5791_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root75Child25 : search edge 3 [397, 5791] root75Candidates25 6188 winner = winner := by decide
theorem root75Step25 : scan edge (search edge 3) 4 [397] 397 root75Tail25 winner =
    scan edge (search edge 3) 4 [397] 397 root75Tail26 winner := by
  rw [root75Tail25, scan, if_neg (by decide), root75Candidates25_checked]
  change scan edge (search edge 3) 4 [397] 397 root75Tail26
    (if 3 ≤ root75Candidates25.length then search edge 3 [397, 5791] root75Candidates25 6188 winner else winner) = _
  rw [if_pos (by decide), root75Child25]
def root75Candidates26 : List Nat := []
set_option maxRecDepth 8192 in
theorem root75Candidates26_checked : candidates edge 4 397 6121 root75Tail27 winner.bound = root75Candidates26 := by
  unfold candidates
  have row : edge 6121 = (SortedNatTree.contains FastRows.row6121) := by
    change Data.row6121.contains = (SortedNatTree.contains FastRows.row6121)
    exact FastRows.row6121_checked.symm
  rw [row]
  decide
theorem root75Step26 : scan edge (search edge 3) 4 [397] 397 root75Tail26 winner =
    scan edge (search edge 3) 4 [397] 397 root75Tail27 winner := by
  rw [root75Tail26, scan, if_neg (by decide), root75Candidates26_checked]
  change scan edge (search edge 3) 4 [397] 397 root75Tail27
    (if 3 ≤ root75Candidates26.length then search edge 3 [397, 6121] root75Candidates26 6518 winner else winner) = _
  rw [if_neg (by decide)]
def root75Candidates27 : List Nat := [14797, 17569]
set_option maxRecDepth 8192 in
theorem root75Candidates27_checked : candidates edge 4 397 6397 root75Tail28 winner.bound = root75Candidates27 := by
  unfold candidates
  have row : edge 6397 = (SortedNatTree.contains FastRows.row6397) := by
    change Data.row6397.contains = (SortedNatTree.contains FastRows.row6397)
    exact FastRows.row6397_checked.symm
  rw [row]
  decide
theorem root75Step27 : scan edge (search edge 3) 4 [397] 397 root75Tail27 winner =
    scan edge (search edge 3) 4 [397] 397 root75Tail28 winner := by
  rw [root75Tail27, scan, if_neg (by decide), root75Candidates27_checked]
  change scan edge (search edge 3) 4 [397] 397 root75Tail28
    (if 3 ≤ root75Candidates27.length then search edge 3 [397, 6397] root75Candidates27 6794 winner else winner) = _
  rw [if_neg (by decide)]
theorem root75Scan28 : scan edge (search edge 3) 4 [397] 397 root75Tail28 winner = winner := by rw [root75Tail28, scan, if_pos (by decide)]
theorem root75Scan27 : scan edge (search edge 3) 4 [397] 397 root75Tail27 winner = winner :=
  (root75Step27).trans root75Scan28
theorem root75Scan26 : scan edge (search edge 3) 4 [397] 397 root75Tail26 winner = winner :=
  (root75Step26).trans root75Scan27
theorem root75Scan25 : scan edge (search edge 3) 4 [397] 397 root75Tail25 winner = winner :=
  (root75Step25).trans root75Scan26
theorem root75Scan24 : scan edge (search edge 3) 4 [397] 397 root75Tail24 winner = winner :=
  (root75Step24).trans root75Scan25
theorem root75Scan23 : scan edge (search edge 3) 4 [397] 397 root75Tail23 winner = winner :=
  (root75Step23).trans root75Scan24
theorem root75Scan22 : scan edge (search edge 3) 4 [397] 397 root75Tail22 winner = winner :=
  (root75Step22).trans root75Scan23
theorem root75Scan21 : scan edge (search edge 3) 4 [397] 397 root75Tail21 winner = winner :=
  (root75Step21).trans root75Scan22
theorem root75Scan20 : scan edge (search edge 3) 4 [397] 397 root75Tail20 winner = winner :=
  (root75Step20).trans root75Scan21
theorem root75Scan19 : scan edge (search edge 3) 4 [397] 397 root75Tail19 winner = winner :=
  (root75Step19).trans root75Scan20
theorem root75Scan18 : scan edge (search edge 3) 4 [397] 397 root75Tail18 winner = winner :=
  (root75Step18).trans root75Scan19
theorem root75Scan17 : scan edge (search edge 3) 4 [397] 397 root75Tail17 winner = winner :=
  (root75Step17).trans root75Scan18
theorem root75Scan16 : scan edge (search edge 3) 4 [397] 397 root75Tail16 winner = winner :=
  (root75Step16).trans root75Scan17
theorem root75Scan15 : scan edge (search edge 3) 4 [397] 397 root75Tail15 winner = winner :=
  (root75Step15).trans root75Scan16
theorem root75Scan14 : scan edge (search edge 3) 4 [397] 397 root75Tail14 winner = winner :=
  (root75Step14).trans root75Scan15
theorem root75Scan13 : scan edge (search edge 3) 4 [397] 397 root75Tail13 winner = winner :=
  (root75Step13).trans root75Scan14
theorem root75Scan12 : scan edge (search edge 3) 4 [397] 397 root75Tail12 winner = winner :=
  (root75Step12).trans root75Scan13
theorem root75Scan11 : scan edge (search edge 3) 4 [397] 397 root75Tail11 winner = winner :=
  (root75Step11).trans root75Scan12
theorem root75Scan10 : scan edge (search edge 3) 4 [397] 397 root75Tail10 winner = winner :=
  (root75Step10).trans root75Scan11
theorem root75Scan9 : scan edge (search edge 3) 4 [397] 397 root75Tail9 winner = winner :=
  (root75Step9).trans root75Scan10
theorem root75Scan8 : scan edge (search edge 3) 4 [397] 397 root75Tail8 winner = winner :=
  (root75Step8).trans root75Scan9
theorem root75Scan7 : scan edge (search edge 3) 4 [397] 397 root75Tail7 winner = winner :=
  (root75Step7).trans root75Scan8
theorem root75Scan6 : scan edge (search edge 3) 4 [397] 397 root75Tail6 winner = winner :=
  (root75Step6).trans root75Scan7
theorem root75Scan5 : scan edge (search edge 3) 4 [397] 397 root75Tail5 winner = winner :=
  (root75Step5).trans root75Scan6
theorem root75Scan4 : scan edge (search edge 3) 4 [397] 397 root75Tail4 winner = winner :=
  (root75Step4).trans root75Scan5
theorem root75Scan3 : scan edge (search edge 3) 4 [397] 397 root75Tail3 winner = winner :=
  (root75Step3).trans root75Scan4
theorem root75Scan2 : scan edge (search edge 3) 4 [397] 397 root75Tail2 winner = winner :=
  (root75Step2).trans root75Scan3
theorem root75Scan1 : scan edge (search edge 3) 4 [397] 397 root75Tail1 winner = winner :=
  (root75Step1).trans root75Scan2
theorem root75Scan0 : scan edge (search edge 3) 4 [397] 397 root75Tail0 winner = winner :=
  (root75Step0).trans root75Scan1
set_option maxRecDepth 4096 in
theorem root75Branch : search edge 4 [397] root75Candidates 397 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [397] 397 root75Tail0 winner = winner
  exact root75Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
