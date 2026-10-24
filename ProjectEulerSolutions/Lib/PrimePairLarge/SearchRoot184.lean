import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root184Candidates : List Nat := Data.row1117
set_option maxRecDepth 8192 in
theorem root184Candidates_checked : candidates edge 5 0 1117 tail185 winner.bound = root184Candidates := by
  apply candidates_eq_row edge 5 0 1117 tail185 root184Candidates winner.bound tail185_ordered
  · decide
  · decide
  · funext q
    rfl
def root184Tail67 : List Nat := []
def root184Tail66 : List Nat := 23977 :: root184Tail67
def root184Tail65 : List Nat := 23497 :: root184Tail66
def root184Tail64 : List Nat := 21031 :: root184Tail65
def root184Tail63 : List Nat := 20857 :: root184Tail64
def root184Tail62 : List Nat := 20743 :: root184Tail63
def root184Tail61 : List Nat := 20233 :: root184Tail62
def root184Tail60 : List Nat := 20143 :: root184Tail61
def root184Tail59 : List Nat := 19681 :: root184Tail60
def root184Tail58 : List Nat := 19603 :: root184Tail59
def root184Tail57 : List Nat := 19429 :: root184Tail58
def root184Tail56 : List Nat := 18583 :: root184Tail57
def root184Tail55 : List Nat := 18433 :: root184Tail56
def root184Tail54 : List Nat := 18049 :: root184Tail55
def root184Tail53 : List Nat := 17971 :: root184Tail54
def root184Tail52 : List Nat := 16573 :: root184Tail53
def root184Tail51 : List Nat := 16567 :: root184Tail52
def root184Tail50 : List Nat := 15319 :: root184Tail51
def root184Tail49 : List Nat := 15061 :: root184Tail50
def root184Tail48 : List Nat := 15013 :: root184Tail49
def root184Tail47 : List Nat := 14593 :: root184Tail48
def root184Tail46 : List Nat := 14011 :: root184Tail47
def root184Tail45 : List Nat := 13759 :: root184Tail46
def root184Tail44 : List Nat := 12637 :: root184Tail45
def root184Tail43 : List Nat := 12511 :: root184Tail44
def root184Tail42 : List Nat := 12451 :: root184Tail43
def root184Tail41 : List Nat := 12421 :: root184Tail42
def root184Tail40 : List Nat := 12301 :: root184Tail41
def root184Tail39 : List Nat := 12157 :: root184Tail40
def root184Tail38 : List Nat := 12043 :: root184Tail39
def root184Tail37 : List Nat := 10993 :: root184Tail38
def root184Tail36 : List Nat := 10663 :: root184Tail37
def root184Tail35 : List Nat := 10657 :: root184Tail36
def root184Tail34 : List Nat := 10531 :: root184Tail35
def root184Tail33 : List Nat := 10369 :: root184Tail34
def root184Tail32 : List Nat := 10321 :: root184Tail33
def root184Tail31 : List Nat := 9907 :: root184Tail32
def root184Tail30 : List Nat := 9859 :: root184Tail31
def root184Tail29 : List Nat := 9601 :: root184Tail30
def root184Tail28 : List Nat := 9199 :: root184Tail29
def root184Tail27 : List Nat := 8887 :: root184Tail28
def root184Tail26 : List Nat := 7591 :: root184Tail27
def root184Tail25 : List Nat := 6679 :: root184Tail26
def root184Tail24 : List Nat := 6637 :: root184Tail25
def root184Tail23 : List Nat := 6211 :: root184Tail24
def root184Tail22 : List Nat := 6133 :: root184Tail23
def root184Tail21 : List Nat := 5869 :: root184Tail22
def root184Tail20 : List Nat := 5323 :: root184Tail21
def root184Tail19 : List Nat := 5167 :: root184Tail20
def root184Tail18 : List Nat := 5077 :: root184Tail19
def root184Tail17 : List Nat := 4993 :: root184Tail18
def root184Tail16 : List Nat := 4561 :: root184Tail17
def root184Tail15 : List Nat := 4363 :: root184Tail16
def root184Tail14 : List Nat := 4357 :: root184Tail15
def root184Tail13 : List Nat := 3823 :: root184Tail14
def root184Tail12 : List Nat := 3637 :: root184Tail13
def root184Tail11 : List Nat := 3511 :: root184Tail12
def root184Tail10 : List Nat := 3343 :: root184Tail11
def root184Tail9 : List Nat := 3229 :: root184Tail10
def root184Tail8 : List Nat := 3067 :: root184Tail9
def root184Tail7 : List Nat := 2851 :: root184Tail8
def root184Tail6 : List Nat := 2797 :: root184Tail7
def root184Tail5 : List Nat := 2731 :: root184Tail6
def root184Tail4 : List Nat := 2341 :: root184Tail5
def root184Tail3 : List Nat := 2269 :: root184Tail4
def root184Tail2 : List Nat := 1429 :: root184Tail3
def root184Tail1 : List Nat := 1327 :: root184Tail2
def root184Tail0 : List Nat := 1201 :: root184Tail1
def root184Candidates0 : List Nat := [3229, 3637, 5167]
set_option maxRecDepth 8192 in
theorem root184Candidates0_checked : candidates edge 4 1117 1201 root184Tail1 winner.bound = root184Candidates0 := by
  unfold candidates
  have row : edge 1201 = (SortedNatTree.contains FastRows.row1201) := by
    change Data.row1201.contains = (SortedNatTree.contains FastRows.row1201)
    exact FastRows.row1201_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root184Child0 : search edge 3 [1117, 1201] root184Candidates0 2318 winner = winner := by decide
theorem root184Step0 : scan edge (search edge 3) 4 [1117] 1117 root184Tail0 winner =
    scan edge (search edge 3) 4 [1117] 1117 root184Tail1 winner := by
  rw [root184Tail0, scan, if_neg (by decide), root184Candidates0_checked]
  change scan edge (search edge 3) 4 [1117] 1117 root184Tail1
    (if 3 ≤ root184Candidates0.length then search edge 3 [1117, 1201] root184Candidates0 2318 winner else winner) = _
  rw [if_pos (by decide), root184Child0]
def root184Candidates1 : List Nat := [1429, 4357, 19603, 23497]
set_option maxRecDepth 8192 in
theorem root184Candidates1_checked : candidates edge 4 1117 1327 root184Tail2 winner.bound = root184Candidates1 := by
  unfold candidates
  have row : edge 1327 = (SortedNatTree.contains FastRows.row1327) := by
    change Data.row1327.contains = (SortedNatTree.contains FastRows.row1327)
    exact FastRows.row1327_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root184Child1 : search edge 3 [1117, 1327] root184Candidates1 2444 winner = winner := by decide
theorem root184Step1 : scan edge (search edge 3) 4 [1117] 1117 root184Tail1 winner =
    scan edge (search edge 3) 4 [1117] 1117 root184Tail2 winner := by
  rw [root184Tail1, scan, if_neg (by decide), root184Candidates1_checked]
  change scan edge (search edge 3) 4 [1117] 1117 root184Tail2
    (if 3 ≤ root184Candidates1.length then search edge 3 [1117, 1327] root184Candidates1 2444 winner else winner) = _
  rw [if_pos (by decide), root184Child1]
def root184Candidates2 : List Nat := [6211, 10321]
set_option maxRecDepth 8192 in
theorem root184Candidates2_checked : candidates edge 4 1117 1429 root184Tail3 winner.bound = root184Candidates2 := by
  unfold candidates
  have row : edge 1429 = (SortedNatTree.contains FastRows.row1429) := by
    change Data.row1429.contains = (SortedNatTree.contains FastRows.row1429)
    exact FastRows.row1429_checked.symm
  rw [row]
  decide
theorem root184Step2 : scan edge (search edge 3) 4 [1117] 1117 root184Tail2 winner =
    scan edge (search edge 3) 4 [1117] 1117 root184Tail3 winner := by
  rw [root184Tail2, scan, if_neg (by decide), root184Candidates2_checked]
  change scan edge (search edge 3) 4 [1117] 1117 root184Tail3
    (if 3 ≤ root184Candidates2.length then search edge 3 [1117, 1429] root184Candidates2 2546 winner else winner) = _
  rw [if_neg (by decide)]
def root184Candidates3 : List Nat := [3067]
set_option maxRecDepth 8192 in
theorem root184Candidates3_checked : candidates edge 4 1117 2269 root184Tail4 winner.bound = root184Candidates3 := by
  unfold candidates
  have row : edge 2269 = (SortedNatTree.contains FastRows.row2269) := by
    change Data.row2269.contains = (SortedNatTree.contains FastRows.row2269)
    exact FastRows.row2269_checked.symm
  rw [row]
  decide
theorem root184Step3 : scan edge (search edge 3) 4 [1117] 1117 root184Tail3 winner =
    scan edge (search edge 3) 4 [1117] 1117 root184Tail4 winner := by
  rw [root184Tail3, scan, if_neg (by decide), root184Candidates3_checked]
  change scan edge (search edge 3) 4 [1117] 1117 root184Tail4
    (if 3 ≤ root184Candidates3.length then search edge 3 [1117, 2269] root184Candidates3 3386 winner else winner) = _
  rw [if_neg (by decide)]
def root184Candidates4 : List Nat := [9859, 18433]
set_option maxRecDepth 8192 in
theorem root184Candidates4_checked : candidates edge 4 1117 2341 root184Tail5 winner.bound = root184Candidates4 := by
  unfold candidates
  have row : edge 2341 = (SortedNatTree.contains FastRows.row2341) := by
    change Data.row2341.contains = (SortedNatTree.contains FastRows.row2341)
    exact FastRows.row2341_checked.symm
  rw [row]
  decide
theorem root184Step4 : scan edge (search edge 3) 4 [1117] 1117 root184Tail4 winner =
    scan edge (search edge 3) 4 [1117] 1117 root184Tail5 winner := by
  rw [root184Tail4, scan, if_neg (by decide), root184Candidates4_checked]
  change scan edge (search edge 3) 4 [1117] 1117 root184Tail5
    (if 3 ≤ root184Candidates4.length then search edge 3 [1117, 2341] root184Candidates4 3458 winner else winner) = _
  rw [if_neg (by decide)]
def root184Candidates5 : List Nat := [10531, 18049]
set_option maxRecDepth 8192 in
theorem root184Candidates5_checked : candidates edge 4 1117 2731 root184Tail6 winner.bound = root184Candidates5 := by
  unfold candidates
  have row : edge 2731 = (SortedNatTree.contains FastRows.row2731) := by
    change Data.row2731.contains = (SortedNatTree.contains FastRows.row2731)
    exact FastRows.row2731_checked.symm
  rw [row]
  decide
theorem root184Step5 : scan edge (search edge 3) 4 [1117] 1117 root184Tail5 winner =
    scan edge (search edge 3) 4 [1117] 1117 root184Tail6 winner := by
  rw [root184Tail5, scan, if_neg (by decide), root184Candidates5_checked]
  change scan edge (search edge 3) 4 [1117] 1117 root184Tail6
    (if 3 ≤ root184Candidates5.length then search edge 3 [1117, 2731] root184Candidates5 3848 winner else winner) = _
  rw [if_neg (by decide)]
def root184Candidates6 : List Nat := [4561]
set_option maxRecDepth 8192 in
theorem root184Candidates6_checked : candidates edge 4 1117 2797 root184Tail7 winner.bound = root184Candidates6 := by
  unfold candidates
  have row : edge 2797 = (SortedNatTree.contains FastRows.row2797) := by
    change Data.row2797.contains = (SortedNatTree.contains FastRows.row2797)
    exact FastRows.row2797_checked.symm
  rw [row]
  decide
theorem root184Step6 : scan edge (search edge 3) 4 [1117] 1117 root184Tail6 winner =
    scan edge (search edge 3) 4 [1117] 1117 root184Tail7 winner := by
  rw [root184Tail6, scan, if_neg (by decide), root184Candidates6_checked]
  change scan edge (search edge 3) 4 [1117] 1117 root184Tail7
    (if 3 ≤ root184Candidates6.length then search edge 3 [1117, 2797] root184Candidates6 3914 winner else winner) = _
  rw [if_neg (by decide)]
def root184Candidates7 : List Nat := [12511]
set_option maxRecDepth 8192 in
theorem root184Candidates7_checked : candidates edge 4 1117 2851 root184Tail8 winner.bound = root184Candidates7 := by
  unfold candidates
  have row : edge 2851 = (SortedNatTree.contains FastRows.row2851) := by
    change Data.row2851.contains = (SortedNatTree.contains FastRows.row2851)
    exact FastRows.row2851_checked.symm
  rw [row]
  decide
theorem root184Step7 : scan edge (search edge 3) 4 [1117] 1117 root184Tail7 winner =
    scan edge (search edge 3) 4 [1117] 1117 root184Tail8 winner := by
  rw [root184Tail7, scan, if_neg (by decide), root184Candidates7_checked]
  change scan edge (search edge 3) 4 [1117] 1117 root184Tail8
    (if 3 ≤ root184Candidates7.length then search edge 3 [1117, 2851] root184Candidates7 3968 winner else winner) = _
  rw [if_neg (by decide)]
def root184Candidates8 : List Nat := [5167]
set_option maxRecDepth 8192 in
theorem root184Candidates8_checked : candidates edge 4 1117 3067 root184Tail9 winner.bound = root184Candidates8 := by
  unfold candidates
  have row : edge 3067 = (SortedNatTree.contains FastRows.row3067) := by
    change Data.row3067.contains = (SortedNatTree.contains FastRows.row3067)
    exact FastRows.row3067_checked.symm
  rw [row]
  decide
theorem root184Step8 : scan edge (search edge 3) 4 [1117] 1117 root184Tail8 winner =
    scan edge (search edge 3) 4 [1117] 1117 root184Tail9 winner := by
  rw [root184Tail8, scan, if_neg (by decide), root184Candidates8_checked]
  change scan edge (search edge 3) 4 [1117] 1117 root184Tail9
    (if 3 ≤ root184Candidates8.length then search edge 3 [1117, 3067] root184Candidates8 4184 winner else winner) = _
  rw [if_neg (by decide)]
def root184Candidates9 : List Nat := [4993, 12451, 20743]
set_option maxRecDepth 8192 in
theorem root184Candidates9_checked : candidates edge 4 1117 3229 root184Tail10 winner.bound = root184Candidates9 := by
  unfold candidates
  have row : edge 3229 = (SortedNatTree.contains FastRows.row3229) := by
    change Data.row3229.contains = (SortedNatTree.contains FastRows.row3229)
    exact FastRows.row3229_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root184Child9 : search edge 3 [1117, 3229] root184Candidates9 4346 winner = winner := by decide
theorem root184Step9 : scan edge (search edge 3) 4 [1117] 1117 root184Tail9 winner =
    scan edge (search edge 3) 4 [1117] 1117 root184Tail10 winner := by
  rw [root184Tail9, scan, if_neg (by decide), root184Candidates9_checked]
  change scan edge (search edge 3) 4 [1117] 1117 root184Tail10
    (if 3 ≤ root184Candidates9.length then search edge 3 [1117, 3229] root184Candidates9 4346 winner else winner) = _
  rw [if_pos (by decide), root184Child9]
def root184Candidates10 : List Nat := [3823, 4561, 14593, 16573]
set_option maxRecDepth 8192 in
theorem root184Candidates10_checked : candidates edge 4 1117 3343 root184Tail11 winner.bound = root184Candidates10 := by
  unfold candidates
  have row : edge 3343 = (SortedNatTree.contains FastRows.row3343) := by
    change Data.row3343.contains = (SortedNatTree.contains FastRows.row3343)
    exact FastRows.row3343_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root184Child10 : search edge 3 [1117, 3343] root184Candidates10 4460 winner = winner := by decide
theorem root184Step10 : scan edge (search edge 3) 4 [1117] 1117 root184Tail10 winner =
    scan edge (search edge 3) 4 [1117] 1117 root184Tail11 winner := by
  rw [root184Tail10, scan, if_neg (by decide), root184Candidates10_checked]
  change scan edge (search edge 3) 4 [1117] 1117 root184Tail11
    (if 3 ≤ root184Candidates10.length then search edge 3 [1117, 3343] root184Candidates10 4460 winner else winner) = _
  rw [if_pos (by decide), root184Child10]
def root184Candidates11 : List Nat := [12451, 18433, 19429, 20233, 20857]
set_option maxRecDepth 8192 in
theorem root184Candidates11_checked : candidates edge 4 1117 3511 root184Tail12 winner.bound = root184Candidates11 := by
  unfold candidates
  have row : edge 3511 = (SortedNatTree.contains FastRows.row3511) := by
    change Data.row3511.contains = (SortedNatTree.contains FastRows.row3511)
    exact FastRows.row3511_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root184Child11 : search edge 3 [1117, 3511] root184Candidates11 4628 winner = winner := by decide
theorem root184Step11 : scan edge (search edge 3) 4 [1117] 1117 root184Tail11 winner =
    scan edge (search edge 3) 4 [1117] 1117 root184Tail12 winner := by
  rw [root184Tail11, scan, if_neg (by decide), root184Candidates11_checked]
  change scan edge (search edge 3) 4 [1117] 1117 root184Tail12
    (if 3 ≤ root184Candidates11.length then search edge 3 [1117, 3511] root184Candidates11 4628 winner else winner) = _
  rw [if_pos (by decide), root184Child11]
def root184Candidates12 : List Nat := [12451, 14011]
set_option maxRecDepth 8192 in
theorem root184Candidates12_checked : candidates edge 4 1117 3637 root184Tail13 winner.bound = root184Candidates12 := by
  unfold candidates
  have row : edge 3637 = (SortedNatTree.contains FastRows.row3637) := by
    change Data.row3637.contains = (SortedNatTree.contains FastRows.row3637)
    exact FastRows.row3637_checked.symm
  rw [row]
  decide
theorem root184Step12 : scan edge (search edge 3) 4 [1117] 1117 root184Tail12 winner =
    scan edge (search edge 3) 4 [1117] 1117 root184Tail13 winner := by
  rw [root184Tail12, scan, if_neg (by decide), root184Candidates12_checked]
  change scan edge (search edge 3) 4 [1117] 1117 root184Tail13
    (if 3 ≤ root184Candidates12.length then search edge 3 [1117, 3637] root184Candidates12 4754 winner else winner) = _
  rw [if_neg (by decide)]
def root184Candidates13 : List Nat := [5167]
set_option maxRecDepth 8192 in
theorem root184Candidates13_checked : candidates edge 4 1117 3823 root184Tail14 winner.bound = root184Candidates13 := by
  unfold candidates
  have row : edge 3823 = (SortedNatTree.contains FastRows.row3823) := by
    change Data.row3823.contains = (SortedNatTree.contains FastRows.row3823)
    exact FastRows.row3823_checked.symm
  rw [row]
  decide
theorem root184Step13 : scan edge (search edge 3) 4 [1117] 1117 root184Tail13 winner =
    scan edge (search edge 3) 4 [1117] 1117 root184Tail14 winner := by
  rw [root184Tail13, scan, if_neg (by decide), root184Candidates13_checked]
  change scan edge (search edge 3) 4 [1117] 1117 root184Tail14
    (if 3 ≤ root184Candidates13.length then search edge 3 [1117, 3823] root184Candidates13 4940 winner else winner) = _
  rw [if_neg (by decide)]
def root184Candidates14 : List Nat := [9859, 16573]
set_option maxRecDepth 8192 in
theorem root184Candidates14_checked : candidates edge 4 1117 4357 root184Tail15 winner.bound = root184Candidates14 := by
  unfold candidates
  have row : edge 4357 = (SortedNatTree.contains FastRows.row4357) := by
    change Data.row4357.contains = (SortedNatTree.contains FastRows.row4357)
    exact FastRows.row4357_checked.symm
  rw [row]
  decide
theorem root184Step14 : scan edge (search edge 3) 4 [1117] 1117 root184Tail14 winner =
    scan edge (search edge 3) 4 [1117] 1117 root184Tail15 winner := by
  rw [root184Tail14, scan, if_neg (by decide), root184Candidates14_checked]
  change scan edge (search edge 3) 4 [1117] 1117 root184Tail15
    (if 3 ≤ root184Candidates14.length then search edge 3 [1117, 4357] root184Candidates14 5474 winner else winner) = _
  rw [if_neg (by decide)]
def root184Candidates15 : List Nat := [6133, 9199]
set_option maxRecDepth 8192 in
theorem root184Candidates15_checked : candidates edge 4 1117 4363 root184Tail16 winner.bound = root184Candidates15 := by
  unfold candidates
  have row : edge 4363 = (SortedNatTree.contains FastRows.row4363) := by
    change Data.row4363.contains = (SortedNatTree.contains FastRows.row4363)
    exact FastRows.row4363_checked.symm
  rw [row]
  decide
theorem root184Step15 : scan edge (search edge 3) 4 [1117] 1117 root184Tail15 winner =
    scan edge (search edge 3) 4 [1117] 1117 root184Tail16 winner := by
  rw [root184Tail15, scan, if_neg (by decide), root184Candidates15_checked]
  change scan edge (search edge 3) 4 [1117] 1117 root184Tail16
    (if 3 ≤ root184Candidates15.length then search edge 3 [1117, 4363] root184Candidates15 5480 winner else winner) = _
  rw [if_neg (by decide)]
def root184Candidates16 : List Nat := [9907, 10321, 19429]
set_option maxRecDepth 8192 in
theorem root184Candidates16_checked : candidates edge 4 1117 4561 root184Tail17 winner.bound = root184Candidates16 := by
  unfold candidates
  have row : edge 4561 = (SortedNatTree.contains FastRows.row4561) := by
    change Data.row4561.contains = (SortedNatTree.contains FastRows.row4561)
    exact FastRows.row4561_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root184Child16 : search edge 3 [1117, 4561] root184Candidates16 5678 winner = winner := by decide
theorem root184Step16 : scan edge (search edge 3) 4 [1117] 1117 root184Tail16 winner =
    scan edge (search edge 3) 4 [1117] 1117 root184Tail17 winner := by
  rw [root184Tail16, scan, if_neg (by decide), root184Candidates16_checked]
  change scan edge (search edge 3) 4 [1117] 1117 root184Tail17
    (if 3 ≤ root184Candidates16.length then search edge 3 [1117, 4561] root184Candidates16 5678 winner else winner) = _
  rw [if_pos (by decide), root184Child16]
def root184Candidates17 : List Nat := [8887, 10369]
set_option maxRecDepth 8192 in
theorem root184Candidates17_checked : candidates edge 4 1117 4993 root184Tail18 winner.bound = root184Candidates17 := by
  unfold candidates
  have row : edge 4993 = (SortedNatTree.contains FastRows.row4993) := by
    change Data.row4993.contains = (SortedNatTree.contains FastRows.row4993)
    exact FastRows.row4993_checked.symm
  rw [row]
  decide
theorem root184Step17 : scan edge (search edge 3) 4 [1117] 1117 root184Tail17 winner =
    scan edge (search edge 3) 4 [1117] 1117 root184Tail18 winner := by
  rw [root184Tail17, scan, if_neg (by decide), root184Candidates17_checked]
  change scan edge (search edge 3) 4 [1117] 1117 root184Tail18
    (if 3 ≤ root184Candidates17.length then search edge 3 [1117, 4993] root184Candidates17 6110 winner else winner) = _
  rw [if_neg (by decide)]
def root184Candidates18 : List Nat := [9907, 10663]
set_option maxRecDepth 8192 in
theorem root184Candidates18_checked : candidates edge 4 1117 5077 root184Tail19 winner.bound = root184Candidates18 := by
  unfold candidates
  have row : edge 5077 = (SortedNatTree.contains FastRows.row5077) := by
    change Data.row5077.contains = (SortedNatTree.contains FastRows.row5077)
    exact FastRows.row5077_checked.symm
  rw [row]
  decide
theorem root184Step18 : scan edge (search edge 3) 4 [1117] 1117 root184Tail18 winner =
    scan edge (search edge 3) 4 [1117] 1117 root184Tail19 winner := by
  rw [root184Tail18, scan, if_neg (by decide), root184Candidates18_checked]
  change scan edge (search edge 3) 4 [1117] 1117 root184Tail19
    (if 3 ≤ root184Candidates18.length then search edge 3 [1117, 5077] root184Candidates18 6194 winner else winner) = _
  rw [if_neg (by decide)]
def root184Candidates19 : List Nat := [10657, 15061, 15319]
set_option maxRecDepth 8192 in
theorem root184Candidates19_checked : candidates edge 4 1117 5167 root184Tail20 winner.bound = root184Candidates19 := by
  unfold candidates
  have row : edge 5167 = (SortedNatTree.contains FastRows.row5167) := by
    change Data.row5167.contains = (SortedNatTree.contains FastRows.row5167)
    exact FastRows.row5167_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root184Child19 : search edge 3 [1117, 5167] root184Candidates19 6284 winner = winner := by decide
theorem root184Step19 : scan edge (search edge 3) 4 [1117] 1117 root184Tail19 winner =
    scan edge (search edge 3) 4 [1117] 1117 root184Tail20 winner := by
  rw [root184Tail19, scan, if_neg (by decide), root184Candidates19_checked]
  change scan edge (search edge 3) 4 [1117] 1117 root184Tail20
    (if 3 ≤ root184Candidates19.length then search edge 3 [1117, 5167] root184Candidates19 6284 winner else winner) = _
  rw [if_pos (by decide), root184Child19]
def root184Candidates20 : List Nat := [10657, 13759, 14593]
set_option maxRecDepth 8192 in
theorem root184Candidates20_checked : candidates edge 4 1117 5323 root184Tail21 winner.bound = root184Candidates20 := by
  unfold candidates
  have row : edge 5323 = (SortedNatTree.contains FastRows.row5323) := by
    change Data.row5323.contains = (SortedNatTree.contains FastRows.row5323)
    exact FastRows.row5323_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root184Child20 : search edge 3 [1117, 5323] root184Candidates20 6440 winner = winner := by decide
theorem root184Step20 : scan edge (search edge 3) 4 [1117] 1117 root184Tail20 winner =
    scan edge (search edge 3) 4 [1117] 1117 root184Tail21 winner := by
  rw [root184Tail20, scan, if_neg (by decide), root184Candidates20_checked]
  change scan edge (search edge 3) 4 [1117] 1117 root184Tail21
    (if 3 ≤ root184Candidates20.length then search edge 3 [1117, 5323] root184Candidates20 6440 winner else winner) = _
  rw [if_pos (by decide), root184Child20]
def root184Candidates21 : List Nat := [10657, 10993]
set_option maxRecDepth 8192 in
theorem root184Candidates21_checked : candidates edge 4 1117 5869 root184Tail22 winner.bound = root184Candidates21 := by
  unfold candidates
  have row : edge 5869 = (SortedNatTree.contains FastRows.row5869) := by
    change Data.row5869.contains = (SortedNatTree.contains FastRows.row5869)
    exact FastRows.row5869_checked.symm
  rw [row]
  decide
theorem root184Step21 : scan edge (search edge 3) 4 [1117] 1117 root184Tail21 winner =
    scan edge (search edge 3) 4 [1117] 1117 root184Tail22 winner := by
  rw [root184Tail21, scan, if_neg (by decide), root184Candidates21_checked]
  change scan edge (search edge 3) 4 [1117] 1117 root184Tail22
    (if 3 ≤ root184Candidates21.length then search edge 3 [1117, 5869] root184Candidates21 6986 winner else winner) = _
  rw [if_neg (by decide)]
def root184Candidates22 : List Nat := []
set_option maxRecDepth 8192 in
theorem root184Candidates22_checked : candidates edge 4 1117 6133 root184Tail23 winner.bound = root184Candidates22 := by
  unfold candidates
  have row : edge 6133 = (SortedNatTree.contains FastRows.row6133) := by
    change Data.row6133.contains = (SortedNatTree.contains FastRows.row6133)
    exact FastRows.row6133_checked.symm
  rw [row]
  decide
theorem root184Step22 : scan edge (search edge 3) 4 [1117] 1117 root184Tail22 winner =
    scan edge (search edge 3) 4 [1117] 1117 root184Tail23 winner := by
  rw [root184Tail22, scan, if_neg (by decide), root184Candidates22_checked]
  change scan edge (search edge 3) 4 [1117] 1117 root184Tail23
    (if 3 ≤ root184Candidates22.length then search edge 3 [1117, 6133] root184Candidates22 7250 winner else winner) = _
  rw [if_neg (by decide)]
def root184Candidates23 : List Nat := [8887]
set_option maxRecDepth 8192 in
theorem root184Candidates23_checked : candidates edge 4 1117 6211 root184Tail24 winner.bound = root184Candidates23 := by
  unfold candidates
  have row : edge 6211 = (SortedNatTree.contains FastRows.row6211) := by
    change Data.row6211.contains = (SortedNatTree.contains FastRows.row6211)
    exact FastRows.row6211_checked.symm
  rw [row]
  decide
theorem root184Step23 : scan edge (search edge 3) 4 [1117] 1117 root184Tail23 winner =
    scan edge (search edge 3) 4 [1117] 1117 root184Tail24 winner := by
  rw [root184Tail23, scan, if_neg (by decide), root184Candidates23_checked]
  change scan edge (search edge 3) 4 [1117] 1117 root184Tail24
    (if 3 ≤ root184Candidates23.length then search edge 3 [1117, 6211] root184Candidates23 7328 winner else winner) = _
  rw [if_neg (by decide)]
theorem root184Scan24 : scan edge (search edge 3) 4 [1117] 1117 root184Tail24 winner = winner := by rw [root184Tail24, scan, if_pos (by decide)]
theorem root184Scan23 : scan edge (search edge 3) 4 [1117] 1117 root184Tail23 winner = winner :=
  (root184Step23).trans root184Scan24
theorem root184Scan22 : scan edge (search edge 3) 4 [1117] 1117 root184Tail22 winner = winner :=
  (root184Step22).trans root184Scan23
theorem root184Scan21 : scan edge (search edge 3) 4 [1117] 1117 root184Tail21 winner = winner :=
  (root184Step21).trans root184Scan22
theorem root184Scan20 : scan edge (search edge 3) 4 [1117] 1117 root184Tail20 winner = winner :=
  (root184Step20).trans root184Scan21
theorem root184Scan19 : scan edge (search edge 3) 4 [1117] 1117 root184Tail19 winner = winner :=
  (root184Step19).trans root184Scan20
theorem root184Scan18 : scan edge (search edge 3) 4 [1117] 1117 root184Tail18 winner = winner :=
  (root184Step18).trans root184Scan19
theorem root184Scan17 : scan edge (search edge 3) 4 [1117] 1117 root184Tail17 winner = winner :=
  (root184Step17).trans root184Scan18
theorem root184Scan16 : scan edge (search edge 3) 4 [1117] 1117 root184Tail16 winner = winner :=
  (root184Step16).trans root184Scan17
theorem root184Scan15 : scan edge (search edge 3) 4 [1117] 1117 root184Tail15 winner = winner :=
  (root184Step15).trans root184Scan16
theorem root184Scan14 : scan edge (search edge 3) 4 [1117] 1117 root184Tail14 winner = winner :=
  (root184Step14).trans root184Scan15
theorem root184Scan13 : scan edge (search edge 3) 4 [1117] 1117 root184Tail13 winner = winner :=
  (root184Step13).trans root184Scan14
theorem root184Scan12 : scan edge (search edge 3) 4 [1117] 1117 root184Tail12 winner = winner :=
  (root184Step12).trans root184Scan13
theorem root184Scan11 : scan edge (search edge 3) 4 [1117] 1117 root184Tail11 winner = winner :=
  (root184Step11).trans root184Scan12
theorem root184Scan10 : scan edge (search edge 3) 4 [1117] 1117 root184Tail10 winner = winner :=
  (root184Step10).trans root184Scan11
theorem root184Scan9 : scan edge (search edge 3) 4 [1117] 1117 root184Tail9 winner = winner :=
  (root184Step9).trans root184Scan10
theorem root184Scan8 : scan edge (search edge 3) 4 [1117] 1117 root184Tail8 winner = winner :=
  (root184Step8).trans root184Scan9
theorem root184Scan7 : scan edge (search edge 3) 4 [1117] 1117 root184Tail7 winner = winner :=
  (root184Step7).trans root184Scan8
theorem root184Scan6 : scan edge (search edge 3) 4 [1117] 1117 root184Tail6 winner = winner :=
  (root184Step6).trans root184Scan7
theorem root184Scan5 : scan edge (search edge 3) 4 [1117] 1117 root184Tail5 winner = winner :=
  (root184Step5).trans root184Scan6
theorem root184Scan4 : scan edge (search edge 3) 4 [1117] 1117 root184Tail4 winner = winner :=
  (root184Step4).trans root184Scan5
theorem root184Scan3 : scan edge (search edge 3) 4 [1117] 1117 root184Tail3 winner = winner :=
  (root184Step3).trans root184Scan4
theorem root184Scan2 : scan edge (search edge 3) 4 [1117] 1117 root184Tail2 winner = winner :=
  (root184Step2).trans root184Scan3
theorem root184Scan1 : scan edge (search edge 3) 4 [1117] 1117 root184Tail1 winner = winner :=
  (root184Step1).trans root184Scan2
theorem root184Scan0 : scan edge (search edge 3) 4 [1117] 1117 root184Tail0 winner = winner :=
  (root184Step0).trans root184Scan1
set_option maxRecDepth 4096 in
theorem root184Branch : search edge 4 [1117] root184Candidates 1117 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1117] 1117 root184Tail0 winner = winner
  exact root184Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
