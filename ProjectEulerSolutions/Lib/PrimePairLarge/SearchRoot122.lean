import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root122Candidates : List Nat := Data.row691
set_option maxRecDepth 8192 in
theorem root122Candidates_checked : candidates edge 5 0 691 tail123 winner.bound = root122Candidates := by
  apply candidates_eq_row edge 5 0 691 tail123 root122Candidates winner.bound tail123_ordered
  · decide
  · decide
  · funext q
    rfl
def root122Tail78 : List Nat := []
def root122Tail77 : List Nat := 25153 :: root122Tail78
def root122Tail76 : List Nat := 24859 :: root122Tail77
def root122Tail75 : List Nat := 24709 :: root122Tail76
def root122Tail74 : List Nat := 24049 :: root122Tail75
def root122Tail73 : List Nat := 23911 :: root122Tail74
def root122Tail72 : List Nat := 23293 :: root122Tail73
def root122Tail71 : List Nat := 23167 :: root122Tail72
def root122Tail70 : List Nat := 23011 :: root122Tail71
def root122Tail69 : List Nat := 22189 :: root122Tail70
def root122Tail68 : List Nat := 21991 :: root122Tail69
def root122Tail67 : List Nat := 21841 :: root122Tail68
def root122Tail66 : List Nat := 21499 :: root122Tail67
def root122Tail65 : List Nat := 20563 :: root122Tail66
def root122Tail64 : List Nat := 20323 :: root122Tail65
def root122Tail63 : List Nat := 20089 :: root122Tail64
def root122Tail62 : List Nat := 20071 :: root122Tail63
def root122Tail61 : List Nat := 19867 :: root122Tail62
def root122Tail60 : List Nat := 19417 :: root122Tail61
def root122Tail59 : List Nat := 18451 :: root122Tail60
def root122Tail58 : List Nat := 18199 :: root122Tail59
def root122Tail57 : List Nat := 17959 :: root122Tail58
def root122Tail56 : List Nat := 17791 :: root122Tail57
def root122Tail55 : List Nat := 17299 :: root122Tail56
def root122Tail54 : List Nat := 16519 :: root122Tail55
def root122Tail53 : List Nat := 16141 :: root122Tail54
def root122Tail52 : List Nat := 15973 :: root122Tail53
def root122Tail51 : List Nat := 15877 :: root122Tail52
def root122Tail50 : List Nat := 14449 :: root122Tail51
def root122Tail49 : List Nat := 14347 :: root122Tail50
def root122Tail48 : List Nat := 14293 :: root122Tail49
def root122Tail47 : List Nat := 13903 :: root122Tail48
def root122Tail46 : List Nat := 13267 :: root122Tail47
def root122Tail45 : List Nat := 12553 :: root122Tail46
def root122Tail44 : List Nat := 11839 :: root122Tail45
def root122Tail43 : List Nat := 11467 :: root122Tail44
def root122Tail42 : List Nat := 11083 :: root122Tail43
def root122Tail41 : List Nat := 10957 :: root122Tail42
def root122Tail40 : List Nat := 10243 :: root122Tail41
def root122Tail39 : List Nat := 9967 :: root122Tail40
def root122Tail38 : List Nat := 9421 :: root122Tail39
def root122Tail37 : List Nat := 9181 :: root122Tail38
def root122Tail36 : List Nat := 9043 :: root122Tail37
def root122Tail35 : List Nat := 8761 :: root122Tail36
def root122Tail34 : List Nat := 8599 :: root122Tail35
def root122Tail33 : List Nat := 8389 :: root122Tail34
def root122Tail32 : List Nat := 7177 :: root122Tail33
def root122Tail31 : List Nat := 6673 :: root122Tail32
def root122Tail30 : List Nat := 6451 :: root122Tail31
def root122Tail29 : List Nat := 6397 :: root122Tail30
def root122Tail28 : List Nat := 6199 :: root122Tail29
def root122Tail27 : List Nat := 5869 :: root122Tail28
def root122Tail26 : List Nat := 5851 :: root122Tail27
def root122Tail25 : List Nat := 5827 :: root122Tail26
def root122Tail24 : List Nat := 5743 :: root122Tail25
def root122Tail23 : List Nat := 5431 :: root122Tail24
def root122Tail22 : List Nat := 5197 :: root122Tail23
def root122Tail21 : List Nat := 4951 :: root122Tail22
def root122Tail20 : List Nat := 4861 :: root122Tail21
def root122Tail19 : List Nat := 4729 :: root122Tail20
def root122Tail18 : List Nat := 4657 :: root122Tail19
def root122Tail17 : List Nat := 4603 :: root122Tail18
def root122Tail16 : List Nat := 3943 :: root122Tail17
def root122Tail15 : List Nat := 3607 :: root122Tail16
def root122Tail14 : List Nat := 3541 :: root122Tail15
def root122Tail13 : List Nat := 3469 :: root122Tail14
def root122Tail12 : List Nat := 3433 :: root122Tail13
def root122Tail11 : List Nat := 3229 :: root122Tail12
def root122Tail10 : List Nat := 3121 :: root122Tail11
def root122Tail9 : List Nat := 2593 :: root122Tail10
def root122Tail8 : List Nat := 2131 :: root122Tail9
def root122Tail7 : List Nat := 1693 :: root122Tail8
def root122Tail6 : List Nat := 1423 :: root122Tail7
def root122Tail5 : List Nat := 1291 :: root122Tail6
def root122Tail4 : List Nat := 1249 :: root122Tail5
def root122Tail3 : List Nat := 919 :: root122Tail4
def root122Tail2 : List Nat := 907 :: root122Tail3
def root122Tail1 : List Nat := 727 :: root122Tail2
def root122Tail0 : List Nat := 709 :: root122Tail1
def root122Candidates0 : List Nat := [3469, 19867, 20089]
set_option maxRecDepth 8192 in
theorem root122Candidates0_checked : candidates edge 4 691 709 root122Tail1 winner.bound = root122Candidates0 := by
  unfold candidates
  have row : edge 709 = (SortedNatTree.contains FastRows.row709) := by
    change Data.row709.contains = (SortedNatTree.contains FastRows.row709)
    exact FastRows.row709_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root122Child0 : search edge 3 [691, 709] root122Candidates0 1400 winner = winner := by decide
theorem root122Step0 : scan edge (search edge 3) 4 [691] 691 root122Tail0 winner =
    scan edge (search edge 3) 4 [691] 691 root122Tail1 winner := by
  rw [root122Tail0, scan, if_neg (by decide), root122Candidates0_checked]
  change scan edge (search edge 3) 4 [691] 691 root122Tail1
    (if 3 ≤ root122Candidates0.length then search edge 3 [691, 709] root122Candidates0 1400 winner else winner) = _
  rw [if_pos (by decide), root122Child0]
def root122Candidates1 : List Nat := [5197, 8389, 14347]
set_option maxRecDepth 8192 in
theorem root122Candidates1_checked : candidates edge 4 691 727 root122Tail2 winner.bound = root122Candidates1 := by
  unfold candidates
  have row : edge 727 = (SortedNatTree.contains FastRows.row727) := by
    change Data.row727.contains = (SortedNatTree.contains FastRows.row727)
    exact FastRows.row727_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root122Child1 : search edge 3 [691, 727] root122Candidates1 1418 winner = winner := by decide
theorem root122Step1 : scan edge (search edge 3) 4 [691] 691 root122Tail1 winner =
    scan edge (search edge 3) 4 [691] 691 root122Tail2 winner := by
  rw [root122Tail1, scan, if_neg (by decide), root122Candidates1_checked]
  change scan edge (search edge 3) 4 [691] 691 root122Tail2
    (if 3 ≤ root122Candidates1.length then search edge 3 [691, 727] root122Candidates1 1418 winner else winner) = _
  rw [if_pos (by decide), root122Child1]
def root122Candidates2 : List Nat := [4861, 5851, 16141, 16519, 17299]
set_option maxRecDepth 8192 in
theorem root122Candidates2_checked : candidates edge 4 691 907 root122Tail3 winner.bound = root122Candidates2 := by
  unfold candidates
  have row : edge 907 = (SortedNatTree.contains FastRows.row907) := by
    change Data.row907.contains = (SortedNatTree.contains FastRows.row907)
    exact FastRows.row907_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root122Child2 : search edge 3 [691, 907] root122Candidates2 1598 winner = winner := by decide
theorem root122Step2 : scan edge (search edge 3) 4 [691] 691 root122Tail2 winner =
    scan edge (search edge 3) 4 [691] 691 root122Tail3 winner := by
  rw [root122Tail2, scan, if_neg (by decide), root122Candidates2_checked]
  change scan edge (search edge 3) 4 [691] 691 root122Tail3
    (if 3 ≤ root122Candidates2.length then search edge 3 [691, 907] root122Candidates2 1598 winner else winner) = _
  rw [if_pos (by decide), root122Child2]
def root122Candidates3 : List Nat := [11467, 14293, 18199, 22189]
set_option maxRecDepth 8192 in
theorem root122Candidates3_checked : candidates edge 4 691 919 root122Tail4 winner.bound = root122Candidates3 := by
  unfold candidates
  have row : edge 919 = (SortedNatTree.contains FastRows.row919) := by
    change Data.row919.contains = (SortedNatTree.contains FastRows.row919)
    exact FastRows.row919_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root122Child3 : search edge 3 [691, 919] root122Candidates3 1610 winner = winner := by decide
theorem root122Step3 : scan edge (search edge 3) 4 [691] 691 root122Tail3 winner =
    scan edge (search edge 3) 4 [691] 691 root122Tail4 winner := by
  rw [root122Tail3, scan, if_neg (by decide), root122Candidates3_checked]
  change scan edge (search edge 3) 4 [691] 691 root122Tail4
    (if 3 ≤ root122Candidates3.length then search edge 3 [691, 919] root122Candidates3 1610 winner else winner) = _
  rw [if_pos (by decide), root122Child3]
def root122Candidates4 : List Nat := [2131, 5851]
set_option maxRecDepth 8192 in
theorem root122Candidates4_checked : candidates edge 4 691 1249 root122Tail5 winner.bound = root122Candidates4 := by
  unfold candidates
  have row : edge 1249 = (SortedNatTree.contains FastRows.row1249) := by
    change Data.row1249.contains = (SortedNatTree.contains FastRows.row1249)
    exact FastRows.row1249_checked.symm
  rw [row]
  decide
theorem root122Step4 : scan edge (search edge 3) 4 [691] 691 root122Tail4 winner =
    scan edge (search edge 3) 4 [691] 691 root122Tail5 winner := by
  rw [root122Tail4, scan, if_neg (by decide), root122Candidates4_checked]
  change scan edge (search edge 3) 4 [691] 691 root122Tail5
    (if 3 ≤ root122Candidates4.length then search edge 3 [691, 1249] root122Candidates4 1940 winner else winner) = _
  rw [if_neg (by decide)]
def root122Candidates5 : List Nat := [9043, 10243, 11083]
set_option maxRecDepth 8192 in
theorem root122Candidates5_checked : candidates edge 4 691 1291 root122Tail6 winner.bound = root122Candidates5 := by
  unfold candidates
  have row : edge 1291 = (SortedNatTree.contains FastRows.row1291) := by
    change Data.row1291.contains = (SortedNatTree.contains FastRows.row1291)
    exact FastRows.row1291_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root122Child5 : search edge 3 [691, 1291] root122Candidates5 1982 winner = winner := by decide
theorem root122Step5 : scan edge (search edge 3) 4 [691] 691 root122Tail5 winner =
    scan edge (search edge 3) 4 [691] 691 root122Tail6 winner := by
  rw [root122Tail5, scan, if_neg (by decide), root122Candidates5_checked]
  change scan edge (search edge 3) 4 [691] 691 root122Tail6
    (if 3 ≤ root122Candidates5.length then search edge 3 [691, 1291] root122Candidates5 1982 winner else winner) = _
  rw [if_pos (by decide), root122Child5]
def root122Candidates6 : List Nat := [23911]
set_option maxRecDepth 8192 in
theorem root122Candidates6_checked : candidates edge 4 691 1423 root122Tail7 winner.bound = root122Candidates6 := by
  unfold candidates
  have row : edge 1423 = (SortedNatTree.contains FastRows.row1423) := by
    change Data.row1423.contains = (SortedNatTree.contains FastRows.row1423)
    exact FastRows.row1423_checked.symm
  rw [row]
  decide
theorem root122Step6 : scan edge (search edge 3) 4 [691] 691 root122Tail6 winner =
    scan edge (search edge 3) 4 [691] 691 root122Tail7 winner := by
  rw [root122Tail6, scan, if_neg (by decide), root122Candidates6_checked]
  change scan edge (search edge 3) 4 [691] 691 root122Tail7
    (if 3 ≤ root122Candidates6.length then search edge 3 [691, 1423] root122Candidates6 2114 winner else winner) = _
  rw [if_neg (by decide)]
def root122Candidates7 : List Nat := [2593, 20563]
set_option maxRecDepth 8192 in
theorem root122Candidates7_checked : candidates edge 4 691 1693 root122Tail8 winner.bound = root122Candidates7 := by
  unfold candidates
  have row : edge 1693 = (SortedNatTree.contains FastRows.row1693) := by
    change Data.row1693.contains = (SortedNatTree.contains FastRows.row1693)
    exact FastRows.row1693_checked.symm
  rw [row]
  decide
theorem root122Step7 : scan edge (search edge 3) 4 [691] 691 root122Tail7 winner =
    scan edge (search edge 3) 4 [691] 691 root122Tail8 winner := by
  rw [root122Tail7, scan, if_neg (by decide), root122Candidates7_checked]
  change scan edge (search edge 3) 4 [691] 691 root122Tail8
    (if 3 ≤ root122Candidates7.length then search edge 3 [691, 1693] root122Candidates7 2384 winner else winner) = _
  rw [if_neg (by decide)]
def root122Candidates8 : List Nat := [3541, 20563]
set_option maxRecDepth 8192 in
theorem root122Candidates8_checked : candidates edge 4 691 2131 root122Tail9 winner.bound = root122Candidates8 := by
  unfold candidates
  have row : edge 2131 = (SortedNatTree.contains FastRows.row2131) := by
    change Data.row2131.contains = (SortedNatTree.contains FastRows.row2131)
    exact FastRows.row2131_checked.symm
  rw [row]
  decide
theorem root122Step8 : scan edge (search edge 3) 4 [691] 691 root122Tail8 winner =
    scan edge (search edge 3) 4 [691] 691 root122Tail9 winner := by
  rw [root122Tail8, scan, if_neg (by decide), root122Candidates8_checked]
  change scan edge (search edge 3) 4 [691] 691 root122Tail9
    (if 3 ≤ root122Candidates8.length then search edge 3 [691, 2131] root122Candidates8 2822 winner else winner) = _
  rw [if_neg (by decide)]
def root122Candidates9 : List Nat := [9181, 9421]
set_option maxRecDepth 8192 in
theorem root122Candidates9_checked : candidates edge 4 691 2593 root122Tail10 winner.bound = root122Candidates9 := by
  unfold candidates
  have row : edge 2593 = (SortedNatTree.contains FastRows.row2593) := by
    change Data.row2593.contains = (SortedNatTree.contains FastRows.row2593)
    exact FastRows.row2593_checked.symm
  rw [row]
  decide
theorem root122Step9 : scan edge (search edge 3) 4 [691] 691 root122Tail9 winner =
    scan edge (search edge 3) 4 [691] 691 root122Tail10 winner := by
  rw [root122Tail9, scan, if_neg (by decide), root122Candidates9_checked]
  change scan edge (search edge 3) 4 [691] 691 root122Tail10
    (if 3 ≤ root122Candidates9.length then search edge 3 [691, 2593] root122Candidates9 3284 winner else winner) = _
  rw [if_neg (by decide)]
def root122Candidates10 : List Nat := [3433, 5851, 8599, 9043, 13903]
set_option maxRecDepth 8192 in
theorem root122Candidates10_checked : candidates edge 4 691 3121 root122Tail11 winner.bound = root122Candidates10 := by
  unfold candidates
  have row : edge 3121 = (SortedNatTree.contains FastRows.row3121) := by
    change Data.row3121.contains = (SortedNatTree.contains FastRows.row3121)
    exact FastRows.row3121_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root122Child10 : search edge 3 [691, 3121] root122Candidates10 3812 winner = winner := by decide
theorem root122Step10 : scan edge (search edge 3) 4 [691] 691 root122Tail10 winner =
    scan edge (search edge 3) 4 [691] 691 root122Tail11 winner := by
  rw [root122Tail10, scan, if_neg (by decide), root122Candidates10_checked]
  change scan edge (search edge 3) 4 [691] 691 root122Tail11
    (if 3 ≤ root122Candidates10.length then search edge 3 [691, 3121] root122Candidates10 3812 winner else winner) = _
  rw [if_pos (by decide), root122Child10]
def root122Candidates11 : List Nat := [6673, 19417, 20089]
set_option maxRecDepth 8192 in
theorem root122Candidates11_checked : candidates edge 4 691 3229 root122Tail12 winner.bound = root122Candidates11 := by
  unfold candidates
  have row : edge 3229 = (SortedNatTree.contains FastRows.row3229) := by
    change Data.row3229.contains = (SortedNatTree.contains FastRows.row3229)
    exact FastRows.row3229_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root122Child11 : search edge 3 [691, 3229] root122Candidates11 3920 winner = winner := by decide
theorem root122Step11 : scan edge (search edge 3) 4 [691] 691 root122Tail11 winner =
    scan edge (search edge 3) 4 [691] 691 root122Tail12 winner := by
  rw [root122Tail11, scan, if_neg (by decide), root122Candidates11_checked]
  change scan edge (search edge 3) 4 [691] 691 root122Tail12
    (if 3 ≤ root122Candidates11.length then search edge 3 [691, 3229] root122Candidates11 3920 winner else winner) = _
  rw [if_pos (by decide), root122Child11]
def root122Candidates12 : List Nat := [6451]
set_option maxRecDepth 8192 in
theorem root122Candidates12_checked : candidates edge 4 691 3433 root122Tail13 winner.bound = root122Candidates12 := by
  unfold candidates
  have row : edge 3433 = (SortedNatTree.contains FastRows.row3433) := by
    change Data.row3433.contains = (SortedNatTree.contains FastRows.row3433)
    exact FastRows.row3433_checked.symm
  rw [row]
  decide
theorem root122Step12 : scan edge (search edge 3) 4 [691] 691 root122Tail12 winner =
    scan edge (search edge 3) 4 [691] 691 root122Tail13 winner := by
  rw [root122Tail12, scan, if_neg (by decide), root122Candidates12_checked]
  change scan edge (search edge 3) 4 [691] 691 root122Tail13
    (if 3 ≤ root122Candidates12.length then search edge 3 [691, 3433] root122Candidates12 4124 winner else winner) = _
  rw [if_neg (by decide)]
def root122Candidates13 : List Nat := [11839]
set_option maxRecDepth 8192 in
theorem root122Candidates13_checked : candidates edge 4 691 3469 root122Tail14 winner.bound = root122Candidates13 := by
  unfold candidates
  have row : edge 3469 = (SortedNatTree.contains FastRows.row3469) := by
    change Data.row3469.contains = (SortedNatTree.contains FastRows.row3469)
    exact FastRows.row3469_checked.symm
  rw [row]
  decide
theorem root122Step13 : scan edge (search edge 3) 4 [691] 691 root122Tail13 winner =
    scan edge (search edge 3) 4 [691] 691 root122Tail14 winner := by
  rw [root122Tail13, scan, if_neg (by decide), root122Candidates13_checked]
  change scan edge (search edge 3) 4 [691] 691 root122Tail14
    (if 3 ≤ root122Candidates13.length then search edge 3 [691, 3469] root122Candidates13 4160 winner else winner) = _
  rw [if_neg (by decide)]
def root122Candidates14 : List Nat := [20089, 20563]
set_option maxRecDepth 8192 in
theorem root122Candidates14_checked : candidates edge 4 691 3541 root122Tail15 winner.bound = root122Candidates14 := by
  unfold candidates
  have row : edge 3541 = (SortedNatTree.contains FastRows.row3541) := by
    change Data.row3541.contains = (SortedNatTree.contains FastRows.row3541)
    exact FastRows.row3541_checked.symm
  rw [row]
  decide
theorem root122Step14 : scan edge (search edge 3) 4 [691] 691 root122Tail14 winner =
    scan edge (search edge 3) 4 [691] 691 root122Tail15 winner := by
  rw [root122Tail14, scan, if_neg (by decide), root122Candidates14_checked]
  change scan edge (search edge 3) 4 [691] 691 root122Tail15
    (if 3 ≤ root122Candidates14.length then search edge 3 [691, 3541] root122Candidates14 4232 winner else winner) = _
  rw [if_neg (by decide)]
def root122Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root122Candidates15_checked : candidates edge 4 691 3607 root122Tail16 winner.bound = root122Candidates15 := by
  unfold candidates
  have row : edge 3607 = (SortedNatTree.contains FastRows.row3607) := by
    change Data.row3607.contains = (SortedNatTree.contains FastRows.row3607)
    exact FastRows.row3607_checked.symm
  rw [row]
  decide
theorem root122Step15 : scan edge (search edge 3) 4 [691] 691 root122Tail15 winner =
    scan edge (search edge 3) 4 [691] 691 root122Tail16 winner := by
  rw [root122Tail15, scan, if_neg (by decide), root122Candidates15_checked]
  change scan edge (search edge 3) 4 [691] 691 root122Tail16
    (if 3 ≤ root122Candidates15.length then search edge 3 [691, 3607] root122Candidates15 4298 winner else winner) = _
  rw [if_neg (by decide)]
def root122Candidates16 : List Nat := [12553]
set_option maxRecDepth 8192 in
theorem root122Candidates16_checked : candidates edge 4 691 3943 root122Tail17 winner.bound = root122Candidates16 := by
  unfold candidates
  have row : edge 3943 = (SortedNatTree.contains FastRows.row3943) := by
    change Data.row3943.contains = (SortedNatTree.contains FastRows.row3943)
    exact FastRows.row3943_checked.symm
  rw [row]
  decide
theorem root122Step16 : scan edge (search edge 3) 4 [691] 691 root122Tail16 winner =
    scan edge (search edge 3) 4 [691] 691 root122Tail17 winner := by
  rw [root122Tail16, scan, if_neg (by decide), root122Candidates16_checked]
  change scan edge (search edge 3) 4 [691] 691 root122Tail17
    (if 3 ≤ root122Candidates16.length then search edge 3 [691, 3943] root122Candidates16 4634 winner else winner) = _
  rw [if_neg (by decide)]
def root122Candidates17 : List Nat := [8599, 8761]
set_option maxRecDepth 8192 in
theorem root122Candidates17_checked : candidates edge 4 691 4603 root122Tail18 winner.bound = root122Candidates17 := by
  unfold candidates
  have row : edge 4603 = (SortedNatTree.contains FastRows.row4603) := by
    change Data.row4603.contains = (SortedNatTree.contains FastRows.row4603)
    exact FastRows.row4603_checked.symm
  rw [row]
  decide
theorem root122Step17 : scan edge (search edge 3) 4 [691] 691 root122Tail17 winner =
    scan edge (search edge 3) 4 [691] 691 root122Tail18 winner := by
  rw [root122Tail17, scan, if_neg (by decide), root122Candidates17_checked]
  change scan edge (search edge 3) 4 [691] 691 root122Tail18
    (if 3 ≤ root122Candidates17.length then search edge 3 [691, 4603] root122Candidates17 5294 winner else winner) = _
  rw [if_neg (by decide)]
def root122Candidates18 : List Nat := [7177, 13903, 19867]
set_option maxRecDepth 8192 in
theorem root122Candidates18_checked : candidates edge 4 691 4657 root122Tail19 winner.bound = root122Candidates18 := by
  unfold candidates
  have row : edge 4657 = (SortedNatTree.contains FastRows.row4657) := by
    change Data.row4657.contains = (SortedNatTree.contains FastRows.row4657)
    exact FastRows.row4657_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root122Child18 : search edge 3 [691, 4657] root122Candidates18 5348 winner = winner := by decide
theorem root122Step18 : scan edge (search edge 3) 4 [691] 691 root122Tail18 winner =
    scan edge (search edge 3) 4 [691] 691 root122Tail19 winner := by
  rw [root122Tail18, scan, if_neg (by decide), root122Candidates18_checked]
  change scan edge (search edge 3) 4 [691] 691 root122Tail19
    (if 3 ≤ root122Candidates18.length then search edge 3 [691, 4657] root122Candidates18 5348 winner else winner) = _
  rw [if_pos (by decide), root122Child18]
def root122Candidates19 : List Nat := [9181]
set_option maxRecDepth 8192 in
theorem root122Candidates19_checked : candidates edge 4 691 4729 root122Tail20 winner.bound = root122Candidates19 := by
  unfold candidates
  have row : edge 4729 = (SortedNatTree.contains FastRows.row4729) := by
    change Data.row4729.contains = (SortedNatTree.contains FastRows.row4729)
    exact FastRows.row4729_checked.symm
  rw [row]
  decide
theorem root122Step19 : scan edge (search edge 3) 4 [691] 691 root122Tail19 winner =
    scan edge (search edge 3) 4 [691] 691 root122Tail20 winner := by
  rw [root122Tail19, scan, if_neg (by decide), root122Candidates19_checked]
  change scan edge (search edge 3) 4 [691] 691 root122Tail20
    (if 3 ≤ root122Candidates19.length then search edge 3 [691, 4729] root122Candidates19 5420 winner else winner) = _
  rw [if_neg (by decide)]
def root122Candidates20 : List Nat := []
set_option maxRecDepth 8192 in
theorem root122Candidates20_checked : candidates edge 4 691 4861 root122Tail21 winner.bound = root122Candidates20 := by
  unfold candidates
  have row : edge 4861 = (SortedNatTree.contains FastRows.row4861) := by
    change Data.row4861.contains = (SortedNatTree.contains FastRows.row4861)
    exact FastRows.row4861_checked.symm
  rw [row]
  decide
theorem root122Step20 : scan edge (search edge 3) 4 [691] 691 root122Tail20 winner =
    scan edge (search edge 3) 4 [691] 691 root122Tail21 winner := by
  rw [root122Tail20, scan, if_neg (by decide), root122Candidates20_checked]
  change scan edge (search edge 3) 4 [691] 691 root122Tail21
    (if 3 ≤ root122Candidates20.length then search edge 3 [691, 4861] root122Candidates20 5552 winner else winner) = _
  rw [if_neg (by decide)]
def root122Candidates21 : List Nat := [18451]
set_option maxRecDepth 8192 in
theorem root122Candidates21_checked : candidates edge 4 691 4951 root122Tail22 winner.bound = root122Candidates21 := by
  unfold candidates
  have row : edge 4951 = (SortedNatTree.contains FastRows.row4951) := by
    change Data.row4951.contains = (SortedNatTree.contains FastRows.row4951)
    exact FastRows.row4951_checked.symm
  rw [row]
  decide
theorem root122Step21 : scan edge (search edge 3) 4 [691] 691 root122Tail21 winner =
    scan edge (search edge 3) 4 [691] 691 root122Tail22 winner := by
  rw [root122Tail21, scan, if_neg (by decide), root122Candidates21_checked]
  change scan edge (search edge 3) 4 [691] 691 root122Tail22
    (if 3 ≤ root122Candidates21.length then search edge 3 [691, 4951] root122Candidates21 5642 winner else winner) = _
  rw [if_neg (by decide)]
def root122Candidates22 : List Nat := [8389, 17791]
set_option maxRecDepth 8192 in
theorem root122Candidates22_checked : candidates edge 4 691 5197 root122Tail23 winner.bound = root122Candidates22 := by
  unfold candidates
  have row : edge 5197 = (SortedNatTree.contains FastRows.row5197) := by
    change Data.row5197.contains = (SortedNatTree.contains FastRows.row5197)
    exact FastRows.row5197_checked.symm
  rw [row]
  decide
theorem root122Step22 : scan edge (search edge 3) 4 [691] 691 root122Tail22 winner =
    scan edge (search edge 3) 4 [691] 691 root122Tail23 winner := by
  rw [root122Tail22, scan, if_neg (by decide), root122Candidates22_checked]
  change scan edge (search edge 3) 4 [691] 691 root122Tail23
    (if 3 ≤ root122Candidates22.length then search edge 3 [691, 5197] root122Candidates22 5888 winner else winner) = _
  rw [if_neg (by decide)]
def root122Candidates23 : List Nat := [5743, 8599]
set_option maxRecDepth 8192 in
theorem root122Candidates23_checked : candidates edge 4 691 5431 root122Tail24 winner.bound = root122Candidates23 := by
  unfold candidates
  have row : edge 5431 = (SortedNatTree.contains FastRows.row5431) := by
    change Data.row5431.contains = (SortedNatTree.contains FastRows.row5431)
    exact FastRows.row5431_checked.symm
  rw [row]
  decide
theorem root122Step23 : scan edge (search edge 3) 4 [691] 691 root122Tail23 winner =
    scan edge (search edge 3) 4 [691] 691 root122Tail24 winner := by
  rw [root122Tail23, scan, if_neg (by decide), root122Candidates23_checked]
  change scan edge (search edge 3) 4 [691] 691 root122Tail24
    (if 3 ≤ root122Candidates23.length then search edge 3 [691, 5431] root122Candidates23 6122 winner else winner) = _
  rw [if_neg (by decide)]
def root122Candidates24 : List Nat := [9043]
set_option maxRecDepth 8192 in
theorem root122Candidates24_checked : candidates edge 4 691 5743 root122Tail25 winner.bound = root122Candidates24 := by
  unfold candidates
  have row : edge 5743 = (SortedNatTree.contains FastRows.row5743) := by
    change Data.row5743.contains = (SortedNatTree.contains FastRows.row5743)
    exact FastRows.row5743_checked.symm
  rw [row]
  decide
theorem root122Step24 : scan edge (search edge 3) 4 [691] 691 root122Tail24 winner =
    scan edge (search edge 3) 4 [691] 691 root122Tail25 winner := by
  rw [root122Tail24, scan, if_neg (by decide), root122Candidates24_checked]
  change scan edge (search edge 3) 4 [691] 691 root122Tail25
    (if 3 ≤ root122Candidates24.length then search edge 3 [691, 5743] root122Candidates24 6434 winner else winner) = _
  rw [if_neg (by decide)]
def root122Candidates25 : List Nat := []
set_option maxRecDepth 8192 in
theorem root122Candidates25_checked : candidates edge 4 691 5827 root122Tail26 winner.bound = root122Candidates25 := by
  unfold candidates
  have row : edge 5827 = (SortedNatTree.contains FastRows.row5827) := by
    change Data.row5827.contains = (SortedNatTree.contains FastRows.row5827)
    exact FastRows.row5827_checked.symm
  rw [row]
  decide
theorem root122Step25 : scan edge (search edge 3) 4 [691] 691 root122Tail25 winner =
    scan edge (search edge 3) 4 [691] 691 root122Tail26 winner := by
  rw [root122Tail25, scan, if_neg (by decide), root122Candidates25_checked]
  change scan edge (search edge 3) 4 [691] 691 root122Tail26
    (if 3 ≤ root122Candidates25.length then search edge 3 [691, 5827] root122Candidates25 6518 winner else winner) = _
  rw [if_neg (by decide)]
def root122Candidates26 : List Nat := []
set_option maxRecDepth 8192 in
theorem root122Candidates26_checked : candidates edge 4 691 5851 root122Tail27 winner.bound = root122Candidates26 := by
  unfold candidates
  have row : edge 5851 = (SortedNatTree.contains FastRows.row5851) := by
    change Data.row5851.contains = (SortedNatTree.contains FastRows.row5851)
    exact FastRows.row5851_checked.symm
  rw [row]
  decide
theorem root122Step26 : scan edge (search edge 3) 4 [691] 691 root122Tail26 winner =
    scan edge (search edge 3) 4 [691] 691 root122Tail27 winner := by
  rw [root122Tail26, scan, if_neg (by decide), root122Candidates26_checked]
  change scan edge (search edge 3) 4 [691] 691 root122Tail27
    (if 3 ≤ root122Candidates26.length then search edge 3 [691, 5851] root122Candidates26 6542 winner else winner) = _
  rw [if_neg (by decide)]
def root122Candidates27 : List Nat := [9967, 11839]
set_option maxRecDepth 8192 in
theorem root122Candidates27_checked : candidates edge 4 691 5869 root122Tail28 winner.bound = root122Candidates27 := by
  unfold candidates
  have row : edge 5869 = (SortedNatTree.contains FastRows.row5869) := by
    change Data.row5869.contains = (SortedNatTree.contains FastRows.row5869)
    exact FastRows.row5869_checked.symm
  rw [row]
  decide
theorem root122Step27 : scan edge (search edge 3) 4 [691] 691 root122Tail27 winner =
    scan edge (search edge 3) 4 [691] 691 root122Tail28 winner := by
  rw [root122Tail27, scan, if_neg (by decide), root122Candidates27_checked]
  change scan edge (search edge 3) 4 [691] 691 root122Tail28
    (if 3 ≤ root122Candidates27.length then search edge 3 [691, 5869] root122Candidates27 6560 winner else winner) = _
  rw [if_neg (by decide)]
def root122Candidates28 : List Nat := [6673]
set_option maxRecDepth 8192 in
theorem root122Candidates28_checked : candidates edge 4 691 6199 root122Tail29 winner.bound = root122Candidates28 := by
  unfold candidates
  have row : edge 6199 = (SortedNatTree.contains FastRows.row6199) := by
    change Data.row6199.contains = (SortedNatTree.contains FastRows.row6199)
    exact FastRows.row6199_checked.symm
  rw [row]
  decide
theorem root122Step28 : scan edge (search edge 3) 4 [691] 691 root122Tail28 winner =
    scan edge (search edge 3) 4 [691] 691 root122Tail29 winner := by
  rw [root122Tail28, scan, if_neg (by decide), root122Candidates28_checked]
  change scan edge (search edge 3) 4 [691] 691 root122Tail29
    (if 3 ≤ root122Candidates28.length then search edge 3 [691, 6199] root122Candidates28 6890 winner else winner) = _
  rw [if_neg (by decide)]
theorem root122Scan29 : scan edge (search edge 3) 4 [691] 691 root122Tail29 winner = winner := by rw [root122Tail29, scan, if_pos (by decide)]
theorem root122Scan28 : scan edge (search edge 3) 4 [691] 691 root122Tail28 winner = winner :=
  (root122Step28).trans root122Scan29
theorem root122Scan27 : scan edge (search edge 3) 4 [691] 691 root122Tail27 winner = winner :=
  (root122Step27).trans root122Scan28
theorem root122Scan26 : scan edge (search edge 3) 4 [691] 691 root122Tail26 winner = winner :=
  (root122Step26).trans root122Scan27
theorem root122Scan25 : scan edge (search edge 3) 4 [691] 691 root122Tail25 winner = winner :=
  (root122Step25).trans root122Scan26
theorem root122Scan24 : scan edge (search edge 3) 4 [691] 691 root122Tail24 winner = winner :=
  (root122Step24).trans root122Scan25
theorem root122Scan23 : scan edge (search edge 3) 4 [691] 691 root122Tail23 winner = winner :=
  (root122Step23).trans root122Scan24
theorem root122Scan22 : scan edge (search edge 3) 4 [691] 691 root122Tail22 winner = winner :=
  (root122Step22).trans root122Scan23
theorem root122Scan21 : scan edge (search edge 3) 4 [691] 691 root122Tail21 winner = winner :=
  (root122Step21).trans root122Scan22
theorem root122Scan20 : scan edge (search edge 3) 4 [691] 691 root122Tail20 winner = winner :=
  (root122Step20).trans root122Scan21
theorem root122Scan19 : scan edge (search edge 3) 4 [691] 691 root122Tail19 winner = winner :=
  (root122Step19).trans root122Scan20
theorem root122Scan18 : scan edge (search edge 3) 4 [691] 691 root122Tail18 winner = winner :=
  (root122Step18).trans root122Scan19
theorem root122Scan17 : scan edge (search edge 3) 4 [691] 691 root122Tail17 winner = winner :=
  (root122Step17).trans root122Scan18
theorem root122Scan16 : scan edge (search edge 3) 4 [691] 691 root122Tail16 winner = winner :=
  (root122Step16).trans root122Scan17
theorem root122Scan15 : scan edge (search edge 3) 4 [691] 691 root122Tail15 winner = winner :=
  (root122Step15).trans root122Scan16
theorem root122Scan14 : scan edge (search edge 3) 4 [691] 691 root122Tail14 winner = winner :=
  (root122Step14).trans root122Scan15
theorem root122Scan13 : scan edge (search edge 3) 4 [691] 691 root122Tail13 winner = winner :=
  (root122Step13).trans root122Scan14
theorem root122Scan12 : scan edge (search edge 3) 4 [691] 691 root122Tail12 winner = winner :=
  (root122Step12).trans root122Scan13
theorem root122Scan11 : scan edge (search edge 3) 4 [691] 691 root122Tail11 winner = winner :=
  (root122Step11).trans root122Scan12
theorem root122Scan10 : scan edge (search edge 3) 4 [691] 691 root122Tail10 winner = winner :=
  (root122Step10).trans root122Scan11
theorem root122Scan9 : scan edge (search edge 3) 4 [691] 691 root122Tail9 winner = winner :=
  (root122Step9).trans root122Scan10
theorem root122Scan8 : scan edge (search edge 3) 4 [691] 691 root122Tail8 winner = winner :=
  (root122Step8).trans root122Scan9
theorem root122Scan7 : scan edge (search edge 3) 4 [691] 691 root122Tail7 winner = winner :=
  (root122Step7).trans root122Scan8
theorem root122Scan6 : scan edge (search edge 3) 4 [691] 691 root122Tail6 winner = winner :=
  (root122Step6).trans root122Scan7
theorem root122Scan5 : scan edge (search edge 3) 4 [691] 691 root122Tail5 winner = winner :=
  (root122Step5).trans root122Scan6
theorem root122Scan4 : scan edge (search edge 3) 4 [691] 691 root122Tail4 winner = winner :=
  (root122Step4).trans root122Scan5
theorem root122Scan3 : scan edge (search edge 3) 4 [691] 691 root122Tail3 winner = winner :=
  (root122Step3).trans root122Scan4
theorem root122Scan2 : scan edge (search edge 3) 4 [691] 691 root122Tail2 winner = winner :=
  (root122Step2).trans root122Scan3
theorem root122Scan1 : scan edge (search edge 3) 4 [691] 691 root122Tail1 winner = winner :=
  (root122Step1).trans root122Scan2
theorem root122Scan0 : scan edge (search edge 3) 4 [691] 691 root122Tail0 winner = winner :=
  (root122Step0).trans root122Scan1
set_option maxRecDepth 4096 in
theorem root122Branch : search edge 4 [691] root122Candidates 691 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [691] 691 root122Tail0 winner = winner
  exact root122Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
