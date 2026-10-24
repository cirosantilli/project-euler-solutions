import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root226Candidates : List Nat := Data.row1447
set_option maxRecDepth 8192 in
theorem root226Candidates_checked : candidates edge 5 0 1447 tail227 winner.bound = root226Candidates := by
  apply candidates_eq_row edge 5 0 1447 tail227 root226Candidates winner.bound tail227_ordered
  · decide
  · decide
  · funext q
    rfl
def root226Tail57 : List Nat := []
def root226Tail56 : List Nat := 24169 :: root226Tail57
def root226Tail55 : List Nat := 23977 :: root226Tail56
def root226Tail54 : List Nat := 23581 :: root226Tail55
def root226Tail53 : List Nat := 22861 :: root226Tail54
def root226Tail52 : List Nat := 22531 :: root226Tail53
def root226Tail51 : List Nat := 22189 :: root226Tail52
def root226Tail50 : List Nat := 20599 :: root226Tail51
def root226Tail49 : List Nat := 20161 :: root226Tail50
def root226Tail48 : List Nat := 19801 :: root226Tail49
def root226Tail47 : List Nat := 19417 :: root226Tail48
def root226Tail46 : List Nat := 19207 :: root226Tail47
def root226Tail45 : List Nat := 18523 :: root226Tail46
def root226Tail44 : List Nat := 18517 :: root226Tail45
def root226Tail43 : List Nat := 18313 :: root226Tail44
def root226Tail42 : List Nat := 18121 :: root226Tail43
def root226Tail41 : List Nat := 17971 :: root226Tail42
def root226Tail40 : List Nat := 17359 :: root226Tail41
def root226Tail39 : List Nat := 16963 :: root226Tail40
def root226Tail38 : List Nat := 16927 :: root226Tail39
def root226Tail37 : List Nat := 16633 :: root226Tail38
def root226Tail36 : List Nat := 15667 :: root226Tail37
def root226Tail35 : List Nat := 15619 :: root226Tail36
def root226Tail34 : List Nat := 15331 :: root226Tail35
def root226Tail33 : List Nat := 13399 :: root226Tail34
def root226Tail32 : List Nat := 13147 :: root226Tail33
def root226Tail31 : List Nat := 11887 :: root226Tail32
def root226Tail30 : List Nat := 11527 :: root226Tail31
def root226Tail29 : List Nat := 10639 :: root226Tail30
def root226Tail28 : List Nat := 9277 :: root226Tail29
def root226Tail27 : List Nat := 9043 :: root226Tail28
def root226Tail26 : List Nat := 8647 :: root226Tail27
def root226Tail25 : List Nat := 8539 :: root226Tail26
def root226Tail24 : List Nat := 8059 :: root226Tail25
def root226Tail23 : List Nat := 7759 :: root226Tail24
def root226Tail22 : List Nat := 6991 :: root226Tail23
def root226Tail21 : List Nat := 6967 :: root226Tail22
def root226Tail20 : List Nat := 6949 :: root226Tail21
def root226Tail19 : List Nat := 5689 :: root226Tail20
def root226Tail18 : List Nat := 5683 :: root226Tail19
def root226Tail17 : List Nat := 5641 :: root226Tail18
def root226Tail16 : List Nat := 4441 :: root226Tail17
def root226Tail15 : List Nat := 4093 :: root226Tail16
def root226Tail14 : List Nat := 4051 :: root226Tail15
def root226Tail13 : List Nat := 3769 :: root226Tail14
def root226Tail12 : List Nat := 3709 :: root226Tail13
def root226Tail11 : List Nat := 3313 :: root226Tail12
def root226Tail10 : List Nat := 3271 :: root226Tail11
def root226Tail9 : List Nat := 3229 :: root226Tail10
def root226Tail8 : List Nat := 2833 :: root226Tail9
def root226Tail7 : List Nat := 2803 :: root226Tail8
def root226Tail6 : List Nat := 2719 :: root226Tail7
def root226Tail5 : List Nat := 2347 :: root226Tail6
def root226Tail4 : List Nat := 2011 :: root226Tail5
def root226Tail3 : List Nat := 1993 :: root226Tail4
def root226Tail2 : List Nat := 1753 :: root226Tail3
def root226Tail1 : List Nat := 1549 :: root226Tail2
def root226Tail0 : List Nat := 1543 :: root226Tail1
def root226Candidates0 : List Nat := [4093, 5689, 8539, 18517]
set_option maxRecDepth 8192 in
theorem root226Candidates0_checked : candidates edge 4 1447 1543 root226Tail1 winner.bound = root226Candidates0 := by
  unfold candidates
  have row : edge 1543 = (SortedNatTree.contains FastRows.row1543) := by
    change Data.row1543.contains = (SortedNatTree.contains FastRows.row1543)
    exact FastRows.row1543_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root226Child0 : search edge 3 [1447, 1543] root226Candidates0 2990 winner = winner := by decide
theorem root226Step0 : scan edge (search edge 3) 4 [1447] 1447 root226Tail0 winner =
    scan edge (search edge 3) 4 [1447] 1447 root226Tail1 winner := by
  rw [root226Tail0, scan, if_neg (by decide), root226Candidates0_checked]
  change scan edge (search edge 3) 4 [1447] 1447 root226Tail1
    (if 3 ≤ root226Candidates0.length then search edge 3 [1447, 1543] root226Candidates0 2990 winner else winner) = _
  rw [if_pos (by decide), root226Child0]
def root226Candidates1 : List Nat := [2833, 19207]
set_option maxRecDepth 8192 in
theorem root226Candidates1_checked : candidates edge 4 1447 1549 root226Tail2 winner.bound = root226Candidates1 := by
  unfold candidates
  have row : edge 1549 = (SortedNatTree.contains FastRows.row1549) := by
    change Data.row1549.contains = (SortedNatTree.contains FastRows.row1549)
    exact FastRows.row1549_checked.symm
  rw [row]
  decide
theorem root226Step1 : scan edge (search edge 3) 4 [1447] 1447 root226Tail1 winner =
    scan edge (search edge 3) 4 [1447] 1447 root226Tail2 winner := by
  rw [root226Tail1, scan, if_neg (by decide), root226Candidates1_checked]
  change scan edge (search edge 3) 4 [1447] 1447 root226Tail2
    (if 3 ≤ root226Candidates1.length then search edge 3 [1447, 1549] root226Candidates1 2996 winner else winner) = _
  rw [if_neg (by decide)]
def root226Candidates2 : List Nat := [3709, 11527]
set_option maxRecDepth 8192 in
theorem root226Candidates2_checked : candidates edge 4 1447 1753 root226Tail3 winner.bound = root226Candidates2 := by
  unfold candidates
  have row : edge 1753 = (SortedNatTree.contains FastRows.row1753) := by
    change Data.row1753.contains = (SortedNatTree.contains FastRows.row1753)
    exact FastRows.row1753_checked.symm
  rw [row]
  decide
theorem root226Step2 : scan edge (search edge 3) 4 [1447] 1447 root226Tail2 winner =
    scan edge (search edge 3) 4 [1447] 1447 root226Tail3 winner := by
  rw [root226Tail2, scan, if_neg (by decide), root226Candidates2_checked]
  change scan edge (search edge 3) 4 [1447] 1447 root226Tail3
    (if 3 ≤ root226Candidates2.length then search edge 3 [1447, 1753] root226Candidates2 3200 winner else winner) = _
  rw [if_neg (by decide)]
def root226Candidates3 : List Nat := [3271, 4093, 9277]
set_option maxRecDepth 8192 in
theorem root226Candidates3_checked : candidates edge 4 1447 1993 root226Tail4 winner.bound = root226Candidates3 := by
  unfold candidates
  have row : edge 1993 = (SortedNatTree.contains FastRows.row1993) := by
    change Data.row1993.contains = (SortedNatTree.contains FastRows.row1993)
    exact FastRows.row1993_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root226Child3 : search edge 3 [1447, 1993] root226Candidates3 3440 winner = winner := by decide
theorem root226Step3 : scan edge (search edge 3) 4 [1447] 1447 root226Tail3 winner =
    scan edge (search edge 3) 4 [1447] 1447 root226Tail4 winner := by
  rw [root226Tail3, scan, if_neg (by decide), root226Candidates3_checked]
  change scan edge (search edge 3) 4 [1447] 1447 root226Tail4
    (if 3 ≤ root226Candidates3.length then search edge 3 [1447, 1993] root226Candidates3 3440 winner else winner) = _
  rw [if_pos (by decide), root226Child3]
def root226Candidates4 : List Nat := [4093, 15331]
set_option maxRecDepth 8192 in
theorem root226Candidates4_checked : candidates edge 4 1447 2011 root226Tail5 winner.bound = root226Candidates4 := by
  unfold candidates
  have row : edge 2011 = (SortedNatTree.contains FastRows.row2011) := by
    change Data.row2011.contains = (SortedNatTree.contains FastRows.row2011)
    exact FastRows.row2011_checked.symm
  rw [row]
  decide
theorem root226Step4 : scan edge (search edge 3) 4 [1447] 1447 root226Tail4 winner =
    scan edge (search edge 3) 4 [1447] 1447 root226Tail5 winner := by
  rw [root226Tail4, scan, if_neg (by decide), root226Candidates4_checked]
  change scan edge (search edge 3) 4 [1447] 1447 root226Tail5
    (if 3 ≤ root226Candidates4.length then search edge 3 [1447, 2011] root226Candidates4 3458 winner else winner) = _
  rw [if_neg (by decide)]
def root226Candidates5 : List Nat := [2833]
set_option maxRecDepth 8192 in
theorem root226Candidates5_checked : candidates edge 4 1447 2347 root226Tail6 winner.bound = root226Candidates5 := by
  unfold candidates
  have row : edge 2347 = (SortedNatTree.contains FastRows.row2347) := by
    change Data.row2347.contains = (SortedNatTree.contains FastRows.row2347)
    exact FastRows.row2347_checked.symm
  rw [row]
  decide
theorem root226Step5 : scan edge (search edge 3) 4 [1447] 1447 root226Tail5 winner =
    scan edge (search edge 3) 4 [1447] 1447 root226Tail6 winner := by
  rw [root226Tail5, scan, if_neg (by decide), root226Candidates5_checked]
  change scan edge (search edge 3) 4 [1447] 1447 root226Tail6
    (if 3 ≤ root226Candidates5.length then search edge 3 [1447, 2347] root226Candidates5 3794 winner else winner) = _
  rw [if_neg (by decide)]
def root226Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root226Candidates6_checked : candidates edge 4 1447 2719 root226Tail7 winner.bound = root226Candidates6 := by
  unfold candidates
  have row : edge 2719 = (SortedNatTree.contains FastRows.row2719) := by
    change Data.row2719.contains = (SortedNatTree.contains FastRows.row2719)
    exact FastRows.row2719_checked.symm
  rw [row]
  decide
theorem root226Step6 : scan edge (search edge 3) 4 [1447] 1447 root226Tail6 winner =
    scan edge (search edge 3) 4 [1447] 1447 root226Tail7 winner := by
  rw [root226Tail6, scan, if_neg (by decide), root226Candidates6_checked]
  change scan edge (search edge 3) 4 [1447] 1447 root226Tail7
    (if 3 ≤ root226Candidates6.length then search edge 3 [1447, 2719] root226Candidates6 4166 winner else winner) = _
  rw [if_neg (by decide)]
def root226Candidates7 : List Nat := [3769, 4051]
set_option maxRecDepth 8192 in
theorem root226Candidates7_checked : candidates edge 4 1447 2803 root226Tail8 winner.bound = root226Candidates7 := by
  unfold candidates
  have row : edge 2803 = (SortedNatTree.contains FastRows.row2803) := by
    change Data.row2803.contains = (SortedNatTree.contains FastRows.row2803)
    exact FastRows.row2803_checked.symm
  rw [row]
  decide
theorem root226Step7 : scan edge (search edge 3) 4 [1447] 1447 root226Tail7 winner =
    scan edge (search edge 3) 4 [1447] 1447 root226Tail8 winner := by
  rw [root226Tail7, scan, if_neg (by decide), root226Candidates7_checked]
  change scan edge (search edge 3) 4 [1447] 1447 root226Tail8
    (if 3 ≤ root226Candidates7.length then search edge 3 [1447, 2803] root226Candidates7 4250 winner else winner) = _
  rw [if_neg (by decide)]
def root226Candidates8 : List Nat := [3229, 5683, 20161]
set_option maxRecDepth 8192 in
theorem root226Candidates8_checked : candidates edge 4 1447 2833 root226Tail9 winner.bound = root226Candidates8 := by
  unfold candidates
  have row : edge 2833 = (SortedNatTree.contains FastRows.row2833) := by
    change Data.row2833.contains = (SortedNatTree.contains FastRows.row2833)
    exact FastRows.row2833_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root226Child8 : search edge 3 [1447, 2833] root226Candidates8 4280 winner = winner := by decide
theorem root226Step8 : scan edge (search edge 3) 4 [1447] 1447 root226Tail8 winner =
    scan edge (search edge 3) 4 [1447] 1447 root226Tail9 winner := by
  rw [root226Tail8, scan, if_neg (by decide), root226Candidates8_checked]
  change scan edge (search edge 3) 4 [1447] 1447 root226Tail9
    (if 3 ≤ root226Candidates8.length then search edge 3 [1447, 2833] root226Candidates8 4280 winner else winner) = _
  rw [if_pos (by decide), root226Child8]
def root226Candidates9 : List Nat := [9277, 16927, 18517, 19417]
set_option maxRecDepth 8192 in
theorem root226Candidates9_checked : candidates edge 4 1447 3229 root226Tail10 winner.bound = root226Candidates9 := by
  unfold candidates
  have row : edge 3229 = (SortedNatTree.contains FastRows.row3229) := by
    change Data.row3229.contains = (SortedNatTree.contains FastRows.row3229)
    exact FastRows.row3229_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root226Child9 : search edge 3 [1447, 3229] root226Candidates9 4676 winner = winner := by decide
theorem root226Step9 : scan edge (search edge 3) 4 [1447] 1447 root226Tail9 winner =
    scan edge (search edge 3) 4 [1447] 1447 root226Tail10 winner := by
  rw [root226Tail9, scan, if_neg (by decide), root226Candidates9_checked]
  change scan edge (search edge 3) 4 [1447] 1447 root226Tail10
    (if 3 ≤ root226Candidates9.length then search edge 3 [1447, 3229] root226Candidates9 4676 winner else winner) = _
  rw [if_pos (by decide), root226Child9]
def root226Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root226Candidates10_checked : candidates edge 4 1447 3271 root226Tail11 winner.bound = root226Candidates10 := by
  unfold candidates
  have row : edge 3271 = (SortedNatTree.contains FastRows.row3271) := by
    change Data.row3271.contains = (SortedNatTree.contains FastRows.row3271)
    exact FastRows.row3271_checked.symm
  rw [row]
  decide
theorem root226Step10 : scan edge (search edge 3) 4 [1447] 1447 root226Tail10 winner =
    scan edge (search edge 3) 4 [1447] 1447 root226Tail11 winner := by
  rw [root226Tail10, scan, if_neg (by decide), root226Candidates10_checked]
  change scan edge (search edge 3) 4 [1447] 1447 root226Tail11
    (if 3 ≤ root226Candidates10.length then search edge 3 [1447, 3271] root226Candidates10 4718 winner else winner) = _
  rw [if_neg (by decide)]
def root226Candidates11 : List Nat := [8059]
set_option maxRecDepth 8192 in
theorem root226Candidates11_checked : candidates edge 4 1447 3313 root226Tail12 winner.bound = root226Candidates11 := by
  unfold candidates
  have row : edge 3313 = (SortedNatTree.contains FastRows.row3313) := by
    change Data.row3313.contains = (SortedNatTree.contains FastRows.row3313)
    exact FastRows.row3313_checked.symm
  rw [row]
  decide
theorem root226Step11 : scan edge (search edge 3) 4 [1447] 1447 root226Tail11 winner =
    scan edge (search edge 3) 4 [1447] 1447 root226Tail12 winner := by
  rw [root226Tail11, scan, if_neg (by decide), root226Candidates11_checked]
  change scan edge (search edge 3) 4 [1447] 1447 root226Tail12
    (if 3 ≤ root226Candidates11.length then search edge 3 [1447, 3313] root226Candidates11 4760 winner else winner) = _
  rw [if_neg (by decide)]
def root226Candidates12 : List Nat := [6991]
set_option maxRecDepth 8192 in
theorem root226Candidates12_checked : candidates edge 4 1447 3709 root226Tail13 winner.bound = root226Candidates12 := by
  unfold candidates
  have row : edge 3709 = (SortedNatTree.contains FastRows.row3709) := by
    change Data.row3709.contains = (SortedNatTree.contains FastRows.row3709)
    exact FastRows.row3709_checked.symm
  rw [row]
  decide
theorem root226Step12 : scan edge (search edge 3) 4 [1447] 1447 root226Tail12 winner =
    scan edge (search edge 3) 4 [1447] 1447 root226Tail13 winner := by
  rw [root226Tail12, scan, if_neg (by decide), root226Candidates12_checked]
  change scan edge (search edge 3) 4 [1447] 1447 root226Tail13
    (if 3 ≤ root226Candidates12.length then search edge 3 [1447, 3709] root226Candidates12 5156 winner else winner) = _
  rw [if_neg (by decide)]
def root226Candidates13 : List Nat := [4051, 6949]
set_option maxRecDepth 8192 in
theorem root226Candidates13_checked : candidates edge 4 1447 3769 root226Tail14 winner.bound = root226Candidates13 := by
  unfold candidates
  have row : edge 3769 = (SortedNatTree.contains FastRows.row3769) := by
    change Data.row3769.contains = (SortedNatTree.contains FastRows.row3769)
    exact FastRows.row3769_checked.symm
  rw [row]
  decide
theorem root226Step13 : scan edge (search edge 3) 4 [1447] 1447 root226Tail13 winner =
    scan edge (search edge 3) 4 [1447] 1447 root226Tail14 winner := by
  rw [root226Tail13, scan, if_neg (by decide), root226Candidates13_checked]
  change scan edge (search edge 3) 4 [1447] 1447 root226Tail14
    (if 3 ≤ root226Candidates13.length then search edge 3 [1447, 3769] root226Candidates13 5216 winner else winner) = _
  rw [if_neg (by decide)]
def root226Candidates14 : List Nat := [20161]
set_option maxRecDepth 8192 in
theorem root226Candidates14_checked : candidates edge 4 1447 4051 root226Tail15 winner.bound = root226Candidates14 := by
  unfold candidates
  have row : edge 4051 = (SortedNatTree.contains FastRows.row4051) := by
    change Data.row4051.contains = (SortedNatTree.contains FastRows.row4051)
    exact FastRows.row4051_checked.symm
  rw [row]
  decide
theorem root226Step14 : scan edge (search edge 3) 4 [1447] 1447 root226Tail14 winner =
    scan edge (search edge 3) 4 [1447] 1447 root226Tail15 winner := by
  rw [root226Tail14, scan, if_neg (by decide), root226Candidates14_checked]
  change scan edge (search edge 3) 4 [1447] 1447 root226Tail15
    (if 3 ≤ root226Candidates14.length then search edge 3 [1447, 4051] root226Candidates14 5498 winner else winner) = _
  rw [if_neg (by decide)]
def root226Candidates15 : List Nat := [6991, 8647]
set_option maxRecDepth 8192 in
theorem root226Candidates15_checked : candidates edge 4 1447 4093 root226Tail16 winner.bound = root226Candidates15 := by
  unfold candidates
  have row : edge 4093 = (SortedNatTree.contains FastRows.row4093) := by
    change Data.row4093.contains = (SortedNatTree.contains FastRows.row4093)
    exact FastRows.row4093_checked.symm
  rw [row]
  decide
theorem root226Step15 : scan edge (search edge 3) 4 [1447] 1447 root226Tail15 winner =
    scan edge (search edge 3) 4 [1447] 1447 root226Tail16 winner := by
  rw [root226Tail15, scan, if_neg (by decide), root226Candidates15_checked]
  change scan edge (search edge 3) 4 [1447] 1447 root226Tail16
    (if 3 ≤ root226Candidates15.length then search edge 3 [1447, 4093] root226Candidates15 5540 winner else winner) = _
  rw [if_neg (by decide)]
def root226Candidates16 : List Nat := [6949, 7759]
set_option maxRecDepth 8192 in
theorem root226Candidates16_checked : candidates edge 4 1447 4441 root226Tail17 winner.bound = root226Candidates16 := by
  unfold candidates
  have row : edge 4441 = (SortedNatTree.contains FastRows.row4441) := by
    change Data.row4441.contains = (SortedNatTree.contains FastRows.row4441)
    exact FastRows.row4441_checked.symm
  rw [row]
  decide
theorem root226Step16 : scan edge (search edge 3) 4 [1447] 1447 root226Tail16 winner =
    scan edge (search edge 3) 4 [1447] 1447 root226Tail17 winner := by
  rw [root226Tail16, scan, if_neg (by decide), root226Candidates16_checked]
  change scan edge (search edge 3) 4 [1447] 1447 root226Tail17
    (if 3 ≤ root226Candidates16.length then search edge 3 [1447, 4441] root226Candidates16 5888 winner else winner) = _
  rw [if_neg (by decide)]
def root226Candidates17 : List Nat := [7759]
set_option maxRecDepth 8192 in
theorem root226Candidates17_checked : candidates edge 4 1447 5641 root226Tail18 winner.bound = root226Candidates17 := by
  unfold candidates
  have row : edge 5641 = (SortedNatTree.contains FastRows.row5641) := by
    change Data.row5641.contains = (SortedNatTree.contains FastRows.row5641)
    exact FastRows.row5641_checked.symm
  rw [row]
  decide
theorem root226Step17 : scan edge (search edge 3) 4 [1447] 1447 root226Tail17 winner =
    scan edge (search edge 3) 4 [1447] 1447 root226Tail18 winner := by
  rw [root226Tail17, scan, if_neg (by decide), root226Candidates17_checked]
  change scan edge (search edge 3) 4 [1447] 1447 root226Tail18
    (if 3 ≤ root226Candidates17.length then search edge 3 [1447, 5641] root226Candidates17 7088 winner else winner) = _
  rw [if_neg (by decide)]
def root226Candidates18 : List Nat := []
set_option maxRecDepth 8192 in
theorem root226Candidates18_checked : candidates edge 4 1447 5683 root226Tail19 winner.bound = root226Candidates18 := by
  unfold candidates
  have row : edge 5683 = (SortedNatTree.contains FastRows.row5683) := by
    change Data.row5683.contains = (SortedNatTree.contains FastRows.row5683)
    exact FastRows.row5683_checked.symm
  rw [row]
  decide
theorem root226Step18 : scan edge (search edge 3) 4 [1447] 1447 root226Tail18 winner =
    scan edge (search edge 3) 4 [1447] 1447 root226Tail19 winner := by
  rw [root226Tail18, scan, if_neg (by decide), root226Candidates18_checked]
  change scan edge (search edge 3) 4 [1447] 1447 root226Tail19
    (if 3 ≤ root226Candidates18.length then search edge 3 [1447, 5683] root226Candidates18 7130 winner else winner) = _
  rw [if_neg (by decide)]
def root226Candidates19 : List Nat := []
set_option maxRecDepth 8192 in
theorem root226Candidates19_checked : candidates edge 4 1447 5689 root226Tail20 winner.bound = root226Candidates19 := by
  unfold candidates
  have row : edge 5689 = (SortedNatTree.contains FastRows.row5689) := by
    change Data.row5689.contains = (SortedNatTree.contains FastRows.row5689)
    exact FastRows.row5689_checked.symm
  rw [row]
  decide
theorem root226Step19 : scan edge (search edge 3) 4 [1447] 1447 root226Tail19 winner =
    scan edge (search edge 3) 4 [1447] 1447 root226Tail20 winner := by
  rw [root226Tail19, scan, if_neg (by decide), root226Candidates19_checked]
  change scan edge (search edge 3) 4 [1447] 1447 root226Tail20
    (if 3 ≤ root226Candidates19.length then search edge 3 [1447, 5689] root226Candidates19 7136 winner else winner) = _
  rw [if_neg (by decide)]
theorem root226Scan20 : scan edge (search edge 3) 4 [1447] 1447 root226Tail20 winner = winner := by rw [root226Tail20, scan, if_pos (by decide)]
theorem root226Scan19 : scan edge (search edge 3) 4 [1447] 1447 root226Tail19 winner = winner :=
  (root226Step19).trans root226Scan20
theorem root226Scan18 : scan edge (search edge 3) 4 [1447] 1447 root226Tail18 winner = winner :=
  (root226Step18).trans root226Scan19
theorem root226Scan17 : scan edge (search edge 3) 4 [1447] 1447 root226Tail17 winner = winner :=
  (root226Step17).trans root226Scan18
theorem root226Scan16 : scan edge (search edge 3) 4 [1447] 1447 root226Tail16 winner = winner :=
  (root226Step16).trans root226Scan17
theorem root226Scan15 : scan edge (search edge 3) 4 [1447] 1447 root226Tail15 winner = winner :=
  (root226Step15).trans root226Scan16
theorem root226Scan14 : scan edge (search edge 3) 4 [1447] 1447 root226Tail14 winner = winner :=
  (root226Step14).trans root226Scan15
theorem root226Scan13 : scan edge (search edge 3) 4 [1447] 1447 root226Tail13 winner = winner :=
  (root226Step13).trans root226Scan14
theorem root226Scan12 : scan edge (search edge 3) 4 [1447] 1447 root226Tail12 winner = winner :=
  (root226Step12).trans root226Scan13
theorem root226Scan11 : scan edge (search edge 3) 4 [1447] 1447 root226Tail11 winner = winner :=
  (root226Step11).trans root226Scan12
theorem root226Scan10 : scan edge (search edge 3) 4 [1447] 1447 root226Tail10 winner = winner :=
  (root226Step10).trans root226Scan11
theorem root226Scan9 : scan edge (search edge 3) 4 [1447] 1447 root226Tail9 winner = winner :=
  (root226Step9).trans root226Scan10
theorem root226Scan8 : scan edge (search edge 3) 4 [1447] 1447 root226Tail8 winner = winner :=
  (root226Step8).trans root226Scan9
theorem root226Scan7 : scan edge (search edge 3) 4 [1447] 1447 root226Tail7 winner = winner :=
  (root226Step7).trans root226Scan8
theorem root226Scan6 : scan edge (search edge 3) 4 [1447] 1447 root226Tail6 winner = winner :=
  (root226Step6).trans root226Scan7
theorem root226Scan5 : scan edge (search edge 3) 4 [1447] 1447 root226Tail5 winner = winner :=
  (root226Step5).trans root226Scan6
theorem root226Scan4 : scan edge (search edge 3) 4 [1447] 1447 root226Tail4 winner = winner :=
  (root226Step4).trans root226Scan5
theorem root226Scan3 : scan edge (search edge 3) 4 [1447] 1447 root226Tail3 winner = winner :=
  (root226Step3).trans root226Scan4
theorem root226Scan2 : scan edge (search edge 3) 4 [1447] 1447 root226Tail2 winner = winner :=
  (root226Step2).trans root226Scan3
theorem root226Scan1 : scan edge (search edge 3) 4 [1447] 1447 root226Tail1 winner = winner :=
  (root226Step1).trans root226Scan2
theorem root226Scan0 : scan edge (search edge 3) 4 [1447] 1447 root226Tail0 winner = winner :=
  (root226Step0).trans root226Scan1
set_option maxRecDepth 4096 in
theorem root226Branch : search edge 4 [1447] root226Candidates 1447 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1447] 1447 root226Tail0 winner = winner
  exact root226Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
