import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root33Candidates : List Nat := Data.row151
set_option maxRecDepth 8192 in
theorem root33Candidates_checked : candidates edge 5 0 151 tail34 winner.bound = root33Candidates := by
  apply candidates_eq_row edge 5 0 151 tail34 root33Candidates winner.bound tail34_ordered
  · decide
  · decide
  · funext q
    rfl
def root33Tail77 : List Nat := []
def root33Tail76 : List Nat := 25741 :: root33Tail77
def root33Tail75 : List Nat := 24229 :: root33Tail76
def root33Tail74 : List Nat := 24181 :: root33Tail75
def root33Tail73 : List Nat := 23827 :: root33Tail74
def root33Tail72 : List Nat := 23767 :: root33Tail73
def root33Tail71 : List Nat := 22669 :: root33Tail72
def root33Tail70 : List Nat := 22501 :: root33Tail71
def root33Tail69 : List Nat := 22483 :: root33Tail70
def root33Tail68 : List Nat := 22441 :: root33Tail69
def root33Tail67 : List Nat := 22189 :: root33Tail68
def root33Tail66 : List Nat := 21433 :: root33Tail67
def root33Tail65 : List Nat := 21247 :: root33Tail66
def root33Tail64 : List Nat := 21031 :: root33Tail65
def root33Tail63 : List Nat := 19213 :: root33Tail64
def root33Tail62 : List Nat := 18973 :: root33Tail63
def root33Tail61 : List Nat := 18553 :: root33Tail62
def root33Tail60 : List Nat := 18451 :: root33Tail61
def root33Tail59 : List Nat := 18307 :: root33Tail60
def root33Tail58 : List Nat := 18289 :: root33Tail59
def root33Tail57 : List Nat := 18121 :: root33Tail58
def root33Tail56 : List Nat := 18097 :: root33Tail57
def root33Tail55 : List Nat := 17881 :: root33Tail56
def root33Tail54 : List Nat := 17827 :: root33Tail55
def root33Tail53 : List Nat := 17419 :: root33Tail54
def root33Tail52 : List Nat := 16369 :: root33Tail53
def root33Tail51 : List Nat := 16273 :: root33Tail52
def root33Tail50 : List Nat := 15787 :: root33Tail51
def root33Tail49 : List Nat := 15391 :: root33Tail50
def root33Tail48 : List Nat := 15031 :: root33Tail49
def root33Tail47 : List Nat := 14929 :: root33Tail48
def root33Tail46 : List Nat := 14821 :: root33Tail47
def root33Tail45 : List Nat := 14779 :: root33Tail46
def root33Tail44 : List Nat := 14653 :: root33Tail45
def root33Tail43 : List Nat := 14419 :: root33Tail44
def root33Tail42 : List Nat := 14293 :: root33Tail43
def root33Tail41 : List Nat := 13807 :: root33Tail42
def root33Tail40 : List Nat := 13513 :: root33Tail41
def root33Tail39 : List Nat := 13309 :: root33Tail40
def root33Tail38 : List Nat := 13183 :: root33Tail39
def root33Tail37 : List Nat := 12967 :: root33Tail38
def root33Tail36 : List Nat := 11443 :: root33Tail37
def root33Tail35 : List Nat := 10663 :: root33Tail36
def root33Tail34 : List Nat := 10657 :: root33Tail35
def root33Tail33 : List Nat := 10531 :: root33Tail34
def root33Tail32 : List Nat := 10321 :: root33Tail33
def root33Tail31 : List Nat := 9883 :: root33Tail32
def root33Tail30 : List Nat := 9547 :: root33Tail31
def root33Tail29 : List Nat := 8779 :: root33Tail30
def root33Tail28 : List Nat := 7933 :: root33Tail29
def root33Tail27 : List Nat := 6967 :: root33Tail28
def root33Tail26 : List Nat := 6733 :: root33Tail27
def root33Tail25 : List Nat := 6607 :: root33Tail26
def root33Tail24 : List Nat := 6217 :: root33Tail25
def root33Tail23 : List Nat := 6007 :: root33Tail24
def root33Tail22 : List Nat := 5413 :: root33Tail23
def root33Tail21 : List Nat := 5281 :: root33Tail22
def root33Tail20 : List Nat := 5197 :: root33Tail21
def root33Tail19 : List Nat := 5011 :: root33Tail20
def root33Tail18 : List Nat := 4801 :: root33Tail19
def root33Tail17 : List Nat := 4783 :: root33Tail18
def root33Tail16 : List Nat := 4657 :: root33Tail17
def root33Tail15 : List Nat := 4273 :: root33Tail16
def root33Tail14 : List Nat := 4027 :: root33Tail15
def root33Tail13 : List Nat := 3019 :: root33Tail14
def root33Tail12 : List Nat := 2689 :: root33Tail13
def root33Tail11 : List Nat := 2683 :: root33Tail12
def root33Tail10 : List Nat := 2311 :: root33Tail11
def root33Tail9 : List Nat := 2281 :: root33Tail10
def root33Tail8 : List Nat := 1303 :: root33Tail9
def root33Tail7 : List Nat := 937 :: root33Tail8
def root33Tail6 : List Nat := 769 :: root33Tail7
def root33Tail5 : List Nat := 631 :: root33Tail6
def root33Tail4 : List Nat := 607 :: root33Tail5
def root33Tail3 : List Nat := 499 :: root33Tail4
def root33Tail2 : List Nat := 433 :: root33Tail3
def root33Tail1 : List Nat := 397 :: root33Tail2
def root33Tail0 : List Nat := 163 :: root33Tail1
def root33Candidates0 : List Nat := [2311, 16369, 18553]
set_option maxRecDepth 8192 in
theorem root33Candidates0_checked : candidates edge 4 151 163 root33Tail1 winner.bound = root33Candidates0 := by
  unfold candidates
  have row : edge 163 = (SortedNatTree.contains FastRows.row163) := by
    change Data.row163.contains = (SortedNatTree.contains FastRows.row163)
    exact FastRows.row163_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root33Child0 : search edge 3 [151, 163] root33Candidates0 314 winner = winner := by decide
theorem root33Step0 : scan edge (search edge 3) 4 [151] 151 root33Tail0 winner =
    scan edge (search edge 3) 4 [151] 151 root33Tail1 winner := by
  rw [root33Tail0, scan, if_neg (by decide), root33Candidates0_checked]
  change scan edge (search edge 3) 4 [151] 151 root33Tail1
    (if 3 ≤ root33Candidates0.length then search edge 3 [151, 163] root33Candidates0 314 winner else winner) = _
  rw [if_pos (by decide), root33Child0]
def root33Candidates1 : List Nat := [4657, 5197, 7933, 10663, 23767]
set_option maxRecDepth 8192 in
theorem root33Candidates1_checked : candidates edge 4 151 397 root33Tail2 winner.bound = root33Candidates1 := by
  unfold candidates
  have row : edge 397 = (SortedNatTree.contains FastRows.row397) := by
    change Data.row397.contains = (SortedNatTree.contains FastRows.row397)
    exact FastRows.row397_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root33Child1 : search edge 3 [151, 397] root33Candidates1 548 winner = winner := by decide
theorem root33Step1 : scan edge (search edge 3) 4 [151] 151 root33Tail1 winner =
    scan edge (search edge 3) 4 [151] 151 root33Tail2 winner := by
  rw [root33Tail1, scan, if_neg (by decide), root33Candidates1_checked]
  change scan edge (search edge 3) 4 [151] 151 root33Tail2
    (if 3 ≤ root33Candidates1.length then search edge 3 [151, 397] root33Candidates1 548 winner else winner) = _
  rw [if_pos (by decide), root33Child1]
def root33Candidates2 : List Nat := [5281, 14653, 14779, 17419, 18973]
set_option maxRecDepth 8192 in
theorem root33Candidates2_checked : candidates edge 4 151 433 root33Tail3 winner.bound = root33Candidates2 := by
  unfold candidates
  have row : edge 433 = (SortedNatTree.contains FastRows.row433) := by
    change Data.row433.contains = (SortedNatTree.contains FastRows.row433)
    exact FastRows.row433_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root33Child2 : search edge 3 [151, 433] root33Candidates2 584 winner = winner := by decide
theorem root33Step2 : scan edge (search edge 3) 4 [151] 151 root33Tail2 winner =
    scan edge (search edge 3) 4 [151] 151 root33Tail3 winner := by
  rw [root33Tail2, scan, if_neg (by decide), root33Candidates2_checked]
  change scan edge (search edge 3) 4 [151] 151 root33Tail3
    (if 3 ≤ root33Candidates2.length then search edge 3 [151, 433] root33Candidates2 584 winner else winner) = _
  rw [if_pos (by decide), root33Child2]
def root33Candidates3 : List Nat := [1303, 5281, 24181]
set_option maxRecDepth 8192 in
theorem root33Candidates3_checked : candidates edge 4 151 499 root33Tail4 winner.bound = root33Candidates3 := by
  unfold candidates
  have row : edge 499 = (SortedNatTree.contains FastRows.row499) := by
    change Data.row499.contains = (SortedNatTree.contains FastRows.row499)
    exact FastRows.row499_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root33Child3 : search edge 3 [151, 499] root33Candidates3 650 winner = winner := by decide
theorem root33Step3 : scan edge (search edge 3) 4 [151] 151 root33Tail3 winner =
    scan edge (search edge 3) 4 [151] 151 root33Tail4 winner := by
  rw [root33Tail3, scan, if_neg (by decide), root33Candidates3_checked]
  change scan edge (search edge 3) 4 [151] 151 root33Tail4
    (if 3 ≤ root33Candidates3.length then search edge 3 [151, 499] root33Candidates3 650 winner else winner) = _
  rw [if_pos (by decide), root33Child3]
def root33Candidates4 : List Nat := [9547]
set_option maxRecDepth 8192 in
theorem root33Candidates4_checked : candidates edge 4 151 607 root33Tail5 winner.bound = root33Candidates4 := by
  unfold candidates
  have row : edge 607 = (SortedNatTree.contains FastRows.row607) := by
    change Data.row607.contains = (SortedNatTree.contains FastRows.row607)
    exact FastRows.row607_checked.symm
  rw [row]
  decide
theorem root33Step4 : scan edge (search edge 3) 4 [151] 151 root33Tail4 winner =
    scan edge (search edge 3) 4 [151] 151 root33Tail5 winner := by
  rw [root33Tail4, scan, if_neg (by decide), root33Candidates4_checked]
  change scan edge (search edge 3) 4 [151] 151 root33Tail5
    (if 3 ≤ root33Candidates4.length then search edge 3 [151, 607] root33Candidates4 758 winner else winner) = _
  rw [if_neg (by decide)]
def root33Candidates5 : List Nat := [4801, 16369, 17419, 17827]
set_option maxRecDepth 8192 in
theorem root33Candidates5_checked : candidates edge 4 151 631 root33Tail6 winner.bound = root33Candidates5 := by
  unfold candidates
  have row : edge 631 = (SortedNatTree.contains FastRows.row631) := by
    change Data.row631.contains = (SortedNatTree.contains FastRows.row631)
    exact FastRows.row631_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root33Child5 : search edge 3 [151, 631] root33Candidates5 782 winner = winner := by decide
theorem root33Step5 : scan edge (search edge 3) 4 [151] 151 root33Tail5 winner =
    scan edge (search edge 3) 4 [151] 151 root33Tail6 winner := by
  rw [root33Tail5, scan, if_neg (by decide), root33Candidates5_checked]
  change scan edge (search edge 3) 4 [151] 151 root33Tail6
    (if 3 ≤ root33Candidates5.length then search edge 3 [151, 631] root33Candidates5 782 winner else winner) = _
  rw [if_pos (by decide), root33Child5]
def root33Candidates6 : List Nat := [5197, 6967, 12967, 13807, 23767]
set_option maxRecDepth 8192 in
theorem root33Candidates6_checked : candidates edge 4 151 769 root33Tail7 winner.bound = root33Candidates6 := by
  unfold candidates
  have row : edge 769 = (SortedNatTree.contains FastRows.row769) := by
    change Data.row769.contains = (SortedNatTree.contains FastRows.row769)
    exact FastRows.row769_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root33Child6 : search edge 3 [151, 769] root33Candidates6 920 winner = winner := by decide
theorem root33Step6 : scan edge (search edge 3) 4 [151] 151 root33Tail6 winner =
    scan edge (search edge 3) 4 [151] 151 root33Tail7 winner := by
  rw [root33Tail6, scan, if_neg (by decide), root33Candidates6_checked]
  change scan edge (search edge 3) 4 [151] 151 root33Tail7
    (if 3 ≤ root33Candidates6.length then search edge 3 [151, 769] root33Candidates6 920 winner else winner) = _
  rw [if_pos (by decide), root33Child6]
def root33Candidates7 : List Nat := [4273, 5281, 10321, 15031, 21247, 23767]
set_option maxRecDepth 8192 in
theorem root33Candidates7_checked : candidates edge 4 151 937 root33Tail8 winner.bound = root33Candidates7 := by
  unfold candidates
  have row : edge 937 = (SortedNatTree.contains FastRows.row937) := by
    change Data.row937.contains = (SortedNatTree.contains FastRows.row937)
    exact FastRows.row937_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root33Child7 : search edge 3 [151, 937] root33Candidates7 1088 winner = winner := by decide
theorem root33Step7 : scan edge (search edge 3) 4 [151] 151 root33Tail7 winner =
    scan edge (search edge 3) 4 [151] 151 root33Tail8 winner := by
  rw [root33Tail7, scan, if_neg (by decide), root33Candidates7_checked]
  change scan edge (search edge 3) 4 [151] 151 root33Tail8
    (if 3 ≤ root33Candidates7.length then search edge 3 [151, 937] root33Candidates7 1088 winner else winner) = _
  rw [if_pos (by decide), root33Child7]
def root33Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root33Candidates8_checked : candidates edge 4 151 1303 root33Tail9 winner.bound = root33Candidates8 := by
  unfold candidates
  have row : edge 1303 = (SortedNatTree.contains FastRows.row1303) := by
    change Data.row1303.contains = (SortedNatTree.contains FastRows.row1303)
    exact FastRows.row1303_checked.symm
  rw [row]
  decide
theorem root33Step8 : scan edge (search edge 3) 4 [151] 151 root33Tail8 winner =
    scan edge (search edge 3) 4 [151] 151 root33Tail9 winner := by
  rw [root33Tail8, scan, if_neg (by decide), root33Candidates8_checked]
  change scan edge (search edge 3) 4 [151] 151 root33Tail9
    (if 3 ≤ root33Candidates8.length then search edge 3 [151, 1303] root33Candidates8 1454 winner else winner) = _
  rw [if_neg (by decide)]
def root33Candidates9 : List Nat := [4273, 18973, 21031]
set_option maxRecDepth 8192 in
theorem root33Candidates9_checked : candidates edge 4 151 2281 root33Tail10 winner.bound = root33Candidates9 := by
  unfold candidates
  have row : edge 2281 = (SortedNatTree.contains FastRows.row2281) := by
    change Data.row2281.contains = (SortedNatTree.contains FastRows.row2281)
    exact FastRows.row2281_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root33Child9 : search edge 3 [151, 2281] root33Candidates9 2432 winner = winner := by decide
theorem root33Step9 : scan edge (search edge 3) 4 [151] 151 root33Tail9 winner =
    scan edge (search edge 3) 4 [151] 151 root33Tail10 winner := by
  rw [root33Tail9, scan, if_neg (by decide), root33Candidates9_checked]
  change scan edge (search edge 3) 4 [151] 151 root33Tail10
    (if 3 ≤ root33Candidates9.length then search edge 3 [151, 2281] root33Candidates9 2432 winner else winner) = _
  rw [if_pos (by decide), root33Child9]
def root33Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root33Candidates10_checked : candidates edge 4 151 2311 root33Tail11 winner.bound = root33Candidates10 := by
  unfold candidates
  have row : edge 2311 = (SortedNatTree.contains FastRows.row2311) := by
    change Data.row2311.contains = (SortedNatTree.contains FastRows.row2311)
    exact FastRows.row2311_checked.symm
  rw [row]
  decide
theorem root33Step10 : scan edge (search edge 3) 4 [151] 151 root33Tail10 winner =
    scan edge (search edge 3) 4 [151] 151 root33Tail11 winner := by
  rw [root33Tail10, scan, if_neg (by decide), root33Candidates10_checked]
  change scan edge (search edge 3) 4 [151] 151 root33Tail11
    (if 3 ≤ root33Candidates10.length then search edge 3 [151, 2311] root33Candidates10 2462 winner else winner) = _
  rw [if_neg (by decide)]
def root33Candidates11 : List Nat := [14419, 22189]
set_option maxRecDepth 8192 in
theorem root33Candidates11_checked : candidates edge 4 151 2683 root33Tail12 winner.bound = root33Candidates11 := by
  unfold candidates
  have row : edge 2683 = (SortedNatTree.contains FastRows.row2683) := by
    change Data.row2683.contains = (SortedNatTree.contains FastRows.row2683)
    exact FastRows.row2683_checked.symm
  rw [row]
  decide
theorem root33Step11 : scan edge (search edge 3) 4 [151] 151 root33Tail11 winner =
    scan edge (search edge 3) 4 [151] 151 root33Tail12 winner := by
  rw [root33Tail11, scan, if_neg (by decide), root33Candidates11_checked]
  change scan edge (search edge 3) 4 [151] 151 root33Tail12
    (if 3 ≤ root33Candidates11.length then search edge 3 [151, 2683] root33Candidates11 2834 winner else winner) = _
  rw [if_neg (by decide)]
def root33Candidates12 : List Nat := [22189]
set_option maxRecDepth 8192 in
theorem root33Candidates12_checked : candidates edge 4 151 2689 root33Tail13 winner.bound = root33Candidates12 := by
  unfold candidates
  have row : edge 2689 = (SortedNatTree.contains FastRows.row2689) := by
    change Data.row2689.contains = (SortedNatTree.contains FastRows.row2689)
    exact FastRows.row2689_checked.symm
  rw [row]
  decide
theorem root33Step12 : scan edge (search edge 3) 4 [151] 151 root33Tail12 winner =
    scan edge (search edge 3) 4 [151] 151 root33Tail13 winner := by
  rw [root33Tail12, scan, if_neg (by decide), root33Candidates12_checked]
  change scan edge (search edge 3) 4 [151] 151 root33Tail13
    (if 3 ≤ root33Candidates12.length then search edge 3 [151, 2689] root33Candidates12 2840 winner else winner) = _
  rw [if_neg (by decide)]
def root33Candidates13 : List Nat := [5413, 22441]
set_option maxRecDepth 8192 in
theorem root33Candidates13_checked : candidates edge 4 151 3019 root33Tail14 winner.bound = root33Candidates13 := by
  unfold candidates
  have row : edge 3019 = (SortedNatTree.contains FastRows.row3019) := by
    change Data.row3019.contains = (SortedNatTree.contains FastRows.row3019)
    exact FastRows.row3019_checked.symm
  rw [row]
  decide
theorem root33Step13 : scan edge (search edge 3) 4 [151] 151 root33Tail13 winner =
    scan edge (search edge 3) 4 [151] 151 root33Tail14 winner := by
  rw [root33Tail13, scan, if_neg (by decide), root33Candidates13_checked]
  change scan edge (search edge 3) 4 [151] 151 root33Tail14
    (if 3 ≤ root33Candidates13.length then search edge 3 [151, 3019] root33Candidates13 3170 winner else winner) = _
  rw [if_neg (by decide)]
def root33Candidates14 : List Nat := [4783, 5413, 11443, 14419, 14929]
set_option maxRecDepth 8192 in
theorem root33Candidates14_checked : candidates edge 4 151 4027 root33Tail15 winner.bound = root33Candidates14 := by
  unfold candidates
  have row : edge 4027 = (SortedNatTree.contains FastRows.row4027) := by
    change Data.row4027.contains = (SortedNatTree.contains FastRows.row4027)
    exact FastRows.row4027_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root33Child14 : search edge 3 [151, 4027] root33Candidates14 4178 winner = winner := by decide
theorem root33Step14 : scan edge (search edge 3) 4 [151] 151 root33Tail14 winner =
    scan edge (search edge 3) 4 [151] 151 root33Tail15 winner := by
  rw [root33Tail14, scan, if_neg (by decide), root33Candidates14_checked]
  change scan edge (search edge 3) 4 [151] 151 root33Tail15
    (if 3 ≤ root33Candidates14.length then search edge 3 [151, 4027] root33Candidates14 4178 winner else winner) = _
  rw [if_pos (by decide), root33Child14]
def root33Candidates15 : List Nat := [9883]
set_option maxRecDepth 8192 in
theorem root33Candidates15_checked : candidates edge 4 151 4273 root33Tail16 winner.bound = root33Candidates15 := by
  unfold candidates
  have row : edge 4273 = (SortedNatTree.contains FastRows.row4273) := by
    change Data.row4273.contains = (SortedNatTree.contains FastRows.row4273)
    exact FastRows.row4273_checked.symm
  rw [row]
  decide
theorem root33Step15 : scan edge (search edge 3) 4 [151] 151 root33Tail15 winner =
    scan edge (search edge 3) 4 [151] 151 root33Tail16 winner := by
  rw [root33Tail15, scan, if_neg (by decide), root33Candidates15_checked]
  change scan edge (search edge 3) 4 [151] 151 root33Tail16
    (if 3 ≤ root33Candidates15.length then search edge 3 [151, 4273] root33Candidates15 4424 winner else winner) = _
  rw [if_neg (by decide)]
def root33Candidates16 : List Nat := [6217, 7933, 14929]
set_option maxRecDepth 8192 in
theorem root33Candidates16_checked : candidates edge 4 151 4657 root33Tail17 winner.bound = root33Candidates16 := by
  unfold candidates
  have row : edge 4657 = (SortedNatTree.contains FastRows.row4657) := by
    change Data.row4657.contains = (SortedNatTree.contains FastRows.row4657)
    exact FastRows.row4657_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root33Child16 : search edge 3 [151, 4657] root33Candidates16 4808 winner = winner := by decide
theorem root33Step16 : scan edge (search edge 3) 4 [151] 151 root33Tail16 winner =
    scan edge (search edge 3) 4 [151] 151 root33Tail17 winner := by
  rw [root33Tail16, scan, if_neg (by decide), root33Candidates16_checked]
  change scan edge (search edge 3) 4 [151] 151 root33Tail17
    (if 3 ≤ root33Candidates16.length then search edge 3 [151, 4657] root33Candidates16 4808 winner else winner) = _
  rw [if_pos (by decide), root33Child16]
def root33Candidates17 : List Nat := [14293]
set_option maxRecDepth 8192 in
theorem root33Candidates17_checked : candidates edge 4 151 4783 root33Tail18 winner.bound = root33Candidates17 := by
  unfold candidates
  have row : edge 4783 = (SortedNatTree.contains FastRows.row4783) := by
    change Data.row4783.contains = (SortedNatTree.contains FastRows.row4783)
    exact FastRows.row4783_checked.symm
  rw [row]
  decide
theorem root33Step17 : scan edge (search edge 3) 4 [151] 151 root33Tail17 winner =
    scan edge (search edge 3) 4 [151] 151 root33Tail18 winner := by
  rw [root33Tail17, scan, if_neg (by decide), root33Candidates17_checked]
  change scan edge (search edge 3) 4 [151] 151 root33Tail18
    (if 3 ≤ root33Candidates17.length then search edge 3 [151, 4783] root33Candidates17 4934 winner else winner) = _
  rw [if_neg (by decide)]
def root33Candidates18 : List Nat := [5197, 6733, 14419]
set_option maxRecDepth 8192 in
theorem root33Candidates18_checked : candidates edge 4 151 4801 root33Tail19 winner.bound = root33Candidates18 := by
  unfold candidates
  have row : edge 4801 = (SortedNatTree.contains FastRows.row4801) := by
    change Data.row4801.contains = (SortedNatTree.contains FastRows.row4801)
    exact FastRows.row4801_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root33Child18 : search edge 3 [151, 4801] root33Candidates18 4952 winner = winner := by decide
theorem root33Step18 : scan edge (search edge 3) 4 [151] 151 root33Tail18 winner =
    scan edge (search edge 3) 4 [151] 151 root33Tail19 winner := by
  rw [root33Tail18, scan, if_neg (by decide), root33Candidates18_checked]
  change scan edge (search edge 3) 4 [151] 151 root33Tail19
    (if 3 ≤ root33Candidates18.length then search edge 3 [151, 4801] root33Candidates18 4952 winner else winner) = _
  rw [if_pos (by decide), root33Child18]
def root33Candidates19 : List Nat := [10531]
set_option maxRecDepth 8192 in
theorem root33Candidates19_checked : candidates edge 4 151 5011 root33Tail20 winner.bound = root33Candidates19 := by
  unfold candidates
  have row : edge 5011 = (SortedNatTree.contains FastRows.row5011) := by
    change Data.row5011.contains = (SortedNatTree.contains FastRows.row5011)
    exact FastRows.row5011_checked.symm
  rw [row]
  decide
theorem root33Step19 : scan edge (search edge 3) 4 [151] 151 root33Tail19 winner =
    scan edge (search edge 3) 4 [151] 151 root33Tail20 winner := by
  rw [root33Tail19, scan, if_neg (by decide), root33Candidates19_checked]
  change scan edge (search edge 3) 4 [151] 151 root33Tail20
    (if 3 ≤ root33Candidates19.length then search edge 3 [151, 5011] root33Candidates19 5162 winner else winner) = _
  rw [if_neg (by decide)]
def root33Candidates20 : List Nat := [6733, 10657, 11443, 14653, 18553]
set_option maxRecDepth 8192 in
theorem root33Candidates20_checked : candidates edge 4 151 5197 root33Tail21 winner.bound = root33Candidates20 := by
  unfold candidates
  have row : edge 5197 = (SortedNatTree.contains FastRows.row5197) := by
    change Data.row5197.contains = (SortedNatTree.contains FastRows.row5197)
    exact FastRows.row5197_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root33Child20 : search edge 3 [151, 5197] root33Candidates20 5348 winner = winner := by decide
theorem root33Step20 : scan edge (search edge 3) 4 [151] 151 root33Tail20 winner =
    scan edge (search edge 3) 4 [151] 151 root33Tail21 winner := by
  rw [root33Tail20, scan, if_neg (by decide), root33Candidates20_checked]
  change scan edge (search edge 3) 4 [151] 151 root33Tail21
    (if 3 ≤ root33Candidates20.length then search edge 3 [151, 5197] root33Candidates20 5348 winner else winner) = _
  rw [if_pos (by decide), root33Child20]
def root33Candidates21 : List Nat := []
set_option maxRecDepth 8192 in
theorem root33Candidates21_checked : candidates edge 4 151 5281 root33Tail22 winner.bound = root33Candidates21 := by
  unfold candidates
  have row : edge 5281 = (SortedNatTree.contains FastRows.row5281) := by
    change Data.row5281.contains = (SortedNatTree.contains FastRows.row5281)
    exact FastRows.row5281_checked.symm
  rw [row]
  decide
theorem root33Step21 : scan edge (search edge 3) 4 [151] 151 root33Tail21 winner =
    scan edge (search edge 3) 4 [151] 151 root33Tail22 winner := by
  rw [root33Tail21, scan, if_neg (by decide), root33Candidates21_checked]
  change scan edge (search edge 3) 4 [151] 151 root33Tail22
    (if 3 ≤ root33Candidates21.length then search edge 3 [151, 5281] root33Candidates21 5432 winner else winner) = _
  rw [if_neg (by decide)]
def root33Candidates22 : List Nat := []
set_option maxRecDepth 8192 in
theorem root33Candidates22_checked : candidates edge 4 151 5413 root33Tail23 winner.bound = root33Candidates22 := by
  unfold candidates
  have row : edge 5413 = (SortedNatTree.contains FastRows.row5413) := by
    change Data.row5413.contains = (SortedNatTree.contains FastRows.row5413)
    exact FastRows.row5413_checked.symm
  rw [row]
  decide
theorem root33Step22 : scan edge (search edge 3) 4 [151] 151 root33Tail22 winner =
    scan edge (search edge 3) 4 [151] 151 root33Tail23 winner := by
  rw [root33Tail22, scan, if_neg (by decide), root33Candidates22_checked]
  change scan edge (search edge 3) 4 [151] 151 root33Tail23
    (if 3 ≤ root33Candidates22.length then search edge 3 [151, 5413] root33Candidates22 5564 winner else winner) = _
  rw [if_neg (by decide)]
def root33Candidates23 : List Nat := [7933, 18973]
set_option maxRecDepth 8192 in
theorem root33Candidates23_checked : candidates edge 4 151 6007 root33Tail24 winner.bound = root33Candidates23 := by
  unfold candidates
  have row : edge 6007 = (SortedNatTree.contains FastRows.row6007) := by
    change Data.row6007.contains = (SortedNatTree.contains FastRows.row6007)
    exact FastRows.row6007_checked.symm
  rw [row]
  decide
theorem root33Step23 : scan edge (search edge 3) 4 [151] 151 root33Tail23 winner =
    scan edge (search edge 3) 4 [151] 151 root33Tail24 winner := by
  rw [root33Tail23, scan, if_neg (by decide), root33Candidates23_checked]
  change scan edge (search edge 3) 4 [151] 151 root33Tail24
    (if 3 ≤ root33Candidates23.length then search edge 3 [151, 6007] root33Candidates23 6158 winner else winner) = _
  rw [if_neg (by decide)]
def root33Candidates24 : List Nat := [7933, 16369]
set_option maxRecDepth 8192 in
theorem root33Candidates24_checked : candidates edge 4 151 6217 root33Tail25 winner.bound = root33Candidates24 := by
  unfold candidates
  have row : edge 6217 = (SortedNatTree.contains FastRows.row6217) := by
    change Data.row6217.contains = (SortedNatTree.contains FastRows.row6217)
    exact FastRows.row6217_checked.symm
  rw [row]
  decide
theorem root33Step24 : scan edge (search edge 3) 4 [151] 151 root33Tail24 winner =
    scan edge (search edge 3) 4 [151] 151 root33Tail25 winner := by
  rw [root33Tail24, scan, if_neg (by decide), root33Candidates24_checked]
  change scan edge (search edge 3) 4 [151] 151 root33Tail25
    (if 3 ≤ root33Candidates24.length then search edge 3 [151, 6217] root33Candidates24 6368 winner else winner) = _
  rw [if_neg (by decide)]
theorem root33Scan25 : scan edge (search edge 3) 4 [151] 151 root33Tail25 winner = winner := by rw [root33Tail25, scan, if_pos (by decide)]
theorem root33Scan24 : scan edge (search edge 3) 4 [151] 151 root33Tail24 winner = winner :=
  (root33Step24).trans root33Scan25
theorem root33Scan23 : scan edge (search edge 3) 4 [151] 151 root33Tail23 winner = winner :=
  (root33Step23).trans root33Scan24
theorem root33Scan22 : scan edge (search edge 3) 4 [151] 151 root33Tail22 winner = winner :=
  (root33Step22).trans root33Scan23
theorem root33Scan21 : scan edge (search edge 3) 4 [151] 151 root33Tail21 winner = winner :=
  (root33Step21).trans root33Scan22
theorem root33Scan20 : scan edge (search edge 3) 4 [151] 151 root33Tail20 winner = winner :=
  (root33Step20).trans root33Scan21
theorem root33Scan19 : scan edge (search edge 3) 4 [151] 151 root33Tail19 winner = winner :=
  (root33Step19).trans root33Scan20
theorem root33Scan18 : scan edge (search edge 3) 4 [151] 151 root33Tail18 winner = winner :=
  (root33Step18).trans root33Scan19
theorem root33Scan17 : scan edge (search edge 3) 4 [151] 151 root33Tail17 winner = winner :=
  (root33Step17).trans root33Scan18
theorem root33Scan16 : scan edge (search edge 3) 4 [151] 151 root33Tail16 winner = winner :=
  (root33Step16).trans root33Scan17
theorem root33Scan15 : scan edge (search edge 3) 4 [151] 151 root33Tail15 winner = winner :=
  (root33Step15).trans root33Scan16
theorem root33Scan14 : scan edge (search edge 3) 4 [151] 151 root33Tail14 winner = winner :=
  (root33Step14).trans root33Scan15
theorem root33Scan13 : scan edge (search edge 3) 4 [151] 151 root33Tail13 winner = winner :=
  (root33Step13).trans root33Scan14
theorem root33Scan12 : scan edge (search edge 3) 4 [151] 151 root33Tail12 winner = winner :=
  (root33Step12).trans root33Scan13
theorem root33Scan11 : scan edge (search edge 3) 4 [151] 151 root33Tail11 winner = winner :=
  (root33Step11).trans root33Scan12
theorem root33Scan10 : scan edge (search edge 3) 4 [151] 151 root33Tail10 winner = winner :=
  (root33Step10).trans root33Scan11
theorem root33Scan9 : scan edge (search edge 3) 4 [151] 151 root33Tail9 winner = winner :=
  (root33Step9).trans root33Scan10
theorem root33Scan8 : scan edge (search edge 3) 4 [151] 151 root33Tail8 winner = winner :=
  (root33Step8).trans root33Scan9
theorem root33Scan7 : scan edge (search edge 3) 4 [151] 151 root33Tail7 winner = winner :=
  (root33Step7).trans root33Scan8
theorem root33Scan6 : scan edge (search edge 3) 4 [151] 151 root33Tail6 winner = winner :=
  (root33Step6).trans root33Scan7
theorem root33Scan5 : scan edge (search edge 3) 4 [151] 151 root33Tail5 winner = winner :=
  (root33Step5).trans root33Scan6
theorem root33Scan4 : scan edge (search edge 3) 4 [151] 151 root33Tail4 winner = winner :=
  (root33Step4).trans root33Scan5
theorem root33Scan3 : scan edge (search edge 3) 4 [151] 151 root33Tail3 winner = winner :=
  (root33Step3).trans root33Scan4
theorem root33Scan2 : scan edge (search edge 3) 4 [151] 151 root33Tail2 winner = winner :=
  (root33Step2).trans root33Scan3
theorem root33Scan1 : scan edge (search edge 3) 4 [151] 151 root33Tail1 winner = winner :=
  (root33Step1).trans root33Scan2
theorem root33Scan0 : scan edge (search edge 3) 4 [151] 151 root33Tail0 winner = winner :=
  (root33Step0).trans root33Scan1
set_option maxRecDepth 4096 in
theorem root33Branch : search edge 4 [151] root33Candidates 151 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [151] 151 root33Tail0 winner = winner
  exact root33Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
