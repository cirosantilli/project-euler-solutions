import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root68Candidates : List Nat := Data.row353
set_option maxRecDepth 8192 in
theorem root68Candidates_checked : candidates edge 5 0 353 tail69 winner.bound = root68Candidates := by
  apply candidates_eq_row edge 5 0 353 tail69 root68Candidates winner.bound tail69_ordered
  · decide
  · decide
  · funext q
    rfl
def root68Tail78 : List Nat := []
def root68Tail77 : List Nat := 25307 :: root68Tail78
def root68Tail76 : List Nat := 24953 :: root68Tail77
def root68Tail75 : List Nat := 24413 :: root68Tail76
def root68Tail74 : List Nat := 23741 :: root68Tail75
def root68Tail73 : List Nat := 23633 :: root68Tail74
def root68Tail72 : List Nat := 23609 :: root68Tail73
def root68Tail71 : List Nat := 23447 :: root68Tail72
def root68Tail70 : List Nat := 22643 :: root68Tail71
def root68Tail69 : List Nat := 22283 :: root68Tail70
def root68Tail68 : List Nat := 21599 :: root68Tail69
def root68Tail67 : List Nat := 21149 :: root68Tail68
def root68Tail66 : List Nat := 20963 :: root68Tail67
def root68Tail65 : List Nat := 19433 :: root68Tail66
def root68Tail64 : List Nat := 19289 :: root68Tail65
def root68Tail63 : List Nat := 19259 :: root68Tail64
def root68Tail62 : List Nat := 19139 :: root68Tail63
def root68Tail61 : List Nat := 17957 :: root68Tail62
def root68Tail60 : List Nat := 17609 :: root68Tail61
def root68Tail59 : List Nat := 17291 :: root68Tail60
def root68Tail58 : List Nat := 16493 :: root68Tail59
def root68Tail57 : List Nat := 16349 :: root68Tail58
def root68Tail56 : List Nat := 15887 :: root68Tail57
def root68Tail55 : List Nat := 15149 :: root68Tail56
def root68Tail54 : List Nat := 14753 :: root68Tail55
def root68Tail53 : List Nat := 14519 :: root68Tail54
def root68Tail52 : List Nat := 14009 :: root68Tail53
def root68Tail51 : List Nat := 13721 :: root68Tail52
def root68Tail50 : List Nat := 12713 :: root68Tail51
def root68Tail49 : List Nat := 12527 :: root68Tail50
def root68Tail48 : List Nat := 12269 :: root68Tail49
def root68Tail47 : List Nat := 12143 :: root68Tail48
def root68Tail46 : List Nat := 11909 :: root68Tail47
def root68Tail45 : List Nat := 11489 :: root68Tail46
def root68Tail44 : List Nat := 11321 :: root68Tail45
def root68Tail43 : List Nat := 10973 :: root68Tail44
def root68Tail42 : List Nat := 10889 :: root68Tail43
def root68Tail41 : List Nat := 10859 :: root68Tail42
def root68Tail40 : List Nat := 10739 :: root68Tail41
def root68Tail39 : List Nat := 10427 :: root68Tail40
def root68Tail38 : List Nat := 9839 :: root68Tail39
def root68Tail37 : List Nat := 9743 :: root68Tail38
def root68Tail36 : List Nat := 9623 :: root68Tail37
def root68Tail35 : List Nat := 9521 :: root68Tail36
def root68Tail34 : List Nat := 9413 :: root68Tail35
def root68Tail33 : List Nat := 9059 :: root68Tail34
def root68Tail32 : List Nat := 8831 :: root68Tail33
def root68Tail31 : List Nat := 8219 :: root68Tail32
def root68Tail30 : List Nat := 7907 :: root68Tail31
def root68Tail29 : List Nat := 7559 :: root68Tail30
def root68Tail28 : List Nat := 7517 :: root68Tail29
def root68Tail27 : List Nat := 6791 :: root68Tail28
def root68Tail26 : List Nat := 6551 :: root68Tail27
def root68Tail25 : List Nat := 6389 :: root68Tail26
def root68Tail24 : List Nat := 6131 :: root68Tail25
def root68Tail23 : List Nat := 5879 :: root68Tail24
def root68Tail22 : List Nat := 5849 :: root68Tail23
def root68Tail21 : List Nat := 5573 :: root68Tail22
def root68Tail20 : List Nat := 5087 :: root68Tail21
def root68Tail19 : List Nat := 4967 :: root68Tail20
def root68Tail18 : List Nat := 4919 :: root68Tail19
def root68Tail17 : List Nat := 4673 :: root68Tail18
def root68Tail16 : List Nat := 4493 :: root68Tail17
def root68Tail15 : List Nat := 4397 :: root68Tail16
def root68Tail14 : List Nat := 4337 :: root68Tail15
def root68Tail13 : List Nat := 3779 :: root68Tail14
def root68Tail12 : List Nat := 3701 :: root68Tail13
def root68Tail11 : List Nat := 3371 :: root68Tail12
def root68Tail10 : List Nat := 3203 :: root68Tail11
def root68Tail9 : List Nat := 2741 :: root68Tail10
def root68Tail8 : List Nat := 2687 :: root68Tail9
def root68Tail7 : List Nat := 2213 :: root68Tail8
def root68Tail6 : List Nat := 1877 :: root68Tail7
def root68Tail5 : List Nat := 1499 :: root68Tail6
def root68Tail4 : List Nat := 1481 :: root68Tail5
def root68Tail3 : List Nat := 1433 :: root68Tail4
def root68Tail2 : List Nat := 797 :: root68Tail3
def root68Tail1 : List Nat := 443 :: root68Tail2
def root68Tail0 : List Nat := 359 :: root68Tail1
def root68Candidates0 : List Nat := [1499, 3203, 7907, 16493]
set_option maxRecDepth 8192 in
theorem root68Candidates0_checked : candidates edge 4 353 359 root68Tail1 winner.bound = root68Candidates0 := by
  unfold candidates
  have row : edge 359 = (SortedNatTree.contains FastRows.row359) := by
    change Data.row359.contains = (SortedNatTree.contains FastRows.row359)
    exact FastRows.row359_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root68Child0 : search edge 3 [353, 359] root68Candidates0 712 winner = winner := by decide
theorem root68Step0 : scan edge (search edge 3) 4 [353] 353 root68Tail0 winner =
    scan edge (search edge 3) 4 [353] 353 root68Tail1 winner := by
  rw [root68Tail0, scan, if_neg (by decide), root68Candidates0_checked]
  change scan edge (search edge 3) 4 [353] 353 root68Tail1
    (if 3 ≤ root68Candidates0.length then search edge 3 [353, 359] root68Candidates0 712 winner else winner) = _
  rw [if_pos (by decide), root68Child0]
def root68Candidates1 : List Nat := [3203, 4673, 6389]
set_option maxRecDepth 8192 in
theorem root68Candidates1_checked : candidates edge 4 353 443 root68Tail2 winner.bound = root68Candidates1 := by
  unfold candidates
  have row : edge 443 = (SortedNatTree.contains FastRows.row443) := by
    change Data.row443.contains = (SortedNatTree.contains FastRows.row443)
    exact FastRows.row443_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root68Child1 : search edge 3 [353, 443] root68Candidates1 796 winner = winner := by decide
theorem root68Step1 : scan edge (search edge 3) 4 [353] 353 root68Tail1 winner =
    scan edge (search edge 3) 4 [353] 353 root68Tail2 winner := by
  rw [root68Tail1, scan, if_neg (by decide), root68Candidates1_checked]
  change scan edge (search edge 3) 4 [353] 353 root68Tail2
    (if 3 ≤ root68Candidates1.length then search edge 3 [353, 443] root68Candidates1 796 winner else winner) = _
  rw [if_pos (by decide), root68Child1]
def root68Candidates2 : List Nat := [6389, 10973, 21149]
set_option maxRecDepth 8192 in
theorem root68Candidates2_checked : candidates edge 4 353 797 root68Tail3 winner.bound = root68Candidates2 := by
  unfold candidates
  have row : edge 797 = (SortedNatTree.contains FastRows.row797) := by
    change Data.row797.contains = (SortedNatTree.contains FastRows.row797)
    exact FastRows.row797_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root68Child2 : search edge 3 [353, 797] root68Candidates2 1150 winner = winner := by decide
theorem root68Step2 : scan edge (search edge 3) 4 [353] 353 root68Tail2 winner =
    scan edge (search edge 3) 4 [353] 353 root68Tail3 winner := by
  rw [root68Tail2, scan, if_neg (by decide), root68Candidates2_checked]
  change scan edge (search edge 3) 4 [353] 353 root68Tail3
    (if 3 ≤ root68Candidates2.length then search edge 3 [353, 797] root68Candidates2 1150 winner else winner) = _
  rw [if_pos (by decide), root68Child2]
def root68Candidates3 : List Nat := [17957]
set_option maxRecDepth 8192 in
theorem root68Candidates3_checked : candidates edge 4 353 1433 root68Tail4 winner.bound = root68Candidates3 := by
  unfold candidates
  have row : edge 1433 = (SortedNatTree.contains FastRows.row1433) := by
    change Data.row1433.contains = (SortedNatTree.contains FastRows.row1433)
    exact FastRows.row1433_checked.symm
  rw [row]
  decide
theorem root68Step3 : scan edge (search edge 3) 4 [353] 353 root68Tail3 winner =
    scan edge (search edge 3) 4 [353] 353 root68Tail4 winner := by
  rw [root68Tail3, scan, if_neg (by decide), root68Candidates3_checked]
  change scan edge (search edge 3) 4 [353] 353 root68Tail4
    (if 3 ≤ root68Candidates3.length then search edge 3 [353, 1433] root68Candidates3 1786 winner else winner) = _
  rw [if_neg (by decide)]
def root68Candidates4 : List Nat := [10427, 23741]
set_option maxRecDepth 8192 in
theorem root68Candidates4_checked : candidates edge 4 353 1481 root68Tail5 winner.bound = root68Candidates4 := by
  unfold candidates
  have row : edge 1481 = (SortedNatTree.contains FastRows.row1481) := by
    change Data.row1481.contains = (SortedNatTree.contains FastRows.row1481)
    exact FastRows.row1481_checked.symm
  rw [row]
  decide
theorem root68Step4 : scan edge (search edge 3) 4 [353] 353 root68Tail4 winner =
    scan edge (search edge 3) 4 [353] 353 root68Tail5 winner := by
  rw [root68Tail4, scan, if_neg (by decide), root68Candidates4_checked]
  change scan edge (search edge 3) 4 [353] 353 root68Tail5
    (if 3 ≤ root68Candidates4.length then search edge 3 [353, 1481] root68Candidates4 1834 winner else winner) = _
  rw [if_neg (by decide)]
def root68Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root68Candidates5_checked : candidates edge 4 353 1499 root68Tail6 winner.bound = root68Candidates5 := by
  unfold candidates
  have row : edge 1499 = (SortedNatTree.contains FastRows.row1499) := by
    change Data.row1499.contains = (SortedNatTree.contains FastRows.row1499)
    exact FastRows.row1499_checked.symm
  rw [row]
  decide
theorem root68Step5 : scan edge (search edge 3) 4 [353] 353 root68Tail5 winner =
    scan edge (search edge 3) 4 [353] 353 root68Tail6 winner := by
  rw [root68Tail5, scan, if_neg (by decide), root68Candidates5_checked]
  change scan edge (search edge 3) 4 [353] 353 root68Tail6
    (if 3 ≤ root68Candidates5.length then search edge 3 [353, 1499] root68Candidates5 1852 winner else winner) = _
  rw [if_neg (by decide)]
def root68Candidates6 : List Nat := [10739]
set_option maxRecDepth 8192 in
theorem root68Candidates6_checked : candidates edge 4 353 1877 root68Tail7 winner.bound = root68Candidates6 := by
  unfold candidates
  have row : edge 1877 = (SortedNatTree.contains FastRows.row1877) := by
    change Data.row1877.contains = (SortedNatTree.contains FastRows.row1877)
    exact FastRows.row1877_checked.symm
  rw [row]
  decide
theorem root68Step6 : scan edge (search edge 3) 4 [353] 353 root68Tail6 winner =
    scan edge (search edge 3) 4 [353] 353 root68Tail7 winner := by
  rw [root68Tail6, scan, if_neg (by decide), root68Candidates6_checked]
  change scan edge (search edge 3) 4 [353] 353 root68Tail7
    (if 3 ≤ root68Candidates6.length then search edge 3 [353, 1877] root68Candidates6 2230 winner else winner) = _
  rw [if_neg (by decide)]
def root68Candidates7 : List Nat := [14519]
set_option maxRecDepth 8192 in
theorem root68Candidates7_checked : candidates edge 4 353 2213 root68Tail8 winner.bound = root68Candidates7 := by
  unfold candidates
  have row : edge 2213 = (SortedNatTree.contains FastRows.row2213) := by
    change Data.row2213.contains = (SortedNatTree.contains FastRows.row2213)
    exact FastRows.row2213_checked.symm
  rw [row]
  decide
theorem root68Step7 : scan edge (search edge 3) 4 [353] 353 root68Tail7 winner =
    scan edge (search edge 3) 4 [353] 353 root68Tail8 winner := by
  rw [root68Tail7, scan, if_neg (by decide), root68Candidates7_checked]
  change scan edge (search edge 3) 4 [353] 353 root68Tail8
    (if 3 ≤ root68Candidates7.length then search edge 3 [353, 2213] root68Candidates7 2566 winner else winner) = _
  rw [if_neg (by decide)]
def root68Candidates8 : List Nat := [16493, 17291]
set_option maxRecDepth 8192 in
theorem root68Candidates8_checked : candidates edge 4 353 2687 root68Tail9 winner.bound = root68Candidates8 := by
  unfold candidates
  have row : edge 2687 = (SortedNatTree.contains FastRows.row2687) := by
    change Data.row2687.contains = (SortedNatTree.contains FastRows.row2687)
    exact FastRows.row2687_checked.symm
  rw [row]
  decide
theorem root68Step8 : scan edge (search edge 3) 4 [353] 353 root68Tail8 winner =
    scan edge (search edge 3) 4 [353] 353 root68Tail9 winner := by
  rw [root68Tail8, scan, if_neg (by decide), root68Candidates8_checked]
  change scan edge (search edge 3) 4 [353] 353 root68Tail9
    (if 3 ≤ root68Candidates8.length then search edge 3 [353, 2687] root68Candidates8 3040 winner else winner) = _
  rw [if_neg (by decide)]
def root68Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root68Candidates9_checked : candidates edge 4 353 2741 root68Tail10 winner.bound = root68Candidates9 := by
  unfold candidates
  have row : edge 2741 = (SortedNatTree.contains FastRows.row2741) := by
    change Data.row2741.contains = (SortedNatTree.contains FastRows.row2741)
    exact FastRows.row2741_checked.symm
  rw [row]
  decide
theorem root68Step9 : scan edge (search edge 3) 4 [353] 353 root68Tail9 winner =
    scan edge (search edge 3) 4 [353] 353 root68Tail10 winner := by
  rw [root68Tail9, scan, if_neg (by decide), root68Candidates9_checked]
  change scan edge (search edge 3) 4 [353] 353 root68Tail10
    (if 3 ≤ root68Candidates9.length then search edge 3 [353, 2741] root68Candidates9 3094 winner else winner) = _
  rw [if_neg (by decide)]
def root68Candidates10 : List Nat := [8219]
set_option maxRecDepth 8192 in
theorem root68Candidates10_checked : candidates edge 4 353 3203 root68Tail11 winner.bound = root68Candidates10 := by
  unfold candidates
  have row : edge 3203 = (SortedNatTree.contains FastRows.row3203) := by
    change Data.row3203.contains = (SortedNatTree.contains FastRows.row3203)
    exact FastRows.row3203_checked.symm
  rw [row]
  decide
theorem root68Step10 : scan edge (search edge 3) 4 [353] 353 root68Tail10 winner =
    scan edge (search edge 3) 4 [353] 353 root68Tail11 winner := by
  rw [root68Tail10, scan, if_neg (by decide), root68Candidates10_checked]
  change scan edge (search edge 3) 4 [353] 353 root68Tail11
    (if 3 ≤ root68Candidates10.length then search edge 3 [353, 3203] root68Candidates10 3556 winner else winner) = _
  rw [if_neg (by decide)]
def root68Candidates11 : List Nat := [5849, 17609, 20963]
set_option maxRecDepth 8192 in
theorem root68Candidates11_checked : candidates edge 4 353 3371 root68Tail12 winner.bound = root68Candidates11 := by
  unfold candidates
  have row : edge 3371 = (SortedNatTree.contains FastRows.row3371) := by
    change Data.row3371.contains = (SortedNatTree.contains FastRows.row3371)
    exact FastRows.row3371_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root68Child11 : search edge 3 [353, 3371] root68Candidates11 3724 winner = winner := by decide
theorem root68Step11 : scan edge (search edge 3) 4 [353] 353 root68Tail11 winner =
    scan edge (search edge 3) 4 [353] 353 root68Tail12 winner := by
  rw [root68Tail11, scan, if_neg (by decide), root68Candidates11_checked]
  change scan edge (search edge 3) 4 [353] 353 root68Tail12
    (if 3 ≤ root68Candidates11.length then search edge 3 [353, 3371] root68Candidates11 3724 winner else winner) = _
  rw [if_pos (by decide), root68Child11]
def root68Candidates12 : List Nat := [4493, 4919, 9623]
set_option maxRecDepth 8192 in
theorem root68Candidates12_checked : candidates edge 4 353 3701 root68Tail13 winner.bound = root68Candidates12 := by
  unfold candidates
  have row : edge 3701 = (SortedNatTree.contains FastRows.row3701) := by
    change Data.row3701.contains = (SortedNatTree.contains FastRows.row3701)
    exact FastRows.row3701_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root68Child12 : search edge 3 [353, 3701] root68Candidates12 4054 winner = winner := by decide
theorem root68Step12 : scan edge (search edge 3) 4 [353] 353 root68Tail12 winner =
    scan edge (search edge 3) 4 [353] 353 root68Tail13 winner := by
  rw [root68Tail12, scan, if_neg (by decide), root68Candidates12_checked]
  change scan edge (search edge 3) 4 [353] 353 root68Tail13
    (if 3 ≤ root68Candidates12.length then search edge 3 [353, 3701] root68Candidates12 4054 winner else winner) = _
  rw [if_pos (by decide), root68Child12]
def root68Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root68Candidates13_checked : candidates edge 4 353 3779 root68Tail14 winner.bound = root68Candidates13 := by
  unfold candidates
  have row : edge 3779 = (SortedNatTree.contains FastRows.row3779) := by
    change Data.row3779.contains = (SortedNatTree.contains FastRows.row3779)
    exact FastRows.row3779_checked.symm
  rw [row]
  decide
theorem root68Step13 : scan edge (search edge 3) 4 [353] 353 root68Tail13 winner =
    scan edge (search edge 3) 4 [353] 353 root68Tail14 winner := by
  rw [root68Tail13, scan, if_neg (by decide), root68Candidates13_checked]
  change scan edge (search edge 3) 4 [353] 353 root68Tail14
    (if 3 ≤ root68Candidates13.length then search edge 3 [353, 3779] root68Candidates13 4132 winner else winner) = _
  rw [if_neg (by decide)]
def root68Candidates14 : List Nat := [11321, 12269]
set_option maxRecDepth 8192 in
theorem root68Candidates14_checked : candidates edge 4 353 4337 root68Tail15 winner.bound = root68Candidates14 := by
  unfold candidates
  have row : edge 4337 = (SortedNatTree.contains FastRows.row4337) := by
    change Data.row4337.contains = (SortedNatTree.contains FastRows.row4337)
    exact FastRows.row4337_checked.symm
  rw [row]
  decide
theorem root68Step14 : scan edge (search edge 3) 4 [353] 353 root68Tail14 winner =
    scan edge (search edge 3) 4 [353] 353 root68Tail15 winner := by
  rw [root68Tail14, scan, if_neg (by decide), root68Candidates14_checked]
  change scan edge (search edge 3) 4 [353] 353 root68Tail15
    (if 3 ≤ root68Candidates14.length then search edge 3 [353, 4337] root68Candidates14 4690 winner else winner) = _
  rw [if_neg (by decide)]
def root68Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root68Candidates15_checked : candidates edge 4 353 4397 root68Tail16 winner.bound = root68Candidates15 := by
  unfold candidates
  have row : edge 4397 = (SortedNatTree.contains FastRows.row4397) := by
    change Data.row4397.contains = (SortedNatTree.contains FastRows.row4397)
    exact FastRows.row4397_checked.symm
  rw [row]
  decide
theorem root68Step15 : scan edge (search edge 3) 4 [353] 353 root68Tail15 winner =
    scan edge (search edge 3) 4 [353] 353 root68Tail16 winner := by
  rw [root68Tail15, scan, if_neg (by decide), root68Candidates15_checked]
  change scan edge (search edge 3) 4 [353] 353 root68Tail16
    (if 3 ≤ root68Candidates15.length then search edge 3 [353, 4397] root68Candidates15 4750 winner else winner) = _
  rw [if_neg (by decide)]
def root68Candidates16 : List Nat := [10889]
set_option maxRecDepth 8192 in
theorem root68Candidates16_checked : candidates edge 4 353 4493 root68Tail17 winner.bound = root68Candidates16 := by
  unfold candidates
  have row : edge 4493 = (SortedNatTree.contains FastRows.row4493) := by
    change Data.row4493.contains = (SortedNatTree.contains FastRows.row4493)
    exact FastRows.row4493_checked.symm
  rw [row]
  decide
theorem root68Step16 : scan edge (search edge 3) 4 [353] 353 root68Tail16 winner =
    scan edge (search edge 3) 4 [353] 353 root68Tail17 winner := by
  rw [root68Tail16, scan, if_neg (by decide), root68Candidates16_checked]
  change scan edge (search edge 3) 4 [353] 353 root68Tail17
    (if 3 ≤ root68Candidates16.length then search edge 3 [353, 4493] root68Candidates16 4846 winner else winner) = _
  rw [if_neg (by decide)]
def root68Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root68Candidates17_checked : candidates edge 4 353 4673 root68Tail18 winner.bound = root68Candidates17 := by
  unfold candidates
  have row : edge 4673 = (SortedNatTree.contains FastRows.row4673) := by
    change Data.row4673.contains = (SortedNatTree.contains FastRows.row4673)
    exact FastRows.row4673_checked.symm
  rw [row]
  decide
theorem root68Step17 : scan edge (search edge 3) 4 [353] 353 root68Tail17 winner =
    scan edge (search edge 3) 4 [353] 353 root68Tail18 winner := by
  rw [root68Tail17, scan, if_neg (by decide), root68Candidates17_checked]
  change scan edge (search edge 3) 4 [353] 353 root68Tail18
    (if 3 ≤ root68Candidates17.length then search edge 3 [353, 4673] root68Candidates17 5026 winner else winner) = _
  rw [if_neg (by decide)]
def root68Candidates18 : List Nat := [7559, 10739]
set_option maxRecDepth 8192 in
theorem root68Candidates18_checked : candidates edge 4 353 4919 root68Tail19 winner.bound = root68Candidates18 := by
  unfold candidates
  have row : edge 4919 = (SortedNatTree.contains FastRows.row4919) := by
    change Data.row4919.contains = (SortedNatTree.contains FastRows.row4919)
    exact FastRows.row4919_checked.symm
  rw [row]
  decide
theorem root68Step18 : scan edge (search edge 3) 4 [353] 353 root68Tail18 winner =
    scan edge (search edge 3) 4 [353] 353 root68Tail19 winner := by
  rw [root68Tail18, scan, if_neg (by decide), root68Candidates18_checked]
  change scan edge (search edge 3) 4 [353] 353 root68Tail19
    (if 3 ≤ root68Candidates18.length then search edge 3 [353, 4919] root68Candidates18 5272 winner else winner) = _
  rw [if_neg (by decide)]
def root68Candidates19 : List Nat := [5573, 5849, 8219]
set_option maxRecDepth 8192 in
theorem root68Candidates19_checked : candidates edge 4 353 4967 root68Tail20 winner.bound = root68Candidates19 := by
  unfold candidates
  have row : edge 4967 = (SortedNatTree.contains FastRows.row4967) := by
    change Data.row4967.contains = (SortedNatTree.contains FastRows.row4967)
    exact FastRows.row4967_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root68Child19 : search edge 3 [353, 4967] root68Candidates19 5320 winner = winner := by decide
theorem root68Step19 : scan edge (search edge 3) 4 [353] 353 root68Tail19 winner =
    scan edge (search edge 3) 4 [353] 353 root68Tail20 winner := by
  rw [root68Tail19, scan, if_neg (by decide), root68Candidates19_checked]
  change scan edge (search edge 3) 4 [353] 353 root68Tail20
    (if 3 ≤ root68Candidates19.length then search edge 3 [353, 4967] root68Candidates19 5320 winner else winner) = _
  rw [if_pos (by decide), root68Child19]
def root68Candidates20 : List Nat := [15149]
set_option maxRecDepth 8192 in
theorem root68Candidates20_checked : candidates edge 4 353 5087 root68Tail21 winner.bound = root68Candidates20 := by
  unfold candidates
  have row : edge 5087 = (SortedNatTree.contains FastRows.row5087) := by
    change Data.row5087.contains = (SortedNatTree.contains FastRows.row5087)
    exact FastRows.row5087_checked.symm
  rw [row]
  decide
theorem root68Step20 : scan edge (search edge 3) 4 [353] 353 root68Tail20 winner =
    scan edge (search edge 3) 4 [353] 353 root68Tail21 winner := by
  rw [root68Tail20, scan, if_neg (by decide), root68Candidates20_checked]
  change scan edge (search edge 3) 4 [353] 353 root68Tail21
    (if 3 ≤ root68Candidates20.length then search edge 3 [353, 5087] root68Candidates20 5440 winner else winner) = _
  rw [if_neg (by decide)]
def root68Candidates21 : List Nat := [8219, 12527]
set_option maxRecDepth 8192 in
theorem root68Candidates21_checked : candidates edge 4 353 5573 root68Tail22 winner.bound = root68Candidates21 := by
  unfold candidates
  have row : edge 5573 = (SortedNatTree.contains FastRows.row5573) := by
    change Data.row5573.contains = (SortedNatTree.contains FastRows.row5573)
    exact FastRows.row5573_checked.symm
  rw [row]
  decide
theorem root68Step21 : scan edge (search edge 3) 4 [353] 353 root68Tail21 winner =
    scan edge (search edge 3) 4 [353] 353 root68Tail22 winner := by
  rw [root68Tail21, scan, if_neg (by decide), root68Candidates21_checked]
  change scan edge (search edge 3) 4 [353] 353 root68Tail22
    (if 3 ≤ root68Candidates21.length then search edge 3 [353, 5573] root68Candidates21 5926 winner else winner) = _
  rw [if_neg (by decide)]
def root68Candidates22 : List Nat := [10427, 12527]
set_option maxRecDepth 8192 in
theorem root68Candidates22_checked : candidates edge 4 353 5849 root68Tail23 winner.bound = root68Candidates22 := by
  unfold candidates
  have row : edge 5849 = (SortedNatTree.contains FastRows.row5849) := by
    change Data.row5849.contains = (SortedNatTree.contains FastRows.row5849)
    exact FastRows.row5849_checked.symm
  rw [row]
  decide
theorem root68Step22 : scan edge (search edge 3) 4 [353] 353 root68Tail22 winner =
    scan edge (search edge 3) 4 [353] 353 root68Tail23 winner := by
  rw [root68Tail22, scan, if_neg (by decide), root68Candidates22_checked]
  change scan edge (search edge 3) 4 [353] 353 root68Tail23
    (if 3 ≤ root68Candidates22.length then search edge 3 [353, 5849] root68Candidates22 6202 winner else winner) = _
  rw [if_neg (by decide)]
def root68Candidates23 : List Nat := []
set_option maxRecDepth 8192 in
theorem root68Candidates23_checked : candidates edge 4 353 5879 root68Tail24 winner.bound = root68Candidates23 := by
  unfold candidates
  have row : edge 5879 = (SortedNatTree.contains FastRows.row5879) := by
    change Data.row5879.contains = (SortedNatTree.contains FastRows.row5879)
    exact FastRows.row5879_checked.symm
  rw [row]
  decide
theorem root68Step23 : scan edge (search edge 3) 4 [353] 353 root68Tail23 winner =
    scan edge (search edge 3) 4 [353] 353 root68Tail24 winner := by
  rw [root68Tail23, scan, if_neg (by decide), root68Candidates23_checked]
  change scan edge (search edge 3) 4 [353] 353 root68Tail24
    (if 3 ≤ root68Candidates23.length then search edge 3 [353, 5879] root68Candidates23 6232 winner else winner) = _
  rw [if_neg (by decide)]
def root68Candidates24 : List Nat := [9059]
set_option maxRecDepth 8192 in
theorem root68Candidates24_checked : candidates edge 4 353 6131 root68Tail25 winner.bound = root68Candidates24 := by
  unfold candidates
  have row : edge 6131 = (SortedNatTree.contains FastRows.row6131) := by
    change Data.row6131.contains = (SortedNatTree.contains FastRows.row6131)
    exact FastRows.row6131_checked.symm
  rw [row]
  decide
theorem root68Step24 : scan edge (search edge 3) 4 [353] 353 root68Tail24 winner =
    scan edge (search edge 3) 4 [353] 353 root68Tail25 winner := by
  rw [root68Tail24, scan, if_neg (by decide), root68Candidates24_checked]
  change scan edge (search edge 3) 4 [353] 353 root68Tail25
    (if 3 ≤ root68Candidates24.length then search edge 3 [353, 6131] root68Candidates24 6484 winner else winner) = _
  rw [if_neg (by decide)]
def root68Candidates25 : List Nat := []
set_option maxRecDepth 8192 in
theorem root68Candidates25_checked : candidates edge 4 353 6389 root68Tail26 winner.bound = root68Candidates25 := by
  unfold candidates
  have row : edge 6389 = (SortedNatTree.contains FastRows.row6389) := by
    change Data.row6389.contains = (SortedNatTree.contains FastRows.row6389)
    exact FastRows.row6389_checked.symm
  rw [row]
  decide
theorem root68Step25 : scan edge (search edge 3) 4 [353] 353 root68Tail25 winner =
    scan edge (search edge 3) 4 [353] 353 root68Tail26 winner := by
  rw [root68Tail25, scan, if_neg (by decide), root68Candidates25_checked]
  change scan edge (search edge 3) 4 [353] 353 root68Tail26
    (if 3 ≤ root68Candidates25.length then search edge 3 [353, 6389] root68Candidates25 6742 winner else winner) = _
  rw [if_neg (by decide)]
theorem root68Scan26 : scan edge (search edge 3) 4 [353] 353 root68Tail26 winner = winner := by rw [root68Tail26, scan, if_pos (by decide)]
theorem root68Scan25 : scan edge (search edge 3) 4 [353] 353 root68Tail25 winner = winner :=
  (root68Step25).trans root68Scan26
theorem root68Scan24 : scan edge (search edge 3) 4 [353] 353 root68Tail24 winner = winner :=
  (root68Step24).trans root68Scan25
theorem root68Scan23 : scan edge (search edge 3) 4 [353] 353 root68Tail23 winner = winner :=
  (root68Step23).trans root68Scan24
theorem root68Scan22 : scan edge (search edge 3) 4 [353] 353 root68Tail22 winner = winner :=
  (root68Step22).trans root68Scan23
theorem root68Scan21 : scan edge (search edge 3) 4 [353] 353 root68Tail21 winner = winner :=
  (root68Step21).trans root68Scan22
theorem root68Scan20 : scan edge (search edge 3) 4 [353] 353 root68Tail20 winner = winner :=
  (root68Step20).trans root68Scan21
theorem root68Scan19 : scan edge (search edge 3) 4 [353] 353 root68Tail19 winner = winner :=
  (root68Step19).trans root68Scan20
theorem root68Scan18 : scan edge (search edge 3) 4 [353] 353 root68Tail18 winner = winner :=
  (root68Step18).trans root68Scan19
theorem root68Scan17 : scan edge (search edge 3) 4 [353] 353 root68Tail17 winner = winner :=
  (root68Step17).trans root68Scan18
theorem root68Scan16 : scan edge (search edge 3) 4 [353] 353 root68Tail16 winner = winner :=
  (root68Step16).trans root68Scan17
theorem root68Scan15 : scan edge (search edge 3) 4 [353] 353 root68Tail15 winner = winner :=
  (root68Step15).trans root68Scan16
theorem root68Scan14 : scan edge (search edge 3) 4 [353] 353 root68Tail14 winner = winner :=
  (root68Step14).trans root68Scan15
theorem root68Scan13 : scan edge (search edge 3) 4 [353] 353 root68Tail13 winner = winner :=
  (root68Step13).trans root68Scan14
theorem root68Scan12 : scan edge (search edge 3) 4 [353] 353 root68Tail12 winner = winner :=
  (root68Step12).trans root68Scan13
theorem root68Scan11 : scan edge (search edge 3) 4 [353] 353 root68Tail11 winner = winner :=
  (root68Step11).trans root68Scan12
theorem root68Scan10 : scan edge (search edge 3) 4 [353] 353 root68Tail10 winner = winner :=
  (root68Step10).trans root68Scan11
theorem root68Scan9 : scan edge (search edge 3) 4 [353] 353 root68Tail9 winner = winner :=
  (root68Step9).trans root68Scan10
theorem root68Scan8 : scan edge (search edge 3) 4 [353] 353 root68Tail8 winner = winner :=
  (root68Step8).trans root68Scan9
theorem root68Scan7 : scan edge (search edge 3) 4 [353] 353 root68Tail7 winner = winner :=
  (root68Step7).trans root68Scan8
theorem root68Scan6 : scan edge (search edge 3) 4 [353] 353 root68Tail6 winner = winner :=
  (root68Step6).trans root68Scan7
theorem root68Scan5 : scan edge (search edge 3) 4 [353] 353 root68Tail5 winner = winner :=
  (root68Step5).trans root68Scan6
theorem root68Scan4 : scan edge (search edge 3) 4 [353] 353 root68Tail4 winner = winner :=
  (root68Step4).trans root68Scan5
theorem root68Scan3 : scan edge (search edge 3) 4 [353] 353 root68Tail3 winner = winner :=
  (root68Step3).trans root68Scan4
theorem root68Scan2 : scan edge (search edge 3) 4 [353] 353 root68Tail2 winner = winner :=
  (root68Step2).trans root68Scan3
theorem root68Scan1 : scan edge (search edge 3) 4 [353] 353 root68Tail1 winner = winner :=
  (root68Step1).trans root68Scan2
theorem root68Scan0 : scan edge (search edge 3) 4 [353] 353 root68Tail0 winner = winner :=
  (root68Step0).trans root68Scan1
set_option maxRecDepth 4096 in
theorem root68Branch : search edge 4 [353] root68Candidates 353 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [353] 353 root68Tail0 winner = winner
  exact root68Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
