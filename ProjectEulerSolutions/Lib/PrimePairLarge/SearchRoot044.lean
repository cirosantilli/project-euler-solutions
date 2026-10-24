import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root44Candidates : List Nat := Data.row211
set_option maxRecDepth 8192 in
theorem root44Candidates_checked : candidates edge 5 0 211 tail45 winner.bound = root44Candidates := by
  apply candidates_eq_row edge 5 0 211 tail45 root44Candidates winner.bound tail45_ordered
  · decide
  · decide
  · funext q
    rfl
def root44Tail70 : List Nat := []
def root44Tail69 : List Nat := 25693 :: root44Tail70
def root44Tail68 : List Nat := 25633 :: root44Tail69
def root44Tail67 : List Nat := 25603 :: root44Tail68
def root44Tail66 : List Nat := 25303 :: root44Tail67
def root44Tail65 : List Nat := 25057 :: root44Tail66
def root44Tail64 : List Nat := 24967 :: root44Tail65
def root44Tail63 : List Nat := 24007 :: root44Tail64
def root44Tail62 : List Nat := 23761 :: root44Tail63
def root44Tail61 : List Nat := 23293 :: root44Tail62
def root44Tail60 : List Nat := 22921 :: root44Tail61
def root44Tail59 : List Nat := 22291 :: root44Tail60
def root44Tail58 : List Nat := 21319 :: root44Tail59
def root44Tail57 : List Nat := 21277 :: root44Tail58
def root44Tail56 : List Nat := 21187 :: root44Tail57
def root44Tail55 : List Nat := 20947 :: root44Tail56
def root44Tail54 : List Nat := 20431 :: root44Tail55
def root44Tail53 : List Nat := 20089 :: root44Tail54
def root44Tail52 : List Nat := 19381 :: root44Tail53
def root44Tail51 : List Nat := 19183 :: root44Tail52
def root44Tail50 : List Nat := 19081 :: root44Tail51
def root44Tail49 : List Nat := 18787 :: root44Tail50
def root44Tail48 : List Nat := 17881 :: root44Tail49
def root44Tail47 : List Nat := 17443 :: root44Tail48
def root44Tail46 : List Nat := 17359 :: root44Tail47
def root44Tail45 : List Nat := 17029 :: root44Tail46
def root44Tail44 : List Nat := 16699 :: root44Tail45
def root44Tail43 : List Nat := 16561 :: root44Tail44
def root44Tail42 : List Nat := 16477 :: root44Tail43
def root44Tail41 : List Nat := 16411 :: root44Tail42
def root44Tail40 : List Nat := 16063 :: root44Tail41
def root44Tail39 : List Nat := 15727 :: root44Tail40
def root44Tail38 : List Nat := 15439 :: root44Tail39
def root44Tail37 : List Nat := 15091 :: root44Tail38
def root44Tail36 : List Nat := 14683 :: root44Tail37
def root44Tail35 : List Nat := 14461 :: root44Tail36
def root44Tail34 : List Nat := 14341 :: root44Tail35
def root44Tail33 : List Nat := 13597 :: root44Tail34
def root44Tail32 : List Nat := 12043 :: root44Tail33
def root44Tail31 : List Nat := 10861 :: root44Tail32
def root44Tail30 : List Nat := 10513 :: root44Tail31
def root44Tail29 : List Nat := 10093 :: root44Tail30
def root44Tail28 : List Nat := 9907 :: root44Tail29
def root44Tail27 : List Nat := 8581 :: root44Tail28
def root44Tail26 : List Nat := 8269 :: root44Tail27
def root44Tail25 : List Nat := 8209 :: root44Tail26
def root44Tail24 : List Nat := 7039 :: root44Tail25
def root44Tail23 : List Nat := 6397 :: root44Tail24
def root44Tail22 : List Nat := 6199 :: root44Tail23
def root44Tail21 : List Nat := 5791 :: root44Tail22
def root44Tail20 : List Nat := 5653 :: root44Tail21
def root44Tail19 : List Nat := 5101 :: root44Tail20
def root44Tail18 : List Nat := 5059 :: root44Tail19
def root44Tail17 : List Nat := 4549 :: root44Tail18
def root44Tail16 : List Nat := 4507 :: root44Tail17
def root44Tail15 : List Nat := 4027 :: root44Tail16
def root44Tail14 : List Nat := 3733 :: root44Tail15
def root44Tail13 : List Nat := 2833 :: root44Tail14
def root44Tail12 : List Nat := 2683 :: root44Tail13
def root44Tail11 : List Nat := 2053 :: root44Tail12
def root44Tail10 : List Nat := 1873 :: root44Tail11
def root44Tail9 : List Nat := 1783 :: root44Tail10
def root44Tail8 : List Nat := 1129 :: root44Tail9
def root44Tail7 : List Nat := 727 :: root44Tail8
def root44Tail6 : List Nat := 571 :: root44Tail7
def root44Tail5 : List Nat := 499 :: root44Tail6
def root44Tail4 : List Nat := 373 :: root44Tail5
def root44Tail3 : List Nat := 349 :: root44Tail4
def root44Tail2 : List Nat := 313 :: root44Tail3
def root44Tail1 : List Nat := 283 :: root44Tail2
def root44Tail0 : List Nat := 271 :: root44Tail1
def root44Candidates0 : List Nat := [1873, 8269, 10513, 16561, 23293]
set_option maxRecDepth 8192 in
theorem root44Candidates0_checked : candidates edge 4 211 271 root44Tail1 winner.bound = root44Candidates0 := by
  unfold candidates
  have row : edge 271 = (SortedNatTree.contains FastRows.row271) := by
    change Data.row271.contains = (SortedNatTree.contains FastRows.row271)
    exact FastRows.row271_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root44Child0 : search edge 3 [211, 271] root44Candidates0 482 winner = winner := by decide
theorem root44Step0 : scan edge (search edge 3) 4 [211] 211 root44Tail0 winner =
    scan edge (search edge 3) 4 [211] 211 root44Tail1 winner := by
  rw [root44Tail0, scan, if_neg (by decide), root44Candidates0_checked]
  change scan edge (search edge 3) 4 [211] 211 root44Tail1
    (if 3 ≤ root44Candidates0.length then search edge 3 [211, 271] root44Candidates0 482 winner else winner) = _
  rw [if_pos (by decide), root44Child0]
def root44Candidates1 : List Nat := [5653, 6397, 7039, 16699]
set_option maxRecDepth 8192 in
theorem root44Candidates1_checked : candidates edge 4 211 283 root44Tail2 winner.bound = root44Candidates1 := by
  unfold candidates
  have row : edge 283 = (SortedNatTree.contains FastRows.row283) := by
    change Data.row283.contains = (SortedNatTree.contains FastRows.row283)
    exact FastRows.row283_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root44Child1 : search edge 3 [211, 283] root44Candidates1 494 winner = winner := by decide
theorem root44Step1 : scan edge (search edge 3) 4 [211] 211 root44Tail1 winner =
    scan edge (search edge 3) 4 [211] 211 root44Tail2 winner := by
  rw [root44Tail1, scan, if_neg (by decide), root44Candidates1_checked]
  change scan edge (search edge 3) 4 [211] 211 root44Tail2
    (if 3 ≤ root44Candidates1.length then search edge 3 [211, 283] root44Candidates1 494 winner else winner) = _
  rw [if_pos (by decide), root44Child1]
def root44Candidates2 : List Nat := [727, 3733, 14683, 16699]
set_option maxRecDepth 8192 in
theorem root44Candidates2_checked : candidates edge 4 211 313 root44Tail3 winner.bound = root44Candidates2 := by
  unfold candidates
  have row : edge 313 = (SortedNatTree.contains FastRows.row313) := by
    change Data.row313.contains = (SortedNatTree.contains FastRows.row313)
    exact FastRows.row313_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root44Child2 : search edge 3 [211, 313] root44Candidates2 524 winner = winner := by decide
theorem root44Step2 : scan edge (search edge 3) 4 [211] 211 root44Tail2 winner =
    scan edge (search edge 3) 4 [211] 211 root44Tail3 winner := by
  rw [root44Tail2, scan, if_neg (by decide), root44Candidates2_checked]
  change scan edge (search edge 3) 4 [211] 211 root44Tail3
    (if 3 ≤ root44Candidates2.length then search edge 3 [211, 313] root44Candidates2 524 winner else winner) = _
  rw [if_pos (by decide), root44Child2]
def root44Candidates3 : List Nat := [373, 499, 25057]
set_option maxRecDepth 8192 in
theorem root44Candidates3_checked : candidates edge 4 211 349 root44Tail4 winner.bound = root44Candidates3 := by
  unfold candidates
  have row : edge 349 = (SortedNatTree.contains FastRows.row349) := by
    change Data.row349.contains = (SortedNatTree.contains FastRows.row349)
    exact FastRows.row349_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root44Child3 : search edge 3 [211, 349] root44Candidates3 560 winner = winner := by decide
theorem root44Step3 : scan edge (search edge 3) 4 [211] 211 root44Tail3 winner =
    scan edge (search edge 3) 4 [211] 211 root44Tail4 winner := by
  rw [root44Tail3, scan, if_neg (by decide), root44Candidates3_checked]
  change scan edge (search edge 3) 4 [211] 211 root44Tail4
    (if 3 ≤ root44Candidates3.length then search edge 3 [211, 349] root44Candidates3 560 winner else winner) = _
  rw [if_pos (by decide), root44Child3]
def root44Candidates4 : List Nat := [5101, 14341, 17443]
set_option maxRecDepth 8192 in
theorem root44Candidates4_checked : candidates edge 4 211 373 root44Tail5 winner.bound = root44Candidates4 := by
  unfold candidates
  have row : edge 373 = (SortedNatTree.contains FastRows.row373) := by
    change Data.row373.contains = (SortedNatTree.contains FastRows.row373)
    exact FastRows.row373_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root44Child4 : search edge 3 [211, 373] root44Candidates4 584 winner = winner := by decide
theorem root44Step4 : scan edge (search edge 3) 4 [211] 211 root44Tail4 winner =
    scan edge (search edge 3) 4 [211] 211 root44Tail5 winner := by
  rw [root44Tail4, scan, if_neg (by decide), root44Candidates4_checked]
  change scan edge (search edge 3) 4 [211] 211 root44Tail5
    (if 3 ≤ root44Candidates4.length then search edge 3 [211, 373] root44Candidates4 584 winner else winner) = _
  rw [if_pos (by decide), root44Child4]
def root44Candidates5 : List Nat := [15091, 20947]
set_option maxRecDepth 8192 in
theorem root44Candidates5_checked : candidates edge 4 211 499 root44Tail6 winner.bound = root44Candidates5 := by
  unfold candidates
  have row : edge 499 = (SortedNatTree.contains FastRows.row499) := by
    change Data.row499.contains = (SortedNatTree.contains FastRows.row499)
    exact FastRows.row499_checked.symm
  rw [row]
  decide
theorem root44Step5 : scan edge (search edge 3) 4 [211] 211 root44Tail5 winner =
    scan edge (search edge 3) 4 [211] 211 root44Tail6 winner := by
  rw [root44Tail5, scan, if_neg (by decide), root44Candidates5_checked]
  change scan edge (search edge 3) 4 [211] 211 root44Tail6
    (if 3 ≤ root44Candidates5.length then search edge 3 [211, 499] root44Candidates5 710 winner else winner) = _
  rw [if_neg (by decide)]
def root44Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root44Candidates6_checked : candidates edge 4 211 571 root44Tail7 winner.bound = root44Candidates6 := by
  unfold candidates
  have row : edge 571 = (SortedNatTree.contains FastRows.row571) := by
    change Data.row571.contains = (SortedNatTree.contains FastRows.row571)
    exact FastRows.row571_checked.symm
  rw [row]
  decide
theorem root44Step6 : scan edge (search edge 3) 4 [211] 211 root44Tail6 winner =
    scan edge (search edge 3) 4 [211] 211 root44Tail7 winner := by
  rw [root44Tail6, scan, if_neg (by decide), root44Candidates6_checked]
  change scan edge (search edge 3) 4 [211] 211 root44Tail7
    (if 3 ≤ root44Candidates6.length then search edge 3 [211, 571] root44Candidates6 782 winner else winner) = _
  rw [if_neg (by decide)]
def root44Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root44Candidates7_checked : candidates edge 4 211 727 root44Tail8 winner.bound = root44Candidates7 := by
  unfold candidates
  have row : edge 727 = (SortedNatTree.contains FastRows.row727) := by
    change Data.row727.contains = (SortedNatTree.contains FastRows.row727)
    exact FastRows.row727_checked.symm
  rw [row]
  decide
theorem root44Step7 : scan edge (search edge 3) 4 [211] 211 root44Tail7 winner =
    scan edge (search edge 3) 4 [211] 211 root44Tail8 winner := by
  rw [root44Tail7, scan, if_neg (by decide), root44Candidates7_checked]
  change scan edge (search edge 3) 4 [211] 211 root44Tail8
    (if 3 ≤ root44Candidates7.length then search edge 3 [211, 727] root44Candidates7 938 winner else winner) = _
  rw [if_neg (by decide)]
def root44Candidates8 : List Nat := [16561, 19381]
set_option maxRecDepth 8192 in
theorem root44Candidates8_checked : candidates edge 4 211 1129 root44Tail9 winner.bound = root44Candidates8 := by
  unfold candidates
  have row : edge 1129 = (SortedNatTree.contains FastRows.row1129) := by
    change Data.row1129.contains = (SortedNatTree.contains FastRows.row1129)
    exact FastRows.row1129_checked.symm
  rw [row]
  decide
theorem root44Step8 : scan edge (search edge 3) 4 [211] 211 root44Tail8 winner =
    scan edge (search edge 3) 4 [211] 211 root44Tail9 winner := by
  rw [root44Tail8, scan, if_neg (by decide), root44Candidates8_checked]
  change scan edge (search edge 3) 4 [211] 211 root44Tail9
    (if 3 ≤ root44Candidates8.length then search edge 3 [211, 1129] root44Candidates8 1340 winner else winner) = _
  rw [if_neg (by decide)]
def root44Candidates9 : List Nat := [2053, 8209, 14341, 15727, 17443]
set_option maxRecDepth 8192 in
theorem root44Candidates9_checked : candidates edge 4 211 1783 root44Tail10 winner.bound = root44Candidates9 := by
  unfold candidates
  have row : edge 1783 = (SortedNatTree.contains FastRows.row1783) := by
    change Data.row1783.contains = (SortedNatTree.contains FastRows.row1783)
    exact FastRows.row1783_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root44Child9 : search edge 3 [211, 1783] root44Candidates9 1994 winner = winner := by decide
theorem root44Step9 : scan edge (search edge 3) 4 [211] 211 root44Tail9 winner =
    scan edge (search edge 3) 4 [211] 211 root44Tail10 winner := by
  rw [root44Tail9, scan, if_neg (by decide), root44Candidates9_checked]
  change scan edge (search edge 3) 4 [211] 211 root44Tail10
    (if 3 ≤ root44Candidates9.length then search edge 3 [211, 1783] root44Candidates9 1994 winner else winner) = _
  rw [if_pos (by decide), root44Child9]
def root44Candidates10 : List Nat := [9907]
set_option maxRecDepth 8192 in
theorem root44Candidates10_checked : candidates edge 4 211 1873 root44Tail11 winner.bound = root44Candidates10 := by
  unfold candidates
  have row : edge 1873 = (SortedNatTree.contains FastRows.row1873) := by
    change Data.row1873.contains = (SortedNatTree.contains FastRows.row1873)
    exact FastRows.row1873_checked.symm
  rw [row]
  decide
theorem root44Step10 : scan edge (search edge 3) 4 [211] 211 root44Tail10 winner =
    scan edge (search edge 3) 4 [211] 211 root44Tail11 winner := by
  rw [root44Tail10, scan, if_neg (by decide), root44Candidates10_checked]
  change scan edge (search edge 3) 4 [211] 211 root44Tail11
    (if 3 ≤ root44Candidates10.length then search edge 3 [211, 1873] root44Candidates10 2084 winner else winner) = _
  rw [if_neg (by decide)]
def root44Candidates11 : List Nat := [5653, 18787]
set_option maxRecDepth 8192 in
theorem root44Candidates11_checked : candidates edge 4 211 2053 root44Tail12 winner.bound = root44Candidates11 := by
  unfold candidates
  have row : edge 2053 = (SortedNatTree.contains FastRows.row2053) := by
    change Data.row2053.contains = (SortedNatTree.contains FastRows.row2053)
    exact FastRows.row2053_checked.symm
  rw [row]
  decide
theorem root44Step11 : scan edge (search edge 3) 4 [211] 211 root44Tail11 winner =
    scan edge (search edge 3) 4 [211] 211 root44Tail12 winner := by
  rw [root44Tail11, scan, if_neg (by decide), root44Candidates11_checked]
  change scan edge (search edge 3) 4 [211] 211 root44Tail12
    (if 3 ≤ root44Candidates11.length then search edge 3 [211, 2053] root44Candidates11 2264 winner else winner) = _
  rw [if_neg (by decide)]
def root44Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root44Candidates12_checked : candidates edge 4 211 2683 root44Tail13 winner.bound = root44Candidates12 := by
  unfold candidates
  have row : edge 2683 = (SortedNatTree.contains FastRows.row2683) := by
    change Data.row2683.contains = (SortedNatTree.contains FastRows.row2683)
    exact FastRows.row2683_checked.symm
  rw [row]
  decide
theorem root44Step12 : scan edge (search edge 3) 4 [211] 211 root44Tail12 winner =
    scan edge (search edge 3) 4 [211] 211 root44Tail13 winner := by
  rw [root44Tail12, scan, if_neg (by decide), root44Candidates12_checked]
  change scan edge (search edge 3) 4 [211] 211 root44Tail13
    (if 3 ≤ root44Candidates12.length then search edge 3 [211, 2683] root44Candidates12 2894 winner else winner) = _
  rw [if_neg (by decide)]
def root44Candidates13 : List Nat := [8581]
set_option maxRecDepth 8192 in
theorem root44Candidates13_checked : candidates edge 4 211 2833 root44Tail14 winner.bound = root44Candidates13 := by
  unfold candidates
  have row : edge 2833 = (SortedNatTree.contains FastRows.row2833) := by
    change Data.row2833.contains = (SortedNatTree.contains FastRows.row2833)
    exact FastRows.row2833_checked.symm
  rw [row]
  decide
theorem root44Step13 : scan edge (search edge 3) 4 [211] 211 root44Tail13 winner =
    scan edge (search edge 3) 4 [211] 211 root44Tail14 winner := by
  rw [root44Tail13, scan, if_neg (by decide), root44Candidates13_checked]
  change scan edge (search edge 3) 4 [211] 211 root44Tail14
    (if 3 ≤ root44Candidates13.length then search edge 3 [211, 2833] root44Candidates13 3044 winner else winner) = _
  rw [if_neg (by decide)]
def root44Candidates14 : List Nat := [10513, 15091]
set_option maxRecDepth 8192 in
theorem root44Candidates14_checked : candidates edge 4 211 3733 root44Tail15 winner.bound = root44Candidates14 := by
  unfold candidates
  have row : edge 3733 = (SortedNatTree.contains FastRows.row3733) := by
    change Data.row3733.contains = (SortedNatTree.contains FastRows.row3733)
    exact FastRows.row3733_checked.symm
  rw [row]
  decide
theorem root44Step14 : scan edge (search edge 3) 4 [211] 211 root44Tail14 winner =
    scan edge (search edge 3) 4 [211] 211 root44Tail15 winner := by
  rw [root44Tail14, scan, if_neg (by decide), root44Candidates14_checked]
  change scan edge (search edge 3) 4 [211] 211 root44Tail15
    (if 3 ≤ root44Candidates14.length then search edge 3 [211, 3733] root44Candidates14 3944 winner else winner) = _
  rw [if_neg (by decide)]
def root44Candidates15 : List Nat := [6199, 14461, 15727]
set_option maxRecDepth 8192 in
theorem root44Candidates15_checked : candidates edge 4 211 4027 root44Tail16 winner.bound = root44Candidates15 := by
  unfold candidates
  have row : edge 4027 = (SortedNatTree.contains FastRows.row4027) := by
    change Data.row4027.contains = (SortedNatTree.contains FastRows.row4027)
    exact FastRows.row4027_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root44Child15 : search edge 3 [211, 4027] root44Candidates15 4238 winner = winner := by decide
theorem root44Step15 : scan edge (search edge 3) 4 [211] 211 root44Tail15 winner =
    scan edge (search edge 3) 4 [211] 211 root44Tail16 winner := by
  rw [root44Tail15, scan, if_neg (by decide), root44Candidates15_checked]
  change scan edge (search edge 3) 4 [211] 211 root44Tail16
    (if 3 ≤ root44Candidates15.length then search edge 3 [211, 4027] root44Candidates15 4238 winner else winner) = _
  rw [if_pos (by decide), root44Child15]
def root44Candidates16 : List Nat := [15439]
set_option maxRecDepth 8192 in
theorem root44Candidates16_checked : candidates edge 4 211 4507 root44Tail17 winner.bound = root44Candidates16 := by
  unfold candidates
  have row : edge 4507 = (SortedNatTree.contains FastRows.row4507) := by
    change Data.row4507.contains = (SortedNatTree.contains FastRows.row4507)
    exact FastRows.row4507_checked.symm
  rw [row]
  decide
theorem root44Step16 : scan edge (search edge 3) 4 [211] 211 root44Tail16 winner =
    scan edge (search edge 3) 4 [211] 211 root44Tail17 winner := by
  rw [root44Tail16, scan, if_neg (by decide), root44Candidates16_checked]
  change scan edge (search edge 3) 4 [211] 211 root44Tail17
    (if 3 ≤ root44Candidates16.length then search edge 3 [211, 4507] root44Candidates16 4718 winner else winner) = _
  rw [if_neg (by decide)]
def root44Candidates17 : List Nat := [14461]
set_option maxRecDepth 8192 in
theorem root44Candidates17_checked : candidates edge 4 211 4549 root44Tail18 winner.bound = root44Candidates17 := by
  unfold candidates
  have row : edge 4549 = (SortedNatTree.contains FastRows.row4549) := by
    change Data.row4549.contains = (SortedNatTree.contains FastRows.row4549)
    exact FastRows.row4549_checked.symm
  rw [row]
  decide
theorem root44Step17 : scan edge (search edge 3) 4 [211] 211 root44Tail17 winner =
    scan edge (search edge 3) 4 [211] 211 root44Tail18 winner := by
  rw [root44Tail17, scan, if_neg (by decide), root44Candidates17_checked]
  change scan edge (search edge 3) 4 [211] 211 root44Tail18
    (if 3 ≤ root44Candidates17.length then search edge 3 [211, 4549] root44Candidates17 4760 winner else winner) = _
  rw [if_neg (by decide)]
def root44Candidates18 : List Nat := [8269]
set_option maxRecDepth 8192 in
theorem root44Candidates18_checked : candidates edge 4 211 5059 root44Tail19 winner.bound = root44Candidates18 := by
  unfold candidates
  have row : edge 5059 = (SortedNatTree.contains FastRows.row5059) := by
    change Data.row5059.contains = (SortedNatTree.contains FastRows.row5059)
    exact FastRows.row5059_checked.symm
  rw [row]
  decide
theorem root44Step18 : scan edge (search edge 3) 4 [211] 211 root44Tail18 winner =
    scan edge (search edge 3) 4 [211] 211 root44Tail19 winner := by
  rw [root44Tail18, scan, if_neg (by decide), root44Candidates18_checked]
  change scan edge (search edge 3) 4 [211] 211 root44Tail19
    (if 3 ≤ root44Candidates18.length then search edge 3 [211, 5059] root44Candidates18 5270 winner else winner) = _
  rw [if_neg (by decide)]
def root44Candidates19 : List Nat := [6397, 16561, 17359]
set_option maxRecDepth 8192 in
theorem root44Candidates19_checked : candidates edge 4 211 5101 root44Tail20 winner.bound = root44Candidates19 := by
  unfold candidates
  have row : edge 5101 = (SortedNatTree.contains FastRows.row5101) := by
    change Data.row5101.contains = (SortedNatTree.contains FastRows.row5101)
    exact FastRows.row5101_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root44Child19 : search edge 3 [211, 5101] root44Candidates19 5312 winner = winner := by decide
theorem root44Step19 : scan edge (search edge 3) 4 [211] 211 root44Tail19 winner =
    scan edge (search edge 3) 4 [211] 211 root44Tail20 winner := by
  rw [root44Tail19, scan, if_neg (by decide), root44Candidates19_checked]
  change scan edge (search edge 3) 4 [211] 211 root44Tail20
    (if 3 ≤ root44Candidates19.length then search edge 3 [211, 5101] root44Candidates19 5312 winner else winner) = _
  rw [if_pos (by decide), root44Child19]
def root44Candidates20 : List Nat := [8269]
set_option maxRecDepth 8192 in
theorem root44Candidates20_checked : candidates edge 4 211 5653 root44Tail21 winner.bound = root44Candidates20 := by
  unfold candidates
  have row : edge 5653 = (SortedNatTree.contains FastRows.row5653) := by
    change Data.row5653.contains = (SortedNatTree.contains FastRows.row5653)
    exact FastRows.row5653_checked.symm
  rw [row]
  decide
theorem root44Step20 : scan edge (search edge 3) 4 [211] 211 root44Tail20 winner =
    scan edge (search edge 3) 4 [211] 211 root44Tail21 winner := by
  rw [root44Tail20, scan, if_neg (by decide), root44Candidates20_checked]
  change scan edge (search edge 3) 4 [211] 211 root44Tail21
    (if 3 ≤ root44Candidates20.length then search edge 3 [211, 5653] root44Candidates20 5864 winner else winner) = _
  rw [if_neg (by decide)]
def root44Candidates21 : List Nat := [14461]
set_option maxRecDepth 8192 in
theorem root44Candidates21_checked : candidates edge 4 211 5791 root44Tail22 winner.bound = root44Candidates21 := by
  unfold candidates
  have row : edge 5791 = (SortedNatTree.contains FastRows.row5791) := by
    change Data.row5791.contains = (SortedNatTree.contains FastRows.row5791)
    exact FastRows.row5791_checked.symm
  rw [row]
  decide
theorem root44Step21 : scan edge (search edge 3) 4 [211] 211 root44Tail21 winner =
    scan edge (search edge 3) 4 [211] 211 root44Tail22 winner := by
  rw [root44Tail21, scan, if_neg (by decide), root44Candidates21_checked]
  change scan edge (search edge 3) 4 [211] 211 root44Tail22
    (if 3 ≤ root44Candidates21.length then search edge 3 [211, 5791] root44Candidates21 6002 winner else winner) = _
  rw [if_neg (by decide)]
def root44Candidates22 : List Nat := []
set_option maxRecDepth 8192 in
theorem root44Candidates22_checked : candidates edge 4 211 6199 root44Tail23 winner.bound = root44Candidates22 := by
  unfold candidates
  have row : edge 6199 = (SortedNatTree.contains FastRows.row6199) := by
    change Data.row6199.contains = (SortedNatTree.contains FastRows.row6199)
    exact FastRows.row6199_checked.symm
  rw [row]
  decide
theorem root44Step22 : scan edge (search edge 3) 4 [211] 211 root44Tail22 winner =
    scan edge (search edge 3) 4 [211] 211 root44Tail23 winner := by
  rw [root44Tail22, scan, if_neg (by decide), root44Candidates22_checked]
  change scan edge (search edge 3) 4 [211] 211 root44Tail23
    (if 3 ≤ root44Candidates22.length then search edge 3 [211, 6199] root44Candidates22 6410 winner else winner) = _
  rw [if_neg (by decide)]
def root44Candidates23 : List Nat := [14461]
set_option maxRecDepth 8192 in
theorem root44Candidates23_checked : candidates edge 4 211 6397 root44Tail24 winner.bound = root44Candidates23 := by
  unfold candidates
  have row : edge 6397 = (SortedNatTree.contains FastRows.row6397) := by
    change Data.row6397.contains = (SortedNatTree.contains FastRows.row6397)
    exact FastRows.row6397_checked.symm
  rw [row]
  decide
theorem root44Step23 : scan edge (search edge 3) 4 [211] 211 root44Tail23 winner =
    scan edge (search edge 3) 4 [211] 211 root44Tail24 winner := by
  rw [root44Tail23, scan, if_neg (by decide), root44Candidates23_checked]
  change scan edge (search edge 3) 4 [211] 211 root44Tail24
    (if 3 ≤ root44Candidates23.length then search edge 3 [211, 6397] root44Candidates23 6608 winner else winner) = _
  rw [if_neg (by decide)]
theorem root44Scan24 : scan edge (search edge 3) 4 [211] 211 root44Tail24 winner = winner := by rw [root44Tail24, scan, if_pos (by decide)]
theorem root44Scan23 : scan edge (search edge 3) 4 [211] 211 root44Tail23 winner = winner :=
  (root44Step23).trans root44Scan24
theorem root44Scan22 : scan edge (search edge 3) 4 [211] 211 root44Tail22 winner = winner :=
  (root44Step22).trans root44Scan23
theorem root44Scan21 : scan edge (search edge 3) 4 [211] 211 root44Tail21 winner = winner :=
  (root44Step21).trans root44Scan22
theorem root44Scan20 : scan edge (search edge 3) 4 [211] 211 root44Tail20 winner = winner :=
  (root44Step20).trans root44Scan21
theorem root44Scan19 : scan edge (search edge 3) 4 [211] 211 root44Tail19 winner = winner :=
  (root44Step19).trans root44Scan20
theorem root44Scan18 : scan edge (search edge 3) 4 [211] 211 root44Tail18 winner = winner :=
  (root44Step18).trans root44Scan19
theorem root44Scan17 : scan edge (search edge 3) 4 [211] 211 root44Tail17 winner = winner :=
  (root44Step17).trans root44Scan18
theorem root44Scan16 : scan edge (search edge 3) 4 [211] 211 root44Tail16 winner = winner :=
  (root44Step16).trans root44Scan17
theorem root44Scan15 : scan edge (search edge 3) 4 [211] 211 root44Tail15 winner = winner :=
  (root44Step15).trans root44Scan16
theorem root44Scan14 : scan edge (search edge 3) 4 [211] 211 root44Tail14 winner = winner :=
  (root44Step14).trans root44Scan15
theorem root44Scan13 : scan edge (search edge 3) 4 [211] 211 root44Tail13 winner = winner :=
  (root44Step13).trans root44Scan14
theorem root44Scan12 : scan edge (search edge 3) 4 [211] 211 root44Tail12 winner = winner :=
  (root44Step12).trans root44Scan13
theorem root44Scan11 : scan edge (search edge 3) 4 [211] 211 root44Tail11 winner = winner :=
  (root44Step11).trans root44Scan12
theorem root44Scan10 : scan edge (search edge 3) 4 [211] 211 root44Tail10 winner = winner :=
  (root44Step10).trans root44Scan11
theorem root44Scan9 : scan edge (search edge 3) 4 [211] 211 root44Tail9 winner = winner :=
  (root44Step9).trans root44Scan10
theorem root44Scan8 : scan edge (search edge 3) 4 [211] 211 root44Tail8 winner = winner :=
  (root44Step8).trans root44Scan9
theorem root44Scan7 : scan edge (search edge 3) 4 [211] 211 root44Tail7 winner = winner :=
  (root44Step7).trans root44Scan8
theorem root44Scan6 : scan edge (search edge 3) 4 [211] 211 root44Tail6 winner = winner :=
  (root44Step6).trans root44Scan7
theorem root44Scan5 : scan edge (search edge 3) 4 [211] 211 root44Tail5 winner = winner :=
  (root44Step5).trans root44Scan6
theorem root44Scan4 : scan edge (search edge 3) 4 [211] 211 root44Tail4 winner = winner :=
  (root44Step4).trans root44Scan5
theorem root44Scan3 : scan edge (search edge 3) 4 [211] 211 root44Tail3 winner = winner :=
  (root44Step3).trans root44Scan4
theorem root44Scan2 : scan edge (search edge 3) 4 [211] 211 root44Tail2 winner = winner :=
  (root44Step2).trans root44Scan3
theorem root44Scan1 : scan edge (search edge 3) 4 [211] 211 root44Tail1 winner = winner :=
  (root44Step1).trans root44Scan2
theorem root44Scan0 : scan edge (search edge 3) 4 [211] 211 root44Tail0 winner = winner :=
  (root44Step0).trans root44Scan1
set_option maxRecDepth 4096 in
theorem root44Branch : search edge 4 [211] root44Candidates 211 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [211] 211 root44Tail0 winner = winner
  exact root44Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
