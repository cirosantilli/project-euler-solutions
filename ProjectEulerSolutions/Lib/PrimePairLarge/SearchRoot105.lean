import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root105Candidates : List Nat := Data.row593
set_option maxRecDepth 8192 in
theorem root105Candidates_checked : candidates edge 5 0 593 tail106 winner.bound = root105Candidates := by
  apply candidates_eq_row edge 5 0 593 tail106 root105Candidates winner.bound tail106_ordered
  · decide
  · decide
  · funext q
    rfl
def root105Tail69 : List Nat := []
def root105Tail68 : List Nat := 24371 :: root105Tail69
def root105Tail67 : List Nat := 23741 :: root105Tail68
def root105Tail66 : List Nat := 23633 :: root105Tail67
def root105Tail65 : List Nat := 22943 :: root105Tail66
def root105Tail64 : List Nat := 22229 :: root105Tail65
def root105Tail63 : List Nat := 22157 :: root105Tail64
def root105Tail62 : List Nat := 22031 :: root105Tail63
def root105Tail61 : List Nat := 21557 :: root105Tail62
def root105Tail60 : List Nat := 21377 :: root105Tail61
def root105Tail59 : List Nat := 20981 :: root105Tail60
def root105Tail58 : List Nat := 20663 :: root105Tail59
def root105Tail57 : List Nat := 20549 :: root105Tail58
def root105Tail56 : List Nat := 20477 :: root105Tail57
def root105Tail55 : List Nat := 20441 :: root105Tail56
def root105Tail54 : List Nat := 20411 :: root105Tail55
def root105Tail53 : List Nat := 19961 :: root105Tail54
def root105Tail52 : List Nat := 19853 :: root105Tail53
def root105Tail51 : List Nat := 19457 :: root105Tail52
def root105Tail50 : List Nat := 18461 :: root105Tail51
def root105Tail49 : List Nat := 18131 :: root105Tail50
def root105Tail48 : List Nat := 18041 :: root105Tail49
def root105Tail47 : List Nat := 16001 :: root105Tail48
def root105Tail46 : List Nat := 15749 :: root105Tail47
def root105Tail45 : List Nat := 14153 :: root105Tail46
def root105Tail44 : List Nat := 14051 :: root105Tail45
def root105Tail43 : List Nat := 13679 :: root105Tail44
def root105Tail42 : List Nat := 13523 :: root105Tail43
def root105Tail41 : List Nat := 12809 :: root105Tail42
def root105Tail40 : List Nat := 12497 :: root105Tail41
def root105Tail39 : List Nat := 12401 :: root105Tail40
def root105Tail38 : List Nat := 12263 :: root105Tail39
def root105Tail37 : List Nat := 11393 :: root105Tail38
def root105Tail36 : List Nat := 10949 :: root105Tail37
def root105Tail35 : List Nat := 10709 :: root105Tail36
def root105Tail34 : List Nat := 10607 :: root105Tail35
def root105Tail33 : List Nat := 10301 :: root105Tail34
def root105Tail32 : List Nat := 9839 :: root105Tail33
def root105Tail31 : List Nat := 9743 :: root105Tail32
def root105Tail30 : List Nat := 9431 :: root105Tail31
def root105Tail29 : List Nat := 9209 :: root105Tail30
def root105Tail28 : List Nat := 8663 :: root105Tail29
def root105Tail27 : List Nat := 8123 :: root105Tail28
def root105Tail26 : List Nat := 7949 :: root105Tail27
def root105Tail25 : List Nat := 7577 :: root105Tail26
def root105Tail24 : List Nat := 7559 :: root105Tail25
def root105Tail23 : List Nat := 7433 :: root105Tail24
def root105Tail22 : List Nat := 6857 :: root105Tail23
def root105Tail21 : List Nat := 6653 :: root105Tail22
def root105Tail20 : List Nat := 5987 :: root105Tail21
def root105Tail19 : List Nat := 5879 :: root105Tail20
def root105Tail18 : List Nat := 5861 :: root105Tail19
def root105Tail17 : List Nat := 5639 :: root105Tail18
def root105Tail16 : List Nat := 5399 :: root105Tail17
def root105Tail15 : List Nat := 5393 :: root105Tail16
def root105Tail14 : List Nat := 4211 :: root105Tail15
def root105Tail13 : List Nat := 4133 :: root105Tail14
def root105Tail12 : List Nat := 4001 :: root105Tail13
def root105Tail11 : List Nat := 3581 :: root105Tail12
def root105Tail10 : List Nat := 3539 :: root105Tail11
def root105Tail9 : List Nat := 3467 :: root105Tail10
def root105Tail8 : List Nat := 3461 :: root105Tail9
def root105Tail7 : List Nat := 3251 :: root105Tail8
def root105Tail6 : List Nat := 3203 :: root105Tail7
def root105Tail5 : List Nat := 2417 :: root105Tail6
def root105Tail4 : List Nat := 2411 :: root105Tail5
def root105Tail3 : List Nat := 2141 :: root105Tail4
def root105Tail2 : List Nat := 977 :: root105Tail3
def root105Tail1 : List Nat := 863 :: root105Tail2
def root105Tail0 : List Nat := 647 :: root105Tail1
def root105Candidates0 : List Nat := [5879, 7949, 10301]
set_option maxRecDepth 8192 in
theorem root105Candidates0_checked : candidates edge 4 593 647 root105Tail1 winner.bound = root105Candidates0 := by
  unfold candidates
  have row : edge 647 = (SortedNatTree.contains FastRows.row647) := by
    change Data.row647.contains = (SortedNatTree.contains FastRows.row647)
    exact FastRows.row647_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root105Child0 : search edge 3 [593, 647] root105Candidates0 1240 winner = winner := by decide
theorem root105Step0 : scan edge (search edge 3) 4 [593] 593 root105Tail0 winner =
    scan edge (search edge 3) 4 [593] 593 root105Tail1 winner := by
  rw [root105Tail0, scan, if_neg (by decide), root105Candidates0_checked]
  change scan edge (search edge 3) 4 [593] 593 root105Tail1
    (if 3 ≤ root105Candidates0.length then search edge 3 [593, 647] root105Candidates0 1240 winner else winner) = _
  rw [if_pos (by decide), root105Child0]
def root105Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root105Candidates1_checked : candidates edge 4 593 863 root105Tail2 winner.bound = root105Candidates1 := by
  unfold candidates
  have row : edge 863 = (SortedNatTree.contains FastRows.row863) := by
    change Data.row863.contains = (SortedNatTree.contains FastRows.row863)
    exact FastRows.row863_checked.symm
  rw [row]
  decide
theorem root105Step1 : scan edge (search edge 3) 4 [593] 593 root105Tail1 winner =
    scan edge (search edge 3) 4 [593] 593 root105Tail2 winner := by
  rw [root105Tail1, scan, if_neg (by decide), root105Candidates1_checked]
  change scan edge (search edge 3) 4 [593] 593 root105Tail2
    (if 3 ≤ root105Candidates1.length then search edge 3 [593, 863] root105Candidates1 1456 winner else winner) = _
  rw [if_neg (by decide)]
def root105Candidates2 : List Nat := [2141, 10607]
set_option maxRecDepth 8192 in
theorem root105Candidates2_checked : candidates edge 4 593 977 root105Tail3 winner.bound = root105Candidates2 := by
  unfold candidates
  have row : edge 977 = (SortedNatTree.contains FastRows.row977) := by
    change Data.row977.contains = (SortedNatTree.contains FastRows.row977)
    exact FastRows.row977_checked.symm
  rw [row]
  decide
theorem root105Step2 : scan edge (search edge 3) 4 [593] 593 root105Tail2 winner =
    scan edge (search edge 3) 4 [593] 593 root105Tail3 winner := by
  rw [root105Tail2, scan, if_neg (by decide), root105Candidates2_checked]
  change scan edge (search edge 3) 4 [593] 593 root105Tail3
    (if 3 ≤ root105Candidates2.length then search edge 3 [593, 977] root105Candidates2 1570 winner else winner) = _
  rw [if_neg (by decide)]
def root105Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root105Candidates3_checked : candidates edge 4 593 2141 root105Tail4 winner.bound = root105Candidates3 := by
  unfold candidates
  have row : edge 2141 = (SortedNatTree.contains FastRows.row2141) := by
    change Data.row2141.contains = (SortedNatTree.contains FastRows.row2141)
    exact FastRows.row2141_checked.symm
  rw [row]
  decide
theorem root105Step3 : scan edge (search edge 3) 4 [593] 593 root105Tail3 winner =
    scan edge (search edge 3) 4 [593] 593 root105Tail4 winner := by
  rw [root105Tail3, scan, if_neg (by decide), root105Candidates3_checked]
  change scan edge (search edge 3) 4 [593] 593 root105Tail4
    (if 3 ≤ root105Candidates3.length then search edge 3 [593, 2141] root105Candidates3 2734 winner else winner) = _
  rw [if_neg (by decide)]
def root105Candidates4 : List Nat := [5879, 7433]
set_option maxRecDepth 8192 in
theorem root105Candidates4_checked : candidates edge 4 593 2411 root105Tail5 winner.bound = root105Candidates4 := by
  unfold candidates
  have row : edge 2411 = (SortedNatTree.contains FastRows.row2411) := by
    change Data.row2411.contains = (SortedNatTree.contains FastRows.row2411)
    exact FastRows.row2411_checked.symm
  rw [row]
  decide
theorem root105Step4 : scan edge (search edge 3) 4 [593] 593 root105Tail4 winner =
    scan edge (search edge 3) 4 [593] 593 root105Tail5 winner := by
  rw [root105Tail4, scan, if_neg (by decide), root105Candidates4_checked]
  change scan edge (search edge 3) 4 [593] 593 root105Tail5
    (if 3 ≤ root105Candidates4.length then search edge 3 [593, 2411] root105Candidates4 3004 winner else winner) = _
  rw [if_neg (by decide)]
def root105Candidates5 : List Nat := [4001, 6857, 13523]
set_option maxRecDepth 8192 in
theorem root105Candidates5_checked : candidates edge 4 593 2417 root105Tail6 winner.bound = root105Candidates5 := by
  unfold candidates
  have row : edge 2417 = (SortedNatTree.contains FastRows.row2417) := by
    change Data.row2417.contains = (SortedNatTree.contains FastRows.row2417)
    exact FastRows.row2417_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root105Child5 : search edge 3 [593, 2417] root105Candidates5 3010 winner = winner := by decide
theorem root105Step5 : scan edge (search edge 3) 4 [593] 593 root105Tail5 winner =
    scan edge (search edge 3) 4 [593] 593 root105Tail6 winner := by
  rw [root105Tail5, scan, if_neg (by decide), root105Candidates5_checked]
  change scan edge (search edge 3) 4 [593] 593 root105Tail6
    (if 3 ≤ root105Candidates5.length then search edge 3 [593, 2417] root105Candidates5 3010 winner else winner) = _
  rw [if_pos (by decide), root105Child5]
def root105Candidates6 : List Nat := [3467, 7949, 9209]
set_option maxRecDepth 8192 in
theorem root105Candidates6_checked : candidates edge 4 593 3203 root105Tail7 winner.bound = root105Candidates6 := by
  unfold candidates
  have row : edge 3203 = (SortedNatTree.contains FastRows.row3203) := by
    change Data.row3203.contains = (SortedNatTree.contains FastRows.row3203)
    exact FastRows.row3203_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root105Child6 : search edge 3 [593, 3203] root105Candidates6 3796 winner = winner := by decide
theorem root105Step6 : scan edge (search edge 3) 4 [593] 593 root105Tail6 winner =
    scan edge (search edge 3) 4 [593] 593 root105Tail7 winner := by
  rw [root105Tail6, scan, if_neg (by decide), root105Candidates6_checked]
  change scan edge (search edge 3) 4 [593] 593 root105Tail7
    (if 3 ≤ root105Candidates6.length then search edge 3 [593, 3203] root105Candidates6 3796 winner else winner) = _
  rw [if_pos (by decide), root105Child6]
def root105Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root105Candidates7_checked : candidates edge 4 593 3251 root105Tail8 winner.bound = root105Candidates7 := by
  unfold candidates
  have row : edge 3251 = (SortedNatTree.contains FastRows.row3251) := by
    change Data.row3251.contains = (SortedNatTree.contains FastRows.row3251)
    exact FastRows.row3251_checked.symm
  rw [row]
  decide
theorem root105Step7 : scan edge (search edge 3) 4 [593] 593 root105Tail7 winner =
    scan edge (search edge 3) 4 [593] 593 root105Tail8 winner := by
  rw [root105Tail7, scan, if_neg (by decide), root105Candidates7_checked]
  change scan edge (search edge 3) 4 [593] 593 root105Tail8
    (if 3 ≤ root105Candidates7.length then search edge 3 [593, 3251] root105Candidates7 3844 winner else winner) = _
  rw [if_neg (by decide)]
def root105Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root105Candidates8_checked : candidates edge 4 593 3461 root105Tail9 winner.bound = root105Candidates8 := by
  unfold candidates
  have row : edge 3461 = (SortedNatTree.contains FastRows.row3461) := by
    change Data.row3461.contains = (SortedNatTree.contains FastRows.row3461)
    exact FastRows.row3461_checked.symm
  rw [row]
  decide
theorem root105Step8 : scan edge (search edge 3) 4 [593] 593 root105Tail8 winner =
    scan edge (search edge 3) 4 [593] 593 root105Tail9 winner := by
  rw [root105Tail8, scan, if_neg (by decide), root105Candidates8_checked]
  change scan edge (search edge 3) 4 [593] 593 root105Tail9
    (if 3 ≤ root105Candidates8.length then search edge 3 [593, 3461] root105Candidates8 4054 winner else winner) = _
  rw [if_neg (by decide)]
def root105Candidates9 : List Nat := [7949, 8663]
set_option maxRecDepth 8192 in
theorem root105Candidates9_checked : candidates edge 4 593 3467 root105Tail10 winner.bound = root105Candidates9 := by
  unfold candidates
  have row : edge 3467 = (SortedNatTree.contains FastRows.row3467) := by
    change Data.row3467.contains = (SortedNatTree.contains FastRows.row3467)
    exact FastRows.row3467_checked.symm
  rw [row]
  decide
theorem root105Step9 : scan edge (search edge 3) 4 [593] 593 root105Tail9 winner =
    scan edge (search edge 3) 4 [593] 593 root105Tail10 winner := by
  rw [root105Tail9, scan, if_neg (by decide), root105Candidates9_checked]
  change scan edge (search edge 3) 4 [593] 593 root105Tail10
    (if 3 ≤ root105Candidates9.length then search edge 3 [593, 3467] root105Candidates9 4060 winner else winner) = _
  rw [if_neg (by decide)]
def root105Candidates10 : List Nat := [5861, 19457]
set_option maxRecDepth 8192 in
theorem root105Candidates10_checked : candidates edge 4 593 3539 root105Tail11 winner.bound = root105Candidates10 := by
  unfold candidates
  have row : edge 3539 = (SortedNatTree.contains FastRows.row3539) := by
    change Data.row3539.contains = (SortedNatTree.contains FastRows.row3539)
    exact FastRows.row3539_checked.symm
  rw [row]
  decide
theorem root105Step10 : scan edge (search edge 3) 4 [593] 593 root105Tail10 winner =
    scan edge (search edge 3) 4 [593] 593 root105Tail11 winner := by
  rw [root105Tail10, scan, if_neg (by decide), root105Candidates10_checked]
  change scan edge (search edge 3) 4 [593] 593 root105Tail11
    (if 3 ≤ root105Candidates10.length then search edge 3 [593, 3539] root105Candidates10 4132 winner else winner) = _
  rw [if_neg (by decide)]
def root105Candidates11 : List Nat := [4133, 12401]
set_option maxRecDepth 8192 in
theorem root105Candidates11_checked : candidates edge 4 593 3581 root105Tail12 winner.bound = root105Candidates11 := by
  unfold candidates
  have row : edge 3581 = (SortedNatTree.contains FastRows.row3581) := by
    change Data.row3581.contains = (SortedNatTree.contains FastRows.row3581)
    exact FastRows.row3581_checked.symm
  rw [row]
  decide
theorem root105Step11 : scan edge (search edge 3) 4 [593] 593 root105Tail11 winner =
    scan edge (search edge 3) 4 [593] 593 root105Tail12 winner := by
  rw [root105Tail11, scan, if_neg (by decide), root105Candidates11_checked]
  change scan edge (search edge 3) 4 [593] 593 root105Tail12
    (if 3 ≤ root105Candidates11.length then search edge 3 [593, 3581] root105Candidates11 4174 winner else winner) = _
  rw [if_neg (by decide)]
def root105Candidates12 : List Nat := [7577]
set_option maxRecDepth 8192 in
theorem root105Candidates12_checked : candidates edge 4 593 4001 root105Tail13 winner.bound = root105Candidates12 := by
  unfold candidates
  have row : edge 4001 = (SortedNatTree.contains FastRows.row4001) := by
    change Data.row4001.contains = (SortedNatTree.contains FastRows.row4001)
    exact FastRows.row4001_checked.symm
  rw [row]
  decide
theorem root105Step12 : scan edge (search edge 3) 4 [593] 593 root105Tail12 winner =
    scan edge (search edge 3) 4 [593] 593 root105Tail13 winner := by
  rw [root105Tail12, scan, if_neg (by decide), root105Candidates12_checked]
  change scan edge (search edge 3) 4 [593] 593 root105Tail13
    (if 3 ≤ root105Candidates12.length then search edge 3 [593, 4001] root105Candidates12 4594 winner else winner) = _
  rw [if_neg (by decide)]
def root105Candidates13 : List Nat := [7433, 14051]
set_option maxRecDepth 8192 in
theorem root105Candidates13_checked : candidates edge 4 593 4133 root105Tail14 winner.bound = root105Candidates13 := by
  unfold candidates
  have row : edge 4133 = (SortedNatTree.contains FastRows.row4133) := by
    change Data.row4133.contains = (SortedNatTree.contains FastRows.row4133)
    exact FastRows.row4133_checked.symm
  rw [row]
  decide
theorem root105Step13 : scan edge (search edge 3) 4 [593] 593 root105Tail13 winner =
    scan edge (search edge 3) 4 [593] 593 root105Tail14 winner := by
  rw [root105Tail13, scan, if_neg (by decide), root105Candidates13_checked]
  change scan edge (search edge 3) 4 [593] 593 root105Tail14
    (if 3 ≤ root105Candidates13.length then search edge 3 [593, 4133] root105Candidates13 4726 winner else winner) = _
  rw [if_neg (by decide)]
def root105Candidates14 : List Nat := [10607, 18041]
set_option maxRecDepth 8192 in
theorem root105Candidates14_checked : candidates edge 4 593 4211 root105Tail15 winner.bound = root105Candidates14 := by
  unfold candidates
  have row : edge 4211 = (SortedNatTree.contains FastRows.row4211) := by
    change Data.row4211.contains = (SortedNatTree.contains FastRows.row4211)
    exact FastRows.row4211_checked.symm
  rw [row]
  decide
theorem root105Step14 : scan edge (search edge 3) 4 [593] 593 root105Tail14 winner =
    scan edge (search edge 3) 4 [593] 593 root105Tail15 winner := by
  rw [root105Tail14, scan, if_neg (by decide), root105Candidates14_checked]
  change scan edge (search edge 3) 4 [593] 593 root105Tail15
    (if 3 ≤ root105Candidates14.length then search edge 3 [593, 4211] root105Candidates14 4804 winner else winner) = _
  rw [if_neg (by decide)]
def root105Candidates15 : List Nat := [19457]
set_option maxRecDepth 8192 in
theorem root105Candidates15_checked : candidates edge 4 593 5393 root105Tail16 winner.bound = root105Candidates15 := by
  unfold candidates
  have row : edge 5393 = (SortedNatTree.contains FastRows.row5393) := by
    change Data.row5393.contains = (SortedNatTree.contains FastRows.row5393)
    exact FastRows.row5393_checked.symm
  rw [row]
  decide
theorem root105Step15 : scan edge (search edge 3) 4 [593] 593 root105Tail15 winner =
    scan edge (search edge 3) 4 [593] 593 root105Tail16 winner := by
  rw [root105Tail15, scan, if_neg (by decide), root105Candidates15_checked]
  change scan edge (search edge 3) 4 [593] 593 root105Tail16
    (if 3 ≤ root105Candidates15.length then search edge 3 [593, 5393] root105Candidates15 5986 winner else winner) = _
  rw [if_neg (by decide)]
def root105Candidates16 : List Nat := [9431, 15749, 18131]
set_option maxRecDepth 8192 in
theorem root105Candidates16_checked : candidates edge 4 593 5399 root105Tail17 winner.bound = root105Candidates16 := by
  unfold candidates
  have row : edge 5399 = (SortedNatTree.contains FastRows.row5399) := by
    change Data.row5399.contains = (SortedNatTree.contains FastRows.row5399)
    exact FastRows.row5399_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root105Child16 : search edge 3 [593, 5399] root105Candidates16 5992 winner = winner := by decide
theorem root105Step16 : scan edge (search edge 3) 4 [593] 593 root105Tail16 winner =
    scan edge (search edge 3) 4 [593] 593 root105Tail17 winner := by
  rw [root105Tail16, scan, if_neg (by decide), root105Candidates16_checked]
  change scan edge (search edge 3) 4 [593] 593 root105Tail17
    (if 3 ≤ root105Candidates16.length then search edge 3 [593, 5399] root105Candidates16 5992 winner else winner) = _
  rw [if_pos (by decide), root105Child16]
def root105Candidates17 : List Nat := [9839]
set_option maxRecDepth 8192 in
theorem root105Candidates17_checked : candidates edge 4 593 5639 root105Tail18 winner.bound = root105Candidates17 := by
  unfold candidates
  have row : edge 5639 = (SortedNatTree.contains FastRows.row5639) := by
    change Data.row5639.contains = (SortedNatTree.contains FastRows.row5639)
    exact FastRows.row5639_checked.symm
  rw [row]
  decide
theorem root105Step17 : scan edge (search edge 3) 4 [593] 593 root105Tail17 winner =
    scan edge (search edge 3) 4 [593] 593 root105Tail18 winner := by
  rw [root105Tail17, scan, if_neg (by decide), root105Candidates17_checked]
  change scan edge (search edge 3) 4 [593] 593 root105Tail18
    (if 3 ≤ root105Candidates17.length then search edge 3 [593, 5639] root105Candidates17 6232 winner else winner) = _
  rw [if_neg (by decide)]
def root105Candidates18 : List Nat := [5987, 13679]
set_option maxRecDepth 8192 in
theorem root105Candidates18_checked : candidates edge 4 593 5861 root105Tail19 winner.bound = root105Candidates18 := by
  unfold candidates
  have row : edge 5861 = (SortedNatTree.contains FastRows.row5861) := by
    change Data.row5861.contains = (SortedNatTree.contains FastRows.row5861)
    exact FastRows.row5861_checked.symm
  rw [row]
  decide
theorem root105Step18 : scan edge (search edge 3) 4 [593] 593 root105Tail18 winner =
    scan edge (search edge 3) 4 [593] 593 root105Tail19 winner := by
  rw [root105Tail18, scan, if_neg (by decide), root105Candidates18_checked]
  change scan edge (search edge 3) 4 [593] 593 root105Tail19
    (if 3 ≤ root105Candidates18.length then search edge 3 [593, 5861] root105Candidates18 6454 winner else winner) = _
  rw [if_neg (by decide)]
def root105Candidates19 : List Nat := []
set_option maxRecDepth 8192 in
theorem root105Candidates19_checked : candidates edge 4 593 5879 root105Tail20 winner.bound = root105Candidates19 := by
  unfold candidates
  have row : edge 5879 = (SortedNatTree.contains FastRows.row5879) := by
    change Data.row5879.contains = (SortedNatTree.contains FastRows.row5879)
    exact FastRows.row5879_checked.symm
  rw [row]
  decide
theorem root105Step19 : scan edge (search edge 3) 4 [593] 593 root105Tail19 winner =
    scan edge (search edge 3) 4 [593] 593 root105Tail20 winner := by
  rw [root105Tail19, scan, if_neg (by decide), root105Candidates19_checked]
  change scan edge (search edge 3) 4 [593] 593 root105Tail20
    (if 3 ≤ root105Candidates19.length then search edge 3 [593, 5879] root105Candidates19 6472 winner else winner) = _
  rw [if_neg (by decide)]
def root105Candidates20 : List Nat := [12497]
set_option maxRecDepth 8192 in
theorem root105Candidates20_checked : candidates edge 4 593 5987 root105Tail21 winner.bound = root105Candidates20 := by
  unfold candidates
  have row : edge 5987 = (SortedNatTree.contains FastRows.row5987) := by
    change Data.row5987.contains = (SortedNatTree.contains FastRows.row5987)
    exact FastRows.row5987_checked.symm
  rw [row]
  decide
theorem root105Step20 : scan edge (search edge 3) 4 [593] 593 root105Tail20 winner =
    scan edge (search edge 3) 4 [593] 593 root105Tail21 winner := by
  rw [root105Tail20, scan, if_neg (by decide), root105Candidates20_checked]
  change scan edge (search edge 3) 4 [593] 593 root105Tail21
    (if 3 ≤ root105Candidates20.length then search edge 3 [593, 5987] root105Candidates20 6580 winner else winner) = _
  rw [if_neg (by decide)]
theorem root105Scan21 : scan edge (search edge 3) 4 [593] 593 root105Tail21 winner = winner := by rw [root105Tail21, scan, if_pos (by decide)]
theorem root105Scan20 : scan edge (search edge 3) 4 [593] 593 root105Tail20 winner = winner :=
  (root105Step20).trans root105Scan21
theorem root105Scan19 : scan edge (search edge 3) 4 [593] 593 root105Tail19 winner = winner :=
  (root105Step19).trans root105Scan20
theorem root105Scan18 : scan edge (search edge 3) 4 [593] 593 root105Tail18 winner = winner :=
  (root105Step18).trans root105Scan19
theorem root105Scan17 : scan edge (search edge 3) 4 [593] 593 root105Tail17 winner = winner :=
  (root105Step17).trans root105Scan18
theorem root105Scan16 : scan edge (search edge 3) 4 [593] 593 root105Tail16 winner = winner :=
  (root105Step16).trans root105Scan17
theorem root105Scan15 : scan edge (search edge 3) 4 [593] 593 root105Tail15 winner = winner :=
  (root105Step15).trans root105Scan16
theorem root105Scan14 : scan edge (search edge 3) 4 [593] 593 root105Tail14 winner = winner :=
  (root105Step14).trans root105Scan15
theorem root105Scan13 : scan edge (search edge 3) 4 [593] 593 root105Tail13 winner = winner :=
  (root105Step13).trans root105Scan14
theorem root105Scan12 : scan edge (search edge 3) 4 [593] 593 root105Tail12 winner = winner :=
  (root105Step12).trans root105Scan13
theorem root105Scan11 : scan edge (search edge 3) 4 [593] 593 root105Tail11 winner = winner :=
  (root105Step11).trans root105Scan12
theorem root105Scan10 : scan edge (search edge 3) 4 [593] 593 root105Tail10 winner = winner :=
  (root105Step10).trans root105Scan11
theorem root105Scan9 : scan edge (search edge 3) 4 [593] 593 root105Tail9 winner = winner :=
  (root105Step9).trans root105Scan10
theorem root105Scan8 : scan edge (search edge 3) 4 [593] 593 root105Tail8 winner = winner :=
  (root105Step8).trans root105Scan9
theorem root105Scan7 : scan edge (search edge 3) 4 [593] 593 root105Tail7 winner = winner :=
  (root105Step7).trans root105Scan8
theorem root105Scan6 : scan edge (search edge 3) 4 [593] 593 root105Tail6 winner = winner :=
  (root105Step6).trans root105Scan7
theorem root105Scan5 : scan edge (search edge 3) 4 [593] 593 root105Tail5 winner = winner :=
  (root105Step5).trans root105Scan6
theorem root105Scan4 : scan edge (search edge 3) 4 [593] 593 root105Tail4 winner = winner :=
  (root105Step4).trans root105Scan5
theorem root105Scan3 : scan edge (search edge 3) 4 [593] 593 root105Tail3 winner = winner :=
  (root105Step3).trans root105Scan4
theorem root105Scan2 : scan edge (search edge 3) 4 [593] 593 root105Tail2 winner = winner :=
  (root105Step2).trans root105Scan3
theorem root105Scan1 : scan edge (search edge 3) 4 [593] 593 root105Tail1 winner = winner :=
  (root105Step1).trans root105Scan2
theorem root105Scan0 : scan edge (search edge 3) 4 [593] 593 root105Tail0 winner = winner :=
  (root105Step0).trans root105Scan1
set_option maxRecDepth 4096 in
theorem root105Branch : search edge 4 [593] root105Candidates 593 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [593] 593 root105Tail0 winner = winner
  exact root105Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
