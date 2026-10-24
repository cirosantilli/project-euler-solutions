import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root31Candidates : List Nat := Data.row139
set_option maxRecDepth 8192 in
theorem root31Candidates_checked : candidates edge 5 0 139 tail32 winner.bound = root31Candidates := by
  apply candidates_eq_row edge 5 0 139 tail32 root31Candidates winner.bound tail32_ordered
  · decide
  · decide
  · funext q
    rfl
def root31Tail78 : List Nat := []
def root31Tail77 : List Nat := 25447 :: root31Tail78
def root31Tail76 : List Nat := 24571 :: root31Tail77
def root31Tail75 : List Nat := 24439 :: root31Tail76
def root31Tail74 : List Nat := 24181 :: root31Tail75
def root31Tail73 : List Nat := 24109 :: root31Tail74
def root31Tail72 : List Nat := 24061 :: root31Tail73
def root31Tail71 : List Nat := 23893 :: root31Tail72
def root31Tail70 : List Nat := 23473 :: root31Tail71
def root31Tail69 : List Nat := 23431 :: root31Tail70
def root31Tail68 : List Nat := 23011 :: root31Tail69
def root31Tail67 : List Nat := 22807 :: root31Tail68
def root31Tail66 : List Nat := 22453 :: root31Tail67
def root31Tail65 : List Nat := 22159 :: root31Tail66
def root31Tail64 : List Nat := 22039 :: root31Tail65
def root31Tail63 : List Nat := 21493 :: root31Tail64
def root31Tail62 : List Nat := 21211 :: root31Tail63
def root31Tail61 : List Nat := 21163 :: root31Tail62
def root31Tail60 : List Nat := 20479 :: root31Tail61
def root31Tail59 : List Nat := 20323 :: root31Tail60
def root31Tail58 : List Nat := 20113 :: root31Tail59
def root31Tail57 : List Nat := 19819 :: root31Tail58
def root31Tail56 : List Nat := 19387 :: root31Tail57
def root31Tail55 : List Nat := 19051 :: root31Tail56
def root31Tail54 : List Nat := 18133 :: root31Tail55
def root31Tail53 : List Nat := 17581 :: root31Tail54
def root31Tail52 : List Nat := 17467 :: root31Tail53
def root31Tail51 : List Nat := 13297 :: root31Tail52
def root31Tail50 : List Nat := 13219 :: root31Tail51
def root31Tail49 : List Nat := 13093 :: root31Tail50
def root31Tail48 : List Nat := 12979 :: root31Tail49
def root31Tail47 : List Nat := 12841 :: root31Tail48
def root31Tail46 : List Nat := 12757 :: root31Tail47
def root31Tail45 : List Nat := 12601 :: root31Tail46
def root31Tail44 : List Nat := 11719 :: root31Tail45
def root31Tail43 : List Nat := 11497 :: root31Tail44
def root31Tail42 : List Nat := 11383 :: root31Tail43
def root31Tail41 : List Nat := 11047 :: root31Tail42
def root31Tail40 : List Nat := 10993 :: root31Tail41
def root31Tail39 : List Nat := 10531 :: root31Tail40
def root31Tail38 : List Nat := 9403 :: root31Tail39
def root31Tail37 : List Nat := 8779 :: root31Tail38
def root31Tail36 : List Nat := 8737 :: root31Tail37
def root31Tail35 : List Nat := 8731 :: root31Tail36
def root31Tail34 : List Nat := 8269 :: root31Tail35
def root31Tail33 : List Nat := 8053 :: root31Tail34
def root31Tail32 : List Nat := 8011 :: root31Tail33
def root31Tail31 : List Nat := 7933 :: root31Tail32
def root31Tail30 : List Nat := 7219 :: root31Tail31
def root31Tail29 : List Nat := 7177 :: root31Tail30
def root31Tail28 : List Nat := 7057 :: root31Tail29
def root31Tail27 : List Nat := 6949 :: root31Tail28
def root31Tail26 : List Nat := 6841 :: root31Tail27
def root31Tail25 : List Nat := 6679 :: root31Tail26
def root31Tail24 : List Nat := 5869 :: root31Tail25
def root31Tail23 : List Nat := 5743 :: root31Tail24
def root31Tail22 : List Nat := 5323 :: root31Tail23
def root31Tail21 : List Nat := 5167 :: root31Tail22
def root31Tail20 : List Nat := 5059 :: root31Tail21
def root31Tail19 : List Nat := 4177 :: root31Tail20
def root31Tail18 : List Nat := 3739 :: root31Tail19
def root31Tail17 : List Nat := 3697 :: root31Tail18
def root31Tail16 : List Nat := 3313 :: root31Tail17
def root31Tail15 : List Nat := 3253 :: root31Tail16
def root31Tail14 : List Nat := 3229 :: root31Tail15
def root31Tail13 : List Nat := 2803 :: root31Tail14
def root31Tail12 : List Nat := 1597 :: root31Tail13
def root31Tail11 : List Nat := 1549 :: root31Tail12
def root31Tail10 : List Nat := 1447 :: root31Tail11
def root31Tail9 : List Nat := 1051 :: root31Tail10
def root31Tail8 : List Nat := 967 :: root31Tail9
def root31Tail7 : List Nat := 907 :: root31Tail8
def root31Tail6 : List Nat := 787 :: root31Tail7
def root31Tail5 : List Nat := 709 :: root31Tail6
def root31Tail4 : List Nat := 661 :: root31Tail5
def root31Tail3 : List Nat := 619 :: root31Tail4
def root31Tail2 : List Nat := 547 :: root31Tail3
def root31Tail1 : List Nat := 457 :: root31Tail2
def root31Tail0 : List Nat := 367 :: root31Tail1
def root31Candidates0 : List Nat := [457, 5059, 13297, 23473]
set_option maxRecDepth 8192 in
theorem root31Candidates0_checked : candidates edge 4 139 367 root31Tail1 winner.bound = root31Candidates0 := by
  unfold candidates
  have row : edge 367 = (SortedNatTree.contains FastRows.row367) := by
    change Data.row367.contains = (SortedNatTree.contains FastRows.row367)
    exact FastRows.row367_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root31Child0 : search edge 3 [139, 367] root31Candidates0 506 winner = winner := by decide
theorem root31Step0 : scan edge (search edge 3) 4 [139] 139 root31Tail0 winner =
    scan edge (search edge 3) 4 [139] 139 root31Tail1 winner := by
  rw [root31Tail0, scan, if_neg (by decide), root31Candidates0_checked]
  change scan edge (search edge 3) 4 [139] 139 root31Tail1
    (if 3 ≤ root31Candidates0.length then search edge 3 [139, 367] root31Candidates0 506 winner else winner) = _
  rw [if_pos (by decide), root31Child0]
def root31Candidates1 : List Nat := [1597, 24571]
set_option maxRecDepth 8192 in
theorem root31Candidates1_checked : candidates edge 4 139 457 root31Tail2 winner.bound = root31Candidates1 := by
  unfold candidates
  have row : edge 457 = (SortedNatTree.contains FastRows.row457) := by
    change Data.row457.contains = (SortedNatTree.contains FastRows.row457)
    exact FastRows.row457_checked.symm
  rw [row]
  decide
theorem root31Step1 : scan edge (search edge 3) 4 [139] 139 root31Tail1 winner =
    scan edge (search edge 3) 4 [139] 139 root31Tail2 winner := by
  rw [root31Tail1, scan, if_neg (by decide), root31Candidates1_checked]
  change scan edge (search edge 3) 4 [139] 139 root31Tail2
    (if 3 ≤ root31Candidates1.length then search edge 3 [139, 457] root31Candidates1 596 winner else winner) = _
  rw [if_neg (by decide)]
def root31Candidates2 : List Nat := [661, 709, 787, 3229, 13093, 22039]
set_option maxRecDepth 8192 in
theorem root31Candidates2_checked : candidates edge 4 139 547 root31Tail3 winner.bound = root31Candidates2 := by
  unfold candidates
  have row : edge 547 = (SortedNatTree.contains FastRows.row547) := by
    change Data.row547.contains = (SortedNatTree.contains FastRows.row547)
    exact FastRows.row547_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root31Child2 : search edge 3 [139, 547] root31Candidates2 686 winner = winner := by decide
theorem root31Step2 : scan edge (search edge 3) 4 [139] 139 root31Tail2 winner =
    scan edge (search edge 3) 4 [139] 139 root31Tail3 winner := by
  rw [root31Tail2, scan, if_neg (by decide), root31Candidates2_checked]
  change scan edge (search edge 3) 4 [139] 139 root31Tail3
    (if 3 ≤ root31Candidates2.length then search edge 3 [139, 547] root31Candidates2 686 winner else winner) = _
  rw [if_pos (by decide), root31Child2]
def root31Candidates3 : List Nat := [1051, 1597]
set_option maxRecDepth 8192 in
theorem root31Candidates3_checked : candidates edge 4 139 619 root31Tail4 winner.bound = root31Candidates3 := by
  unfold candidates
  have row : edge 619 = (SortedNatTree.contains FastRows.row619) := by
    change Data.row619.contains = (SortedNatTree.contains FastRows.row619)
    exact FastRows.row619_checked.symm
  rw [row]
  decide
theorem root31Step3 : scan edge (search edge 3) 4 [139] 139 root31Tail3 winner =
    scan edge (search edge 3) 4 [139] 139 root31Tail4 winner := by
  rw [root31Tail3, scan, if_neg (by decide), root31Candidates3_checked]
  change scan edge (search edge 3) 4 [139] 139 root31Tail4
    (if 3 ≤ root31Candidates3.length then search edge 3 [139, 619] root31Candidates3 758 winner else winner) = _
  rw [if_neg (by decide)]
def root31Candidates4 : List Nat := [1447, 11719, 19819]
set_option maxRecDepth 8192 in
theorem root31Candidates4_checked : candidates edge 4 139 661 root31Tail5 winner.bound = root31Candidates4 := by
  unfold candidates
  have row : edge 661 = (SortedNatTree.contains FastRows.row661) := by
    change Data.row661.contains = (SortedNatTree.contains FastRows.row661)
    exact FastRows.row661_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root31Child4 : search edge 3 [139, 661] root31Candidates4 800 winner = winner := by decide
theorem root31Step4 : scan edge (search edge 3) 4 [139] 139 root31Tail4 winner =
    scan edge (search edge 3) 4 [139] 139 root31Tail5 winner := by
  rw [root31Tail4, scan, if_neg (by decide), root31Candidates4_checked]
  change scan edge (search edge 3) 4 [139] 139 root31Tail5
    (if 3 ≤ root31Candidates4.length then search edge 3 [139, 661] root31Candidates4 800 winner else winner) = _
  rw [if_pos (by decide), root31Child4]
def root31Candidates5 : List Nat := [967, 4177, 8737, 11497, 22039, 22807]
set_option maxRecDepth 8192 in
theorem root31Candidates5_checked : candidates edge 4 139 709 root31Tail6 winner.bound = root31Candidates5 := by
  unfold candidates
  have row : edge 709 = (SortedNatTree.contains FastRows.row709) := by
    change Data.row709.contains = (SortedNatTree.contains FastRows.row709)
    exact FastRows.row709_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root31Child5 : search edge 3 [139, 709] root31Candidates5 848 winner = winner := by decide
theorem root31Step5 : scan edge (search edge 3) 4 [139] 139 root31Tail5 winner =
    scan edge (search edge 3) 4 [139] 139 root31Tail6 winner := by
  rw [root31Tail5, scan, if_neg (by decide), root31Candidates5_checked]
  change scan edge (search edge 3) 4 [139] 139 root31Tail6
    (if 3 ≤ root31Candidates5.length then search edge 3 [139, 709] root31Candidates5 848 winner else winner) = _
  rw [if_pos (by decide), root31Child5]
def root31Candidates6 : List Nat := [4177]
set_option maxRecDepth 8192 in
theorem root31Candidates6_checked : candidates edge 4 139 787 root31Tail7 winner.bound = root31Candidates6 := by
  unfold candidates
  have row : edge 787 = (SortedNatTree.contains FastRows.row787) := by
    change Data.row787.contains = (SortedNatTree.contains FastRows.row787)
    exact FastRows.row787_checked.symm
  rw [row]
  decide
theorem root31Step6 : scan edge (search edge 3) 4 [139] 139 root31Tail6 winner =
    scan edge (search edge 3) 4 [139] 139 root31Tail7 winner := by
  rw [root31Tail6, scan, if_neg (by decide), root31Candidates6_checked]
  change scan edge (search edge 3) 4 [139] 139 root31Tail7
    (if 3 ≤ root31Candidates6.length then search edge 3 [139, 787] root31Candidates6 926 winner else winner) = _
  rw [if_neg (by decide)]
def root31Candidates7 : List Nat := [6841]
set_option maxRecDepth 8192 in
theorem root31Candidates7_checked : candidates edge 4 139 907 root31Tail8 winner.bound = root31Candidates7 := by
  unfold candidates
  have row : edge 907 = (SortedNatTree.contains FastRows.row907) := by
    change Data.row907.contains = (SortedNatTree.contains FastRows.row907)
    exact FastRows.row907_checked.symm
  rw [row]
  decide
theorem root31Step7 : scan edge (search edge 3) 4 [139] 139 root31Tail7 winner =
    scan edge (search edge 3) 4 [139] 139 root31Tail8 winner := by
  rw [root31Tail7, scan, if_neg (by decide), root31Candidates7_checked]
  change scan edge (search edge 3) 4 [139] 139 root31Tail8
    (if 3 ≤ root31Candidates7.length then search edge 3 [139, 907] root31Candidates7 1046 winner else winner) = _
  rw [if_neg (by decide)]
def root31Candidates8 : List Nat := [6949, 8269, 10993]
set_option maxRecDepth 8192 in
theorem root31Candidates8_checked : candidates edge 4 139 967 root31Tail9 winner.bound = root31Candidates8 := by
  unfold candidates
  have row : edge 967 = (SortedNatTree.contains FastRows.row967) := by
    change Data.row967.contains = (SortedNatTree.contains FastRows.row967)
    exact FastRows.row967_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root31Child8 : search edge 3 [139, 967] root31Candidates8 1106 winner = winner := by decide
theorem root31Step8 : scan edge (search edge 3) 4 [139] 139 root31Tail8 winner =
    scan edge (search edge 3) 4 [139] 139 root31Tail9 winner := by
  rw [root31Tail8, scan, if_neg (by decide), root31Candidates8_checked]
  change scan edge (search edge 3) 4 [139] 139 root31Tail9
    (if 3 ≤ root31Candidates8.length then search edge 3 [139, 967] root31Candidates8 1106 winner else winner) = _
  rw [if_pos (by decide), root31Child8]
def root31Candidates9 : List Nat := [3739, 13297, 22453]
set_option maxRecDepth 8192 in
theorem root31Candidates9_checked : candidates edge 4 139 1051 root31Tail10 winner.bound = root31Candidates9 := by
  unfold candidates
  have row : edge 1051 = (SortedNatTree.contains FastRows.row1051) := by
    change Data.row1051.contains = (SortedNatTree.contains FastRows.row1051)
    exact FastRows.row1051_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root31Child9 : search edge 3 [139, 1051] root31Candidates9 1190 winner = winner := by decide
theorem root31Step9 : scan edge (search edge 3) 4 [139] 139 root31Tail9 winner =
    scan edge (search edge 3) 4 [139] 139 root31Tail10 winner := by
  rw [root31Tail9, scan, if_neg (by decide), root31Candidates9_checked]
  change scan edge (search edge 3) 4 [139] 139 root31Tail10
    (if 3 ≤ root31Candidates9.length then search edge 3 [139, 1051] root31Candidates9 1190 winner else winner) = _
  rw [if_pos (by decide), root31Child9]
def root31Candidates10 : List Nat := [1549, 2803, 3229, 3313, 6949]
set_option maxRecDepth 8192 in
theorem root31Candidates10_checked : candidates edge 4 139 1447 root31Tail11 winner.bound = root31Candidates10 := by
  unfold candidates
  have row : edge 1447 = (SortedNatTree.contains FastRows.row1447) := by
    change Data.row1447.contains = (SortedNatTree.contains FastRows.row1447)
    exact FastRows.row1447_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root31Child10 : search edge 3 [139, 1447] root31Candidates10 1586 winner = winner := by decide
theorem root31Step10 : scan edge (search edge 3) 4 [139] 139 root31Tail10 winner =
    scan edge (search edge 3) 4 [139] 139 root31Tail11 winner := by
  rw [root31Tail10, scan, if_neg (by decide), root31Candidates10_checked]
  change scan edge (search edge 3) 4 [139] 139 root31Tail11
    (if 3 ≤ root31Candidates10.length then search edge 3 [139, 1447] root31Candidates10 1586 winner else winner) = _
  rw [if_pos (by decide), root31Child10]
def root31Candidates11 : List Nat := [3253, 5059, 11719, 19819]
set_option maxRecDepth 8192 in
theorem root31Candidates11_checked : candidates edge 4 139 1549 root31Tail12 winner.bound = root31Candidates11 := by
  unfold candidates
  have row : edge 1549 = (SortedNatTree.contains FastRows.row1549) := by
    change Data.row1549.contains = (SortedNatTree.contains FastRows.row1549)
    exact FastRows.row1549_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root31Child11 : search edge 3 [139, 1549] root31Candidates11 1688 winner = winner := by decide
theorem root31Step11 : scan edge (search edge 3) 4 [139] 139 root31Tail11 winner =
    scan edge (search edge 3) 4 [139] 139 root31Tail12 winner := by
  rw [root31Tail11, scan, if_neg (by decide), root31Candidates11_checked]
  change scan edge (search edge 3) 4 [139] 139 root31Tail12
    (if 3 ≤ root31Candidates11.length then search edge 3 [139, 1549] root31Candidates11 1688 winner else winner) = _
  rw [if_pos (by decide), root31Child11]
def root31Candidates12 : List Nat := [8269, 10531]
set_option maxRecDepth 8192 in
theorem root31Candidates12_checked : candidates edge 4 139 1597 root31Tail13 winner.bound = root31Candidates12 := by
  unfold candidates
  have row : edge 1597 = (SortedNatTree.contains FastRows.row1597) := by
    change Data.row1597.contains = (SortedNatTree.contains FastRows.row1597)
    exact FastRows.row1597_checked.symm
  rw [row]
  decide
theorem root31Step12 : scan edge (search edge 3) 4 [139] 139 root31Tail12 winner =
    scan edge (search edge 3) 4 [139] 139 root31Tail13 winner := by
  rw [root31Tail12, scan, if_neg (by decide), root31Candidates12_checked]
  change scan edge (search edge 3) 4 [139] 139 root31Tail13
    (if 3 ≤ root31Candidates12.length then search edge 3 [139, 1597] root31Candidates12 1736 winner else winner) = _
  rw [if_neg (by decide)]
def root31Candidates13 : List Nat := [19387]
set_option maxRecDepth 8192 in
theorem root31Candidates13_checked : candidates edge 4 139 2803 root31Tail14 winner.bound = root31Candidates13 := by
  unfold candidates
  have row : edge 2803 = (SortedNatTree.contains FastRows.row2803) := by
    change Data.row2803.contains = (SortedNatTree.contains FastRows.row2803)
    exact FastRows.row2803_checked.symm
  rw [row]
  decide
theorem root31Step13 : scan edge (search edge 3) 4 [139] 139 root31Tail13 winner =
    scan edge (search edge 3) 4 [139] 139 root31Tail14 winner := by
  rw [root31Tail13, scan, if_neg (by decide), root31Candidates13_checked]
  change scan edge (search edge 3) 4 [139] 139 root31Tail14
    (if 3 ≤ root31Candidates13.length then search edge 3 [139, 2803] root31Candidates13 2942 winner else winner) = _
  rw [if_neg (by decide)]
def root31Candidates14 : List Nat := [6841, 17467]
set_option maxRecDepth 8192 in
theorem root31Candidates14_checked : candidates edge 4 139 3229 root31Tail15 winner.bound = root31Candidates14 := by
  unfold candidates
  have row : edge 3229 = (SortedNatTree.contains FastRows.row3229) := by
    change Data.row3229.contains = (SortedNatTree.contains FastRows.row3229)
    exact FastRows.row3229_checked.symm
  rw [row]
  decide
theorem root31Step14 : scan edge (search edge 3) 4 [139] 139 root31Tail14 winner =
    scan edge (search edge 3) 4 [139] 139 root31Tail15 winner := by
  rw [root31Tail14, scan, if_neg (by decide), root31Candidates14_checked]
  change scan edge (search edge 3) 4 [139] 139 root31Tail15
    (if 3 ≤ root31Candidates14.length then search edge 3 [139, 3229] root31Candidates14 3368 winner else winner) = _
  rw [if_neg (by decide)]
def root31Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root31Candidates15_checked : candidates edge 4 139 3253 root31Tail16 winner.bound = root31Candidates15 := by
  unfold candidates
  have row : edge 3253 = (SortedNatTree.contains FastRows.row3253) := by
    change Data.row3253.contains = (SortedNatTree.contains FastRows.row3253)
    exact FastRows.row3253_checked.symm
  rw [row]
  decide
theorem root31Step15 : scan edge (search edge 3) 4 [139] 139 root31Tail15 winner =
    scan edge (search edge 3) 4 [139] 139 root31Tail16 winner := by
  rw [root31Tail15, scan, if_neg (by decide), root31Candidates15_checked]
  change scan edge (search edge 3) 4 [139] 139 root31Tail16
    (if 3 ≤ root31Candidates15.length then search edge 3 [139, 3253] root31Candidates15 3392 winner else winner) = _
  rw [if_neg (by decide)]
def root31Candidates16 : List Nat := [8779, 12601]
set_option maxRecDepth 8192 in
theorem root31Candidates16_checked : candidates edge 4 139 3313 root31Tail17 winner.bound = root31Candidates16 := by
  unfold candidates
  have row : edge 3313 = (SortedNatTree.contains FastRows.row3313) := by
    change Data.row3313.contains = (SortedNatTree.contains FastRows.row3313)
    exact FastRows.row3313_checked.symm
  rw [row]
  decide
theorem root31Step16 : scan edge (search edge 3) 4 [139] 139 root31Tail16 winner =
    scan edge (search edge 3) 4 [139] 139 root31Tail17 winner := by
  rw [root31Tail16, scan, if_neg (by decide), root31Candidates16_checked]
  change scan edge (search edge 3) 4 [139] 139 root31Tail17
    (if 3 ≤ root31Candidates16.length then search edge 3 [139, 3313] root31Candidates16 3452 winner else winner) = _
  rw [if_neg (by decide)]
def root31Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root31Candidates17_checked : candidates edge 4 139 3697 root31Tail18 winner.bound = root31Candidates17 := by
  unfold candidates
  have row : edge 3697 = (SortedNatTree.contains FastRows.row3697) := by
    change Data.row3697.contains = (SortedNatTree.contains FastRows.row3697)
    exact FastRows.row3697_checked.symm
  rw [row]
  decide
theorem root31Step17 : scan edge (search edge 3) 4 [139] 139 root31Tail17 winner =
    scan edge (search edge 3) 4 [139] 139 root31Tail18 winner := by
  rw [root31Tail17, scan, if_neg (by decide), root31Candidates17_checked]
  change scan edge (search edge 3) 4 [139] 139 root31Tail18
    (if 3 ≤ root31Candidates17.length then search edge 3 [139, 3697] root31Candidates17 3836 winner else winner) = _
  rw [if_neg (by decide)]
def root31Candidates18 : List Nat := [6679, 10993]
set_option maxRecDepth 8192 in
theorem root31Candidates18_checked : candidates edge 4 139 3739 root31Tail19 winner.bound = root31Candidates18 := by
  unfold candidates
  have row : edge 3739 = (SortedNatTree.contains FastRows.row3739) := by
    change Data.row3739.contains = (SortedNatTree.contains FastRows.row3739)
    exact FastRows.row3739_checked.symm
  rw [row]
  decide
theorem root31Step18 : scan edge (search edge 3) 4 [139] 139 root31Tail18 winner =
    scan edge (search edge 3) 4 [139] 139 root31Tail19 winner := by
  rw [root31Tail18, scan, if_neg (by decide), root31Candidates18_checked]
  change scan edge (search edge 3) 4 [139] 139 root31Tail19
    (if 3 ≤ root31Candidates18.length then search edge 3 [139, 3739] root31Candidates18 3878 winner else winner) = _
  rw [if_neg (by decide)]
def root31Candidates19 : List Nat := [5323, 8053]
set_option maxRecDepth 8192 in
theorem root31Candidates19_checked : candidates edge 4 139 4177 root31Tail20 winner.bound = root31Candidates19 := by
  unfold candidates
  have row : edge 4177 = (SortedNatTree.contains FastRows.row4177) := by
    change Data.row4177.contains = (SortedNatTree.contains FastRows.row4177)
    exact FastRows.row4177_checked.symm
  rw [row]
  decide
theorem root31Step19 : scan edge (search edge 3) 4 [139] 139 root31Tail19 winner =
    scan edge (search edge 3) 4 [139] 139 root31Tail20 winner := by
  rw [root31Tail19, scan, if_neg (by decide), root31Candidates19_checked]
  change scan edge (search edge 3) 4 [139] 139 root31Tail20
    (if 3 ≤ root31Candidates19.length then search edge 3 [139, 4177] root31Candidates19 4316 winner else winner) = _
  rw [if_neg (by decide)]
def root31Candidates20 : List Nat := [5323, 8269]
set_option maxRecDepth 8192 in
theorem root31Candidates20_checked : candidates edge 4 139 5059 root31Tail21 winner.bound = root31Candidates20 := by
  unfold candidates
  have row : edge 5059 = (SortedNatTree.contains FastRows.row5059) := by
    change Data.row5059.contains = (SortedNatTree.contains FastRows.row5059)
    exact FastRows.row5059_checked.symm
  rw [row]
  decide
theorem root31Step20 : scan edge (search edge 3) 4 [139] 139 root31Tail20 winner =
    scan edge (search edge 3) 4 [139] 139 root31Tail21 winner := by
  rw [root31Tail20, scan, if_neg (by decide), root31Candidates20_checked]
  change scan edge (search edge 3) 4 [139] 139 root31Tail21
    (if 3 ≤ root31Candidates20.length then search edge 3 [139, 5059] root31Candidates20 5198 winner else winner) = _
  rw [if_neg (by decide)]
def root31Candidates21 : List Nat := [7177, 19051]
set_option maxRecDepth 8192 in
theorem root31Candidates21_checked : candidates edge 4 139 5167 root31Tail22 winner.bound = root31Candidates21 := by
  unfold candidates
  have row : edge 5167 = (SortedNatTree.contains FastRows.row5167) := by
    change Data.row5167.contains = (SortedNatTree.contains FastRows.row5167)
    exact FastRows.row5167_checked.symm
  rw [row]
  decide
theorem root31Step21 : scan edge (search edge 3) 4 [139] 139 root31Tail21 winner =
    scan edge (search edge 3) 4 [139] 139 root31Tail22 winner := by
  rw [root31Tail21, scan, if_neg (by decide), root31Candidates21_checked]
  change scan edge (search edge 3) 4 [139] 139 root31Tail22
    (if 3 ≤ root31Candidates21.length then search edge 3 [139, 5167] root31Candidates21 5306 winner else winner) = _
  rw [if_neg (by decide)]
def root31Candidates22 : List Nat := [19387]
set_option maxRecDepth 8192 in
theorem root31Candidates22_checked : candidates edge 4 139 5323 root31Tail23 winner.bound = root31Candidates22 := by
  unfold candidates
  have row : edge 5323 = (SortedNatTree.contains FastRows.row5323) := by
    change Data.row5323.contains = (SortedNatTree.contains FastRows.row5323)
    exact FastRows.row5323_checked.symm
  rw [row]
  decide
theorem root31Step22 : scan edge (search edge 3) 4 [139] 139 root31Tail22 winner =
    scan edge (search edge 3) 4 [139] 139 root31Tail23 winner := by
  rw [root31Tail22, scan, if_neg (by decide), root31Candidates22_checked]
  change scan edge (search edge 3) 4 [139] 139 root31Tail23
    (if 3 ≤ root31Candidates22.length then search edge 3 [139, 5323] root31Candidates22 5462 winner else winner) = _
  rw [if_neg (by decide)]
def root31Candidates23 : List Nat := []
set_option maxRecDepth 8192 in
theorem root31Candidates23_checked : candidates edge 4 139 5743 root31Tail24 winner.bound = root31Candidates23 := by
  unfold candidates
  have row : edge 5743 = (SortedNatTree.contains FastRows.row5743) := by
    change Data.row5743.contains = (SortedNatTree.contains FastRows.row5743)
    exact FastRows.row5743_checked.symm
  rw [row]
  decide
theorem root31Step23 : scan edge (search edge 3) 4 [139] 139 root31Tail23 winner =
    scan edge (search edge 3) 4 [139] 139 root31Tail24 winner := by
  rw [root31Tail23, scan, if_neg (by decide), root31Candidates23_checked]
  change scan edge (search edge 3) 4 [139] 139 root31Tail24
    (if 3 ≤ root31Candidates23.length then search edge 3 [139, 5743] root31Candidates23 5882 winner else winner) = _
  rw [if_neg (by decide)]
def root31Candidates24 : List Nat := [10993]
set_option maxRecDepth 8192 in
theorem root31Candidates24_checked : candidates edge 4 139 5869 root31Tail25 winner.bound = root31Candidates24 := by
  unfold candidates
  have row : edge 5869 = (SortedNatTree.contains FastRows.row5869) := by
    change Data.row5869.contains = (SortedNatTree.contains FastRows.row5869)
    exact FastRows.row5869_checked.symm
  rw [row]
  decide
theorem root31Step24 : scan edge (search edge 3) 4 [139] 139 root31Tail24 winner =
    scan edge (search edge 3) 4 [139] 139 root31Tail25 winner := by
  rw [root31Tail24, scan, if_neg (by decide), root31Candidates24_checked]
  change scan edge (search edge 3) 4 [139] 139 root31Tail25
    (if 3 ≤ root31Candidates24.length then search edge 3 [139, 5869] root31Candidates24 6008 winner else winner) = _
  rw [if_neg (by decide)]
theorem root31Scan25 : scan edge (search edge 3) 4 [139] 139 root31Tail25 winner = winner := by rw [root31Tail25, scan, if_pos (by decide)]
theorem root31Scan24 : scan edge (search edge 3) 4 [139] 139 root31Tail24 winner = winner :=
  (root31Step24).trans root31Scan25
theorem root31Scan23 : scan edge (search edge 3) 4 [139] 139 root31Tail23 winner = winner :=
  (root31Step23).trans root31Scan24
theorem root31Scan22 : scan edge (search edge 3) 4 [139] 139 root31Tail22 winner = winner :=
  (root31Step22).trans root31Scan23
theorem root31Scan21 : scan edge (search edge 3) 4 [139] 139 root31Tail21 winner = winner :=
  (root31Step21).trans root31Scan22
theorem root31Scan20 : scan edge (search edge 3) 4 [139] 139 root31Tail20 winner = winner :=
  (root31Step20).trans root31Scan21
theorem root31Scan19 : scan edge (search edge 3) 4 [139] 139 root31Tail19 winner = winner :=
  (root31Step19).trans root31Scan20
theorem root31Scan18 : scan edge (search edge 3) 4 [139] 139 root31Tail18 winner = winner :=
  (root31Step18).trans root31Scan19
theorem root31Scan17 : scan edge (search edge 3) 4 [139] 139 root31Tail17 winner = winner :=
  (root31Step17).trans root31Scan18
theorem root31Scan16 : scan edge (search edge 3) 4 [139] 139 root31Tail16 winner = winner :=
  (root31Step16).trans root31Scan17
theorem root31Scan15 : scan edge (search edge 3) 4 [139] 139 root31Tail15 winner = winner :=
  (root31Step15).trans root31Scan16
theorem root31Scan14 : scan edge (search edge 3) 4 [139] 139 root31Tail14 winner = winner :=
  (root31Step14).trans root31Scan15
theorem root31Scan13 : scan edge (search edge 3) 4 [139] 139 root31Tail13 winner = winner :=
  (root31Step13).trans root31Scan14
theorem root31Scan12 : scan edge (search edge 3) 4 [139] 139 root31Tail12 winner = winner :=
  (root31Step12).trans root31Scan13
theorem root31Scan11 : scan edge (search edge 3) 4 [139] 139 root31Tail11 winner = winner :=
  (root31Step11).trans root31Scan12
theorem root31Scan10 : scan edge (search edge 3) 4 [139] 139 root31Tail10 winner = winner :=
  (root31Step10).trans root31Scan11
theorem root31Scan9 : scan edge (search edge 3) 4 [139] 139 root31Tail9 winner = winner :=
  (root31Step9).trans root31Scan10
theorem root31Scan8 : scan edge (search edge 3) 4 [139] 139 root31Tail8 winner = winner :=
  (root31Step8).trans root31Scan9
theorem root31Scan7 : scan edge (search edge 3) 4 [139] 139 root31Tail7 winner = winner :=
  (root31Step7).trans root31Scan8
theorem root31Scan6 : scan edge (search edge 3) 4 [139] 139 root31Tail6 winner = winner :=
  (root31Step6).trans root31Scan7
theorem root31Scan5 : scan edge (search edge 3) 4 [139] 139 root31Tail5 winner = winner :=
  (root31Step5).trans root31Scan6
theorem root31Scan4 : scan edge (search edge 3) 4 [139] 139 root31Tail4 winner = winner :=
  (root31Step4).trans root31Scan5
theorem root31Scan3 : scan edge (search edge 3) 4 [139] 139 root31Tail3 winner = winner :=
  (root31Step3).trans root31Scan4
theorem root31Scan2 : scan edge (search edge 3) 4 [139] 139 root31Tail2 winner = winner :=
  (root31Step2).trans root31Scan3
theorem root31Scan1 : scan edge (search edge 3) 4 [139] 139 root31Tail1 winner = winner :=
  (root31Step1).trans root31Scan2
theorem root31Scan0 : scan edge (search edge 3) 4 [139] 139 root31Tail0 winner = winner :=
  (root31Step0).trans root31Scan1
set_option maxRecDepth 4096 in
theorem root31Branch : search edge 4 [139] root31Candidates 139 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [139] 139 root31Tail0 winner = winner
  exact root31Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
