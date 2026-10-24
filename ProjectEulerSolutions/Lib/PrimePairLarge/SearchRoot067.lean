import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root67Candidates : List Nat := Data.row349
set_option maxRecDepth 8192 in
theorem root67Candidates_checked : candidates edge 5 0 349 tail68 winner.bound = root67Candidates := by
  apply candidates_eq_row edge 5 0 349 tail68 root67Candidates winner.bound tail68_ordered
  · decide
  · decide
  · funext q
    rfl
def root67Tail73 : List Nat := []
def root67Tail72 : List Nat := 25603 :: root67Tail73
def root67Tail71 : List Nat := 25057 :: root67Tail72
def root67Tail70 : List Nat := 24979 :: root67Tail71
def root67Tail69 : List Nat := 24691 :: root67Tail70
def root67Tail68 : List Nat := 24229 :: root67Tail69
def root67Tail67 : List Nat := 23563 :: root67Tail68
def root67Tail66 : List Nat := 23557 :: root67Tail67
def root67Tail65 : List Nat := 22639 :: root67Tail66
def root67Tail64 : List Nat := 21589 :: root67Tail65
def root67Tail63 : List Nat := 21499 :: root67Tail64
def root67Tail62 : List Nat := 21121 :: root67Tail63
def root67Tail61 : List Nat := 20101 :: root67Tail62
def root67Tail60 : List Nat := 18517 :: root67Tail61
def root67Tail59 : List Nat := 18379 :: root67Tail60
def root67Tail58 : List Nat := 18253 :: root67Tail59
def root67Tail57 : List Nat := 18223 :: root67Tail58
def root67Tail56 : List Nat := 18061 :: root67Tail57
def root67Tail55 : List Nat := 17959 :: root67Tail56
def root67Tail54 : List Nat := 17923 :: root67Tail55
def root67Tail53 : List Nat := 17713 :: root67Tail54
def root67Tail52 : List Nat := 17299 :: root67Tail53
def root67Tail51 : List Nat := 17209 :: root67Tail52
def root67Tail50 : List Nat := 16033 :: root67Tail51
def root67Tail49 : List Nat := 15493 :: root67Tail50
def root67Tail48 : List Nat := 14983 :: root67Tail49
def root67Tail47 : List Nat := 14947 :: root67Tail48
def root67Tail46 : List Nat := 14653 :: root67Tail47
def root67Tail45 : List Nat := 14281 :: root67Tail46
def root67Tail44 : List Nat := 14149 :: root67Tail45
def root67Tail43 : List Nat := 14071 :: root67Tail44
def root67Tail42 : List Nat := 13183 :: root67Tail43
def root67Tail41 : List Nat := 12799 :: root67Tail42
def root67Tail40 : List Nat := 12763 :: root67Tail41
def root67Tail39 : List Nat := 12421 :: root67Tail40
def root67Tail38 : List Nat := 11827 :: root67Tail39
def root67Tail37 : List Nat := 11587 :: root67Tail38
def root67Tail36 : List Nat := 11467 :: root67Tail37
def root67Tail35 : List Nat := 11329 :: root67Tail36
def root67Tail34 : List Nat := 10753 :: root67Tail35
def root67Tail33 : List Nat := 10627 :: root67Tail34
def root67Tail32 : List Nat := 10273 :: root67Tail33
def root67Tail31 : List Nat := 9973 :: root67Tail32
def root67Tail30 : List Nat := 9679 :: root67Tail31
def root67Tail29 : List Nat := 8377 :: root67Tail30
def root67Tail28 : List Nat := 8293 :: root67Tail29
def root67Tail27 : List Nat := 6709 :: root67Tail28
def root67Tail26 : List Nat := 6151 :: root67Tail27
def root67Tail25 : List Nat := 6007 :: root67Tail26
def root67Tail24 : List Nat := 5743 :: root67Tail25
def root67Tail23 : List Nat := 5581 :: root67Tail24
def root67Tail22 : List Nat := 4657 :: root67Tail23
def root67Tail21 : List Nat := 4357 :: root67Tail22
def root67Tail20 : List Nat := 4159 :: root67Tail21
def root67Tail19 : List Nat := 3433 :: root67Tail20
def root67Tail18 : List Nat := 2971 :: root67Tail19
def root67Tail17 : List Nat := 2719 :: root67Tail18
def root67Tail16 : List Nat := 2707 :: root67Tail17
def root67Tail15 : List Nat := 2551 :: root67Tail16
def root67Tail14 : List Nat := 2341 :: root67Tail15
def root67Tail13 : List Nat := 2161 :: root67Tail14
def root67Tail12 : List Nat := 1447 :: root67Tail13
def root67Tail11 : List Nat := 1429 :: root67Tail12
def root67Tail10 : List Nat := 1291 :: root67Tail11
def root67Tail9 : List Nat := 1237 :: root67Tail10
def root67Tail8 : List Nat := 1123 :: root67Tail9
def root67Tail7 : List Nat := 1087 :: root67Tail8
def root67Tail6 : List Nat := 967 :: root67Tail7
def root67Tail5 : List Nat := 919 :: root67Tail6
def root67Tail4 : List Nat := 829 :: root67Tail5
def root67Tail3 : List Nat := 709 :: root67Tail4
def root67Tail2 : List Nat := 499 :: root67Tail3
def root67Tail1 : List Nat := 409 :: root67Tail2
def root67Tail0 : List Nat := 373 :: root67Tail1
def root67Candidates0 : List Nat := [1237, 1447, 2161, 14071]
set_option maxRecDepth 8192 in
theorem root67Candidates0_checked : candidates edge 4 349 373 root67Tail1 winner.bound = root67Candidates0 := by
  unfold candidates
  have row : edge 373 = (SortedNatTree.contains FastRows.row373) := by
    change Data.row373.contains = (SortedNatTree.contains FastRows.row373)
    exact FastRows.row373_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root67Child0 : search edge 3 [349, 373] root67Candidates0 722 winner = winner := by decide
theorem root67Step0 : scan edge (search edge 3) 4 [349] 349 root67Tail0 winner =
    scan edge (search edge 3) 4 [349] 349 root67Tail1 winner := by
  rw [root67Tail0, scan, if_neg (by decide), root67Candidates0_checked]
  change scan edge (search edge 3) 4 [349] 349 root67Tail1
    (if 3 ≤ root67Candidates0.length then search edge 3 [349, 373] root67Candidates0 722 winner else winner) = _
  rw [if_pos (by decide), root67Child0]
def root67Candidates1 : List Nat := [709, 13183, 21589]
set_option maxRecDepth 8192 in
theorem root67Candidates1_checked : candidates edge 4 349 409 root67Tail2 winner.bound = root67Candidates1 := by
  unfold candidates
  have row : edge 409 = (SortedNatTree.contains FastRows.row409) := by
    change Data.row409.contains = (SortedNatTree.contains FastRows.row409)
    exact FastRows.row409_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root67Child1 : search edge 3 [349, 409] root67Candidates1 758 winner = winner := by decide
theorem root67Step1 : scan edge (search edge 3) 4 [349] 349 root67Tail1 winner =
    scan edge (search edge 3) 4 [349] 349 root67Tail2 winner := by
  rw [root67Tail1, scan, if_neg (by decide), root67Candidates1_checked]
  change scan edge (search edge 3) 4 [349] 349 root67Tail2
    (if 3 ≤ root67Candidates1.length then search edge 3 [349, 409] root67Candidates1 758 winner else winner) = _
  rw [if_pos (by decide), root67Child1]
def root67Candidates2 : List Nat := [1237, 2551, 2707, 4357, 14281, 14983, 23563]
set_option maxRecDepth 8192 in
theorem root67Candidates2_checked : candidates edge 4 349 499 root67Tail3 winner.bound = root67Candidates2 := by
  unfold candidates
  have row : edge 499 = (SortedNatTree.contains FastRows.row499) := by
    change Data.row499.contains = (SortedNatTree.contains FastRows.row499)
    exact FastRows.row499_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root67Child2 : search edge 3 [349, 499] root67Candidates2 848 winner = winner := by decide
theorem root67Step2 : scan edge (search edge 3) 4 [349] 349 root67Tail2 winner =
    scan edge (search edge 3) 4 [349] 349 root67Tail3 winner := by
  rw [root67Tail2, scan, if_neg (by decide), root67Candidates2_checked]
  change scan edge (search edge 3) 4 [349] 349 root67Tail3
    (if 3 ≤ root67Candidates2.length then search edge 3 [349, 499] root67Candidates2 848 winner else winner) = _
  rw [if_pos (by decide), root67Child2]
def root67Candidates3 : List Nat := [967, 10627, 12421, 17923]
set_option maxRecDepth 8192 in
theorem root67Candidates3_checked : candidates edge 4 349 709 root67Tail4 winner.bound = root67Candidates3 := by
  unfold candidates
  have row : edge 709 = (SortedNatTree.contains FastRows.row709) := by
    change Data.row709.contains = (SortedNatTree.contains FastRows.row709)
    exact FastRows.row709_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root67Child3 : search edge 3 [349, 709] root67Candidates3 1058 winner = winner := by decide
theorem root67Step3 : scan edge (search edge 3) 4 [349] 349 root67Tail3 winner =
    scan edge (search edge 3) 4 [349] 349 root67Tail4 winner := by
  rw [root67Tail3, scan, if_neg (by decide), root67Candidates3_checked]
  change scan edge (search edge 3) 4 [349] 349 root67Tail4
    (if 3 ≤ root67Candidates3.length then search edge 3 [349, 709] root67Candidates3 1058 winner else winner) = _
  rw [if_pos (by decide), root67Child3]
def root67Candidates4 : List Nat := [1237, 10273, 11467, 14071]
set_option maxRecDepth 8192 in
theorem root67Candidates4_checked : candidates edge 4 349 829 root67Tail5 winner.bound = root67Candidates4 := by
  unfold candidates
  have row : edge 829 = (SortedNatTree.contains FastRows.row829) := by
    change Data.row829.contains = (SortedNatTree.contains FastRows.row829)
    exact FastRows.row829_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root67Child4 : search edge 3 [349, 829] root67Candidates4 1178 winner = winner := by decide
theorem root67Step4 : scan edge (search edge 3) 4 [349] 349 root67Tail4 winner =
    scan edge (search edge 3) 4 [349] 349 root67Tail5 winner := by
  rw [root67Tail4, scan, if_neg (by decide), root67Candidates4_checked]
  change scan edge (search edge 3) 4 [349] 349 root67Tail5
    (if 3 ≤ root67Candidates4.length then search edge 3 [349, 829] root67Candidates4 1178 winner else winner) = _
  rw [if_pos (by decide), root67Child4]
def root67Candidates5 : List Nat := [11467, 17209]
set_option maxRecDepth 8192 in
theorem root67Candidates5_checked : candidates edge 4 349 919 root67Tail6 winner.bound = root67Candidates5 := by
  unfold candidates
  have row : edge 919 = (SortedNatTree.contains FastRows.row919) := by
    change Data.row919.contains = (SortedNatTree.contains FastRows.row919)
    exact FastRows.row919_checked.symm
  rw [row]
  decide
theorem root67Step5 : scan edge (search edge 3) 4 [349] 349 root67Tail5 winner =
    scan edge (search edge 3) 4 [349] 349 root67Tail6 winner := by
  rw [root67Tail5, scan, if_neg (by decide), root67Candidates5_checked]
  change scan edge (search edge 3) 4 [349] 349 root67Tail6
    (if 3 ≤ root67Candidates5.length then search edge 3 [349, 919] root67Candidates5 1268 winner else winner) = _
  rw [if_neg (by decide)]
def root67Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root67Candidates6_checked : candidates edge 4 349 967 root67Tail7 winner.bound = root67Candidates6 := by
  unfold candidates
  have row : edge 967 = (SortedNatTree.contains FastRows.row967) := by
    change Data.row967.contains = (SortedNatTree.contains FastRows.row967)
    exact FastRows.row967_checked.symm
  rw [row]
  decide
theorem root67Step6 : scan edge (search edge 3) 4 [349] 349 root67Tail6 winner =
    scan edge (search edge 3) 4 [349] 349 root67Tail7 winner := by
  rw [root67Tail6, scan, if_neg (by decide), root67Candidates6_checked]
  change scan edge (search edge 3) 4 [349] 349 root67Tail7
    (if 3 ≤ root67Candidates6.length then search edge 3 [349, 967] root67Candidates6 1316 winner else winner) = _
  rw [if_neg (by decide)]
def root67Candidates7 : List Nat := [21499]
set_option maxRecDepth 8192 in
theorem root67Candidates7_checked : candidates edge 4 349 1087 root67Tail8 winner.bound = root67Candidates7 := by
  unfold candidates
  have row : edge 1087 = (SortedNatTree.contains FastRows.row1087) := by
    change Data.row1087.contains = (SortedNatTree.contains FastRows.row1087)
    exact FastRows.row1087_checked.symm
  rw [row]
  decide
theorem root67Step7 : scan edge (search edge 3) 4 [349] 349 root67Tail7 winner =
    scan edge (search edge 3) 4 [349] 349 root67Tail8 winner := by
  rw [root67Tail7, scan, if_neg (by decide), root67Candidates7_checked]
  change scan edge (search edge 3) 4 [349] 349 root67Tail8
    (if 3 ≤ root67Candidates7.length then search edge 3 [349, 1087] root67Candidates7 1436 winner else winner) = _
  rw [if_neg (by decide)]
def root67Candidates8 : List Nat := [3433, 16033]
set_option maxRecDepth 8192 in
theorem root67Candidates8_checked : candidates edge 4 349 1123 root67Tail9 winner.bound = root67Candidates8 := by
  unfold candidates
  have row : edge 1123 = (SortedNatTree.contains FastRows.row1123) := by
    change Data.row1123.contains = (SortedNatTree.contains FastRows.row1123)
    exact FastRows.row1123_checked.symm
  rw [row]
  decide
theorem root67Step8 : scan edge (search edge 3) 4 [349] 349 root67Tail8 winner =
    scan edge (search edge 3) 4 [349] 349 root67Tail9 winner := by
  rw [root67Tail8, scan, if_neg (by decide), root67Candidates8_checked]
  change scan edge (search edge 3) 4 [349] 349 root67Tail9
    (if 3 ≤ root67Candidates8.length then search edge 3 [349, 1123] root67Candidates8 1472 winner else winner) = _
  rw [if_neg (by decide)]
def root67Candidates9 : List Nat := [2341, 5743, 9973, 14653, 22639]
set_option maxRecDepth 8192 in
theorem root67Candidates9_checked : candidates edge 4 349 1237 root67Tail10 winner.bound = root67Candidates9 := by
  unfold candidates
  have row : edge 1237 = (SortedNatTree.contains FastRows.row1237) := by
    change Data.row1237.contains = (SortedNatTree.contains FastRows.row1237)
    exact FastRows.row1237_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root67Child9 : search edge 3 [349, 1237] root67Candidates9 1586 winner = winner := by decide
theorem root67Step9 : scan edge (search edge 3) 4 [349] 349 root67Tail9 winner =
    scan edge (search edge 3) 4 [349] 349 root67Tail10 winner := by
  rw [root67Tail9, scan, if_neg (by decide), root67Candidates9_checked]
  change scan edge (search edge 3) 4 [349] 349 root67Tail10
    (if 3 ≤ root67Candidates9.length then search edge 3 [349, 1237] root67Candidates9 1586 winner else winner) = _
  rw [if_pos (by decide), root67Child9]
def root67Candidates10 : List Nat := [8293]
set_option maxRecDepth 8192 in
theorem root67Candidates10_checked : candidates edge 4 349 1291 root67Tail11 winner.bound = root67Candidates10 := by
  unfold candidates
  have row : edge 1291 = (SortedNatTree.contains FastRows.row1291) := by
    change Data.row1291.contains = (SortedNatTree.contains FastRows.row1291)
    exact FastRows.row1291_checked.symm
  rw [row]
  decide
theorem root67Step10 : scan edge (search edge 3) 4 [349] 349 root67Tail10 winner =
    scan edge (search edge 3) 4 [349] 349 root67Tail11 winner := by
  rw [root67Tail10, scan, if_neg (by decide), root67Candidates10_checked]
  change scan edge (search edge 3) 4 [349] 349 root67Tail11
    (if 3 ≤ root67Candidates10.length then search edge 3 [349, 1291] root67Candidates10 1640 winner else winner) = _
  rw [if_neg (by decide)]
def root67Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root67Candidates11_checked : candidates edge 4 349 1429 root67Tail12 winner.bound = root67Candidates11 := by
  unfold candidates
  have row : edge 1429 = (SortedNatTree.contains FastRows.row1429) := by
    change Data.row1429.contains = (SortedNatTree.contains FastRows.row1429)
    exact FastRows.row1429_checked.symm
  rw [row]
  decide
theorem root67Step11 : scan edge (search edge 3) 4 [349] 349 root67Tail11 winner =
    scan edge (search edge 3) 4 [349] 349 root67Tail12 winner := by
  rw [root67Tail11, scan, if_neg (by decide), root67Candidates11_checked]
  change scan edge (search edge 3) 4 [349] 349 root67Tail12
    (if 3 ≤ root67Candidates11.length then search edge 3 [349, 1429] root67Candidates11 1778 winner else winner) = _
  rw [if_neg (by decide)]
def root67Candidates12 : List Nat := [2719, 18517]
set_option maxRecDepth 8192 in
theorem root67Candidates12_checked : candidates edge 4 349 1447 root67Tail13 winner.bound = root67Candidates12 := by
  unfold candidates
  have row : edge 1447 = (SortedNatTree.contains FastRows.row1447) := by
    change Data.row1447.contains = (SortedNatTree.contains FastRows.row1447)
    exact FastRows.row1447_checked.symm
  rw [row]
  decide
theorem root67Step12 : scan edge (search edge 3) 4 [349] 349 root67Tail12 winner =
    scan edge (search edge 3) 4 [349] 349 root67Tail13 winner := by
  rw [root67Tail12, scan, if_neg (by decide), root67Candidates12_checked]
  change scan edge (search edge 3) 4 [349] 349 root67Tail13
    (if 3 ≤ root67Candidates12.length then search edge 3 [349, 1447] root67Candidates12 1796 winner else winner) = _
  rw [if_neg (by decide)]
def root67Candidates13 : List Nat := [4357, 22639]
set_option maxRecDepth 8192 in
theorem root67Candidates13_checked : candidates edge 4 349 2161 root67Tail14 winner.bound = root67Candidates13 := by
  unfold candidates
  have row : edge 2161 = (SortedNatTree.contains FastRows.row2161) := by
    change Data.row2161.contains = (SortedNatTree.contains FastRows.row2161)
    exact FastRows.row2161_checked.symm
  rw [row]
  decide
theorem root67Step13 : scan edge (search edge 3) 4 [349] 349 root67Tail13 winner =
    scan edge (search edge 3) 4 [349] 349 root67Tail14 winner := by
  rw [root67Tail13, scan, if_neg (by decide), root67Candidates13_checked]
  change scan edge (search edge 3) 4 [349] 349 root67Tail14
    (if 3 ≤ root67Candidates13.length then search edge 3 [349, 2161] root67Candidates13 2510 winner else winner) = _
  rw [if_neg (by decide)]
def root67Candidates14 : List Nat := [8377, 9679]
set_option maxRecDepth 8192 in
theorem root67Candidates14_checked : candidates edge 4 349 2341 root67Tail15 winner.bound = root67Candidates14 := by
  unfold candidates
  have row : edge 2341 = (SortedNatTree.contains FastRows.row2341) := by
    change Data.row2341.contains = (SortedNatTree.contains FastRows.row2341)
    exact FastRows.row2341_checked.symm
  rw [row]
  decide
theorem root67Step14 : scan edge (search edge 3) 4 [349] 349 root67Tail14 winner =
    scan edge (search edge 3) 4 [349] 349 root67Tail15 winner := by
  rw [root67Tail14, scan, if_neg (by decide), root67Candidates14_checked]
  change scan edge (search edge 3) 4 [349] 349 root67Tail15
    (if 3 ≤ root67Candidates14.length then search edge 3 [349, 2341] root67Candidates14 2690 winner else winner) = _
  rw [if_neg (by decide)]
def root67Candidates15 : List Nat := [12763, 12799, 20101]
set_option maxRecDepth 8192 in
theorem root67Candidates15_checked : candidates edge 4 349 2551 root67Tail16 winner.bound = root67Candidates15 := by
  unfold candidates
  have row : edge 2551 = (SortedNatTree.contains FastRows.row2551) := by
    change Data.row2551.contains = (SortedNatTree.contains FastRows.row2551)
    exact FastRows.row2551_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root67Child15 : search edge 3 [349, 2551] root67Candidates15 2900 winner = winner := by decide
theorem root67Step15 : scan edge (search edge 3) 4 [349] 349 root67Tail15 winner =
    scan edge (search edge 3) 4 [349] 349 root67Tail16 winner := by
  rw [root67Tail15, scan, if_neg (by decide), root67Candidates15_checked]
  change scan edge (search edge 3) 4 [349] 349 root67Tail16
    (if 3 ≤ root67Candidates15.length then search edge 3 [349, 2551] root67Candidates15 2900 winner else winner) = _
  rw [if_pos (by decide), root67Child15]
def root67Candidates16 : List Nat := [9973, 14281]
set_option maxRecDepth 8192 in
theorem root67Candidates16_checked : candidates edge 4 349 2707 root67Tail17 winner.bound = root67Candidates16 := by
  unfold candidates
  have row : edge 2707 = (SortedNatTree.contains FastRows.row2707) := by
    change Data.row2707.contains = (SortedNatTree.contains FastRows.row2707)
    exact FastRows.row2707_checked.symm
  rw [row]
  decide
theorem root67Step16 : scan edge (search edge 3) 4 [349] 349 root67Tail16 winner =
    scan edge (search edge 3) 4 [349] 349 root67Tail17 winner := by
  rw [root67Tail16, scan, if_neg (by decide), root67Candidates16_checked]
  change scan edge (search edge 3) 4 [349] 349 root67Tail17
    (if 3 ≤ root67Candidates16.length then search edge 3 [349, 2707] root67Candidates16 3056 winner else winner) = _
  rw [if_neg (by decide)]
def root67Candidates17 : List Nat := [17299, 17713]
set_option maxRecDepth 8192 in
theorem root67Candidates17_checked : candidates edge 4 349 2719 root67Tail18 winner.bound = root67Candidates17 := by
  unfold candidates
  have row : edge 2719 = (SortedNatTree.contains FastRows.row2719) := by
    change Data.row2719.contains = (SortedNatTree.contains FastRows.row2719)
    exact FastRows.row2719_checked.symm
  rw [row]
  decide
theorem root67Step17 : scan edge (search edge 3) 4 [349] 349 root67Tail17 winner =
    scan edge (search edge 3) 4 [349] 349 root67Tail18 winner := by
  rw [root67Tail17, scan, if_neg (by decide), root67Candidates17_checked]
  change scan edge (search edge 3) 4 [349] 349 root67Tail18
    (if 3 ≤ root67Candidates17.length then search edge 3 [349, 2719] root67Candidates17 3068 winner else winner) = _
  rw [if_neg (by decide)]
def root67Candidates18 : List Nat := [12421, 17713]
set_option maxRecDepth 8192 in
theorem root67Candidates18_checked : candidates edge 4 349 2971 root67Tail19 winner.bound = root67Candidates18 := by
  unfold candidates
  have row : edge 2971 = (SortedNatTree.contains FastRows.row2971) := by
    change Data.row2971.contains = (SortedNatTree.contains FastRows.row2971)
    exact FastRows.row2971_checked.symm
  rw [row]
  decide
theorem root67Step18 : scan edge (search edge 3) 4 [349] 349 root67Tail18 winner =
    scan edge (search edge 3) 4 [349] 349 root67Tail19 winner := by
  rw [root67Tail18, scan, if_neg (by decide), root67Candidates18_checked]
  change scan edge (search edge 3) 4 [349] 349 root67Tail19
    (if 3 ≤ root67Candidates18.length then search edge 3 [349, 2971] root67Candidates18 3320 winner else winner) = _
  rw [if_neg (by decide)]
def root67Candidates19 : List Nat := []
set_option maxRecDepth 8192 in
theorem root67Candidates19_checked : candidates edge 4 349 3433 root67Tail20 winner.bound = root67Candidates19 := by
  unfold candidates
  have row : edge 3433 = (SortedNatTree.contains FastRows.row3433) := by
    change Data.row3433.contains = (SortedNatTree.contains FastRows.row3433)
    exact FastRows.row3433_checked.symm
  rw [row]
  decide
theorem root67Step19 : scan edge (search edge 3) 4 [349] 349 root67Tail19 winner =
    scan edge (search edge 3) 4 [349] 349 root67Tail20 winner := by
  rw [root67Tail19, scan, if_neg (by decide), root67Candidates19_checked]
  change scan edge (search edge 3) 4 [349] 349 root67Tail20
    (if 3 ≤ root67Candidates19.length then search edge 3 [349, 3433] root67Candidates19 3782 winner else winner) = _
  rw [if_neg (by decide)]
def root67Candidates20 : List Nat := [17209]
set_option maxRecDepth 8192 in
theorem root67Candidates20_checked : candidates edge 4 349 4159 root67Tail21 winner.bound = root67Candidates20 := by
  unfold candidates
  have row : edge 4159 = (SortedNatTree.contains FastRows.row4159) := by
    change Data.row4159.contains = (SortedNatTree.contains FastRows.row4159)
    exact FastRows.row4159_checked.symm
  rw [row]
  decide
theorem root67Step20 : scan edge (search edge 3) 4 [349] 349 root67Tail20 winner =
    scan edge (search edge 3) 4 [349] 349 root67Tail21 winner := by
  rw [root67Tail20, scan, if_neg (by decide), root67Candidates20_checked]
  change scan edge (search edge 3) 4 [349] 349 root67Tail21
    (if 3 ≤ root67Candidates20.length then search edge 3 [349, 4159] root67Candidates20 4508 winner else winner) = _
  rw [if_neg (by decide)]
def root67Candidates21 : List Nat := [12799, 13183, 17959]
set_option maxRecDepth 8192 in
theorem root67Candidates21_checked : candidates edge 4 349 4357 root67Tail22 winner.bound = root67Candidates21 := by
  unfold candidates
  have row : edge 4357 = (SortedNatTree.contains FastRows.row4357) := by
    change Data.row4357.contains = (SortedNatTree.contains FastRows.row4357)
    exact FastRows.row4357_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root67Child21 : search edge 3 [349, 4357] root67Candidates21 4706 winner = winner := by decide
theorem root67Step21 : scan edge (search edge 3) 4 [349] 349 root67Tail21 winner =
    scan edge (search edge 3) 4 [349] 349 root67Tail22 winner := by
  rw [root67Tail21, scan, if_neg (by decide), root67Candidates21_checked]
  change scan edge (search edge 3) 4 [349] 349 root67Tail22
    (if 3 ≤ root67Candidates21.length then search edge 3 [349, 4357] root67Candidates21 4706 winner else winner) = _
  rw [if_pos (by decide), root67Child21]
def root67Candidates22 : List Nat := []
set_option maxRecDepth 8192 in
theorem root67Candidates22_checked : candidates edge 4 349 4657 root67Tail23 winner.bound = root67Candidates22 := by
  unfold candidates
  have row : edge 4657 = (SortedNatTree.contains FastRows.row4657) := by
    change Data.row4657.contains = (SortedNatTree.contains FastRows.row4657)
    exact FastRows.row4657_checked.symm
  rw [row]
  decide
theorem root67Step22 : scan edge (search edge 3) 4 [349] 349 root67Tail22 winner =
    scan edge (search edge 3) 4 [349] 349 root67Tail23 winner := by
  rw [root67Tail22, scan, if_neg (by decide), root67Candidates22_checked]
  change scan edge (search edge 3) 4 [349] 349 root67Tail23
    (if 3 ≤ root67Candidates22.length then search edge 3 [349, 4657] root67Candidates22 5006 winner else winner) = _
  rw [if_neg (by decide)]
def root67Candidates23 : List Nat := [18061]
set_option maxRecDepth 8192 in
theorem root67Candidates23_checked : candidates edge 4 349 5581 root67Tail24 winner.bound = root67Candidates23 := by
  unfold candidates
  have row : edge 5581 = (SortedNatTree.contains FastRows.row5581) := by
    change Data.row5581.contains = (SortedNatTree.contains FastRows.row5581)
    exact FastRows.row5581_checked.symm
  rw [row]
  decide
theorem root67Step23 : scan edge (search edge 3) 4 [349] 349 root67Tail23 winner =
    scan edge (search edge 3) 4 [349] 349 root67Tail24 winner := by
  rw [root67Tail23, scan, if_neg (by decide), root67Candidates23_checked]
  change scan edge (search edge 3) 4 [349] 349 root67Tail24
    (if 3 ≤ root67Candidates23.length then search edge 3 [349, 5581] root67Candidates23 5930 winner else winner) = _
  rw [if_neg (by decide)]
def root67Candidates24 : List Nat := [8377, 12799, 17713]
set_option maxRecDepth 8192 in
theorem root67Candidates24_checked : candidates edge 4 349 5743 root67Tail25 winner.bound = root67Candidates24 := by
  unfold candidates
  have row : edge 5743 = (SortedNatTree.contains FastRows.row5743) := by
    change Data.row5743.contains = (SortedNatTree.contains FastRows.row5743)
    exact FastRows.row5743_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root67Child24 : search edge 3 [349, 5743] root67Candidates24 6092 winner = winner := by decide
theorem root67Step24 : scan edge (search edge 3) 4 [349] 349 root67Tail24 winner =
    scan edge (search edge 3) 4 [349] 349 root67Tail25 winner := by
  rw [root67Tail24, scan, if_neg (by decide), root67Candidates24_checked]
  change scan edge (search edge 3) 4 [349] 349 root67Tail25
    (if 3 ≤ root67Candidates24.length then search edge 3 [349, 5743] root67Candidates24 6092 winner else winner) = _
  rw [if_pos (by decide), root67Child24]
def root67Candidates25 : List Nat := [14281]
set_option maxRecDepth 8192 in
theorem root67Candidates25_checked : candidates edge 4 349 6007 root67Tail26 winner.bound = root67Candidates25 := by
  unfold candidates
  have row : edge 6007 = (SortedNatTree.contains FastRows.row6007) := by
    change Data.row6007.contains = (SortedNatTree.contains FastRows.row6007)
    exact FastRows.row6007_checked.symm
  rw [row]
  decide
theorem root67Step25 : scan edge (search edge 3) 4 [349] 349 root67Tail25 winner =
    scan edge (search edge 3) 4 [349] 349 root67Tail26 winner := by
  rw [root67Tail25, scan, if_neg (by decide), root67Candidates25_checked]
  change scan edge (search edge 3) 4 [349] 349 root67Tail26
    (if 3 ≤ root67Candidates25.length then search edge 3 [349, 6007] root67Candidates25 6356 winner else winner) = _
  rw [if_neg (by decide)]
def root67Candidates26 : List Nat := [12763, 15493]
set_option maxRecDepth 8192 in
theorem root67Candidates26_checked : candidates edge 4 349 6151 root67Tail27 winner.bound = root67Candidates26 := by
  unfold candidates
  have row : edge 6151 = (SortedNatTree.contains FastRows.row6151) := by
    change Data.row6151.contains = (SortedNatTree.contains FastRows.row6151)
    exact FastRows.row6151_checked.symm
  rw [row]
  decide
theorem root67Step26 : scan edge (search edge 3) 4 [349] 349 root67Tail26 winner =
    scan edge (search edge 3) 4 [349] 349 root67Tail27 winner := by
  rw [root67Tail26, scan, if_neg (by decide), root67Candidates26_checked]
  change scan edge (search edge 3) 4 [349] 349 root67Tail27
    (if 3 ≤ root67Candidates26.length then search edge 3 [349, 6151] root67Candidates26 6500 winner else winner) = _
  rw [if_neg (by decide)]
theorem root67Scan27 : scan edge (search edge 3) 4 [349] 349 root67Tail27 winner = winner := by rw [root67Tail27, scan, if_pos (by decide)]
theorem root67Scan26 : scan edge (search edge 3) 4 [349] 349 root67Tail26 winner = winner :=
  (root67Step26).trans root67Scan27
theorem root67Scan25 : scan edge (search edge 3) 4 [349] 349 root67Tail25 winner = winner :=
  (root67Step25).trans root67Scan26
theorem root67Scan24 : scan edge (search edge 3) 4 [349] 349 root67Tail24 winner = winner :=
  (root67Step24).trans root67Scan25
theorem root67Scan23 : scan edge (search edge 3) 4 [349] 349 root67Tail23 winner = winner :=
  (root67Step23).trans root67Scan24
theorem root67Scan22 : scan edge (search edge 3) 4 [349] 349 root67Tail22 winner = winner :=
  (root67Step22).trans root67Scan23
theorem root67Scan21 : scan edge (search edge 3) 4 [349] 349 root67Tail21 winner = winner :=
  (root67Step21).trans root67Scan22
theorem root67Scan20 : scan edge (search edge 3) 4 [349] 349 root67Tail20 winner = winner :=
  (root67Step20).trans root67Scan21
theorem root67Scan19 : scan edge (search edge 3) 4 [349] 349 root67Tail19 winner = winner :=
  (root67Step19).trans root67Scan20
theorem root67Scan18 : scan edge (search edge 3) 4 [349] 349 root67Tail18 winner = winner :=
  (root67Step18).trans root67Scan19
theorem root67Scan17 : scan edge (search edge 3) 4 [349] 349 root67Tail17 winner = winner :=
  (root67Step17).trans root67Scan18
theorem root67Scan16 : scan edge (search edge 3) 4 [349] 349 root67Tail16 winner = winner :=
  (root67Step16).trans root67Scan17
theorem root67Scan15 : scan edge (search edge 3) 4 [349] 349 root67Tail15 winner = winner :=
  (root67Step15).trans root67Scan16
theorem root67Scan14 : scan edge (search edge 3) 4 [349] 349 root67Tail14 winner = winner :=
  (root67Step14).trans root67Scan15
theorem root67Scan13 : scan edge (search edge 3) 4 [349] 349 root67Tail13 winner = winner :=
  (root67Step13).trans root67Scan14
theorem root67Scan12 : scan edge (search edge 3) 4 [349] 349 root67Tail12 winner = winner :=
  (root67Step12).trans root67Scan13
theorem root67Scan11 : scan edge (search edge 3) 4 [349] 349 root67Tail11 winner = winner :=
  (root67Step11).trans root67Scan12
theorem root67Scan10 : scan edge (search edge 3) 4 [349] 349 root67Tail10 winner = winner :=
  (root67Step10).trans root67Scan11
theorem root67Scan9 : scan edge (search edge 3) 4 [349] 349 root67Tail9 winner = winner :=
  (root67Step9).trans root67Scan10
theorem root67Scan8 : scan edge (search edge 3) 4 [349] 349 root67Tail8 winner = winner :=
  (root67Step8).trans root67Scan9
theorem root67Scan7 : scan edge (search edge 3) 4 [349] 349 root67Tail7 winner = winner :=
  (root67Step7).trans root67Scan8
theorem root67Scan6 : scan edge (search edge 3) 4 [349] 349 root67Tail6 winner = winner :=
  (root67Step6).trans root67Scan7
theorem root67Scan5 : scan edge (search edge 3) 4 [349] 349 root67Tail5 winner = winner :=
  (root67Step5).trans root67Scan6
theorem root67Scan4 : scan edge (search edge 3) 4 [349] 349 root67Tail4 winner = winner :=
  (root67Step4).trans root67Scan5
theorem root67Scan3 : scan edge (search edge 3) 4 [349] 349 root67Tail3 winner = winner :=
  (root67Step3).trans root67Scan4
theorem root67Scan2 : scan edge (search edge 3) 4 [349] 349 root67Tail2 winner = winner :=
  (root67Step2).trans root67Scan3
theorem root67Scan1 : scan edge (search edge 3) 4 [349] 349 root67Tail1 winner = winner :=
  (root67Step1).trans root67Scan2
theorem root67Scan0 : scan edge (search edge 3) 4 [349] 349 root67Tail0 winner = winner :=
  (root67Step0).trans root67Scan1
set_option maxRecDepth 4096 in
theorem root67Branch : search edge 4 [349] root67Candidates 349 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [349] 349 root67Tail0 winner = winner
  exact root67Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
