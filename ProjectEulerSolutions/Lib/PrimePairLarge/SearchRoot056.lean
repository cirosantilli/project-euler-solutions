import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root56Candidates : List Nat := Data.row277
set_option maxRecDepth 8192 in
theorem root56Candidates_checked : candidates edge 5 0 277 tail57 winner.bound = root56Candidates := by
  apply candidates_eq_row edge 5 0 277 tail57 root56Candidates winner.bound tail57_ordered
  · decide
  · decide
  · funext q
    rfl
def root56Tail65 : List Nat := []
def root56Tail64 : List Nat := 25447 :: root56Tail65
def root56Tail63 : List Nat := 24967 :: root56Tail64
def root56Tail62 : List Nat := 24547 :: root56Tail63
def root56Tail61 : List Nat := 24373 :: root56Tail62
def root56Tail60 : List Nat := 24019 :: root56Tail61
def root56Tail59 : List Nat := 23893 :: root56Tail60
def root56Tail58 : List Nat := 23539 :: root56Tail59
def root56Tail57 : List Nat := 23497 :: root56Tail58
def root56Tail56 : List Nat := 23431 :: root56Tail57
def root56Tail55 : List Nat := 23197 :: root56Tail56
def root56Tail54 : List Nat := 22993 :: root56Tail55
def root56Tail53 : List Nat := 19717 :: root56Tail54
def root56Tail52 : List Nat := 19381 :: root56Tail53
def root56Tail51 : List Nat := 18913 :: root56Tail52
def root56Tail50 : List Nat := 18367 :: root56Tail51
def root56Tail49 : List Nat := 17509 :: root56Tail50
def root56Tail48 : List Nat := 17383 :: root56Tail49
def root56Tail47 : List Nat := 17341 :: root56Tail48
def root56Tail46 : List Nat := 17299 :: root56Tail47
def root56Tail45 : List Nat := 16603 :: root56Tail46
def root56Tail44 : List Nat := 16453 :: root56Tail45
def root56Tail43 : List Nat := 14419 :: root56Tail44
def root56Tail42 : List Nat := 13831 :: root56Tail43
def root56Tail41 : List Nat := 12409 :: root56Tail42
def root56Tail40 : List Nat := 12073 :: root56Tail41
def root56Tail39 : List Nat := 11689 :: root56Tail40
def root56Tail38 : List Nat := 11503 :: root56Tail39
def root56Tail37 : List Nat := 10939 :: root56Tail38
def root56Tail36 : List Nat := 10831 :: root56Tail37
def root56Tail35 : List Nat := 10687 :: root56Tail36
def root56Tail34 : List Nat := 10429 :: root56Tail35
def root56Tail33 : List Nat := 10321 :: root56Tail34
def root56Tail32 : List Nat := 10069 :: root56Tail33
def root56Tail31 : List Nat := 9859 :: root56Tail32
def root56Tail30 : List Nat := 9613 :: root56Tail31
def root56Tail29 : List Nat := 9001 :: root56Tail30
def root56Tail28 : List Nat := 8647 :: root56Tail29
def root56Tail27 : List Nat := 8539 :: root56Tail28
def root56Tail26 : List Nat := 7993 :: root56Tail27
def root56Tail25 : List Nat := 7741 :: root56Tail26
def root56Tail24 : List Nat := 7213 :: root56Tail25
def root56Tail23 : List Nat := 6841 :: root56Tail24
def root56Tail22 : List Nat := 6427 :: root56Tail23
def root56Tail21 : List Nat := 6343 :: root56Tail22
def root56Tail20 : List Nat := 5431 :: root56Tail21
def root56Tail19 : List Nat := 5323 :: root56Tail20
def root56Tail18 : List Nat := 5119 :: root56Tail19
def root56Tail17 : List Nat := 4483 :: root56Tail18
def root56Tail16 : List Nat := 4327 :: root56Tail17
def root56Tail15 : List Nat := 3919 :: root56Tail16
def root56Tail14 : List Nat := 3307 :: root56Tail15
def root56Tail13 : List Nat := 3079 :: root56Tail14
def root56Tail12 : List Nat := 3037 :: root56Tail13
def root56Tail11 : List Nat := 2593 :: root56Tail12
def root56Tail10 : List Nat := 2437 :: root56Tail11
def root56Tail9 : List Nat := 2281 :: root56Tail10
def root56Tail8 : List Nat := 1933 :: root56Tail9
def root56Tail7 : List Nat := 1693 :: root56Tail8
def root56Tail6 : List Nat := 1471 :: root56Tail7
def root56Tail5 : List Nat := 1381 :: root56Tail6
def root56Tail4 : List Nat := 859 :: root56Tail5
def root56Tail3 : List Nat := 787 :: root56Tail4
def root56Tail2 : List Nat := 751 :: root56Tail3
def root56Tail1 : List Nat := 499 :: root56Tail2
def root56Tail0 : List Nat := 331 :: root56Tail1
def root56Candidates0 : List Nat := [1933, 6427, 10321, 10687, 24373]
set_option maxRecDepth 8192 in
theorem root56Candidates0_checked : candidates edge 4 277 331 root56Tail1 winner.bound = root56Candidates0 := by
  unfold candidates
  have row : edge 331 = (SortedNatTree.contains FastRows.row331) := by
    change Data.row331.contains = (SortedNatTree.contains FastRows.row331)
    exact FastRows.row331_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root56Child0 : search edge 3 [277, 331] root56Candidates0 608 winner = winner := by decide
theorem root56Step0 : scan edge (search edge 3) 4 [277] 277 root56Tail0 winner =
    scan edge (search edge 3) 4 [277] 277 root56Tail1 winner := by
  rw [root56Tail0, scan, if_neg (by decide), root56Candidates0_checked]
  change scan edge (search edge 3) 4 [277] 277 root56Tail1
    (if 3 ≤ root56Candidates0.length then search edge 3 [277, 331] root56Candidates0 608 winner else winner) = _
  rw [if_pos (by decide), root56Child0]
def root56Candidates1 : List Nat := [1933, 24373]
set_option maxRecDepth 8192 in
theorem root56Candidates1_checked : candidates edge 4 277 499 root56Tail2 winner.bound = root56Candidates1 := by
  unfold candidates
  have row : edge 499 = (SortedNatTree.contains FastRows.row499) := by
    change Data.row499.contains = (SortedNatTree.contains FastRows.row499)
    exact FastRows.row499_checked.symm
  rw [row]
  decide
theorem root56Step1 : scan edge (search edge 3) 4 [277] 277 root56Tail1 winner =
    scan edge (search edge 3) 4 [277] 277 root56Tail2 winner := by
  rw [root56Tail1, scan, if_neg (by decide), root56Candidates1_checked]
  change scan edge (search edge 3) 4 [277] 277 root56Tail2
    (if 3 ≤ root56Candidates1.length then search edge 3 [277, 499] root56Candidates1 776 winner else winner) = _
  rw [if_neg (by decide)]
def root56Candidates2 : List Nat := [787, 2437, 11689]
set_option maxRecDepth 8192 in
theorem root56Candidates2_checked : candidates edge 4 277 751 root56Tail3 winner.bound = root56Candidates2 := by
  unfold candidates
  have row : edge 751 = (SortedNatTree.contains FastRows.row751) := by
    change Data.row751.contains = (SortedNatTree.contains FastRows.row751)
    exact FastRows.row751_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root56Child2 : search edge 3 [277, 751] root56Candidates2 1028 winner = winner := by decide
theorem root56Step2 : scan edge (search edge 3) 4 [277] 277 root56Tail2 winner =
    scan edge (search edge 3) 4 [277] 277 root56Tail3 winner := by
  rw [root56Tail2, scan, if_neg (by decide), root56Candidates2_checked]
  change scan edge (search edge 3) 4 [277] 277 root56Tail3
    (if 3 ≤ root56Candidates2.length then search edge 3 [277, 751] root56Candidates2 1028 winner else winner) = _
  rw [if_pos (by decide), root56Child2]
def root56Candidates3 : List Nat := [7993, 17341]
set_option maxRecDepth 8192 in
theorem root56Candidates3_checked : candidates edge 4 277 787 root56Tail4 winner.bound = root56Candidates3 := by
  unfold candidates
  have row : edge 787 = (SortedNatTree.contains FastRows.row787) := by
    change Data.row787.contains = (SortedNatTree.contains FastRows.row787)
    exact FastRows.row787_checked.symm
  rw [row]
  decide
theorem root56Step3 : scan edge (search edge 3) 4 [277] 277 root56Tail3 winner =
    scan edge (search edge 3) 4 [277] 277 root56Tail4 winner := by
  rw [root56Tail3, scan, if_neg (by decide), root56Candidates3_checked]
  change scan edge (search edge 3) 4 [277] 277 root56Tail4
    (if 3 ≤ root56Candidates3.length then search edge 3 [277, 787] root56Candidates3 1064 winner else winner) = _
  rw [if_neg (by decide)]
def root56Candidates4 : List Nat := [1381, 3307, 7213, 11689]
set_option maxRecDepth 8192 in
theorem root56Candidates4_checked : candidates edge 4 277 859 root56Tail5 winner.bound = root56Candidates4 := by
  unfold candidates
  have row : edge 859 = (SortedNatTree.contains FastRows.row859) := by
    change Data.row859.contains = (SortedNatTree.contains FastRows.row859)
    exact FastRows.row859_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root56Child4 : search edge 3 [277, 859] root56Candidates4 1136 winner = winner := by decide
theorem root56Step4 : scan edge (search edge 3) 4 [277] 277 root56Tail4 winner =
    scan edge (search edge 3) 4 [277] 277 root56Tail5 winner := by
  rw [root56Tail4, scan, if_neg (by decide), root56Candidates4_checked]
  change scan edge (search edge 3) 4 [277] 277 root56Tail5
    (if 3 ≤ root56Candidates4.length then search edge 3 [277, 859] root56Candidates4 1136 winner else winner) = _
  rw [if_pos (by decide), root56Child4]
def root56Candidates5 : List Nat := [22993]
set_option maxRecDepth 8192 in
theorem root56Candidates5_checked : candidates edge 4 277 1381 root56Tail6 winner.bound = root56Candidates5 := by
  unfold candidates
  have row : edge 1381 = (SortedNatTree.contains FastRows.row1381) := by
    change Data.row1381.contains = (SortedNatTree.contains FastRows.row1381)
    exact FastRows.row1381_checked.symm
  rw [row]
  decide
theorem root56Step5 : scan edge (search edge 3) 4 [277] 277 root56Tail5 winner =
    scan edge (search edge 3) 4 [277] 277 root56Tail6 winner := by
  rw [root56Tail5, scan, if_neg (by decide), root56Candidates5_checked]
  change scan edge (search edge 3) 4 [277] 277 root56Tail6
    (if 3 ≤ root56Candidates5.length then search edge 3 [277, 1381] root56Candidates5 1658 winner else winner) = _
  rw [if_neg (by decide)]
def root56Candidates6 : List Nat := [18913]
set_option maxRecDepth 8192 in
theorem root56Candidates6_checked : candidates edge 4 277 1471 root56Tail7 winner.bound = root56Candidates6 := by
  unfold candidates
  have row : edge 1471 = (SortedNatTree.contains FastRows.row1471) := by
    change Data.row1471.contains = (SortedNatTree.contains FastRows.row1471)
    exact FastRows.row1471_checked.symm
  rw [row]
  decide
theorem root56Step6 : scan edge (search edge 3) 4 [277] 277 root56Tail6 winner =
    scan edge (search edge 3) 4 [277] 277 root56Tail7 winner := by
  rw [root56Tail6, scan, if_neg (by decide), root56Candidates6_checked]
  change scan edge (search edge 3) 4 [277] 277 root56Tail7
    (if 3 ≤ root56Candidates6.length then search edge 3 [277, 1471] root56Candidates6 1748 winner else winner) = _
  rw [if_neg (by decide)]
def root56Candidates7 : List Nat := [2593, 3919, 4327]
set_option maxRecDepth 8192 in
theorem root56Candidates7_checked : candidates edge 4 277 1693 root56Tail8 winner.bound = root56Candidates7 := by
  unfold candidates
  have row : edge 1693 = (SortedNatTree.contains FastRows.row1693) := by
    change Data.row1693.contains = (SortedNatTree.contains FastRows.row1693)
    exact FastRows.row1693_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root56Child7 : search edge 3 [277, 1693] root56Candidates7 1970 winner = winner := by decide
theorem root56Step7 : scan edge (search edge 3) 4 [277] 277 root56Tail7 winner =
    scan edge (search edge 3) 4 [277] 277 root56Tail8 winner := by
  rw [root56Tail7, scan, if_neg (by decide), root56Candidates7_checked]
  change scan edge (search edge 3) 4 [277] 277 root56Tail8
    (if 3 ≤ root56Candidates7.length then search edge 3 [277, 1693] root56Candidates7 1970 winner else winner) = _
  rw [if_pos (by decide), root56Child7]
def root56Candidates8 : List Nat := [6841, 23197]
set_option maxRecDepth 8192 in
theorem root56Candidates8_checked : candidates edge 4 277 1933 root56Tail9 winner.bound = root56Candidates8 := by
  unfold candidates
  have row : edge 1933 = (SortedNatTree.contains FastRows.row1933) := by
    change Data.row1933.contains = (SortedNatTree.contains FastRows.row1933)
    exact FastRows.row1933_checked.symm
  rw [row]
  decide
theorem root56Step8 : scan edge (search edge 3) 4 [277] 277 root56Tail8 winner =
    scan edge (search edge 3) 4 [277] 277 root56Tail9 winner := by
  rw [root56Tail8, scan, if_neg (by decide), root56Candidates8_checked]
  change scan edge (search edge 3) 4 [277] 277 root56Tail9
    (if 3 ≤ root56Candidates8.length then search edge 3 [277, 1933] root56Candidates8 2210 winner else winner) = _
  rw [if_neg (by decide)]
def root56Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root56Candidates9_checked : candidates edge 4 277 2281 root56Tail10 winner.bound = root56Candidates9 := by
  unfold candidates
  have row : edge 2281 = (SortedNatTree.contains FastRows.row2281) := by
    change Data.row2281.contains = (SortedNatTree.contains FastRows.row2281)
    exact FastRows.row2281_checked.symm
  rw [row]
  decide
theorem root56Step9 : scan edge (search edge 3) 4 [277] 277 root56Tail9 winner =
    scan edge (search edge 3) 4 [277] 277 root56Tail10 winner := by
  rw [root56Tail9, scan, if_neg (by decide), root56Candidates9_checked]
  change scan edge (search edge 3) 4 [277] 277 root56Tail10
    (if 3 ≤ root56Candidates9.length then search edge 3 [277, 2281] root56Candidates9 2558 winner else winner) = _
  rw [if_neg (by decide)]
def root56Candidates10 : List Nat := [10429]
set_option maxRecDepth 8192 in
theorem root56Candidates10_checked : candidates edge 4 277 2437 root56Tail11 winner.bound = root56Candidates10 := by
  unfold candidates
  have row : edge 2437 = (SortedNatTree.contains FastRows.row2437) := by
    change Data.row2437.contains = (SortedNatTree.contains FastRows.row2437)
    exact FastRows.row2437_checked.symm
  rw [row]
  decide
theorem root56Step10 : scan edge (search edge 3) 4 [277] 277 root56Tail10 winner =
    scan edge (search edge 3) 4 [277] 277 root56Tail11 winner := by
  rw [root56Tail10, scan, if_neg (by decide), root56Candidates10_checked]
  change scan edge (search edge 3) 4 [277] 277 root56Tail11
    (if 3 ≤ root56Candidates10.length then search edge 3 [277, 2437] root56Candidates10 2714 winner else winner) = _
  rw [if_neg (by decide)]
def root56Candidates11 : List Nat := [3307]
set_option maxRecDepth 8192 in
theorem root56Candidates11_checked : candidates edge 4 277 2593 root56Tail12 winner.bound = root56Candidates11 := by
  unfold candidates
  have row : edge 2593 = (SortedNatTree.contains FastRows.row2593) := by
    change Data.row2593.contains = (SortedNatTree.contains FastRows.row2593)
    exact FastRows.row2593_checked.symm
  rw [row]
  decide
theorem root56Step11 : scan edge (search edge 3) 4 [277] 277 root56Tail11 winner =
    scan edge (search edge 3) 4 [277] 277 root56Tail12 winner := by
  rw [root56Tail11, scan, if_neg (by decide), root56Candidates11_checked]
  change scan edge (search edge 3) 4 [277] 277 root56Tail12
    (if 3 ≤ root56Candidates11.length then search edge 3 [277, 2593] root56Candidates11 2870 winner else winner) = _
  rw [if_neg (by decide)]
def root56Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root56Candidates12_checked : candidates edge 4 277 3037 root56Tail13 winner.bound = root56Candidates12 := by
  unfold candidates
  have row : edge 3037 = (SortedNatTree.contains FastRows.row3037) := by
    change Data.row3037.contains = (SortedNatTree.contains FastRows.row3037)
    exact FastRows.row3037_checked.symm
  rw [row]
  decide
theorem root56Step12 : scan edge (search edge 3) 4 [277] 277 root56Tail12 winner =
    scan edge (search edge 3) 4 [277] 277 root56Tail13 winner := by
  rw [root56Tail12, scan, if_neg (by decide), root56Candidates12_checked]
  change scan edge (search edge 3) 4 [277] 277 root56Tail13
    (if 3 ≤ root56Candidates12.length then search edge 3 [277, 3037] root56Candidates12 3314 winner else winner) = _
  rw [if_neg (by decide)]
def root56Candidates13 : List Nat := [11689, 14419]
set_option maxRecDepth 8192 in
theorem root56Candidates13_checked : candidates edge 4 277 3079 root56Tail14 winner.bound = root56Candidates13 := by
  unfold candidates
  have row : edge 3079 = (SortedNatTree.contains FastRows.row3079) := by
    change Data.row3079.contains = (SortedNatTree.contains FastRows.row3079)
    exact FastRows.row3079_checked.symm
  rw [row]
  decide
theorem root56Step13 : scan edge (search edge 3) 4 [277] 277 root56Tail13 winner =
    scan edge (search edge 3) 4 [277] 277 root56Tail14 winner := by
  rw [root56Tail13, scan, if_neg (by decide), root56Candidates13_checked]
  change scan edge (search edge 3) 4 [277] 277 root56Tail14
    (if 3 ≤ root56Candidates13.length then search edge 3 [277, 3079] root56Candidates13 3356 winner else winner) = _
  rw [if_neg (by decide)]
def root56Candidates14 : List Nat := [12073]
set_option maxRecDepth 8192 in
theorem root56Candidates14_checked : candidates edge 4 277 3307 root56Tail15 winner.bound = root56Candidates14 := by
  unfold candidates
  have row : edge 3307 = (SortedNatTree.contains FastRows.row3307) := by
    change Data.row3307.contains = (SortedNatTree.contains FastRows.row3307)
    exact FastRows.row3307_checked.symm
  rw [row]
  decide
theorem root56Step14 : scan edge (search edge 3) 4 [277] 277 root56Tail14 winner =
    scan edge (search edge 3) 4 [277] 277 root56Tail15 winner := by
  rw [root56Tail14, scan, if_neg (by decide), root56Candidates14_checked]
  change scan edge (search edge 3) 4 [277] 277 root56Tail15
    (if 3 ≤ root56Candidates14.length then search edge 3 [277, 3307] root56Candidates14 3584 winner else winner) = _
  rw [if_neg (by decide)]
def root56Candidates15 : List Nat := [13831]
set_option maxRecDepth 8192 in
theorem root56Candidates15_checked : candidates edge 4 277 3919 root56Tail16 winner.bound = root56Candidates15 := by
  unfold candidates
  have row : edge 3919 = (SortedNatTree.contains FastRows.row3919) := by
    change Data.row3919.contains = (SortedNatTree.contains FastRows.row3919)
    exact FastRows.row3919_checked.symm
  rw [row]
  decide
theorem root56Step15 : scan edge (search edge 3) 4 [277] 277 root56Tail15 winner =
    scan edge (search edge 3) 4 [277] 277 root56Tail16 winner := by
  rw [root56Tail15, scan, if_neg (by decide), root56Candidates15_checked]
  change scan edge (search edge 3) 4 [277] 277 root56Tail16
    (if 3 ≤ root56Candidates15.length then search edge 3 [277, 3919] root56Candidates15 4196 winner else winner) = _
  rw [if_neg (by decide)]
def root56Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root56Candidates16_checked : candidates edge 4 277 4327 root56Tail17 winner.bound = root56Candidates16 := by
  unfold candidates
  have row : edge 4327 = (SortedNatTree.contains FastRows.row4327) := by
    change Data.row4327.contains = (SortedNatTree.contains FastRows.row4327)
    exact FastRows.row4327_checked.symm
  rw [row]
  decide
theorem root56Step16 : scan edge (search edge 3) 4 [277] 277 root56Tail16 winner =
    scan edge (search edge 3) 4 [277] 277 root56Tail17 winner := by
  rw [root56Tail16, scan, if_neg (by decide), root56Candidates16_checked]
  change scan edge (search edge 3) 4 [277] 277 root56Tail17
    (if 3 ≤ root56Candidates16.length then search edge 3 [277, 4327] root56Candidates16 4604 winner else winner) = _
  rw [if_neg (by decide)]
def root56Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root56Candidates17_checked : candidates edge 4 277 4483 root56Tail18 winner.bound = root56Candidates17 := by
  unfold candidates
  have row : edge 4483 = (SortedNatTree.contains FastRows.row4483) := by
    change Data.row4483.contains = (SortedNatTree.contains FastRows.row4483)
    exact FastRows.row4483_checked.symm
  rw [row]
  decide
theorem root56Step17 : scan edge (search edge 3) 4 [277] 277 root56Tail17 winner =
    scan edge (search edge 3) 4 [277] 277 root56Tail18 winner := by
  rw [root56Tail17, scan, if_neg (by decide), root56Candidates17_checked]
  change scan edge (search edge 3) 4 [277] 277 root56Tail18
    (if 3 ≤ root56Candidates17.length then search edge 3 [277, 4483] root56Candidates17 4760 winner else winner) = _
  rw [if_neg (by decide)]
def root56Candidates18 : List Nat := [8647]
set_option maxRecDepth 8192 in
theorem root56Candidates18_checked : candidates edge 4 277 5119 root56Tail19 winner.bound = root56Candidates18 := by
  unfold candidates
  have row : edge 5119 = (SortedNatTree.contains FastRows.row5119) := by
    change Data.row5119.contains = (SortedNatTree.contains FastRows.row5119)
    exact FastRows.row5119_checked.symm
  rw [row]
  decide
theorem root56Step18 : scan edge (search edge 3) 4 [277] 277 root56Tail18 winner =
    scan edge (search edge 3) 4 [277] 277 root56Tail19 winner := by
  rw [root56Tail18, scan, if_neg (by decide), root56Candidates18_checked]
  change scan edge (search edge 3) 4 [277] 277 root56Tail19
    (if 3 ≤ root56Candidates18.length then search edge 3 [277, 5119] root56Candidates18 5396 winner else winner) = _
  rw [if_neg (by decide)]
def root56Candidates19 : List Nat := [16453]
set_option maxRecDepth 8192 in
theorem root56Candidates19_checked : candidates edge 4 277 5323 root56Tail20 winner.bound = root56Candidates19 := by
  unfold candidates
  have row : edge 5323 = (SortedNatTree.contains FastRows.row5323) := by
    change Data.row5323.contains = (SortedNatTree.contains FastRows.row5323)
    exact FastRows.row5323_checked.symm
  rw [row]
  decide
theorem root56Step19 : scan edge (search edge 3) 4 [277] 277 root56Tail19 winner =
    scan edge (search edge 3) 4 [277] 277 root56Tail20 winner := by
  rw [root56Tail19, scan, if_neg (by decide), root56Candidates19_checked]
  change scan edge (search edge 3) 4 [277] 277 root56Tail20
    (if 3 ≤ root56Candidates19.length then search edge 3 [277, 5323] root56Candidates19 5600 winner else winner) = _
  rw [if_neg (by decide)]
def root56Candidates20 : List Nat := [18913]
set_option maxRecDepth 8192 in
theorem root56Candidates20_checked : candidates edge 4 277 5431 root56Tail21 winner.bound = root56Candidates20 := by
  unfold candidates
  have row : edge 5431 = (SortedNatTree.contains FastRows.row5431) := by
    change Data.row5431.contains = (SortedNatTree.contains FastRows.row5431)
    exact FastRows.row5431_checked.symm
  rw [row]
  decide
theorem root56Step20 : scan edge (search edge 3) 4 [277] 277 root56Tail20 winner =
    scan edge (search edge 3) 4 [277] 277 root56Tail21 winner := by
  rw [root56Tail20, scan, if_neg (by decide), root56Candidates20_checked]
  change scan edge (search edge 3) 4 [277] 277 root56Tail21
    (if 3 ≤ root56Candidates20.length then search edge 3 [277, 5431] root56Candidates20 5708 winner else winner) = _
  rw [if_neg (by decide)]
def root56Candidates21 : List Nat := [10321]
set_option maxRecDepth 8192 in
theorem root56Candidates21_checked : candidates edge 4 277 6343 root56Tail22 winner.bound = root56Candidates21 := by
  unfold candidates
  have row : edge 6343 = (SortedNatTree.contains FastRows.row6343) := by
    change Data.row6343.contains = (SortedNatTree.contains FastRows.row6343)
    exact FastRows.row6343_checked.symm
  rw [row]
  decide
theorem root56Step21 : scan edge (search edge 3) 4 [277] 277 root56Tail21 winner =
    scan edge (search edge 3) 4 [277] 277 root56Tail22 winner := by
  rw [root56Tail21, scan, if_neg (by decide), root56Candidates21_checked]
  change scan edge (search edge 3) 4 [277] 277 root56Tail22
    (if 3 ≤ root56Candidates21.length then search edge 3 [277, 6343] root56Candidates21 6620 winner else winner) = _
  rw [if_neg (by decide)]
def root56Candidates22 : List Nat := []
set_option maxRecDepth 8192 in
theorem root56Candidates22_checked : candidates edge 4 277 6427 root56Tail23 winner.bound = root56Candidates22 := by
  unfold candidates
  have row : edge 6427 = (SortedNatTree.contains FastRows.row6427) := by
    change Data.row6427.contains = (SortedNatTree.contains FastRows.row6427)
    exact FastRows.row6427_checked.symm
  rw [row]
  decide
theorem root56Step22 : scan edge (search edge 3) 4 [277] 277 root56Tail22 winner =
    scan edge (search edge 3) 4 [277] 277 root56Tail23 winner := by
  rw [root56Tail22, scan, if_neg (by decide), root56Candidates22_checked]
  change scan edge (search edge 3) 4 [277] 277 root56Tail23
    (if 3 ≤ root56Candidates22.length then search edge 3 [277, 6427] root56Candidates22 6704 winner else winner) = _
  rw [if_neg (by decide)]
theorem root56Scan23 : scan edge (search edge 3) 4 [277] 277 root56Tail23 winner = winner := by rw [root56Tail23, scan, if_pos (by decide)]
theorem root56Scan22 : scan edge (search edge 3) 4 [277] 277 root56Tail22 winner = winner :=
  (root56Step22).trans root56Scan23
theorem root56Scan21 : scan edge (search edge 3) 4 [277] 277 root56Tail21 winner = winner :=
  (root56Step21).trans root56Scan22
theorem root56Scan20 : scan edge (search edge 3) 4 [277] 277 root56Tail20 winner = winner :=
  (root56Step20).trans root56Scan21
theorem root56Scan19 : scan edge (search edge 3) 4 [277] 277 root56Tail19 winner = winner :=
  (root56Step19).trans root56Scan20
theorem root56Scan18 : scan edge (search edge 3) 4 [277] 277 root56Tail18 winner = winner :=
  (root56Step18).trans root56Scan19
theorem root56Scan17 : scan edge (search edge 3) 4 [277] 277 root56Tail17 winner = winner :=
  (root56Step17).trans root56Scan18
theorem root56Scan16 : scan edge (search edge 3) 4 [277] 277 root56Tail16 winner = winner :=
  (root56Step16).trans root56Scan17
theorem root56Scan15 : scan edge (search edge 3) 4 [277] 277 root56Tail15 winner = winner :=
  (root56Step15).trans root56Scan16
theorem root56Scan14 : scan edge (search edge 3) 4 [277] 277 root56Tail14 winner = winner :=
  (root56Step14).trans root56Scan15
theorem root56Scan13 : scan edge (search edge 3) 4 [277] 277 root56Tail13 winner = winner :=
  (root56Step13).trans root56Scan14
theorem root56Scan12 : scan edge (search edge 3) 4 [277] 277 root56Tail12 winner = winner :=
  (root56Step12).trans root56Scan13
theorem root56Scan11 : scan edge (search edge 3) 4 [277] 277 root56Tail11 winner = winner :=
  (root56Step11).trans root56Scan12
theorem root56Scan10 : scan edge (search edge 3) 4 [277] 277 root56Tail10 winner = winner :=
  (root56Step10).trans root56Scan11
theorem root56Scan9 : scan edge (search edge 3) 4 [277] 277 root56Tail9 winner = winner :=
  (root56Step9).trans root56Scan10
theorem root56Scan8 : scan edge (search edge 3) 4 [277] 277 root56Tail8 winner = winner :=
  (root56Step8).trans root56Scan9
theorem root56Scan7 : scan edge (search edge 3) 4 [277] 277 root56Tail7 winner = winner :=
  (root56Step7).trans root56Scan8
theorem root56Scan6 : scan edge (search edge 3) 4 [277] 277 root56Tail6 winner = winner :=
  (root56Step6).trans root56Scan7
theorem root56Scan5 : scan edge (search edge 3) 4 [277] 277 root56Tail5 winner = winner :=
  (root56Step5).trans root56Scan6
theorem root56Scan4 : scan edge (search edge 3) 4 [277] 277 root56Tail4 winner = winner :=
  (root56Step4).trans root56Scan5
theorem root56Scan3 : scan edge (search edge 3) 4 [277] 277 root56Tail3 winner = winner :=
  (root56Step3).trans root56Scan4
theorem root56Scan2 : scan edge (search edge 3) 4 [277] 277 root56Tail2 winner = winner :=
  (root56Step2).trans root56Scan3
theorem root56Scan1 : scan edge (search edge 3) 4 [277] 277 root56Tail1 winner = winner :=
  (root56Step1).trans root56Scan2
theorem root56Scan0 : scan edge (search edge 3) 4 [277] 277 root56Tail0 winner = winner :=
  (root56Step0).trans root56Scan1
set_option maxRecDepth 4096 in
theorem root56Branch : search edge 4 [277] root56Candidates 277 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [277] 277 root56Tail0 winner = winner
  exact root56Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
