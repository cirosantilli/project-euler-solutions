import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root32Candidates : List Nat := Data.row149
set_option maxRecDepth 8192 in
theorem root32Candidates_checked : candidates edge 5 0 149 tail33 winner.bound = root32Candidates := by
  apply candidates_eq_row edge 5 0 149 tail33 root32Candidates winner.bound tail33_ordered
  · decide
  · decide
  · funext q
    rfl
def root32Tail67 : List Nat := []
def root32Tail66 : List Nat := 25409 :: root32Tail67
def root32Tail65 : List Nat := 25373 :: root32Tail66
def root32Tail64 : List Nat := 22727 :: root32Tail65
def root32Tail63 : List Nat := 22571 :: root32Tail64
def root32Tail62 : List Nat := 22511 :: root32Tail63
def root32Tail61 : List Nat := 22013 :: root32Tail62
def root32Tail60 : List Nat := 21587 :: root32Tail61
def root32Tail59 : List Nat := 21017 :: root32Tail60
def root32Tail58 : List Nat := 20693 :: root32Tail59
def root32Tail57 : List Nat := 20357 :: root32Tail58
def root32Tail56 : List Nat := 20327 :: root32Tail57
def root32Tail55 : List Nat := 20231 :: root32Tail56
def root32Tail54 : List Nat := 19211 :: root32Tail55
def root32Tail53 : List Nat := 18899 :: root32Tail54
def root32Tail52 : List Nat := 17351 :: root32Tail53
def root32Tail51 : List Nat := 16703 :: root32Tail52
def root32Tail50 : List Nat := 16553 :: root32Tail51
def root32Tail49 : List Nat := 15767 :: root32Tail50
def root32Tail48 : List Nat := 15161 :: root32Tail49
def root32Tail47 : List Nat := 14951 :: root32Tail48
def root32Tail46 : List Nat := 14747 :: root32Tail47
def root32Tail45 : List Nat := 14717 :: root32Tail46
def root32Tail44 : List Nat := 13751 :: root32Tail45
def root32Tail43 : List Nat := 12503 :: root32Tail44
def root32Tail42 : List Nat := 12263 :: root32Tail43
def root32Tail41 : List Nat := 12071 :: root32Tail42
def root32Tail40 : List Nat := 12041 :: root32Tail41
def root32Tail39 : List Nat := 11801 :: root32Tail40
def root32Tail38 : List Nat := 11243 :: root32Tail39
def root32Tail37 : List Nat := 10559 :: root32Tail38
def root32Tail36 : List Nat := 10457 :: root32Tail37
def root32Tail35 : List Nat := 10253 :: root32Tail36
def root32Tail34 : List Nat := 9857 :: root32Tail35
def root32Tail33 : List Nat := 9551 :: root32Tail34
def root32Tail32 : List Nat := 9419 :: root32Tail33
def root32Tail31 : List Nat := 9227 :: root32Tail32
def root32Tail30 : List Nat := 9161 :: root32Tail31
def root32Tail29 : List Nat := 8951 :: root32Tail30
def root32Tail28 : List Nat := 8741 :: root32Tail29
def root32Tail27 : List Nat := 8543 :: root32Tail28
def root32Tail26 : List Nat := 8009 :: root32Tail27
def root32Tail25 : List Nat := 7283 :: root32Tail26
def root32Tail24 : List Nat := 7193 :: root32Tail25
def root32Tail23 : List Nat := 7151 :: root32Tail24
def root32Tail22 : List Nat := 7019 :: root32Tail23
def root32Tail21 : List Nat := 6917 :: root32Tail22
def root32Tail20 : List Nat := 6791 :: root32Tail21
def root32Tail19 : List Nat := 6581 :: root32Tail20
def root32Tail18 : List Nat := 5939 :: root32Tail19
def root32Tail17 : List Nat := 5867 :: root32Tail18
def root32Tail16 : List Nat := 5717 :: root32Tail17
def root32Tail15 : List Nat := 5279 :: root32Tail16
def root32Tail14 : List Nat := 5009 :: root32Tail15
def root32Tail13 : List Nat := 4289 :: root32Tail14
def root32Tail12 : List Nat := 4133 :: root32Tail13
def root32Tail11 : List Nat := 3617 :: root32Tail12
def root32Tail10 : List Nat := 3491 :: root32Tail11
def root32Tail9 : List Nat := 2819 :: root32Tail10
def root32Tail8 : List Nat := 1979 :: root32Tail9
def root32Tail7 : List Nat := 1583 :: root32Tail8
def root32Tail6 : List Nat := 1571 :: root32Tail7
def root32Tail5 : List Nat := 971 :: root32Tail6
def root32Tail4 : List Nat := 953 :: root32Tail5
def root32Tail3 : List Nat := 563 :: root32Tail4
def root32Tail2 : List Nat := 491 :: root32Tail3
def root32Tail1 : List Nat := 251 :: root32Tail2
def root32Tail0 : List Nat := 173 :: root32Tail1
def root32Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root32Candidates0_checked : candidates edge 4 149 173 root32Tail1 winner.bound = root32Candidates0 := by
  unfold candidates
  have row : edge 173 = (SortedNatTree.contains FastRows.row173) := by
    change Data.row173.contains = (SortedNatTree.contains FastRows.row173)
    exact FastRows.row173_checked.symm
  rw [row]
  decide
theorem root32Step0 : scan edge (search edge 3) 4 [149] 149 root32Tail0 winner =
    scan edge (search edge 3) 4 [149] 149 root32Tail1 winner := by
  rw [root32Tail0, scan, if_neg (by decide), root32Candidates0_checked]
  change scan edge (search edge 3) 4 [149] 149 root32Tail1
    (if 3 ≤ root32Candidates0.length then search edge 3 [149, 173] root32Candidates0 322 winner else winner) = _
  rw [if_neg (by decide)]
def root32Candidates1 : List Nat := [491, 971, 1979, 6581, 8951]
set_option maxRecDepth 8192 in
theorem root32Candidates1_checked : candidates edge 4 149 251 root32Tail2 winner.bound = root32Candidates1 := by
  unfold candidates
  have row : edge 251 = (SortedNatTree.contains FastRows.row251) := by
    change Data.row251.contains = (SortedNatTree.contains FastRows.row251)
    exact FastRows.row251_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root32Child1 : search edge 3 [149, 251] root32Candidates1 400 winner = winner := by decide
theorem root32Step1 : scan edge (search edge 3) 4 [149] 149 root32Tail1 winner =
    scan edge (search edge 3) 4 [149] 149 root32Tail2 winner := by
  rw [root32Tail1, scan, if_neg (by decide), root32Candidates1_checked]
  change scan edge (search edge 3) 4 [149] 149 root32Tail2
    (if 3 ≤ root32Candidates1.length then search edge 3 [149, 251] root32Candidates1 400 winner else winner) = _
  rw [if_pos (by decide), root32Child1]
def root32Candidates2 : List Nat := [3617, 10559, 21587, 22013]
set_option maxRecDepth 8192 in
theorem root32Candidates2_checked : candidates edge 4 149 491 root32Tail3 winner.bound = root32Candidates2 := by
  unfold candidates
  have row : edge 491 = (SortedNatTree.contains FastRows.row491) := by
    change Data.row491.contains = (SortedNatTree.contains FastRows.row491)
    exact FastRows.row491_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root32Child2 : search edge 3 [149, 491] root32Candidates2 640 winner = winner := by decide
theorem root32Step2 : scan edge (search edge 3) 4 [149] 149 root32Tail2 winner =
    scan edge (search edge 3) 4 [149] 149 root32Tail3 winner := by
  rw [root32Tail2, scan, if_neg (by decide), root32Candidates2_checked]
  change scan edge (search edge 3) 4 [149] 149 root32Tail3
    (if 3 ≤ root32Candidates2.length then search edge 3 [149, 491] root32Candidates2 640 winner else winner) = _
  rw [if_pos (by decide), root32Child2]
def root32Candidates3 : List Nat := [971, 1979, 5717, 6581, 6791, 7019, 14747]
set_option maxRecDepth 8192 in
theorem root32Candidates3_checked : candidates edge 4 149 563 root32Tail4 winner.bound = root32Candidates3 := by
  unfold candidates
  have row : edge 563 = (SortedNatTree.contains FastRows.row563) := by
    change Data.row563.contains = (SortedNatTree.contains FastRows.row563)
    exact FastRows.row563_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root32Child3 : search edge 3 [149, 563] root32Candidates3 712 winner = winner := by decide
theorem root32Step3 : scan edge (search edge 3) 4 [149] 149 root32Tail3 winner =
    scan edge (search edge 3) 4 [149] 149 root32Tail4 winner := by
  rw [root32Tail3, scan, if_neg (by decide), root32Candidates3_checked]
  change scan edge (search edge 3) 4 [149] 149 root32Tail4
    (if 3 ≤ root32Candidates3.length then search edge 3 [149, 563] root32Candidates3 712 winner else winner) = _
  rw [if_pos (by decide), root32Child3]
def root32Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root32Candidates4_checked : candidates edge 4 149 953 root32Tail5 winner.bound = root32Candidates4 := by
  unfold candidates
  have row : edge 953 = (SortedNatTree.contains FastRows.row953) := by
    change Data.row953.contains = (SortedNatTree.contains FastRows.row953)
    exact FastRows.row953_checked.symm
  rw [row]
  decide
theorem root32Step4 : scan edge (search edge 3) 4 [149] 149 root32Tail4 winner =
    scan edge (search edge 3) 4 [149] 149 root32Tail5 winner := by
  rw [root32Tail4, scan, if_neg (by decide), root32Candidates4_checked]
  change scan edge (search edge 3) 4 [149] 149 root32Tail5
    (if 3 ≤ root32Candidates4.length then search edge 3 [149, 953] root32Candidates4 1102 winner else winner) = _
  rw [if_neg (by decide)]
def root32Candidates5 : List Nat := [5009, 7193, 14951]
set_option maxRecDepth 8192 in
theorem root32Candidates5_checked : candidates edge 4 149 971 root32Tail6 winner.bound = root32Candidates5 := by
  unfold candidates
  have row : edge 971 = (SortedNatTree.contains FastRows.row971) := by
    change Data.row971.contains = (SortedNatTree.contains FastRows.row971)
    exact FastRows.row971_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root32Child5 : search edge 3 [149, 971] root32Candidates5 1120 winner = winner := by decide
theorem root32Step5 : scan edge (search edge 3) 4 [149] 149 root32Tail5 winner =
    scan edge (search edge 3) 4 [149] 149 root32Tail6 winner := by
  rw [root32Tail5, scan, if_neg (by decide), root32Candidates5_checked]
  change scan edge (search edge 3) 4 [149] 149 root32Tail6
    (if 3 ≤ root32Candidates5.length then search edge 3 [149, 971] root32Candidates5 1120 winner else winner) = _
  rw [if_pos (by decide), root32Child5]
def root32Candidates6 : List Nat := [2819, 4133]
set_option maxRecDepth 8192 in
theorem root32Candidates6_checked : candidates edge 4 149 1571 root32Tail7 winner.bound = root32Candidates6 := by
  unfold candidates
  have row : edge 1571 = (SortedNatTree.contains FastRows.row1571) := by
    change Data.row1571.contains = (SortedNatTree.contains FastRows.row1571)
    exact FastRows.row1571_checked.symm
  rw [row]
  decide
theorem root32Step6 : scan edge (search edge 3) 4 [149] 149 root32Tail6 winner =
    scan edge (search edge 3) 4 [149] 149 root32Tail7 winner := by
  rw [root32Tail6, scan, if_neg (by decide), root32Candidates6_checked]
  change scan edge (search edge 3) 4 [149] 149 root32Tail7
    (if 3 ≤ root32Candidates6.length then search edge 3 [149, 1571] root32Candidates6 1720 winner else winner) = _
  rw [if_neg (by decide)]
def root32Candidates7 : List Nat := [5009, 20693]
set_option maxRecDepth 8192 in
theorem root32Candidates7_checked : candidates edge 4 149 1583 root32Tail8 winner.bound = root32Candidates7 := by
  unfold candidates
  have row : edge 1583 = (SortedNatTree.contains FastRows.row1583) := by
    change Data.row1583.contains = (SortedNatTree.contains FastRows.row1583)
    exact FastRows.row1583_checked.symm
  rw [row]
  decide
theorem root32Step7 : scan edge (search edge 3) 4 [149] 149 root32Tail7 winner =
    scan edge (search edge 3) 4 [149] 149 root32Tail8 winner := by
  rw [root32Tail7, scan, if_neg (by decide), root32Candidates7_checked]
  change scan edge (search edge 3) 4 [149] 149 root32Tail8
    (if 3 ≤ root32Candidates7.length then search edge 3 [149, 1583] root32Candidates7 1732 winner else winner) = _
  rw [if_neg (by decide)]
def root32Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root32Candidates8_checked : candidates edge 4 149 1979 root32Tail9 winner.bound = root32Candidates8 := by
  unfold candidates
  have row : edge 1979 = (SortedNatTree.contains FastRows.row1979) := by
    change Data.row1979.contains = (SortedNatTree.contains FastRows.row1979)
    exact FastRows.row1979_checked.symm
  rw [row]
  decide
theorem root32Step8 : scan edge (search edge 3) 4 [149] 149 root32Tail8 winner =
    scan edge (search edge 3) 4 [149] 149 root32Tail9 winner := by
  rw [root32Tail8, scan, if_neg (by decide), root32Candidates8_checked]
  change scan edge (search edge 3) 4 [149] 149 root32Tail9
    (if 3 ≤ root32Candidates8.length then search edge 3 [149, 1979] root32Candidates8 2128 winner else winner) = _
  rw [if_neg (by decide)]
def root32Candidates9 : List Nat := [3617, 21587]
set_option maxRecDepth 8192 in
theorem root32Candidates9_checked : candidates edge 4 149 2819 root32Tail10 winner.bound = root32Candidates9 := by
  unfold candidates
  have row : edge 2819 = (SortedNatTree.contains FastRows.row2819) := by
    change Data.row2819.contains = (SortedNatTree.contains FastRows.row2819)
    exact FastRows.row2819_checked.symm
  rw [row]
  decide
theorem root32Step9 : scan edge (search edge 3) 4 [149] 149 root32Tail9 winner =
    scan edge (search edge 3) 4 [149] 149 root32Tail10 winner := by
  rw [root32Tail9, scan, if_neg (by decide), root32Candidates9_checked]
  change scan edge (search edge 3) 4 [149] 149 root32Tail10
    (if 3 ≤ root32Candidates9.length then search edge 3 [149, 2819] root32Candidates9 2968 winner else winner) = _
  rw [if_neg (by decide)]
def root32Candidates10 : List Nat := [15161, 18899]
set_option maxRecDepth 8192 in
theorem root32Candidates10_checked : candidates edge 4 149 3491 root32Tail11 winner.bound = root32Candidates10 := by
  unfold candidates
  have row : edge 3491 = (SortedNatTree.contains FastRows.row3491) := by
    change Data.row3491.contains = (SortedNatTree.contains FastRows.row3491)
    exact FastRows.row3491_checked.symm
  rw [row]
  decide
theorem root32Step10 : scan edge (search edge 3) 4 [149] 149 root32Tail10 winner =
    scan edge (search edge 3) 4 [149] 149 root32Tail11 winner := by
  rw [root32Tail10, scan, if_neg (by decide), root32Candidates10_checked]
  change scan edge (search edge 3) 4 [149] 149 root32Tail11
    (if 3 ≤ root32Candidates10.length then search edge 3 [149, 3491] root32Candidates10 3640 winner else winner) = _
  rw [if_neg (by decide)]
def root32Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root32Candidates11_checked : candidates edge 4 149 3617 root32Tail12 winner.bound = root32Candidates11 := by
  unfold candidates
  have row : edge 3617 = (SortedNatTree.contains FastRows.row3617) := by
    change Data.row3617.contains = (SortedNatTree.contains FastRows.row3617)
    exact FastRows.row3617_checked.symm
  rw [row]
  decide
theorem root32Step11 : scan edge (search edge 3) 4 [149] 149 root32Tail11 winner =
    scan edge (search edge 3) 4 [149] 149 root32Tail12 winner := by
  rw [root32Tail11, scan, if_neg (by decide), root32Candidates11_checked]
  change scan edge (search edge 3) 4 [149] 149 root32Tail12
    (if 3 ≤ root32Candidates11.length then search edge 3 [149, 3617] root32Candidates11 3766 winner else winner) = _
  rw [if_neg (by decide)]
def root32Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root32Candidates12_checked : candidates edge 4 149 4133 root32Tail13 winner.bound = root32Candidates12 := by
  unfold candidates
  have row : edge 4133 = (SortedNatTree.contains FastRows.row4133) := by
    change Data.row4133.contains = (SortedNatTree.contains FastRows.row4133)
    exact FastRows.row4133_checked.symm
  rw [row]
  decide
theorem root32Step12 : scan edge (search edge 3) 4 [149] 149 root32Tail12 winner =
    scan edge (search edge 3) 4 [149] 149 root32Tail13 winner := by
  rw [root32Tail12, scan, if_neg (by decide), root32Candidates12_checked]
  change scan edge (search edge 3) 4 [149] 149 root32Tail13
    (if 3 ≤ root32Candidates12.length then search edge 3 [149, 4133] root32Candidates12 4282 winner else winner) = _
  rw [if_neg (by decide)]
def root32Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root32Candidates13_checked : candidates edge 4 149 4289 root32Tail14 winner.bound = root32Candidates13 := by
  unfold candidates
  have row : edge 4289 = (SortedNatTree.contains FastRows.row4289) := by
    change Data.row4289.contains = (SortedNatTree.contains FastRows.row4289)
    exact FastRows.row4289_checked.symm
  rw [row]
  decide
theorem root32Step13 : scan edge (search edge 3) 4 [149] 149 root32Tail13 winner =
    scan edge (search edge 3) 4 [149] 149 root32Tail14 winner := by
  rw [root32Tail13, scan, if_neg (by decide), root32Candidates13_checked]
  change scan edge (search edge 3) 4 [149] 149 root32Tail14
    (if 3 ≤ root32Candidates13.length then search edge 3 [149, 4289] root32Candidates13 4438 winner else winner) = _
  rw [if_neg (by decide)]
def root32Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root32Candidates14_checked : candidates edge 4 149 5009 root32Tail15 winner.bound = root32Candidates14 := by
  unfold candidates
  have row : edge 5009 = (SortedNatTree.contains FastRows.row5009) := by
    change Data.row5009.contains = (SortedNatTree.contains FastRows.row5009)
    exact FastRows.row5009_checked.symm
  rw [row]
  decide
theorem root32Step14 : scan edge (search edge 3) 4 [149] 149 root32Tail14 winner =
    scan edge (search edge 3) 4 [149] 149 root32Tail15 winner := by
  rw [root32Tail14, scan, if_neg (by decide), root32Candidates14_checked]
  change scan edge (search edge 3) 4 [149] 149 root32Tail15
    (if 3 ≤ root32Candidates14.length then search edge 3 [149, 5009] root32Candidates14 5158 winner else winner) = _
  rw [if_neg (by decide)]
def root32Candidates15 : List Nat := [12503]
set_option maxRecDepth 8192 in
theorem root32Candidates15_checked : candidates edge 4 149 5279 root32Tail16 winner.bound = root32Candidates15 := by
  unfold candidates
  have row : edge 5279 = (SortedNatTree.contains FastRows.row5279) := by
    change Data.row5279.contains = (SortedNatTree.contains FastRows.row5279)
    exact FastRows.row5279_checked.symm
  rw [row]
  decide
theorem root32Step15 : scan edge (search edge 3) 4 [149] 149 root32Tail15 winner =
    scan edge (search edge 3) 4 [149] 149 root32Tail16 winner := by
  rw [root32Tail15, scan, if_neg (by decide), root32Candidates15_checked]
  change scan edge (search edge 3) 4 [149] 149 root32Tail16
    (if 3 ≤ root32Candidates15.length then search edge 3 [149, 5279] root32Candidates15 5428 winner else winner) = _
  rw [if_neg (by decide)]
def root32Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root32Candidates16_checked : candidates edge 4 149 5717 root32Tail17 winner.bound = root32Candidates16 := by
  unfold candidates
  have row : edge 5717 = (SortedNatTree.contains FastRows.row5717) := by
    change Data.row5717.contains = (SortedNatTree.contains FastRows.row5717)
    exact FastRows.row5717_checked.symm
  rw [row]
  decide
theorem root32Step16 : scan edge (search edge 3) 4 [149] 149 root32Tail16 winner =
    scan edge (search edge 3) 4 [149] 149 root32Tail17 winner := by
  rw [root32Tail16, scan, if_neg (by decide), root32Candidates16_checked]
  change scan edge (search edge 3) 4 [149] 149 root32Tail17
    (if 3 ≤ root32Candidates16.length then search edge 3 [149, 5717] root32Candidates16 5866 winner else winner) = _
  rw [if_neg (by decide)]
def root32Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root32Candidates17_checked : candidates edge 4 149 5867 root32Tail18 winner.bound = root32Candidates17 := by
  unfold candidates
  have row : edge 5867 = (SortedNatTree.contains FastRows.row5867) := by
    change Data.row5867.contains = (SortedNatTree.contains FastRows.row5867)
    exact FastRows.row5867_checked.symm
  rw [row]
  decide
theorem root32Step17 : scan edge (search edge 3) 4 [149] 149 root32Tail17 winner =
    scan edge (search edge 3) 4 [149] 149 root32Tail18 winner := by
  rw [root32Tail17, scan, if_neg (by decide), root32Candidates17_checked]
  change scan edge (search edge 3) 4 [149] 149 root32Tail18
    (if 3 ≤ root32Candidates17.length then search edge 3 [149, 5867] root32Candidates17 6016 winner else winner) = _
  rw [if_neg (by decide)]
def root32Candidates18 : List Nat := []
set_option maxRecDepth 8192 in
theorem root32Candidates18_checked : candidates edge 4 149 5939 root32Tail19 winner.bound = root32Candidates18 := by
  unfold candidates
  have row : edge 5939 = (SortedNatTree.contains FastRows.row5939) := by
    change Data.row5939.contains = (SortedNatTree.contains FastRows.row5939)
    exact FastRows.row5939_checked.symm
  rw [row]
  decide
theorem root32Step18 : scan edge (search edge 3) 4 [149] 149 root32Tail18 winner =
    scan edge (search edge 3) 4 [149] 149 root32Tail19 winner := by
  rw [root32Tail18, scan, if_neg (by decide), root32Candidates18_checked]
  change scan edge (search edge 3) 4 [149] 149 root32Tail19
    (if 3 ≤ root32Candidates18.length then search edge 3 [149, 5939] root32Candidates18 6088 winner else winner) = _
  rw [if_neg (by decide)]
theorem root32Scan19 : scan edge (search edge 3) 4 [149] 149 root32Tail19 winner = winner := by rw [root32Tail19, scan, if_pos (by decide)]
theorem root32Scan18 : scan edge (search edge 3) 4 [149] 149 root32Tail18 winner = winner :=
  (root32Step18).trans root32Scan19
theorem root32Scan17 : scan edge (search edge 3) 4 [149] 149 root32Tail17 winner = winner :=
  (root32Step17).trans root32Scan18
theorem root32Scan16 : scan edge (search edge 3) 4 [149] 149 root32Tail16 winner = winner :=
  (root32Step16).trans root32Scan17
theorem root32Scan15 : scan edge (search edge 3) 4 [149] 149 root32Tail15 winner = winner :=
  (root32Step15).trans root32Scan16
theorem root32Scan14 : scan edge (search edge 3) 4 [149] 149 root32Tail14 winner = winner :=
  (root32Step14).trans root32Scan15
theorem root32Scan13 : scan edge (search edge 3) 4 [149] 149 root32Tail13 winner = winner :=
  (root32Step13).trans root32Scan14
theorem root32Scan12 : scan edge (search edge 3) 4 [149] 149 root32Tail12 winner = winner :=
  (root32Step12).trans root32Scan13
theorem root32Scan11 : scan edge (search edge 3) 4 [149] 149 root32Tail11 winner = winner :=
  (root32Step11).trans root32Scan12
theorem root32Scan10 : scan edge (search edge 3) 4 [149] 149 root32Tail10 winner = winner :=
  (root32Step10).trans root32Scan11
theorem root32Scan9 : scan edge (search edge 3) 4 [149] 149 root32Tail9 winner = winner :=
  (root32Step9).trans root32Scan10
theorem root32Scan8 : scan edge (search edge 3) 4 [149] 149 root32Tail8 winner = winner :=
  (root32Step8).trans root32Scan9
theorem root32Scan7 : scan edge (search edge 3) 4 [149] 149 root32Tail7 winner = winner :=
  (root32Step7).trans root32Scan8
theorem root32Scan6 : scan edge (search edge 3) 4 [149] 149 root32Tail6 winner = winner :=
  (root32Step6).trans root32Scan7
theorem root32Scan5 : scan edge (search edge 3) 4 [149] 149 root32Tail5 winner = winner :=
  (root32Step5).trans root32Scan6
theorem root32Scan4 : scan edge (search edge 3) 4 [149] 149 root32Tail4 winner = winner :=
  (root32Step4).trans root32Scan5
theorem root32Scan3 : scan edge (search edge 3) 4 [149] 149 root32Tail3 winner = winner :=
  (root32Step3).trans root32Scan4
theorem root32Scan2 : scan edge (search edge 3) 4 [149] 149 root32Tail2 winner = winner :=
  (root32Step2).trans root32Scan3
theorem root32Scan1 : scan edge (search edge 3) 4 [149] 149 root32Tail1 winner = winner :=
  (root32Step1).trans root32Scan2
theorem root32Scan0 : scan edge (search edge 3) 4 [149] 149 root32Tail0 winner = winner :=
  (root32Step0).trans root32Scan1
set_option maxRecDepth 4096 in
theorem root32Branch : search edge 4 [149] root32Candidates 149 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [149] 149 root32Tail0 winner = winner
  exact root32Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
