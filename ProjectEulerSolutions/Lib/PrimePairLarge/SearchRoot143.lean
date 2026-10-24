import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root143Candidates : List Nat := Data.row839
set_option maxRecDepth 8192 in
theorem root143Candidates_checked : candidates edge 5 0 839 tail144 winner.bound = root143Candidates := by
  apply candidates_eq_row edge 5 0 839 tail144 root143Candidates winner.bound tail144_ordered
  · decide
  · decide
  · funext q
    rfl
def root143Tail65 : List Nat := []
def root143Tail64 : List Nat := 25169 :: root143Tail65
def root143Tail63 : List Nat := 24749 :: root143Tail64
def root143Tail62 : List Nat := 24179 :: root143Tail63
def root143Tail61 : List Nat := 24077 :: root143Tail62
def root143Tail60 : List Nat := 23753 :: root143Tail61
def root143Tail59 : List Nat := 23663 :: root143Tail60
def root143Tail58 : List Nat := 22571 :: root143Tail59
def root143Tail57 : List Nat := 21647 :: root143Tail58
def root143Tail56 : List Nat := 21521 :: root143Tail57
def root143Tail55 : List Nat := 21011 :: root143Tail56
def root143Tail54 : List Nat := 20717 :: root143Tail55
def root143Tail53 : List Nat := 20231 :: root143Tail54
def root143Tail52 : List Nat := 19211 :: root143Tail53
def root143Tail51 : List Nat := 19001 :: root143Tail52
def root143Tail50 : List Nat := 18803 :: root143Tail51
def root143Tail49 : List Nat := 18503 :: root143Tail50
def root143Tail48 : List Nat := 18089 :: root143Tail49
def root143Tail47 : List Nat := 17489 :: root143Tail48
def root143Tail46 : List Nat := 17033 :: root143Tail47
def root143Tail45 : List Nat := 16529 :: root143Tail46
def root143Tail44 : List Nat := 16451 :: root143Tail45
def root143Tail43 : List Nat := 14009 :: root143Tail44
def root143Tail42 : List Nat := 13763 :: root143Tail43
def root143Tail41 : List Nat := 13421 :: root143Tail42
def root143Tail40 : List Nat := 13331 :: root143Tail41
def root143Tail39 : List Nat := 13259 :: root143Tail40
def root143Tail38 : List Nat := 13217 :: root143Tail39
def root143Tail37 : List Nat := 13127 :: root143Tail38
def root143Tail36 : List Nat := 12713 :: root143Tail37
def root143Tail35 : List Nat := 12239 :: root143Tail36
def root143Tail34 : List Nat := 12161 :: root143Tail35
def root143Tail33 : List Nat := 10781 :: root143Tail34
def root143Tail32 : List Nat := 10259 :: root143Tail33
def root143Tail31 : List Nat := 10223 :: root143Tail32
def root143Tail30 : List Nat := 10193 :: root143Tail31
def root143Tail29 : List Nat := 10007 :: root143Tail30
def root143Tail28 : List Nat := 9257 :: root143Tail29
def root143Tail27 : List Nat := 8543 :: root143Tail28
def root143Tail26 : List Nat := 8447 :: root143Tail27
def root143Tail25 : List Nat := 8387 :: root143Tail26
def root143Tail24 : List Nat := 7877 :: root143Tail25
def root143Tail23 : List Nat := 7841 :: root143Tail24
def root143Tail22 : List Nat := 7457 :: root143Tail23
def root143Tail21 : List Nat := 7331 :: root143Tail22
def root143Tail20 : List Nat := 7283 :: root143Tail21
def root143Tail19 : List Nat := 7121 :: root143Tail20
def root143Tail18 : List Nat := 6599 :: root143Tail19
def root143Tail17 : List Nat := 6263 :: root143Tail18
def root143Tail16 : List Nat := 5741 :: root143Tail17
def root143Tail15 : List Nat := 5351 :: root143Tail16
def root143Tail14 : List Nat := 5273 :: root143Tail15
def root143Tail13 : List Nat := 5237 :: root143Tail14
def root143Tail12 : List Nat := 4721 :: root143Tail13
def root143Tail11 : List Nat := 4679 :: root143Tail12
def root143Tail10 : List Nat := 3923 :: root143Tail11
def root143Tail9 : List Nat := 3533 :: root143Tail10
def root143Tail8 : List Nat := 3251 :: root143Tail9
def root143Tail7 : List Nat := 2333 :: root143Tail8
def root143Tail6 : List Nat := 2063 :: root143Tail7
def root143Tail5 : List Nat := 1949 :: root143Tail6
def root143Tail4 : List Nat := 1289 :: root143Tail5
def root143Tail3 : List Nat := 1283 :: root143Tail4
def root143Tail2 : List Nat := 1109 :: root143Tail3
def root143Tail1 : List Nat := 911 :: root143Tail2
def root143Tail0 : List Nat := 887 :: root143Tail1
def root143Candidates0 : List Nat := [3251, 12161, 13127, 13421]
set_option maxRecDepth 8192 in
theorem root143Candidates0_checked : candidates edge 4 839 887 root143Tail1 winner.bound = root143Candidates0 := by
  unfold candidates
  have row : edge 887 = (SortedNatTree.contains FastRows.row887) := by
    change Data.row887.contains = (SortedNatTree.contains FastRows.row887)
    exact FastRows.row887_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root143Child0 : search edge 3 [839, 887] root143Candidates0 1726 winner = winner := by decide
theorem root143Step0 : scan edge (search edge 3) 4 [839] 839 root143Tail0 winner =
    scan edge (search edge 3) 4 [839] 839 root143Tail1 winner := by
  rw [root143Tail0, scan, if_neg (by decide), root143Candidates0_checked]
  change scan edge (search edge 3) 4 [839] 839 root143Tail1
    (if 3 ≤ root143Candidates0.length then search edge 3 [839, 887] root143Candidates0 1726 winner else winner) = _
  rw [if_pos (by decide), root143Child0]
def root143Candidates1 : List Nat := [21647]
set_option maxRecDepth 8192 in
theorem root143Candidates1_checked : candidates edge 4 839 911 root143Tail2 winner.bound = root143Candidates1 := by
  unfold candidates
  have row : edge 911 = (SortedNatTree.contains FastRows.row911) := by
    change Data.row911.contains = (SortedNatTree.contains FastRows.row911)
    exact FastRows.row911_checked.symm
  rw [row]
  decide
theorem root143Step1 : scan edge (search edge 3) 4 [839] 839 root143Tail1 winner =
    scan edge (search edge 3) 4 [839] 839 root143Tail2 winner := by
  rw [root143Tail1, scan, if_neg (by decide), root143Candidates1_checked]
  change scan edge (search edge 3) 4 [839] 839 root143Tail2
    (if 3 ≤ root143Candidates1.length then search edge 3 [839, 911] root143Candidates1 1750 winner else winner) = _
  rw [if_neg (by decide)]
def root143Candidates2 : List Nat := [3923]
set_option maxRecDepth 8192 in
theorem root143Candidates2_checked : candidates edge 4 839 1109 root143Tail3 winner.bound = root143Candidates2 := by
  unfold candidates
  have row : edge 1109 = (SortedNatTree.contains FastRows.row1109) := by
    change Data.row1109.contains = (SortedNatTree.contains FastRows.row1109)
    exact FastRows.row1109_checked.symm
  rw [row]
  decide
theorem root143Step2 : scan edge (search edge 3) 4 [839] 839 root143Tail2 winner =
    scan edge (search edge 3) 4 [839] 839 root143Tail3 winner := by
  rw [root143Tail2, scan, if_neg (by decide), root143Candidates2_checked]
  change scan edge (search edge 3) 4 [839] 839 root143Tail3
    (if 3 ≤ root143Candidates2.length then search edge 3 [839, 1109] root143Candidates2 1948 winner else winner) = _
  rw [if_neg (by decide)]
def root143Candidates3 : List Nat := [13217, 16451]
set_option maxRecDepth 8192 in
theorem root143Candidates3_checked : candidates edge 4 839 1283 root143Tail4 winner.bound = root143Candidates3 := by
  unfold candidates
  have row : edge 1283 = (SortedNatTree.contains FastRows.row1283) := by
    change Data.row1283.contains = (SortedNatTree.contains FastRows.row1283)
    exact FastRows.row1283_checked.symm
  rw [row]
  decide
theorem root143Step3 : scan edge (search edge 3) 4 [839] 839 root143Tail3 winner =
    scan edge (search edge 3) 4 [839] 839 root143Tail4 winner := by
  rw [root143Tail3, scan, if_neg (by decide), root143Candidates3_checked]
  change scan edge (search edge 3) 4 [839] 839 root143Tail4
    (if 3 ≤ root143Candidates3.length then search edge 3 [839, 1283] root143Candidates3 2122 winner else winner) = _
  rw [if_neg (by decide)]
def root143Candidates4 : List Nat := [10781, 12713, 13259, 22571]
set_option maxRecDepth 8192 in
theorem root143Candidates4_checked : candidates edge 4 839 1289 root143Tail5 winner.bound = root143Candidates4 := by
  unfold candidates
  have row : edge 1289 = (SortedNatTree.contains FastRows.row1289) := by
    change Data.row1289.contains = (SortedNatTree.contains FastRows.row1289)
    exact FastRows.row1289_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root143Child4 : search edge 3 [839, 1289] root143Candidates4 2128 winner = winner := by decide
theorem root143Step4 : scan edge (search edge 3) 4 [839] 839 root143Tail4 winner =
    scan edge (search edge 3) 4 [839] 839 root143Tail5 winner := by
  rw [root143Tail4, scan, if_neg (by decide), root143Candidates4_checked]
  change scan edge (search edge 3) 4 [839] 839 root143Tail5
    (if 3 ≤ root143Candidates4.length then search edge 3 [839, 1289] root143Candidates4 2128 winner else winner) = _
  rw [if_pos (by decide), root143Child4]
def root143Candidates5 : List Nat := [3923, 10007, 10259]
set_option maxRecDepth 8192 in
theorem root143Candidates5_checked : candidates edge 4 839 1949 root143Tail6 winner.bound = root143Candidates5 := by
  unfold candidates
  have row : edge 1949 = (SortedNatTree.contains FastRows.row1949) := by
    change Data.row1949.contains = (SortedNatTree.contains FastRows.row1949)
    exact FastRows.row1949_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root143Child5 : search edge 3 [839, 1949] root143Candidates5 2788 winner = winner := by decide
theorem root143Step5 : scan edge (search edge 3) 4 [839] 839 root143Tail5 winner =
    scan edge (search edge 3) 4 [839] 839 root143Tail6 winner := by
  rw [root143Tail5, scan, if_neg (by decide), root143Candidates5_checked]
  change scan edge (search edge 3) 4 [839] 839 root143Tail6
    (if 3 ≤ root143Candidates5.length then search edge 3 [839, 1949] root143Candidates5 2788 winner else winner) = _
  rw [if_pos (by decide), root143Child5]
def root143Candidates6 : List Nat := [2333, 6599, 16529]
set_option maxRecDepth 8192 in
theorem root143Candidates6_checked : candidates edge 4 839 2063 root143Tail7 winner.bound = root143Candidates6 := by
  unfold candidates
  have row : edge 2063 = (SortedNatTree.contains FastRows.row2063) := by
    change Data.row2063.contains = (SortedNatTree.contains FastRows.row2063)
    exact FastRows.row2063_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root143Child6 : search edge 3 [839, 2063] root143Candidates6 2902 winner = winner := by decide
theorem root143Step6 : scan edge (search edge 3) 4 [839] 839 root143Tail6 winner =
    scan edge (search edge 3) 4 [839] 839 root143Tail7 winner := by
  rw [root143Tail6, scan, if_neg (by decide), root143Candidates6_checked]
  change scan edge (search edge 3) 4 [839] 839 root143Tail7
    (if 3 ≤ root143Candidates6.length then search edge 3 [839, 2063] root143Candidates6 2902 winner else winner) = _
  rw [if_pos (by decide), root143Child6]
def root143Candidates7 : List Nat := [3533, 8387, 16451, 20231]
set_option maxRecDepth 8192 in
theorem root143Candidates7_checked : candidates edge 4 839 2333 root143Tail8 winner.bound = root143Candidates7 := by
  unfold candidates
  have row : edge 2333 = (SortedNatTree.contains FastRows.row2333) := by
    change Data.row2333.contains = (SortedNatTree.contains FastRows.row2333)
    exact FastRows.row2333_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root143Child7 : search edge 3 [839, 2333] root143Candidates7 3172 winner = winner := by decide
theorem root143Step7 : scan edge (search edge 3) 4 [839] 839 root143Tail7 winner =
    scan edge (search edge 3) 4 [839] 839 root143Tail8 winner := by
  rw [root143Tail7, scan, if_neg (by decide), root143Candidates7_checked]
  change scan edge (search edge 3) 4 [839] 839 root143Tail8
    (if 3 ≤ root143Candidates7.length then search edge 3 [839, 2333] root143Candidates7 3172 winner else winner) = _
  rw [if_pos (by decide), root143Child7]
def root143Candidates8 : List Nat := [4721]
set_option maxRecDepth 8192 in
theorem root143Candidates8_checked : candidates edge 4 839 3251 root143Tail9 winner.bound = root143Candidates8 := by
  unfold candidates
  have row : edge 3251 = (SortedNatTree.contains FastRows.row3251) := by
    change Data.row3251.contains = (SortedNatTree.contains FastRows.row3251)
    exact FastRows.row3251_checked.symm
  rw [row]
  decide
theorem root143Step8 : scan edge (search edge 3) 4 [839] 839 root143Tail8 winner =
    scan edge (search edge 3) 4 [839] 839 root143Tail9 winner := by
  rw [root143Tail8, scan, if_neg (by decide), root143Candidates8_checked]
  change scan edge (search edge 3) 4 [839] 839 root143Tail9
    (if 3 ≤ root143Candidates8.length then search edge 3 [839, 3251] root143Candidates8 4090 winner else winner) = _
  rw [if_neg (by decide)]
def root143Candidates9 : List Nat := [4679]
set_option maxRecDepth 8192 in
theorem root143Candidates9_checked : candidates edge 4 839 3533 root143Tail10 winner.bound = root143Candidates9 := by
  unfold candidates
  have row : edge 3533 = (SortedNatTree.contains FastRows.row3533) := by
    change Data.row3533.contains = (SortedNatTree.contains FastRows.row3533)
    exact FastRows.row3533_checked.symm
  rw [row]
  decide
theorem root143Step9 : scan edge (search edge 3) 4 [839] 839 root143Tail9 winner =
    scan edge (search edge 3) 4 [839] 839 root143Tail10 winner := by
  rw [root143Tail9, scan, if_neg (by decide), root143Candidates9_checked]
  change scan edge (search edge 3) 4 [839] 839 root143Tail10
    (if 3 ≤ root143Candidates9.length then search edge 3 [839, 3533] root143Candidates9 4372 winner else winner) = _
  rw [if_neg (by decide)]
def root143Candidates10 : List Nat := [4721]
set_option maxRecDepth 8192 in
theorem root143Candidates10_checked : candidates edge 4 839 3923 root143Tail11 winner.bound = root143Candidates10 := by
  unfold candidates
  have row : edge 3923 = (SortedNatTree.contains FastRows.row3923) := by
    change Data.row3923.contains = (SortedNatTree.contains FastRows.row3923)
    exact FastRows.row3923_checked.symm
  rw [row]
  decide
theorem root143Step10 : scan edge (search edge 3) 4 [839] 839 root143Tail10 winner =
    scan edge (search edge 3) 4 [839] 839 root143Tail11 winner := by
  rw [root143Tail10, scan, if_neg (by decide), root143Candidates10_checked]
  change scan edge (search edge 3) 4 [839] 839 root143Tail11
    (if 3 ≤ root143Candidates10.length then search edge 3 [839, 3923] root143Candidates10 4762 winner else winner) = _
  rw [if_neg (by decide)]
def root143Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root143Candidates11_checked : candidates edge 4 839 4679 root143Tail12 winner.bound = root143Candidates11 := by
  unfold candidates
  have row : edge 4679 = (SortedNatTree.contains FastRows.row4679) := by
    change Data.row4679.contains = (SortedNatTree.contains FastRows.row4679)
    exact FastRows.row4679_checked.symm
  rw [row]
  decide
theorem root143Step11 : scan edge (search edge 3) 4 [839] 839 root143Tail11 winner =
    scan edge (search edge 3) 4 [839] 839 root143Tail12 winner := by
  rw [root143Tail11, scan, if_neg (by decide), root143Candidates11_checked]
  change scan edge (search edge 3) 4 [839] 839 root143Tail12
    (if 3 ≤ root143Candidates11.length then search edge 3 [839, 4679] root143Candidates11 5518 winner else winner) = _
  rw [if_neg (by decide)]
def root143Candidates12 : List Nat := [9257]
set_option maxRecDepth 8192 in
theorem root143Candidates12_checked : candidates edge 4 839 4721 root143Tail13 winner.bound = root143Candidates12 := by
  unfold candidates
  have row : edge 4721 = (SortedNatTree.contains FastRows.row4721) := by
    change Data.row4721.contains = (SortedNatTree.contains FastRows.row4721)
    exact FastRows.row4721_checked.symm
  rw [row]
  decide
theorem root143Step12 : scan edge (search edge 3) 4 [839] 839 root143Tail12 winner =
    scan edge (search edge 3) 4 [839] 839 root143Tail13 winner := by
  rw [root143Tail12, scan, if_neg (by decide), root143Candidates12_checked]
  change scan edge (search edge 3) 4 [839] 839 root143Tail13
    (if 3 ≤ root143Candidates12.length then search edge 3 [839, 4721] root143Candidates12 5560 winner else winner) = _
  rw [if_neg (by decide)]
def root143Candidates13 : List Nat := [13217, 16451]
set_option maxRecDepth 8192 in
theorem root143Candidates13_checked : candidates edge 4 839 5237 root143Tail14 winner.bound = root143Candidates13 := by
  unfold candidates
  have row : edge 5237 = (SortedNatTree.contains FastRows.row5237) := by
    change Data.row5237.contains = (SortedNatTree.contains FastRows.row5237)
    exact FastRows.row5237_checked.symm
  rw [row]
  decide
theorem root143Step13 : scan edge (search edge 3) 4 [839] 839 root143Tail13 winner =
    scan edge (search edge 3) 4 [839] 839 root143Tail14 winner := by
  rw [root143Tail13, scan, if_neg (by decide), root143Candidates13_checked]
  change scan edge (search edge 3) 4 [839] 839 root143Tail14
    (if 3 ≤ root143Candidates13.length then search edge 3 [839, 5237] root143Candidates13 6076 winner else winner) = _
  rw [if_neg (by decide)]
def root143Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root143Candidates14_checked : candidates edge 4 839 5273 root143Tail15 winner.bound = root143Candidates14 := by
  unfold candidates
  have row : edge 5273 = (SortedNatTree.contains FastRows.row5273) := by
    change Data.row5273.contains = (SortedNatTree.contains FastRows.row5273)
    exact FastRows.row5273_checked.symm
  rw [row]
  decide
theorem root143Step14 : scan edge (search edge 3) 4 [839] 839 root143Tail14 winner =
    scan edge (search edge 3) 4 [839] 839 root143Tail15 winner := by
  rw [root143Tail14, scan, if_neg (by decide), root143Candidates14_checked]
  change scan edge (search edge 3) 4 [839] 839 root143Tail15
    (if 3 ≤ root143Candidates14.length then search edge 3 [839, 5273] root143Candidates14 6112 winner else winner) = _
  rw [if_neg (by decide)]
def root143Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root143Candidates15_checked : candidates edge 4 839 5351 root143Tail16 winner.bound = root143Candidates15 := by
  unfold candidates
  have row : edge 5351 = (SortedNatTree.contains FastRows.row5351) := by
    change Data.row5351.contains = (SortedNatTree.contains FastRows.row5351)
    exact FastRows.row5351_checked.symm
  rw [row]
  decide
theorem root143Step15 : scan edge (search edge 3) 4 [839] 839 root143Tail15 winner =
    scan edge (search edge 3) 4 [839] 839 root143Tail16 winner := by
  rw [root143Tail15, scan, if_neg (by decide), root143Candidates15_checked]
  change scan edge (search edge 3) 4 [839] 839 root143Tail16
    (if 3 ≤ root143Candidates15.length then search edge 3 [839, 5351] root143Candidates15 6190 winner else winner) = _
  rw [if_neg (by decide)]
def root143Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root143Candidates16_checked : candidates edge 4 839 5741 root143Tail17 winner.bound = root143Candidates16 := by
  unfold candidates
  have row : edge 5741 = (SortedNatTree.contains FastRows.row5741) := by
    change Data.row5741.contains = (SortedNatTree.contains FastRows.row5741)
    exact FastRows.row5741_checked.symm
  rw [row]
  decide
theorem root143Step16 : scan edge (search edge 3) 4 [839] 839 root143Tail16 winner =
    scan edge (search edge 3) 4 [839] 839 root143Tail17 winner := by
  rw [root143Tail16, scan, if_neg (by decide), root143Candidates16_checked]
  change scan edge (search edge 3) 4 [839] 839 root143Tail17
    (if 3 ≤ root143Candidates16.length then search edge 3 [839, 5741] root143Candidates16 6580 winner else winner) = _
  rw [if_neg (by decide)]
def root143Candidates17 : List Nat := [7121]
set_option maxRecDepth 8192 in
theorem root143Candidates17_checked : candidates edge 4 839 6263 root143Tail18 winner.bound = root143Candidates17 := by
  unfold candidates
  have row : edge 6263 = (SortedNatTree.contains FastRows.row6263) := by
    change Data.row6263.contains = (SortedNatTree.contains FastRows.row6263)
    exact FastRows.row6263_checked.symm
  rw [row]
  decide
theorem root143Step17 : scan edge (search edge 3) 4 [839] 839 root143Tail17 winner =
    scan edge (search edge 3) 4 [839] 839 root143Tail18 winner := by
  rw [root143Tail17, scan, if_neg (by decide), root143Candidates17_checked]
  change scan edge (search edge 3) 4 [839] 839 root143Tail18
    (if 3 ≤ root143Candidates17.length then search edge 3 [839, 6263] root143Candidates17 7102 winner else winner) = _
  rw [if_neg (by decide)]
theorem root143Scan18 : scan edge (search edge 3) 4 [839] 839 root143Tail18 winner = winner := by rw [root143Tail18, scan, if_pos (by decide)]
theorem root143Scan17 : scan edge (search edge 3) 4 [839] 839 root143Tail17 winner = winner :=
  (root143Step17).trans root143Scan18
theorem root143Scan16 : scan edge (search edge 3) 4 [839] 839 root143Tail16 winner = winner :=
  (root143Step16).trans root143Scan17
theorem root143Scan15 : scan edge (search edge 3) 4 [839] 839 root143Tail15 winner = winner :=
  (root143Step15).trans root143Scan16
theorem root143Scan14 : scan edge (search edge 3) 4 [839] 839 root143Tail14 winner = winner :=
  (root143Step14).trans root143Scan15
theorem root143Scan13 : scan edge (search edge 3) 4 [839] 839 root143Tail13 winner = winner :=
  (root143Step13).trans root143Scan14
theorem root143Scan12 : scan edge (search edge 3) 4 [839] 839 root143Tail12 winner = winner :=
  (root143Step12).trans root143Scan13
theorem root143Scan11 : scan edge (search edge 3) 4 [839] 839 root143Tail11 winner = winner :=
  (root143Step11).trans root143Scan12
theorem root143Scan10 : scan edge (search edge 3) 4 [839] 839 root143Tail10 winner = winner :=
  (root143Step10).trans root143Scan11
theorem root143Scan9 : scan edge (search edge 3) 4 [839] 839 root143Tail9 winner = winner :=
  (root143Step9).trans root143Scan10
theorem root143Scan8 : scan edge (search edge 3) 4 [839] 839 root143Tail8 winner = winner :=
  (root143Step8).trans root143Scan9
theorem root143Scan7 : scan edge (search edge 3) 4 [839] 839 root143Tail7 winner = winner :=
  (root143Step7).trans root143Scan8
theorem root143Scan6 : scan edge (search edge 3) 4 [839] 839 root143Tail6 winner = winner :=
  (root143Step6).trans root143Scan7
theorem root143Scan5 : scan edge (search edge 3) 4 [839] 839 root143Tail5 winner = winner :=
  (root143Step5).trans root143Scan6
theorem root143Scan4 : scan edge (search edge 3) 4 [839] 839 root143Tail4 winner = winner :=
  (root143Step4).trans root143Scan5
theorem root143Scan3 : scan edge (search edge 3) 4 [839] 839 root143Tail3 winner = winner :=
  (root143Step3).trans root143Scan4
theorem root143Scan2 : scan edge (search edge 3) 4 [839] 839 root143Tail2 winner = winner :=
  (root143Step2).trans root143Scan3
theorem root143Scan1 : scan edge (search edge 3) 4 [839] 839 root143Tail1 winner = winner :=
  (root143Step1).trans root143Scan2
theorem root143Scan0 : scan edge (search edge 3) 4 [839] 839 root143Tail0 winner = winner :=
  (root143Step0).trans root143Scan1
set_option maxRecDepth 4096 in
theorem root143Branch : search edge 4 [839] root143Candidates 839 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [839] 839 root143Tail0 winner = winner
  exact root143Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
