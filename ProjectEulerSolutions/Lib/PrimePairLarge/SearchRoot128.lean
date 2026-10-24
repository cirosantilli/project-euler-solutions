import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root128Candidates : List Nat := Data.row739
set_option maxRecDepth 8192 in
theorem root128Candidates_checked : candidates edge 5 0 739 tail129 winner.bound = root128Candidates := by
  apply candidates_eq_row edge 5 0 739 tail129 root128Candidates winner.bound tail129_ordered
  · decide
  · decide
  · funext q
    rfl
def root128Tail73 : List Nat := []
def root128Tail72 : List Nat := 24247 :: root128Tail73
def root128Tail71 : List Nat := 23893 :: root128Tail72
def root128Tail70 : List Nat := 23869 :: root128Tail71
def root128Tail69 : List Nat := 23071 :: root128Tail70
def root128Tail68 : List Nat := 22699 :: root128Tail69
def root128Tail67 : List Nat := 22669 :: root128Tail68
def root128Tail66 : List Nat := 22291 :: root128Tail67
def root128Tail65 : List Nat := 22129 :: root128Tail66
def root128Tail64 : List Nat := 22003 :: root128Tail65
def root128Tail63 : List Nat := 21871 :: root128Tail64
def root128Tail62 : List Nat := 21481 :: root128Tail63
def root128Tail61 : List Nat := 21283 :: root128Tail62
def root128Tail60 : List Nat := 21157 :: root128Tail61
def root128Tail59 : List Nat := 20719 :: root128Tail60
def root128Tail58 : List Nat := 20347 :: root128Tail59
def root128Tail57 : List Nat := 20173 :: root128Tail58
def root128Tail56 : List Nat := 19609 :: root128Tail57
def root128Tail55 : List Nat := 18661 :: root128Tail56
def root128Tail54 : List Nat := 18301 :: root128Tail55
def root128Tail53 : List Nat := 18181 :: root128Tail54
def root128Tail52 : List Nat := 17911 :: root128Tail53
def root128Tail51 : List Nat := 17713 :: root128Tail52
def root128Tail50 : List Nat := 17209 :: root128Tail51
def root128Tail49 : List Nat := 16987 :: root128Tail50
def root128Tail48 : List Nat := 16333 :: root128Tail49
def root128Tail47 : List Nat := 16189 :: root128Tail48
def root128Tail46 : List Nat := 16057 :: root128Tail47
def root128Tail45 : List Nat := 15937 :: root128Tail46
def root128Tail44 : List Nat := 15823 :: root128Tail45
def root128Tail43 : List Nat := 15679 :: root128Tail44
def root128Tail42 : List Nat := 15619 :: root128Tail43
def root128Tail41 : List Nat := 14983 :: root128Tail42
def root128Tail40 : List Nat := 14713 :: root128Tail41
def root128Tail39 : List Nat := 14563 :: root128Tail40
def root128Tail38 : List Nat := 14293 :: root128Tail39
def root128Tail37 : List Nat := 14149 :: root128Tail38
def root128Tail36 : List Nat := 13831 :: root128Tail37
def root128Tail35 : List Nat := 13711 :: root128Tail36
def root128Tail34 : List Nat := 13327 :: root128Tail35
def root128Tail33 : List Nat := 12763 :: root128Tail34
def root128Tail32 : List Nat := 12757 :: root128Tail33
def root128Tail31 : List Nat := 11317 :: root128Tail32
def root128Tail30 : List Nat := 11059 :: root128Tail31
def root128Tail29 : List Nat := 10939 :: root128Tail30
def root128Tail28 : List Nat := 10663 :: root128Tail29
def root128Tail27 : List Nat := 10597 :: root128Tail28
def root128Tail26 : List Nat := 10429 :: root128Tail27
def root128Tail25 : List Nat := 9697 :: root128Tail26
def root128Tail24 : List Nat := 9463 :: root128Tail25
def root128Tail23 : List Nat := 9283 :: root128Tail24
def root128Tail22 : List Nat := 8647 :: root128Tail23
def root128Tail21 : List Nat := 8431 :: root128Tail22
def root128Tail20 : List Nat := 7699 :: root128Tail21
def root128Tail19 : List Nat := 7393 :: root128Tail20
def root128Tail18 : List Nat := 7219 :: root128Tail19
def root128Tail17 : List Nat := 6841 :: root128Tail18
def root128Tail16 : List Nat := 6793 :: root128Tail17
def root128Tail15 : List Nat := 6673 :: root128Tail16
def root128Tail14 : List Nat := 6007 :: root128Tail15
def root128Tail13 : List Nat := 4909 :: root128Tail14
def root128Tail12 : List Nat := 4759 :: root128Tail13
def root128Tail11 : List Nat := 4549 :: root128Tail12
def root128Tail10 : List Nat := 4357 :: root128Tail11
def root128Tail9 : List Nat := 3733 :: root128Tail10
def root128Tail8 : List Nat := 3613 :: root128Tail9
def root128Tail7 : List Nat := 3187 :: root128Tail8
def root128Tail6 : List Nat := 3121 :: root128Tail7
def root128Tail5 : List Nat := 1381 :: root128Tail6
def root128Tail4 : List Nat := 1123 :: root128Tail5
def root128Tail3 : List Nat := 1093 :: root128Tail4
def root128Tail2 : List Nat := 967 :: root128Tail3
def root128Tail1 : List Nat := 853 :: root128Tail2
def root128Tail0 : List Nat := 751 :: root128Tail1
def root128Candidates0 : List Nat := [8431, 16057, 21481]
set_option maxRecDepth 8192 in
theorem root128Candidates0_checked : candidates edge 4 739 751 root128Tail1 winner.bound = root128Candidates0 := by
  unfold candidates
  have row : edge 751 = (SortedNatTree.contains FastRows.row751) := by
    change Data.row751.contains = (SortedNatTree.contains FastRows.row751)
    exact FastRows.row751_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root128Child0 : search edge 3 [739, 751] root128Candidates0 1490 winner = winner := by decide
theorem root128Step0 : scan edge (search edge 3) 4 [739] 739 root128Tail0 winner =
    scan edge (search edge 3) 4 [739] 739 root128Tail1 winner := by
  rw [root128Tail0, scan, if_neg (by decide), root128Candidates0_checked]
  change scan edge (search edge 3) 4 [739] 739 root128Tail1
    (if 3 ≤ root128Candidates0.length then search edge 3 [739, 751] root128Candidates0 1490 winner else winner) = _
  rw [if_pos (by decide), root128Child0]
def root128Candidates1 : List Nat := [17209, 22129]
set_option maxRecDepth 8192 in
theorem root128Candidates1_checked : candidates edge 4 739 853 root128Tail2 winner.bound = root128Candidates1 := by
  unfold candidates
  have row : edge 853 = (SortedNatTree.contains FastRows.row853) := by
    change Data.row853.contains = (SortedNatTree.contains FastRows.row853)
    exact FastRows.row853_checked.symm
  rw [row]
  decide
theorem root128Step1 : scan edge (search edge 3) 4 [739] 739 root128Tail1 winner =
    scan edge (search edge 3) 4 [739] 739 root128Tail2 winner := by
  rw [root128Tail1, scan, if_neg (by decide), root128Candidates1_checked]
  change scan edge (search edge 3) 4 [739] 739 root128Tail2
    (if 3 ≤ root128Candidates1.length then search edge 3 [739, 853] root128Candidates1 1592 winner else winner) = _
  rw [if_neg (by decide)]
def root128Candidates2 : List Nat := [1381, 14713, 20719]
set_option maxRecDepth 8192 in
theorem root128Candidates2_checked : candidates edge 4 739 967 root128Tail3 winner.bound = root128Candidates2 := by
  unfold candidates
  have row : edge 967 = (SortedNatTree.contains FastRows.row967) := by
    change Data.row967.contains = (SortedNatTree.contains FastRows.row967)
    exact FastRows.row967_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root128Child2 : search edge 3 [739, 967] root128Candidates2 1706 winner = winner := by decide
theorem root128Step2 : scan edge (search edge 3) 4 [739] 739 root128Tail2 winner =
    scan edge (search edge 3) 4 [739] 739 root128Tail3 winner := by
  rw [root128Tail2, scan, if_neg (by decide), root128Candidates2_checked]
  change scan edge (search edge 3) 4 [739] 739 root128Tail3
    (if 3 ≤ root128Candidates2.length then search edge 3 [739, 967] root128Candidates2 1706 winner else winner) = _
  rw [if_pos (by decide), root128Child2]
def root128Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root128Candidates3_checked : candidates edge 4 739 1093 root128Tail4 winner.bound = root128Candidates3 := by
  unfold candidates
  have row : edge 1093 = (SortedNatTree.contains FastRows.row1093) := by
    change Data.row1093.contains = (SortedNatTree.contains FastRows.row1093)
    exact FastRows.row1093_checked.symm
  rw [row]
  decide
theorem root128Step3 : scan edge (search edge 3) 4 [739] 739 root128Tail3 winner =
    scan edge (search edge 3) 4 [739] 739 root128Tail4 winner := by
  rw [root128Tail3, scan, if_neg (by decide), root128Candidates3_checked]
  change scan edge (search edge 3) 4 [739] 739 root128Tail4
    (if 3 ≤ root128Candidates3.length then search edge 3 [739, 1093] root128Candidates3 1832 winner else winner) = _
  rw [if_neg (by decide)]
def root128Candidates4 : List Nat := [15937]
set_option maxRecDepth 8192 in
theorem root128Candidates4_checked : candidates edge 4 739 1123 root128Tail5 winner.bound = root128Candidates4 := by
  unfold candidates
  have row : edge 1123 = (SortedNatTree.contains FastRows.row1123) := by
    change Data.row1123.contains = (SortedNatTree.contains FastRows.row1123)
    exact FastRows.row1123_checked.symm
  rw [row]
  decide
theorem root128Step4 : scan edge (search edge 3) 4 [739] 739 root128Tail4 winner =
    scan edge (search edge 3) 4 [739] 739 root128Tail5 winner := by
  rw [root128Tail4, scan, if_neg (by decide), root128Candidates4_checked]
  change scan edge (search edge 3) 4 [739] 739 root128Tail5
    (if 3 ≤ root128Candidates4.length then search edge 3 [739, 1123] root128Candidates4 1862 winner else winner) = _
  rw [if_neg (by decide)]
def root128Candidates5 : List Nat := [10597, 16333, 18181, 22129]
set_option maxRecDepth 8192 in
theorem root128Candidates5_checked : candidates edge 4 739 1381 root128Tail6 winner.bound = root128Candidates5 := by
  unfold candidates
  have row : edge 1381 = (SortedNatTree.contains FastRows.row1381) := by
    change Data.row1381.contains = (SortedNatTree.contains FastRows.row1381)
    exact FastRows.row1381_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root128Child5 : search edge 3 [739, 1381] root128Candidates5 2120 winner = winner := by decide
theorem root128Step5 : scan edge (search edge 3) 4 [739] 739 root128Tail5 winner =
    scan edge (search edge 3) 4 [739] 739 root128Tail6 winner := by
  rw [root128Tail5, scan, if_neg (by decide), root128Candidates5_checked]
  change scan edge (search edge 3) 4 [739] 739 root128Tail6
    (if 3 ≤ root128Candidates5.length then search edge 3 [739, 1381] root128Candidates5 2120 winner else winner) = _
  rw [if_pos (by decide), root128Child5]
def root128Candidates6 : List Nat := [4357, 7393, 14563]
set_option maxRecDepth 8192 in
theorem root128Candidates6_checked : candidates edge 4 739 3121 root128Tail7 winner.bound = root128Candidates6 := by
  unfold candidates
  have row : edge 3121 = (SortedNatTree.contains FastRows.row3121) := by
    change Data.row3121.contains = (SortedNatTree.contains FastRows.row3121)
    exact FastRows.row3121_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root128Child6 : search edge 3 [739, 3121] root128Candidates6 3860 winner = winner := by decide
theorem root128Step6 : scan edge (search edge 3) 4 [739] 739 root128Tail6 winner =
    scan edge (search edge 3) 4 [739] 739 root128Tail7 winner := by
  rw [root128Tail6, scan, if_neg (by decide), root128Candidates6_checked]
  change scan edge (search edge 3) 4 [739] 739 root128Tail7
    (if 3 ≤ root128Candidates6.length then search edge 3 [739, 3121] root128Candidates6 3860 winner else winner) = _
  rw [if_pos (by decide), root128Child6]
def root128Candidates7 : List Nat := [18181, 20173]
set_option maxRecDepth 8192 in
theorem root128Candidates7_checked : candidates edge 4 739 3187 root128Tail8 winner.bound = root128Candidates7 := by
  unfold candidates
  have row : edge 3187 = (SortedNatTree.contains FastRows.row3187) := by
    change Data.row3187.contains = (SortedNatTree.contains FastRows.row3187)
    exact FastRows.row3187_checked.symm
  rw [row]
  decide
theorem root128Step7 : scan edge (search edge 3) 4 [739] 739 root128Tail7 winner =
    scan edge (search edge 3) 4 [739] 739 root128Tail8 winner := by
  rw [root128Tail7, scan, if_neg (by decide), root128Candidates7_checked]
  change scan edge (search edge 3) 4 [739] 739 root128Tail8
    (if 3 ≤ root128Candidates7.length then search edge 3 [739, 3187] root128Candidates7 3926 winner else winner) = _
  rw [if_neg (by decide)]
def root128Candidates8 : List Nat := [14293, 14983]
set_option maxRecDepth 8192 in
theorem root128Candidates8_checked : candidates edge 4 739 3613 root128Tail9 winner.bound = root128Candidates8 := by
  unfold candidates
  have row : edge 3613 = (SortedNatTree.contains FastRows.row3613) := by
    change Data.row3613.contains = (SortedNatTree.contains FastRows.row3613)
    exact FastRows.row3613_checked.symm
  rw [row]
  decide
theorem root128Step8 : scan edge (search edge 3) 4 [739] 739 root128Tail8 winner =
    scan edge (search edge 3) 4 [739] 739 root128Tail9 winner := by
  rw [root128Tail8, scan, if_neg (by decide), root128Candidates8_checked]
  change scan edge (search edge 3) 4 [739] 739 root128Tail9
    (if 3 ≤ root128Candidates8.length then search edge 3 [739, 3613] root128Candidates8 4352 winner else winner) = _
  rw [if_neg (by decide)]
def root128Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root128Candidates9_checked : candidates edge 4 739 3733 root128Tail10 winner.bound = root128Candidates9 := by
  unfold candidates
  have row : edge 3733 = (SortedNatTree.contains FastRows.row3733) := by
    change Data.row3733.contains = (SortedNatTree.contains FastRows.row3733)
    exact FastRows.row3733_checked.symm
  rw [row]
  decide
theorem root128Step9 : scan edge (search edge 3) 4 [739] 739 root128Tail9 winner =
    scan edge (search edge 3) 4 [739] 739 root128Tail10 winner := by
  rw [root128Tail9, scan, if_neg (by decide), root128Candidates9_checked]
  change scan edge (search edge 3) 4 [739] 739 root128Tail10
    (if 3 ≤ root128Candidates9.length then search edge 3 [739, 3733] root128Candidates9 4472 winner else winner) = _
  rw [if_neg (by decide)]
def root128Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root128Candidates10_checked : candidates edge 4 739 4357 root128Tail11 winner.bound = root128Candidates10 := by
  unfold candidates
  have row : edge 4357 = (SortedNatTree.contains FastRows.row4357) := by
    change Data.row4357.contains = (SortedNatTree.contains FastRows.row4357)
    exact FastRows.row4357_checked.symm
  rw [row]
  decide
theorem root128Step10 : scan edge (search edge 3) 4 [739] 739 root128Tail10 winner =
    scan edge (search edge 3) 4 [739] 739 root128Tail11 winner := by
  rw [root128Tail10, scan, if_neg (by decide), root128Candidates10_checked]
  change scan edge (search edge 3) 4 [739] 739 root128Tail11
    (if 3 ≤ root128Candidates10.length then search edge 3 [739, 4357] root128Candidates10 5096 winner else winner) = _
  rw [if_neg (by decide)]
def root128Candidates11 : List Nat := [18301]
set_option maxRecDepth 8192 in
theorem root128Candidates11_checked : candidates edge 4 739 4549 root128Tail12 winner.bound = root128Candidates11 := by
  unfold candidates
  have row : edge 4549 = (SortedNatTree.contains FastRows.row4549) := by
    change Data.row4549.contains = (SortedNatTree.contains FastRows.row4549)
    exact FastRows.row4549_checked.symm
  rw [row]
  decide
theorem root128Step11 : scan edge (search edge 3) 4 [739] 739 root128Tail11 winner =
    scan edge (search edge 3) 4 [739] 739 root128Tail12 winner := by
  rw [root128Tail11, scan, if_neg (by decide), root128Candidates11_checked]
  change scan edge (search edge 3) 4 [739] 739 root128Tail12
    (if 3 ≤ root128Candidates11.length then search edge 3 [739, 4549] root128Candidates11 5288 winner else winner) = _
  rw [if_neg (by decide)]
def root128Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root128Candidates12_checked : candidates edge 4 739 4759 root128Tail13 winner.bound = root128Candidates12 := by
  unfold candidates
  have row : edge 4759 = (SortedNatTree.contains FastRows.row4759) := by
    change Data.row4759.contains = (SortedNatTree.contains FastRows.row4759)
    exact FastRows.row4759_checked.symm
  rw [row]
  decide
theorem root128Step12 : scan edge (search edge 3) 4 [739] 739 root128Tail12 winner =
    scan edge (search edge 3) 4 [739] 739 root128Tail13 winner := by
  rw [root128Tail12, scan, if_neg (by decide), root128Candidates12_checked]
  change scan edge (search edge 3) 4 [739] 739 root128Tail13
    (if 3 ≤ root128Candidates12.length then search edge 3 [739, 4759] root128Candidates12 5498 winner else winner) = _
  rw [if_neg (by decide)]
def root128Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root128Candidates13_checked : candidates edge 4 739 4909 root128Tail14 winner.bound = root128Candidates13 := by
  unfold candidates
  have row : edge 4909 = (SortedNatTree.contains FastRows.row4909) := by
    change Data.row4909.contains = (SortedNatTree.contains FastRows.row4909)
    exact FastRows.row4909_checked.symm
  rw [row]
  decide
theorem root128Step13 : scan edge (search edge 3) 4 [739] 739 root128Tail13 winner =
    scan edge (search edge 3) 4 [739] 739 root128Tail14 winner := by
  rw [root128Tail13, scan, if_neg (by decide), root128Candidates13_checked]
  change scan edge (search edge 3) 4 [739] 739 root128Tail14
    (if 3 ≤ root128Candidates13.length then search edge 3 [739, 4909] root128Candidates13 5648 winner else winner) = _
  rw [if_neg (by decide)]
def root128Candidates14 : List Nat := [10597, 13831, 18661]
set_option maxRecDepth 8192 in
theorem root128Candidates14_checked : candidates edge 4 739 6007 root128Tail15 winner.bound = root128Candidates14 := by
  unfold candidates
  have row : edge 6007 = (SortedNatTree.contains FastRows.row6007) := by
    change Data.row6007.contains = (SortedNatTree.contains FastRows.row6007)
    exact FastRows.row6007_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root128Child14 : search edge 3 [739, 6007] root128Candidates14 6746 winner = winner := by decide
theorem root128Step14 : scan edge (search edge 3) 4 [739] 739 root128Tail14 winner =
    scan edge (search edge 3) 4 [739] 739 root128Tail15 winner := by
  rw [root128Tail14, scan, if_neg (by decide), root128Candidates14_checked]
  change scan edge (search edge 3) 4 [739] 739 root128Tail15
    (if 3 ≤ root128Candidates14.length then search edge 3 [739, 6007] root128Candidates14 6746 winner else winner) = _
  rw [if_pos (by decide), root128Child14]
theorem root128Scan15 : scan edge (search edge 3) 4 [739] 739 root128Tail15 winner = winner := by rw [root128Tail15, scan, if_pos (by decide)]
theorem root128Scan14 : scan edge (search edge 3) 4 [739] 739 root128Tail14 winner = winner :=
  (root128Step14).trans root128Scan15
theorem root128Scan13 : scan edge (search edge 3) 4 [739] 739 root128Tail13 winner = winner :=
  (root128Step13).trans root128Scan14
theorem root128Scan12 : scan edge (search edge 3) 4 [739] 739 root128Tail12 winner = winner :=
  (root128Step12).trans root128Scan13
theorem root128Scan11 : scan edge (search edge 3) 4 [739] 739 root128Tail11 winner = winner :=
  (root128Step11).trans root128Scan12
theorem root128Scan10 : scan edge (search edge 3) 4 [739] 739 root128Tail10 winner = winner :=
  (root128Step10).trans root128Scan11
theorem root128Scan9 : scan edge (search edge 3) 4 [739] 739 root128Tail9 winner = winner :=
  (root128Step9).trans root128Scan10
theorem root128Scan8 : scan edge (search edge 3) 4 [739] 739 root128Tail8 winner = winner :=
  (root128Step8).trans root128Scan9
theorem root128Scan7 : scan edge (search edge 3) 4 [739] 739 root128Tail7 winner = winner :=
  (root128Step7).trans root128Scan8
theorem root128Scan6 : scan edge (search edge 3) 4 [739] 739 root128Tail6 winner = winner :=
  (root128Step6).trans root128Scan7
theorem root128Scan5 : scan edge (search edge 3) 4 [739] 739 root128Tail5 winner = winner :=
  (root128Step5).trans root128Scan6
theorem root128Scan4 : scan edge (search edge 3) 4 [739] 739 root128Tail4 winner = winner :=
  (root128Step4).trans root128Scan5
theorem root128Scan3 : scan edge (search edge 3) 4 [739] 739 root128Tail3 winner = winner :=
  (root128Step3).trans root128Scan4
theorem root128Scan2 : scan edge (search edge 3) 4 [739] 739 root128Tail2 winner = winner :=
  (root128Step2).trans root128Scan3
theorem root128Scan1 : scan edge (search edge 3) 4 [739] 739 root128Tail1 winner = winner :=
  (root128Step1).trans root128Scan2
theorem root128Scan0 : scan edge (search edge 3) 4 [739] 739 root128Tail0 winner = winner :=
  (root128Step0).trans root128Scan1
set_option maxRecDepth 4096 in
theorem root128Branch : search edge 4 [739] root128Candidates 739 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [739] 739 root128Tail0 winner = winner
  exact root128Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
