import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root253Candidates : List Nat := Data.row1619
set_option maxRecDepth 8192 in
theorem root253Candidates_checked : candidates edge 5 0 1619 tail254 winner.bound = root253Candidates := by
  apply candidates_eq_row edge 5 0 1619 tail254 root253Candidates winner.bound tail254_ordered
  · decide
  · decide
  · funext q
    rfl
def root253Tail54 : List Nat := []
def root253Tail53 : List Nat := 24239 :: root253Tail54
def root253Tail52 : List Nat := 23627 :: root253Tail53
def root253Tail51 : List Nat := 23291 :: root253Tail52
def root253Tail50 : List Nat := 22961 :: root253Tail51
def root253Tail49 : List Nat := 22247 :: root253Tail50
def root253Tail48 : List Nat := 21737 :: root253Tail49
def root253Tail47 : List Nat := 21191 :: root253Tail48
def root253Tail46 : List Nat := 20921 :: root253Tail47
def root253Tail45 : List Nat := 19949 :: root253Tail46
def root253Tail44 : List Nat := 19937 :: root253Tail45
def root253Tail43 : List Nat := 19739 :: root253Tail44
def root253Tail42 : List Nat := 19181 :: root253Tail43
def root253Tail41 : List Nat := 18671 :: root253Tail42
def root253Tail40 : List Nat := 18311 :: root253Tail41
def root253Tail39 : List Nat := 17099 :: root253Tail40
def root253Tail38 : List Nat := 16673 :: root253Tail39
def root253Tail37 : List Nat := 16451 :: root253Tail38
def root253Tail36 : List Nat := 15107 :: root253Tail37
def root253Tail35 : List Nat := 14303 :: root253Tail36
def root253Tail34 : List Nat := 13109 :: root253Tail35
def root253Tail33 : List Nat := 12269 :: root253Tail34
def root253Tail32 : List Nat := 12119 :: root253Tail33
def root253Tail31 : List Nat := 11447 :: root253Tail32
def root253Tail30 : List Nat := 10733 :: root253Tail31
def root253Tail29 : List Nat := 10289 :: root253Tail30
def root253Tail28 : List Nat := 9749 :: root253Tail29
def root253Tail27 : List Nat := 9041 :: root253Tail28
def root253Tail26 : List Nat := 8951 :: root253Tail27
def root253Tail25 : List Nat := 8783 :: root253Tail26
def root253Tail24 : List Nat := 8747 :: root253Tail25
def root253Tail23 : List Nat := 8513 :: root253Tail24
def root253Tail22 : List Nat := 7949 :: root253Tail23
def root253Tail21 : List Nat := 7331 :: root253Tail22
def root253Tail20 : List Nat := 6701 :: root253Tail21
def root253Tail19 : List Nat := 6197 :: root253Tail20
def root253Tail18 : List Nat := 6143 :: root253Tail19
def root253Tail17 : List Nat := 5987 :: root253Tail18
def root253Tail16 : List Nat := 5903 :: root253Tail17
def root253Tail15 : List Nat := 5801 :: root253Tail16
def root253Tail14 : List Nat := 5573 :: root253Tail15
def root253Tail13 : List Nat := 5279 :: root253Tail14
def root253Tail12 : List Nat := 4943 :: root253Tail13
def root253Tail11 : List Nat := 4817 :: root253Tail12
def root253Tail10 : List Nat := 4547 :: root253Tail11
def root253Tail9 : List Nat := 4127 :: root253Tail10
def root253Tail8 : List Nat := 3677 :: root253Tail9
def root253Tail7 : List Nat := 3407 :: root253Tail8
def root253Tail6 : List Nat := 3257 :: root253Tail7
def root253Tail5 : List Nat := 3119 :: root253Tail6
def root253Tail4 : List Nat := 2777 :: root253Tail5
def root253Tail3 : List Nat := 2531 :: root253Tail4
def root253Tail2 : List Nat := 2339 :: root253Tail3
def root253Tail1 : List Nat := 2213 :: root253Tail2
def root253Tail0 : List Nat := 2003 :: root253Tail1
def root253Candidates0 : List Nat := [8747, 16673, 19937]
set_option maxRecDepth 8192 in
theorem root253Candidates0_checked : candidates edge 4 1619 2003 root253Tail1 winner.bound = root253Candidates0 := by
  unfold candidates
  have row : edge 2003 = (SortedNatTree.contains FastRows.row2003) := by
    change Data.row2003.contains = (SortedNatTree.contains FastRows.row2003)
    exact FastRows.row2003_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root253Child0 : search edge 3 [1619, 2003] root253Candidates0 3622 winner = winner := by decide
theorem root253Step0 : scan edge (search edge 3) 4 [1619] 1619 root253Tail0 winner =
    scan edge (search edge 3) 4 [1619] 1619 root253Tail1 winner := by
  rw [root253Tail0, scan, if_neg (by decide), root253Candidates0_checked]
  change scan edge (search edge 3) 4 [1619] 1619 root253Tail1
    (if 3 ≤ root253Candidates0.length then search edge 3 [1619, 2003] root253Candidates0 3622 winner else winner) = _
  rw [if_pos (by decide), root253Child0]
def root253Candidates1 : List Nat := [19181]
set_option maxRecDepth 8192 in
theorem root253Candidates1_checked : candidates edge 4 1619 2213 root253Tail2 winner.bound = root253Candidates1 := by
  unfold candidates
  have row : edge 2213 = (SortedNatTree.contains FastRows.row2213) := by
    change Data.row2213.contains = (SortedNatTree.contains FastRows.row2213)
    exact FastRows.row2213_checked.symm
  rw [row]
  decide
theorem root253Step1 : scan edge (search edge 3) 4 [1619] 1619 root253Tail1 winner =
    scan edge (search edge 3) 4 [1619] 1619 root253Tail2 winner := by
  rw [root253Tail1, scan, if_neg (by decide), root253Candidates1_checked]
  change scan edge (search edge 3) 4 [1619] 1619 root253Tail2
    (if 3 ≤ root253Candidates1.length then search edge 3 [1619, 2213] root253Candidates1 3832 winner else winner) = _
  rw [if_neg (by decide)]
def root253Candidates2 : List Nat := [4127, 16451]
set_option maxRecDepth 8192 in
theorem root253Candidates2_checked : candidates edge 4 1619 2339 root253Tail3 winner.bound = root253Candidates2 := by
  unfold candidates
  have row : edge 2339 = (SortedNatTree.contains FastRows.row2339) := by
    change Data.row2339.contains = (SortedNatTree.contains FastRows.row2339)
    exact FastRows.row2339_checked.symm
  rw [row]
  decide
theorem root253Step2 : scan edge (search edge 3) 4 [1619] 1619 root253Tail2 winner =
    scan edge (search edge 3) 4 [1619] 1619 root253Tail3 winner := by
  rw [root253Tail2, scan, if_neg (by decide), root253Candidates2_checked]
  change scan edge (search edge 3) 4 [1619] 1619 root253Tail3
    (if 3 ≤ root253Candidates2.length then search edge 3 [1619, 2339] root253Candidates2 3958 winner else winner) = _
  rw [if_neg (by decide)]
def root253Candidates3 : List Nat := [6701, 17099]
set_option maxRecDepth 8192 in
theorem root253Candidates3_checked : candidates edge 4 1619 2531 root253Tail4 winner.bound = root253Candidates3 := by
  unfold candidates
  have row : edge 2531 = (SortedNatTree.contains FastRows.row2531) := by
    change Data.row2531.contains = (SortedNatTree.contains FastRows.row2531)
    exact FastRows.row2531_checked.symm
  rw [row]
  decide
theorem root253Step3 : scan edge (search edge 3) 4 [1619] 1619 root253Tail3 winner =
    scan edge (search edge 3) 4 [1619] 1619 root253Tail4 winner := by
  rw [root253Tail3, scan, if_neg (by decide), root253Candidates3_checked]
  change scan edge (search edge 3) 4 [1619] 1619 root253Tail4
    (if 3 ≤ root253Candidates3.length then search edge 3 [1619, 2531] root253Candidates3 4150 winner else winner) = _
  rw [if_neg (by decide)]
def root253Candidates4 : List Nat := [18671]
set_option maxRecDepth 8192 in
theorem root253Candidates4_checked : candidates edge 4 1619 2777 root253Tail5 winner.bound = root253Candidates4 := by
  unfold candidates
  have row : edge 2777 = (SortedNatTree.contains FastRows.row2777) := by
    change Data.row2777.contains = (SortedNatTree.contains FastRows.row2777)
    exact FastRows.row2777_checked.symm
  rw [row]
  decide
theorem root253Step4 : scan edge (search edge 3) 4 [1619] 1619 root253Tail4 winner =
    scan edge (search edge 3) 4 [1619] 1619 root253Tail5 winner := by
  rw [root253Tail4, scan, if_neg (by decide), root253Candidates4_checked]
  change scan edge (search edge 3) 4 [1619] 1619 root253Tail5
    (if 3 ≤ root253Candidates4.length then search edge 3 [1619, 2777] root253Candidates4 4396 winner else winner) = _
  rw [if_neg (by decide)]
def root253Candidates5 : List Nat := [3677]
set_option maxRecDepth 8192 in
theorem root253Candidates5_checked : candidates edge 4 1619 3119 root253Tail6 winner.bound = root253Candidates5 := by
  unfold candidates
  have row : edge 3119 = (SortedNatTree.contains FastRows.row3119) := by
    change Data.row3119.contains = (SortedNatTree.contains FastRows.row3119)
    exact FastRows.row3119_checked.symm
  rw [row]
  decide
theorem root253Step5 : scan edge (search edge 3) 4 [1619] 1619 root253Tail5 winner =
    scan edge (search edge 3) 4 [1619] 1619 root253Tail6 winner := by
  rw [root253Tail5, scan, if_neg (by decide), root253Candidates5_checked]
  change scan edge (search edge 3) 4 [1619] 1619 root253Tail6
    (if 3 ≤ root253Candidates5.length then search edge 3 [1619, 3119] root253Candidates5 4738 winner else winner) = _
  rw [if_neg (by decide)]
def root253Candidates6 : List Nat := [3407, 4547, 7949]
set_option maxRecDepth 8192 in
theorem root253Candidates6_checked : candidates edge 4 1619 3257 root253Tail7 winner.bound = root253Candidates6 := by
  unfold candidates
  have row : edge 3257 = (SortedNatTree.contains FastRows.row3257) := by
    change Data.row3257.contains = (SortedNatTree.contains FastRows.row3257)
    exact FastRows.row3257_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root253Child6 : search edge 3 [1619, 3257] root253Candidates6 4876 winner = winner := by decide
theorem root253Step6 : scan edge (search edge 3) 4 [1619] 1619 root253Tail6 winner =
    scan edge (search edge 3) 4 [1619] 1619 root253Tail7 winner := by
  rw [root253Tail6, scan, if_neg (by decide), root253Candidates6_checked]
  change scan edge (search edge 3) 4 [1619] 1619 root253Tail7
    (if 3 ≤ root253Candidates6.length then search edge 3 [1619, 3257] root253Candidates6 4876 winner else winner) = _
  rw [if_pos (by decide), root253Child6]
def root253Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root253Candidates7_checked : candidates edge 4 1619 3407 root253Tail8 winner.bound = root253Candidates7 := by
  unfold candidates
  have row : edge 3407 = (SortedNatTree.contains FastRows.row3407) := by
    change Data.row3407.contains = (SortedNatTree.contains FastRows.row3407)
    exact FastRows.row3407_checked.symm
  rw [row]
  decide
theorem root253Step7 : scan edge (search edge 3) 4 [1619] 1619 root253Tail7 winner =
    scan edge (search edge 3) 4 [1619] 1619 root253Tail8 winner := by
  rw [root253Tail7, scan, if_neg (by decide), root253Candidates7_checked]
  change scan edge (search edge 3) 4 [1619] 1619 root253Tail8
    (if 3 ≤ root253Candidates7.length then search edge 3 [1619, 3407] root253Candidates7 5026 winner else winner) = _
  rw [if_neg (by decide)]
def root253Candidates8 : List Nat := [5903, 12269]
set_option maxRecDepth 8192 in
theorem root253Candidates8_checked : candidates edge 4 1619 3677 root253Tail9 winner.bound = root253Candidates8 := by
  unfold candidates
  have row : edge 3677 = (SortedNatTree.contains FastRows.row3677) := by
    change Data.row3677.contains = (SortedNatTree.contains FastRows.row3677)
    exact FastRows.row3677_checked.symm
  rw [row]
  decide
theorem root253Step8 : scan edge (search edge 3) 4 [1619] 1619 root253Tail8 winner =
    scan edge (search edge 3) 4 [1619] 1619 root253Tail9 winner := by
  rw [root253Tail8, scan, if_neg (by decide), root253Candidates8_checked]
  change scan edge (search edge 3) 4 [1619] 1619 root253Tail9
    (if 3 ≤ root253Candidates8.length then search edge 3 [1619, 3677] root253Candidates8 5296 winner else winner) = _
  rw [if_neg (by decide)]
def root253Candidates9 : List Nat := [7949]
set_option maxRecDepth 8192 in
theorem root253Candidates9_checked : candidates edge 4 1619 4127 root253Tail10 winner.bound = root253Candidates9 := by
  unfold candidates
  have row : edge 4127 = (SortedNatTree.contains FastRows.row4127) := by
    change Data.row4127.contains = (SortedNatTree.contains FastRows.row4127)
    exact FastRows.row4127_checked.symm
  rw [row]
  decide
theorem root253Step9 : scan edge (search edge 3) 4 [1619] 1619 root253Tail9 winner =
    scan edge (search edge 3) 4 [1619] 1619 root253Tail10 winner := by
  rw [root253Tail9, scan, if_neg (by decide), root253Candidates9_checked]
  change scan edge (search edge 3) 4 [1619] 1619 root253Tail10
    (if 3 ≤ root253Candidates9.length then search edge 3 [1619, 4127] root253Candidates9 5746 winner else winner) = _
  rw [if_neg (by decide)]
def root253Candidates10 : List Nat := [4943]
set_option maxRecDepth 8192 in
theorem root253Candidates10_checked : candidates edge 4 1619 4547 root253Tail11 winner.bound = root253Candidates10 := by
  unfold candidates
  have row : edge 4547 = (SortedNatTree.contains FastRows.row4547) := by
    change Data.row4547.contains = (SortedNatTree.contains FastRows.row4547)
    exact FastRows.row4547_checked.symm
  rw [row]
  decide
theorem root253Step10 : scan edge (search edge 3) 4 [1619] 1619 root253Tail10 winner =
    scan edge (search edge 3) 4 [1619] 1619 root253Tail11 winner := by
  rw [root253Tail10, scan, if_neg (by decide), root253Candidates10_checked]
  change scan edge (search edge 3) 4 [1619] 1619 root253Tail11
    (if 3 ≤ root253Candidates10.length then search edge 3 [1619, 4547] root253Candidates10 6166 winner else winner) = _
  rw [if_neg (by decide)]
def root253Candidates11 : List Nat := [5903, 6197]
set_option maxRecDepth 8192 in
theorem root253Candidates11_checked : candidates edge 4 1619 4817 root253Tail12 winner.bound = root253Candidates11 := by
  unfold candidates
  have row : edge 4817 = (SortedNatTree.contains FastRows.row4817) := by
    change Data.row4817.contains = (SortedNatTree.contains FastRows.row4817)
    exact FastRows.row4817_checked.symm
  rw [row]
  decide
theorem root253Step11 : scan edge (search edge 3) 4 [1619] 1619 root253Tail11 winner =
    scan edge (search edge 3) 4 [1619] 1619 root253Tail12 winner := by
  rw [root253Tail11, scan, if_neg (by decide), root253Candidates11_checked]
  change scan edge (search edge 3) 4 [1619] 1619 root253Tail12
    (if 3 ≤ root253Candidates11.length then search edge 3 [1619, 4817] root253Candidates11 6436 winner else winner) = _
  rw [if_neg (by decide)]
def root253Candidates12 : List Nat := [6701]
set_option maxRecDepth 8192 in
theorem root253Candidates12_checked : candidates edge 4 1619 4943 root253Tail13 winner.bound = root253Candidates12 := by
  unfold candidates
  have row : edge 4943 = (SortedNatTree.contains FastRows.row4943) := by
    change Data.row4943.contains = (SortedNatTree.contains FastRows.row4943)
    exact FastRows.row4943_checked.symm
  rw [row]
  decide
theorem root253Step12 : scan edge (search edge 3) 4 [1619] 1619 root253Tail12 winner =
    scan edge (search edge 3) 4 [1619] 1619 root253Tail13 winner := by
  rw [root253Tail12, scan, if_neg (by decide), root253Candidates12_checked]
  change scan edge (search edge 3) 4 [1619] 1619 root253Tail13
    (if 3 ≤ root253Candidates12.length then search edge 3 [1619, 4943] root253Candidates12 6562 winner else winner) = _
  rw [if_neg (by decide)]
def root253Candidates13 : List Nat := [15107]
set_option maxRecDepth 8192 in
theorem root253Candidates13_checked : candidates edge 4 1619 5279 root253Tail14 winner.bound = root253Candidates13 := by
  unfold candidates
  have row : edge 5279 = (SortedNatTree.contains FastRows.row5279) := by
    change Data.row5279.contains = (SortedNatTree.contains FastRows.row5279)
    exact FastRows.row5279_checked.symm
  rw [row]
  decide
theorem root253Step13 : scan edge (search edge 3) 4 [1619] 1619 root253Tail13 winner =
    scan edge (search edge 3) 4 [1619] 1619 root253Tail14 winner := by
  rw [root253Tail13, scan, if_neg (by decide), root253Candidates13_checked]
  change scan edge (search edge 3) 4 [1619] 1619 root253Tail14
    (if 3 ≤ root253Candidates13.length then search edge 3 [1619, 5279] root253Candidates13 6898 winner else winner) = _
  rw [if_neg (by decide)]
def root253Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root253Candidates14_checked : candidates edge 4 1619 5573 root253Tail15 winner.bound = root253Candidates14 := by
  unfold candidates
  have row : edge 5573 = (SortedNatTree.contains FastRows.row5573) := by
    change Data.row5573.contains = (SortedNatTree.contains FastRows.row5573)
    exact FastRows.row5573_checked.symm
  rw [row]
  decide
theorem root253Step14 : scan edge (search edge 3) 4 [1619] 1619 root253Tail14 winner =
    scan edge (search edge 3) 4 [1619] 1619 root253Tail15 winner := by
  rw [root253Tail14, scan, if_neg (by decide), root253Candidates14_checked]
  change scan edge (search edge 3) 4 [1619] 1619 root253Tail15
    (if 3 ≤ root253Candidates14.length then search edge 3 [1619, 5573] root253Candidates14 7192 winner else winner) = _
  rw [if_neg (by decide)]
def root253Candidates15 : List Nat := [8783]
set_option maxRecDepth 8192 in
theorem root253Candidates15_checked : candidates edge 4 1619 5801 root253Tail16 winner.bound = root253Candidates15 := by
  unfold candidates
  have row : edge 5801 = (SortedNatTree.contains FastRows.row5801) := by
    change Data.row5801.contains = (SortedNatTree.contains FastRows.row5801)
    exact FastRows.row5801_checked.symm
  rw [row]
  decide
theorem root253Step15 : scan edge (search edge 3) 4 [1619] 1619 root253Tail15 winner =
    scan edge (search edge 3) 4 [1619] 1619 root253Tail16 winner := by
  rw [root253Tail15, scan, if_neg (by decide), root253Candidates15_checked]
  change scan edge (search edge 3) 4 [1619] 1619 root253Tail16
    (if 3 ≤ root253Candidates15.length then search edge 3 [1619, 5801] root253Candidates15 7420 winner else winner) = _
  rw [if_neg (by decide)]
def root253Candidates16 : List Nat := [6143]
set_option maxRecDepth 8192 in
theorem root253Candidates16_checked : candidates edge 4 1619 5903 root253Tail17 winner.bound = root253Candidates16 := by
  unfold candidates
  have row : edge 5903 = (SortedNatTree.contains FastRows.row5903) := by
    change Data.row5903.contains = (SortedNatTree.contains FastRows.row5903)
    exact FastRows.row5903_checked.symm
  rw [row]
  decide
theorem root253Step16 : scan edge (search edge 3) 4 [1619] 1619 root253Tail16 winner =
    scan edge (search edge 3) 4 [1619] 1619 root253Tail17 winner := by
  rw [root253Tail16, scan, if_neg (by decide), root253Candidates16_checked]
  change scan edge (search edge 3) 4 [1619] 1619 root253Tail17
    (if 3 ≤ root253Candidates16.length then search edge 3 [1619, 5903] root253Candidates16 7522 winner else winner) = _
  rw [if_neg (by decide)]
def root253Candidates17 : List Nat := [8513, 11447, 17099]
set_option maxRecDepth 8192 in
theorem root253Candidates17_checked : candidates edge 4 1619 5987 root253Tail18 winner.bound = root253Candidates17 := by
  unfold candidates
  have row : edge 5987 = (SortedNatTree.contains FastRows.row5987) := by
    change Data.row5987.contains = (SortedNatTree.contains FastRows.row5987)
    exact FastRows.row5987_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root253Child17 : search edge 3 [1619, 5987] root253Candidates17 7606 winner = winner := by decide
theorem root253Step17 : scan edge (search edge 3) 4 [1619] 1619 root253Tail17 winner =
    scan edge (search edge 3) 4 [1619] 1619 root253Tail18 winner := by
  rw [root253Tail17, scan, if_neg (by decide), root253Candidates17_checked]
  change scan edge (search edge 3) 4 [1619] 1619 root253Tail18
    (if 3 ≤ root253Candidates17.length then search edge 3 [1619, 5987] root253Candidates17 7606 winner else winner) = _
  rw [if_pos (by decide), root253Child17]
theorem root253Scan18 : scan edge (search edge 3) 4 [1619] 1619 root253Tail18 winner = winner := by rw [root253Tail18, scan, if_pos (by decide)]
theorem root253Scan17 : scan edge (search edge 3) 4 [1619] 1619 root253Tail17 winner = winner :=
  (root253Step17).trans root253Scan18
theorem root253Scan16 : scan edge (search edge 3) 4 [1619] 1619 root253Tail16 winner = winner :=
  (root253Step16).trans root253Scan17
theorem root253Scan15 : scan edge (search edge 3) 4 [1619] 1619 root253Tail15 winner = winner :=
  (root253Step15).trans root253Scan16
theorem root253Scan14 : scan edge (search edge 3) 4 [1619] 1619 root253Tail14 winner = winner :=
  (root253Step14).trans root253Scan15
theorem root253Scan13 : scan edge (search edge 3) 4 [1619] 1619 root253Tail13 winner = winner :=
  (root253Step13).trans root253Scan14
theorem root253Scan12 : scan edge (search edge 3) 4 [1619] 1619 root253Tail12 winner = winner :=
  (root253Step12).trans root253Scan13
theorem root253Scan11 : scan edge (search edge 3) 4 [1619] 1619 root253Tail11 winner = winner :=
  (root253Step11).trans root253Scan12
theorem root253Scan10 : scan edge (search edge 3) 4 [1619] 1619 root253Tail10 winner = winner :=
  (root253Step10).trans root253Scan11
theorem root253Scan9 : scan edge (search edge 3) 4 [1619] 1619 root253Tail9 winner = winner :=
  (root253Step9).trans root253Scan10
theorem root253Scan8 : scan edge (search edge 3) 4 [1619] 1619 root253Tail8 winner = winner :=
  (root253Step8).trans root253Scan9
theorem root253Scan7 : scan edge (search edge 3) 4 [1619] 1619 root253Tail7 winner = winner :=
  (root253Step7).trans root253Scan8
theorem root253Scan6 : scan edge (search edge 3) 4 [1619] 1619 root253Tail6 winner = winner :=
  (root253Step6).trans root253Scan7
theorem root253Scan5 : scan edge (search edge 3) 4 [1619] 1619 root253Tail5 winner = winner :=
  (root253Step5).trans root253Scan6
theorem root253Scan4 : scan edge (search edge 3) 4 [1619] 1619 root253Tail4 winner = winner :=
  (root253Step4).trans root253Scan5
theorem root253Scan3 : scan edge (search edge 3) 4 [1619] 1619 root253Tail3 winner = winner :=
  (root253Step3).trans root253Scan4
theorem root253Scan2 : scan edge (search edge 3) 4 [1619] 1619 root253Tail2 winner = winner :=
  (root253Step2).trans root253Scan3
theorem root253Scan1 : scan edge (search edge 3) 4 [1619] 1619 root253Tail1 winner = winner :=
  (root253Step1).trans root253Scan2
theorem root253Scan0 : scan edge (search edge 3) 4 [1619] 1619 root253Tail0 winner = winner :=
  (root253Step0).trans root253Scan1
set_option maxRecDepth 4096 in
theorem root253Branch : search edge 4 [1619] root253Candidates 1619 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1619] 1619 root253Tail0 winner = winner
  exact root253Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
