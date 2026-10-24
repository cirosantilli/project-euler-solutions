import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root254Candidates : List Nat := Data.row1621
set_option maxRecDepth 8192 in
theorem root254Candidates_checked : candidates edge 5 0 1621 tail255 winner.bound = root254Candidates := by
  apply candidates_eq_row edge 5 0 1621 tail255 root254Candidates winner.bound tail255_ordered
  · decide
  · decide
  · funext q
    rfl
def root254Tail53 : List Nat := []
def root254Tail52 : List Nat := 22291 :: root254Tail53
def root254Tail51 : List Nat := 21649 :: root254Tail52
def root254Tail50 : List Nat := 21067 :: root254Tail51
def root254Tail49 : List Nat := 20149 :: root254Tail50
def root254Tail48 : List Nat := 19777 :: root254Tail49
def root254Tail47 : List Nat := 19423 :: root254Tail48
def root254Tail46 : List Nat := 19051 :: root254Tail47
def root254Tail45 : List Nat := 17881 :: root254Tail46
def root254Tail44 : List Nat := 17713 :: root254Tail45
def root254Tail43 : List Nat := 17581 :: root254Tail44
def root254Tail42 : List Nat := 17467 :: root254Tail43
def root254Tail41 : List Nat := 17341 :: root254Tail42
def root254Tail40 : List Nat := 16651 :: root254Tail41
def root254Tail39 : List Nat := 16447 :: root254Tail40
def root254Tail38 : List Nat := 16411 :: root254Tail39
def root254Tail37 : List Nat := 16249 :: root254Tail38
def root254Tail36 : List Nat := 16033 :: root254Tail37
def root254Tail35 : List Nat := 15193 :: root254Tail36
def root254Tail34 : List Nat := 14731 :: root254Tail35
def root254Tail33 : List Nat := 12889 :: root254Tail34
def root254Tail32 : List Nat := 12037 :: root254Tail33
def root254Tail31 : List Nat := 11149 :: root254Tail32
def root254Tail30 : List Nat := 11083 :: root254Tail31
def root254Tail29 : List Nat := 10651 :: root254Tail30
def root254Tail28 : List Nat := 10321 :: root254Tail29
def root254Tail27 : List Nat := 9769 :: root254Tail28
def root254Tail26 : List Nat := 9403 :: root254Tail27
def root254Tail25 : List Nat := 9109 :: root254Tail26
def root254Tail24 : List Nat := 8971 :: root254Tail25
def root254Tail23 : List Nat := 7933 :: root254Tail24
def root254Tail22 : List Nat := 7687 :: root254Tail23
def root254Tail21 : List Nat := 7639 :: root254Tail22
def root254Tail20 : List Nat := 7603 :: root254Tail21
def root254Tail19 : List Nat := 7477 :: root254Tail20
def root254Tail18 : List Nat := 6247 :: root254Tail19
def root254Tail17 : List Nat := 6073 :: root254Tail18
def root254Tail16 : List Nat := 6037 :: root254Tail17
def root254Tail15 : List Nat := 5527 :: root254Tail16
def root254Tail14 : List Nat := 5503 :: root254Tail15
def root254Tail13 : List Nat := 5443 :: root254Tail14
def root254Tail12 : List Nat := 4441 :: root254Tail13
def root254Tail11 : List Nat := 4153 :: root254Tail12
def root254Tail10 : List Nat := 4021 :: root254Tail11
def root254Tail9 : List Nat := 3943 :: root254Tail10
def root254Tail8 : List Nat := 3823 :: root254Tail9
def root254Tail7 : List Nat := 3313 :: root254Tail8
def root254Tail6 : List Nat := 3187 :: root254Tail7
def root254Tail5 : List Nat := 2851 :: root254Tail6
def root254Tail4 : List Nat := 2803 :: root254Tail5
def root254Tail3 : List Nat := 2767 :: root254Tail4
def root254Tail2 : List Nat := 2437 :: root254Tail3
def root254Tail1 : List Nat := 2083 :: root254Tail2
def root254Tail0 : List Nat := 1801 :: root254Tail1
def root254Candidates0 : List Nat := [3823, 11149]
set_option maxRecDepth 8192 in
theorem root254Candidates0_checked : candidates edge 4 1621 1801 root254Tail1 winner.bound = root254Candidates0 := by
  unfold candidates
  have row : edge 1801 = (SortedNatTree.contains FastRows.row1801) := by
    change Data.row1801.contains = (SortedNatTree.contains FastRows.row1801)
    exact FastRows.row1801_checked.symm
  rw [row]
  decide
theorem root254Step0 : scan edge (search edge 3) 4 [1621] 1621 root254Tail0 winner =
    scan edge (search edge 3) 4 [1621] 1621 root254Tail1 winner := by
  rw [root254Tail0, scan, if_neg (by decide), root254Candidates0_checked]
  change scan edge (search edge 3) 4 [1621] 1621 root254Tail1
    (if 3 ≤ root254Candidates0.length then search edge 3 [1621, 1801] root254Candidates0 3422 winner else winner) = _
  rw [if_neg (by decide)]
def root254Candidates1 : List Nat := [5443, 9109, 17581]
set_option maxRecDepth 8192 in
theorem root254Candidates1_checked : candidates edge 4 1621 2083 root254Tail2 winner.bound = root254Candidates1 := by
  unfold candidates
  have row : edge 2083 = (SortedNatTree.contains FastRows.row2083) := by
    change Data.row2083.contains = (SortedNatTree.contains FastRows.row2083)
    exact FastRows.row2083_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root254Child1 : search edge 3 [1621, 2083] root254Candidates1 3704 winner = winner := by decide
theorem root254Step1 : scan edge (search edge 3) 4 [1621] 1621 root254Tail1 winner =
    scan edge (search edge 3) 4 [1621] 1621 root254Tail2 winner := by
  rw [root254Tail1, scan, if_neg (by decide), root254Candidates1_checked]
  change scan edge (search edge 3) 4 [1621] 1621 root254Tail2
    (if 3 ≤ root254Candidates1.length then search edge 3 [1621, 2083] root254Candidates1 3704 winner else winner) = _
  rw [if_pos (by decide), root254Child1]
def root254Candidates2 : List Nat := [6073]
set_option maxRecDepth 8192 in
theorem root254Candidates2_checked : candidates edge 4 1621 2437 root254Tail3 winner.bound = root254Candidates2 := by
  unfold candidates
  have row : edge 2437 = (SortedNatTree.contains FastRows.row2437) := by
    change Data.row2437.contains = (SortedNatTree.contains FastRows.row2437)
    exact FastRows.row2437_checked.symm
  rw [row]
  decide
theorem root254Step2 : scan edge (search edge 3) 4 [1621] 1621 root254Tail2 winner =
    scan edge (search edge 3) 4 [1621] 1621 root254Tail3 winner := by
  rw [root254Tail2, scan, if_neg (by decide), root254Candidates2_checked]
  change scan edge (search edge 3) 4 [1621] 1621 root254Tail3
    (if 3 ≤ root254Candidates2.length then search edge 3 [1621, 2437] root254Candidates2 4058 winner else winner) = _
  rw [if_neg (by decide)]
def root254Candidates3 : List Nat := [3187, 7477, 7933, 16249]
set_option maxRecDepth 8192 in
theorem root254Candidates3_checked : candidates edge 4 1621 2767 root254Tail4 winner.bound = root254Candidates3 := by
  unfold candidates
  have row : edge 2767 = (SortedNatTree.contains FastRows.row2767) := by
    change Data.row2767.contains = (SortedNatTree.contains FastRows.row2767)
    exact FastRows.row2767_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root254Child3 : search edge 3 [1621, 2767] root254Candidates3 4388 winner = winner := by decide
theorem root254Step3 : scan edge (search edge 3) 4 [1621] 1621 root254Tail3 winner =
    scan edge (search edge 3) 4 [1621] 1621 root254Tail4 winner := by
  rw [root254Tail3, scan, if_neg (by decide), root254Candidates3_checked]
  change scan edge (search edge 3) 4 [1621] 1621 root254Tail4
    (if 3 ≤ root254Candidates3.length then search edge 3 [1621, 2767] root254Candidates3 4388 winner else winner) = _
  rw [if_pos (by decide), root254Child3]
def root254Candidates4 : List Nat := [8971]
set_option maxRecDepth 8192 in
theorem root254Candidates4_checked : candidates edge 4 1621 2803 root254Tail5 winner.bound = root254Candidates4 := by
  unfold candidates
  have row : edge 2803 = (SortedNatTree.contains FastRows.row2803) := by
    change Data.row2803.contains = (SortedNatTree.contains FastRows.row2803)
    exact FastRows.row2803_checked.symm
  rw [row]
  decide
theorem root254Step4 : scan edge (search edge 3) 4 [1621] 1621 root254Tail4 winner =
    scan edge (search edge 3) 4 [1621] 1621 root254Tail5 winner := by
  rw [root254Tail4, scan, if_neg (by decide), root254Candidates4_checked]
  change scan edge (search edge 3) 4 [1621] 1621 root254Tail5
    (if 3 ≤ root254Candidates4.length then search edge 3 [1621, 2803] root254Candidates4 4424 winner else winner) = _
  rw [if_neg (by decide)]
def root254Candidates5 : List Nat := [3187, 3313, 19423]
set_option maxRecDepth 8192 in
theorem root254Candidates5_checked : candidates edge 4 1621 2851 root254Tail6 winner.bound = root254Candidates5 := by
  unfold candidates
  have row : edge 2851 = (SortedNatTree.contains FastRows.row2851) := by
    change Data.row2851.contains = (SortedNatTree.contains FastRows.row2851)
    exact FastRows.row2851_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root254Child5 : search edge 3 [1621, 2851] root254Candidates5 4472 winner = winner := by decide
theorem root254Step5 : scan edge (search edge 3) 4 [1621] 1621 root254Tail5 winner =
    scan edge (search edge 3) 4 [1621] 1621 root254Tail6 winner := by
  rw [root254Tail5, scan, if_neg (by decide), root254Candidates5_checked]
  change scan edge (search edge 3) 4 [1621] 1621 root254Tail6
    (if 3 ≤ root254Candidates5.length then search edge 3 [1621, 2851] root254Candidates5 4472 winner else winner) = _
  rw [if_pos (by decide), root254Child5]
def root254Candidates6 : List Nat := [7477, 9109, 17341]
set_option maxRecDepth 8192 in
theorem root254Candidates6_checked : candidates edge 4 1621 3187 root254Tail7 winner.bound = root254Candidates6 := by
  unfold candidates
  have row : edge 3187 = (SortedNatTree.contains FastRows.row3187) := by
    change Data.row3187.contains = (SortedNatTree.contains FastRows.row3187)
    exact FastRows.row3187_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root254Child6 : search edge 3 [1621, 3187] root254Candidates6 4808 winner = winner := by decide
theorem root254Step6 : scan edge (search edge 3) 4 [1621] 1621 root254Tail6 winner =
    scan edge (search edge 3) 4 [1621] 1621 root254Tail7 winner := by
  rw [root254Tail6, scan, if_neg (by decide), root254Candidates6_checked]
  change scan edge (search edge 3) 4 [1621] 1621 root254Tail7
    (if 3 ≤ root254Candidates6.length then search edge 3 [1621, 3187] root254Candidates6 4808 winner else winner) = _
  rw [if_pos (by decide), root254Child6]
def root254Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root254Candidates7_checked : candidates edge 4 1621 3313 root254Tail8 winner.bound = root254Candidates7 := by
  unfold candidates
  have row : edge 3313 = (SortedNatTree.contains FastRows.row3313) := by
    change Data.row3313.contains = (SortedNatTree.contains FastRows.row3313)
    exact FastRows.row3313_checked.symm
  rw [row]
  decide
theorem root254Step7 : scan edge (search edge 3) 4 [1621] 1621 root254Tail7 winner =
    scan edge (search edge 3) 4 [1621] 1621 root254Tail8 winner := by
  rw [root254Tail7, scan, if_neg (by decide), root254Candidates7_checked]
  change scan edge (search edge 3) 4 [1621] 1621 root254Tail8
    (if 3 ≤ root254Candidates7.length then search edge 3 [1621, 3313] root254Candidates7 4934 winner else winner) = _
  rw [if_neg (by decide)]
def root254Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root254Candidates8_checked : candidates edge 4 1621 3823 root254Tail9 winner.bound = root254Candidates8 := by
  unfold candidates
  have row : edge 3823 = (SortedNatTree.contains FastRows.row3823) := by
    change Data.row3823.contains = (SortedNatTree.contains FastRows.row3823)
    exact FastRows.row3823_checked.symm
  rw [row]
  decide
theorem root254Step8 : scan edge (search edge 3) 4 [1621] 1621 root254Tail8 winner =
    scan edge (search edge 3) 4 [1621] 1621 root254Tail9 winner := by
  rw [root254Tail8, scan, if_neg (by decide), root254Candidates8_checked]
  change scan edge (search edge 3) 4 [1621] 1621 root254Tail9
    (if 3 ≤ root254Candidates8.length then search edge 3 [1621, 3823] root254Candidates8 5444 winner else winner) = _
  rw [if_neg (by decide)]
def root254Candidates9 : List Nat := [9769]
set_option maxRecDepth 8192 in
theorem root254Candidates9_checked : candidates edge 4 1621 3943 root254Tail10 winner.bound = root254Candidates9 := by
  unfold candidates
  have row : edge 3943 = (SortedNatTree.contains FastRows.row3943) := by
    change Data.row3943.contains = (SortedNatTree.contains FastRows.row3943)
    exact FastRows.row3943_checked.symm
  rw [row]
  decide
theorem root254Step9 : scan edge (search edge 3) 4 [1621] 1621 root254Tail9 winner =
    scan edge (search edge 3) 4 [1621] 1621 root254Tail10 winner := by
  rw [root254Tail9, scan, if_neg (by decide), root254Candidates9_checked]
  change scan edge (search edge 3) 4 [1621] 1621 root254Tail10
    (if 3 ≤ root254Candidates9.length then search edge 3 [1621, 3943] root254Candidates9 5564 winner else winner) = _
  rw [if_neg (by decide)]
def root254Candidates10 : List Nat := [4441, 12037, 16651]
set_option maxRecDepth 8192 in
theorem root254Candidates10_checked : candidates edge 4 1621 4021 root254Tail11 winner.bound = root254Candidates10 := by
  unfold candidates
  have row : edge 4021 = (SortedNatTree.contains FastRows.row4021) := by
    change Data.row4021.contains = (SortedNatTree.contains FastRows.row4021)
    exact FastRows.row4021_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root254Child10 : search edge 3 [1621, 4021] root254Candidates10 5642 winner = winner := by decide
theorem root254Step10 : scan edge (search edge 3) 4 [1621] 1621 root254Tail10 winner =
    scan edge (search edge 3) 4 [1621] 1621 root254Tail11 winner := by
  rw [root254Tail10, scan, if_neg (by decide), root254Candidates10_checked]
  change scan edge (search edge 3) 4 [1621] 1621 root254Tail11
    (if 3 ≤ root254Candidates10.length then search edge 3 [1621, 4021] root254Candidates10 5642 winner else winner) = _
  rw [if_pos (by decide), root254Child10]
def root254Candidates11 : List Nat := [5503]
set_option maxRecDepth 8192 in
theorem root254Candidates11_checked : candidates edge 4 1621 4153 root254Tail12 winner.bound = root254Candidates11 := by
  unfold candidates
  have row : edge 4153 = (SortedNatTree.contains FastRows.row4153) := by
    change Data.row4153.contains = (SortedNatTree.contains FastRows.row4153)
    exact FastRows.row4153_checked.symm
  rw [row]
  decide
theorem root254Step11 : scan edge (search edge 3) 4 [1621] 1621 root254Tail11 winner =
    scan edge (search edge 3) 4 [1621] 1621 root254Tail12 winner := by
  rw [root254Tail11, scan, if_neg (by decide), root254Candidates11_checked]
  change scan edge (search edge 3) 4 [1621] 1621 root254Tail12
    (if 3 ≤ root254Candidates11.length then search edge 3 [1621, 4153] root254Candidates11 5774 winner else winner) = _
  rw [if_neg (by decide)]
def root254Candidates12 : List Nat := [6247]
set_option maxRecDepth 8192 in
theorem root254Candidates12_checked : candidates edge 4 1621 4441 root254Tail13 winner.bound = root254Candidates12 := by
  unfold candidates
  have row : edge 4441 = (SortedNatTree.contains FastRows.row4441) := by
    change Data.row4441.contains = (SortedNatTree.contains FastRows.row4441)
    exact FastRows.row4441_checked.symm
  rw [row]
  decide
theorem root254Step12 : scan edge (search edge 3) 4 [1621] 1621 root254Tail12 winner =
    scan edge (search edge 3) 4 [1621] 1621 root254Tail13 winner := by
  rw [root254Tail12, scan, if_neg (by decide), root254Candidates12_checked]
  change scan edge (search edge 3) 4 [1621] 1621 root254Tail13
    (if 3 ≤ root254Candidates12.length then search edge 3 [1621, 4441] root254Candidates12 6062 winner else winner) = _
  rw [if_neg (by decide)]
def root254Candidates13 : List Nat := [12037, 16651]
set_option maxRecDepth 8192 in
theorem root254Candidates13_checked : candidates edge 4 1621 5443 root254Tail14 winner.bound = root254Candidates13 := by
  unfold candidates
  have row : edge 5443 = (SortedNatTree.contains FastRows.row5443) := by
    change Data.row5443.contains = (SortedNatTree.contains FastRows.row5443)
    exact FastRows.row5443_checked.symm
  rw [row]
  decide
theorem root254Step13 : scan edge (search edge 3) 4 [1621] 1621 root254Tail13 winner =
    scan edge (search edge 3) 4 [1621] 1621 root254Tail14 winner := by
  rw [root254Tail13, scan, if_neg (by decide), root254Candidates13_checked]
  change scan edge (search edge 3) 4 [1621] 1621 root254Tail14
    (if 3 ≤ root254Candidates13.length then search edge 3 [1621, 5443] root254Candidates13 7064 winner else winner) = _
  rw [if_neg (by decide)]
def root254Candidates14 : List Nat := [7603, 16411]
set_option maxRecDepth 8192 in
theorem root254Candidates14_checked : candidates edge 4 1621 5503 root254Tail15 winner.bound = root254Candidates14 := by
  unfold candidates
  have row : edge 5503 = (SortedNatTree.contains FastRows.row5503) := by
    change Data.row5503.contains = (SortedNatTree.contains FastRows.row5503)
    exact FastRows.row5503_checked.symm
  rw [row]
  decide
theorem root254Step14 : scan edge (search edge 3) 4 [1621] 1621 root254Tail14 winner =
    scan edge (search edge 3) 4 [1621] 1621 root254Tail15 winner := by
  rw [root254Tail14, scan, if_neg (by decide), root254Candidates14_checked]
  change scan edge (search edge 3) 4 [1621] 1621 root254Tail15
    (if 3 ≤ root254Candidates14.length then search edge 3 [1621, 5503] root254Candidates14 7124 winner else winner) = _
  rw [if_neg (by decide)]
def root254Candidates15 : List Nat := [9403]
set_option maxRecDepth 8192 in
theorem root254Candidates15_checked : candidates edge 4 1621 5527 root254Tail16 winner.bound = root254Candidates15 := by
  unfold candidates
  have row : edge 5527 = (SortedNatTree.contains FastRows.row5527) := by
    change Data.row5527.contains = (SortedNatTree.contains FastRows.row5527)
    exact FastRows.row5527_checked.symm
  rw [row]
  decide
theorem root254Step15 : scan edge (search edge 3) 4 [1621] 1621 root254Tail15 winner =
    scan edge (search edge 3) 4 [1621] 1621 root254Tail16 winner := by
  rw [root254Tail15, scan, if_neg (by decide), root254Candidates15_checked]
  change scan edge (search edge 3) 4 [1621] 1621 root254Tail16
    (if 3 ≤ root254Candidates15.length then search edge 3 [1621, 5527] root254Candidates15 7148 winner else winner) = _
  rw [if_neg (by decide)]
def root254Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root254Candidates16_checked : candidates edge 4 1621 6037 root254Tail17 winner.bound = root254Candidates16 := by
  unfold candidates
  have row : edge 6037 = (SortedNatTree.contains FastRows.row6037) := by
    change Data.row6037.contains = (SortedNatTree.contains FastRows.row6037)
    exact FastRows.row6037_checked.symm
  rw [row]
  decide
theorem root254Step16 : scan edge (search edge 3) 4 [1621] 1621 root254Tail16 winner =
    scan edge (search edge 3) 4 [1621] 1621 root254Tail17 winner := by
  rw [root254Tail16, scan, if_neg (by decide), root254Candidates16_checked]
  change scan edge (search edge 3) 4 [1621] 1621 root254Tail17
    (if 3 ≤ root254Candidates16.length then search edge 3 [1621, 6037] root254Candidates16 7658 winner else winner) = _
  rw [if_neg (by decide)]
def root254Candidates17 : List Nat := [9769, 14731, 16249]
set_option maxRecDepth 8192 in
theorem root254Candidates17_checked : candidates edge 4 1621 6073 root254Tail18 winner.bound = root254Candidates17 := by
  unfold candidates
  have row : edge 6073 = (SortedNatTree.contains FastRows.row6073) := by
    change Data.row6073.contains = (SortedNatTree.contains FastRows.row6073)
    exact FastRows.row6073_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root254Child17 : search edge 3 [1621, 6073] root254Candidates17 7694 winner = winner := by decide
theorem root254Step17 : scan edge (search edge 3) 4 [1621] 1621 root254Tail17 winner =
    scan edge (search edge 3) 4 [1621] 1621 root254Tail18 winner := by
  rw [root254Tail17, scan, if_neg (by decide), root254Candidates17_checked]
  change scan edge (search edge 3) 4 [1621] 1621 root254Tail18
    (if 3 ≤ root254Candidates17.length then search edge 3 [1621, 6073] root254Candidates17 7694 winner else winner) = _
  rw [if_pos (by decide), root254Child17]
theorem root254Scan18 : scan edge (search edge 3) 4 [1621] 1621 root254Tail18 winner = winner := by rw [root254Tail18, scan, if_pos (by decide)]
theorem root254Scan17 : scan edge (search edge 3) 4 [1621] 1621 root254Tail17 winner = winner :=
  (root254Step17).trans root254Scan18
theorem root254Scan16 : scan edge (search edge 3) 4 [1621] 1621 root254Tail16 winner = winner :=
  (root254Step16).trans root254Scan17
theorem root254Scan15 : scan edge (search edge 3) 4 [1621] 1621 root254Tail15 winner = winner :=
  (root254Step15).trans root254Scan16
theorem root254Scan14 : scan edge (search edge 3) 4 [1621] 1621 root254Tail14 winner = winner :=
  (root254Step14).trans root254Scan15
theorem root254Scan13 : scan edge (search edge 3) 4 [1621] 1621 root254Tail13 winner = winner :=
  (root254Step13).trans root254Scan14
theorem root254Scan12 : scan edge (search edge 3) 4 [1621] 1621 root254Tail12 winner = winner :=
  (root254Step12).trans root254Scan13
theorem root254Scan11 : scan edge (search edge 3) 4 [1621] 1621 root254Tail11 winner = winner :=
  (root254Step11).trans root254Scan12
theorem root254Scan10 : scan edge (search edge 3) 4 [1621] 1621 root254Tail10 winner = winner :=
  (root254Step10).trans root254Scan11
theorem root254Scan9 : scan edge (search edge 3) 4 [1621] 1621 root254Tail9 winner = winner :=
  (root254Step9).trans root254Scan10
theorem root254Scan8 : scan edge (search edge 3) 4 [1621] 1621 root254Tail8 winner = winner :=
  (root254Step8).trans root254Scan9
theorem root254Scan7 : scan edge (search edge 3) 4 [1621] 1621 root254Tail7 winner = winner :=
  (root254Step7).trans root254Scan8
theorem root254Scan6 : scan edge (search edge 3) 4 [1621] 1621 root254Tail6 winner = winner :=
  (root254Step6).trans root254Scan7
theorem root254Scan5 : scan edge (search edge 3) 4 [1621] 1621 root254Tail5 winner = winner :=
  (root254Step5).trans root254Scan6
theorem root254Scan4 : scan edge (search edge 3) 4 [1621] 1621 root254Tail4 winner = winner :=
  (root254Step4).trans root254Scan5
theorem root254Scan3 : scan edge (search edge 3) 4 [1621] 1621 root254Tail3 winner = winner :=
  (root254Step3).trans root254Scan4
theorem root254Scan2 : scan edge (search edge 3) 4 [1621] 1621 root254Tail2 winner = winner :=
  (root254Step2).trans root254Scan3
theorem root254Scan1 : scan edge (search edge 3) 4 [1621] 1621 root254Tail1 winner = winner :=
  (root254Step1).trans root254Scan2
theorem root254Scan0 : scan edge (search edge 3) 4 [1621] 1621 root254Tail0 winner = winner :=
  (root254Step0).trans root254Scan1
set_option maxRecDepth 4096 in
theorem root254Branch : search edge 4 [1621] root254Candidates 1621 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1621] 1621 root254Tail0 winner = winner
  exact root254Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
