import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root211Candidates : List Nat := Data.row1307
set_option maxRecDepth 8192 in
theorem root211Candidates_checked : candidates edge 5 0 1307 tail212 winner.bound = root211Candidates := by
  apply candidates_eq_row edge 5 0 1307 tail212 root211Candidates winner.bound tail212_ordered
  · decide
  · decide
  · funext q
    rfl
def root211Tail55 : List Nat := []
def root211Tail54 : List Nat := 24683 :: root211Tail55
def root211Tail53 : List Nat := 23687 :: root211Tail54
def root211Tail52 : List Nat := 23057 :: root211Tail53
def root211Tail51 : List Nat := 21839 :: root211Tail52
def root211Tail50 : List Nat := 20981 :: root211Tail51
def root211Tail49 : List Nat := 20663 :: root211Tail50
def root211Tail48 : List Nat := 19991 :: root211Tail49
def root211Tail47 : List Nat := 19727 :: root211Tail48
def root211Tail46 : List Nat := 19469 :: root211Tail47
def root211Tail45 : List Nat := 19427 :: root211Tail46
def root211Tail44 : List Nat := 18149 :: root211Tail45
def root211Tail43 : List Nat := 16673 :: root211Tail44
def root211Tail42 : List Nat := 16661 :: root211Tail43
def root211Tail41 : List Nat := 16529 :: root211Tail42
def root211Tail40 : List Nat := 16493 :: root211Tail41
def root211Tail39 : List Nat := 16319 :: root211Tail40
def root211Tail38 : List Nat := 15881 :: root211Tail39
def root211Tail37 : List Nat := 15443 :: root211Tail38
def root211Tail36 : List Nat := 15401 :: root211Tail37
def root211Tail35 : List Nat := 13931 :: root211Tail36
def root211Tail34 : List Nat := 13841 :: root211Tail35
def root211Tail33 : List Nat := 13553 :: root211Tail34
def root211Tail32 : List Nat := 13523 :: root211Tail33
def root211Tail31 : List Nat := 13397 :: root211Tail32
def root211Tail30 : List Nat := 13151 :: root211Tail31
def root211Tail29 : List Nat := 13007 :: root211Tail30
def root211Tail28 : List Nat := 12941 :: root211Tail29
def root211Tail27 : List Nat := 12899 :: root211Tail28
def root211Tail26 : List Nat := 12809 :: root211Tail27
def root211Tail25 : List Nat := 11681 :: root211Tail26
def root211Tail24 : List Nat := 10949 :: root211Tail25
def root211Tail23 : List Nat := 9497 :: root211Tail24
def root211Tail22 : List Nat := 8171 :: root211Tail23
def root211Tail21 : List Nat := 7901 :: root211Tail22
def root211Tail20 : List Nat := 7583 :: root211Tail21
def root211Tail19 : List Nat := 7541 :: root211Tail20
def root211Tail18 : List Nat := 7499 :: root211Tail19
def root211Tail17 : List Nat := 6863 :: root211Tail18
def root211Tail16 : List Nat := 6053 :: root211Tail17
def root211Tail15 : List Nat := 5939 :: root211Tail16
def root211Tail14 : List Nat := 5927 :: root211Tail15
def root211Tail13 : List Nat := 5861 :: root211Tail14
def root211Tail12 : List Nat := 4583 :: root211Tail13
def root211Tail11 : List Nat := 4133 :: root211Tail12
def root211Tail10 : List Nat := 4079 :: root211Tail11
def root211Tail9 : List Nat := 3989 :: root211Tail10
def root211Tail8 : List Nat := 3719 :: root211Tail9
def root211Tail7 : List Nat := 3701 :: root211Tail8
def root211Tail6 : List Nat := 3491 :: root211Tail7
def root211Tail5 : List Nat := 3023 :: root211Tail6
def root211Tail4 : List Nat := 2837 :: root211Tail5
def root211Tail3 : List Nat := 2663 :: root211Tail4
def root211Tail2 : List Nat := 2357 :: root211Tail3
def root211Tail1 : List Nat := 1979 :: root211Tail2
def root211Tail0 : List Nat := 1787 :: root211Tail1
def root211Candidates0 : List Nat := [5927, 7901, 13553]
set_option maxRecDepth 8192 in
theorem root211Candidates0_checked : candidates edge 4 1307 1787 root211Tail1 winner.bound = root211Candidates0 := by
  unfold candidates
  have row : edge 1787 = (SortedNatTree.contains FastRows.row1787) := by
    change Data.row1787.contains = (SortedNatTree.contains FastRows.row1787)
    exact FastRows.row1787_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root211Child0 : search edge 3 [1307, 1787] root211Candidates0 3094 winner = winner := by decide
theorem root211Step0 : scan edge (search edge 3) 4 [1307] 1307 root211Tail0 winner =
    scan edge (search edge 3) 4 [1307] 1307 root211Tail1 winner := by
  rw [root211Tail0, scan, if_neg (by decide), root211Candidates0_checked]
  change scan edge (search edge 3) 4 [1307] 1307 root211Tail1
    (if 3 ≤ root211Candidates0.length then search edge 3 [1307, 1787] root211Candidates0 3094 winner else winner) = _
  rw [if_pos (by decide), root211Child0]
def root211Candidates1 : List Nat := [12941, 13523, 16661]
set_option maxRecDepth 8192 in
theorem root211Candidates1_checked : candidates edge 4 1307 1979 root211Tail2 winner.bound = root211Candidates1 := by
  unfold candidates
  have row : edge 1979 = (SortedNatTree.contains FastRows.row1979) := by
    change Data.row1979.contains = (SortedNatTree.contains FastRows.row1979)
    exact FastRows.row1979_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root211Child1 : search edge 3 [1307, 1979] root211Candidates1 3286 winner = winner := by decide
theorem root211Step1 : scan edge (search edge 3) 4 [1307] 1307 root211Tail1 winner =
    scan edge (search edge 3) 4 [1307] 1307 root211Tail2 winner := by
  rw [root211Tail1, scan, if_neg (by decide), root211Candidates1_checked]
  change scan edge (search edge 3) 4 [1307] 1307 root211Tail2
    (if 3 ≤ root211Candidates1.length then search edge 3 [1307, 1979] root211Candidates1 3286 winner else winner) = _
  rw [if_pos (by decide), root211Child1]
def root211Candidates2 : List Nat := [7901, 16661, 20981]
set_option maxRecDepth 8192 in
theorem root211Candidates2_checked : candidates edge 4 1307 2357 root211Tail3 winner.bound = root211Candidates2 := by
  unfold candidates
  have row : edge 2357 = (SortedNatTree.contains FastRows.row2357) := by
    change Data.row2357.contains = (SortedNatTree.contains FastRows.row2357)
    exact FastRows.row2357_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root211Child2 : search edge 3 [1307, 2357] root211Candidates2 3664 winner = winner := by decide
theorem root211Step2 : scan edge (search edge 3) 4 [1307] 1307 root211Tail2 winner =
    scan edge (search edge 3) 4 [1307] 1307 root211Tail3 winner := by
  rw [root211Tail2, scan, if_neg (by decide), root211Candidates2_checked]
  change scan edge (search edge 3) 4 [1307] 1307 root211Tail3
    (if 3 ≤ root211Candidates2.length then search edge 3 [1307, 2357] root211Candidates2 3664 winner else winner) = _
  rw [if_pos (by decide), root211Child2]
def root211Candidates3 : List Nat := [7901]
set_option maxRecDepth 8192 in
theorem root211Candidates3_checked : candidates edge 4 1307 2663 root211Tail4 winner.bound = root211Candidates3 := by
  unfold candidates
  have row : edge 2663 = (SortedNatTree.contains FastRows.row2663) := by
    change Data.row2663.contains = (SortedNatTree.contains FastRows.row2663)
    exact FastRows.row2663_checked.symm
  rw [row]
  decide
theorem root211Step3 : scan edge (search edge 3) 4 [1307] 1307 root211Tail3 winner =
    scan edge (search edge 3) 4 [1307] 1307 root211Tail4 winner := by
  rw [root211Tail3, scan, if_neg (by decide), root211Candidates3_checked]
  change scan edge (search edge 3) 4 [1307] 1307 root211Tail4
    (if 3 ≤ root211Candidates3.length then search edge 3 [1307, 2663] root211Candidates3 3970 winner else winner) = _
  rw [if_neg (by decide)]
def root211Candidates4 : List Nat := [13841]
set_option maxRecDepth 8192 in
theorem root211Candidates4_checked : candidates edge 4 1307 2837 root211Tail5 winner.bound = root211Candidates4 := by
  unfold candidates
  have row : edge 2837 = (SortedNatTree.contains FastRows.row2837) := by
    change Data.row2837.contains = (SortedNatTree.contains FastRows.row2837)
    exact FastRows.row2837_checked.symm
  rw [row]
  decide
theorem root211Step4 : scan edge (search edge 3) 4 [1307] 1307 root211Tail4 winner =
    scan edge (search edge 3) 4 [1307] 1307 root211Tail5 winner := by
  rw [root211Tail4, scan, if_neg (by decide), root211Candidates4_checked]
  change scan edge (search edge 3) 4 [1307] 1307 root211Tail5
    (if 3 ≤ root211Candidates4.length then search edge 3 [1307, 2837] root211Candidates4 4144 winner else winner) = _
  rw [if_neg (by decide)]
def root211Candidates5 : List Nat := [3989, 9497, 16673]
set_option maxRecDepth 8192 in
theorem root211Candidates5_checked : candidates edge 4 1307 3023 root211Tail6 winner.bound = root211Candidates5 := by
  unfold candidates
  have row : edge 3023 = (SortedNatTree.contains FastRows.row3023) := by
    change Data.row3023.contains = (SortedNatTree.contains FastRows.row3023)
    exact FastRows.row3023_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root211Child5 : search edge 3 [1307, 3023] root211Candidates5 4330 winner = winner := by decide
theorem root211Step5 : scan edge (search edge 3) 4 [1307] 1307 root211Tail5 winner =
    scan edge (search edge 3) 4 [1307] 1307 root211Tail6 winner := by
  rw [root211Tail5, scan, if_neg (by decide), root211Candidates5_checked]
  change scan edge (search edge 3) 4 [1307] 1307 root211Tail6
    (if 3 ≤ root211Candidates5.length then search edge 3 [1307, 3023] root211Candidates5 4330 winner else winner) = _
  rw [if_pos (by decide), root211Child5]
def root211Candidates6 : List Nat := [16319]
set_option maxRecDepth 8192 in
theorem root211Candidates6_checked : candidates edge 4 1307 3491 root211Tail7 winner.bound = root211Candidates6 := by
  unfold candidates
  have row : edge 3491 = (SortedNatTree.contains FastRows.row3491) := by
    change Data.row3491.contains = (SortedNatTree.contains FastRows.row3491)
    exact FastRows.row3491_checked.symm
  rw [row]
  decide
theorem root211Step6 : scan edge (search edge 3) 4 [1307] 1307 root211Tail6 winner =
    scan edge (search edge 3) 4 [1307] 1307 root211Tail7 winner := by
  rw [root211Tail6, scan, if_neg (by decide), root211Candidates6_checked]
  change scan edge (search edge 3) 4 [1307] 1307 root211Tail7
    (if 3 ≤ root211Candidates6.length then search edge 3 [1307, 3491] root211Candidates6 4798 winner else winner) = _
  rw [if_neg (by decide)]
def root211Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root211Candidates7_checked : candidates edge 4 1307 3701 root211Tail8 winner.bound = root211Candidates7 := by
  unfold candidates
  have row : edge 3701 = (SortedNatTree.contains FastRows.row3701) := by
    change Data.row3701.contains = (SortedNatTree.contains FastRows.row3701)
    exact FastRows.row3701_checked.symm
  rw [row]
  decide
theorem root211Step7 : scan edge (search edge 3) 4 [1307] 1307 root211Tail7 winner =
    scan edge (search edge 3) 4 [1307] 1307 root211Tail8 winner := by
  rw [root211Tail7, scan, if_neg (by decide), root211Candidates7_checked]
  change scan edge (search edge 3) 4 [1307] 1307 root211Tail8
    (if 3 ≤ root211Candidates7.length then search edge 3 [1307, 3701] root211Candidates7 5008 winner else winner) = _
  rw [if_neg (by decide)]
def root211Candidates8 : List Nat := [12899]
set_option maxRecDepth 8192 in
theorem root211Candidates8_checked : candidates edge 4 1307 3719 root211Tail9 winner.bound = root211Candidates8 := by
  unfold candidates
  have row : edge 3719 = (SortedNatTree.contains FastRows.row3719) := by
    change Data.row3719.contains = (SortedNatTree.contains FastRows.row3719)
    exact FastRows.row3719_checked.symm
  rw [row]
  decide
theorem root211Step8 : scan edge (search edge 3) 4 [1307] 1307 root211Tail8 winner =
    scan edge (search edge 3) 4 [1307] 1307 root211Tail9 winner := by
  rw [root211Tail8, scan, if_neg (by decide), root211Candidates8_checked]
  change scan edge (search edge 3) 4 [1307] 1307 root211Tail9
    (if 3 ≤ root211Candidates8.length then search edge 3 [1307, 3719] root211Candidates8 5026 winner else winner) = _
  rw [if_neg (by decide)]
def root211Candidates9 : List Nat := [5927]
set_option maxRecDepth 8192 in
theorem root211Candidates9_checked : candidates edge 4 1307 3989 root211Tail10 winner.bound = root211Candidates9 := by
  unfold candidates
  have row : edge 3989 = (SortedNatTree.contains FastRows.row3989) := by
    change Data.row3989.contains = (SortedNatTree.contains FastRows.row3989)
    exact FastRows.row3989_checked.symm
  rw [row]
  decide
theorem root211Step9 : scan edge (search edge 3) 4 [1307] 1307 root211Tail9 winner =
    scan edge (search edge 3) 4 [1307] 1307 root211Tail10 winner := by
  rw [root211Tail9, scan, if_neg (by decide), root211Candidates9_checked]
  change scan edge (search edge 3) 4 [1307] 1307 root211Tail10
    (if 3 ≤ root211Candidates9.length then search edge 3 [1307, 3989] root211Candidates9 5296 winner else winner) = _
  rw [if_neg (by decide)]
def root211Candidates10 : List Nat := [5927, 6053, 6863, 13151, 13553]
set_option maxRecDepth 8192 in
theorem root211Candidates10_checked : candidates edge 4 1307 4079 root211Tail11 winner.bound = root211Candidates10 := by
  unfold candidates
  have row : edge 4079 = (SortedNatTree.contains FastRows.row4079) := by
    change Data.row4079.contains = (SortedNatTree.contains FastRows.row4079)
    exact FastRows.row4079_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root211Child10 : search edge 3 [1307, 4079] root211Candidates10 5386 winner = winner := by decide
theorem root211Step10 : scan edge (search edge 3) 4 [1307] 1307 root211Tail10 winner =
    scan edge (search edge 3) 4 [1307] 1307 root211Tail11 winner := by
  rw [root211Tail10, scan, if_neg (by decide), root211Candidates10_checked]
  change scan edge (search edge 3) 4 [1307] 1307 root211Tail11
    (if 3 ≤ root211Candidates10.length then search edge 3 [1307, 4079] root211Candidates10 5386 winner else winner) = _
  rw [if_pos (by decide), root211Child10]
def root211Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root211Candidates11_checked : candidates edge 4 1307 4133 root211Tail12 winner.bound = root211Candidates11 := by
  unfold candidates
  have row : edge 4133 = (SortedNatTree.contains FastRows.row4133) := by
    change Data.row4133.contains = (SortedNatTree.contains FastRows.row4133)
    exact FastRows.row4133_checked.symm
  rw [row]
  decide
theorem root211Step11 : scan edge (search edge 3) 4 [1307] 1307 root211Tail11 winner =
    scan edge (search edge 3) 4 [1307] 1307 root211Tail12 winner := by
  rw [root211Tail11, scan, if_neg (by decide), root211Candidates11_checked]
  change scan edge (search edge 3) 4 [1307] 1307 root211Tail12
    (if 3 ≤ root211Candidates11.length then search edge 3 [1307, 4133] root211Candidates11 5440 winner else winner) = _
  rw [if_neg (by decide)]
def root211Candidates12 : List Nat := [13007]
set_option maxRecDepth 8192 in
theorem root211Candidates12_checked : candidates edge 4 1307 4583 root211Tail13 winner.bound = root211Candidates12 := by
  unfold candidates
  have row : edge 4583 = (SortedNatTree.contains FastRows.row4583) := by
    change Data.row4583.contains = (SortedNatTree.contains FastRows.row4583)
    exact FastRows.row4583_checked.symm
  rw [row]
  decide
theorem root211Step12 : scan edge (search edge 3) 4 [1307] 1307 root211Tail12 winner =
    scan edge (search edge 3) 4 [1307] 1307 root211Tail13 winner := by
  rw [root211Tail12, scan, if_neg (by decide), root211Candidates12_checked]
  change scan edge (search edge 3) 4 [1307] 1307 root211Tail13
    (if 3 ≤ root211Candidates12.length then search edge 3 [1307, 4583] root211Candidates12 5890 winner else winner) = _
  rw [if_neg (by decide)]
def root211Candidates13 : List Nat := [8171, 11681, 12941]
set_option maxRecDepth 8192 in
theorem root211Candidates13_checked : candidates edge 4 1307 5861 root211Tail14 winner.bound = root211Candidates13 := by
  unfold candidates
  have row : edge 5861 = (SortedNatTree.contains FastRows.row5861) := by
    change Data.row5861.contains = (SortedNatTree.contains FastRows.row5861)
    exact FastRows.row5861_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root211Child13 : search edge 3 [1307, 5861] root211Candidates13 7168 winner = winner := by decide
theorem root211Step13 : scan edge (search edge 3) 4 [1307] 1307 root211Tail13 winner =
    scan edge (search edge 3) 4 [1307] 1307 root211Tail14 winner := by
  rw [root211Tail13, scan, if_neg (by decide), root211Candidates13_checked]
  change scan edge (search edge 3) 4 [1307] 1307 root211Tail14
    (if 3 ≤ root211Candidates13.length then search edge 3 [1307, 5861] root211Candidates13 7168 winner else winner) = _
  rw [if_pos (by decide), root211Child13]
def root211Candidates14 : List Nat := [5939, 7499, 7583, 13523]
set_option maxRecDepth 8192 in
theorem root211Candidates14_checked : candidates edge 4 1307 5927 root211Tail15 winner.bound = root211Candidates14 := by
  unfold candidates
  have row : edge 5927 = (SortedNatTree.contains FastRows.row5927) := by
    change Data.row5927.contains = (SortedNatTree.contains FastRows.row5927)
    exact FastRows.row5927_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root211Child14 : search edge 3 [1307, 5927] root211Candidates14 7234 winner = winner := by decide
theorem root211Step14 : scan edge (search edge 3) 4 [1307] 1307 root211Tail14 winner =
    scan edge (search edge 3) 4 [1307] 1307 root211Tail15 winner := by
  rw [root211Tail14, scan, if_neg (by decide), root211Candidates14_checked]
  change scan edge (search edge 3) 4 [1307] 1307 root211Tail15
    (if 3 ≤ root211Candidates14.length then search edge 3 [1307, 5927] root211Candidates14 7234 winner else winner) = _
  rw [if_pos (by decide), root211Child14]
def root211Candidates15 : List Nat := [6053, 9497, 12899]
set_option maxRecDepth 8192 in
theorem root211Candidates15_checked : candidates edge 4 1307 5939 root211Tail16 winner.bound = root211Candidates15 := by
  unfold candidates
  have row : edge 5939 = (SortedNatTree.contains FastRows.row5939) := by
    change Data.row5939.contains = (SortedNatTree.contains FastRows.row5939)
    exact FastRows.row5939_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root211Child15 : search edge 3 [1307, 5939] root211Candidates15 7246 winner = winner := by decide
theorem root211Step15 : scan edge (search edge 3) 4 [1307] 1307 root211Tail15 winner =
    scan edge (search edge 3) 4 [1307] 1307 root211Tail16 winner := by
  rw [root211Tail15, scan, if_neg (by decide), root211Candidates15_checked]
  change scan edge (search edge 3) 4 [1307] 1307 root211Tail16
    (if 3 ≤ root211Candidates15.length then search edge 3 [1307, 5939] root211Candidates15 7246 winner else winner) = _
  rw [if_pos (by decide), root211Child15]
def root211Candidates16 : List Nat := [8171]
set_option maxRecDepth 8192 in
theorem root211Candidates16_checked : candidates edge 4 1307 6053 root211Tail17 winner.bound = root211Candidates16 := by
  unfold candidates
  have row : edge 6053 = (SortedNatTree.contains FastRows.row6053) := by
    change Data.row6053.contains = (SortedNatTree.contains FastRows.row6053)
    exact FastRows.row6053_checked.symm
  rw [row]
  decide
theorem root211Step16 : scan edge (search edge 3) 4 [1307] 1307 root211Tail16 winner =
    scan edge (search edge 3) 4 [1307] 1307 root211Tail17 winner := by
  rw [root211Tail16, scan, if_neg (by decide), root211Candidates16_checked]
  change scan edge (search edge 3) 4 [1307] 1307 root211Tail17
    (if 3 ≤ root211Candidates16.length then search edge 3 [1307, 6053] root211Candidates16 7360 winner else winner) = _
  rw [if_neg (by decide)]
theorem root211Scan17 : scan edge (search edge 3) 4 [1307] 1307 root211Tail17 winner = winner := by rw [root211Tail17, scan, if_pos (by decide)]
theorem root211Scan16 : scan edge (search edge 3) 4 [1307] 1307 root211Tail16 winner = winner :=
  (root211Step16).trans root211Scan17
theorem root211Scan15 : scan edge (search edge 3) 4 [1307] 1307 root211Tail15 winner = winner :=
  (root211Step15).trans root211Scan16
theorem root211Scan14 : scan edge (search edge 3) 4 [1307] 1307 root211Tail14 winner = winner :=
  (root211Step14).trans root211Scan15
theorem root211Scan13 : scan edge (search edge 3) 4 [1307] 1307 root211Tail13 winner = winner :=
  (root211Step13).trans root211Scan14
theorem root211Scan12 : scan edge (search edge 3) 4 [1307] 1307 root211Tail12 winner = winner :=
  (root211Step12).trans root211Scan13
theorem root211Scan11 : scan edge (search edge 3) 4 [1307] 1307 root211Tail11 winner = winner :=
  (root211Step11).trans root211Scan12
theorem root211Scan10 : scan edge (search edge 3) 4 [1307] 1307 root211Tail10 winner = winner :=
  (root211Step10).trans root211Scan11
theorem root211Scan9 : scan edge (search edge 3) 4 [1307] 1307 root211Tail9 winner = winner :=
  (root211Step9).trans root211Scan10
theorem root211Scan8 : scan edge (search edge 3) 4 [1307] 1307 root211Tail8 winner = winner :=
  (root211Step8).trans root211Scan9
theorem root211Scan7 : scan edge (search edge 3) 4 [1307] 1307 root211Tail7 winner = winner :=
  (root211Step7).trans root211Scan8
theorem root211Scan6 : scan edge (search edge 3) 4 [1307] 1307 root211Tail6 winner = winner :=
  (root211Step6).trans root211Scan7
theorem root211Scan5 : scan edge (search edge 3) 4 [1307] 1307 root211Tail5 winner = winner :=
  (root211Step5).trans root211Scan6
theorem root211Scan4 : scan edge (search edge 3) 4 [1307] 1307 root211Tail4 winner = winner :=
  (root211Step4).trans root211Scan5
theorem root211Scan3 : scan edge (search edge 3) 4 [1307] 1307 root211Tail3 winner = winner :=
  (root211Step3).trans root211Scan4
theorem root211Scan2 : scan edge (search edge 3) 4 [1307] 1307 root211Tail2 winner = winner :=
  (root211Step2).trans root211Scan3
theorem root211Scan1 : scan edge (search edge 3) 4 [1307] 1307 root211Tail1 winner = winner :=
  (root211Step1).trans root211Scan2
theorem root211Scan0 : scan edge (search edge 3) 4 [1307] 1307 root211Tail0 winner = winner :=
  (root211Step0).trans root211Scan1
set_option maxRecDepth 4096 in
theorem root211Branch : search edge 4 [1307] root211Candidates 1307 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1307] 1307 root211Tail0 winner = winner
  exact root211Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
