import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root212Candidates : List Nat := Data.row1319
set_option maxRecDepth 8192 in
theorem root212Candidates_checked : candidates edge 5 0 1319 tail213 winner.bound = root212Candidates := by
  apply candidates_eq_row edge 5 0 1319 tail213 root212Candidates winner.bound tail213_ordered
  · decide
  · decide
  · funext q
    rfl
def root212Tail47 : List Nat := []
def root212Tail46 : List Nat := 22613 :: root212Tail47
def root212Tail45 : List Nat := 21269 :: root212Tail46
def root212Tail44 : List Nat := 20807 :: root212Tail45
def root212Tail43 : List Nat := 17189 :: root212Tail44
def root212Tail42 : List Nat := 17033 :: root212Tail43
def root212Tail41 : List Nat := 16553 :: root212Tail42
def root212Tail40 : List Nat := 16361 :: root212Tail41
def root212Tail39 : List Nat := 16007 :: root212Tail40
def root212Tail38 : List Nat := 14633 :: root212Tail39
def root212Tail37 : List Nat := 14621 :: root212Tail38
def root212Tail36 : List Nat := 13877 :: root212Tail37
def root212Tail35 : List Nat := 13697 :: root212Tail36
def root212Tail34 : List Nat := 12899 :: root212Tail35
def root212Tail33 : List Nat := 12791 :: root212Tail34
def root212Tail32 : List Nat := 12743 :: root212Tail33
def root212Tail31 : List Nat := 12281 :: root212Tail32
def root212Tail30 : List Nat := 11933 :: root212Tail31
def root212Tail29 : List Nat := 11777 :: root212Tail30
def root212Tail28 : List Nat := 11027 :: root212Tail29
def root212Tail27 : List Nat := 10169 :: root212Tail28
def root212Tail26 : List Nat := 10091 :: root212Tail27
def root212Tail25 : List Nat := 9419 :: root212Tail26
def root212Tail24 : List Nat := 9341 :: root212Tail25
def root212Tail23 : List Nat := 9209 :: root212Tail24
def root212Tail22 : List Nat := 8699 :: root212Tail23
def root212Tail21 : List Nat := 8501 :: root212Tail22
def root212Tail20 : List Nat := 8363 :: root212Tail21
def root212Tail19 : List Nat := 8039 :: root212Tail20
def root212Tail18 : List Nat := 7649 :: root212Tail19
def root212Tail17 : List Nat := 6971 :: root212Tail18
def root212Tail16 : List Nat := 5981 :: root212Tail17
def root212Tail15 : List Nat := 5843 :: root212Tail16
def root212Tail14 : List Nat := 4283 :: root212Tail15
def root212Tail13 : List Nat := 4253 :: root212Tail14
def root212Tail12 : List Nat := 4133 :: root212Tail13
def root212Tail11 : List Nat := 4007 :: root212Tail12
def root212Tail10 : List Nat := 3863 :: root212Tail11
def root212Tail9 : List Nat := 3623 :: root212Tail10
def root212Tail8 : List Nat := 3533 :: root212Tail9
def root212Tail7 : List Nat := 3407 :: root212Tail8
def root212Tail6 : List Nat := 2789 :: root212Tail7
def root212Tail5 : List Nat := 2243 :: root212Tail6
def root212Tail4 : List Nat := 1979 :: root212Tail5
def root212Tail3 : List Nat := 1901 :: root212Tail4
def root212Tail2 : List Nat := 1637 :: root212Tail3
def root212Tail1 : List Nat := 1613 :: root212Tail2
def root212Tail0 : List Nat := 1361 :: root212Tail1
def root212Candidates0 : List Nat := [8699, 9341]
set_option maxRecDepth 8192 in
theorem root212Candidates0_checked : candidates edge 4 1319 1361 root212Tail1 winner.bound = root212Candidates0 := by
  unfold candidates
  have row : edge 1361 = (SortedNatTree.contains FastRows.row1361) := by
    change Data.row1361.contains = (SortedNatTree.contains FastRows.row1361)
    exact FastRows.row1361_checked.symm
  rw [row]
  decide
theorem root212Step0 : scan edge (search edge 3) 4 [1319] 1319 root212Tail0 winner =
    scan edge (search edge 3) 4 [1319] 1319 root212Tail1 winner := by
  rw [root212Tail0, scan, if_neg (by decide), root212Candidates0_checked]
  change scan edge (search edge 3) 4 [1319] 1319 root212Tail1
    (if 3 ≤ root212Candidates0.length then search edge 3 [1319, 1361] root212Candidates0 2680 winner else winner) = _
  rw [if_neg (by decide)]
def root212Candidates1 : List Nat := [1637, 1979, 4133, 6971, 11027, 12791]
set_option maxRecDepth 8192 in
theorem root212Candidates1_checked : candidates edge 4 1319 1613 root212Tail2 winner.bound = root212Candidates1 := by
  unfold candidates
  have row : edge 1613 = (SortedNatTree.contains FastRows.row1613) := by
    change Data.row1613.contains = (SortedNatTree.contains FastRows.row1613)
    exact FastRows.row1613_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root212Child1 : search edge 3 [1319, 1613] root212Candidates1 2932 winner = winner := by decide
theorem root212Step1 : scan edge (search edge 3) 4 [1319] 1319 root212Tail1 winner =
    scan edge (search edge 3) 4 [1319] 1319 root212Tail2 winner := by
  rw [root212Tail1, scan, if_neg (by decide), root212Candidates1_checked]
  change scan edge (search edge 3) 4 [1319] 1319 root212Tail2
    (if 3 ≤ root212Candidates1.length then search edge 3 [1319, 1613] root212Candidates1 2932 winner else winner) = _
  rw [if_pos (by decide), root212Child1]
def root212Candidates2 : List Nat := [8039, 8363, 11777, 14633]
set_option maxRecDepth 8192 in
theorem root212Candidates2_checked : candidates edge 4 1319 1637 root212Tail3 winner.bound = root212Candidates2 := by
  unfold candidates
  have row : edge 1637 = (SortedNatTree.contains FastRows.row1637) := by
    change Data.row1637.contains = (SortedNatTree.contains FastRows.row1637)
    exact FastRows.row1637_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root212Child2 : search edge 3 [1319, 1637] root212Candidates2 2956 winner = winner := by decide
theorem root212Step2 : scan edge (search edge 3) 4 [1319] 1319 root212Tail2 winner =
    scan edge (search edge 3) 4 [1319] 1319 root212Tail3 winner := by
  rw [root212Tail2, scan, if_neg (by decide), root212Candidates2_checked]
  change scan edge (search edge 3) 4 [1319] 1319 root212Tail3
    (if 3 ≤ root212Candidates2.length then search edge 3 [1319, 1637] root212Candidates2 2956 winner else winner) = _
  rw [if_pos (by decide), root212Child2]
def root212Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root212Candidates3_checked : candidates edge 4 1319 1901 root212Tail4 winner.bound = root212Candidates3 := by
  unfold candidates
  have row : edge 1901 = (SortedNatTree.contains FastRows.row1901) := by
    change Data.row1901.contains = (SortedNatTree.contains FastRows.row1901)
    exact FastRows.row1901_checked.symm
  rw [row]
  decide
theorem root212Step3 : scan edge (search edge 3) 4 [1319] 1319 root212Tail3 winner =
    scan edge (search edge 3) 4 [1319] 1319 root212Tail4 winner := by
  rw [root212Tail3, scan, if_neg (by decide), root212Candidates3_checked]
  change scan edge (search edge 3) 4 [1319] 1319 root212Tail4
    (if 3 ≤ root212Candidates3.length then search edge 3 [1319, 1901] root212Candidates3 3220 winner else winner) = _
  rw [if_neg (by decide)]
def root212Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root212Candidates4_checked : candidates edge 4 1319 1979 root212Tail5 winner.bound = root212Candidates4 := by
  unfold candidates
  have row : edge 1979 = (SortedNatTree.contains FastRows.row1979) := by
    change Data.row1979.contains = (SortedNatTree.contains FastRows.row1979)
    exact FastRows.row1979_checked.symm
  rw [row]
  decide
theorem root212Step4 : scan edge (search edge 3) 4 [1319] 1319 root212Tail4 winner =
    scan edge (search edge 3) 4 [1319] 1319 root212Tail5 winner := by
  rw [root212Tail4, scan, if_neg (by decide), root212Candidates4_checked]
  change scan edge (search edge 3) 4 [1319] 1319 root212Tail5
    (if 3 ≤ root212Candidates4.length then search edge 3 [1319, 1979] root212Candidates4 3298 winner else winner) = _
  rw [if_neg (by decide)]
def root212Candidates5 : List Nat := [2789, 4133, 12743, 17033, 17189]
set_option maxRecDepth 8192 in
theorem root212Candidates5_checked : candidates edge 4 1319 2243 root212Tail6 winner.bound = root212Candidates5 := by
  unfold candidates
  have row : edge 2243 = (SortedNatTree.contains FastRows.row2243) := by
    change Data.row2243.contains = (SortedNatTree.contains FastRows.row2243)
    exact FastRows.row2243_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root212Child5 : search edge 3 [1319, 2243] root212Candidates5 3562 winner = winner := by decide
theorem root212Step5 : scan edge (search edge 3) 4 [1319] 1319 root212Tail5 winner =
    scan edge (search edge 3) 4 [1319] 1319 root212Tail6 winner := by
  rw [root212Tail5, scan, if_neg (by decide), root212Candidates5_checked]
  change scan edge (search edge 3) 4 [1319] 1319 root212Tail6
    (if 3 ≤ root212Candidates5.length then search edge 3 [1319, 2243] root212Candidates5 3562 winner else winner) = _
  rw [if_pos (by decide), root212Child5]
def root212Candidates6 : List Nat := [4007, 4133]
set_option maxRecDepth 8192 in
theorem root212Candidates6_checked : candidates edge 4 1319 2789 root212Tail7 winner.bound = root212Candidates6 := by
  unfold candidates
  have row : edge 2789 = (SortedNatTree.contains FastRows.row2789) := by
    change Data.row2789.contains = (SortedNatTree.contains FastRows.row2789)
    exact FastRows.row2789_checked.symm
  rw [row]
  decide
theorem root212Step6 : scan edge (search edge 3) 4 [1319] 1319 root212Tail6 winner =
    scan edge (search edge 3) 4 [1319] 1319 root212Tail7 winner := by
  rw [root212Tail6, scan, if_neg (by decide), root212Candidates6_checked]
  change scan edge (search edge 3) 4 [1319] 1319 root212Tail7
    (if 3 ≤ root212Candidates6.length then search edge 3 [1319, 2789] root212Candidates6 4108 winner else winner) = _
  rw [if_neg (by decide)]
def root212Candidates7 : List Nat := [8699]
set_option maxRecDepth 8192 in
theorem root212Candidates7_checked : candidates edge 4 1319 3407 root212Tail8 winner.bound = root212Candidates7 := by
  unfold candidates
  have row : edge 3407 = (SortedNatTree.contains FastRows.row3407) := by
    change Data.row3407.contains = (SortedNatTree.contains FastRows.row3407)
    exact FastRows.row3407_checked.symm
  rw [row]
  decide
theorem root212Step7 : scan edge (search edge 3) 4 [1319] 1319 root212Tail7 winner =
    scan edge (search edge 3) 4 [1319] 1319 root212Tail8 winner := by
  rw [root212Tail7, scan, if_neg (by decide), root212Candidates7_checked]
  change scan edge (search edge 3) 4 [1319] 1319 root212Tail8
    (if 3 ≤ root212Candidates7.length then search edge 3 [1319, 3407] root212Candidates7 4726 winner else winner) = _
  rw [if_neg (by decide)]
def root212Candidates8 : List Nat := [4133, 9341, 10169]
set_option maxRecDepth 8192 in
theorem root212Candidates8_checked : candidates edge 4 1319 3533 root212Tail9 winner.bound = root212Candidates8 := by
  unfold candidates
  have row : edge 3533 = (SortedNatTree.contains FastRows.row3533) := by
    change Data.row3533.contains = (SortedNatTree.contains FastRows.row3533)
    exact FastRows.row3533_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root212Child8 : search edge 3 [1319, 3533] root212Candidates8 4852 winner = winner := by decide
theorem root212Step8 : scan edge (search edge 3) 4 [1319] 1319 root212Tail8 winner =
    scan edge (search edge 3) 4 [1319] 1319 root212Tail9 winner := by
  rw [root212Tail8, scan, if_neg (by decide), root212Candidates8_checked]
  change scan edge (search edge 3) 4 [1319] 1319 root212Tail9
    (if 3 ≤ root212Candidates8.length then search edge 3 [1319, 3533] root212Candidates8 4852 winner else winner) = _
  rw [if_pos (by decide), root212Child8]
def root212Candidates9 : List Nat := [5981]
set_option maxRecDepth 8192 in
theorem root212Candidates9_checked : candidates edge 4 1319 3623 root212Tail10 winner.bound = root212Candidates9 := by
  unfold candidates
  have row : edge 3623 = (SortedNatTree.contains FastRows.row3623) := by
    change Data.row3623.contains = (SortedNatTree.contains FastRows.row3623)
    exact FastRows.row3623_checked.symm
  rw [row]
  decide
theorem root212Step9 : scan edge (search edge 3) 4 [1319] 1319 root212Tail9 winner =
    scan edge (search edge 3) 4 [1319] 1319 root212Tail10 winner := by
  rw [root212Tail9, scan, if_neg (by decide), root212Candidates9_checked]
  change scan edge (search edge 3) 4 [1319] 1319 root212Tail10
    (if 3 ≤ root212Candidates9.length then search edge 3 [1319, 3623] root212Candidates9 4942 winner else winner) = _
  rw [if_neg (by decide)]
def root212Candidates10 : List Nat := [4007, 11777]
set_option maxRecDepth 8192 in
theorem root212Candidates10_checked : candidates edge 4 1319 3863 root212Tail11 winner.bound = root212Candidates10 := by
  unfold candidates
  have row : edge 3863 = (SortedNatTree.contains FastRows.row3863) := by
    change Data.row3863.contains = (SortedNatTree.contains FastRows.row3863)
    exact FastRows.row3863_checked.symm
  rw [row]
  decide
theorem root212Step10 : scan edge (search edge 3) 4 [1319] 1319 root212Tail10 winner =
    scan edge (search edge 3) 4 [1319] 1319 root212Tail11 winner := by
  rw [root212Tail10, scan, if_neg (by decide), root212Candidates10_checked]
  change scan edge (search edge 3) 4 [1319] 1319 root212Tail11
    (if 3 ≤ root212Candidates10.length then search edge 3 [1319, 3863] root212Candidates10 5182 winner else winner) = _
  rw [if_neg (by decide)]
def root212Candidates11 : List Nat := [16553]
set_option maxRecDepth 8192 in
theorem root212Candidates11_checked : candidates edge 4 1319 4007 root212Tail12 winner.bound = root212Candidates11 := by
  unfold candidates
  have row : edge 4007 = (SortedNatTree.contains FastRows.row4007) := by
    change Data.row4007.contains = (SortedNatTree.contains FastRows.row4007)
    exact FastRows.row4007_checked.symm
  rw [row]
  decide
theorem root212Step11 : scan edge (search edge 3) 4 [1319] 1319 root212Tail11 winner =
    scan edge (search edge 3) 4 [1319] 1319 root212Tail12 winner := by
  rw [root212Tail11, scan, if_neg (by decide), root212Candidates11_checked]
  change scan edge (search edge 3) 4 [1319] 1319 root212Tail12
    (if 3 ≤ root212Candidates11.length then search edge 3 [1319, 4007] root212Candidates11 5326 winner else winner) = _
  rw [if_neg (by decide)]
def root212Candidates12 : List Nat := [10169]
set_option maxRecDepth 8192 in
theorem root212Candidates12_checked : candidates edge 4 1319 4133 root212Tail13 winner.bound = root212Candidates12 := by
  unfold candidates
  have row : edge 4133 = (SortedNatTree.contains FastRows.row4133) := by
    change Data.row4133.contains = (SortedNatTree.contains FastRows.row4133)
    exact FastRows.row4133_checked.symm
  rw [row]
  decide
theorem root212Step12 : scan edge (search edge 3) 4 [1319] 1319 root212Tail12 winner =
    scan edge (search edge 3) 4 [1319] 1319 root212Tail13 winner := by
  rw [root212Tail12, scan, if_neg (by decide), root212Candidates12_checked]
  change scan edge (search edge 3) 4 [1319] 1319 root212Tail13
    (if 3 ≤ root212Candidates12.length then search edge 3 [1319, 4133] root212Candidates12 5452 winner else winner) = _
  rw [if_neg (by decide)]
def root212Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root212Candidates13_checked : candidates edge 4 1319 4253 root212Tail14 winner.bound = root212Candidates13 := by
  unfold candidates
  have row : edge 4253 = (SortedNatTree.contains FastRows.row4253) := by
    change Data.row4253.contains = (SortedNatTree.contains FastRows.row4253)
    exact FastRows.row4253_checked.symm
  rw [row]
  decide
theorem root212Step13 : scan edge (search edge 3) 4 [1319] 1319 root212Tail13 winner =
    scan edge (search edge 3) 4 [1319] 1319 root212Tail14 winner := by
  rw [root212Tail13, scan, if_neg (by decide), root212Candidates13_checked]
  change scan edge (search edge 3) 4 [1319] 1319 root212Tail14
    (if 3 ≤ root212Candidates13.length then search edge 3 [1319, 4253] root212Candidates13 5572 winner else winner) = _
  rw [if_neg (by decide)]
def root212Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root212Candidates14_checked : candidates edge 4 1319 4283 root212Tail15 winner.bound = root212Candidates14 := by
  unfold candidates
  have row : edge 4283 = (SortedNatTree.contains FastRows.row4283) := by
    change Data.row4283.contains = (SortedNatTree.contains FastRows.row4283)
    exact FastRows.row4283_checked.symm
  rw [row]
  decide
theorem root212Step14 : scan edge (search edge 3) 4 [1319] 1319 root212Tail14 winner =
    scan edge (search edge 3) 4 [1319] 1319 root212Tail15 winner := by
  rw [root212Tail14, scan, if_neg (by decide), root212Candidates14_checked]
  change scan edge (search edge 3) 4 [1319] 1319 root212Tail15
    (if 3 ≤ root212Candidates14.length then search edge 3 [1319, 4283] root212Candidates14 5602 winner else winner) = _
  rw [if_neg (by decide)]
def root212Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root212Candidates15_checked : candidates edge 4 1319 5843 root212Tail16 winner.bound = root212Candidates15 := by
  unfold candidates
  have row : edge 5843 = (SortedNatTree.contains FastRows.row5843) := by
    change Data.row5843.contains = (SortedNatTree.contains FastRows.row5843)
    exact FastRows.row5843_checked.symm
  rw [row]
  decide
theorem root212Step15 : scan edge (search edge 3) 4 [1319] 1319 root212Tail15 winner =
    scan edge (search edge 3) 4 [1319] 1319 root212Tail16 winner := by
  rw [root212Tail15, scan, if_neg (by decide), root212Candidates15_checked]
  change scan edge (search edge 3) 4 [1319] 1319 root212Tail16
    (if 3 ≤ root212Candidates15.length then search edge 3 [1319, 5843] root212Candidates15 7162 winner else winner) = _
  rw [if_neg (by decide)]
def root212Candidates16 : List Nat := [9341, 11933, 12899, 16361, 17033]
set_option maxRecDepth 8192 in
theorem root212Candidates16_checked : candidates edge 4 1319 5981 root212Tail17 winner.bound = root212Candidates16 := by
  unfold candidates
  have row : edge 5981 = (SortedNatTree.contains FastRows.row5981) := by
    change Data.row5981.contains = (SortedNatTree.contains FastRows.row5981)
    exact FastRows.row5981_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root212Child16 : search edge 3 [1319, 5981] root212Candidates16 7300 winner = winner := by decide
theorem root212Step16 : scan edge (search edge 3) 4 [1319] 1319 root212Tail16 winner =
    scan edge (search edge 3) 4 [1319] 1319 root212Tail17 winner := by
  rw [root212Tail16, scan, if_neg (by decide), root212Candidates16_checked]
  change scan edge (search edge 3) 4 [1319] 1319 root212Tail17
    (if 3 ≤ root212Candidates16.length then search edge 3 [1319, 5981] root212Candidates16 7300 winner else winner) = _
  rw [if_pos (by decide), root212Child16]
theorem root212Scan17 : scan edge (search edge 3) 4 [1319] 1319 root212Tail17 winner = winner := by rw [root212Tail17, scan, if_pos (by decide)]
theorem root212Scan16 : scan edge (search edge 3) 4 [1319] 1319 root212Tail16 winner = winner :=
  (root212Step16).trans root212Scan17
theorem root212Scan15 : scan edge (search edge 3) 4 [1319] 1319 root212Tail15 winner = winner :=
  (root212Step15).trans root212Scan16
theorem root212Scan14 : scan edge (search edge 3) 4 [1319] 1319 root212Tail14 winner = winner :=
  (root212Step14).trans root212Scan15
theorem root212Scan13 : scan edge (search edge 3) 4 [1319] 1319 root212Tail13 winner = winner :=
  (root212Step13).trans root212Scan14
theorem root212Scan12 : scan edge (search edge 3) 4 [1319] 1319 root212Tail12 winner = winner :=
  (root212Step12).trans root212Scan13
theorem root212Scan11 : scan edge (search edge 3) 4 [1319] 1319 root212Tail11 winner = winner :=
  (root212Step11).trans root212Scan12
theorem root212Scan10 : scan edge (search edge 3) 4 [1319] 1319 root212Tail10 winner = winner :=
  (root212Step10).trans root212Scan11
theorem root212Scan9 : scan edge (search edge 3) 4 [1319] 1319 root212Tail9 winner = winner :=
  (root212Step9).trans root212Scan10
theorem root212Scan8 : scan edge (search edge 3) 4 [1319] 1319 root212Tail8 winner = winner :=
  (root212Step8).trans root212Scan9
theorem root212Scan7 : scan edge (search edge 3) 4 [1319] 1319 root212Tail7 winner = winner :=
  (root212Step7).trans root212Scan8
theorem root212Scan6 : scan edge (search edge 3) 4 [1319] 1319 root212Tail6 winner = winner :=
  (root212Step6).trans root212Scan7
theorem root212Scan5 : scan edge (search edge 3) 4 [1319] 1319 root212Tail5 winner = winner :=
  (root212Step5).trans root212Scan6
theorem root212Scan4 : scan edge (search edge 3) 4 [1319] 1319 root212Tail4 winner = winner :=
  (root212Step4).trans root212Scan5
theorem root212Scan3 : scan edge (search edge 3) 4 [1319] 1319 root212Tail3 winner = winner :=
  (root212Step3).trans root212Scan4
theorem root212Scan2 : scan edge (search edge 3) 4 [1319] 1319 root212Tail2 winner = winner :=
  (root212Step2).trans root212Scan3
theorem root212Scan1 : scan edge (search edge 3) 4 [1319] 1319 root212Tail1 winner = winner :=
  (root212Step1).trans root212Scan2
theorem root212Scan0 : scan edge (search edge 3) 4 [1319] 1319 root212Tail0 winner = winner :=
  (root212Step0).trans root212Scan1
set_option maxRecDepth 4096 in
theorem root212Branch : search edge 4 [1319] root212Candidates 1319 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1319] 1319 root212Tail0 winner = winner
  exact root212Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
