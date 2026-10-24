import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root281Candidates : List Nat := Data.row1861
set_option maxRecDepth 8192 in
theorem root281Candidates_checked : candidates edge 5 0 1861 tail282 winner.bound = root281Candidates := by
  apply candidates_eq_row edge 5 0 1861 tail282 root281Candidates winner.bound tail282_ordered
  · decide
  · decide
  · funext q
    rfl
def root281Tail41 : List Nat := []
def root281Tail40 : List Nat := 22567 :: root281Tail41
def root281Tail39 : List Nat := 22291 :: root281Tail40
def root281Tail38 : List Nat := 22159 :: root281Tail39
def root281Tail37 : List Nat := 19993 :: root281Tail38
def root281Tail36 : List Nat := 18043 :: root281Tail37
def root281Tail35 : List Nat := 17599 :: root281Tail36
def root281Tail34 : List Nat := 17137 :: root281Tail35
def root281Tail33 : List Nat := 16747 :: root281Tail34
def root281Tail32 : List Nat := 16477 :: root281Tail33
def root281Tail31 : List Nat := 15271 :: root281Tail32
def root281Tail30 : List Nat := 15187 :: root281Tail31
def root281Tail29 : List Nat := 14983 :: root281Tail30
def root281Tail28 : List Nat := 14869 :: root281Tail29
def root281Tail27 : List Nat := 13591 :: root281Tail28
def root281Tail26 : List Nat := 12211 :: root281Tail27
def root281Tail25 : List Nat := 11551 :: root281Tail26
def root281Tail24 : List Nat := 11437 :: root281Tail25
def root281Tail23 : List Nat := 10627 :: root281Tail24
def root281Tail22 : List Nat := 10501 :: root281Tail23
def root281Tail21 : List Nat := 9973 :: root281Tail22
def root281Tail20 : List Nat := 7867 :: root281Tail21
def root281Tail19 : List Nat := 7213 :: root281Tail20
def root281Tail18 : List Nat := 6907 :: root281Tail19
def root281Tail17 : List Nat := 6133 :: root281Tail18
def root281Tail16 : List Nat := 6121 :: root281Tail17
def root281Tail15 : List Nat := 6007 :: root281Tail16
def root281Tail14 : List Nat := 5881 :: root281Tail15
def root281Tail13 : List Nat := 5521 :: root281Tail14
def root281Tail12 : List Nat := 5479 :: root281Tail13
def root281Tail11 : List Nat := 5059 :: root281Tail12
def root281Tail10 : List Nat := 5011 :: root281Tail11
def root281Tail9 : List Nat := 4789 :: root281Tail10
def root281Tail8 : List Nat := 3769 :: root281Tail9
def root281Tail7 : List Nat := 3709 :: root281Tail8
def root281Tail6 : List Nat := 3433 :: root281Tail7
def root281Tail5 : List Nat := 3163 :: root281Tail6
def root281Tail4 : List Nat := 2791 :: root281Tail5
def root281Tail3 : List Nat := 2383 :: root281Tail4
def root281Tail2 : List Nat := 2371 :: root281Tail3
def root281Tail1 : List Nat := 1951 :: root281Tail2
def root281Tail0 : List Nat := 1933 :: root281Tail1
def root281Candidates0 : List Nat := [5011, 10627, 19993]
set_option maxRecDepth 8192 in
theorem root281Candidates0_checked : candidates edge 4 1861 1933 root281Tail1 winner.bound = root281Candidates0 := by
  unfold candidates
  have row : edge 1933 = (SortedNatTree.contains FastRows.row1933) := by
    change Data.row1933.contains = (SortedNatTree.contains FastRows.row1933)
    exact FastRows.row1933_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root281Child0 : search edge 3 [1861, 1933] root281Candidates0 3794 winner = winner := by decide
theorem root281Step0 : scan edge (search edge 3) 4 [1861] 1861 root281Tail0 winner =
    scan edge (search edge 3) 4 [1861] 1861 root281Tail1 winner := by
  rw [root281Tail0, scan, if_neg (by decide), root281Candidates0_checked]
  change scan edge (search edge 3) 4 [1861] 1861 root281Tail1
    (if 3 ≤ root281Candidates0.length then search edge 3 [1861, 1933] root281Candidates0 3794 winner else winner) = _
  rw [if_pos (by decide), root281Child0]
def root281Candidates1 : List Nat := [17599]
set_option maxRecDepth 8192 in
theorem root281Candidates1_checked : candidates edge 4 1861 1951 root281Tail2 winner.bound = root281Candidates1 := by
  unfold candidates
  have row : edge 1951 = (SortedNatTree.contains FastRows.row1951) := by
    change Data.row1951.contains = (SortedNatTree.contains FastRows.row1951)
    exact FastRows.row1951_checked.symm
  rw [row]
  decide
theorem root281Step1 : scan edge (search edge 3) 4 [1861] 1861 root281Tail1 winner =
    scan edge (search edge 3) 4 [1861] 1861 root281Tail2 winner := by
  rw [root281Tail1, scan, if_neg (by decide), root281Candidates1_checked]
  change scan edge (search edge 3) 4 [1861] 1861 root281Tail2
    (if 3 ≤ root281Candidates1.length then search edge 3 [1861, 1951] root281Candidates1 3812 winner else winner) = _
  rw [if_neg (by decide)]
def root281Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root281Candidates2_checked : candidates edge 4 1861 2371 root281Tail3 winner.bound = root281Candidates2 := by
  unfold candidates
  have row : edge 2371 = (SortedNatTree.contains FastRows.row2371) := by
    change Data.row2371.contains = (SortedNatTree.contains FastRows.row2371)
    exact FastRows.row2371_checked.symm
  rw [row]
  decide
theorem root281Step2 : scan edge (search edge 3) 4 [1861] 1861 root281Tail2 winner =
    scan edge (search edge 3) 4 [1861] 1861 root281Tail3 winner := by
  rw [root281Tail2, scan, if_neg (by decide), root281Candidates2_checked]
  change scan edge (search edge 3) 4 [1861] 1861 root281Tail3
    (if 3 ≤ root281Candidates2.length then search edge 3 [1861, 2371] root281Candidates2 4232 winner else winner) = _
  rw [if_neg (by decide)]
def root281Candidates3 : List Nat := [9973, 13591, 18043]
set_option maxRecDepth 8192 in
theorem root281Candidates3_checked : candidates edge 4 1861 2383 root281Tail4 winner.bound = root281Candidates3 := by
  unfold candidates
  have row : edge 2383 = (SortedNatTree.contains FastRows.row2383) := by
    change Data.row2383.contains = (SortedNatTree.contains FastRows.row2383)
    exact FastRows.row2383_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root281Child3 : search edge 3 [1861, 2383] root281Candidates3 4244 winner = winner := by decide
theorem root281Step3 : scan edge (search edge 3) 4 [1861] 1861 root281Tail3 winner =
    scan edge (search edge 3) 4 [1861] 1861 root281Tail4 winner := by
  rw [root281Tail3, scan, if_neg (by decide), root281Candidates3_checked]
  change scan edge (search edge 3) 4 [1861] 1861 root281Tail4
    (if 3 ≤ root281Candidates3.length then search edge 3 [1861, 2383] root281Candidates3 4244 winner else winner) = _
  rw [if_pos (by decide), root281Child3]
def root281Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root281Candidates4_checked : candidates edge 4 1861 2791 root281Tail5 winner.bound = root281Candidates4 := by
  unfold candidates
  have row : edge 2791 = (SortedNatTree.contains FastRows.row2791) := by
    change Data.row2791.contains = (SortedNatTree.contains FastRows.row2791)
    exact FastRows.row2791_checked.symm
  rw [row]
  decide
theorem root281Step4 : scan edge (search edge 3) 4 [1861] 1861 root281Tail4 winner =
    scan edge (search edge 3) 4 [1861] 1861 root281Tail5 winner := by
  rw [root281Tail4, scan, if_neg (by decide), root281Candidates4_checked]
  change scan edge (search edge 3) 4 [1861] 1861 root281Tail5
    (if 3 ≤ root281Candidates4.length then search edge 3 [1861, 2791] root281Candidates4 4652 winner else winner) = _
  rw [if_neg (by decide)]
def root281Candidates5 : List Nat := [3769]
set_option maxRecDepth 8192 in
theorem root281Candidates5_checked : candidates edge 4 1861 3163 root281Tail6 winner.bound = root281Candidates5 := by
  unfold candidates
  have row : edge 3163 = (SortedNatTree.contains FastRows.row3163) := by
    change Data.row3163.contains = (SortedNatTree.contains FastRows.row3163)
    exact FastRows.row3163_checked.symm
  rw [row]
  decide
theorem root281Step5 : scan edge (search edge 3) 4 [1861] 1861 root281Tail5 winner =
    scan edge (search edge 3) 4 [1861] 1861 root281Tail6 winner := by
  rw [root281Tail5, scan, if_neg (by decide), root281Candidates5_checked]
  change scan edge (search edge 3) 4 [1861] 1861 root281Tail6
    (if 3 ≤ root281Candidates5.length then search edge 3 [1861, 3163] root281Candidates5 5024 winner else winner) = _
  rw [if_neg (by decide)]
def root281Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root281Candidates6_checked : candidates edge 4 1861 3433 root281Tail7 winner.bound = root281Candidates6 := by
  unfold candidates
  have row : edge 3433 = (SortedNatTree.contains FastRows.row3433) := by
    change Data.row3433.contains = (SortedNatTree.contains FastRows.row3433)
    exact FastRows.row3433_checked.symm
  rw [row]
  decide
theorem root281Step6 : scan edge (search edge 3) 4 [1861] 1861 root281Tail6 winner =
    scan edge (search edge 3) 4 [1861] 1861 root281Tail7 winner := by
  rw [root281Tail6, scan, if_neg (by decide), root281Candidates6_checked]
  change scan edge (search edge 3) 4 [1861] 1861 root281Tail7
    (if 3 ≤ root281Candidates6.length then search edge 3 [1861, 3433] root281Candidates6 5294 winner else winner) = _
  rw [if_neg (by decide)]
def root281Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root281Candidates7_checked : candidates edge 4 1861 3709 root281Tail8 winner.bound = root281Candidates7 := by
  unfold candidates
  have row : edge 3709 = (SortedNatTree.contains FastRows.row3709) := by
    change Data.row3709.contains = (SortedNatTree.contains FastRows.row3709)
    exact FastRows.row3709_checked.symm
  rw [row]
  decide
theorem root281Step7 : scan edge (search edge 3) 4 [1861] 1861 root281Tail7 winner =
    scan edge (search edge 3) 4 [1861] 1861 root281Tail8 winner := by
  rw [root281Tail7, scan, if_neg (by decide), root281Candidates7_checked]
  change scan edge (search edge 3) 4 [1861] 1861 root281Tail8
    (if 3 ≤ root281Candidates7.length then search edge 3 [1861, 3709] root281Candidates7 5570 winner else winner) = _
  rw [if_neg (by decide)]
def root281Candidates8 : List Nat := [16747]
set_option maxRecDepth 8192 in
theorem root281Candidates8_checked : candidates edge 4 1861 3769 root281Tail9 winner.bound = root281Candidates8 := by
  unfold candidates
  have row : edge 3769 = (SortedNatTree.contains FastRows.row3769) := by
    change Data.row3769.contains = (SortedNatTree.contains FastRows.row3769)
    exact FastRows.row3769_checked.symm
  rw [row]
  decide
theorem root281Step8 : scan edge (search edge 3) 4 [1861] 1861 root281Tail8 winner =
    scan edge (search edge 3) 4 [1861] 1861 root281Tail9 winner := by
  rw [root281Tail8, scan, if_neg (by decide), root281Candidates8_checked]
  change scan edge (search edge 3) 4 [1861] 1861 root281Tail9
    (if 3 ≤ root281Candidates8.length then search edge 3 [1861, 3769] root281Candidates8 5630 winner else winner) = _
  rw [if_neg (by decide)]
def root281Candidates9 : List Nat := [17599]
set_option maxRecDepth 8192 in
theorem root281Candidates9_checked : candidates edge 4 1861 4789 root281Tail10 winner.bound = root281Candidates9 := by
  unfold candidates
  have row : edge 4789 = (SortedNatTree.contains FastRows.row4789) := by
    change Data.row4789.contains = (SortedNatTree.contains FastRows.row4789)
    exact FastRows.row4789_checked.symm
  rw [row]
  decide
theorem root281Step9 : scan edge (search edge 3) 4 [1861] 1861 root281Tail9 winner =
    scan edge (search edge 3) 4 [1861] 1861 root281Tail10 winner := by
  rw [root281Tail9, scan, if_neg (by decide), root281Candidates9_checked]
  change scan edge (search edge 3) 4 [1861] 1861 root281Tail10
    (if 3 ≤ root281Candidates9.length then search edge 3 [1861, 4789] root281Candidates9 6650 winner else winner) = _
  rw [if_neg (by decide)]
def root281Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root281Candidates10_checked : candidates edge 4 1861 5011 root281Tail11 winner.bound = root281Candidates10 := by
  unfold candidates
  have row : edge 5011 = (SortedNatTree.contains FastRows.row5011) := by
    change Data.row5011.contains = (SortedNatTree.contains FastRows.row5011)
    exact FastRows.row5011_checked.symm
  rw [row]
  decide
theorem root281Step10 : scan edge (search edge 3) 4 [1861] 1861 root281Tail10 winner =
    scan edge (search edge 3) 4 [1861] 1861 root281Tail11 winner := by
  rw [root281Tail10, scan, if_neg (by decide), root281Candidates10_checked]
  change scan edge (search edge 3) 4 [1861] 1861 root281Tail11
    (if 3 ≤ root281Candidates10.length then search edge 3 [1861, 5011] root281Candidates10 6872 winner else winner) = _
  rw [if_neg (by decide)]
def root281Candidates11 : List Nat := [7213, 11437]
set_option maxRecDepth 8192 in
theorem root281Candidates11_checked : candidates edge 4 1861 5059 root281Tail12 winner.bound = root281Candidates11 := by
  unfold candidates
  have row : edge 5059 = (SortedNatTree.contains FastRows.row5059) := by
    change Data.row5059.contains = (SortedNatTree.contains FastRows.row5059)
    exact FastRows.row5059_checked.symm
  rw [row]
  decide
theorem root281Step11 : scan edge (search edge 3) 4 [1861] 1861 root281Tail11 winner =
    scan edge (search edge 3) 4 [1861] 1861 root281Tail12 winner := by
  rw [root281Tail11, scan, if_neg (by decide), root281Candidates11_checked]
  change scan edge (search edge 3) 4 [1861] 1861 root281Tail12
    (if 3 ≤ root281Candidates11.length then search edge 3 [1861, 5059] root281Candidates11 6920 winner else winner) = _
  rw [if_neg (by decide)]
def root281Candidates12 : List Nat := [9973, 10501]
set_option maxRecDepth 8192 in
theorem root281Candidates12_checked : candidates edge 4 1861 5479 root281Tail13 winner.bound = root281Candidates12 := by
  unfold candidates
  have row : edge 5479 = (SortedNatTree.contains FastRows.row5479) := by
    change Data.row5479.contains = (SortedNatTree.contains FastRows.row5479)
    exact FastRows.row5479_checked.symm
  rw [row]
  decide
theorem root281Step12 : scan edge (search edge 3) 4 [1861] 1861 root281Tail12 winner =
    scan edge (search edge 3) 4 [1861] 1861 root281Tail13 winner := by
  rw [root281Tail12, scan, if_neg (by decide), root281Candidates12_checked]
  change scan edge (search edge 3) 4 [1861] 1861 root281Tail13
    (if 3 ≤ root281Candidates12.length then search edge 3 [1861, 5479] root281Candidates12 7340 winner else winner) = _
  rw [if_neg (by decide)]
def root281Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root281Candidates13_checked : candidates edge 4 1861 5521 root281Tail14 winner.bound = root281Candidates13 := by
  unfold candidates
  have row : edge 5521 = (SortedNatTree.contains FastRows.row5521) := by
    change Data.row5521.contains = (SortedNatTree.contains FastRows.row5521)
    exact FastRows.row5521_checked.symm
  rw [row]
  decide
theorem root281Step13 : scan edge (search edge 3) 4 [1861] 1861 root281Tail13 winner =
    scan edge (search edge 3) 4 [1861] 1861 root281Tail14 winner := by
  rw [root281Tail13, scan, if_neg (by decide), root281Candidates13_checked]
  change scan edge (search edge 3) 4 [1861] 1861 root281Tail14
    (if 3 ≤ root281Candidates13.length then search edge 3 [1861, 5521] root281Candidates13 7382 winner else winner) = _
  rw [if_neg (by decide)]
def root281Candidates14 : List Nat := [12211]
set_option maxRecDepth 8192 in
theorem root281Candidates14_checked : candidates edge 4 1861 5881 root281Tail15 winner.bound = root281Candidates14 := by
  unfold candidates
  have row : edge 5881 = (SortedNatTree.contains FastRows.row5881) := by
    change Data.row5881.contains = (SortedNatTree.contains FastRows.row5881)
    exact FastRows.row5881_checked.symm
  rw [row]
  decide
theorem root281Step14 : scan edge (search edge 3) 4 [1861] 1861 root281Tail14 winner =
    scan edge (search edge 3) 4 [1861] 1861 root281Tail15 winner := by
  rw [root281Tail14, scan, if_neg (by decide), root281Candidates14_checked]
  change scan edge (search edge 3) 4 [1861] 1861 root281Tail15
    (if 3 ≤ root281Candidates14.length then search edge 3 [1861, 5881] root281Candidates14 7742 winner else winner) = _
  rw [if_neg (by decide)]
def root281Candidates15 : List Nat := [15187]
set_option maxRecDepth 8192 in
theorem root281Candidates15_checked : candidates edge 4 1861 6007 root281Tail16 winner.bound = root281Candidates15 := by
  unfold candidates
  have row : edge 6007 = (SortedNatTree.contains FastRows.row6007) := by
    change Data.row6007.contains = (SortedNatTree.contains FastRows.row6007)
    exact FastRows.row6007_checked.symm
  rw [row]
  decide
theorem root281Step15 : scan edge (search edge 3) 4 [1861] 1861 root281Tail15 winner =
    scan edge (search edge 3) 4 [1861] 1861 root281Tail16 winner := by
  rw [root281Tail15, scan, if_neg (by decide), root281Candidates15_checked]
  change scan edge (search edge 3) 4 [1861] 1861 root281Tail16
    (if 3 ≤ root281Candidates15.length then search edge 3 [1861, 6007] root281Candidates15 7868 winner else winner) = _
  rw [if_neg (by decide)]
theorem root281Scan16 : scan edge (search edge 3) 4 [1861] 1861 root281Tail16 winner = winner := by rw [root281Tail16, scan, if_pos (by decide)]
theorem root281Scan15 : scan edge (search edge 3) 4 [1861] 1861 root281Tail15 winner = winner :=
  (root281Step15).trans root281Scan16
theorem root281Scan14 : scan edge (search edge 3) 4 [1861] 1861 root281Tail14 winner = winner :=
  (root281Step14).trans root281Scan15
theorem root281Scan13 : scan edge (search edge 3) 4 [1861] 1861 root281Tail13 winner = winner :=
  (root281Step13).trans root281Scan14
theorem root281Scan12 : scan edge (search edge 3) 4 [1861] 1861 root281Tail12 winner = winner :=
  (root281Step12).trans root281Scan13
theorem root281Scan11 : scan edge (search edge 3) 4 [1861] 1861 root281Tail11 winner = winner :=
  (root281Step11).trans root281Scan12
theorem root281Scan10 : scan edge (search edge 3) 4 [1861] 1861 root281Tail10 winner = winner :=
  (root281Step10).trans root281Scan11
theorem root281Scan9 : scan edge (search edge 3) 4 [1861] 1861 root281Tail9 winner = winner :=
  (root281Step9).trans root281Scan10
theorem root281Scan8 : scan edge (search edge 3) 4 [1861] 1861 root281Tail8 winner = winner :=
  (root281Step8).trans root281Scan9
theorem root281Scan7 : scan edge (search edge 3) 4 [1861] 1861 root281Tail7 winner = winner :=
  (root281Step7).trans root281Scan8
theorem root281Scan6 : scan edge (search edge 3) 4 [1861] 1861 root281Tail6 winner = winner :=
  (root281Step6).trans root281Scan7
theorem root281Scan5 : scan edge (search edge 3) 4 [1861] 1861 root281Tail5 winner = winner :=
  (root281Step5).trans root281Scan6
theorem root281Scan4 : scan edge (search edge 3) 4 [1861] 1861 root281Tail4 winner = winner :=
  (root281Step4).trans root281Scan5
theorem root281Scan3 : scan edge (search edge 3) 4 [1861] 1861 root281Tail3 winner = winner :=
  (root281Step3).trans root281Scan4
theorem root281Scan2 : scan edge (search edge 3) 4 [1861] 1861 root281Tail2 winner = winner :=
  (root281Step2).trans root281Scan3
theorem root281Scan1 : scan edge (search edge 3) 4 [1861] 1861 root281Tail1 winner = winner :=
  (root281Step1).trans root281Scan2
theorem root281Scan0 : scan edge (search edge 3) 4 [1861] 1861 root281Tail0 winner = winner :=
  (root281Step0).trans root281Scan1
set_option maxRecDepth 4096 in
theorem root281Branch : search edge 4 [1861] root281Candidates 1861 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1861] 1861 root281Tail0 winner = winner
  exact root281Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
