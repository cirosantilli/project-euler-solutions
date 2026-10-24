import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root259Candidates : List Nat := Data.row1667
set_option maxRecDepth 8192 in
theorem root259Candidates_checked : candidates edge 5 0 1667 tail260 winner.bound = root259Candidates := by
  apply candidates_eq_row edge 5 0 1667 tail260 root259Candidates winner.bound tail260_ordered
  · decide
  · decide
  · funext q
    rfl
def root259Tail50 : List Nat := []
def root259Tail49 : List Nat := 23873 :: root259Tail50
def root259Tail48 : List Nat := 23747 :: root259Tail49
def root259Tail47 : List Nat := 23741 :: root259Tail48
def root259Tail46 : List Nat := 23663 :: root259Tail47
def root259Tail45 : List Nat := 22877 :: root259Tail46
def root259Tail44 : List Nat := 22091 :: root259Tail45
def root259Tail43 : List Nat := 21881 :: root259Tail44
def root259Tail42 : List Nat := 21773 :: root259Tail43
def root259Tail41 : List Nat := 21701 :: root259Tail42
def root259Tail40 : List Nat := 20639 :: root259Tail41
def root259Tail39 : List Nat := 20129 :: root259Tail40
def root259Tail38 : List Nat := 19181 :: root259Tail39
def root259Tail37 : List Nat := 18701 :: root259Tail38
def root259Tail36 : List Nat := 18059 :: root259Tail37
def root259Tail35 : List Nat := 17987 :: root259Tail36
def root259Tail34 : List Nat := 16883 :: root259Tail35
def root259Tail33 : List Nat := 16787 :: root259Tail34
def root259Tail32 : List Nat := 16631 :: root259Tail33
def root259Tail31 : List Nat := 16349 :: root259Tail32
def root259Tail30 : List Nat := 15233 :: root259Tail31
def root259Tail29 : List Nat := 14447 :: root259Tail30
def root259Tail28 : List Nat := 14321 :: root259Tail29
def root259Tail27 : List Nat := 13457 :: root259Tail28
def root259Tail26 : List Nat := 12113 :: root259Tail27
def root259Tail25 : List Nat := 10559 :: root259Tail26
def root259Tail24 : List Nat := 10259 :: root259Tail25
def root259Tail23 : List Nat := 10091 :: root259Tail24
def root259Tail22 : List Nat := 9137 :: root259Tail23
def root259Tail21 : List Nat := 8663 :: root259Tail22
def root259Tail20 : List Nat := 6701 :: root259Tail21
def root259Tail19 : List Nat := 6353 :: root259Tail20
def root259Tail18 : List Nat := 6329 :: root259Tail19
def root259Tail17 : List Nat := 6323 :: root259Tail18
def root259Tail16 : List Nat := 6287 :: root259Tail17
def root259Tail15 : List Nat := 5669 :: root259Tail16
def root259Tail14 : List Nat := 5573 :: root259Tail15
def root259Tail13 : List Nat := 5531 :: root259Tail14
def root259Tail12 : List Nat := 5051 :: root259Tail13
def root259Tail11 : List Nat := 4967 :: root259Tail12
def root259Tail10 : List Nat := 4817 :: root259Tail11
def root259Tail9 : List Nat := 4703 :: root259Tail10
def root259Tail8 : List Nat := 4481 :: root259Tail9
def root259Tail7 : List Nat := 4157 :: root259Tail8
def root259Tail6 : List Nat := 3929 :: root259Tail7
def root259Tail5 : List Nat := 3623 :: root259Tail6
def root259Tail4 : List Nat := 2657 :: root259Tail5
def root259Tail3 : List Nat := 2267 :: root259Tail4
def root259Tail2 : List Nat := 2153 :: root259Tail3
def root259Tail1 : List Nat := 1973 :: root259Tail2
def root259Tail0 : List Nat := 1901 :: root259Tail1
def root259Candidates0 : List Nat := [2153, 4817, 8663]
set_option maxRecDepth 8192 in
theorem root259Candidates0_checked : candidates edge 4 1667 1901 root259Tail1 winner.bound = root259Candidates0 := by
  unfold candidates
  have row : edge 1901 = (SortedNatTree.contains FastRows.row1901) := by
    change Data.row1901.contains = (SortedNatTree.contains FastRows.row1901)
    exact FastRows.row1901_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root259Child0 : search edge 3 [1667, 1901] root259Candidates0 3568 winner = winner := by decide
theorem root259Step0 : scan edge (search edge 3) 4 [1667] 1667 root259Tail0 winner =
    scan edge (search edge 3) 4 [1667] 1667 root259Tail1 winner := by
  rw [root259Tail0, scan, if_neg (by decide), root259Candidates0_checked]
  change scan edge (search edge 3) 4 [1667] 1667 root259Tail1
    (if 3 ≤ root259Candidates0.length then search edge 3 [1667, 1901] root259Candidates0 3568 winner else winner) = _
  rw [if_pos (by decide), root259Child0]
def root259Candidates1 : List Nat := [15233, 17987, 21881]
set_option maxRecDepth 8192 in
theorem root259Candidates1_checked : candidates edge 4 1667 1973 root259Tail2 winner.bound = root259Candidates1 := by
  unfold candidates
  have row : edge 1973 = (SortedNatTree.contains FastRows.row1973) := by
    change Data.row1973.contains = (SortedNatTree.contains FastRows.row1973)
    exact FastRows.row1973_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root259Child1 : search edge 3 [1667, 1973] root259Candidates1 3640 winner = winner := by decide
theorem root259Step1 : scan edge (search edge 3) 4 [1667] 1667 root259Tail1 winner =
    scan edge (search edge 3) 4 [1667] 1667 root259Tail2 winner := by
  rw [root259Tail1, scan, if_neg (by decide), root259Candidates1_checked]
  change scan edge (search edge 3) 4 [1667] 1667 root259Tail2
    (if 3 ≤ root259Candidates1.length then search edge 3 [1667, 1973] root259Candidates1 3640 winner else winner) = _
  rw [if_pos (by decide), root259Child1]
def root259Candidates2 : List Nat := [4967, 9137]
set_option maxRecDepth 8192 in
theorem root259Candidates2_checked : candidates edge 4 1667 2153 root259Tail3 winner.bound = root259Candidates2 := by
  unfold candidates
  have row : edge 2153 = (SortedNatTree.contains FastRows.row2153) := by
    change Data.row2153.contains = (SortedNatTree.contains FastRows.row2153)
    exact FastRows.row2153_checked.symm
  rw [row]
  decide
theorem root259Step2 : scan edge (search edge 3) 4 [1667] 1667 root259Tail2 winner =
    scan edge (search edge 3) 4 [1667] 1667 root259Tail3 winner := by
  rw [root259Tail2, scan, if_neg (by decide), root259Candidates2_checked]
  change scan edge (search edge 3) 4 [1667] 1667 root259Tail3
    (if 3 ≤ root259Candidates2.length then search edge 3 [1667, 2153] root259Candidates2 3820 winner else winner) = _
  rw [if_neg (by decide)]
def root259Candidates3 : List Nat := [5669, 16883]
set_option maxRecDepth 8192 in
theorem root259Candidates3_checked : candidates edge 4 1667 2267 root259Tail4 winner.bound = root259Candidates3 := by
  unfold candidates
  have row : edge 2267 = (SortedNatTree.contains FastRows.row2267) := by
    change Data.row2267.contains = (SortedNatTree.contains FastRows.row2267)
    exact FastRows.row2267_checked.symm
  rw [row]
  decide
theorem root259Step3 : scan edge (search edge 3) 4 [1667] 1667 root259Tail3 winner =
    scan edge (search edge 3) 4 [1667] 1667 root259Tail4 winner := by
  rw [root259Tail3, scan, if_neg (by decide), root259Candidates3_checked]
  change scan edge (search edge 3) 4 [1667] 1667 root259Tail4
    (if 3 ≤ root259Candidates3.length then search edge 3 [1667, 2267] root259Candidates3 3934 winner else winner) = _
  rw [if_neg (by decide)]
def root259Candidates4 : List Nat := [20639]
set_option maxRecDepth 8192 in
theorem root259Candidates4_checked : candidates edge 4 1667 2657 root259Tail5 winner.bound = root259Candidates4 := by
  unfold candidates
  have row : edge 2657 = (SortedNatTree.contains FastRows.row2657) := by
    change Data.row2657.contains = (SortedNatTree.contains FastRows.row2657)
    exact FastRows.row2657_checked.symm
  rw [row]
  decide
theorem root259Step4 : scan edge (search edge 3) 4 [1667] 1667 root259Tail4 winner =
    scan edge (search edge 3) 4 [1667] 1667 root259Tail5 winner := by
  rw [root259Tail4, scan, if_neg (by decide), root259Candidates4_checked]
  change scan edge (search edge 3) 4 [1667] 1667 root259Tail5
    (if 3 ≤ root259Candidates4.length then search edge 3 [1667, 2657] root259Candidates4 4324 winner else winner) = _
  rw [if_neg (by decide)]
def root259Candidates5 : List Nat := [4967]
set_option maxRecDepth 8192 in
theorem root259Candidates5_checked : candidates edge 4 1667 3623 root259Tail6 winner.bound = root259Candidates5 := by
  unfold candidates
  have row : edge 3623 = (SortedNatTree.contains FastRows.row3623) := by
    change Data.row3623.contains = (SortedNatTree.contains FastRows.row3623)
    exact FastRows.row3623_checked.symm
  rw [row]
  decide
theorem root259Step5 : scan edge (search edge 3) 4 [1667] 1667 root259Tail5 winner =
    scan edge (search edge 3) 4 [1667] 1667 root259Tail6 winner := by
  rw [root259Tail5, scan, if_neg (by decide), root259Candidates5_checked]
  change scan edge (search edge 3) 4 [1667] 1667 root259Tail6
    (if 3 ≤ root259Candidates5.length then search edge 3 [1667, 3623] root259Candidates5 5290 winner else winner) = _
  rw [if_neg (by decide)]
def root259Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root259Candidates6_checked : candidates edge 4 1667 3929 root259Tail7 winner.bound = root259Candidates6 := by
  unfold candidates
  have row : edge 3929 = (SortedNatTree.contains FastRows.row3929) := by
    change Data.row3929.contains = (SortedNatTree.contains FastRows.row3929)
    exact FastRows.row3929_checked.symm
  rw [row]
  decide
theorem root259Step6 : scan edge (search edge 3) 4 [1667] 1667 root259Tail6 winner =
    scan edge (search edge 3) 4 [1667] 1667 root259Tail7 winner := by
  rw [root259Tail6, scan, if_neg (by decide), root259Candidates6_checked]
  change scan edge (search edge 3) 4 [1667] 1667 root259Tail7
    (if 3 ≤ root259Candidates6.length then search edge 3 [1667, 3929] root259Candidates6 5596 winner else winner) = _
  rw [if_neg (by decide)]
def root259Candidates7 : List Nat := [10259]
set_option maxRecDepth 8192 in
theorem root259Candidates7_checked : candidates edge 4 1667 4157 root259Tail8 winner.bound = root259Candidates7 := by
  unfold candidates
  have row : edge 4157 = (SortedNatTree.contains FastRows.row4157) := by
    change Data.row4157.contains = (SortedNatTree.contains FastRows.row4157)
    exact FastRows.row4157_checked.symm
  rw [row]
  decide
theorem root259Step7 : scan edge (search edge 3) 4 [1667] 1667 root259Tail7 winner =
    scan edge (search edge 3) 4 [1667] 1667 root259Tail8 winner := by
  rw [root259Tail7, scan, if_neg (by decide), root259Candidates7_checked]
  change scan edge (search edge 3) 4 [1667] 1667 root259Tail8
    (if 3 ≤ root259Candidates7.length then search edge 3 [1667, 4157] root259Candidates7 5824 winner else winner) = _
  rw [if_neg (by decide)]
def root259Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root259Candidates8_checked : candidates edge 4 1667 4481 root259Tail9 winner.bound = root259Candidates8 := by
  unfold candidates
  have row : edge 4481 = (SortedNatTree.contains FastRows.row4481) := by
    change Data.row4481.contains = (SortedNatTree.contains FastRows.row4481)
    exact FastRows.row4481_checked.symm
  rw [row]
  decide
theorem root259Step8 : scan edge (search edge 3) 4 [1667] 1667 root259Tail8 winner =
    scan edge (search edge 3) 4 [1667] 1667 root259Tail9 winner := by
  rw [root259Tail8, scan, if_neg (by decide), root259Candidates8_checked]
  change scan edge (search edge 3) 4 [1667] 1667 root259Tail9
    (if 3 ≤ root259Candidates8.length then search edge 3 [1667, 4481] root259Candidates8 6148 winner else winner) = _
  rw [if_neg (by decide)]
def root259Candidates9 : List Nat := [5531, 6329]
set_option maxRecDepth 8192 in
theorem root259Candidates9_checked : candidates edge 4 1667 4703 root259Tail10 winner.bound = root259Candidates9 := by
  unfold candidates
  have row : edge 4703 = (SortedNatTree.contains FastRows.row4703) := by
    change Data.row4703.contains = (SortedNatTree.contains FastRows.row4703)
    exact FastRows.row4703_checked.symm
  rw [row]
  decide
theorem root259Step9 : scan edge (search edge 3) 4 [1667] 1667 root259Tail9 winner =
    scan edge (search edge 3) 4 [1667] 1667 root259Tail10 winner := by
  rw [root259Tail9, scan, if_neg (by decide), root259Candidates9_checked]
  change scan edge (search edge 3) 4 [1667] 1667 root259Tail10
    (if 3 ≤ root259Candidates9.length then search edge 3 [1667, 4703] root259Candidates9 6370 winner else winner) = _
  rw [if_neg (by decide)]
def root259Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root259Candidates10_checked : candidates edge 4 1667 4817 root259Tail11 winner.bound = root259Candidates10 := by
  unfold candidates
  have row : edge 4817 = (SortedNatTree.contains FastRows.row4817) := by
    change Data.row4817.contains = (SortedNatTree.contains FastRows.row4817)
    exact FastRows.row4817_checked.symm
  rw [row]
  decide
theorem root259Step10 : scan edge (search edge 3) 4 [1667] 1667 root259Tail10 winner =
    scan edge (search edge 3) 4 [1667] 1667 root259Tail11 winner := by
  rw [root259Tail10, scan, if_neg (by decide), root259Candidates10_checked]
  change scan edge (search edge 3) 4 [1667] 1667 root259Tail11
    (if 3 ≤ root259Candidates10.length then search edge 3 [1667, 4817] root259Candidates10 6484 winner else winner) = _
  rw [if_neg (by decide)]
def root259Candidates11 : List Nat := [5531, 5573]
set_option maxRecDepth 8192 in
theorem root259Candidates11_checked : candidates edge 4 1667 4967 root259Tail12 winner.bound = root259Candidates11 := by
  unfold candidates
  have row : edge 4967 = (SortedNatTree.contains FastRows.row4967) := by
    change Data.row4967.contains = (SortedNatTree.contains FastRows.row4967)
    exact FastRows.row4967_checked.symm
  rw [row]
  decide
theorem root259Step11 : scan edge (search edge 3) 4 [1667] 1667 root259Tail11 winner =
    scan edge (search edge 3) 4 [1667] 1667 root259Tail12 winner := by
  rw [root259Tail11, scan, if_neg (by decide), root259Candidates11_checked]
  change scan edge (search edge 3) 4 [1667] 1667 root259Tail12
    (if 3 ≤ root259Candidates11.length then search edge 3 [1667, 4967] root259Candidates11 6634 winner else winner) = _
  rw [if_neg (by decide)]
def root259Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root259Candidates12_checked : candidates edge 4 1667 5051 root259Tail13 winner.bound = root259Candidates12 := by
  unfold candidates
  have row : edge 5051 = (SortedNatTree.contains FastRows.row5051) := by
    change Data.row5051.contains = (SortedNatTree.contains FastRows.row5051)
    exact FastRows.row5051_checked.symm
  rw [row]
  decide
theorem root259Step12 : scan edge (search edge 3) 4 [1667] 1667 root259Tail12 winner =
    scan edge (search edge 3) 4 [1667] 1667 root259Tail13 winner := by
  rw [root259Tail12, scan, if_neg (by decide), root259Candidates12_checked]
  change scan edge (search edge 3) 4 [1667] 1667 root259Tail13
    (if 3 ≤ root259Candidates12.length then search edge 3 [1667, 5051] root259Candidates12 6718 winner else winner) = _
  rw [if_neg (by decide)]
def root259Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root259Candidates13_checked : candidates edge 4 1667 5531 root259Tail14 winner.bound = root259Candidates13 := by
  unfold candidates
  have row : edge 5531 = (SortedNatTree.contains FastRows.row5531) := by
    change Data.row5531.contains = (SortedNatTree.contains FastRows.row5531)
    exact FastRows.row5531_checked.symm
  rw [row]
  decide
theorem root259Step13 : scan edge (search edge 3) 4 [1667] 1667 root259Tail13 winner =
    scan edge (search edge 3) 4 [1667] 1667 root259Tail14 winner := by
  rw [root259Tail13, scan, if_neg (by decide), root259Candidates13_checked]
  change scan edge (search edge 3) 4 [1667] 1667 root259Tail14
    (if 3 ≤ root259Candidates13.length then search edge 3 [1667, 5531] root259Candidates13 7198 winner else winner) = _
  rw [if_neg (by decide)]
def root259Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root259Candidates14_checked : candidates edge 4 1667 5573 root259Tail15 winner.bound = root259Candidates14 := by
  unfold candidates
  have row : edge 5573 = (SortedNatTree.contains FastRows.row5573) := by
    change Data.row5573.contains = (SortedNatTree.contains FastRows.row5573)
    exact FastRows.row5573_checked.symm
  rw [row]
  decide
theorem root259Step14 : scan edge (search edge 3) 4 [1667] 1667 root259Tail14 winner =
    scan edge (search edge 3) 4 [1667] 1667 root259Tail15 winner := by
  rw [root259Tail14, scan, if_neg (by decide), root259Candidates14_checked]
  change scan edge (search edge 3) 4 [1667] 1667 root259Tail15
    (if 3 ≤ root259Candidates14.length then search edge 3 [1667, 5573] root259Candidates14 7240 winner else winner) = _
  rw [if_neg (by decide)]
def root259Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root259Candidates15_checked : candidates edge 4 1667 5669 root259Tail16 winner.bound = root259Candidates15 := by
  unfold candidates
  have row : edge 5669 = (SortedNatTree.contains FastRows.row5669) := by
    change Data.row5669.contains = (SortedNatTree.contains FastRows.row5669)
    exact FastRows.row5669_checked.symm
  rw [row]
  decide
theorem root259Step15 : scan edge (search edge 3) 4 [1667] 1667 root259Tail15 winner =
    scan edge (search edge 3) 4 [1667] 1667 root259Tail16 winner := by
  rw [root259Tail15, scan, if_neg (by decide), root259Candidates15_checked]
  change scan edge (search edge 3) 4 [1667] 1667 root259Tail16
    (if 3 ≤ root259Candidates15.length then search edge 3 [1667, 5669] root259Candidates15 7336 winner else winner) = _
  rw [if_neg (by decide)]
theorem root259Scan16 : scan edge (search edge 3) 4 [1667] 1667 root259Tail16 winner = winner := by rw [root259Tail16, scan, if_pos (by decide)]
theorem root259Scan15 : scan edge (search edge 3) 4 [1667] 1667 root259Tail15 winner = winner :=
  (root259Step15).trans root259Scan16
theorem root259Scan14 : scan edge (search edge 3) 4 [1667] 1667 root259Tail14 winner = winner :=
  (root259Step14).trans root259Scan15
theorem root259Scan13 : scan edge (search edge 3) 4 [1667] 1667 root259Tail13 winner = winner :=
  (root259Step13).trans root259Scan14
theorem root259Scan12 : scan edge (search edge 3) 4 [1667] 1667 root259Tail12 winner = winner :=
  (root259Step12).trans root259Scan13
theorem root259Scan11 : scan edge (search edge 3) 4 [1667] 1667 root259Tail11 winner = winner :=
  (root259Step11).trans root259Scan12
theorem root259Scan10 : scan edge (search edge 3) 4 [1667] 1667 root259Tail10 winner = winner :=
  (root259Step10).trans root259Scan11
theorem root259Scan9 : scan edge (search edge 3) 4 [1667] 1667 root259Tail9 winner = winner :=
  (root259Step9).trans root259Scan10
theorem root259Scan8 : scan edge (search edge 3) 4 [1667] 1667 root259Tail8 winner = winner :=
  (root259Step8).trans root259Scan9
theorem root259Scan7 : scan edge (search edge 3) 4 [1667] 1667 root259Tail7 winner = winner :=
  (root259Step7).trans root259Scan8
theorem root259Scan6 : scan edge (search edge 3) 4 [1667] 1667 root259Tail6 winner = winner :=
  (root259Step6).trans root259Scan7
theorem root259Scan5 : scan edge (search edge 3) 4 [1667] 1667 root259Tail5 winner = winner :=
  (root259Step5).trans root259Scan6
theorem root259Scan4 : scan edge (search edge 3) 4 [1667] 1667 root259Tail4 winner = winner :=
  (root259Step4).trans root259Scan5
theorem root259Scan3 : scan edge (search edge 3) 4 [1667] 1667 root259Tail3 winner = winner :=
  (root259Step3).trans root259Scan4
theorem root259Scan2 : scan edge (search edge 3) 4 [1667] 1667 root259Tail2 winner = winner :=
  (root259Step2).trans root259Scan3
theorem root259Scan1 : scan edge (search edge 3) 4 [1667] 1667 root259Tail1 winner = winner :=
  (root259Step1).trans root259Scan2
theorem root259Scan0 : scan edge (search edge 3) 4 [1667] 1667 root259Tail0 winner = winner :=
  (root259Step0).trans root259Scan1
set_option maxRecDepth 4096 in
theorem root259Branch : search edge 4 [1667] root259Candidates 1667 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1667] 1667 root259Tail0 winner = winner
  exact root259Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
