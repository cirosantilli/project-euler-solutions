import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root178Candidates : List Nat := Data.row1087
set_option maxRecDepth 8192 in
theorem root178Candidates_checked : candidates edge 5 0 1087 tail179 winner.bound = root178Candidates := by
  apply candidates_eq_row edge 5 0 1087 tail179 root178Candidates winner.bound tail179_ordered
  · decide
  · decide
  · funext q
    rfl
def root178Tail52 : List Nat := []
def root178Tail51 : List Nat := 24943 :: root178Tail52
def root178Tail50 : List Nat := 24481 :: root178Tail51
def root178Tail49 : List Nat := 23071 :: root178Tail50
def root178Tail48 : List Nat := 22063 :: root178Tail49
def root178Tail47 : List Nat := 21757 :: root178Tail48
def root178Tail46 : List Nat := 21751 :: root178Tail47
def root178Tail45 : List Nat := 21517 :: root178Tail46
def root178Tail44 : List Nat := 21499 :: root178Tail45
def root178Tail43 : List Nat := 21139 :: root178Tail44
def root178Tail42 : List Nat := 20749 :: root178Tail43
def root178Tail41 : List Nat := 20047 :: root178Tail42
def root178Tail40 : List Nat := 19423 :: root178Tail41
def root178Tail39 : List Nat := 18307 :: root178Tail40
def root178Tail38 : List Nat := 18289 :: root178Tail39
def root178Tail37 : List Nat := 17971 :: root178Tail38
def root178Tail36 : List Nat := 17443 :: root178Tail37
def root178Tail35 : List Nat := 16987 :: root178Tail36
def root178Tail34 : List Nat := 16519 :: root178Tail35
def root178Tail33 : List Nat := 15199 :: root178Tail34
def root178Tail32 : List Nat := 14869 :: root178Tail33
def root178Tail31 : List Nat := 13921 :: root178Tail32
def root178Tail30 : List Nat := 13879 :: root178Tail31
def root178Tail29 : List Nat := 13831 :: root178Tail30
def root178Tail28 : List Nat := 13537 :: root178Tail29
def root178Tail27 : List Nat := 13327 :: root178Tail28
def root178Tail26 : List Nat := 12979 :: root178Tail27
def root178Tail25 : List Nat := 12829 :: root178Tail26
def root178Tail24 : List Nat := 10663 :: root178Tail25
def root178Tail23 : List Nat := 10453 :: root178Tail24
def root178Tail22 : List Nat := 10429 :: root178Tail23
def root178Tail21 : List Nat := 9511 :: root178Tail22
def root178Tail20 : List Nat := 9403 :: root178Tail21
def root178Tail19 : List Nat := 9241 :: root178Tail20
def root178Tail18 : List Nat := 9199 :: root178Tail19
def root178Tail17 : List Nat := 9007 :: root178Tail18
def root178Tail16 : List Nat := 6967 :: root178Tail17
def root178Tail15 : List Nat := 6763 :: root178Tail16
def root178Tail14 : List Nat := 6451 :: root178Tail15
def root178Tail13 : List Nat := 6091 :: root178Tail14
def root178Tail12 : List Nat := 5569 :: root178Tail13
def root178Tail11 : List Nat := 5413 :: root178Tail12
def root178Tail10 : List Nat := 5107 :: root178Tail11
def root178Tail9 : List Nat := 4957 :: root178Tail10
def root178Tail8 : List Nat := 4789 :: root178Tail9
def root178Tail7 : List Nat := 4447 :: root178Tail8
def root178Tail6 : List Nat := 3847 :: root178Tail7
def root178Tail5 : List Nat := 3271 :: root178Tail6
def root178Tail4 : List Nat := 2833 :: root178Tail5
def root178Tail3 : List Nat := 2179 :: root178Tail4
def root178Tail2 : List Nat := 2137 :: root178Tail3
def root178Tail1 : List Nat := 1453 :: root178Tail2
def root178Tail0 : List Nat := 1117 :: root178Tail1
def root178Candidates0 : List Nat := [9199, 10663, 17971]
set_option maxRecDepth 8192 in
theorem root178Candidates0_checked : candidates edge 4 1087 1117 root178Tail1 winner.bound = root178Candidates0 := by
  unfold candidates
  have row : edge 1117 = (SortedNatTree.contains FastRows.row1117) := by
    change Data.row1117.contains = (SortedNatTree.contains FastRows.row1117)
    exact FastRows.row1117_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root178Child0 : search edge 3 [1087, 1117] root178Candidates0 2204 winner = winner := by decide
theorem root178Step0 : scan edge (search edge 3) 4 [1087] 1087 root178Tail0 winner =
    scan edge (search edge 3) 4 [1087] 1087 root178Tail1 winner := by
  rw [root178Tail0, scan, if_neg (by decide), root178Candidates0_checked]
  change scan edge (search edge 3) 4 [1087] 1087 root178Tail1
    (if 3 ≤ root178Candidates0.length then search edge 3 [1087, 1117] root178Candidates0 2204 winner else winner) = _
  rw [if_pos (by decide), root178Child0]
def root178Candidates1 : List Nat := [6763, 17971]
set_option maxRecDepth 8192 in
theorem root178Candidates1_checked : candidates edge 4 1087 1453 root178Tail2 winner.bound = root178Candidates1 := by
  unfold candidates
  have row : edge 1453 = (SortedNatTree.contains FastRows.row1453) := by
    change Data.row1453.contains = (SortedNatTree.contains FastRows.row1453)
    exact FastRows.row1453_checked.symm
  rw [row]
  decide
theorem root178Step1 : scan edge (search edge 3) 4 [1087] 1087 root178Tail1 winner =
    scan edge (search edge 3) 4 [1087] 1087 root178Tail2 winner := by
  rw [root178Tail1, scan, if_neg (by decide), root178Candidates1_checked]
  change scan edge (search edge 3) 4 [1087] 1087 root178Tail2
    (if 3 ≤ root178Candidates1.length then search edge 3 [1087, 1453] root178Candidates1 2540 winner else winner) = _
  rw [if_neg (by decide)]
def root178Candidates2 : List Nat := [13921]
set_option maxRecDepth 8192 in
theorem root178Candidates2_checked : candidates edge 4 1087 2137 root178Tail3 winner.bound = root178Candidates2 := by
  unfold candidates
  have row : edge 2137 = (SortedNatTree.contains FastRows.row2137) := by
    change Data.row2137.contains = (SortedNatTree.contains FastRows.row2137)
    exact FastRows.row2137_checked.symm
  rw [row]
  decide
theorem root178Step2 : scan edge (search edge 3) 4 [1087] 1087 root178Tail2 winner =
    scan edge (search edge 3) 4 [1087] 1087 root178Tail3 winner := by
  rw [root178Tail2, scan, if_neg (by decide), root178Candidates2_checked]
  change scan edge (search edge 3) 4 [1087] 1087 root178Tail3
    (if 3 ≤ root178Candidates2.length then search edge 3 [1087, 2137] root178Candidates2 3224 winner else winner) = _
  rw [if_neg (by decide)]
def root178Candidates3 : List Nat := [9241, 10453, 18289, 19423]
set_option maxRecDepth 8192 in
theorem root178Candidates3_checked : candidates edge 4 1087 2179 root178Tail4 winner.bound = root178Candidates3 := by
  unfold candidates
  have row : edge 2179 = (SortedNatTree.contains FastRows.row2179) := by
    change Data.row2179.contains = (SortedNatTree.contains FastRows.row2179)
    exact FastRows.row2179_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root178Child3 : search edge 3 [1087, 2179] root178Candidates3 3266 winner = winner := by decide
theorem root178Step3 : scan edge (search edge 3) 4 [1087] 1087 root178Tail3 winner =
    scan edge (search edge 3) 4 [1087] 1087 root178Tail4 winner := by
  rw [root178Tail3, scan, if_neg (by decide), root178Candidates3_checked]
  change scan edge (search edge 3) 4 [1087] 1087 root178Tail4
    (if 3 ≤ root178Candidates3.length then search edge 3 [1087, 2179] root178Candidates3 3266 winner else winner) = _
  rw [if_pos (by decide), root178Child3]
def root178Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root178Candidates4_checked : candidates edge 4 1087 2833 root178Tail5 winner.bound = root178Candidates4 := by
  unfold candidates
  have row : edge 2833 = (SortedNatTree.contains FastRows.row2833) := by
    change Data.row2833.contains = (SortedNatTree.contains FastRows.row2833)
    exact FastRows.row2833_checked.symm
  rw [row]
  decide
theorem root178Step4 : scan edge (search edge 3) 4 [1087] 1087 root178Tail4 winner =
    scan edge (search edge 3) 4 [1087] 1087 root178Tail5 winner := by
  rw [root178Tail4, scan, if_neg (by decide), root178Candidates4_checked]
  change scan edge (search edge 3) 4 [1087] 1087 root178Tail5
    (if 3 ≤ root178Candidates4.length then search edge 3 [1087, 2833] root178Candidates4 3920 winner else winner) = _
  rw [if_neg (by decide)]
def root178Candidates5 : List Nat := [9199, 13921]
set_option maxRecDepth 8192 in
theorem root178Candidates5_checked : candidates edge 4 1087 3271 root178Tail6 winner.bound = root178Candidates5 := by
  unfold candidates
  have row : edge 3271 = (SortedNatTree.contains FastRows.row3271) := by
    change Data.row3271.contains = (SortedNatTree.contains FastRows.row3271)
    exact FastRows.row3271_checked.symm
  rw [row]
  decide
theorem root178Step5 : scan edge (search edge 3) 4 [1087] 1087 root178Tail5 winner =
    scan edge (search edge 3) 4 [1087] 1087 root178Tail6 winner := by
  rw [root178Tail5, scan, if_neg (by decide), root178Candidates5_checked]
  change scan edge (search edge 3) 4 [1087] 1087 root178Tail6
    (if 3 ≤ root178Candidates5.length then search edge 3 [1087, 3271] root178Candidates5 4358 winner else winner) = _
  rw [if_neg (by decide)]
def root178Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root178Candidates6_checked : candidates edge 4 1087 3847 root178Tail7 winner.bound = root178Candidates6 := by
  unfold candidates
  have row : edge 3847 = (SortedNatTree.contains FastRows.row3847) := by
    change Data.row3847.contains = (SortedNatTree.contains FastRows.row3847)
    exact FastRows.row3847_checked.symm
  rw [row]
  decide
theorem root178Step6 : scan edge (search edge 3) 4 [1087] 1087 root178Tail6 winner =
    scan edge (search edge 3) 4 [1087] 1087 root178Tail7 winner := by
  rw [root178Tail6, scan, if_neg (by decide), root178Candidates6_checked]
  change scan edge (search edge 3) 4 [1087] 1087 root178Tail7
    (if 3 ≤ root178Candidates6.length then search edge 3 [1087, 3847] root178Candidates6 4934 winner else winner) = _
  rw [if_neg (by decide)]
def root178Candidates7 : List Nat := [4789, 18289, 20047]
set_option maxRecDepth 8192 in
theorem root178Candidates7_checked : candidates edge 4 1087 4447 root178Tail8 winner.bound = root178Candidates7 := by
  unfold candidates
  have row : edge 4447 = (SortedNatTree.contains FastRows.row4447) := by
    change Data.row4447.contains = (SortedNatTree.contains FastRows.row4447)
    exact FastRows.row4447_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root178Child7 : search edge 3 [1087, 4447] root178Candidates7 5534 winner = winner := by decide
theorem root178Step7 : scan edge (search edge 3) 4 [1087] 1087 root178Tail7 winner =
    scan edge (search edge 3) 4 [1087] 1087 root178Tail8 winner := by
  rw [root178Tail7, scan, if_neg (by decide), root178Candidates7_checked]
  change scan edge (search edge 3) 4 [1087] 1087 root178Tail8
    (if 3 ≤ root178Candidates7.length then search edge 3 [1087, 4447] root178Candidates7 5534 winner else winner) = _
  rw [if_pos (by decide), root178Child7]
def root178Candidates8 : List Nat := [12979]
set_option maxRecDepth 8192 in
theorem root178Candidates8_checked : candidates edge 4 1087 4789 root178Tail9 winner.bound = root178Candidates8 := by
  unfold candidates
  have row : edge 4789 = (SortedNatTree.contains FastRows.row4789) := by
    change Data.row4789.contains = (SortedNatTree.contains FastRows.row4789)
    exact FastRows.row4789_checked.symm
  rw [row]
  decide
theorem root178Step8 : scan edge (search edge 3) 4 [1087] 1087 root178Tail8 winner =
    scan edge (search edge 3) 4 [1087] 1087 root178Tail9 winner := by
  rw [root178Tail8, scan, if_neg (by decide), root178Candidates8_checked]
  change scan edge (search edge 3) 4 [1087] 1087 root178Tail9
    (if 3 ≤ root178Candidates8.length then search edge 3 [1087, 4789] root178Candidates8 5876 winner else winner) = _
  rw [if_neg (by decide)]
def root178Candidates9 : List Nat := [5569]
set_option maxRecDepth 8192 in
theorem root178Candidates9_checked : candidates edge 4 1087 4957 root178Tail10 winner.bound = root178Candidates9 := by
  unfold candidates
  have row : edge 4957 = (SortedNatTree.contains FastRows.row4957) := by
    change Data.row4957.contains = (SortedNatTree.contains FastRows.row4957)
    exact FastRows.row4957_checked.symm
  rw [row]
  decide
theorem root178Step9 : scan edge (search edge 3) 4 [1087] 1087 root178Tail9 winner =
    scan edge (search edge 3) 4 [1087] 1087 root178Tail10 winner := by
  rw [root178Tail9, scan, if_neg (by decide), root178Candidates9_checked]
  change scan edge (search edge 3) 4 [1087] 1087 root178Tail10
    (if 3 ≤ root178Candidates9.length then search edge 3 [1087, 4957] root178Candidates9 6044 winner else winner) = _
  rw [if_neg (by decide)]
def root178Candidates10 : List Nat := [6763, 9403]
set_option maxRecDepth 8192 in
theorem root178Candidates10_checked : candidates edge 4 1087 5107 root178Tail11 winner.bound = root178Candidates10 := by
  unfold candidates
  have row : edge 5107 = (SortedNatTree.contains FastRows.row5107) := by
    change Data.row5107.contains = (SortedNatTree.contains FastRows.row5107)
    exact FastRows.row5107_checked.symm
  rw [row]
  decide
theorem root178Step10 : scan edge (search edge 3) 4 [1087] 1087 root178Tail10 winner =
    scan edge (search edge 3) 4 [1087] 1087 root178Tail11 winner := by
  rw [root178Tail10, scan, if_neg (by decide), root178Candidates10_checked]
  change scan edge (search edge 3) 4 [1087] 1087 root178Tail11
    (if 3 ≤ root178Candidates10.length then search edge 3 [1087, 5107] root178Candidates10 6194 winner else winner) = _
  rw [if_neg (by decide)]
def root178Candidates11 : List Nat := [5569, 6451]
set_option maxRecDepth 8192 in
theorem root178Candidates11_checked : candidates edge 4 1087 5413 root178Tail12 winner.bound = root178Candidates11 := by
  unfold candidates
  have row : edge 5413 = (SortedNatTree.contains FastRows.row5413) := by
    change Data.row5413.contains = (SortedNatTree.contains FastRows.row5413)
    exact FastRows.row5413_checked.symm
  rw [row]
  decide
theorem root178Step11 : scan edge (search edge 3) 4 [1087] 1087 root178Tail11 winner =
    scan edge (search edge 3) 4 [1087] 1087 root178Tail12 winner := by
  rw [root178Tail11, scan, if_neg (by decide), root178Candidates11_checked]
  change scan edge (search edge 3) 4 [1087] 1087 root178Tail12
    (if 3 ≤ root178Candidates11.length then search edge 3 [1087, 5413] root178Candidates11 6500 winner else winner) = _
  rw [if_neg (by decide)]
def root178Candidates12 : List Nat := [6451]
set_option maxRecDepth 8192 in
theorem root178Candidates12_checked : candidates edge 4 1087 5569 root178Tail13 winner.bound = root178Candidates12 := by
  unfold candidates
  have row : edge 5569 = (SortedNatTree.contains FastRows.row5569) := by
    change Data.row5569.contains = (SortedNatTree.contains FastRows.row5569)
    exact FastRows.row5569_checked.symm
  rw [row]
  decide
theorem root178Step12 : scan edge (search edge 3) 4 [1087] 1087 root178Tail12 winner =
    scan edge (search edge 3) 4 [1087] 1087 root178Tail13 winner := by
  rw [root178Tail12, scan, if_neg (by decide), root178Candidates12_checked]
  change scan edge (search edge 3) 4 [1087] 1087 root178Tail13
    (if 3 ≤ root178Candidates12.length then search edge 3 [1087, 5569] root178Candidates12 6656 winner else winner) = _
  rw [if_neg (by decide)]
def root178Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root178Candidates13_checked : candidates edge 4 1087 6091 root178Tail14 winner.bound = root178Candidates13 := by
  unfold candidates
  have row : edge 6091 = (SortedNatTree.contains FastRows.row6091) := by
    change Data.row6091.contains = (SortedNatTree.contains FastRows.row6091)
    exact FastRows.row6091_checked.symm
  rw [row]
  decide
theorem root178Step13 : scan edge (search edge 3) 4 [1087] 1087 root178Tail13 winner =
    scan edge (search edge 3) 4 [1087] 1087 root178Tail14 winner := by
  rw [root178Tail13, scan, if_neg (by decide), root178Candidates13_checked]
  change scan edge (search edge 3) 4 [1087] 1087 root178Tail14
    (if 3 ≤ root178Candidates13.length then search edge 3 [1087, 6091] root178Candidates13 7178 winner else winner) = _
  rw [if_neg (by decide)]
theorem root178Scan14 : scan edge (search edge 3) 4 [1087] 1087 root178Tail14 winner = winner := by rw [root178Tail14, scan, if_pos (by decide)]
theorem root178Scan13 : scan edge (search edge 3) 4 [1087] 1087 root178Tail13 winner = winner :=
  (root178Step13).trans root178Scan14
theorem root178Scan12 : scan edge (search edge 3) 4 [1087] 1087 root178Tail12 winner = winner :=
  (root178Step12).trans root178Scan13
theorem root178Scan11 : scan edge (search edge 3) 4 [1087] 1087 root178Tail11 winner = winner :=
  (root178Step11).trans root178Scan12
theorem root178Scan10 : scan edge (search edge 3) 4 [1087] 1087 root178Tail10 winner = winner :=
  (root178Step10).trans root178Scan11
theorem root178Scan9 : scan edge (search edge 3) 4 [1087] 1087 root178Tail9 winner = winner :=
  (root178Step9).trans root178Scan10
theorem root178Scan8 : scan edge (search edge 3) 4 [1087] 1087 root178Tail8 winner = winner :=
  (root178Step8).trans root178Scan9
theorem root178Scan7 : scan edge (search edge 3) 4 [1087] 1087 root178Tail7 winner = winner :=
  (root178Step7).trans root178Scan8
theorem root178Scan6 : scan edge (search edge 3) 4 [1087] 1087 root178Tail6 winner = winner :=
  (root178Step6).trans root178Scan7
theorem root178Scan5 : scan edge (search edge 3) 4 [1087] 1087 root178Tail5 winner = winner :=
  (root178Step5).trans root178Scan6
theorem root178Scan4 : scan edge (search edge 3) 4 [1087] 1087 root178Tail4 winner = winner :=
  (root178Step4).trans root178Scan5
theorem root178Scan3 : scan edge (search edge 3) 4 [1087] 1087 root178Tail3 winner = winner :=
  (root178Step3).trans root178Scan4
theorem root178Scan2 : scan edge (search edge 3) 4 [1087] 1087 root178Tail2 winner = winner :=
  (root178Step2).trans root178Scan3
theorem root178Scan1 : scan edge (search edge 3) 4 [1087] 1087 root178Tail1 winner = winner :=
  (root178Step1).trans root178Scan2
theorem root178Scan0 : scan edge (search edge 3) 4 [1087] 1087 root178Tail0 winner = winner :=
  (root178Step0).trans root178Scan1
set_option maxRecDepth 4096 in
theorem root178Branch : search edge 4 [1087] root178Candidates 1087 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1087] 1087 root178Tail0 winner = winner
  exact root178Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
