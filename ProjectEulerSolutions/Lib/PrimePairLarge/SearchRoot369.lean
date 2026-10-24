import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root369Candidates : List Nat := Data.row2543
set_option maxRecDepth 8192 in
theorem root369Candidates_checked : candidates edge 5 0 2543 tail370 winner.bound = root369Candidates := by
  apply candidates_eq_row edge 5 0 2543 tail370 root369Candidates winner.bound tail370_ordered
  · decide
  · decide
  · funext q
    rfl
def root369Tail51 : List Nat := []
def root369Tail50 : List Nat := 22787 :: root369Tail51
def root369Tail49 : List Nat := 22709 :: root369Tail50
def root369Tail48 : List Nat := 21713 :: root369Tail49
def root369Tail47 : List Nat := 19973 :: root369Tail48
def root369Tail46 : List Nat := 17891 :: root369Tail47
def root369Tail45 : List Nat := 17519 :: root369Tail46
def root369Tail44 : List Nat := 17291 :: root369Tail45
def root369Tail43 : List Nat := 17123 :: root369Tail44
def root369Tail42 : List Nat := 16301 :: root369Tail43
def root369Tail41 : List Nat := 16253 :: root369Tail42
def root369Tail40 : List Nat := 15329 :: root369Tail41
def root369Tail39 : List Nat := 15107 :: root369Tail40
def root369Tail38 : List Nat := 14897 :: root369Tail39
def root369Tail37 : List Nat := 14831 :: root369Tail38
def root369Tail36 : List Nat := 14537 :: root369Tail37
def root369Tail35 : List Nat := 14153 :: root369Tail36
def root369Tail34 : List Nat := 13259 :: root369Tail35
def root369Tail33 : List Nat := 12917 :: root369Tail34
def root369Tail32 : List Nat := 12893 :: root369Tail33
def root369Tail31 : List Nat := 11579 :: root369Tail32
def root369Tail30 : List Nat := 11261 :: root369Tail31
def root369Tail29 : List Nat := 10499 :: root369Tail30
def root369Tail28 : List Nat := 10487 :: root369Tail29
def root369Tail27 : List Nat := 10313 :: root369Tail28
def root369Tail26 : List Nat := 10253 :: root369Tail27
def root369Tail25 : List Nat := 10163 :: root369Tail26
def root369Tail24 : List Nat := 9851 :: root369Tail25
def root369Tail23 : List Nat := 9539 :: root369Tail24
def root369Tail22 : List Nat := 9521 :: root369Tail23
def root369Tail21 : List Nat := 9413 :: root369Tail22
def root369Tail20 : List Nat := 9059 :: root369Tail21
def root369Tail19 : List Nat := 7649 :: root369Tail20
def root369Tail18 : List Nat := 7541 :: root369Tail19
def root369Tail17 : List Nat := 7331 :: root369Tail18
def root369Tail16 : List Nat := 6911 :: root369Tail17
def root369Tail15 : List Nat := 6791 :: root369Tail16
def root369Tail14 : List Nat := 6011 :: root369Tail15
def root369Tail13 : List Nat := 5591 :: root369Tail14
def root369Tail12 : List Nat := 5483 :: root369Tail13
def root369Tail11 : List Nat := 5441 :: root369Tail12
def root369Tail10 : List Nat := 5237 :: root369Tail11
def root369Tail9 : List Nat := 5153 :: root369Tail10
def root369Tail8 : List Nat := 5147 :: root369Tail9
def root369Tail7 : List Nat := 5051 :: root369Tail8
def root369Tail6 : List Nat := 4523 :: root369Tail7
def root369Tail5 : List Nat := 4349 :: root369Tail6
def root369Tail4 : List Nat := 4337 :: root369Tail5
def root369Tail3 : List Nat := 3851 :: root369Tail4
def root369Tail2 : List Nat := 3539 :: root369Tail3
def root369Tail1 : List Nat := 3371 :: root369Tail2
def root369Tail0 : List Nat := 2609 :: root369Tail1
def root369Candidates0 : List Nat := [10313, 15107]
set_option maxRecDepth 8192 in
theorem root369Candidates0_checked : candidates edge 4 2543 2609 root369Tail1 winner.bound = root369Candidates0 := by
  unfold candidates
  have row : edge 2609 = (SortedNatTree.contains FastRows.row2609) := by
    change Data.row2609.contains = (SortedNatTree.contains FastRows.row2609)
    exact FastRows.row2609_checked.symm
  rw [row]
  decide
theorem root369Step0 : scan edge (search edge 3) 4 [2543] 2543 root369Tail0 winner =
    scan edge (search edge 3) 4 [2543] 2543 root369Tail1 winner := by
  rw [root369Tail0, scan, if_neg (by decide), root369Candidates0_checked]
  change scan edge (search edge 3) 4 [2543] 2543 root369Tail1
    (if 3 ≤ root369Candidates0.length then search edge 3 [2543, 2609] root369Candidates0 5152 winner else winner) = _
  rw [if_neg (by decide)]
def root369Candidates1 : List Nat := [5153]
set_option maxRecDepth 8192 in
theorem root369Candidates1_checked : candidates edge 4 2543 3371 root369Tail2 winner.bound = root369Candidates1 := by
  unfold candidates
  have row : edge 3371 = (SortedNatTree.contains FastRows.row3371) := by
    change Data.row3371.contains = (SortedNatTree.contains FastRows.row3371)
    exact FastRows.row3371_checked.symm
  rw [row]
  decide
theorem root369Step1 : scan edge (search edge 3) 4 [2543] 2543 root369Tail1 winner =
    scan edge (search edge 3) 4 [2543] 2543 root369Tail2 winner := by
  rw [root369Tail1, scan, if_neg (by decide), root369Candidates1_checked]
  change scan edge (search edge 3) 4 [2543] 2543 root369Tail2
    (if 3 ≤ root369Candidates1.length then search edge 3 [2543, 3371] root369Candidates1 5914 winner else winner) = _
  rw [if_neg (by decide)]
def root369Candidates2 : List Nat := [5441]
set_option maxRecDepth 8192 in
theorem root369Candidates2_checked : candidates edge 4 2543 3539 root369Tail3 winner.bound = root369Candidates2 := by
  unfold candidates
  have row : edge 3539 = (SortedNatTree.contains FastRows.row3539) := by
    change Data.row3539.contains = (SortedNatTree.contains FastRows.row3539)
    exact FastRows.row3539_checked.symm
  rw [row]
  decide
theorem root369Step2 : scan edge (search edge 3) 4 [2543] 2543 root369Tail2 winner =
    scan edge (search edge 3) 4 [2543] 2543 root369Tail3 winner := by
  rw [root369Tail2, scan, if_neg (by decide), root369Candidates2_checked]
  change scan edge (search edge 3) 4 [2543] 2543 root369Tail3
    (if 3 ≤ root369Candidates2.length then search edge 3 [2543, 3539] root369Candidates2 6082 winner else winner) = _
  rw [if_neg (by decide)]
def root369Candidates3 : List Nat := [9521, 17891]
set_option maxRecDepth 8192 in
theorem root369Candidates3_checked : candidates edge 4 2543 3851 root369Tail4 winner.bound = root369Candidates3 := by
  unfold candidates
  have row : edge 3851 = (SortedNatTree.contains FastRows.row3851) := by
    change Data.row3851.contains = (SortedNatTree.contains FastRows.row3851)
    exact FastRows.row3851_checked.symm
  rw [row]
  decide
theorem root369Step3 : scan edge (search edge 3) 4 [2543] 2543 root369Tail3 winner =
    scan edge (search edge 3) 4 [2543] 2543 root369Tail4 winner := by
  rw [root369Tail3, scan, if_neg (by decide), root369Candidates3_checked]
  change scan edge (search edge 3) 4 [2543] 2543 root369Tail4
    (if 3 ≤ root369Candidates3.length then search edge 3 [2543, 3851] root369Candidates3 6394 winner else winner) = _
  rw [if_neg (by decide)]
def root369Candidates4 : List Nat := [4523, 5237]
set_option maxRecDepth 8192 in
theorem root369Candidates4_checked : candidates edge 4 2543 4337 root369Tail5 winner.bound = root369Candidates4 := by
  unfold candidates
  have row : edge 4337 = (SortedNatTree.contains FastRows.row4337) := by
    change Data.row4337.contains = (SortedNatTree.contains FastRows.row4337)
    exact FastRows.row4337_checked.symm
  rw [row]
  decide
theorem root369Step4 : scan edge (search edge 3) 4 [2543] 2543 root369Tail4 winner =
    scan edge (search edge 3) 4 [2543] 2543 root369Tail5 winner := by
  rw [root369Tail4, scan, if_neg (by decide), root369Candidates4_checked]
  change scan edge (search edge 3) 4 [2543] 2543 root369Tail5
    (if 3 ≤ root369Candidates4.length then search edge 3 [2543, 4337] root369Candidates4 6880 winner else winner) = _
  rw [if_neg (by decide)]
def root369Candidates5 : List Nat := [5051, 5483, 10313, 12893, 16301]
set_option maxRecDepth 8192 in
theorem root369Candidates5_checked : candidates edge 4 2543 4349 root369Tail6 winner.bound = root369Candidates5 := by
  unfold candidates
  have row : edge 4349 = (SortedNatTree.contains FastRows.row4349) := by
    change Data.row4349.contains = (SortedNatTree.contains FastRows.row4349)
    exact FastRows.row4349_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root369Child5 : search edge 3 [2543, 4349] root369Candidates5 6892 winner = winner := by decide
theorem root369Step5 : scan edge (search edge 3) 4 [2543] 2543 root369Tail5 winner =
    scan edge (search edge 3) 4 [2543] 2543 root369Tail6 winner := by
  rw [root369Tail5, scan, if_neg (by decide), root369Candidates5_checked]
  change scan edge (search edge 3) 4 [2543] 2543 root369Tail6
    (if 3 ≤ root369Candidates5.length then search edge 3 [2543, 4349] root369Candidates5 6892 winner else winner) = _
  rw [if_pos (by decide), root369Child5]
def root369Candidates6 : List Nat := [5153, 6911, 17891]
set_option maxRecDepth 8192 in
theorem root369Candidates6_checked : candidates edge 4 2543 4523 root369Tail7 winner.bound = root369Candidates6 := by
  unfold candidates
  have row : edge 4523 = (SortedNatTree.contains FastRows.row4523) := by
    change Data.row4523.contains = (SortedNatTree.contains FastRows.row4523)
    exact FastRows.row4523_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root369Child6 : search edge 3 [2543, 4523] root369Candidates6 7066 winner = winner := by decide
theorem root369Step6 : scan edge (search edge 3) 4 [2543] 2543 root369Tail6 winner =
    scan edge (search edge 3) 4 [2543] 2543 root369Tail7 winner := by
  rw [root369Tail6, scan, if_neg (by decide), root369Candidates6_checked]
  change scan edge (search edge 3) 4 [2543] 2543 root369Tail7
    (if 3 ≤ root369Candidates6.length then search edge 3 [2543, 4523] root369Candidates6 7066 winner else winner) = _
  rw [if_pos (by decide), root369Child6]
def root369Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root369Candidates7_checked : candidates edge 4 2543 5051 root369Tail8 winner.bound = root369Candidates7 := by
  unfold candidates
  have row : edge 5051 = (SortedNatTree.contains FastRows.row5051) := by
    change Data.row5051.contains = (SortedNatTree.contains FastRows.row5051)
    exact FastRows.row5051_checked.symm
  rw [row]
  decide
theorem root369Step7 : scan edge (search edge 3) 4 [2543] 2543 root369Tail7 winner =
    scan edge (search edge 3) 4 [2543] 2543 root369Tail8 winner := by
  rw [root369Tail7, scan, if_neg (by decide), root369Candidates7_checked]
  change scan edge (search edge 3) 4 [2543] 2543 root369Tail8
    (if 3 ≤ root369Candidates7.length then search edge 3 [2543, 5051] root369Candidates7 7594 winner else winner) = _
  rw [if_neg (by decide)]
def root369Candidates8 : List Nat := [7331, 9851, 13259, 14537, 15329]
set_option maxRecDepth 8192 in
theorem root369Candidates8_checked : candidates edge 4 2543 5147 root369Tail9 winner.bound = root369Candidates8 := by
  unfold candidates
  have row : edge 5147 = (SortedNatTree.contains FastRows.row5147) := by
    change Data.row5147.contains = (SortedNatTree.contains FastRows.row5147)
    exact FastRows.row5147_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root369Child8 : search edge 3 [2543, 5147] root369Candidates8 7690 winner = winner := by decide
theorem root369Step8 : scan edge (search edge 3) 4 [2543] 2543 root369Tail8 winner =
    scan edge (search edge 3) 4 [2543] 2543 root369Tail9 winner := by
  rw [root369Tail8, scan, if_neg (by decide), root369Candidates8_checked]
  change scan edge (search edge 3) 4 [2543] 2543 root369Tail9
    (if 3 ≤ root369Candidates8.length then search edge 3 [2543, 5147] root369Candidates8 7690 winner else winner) = _
  rw [if_pos (by decide), root369Child8]
def root369Candidates9 : List Nat := [10487, 11579, 15107]
set_option maxRecDepth 8192 in
theorem root369Candidates9_checked : candidates edge 4 2543 5153 root369Tail10 winner.bound = root369Candidates9 := by
  unfold candidates
  have row : edge 5153 = (SortedNatTree.contains FastRows.row5153) := by
    change Data.row5153.contains = (SortedNatTree.contains FastRows.row5153)
    exact FastRows.row5153_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root369Child9 : search edge 3 [2543, 5153] root369Candidates9 7696 winner = winner := by decide
theorem root369Step9 : scan edge (search edge 3) 4 [2543] 2543 root369Tail9 winner =
    scan edge (search edge 3) 4 [2543] 2543 root369Tail10 winner := by
  rw [root369Tail9, scan, if_neg (by decide), root369Candidates9_checked]
  change scan edge (search edge 3) 4 [2543] 2543 root369Tail10
    (if 3 ≤ root369Candidates9.length then search edge 3 [2543, 5153] root369Candidates9 7696 winner else winner) = _
  rw [if_pos (by decide), root369Child9]
def root369Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root369Candidates10_checked : candidates edge 4 2543 5237 root369Tail11 winner.bound = root369Candidates10 := by
  unfold candidates
  have row : edge 5237 = (SortedNatTree.contains FastRows.row5237) := by
    change Data.row5237.contains = (SortedNatTree.contains FastRows.row5237)
    exact FastRows.row5237_checked.symm
  rw [row]
  decide
theorem root369Step10 : scan edge (search edge 3) 4 [2543] 2543 root369Tail10 winner =
    scan edge (search edge 3) 4 [2543] 2543 root369Tail11 winner := by
  rw [root369Tail10, scan, if_neg (by decide), root369Candidates10_checked]
  change scan edge (search edge 3) 4 [2543] 2543 root369Tail11
    (if 3 ≤ root369Candidates10.length then search edge 3 [2543, 5237] root369Candidates10 7780 winner else winner) = _
  rw [if_neg (by decide)]
def root369Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root369Candidates11_checked : candidates edge 4 2543 5441 root369Tail12 winner.bound = root369Candidates11 := by
  unfold candidates
  have row : edge 5441 = (SortedNatTree.contains FastRows.row5441) := by
    change Data.row5441.contains = (SortedNatTree.contains FastRows.row5441)
    exact FastRows.row5441_checked.symm
  rw [row]
  decide
theorem root369Step11 : scan edge (search edge 3) 4 [2543] 2543 root369Tail11 winner =
    scan edge (search edge 3) 4 [2543] 2543 root369Tail12 winner := by
  rw [root369Tail11, scan, if_neg (by decide), root369Candidates11_checked]
  change scan edge (search edge 3) 4 [2543] 2543 root369Tail12
    (if 3 ≤ root369Candidates11.length then search edge 3 [2543, 5441] root369Candidates11 7984 winner else winner) = _
  rw [if_neg (by decide)]
def root369Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root369Candidates12_checked : candidates edge 4 2543 5483 root369Tail13 winner.bound = root369Candidates12 := by
  unfold candidates
  have row : edge 5483 = (SortedNatTree.contains FastRows.row5483) := by
    change Data.row5483.contains = (SortedNatTree.contains FastRows.row5483)
    exact FastRows.row5483_checked.symm
  rw [row]
  decide
theorem root369Step12 : scan edge (search edge 3) 4 [2543] 2543 root369Tail12 winner =
    scan edge (search edge 3) 4 [2543] 2543 root369Tail13 winner := by
  rw [root369Tail12, scan, if_neg (by decide), root369Candidates12_checked]
  change scan edge (search edge 3) 4 [2543] 2543 root369Tail13
    (if 3 ≤ root369Candidates12.length then search edge 3 [2543, 5483] root369Candidates12 8026 winner else winner) = _
  rw [if_neg (by decide)]
def root369Candidates13 : List Nat := [13259]
set_option maxRecDepth 8192 in
theorem root369Candidates13_checked : candidates edge 4 2543 5591 root369Tail14 winner.bound = root369Candidates13 := by
  unfold candidates
  have row : edge 5591 = (SortedNatTree.contains FastRows.row5591) := by
    change Data.row5591.contains = (SortedNatTree.contains FastRows.row5591)
    exact FastRows.row5591_checked.symm
  rw [row]
  decide
theorem root369Step13 : scan edge (search edge 3) 4 [2543] 2543 root369Tail13 winner =
    scan edge (search edge 3) 4 [2543] 2543 root369Tail14 winner := by
  rw [root369Tail13, scan, if_neg (by decide), root369Candidates13_checked]
  change scan edge (search edge 3) 4 [2543] 2543 root369Tail14
    (if 3 ≤ root369Candidates13.length then search edge 3 [2543, 5591] root369Candidates13 8134 winner else winner) = _
  rw [if_neg (by decide)]
theorem root369Scan14 : scan edge (search edge 3) 4 [2543] 2543 root369Tail14 winner = winner := by rw [root369Tail14, scan, if_pos (by decide)]
theorem root369Scan13 : scan edge (search edge 3) 4 [2543] 2543 root369Tail13 winner = winner :=
  (root369Step13).trans root369Scan14
theorem root369Scan12 : scan edge (search edge 3) 4 [2543] 2543 root369Tail12 winner = winner :=
  (root369Step12).trans root369Scan13
theorem root369Scan11 : scan edge (search edge 3) 4 [2543] 2543 root369Tail11 winner = winner :=
  (root369Step11).trans root369Scan12
theorem root369Scan10 : scan edge (search edge 3) 4 [2543] 2543 root369Tail10 winner = winner :=
  (root369Step10).trans root369Scan11
theorem root369Scan9 : scan edge (search edge 3) 4 [2543] 2543 root369Tail9 winner = winner :=
  (root369Step9).trans root369Scan10
theorem root369Scan8 : scan edge (search edge 3) 4 [2543] 2543 root369Tail8 winner = winner :=
  (root369Step8).trans root369Scan9
theorem root369Scan7 : scan edge (search edge 3) 4 [2543] 2543 root369Tail7 winner = winner :=
  (root369Step7).trans root369Scan8
theorem root369Scan6 : scan edge (search edge 3) 4 [2543] 2543 root369Tail6 winner = winner :=
  (root369Step6).trans root369Scan7
theorem root369Scan5 : scan edge (search edge 3) 4 [2543] 2543 root369Tail5 winner = winner :=
  (root369Step5).trans root369Scan6
theorem root369Scan4 : scan edge (search edge 3) 4 [2543] 2543 root369Tail4 winner = winner :=
  (root369Step4).trans root369Scan5
theorem root369Scan3 : scan edge (search edge 3) 4 [2543] 2543 root369Tail3 winner = winner :=
  (root369Step3).trans root369Scan4
theorem root369Scan2 : scan edge (search edge 3) 4 [2543] 2543 root369Tail2 winner = winner :=
  (root369Step2).trans root369Scan3
theorem root369Scan1 : scan edge (search edge 3) 4 [2543] 2543 root369Tail1 winner = winner :=
  (root369Step1).trans root369Scan2
theorem root369Scan0 : scan edge (search edge 3) 4 [2543] 2543 root369Tail0 winner = winner :=
  (root369Step0).trans root369Scan1
set_option maxRecDepth 4096 in
theorem root369Branch : search edge 4 [2543] root369Candidates 2543 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2543] 2543 root369Tail0 winner = winner
  exact root369Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
