import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root217Candidates : List Nat := Data.row1373
set_option maxRecDepth 8192 in
theorem root217Candidates_checked : candidates edge 5 0 1373 tail218 winner.bound = root217Candidates := by
  apply candidates_eq_row edge 5 0 1373 tail218 root217Candidates winner.bound tail218_ordered
  · decide
  · decide
  · funext q
    rfl
def root217Tail51 : List Nat := []
def root217Tail50 : List Nat := 24371 :: root217Tail51
def root217Tail49 : List Nat := 24137 :: root217Tail50
def root217Tail48 : List Nat := 22943 :: root217Tail49
def root217Tail47 : List Nat := 22247 :: root217Tail48
def root217Tail46 : List Nat := 20903 :: root217Tail47
def root217Tail45 : List Nat := 20021 :: root217Tail46
def root217Tail44 : List Nat := 19139 :: root217Tail45
def root217Tail43 : List Nat := 19001 :: root217Tail44
def root217Tail42 : List Nat := 18593 :: root217Tail43
def root217Tail41 : List Nat := 18131 :: root217Tail42
def root217Tail40 : List Nat := 17903 :: root217Tail41
def root217Tail39 : List Nat := 17093 :: root217Tail40
def root217Tail38 : List Nat := 16001 :: root217Tail39
def root217Tail37 : List Nat := 15287 :: root217Tail38
def root217Tail36 : List Nat := 14939 :: root217Tail37
def root217Tail35 : List Nat := 14741 :: root217Tail36
def root217Tail34 : List Nat := 14591 :: root217Tail35
def root217Tail33 : List Nat := 14249 :: root217Tail34
def root217Tail32 : List Nat := 14081 :: root217Tail33
def root217Tail31 : List Nat := 12611 :: root217Tail32
def root217Tail30 : List Nat := 12491 :: root217Tail31
def root217Tail29 : List Nat := 11777 :: root217Tail30
def root217Tail28 : List Nat := 11411 :: root217Tail29
def root217Tail27 : List Nat := 11087 :: root217Tail28
def root217Tail26 : List Nat := 10559 :: root217Tail27
def root217Tail25 : List Nat := 10487 :: root217Tail26
def root217Tail24 : List Nat := 10301 :: root217Tail25
def root217Tail23 : List Nat := 10247 :: root217Tail24
def root217Tail22 : List Nat := 9941 :: root217Tail23
def root217Tail21 : List Nat := 8537 :: root217Tail22
def root217Tail20 : List Nat := 8423 :: root217Tail21
def root217Tail19 : List Nat := 7937 :: root217Tail20
def root217Tail18 : List Nat := 7907 :: root217Tail19
def root217Tail17 : List Nat := 7727 :: root217Tail18
def root217Tail16 : List Nat := 7541 :: root217Tail17
def root217Tail15 : List Nat := 6803 :: root217Tail16
def root217Tail14 : List Nat := 6491 :: root217Tail15
def root217Tail13 : List Nat := 6143 :: root217Tail14
def root217Tail12 : List Nat := 5333 :: root217Tail13
def root217Tail11 : List Nat := 4493 :: root217Tail12
def root217Tail10 : List Nat := 3539 :: root217Tail11
def root217Tail9 : List Nat := 3023 :: root217Tail10
def root217Tail8 : List Nat := 2591 :: root217Tail9
def root217Tail7 : List Nat := 2531 :: root217Tail8
def root217Tail6 : List Nat := 2423 :: root217Tail7
def root217Tail5 : List Nat := 1901 :: root217Tail6
def root217Tail4 : List Nat := 1637 :: root217Tail5
def root217Tail3 : List Nat := 1511 :: root217Tail4
def root217Tail2 : List Nat := 1499 :: root217Tail3
def root217Tail1 : List Nat := 1451 :: root217Tail2
def root217Tail0 : List Nat := 1439 :: root217Tail1
def root217Candidates0 : List Nat := [10301]
set_option maxRecDepth 8192 in
theorem root217Candidates0_checked : candidates edge 4 1373 1439 root217Tail1 winner.bound = root217Candidates0 := by
  unfold candidates
  have row : edge 1439 = (SortedNatTree.contains FastRows.row1439) := by
    change Data.row1439.contains = (SortedNatTree.contains FastRows.row1439)
    exact FastRows.row1439_checked.symm
  rw [row]
  decide
theorem root217Step0 : scan edge (search edge 3) 4 [1373] 1373 root217Tail0 winner =
    scan edge (search edge 3) 4 [1373] 1373 root217Tail1 winner := by
  rw [root217Tail0, scan, if_neg (by decide), root217Candidates0_checked]
  change scan edge (search edge 3) 4 [1373] 1373 root217Tail1
    (if 3 ≤ root217Candidates0.length then search edge 3 [1373, 1439] root217Candidates0 2812 winner else winner) = _
  rw [if_neg (by decide)]
def root217Candidates1 : List Nat := [7907, 11411]
set_option maxRecDepth 8192 in
theorem root217Candidates1_checked : candidates edge 4 1373 1451 root217Tail2 winner.bound = root217Candidates1 := by
  unfold candidates
  have row : edge 1451 = (SortedNatTree.contains FastRows.row1451) := by
    change Data.row1451.contains = (SortedNatTree.contains FastRows.row1451)
    exact FastRows.row1451_checked.symm
  rw [row]
  decide
theorem root217Step1 : scan edge (search edge 3) 4 [1373] 1373 root217Tail1 winner =
    scan edge (search edge 3) 4 [1373] 1373 root217Tail2 winner := by
  rw [root217Tail1, scan, if_neg (by decide), root217Candidates1_checked]
  change scan edge (search edge 3) 4 [1373] 1373 root217Tail2
    (if 3 ≤ root217Candidates1.length then search edge 3 [1373, 1451] root217Candidates1 2824 winner else winner) = _
  rw [if_neg (by decide)]
def root217Candidates2 : List Nat := [10487, 17903]
set_option maxRecDepth 8192 in
theorem root217Candidates2_checked : candidates edge 4 1373 1499 root217Tail3 winner.bound = root217Candidates2 := by
  unfold candidates
  have row : edge 1499 = (SortedNatTree.contains FastRows.row1499) := by
    change Data.row1499.contains = (SortedNatTree.contains FastRows.row1499)
    exact FastRows.row1499_checked.symm
  rw [row]
  decide
theorem root217Step2 : scan edge (search edge 3) 4 [1373] 1373 root217Tail2 winner =
    scan edge (search edge 3) 4 [1373] 1373 root217Tail3 winner := by
  rw [root217Tail2, scan, if_neg (by decide), root217Candidates2_checked]
  change scan edge (search edge 3) 4 [1373] 1373 root217Tail3
    (if 3 ≤ root217Candidates2.length then search edge 3 [1373, 1499] root217Candidates2 2872 winner else winner) = _
  rw [if_neg (by decide)]
def root217Candidates3 : List Nat := [10559, 14591, 17903]
set_option maxRecDepth 8192 in
theorem root217Candidates3_checked : candidates edge 4 1373 1511 root217Tail4 winner.bound = root217Candidates3 := by
  unfold candidates
  have row : edge 1511 = (SortedNatTree.contains FastRows.row1511) := by
    change Data.row1511.contains = (SortedNatTree.contains FastRows.row1511)
    exact FastRows.row1511_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root217Child3 : search edge 3 [1373, 1511] root217Candidates3 2884 winner = winner := by decide
theorem root217Step3 : scan edge (search edge 3) 4 [1373] 1373 root217Tail3 winner =
    scan edge (search edge 3) 4 [1373] 1373 root217Tail4 winner := by
  rw [root217Tail3, scan, if_neg (by decide), root217Candidates3_checked]
  change scan edge (search edge 3) 4 [1373] 1373 root217Tail4
    (if 3 ≤ root217Candidates3.length then search edge 3 [1373, 1511] root217Candidates3 2884 winner else winner) = _
  rw [if_pos (by decide), root217Child3]
def root217Candidates4 : List Nat := [11777, 14741]
set_option maxRecDepth 8192 in
theorem root217Candidates4_checked : candidates edge 4 1373 1637 root217Tail5 winner.bound = root217Candidates4 := by
  unfold candidates
  have row : edge 1637 = (SortedNatTree.contains FastRows.row1637) := by
    change Data.row1637.contains = (SortedNatTree.contains FastRows.row1637)
    exact FastRows.row1637_checked.symm
  rw [row]
  decide
theorem root217Step4 : scan edge (search edge 3) 4 [1373] 1373 root217Tail4 winner =
    scan edge (search edge 3) 4 [1373] 1373 root217Tail5 winner := by
  rw [root217Tail4, scan, if_neg (by decide), root217Candidates4_checked]
  change scan edge (search edge 3) 4 [1373] 1373 root217Tail5
    (if 3 ≤ root217Candidates4.length then search edge 3 [1373, 1637] root217Candidates4 3010 winner else winner) = _
  rw [if_neg (by decide)]
def root217Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root217Candidates5_checked : candidates edge 4 1373 1901 root217Tail6 winner.bound = root217Candidates5 := by
  unfold candidates
  have row : edge 1901 = (SortedNatTree.contains FastRows.row1901) := by
    change Data.row1901.contains = (SortedNatTree.contains FastRows.row1901)
    exact FastRows.row1901_checked.symm
  rw [row]
  decide
theorem root217Step5 : scan edge (search edge 3) 4 [1373] 1373 root217Tail5 winner =
    scan edge (search edge 3) 4 [1373] 1373 root217Tail6 winner := by
  rw [root217Tail5, scan, if_neg (by decide), root217Candidates5_checked]
  change scan edge (search edge 3) 4 [1373] 1373 root217Tail6
    (if 3 ≤ root217Candidates5.length then search edge 3 [1373, 1901] root217Candidates5 3274 winner else winner) = _
  rw [if_neg (by decide)]
def root217Candidates6 : List Nat := [7727]
set_option maxRecDepth 8192 in
theorem root217Candidates6_checked : candidates edge 4 1373 2423 root217Tail7 winner.bound = root217Candidates6 := by
  unfold candidates
  have row : edge 2423 = (SortedNatTree.contains FastRows.row2423) := by
    change Data.row2423.contains = (SortedNatTree.contains FastRows.row2423)
    exact FastRows.row2423_checked.symm
  rw [row]
  decide
theorem root217Step6 : scan edge (search edge 3) 4 [1373] 1373 root217Tail6 winner =
    scan edge (search edge 3) 4 [1373] 1373 root217Tail7 winner := by
  rw [root217Tail6, scan, if_neg (by decide), root217Candidates6_checked]
  change scan edge (search edge 3) 4 [1373] 1373 root217Tail7
    (if 3 ≤ root217Candidates6.length then search edge 3 [1373, 2423] root217Candidates6 3796 winner else winner) = _
  rw [if_neg (by decide)]
def root217Candidates7 : List Nat := [3539, 12611, 14249]
set_option maxRecDepth 8192 in
theorem root217Candidates7_checked : candidates edge 4 1373 2531 root217Tail8 winner.bound = root217Candidates7 := by
  unfold candidates
  have row : edge 2531 = (SortedNatTree.contains FastRows.row2531) := by
    change Data.row2531.contains = (SortedNatTree.contains FastRows.row2531)
    exact FastRows.row2531_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root217Child7 : search edge 3 [1373, 2531] root217Candidates7 3904 winner = winner := by decide
theorem root217Step7 : scan edge (search edge 3) 4 [1373] 1373 root217Tail7 winner =
    scan edge (search edge 3) 4 [1373] 1373 root217Tail8 winner := by
  rw [root217Tail7, scan, if_neg (by decide), root217Candidates7_checked]
  change scan edge (search edge 3) 4 [1373] 1373 root217Tail8
    (if 3 ≤ root217Candidates7.length then search edge 3 [1373, 2531] root217Candidates7 3904 winner else winner) = _
  rw [if_pos (by decide), root217Child7]
def root217Candidates8 : List Nat := [7937, 8423, 9941, 14249]
set_option maxRecDepth 8192 in
theorem root217Candidates8_checked : candidates edge 4 1373 2591 root217Tail9 winner.bound = root217Candidates8 := by
  unfold candidates
  have row : edge 2591 = (SortedNatTree.contains FastRows.row2591) := by
    change Data.row2591.contains = (SortedNatTree.contains FastRows.row2591)
    exact FastRows.row2591_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root217Child8 : search edge 3 [1373, 2591] root217Candidates8 3964 winner = winner := by decide
theorem root217Step8 : scan edge (search edge 3) 4 [1373] 1373 root217Tail8 winner =
    scan edge (search edge 3) 4 [1373] 1373 root217Tail9 winner := by
  rw [root217Tail8, scan, if_neg (by decide), root217Candidates8_checked]
  change scan edge (search edge 3) 4 [1373] 1373 root217Tail9
    (if 3 ≤ root217Candidates8.length then search edge 3 [1373, 2591] root217Candidates8 3964 winner else winner) = _
  rw [if_pos (by decide), root217Child8]
def root217Candidates9 : List Nat := [18131, 19001]
set_option maxRecDepth 8192 in
theorem root217Candidates9_checked : candidates edge 4 1373 3023 root217Tail10 winner.bound = root217Candidates9 := by
  unfold candidates
  have row : edge 3023 = (SortedNatTree.contains FastRows.row3023) := by
    change Data.row3023.contains = (SortedNatTree.contains FastRows.row3023)
    exact FastRows.row3023_checked.symm
  rw [row]
  decide
theorem root217Step9 : scan edge (search edge 3) 4 [1373] 1373 root217Tail9 winner =
    scan edge (search edge 3) 4 [1373] 1373 root217Tail10 winner := by
  rw [root217Tail9, scan, if_neg (by decide), root217Candidates9_checked]
  change scan edge (search edge 3) 4 [1373] 1373 root217Tail10
    (if 3 ≤ root217Candidates9.length then search edge 3 [1373, 3023] root217Candidates9 4396 winner else winner) = _
  rw [if_neg (by decide)]
def root217Candidates10 : List Nat := [17903]
set_option maxRecDepth 8192 in
theorem root217Candidates10_checked : candidates edge 4 1373 3539 root217Tail11 winner.bound = root217Candidates10 := by
  unfold candidates
  have row : edge 3539 = (SortedNatTree.contains FastRows.row3539) := by
    change Data.row3539.contains = (SortedNatTree.contains FastRows.row3539)
    exact FastRows.row3539_checked.symm
  rw [row]
  decide
theorem root217Step10 : scan edge (search edge 3) 4 [1373] 1373 root217Tail10 winner =
    scan edge (search edge 3) 4 [1373] 1373 root217Tail11 winner := by
  rw [root217Tail10, scan, if_neg (by decide), root217Candidates10_checked]
  change scan edge (search edge 3) 4 [1373] 1373 root217Tail11
    (if 3 ≤ root217Candidates10.length then search edge 3 [1373, 3539] root217Candidates10 4912 winner else winner) = _
  rw [if_neg (by decide)]
def root217Candidates11 : List Nat := [11777, 14741]
set_option maxRecDepth 8192 in
theorem root217Candidates11_checked : candidates edge 4 1373 4493 root217Tail12 winner.bound = root217Candidates11 := by
  unfold candidates
  have row : edge 4493 = (SortedNatTree.contains FastRows.row4493) := by
    change Data.row4493.contains = (SortedNatTree.contains FastRows.row4493)
    exact FastRows.row4493_checked.symm
  rw [row]
  decide
theorem root217Step11 : scan edge (search edge 3) 4 [1373] 1373 root217Tail11 winner =
    scan edge (search edge 3) 4 [1373] 1373 root217Tail12 winner := by
  rw [root217Tail11, scan, if_neg (by decide), root217Candidates11_checked]
  change scan edge (search edge 3) 4 [1373] 1373 root217Tail12
    (if 3 ≤ root217Candidates11.length then search edge 3 [1373, 4493] root217Candidates11 5866 winner else winner) = _
  rw [if_neg (by decide)]
def root217Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root217Candidates12_checked : candidates edge 4 1373 5333 root217Tail13 winner.bound = root217Candidates12 := by
  unfold candidates
  have row : edge 5333 = (SortedNatTree.contains FastRows.row5333) := by
    change Data.row5333.contains = (SortedNatTree.contains FastRows.row5333)
    exact FastRows.row5333_checked.symm
  rw [row]
  decide
theorem root217Step12 : scan edge (search edge 3) 4 [1373] 1373 root217Tail12 winner =
    scan edge (search edge 3) 4 [1373] 1373 root217Tail13 winner := by
  rw [root217Tail12, scan, if_neg (by decide), root217Candidates12_checked]
  change scan edge (search edge 3) 4 [1373] 1373 root217Tail13
    (if 3 ≤ root217Candidates12.length then search edge 3 [1373, 5333] root217Candidates12 6706 winner else winner) = _
  rw [if_neg (by decide)]
def root217Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root217Candidates13_checked : candidates edge 4 1373 6143 root217Tail14 winner.bound = root217Candidates13 := by
  unfold candidates
  have row : edge 6143 = (SortedNatTree.contains FastRows.row6143) := by
    change Data.row6143.contains = (SortedNatTree.contains FastRows.row6143)
    exact FastRows.row6143_checked.symm
  rw [row]
  decide
theorem root217Step13 : scan edge (search edge 3) 4 [1373] 1373 root217Tail13 winner =
    scan edge (search edge 3) 4 [1373] 1373 root217Tail14 winner := by
  rw [root217Tail13, scan, if_neg (by decide), root217Candidates13_checked]
  change scan edge (search edge 3) 4 [1373] 1373 root217Tail14
    (if 3 ≤ root217Candidates13.length then search edge 3 [1373, 6143] root217Candidates13 7516 winner else winner) = _
  rw [if_neg (by decide)]
theorem root217Scan14 : scan edge (search edge 3) 4 [1373] 1373 root217Tail14 winner = winner := by rw [root217Tail14, scan, if_pos (by decide)]
theorem root217Scan13 : scan edge (search edge 3) 4 [1373] 1373 root217Tail13 winner = winner :=
  (root217Step13).trans root217Scan14
theorem root217Scan12 : scan edge (search edge 3) 4 [1373] 1373 root217Tail12 winner = winner :=
  (root217Step12).trans root217Scan13
theorem root217Scan11 : scan edge (search edge 3) 4 [1373] 1373 root217Tail11 winner = winner :=
  (root217Step11).trans root217Scan12
theorem root217Scan10 : scan edge (search edge 3) 4 [1373] 1373 root217Tail10 winner = winner :=
  (root217Step10).trans root217Scan11
theorem root217Scan9 : scan edge (search edge 3) 4 [1373] 1373 root217Tail9 winner = winner :=
  (root217Step9).trans root217Scan10
theorem root217Scan8 : scan edge (search edge 3) 4 [1373] 1373 root217Tail8 winner = winner :=
  (root217Step8).trans root217Scan9
theorem root217Scan7 : scan edge (search edge 3) 4 [1373] 1373 root217Tail7 winner = winner :=
  (root217Step7).trans root217Scan8
theorem root217Scan6 : scan edge (search edge 3) 4 [1373] 1373 root217Tail6 winner = winner :=
  (root217Step6).trans root217Scan7
theorem root217Scan5 : scan edge (search edge 3) 4 [1373] 1373 root217Tail5 winner = winner :=
  (root217Step5).trans root217Scan6
theorem root217Scan4 : scan edge (search edge 3) 4 [1373] 1373 root217Tail4 winner = winner :=
  (root217Step4).trans root217Scan5
theorem root217Scan3 : scan edge (search edge 3) 4 [1373] 1373 root217Tail3 winner = winner :=
  (root217Step3).trans root217Scan4
theorem root217Scan2 : scan edge (search edge 3) 4 [1373] 1373 root217Tail2 winner = winner :=
  (root217Step2).trans root217Scan3
theorem root217Scan1 : scan edge (search edge 3) 4 [1373] 1373 root217Tail1 winner = winner :=
  (root217Step1).trans root217Scan2
theorem root217Scan0 : scan edge (search edge 3) 4 [1373] 1373 root217Tail0 winner = winner :=
  (root217Step0).trans root217Scan1
set_option maxRecDepth 4096 in
theorem root217Branch : search edge 4 [1373] root217Candidates 1373 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1373] 1373 root217Tail0 winner = winner
  exact root217Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
