import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root400Candidates : List Nat := Data.row2767
set_option maxRecDepth 8192 in
theorem root400Candidates_checked : candidates edge 5 0 2767 tail401 winner.bound = root400Candidates := by
  apply candidates_eq_row edge 5 0 2767 tail401 root400Candidates winner.bound tail401_ordered
  · decide
  · decide
  · funext q
    rfl
def root400Tail50 : List Nat := []
def root400Tail49 : List Nat := 23203 :: root400Tail50
def root400Tail48 : List Nat := 22189 :: root400Tail49
def root400Tail47 : List Nat := 21397 :: root400Tail48
def root400Tail46 : List Nat := 20551 :: root400Tail47
def root400Tail45 : List Nat := 20071 :: root400Tail46
def root400Tail44 : List Nat := 19543 :: root400Tail45
def root400Tail43 : List Nat := 18919 :: root400Tail44
def root400Tail42 : List Nat := 18433 :: root400Tail43
def root400Tail41 : List Nat := 18289 :: root400Tail42
def root400Tail40 : List Nat := 16477 :: root400Tail41
def root400Tail39 : List Nat := 16249 :: root400Tail40
def root400Tail38 : List Nat := 15679 :: root400Tail39
def root400Tail37 : List Nat := 15619 :: root400Tail38
def root400Tail36 : List Nat := 15121 :: root400Tail37
def root400Tail35 : List Nat := 14923 :: root400Tail36
def root400Tail34 : List Nat := 14551 :: root400Tail35
def root400Tail33 : List Nat := 14293 :: root400Tail34
def root400Tail32 : List Nat := 13687 :: root400Tail33
def root400Tail31 : List Nat := 13399 :: root400Tail32
def root400Tail30 : List Nat := 12781 :: root400Tail31
def root400Tail29 : List Nat := 11971 :: root400Tail30
def root400Tail28 : List Nat := 10789 :: root400Tail29
def root400Tail27 : List Nat := 10627 :: root400Tail28
def root400Tail26 : List Nat := 9721 :: root400Tail27
def root400Tail25 : List Nat := 9661 :: root400Tail26
def root400Tail24 : List Nat := 8527 :: root400Tail25
def root400Tail23 : List Nat := 8419 :: root400Tail24
def root400Tail22 : List Nat := 7933 :: root400Tail23
def root400Tail21 : List Nat := 7723 :: root400Tail22
def root400Tail20 : List Nat := 7621 :: root400Tail21
def root400Tail19 : List Nat := 7561 :: root400Tail20
def root400Tail18 : List Nat := 7537 :: root400Tail19
def root400Tail17 : List Nat := 7477 :: root400Tail18
def root400Tail16 : List Nat := 7459 :: root400Tail17
def root400Tail15 : List Nat := 6967 :: root400Tail16
def root400Tail14 : List Nat := 6679 :: root400Tail15
def root400Tail13 : List Nat := 6361 :: root400Tail14
def root400Tail12 : List Nat := 5737 :: root400Tail13
def root400Tail11 : List Nat := 5689 :: root400Tail12
def root400Tail10 : List Nat := 5119 :: root400Tail11
def root400Tail9 : List Nat := 5107 :: root400Tail10
def root400Tail8 : List Nat := 5059 :: root400Tail9
def root400Tail7 : List Nat := 4783 :: root400Tail8
def root400Tail6 : List Nat := 4513 :: root400Tail7
def root400Tail5 : List Nat := 4099 :: root400Tail6
def root400Tail4 : List Nat := 3967 :: root400Tail5
def root400Tail3 : List Nat := 3889 :: root400Tail4
def root400Tail2 : List Nat := 3733 :: root400Tail3
def root400Tail1 : List Nat := 3187 :: root400Tail2
def root400Tail0 : List Nat := 3109 :: root400Tail1
def root400Candidates0 : List Nat := [4099]
set_option maxRecDepth 8192 in
theorem root400Candidates0_checked : candidates edge 4 2767 3109 root400Tail1 winner.bound = root400Candidates0 := by
  unfold candidates
  have row : edge 3109 = (SortedNatTree.contains FastRows.row3109) := by
    change Data.row3109.contains = (SortedNatTree.contains FastRows.row3109)
    exact FastRows.row3109_checked.symm
  rw [row]
  decide
theorem root400Step0 : scan edge (search edge 3) 4 [2767] 2767 root400Tail0 winner =
    scan edge (search edge 3) 4 [2767] 2767 root400Tail1 winner := by
  rw [root400Tail0, scan, if_neg (by decide), root400Candidates0_checked]
  change scan edge (search edge 3) 4 [2767] 2767 root400Tail1
    (if 3 ≤ root400Candidates0.length then search edge 3 [2767, 3109] root400Candidates0 5876 winner else winner) = _
  rw [if_neg (by decide)]
def root400Candidates1 : List Nat := [7477, 8527, 19543, 20071]
set_option maxRecDepth 8192 in
theorem root400Candidates1_checked : candidates edge 4 2767 3187 root400Tail2 winner.bound = root400Candidates1 := by
  unfold candidates
  have row : edge 3187 = (SortedNatTree.contains FastRows.row3187) := by
    change Data.row3187.contains = (SortedNatTree.contains FastRows.row3187)
    exact FastRows.row3187_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root400Child1 : search edge 3 [2767, 3187] root400Candidates1 5954 winner = winner := by decide
theorem root400Step1 : scan edge (search edge 3) 4 [2767] 2767 root400Tail1 winner =
    scan edge (search edge 3) 4 [2767] 2767 root400Tail2 winner := by
  rw [root400Tail1, scan, if_neg (by decide), root400Candidates1_checked]
  change scan edge (search edge 3) 4 [2767] 2767 root400Tail2
    (if 3 ≤ root400Candidates1.length then search edge 3 [2767, 3187] root400Candidates1 5954 winner else winner) = _
  rw [if_pos (by decide), root400Child1]
def root400Candidates2 : List Nat := [10627, 11971, 13399]
set_option maxRecDepth 8192 in
theorem root400Candidates2_checked : candidates edge 4 2767 3733 root400Tail3 winner.bound = root400Candidates2 := by
  unfold candidates
  have row : edge 3733 = (SortedNatTree.contains FastRows.row3733) := by
    change Data.row3733.contains = (SortedNatTree.contains FastRows.row3733)
    exact FastRows.row3733_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root400Child2 : search edge 3 [2767, 3733] root400Candidates2 6500 winner = winner := by decide
theorem root400Step2 : scan edge (search edge 3) 4 [2767] 2767 root400Tail2 winner =
    scan edge (search edge 3) 4 [2767] 2767 root400Tail3 winner := by
  rw [root400Tail2, scan, if_neg (by decide), root400Candidates2_checked]
  change scan edge (search edge 3) 4 [2767] 2767 root400Tail3
    (if 3 ≤ root400Candidates2.length then search edge 3 [2767, 3733] root400Candidates2 6500 winner else winner) = _
  rw [if_pos (by decide), root400Child2]
def root400Candidates3 : List Nat := [7477]
set_option maxRecDepth 8192 in
theorem root400Candidates3_checked : candidates edge 4 2767 3889 root400Tail4 winner.bound = root400Candidates3 := by
  unfold candidates
  have row : edge 3889 = (SortedNatTree.contains FastRows.row3889) := by
    change Data.row3889.contains = (SortedNatTree.contains FastRows.row3889)
    exact FastRows.row3889_checked.symm
  rw [row]
  decide
theorem root400Step3 : scan edge (search edge 3) 4 [2767] 2767 root400Tail3 winner =
    scan edge (search edge 3) 4 [2767] 2767 root400Tail4 winner := by
  rw [root400Tail3, scan, if_neg (by decide), root400Candidates3_checked]
  change scan edge (search edge 3) 4 [2767] 2767 root400Tail4
    (if 3 ≤ root400Candidates3.length then search edge 3 [2767, 3889] root400Candidates3 6656 winner else winner) = _
  rw [if_neg (by decide)]
def root400Candidates4 : List Nat := [4783, 7933, 11971]
set_option maxRecDepth 8192 in
theorem root400Candidates4_checked : candidates edge 4 2767 3967 root400Tail5 winner.bound = root400Candidates4 := by
  unfold candidates
  have row : edge 3967 = (SortedNatTree.contains FastRows.row3967) := by
    change Data.row3967.contains = (SortedNatTree.contains FastRows.row3967)
    exact FastRows.row3967_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root400Child4 : search edge 3 [2767, 3967] root400Candidates4 6734 winner = winner := by decide
theorem root400Step4 : scan edge (search edge 3) 4 [2767] 2767 root400Tail4 winner =
    scan edge (search edge 3) 4 [2767] 2767 root400Tail5 winner := by
  rw [root400Tail4, scan, if_neg (by decide), root400Candidates4_checked]
  change scan edge (search edge 3) 4 [2767] 2767 root400Tail5
    (if 3 ≤ root400Candidates4.length then search edge 3 [2767, 3967] root400Candidates4 6734 winner else winner) = _
  rw [if_pos (by decide), root400Child4]
def root400Candidates5 : List Nat := [4783, 14551]
set_option maxRecDepth 8192 in
theorem root400Candidates5_checked : candidates edge 4 2767 4099 root400Tail6 winner.bound = root400Candidates5 := by
  unfold candidates
  have row : edge 4099 = (SortedNatTree.contains FastRows.row4099) := by
    change Data.row4099.contains = (SortedNatTree.contains FastRows.row4099)
    exact FastRows.row4099_checked.symm
  rw [row]
  decide
theorem root400Step5 : scan edge (search edge 3) 4 [2767] 2767 root400Tail5 winner =
    scan edge (search edge 3) 4 [2767] 2767 root400Tail6 winner := by
  rw [root400Tail5, scan, if_neg (by decide), root400Candidates5_checked]
  change scan edge (search edge 3) 4 [2767] 2767 root400Tail6
    (if 3 ≤ root400Candidates5.length then search edge 3 [2767, 4099] root400Candidates5 6866 winner else winner) = _
  rw [if_neg (by decide)]
def root400Candidates6 : List Nat := [7723]
set_option maxRecDepth 8192 in
theorem root400Candidates6_checked : candidates edge 4 2767 4513 root400Tail7 winner.bound = root400Candidates6 := by
  unfold candidates
  have row : edge 4513 = (SortedNatTree.contains FastRows.row4513) := by
    change Data.row4513.contains = (SortedNatTree.contains FastRows.row4513)
    exact FastRows.row4513_checked.symm
  rw [row]
  decide
theorem root400Step6 : scan edge (search edge 3) 4 [2767] 2767 root400Tail6 winner =
    scan edge (search edge 3) 4 [2767] 2767 root400Tail7 winner := by
  rw [root400Tail6, scan, if_neg (by decide), root400Candidates6_checked]
  change scan edge (search edge 3) 4 [2767] 2767 root400Tail7
    (if 3 ≤ root400Candidates6.length then search edge 3 [2767, 4513] root400Candidates6 7280 winner else winner) = _
  rw [if_neg (by decide)]
def root400Candidates7 : List Nat := [14293]
set_option maxRecDepth 8192 in
theorem root400Candidates7_checked : candidates edge 4 2767 4783 root400Tail8 winner.bound = root400Candidates7 := by
  unfold candidates
  have row : edge 4783 = (SortedNatTree.contains FastRows.row4783) := by
    change Data.row4783.contains = (SortedNatTree.contains FastRows.row4783)
    exact FastRows.row4783_checked.symm
  rw [row]
  decide
theorem root400Step7 : scan edge (search edge 3) 4 [2767] 2767 root400Tail7 winner =
    scan edge (search edge 3) 4 [2767] 2767 root400Tail8 winner := by
  rw [root400Tail7, scan, if_neg (by decide), root400Candidates7_checked]
  change scan edge (search edge 3) 4 [2767] 2767 root400Tail8
    (if 3 ≤ root400Candidates7.length then search edge 3 [2767, 4783] root400Candidates7 7550 winner else winner) = _
  rw [if_neg (by decide)]
def root400Candidates8 : List Nat := [10789, 14551]
set_option maxRecDepth 8192 in
theorem root400Candidates8_checked : candidates edge 4 2767 5059 root400Tail9 winner.bound = root400Candidates8 := by
  unfold candidates
  have row : edge 5059 = (SortedNatTree.contains FastRows.row5059) := by
    change Data.row5059.contains = (SortedNatTree.contains FastRows.row5059)
    exact FastRows.row5059_checked.symm
  rw [row]
  decide
theorem root400Step8 : scan edge (search edge 3) 4 [2767] 2767 root400Tail8 winner =
    scan edge (search edge 3) 4 [2767] 2767 root400Tail9 winner := by
  rw [root400Tail8, scan, if_neg (by decide), root400Candidates8_checked]
  change scan edge (search edge 3) 4 [2767] 2767 root400Tail9
    (if 3 ≤ root400Candidates8.length then search edge 3 [2767, 5059] root400Candidates8 7826 winner else winner) = _
  rw [if_neg (by decide)]
def root400Candidates9 : List Nat := [14551]
set_option maxRecDepth 8192 in
theorem root400Candidates9_checked : candidates edge 4 2767 5107 root400Tail10 winner.bound = root400Candidates9 := by
  unfold candidates
  have row : edge 5107 = (SortedNatTree.contains FastRows.row5107) := by
    change Data.row5107.contains = (SortedNatTree.contains FastRows.row5107)
    exact FastRows.row5107_checked.symm
  rw [row]
  decide
theorem root400Step9 : scan edge (search edge 3) 4 [2767] 2767 root400Tail9 winner =
    scan edge (search edge 3) 4 [2767] 2767 root400Tail10 winner := by
  rw [root400Tail9, scan, if_neg (by decide), root400Candidates9_checked]
  change scan edge (search edge 3) 4 [2767] 2767 root400Tail10
    (if 3 ≤ root400Candidates9.length then search edge 3 [2767, 5107] root400Candidates9 7874 winner else winner) = _
  rw [if_neg (by decide)]
def root400Candidates10 : List Nat := [7459, 7537, 7621]
set_option maxRecDepth 8192 in
theorem root400Candidates10_checked : candidates edge 4 2767 5119 root400Tail11 winner.bound = root400Candidates10 := by
  unfold candidates
  have row : edge 5119 = (SortedNatTree.contains FastRows.row5119) := by
    change Data.row5119.contains = (SortedNatTree.contains FastRows.row5119)
    exact FastRows.row5119_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root400Child10 : search edge 3 [2767, 5119] root400Candidates10 7886 winner = winner := by decide
theorem root400Step10 : scan edge (search edge 3) 4 [2767] 2767 root400Tail10 winner =
    scan edge (search edge 3) 4 [2767] 2767 root400Tail11 winner := by
  rw [root400Tail10, scan, if_neg (by decide), root400Candidates10_checked]
  change scan edge (search edge 3) 4 [2767] 2767 root400Tail11
    (if 3 ≤ root400Candidates10.length then search edge 3 [2767, 5119] root400Candidates10 7886 winner else winner) = _
  rw [if_pos (by decide), root400Child10]
def root400Candidates11 : List Nat := [8419, 15121]
set_option maxRecDepth 8192 in
theorem root400Candidates11_checked : candidates edge 4 2767 5689 root400Tail12 winner.bound = root400Candidates11 := by
  unfold candidates
  have row : edge 5689 = (SortedNatTree.contains FastRows.row5689) := by
    change Data.row5689.contains = (SortedNatTree.contains FastRows.row5689)
    exact FastRows.row5689_checked.symm
  rw [row]
  decide
theorem root400Step11 : scan edge (search edge 3) 4 [2767] 2767 root400Tail11 winner =
    scan edge (search edge 3) 4 [2767] 2767 root400Tail12 winner := by
  rw [root400Tail11, scan, if_neg (by decide), root400Candidates11_checked]
  change scan edge (search edge 3) 4 [2767] 2767 root400Tail12
    (if 3 ≤ root400Candidates11.length then search edge 3 [2767, 5689] root400Candidates11 8456 winner else winner) = _
  rw [if_neg (by decide)]
def root400Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root400Candidates12_checked : candidates edge 4 2767 5737 root400Tail13 winner.bound = root400Candidates12 := by
  unfold candidates
  have row : edge 5737 = (SortedNatTree.contains FastRows.row5737) := by
    change Data.row5737.contains = (SortedNatTree.contains FastRows.row5737)
    exact FastRows.row5737_checked.symm
  rw [row]
  decide
theorem root400Step12 : scan edge (search edge 3) 4 [2767] 2767 root400Tail12 winner =
    scan edge (search edge 3) 4 [2767] 2767 root400Tail13 winner := by
  rw [root400Tail12, scan, if_neg (by decide), root400Candidates12_checked]
  change scan edge (search edge 3) 4 [2767] 2767 root400Tail13
    (if 3 ≤ root400Candidates12.length then search edge 3 [2767, 5737] root400Candidates12 8504 winner else winner) = _
  rw [if_neg (by decide)]
theorem root400Scan13 : scan edge (search edge 3) 4 [2767] 2767 root400Tail13 winner = winner := by rw [root400Tail13, scan, if_pos (by decide)]
theorem root400Scan12 : scan edge (search edge 3) 4 [2767] 2767 root400Tail12 winner = winner :=
  (root400Step12).trans root400Scan13
theorem root400Scan11 : scan edge (search edge 3) 4 [2767] 2767 root400Tail11 winner = winner :=
  (root400Step11).trans root400Scan12
theorem root400Scan10 : scan edge (search edge 3) 4 [2767] 2767 root400Tail10 winner = winner :=
  (root400Step10).trans root400Scan11
theorem root400Scan9 : scan edge (search edge 3) 4 [2767] 2767 root400Tail9 winner = winner :=
  (root400Step9).trans root400Scan10
theorem root400Scan8 : scan edge (search edge 3) 4 [2767] 2767 root400Tail8 winner = winner :=
  (root400Step8).trans root400Scan9
theorem root400Scan7 : scan edge (search edge 3) 4 [2767] 2767 root400Tail7 winner = winner :=
  (root400Step7).trans root400Scan8
theorem root400Scan6 : scan edge (search edge 3) 4 [2767] 2767 root400Tail6 winner = winner :=
  (root400Step6).trans root400Scan7
theorem root400Scan5 : scan edge (search edge 3) 4 [2767] 2767 root400Tail5 winner = winner :=
  (root400Step5).trans root400Scan6
theorem root400Scan4 : scan edge (search edge 3) 4 [2767] 2767 root400Tail4 winner = winner :=
  (root400Step4).trans root400Scan5
theorem root400Scan3 : scan edge (search edge 3) 4 [2767] 2767 root400Tail3 winner = winner :=
  (root400Step3).trans root400Scan4
theorem root400Scan2 : scan edge (search edge 3) 4 [2767] 2767 root400Tail2 winner = winner :=
  (root400Step2).trans root400Scan3
theorem root400Scan1 : scan edge (search edge 3) 4 [2767] 2767 root400Tail1 winner = winner :=
  (root400Step1).trans root400Scan2
theorem root400Scan0 : scan edge (search edge 3) 4 [2767] 2767 root400Tail0 winner = winner :=
  (root400Step0).trans root400Scan1
set_option maxRecDepth 4096 in
theorem root400Branch : search edge 4 [2767] root400Candidates 2767 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2767] 2767 root400Tail0 winner = winner
  exact root400Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
