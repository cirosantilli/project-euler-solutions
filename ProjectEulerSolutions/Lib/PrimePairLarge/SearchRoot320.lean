import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root320Candidates : List Nat := Data.row2141
set_option maxRecDepth 8192 in
theorem root320Candidates_checked : candidates edge 5 0 2141 tail321 winner.bound = root320Candidates := by
  apply candidates_eq_row edge 5 0 2141 tail321 root320Candidates winner.bound tail321_ordered
  · decide
  · decide
  · funext q
    rfl
def root320Tail45 : List Nat := []
def root320Tail44 : List Nat := 23549 :: root320Tail45
def root320Tail43 : List Nat := 22751 :: root320Tail44
def root320Tail42 : List Nat := 21893 :: root320Tail43
def root320Tail41 : List Nat := 21269 :: root320Tail42
def root320Tail40 : List Nat := 20903 :: root320Tail41
def root320Tail39 : List Nat := 20789 :: root320Tail40
def root320Tail38 : List Nat := 20261 :: root320Tail39
def root320Tail37 : List Nat := 19421 :: root320Tail38
def root320Tail36 : List Nat := 19013 :: root320Tail37
def root320Tail35 : List Nat := 18413 :: root320Tail36
def root320Tail34 : List Nat := 17627 :: root320Tail35
def root320Tail33 : List Nat := 17207 :: root320Tail34
def root320Tail32 : List Nat := 16127 :: root320Tail33
def root320Tail31 : List Nat := 15641 :: root320Tail32
def root320Tail30 : List Nat := 14969 :: root320Tail31
def root320Tail29 : List Nat := 14951 :: root320Tail30
def root320Tail28 : List Nat := 13901 :: root320Tail29
def root320Tail27 : List Nat := 12281 :: root320Tail28
def root320Tail26 : List Nat := 11903 :: root320Tail27
def root320Tail25 : List Nat := 11813 :: root320Tail26
def root320Tail24 : List Nat := 11351 :: root320Tail25
def root320Tail23 : List Nat := 10139 :: root320Tail24
def root320Tail22 : List Nat := 9623 :: root320Tail23
def root320Tail21 : List Nat := 9551 :: root320Tail22
def root320Tail20 : List Nat := 8951 :: root320Tail21
def root320Tail19 : List Nat := 8693 :: root320Tail20
def root320Tail18 : List Nat := 7727 :: root320Tail19
def root320Tail17 : List Nat := 7691 :: root320Tail18
def root320Tail16 : List Nat := 7547 :: root320Tail17
def root320Tail15 : List Nat := 7349 :: root320Tail16
def root320Tail14 : List Nat := 6737 :: root320Tail15
def root320Tail13 : List Nat := 6053 :: root320Tail14
def root320Tail12 : List Nat := 5519 :: root320Tail13
def root320Tail11 : List Nat := 5153 :: root320Tail12
def root320Tail10 : List Nat := 4649 :: root320Tail11
def root320Tail9 : List Nat := 4091 :: root320Tail10
def root320Tail8 : List Nat := 4049 :: root320Tail9
def root320Tail7 : List Nat := 3947 :: root320Tail8
def root320Tail6 : List Nat := 3881 :: root320Tail7
def root320Tail5 : List Nat := 3191 :: root320Tail6
def root320Tail4 : List Nat := 3167 :: root320Tail5
def root320Tail3 : List Nat := 2663 :: root320Tail4
def root320Tail2 : List Nat := 2621 :: root320Tail3
def root320Tail1 : List Nat := 2309 :: root320Tail2
def root320Tail0 : List Nat := 2267 :: root320Tail1
def root320Candidates0 : List Nat := [6053, 9551]
set_option maxRecDepth 8192 in
theorem root320Candidates0_checked : candidates edge 4 2141 2267 root320Tail1 winner.bound = root320Candidates0 := by
  unfold candidates
  have row : edge 2267 = (SortedNatTree.contains FastRows.row2267) := by
    change Data.row2267.contains = (SortedNatTree.contains FastRows.row2267)
    exact FastRows.row2267_checked.symm
  rw [row]
  decide
theorem root320Step0 : scan edge (search edge 3) 4 [2141] 2141 root320Tail0 winner =
    scan edge (search edge 3) 4 [2141] 2141 root320Tail1 winner := by
  rw [root320Tail0, scan, if_neg (by decide), root320Candidates0_checked]
  change scan edge (search edge 3) 4 [2141] 2141 root320Tail1
    (if 3 ≤ root320Candidates0.length then search edge 3 [2141, 2267] root320Candidates0 4408 winner else winner) = _
  rw [if_neg (by decide)]
def root320Candidates1 : List Nat := [7691]
set_option maxRecDepth 8192 in
theorem root320Candidates1_checked : candidates edge 4 2141 2309 root320Tail2 winner.bound = root320Candidates1 := by
  unfold candidates
  have row : edge 2309 = (SortedNatTree.contains FastRows.row2309) := by
    change Data.row2309.contains = (SortedNatTree.contains FastRows.row2309)
    exact FastRows.row2309_checked.symm
  rw [row]
  decide
theorem root320Step1 : scan edge (search edge 3) 4 [2141] 2141 root320Tail1 winner =
    scan edge (search edge 3) 4 [2141] 2141 root320Tail2 winner := by
  rw [root320Tail1, scan, if_neg (by decide), root320Candidates1_checked]
  change scan edge (search edge 3) 4 [2141] 2141 root320Tail2
    (if 3 ≤ root320Candidates1.length then search edge 3 [2141, 2309] root320Candidates1 4450 winner else winner) = _
  rw [if_neg (by decide)]
def root320Candidates2 : List Nat := [6053, 11813, 18413]
set_option maxRecDepth 8192 in
theorem root320Candidates2_checked : candidates edge 4 2141 2621 root320Tail3 winner.bound = root320Candidates2 := by
  unfold candidates
  have row : edge 2621 = (SortedNatTree.contains FastRows.row2621) := by
    change Data.row2621.contains = (SortedNatTree.contains FastRows.row2621)
    exact FastRows.row2621_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root320Child2 : search edge 3 [2141, 2621] root320Candidates2 4762 winner = winner := by decide
theorem root320Step2 : scan edge (search edge 3) 4 [2141] 2141 root320Tail2 winner =
    scan edge (search edge 3) 4 [2141] 2141 root320Tail3 winner := by
  rw [root320Tail2, scan, if_neg (by decide), root320Candidates2_checked]
  change scan edge (search edge 3) 4 [2141] 2141 root320Tail3
    (if 3 ≤ root320Candidates2.length then search edge 3 [2141, 2621] root320Candidates2 4762 winner else winner) = _
  rw [if_pos (by decide), root320Child2]
def root320Candidates3 : List Nat := [19421]
set_option maxRecDepth 8192 in
theorem root320Candidates3_checked : candidates edge 4 2141 2663 root320Tail4 winner.bound = root320Candidates3 := by
  unfold candidates
  have row : edge 2663 = (SortedNatTree.contains FastRows.row2663) := by
    change Data.row2663.contains = (SortedNatTree.contains FastRows.row2663)
    exact FastRows.row2663_checked.symm
  rw [row]
  decide
theorem root320Step3 : scan edge (search edge 3) 4 [2141] 2141 root320Tail3 winner =
    scan edge (search edge 3) 4 [2141] 2141 root320Tail4 winner := by
  rw [root320Tail3, scan, if_neg (by decide), root320Candidates3_checked]
  change scan edge (search edge 3) 4 [2141] 2141 root320Tail4
    (if 3 ≤ root320Candidates3.length then search edge 3 [2141, 2663] root320Candidates3 4804 winner else winner) = _
  rw [if_neg (by decide)]
def root320Candidates4 : List Nat := [8951, 19421]
set_option maxRecDepth 8192 in
theorem root320Candidates4_checked : candidates edge 4 2141 3167 root320Tail5 winner.bound = root320Candidates4 := by
  unfold candidates
  have row : edge 3167 = (SortedNatTree.contains FastRows.row3167) := by
    change Data.row3167.contains = (SortedNatTree.contains FastRows.row3167)
    exact FastRows.row3167_checked.symm
  rw [row]
  decide
theorem root320Step4 : scan edge (search edge 3) 4 [2141] 2141 root320Tail4 winner =
    scan edge (search edge 3) 4 [2141] 2141 root320Tail5 winner := by
  rw [root320Tail4, scan, if_neg (by decide), root320Candidates4_checked]
  change scan edge (search edge 3) 4 [2141] 2141 root320Tail5
    (if 3 ≤ root320Candidates4.length then search edge 3 [2141, 3167] root320Candidates4 5308 winner else winner) = _
  rw [if_neg (by decide)]
def root320Candidates5 : List Nat := [7547]
set_option maxRecDepth 8192 in
theorem root320Candidates5_checked : candidates edge 4 2141 3191 root320Tail6 winner.bound = root320Candidates5 := by
  unfold candidates
  have row : edge 3191 = (SortedNatTree.contains FastRows.row3191) := by
    change Data.row3191.contains = (SortedNatTree.contains FastRows.row3191)
    exact FastRows.row3191_checked.symm
  rw [row]
  decide
theorem root320Step5 : scan edge (search edge 3) 4 [2141] 2141 root320Tail5 winner =
    scan edge (search edge 3) 4 [2141] 2141 root320Tail6 winner := by
  rw [root320Tail5, scan, if_neg (by decide), root320Candidates5_checked]
  change scan edge (search edge 3) 4 [2141] 2141 root320Tail6
    (if 3 ≤ root320Candidates5.length then search edge 3 [2141, 3191] root320Candidates5 5332 winner else winner) = _
  rw [if_neg (by decide)]
def root320Candidates6 : List Nat := [5519]
set_option maxRecDepth 8192 in
theorem root320Candidates6_checked : candidates edge 4 2141 3881 root320Tail7 winner.bound = root320Candidates6 := by
  unfold candidates
  have row : edge 3881 = (SortedNatTree.contains FastRows.row3881) := by
    change Data.row3881.contains = (SortedNatTree.contains FastRows.row3881)
    exact FastRows.row3881_checked.symm
  rw [row]
  decide
theorem root320Step6 : scan edge (search edge 3) 4 [2141] 2141 root320Tail6 winner =
    scan edge (search edge 3) 4 [2141] 2141 root320Tail7 winner := by
  rw [root320Tail6, scan, if_neg (by decide), root320Candidates6_checked]
  change scan edge (search edge 3) 4 [2141] 2141 root320Tail7
    (if 3 ≤ root320Candidates6.length then search edge 3 [2141, 3881] root320Candidates6 6022 winner else winner) = _
  rw [if_neg (by decide)]
def root320Candidates7 : List Nat := [11813, 17627]
set_option maxRecDepth 8192 in
theorem root320Candidates7_checked : candidates edge 4 2141 3947 root320Tail8 winner.bound = root320Candidates7 := by
  unfold candidates
  have row : edge 3947 = (SortedNatTree.contains FastRows.row3947) := by
    change Data.row3947.contains = (SortedNatTree.contains FastRows.row3947)
    exact FastRows.row3947_checked.symm
  rw [row]
  decide
theorem root320Step7 : scan edge (search edge 3) 4 [2141] 2141 root320Tail7 winner =
    scan edge (search edge 3) 4 [2141] 2141 root320Tail8 winner := by
  rw [root320Tail7, scan, if_neg (by decide), root320Candidates7_checked]
  change scan edge (search edge 3) 4 [2141] 2141 root320Tail8
    (if 3 ≤ root320Candidates7.length then search edge 3 [2141, 3947] root320Candidates7 6088 winner else winner) = _
  rw [if_neg (by decide)]
def root320Candidates8 : List Nat := [9623, 10139]
set_option maxRecDepth 8192 in
theorem root320Candidates8_checked : candidates edge 4 2141 4049 root320Tail9 winner.bound = root320Candidates8 := by
  unfold candidates
  have row : edge 4049 = (SortedNatTree.contains FastRows.row4049) := by
    change Data.row4049.contains = (SortedNatTree.contains FastRows.row4049)
    exact FastRows.row4049_checked.symm
  rw [row]
  decide
theorem root320Step8 : scan edge (search edge 3) 4 [2141] 2141 root320Tail8 winner =
    scan edge (search edge 3) 4 [2141] 2141 root320Tail9 winner := by
  rw [root320Tail8, scan, if_neg (by decide), root320Candidates8_checked]
  change scan edge (search edge 3) 4 [2141] 2141 root320Tail9
    (if 3 ≤ root320Candidates8.length then search edge 3 [2141, 4049] root320Candidates8 6190 winner else winner) = _
  rw [if_neg (by decide)]
def root320Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root320Candidates9_checked : candidates edge 4 2141 4091 root320Tail10 winner.bound = root320Candidates9 := by
  unfold candidates
  have row : edge 4091 = (SortedNatTree.contains FastRows.row4091) := by
    change Data.row4091.contains = (SortedNatTree.contains FastRows.row4091)
    exact FastRows.row4091_checked.symm
  rw [row]
  decide
theorem root320Step9 : scan edge (search edge 3) 4 [2141] 2141 root320Tail9 winner =
    scan edge (search edge 3) 4 [2141] 2141 root320Tail10 winner := by
  rw [root320Tail9, scan, if_neg (by decide), root320Candidates9_checked]
  change scan edge (search edge 3) 4 [2141] 2141 root320Tail10
    (if 3 ≤ root320Candidates9.length then search edge 3 [2141, 4091] root320Candidates9 6232 winner else winner) = _
  rw [if_neg (by decide)]
def root320Candidates10 : List Nat := [7727, 14951]
set_option maxRecDepth 8192 in
theorem root320Candidates10_checked : candidates edge 4 2141 4649 root320Tail11 winner.bound = root320Candidates10 := by
  unfold candidates
  have row : edge 4649 = (SortedNatTree.contains FastRows.row4649) := by
    change Data.row4649.contains = (SortedNatTree.contains FastRows.row4649)
    exact FastRows.row4649_checked.symm
  rw [row]
  decide
theorem root320Step10 : scan edge (search edge 3) 4 [2141] 2141 root320Tail10 winner =
    scan edge (search edge 3) 4 [2141] 2141 root320Tail11 winner := by
  rw [root320Tail10, scan, if_neg (by decide), root320Candidates10_checked]
  change scan edge (search edge 3) 4 [2141] 2141 root320Tail11
    (if 3 ≤ root320Candidates10.length then search edge 3 [2141, 4649] root320Candidates10 6790 winner else winner) = _
  rw [if_neg (by decide)]
def root320Candidates11 : List Nat := [8693]
set_option maxRecDepth 8192 in
theorem root320Candidates11_checked : candidates edge 4 2141 5153 root320Tail12 winner.bound = root320Candidates11 := by
  unfold candidates
  have row : edge 5153 = (SortedNatTree.contains FastRows.row5153) := by
    change Data.row5153.contains = (SortedNatTree.contains FastRows.row5153)
    exact FastRows.row5153_checked.symm
  rw [row]
  decide
theorem root320Step11 : scan edge (search edge 3) 4 [2141] 2141 root320Tail11 winner =
    scan edge (search edge 3) 4 [2141] 2141 root320Tail12 winner := by
  rw [root320Tail11, scan, if_neg (by decide), root320Candidates11_checked]
  change scan edge (search edge 3) 4 [2141] 2141 root320Tail12
    (if 3 ≤ root320Candidates11.length then search edge 3 [2141, 5153] root320Candidates11 7294 winner else winner) = _
  rw [if_neg (by decide)]
def root320Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root320Candidates12_checked : candidates edge 4 2141 5519 root320Tail13 winner.bound = root320Candidates12 := by
  unfold candidates
  have row : edge 5519 = (SortedNatTree.contains FastRows.row5519) := by
    change Data.row5519.contains = (SortedNatTree.contains FastRows.row5519)
    exact FastRows.row5519_checked.symm
  rw [row]
  decide
theorem root320Step12 : scan edge (search edge 3) 4 [2141] 2141 root320Tail12 winner =
    scan edge (search edge 3) 4 [2141] 2141 root320Tail13 winner := by
  rw [root320Tail12, scan, if_neg (by decide), root320Candidates12_checked]
  change scan edge (search edge 3) 4 [2141] 2141 root320Tail13
    (if 3 ≤ root320Candidates12.length then search edge 3 [2141, 5519] root320Candidates12 7660 winner else winner) = _
  rw [if_neg (by decide)]
theorem root320Scan13 : scan edge (search edge 3) 4 [2141] 2141 root320Tail13 winner = winner := by rw [root320Tail13, scan, if_pos (by decide)]
theorem root320Scan12 : scan edge (search edge 3) 4 [2141] 2141 root320Tail12 winner = winner :=
  (root320Step12).trans root320Scan13
theorem root320Scan11 : scan edge (search edge 3) 4 [2141] 2141 root320Tail11 winner = winner :=
  (root320Step11).trans root320Scan12
theorem root320Scan10 : scan edge (search edge 3) 4 [2141] 2141 root320Tail10 winner = winner :=
  (root320Step10).trans root320Scan11
theorem root320Scan9 : scan edge (search edge 3) 4 [2141] 2141 root320Tail9 winner = winner :=
  (root320Step9).trans root320Scan10
theorem root320Scan8 : scan edge (search edge 3) 4 [2141] 2141 root320Tail8 winner = winner :=
  (root320Step8).trans root320Scan9
theorem root320Scan7 : scan edge (search edge 3) 4 [2141] 2141 root320Tail7 winner = winner :=
  (root320Step7).trans root320Scan8
theorem root320Scan6 : scan edge (search edge 3) 4 [2141] 2141 root320Tail6 winner = winner :=
  (root320Step6).trans root320Scan7
theorem root320Scan5 : scan edge (search edge 3) 4 [2141] 2141 root320Tail5 winner = winner :=
  (root320Step5).trans root320Scan6
theorem root320Scan4 : scan edge (search edge 3) 4 [2141] 2141 root320Tail4 winner = winner :=
  (root320Step4).trans root320Scan5
theorem root320Scan3 : scan edge (search edge 3) 4 [2141] 2141 root320Tail3 winner = winner :=
  (root320Step3).trans root320Scan4
theorem root320Scan2 : scan edge (search edge 3) 4 [2141] 2141 root320Tail2 winner = winner :=
  (root320Step2).trans root320Scan3
theorem root320Scan1 : scan edge (search edge 3) 4 [2141] 2141 root320Tail1 winner = winner :=
  (root320Step1).trans root320Scan2
theorem root320Scan0 : scan edge (search edge 3) 4 [2141] 2141 root320Tail0 winner = winner :=
  (root320Step0).trans root320Scan1
set_option maxRecDepth 4096 in
theorem root320Branch : search edge 4 [2141] root320Candidates 2141 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2141] 2141 root320Tail0 winner = winner
  exact root320Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
