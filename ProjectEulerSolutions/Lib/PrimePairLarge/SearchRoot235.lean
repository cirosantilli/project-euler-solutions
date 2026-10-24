import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root235Candidates : List Nat := Data.row1493
set_option maxRecDepth 8192 in
theorem root235Candidates_checked : candidates edge 5 0 1493 tail236 winner.bound = root235Candidates := by
  apply candidates_eq_row edge 5 0 1493 tail236 root235Candidates winner.bound tail236_ordered
  · decide
  · decide
  · funext q
    rfl
def root235Tail52 : List Nat := []
def root235Tail51 : List Nat := 23819 :: root235Tail52
def root235Tail50 : List Nat := 23297 :: root235Tail51
def root235Tail49 : List Nat := 23003 :: root235Tail50
def root235Tail48 : List Nat := 22973 :: root235Tail49
def root235Tail47 : List Nat := 21803 :: root235Tail48
def root235Tail46 : List Nat := 21089 :: root235Tail47
def root235Tail45 : List Nat := 20921 :: root235Tail46
def root235Tail44 : List Nat := 19571 :: root235Tail45
def root235Tail43 : List Nat := 19139 :: root235Tail44
def root235Tail42 : List Nat := 18401 :: root235Tail43
def root235Tail41 : List Nat := 17921 :: root235Tail42
def root235Tail40 : List Nat := 17891 :: root235Tail41
def root235Tail39 : List Nat := 17417 :: root235Tail40
def root235Tail38 : List Nat := 17159 :: root235Tail39
def root235Tail37 : List Nat := 16703 :: root235Tail38
def root235Tail36 : List Nat := 16451 :: root235Tail37
def root235Tail35 : List Nat := 15581 :: root235Tail36
def root235Tail34 : List Nat := 15443 :: root235Tail35
def root235Tail33 : List Nat := 15377 :: root235Tail34
def root235Tail32 : List Nat := 14951 :: root235Tail33
def root235Tail31 : List Nat := 14657 :: root235Tail32
def root235Tail30 : List Nat := 13691 :: root235Tail31
def root235Tail29 : List Nat := 13649 :: root235Tail30
def root235Tail28 : List Nat := 13487 :: root235Tail29
def root235Tail27 : List Nat := 13127 :: root235Tail28
def root235Tail26 : List Nat := 12791 :: root235Tail27
def root235Tail25 : List Nat := 12101 :: root235Tail26
def root235Tail24 : List Nat := 11489 :: root235Tail25
def root235Tail23 : List Nat := 11171 :: root235Tail24
def root235Tail22 : List Nat := 9851 :: root235Tail23
def root235Tail21 : List Nat := 9473 :: root235Tail22
def root235Tail20 : List Nat := 9461 :: root235Tail21
def root235Tail19 : List Nat := 9059 :: root235Tail20
def root235Tail18 : List Nat := 8999 :: root235Tail19
def root235Tail17 : List Nat := 7607 :: root235Tail18
def root235Tail16 : List Nat := 7541 :: root235Tail17
def root235Tail15 : List Nat := 7457 :: root235Tail16
def root235Tail14 : List Nat := 7229 :: root235Tail15
def root235Tail13 : List Nat := 6599 :: root235Tail14
def root235Tail12 : List Nat := 5867 :: root235Tail13
def root235Tail11 : List Nat := 5651 :: root235Tail12
def root235Tail10 : List Nat := 5279 :: root235Tail11
def root235Tail9 : List Nat := 5189 :: root235Tail10
def root235Tail8 : List Nat := 5021 :: root235Tail9
def root235Tail7 : List Nat := 4943 :: root235Tail8
def root235Tail6 : List Nat := 4649 :: root235Tail7
def root235Tail5 : List Nat := 3623 :: root235Tail6
def root235Tail4 : List Nat := 3329 :: root235Tail5
def root235Tail3 : List Nat := 2609 :: root235Tail4
def root235Tail2 : List Nat := 2441 :: root235Tail3
def root235Tail1 : List Nat := 2339 :: root235Tail2
def root235Tail0 : List Nat := 1523 :: root235Tail1
def root235Candidates0 : List Nat := [5279]
set_option maxRecDepth 8192 in
theorem root235Candidates0_checked : candidates edge 4 1493 1523 root235Tail1 winner.bound = root235Candidates0 := by
  unfold candidates
  have row : edge 1523 = (SortedNatTree.contains FastRows.row1523) := by
    change Data.row1523.contains = (SortedNatTree.contains FastRows.row1523)
    exact FastRows.row1523_checked.symm
  rw [row]
  decide
theorem root235Step0 : scan edge (search edge 3) 4 [1493] 1493 root235Tail0 winner =
    scan edge (search edge 3) 4 [1493] 1493 root235Tail1 winner := by
  rw [root235Tail0, scan, if_neg (by decide), root235Candidates0_checked]
  change scan edge (search edge 3) 4 [1493] 1493 root235Tail1
    (if 3 ≤ root235Candidates0.length then search edge 3 [1493, 1523] root235Candidates0 3016 winner else winner) = _
  rw [if_neg (by decide)]
def root235Candidates1 : List Nat := [16451]
set_option maxRecDepth 8192 in
theorem root235Candidates1_checked : candidates edge 4 1493 2339 root235Tail2 winner.bound = root235Candidates1 := by
  unfold candidates
  have row : edge 2339 = (SortedNatTree.contains FastRows.row2339) := by
    change Data.row2339.contains = (SortedNatTree.contains FastRows.row2339)
    exact FastRows.row2339_checked.symm
  rw [row]
  decide
theorem root235Step1 : scan edge (search edge 3) 4 [1493] 1493 root235Tail1 winner =
    scan edge (search edge 3) 4 [1493] 1493 root235Tail2 winner := by
  rw [root235Tail1, scan, if_neg (by decide), root235Candidates1_checked]
  change scan edge (search edge 3) 4 [1493] 1493 root235Tail2
    (if 3 ≤ root235Candidates1.length then search edge 3 [1493, 2339] root235Candidates1 3832 winner else winner) = _
  rw [if_neg (by decide)]
def root235Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root235Candidates2_checked : candidates edge 4 1493 2441 root235Tail3 winner.bound = root235Candidates2 := by
  unfold candidates
  have row : edge 2441 = (SortedNatTree.contains FastRows.row2441) := by
    change Data.row2441.contains = (SortedNatTree.contains FastRows.row2441)
    exact FastRows.row2441_checked.symm
  rw [row]
  decide
theorem root235Step2 : scan edge (search edge 3) 4 [1493] 1493 root235Tail2 winner =
    scan edge (search edge 3) 4 [1493] 1493 root235Tail3 winner := by
  rw [root235Tail2, scan, if_neg (by decide), root235Candidates2_checked]
  change scan edge (search edge 3) 4 [1493] 1493 root235Tail3
    (if 3 ≤ root235Candidates2.length then search edge 3 [1493, 2441] root235Candidates2 3934 winner else winner) = _
  rw [if_neg (by decide)]
def root235Candidates3 : List Nat := [5651, 6599, 21803]
set_option maxRecDepth 8192 in
theorem root235Candidates3_checked : candidates edge 4 1493 2609 root235Tail4 winner.bound = root235Candidates3 := by
  unfold candidates
  have row : edge 2609 = (SortedNatTree.contains FastRows.row2609) := by
    change Data.row2609.contains = (SortedNatTree.contains FastRows.row2609)
    exact FastRows.row2609_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root235Child3 : search edge 3 [1493, 2609] root235Candidates3 4102 winner = winner := by decide
theorem root235Step3 : scan edge (search edge 3) 4 [1493] 1493 root235Tail3 winner =
    scan edge (search edge 3) 4 [1493] 1493 root235Tail4 winner := by
  rw [root235Tail3, scan, if_neg (by decide), root235Candidates3_checked]
  change scan edge (search edge 3) 4 [1493] 1493 root235Tail4
    (if 3 ≤ root235Candidates3.length then search edge 3 [1493, 2609] root235Candidates3 4102 winner else winner) = _
  rw [if_pos (by decide), root235Child3]
def root235Candidates4 : List Nat := [4649]
set_option maxRecDepth 8192 in
theorem root235Candidates4_checked : candidates edge 4 1493 3329 root235Tail5 winner.bound = root235Candidates4 := by
  unfold candidates
  have row : edge 3329 = (SortedNatTree.contains FastRows.row3329) := by
    change Data.row3329.contains = (SortedNatTree.contains FastRows.row3329)
    exact FastRows.row3329_checked.symm
  rw [row]
  decide
theorem root235Step4 : scan edge (search edge 3) 4 [1493] 1493 root235Tail4 winner =
    scan edge (search edge 3) 4 [1493] 1493 root235Tail5 winner := by
  rw [root235Tail4, scan, if_neg (by decide), root235Candidates4_checked]
  change scan edge (search edge 3) 4 [1493] 1493 root235Tail5
    (if 3 ≤ root235Candidates4.length then search edge 3 [1493, 3329] root235Candidates4 4822 winner else winner) = _
  rw [if_neg (by decide)]
def root235Candidates5 : List Nat := [9059, 18401]
set_option maxRecDepth 8192 in
theorem root235Candidates5_checked : candidates edge 4 1493 3623 root235Tail6 winner.bound = root235Candidates5 := by
  unfold candidates
  have row : edge 3623 = (SortedNatTree.contains FastRows.row3623) := by
    change Data.row3623.contains = (SortedNatTree.contains FastRows.row3623)
    exact FastRows.row3623_checked.symm
  rw [row]
  decide
theorem root235Step5 : scan edge (search edge 3) 4 [1493] 1493 root235Tail5 winner =
    scan edge (search edge 3) 4 [1493] 1493 root235Tail6 winner := by
  rw [root235Tail5, scan, if_neg (by decide), root235Candidates5_checked]
  change scan edge (search edge 3) 4 [1493] 1493 root235Tail6
    (if 3 ≤ root235Candidates5.length then search edge 3 [1493, 3623] root235Candidates5 5116 winner else winner) = _
  rw [if_neg (by decide)]
def root235Candidates6 : List Nat := [9461, 14951]
set_option maxRecDepth 8192 in
theorem root235Candidates6_checked : candidates edge 4 1493 4649 root235Tail7 winner.bound = root235Candidates6 := by
  unfold candidates
  have row : edge 4649 = (SortedNatTree.contains FastRows.row4649) := by
    change Data.row4649.contains = (SortedNatTree.contains FastRows.row4649)
    exact FastRows.row4649_checked.symm
  rw [row]
  decide
theorem root235Step6 : scan edge (search edge 3) 4 [1493] 1493 root235Tail6 winner =
    scan edge (search edge 3) 4 [1493] 1493 root235Tail7 winner := by
  rw [root235Tail6, scan, if_neg (by decide), root235Candidates6_checked]
  change scan edge (search edge 3) 4 [1493] 1493 root235Tail7
    (if 3 ≤ root235Candidates6.length then search edge 3 [1493, 4649] root235Candidates6 6142 winner else winner) = _
  rw [if_neg (by decide)]
def root235Candidates7 : List Nat := [5021, 5189]
set_option maxRecDepth 8192 in
theorem root235Candidates7_checked : candidates edge 4 1493 4943 root235Tail8 winner.bound = root235Candidates7 := by
  unfold candidates
  have row : edge 4943 = (SortedNatTree.contains FastRows.row4943) := by
    change Data.row4943.contains = (SortedNatTree.contains FastRows.row4943)
    exact FastRows.row4943_checked.symm
  rw [row]
  decide
theorem root235Step7 : scan edge (search edge 3) 4 [1493] 1493 root235Tail7 winner =
    scan edge (search edge 3) 4 [1493] 1493 root235Tail8 winner := by
  rw [root235Tail7, scan, if_neg (by decide), root235Candidates7_checked]
  change scan edge (search edge 3) 4 [1493] 1493 root235Tail8
    (if 3 ≤ root235Candidates7.length then search edge 3 [1493, 4943] root235Candidates7 6436 winner else winner) = _
  rw [if_neg (by decide)]
def root235Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root235Candidates8_checked : candidates edge 4 1493 5021 root235Tail9 winner.bound = root235Candidates8 := by
  unfold candidates
  have row : edge 5021 = (SortedNatTree.contains FastRows.row5021) := by
    change Data.row5021.contains = (SortedNatTree.contains FastRows.row5021)
    exact FastRows.row5021_checked.symm
  rw [row]
  decide
theorem root235Step8 : scan edge (search edge 3) 4 [1493] 1493 root235Tail8 winner =
    scan edge (search edge 3) 4 [1493] 1493 root235Tail9 winner := by
  rw [root235Tail8, scan, if_neg (by decide), root235Candidates8_checked]
  change scan edge (search edge 3) 4 [1493] 1493 root235Tail9
    (if 3 ≤ root235Candidates8.length then search edge 3 [1493, 5021] root235Candidates8 6514 winner else winner) = _
  rw [if_neg (by decide)]
def root235Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root235Candidates9_checked : candidates edge 4 1493 5189 root235Tail10 winner.bound = root235Candidates9 := by
  unfold candidates
  have row : edge 5189 = (SortedNatTree.contains FastRows.row5189) := by
    change Data.row5189.contains = (SortedNatTree.contains FastRows.row5189)
    exact FastRows.row5189_checked.symm
  rw [row]
  decide
theorem root235Step9 : scan edge (search edge 3) 4 [1493] 1493 root235Tail9 winner =
    scan edge (search edge 3) 4 [1493] 1493 root235Tail10 winner := by
  rw [root235Tail9, scan, if_neg (by decide), root235Candidates9_checked]
  change scan edge (search edge 3) 4 [1493] 1493 root235Tail10
    (if 3 ≤ root235Candidates9.length then search edge 3 [1493, 5189] root235Candidates9 6682 winner else winner) = _
  rw [if_neg (by decide)]
def root235Candidates10 : List Nat := [5651]
set_option maxRecDepth 8192 in
theorem root235Candidates10_checked : candidates edge 4 1493 5279 root235Tail11 winner.bound = root235Candidates10 := by
  unfold candidates
  have row : edge 5279 = (SortedNatTree.contains FastRows.row5279) := by
    change Data.row5279.contains = (SortedNatTree.contains FastRows.row5279)
    exact FastRows.row5279_checked.symm
  rw [row]
  decide
theorem root235Step10 : scan edge (search edge 3) 4 [1493] 1493 root235Tail10 winner =
    scan edge (search edge 3) 4 [1493] 1493 root235Tail11 winner := by
  rw [root235Tail10, scan, if_neg (by decide), root235Candidates10_checked]
  change scan edge (search edge 3) 4 [1493] 1493 root235Tail11
    (if 3 ≤ root235Candidates10.length then search edge 3 [1493, 5279] root235Candidates10 6772 winner else winner) = _
  rw [if_neg (by decide)]
def root235Candidates11 : List Nat := [14657]
set_option maxRecDepth 8192 in
theorem root235Candidates11_checked : candidates edge 4 1493 5651 root235Tail12 winner.bound = root235Candidates11 := by
  unfold candidates
  have row : edge 5651 = (SortedNatTree.contains FastRows.row5651) := by
    change Data.row5651.contains = (SortedNatTree.contains FastRows.row5651)
    exact FastRows.row5651_checked.symm
  rw [row]
  decide
theorem root235Step11 : scan edge (search edge 3) 4 [1493] 1493 root235Tail11 winner =
    scan edge (search edge 3) 4 [1493] 1493 root235Tail12 winner := by
  rw [root235Tail11, scan, if_neg (by decide), root235Candidates11_checked]
  change scan edge (search edge 3) 4 [1493] 1493 root235Tail12
    (if 3 ≤ root235Candidates11.length then search edge 3 [1493, 5651] root235Candidates11 7144 winner else winner) = _
  rw [if_neg (by decide)]
def root235Candidates12 : List Nat := [14657]
set_option maxRecDepth 8192 in
theorem root235Candidates12_checked : candidates edge 4 1493 5867 root235Tail13 winner.bound = root235Candidates12 := by
  unfold candidates
  have row : edge 5867 = (SortedNatTree.contains FastRows.row5867) := by
    change Data.row5867.contains = (SortedNatTree.contains FastRows.row5867)
    exact FastRows.row5867_checked.symm
  rw [row]
  decide
theorem root235Step12 : scan edge (search edge 3) 4 [1493] 1493 root235Tail12 winner =
    scan edge (search edge 3) 4 [1493] 1493 root235Tail13 winner := by
  rw [root235Tail12, scan, if_neg (by decide), root235Candidates12_checked]
  change scan edge (search edge 3) 4 [1493] 1493 root235Tail13
    (if 3 ≤ root235Candidates12.length then search edge 3 [1493, 5867] root235Candidates12 7360 winner else winner) = _
  rw [if_neg (by decide)]
theorem root235Scan13 : scan edge (search edge 3) 4 [1493] 1493 root235Tail13 winner = winner := by rw [root235Tail13, scan, if_pos (by decide)]
theorem root235Scan12 : scan edge (search edge 3) 4 [1493] 1493 root235Tail12 winner = winner :=
  (root235Step12).trans root235Scan13
theorem root235Scan11 : scan edge (search edge 3) 4 [1493] 1493 root235Tail11 winner = winner :=
  (root235Step11).trans root235Scan12
theorem root235Scan10 : scan edge (search edge 3) 4 [1493] 1493 root235Tail10 winner = winner :=
  (root235Step10).trans root235Scan11
theorem root235Scan9 : scan edge (search edge 3) 4 [1493] 1493 root235Tail9 winner = winner :=
  (root235Step9).trans root235Scan10
theorem root235Scan8 : scan edge (search edge 3) 4 [1493] 1493 root235Tail8 winner = winner :=
  (root235Step8).trans root235Scan9
theorem root235Scan7 : scan edge (search edge 3) 4 [1493] 1493 root235Tail7 winner = winner :=
  (root235Step7).trans root235Scan8
theorem root235Scan6 : scan edge (search edge 3) 4 [1493] 1493 root235Tail6 winner = winner :=
  (root235Step6).trans root235Scan7
theorem root235Scan5 : scan edge (search edge 3) 4 [1493] 1493 root235Tail5 winner = winner :=
  (root235Step5).trans root235Scan6
theorem root235Scan4 : scan edge (search edge 3) 4 [1493] 1493 root235Tail4 winner = winner :=
  (root235Step4).trans root235Scan5
theorem root235Scan3 : scan edge (search edge 3) 4 [1493] 1493 root235Tail3 winner = winner :=
  (root235Step3).trans root235Scan4
theorem root235Scan2 : scan edge (search edge 3) 4 [1493] 1493 root235Tail2 winner = winner :=
  (root235Step2).trans root235Scan3
theorem root235Scan1 : scan edge (search edge 3) 4 [1493] 1493 root235Tail1 winner = winner :=
  (root235Step1).trans root235Scan2
theorem root235Scan0 : scan edge (search edge 3) 4 [1493] 1493 root235Tail0 winner = winner :=
  (root235Step0).trans root235Scan1
set_option maxRecDepth 4096 in
theorem root235Branch : search edge 4 [1493] root235Candidates 1493 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1493] 1493 root235Tail0 winner = winner
  exact root235Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
