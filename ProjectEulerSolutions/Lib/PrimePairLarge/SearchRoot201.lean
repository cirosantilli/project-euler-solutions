import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root201Candidates : List Nat := Data.row1249
set_option maxRecDepth 8192 in
theorem root201Candidates_checked : candidates edge 5 0 1249 tail202 winner.bound = root201Candidates := by
  apply candidates_eq_row edge 5 0 1249 tail202 root201Candidates winner.bound tail202_ordered
  · decide
  · decide
  · funext q
    rfl
def root201Tail46 : List Nat := []
def root201Tail45 : List Nat := 24391 :: root201Tail46
def root201Tail44 : List Nat := 24103 :: root201Tail45
def root201Tail43 : List Nat := 24061 :: root201Tail44
def root201Tail42 : List Nat := 23977 :: root201Tail43
def root201Tail41 : List Nat := 22921 :: root201Tail42
def root201Tail40 : List Nat := 22543 :: root201Tail41
def root201Tail39 : List Nat := 21283 :: root201Tail40
def root201Tail38 : List Nat := 20899 :: root201Tail39
def root201Tail37 : List Nat := 20101 :: root201Tail38
def root201Tail36 : List Nat := 19609 :: root201Tail37
def root201Tail35 : List Nat := 19603 :: root201Tail36
def root201Tail34 : List Nat := 18679 :: root201Tail35
def root201Tail33 : List Nat := 17467 :: root201Tail34
def root201Tail32 : List Nat := 15733 :: root201Tail33
def root201Tail31 : List Nat := 15649 :: root201Tail32
def root201Tail30 : List Nat := 15601 :: root201Tail31
def root201Tail29 : List Nat := 15187 :: root201Tail30
def root201Tail28 : List Nat := 15061 :: root201Tail29
def root201Tail27 : List Nat := 13339 :: root201Tail28
def root201Tail26 : List Nat := 13219 :: root201Tail27
def root201Tail25 : List Nat := 12157 :: root201Tail26
def root201Tail24 : List Nat := 11959 :: root201Tail25
def root201Tail23 : List Nat := 11503 :: root201Tail24
def root201Tail22 : List Nat := 11329 :: root201Tail23
def root201Tail21 : List Nat := 10993 :: root201Tail22
def root201Tail20 : List Nat := 10099 :: root201Tail21
def root201Tail19 : List Nat := 9187 :: root201Tail20
def root201Tail18 : List Nat := 9001 :: root201Tail19
def root201Tail17 : List Nat := 8803 :: root201Tail18
def root201Tail16 : List Nat := 8641 :: root201Tail17
def root201Tail15 : List Nat := 7621 :: root201Tail16
def root201Tail14 : List Nat := 7369 :: root201Tail15
def root201Tail13 : List Nat := 6949 :: root201Tail14
def root201Tail12 : List Nat := 6871 :: root201Tail13
def root201Tail11 : List Nat := 6151 :: root201Tail12
def root201Tail10 : List Nat := 6079 :: root201Tail11
def root201Tail9 : List Nat := 5851 :: root201Tail10
def root201Tail8 : List Nat := 5569 :: root201Tail9
def root201Tail7 : List Nat := 5563 :: root201Tail8
def root201Tail6 : List Nat := 4441 :: root201Tail7
def root201Tail5 : List Nat := 3727 :: root201Tail6
def root201Tail4 : List Nat := 3391 :: root201Tail5
def root201Tail3 : List Nat := 2551 :: root201Tail4
def root201Tail2 : List Nat := 2377 :: root201Tail3
def root201Tail1 : List Nat := 2131 :: root201Tail2
def root201Tail0 : List Nat := 1867 :: root201Tail1
def root201Candidates0 : List Nat := [12157]
set_option maxRecDepth 8192 in
theorem root201Candidates0_checked : candidates edge 4 1249 1867 root201Tail1 winner.bound = root201Candidates0 := by
  unfold candidates
  have row : edge 1867 = (SortedNatTree.contains FastRows.row1867) := by
    change Data.row1867.contains = (SortedNatTree.contains FastRows.row1867)
    exact FastRows.row1867_checked.symm
  rw [row]
  decide
theorem root201Step0 : scan edge (search edge 3) 4 [1249] 1249 root201Tail0 winner =
    scan edge (search edge 3) 4 [1249] 1249 root201Tail1 winner := by
  rw [root201Tail0, scan, if_neg (by decide), root201Candidates0_checked]
  change scan edge (search edge 3) 4 [1249] 1249 root201Tail1
    (if 3 ≤ root201Candidates0.length then search edge 3 [1249, 1867] root201Candidates0 3116 winner else winner) = _
  rw [if_neg (by decide)]
def root201Candidates1 : List Nat := [5563, 13339]
set_option maxRecDepth 8192 in
theorem root201Candidates1_checked : candidates edge 4 1249 2131 root201Tail2 winner.bound = root201Candidates1 := by
  unfold candidates
  have row : edge 2131 = (SortedNatTree.contains FastRows.row2131) := by
    change Data.row2131.contains = (SortedNatTree.contains FastRows.row2131)
    exact FastRows.row2131_checked.symm
  rw [row]
  decide
theorem root201Step1 : scan edge (search edge 3) 4 [1249] 1249 root201Tail1 winner =
    scan edge (search edge 3) 4 [1249] 1249 root201Tail2 winner := by
  rw [root201Tail1, scan, if_neg (by decide), root201Candidates1_checked]
  change scan edge (search edge 3) 4 [1249] 1249 root201Tail2
    (if 3 ≤ root201Candidates1.length then search edge 3 [1249, 2131] root201Candidates1 3380 winner else winner) = _
  rw [if_neg (by decide)]
def root201Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root201Candidates2_checked : candidates edge 4 1249 2377 root201Tail3 winner.bound = root201Candidates2 := by
  unfold candidates
  have row : edge 2377 = (SortedNatTree.contains FastRows.row2377) := by
    change Data.row2377.contains = (SortedNatTree.contains FastRows.row2377)
    exact FastRows.row2377_checked.symm
  rw [row]
  decide
theorem root201Step2 : scan edge (search edge 3) 4 [1249] 1249 root201Tail2 winner =
    scan edge (search edge 3) 4 [1249] 1249 root201Tail3 winner := by
  rw [root201Tail2, scan, if_neg (by decide), root201Candidates2_checked]
  change scan edge (search edge 3) 4 [1249] 1249 root201Tail3
    (if 3 ≤ root201Candidates2.length then search edge 3 [1249, 2377] root201Candidates2 3626 winner else winner) = _
  rw [if_neg (by decide)]
def root201Candidates3 : List Nat := [12157, 20101]
set_option maxRecDepth 8192 in
theorem root201Candidates3_checked : candidates edge 4 1249 2551 root201Tail4 winner.bound = root201Candidates3 := by
  unfold candidates
  have row : edge 2551 = (SortedNatTree.contains FastRows.row2551) := by
    change Data.row2551.contains = (SortedNatTree.contains FastRows.row2551)
    exact FastRows.row2551_checked.symm
  rw [row]
  decide
theorem root201Step3 : scan edge (search edge 3) 4 [1249] 1249 root201Tail3 winner =
    scan edge (search edge 3) 4 [1249] 1249 root201Tail4 winner := by
  rw [root201Tail3, scan, if_neg (by decide), root201Candidates3_checked]
  change scan edge (search edge 3) 4 [1249] 1249 root201Tail4
    (if 3 ≤ root201Candidates3.length then search edge 3 [1249, 2551] root201Candidates3 3800 winner else winner) = _
  rw [if_neg (by decide)]
def root201Candidates4 : List Nat := [6079]
set_option maxRecDepth 8192 in
theorem root201Candidates4_checked : candidates edge 4 1249 3391 root201Tail5 winner.bound = root201Candidates4 := by
  unfold candidates
  have row : edge 3391 = (SortedNatTree.contains FastRows.row3391) := by
    change Data.row3391.contains = (SortedNatTree.contains FastRows.row3391)
    exact FastRows.row3391_checked.symm
  rw [row]
  decide
theorem root201Step4 : scan edge (search edge 3) 4 [1249] 1249 root201Tail4 winner =
    scan edge (search edge 3) 4 [1249] 1249 root201Tail5 winner := by
  rw [root201Tail4, scan, if_neg (by decide), root201Candidates4_checked]
  change scan edge (search edge 3) 4 [1249] 1249 root201Tail5
    (if 3 ≤ root201Candidates4.length then search edge 3 [1249, 3391] root201Candidates4 4640 winner else winner) = _
  rw [if_neg (by decide)]
def root201Candidates5 : List Nat := [6949, 10993, 20899]
set_option maxRecDepth 8192 in
theorem root201Candidates5_checked : candidates edge 4 1249 3727 root201Tail6 winner.bound = root201Candidates5 := by
  unfold candidates
  have row : edge 3727 = (SortedNatTree.contains FastRows.row3727) := by
    change Data.row3727.contains = (SortedNatTree.contains FastRows.row3727)
    exact FastRows.row3727_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root201Child5 : search edge 3 [1249, 3727] root201Candidates5 4976 winner = winner := by decide
theorem root201Step5 : scan edge (search edge 3) 4 [1249] 1249 root201Tail5 winner =
    scan edge (search edge 3) 4 [1249] 1249 root201Tail6 winner := by
  rw [root201Tail5, scan, if_neg (by decide), root201Candidates5_checked]
  change scan edge (search edge 3) 4 [1249] 1249 root201Tail6
    (if 3 ≤ root201Candidates5.length then search edge 3 [1249, 3727] root201Candidates5 4976 winner else winner) = _
  rw [if_pos (by decide), root201Child5]
def root201Candidates6 : List Nat := [6949]
set_option maxRecDepth 8192 in
theorem root201Candidates6_checked : candidates edge 4 1249 4441 root201Tail7 winner.bound = root201Candidates6 := by
  unfold candidates
  have row : edge 4441 = (SortedNatTree.contains FastRows.row4441) := by
    change Data.row4441.contains = (SortedNatTree.contains FastRows.row4441)
    exact FastRows.row4441_checked.symm
  rw [row]
  decide
theorem root201Step6 : scan edge (search edge 3) 4 [1249] 1249 root201Tail6 winner =
    scan edge (search edge 3) 4 [1249] 1249 root201Tail7 winner := by
  rw [root201Tail6, scan, if_neg (by decide), root201Candidates6_checked]
  change scan edge (search edge 3) 4 [1249] 1249 root201Tail7
    (if 3 ≤ root201Candidates6.length then search edge 3 [1249, 4441] root201Candidates6 5690 winner else winner) = _
  rw [if_neg (by decide)]
def root201Candidates7 : List Nat := [6079, 12157]
set_option maxRecDepth 8192 in
theorem root201Candidates7_checked : candidates edge 4 1249 5563 root201Tail8 winner.bound = root201Candidates7 := by
  unfold candidates
  have row : edge 5563 = (SortedNatTree.contains FastRows.row5563) := by
    change Data.row5563.contains = (SortedNatTree.contains FastRows.row5563)
    exact FastRows.row5563_checked.symm
  rw [row]
  decide
theorem root201Step7 : scan edge (search edge 3) 4 [1249] 1249 root201Tail7 winner =
    scan edge (search edge 3) 4 [1249] 1249 root201Tail8 winner := by
  rw [root201Tail7, scan, if_neg (by decide), root201Candidates7_checked]
  change scan edge (search edge 3) 4 [1249] 1249 root201Tail8
    (if 3 ≤ root201Candidates7.length then search edge 3 [1249, 5563] root201Candidates7 6812 winner else winner) = _
  rw [if_neg (by decide)]
def root201Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root201Candidates8_checked : candidates edge 4 1249 5569 root201Tail9 winner.bound = root201Candidates8 := by
  unfold candidates
  have row : edge 5569 = (SortedNatTree.contains FastRows.row5569) := by
    change Data.row5569.contains = (SortedNatTree.contains FastRows.row5569)
    exact FastRows.row5569_checked.symm
  rw [row]
  decide
theorem root201Step8 : scan edge (search edge 3) 4 [1249] 1249 root201Tail8 winner =
    scan edge (search edge 3) 4 [1249] 1249 root201Tail9 winner := by
  rw [root201Tail8, scan, if_neg (by decide), root201Candidates8_checked]
  change scan edge (search edge 3) 4 [1249] 1249 root201Tail9
    (if 3 ≤ root201Candidates8.length then search edge 3 [1249, 5569] root201Candidates8 6818 winner else winner) = _
  rw [if_neg (by decide)]
def root201Candidates9 : List Nat := [11959, 15649]
set_option maxRecDepth 8192 in
theorem root201Candidates9_checked : candidates edge 4 1249 5851 root201Tail10 winner.bound = root201Candidates9 := by
  unfold candidates
  have row : edge 5851 = (SortedNatTree.contains FastRows.row5851) := by
    change Data.row5851.contains = (SortedNatTree.contains FastRows.row5851)
    exact FastRows.row5851_checked.symm
  rw [row]
  decide
theorem root201Step9 : scan edge (search edge 3) 4 [1249] 1249 root201Tail9 winner =
    scan edge (search edge 3) 4 [1249] 1249 root201Tail10 winner := by
  rw [root201Tail9, scan, if_neg (by decide), root201Candidates9_checked]
  change scan edge (search edge 3) 4 [1249] 1249 root201Tail10
    (if 3 ≤ root201Candidates9.length then search edge 3 [1249, 5851] root201Candidates9 7100 winner else winner) = _
  rw [if_neg (by decide)]
def root201Candidates10 : List Nat := [7621]
set_option maxRecDepth 8192 in
theorem root201Candidates10_checked : candidates edge 4 1249 6079 root201Tail11 winner.bound = root201Candidates10 := by
  unfold candidates
  have row : edge 6079 = (SortedNatTree.contains FastRows.row6079) := by
    change Data.row6079.contains = (SortedNatTree.contains FastRows.row6079)
    exact FastRows.row6079_checked.symm
  rw [row]
  decide
theorem root201Step10 : scan edge (search edge 3) 4 [1249] 1249 root201Tail10 winner =
    scan edge (search edge 3) 4 [1249] 1249 root201Tail11 winner := by
  rw [root201Tail10, scan, if_neg (by decide), root201Candidates10_checked]
  change scan edge (search edge 3) 4 [1249] 1249 root201Tail11
    (if 3 ≤ root201Candidates10.length then search edge 3 [1249, 6079] root201Candidates10 7328 winner else winner) = _
  rw [if_neg (by decide)]
def root201Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root201Candidates11_checked : candidates edge 4 1249 6151 root201Tail12 winner.bound = root201Candidates11 := by
  unfold candidates
  have row : edge 6151 = (SortedNatTree.contains FastRows.row6151) := by
    change Data.row6151.contains = (SortedNatTree.contains FastRows.row6151)
    exact FastRows.row6151_checked.symm
  rw [row]
  decide
theorem root201Step11 : scan edge (search edge 3) 4 [1249] 1249 root201Tail11 winner =
    scan edge (search edge 3) 4 [1249] 1249 root201Tail12 winner := by
  rw [root201Tail11, scan, if_neg (by decide), root201Candidates11_checked]
  change scan edge (search edge 3) 4 [1249] 1249 root201Tail12
    (if 3 ≤ root201Candidates11.length then search edge 3 [1249, 6151] root201Candidates11 7400 winner else winner) = _
  rw [if_neg (by decide)]
theorem root201Scan12 : scan edge (search edge 3) 4 [1249] 1249 root201Tail12 winner = winner := by rw [root201Tail12, scan, if_pos (by decide)]
theorem root201Scan11 : scan edge (search edge 3) 4 [1249] 1249 root201Tail11 winner = winner :=
  (root201Step11).trans root201Scan12
theorem root201Scan10 : scan edge (search edge 3) 4 [1249] 1249 root201Tail10 winner = winner :=
  (root201Step10).trans root201Scan11
theorem root201Scan9 : scan edge (search edge 3) 4 [1249] 1249 root201Tail9 winner = winner :=
  (root201Step9).trans root201Scan10
theorem root201Scan8 : scan edge (search edge 3) 4 [1249] 1249 root201Tail8 winner = winner :=
  (root201Step8).trans root201Scan9
theorem root201Scan7 : scan edge (search edge 3) 4 [1249] 1249 root201Tail7 winner = winner :=
  (root201Step7).trans root201Scan8
theorem root201Scan6 : scan edge (search edge 3) 4 [1249] 1249 root201Tail6 winner = winner :=
  (root201Step6).trans root201Scan7
theorem root201Scan5 : scan edge (search edge 3) 4 [1249] 1249 root201Tail5 winner = winner :=
  (root201Step5).trans root201Scan6
theorem root201Scan4 : scan edge (search edge 3) 4 [1249] 1249 root201Tail4 winner = winner :=
  (root201Step4).trans root201Scan5
theorem root201Scan3 : scan edge (search edge 3) 4 [1249] 1249 root201Tail3 winner = winner :=
  (root201Step3).trans root201Scan4
theorem root201Scan2 : scan edge (search edge 3) 4 [1249] 1249 root201Tail2 winner = winner :=
  (root201Step2).trans root201Scan3
theorem root201Scan1 : scan edge (search edge 3) 4 [1249] 1249 root201Tail1 winner = winner :=
  (root201Step1).trans root201Scan2
theorem root201Scan0 : scan edge (search edge 3) 4 [1249] 1249 root201Tail0 winner = winner :=
  (root201Step0).trans root201Scan1
set_option maxRecDepth 4096 in
theorem root201Branch : search edge 4 [1249] root201Candidates 1249 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1249] 1249 root201Tail0 winner = winner
  exact root201Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
