import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root483Candidates : List Nat := Data.row3467
set_option maxRecDepth 8192 in
theorem root483Candidates_checked : candidates edge 5 0 3467 tail484 winner.bound = root483Candidates := by
  apply candidates_eq_row edge 5 0 3467 tail484 root483Candidates winner.bound tail484_ordered
  · decide
  · decide
  · funext q
    rfl
def root483Tail37 : List Nat := []
def root483Tail36 : List Nat := 21911 :: root483Tail37
def root483Tail35 : List Nat := 20123 :: root483Tail36
def root483Tail34 : List Nat := 19073 :: root483Tail35
def root483Tail33 : List Nat := 18947 :: root483Tail34
def root483Tail32 : List Nat := 18311 :: root483Tail33
def root483Tail31 : List Nat := 18149 :: root483Tail32
def root483Tail30 : List Nat := 17123 :: root483Tail31
def root483Tail29 : List Nat := 16661 :: root483Tail30
def root483Tail28 : List Nat := 16631 :: root483Tail29
def root483Tail27 : List Nat := 10691 :: root483Tail28
def root483Tail26 : List Nat := 10589 :: root483Tail27
def root483Tail25 : List Nat := 9833 :: root483Tail26
def root483Tail24 : List Nat := 9479 :: root483Tail25
def root483Tail23 : List Nat := 9059 :: root483Tail24
def root483Tail22 : List Nat := 8951 :: root483Tail23
def root483Tail21 : List Nat := 8663 :: root483Tail22
def root483Tail20 : List Nat := 8573 :: root483Tail21
def root483Tail19 : List Nat := 8117 :: root483Tail20
def root483Tail18 : List Nat := 7949 :: root483Tail19
def root483Tail17 : List Nat := 7481 :: root483Tail18
def root483Tail16 : List Nat := 7001 :: root483Tail17
def root483Tail15 : List Nat := 6959 :: root483Tail16
def root483Tail14 : List Nat := 6599 :: root483Tail15
def root483Tail13 : List Nat := 6113 :: root483Tail14
def root483Tail12 : List Nat := 5807 :: root483Tail13
def root483Tail11 : List Nat := 5591 :: root483Tail12
def root483Tail10 : List Nat := 5471 :: root483Tail11
def root483Tail9 : List Nat := 5297 :: root483Tail10
def root483Tail8 : List Nat := 5273 :: root483Tail9
def root483Tail7 : List Nat := 5231 :: root483Tail8
def root483Tail6 : List Nat := 5009 :: root483Tail7
def root483Tail5 : List Nat := 4919 :: root483Tail6
def root483Tail4 : List Nat := 4817 :: root483Tail5
def root483Tail3 : List Nat := 4799 :: root483Tail4
def root483Tail2 : List Nat := 4457 :: root483Tail3
def root483Tail1 : List Nat := 3851 :: root483Tail2
def root483Tail0 : List Nat := 3623 :: root483Tail1
def root483Candidates0 : List Nat := [9059]
set_option maxRecDepth 8192 in
theorem root483Candidates0_checked : candidates edge 4 3467 3623 root483Tail1 winner.bound = root483Candidates0 := by
  unfold candidates
  have row : edge 3623 = (SortedNatTree.contains FastRows.row3623) := by
    change Data.row3623.contains = (SortedNatTree.contains FastRows.row3623)
    exact FastRows.row3623_checked.symm
  rw [row]
  decide
theorem root483Step0 : scan edge (search edge 3) 4 [3467] 3467 root483Tail0 winner =
    scan edge (search edge 3) 4 [3467] 3467 root483Tail1 winner := by
  rw [root483Tail0, scan, if_neg (by decide), root483Candidates0_checked]
  change scan edge (search edge 3) 4 [3467] 3467 root483Tail1
    (if 3 ≤ root483Candidates0.length then search edge 3 [3467, 3623] root483Candidates0 7090 winner else winner) = _
  rw [if_neg (by decide)]
def root483Candidates1 : List Nat := [8663, 16661]
set_option maxRecDepth 8192 in
theorem root483Candidates1_checked : candidates edge 4 3467 3851 root483Tail2 winner.bound = root483Candidates1 := by
  unfold candidates
  have row : edge 3851 = (SortedNatTree.contains FastRows.row3851) := by
    change Data.row3851.contains = (SortedNatTree.contains FastRows.row3851)
    exact FastRows.row3851_checked.symm
  rw [row]
  decide
theorem root483Step1 : scan edge (search edge 3) 4 [3467] 3467 root483Tail1 winner =
    scan edge (search edge 3) 4 [3467] 3467 root483Tail2 winner := by
  rw [root483Tail1, scan, if_neg (by decide), root483Candidates1_checked]
  change scan edge (search edge 3) 4 [3467] 3467 root483Tail2
    (if 3 ≤ root483Candidates1.length then search edge 3 [3467, 3851] root483Candidates1 7318 winner else winner) = _
  rw [if_neg (by decide)]
def root483Candidates2 : List Nat := [5591]
set_option maxRecDepth 8192 in
theorem root483Candidates2_checked : candidates edge 4 3467 4457 root483Tail3 winner.bound = root483Candidates2 := by
  unfold candidates
  have row : edge 4457 = (SortedNatTree.contains FastRows.row4457) := by
    change Data.row4457.contains = (SortedNatTree.contains FastRows.row4457)
    exact FastRows.row4457_checked.symm
  rw [row]
  decide
theorem root483Step2 : scan edge (search edge 3) 4 [3467] 3467 root483Tail2 winner =
    scan edge (search edge 3) 4 [3467] 3467 root483Tail3 winner := by
  rw [root483Tail2, scan, if_neg (by decide), root483Candidates2_checked]
  change scan edge (search edge 3) 4 [3467] 3467 root483Tail3
    (if 3 ≤ root483Candidates2.length then search edge 3 [3467, 4457] root483Candidates2 7924 winner else winner) = _
  rw [if_neg (by decide)]
def root483Candidates3 : List Nat := [5231]
set_option maxRecDepth 8192 in
theorem root483Candidates3_checked : candidates edge 4 3467 4799 root483Tail4 winner.bound = root483Candidates3 := by
  unfold candidates
  have row : edge 4799 = (SortedNatTree.contains FastRows.row4799) := by
    change Data.row4799.contains = (SortedNatTree.contains FastRows.row4799)
    exact FastRows.row4799_checked.symm
  rw [row]
  decide
theorem root483Step3 : scan edge (search edge 3) 4 [3467] 3467 root483Tail3 winner =
    scan edge (search edge 3) 4 [3467] 3467 root483Tail4 winner := by
  rw [root483Tail3, scan, if_neg (by decide), root483Candidates3_checked]
  change scan edge (search edge 3) 4 [3467] 3467 root483Tail4
    (if 3 ≤ root483Candidates3.length then search edge 3 [3467, 4799] root483Candidates3 8266 winner else winner) = _
  rw [if_neg (by decide)]
def root483Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root483Candidates4_checked : candidates edge 4 3467 4817 root483Tail5 winner.bound = root483Candidates4 := by
  unfold candidates
  have row : edge 4817 = (SortedNatTree.contains FastRows.row4817) := by
    change Data.row4817.contains = (SortedNatTree.contains FastRows.row4817)
    exact FastRows.row4817_checked.symm
  rw [row]
  decide
theorem root483Step4 : scan edge (search edge 3) 4 [3467] 3467 root483Tail4 winner =
    scan edge (search edge 3) 4 [3467] 3467 root483Tail5 winner := by
  rw [root483Tail4, scan, if_neg (by decide), root483Candidates4_checked]
  change scan edge (search edge 3) 4 [3467] 3467 root483Tail5
    (if 3 ≤ root483Candidates4.length then search edge 3 [3467, 4817] root483Candidates4 8284 winner else winner) = _
  rw [if_neg (by decide)]
def root483Candidates5 : List Nat := [6599, 7481]
set_option maxRecDepth 8192 in
theorem root483Candidates5_checked : candidates edge 4 3467 4919 root483Tail6 winner.bound = root483Candidates5 := by
  unfold candidates
  have row : edge 4919 = (SortedNatTree.contains FastRows.row4919) := by
    change Data.row4919.contains = (SortedNatTree.contains FastRows.row4919)
    exact FastRows.row4919_checked.symm
  rw [row]
  decide
theorem root483Step5 : scan edge (search edge 3) 4 [3467] 3467 root483Tail5 winner =
    scan edge (search edge 3) 4 [3467] 3467 root483Tail6 winner := by
  rw [root483Tail5, scan, if_neg (by decide), root483Candidates5_checked]
  change scan edge (search edge 3) 4 [3467] 3467 root483Tail6
    (if 3 ≤ root483Candidates5.length then search edge 3 [3467, 4919] root483Candidates5 8386 winner else winner) = _
  rw [if_neg (by decide)]
def root483Candidates6 : List Nat := [9059]
set_option maxRecDepth 8192 in
theorem root483Candidates6_checked : candidates edge 4 3467 5009 root483Tail7 winner.bound = root483Candidates6 := by
  unfold candidates
  have row : edge 5009 = (SortedNatTree.contains FastRows.row5009) := by
    change Data.row5009.contains = (SortedNatTree.contains FastRows.row5009)
    exact FastRows.row5009_checked.symm
  rw [row]
  decide
theorem root483Step6 : scan edge (search edge 3) 4 [3467] 3467 root483Tail6 winner =
    scan edge (search edge 3) 4 [3467] 3467 root483Tail7 winner := by
  rw [root483Tail6, scan, if_neg (by decide), root483Candidates6_checked]
  change scan edge (search edge 3) 4 [3467] 3467 root483Tail7
    (if 3 ≤ root483Candidates6.length then search edge 3 [3467, 5009] root483Candidates6 8476 winner else winner) = _
  rw [if_neg (by decide)]
def root483Candidates7 : List Nat := [7949, 9833, 10691]
set_option maxRecDepth 8192 in
theorem root483Candidates7_checked : candidates edge 4 3467 5231 root483Tail8 winner.bound = root483Candidates7 := by
  unfold candidates
  have row : edge 5231 = (SortedNatTree.contains FastRows.row5231) := by
    change Data.row5231.contains = (SortedNatTree.contains FastRows.row5231)
    exact FastRows.row5231_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root483Child7 : search edge 3 [3467, 5231] root483Candidates7 8698 winner = winner := by decide
theorem root483Step7 : scan edge (search edge 3) 4 [3467] 3467 root483Tail7 winner =
    scan edge (search edge 3) 4 [3467] 3467 root483Tail8 winner := by
  rw [root483Tail7, scan, if_neg (by decide), root483Candidates7_checked]
  change scan edge (search edge 3) 4 [3467] 3467 root483Tail8
    (if 3 ≤ root483Candidates7.length then search edge 3 [3467, 5231] root483Candidates7 8698 winner else winner) = _
  rw [if_pos (by decide), root483Child7]
def root483Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root483Candidates8_checked : candidates edge 4 3467 5273 root483Tail9 winner.bound = root483Candidates8 := by
  unfold candidates
  have row : edge 5273 = (SortedNatTree.contains FastRows.row5273) := by
    change Data.row5273.contains = (SortedNatTree.contains FastRows.row5273)
    exact FastRows.row5273_checked.symm
  rw [row]
  decide
theorem root483Step8 : scan edge (search edge 3) 4 [3467] 3467 root483Tail8 winner =
    scan edge (search edge 3) 4 [3467] 3467 root483Tail9 winner := by
  rw [root483Tail8, scan, if_neg (by decide), root483Candidates8_checked]
  change scan edge (search edge 3) 4 [3467] 3467 root483Tail9
    (if 3 ≤ root483Candidates8.length then search edge 3 [3467, 5273] root483Candidates8 8740 winner else winner) = _
  rw [if_neg (by decide)]
def root483Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root483Candidates9_checked : candidates edge 4 3467 5297 root483Tail10 winner.bound = root483Candidates9 := by
  unfold candidates
  have row : edge 5297 = (SortedNatTree.contains FastRows.row5297) := by
    change Data.row5297.contains = (SortedNatTree.contains FastRows.row5297)
    exact FastRows.row5297_checked.symm
  rw [row]
  decide
theorem root483Step9 : scan edge (search edge 3) 4 [3467] 3467 root483Tail9 winner =
    scan edge (search edge 3) 4 [3467] 3467 root483Tail10 winner := by
  rw [root483Tail9, scan, if_neg (by decide), root483Candidates9_checked]
  change scan edge (search edge 3) 4 [3467] 3467 root483Tail10
    (if 3 ≤ root483Candidates9.length then search edge 3 [3467, 5297] root483Candidates9 8764 winner else winner) = _
  rw [if_neg (by decide)]
def root483Candidates10 : List Nat := [5807]
set_option maxRecDepth 8192 in
theorem root483Candidates10_checked : candidates edge 4 3467 5471 root483Tail11 winner.bound = root483Candidates10 := by
  unfold candidates
  have row : edge 5471 = (SortedNatTree.contains FastRows.row5471) := by
    change Data.row5471.contains = (SortedNatTree.contains FastRows.row5471)
    exact FastRows.row5471_checked.symm
  rw [row]
  decide
theorem root483Step10 : scan edge (search edge 3) 4 [3467] 3467 root483Tail10 winner =
    scan edge (search edge 3) 4 [3467] 3467 root483Tail11 winner := by
  rw [root483Tail10, scan, if_neg (by decide), root483Candidates10_checked]
  change scan edge (search edge 3) 4 [3467] 3467 root483Tail11
    (if 3 ≤ root483Candidates10.length then search edge 3 [3467, 5471] root483Candidates10 8938 winner else winner) = _
  rw [if_neg (by decide)]
def root483Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root483Candidates11_checked : candidates edge 4 3467 5591 root483Tail12 winner.bound = root483Candidates11 := by
  unfold candidates
  have row : edge 5591 = (SortedNatTree.contains FastRows.row5591) := by
    change Data.row5591.contains = (SortedNatTree.contains FastRows.row5591)
    exact FastRows.row5591_checked.symm
  rw [row]
  decide
theorem root483Step11 : scan edge (search edge 3) 4 [3467] 3467 root483Tail11 winner =
    scan edge (search edge 3) 4 [3467] 3467 root483Tail12 winner := by
  rw [root483Tail11, scan, if_neg (by decide), root483Candidates11_checked]
  change scan edge (search edge 3) 4 [3467] 3467 root483Tail12
    (if 3 ≤ root483Candidates11.length then search edge 3 [3467, 5591] root483Candidates11 9058 winner else winner) = _
  rw [if_neg (by decide)]
theorem root483Scan12 : scan edge (search edge 3) 4 [3467] 3467 root483Tail12 winner = winner := by rw [root483Tail12, scan, if_pos (by decide)]
theorem root483Scan11 : scan edge (search edge 3) 4 [3467] 3467 root483Tail11 winner = winner :=
  (root483Step11).trans root483Scan12
theorem root483Scan10 : scan edge (search edge 3) 4 [3467] 3467 root483Tail10 winner = winner :=
  (root483Step10).trans root483Scan11
theorem root483Scan9 : scan edge (search edge 3) 4 [3467] 3467 root483Tail9 winner = winner :=
  (root483Step9).trans root483Scan10
theorem root483Scan8 : scan edge (search edge 3) 4 [3467] 3467 root483Tail8 winner = winner :=
  (root483Step8).trans root483Scan9
theorem root483Scan7 : scan edge (search edge 3) 4 [3467] 3467 root483Tail7 winner = winner :=
  (root483Step7).trans root483Scan8
theorem root483Scan6 : scan edge (search edge 3) 4 [3467] 3467 root483Tail6 winner = winner :=
  (root483Step6).trans root483Scan7
theorem root483Scan5 : scan edge (search edge 3) 4 [3467] 3467 root483Tail5 winner = winner :=
  (root483Step5).trans root483Scan6
theorem root483Scan4 : scan edge (search edge 3) 4 [3467] 3467 root483Tail4 winner = winner :=
  (root483Step4).trans root483Scan5
theorem root483Scan3 : scan edge (search edge 3) 4 [3467] 3467 root483Tail3 winner = winner :=
  (root483Step3).trans root483Scan4
theorem root483Scan2 : scan edge (search edge 3) 4 [3467] 3467 root483Tail2 winner = winner :=
  (root483Step2).trans root483Scan3
theorem root483Scan1 : scan edge (search edge 3) 4 [3467] 3467 root483Tail1 winner = winner :=
  (root483Step1).trans root483Scan2
theorem root483Scan0 : scan edge (search edge 3) 4 [3467] 3467 root483Tail0 winner = winner :=
  (root483Step0).trans root483Scan1
set_option maxRecDepth 4096 in
theorem root483Branch : search edge 4 [3467] root483Candidates 3467 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3467] 3467 root483Tail0 winner = winner
  exact root483Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
