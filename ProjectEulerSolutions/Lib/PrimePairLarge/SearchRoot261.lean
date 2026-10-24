import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root261Candidates : List Nat := Data.row1693
set_option maxRecDepth 8192 in
theorem root261Candidates_checked : candidates edge 5 0 1693 tail262 winner.bound = root261Candidates := by
  apply candidates_eq_row edge 5 0 1693 tail262 root261Candidates winner.bound tail262_ordered
  · decide
  · decide
  · funext q
    rfl
def root261Tail51 : List Nat := []
def root261Tail50 : List Nat := 24121 :: root261Tail51
def root261Tail49 : List Nat := 23899 :: root261Tail50
def root261Tail48 : List Nat := 23227 :: root261Tail49
def root261Tail47 : List Nat := 22921 :: root261Tail48
def root261Tail46 : List Nat := 22147 :: root261Tail47
def root261Tail45 : List Nat := 22051 :: root261Tail46
def root261Tail44 : List Nat := 21601 :: root261Tail45
def root261Tail43 : List Nat := 20707 :: root261Tail44
def root261Tail42 : List Nat := 20563 :: root261Tail43
def root261Tail41 : List Nat := 20521 :: root261Tail42
def root261Tail40 : List Nat := 20341 :: root261Tail41
def root261Tail39 : List Nat := 20269 :: root261Tail40
def root261Tail38 : List Nat := 19753 :: root261Tail39
def root261Tail37 : List Nat := 18661 :: root261Tail38
def root261Tail36 : List Nat := 18439 :: root261Tail37
def root261Tail35 : List Nat := 16981 :: root261Tail36
def root261Tail34 : List Nat := 16759 :: root261Tail35
def root261Tail33 : List Nat := 15607 :: root261Tail34
def root261Tail32 : List Nat := 15541 :: root261Tail33
def root261Tail31 : List Nat := 15361 :: root261Tail32
def root261Tail30 : List Nat := 14797 :: root261Tail31
def root261Tail29 : List Nat := 14071 :: root261Tail30
def root261Tail28 : List Nat := 13873 :: root261Tail29
def root261Tail27 : List Nat := 13681 :: root261Tail28
def root261Tail26 : List Nat := 13033 :: root261Tail27
def root261Tail25 : List Nat := 12967 :: root261Tail26
def root261Tail24 : List Nat := 12697 :: root261Tail25
def root261Tail23 : List Nat := 12487 :: root261Tail24
def root261Tail22 : List Nat := 11953 :: root261Tail23
def root261Tail21 : List Nat := 11239 :: root261Tail22
def root261Tail20 : List Nat := 10459 :: root261Tail21
def root261Tail19 : List Nat := 9337 :: root261Tail20
def root261Tail18 : List Nat := 9283 :: root261Tail19
def root261Tail17 : List Nat := 7927 :: root261Tail18
def root261Tail16 : List Nat := 7489 :: root261Tail17
def root261Tail15 : List Nat := 7477 :: root261Tail16
def root261Tail14 : List Nat := 7309 :: root261Tail15
def root261Tail13 : List Nat := 6991 :: root261Tail14
def root261Tail12 : List Nat := 6217 :: root261Tail13
def root261Tail11 : List Nat := 4639 :: root261Tail12
def root261Tail10 : List Nat := 4441 :: root261Tail11
def root261Tail9 : List Nat := 4327 :: root261Tail10
def root261Tail8 : List Nat := 3919 :: root261Tail9
def root261Tail7 : List Nat := 3739 :: root261Tail8
def root261Tail6 : List Nat := 3373 :: root261Tail7
def root261Tail5 : List Nat := 3181 :: root261Tail6
def root261Tail4 : List Nat := 2689 :: root261Tail5
def root261Tail3 : List Nat := 2593 :: root261Tail4
def root261Tail2 : List Nat := 2551 :: root261Tail3
def root261Tail1 : List Nat := 2371 :: root261Tail2
def root261Tail0 : List Nat := 1753 :: root261Tail1
def root261Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root261Candidates0_checked : candidates edge 4 1693 1753 root261Tail1 winner.bound = root261Candidates0 := by
  unfold candidates
  have row : edge 1753 = (SortedNatTree.contains FastRows.row1753) := by
    change Data.row1753.contains = (SortedNatTree.contains FastRows.row1753)
    exact FastRows.row1753_checked.symm
  rw [row]
  decide
theorem root261Step0 : scan edge (search edge 3) 4 [1693] 1693 root261Tail0 winner =
    scan edge (search edge 3) 4 [1693] 1693 root261Tail1 winner := by
  rw [root261Tail0, scan, if_neg (by decide), root261Candidates0_checked]
  change scan edge (search edge 3) 4 [1693] 1693 root261Tail1
    (if 3 ≤ root261Candidates0.length then search edge 3 [1693, 1753] root261Candidates0 3446 winner else winner) = _
  rw [if_neg (by decide)]
def root261Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root261Candidates1_checked : candidates edge 4 1693 2371 root261Tail2 winner.bound = root261Candidates1 := by
  unfold candidates
  have row : edge 2371 = (SortedNatTree.contains FastRows.row2371) := by
    change Data.row2371.contains = (SortedNatTree.contains FastRows.row2371)
    exact FastRows.row2371_checked.symm
  rw [row]
  decide
theorem root261Step1 : scan edge (search edge 3) 4 [1693] 1693 root261Tail1 winner =
    scan edge (search edge 3) 4 [1693] 1693 root261Tail2 winner := by
  rw [root261Tail1, scan, if_neg (by decide), root261Candidates1_checked]
  change scan edge (search edge 3) 4 [1693] 1693 root261Tail2
    (if 3 ≤ root261Candidates1.length then search edge 3 [1693, 2371] root261Candidates1 4064 winner else winner) = _
  rw [if_neg (by decide)]
def root261Candidates2 : List Nat := [11953, 13681]
set_option maxRecDepth 8192 in
theorem root261Candidates2_checked : candidates edge 4 1693 2551 root261Tail3 winner.bound = root261Candidates2 := by
  unfold candidates
  have row : edge 2551 = (SortedNatTree.contains FastRows.row2551) := by
    change Data.row2551.contains = (SortedNatTree.contains FastRows.row2551)
    exact FastRows.row2551_checked.symm
  rw [row]
  decide
theorem root261Step2 : scan edge (search edge 3) 4 [1693] 1693 root261Tail2 winner =
    scan edge (search edge 3) 4 [1693] 1693 root261Tail3 winner := by
  rw [root261Tail2, scan, if_neg (by decide), root261Candidates2_checked]
  change scan edge (search edge 3) 4 [1693] 1693 root261Tail3
    (if 3 ≤ root261Candidates2.length then search edge 3 [1693, 2551] root261Candidates2 4244 winner else winner) = _
  rw [if_neg (by decide)]
def root261Candidates3 : List Nat := [3373, 7927, 12967]
set_option maxRecDepth 8192 in
theorem root261Candidates3_checked : candidates edge 4 1693 2593 root261Tail4 winner.bound = root261Candidates3 := by
  unfold candidates
  have row : edge 2593 = (SortedNatTree.contains FastRows.row2593) := by
    change Data.row2593.contains = (SortedNatTree.contains FastRows.row2593)
    exact FastRows.row2593_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root261Child3 : search edge 3 [1693, 2593] root261Candidates3 4286 winner = winner := by decide
theorem root261Step3 : scan edge (search edge 3) 4 [1693] 1693 root261Tail3 winner =
    scan edge (search edge 3) 4 [1693] 1693 root261Tail4 winner := by
  rw [root261Tail3, scan, if_neg (by decide), root261Candidates3_checked]
  change scan edge (search edge 3) 4 [1693] 1693 root261Tail4
    (if 3 ≤ root261Candidates3.length then search edge 3 [1693, 2593] root261Candidates3 4286 winner else winner) = _
  rw [if_pos (by decide), root261Child3]
def root261Candidates4 : List Nat := [7477, 12697]
set_option maxRecDepth 8192 in
theorem root261Candidates4_checked : candidates edge 4 1693 2689 root261Tail5 winner.bound = root261Candidates4 := by
  unfold candidates
  have row : edge 2689 = (SortedNatTree.contains FastRows.row2689) := by
    change Data.row2689.contains = (SortedNatTree.contains FastRows.row2689)
    exact FastRows.row2689_checked.symm
  rw [row]
  decide
theorem root261Step4 : scan edge (search edge 3) 4 [1693] 1693 root261Tail4 winner =
    scan edge (search edge 3) 4 [1693] 1693 root261Tail5 winner := by
  rw [root261Tail4, scan, if_neg (by decide), root261Candidates4_checked]
  change scan edge (search edge 3) 4 [1693] 1693 root261Tail5
    (if 3 ≤ root261Candidates4.length then search edge 3 [1693, 2689] root261Candidates4 4382 winner else winner) = _
  rw [if_neg (by decide)]
def root261Candidates5 : List Nat := [15361]
set_option maxRecDepth 8192 in
theorem root261Candidates5_checked : candidates edge 4 1693 3181 root261Tail6 winner.bound = root261Candidates5 := by
  unfold candidates
  have row : edge 3181 = (SortedNatTree.contains FastRows.row3181) := by
    change Data.row3181.contains = (SortedNatTree.contains FastRows.row3181)
    exact FastRows.row3181_checked.symm
  rw [row]
  decide
theorem root261Step5 : scan edge (search edge 3) 4 [1693] 1693 root261Tail5 winner =
    scan edge (search edge 3) 4 [1693] 1693 root261Tail6 winner := by
  rw [root261Tail5, scan, if_neg (by decide), root261Candidates5_checked]
  change scan edge (search edge 3) 4 [1693] 1693 root261Tail6
    (if 3 ≤ root261Candidates5.length then search edge 3 [1693, 3181] root261Candidates5 4874 winner else winner) = _
  rw [if_neg (by decide)]
def root261Candidates6 : List Nat := [12697, 13681]
set_option maxRecDepth 8192 in
theorem root261Candidates6_checked : candidates edge 4 1693 3373 root261Tail7 winner.bound = root261Candidates6 := by
  unfold candidates
  have row : edge 3373 = (SortedNatTree.contains FastRows.row3373) := by
    change Data.row3373.contains = (SortedNatTree.contains FastRows.row3373)
    exact FastRows.row3373_checked.symm
  rw [row]
  decide
theorem root261Step6 : scan edge (search edge 3) 4 [1693] 1693 root261Tail6 winner =
    scan edge (search edge 3) 4 [1693] 1693 root261Tail7 winner := by
  rw [root261Tail6, scan, if_neg (by decide), root261Candidates6_checked]
  change scan edge (search edge 3) 4 [1693] 1693 root261Tail7
    (if 3 ≤ root261Candidates6.length then search edge 3 [1693, 3373] root261Candidates6 5066 winner else winner) = _
  rw [if_neg (by decide)]
def root261Candidates7 : List Nat := [14071, 20563]
set_option maxRecDepth 8192 in
theorem root261Candidates7_checked : candidates edge 4 1693 3739 root261Tail8 winner.bound = root261Candidates7 := by
  unfold candidates
  have row : edge 3739 = (SortedNatTree.contains FastRows.row3739) := by
    change Data.row3739.contains = (SortedNatTree.contains FastRows.row3739)
    exact FastRows.row3739_checked.symm
  rw [row]
  decide
theorem root261Step7 : scan edge (search edge 3) 4 [1693] 1693 root261Tail7 winner =
    scan edge (search edge 3) 4 [1693] 1693 root261Tail8 winner := by
  rw [root261Tail7, scan, if_neg (by decide), root261Candidates7_checked]
  change scan edge (search edge 3) 4 [1693] 1693 root261Tail8
    (if 3 ≤ root261Candidates7.length then search edge 3 [1693, 3739] root261Candidates7 5432 winner else winner) = _
  rw [if_neg (by decide)]
def root261Candidates8 : List Nat := [7309, 7489]
set_option maxRecDepth 8192 in
theorem root261Candidates8_checked : candidates edge 4 1693 3919 root261Tail9 winner.bound = root261Candidates8 := by
  unfold candidates
  have row : edge 3919 = (SortedNatTree.contains FastRows.row3919) := by
    change Data.row3919.contains = (SortedNatTree.contains FastRows.row3919)
    exact FastRows.row3919_checked.symm
  rw [row]
  decide
theorem root261Step8 : scan edge (search edge 3) 4 [1693] 1693 root261Tail8 winner =
    scan edge (search edge 3) 4 [1693] 1693 root261Tail9 winner := by
  rw [root261Tail8, scan, if_neg (by decide), root261Candidates8_checked]
  change scan edge (search edge 3) 4 [1693] 1693 root261Tail9
    (if 3 ≤ root261Candidates8.length then search edge 3 [1693, 3919] root261Candidates8 5612 winner else winner) = _
  rw [if_neg (by decide)]
def root261Candidates9 : List Nat := [9337, 12967]
set_option maxRecDepth 8192 in
theorem root261Candidates9_checked : candidates edge 4 1693 4327 root261Tail10 winner.bound = root261Candidates9 := by
  unfold candidates
  have row : edge 4327 = (SortedNatTree.contains FastRows.row4327) := by
    change Data.row4327.contains = (SortedNatTree.contains FastRows.row4327)
    exact FastRows.row4327_checked.symm
  rw [row]
  decide
theorem root261Step9 : scan edge (search edge 3) 4 [1693] 1693 root261Tail9 winner =
    scan edge (search edge 3) 4 [1693] 1693 root261Tail10 winner := by
  rw [root261Tail9, scan, if_neg (by decide), root261Candidates9_checked]
  change scan edge (search edge 3) 4 [1693] 1693 root261Tail10
    (if 3 ≤ root261Candidates9.length then search edge 3 [1693, 4327] root261Candidates9 6020 winner else winner) = _
  rw [if_neg (by decide)]
def root261Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root261Candidates10_checked : candidates edge 4 1693 4441 root261Tail11 winner.bound = root261Candidates10 := by
  unfold candidates
  have row : edge 4441 = (SortedNatTree.contains FastRows.row4441) := by
    change Data.row4441.contains = (SortedNatTree.contains FastRows.row4441)
    exact FastRows.row4441_checked.symm
  rw [row]
  decide
theorem root261Step10 : scan edge (search edge 3) 4 [1693] 1693 root261Tail10 winner =
    scan edge (search edge 3) 4 [1693] 1693 root261Tail11 winner := by
  rw [root261Tail10, scan, if_neg (by decide), root261Candidates10_checked]
  change scan edge (search edge 3) 4 [1693] 1693 root261Tail11
    (if 3 ≤ root261Candidates10.length then search edge 3 [1693, 4441] root261Candidates10 6134 winner else winner) = _
  rw [if_neg (by decide)]
def root261Candidates11 : List Nat := [14797]
set_option maxRecDepth 8192 in
theorem root261Candidates11_checked : candidates edge 4 1693 4639 root261Tail12 winner.bound = root261Candidates11 := by
  unfold candidates
  have row : edge 4639 = (SortedNatTree.contains FastRows.row4639) := by
    change Data.row4639.contains = (SortedNatTree.contains FastRows.row4639)
    exact FastRows.row4639_checked.symm
  rw [row]
  decide
theorem root261Step11 : scan edge (search edge 3) 4 [1693] 1693 root261Tail11 winner =
    scan edge (search edge 3) 4 [1693] 1693 root261Tail12 winner := by
  rw [root261Tail11, scan, if_neg (by decide), root261Candidates11_checked]
  change scan edge (search edge 3) 4 [1693] 1693 root261Tail12
    (if 3 ≤ root261Candidates11.length then search edge 3 [1693, 4639] root261Candidates11 6332 winner else winner) = _
  rw [if_neg (by decide)]
theorem root261Scan12 : scan edge (search edge 3) 4 [1693] 1693 root261Tail12 winner = winner := by rw [root261Tail12, scan, if_pos (by decide)]
theorem root261Scan11 : scan edge (search edge 3) 4 [1693] 1693 root261Tail11 winner = winner :=
  (root261Step11).trans root261Scan12
theorem root261Scan10 : scan edge (search edge 3) 4 [1693] 1693 root261Tail10 winner = winner :=
  (root261Step10).trans root261Scan11
theorem root261Scan9 : scan edge (search edge 3) 4 [1693] 1693 root261Tail9 winner = winner :=
  (root261Step9).trans root261Scan10
theorem root261Scan8 : scan edge (search edge 3) 4 [1693] 1693 root261Tail8 winner = winner :=
  (root261Step8).trans root261Scan9
theorem root261Scan7 : scan edge (search edge 3) 4 [1693] 1693 root261Tail7 winner = winner :=
  (root261Step7).trans root261Scan8
theorem root261Scan6 : scan edge (search edge 3) 4 [1693] 1693 root261Tail6 winner = winner :=
  (root261Step6).trans root261Scan7
theorem root261Scan5 : scan edge (search edge 3) 4 [1693] 1693 root261Tail5 winner = winner :=
  (root261Step5).trans root261Scan6
theorem root261Scan4 : scan edge (search edge 3) 4 [1693] 1693 root261Tail4 winner = winner :=
  (root261Step4).trans root261Scan5
theorem root261Scan3 : scan edge (search edge 3) 4 [1693] 1693 root261Tail3 winner = winner :=
  (root261Step3).trans root261Scan4
theorem root261Scan2 : scan edge (search edge 3) 4 [1693] 1693 root261Tail2 winner = winner :=
  (root261Step2).trans root261Scan3
theorem root261Scan1 : scan edge (search edge 3) 4 [1693] 1693 root261Tail1 winner = winner :=
  (root261Step1).trans root261Scan2
theorem root261Scan0 : scan edge (search edge 3) 4 [1693] 1693 root261Tail0 winner = winner :=
  (root261Step0).trans root261Scan1
set_option maxRecDepth 4096 in
theorem root261Branch : search edge 4 [1693] root261Candidates 1693 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1693] 1693 root261Tail0 winner = winner
  exact root261Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
