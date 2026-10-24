import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root272Candidates : List Nat := Data.row1777
set_option maxRecDepth 8192 in
theorem root272Candidates_checked : candidates edge 5 0 1777 tail273 winner.bound = root272Candidates := by
  apply candidates_eq_row edge 5 0 1777 tail273 root272Candidates winner.bound tail273_ordered
  · decide
  · decide
  · funext q
    rfl
def root272Tail47 : List Nat := []
def root272Tail46 : List Nat := 24109 :: root272Tail47
def root272Tail45 : List Nat := 23671 :: root272Tail46
def root272Tail44 : List Nat := 23563 :: root272Tail45
def root272Tail43 : List Nat := 23209 :: root272Tail44
def root272Tail42 : List Nat := 21601 :: root272Tail43
def root272Tail41 : List Nat := 21067 :: root272Tail42
def root272Tail40 : List Nat := 20353 :: root272Tail41
def root272Tail39 : List Nat := 20143 :: root272Tail40
def root272Tail38 : List Nat := 19891 :: root272Tail39
def root272Tail37 : List Nat := 19699 :: root272Tail38
def root272Tail36 : List Nat := 18523 :: root272Tail37
def root272Tail35 : List Nat := 16183 :: root272Tail36
def root272Tail34 : List Nat := 15991 :: root272Tail35
def root272Tail33 : List Nat := 15271 :: root272Tail34
def root272Tail32 : List Nat := 15121 :: root272Tail33
def root272Tail31 : List Nat := 14683 :: root272Tail32
def root272Tail30 : List Nat := 13693 :: root272Tail31
def root272Tail29 : List Nat := 12979 :: root272Tail30
def root272Tail28 : List Nat := 12823 :: root272Tail29
def root272Tail27 : List Nat := 12487 :: root272Tail28
def root272Tail26 : List Nat := 12433 :: root272Tail27
def root272Tail25 : List Nat := 12211 :: root272Tail26
def root272Tail24 : List Nat := 11059 :: root272Tail25
def root272Tail23 : List Nat := 10597 :: root272Tail24
def root272Tail22 : List Nat := 10177 :: root272Tail23
def root272Tail21 : List Nat := 9787 :: root272Tail22
def root272Tail20 : List Nat := 9439 :: root272Tail21
def root272Tail19 : List Nat := 7867 :: root272Tail20
def root272Tail18 : List Nat := 7741 :: root272Tail19
def root272Tail17 : List Nat := 7603 :: root272Tail18
def root272Tail16 : List Nat := 7561 :: root272Tail17
def root272Tail15 : List Nat := 7489 :: root272Tail16
def root272Tail14 : List Nat := 7213 :: root272Tail15
def root272Tail13 : List Nat := 6277 :: root272Tail14
def root272Tail12 : List Nat := 6151 :: root272Tail13
def root272Tail11 : List Nat := 5839 :: root272Tail12
def root272Tail10 : List Nat := 5011 :: root272Tail11
def root272Tail9 : List Nat := 4621 :: root272Tail10
def root272Tail8 : List Nat := 3739 :: root272Tail9
def root272Tail7 : List Nat := 3463 :: root272Tail8
def root272Tail6 : List Nat := 3253 :: root272Tail7
def root272Tail5 : List Nat := 3121 :: root272Tail6
def root272Tail4 : List Nat := 2719 :: root272Tail5
def root272Tail3 : List Nat := 2647 :: root272Tail4
def root272Tail2 : List Nat := 2539 :: root272Tail3
def root272Tail1 : List Nat := 2371 :: root272Tail2
def root272Tail0 : List Nat := 2017 :: root272Tail1
def root272Candidates0 : List Nat := [4621, 7603, 9787]
set_option maxRecDepth 8192 in
theorem root272Candidates0_checked : candidates edge 4 1777 2017 root272Tail1 winner.bound = root272Candidates0 := by
  unfold candidates
  have row : edge 2017 = (SortedNatTree.contains FastRows.row2017) := by
    change Data.row2017.contains = (SortedNatTree.contains FastRows.row2017)
    exact FastRows.row2017_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root272Child0 : search edge 3 [1777, 2017] root272Candidates0 3794 winner = winner := by decide
theorem root272Step0 : scan edge (search edge 3) 4 [1777] 1777 root272Tail0 winner =
    scan edge (search edge 3) 4 [1777] 1777 root272Tail1 winner := by
  rw [root272Tail0, scan, if_neg (by decide), root272Candidates0_checked]
  change scan edge (search edge 3) 4 [1777] 1777 root272Tail1
    (if 3 ≤ root272Candidates0.length then search edge 3 [1777, 2017] root272Candidates0 3794 winner else winner) = _
  rw [if_pos (by decide), root272Child0]
def root272Candidates1 : List Nat := [15121]
set_option maxRecDepth 8192 in
theorem root272Candidates1_checked : candidates edge 4 1777 2371 root272Tail2 winner.bound = root272Candidates1 := by
  unfold candidates
  have row : edge 2371 = (SortedNatTree.contains FastRows.row2371) := by
    change Data.row2371.contains = (SortedNatTree.contains FastRows.row2371)
    exact FastRows.row2371_checked.symm
  rw [row]
  decide
theorem root272Step1 : scan edge (search edge 3) 4 [1777] 1777 root272Tail1 winner =
    scan edge (search edge 3) 4 [1777] 1777 root272Tail2 winner := by
  rw [root272Tail1, scan, if_neg (by decide), root272Candidates1_checked]
  change scan edge (search edge 3) 4 [1777] 1777 root272Tail2
    (if 3 ≤ root272Candidates1.length then search edge 3 [1777, 2371] root272Candidates1 4148 winner else winner) = _
  rw [if_neg (by decide)]
def root272Candidates2 : List Nat := [3463]
set_option maxRecDepth 8192 in
theorem root272Candidates2_checked : candidates edge 4 1777 2539 root272Tail3 winner.bound = root272Candidates2 := by
  unfold candidates
  have row : edge 2539 = (SortedNatTree.contains FastRows.row2539) := by
    change Data.row2539.contains = (SortedNatTree.contains FastRows.row2539)
    exact FastRows.row2539_checked.symm
  rw [row]
  decide
theorem root272Step2 : scan edge (search edge 3) 4 [1777] 1777 root272Tail2 winner =
    scan edge (search edge 3) 4 [1777] 1777 root272Tail3 winner := by
  rw [root272Tail2, scan, if_neg (by decide), root272Candidates2_checked]
  change scan edge (search edge 3) 4 [1777] 1777 root272Tail3
    (if 3 ≤ root272Candidates2.length then search edge 3 [1777, 2539] root272Candidates2 4316 winner else winner) = _
  rw [if_neg (by decide)]
def root272Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root272Candidates3_checked : candidates edge 4 1777 2647 root272Tail4 winner.bound = root272Candidates3 := by
  unfold candidates
  have row : edge 2647 = (SortedNatTree.contains FastRows.row2647) := by
    change Data.row2647.contains = (SortedNatTree.contains FastRows.row2647)
    exact FastRows.row2647_checked.symm
  rw [row]
  decide
theorem root272Step3 : scan edge (search edge 3) 4 [1777] 1777 root272Tail3 winner =
    scan edge (search edge 3) 4 [1777] 1777 root272Tail4 winner := by
  rw [root272Tail3, scan, if_neg (by decide), root272Candidates3_checked]
  change scan edge (search edge 3) 4 [1777] 1777 root272Tail4
    (if 3 ≤ root272Candidates3.length then search edge 3 [1777, 2647] root272Candidates3 4424 winner else winner) = _
  rw [if_neg (by decide)]
def root272Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root272Candidates4_checked : candidates edge 4 1777 2719 root272Tail5 winner.bound = root272Candidates4 := by
  unfold candidates
  have row : edge 2719 = (SortedNatTree.contains FastRows.row2719) := by
    change Data.row2719.contains = (SortedNatTree.contains FastRows.row2719)
    exact FastRows.row2719_checked.symm
  rw [row]
  decide
theorem root272Step4 : scan edge (search edge 3) 4 [1777] 1777 root272Tail4 winner =
    scan edge (search edge 3) 4 [1777] 1777 root272Tail5 winner := by
  rw [root272Tail4, scan, if_neg (by decide), root272Candidates4_checked]
  change scan edge (search edge 3) 4 [1777] 1777 root272Tail5
    (if 3 ≤ root272Candidates4.length then search edge 3 [1777, 2719] root272Candidates4 4496 winner else winner) = _
  rw [if_neg (by decide)]
def root272Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root272Candidates5_checked : candidates edge 4 1777 3121 root272Tail6 winner.bound = root272Candidates5 := by
  unfold candidates
  have row : edge 3121 = (SortedNatTree.contains FastRows.row3121) := by
    change Data.row3121.contains = (SortedNatTree.contains FastRows.row3121)
    exact FastRows.row3121_checked.symm
  rw [row]
  decide
theorem root272Step5 : scan edge (search edge 3) 4 [1777] 1777 root272Tail5 winner =
    scan edge (search edge 3) 4 [1777] 1777 root272Tail6 winner := by
  rw [root272Tail5, scan, if_neg (by decide), root272Candidates5_checked]
  change scan edge (search edge 3) 4 [1777] 1777 root272Tail6
    (if 3 ≤ root272Candidates5.length then search edge 3 [1777, 3121] root272Candidates5 4898 winner else winner) = _
  rw [if_neg (by decide)]
def root272Candidates6 : List Nat := [15121]
set_option maxRecDepth 8192 in
theorem root272Candidates6_checked : candidates edge 4 1777 3253 root272Tail7 winner.bound = root272Candidates6 := by
  unfold candidates
  have row : edge 3253 = (SortedNatTree.contains FastRows.row3253) := by
    change Data.row3253.contains = (SortedNatTree.contains FastRows.row3253)
    exact FastRows.row3253_checked.symm
  rw [row]
  decide
theorem root272Step6 : scan edge (search edge 3) 4 [1777] 1777 root272Tail6 winner =
    scan edge (search edge 3) 4 [1777] 1777 root272Tail7 winner := by
  rw [root272Tail6, scan, if_neg (by decide), root272Candidates6_checked]
  change scan edge (search edge 3) 4 [1777] 1777 root272Tail7
    (if 3 ≤ root272Candidates6.length then search edge 3 [1777, 3253] root272Candidates6 5030 winner else winner) = _
  rw [if_neg (by decide)]
def root272Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root272Candidates7_checked : candidates edge 4 1777 3463 root272Tail8 winner.bound = root272Candidates7 := by
  unfold candidates
  have row : edge 3463 = (SortedNatTree.contains FastRows.row3463) := by
    change Data.row3463.contains = (SortedNatTree.contains FastRows.row3463)
    exact FastRows.row3463_checked.symm
  rw [row]
  decide
theorem root272Step7 : scan edge (search edge 3) 4 [1777] 1777 root272Tail7 winner =
    scan edge (search edge 3) 4 [1777] 1777 root272Tail8 winner := by
  rw [root272Tail7, scan, if_neg (by decide), root272Candidates7_checked]
  change scan edge (search edge 3) 4 [1777] 1777 root272Tail8
    (if 3 ≤ root272Candidates7.length then search edge 3 [1777, 3463] root272Candidates7 5240 winner else winner) = _
  rw [if_neg (by decide)]
def root272Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root272Candidates8_checked : candidates edge 4 1777 3739 root272Tail9 winner.bound = root272Candidates8 := by
  unfold candidates
  have row : edge 3739 = (SortedNatTree.contains FastRows.row3739) := by
    change Data.row3739.contains = (SortedNatTree.contains FastRows.row3739)
    exact FastRows.row3739_checked.symm
  rw [row]
  decide
theorem root272Step8 : scan edge (search edge 3) 4 [1777] 1777 root272Tail8 winner =
    scan edge (search edge 3) 4 [1777] 1777 root272Tail9 winner := by
  rw [root272Tail8, scan, if_neg (by decide), root272Candidates8_checked]
  change scan edge (search edge 3) 4 [1777] 1777 root272Tail9
    (if 3 ≤ root272Candidates8.length then search edge 3 [1777, 3739] root272Candidates8 5516 winner else winner) = _
  rw [if_neg (by decide)]
def root272Candidates9 : List Nat := [7489]
set_option maxRecDepth 8192 in
theorem root272Candidates9_checked : candidates edge 4 1777 4621 root272Tail10 winner.bound = root272Candidates9 := by
  unfold candidates
  have row : edge 4621 = (SortedNatTree.contains FastRows.row4621) := by
    change Data.row4621.contains = (SortedNatTree.contains FastRows.row4621)
    exact FastRows.row4621_checked.symm
  rw [row]
  decide
theorem root272Step9 : scan edge (search edge 3) 4 [1777] 1777 root272Tail9 winner =
    scan edge (search edge 3) 4 [1777] 1777 root272Tail10 winner := by
  rw [root272Tail9, scan, if_neg (by decide), root272Candidates9_checked]
  change scan edge (search edge 3) 4 [1777] 1777 root272Tail10
    (if 3 ≤ root272Candidates9.length then search edge 3 [1777, 4621] root272Candidates9 6398 winner else winner) = _
  rw [if_neg (by decide)]
def root272Candidates10 : List Nat := [12823]
set_option maxRecDepth 8192 in
theorem root272Candidates10_checked : candidates edge 4 1777 5011 root272Tail11 winner.bound = root272Candidates10 := by
  unfold candidates
  have row : edge 5011 = (SortedNatTree.contains FastRows.row5011) := by
    change Data.row5011.contains = (SortedNatTree.contains FastRows.row5011)
    exact FastRows.row5011_checked.symm
  rw [row]
  decide
theorem root272Step10 : scan edge (search edge 3) 4 [1777] 1777 root272Tail10 winner =
    scan edge (search edge 3) 4 [1777] 1777 root272Tail11 winner := by
  rw [root272Tail10, scan, if_neg (by decide), root272Candidates10_checked]
  change scan edge (search edge 3) 4 [1777] 1777 root272Tail11
    (if 3 ≤ root272Candidates10.length then search edge 3 [1777, 5011] root272Candidates10 6788 winner else winner) = _
  rw [if_neg (by decide)]
def root272Candidates11 : List Nat := [10597, 15121]
set_option maxRecDepth 8192 in
theorem root272Candidates11_checked : candidates edge 4 1777 5839 root272Tail12 winner.bound = root272Candidates11 := by
  unfold candidates
  have row : edge 5839 = (SortedNatTree.contains FastRows.row5839) := by
    change Data.row5839.contains = (SortedNatTree.contains FastRows.row5839)
    exact FastRows.row5839_checked.symm
  rw [row]
  decide
theorem root272Step11 : scan edge (search edge 3) 4 [1777] 1777 root272Tail11 winner =
    scan edge (search edge 3) 4 [1777] 1777 root272Tail12 winner := by
  rw [root272Tail11, scan, if_neg (by decide), root272Candidates11_checked]
  change scan edge (search edge 3) 4 [1777] 1777 root272Tail12
    (if 3 ≤ root272Candidates11.length then search edge 3 [1777, 5839] root272Candidates11 7616 winner else winner) = _
  rw [if_neg (by decide)]
theorem root272Scan12 : scan edge (search edge 3) 4 [1777] 1777 root272Tail12 winner = winner := by rw [root272Tail12, scan, if_pos (by decide)]
theorem root272Scan11 : scan edge (search edge 3) 4 [1777] 1777 root272Tail11 winner = winner :=
  (root272Step11).trans root272Scan12
theorem root272Scan10 : scan edge (search edge 3) 4 [1777] 1777 root272Tail10 winner = winner :=
  (root272Step10).trans root272Scan11
theorem root272Scan9 : scan edge (search edge 3) 4 [1777] 1777 root272Tail9 winner = winner :=
  (root272Step9).trans root272Scan10
theorem root272Scan8 : scan edge (search edge 3) 4 [1777] 1777 root272Tail8 winner = winner :=
  (root272Step8).trans root272Scan9
theorem root272Scan7 : scan edge (search edge 3) 4 [1777] 1777 root272Tail7 winner = winner :=
  (root272Step7).trans root272Scan8
theorem root272Scan6 : scan edge (search edge 3) 4 [1777] 1777 root272Tail6 winner = winner :=
  (root272Step6).trans root272Scan7
theorem root272Scan5 : scan edge (search edge 3) 4 [1777] 1777 root272Tail5 winner = winner :=
  (root272Step5).trans root272Scan6
theorem root272Scan4 : scan edge (search edge 3) 4 [1777] 1777 root272Tail4 winner = winner :=
  (root272Step4).trans root272Scan5
theorem root272Scan3 : scan edge (search edge 3) 4 [1777] 1777 root272Tail3 winner = winner :=
  (root272Step3).trans root272Scan4
theorem root272Scan2 : scan edge (search edge 3) 4 [1777] 1777 root272Tail2 winner = winner :=
  (root272Step2).trans root272Scan3
theorem root272Scan1 : scan edge (search edge 3) 4 [1777] 1777 root272Tail1 winner = winner :=
  (root272Step1).trans root272Scan2
theorem root272Scan0 : scan edge (search edge 3) 4 [1777] 1777 root272Tail0 winner = winner :=
  (root272Step0).trans root272Scan1
set_option maxRecDepth 4096 in
theorem root272Branch : search edge 4 [1777] root272Candidates 1777 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1777] 1777 root272Tail0 winner = winner
  exact root272Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
