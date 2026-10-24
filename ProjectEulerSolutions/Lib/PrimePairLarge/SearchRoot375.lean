import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root375Candidates : List Nat := Data.row2593
set_option maxRecDepth 8192 in
theorem root375Candidates_checked : candidates edge 5 0 2593 tail376 winner.bound = root375Candidates := by
  apply candidates_eq_row edge 5 0 2593 tail376 root375Candidates winner.bound tail376_ordered
  · decide
  · decide
  · funext q
    rfl
def root375Tail54 : List Nat := []
def root375Tail53 : List Nat := 23293 :: root375Tail54
def root375Tail52 : List Nat := 23209 :: root375Tail53
def root375Tail51 : List Nat := 22291 :: root375Tail52
def root375Tail50 : List Nat := 22027 :: root375Tail51
def root375Tail49 : List Nat := 21943 :: root375Tail50
def root375Tail48 : List Nat := 21613 :: root375Tail49
def root375Tail47 : List Nat := 21523 :: root375Tail48
def root375Tail46 : List Nat := 20389 :: root375Tail47
def root375Tail45 : List Nat := 19861 :: root375Tail46
def root375Tail44 : List Nat := 19507 :: root375Tail45
def root375Tail43 : List Nat := 19441 :: root375Tail44
def root375Tail42 : List Nat := 18427 :: root375Tail43
def root375Tail41 : List Nat := 18013 :: root375Tail42
def root375Tail40 : List Nat := 17293 :: root375Tail41
def root375Tail39 : List Nat := 17239 :: root375Tail40
def root375Tail38 : List Nat := 17209 :: root375Tail39
def root375Tail37 : List Nat := 16903 :: root375Tail38
def root375Tail36 : List Nat := 16699 :: root375Tail37
def root375Tail35 : List Nat := 16411 :: root375Tail36
def root375Tail34 : List Nat := 15619 :: root375Tail35
def root375Tail33 : List Nat := 14851 :: root375Tail34
def root375Tail32 : List Nat := 13591 :: root375Tail33
def root375Tail31 : List Nat := 12967 :: root375Tail32
def root375Tail30 : List Nat := 12841 :: root375Tail31
def root375Tail29 : List Nat := 12637 :: root375Tail30
def root375Tail28 : List Nat := 12541 :: root375Tail29
def root375Tail27 : List Nat := 12421 :: root375Tail28
def root375Tail26 : List Nat := 10789 :: root375Tail27
def root375Tail25 : List Nat := 10111 :: root375Tail26
def root375Tail24 : List Nat := 9721 :: root375Tail25
def root375Tail23 : List Nat := 9547 :: root375Tail24
def root375Tail22 : List Nat := 9421 :: root375Tail23
def root375Tail21 : List Nat := 9181 :: root375Tail22
def root375Tail20 : List Nat := 8293 :: root375Tail21
def root375Tail19 : List Nat := 7927 :: root375Tail20
def root375Tail18 : List Nat := 7753 :: root375Tail19
def root375Tail17 : List Nat := 7573 :: root375Tail18
def root375Tail16 : List Nat := 7039 :: root375Tail17
def root375Tail15 : List Nat := 7027 :: root375Tail16
def root375Tail14 : List Nat := 5683 :: root375Tail15
def root375Tail13 : List Nat := 5443 :: root375Tail14
def root375Tail12 : List Nat := 5281 :: root375Tail13
def root375Tail11 : List Nat := 5101 :: root375Tail12
def root375Tail10 : List Nat := 5011 :: root375Tail11
def root375Tail9 : List Nat := 3847 :: root375Tail10
def root375Tail8 : List Nat := 3709 :: root375Tail9
def root375Tail7 : List Nat := 3373 :: root375Tail8
def root375Tail6 : List Nat := 3307 :: root375Tail7
def root375Tail5 : List Nat := 3169 :: root375Tail6
def root375Tail4 : List Nat := 2971 :: root375Tail5
def root375Tail3 : List Nat := 2887 :: root375Tail4
def root375Tail2 : List Nat := 2719 :: root375Tail3
def root375Tail1 : List Nat := 2713 :: root375Tail2
def root375Tail0 : List Nat := 2617 :: root375Tail1
def root375Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root375Candidates0_checked : candidates edge 4 2593 2617 root375Tail1 winner.bound = root375Candidates0 := by
  unfold candidates
  have row : edge 2617 = (SortedNatTree.contains FastRows.row2617) := by
    change Data.row2617.contains = (SortedNatTree.contains FastRows.row2617)
    exact FastRows.row2617_checked.symm
  rw [row]
  decide
theorem root375Step0 : scan edge (search edge 3) 4 [2593] 2593 root375Tail0 winner =
    scan edge (search edge 3) 4 [2593] 2593 root375Tail1 winner := by
  rw [root375Tail0, scan, if_neg (by decide), root375Candidates0_checked]
  change scan edge (search edge 3) 4 [2593] 2593 root375Tail1
    (if 3 ≤ root375Candidates0.length then search edge 3 [2593, 2617] root375Candidates0 5210 winner else winner) = _
  rw [if_neg (by decide)]
def root375Candidates1 : List Nat := [5443]
set_option maxRecDepth 8192 in
theorem root375Candidates1_checked : candidates edge 4 2593 2713 root375Tail2 winner.bound = root375Candidates1 := by
  unfold candidates
  have row : edge 2713 = (SortedNatTree.contains FastRows.row2713) := by
    change Data.row2713.contains = (SortedNatTree.contains FastRows.row2713)
    exact FastRows.row2713_checked.symm
  rw [row]
  decide
theorem root375Step1 : scan edge (search edge 3) 4 [2593] 2593 root375Tail1 winner =
    scan edge (search edge 3) 4 [2593] 2593 root375Tail2 winner := by
  rw [root375Tail1, scan, if_neg (by decide), root375Candidates1_checked]
  change scan edge (search edge 3) 4 [2593] 2593 root375Tail2
    (if 3 ≤ root375Candidates1.length then search edge 3 [2593, 2713] root375Candidates1 5306 winner else winner) = _
  rw [if_neg (by decide)]
def root375Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root375Candidates2_checked : candidates edge 4 2593 2719 root375Tail3 winner.bound = root375Candidates2 := by
  unfold candidates
  have row : edge 2719 = (SortedNatTree.contains FastRows.row2719) := by
    change Data.row2719.contains = (SortedNatTree.contains FastRows.row2719)
    exact FastRows.row2719_checked.symm
  rw [row]
  decide
theorem root375Step2 : scan edge (search edge 3) 4 [2593] 2593 root375Tail2 winner =
    scan edge (search edge 3) 4 [2593] 2593 root375Tail3 winner := by
  rw [root375Tail2, scan, if_neg (by decide), root375Candidates2_checked]
  change scan edge (search edge 3) 4 [2593] 2593 root375Tail3
    (if 3 ≤ root375Candidates2.length then search edge 3 [2593, 2719] root375Candidates2 5312 winner else winner) = _
  rw [if_neg (by decide)]
def root375Candidates3 : List Nat := [7573, 12967, 14851]
set_option maxRecDepth 8192 in
theorem root375Candidates3_checked : candidates edge 4 2593 2887 root375Tail4 winner.bound = root375Candidates3 := by
  unfold candidates
  have row : edge 2887 = (SortedNatTree.contains FastRows.row2887) := by
    change Data.row2887.contains = (SortedNatTree.contains FastRows.row2887)
    exact FastRows.row2887_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root375Child3 : search edge 3 [2593, 2887] root375Candidates3 5480 winner = winner := by decide
theorem root375Step3 : scan edge (search edge 3) 4 [2593] 2593 root375Tail3 winner =
    scan edge (search edge 3) 4 [2593] 2593 root375Tail4 winner := by
  rw [root375Tail3, scan, if_neg (by decide), root375Candidates3_checked]
  change scan edge (search edge 3) 4 [2593] 2593 root375Tail4
    (if 3 ≤ root375Candidates3.length then search edge 3 [2593, 2887] root375Candidates3 5480 winner else winner) = _
  rw [if_pos (by decide), root375Child3]
def root375Candidates4 : List Nat := [3709, 7573, 12421, 13591]
set_option maxRecDepth 8192 in
theorem root375Candidates4_checked : candidates edge 4 2593 2971 root375Tail5 winner.bound = root375Candidates4 := by
  unfold candidates
  have row : edge 2971 = (SortedNatTree.contains FastRows.row2971) := by
    change Data.row2971.contains = (SortedNatTree.contains FastRows.row2971)
    exact FastRows.row2971_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root375Child4 : search edge 3 [2593, 2971] root375Candidates4 5564 winner = winner := by decide
theorem root375Step4 : scan edge (search edge 3) 4 [2593] 2593 root375Tail4 winner =
    scan edge (search edge 3) 4 [2593] 2593 root375Tail5 winner := by
  rw [root375Tail4, scan, if_neg (by decide), root375Candidates4_checked]
  change scan edge (search edge 3) 4 [2593] 2593 root375Tail5
    (if 3 ≤ root375Candidates4.length then search edge 3 [2593, 2971] root375Candidates4 5564 winner else winner) = _
  rw [if_pos (by decide), root375Child4]
def root375Candidates5 : List Nat := [5101]
set_option maxRecDepth 8192 in
theorem root375Candidates5_checked : candidates edge 4 2593 3169 root375Tail6 winner.bound = root375Candidates5 := by
  unfold candidates
  have row : edge 3169 = (SortedNatTree.contains FastRows.row3169) := by
    change Data.row3169.contains = (SortedNatTree.contains FastRows.row3169)
    exact FastRows.row3169_checked.symm
  rw [row]
  decide
theorem root375Step5 : scan edge (search edge 3) 4 [2593] 2593 root375Tail5 winner =
    scan edge (search edge 3) 4 [2593] 2593 root375Tail6 winner := by
  rw [root375Tail5, scan, if_neg (by decide), root375Candidates5_checked]
  change scan edge (search edge 3) 4 [2593] 2593 root375Tail6
    (if 3 ≤ root375Candidates5.length then search edge 3 [2593, 3169] root375Candidates5 5762 winner else winner) = _
  rw [if_neg (by decide)]
def root375Candidates6 : List Nat := [5443, 7039, 9181]
set_option maxRecDepth 8192 in
theorem root375Candidates6_checked : candidates edge 4 2593 3307 root375Tail7 winner.bound = root375Candidates6 := by
  unfold candidates
  have row : edge 3307 = (SortedNatTree.contains FastRows.row3307) := by
    change Data.row3307.contains = (SortedNatTree.contains FastRows.row3307)
    exact FastRows.row3307_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root375Child6 : search edge 3 [2593, 3307] root375Candidates6 5900 winner = winner := by decide
theorem root375Step6 : scan edge (search edge 3) 4 [2593] 2593 root375Tail6 winner =
    scan edge (search edge 3) 4 [2593] 2593 root375Tail7 winner := by
  rw [root375Tail6, scan, if_neg (by decide), root375Candidates6_checked]
  change scan edge (search edge 3) 4 [2593] 2593 root375Tail7
    (if 3 ≤ root375Candidates6.length then search edge 3 [2593, 3307] root375Candidates6 5900 winner else winner) = _
  rw [if_pos (by decide), root375Child6]
def root375Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root375Candidates7_checked : candidates edge 4 2593 3373 root375Tail8 winner.bound = root375Candidates7 := by
  unfold candidates
  have row : edge 3373 = (SortedNatTree.contains FastRows.row3373) := by
    change Data.row3373.contains = (SortedNatTree.contains FastRows.row3373)
    exact FastRows.row3373_checked.symm
  rw [row]
  decide
theorem root375Step7 : scan edge (search edge 3) 4 [2593] 2593 root375Tail7 winner =
    scan edge (search edge 3) 4 [2593] 2593 root375Tail8 winner := by
  rw [root375Tail7, scan, if_neg (by decide), root375Candidates7_checked]
  change scan edge (search edge 3) 4 [2593] 2593 root375Tail8
    (if 3 ≤ root375Candidates7.length then search edge 3 [2593, 3373] root375Candidates7 5966 winner else winner) = _
  rw [if_neg (by decide)]
def root375Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root375Candidates8_checked : candidates edge 4 2593 3709 root375Tail9 winner.bound = root375Candidates8 := by
  unfold candidates
  have row : edge 3709 = (SortedNatTree.contains FastRows.row3709) := by
    change Data.row3709.contains = (SortedNatTree.contains FastRows.row3709)
    exact FastRows.row3709_checked.symm
  rw [row]
  decide
theorem root375Step8 : scan edge (search edge 3) 4 [2593] 2593 root375Tail8 winner =
    scan edge (search edge 3) 4 [2593] 2593 root375Tail9 winner := by
  rw [root375Tail8, scan, if_neg (by decide), root375Candidates8_checked]
  change scan edge (search edge 3) 4 [2593] 2593 root375Tail9
    (if 3 ≤ root375Candidates8.length then search edge 3 [2593, 3709] root375Candidates8 6302 winner else winner) = _
  rw [if_neg (by decide)]
def root375Candidates9 : List Nat := [10789, 12967, 17239, 19507]
set_option maxRecDepth 8192 in
theorem root375Candidates9_checked : candidates edge 4 2593 3847 root375Tail10 winner.bound = root375Candidates9 := by
  unfold candidates
  have row : edge 3847 = (SortedNatTree.contains FastRows.row3847) := by
    change Data.row3847.contains = (SortedNatTree.contains FastRows.row3847)
    exact FastRows.row3847_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root375Child9 : search edge 3 [2593, 3847] root375Candidates9 6440 winner = winner := by decide
theorem root375Step9 : scan edge (search edge 3) 4 [2593] 2593 root375Tail9 winner =
    scan edge (search edge 3) 4 [2593] 2593 root375Tail10 winner := by
  rw [root375Tail9, scan, if_neg (by decide), root375Candidates9_checked]
  change scan edge (search edge 3) 4 [2593] 2593 root375Tail10
    (if 3 ≤ root375Candidates9.length then search edge 3 [2593, 3847] root375Candidates9 6440 winner else winner) = _
  rw [if_pos (by decide), root375Child9]
def root375Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root375Candidates10_checked : candidates edge 4 2593 5011 root375Tail11 winner.bound = root375Candidates10 := by
  unfold candidates
  have row : edge 5011 = (SortedNatTree.contains FastRows.row5011) := by
    change Data.row5011.contains = (SortedNatTree.contains FastRows.row5011)
    exact FastRows.row5011_checked.symm
  rw [row]
  decide
theorem root375Step10 : scan edge (search edge 3) 4 [2593] 2593 root375Tail10 winner =
    scan edge (search edge 3) 4 [2593] 2593 root375Tail11 winner := by
  rw [root375Tail10, scan, if_neg (by decide), root375Candidates10_checked]
  change scan edge (search edge 3) 4 [2593] 2593 root375Tail11
    (if 3 ≤ root375Candidates10.length then search edge 3 [2593, 5011] root375Candidates10 7604 winner else winner) = _
  rw [if_neg (by decide)]
def root375Candidates11 : List Nat := [8293, 16903, 18013]
set_option maxRecDepth 8192 in
theorem root375Candidates11_checked : candidates edge 4 2593 5101 root375Tail12 winner.bound = root375Candidates11 := by
  unfold candidates
  have row : edge 5101 = (SortedNatTree.contains FastRows.row5101) := by
    change Data.row5101.contains = (SortedNatTree.contains FastRows.row5101)
    exact FastRows.row5101_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root375Child11 : search edge 3 [2593, 5101] root375Candidates11 7694 winner = winner := by decide
theorem root375Step11 : scan edge (search edge 3) 4 [2593] 2593 root375Tail11 winner =
    scan edge (search edge 3) 4 [2593] 2593 root375Tail12 winner := by
  rw [root375Tail11, scan, if_neg (by decide), root375Candidates11_checked]
  change scan edge (search edge 3) 4 [2593] 2593 root375Tail12
    (if 3 ≤ root375Candidates11.length then search edge 3 [2593, 5101] root375Candidates11 7694 winner else winner) = _
  rw [if_pos (by decide), root375Child11]
def root375Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root375Candidates12_checked : candidates edge 4 2593 5281 root375Tail13 winner.bound = root375Candidates12 := by
  unfold candidates
  have row : edge 5281 = (SortedNatTree.contains FastRows.row5281) := by
    change Data.row5281.contains = (SortedNatTree.contains FastRows.row5281)
    exact FastRows.row5281_checked.symm
  rw [row]
  decide
theorem root375Step12 : scan edge (search edge 3) 4 [2593] 2593 root375Tail12 winner =
    scan edge (search edge 3) 4 [2593] 2593 root375Tail13 winner := by
  rw [root375Tail12, scan, if_neg (by decide), root375Candidates12_checked]
  change scan edge (search edge 3) 4 [2593] 2593 root375Tail13
    (if 3 ≤ root375Candidates12.length then search edge 3 [2593, 5281] root375Candidates12 7874 winner else winner) = _
  rw [if_neg (by decide)]
def root375Candidates13 : List Nat := [7039]
set_option maxRecDepth 8192 in
theorem root375Candidates13_checked : candidates edge 4 2593 5443 root375Tail14 winner.bound = root375Candidates13 := by
  unfold candidates
  have row : edge 5443 = (SortedNatTree.contains FastRows.row5443) := by
    change Data.row5443.contains = (SortedNatTree.contains FastRows.row5443)
    exact FastRows.row5443_checked.symm
  rw [row]
  decide
theorem root375Step13 : scan edge (search edge 3) 4 [2593] 2593 root375Tail13 winner =
    scan edge (search edge 3) 4 [2593] 2593 root375Tail14 winner := by
  rw [root375Tail13, scan, if_neg (by decide), root375Candidates13_checked]
  change scan edge (search edge 3) 4 [2593] 2593 root375Tail14
    (if 3 ≤ root375Candidates13.length then search edge 3 [2593, 5443] root375Candidates13 8036 winner else winner) = _
  rw [if_neg (by decide)]
def root375Candidates14 : List Nat := [12541]
set_option maxRecDepth 8192 in
theorem root375Candidates14_checked : candidates edge 4 2593 5683 root375Tail15 winner.bound = root375Candidates14 := by
  unfold candidates
  have row : edge 5683 = (SortedNatTree.contains FastRows.row5683) := by
    change Data.row5683.contains = (SortedNatTree.contains FastRows.row5683)
    exact FastRows.row5683_checked.symm
  rw [row]
  decide
theorem root375Step14 : scan edge (search edge 3) 4 [2593] 2593 root375Tail14 winner =
    scan edge (search edge 3) 4 [2593] 2593 root375Tail15 winner := by
  rw [root375Tail14, scan, if_neg (by decide), root375Candidates14_checked]
  change scan edge (search edge 3) 4 [2593] 2593 root375Tail15
    (if 3 ≤ root375Candidates14.length then search edge 3 [2593, 5683] root375Candidates14 8276 winner else winner) = _
  rw [if_neg (by decide)]
theorem root375Scan15 : scan edge (search edge 3) 4 [2593] 2593 root375Tail15 winner = winner := by rw [root375Tail15, scan, if_pos (by decide)]
theorem root375Scan14 : scan edge (search edge 3) 4 [2593] 2593 root375Tail14 winner = winner :=
  (root375Step14).trans root375Scan15
theorem root375Scan13 : scan edge (search edge 3) 4 [2593] 2593 root375Tail13 winner = winner :=
  (root375Step13).trans root375Scan14
theorem root375Scan12 : scan edge (search edge 3) 4 [2593] 2593 root375Tail12 winner = winner :=
  (root375Step12).trans root375Scan13
theorem root375Scan11 : scan edge (search edge 3) 4 [2593] 2593 root375Tail11 winner = winner :=
  (root375Step11).trans root375Scan12
theorem root375Scan10 : scan edge (search edge 3) 4 [2593] 2593 root375Tail10 winner = winner :=
  (root375Step10).trans root375Scan11
theorem root375Scan9 : scan edge (search edge 3) 4 [2593] 2593 root375Tail9 winner = winner :=
  (root375Step9).trans root375Scan10
theorem root375Scan8 : scan edge (search edge 3) 4 [2593] 2593 root375Tail8 winner = winner :=
  (root375Step8).trans root375Scan9
theorem root375Scan7 : scan edge (search edge 3) 4 [2593] 2593 root375Tail7 winner = winner :=
  (root375Step7).trans root375Scan8
theorem root375Scan6 : scan edge (search edge 3) 4 [2593] 2593 root375Tail6 winner = winner :=
  (root375Step6).trans root375Scan7
theorem root375Scan5 : scan edge (search edge 3) 4 [2593] 2593 root375Tail5 winner = winner :=
  (root375Step5).trans root375Scan6
theorem root375Scan4 : scan edge (search edge 3) 4 [2593] 2593 root375Tail4 winner = winner :=
  (root375Step4).trans root375Scan5
theorem root375Scan3 : scan edge (search edge 3) 4 [2593] 2593 root375Tail3 winner = winner :=
  (root375Step3).trans root375Scan4
theorem root375Scan2 : scan edge (search edge 3) 4 [2593] 2593 root375Tail2 winner = winner :=
  (root375Step2).trans root375Scan3
theorem root375Scan1 : scan edge (search edge 3) 4 [2593] 2593 root375Tail1 winner = winner :=
  (root375Step1).trans root375Scan2
theorem root375Scan0 : scan edge (search edge 3) 4 [2593] 2593 root375Tail0 winner = winner :=
  (root375Step0).trans root375Scan1
set_option maxRecDepth 4096 in
theorem root375Branch : search edge 4 [2593] root375Candidates 2593 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2593] 2593 root375Tail0 winner = winner
  exact root375Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
