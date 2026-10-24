import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root410Candidates : List Nat := Data.row2843
set_option maxRecDepth 8192 in
theorem root410Candidates_checked : candidates edge 5 0 2843 tail411 winner.bound = root410Candidates := by
  apply candidates_eq_row edge 5 0 2843 tail411 root410Candidates winner.bound tail411_ordered
  · decide
  · decide
  · funext q
    rfl
def root410Tail43 : List Nat := []
def root410Tail42 : List Nat := 23099 :: root410Tail43
def root410Tail41 : List Nat := 22691 :: root410Tail42
def root410Tail40 : List Nat := 22397 :: root410Tail41
def root410Tail39 : List Nat := 22277 :: root410Tail40
def root410Tail38 : List Nat := 21407 :: root410Tail39
def root410Tail37 : List Nat := 21227 :: root410Tail38
def root410Tail36 : List Nat := 20717 :: root410Tail37
def root410Tail35 : List Nat := 19463 :: root410Tail36
def root410Tail34 : List Nat := 19163 :: root410Tail35
def root410Tail33 : List Nat := 17909 :: root410Tail34
def root410Tail32 : List Nat := 17669 :: root410Tail33
def root410Tail31 : List Nat := 17471 :: root410Tail32
def root410Tail30 : List Nat := 16943 :: root410Tail31
def root410Tail29 : List Nat := 15149 :: root410Tail30
def root410Tail28 : List Nat := 15137 :: root410Tail29
def root410Tail27 : List Nat := 14321 :: root410Tail28
def root410Tail26 : List Nat := 13313 :: root410Tail27
def root410Tail25 : List Nat := 13163 :: root410Tail26
def root410Tail24 : List Nat := 12653 :: root410Tail25
def root410Tail23 : List Nat := 12569 :: root410Tail24
def root410Tail22 : List Nat := 11939 :: root410Tail23
def root410Tail21 : List Nat := 10667 :: root410Tail22
def root410Tail20 : List Nat := 10457 :: root410Tail21
def root410Tail19 : List Nat := 10427 :: root410Tail20
def root410Tail18 : List Nat := 10139 :: root410Tail19
def root410Tail17 : List Nat := 9857 :: root410Tail18
def root410Tail16 : List Nat := 9473 :: root410Tail17
def root410Tail15 : List Nat := 8831 :: root410Tail16
def root410Tail14 : List Nat := 8273 :: root410Tail15
def root410Tail13 : List Nat := 8171 :: root410Tail14
def root410Tail12 : List Nat := 7841 :: root410Tail13
def root410Tail11 : List Nat := 6899 :: root410Tail12
def root410Tail10 : List Nat := 6047 :: root410Tail11
def root410Tail9 : List Nat := 5807 :: root410Tail10
def root410Tail8 : List Nat := 5783 :: root410Tail9
def root410Tail7 : List Nat := 5657 :: root410Tail8
def root410Tail6 : List Nat := 4817 :: root410Tail7
def root410Tail5 : List Nat := 4253 :: root410Tail6
def root410Tail4 : List Nat := 3989 :: root410Tail5
def root410Tail3 : List Nat := 3947 :: root410Tail4
def root410Tail2 : List Nat := 3389 :: root410Tail3
def root410Tail1 : List Nat := 3203 :: root410Tail2
def root410Tail0 : List Nat := 3089 :: root410Tail1
def root410Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root410Candidates0_checked : candidates edge 4 2843 3089 root410Tail1 winner.bound = root410Candidates0 := by
  unfold candidates
  have row : edge 3089 = (SortedNatTree.contains FastRows.row3089) := by
    change Data.row3089.contains = (SortedNatTree.contains FastRows.row3089)
    exact FastRows.row3089_checked.symm
  rw [row]
  decide
theorem root410Step0 : scan edge (search edge 3) 4 [2843] 2843 root410Tail0 winner =
    scan edge (search edge 3) 4 [2843] 2843 root410Tail1 winner := by
  rw [root410Tail0, scan, if_neg (by decide), root410Candidates0_checked]
  change scan edge (search edge 3) 4 [2843] 2843 root410Tail1
    (if 3 ≤ root410Candidates0.length then search edge 3 [2843, 3089] root410Candidates0 5932 winner else winner) = _
  rw [if_neg (by decide)]
def root410Candidates1 : List Nat := [19163, 19463]
set_option maxRecDepth 8192 in
theorem root410Candidates1_checked : candidates edge 4 2843 3203 root410Tail2 winner.bound = root410Candidates1 := by
  unfold candidates
  have row : edge 3203 = (SortedNatTree.contains FastRows.row3203) := by
    change Data.row3203.contains = (SortedNatTree.contains FastRows.row3203)
    exact FastRows.row3203_checked.symm
  rw [row]
  decide
theorem root410Step1 : scan edge (search edge 3) 4 [2843] 2843 root410Tail1 winner =
    scan edge (search edge 3) 4 [2843] 2843 root410Tail2 winner := by
  rw [root410Tail1, scan, if_neg (by decide), root410Candidates1_checked]
  change scan edge (search edge 3) 4 [2843] 2843 root410Tail2
    (if 3 ≤ root410Candidates1.length then search edge 3 [2843, 3203] root410Candidates1 6046 winner else winner) = _
  rw [if_neg (by decide)]
def root410Candidates2 : List Nat := [7841, 12653, 13313, 16943]
set_option maxRecDepth 8192 in
theorem root410Candidates2_checked : candidates edge 4 2843 3389 root410Tail3 winner.bound = root410Candidates2 := by
  unfold candidates
  have row : edge 3389 = (SortedNatTree.contains FastRows.row3389) := by
    change Data.row3389.contains = (SortedNatTree.contains FastRows.row3389)
    exact FastRows.row3389_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root410Child2 : search edge 3 [2843, 3389] root410Candidates2 6232 winner = winner := by decide
theorem root410Step2 : scan edge (search edge 3) 4 [2843] 2843 root410Tail2 winner =
    scan edge (search edge 3) 4 [2843] 2843 root410Tail3 winner := by
  rw [root410Tail2, scan, if_neg (by decide), root410Candidates2_checked]
  change scan edge (search edge 3) 4 [2843] 2843 root410Tail3
    (if 3 ≤ root410Candidates2.length then search edge 3 [2843, 3389] root410Candidates2 6232 winner else winner) = _
  rw [if_pos (by decide), root410Child2]
def root410Candidates3 : List Nat := [9857, 15137, 17909]
set_option maxRecDepth 8192 in
theorem root410Candidates3_checked : candidates edge 4 2843 3947 root410Tail4 winner.bound = root410Candidates3 := by
  unfold candidates
  have row : edge 3947 = (SortedNatTree.contains FastRows.row3947) := by
    change Data.row3947.contains = (SortedNatTree.contains FastRows.row3947)
    exact FastRows.row3947_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root410Child3 : search edge 3 [2843, 3947] root410Candidates3 6790 winner = winner := by decide
theorem root410Step3 : scan edge (search edge 3) 4 [2843] 2843 root410Tail3 winner =
    scan edge (search edge 3) 4 [2843] 2843 root410Tail4 winner := by
  rw [root410Tail3, scan, if_neg (by decide), root410Candidates3_checked]
  change scan edge (search edge 3) 4 [2843] 2843 root410Tail4
    (if 3 ≤ root410Candidates3.length then search edge 3 [2843, 3947] root410Candidates3 6790 winner else winner) = _
  rw [if_pos (by decide), root410Child3]
def root410Candidates4 : List Nat := [5783]
set_option maxRecDepth 8192 in
theorem root410Candidates4_checked : candidates edge 4 2843 3989 root410Tail5 winner.bound = root410Candidates4 := by
  unfold candidates
  have row : edge 3989 = (SortedNatTree.contains FastRows.row3989) := by
    change Data.row3989.contains = (SortedNatTree.contains FastRows.row3989)
    exact FastRows.row3989_checked.symm
  rw [row]
  decide
theorem root410Step4 : scan edge (search edge 3) 4 [2843] 2843 root410Tail4 winner =
    scan edge (search edge 3) 4 [2843] 2843 root410Tail5 winner := by
  rw [root410Tail4, scan, if_neg (by decide), root410Candidates4_checked]
  change scan edge (search edge 3) 4 [2843] 2843 root410Tail5
    (if 3 ≤ root410Candidates4.length then search edge 3 [2843, 3989] root410Candidates4 6832 winner else winner) = _
  rw [if_neg (by decide)]
def root410Candidates5 : List Nat := [5657, 8831]
set_option maxRecDepth 8192 in
theorem root410Candidates5_checked : candidates edge 4 2843 4253 root410Tail6 winner.bound = root410Candidates5 := by
  unfold candidates
  have row : edge 4253 = (SortedNatTree.contains FastRows.row4253) := by
    change Data.row4253.contains = (SortedNatTree.contains FastRows.row4253)
    exact FastRows.row4253_checked.symm
  rw [row]
  decide
theorem root410Step5 : scan edge (search edge 3) 4 [2843] 2843 root410Tail5 winner =
    scan edge (search edge 3) 4 [2843] 2843 root410Tail6 winner := by
  rw [root410Tail5, scan, if_neg (by decide), root410Candidates5_checked]
  change scan edge (search edge 3) 4 [2843] 2843 root410Tail6
    (if 3 ≤ root410Candidates5.length then search edge 3 [2843, 4253] root410Candidates5 7096 winner else winner) = _
  rw [if_neg (by decide)]
def root410Candidates6 : List Nat := [5783]
set_option maxRecDepth 8192 in
theorem root410Candidates6_checked : candidates edge 4 2843 4817 root410Tail7 winner.bound = root410Candidates6 := by
  unfold candidates
  have row : edge 4817 = (SortedNatTree.contains FastRows.row4817) := by
    change Data.row4817.contains = (SortedNatTree.contains FastRows.row4817)
    exact FastRows.row4817_checked.symm
  rw [row]
  decide
theorem root410Step6 : scan edge (search edge 3) 4 [2843] 2843 root410Tail6 winner =
    scan edge (search edge 3) 4 [2843] 2843 root410Tail7 winner := by
  rw [root410Tail6, scan, if_neg (by decide), root410Candidates6_checked]
  change scan edge (search edge 3) 4 [2843] 2843 root410Tail7
    (if 3 ≤ root410Candidates6.length then search edge 3 [2843, 4817] root410Candidates6 7660 winner else winner) = _
  rw [if_neg (by decide)]
def root410Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root410Candidates7_checked : candidates edge 4 2843 5657 root410Tail8 winner.bound = root410Candidates7 := by
  unfold candidates
  have row : edge 5657 = (SortedNatTree.contains FastRows.row5657) := by
    change Data.row5657.contains = (SortedNatTree.contains FastRows.row5657)
    exact FastRows.row5657_checked.symm
  rw [row]
  decide
theorem root410Step7 : scan edge (search edge 3) 4 [2843] 2843 root410Tail7 winner =
    scan edge (search edge 3) 4 [2843] 2843 root410Tail8 winner := by
  rw [root410Tail7, scan, if_neg (by decide), root410Candidates7_checked]
  change scan edge (search edge 3) 4 [2843] 2843 root410Tail8
    (if 3 ≤ root410Candidates7.length then search edge 3 [2843, 5657] root410Candidates7 8500 winner else winner) = _
  rw [if_neg (by decide)]
def root410Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root410Candidates8_checked : candidates edge 4 2843 5783 root410Tail9 winner.bound = root410Candidates8 := by
  unfold candidates
  have row : edge 5783 = (SortedNatTree.contains FastRows.row5783) := by
    change Data.row5783.contains = (SortedNatTree.contains FastRows.row5783)
    exact FastRows.row5783_checked.symm
  rw [row]
  decide
theorem root410Step8 : scan edge (search edge 3) 4 [2843] 2843 root410Tail8 winner =
    scan edge (search edge 3) 4 [2843] 2843 root410Tail9 winner := by
  rw [root410Tail8, scan, if_neg (by decide), root410Candidates8_checked]
  change scan edge (search edge 3) 4 [2843] 2843 root410Tail9
    (if 3 ≤ root410Candidates8.length then search edge 3 [2843, 5783] root410Candidates8 8626 winner else winner) = _
  rw [if_neg (by decide)]
theorem root410Scan9 : scan edge (search edge 3) 4 [2843] 2843 root410Tail9 winner = winner := by rw [root410Tail9, scan, if_pos (by decide)]
theorem root410Scan8 : scan edge (search edge 3) 4 [2843] 2843 root410Tail8 winner = winner :=
  (root410Step8).trans root410Scan9
theorem root410Scan7 : scan edge (search edge 3) 4 [2843] 2843 root410Tail7 winner = winner :=
  (root410Step7).trans root410Scan8
theorem root410Scan6 : scan edge (search edge 3) 4 [2843] 2843 root410Tail6 winner = winner :=
  (root410Step6).trans root410Scan7
theorem root410Scan5 : scan edge (search edge 3) 4 [2843] 2843 root410Tail5 winner = winner :=
  (root410Step5).trans root410Scan6
theorem root410Scan4 : scan edge (search edge 3) 4 [2843] 2843 root410Tail4 winner = winner :=
  (root410Step4).trans root410Scan5
theorem root410Scan3 : scan edge (search edge 3) 4 [2843] 2843 root410Tail3 winner = winner :=
  (root410Step3).trans root410Scan4
theorem root410Scan2 : scan edge (search edge 3) 4 [2843] 2843 root410Tail2 winner = winner :=
  (root410Step2).trans root410Scan3
theorem root410Scan1 : scan edge (search edge 3) 4 [2843] 2843 root410Tail1 winner = winner :=
  (root410Step1).trans root410Scan2
theorem root410Scan0 : scan edge (search edge 3) 4 [2843] 2843 root410Tail0 winner = winner :=
  (root410Step0).trans root410Scan1
set_option maxRecDepth 4096 in
theorem root410Branch : search edge 4 [2843] root410Candidates 2843 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2843] 2843 root410Tail0 winner = winner
  exact root410Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
