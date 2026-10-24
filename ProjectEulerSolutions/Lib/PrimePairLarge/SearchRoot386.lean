import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root386Candidates : List Nat := Data.row2683
set_option maxRecDepth 8192 in
theorem root386Candidates_checked : candidates edge 5 0 2683 tail387 winner.bound = root386Candidates := by
  apply candidates_eq_row edge 5 0 2683 tail387 root386Candidates winner.bound tail387_ordered
  · decide
  · decide
  · funext q
    rfl
def root386Tail38 : List Nat := []
def root386Tail37 : List Nat := 22777 :: root386Tail38
def root386Tail36 : List Nat := 22651 :: root386Tail37
def root386Tail35 : List Nat := 22447 :: root386Tail36
def root386Tail34 : List Nat := 22189 :: root386Tail35
def root386Tail33 : List Nat := 19489 :: root386Tail34
def root386Tail32 : List Nat := 17659 :: root386Tail33
def root386Tail31 : List Nat := 17551 :: root386Tail32
def root386Tail30 : List Nat := 17317 :: root386Tail31
def root386Tail29 : List Nat := 16141 :: root386Tail30
def root386Tail28 : List Nat := 15349 :: root386Tail29
def root386Tail27 : List Nat := 14887 :: root386Tail28
def root386Tail26 : List Nat := 14419 :: root386Tail27
def root386Tail25 : List Nat := 13411 :: root386Tail26
def root386Tail24 : List Nat := 12973 :: root386Tail25
def root386Tail23 : List Nat := 12601 :: root386Tail24
def root386Tail22 : List Nat := 11821 :: root386Tail23
def root386Tail21 : List Nat := 11689 :: root386Tail22
def root386Tail20 : List Nat := 11467 :: root386Tail21
def root386Tail19 : List Nat := 11149 :: root386Tail20
def root386Tail18 : List Nat := 11047 :: root386Tail19
def root386Tail17 : List Nat := 10831 :: root386Tail18
def root386Tail16 : List Nat := 10687 :: root386Tail17
def root386Tail15 : List Nat := 9067 :: root386Tail16
def root386Tail14 : List Nat := 8803 :: root386Tail15
def root386Tail13 : List Nat := 8293 :: root386Tail14
def root386Tail12 : List Nat := 8167 :: root386Tail13
def root386Tail11 : List Nat := 8059 :: root386Tail12
def root386Tail10 : List Nat := 6619 :: root386Tail11
def root386Tail9 : List Nat := 6451 :: root386Tail10
def root386Tail8 : List Nat := 5563 :: root386Tail9
def root386Tail7 : List Nat := 5209 :: root386Tail8
def root386Tail6 : List Nat := 5023 :: root386Tail7
def root386Tail5 : List Nat := 4447 :: root386Tail6
def root386Tail4 : List Nat := 3931 :: root386Tail5
def root386Tail3 : List Nat := 3823 :: root386Tail4
def root386Tail2 : List Nat := 3253 :: root386Tail3
def root386Tail1 : List Nat := 2797 :: root386Tail2
def root386Tail0 : List Nat := 2791 :: root386Tail1
def root386Candidates0 : List Nat := [11467]
set_option maxRecDepth 8192 in
theorem root386Candidates0_checked : candidates edge 4 2683 2791 root386Tail1 winner.bound = root386Candidates0 := by
  unfold candidates
  have row : edge 2791 = (SortedNatTree.contains FastRows.row2791) := by
    change Data.row2791.contains = (SortedNatTree.contains FastRows.row2791)
    exact FastRows.row2791_checked.symm
  rw [row]
  decide
theorem root386Step0 : scan edge (search edge 3) 4 [2683] 2683 root386Tail0 winner =
    scan edge (search edge 3) 4 [2683] 2683 root386Tail1 winner := by
  rw [root386Tail0, scan, if_neg (by decide), root386Candidates0_checked]
  change scan edge (search edge 3) 4 [2683] 2683 root386Tail1
    (if 3 ≤ root386Candidates0.length then search edge 3 [2683, 2791] root386Candidates0 5474 winner else winner) = _
  rw [if_neg (by decide)]
def root386Candidates1 : List Nat := [5023]
set_option maxRecDepth 8192 in
theorem root386Candidates1_checked : candidates edge 4 2683 2797 root386Tail2 winner.bound = root386Candidates1 := by
  unfold candidates
  have row : edge 2797 = (SortedNatTree.contains FastRows.row2797) := by
    change Data.row2797.contains = (SortedNatTree.contains FastRows.row2797)
    exact FastRows.row2797_checked.symm
  rw [row]
  decide
theorem root386Step1 : scan edge (search edge 3) 4 [2683] 2683 root386Tail1 winner =
    scan edge (search edge 3) 4 [2683] 2683 root386Tail2 winner := by
  rw [root386Tail1, scan, if_neg (by decide), root386Candidates1_checked]
  change scan edge (search edge 3) 4 [2683] 2683 root386Tail2
    (if 3 ≤ root386Candidates1.length then search edge 3 [2683, 2797] root386Candidates1 5480 winner else winner) = _
  rw [if_neg (by decide)]
def root386Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root386Candidates2_checked : candidates edge 4 2683 3253 root386Tail3 winner.bound = root386Candidates2 := by
  unfold candidates
  have row : edge 3253 = (SortedNatTree.contains FastRows.row3253) := by
    change Data.row3253.contains = (SortedNatTree.contains FastRows.row3253)
    exact FastRows.row3253_checked.symm
  rw [row]
  decide
theorem root386Step2 : scan edge (search edge 3) 4 [2683] 2683 root386Tail2 winner =
    scan edge (search edge 3) 4 [2683] 2683 root386Tail3 winner := by
  rw [root386Tail2, scan, if_neg (by decide), root386Candidates2_checked]
  change scan edge (search edge 3) 4 [2683] 2683 root386Tail3
    (if 3 ≤ root386Candidates2.length then search edge 3 [2683, 3253] root386Candidates2 5936 winner else winner) = _
  rw [if_neg (by decide)]
def root386Candidates3 : List Nat := [5209, 19489]
set_option maxRecDepth 8192 in
theorem root386Candidates3_checked : candidates edge 4 2683 3823 root386Tail4 winner.bound = root386Candidates3 := by
  unfold candidates
  have row : edge 3823 = (SortedNatTree.contains FastRows.row3823) := by
    change Data.row3823.contains = (SortedNatTree.contains FastRows.row3823)
    exact FastRows.row3823_checked.symm
  rw [row]
  decide
theorem root386Step3 : scan edge (search edge 3) 4 [2683] 2683 root386Tail3 winner =
    scan edge (search edge 3) 4 [2683] 2683 root386Tail4 winner := by
  rw [root386Tail3, scan, if_neg (by decide), root386Candidates3_checked]
  change scan edge (search edge 3) 4 [2683] 2683 root386Tail4
    (if 3 ≤ root386Candidates3.length then search edge 3 [2683, 3823] root386Candidates3 6506 winner else winner) = _
  rw [if_neg (by decide)]
def root386Candidates4 : List Nat := [8803]
set_option maxRecDepth 8192 in
theorem root386Candidates4_checked : candidates edge 4 2683 3931 root386Tail5 winner.bound = root386Candidates4 := by
  unfold candidates
  have row : edge 3931 = (SortedNatTree.contains FastRows.row3931) := by
    change Data.row3931.contains = (SortedNatTree.contains FastRows.row3931)
    exact FastRows.row3931_checked.symm
  rw [row]
  decide
theorem root386Step4 : scan edge (search edge 3) 4 [2683] 2683 root386Tail4 winner =
    scan edge (search edge 3) 4 [2683] 2683 root386Tail5 winner := by
  rw [root386Tail4, scan, if_neg (by decide), root386Candidates4_checked]
  change scan edge (search edge 3) 4 [2683] 2683 root386Tail5
    (if 3 ≤ root386Candidates4.length then search edge 3 [2683, 3931] root386Candidates4 6614 winner else winner) = _
  rw [if_neg (by decide)]
def root386Candidates5 : List Nat := [5023, 8293, 10831, 14419]
set_option maxRecDepth 8192 in
theorem root386Candidates5_checked : candidates edge 4 2683 4447 root386Tail6 winner.bound = root386Candidates5 := by
  unfold candidates
  have row : edge 4447 = (SortedNatTree.contains FastRows.row4447) := by
    change Data.row4447.contains = (SortedNatTree.contains FastRows.row4447)
    exact FastRows.row4447_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root386Child5 : search edge 3 [2683, 4447] root386Candidates5 7130 winner = winner := by decide
theorem root386Step5 : scan edge (search edge 3) 4 [2683] 2683 root386Tail5 winner =
    scan edge (search edge 3) 4 [2683] 2683 root386Tail6 winner := by
  rw [root386Tail5, scan, if_neg (by decide), root386Candidates5_checked]
  change scan edge (search edge 3) 4 [2683] 2683 root386Tail6
    (if 3 ≤ root386Candidates5.length then search edge 3 [2683, 4447] root386Candidates5 7130 winner else winner) = _
  rw [if_pos (by decide), root386Child5]
def root386Candidates6 : List Nat := [8059, 10687]
set_option maxRecDepth 8192 in
theorem root386Candidates6_checked : candidates edge 4 2683 5023 root386Tail7 winner.bound = root386Candidates6 := by
  unfold candidates
  have row : edge 5023 = (SortedNatTree.contains FastRows.row5023) := by
    change Data.row5023.contains = (SortedNatTree.contains FastRows.row5023)
    exact FastRows.row5023_checked.symm
  rw [row]
  decide
theorem root386Step6 : scan edge (search edge 3) 4 [2683] 2683 root386Tail6 winner =
    scan edge (search edge 3) 4 [2683] 2683 root386Tail7 winner := by
  rw [root386Tail6, scan, if_neg (by decide), root386Candidates6_checked]
  change scan edge (search edge 3) 4 [2683] 2683 root386Tail7
    (if 3 ≤ root386Candidates6.length then search edge 3 [2683, 5023] root386Candidates6 7706 winner else winner) = _
  rw [if_neg (by decide)]
def root386Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root386Candidates7_checked : candidates edge 4 2683 5209 root386Tail8 winner.bound = root386Candidates7 := by
  unfold candidates
  have row : edge 5209 = (SortedNatTree.contains FastRows.row5209) := by
    change Data.row5209.contains = (SortedNatTree.contains FastRows.row5209)
    exact FastRows.row5209_checked.symm
  rw [row]
  decide
theorem root386Step7 : scan edge (search edge 3) 4 [2683] 2683 root386Tail7 winner =
    scan edge (search edge 3) 4 [2683] 2683 root386Tail8 winner := by
  rw [root386Tail7, scan, if_neg (by decide), root386Candidates7_checked]
  change scan edge (search edge 3) 4 [2683] 2683 root386Tail8
    (if 3 ≤ root386Candidates7.length then search edge 3 [2683, 5209] root386Candidates7 7892 winner else winner) = _
  rw [if_neg (by decide)]
def root386Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root386Candidates8_checked : candidates edge 4 2683 5563 root386Tail9 winner.bound = root386Candidates8 := by
  unfold candidates
  have row : edge 5563 = (SortedNatTree.contains FastRows.row5563) := by
    change Data.row5563.contains = (SortedNatTree.contains FastRows.row5563)
    exact FastRows.row5563_checked.symm
  rw [row]
  decide
theorem root386Step8 : scan edge (search edge 3) 4 [2683] 2683 root386Tail8 winner =
    scan edge (search edge 3) 4 [2683] 2683 root386Tail9 winner := by
  rw [root386Tail8, scan, if_neg (by decide), root386Candidates8_checked]
  change scan edge (search edge 3) 4 [2683] 2683 root386Tail9
    (if 3 ≤ root386Candidates8.length then search edge 3 [2683, 5563] root386Candidates8 8246 winner else winner) = _
  rw [if_neg (by decide)]
theorem root386Scan9 : scan edge (search edge 3) 4 [2683] 2683 root386Tail9 winner = winner := by rw [root386Tail9, scan, if_pos (by decide)]
theorem root386Scan8 : scan edge (search edge 3) 4 [2683] 2683 root386Tail8 winner = winner :=
  (root386Step8).trans root386Scan9
theorem root386Scan7 : scan edge (search edge 3) 4 [2683] 2683 root386Tail7 winner = winner :=
  (root386Step7).trans root386Scan8
theorem root386Scan6 : scan edge (search edge 3) 4 [2683] 2683 root386Tail6 winner = winner :=
  (root386Step6).trans root386Scan7
theorem root386Scan5 : scan edge (search edge 3) 4 [2683] 2683 root386Tail5 winner = winner :=
  (root386Step5).trans root386Scan6
theorem root386Scan4 : scan edge (search edge 3) 4 [2683] 2683 root386Tail4 winner = winner :=
  (root386Step4).trans root386Scan5
theorem root386Scan3 : scan edge (search edge 3) 4 [2683] 2683 root386Tail3 winner = winner :=
  (root386Step3).trans root386Scan4
theorem root386Scan2 : scan edge (search edge 3) 4 [2683] 2683 root386Tail2 winner = winner :=
  (root386Step2).trans root386Scan3
theorem root386Scan1 : scan edge (search edge 3) 4 [2683] 2683 root386Tail1 winner = winner :=
  (root386Step1).trans root386Scan2
theorem root386Scan0 : scan edge (search edge 3) 4 [2683] 2683 root386Tail0 winner = winner :=
  (root386Step0).trans root386Scan1
set_option maxRecDepth 4096 in
theorem root386Branch : search edge 4 [2683] root386Candidates 2683 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2683] 2683 root386Tail0 winner = winner
  exact root386Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
