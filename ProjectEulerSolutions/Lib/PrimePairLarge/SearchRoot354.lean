import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root354Candidates : List Nat := Data.row2399
set_option maxRecDepth 8192 in
theorem root354Candidates_checked : candidates edge 5 0 2399 tail355 winner.bound = root354Candidates := by
  apply candidates_eq_row edge 5 0 2399 tail355 root354Candidates winner.bound tail355_ordered
  · decide
  · decide
  · funext q
    rfl
def root354Tail46 : List Nat := []
def root354Tail45 : List Nat := 23609 :: root354Tail46
def root354Tail44 : List Nat := 23339 :: root354Tail45
def root354Tail43 : List Nat := 22643 :: root354Tail44
def root354Tail42 : List Nat := 21863 :: root354Tail43
def root354Tail41 : List Nat := 21701 :: root354Tail42
def root354Tail40 : List Nat := 21587 :: root354Tail41
def root354Tail39 : List Nat := 20939 :: root354Tail40
def root354Tail38 : List Nat := 20123 :: root354Tail39
def root354Tail37 : List Nat := 19727 :: root354Tail38
def root354Tail36 : List Nat := 19553 :: root354Tail37
def root354Tail35 : List Nat := 15959 :: root354Tail36
def root354Tail34 : List Nat := 15581 :: root354Tail35
def root354Tail33 : List Nat := 15233 :: root354Tail34
def root354Tail32 : List Nat := 15227 :: root354Tail33
def root354Tail31 : List Nat := 15077 :: root354Tail32
def root354Tail30 : List Nat := 14699 :: root354Tail31
def root354Tail29 : List Nat := 14639 :: root354Tail30
def root354Tail28 : List Nat := 14561 :: root354Tail29
def root354Tail27 : List Nat := 14447 :: root354Tail28
def root354Tail26 : List Nat := 14153 :: root354Tail27
def root354Tail25 : List Nat := 13799 :: root354Tail26
def root354Tail24 : List Nat := 13151 :: root354Tail25
def root354Tail23 : List Nat := 11807 :: root354Tail24
def root354Tail22 : List Nat := 11801 :: root354Tail23
def root354Tail21 : List Nat := 11447 :: root354Tail22
def root354Tail20 : List Nat := 11369 :: root354Tail21
def root354Tail19 : List Nat := 10709 :: root354Tail20
def root354Tail18 : List Nat := 9623 :: root354Tail19
def root354Tail17 : List Nat := 8783 :: root354Tail18
def root354Tail16 : List Nat := 8363 :: root354Tail17
def root354Tail15 : List Nat := 8087 :: root354Tail16
def root354Tail14 : List Nat := 8081 :: root354Tail15
def root354Tail13 : List Nat := 7283 :: root354Tail14
def root354Tail12 : List Nat := 6977 :: root354Tail13
def root354Tail11 : List Nat := 6761 :: root354Tail12
def root354Tail10 : List Nat := 6737 :: root354Tail11
def root354Tail9 : List Nat := 6701 :: root354Tail10
def root354Tail8 : List Nat := 6131 :: root354Tail9
def root354Tail7 : List Nat := 5717 :: root354Tail8
def root354Tail6 : List Nat := 4517 :: root354Tail7
def root354Tail5 : List Nat := 4457 :: root354Tail6
def root354Tail4 : List Nat := 3911 :: root354Tail5
def root354Tail3 : List Nat := 3821 :: root354Tail4
def root354Tail2 : List Nat := 3467 :: root354Tail3
def root354Tail1 : List Nat := 3251 :: root354Tail2
def root354Tail0 : List Nat := 2663 :: root354Tail1
def root354Candidates0 : List Nat := [3251, 4457, 8783]
set_option maxRecDepth 8192 in
theorem root354Candidates0_checked : candidates edge 4 2399 2663 root354Tail1 winner.bound = root354Candidates0 := by
  unfold candidates
  have row : edge 2663 = (SortedNatTree.contains FastRows.row2663) := by
    change Data.row2663.contains = (SortedNatTree.contains FastRows.row2663)
    exact FastRows.row2663_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root354Child0 : search edge 3 [2399, 2663] root354Candidates0 5062 winner = winner := by decide
theorem root354Step0 : scan edge (search edge 3) 4 [2399] 2399 root354Tail0 winner =
    scan edge (search edge 3) 4 [2399] 2399 root354Tail1 winner := by
  rw [root354Tail0, scan, if_neg (by decide), root354Candidates0_checked]
  change scan edge (search edge 3) 4 [2399] 2399 root354Tail1
    (if 3 ≤ root354Candidates0.length then search edge 3 [2399, 2663] root354Candidates0 5062 winner else winner) = _
  rw [if_pos (by decide), root354Child0]
def root354Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root354Candidates1_checked : candidates edge 4 2399 3251 root354Tail2 winner.bound = root354Candidates1 := by
  unfold candidates
  have row : edge 3251 = (SortedNatTree.contains FastRows.row3251) := by
    change Data.row3251.contains = (SortedNatTree.contains FastRows.row3251)
    exact FastRows.row3251_checked.symm
  rw [row]
  decide
theorem root354Step1 : scan edge (search edge 3) 4 [2399] 2399 root354Tail1 winner =
    scan edge (search edge 3) 4 [2399] 2399 root354Tail2 winner := by
  rw [root354Tail1, scan, if_neg (by decide), root354Candidates1_checked]
  change scan edge (search edge 3) 4 [2399] 2399 root354Tail2
    (if 3 ≤ root354Candidates1.length then search edge 3 [2399, 3251] root354Candidates1 5650 winner else winner) = _
  rw [if_neg (by decide)]
def root354Candidates2 : List Nat := [4457, 20123]
set_option maxRecDepth 8192 in
theorem root354Candidates2_checked : candidates edge 4 2399 3467 root354Tail3 winner.bound = root354Candidates2 := by
  unfold candidates
  have row : edge 3467 = (SortedNatTree.contains FastRows.row3467) := by
    change Data.row3467.contains = (SortedNatTree.contains FastRows.row3467)
    exact FastRows.row3467_checked.symm
  rw [row]
  decide
theorem root354Step2 : scan edge (search edge 3) 4 [2399] 2399 root354Tail2 winner =
    scan edge (search edge 3) 4 [2399] 2399 root354Tail3 winner := by
  rw [root354Tail2, scan, if_neg (by decide), root354Candidates2_checked]
  change scan edge (search edge 3) 4 [2399] 2399 root354Tail3
    (if 3 ≤ root354Candidates2.length then search edge 3 [2399, 3467] root354Candidates2 5866 winner else winner) = _
  rw [if_neg (by decide)]
def root354Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root354Candidates3_checked : candidates edge 4 2399 3821 root354Tail4 winner.bound = root354Candidates3 := by
  unfold candidates
  have row : edge 3821 = (SortedNatTree.contains FastRows.row3821) := by
    change Data.row3821.contains = (SortedNatTree.contains FastRows.row3821)
    exact FastRows.row3821_checked.symm
  rw [row]
  decide
theorem root354Step3 : scan edge (search edge 3) 4 [2399] 2399 root354Tail3 winner =
    scan edge (search edge 3) 4 [2399] 2399 root354Tail4 winner := by
  rw [root354Tail3, scan, if_neg (by decide), root354Candidates3_checked]
  change scan edge (search edge 3) 4 [2399] 2399 root354Tail4
    (if 3 ≤ root354Candidates3.length then search edge 3 [2399, 3821] root354Candidates3 6220 winner else winner) = _
  rw [if_neg (by decide)]
def root354Candidates4 : List Nat := [4457, 5717, 15233]
set_option maxRecDepth 8192 in
theorem root354Candidates4_checked : candidates edge 4 2399 3911 root354Tail5 winner.bound = root354Candidates4 := by
  unfold candidates
  have row : edge 3911 = (SortedNatTree.contains FastRows.row3911) := by
    change Data.row3911.contains = (SortedNatTree.contains FastRows.row3911)
    exact FastRows.row3911_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root354Child4 : search edge 3 [2399, 3911] root354Candidates4 6310 winner = winner := by decide
theorem root354Step4 : scan edge (search edge 3) 4 [2399] 2399 root354Tail4 winner =
    scan edge (search edge 3) 4 [2399] 2399 root354Tail5 winner := by
  rw [root354Tail4, scan, if_neg (by decide), root354Candidates4_checked]
  change scan edge (search edge 3) 4 [2399] 2399 root354Tail5
    (if 3 ≤ root354Candidates4.length then search edge 3 [2399, 3911] root354Candidates4 6310 winner else winner) = _
  rw [if_pos (by decide), root354Child4]
def root354Candidates5 : List Nat := [14561]
set_option maxRecDepth 8192 in
theorem root354Candidates5_checked : candidates edge 4 2399 4457 root354Tail6 winner.bound = root354Candidates5 := by
  unfold candidates
  have row : edge 4457 = (SortedNatTree.contains FastRows.row4457) := by
    change Data.row4457.contains = (SortedNatTree.contains FastRows.row4457)
    exact FastRows.row4457_checked.symm
  rw [row]
  decide
theorem root354Step5 : scan edge (search edge 3) 4 [2399] 2399 root354Tail5 winner =
    scan edge (search edge 3) 4 [2399] 2399 root354Tail6 winner := by
  rw [root354Tail5, scan, if_neg (by decide), root354Candidates5_checked]
  change scan edge (search edge 3) 4 [2399] 2399 root354Tail6
    (if 3 ≤ root354Candidates5.length then search edge 3 [2399, 4457] root354Candidates5 6856 winner else winner) = _
  rw [if_neg (by decide)]
def root354Candidates6 : List Nat := [8087]
set_option maxRecDepth 8192 in
theorem root354Candidates6_checked : candidates edge 4 2399 4517 root354Tail7 winner.bound = root354Candidates6 := by
  unfold candidates
  have row : edge 4517 = (SortedNatTree.contains FastRows.row4517) := by
    change Data.row4517.contains = (SortedNatTree.contains FastRows.row4517)
    exact FastRows.row4517_checked.symm
  rw [row]
  decide
theorem root354Step6 : scan edge (search edge 3) 4 [2399] 2399 root354Tail6 winner =
    scan edge (search edge 3) 4 [2399] 2399 root354Tail7 winner := by
  rw [root354Tail6, scan, if_neg (by decide), root354Candidates6_checked]
  change scan edge (search edge 3) 4 [2399] 2399 root354Tail7
    (if 3 ≤ root354Candidates6.length then search edge 3 [2399, 4517] root354Candidates6 6916 winner else winner) = _
  rw [if_neg (by decide)]
def root354Candidates7 : List Nat := [15233]
set_option maxRecDepth 8192 in
theorem root354Candidates7_checked : candidates edge 4 2399 5717 root354Tail8 winner.bound = root354Candidates7 := by
  unfold candidates
  have row : edge 5717 = (SortedNatTree.contains FastRows.row5717) := by
    change Data.row5717.contains = (SortedNatTree.contains FastRows.row5717)
    exact FastRows.row5717_checked.symm
  rw [row]
  decide
theorem root354Step7 : scan edge (search edge 3) 4 [2399] 2399 root354Tail7 winner =
    scan edge (search edge 3) 4 [2399] 2399 root354Tail8 winner := by
  rw [root354Tail7, scan, if_neg (by decide), root354Candidates7_checked]
  change scan edge (search edge 3) 4 [2399] 2399 root354Tail8
    (if 3 ≤ root354Candidates7.length then search edge 3 [2399, 5717] root354Candidates7 8116 winner else winner) = _
  rw [if_neg (by decide)]
theorem root354Scan8 : scan edge (search edge 3) 4 [2399] 2399 root354Tail8 winner = winner := by rw [root354Tail8, scan, if_pos (by decide)]
theorem root354Scan7 : scan edge (search edge 3) 4 [2399] 2399 root354Tail7 winner = winner :=
  (root354Step7).trans root354Scan8
theorem root354Scan6 : scan edge (search edge 3) 4 [2399] 2399 root354Tail6 winner = winner :=
  (root354Step6).trans root354Scan7
theorem root354Scan5 : scan edge (search edge 3) 4 [2399] 2399 root354Tail5 winner = winner :=
  (root354Step5).trans root354Scan6
theorem root354Scan4 : scan edge (search edge 3) 4 [2399] 2399 root354Tail4 winner = winner :=
  (root354Step4).trans root354Scan5
theorem root354Scan3 : scan edge (search edge 3) 4 [2399] 2399 root354Tail3 winner = winner :=
  (root354Step3).trans root354Scan4
theorem root354Scan2 : scan edge (search edge 3) 4 [2399] 2399 root354Tail2 winner = winner :=
  (root354Step2).trans root354Scan3
theorem root354Scan1 : scan edge (search edge 3) 4 [2399] 2399 root354Tail1 winner = winner :=
  (root354Step1).trans root354Scan2
theorem root354Scan0 : scan edge (search edge 3) 4 [2399] 2399 root354Tail0 winner = winner :=
  (root354Step0).trans root354Scan1
set_option maxRecDepth 4096 in
theorem root354Branch : search edge 4 [2399] root354Candidates 2399 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2399] 2399 root354Tail0 winner = winner
  exact root354Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
