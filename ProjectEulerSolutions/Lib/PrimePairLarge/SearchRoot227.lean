import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root227Candidates : List Nat := Data.row1451
set_option maxRecDepth 8192 in
theorem root227Candidates_checked : candidates edge 5 0 1451 tail228 winner.bound = root227Candidates := by
  apply candidates_eq_row edge 5 0 1451 tail228 root227Candidates winner.bound tail228_ordered
  · decide
  · decide
  · funext q
    rfl
def root227Tail50 : List Nat := []
def root227Tail49 : List Nat := 23417 :: root227Tail50
def root227Tail48 : List Nat := 23291 :: root227Tail49
def root227Tail47 : List Nat := 22937 :: root227Tail48
def root227Tail46 : List Nat := 22751 :: root227Tail47
def root227Tail45 : List Nat := 22619 :: root227Tail46
def root227Tail44 : List Nat := 21929 :: root227Tail45
def root227Tail43 : List Nat := 20939 :: root227Tail44
def root227Tail42 : List Nat := 19913 :: root227Tail43
def root227Tail41 : List Nat := 18587 :: root227Tail42
def root227Tail40 : List Nat := 18521 :: root227Tail41
def root227Tail39 : List Nat := 18191 :: root227Tail40
def root227Tail38 : List Nat := 17837 :: root227Tail39
def root227Tail37 : List Nat := 17033 :: root227Tail38
def root227Tail36 : List Nat := 16253 :: root227Tail37
def root227Tail35 : List Nat := 16007 :: root227Tail36
def root227Tail34 : List Nat := 15959 :: root227Tail35
def root227Tail33 : List Nat := 15329 :: root227Tail34
def root227Tail32 : List Nat := 15227 :: root227Tail33
def root227Tail31 : List Nat := 14747 :: root227Tail32
def root227Tail30 : List Nat := 14327 :: root227Tail31
def root227Tail29 : List Nat := 14243 :: root227Tail30
def root227Tail28 : List Nat := 14009 :: root227Tail29
def root227Tail27 : List Nat := 13151 :: root227Tail28
def root227Tail26 : List Nat := 12983 :: root227Tail27
def root227Tail25 : List Nat := 12203 :: root227Tail26
def root227Tail24 : List Nat := 11411 :: root227Tail25
def root227Tail23 : List Nat := 10613 :: root227Tail24
def root227Tail22 : List Nat := 10463 :: root227Tail23
def root227Tail21 : List Nat := 9419 :: root227Tail22
def root227Tail20 : List Nat := 9413 :: root227Tail21
def root227Tail19 : List Nat := 8741 :: root227Tail20
def root227Tail18 : List Nat := 8597 :: root227Tail19
def root227Tail17 : List Nat := 7907 :: root227Tail18
def root227Tail16 : List Nat := 7883 :: root227Tail17
def root227Tail15 : List Nat := 7823 :: root227Tail16
def root227Tail14 : List Nat := 7499 :: root227Tail15
def root227Tail13 : List Nat := 7193 :: root227Tail14
def root227Tail12 : List Nat := 7001 :: root227Tail13
def root227Tail11 : List Nat := 6869 :: root227Tail12
def root227Tail10 : List Nat := 6659 :: root227Tail11
def root227Tail9 : List Nat := 6581 :: root227Tail10
def root227Tail8 : List Nat := 5147 :: root227Tail9
def root227Tail7 : List Nat := 4457 :: root227Tail8
def root227Tail6 : List Nat := 4373 :: root227Tail7
def root227Tail5 : List Nat := 3761 :: root227Tail6
def root227Tail4 : List Nat := 3557 :: root227Tail5
def root227Tail3 : List Nat := 3413 :: root227Tail4
def root227Tail2 : List Nat := 2699 :: root227Tail3
def root227Tail1 : List Nat := 2039 :: root227Tail2
def root227Tail0 : List Nat := 1481 :: root227Tail1
def root227Candidates0 : List Nat := [6869, 7823, 8597, 14327]
set_option maxRecDepth 8192 in
theorem root227Candidates0_checked : candidates edge 4 1451 1481 root227Tail1 winner.bound = root227Candidates0 := by
  unfold candidates
  have row : edge 1481 = (SortedNatTree.contains FastRows.row1481) := by
    change Data.row1481.contains = (SortedNatTree.contains FastRows.row1481)
    exact FastRows.row1481_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root227Child0 : search edge 3 [1451, 1481] root227Candidates0 2932 winner = winner := by decide
theorem root227Step0 : scan edge (search edge 3) 4 [1451] 1451 root227Tail0 winner =
    scan edge (search edge 3) 4 [1451] 1451 root227Tail1 winner := by
  rw [root227Tail0, scan, if_neg (by decide), root227Candidates0_checked]
  change scan edge (search edge 3) 4 [1451] 1451 root227Tail1
    (if 3 ≤ root227Candidates0.length then search edge 3 [1451, 1481] root227Candidates0 2932 winner else winner) = _
  rw [if_pos (by decide), root227Child0]
def root227Candidates1 : List Nat := [10463, 15329, 16007]
set_option maxRecDepth 8192 in
theorem root227Candidates1_checked : candidates edge 4 1451 2039 root227Tail2 winner.bound = root227Candidates1 := by
  unfold candidates
  have row : edge 2039 = (SortedNatTree.contains FastRows.row2039) := by
    change Data.row2039.contains = (SortedNatTree.contains FastRows.row2039)
    exact FastRows.row2039_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root227Child1 : search edge 3 [1451, 2039] root227Candidates1 3490 winner = winner := by decide
theorem root227Step1 : scan edge (search edge 3) 4 [1451] 1451 root227Tail1 winner =
    scan edge (search edge 3) 4 [1451] 1451 root227Tail2 winner := by
  rw [root227Tail1, scan, if_neg (by decide), root227Candidates1_checked]
  change scan edge (search edge 3) 4 [1451] 1451 root227Tail2
    (if 3 ≤ root227Candidates1.length then search edge 3 [1451, 2039] root227Candidates1 3490 winner else winner) = _
  rw [if_pos (by decide), root227Child1]
def root227Candidates2 : List Nat := [3413, 3761]
set_option maxRecDepth 8192 in
theorem root227Candidates2_checked : candidates edge 4 1451 2699 root227Tail3 winner.bound = root227Candidates2 := by
  unfold candidates
  have row : edge 2699 = (SortedNatTree.contains FastRows.row2699) := by
    change Data.row2699.contains = (SortedNatTree.contains FastRows.row2699)
    exact FastRows.row2699_checked.symm
  rw [row]
  decide
theorem root227Step2 : scan edge (search edge 3) 4 [1451] 1451 root227Tail2 winner =
    scan edge (search edge 3) 4 [1451] 1451 root227Tail3 winner := by
  rw [root227Tail2, scan, if_neg (by decide), root227Candidates2_checked]
  change scan edge (search edge 3) 4 [1451] 1451 root227Tail3
    (if 3 ≤ root227Candidates2.length then search edge 3 [1451, 2699] root227Candidates2 4150 winner else winner) = _
  rw [if_neg (by decide)]
def root227Candidates3 : List Nat := [3761, 7001, 16007]
set_option maxRecDepth 8192 in
theorem root227Candidates3_checked : candidates edge 4 1451 3413 root227Tail4 winner.bound = root227Candidates3 := by
  unfold candidates
  have row : edge 3413 = (SortedNatTree.contains FastRows.row3413) := by
    change Data.row3413.contains = (SortedNatTree.contains FastRows.row3413)
    exact FastRows.row3413_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root227Child3 : search edge 3 [1451, 3413] root227Candidates3 4864 winner = winner := by decide
theorem root227Step3 : scan edge (search edge 3) 4 [1451] 1451 root227Tail3 winner =
    scan edge (search edge 3) 4 [1451] 1451 root227Tail4 winner := by
  rw [root227Tail3, scan, if_neg (by decide), root227Candidates3_checked]
  change scan edge (search edge 3) 4 [1451] 1451 root227Tail4
    (if 3 ≤ root227Candidates3.length then search edge 3 [1451, 3413] root227Candidates3 4864 winner else winner) = _
  rw [if_pos (by decide), root227Child3]
def root227Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root227Candidates4_checked : candidates edge 4 1451 3557 root227Tail5 winner.bound = root227Candidates4 := by
  unfold candidates
  have row : edge 3557 = (SortedNatTree.contains FastRows.row3557) := by
    change Data.row3557.contains = (SortedNatTree.contains FastRows.row3557)
    exact FastRows.row3557_checked.symm
  rw [row]
  decide
theorem root227Step4 : scan edge (search edge 3) 4 [1451] 1451 root227Tail4 winner =
    scan edge (search edge 3) 4 [1451] 1451 root227Tail5 winner := by
  rw [root227Tail4, scan, if_neg (by decide), root227Candidates4_checked]
  change scan edge (search edge 3) 4 [1451] 1451 root227Tail5
    (if 3 ≤ root227Candidates4.length then search edge 3 [1451, 3557] root227Candidates4 5008 winner else winner) = _
  rw [if_neg (by decide)]
def root227Candidates5 : List Nat := [7499]
set_option maxRecDepth 8192 in
theorem root227Candidates5_checked : candidates edge 4 1451 3761 root227Tail6 winner.bound = root227Candidates5 := by
  unfold candidates
  have row : edge 3761 = (SortedNatTree.contains FastRows.row3761) := by
    change Data.row3761.contains = (SortedNatTree.contains FastRows.row3761)
    exact FastRows.row3761_checked.symm
  rw [row]
  decide
theorem root227Step5 : scan edge (search edge 3) 4 [1451] 1451 root227Tail5 winner =
    scan edge (search edge 3) 4 [1451] 1451 root227Tail6 winner := by
  rw [root227Tail5, scan, if_neg (by decide), root227Candidates5_checked]
  change scan edge (search edge 3) 4 [1451] 1451 root227Tail6
    (if 3 ≤ root227Candidates5.length then search edge 3 [1451, 3761] root227Candidates5 5212 winner else winner) = _
  rw [if_neg (by decide)]
def root227Candidates6 : List Nat := [10463, 12983, 14747]
set_option maxRecDepth 8192 in
theorem root227Candidates6_checked : candidates edge 4 1451 4373 root227Tail7 winner.bound = root227Candidates6 := by
  unfold candidates
  have row : edge 4373 = (SortedNatTree.contains FastRows.row4373) := by
    change Data.row4373.contains = (SortedNatTree.contains FastRows.row4373)
    exact FastRows.row4373_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root227Child6 : search edge 3 [1451, 4373] root227Candidates6 5824 winner = winner := by decide
theorem root227Step6 : scan edge (search edge 3) 4 [1451] 1451 root227Tail6 winner =
    scan edge (search edge 3) 4 [1451] 1451 root227Tail7 winner := by
  rw [root227Tail6, scan, if_neg (by decide), root227Candidates6_checked]
  change scan edge (search edge 3) 4 [1451] 1451 root227Tail7
    (if 3 ≤ root227Candidates6.length then search edge 3 [1451, 4373] root227Candidates6 5824 winner else winner) = _
  rw [if_pos (by decide), root227Child6]
def root227Candidates7 : List Nat := [7499]
set_option maxRecDepth 8192 in
theorem root227Candidates7_checked : candidates edge 4 1451 4457 root227Tail8 winner.bound = root227Candidates7 := by
  unfold candidates
  have row : edge 4457 = (SortedNatTree.contains FastRows.row4457) := by
    change Data.row4457.contains = (SortedNatTree.contains FastRows.row4457)
    exact FastRows.row4457_checked.symm
  rw [row]
  decide
theorem root227Step7 : scan edge (search edge 3) 4 [1451] 1451 root227Tail7 winner =
    scan edge (search edge 3) 4 [1451] 1451 root227Tail8 winner := by
  rw [root227Tail7, scan, if_neg (by decide), root227Candidates7_checked]
  change scan edge (search edge 3) 4 [1451] 1451 root227Tail8
    (if 3 ≤ root227Candidates7.length then search edge 3 [1451, 4457] root227Candidates7 5908 winner else winner) = _
  rw [if_neg (by decide)]
def root227Candidates8 : List Nat := [7823, 15329]
set_option maxRecDepth 8192 in
theorem root227Candidates8_checked : candidates edge 4 1451 5147 root227Tail9 winner.bound = root227Candidates8 := by
  unfold candidates
  have row : edge 5147 = (SortedNatTree.contains FastRows.row5147) := by
    change Data.row5147.contains = (SortedNatTree.contains FastRows.row5147)
    exact FastRows.row5147_checked.symm
  rw [row]
  decide
theorem root227Step8 : scan edge (search edge 3) 4 [1451] 1451 root227Tail8 winner =
    scan edge (search edge 3) 4 [1451] 1451 root227Tail9 winner := by
  rw [root227Tail8, scan, if_neg (by decide), root227Candidates8_checked]
  change scan edge (search edge 3) 4 [1451] 1451 root227Tail9
    (if 3 ≤ root227Candidates8.length then search edge 3 [1451, 5147] root227Candidates8 6598 winner else winner) = _
  rw [if_neg (by decide)]
theorem root227Scan9 : scan edge (search edge 3) 4 [1451] 1451 root227Tail9 winner = winner := by rw [root227Tail9, scan, if_pos (by decide)]
theorem root227Scan8 : scan edge (search edge 3) 4 [1451] 1451 root227Tail8 winner = winner :=
  (root227Step8).trans root227Scan9
theorem root227Scan7 : scan edge (search edge 3) 4 [1451] 1451 root227Tail7 winner = winner :=
  (root227Step7).trans root227Scan8
theorem root227Scan6 : scan edge (search edge 3) 4 [1451] 1451 root227Tail6 winner = winner :=
  (root227Step6).trans root227Scan7
theorem root227Scan5 : scan edge (search edge 3) 4 [1451] 1451 root227Tail5 winner = winner :=
  (root227Step5).trans root227Scan6
theorem root227Scan4 : scan edge (search edge 3) 4 [1451] 1451 root227Tail4 winner = winner :=
  (root227Step4).trans root227Scan5
theorem root227Scan3 : scan edge (search edge 3) 4 [1451] 1451 root227Tail3 winner = winner :=
  (root227Step3).trans root227Scan4
theorem root227Scan2 : scan edge (search edge 3) 4 [1451] 1451 root227Tail2 winner = winner :=
  (root227Step2).trans root227Scan3
theorem root227Scan1 : scan edge (search edge 3) 4 [1451] 1451 root227Tail1 winner = winner :=
  (root227Step1).trans root227Scan2
theorem root227Scan0 : scan edge (search edge 3) 4 [1451] 1451 root227Tail0 winner = winner :=
  (root227Step0).trans root227Scan1
set_option maxRecDepth 4096 in
theorem root227Branch : search edge 4 [1451] root227Candidates 1451 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1451] 1451 root227Tail0 winner = winner
  exact root227Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
