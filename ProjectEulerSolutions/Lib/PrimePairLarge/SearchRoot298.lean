import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root298Candidates : List Nat := Data.row1993
set_option maxRecDepth 8192 in
theorem root298Candidates_checked : candidates edge 5 0 1993 tail299 winner.bound = root298Candidates := by
  apply candidates_eq_row edge 5 0 1993 tail299 root298Candidates winner.bound tail299_ordered
  · decide
  · decide
  · funext q
    rfl
def root298Tail45 : List Nat := []
def root298Tail44 : List Nat := 23563 :: root298Tail45
def root298Tail43 : List Nat := 22639 :: root298Tail44
def root298Tail42 : List Nat := 22147 :: root298Tail43
def root298Tail41 : List Nat := 22003 :: root298Tail42
def root298Tail40 : List Nat := 21169 :: root298Tail41
def root298Tail39 : List Nat := 21139 :: root298Tail40
def root298Tail38 : List Nat := 21061 :: root298Tail39
def root298Tail37 : List Nat := 20071 :: root298Tail38
def root298Tail36 : List Nat := 19993 :: root298Tail37
def root298Tail35 : List Nat := 19531 :: root298Tail36
def root298Tail34 : List Nat := 18919 :: root298Tail35
def root298Tail33 : List Nat := 18439 :: root298Tail34
def root298Tail32 : List Nat := 17491 :: root298Tail33
def root298Tail31 : List Nat := 17431 :: root298Tail32
def root298Tail30 : List Nat := 16477 :: root298Tail31
def root298Tail29 : List Nat := 16063 :: root298Tail30
def root298Tail28 : List Nat := 15601 :: root298Tail29
def root298Tail27 : List Nat := 15241 :: root298Tail28
def root298Tail26 : List Nat := 14197 :: root298Tail27
def root298Tail25 : List Nat := 13063 :: root298Tail26
def root298Tail24 : List Nat := 12823 :: root298Tail25
def root298Tail23 : List Nat := 12577 :: root298Tail24
def root298Tail22 : List Nat := 12049 :: root298Tail23
def root298Tail21 : List Nat := 11941 :: root298Tail22
def root298Tail20 : List Nat := 11863 :: root298Tail21
def root298Tail19 : List Nat := 9973 :: root298Tail20
def root298Tail18 : List Nat := 9277 :: root298Tail19
def root298Tail17 : List Nat := 8377 :: root298Tail18
def root298Tail16 : List Nat := 8287 :: root298Tail17
def root298Tail15 : List Nat := 8209 :: root298Tail16
def root298Tail14 : List Nat := 7507 :: root298Tail15
def root298Tail13 : List Nat := 6781 :: root298Tail14
def root298Tail12 : List Nat := 6481 :: root298Tail13
def root298Tail11 : List Nat := 6337 :: root298Tail12
def root298Tail10 : List Nat := 6211 :: root298Tail11
def root298Tail9 : List Nat := 6151 :: root298Tail10
def root298Tail8 : List Nat := 5869 :: root298Tail9
def root298Tail7 : List Nat := 5743 :: root298Tail8
def root298Tail6 : List Nat := 5581 :: root298Tail7
def root298Tail5 : List Nat := 5011 :: root298Tail6
def root298Tail4 : List Nat := 4483 :: root298Tail5
def root298Tail3 : List Nat := 4093 :: root298Tail4
def root298Tail2 : List Nat := 3457 :: root298Tail3
def root298Tail1 : List Nat := 3271 :: root298Tail2
def root298Tail0 : List Nat := 2221 :: root298Tail1
def root298Candidates0 : List Nat := [5581, 9973, 15601]
set_option maxRecDepth 8192 in
theorem root298Candidates0_checked : candidates edge 4 1993 2221 root298Tail1 winner.bound = root298Candidates0 := by
  unfold candidates
  have row : edge 2221 = (SortedNatTree.contains FastRows.row2221) := by
    change Data.row2221.contains = (SortedNatTree.contains FastRows.row2221)
    exact FastRows.row2221_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root298Child0 : search edge 3 [1993, 2221] root298Candidates0 4214 winner = winner := by decide
theorem root298Step0 : scan edge (search edge 3) 4 [1993] 1993 root298Tail0 winner =
    scan edge (search edge 3) 4 [1993] 1993 root298Tail1 winner := by
  rw [root298Tail0, scan, if_neg (by decide), root298Candidates0_checked]
  change scan edge (search edge 3) 4 [1993] 1993 root298Tail1
    (if 3 ≤ root298Candidates0.length then search edge 3 [1993, 2221] root298Candidates0 4214 winner else winner) = _
  rw [if_pos (by decide), root298Child0]
def root298Candidates1 : List Nat := [8377]
set_option maxRecDepth 8192 in
theorem root298Candidates1_checked : candidates edge 4 1993 3271 root298Tail2 winner.bound = root298Candidates1 := by
  unfold candidates
  have row : edge 3271 = (SortedNatTree.contains FastRows.row3271) := by
    change Data.row3271.contains = (SortedNatTree.contains FastRows.row3271)
    exact FastRows.row3271_checked.symm
  rw [row]
  decide
theorem root298Step1 : scan edge (search edge 3) 4 [1993] 1993 root298Tail1 winner =
    scan edge (search edge 3) 4 [1993] 1993 root298Tail2 winner := by
  rw [root298Tail1, scan, if_neg (by decide), root298Candidates1_checked]
  change scan edge (search edge 3) 4 [1993] 1993 root298Tail2
    (if 3 ≤ root298Candidates1.length then search edge 3 [1993, 3271] root298Candidates1 5264 winner else winner) = _
  rw [if_neg (by decide)]
def root298Candidates2 : List Nat := [5869, 8209, 8377]
set_option maxRecDepth 8192 in
theorem root298Candidates2_checked : candidates edge 4 1993 3457 root298Tail3 winner.bound = root298Candidates2 := by
  unfold candidates
  have row : edge 3457 = (SortedNatTree.contains FastRows.row3457) := by
    change Data.row3457.contains = (SortedNatTree.contains FastRows.row3457)
    exact FastRows.row3457_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root298Child2 : search edge 3 [1993, 3457] root298Candidates2 5450 winner = winner := by decide
theorem root298Step2 : scan edge (search edge 3) 4 [1993] 1993 root298Tail2 winner =
    scan edge (search edge 3) 4 [1993] 1993 root298Tail3 winner := by
  rw [root298Tail2, scan, if_neg (by decide), root298Candidates2_checked]
  change scan edge (search edge 3) 4 [1993] 1993 root298Tail3
    (if 3 ≤ root298Candidates2.length then search edge 3 [1993, 3457] root298Candidates2 5450 winner else winner) = _
  rw [if_pos (by decide), root298Child2]
def root298Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root298Candidates3_checked : candidates edge 4 1993 4093 root298Tail4 winner.bound = root298Candidates3 := by
  unfold candidates
  have row : edge 4093 = (SortedNatTree.contains FastRows.row4093) := by
    change Data.row4093.contains = (SortedNatTree.contains FastRows.row4093)
    exact FastRows.row4093_checked.symm
  rw [row]
  decide
theorem root298Step3 : scan edge (search edge 3) 4 [1993] 1993 root298Tail3 winner =
    scan edge (search edge 3) 4 [1993] 1993 root298Tail4 winner := by
  rw [root298Tail3, scan, if_neg (by decide), root298Candidates3_checked]
  change scan edge (search edge 3) 4 [1993] 1993 root298Tail4
    (if 3 ≤ root298Candidates3.length then search edge 3 [1993, 4093] root298Candidates3 6086 winner else winner) = _
  rw [if_neg (by decide)]
def root298Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root298Candidates4_checked : candidates edge 4 1993 4483 root298Tail5 winner.bound = root298Candidates4 := by
  unfold candidates
  have row : edge 4483 = (SortedNatTree.contains FastRows.row4483) := by
    change Data.row4483.contains = (SortedNatTree.contains FastRows.row4483)
    exact FastRows.row4483_checked.symm
  rw [row]
  decide
theorem root298Step4 : scan edge (search edge 3) 4 [1993] 1993 root298Tail4 winner =
    scan edge (search edge 3) 4 [1993] 1993 root298Tail5 winner := by
  rw [root298Tail4, scan, if_neg (by decide), root298Candidates4_checked]
  change scan edge (search edge 3) 4 [1993] 1993 root298Tail5
    (if 3 ≤ root298Candidates4.length then search edge 3 [1993, 4483] root298Candidates4 6476 winner else winner) = _
  rw [if_neg (by decide)]
def root298Candidates5 : List Nat := [12823]
set_option maxRecDepth 8192 in
theorem root298Candidates5_checked : candidates edge 4 1993 5011 root298Tail6 winner.bound = root298Candidates5 := by
  unfold candidates
  have row : edge 5011 = (SortedNatTree.contains FastRows.row5011) := by
    change Data.row5011.contains = (SortedNatTree.contains FastRows.row5011)
    exact FastRows.row5011_checked.symm
  rw [row]
  decide
theorem root298Step5 : scan edge (search edge 3) 4 [1993] 1993 root298Tail5 winner =
    scan edge (search edge 3) 4 [1993] 1993 root298Tail6 winner := by
  rw [root298Tail5, scan, if_neg (by decide), root298Candidates5_checked]
  change scan edge (search edge 3) 4 [1993] 1993 root298Tail6
    (if 3 ≤ root298Candidates5.length then search edge 3 [1993, 5011] root298Candidates5 7004 winner else winner) = _
  rw [if_neg (by decide)]
def root298Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root298Candidates6_checked : candidates edge 4 1993 5581 root298Tail7 winner.bound = root298Candidates6 := by
  unfold candidates
  have row : edge 5581 = (SortedNatTree.contains FastRows.row5581) := by
    change Data.row5581.contains = (SortedNatTree.contains FastRows.row5581)
    exact FastRows.row5581_checked.symm
  rw [row]
  decide
theorem root298Step6 : scan edge (search edge 3) 4 [1993] 1993 root298Tail6 winner =
    scan edge (search edge 3) 4 [1993] 1993 root298Tail7 winner := by
  rw [root298Tail6, scan, if_neg (by decide), root298Candidates6_checked]
  change scan edge (search edge 3) 4 [1993] 1993 root298Tail7
    (if 3 ≤ root298Candidates6.length then search edge 3 [1993, 5581] root298Candidates6 7574 winner else winner) = _
  rw [if_neg (by decide)]
def root298Candidates7 : List Nat := [7507, 8287, 8377, 12049]
set_option maxRecDepth 8192 in
theorem root298Candidates7_checked : candidates edge 4 1993 5743 root298Tail8 winner.bound = root298Candidates7 := by
  unfold candidates
  have row : edge 5743 = (SortedNatTree.contains FastRows.row5743) := by
    change Data.row5743.contains = (SortedNatTree.contains FastRows.row5743)
    exact FastRows.row5743_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root298Child7 : search edge 3 [1993, 5743] root298Candidates7 7736 winner = winner := by decide
theorem root298Step7 : scan edge (search edge 3) 4 [1993] 1993 root298Tail7 winner =
    scan edge (search edge 3) 4 [1993] 1993 root298Tail8 winner := by
  rw [root298Tail7, scan, if_neg (by decide), root298Candidates7_checked]
  change scan edge (search edge 3) 4 [1993] 1993 root298Tail8
    (if 3 ≤ root298Candidates7.length then search edge 3 [1993, 5743] root298Candidates7 7736 winner else winner) = _
  rw [if_pos (by decide), root298Child7]
def root298Candidates8 : List Nat := [8209, 12577]
set_option maxRecDepth 8192 in
theorem root298Candidates8_checked : candidates edge 4 1993 5869 root298Tail9 winner.bound = root298Candidates8 := by
  unfold candidates
  have row : edge 5869 = (SortedNatTree.contains FastRows.row5869) := by
    change Data.row5869.contains = (SortedNatTree.contains FastRows.row5869)
    exact FastRows.row5869_checked.symm
  rw [row]
  decide
theorem root298Step8 : scan edge (search edge 3) 4 [1993] 1993 root298Tail8 winner =
    scan edge (search edge 3) 4 [1993] 1993 root298Tail9 winner := by
  rw [root298Tail8, scan, if_neg (by decide), root298Candidates8_checked]
  change scan edge (search edge 3) 4 [1993] 1993 root298Tail9
    (if 3 ≤ root298Candidates8.length then search edge 3 [1993, 5869] root298Candidates8 7862 winner else winner) = _
  rw [if_neg (by decide)]
theorem root298Scan9 : scan edge (search edge 3) 4 [1993] 1993 root298Tail9 winner = winner := by rw [root298Tail9, scan, if_pos (by decide)]
theorem root298Scan8 : scan edge (search edge 3) 4 [1993] 1993 root298Tail8 winner = winner :=
  (root298Step8).trans root298Scan9
theorem root298Scan7 : scan edge (search edge 3) 4 [1993] 1993 root298Tail7 winner = winner :=
  (root298Step7).trans root298Scan8
theorem root298Scan6 : scan edge (search edge 3) 4 [1993] 1993 root298Tail6 winner = winner :=
  (root298Step6).trans root298Scan7
theorem root298Scan5 : scan edge (search edge 3) 4 [1993] 1993 root298Tail5 winner = winner :=
  (root298Step5).trans root298Scan6
theorem root298Scan4 : scan edge (search edge 3) 4 [1993] 1993 root298Tail4 winner = winner :=
  (root298Step4).trans root298Scan5
theorem root298Scan3 : scan edge (search edge 3) 4 [1993] 1993 root298Tail3 winner = winner :=
  (root298Step3).trans root298Scan4
theorem root298Scan2 : scan edge (search edge 3) 4 [1993] 1993 root298Tail2 winner = winner :=
  (root298Step2).trans root298Scan3
theorem root298Scan1 : scan edge (search edge 3) 4 [1993] 1993 root298Tail1 winner = winner :=
  (root298Step1).trans root298Scan2
theorem root298Scan0 : scan edge (search edge 3) 4 [1993] 1993 root298Tail0 winner = winner :=
  (root298Step0).trans root298Scan1
set_option maxRecDepth 4096 in
theorem root298Branch : search edge 4 [1993] root298Candidates 1993 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1993] 1993 root298Tail0 winner = winner
  exact root298Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
