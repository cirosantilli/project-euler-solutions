import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root472Candidates : List Nat := Data.row3371
set_option maxRecDepth 8192 in
theorem root472Candidates_checked : candidates edge 5 0 3371 tail473 winner.bound = root472Candidates := by
  apply candidates_eq_row edge 5 0 3371 tail473 root472Candidates winner.bound tail473_ordered
  · decide
  · decide
  · funext q
    rfl
def root472Tail38 : List Nat := []
def root472Tail37 : List Nat := 21737 :: root472Tail38
def root472Tail36 : List Nat := 21383 :: root472Tail37
def root472Tail35 : List Nat := 20963 :: root472Tail36
def root472Tail34 : List Nat := 20549 :: root472Tail35
def root472Tail33 : List Nat := 20507 :: root472Tail34
def root472Tail32 : List Nat := 19583 :: root472Tail33
def root472Tail31 : List Nat := 19301 :: root472Tail32
def root472Tail30 : List Nat := 18461 :: root472Tail31
def root472Tail29 : List Nat := 18443 :: root472Tail30
def root472Tail28 : List Nat := 17747 :: root472Tail29
def root472Tail27 : List Nat := 17609 :: root472Tail28
def root472Tail26 : List Nat := 17321 :: root472Tail27
def root472Tail25 : List Nat := 16931 :: root472Tail26
def root472Tail24 : List Nat := 16823 :: root472Tail25
def root472Tail23 : List Nat := 16691 :: root472Tail24
def root472Tail22 : List Nat := 16067 :: root472Tail23
def root472Tail21 : List Nat := 15809 :: root472Tail22
def root472Tail20 : List Nat := 15299 :: root472Tail21
def root472Tail19 : List Nat := 15137 :: root472Tail20
def root472Tail18 : List Nat := 12239 :: root472Tail19
def root472Tail17 : List Nat := 12101 :: root472Tail18
def root472Tail16 : List Nat := 12071 :: root472Tail17
def root472Tail15 : List Nat := 11987 :: root472Tail16
def root472Tail14 : List Nat := 11273 :: root472Tail15
def root472Tail13 : List Nat := 10259 :: root472Tail14
def root472Tail12 : List Nat := 9677 :: root472Tail13
def root472Tail11 : List Nat := 6983 :: root472Tail12
def root472Tail10 : List Nat := 6359 :: root472Tail11
def root472Tail9 : List Nat := 6029 :: root472Tail10
def root472Tail8 : List Nat := 5849 :: root472Tail9
def root472Tail7 : List Nat := 5393 :: root472Tail8
def root472Tail6 : List Nat := 5153 :: root472Tail7
def root472Tail5 : List Nat := 5009 :: root472Tail6
def root472Tail4 : List Nat := 4973 :: root472Tail5
def root472Tail3 : List Nat := 4799 :: root472Tail4
def root472Tail2 : List Nat := 4649 :: root472Tail3
def root472Tail1 : List Nat := 4229 :: root472Tail2
def root472Tail0 : List Nat := 3413 :: root472Tail1
def root472Candidates0 : List Nat := [4649, 5009, 12239]
set_option maxRecDepth 8192 in
theorem root472Candidates0_checked : candidates edge 4 3371 3413 root472Tail1 winner.bound = root472Candidates0 := by
  unfold candidates
  have row : edge 3413 = (SortedNatTree.contains FastRows.row3413) := by
    change Data.row3413.contains = (SortedNatTree.contains FastRows.row3413)
    exact FastRows.row3413_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root472Child0 : search edge 3 [3371, 3413] root472Candidates0 6784 winner = winner := by decide
theorem root472Step0 : scan edge (search edge 3) 4 [3371] 3371 root472Tail0 winner =
    scan edge (search edge 3) 4 [3371] 3371 root472Tail1 winner := by
  rw [root472Tail0, scan, if_neg (by decide), root472Candidates0_checked]
  change scan edge (search edge 3) 4 [3371] 3371 root472Tail1
    (if 3 ≤ root472Candidates0.length then search edge 3 [3371, 3413] root472Candidates0 6784 winner else winner) = _
  rw [if_pos (by decide), root472Child0]
def root472Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root472Candidates1_checked : candidates edge 4 3371 4229 root472Tail2 winner.bound = root472Candidates1 := by
  unfold candidates
  have row : edge 4229 = (SortedNatTree.contains FastRows.row4229) := by
    change Data.row4229.contains = (SortedNatTree.contains FastRows.row4229)
    exact FastRows.row4229_checked.symm
  rw [row]
  decide
theorem root472Step1 : scan edge (search edge 3) 4 [3371] 3371 root472Tail1 winner =
    scan edge (search edge 3) 4 [3371] 3371 root472Tail2 winner := by
  rw [root472Tail1, scan, if_neg (by decide), root472Candidates1_checked]
  change scan edge (search edge 3) 4 [3371] 3371 root472Tail2
    (if 3 ≤ root472Candidates1.length then search edge 3 [3371, 4229] root472Candidates1 7600 winner else winner) = _
  rw [if_neg (by decide)]
def root472Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root472Candidates2_checked : candidates edge 4 3371 4649 root472Tail3 winner.bound = root472Candidates2 := by
  unfold candidates
  have row : edge 4649 = (SortedNatTree.contains FastRows.row4649) := by
    change Data.row4649.contains = (SortedNatTree.contains FastRows.row4649)
    exact FastRows.row4649_checked.symm
  rw [row]
  decide
theorem root472Step2 : scan edge (search edge 3) 4 [3371] 3371 root472Tail2 winner =
    scan edge (search edge 3) 4 [3371] 3371 root472Tail3 winner := by
  rw [root472Tail2, scan, if_neg (by decide), root472Candidates2_checked]
  change scan edge (search edge 3) 4 [3371] 3371 root472Tail3
    (if 3 ≤ root472Candidates2.length then search edge 3 [3371, 4649] root472Candidates2 8020 winner else winner) = _
  rw [if_neg (by decide)]
def root472Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root472Candidates3_checked : candidates edge 4 3371 4799 root472Tail4 winner.bound = root472Candidates3 := by
  unfold candidates
  have row : edge 4799 = (SortedNatTree.contains FastRows.row4799) := by
    change Data.row4799.contains = (SortedNatTree.contains FastRows.row4799)
    exact FastRows.row4799_checked.symm
  rw [row]
  decide
theorem root472Step3 : scan edge (search edge 3) 4 [3371] 3371 root472Tail3 winner =
    scan edge (search edge 3) 4 [3371] 3371 root472Tail4 winner := by
  rw [root472Tail3, scan, if_neg (by decide), root472Candidates3_checked]
  change scan edge (search edge 3) 4 [3371] 3371 root472Tail4
    (if 3 ≤ root472Candidates3.length then search edge 3 [3371, 4799] root472Candidates3 8170 winner else winner) = _
  rw [if_neg (by decide)]
def root472Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root472Candidates4_checked : candidates edge 4 3371 4973 root472Tail5 winner.bound = root472Candidates4 := by
  unfold candidates
  have row : edge 4973 = (SortedNatTree.contains FastRows.row4973) := by
    change Data.row4973.contains = (SortedNatTree.contains FastRows.row4973)
    exact FastRows.row4973_checked.symm
  rw [row]
  decide
theorem root472Step4 : scan edge (search edge 3) 4 [3371] 3371 root472Tail4 winner =
    scan edge (search edge 3) 4 [3371] 3371 root472Tail5 winner := by
  rw [root472Tail4, scan, if_neg (by decide), root472Candidates4_checked]
  change scan edge (search edge 3) 4 [3371] 3371 root472Tail5
    (if 3 ≤ root472Candidates4.length then search edge 3 [3371, 4973] root472Candidates4 8344 winner else winner) = _
  rw [if_neg (by decide)]
def root472Candidates5 : List Nat := [5849, 6029]
set_option maxRecDepth 8192 in
theorem root472Candidates5_checked : candidates edge 4 3371 5009 root472Tail6 winner.bound = root472Candidates5 := by
  unfold candidates
  have row : edge 5009 = (SortedNatTree.contains FastRows.row5009) := by
    change Data.row5009.contains = (SortedNatTree.contains FastRows.row5009)
    exact FastRows.row5009_checked.symm
  rw [row]
  decide
theorem root472Step5 : scan edge (search edge 3) 4 [3371] 3371 root472Tail5 winner =
    scan edge (search edge 3) 4 [3371] 3371 root472Tail6 winner := by
  rw [root472Tail5, scan, if_neg (by decide), root472Candidates5_checked]
  change scan edge (search edge 3) 4 [3371] 3371 root472Tail6
    (if 3 ≤ root472Candidates5.length then search edge 3 [3371, 5009] root472Candidates5 8380 winner else winner) = _
  rw [if_neg (by decide)]
def root472Candidates6 : List Nat := [5849]
set_option maxRecDepth 8192 in
theorem root472Candidates6_checked : candidates edge 4 3371 5153 root472Tail7 winner.bound = root472Candidates6 := by
  unfold candidates
  have row : edge 5153 = (SortedNatTree.contains FastRows.row5153) := by
    change Data.row5153.contains = (SortedNatTree.contains FastRows.row5153)
    exact FastRows.row5153_checked.symm
  rw [row]
  decide
theorem root472Step6 : scan edge (search edge 3) 4 [3371] 3371 root472Tail6 winner =
    scan edge (search edge 3) 4 [3371] 3371 root472Tail7 winner := by
  rw [root472Tail6, scan, if_neg (by decide), root472Candidates6_checked]
  change scan edge (search edge 3) 4 [3371] 3371 root472Tail7
    (if 3 ≤ root472Candidates6.length then search edge 3 [3371, 5153] root472Candidates6 8524 winner else winner) = _
  rw [if_neg (by decide)]
def root472Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root472Candidates7_checked : candidates edge 4 3371 5393 root472Tail8 winner.bound = root472Candidates7 := by
  unfold candidates
  have row : edge 5393 = (SortedNatTree.contains FastRows.row5393) := by
    change Data.row5393.contains = (SortedNatTree.contains FastRows.row5393)
    exact FastRows.row5393_checked.symm
  rw [row]
  decide
theorem root472Step7 : scan edge (search edge 3) 4 [3371] 3371 root472Tail7 winner =
    scan edge (search edge 3) 4 [3371] 3371 root472Tail8 winner := by
  rw [root472Tail7, scan, if_neg (by decide), root472Candidates7_checked]
  change scan edge (search edge 3) 4 [3371] 3371 root472Tail8
    (if 3 ≤ root472Candidates7.length then search edge 3 [3371, 5393] root472Candidates7 8764 winner else winner) = _
  rw [if_neg (by decide)]
theorem root472Scan8 : scan edge (search edge 3) 4 [3371] 3371 root472Tail8 winner = winner := by rw [root472Tail8, scan, if_pos (by decide)]
theorem root472Scan7 : scan edge (search edge 3) 4 [3371] 3371 root472Tail7 winner = winner :=
  (root472Step7).trans root472Scan8
theorem root472Scan6 : scan edge (search edge 3) 4 [3371] 3371 root472Tail6 winner = winner :=
  (root472Step6).trans root472Scan7
theorem root472Scan5 : scan edge (search edge 3) 4 [3371] 3371 root472Tail5 winner = winner :=
  (root472Step5).trans root472Scan6
theorem root472Scan4 : scan edge (search edge 3) 4 [3371] 3371 root472Tail4 winner = winner :=
  (root472Step4).trans root472Scan5
theorem root472Scan3 : scan edge (search edge 3) 4 [3371] 3371 root472Tail3 winner = winner :=
  (root472Step3).trans root472Scan4
theorem root472Scan2 : scan edge (search edge 3) 4 [3371] 3371 root472Tail2 winner = winner :=
  (root472Step2).trans root472Scan3
theorem root472Scan1 : scan edge (search edge 3) 4 [3371] 3371 root472Tail1 winner = winner :=
  (root472Step1).trans root472Scan2
theorem root472Scan0 : scan edge (search edge 3) 4 [3371] 3371 root472Tail0 winner = winner :=
  (root472Step0).trans root472Scan1
set_option maxRecDepth 4096 in
theorem root472Branch : search edge 4 [3371] root472Candidates 3371 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3371] 3371 root472Tail0 winner = winner
  exact root472Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
