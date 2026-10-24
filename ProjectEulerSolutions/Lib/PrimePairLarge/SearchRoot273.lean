import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root273Candidates : List Nat := Data.row1783
set_option maxRecDepth 8192 in
theorem root273Candidates_checked : candidates edge 5 0 1783 tail274 winner.bound = root273Candidates := by
  apply candidates_eq_row edge 5 0 1783 tail274 root273Candidates winner.bound tail274_ordered
  · decide
  · decide
  · funext q
    rfl
def root273Tail48 : List Nat := []
def root273Tail47 : List Nat := 24151 :: root273Tail48
def root273Tail46 : List Nat := 24061 :: root273Tail47
def root273Tail45 : List Nat := 23773 :: root273Tail46
def root273Tail44 : List Nat := 23767 :: root273Tail45
def root273Tail43 : List Nat := 22003 :: root273Tail44
def root273Tail42 : List Nat := 21937 :: root273Tail43
def root273Tail41 : List Nat := 21391 :: root273Tail42
def root273Tail40 : List Nat := 20551 :: root273Tail41
def root273Tail39 : List Nat := 20161 :: root273Tail40
def root273Tail38 : List Nat := 19489 :: root273Tail39
def root273Tail37 : List Nat := 18367 :: root273Tail38
def root273Tail36 : List Nat := 17443 :: root273Tail37
def root273Tail35 : List Nat := 16903 :: root273Tail36
def root273Tail34 : List Nat := 15919 :: root273Tail35
def root273Tail33 : List Nat := 15727 :: root273Tail34
def root273Tail32 : List Nat := 15583 :: root273Tail33
def root273Tail31 : List Nat := 15121 :: root273Tail32
def root273Tail30 : List Nat := 14947 :: root273Tail31
def root273Tail29 : List Nat := 14713 :: root273Tail30
def root273Tail28 : List Nat := 14551 :: root273Tail29
def root273Tail27 : List Nat := 14341 :: root273Tail28
def root273Tail26 : List Nat := 14293 :: root273Tail27
def root273Tail25 : List Nat := 12391 :: root273Tail26
def root273Tail24 : List Nat := 12109 :: root273Tail25
def root273Tail23 : List Nat := 11719 :: root273Tail24
def root273Tail22 : List Nat := 11689 :: root273Tail23
def root273Tail21 : List Nat := 11587 :: root273Tail22
def root273Tail20 : List Nat := 11083 :: root273Tail21
def root273Tail19 : List Nat := 9721 :: root273Tail20
def root273Tail18 : List Nat := 9067 :: root273Tail19
def root273Tail17 : List Nat := 8647 :: root273Tail18
def root273Tail16 : List Nat := 8461 :: root273Tail17
def root273Tail15 : List Nat := 8209 :: root273Tail16
def root273Tail14 : List Nat := 7129 :: root273Tail15
def root273Tail13 : List Nat := 6823 :: root273Tail14
def root273Tail12 : List Nat := 6373 :: root273Tail13
def root273Tail11 : List Nat := 6277 :: root273Tail12
def root273Tail10 : List Nat := 6211 :: root273Tail11
def root273Tail9 : List Nat := 6121 :: root273Tail10
def root273Tail8 : List Nat := 5011 :: root273Tail9
def root273Tail7 : List Nat := 4639 :: root273Tail8
def root273Tail6 : List Nat := 4273 :: root273Tail7
def root273Tail5 : List Nat := 3769 :: root273Tail6
def root273Tail4 : List Nat := 3643 :: root273Tail5
def root273Tail3 : List Nat := 2953 :: root273Tail4
def root273Tail2 : List Nat := 2791 :: root273Tail3
def root273Tail1 : List Nat := 2707 :: root273Tail2
def root273Tail0 : List Nat := 2053 :: root273Tail1
def root273Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root273Candidates0_checked : candidates edge 4 1783 2053 root273Tail1 winner.bound = root273Candidates0 := by
  unfold candidates
  have row : edge 2053 = (SortedNatTree.contains FastRows.row2053) := by
    change Data.row2053.contains = (SortedNatTree.contains FastRows.row2053)
    exact FastRows.row2053_checked.symm
  rw [row]
  decide
theorem root273Step0 : scan edge (search edge 3) 4 [1783] 1783 root273Tail0 winner =
    scan edge (search edge 3) 4 [1783] 1783 root273Tail1 winner := by
  rw [root273Tail0, scan, if_neg (by decide), root273Candidates0_checked]
  change scan edge (search edge 3) 4 [1783] 1783 root273Tail1
    (if 3 ≤ root273Candidates0.length then search edge 3 [1783, 2053] root273Candidates0 3836 winner else winner) = _
  rw [if_neg (by decide)]
def root273Candidates1 : List Nat := [2791, 4273, 12109, 15583, 16903, 19489]
set_option maxRecDepth 8192 in
theorem root273Candidates1_checked : candidates edge 4 1783 2707 root273Tail2 winner.bound = root273Candidates1 := by
  unfold candidates
  have row : edge 2707 = (SortedNatTree.contains FastRows.row2707) := by
    change Data.row2707.contains = (SortedNatTree.contains FastRows.row2707)
    exact FastRows.row2707_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root273Child1 : search edge 3 [1783, 2707] root273Candidates1 4490 winner = winner := by decide
theorem root273Step1 : scan edge (search edge 3) 4 [1783] 1783 root273Tail1 winner =
    scan edge (search edge 3) 4 [1783] 1783 root273Tail2 winner := by
  rw [root273Tail1, scan, if_neg (by decide), root273Candidates1_checked]
  change scan edge (search edge 3) 4 [1783] 1783 root273Tail2
    (if 3 ≤ root273Candidates1.length then search edge 3 [1783, 2707] root273Candidates1 4490 winner else winner) = _
  rw [if_pos (by decide), root273Child1]
def root273Candidates2 : List Nat := [8647, 12391]
set_option maxRecDepth 8192 in
theorem root273Candidates2_checked : candidates edge 4 1783 2791 root273Tail3 winner.bound = root273Candidates2 := by
  unfold candidates
  have row : edge 2791 = (SortedNatTree.contains FastRows.row2791) := by
    change Data.row2791.contains = (SortedNatTree.contains FastRows.row2791)
    exact FastRows.row2791_checked.symm
  rw [row]
  decide
theorem root273Step2 : scan edge (search edge 3) 4 [1783] 1783 root273Tail2 winner =
    scan edge (search edge 3) 4 [1783] 1783 root273Tail3 winner := by
  rw [root273Tail2, scan, if_neg (by decide), root273Candidates2_checked]
  change scan edge (search edge 3) 4 [1783] 1783 root273Tail3
    (if 3 ≤ root273Candidates2.length then search edge 3 [1783, 2791] root273Candidates2 4574 winner else winner) = _
  rw [if_neg (by decide)]
def root273Candidates3 : List Nat := [3643, 4273, 8461, 15919]
set_option maxRecDepth 8192 in
theorem root273Candidates3_checked : candidates edge 4 1783 2953 root273Tail4 winner.bound = root273Candidates3 := by
  unfold candidates
  have row : edge 2953 = (SortedNatTree.contains FastRows.row2953) := by
    change Data.row2953.contains = (SortedNatTree.contains FastRows.row2953)
    exact FastRows.row2953_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root273Child3 : search edge 3 [1783, 2953] root273Candidates3 4736 winner = winner := by decide
theorem root273Step3 : scan edge (search edge 3) 4 [1783] 1783 root273Tail3 winner =
    scan edge (search edge 3) 4 [1783] 1783 root273Tail4 winner := by
  rw [root273Tail3, scan, if_neg (by decide), root273Candidates3_checked]
  change scan edge (search edge 3) 4 [1783] 1783 root273Tail4
    (if 3 ≤ root273Candidates3.length then search edge 3 [1783, 2953] root273Candidates3 4736 winner else winner) = _
  rw [if_pos (by decide), root273Child3]
def root273Candidates4 : List Nat := [4273, 11587]
set_option maxRecDepth 8192 in
theorem root273Candidates4_checked : candidates edge 4 1783 3643 root273Tail5 winner.bound = root273Candidates4 := by
  unfold candidates
  have row : edge 3643 = (SortedNatTree.contains FastRows.row3643) := by
    change Data.row3643.contains = (SortedNatTree.contains FastRows.row3643)
    exact FastRows.row3643_checked.symm
  rw [row]
  decide
theorem root273Step4 : scan edge (search edge 3) 4 [1783] 1783 root273Tail4 winner =
    scan edge (search edge 3) 4 [1783] 1783 root273Tail5 winner := by
  rw [root273Tail4, scan, if_neg (by decide), root273Candidates4_checked]
  change scan edge (search edge 3) 4 [1783] 1783 root273Tail5
    (if 3 ≤ root273Candidates4.length then search edge 3 [1783, 3643] root273Candidates4 5426 winner else winner) = _
  rw [if_neg (by decide)]
def root273Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root273Candidates5_checked : candidates edge 4 1783 3769 root273Tail6 winner.bound = root273Candidates5 := by
  unfold candidates
  have row : edge 3769 = (SortedNatTree.contains FastRows.row3769) := by
    change Data.row3769.contains = (SortedNatTree.contains FastRows.row3769)
    exact FastRows.row3769_checked.symm
  rw [row]
  decide
theorem root273Step5 : scan edge (search edge 3) 4 [1783] 1783 root273Tail5 winner =
    scan edge (search edge 3) 4 [1783] 1783 root273Tail6 winner := by
  rw [root273Tail5, scan, if_neg (by decide), root273Candidates5_checked]
  change scan edge (search edge 3) 4 [1783] 1783 root273Tail6
    (if 3 ≤ root273Candidates5.length then search edge 3 [1783, 3769] root273Candidates5 5552 winner else winner) = _
  rw [if_neg (by decide)]
def root273Candidates6 : List Nat := [6373, 8461]
set_option maxRecDepth 8192 in
theorem root273Candidates6_checked : candidates edge 4 1783 4273 root273Tail7 winner.bound = root273Candidates6 := by
  unfold candidates
  have row : edge 4273 = (SortedNatTree.contains FastRows.row4273) := by
    change Data.row4273.contains = (SortedNatTree.contains FastRows.row4273)
    exact FastRows.row4273_checked.symm
  rw [row]
  decide
theorem root273Step6 : scan edge (search edge 3) 4 [1783] 1783 root273Tail6 winner =
    scan edge (search edge 3) 4 [1783] 1783 root273Tail7 winner := by
  rw [root273Tail6, scan, if_neg (by decide), root273Candidates6_checked]
  change scan edge (search edge 3) 4 [1783] 1783 root273Tail7
    (if 3 ≤ root273Candidates6.length then search edge 3 [1783, 4273] root273Candidates6 6056 winner else winner) = _
  rw [if_neg (by decide)]
def root273Candidates7 : List Nat := [6121, 6211, 6373]
set_option maxRecDepth 8192 in
theorem root273Candidates7_checked : candidates edge 4 1783 4639 root273Tail8 winner.bound = root273Candidates7 := by
  unfold candidates
  have row : edge 4639 = (SortedNatTree.contains FastRows.row4639) := by
    change Data.row4639.contains = (SortedNatTree.contains FastRows.row4639)
    exact FastRows.row4639_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root273Child7 : search edge 3 [1783, 4639] root273Candidates7 6422 winner = winner := by decide
theorem root273Step7 : scan edge (search edge 3) 4 [1783] 1783 root273Tail7 winner =
    scan edge (search edge 3) 4 [1783] 1783 root273Tail8 winner := by
  rw [root273Tail7, scan, if_neg (by decide), root273Candidates7_checked]
  change scan edge (search edge 3) 4 [1783] 1783 root273Tail8
    (if 3 ≤ root273Candidates7.length then search edge 3 [1783, 4639] root273Candidates7 6422 winner else winner) = _
  rw [if_pos (by decide), root273Child7]
def root273Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root273Candidates8_checked : candidates edge 4 1783 5011 root273Tail9 winner.bound = root273Candidates8 := by
  unfold candidates
  have row : edge 5011 = (SortedNatTree.contains FastRows.row5011) := by
    change Data.row5011.contains = (SortedNatTree.contains FastRows.row5011)
    exact FastRows.row5011_checked.symm
  rw [row]
  decide
theorem root273Step8 : scan edge (search edge 3) 4 [1783] 1783 root273Tail8 winner =
    scan edge (search edge 3) 4 [1783] 1783 root273Tail9 winner := by
  rw [root273Tail8, scan, if_neg (by decide), root273Candidates8_checked]
  change scan edge (search edge 3) 4 [1783] 1783 root273Tail9
    (if 3 ≤ root273Candidates8.length then search edge 3 [1783, 5011] root273Candidates8 6794 winner else winner) = _
  rw [if_neg (by decide)]
theorem root273Scan9 : scan edge (search edge 3) 4 [1783] 1783 root273Tail9 winner = winner := by rw [root273Tail9, scan, if_pos (by decide)]
theorem root273Scan8 : scan edge (search edge 3) 4 [1783] 1783 root273Tail8 winner = winner :=
  (root273Step8).trans root273Scan9
theorem root273Scan7 : scan edge (search edge 3) 4 [1783] 1783 root273Tail7 winner = winner :=
  (root273Step7).trans root273Scan8
theorem root273Scan6 : scan edge (search edge 3) 4 [1783] 1783 root273Tail6 winner = winner :=
  (root273Step6).trans root273Scan7
theorem root273Scan5 : scan edge (search edge 3) 4 [1783] 1783 root273Tail5 winner = winner :=
  (root273Step5).trans root273Scan6
theorem root273Scan4 : scan edge (search edge 3) 4 [1783] 1783 root273Tail4 winner = winner :=
  (root273Step4).trans root273Scan5
theorem root273Scan3 : scan edge (search edge 3) 4 [1783] 1783 root273Tail3 winner = winner :=
  (root273Step3).trans root273Scan4
theorem root273Scan2 : scan edge (search edge 3) 4 [1783] 1783 root273Tail2 winner = winner :=
  (root273Step2).trans root273Scan3
theorem root273Scan1 : scan edge (search edge 3) 4 [1783] 1783 root273Tail1 winner = winner :=
  (root273Step1).trans root273Scan2
theorem root273Scan0 : scan edge (search edge 3) 4 [1783] 1783 root273Tail0 winner = winner :=
  (root273Step0).trans root273Scan1
set_option maxRecDepth 4096 in
theorem root273Branch : search edge 4 [1783] root273Candidates 1783 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1783] 1783 root273Tail0 winner = winner
  exact root273Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
