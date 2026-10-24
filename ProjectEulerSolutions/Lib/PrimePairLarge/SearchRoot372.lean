import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root372Candidates : List Nat := Data.row2557
set_option maxRecDepth 8192 in
theorem root372Candidates_checked : candidates edge 5 0 2557 tail373 winner.bound = root372Candidates := by
  apply candidates_eq_row edge 5 0 2557 tail373 root372Candidates winner.bound tail373_ordered
  · decide
  · decide
  · funext q
    rfl
def root372Tail36 : List Nat := []
def root372Tail35 : List Nat := 23311 :: root372Tail36
def root372Tail34 : List Nat := 22639 :: root372Tail35
def root372Tail33 : List Nat := 19963 :: root372Tail34
def root372Tail32 : List Nat := 19801 :: root372Tail33
def root372Tail31 : List Nat := 19483 :: root372Tail32
def root372Tail30 : List Nat := 18313 :: root372Tail31
def root372Tail29 : List Nat := 16987 :: root372Tail30
def root372Tail28 : List Nat := 16843 :: root372Tail29
def root372Tail27 : List Nat := 15877 :: root372Tail28
def root372Tail26 : List Nat := 13873 :: root372Tail27
def root372Tail25 : List Nat := 13477 :: root372Tail26
def root372Tail24 : List Nat := 12973 :: root372Tail25
def root372Tail23 : List Nat := 12157 :: root372Tail24
def root372Tail22 : List Nat := 12097 :: root372Tail23
def root372Tail21 : List Nat := 12049 :: root372Tail22
def root372Tail20 : List Nat := 11689 :: root372Tail21
def root372Tail19 : List Nat := 11527 :: root372Tail20
def root372Tail18 : List Nat := 11173 :: root372Tail19
def root372Tail17 : List Nat := 10009 :: root372Tail18
def root372Tail16 : List Nat := 8821 :: root372Tail17
def root372Tail15 : List Nat := 8521 :: root372Tail16
def root372Tail14 : List Nat := 8191 :: root372Tail15
def root372Tail13 : List Nat := 7873 :: root372Tail14
def root372Tail12 : List Nat := 7789 :: root372Tail13
def root372Tail11 : List Nat := 7687 :: root372Tail12
def root372Tail10 : List Nat := 7369 :: root372Tail11
def root372Tail9 : List Nat := 6997 :: root372Tail10
def root372Tail8 : List Nat := 6871 :: root372Tail9
def root372Tail7 : List Nat := 6763 :: root372Tail8
def root372Tail6 : List Nat := 5749 :: root372Tail7
def root372Tail5 : List Nat := 4957 :: root372Tail6
def root372Tail4 : List Nat := 4657 :: root372Tail5
def root372Tail3 : List Nat := 4639 :: root372Tail4
def root372Tail2 : List Nat := 4243 :: root372Tail3
def root372Tail1 : List Nat := 3313 :: root372Tail2
def root372Tail0 : List Nat := 2683 :: root372Tail1
def root372Candidates0 : List Nat := [11689, 12973]
set_option maxRecDepth 8192 in
theorem root372Candidates0_checked : candidates edge 4 2557 2683 root372Tail1 winner.bound = root372Candidates0 := by
  unfold candidates
  have row : edge 2683 = (SortedNatTree.contains FastRows.row2683) := by
    change Data.row2683.contains = (SortedNatTree.contains FastRows.row2683)
    exact FastRows.row2683_checked.symm
  rw [row]
  decide
theorem root372Step0 : scan edge (search edge 3) 4 [2557] 2557 root372Tail0 winner =
    scan edge (search edge 3) 4 [2557] 2557 root372Tail1 winner := by
  rw [root372Tail0, scan, if_neg (by decide), root372Candidates0_checked]
  change scan edge (search edge 3) 4 [2557] 2557 root372Tail1
    (if 3 ≤ root372Candidates0.length then search edge 3 [2557, 2683] root372Candidates0 5240 winner else winner) = _
  rw [if_neg (by decide)]
def root372Candidates1 : List Nat := [6871]
set_option maxRecDepth 8192 in
theorem root372Candidates1_checked : candidates edge 4 2557 3313 root372Tail2 winner.bound = root372Candidates1 := by
  unfold candidates
  have row : edge 3313 = (SortedNatTree.contains FastRows.row3313) := by
    change Data.row3313.contains = (SortedNatTree.contains FastRows.row3313)
    exact FastRows.row3313_checked.symm
  rw [row]
  decide
theorem root372Step1 : scan edge (search edge 3) 4 [2557] 2557 root372Tail1 winner =
    scan edge (search edge 3) 4 [2557] 2557 root372Tail2 winner := by
  rw [root372Tail1, scan, if_neg (by decide), root372Candidates1_checked]
  change scan edge (search edge 3) 4 [2557] 2557 root372Tail2
    (if 3 ≤ root372Candidates1.length then search edge 3 [2557, 3313] root372Candidates1 5870 winner else winner) = _
  rw [if_neg (by decide)]
def root372Candidates2 : List Nat := [6997]
set_option maxRecDepth 8192 in
theorem root372Candidates2_checked : candidates edge 4 2557 4243 root372Tail3 winner.bound = root372Candidates2 := by
  unfold candidates
  have row : edge 4243 = (SortedNatTree.contains FastRows.row4243) := by
    change Data.row4243.contains = (SortedNatTree.contains FastRows.row4243)
    exact FastRows.row4243_checked.symm
  rw [row]
  decide
theorem root372Step2 : scan edge (search edge 3) 4 [2557] 2557 root372Tail2 winner =
    scan edge (search edge 3) 4 [2557] 2557 root372Tail3 winner := by
  rw [root372Tail2, scan, if_neg (by decide), root372Candidates2_checked]
  change scan edge (search edge 3) 4 [2557] 2557 root372Tail3
    (if 3 ≤ root372Candidates2.length then search edge 3 [2557, 4243] root372Candidates2 6800 winner else winner) = _
  rw [if_neg (by decide)]
def root372Candidates3 : List Nat := [5749]
set_option maxRecDepth 8192 in
theorem root372Candidates3_checked : candidates edge 4 2557 4639 root372Tail4 winner.bound = root372Candidates3 := by
  unfold candidates
  have row : edge 4639 = (SortedNatTree.contains FastRows.row4639) := by
    change Data.row4639.contains = (SortedNatTree.contains FastRows.row4639)
    exact FastRows.row4639_checked.symm
  rw [row]
  decide
theorem root372Step3 : scan edge (search edge 3) 4 [2557] 2557 root372Tail3 winner =
    scan edge (search edge 3) 4 [2557] 2557 root372Tail4 winner := by
  rw [root372Tail3, scan, if_neg (by decide), root372Candidates3_checked]
  change scan edge (search edge 3) 4 [2557] 2557 root372Tail4
    (if 3 ≤ root372Candidates3.length then search edge 3 [2557, 4639] root372Candidates3 7196 winner else winner) = _
  rw [if_neg (by decide)]
def root372Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root372Candidates4_checked : candidates edge 4 2557 4657 root372Tail5 winner.bound = root372Candidates4 := by
  unfold candidates
  have row : edge 4657 = (SortedNatTree.contains FastRows.row4657) := by
    change Data.row4657.contains = (SortedNatTree.contains FastRows.row4657)
    exact FastRows.row4657_checked.symm
  rw [row]
  decide
theorem root372Step4 : scan edge (search edge 3) 4 [2557] 2557 root372Tail4 winner =
    scan edge (search edge 3) 4 [2557] 2557 root372Tail5 winner := by
  rw [root372Tail4, scan, if_neg (by decide), root372Candidates4_checked]
  change scan edge (search edge 3) 4 [2557] 2557 root372Tail5
    (if 3 ≤ root372Candidates4.length then search edge 3 [2557, 4657] root372Candidates4 7214 winner else winner) = _
  rw [if_neg (by decide)]
def root372Candidates5 : List Nat := [18313]
set_option maxRecDepth 8192 in
theorem root372Candidates5_checked : candidates edge 4 2557 4957 root372Tail6 winner.bound = root372Candidates5 := by
  unfold candidates
  have row : edge 4957 = (SortedNatTree.contains FastRows.row4957) := by
    change Data.row4957.contains = (SortedNatTree.contains FastRows.row4957)
    exact FastRows.row4957_checked.symm
  rw [row]
  decide
theorem root372Step5 : scan edge (search edge 3) 4 [2557] 2557 root372Tail5 winner =
    scan edge (search edge 3) 4 [2557] 2557 root372Tail6 winner := by
  rw [root372Tail5, scan, if_neg (by decide), root372Candidates5_checked]
  change scan edge (search edge 3) 4 [2557] 2557 root372Tail6
    (if 3 ≤ root372Candidates5.length then search edge 3 [2557, 4957] root372Candidates5 7514 winner else winner) = _
  rw [if_neg (by decide)]
def root372Candidates6 : List Nat := [13873, 16987]
set_option maxRecDepth 8192 in
theorem root372Candidates6_checked : candidates edge 4 2557 5749 root372Tail7 winner.bound = root372Candidates6 := by
  unfold candidates
  have row : edge 5749 = (SortedNatTree.contains FastRows.row5749) := by
    change Data.row5749.contains = (SortedNatTree.contains FastRows.row5749)
    exact FastRows.row5749_checked.symm
  rw [row]
  decide
theorem root372Step6 : scan edge (search edge 3) 4 [2557] 2557 root372Tail6 winner =
    scan edge (search edge 3) 4 [2557] 2557 root372Tail7 winner := by
  rw [root372Tail6, scan, if_neg (by decide), root372Candidates6_checked]
  change scan edge (search edge 3) 4 [2557] 2557 root372Tail7
    (if 3 ≤ root372Candidates6.length then search edge 3 [2557, 5749] root372Candidates6 8306 winner else winner) = _
  rw [if_neg (by decide)]
theorem root372Scan7 : scan edge (search edge 3) 4 [2557] 2557 root372Tail7 winner = winner := by rw [root372Tail7, scan, if_pos (by decide)]
theorem root372Scan6 : scan edge (search edge 3) 4 [2557] 2557 root372Tail6 winner = winner :=
  (root372Step6).trans root372Scan7
theorem root372Scan5 : scan edge (search edge 3) 4 [2557] 2557 root372Tail5 winner = winner :=
  (root372Step5).trans root372Scan6
theorem root372Scan4 : scan edge (search edge 3) 4 [2557] 2557 root372Tail4 winner = winner :=
  (root372Step4).trans root372Scan5
theorem root372Scan3 : scan edge (search edge 3) 4 [2557] 2557 root372Tail3 winner = winner :=
  (root372Step3).trans root372Scan4
theorem root372Scan2 : scan edge (search edge 3) 4 [2557] 2557 root372Tail2 winner = winner :=
  (root372Step2).trans root372Scan3
theorem root372Scan1 : scan edge (search edge 3) 4 [2557] 2557 root372Tail1 winner = winner :=
  (root372Step1).trans root372Scan2
theorem root372Scan0 : scan edge (search edge 3) 4 [2557] 2557 root372Tail0 winner = winner :=
  (root372Step0).trans root372Scan1
set_option maxRecDepth 4096 in
theorem root372Branch : search edge 4 [2557] root372Candidates 2557 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2557] 2557 root372Tail0 winner = winner
  exact root372Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
