import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root385Candidates : List Nat := Data.row2677
set_option maxRecDepth 8192 in
theorem root385Candidates_checked : candidates edge 5 0 2677 tail386 winner.bound = root385Candidates := by
  apply candidates_eq_row edge 5 0 2677 tail386 root385Candidates winner.bound tail386_ordered
  · decide
  · decide
  · funext q
    rfl
def root385Tail34 : List Nat := []
def root385Tail33 : List Nat := 22153 :: root385Tail34
def root385Tail32 : List Nat := 21613 :: root385Tail33
def root385Tail31 : List Nat := 21031 :: root385Tail32
def root385Tail30 : List Nat := 20443 :: root385Tail31
def root385Tail29 : List Nat := 19819 :: root385Tail30
def root385Tail28 : List Nat := 18097 :: root385Tail29
def root385Tail27 : List Nat := 18013 :: root385Tail28
def root385Tail26 : List Nat := 17929 :: root385Tail27
def root385Tail25 : List Nat := 17623 :: root385Tail26
def root385Tail24 : List Nat := 15493 :: root385Tail25
def root385Tail23 : List Nat := 14767 :: root385Tail24
def root385Tail22 : List Nat := 13879 :: root385Tail23
def root385Tail21 : List Nat := 13291 :: root385Tail22
def root385Tail20 : List Nat := 13177 :: root385Tail21
def root385Tail19 : List Nat := 12007 :: root385Tail20
def root385Tail18 : List Nat := 11701 :: root385Tail19
def root385Tail17 : List Nat := 9859 :: root385Tail18
def root385Tail16 : List Nat := 9043 :: root385Tail17
def root385Tail15 : List Nat := 9007 :: root385Tail16
def root385Tail14 : List Nat := 8707 :: root385Tail15
def root385Tail13 : List Nat := 8419 :: root385Tail14
def root385Tail12 : List Nat := 7741 :: root385Tail13
def root385Tail11 : List Nat := 7369 :: root385Tail12
def root385Tail10 : List Nat := 6961 :: root385Tail11
def root385Tail9 : List Nat := 6247 :: root385Tail10
def root385Tail8 : List Nat := 5851 :: root385Tail9
def root385Tail7 : List Nat := 5737 :: root385Tail8
def root385Tail6 : List Nat := 5569 :: root385Tail7
def root385Tail5 : List Nat := 5563 :: root385Tail6
def root385Tail4 : List Nat := 4261 :: root385Tail5
def root385Tail3 : List Nat := 3469 :: root385Tail4
def root385Tail2 : List Nat := 3049 :: root385Tail3
def root385Tail1 : List Nat := 2971 :: root385Tail2
def root385Tail0 : List Nat := 2833 :: root385Tail1
def root385Candidates0 : List Nat := [5851, 13291, 18097]
set_option maxRecDepth 8192 in
theorem root385Candidates0_checked : candidates edge 4 2677 2833 root385Tail1 winner.bound = root385Candidates0 := by
  unfold candidates
  have row : edge 2833 = (SortedNatTree.contains FastRows.row2833) := by
    change Data.row2833.contains = (SortedNatTree.contains FastRows.row2833)
    exact FastRows.row2833_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root385Child0 : search edge 3 [2677, 2833] root385Candidates0 5510 winner = winner := by decide
theorem root385Step0 : scan edge (search edge 3) 4 [2677] 2677 root385Tail0 winner =
    scan edge (search edge 3) 4 [2677] 2677 root385Tail1 winner := by
  rw [root385Tail0, scan, if_neg (by decide), root385Candidates0_checked]
  change scan edge (search edge 3) 4 [2677] 2677 root385Tail1
    (if 3 ≤ root385Candidates0.length then search edge 3 [2677, 2833] root385Candidates0 5510 winner else winner) = _
  rw [if_pos (by decide), root385Child0]
def root385Candidates1 : List Nat := [6961]
set_option maxRecDepth 8192 in
theorem root385Candidates1_checked : candidates edge 4 2677 2971 root385Tail2 winner.bound = root385Candidates1 := by
  unfold candidates
  have row : edge 2971 = (SortedNatTree.contains FastRows.row2971) := by
    change Data.row2971.contains = (SortedNatTree.contains FastRows.row2971)
    exact FastRows.row2971_checked.symm
  rw [row]
  decide
theorem root385Step1 : scan edge (search edge 3) 4 [2677] 2677 root385Tail1 winner =
    scan edge (search edge 3) 4 [2677] 2677 root385Tail2 winner := by
  rw [root385Tail1, scan, if_neg (by decide), root385Candidates1_checked]
  change scan edge (search edge 3) 4 [2677] 2677 root385Tail2
    (if 3 ≤ root385Candidates1.length then search edge 3 [2677, 2971] root385Candidates1 5648 winner else winner) = _
  rw [if_neg (by decide)]
def root385Candidates2 : List Nat := [12007, 19819]
set_option maxRecDepth 8192 in
theorem root385Candidates2_checked : candidates edge 4 2677 3049 root385Tail3 winner.bound = root385Candidates2 := by
  unfold candidates
  have row : edge 3049 = (SortedNatTree.contains FastRows.row3049) := by
    change Data.row3049.contains = (SortedNatTree.contains FastRows.row3049)
    exact FastRows.row3049_checked.symm
  rw [row]
  decide
theorem root385Step2 : scan edge (search edge 3) 4 [2677] 2677 root385Tail2 winner =
    scan edge (search edge 3) 4 [2677] 2677 root385Tail3 winner := by
  rw [root385Tail2, scan, if_neg (by decide), root385Candidates2_checked]
  change scan edge (search edge 3) 4 [2677] 2677 root385Tail3
    (if 3 ≤ root385Candidates2.length then search edge 3 [2677, 3049] root385Candidates2 5726 winner else winner) = _
  rw [if_neg (by decide)]
def root385Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root385Candidates3_checked : candidates edge 4 2677 3469 root385Tail4 winner.bound = root385Candidates3 := by
  unfold candidates
  have row : edge 3469 = (SortedNatTree.contains FastRows.row3469) := by
    change Data.row3469.contains = (SortedNatTree.contains FastRows.row3469)
    exact FastRows.row3469_checked.symm
  rw [row]
  decide
theorem root385Step3 : scan edge (search edge 3) 4 [2677] 2677 root385Tail3 winner =
    scan edge (search edge 3) 4 [2677] 2677 root385Tail4 winner := by
  rw [root385Tail3, scan, if_neg (by decide), root385Candidates3_checked]
  change scan edge (search edge 3) 4 [2677] 2677 root385Tail4
    (if 3 ≤ root385Candidates3.length then search edge 3 [2677, 3469] root385Candidates3 6146 winner else winner) = _
  rw [if_neg (by decide)]
def root385Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root385Candidates4_checked : candidates edge 4 2677 4261 root385Tail5 winner.bound = root385Candidates4 := by
  unfold candidates
  have row : edge 4261 = (SortedNatTree.contains FastRows.row4261) := by
    change Data.row4261.contains = (SortedNatTree.contains FastRows.row4261)
    exact FastRows.row4261_checked.symm
  rw [row]
  decide
theorem root385Step4 : scan edge (search edge 3) 4 [2677] 2677 root385Tail4 winner =
    scan edge (search edge 3) 4 [2677] 2677 root385Tail5 winner := by
  rw [root385Tail4, scan, if_neg (by decide), root385Candidates4_checked]
  change scan edge (search edge 3) 4 [2677] 2677 root385Tail5
    (if 3 ≤ root385Candidates4.length then search edge 3 [2677, 4261] root385Candidates4 6938 winner else winner) = _
  rw [if_neg (by decide)]
def root385Candidates5 : List Nat := [15493]
set_option maxRecDepth 8192 in
theorem root385Candidates5_checked : candidates edge 4 2677 5563 root385Tail6 winner.bound = root385Candidates5 := by
  unfold candidates
  have row : edge 5563 = (SortedNatTree.contains FastRows.row5563) := by
    change Data.row5563.contains = (SortedNatTree.contains FastRows.row5563)
    exact FastRows.row5563_checked.symm
  rw [row]
  decide
theorem root385Step5 : scan edge (search edge 3) 4 [2677] 2677 root385Tail5 winner =
    scan edge (search edge 3) 4 [2677] 2677 root385Tail6 winner := by
  rw [root385Tail5, scan, if_neg (by decide), root385Candidates5_checked]
  change scan edge (search edge 3) 4 [2677] 2677 root385Tail6
    (if 3 ≤ root385Candidates5.length then search edge 3 [2677, 5563] root385Candidates5 8240 winner else winner) = _
  rw [if_neg (by decide)]
def root385Candidates6 : List Nat := [12007, 13177]
set_option maxRecDepth 8192 in
theorem root385Candidates6_checked : candidates edge 4 2677 5569 root385Tail7 winner.bound = root385Candidates6 := by
  unfold candidates
  have row : edge 5569 = (SortedNatTree.contains FastRows.row5569) := by
    change Data.row5569.contains = (SortedNatTree.contains FastRows.row5569)
    exact FastRows.row5569_checked.symm
  rw [row]
  decide
theorem root385Step6 : scan edge (search edge 3) 4 [2677] 2677 root385Tail6 winner =
    scan edge (search edge 3) 4 [2677] 2677 root385Tail7 winner := by
  rw [root385Tail6, scan, if_neg (by decide), root385Candidates6_checked]
  change scan edge (search edge 3) 4 [2677] 2677 root385Tail7
    (if 3 ≤ root385Candidates6.length then search edge 3 [2677, 5569] root385Candidates6 8246 winner else winner) = _
  rw [if_neg (by decide)]
def root385Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root385Candidates7_checked : candidates edge 4 2677 5737 root385Tail8 winner.bound = root385Candidates7 := by
  unfold candidates
  have row : edge 5737 = (SortedNatTree.contains FastRows.row5737) := by
    change Data.row5737.contains = (SortedNatTree.contains FastRows.row5737)
    exact FastRows.row5737_checked.symm
  rw [row]
  decide
theorem root385Step7 : scan edge (search edge 3) 4 [2677] 2677 root385Tail7 winner =
    scan edge (search edge 3) 4 [2677] 2677 root385Tail8 winner := by
  rw [root385Tail7, scan, if_neg (by decide), root385Candidates7_checked]
  change scan edge (search edge 3) 4 [2677] 2677 root385Tail8
    (if 3 ≤ root385Candidates7.length then search edge 3 [2677, 5737] root385Candidates7 8414 winner else winner) = _
  rw [if_neg (by decide)]
theorem root385Scan8 : scan edge (search edge 3) 4 [2677] 2677 root385Tail8 winner = winner := by rw [root385Tail8, scan, if_pos (by decide)]
theorem root385Scan7 : scan edge (search edge 3) 4 [2677] 2677 root385Tail7 winner = winner :=
  (root385Step7).trans root385Scan8
theorem root385Scan6 : scan edge (search edge 3) 4 [2677] 2677 root385Tail6 winner = winner :=
  (root385Step6).trans root385Scan7
theorem root385Scan5 : scan edge (search edge 3) 4 [2677] 2677 root385Tail5 winner = winner :=
  (root385Step5).trans root385Scan6
theorem root385Scan4 : scan edge (search edge 3) 4 [2677] 2677 root385Tail4 winner = winner :=
  (root385Step4).trans root385Scan5
theorem root385Scan3 : scan edge (search edge 3) 4 [2677] 2677 root385Tail3 winner = winner :=
  (root385Step3).trans root385Scan4
theorem root385Scan2 : scan edge (search edge 3) 4 [2677] 2677 root385Tail2 winner = winner :=
  (root385Step2).trans root385Scan3
theorem root385Scan1 : scan edge (search edge 3) 4 [2677] 2677 root385Tail1 winner = winner :=
  (root385Step1).trans root385Scan2
theorem root385Scan0 : scan edge (search edge 3) 4 [2677] 2677 root385Tail0 winner = winner :=
  (root385Step0).trans root385Scan1
set_option maxRecDepth 4096 in
theorem root385Branch : search edge 4 [2677] root385Candidates 2677 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2677] 2677 root385Tail0 winner = winner
  exact root385Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
