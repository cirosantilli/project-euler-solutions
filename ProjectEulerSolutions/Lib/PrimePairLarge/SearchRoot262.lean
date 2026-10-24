import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root262Candidates : List Nat := Data.row1697
set_option maxRecDepth 8192 in
theorem root262Candidates_checked : candidates edge 5 0 1697 tail263 winner.bound = root262Candidates := by
  apply candidates_eq_row edge 5 0 1697 tail263 root262Candidates winner.bound tail263_ordered
  · decide
  · decide
  · funext q
    rfl
def root262Tail41 : List Nat := []
def root262Tail40 : List Nat := 23327 :: root262Tail41
def root262Tail39 : List Nat := 22901 :: root262Tail40
def root262Tail38 : List Nat := 22109 :: root262Tail39
def root262Tail37 : List Nat := 21401 :: root262Tail38
def root262Tail36 : List Nat := 21011 :: root262Tail37
def root262Tail35 : List Nat := 19661 :: root262Tail36
def root262Tail34 : List Nat := 18797 :: root262Tail35
def root262Tail33 : List Nat := 18713 :: root262Tail34
def root262Tail32 : List Nat := 18617 :: root262Tail33
def root262Tail31 : List Nat := 16901 :: root262Tail32
def root262Tail30 : List Nat := 16061 :: root262Tail31
def root262Tail29 : List Nat := 15629 :: root262Tail30
def root262Tail28 : List Nat := 14879 :: root262Tail29
def root262Tail27 : List Nat := 13697 :: root262Tail28
def root262Tail26 : List Nat := 13421 :: root262Tail27
def root262Tail25 : List Nat := 13151 :: root262Tail26
def root262Tail24 : List Nat := 12953 :: root262Tail25
def root262Tail23 : List Nat := 12527 :: root262Tail24
def root262Tail22 : List Nat := 12113 :: root262Tail23
def root262Tail21 : List Nat := 11813 :: root262Tail22
def root262Tail20 : List Nat := 11681 :: root262Tail21
def root262Tail19 : List Nat := 11483 :: root262Tail20
def root262Tail18 : List Nat := 10391 :: root262Tail19
def root262Tail17 : List Nat := 10253 :: root262Tail18
def root262Tail16 : List Nat := 9887 :: root262Tail17
def root262Tail15 : List Nat := 9467 :: root262Tail16
def root262Tail14 : List Nat := 9419 :: root262Tail15
def root262Tail13 : List Nat := 8237 :: root262Tail14
def root262Tail12 : List Nat := 7901 :: root262Tail13
def root262Tail11 : List Nat := 7643 :: root262Tail12
def root262Tail10 : List Nat := 7193 :: root262Tail11
def root262Tail9 : List Nat := 6689 :: root262Tail10
def root262Tail8 : List Nat := 5477 :: root262Tail9
def root262Tail7 : List Nat := 5099 :: root262Tail8
def root262Tail6 : List Nat := 5087 :: root262Tail7
def root262Tail5 : List Nat := 4931 :: root262Tail6
def root262Tail4 : List Nat := 3527 :: root262Tail5
def root262Tail3 : List Nat := 3041 :: root262Tail4
def root262Tail2 : List Nat := 2741 :: root262Tail3
def root262Tail1 : List Nat := 2027 :: root262Tail2
def root262Tail0 : List Nat := 1979 :: root262Tail1
def root262Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root262Candidates0_checked : candidates edge 4 1697 1979 root262Tail1 winner.bound = root262Candidates0 := by
  unfold candidates
  have row : edge 1979 = (SortedNatTree.contains FastRows.row1979) := by
    change Data.row1979.contains = (SortedNatTree.contains FastRows.row1979)
    exact FastRows.row1979_checked.symm
  rw [row]
  decide
theorem root262Step0 : scan edge (search edge 3) 4 [1697] 1697 root262Tail0 winner =
    scan edge (search edge 3) 4 [1697] 1697 root262Tail1 winner := by
  rw [root262Tail0, scan, if_neg (by decide), root262Candidates0_checked]
  change scan edge (search edge 3) 4 [1697] 1697 root262Tail1
    (if 3 ≤ root262Candidates0.length then search edge 3 [1697, 1979] root262Candidates0 3676 winner else winner) = _
  rw [if_neg (by decide)]
def root262Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root262Candidates1_checked : candidates edge 4 1697 2027 root262Tail2 winner.bound = root262Candidates1 := by
  unfold candidates
  have row : edge 2027 = (SortedNatTree.contains FastRows.row2027) := by
    change Data.row2027.contains = (SortedNatTree.contains FastRows.row2027)
    exact FastRows.row2027_checked.symm
  rw [row]
  decide
theorem root262Step1 : scan edge (search edge 3) 4 [1697] 1697 root262Tail1 winner =
    scan edge (search edge 3) 4 [1697] 1697 root262Tail2 winner := by
  rw [root262Tail1, scan, if_neg (by decide), root262Candidates1_checked]
  change scan edge (search edge 3) 4 [1697] 1697 root262Tail2
    (if 3 ≤ root262Candidates1.length then search edge 3 [1697, 2027] root262Candidates1 3724 winner else winner) = _
  rw [if_neg (by decide)]
def root262Candidates2 : List Nat := [6689, 9467]
set_option maxRecDepth 8192 in
theorem root262Candidates2_checked : candidates edge 4 1697 2741 root262Tail3 winner.bound = root262Candidates2 := by
  unfold candidates
  have row : edge 2741 = (SortedNatTree.contains FastRows.row2741) := by
    change Data.row2741.contains = (SortedNatTree.contains FastRows.row2741)
    exact FastRows.row2741_checked.symm
  rw [row]
  decide
theorem root262Step2 : scan edge (search edge 3) 4 [1697] 1697 root262Tail2 winner =
    scan edge (search edge 3) 4 [1697] 1697 root262Tail3 winner := by
  rw [root262Tail2, scan, if_neg (by decide), root262Candidates2_checked]
  change scan edge (search edge 3) 4 [1697] 1697 root262Tail3
    (if 3 ≤ root262Candidates2.length then search edge 3 [1697, 2741] root262Candidates2 4438 winner else winner) = _
  rw [if_neg (by decide)]
def root262Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root262Candidates3_checked : candidates edge 4 1697 3041 root262Tail4 winner.bound = root262Candidates3 := by
  unfold candidates
  have row : edge 3041 = (SortedNatTree.contains FastRows.row3041) := by
    change Data.row3041.contains = (SortedNatTree.contains FastRows.row3041)
    exact FastRows.row3041_checked.symm
  rw [row]
  decide
theorem root262Step3 : scan edge (search edge 3) 4 [1697] 1697 root262Tail3 winner =
    scan edge (search edge 3) 4 [1697] 1697 root262Tail4 winner := by
  rw [root262Tail3, scan, if_neg (by decide), root262Candidates3_checked]
  change scan edge (search edge 3) 4 [1697] 1697 root262Tail4
    (if 3 ≤ root262Candidates3.length then search edge 3 [1697, 3041] root262Candidates3 4738 winner else winner) = _
  rw [if_neg (by decide)]
def root262Candidates4 : List Nat := [7901, 18617]
set_option maxRecDepth 8192 in
theorem root262Candidates4_checked : candidates edge 4 1697 3527 root262Tail5 winner.bound = root262Candidates4 := by
  unfold candidates
  have row : edge 3527 = (SortedNatTree.contains FastRows.row3527) := by
    change Data.row3527.contains = (SortedNatTree.contains FastRows.row3527)
    exact FastRows.row3527_checked.symm
  rw [row]
  decide
theorem root262Step4 : scan edge (search edge 3) 4 [1697] 1697 root262Tail4 winner =
    scan edge (search edge 3) 4 [1697] 1697 root262Tail5 winner := by
  rw [root262Tail4, scan, if_neg (by decide), root262Candidates4_checked]
  change scan edge (search edge 3) 4 [1697] 1697 root262Tail5
    (if 3 ≤ root262Candidates4.length then search edge 3 [1697, 3527] root262Candidates4 5224 winner else winner) = _
  rw [if_neg (by decide)]
def root262Candidates5 : List Nat := [10253, 11681, 12953]
set_option maxRecDepth 8192 in
theorem root262Candidates5_checked : candidates edge 4 1697 4931 root262Tail6 winner.bound = root262Candidates5 := by
  unfold candidates
  have row : edge 4931 = (SortedNatTree.contains FastRows.row4931) := by
    change Data.row4931.contains = (SortedNatTree.contains FastRows.row4931)
    exact FastRows.row4931_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root262Child5 : search edge 3 [1697, 4931] root262Candidates5 6628 winner = winner := by decide
theorem root262Step5 : scan edge (search edge 3) 4 [1697] 1697 root262Tail5 winner =
    scan edge (search edge 3) 4 [1697] 1697 root262Tail6 winner := by
  rw [root262Tail5, scan, if_neg (by decide), root262Candidates5_checked]
  change scan edge (search edge 3) 4 [1697] 1697 root262Tail6
    (if 3 ≤ root262Candidates5.length then search edge 3 [1697, 4931] root262Candidates5 6628 winner else winner) = _
  rw [if_pos (by decide), root262Child5]
def root262Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root262Candidates6_checked : candidates edge 4 1697 5087 root262Tail7 winner.bound = root262Candidates6 := by
  unfold candidates
  have row : edge 5087 = (SortedNatTree.contains FastRows.row5087) := by
    change Data.row5087.contains = (SortedNatTree.contains FastRows.row5087)
    exact FastRows.row5087_checked.symm
  rw [row]
  decide
theorem root262Step6 : scan edge (search edge 3) 4 [1697] 1697 root262Tail6 winner =
    scan edge (search edge 3) 4 [1697] 1697 root262Tail7 winner := by
  rw [root262Tail6, scan, if_neg (by decide), root262Candidates6_checked]
  change scan edge (search edge 3) 4 [1697] 1697 root262Tail7
    (if 3 ≤ root262Candidates6.length then search edge 3 [1697, 5087] root262Candidates6 6784 winner else winner) = _
  rw [if_neg (by decide)]
def root262Candidates7 : List Nat := [9467]
set_option maxRecDepth 8192 in
theorem root262Candidates7_checked : candidates edge 4 1697 5099 root262Tail8 winner.bound = root262Candidates7 := by
  unfold candidates
  have row : edge 5099 = (SortedNatTree.contains FastRows.row5099) := by
    change Data.row5099.contains = (SortedNatTree.contains FastRows.row5099)
    exact FastRows.row5099_checked.symm
  rw [row]
  decide
theorem root262Step7 : scan edge (search edge 3) 4 [1697] 1697 root262Tail7 winner =
    scan edge (search edge 3) 4 [1697] 1697 root262Tail8 winner := by
  rw [root262Tail7, scan, if_neg (by decide), root262Candidates7_checked]
  change scan edge (search edge 3) 4 [1697] 1697 root262Tail8
    (if 3 ≤ root262Candidates7.length then search edge 3 [1697, 5099] root262Candidates7 6796 winner else winner) = _
  rw [if_neg (by decide)]
def root262Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root262Candidates8_checked : candidates edge 4 1697 5477 root262Tail9 winner.bound = root262Candidates8 := by
  unfold candidates
  have row : edge 5477 = (SortedNatTree.contains FastRows.row5477) := by
    change Data.row5477.contains = (SortedNatTree.contains FastRows.row5477)
    exact FastRows.row5477_checked.symm
  rw [row]
  decide
theorem root262Step8 : scan edge (search edge 3) 4 [1697] 1697 root262Tail8 winner =
    scan edge (search edge 3) 4 [1697] 1697 root262Tail9 winner := by
  rw [root262Tail8, scan, if_neg (by decide), root262Candidates8_checked]
  change scan edge (search edge 3) 4 [1697] 1697 root262Tail9
    (if 3 ≤ root262Candidates8.length then search edge 3 [1697, 5477] root262Candidates8 7174 winner else winner) = _
  rw [if_neg (by decide)]
theorem root262Scan9 : scan edge (search edge 3) 4 [1697] 1697 root262Tail9 winner = winner := by rw [root262Tail9, scan, if_pos (by decide)]
theorem root262Scan8 : scan edge (search edge 3) 4 [1697] 1697 root262Tail8 winner = winner :=
  (root262Step8).trans root262Scan9
theorem root262Scan7 : scan edge (search edge 3) 4 [1697] 1697 root262Tail7 winner = winner :=
  (root262Step7).trans root262Scan8
theorem root262Scan6 : scan edge (search edge 3) 4 [1697] 1697 root262Tail6 winner = winner :=
  (root262Step6).trans root262Scan7
theorem root262Scan5 : scan edge (search edge 3) 4 [1697] 1697 root262Tail5 winner = winner :=
  (root262Step5).trans root262Scan6
theorem root262Scan4 : scan edge (search edge 3) 4 [1697] 1697 root262Tail4 winner = winner :=
  (root262Step4).trans root262Scan5
theorem root262Scan3 : scan edge (search edge 3) 4 [1697] 1697 root262Tail3 winner = winner :=
  (root262Step3).trans root262Scan4
theorem root262Scan2 : scan edge (search edge 3) 4 [1697] 1697 root262Tail2 winner = winner :=
  (root262Step2).trans root262Scan3
theorem root262Scan1 : scan edge (search edge 3) 4 [1697] 1697 root262Tail1 winner = winner :=
  (root262Step1).trans root262Scan2
theorem root262Scan0 : scan edge (search edge 3) 4 [1697] 1697 root262Tail0 winner = winner :=
  (root262Step0).trans root262Scan1
set_option maxRecDepth 4096 in
theorem root262Branch : search edge 4 [1697] root262Candidates 1697 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1697] 1697 root262Tail0 winner = winner
  exact root262Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
