import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root166Candidates : List Nat := Data.row1009
set_option maxRecDepth 8192 in
theorem root166Candidates_checked : candidates edge 5 0 1009 tail167 winner.bound = root166Candidates := by
  apply candidates_eq_row edge 5 0 1009 tail167 root166Candidates winner.bound tail167_ordered
  · decide
  · decide
  · funext q
    rfl
def root166Tail45 : List Nat := []
def root166Tail44 : List Nat := 24133 :: root166Tail45
def root166Tail43 : List Nat := 23071 :: root166Tail44
def root166Tail42 : List Nat := 22543 :: root166Tail43
def root166Tail41 : List Nat := 21517 :: root166Tail42
def root166Tail40 : List Nat := 20707 :: root166Tail41
def root166Tail39 : List Nat := 20161 :: root166Tail40
def root166Tail38 : List Nat := 19813 :: root166Tail39
def root166Tail37 : List Nat := 19141 :: root166Tail38
def root166Tail36 : List Nat := 18691 :: root166Tail37
def root166Tail35 : List Nat := 18523 :: root166Tail36
def root166Tail34 : List Nat := 17107 :: root166Tail35
def root166Tail33 : List Nat := 16843 :: root166Tail34
def root166Tail32 : List Nat := 16831 :: root166Tail33
def root166Tail31 : List Nat := 16759 :: root166Tail32
def root166Tail30 : List Nat := 16741 :: root166Tail31
def root166Tail29 : List Nat := 16381 :: root166Tail30
def root166Tail28 : List Nat := 16183 :: root166Tail29
def root166Tail27 : List Nat := 15889 :: root166Tail28
def root166Tail26 : List Nat := 15823 :: root166Tail27
def root166Tail25 : List Nat := 13327 :: root166Tail26
def root166Tail24 : List Nat := 12841 :: root166Tail25
def root166Tail23 : List Nat := 11467 :: root166Tail24
def root166Tail22 : List Nat := 11059 :: root166Tail23
def root166Tail21 : List Nat := 9433 :: root166Tail22
def root166Tail20 : List Nat := 8821 :: root166Tail21
def root166Tail19 : List Nat := 7753 :: root166Tail20
def root166Tail18 : List Nat := 7393 :: root166Tail19
def root166Tail17 : List Nat := 7309 :: root166Tail18
def root166Tail16 : List Nat := 7027 :: root166Tail17
def root166Tail15 : List Nat := 6673 :: root166Tail16
def root166Tail14 : List Nat := 6577 :: root166Tail15
def root166Tail13 : List Nat := 6481 :: root166Tail14
def root166Tail12 : List Nat := 6397 :: root166Tail13
def root166Tail11 : List Nat := 6367 :: root166Tail12
def root166Tail10 : List Nat := 6043 :: root166Tail11
def root166Tail9 : List Nat := 5653 :: root166Tail10
def root166Tail8 : List Nat := 4657 :: root166Tail9
def root166Tail7 : List Nat := 4099 :: root166Tail8
def root166Tail6 : List Nat := 3823 :: root166Tail7
def root166Tail5 : List Nat := 3613 :: root166Tail6
def root166Tail4 : List Nat := 2851 :: root166Tail5
def root166Tail3 : List Nat := 2473 :: root166Tail4
def root166Tail2 : List Nat := 2221 :: root166Tail3
def root166Tail1 : List Nat := 1933 :: root166Tail2
def root166Tail0 : List Nat := 1789 :: root166Tail1
def root166Candidates0 : List Nat := [7393]
set_option maxRecDepth 8192 in
theorem root166Candidates0_checked : candidates edge 4 1009 1789 root166Tail1 winner.bound = root166Candidates0 := by
  unfold candidates
  have row : edge 1789 = (SortedNatTree.contains FastRows.row1789) := by
    change Data.row1789.contains = (SortedNatTree.contains FastRows.row1789)
    exact FastRows.row1789_checked.symm
  rw [row]
  decide
theorem root166Step0 : scan edge (search edge 3) 4 [1009] 1009 root166Tail0 winner =
    scan edge (search edge 3) 4 [1009] 1009 root166Tail1 winner := by
  rw [root166Tail0, scan, if_neg (by decide), root166Candidates0_checked]
  change scan edge (search edge 3) 4 [1009] 1009 root166Tail1
    (if 3 ≤ root166Candidates0.length then search edge 3 [1009, 1789] root166Candidates0 2798 winner else winner) = _
  rw [if_neg (by decide)]
def root166Candidates1 : List Nat := [17107, 18523, 23071]
set_option maxRecDepth 8192 in
theorem root166Candidates1_checked : candidates edge 4 1009 1933 root166Tail2 winner.bound = root166Candidates1 := by
  unfold candidates
  have row : edge 1933 = (SortedNatTree.contains FastRows.row1933) := by
    change Data.row1933.contains = (SortedNatTree.contains FastRows.row1933)
    exact FastRows.row1933_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root166Child1 : search edge 3 [1009, 1933] root166Candidates1 2942 winner = winner := by decide
theorem root166Step1 : scan edge (search edge 3) 4 [1009] 1009 root166Tail1 winner =
    scan edge (search edge 3) 4 [1009] 1009 root166Tail2 winner := by
  rw [root166Tail1, scan, if_neg (by decide), root166Candidates1_checked]
  change scan edge (search edge 3) 4 [1009] 1009 root166Tail2
    (if 3 ≤ root166Candidates1.length then search edge 3 [1009, 1933] root166Candidates1 2942 winner else winner) = _
  rw [if_pos (by decide), root166Child1]
def root166Candidates2 : List Nat := [6043, 6673, 20161]
set_option maxRecDepth 8192 in
theorem root166Candidates2_checked : candidates edge 4 1009 2221 root166Tail3 winner.bound = root166Candidates2 := by
  unfold candidates
  have row : edge 2221 = (SortedNatTree.contains FastRows.row2221) := by
    change Data.row2221.contains = (SortedNatTree.contains FastRows.row2221)
    exact FastRows.row2221_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root166Child2 : search edge 3 [1009, 2221] root166Candidates2 3230 winner = winner := by decide
theorem root166Step2 : scan edge (search edge 3) 4 [1009] 1009 root166Tail2 winner =
    scan edge (search edge 3) 4 [1009] 1009 root166Tail3 winner := by
  rw [root166Tail2, scan, if_neg (by decide), root166Candidates2_checked]
  change scan edge (search edge 3) 4 [1009] 1009 root166Tail3
    (if 3 ≤ root166Candidates2.length then search edge 3 [1009, 2221] root166Candidates2 3230 winner else winner) = _
  rw [if_pos (by decide), root166Child2]
def root166Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root166Candidates3_checked : candidates edge 4 1009 2473 root166Tail4 winner.bound = root166Candidates3 := by
  unfold candidates
  have row : edge 2473 = (SortedNatTree.contains FastRows.row2473) := by
    change Data.row2473.contains = (SortedNatTree.contains FastRows.row2473)
    exact FastRows.row2473_checked.symm
  rw [row]
  decide
theorem root166Step3 : scan edge (search edge 3) 4 [1009] 1009 root166Tail3 winner =
    scan edge (search edge 3) 4 [1009] 1009 root166Tail4 winner := by
  rw [root166Tail3, scan, if_neg (by decide), root166Candidates3_checked]
  change scan edge (search edge 3) 4 [1009] 1009 root166Tail4
    (if 3 ≤ root166Candidates3.length then search edge 3 [1009, 2473] root166Candidates3 3482 winner else winner) = _
  rw [if_neg (by decide)]
def root166Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root166Candidates4_checked : candidates edge 4 1009 2851 root166Tail5 winner.bound = root166Candidates4 := by
  unfold candidates
  have row : edge 2851 = (SortedNatTree.contains FastRows.row2851) := by
    change Data.row2851.contains = (SortedNatTree.contains FastRows.row2851)
    exact FastRows.row2851_checked.symm
  rw [row]
  decide
theorem root166Step4 : scan edge (search edge 3) 4 [1009] 1009 root166Tail4 winner =
    scan edge (search edge 3) 4 [1009] 1009 root166Tail5 winner := by
  rw [root166Tail4, scan, if_neg (by decide), root166Candidates4_checked]
  change scan edge (search edge 3) 4 [1009] 1009 root166Tail5
    (if 3 ≤ root166Candidates4.length then search edge 3 [1009, 2851] root166Candidates4 3860 winner else winner) = _
  rw [if_neg (by decide)]
def root166Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root166Candidates5_checked : candidates edge 4 1009 3613 root166Tail6 winner.bound = root166Candidates5 := by
  unfold candidates
  have row : edge 3613 = (SortedNatTree.contains FastRows.row3613) := by
    change Data.row3613.contains = (SortedNatTree.contains FastRows.row3613)
    exact FastRows.row3613_checked.symm
  rw [row]
  decide
theorem root166Step5 : scan edge (search edge 3) 4 [1009] 1009 root166Tail5 winner =
    scan edge (search edge 3) 4 [1009] 1009 root166Tail6 winner := by
  rw [root166Tail5, scan, if_neg (by decide), root166Candidates5_checked]
  change scan edge (search edge 3) 4 [1009] 1009 root166Tail6
    (if 3 ≤ root166Candidates5.length then search edge 3 [1009, 3613] root166Candidates5 4622 winner else winner) = _
  rw [if_neg (by decide)]
def root166Candidates6 : List Nat := [7393]
set_option maxRecDepth 8192 in
theorem root166Candidates6_checked : candidates edge 4 1009 3823 root166Tail7 winner.bound = root166Candidates6 := by
  unfold candidates
  have row : edge 3823 = (SortedNatTree.contains FastRows.row3823) := by
    change Data.row3823.contains = (SortedNatTree.contains FastRows.row3823)
    exact FastRows.row3823_checked.symm
  rw [row]
  decide
theorem root166Step6 : scan edge (search edge 3) 4 [1009] 1009 root166Tail6 winner =
    scan edge (search edge 3) 4 [1009] 1009 root166Tail7 winner := by
  rw [root166Tail6, scan, if_neg (by decide), root166Candidates6_checked]
  change scan edge (search edge 3) 4 [1009] 1009 root166Tail7
    (if 3 ≤ root166Candidates6.length then search edge 3 [1009, 3823] root166Candidates6 4832 winner else winner) = _
  rw [if_neg (by decide)]
def root166Candidates7 : List Nat := [8821, 19141]
set_option maxRecDepth 8192 in
theorem root166Candidates7_checked : candidates edge 4 1009 4099 root166Tail8 winner.bound = root166Candidates7 := by
  unfold candidates
  have row : edge 4099 = (SortedNatTree.contains FastRows.row4099) := by
    change Data.row4099.contains = (SortedNatTree.contains FastRows.row4099)
    exact FastRows.row4099_checked.symm
  rw [row]
  decide
theorem root166Step7 : scan edge (search edge 3) 4 [1009] 1009 root166Tail7 winner =
    scan edge (search edge 3) 4 [1009] 1009 root166Tail8 winner := by
  rw [root166Tail7, scan, if_neg (by decide), root166Candidates7_checked]
  change scan edge (search edge 3) 4 [1009] 1009 root166Tail8
    (if 3 ≤ root166Candidates7.length then search edge 3 [1009, 4099] root166Candidates7 5108 winner else winner) = _
  rw [if_neg (by decide)]
def root166Candidates8 : List Nat := [7753, 9433]
set_option maxRecDepth 8192 in
theorem root166Candidates8_checked : candidates edge 4 1009 4657 root166Tail9 winner.bound = root166Candidates8 := by
  unfold candidates
  have row : edge 4657 = (SortedNatTree.contains FastRows.row4657) := by
    change Data.row4657.contains = (SortedNatTree.contains FastRows.row4657)
    exact FastRows.row4657_checked.symm
  rw [row]
  decide
theorem root166Step8 : scan edge (search edge 3) 4 [1009] 1009 root166Tail8 winner =
    scan edge (search edge 3) 4 [1009] 1009 root166Tail9 winner := by
  rw [root166Tail8, scan, if_neg (by decide), root166Candidates8_checked]
  change scan edge (search edge 3) 4 [1009] 1009 root166Tail9
    (if 3 ≤ root166Candidates8.length then search edge 3 [1009, 4657] root166Candidates8 5666 winner else winner) = _
  rw [if_neg (by decide)]
def root166Candidates9 : List Nat := [6367]
set_option maxRecDepth 8192 in
theorem root166Candidates9_checked : candidates edge 4 1009 5653 root166Tail10 winner.bound = root166Candidates9 := by
  unfold candidates
  have row : edge 5653 = (SortedNatTree.contains FastRows.row5653) := by
    change Data.row5653.contains = (SortedNatTree.contains FastRows.row5653)
    exact FastRows.row5653_checked.symm
  rw [row]
  decide
theorem root166Step9 : scan edge (search edge 3) 4 [1009] 1009 root166Tail9 winner =
    scan edge (search edge 3) 4 [1009] 1009 root166Tail10 winner := by
  rw [root166Tail9, scan, if_neg (by decide), root166Candidates9_checked]
  change scan edge (search edge 3) 4 [1009] 1009 root166Tail10
    (if 3 ≤ root166Candidates9.length then search edge 3 [1009, 5653] root166Candidates9 6662 winner else winner) = _
  rw [if_neg (by decide)]
def root166Candidates10 : List Nat := [16183]
set_option maxRecDepth 8192 in
theorem root166Candidates10_checked : candidates edge 4 1009 6043 root166Tail11 winner.bound = root166Candidates10 := by
  unfold candidates
  have row : edge 6043 = (SortedNatTree.contains FastRows.row6043) := by
    change Data.row6043.contains = (SortedNatTree.contains FastRows.row6043)
    exact FastRows.row6043_checked.symm
  rw [row]
  decide
theorem root166Step10 : scan edge (search edge 3) 4 [1009] 1009 root166Tail10 winner =
    scan edge (search edge 3) 4 [1009] 1009 root166Tail11 winner := by
  rw [root166Tail10, scan, if_neg (by decide), root166Candidates10_checked]
  change scan edge (search edge 3) 4 [1009] 1009 root166Tail11
    (if 3 ≤ root166Candidates10.length then search edge 3 [1009, 6043] root166Candidates10 7052 winner else winner) = _
  rw [if_neg (by decide)]
theorem root166Scan11 : scan edge (search edge 3) 4 [1009] 1009 root166Tail11 winner = winner := by rw [root166Tail11, scan, if_pos (by decide)]
theorem root166Scan10 : scan edge (search edge 3) 4 [1009] 1009 root166Tail10 winner = winner :=
  (root166Step10).trans root166Scan11
theorem root166Scan9 : scan edge (search edge 3) 4 [1009] 1009 root166Tail9 winner = winner :=
  (root166Step9).trans root166Scan10
theorem root166Scan8 : scan edge (search edge 3) 4 [1009] 1009 root166Tail8 winner = winner :=
  (root166Step8).trans root166Scan9
theorem root166Scan7 : scan edge (search edge 3) 4 [1009] 1009 root166Tail7 winner = winner :=
  (root166Step7).trans root166Scan8
theorem root166Scan6 : scan edge (search edge 3) 4 [1009] 1009 root166Tail6 winner = winner :=
  (root166Step6).trans root166Scan7
theorem root166Scan5 : scan edge (search edge 3) 4 [1009] 1009 root166Tail5 winner = winner :=
  (root166Step5).trans root166Scan6
theorem root166Scan4 : scan edge (search edge 3) 4 [1009] 1009 root166Tail4 winner = winner :=
  (root166Step4).trans root166Scan5
theorem root166Scan3 : scan edge (search edge 3) 4 [1009] 1009 root166Tail3 winner = winner :=
  (root166Step3).trans root166Scan4
theorem root166Scan2 : scan edge (search edge 3) 4 [1009] 1009 root166Tail2 winner = winner :=
  (root166Step2).trans root166Scan3
theorem root166Scan1 : scan edge (search edge 3) 4 [1009] 1009 root166Tail1 winner = winner :=
  (root166Step1).trans root166Scan2
theorem root166Scan0 : scan edge (search edge 3) 4 [1009] 1009 root166Tail0 winner = winner :=
  (root166Step0).trans root166Scan1
set_option maxRecDepth 4096 in
theorem root166Branch : search edge 4 [1009] root166Candidates 1009 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1009] 1009 root166Tail0 winner = winner
  exact root166Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
