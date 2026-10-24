import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root284Candidates : List Nat := Data.row1873
set_option maxRecDepth 8192 in
theorem root284Candidates_checked : candidates edge 5 0 1873 tail285 winner.bound = root284Candidates := by
  apply candidates_eq_row edge 5 0 1873 tail285 root284Candidates winner.bound tail285_ordered
  · decide
  · decide
  · funext q
    rfl
def root284Tail45 : List Nat := []
def root284Tail44 : List Nat := 24091 :: root284Tail45
def root284Tail43 : List Nat := 23167 :: root284Tail44
def root284Tail42 : List Nat := 23011 :: root284Tail43
def root284Tail41 : List Nat := 21943 :: root284Tail42
def root284Tail40 : List Nat := 21613 :: root284Tail41
def root284Tail39 : List Nat := 20143 :: root284Tail40
def root284Tail38 : List Nat := 15607 :: root284Tail39
def root284Tail37 : List Nat := 15493 :: root284Tail38
def root284Tail36 : List Nat := 15391 :: root284Tail37
def root284Tail35 : List Nat := 14947 :: root284Tail36
def root284Tail34 : List Nat := 14851 :: root284Tail35
def root284Tail33 : List Nat := 14821 :: root284Tail34
def root284Tail32 : List Nat := 14437 :: root284Tail33
def root284Tail31 : List Nat := 14107 :: root284Tail32
def root284Tail30 : List Nat := 13627 :: root284Tail31
def root284Tail29 : List Nat := 13003 :: root284Tail30
def root284Tail28 : List Nat := 12967 :: root284Tail29
def root284Tail27 : List Nat := 12049 :: root284Tail28
def root284Tail26 : List Nat := 11251 :: root284Tail27
def root284Tail25 : List Nat := 10867 :: root284Tail26
def root284Tail24 : List Nat := 10657 :: root284Tail25
def root284Tail23 : List Nat := 9907 :: root284Tail24
def root284Tail22 : List Nat := 9769 :: root284Tail23
def root284Tail21 : List Nat := 9697 :: root284Tail22
def root284Tail20 : List Nat := 9433 :: root284Tail21
def root284Tail19 : List Nat := 9241 :: root284Tail20
def root284Tail18 : List Nat := 9067 :: root284Tail19
def root284Tail17 : List Nat := 8179 :: root284Tail18
def root284Tail16 : List Nat := 8059 :: root284Tail17
def root284Tail15 : List Nat := 8011 :: root284Tail16
def root284Tail14 : List Nat := 7963 :: root284Tail15
def root284Tail13 : List Nat := 7507 :: root284Tail14
def root284Tail12 : List Nat := 6997 :: root284Tail13
def root284Tail11 : List Nat := 6961 :: root284Tail12
def root284Tail10 : List Nat := 6841 :: root284Tail11
def root284Tail9 : List Nat := 6709 :: root284Tail10
def root284Tail8 : List Nat := 6427 :: root284Tail9
def root284Tail7 : List Nat := 6301 :: root284Tail8
def root284Tail6 : List Nat := 5569 :: root284Tail7
def root284Tail5 : List Nat := 5527 :: root284Tail6
def root284Tail4 : List Nat := 3511 :: root284Tail5
def root284Tail3 : List Nat := 2719 :: root284Tail4
def root284Tail2 : List Nat := 2437 :: root284Tail3
def root284Tail1 : List Nat := 2269 :: root284Tail2
def root284Tail0 : List Nat := 2017 :: root284Tail1
def root284Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root284Candidates0_checked : candidates edge 4 1873 2017 root284Tail1 winner.bound = root284Candidates0 := by
  unfold candidates
  have row : edge 2017 = (SortedNatTree.contains FastRows.row2017) := by
    change Data.row2017.contains = (SortedNatTree.contains FastRows.row2017)
    exact FastRows.row2017_checked.symm
  rw [row]
  decide
theorem root284Step0 : scan edge (search edge 3) 4 [1873] 1873 root284Tail0 winner =
    scan edge (search edge 3) 4 [1873] 1873 root284Tail1 winner := by
  rw [root284Tail0, scan, if_neg (by decide), root284Candidates0_checked]
  change scan edge (search edge 3) 4 [1873] 1873 root284Tail1
    (if 3 ≤ root284Candidates0.length then search edge 3 [1873, 2017] root284Candidates0 3890 winner else winner) = _
  rw [if_neg (by decide)]
def root284Candidates1 : List Nat := [9697]
set_option maxRecDepth 8192 in
theorem root284Candidates1_checked : candidates edge 4 1873 2269 root284Tail2 winner.bound = root284Candidates1 := by
  unfold candidates
  have row : edge 2269 = (SortedNatTree.contains FastRows.row2269) := by
    change Data.row2269.contains = (SortedNatTree.contains FastRows.row2269)
    exact FastRows.row2269_checked.symm
  rw [row]
  decide
theorem root284Step1 : scan edge (search edge 3) 4 [1873] 1873 root284Tail1 winner =
    scan edge (search edge 3) 4 [1873] 1873 root284Tail2 winner := by
  rw [root284Tail1, scan, if_neg (by decide), root284Candidates1_checked]
  change scan edge (search edge 3) 4 [1873] 1873 root284Tail2
    (if 3 ≤ root284Candidates1.length then search edge 3 [1873, 2269] root284Candidates1 4142 winner else winner) = _
  rw [if_neg (by decide)]
def root284Candidates2 : List Nat := [6301]
set_option maxRecDepth 8192 in
theorem root284Candidates2_checked : candidates edge 4 1873 2437 root284Tail3 winner.bound = root284Candidates2 := by
  unfold candidates
  have row : edge 2437 = (SortedNatTree.contains FastRows.row2437) := by
    change Data.row2437.contains = (SortedNatTree.contains FastRows.row2437)
    exact FastRows.row2437_checked.symm
  rw [row]
  decide
theorem root284Step2 : scan edge (search edge 3) 4 [1873] 1873 root284Tail2 winner =
    scan edge (search edge 3) 4 [1873] 1873 root284Tail3 winner := by
  rw [root284Tail2, scan, if_neg (by decide), root284Candidates2_checked]
  change scan edge (search edge 3) 4 [1873] 1873 root284Tail3
    (if 3 ≤ root284Candidates2.length then search edge 3 [1873, 2437] root284Candidates2 4310 winner else winner) = _
  rw [if_neg (by decide)]
def root284Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root284Candidates3_checked : candidates edge 4 1873 2719 root284Tail4 winner.bound = root284Candidates3 := by
  unfold candidates
  have row : edge 2719 = (SortedNatTree.contains FastRows.row2719) := by
    change Data.row2719.contains = (SortedNatTree.contains FastRows.row2719)
    exact FastRows.row2719_checked.symm
  rw [row]
  decide
theorem root284Step3 : scan edge (search edge 3) 4 [1873] 1873 root284Tail3 winner =
    scan edge (search edge 3) 4 [1873] 1873 root284Tail4 winner := by
  rw [root284Tail3, scan, if_neg (by decide), root284Candidates3_checked]
  change scan edge (search edge 3) 4 [1873] 1873 root284Tail4
    (if 3 ≤ root284Candidates3.length then search edge 3 [1873, 2719] root284Candidates3 4592 winner else winner) = _
  rw [if_neg (by decide)]
def root284Candidates4 : List Nat := [7963, 8011]
set_option maxRecDepth 8192 in
theorem root284Candidates4_checked : candidates edge 4 1873 3511 root284Tail5 winner.bound = root284Candidates4 := by
  unfold candidates
  have row : edge 3511 = (SortedNatTree.contains FastRows.row3511) := by
    change Data.row3511.contains = (SortedNatTree.contains FastRows.row3511)
    exact FastRows.row3511_checked.symm
  rw [row]
  decide
theorem root284Step4 : scan edge (search edge 3) 4 [1873] 1873 root284Tail4 winner =
    scan edge (search edge 3) 4 [1873] 1873 root284Tail5 winner := by
  rw [root284Tail4, scan, if_neg (by decide), root284Candidates4_checked]
  change scan edge (search edge 3) 4 [1873] 1873 root284Tail5
    (if 3 ≤ root284Candidates4.length then search edge 3 [1873, 3511] root284Candidates4 5384 winner else winner) = _
  rw [if_neg (by decide)]
def root284Candidates5 : List Nat := [10867]
set_option maxRecDepth 8192 in
theorem root284Candidates5_checked : candidates edge 4 1873 5527 root284Tail6 winner.bound = root284Candidates5 := by
  unfold candidates
  have row : edge 5527 = (SortedNatTree.contains FastRows.row5527) := by
    change Data.row5527.contains = (SortedNatTree.contains FastRows.row5527)
    exact FastRows.row5527_checked.symm
  rw [row]
  decide
theorem root284Step5 : scan edge (search edge 3) 4 [1873] 1873 root284Tail5 winner =
    scan edge (search edge 3) 4 [1873] 1873 root284Tail6 winner := by
  rw [root284Tail5, scan, if_neg (by decide), root284Candidates5_checked]
  change scan edge (search edge 3) 4 [1873] 1873 root284Tail6
    (if 3 ≤ root284Candidates5.length then search edge 3 [1873, 5527] root284Candidates5 7400 winner else winner) = _
  rw [if_neg (by decide)]
def root284Candidates6 : List Nat := [9433]
set_option maxRecDepth 8192 in
theorem root284Candidates6_checked : candidates edge 4 1873 5569 root284Tail7 winner.bound = root284Candidates6 := by
  unfold candidates
  have row : edge 5569 = (SortedNatTree.contains FastRows.row5569) := by
    change Data.row5569.contains = (SortedNatTree.contains FastRows.row5569)
    exact FastRows.row5569_checked.symm
  rw [row]
  decide
theorem root284Step6 : scan edge (search edge 3) 4 [1873] 1873 root284Tail6 winner =
    scan edge (search edge 3) 4 [1873] 1873 root284Tail7 winner := by
  rw [root284Tail6, scan, if_neg (by decide), root284Candidates6_checked]
  change scan edge (search edge 3) 4 [1873] 1873 root284Tail7
    (if 3 ≤ root284Candidates6.length then search edge 3 [1873, 5569] root284Candidates6 7442 winner else winner) = _
  rw [if_neg (by decide)]
theorem root284Scan7 : scan edge (search edge 3) 4 [1873] 1873 root284Tail7 winner = winner := by rw [root284Tail7, scan, if_pos (by decide)]
theorem root284Scan6 : scan edge (search edge 3) 4 [1873] 1873 root284Tail6 winner = winner :=
  (root284Step6).trans root284Scan7
theorem root284Scan5 : scan edge (search edge 3) 4 [1873] 1873 root284Tail5 winner = winner :=
  (root284Step5).trans root284Scan6
theorem root284Scan4 : scan edge (search edge 3) 4 [1873] 1873 root284Tail4 winner = winner :=
  (root284Step4).trans root284Scan5
theorem root284Scan3 : scan edge (search edge 3) 4 [1873] 1873 root284Tail3 winner = winner :=
  (root284Step3).trans root284Scan4
theorem root284Scan2 : scan edge (search edge 3) 4 [1873] 1873 root284Tail2 winner = winner :=
  (root284Step2).trans root284Scan3
theorem root284Scan1 : scan edge (search edge 3) 4 [1873] 1873 root284Tail1 winner = winner :=
  (root284Step1).trans root284Scan2
theorem root284Scan0 : scan edge (search edge 3) 4 [1873] 1873 root284Tail0 winner = winner :=
  (root284Step0).trans root284Scan1
set_option maxRecDepth 4096 in
theorem root284Branch : search edge 4 [1873] root284Candidates 1873 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1873] 1873 root284Tail0 winner = winner
  exact root284Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
