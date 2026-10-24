import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root282Candidates : List Nat := Data.row1867
set_option maxRecDepth 8192 in
theorem root282Candidates_checked : candidates edge 5 0 1867 tail283 winner.bound = root282Candidates := by
  apply candidates_eq_row edge 5 0 1867 tail283 root282Candidates winner.bound tail283_ordered
  · decide
  · decide
  · funext q
    rfl
def root282Tail44 : List Nat := []
def root282Tail43 : List Nat := 24151 :: root282Tail44
def root282Tail42 : List Nat := 23371 :: root282Tail43
def root282Tail41 : List Nat := 22447 :: root282Tail42
def root282Tail40 : List Nat := 21019 :: root282Tail41
def root282Tail39 : List Nat := 20407 :: root282Tail40
def root282Tail38 : List Nat := 20071 :: root282Tail39
def root282Tail37 : List Nat := 19447 :: root282Tail38
def root282Tail36 : List Nat := 19333 :: root282Tail37
def root282Tail35 : List Nat := 19237 :: root282Tail36
def root282Tail34 : List Nat := 19069 :: root282Tail35
def root282Tail33 : List Nat := 18913 :: root282Tail34
def root282Tail32 : List Nat := 18313 :: root282Tail33
def root282Tail31 : List Nat := 18061 :: root282Tail32
def root282Tail30 : List Nat := 16693 :: root282Tail31
def root282Tail29 : List Nat := 16381 :: root282Tail30
def root282Tail28 : List Nat := 15241 :: root282Tail29
def root282Tail27 : List Nat := 15091 :: root282Tail28
def root282Tail26 : List Nat := 15031 :: root282Tail27
def root282Tail25 : List Nat := 14923 :: root282Tail26
def root282Tail24 : List Nat := 14029 :: root282Tail25
def root282Tail23 : List Nat := 13831 :: root282Tail24
def root282Tail22 : List Nat := 12157 :: root282Tail23
def root282Tail21 : List Nat := 11887 :: root282Tail22
def root282Tail20 : List Nat := 11353 :: root282Tail21
def root282Tail19 : List Nat := 11299 :: root282Tail20
def root282Tail18 : List Nat := 11131 :: root282Tail19
def root282Tail17 : List Nat := 10831 :: root282Tail18
def root282Tail16 : List Nat := 10789 :: root282Tail17
def root282Tail15 : List Nat := 10753 :: root282Tail16
def root282Tail14 : List Nat := 8731 :: root282Tail15
def root282Tail13 : List Nat := 8713 :: root282Tail14
def root282Tail12 : List Nat := 8293 :: root282Tail13
def root282Tail11 : List Nat := 7873 :: root282Tail12
def root282Tail10 : List Nat := 7639 :: root282Tail11
def root282Tail9 : List Nat := 7129 :: root282Tail10
def root282Tail8 : List Nat := 6733 :: root282Tail9
def root282Tail7 : List Nat := 6121 :: root282Tail8
def root282Tail6 : List Nat := 5749 :: root282Tail7
def root282Tail5 : List Nat := 5659 :: root282Tail6
def root282Tail4 : List Nat := 5281 :: root282Tail5
def root282Tail3 : List Nat := 4111 :: root282Tail4
def root282Tail2 : List Nat := 3673 :: root282Tail3
def root282Tail1 : List Nat := 2221 :: root282Tail2
def root282Tail0 : List Nat := 2053 :: root282Tail1
def root282Candidates0 : List Nat := [19069]
set_option maxRecDepth 8192 in
theorem root282Candidates0_checked : candidates edge 4 1867 2053 root282Tail1 winner.bound = root282Candidates0 := by
  unfold candidates
  have row : edge 2053 = (SortedNatTree.contains FastRows.row2053) := by
    change Data.row2053.contains = (SortedNatTree.contains FastRows.row2053)
    exact FastRows.row2053_checked.symm
  rw [row]
  decide
theorem root282Step0 : scan edge (search edge 3) 4 [1867] 1867 root282Tail0 winner =
    scan edge (search edge 3) 4 [1867] 1867 root282Tail1 winner := by
  rw [root282Tail0, scan, if_neg (by decide), root282Candidates0_checked]
  change scan edge (search edge 3) 4 [1867] 1867 root282Tail1
    (if 3 ≤ root282Candidates0.length then search edge 3 [1867, 2053] root282Candidates0 3920 winner else winner) = _
  rw [if_neg (by decide)]
def root282Candidates1 : List Nat := [4111, 7639, 7873]
set_option maxRecDepth 8192 in
theorem root282Candidates1_checked : candidates edge 4 1867 2221 root282Tail2 winner.bound = root282Candidates1 := by
  unfold candidates
  have row : edge 2221 = (SortedNatTree.contains FastRows.row2221) := by
    change Data.row2221.contains = (SortedNatTree.contains FastRows.row2221)
    exact FastRows.row2221_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root282Child1 : search edge 3 [1867, 2221] root282Candidates1 4088 winner = winner := by decide
theorem root282Step1 : scan edge (search edge 3) 4 [1867] 1867 root282Tail1 winner =
    scan edge (search edge 3) 4 [1867] 1867 root282Tail2 winner := by
  rw [root282Tail1, scan, if_neg (by decide), root282Candidates1_checked]
  change scan edge (search edge 3) 4 [1867] 1867 root282Tail2
    (if 3 ≤ root282Candidates1.length then search edge 3 [1867, 2221] root282Candidates1 4088 winner else winner) = _
  rw [if_pos (by decide), root282Child1]
def root282Candidates2 : List Nat := [8293]
set_option maxRecDepth 8192 in
theorem root282Candidates2_checked : candidates edge 4 1867 3673 root282Tail3 winner.bound = root282Candidates2 := by
  unfold candidates
  have row : edge 3673 = (SortedNatTree.contains FastRows.row3673) := by
    change Data.row3673.contains = (SortedNatTree.contains FastRows.row3673)
    exact FastRows.row3673_checked.symm
  rw [row]
  decide
theorem root282Step2 : scan edge (search edge 3) 4 [1867] 1867 root282Tail2 winner =
    scan edge (search edge 3) 4 [1867] 1867 root282Tail3 winner := by
  rw [root282Tail2, scan, if_neg (by decide), root282Candidates2_checked]
  change scan edge (search edge 3) 4 [1867] 1867 root282Tail3
    (if 3 ≤ root282Candidates2.length then search edge 3 [1867, 3673] root282Candidates2 5540 winner else winner) = _
  rw [if_neg (by decide)]
def root282Candidates3 : List Nat := [11353, 12157]
set_option maxRecDepth 8192 in
theorem root282Candidates3_checked : candidates edge 4 1867 4111 root282Tail4 winner.bound = root282Candidates3 := by
  unfold candidates
  have row : edge 4111 = (SortedNatTree.contains FastRows.row4111) := by
    change Data.row4111.contains = (SortedNatTree.contains FastRows.row4111)
    exact FastRows.row4111_checked.symm
  rw [row]
  decide
theorem root282Step3 : scan edge (search edge 3) 4 [1867] 1867 root282Tail3 winner =
    scan edge (search edge 3) 4 [1867] 1867 root282Tail4 winner := by
  rw [root282Tail3, scan, if_neg (by decide), root282Candidates3_checked]
  change scan edge (search edge 3) 4 [1867] 1867 root282Tail4
    (if 3 ≤ root282Candidates3.length then search edge 3 [1867, 4111] root282Candidates3 5978 winner else winner) = _
  rw [if_neg (by decide)]
def root282Candidates4 : List Nat := [6121, 7873]
set_option maxRecDepth 8192 in
theorem root282Candidates4_checked : candidates edge 4 1867 5281 root282Tail5 winner.bound = root282Candidates4 := by
  unfold candidates
  have row : edge 5281 = (SortedNatTree.contains FastRows.row5281) := by
    change Data.row5281.contains = (SortedNatTree.contains FastRows.row5281)
    exact FastRows.row5281_checked.symm
  rw [row]
  decide
theorem root282Step4 : scan edge (search edge 3) 4 [1867] 1867 root282Tail4 winner =
    scan edge (search edge 3) 4 [1867] 1867 root282Tail5 winner := by
  rw [root282Tail4, scan, if_neg (by decide), root282Candidates4_checked]
  change scan edge (search edge 3) 4 [1867] 1867 root282Tail5
    (if 3 ≤ root282Candidates4.length then search edge 3 [1867, 5281] root282Candidates4 7148 winner else winner) = _
  rw [if_neg (by decide)]
def root282Candidates5 : List Nat := [6121]
set_option maxRecDepth 8192 in
theorem root282Candidates5_checked : candidates edge 4 1867 5659 root282Tail6 winner.bound = root282Candidates5 := by
  unfold candidates
  have row : edge 5659 = (SortedNatTree.contains FastRows.row5659) := by
    change Data.row5659.contains = (SortedNatTree.contains FastRows.row5659)
    exact FastRows.row5659_checked.symm
  rw [row]
  decide
theorem root282Step5 : scan edge (search edge 3) 4 [1867] 1867 root282Tail5 winner =
    scan edge (search edge 3) 4 [1867] 1867 root282Tail6 winner := by
  rw [root282Tail5, scan, if_neg (by decide), root282Candidates5_checked]
  change scan edge (search edge 3) 4 [1867] 1867 root282Tail6
    (if 3 ≤ root282Candidates5.length then search edge 3 [1867, 5659] root282Candidates5 7526 winner else winner) = _
  rw [if_neg (by decide)]
def root282Candidates6 : List Nat := [11131, 18061]
set_option maxRecDepth 8192 in
theorem root282Candidates6_checked : candidates edge 4 1867 5749 root282Tail7 winner.bound = root282Candidates6 := by
  unfold candidates
  have row : edge 5749 = (SortedNatTree.contains FastRows.row5749) := by
    change Data.row5749.contains = (SortedNatTree.contains FastRows.row5749)
    exact FastRows.row5749_checked.symm
  rw [row]
  decide
theorem root282Step6 : scan edge (search edge 3) 4 [1867] 1867 root282Tail6 winner =
    scan edge (search edge 3) 4 [1867] 1867 root282Tail7 winner := by
  rw [root282Tail6, scan, if_neg (by decide), root282Candidates6_checked]
  change scan edge (search edge 3) 4 [1867] 1867 root282Tail7
    (if 3 ≤ root282Candidates6.length then search edge 3 [1867, 5749] root282Candidates6 7616 winner else winner) = _
  rw [if_neg (by decide)]
theorem root282Scan7 : scan edge (search edge 3) 4 [1867] 1867 root282Tail7 winner = winner := by rw [root282Tail7, scan, if_pos (by decide)]
theorem root282Scan6 : scan edge (search edge 3) 4 [1867] 1867 root282Tail6 winner = winner :=
  (root282Step6).trans root282Scan7
theorem root282Scan5 : scan edge (search edge 3) 4 [1867] 1867 root282Tail5 winner = winner :=
  (root282Step5).trans root282Scan6
theorem root282Scan4 : scan edge (search edge 3) 4 [1867] 1867 root282Tail4 winner = winner :=
  (root282Step4).trans root282Scan5
theorem root282Scan3 : scan edge (search edge 3) 4 [1867] 1867 root282Tail3 winner = winner :=
  (root282Step3).trans root282Scan4
theorem root282Scan2 : scan edge (search edge 3) 4 [1867] 1867 root282Tail2 winner = winner :=
  (root282Step2).trans root282Scan3
theorem root282Scan1 : scan edge (search edge 3) 4 [1867] 1867 root282Tail1 winner = winner :=
  (root282Step1).trans root282Scan2
theorem root282Scan0 : scan edge (search edge 3) 4 [1867] 1867 root282Tail0 winner = winner :=
  (root282Step0).trans root282Scan1
set_option maxRecDepth 4096 in
theorem root282Branch : search edge 4 [1867] root282Candidates 1867 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1867] 1867 root282Tail0 winner = winner
  exact root282Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
