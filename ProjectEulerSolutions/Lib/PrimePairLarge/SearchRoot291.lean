import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root291Candidates : List Nat := Data.row1931
set_option maxRecDepth 8192 in
theorem root291Candidates_checked : candidates edge 5 0 1931 tail292 winner.bound = root291Candidates := by
  apply candidates_eq_row edge 5 0 1931 tail292 root291Candidates winner.bound tail292_ordered
  · decide
  · decide
  · funext q
    rfl
def root291Tail44 : List Nat := []
def root291Tail43 : List Nat := 23603 :: root291Tail44
def root291Tail42 : List Nat := 23549 :: root291Tail43
def root291Tail41 : List Nat := 23531 :: root291Tail42
def root291Tail40 : List Nat := 22679 :: root291Tail41
def root291Tail39 : List Nat := 21893 :: root291Tail40
def root291Tail38 : List Nat := 19841 :: root291Tail39
def root291Tail37 : List Nat := 19583 :: root291Tail38
def root291Tail36 : List Nat := 19571 :: root291Tail37
def root291Tail35 : List Nat := 19037 :: root291Tail36
def root291Tail34 : List Nat := 18773 :: root291Tail35
def root291Tail33 : List Nat := 17921 :: root291Tail34
def root291Tail32 : List Nat := 17471 :: root291Tail33
def root291Tail31 : List Nat := 16673 :: root291Tail32
def root291Tail30 : List Nat := 15551 :: root291Tail31
def root291Tail29 : List Nat := 15401 :: root291Tail30
def root291Tail28 : List Nat := 15287 :: root291Tail29
def root291Tail27 : List Nat := 14759 :: root291Tail28
def root291Tail26 : List Nat := 14447 :: root291Tail27
def root291Tail25 : List Nat := 14057 :: root291Tail26
def root291Tail24 : List Nat := 13331 :: root291Tail25
def root291Tail23 : List Nat := 13163 :: root291Tail24
def root291Tail22 : List Nat := 13103 :: root291Tail23
def root291Tail21 : List Nat := 11813 :: root291Tail22
def root291Tail20 : List Nat := 11393 :: root291Tail21
def root291Tail19 : List Nat := 11213 :: root291Tail20
def root291Tail18 : List Nat := 11057 :: root291Tail19
def root291Tail17 : List Nat := 10457 :: root291Tail18
def root291Tail16 : List Nat := 10343 :: root291Tail17
def root291Tail15 : List Nat := 10133 :: root291Tail16
def root291Tail14 : List Nat := 10007 :: root291Tail15
def root291Tail13 : List Nat := 9929 :: root291Tail14
def root291Tail12 : List Nat := 9533 :: root291Tail13
def root291Tail11 : List Nat := 9437 :: root291Tail12
def root291Tail10 : List Nat := 8753 :: root291Tail11
def root291Tail9 : List Nat := 7043 :: root291Tail10
def root291Tail8 : List Nat := 6719 :: root291Tail9
def root291Tail7 : List Nat := 5477 :: root291Tail8
def root291Tail6 : List Nat := 5297 :: root291Tail7
def root291Tail5 : List Nat := 5279 :: root291Tail6
def root291Tail4 : List Nat := 5171 :: root291Tail5
def root291Tail3 : List Nat := 4943 :: root291Tail4
def root291Tail2 : List Nat := 4289 :: root291Tail3
def root291Tail1 : List Nat := 2957 :: root291Tail2
def root291Tail0 : List Nat := 2399 :: root291Tail1
def root291Candidates0 : List Nat := [14447]
set_option maxRecDepth 8192 in
theorem root291Candidates0_checked : candidates edge 4 1931 2399 root291Tail1 winner.bound = root291Candidates0 := by
  unfold candidates
  have row : edge 2399 = (SortedNatTree.contains FastRows.row2399) := by
    change Data.row2399.contains = (SortedNatTree.contains FastRows.row2399)
    exact FastRows.row2399_checked.symm
  rw [row]
  decide
theorem root291Step0 : scan edge (search edge 3) 4 [1931] 1931 root291Tail0 winner =
    scan edge (search edge 3) 4 [1931] 1931 root291Tail1 winner := by
  rw [root291Tail0, scan, if_neg (by decide), root291Candidates0_checked]
  change scan edge (search edge 3) 4 [1931] 1931 root291Tail1
    (if 3 ≤ root291Candidates0.length then search edge 3 [1931, 2399] root291Candidates0 4330 winner else winner) = _
  rw [if_neg (by decide)]
def root291Candidates1 : List Nat := [14759, 19841]
set_option maxRecDepth 8192 in
theorem root291Candidates1_checked : candidates edge 4 1931 2957 root291Tail2 winner.bound = root291Candidates1 := by
  unfold candidates
  have row : edge 2957 = (SortedNatTree.contains FastRows.row2957) := by
    change Data.row2957.contains = (SortedNatTree.contains FastRows.row2957)
    exact FastRows.row2957_checked.symm
  rw [row]
  decide
theorem root291Step1 : scan edge (search edge 3) 4 [1931] 1931 root291Tail1 winner =
    scan edge (search edge 3) 4 [1931] 1931 root291Tail2 winner := by
  rw [root291Tail1, scan, if_neg (by decide), root291Candidates1_checked]
  change scan edge (search edge 3) 4 [1931] 1931 root291Tail2
    (if 3 ≤ root291Candidates1.length then search edge 3 [1931, 2957] root291Candidates1 4888 winner else winner) = _
  rw [if_neg (by decide)]
def root291Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root291Candidates2_checked : candidates edge 4 1931 4289 root291Tail3 winner.bound = root291Candidates2 := by
  unfold candidates
  have row : edge 4289 = (SortedNatTree.contains FastRows.row4289) := by
    change Data.row4289.contains = (SortedNatTree.contains FastRows.row4289)
    exact FastRows.row4289_checked.symm
  rw [row]
  decide
theorem root291Step2 : scan edge (search edge 3) 4 [1931] 1931 root291Tail2 winner =
    scan edge (search edge 3) 4 [1931] 1931 root291Tail3 winner := by
  rw [root291Tail2, scan, if_neg (by decide), root291Candidates2_checked]
  change scan edge (search edge 3) 4 [1931] 1931 root291Tail3
    (if 3 ≤ root291Candidates2.length then search edge 3 [1931, 4289] root291Candidates2 6220 winner else winner) = _
  rw [if_neg (by decide)]
def root291Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root291Candidates3_checked : candidates edge 4 1931 4943 root291Tail4 winner.bound = root291Candidates3 := by
  unfold candidates
  have row : edge 4943 = (SortedNatTree.contains FastRows.row4943) := by
    change Data.row4943.contains = (SortedNatTree.contains FastRows.row4943)
    exact FastRows.row4943_checked.symm
  rw [row]
  decide
theorem root291Step3 : scan edge (search edge 3) 4 [1931] 1931 root291Tail3 winner =
    scan edge (search edge 3) 4 [1931] 1931 root291Tail4 winner := by
  rw [root291Tail3, scan, if_neg (by decide), root291Candidates3_checked]
  change scan edge (search edge 3) 4 [1931] 1931 root291Tail4
    (if 3 ≤ root291Candidates3.length then search edge 3 [1931, 4943] root291Candidates3 6874 winner else winner) = _
  rw [if_neg (by decide)]
def root291Candidates4 : List Nat := [5297, 10457, 14447]
set_option maxRecDepth 8192 in
theorem root291Candidates4_checked : candidates edge 4 1931 5171 root291Tail5 winner.bound = root291Candidates4 := by
  unfold candidates
  have row : edge 5171 = (SortedNatTree.contains FastRows.row5171) := by
    change Data.row5171.contains = (SortedNatTree.contains FastRows.row5171)
    exact FastRows.row5171_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root291Child4 : search edge 3 [1931, 5171] root291Candidates4 7102 winner = winner := by decide
theorem root291Step4 : scan edge (search edge 3) 4 [1931] 1931 root291Tail4 winner =
    scan edge (search edge 3) 4 [1931] 1931 root291Tail5 winner := by
  rw [root291Tail4, scan, if_neg (by decide), root291Candidates4_checked]
  change scan edge (search edge 3) 4 [1931] 1931 root291Tail5
    (if 3 ≤ root291Candidates4.length then search edge 3 [1931, 5171] root291Candidates4 7102 winner else winner) = _
  rw [if_pos (by decide), root291Child4]
def root291Candidates5 : List Nat := [6719, 15287]
set_option maxRecDepth 8192 in
theorem root291Candidates5_checked : candidates edge 4 1931 5279 root291Tail6 winner.bound = root291Candidates5 := by
  unfold candidates
  have row : edge 5279 = (SortedNatTree.contains FastRows.row5279) := by
    change Data.row5279.contains = (SortedNatTree.contains FastRows.row5279)
    exact FastRows.row5279_checked.symm
  rw [row]
  decide
theorem root291Step5 : scan edge (search edge 3) 4 [1931] 1931 root291Tail5 winner =
    scan edge (search edge 3) 4 [1931] 1931 root291Tail6 winner := by
  rw [root291Tail5, scan, if_neg (by decide), root291Candidates5_checked]
  change scan edge (search edge 3) 4 [1931] 1931 root291Tail6
    (if 3 ≤ root291Candidates5.length then search edge 3 [1931, 5279] root291Candidates5 7210 winner else winner) = _
  rw [if_neg (by decide)]
def root291Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root291Candidates6_checked : candidates edge 4 1931 5297 root291Tail7 winner.bound = root291Candidates6 := by
  unfold candidates
  have row : edge 5297 = (SortedNatTree.contains FastRows.row5297) := by
    change Data.row5297.contains = (SortedNatTree.contains FastRows.row5297)
    exact FastRows.row5297_checked.symm
  rw [row]
  decide
theorem root291Step6 : scan edge (search edge 3) 4 [1931] 1931 root291Tail6 winner =
    scan edge (search edge 3) 4 [1931] 1931 root291Tail7 winner := by
  rw [root291Tail6, scan, if_neg (by decide), root291Candidates6_checked]
  change scan edge (search edge 3) 4 [1931] 1931 root291Tail7
    (if 3 ≤ root291Candidates6.length then search edge 3 [1931, 5297] root291Candidates6 7228 winner else winner) = _
  rw [if_neg (by decide)]
def root291Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root291Candidates7_checked : candidates edge 4 1931 5477 root291Tail8 winner.bound = root291Candidates7 := by
  unfold candidates
  have row : edge 5477 = (SortedNatTree.contains FastRows.row5477) := by
    change Data.row5477.contains = (SortedNatTree.contains FastRows.row5477)
    exact FastRows.row5477_checked.symm
  rw [row]
  decide
theorem root291Step7 : scan edge (search edge 3) 4 [1931] 1931 root291Tail7 winner =
    scan edge (search edge 3) 4 [1931] 1931 root291Tail8 winner := by
  rw [root291Tail7, scan, if_neg (by decide), root291Candidates7_checked]
  change scan edge (search edge 3) 4 [1931] 1931 root291Tail8
    (if 3 ≤ root291Candidates7.length then search edge 3 [1931, 5477] root291Candidates7 7408 winner else winner) = _
  rw [if_neg (by decide)]
theorem root291Scan8 : scan edge (search edge 3) 4 [1931] 1931 root291Tail8 winner = winner := by rw [root291Tail8, scan, if_pos (by decide)]
theorem root291Scan7 : scan edge (search edge 3) 4 [1931] 1931 root291Tail7 winner = winner :=
  (root291Step7).trans root291Scan8
theorem root291Scan6 : scan edge (search edge 3) 4 [1931] 1931 root291Tail6 winner = winner :=
  (root291Step6).trans root291Scan7
theorem root291Scan5 : scan edge (search edge 3) 4 [1931] 1931 root291Tail5 winner = winner :=
  (root291Step5).trans root291Scan6
theorem root291Scan4 : scan edge (search edge 3) 4 [1931] 1931 root291Tail4 winner = winner :=
  (root291Step4).trans root291Scan5
theorem root291Scan3 : scan edge (search edge 3) 4 [1931] 1931 root291Tail3 winner = winner :=
  (root291Step3).trans root291Scan4
theorem root291Scan2 : scan edge (search edge 3) 4 [1931] 1931 root291Tail2 winner = winner :=
  (root291Step2).trans root291Scan3
theorem root291Scan1 : scan edge (search edge 3) 4 [1931] 1931 root291Tail1 winner = winner :=
  (root291Step1).trans root291Scan2
theorem root291Scan0 : scan edge (search edge 3) 4 [1931] 1931 root291Tail0 winner = winner :=
  (root291Step0).trans root291Scan1
set_option maxRecDepth 4096 in
theorem root291Branch : search edge 4 [1931] root291Candidates 1931 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1931] 1931 root291Tail0 winner = winner
  exact root291Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
