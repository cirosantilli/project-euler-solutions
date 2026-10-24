import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root338Candidates : List Nat := Data.row2293
set_option maxRecDepth 8192 in
theorem root338Candidates_checked : candidates edge 5 0 2293 tail339 winner.bound = root338Candidates := by
  apply candidates_eq_row edge 5 0 2293 tail339 root338Candidates winner.bound tail339_ordered
  · decide
  · decide
  · funext q
    rfl
def root338Tail29 : List Nat := []
def root338Tail28 : List Nat := 23671 :: root338Tail29
def root338Tail27 : List Nat := 21943 :: root338Tail28
def root338Tail26 : List Nat := 21019 :: root338Tail27
def root338Tail25 : List Nat := 20353 :: root338Tail26
def root338Tail24 : List Nat := 19309 :: root338Tail25
def root338Tail23 : List Nat := 18517 :: root338Tail24
def root338Tail22 : List Nat := 17827 :: root338Tail23
def root338Tail21 : List Nat := 17467 :: root338Tail22
def root338Tail20 : List Nat := 16447 :: root338Tail21
def root338Tail19 : List Nat := 14851 :: root338Tail20
def root338Tail18 : List Nat := 14563 :: root338Tail19
def root338Tail17 : List Nat := 13093 :: root338Tail18
def root338Tail16 : List Nat := 12211 :: root338Tail17
def root338Tail15 : List Nat := 9277 :: root338Tail16
def root338Tail14 : List Nat := 9199 :: root338Tail15
def root338Tail13 : List Nat := 9187 :: root338Tail14
def root338Tail12 : List Nat := 8011 :: root338Tail13
def root338Tail11 : List Nat := 7333 :: root338Tail12
def root338Tail10 : List Nat := 7039 :: root338Tail11
def root338Tail9 : List Nat := 6829 :: root338Tail10
def root338Tail8 : List Nat := 5827 :: root338Tail9
def root338Tail7 : List Nat := 5749 :: root338Tail8
def root338Tail6 : List Nat := 5569 :: root338Tail7
def root338Tail5 : List Nat := 5281 :: root338Tail6
def root338Tail4 : List Nat := 5197 :: root338Tail5
def root338Tail3 : List Nat := 3853 :: root338Tail4
def root338Tail2 : List Nat := 3727 :: root338Tail3
def root338Tail1 : List Nat := 2683 :: root338Tail2
def root338Tail0 : List Nat := 2671 :: root338Tail1
def root338Candidates0 : List Nat := [3853, 6829]
set_option maxRecDepth 8192 in
theorem root338Candidates0_checked : candidates edge 4 2293 2671 root338Tail1 winner.bound = root338Candidates0 := by
  unfold candidates
  have row : edge 2671 = (SortedNatTree.contains FastRows.row2671) := by
    change Data.row2671.contains = (SortedNatTree.contains FastRows.row2671)
    exact FastRows.row2671_checked.symm
  rw [row]
  decide
theorem root338Step0 : scan edge (search edge 3) 4 [2293] 2293 root338Tail0 winner =
    scan edge (search edge 3) 4 [2293] 2293 root338Tail1 winner := by
  rw [root338Tail0, scan, if_neg (by decide), root338Candidates0_checked]
  change scan edge (search edge 3) 4 [2293] 2293 root338Tail1
    (if 3 ≤ root338Candidates0.length then search edge 3 [2293, 2671] root338Candidates0 4964 winner else winner) = _
  rw [if_neg (by decide)]
def root338Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root338Candidates1_checked : candidates edge 4 2293 2683 root338Tail2 winner.bound = root338Candidates1 := by
  unfold candidates
  have row : edge 2683 = (SortedNatTree.contains FastRows.row2683) := by
    change Data.row2683.contains = (SortedNatTree.contains FastRows.row2683)
    exact FastRows.row2683_checked.symm
  rw [row]
  decide
theorem root338Step1 : scan edge (search edge 3) 4 [2293] 2293 root338Tail1 winner =
    scan edge (search edge 3) 4 [2293] 2293 root338Tail2 winner := by
  rw [root338Tail1, scan, if_neg (by decide), root338Candidates1_checked]
  change scan edge (search edge 3) 4 [2293] 2293 root338Tail2
    (if 3 ≤ root338Candidates1.length then search edge 3 [2293, 2683] root338Candidates1 4976 winner else winner) = _
  rw [if_neg (by decide)]
def root338Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root338Candidates2_checked : candidates edge 4 2293 3727 root338Tail3 winner.bound = root338Candidates2 := by
  unfold candidates
  have row : edge 3727 = (SortedNatTree.contains FastRows.row3727) := by
    change Data.row3727.contains = (SortedNatTree.contains FastRows.row3727)
    exact FastRows.row3727_checked.symm
  rw [row]
  decide
theorem root338Step2 : scan edge (search edge 3) 4 [2293] 2293 root338Tail2 winner =
    scan edge (search edge 3) 4 [2293] 2293 root338Tail3 winner := by
  rw [root338Tail2, scan, if_neg (by decide), root338Candidates2_checked]
  change scan edge (search edge 3) 4 [2293] 2293 root338Tail3
    (if 3 ≤ root338Candidates2.length then search edge 3 [2293, 3727] root338Candidates2 6020 winner else winner) = _
  rw [if_neg (by decide)]
def root338Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root338Candidates3_checked : candidates edge 4 2293 3853 root338Tail4 winner.bound = root338Candidates3 := by
  unfold candidates
  have row : edge 3853 = (SortedNatTree.contains FastRows.row3853) := by
    change Data.row3853.contains = (SortedNatTree.contains FastRows.row3853)
    exact FastRows.row3853_checked.symm
  rw [row]
  decide
theorem root338Step3 : scan edge (search edge 3) 4 [2293] 2293 root338Tail3 winner =
    scan edge (search edge 3) 4 [2293] 2293 root338Tail4 winner := by
  rw [root338Tail3, scan, if_neg (by decide), root338Candidates3_checked]
  change scan edge (search edge 3) 4 [2293] 2293 root338Tail4
    (if 3 ≤ root338Candidates3.length then search edge 3 [2293, 3853] root338Candidates3 6146 winner else winner) = _
  rw [if_neg (by decide)]
def root338Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root338Candidates4_checked : candidates edge 4 2293 5197 root338Tail5 winner.bound = root338Candidates4 := by
  unfold candidates
  have row : edge 5197 = (SortedNatTree.contains FastRows.row5197) := by
    change Data.row5197.contains = (SortedNatTree.contains FastRows.row5197)
    exact FastRows.row5197_checked.symm
  rw [row]
  decide
theorem root338Step4 : scan edge (search edge 3) 4 [2293] 2293 root338Tail4 winner =
    scan edge (search edge 3) 4 [2293] 2293 root338Tail5 winner := by
  rw [root338Tail4, scan, if_neg (by decide), root338Candidates4_checked]
  change scan edge (search edge 3) 4 [2293] 2293 root338Tail5
    (if 3 ≤ root338Candidates4.length then search edge 3 [2293, 5197] root338Candidates4 7490 winner else winner) = _
  rw [if_neg (by decide)]
def root338Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root338Candidates5_checked : candidates edge 4 2293 5281 root338Tail6 winner.bound = root338Candidates5 := by
  unfold candidates
  have row : edge 5281 = (SortedNatTree.contains FastRows.row5281) := by
    change Data.row5281.contains = (SortedNatTree.contains FastRows.row5281)
    exact FastRows.row5281_checked.symm
  rw [row]
  decide
theorem root338Step5 : scan edge (search edge 3) 4 [2293] 2293 root338Tail5 winner =
    scan edge (search edge 3) 4 [2293] 2293 root338Tail6 winner := by
  rw [root338Tail5, scan, if_neg (by decide), root338Candidates5_checked]
  change scan edge (search edge 3) 4 [2293] 2293 root338Tail6
    (if 3 ≤ root338Candidates5.length then search edge 3 [2293, 5281] root338Candidates5 7574 winner else winner) = _
  rw [if_neg (by decide)]
def root338Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root338Candidates6_checked : candidates edge 4 2293 5569 root338Tail7 winner.bound = root338Candidates6 := by
  unfold candidates
  have row : edge 5569 = (SortedNatTree.contains FastRows.row5569) := by
    change Data.row5569.contains = (SortedNatTree.contains FastRows.row5569)
    exact FastRows.row5569_checked.symm
  rw [row]
  decide
theorem root338Step6 : scan edge (search edge 3) 4 [2293] 2293 root338Tail6 winner =
    scan edge (search edge 3) 4 [2293] 2293 root338Tail7 winner := by
  rw [root338Tail6, scan, if_neg (by decide), root338Candidates6_checked]
  change scan edge (search edge 3) 4 [2293] 2293 root338Tail7
    (if 3 ≤ root338Candidates6.length then search edge 3 [2293, 5569] root338Candidates6 7862 winner else winner) = _
  rw [if_neg (by decide)]
def root338Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root338Candidates7_checked : candidates edge 4 2293 5749 root338Tail8 winner.bound = root338Candidates7 := by
  unfold candidates
  have row : edge 5749 = (SortedNatTree.contains FastRows.row5749) := by
    change Data.row5749.contains = (SortedNatTree.contains FastRows.row5749)
    exact FastRows.row5749_checked.symm
  rw [row]
  decide
theorem root338Step7 : scan edge (search edge 3) 4 [2293] 2293 root338Tail7 winner =
    scan edge (search edge 3) 4 [2293] 2293 root338Tail8 winner := by
  rw [root338Tail7, scan, if_neg (by decide), root338Candidates7_checked]
  change scan edge (search edge 3) 4 [2293] 2293 root338Tail8
    (if 3 ≤ root338Candidates7.length then search edge 3 [2293, 5749] root338Candidates7 8042 winner else winner) = _
  rw [if_neg (by decide)]
def root338Candidates8 : List Nat := [14563]
set_option maxRecDepth 8192 in
theorem root338Candidates8_checked : candidates edge 4 2293 5827 root338Tail9 winner.bound = root338Candidates8 := by
  unfold candidates
  have row : edge 5827 = (SortedNatTree.contains FastRows.row5827) := by
    change Data.row5827.contains = (SortedNatTree.contains FastRows.row5827)
    exact FastRows.row5827_checked.symm
  rw [row]
  decide
theorem root338Step8 : scan edge (search edge 3) 4 [2293] 2293 root338Tail8 winner =
    scan edge (search edge 3) 4 [2293] 2293 root338Tail9 winner := by
  rw [root338Tail8, scan, if_neg (by decide), root338Candidates8_checked]
  change scan edge (search edge 3) 4 [2293] 2293 root338Tail9
    (if 3 ≤ root338Candidates8.length then search edge 3 [2293, 5827] root338Candidates8 8120 winner else winner) = _
  rw [if_neg (by decide)]
theorem root338Scan9 : scan edge (search edge 3) 4 [2293] 2293 root338Tail9 winner = winner := by rw [root338Tail9, scan, if_pos (by decide)]
theorem root338Scan8 : scan edge (search edge 3) 4 [2293] 2293 root338Tail8 winner = winner :=
  (root338Step8).trans root338Scan9
theorem root338Scan7 : scan edge (search edge 3) 4 [2293] 2293 root338Tail7 winner = winner :=
  (root338Step7).trans root338Scan8
theorem root338Scan6 : scan edge (search edge 3) 4 [2293] 2293 root338Tail6 winner = winner :=
  (root338Step6).trans root338Scan7
theorem root338Scan5 : scan edge (search edge 3) 4 [2293] 2293 root338Tail5 winner = winner :=
  (root338Step5).trans root338Scan6
theorem root338Scan4 : scan edge (search edge 3) 4 [2293] 2293 root338Tail4 winner = winner :=
  (root338Step4).trans root338Scan5
theorem root338Scan3 : scan edge (search edge 3) 4 [2293] 2293 root338Tail3 winner = winner :=
  (root338Step3).trans root338Scan4
theorem root338Scan2 : scan edge (search edge 3) 4 [2293] 2293 root338Tail2 winner = winner :=
  (root338Step2).trans root338Scan3
theorem root338Scan1 : scan edge (search edge 3) 4 [2293] 2293 root338Tail1 winner = winner :=
  (root338Step1).trans root338Scan2
theorem root338Scan0 : scan edge (search edge 3) 4 [2293] 2293 root338Tail0 winner = winner :=
  (root338Step0).trans root338Scan1
set_option maxRecDepth 4096 in
theorem root338Branch : search edge 4 [2293] root338Candidates 2293 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2293] 2293 root338Tail0 winner = winner
  exact root338Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
