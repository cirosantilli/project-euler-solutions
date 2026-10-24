import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root463Candidates : List Nat := Data.row3313
set_option maxRecDepth 8192 in
theorem root463Candidates_checked : candidates edge 5 0 3313 tail464 winner.bound = root463Candidates := by
  apply candidates_eq_row edge 5 0 3313 tail464 root463Candidates winner.bound tail464_ordered
  · decide
  · decide
  · funext q
    rfl
def root463Tail35 : List Nat := []
def root463Tail34 : List Nat := 22447 :: root463Tail35
def root463Tail33 : List Nat := 21121 :: root463Tail34
def root463Tail32 : List Nat := 21061 :: root463Tail33
def root463Tail31 : List Nat := 20551 :: root463Tail32
def root463Tail30 : List Nat := 20173 :: root463Tail31
def root463Tail29 : List Nat := 19507 :: root463Tail30
def root463Tail28 : List Nat := 19447 :: root463Tail29
def root463Tail27 : List Nat := 18919 :: root463Tail28
def root463Tail26 : List Nat := 18541 :: root463Tail27
def root463Tail25 : List Nat := 17683 :: root463Tail26
def root463Tail24 : List Nat := 17299 :: root463Tail25
def root463Tail23 : List Nat := 15679 :: root463Tail24
def root463Tail22 : List Nat := 14923 :: root463Tail23
def root463Tail21 : List Nat := 14449 :: root463Tail22
def root463Tail20 : List Nat := 13831 :: root463Tail21
def root463Tail19 : List Nat := 12781 :: root463Tail20
def root463Tail18 : List Nat := 12601 :: root463Tail19
def root463Tail17 : List Nat := 12301 :: root463Tail18
def root463Tail16 : List Nat := 9739 :: root463Tail17
def root463Tail15 : List Nat := 9349 :: root463Tail16
def root463Tail14 : List Nat := 8779 :: root463Tail15
def root463Tail13 : List Nat := 8629 :: root463Tail14
def root463Tail12 : List Nat := 8089 :: root463Tail13
def root463Tail11 : List Nat := 8059 :: root463Tail12
def root463Tail10 : List Nat := 6871 :: root463Tail11
def root463Tail9 : List Nat := 6637 :: root463Tail10
def root463Tail8 : List Nat := 6421 :: root463Tail9
def root463Tail7 : List Nat := 4951 :: root463Tail8
def root463Tail6 : List Nat := 4831 :: root463Tail7
def root463Tail5 : List Nat := 4723 :: root463Tail6
def root463Tail4 : List Nat := 4663 :: root463Tail5
def root463Tail3 : List Nat := 4159 :: root463Tail4
def root463Tail2 : List Nat := 4111 :: root463Tail3
def root463Tail1 : List Nat := 3733 :: root463Tail2
def root463Tail0 : List Nat := 3673 :: root463Tail1
def root463Candidates0 : List Nat := [8629]
set_option maxRecDepth 8192 in
theorem root463Candidates0_checked : candidates edge 4 3313 3673 root463Tail1 winner.bound = root463Candidates0 := by
  unfold candidates
  have row : edge 3673 = (SortedNatTree.contains FastRows.row3673) := by
    change Data.row3673.contains = (SortedNatTree.contains FastRows.row3673)
    exact FastRows.row3673_checked.symm
  rw [row]
  decide
theorem root463Step0 : scan edge (search edge 3) 4 [3313] 3313 root463Tail0 winner =
    scan edge (search edge 3) 4 [3313] 3313 root463Tail1 winner := by
  rw [root463Tail0, scan, if_neg (by decide), root463Candidates0_checked]
  change scan edge (search edge 3) 4 [3313] 3313 root463Tail1
    (if 3 ≤ root463Candidates0.length then search edge 3 [3313, 3673] root463Candidates0 6986 winner else winner) = _
  rw [if_neg (by decide)]
def root463Candidates1 : List Nat := [6421, 8089, 9349]
set_option maxRecDepth 8192 in
theorem root463Candidates1_checked : candidates edge 4 3313 3733 root463Tail2 winner.bound = root463Candidates1 := by
  unfold candidates
  have row : edge 3733 = (SortedNatTree.contains FastRows.row3733) := by
    change Data.row3733.contains = (SortedNatTree.contains FastRows.row3733)
    exact FastRows.row3733_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root463Child1 : search edge 3 [3313, 3733] root463Candidates1 7046 winner = winner := by decide
theorem root463Step1 : scan edge (search edge 3) 4 [3313] 3313 root463Tail1 winner =
    scan edge (search edge 3) 4 [3313] 3313 root463Tail2 winner := by
  rw [root463Tail1, scan, if_neg (by decide), root463Candidates1_checked]
  change scan edge (search edge 3) 4 [3313] 3313 root463Tail2
    (if 3 ≤ root463Candidates1.length then search edge 3 [3313, 3733] root463Candidates1 7046 winner else winner) = _
  rw [if_pos (by decide), root463Child1]
def root463Candidates2 : List Nat := [6637, 9349]
set_option maxRecDepth 8192 in
theorem root463Candidates2_checked : candidates edge 4 3313 4111 root463Tail3 winner.bound = root463Candidates2 := by
  unfold candidates
  have row : edge 4111 = (SortedNatTree.contains FastRows.row4111) := by
    change Data.row4111.contains = (SortedNatTree.contains FastRows.row4111)
    exact FastRows.row4111_checked.symm
  rw [row]
  decide
theorem root463Step2 : scan edge (search edge 3) 4 [3313] 3313 root463Tail2 winner =
    scan edge (search edge 3) 4 [3313] 3313 root463Tail3 winner := by
  rw [root463Tail2, scan, if_neg (by decide), root463Candidates2_checked]
  change scan edge (search edge 3) 4 [3313] 3313 root463Tail3
    (if 3 ≤ root463Candidates2.length then search edge 3 [3313, 4111] root463Candidates2 7424 winner else winner) = _
  rw [if_neg (by decide)]
def root463Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root463Candidates3_checked : candidates edge 4 3313 4159 root463Tail4 winner.bound = root463Candidates3 := by
  unfold candidates
  have row : edge 4159 = (SortedNatTree.contains FastRows.row4159) := by
    change Data.row4159.contains = (SortedNatTree.contains FastRows.row4159)
    exact FastRows.row4159_checked.symm
  rw [row]
  decide
theorem root463Step3 : scan edge (search edge 3) 4 [3313] 3313 root463Tail3 winner =
    scan edge (search edge 3) 4 [3313] 3313 root463Tail4 winner := by
  rw [root463Tail3, scan, if_neg (by decide), root463Candidates3_checked]
  change scan edge (search edge 3) 4 [3313] 3313 root463Tail4
    (if 3 ≤ root463Candidates3.length then search edge 3 [3313, 4159] root463Candidates3 7472 winner else winner) = _
  rw [if_neg (by decide)]
def root463Candidates4 : List Nat := [6637, 14923]
set_option maxRecDepth 8192 in
theorem root463Candidates4_checked : candidates edge 4 3313 4663 root463Tail5 winner.bound = root463Candidates4 := by
  unfold candidates
  have row : edge 4663 = (SortedNatTree.contains FastRows.row4663) := by
    change Data.row4663.contains = (SortedNatTree.contains FastRows.row4663)
    exact FastRows.row4663_checked.symm
  rw [row]
  decide
theorem root463Step4 : scan edge (search edge 3) 4 [3313] 3313 root463Tail4 winner =
    scan edge (search edge 3) 4 [3313] 3313 root463Tail5 winner := by
  rw [root463Tail4, scan, if_neg (by decide), root463Candidates4_checked]
  change scan edge (search edge 3) 4 [3313] 3313 root463Tail5
    (if 3 ≤ root463Candidates4.length then search edge 3 [3313, 4663] root463Candidates4 7976 winner else winner) = _
  rw [if_neg (by decide)]
def root463Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root463Candidates5_checked : candidates edge 4 3313 4723 root463Tail6 winner.bound = root463Candidates5 := by
  unfold candidates
  have row : edge 4723 = (SortedNatTree.contains FastRows.row4723) := by
    change Data.row4723.contains = (SortedNatTree.contains FastRows.row4723)
    exact FastRows.row4723_checked.symm
  rw [row]
  decide
theorem root463Step5 : scan edge (search edge 3) 4 [3313] 3313 root463Tail5 winner =
    scan edge (search edge 3) 4 [3313] 3313 root463Tail6 winner := by
  rw [root463Tail5, scan, if_neg (by decide), root463Candidates5_checked]
  change scan edge (search edge 3) 4 [3313] 3313 root463Tail6
    (if 3 ≤ root463Candidates5.length then search edge 3 [3313, 4723] root463Candidates5 8036 winner else winner) = _
  rw [if_neg (by decide)]
def root463Candidates6 : List Nat := [4951]
set_option maxRecDepth 8192 in
theorem root463Candidates6_checked : candidates edge 4 3313 4831 root463Tail7 winner.bound = root463Candidates6 := by
  unfold candidates
  have row : edge 4831 = (SortedNatTree.contains FastRows.row4831) := by
    change Data.row4831.contains = (SortedNatTree.contains FastRows.row4831)
    exact FastRows.row4831_checked.symm
  rw [row]
  decide
theorem root463Step6 : scan edge (search edge 3) 4 [3313] 3313 root463Tail6 winner =
    scan edge (search edge 3) 4 [3313] 3313 root463Tail7 winner := by
  rw [root463Tail6, scan, if_neg (by decide), root463Candidates6_checked]
  change scan edge (search edge 3) 4 [3313] 3313 root463Tail7
    (if 3 ≤ root463Candidates6.length then search edge 3 [3313, 4831] root463Candidates6 8144 winner else winner) = _
  rw [if_neg (by decide)]
def root463Candidates7 : List Nat := [8779]
set_option maxRecDepth 8192 in
theorem root463Candidates7_checked : candidates edge 4 3313 4951 root463Tail8 winner.bound = root463Candidates7 := by
  unfold candidates
  have row : edge 4951 = (SortedNatTree.contains FastRows.row4951) := by
    change Data.row4951.contains = (SortedNatTree.contains FastRows.row4951)
    exact FastRows.row4951_checked.symm
  rw [row]
  decide
theorem root463Step7 : scan edge (search edge 3) 4 [3313] 3313 root463Tail7 winner =
    scan edge (search edge 3) 4 [3313] 3313 root463Tail8 winner := by
  rw [root463Tail7, scan, if_neg (by decide), root463Candidates7_checked]
  change scan edge (search edge 3) 4 [3313] 3313 root463Tail8
    (if 3 ≤ root463Candidates7.length then search edge 3 [3313, 4951] root463Candidates7 8264 winner else winner) = _
  rw [if_neg (by decide)]
theorem root463Scan8 : scan edge (search edge 3) 4 [3313] 3313 root463Tail8 winner = winner := by rw [root463Tail8, scan, if_pos (by decide)]
theorem root463Scan7 : scan edge (search edge 3) 4 [3313] 3313 root463Tail7 winner = winner :=
  (root463Step7).trans root463Scan8
theorem root463Scan6 : scan edge (search edge 3) 4 [3313] 3313 root463Tail6 winner = winner :=
  (root463Step6).trans root463Scan7
theorem root463Scan5 : scan edge (search edge 3) 4 [3313] 3313 root463Tail5 winner = winner :=
  (root463Step5).trans root463Scan6
theorem root463Scan4 : scan edge (search edge 3) 4 [3313] 3313 root463Tail4 winner = winner :=
  (root463Step4).trans root463Scan5
theorem root463Scan3 : scan edge (search edge 3) 4 [3313] 3313 root463Tail3 winner = winner :=
  (root463Step3).trans root463Scan4
theorem root463Scan2 : scan edge (search edge 3) 4 [3313] 3313 root463Tail2 winner = winner :=
  (root463Step2).trans root463Scan3
theorem root463Scan1 : scan edge (search edge 3) 4 [3313] 3313 root463Tail1 winner = winner :=
  (root463Step1).trans root463Scan2
theorem root463Scan0 : scan edge (search edge 3) 4 [3313] 3313 root463Tail0 winner = winner :=
  (root463Step0).trans root463Scan1
set_option maxRecDepth 4096 in
theorem root463Branch : search edge 4 [3313] root463Candidates 3313 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3313] 3313 root463Tail0 winner = winner
  exact root463Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
