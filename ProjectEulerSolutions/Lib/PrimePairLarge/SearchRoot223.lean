import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root223Candidates : List Nat := Data.row1429
set_option maxRecDepth 8192 in
theorem root223Candidates_checked : candidates edge 5 0 1429 tail224 winner.bound = root223Candidates := by
  apply candidates_eq_row edge 5 0 1429 tail224 root223Candidates winner.bound tail224_ordered
  · decide
  · decide
  · funext q
    rfl
def root223Tail43 : List Nat := []
def root223Tail42 : List Nat := 23899 :: root223Tail43
def root223Tail41 : List Nat := 23017 :: root223Tail42
def root223Tail40 : List Nat := 22963 :: root223Tail41
def root223Tail39 : List Nat := 22861 :: root223Tail40
def root223Tail38 : List Nat := 22543 :: root223Tail39
def root223Tail37 : List Nat := 21799 :: root223Tail38
def root223Tail36 : List Nat := 20611 :: root223Tail37
def root223Tail35 : List Nat := 19489 :: root223Tail36
def root223Tail34 : List Nat := 19267 :: root223Tail35
def root223Tail33 : List Nat := 18397 :: root223Tail34
def root223Tail32 : List Nat := 18229 :: root223Tail33
def root223Tail31 : List Nat := 17041 :: root223Tail32
def root223Tail30 : List Nat := 16339 :: root223Tail31
def root223Tail29 : List Nat := 15439 :: root223Tail30
def root223Tail28 : List Nat := 14797 :: root223Tail29
def root223Tail27 : List Nat := 14221 :: root223Tail28
def root223Tail26 : List Nat := 11497 :: root223Tail27
def root223Tail25 : List Nat := 11113 :: root223Tail26
def root223Tail24 : List Nat := 10399 :: root223Tail25
def root223Tail23 : List Nat := 10321 :: root223Tail24
def root223Tail22 : List Nat := 10069 :: root223Tail23
def root223Tail21 : List Nat := 9619 :: root223Tail22
def root223Tail20 : List Nat := 9433 :: root223Tail21
def root223Tail19 : List Nat := 8737 :: root223Tail20
def root223Tail18 : List Nat := 8527 :: root223Tail19
def root223Tail17 : List Nat := 8101 :: root223Tail18
def root223Tail16 : List Nat := 7963 :: root223Tail17
def root223Tail15 : List Nat := 7477 :: root223Tail16
def root223Tail14 : List Nat := 7351 :: root223Tail15
def root223Tail13 : List Nat := 6967 :: root223Tail14
def root223Tail12 : List Nat := 6619 :: root223Tail13
def root223Tail11 : List Nat := 6553 :: root223Tail12
def root223Tail10 : List Nat := 6427 :: root223Tail11
def root223Tail9 : List Nat := 6343 :: root223Tail10
def root223Tail8 : List Nat := 6211 :: root223Tail9
def root223Tail7 : List Nat := 4957 :: root223Tail8
def root223Tail6 : List Nat := 3919 :: root223Tail7
def root223Tail5 : List Nat := 3739 :: root223Tail6
def root223Tail4 : List Nat := 3691 :: root223Tail5
def root223Tail3 : List Nat := 3313 :: root223Tail4
def root223Tail2 : List Nat := 2221 :: root223Tail3
def root223Tail1 : List Nat := 1831 :: root223Tail2
def root223Tail0 : List Nat := 1723 :: root223Tail1
def root223Candidates0 : List Nat := [1831]
set_option maxRecDepth 8192 in
theorem root223Candidates0_checked : candidates edge 4 1429 1723 root223Tail1 winner.bound = root223Candidates0 := by
  unfold candidates
  have row : edge 1723 = (SortedNatTree.contains FastRows.row1723) := by
    change Data.row1723.contains = (SortedNatTree.contains FastRows.row1723)
    exact FastRows.row1723_checked.symm
  rw [row]
  decide
theorem root223Step0 : scan edge (search edge 3) 4 [1429] 1429 root223Tail0 winner =
    scan edge (search edge 3) 4 [1429] 1429 root223Tail1 winner := by
  rw [root223Tail0, scan, if_neg (by decide), root223Candidates0_checked]
  change scan edge (search edge 3) 4 [1429] 1429 root223Tail1
    (if 3 ≤ root223Candidates0.length then search edge 3 [1429, 1723] root223Candidates0 3152 winner else winner) = _
  rw [if_neg (by decide)]
def root223Candidates1 : List Nat := [20611]
set_option maxRecDepth 8192 in
theorem root223Candidates1_checked : candidates edge 4 1429 1831 root223Tail2 winner.bound = root223Candidates1 := by
  unfold candidates
  have row : edge 1831 = (SortedNatTree.contains FastRows.row1831) := by
    change Data.row1831.contains = (SortedNatTree.contains FastRows.row1831)
    exact FastRows.row1831_checked.symm
  rw [row]
  decide
theorem root223Step1 : scan edge (search edge 3) 4 [1429] 1429 root223Tail1 winner =
    scan edge (search edge 3) 4 [1429] 1429 root223Tail2 winner := by
  rw [root223Tail1, scan, if_neg (by decide), root223Candidates1_checked]
  change scan edge (search edge 3) 4 [1429] 1429 root223Tail2
    (if 3 ≤ root223Candidates1.length then search edge 3 [1429, 1831] root223Candidates1 3260 winner else winner) = _
  rw [if_neg (by decide)]
def root223Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root223Candidates2_checked : candidates edge 4 1429 2221 root223Tail3 winner.bound = root223Candidates2 := by
  unfold candidates
  have row : edge 2221 = (SortedNatTree.contains FastRows.row2221) := by
    change Data.row2221.contains = (SortedNatTree.contains FastRows.row2221)
    exact FastRows.row2221_checked.symm
  rw [row]
  decide
theorem root223Step2 : scan edge (search edge 3) 4 [1429] 1429 root223Tail2 winner =
    scan edge (search edge 3) 4 [1429] 1429 root223Tail3 winner := by
  rw [root223Tail2, scan, if_neg (by decide), root223Candidates2_checked]
  change scan edge (search edge 3) 4 [1429] 1429 root223Tail3
    (if 3 ≤ root223Candidates2.length then search edge 3 [1429, 2221] root223Candidates2 3650 winner else winner) = _
  rw [if_neg (by decide)]
def root223Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root223Candidates3_checked : candidates edge 4 1429 3313 root223Tail4 winner.bound = root223Candidates3 := by
  unfold candidates
  have row : edge 3313 = (SortedNatTree.contains FastRows.row3313) := by
    change Data.row3313.contains = (SortedNatTree.contains FastRows.row3313)
    exact FastRows.row3313_checked.symm
  rw [row]
  decide
theorem root223Step3 : scan edge (search edge 3) 4 [1429] 1429 root223Tail3 winner =
    scan edge (search edge 3) 4 [1429] 1429 root223Tail4 winner := by
  rw [root223Tail3, scan, if_neg (by decide), root223Candidates3_checked]
  change scan edge (search edge 3) 4 [1429] 1429 root223Tail4
    (if 3 ≤ root223Candidates3.length then search edge 3 [1429, 3313] root223Candidates3 4742 winner else winner) = _
  rw [if_neg (by decide)]
def root223Candidates4 : List Nat := [7477, 9433]
set_option maxRecDepth 8192 in
theorem root223Candidates4_checked : candidates edge 4 1429 3691 root223Tail5 winner.bound = root223Candidates4 := by
  unfold candidates
  have row : edge 3691 = (SortedNatTree.contains FastRows.row3691) := by
    change Data.row3691.contains = (SortedNatTree.contains FastRows.row3691)
    exact FastRows.row3691_checked.symm
  rw [row]
  decide
theorem root223Step4 : scan edge (search edge 3) 4 [1429] 1429 root223Tail4 winner =
    scan edge (search edge 3) 4 [1429] 1429 root223Tail5 winner := by
  rw [root223Tail4, scan, if_neg (by decide), root223Candidates4_checked]
  change scan edge (search edge 3) 4 [1429] 1429 root223Tail5
    (if 3 ≤ root223Candidates4.length then search edge 3 [1429, 3691] root223Candidates4 5120 winner else winner) = _
  rw [if_neg (by decide)]
def root223Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root223Candidates5_checked : candidates edge 4 1429 3739 root223Tail6 winner.bound = root223Candidates5 := by
  unfold candidates
  have row : edge 3739 = (SortedNatTree.contains FastRows.row3739) := by
    change Data.row3739.contains = (SortedNatTree.contains FastRows.row3739)
    exact FastRows.row3739_checked.symm
  rw [row]
  decide
theorem root223Step5 : scan edge (search edge 3) 4 [1429] 1429 root223Tail5 winner =
    scan edge (search edge 3) 4 [1429] 1429 root223Tail6 winner := by
  rw [root223Tail5, scan, if_neg (by decide), root223Candidates5_checked]
  change scan edge (search edge 3) 4 [1429] 1429 root223Tail6
    (if 3 ≤ root223Candidates5.length then search edge 3 [1429, 3739] root223Candidates5 5168 winner else winner) = _
  rw [if_neg (by decide)]
def root223Candidates6 : List Nat := [4957]
set_option maxRecDepth 8192 in
theorem root223Candidates6_checked : candidates edge 4 1429 3919 root223Tail7 winner.bound = root223Candidates6 := by
  unfold candidates
  have row : edge 3919 = (SortedNatTree.contains FastRows.row3919) := by
    change Data.row3919.contains = (SortedNatTree.contains FastRows.row3919)
    exact FastRows.row3919_checked.symm
  rw [row]
  decide
theorem root223Step6 : scan edge (search edge 3) 4 [1429] 1429 root223Tail6 winner =
    scan edge (search edge 3) 4 [1429] 1429 root223Tail7 winner := by
  rw [root223Tail6, scan, if_neg (by decide), root223Candidates6_checked]
  change scan edge (search edge 3) 4 [1429] 1429 root223Tail7
    (if 3 ≤ root223Candidates6.length then search edge 3 [1429, 3919] root223Candidates6 5348 winner else winner) = _
  rw [if_neg (by decide)]
def root223Candidates7 : List Nat := [17041, 18397]
set_option maxRecDepth 8192 in
theorem root223Candidates7_checked : candidates edge 4 1429 4957 root223Tail8 winner.bound = root223Candidates7 := by
  unfold candidates
  have row : edge 4957 = (SortedNatTree.contains FastRows.row4957) := by
    change Data.row4957.contains = (SortedNatTree.contains FastRows.row4957)
    exact FastRows.row4957_checked.symm
  rw [row]
  decide
theorem root223Step7 : scan edge (search edge 3) 4 [1429] 1429 root223Tail7 winner =
    scan edge (search edge 3) 4 [1429] 1429 root223Tail8 winner := by
  rw [root223Tail7, scan, if_neg (by decide), root223Candidates7_checked]
  change scan edge (search edge 3) 4 [1429] 1429 root223Tail8
    (if 3 ≤ root223Candidates7.length then search edge 3 [1429, 4957] root223Candidates7 6386 winner else winner) = _
  rw [if_neg (by decide)]
theorem root223Scan8 : scan edge (search edge 3) 4 [1429] 1429 root223Tail8 winner = winner := by rw [root223Tail8, scan, if_pos (by decide)]
theorem root223Scan7 : scan edge (search edge 3) 4 [1429] 1429 root223Tail7 winner = winner :=
  (root223Step7).trans root223Scan8
theorem root223Scan6 : scan edge (search edge 3) 4 [1429] 1429 root223Tail6 winner = winner :=
  (root223Step6).trans root223Scan7
theorem root223Scan5 : scan edge (search edge 3) 4 [1429] 1429 root223Tail5 winner = winner :=
  (root223Step5).trans root223Scan6
theorem root223Scan4 : scan edge (search edge 3) 4 [1429] 1429 root223Tail4 winner = winner :=
  (root223Step4).trans root223Scan5
theorem root223Scan3 : scan edge (search edge 3) 4 [1429] 1429 root223Tail3 winner = winner :=
  (root223Step3).trans root223Scan4
theorem root223Scan2 : scan edge (search edge 3) 4 [1429] 1429 root223Tail2 winner = winner :=
  (root223Step2).trans root223Scan3
theorem root223Scan1 : scan edge (search edge 3) 4 [1429] 1429 root223Tail1 winner = winner :=
  (root223Step1).trans root223Scan2
theorem root223Scan0 : scan edge (search edge 3) 4 [1429] 1429 root223Tail0 winner = winner :=
  (root223Step0).trans root223Scan1
set_option maxRecDepth 4096 in
theorem root223Branch : search edge 4 [1429] root223Candidates 1429 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1429] 1429 root223Tail0 winner = winner
  exact root223Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
