import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root541Candidates : List Nat := Data.row3923
set_option maxRecDepth 8192 in
theorem root541Candidates_checked : candidates edge 5 0 3923 tail542 winner.bound = root541Candidates := by
  apply candidates_eq_row edge 5 0 3923 tail542 root541Candidates winner.bound tail542_ordered
  · decide
  · decide
  · funext q
    rfl
def root541Tail35 : List Nat := []
def root541Tail34 : List Nat := 20771 :: root541Tail35
def root541Tail33 : List Nat := 20297 :: root541Tail34
def root541Tail32 : List Nat := 19997 :: root541Tail33
def root541Tail31 : List Nat := 19583 :: root541Tail32
def root541Tail30 : List Nat := 19427 :: root541Tail31
def root541Tail29 : List Nat := 19373 :: root541Tail30
def root541Tail28 : List Nat := 18233 :: root541Tail29
def root541Tail27 : List Nat := 16103 :: root541Tail28
def root541Tail26 : List Nat := 14411 :: root541Tail27
def root541Tail25 : List Nat := 13241 :: root541Tail26
def root541Tail24 : List Nat := 12497 :: root541Tail25
def root541Tail23 : List Nat := 11783 :: root541Tail24
def root541Tail22 : List Nat := 11399 :: root541Tail23
def root541Tail21 : List Nat := 10937 :: root541Tail22
def root541Tail20 : List Nat := 10313 :: root541Tail21
def root541Tail19 : List Nat := 9467 :: root541Tail20
def root541Tail18 : List Nat := 9377 :: root541Tail19
def root541Tail17 : List Nat := 9041 :: root541Tail18
def root541Tail16 : List Nat := 8861 :: root541Tail17
def root541Tail15 : List Nat := 8783 :: root541Tail16
def root541Tail14 : List Nat := 8663 :: root541Tail15
def root541Tail13 : List Nat := 8627 :: root541Tail14
def root541Tail12 : List Nat := 8237 :: root541Tail13
def root541Tail11 : List Nat := 7949 :: root541Tail12
def root541Tail10 : List Nat := 7643 :: root541Tail11
def root541Tail9 : List Nat := 6803 :: root541Tail10
def root541Tail8 : List Nat := 6011 :: root541Tail9
def root541Tail7 : List Nat := 5519 :: root541Tail8
def root541Tail6 : List Nat := 4973 :: root541Tail7
def root541Tail5 : List Nat := 4919 :: root541Tail6
def root541Tail4 : List Nat := 4721 :: root541Tail5
def root541Tail3 : List Nat := 4703 :: root541Tail4
def root541Tail2 : List Nat := 4211 :: root541Tail3
def root541Tail1 : List Nat := 4133 :: root541Tail2
def root541Tail0 : List Nat := 3989 :: root541Tail1
def root541Candidates0 : List Nat := [8861]
set_option maxRecDepth 8192 in
theorem root541Candidates0_checked : candidates edge 4 3923 3989 root541Tail1 winner.bound = root541Candidates0 := by
  unfold candidates
  have row : edge 3989 = (SortedNatTree.contains FastRows.row3989) := by
    change Data.row3989.contains = (SortedNatTree.contains FastRows.row3989)
    exact FastRows.row3989_checked.symm
  rw [row]
  decide
theorem root541Step0 : scan edge (search edge 3) 4 [3923] 3923 root541Tail0 winner =
    scan edge (search edge 3) 4 [3923] 3923 root541Tail1 winner := by
  rw [root541Tail0, scan, if_neg (by decide), root541Candidates0_checked]
  change scan edge (search edge 3) 4 [3923] 3923 root541Tail1
    (if 3 ≤ root541Candidates0.length then search edge 3 [3923, 3989] root541Candidates0 7912 winner else winner) = _
  rw [if_neg (by decide)]
def root541Candidates1 : List Nat := [4721]
set_option maxRecDepth 8192 in
theorem root541Candidates1_checked : candidates edge 4 3923 4133 root541Tail2 winner.bound = root541Candidates1 := by
  unfold candidates
  have row : edge 4133 = (SortedNatTree.contains FastRows.row4133) := by
    change Data.row4133.contains = (SortedNatTree.contains FastRows.row4133)
    exact FastRows.row4133_checked.symm
  rw [row]
  decide
theorem root541Step1 : scan edge (search edge 3) 4 [3923] 3923 root541Tail1 winner =
    scan edge (search edge 3) 4 [3923] 3923 root541Tail2 winner := by
  rw [root541Tail1, scan, if_neg (by decide), root541Candidates1_checked]
  change scan edge (search edge 3) 4 [3923] 3923 root541Tail2
    (if 3 ≤ root541Candidates1.length then search edge 3 [3923, 4133] root541Candidates1 8056 winner else winner) = _
  rw [if_neg (by decide)]
def root541Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root541Candidates2_checked : candidates edge 4 3923 4211 root541Tail3 winner.bound = root541Candidates2 := by
  unfold candidates
  have row : edge 4211 = (SortedNatTree.contains FastRows.row4211) := by
    change Data.row4211.contains = (SortedNatTree.contains FastRows.row4211)
    exact FastRows.row4211_checked.symm
  rw [row]
  decide
theorem root541Step2 : scan edge (search edge 3) 4 [3923] 3923 root541Tail2 winner =
    scan edge (search edge 3) 4 [3923] 3923 root541Tail3 winner := by
  rw [root541Tail2, scan, if_neg (by decide), root541Candidates2_checked]
  change scan edge (search edge 3) 4 [3923] 3923 root541Tail3
    (if 3 ≤ root541Candidates2.length then search edge 3 [3923, 4211] root541Candidates2 8134 winner else winner) = _
  rw [if_neg (by decide)]
def root541Candidates3 : List Nat := [8627]
set_option maxRecDepth 8192 in
theorem root541Candidates3_checked : candidates edge 4 3923 4703 root541Tail4 winner.bound = root541Candidates3 := by
  unfold candidates
  have row : edge 4703 = (SortedNatTree.contains FastRows.row4703) := by
    change Data.row4703.contains = (SortedNatTree.contains FastRows.row4703)
    exact FastRows.row4703_checked.symm
  rw [row]
  decide
theorem root541Step3 : scan edge (search edge 3) 4 [3923] 3923 root541Tail3 winner =
    scan edge (search edge 3) 4 [3923] 3923 root541Tail4 winner := by
  rw [root541Tail3, scan, if_neg (by decide), root541Candidates3_checked]
  change scan edge (search edge 3) 4 [3923] 3923 root541Tail4
    (if 3 ≤ root541Candidates3.length then search edge 3 [3923, 4703] root541Candidates3 8626 winner else winner) = _
  rw [if_neg (by decide)]
def root541Candidates4 : List Nat := [4919, 7949]
set_option maxRecDepth 8192 in
theorem root541Candidates4_checked : candidates edge 4 3923 4721 root541Tail5 winner.bound = root541Candidates4 := by
  unfold candidates
  have row : edge 4721 = (SortedNatTree.contains FastRows.row4721) := by
    change Data.row4721.contains = (SortedNatTree.contains FastRows.row4721)
    exact FastRows.row4721_checked.symm
  rw [row]
  decide
theorem root541Step4 : scan edge (search edge 3) 4 [3923] 3923 root541Tail4 winner =
    scan edge (search edge 3) 4 [3923] 3923 root541Tail5 winner := by
  rw [root541Tail4, scan, if_neg (by decide), root541Candidates4_checked]
  change scan edge (search edge 3) 4 [3923] 3923 root541Tail5
    (if 3 ≤ root541Candidates4.length then search edge 3 [3923, 4721] root541Candidates4 8644 winner else winner) = _
  rw [if_neg (by decide)]
def root541Candidates5 : List Nat := [6803, 8627, 8783]
set_option maxRecDepth 8192 in
theorem root541Candidates5_checked : candidates edge 4 3923 4919 root541Tail6 winner.bound = root541Candidates5 := by
  unfold candidates
  have row : edge 4919 = (SortedNatTree.contains FastRows.row4919) := by
    change Data.row4919.contains = (SortedNatTree.contains FastRows.row4919)
    exact FastRows.row4919_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root541Child5 : search edge 3 [3923, 4919] root541Candidates5 8842 winner = winner := by decide
theorem root541Step5 : scan edge (search edge 3) 4 [3923] 3923 root541Tail5 winner =
    scan edge (search edge 3) 4 [3923] 3923 root541Tail6 winner := by
  rw [root541Tail5, scan, if_neg (by decide), root541Candidates5_checked]
  change scan edge (search edge 3) 4 [3923] 3923 root541Tail6
    (if 3 ≤ root541Candidates5.length then search edge 3 [3923, 4919] root541Candidates5 8842 winner else winner) = _
  rw [if_pos (by decide), root541Child5]
def root541Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root541Candidates6_checked : candidates edge 4 3923 4973 root541Tail7 winner.bound = root541Candidates6 := by
  unfold candidates
  have row : edge 4973 = (SortedNatTree.contains FastRows.row4973) := by
    change Data.row4973.contains = (SortedNatTree.contains FastRows.row4973)
    exact FastRows.row4973_checked.symm
  rw [row]
  decide
theorem root541Step6 : scan edge (search edge 3) 4 [3923] 3923 root541Tail6 winner =
    scan edge (search edge 3) 4 [3923] 3923 root541Tail7 winner := by
  rw [root541Tail6, scan, if_neg (by decide), root541Candidates6_checked]
  change scan edge (search edge 3) 4 [3923] 3923 root541Tail7
    (if 3 ≤ root541Candidates6.length then search edge 3 [3923, 4973] root541Candidates6 8896 winner else winner) = _
  rw [if_neg (by decide)]
def root541Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root541Candidates7_checked : candidates edge 4 3923 5519 root541Tail8 winner.bound = root541Candidates7 := by
  unfold candidates
  have row : edge 5519 = (SortedNatTree.contains FastRows.row5519) := by
    change Data.row5519.contains = (SortedNatTree.contains FastRows.row5519)
    exact FastRows.row5519_checked.symm
  rw [row]
  decide
theorem root541Step7 : scan edge (search edge 3) 4 [3923] 3923 root541Tail7 winner =
    scan edge (search edge 3) 4 [3923] 3923 root541Tail8 winner := by
  rw [root541Tail7, scan, if_neg (by decide), root541Candidates7_checked]
  change scan edge (search edge 3) 4 [3923] 3923 root541Tail8
    (if 3 ≤ root541Candidates7.length then search edge 3 [3923, 5519] root541Candidates7 9442 winner else winner) = _
  rw [if_neg (by decide)]
theorem root541Scan8 : scan edge (search edge 3) 4 [3923] 3923 root541Tail8 winner = winner := by rw [root541Tail8, scan, if_pos (by decide)]
theorem root541Scan7 : scan edge (search edge 3) 4 [3923] 3923 root541Tail7 winner = winner :=
  (root541Step7).trans root541Scan8
theorem root541Scan6 : scan edge (search edge 3) 4 [3923] 3923 root541Tail6 winner = winner :=
  (root541Step6).trans root541Scan7
theorem root541Scan5 : scan edge (search edge 3) 4 [3923] 3923 root541Tail5 winner = winner :=
  (root541Step5).trans root541Scan6
theorem root541Scan4 : scan edge (search edge 3) 4 [3923] 3923 root541Tail4 winner = winner :=
  (root541Step4).trans root541Scan5
theorem root541Scan3 : scan edge (search edge 3) 4 [3923] 3923 root541Tail3 winner = winner :=
  (root541Step3).trans root541Scan4
theorem root541Scan2 : scan edge (search edge 3) 4 [3923] 3923 root541Tail2 winner = winner :=
  (root541Step2).trans root541Scan3
theorem root541Scan1 : scan edge (search edge 3) 4 [3923] 3923 root541Tail1 winner = winner :=
  (root541Step1).trans root541Scan2
theorem root541Scan0 : scan edge (search edge 3) 4 [3923] 3923 root541Tail0 winner = winner :=
  (root541Step0).trans root541Scan1
set_option maxRecDepth 4096 in
theorem root541Branch : search edge 4 [3923] root541Candidates 3923 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3923] 3923 root541Tail0 winner = winner
  exact root541Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
