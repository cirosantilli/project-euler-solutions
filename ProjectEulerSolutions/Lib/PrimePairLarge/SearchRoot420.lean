import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root420Candidates : List Nat := Data.row2927
set_option maxRecDepth 8192 in
theorem root420Candidates_checked : candidates edge 5 0 2927 tail421 winner.bound = root420Candidates := by
  apply candidates_eq_row edge 5 0 2927 tail421 root420Candidates winner.bound tail421_ordered
  · decide
  · decide
  · funext q
    rfl
def root420Tail32 : List Nat := []
def root420Tail31 : List Nat := 23027 :: root420Tail32
def root420Tail30 : List Nat := 22751 :: root420Tail31
def root420Tail29 : List Nat := 22709 :: root420Tail30
def root420Tail28 : List Nat := 22343 :: root420Tail29
def root420Tail27 : List Nat := 21557 :: root420Tail28
def root420Tail26 : List Nat := 20849 :: root420Tail27
def root420Tail25 : List Nat := 18749 :: root420Tail26
def root420Tail24 : List Nat := 18131 :: root420Tail25
def root420Tail23 : List Nat := 15263 :: root420Tail24
def root420Tail22 : List Nat := 13457 :: root420Tail23
def root420Tail21 : List Nat := 13103 :: root420Tail22
def root420Tail20 : List Nat := 12611 :: root420Tail21
def root420Tail19 : List Nat := 11987 :: root420Tail20
def root420Tail18 : List Nat := 11519 :: root420Tail19
def root420Tail17 : List Nat := 11057 :: root420Tail18
def root420Tail16 : List Nat := 10853 :: root420Tail17
def root420Tail15 : List Nat := 10037 :: root420Tail16
def root420Tail14 : List Nat := 9227 :: root420Tail15
def root420Tail13 : List Nat := 9011 :: root420Tail14
def root420Tail12 : List Nat := 8999 :: root420Tail13
def root420Tail11 : List Nat := 8243 :: root420Tail12
def root420Tail10 : List Nat := 8009 :: root420Tail11
def root420Tail9 : List Nat := 6983 :: root420Tail10
def root420Tail8 : List Nat := 6257 :: root420Tail9
def root420Tail7 : List Nat := 6029 :: root420Tail8
def root420Tail6 : List Nat := 5693 :: root420Tail7
def root420Tail5 : List Nat := 5531 :: root420Tail6
def root420Tail4 : List Nat := 5009 :: root420Tail5
def root420Tail3 : List Nat := 3863 :: root420Tail4
def root420Tail2 : List Nat := 3593 :: root420Tail3
def root420Tail1 : List Nat := 3203 :: root420Tail2
def root420Tail0 : List Nat := 3137 :: root420Tail1
def root420Candidates0 : List Nat := [3593, 5009]
set_option maxRecDepth 8192 in
theorem root420Candidates0_checked : candidates edge 4 2927 3137 root420Tail1 winner.bound = root420Candidates0 := by
  unfold candidates
  have row : edge 3137 = (SortedNatTree.contains FastRows.row3137) := by
    change Data.row3137.contains = (SortedNatTree.contains FastRows.row3137)
    exact FastRows.row3137_checked.symm
  rw [row]
  decide
theorem root420Step0 : scan edge (search edge 3) 4 [2927] 2927 root420Tail0 winner =
    scan edge (search edge 3) 4 [2927] 2927 root420Tail1 winner := by
  rw [root420Tail0, scan, if_neg (by decide), root420Candidates0_checked]
  change scan edge (search edge 3) 4 [2927] 2927 root420Tail1
    (if 3 ≤ root420Candidates0.length then search edge 3 [2927, 3137] root420Candidates0 6064 winner else winner) = _
  rw [if_neg (by decide)]
def root420Candidates1 : List Nat := [15263, 18749]
set_option maxRecDepth 8192 in
theorem root420Candidates1_checked : candidates edge 4 2927 3203 root420Tail2 winner.bound = root420Candidates1 := by
  unfold candidates
  have row : edge 3203 = (SortedNatTree.contains FastRows.row3203) := by
    change Data.row3203.contains = (SortedNatTree.contains FastRows.row3203)
    exact FastRows.row3203_checked.symm
  rw [row]
  decide
theorem root420Step1 : scan edge (search edge 3) 4 [2927] 2927 root420Tail1 winner =
    scan edge (search edge 3) 4 [2927] 2927 root420Tail2 winner := by
  rw [root420Tail1, scan, if_neg (by decide), root420Candidates1_checked]
  change scan edge (search edge 3) 4 [2927] 2927 root420Tail2
    (if 3 ≤ root420Candidates1.length then search edge 3 [2927, 3203] root420Candidates1 6130 winner else winner) = _
  rw [if_neg (by decide)]
def root420Candidates2 : List Nat := [5693, 18131]
set_option maxRecDepth 8192 in
theorem root420Candidates2_checked : candidates edge 4 2927 3593 root420Tail3 winner.bound = root420Candidates2 := by
  unfold candidates
  have row : edge 3593 = (SortedNatTree.contains FastRows.row3593) := by
    change Data.row3593.contains = (SortedNatTree.contains FastRows.row3593)
    exact FastRows.row3593_checked.symm
  rw [row]
  decide
theorem root420Step2 : scan edge (search edge 3) 4 [2927] 2927 root420Tail2 winner =
    scan edge (search edge 3) 4 [2927] 2927 root420Tail3 winner := by
  rw [root420Tail2, scan, if_neg (by decide), root420Candidates2_checked]
  change scan edge (search edge 3) 4 [2927] 2927 root420Tail3
    (if 3 ≤ root420Candidates2.length then search edge 3 [2927, 3593] root420Candidates2 6520 winner else winner) = _
  rw [if_neg (by decide)]
def root420Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root420Candidates3_checked : candidates edge 4 2927 3863 root420Tail4 winner.bound = root420Candidates3 := by
  unfold candidates
  have row : edge 3863 = (SortedNatTree.contains FastRows.row3863) := by
    change Data.row3863.contains = (SortedNatTree.contains FastRows.row3863)
    exact FastRows.row3863_checked.symm
  rw [row]
  decide
theorem root420Step3 : scan edge (search edge 3) 4 [2927] 2927 root420Tail3 winner =
    scan edge (search edge 3) 4 [2927] 2927 root420Tail4 winner := by
  rw [root420Tail3, scan, if_neg (by decide), root420Candidates3_checked]
  change scan edge (search edge 3) 4 [2927] 2927 root420Tail4
    (if 3 ≤ root420Candidates3.length then search edge 3 [2927, 3863] root420Candidates3 6790 winner else winner) = _
  rw [if_neg (by decide)]
def root420Candidates4 : List Nat := [6029, 11057]
set_option maxRecDepth 8192 in
theorem root420Candidates4_checked : candidates edge 4 2927 5009 root420Tail5 winner.bound = root420Candidates4 := by
  unfold candidates
  have row : edge 5009 = (SortedNatTree.contains FastRows.row5009) := by
    change Data.row5009.contains = (SortedNatTree.contains FastRows.row5009)
    exact FastRows.row5009_checked.symm
  rw [row]
  decide
theorem root420Step4 : scan edge (search edge 3) 4 [2927] 2927 root420Tail4 winner =
    scan edge (search edge 3) 4 [2927] 2927 root420Tail5 winner := by
  rw [root420Tail4, scan, if_neg (by decide), root420Candidates4_checked]
  change scan edge (search edge 3) 4 [2927] 2927 root420Tail5
    (if 3 ≤ root420Candidates4.length then search edge 3 [2927, 5009] root420Candidates4 7936 winner else winner) = _
  rw [if_neg (by decide)]
def root420Candidates5 : List Nat := [8009]
set_option maxRecDepth 8192 in
theorem root420Candidates5_checked : candidates edge 4 2927 5531 root420Tail6 winner.bound = root420Candidates5 := by
  unfold candidates
  have row : edge 5531 = (SortedNatTree.contains FastRows.row5531) := by
    change Data.row5531.contains = (SortedNatTree.contains FastRows.row5531)
    exact FastRows.row5531_checked.symm
  rw [row]
  decide
theorem root420Step5 : scan edge (search edge 3) 4 [2927] 2927 root420Tail5 winner =
    scan edge (search edge 3) 4 [2927] 2927 root420Tail6 winner := by
  rw [root420Tail5, scan, if_neg (by decide), root420Candidates5_checked]
  change scan edge (search edge 3) 4 [2927] 2927 root420Tail6
    (if 3 ≤ root420Candidates5.length then search edge 3 [2927, 5531] root420Candidates5 8458 winner else winner) = _
  rw [if_neg (by decide)]
def root420Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root420Candidates6_checked : candidates edge 4 2927 5693 root420Tail7 winner.bound = root420Candidates6 := by
  unfold candidates
  have row : edge 5693 = (SortedNatTree.contains FastRows.row5693) := by
    change Data.row5693.contains = (SortedNatTree.contains FastRows.row5693)
    exact FastRows.row5693_checked.symm
  rw [row]
  decide
theorem root420Step6 : scan edge (search edge 3) 4 [2927] 2927 root420Tail6 winner =
    scan edge (search edge 3) 4 [2927] 2927 root420Tail7 winner := by
  rw [root420Tail6, scan, if_neg (by decide), root420Candidates6_checked]
  change scan edge (search edge 3) 4 [2927] 2927 root420Tail7
    (if 3 ≤ root420Candidates6.length then search edge 3 [2927, 5693] root420Candidates6 8620 winner else winner) = _
  rw [if_neg (by decide)]
theorem root420Scan7 : scan edge (search edge 3) 4 [2927] 2927 root420Tail7 winner = winner := by rw [root420Tail7, scan, if_pos (by decide)]
theorem root420Scan6 : scan edge (search edge 3) 4 [2927] 2927 root420Tail6 winner = winner :=
  (root420Step6).trans root420Scan7
theorem root420Scan5 : scan edge (search edge 3) 4 [2927] 2927 root420Tail5 winner = winner :=
  (root420Step5).trans root420Scan6
theorem root420Scan4 : scan edge (search edge 3) 4 [2927] 2927 root420Tail4 winner = winner :=
  (root420Step4).trans root420Scan5
theorem root420Scan3 : scan edge (search edge 3) 4 [2927] 2927 root420Tail3 winner = winner :=
  (root420Step3).trans root420Scan4
theorem root420Scan2 : scan edge (search edge 3) 4 [2927] 2927 root420Tail2 winner = winner :=
  (root420Step2).trans root420Scan3
theorem root420Scan1 : scan edge (search edge 3) 4 [2927] 2927 root420Tail1 winner = winner :=
  (root420Step1).trans root420Scan2
theorem root420Scan0 : scan edge (search edge 3) 4 [2927] 2927 root420Tail0 winner = winner :=
  (root420Step0).trans root420Scan1
set_option maxRecDepth 4096 in
theorem root420Branch : search edge 4 [2927] root420Candidates 2927 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2927] 2927 root420Tail0 winner = winner
  exact root420Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
