import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root439Candidates : List Nat := Data.row3089
set_option maxRecDepth 8192 in
theorem root439Candidates_checked : candidates edge 5 0 3089 tail440 winner.bound = root439Candidates := by
  apply candidates_eq_row edge 5 0 3089 tail440 root439Candidates winner.bound tail440_ordered
  · decide
  · decide
  · funext q
    rfl
def root439Tail30 : List Nat := []
def root439Tail29 : List Nat := 22619 :: root439Tail30
def root439Tail28 : List Nat := 22511 :: root439Tail29
def root439Tail27 : List Nat := 22433 :: root439Tail28
def root439Tail26 : List Nat := 22091 :: root439Tail27
def root439Tail25 : List Nat := 21611 :: root439Tail26
def root439Tail24 : List Nat := 20327 :: root439Tail25
def root439Tail23 : List Nat := 19973 :: root439Tail24
def root439Tail22 : List Nat := 18803 :: root439Tail23
def root439Tail21 : List Nat := 18341 :: root439Tail22
def root439Tail20 : List Nat := 18257 :: root439Tail21
def root439Tail19 : List Nat := 16889 :: root439Tail20
def root439Tail18 : List Nat := 15287 :: root439Tail19
def root439Tail17 : List Nat := 14741 :: root439Tail18
def root439Tail16 : List Nat := 11789 :: root439Tail17
def root439Tail15 : List Nat := 11579 :: root439Tail16
def root439Tail14 : List Nat := 11279 :: root439Tail15
def root439Tail13 : List Nat := 10169 :: root439Tail14
def root439Tail12 : List Nat := 9629 :: root439Tail13
def root439Tail11 : List Nat := 9203 :: root439Tail12
def root439Tail10 : List Nat := 8093 :: root439Tail11
def root439Tail9 : List Nat := 7541 :: root439Tail10
def root439Tail8 : List Nat := 7457 :: root439Tail9
def root439Tail7 : List Nat := 6689 :: root439Tail8
def root439Tail6 : List Nat := 5867 :: root439Tail7
def root439Tail5 : List Nat := 5639 :: root439Tail6
def root439Tail4 : List Nat := 5303 :: root439Tail5
def root439Tail3 : List Nat := 5147 :: root439Tail4
def root439Tail2 : List Nat := 3917 :: root439Tail3
def root439Tail1 : List Nat := 3851 :: root439Tail2
def root439Tail0 : List Nat := 3299 :: root439Tail1
def root439Candidates0 : List Nat := [9203]
set_option maxRecDepth 8192 in
theorem root439Candidates0_checked : candidates edge 4 3089 3299 root439Tail1 winner.bound = root439Candidates0 := by
  unfold candidates
  have row : edge 3299 = (SortedNatTree.contains FastRows.row3299) := by
    change Data.row3299.contains = (SortedNatTree.contains FastRows.row3299)
    exact FastRows.row3299_checked.symm
  rw [row]
  decide
theorem root439Step0 : scan edge (search edge 3) 4 [3089] 3089 root439Tail0 winner =
    scan edge (search edge 3) 4 [3089] 3089 root439Tail1 winner := by
  rw [root439Tail0, scan, if_neg (by decide), root439Candidates0_checked]
  change scan edge (search edge 3) 4 [3089] 3089 root439Tail1
    (if 3 ≤ root439Candidates0.length then search edge 3 [3089, 3299] root439Candidates0 6388 winner else winner) = _
  rw [if_neg (by decide)]
def root439Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root439Candidates1_checked : candidates edge 4 3089 3851 root439Tail2 winner.bound = root439Candidates1 := by
  unfold candidates
  have row : edge 3851 = (SortedNatTree.contains FastRows.row3851) := by
    change Data.row3851.contains = (SortedNatTree.contains FastRows.row3851)
    exact FastRows.row3851_checked.symm
  rw [row]
  decide
theorem root439Step1 : scan edge (search edge 3) 4 [3089] 3089 root439Tail1 winner =
    scan edge (search edge 3) 4 [3089] 3089 root439Tail2 winner := by
  rw [root439Tail1, scan, if_neg (by decide), root439Candidates1_checked]
  change scan edge (search edge 3) 4 [3089] 3089 root439Tail2
    (if 3 ≤ root439Candidates1.length then search edge 3 [3089, 3851] root439Candidates1 6940 winner else winner) = _
  rw [if_neg (by decide)]
def root439Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root439Candidates2_checked : candidates edge 4 3089 3917 root439Tail3 winner.bound = root439Candidates2 := by
  unfold candidates
  have row : edge 3917 = (SortedNatTree.contains FastRows.row3917) := by
    change Data.row3917.contains = (SortedNatTree.contains FastRows.row3917)
    exact FastRows.row3917_checked.symm
  rw [row]
  decide
theorem root439Step2 : scan edge (search edge 3) 4 [3089] 3089 root439Tail2 winner =
    scan edge (search edge 3) 4 [3089] 3089 root439Tail3 winner := by
  rw [root439Tail2, scan, if_neg (by decide), root439Candidates2_checked]
  change scan edge (search edge 3) 4 [3089] 3089 root439Tail3
    (if 3 ≤ root439Candidates2.length then search edge 3 [3089, 3917] root439Candidates2 7006 winner else winner) = _
  rw [if_neg (by decide)]
def root439Candidates3 : List Nat := [9203]
set_option maxRecDepth 8192 in
theorem root439Candidates3_checked : candidates edge 4 3089 5147 root439Tail4 winner.bound = root439Candidates3 := by
  unfold candidates
  have row : edge 5147 = (SortedNatTree.contains FastRows.row5147) := by
    change Data.row5147.contains = (SortedNatTree.contains FastRows.row5147)
    exact FastRows.row5147_checked.symm
  rw [row]
  decide
theorem root439Step3 : scan edge (search edge 3) 4 [3089] 3089 root439Tail3 winner =
    scan edge (search edge 3) 4 [3089] 3089 root439Tail4 winner := by
  rw [root439Tail3, scan, if_neg (by decide), root439Candidates3_checked]
  change scan edge (search edge 3) 4 [3089] 3089 root439Tail4
    (if 3 ≤ root439Candidates3.length then search edge 3 [3089, 5147] root439Candidates3 8236 winner else winner) = _
  rw [if_neg (by decide)]
def root439Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root439Candidates4_checked : candidates edge 4 3089 5303 root439Tail5 winner.bound = root439Candidates4 := by
  unfold candidates
  have row : edge 5303 = (SortedNatTree.contains FastRows.row5303) := by
    change Data.row5303.contains = (SortedNatTree.contains FastRows.row5303)
    exact FastRows.row5303_checked.symm
  rw [row]
  decide
theorem root439Step4 : scan edge (search edge 3) 4 [3089] 3089 root439Tail4 winner =
    scan edge (search edge 3) 4 [3089] 3089 root439Tail5 winner := by
  rw [root439Tail4, scan, if_neg (by decide), root439Candidates4_checked]
  change scan edge (search edge 3) 4 [3089] 3089 root439Tail5
    (if 3 ≤ root439Candidates4.length then search edge 3 [3089, 5303] root439Candidates4 8392 winner else winner) = _
  rw [if_neg (by decide)]
def root439Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root439Candidates5_checked : candidates edge 4 3089 5639 root439Tail6 winner.bound = root439Candidates5 := by
  unfold candidates
  have row : edge 5639 = (SortedNatTree.contains FastRows.row5639) := by
    change Data.row5639.contains = (SortedNatTree.contains FastRows.row5639)
    exact FastRows.row5639_checked.symm
  rw [row]
  decide
theorem root439Step5 : scan edge (search edge 3) 4 [3089] 3089 root439Tail5 winner =
    scan edge (search edge 3) 4 [3089] 3089 root439Tail6 winner := by
  rw [root439Tail5, scan, if_neg (by decide), root439Candidates5_checked]
  change scan edge (search edge 3) 4 [3089] 3089 root439Tail6
    (if 3 ≤ root439Candidates5.length then search edge 3 [3089, 5639] root439Candidates5 8728 winner else winner) = _
  rw [if_neg (by decide)]
theorem root439Scan6 : scan edge (search edge 3) 4 [3089] 3089 root439Tail6 winner = winner := by rw [root439Tail6, scan, if_pos (by decide)]
theorem root439Scan5 : scan edge (search edge 3) 4 [3089] 3089 root439Tail5 winner = winner :=
  (root439Step5).trans root439Scan6
theorem root439Scan4 : scan edge (search edge 3) 4 [3089] 3089 root439Tail4 winner = winner :=
  (root439Step4).trans root439Scan5
theorem root439Scan3 : scan edge (search edge 3) 4 [3089] 3089 root439Tail3 winner = winner :=
  (root439Step3).trans root439Scan4
theorem root439Scan2 : scan edge (search edge 3) 4 [3089] 3089 root439Tail2 winner = winner :=
  (root439Step2).trans root439Scan3
theorem root439Scan1 : scan edge (search edge 3) 4 [3089] 3089 root439Tail1 winner = winner :=
  (root439Step1).trans root439Scan2
theorem root439Scan0 : scan edge (search edge 3) 4 [3089] 3089 root439Tail0 winner = winner :=
  (root439Step0).trans root439Scan1
set_option maxRecDepth 4096 in
theorem root439Branch : search edge 4 [3089] root439Candidates 3089 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3089] 3089 root439Tail0 winner = winner
  exact root439Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
