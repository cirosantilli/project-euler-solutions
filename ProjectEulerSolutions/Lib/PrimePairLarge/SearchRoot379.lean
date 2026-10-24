import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root379Candidates : List Nat := Data.row2633
set_option maxRecDepth 8192 in
theorem root379Candidates_checked : candidates edge 5 0 2633 tail380 winner.bound = root379Candidates := by
  apply candidates_eq_row edge 5 0 2633 tail380 root379Candidates winner.bound tail380_ordered
  · decide
  · decide
  · funext q
    rfl
def root379Tail38 : List Nat := []
def root379Tail37 : List Nat := 23003 :: root379Tail38
def root379Tail36 : List Nat := 21713 :: root379Tail37
def root379Tail35 : List Nat := 20147 :: root379Tail36
def root379Tail34 : List Nat := 20051 :: root379Tail35
def root379Tail33 : List Nat := 19949 :: root379Tail34
def root379Tail32 : List Nat := 19751 :: root379Tail33
def root379Tail31 : List Nat := 17027 :: root379Tail32
def root379Tail30 : List Nat := 16631 :: root379Tail31
def root379Tail29 : List Nat := 16253 :: root379Tail30
def root379Tail28 : List Nat := 16223 :: root379Tail29
def root379Tail27 : List Nat := 15959 :: root379Tail28
def root379Tail26 : List Nat := 14771 :: root379Tail27
def root379Tail25 : List Nat := 13859 :: root379Tail26
def root379Tail24 : List Nat := 13763 :: root379Tail25
def root379Tail23 : List Nat := 13679 :: root379Tail24
def root379Tail22 : List Nat := 13577 :: root379Tail23
def root379Tail21 : List Nat := 13469 :: root379Tail22
def root379Tail20 : List Nat := 13313 :: root379Tail21
def root379Tail19 : List Nat := 12149 :: root379Tail20
def root379Tail18 : List Nat := 11549 :: root379Tail19
def root379Tail17 : List Nat := 11423 :: root379Tail18
def root379Tail16 : List Nat := 10847 :: root379Tail17
def root379Tail15 : List Nat := 10613 :: root379Tail16
def root379Tail14 : List Nat := 8999 :: root379Tail15
def root379Tail13 : List Nat := 8369 :: root379Tail14
def root379Tail12 : List Nat := 7547 :: root379Tail13
def root379Tail11 : List Nat := 7517 :: root379Tail12
def root379Tail10 : List Nat := 7193 :: root379Tail11
def root379Tail9 : List Nat := 7079 :: root379Tail10
def root379Tail8 : List Nat := 6581 :: root379Tail9
def root379Tail7 : List Nat := 5867 :: root379Tail8
def root379Tail6 : List Nat := 5807 :: root379Tail7
def root379Tail5 : List Nat := 5693 :: root379Tail6
def root379Tail4 : List Nat := 5147 :: root379Tail5
def root379Tail3 : List Nat := 5051 :: root379Tail4
def root379Tail2 : List Nat := 4337 :: root379Tail3
def root379Tail1 : List Nat := 3929 :: root379Tail2
def root379Tail0 : List Nat := 2699 :: root379Tail1
def root379Candidates0 : List Nat := [4337]
set_option maxRecDepth 8192 in
theorem root379Candidates0_checked : candidates edge 4 2633 2699 root379Tail1 winner.bound = root379Candidates0 := by
  unfold candidates
  have row : edge 2699 = (SortedNatTree.contains FastRows.row2699) := by
    change Data.row2699.contains = (SortedNatTree.contains FastRows.row2699)
    exact FastRows.row2699_checked.symm
  rw [row]
  decide
theorem root379Step0 : scan edge (search edge 3) 4 [2633] 2633 root379Tail0 winner =
    scan edge (search edge 3) 4 [2633] 2633 root379Tail1 winner := by
  rw [root379Tail0, scan, if_neg (by decide), root379Candidates0_checked]
  change scan edge (search edge 3) 4 [2633] 2633 root379Tail1
    (if 3 ≤ root379Candidates0.length then search edge 3 [2633, 2699] root379Candidates0 5332 winner else winner) = _
  rw [if_neg (by decide)]
def root379Candidates1 : List Nat := [11549]
set_option maxRecDepth 8192 in
theorem root379Candidates1_checked : candidates edge 4 2633 3929 root379Tail2 winner.bound = root379Candidates1 := by
  unfold candidates
  have row : edge 3929 = (SortedNatTree.contains FastRows.row3929) := by
    change Data.row3929.contains = (SortedNatTree.contains FastRows.row3929)
    exact FastRows.row3929_checked.symm
  rw [row]
  decide
theorem root379Step1 : scan edge (search edge 3) 4 [2633] 2633 root379Tail1 winner =
    scan edge (search edge 3) 4 [2633] 2633 root379Tail2 winner := by
  rw [root379Tail1, scan, if_neg (by decide), root379Candidates1_checked]
  change scan edge (search edge 3) 4 [2633] 2633 root379Tail2
    (if 3 ≤ root379Candidates1.length then search edge 3 [2633, 3929] root379Candidates1 6562 winner else winner) = _
  rw [if_neg (by decide)]
def root379Candidates2 : List Nat := [7547]
set_option maxRecDepth 8192 in
theorem root379Candidates2_checked : candidates edge 4 2633 4337 root379Tail3 winner.bound = root379Candidates2 := by
  unfold candidates
  have row : edge 4337 = (SortedNatTree.contains FastRows.row4337) := by
    change Data.row4337.contains = (SortedNatTree.contains FastRows.row4337)
    exact FastRows.row4337_checked.symm
  rw [row]
  decide
theorem root379Step2 : scan edge (search edge 3) 4 [2633] 2633 root379Tail2 winner =
    scan edge (search edge 3) 4 [2633] 2633 root379Tail3 winner := by
  rw [root379Tail2, scan, if_neg (by decide), root379Candidates2_checked]
  change scan edge (search edge 3) 4 [2633] 2633 root379Tail3
    (if 3 ≤ root379Candidates2.length then search edge 3 [2633, 4337] root379Candidates2 6970 winner else winner) = _
  rw [if_neg (by decide)]
def root379Candidates3 : List Nat := [5867]
set_option maxRecDepth 8192 in
theorem root379Candidates3_checked : candidates edge 4 2633 5051 root379Tail4 winner.bound = root379Candidates3 := by
  unfold candidates
  have row : edge 5051 = (SortedNatTree.contains FastRows.row5051) := by
    change Data.row5051.contains = (SortedNatTree.contains FastRows.row5051)
    exact FastRows.row5051_checked.symm
  rw [row]
  decide
theorem root379Step3 : scan edge (search edge 3) 4 [2633] 2633 root379Tail3 winner =
    scan edge (search edge 3) 4 [2633] 2633 root379Tail4 winner := by
  rw [root379Tail3, scan, if_neg (by decide), root379Candidates3_checked]
  change scan edge (search edge 3) 4 [2633] 2633 root379Tail4
    (if 3 ≤ root379Candidates3.length then search edge 3 [2633, 5051] root379Candidates3 7684 winner else winner) = _
  rw [if_neg (by decide)]
def root379Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root379Candidates4_checked : candidates edge 4 2633 5147 root379Tail5 winner.bound = root379Candidates4 := by
  unfold candidates
  have row : edge 5147 = (SortedNatTree.contains FastRows.row5147) := by
    change Data.row5147.contains = (SortedNatTree.contains FastRows.row5147)
    exact FastRows.row5147_checked.symm
  rw [row]
  decide
theorem root379Step4 : scan edge (search edge 3) 4 [2633] 2633 root379Tail4 winner =
    scan edge (search edge 3) 4 [2633] 2633 root379Tail5 winner := by
  rw [root379Tail4, scan, if_neg (by decide), root379Candidates4_checked]
  change scan edge (search edge 3) 4 [2633] 2633 root379Tail5
    (if 3 ≤ root379Candidates4.length then search edge 3 [2633, 5147] root379Candidates4 7780 winner else winner) = _
  rw [if_neg (by decide)]
def root379Candidates5 : List Nat := [14771]
set_option maxRecDepth 8192 in
theorem root379Candidates5_checked : candidates edge 4 2633 5693 root379Tail6 winner.bound = root379Candidates5 := by
  unfold candidates
  have row : edge 5693 = (SortedNatTree.contains FastRows.row5693) := by
    change Data.row5693.contains = (SortedNatTree.contains FastRows.row5693)
    exact FastRows.row5693_checked.symm
  rw [row]
  decide
theorem root379Step5 : scan edge (search edge 3) 4 [2633] 2633 root379Tail5 winner =
    scan edge (search edge 3) 4 [2633] 2633 root379Tail6 winner := by
  rw [root379Tail5, scan, if_neg (by decide), root379Candidates5_checked]
  change scan edge (search edge 3) 4 [2633] 2633 root379Tail6
    (if 3 ≤ root379Candidates5.length then search edge 3 [2633, 5693] root379Candidates5 8326 winner else winner) = _
  rw [if_neg (by decide)]
def root379Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root379Candidates6_checked : candidates edge 4 2633 5807 root379Tail7 winner.bound = root379Candidates6 := by
  unfold candidates
  have row : edge 5807 = (SortedNatTree.contains FastRows.row5807) := by
    change Data.row5807.contains = (SortedNatTree.contains FastRows.row5807)
    exact FastRows.row5807_checked.symm
  rw [row]
  decide
theorem root379Step6 : scan edge (search edge 3) 4 [2633] 2633 root379Tail6 winner =
    scan edge (search edge 3) 4 [2633] 2633 root379Tail7 winner := by
  rw [root379Tail6, scan, if_neg (by decide), root379Candidates6_checked]
  change scan edge (search edge 3) 4 [2633] 2633 root379Tail7
    (if 3 ≤ root379Candidates6.length then search edge 3 [2633, 5807] root379Candidates6 8440 winner else winner) = _
  rw [if_neg (by decide)]
theorem root379Scan7 : scan edge (search edge 3) 4 [2633] 2633 root379Tail7 winner = winner := by rw [root379Tail7, scan, if_pos (by decide)]
theorem root379Scan6 : scan edge (search edge 3) 4 [2633] 2633 root379Tail6 winner = winner :=
  (root379Step6).trans root379Scan7
theorem root379Scan5 : scan edge (search edge 3) 4 [2633] 2633 root379Tail5 winner = winner :=
  (root379Step5).trans root379Scan6
theorem root379Scan4 : scan edge (search edge 3) 4 [2633] 2633 root379Tail4 winner = winner :=
  (root379Step4).trans root379Scan5
theorem root379Scan3 : scan edge (search edge 3) 4 [2633] 2633 root379Tail3 winner = winner :=
  (root379Step3).trans root379Scan4
theorem root379Scan2 : scan edge (search edge 3) 4 [2633] 2633 root379Tail2 winner = winner :=
  (root379Step2).trans root379Scan3
theorem root379Scan1 : scan edge (search edge 3) 4 [2633] 2633 root379Tail1 winner = winner :=
  (root379Step1).trans root379Scan2
theorem root379Scan0 : scan edge (search edge 3) 4 [2633] 2633 root379Tail0 winner = winner :=
  (root379Step0).trans root379Scan1
set_option maxRecDepth 4096 in
theorem root379Branch : search edge 4 [2633] root379Candidates 2633 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2633] 2633 root379Tail0 winner = winner
  exact root379Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
