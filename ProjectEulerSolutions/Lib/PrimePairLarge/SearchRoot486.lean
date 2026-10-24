import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root486Candidates : List Nat := Data.row3499
set_option maxRecDepth 8192 in
theorem root486Candidates_checked : candidates edge 5 0 3499 tail487 winner.bound = root486Candidates := by
  apply candidates_eq_row edge 5 0 3499 tail487 root486Candidates winner.bound tail487_ordered
  · decide
  · decide
  · funext q
    rfl
def root486Tail45 : List Nat := []
def root486Tail44 : List Nat := 22453 :: root486Tail45
def root486Tail43 : List Nat := 20521 :: root486Tail44
def root486Tail42 : List Nat := 20101 :: root486Tail43
def root486Tail41 : List Nat := 20047 :: root486Tail42
def root486Tail40 : List Nat := 19387 :: root486Tail41
def root486Tail39 : List Nat := 19081 :: root486Tail40
def root486Tail38 : List Nat := 18691 :: root486Tail39
def root486Tail37 : List Nat := 18199 :: root486Tail38
def root486Tail36 : List Nat := 17977 :: root486Tail37
def root486Tail35 : List Nat := 16381 :: root486Tail36
def root486Tail34 : List Nat := 15739 :: root486Tail35
def root486Tail33 : List Nat := 15331 :: root486Tail34
def root486Tail32 : List Nat := 15277 :: root486Tail33
def root486Tail31 : List Nat := 15217 :: root486Tail32
def root486Tail30 : List Nat := 14797 :: root486Tail31
def root486Tail29 : List Nat := 14713 :: root486Tail30
def root486Tail28 : List Nat := 14437 :: root486Tail29
def root486Tail27 : List Nat := 14419 :: root486Tail28
def root486Tail26 : List Nat := 14293 :: root486Tail27
def root486Tail25 : List Nat := 13339 :: root486Tail26
def root486Tail24 : List Nat := 12757 :: root486Tail25
def root486Tail23 : List Nat := 12487 :: root486Tail24
def root486Tail22 : List Nat := 12373 :: root486Tail23
def root486Tail21 : List Nat := 11437 :: root486Tail22
def root486Tail20 : List Nat := 10333 :: root486Tail21
def root486Tail19 : List Nat := 9931 :: root486Tail20
def root486Tail18 : List Nat := 9871 :: root486Tail19
def root486Tail17 : List Nat := 9619 :: root486Tail18
def root486Tail16 : List Nat := 9343 :: root486Tail17
def root486Tail15 : List Nat := 9157 :: root486Tail16
def root486Tail14 : List Nat := 8599 :: root486Tail15
def root486Tail13 : List Nat := 8377 :: root486Tail14
def root486Tail12 : List Nat := 8191 :: root486Tail13
def root486Tail11 : List Nat := 7993 :: root486Tail12
def root486Tail10 : List Nat := 7759 :: root486Tail11
def root486Tail9 : List Nat := 7321 :: root486Tail10
def root486Tail8 : List Nat := 6361 :: root486Tail9
def root486Tail7 : List Nat := 6277 :: root486Tail8
def root486Tail6 : List Nat := 5881 :: root486Tail7
def root486Tail5 : List Nat := 5449 :: root486Tail6
def root486Tail4 : List Nat := 5407 :: root486Tail5
def root486Tail3 : List Nat := 5227 :: root486Tail4
def root486Tail2 : List Nat := 5011 :: root486Tail3
def root486Tail1 : List Nat := 4651 :: root486Tail2
def root486Tail0 : List Nat := 3613 :: root486Tail1
def root486Candidates0 : List Nat := [14293]
set_option maxRecDepth 8192 in
theorem root486Candidates0_checked : candidates edge 4 3499 3613 root486Tail1 winner.bound = root486Candidates0 := by
  unfold candidates
  have row : edge 3613 = (SortedNatTree.contains FastRows.row3613) := by
    change Data.row3613.contains = (SortedNatTree.contains FastRows.row3613)
    exact FastRows.row3613_checked.symm
  rw [row]
  decide
theorem root486Step0 : scan edge (search edge 3) 4 [3499] 3499 root486Tail0 winner =
    scan edge (search edge 3) 4 [3499] 3499 root486Tail1 winner := by
  rw [root486Tail0, scan, if_neg (by decide), root486Candidates0_checked]
  change scan edge (search edge 3) 4 [3499] 3499 root486Tail1
    (if 3 ≤ root486Candidates0.length then search edge 3 [3499, 3613] root486Candidates0 7112 winner else winner) = _
  rw [if_neg (by decide)]
def root486Candidates1 : List Nat := [14437]
set_option maxRecDepth 8192 in
theorem root486Candidates1_checked : candidates edge 4 3499 4651 root486Tail2 winner.bound = root486Candidates1 := by
  unfold candidates
  have row : edge 4651 = (SortedNatTree.contains FastRows.row4651) := by
    change Data.row4651.contains = (SortedNatTree.contains FastRows.row4651)
    exact FastRows.row4651_checked.symm
  rw [row]
  decide
theorem root486Step1 : scan edge (search edge 3) 4 [3499] 3499 root486Tail1 winner =
    scan edge (search edge 3) 4 [3499] 3499 root486Tail2 winner := by
  rw [root486Tail1, scan, if_neg (by decide), root486Candidates1_checked]
  change scan edge (search edge 3) 4 [3499] 3499 root486Tail2
    (if 3 ≤ root486Candidates1.length then search edge 3 [3499, 4651] root486Candidates1 8150 winner else winner) = _
  rw [if_neg (by decide)]
def root486Candidates2 : List Nat := [8191]
set_option maxRecDepth 8192 in
theorem root486Candidates2_checked : candidates edge 4 3499 5011 root486Tail3 winner.bound = root486Candidates2 := by
  unfold candidates
  have row : edge 5011 = (SortedNatTree.contains FastRows.row5011) := by
    change Data.row5011.contains = (SortedNatTree.contains FastRows.row5011)
    exact FastRows.row5011_checked.symm
  rw [row]
  decide
theorem root486Step2 : scan edge (search edge 3) 4 [3499] 3499 root486Tail2 winner =
    scan edge (search edge 3) 4 [3499] 3499 root486Tail3 winner := by
  rw [root486Tail2, scan, if_neg (by decide), root486Candidates2_checked]
  change scan edge (search edge 3) 4 [3499] 3499 root486Tail3
    (if 3 ≤ root486Candidates2.length then search edge 3 [3499, 5011] root486Candidates2 8510 winner else winner) = _
  rw [if_neg (by decide)]
def root486Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root486Candidates3_checked : candidates edge 4 3499 5227 root486Tail4 winner.bound = root486Candidates3 := by
  unfold candidates
  have row : edge 5227 = (SortedNatTree.contains FastRows.row5227) := by
    change Data.row5227.contains = (SortedNatTree.contains FastRows.row5227)
    exact FastRows.row5227_checked.symm
  rw [row]
  decide
theorem root486Step3 : scan edge (search edge 3) 4 [3499] 3499 root486Tail3 winner =
    scan edge (search edge 3) 4 [3499] 3499 root486Tail4 winner := by
  rw [root486Tail3, scan, if_neg (by decide), root486Candidates3_checked]
  change scan edge (search edge 3) 4 [3499] 3499 root486Tail4
    (if 3 ≤ root486Candidates3.length then search edge 3 [3499, 5227] root486Candidates3 8726 winner else winner) = _
  rw [if_neg (by decide)]
def root486Candidates4 : List Nat := [5449]
set_option maxRecDepth 8192 in
theorem root486Candidates4_checked : candidates edge 4 3499 5407 root486Tail5 winner.bound = root486Candidates4 := by
  unfold candidates
  have row : edge 5407 = (SortedNatTree.contains FastRows.row5407) := by
    change Data.row5407.contains = (SortedNatTree.contains FastRows.row5407)
    exact FastRows.row5407_checked.symm
  rw [row]
  decide
theorem root486Step4 : scan edge (search edge 3) 4 [3499] 3499 root486Tail4 winner =
    scan edge (search edge 3) 4 [3499] 3499 root486Tail5 winner := by
  rw [root486Tail4, scan, if_neg (by decide), root486Candidates4_checked]
  change scan edge (search edge 3) 4 [3499] 3499 root486Tail5
    (if 3 ≤ root486Candidates4.length then search edge 3 [3499, 5407] root486Candidates4 8906 winner else winner) = _
  rw [if_neg (by decide)]
def root486Candidates5 : List Nat := [7321]
set_option maxRecDepth 8192 in
theorem root486Candidates5_checked : candidates edge 4 3499 5449 root486Tail6 winner.bound = root486Candidates5 := by
  unfold candidates
  have row : edge 5449 = (SortedNatTree.contains FastRows.row5449) := by
    change Data.row5449.contains = (SortedNatTree.contains FastRows.row5449)
    exact FastRows.row5449_checked.symm
  rw [row]
  decide
theorem root486Step5 : scan edge (search edge 3) 4 [3499] 3499 root486Tail5 winner =
    scan edge (search edge 3) 4 [3499] 3499 root486Tail6 winner := by
  rw [root486Tail5, scan, if_neg (by decide), root486Candidates5_checked]
  change scan edge (search edge 3) 4 [3499] 3499 root486Tail6
    (if 3 ≤ root486Candidates5.length then search edge 3 [3499, 5449] root486Candidates5 8948 winner else winner) = _
  rw [if_neg (by decide)]
theorem root486Scan6 : scan edge (search edge 3) 4 [3499] 3499 root486Tail6 winner = winner := by rw [root486Tail6, scan, if_pos (by decide)]
theorem root486Scan5 : scan edge (search edge 3) 4 [3499] 3499 root486Tail5 winner = winner :=
  (root486Step5).trans root486Scan6
theorem root486Scan4 : scan edge (search edge 3) 4 [3499] 3499 root486Tail4 winner = winner :=
  (root486Step4).trans root486Scan5
theorem root486Scan3 : scan edge (search edge 3) 4 [3499] 3499 root486Tail3 winner = winner :=
  (root486Step3).trans root486Scan4
theorem root486Scan2 : scan edge (search edge 3) 4 [3499] 3499 root486Tail2 winner = winner :=
  (root486Step2).trans root486Scan3
theorem root486Scan1 : scan edge (search edge 3) 4 [3499] 3499 root486Tail1 winner = winner :=
  (root486Step1).trans root486Scan2
theorem root486Scan0 : scan edge (search edge 3) 4 [3499] 3499 root486Tail0 winner = winner :=
  (root486Step0).trans root486Scan1
set_option maxRecDepth 4096 in
theorem root486Branch : search edge 4 [3499] root486Candidates 3499 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3499] 3499 root486Tail0 winner = winner
  exact root486Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
