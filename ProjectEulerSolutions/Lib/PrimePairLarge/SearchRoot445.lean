import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root445Candidates : List Nat := Data.row3167
set_option maxRecDepth 8192 in
theorem root445Candidates_checked : candidates edge 5 0 3167 tail446 winner.bound = root445Candidates := by
  apply candidates_eq_row edge 5 0 3167 tail446 root445Candidates winner.bound tail446_ordered
  · decide
  · decide
  · funext q
    rfl
def root445Tail31 : List Nat := []
def root445Tail30 : List Nat := 22619 :: root445Tail31
def root445Tail29 : List Nat := 20753 :: root445Tail30
def root445Tail28 : List Nat := 19421 :: root445Tail29
def root445Tail27 : List Nat := 19289 :: root445Tail28
def root445Tail26 : List Nat := 17189 :: root445Tail27
def root445Tail25 : List Nat := 16979 :: root445Tail26
def root445Tail24 : List Nat := 16811 :: root445Tail25
def root445Tail23 : List Nat := 16223 :: root445Tail24
def root445Tail22 : List Nat := 14831 :: root445Tail23
def root445Tail21 : List Nat := 14753 :: root445Tail22
def root445Tail20 : List Nat := 13967 :: root445Tail21
def root445Tail19 : List Nat := 12917 :: root445Tail20
def root445Tail18 : List Nat := 12269 :: root445Tail19
def root445Tail17 : List Nat := 12113 :: root445Tail18
def root445Tail16 : List Nat := 11657 :: root445Tail17
def root445Tail15 : List Nat := 10859 :: root445Tail16
def root445Tail14 : List Nat := 10463 :: root445Tail15
def root445Tail13 : List Nat := 9413 :: root445Tail14
def root445Tail12 : List Nat := 9377 :: root445Tail13
def root445Tail11 : List Nat := 9239 :: root445Tail12
def root445Tail10 : List Nat := 8951 :: root445Tail11
def root445Tail9 : List Nat := 8039 :: root445Tail10
def root445Tail8 : List Nat := 6521 :: root445Tail9
def root445Tail7 : List Nat := 6299 :: root445Tail8
def root445Tail6 : List Nat := 6269 :: root445Tail7
def root445Tail5 : List Nat := 5711 :: root445Tail6
def root445Tail4 : List Nat := 5309 :: root445Tail5
def root445Tail3 : List Nat := 4421 :: root445Tail4
def root445Tail2 : List Nat := 4217 :: root445Tail3
def root445Tail1 : List Nat := 4001 :: root445Tail2
def root445Tail0 : List Nat := 3527 :: root445Tail1
def root445Candidates0 : List Nat := [9239]
set_option maxRecDepth 8192 in
theorem root445Candidates0_checked : candidates edge 4 3167 3527 root445Tail1 winner.bound = root445Candidates0 := by
  unfold candidates
  have row : edge 3527 = (SortedNatTree.contains FastRows.row3527) := by
    change Data.row3527.contains = (SortedNatTree.contains FastRows.row3527)
    exact FastRows.row3527_checked.symm
  rw [row]
  decide
theorem root445Step0 : scan edge (search edge 3) 4 [3167] 3167 root445Tail0 winner =
    scan edge (search edge 3) 4 [3167] 3167 root445Tail1 winner := by
  rw [root445Tail0, scan, if_neg (by decide), root445Candidates0_checked]
  change scan edge (search edge 3) 4 [3167] 3167 root445Tail1
    (if 3 ≤ root445Candidates0.length then search edge 3 [3167, 3527] root445Candidates0 6694 winner else winner) = _
  rw [if_neg (by decide)]
def root445Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root445Candidates1_checked : candidates edge 4 3167 4001 root445Tail2 winner.bound = root445Candidates1 := by
  unfold candidates
  have row : edge 4001 = (SortedNatTree.contains FastRows.row4001) := by
    change Data.row4001.contains = (SortedNatTree.contains FastRows.row4001)
    exact FastRows.row4001_checked.symm
  rw [row]
  decide
theorem root445Step1 : scan edge (search edge 3) 4 [3167] 3167 root445Tail1 winner =
    scan edge (search edge 3) 4 [3167] 3167 root445Tail2 winner := by
  rw [root445Tail1, scan, if_neg (by decide), root445Candidates1_checked]
  change scan edge (search edge 3) 4 [3167] 3167 root445Tail2
    (if 3 ≤ root445Candidates1.length then search edge 3 [3167, 4001] root445Candidates1 7168 winner else winner) = _
  rw [if_neg (by decide)]
def root445Candidates2 : List Nat := [4421, 9377]
set_option maxRecDepth 8192 in
theorem root445Candidates2_checked : candidates edge 4 3167 4217 root445Tail3 winner.bound = root445Candidates2 := by
  unfold candidates
  have row : edge 4217 = (SortedNatTree.contains FastRows.row4217) := by
    change Data.row4217.contains = (SortedNatTree.contains FastRows.row4217)
    exact FastRows.row4217_checked.symm
  rw [row]
  decide
theorem root445Step2 : scan edge (search edge 3) 4 [3167] 3167 root445Tail2 winner =
    scan edge (search edge 3) 4 [3167] 3167 root445Tail3 winner := by
  rw [root445Tail2, scan, if_neg (by decide), root445Candidates2_checked]
  change scan edge (search edge 3) 4 [3167] 3167 root445Tail3
    (if 3 ≤ root445Candidates2.length then search edge 3 [3167, 4217] root445Candidates2 7384 winner else winner) = _
  rw [if_neg (by decide)]
def root445Candidates3 : List Nat := [8039, 10463]
set_option maxRecDepth 8192 in
theorem root445Candidates3_checked : candidates edge 4 3167 4421 root445Tail4 winner.bound = root445Candidates3 := by
  unfold candidates
  have row : edge 4421 = (SortedNatTree.contains FastRows.row4421) := by
    change Data.row4421.contains = (SortedNatTree.contains FastRows.row4421)
    exact FastRows.row4421_checked.symm
  rw [row]
  decide
theorem root445Step3 : scan edge (search edge 3) 4 [3167] 3167 root445Tail3 winner =
    scan edge (search edge 3) 4 [3167] 3167 root445Tail4 winner := by
  rw [root445Tail3, scan, if_neg (by decide), root445Candidates3_checked]
  change scan edge (search edge 3) 4 [3167] 3167 root445Tail4
    (if 3 ≤ root445Candidates3.length then search edge 3 [3167, 4421] root445Candidates3 7588 winner else winner) = _
  rw [if_neg (by decide)]
def root445Candidates4 : List Nat := [9413, 10463]
set_option maxRecDepth 8192 in
theorem root445Candidates4_checked : candidates edge 4 3167 5309 root445Tail5 winner.bound = root445Candidates4 := by
  unfold candidates
  have row : edge 5309 = (SortedNatTree.contains FastRows.row5309) := by
    change Data.row5309.contains = (SortedNatTree.contains FastRows.row5309)
    exact FastRows.row5309_checked.symm
  rw [row]
  decide
theorem root445Step4 : scan edge (search edge 3) 4 [3167] 3167 root445Tail4 winner =
    scan edge (search edge 3) 4 [3167] 3167 root445Tail5 winner := by
  rw [root445Tail4, scan, if_neg (by decide), root445Candidates4_checked]
  change scan edge (search edge 3) 4 [3167] 3167 root445Tail5
    (if 3 ≤ root445Candidates4.length then search edge 3 [3167, 5309] root445Candidates4 8476 winner else winner) = _
  rw [if_neg (by decide)]
def root445Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root445Candidates5_checked : candidates edge 4 3167 5711 root445Tail6 winner.bound = root445Candidates5 := by
  unfold candidates
  have row : edge 5711 = (SortedNatTree.contains FastRows.row5711) := by
    change Data.row5711.contains = (SortedNatTree.contains FastRows.row5711)
    exact FastRows.row5711_checked.symm
  rw [row]
  decide
theorem root445Step5 : scan edge (search edge 3) 4 [3167] 3167 root445Tail5 winner =
    scan edge (search edge 3) 4 [3167] 3167 root445Tail6 winner := by
  rw [root445Tail5, scan, if_neg (by decide), root445Candidates5_checked]
  change scan edge (search edge 3) 4 [3167] 3167 root445Tail6
    (if 3 ≤ root445Candidates5.length then search edge 3 [3167, 5711] root445Candidates5 8878 winner else winner) = _
  rw [if_neg (by decide)]
theorem root445Scan6 : scan edge (search edge 3) 4 [3167] 3167 root445Tail6 winner = winner := by rw [root445Tail6, scan, if_pos (by decide)]
theorem root445Scan5 : scan edge (search edge 3) 4 [3167] 3167 root445Tail5 winner = winner :=
  (root445Step5).trans root445Scan6
theorem root445Scan4 : scan edge (search edge 3) 4 [3167] 3167 root445Tail4 winner = winner :=
  (root445Step4).trans root445Scan5
theorem root445Scan3 : scan edge (search edge 3) 4 [3167] 3167 root445Tail3 winner = winner :=
  (root445Step3).trans root445Scan4
theorem root445Scan2 : scan edge (search edge 3) 4 [3167] 3167 root445Tail2 winner = winner :=
  (root445Step2).trans root445Scan3
theorem root445Scan1 : scan edge (search edge 3) 4 [3167] 3167 root445Tail1 winner = winner :=
  (root445Step1).trans root445Scan2
theorem root445Scan0 : scan edge (search edge 3) 4 [3167] 3167 root445Tail0 winner = winner :=
  (root445Step0).trans root445Scan1
set_option maxRecDepth 4096 in
theorem root445Branch : search edge 4 [3167] root445Candidates 3167 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3167] 3167 root445Tail0 winner = winner
  exact root445Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
