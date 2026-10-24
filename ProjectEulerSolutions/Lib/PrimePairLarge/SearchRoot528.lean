import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root528Candidates : List Nat := Data.row3823
set_option maxRecDepth 8192 in
theorem root528Candidates_checked : candidates edge 5 0 3823 tail529 winner.bound = root528Candidates := by
  apply candidates_eq_row edge 5 0 3823 tail529 root528Candidates winner.bound tail529_ordered
  · decide
  · decide
  · funext q
    rfl
def root528Tail39 : List Nat := []
def root528Tail38 : List Nat := 21559 :: root528Tail39
def root528Tail37 : List Nat := 21001 :: root528Tail38
def root528Tail36 : List Nat := 20899 :: root528Tail37
def root528Tail35 : List Nat := 20269 :: root528Tail36
def root528Tail34 : List Nat := 19489 :: root528Tail35
def root528Tail33 : List Nat := 19237 :: root528Tail34
def root528Tail32 : List Nat := 18787 :: root528Tail33
def root528Tail31 : List Nat := 18427 :: root528Tail32
def root528Tail30 : List Nat := 16369 :: root528Tail31
def root528Tail29 : List Nat := 15859 :: root528Tail30
def root528Tail28 : List Nat := 14923 :: root528Tail29
def root528Tail27 : List Nat := 14869 :: root528Tail28
def root528Tail26 : List Nat := 14563 :: root528Tail27
def root528Tail25 : List Nat := 13831 :: root528Tail26
def root528Tail24 : List Nat := 13093 :: root528Tail25
def root528Tail23 : List Nat := 12517 :: root528Tail24
def root528Tail22 : List Nat := 12163 :: root528Tail23
def root528Tail21 : List Nat := 11833 :: root528Tail22
def root528Tail20 : List Nat := 11617 :: root528Tail21
def root528Tail19 : List Nat := 11131 :: root528Tail20
def root528Tail18 : List Nat := 10711 :: root528Tail19
def root528Tail17 : List Nat := 10597 :: root528Tail18
def root528Tail16 : List Nat := 10069 :: root528Tail17
def root528Tail15 : List Nat := 9181 :: root528Tail16
def root528Tail14 : List Nat := 8689 :: root528Tail15
def root528Tail13 : List Nat := 7993 :: root528Tail14
def root528Tail12 : List Nat := 7417 :: root528Tail13
def root528Tail11 : List Nat := 7393 :: root528Tail12
def root528Tail10 : List Nat := 6841 :: root528Tail11
def root528Tail9 : List Nat := 6199 :: root528Tail10
def root528Tail8 : List Nat := 5779 :: root528Tail9
def root528Tail7 : List Nat := 5233 :: root528Tail8
def root528Tail6 : List Nat := 5209 :: root528Tail7
def root528Tail5 : List Nat := 5167 :: root528Tail6
def root528Tail4 : List Nat := 4759 :: root528Tail5
def root528Tail3 : List Nat := 4549 :: root528Tail4
def root528Tail2 : List Nat := 4297 :: root528Tail3
def root528Tail1 : List Nat := 4201 :: root528Tail2
def root528Tail0 : List Nat := 4159 :: root528Tail1
def root528Candidates0 : List Nat := [5779, 10069, 12163]
set_option maxRecDepth 8192 in
theorem root528Candidates0_checked : candidates edge 4 3823 4159 root528Tail1 winner.bound = root528Candidates0 := by
  unfold candidates
  have row : edge 4159 = (SortedNatTree.contains FastRows.row4159) := by
    change Data.row4159.contains = (SortedNatTree.contains FastRows.row4159)
    exact FastRows.row4159_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root528Child0 : search edge 3 [3823, 4159] root528Candidates0 7982 winner = winner := by decide
theorem root528Step0 : scan edge (search edge 3) 4 [3823] 3823 root528Tail0 winner =
    scan edge (search edge 3) 4 [3823] 3823 root528Tail1 winner := by
  rw [root528Tail0, scan, if_neg (by decide), root528Candidates0_checked]
  change scan edge (search edge 3) 4 [3823] 3823 root528Tail1
    (if 3 ≤ root528Candidates0.length then search edge 3 [3823, 4159] root528Candidates0 7982 winner else winner) = _
  rw [if_pos (by decide), root528Child0]
def root528Candidates1 : List Nat := [4759, 6199, 10597]
set_option maxRecDepth 8192 in
theorem root528Candidates1_checked : candidates edge 4 3823 4201 root528Tail2 winner.bound = root528Candidates1 := by
  unfold candidates
  have row : edge 4201 = (SortedNatTree.contains FastRows.row4201) := by
    change Data.row4201.contains = (SortedNatTree.contains FastRows.row4201)
    exact FastRows.row4201_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root528Child1 : search edge 3 [3823, 4201] root528Candidates1 8024 winner = winner := by decide
theorem root528Step1 : scan edge (search edge 3) 4 [3823] 3823 root528Tail1 winner =
    scan edge (search edge 3) 4 [3823] 3823 root528Tail2 winner := by
  rw [root528Tail1, scan, if_neg (by decide), root528Candidates1_checked]
  change scan edge (search edge 3) 4 [3823] 3823 root528Tail2
    (if 3 ≤ root528Candidates1.length then search edge 3 [3823, 4201] root528Candidates1 8024 winner else winner) = _
  rw [if_pos (by decide), root528Child1]
def root528Candidates2 : List Nat := [14869]
set_option maxRecDepth 8192 in
theorem root528Candidates2_checked : candidates edge 4 3823 4297 root528Tail3 winner.bound = root528Candidates2 := by
  unfold candidates
  have row : edge 4297 = (SortedNatTree.contains FastRows.row4297) := by
    change Data.row4297.contains = (SortedNatTree.contains FastRows.row4297)
    exact FastRows.row4297_checked.symm
  rw [row]
  decide
theorem root528Step2 : scan edge (search edge 3) 4 [3823] 3823 root528Tail2 winner =
    scan edge (search edge 3) 4 [3823] 3823 root528Tail3 winner := by
  rw [root528Tail2, scan, if_neg (by decide), root528Candidates2_checked]
  change scan edge (search edge 3) 4 [3823] 3823 root528Tail3
    (if 3 ≤ root528Candidates2.length then search edge 3 [3823, 4297] root528Candidates2 8120 winner else winner) = _
  rw [if_neg (by decide)]
def root528Candidates3 : List Nat := [15859]
set_option maxRecDepth 8192 in
theorem root528Candidates3_checked : candidates edge 4 3823 4549 root528Tail4 winner.bound = root528Candidates3 := by
  unfold candidates
  have row : edge 4549 = (SortedNatTree.contains FastRows.row4549) := by
    change Data.row4549.contains = (SortedNatTree.contains FastRows.row4549)
    exact FastRows.row4549_checked.symm
  rw [row]
  decide
theorem root528Step3 : scan edge (search edge 3) 4 [3823] 3823 root528Tail3 winner =
    scan edge (search edge 3) 4 [3823] 3823 root528Tail4 winner := by
  rw [root528Tail3, scan, if_neg (by decide), root528Candidates3_checked]
  change scan edge (search edge 3) 4 [3823] 3823 root528Tail4
    (if 3 ≤ root528Candidates3.length then search edge 3 [3823, 4549] root528Candidates3 8372 winner else winner) = _
  rw [if_neg (by decide)]
def root528Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root528Candidates4_checked : candidates edge 4 3823 4759 root528Tail5 winner.bound = root528Candidates4 := by
  unfold candidates
  have row : edge 4759 = (SortedNatTree.contains FastRows.row4759) := by
    change Data.row4759.contains = (SortedNatTree.contains FastRows.row4759)
    exact FastRows.row4759_checked.symm
  rw [row]
  decide
theorem root528Step4 : scan edge (search edge 3) 4 [3823] 3823 root528Tail4 winner =
    scan edge (search edge 3) 4 [3823] 3823 root528Tail5 winner := by
  rw [root528Tail4, scan, if_neg (by decide), root528Candidates4_checked]
  change scan edge (search edge 3) 4 [3823] 3823 root528Tail5
    (if 3 ≤ root528Candidates4.length then search edge 3 [3823, 4759] root528Candidates4 8582 winner else winner) = _
  rw [if_neg (by decide)]
def root528Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root528Candidates5_checked : candidates edge 4 3823 5167 root528Tail6 winner.bound = root528Candidates5 := by
  unfold candidates
  have row : edge 5167 = (SortedNatTree.contains FastRows.row5167) := by
    change Data.row5167.contains = (SortedNatTree.contains FastRows.row5167)
    exact FastRows.row5167_checked.symm
  rw [row]
  decide
theorem root528Step5 : scan edge (search edge 3) 4 [3823] 3823 root528Tail5 winner =
    scan edge (search edge 3) 4 [3823] 3823 root528Tail6 winner := by
  rw [root528Tail5, scan, if_neg (by decide), root528Candidates5_checked]
  change scan edge (search edge 3) 4 [3823] 3823 root528Tail6
    (if 3 ≤ root528Candidates5.length then search edge 3 [3823, 5167] root528Candidates5 8990 winner else winner) = _
  rw [if_neg (by decide)]
def root528Candidates6 : List Nat := [7417, 10069]
set_option maxRecDepth 8192 in
theorem root528Candidates6_checked : candidates edge 4 3823 5209 root528Tail7 winner.bound = root528Candidates6 := by
  unfold candidates
  have row : edge 5209 = (SortedNatTree.contains FastRows.row5209) := by
    change Data.row5209.contains = (SortedNatTree.contains FastRows.row5209)
    exact FastRows.row5209_checked.symm
  rw [row]
  decide
theorem root528Step6 : scan edge (search edge 3) 4 [3823] 3823 root528Tail6 winner =
    scan edge (search edge 3) 4 [3823] 3823 root528Tail7 winner := by
  rw [root528Tail6, scan, if_neg (by decide), root528Candidates6_checked]
  change scan edge (search edge 3) 4 [3823] 3823 root528Tail7
    (if 3 ≤ root528Candidates6.length then search edge 3 [3823, 5209] root528Candidates6 9032 winner else winner) = _
  rw [if_neg (by decide)]
def root528Candidates7 : List Nat := [5779]
set_option maxRecDepth 8192 in
theorem root528Candidates7_checked : candidates edge 4 3823 5233 root528Tail8 winner.bound = root528Candidates7 := by
  unfold candidates
  have row : edge 5233 = (SortedNatTree.contains FastRows.row5233) := by
    change Data.row5233.contains = (SortedNatTree.contains FastRows.row5233)
    exact FastRows.row5233_checked.symm
  rw [row]
  decide
theorem root528Step7 : scan edge (search edge 3) 4 [3823] 3823 root528Tail7 winner =
    scan edge (search edge 3) 4 [3823] 3823 root528Tail8 winner := by
  rw [root528Tail7, scan, if_neg (by decide), root528Candidates7_checked]
  change scan edge (search edge 3) 4 [3823] 3823 root528Tail8
    (if 3 ≤ root528Candidates7.length then search edge 3 [3823, 5233] root528Candidates7 9056 winner else winner) = _
  rw [if_neg (by decide)]
theorem root528Scan8 : scan edge (search edge 3) 4 [3823] 3823 root528Tail8 winner = winner := by rw [root528Tail8, scan, if_pos (by decide)]
theorem root528Scan7 : scan edge (search edge 3) 4 [3823] 3823 root528Tail7 winner = winner :=
  (root528Step7).trans root528Scan8
theorem root528Scan6 : scan edge (search edge 3) 4 [3823] 3823 root528Tail6 winner = winner :=
  (root528Step6).trans root528Scan7
theorem root528Scan5 : scan edge (search edge 3) 4 [3823] 3823 root528Tail5 winner = winner :=
  (root528Step5).trans root528Scan6
theorem root528Scan4 : scan edge (search edge 3) 4 [3823] 3823 root528Tail4 winner = winner :=
  (root528Step4).trans root528Scan5
theorem root528Scan3 : scan edge (search edge 3) 4 [3823] 3823 root528Tail3 winner = winner :=
  (root528Step3).trans root528Scan4
theorem root528Scan2 : scan edge (search edge 3) 4 [3823] 3823 root528Tail2 winner = winner :=
  (root528Step2).trans root528Scan3
theorem root528Scan1 : scan edge (search edge 3) 4 [3823] 3823 root528Tail1 winner = winner :=
  (root528Step1).trans root528Scan2
theorem root528Scan0 : scan edge (search edge 3) 4 [3823] 3823 root528Tail0 winner = winner :=
  (root528Step0).trans root528Scan1
set_option maxRecDepth 4096 in
theorem root528Branch : search edge 4 [3823] root528Candidates 3823 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3823] 3823 root528Tail0 winner = winner
  exact root528Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
