import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root327Candidates : List Nat := Data.row2213
set_option maxRecDepth 8192 in
theorem root327Candidates_checked : candidates edge 5 0 2213 tail328 winner.bound = root327Candidates := by
  apply candidates_eq_row edge 5 0 2213 tail328 root327Candidates winner.bound tail328_ordered
  · decide
  · decide
  · funext q
    rfl
def root327Tail33 : List Nat := []
def root327Tail32 : List Nat := 23741 :: root327Tail33
def root327Tail31 : List Nat := 22901 :: root327Tail32
def root327Tail30 : List Nat := 22397 :: root327Tail31
def root327Tail29 : List Nat := 21647 :: root327Tail30
def root327Tail28 : List Nat := 21323 :: root327Tail29
def root327Tail27 : List Nat := 20849 :: root327Tail28
def root327Tail26 : List Nat := 20681 :: root327Tail27
def root327Tail25 : List Nat := 20219 :: root327Tail26
def root327Tail24 : List Nat := 19181 :: root327Tail25
def root327Tail23 : List Nat := 18059 :: root327Tail24
def root327Tail22 : List Nat := 15767 :: root327Tail23
def root327Tail21 : List Nat := 14669 :: root327Tail22
def root327Tail20 : List Nat := 14519 :: root327Tail21
def root327Tail19 : List Nat := 14447 :: root327Tail20
def root327Tail18 : List Nat := 13241 :: root327Tail19
def root327Tail17 : List Nat := 12809 :: root327Tail18
def root327Tail16 : List Nat := 12671 :: root327Tail17
def root327Tail15 : List Nat := 12251 :: root327Tail16
def root327Tail14 : List Nat := 11813 :: root327Tail15
def root327Tail13 : List Nat := 10091 :: root327Tail14
def root327Tail12 : List Nat := 9419 :: root327Tail13
def root327Tail11 : List Nat := 8243 :: root327Tail12
def root327Tail10 : List Nat := 8147 :: root327Tail11
def root327Tail9 : List Nat := 8093 :: root327Tail10
def root327Tail8 : List Nat := 7547 :: root327Tail9
def root327Tail7 : List Nat := 7457 :: root327Tail8
def root327Tail6 : List Nat := 6029 :: root327Tail7
def root327Tail5 : List Nat := 5417 :: root327Tail6
def root327Tail4 : List Nat := 4691 :: root327Tail5
def root327Tail3 : List Nat := 4451 :: root327Tail4
def root327Tail2 : List Nat := 4253 :: root327Tail3
def root327Tail1 : List Nat := 3581 :: root327Tail2
def root327Tail0 : List Nat := 2711 :: root327Tail1
def root327Candidates0 : List Nat := [7457, 12251]
set_option maxRecDepth 8192 in
theorem root327Candidates0_checked : candidates edge 4 2213 2711 root327Tail1 winner.bound = root327Candidates0 := by
  unfold candidates
  have row : edge 2711 = (SortedNatTree.contains FastRows.row2711) := by
    change Data.row2711.contains = (SortedNatTree.contains FastRows.row2711)
    exact FastRows.row2711_checked.symm
  rw [row]
  decide
theorem root327Step0 : scan edge (search edge 3) 4 [2213] 2213 root327Tail0 winner =
    scan edge (search edge 3) 4 [2213] 2213 root327Tail1 winner := by
  rw [root327Tail0, scan, if_neg (by decide), root327Candidates0_checked]
  change scan edge (search edge 3) 4 [2213] 2213 root327Tail1
    (if 3 ≤ root327Candidates0.length then search edge 3 [2213, 2711] root327Candidates0 4924 winner else winner) = _
  rw [if_neg (by decide)]
def root327Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root327Candidates1_checked : candidates edge 4 2213 3581 root327Tail2 winner.bound = root327Candidates1 := by
  unfold candidates
  have row : edge 3581 = (SortedNatTree.contains FastRows.row3581) := by
    change Data.row3581.contains = (SortedNatTree.contains FastRows.row3581)
    exact FastRows.row3581_checked.symm
  rw [row]
  decide
theorem root327Step1 : scan edge (search edge 3) 4 [2213] 2213 root327Tail1 winner =
    scan edge (search edge 3) 4 [2213] 2213 root327Tail2 winner := by
  rw [root327Tail1, scan, if_neg (by decide), root327Candidates1_checked]
  change scan edge (search edge 3) 4 [2213] 2213 root327Tail2
    (if 3 ≤ root327Candidates1.length then search edge 3 [2213, 3581] root327Candidates1 5794 winner else winner) = _
  rw [if_neg (by decide)]
def root327Candidates2 : List Nat := [11813]
set_option maxRecDepth 8192 in
theorem root327Candidates2_checked : candidates edge 4 2213 4253 root327Tail3 winner.bound = root327Candidates2 := by
  unfold candidates
  have row : edge 4253 = (SortedNatTree.contains FastRows.row4253) := by
    change Data.row4253.contains = (SortedNatTree.contains FastRows.row4253)
    exact FastRows.row4253_checked.symm
  rw [row]
  decide
theorem root327Step2 : scan edge (search edge 3) 4 [2213] 2213 root327Tail2 winner =
    scan edge (search edge 3) 4 [2213] 2213 root327Tail3 winner := by
  rw [root327Tail2, scan, if_neg (by decide), root327Candidates2_checked]
  change scan edge (search edge 3) 4 [2213] 2213 root327Tail3
    (if 3 ≤ root327Candidates2.length then search edge 3 [2213, 4253] root327Candidates2 6466 winner else winner) = _
  rw [if_neg (by decide)]
def root327Candidates3 : List Nat := [14669]
set_option maxRecDepth 8192 in
theorem root327Candidates3_checked : candidates edge 4 2213 4451 root327Tail4 winner.bound = root327Candidates3 := by
  unfold candidates
  have row : edge 4451 = (SortedNatTree.contains FastRows.row4451) := by
    change Data.row4451.contains = (SortedNatTree.contains FastRows.row4451)
    exact FastRows.row4451_checked.symm
  rw [row]
  decide
theorem root327Step3 : scan edge (search edge 3) 4 [2213] 2213 root327Tail3 winner =
    scan edge (search edge 3) 4 [2213] 2213 root327Tail4 winner := by
  rw [root327Tail3, scan, if_neg (by decide), root327Candidates3_checked]
  change scan edge (search edge 3) 4 [2213] 2213 root327Tail4
    (if 3 ≤ root327Candidates3.length then search edge 3 [2213, 4451] root327Candidates3 6664 winner else winner) = _
  rw [if_neg (by decide)]
def root327Candidates4 : List Nat := [7457, 14519]
set_option maxRecDepth 8192 in
theorem root327Candidates4_checked : candidates edge 4 2213 4691 root327Tail5 winner.bound = root327Candidates4 := by
  unfold candidates
  have row : edge 4691 = (SortedNatTree.contains FastRows.row4691) := by
    change Data.row4691.contains = (SortedNatTree.contains FastRows.row4691)
    exact FastRows.row4691_checked.symm
  rw [row]
  decide
theorem root327Step4 : scan edge (search edge 3) 4 [2213] 2213 root327Tail4 winner =
    scan edge (search edge 3) 4 [2213] 2213 root327Tail5 winner := by
  rw [root327Tail4, scan, if_neg (by decide), root327Candidates4_checked]
  change scan edge (search edge 3) 4 [2213] 2213 root327Tail5
    (if 3 ≤ root327Candidates4.length then search edge 3 [2213, 4691] root327Candidates4 6904 winner else winner) = _
  rw [if_neg (by decide)]
def root327Candidates5 : List Nat := [11813, 18059]
set_option maxRecDepth 8192 in
theorem root327Candidates5_checked : candidates edge 4 2213 5417 root327Tail6 winner.bound = root327Candidates5 := by
  unfold candidates
  have row : edge 5417 = (SortedNatTree.contains FastRows.row5417) := by
    change Data.row5417.contains = (SortedNatTree.contains FastRows.row5417)
    exact FastRows.row5417_checked.symm
  rw [row]
  decide
theorem root327Step5 : scan edge (search edge 3) 4 [2213] 2213 root327Tail5 winner =
    scan edge (search edge 3) 4 [2213] 2213 root327Tail6 winner := by
  rw [root327Tail5, scan, if_neg (by decide), root327Candidates5_checked]
  change scan edge (search edge 3) 4 [2213] 2213 root327Tail6
    (if 3 ≤ root327Candidates5.length then search edge 3 [2213, 5417] root327Candidates5 7630 winner else winner) = _
  rw [if_neg (by decide)]
theorem root327Scan6 : scan edge (search edge 3) 4 [2213] 2213 root327Tail6 winner = winner := by rw [root327Tail6, scan, if_pos (by decide)]
theorem root327Scan5 : scan edge (search edge 3) 4 [2213] 2213 root327Tail5 winner = winner :=
  (root327Step5).trans root327Scan6
theorem root327Scan4 : scan edge (search edge 3) 4 [2213] 2213 root327Tail4 winner = winner :=
  (root327Step4).trans root327Scan5
theorem root327Scan3 : scan edge (search edge 3) 4 [2213] 2213 root327Tail3 winner = winner :=
  (root327Step3).trans root327Scan4
theorem root327Scan2 : scan edge (search edge 3) 4 [2213] 2213 root327Tail2 winner = winner :=
  (root327Step2).trans root327Scan3
theorem root327Scan1 : scan edge (search edge 3) 4 [2213] 2213 root327Tail1 winner = winner :=
  (root327Step1).trans root327Scan2
theorem root327Scan0 : scan edge (search edge 3) 4 [2213] 2213 root327Tail0 winner = winner :=
  (root327Step0).trans root327Scan1
set_option maxRecDepth 4096 in
theorem root327Branch : search edge 4 [2213] root327Candidates 2213 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2213] 2213 root327Tail0 winner = winner
  exact root327Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
