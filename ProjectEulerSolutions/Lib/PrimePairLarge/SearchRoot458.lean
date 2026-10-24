import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root458Candidates : List Nat := Data.row3259
set_option maxRecDepth 8192 in
theorem root458Candidates_checked : candidates edge 5 0 3259 tail459 winner.bound = root458Candidates := by
  apply candidates_eq_row edge 5 0 3259 tail459 root458Candidates winner.bound tail459_ordered
  · decide
  · decide
  · funext q
    rfl
def root458Tail30 : List Nat := []
def root458Tail29 : List Nat := 21121 :: root458Tail30
def root458Tail28 : List Nat := 20857 :: root458Tail29
def root458Tail27 : List Nat := 20731 :: root458Tail28
def root458Tail26 : List Nat := 19387 :: root458Tail27
def root458Tail25 : List Nat := 19219 :: root458Tail26
def root458Tail24 : List Nat := 18757 :: root458Tail25
def root458Tail23 : List Nat := 18013 :: root458Tail24
def root458Tail22 : List Nat := 17467 :: root458Tail23
def root458Tail21 : List Nat := 15787 :: root458Tail22
def root458Tail20 : List Nat := 12721 :: root458Tail21
def root458Tail19 : List Nat := 12583 :: root458Tail20
def root458Tail18 : List Nat := 12343 :: root458Tail19
def root458Tail17 : List Nat := 11953 :: root458Tail18
def root458Tail16 : List Nat := 11527 :: root458Tail17
def root458Tail15 : List Nat := 10771 :: root458Tail16
def root458Tail14 : List Nat := 9067 :: root458Tail15
def root458Tail13 : List Nat := 8689 :: root458Tail14
def root458Tail12 : List Nat := 8641 :: root458Tail13
def root458Tail11 : List Nat := 7963 :: root458Tail12
def root458Tail10 : List Nat := 7717 :: root458Tail11
def root458Tail9 : List Nat := 7207 :: root458Tail10
def root458Tail8 : List Nat := 6247 :: root458Tail9
def root458Tail7 : List Nat := 6199 :: root458Tail8
def root458Tail6 : List Nat := 5569 :: root458Tail7
def root458Tail5 : List Nat := 5503 :: root458Tail6
def root458Tail4 : List Nat := 5107 :: root458Tail5
def root458Tail3 : List Nat := 4567 :: root458Tail4
def root458Tail2 : List Nat := 4327 :: root458Tail3
def root458Tail1 : List Nat := 4021 :: root458Tail2
def root458Tail0 : List Nat := 3271 :: root458Tail1
def root458Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root458Candidates0_checked : candidates edge 4 3259 3271 root458Tail1 winner.bound = root458Candidates0 := by
  unfold candidates
  have row : edge 3271 = (SortedNatTree.contains FastRows.row3271) := by
    change Data.row3271.contains = (SortedNatTree.contains FastRows.row3271)
    exact FastRows.row3271_checked.symm
  rw [row]
  decide
theorem root458Step0 : scan edge (search edge 3) 4 [3259] 3259 root458Tail0 winner =
    scan edge (search edge 3) 4 [3259] 3259 root458Tail1 winner := by
  rw [root458Tail0, scan, if_neg (by decide), root458Candidates0_checked]
  change scan edge (search edge 3) 4 [3259] 3259 root458Tail1
    (if 3 ≤ root458Candidates0.length then search edge 3 [3259, 3271] root458Candidates0 6530 winner else winner) = _
  rw [if_neg (by decide)]
def root458Candidates1 : List Nat := [7207]
set_option maxRecDepth 8192 in
theorem root458Candidates1_checked : candidates edge 4 3259 4021 root458Tail2 winner.bound = root458Candidates1 := by
  unfold candidates
  have row : edge 4021 = (SortedNatTree.contains FastRows.row4021) := by
    change Data.row4021.contains = (SortedNatTree.contains FastRows.row4021)
    exact FastRows.row4021_checked.symm
  rw [row]
  decide
theorem root458Step1 : scan edge (search edge 3) 4 [3259] 3259 root458Tail1 winner =
    scan edge (search edge 3) 4 [3259] 3259 root458Tail2 winner := by
  rw [root458Tail1, scan, if_neg (by decide), root458Candidates1_checked]
  change scan edge (search edge 3) 4 [3259] 3259 root458Tail2
    (if 3 ≤ root458Candidates1.length then search edge 3 [3259, 4021] root458Candidates1 7280 winner else winner) = _
  rw [if_neg (by decide)]
def root458Candidates2 : List Nat := [5107, 5503, 7963]
set_option maxRecDepth 8192 in
theorem root458Candidates2_checked : candidates edge 4 3259 4327 root458Tail3 winner.bound = root458Candidates2 := by
  unfold candidates
  have row : edge 4327 = (SortedNatTree.contains FastRows.row4327) := by
    change Data.row4327.contains = (SortedNatTree.contains FastRows.row4327)
    exact FastRows.row4327_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root458Child2 : search edge 3 [3259, 4327] root458Candidates2 7586 winner = winner := by decide
theorem root458Step2 : scan edge (search edge 3) 4 [3259] 3259 root458Tail2 winner =
    scan edge (search edge 3) 4 [3259] 3259 root458Tail3 winner := by
  rw [root458Tail2, scan, if_neg (by decide), root458Candidates2_checked]
  change scan edge (search edge 3) 4 [3259] 3259 root458Tail3
    (if 3 ≤ root458Candidates2.length then search edge 3 [3259, 4327] root458Candidates2 7586 winner else winner) = _
  rw [if_pos (by decide), root458Child2]
def root458Candidates3 : List Nat := [17467]
set_option maxRecDepth 8192 in
theorem root458Candidates3_checked : candidates edge 4 3259 4567 root458Tail4 winner.bound = root458Candidates3 := by
  unfold candidates
  have row : edge 4567 = (SortedNatTree.contains FastRows.row4567) := by
    change Data.row4567.contains = (SortedNatTree.contains FastRows.row4567)
    exact FastRows.row4567_checked.symm
  rw [row]
  decide
theorem root458Step3 : scan edge (search edge 3) 4 [3259] 3259 root458Tail3 winner =
    scan edge (search edge 3) 4 [3259] 3259 root458Tail4 winner := by
  rw [root458Tail3, scan, if_neg (by decide), root458Candidates3_checked]
  change scan edge (search edge 3) 4 [3259] 3259 root458Tail4
    (if 3 ≤ root458Candidates3.length then search edge 3 [3259, 4567] root458Candidates3 7826 winner else winner) = _
  rw [if_neg (by decide)]
def root458Candidates4 : List Nat := [7717, 8641, 12721]
set_option maxRecDepth 8192 in
theorem root458Candidates4_checked : candidates edge 4 3259 5107 root458Tail5 winner.bound = root458Candidates4 := by
  unfold candidates
  have row : edge 5107 = (SortedNatTree.contains FastRows.row5107) := by
    change Data.row5107.contains = (SortedNatTree.contains FastRows.row5107)
    exact FastRows.row5107_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root458Child4 : search edge 3 [3259, 5107] root458Candidates4 8366 winner = winner := by decide
theorem root458Step4 : scan edge (search edge 3) 4 [3259] 3259 root458Tail4 winner =
    scan edge (search edge 3) 4 [3259] 3259 root458Tail5 winner := by
  rw [root458Tail4, scan, if_neg (by decide), root458Candidates4_checked]
  change scan edge (search edge 3) 4 [3259] 3259 root458Tail5
    (if 3 ≤ root458Candidates4.length then search edge 3 [3259, 5107] root458Candidates4 8366 winner else winner) = _
  rw [if_pos (by decide), root458Child4]
def root458Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root458Candidates5_checked : candidates edge 4 3259 5503 root458Tail6 winner.bound = root458Candidates5 := by
  unfold candidates
  have row : edge 5503 = (SortedNatTree.contains FastRows.row5503) := by
    change Data.row5503.contains = (SortedNatTree.contains FastRows.row5503)
    exact FastRows.row5503_checked.symm
  rw [row]
  decide
theorem root458Step5 : scan edge (search edge 3) 4 [3259] 3259 root458Tail5 winner =
    scan edge (search edge 3) 4 [3259] 3259 root458Tail6 winner := by
  rw [root458Tail5, scan, if_neg (by decide), root458Candidates5_checked]
  change scan edge (search edge 3) 4 [3259] 3259 root458Tail6
    (if 3 ≤ root458Candidates5.length then search edge 3 [3259, 5503] root458Candidates5 8762 winner else winner) = _
  rw [if_neg (by decide)]
def root458Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root458Candidates6_checked : candidates edge 4 3259 5569 root458Tail7 winner.bound = root458Candidates6 := by
  unfold candidates
  have row : edge 5569 = (SortedNatTree.contains FastRows.row5569) := by
    change Data.row5569.contains = (SortedNatTree.contains FastRows.row5569)
    exact FastRows.row5569_checked.symm
  rw [row]
  decide
theorem root458Step6 : scan edge (search edge 3) 4 [3259] 3259 root458Tail6 winner =
    scan edge (search edge 3) 4 [3259] 3259 root458Tail7 winner := by
  rw [root458Tail6, scan, if_neg (by decide), root458Candidates6_checked]
  change scan edge (search edge 3) 4 [3259] 3259 root458Tail7
    (if 3 ≤ root458Candidates6.length then search edge 3 [3259, 5569] root458Candidates6 8828 winner else winner) = _
  rw [if_neg (by decide)]
theorem root458Scan7 : scan edge (search edge 3) 4 [3259] 3259 root458Tail7 winner = winner := by rw [root458Tail7, scan, if_pos (by decide)]
theorem root458Scan6 : scan edge (search edge 3) 4 [3259] 3259 root458Tail6 winner = winner :=
  (root458Step6).trans root458Scan7
theorem root458Scan5 : scan edge (search edge 3) 4 [3259] 3259 root458Tail5 winner = winner :=
  (root458Step5).trans root458Scan6
theorem root458Scan4 : scan edge (search edge 3) 4 [3259] 3259 root458Tail4 winner = winner :=
  (root458Step4).trans root458Scan5
theorem root458Scan3 : scan edge (search edge 3) 4 [3259] 3259 root458Tail3 winner = winner :=
  (root458Step3).trans root458Scan4
theorem root458Scan2 : scan edge (search edge 3) 4 [3259] 3259 root458Tail2 winner = winner :=
  (root458Step2).trans root458Scan3
theorem root458Scan1 : scan edge (search edge 3) 4 [3259] 3259 root458Tail1 winner = winner :=
  (root458Step1).trans root458Scan2
theorem root458Scan0 : scan edge (search edge 3) 4 [3259] 3259 root458Tail0 winner = winner :=
  (root458Step0).trans root458Scan1
set_option maxRecDepth 4096 in
theorem root458Branch : search edge 4 [3259] root458Candidates 3259 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3259] 3259 root458Tail0 winner = winner
  exact root458Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
