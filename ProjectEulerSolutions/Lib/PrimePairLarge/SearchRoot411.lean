import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root411Candidates : List Nat := Data.row2851
set_option maxRecDepth 8192 in
theorem root411Candidates_checked : candidates edge 5 0 2851 tail412 winner.bound = root411Candidates := by
  apply candidates_eq_row edge 5 0 2851 tail412 root411Candidates winner.bound tail412_ordered
  · decide
  · decide
  · funext q
    rfl
def root411Tail43 : List Nat := []
def root411Tail42 : List Nat := 22741 :: root411Tail43
def root411Tail41 : List Nat := 22483 :: root411Tail42
def root411Tail40 : List Nat := 21589 :: root411Tail41
def root411Tail39 : List Nat := 21559 :: root411Tail40
def root411Tail38 : List Nat := 20047 :: root411Tail39
def root411Tail37 : List Nat := 19609 :: root411Tail38
def root411Tail36 : List Nat := 19423 :: root411Tail37
def root411Tail35 : List Nat := 18913 :: root411Tail36
def root411Tail34 : List Nat := 16921 :: root411Tail35
def root411Tail33 : List Nat := 16879 :: root411Tail34
def root411Tail32 : List Nat := 16141 :: root411Tail33
def root411Tail31 : List Nat := 15913 :: root411Tail32
def root411Tail30 : List Nat := 15601 :: root411Tail31
def root411Tail29 : List Nat := 15427 :: root411Tail30
def root411Tail28 : List Nat := 15139 :: root411Tail29
def root411Tail27 : List Nat := 14887 :: root411Tail28
def root411Tail26 : List Nat := 14449 :: root411Tail27
def root411Tail25 : List Nat := 13879 :: root411Tail26
def root411Tail24 : List Nat := 12739 :: root411Tail25
def root411Tail23 : List Nat := 12511 :: root411Tail24
def root411Tail22 : List Nat := 12007 :: root411Tail23
def root411Tail21 : List Nat := 11839 :: root411Tail22
def root411Tail20 : List Nat := 11719 :: root411Tail21
def root411Tail19 : List Nat := 11527 :: root411Tail20
def root411Tail18 : List Nat := 11257 :: root411Tail19
def root411Tail17 : List Nat := 10939 :: root411Tail18
def root411Tail16 : List Nat := 10831 :: root411Tail17
def root411Tail15 : List Nat := 10729 :: root411Tail16
def root411Tail14 : List Nat := 9739 :: root411Tail15
def root411Tail13 : List Nat := 9697 :: root411Tail14
def root411Tail12 : List Nat := 9613 :: root411Tail13
def root411Tail11 : List Nat := 8647 :: root411Tail12
def root411Tail10 : List Nat := 7219 :: root411Tail11
def root411Tail9 : List Nat := 6949 :: root411Tail10
def root411Tail8 : List Nat := 6277 :: root411Tail9
def root411Tail7 : List Nat := 6091 :: root411Tail8
def root411Tail6 : List Nat := 5737 :: root411Tail7
def root411Tail5 : List Nat := 5119 :: root411Tail6
def root411Tail4 : List Nat := 5107 :: root411Tail5
def root411Tail3 : List Nat := 4951 :: root411Tail4
def root411Tail2 : List Nat := 3733 :: root411Tail3
def root411Tail1 : List Nat := 3313 :: root411Tail2
def root411Tail0 : List Nat := 3187 :: root411Tail1
def root411Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root411Candidates0_checked : candidates edge 4 2851 3187 root411Tail1 winner.bound = root411Candidates0 := by
  unfold candidates
  have row : edge 3187 = (SortedNatTree.contains FastRows.row3187) := by
    change Data.row3187.contains = (SortedNatTree.contains FastRows.row3187)
    exact FastRows.row3187_checked.symm
  rw [row]
  decide
theorem root411Step0 : scan edge (search edge 3) 4 [2851] 2851 root411Tail0 winner =
    scan edge (search edge 3) 4 [2851] 2851 root411Tail1 winner := by
  rw [root411Tail0, scan, if_neg (by decide), root411Candidates0_checked]
  change scan edge (search edge 3) 4 [2851] 2851 root411Tail1
    (if 3 ≤ root411Candidates0.length then search edge 3 [2851, 3187] root411Candidates0 6038 winner else winner) = _
  rw [if_neg (by decide)]
def root411Candidates1 : List Nat := [3733, 4951, 9739, 14449]
set_option maxRecDepth 8192 in
theorem root411Candidates1_checked : candidates edge 4 2851 3313 root411Tail2 winner.bound = root411Candidates1 := by
  unfold candidates
  have row : edge 3313 = (SortedNatTree.contains FastRows.row3313) := by
    change Data.row3313.contains = (SortedNatTree.contains FastRows.row3313)
    exact FastRows.row3313_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root411Child1 : search edge 3 [2851, 3313] root411Candidates1 6164 winner = winner := by decide
theorem root411Step1 : scan edge (search edge 3) 4 [2851] 2851 root411Tail1 winner =
    scan edge (search edge 3) 4 [2851] 2851 root411Tail2 winner := by
  rw [root411Tail1, scan, if_neg (by decide), root411Candidates1_checked]
  change scan edge (search edge 3) 4 [2851] 2851 root411Tail2
    (if 3 ≤ root411Candidates1.length then search edge 3 [2851, 3313] root411Candidates1 6164 winner else winner) = _
  rw [if_pos (by decide), root411Child1]
def root411Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root411Candidates2_checked : candidates edge 4 2851 3733 root411Tail3 winner.bound = root411Candidates2 := by
  unfold candidates
  have row : edge 3733 = (SortedNatTree.contains FastRows.row3733) := by
    change Data.row3733.contains = (SortedNatTree.contains FastRows.row3733)
    exact FastRows.row3733_checked.symm
  rw [row]
  decide
theorem root411Step2 : scan edge (search edge 3) 4 [2851] 2851 root411Tail2 winner =
    scan edge (search edge 3) 4 [2851] 2851 root411Tail3 winner := by
  rw [root411Tail2, scan, if_neg (by decide), root411Candidates2_checked]
  change scan edge (search edge 3) 4 [2851] 2851 root411Tail3
    (if 3 ≤ root411Candidates2.length then search edge 3 [2851, 3733] root411Candidates2 6584 winner else winner) = _
  rw [if_neg (by decide)]
def root411Candidates3 : List Nat := [5107, 9697]
set_option maxRecDepth 8192 in
theorem root411Candidates3_checked : candidates edge 4 2851 4951 root411Tail4 winner.bound = root411Candidates3 := by
  unfold candidates
  have row : edge 4951 = (SortedNatTree.contains FastRows.row4951) := by
    change Data.row4951.contains = (SortedNatTree.contains FastRows.row4951)
    exact FastRows.row4951_checked.symm
  rw [row]
  decide
theorem root411Step3 : scan edge (search edge 3) 4 [2851] 2851 root411Tail3 winner =
    scan edge (search edge 3) 4 [2851] 2851 root411Tail4 winner := by
  rw [root411Tail3, scan, if_neg (by decide), root411Candidates3_checked]
  change scan edge (search edge 3) 4 [2851] 2851 root411Tail4
    (if 3 ≤ root411Candidates3.length then search edge 3 [2851, 4951] root411Candidates3 7802 winner else winner) = _
  rw [if_neg (by decide)]
def root411Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root411Candidates4_checked : candidates edge 4 2851 5107 root411Tail5 winner.bound = root411Candidates4 := by
  unfold candidates
  have row : edge 5107 = (SortedNatTree.contains FastRows.row5107) := by
    change Data.row5107.contains = (SortedNatTree.contains FastRows.row5107)
    exact FastRows.row5107_checked.symm
  rw [row]
  decide
theorem root411Step4 : scan edge (search edge 3) 4 [2851] 2851 root411Tail4 winner =
    scan edge (search edge 3) 4 [2851] 2851 root411Tail5 winner := by
  rw [root411Tail4, scan, if_neg (by decide), root411Candidates4_checked]
  change scan edge (search edge 3) 4 [2851] 2851 root411Tail5
    (if 3 ≤ root411Candidates4.length then search edge 3 [2851, 5107] root411Candidates4 7958 winner else winner) = _
  rw [if_neg (by decide)]
def root411Candidates5 : List Nat := [8647]
set_option maxRecDepth 8192 in
theorem root411Candidates5_checked : candidates edge 4 2851 5119 root411Tail6 winner.bound = root411Candidates5 := by
  unfold candidates
  have row : edge 5119 = (SortedNatTree.contains FastRows.row5119) := by
    change Data.row5119.contains = (SortedNatTree.contains FastRows.row5119)
    exact FastRows.row5119_checked.symm
  rw [row]
  decide
theorem root411Step5 : scan edge (search edge 3) 4 [2851] 2851 root411Tail5 winner =
    scan edge (search edge 3) 4 [2851] 2851 root411Tail6 winner := by
  rw [root411Tail5, scan, if_neg (by decide), root411Candidates5_checked]
  change scan edge (search edge 3) 4 [2851] 2851 root411Tail6
    (if 3 ≤ root411Candidates5.length then search edge 3 [2851, 5119] root411Candidates5 7970 winner else winner) = _
  rw [if_neg (by decide)]
def root411Candidates6 : List Nat := [9613, 12511]
set_option maxRecDepth 8192 in
theorem root411Candidates6_checked : candidates edge 4 2851 5737 root411Tail7 winner.bound = root411Candidates6 := by
  unfold candidates
  have row : edge 5737 = (SortedNatTree.contains FastRows.row5737) := by
    change Data.row5737.contains = (SortedNatTree.contains FastRows.row5737)
    exact FastRows.row5737_checked.symm
  rw [row]
  decide
theorem root411Step6 : scan edge (search edge 3) 4 [2851] 2851 root411Tail6 winner =
    scan edge (search edge 3) 4 [2851] 2851 root411Tail7 winner := by
  rw [root411Tail6, scan, if_neg (by decide), root411Candidates6_checked]
  change scan edge (search edge 3) 4 [2851] 2851 root411Tail7
    (if 3 ≤ root411Candidates6.length then search edge 3 [2851, 5737] root411Candidates6 8588 winner else winner) = _
  rw [if_neg (by decide)]
theorem root411Scan7 : scan edge (search edge 3) 4 [2851] 2851 root411Tail7 winner = winner := by rw [root411Tail7, scan, if_pos (by decide)]
theorem root411Scan6 : scan edge (search edge 3) 4 [2851] 2851 root411Tail6 winner = winner :=
  (root411Step6).trans root411Scan7
theorem root411Scan5 : scan edge (search edge 3) 4 [2851] 2851 root411Tail5 winner = winner :=
  (root411Step5).trans root411Scan6
theorem root411Scan4 : scan edge (search edge 3) 4 [2851] 2851 root411Tail4 winner = winner :=
  (root411Step4).trans root411Scan5
theorem root411Scan3 : scan edge (search edge 3) 4 [2851] 2851 root411Tail3 winner = winner :=
  (root411Step3).trans root411Scan4
theorem root411Scan2 : scan edge (search edge 3) 4 [2851] 2851 root411Tail2 winner = winner :=
  (root411Step2).trans root411Scan3
theorem root411Scan1 : scan edge (search edge 3) 4 [2851] 2851 root411Tail1 winner = winner :=
  (root411Step1).trans root411Scan2
theorem root411Scan0 : scan edge (search edge 3) 4 [2851] 2851 root411Tail0 winner = winner :=
  (root411Step0).trans root411Scan1
set_option maxRecDepth 4096 in
theorem root411Branch : search edge 4 [2851] root411Candidates 2851 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2851] 2851 root411Tail0 winner = winner
  exact root411Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
