import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root471Candidates : List Nat := Data.row3361
set_option maxRecDepth 8192 in
theorem root471Candidates_checked : candidates edge 5 0 3361 tail472 winner.bound = root471Candidates := by
  apply candidates_eq_row edge 5 0 3361 tail472 root471Candidates winner.bound tail472_ordered
  · decide
  · decide
  · funext q
    rfl
def root471Tail28 : List Nat := []
def root471Tail27 : List Nat := 22543 :: root471Tail28
def root471Tail26 : List Nat := 22123 :: root471Tail27
def root471Tail25 : List Nat := 21661 :: root471Tail26
def root471Tail24 : List Nat := 21613 :: root471Tail25
def root471Tail23 : List Nat := 21277 :: root471Tail24
def root471Tail22 : List Nat := 20563 :: root471Tail23
def root471Tail21 : List Nat := 17389 :: root471Tail22
def root471Tail20 : List Nat := 16069 :: root471Tail21
def root471Tail19 : List Nat := 15607 :: root471Tail20
def root471Tail18 : List Nat := 14983 :: root471Tail19
def root471Tail17 : List Nat := 14533 :: root471Tail18
def root471Tail16 : List Nat := 14083 :: root471Tail17
def root471Tail15 : List Nat := 12487 :: root471Tail16
def root471Tail14 : List Nat := 11551 :: root471Tail15
def root471Tail13 : List Nat := 11257 :: root471Tail14
def root471Tail12 : List Nat := 9811 :: root471Tail13
def root471Tail11 : List Nat := 8971 :: root471Tail12
def root471Tail10 : List Nat := 8887 :: root471Tail11
def root471Tail9 : List Nat := 8737 :: root471Tail10
def root471Tail8 : List Nat := 8719 :: root471Tail9
def root471Tail7 : List Nat := 6343 :: root471Tail8
def root471Tail6 : List Nat := 6199 :: root471Tail7
def root471Tail5 : List Nat := 5557 :: root471Tail6
def root471Tail4 : List Nat := 5419 :: root471Tail5
def root471Tail3 : List Nat := 4951 :: root471Tail4
def root471Tail2 : List Nat := 4507 :: root471Tail3
def root471Tail1 : List Nat := 3919 :: root471Tail2
def root471Tail0 : List Nat := 3739 :: root471Tail1
def root471Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root471Candidates0_checked : candidates edge 4 3361 3739 root471Tail1 winner.bound = root471Candidates0 := by
  unfold candidates
  have row : edge 3739 = (SortedNatTree.contains FastRows.row3739) := by
    change Data.row3739.contains = (SortedNatTree.contains FastRows.row3739)
    exact FastRows.row3739_checked.symm
  rw [row]
  decide
theorem root471Step0 : scan edge (search edge 3) 4 [3361] 3361 root471Tail0 winner =
    scan edge (search edge 3) 4 [3361] 3361 root471Tail1 winner := by
  rw [root471Tail0, scan, if_neg (by decide), root471Candidates0_checked]
  change scan edge (search edge 3) 4 [3361] 3361 root471Tail1
    (if 3 ≤ root471Candidates0.length then search edge 3 [3361, 3739] root471Candidates0 7100 winner else winner) = _
  rw [if_neg (by decide)]
def root471Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root471Candidates1_checked : candidates edge 4 3361 3919 root471Tail2 winner.bound = root471Candidates1 := by
  unfold candidates
  have row : edge 3919 = (SortedNatTree.contains FastRows.row3919) := by
    change Data.row3919.contains = (SortedNatTree.contains FastRows.row3919)
    exact FastRows.row3919_checked.symm
  rw [row]
  decide
theorem root471Step1 : scan edge (search edge 3) 4 [3361] 3361 root471Tail1 winner =
    scan edge (search edge 3) 4 [3361] 3361 root471Tail2 winner := by
  rw [root471Tail1, scan, if_neg (by decide), root471Candidates1_checked]
  change scan edge (search edge 3) 4 [3361] 3361 root471Tail2
    (if 3 ≤ root471Candidates1.length then search edge 3 [3361, 3919] root471Candidates1 7280 winner else winner) = _
  rw [if_neg (by decide)]
def root471Candidates2 : List Nat := [5419, 11257, 14533]
set_option maxRecDepth 8192 in
theorem root471Candidates2_checked : candidates edge 4 3361 4507 root471Tail3 winner.bound = root471Candidates2 := by
  unfold candidates
  have row : edge 4507 = (SortedNatTree.contains FastRows.row4507) := by
    change Data.row4507.contains = (SortedNatTree.contains FastRows.row4507)
    exact FastRows.row4507_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root471Child2 : search edge 3 [3361, 4507] root471Candidates2 7868 winner = winner := by decide
theorem root471Step2 : scan edge (search edge 3) 4 [3361] 3361 root471Tail2 winner =
    scan edge (search edge 3) 4 [3361] 3361 root471Tail3 winner := by
  rw [root471Tail2, scan, if_neg (by decide), root471Candidates2_checked]
  change scan edge (search edge 3) 4 [3361] 3361 root471Tail3
    (if 3 ≤ root471Candidates2.length then search edge 3 [3361, 4507] root471Candidates2 7868 winner else winner) = _
  rw [if_pos (by decide), root471Child2]
def root471Candidates3 : List Nat := [6343, 8719]
set_option maxRecDepth 8192 in
theorem root471Candidates3_checked : candidates edge 4 3361 4951 root471Tail4 winner.bound = root471Candidates3 := by
  unfold candidates
  have row : edge 4951 = (SortedNatTree.contains FastRows.row4951) := by
    change Data.row4951.contains = (SortedNatTree.contains FastRows.row4951)
    exact FastRows.row4951_checked.symm
  rw [row]
  decide
theorem root471Step3 : scan edge (search edge 3) 4 [3361] 3361 root471Tail3 winner =
    scan edge (search edge 3) 4 [3361] 3361 root471Tail4 winner := by
  rw [root471Tail3, scan, if_neg (by decide), root471Candidates3_checked]
  change scan edge (search edge 3) 4 [3361] 3361 root471Tail4
    (if 3 ≤ root471Candidates3.length then search edge 3 [3361, 4951] root471Candidates3 8312 winner else winner) = _
  rw [if_neg (by decide)]
def root471Candidates4 : List Nat := [8719, 14533]
set_option maxRecDepth 8192 in
theorem root471Candidates4_checked : candidates edge 4 3361 5419 root471Tail5 winner.bound = root471Candidates4 := by
  unfold candidates
  have row : edge 5419 = (SortedNatTree.contains FastRows.row5419) := by
    change Data.row5419.contains = (SortedNatTree.contains FastRows.row5419)
    exact FastRows.row5419_checked.symm
  rw [row]
  decide
theorem root471Step4 : scan edge (search edge 3) 4 [3361] 3361 root471Tail4 winner =
    scan edge (search edge 3) 4 [3361] 3361 root471Tail5 winner := by
  rw [root471Tail4, scan, if_neg (by decide), root471Candidates4_checked]
  change scan edge (search edge 3) 4 [3361] 3361 root471Tail5
    (if 3 ≤ root471Candidates4.length then search edge 3 [3361, 5419] root471Candidates4 8780 winner else winner) = _
  rw [if_neg (by decide)]
def root471Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root471Candidates5_checked : candidates edge 4 3361 5557 root471Tail6 winner.bound = root471Candidates5 := by
  unfold candidates
  have row : edge 5557 = (SortedNatTree.contains FastRows.row5557) := by
    change Data.row5557.contains = (SortedNatTree.contains FastRows.row5557)
    exact FastRows.row5557_checked.symm
  rw [row]
  decide
theorem root471Step5 : scan edge (search edge 3) 4 [3361] 3361 root471Tail5 winner =
    scan edge (search edge 3) 4 [3361] 3361 root471Tail6 winner := by
  rw [root471Tail5, scan, if_neg (by decide), root471Candidates5_checked]
  change scan edge (search edge 3) 4 [3361] 3361 root471Tail6
    (if 3 ≤ root471Candidates5.length then search edge 3 [3361, 5557] root471Candidates5 8918 winner else winner) = _
  rw [if_neg (by decide)]
theorem root471Scan6 : scan edge (search edge 3) 4 [3361] 3361 root471Tail6 winner = winner := by rw [root471Tail6, scan, if_pos (by decide)]
theorem root471Scan5 : scan edge (search edge 3) 4 [3361] 3361 root471Tail5 winner = winner :=
  (root471Step5).trans root471Scan6
theorem root471Scan4 : scan edge (search edge 3) 4 [3361] 3361 root471Tail4 winner = winner :=
  (root471Step4).trans root471Scan5
theorem root471Scan3 : scan edge (search edge 3) 4 [3361] 3361 root471Tail3 winner = winner :=
  (root471Step3).trans root471Scan4
theorem root471Scan2 : scan edge (search edge 3) 4 [3361] 3361 root471Tail2 winner = winner :=
  (root471Step2).trans root471Scan3
theorem root471Scan1 : scan edge (search edge 3) 4 [3361] 3361 root471Tail1 winner = winner :=
  (root471Step1).trans root471Scan2
theorem root471Scan0 : scan edge (search edge 3) 4 [3361] 3361 root471Tail0 winner = winner :=
  (root471Step0).trans root471Scan1
set_option maxRecDepth 4096 in
theorem root471Branch : search edge 4 [3361] root471Candidates 3361 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3361] 3361 root471Tail0 winner = winner
  exact root471Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
