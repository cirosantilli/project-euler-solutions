import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root495Candidates : List Nat := Data.row3557
set_option maxRecDepth 8192 in
theorem root495Candidates_checked : candidates edge 5 0 3557 tail496 winner.bound = root495Candidates := by
  apply candidates_eq_row edge 5 0 3557 tail496 root495Candidates winner.bound tail496_ordered
  · decide
  · decide
  · funext q
    rfl
def root495Tail36 : List Nat := []
def root495Tail35 : List Nat := 21701 :: root495Tail36
def root495Tail34 : List Nat := 21503 :: root495Tail35
def root495Tail33 : List Nat := 21269 :: root495Tail34
def root495Tail32 : List Nat := 20063 :: root495Tail33
def root495Tail31 : List Nat := 19751 :: root495Tail32
def root495Tail30 : List Nat := 19421 :: root495Tail31
def root495Tail29 : List Nat := 19391 :: root495Tail30
def root495Tail28 : List Nat := 19163 :: root495Tail29
def root495Tail27 : List Nat := 17987 :: root495Tail28
def root495Tail26 : List Nat := 16937 :: root495Tail27
def root495Tail25 : List Nat := 16883 :: root495Tail26
def root495Tail24 : List Nat := 16811 :: root495Tail25
def root495Tail23 : List Nat := 14897 :: root495Tail24
def root495Tail22 : List Nat := 14321 :: root495Tail23
def root495Tail21 : List Nat := 13751 :: root495Tail22
def root495Tail20 : List Nat := 13679 :: root495Tail21
def root495Tail19 : List Nat := 13037 :: root495Tail20
def root495Tail18 : List Nat := 12491 :: root495Tail19
def root495Tail17 : List Nat := 11789 :: root495Tail18
def root495Tail16 : List Nat := 11717 :: root495Tail17
def root495Tail15 : List Nat := 10529 :: root495Tail16
def root495Tail14 : List Nat := 10007 :: root495Tail15
def root495Tail13 : List Nat := 9857 :: root495Tail14
def root495Tail12 : List Nat := 9629 :: root495Tail13
def root495Tail11 : List Nat := 9437 :: root495Tail12
def root495Tail10 : List Nat := 8681 :: root495Tail11
def root495Tail9 : List Nat := 6551 :: root495Tail10
def root495Tail8 : List Nat := 6329 :: root495Tail9
def root495Tail7 : List Nat := 6131 :: root495Tail8
def root495Tail6 : List Nat := 5783 :: root495Tail7
def root495Tail5 : List Nat := 5417 :: root495Tail6
def root495Tail4 : List Nat := 5279 :: root495Tail5
def root495Tail3 : List Nat := 5051 :: root495Tail4
def root495Tail2 : List Nat := 5021 :: root495Tail3
def root495Tail1 : List Nat := 4139 :: root495Tail2
def root495Tail0 : List Nat := 3803 :: root495Tail1
def root495Candidates0 : List Nat := [5051]
set_option maxRecDepth 8192 in
theorem root495Candidates0_checked : candidates edge 4 3557 3803 root495Tail1 winner.bound = root495Candidates0 := by
  unfold candidates
  have row : edge 3803 = (SortedNatTree.contains FastRows.row3803) := by
    change Data.row3803.contains = (SortedNatTree.contains FastRows.row3803)
    exact FastRows.row3803_checked.symm
  rw [row]
  decide
theorem root495Step0 : scan edge (search edge 3) 4 [3557] 3557 root495Tail0 winner =
    scan edge (search edge 3) 4 [3557] 3557 root495Tail1 winner := by
  rw [root495Tail0, scan, if_neg (by decide), root495Candidates0_checked]
  change scan edge (search edge 3) 4 [3557] 3557 root495Tail1
    (if 3 ≤ root495Candidates0.length then search edge 3 [3557, 3803] root495Candidates0 7360 winner else winner) = _
  rw [if_neg (by decide)]
def root495Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root495Candidates1_checked : candidates edge 4 3557 4139 root495Tail2 winner.bound = root495Candidates1 := by
  unfold candidates
  have row : edge 4139 = (SortedNatTree.contains FastRows.row4139) := by
    change Data.row4139.contains = (SortedNatTree.contains FastRows.row4139)
    exact FastRows.row4139_checked.symm
  rw [row]
  decide
theorem root495Step1 : scan edge (search edge 3) 4 [3557] 3557 root495Tail1 winner =
    scan edge (search edge 3) 4 [3557] 3557 root495Tail2 winner := by
  rw [root495Tail1, scan, if_neg (by decide), root495Candidates1_checked]
  change scan edge (search edge 3) 4 [3557] 3557 root495Tail2
    (if 3 ≤ root495Candidates1.length then search edge 3 [3557, 4139] root495Candidates1 7696 winner else winner) = _
  rw [if_neg (by decide)]
def root495Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root495Candidates2_checked : candidates edge 4 3557 5021 root495Tail3 winner.bound = root495Candidates2 := by
  unfold candidates
  have row : edge 5021 = (SortedNatTree.contains FastRows.row5021) := by
    change Data.row5021.contains = (SortedNatTree.contains FastRows.row5021)
    exact FastRows.row5021_checked.symm
  rw [row]
  decide
theorem root495Step2 : scan edge (search edge 3) 4 [3557] 3557 root495Tail2 winner =
    scan edge (search edge 3) 4 [3557] 3557 root495Tail3 winner := by
  rw [root495Tail2, scan, if_neg (by decide), root495Candidates2_checked]
  change scan edge (search edge 3) 4 [3557] 3557 root495Tail3
    (if 3 ≤ root495Candidates2.length then search edge 3 [3557, 5021] root495Candidates2 8578 winner else winner) = _
  rw [if_neg (by decide)]
def root495Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root495Candidates3_checked : candidates edge 4 3557 5051 root495Tail4 winner.bound = root495Candidates3 := by
  unfold candidates
  have row : edge 5051 = (SortedNatTree.contains FastRows.row5051) := by
    change Data.row5051.contains = (SortedNatTree.contains FastRows.row5051)
    exact FastRows.row5051_checked.symm
  rw [row]
  decide
theorem root495Step3 : scan edge (search edge 3) 4 [3557] 3557 root495Tail3 winner =
    scan edge (search edge 3) 4 [3557] 3557 root495Tail4 winner := by
  rw [root495Tail3, scan, if_neg (by decide), root495Candidates3_checked]
  change scan edge (search edge 3) 4 [3557] 3557 root495Tail4
    (if 3 ≤ root495Candidates3.length then search edge 3 [3557, 5051] root495Candidates3 8608 winner else winner) = _
  rw [if_neg (by decide)]
def root495Candidates4 : List Nat := [5783]
set_option maxRecDepth 8192 in
theorem root495Candidates4_checked : candidates edge 4 3557 5279 root495Tail5 winner.bound = root495Candidates4 := by
  unfold candidates
  have row : edge 5279 = (SortedNatTree.contains FastRows.row5279) := by
    change Data.row5279.contains = (SortedNatTree.contains FastRows.row5279)
    exact FastRows.row5279_checked.symm
  rw [row]
  decide
theorem root495Step4 : scan edge (search edge 3) 4 [3557] 3557 root495Tail4 winner =
    scan edge (search edge 3) 4 [3557] 3557 root495Tail5 winner := by
  rw [root495Tail4, scan, if_neg (by decide), root495Candidates4_checked]
  change scan edge (search edge 3) 4 [3557] 3557 root495Tail5
    (if 3 ≤ root495Candidates4.length then search edge 3 [3557, 5279] root495Candidates4 8836 winner else winner) = _
  rw [if_neg (by decide)]
def root495Candidates5 : List Nat := [6131, 13037]
set_option maxRecDepth 8192 in
theorem root495Candidates5_checked : candidates edge 4 3557 5417 root495Tail6 winner.bound = root495Candidates5 := by
  unfold candidates
  have row : edge 5417 = (SortedNatTree.contains FastRows.row5417) := by
    change Data.row5417.contains = (SortedNatTree.contains FastRows.row5417)
    exact FastRows.row5417_checked.symm
  rw [row]
  decide
theorem root495Step5 : scan edge (search edge 3) 4 [3557] 3557 root495Tail5 winner =
    scan edge (search edge 3) 4 [3557] 3557 root495Tail6 winner := by
  rw [root495Tail5, scan, if_neg (by decide), root495Candidates5_checked]
  change scan edge (search edge 3) 4 [3557] 3557 root495Tail6
    (if 3 ≤ root495Candidates5.length then search edge 3 [3557, 5417] root495Candidates5 8974 winner else winner) = _
  rw [if_neg (by decide)]
theorem root495Scan6 : scan edge (search edge 3) 4 [3557] 3557 root495Tail6 winner = winner := by rw [root495Tail6, scan, if_pos (by decide)]
theorem root495Scan5 : scan edge (search edge 3) 4 [3557] 3557 root495Tail5 winner = winner :=
  (root495Step5).trans root495Scan6
theorem root495Scan4 : scan edge (search edge 3) 4 [3557] 3557 root495Tail4 winner = winner :=
  (root495Step4).trans root495Scan5
theorem root495Scan3 : scan edge (search edge 3) 4 [3557] 3557 root495Tail3 winner = winner :=
  (root495Step3).trans root495Scan4
theorem root495Scan2 : scan edge (search edge 3) 4 [3557] 3557 root495Tail2 winner = winner :=
  (root495Step2).trans root495Scan3
theorem root495Scan1 : scan edge (search edge 3) 4 [3557] 3557 root495Tail1 winner = winner :=
  (root495Step1).trans root495Scan2
theorem root495Scan0 : scan edge (search edge 3) 4 [3557] 3557 root495Tail0 winner = winner :=
  (root495Step0).trans root495Scan1
set_option maxRecDepth 4096 in
theorem root495Branch : search edge 4 [3557] root495Candidates 3557 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3557] 3557 root495Tail0 winner = winner
  exact root495Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
