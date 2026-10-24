import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root467Candidates : List Nat := Data.row3331
set_option maxRecDepth 8192 in
theorem root467Candidates_checked : candidates edge 5 0 3331 tail468 winner.bound = root467Candidates := by
  apply candidates_eq_row edge 5 0 3331 tail468 root467Candidates winner.bound tail468_ordered
  · decide
  · decide
  · funext q
    rfl
def root467Tail40 : List Nat := []
def root467Tail39 : List Nat := 22483 :: root467Tail40
def root467Tail38 : List Nat := 20857 :: root467Tail39
def root467Tail37 : List Nat := 20233 :: root467Tail38
def root467Tail36 : List Nat := 19009 :: root467Tail37
def root467Tail35 : List Nat := 18523 :: root467Tail36
def root467Tail34 : List Nat := 18397 :: root467Tail35
def root467Tail33 : List Nat := 17011 :: root467Tail34
def root467Tail32 : List Nat := 16747 :: root467Tail33
def root467Tail31 : List Nat := 16699 :: root467Tail32
def root467Tail30 : List Nat := 16519 :: root467Tail31
def root467Tail29 : List Nat := 14449 :: root467Tail30
def root467Tail28 : List Nat := 14011 :: root467Tail29
def root467Tail27 : List Nat := 13933 :: root467Tail28
def root467Tail26 : List Nat := 13903 :: root467Tail27
def root467Tail25 : List Nat := 13159 :: root467Tail26
def root467Tail24 : List Nat := 12757 :: root467Tail25
def root467Tail23 : List Nat := 12613 :: root467Tail24
def root467Tail22 : List Nat := 12097 :: root467Tail23
def root467Tail21 : List Nat := 10333 :: root467Tail22
def root467Tail20 : List Nat := 9679 :: root467Tail21
def root467Tail19 : List Nat := 9421 :: root467Tail20
def root467Tail18 : List Nat := 9283 :: root467Tail19
def root467Tail17 : List Nat := 9001 :: root467Tail18
def root467Tail16 : List Nat := 8737 :: root467Tail17
def root467Tail15 : List Nat := 8713 :: root467Tail16
def root467Tail14 : List Nat := 8599 :: root467Tail15
def root467Tail13 : List Nat := 8221 :: root467Tail14
def root467Tail12 : List Nat := 7993 :: root467Tail13
def root467Tail11 : List Nat := 7927 :: root467Tail12
def root467Tail10 : List Nat := 7717 :: root467Tail11
def root467Tail9 : List Nat := 7699 :: root467Tail10
def root467Tail8 : List Nat := 7159 :: root467Tail9
def root467Tail7 : List Nat := 5701 :: root467Tail8
def root467Tail6 : List Nat := 5647 :: root467Tail7
def root467Tail5 : List Nat := 5197 :: root467Tail6
def root467Tail4 : List Nat := 4903 :: root467Tail5
def root467Tail3 : List Nat := 4621 :: root467Tail4
def root467Tail2 : List Nat := 4261 :: root467Tail3
def root467Tail1 : List Nat := 4231 :: root467Tail2
def root467Tail0 : List Nat := 3559 :: root467Tail1
def root467Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root467Candidates0_checked : candidates edge 4 3331 3559 root467Tail1 winner.bound = root467Candidates0 := by
  unfold candidates
  have row : edge 3559 = (SortedNatTree.contains FastRows.row3559) := by
    change Data.row3559.contains = (SortedNatTree.contains FastRows.row3559)
    exact FastRows.row3559_checked.symm
  rw [row]
  decide
theorem root467Step0 : scan edge (search edge 3) 4 [3331] 3331 root467Tail0 winner =
    scan edge (search edge 3) 4 [3331] 3331 root467Tail1 winner := by
  rw [root467Tail0, scan, if_neg (by decide), root467Candidates0_checked]
  change scan edge (search edge 3) 4 [3331] 3331 root467Tail1
    (if 3 ≤ root467Candidates0.length then search edge 3 [3331, 3559] root467Candidates0 6890 winner else winner) = _
  rw [if_neg (by decide)]
def root467Candidates1 : List Nat := [5197, 14011]
set_option maxRecDepth 8192 in
theorem root467Candidates1_checked : candidates edge 4 3331 4231 root467Tail2 winner.bound = root467Candidates1 := by
  unfold candidates
  have row : edge 4231 = (SortedNatTree.contains FastRows.row4231) := by
    change Data.row4231.contains = (SortedNatTree.contains FastRows.row4231)
    exact FastRows.row4231_checked.symm
  rw [row]
  decide
theorem root467Step1 : scan edge (search edge 3) 4 [3331] 3331 root467Tail1 winner =
    scan edge (search edge 3) 4 [3331] 3331 root467Tail2 winner := by
  rw [root467Tail1, scan, if_neg (by decide), root467Candidates1_checked]
  change scan edge (search edge 3) 4 [3331] 3331 root467Tail2
    (if 3 ≤ root467Candidates1.length then search edge 3 [3331, 4231] root467Candidates1 7562 winner else winner) = _
  rw [if_neg (by decide)]
def root467Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root467Candidates2_checked : candidates edge 4 3331 4261 root467Tail3 winner.bound = root467Candidates2 := by
  unfold candidates
  have row : edge 4261 = (SortedNatTree.contains FastRows.row4261) := by
    change Data.row4261.contains = (SortedNatTree.contains FastRows.row4261)
    exact FastRows.row4261_checked.symm
  rw [row]
  decide
theorem root467Step2 : scan edge (search edge 3) 4 [3331] 3331 root467Tail2 winner =
    scan edge (search edge 3) 4 [3331] 3331 root467Tail3 winner := by
  rw [root467Tail2, scan, if_neg (by decide), root467Candidates2_checked]
  change scan edge (search edge 3) 4 [3331] 3331 root467Tail3
    (if 3 ≤ root467Candidates2.length then search edge 3 [3331, 4261] root467Candidates2 7592 winner else winner) = _
  rw [if_neg (by decide)]
def root467Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root467Candidates3_checked : candidates edge 4 3331 4621 root467Tail4 winner.bound = root467Candidates3 := by
  unfold candidates
  have row : edge 4621 = (SortedNatTree.contains FastRows.row4621) := by
    change Data.row4621.contains = (SortedNatTree.contains FastRows.row4621)
    exact FastRows.row4621_checked.symm
  rw [row]
  decide
theorem root467Step3 : scan edge (search edge 3) 4 [3331] 3331 root467Tail3 winner =
    scan edge (search edge 3) 4 [3331] 3331 root467Tail4 winner := by
  rw [root467Tail3, scan, if_neg (by decide), root467Candidates3_checked]
  change scan edge (search edge 3) 4 [3331] 3331 root467Tail4
    (if 3 ≤ root467Candidates3.length then search edge 3 [3331, 4621] root467Candidates3 7952 winner else winner) = _
  rw [if_neg (by decide)]
def root467Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root467Candidates4_checked : candidates edge 4 3331 4903 root467Tail5 winner.bound = root467Candidates4 := by
  unfold candidates
  have row : edge 4903 = (SortedNatTree.contains FastRows.row4903) := by
    change Data.row4903.contains = (SortedNatTree.contains FastRows.row4903)
    exact FastRows.row4903_checked.symm
  rw [row]
  decide
theorem root467Step4 : scan edge (search edge 3) 4 [3331] 3331 root467Tail4 winner =
    scan edge (search edge 3) 4 [3331] 3331 root467Tail5 winner := by
  rw [root467Tail4, scan, if_neg (by decide), root467Candidates4_checked]
  change scan edge (search edge 3) 4 [3331] 3331 root467Tail5
    (if 3 ≤ root467Candidates4.length then search edge 3 [3331, 4903] root467Candidates4 8234 winner else winner) = _
  rw [if_neg (by decide)]
def root467Candidates5 : List Nat := [5701, 8221, 9679, 13933]
set_option maxRecDepth 8192 in
theorem root467Candidates5_checked : candidates edge 4 3331 5197 root467Tail6 winner.bound = root467Candidates5 := by
  unfold candidates
  have row : edge 5197 = (SortedNatTree.contains FastRows.row5197) := by
    change Data.row5197.contains = (SortedNatTree.contains FastRows.row5197)
    exact FastRows.row5197_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root467Child5 : search edge 3 [3331, 5197] root467Candidates5 8528 winner = winner := by decide
theorem root467Step5 : scan edge (search edge 3) 4 [3331] 3331 root467Tail5 winner =
    scan edge (search edge 3) 4 [3331] 3331 root467Tail6 winner := by
  rw [root467Tail5, scan, if_neg (by decide), root467Candidates5_checked]
  change scan edge (search edge 3) 4 [3331] 3331 root467Tail6
    (if 3 ≤ root467Candidates5.length then search edge 3 [3331, 5197] root467Candidates5 8528 winner else winner) = _
  rw [if_pos (by decide), root467Child5]
def root467Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root467Candidates6_checked : candidates edge 4 3331 5647 root467Tail7 winner.bound = root467Candidates6 := by
  unfold candidates
  have row : edge 5647 = (SortedNatTree.contains FastRows.row5647) := by
    change Data.row5647.contains = (SortedNatTree.contains FastRows.row5647)
    exact FastRows.row5647_checked.symm
  rw [row]
  decide
theorem root467Step6 : scan edge (search edge 3) 4 [3331] 3331 root467Tail6 winner =
    scan edge (search edge 3) 4 [3331] 3331 root467Tail7 winner := by
  rw [root467Tail6, scan, if_neg (by decide), root467Candidates6_checked]
  change scan edge (search edge 3) 4 [3331] 3331 root467Tail7
    (if 3 ≤ root467Candidates6.length then search edge 3 [3331, 5647] root467Candidates6 8978 winner else winner) = _
  rw [if_neg (by decide)]
theorem root467Scan7 : scan edge (search edge 3) 4 [3331] 3331 root467Tail7 winner = winner := by rw [root467Tail7, scan, if_pos (by decide)]
theorem root467Scan6 : scan edge (search edge 3) 4 [3331] 3331 root467Tail6 winner = winner :=
  (root467Step6).trans root467Scan7
theorem root467Scan5 : scan edge (search edge 3) 4 [3331] 3331 root467Tail5 winner = winner :=
  (root467Step5).trans root467Scan6
theorem root467Scan4 : scan edge (search edge 3) 4 [3331] 3331 root467Tail4 winner = winner :=
  (root467Step4).trans root467Scan5
theorem root467Scan3 : scan edge (search edge 3) 4 [3331] 3331 root467Tail3 winner = winner :=
  (root467Step3).trans root467Scan4
theorem root467Scan2 : scan edge (search edge 3) 4 [3331] 3331 root467Tail2 winner = winner :=
  (root467Step2).trans root467Scan3
theorem root467Scan1 : scan edge (search edge 3) 4 [3331] 3331 root467Tail1 winner = winner :=
  (root467Step1).trans root467Scan2
theorem root467Scan0 : scan edge (search edge 3) 4 [3331] 3331 root467Tail0 winner = winner :=
  (root467Step0).trans root467Scan1
set_option maxRecDepth 4096 in
theorem root467Branch : search edge 4 [3331] root467Candidates 3331 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3331] 3331 root467Tail0 winner = winner
  exact root467Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
