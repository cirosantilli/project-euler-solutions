import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root319Candidates : List Nat := Data.row2137
set_option maxRecDepth 8192 in
theorem root319Candidates_checked : candidates edge 5 0 2137 tail320 winner.bound = root319Candidates := by
  apply candidates_eq_row edge 5 0 2137 tail320 root319Candidates winner.bound tail320_ordered
  · decide
  · decide
  · funext q
    rfl
def root319Tail47 : List Nat := []
def root319Tail46 : List Nat := 23623 :: root319Tail47
def root319Tail45 : List Nat := 22441 :: root319Tail46
def root319Tail44 : List Nat := 22303 :: root319Tail45
def root319Tail43 : List Nat := 20407 :: root319Tail44
def root319Tail42 : List Nat := 19777 :: root319Tail43
def root319Tail41 : List Nat := 19501 :: root319Tail42
def root319Tail40 : List Nat := 18787 :: root319Tail41
def root319Tail39 : List Nat := 18481 :: root319Tail40
def root319Tail38 : List Nat := 18397 :: root319Tail39
def root319Tail37 : List Nat := 17929 :: root319Tail38
def root319Tail36 : List Nat := 17341 :: root319Tail37
def root319Tail35 : List Nat := 17137 :: root319Tail36
def root319Tail34 : List Nat := 16903 :: root319Tail35
def root319Tail33 : List Nat := 14797 :: root319Tail34
def root319Tail32 : List Nat := 14251 :: root319Tail33
def root319Tail31 : List Nat := 14173 :: root319Tail32
def root319Tail30 : List Nat := 13921 :: root319Tail31
def root319Tail29 : List Nat := 13063 :: root319Tail30
def root319Tail28 : List Nat := 11779 :: root319Tail29
def root319Tail27 : List Nat := 11299 :: root319Tail28
def root319Tail26 : List Nat := 11257 :: root319Tail27
def root319Tail25 : List Nat := 10957 :: root319Tail26
def root319Tail24 : List Nat := 10597 :: root319Tail25
def root319Tail23 : List Nat := 10099 :: root319Tail24
def root319Tail22 : List Nat := 9973 :: root319Tail23
def root319Tail21 : List Nat := 9283 :: root319Tail22
def root319Tail20 : List Nat := 9157 :: root319Tail21
def root319Tail19 : List Nat := 8527 :: root319Tail20
def root319Tail18 : List Nat := 8419 :: root319Tail19
def root319Tail17 : List Nat := 8317 :: root319Tail18
def root319Tail16 : List Nat := 8233 :: root319Tail17
def root319Tail15 : List Nat := 7561 :: root319Tail16
def root319Tail14 : List Nat := 7459 :: root319Tail15
def root319Tail13 : List Nat := 7159 :: root319Tail14
def root319Tail12 : List Nat := 7039 :: root319Tail13
def root319Tail11 : List Nat := 6991 :: root319Tail12
def root319Tail10 : List Nat := 6571 :: root319Tail11
def root319Tail9 : List Nat := 6217 :: root319Tail10
def root319Tail8 : List Nat := 6133 :: root319Tail9
def root319Tail7 : List Nat := 6073 :: root319Tail8
def root319Tail6 : List Nat := 5503 :: root319Tail7
def root319Tail5 : List Nat := 4513 :: root319Tail6
def root319Tail4 : List Nat := 4051 :: root319Tail5
def root319Tail3 : List Nat := 3697 :: root319Tail4
def root319Tail2 : List Nat := 3061 :: root319Tail3
def root319Tail1 : List Nat := 2539 :: root319Tail2
def root319Tail0 : List Nat := 2293 :: root319Tail1
def root319Candidates0 : List Nat := [7039]
set_option maxRecDepth 8192 in
theorem root319Candidates0_checked : candidates edge 4 2137 2293 root319Tail1 winner.bound = root319Candidates0 := by
  unfold candidates
  have row : edge 2293 = (SortedNatTree.contains FastRows.row2293) := by
    change Data.row2293.contains = (SortedNatTree.contains FastRows.row2293)
    exact FastRows.row2293_checked.symm
  rw [row]
  decide
theorem root319Step0 : scan edge (search edge 3) 4 [2137] 2137 root319Tail0 winner =
    scan edge (search edge 3) 4 [2137] 2137 root319Tail1 winner := by
  rw [root319Tail0, scan, if_neg (by decide), root319Candidates0_checked]
  change scan edge (search edge 3) 4 [2137] 2137 root319Tail1
    (if 3 ≤ root319Candidates0.length then search edge 3 [2137, 2293] root319Candidates0 4430 winner else winner) = _
  rw [if_neg (by decide)]
def root319Candidates1 : List Nat := [7159, 17341]
set_option maxRecDepth 8192 in
theorem root319Candidates1_checked : candidates edge 4 2137 2539 root319Tail2 winner.bound = root319Candidates1 := by
  unfold candidates
  have row : edge 2539 = (SortedNatTree.contains FastRows.row2539) := by
    change Data.row2539.contains = (SortedNatTree.contains FastRows.row2539)
    exact FastRows.row2539_checked.symm
  rw [row]
  decide
theorem root319Step1 : scan edge (search edge 3) 4 [2137] 2137 root319Tail1 winner =
    scan edge (search edge 3) 4 [2137] 2137 root319Tail2 winner := by
  rw [root319Tail1, scan, if_neg (by decide), root319Candidates1_checked]
  change scan edge (search edge 3) 4 [2137] 2137 root319Tail2
    (if 3 ≤ root319Candidates1.length then search edge 3 [2137, 2539] root319Candidates1 4676 winner else winner) = _
  rw [if_neg (by decide)]
def root319Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root319Candidates2_checked : candidates edge 4 2137 3061 root319Tail3 winner.bound = root319Candidates2 := by
  unfold candidates
  have row : edge 3061 = (SortedNatTree.contains FastRows.row3061) := by
    change Data.row3061.contains = (SortedNatTree.contains FastRows.row3061)
    exact FastRows.row3061_checked.symm
  rw [row]
  decide
theorem root319Step2 : scan edge (search edge 3) 4 [2137] 2137 root319Tail2 winner =
    scan edge (search edge 3) 4 [2137] 2137 root319Tail3 winner := by
  rw [root319Tail2, scan, if_neg (by decide), root319Candidates2_checked]
  change scan edge (search edge 3) 4 [2137] 2137 root319Tail3
    (if 3 ≤ root319Candidates2.length then search edge 3 [2137, 3061] root319Candidates2 5198 winner else winner) = _
  rw [if_neg (by decide)]
def root319Candidates3 : List Nat := [7459]
set_option maxRecDepth 8192 in
theorem root319Candidates3_checked : candidates edge 4 2137 3697 root319Tail4 winner.bound = root319Candidates3 := by
  unfold candidates
  have row : edge 3697 = (SortedNatTree.contains FastRows.row3697) := by
    change Data.row3697.contains = (SortedNatTree.contains FastRows.row3697)
    exact FastRows.row3697_checked.symm
  rw [row]
  decide
theorem root319Step3 : scan edge (search edge 3) 4 [2137] 2137 root319Tail3 winner =
    scan edge (search edge 3) 4 [2137] 2137 root319Tail4 winner := by
  rw [root319Tail3, scan, if_neg (by decide), root319Candidates3_checked]
  change scan edge (search edge 3) 4 [2137] 2137 root319Tail4
    (if 3 ≤ root319Candidates3.length then search edge 3 [2137, 3697] root319Candidates3 5834 winner else winner) = _
  rw [if_neg (by decide)]
def root319Candidates4 : List Nat := [10957, 14173]
set_option maxRecDepth 8192 in
theorem root319Candidates4_checked : candidates edge 4 2137 4051 root319Tail5 winner.bound = root319Candidates4 := by
  unfold candidates
  have row : edge 4051 = (SortedNatTree.contains FastRows.row4051) := by
    change Data.row4051.contains = (SortedNatTree.contains FastRows.row4051)
    exact FastRows.row4051_checked.symm
  rw [row]
  decide
theorem root319Step4 : scan edge (search edge 3) 4 [2137] 2137 root319Tail4 winner =
    scan edge (search edge 3) 4 [2137] 2137 root319Tail5 winner := by
  rw [root319Tail4, scan, if_neg (by decide), root319Candidates4_checked]
  change scan edge (search edge 3) 4 [2137] 2137 root319Tail5
    (if 3 ≤ root319Candidates4.length then search edge 3 [2137, 4051] root319Candidates4 6188 winner else winner) = _
  rw [if_neg (by decide)]
def root319Candidates5 : List Nat := [10099, 10597, 17341]
set_option maxRecDepth 8192 in
theorem root319Candidates5_checked : candidates edge 4 2137 4513 root319Tail6 winner.bound = root319Candidates5 := by
  unfold candidates
  have row : edge 4513 = (SortedNatTree.contains FastRows.row4513) := by
    change Data.row4513.contains = (SortedNatTree.contains FastRows.row4513)
    exact FastRows.row4513_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root319Child5 : search edge 3 [2137, 4513] root319Candidates5 6650 winner = winner := by decide
theorem root319Step5 : scan edge (search edge 3) 4 [2137] 2137 root319Tail5 winner =
    scan edge (search edge 3) 4 [2137] 2137 root319Tail6 winner := by
  rw [root319Tail5, scan, if_neg (by decide), root319Candidates5_checked]
  change scan edge (search edge 3) 4 [2137] 2137 root319Tail6
    (if 3 ≤ root319Candidates5.length then search edge 3 [2137, 4513] root319Candidates5 6650 winner else winner) = _
  rw [if_pos (by decide), root319Child5]
def root319Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root319Candidates6_checked : candidates edge 4 2137 5503 root319Tail7 winner.bound = root319Candidates6 := by
  unfold candidates
  have row : edge 5503 = (SortedNatTree.contains FastRows.row5503) := by
    change Data.row5503.contains = (SortedNatTree.contains FastRows.row5503)
    exact FastRows.row5503_checked.symm
  rw [row]
  decide
theorem root319Step6 : scan edge (search edge 3) 4 [2137] 2137 root319Tail6 winner =
    scan edge (search edge 3) 4 [2137] 2137 root319Tail7 winner := by
  rw [root319Tail6, scan, if_neg (by decide), root319Candidates6_checked]
  change scan edge (search edge 3) 4 [2137] 2137 root319Tail7
    (if 3 ≤ root319Candidates6.length then search edge 3 [2137, 5503] root319Candidates6 7640 winner else winner) = _
  rw [if_neg (by decide)]
theorem root319Scan7 : scan edge (search edge 3) 4 [2137] 2137 root319Tail7 winner = winner := by rw [root319Tail7, scan, if_pos (by decide)]
theorem root319Scan6 : scan edge (search edge 3) 4 [2137] 2137 root319Tail6 winner = winner :=
  (root319Step6).trans root319Scan7
theorem root319Scan5 : scan edge (search edge 3) 4 [2137] 2137 root319Tail5 winner = winner :=
  (root319Step5).trans root319Scan6
theorem root319Scan4 : scan edge (search edge 3) 4 [2137] 2137 root319Tail4 winner = winner :=
  (root319Step4).trans root319Scan5
theorem root319Scan3 : scan edge (search edge 3) 4 [2137] 2137 root319Tail3 winner = winner :=
  (root319Step3).trans root319Scan4
theorem root319Scan2 : scan edge (search edge 3) 4 [2137] 2137 root319Tail2 winner = winner :=
  (root319Step2).trans root319Scan3
theorem root319Scan1 : scan edge (search edge 3) 4 [2137] 2137 root319Tail1 winner = winner :=
  (root319Step1).trans root319Scan2
theorem root319Scan0 : scan edge (search edge 3) 4 [2137] 2137 root319Tail0 winner = winner :=
  (root319Step0).trans root319Scan1
set_option maxRecDepth 4096 in
theorem root319Branch : search edge 4 [2137] root319Candidates 2137 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2137] 2137 root319Tail0 winner = winner
  exact root319Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
