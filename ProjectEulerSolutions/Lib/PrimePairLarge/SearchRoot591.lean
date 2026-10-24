import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root591Candidates : List Nat := Data.row4349
set_option maxRecDepth 8192 in
theorem root591Candidates_checked : candidates edge 5 0 4349 tail592 winner.bound = root591Candidates := by
  apply candidates_eq_row edge 5 0 4349 tail592 root591Candidates winner.bound tail592_ordered
  · decide
  · decide
  · funext q
    rfl
def root591Tail40 : List Nat := []
def root591Tail39 : List Nat := 21107 :: root591Tail40
def root591Tail38 : List Nat := 21089 :: root591Tail39
def root591Tail37 : List Nat := 20873 :: root591Tail38
def root591Tail36 : List Nat := 20663 :: root591Tail37
def root591Tail35 : List Nat := 19889 :: root591Tail36
def root591Tail34 : List Nat := 19427 :: root591Tail35
def root591Tail33 : List Nat := 18233 :: root591Tail34
def root591Tail32 : List Nat := 17471 :: root591Tail33
def root591Tail31 : List Nat := 17207 :: root591Tail32
def root591Tail30 : List Nat := 16301 :: root591Tail31
def root591Tail29 : List Nat := 15629 :: root591Tail30
def root591Tail28 : List Nat := 14411 :: root591Tail29
def root591Tail27 : List Nat := 14321 :: root591Tail28
def root591Tail26 : List Nat := 13967 :: root591Tail27
def root591Tail25 : List Nat := 13697 :: root591Tail26
def root591Tail24 : List Nat := 13217 :: root591Tail25
def root591Tail23 : List Nat := 13001 :: root591Tail24
def root591Tail22 : List Nat := 12893 :: root591Tail23
def root591Tail21 : List Nat := 12713 :: root591Tail22
def root591Tail20 : List Nat := 12227 :: root591Tail21
def root591Tail19 : List Nat := 12119 :: root591Tail20
def root591Tail18 : List Nat := 11177 :: root591Tail19
def root591Tail17 : List Nat := 11171 :: root591Tail18
def root591Tail16 : List Nat := 11027 :: root591Tail17
def root591Tail15 : List Nat := 10607 :: root591Tail16
def root591Tail14 : List Nat := 10313 :: root591Tail15
def root591Tail13 : List Nat := 9629 :: root591Tail14
def root591Tail12 : List Nat := 9041 :: root591Tail13
def root591Tail11 : List Nat := 8093 :: root591Tail12
def root591Tail10 : List Nat := 7793 :: root591Tail11
def root591Tail9 : List Nat := 7523 :: root591Tail10
def root591Tail8 : List Nat := 7019 :: root591Tail9
def root591Tail7 : List Nat := 6581 :: root591Tail8
def root591Tail6 : List Nat := 5861 :: root591Tail7
def root591Tail5 : List Nat := 5531 :: root591Tail6
def root591Tail4 : List Nat := 5483 :: root591Tail5
def root591Tail3 : List Nat := 5303 :: root591Tail4
def root591Tail2 : List Nat := 5297 :: root591Tail3
def root591Tail1 : List Nat := 5051 :: root591Tail2
def root591Tail0 : List Nat := 4967 :: root591Tail1
def root591Candidates0 : List Nat := [5531]
set_option maxRecDepth 8192 in
theorem root591Candidates0_checked : candidates edge 4 4349 4967 root591Tail1 winner.bound = root591Candidates0 := by
  unfold candidates
  have row : edge 4967 = (SortedNatTree.contains FastRows.row4967) := by
    change Data.row4967.contains = (SortedNatTree.contains FastRows.row4967)
    exact FastRows.row4967_checked.symm
  rw [row]
  decide
theorem root591Step0 : scan edge (search edge 3) 4 [4349] 4349 root591Tail0 winner =
    scan edge (search edge 3) 4 [4349] 4349 root591Tail1 winner := by
  rw [root591Tail0, scan, if_neg (by decide), root591Candidates0_checked]
  change scan edge (search edge 3) 4 [4349] 4349 root591Tail1
    (if 3 ≤ root591Candidates0.length then search edge 3 [4349, 4967] root591Candidates0 9316 winner else winner) = _
  rw [if_neg (by decide)]
def root591Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root591Candidates1_checked : candidates edge 4 4349 5051 root591Tail2 winner.bound = root591Candidates1 := by
  unfold candidates
  have row : edge 5051 = (SortedNatTree.contains FastRows.row5051) := by
    change Data.row5051.contains = (SortedNatTree.contains FastRows.row5051)
    exact FastRows.row5051_checked.symm
  rw [row]
  decide
theorem root591Step1 : scan edge (search edge 3) 4 [4349] 4349 root591Tail1 winner =
    scan edge (search edge 3) 4 [4349] 4349 root591Tail2 winner := by
  rw [root591Tail1, scan, if_neg (by decide), root591Candidates1_checked]
  change scan edge (search edge 3) 4 [4349] 4349 root591Tail2
    (if 3 ≤ root591Candidates1.length then search edge 3 [4349, 5051] root591Candidates1 9400 winner else winner) = _
  rw [if_neg (by decide)]
def root591Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root591Candidates2_checked : candidates edge 4 4349 5297 root591Tail3 winner.bound = root591Candidates2 := by
  unfold candidates
  have row : edge 5297 = (SortedNatTree.contains FastRows.row5297) := by
    change Data.row5297.contains = (SortedNatTree.contains FastRows.row5297)
    exact FastRows.row5297_checked.symm
  rw [row]
  decide
theorem root591Step2 : scan edge (search edge 3) 4 [4349] 4349 root591Tail2 winner =
    scan edge (search edge 3) 4 [4349] 4349 root591Tail3 winner := by
  rw [root591Tail2, scan, if_neg (by decide), root591Candidates2_checked]
  change scan edge (search edge 3) 4 [4349] 4349 root591Tail3
    (if 3 ≤ root591Candidates2.length then search edge 3 [4349, 5297] root591Candidates2 9646 winner else winner) = _
  rw [if_neg (by decide)]
def root591Candidates3 : List Nat := [10607, 11027]
set_option maxRecDepth 8192 in
theorem root591Candidates3_checked : candidates edge 4 4349 5303 root591Tail4 winner.bound = root591Candidates3 := by
  unfold candidates
  have row : edge 5303 = (SortedNatTree.contains FastRows.row5303) := by
    change Data.row5303.contains = (SortedNatTree.contains FastRows.row5303)
    exact FastRows.row5303_checked.symm
  rw [row]
  decide
theorem root591Step3 : scan edge (search edge 3) 4 [4349] 4349 root591Tail3 winner =
    scan edge (search edge 3) 4 [4349] 4349 root591Tail4 winner := by
  rw [root591Tail3, scan, if_neg (by decide), root591Candidates3_checked]
  change scan edge (search edge 3) 4 [4349] 4349 root591Tail4
    (if 3 ≤ root591Candidates3.length then search edge 3 [4349, 5303] root591Candidates3 9652 winner else winner) = _
  rw [if_neg (by decide)]
theorem root591Scan4 : scan edge (search edge 3) 4 [4349] 4349 root591Tail4 winner = winner := by rw [root591Tail4, scan, if_pos (by decide)]
theorem root591Scan3 : scan edge (search edge 3) 4 [4349] 4349 root591Tail3 winner = winner :=
  (root591Step3).trans root591Scan4
theorem root591Scan2 : scan edge (search edge 3) 4 [4349] 4349 root591Tail2 winner = winner :=
  (root591Step2).trans root591Scan3
theorem root591Scan1 : scan edge (search edge 3) 4 [4349] 4349 root591Tail1 winner = winner :=
  (root591Step1).trans root591Scan2
theorem root591Scan0 : scan edge (search edge 3) 4 [4349] 4349 root591Tail0 winner = winner :=
  (root591Step0).trans root591Scan1
set_option maxRecDepth 4096 in
theorem root591Branch : search edge 4 [4349] root591Candidates 4349 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4349] 4349 root591Tail0 winner = winner
  exact root591Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
