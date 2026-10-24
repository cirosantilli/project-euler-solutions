import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root484Candidates : List Nat := Data.row3469
set_option maxRecDepth 8192 in
theorem root484Candidates_checked : candidates edge 5 0 3469 tail485 winner.bound = root484Candidates := by
  apply candidates_eq_row edge 5 0 3469 tail485 root484Candidates winner.bound tail485_ordered
  · decide
  · decide
  · funext q
    rfl
def root484Tail33 : List Nat := []
def root484Tail32 : List Nat := 21739 :: root484Tail33
def root484Tail31 : List Nat := 19861 :: root484Tail32
def root484Tail30 : List Nat := 19273 :: root484Tail31
def root484Tail29 : List Nat := 18757 :: root484Tail30
def root484Tail28 : List Nat := 18043 :: root484Tail29
def root484Tail27 : List Nat := 16963 :: root484Tail28
def root484Tail26 : List Nat := 16831 :: root484Tail27
def root484Tail25 : List Nat := 15937 :: root484Tail26
def root484Tail24 : List Nat := 15643 :: root484Tail25
def root484Tail23 : List Nat := 14107 :: root484Tail24
def root484Tail22 : List Nat := 14011 :: root484Tail23
def root484Tail21 : List Nat := 13513 :: root484Tail22
def root484Tail20 : List Nat := 12799 :: root484Tail21
def root484Tail19 : List Nat := 12301 :: root484Tail20
def root484Tail18 : List Nat := 11959 :: root484Tail19
def root484Tail17 : List Nat := 11839 :: root484Tail18
def root484Tail16 : List Nat := 11317 :: root484Tail17
def root484Tail15 : List Nat := 10453 :: root484Tail16
def root484Tail14 : List Nat := 10273 :: root484Tail15
def root484Tail13 : List Nat := 9619 :: root484Tail14
def root484Tail12 : List Nat := 9403 :: root484Tail13
def root484Tail11 : List Nat := 8821 :: root484Tail12
def root484Tail10 : List Nat := 8311 :: root484Tail11
def root484Tail9 : List Nat := 7687 :: root484Tail10
def root484Tail8 : List Nat := 7309 :: root484Tail9
def root484Tail7 : List Nat := 7219 :: root484Tail8
def root484Tail6 : List Nat := 6703 :: root484Tail7
def root484Tail5 : List Nat := 5839 :: root484Tail6
def root484Tail4 : List Nat := 5503 :: root484Tail5
def root484Tail3 : List Nat := 4903 :: root484Tail4
def root484Tail2 : List Nat := 4357 :: root484Tail3
def root484Tail1 : List Nat := 4021 :: root484Tail2
def root484Tail0 : List Nat := 3517 :: root484Tail1
def root484Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root484Candidates0_checked : candidates edge 4 3469 3517 root484Tail1 winner.bound = root484Candidates0 := by
  unfold candidates
  have row : edge 3517 = (SortedNatTree.contains FastRows.row3517) := by
    change Data.row3517.contains = (SortedNatTree.contains FastRows.row3517)
    exact FastRows.row3517_checked.symm
  rw [row]
  decide
theorem root484Step0 : scan edge (search edge 3) 4 [3469] 3469 root484Tail0 winner =
    scan edge (search edge 3) 4 [3469] 3469 root484Tail1 winner := by
  rw [root484Tail0, scan, if_neg (by decide), root484Candidates0_checked]
  change scan edge (search edge 3) 4 [3469] 3469 root484Tail1
    (if 3 ≤ root484Candidates0.length then search edge 3 [3469, 3517] root484Candidates0 6986 winner else winner) = _
  rw [if_neg (by decide)]
def root484Candidates1 : List Nat := [4903, 10453, 16831]
set_option maxRecDepth 8192 in
theorem root484Candidates1_checked : candidates edge 4 3469 4021 root484Tail2 winner.bound = root484Candidates1 := by
  unfold candidates
  have row : edge 4021 = (SortedNatTree.contains FastRows.row4021) := by
    change Data.row4021.contains = (SortedNatTree.contains FastRows.row4021)
    exact FastRows.row4021_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root484Child1 : search edge 3 [3469, 4021] root484Candidates1 7490 winner = winner := by decide
theorem root484Step1 : scan edge (search edge 3) 4 [3469] 3469 root484Tail1 winner =
    scan edge (search edge 3) 4 [3469] 3469 root484Tail2 winner := by
  rw [root484Tail1, scan, if_neg (by decide), root484Candidates1_checked]
  change scan edge (search edge 3) 4 [3469] 3469 root484Tail2
    (if 3 ≤ root484Candidates1.length then search edge 3 [3469, 4021] root484Candidates1 7490 winner else winner) = _
  rw [if_pos (by decide), root484Child1]
def root484Candidates2 : List Nat := [12799, 18043]
set_option maxRecDepth 8192 in
theorem root484Candidates2_checked : candidates edge 4 3469 4357 root484Tail3 winner.bound = root484Candidates2 := by
  unfold candidates
  have row : edge 4357 = (SortedNatTree.contains FastRows.row4357) := by
    change Data.row4357.contains = (SortedNatTree.contains FastRows.row4357)
    exact FastRows.row4357_checked.symm
  rw [row]
  decide
theorem root484Step2 : scan edge (search edge 3) 4 [3469] 3469 root484Tail2 winner =
    scan edge (search edge 3) 4 [3469] 3469 root484Tail3 winner := by
  rw [root484Tail2, scan, if_neg (by decide), root484Candidates2_checked]
  change scan edge (search edge 3) 4 [3469] 3469 root484Tail3
    (if 3 ≤ root484Candidates2.length then search edge 3 [3469, 4357] root484Candidates2 7826 winner else winner) = _
  rw [if_neg (by decide)]
def root484Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root484Candidates3_checked : candidates edge 4 3469 4903 root484Tail4 winner.bound = root484Candidates3 := by
  unfold candidates
  have row : edge 4903 = (SortedNatTree.contains FastRows.row4903) := by
    change Data.row4903.contains = (SortedNatTree.contains FastRows.row4903)
    exact FastRows.row4903_checked.symm
  rw [row]
  decide
theorem root484Step3 : scan edge (search edge 3) 4 [3469] 3469 root484Tail3 winner =
    scan edge (search edge 3) 4 [3469] 3469 root484Tail4 winner := by
  rw [root484Tail3, scan, if_neg (by decide), root484Candidates3_checked]
  change scan edge (search edge 3) 4 [3469] 3469 root484Tail4
    (if 3 ≤ root484Candidates3.length then search edge 3 [3469, 4903] root484Candidates3 8372 winner else winner) = _
  rw [if_neg (by decide)]
def root484Candidates4 : List Nat := [8311, 11959]
set_option maxRecDepth 8192 in
theorem root484Candidates4_checked : candidates edge 4 3469 5503 root484Tail5 winner.bound = root484Candidates4 := by
  unfold candidates
  have row : edge 5503 = (SortedNatTree.contains FastRows.row5503) := by
    change Data.row5503.contains = (SortedNatTree.contains FastRows.row5503)
    exact FastRows.row5503_checked.symm
  rw [row]
  decide
theorem root484Step4 : scan edge (search edge 3) 4 [3469] 3469 root484Tail4 winner =
    scan edge (search edge 3) 4 [3469] 3469 root484Tail5 winner := by
  rw [root484Tail4, scan, if_neg (by decide), root484Candidates4_checked]
  change scan edge (search edge 3) 4 [3469] 3469 root484Tail5
    (if 3 ≤ root484Candidates4.length then search edge 3 [3469, 5503] root484Candidates4 8972 winner else winner) = _
  rw [if_neg (by decide)]
theorem root484Scan5 : scan edge (search edge 3) 4 [3469] 3469 root484Tail5 winner = winner := by rw [root484Tail5, scan, if_pos (by decide)]
theorem root484Scan4 : scan edge (search edge 3) 4 [3469] 3469 root484Tail4 winner = winner :=
  (root484Step4).trans root484Scan5
theorem root484Scan3 : scan edge (search edge 3) 4 [3469] 3469 root484Tail3 winner = winner :=
  (root484Step3).trans root484Scan4
theorem root484Scan2 : scan edge (search edge 3) 4 [3469] 3469 root484Tail2 winner = winner :=
  (root484Step2).trans root484Scan3
theorem root484Scan1 : scan edge (search edge 3) 4 [3469] 3469 root484Tail1 winner = winner :=
  (root484Step1).trans root484Scan2
theorem root484Scan0 : scan edge (search edge 3) 4 [3469] 3469 root484Tail0 winner = winner :=
  (root484Step0).trans root484Scan1
set_option maxRecDepth 4096 in
theorem root484Branch : search edge 4 [3469] root484Candidates 3469 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3469] 3469 root484Tail0 winner = winner
  exact root484Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
