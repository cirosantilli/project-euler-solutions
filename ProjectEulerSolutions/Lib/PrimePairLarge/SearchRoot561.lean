import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root561Candidates : List Nat := Data.row4093
set_option maxRecDepth 8192 in
theorem root561Candidates_checked : candidates edge 5 0 4093 tail562 winner.bound = root561Candidates := by
  apply candidates_eq_row edge 5 0 4093 tail562 root561Candidates winner.bound tail562_ordered
  · decide
  · decide
  · funext q
    rfl
def root561Tail31 : List Nat := []
def root561Tail30 : List Nat := 18181 :: root561Tail31
def root561Tail29 : List Nat := 16417 :: root561Tail30
def root561Tail28 : List Nat := 16189 :: root561Tail29
def root561Tail27 : List Nat := 16057 :: root561Tail28
def root561Tail26 : List Nat := 15877 :: root561Tail27
def root561Tail25 : List Nat := 15139 :: root561Tail26
def root561Tail24 : List Nat := 14449 :: root561Tail25
def root561Tail23 : List Nat := 13687 :: root561Tail24
def root561Tail22 : List Nat := 13183 :: root561Tail23
def root561Tail21 : List Nat := 12703 :: root561Tail22
def root561Tail20 : List Nat := 12601 :: root561Tail21
def root561Tail19 : List Nat := 12391 :: root561Tail20
def root561Tail18 : List Nat := 12109 :: root561Tail19
def root561Tail17 : List Nat := 11971 :: root561Tail18
def root561Tail16 : List Nat := 11587 :: root561Tail17
def root561Tail15 : List Nat := 11503 :: root561Tail16
def root561Tail14 : List Nat := 11467 :: root561Tail15
def root561Tail13 : List Nat := 11173 :: root561Tail14
def root561Tail12 : List Nat := 9721 :: root561Tail13
def root561Tail11 : List Nat := 9403 :: root561Tail12
def root561Tail10 : List Nat := 8647 :: root561Tail11
def root561Tail9 : List Nat := 8167 :: root561Tail10
def root561Tail8 : List Nat := 7933 :: root561Tail9
def root561Tail7 : List Nat := 7321 :: root561Tail8
def root561Tail6 : List Nat := 6991 :: root561Tail7
def root561Tail5 : List Nat := 6823 :: root561Tail6
def root561Tail4 : List Nat := 5749 :: root561Tail5
def root561Tail3 : List Nat := 5659 :: root561Tail4
def root561Tail2 : List Nat := 5119 :: root561Tail3
def root561Tail1 : List Nat := 5077 :: root561Tail2
def root561Tail0 : List Nat := 4567 :: root561Tail1
def root561Candidates0 : List Nat := [16189, 16417]
set_option maxRecDepth 8192 in
theorem root561Candidates0_checked : candidates edge 4 4093 4567 root561Tail1 winner.bound = root561Candidates0 := by
  unfold candidates
  have row : edge 4567 = (SortedNatTree.contains FastRows.row4567) := by
    change Data.row4567.contains = (SortedNatTree.contains FastRows.row4567)
    exact FastRows.row4567_checked.symm
  rw [row]
  decide
theorem root561Step0 : scan edge (search edge 3) 4 [4093] 4093 root561Tail0 winner =
    scan edge (search edge 3) 4 [4093] 4093 root561Tail1 winner := by
  rw [root561Tail0, scan, if_neg (by decide), root561Candidates0_checked]
  change scan edge (search edge 3) 4 [4093] 4093 root561Tail1
    (if 3 ≤ root561Candidates0.length then search edge 3 [4093, 4567] root561Candidates0 8660 winner else winner) = _
  rw [if_neg (by decide)]
def root561Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root561Candidates1_checked : candidates edge 4 4093 5077 root561Tail2 winner.bound = root561Candidates1 := by
  unfold candidates
  have row : edge 5077 = (SortedNatTree.contains FastRows.row5077) := by
    change Data.row5077.contains = (SortedNatTree.contains FastRows.row5077)
    exact FastRows.row5077_checked.symm
  rw [row]
  decide
theorem root561Step1 : scan edge (search edge 3) 4 [4093] 4093 root561Tail1 winner =
    scan edge (search edge 3) 4 [4093] 4093 root561Tail2 winner := by
  rw [root561Tail1, scan, if_neg (by decide), root561Candidates1_checked]
  change scan edge (search edge 3) 4 [4093] 4093 root561Tail2
    (if 3 ≤ root561Candidates1.length then search edge 3 [4093, 5077] root561Candidates1 9170 winner else winner) = _
  rw [if_neg (by decide)]
def root561Candidates2 : List Nat := [8647, 12601]
set_option maxRecDepth 8192 in
theorem root561Candidates2_checked : candidates edge 4 4093 5119 root561Tail3 winner.bound = root561Candidates2 := by
  unfold candidates
  have row : edge 5119 = (SortedNatTree.contains FastRows.row5119) := by
    change Data.row5119.contains = (SortedNatTree.contains FastRows.row5119)
    exact FastRows.row5119_checked.symm
  rw [row]
  decide
theorem root561Step2 : scan edge (search edge 3) 4 [4093] 4093 root561Tail2 winner =
    scan edge (search edge 3) 4 [4093] 4093 root561Tail3 winner := by
  rw [root561Tail2, scan, if_neg (by decide), root561Candidates2_checked]
  change scan edge (search edge 3) 4 [4093] 4093 root561Tail3
    (if 3 ≤ root561Candidates2.length then search edge 3 [4093, 5119] root561Candidates2 9212 winner else winner) = _
  rw [if_neg (by decide)]
theorem root561Scan3 : scan edge (search edge 3) 4 [4093] 4093 root561Tail3 winner = winner := by rw [root561Tail3, scan, if_pos (by decide)]
theorem root561Scan2 : scan edge (search edge 3) 4 [4093] 4093 root561Tail2 winner = winner :=
  (root561Step2).trans root561Scan3
theorem root561Scan1 : scan edge (search edge 3) 4 [4093] 4093 root561Tail1 winner = winner :=
  (root561Step1).trans root561Scan2
theorem root561Scan0 : scan edge (search edge 3) 4 [4093] 4093 root561Tail0 winner = winner :=
  (root561Step0).trans root561Scan1
set_option maxRecDepth 4096 in
theorem root561Branch : search edge 4 [4093] root561Candidates 4093 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4093] 4093 root561Tail0 winner = winner
  exact root561Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
