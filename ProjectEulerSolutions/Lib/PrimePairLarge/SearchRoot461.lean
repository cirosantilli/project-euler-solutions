import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root461Candidates : List Nat := Data.row3301
set_option maxRecDepth 8192 in
theorem root461Candidates_checked : candidates edge 5 0 3301 tail462 winner.bound = root461Candidates := by
  apply candidates_eq_row edge 5 0 3301 tail462 root461Candidates winner.bound tail462_ordered
  · decide
  · decide
  · funext q
    rfl
def root461Tail33 : List Nat := []
def root461Tail32 : List Nat := 22669 :: root461Tail33
def root461Tail31 : List Nat := 21991 :: root461Tail32
def root461Tail30 : List Nat := 21193 :: root461Tail31
def root461Tail29 : List Nat := 21061 :: root461Tail30
def root461Tail28 : List Nat := 20743 :: root461Tail29
def root461Tail27 : List Nat := 20521 :: root461Tail28
def root461Tail26 : List Nat := 20143 :: root461Tail27
def root461Tail25 : List Nat := 18583 :: root461Tail26
def root461Tail24 : List Nat := 18541 :: root461Tail25
def root461Tail23 : List Nat := 17239 :: root461Tail24
def root461Tail22 : List Nat := 16651 :: root461Tail23
def root461Tail21 : List Nat := 16333 :: root461Tail22
def root461Tail20 : List Nat := 15031 :: root461Tail21
def root461Tail19 : List Nat := 14593 :: root461Tail20
def root461Tail18 : List Nat := 14011 :: root461Tail19
def root461Tail17 : List Nat := 13879 :: root461Tail18
def root461Tail16 : List Nat := 13219 :: root461Tail17
def root461Tail15 : List Nat := 12637 :: root461Tail16
def root461Tail14 : List Nat := 12421 :: root461Tail15
def root461Tail13 : List Nat := 10861 :: root461Tail14
def root461Tail12 : List Nat := 10531 :: root461Tail13
def root461Tail11 : List Nat := 8893 :: root461Tail12
def root461Tail10 : List Nat := 7297 :: root461Tail11
def root461Tail9 : List Nat := 6967 :: root461Tail10
def root461Tail8 : List Nat := 6961 :: root461Tail9
def root461Tail7 : List Nat := 6343 :: root461Tail8
def root461Tail6 : List Nat := 5881 :: root461Tail7
def root461Tail5 : List Nat := 5779 :: root461Tail6
def root461Tail4 : List Nat := 4651 :: root461Tail5
def root461Tail3 : List Nat := 4567 :: root461Tail4
def root461Tail2 : List Nat := 4447 :: root461Tail3
def root461Tail1 : List Nat := 4363 :: root461Tail2
def root461Tail0 : List Nat := 3517 :: root461Tail1
def root461Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root461Candidates0_checked : candidates edge 4 3301 3517 root461Tail1 winner.bound = root461Candidates0 := by
  unfold candidates
  have row : edge 3517 = (SortedNatTree.contains FastRows.row3517) := by
    change Data.row3517.contains = (SortedNatTree.contains FastRows.row3517)
    exact FastRows.row3517_checked.symm
  rw [row]
  decide
theorem root461Step0 : scan edge (search edge 3) 4 [3301] 3301 root461Tail0 winner =
    scan edge (search edge 3) 4 [3301] 3301 root461Tail1 winner := by
  rw [root461Tail0, scan, if_neg (by decide), root461Candidates0_checked]
  change scan edge (search edge 3) 4 [3301] 3301 root461Tail1
    (if 3 ≤ root461Candidates0.length then search edge 3 [3301, 3517] root461Candidates0 6818 winner else winner) = _
  rw [if_neg (by decide)]
def root461Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root461Candidates1_checked : candidates edge 4 3301 4363 root461Tail2 winner.bound = root461Candidates1 := by
  unfold candidates
  have row : edge 4363 = (SortedNatTree.contains FastRows.row4363) := by
    change Data.row4363.contains = (SortedNatTree.contains FastRows.row4363)
    exact FastRows.row4363_checked.symm
  rw [row]
  decide
theorem root461Step1 : scan edge (search edge 3) 4 [3301] 3301 root461Tail1 winner =
    scan edge (search edge 3) 4 [3301] 3301 root461Tail2 winner := by
  rw [root461Tail1, scan, if_neg (by decide), root461Candidates1_checked]
  change scan edge (search edge 3) 4 [3301] 3301 root461Tail2
    (if 3 ≤ root461Candidates1.length then search edge 3 [3301, 4363] root461Candidates1 7664 winner else winner) = _
  rw [if_neg (by decide)]
def root461Candidates2 : List Nat := [15031, 17239]
set_option maxRecDepth 8192 in
theorem root461Candidates2_checked : candidates edge 4 3301 4447 root461Tail3 winner.bound = root461Candidates2 := by
  unfold candidates
  have row : edge 4447 = (SortedNatTree.contains FastRows.row4447) := by
    change Data.row4447.contains = (SortedNatTree.contains FastRows.row4447)
    exact FastRows.row4447_checked.symm
  rw [row]
  decide
theorem root461Step2 : scan edge (search edge 3) 4 [3301] 3301 root461Tail2 winner =
    scan edge (search edge 3) 4 [3301] 3301 root461Tail3 winner := by
  rw [root461Tail2, scan, if_neg (by decide), root461Candidates2_checked]
  change scan edge (search edge 3) 4 [3301] 3301 root461Tail3
    (if 3 ≤ root461Candidates2.length then search edge 3 [3301, 4447] root461Candidates2 7748 winner else winner) = _
  rw [if_neg (by decide)]
def root461Candidates3 : List Nat := [10861]
set_option maxRecDepth 8192 in
theorem root461Candidates3_checked : candidates edge 4 3301 4567 root461Tail4 winner.bound = root461Candidates3 := by
  unfold candidates
  have row : edge 4567 = (SortedNatTree.contains FastRows.row4567) := by
    change Data.row4567.contains = (SortedNatTree.contains FastRows.row4567)
    exact FastRows.row4567_checked.symm
  rw [row]
  decide
theorem root461Step3 : scan edge (search edge 3) 4 [3301] 3301 root461Tail3 winner =
    scan edge (search edge 3) 4 [3301] 3301 root461Tail4 winner := by
  rw [root461Tail3, scan, if_neg (by decide), root461Candidates3_checked]
  change scan edge (search edge 3) 4 [3301] 3301 root461Tail4
    (if 3 ≤ root461Candidates3.length then search edge 3 [3301, 4567] root461Candidates3 7868 winner else winner) = _
  rw [if_neg (by decide)]
def root461Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root461Candidates4_checked : candidates edge 4 3301 4651 root461Tail5 winner.bound = root461Candidates4 := by
  unfold candidates
  have row : edge 4651 = (SortedNatTree.contains FastRows.row4651) := by
    change Data.row4651.contains = (SortedNatTree.contains FastRows.row4651)
    exact FastRows.row4651_checked.symm
  rw [row]
  decide
theorem root461Step4 : scan edge (search edge 3) 4 [3301] 3301 root461Tail4 winner =
    scan edge (search edge 3) 4 [3301] 3301 root461Tail5 winner := by
  rw [root461Tail4, scan, if_neg (by decide), root461Candidates4_checked]
  change scan edge (search edge 3) 4 [3301] 3301 root461Tail5
    (if 3 ≤ root461Candidates4.length then search edge 3 [3301, 4651] root461Candidates4 7952 winner else winner) = _
  rw [if_neg (by decide)]
theorem root461Scan5 : scan edge (search edge 3) 4 [3301] 3301 root461Tail5 winner = winner := by rw [root461Tail5, scan, if_pos (by decide)]
theorem root461Scan4 : scan edge (search edge 3) 4 [3301] 3301 root461Tail4 winner = winner :=
  (root461Step4).trans root461Scan5
theorem root461Scan3 : scan edge (search edge 3) 4 [3301] 3301 root461Tail3 winner = winner :=
  (root461Step3).trans root461Scan4
theorem root461Scan2 : scan edge (search edge 3) 4 [3301] 3301 root461Tail2 winner = winner :=
  (root461Step2).trans root461Scan3
theorem root461Scan1 : scan edge (search edge 3) 4 [3301] 3301 root461Tail1 winner = winner :=
  (root461Step1).trans root461Scan2
theorem root461Scan0 : scan edge (search edge 3) 4 [3301] 3301 root461Tail0 winner = winner :=
  (root461Step0).trans root461Scan1
set_option maxRecDepth 4096 in
theorem root461Branch : search edge 4 [3301] root461Candidates 3301 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3301] 3301 root461Tail0 winner = winner
  exact root461Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
