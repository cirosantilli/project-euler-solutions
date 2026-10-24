import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root434Candidates : List Nat := Data.row3049
set_option maxRecDepth 8192 in
theorem root434Candidates_checked : candidates edge 5 0 3049 tail435 winner.bound = root434Candidates := by
  apply candidates_eq_row edge 5 0 3049 tail435 root434Candidates winner.bound tail435_ordered
  · decide
  · decide
  · funext q
    rfl
def root434Tail37 : List Nat := []
def root434Tail36 : List Nat := 21247 :: root434Tail37
def root434Tail35 : List Nat := 21187 :: root434Tail36
def root434Tail34 : List Nat := 20947 :: root434Tail35
def root434Tail33 : List Nat := 20899 :: root434Tail34
def root434Tail32 : List Nat := 20323 :: root434Tail33
def root434Tail31 : List Nat := 20233 :: root434Tail32
def root434Tail30 : List Nat := 19819 :: root434Tail31
def root434Tail29 : List Nat := 18637 :: root434Tail30
def root434Tail28 : List Nat := 18253 :: root434Tail29
def root434Tail27 : List Nat := 18217 :: root434Tail28
def root434Tail26 : List Nat := 17791 :: root434Tail27
def root434Tail25 : List Nat := 17761 :: root434Tail26
def root434Tail24 : List Nat := 16657 :: root434Tail25
def root434Tail23 : List Nat := 16111 :: root434Tail24
def root434Tail22 : List Nat := 15031 :: root434Tail23
def root434Tail21 : List Nat := 13723 :: root434Tail22
def root434Tail20 : List Nat := 12889 :: root434Tail21
def root434Tail19 : List Nat := 12007 :: root434Tail20
def root434Tail18 : List Nat := 11743 :: root434Tail19
def root434Tail17 : List Nat := 10663 :: root434Tail18
def root434Tail16 : List Nat := 10357 :: root434Tail17
def root434Tail15 : List Nat := 9949 :: root434Tail16
def root434Tail14 : List Nat := 9871 :: root434Tail15
def root434Tail13 : List Nat := 9811 :: root434Tail14
def root434Tail12 : List Nat := 9787 :: root434Tail13
def root434Tail11 : List Nat := 8737 :: root434Tail12
def root434Tail10 : List Nat := 8563 :: root434Tail11
def root434Tail9 : List Nat := 8191 :: root434Tail10
def root434Tail8 : List Nat := 8089 :: root434Tail9
def root434Tail7 : List Nat := 7297 :: root434Tail8
def root434Tail6 : List Nat := 7177 :: root434Tail7
def root434Tail5 : List Nat := 6379 :: root434Tail6
def root434Tail4 : List Nat := 5581 :: root434Tail5
def root434Tail3 : List Nat := 5167 :: root434Tail4
def root434Tail2 : List Nat := 4831 :: root434Tail3
def root434Tail1 : List Nat := 4153 :: root434Tail2
def root434Tail0 : List Nat := 3259 :: root434Tail1
def root434Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root434Candidates0_checked : candidates edge 4 3049 3259 root434Tail1 winner.bound = root434Candidates0 := by
  unfold candidates
  have row : edge 3259 = (SortedNatTree.contains FastRows.row3259) := by
    change Data.row3259.contains = (SortedNatTree.contains FastRows.row3259)
    exact FastRows.row3259_checked.symm
  rw [row]
  decide
theorem root434Step0 : scan edge (search edge 3) 4 [3049] 3049 root434Tail0 winner =
    scan edge (search edge 3) 4 [3049] 3049 root434Tail1 winner := by
  rw [root434Tail0, scan, if_neg (by decide), root434Candidates0_checked]
  change scan edge (search edge 3) 4 [3049] 3049 root434Tail1
    (if 3 ≤ root434Candidates0.length then search edge 3 [3049, 3259] root434Candidates0 6308 winner else winner) = _
  rw [if_neg (by decide)]
def root434Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root434Candidates1_checked : candidates edge 4 3049 4153 root434Tail2 winner.bound = root434Candidates1 := by
  unfold candidates
  have row : edge 4153 = (SortedNatTree.contains FastRows.row4153) := by
    change Data.row4153.contains = (SortedNatTree.contains FastRows.row4153)
    exact FastRows.row4153_checked.symm
  rw [row]
  decide
theorem root434Step1 : scan edge (search edge 3) 4 [3049] 3049 root434Tail1 winner =
    scan edge (search edge 3) 4 [3049] 3049 root434Tail2 winner := by
  rw [root434Tail1, scan, if_neg (by decide), root434Candidates1_checked]
  change scan edge (search edge 3) 4 [3049] 3049 root434Tail2
    (if 3 ≤ root434Candidates1.length then search edge 3 [3049, 4153] root434Candidates1 7202 winner else winner) = _
  rw [if_neg (by decide)]
def root434Candidates2 : List Nat := [8191, 8737]
set_option maxRecDepth 8192 in
theorem root434Candidates2_checked : candidates edge 4 3049 4831 root434Tail3 winner.bound = root434Candidates2 := by
  unfold candidates
  have row : edge 4831 = (SortedNatTree.contains FastRows.row4831) := by
    change Data.row4831.contains = (SortedNatTree.contains FastRows.row4831)
    exact FastRows.row4831_checked.symm
  rw [row]
  decide
theorem root434Step2 : scan edge (search edge 3) 4 [3049] 3049 root434Tail2 winner =
    scan edge (search edge 3) 4 [3049] 3049 root434Tail3 winner := by
  rw [root434Tail2, scan, if_neg (by decide), root434Candidates2_checked]
  change scan edge (search edge 3) 4 [3049] 3049 root434Tail3
    (if 3 ≤ root434Candidates2.length then search edge 3 [3049, 4831] root434Candidates2 7880 winner else winner) = _
  rw [if_neg (by decide)]
def root434Candidates3 : List Nat := [7177]
set_option maxRecDepth 8192 in
theorem root434Candidates3_checked : candidates edge 4 3049 5167 root434Tail4 winner.bound = root434Candidates3 := by
  unfold candidates
  have row : edge 5167 = (SortedNatTree.contains FastRows.row5167) := by
    change Data.row5167.contains = (SortedNatTree.contains FastRows.row5167)
    exact FastRows.row5167_checked.symm
  rw [row]
  decide
theorem root434Step3 : scan edge (search edge 3) 4 [3049] 3049 root434Tail3 winner =
    scan edge (search edge 3) 4 [3049] 3049 root434Tail4 winner := by
  rw [root434Tail3, scan, if_neg (by decide), root434Candidates3_checked]
  change scan edge (search edge 3) 4 [3049] 3049 root434Tail4
    (if 3 ≤ root434Candidates3.length then search edge 3 [3049, 5167] root434Candidates3 8216 winner else winner) = _
  rw [if_neg (by decide)]
def root434Candidates4 : List Nat := [9949]
set_option maxRecDepth 8192 in
theorem root434Candidates4_checked : candidates edge 4 3049 5581 root434Tail5 winner.bound = root434Candidates4 := by
  unfold candidates
  have row : edge 5581 = (SortedNatTree.contains FastRows.row5581) := by
    change Data.row5581.contains = (SortedNatTree.contains FastRows.row5581)
    exact FastRows.row5581_checked.symm
  rw [row]
  decide
theorem root434Step4 : scan edge (search edge 3) 4 [3049] 3049 root434Tail4 winner =
    scan edge (search edge 3) 4 [3049] 3049 root434Tail5 winner := by
  rw [root434Tail4, scan, if_neg (by decide), root434Candidates4_checked]
  change scan edge (search edge 3) 4 [3049] 3049 root434Tail5
    (if 3 ≤ root434Candidates4.length then search edge 3 [3049, 5581] root434Candidates4 8630 winner else winner) = _
  rw [if_neg (by decide)]
theorem root434Scan5 : scan edge (search edge 3) 4 [3049] 3049 root434Tail5 winner = winner := by rw [root434Tail5, scan, if_pos (by decide)]
theorem root434Scan4 : scan edge (search edge 3) 4 [3049] 3049 root434Tail4 winner = winner :=
  (root434Step4).trans root434Scan5
theorem root434Scan3 : scan edge (search edge 3) 4 [3049] 3049 root434Tail3 winner = winner :=
  (root434Step3).trans root434Scan4
theorem root434Scan2 : scan edge (search edge 3) 4 [3049] 3049 root434Tail2 winner = winner :=
  (root434Step2).trans root434Scan3
theorem root434Scan1 : scan edge (search edge 3) 4 [3049] 3049 root434Tail1 winner = winner :=
  (root434Step1).trans root434Scan2
theorem root434Scan0 : scan edge (search edge 3) 4 [3049] 3049 root434Tail0 winner = winner :=
  (root434Step0).trans root434Scan1
set_option maxRecDepth 4096 in
theorem root434Branch : search edge 4 [3049] root434Candidates 3049 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3049] 3049 root434Tail0 winner = winner
  exact root434Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
