import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root345Candidates : List Nat := Data.row2347
set_option maxRecDepth 8192 in
theorem root345Candidates_checked : candidates edge 5 0 2347 tail346 winner.bound = root345Candidates := by
  apply candidates_eq_row edge 5 0 2347 tail346 root345Candidates winner.bound tail346_ordered
  · decide
  · decide
  · funext q
    rfl
def root345Tail44 : List Nat := []
def root345Tail43 : List Nat := 22861 :: root345Tail44
def root345Tail42 : List Nat := 22717 :: root345Tail43
def root345Tail41 : List Nat := 22273 :: root345Tail42
def root345Tail40 : List Nat := 22003 :: root345Tail41
def root345Tail39 : List Nat := 21247 :: root345Tail40
def root345Tail38 : List Nat := 20749 :: root345Tail39
def root345Tail37 : List Nat := 19819 :: root345Tail38
def root345Tail36 : List Nat := 19753 :: root345Tail37
def root345Tail35 : List Nat := 19717 :: root345Tail36
def root345Tail34 : List Nat := 19249 :: root345Tail35
def root345Tail33 : List Nat := 16561 :: root345Tail34
def root345Tail32 : List Nat := 15643 :: root345Tail33
def root345Tail31 : List Nat := 15289 :: root345Tail32
def root345Tail30 : List Nat := 14713 :: root345Tail31
def root345Tail29 : List Nat := 14533 :: root345Tail30
def root345Tail28 : List Nat := 13693 :: root345Tail29
def root345Tail27 : List Nat := 12409 :: root345Tail28
def root345Tail26 : List Nat := 12343 :: root345Tail27
def root345Tail25 : List Nat := 12277 :: root345Tail26
def root345Tail24 : List Nat := 11311 :: root345Tail25
def root345Tail23 : List Nat := 10429 :: root345Tail24
def root345Tail22 : List Nat := 9871 :: root345Tail23
def root345Tail21 : List Nat := 9829 :: root345Tail22
def root345Tail20 : List Nat := 9769 :: root345Tail21
def root345Tail19 : List Nat := 9151 :: root345Tail20
def root345Tail18 : List Nat := 8821 :: root345Tail19
def root345Tail17 : List Nat := 7879 :: root345Tail18
def root345Tail16 : List Nat := 7873 :: root345Tail17
def root345Tail15 : List Nat := 7681 :: root345Tail16
def root345Tail14 : List Nat := 7417 :: root345Tail15
def root345Tail13 : List Nat := 7057 :: root345Tail14
def root345Tail12 : List Nat := 6883 :: root345Tail13
def root345Tail11 : List Nat := 6469 :: root345Tail12
def root345Tail10 : List Nat := 6199 :: root345Tail11
def root345Tail9 : List Nat := 4663 :: root345Tail10
def root345Tail8 : List Nat := 4639 :: root345Tail9
def root345Tail7 : List Nat := 4327 :: root345Tail8
def root345Tail6 : List Nat := 3253 :: root345Tail7
def root345Tail5 : List Nat := 3067 :: root345Tail6
def root345Tail4 : List Nat := 3049 :: root345Tail5
def root345Tail3 : List Nat := 2851 :: root345Tail4
def root345Tail2 : List Nat := 2833 :: root345Tail3
def root345Tail1 : List Nat := 2749 :: root345Tail2
def root345Tail0 : List Nat := 2731 :: root345Tail1
def root345Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root345Candidates0_checked : candidates edge 4 2347 2731 root345Tail1 winner.bound = root345Candidates0 := by
  unfold candidates
  have row : edge 2731 = (SortedNatTree.contains FastRows.row2731) := by
    change Data.row2731.contains = (SortedNatTree.contains FastRows.row2731)
    exact FastRows.row2731_checked.symm
  rw [row]
  decide
theorem root345Step0 : scan edge (search edge 3) 4 [2347] 2347 root345Tail0 winner =
    scan edge (search edge 3) 4 [2347] 2347 root345Tail1 winner := by
  rw [root345Tail0, scan, if_neg (by decide), root345Candidates0_checked]
  change scan edge (search edge 3) 4 [2347] 2347 root345Tail1
    (if 3 ≤ root345Candidates0.length then search edge 3 [2347, 2731] root345Candidates0 5078 winner else winner) = _
  rw [if_neg (by decide)]
def root345Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root345Candidates1_checked : candidates edge 4 2347 2749 root345Tail2 winner.bound = root345Candidates1 := by
  unfold candidates
  have row : edge 2749 = (SortedNatTree.contains FastRows.row2749) := by
    change Data.row2749.contains = (SortedNatTree.contains FastRows.row2749)
    exact FastRows.row2749_checked.symm
  rw [row]
  decide
theorem root345Step1 : scan edge (search edge 3) 4 [2347] 2347 root345Tail1 winner =
    scan edge (search edge 3) 4 [2347] 2347 root345Tail2 winner := by
  rw [root345Tail1, scan, if_neg (by decide), root345Candidates1_checked]
  change scan edge (search edge 3) 4 [2347] 2347 root345Tail2
    (if 3 ≤ root345Candidates1.length then search edge 3 [2347, 2749] root345Candidates1 5096 winner else winner) = _
  rw [if_neg (by decide)]
def root345Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root345Candidates2_checked : candidates edge 4 2347 2833 root345Tail3 winner.bound = root345Candidates2 := by
  unfold candidates
  have row : edge 2833 = (SortedNatTree.contains FastRows.row2833) := by
    change Data.row2833.contains = (SortedNatTree.contains FastRows.row2833)
    exact FastRows.row2833_checked.symm
  rw [row]
  decide
theorem root345Step2 : scan edge (search edge 3) 4 [2347] 2347 root345Tail2 winner =
    scan edge (search edge 3) 4 [2347] 2347 root345Tail3 winner := by
  rw [root345Tail2, scan, if_neg (by decide), root345Candidates2_checked]
  change scan edge (search edge 3) 4 [2347] 2347 root345Tail3
    (if 3 ≤ root345Candidates2.length then search edge 3 [2347, 2833] root345Candidates2 5180 winner else winner) = _
  rw [if_neg (by decide)]
def root345Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root345Candidates3_checked : candidates edge 4 2347 2851 root345Tail4 winner.bound = root345Candidates3 := by
  unfold candidates
  have row : edge 2851 = (SortedNatTree.contains FastRows.row2851) := by
    change Data.row2851.contains = (SortedNatTree.contains FastRows.row2851)
    exact FastRows.row2851_checked.symm
  rw [row]
  decide
theorem root345Step3 : scan edge (search edge 3) 4 [2347] 2347 root345Tail3 winner =
    scan edge (search edge 3) 4 [2347] 2347 root345Tail4 winner := by
  rw [root345Tail3, scan, if_neg (by decide), root345Candidates3_checked]
  change scan edge (search edge 3) 4 [2347] 2347 root345Tail4
    (if 3 ≤ root345Candidates3.length then search edge 3 [2347, 2851] root345Candidates3 5198 winner else winner) = _
  rw [if_neg (by decide)]
def root345Candidates4 : List Nat := [9871, 19819]
set_option maxRecDepth 8192 in
theorem root345Candidates4_checked : candidates edge 4 2347 3049 root345Tail5 winner.bound = root345Candidates4 := by
  unfold candidates
  have row : edge 3049 = (SortedNatTree.contains FastRows.row3049) := by
    change Data.row3049.contains = (SortedNatTree.contains FastRows.row3049)
    exact FastRows.row3049_checked.symm
  rw [row]
  decide
theorem root345Step4 : scan edge (search edge 3) 4 [2347] 2347 root345Tail4 winner =
    scan edge (search edge 3) 4 [2347] 2347 root345Tail5 winner := by
  rw [root345Tail4, scan, if_neg (by decide), root345Candidates4_checked]
  change scan edge (search edge 3) 4 [2347] 2347 root345Tail5
    (if 3 ≤ root345Candidates4.length then search edge 3 [2347, 3049] root345Candidates4 5396 winner else winner) = _
  rw [if_neg (by decide)]
def root345Candidates5 : List Nat := [9829]
set_option maxRecDepth 8192 in
theorem root345Candidates5_checked : candidates edge 4 2347 3067 root345Tail6 winner.bound = root345Candidates5 := by
  unfold candidates
  have row : edge 3067 = (SortedNatTree.contains FastRows.row3067) := by
    change Data.row3067.contains = (SortedNatTree.contains FastRows.row3067)
    exact FastRows.row3067_checked.symm
  rw [row]
  decide
theorem root345Step5 : scan edge (search edge 3) 4 [2347] 2347 root345Tail5 winner =
    scan edge (search edge 3) 4 [2347] 2347 root345Tail6 winner := by
  rw [root345Tail5, scan, if_neg (by decide), root345Candidates5_checked]
  change scan edge (search edge 3) 4 [2347] 2347 root345Tail6
    (if 3 ≤ root345Candidates5.length then search edge 3 [2347, 3067] root345Candidates5 5414 winner else winner) = _
  rw [if_neg (by decide)]
def root345Candidates6 : List Nat := [7873, 19717]
set_option maxRecDepth 8192 in
theorem root345Candidates6_checked : candidates edge 4 2347 3253 root345Tail7 winner.bound = root345Candidates6 := by
  unfold candidates
  have row : edge 3253 = (SortedNatTree.contains FastRows.row3253) := by
    change Data.row3253.contains = (SortedNatTree.contains FastRows.row3253)
    exact FastRows.row3253_checked.symm
  rw [row]
  decide
theorem root345Step6 : scan edge (search edge 3) 4 [2347] 2347 root345Tail6 winner =
    scan edge (search edge 3) 4 [2347] 2347 root345Tail7 winner := by
  rw [root345Tail6, scan, if_neg (by decide), root345Candidates6_checked]
  change scan edge (search edge 3) 4 [2347] 2347 root345Tail7
    (if 3 ≤ root345Candidates6.length then search edge 3 [2347, 3253] root345Candidates6 5600 winner else winner) = _
  rw [if_neg (by decide)]
def root345Candidates7 : List Nat := [9871]
set_option maxRecDepth 8192 in
theorem root345Candidates7_checked : candidates edge 4 2347 4327 root345Tail8 winner.bound = root345Candidates7 := by
  unfold candidates
  have row : edge 4327 = (SortedNatTree.contains FastRows.row4327) := by
    change Data.row4327.contains = (SortedNatTree.contains FastRows.row4327)
    exact FastRows.row4327_checked.symm
  rw [row]
  decide
theorem root345Step7 : scan edge (search edge 3) 4 [2347] 2347 root345Tail7 winner =
    scan edge (search edge 3) 4 [2347] 2347 root345Tail8 winner := by
  rw [root345Tail7, scan, if_neg (by decide), root345Candidates7_checked]
  change scan edge (search edge 3) 4 [2347] 2347 root345Tail8
    (if 3 ≤ root345Candidates7.length then search edge 3 [2347, 4327] root345Candidates7 6674 winner else winner) = _
  rw [if_neg (by decide)]
def root345Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root345Candidates8_checked : candidates edge 4 2347 4639 root345Tail9 winner.bound = root345Candidates8 := by
  unfold candidates
  have row : edge 4639 = (SortedNatTree.contains FastRows.row4639) := by
    change Data.row4639.contains = (SortedNatTree.contains FastRows.row4639)
    exact FastRows.row4639_checked.symm
  rw [row]
  decide
theorem root345Step8 : scan edge (search edge 3) 4 [2347] 2347 root345Tail8 winner =
    scan edge (search edge 3) 4 [2347] 2347 root345Tail9 winner := by
  rw [root345Tail8, scan, if_neg (by decide), root345Candidates8_checked]
  change scan edge (search edge 3) 4 [2347] 2347 root345Tail9
    (if 3 ≤ root345Candidates8.length then search edge 3 [2347, 4639] root345Candidates8 6986 winner else winner) = _
  rw [if_neg (by decide)]
def root345Candidates9 : List Nat := [7879, 14533]
set_option maxRecDepth 8192 in
theorem root345Candidates9_checked : candidates edge 4 2347 4663 root345Tail10 winner.bound = root345Candidates9 := by
  unfold candidates
  have row : edge 4663 = (SortedNatTree.contains FastRows.row4663) := by
    change Data.row4663.contains = (SortedNatTree.contains FastRows.row4663)
    exact FastRows.row4663_checked.symm
  rw [row]
  decide
theorem root345Step9 : scan edge (search edge 3) 4 [2347] 2347 root345Tail9 winner =
    scan edge (search edge 3) 4 [2347] 2347 root345Tail10 winner := by
  rw [root345Tail9, scan, if_neg (by decide), root345Candidates9_checked]
  change scan edge (search edge 3) 4 [2347] 2347 root345Tail10
    (if 3 ≤ root345Candidates9.length then search edge 3 [2347, 4663] root345Candidates9 7010 winner else winner) = _
  rw [if_neg (by decide)]
theorem root345Scan10 : scan edge (search edge 3) 4 [2347] 2347 root345Tail10 winner = winner := by rw [root345Tail10, scan, if_pos (by decide)]
theorem root345Scan9 : scan edge (search edge 3) 4 [2347] 2347 root345Tail9 winner = winner :=
  (root345Step9).trans root345Scan10
theorem root345Scan8 : scan edge (search edge 3) 4 [2347] 2347 root345Tail8 winner = winner :=
  (root345Step8).trans root345Scan9
theorem root345Scan7 : scan edge (search edge 3) 4 [2347] 2347 root345Tail7 winner = winner :=
  (root345Step7).trans root345Scan8
theorem root345Scan6 : scan edge (search edge 3) 4 [2347] 2347 root345Tail6 winner = winner :=
  (root345Step6).trans root345Scan7
theorem root345Scan5 : scan edge (search edge 3) 4 [2347] 2347 root345Tail5 winner = winner :=
  (root345Step5).trans root345Scan6
theorem root345Scan4 : scan edge (search edge 3) 4 [2347] 2347 root345Tail4 winner = winner :=
  (root345Step4).trans root345Scan5
theorem root345Scan3 : scan edge (search edge 3) 4 [2347] 2347 root345Tail3 winner = winner :=
  (root345Step3).trans root345Scan4
theorem root345Scan2 : scan edge (search edge 3) 4 [2347] 2347 root345Tail2 winner = winner :=
  (root345Step2).trans root345Scan3
theorem root345Scan1 : scan edge (search edge 3) 4 [2347] 2347 root345Tail1 winner = winner :=
  (root345Step1).trans root345Scan2
theorem root345Scan0 : scan edge (search edge 3) 4 [2347] 2347 root345Tail0 winner = winner :=
  (root345Step0).trans root345Scan1
set_option maxRecDepth 4096 in
theorem root345Branch : search edge 4 [2347] root345Candidates 2347 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2347] 2347 root345Tail0 winner = winner
  exact root345Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
