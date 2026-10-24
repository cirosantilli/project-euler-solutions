import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root448Candidates : List Nat := Data.row3187
set_option maxRecDepth 8192 in
theorem root448Candidates_checked : candidates edge 5 0 3187 tail449 winner.bound = root448Candidates := by
  apply candidates_eq_row edge 5 0 3187 tail449 root448Candidates winner.bound tail449_ordered
  · decide
  · decide
  · funext q
    rfl
def root448Tail49 : List Nat := []
def root448Tail48 : List Nat := 22153 :: root448Tail49
def root448Tail47 : List Nat := 21817 :: root448Tail48
def root448Tail46 : List Nat := 21019 :: root448Tail47
def root448Tail45 : List Nat := 20887 :: root448Tail46
def root448Tail44 : List Nat := 20599 :: root448Tail45
def root448Tail43 : List Nat := 20173 :: root448Tail44
def root448Tail42 : List Nat := 20161 :: root448Tail43
def root448Tail41 : List Nat := 20071 :: root448Tail42
def root448Tail40 : List Nat := 19543 :: root448Tail41
def root448Tail39 : List Nat := 19447 :: root448Tail40
def root448Tail38 : List Nat := 19273 :: root448Tail39
def root448Tail37 : List Nat := 19207 :: root448Tail38
def root448Tail36 : List Nat := 18181 :: root448Tail37
def root448Tail35 : List Nat := 17551 :: root448Tail36
def root448Tail34 : List Nat := 17341 :: root448Tail35
def root448Tail33 : List Nat := 17299 :: root448Tail34
def root448Tail32 : List Nat := 17137 :: root448Tail33
def root448Tail31 : List Nat := 15259 :: root448Tail32
def root448Tail30 : List Nat := 14947 :: root448Tail31
def root448Tail29 : List Nat := 13147 :: root448Tail30
def root448Tail28 : List Nat := 12637 :: root448Tail29
def root448Tail27 : List Nat := 11593 :: root448Tail28
def root448Tail26 : List Nat := 11587 :: root448Tail27
def root448Tail25 : List Nat := 11353 :: root448Tail26
def root448Tail24 : List Nat := 10159 :: root448Tail25
def root448Tail23 : List Nat := 9643 :: root448Tail24
def root448Tail22 : List Nat := 9391 :: root448Tail23
def root448Tail21 : List Nat := 9151 :: root448Tail22
def root448Tail20 : List Nat := 9109 :: root448Tail21
def root448Tail19 : List Nat := 9049 :: root448Tail20
def root448Tail18 : List Nat := 8563 :: root448Tail19
def root448Tail17 : List Nat := 8527 :: root448Tail18
def root448Tail16 : List Nat := 8233 :: root448Tail17
def root448Tail15 : List Nat := 8101 :: root448Tail16
def root448Tail14 : List Nat := 8017 :: root448Tail15
def root448Tail13 : List Nat := 7477 :: root448Tail14
def root448Tail12 : List Nat := 7411 :: root448Tail13
def root448Tail11 : List Nat := 6907 :: root448Tail12
def root448Tail10 : List Nat := 6379 :: root448Tail11
def root448Tail9 : List Nat := 5479 :: root448Tail10
def root448Tail8 : List Nat := 5227 :: root448Tail9
def root448Tail7 : List Nat := 5209 :: root448Tail8
def root448Tail6 : List Nat := 4957 :: root448Tail7
def root448Tail5 : List Nat := 4729 :: root448Tail6
def root448Tail4 : List Nat := 4657 :: root448Tail5
def root448Tail3 : List Nat := 4561 :: root448Tail4
def root448Tail2 : List Nat := 4093 :: root448Tail3
def root448Tail1 : List Nat := 3529 :: root448Tail2
def root448Tail0 : List Nat := 3229 :: root448Tail1
def root448Candidates0 : List Nat := [3529, 8527, 9109, 9151]
set_option maxRecDepth 8192 in
theorem root448Candidates0_checked : candidates edge 4 3187 3229 root448Tail1 winner.bound = root448Candidates0 := by
  unfold candidates
  have row : edge 3229 = (SortedNatTree.contains FastRows.row3229) := by
    change Data.row3229.contains = (SortedNatTree.contains FastRows.row3229)
    exact FastRows.row3229_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root448Child0 : search edge 3 [3187, 3229] root448Candidates0 6416 winner = winner := by decide
theorem root448Step0 : scan edge (search edge 3) 4 [3187] 3187 root448Tail0 winner =
    scan edge (search edge 3) 4 [3187] 3187 root448Tail1 winner := by
  rw [root448Tail0, scan, if_neg (by decide), root448Candidates0_checked]
  change scan edge (search edge 3) 4 [3187] 3187 root448Tail1
    (if 3 ≤ root448Candidates0.length then search edge 3 [3187, 3229] root448Candidates0 6416 winner else winner) = _
  rw [if_pos (by decide), root448Child0]
def root448Candidates1 : List Nat := [4657]
set_option maxRecDepth 8192 in
theorem root448Candidates1_checked : candidates edge 4 3187 3529 root448Tail2 winner.bound = root448Candidates1 := by
  unfold candidates
  have row : edge 3529 = (SortedNatTree.contains FastRows.row3529) := by
    change Data.row3529.contains = (SortedNatTree.contains FastRows.row3529)
    exact FastRows.row3529_checked.symm
  rw [row]
  decide
theorem root448Step1 : scan edge (search edge 3) 4 [3187] 3187 root448Tail1 winner =
    scan edge (search edge 3) 4 [3187] 3187 root448Tail2 winner := by
  rw [root448Tail1, scan, if_neg (by decide), root448Candidates1_checked]
  change scan edge (search edge 3) 4 [3187] 3187 root448Tail2
    (if 3 ≤ root448Candidates1.length then search edge 3 [3187, 3529] root448Candidates1 6716 winner else winner) = _
  rw [if_neg (by decide)]
def root448Candidates2 : List Nat := [11587, 18181]
set_option maxRecDepth 8192 in
theorem root448Candidates2_checked : candidates edge 4 3187 4093 root448Tail3 winner.bound = root448Candidates2 := by
  unfold candidates
  have row : edge 4093 = (SortedNatTree.contains FastRows.row4093) := by
    change Data.row4093.contains = (SortedNatTree.contains FastRows.row4093)
    exact FastRows.row4093_checked.symm
  rw [row]
  decide
theorem root448Step2 : scan edge (search edge 3) 4 [3187] 3187 root448Tail2 winner =
    scan edge (search edge 3) 4 [3187] 3187 root448Tail3 winner := by
  rw [root448Tail2, scan, if_neg (by decide), root448Candidates2_checked]
  change scan edge (search edge 3) 4 [3187] 3187 root448Tail3
    (if 3 ≤ root448Candidates2.length then search edge 3 [3187, 4093] root448Candidates2 7280 winner else winner) = _
  rw [if_neg (by decide)]
def root448Candidates3 : List Nat := [6379, 8017, 8101]
set_option maxRecDepth 8192 in
theorem root448Candidates3_checked : candidates edge 4 3187 4561 root448Tail4 winner.bound = root448Candidates3 := by
  unfold candidates
  have row : edge 4561 = (SortedNatTree.contains FastRows.row4561) := by
    change Data.row4561.contains = (SortedNatTree.contains FastRows.row4561)
    exact FastRows.row4561_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root448Child3 : search edge 3 [3187, 4561] root448Candidates3 7748 winner = winner := by decide
theorem root448Step3 : scan edge (search edge 3) 4 [3187] 3187 root448Tail3 winner =
    scan edge (search edge 3) 4 [3187] 3187 root448Tail4 winner := by
  rw [root448Tail3, scan, if_neg (by decide), root448Candidates3_checked]
  change scan edge (search edge 3) 4 [3187] 3187 root448Tail4
    (if 3 ≤ root448Candidates3.length then search edge 3 [3187, 4561] root448Candidates3 7748 winner else winner) = _
  rw [if_pos (by decide), root448Child3]
def root448Candidates4 : List Nat := [5227]
set_option maxRecDepth 8192 in
theorem root448Candidates4_checked : candidates edge 4 3187 4657 root448Tail5 winner.bound = root448Candidates4 := by
  unfold candidates
  have row : edge 4657 = (SortedNatTree.contains FastRows.row4657) := by
    change Data.row4657.contains = (SortedNatTree.contains FastRows.row4657)
    exact FastRows.row4657_checked.symm
  rw [row]
  decide
theorem root448Step4 : scan edge (search edge 3) 4 [3187] 3187 root448Tail4 winner =
    scan edge (search edge 3) 4 [3187] 3187 root448Tail5 winner := by
  rw [root448Tail4, scan, if_neg (by decide), root448Candidates4_checked]
  change scan edge (search edge 3) 4 [3187] 3187 root448Tail5
    (if 3 ≤ root448Candidates4.length then search edge 3 [3187, 4657] root448Candidates4 7844 winner else winner) = _
  rw [if_neg (by decide)]
def root448Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root448Candidates5_checked : candidates edge 4 3187 4729 root448Tail6 winner.bound = root448Candidates5 := by
  unfold candidates
  have row : edge 4729 = (SortedNatTree.contains FastRows.row4729) := by
    change Data.row4729.contains = (SortedNatTree.contains FastRows.row4729)
    exact FastRows.row4729_checked.symm
  rw [row]
  decide
theorem root448Step5 : scan edge (search edge 3) 4 [3187] 3187 root448Tail5 winner =
    scan edge (search edge 3) 4 [3187] 3187 root448Tail6 winner := by
  rw [root448Tail5, scan, if_neg (by decide), root448Candidates5_checked]
  change scan edge (search edge 3) 4 [3187] 3187 root448Tail6
    (if 3 ≤ root448Candidates5.length then search edge 3 [3187, 4729] root448Candidates5 7916 winner else winner) = _
  rw [if_neg (by decide)]
def root448Candidates6 : List Nat := [8017, 8233]
set_option maxRecDepth 8192 in
theorem root448Candidates6_checked : candidates edge 4 3187 4957 root448Tail7 winner.bound = root448Candidates6 := by
  unfold candidates
  have row : edge 4957 = (SortedNatTree.contains FastRows.row4957) := by
    change Data.row4957.contains = (SortedNatTree.contains FastRows.row4957)
    exact FastRows.row4957_checked.symm
  rw [row]
  decide
theorem root448Step6 : scan edge (search edge 3) 4 [3187] 3187 root448Tail6 winner =
    scan edge (search edge 3) 4 [3187] 3187 root448Tail7 winner := by
  rw [root448Tail6, scan, if_neg (by decide), root448Candidates6_checked]
  change scan edge (search edge 3) 4 [3187] 3187 root448Tail7
    (if 3 ≤ root448Candidates6.length then search edge 3 [3187, 4957] root448Candidates6 8144 winner else winner) = _
  rw [if_neg (by decide)]
def root448Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root448Candidates7_checked : candidates edge 4 3187 5209 root448Tail8 winner.bound = root448Candidates7 := by
  unfold candidates
  have row : edge 5209 = (SortedNatTree.contains FastRows.row5209) := by
    change Data.row5209.contains = (SortedNatTree.contains FastRows.row5209)
    exact FastRows.row5209_checked.symm
  rw [row]
  decide
theorem root448Step7 : scan edge (search edge 3) 4 [3187] 3187 root448Tail7 winner =
    scan edge (search edge 3) 4 [3187] 3187 root448Tail8 winner := by
  rw [root448Tail7, scan, if_neg (by decide), root448Candidates7_checked]
  change scan edge (search edge 3) 4 [3187] 3187 root448Tail8
    (if 3 ≤ root448Candidates7.length then search edge 3 [3187, 5209] root448Candidates7 8396 winner else winner) = _
  rw [if_neg (by decide)]
def root448Candidates8 : List Nat := [6379, 17137, 17551]
set_option maxRecDepth 8192 in
theorem root448Candidates8_checked : candidates edge 4 3187 5227 root448Tail9 winner.bound = root448Candidates8 := by
  unfold candidates
  have row : edge 5227 = (SortedNatTree.contains FastRows.row5227) := by
    change Data.row5227.contains = (SortedNatTree.contains FastRows.row5227)
    exact FastRows.row5227_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root448Child8 : search edge 3 [3187, 5227] root448Candidates8 8414 winner = winner := by decide
theorem root448Step8 : scan edge (search edge 3) 4 [3187] 3187 root448Tail8 winner =
    scan edge (search edge 3) 4 [3187] 3187 root448Tail9 winner := by
  rw [root448Tail8, scan, if_neg (by decide), root448Candidates8_checked]
  change scan edge (search edge 3) 4 [3187] 3187 root448Tail9
    (if 3 ≤ root448Candidates8.length then search edge 3 [3187, 5227] root448Candidates8 8414 winner else winner) = _
  rw [if_pos (by decide), root448Child8]
def root448Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root448Candidates9_checked : candidates edge 4 3187 5479 root448Tail10 winner.bound = root448Candidates9 := by
  unfold candidates
  have row : edge 5479 = (SortedNatTree.contains FastRows.row5479) := by
    change Data.row5479.contains = (SortedNatTree.contains FastRows.row5479)
    exact FastRows.row5479_checked.symm
  rw [row]
  decide
theorem root448Step9 : scan edge (search edge 3) 4 [3187] 3187 root448Tail9 winner =
    scan edge (search edge 3) 4 [3187] 3187 root448Tail10 winner := by
  rw [root448Tail9, scan, if_neg (by decide), root448Candidates9_checked]
  change scan edge (search edge 3) 4 [3187] 3187 root448Tail10
    (if 3 ≤ root448Candidates9.length then search edge 3 [3187, 5479] root448Candidates9 8666 winner else winner) = _
  rw [if_neg (by decide)]
theorem root448Scan10 : scan edge (search edge 3) 4 [3187] 3187 root448Tail10 winner = winner := by rw [root448Tail10, scan, if_pos (by decide)]
theorem root448Scan9 : scan edge (search edge 3) 4 [3187] 3187 root448Tail9 winner = winner :=
  (root448Step9).trans root448Scan10
theorem root448Scan8 : scan edge (search edge 3) 4 [3187] 3187 root448Tail8 winner = winner :=
  (root448Step8).trans root448Scan9
theorem root448Scan7 : scan edge (search edge 3) 4 [3187] 3187 root448Tail7 winner = winner :=
  (root448Step7).trans root448Scan8
theorem root448Scan6 : scan edge (search edge 3) 4 [3187] 3187 root448Tail6 winner = winner :=
  (root448Step6).trans root448Scan7
theorem root448Scan5 : scan edge (search edge 3) 4 [3187] 3187 root448Tail5 winner = winner :=
  (root448Step5).trans root448Scan6
theorem root448Scan4 : scan edge (search edge 3) 4 [3187] 3187 root448Tail4 winner = winner :=
  (root448Step4).trans root448Scan5
theorem root448Scan3 : scan edge (search edge 3) 4 [3187] 3187 root448Tail3 winner = winner :=
  (root448Step3).trans root448Scan4
theorem root448Scan2 : scan edge (search edge 3) 4 [3187] 3187 root448Tail2 winner = winner :=
  (root448Step2).trans root448Scan3
theorem root448Scan1 : scan edge (search edge 3) 4 [3187] 3187 root448Tail1 winner = winner :=
  (root448Step1).trans root448Scan2
theorem root448Scan0 : scan edge (search edge 3) 4 [3187] 3187 root448Tail0 winner = winner :=
  (root448Step0).trans root448Scan1
set_option maxRecDepth 4096 in
theorem root448Branch : search edge 4 [3187] root448Candidates 3187 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3187] 3187 root448Tail0 winner = winner
  exact root448Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
