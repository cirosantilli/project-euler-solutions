import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root202Candidates : List Nat := Data.row1259
set_option maxRecDepth 8192 in
theorem root202Candidates_checked : candidates edge 5 0 1259 tail203 winner.bound = root202Candidates := by
  apply candidates_eq_row edge 5 0 1259 tail203 root202Candidates winner.bound tail203_ordered
  · decide
  · decide
  · funext q
    rfl
def root202Tail40 : List Nat := []
def root202Tail39 : List Nat := 22817 :: root202Tail40
def root202Tail38 : List Nat := 21347 :: root202Tail39
def root202Tail37 : List Nat := 20903 :: root202Tail38
def root202Tail36 : List Nat := 20789 :: root202Tail37
def root202Tail35 : List Nat := 20177 :: root202Tail36
def root202Tail34 : List Nat := 18773 :: root202Tail35
def root202Tail33 : List Nat := 18059 :: root202Tail34
def root202Tail32 : List Nat := 17783 :: root202Tail33
def root202Tail31 : List Nat := 17669 :: root202Tail32
def root202Tail30 : List Nat := 17489 :: root202Tail31
def root202Tail29 : List Nat := 16787 :: root202Tail30
def root202Tail28 : List Nat := 14771 :: root202Tail29
def root202Tail27 : List Nat := 14753 :: root202Tail28
def root202Tail26 : List Nat := 14741 :: root202Tail27
def root202Tail25 : List Nat := 14447 :: root202Tail26
def root202Tail24 : List Nat := 14249 :: root202Tail25
def root202Tail23 : List Nat := 13721 :: root202Tail24
def root202Tail22 : List Nat := 12263 :: root202Tail23
def root202Tail21 : List Nat := 12239 :: root202Tail22
def root202Tail20 : List Nat := 12011 :: root202Tail21
def root202Tail19 : List Nat := 11351 :: root202Tail20
def root202Tail18 : List Nat := 11087 :: root202Tail19
def root202Tail17 : List Nat := 9689 :: root202Tail18
def root202Tail16 : List Nat := 9467 :: root202Tail17
def root202Tail15 : List Nat := 9437 :: root202Tail16
def root202Tail14 : List Nat := 8951 :: root202Tail15
def root202Tail13 : List Nat := 8291 :: root202Tail14
def root202Tail12 : List Nat := 7283 :: root202Tail13
def root202Tail11 : List Nat := 6959 :: root202Tail12
def root202Tail10 : List Nat := 6599 :: root202Tail11
def root202Tail9 : List Nat := 5507 :: root202Tail10
def root202Tail8 : List Nat := 5039 :: root202Tail9
def root202Tail7 : List Nat := 3989 :: root202Tail8
def root202Tail6 : List Nat := 2819 :: root202Tail7
def root202Tail5 : List Nat := 2633 :: root202Tail6
def root202Tail4 : List Nat := 2609 :: root202Tail5
def root202Tail3 : List Nat := 2357 :: root202Tail4
def root202Tail2 : List Nat := 2297 :: root202Tail3
def root202Tail1 : List Nat := 2273 :: root202Tail2
def root202Tail0 : List Nat := 2213 :: root202Tail1
def root202Candidates0 : List Nat := [14447, 18059]
set_option maxRecDepth 8192 in
theorem root202Candidates0_checked : candidates edge 4 1259 2213 root202Tail1 winner.bound = root202Candidates0 := by
  unfold candidates
  have row : edge 2213 = (SortedNatTree.contains FastRows.row2213) := by
    change Data.row2213.contains = (SortedNatTree.contains FastRows.row2213)
    exact FastRows.row2213_checked.symm
  rw [row]
  decide
theorem root202Step0 : scan edge (search edge 3) 4 [1259] 1259 root202Tail0 winner =
    scan edge (search edge 3) 4 [1259] 1259 root202Tail1 winner := by
  rw [root202Tail0, scan, if_neg (by decide), root202Candidates0_checked]
  change scan edge (search edge 3) 4 [1259] 1259 root202Tail1
    (if 3 ≤ root202Candidates0.length then search edge 3 [1259, 2213] root202Candidates0 3472 winner else winner) = _
  rw [if_neg (by decide)]
def root202Candidates1 : List Nat := [2609, 7283, 11087, 16787]
set_option maxRecDepth 8192 in
theorem root202Candidates1_checked : candidates edge 4 1259 2273 root202Tail2 winner.bound = root202Candidates1 := by
  unfold candidates
  have row : edge 2273 = (SortedNatTree.contains FastRows.row2273) := by
    change Data.row2273.contains = (SortedNatTree.contains FastRows.row2273)
    exact FastRows.row2273_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root202Child1 : search edge 3 [1259, 2273] root202Candidates1 3532 winner = winner := by decide
theorem root202Step1 : scan edge (search edge 3) 4 [1259] 1259 root202Tail1 winner =
    scan edge (search edge 3) 4 [1259] 1259 root202Tail2 winner := by
  rw [root202Tail1, scan, if_neg (by decide), root202Candidates1_checked]
  change scan edge (search edge 3) 4 [1259] 1259 root202Tail2
    (if 3 ≤ root202Candidates1.length then search edge 3 [1259, 2273] root202Candidates1 3532 winner else winner) = _
  rw [if_pos (by decide), root202Child1]
def root202Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root202Candidates2_checked : candidates edge 4 1259 2297 root202Tail3 winner.bound = root202Candidates2 := by
  unfold candidates
  have row : edge 2297 = (SortedNatTree.contains FastRows.row2297) := by
    change Data.row2297.contains = (SortedNatTree.contains FastRows.row2297)
    exact FastRows.row2297_checked.symm
  rw [row]
  decide
theorem root202Step2 : scan edge (search edge 3) 4 [1259] 1259 root202Tail2 winner =
    scan edge (search edge 3) 4 [1259] 1259 root202Tail3 winner := by
  rw [root202Tail2, scan, if_neg (by decide), root202Candidates2_checked]
  change scan edge (search edge 3) 4 [1259] 1259 root202Tail3
    (if 3 ≤ root202Candidates2.length then search edge 3 [1259, 2297] root202Candidates2 3556 winner else winner) = _
  rw [if_neg (by decide)]
def root202Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root202Candidates3_checked : candidates edge 4 1259 2357 root202Tail4 winner.bound = root202Candidates3 := by
  unfold candidates
  have row : edge 2357 = (SortedNatTree.contains FastRows.row2357) := by
    change Data.row2357.contains = (SortedNatTree.contains FastRows.row2357)
    exact FastRows.row2357_checked.symm
  rw [row]
  decide
theorem root202Step3 : scan edge (search edge 3) 4 [1259] 1259 root202Tail3 winner =
    scan edge (search edge 3) 4 [1259] 1259 root202Tail4 winner := by
  rw [root202Tail3, scan, if_neg (by decide), root202Candidates3_checked]
  change scan edge (search edge 3) 4 [1259] 1259 root202Tail4
    (if 3 ≤ root202Candidates3.length then search edge 3 [1259, 2357] root202Candidates3 3616 winner else winner) = _
  rw [if_neg (by decide)]
def root202Candidates4 : List Nat := [6599]
set_option maxRecDepth 8192 in
theorem root202Candidates4_checked : candidates edge 4 1259 2609 root202Tail5 winner.bound = root202Candidates4 := by
  unfold candidates
  have row : edge 2609 = (SortedNatTree.contains FastRows.row2609) := by
    change Data.row2609.contains = (SortedNatTree.contains FastRows.row2609)
    exact FastRows.row2609_checked.symm
  rw [row]
  decide
theorem root202Step4 : scan edge (search edge 3) 4 [1259] 1259 root202Tail4 winner =
    scan edge (search edge 3) 4 [1259] 1259 root202Tail5 winner := by
  rw [root202Tail4, scan, if_neg (by decide), root202Candidates4_checked]
  change scan edge (search edge 3) 4 [1259] 1259 root202Tail5
    (if 3 ≤ root202Candidates4.length then search edge 3 [1259, 2609] root202Candidates4 3868 winner else winner) = _
  rw [if_neg (by decide)]
def root202Candidates5 : List Nat := [14771]
set_option maxRecDepth 8192 in
theorem root202Candidates5_checked : candidates edge 4 1259 2633 root202Tail6 winner.bound = root202Candidates5 := by
  unfold candidates
  have row : edge 2633 = (SortedNatTree.contains FastRows.row2633) := by
    change Data.row2633.contains = (SortedNatTree.contains FastRows.row2633)
    exact FastRows.row2633_checked.symm
  rw [row]
  decide
theorem root202Step5 : scan edge (search edge 3) 4 [1259] 1259 root202Tail5 winner =
    scan edge (search edge 3) 4 [1259] 1259 root202Tail6 winner := by
  rw [root202Tail5, scan, if_neg (by decide), root202Candidates5_checked]
  change scan edge (search edge 3) 4 [1259] 1259 root202Tail6
    (if 3 ≤ root202Candidates5.length then search edge 3 [1259, 2633] root202Candidates5 3892 winner else winner) = _
  rw [if_neg (by decide)]
def root202Candidates6 : List Nat := [5039]
set_option maxRecDepth 8192 in
theorem root202Candidates6_checked : candidates edge 4 1259 2819 root202Tail7 winner.bound = root202Candidates6 := by
  unfold candidates
  have row : edge 2819 = (SortedNatTree.contains FastRows.row2819) := by
    change Data.row2819.contains = (SortedNatTree.contains FastRows.row2819)
    exact FastRows.row2819_checked.symm
  rw [row]
  decide
theorem root202Step6 : scan edge (search edge 3) 4 [1259] 1259 root202Tail6 winner =
    scan edge (search edge 3) 4 [1259] 1259 root202Tail7 winner := by
  rw [root202Tail6, scan, if_neg (by decide), root202Candidates6_checked]
  change scan edge (search edge 3) 4 [1259] 1259 root202Tail7
    (if 3 ≤ root202Candidates6.length then search edge 3 [1259, 2819] root202Candidates6 4078 winner else winner) = _
  rw [if_neg (by decide)]
def root202Candidates7 : List Nat := [8291, 20177]
set_option maxRecDepth 8192 in
theorem root202Candidates7_checked : candidates edge 4 1259 3989 root202Tail8 winner.bound = root202Candidates7 := by
  unfold candidates
  have row : edge 3989 = (SortedNatTree.contains FastRows.row3989) := by
    change Data.row3989.contains = (SortedNatTree.contains FastRows.row3989)
    exact FastRows.row3989_checked.symm
  rw [row]
  decide
theorem root202Step7 : scan edge (search edge 3) 4 [1259] 1259 root202Tail7 winner =
    scan edge (search edge 3) 4 [1259] 1259 root202Tail8 winner := by
  rw [root202Tail7, scan, if_neg (by decide), root202Candidates7_checked]
  change scan edge (search edge 3) 4 [1259] 1259 root202Tail8
    (if 3 ≤ root202Candidates7.length then search edge 3 [1259, 3989] root202Candidates7 5248 winner else winner) = _
  rw [if_neg (by decide)]
def root202Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root202Candidates8_checked : candidates edge 4 1259 5039 root202Tail9 winner.bound = root202Candidates8 := by
  unfold candidates
  have row : edge 5039 = (SortedNatTree.contains FastRows.row5039) := by
    change Data.row5039.contains = (SortedNatTree.contains FastRows.row5039)
    exact FastRows.row5039_checked.symm
  rw [row]
  decide
theorem root202Step8 : scan edge (search edge 3) 4 [1259] 1259 root202Tail8 winner =
    scan edge (search edge 3) 4 [1259] 1259 root202Tail9 winner := by
  rw [root202Tail8, scan, if_neg (by decide), root202Candidates8_checked]
  change scan edge (search edge 3) 4 [1259] 1259 root202Tail9
    (if 3 ≤ root202Candidates8.length then search edge 3 [1259, 5039] root202Candidates8 6298 winner else winner) = _
  rw [if_neg (by decide)]
def root202Candidates9 : List Nat := [8291]
set_option maxRecDepth 8192 in
theorem root202Candidates9_checked : candidates edge 4 1259 5507 root202Tail10 winner.bound = root202Candidates9 := by
  unfold candidates
  have row : edge 5507 = (SortedNatTree.contains FastRows.row5507) := by
    change Data.row5507.contains = (SortedNatTree.contains FastRows.row5507)
    exact FastRows.row5507_checked.symm
  rw [row]
  decide
theorem root202Step9 : scan edge (search edge 3) 4 [1259] 1259 root202Tail9 winner =
    scan edge (search edge 3) 4 [1259] 1259 root202Tail10 winner := by
  rw [root202Tail9, scan, if_neg (by decide), root202Candidates9_checked]
  change scan edge (search edge 3) 4 [1259] 1259 root202Tail10
    (if 3 ≤ root202Candidates9.length then search edge 3 [1259, 5507] root202Candidates9 6766 winner else winner) = _
  rw [if_neg (by decide)]
theorem root202Scan10 : scan edge (search edge 3) 4 [1259] 1259 root202Tail10 winner = winner := by rw [root202Tail10, scan, if_pos (by decide)]
theorem root202Scan9 : scan edge (search edge 3) 4 [1259] 1259 root202Tail9 winner = winner :=
  (root202Step9).trans root202Scan10
theorem root202Scan8 : scan edge (search edge 3) 4 [1259] 1259 root202Tail8 winner = winner :=
  (root202Step8).trans root202Scan9
theorem root202Scan7 : scan edge (search edge 3) 4 [1259] 1259 root202Tail7 winner = winner :=
  (root202Step7).trans root202Scan8
theorem root202Scan6 : scan edge (search edge 3) 4 [1259] 1259 root202Tail6 winner = winner :=
  (root202Step6).trans root202Scan7
theorem root202Scan5 : scan edge (search edge 3) 4 [1259] 1259 root202Tail5 winner = winner :=
  (root202Step5).trans root202Scan6
theorem root202Scan4 : scan edge (search edge 3) 4 [1259] 1259 root202Tail4 winner = winner :=
  (root202Step4).trans root202Scan5
theorem root202Scan3 : scan edge (search edge 3) 4 [1259] 1259 root202Tail3 winner = winner :=
  (root202Step3).trans root202Scan4
theorem root202Scan2 : scan edge (search edge 3) 4 [1259] 1259 root202Tail2 winner = winner :=
  (root202Step2).trans root202Scan3
theorem root202Scan1 : scan edge (search edge 3) 4 [1259] 1259 root202Tail1 winner = winner :=
  (root202Step1).trans root202Scan2
theorem root202Scan0 : scan edge (search edge 3) 4 [1259] 1259 root202Tail0 winner = winner :=
  (root202Step0).trans root202Scan1
set_option maxRecDepth 4096 in
theorem root202Branch : search edge 4 [1259] root202Candidates 1259 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1259] 1259 root202Tail0 winner = winner
  exact root202Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
