import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root206Candidates : List Nat := Data.row1289
set_option maxRecDepth 8192 in
theorem root206Candidates_checked : candidates edge 5 0 1289 tail207 winner.bound = root206Candidates := by
  apply candidates_eq_row edge 5 0 1289 tail207 root206Candidates winner.bound tail207_ordered
  · decide
  · decide
  · funext q
    rfl
def root206Tail40 : List Nat := []
def root206Tail39 : List Nat := 24359 :: root206Tail40
def root206Tail38 : List Nat := 24137 :: root206Tail39
def root206Tail37 : List Nat := 22571 :: root206Tail38
def root206Tail36 : List Nat := 22193 :: root206Tail37
def root206Tail35 : List Nat := 22109 :: root206Tail36
def root206Tail34 : List Nat := 21563 :: root206Tail35
def root206Tail33 : List Nat := 21227 :: root206Tail34
def root206Tail32 : List Nat := 20177 :: root206Tail33
def root206Tail31 : List Nat := 19727 :: root206Tail32
def root206Tail30 : List Nat := 19463 :: root206Tail31
def root206Tail29 : List Nat := 18047 :: root206Tail30
def root206Tail28 : List Nat := 17291 :: root206Tail29
def root206Tail27 : List Nat := 17027 :: root206Tail28
def root206Tail26 : List Nat := 15443 :: root206Tail27
def root206Tail25 : List Nat := 14249 :: root206Tail26
def root206Tail24 : List Nat := 13259 :: root206Tail25
def root206Tail23 : List Nat := 13241 :: root206Tail24
def root206Tail22 : List Nat := 12713 :: root206Tail23
def root206Tail21 : List Nat := 12611 :: root206Tail22
def root206Tail20 : List Nat := 11621 :: root206Tail21
def root206Tail19 : List Nat := 10781 :: root206Tail20
def root206Tail18 : List Nat := 10091 :: root206Tail19
def root206Tail17 : List Nat := 9431 :: root206Tail18
def root206Tail16 : List Nat := 9323 :: root206Tail17
def root206Tail15 : List Nat := 8933 :: root206Tail16
def root206Tail14 : List Nat := 7919 :: root206Tail15
def root206Tail13 : List Nat := 7853 :: root206Tail14
def root206Tail12 : List Nat := 7541 :: root206Tail13
def root206Tail11 : List Nat := 6449 :: root206Tail12
def root206Tail10 : List Nat := 6311 :: root206Tail11
def root206Tail9 : List Nat := 4733 :: root206Tail10
def root206Tail8 : List Nat := 4139 :: root206Tail9
def root206Tail7 : List Nat := 3833 :: root206Tail8
def root206Tail6 : List Nat := 3767 :: root206Tail7
def root206Tail5 : List Nat := 2801 :: root206Tail6
def root206Tail4 : List Nat := 2237 :: root206Tail5
def root206Tail3 : List Nat := 2039 :: root206Tail4
def root206Tail2 : List Nat := 2027 :: root206Tail3
def root206Tail1 : List Nat := 1721 :: root206Tail2
def root206Tail0 : List Nat := 1409 :: root206Tail1
def root206Candidates0 : List Nat := [12713]
set_option maxRecDepth 8192 in
theorem root206Candidates0_checked : candidates edge 4 1289 1409 root206Tail1 winner.bound = root206Candidates0 := by
  unfold candidates
  have row : edge 1409 = (SortedNatTree.contains FastRows.row1409) := by
    change Data.row1409.contains = (SortedNatTree.contains FastRows.row1409)
    exact FastRows.row1409_checked.symm
  rw [row]
  decide
theorem root206Step0 : scan edge (search edge 3) 4 [1289] 1289 root206Tail0 winner =
    scan edge (search edge 3) 4 [1289] 1289 root206Tail1 winner := by
  rw [root206Tail0, scan, if_neg (by decide), root206Candidates0_checked]
  change scan edge (search edge 3) 4 [1289] 1289 root206Tail1
    (if 3 ≤ root206Candidates0.length then search edge 3 [1289, 1409] root206Candidates0 2698 winner else winner) = _
  rw [if_neg (by decide)]
def root206Candidates1 : List Nat := [19463, 20177]
set_option maxRecDepth 8192 in
theorem root206Candidates1_checked : candidates edge 4 1289 1721 root206Tail2 winner.bound = root206Candidates1 := by
  unfold candidates
  have row : edge 1721 = (SortedNatTree.contains FastRows.row1721) := by
    change Data.row1721.contains = (SortedNatTree.contains FastRows.row1721)
    exact FastRows.row1721_checked.symm
  rw [row]
  decide
theorem root206Step1 : scan edge (search edge 3) 4 [1289] 1289 root206Tail1 winner =
    scan edge (search edge 3) 4 [1289] 1289 root206Tail2 winner := by
  rw [root206Tail1, scan, if_neg (by decide), root206Candidates1_checked]
  change scan edge (search edge 3) 4 [1289] 1289 root206Tail2
    (if 3 ≤ root206Candidates1.length then search edge 3 [1289, 1721] root206Candidates1 3010 winner else winner) = _
  rw [if_neg (by decide)]
def root206Candidates2 : List Nat := [12611]
set_option maxRecDepth 8192 in
theorem root206Candidates2_checked : candidates edge 4 1289 2027 root206Tail3 winner.bound = root206Candidates2 := by
  unfold candidates
  have row : edge 2027 = (SortedNatTree.contains FastRows.row2027) := by
    change Data.row2027.contains = (SortedNatTree.contains FastRows.row2027)
    exact FastRows.row2027_checked.symm
  rw [row]
  decide
theorem root206Step2 : scan edge (search edge 3) 4 [1289] 1289 root206Tail2 winner =
    scan edge (search edge 3) 4 [1289] 1289 root206Tail3 winner := by
  rw [root206Tail2, scan, if_neg (by decide), root206Candidates2_checked]
  change scan edge (search edge 3) 4 [1289] 1289 root206Tail3
    (if 3 ≤ root206Candidates2.length then search edge 3 [1289, 2027] root206Candidates2 3316 winner else winner) = _
  rw [if_neg (by decide)]
def root206Candidates3 : List Nat := [3767]
set_option maxRecDepth 8192 in
theorem root206Candidates3_checked : candidates edge 4 1289 2039 root206Tail4 winner.bound = root206Candidates3 := by
  unfold candidates
  have row : edge 2039 = (SortedNatTree.contains FastRows.row2039) := by
    change Data.row2039.contains = (SortedNatTree.contains FastRows.row2039)
    exact FastRows.row2039_checked.symm
  rw [row]
  decide
theorem root206Step3 : scan edge (search edge 3) 4 [1289] 1289 root206Tail3 winner =
    scan edge (search edge 3) 4 [1289] 1289 root206Tail4 winner := by
  rw [root206Tail3, scan, if_neg (by decide), root206Candidates3_checked]
  change scan edge (search edge 3) 4 [1289] 1289 root206Tail4
    (if 3 ≤ root206Candidates3.length then search edge 3 [1289, 2039] root206Candidates3 3328 winner else winner) = _
  rw [if_neg (by decide)]
def root206Candidates4 : List Nat := [4139, 9431, 13241]
set_option maxRecDepth 8192 in
theorem root206Candidates4_checked : candidates edge 4 1289 2237 root206Tail5 winner.bound = root206Candidates4 := by
  unfold candidates
  have row : edge 2237 = (SortedNatTree.contains FastRows.row2237) := by
    change Data.row2237.contains = (SortedNatTree.contains FastRows.row2237)
    exact FastRows.row2237_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root206Child4 : search edge 3 [1289, 2237] root206Candidates4 3526 winner = winner := by decide
theorem root206Step4 : scan edge (search edge 3) 4 [1289] 1289 root206Tail4 winner =
    scan edge (search edge 3) 4 [1289] 1289 root206Tail5 winner := by
  rw [root206Tail4, scan, if_neg (by decide), root206Candidates4_checked]
  change scan edge (search edge 3) 4 [1289] 1289 root206Tail5
    (if 3 ≤ root206Candidates4.length then search edge 3 [1289, 2237] root206Candidates4 3526 winner else winner) = _
  rw [if_pos (by decide), root206Child4]
def root206Candidates5 : List Nat := [4733, 6311, 7919, 20177]
set_option maxRecDepth 8192 in
theorem root206Candidates5_checked : candidates edge 4 1289 2801 root206Tail6 winner.bound = root206Candidates5 := by
  unfold candidates
  have row : edge 2801 = (SortedNatTree.contains FastRows.row2801) := by
    change Data.row2801.contains = (SortedNatTree.contains FastRows.row2801)
    exact FastRows.row2801_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root206Child5 : search edge 3 [1289, 2801] root206Candidates5 4090 winner = winner := by decide
theorem root206Step5 : scan edge (search edge 3) 4 [1289] 1289 root206Tail5 winner =
    scan edge (search edge 3) 4 [1289] 1289 root206Tail6 winner := by
  rw [root206Tail5, scan, if_neg (by decide), root206Candidates5_checked]
  change scan edge (search edge 3) 4 [1289] 1289 root206Tail6
    (if 3 ≤ root206Candidates5.length then search edge 3 [1289, 2801] root206Candidates5 4090 winner else winner) = _
  rw [if_pos (by decide), root206Child5]
def root206Candidates6 : List Nat := [8933, 19727]
set_option maxRecDepth 8192 in
theorem root206Candidates6_checked : candidates edge 4 1289 3767 root206Tail7 winner.bound = root206Candidates6 := by
  unfold candidates
  have row : edge 3767 = (SortedNatTree.contains FastRows.row3767) := by
    change Data.row3767.contains = (SortedNatTree.contains FastRows.row3767)
    exact FastRows.row3767_checked.symm
  rw [row]
  decide
theorem root206Step6 : scan edge (search edge 3) 4 [1289] 1289 root206Tail6 winner =
    scan edge (search edge 3) 4 [1289] 1289 root206Tail7 winner := by
  rw [root206Tail6, scan, if_neg (by decide), root206Candidates6_checked]
  change scan edge (search edge 3) 4 [1289] 1289 root206Tail7
    (if 3 ≤ root206Candidates6.length then search edge 3 [1289, 3767] root206Candidates6 5056 winner else winner) = _
  rw [if_neg (by decide)]
def root206Candidates7 : List Nat := [4733, 6449]
set_option maxRecDepth 8192 in
theorem root206Candidates7_checked : candidates edge 4 1289 3833 root206Tail8 winner.bound = root206Candidates7 := by
  unfold candidates
  have row : edge 3833 = (SortedNatTree.contains FastRows.row3833) := by
    change Data.row3833.contains = (SortedNatTree.contains FastRows.row3833)
    exact FastRows.row3833_checked.symm
  rw [row]
  decide
theorem root206Step7 : scan edge (search edge 3) 4 [1289] 1289 root206Tail7 winner =
    scan edge (search edge 3) 4 [1289] 1289 root206Tail8 winner := by
  rw [root206Tail7, scan, if_neg (by decide), root206Candidates7_checked]
  change scan edge (search edge 3) 4 [1289] 1289 root206Tail8
    (if 3 ≤ root206Candidates7.length then search edge 3 [1289, 3833] root206Candidates7 5122 winner else winner) = _
  rw [if_neg (by decide)]
def root206Candidates8 : List Nat := [7541, 15443]
set_option maxRecDepth 8192 in
theorem root206Candidates8_checked : candidates edge 4 1289 4139 root206Tail9 winner.bound = root206Candidates8 := by
  unfold candidates
  have row : edge 4139 = (SortedNatTree.contains FastRows.row4139) := by
    change Data.row4139.contains = (SortedNatTree.contains FastRows.row4139)
    exact FastRows.row4139_checked.symm
  rw [row]
  decide
theorem root206Step8 : scan edge (search edge 3) 4 [1289] 1289 root206Tail8 winner =
    scan edge (search edge 3) 4 [1289] 1289 root206Tail9 winner := by
  rw [root206Tail8, scan, if_neg (by decide), root206Candidates8_checked]
  change scan edge (search edge 3) 4 [1289] 1289 root206Tail9
    (if 3 ≤ root206Candidates8.length then search edge 3 [1289, 4139] root206Candidates8 5428 winner else winner) = _
  rw [if_neg (by decide)]
def root206Candidates9 : List Nat := [7919]
set_option maxRecDepth 8192 in
theorem root206Candidates9_checked : candidates edge 4 1289 4733 root206Tail10 winner.bound = root206Candidates9 := by
  unfold candidates
  have row : edge 4733 = (SortedNatTree.contains FastRows.row4733) := by
    change Data.row4733.contains = (SortedNatTree.contains FastRows.row4733)
    exact FastRows.row4733_checked.symm
  rw [row]
  decide
theorem root206Step9 : scan edge (search edge 3) 4 [1289] 1289 root206Tail9 winner =
    scan edge (search edge 3) 4 [1289] 1289 root206Tail10 winner := by
  rw [root206Tail9, scan, if_neg (by decide), root206Candidates9_checked]
  change scan edge (search edge 3) 4 [1289] 1289 root206Tail10
    (if 3 ≤ root206Candidates9.length then search edge 3 [1289, 4733] root206Candidates9 6022 winner else winner) = _
  rw [if_neg (by decide)]
theorem root206Scan10 : scan edge (search edge 3) 4 [1289] 1289 root206Tail10 winner = winner := by rw [root206Tail10, scan, if_pos (by decide)]
theorem root206Scan9 : scan edge (search edge 3) 4 [1289] 1289 root206Tail9 winner = winner :=
  (root206Step9).trans root206Scan10
theorem root206Scan8 : scan edge (search edge 3) 4 [1289] 1289 root206Tail8 winner = winner :=
  (root206Step8).trans root206Scan9
theorem root206Scan7 : scan edge (search edge 3) 4 [1289] 1289 root206Tail7 winner = winner :=
  (root206Step7).trans root206Scan8
theorem root206Scan6 : scan edge (search edge 3) 4 [1289] 1289 root206Tail6 winner = winner :=
  (root206Step6).trans root206Scan7
theorem root206Scan5 : scan edge (search edge 3) 4 [1289] 1289 root206Tail5 winner = winner :=
  (root206Step5).trans root206Scan6
theorem root206Scan4 : scan edge (search edge 3) 4 [1289] 1289 root206Tail4 winner = winner :=
  (root206Step4).trans root206Scan5
theorem root206Scan3 : scan edge (search edge 3) 4 [1289] 1289 root206Tail3 winner = winner :=
  (root206Step3).trans root206Scan4
theorem root206Scan2 : scan edge (search edge 3) 4 [1289] 1289 root206Tail2 winner = winner :=
  (root206Step2).trans root206Scan3
theorem root206Scan1 : scan edge (search edge 3) 4 [1289] 1289 root206Tail1 winner = winner :=
  (root206Step1).trans root206Scan2
theorem root206Scan0 : scan edge (search edge 3) 4 [1289] 1289 root206Tail0 winner = winner :=
  (root206Step0).trans root206Scan1
set_option maxRecDepth 4096 in
theorem root206Branch : search edge 4 [1289] root206Candidates 1289 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1289] 1289 root206Tail0 winner = winner
  exact root206Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
