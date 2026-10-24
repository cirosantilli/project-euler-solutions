import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root213Candidates : List Nat := Data.row1321
set_option maxRecDepth 8192 in
theorem root213Candidates_checked : candidates edge 5 0 1321 tail214 winner.bound = root213Candidates := by
  apply candidates_eq_row edge 5 0 1321 tail214 root213Candidates winner.bound tail214_ordered
  · decide
  · decide
  · funext q
    rfl
def root213Tail43 : List Nat := []
def root213Tail42 : List Nat := 24631 :: root213Tail43
def root213Tail41 : List Nat := 24169 :: root213Tail42
def root213Tail40 : List Nat := 24061 :: root213Tail41
def root213Tail39 : List Nat := 22129 :: root213Tail40
def root213Tail38 : List Nat := 22027 :: root213Tail39
def root213Tail37 : List Nat := 21169 :: root213Tail38
def root213Tail36 : List Nat := 20929 :: root213Tail37
def root213Tail35 : List Nat := 19993 :: root213Tail36
def root213Tail34 : List Nat := 19423 :: root213Tail35
def root213Tail33 : List Nat := 18979 :: root213Tail34
def root213Tail32 : List Nat := 18199 :: root213Tail33
def root213Tail31 : List Nat := 18013 :: root213Tail32
def root213Tail30 : List Nat := 17239 :: root213Tail31
def root213Tail29 : List Nat := 16417 :: root213Tail30
def root213Tail28 : List Nat := 15679 :: root213Tail29
def root213Tail27 : List Nat := 15559 :: root213Tail28
def root213Tail26 : List Nat := 15391 :: root213Tail27
def root213Tail25 : List Nat := 15259 :: root213Tail26
def root213Tail24 : List Nat := 14779 :: root213Tail25
def root213Tail23 : List Nat := 14551 :: root213Tail24
def root213Tail22 : List Nat := 13249 :: root213Tail23
def root213Tail21 : List Nat := 12823 :: root213Tail22
def root213Tail20 : List Nat := 12601 :: root213Tail21
def root213Tail19 : List Nat := 12433 :: root213Tail20
def root213Tail18 : List Nat := 11821 :: root213Tail19
def root213Tail17 : List Nat := 11059 :: root213Tail18
def root213Tail16 : List Nat := 10267 :: root213Tail17
def root213Tail15 : List Nat := 9901 :: root213Tail16
def root213Tail14 : List Nat := 9337 :: root213Tail15
def root213Tail13 : List Nat := 8263 :: root213Tail14
def root213Tail12 : List Nat := 7759 :: root213Tail13
def root213Tail11 : List Nat := 7537 :: root213Tail12
def root213Tail10 : List Nat := 6607 :: root213Tail11
def root213Tail9 : List Nat := 5581 :: root213Tail10
def root213Tail8 : List Nat := 4507 :: root213Tail9
def root213Tail7 : List Nat := 4447 :: root213Tail8
def root213Tail6 : List Nat := 4273 :: root213Tail7
def root213Tail5 : List Nat := 4261 :: root213Tail6
def root213Tail4 : List Nat := 3001 :: root213Tail5
def root213Tail3 : List Nat := 2917 :: root213Tail4
def root213Tail2 : List Nat := 2803 :: root213Tail3
def root213Tail1 : List Nat := 2767 :: root213Tail2
def root213Tail0 : List Nat := 1381 :: root213Tail1
def root213Candidates0 : List Nat := [2767, 22129]
set_option maxRecDepth 8192 in
theorem root213Candidates0_checked : candidates edge 4 1321 1381 root213Tail1 winner.bound = root213Candidates0 := by
  unfold candidates
  have row : edge 1381 = (SortedNatTree.contains FastRows.row1381) := by
    change Data.row1381.contains = (SortedNatTree.contains FastRows.row1381)
    exact FastRows.row1381_checked.symm
  rw [row]
  decide
theorem root213Step0 : scan edge (search edge 3) 4 [1321] 1321 root213Tail0 winner =
    scan edge (search edge 3) 4 [1321] 1321 root213Tail1 winner := by
  rw [root213Tail0, scan, if_neg (by decide), root213Candidates0_checked]
  change scan edge (search edge 3) 4 [1321] 1321 root213Tail1
    (if 3 ≤ root213Candidates0.length then search edge 3 [1321, 1381] root213Candidates0 2702 winner else winner) = _
  rw [if_neg (by decide)]
def root213Candidates1 : List Nat := [7537, 14551, 15679]
set_option maxRecDepth 8192 in
theorem root213Candidates1_checked : candidates edge 4 1321 2767 root213Tail2 winner.bound = root213Candidates1 := by
  unfold candidates
  have row : edge 2767 = (SortedNatTree.contains FastRows.row2767) := by
    change Data.row2767.contains = (SortedNatTree.contains FastRows.row2767)
    exact FastRows.row2767_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root213Child1 : search edge 3 [1321, 2767] root213Candidates1 4088 winner = winner := by decide
theorem root213Step1 : scan edge (search edge 3) 4 [1321] 1321 root213Tail1 winner =
    scan edge (search edge 3) 4 [1321] 1321 root213Tail2 winner := by
  rw [root213Tail1, scan, if_neg (by decide), root213Candidates1_checked]
  change scan edge (search edge 3) 4 [1321] 1321 root213Tail2
    (if 3 ≤ root213Candidates1.length then search edge 3 [1321, 2767] root213Candidates1 4088 winner else winner) = _
  rw [if_pos (by decide), root213Child1]
def root213Candidates2 : List Nat := [4507]
set_option maxRecDepth 8192 in
theorem root213Candidates2_checked : candidates edge 4 1321 2803 root213Tail3 winner.bound = root213Candidates2 := by
  unfold candidates
  have row : edge 2803 = (SortedNatTree.contains FastRows.row2803) := by
    change Data.row2803.contains = (SortedNatTree.contains FastRows.row2803)
    exact FastRows.row2803_checked.symm
  rw [row]
  decide
theorem root213Step2 : scan edge (search edge 3) 4 [1321] 1321 root213Tail2 winner =
    scan edge (search edge 3) 4 [1321] 1321 root213Tail3 winner := by
  rw [root213Tail2, scan, if_neg (by decide), root213Candidates2_checked]
  change scan edge (search edge 3) 4 [1321] 1321 root213Tail3
    (if 3 ≤ root213Candidates2.length then search edge 3 [1321, 2803] root213Candidates2 4124 winner else winner) = _
  rw [if_neg (by decide)]
def root213Candidates3 : List Nat := [10267]
set_option maxRecDepth 8192 in
theorem root213Candidates3_checked : candidates edge 4 1321 2917 root213Tail4 winner.bound = root213Candidates3 := by
  unfold candidates
  have row : edge 2917 = (SortedNatTree.contains FastRows.row2917) := by
    change Data.row2917.contains = (SortedNatTree.contains FastRows.row2917)
    exact FastRows.row2917_checked.symm
  rw [row]
  decide
theorem root213Step3 : scan edge (search edge 3) 4 [1321] 1321 root213Tail3 winner =
    scan edge (search edge 3) 4 [1321] 1321 root213Tail4 winner := by
  rw [root213Tail3, scan, if_neg (by decide), root213Candidates3_checked]
  change scan edge (search edge 3) 4 [1321] 1321 root213Tail4
    (if 3 ≤ root213Candidates3.length then search edge 3 [1321, 2917] root213Candidates3 4238 winner else winner) = _
  rw [if_neg (by decide)]
def root213Candidates4 : List Nat := [6607, 10267, 11059]
set_option maxRecDepth 8192 in
theorem root213Candidates4_checked : candidates edge 4 1321 3001 root213Tail5 winner.bound = root213Candidates4 := by
  unfold candidates
  have row : edge 3001 = (SortedNatTree.contains FastRows.row3001) := by
    change Data.row3001.contains = (SortedNatTree.contains FastRows.row3001)
    exact FastRows.row3001_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root213Child4 : search edge 3 [1321, 3001] root213Candidates4 4322 winner = winner := by decide
theorem root213Step4 : scan edge (search edge 3) 4 [1321] 1321 root213Tail4 winner =
    scan edge (search edge 3) 4 [1321] 1321 root213Tail5 winner := by
  rw [root213Tail4, scan, if_neg (by decide), root213Candidates4_checked]
  change scan edge (search edge 3) 4 [1321] 1321 root213Tail5
    (if 3 ≤ root213Candidates4.length then search edge 3 [1321, 3001] root213Candidates4 4322 winner else winner) = _
  rw [if_pos (by decide), root213Child4]
def root213Candidates5 : List Nat := [12433]
set_option maxRecDepth 8192 in
theorem root213Candidates5_checked : candidates edge 4 1321 4261 root213Tail6 winner.bound = root213Candidates5 := by
  unfold candidates
  have row : edge 4261 = (SortedNatTree.contains FastRows.row4261) := by
    change Data.row4261.contains = (SortedNatTree.contains FastRows.row4261)
    exact FastRows.row4261_checked.symm
  rw [row]
  decide
theorem root213Step5 : scan edge (search edge 3) 4 [1321] 1321 root213Tail5 winner =
    scan edge (search edge 3) 4 [1321] 1321 root213Tail6 winner := by
  rw [root213Tail5, scan, if_neg (by decide), root213Candidates5_checked]
  change scan edge (search edge 3) 4 [1321] 1321 root213Tail6
    (if 3 ≤ root213Candidates5.length then search edge 3 [1321, 4261] root213Candidates5 5582 winner else winner) = _
  rw [if_neg (by decide)]
def root213Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root213Candidates6_checked : candidates edge 4 1321 4273 root213Tail7 winner.bound = root213Candidates6 := by
  unfold candidates
  have row : edge 4273 = (SortedNatTree.contains FastRows.row4273) := by
    change Data.row4273.contains = (SortedNatTree.contains FastRows.row4273)
    exact FastRows.row4273_checked.symm
  rw [row]
  decide
theorem root213Step6 : scan edge (search edge 3) 4 [1321] 1321 root213Tail6 winner =
    scan edge (search edge 3) 4 [1321] 1321 root213Tail7 winner := by
  rw [root213Tail6, scan, if_neg (by decide), root213Candidates6_checked]
  change scan edge (search edge 3) 4 [1321] 1321 root213Tail7
    (if 3 ≤ root213Candidates6.length then search edge 3 [1321, 4273] root213Candidates6 5594 winner else winner) = _
  rw [if_neg (by decide)]
def root213Candidates7 : List Nat := [15259, 17239]
set_option maxRecDepth 8192 in
theorem root213Candidates7_checked : candidates edge 4 1321 4447 root213Tail8 winner.bound = root213Candidates7 := by
  unfold candidates
  have row : edge 4447 = (SortedNatTree.contains FastRows.row4447) := by
    change Data.row4447.contains = (SortedNatTree.contains FastRows.row4447)
    exact FastRows.row4447_checked.symm
  rw [row]
  decide
theorem root213Step7 : scan edge (search edge 3) 4 [1321] 1321 root213Tail7 winner =
    scan edge (search edge 3) 4 [1321] 1321 root213Tail8 winner := by
  rw [root213Tail7, scan, if_neg (by decide), root213Candidates7_checked]
  change scan edge (search edge 3) 4 [1321] 1321 root213Tail8
    (if 3 ≤ root213Candidates7.length then search edge 3 [1321, 4447] root213Candidates7 5768 winner else winner) = _
  rw [if_neg (by decide)]
def root213Candidates8 : List Nat := [7759, 9901]
set_option maxRecDepth 8192 in
theorem root213Candidates8_checked : candidates edge 4 1321 4507 root213Tail9 winner.bound = root213Candidates8 := by
  unfold candidates
  have row : edge 4507 = (SortedNatTree.contains FastRows.row4507) := by
    change Data.row4507.contains = (SortedNatTree.contains FastRows.row4507)
    exact FastRows.row4507_checked.symm
  rw [row]
  decide
theorem root213Step8 : scan edge (search edge 3) 4 [1321] 1321 root213Tail8 winner =
    scan edge (search edge 3) 4 [1321] 1321 root213Tail9 winner := by
  rw [root213Tail8, scan, if_neg (by decide), root213Candidates8_checked]
  change scan edge (search edge 3) 4 [1321] 1321 root213Tail9
    (if 3 ≤ root213Candidates8.length then search edge 3 [1321, 4507] root213Candidates8 5828 winner else winner) = _
  rw [if_neg (by decide)]
def root213Candidates9 : List Nat := [15391]
set_option maxRecDepth 8192 in
theorem root213Candidates9_checked : candidates edge 4 1321 5581 root213Tail10 winner.bound = root213Candidates9 := by
  unfold candidates
  have row : edge 5581 = (SortedNatTree.contains FastRows.row5581) := by
    change Data.row5581.contains = (SortedNatTree.contains FastRows.row5581)
    exact FastRows.row5581_checked.symm
  rw [row]
  decide
theorem root213Step9 : scan edge (search edge 3) 4 [1321] 1321 root213Tail9 winner =
    scan edge (search edge 3) 4 [1321] 1321 root213Tail10 winner := by
  rw [root213Tail9, scan, if_neg (by decide), root213Candidates9_checked]
  change scan edge (search edge 3) 4 [1321] 1321 root213Tail10
    (if 3 ≤ root213Candidates9.length then search edge 3 [1321, 5581] root213Candidates9 6902 winner else winner) = _
  rw [if_neg (by decide)]
theorem root213Scan10 : scan edge (search edge 3) 4 [1321] 1321 root213Tail10 winner = winner := by rw [root213Tail10, scan, if_pos (by decide)]
theorem root213Scan9 : scan edge (search edge 3) 4 [1321] 1321 root213Tail9 winner = winner :=
  (root213Step9).trans root213Scan10
theorem root213Scan8 : scan edge (search edge 3) 4 [1321] 1321 root213Tail8 winner = winner :=
  (root213Step8).trans root213Scan9
theorem root213Scan7 : scan edge (search edge 3) 4 [1321] 1321 root213Tail7 winner = winner :=
  (root213Step7).trans root213Scan8
theorem root213Scan6 : scan edge (search edge 3) 4 [1321] 1321 root213Tail6 winner = winner :=
  (root213Step6).trans root213Scan7
theorem root213Scan5 : scan edge (search edge 3) 4 [1321] 1321 root213Tail5 winner = winner :=
  (root213Step5).trans root213Scan6
theorem root213Scan4 : scan edge (search edge 3) 4 [1321] 1321 root213Tail4 winner = winner :=
  (root213Step4).trans root213Scan5
theorem root213Scan3 : scan edge (search edge 3) 4 [1321] 1321 root213Tail3 winner = winner :=
  (root213Step3).trans root213Scan4
theorem root213Scan2 : scan edge (search edge 3) 4 [1321] 1321 root213Tail2 winner = winner :=
  (root213Step2).trans root213Scan3
theorem root213Scan1 : scan edge (search edge 3) 4 [1321] 1321 root213Tail1 winner = winner :=
  (root213Step1).trans root213Scan2
theorem root213Scan0 : scan edge (search edge 3) 4 [1321] 1321 root213Tail0 winner = winner :=
  (root213Step0).trans root213Scan1
set_option maxRecDepth 4096 in
theorem root213Branch : search edge 4 [1321] root213Candidates 1321 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1321] 1321 root213Tail0 winner = winner
  exact root213Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
