import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root267Candidates : List Nat := Data.row1733
set_option maxRecDepth 8192 in
theorem root267Candidates_checked : candidates edge 5 0 1733 tail268 winner.bound = root267Candidates := by
  apply candidates_eq_row edge 5 0 1733 tail268 root267Candidates winner.bound tail268_ordered
  · decide
  · decide
  · funext q
    rfl
def root267Tail47 : List Nat := []
def root267Tail46 : List Nat := 24083 :: root267Tail47
def root267Tail45 : List Nat := 23531 :: root267Tail46
def root267Tail44 : List Nat := 23327 :: root267Tail45
def root267Tail43 : List Nat := 22901 :: root267Tail44
def root267Tail42 : List Nat := 21611 :: root267Tail43
def root267Tail41 : List Nat := 21149 :: root267Tail42
def root267Tail40 : List Nat := 20183 :: root267Tail41
def root267Tail39 : List Nat := 19379 :: root267Tail40
def root267Tail38 : List Nat := 18911 :: root267Tail39
def root267Tail37 : List Nat := 18503 :: root267Tail38
def root267Tail36 : List Nat := 18329 :: root267Tail37
def root267Tail35 : List Nat := 17393 :: root267Tail36
def root267Tail34 : List Nat := 16607 :: root267Tail35
def root267Tail33 : List Nat := 16427 :: root267Tail34
def root267Tail32 : List Nat := 14891 :: root267Tail33
def root267Tail31 : List Nat := 13457 :: root267Tail32
def root267Tail30 : List Nat := 12401 :: root267Tail31
def root267Tail29 : List Nat := 12203 :: root267Tail30
def root267Tail28 : List Nat := 11273 :: root267Tail29
def root267Tail27 : List Nat := 10463 :: root267Tail28
def root267Tail26 : List Nat := 10337 :: root267Tail27
def root267Tail25 : List Nat := 9929 :: root267Tail26
def root267Tail24 : List Nat := 9719 :: root267Tail25
def root267Tail23 : List Nat := 8969 :: root267Tail24
def root267Tail22 : List Nat := 8807 :: root267Tail23
def root267Tail21 : List Nat := 8627 :: root267Tail22
def root267Tail20 : List Nat := 7949 :: root267Tail21
def root267Tail19 : List Nat := 7757 :: root267Tail20
def root267Tail18 : List Nat := 7043 :: root267Tail19
def root267Tail17 : List Nat := 6959 :: root267Tail18
def root267Tail16 : List Nat := 6521 :: root267Tail17
def root267Tail15 : List Nat := 6323 :: root267Tail16
def root267Tail14 : List Nat := 6101 :: root267Tail15
def root267Tail13 : List Nat := 5897 :: root267Tail14
def root267Tail12 : List Nat := 5867 :: root267Tail13
def root267Tail11 : List Nat := 5273 :: root267Tail12
def root267Tail10 : List Nat := 5051 :: root267Tail11
def root267Tail9 : List Nat := 4817 :: root267Tail10
def root267Tail8 : List Nat := 4133 :: root267Tail9
def root267Tail7 : List Nat := 4001 :: root267Tail8
def root267Tail6 : List Nat := 3917 :: root267Tail7
def root267Tail5 : List Nat := 3593 :: root267Tail6
def root267Tail4 : List Nat := 3347 :: root267Tail5
def root267Tail3 : List Nat := 2927 :: root267Tail4
def root267Tail2 : List Nat := 2663 :: root267Tail3
def root267Tail1 : List Nat := 2657 :: root267Tail2
def root267Tail0 : List Nat := 2411 :: root267Tail1
def root267Candidates0 : List Nat := [3917]
set_option maxRecDepth 8192 in
theorem root267Candidates0_checked : candidates edge 4 1733 2411 root267Tail1 winner.bound = root267Candidates0 := by
  unfold candidates
  have row : edge 2411 = (SortedNatTree.contains FastRows.row2411) := by
    change Data.row2411.contains = (SortedNatTree.contains FastRows.row2411)
    exact FastRows.row2411_checked.symm
  rw [row]
  decide
theorem root267Step0 : scan edge (search edge 3) 4 [1733] 1733 root267Tail0 winner =
    scan edge (search edge 3) 4 [1733] 1733 root267Tail1 winner := by
  rw [root267Tail0, scan, if_neg (by decide), root267Candidates0_checked]
  change scan edge (search edge 3) 4 [1733] 1733 root267Tail1
    (if 3 ≤ root267Candidates0.length then search edge 3 [1733, 2411] root267Candidates0 4144 winner else winner) = _
  rw [if_neg (by decide)]
def root267Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root267Candidates1_checked : candidates edge 4 1733 2657 root267Tail2 winner.bound = root267Candidates1 := by
  unfold candidates
  have row : edge 2657 = (SortedNatTree.contains FastRows.row2657) := by
    change Data.row2657.contains = (SortedNatTree.contains FastRows.row2657)
    exact FastRows.row2657_checked.symm
  rw [row]
  decide
theorem root267Step1 : scan edge (search edge 3) 4 [1733] 1733 root267Tail1 winner =
    scan edge (search edge 3) 4 [1733] 1733 root267Tail2 winner := by
  rw [root267Tail1, scan, if_neg (by decide), root267Candidates1_checked]
  change scan edge (search edge 3) 4 [1733] 1733 root267Tail2
    (if 3 ≤ root267Candidates1.length then search edge 3 [1733, 2657] root267Candidates1 4390 winner else winner) = _
  rw [if_neg (by decide)]
def root267Candidates2 : List Nat := [14891]
set_option maxRecDepth 8192 in
theorem root267Candidates2_checked : candidates edge 4 1733 2663 root267Tail3 winner.bound = root267Candidates2 := by
  unfold candidates
  have row : edge 2663 = (SortedNatTree.contains FastRows.row2663) := by
    change Data.row2663.contains = (SortedNatTree.contains FastRows.row2663)
    exact FastRows.row2663_checked.symm
  rw [row]
  decide
theorem root267Step2 : scan edge (search edge 3) 4 [1733] 1733 root267Tail2 winner =
    scan edge (search edge 3) 4 [1733] 1733 root267Tail3 winner := by
  rw [root267Tail2, scan, if_neg (by decide), root267Candidates2_checked]
  change scan edge (search edge 3) 4 [1733] 1733 root267Tail3
    (if 3 ≤ root267Candidates2.length then search edge 3 [1733, 2663] root267Candidates2 4396 winner else winner) = _
  rw [if_neg (by decide)]
def root267Candidates3 : List Nat := [3593, 13457]
set_option maxRecDepth 8192 in
theorem root267Candidates3_checked : candidates edge 4 1733 2927 root267Tail4 winner.bound = root267Candidates3 := by
  unfold candidates
  have row : edge 2927 = (SortedNatTree.contains FastRows.row2927) := by
    change Data.row2927.contains = (SortedNatTree.contains FastRows.row2927)
    exact FastRows.row2927_checked.symm
  rw [row]
  decide
theorem root267Step3 : scan edge (search edge 3) 4 [1733] 1733 root267Tail3 winner =
    scan edge (search edge 3) 4 [1733] 1733 root267Tail4 winner := by
  rw [root267Tail3, scan, if_neg (by decide), root267Candidates3_checked]
  change scan edge (search edge 3) 4 [1733] 1733 root267Tail4
    (if 3 ≤ root267Candidates3.length then search edge 3 [1733, 2927] root267Candidates3 4660 winner else winner) = _
  rw [if_neg (by decide)]
def root267Candidates4 : List Nat := [6101, 6959, 7043]
set_option maxRecDepth 8192 in
theorem root267Candidates4_checked : candidates edge 4 1733 3347 root267Tail5 winner.bound = root267Candidates4 := by
  unfold candidates
  have row : edge 3347 = (SortedNatTree.contains FastRows.row3347) := by
    change Data.row3347.contains = (SortedNatTree.contains FastRows.row3347)
    exact FastRows.row3347_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root267Child4 : search edge 3 [1733, 3347] root267Candidates4 5080 winner = winner := by decide
theorem root267Step4 : scan edge (search edge 3) 4 [1733] 1733 root267Tail4 winner =
    scan edge (search edge 3) 4 [1733] 1733 root267Tail5 winner := by
  rw [root267Tail4, scan, if_neg (by decide), root267Candidates4_checked]
  change scan edge (search edge 3) 4 [1733] 1733 root267Tail5
    (if 3 ≤ root267Candidates4.length then search edge 3 [1733, 3347] root267Candidates4 5080 winner else winner) = _
  rw [if_pos (by decide), root267Child4]
def root267Candidates5 : List Nat := [4001, 8807]
set_option maxRecDepth 8192 in
theorem root267Candidates5_checked : candidates edge 4 1733 3593 root267Tail6 winner.bound = root267Candidates5 := by
  unfold candidates
  have row : edge 3593 = (SortedNatTree.contains FastRows.row3593) := by
    change Data.row3593.contains = (SortedNatTree.contains FastRows.row3593)
    exact FastRows.row3593_checked.symm
  rw [row]
  decide
theorem root267Step5 : scan edge (search edge 3) 4 [1733] 1733 root267Tail5 winner =
    scan edge (search edge 3) 4 [1733] 1733 root267Tail6 winner := by
  rw [root267Tail5, scan, if_neg (by decide), root267Candidates5_checked]
  change scan edge (search edge 3) 4 [1733] 1733 root267Tail6
    (if 3 ≤ root267Candidates5.length then search edge 3 [1733, 3593] root267Candidates5 5326 winner else winner) = _
  rw [if_neg (by decide)]
def root267Candidates6 : List Nat := [6323, 13457, 20183]
set_option maxRecDepth 8192 in
theorem root267Candidates6_checked : candidates edge 4 1733 3917 root267Tail7 winner.bound = root267Candidates6 := by
  unfold candidates
  have row : edge 3917 = (SortedNatTree.contains FastRows.row3917) := by
    change Data.row3917.contains = (SortedNatTree.contains FastRows.row3917)
    exact FastRows.row3917_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root267Child6 : search edge 3 [1733, 3917] root267Candidates6 5650 winner = winner := by decide
theorem root267Step6 : scan edge (search edge 3) 4 [1733] 1733 root267Tail6 winner =
    scan edge (search edge 3) 4 [1733] 1733 root267Tail7 winner := by
  rw [root267Tail6, scan, if_neg (by decide), root267Candidates6_checked]
  change scan edge (search edge 3) 4 [1733] 1733 root267Tail7
    (if 3 ≤ root267Candidates6.length then search edge 3 [1733, 3917] root267Candidates6 5650 winner else winner) = _
  rw [if_pos (by decide), root267Child6]
def root267Candidates7 : List Nat := [6323]
set_option maxRecDepth 8192 in
theorem root267Candidates7_checked : candidates edge 4 1733 4001 root267Tail8 winner.bound = root267Candidates7 := by
  unfold candidates
  have row : edge 4001 = (SortedNatTree.contains FastRows.row4001) := by
    change Data.row4001.contains = (SortedNatTree.contains FastRows.row4001)
    exact FastRows.row4001_checked.symm
  rw [row]
  decide
theorem root267Step7 : scan edge (search edge 3) 4 [1733] 1733 root267Tail7 winner =
    scan edge (search edge 3) 4 [1733] 1733 root267Tail8 winner := by
  rw [root267Tail7, scan, if_neg (by decide), root267Candidates7_checked]
  change scan edge (search edge 3) 4 [1733] 1733 root267Tail8
    (if 3 ≤ root267Candidates7.length then search edge 3 [1733, 4001] root267Candidates7 5734 winner else winner) = _
  rw [if_neg (by decide)]
def root267Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root267Candidates8_checked : candidates edge 4 1733 4133 root267Tail9 winner.bound = root267Candidates8 := by
  unfold candidates
  have row : edge 4133 = (SortedNatTree.contains FastRows.row4133) := by
    change Data.row4133.contains = (SortedNatTree.contains FastRows.row4133)
    exact FastRows.row4133_checked.symm
  rw [row]
  decide
theorem root267Step8 : scan edge (search edge 3) 4 [1733] 1733 root267Tail8 winner =
    scan edge (search edge 3) 4 [1733] 1733 root267Tail9 winner := by
  rw [root267Tail8, scan, if_neg (by decide), root267Candidates8_checked]
  change scan edge (search edge 3) 4 [1733] 1733 root267Tail9
    (if 3 ≤ root267Candidates8.length then search edge 3 [1733, 4133] root267Candidates8 5866 winner else winner) = _
  rw [if_neg (by decide)]
def root267Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root267Candidates9_checked : candidates edge 4 1733 4817 root267Tail10 winner.bound = root267Candidates9 := by
  unfold candidates
  have row : edge 4817 = (SortedNatTree.contains FastRows.row4817) := by
    change Data.row4817.contains = (SortedNatTree.contains FastRows.row4817)
    exact FastRows.row4817_checked.symm
  rw [row]
  decide
theorem root267Step9 : scan edge (search edge 3) 4 [1733] 1733 root267Tail9 winner =
    scan edge (search edge 3) 4 [1733] 1733 root267Tail10 winner := by
  rw [root267Tail9, scan, if_neg (by decide), root267Candidates9_checked]
  change scan edge (search edge 3) 4 [1733] 1733 root267Tail10
    (if 3 ≤ root267Candidates9.length then search edge 3 [1733, 4817] root267Candidates9 6550 winner else winner) = _
  rw [if_neg (by decide)]
def root267Candidates10 : List Nat := [5867, 6101, 7949, 8969, 9929]
set_option maxRecDepth 8192 in
theorem root267Candidates10_checked : candidates edge 4 1733 5051 root267Tail11 winner.bound = root267Candidates10 := by
  unfold candidates
  have row : edge 5051 = (SortedNatTree.contains FastRows.row5051) := by
    change Data.row5051.contains = (SortedNatTree.contains FastRows.row5051)
    exact FastRows.row5051_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root267Child10 : search edge 3 [1733, 5051] root267Candidates10 6784 winner = winner := by decide
theorem root267Step10 : scan edge (search edge 3) 4 [1733] 1733 root267Tail10 winner =
    scan edge (search edge 3) 4 [1733] 1733 root267Tail11 winner := by
  rw [root267Tail10, scan, if_neg (by decide), root267Candidates10_checked]
  change scan edge (search edge 3) 4 [1733] 1733 root267Tail11
    (if 3 ≤ root267Candidates10.length then search edge 3 [1733, 5051] root267Candidates10 6784 winner else winner) = _
  rw [if_pos (by decide), root267Child10]
def root267Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root267Candidates11_checked : candidates edge 4 1733 5273 root267Tail12 winner.bound = root267Candidates11 := by
  unfold candidates
  have row : edge 5273 = (SortedNatTree.contains FastRows.row5273) := by
    change Data.row5273.contains = (SortedNatTree.contains FastRows.row5273)
    exact FastRows.row5273_checked.symm
  rw [row]
  decide
theorem root267Step11 : scan edge (search edge 3) 4 [1733] 1733 root267Tail11 winner =
    scan edge (search edge 3) 4 [1733] 1733 root267Tail12 winner := by
  rw [root267Tail11, scan, if_neg (by decide), root267Candidates11_checked]
  change scan edge (search edge 3) 4 [1733] 1733 root267Tail12
    (if 3 ≤ root267Candidates11.length then search edge 3 [1733, 5273] root267Candidates11 7006 winner else winner) = _
  rw [if_neg (by decide)]
def root267Candidates12 : List Nat := [18329]
set_option maxRecDepth 8192 in
theorem root267Candidates12_checked : candidates edge 4 1733 5867 root267Tail13 winner.bound = root267Candidates12 := by
  unfold candidates
  have row : edge 5867 = (SortedNatTree.contains FastRows.row5867) := by
    change Data.row5867.contains = (SortedNatTree.contains FastRows.row5867)
    exact FastRows.row5867_checked.symm
  rw [row]
  decide
theorem root267Step12 : scan edge (search edge 3) 4 [1733] 1733 root267Tail12 winner =
    scan edge (search edge 3) 4 [1733] 1733 root267Tail13 winner := by
  rw [root267Tail12, scan, if_neg (by decide), root267Candidates12_checked]
  change scan edge (search edge 3) 4 [1733] 1733 root267Tail13
    (if 3 ≤ root267Candidates12.length then search edge 3 [1733, 5867] root267Candidates12 7600 winner else winner) = _
  rw [if_neg (by decide)]
def root267Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root267Candidates13_checked : candidates edge 4 1733 5897 root267Tail14 winner.bound = root267Candidates13 := by
  unfold candidates
  have row : edge 5897 = (SortedNatTree.contains FastRows.row5897) := by
    change Data.row5897.contains = (SortedNatTree.contains FastRows.row5897)
    exact FastRows.row5897_checked.symm
  rw [row]
  decide
theorem root267Step13 : scan edge (search edge 3) 4 [1733] 1733 root267Tail13 winner =
    scan edge (search edge 3) 4 [1733] 1733 root267Tail14 winner := by
  rw [root267Tail13, scan, if_neg (by decide), root267Candidates13_checked]
  change scan edge (search edge 3) 4 [1733] 1733 root267Tail14
    (if 3 ≤ root267Candidates13.length then search edge 3 [1733, 5897] root267Candidates13 7630 winner else winner) = _
  rw [if_neg (by decide)]
theorem root267Scan14 : scan edge (search edge 3) 4 [1733] 1733 root267Tail14 winner = winner := by rw [root267Tail14, scan, if_pos (by decide)]
theorem root267Scan13 : scan edge (search edge 3) 4 [1733] 1733 root267Tail13 winner = winner :=
  (root267Step13).trans root267Scan14
theorem root267Scan12 : scan edge (search edge 3) 4 [1733] 1733 root267Tail12 winner = winner :=
  (root267Step12).trans root267Scan13
theorem root267Scan11 : scan edge (search edge 3) 4 [1733] 1733 root267Tail11 winner = winner :=
  (root267Step11).trans root267Scan12
theorem root267Scan10 : scan edge (search edge 3) 4 [1733] 1733 root267Tail10 winner = winner :=
  (root267Step10).trans root267Scan11
theorem root267Scan9 : scan edge (search edge 3) 4 [1733] 1733 root267Tail9 winner = winner :=
  (root267Step9).trans root267Scan10
theorem root267Scan8 : scan edge (search edge 3) 4 [1733] 1733 root267Tail8 winner = winner :=
  (root267Step8).trans root267Scan9
theorem root267Scan7 : scan edge (search edge 3) 4 [1733] 1733 root267Tail7 winner = winner :=
  (root267Step7).trans root267Scan8
theorem root267Scan6 : scan edge (search edge 3) 4 [1733] 1733 root267Tail6 winner = winner :=
  (root267Step6).trans root267Scan7
theorem root267Scan5 : scan edge (search edge 3) 4 [1733] 1733 root267Tail5 winner = winner :=
  (root267Step5).trans root267Scan6
theorem root267Scan4 : scan edge (search edge 3) 4 [1733] 1733 root267Tail4 winner = winner :=
  (root267Step4).trans root267Scan5
theorem root267Scan3 : scan edge (search edge 3) 4 [1733] 1733 root267Tail3 winner = winner :=
  (root267Step3).trans root267Scan4
theorem root267Scan2 : scan edge (search edge 3) 4 [1733] 1733 root267Tail2 winner = winner :=
  (root267Step2).trans root267Scan3
theorem root267Scan1 : scan edge (search edge 3) 4 [1733] 1733 root267Tail1 winner = winner :=
  (root267Step1).trans root267Scan2
theorem root267Scan0 : scan edge (search edge 3) 4 [1733] 1733 root267Tail0 winner = winner :=
  (root267Step0).trans root267Scan1
set_option maxRecDepth 4096 in
theorem root267Branch : search edge 4 [1733] root267Candidates 1733 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1733] 1733 root267Tail0 winner = winner
  exact root267Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
