import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root207Candidates : List Nat := Data.row1291
set_option maxRecDepth 8192 in
theorem root207Candidates_checked : candidates edge 5 0 1291 tail208 winner.bound = root207Candidates := by
  apply candidates_eq_row edge 5 0 1291 tail208 root207Candidates winner.bound tail208_ordered
  · decide
  · decide
  · funext q
    rfl
def root207Tail45 : List Nat := []
def root207Tail44 : List Nat := 24337 :: root207Tail45
def root207Tail43 : List Nat := 24091 :: root207Tail44
def root207Tail42 : List Nat := 22717 :: root207Tail43
def root207Tail41 : List Nat := 22573 :: root207Tail42
def root207Tail40 : List Nat := 22549 :: root207Tail41
def root207Tail39 : List Nat := 22369 :: root207Tail40
def root207Tail38 : List Nat := 21859 :: root207Tail39
def root207Tail37 : List Nat := 20947 :: root207Tail38
def root207Tail36 : List Nat := 20809 :: root207Tail37
def root207Tail35 : List Nat := 19843 :: root207Tail36
def root207Tail34 : List Nat := 19759 :: root207Tail35
def root207Tail33 : List Nat := 19273 :: root207Tail34
def root207Tail32 : List Nat := 16987 :: root207Tail33
def root207Tail31 : List Nat := 16363 :: root207Tail32
def root207Tail30 : List Nat := 16063 :: root207Tail31
def root207Tail29 : List Nat := 15817 :: root207Tail30
def root207Tail28 : List Nat := 15661 :: root207Tail29
def root207Tail27 : List Nat := 14389 :: root207Tail28
def root207Tail26 : List Nat := 13711 :: root207Tail27
def root207Tail25 : List Nat := 12829 :: root207Tail26
def root207Tail24 : List Nat := 12343 :: root207Tail25
def root207Tail23 : List Nat := 11317 :: root207Tail24
def root207Tail22 : List Nat := 11113 :: root207Tail23
def root207Tail21 : List Nat := 11083 :: root207Tail22
def root207Tail20 : List Nat := 10771 :: root207Tail21
def root207Tail19 : List Nat := 10243 :: root207Tail20
def root207Tail18 : List Nat := 10141 :: root207Tail19
def root207Tail17 : List Nat := 9463 :: root207Tail18
def root207Tail16 : List Nat := 9043 :: root207Tail17
def root207Tail15 : List Nat := 8923 :: root207Tail16
def root207Tail14 : List Nat := 8293 :: root207Tail15
def root207Tail13 : List Nat := 8209 :: root207Tail14
def root207Tail12 : List Nat := 7741 :: root207Tail13
def root207Tail11 : List Nat := 6907 :: root207Tail12
def root207Tail10 : List Nat := 6733 :: root207Tail11
def root207Tail9 : List Nat := 5557 :: root207Tail10
def root207Tail8 : List Nat := 5059 :: root207Tail9
def root207Tail7 : List Nat := 4903 :: root207Tail8
def root207Tail6 : List Nat := 4513 :: root207Tail7
def root207Tail5 : List Nat := 4273 :: root207Tail6
def root207Tail4 : List Nat := 3793 :: root207Tail5
def root207Tail3 : List Nat := 3457 :: root207Tail4
def root207Tail2 : List Nat := 2293 :: root207Tail3
def root207Tail1 : List Nat := 2281 :: root207Tail2
def root207Tail0 : List Nat := 1777 :: root207Tail1
def root207Candidates0 : List Nat := [7741]
set_option maxRecDepth 8192 in
theorem root207Candidates0_checked : candidates edge 4 1291 1777 root207Tail1 winner.bound = root207Candidates0 := by
  unfold candidates
  have row : edge 1777 = (SortedNatTree.contains FastRows.row1777) := by
    change Data.row1777.contains = (SortedNatTree.contains FastRows.row1777)
    exact FastRows.row1777_checked.symm
  rw [row]
  decide
theorem root207Step0 : scan edge (search edge 3) 4 [1291] 1291 root207Tail0 winner =
    scan edge (search edge 3) 4 [1291] 1291 root207Tail1 winner := by
  rw [root207Tail0, scan, if_neg (by decide), root207Candidates0_checked]
  change scan edge (search edge 3) 4 [1291] 1291 root207Tail1
    (if 3 ≤ root207Candidates0.length then search edge 3 [1291, 1777] root207Candidates0 3068 winner else winner) = _
  rw [if_neg (by decide)]
def root207Candidates1 : List Nat := [4273]
set_option maxRecDepth 8192 in
theorem root207Candidates1_checked : candidates edge 4 1291 2281 root207Tail2 winner.bound = root207Candidates1 := by
  unfold candidates
  have row : edge 2281 = (SortedNatTree.contains FastRows.row2281) := by
    change Data.row2281.contains = (SortedNatTree.contains FastRows.row2281)
    exact FastRows.row2281_checked.symm
  rw [row]
  decide
theorem root207Step1 : scan edge (search edge 3) 4 [1291] 1291 root207Tail1 winner =
    scan edge (search edge 3) 4 [1291] 1291 root207Tail2 winner := by
  rw [root207Tail1, scan, if_neg (by decide), root207Candidates1_checked]
  change scan edge (search edge 3) 4 [1291] 1291 root207Tail2
    (if 3 ≤ root207Candidates1.length then search edge 3 [1291, 2281] root207Candidates1 3572 winner else winner) = _
  rw [if_neg (by decide)]
def root207Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root207Candidates2_checked : candidates edge 4 1291 2293 root207Tail3 winner.bound = root207Candidates2 := by
  unfold candidates
  have row : edge 2293 = (SortedNatTree.contains FastRows.row2293) := by
    change Data.row2293.contains = (SortedNatTree.contains FastRows.row2293)
    exact FastRows.row2293_checked.symm
  rw [row]
  decide
theorem root207Step2 : scan edge (search edge 3) 4 [1291] 1291 root207Tail2 winner =
    scan edge (search edge 3) 4 [1291] 1291 root207Tail3 winner := by
  rw [root207Tail2, scan, if_neg (by decide), root207Candidates2_checked]
  change scan edge (search edge 3) 4 [1291] 1291 root207Tail3
    (if 3 ≤ root207Candidates2.length then search edge 3 [1291, 2293] root207Candidates2 3584 winner else winner) = _
  rw [if_neg (by decide)]
def root207Candidates3 : List Nat := [8209]
set_option maxRecDepth 8192 in
theorem root207Candidates3_checked : candidates edge 4 1291 3457 root207Tail4 winner.bound = root207Candidates3 := by
  unfold candidates
  have row : edge 3457 = (SortedNatTree.contains FastRows.row3457) := by
    change Data.row3457.contains = (SortedNatTree.contains FastRows.row3457)
    exact FastRows.row3457_checked.symm
  rw [row]
  decide
theorem root207Step3 : scan edge (search edge 3) 4 [1291] 1291 root207Tail3 winner =
    scan edge (search edge 3) 4 [1291] 1291 root207Tail4 winner := by
  rw [root207Tail3, scan, if_neg (by decide), root207Candidates3_checked]
  change scan edge (search edge 3) 4 [1291] 1291 root207Tail4
    (if 3 ≤ root207Candidates3.length then search edge 3 [1291, 3457] root207Candidates3 4748 winner else winner) = _
  rw [if_neg (by decide)]
def root207Candidates4 : List Nat := [6733, 6907, 16363]
set_option maxRecDepth 8192 in
theorem root207Candidates4_checked : candidates edge 4 1291 3793 root207Tail5 winner.bound = root207Candidates4 := by
  unfold candidates
  have row : edge 3793 = (SortedNatTree.contains FastRows.row3793) := by
    change Data.row3793.contains = (SortedNatTree.contains FastRows.row3793)
    exact FastRows.row3793_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root207Child4 : search edge 3 [1291, 3793] root207Candidates4 5084 winner = winner := by decide
theorem root207Step4 : scan edge (search edge 3) 4 [1291] 1291 root207Tail4 winner =
    scan edge (search edge 3) 4 [1291] 1291 root207Tail5 winner := by
  rw [root207Tail4, scan, if_neg (by decide), root207Candidates4_checked]
  change scan edge (search edge 3) 4 [1291] 1291 root207Tail5
    (if 3 ≤ root207Candidates4.length then search edge 3 [1291, 3793] root207Candidates4 5084 winner else winner) = _
  rw [if_pos (by decide), root207Child4]
def root207Candidates5 : List Nat := [10243]
set_option maxRecDepth 8192 in
theorem root207Candidates5_checked : candidates edge 4 1291 4273 root207Tail6 winner.bound = root207Candidates5 := by
  unfold candidates
  have row : edge 4273 = (SortedNatTree.contains FastRows.row4273) := by
    change Data.row4273.contains = (SortedNatTree.contains FastRows.row4273)
    exact FastRows.row4273_checked.symm
  rw [row]
  decide
theorem root207Step5 : scan edge (search edge 3) 4 [1291] 1291 root207Tail5 winner =
    scan edge (search edge 3) 4 [1291] 1291 root207Tail6 winner := by
  rw [root207Tail5, scan, if_neg (by decide), root207Candidates5_checked]
  change scan edge (search edge 3) 4 [1291] 1291 root207Tail6
    (if 3 ≤ root207Candidates5.length then search edge 3 [1291, 4273] root207Candidates5 5564 winner else winner) = _
  rw [if_neg (by decide)]
def root207Candidates6 : List Nat := [10771]
set_option maxRecDepth 8192 in
theorem root207Candidates6_checked : candidates edge 4 1291 4513 root207Tail7 winner.bound = root207Candidates6 := by
  unfold candidates
  have row : edge 4513 = (SortedNatTree.contains FastRows.row4513) := by
    change Data.row4513.contains = (SortedNatTree.contains FastRows.row4513)
    exact FastRows.row4513_checked.symm
  rw [row]
  decide
theorem root207Step6 : scan edge (search edge 3) 4 [1291] 1291 root207Tail6 winner =
    scan edge (search edge 3) 4 [1291] 1291 root207Tail7 winner := by
  rw [root207Tail6, scan, if_neg (by decide), root207Candidates6_checked]
  change scan edge (search edge 3) 4 [1291] 1291 root207Tail7
    (if 3 ≤ root207Candidates6.length then search edge 3 [1291, 4513] root207Candidates6 5804 winner else winner) = _
  rw [if_neg (by decide)]
def root207Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root207Candidates7_checked : candidates edge 4 1291 4903 root207Tail8 winner.bound = root207Candidates7 := by
  unfold candidates
  have row : edge 4903 = (SortedNatTree.contains FastRows.row4903) := by
    change Data.row4903.contains = (SortedNatTree.contains FastRows.row4903)
    exact FastRows.row4903_checked.symm
  rw [row]
  decide
theorem root207Step7 : scan edge (search edge 3) 4 [1291] 1291 root207Tail7 winner =
    scan edge (search edge 3) 4 [1291] 1291 root207Tail8 winner := by
  rw [root207Tail7, scan, if_neg (by decide), root207Candidates7_checked]
  change scan edge (search edge 3) 4 [1291] 1291 root207Tail8
    (if 3 ≤ root207Candidates7.length then search edge 3 [1291, 4903] root207Candidates7 6194 winner else winner) = _
  rw [if_neg (by decide)]
def root207Candidates8 : List Nat := [13711]
set_option maxRecDepth 8192 in
theorem root207Candidates8_checked : candidates edge 4 1291 5059 root207Tail9 winner.bound = root207Candidates8 := by
  unfold candidates
  have row : edge 5059 = (SortedNatTree.contains FastRows.row5059) := by
    change Data.row5059.contains = (SortedNatTree.contains FastRows.row5059)
    exact FastRows.row5059_checked.symm
  rw [row]
  decide
theorem root207Step8 : scan edge (search edge 3) 4 [1291] 1291 root207Tail8 winner =
    scan edge (search edge 3) 4 [1291] 1291 root207Tail9 winner := by
  rw [root207Tail8, scan, if_neg (by decide), root207Candidates8_checked]
  change scan edge (search edge 3) 4 [1291] 1291 root207Tail9
    (if 3 ≤ root207Candidates8.length then search edge 3 [1291, 5059] root207Candidates8 6350 winner else winner) = _
  rw [if_neg (by decide)]
def root207Candidates9 : List Nat := [8293]
set_option maxRecDepth 8192 in
theorem root207Candidates9_checked : candidates edge 4 1291 5557 root207Tail10 winner.bound = root207Candidates9 := by
  unfold candidates
  have row : edge 5557 = (SortedNatTree.contains FastRows.row5557) := by
    change Data.row5557.contains = (SortedNatTree.contains FastRows.row5557)
    exact FastRows.row5557_checked.symm
  rw [row]
  decide
theorem root207Step9 : scan edge (search edge 3) 4 [1291] 1291 root207Tail9 winner =
    scan edge (search edge 3) 4 [1291] 1291 root207Tail10 winner := by
  rw [root207Tail9, scan, if_neg (by decide), root207Candidates9_checked]
  change scan edge (search edge 3) 4 [1291] 1291 root207Tail10
    (if 3 ≤ root207Candidates9.length then search edge 3 [1291, 5557] root207Candidates9 6848 winner else winner) = _
  rw [if_neg (by decide)]
theorem root207Scan10 : scan edge (search edge 3) 4 [1291] 1291 root207Tail10 winner = winner := by rw [root207Tail10, scan, if_pos (by decide)]
theorem root207Scan9 : scan edge (search edge 3) 4 [1291] 1291 root207Tail9 winner = winner :=
  (root207Step9).trans root207Scan10
theorem root207Scan8 : scan edge (search edge 3) 4 [1291] 1291 root207Tail8 winner = winner :=
  (root207Step8).trans root207Scan9
theorem root207Scan7 : scan edge (search edge 3) 4 [1291] 1291 root207Tail7 winner = winner :=
  (root207Step7).trans root207Scan8
theorem root207Scan6 : scan edge (search edge 3) 4 [1291] 1291 root207Tail6 winner = winner :=
  (root207Step6).trans root207Scan7
theorem root207Scan5 : scan edge (search edge 3) 4 [1291] 1291 root207Tail5 winner = winner :=
  (root207Step5).trans root207Scan6
theorem root207Scan4 : scan edge (search edge 3) 4 [1291] 1291 root207Tail4 winner = winner :=
  (root207Step4).trans root207Scan5
theorem root207Scan3 : scan edge (search edge 3) 4 [1291] 1291 root207Tail3 winner = winner :=
  (root207Step3).trans root207Scan4
theorem root207Scan2 : scan edge (search edge 3) 4 [1291] 1291 root207Tail2 winner = winner :=
  (root207Step2).trans root207Scan3
theorem root207Scan1 : scan edge (search edge 3) 4 [1291] 1291 root207Tail1 winner = winner :=
  (root207Step1).trans root207Scan2
theorem root207Scan0 : scan edge (search edge 3) 4 [1291] 1291 root207Tail0 winner = winner :=
  (root207Step0).trans root207Scan1
set_option maxRecDepth 4096 in
theorem root207Branch : search edge 4 [1291] root207Candidates 1291 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1291] 1291 root207Tail0 winner = winner
  exact root207Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
