import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root342Candidates : List Nat := Data.row2333
set_option maxRecDepth 8192 in
theorem root342Candidates_checked : candidates edge 5 0 2333 tail343 winner.bound = root342Candidates := by
  apply candidates_eq_row edge 5 0 2333 tail343 root342Candidates winner.bound tail343_ordered
  · decide
  · decide
  · funext q
    rfl
def root342Tail55 : List Nat := []
def root342Tail54 : List Nat := 23609 :: root342Tail55
def root342Tail53 : List Nat := 23399 :: root342Tail54
def root342Tail52 : List Nat := 22751 :: root342Tail53
def root342Tail51 : List Nat := 21701 :: root342Tail52
def root342Tail50 : List Nat := 20963 :: root342Tail51
def root342Tail49 : List Nat := 20753 :: root342Tail50
def root342Tail48 : List Nat := 20231 :: root342Tail49
def root342Tail47 : List Nat := 19949 :: root342Tail48
def root342Tail46 : List Nat := 19913 :: root342Tail47
def root342Tail45 : List Nat := 19301 :: root342Tail46
def root342Tail44 : List Nat := 19157 :: root342Tail45
def root342Tail43 : List Nat := 18143 :: root342Tail44
def root342Tail42 : List Nat := 18131 :: root342Tail43
def root342Tail41 : List Nat := 17837 :: root342Tail42
def root342Tail40 : List Nat := 17627 :: root342Tail41
def root342Tail39 : List Nat := 16553 :: root342Tail40
def root342Tail38 : List Nat := 16451 :: root342Tail39
def root342Tail37 : List Nat := 16361 :: root342Tail38
def root342Tail36 : List Nat := 16253 :: root342Tail37
def root342Tail35 : List Nat := 15881 :: root342Tail36
def root342Tail34 : List Nat := 15737 :: root342Tail35
def root342Tail33 : List Nat := 15569 :: root342Tail34
def root342Tail32 : List Nat := 15413 :: root342Tail33
def root342Tail31 : List Nat := 15101 :: root342Tail32
def root342Tail30 : List Nat := 14867 :: root342Tail31
def root342Tail29 : List Nat := 14327 :: root342Tail30
def root342Tail28 : List Nat := 13109 :: root342Tail29
def root342Tail27 : List Nat := 12503 :: root342Tail28
def root342Tail26 : List Nat := 12203 :: root342Tail27
def root342Tail25 : List Nat := 12119 :: root342Tail26
def root342Tail24 : List Nat := 11489 :: root342Tail25
def root342Tail23 : List Nat := 9479 :: root342Tail24
def root342Tail22 : List Nat := 9473 :: root342Tail23
def root342Tail21 : List Nat := 8867 :: root342Tail22
def root342Tail20 : List Nat := 8831 :: root342Tail21
def root342Tail19 : List Nat := 8429 :: root342Tail20
def root342Tail18 : List Nat := 8387 :: root342Tail19
def root342Tail17 : List Nat := 7907 :: root342Tail18
def root342Tail16 : List Nat := 7253 :: root342Tail17
def root342Tail15 : List Nat := 7229 :: root342Tail16
def root342Tail14 : List Nat := 6701 :: root342Tail15
def root342Tail13 : List Nat := 6311 :: root342Tail14
def root342Tail12 : List Nat := 5987 :: root342Tail13
def root342Tail11 : List Nat := 5927 :: root342Tail12
def root342Tail10 : List Nat := 5297 :: root342Tail11
def root342Tail9 : List Nat := 5261 :: root342Tail10
def root342Tail8 : List Nat := 5189 :: root342Tail9
def root342Tail7 : List Nat := 3701 :: root342Tail8
def root342Tail6 : List Nat := 3533 :: root342Tail7
def root342Tail5 : List Nat := 3221 :: root342Tail6
def root342Tail4 : List Nat := 3137 :: root342Tail5
def root342Tail3 : List Nat := 2963 :: root342Tail4
def root342Tail2 : List Nat := 2843 :: root342Tail3
def root342Tail1 : List Nat := 2423 :: root342Tail2
def root342Tail0 : List Nat := 2339 :: root342Tail1
def root342Candidates0 : List Nat := [16253, 16451]
set_option maxRecDepth 8192 in
theorem root342Candidates0_checked : candidates edge 4 2333 2339 root342Tail1 winner.bound = root342Candidates0 := by
  unfold candidates
  have row : edge 2339 = (SortedNatTree.contains FastRows.row2339) := by
    change Data.row2339.contains = (SortedNatTree.contains FastRows.row2339)
    exact FastRows.row2339_checked.symm
  rw [row]
  decide
theorem root342Step0 : scan edge (search edge 3) 4 [2333] 2333 root342Tail0 winner =
    scan edge (search edge 3) 4 [2333] 2333 root342Tail1 winner := by
  rw [root342Tail0, scan, if_neg (by decide), root342Candidates0_checked]
  change scan edge (search edge 3) 4 [2333] 2333 root342Tail1
    (if 3 ≤ root342Candidates0.length then search edge 3 [2333, 2339] root342Candidates0 4672 winner else winner) = _
  rw [if_neg (by decide)]
def root342Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root342Candidates1_checked : candidates edge 4 2333 2423 root342Tail2 winner.bound = root342Candidates1 := by
  unfold candidates
  have row : edge 2423 = (SortedNatTree.contains FastRows.row2423) := by
    change Data.row2423.contains = (SortedNatTree.contains FastRows.row2423)
    exact FastRows.row2423_checked.symm
  rw [row]
  decide
theorem root342Step1 : scan edge (search edge 3) 4 [2333] 2333 root342Tail1 winner =
    scan edge (search edge 3) 4 [2333] 2333 root342Tail2 winner := by
  rw [root342Tail1, scan, if_neg (by decide), root342Candidates1_checked]
  change scan edge (search edge 3) 4 [2333] 2333 root342Tail2
    (if 3 ≤ root342Candidates1.length then search edge 3 [2333, 2423] root342Candidates1 4756 winner else winner) = _
  rw [if_neg (by decide)]
def root342Candidates2 : List Nat := [8831, 9473]
set_option maxRecDepth 8192 in
theorem root342Candidates2_checked : candidates edge 4 2333 2843 root342Tail3 winner.bound = root342Candidates2 := by
  unfold candidates
  have row : edge 2843 = (SortedNatTree.contains FastRows.row2843) := by
    change Data.row2843.contains = (SortedNatTree.contains FastRows.row2843)
    exact FastRows.row2843_checked.symm
  rw [row]
  decide
theorem root342Step2 : scan edge (search edge 3) 4 [2333] 2333 root342Tail2 winner =
    scan edge (search edge 3) 4 [2333] 2333 root342Tail3 winner := by
  rw [root342Tail2, scan, if_neg (by decide), root342Candidates2_checked]
  change scan edge (search edge 3) 4 [2333] 2333 root342Tail3
    (if 3 ≤ root342Candidates2.length then search edge 3 [2333, 2843] root342Candidates2 5176 winner else winner) = _
  rw [if_neg (by decide)]
def root342Candidates3 : List Nat := [17627]
set_option maxRecDepth 8192 in
theorem root342Candidates3_checked : candidates edge 4 2333 2963 root342Tail4 winner.bound = root342Candidates3 := by
  unfold candidates
  have row : edge 2963 = (SortedNatTree.contains FastRows.row2963) := by
    change Data.row2963.contains = (SortedNatTree.contains FastRows.row2963)
    exact FastRows.row2963_checked.symm
  rw [row]
  decide
theorem root342Step3 : scan edge (search edge 3) 4 [2333] 2333 root342Tail3 winner =
    scan edge (search edge 3) 4 [2333] 2333 root342Tail4 winner := by
  rw [root342Tail3, scan, if_neg (by decide), root342Candidates3_checked]
  change scan edge (search edge 3) 4 [2333] 2333 root342Tail4
    (if 3 ≤ root342Candidates3.length then search edge 3 [2333, 2963] root342Candidates3 5296 winner else winner) = _
  rw [if_neg (by decide)]
def root342Candidates4 : List Nat := [5189, 6311, 8387, 14867]
set_option maxRecDepth 8192 in
theorem root342Candidates4_checked : candidates edge 4 2333 3137 root342Tail5 winner.bound = root342Candidates4 := by
  unfold candidates
  have row : edge 3137 = (SortedNatTree.contains FastRows.row3137) := by
    change Data.row3137.contains = (SortedNatTree.contains FastRows.row3137)
    exact FastRows.row3137_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root342Child4 : search edge 3 [2333, 3137] root342Candidates4 5470 winner = winner := by decide
theorem root342Step4 : scan edge (search edge 3) 4 [2333] 2333 root342Tail4 winner =
    scan edge (search edge 3) 4 [2333] 2333 root342Tail5 winner := by
  rw [root342Tail4, scan, if_neg (by decide), root342Candidates4_checked]
  change scan edge (search edge 3) 4 [2333] 2333 root342Tail5
    (if 3 ≤ root342Candidates4.length then search edge 3 [2333, 3137] root342Candidates4 5470 winner else winner) = _
  rw [if_pos (by decide), root342Child4]
def root342Candidates5 : List Nat := [6311, 8429, 18143]
set_option maxRecDepth 8192 in
theorem root342Candidates5_checked : candidates edge 4 2333 3221 root342Tail6 winner.bound = root342Candidates5 := by
  unfold candidates
  have row : edge 3221 = (SortedNatTree.contains FastRows.row3221) := by
    change Data.row3221.contains = (SortedNatTree.contains FastRows.row3221)
    exact FastRows.row3221_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root342Child5 : search edge 3 [2333, 3221] root342Candidates5 5554 winner = winner := by decide
theorem root342Step5 : scan edge (search edge 3) 4 [2333] 2333 root342Tail5 winner =
    scan edge (search edge 3) 4 [2333] 2333 root342Tail6 winner := by
  rw [root342Tail5, scan, if_neg (by decide), root342Candidates5_checked]
  change scan edge (search edge 3) 4 [2333] 2333 root342Tail6
    (if 3 ≤ root342Candidates5.length then search edge 3 [2333, 3221] root342Candidates5 5554 winner else winner) = _
  rw [if_pos (by decide), root342Child5]
def root342Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root342Candidates6_checked : candidates edge 4 2333 3533 root342Tail7 winner.bound = root342Candidates6 := by
  unfold candidates
  have row : edge 3533 = (SortedNatTree.contains FastRows.row3533) := by
    change Data.row3533.contains = (SortedNatTree.contains FastRows.row3533)
    exact FastRows.row3533_checked.symm
  rw [row]
  decide
theorem root342Step6 : scan edge (search edge 3) 4 [2333] 2333 root342Tail6 winner =
    scan edge (search edge 3) 4 [2333] 2333 root342Tail7 winner := by
  rw [root342Tail6, scan, if_neg (by decide), root342Candidates6_checked]
  change scan edge (search edge 3) 4 [2333] 2333 root342Tail7
    (if 3 ≤ root342Candidates6.length then search edge 3 [2333, 3533] root342Candidates6 5866 winner else winner) = _
  rw [if_neg (by decide)]
def root342Candidates7 : List Nat := [9473]
set_option maxRecDepth 8192 in
theorem root342Candidates7_checked : candidates edge 4 2333 3701 root342Tail8 winner.bound = root342Candidates7 := by
  unfold candidates
  have row : edge 3701 = (SortedNatTree.contains FastRows.row3701) := by
    change Data.row3701.contains = (SortedNatTree.contains FastRows.row3701)
    exact FastRows.row3701_checked.symm
  rw [row]
  decide
theorem root342Step7 : scan edge (search edge 3) 4 [2333] 2333 root342Tail7 winner =
    scan edge (search edge 3) 4 [2333] 2333 root342Tail8 winner := by
  rw [root342Tail7, scan, if_neg (by decide), root342Candidates7_checked]
  change scan edge (search edge 3) 4 [2333] 2333 root342Tail8
    (if 3 ≤ root342Candidates7.length then search edge 3 [2333, 3701] root342Candidates7 6034 winner else winner) = _
  rw [if_neg (by decide)]
def root342Candidates8 : List Nat := [16553]
set_option maxRecDepth 8192 in
theorem root342Candidates8_checked : candidates edge 4 2333 5189 root342Tail9 winner.bound = root342Candidates8 := by
  unfold candidates
  have row : edge 5189 = (SortedNatTree.contains FastRows.row5189) := by
    change Data.row5189.contains = (SortedNatTree.contains FastRows.row5189)
    exact FastRows.row5189_checked.symm
  rw [row]
  decide
theorem root342Step8 : scan edge (search edge 3) 4 [2333] 2333 root342Tail8 winner =
    scan edge (search edge 3) 4 [2333] 2333 root342Tail9 winner := by
  rw [root342Tail8, scan, if_neg (by decide), root342Candidates8_checked]
  change scan edge (search edge 3) 4 [2333] 2333 root342Tail9
    (if 3 ≤ root342Candidates8.length then search edge 3 [2333, 5189] root342Candidates8 7522 winner else winner) = _
  rw [if_neg (by decide)]
def root342Candidates9 : List Nat := [5987, 8387, 8831, 9473]
set_option maxRecDepth 8192 in
theorem root342Candidates9_checked : candidates edge 4 2333 5261 root342Tail10 winner.bound = root342Candidates9 := by
  unfold candidates
  have row : edge 5261 = (SortedNatTree.contains FastRows.row5261) := by
    change Data.row5261.contains = (SortedNatTree.contains FastRows.row5261)
    exact FastRows.row5261_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root342Child9 : search edge 3 [2333, 5261] root342Candidates9 7594 winner = winner := by decide
theorem root342Step9 : scan edge (search edge 3) 4 [2333] 2333 root342Tail9 winner =
    scan edge (search edge 3) 4 [2333] 2333 root342Tail10 winner := by
  rw [root342Tail9, scan, if_neg (by decide), root342Candidates9_checked]
  change scan edge (search edge 3) 4 [2333] 2333 root342Tail10
    (if 3 ≤ root342Candidates9.length then search edge 3 [2333, 5261] root342Candidates9 7594 winner else winner) = _
  rw [if_pos (by decide), root342Child9]
def root342Candidates10 : List Nat := [6701]
set_option maxRecDepth 8192 in
theorem root342Candidates10_checked : candidates edge 4 2333 5297 root342Tail11 winner.bound = root342Candidates10 := by
  unfold candidates
  have row : edge 5297 = (SortedNatTree.contains FastRows.row5297) := by
    change Data.row5297.contains = (SortedNatTree.contains FastRows.row5297)
    exact FastRows.row5297_checked.symm
  rw [row]
  decide
theorem root342Step10 : scan edge (search edge 3) 4 [2333] 2333 root342Tail10 winner =
    scan edge (search edge 3) 4 [2333] 2333 root342Tail11 winner := by
  rw [root342Tail10, scan, if_neg (by decide), root342Candidates10_checked]
  change scan edge (search edge 3) 4 [2333] 2333 root342Tail11
    (if 3 ≤ root342Candidates10.length then search edge 3 [2333, 5297] root342Candidates10 7630 winner else winner) = _
  rw [if_neg (by decide)]
theorem root342Scan11 : scan edge (search edge 3) 4 [2333] 2333 root342Tail11 winner = winner := by rw [root342Tail11, scan, if_pos (by decide)]
theorem root342Scan10 : scan edge (search edge 3) 4 [2333] 2333 root342Tail10 winner = winner :=
  (root342Step10).trans root342Scan11
theorem root342Scan9 : scan edge (search edge 3) 4 [2333] 2333 root342Tail9 winner = winner :=
  (root342Step9).trans root342Scan10
theorem root342Scan8 : scan edge (search edge 3) 4 [2333] 2333 root342Tail8 winner = winner :=
  (root342Step8).trans root342Scan9
theorem root342Scan7 : scan edge (search edge 3) 4 [2333] 2333 root342Tail7 winner = winner :=
  (root342Step7).trans root342Scan8
theorem root342Scan6 : scan edge (search edge 3) 4 [2333] 2333 root342Tail6 winner = winner :=
  (root342Step6).trans root342Scan7
theorem root342Scan5 : scan edge (search edge 3) 4 [2333] 2333 root342Tail5 winner = winner :=
  (root342Step5).trans root342Scan6
theorem root342Scan4 : scan edge (search edge 3) 4 [2333] 2333 root342Tail4 winner = winner :=
  (root342Step4).trans root342Scan5
theorem root342Scan3 : scan edge (search edge 3) 4 [2333] 2333 root342Tail3 winner = winner :=
  (root342Step3).trans root342Scan4
theorem root342Scan2 : scan edge (search edge 3) 4 [2333] 2333 root342Tail2 winner = winner :=
  (root342Step2).trans root342Scan3
theorem root342Scan1 : scan edge (search edge 3) 4 [2333] 2333 root342Tail1 winner = winner :=
  (root342Step1).trans root342Scan2
theorem root342Scan0 : scan edge (search edge 3) 4 [2333] 2333 root342Tail0 winner = winner :=
  (root342Step0).trans root342Scan1
set_option maxRecDepth 4096 in
theorem root342Branch : search edge 4 [2333] root342Candidates 2333 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2333] 2333 root342Tail0 winner = winner
  exact root342Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
