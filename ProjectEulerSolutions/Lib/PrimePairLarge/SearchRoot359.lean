import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root359Candidates : List Nat := Data.row2441
set_option maxRecDepth 8192 in
theorem root359Candidates_checked : candidates edge 5 0 2441 tail360 winner.bound = root359Candidates := by
  apply candidates_eq_row edge 5 0 2441 tail360 root359Candidates winner.bound tail360_ordered
  · decide
  · decide
  · funext q
    rfl
def root359Tail47 : List Nat := []
def root359Tail46 : List Nat := 23333 :: root359Tail47
def root359Tail45 : List Nat := 22709 :: root359Tail46
def root359Tail44 : List Nat := 21863 :: root359Tail45
def root359Tail43 : List Nat := 21023 :: root359Tail44
def root359Tail42 : List Nat := 20897 :: root359Tail43
def root359Tail41 : List Nat := 20477 :: root359Tail42
def root359Tail40 : List Nat := 19973 :: root359Tail41
def root359Tail39 : List Nat := 19181 :: root359Tail40
def root359Tail38 : List Nat := 17597 :: root359Tail39
def root359Tail37 : List Nat := 17387 :: root359Tail38
def root359Tail36 : List Nat := 17351 :: root359Tail37
def root359Tail35 : List Nat := 17207 :: root359Tail36
def root359Tail34 : List Nat := 15959 :: root359Tail35
def root359Tail33 : List Nat := 15671 :: root359Tail34
def root359Tail32 : List Nat := 15413 :: root359Tail33
def root359Tail31 : List Nat := 15329 :: root359Tail32
def root359Tail30 : List Nat := 15149 :: root359Tail31
def root359Tail29 : List Nat := 15083 :: root359Tail30
def root359Tail28 : List Nat := 14867 :: root359Tail29
def root359Tail27 : List Nat := 14051 :: root359Tail28
def root359Tail26 : List Nat := 11423 :: root359Tail27
def root359Tail25 : List Nat := 11321 :: root359Tail26
def root359Tail24 : List Nat := 10457 :: root359Tail25
def root359Tail23 : List Nat := 10169 :: root359Tail24
def root359Tail22 : List Nat := 9929 :: root359Tail23
def root359Tail21 : List Nat := 9341 :: root359Tail22
def root359Tail20 : List Nat := 8423 :: root359Tail21
def root359Tail19 : List Nat := 8231 :: root359Tail20
def root359Tail18 : List Nat := 7727 :: root359Tail19
def root359Tail17 : List Nat := 7691 :: root359Tail18
def root359Tail16 : List Nat := 7643 :: root359Tail17
def root359Tail15 : List Nat := 7121 :: root359Tail16
def root359Tail14 : List Nat := 6659 :: root359Tail15
def root359Tail13 : List Nat := 6113 :: root359Tail14
def root359Tail12 : List Nat := 6089 :: root359Tail13
def root359Tail11 : List Nat := 5381 :: root359Tail12
def root359Tail10 : List Nat := 5333 :: root359Tail11
def root359Tail9 : List Nat := 5297 :: root359Tail10
def root359Tail8 : List Nat := 5147 :: root359Tail9
def root359Tail7 : List Nat := 5087 :: root359Tail8
def root359Tail6 : List Nat := 4787 :: root359Tail7
def root359Tail5 : List Nat := 4349 :: root359Tail6
def root359Tail4 : List Nat := 4157 :: root359Tail5
def root359Tail3 : List Nat := 3677 :: root359Tail4
def root359Tail2 : List Nat := 3671 :: root359Tail3
def root359Tail1 : List Nat := 3209 :: root359Tail2
def root359Tail0 : List Nat := 2543 :: root359Tail1
def root359Candidates0 : List Nat := [4349, 5147, 15329, 19973]
set_option maxRecDepth 8192 in
theorem root359Candidates0_checked : candidates edge 4 2441 2543 root359Tail1 winner.bound = root359Candidates0 := by
  unfold candidates
  have row : edge 2543 = (SortedNatTree.contains FastRows.row2543) := by
    change Data.row2543.contains = (SortedNatTree.contains FastRows.row2543)
    exact FastRows.row2543_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root359Child0 : search edge 3 [2441, 2543] root359Candidates0 4984 winner = winner := by decide
theorem root359Step0 : scan edge (search edge 3) 4 [2441] 2441 root359Tail0 winner =
    scan edge (search edge 3) 4 [2441] 2441 root359Tail1 winner := by
  rw [root359Tail0, scan, if_neg (by decide), root359Candidates0_checked]
  change scan edge (search edge 3) 4 [2441] 2441 root359Tail1
    (if 3 ≤ root359Candidates0.length then search edge 3 [2441, 2543] root359Candidates0 4984 winner else winner) = _
  rw [if_pos (by decide), root359Child0]
def root359Candidates1 : List Nat := [14051]
set_option maxRecDepth 8192 in
theorem root359Candidates1_checked : candidates edge 4 2441 3209 root359Tail2 winner.bound = root359Candidates1 := by
  unfold candidates
  have row : edge 3209 = (SortedNatTree.contains FastRows.row3209) := by
    change Data.row3209.contains = (SortedNatTree.contains FastRows.row3209)
    exact FastRows.row3209_checked.symm
  rw [row]
  decide
theorem root359Step1 : scan edge (search edge 3) 4 [2441] 2441 root359Tail1 winner =
    scan edge (search edge 3) 4 [2441] 2441 root359Tail2 winner := by
  rw [root359Tail1, scan, if_neg (by decide), root359Candidates1_checked]
  change scan edge (search edge 3) 4 [2441] 2441 root359Tail2
    (if 3 ≤ root359Candidates1.length then search edge 3 [2441, 3209] root359Candidates1 5650 winner else winner) = _
  rw [if_neg (by decide)]
def root359Candidates2 : List Nat := [5381, 7691, 9929]
set_option maxRecDepth 8192 in
theorem root359Candidates2_checked : candidates edge 4 2441 3671 root359Tail3 winner.bound = root359Candidates2 := by
  unfold candidates
  have row : edge 3671 = (SortedNatTree.contains FastRows.row3671) := by
    change Data.row3671.contains = (SortedNatTree.contains FastRows.row3671)
    exact FastRows.row3671_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root359Child2 : search edge 3 [2441, 3671] root359Candidates2 6112 winner = winner := by decide
theorem root359Step2 : scan edge (search edge 3) 4 [2441] 2441 root359Tail2 winner =
    scan edge (search edge 3) 4 [2441] 2441 root359Tail3 winner := by
  rw [root359Tail2, scan, if_neg (by decide), root359Candidates2_checked]
  change scan edge (search edge 3) 4 [2441] 2441 root359Tail3
    (if 3 ≤ root359Candidates2.length then search edge 3 [2441, 3671] root359Candidates2 6112 winner else winner) = _
  rw [if_pos (by decide), root359Child2]
def root359Candidates3 : List Nat := [6659, 11321]
set_option maxRecDepth 8192 in
theorem root359Candidates3_checked : candidates edge 4 2441 3677 root359Tail4 winner.bound = root359Candidates3 := by
  unfold candidates
  have row : edge 3677 = (SortedNatTree.contains FastRows.row3677) := by
    change Data.row3677.contains = (SortedNatTree.contains FastRows.row3677)
    exact FastRows.row3677_checked.symm
  rw [row]
  decide
theorem root359Step3 : scan edge (search edge 3) 4 [2441] 2441 root359Tail3 winner =
    scan edge (search edge 3) 4 [2441] 2441 root359Tail4 winner := by
  rw [root359Tail3, scan, if_neg (by decide), root359Candidates3_checked]
  change scan edge (search edge 3) 4 [2441] 2441 root359Tail4
    (if 3 ≤ root359Candidates3.length then search edge 3 [2441, 3677] root359Candidates3 6118 winner else winner) = _
  rw [if_neg (by decide)]
def root359Candidates4 : List Nat := [5333]
set_option maxRecDepth 8192 in
theorem root359Candidates4_checked : candidates edge 4 2441 4157 root359Tail5 winner.bound = root359Candidates4 := by
  unfold candidates
  have row : edge 4157 = (SortedNatTree.contains FastRows.row4157) := by
    change Data.row4157.contains = (SortedNatTree.contains FastRows.row4157)
    exact FastRows.row4157_checked.symm
  rw [row]
  decide
theorem root359Step4 : scan edge (search edge 3) 4 [2441] 2441 root359Tail4 winner =
    scan edge (search edge 3) 4 [2441] 2441 root359Tail5 winner := by
  rw [root359Tail4, scan, if_neg (by decide), root359Candidates4_checked]
  change scan edge (search edge 3) 4 [2441] 2441 root359Tail5
    (if 3 ≤ root359Candidates4.length then search edge 3 [2441, 4157] root359Candidates4 6598 winner else winner) = _
  rw [if_neg (by decide)]
def root359Candidates5 : List Nat := [5297, 17207]
set_option maxRecDepth 8192 in
theorem root359Candidates5_checked : candidates edge 4 2441 4349 root359Tail6 winner.bound = root359Candidates5 := by
  unfold candidates
  have row : edge 4349 = (SortedNatTree.contains FastRows.row4349) := by
    change Data.row4349.contains = (SortedNatTree.contains FastRows.row4349)
    exact FastRows.row4349_checked.symm
  rw [row]
  decide
theorem root359Step5 : scan edge (search edge 3) 4 [2441] 2441 root359Tail5 winner =
    scan edge (search edge 3) 4 [2441] 2441 root359Tail6 winner := by
  rw [root359Tail5, scan, if_neg (by decide), root359Candidates5_checked]
  change scan edge (search edge 3) 4 [2441] 2441 root359Tail6
    (if 3 ≤ root359Candidates5.length then search edge 3 [2441, 4349] root359Candidates5 6790 winner else winner) = _
  rw [if_neg (by decide)]
def root359Candidates6 : List Nat := [9929, 10457, 17387]
set_option maxRecDepth 8192 in
theorem root359Candidates6_checked : candidates edge 4 2441 4787 root359Tail7 winner.bound = root359Candidates6 := by
  unfold candidates
  have row : edge 4787 = (SortedNatTree.contains FastRows.row4787) := by
    change Data.row4787.contains = (SortedNatTree.contains FastRows.row4787)
    exact FastRows.row4787_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root359Child6 : search edge 3 [2441, 4787] root359Candidates6 7228 winner = winner := by decide
theorem root359Step6 : scan edge (search edge 3) 4 [2441] 2441 root359Tail6 winner =
    scan edge (search edge 3) 4 [2441] 2441 root359Tail7 winner := by
  rw [root359Tail6, scan, if_neg (by decide), root359Candidates6_checked]
  change scan edge (search edge 3) 4 [2441] 2441 root359Tail7
    (if 3 ≤ root359Candidates6.length then search edge 3 [2441, 4787] root359Candidates6 7228 winner else winner) = _
  rw [if_pos (by decide), root359Child6]
def root359Candidates7 : List Nat := [10457, 15149, 17207, 17597]
set_option maxRecDepth 8192 in
theorem root359Candidates7_checked : candidates edge 4 2441 5087 root359Tail8 winner.bound = root359Candidates7 := by
  unfold candidates
  have row : edge 5087 = (SortedNatTree.contains FastRows.row5087) := by
    change Data.row5087.contains = (SortedNatTree.contains FastRows.row5087)
    exact FastRows.row5087_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root359Child7 : search edge 3 [2441, 5087] root359Candidates7 7528 winner = winner := by decide
theorem root359Step7 : scan edge (search edge 3) 4 [2441] 2441 root359Tail7 winner =
    scan edge (search edge 3) 4 [2441] 2441 root359Tail8 winner := by
  rw [root359Tail7, scan, if_neg (by decide), root359Candidates7_checked]
  change scan edge (search edge 3) 4 [2441] 2441 root359Tail8
    (if 3 ≤ root359Candidates7.length then search edge 3 [2441, 5087] root359Candidates7 7528 winner else winner) = _
  rw [if_pos (by decide), root359Child7]
def root359Candidates8 : List Nat := [5297, 15329, 15671]
set_option maxRecDepth 8192 in
theorem root359Candidates8_checked : candidates edge 4 2441 5147 root359Tail9 winner.bound = root359Candidates8 := by
  unfold candidates
  have row : edge 5147 = (SortedNatTree.contains FastRows.row5147) := by
    change Data.row5147.contains = (SortedNatTree.contains FastRows.row5147)
    exact FastRows.row5147_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root359Child8 : search edge 3 [2441, 5147] root359Candidates8 7588 winner = winner := by decide
theorem root359Step8 : scan edge (search edge 3) 4 [2441] 2441 root359Tail8 winner =
    scan edge (search edge 3) 4 [2441] 2441 root359Tail9 winner := by
  rw [root359Tail8, scan, if_neg (by decide), root359Candidates8_checked]
  change scan edge (search edge 3) 4 [2441] 2441 root359Tail9
    (if 3 ≤ root359Candidates8.length then search edge 3 [2441, 5147] root359Candidates8 7588 winner else winner) = _
  rw [if_pos (by decide), root359Child8]
def root359Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root359Candidates9_checked : candidates edge 4 2441 5297 root359Tail10 winner.bound = root359Candidates9 := by
  unfold candidates
  have row : edge 5297 = (SortedNatTree.contains FastRows.row5297) := by
    change Data.row5297.contains = (SortedNatTree.contains FastRows.row5297)
    exact FastRows.row5297_checked.symm
  rw [row]
  decide
theorem root359Step9 : scan edge (search edge 3) 4 [2441] 2441 root359Tail9 winner =
    scan edge (search edge 3) 4 [2441] 2441 root359Tail10 winner := by
  rw [root359Tail9, scan, if_neg (by decide), root359Candidates9_checked]
  change scan edge (search edge 3) 4 [2441] 2441 root359Tail10
    (if 3 ≤ root359Candidates9.length then search edge 3 [2441, 5297] root359Candidates9 7738 winner else winner) = _
  rw [if_neg (by decide)]
def root359Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root359Candidates10_checked : candidates edge 4 2441 5333 root359Tail11 winner.bound = root359Candidates10 := by
  unfold candidates
  have row : edge 5333 = (SortedNatTree.contains FastRows.row5333) := by
    change Data.row5333.contains = (SortedNatTree.contains FastRows.row5333)
    exact FastRows.row5333_checked.symm
  rw [row]
  decide
theorem root359Step10 : scan edge (search edge 3) 4 [2441] 2441 root359Tail10 winner =
    scan edge (search edge 3) 4 [2441] 2441 root359Tail11 winner := by
  rw [root359Tail10, scan, if_neg (by decide), root359Candidates10_checked]
  change scan edge (search edge 3) 4 [2441] 2441 root359Tail11
    (if 3 ≤ root359Candidates10.length then search edge 3 [2441, 5333] root359Candidates10 7774 winner else winner) = _
  rw [if_neg (by decide)]
def root359Candidates11 : List Nat := [9929]
set_option maxRecDepth 8192 in
theorem root359Candidates11_checked : candidates edge 4 2441 5381 root359Tail12 winner.bound = root359Candidates11 := by
  unfold candidates
  have row : edge 5381 = (SortedNatTree.contains FastRows.row5381) := by
    change Data.row5381.contains = (SortedNatTree.contains FastRows.row5381)
    exact FastRows.row5381_checked.symm
  rw [row]
  decide
theorem root359Step11 : scan edge (search edge 3) 4 [2441] 2441 root359Tail11 winner =
    scan edge (search edge 3) 4 [2441] 2441 root359Tail12 winner := by
  rw [root359Tail11, scan, if_neg (by decide), root359Candidates11_checked]
  change scan edge (search edge 3) 4 [2441] 2441 root359Tail12
    (if 3 ≤ root359Candidates11.length then search edge 3 [2441, 5381] root359Candidates11 7822 winner else winner) = _
  rw [if_neg (by decide)]
theorem root359Scan12 : scan edge (search edge 3) 4 [2441] 2441 root359Tail12 winner = winner := by rw [root359Tail12, scan, if_pos (by decide)]
theorem root359Scan11 : scan edge (search edge 3) 4 [2441] 2441 root359Tail11 winner = winner :=
  (root359Step11).trans root359Scan12
theorem root359Scan10 : scan edge (search edge 3) 4 [2441] 2441 root359Tail10 winner = winner :=
  (root359Step10).trans root359Scan11
theorem root359Scan9 : scan edge (search edge 3) 4 [2441] 2441 root359Tail9 winner = winner :=
  (root359Step9).trans root359Scan10
theorem root359Scan8 : scan edge (search edge 3) 4 [2441] 2441 root359Tail8 winner = winner :=
  (root359Step8).trans root359Scan9
theorem root359Scan7 : scan edge (search edge 3) 4 [2441] 2441 root359Tail7 winner = winner :=
  (root359Step7).trans root359Scan8
theorem root359Scan6 : scan edge (search edge 3) 4 [2441] 2441 root359Tail6 winner = winner :=
  (root359Step6).trans root359Scan7
theorem root359Scan5 : scan edge (search edge 3) 4 [2441] 2441 root359Tail5 winner = winner :=
  (root359Step5).trans root359Scan6
theorem root359Scan4 : scan edge (search edge 3) 4 [2441] 2441 root359Tail4 winner = winner :=
  (root359Step4).trans root359Scan5
theorem root359Scan3 : scan edge (search edge 3) 4 [2441] 2441 root359Tail3 winner = winner :=
  (root359Step3).trans root359Scan4
theorem root359Scan2 : scan edge (search edge 3) 4 [2441] 2441 root359Tail2 winner = winner :=
  (root359Step2).trans root359Scan3
theorem root359Scan1 : scan edge (search edge 3) 4 [2441] 2441 root359Tail1 winner = winner :=
  (root359Step1).trans root359Scan2
theorem root359Scan0 : scan edge (search edge 3) 4 [2441] 2441 root359Tail0 winner = winner :=
  (root359Step0).trans root359Scan1
set_option maxRecDepth 4096 in
theorem root359Branch : search edge 4 [2441] root359Candidates 2441 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2441] 2441 root359Tail0 winner = winner
  exact root359Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
