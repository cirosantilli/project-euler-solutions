import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root475Candidates : List Nat := Data.row3391
set_option maxRecDepth 8192 in
theorem root475Candidates_checked : candidates edge 5 0 3391 tail476 winner.bound = root475Candidates := by
  apply candidates_eq_row edge 5 0 3391 tail476 root475Candidates winner.bound tail476_ordered
  · decide
  · decide
  · funext q
    rfl
def root475Tail47 : List Nat := []
def root475Tail46 : List Nat := 22369 :: root475Tail47
def root475Tail45 : List Nat := 21961 :: root475Tail46
def root475Tail44 : List Nat := 21067 :: root475Tail45
def root475Tail43 : List Nat := 20323 :: root475Tail44
def root475Tail42 : List Nat := 19861 :: root475Tail43
def root475Tail41 : List Nat := 18427 :: root475Tail42
def root475Tail40 : List Nat := 18133 :: root475Tail41
def root475Tail39 : List Nat := 17341 :: root475Tail40
def root475Tail38 : List Nat := 17209 :: root475Tail39
def root475Tail37 : List Nat := 16903 :: root475Tail38
def root475Tail36 : List Nat := 16231 :: root475Tail37
def root475Tail35 : List Nat := 15817 :: root475Tail36
def root475Tail34 : List Nat := 15739 :: root475Tail35
def root475Tail33 : List Nat := 15559 :: root475Tail34
def root475Tail32 : List Nat := 14947 :: root475Tail33
def root475Tail31 : List Nat := 14887 :: root475Tail32
def root475Tail30 : List Nat := 14767 :: root475Tail31
def root475Tail29 : List Nat := 14437 :: root475Tail30
def root475Tail28 : List Nat := 13627 :: root475Tail29
def root475Tail27 : List Nat := 13597 :: root475Tail28
def root475Tail26 : List Nat := 12547 :: root475Tail27
def root475Tail25 : List Nat := 12421 :: root475Tail26
def root475Tail24 : List Nat := 12049 :: root475Tail25
def root475Tail23 : List Nat := 11491 :: root475Tail24
def root475Tail22 : List Nat := 10939 :: root475Tail23
def root475Tail21 : List Nat := 10531 :: root475Tail22
def root475Tail20 : List Nat := 10267 :: root475Tail21
def root475Tail19 : List Nat := 9817 :: root475Tail20
def root475Tail18 : List Nat := 9463 :: root475Tail19
def root475Tail17 : List Nat := 9439 :: root475Tail18
def root475Tail16 : List Nat := 8431 :: root475Tail17
def root475Tail15 : List Nat := 8329 :: root475Tail16
def root475Tail14 : List Nat := 7237 :: root475Tail15
def root475Tail13 : List Nat := 6997 :: root475Tail14
def root475Tail12 : List Nat := 6907 :: root475Tail13
def root475Tail11 : List Nat := 6733 :: root475Tail12
def root475Tail10 : List Nat := 6607 :: root475Tail11
def root475Tail9 : List Nat := 6361 :: root475Tail10
def root475Tail8 : List Nat := 6199 :: root475Tail9
def root475Tail7 : List Nat := 6079 :: root475Tail8
def root475Tail6 : List Nat := 5443 :: root475Tail7
def root475Tail5 : List Nat := 5227 :: root475Tail6
def root475Tail4 : List Nat := 5113 :: root475Tail5
def root475Tail3 : List Nat := 5107 :: root475Tail4
def root475Tail2 : List Nat := 3643 :: root475Tail3
def root475Tail1 : List Nat := 3511 :: root475Tail2
def root475Tail0 : List Nat := 3433 :: root475Tail1
def root475Candidates0 : List Nat := [3643, 6607, 6733, 14887, 18427]
set_option maxRecDepth 8192 in
theorem root475Candidates0_checked : candidates edge 4 3391 3433 root475Tail1 winner.bound = root475Candidates0 := by
  unfold candidates
  have row : edge 3433 = (SortedNatTree.contains FastRows.row3433) := by
    change Data.row3433.contains = (SortedNatTree.contains FastRows.row3433)
    exact FastRows.row3433_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root475Child0 : search edge 3 [3391, 3433] root475Candidates0 6824 winner = winner := by decide
theorem root475Step0 : scan edge (search edge 3) 4 [3391] 3391 root475Tail0 winner =
    scan edge (search edge 3) 4 [3391] 3391 root475Tail1 winner := by
  rw [root475Tail0, scan, if_neg (by decide), root475Candidates0_checked]
  change scan edge (search edge 3) 4 [3391] 3391 root475Tail1
    (if 3 ≤ root475Candidates0.length then search edge 3 [3391, 3433] root475Candidates0 6824 winner else winner) = _
  rw [if_pos (by decide), root475Child0]
def root475Candidates1 : List Nat := [9439, 16231]
set_option maxRecDepth 8192 in
theorem root475Candidates1_checked : candidates edge 4 3391 3511 root475Tail2 winner.bound = root475Candidates1 := by
  unfold candidates
  have row : edge 3511 = (SortedNatTree.contains FastRows.row3511) := by
    change Data.row3511.contains = (SortedNatTree.contains FastRows.row3511)
    exact FastRows.row3511_checked.symm
  rw [row]
  decide
theorem root475Step1 : scan edge (search edge 3) 4 [3391] 3391 root475Tail1 winner =
    scan edge (search edge 3) 4 [3391] 3391 root475Tail2 winner := by
  rw [root475Tail1, scan, if_neg (by decide), root475Candidates1_checked]
  change scan edge (search edge 3) 4 [3391] 3391 root475Tail2
    (if 3 ≤ root475Candidates1.length then search edge 3 [3391, 3511] root475Candidates1 6902 winner else winner) = _
  rw [if_neg (by decide)]
def root475Candidates2 : List Nat := [6199, 6607, 9817, 15817, 16231]
set_option maxRecDepth 8192 in
theorem root475Candidates2_checked : candidates edge 4 3391 3643 root475Tail3 winner.bound = root475Candidates2 := by
  unfold candidates
  have row : edge 3643 = (SortedNatTree.contains FastRows.row3643) := by
    change Data.row3643.contains = (SortedNatTree.contains FastRows.row3643)
    exact FastRows.row3643_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root475Child2 : search edge 3 [3391, 3643] root475Candidates2 7034 winner = winner := by decide
theorem root475Step2 : scan edge (search edge 3) 4 [3391] 3391 root475Tail2 winner =
    scan edge (search edge 3) 4 [3391] 3391 root475Tail3 winner := by
  rw [root475Tail2, scan, if_neg (by decide), root475Candidates2_checked]
  change scan edge (search edge 3) 4 [3391] 3391 root475Tail3
    (if 3 ≤ root475Candidates2.length then search edge 3 [3391, 3643] root475Candidates2 7034 winner else winner) = _
  rw [if_pos (by decide), root475Child2]
def root475Candidates3 : List Nat := [16231, 17341]
set_option maxRecDepth 8192 in
theorem root475Candidates3_checked : candidates edge 4 3391 5107 root475Tail4 winner.bound = root475Candidates3 := by
  unfold candidates
  have row : edge 5107 = (SortedNatTree.contains FastRows.row5107) := by
    change Data.row5107.contains = (SortedNatTree.contains FastRows.row5107)
    exact FastRows.row5107_checked.symm
  rw [row]
  decide
theorem root475Step3 : scan edge (search edge 3) 4 [3391] 3391 root475Tail3 winner =
    scan edge (search edge 3) 4 [3391] 3391 root475Tail4 winner := by
  rw [root475Tail3, scan, if_neg (by decide), root475Candidates3_checked]
  change scan edge (search edge 3) 4 [3391] 3391 root475Tail4
    (if 3 ≤ root475Candidates3.length then search edge 3 [3391, 5107] root475Candidates3 8498 winner else winner) = _
  rw [if_neg (by decide)]
def root475Candidates4 : List Nat := [14767]
set_option maxRecDepth 8192 in
theorem root475Candidates4_checked : candidates edge 4 3391 5113 root475Tail5 winner.bound = root475Candidates4 := by
  unfold candidates
  have row : edge 5113 = (SortedNatTree.contains FastRows.row5113) := by
    change Data.row5113.contains = (SortedNatTree.contains FastRows.row5113)
    exact FastRows.row5113_checked.symm
  rw [row]
  decide
theorem root475Step4 : scan edge (search edge 3) 4 [3391] 3391 root475Tail4 winner =
    scan edge (search edge 3) 4 [3391] 3391 root475Tail5 winner := by
  rw [root475Tail4, scan, if_neg (by decide), root475Candidates4_checked]
  change scan edge (search edge 3) 4 [3391] 3391 root475Tail5
    (if 3 ≤ root475Candidates4.length then search edge 3 [3391, 5113] root475Candidates4 8504 winner else winner) = _
  rw [if_neg (by decide)]
def root475Candidates5 : List Nat := [9817]
set_option maxRecDepth 8192 in
theorem root475Candidates5_checked : candidates edge 4 3391 5227 root475Tail6 winner.bound = root475Candidates5 := by
  unfold candidates
  have row : edge 5227 = (SortedNatTree.contains FastRows.row5227) := by
    change Data.row5227.contains = (SortedNatTree.contains FastRows.row5227)
    exact FastRows.row5227_checked.symm
  rw [row]
  decide
theorem root475Step5 : scan edge (search edge 3) 4 [3391] 3391 root475Tail5 winner =
    scan edge (search edge 3) 4 [3391] 3391 root475Tail6 winner := by
  rw [root475Tail5, scan, if_neg (by decide), root475Candidates5_checked]
  change scan edge (search edge 3) 4 [3391] 3391 root475Tail6
    (if 3 ≤ root475Candidates5.length then search edge 3 [3391, 5227] root475Candidates5 8618 winner else winner) = _
  rw [if_neg (by decide)]
def root475Candidates6 : List Nat := [6079, 12547, 15559]
set_option maxRecDepth 8192 in
theorem root475Candidates6_checked : candidates edge 4 3391 5443 root475Tail7 winner.bound = root475Candidates6 := by
  unfold candidates
  have row : edge 5443 = (SortedNatTree.contains FastRows.row5443) := by
    change Data.row5443.contains = (SortedNatTree.contains FastRows.row5443)
    exact FastRows.row5443_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root475Child6 : search edge 3 [3391, 5443] root475Candidates6 8834 winner = winner := by decide
theorem root475Step6 : scan edge (search edge 3) 4 [3391] 3391 root475Tail6 winner =
    scan edge (search edge 3) 4 [3391] 3391 root475Tail7 winner := by
  rw [root475Tail6, scan, if_neg (by decide), root475Candidates6_checked]
  change scan edge (search edge 3) 4 [3391] 3391 root475Tail7
    (if 3 ≤ root475Candidates6.length then search edge 3 [3391, 5443] root475Candidates6 8834 winner else winner) = _
  rw [if_pos (by decide), root475Child6]
theorem root475Scan7 : scan edge (search edge 3) 4 [3391] 3391 root475Tail7 winner = winner := by rw [root475Tail7, scan, if_pos (by decide)]
theorem root475Scan6 : scan edge (search edge 3) 4 [3391] 3391 root475Tail6 winner = winner :=
  (root475Step6).trans root475Scan7
theorem root475Scan5 : scan edge (search edge 3) 4 [3391] 3391 root475Tail5 winner = winner :=
  (root475Step5).trans root475Scan6
theorem root475Scan4 : scan edge (search edge 3) 4 [3391] 3391 root475Tail4 winner = winner :=
  (root475Step4).trans root475Scan5
theorem root475Scan3 : scan edge (search edge 3) 4 [3391] 3391 root475Tail3 winner = winner :=
  (root475Step3).trans root475Scan4
theorem root475Scan2 : scan edge (search edge 3) 4 [3391] 3391 root475Tail2 winner = winner :=
  (root475Step2).trans root475Scan3
theorem root475Scan1 : scan edge (search edge 3) 4 [3391] 3391 root475Tail1 winner = winner :=
  (root475Step1).trans root475Scan2
theorem root475Scan0 : scan edge (search edge 3) 4 [3391] 3391 root475Tail0 winner = winner :=
  (root475Step0).trans root475Scan1
set_option maxRecDepth 4096 in
theorem root475Branch : search edge 4 [3391] root475Candidates 3391 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3391] 3391 root475Tail0 winner = winner
  exact root475Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
