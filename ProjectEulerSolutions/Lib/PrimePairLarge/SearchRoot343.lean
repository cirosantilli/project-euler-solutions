import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root343Candidates : List Nat := Data.row2339
set_option maxRecDepth 8192 in
theorem root343Candidates_checked : candidates edge 5 0 2339 tail344 winner.bound = root343Candidates := by
  apply candidates_eq_row edge 5 0 2339 tail344 root343Candidates winner.bound tail344_ordered
  · decide
  · decide
  · funext q
    rfl
def root343Tail48 : List Nat := []
def root343Tail47 : List Nat := 23663 :: root343Tail48
def root343Tail46 : List Nat := 23603 :: root343Tail47
def root343Tail45 : List Nat := 22679 :: root343Tail46
def root343Tail44 : List Nat := 21503 :: root343Tail45
def root343Tail43 : List Nat := 21227 :: root343Tail44
def root343Tail42 : List Nat := 20807 :: root343Tail43
def root343Tail41 : List Nat := 20639 :: root343Tail42
def root343Tail40 : List Nat := 19319 :: root343Tail41
def root343Tail39 : List Nat := 18917 :: root343Tail40
def root343Tail38 : List Nat := 17903 :: root343Tail39
def root343Tail37 : List Nat := 17471 :: root343Tail38
def root343Tail36 : List Nat := 16451 :: root343Tail37
def root343Tail35 : List Nat := 16253 :: root343Tail36
def root343Tail34 : List Nat := 16127 :: root343Tail35
def root343Tail33 : List Nat := 15959 :: root343Tail34
def root343Tail32 : List Nat := 15641 :: root343Tail33
def root343Tail31 : List Nat := 15467 :: root343Tail32
def root343Tail30 : List Nat := 14939 :: root343Tail31
def root343Tail29 : List Nat := 14741 :: root343Tail30
def root343Tail28 : List Nat := 12893 :: root343Tail29
def root343Tail27 : List Nat := 11927 :: root343Tail28
def root343Tail26 : List Nat := 11351 :: root343Tail27
def root343Tail25 : List Nat := 10709 :: root343Tail26
def root343Tail24 : List Nat := 10037 :: root343Tail25
def root343Tail23 : List Nat := 9767 :: root343Tail24
def root343Tail22 : List Nat := 9743 :: root343Tail23
def root343Tail21 : List Nat := 9209 :: root343Tail22
def root343Tail20 : List Nat := 9011 :: root343Tail21
def root343Tail19 : List Nat := 8369 :: root343Tail20
def root343Tail18 : List Nat := 7841 :: root343Tail19
def root343Tail17 : List Nat := 7499 :: root343Tail18
def root343Tail16 : List Nat := 7487 :: root343Tail17
def root343Tail15 : List Nat := 7307 :: root343Tail16
def root343Tail14 : List Nat := 7247 :: root343Tail15
def root343Tail13 : List Nat := 7013 :: root343Tail14
def root343Tail12 : List Nat := 7001 :: root343Tail13
def root343Tail11 : List Nat := 6287 :: root343Tail12
def root343Tail10 : List Nat := 6101 :: root343Tail11
def root343Tail9 : List Nat := 6047 :: root343Tail10
def root343Tail8 : List Nat := 5303 :: root343Tail9
def root343Tail7 : List Nat := 5081 :: root343Tail8
def root343Tail6 : List Nat := 5039 :: root343Tail7
def root343Tail5 : List Nat := 4337 :: root343Tail6
def root343Tail4 : List Nat := 4127 :: root343Tail5
def root343Tail3 : List Nat := 3371 :: root343Tail4
def root343Tail2 : List Nat := 3347 :: root343Tail3
def root343Tail1 : List Nat := 2729 :: root343Tail2
def root343Tail0 : List Nat := 2543 :: root343Tail1
def root343Candidates0 : List Nat := [3371, 4337, 12893, 16253]
set_option maxRecDepth 8192 in
theorem root343Candidates0_checked : candidates edge 4 2339 2543 root343Tail1 winner.bound = root343Candidates0 := by
  unfold candidates
  have row : edge 2543 = (SortedNatTree.contains FastRows.row2543) := by
    change Data.row2543.contains = (SortedNatTree.contains FastRows.row2543)
    exact FastRows.row2543_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root343Child0 : search edge 3 [2339, 2543] root343Candidates0 4882 winner = winner := by decide
theorem root343Step0 : scan edge (search edge 3) 4 [2339] 2339 root343Tail0 winner =
    scan edge (search edge 3) 4 [2339] 2339 root343Tail1 winner := by
  rw [root343Tail0, scan, if_neg (by decide), root343Candidates0_checked]
  change scan edge (search edge 3) 4 [2339] 2339 root343Tail1
    (if 3 ≤ root343Candidates0.length then search edge 3 [2339, 2543] root343Candidates0 4882 winner else winner) = _
  rw [if_pos (by decide), root343Child0]
def root343Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root343Candidates1_checked : candidates edge 4 2339 2729 root343Tail2 winner.bound = root343Candidates1 := by
  unfold candidates
  have row : edge 2729 = (SortedNatTree.contains FastRows.row2729) := by
    change Data.row2729.contains = (SortedNatTree.contains FastRows.row2729)
    exact FastRows.row2729_checked.symm
  rw [row]
  decide
theorem root343Step1 : scan edge (search edge 3) 4 [2339] 2339 root343Tail1 winner =
    scan edge (search edge 3) 4 [2339] 2339 root343Tail2 winner := by
  rw [root343Tail1, scan, if_neg (by decide), root343Candidates1_checked]
  change scan edge (search edge 3) 4 [2339] 2339 root343Tail2
    (if 3 ≤ root343Candidates1.length then search edge 3 [2339, 2729] root343Candidates1 5068 winner else winner) = _
  rw [if_neg (by decide)]
def root343Candidates2 : List Nat := [6101]
set_option maxRecDepth 8192 in
theorem root343Candidates2_checked : candidates edge 4 2339 3347 root343Tail3 winner.bound = root343Candidates2 := by
  unfold candidates
  have row : edge 3347 = (SortedNatTree.contains FastRows.row3347) := by
    change Data.row3347.contains = (SortedNatTree.contains FastRows.row3347)
    exact FastRows.row3347_checked.symm
  rw [row]
  decide
theorem root343Step2 : scan edge (search edge 3) 4 [2339] 2339 root343Tail2 winner =
    scan edge (search edge 3) 4 [2339] 2339 root343Tail3 winner := by
  rw [root343Tail2, scan, if_neg (by decide), root343Candidates2_checked]
  change scan edge (search edge 3) 4 [2339] 2339 root343Tail3
    (if 3 ≤ root343Candidates2.length then search edge 3 [2339, 3347] root343Candidates2 5686 winner else winner) = _
  rw [if_neg (by decide)]
def root343Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root343Candidates3_checked : candidates edge 4 2339 3371 root343Tail4 winner.bound = root343Candidates3 := by
  unfold candidates
  have row : edge 3371 = (SortedNatTree.contains FastRows.row3371) := by
    change Data.row3371.contains = (SortedNatTree.contains FastRows.row3371)
    exact FastRows.row3371_checked.symm
  rw [row]
  decide
theorem root343Step3 : scan edge (search edge 3) 4 [2339] 2339 root343Tail3 winner =
    scan edge (search edge 3) 4 [2339] 2339 root343Tail4 winner := by
  rw [root343Tail3, scan, if_neg (by decide), root343Candidates3_checked]
  change scan edge (search edge 3) 4 [2339] 2339 root343Tail4
    (if 3 ≤ root343Candidates3.length then search edge 3 [2339, 3371] root343Candidates3 5710 winner else winner) = _
  rw [if_neg (by decide)]
def root343Candidates4 : List Nat := [9767]
set_option maxRecDepth 8192 in
theorem root343Candidates4_checked : candidates edge 4 2339 4127 root343Tail5 winner.bound = root343Candidates4 := by
  unfold candidates
  have row : edge 4127 = (SortedNatTree.contains FastRows.row4127) := by
    change Data.row4127.contains = (SortedNatTree.contains FastRows.row4127)
    exact FastRows.row4127_checked.symm
  rw [row]
  decide
theorem root343Step4 : scan edge (search edge 3) 4 [2339] 2339 root343Tail4 winner =
    scan edge (search edge 3) 4 [2339] 2339 root343Tail5 winner := by
  rw [root343Tail4, scan, if_neg (by decide), root343Candidates4_checked]
  change scan edge (search edge 3) 4 [2339] 2339 root343Tail5
    (if 3 ≤ root343Candidates4.length then search edge 3 [2339, 4127] root343Candidates4 6466 winner else winner) = _
  rw [if_neg (by decide)]
def root343Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root343Candidates5_checked : candidates edge 4 2339 4337 root343Tail6 winner.bound = root343Candidates5 := by
  unfold candidates
  have row : edge 4337 = (SortedNatTree.contains FastRows.row4337) := by
    change Data.row4337.contains = (SortedNatTree.contains FastRows.row4337)
    exact FastRows.row4337_checked.symm
  rw [row]
  decide
theorem root343Step5 : scan edge (search edge 3) 4 [2339] 2339 root343Tail5 winner =
    scan edge (search edge 3) 4 [2339] 2339 root343Tail6 winner := by
  rw [root343Tail5, scan, if_neg (by decide), root343Candidates5_checked]
  change scan edge (search edge 3) 4 [2339] 2339 root343Tail6
    (if 3 ≤ root343Candidates5.length then search edge 3 [2339, 4337] root343Candidates5 6676 winner else winner) = _
  rw [if_neg (by decide)]
def root343Candidates6 : List Nat := [7001]
set_option maxRecDepth 8192 in
theorem root343Candidates6_checked : candidates edge 4 2339 5039 root343Tail7 winner.bound = root343Candidates6 := by
  unfold candidates
  have row : edge 5039 = (SortedNatTree.contains FastRows.row5039) := by
    change Data.row5039.contains = (SortedNatTree.contains FastRows.row5039)
    exact FastRows.row5039_checked.symm
  rw [row]
  decide
theorem root343Step6 : scan edge (search edge 3) 4 [2339] 2339 root343Tail6 winner =
    scan edge (search edge 3) 4 [2339] 2339 root343Tail7 winner := by
  rw [root343Tail6, scan, if_neg (by decide), root343Candidates6_checked]
  change scan edge (search edge 3) 4 [2339] 2339 root343Tail7
    (if 3 ≤ root343Candidates6.length then search edge 3 [2339, 5039] root343Candidates6 7378 winner else winner) = _
  rw [if_neg (by decide)]
def root343Candidates7 : List Nat := [7307]
set_option maxRecDepth 8192 in
theorem root343Candidates7_checked : candidates edge 4 2339 5081 root343Tail8 winner.bound = root343Candidates7 := by
  unfold candidates
  have row : edge 5081 = (SortedNatTree.contains FastRows.row5081) := by
    change Data.row5081.contains = (SortedNatTree.contains FastRows.row5081)
    exact FastRows.row5081_checked.symm
  rw [row]
  decide
theorem root343Step7 : scan edge (search edge 3) 4 [2339] 2339 root343Tail7 winner =
    scan edge (search edge 3) 4 [2339] 2339 root343Tail8 winner := by
  rw [root343Tail7, scan, if_neg (by decide), root343Candidates7_checked]
  change scan edge (search edge 3) 4 [2339] 2339 root343Tail8
    (if 3 ≤ root343Candidates7.length then search edge 3 [2339, 5081] root343Candidates7 7420 winner else winner) = _
  rw [if_neg (by decide)]
def root343Candidates8 : List Nat := [6101]
set_option maxRecDepth 8192 in
theorem root343Candidates8_checked : candidates edge 4 2339 5303 root343Tail9 winner.bound = root343Candidates8 := by
  unfold candidates
  have row : edge 5303 = (SortedNatTree.contains FastRows.row5303) := by
    change Data.row5303.contains = (SortedNatTree.contains FastRows.row5303)
    exact FastRows.row5303_checked.symm
  rw [row]
  decide
theorem root343Step8 : scan edge (search edge 3) 4 [2339] 2339 root343Tail8 winner =
    scan edge (search edge 3) 4 [2339] 2339 root343Tail9 winner := by
  rw [root343Tail8, scan, if_neg (by decide), root343Candidates8_checked]
  change scan edge (search edge 3) 4 [2339] 2339 root343Tail9
    (if 3 ≤ root343Candidates8.length then search edge 3 [2339, 5303] root343Candidates8 7642 winner else winner) = _
  rw [if_neg (by decide)]
theorem root343Scan9 : scan edge (search edge 3) 4 [2339] 2339 root343Tail9 winner = winner := by rw [root343Tail9, scan, if_pos (by decide)]
theorem root343Scan8 : scan edge (search edge 3) 4 [2339] 2339 root343Tail8 winner = winner :=
  (root343Step8).trans root343Scan9
theorem root343Scan7 : scan edge (search edge 3) 4 [2339] 2339 root343Tail7 winner = winner :=
  (root343Step7).trans root343Scan8
theorem root343Scan6 : scan edge (search edge 3) 4 [2339] 2339 root343Tail6 winner = winner :=
  (root343Step6).trans root343Scan7
theorem root343Scan5 : scan edge (search edge 3) 4 [2339] 2339 root343Tail5 winner = winner :=
  (root343Step5).trans root343Scan6
theorem root343Scan4 : scan edge (search edge 3) 4 [2339] 2339 root343Tail4 winner = winner :=
  (root343Step4).trans root343Scan5
theorem root343Scan3 : scan edge (search edge 3) 4 [2339] 2339 root343Tail3 winner = winner :=
  (root343Step3).trans root343Scan4
theorem root343Scan2 : scan edge (search edge 3) 4 [2339] 2339 root343Tail2 winner = winner :=
  (root343Step2).trans root343Scan3
theorem root343Scan1 : scan edge (search edge 3) 4 [2339] 2339 root343Tail1 winner = winner :=
  (root343Step1).trans root343Scan2
theorem root343Scan0 : scan edge (search edge 3) 4 [2339] 2339 root343Tail0 winner = winner :=
  (root343Step0).trans root343Scan1
set_option maxRecDepth 4096 in
theorem root343Branch : search edge 4 [2339] root343Candidates 2339 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2339] 2339 root343Tail0 winner = winner
  exact root343Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
