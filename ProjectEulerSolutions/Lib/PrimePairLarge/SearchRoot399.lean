import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root399Candidates : List Nat := Data.row2753
set_option maxRecDepth 8192 in
theorem root399Candidates_checked : candidates edge 5 0 2753 tail400 winner.bound = root399Candidates := by
  apply candidates_eq_row edge 5 0 2753 tail400 root399Candidates winner.bound tail400_ordered
  · decide
  · decide
  · funext q
    rfl
def root399Tail58 : List Nat := []
def root399Tail57 : List Nat := 22727 :: root399Tail58
def root399Tail56 : List Nat := 22643 :: root399Tail57
def root399Tail55 : List Nat := 22247 :: root399Tail56
def root399Tail54 : List Nat := 21611 :: root399Tail55
def root399Tail53 : List Nat := 20939 :: root399Tail54
def root399Tail52 : List Nat := 20249 :: root399Tail53
def root399Tail51 : List Nat := 19739 :: root399Tail52
def root399Tail50 : List Nat := 19571 :: root399Tail51
def root399Tail49 : List Nat := 18089 :: root399Tail50
def root399Tail48 : List Nat := 18041 :: root399Tail49
def root399Tail47 : List Nat := 17891 :: root399Tail48
def root399Tail46 : List Nat := 17477 :: root399Tail47
def root399Tail45 : List Nat := 16661 :: root399Tail46
def root399Tail44 : List Nat := 16253 :: root399Tail45
def root399Tail43 : List Nat := 15791 :: root399Tail44
def root399Tail42 : List Nat := 15737 :: root399Tail43
def root399Tail41 : List Nat := 15569 :: root399Tail42
def root399Tail40 : List Nat := 15413 :: root399Tail41
def root399Tail39 : List Nat := 15101 :: root399Tail40
def root399Tail38 : List Nat := 14867 :: root399Tail39
def root399Tail37 : List Nat := 14771 :: root399Tail38
def root399Tail36 : List Nat := 14519 :: root399Tail37
def root399Tail35 : List Nat := 14387 :: root399Tail36
def root399Tail34 : List Nat := 14321 :: root399Tail35
def root399Tail33 : List Nat := 14243 :: root399Tail34
def root399Tail32 : List Nat := 14051 :: root399Tail33
def root399Tail31 : List Nat := 13721 :: root399Tail32
def root399Tail30 : List Nat := 12497 :: root399Tail31
def root399Tail29 : List Nat := 12473 :: root399Tail30
def root399Tail28 : List Nat := 12347 :: root399Tail29
def root399Tail27 : List Nat := 12329 :: root399Tail28
def root399Tail26 : List Nat := 12269 :: root399Tail27
def root399Tail25 : List Nat := 11579 :: root399Tail26
def root399Tail24 : List Nat := 9923 :: root399Tail25
def root399Tail23 : List Nat := 9689 :: root399Tail24
def root399Tail22 : List Nat := 9293 :: root399Tail23
def root399Tail21 : List Nat := 9203 :: root399Tail22
def root399Tail20 : List Nat := 9059 :: root399Tail21
def root399Tail19 : List Nat := 8933 :: root399Tail20
def root399Tail18 : List Nat := 8831 :: root399Tail19
def root399Tail17 : List Nat := 8597 :: root399Tail18
def root399Tail16 : List Nat := 7919 :: root399Tail17
def root399Tail15 : List Nat := 7529 :: root399Tail16
def root399Tail14 : List Nat := 7487 :: root399Tail15
def root399Tail13 : List Nat := 7481 :: root399Tail14
def root399Tail12 : List Nat := 7457 :: root399Tail13
def root399Tail11 : List Nat := 7001 :: root399Tail12
def root399Tail10 : List Nat := 6791 :: root399Tail11
def root399Tail9 : List Nat := 6287 :: root399Tail10
def root399Tail8 : List Nat := 6101 :: root399Tail9
def root399Tail7 : List Nat := 5171 :: root399Tail8
def root399Tail6 : List Nat := 4799 :: root399Tail7
def root399Tail5 : List Nat := 4583 :: root399Tail6
def root399Tail4 : List Nat := 3371 :: root399Tail5
def root399Tail3 : List Nat := 3203 :: root399Tail4
def root399Tail2 : List Nat := 3089 :: root399Tail3
def root399Tail1 : List Nat := 2963 :: root399Tail2
def root399Tail0 : List Nat := 2927 :: root399Tail1
def root399Candidates0 : List Nat := [3203]
set_option maxRecDepth 8192 in
theorem root399Candidates0_checked : candidates edge 4 2753 2927 root399Tail1 winner.bound = root399Candidates0 := by
  unfold candidates
  have row : edge 2927 = (SortedNatTree.contains FastRows.row2927) := by
    change Data.row2927.contains = (SortedNatTree.contains FastRows.row2927)
    exact FastRows.row2927_checked.symm
  rw [row]
  decide
theorem root399Step0 : scan edge (search edge 3) 4 [2753] 2753 root399Tail0 winner =
    scan edge (search edge 3) 4 [2753] 2753 root399Tail1 winner := by
  rw [root399Tail0, scan, if_neg (by decide), root399Candidates0_checked]
  change scan edge (search edge 3) 4 [2753] 2753 root399Tail1
    (if 3 ≤ root399Candidates0.length then search edge 3 [2753, 2927] root399Candidates0 5680 winner else winner) = _
  rw [if_neg (by decide)]
def root399Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root399Candidates1_checked : candidates edge 4 2753 2963 root399Tail2 winner.bound = root399Candidates1 := by
  unfold candidates
  have row : edge 2963 = (SortedNatTree.contains FastRows.row2963) := by
    change Data.row2963.contains = (SortedNatTree.contains FastRows.row2963)
    exact FastRows.row2963_checked.symm
  rw [row]
  decide
theorem root399Step1 : scan edge (search edge 3) 4 [2753] 2753 root399Tail1 winner =
    scan edge (search edge 3) 4 [2753] 2753 root399Tail2 winner := by
  rw [root399Tail1, scan, if_neg (by decide), root399Candidates1_checked]
  change scan edge (search edge 3) 4 [2753] 2753 root399Tail2
    (if 3 ≤ root399Candidates1.length then search edge 3 [2753, 2963] root399Candidates1 5716 winner else winner) = _
  rw [if_neg (by decide)]
def root399Candidates2 : List Nat := [7457, 9203, 11579]
set_option maxRecDepth 8192 in
theorem root399Candidates2_checked : candidates edge 4 2753 3089 root399Tail3 winner.bound = root399Candidates2 := by
  unfold candidates
  have row : edge 3089 = (SortedNatTree.contains FastRows.row3089) := by
    change Data.row3089.contains = (SortedNatTree.contains FastRows.row3089)
    exact FastRows.row3089_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root399Child2 : search edge 3 [2753, 3089] root399Candidates2 5842 winner = winner := by decide
theorem root399Step2 : scan edge (search edge 3) 4 [2753] 2753 root399Tail2 winner =
    scan edge (search edge 3) 4 [2753] 2753 root399Tail3 winner := by
  rw [root399Tail2, scan, if_neg (by decide), root399Candidates2_checked]
  change scan edge (search edge 3) 4 [2753] 2753 root399Tail3
    (if 3 ≤ root399Candidates2.length then search edge 3 [2753, 3089] root399Candidates2 5842 winner else winner) = _
  rw [if_pos (by decide), root399Child2]
def root399Candidates3 : List Nat := [6101, 8597, 8933, 9923]
set_option maxRecDepth 8192 in
theorem root399Candidates3_checked : candidates edge 4 2753 3203 root399Tail4 winner.bound = root399Candidates3 := by
  unfold candidates
  have row : edge 3203 = (SortedNatTree.contains FastRows.row3203) := by
    change Data.row3203.contains = (SortedNatTree.contains FastRows.row3203)
    exact FastRows.row3203_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root399Child3 : search edge 3 [2753, 3203] root399Candidates3 5956 winner = winner := by decide
theorem root399Step3 : scan edge (search edge 3) 4 [2753] 2753 root399Tail3 winner =
    scan edge (search edge 3) 4 [2753] 2753 root399Tail4 winner := by
  rw [root399Tail3, scan, if_neg (by decide), root399Candidates3_checked]
  change scan edge (search edge 3) 4 [2753] 2753 root399Tail4
    (if 3 ≤ root399Candidates3.length then search edge 3 [2753, 3203] root399Candidates3 5956 winner else winner) = _
  rw [if_pos (by decide), root399Child3]
def root399Candidates4 : List Nat := [4799]
set_option maxRecDepth 8192 in
theorem root399Candidates4_checked : candidates edge 4 2753 3371 root399Tail5 winner.bound = root399Candidates4 := by
  unfold candidates
  have row : edge 3371 = (SortedNatTree.contains FastRows.row3371) := by
    change Data.row3371.contains = (SortedNatTree.contains FastRows.row3371)
    exact FastRows.row3371_checked.symm
  rw [row]
  decide
theorem root399Step4 : scan edge (search edge 3) 4 [2753] 2753 root399Tail4 winner =
    scan edge (search edge 3) 4 [2753] 2753 root399Tail5 winner := by
  rw [root399Tail4, scan, if_neg (by decide), root399Candidates4_checked]
  change scan edge (search edge 3) 4 [2753] 2753 root399Tail5
    (if 3 ≤ root399Candidates4.length then search edge 3 [2753, 3371] root399Candidates4 6124 winner else winner) = _
  rw [if_neg (by decide)]
def root399Candidates5 : List Nat := [17477]
set_option maxRecDepth 8192 in
theorem root399Candidates5_checked : candidates edge 4 2753 4583 root399Tail6 winner.bound = root399Candidates5 := by
  unfold candidates
  have row : edge 4583 = (SortedNatTree.contains FastRows.row4583) := by
    change Data.row4583.contains = (SortedNatTree.contains FastRows.row4583)
    exact FastRows.row4583_checked.symm
  rw [row]
  decide
theorem root399Step5 : scan edge (search edge 3) 4 [2753] 2753 root399Tail5 winner =
    scan edge (search edge 3) 4 [2753] 2753 root399Tail6 winner := by
  rw [root399Tail5, scan, if_neg (by decide), root399Candidates5_checked]
  change scan edge (search edge 3) 4 [2753] 2753 root399Tail6
    (if 3 ≤ root399Candidates5.length then search edge 3 [2753, 4583] root399Candidates5 7336 winner else winner) = _
  rw [if_neg (by decide)]
def root399Candidates6 : List Nat := [7487, 8597]
set_option maxRecDepth 8192 in
theorem root399Candidates6_checked : candidates edge 4 2753 4799 root399Tail7 winner.bound = root399Candidates6 := by
  unfold candidates
  have row : edge 4799 = (SortedNatTree.contains FastRows.row4799) := by
    change Data.row4799.contains = (SortedNatTree.contains FastRows.row4799)
    exact FastRows.row4799_checked.symm
  rw [row]
  decide
theorem root399Step6 : scan edge (search edge 3) 4 [2753] 2753 root399Tail6 winner =
    scan edge (search edge 3) 4 [2753] 2753 root399Tail7 winner := by
  rw [root399Tail6, scan, if_neg (by decide), root399Candidates6_checked]
  change scan edge (search edge 3) 4 [2753] 2753 root399Tail7
    (if 3 ≤ root399Candidates6.length then search edge 3 [2753, 4799] root399Candidates6 7552 winner else winner) = _
  rw [if_neg (by decide)]
def root399Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root399Candidates7_checked : candidates edge 4 2753 5171 root399Tail8 winner.bound = root399Candidates7 := by
  unfold candidates
  have row : edge 5171 = (SortedNatTree.contains FastRows.row5171) := by
    change Data.row5171.contains = (SortedNatTree.contains FastRows.row5171)
    exact FastRows.row5171_checked.symm
  rw [row]
  decide
theorem root399Step7 : scan edge (search edge 3) 4 [2753] 2753 root399Tail7 winner =
    scan edge (search edge 3) 4 [2753] 2753 root399Tail8 winner := by
  rw [root399Tail7, scan, if_neg (by decide), root399Candidates7_checked]
  change scan edge (search edge 3) 4 [2753] 2753 root399Tail8
    (if 3 ≤ root399Candidates7.length then search edge 3 [2753, 5171] root399Candidates7 7924 winner else winner) = _
  rw [if_neg (by decide)]
theorem root399Scan8 : scan edge (search edge 3) 4 [2753] 2753 root399Tail8 winner = winner := by rw [root399Tail8, scan, if_pos (by decide)]
theorem root399Scan7 : scan edge (search edge 3) 4 [2753] 2753 root399Tail7 winner = winner :=
  (root399Step7).trans root399Scan8
theorem root399Scan6 : scan edge (search edge 3) 4 [2753] 2753 root399Tail6 winner = winner :=
  (root399Step6).trans root399Scan7
theorem root399Scan5 : scan edge (search edge 3) 4 [2753] 2753 root399Tail5 winner = winner :=
  (root399Step5).trans root399Scan6
theorem root399Scan4 : scan edge (search edge 3) 4 [2753] 2753 root399Tail4 winner = winner :=
  (root399Step4).trans root399Scan5
theorem root399Scan3 : scan edge (search edge 3) 4 [2753] 2753 root399Tail3 winner = winner :=
  (root399Step3).trans root399Scan4
theorem root399Scan2 : scan edge (search edge 3) 4 [2753] 2753 root399Tail2 winner = winner :=
  (root399Step2).trans root399Scan3
theorem root399Scan1 : scan edge (search edge 3) 4 [2753] 2753 root399Tail1 winner = winner :=
  (root399Step1).trans root399Scan2
theorem root399Scan0 : scan edge (search edge 3) 4 [2753] 2753 root399Tail0 winner = winner :=
  (root399Step0).trans root399Scan1
set_option maxRecDepth 4096 in
theorem root399Branch : search edge 4 [2753] root399Candidates 2753 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2753] 2753 root399Tail0 winner = winner
  exact root399Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
