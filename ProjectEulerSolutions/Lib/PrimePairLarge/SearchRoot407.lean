import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root407Candidates : List Nat := Data.row2819
set_option maxRecDepth 8192 in
theorem root407Candidates_checked : candidates edge 5 0 2819 tail408 winner.bound = root407Candidates := by
  apply candidates_eq_row edge 5 0 2819 tail408 root407Candidates winner.bound tail408_ordered
  · decide
  · decide
  · funext q
    rfl
def root407Tail39 : List Nat := []
def root407Tail38 : List Nat := 23039 :: root407Tail39
def root407Tail37 : List Nat := 22787 :: root407Tail38
def root407Tail36 : List Nat := 22307 :: root407Tail37
def root407Tail35 : List Nat := 22073 :: root407Tail36
def root407Tail34 : List Nat := 22031 :: root407Tail35
def root407Tail33 : List Nat := 21587 :: root407Tail34
def root407Tail32 : List Nat := 21407 :: root407Tail33
def root407Tail31 : List Nat := 20183 :: root407Tail32
def root407Tail30 : List Nat := 19139 :: root407Tail31
def root407Tail29 : List Nat := 18047 :: root407Tail30
def root407Tail28 : List Nat := 18041 :: root407Tail29
def root407Tail27 : List Nat := 17981 :: root407Tail28
def root407Tail26 : List Nat := 17921 :: root407Tail27
def root407Tail25 : List Nat := 16829 :: root407Tail26
def root407Tail24 : List Nat := 16301 :: root407Tail25
def root407Tail23 : List Nat := 16001 :: root407Tail24
def root407Tail22 : List Nat := 14411 :: root407Tail23
def root407Tail21 : List Nat := 14177 :: root407Tail22
def root407Tail20 : List Nat := 13799 :: root407Tail21
def root407Tail19 : List Nat := 13487 :: root407Tail20
def root407Tail18 : List Nat := 12539 :: root407Tail19
def root407Tail17 : List Nat := 11423 :: root407Tail18
def root407Tail16 : List Nat := 9767 :: root407Tail17
def root407Tail15 : List Nat := 9203 :: root407Tail16
def root407Tail14 : List Nat := 8081 :: root407Tail15
def root407Tail13 : List Nat := 7607 :: root407Tail14
def root407Tail12 : List Nat := 7523 :: root407Tail13
def root407Tail11 : List Nat := 7187 :: root407Tail12
def root407Tail10 : List Nat := 6653 :: root407Tail11
def root407Tail9 : List Nat := 5927 :: root407Tail10
def root407Tail8 : List Nat := 5879 :: root407Tail9
def root407Tail7 : List Nat := 5477 :: root407Tail8
def root407Tail6 : List Nat := 5381 :: root407Tail7
def root407Tail5 : List Nat := 5351 :: root407Tail6
def root407Tail4 : List Nat := 5039 :: root407Tail5
def root407Tail3 : List Nat := 4787 :: root407Tail4
def root407Tail2 : List Nat := 4079 :: root407Tail3
def root407Tail1 : List Nat := 3617 :: root407Tail2
def root407Tail0 : List Nat := 2897 :: root407Tail1
def root407Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root407Candidates0_checked : candidates edge 4 2819 2897 root407Tail1 winner.bound = root407Candidates0 := by
  unfold candidates
  have row : edge 2897 = (SortedNatTree.contains FastRows.row2897) := by
    change Data.row2897.contains = (SortedNatTree.contains FastRows.row2897)
    exact FastRows.row2897_checked.symm
  rw [row]
  decide
theorem root407Step0 : scan edge (search edge 3) 4 [2819] 2819 root407Tail0 winner =
    scan edge (search edge 3) 4 [2819] 2819 root407Tail1 winner := by
  rw [root407Tail0, scan, if_neg (by decide), root407Candidates0_checked]
  change scan edge (search edge 3) 4 [2819] 2819 root407Tail1
    (if 3 ≤ root407Candidates0.length then search edge 3 [2819, 2897] root407Candidates0 5716 winner else winner) = _
  rw [if_neg (by decide)]
def root407Candidates1 : List Nat := [7607, 16001]
set_option maxRecDepth 8192 in
theorem root407Candidates1_checked : candidates edge 4 2819 3617 root407Tail2 winner.bound = root407Candidates1 := by
  unfold candidates
  have row : edge 3617 = (SortedNatTree.contains FastRows.row3617) := by
    change Data.row3617.contains = (SortedNatTree.contains FastRows.row3617)
    exact FastRows.row3617_checked.symm
  rw [row]
  decide
theorem root407Step1 : scan edge (search edge 3) 4 [2819] 2819 root407Tail1 winner =
    scan edge (search edge 3) 4 [2819] 2819 root407Tail2 winner := by
  rw [root407Tail1, scan, if_neg (by decide), root407Candidates1_checked]
  change scan edge (search edge 3) 4 [2819] 2819 root407Tail2
    (if 3 ≤ root407Candidates1.length then search edge 3 [2819, 3617] root407Candidates1 6436 winner else winner) = _
  rw [if_neg (by decide)]
def root407Candidates2 : List Nat := [5879, 5927, 18047]
set_option maxRecDepth 8192 in
theorem root407Candidates2_checked : candidates edge 4 2819 4079 root407Tail3 winner.bound = root407Candidates2 := by
  unfold candidates
  have row : edge 4079 = (SortedNatTree.contains FastRows.row4079) := by
    change Data.row4079.contains = (SortedNatTree.contains FastRows.row4079)
    exact FastRows.row4079_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root407Child2 : search edge 3 [2819, 4079] root407Candidates2 6898 winner = winner := by decide
theorem root407Step2 : scan edge (search edge 3) 4 [2819] 2819 root407Tail2 winner =
    scan edge (search edge 3) 4 [2819] 2819 root407Tail3 winner := by
  rw [root407Tail2, scan, if_neg (by decide), root407Candidates2_checked]
  change scan edge (search edge 3) 4 [2819] 2819 root407Tail3
    (if 3 ≤ root407Candidates2.length then search edge 3 [2819, 4079] root407Candidates2 6898 winner else winner) = _
  rw [if_pos (by decide), root407Child2]
def root407Candidates3 : List Nat := [7187]
set_option maxRecDepth 8192 in
theorem root407Candidates3_checked : candidates edge 4 2819 4787 root407Tail4 winner.bound = root407Candidates3 := by
  unfold candidates
  have row : edge 4787 = (SortedNatTree.contains FastRows.row4787) := by
    change Data.row4787.contains = (SortedNatTree.contains FastRows.row4787)
    exact FastRows.row4787_checked.symm
  rw [row]
  decide
theorem root407Step3 : scan edge (search edge 3) 4 [2819] 2819 root407Tail3 winner =
    scan edge (search edge 3) 4 [2819] 2819 root407Tail4 winner := by
  rw [root407Tail3, scan, if_neg (by decide), root407Candidates3_checked]
  change scan edge (search edge 3) 4 [2819] 2819 root407Tail4
    (if 3 ≤ root407Candidates3.length then search edge 3 [2819, 4787] root407Candidates3 7606 winner else winner) = _
  rw [if_neg (by decide)]
def root407Candidates4 : List Nat := [16001]
set_option maxRecDepth 8192 in
theorem root407Candidates4_checked : candidates edge 4 2819 5039 root407Tail5 winner.bound = root407Candidates4 := by
  unfold candidates
  have row : edge 5039 = (SortedNatTree.contains FastRows.row5039) := by
    change Data.row5039.contains = (SortedNatTree.contains FastRows.row5039)
    exact FastRows.row5039_checked.symm
  rw [row]
  decide
theorem root407Step4 : scan edge (search edge 3) 4 [2819] 2819 root407Tail4 winner =
    scan edge (search edge 3) 4 [2819] 2819 root407Tail5 winner := by
  rw [root407Tail4, scan, if_neg (by decide), root407Candidates4_checked]
  change scan edge (search edge 3) 4 [2819] 2819 root407Tail5
    (if 3 ≤ root407Candidates4.length then search edge 3 [2819, 5039] root407Candidates4 7858 winner else winner) = _
  rw [if_neg (by decide)]
def root407Candidates5 : List Nat := [5879, 5927]
set_option maxRecDepth 8192 in
theorem root407Candidates5_checked : candidates edge 4 2819 5351 root407Tail6 winner.bound = root407Candidates5 := by
  unfold candidates
  have row : edge 5351 = (SortedNatTree.contains FastRows.row5351) := by
    change Data.row5351.contains = (SortedNatTree.contains FastRows.row5351)
    exact FastRows.row5351_checked.symm
  rw [row]
  decide
theorem root407Step5 : scan edge (search edge 3) 4 [2819] 2819 root407Tail5 winner =
    scan edge (search edge 3) 4 [2819] 2819 root407Tail6 winner := by
  rw [root407Tail5, scan, if_neg (by decide), root407Candidates5_checked]
  change scan edge (search edge 3) 4 [2819] 2819 root407Tail6
    (if 3 ≤ root407Candidates5.length then search edge 3 [2819, 5351] root407Candidates5 8170 winner else winner) = _
  rw [if_neg (by decide)]
def root407Candidates6 : List Nat := [8081, 14411, 16001]
set_option maxRecDepth 8192 in
theorem root407Candidates6_checked : candidates edge 4 2819 5381 root407Tail7 winner.bound = root407Candidates6 := by
  unfold candidates
  have row : edge 5381 = (SortedNatTree.contains FastRows.row5381) := by
    change Data.row5381.contains = (SortedNatTree.contains FastRows.row5381)
    exact FastRows.row5381_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root407Child6 : search edge 3 [2819, 5381] root407Candidates6 8200 winner = winner := by decide
theorem root407Step6 : scan edge (search edge 3) 4 [2819] 2819 root407Tail6 winner =
    scan edge (search edge 3) 4 [2819] 2819 root407Tail7 winner := by
  rw [root407Tail6, scan, if_neg (by decide), root407Candidates6_checked]
  change scan edge (search edge 3) 4 [2819] 2819 root407Tail7
    (if 3 ≤ root407Candidates6.length then search edge 3 [2819, 5381] root407Candidates6 8200 winner else winner) = _
  rw [if_pos (by decide), root407Child6]
def root407Candidates7 : List Nat := [5927, 8081]
set_option maxRecDepth 8192 in
theorem root407Candidates7_checked : candidates edge 4 2819 5477 root407Tail8 winner.bound = root407Candidates7 := by
  unfold candidates
  have row : edge 5477 = (SortedNatTree.contains FastRows.row5477) := by
    change Data.row5477.contains = (SortedNatTree.contains FastRows.row5477)
    exact FastRows.row5477_checked.symm
  rw [row]
  decide
theorem root407Step7 : scan edge (search edge 3) 4 [2819] 2819 root407Tail7 winner =
    scan edge (search edge 3) 4 [2819] 2819 root407Tail8 winner := by
  rw [root407Tail7, scan, if_neg (by decide), root407Candidates7_checked]
  change scan edge (search edge 3) 4 [2819] 2819 root407Tail8
    (if 3 ≤ root407Candidates7.length then search edge 3 [2819, 5477] root407Candidates7 8296 winner else winner) = _
  rw [if_neg (by decide)]
theorem root407Scan8 : scan edge (search edge 3) 4 [2819] 2819 root407Tail8 winner = winner := by rw [root407Tail8, scan, if_pos (by decide)]
theorem root407Scan7 : scan edge (search edge 3) 4 [2819] 2819 root407Tail7 winner = winner :=
  (root407Step7).trans root407Scan8
theorem root407Scan6 : scan edge (search edge 3) 4 [2819] 2819 root407Tail6 winner = winner :=
  (root407Step6).trans root407Scan7
theorem root407Scan5 : scan edge (search edge 3) 4 [2819] 2819 root407Tail5 winner = winner :=
  (root407Step5).trans root407Scan6
theorem root407Scan4 : scan edge (search edge 3) 4 [2819] 2819 root407Tail4 winner = winner :=
  (root407Step4).trans root407Scan5
theorem root407Scan3 : scan edge (search edge 3) 4 [2819] 2819 root407Tail3 winner = winner :=
  (root407Step3).trans root407Scan4
theorem root407Scan2 : scan edge (search edge 3) 4 [2819] 2819 root407Tail2 winner = winner :=
  (root407Step2).trans root407Scan3
theorem root407Scan1 : scan edge (search edge 3) 4 [2819] 2819 root407Tail1 winner = winner :=
  (root407Step1).trans root407Scan2
theorem root407Scan0 : scan edge (search edge 3) 4 [2819] 2819 root407Tail0 winner = winner :=
  (root407Step0).trans root407Scan1
set_option maxRecDepth 4096 in
theorem root407Branch : search edge 4 [2819] root407Candidates 2819 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2819] 2819 root407Tail0 winner = winner
  exact root407Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
