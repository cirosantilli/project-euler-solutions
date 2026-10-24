import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root498Candidates : List Nat := Data.row3581
set_option maxRecDepth 8192 in
theorem root498Candidates_checked : candidates edge 5 0 3581 tail499 winner.bound = root498Candidates := by
  apply candidates_eq_row edge 5 0 3581 tail499 root498Candidates winner.bound tail499_ordered
  · decide
  · decide
  · funext q
    rfl
def root498Tail43 : List Nat := []
def root498Tail42 : List Nat := 21143 :: root498Tail43
def root498Tail41 : List Nat := 21011 :: root498Tail42
def root498Tail40 : List Nat := 20849 :: root498Tail41
def root498Tail39 : List Nat := 20753 :: root498Tail40
def root498Tail38 : List Nat := 19997 :: root498Tail39
def root498Tail37 : List Nat := 19763 :: root498Tail38
def root498Tail36 : List Nat := 19661 :: root498Tail37
def root498Tail35 : List Nat := 19469 :: root498Tail36
def root498Tail34 : List Nat := 19031 :: root498Tail35
def root498Tail33 : List Nat := 18443 :: root498Tail34
def root498Tail32 : List Nat := 17351 :: root498Tail33
def root498Tail31 : List Nat := 16763 :: root498Tail32
def root498Tail30 : List Nat := 16487 :: root498Tail31
def root498Tail29 : List Nat := 16223 :: root498Tail30
def root498Tail28 : List Nat := 16217 :: root498Tail29
def root498Tail27 : List Nat := 14411 :: root498Tail28
def root498Tail26 : List Nat := 13487 :: root498Tail27
def root498Tail25 : List Nat := 12917 :: root498Tail26
def root498Tail24 : List Nat := 12791 :: root498Tail25
def root498Tail23 : List Nat := 12401 :: root498Tail24
def root498Tail22 : List Nat := 12239 :: root498Tail23
def root498Tail21 : List Nat := 11657 :: root498Tail22
def root498Tail20 : List Nat := 9941 :: root498Tail21
def root498Tail19 : List Nat := 9803 :: root498Tail20
def root498Tail18 : List Nat := 8807 :: root498Tail19
def root498Tail17 : List Nat := 7829 :: root498Tail18
def root498Tail16 : List Nat := 7253 :: root498Tail17
def root498Tail15 : List Nat := 6983 :: root498Tail16
def root498Tail14 : List Nat := 6701 :: root498Tail15
def root498Tail13 : List Nat := 6689 :: root498Tail14
def root498Tail12 : List Nat := 6101 :: root498Tail13
def root498Tail11 : List Nat := 5903 :: root498Tail12
def root498Tail10 : List Nat := 5849 :: root498Tail11
def root498Tail9 : List Nat := 5387 :: root498Tail10
def root498Tail8 : List Nat := 5303 :: root498Tail9
def root498Tail7 : List Nat := 4931 :: root498Tail8
def root498Tail6 : List Nat := 4679 :: root498Tail7
def root498Tail5 : List Nat := 4643 :: root498Tail6
def root498Tail4 : List Nat := 4421 :: root498Tail5
def root498Tail3 : List Nat := 4139 :: root498Tail4
def root498Tail2 : List Nat := 4133 :: root498Tail3
def root498Tail1 : List Nat := 4091 :: root498Tail2
def root498Tail0 : List Nat := 3851 :: root498Tail1
def root498Candidates0 : List Nat := [4139, 5903, 6983]
set_option maxRecDepth 8192 in
theorem root498Candidates0_checked : candidates edge 4 3581 3851 root498Tail1 winner.bound = root498Candidates0 := by
  unfold candidates
  have row : edge 3851 = (SortedNatTree.contains FastRows.row3851) := by
    change Data.row3851.contains = (SortedNatTree.contains FastRows.row3851)
    exact FastRows.row3851_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root498Child0 : search edge 3 [3581, 3851] root498Candidates0 7432 winner = winner := by decide
theorem root498Step0 : scan edge (search edge 3) 4 [3581] 3581 root498Tail0 winner =
    scan edge (search edge 3) 4 [3581] 3581 root498Tail1 winner := by
  rw [root498Tail0, scan, if_neg (by decide), root498Candidates0_checked]
  change scan edge (search edge 3) 4 [3581] 3581 root498Tail1
    (if 3 ≤ root498Candidates0.length then search edge 3 [3581, 3851] root498Candidates0 7432 winner else winner) = _
  rw [if_pos (by decide), root498Child0]
def root498Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root498Candidates1_checked : candidates edge 4 3581 4091 root498Tail2 winner.bound = root498Candidates1 := by
  unfold candidates
  have row : edge 4091 = (SortedNatTree.contains FastRows.row4091) := by
    change Data.row4091.contains = (SortedNatTree.contains FastRows.row4091)
    exact FastRows.row4091_checked.symm
  rw [row]
  decide
theorem root498Step1 : scan edge (search edge 3) 4 [3581] 3581 root498Tail1 winner =
    scan edge (search edge 3) 4 [3581] 3581 root498Tail2 winner := by
  rw [root498Tail1, scan, if_neg (by decide), root498Candidates1_checked]
  change scan edge (search edge 3) 4 [3581] 3581 root498Tail2
    (if 3 ≤ root498Candidates1.length then search edge 3 [3581, 4091] root498Candidates1 7672 winner else winner) = _
  rw [if_neg (by decide)]
def root498Candidates2 : List Nat := [5387]
set_option maxRecDepth 8192 in
theorem root498Candidates2_checked : candidates edge 4 3581 4133 root498Tail3 winner.bound = root498Candidates2 := by
  unfold candidates
  have row : edge 4133 = (SortedNatTree.contains FastRows.row4133) := by
    change Data.row4133.contains = (SortedNatTree.contains FastRows.row4133)
    exact FastRows.row4133_checked.symm
  rw [row]
  decide
theorem root498Step2 : scan edge (search edge 3) 4 [3581] 3581 root498Tail2 winner =
    scan edge (search edge 3) 4 [3581] 3581 root498Tail3 winner := by
  rw [root498Tail2, scan, if_neg (by decide), root498Candidates2_checked]
  change scan edge (search edge 3) 4 [3581] 3581 root498Tail3
    (if 3 ≤ root498Candidates2.length then search edge 3 [3581, 4133] root498Candidates2 7714 winner else winner) = _
  rw [if_neg (by decide)]
def root498Candidates3 : List Nat := [7253]
set_option maxRecDepth 8192 in
theorem root498Candidates3_checked : candidates edge 4 3581 4139 root498Tail4 winner.bound = root498Candidates3 := by
  unfold candidates
  have row : edge 4139 = (SortedNatTree.contains FastRows.row4139) := by
    change Data.row4139.contains = (SortedNatTree.contains FastRows.row4139)
    exact FastRows.row4139_checked.symm
  rw [row]
  decide
theorem root498Step3 : scan edge (search edge 3) 4 [3581] 3581 root498Tail3 winner =
    scan edge (search edge 3) 4 [3581] 3581 root498Tail4 winner := by
  rw [root498Tail3, scan, if_neg (by decide), root498Candidates3_checked]
  change scan edge (search edge 3) 4 [3581] 3581 root498Tail4
    (if 3 ≤ root498Candidates3.length then search edge 3 [3581, 4139] root498Candidates3 7720 winner else winner) = _
  rw [if_neg (by decide)]
def root498Candidates4 : List Nat := [9803, 12401]
set_option maxRecDepth 8192 in
theorem root498Candidates4_checked : candidates edge 4 3581 4421 root498Tail5 winner.bound = root498Candidates4 := by
  unfold candidates
  have row : edge 4421 = (SortedNatTree.contains FastRows.row4421) := by
    change Data.row4421.contains = (SortedNatTree.contains FastRows.row4421)
    exact FastRows.row4421_checked.symm
  rw [row]
  decide
theorem root498Step4 : scan edge (search edge 3) 4 [3581] 3581 root498Tail4 winner =
    scan edge (search edge 3) 4 [3581] 3581 root498Tail5 winner := by
  rw [root498Tail4, scan, if_neg (by decide), root498Candidates4_checked]
  change scan edge (search edge 3) 4 [3581] 3581 root498Tail5
    (if 3 ≤ root498Candidates4.length then search edge 3 [3581, 4421] root498Candidates4 8002 winner else winner) = _
  rw [if_neg (by decide)]
def root498Candidates5 : List Nat := [5849]
set_option maxRecDepth 8192 in
theorem root498Candidates5_checked : candidates edge 4 3581 4643 root498Tail6 winner.bound = root498Candidates5 := by
  unfold candidates
  have row : edge 4643 = (SortedNatTree.contains FastRows.row4643) := by
    change Data.row4643.contains = (SortedNatTree.contains FastRows.row4643)
    exact FastRows.row4643_checked.symm
  rw [row]
  decide
theorem root498Step5 : scan edge (search edge 3) 4 [3581] 3581 root498Tail5 winner =
    scan edge (search edge 3) 4 [3581] 3581 root498Tail6 winner := by
  rw [root498Tail5, scan, if_neg (by decide), root498Candidates5_checked]
  change scan edge (search edge 3) 4 [3581] 3581 root498Tail6
    (if 3 ≤ root498Candidates5.length then search edge 3 [3581, 4643] root498Candidates5 8224 winner else winner) = _
  rw [if_neg (by decide)]
def root498Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root498Candidates6_checked : candidates edge 4 3581 4679 root498Tail7 winner.bound = root498Candidates6 := by
  unfold candidates
  have row : edge 4679 = (SortedNatTree.contains FastRows.row4679) := by
    change Data.row4679.contains = (SortedNatTree.contains FastRows.row4679)
    exact FastRows.row4679_checked.symm
  rw [row]
  decide
theorem root498Step6 : scan edge (search edge 3) 4 [3581] 3581 root498Tail6 winner =
    scan edge (search edge 3) 4 [3581] 3581 root498Tail7 winner := by
  rw [root498Tail6, scan, if_neg (by decide), root498Candidates6_checked]
  change scan edge (search edge 3) 4 [3581] 3581 root498Tail7
    (if 3 ≤ root498Candidates6.length then search edge 3 [3581, 4679] root498Candidates6 8260 winner else winner) = _
  rw [if_neg (by decide)]
def root498Candidates7 : List Nat := [13487, 16223]
set_option maxRecDepth 8192 in
theorem root498Candidates7_checked : candidates edge 4 3581 4931 root498Tail8 winner.bound = root498Candidates7 := by
  unfold candidates
  have row : edge 4931 = (SortedNatTree.contains FastRows.row4931) := by
    change Data.row4931.contains = (SortedNatTree.contains FastRows.row4931)
    exact FastRows.row4931_checked.symm
  rw [row]
  decide
theorem root498Step7 : scan edge (search edge 3) 4 [3581] 3581 root498Tail7 winner =
    scan edge (search edge 3) 4 [3581] 3581 root498Tail8 winner := by
  rw [root498Tail7, scan, if_neg (by decide), root498Candidates7_checked]
  change scan edge (search edge 3) 4 [3581] 3581 root498Tail8
    (if 3 ≤ root498Candidates7.length then search edge 3 [3581, 4931] root498Candidates7 8512 winner else winner) = _
  rw [if_neg (by decide)]
def root498Candidates8 : List Nat := [5849, 6101, 12917, 16487]
set_option maxRecDepth 8192 in
theorem root498Candidates8_checked : candidates edge 4 3581 5303 root498Tail9 winner.bound = root498Candidates8 := by
  unfold candidates
  have row : edge 5303 = (SortedNatTree.contains FastRows.row5303) := by
    change Data.row5303.contains = (SortedNatTree.contains FastRows.row5303)
    exact FastRows.row5303_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root498Child8 : search edge 3 [3581, 5303] root498Candidates8 8884 winner = winner := by decide
theorem root498Step8 : scan edge (search edge 3) 4 [3581] 3581 root498Tail8 winner =
    scan edge (search edge 3) 4 [3581] 3581 root498Tail9 winner := by
  rw [root498Tail8, scan, if_neg (by decide), root498Candidates8_checked]
  change scan edge (search edge 3) 4 [3581] 3581 root498Tail9
    (if 3 ≤ root498Candidates8.length then search edge 3 [3581, 5303] root498Candidates8 8884 winner else winner) = _
  rw [if_pos (by decide), root498Child8]
def root498Candidates9 : List Nat := [16223]
set_option maxRecDepth 8192 in
theorem root498Candidates9_checked : candidates edge 4 3581 5387 root498Tail10 winner.bound = root498Candidates9 := by
  unfold candidates
  have row : edge 5387 = (SortedNatTree.contains FastRows.row5387) := by
    change Data.row5387.contains = (SortedNatTree.contains FastRows.row5387)
    exact FastRows.row5387_checked.symm
  rw [row]
  decide
theorem root498Step9 : scan edge (search edge 3) 4 [3581] 3581 root498Tail9 winner =
    scan edge (search edge 3) 4 [3581] 3581 root498Tail10 winner := by
  rw [root498Tail9, scan, if_neg (by decide), root498Candidates9_checked]
  change scan edge (search edge 3) 4 [3581] 3581 root498Tail10
    (if 3 ≤ root498Candidates9.length then search edge 3 [3581, 5387] root498Candidates9 8968 winner else winner) = _
  rw [if_neg (by decide)]
theorem root498Scan10 : scan edge (search edge 3) 4 [3581] 3581 root498Tail10 winner = winner := by rw [root498Tail10, scan, if_pos (by decide)]
theorem root498Scan9 : scan edge (search edge 3) 4 [3581] 3581 root498Tail9 winner = winner :=
  (root498Step9).trans root498Scan10
theorem root498Scan8 : scan edge (search edge 3) 4 [3581] 3581 root498Tail8 winner = winner :=
  (root498Step8).trans root498Scan9
theorem root498Scan7 : scan edge (search edge 3) 4 [3581] 3581 root498Tail7 winner = winner :=
  (root498Step7).trans root498Scan8
theorem root498Scan6 : scan edge (search edge 3) 4 [3581] 3581 root498Tail6 winner = winner :=
  (root498Step6).trans root498Scan7
theorem root498Scan5 : scan edge (search edge 3) 4 [3581] 3581 root498Tail5 winner = winner :=
  (root498Step5).trans root498Scan6
theorem root498Scan4 : scan edge (search edge 3) 4 [3581] 3581 root498Tail4 winner = winner :=
  (root498Step4).trans root498Scan5
theorem root498Scan3 : scan edge (search edge 3) 4 [3581] 3581 root498Tail3 winner = winner :=
  (root498Step3).trans root498Scan4
theorem root498Scan2 : scan edge (search edge 3) 4 [3581] 3581 root498Tail2 winner = winner :=
  (root498Step2).trans root498Scan3
theorem root498Scan1 : scan edge (search edge 3) 4 [3581] 3581 root498Tail1 winner = winner :=
  (root498Step1).trans root498Scan2
theorem root498Scan0 : scan edge (search edge 3) 4 [3581] 3581 root498Tail0 winner = winner :=
  (root498Step0).trans root498Scan1
set_option maxRecDepth 4096 in
theorem root498Branch : search edge 4 [3581] root498Candidates 3581 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3581] 3581 root498Tail0 winner = winner
  exact root498Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
