import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root444Candidates : List Nat := Data.row3163
set_option maxRecDepth 8192 in
theorem root444Candidates_checked : candidates edge 5 0 3163 tail445 winner.bound = root444Candidates := by
  apply candidates_eq_row edge 5 0 3163 tail445 root444Candidates winner.bound tail445_ordered
  · decide
  · decide
  · funext q
    rfl
def root444Tail55 : List Nat := []
def root444Tail54 : List Nat := 22861 :: root444Tail55
def root444Tail53 : List Nat := 22483 :: root444Tail54
def root444Tail52 : List Nat := 21871 :: root444Tail53
def root444Tail51 : List Nat := 21787 :: root444Tail52
def root444Tail50 : List Nat := 21727 :: root444Tail51
def root444Tail49 : List Nat := 21391 :: root444Tail50
def root444Tail48 : List Nat := 21157 :: root444Tail49
def root444Tail47 : List Nat := 20731 :: root444Tail48
def root444Tail46 : List Nat := 20611 :: root444Tail47
def root444Tail45 : List Nat := 20287 :: root444Tail46
def root444Tail44 : List Nat := 19891 :: root444Tail45
def root444Tail43 : List Nat := 18553 :: root444Tail44
def root444Tail42 : List Nat := 18523 :: root444Tail43
def root444Tail41 : List Nat := 18481 :: root444Tail42
def root444Tail40 : List Nat := 18049 :: root444Tail41
def root444Tail39 : List Nat := 17977 :: root444Tail40
def root444Tail38 : List Nat := 17923 :: root444Tail39
def root444Tail37 : List Nat := 17401 :: root444Tail38
def root444Tail36 : List Nat := 17389 :: root444Tail37
def root444Tail35 : List Nat := 16993 :: root444Tail36
def root444Tail34 : List Nat := 15679 :: root444Tail35
def root444Tail33 : List Nat := 15541 :: root444Tail34
def root444Tail32 : List Nat := 14851 :: root444Tail33
def root444Tail31 : List Nat := 14431 :: root444Tail32
def root444Tail30 : List Nat := 14143 :: root444Tail31
def root444Tail29 : List Nat := 13873 :: root444Tail30
def root444Tail28 : List Nat := 13597 :: root444Tail29
def root444Tail27 : List Nat := 13537 :: root444Tail28
def root444Tail26 : List Nat := 12967 :: root444Tail27
def root444Tail25 : List Nat := 12277 :: root444Tail26
def root444Tail24 : List Nat := 11329 :: root444Tail25
def root444Tail23 : List Nat := 10861 :: root444Tail24
def root444Tail22 : List Nat := 10711 :: root444Tail23
def root444Tail21 : List Nat := 10333 :: root444Tail22
def root444Tail20 : List Nat := 10009 :: root444Tail21
def root444Tail19 : List Nat := 9859 :: root444Tail20
def root444Tail18 : List Nat := 9721 :: root444Tail19
def root444Tail17 : List Nat := 9187 :: root444Tail18
def root444Tail16 : List Nat := 9043 :: root444Tail17
def root444Tail15 : List Nat := 8821 :: root444Tail16
def root444Tail14 : List Nat := 8707 :: root444Tail15
def root444Tail13 : List Nat := 8527 :: root444Tail14
def root444Tail12 : List Nat := 7951 :: root444Tail13
def root444Tail11 : List Nat := 7477 :: root444Tail12
def root444Tail10 : List Nat := 6733 :: root444Tail11
def root444Tail9 : List Nat := 6301 :: root444Tail10
def root444Tail8 : List Nat := 6037 :: root444Tail9
def root444Tail7 : List Nat := 5743 :: root444Tail8
def root444Tail6 : List Nat := 5431 :: root444Tail7
def root444Tail5 : List Nat := 5101 :: root444Tail6
def root444Tail4 : List Nat := 4129 :: root444Tail5
def root444Tail3 : List Nat := 3967 :: root444Tail4
def root444Tail2 : List Nat := 3769 :: root444Tail3
def root444Tail1 : List Nat := 3571 :: root444Tail2
def root444Tail0 : List Nat := 3529 :: root444Tail1
def root444Candidates0 : List Nat := [6733, 8707, 14143]
set_option maxRecDepth 8192 in
theorem root444Candidates0_checked : candidates edge 4 3163 3529 root444Tail1 winner.bound = root444Candidates0 := by
  unfold candidates
  have row : edge 3529 = (SortedNatTree.contains FastRows.row3529) := by
    change Data.row3529.contains = (SortedNatTree.contains FastRows.row3529)
    exact FastRows.row3529_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root444Child0 : search edge 3 [3163, 3529] root444Candidates0 6692 winner = winner := by decide
theorem root444Step0 : scan edge (search edge 3) 4 [3163] 3163 root444Tail0 winner =
    scan edge (search edge 3) 4 [3163] 3163 root444Tail1 winner := by
  rw [root444Tail0, scan, if_neg (by decide), root444Candidates0_checked]
  change scan edge (search edge 3) 4 [3163] 3163 root444Tail1
    (if 3 ≤ root444Candidates0.length then search edge 3 [3163, 3529] root444Candidates0 6692 winner else winner) = _
  rw [if_pos (by decide), root444Child0]
def root444Candidates1 : List Nat := [11329, 14143]
set_option maxRecDepth 8192 in
theorem root444Candidates1_checked : candidates edge 4 3163 3571 root444Tail2 winner.bound = root444Candidates1 := by
  unfold candidates
  have row : edge 3571 = (SortedNatTree.contains FastRows.row3571) := by
    change Data.row3571.contains = (SortedNatTree.contains FastRows.row3571)
    exact FastRows.row3571_checked.symm
  rw [row]
  decide
theorem root444Step1 : scan edge (search edge 3) 4 [3163] 3163 root444Tail1 winner =
    scan edge (search edge 3) 4 [3163] 3163 root444Tail2 winner := by
  rw [root444Tail1, scan, if_neg (by decide), root444Candidates1_checked]
  change scan edge (search edge 3) 4 [3163] 3163 root444Tail2
    (if 3 ≤ root444Candidates1.length then search edge 3 [3163, 3571] root444Candidates1 6734 winner else winner) = _
  rw [if_neg (by decide)]
def root444Candidates2 : List Nat := [16993]
set_option maxRecDepth 8192 in
theorem root444Candidates2_checked : candidates edge 4 3163 3769 root444Tail3 winner.bound = root444Candidates2 := by
  unfold candidates
  have row : edge 3769 = (SortedNatTree.contains FastRows.row3769) := by
    change Data.row3769.contains = (SortedNatTree.contains FastRows.row3769)
    exact FastRows.row3769_checked.symm
  rw [row]
  decide
theorem root444Step2 : scan edge (search edge 3) 4 [3163] 3163 root444Tail2 winner =
    scan edge (search edge 3) 4 [3163] 3163 root444Tail3 winner := by
  rw [root444Tail2, scan, if_neg (by decide), root444Candidates2_checked]
  change scan edge (search edge 3) 4 [3163] 3163 root444Tail3
    (if 3 ≤ root444Candidates2.length then search edge 3 [3163, 3769] root444Candidates2 6932 winner else winner) = _
  rw [if_neg (by decide)]
def root444Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root444Candidates3_checked : candidates edge 4 3163 3967 root444Tail4 winner.bound = root444Candidates3 := by
  unfold candidates
  have row : edge 3967 = (SortedNatTree.contains FastRows.row3967) := by
    change Data.row3967.contains = (SortedNatTree.contains FastRows.row3967)
    exact FastRows.row3967_checked.symm
  rw [row]
  decide
theorem root444Step3 : scan edge (search edge 3) 4 [3163] 3163 root444Tail3 winner =
    scan edge (search edge 3) 4 [3163] 3163 root444Tail4 winner := by
  rw [root444Tail3, scan, if_neg (by decide), root444Candidates3_checked]
  change scan edge (search edge 3) 4 [3163] 3163 root444Tail4
    (if 3 ≤ root444Candidates3.length then search edge 3 [3163, 3967] root444Candidates3 7130 winner else winner) = _
  rw [if_neg (by decide)]
def root444Candidates4 : List Nat := [14431, 17389]
set_option maxRecDepth 8192 in
theorem root444Candidates4_checked : candidates edge 4 3163 4129 root444Tail5 winner.bound = root444Candidates4 := by
  unfold candidates
  have row : edge 4129 = (SortedNatTree.contains FastRows.row4129) := by
    change Data.row4129.contains = (SortedNatTree.contains FastRows.row4129)
    exact FastRows.row4129_checked.symm
  rw [row]
  decide
theorem root444Step4 : scan edge (search edge 3) 4 [3163] 3163 root444Tail4 winner =
    scan edge (search edge 3) 4 [3163] 3163 root444Tail5 winner := by
  rw [root444Tail4, scan, if_neg (by decide), root444Candidates4_checked]
  change scan edge (search edge 3) 4 [3163] 3163 root444Tail5
    (if 3 ≤ root444Candidates4.length then search edge 3 [3163, 4129] root444Candidates4 7292 winner else winner) = _
  rw [if_neg (by decide)]
def root444Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root444Candidates5_checked : candidates edge 4 3163 5101 root444Tail6 winner.bound = root444Candidates5 := by
  unfold candidates
  have row : edge 5101 = (SortedNatTree.contains FastRows.row5101) := by
    change Data.row5101.contains = (SortedNatTree.contains FastRows.row5101)
    exact FastRows.row5101_checked.symm
  rw [row]
  decide
theorem root444Step5 : scan edge (search edge 3) 4 [3163] 3163 root444Tail5 winner =
    scan edge (search edge 3) 4 [3163] 3163 root444Tail6 winner := by
  rw [root444Tail5, scan, if_neg (by decide), root444Candidates5_checked]
  change scan edge (search edge 3) 4 [3163] 3163 root444Tail6
    (if 3 ≤ root444Candidates5.length then search edge 3 [3163, 5101] root444Candidates5 8264 winner else winner) = _
  rw [if_neg (by decide)]
def root444Candidates6 : List Nat := [5743]
set_option maxRecDepth 8192 in
theorem root444Candidates6_checked : candidates edge 4 3163 5431 root444Tail7 winner.bound = root444Candidates6 := by
  unfold candidates
  have row : edge 5431 = (SortedNatTree.contains FastRows.row5431) := by
    change Data.row5431.contains = (SortedNatTree.contains FastRows.row5431)
    exact FastRows.row5431_checked.symm
  rw [row]
  decide
theorem root444Step6 : scan edge (search edge 3) 4 [3163] 3163 root444Tail6 winner =
    scan edge (search edge 3) 4 [3163] 3163 root444Tail7 winner := by
  rw [root444Tail6, scan, if_neg (by decide), root444Candidates6_checked]
  change scan edge (search edge 3) 4 [3163] 3163 root444Tail7
    (if 3 ≤ root444Candidates6.length then search edge 3 [3163, 5431] root444Candidates6 8594 winner else winner) = _
  rw [if_neg (by decide)]
theorem root444Scan7 : scan edge (search edge 3) 4 [3163] 3163 root444Tail7 winner = winner := by rw [root444Tail7, scan, if_pos (by decide)]
theorem root444Scan6 : scan edge (search edge 3) 4 [3163] 3163 root444Tail6 winner = winner :=
  (root444Step6).trans root444Scan7
theorem root444Scan5 : scan edge (search edge 3) 4 [3163] 3163 root444Tail5 winner = winner :=
  (root444Step5).trans root444Scan6
theorem root444Scan4 : scan edge (search edge 3) 4 [3163] 3163 root444Tail4 winner = winner :=
  (root444Step4).trans root444Scan5
theorem root444Scan3 : scan edge (search edge 3) 4 [3163] 3163 root444Tail3 winner = winner :=
  (root444Step3).trans root444Scan4
theorem root444Scan2 : scan edge (search edge 3) 4 [3163] 3163 root444Tail2 winner = winner :=
  (root444Step2).trans root444Scan3
theorem root444Scan1 : scan edge (search edge 3) 4 [3163] 3163 root444Tail1 winner = winner :=
  (root444Step1).trans root444Scan2
theorem root444Scan0 : scan edge (search edge 3) 4 [3163] 3163 root444Tail0 winner = winner :=
  (root444Step0).trans root444Scan1
set_option maxRecDepth 4096 in
theorem root444Branch : search edge 4 [3163] root444Candidates 3163 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3163] 3163 root444Tail0 winner = winner
  exact root444Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
