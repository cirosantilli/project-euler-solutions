import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root215Candidates : List Nat := Data.row1361
set_option maxRecDepth 8192 in
theorem root215Candidates_checked : candidates edge 5 0 1361 tail216 winner.bound = root215Candidates := by
  apply candidates_eq_row edge 5 0 1361 tail216 root215Candidates winner.bound tail216_ordered
  · decide
  · decide
  · funext q
    rfl
def root215Tail54 : List Nat := []
def root215Tail53 : List Nat := 23909 :: root215Tail54
def root215Tail52 : List Nat := 23831 :: root215Tail53
def root215Tail51 : List Nat := 23747 :: root215Tail52
def root215Tail50 : List Nat := 23417 :: root215Tail51
def root215Tail49 : List Nat := 23357 :: root215Tail50
def root215Tail48 : List Nat := 23291 :: root215Tail49
def root215Tail47 : List Nat := 22229 :: root215Tail48
def root215Tail46 : List Nat := 22193 :: root215Tail47
def root215Tail45 : List Nat := 21143 :: root215Tail46
def root215Tail44 : List Nat := 21017 :: root215Tail45
def root215Tail43 : List Nat := 20681 :: root215Tail44
def root215Tail42 : List Nat := 19259 :: root215Tail43
def root215Tail41 : List Nat := 18911 :: root215Tail42
def root215Tail40 : List Nat := 17987 :: root215Tail41
def root215Tail39 : List Nat := 16067 :: root215Tail40
def root215Tail38 : List Nat := 14969 :: root215Tail39
def root215Tail37 : List Nat := 14747 :: root215Tail38
def root215Tail36 : List Nat := 14591 :: root215Tail37
def root215Tail35 : List Nat := 14549 :: root215Tail36
def root215Tail34 : List Nat := 14543 :: root215Tail35
def root215Tail33 : List Nat := 14327 :: root215Tail34
def root215Tail32 : List Nat := 14249 :: root215Tail33
def root215Tail31 : List Nat := 14243 :: root215Tail32
def root215Tail30 : List Nat := 13799 :: root215Tail31
def root215Tail29 : List Nat := 13613 :: root215Tail30
def root215Tail28 : List Nat := 13331 :: root215Tail29
def root215Tail27 : List Nat := 13001 :: root215Tail28
def root215Tail26 : List Nat := 12491 :: root215Tail27
def root215Tail25 : List Nat := 12479 :: root215Tail26
def root215Tail24 : List Nat := 11981 :: root215Tail25
def root215Tail23 : List Nat := 11939 :: root215Tail24
def root215Tail22 : List Nat := 11321 :: root215Tail23
def root215Tail21 : List Nat := 10973 :: root215Tail22
def root215Tail20 : List Nat := 10223 :: root215Tail21
def root215Tail19 : List Nat := 9467 :: root215Tail20
def root215Tail18 : List Nat := 9341 :: root215Tail19
def root215Tail17 : List Nat := 9161 :: root215Tail18
def root215Tail16 : List Nat := 8699 :: root215Tail17
def root215Tail15 : List Nat := 7607 :: root215Tail16
def root215Tail14 : List Nat := 7103 :: root215Tail15
def root215Tail13 : List Nat := 6947 :: root215Tail14
def root215Tail12 : List Nat := 6899 :: root215Tail13
def root215Tail11 : List Nat := 6737 :: root215Tail12
def root215Tail10 : List Nat := 6599 :: root215Tail11
def root215Tail9 : List Nat := 6047 :: root215Tail10
def root215Tail8 : List Nat := 5669 :: root215Tail9
def root215Tail7 : List Nat := 5099 :: root215Tail8
def root215Tail6 : List Nat := 4793 :: root215Tail7
def root215Tail5 : List Nat := 3947 :: root215Tail6
def root215Tail4 : List Nat := 3917 :: root215Tail5
def root215Tail3 : List Nat := 3701 :: root215Tail4
def root215Tail2 : List Nat := 3581 :: root215Tail3
def root215Tail1 : List Nat := 3461 :: root215Tail2
def root215Tail0 : List Nat := 2141 :: root215Tail1
def root215Candidates0 : List Nat := [3947, 6737, 14969]
set_option maxRecDepth 8192 in
theorem root215Candidates0_checked : candidates edge 4 1361 2141 root215Tail1 winner.bound = root215Candidates0 := by
  unfold candidates
  have row : edge 2141 = (SortedNatTree.contains FastRows.row2141) := by
    change Data.row2141.contains = (SortedNatTree.contains FastRows.row2141)
    exact FastRows.row2141_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root215Child0 : search edge 3 [1361, 2141] root215Candidates0 3502 winner = winner := by decide
theorem root215Step0 : scan edge (search edge 3) 4 [1361] 1361 root215Tail0 winner =
    scan edge (search edge 3) 4 [1361] 1361 root215Tail1 winner := by
  rw [root215Tail0, scan, if_neg (by decide), root215Candidates0_checked]
  change scan edge (search edge 3) 4 [1361] 1361 root215Tail1
    (if 3 ≤ root215Candidates0.length then search edge 3 [1361, 2141] root215Candidates0 3502 winner else winner) = _
  rw [if_pos (by decide), root215Child0]
def root215Candidates1 : List Nat := [6737]
set_option maxRecDepth 8192 in
theorem root215Candidates1_checked : candidates edge 4 1361 3461 root215Tail2 winner.bound = root215Candidates1 := by
  unfold candidates
  have row : edge 3461 = (SortedNatTree.contains FastRows.row3461) := by
    change Data.row3461.contains = (SortedNatTree.contains FastRows.row3461)
    exact FastRows.row3461_checked.symm
  rw [row]
  decide
theorem root215Step1 : scan edge (search edge 3) 4 [1361] 1361 root215Tail1 winner =
    scan edge (search edge 3) 4 [1361] 1361 root215Tail2 winner := by
  rw [root215Tail1, scan, if_neg (by decide), root215Candidates1_checked]
  change scan edge (search edge 3) 4 [1361] 1361 root215Tail2
    (if 3 ≤ root215Candidates1.length then search edge 3 [1361, 3461] root215Candidates1 4822 winner else winner) = _
  rw [if_neg (by decide)]
def root215Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root215Candidates2_checked : candidates edge 4 1361 3581 root215Tail3 winner.bound = root215Candidates2 := by
  unfold candidates
  have row : edge 3581 = (SortedNatTree.contains FastRows.row3581) := by
    change Data.row3581.contains = (SortedNatTree.contains FastRows.row3581)
    exact FastRows.row3581_checked.symm
  rw [row]
  decide
theorem root215Step2 : scan edge (search edge 3) 4 [1361] 1361 root215Tail2 winner =
    scan edge (search edge 3) 4 [1361] 1361 root215Tail3 winner := by
  rw [root215Tail2, scan, if_neg (by decide), root215Candidates2_checked]
  change scan edge (search edge 3) 4 [1361] 1361 root215Tail3
    (if 3 ≤ root215Candidates2.length then search edge 3 [1361, 3581] root215Candidates2 4942 winner else winner) = _
  rw [if_neg (by decide)]
def root215Candidates3 : List Nat := [9161]
set_option maxRecDepth 8192 in
theorem root215Candidates3_checked : candidates edge 4 1361 3701 root215Tail4 winner.bound = root215Candidates3 := by
  unfold candidates
  have row : edge 3701 = (SortedNatTree.contains FastRows.row3701) := by
    change Data.row3701.contains = (SortedNatTree.contains FastRows.row3701)
    exact FastRows.row3701_checked.symm
  rw [row]
  decide
theorem root215Step3 : scan edge (search edge 3) 4 [1361] 1361 root215Tail3 winner =
    scan edge (search edge 3) 4 [1361] 1361 root215Tail4 winner := by
  rw [root215Tail3, scan, if_neg (by decide), root215Candidates3_checked]
  change scan edge (search edge 3) 4 [1361] 1361 root215Tail4
    (if 3 ≤ root215Candidates3.length then search edge 3 [1361, 3701] root215Candidates3 5062 winner else winner) = _
  rw [if_neg (by decide)]
def root215Candidates4 : List Nat := [6899, 14249]
set_option maxRecDepth 8192 in
theorem root215Candidates4_checked : candidates edge 4 1361 3917 root215Tail5 winner.bound = root215Candidates4 := by
  unfold candidates
  have row : edge 3917 = (SortedNatTree.contains FastRows.row3917) := by
    change Data.row3917.contains = (SortedNatTree.contains FastRows.row3917)
    exact FastRows.row3917_checked.symm
  rw [row]
  decide
theorem root215Step4 : scan edge (search edge 3) 4 [1361] 1361 root215Tail4 winner =
    scan edge (search edge 3) 4 [1361] 1361 root215Tail5 winner := by
  rw [root215Tail4, scan, if_neg (by decide), root215Candidates4_checked]
  change scan edge (search edge 3) 4 [1361] 1361 root215Tail5
    (if 3 ≤ root215Candidates4.length then search edge 3 [1361, 3917] root215Candidates4 5278 winner else winner) = _
  rw [if_neg (by decide)]
def root215Candidates5 : List Nat := [20681]
set_option maxRecDepth 8192 in
theorem root215Candidates5_checked : candidates edge 4 1361 3947 root215Tail6 winner.bound = root215Candidates5 := by
  unfold candidates
  have row : edge 3947 = (SortedNatTree.contains FastRows.row3947) := by
    change Data.row3947.contains = (SortedNatTree.contains FastRows.row3947)
    exact FastRows.row3947_checked.symm
  rw [row]
  decide
theorem root215Step5 : scan edge (search edge 3) 4 [1361] 1361 root215Tail5 winner =
    scan edge (search edge 3) 4 [1361] 1361 root215Tail6 winner := by
  rw [root215Tail5, scan, if_neg (by decide), root215Candidates5_checked]
  change scan edge (search edge 3) 4 [1361] 1361 root215Tail6
    (if 3 ≤ root215Candidates5.length then search edge 3 [1361, 3947] root215Candidates5 5308 winner else winner) = _
  rw [if_neg (by decide)]
def root215Candidates6 : List Nat := [5099]
set_option maxRecDepth 8192 in
theorem root215Candidates6_checked : candidates edge 4 1361 4793 root215Tail7 winner.bound = root215Candidates6 := by
  unfold candidates
  have row : edge 4793 = (SortedNatTree.contains FastRows.row4793) := by
    change Data.row4793.contains = (SortedNatTree.contains FastRows.row4793)
    exact FastRows.row4793_checked.symm
  rw [row]
  decide
theorem root215Step6 : scan edge (search edge 3) 4 [1361] 1361 root215Tail6 winner =
    scan edge (search edge 3) 4 [1361] 1361 root215Tail7 winner := by
  rw [root215Tail6, scan, if_neg (by decide), root215Candidates6_checked]
  change scan edge (search edge 3) 4 [1361] 1361 root215Tail7
    (if 3 ≤ root215Candidates6.length then search edge 3 [1361, 4793] root215Candidates6 6154 winner else winner) = _
  rw [if_neg (by decide)]
def root215Candidates7 : List Nat := [9467, 11939, 14747]
set_option maxRecDepth 8192 in
theorem root215Candidates7_checked : candidates edge 4 1361 5099 root215Tail8 winner.bound = root215Candidates7 := by
  unfold candidates
  have row : edge 5099 = (SortedNatTree.contains FastRows.row5099) := by
    change Data.row5099.contains = (SortedNatTree.contains FastRows.row5099)
    exact FastRows.row5099_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root215Child7 : search edge 3 [1361, 5099] root215Candidates7 6460 winner = winner := by decide
theorem root215Step7 : scan edge (search edge 3) 4 [1361] 1361 root215Tail7 winner =
    scan edge (search edge 3) 4 [1361] 1361 root215Tail8 winner := by
  rw [root215Tail7, scan, if_neg (by decide), root215Candidates7_checked]
  change scan edge (search edge 3) 4 [1361] 1361 root215Tail8
    (if 3 ≤ root215Candidates7.length then search edge 3 [1361, 5099] root215Candidates7 6460 winner else winner) = _
  rw [if_pos (by decide), root215Child7]
def root215Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root215Candidates8_checked : candidates edge 4 1361 5669 root215Tail9 winner.bound = root215Candidates8 := by
  unfold candidates
  have row : edge 5669 = (SortedNatTree.contains FastRows.row5669) := by
    change Data.row5669.contains = (SortedNatTree.contains FastRows.row5669)
    exact FastRows.row5669_checked.symm
  rw [row]
  decide
theorem root215Step8 : scan edge (search edge 3) 4 [1361] 1361 root215Tail8 winner =
    scan edge (search edge 3) 4 [1361] 1361 root215Tail9 winner := by
  rw [root215Tail8, scan, if_neg (by decide), root215Candidates8_checked]
  change scan edge (search edge 3) 4 [1361] 1361 root215Tail9
    (if 3 ≤ root215Candidates8.length then search edge 3 [1361, 5669] root215Candidates8 7030 winner else winner) = _
  rw [if_neg (by decide)]
def root215Candidates9 : List Nat := [6737, 9467, 14543, 14969]
set_option maxRecDepth 8192 in
theorem root215Candidates9_checked : candidates edge 4 1361 6047 root215Tail10 winner.bound = root215Candidates9 := by
  unfold candidates
  have row : edge 6047 = (SortedNatTree.contains FastRows.row6047) := by
    change Data.row6047.contains = (SortedNatTree.contains FastRows.row6047)
    exact FastRows.row6047_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root215Child9 : search edge 3 [1361, 6047] root215Candidates9 7408 winner = winner := by decide
theorem root215Step9 : scan edge (search edge 3) 4 [1361] 1361 root215Tail9 winner =
    scan edge (search edge 3) 4 [1361] 1361 root215Tail10 winner := by
  rw [root215Tail9, scan, if_neg (by decide), root215Candidates9_checked]
  change scan edge (search edge 3) 4 [1361] 1361 root215Tail10
    (if 3 ≤ root215Candidates9.length then search edge 3 [1361, 6047] root215Candidates9 7408 winner else winner) = _
  rw [if_pos (by decide), root215Child9]
theorem root215Scan10 : scan edge (search edge 3) 4 [1361] 1361 root215Tail10 winner = winner := by rw [root215Tail10, scan, if_pos (by decide)]
theorem root215Scan9 : scan edge (search edge 3) 4 [1361] 1361 root215Tail9 winner = winner :=
  (root215Step9).trans root215Scan10
theorem root215Scan8 : scan edge (search edge 3) 4 [1361] 1361 root215Tail8 winner = winner :=
  (root215Step8).trans root215Scan9
theorem root215Scan7 : scan edge (search edge 3) 4 [1361] 1361 root215Tail7 winner = winner :=
  (root215Step7).trans root215Scan8
theorem root215Scan6 : scan edge (search edge 3) 4 [1361] 1361 root215Tail6 winner = winner :=
  (root215Step6).trans root215Scan7
theorem root215Scan5 : scan edge (search edge 3) 4 [1361] 1361 root215Tail5 winner = winner :=
  (root215Step5).trans root215Scan6
theorem root215Scan4 : scan edge (search edge 3) 4 [1361] 1361 root215Tail4 winner = winner :=
  (root215Step4).trans root215Scan5
theorem root215Scan3 : scan edge (search edge 3) 4 [1361] 1361 root215Tail3 winner = winner :=
  (root215Step3).trans root215Scan4
theorem root215Scan2 : scan edge (search edge 3) 4 [1361] 1361 root215Tail2 winner = winner :=
  (root215Step2).trans root215Scan3
theorem root215Scan1 : scan edge (search edge 3) 4 [1361] 1361 root215Tail1 winner = winner :=
  (root215Step1).trans root215Scan2
theorem root215Scan0 : scan edge (search edge 3) 4 [1361] 1361 root215Tail0 winner = winner :=
  (root215Step0).trans root215Scan1
set_option maxRecDepth 4096 in
theorem root215Branch : search edge 4 [1361] root215Candidates 1361 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1361] 1361 root215Tail0 winner = winner
  exact root215Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
