import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root382Candidates : List Nat := Data.row2659
set_option maxRecDepth 8192 in
theorem root382Candidates_checked : candidates edge 5 0 2659 tail383 winner.bound = root382Candidates := by
  apply candidates_eq_row edge 5 0 2659 tail383 root382Candidates winner.bound tail383_ordered
  · decide
  · decide
  · funext q
    rfl
def root382Tail38 : List Nat := []
def root382Tail37 : List Nat := 22777 :: root382Tail38
def root382Tail36 : List Nat := 22543 :: root382Tail37
def root382Tail35 : List Nat := 21757 :: root382Tail36
def root382Tail34 : List Nat := 20959 :: root382Tail35
def root382Tail33 : List Nat := 20857 :: root382Tail34
def root382Tail32 : List Nat := 20233 :: root382Tail33
def root382Tail31 : List Nat := 18973 :: root382Tail32
def root382Tail30 : List Nat := 18229 :: root382Tail31
def root382Tail29 : List Nat := 17737 :: root382Tail30
def root382Tail28 : List Nat := 17497 :: root382Tail29
def root382Tail27 : List Nat := 16477 :: root382Tail28
def root382Tail26 : List Nat := 14503 :: root382Tail27
def root382Tail25 : List Nat := 12979 :: root382Tail26
def root382Tail24 : List Nat := 12853 :: root382Tail25
def root382Tail23 : List Nat := 12379 :: root382Tail24
def root382Tail22 : List Nat := 12109 :: root382Tail23
def root382Tail21 : List Nat := 11287 :: root382Tail22
def root382Tail20 : List Nat := 10837 :: root382Tail21
def root382Tail19 : List Nat := 10627 :: root382Tail20
def root382Tail18 : List Nat := 9127 :: root382Tail19
def root382Tail17 : List Nat := 9091 :: root382Tail18
def root382Tail16 : List Nat := 8641 :: root382Tail17
def root382Tail15 : List Nat := 7927 :: root382Tail16
def root382Tail14 : List Nat := 7603 :: root382Tail15
def root382Tail13 : List Nat := 6949 :: root382Tail14
def root382Tail12 : List Nat := 5869 :: root382Tail13
def root382Tail11 : List Nat := 5857 :: root382Tail12
def root382Tail10 : List Nat := 5827 :: root382Tail11
def root382Tail9 : List Nat := 5167 :: root382Tail10
def root382Tail8 : List Nat := 4933 :: root382Tail9
def root382Tail7 : List Nat := 4759 :: root382Tail8
def root382Tail6 : List Nat := 4723 :: root382Tail7
def root382Tail5 : List Nat := 4423 :: root382Tail6
def root382Tail4 : List Nat := 3907 :: root382Tail5
def root382Tail3 : List Nat := 3739 :: root382Tail4
def root382Tail2 : List Nat := 3631 :: root382Tail3
def root382Tail1 : List Nat := 3169 :: root382Tail2
def root382Tail0 : List Nat := 3079 :: root382Tail1
def root382Candidates0 : List Nat := [17497]
set_option maxRecDepth 8192 in
theorem root382Candidates0_checked : candidates edge 4 2659 3079 root382Tail1 winner.bound = root382Candidates0 := by
  unfold candidates
  have row : edge 3079 = (SortedNatTree.contains FastRows.row3079) := by
    change Data.row3079.contains = (SortedNatTree.contains FastRows.row3079)
    exact FastRows.row3079_checked.symm
  rw [row]
  decide
theorem root382Step0 : scan edge (search edge 3) 4 [2659] 2659 root382Tail0 winner =
    scan edge (search edge 3) 4 [2659] 2659 root382Tail1 winner := by
  rw [root382Tail0, scan, if_neg (by decide), root382Candidates0_checked]
  change scan edge (search edge 3) 4 [2659] 2659 root382Tail1
    (if 3 ≤ root382Candidates0.length then search edge 3 [2659, 3079] root382Candidates0 5738 winner else winner) = _
  rw [if_neg (by decide)]
def root382Candidates1 : List Nat := [14503]
set_option maxRecDepth 8192 in
theorem root382Candidates1_checked : candidates edge 4 2659 3169 root382Tail2 winner.bound = root382Candidates1 := by
  unfold candidates
  have row : edge 3169 = (SortedNatTree.contains FastRows.row3169) := by
    change Data.row3169.contains = (SortedNatTree.contains FastRows.row3169)
    exact FastRows.row3169_checked.symm
  rw [row]
  decide
theorem root382Step1 : scan edge (search edge 3) 4 [2659] 2659 root382Tail1 winner =
    scan edge (search edge 3) 4 [2659] 2659 root382Tail2 winner := by
  rw [root382Tail1, scan, if_neg (by decide), root382Candidates1_checked]
  change scan edge (search edge 3) 4 [2659] 2659 root382Tail2
    (if 3 ≤ root382Candidates1.length then search edge 3 [2659, 3169] root382Candidates1 5828 winner else winner) = _
  rw [if_neg (by decide)]
def root382Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root382Candidates2_checked : candidates edge 4 2659 3631 root382Tail3 winner.bound = root382Candidates2 := by
  unfold candidates
  have row : edge 3631 = (SortedNatTree.contains FastRows.row3631) := by
    change Data.row3631.contains = (SortedNatTree.contains FastRows.row3631)
    exact FastRows.row3631_checked.symm
  rw [row]
  decide
theorem root382Step2 : scan edge (search edge 3) 4 [2659] 2659 root382Tail2 winner =
    scan edge (search edge 3) 4 [2659] 2659 root382Tail3 winner := by
  rw [root382Tail2, scan, if_neg (by decide), root382Candidates2_checked]
  change scan edge (search edge 3) 4 [2659] 2659 root382Tail3
    (if 3 ≤ root382Candidates2.length then search edge 3 [2659, 3631] root382Candidates2 6290 winner else winner) = _
  rw [if_neg (by decide)]
def root382Candidates3 : List Nat := [12379]
set_option maxRecDepth 8192 in
theorem root382Candidates3_checked : candidates edge 4 2659 3739 root382Tail4 winner.bound = root382Candidates3 := by
  unfold candidates
  have row : edge 3739 = (SortedNatTree.contains FastRows.row3739) := by
    change Data.row3739.contains = (SortedNatTree.contains FastRows.row3739)
    exact FastRows.row3739_checked.symm
  rw [row]
  decide
theorem root382Step3 : scan edge (search edge 3) 4 [2659] 2659 root382Tail3 winner =
    scan edge (search edge 3) 4 [2659] 2659 root382Tail4 winner := by
  rw [root382Tail3, scan, if_neg (by decide), root382Candidates3_checked]
  change scan edge (search edge 3) 4 [2659] 2659 root382Tail4
    (if 3 ≤ root382Candidates3.length then search edge 3 [2659, 3739] root382Candidates3 6398 winner else winner) = _
  rw [if_neg (by decide)]
def root382Candidates4 : List Nat := [14503]
set_option maxRecDepth 8192 in
theorem root382Candidates4_checked : candidates edge 4 2659 3907 root382Tail5 winner.bound = root382Candidates4 := by
  unfold candidates
  have row : edge 3907 = (SortedNatTree.contains FastRows.row3907) := by
    change Data.row3907.contains = (SortedNatTree.contains FastRows.row3907)
    exact FastRows.row3907_checked.symm
  rw [row]
  decide
theorem root382Step4 : scan edge (search edge 3) 4 [2659] 2659 root382Tail4 winner =
    scan edge (search edge 3) 4 [2659] 2659 root382Tail5 winner := by
  rw [root382Tail4, scan, if_neg (by decide), root382Candidates4_checked]
  change scan edge (search edge 3) 4 [2659] 2659 root382Tail5
    (if 3 ≤ root382Candidates4.length then search edge 3 [2659, 3907] root382Candidates4 6566 winner else winner) = _
  rw [if_neg (by decide)]
def root382Candidates5 : List Nat := [5869, 12109]
set_option maxRecDepth 8192 in
theorem root382Candidates5_checked : candidates edge 4 2659 4423 root382Tail6 winner.bound = root382Candidates5 := by
  unfold candidates
  have row : edge 4423 = (SortedNatTree.contains FastRows.row4423) := by
    change Data.row4423.contains = (SortedNatTree.contains FastRows.row4423)
    exact FastRows.row4423_checked.symm
  rw [row]
  decide
theorem root382Step5 : scan edge (search edge 3) 4 [2659] 2659 root382Tail5 winner =
    scan edge (search edge 3) 4 [2659] 2659 root382Tail6 winner := by
  rw [root382Tail5, scan, if_neg (by decide), root382Candidates5_checked]
  change scan edge (search edge 3) 4 [2659] 2659 root382Tail6
    (if 3 ≤ root382Candidates5.length then search edge 3 [2659, 4423] root382Candidates5 7082 winner else winner) = _
  rw [if_neg (by decide)]
def root382Candidates6 : List Nat := [5869, 12979]
set_option maxRecDepth 8192 in
theorem root382Candidates6_checked : candidates edge 4 2659 4723 root382Tail7 winner.bound = root382Candidates6 := by
  unfold candidates
  have row : edge 4723 = (SortedNatTree.contains FastRows.row4723) := by
    change Data.row4723.contains = (SortedNatTree.contains FastRows.row4723)
    exact FastRows.row4723_checked.symm
  rw [row]
  decide
theorem root382Step6 : scan edge (search edge 3) 4 [2659] 2659 root382Tail6 winner =
    scan edge (search edge 3) 4 [2659] 2659 root382Tail7 winner := by
  rw [root382Tail6, scan, if_neg (by decide), root382Candidates6_checked]
  change scan edge (search edge 3) 4 [2659] 2659 root382Tail7
    (if 3 ≤ root382Candidates6.length then search edge 3 [2659, 4723] root382Candidates6 7382 winner else winner) = _
  rw [if_neg (by decide)]
def root382Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root382Candidates7_checked : candidates edge 4 2659 4759 root382Tail8 winner.bound = root382Candidates7 := by
  unfold candidates
  have row : edge 4759 = (SortedNatTree.contains FastRows.row4759) := by
    change Data.row4759.contains = (SortedNatTree.contains FastRows.row4759)
    exact FastRows.row4759_checked.symm
  rw [row]
  decide
theorem root382Step7 : scan edge (search edge 3) 4 [2659] 2659 root382Tail7 winner =
    scan edge (search edge 3) 4 [2659] 2659 root382Tail8 winner := by
  rw [root382Tail7, scan, if_neg (by decide), root382Candidates7_checked]
  change scan edge (search edge 3) 4 [2659] 2659 root382Tail8
    (if 3 ≤ root382Candidates7.length then search edge 3 [2659, 4759] root382Candidates7 7418 winner else winner) = _
  rw [if_neg (by decide)]
def root382Candidates8 : List Nat := [12109]
set_option maxRecDepth 8192 in
theorem root382Candidates8_checked : candidates edge 4 2659 4933 root382Tail9 winner.bound = root382Candidates8 := by
  unfold candidates
  have row : edge 4933 = (SortedNatTree.contains FastRows.row4933) := by
    change Data.row4933.contains = (SortedNatTree.contains FastRows.row4933)
    exact FastRows.row4933_checked.symm
  rw [row]
  decide
theorem root382Step8 : scan edge (search edge 3) 4 [2659] 2659 root382Tail8 winner =
    scan edge (search edge 3) 4 [2659] 2659 root382Tail9 winner := by
  rw [root382Tail8, scan, if_neg (by decide), root382Candidates8_checked]
  change scan edge (search edge 3) 4 [2659] 2659 root382Tail9
    (if 3 ≤ root382Candidates8.length then search edge 3 [2659, 4933] root382Candidates8 7592 winner else winner) = _
  rw [if_neg (by decide)]
def root382Candidates9 : List Nat := [10837]
set_option maxRecDepth 8192 in
theorem root382Candidates9_checked : candidates edge 4 2659 5167 root382Tail10 winner.bound = root382Candidates9 := by
  unfold candidates
  have row : edge 5167 = (SortedNatTree.contains FastRows.row5167) := by
    change Data.row5167.contains = (SortedNatTree.contains FastRows.row5167)
    exact FastRows.row5167_checked.symm
  rw [row]
  decide
theorem root382Step9 : scan edge (search edge 3) 4 [2659] 2659 root382Tail9 winner =
    scan edge (search edge 3) 4 [2659] 2659 root382Tail10 winner := by
  rw [root382Tail9, scan, if_neg (by decide), root382Candidates9_checked]
  change scan edge (search edge 3) 4 [2659] 2659 root382Tail10
    (if 3 ≤ root382Candidates9.length then search edge 3 [2659, 5167] root382Candidates9 7826 winner else winner) = _
  rw [if_neg (by decide)]
def root382Candidates10 : List Nat := [9127]
set_option maxRecDepth 8192 in
theorem root382Candidates10_checked : candidates edge 4 2659 5827 root382Tail11 winner.bound = root382Candidates10 := by
  unfold candidates
  have row : edge 5827 = (SortedNatTree.contains FastRows.row5827) := by
    change Data.row5827.contains = (SortedNatTree.contains FastRows.row5827)
    exact FastRows.row5827_checked.symm
  rw [row]
  decide
theorem root382Step10 : scan edge (search edge 3) 4 [2659] 2659 root382Tail10 winner =
    scan edge (search edge 3) 4 [2659] 2659 root382Tail11 winner := by
  rw [root382Tail10, scan, if_neg (by decide), root382Candidates10_checked]
  change scan edge (search edge 3) 4 [2659] 2659 root382Tail11
    (if 3 ≤ root382Candidates10.length then search edge 3 [2659, 5827] root382Candidates10 8486 winner else winner) = _
  rw [if_neg (by decide)]
theorem root382Scan11 : scan edge (search edge 3) 4 [2659] 2659 root382Tail11 winner = winner := by rw [root382Tail11, scan, if_pos (by decide)]
theorem root382Scan10 : scan edge (search edge 3) 4 [2659] 2659 root382Tail10 winner = winner :=
  (root382Step10).trans root382Scan11
theorem root382Scan9 : scan edge (search edge 3) 4 [2659] 2659 root382Tail9 winner = winner :=
  (root382Step9).trans root382Scan10
theorem root382Scan8 : scan edge (search edge 3) 4 [2659] 2659 root382Tail8 winner = winner :=
  (root382Step8).trans root382Scan9
theorem root382Scan7 : scan edge (search edge 3) 4 [2659] 2659 root382Tail7 winner = winner :=
  (root382Step7).trans root382Scan8
theorem root382Scan6 : scan edge (search edge 3) 4 [2659] 2659 root382Tail6 winner = winner :=
  (root382Step6).trans root382Scan7
theorem root382Scan5 : scan edge (search edge 3) 4 [2659] 2659 root382Tail5 winner = winner :=
  (root382Step5).trans root382Scan6
theorem root382Scan4 : scan edge (search edge 3) 4 [2659] 2659 root382Tail4 winner = winner :=
  (root382Step4).trans root382Scan5
theorem root382Scan3 : scan edge (search edge 3) 4 [2659] 2659 root382Tail3 winner = winner :=
  (root382Step3).trans root382Scan4
theorem root382Scan2 : scan edge (search edge 3) 4 [2659] 2659 root382Tail2 winner = winner :=
  (root382Step2).trans root382Scan3
theorem root382Scan1 : scan edge (search edge 3) 4 [2659] 2659 root382Tail1 winner = winner :=
  (root382Step1).trans root382Scan2
theorem root382Scan0 : scan edge (search edge 3) 4 [2659] 2659 root382Tail0 winner = winner :=
  (root382Step0).trans root382Scan1
set_option maxRecDepth 4096 in
theorem root382Branch : search edge 4 [2659] root382Candidates 2659 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2659] 2659 root382Tail0 winner = winner
  exact root382Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
