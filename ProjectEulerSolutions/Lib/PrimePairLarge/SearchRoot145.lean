import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root145Candidates : List Nat := Data.row857
set_option maxRecDepth 8192 in
theorem root145Candidates_checked : candidates edge 5 0 857 tail146 winner.bound = root145Candidates := by
  apply candidates_eq_row edge 5 0 857 tail146 root145Candidates winner.bound tail146_ordered
  · decide
  · decide
  · funext q
    rfl
def root145Tail54 : List Nat := []
def root145Tail53 : List Nat := 24767 :: root145Tail54
def root145Tail52 : List Nat := 24329 :: root145Tail53
def root145Tail51 : List Nat := 24239 :: root145Tail52
def root145Tail50 : List Nat := 23549 :: root145Tail51
def root145Tail49 : List Nat := 22787 :: root145Tail50
def root145Tail48 : List Nat := 21821 :: root145Tail49
def root145Tail47 : List Nat := 21569 :: root145Tail48
def root145Tail46 : List Nat := 20753 :: root145Tail47
def root145Tail45 : List Nat := 20693 :: root145Tail46
def root145Tail44 : List Nat := 20507 :: root145Tail45
def root145Tail43 : List Nat := 20441 :: root145Tail44
def root145Tail42 : List Nat := 20399 :: root145Tail43
def root145Tail41 : List Nat := 19961 :: root145Tail42
def root145Tail40 : List Nat := 19553 :: root145Tail41
def root145Tail39 : List Nat := 18617 :: root145Tail40
def root145Tail38 : List Nat := 18341 :: root145Tail39
def root145Tail37 : List Nat := 16451 :: root145Tail38
def root145Tail36 : List Nat := 15803 :: root145Tail37
def root145Tail35 : List Nat := 15551 :: root145Tail36
def root145Tail34 : List Nat := 15101 :: root145Tail35
def root145Tail33 : List Nat := 14891 :: root145Tail34
def root145Tail32 : List Nat := 14771 :: root145Tail33
def root145Tail31 : List Nat := 14639 :: root145Tail32
def root145Tail30 : List Nat := 14009 :: root145Tail31
def root145Tail29 : List Nat := 13553 :: root145Tail30
def root145Tail28 : List Nat := 13487 :: root145Tail29
def root145Tail27 : List Nat := 12569 :: root145Tail28
def root145Tail26 : List Nat := 12437 :: root145Tail27
def root145Tail25 : List Nat := 11519 :: root145Tail26
def root145Tail24 : List Nat := 11393 :: root145Tail25
def root145Tail23 : List Nat := 11117 :: root145Tail24
def root145Tail22 : List Nat := 10883 :: root145Tail23
def root145Tail21 : List Nat := 10589 :: root145Tail22
def root145Tail20 : List Nat := 10337 :: root145Tail21
def root145Tail19 : List Nat := 9743 :: root145Tail20
def root145Tail18 : List Nat := 9491 :: root145Tail19
def root145Tail17 : List Nat := 8699 :: root145Tail18
def root145Tail16 : List Nat := 8681 :: root145Tail17
def root145Tail15 : List Nat := 8243 :: root145Tail16
def root145Tail14 : List Nat := 8147 :: root145Tail15
def root145Tail13 : List Nat := 7529 :: root145Tail14
def root145Tail12 : List Nat := 7307 :: root145Tail13
def root145Tail11 : List Nat := 7187 :: root145Tail12
def root145Tail10 : List Nat := 6329 :: root145Tail11
def root145Tail9 : List Nat := 6131 :: root145Tail10
def root145Tail8 : List Nat := 4877 :: root145Tail9
def root145Tail7 : List Nat := 4463 :: root145Tail8
def root145Tail6 : List Nat := 4079 :: root145Tail7
def root145Tail5 : List Nat := 3413 :: root145Tail6
def root145Tail4 : List Nat := 2693 :: root145Tail5
def root145Tail3 : List Nat := 2633 :: root145Tail4
def root145Tail2 : List Nat := 1973 :: root145Tail3
def root145Tail1 : List Nat := 1601 :: root145Tail2
def root145Tail0 : List Nat := 1301 :: root145Tail1
def root145Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root145Candidates0_checked : candidates edge 4 857 1301 root145Tail1 winner.bound = root145Candidates0 := by
  unfold candidates
  have row : edge 1301 = (SortedNatTree.contains FastRows.row1301) := by
    change Data.row1301.contains = (SortedNatTree.contains FastRows.row1301)
    exact FastRows.row1301_checked.symm
  rw [row]
  decide
theorem root145Step0 : scan edge (search edge 3) 4 [857] 857 root145Tail0 winner =
    scan edge (search edge 3) 4 [857] 857 root145Tail1 winner := by
  rw [root145Tail0, scan, if_neg (by decide), root145Candidates0_checked]
  change scan edge (search edge 3) 4 [857] 857 root145Tail1
    (if 3 ≤ root145Candidates0.length then search edge 3 [857, 1301] root145Candidates0 2158 winner else winner) = _
  rw [if_neg (by decide)]
def root145Candidates1 : List Nat := [7307, 21821]
set_option maxRecDepth 8192 in
theorem root145Candidates1_checked : candidates edge 4 857 1601 root145Tail2 winner.bound = root145Candidates1 := by
  unfold candidates
  have row : edge 1601 = (SortedNatTree.contains FastRows.row1601) := by
    change Data.row1601.contains = (SortedNatTree.contains FastRows.row1601)
    exact FastRows.row1601_checked.symm
  rw [row]
  decide
theorem root145Step1 : scan edge (search edge 3) 4 [857] 857 root145Tail1 winner =
    scan edge (search edge 3) 4 [857] 857 root145Tail2 winner := by
  rw [root145Tail1, scan, if_neg (by decide), root145Candidates1_checked]
  change scan edge (search edge 3) 4 [857] 857 root145Tail2
    (if 3 ≤ root145Candidates1.length then search edge 3 [857, 1601] root145Candidates1 2458 winner else winner) = _
  rw [if_neg (by decide)]
def root145Candidates2 : List Nat := [6131, 14771, 16451, 20399, 20441]
set_option maxRecDepth 8192 in
theorem root145Candidates2_checked : candidates edge 4 857 1973 root145Tail3 winner.bound = root145Candidates2 := by
  unfold candidates
  have row : edge 1973 = (SortedNatTree.contains FastRows.row1973) := by
    change Data.row1973.contains = (SortedNatTree.contains FastRows.row1973)
    exact FastRows.row1973_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root145Child2 : search edge 3 [857, 1973] root145Candidates2 2830 winner = winner := by decide
theorem root145Step2 : scan edge (search edge 3) 4 [857] 857 root145Tail2 winner =
    scan edge (search edge 3) 4 [857] 857 root145Tail3 winner := by
  rw [root145Tail2, scan, if_neg (by decide), root145Candidates2_checked]
  change scan edge (search edge 3) 4 [857] 857 root145Tail3
    (if 3 ≤ root145Candidates2.length then search edge 3 [857, 1973] root145Candidates2 2830 winner else winner) = _
  rw [if_pos (by decide), root145Child2]
def root145Candidates3 : List Nat := [14771]
set_option maxRecDepth 8192 in
theorem root145Candidates3_checked : candidates edge 4 857 2633 root145Tail4 winner.bound = root145Candidates3 := by
  unfold candidates
  have row : edge 2633 = (SortedNatTree.contains FastRows.row2633) := by
    change Data.row2633.contains = (SortedNatTree.contains FastRows.row2633)
    exact FastRows.row2633_checked.symm
  rw [row]
  decide
theorem root145Step3 : scan edge (search edge 3) 4 [857] 857 root145Tail3 winner =
    scan edge (search edge 3) 4 [857] 857 root145Tail4 winner := by
  rw [root145Tail3, scan, if_neg (by decide), root145Candidates3_checked]
  change scan edge (search edge 3) 4 [857] 857 root145Tail4
    (if 3 ≤ root145Candidates3.length then search edge 3 [857, 2633] root145Candidates3 3490 winner else winner) = _
  rw [if_neg (by decide)]
def root145Candidates4 : List Nat := [4079, 13553, 18341]
set_option maxRecDepth 8192 in
theorem root145Candidates4_checked : candidates edge 4 857 2693 root145Tail5 winner.bound = root145Candidates4 := by
  unfold candidates
  have row : edge 2693 = (SortedNatTree.contains FastRows.row2693) := by
    change Data.row2693.contains = (SortedNatTree.contains FastRows.row2693)
    exact FastRows.row2693_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root145Child4 : search edge 3 [857, 2693] root145Candidates4 3550 winner = winner := by decide
theorem root145Step4 : scan edge (search edge 3) 4 [857] 857 root145Tail4 winner =
    scan edge (search edge 3) 4 [857] 857 root145Tail5 winner := by
  rw [root145Tail4, scan, if_neg (by decide), root145Candidates4_checked]
  change scan edge (search edge 3) 4 [857] 857 root145Tail5
    (if 3 ≤ root145Candidates4.length then search edge 3 [857, 2693] root145Candidates4 3550 winner else winner) = _
  rw [if_pos (by decide), root145Child4]
def root145Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root145Candidates5_checked : candidates edge 4 857 3413 root145Tail6 winner.bound = root145Candidates5 := by
  unfold candidates
  have row : edge 3413 = (SortedNatTree.contains FastRows.row3413) := by
    change Data.row3413.contains = (SortedNatTree.contains FastRows.row3413)
    exact FastRows.row3413_checked.symm
  rw [row]
  decide
theorem root145Step5 : scan edge (search edge 3) 4 [857] 857 root145Tail5 winner =
    scan edge (search edge 3) 4 [857] 857 root145Tail6 winner := by
  rw [root145Tail5, scan, if_neg (by decide), root145Candidates5_checked]
  change scan edge (search edge 3) 4 [857] 857 root145Tail6
    (if 3 ≤ root145Candidates5.length then search edge 3 [857, 3413] root145Candidates5 4270 winner else winner) = _
  rw [if_neg (by decide)]
def root145Candidates6 : List Nat := [6131, 9743, 13553]
set_option maxRecDepth 8192 in
theorem root145Candidates6_checked : candidates edge 4 857 4079 root145Tail7 winner.bound = root145Candidates6 := by
  unfold candidates
  have row : edge 4079 = (SortedNatTree.contains FastRows.row4079) := by
    change Data.row4079.contains = (SortedNatTree.contains FastRows.row4079)
    exact FastRows.row4079_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root145Child6 : search edge 3 [857, 4079] root145Candidates6 4936 winner = winner := by decide
theorem root145Step6 : scan edge (search edge 3) 4 [857] 857 root145Tail6 winner =
    scan edge (search edge 3) 4 [857] 857 root145Tail7 winner := by
  rw [root145Tail6, scan, if_neg (by decide), root145Candidates6_checked]
  change scan edge (search edge 3) 4 [857] 857 root145Tail7
    (if 3 ≤ root145Candidates6.length then search edge 3 [857, 4079] root145Candidates6 4936 winner else winner) = _
  rw [if_pos (by decide), root145Child6]
def root145Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root145Candidates7_checked : candidates edge 4 857 4463 root145Tail8 winner.bound = root145Candidates7 := by
  unfold candidates
  have row : edge 4463 = (SortedNatTree.contains FastRows.row4463) := by
    change Data.row4463.contains = (SortedNatTree.contains FastRows.row4463)
    exact FastRows.row4463_checked.symm
  rw [row]
  decide
theorem root145Step7 : scan edge (search edge 3) 4 [857] 857 root145Tail7 winner =
    scan edge (search edge 3) 4 [857] 857 root145Tail8 winner := by
  rw [root145Tail7, scan, if_neg (by decide), root145Candidates7_checked]
  change scan edge (search edge 3) 4 [857] 857 root145Tail8
    (if 3 ≤ root145Candidates7.length then search edge 3 [857, 4463] root145Candidates7 5320 winner else winner) = _
  rw [if_neg (by decide)]
def root145Candidates8 : List Nat := [7307, 14771, 14891]
set_option maxRecDepth 8192 in
theorem root145Candidates8_checked : candidates edge 4 857 4877 root145Tail9 winner.bound = root145Candidates8 := by
  unfold candidates
  have row : edge 4877 = (SortedNatTree.contains FastRows.row4877) := by
    change Data.row4877.contains = (SortedNatTree.contains FastRows.row4877)
    exact FastRows.row4877_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root145Child8 : search edge 3 [857, 4877] root145Candidates8 5734 winner = winner := by decide
theorem root145Step8 : scan edge (search edge 3) 4 [857] 857 root145Tail8 winner =
    scan edge (search edge 3) 4 [857] 857 root145Tail9 winner := by
  rw [root145Tail8, scan, if_neg (by decide), root145Candidates8_checked]
  change scan edge (search edge 3) 4 [857] 857 root145Tail9
    (if 3 ≤ root145Candidates8.length then search edge 3 [857, 4877] root145Candidates8 5734 winner else winner) = _
  rw [if_pos (by decide), root145Child8]
def root145Candidates9 : List Nat := [14771]
set_option maxRecDepth 8192 in
theorem root145Candidates9_checked : candidates edge 4 857 6131 root145Tail10 winner.bound = root145Candidates9 := by
  unfold candidates
  have row : edge 6131 = (SortedNatTree.contains FastRows.row6131) := by
    change Data.row6131.contains = (SortedNatTree.contains FastRows.row6131)
    exact FastRows.row6131_checked.symm
  rw [row]
  decide
theorem root145Step9 : scan edge (search edge 3) 4 [857] 857 root145Tail9 winner =
    scan edge (search edge 3) 4 [857] 857 root145Tail10 winner := by
  rw [root145Tail9, scan, if_neg (by decide), root145Candidates9_checked]
  change scan edge (search edge 3) 4 [857] 857 root145Tail10
    (if 3 ≤ root145Candidates9.length then search edge 3 [857, 6131] root145Candidates9 6988 winner else winner) = _
  rw [if_neg (by decide)]
theorem root145Scan10 : scan edge (search edge 3) 4 [857] 857 root145Tail10 winner = winner := by rw [root145Tail10, scan, if_pos (by decide)]
theorem root145Scan9 : scan edge (search edge 3) 4 [857] 857 root145Tail9 winner = winner :=
  (root145Step9).trans root145Scan10
theorem root145Scan8 : scan edge (search edge 3) 4 [857] 857 root145Tail8 winner = winner :=
  (root145Step8).trans root145Scan9
theorem root145Scan7 : scan edge (search edge 3) 4 [857] 857 root145Tail7 winner = winner :=
  (root145Step7).trans root145Scan8
theorem root145Scan6 : scan edge (search edge 3) 4 [857] 857 root145Tail6 winner = winner :=
  (root145Step6).trans root145Scan7
theorem root145Scan5 : scan edge (search edge 3) 4 [857] 857 root145Tail5 winner = winner :=
  (root145Step5).trans root145Scan6
theorem root145Scan4 : scan edge (search edge 3) 4 [857] 857 root145Tail4 winner = winner :=
  (root145Step4).trans root145Scan5
theorem root145Scan3 : scan edge (search edge 3) 4 [857] 857 root145Tail3 winner = winner :=
  (root145Step3).trans root145Scan4
theorem root145Scan2 : scan edge (search edge 3) 4 [857] 857 root145Tail2 winner = winner :=
  (root145Step2).trans root145Scan3
theorem root145Scan1 : scan edge (search edge 3) 4 [857] 857 root145Tail1 winner = winner :=
  (root145Step1).trans root145Scan2
theorem root145Scan0 : scan edge (search edge 3) 4 [857] 857 root145Tail0 winner = winner :=
  (root145Step0).trans root145Scan1
set_option maxRecDepth 4096 in
theorem root145Branch : search edge 4 [857] root145Candidates 857 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [857] 857 root145Tail0 winner = winner
  exact root145Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
