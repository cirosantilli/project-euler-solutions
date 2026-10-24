import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root155Candidates : List Nat := Data.row929
set_option maxRecDepth 8192 in
theorem root155Candidates_checked : candidates edge 5 0 929 tail156 winner.bound = root155Candidates := by
  apply candidates_eq_row edge 5 0 929 tail156 root155Candidates winner.bound tail156_ordered
  · decide
  · decide
  · funext q
    rfl
def root155Tail55 : List Nat := []
def root155Tail54 : List Nat := 23669 :: root155Tail55
def root155Tail53 : List Nat := 23603 :: root155Tail54
def root155Tail52 : List Nat := 22343 :: root155Tail53
def root155Tail51 : List Nat := 21977 :: root155Tail52
def root155Tail50 : List Nat := 21407 :: root155Tail51
def root155Tail49 : List Nat := 21377 :: root155Tail50
def root155Tail48 : List Nat := 21221 :: root155Tail49
def root155Tail47 : List Nat := 20771 :: root155Tail48
def root155Tail46 : List Nat := 20747 :: root155Tail47
def root155Tail45 : List Nat := 20129 :: root155Tail46
def root155Tail44 : List Nat := 19559 :: root155Tail45
def root155Tail43 : List Nat := 18899 :: root155Tail44
def root155Tail42 : List Nat := 17093 :: root155Tail43
def root155Tail41 : List Nat := 16787 :: root155Tail42
def root155Tail40 : List Nat := 15803 :: root155Tail41
def root155Tail39 : List Nat := 15551 :: root155Tail40
def root155Tail38 : List Nat := 15527 :: root155Tail39
def root155Tail37 : List Nat := 15107 :: root155Tail38
def root155Tail36 : List Nat := 14867 :: root155Tail37
def root155Tail35 : List Nat := 14519 :: root155Tail36
def root155Tail34 : List Nat := 14207 :: root155Tail35
def root155Tail33 : List Nat := 13679 :: root155Tail34
def root155Tail32 : List Nat := 13367 :: root155Tail33
def root155Tail31 : List Nat := 13049 :: root155Tail32
def root155Tail30 : List Nat := 12569 :: root155Tail31
def root155Tail29 : List Nat := 12401 :: root155Tail30
def root155Tail28 : List Nat := 12251 :: root155Tail29
def root155Tail27 : List Nat := 12107 :: root155Tail28
def root155Tail26 : List Nat := 11393 :: root155Tail27
def root155Tail25 : List Nat := 11243 :: root155Tail26
def root155Tail24 : List Nat := 10061 :: root155Tail25
def root155Tail23 : List Nat := 9029 :: root155Tail24
def root155Tail22 : List Nat := 8747 :: root155Tail23
def root155Tail21 : List Nat := 8369 :: root155Tail22
def root155Tail20 : List Nat := 8117 :: root155Tail21
def root155Tail19 : List Nat := 7853 :: root155Tail20
def root155Tail18 : List Nat := 7109 :: root155Tail19
def root155Tail17 : List Nat := 6563 :: root155Tail18
def root155Tail16 : List Nat := 5669 :: root155Tail17
def root155Tail15 : List Nat := 5639 :: root155Tail16
def root155Tail14 : List Nat := 4973 :: root155Tail15
def root155Tail13 : List Nat := 4463 :: root155Tail14
def root155Tail12 : List Nat := 4421 :: root155Tail13
def root155Tail11 : List Nat := 4073 :: root155Tail12
def root155Tail10 : List Nat := 4013 :: root155Tail11
def root155Tail9 : List Nat := 3761 :: root155Tail10
def root155Tail8 : List Nat := 3539 :: root155Tail9
def root155Tail7 : List Nat := 3467 :: root155Tail8
def root155Tail6 : List Nat := 3209 :: root155Tail7
def root155Tail5 : List Nat := 2741 :: root155Tail6
def root155Tail4 : List Nat := 2477 :: root155Tail5
def root155Tail3 : List Nat := 2039 :: root155Tail4
def root155Tail2 : List Nat := 983 :: root155Tail3
def root155Tail1 : List Nat := 953 :: root155Tail2
def root155Tail0 : List Nat := 941 :: root155Tail1
def root155Candidates0 : List Nat := [8369, 12107, 12401, 15551, 16787]
set_option maxRecDepth 8192 in
theorem root155Candidates0_checked : candidates edge 4 929 941 root155Tail1 winner.bound = root155Candidates0 := by
  unfold candidates
  have row : edge 941 = (SortedNatTree.contains FastRows.row941) := by
    change Data.row941.contains = (SortedNatTree.contains FastRows.row941)
    exact FastRows.row941_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root155Child0 : search edge 3 [929, 941] root155Candidates0 1870 winner = winner := by decide
theorem root155Step0 : scan edge (search edge 3) 4 [929] 929 root155Tail0 winner =
    scan edge (search edge 3) 4 [929] 929 root155Tail1 winner := by
  rw [root155Tail0, scan, if_neg (by decide), root155Candidates0_checked]
  change scan edge (search edge 3) 4 [929] 929 root155Tail1
    (if 3 ≤ root155Candidates0.length then search edge 3 [929, 941] root155Candidates0 1870 winner else winner) = _
  rw [if_pos (by decide), root155Child0]
def root155Candidates1 : List Nat := [7109, 9029]
set_option maxRecDepth 8192 in
theorem root155Candidates1_checked : candidates edge 4 929 953 root155Tail2 winner.bound = root155Candidates1 := by
  unfold candidates
  have row : edge 953 = (SortedNatTree.contains FastRows.row953) := by
    change Data.row953.contains = (SortedNatTree.contains FastRows.row953)
    exact FastRows.row953_checked.symm
  rw [row]
  decide
theorem root155Step1 : scan edge (search edge 3) 4 [929] 929 root155Tail1 winner =
    scan edge (search edge 3) 4 [929] 929 root155Tail2 winner := by
  rw [root155Tail1, scan, if_neg (by decide), root155Candidates1_checked]
  change scan edge (search edge 3) 4 [929] 929 root155Tail2
    (if 3 ≤ root155Candidates1.length then search edge 3 [929, 953] root155Candidates1 1882 winner else winner) = _
  rw [if_neg (by decide)]
def root155Candidates2 : List Nat := [4073, 7109, 14207]
set_option maxRecDepth 8192 in
theorem root155Candidates2_checked : candidates edge 4 929 983 root155Tail3 winner.bound = root155Candidates2 := by
  unfold candidates
  have row : edge 983 = (SortedNatTree.contains FastRows.row983) := by
    change Data.row983.contains = (SortedNatTree.contains FastRows.row983)
    exact FastRows.row983_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root155Child2 : search edge 3 [929, 983] root155Candidates2 1912 winner = winner := by decide
theorem root155Step2 : scan edge (search edge 3) 4 [929] 929 root155Tail2 winner =
    scan edge (search edge 3) 4 [929] 929 root155Tail3 winner := by
  rw [root155Tail2, scan, if_neg (by decide), root155Candidates2_checked]
  change scan edge (search edge 3) 4 [929] 929 root155Tail3
    (if 3 ≤ root155Candidates2.length then search edge 3 [929, 983] root155Candidates2 1912 winner else winner) = _
  rw [if_pos (by decide), root155Child2]
def root155Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root155Candidates3_checked : candidates edge 4 929 2039 root155Tail4 winner.bound = root155Candidates3 := by
  unfold candidates
  have row : edge 2039 = (SortedNatTree.contains FastRows.row2039) := by
    change Data.row2039.contains = (SortedNatTree.contains FastRows.row2039)
    exact FastRows.row2039_checked.symm
  rw [row]
  decide
theorem root155Step3 : scan edge (search edge 3) 4 [929] 929 root155Tail3 winner =
    scan edge (search edge 3) 4 [929] 929 root155Tail4 winner := by
  rw [root155Tail3, scan, if_neg (by decide), root155Candidates3_checked]
  change scan edge (search edge 3) 4 [929] 929 root155Tail4
    (if 3 ≤ root155Candidates3.length then search edge 3 [929, 2039] root155Candidates3 2968 winner else winner) = _
  rw [if_neg (by decide)]
def root155Candidates4 : List Nat := [4073, 7853, 12251, 15527]
set_option maxRecDepth 8192 in
theorem root155Candidates4_checked : candidates edge 4 929 2477 root155Tail5 winner.bound = root155Candidates4 := by
  unfold candidates
  have row : edge 2477 = (SortedNatTree.contains FastRows.row2477) := by
    change Data.row2477.contains = (SortedNatTree.contains FastRows.row2477)
    exact FastRows.row2477_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root155Child4 : search edge 3 [929, 2477] root155Candidates4 3406 winner = winner := by decide
theorem root155Step4 : scan edge (search edge 3) 4 [929] 929 root155Tail4 winner =
    scan edge (search edge 3) 4 [929] 929 root155Tail5 winner := by
  rw [root155Tail4, scan, if_neg (by decide), root155Candidates4_checked]
  change scan edge (search edge 3) 4 [929] 929 root155Tail5
    (if 3 ≤ root155Candidates4.length then search edge 3 [929, 2477] root155Candidates4 3406 winner else winner) = _
  rw [if_pos (by decide), root155Child4]
def root155Candidates5 : List Nat := [8747]
set_option maxRecDepth 8192 in
theorem root155Candidates5_checked : candidates edge 4 929 2741 root155Tail6 winner.bound = root155Candidates5 := by
  unfold candidates
  have row : edge 2741 = (SortedNatTree.contains FastRows.row2741) := by
    change Data.row2741.contains = (SortedNatTree.contains FastRows.row2741)
    exact FastRows.row2741_checked.symm
  rw [row]
  decide
theorem root155Step5 : scan edge (search edge 3) 4 [929] 929 root155Tail5 winner =
    scan edge (search edge 3) 4 [929] 929 root155Tail6 winner := by
  rw [root155Tail5, scan, if_neg (by decide), root155Candidates5_checked]
  change scan edge (search edge 3) 4 [929] 929 root155Tail6
    (if 3 ≤ root155Candidates5.length then search edge 3 [929, 2741] root155Candidates5 3670 winner else winner) = _
  rw [if_neg (by decide)]
def root155Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root155Candidates6_checked : candidates edge 4 929 3209 root155Tail7 winner.bound = root155Candidates6 := by
  unfold candidates
  have row : edge 3209 = (SortedNatTree.contains FastRows.row3209) := by
    change Data.row3209.contains = (SortedNatTree.contains FastRows.row3209)
    exact FastRows.row3209_checked.symm
  rw [row]
  decide
theorem root155Step6 : scan edge (search edge 3) 4 [929] 929 root155Tail6 winner =
    scan edge (search edge 3) 4 [929] 929 root155Tail7 winner := by
  rw [root155Tail6, scan, if_neg (by decide), root155Candidates6_checked]
  change scan edge (search edge 3) 4 [929] 929 root155Tail7
    (if 3 ≤ root155Candidates6.length then search edge 3 [929, 3209] root155Candidates6 4138 winner else winner) = _
  rw [if_neg (by decide)]
def root155Candidates7 : List Nat := [8117]
set_option maxRecDepth 8192 in
theorem root155Candidates7_checked : candidates edge 4 929 3467 root155Tail8 winner.bound = root155Candidates7 := by
  unfold candidates
  have row : edge 3467 = (SortedNatTree.contains FastRows.row3467) := by
    change Data.row3467.contains = (SortedNatTree.contains FastRows.row3467)
    exact FastRows.row3467_checked.symm
  rw [row]
  decide
theorem root155Step7 : scan edge (search edge 3) 4 [929] 929 root155Tail7 winner =
    scan edge (search edge 3) 4 [929] 929 root155Tail8 winner := by
  rw [root155Tail7, scan, if_neg (by decide), root155Candidates7_checked]
  change scan edge (search edge 3) 4 [929] 929 root155Tail8
    (if 3 ≤ root155Candidates7.length then search edge 3 [929, 3467] root155Candidates7 4396 winner else winner) = _
  rw [if_neg (by decide)]
def root155Candidates8 : List Nat := [4013]
set_option maxRecDepth 8192 in
theorem root155Candidates8_checked : candidates edge 4 929 3539 root155Tail9 winner.bound = root155Candidates8 := by
  unfold candidates
  have row : edge 3539 = (SortedNatTree.contains FastRows.row3539) := by
    change Data.row3539.contains = (SortedNatTree.contains FastRows.row3539)
    exact FastRows.row3539_checked.symm
  rw [row]
  decide
theorem root155Step8 : scan edge (search edge 3) 4 [929] 929 root155Tail8 winner =
    scan edge (search edge 3) 4 [929] 929 root155Tail9 winner := by
  rw [root155Tail8, scan, if_neg (by decide), root155Candidates8_checked]
  change scan edge (search edge 3) 4 [929] 929 root155Tail9
    (if 3 ≤ root155Candidates8.length then search edge 3 [929, 3539] root155Candidates8 4468 winner else winner) = _
  rw [if_neg (by decide)]
def root155Candidates9 : List Nat := [8369]
set_option maxRecDepth 8192 in
theorem root155Candidates9_checked : candidates edge 4 929 3761 root155Tail10 winner.bound = root155Candidates9 := by
  unfold candidates
  have row : edge 3761 = (SortedNatTree.contains FastRows.row3761) := by
    change Data.row3761.contains = (SortedNatTree.contains FastRows.row3761)
    exact FastRows.row3761_checked.symm
  rw [row]
  decide
theorem root155Step9 : scan edge (search edge 3) 4 [929] 929 root155Tail9 winner =
    scan edge (search edge 3) 4 [929] 929 root155Tail10 winner := by
  rw [root155Tail9, scan, if_neg (by decide), root155Candidates9_checked]
  change scan edge (search edge 3) 4 [929] 929 root155Tail10
    (if 3 ≤ root155Candidates9.length then search edge 3 [929, 3761] root155Candidates9 4690 winner else winner) = _
  rw [if_neg (by decide)]
def root155Candidates10 : List Nat := [4421]
set_option maxRecDepth 8192 in
theorem root155Candidates10_checked : candidates edge 4 929 4013 root155Tail11 winner.bound = root155Candidates10 := by
  unfold candidates
  have row : edge 4013 = (SortedNatTree.contains FastRows.row4013) := by
    change Data.row4013.contains = (SortedNatTree.contains FastRows.row4013)
    exact FastRows.row4013_checked.symm
  rw [row]
  decide
theorem root155Step10 : scan edge (search edge 3) 4 [929] 929 root155Tail10 winner =
    scan edge (search edge 3) 4 [929] 929 root155Tail11 winner := by
  rw [root155Tail10, scan, if_neg (by decide), root155Candidates10_checked]
  change scan edge (search edge 3) 4 [929] 929 root155Tail11
    (if 3 ≤ root155Candidates10.length then search edge 3 [929, 4013] root155Candidates10 4942 winner else winner) = _
  rw [if_neg (by decide)]
def root155Candidates11 : List Nat := [9029]
set_option maxRecDepth 8192 in
theorem root155Candidates11_checked : candidates edge 4 929 4073 root155Tail12 winner.bound = root155Candidates11 := by
  unfold candidates
  have row : edge 4073 = (SortedNatTree.contains FastRows.row4073) := by
    change Data.row4073.contains = (SortedNatTree.contains FastRows.row4073)
    exact FastRows.row4073_checked.symm
  rw [row]
  decide
theorem root155Step11 : scan edge (search edge 3) 4 [929] 929 root155Tail11 winner =
    scan edge (search edge 3) 4 [929] 929 root155Tail12 winner := by
  rw [root155Tail11, scan, if_neg (by decide), root155Candidates11_checked]
  change scan edge (search edge 3) 4 [929] 929 root155Tail12
    (if 3 ≤ root155Candidates11.length then search edge 3 [929, 4073] root155Candidates11 5002 winner else winner) = _
  rw [if_neg (by decide)]
def root155Candidates12 : List Nat := [6563, 11393, 12401, 15803]
set_option maxRecDepth 8192 in
theorem root155Candidates12_checked : candidates edge 4 929 4421 root155Tail13 winner.bound = root155Candidates12 := by
  unfold candidates
  have row : edge 4421 = (SortedNatTree.contains FastRows.row4421) := by
    change Data.row4421.contains = (SortedNatTree.contains FastRows.row4421)
    exact FastRows.row4421_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root155Child12 : search edge 3 [929, 4421] root155Candidates12 5350 winner = winner := by decide
theorem root155Step12 : scan edge (search edge 3) 4 [929] 929 root155Tail12 winner =
    scan edge (search edge 3) 4 [929] 929 root155Tail13 winner := by
  rw [root155Tail12, scan, if_neg (by decide), root155Candidates12_checked]
  change scan edge (search edge 3) 4 [929] 929 root155Tail13
    (if 3 ≤ root155Candidates12.length then search edge 3 [929, 4421] root155Candidates12 5350 winner else winner) = _
  rw [if_pos (by decide), root155Child12]
def root155Candidates13 : List Nat := [8369, 12107]
set_option maxRecDepth 8192 in
theorem root155Candidates13_checked : candidates edge 4 929 4463 root155Tail14 winner.bound = root155Candidates13 := by
  unfold candidates
  have row : edge 4463 = (SortedNatTree.contains FastRows.row4463) := by
    change Data.row4463.contains = (SortedNatTree.contains FastRows.row4463)
    exact FastRows.row4463_checked.symm
  rw [row]
  decide
theorem root155Step13 : scan edge (search edge 3) 4 [929] 929 root155Tail13 winner =
    scan edge (search edge 3) 4 [929] 929 root155Tail14 winner := by
  rw [root155Tail13, scan, if_neg (by decide), root155Candidates13_checked]
  change scan edge (search edge 3) 4 [929] 929 root155Tail14
    (if 3 ≤ root155Candidates13.length then search edge 3 [929, 4463] root155Candidates13 5392 winner else winner) = _
  rw [if_neg (by decide)]
def root155Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root155Candidates14_checked : candidates edge 4 929 4973 root155Tail15 winner.bound = root155Candidates14 := by
  unfold candidates
  have row : edge 4973 = (SortedNatTree.contains FastRows.row4973) := by
    change Data.row4973.contains = (SortedNatTree.contains FastRows.row4973)
    exact FastRows.row4973_checked.symm
  rw [row]
  decide
theorem root155Step14 : scan edge (search edge 3) 4 [929] 929 root155Tail14 winner =
    scan edge (search edge 3) 4 [929] 929 root155Tail15 winner := by
  rw [root155Tail14, scan, if_neg (by decide), root155Candidates14_checked]
  change scan edge (search edge 3) 4 [929] 929 root155Tail15
    (if 3 ≤ root155Candidates14.length then search edge 3 [929, 4973] root155Candidates14 5902 winner else winner) = _
  rw [if_neg (by decide)]
def root155Candidates15 : List Nat := [6563]
set_option maxRecDepth 8192 in
theorem root155Candidates15_checked : candidates edge 4 929 5639 root155Tail16 winner.bound = root155Candidates15 := by
  unfold candidates
  have row : edge 5639 = (SortedNatTree.contains FastRows.row5639) := by
    change Data.row5639.contains = (SortedNatTree.contains FastRows.row5639)
    exact FastRows.row5639_checked.symm
  rw [row]
  decide
theorem root155Step15 : scan edge (search edge 3) 4 [929] 929 root155Tail15 winner =
    scan edge (search edge 3) 4 [929] 929 root155Tail16 winner := by
  rw [root155Tail15, scan, if_neg (by decide), root155Candidates15_checked]
  change scan edge (search edge 3) 4 [929] 929 root155Tail16
    (if 3 ≤ root155Candidates15.length then search edge 3 [929, 5639] root155Candidates15 6568 winner else winner) = _
  rw [if_neg (by decide)]
def root155Candidates16 : List Nat := [12251, 17093]
set_option maxRecDepth 8192 in
theorem root155Candidates16_checked : candidates edge 4 929 5669 root155Tail17 winner.bound = root155Candidates16 := by
  unfold candidates
  have row : edge 5669 = (SortedNatTree.contains FastRows.row5669) := by
    change Data.row5669.contains = (SortedNatTree.contains FastRows.row5669)
    exact FastRows.row5669_checked.symm
  rw [row]
  decide
theorem root155Step16 : scan edge (search edge 3) 4 [929] 929 root155Tail16 winner =
    scan edge (search edge 3) 4 [929] 929 root155Tail17 winner := by
  rw [root155Tail16, scan, if_neg (by decide), root155Candidates16_checked]
  change scan edge (search edge 3) 4 [929] 929 root155Tail17
    (if 3 ≤ root155Candidates16.length then search edge 3 [929, 5669] root155Candidates16 6598 winner else winner) = _
  rw [if_neg (by decide)]
theorem root155Scan17 : scan edge (search edge 3) 4 [929] 929 root155Tail17 winner = winner := by rw [root155Tail17, scan, if_pos (by decide)]
theorem root155Scan16 : scan edge (search edge 3) 4 [929] 929 root155Tail16 winner = winner :=
  (root155Step16).trans root155Scan17
theorem root155Scan15 : scan edge (search edge 3) 4 [929] 929 root155Tail15 winner = winner :=
  (root155Step15).trans root155Scan16
theorem root155Scan14 : scan edge (search edge 3) 4 [929] 929 root155Tail14 winner = winner :=
  (root155Step14).trans root155Scan15
theorem root155Scan13 : scan edge (search edge 3) 4 [929] 929 root155Tail13 winner = winner :=
  (root155Step13).trans root155Scan14
theorem root155Scan12 : scan edge (search edge 3) 4 [929] 929 root155Tail12 winner = winner :=
  (root155Step12).trans root155Scan13
theorem root155Scan11 : scan edge (search edge 3) 4 [929] 929 root155Tail11 winner = winner :=
  (root155Step11).trans root155Scan12
theorem root155Scan10 : scan edge (search edge 3) 4 [929] 929 root155Tail10 winner = winner :=
  (root155Step10).trans root155Scan11
theorem root155Scan9 : scan edge (search edge 3) 4 [929] 929 root155Tail9 winner = winner :=
  (root155Step9).trans root155Scan10
theorem root155Scan8 : scan edge (search edge 3) 4 [929] 929 root155Tail8 winner = winner :=
  (root155Step8).trans root155Scan9
theorem root155Scan7 : scan edge (search edge 3) 4 [929] 929 root155Tail7 winner = winner :=
  (root155Step7).trans root155Scan8
theorem root155Scan6 : scan edge (search edge 3) 4 [929] 929 root155Tail6 winner = winner :=
  (root155Step6).trans root155Scan7
theorem root155Scan5 : scan edge (search edge 3) 4 [929] 929 root155Tail5 winner = winner :=
  (root155Step5).trans root155Scan6
theorem root155Scan4 : scan edge (search edge 3) 4 [929] 929 root155Tail4 winner = winner :=
  (root155Step4).trans root155Scan5
theorem root155Scan3 : scan edge (search edge 3) 4 [929] 929 root155Tail3 winner = winner :=
  (root155Step3).trans root155Scan4
theorem root155Scan2 : scan edge (search edge 3) 4 [929] 929 root155Tail2 winner = winner :=
  (root155Step2).trans root155Scan3
theorem root155Scan1 : scan edge (search edge 3) 4 [929] 929 root155Tail1 winner = winner :=
  (root155Step1).trans root155Scan2
theorem root155Scan0 : scan edge (search edge 3) 4 [929] 929 root155Tail0 winner = winner :=
  (root155Step0).trans root155Scan1
set_option maxRecDepth 4096 in
theorem root155Branch : search edge 4 [929] root155Candidates 929 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [929] 929 root155Tail0 winner = winner
  exact root155Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
