import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root173Candidates : List Nat := Data.row1049
set_option maxRecDepth 8192 in
theorem root173Candidates_checked : candidates edge 5 0 1049 tail174 winner.bound = root173Candidates := by
  apply candidates_eq_row edge 5 0 1049 tail174 root173Candidates winner.bound tail174_ordered
  · decide
  · decide
  · funext q
    rfl
def root173Tail60 : List Nat := []
def root173Tail59 : List Nat := 24923 :: root173Tail60
def root173Tail58 : List Nat := 23957 :: root173Tail59
def root173Tail57 : List Nat := 22859 :: root173Tail58
def root173Tail56 : List Nat := 22613 :: root173Tail57
def root173Tail55 : List Nat := 21851 :: root173Tail56
def root173Tail54 : List Nat := 21521 :: root173Tail55
def root173Tail53 : List Nat := 20807 :: root173Tail54
def root173Tail52 : List Nat := 20441 :: root173Tail53
def root173Tail51 : List Nat := 20327 :: root173Tail52
def root173Tail50 : List Nat := 20297 :: root173Tail51
def root173Tail49 : List Nat := 16433 :: root173Tail50
def root173Tail48 : List Nat := 16421 :: root173Tail49
def root173Tail47 : List Nat := 15887 :: root173Tail48
def root173Tail46 : List Nat := 15809 :: root173Tail47
def root173Tail45 : List Nat := 15329 :: root173Tail46
def root173Tail44 : List Nat := 14741 :: root173Tail45
def root173Tail43 : List Nat := 14543 :: root173Tail44
def root173Tail42 : List Nat := 14081 :: root173Tail43
def root173Tail41 : List Nat := 13997 :: root173Tail42
def root173Tail40 : List Nat := 13037 :: root173Tail41
def root173Tail39 : List Nat := 12473 :: root173Tail40
def root173Tail38 : List Nat := 12011 :: root173Tail39
def root173Tail37 : List Nat := 11987 :: root173Tail38
def root173Tail36 : List Nat := 11969 :: root173Tail37
def root173Tail35 : List Nat := 11927 :: root173Tail36
def root173Tail34 : List Nat := 11789 :: root173Tail35
def root173Tail33 : List Nat := 11171 :: root173Tail34
def root173Tail32 : List Nat := 11057 :: root173Tail33
def root173Tail31 : List Nat := 10601 :: root173Tail32
def root173Tail30 : List Nat := 10037 :: root173Tail31
def root173Tail29 : List Nat := 9623 :: root173Tail30
def root173Tail28 : List Nat := 9413 :: root173Tail29
def root173Tail27 : List Nat := 9323 :: root173Tail28
def root173Tail26 : List Nat := 9029 :: root173Tail27
def root173Tail25 : List Nat := 8669 :: root173Tail26
def root173Tail24 : List Nat := 8219 :: root173Tail25
def root173Tail23 : List Nat := 7841 :: root173Tail24
def root173Tail22 : List Nat := 7607 :: root173Tail23
def root173Tail21 : List Nat := 7577 :: root173Tail22
def root173Tail20 : List Nat := 7559 :: root173Tail21
def root173Tail19 : List Nat := 7523 :: root173Tail20
def root173Tail18 : List Nat := 7229 :: root173Tail19
def root173Tail17 : List Nat := 7013 :: root173Tail18
def root173Tail16 : List Nat := 6971 :: root173Tail17
def root173Tail15 : List Nat := 6203 :: root173Tail16
def root173Tail14 : List Nat := 5897 :: root173Tail15
def root173Tail13 : List Nat := 5393 :: root173Tail14
def root173Tail12 : List Nat := 5297 :: root173Tail13
def root173Tail11 : List Nat := 4787 :: root173Tail12
def root173Tail10 : List Nat := 4493 :: root173Tail11
def root173Tail9 : List Nat := 3851 :: root173Tail10
def root173Tail8 : List Nat := 3701 :: root173Tail9
def root173Tail7 : List Nat := 3617 :: root173Tail8
def root173Tail6 : List Nat := 3449 :: root173Tail7
def root173Tail5 : List Nat := 2633 :: root173Tail6
def root173Tail4 : List Nat := 2297 :: root173Tail5
def root173Tail3 : List Nat := 2087 :: root173Tail4
def root173Tail2 : List Nat := 1847 :: root173Tail3
def root173Tail1 : List Nat := 1181 :: root173Tail2
def root173Tail0 : List Nat := 1109 :: root173Tail1
def root173Candidates0 : List Nat := [1181, 4493]
set_option maxRecDepth 8192 in
theorem root173Candidates0_checked : candidates edge 4 1049 1109 root173Tail1 winner.bound = root173Candidates0 := by
  unfold candidates
  have row : edge 1109 = (SortedNatTree.contains FastRows.row1109) := by
    change Data.row1109.contains = (SortedNatTree.contains FastRows.row1109)
    exact FastRows.row1109_checked.symm
  rw [row]
  decide
theorem root173Step0 : scan edge (search edge 3) 4 [1049] 1049 root173Tail0 winner =
    scan edge (search edge 3) 4 [1049] 1049 root173Tail1 winner := by
  rw [root173Tail0, scan, if_neg (by decide), root173Candidates0_checked]
  change scan edge (search edge 3) 4 [1049] 1049 root173Tail1
    (if 3 ≤ root173Candidates0.length then search edge 3 [1049, 1109] root173Candidates0 2158 winner else winner) = _
  rw [if_neg (by decide)]
def root173Candidates1 : List Nat := [1847, 7577, 8669, 16421]
set_option maxRecDepth 8192 in
theorem root173Candidates1_checked : candidates edge 4 1049 1181 root173Tail2 winner.bound = root173Candidates1 := by
  unfold candidates
  have row : edge 1181 = (SortedNatTree.contains FastRows.row1181) := by
    change Data.row1181.contains = (SortedNatTree.contains FastRows.row1181)
    exact FastRows.row1181_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root173Child1 : search edge 3 [1049, 1181] root173Candidates1 2230 winner = winner := by decide
theorem root173Step1 : scan edge (search edge 3) 4 [1049] 1049 root173Tail1 winner =
    scan edge (search edge 3) 4 [1049] 1049 root173Tail2 winner := by
  rw [root173Tail1, scan, if_neg (by decide), root173Candidates1_checked]
  change scan edge (search edge 3) 4 [1049] 1049 root173Tail2
    (if 3 ≤ root173Candidates1.length then search edge 3 [1049, 1181] root173Candidates1 2230 winner else winner) = _
  rw [if_pos (by decide), root173Child1]
def root173Candidates2 : List Nat := [7523, 21521]
set_option maxRecDepth 8192 in
theorem root173Candidates2_checked : candidates edge 4 1049 1847 root173Tail3 winner.bound = root173Candidates2 := by
  unfold candidates
  have row : edge 1847 = (SortedNatTree.contains FastRows.row1847) := by
    change Data.row1847.contains = (SortedNatTree.contains FastRows.row1847)
    exact FastRows.row1847_checked.symm
  rw [row]
  decide
theorem root173Step2 : scan edge (search edge 3) 4 [1049] 1049 root173Tail2 winner =
    scan edge (search edge 3) 4 [1049] 1049 root173Tail3 winner := by
  rw [root173Tail2, scan, if_neg (by decide), root173Candidates2_checked]
  change scan edge (search edge 3) 4 [1049] 1049 root173Tail3
    (if 3 ≤ root173Candidates2.length then search edge 3 [1049, 1847] root173Candidates2 2896 winner else winner) = _
  rw [if_neg (by decide)]
def root173Candidates3 : List Nat := [7559, 9029, 12011, 14543]
set_option maxRecDepth 8192 in
theorem root173Candidates3_checked : candidates edge 4 1049 2087 root173Tail4 winner.bound = root173Candidates3 := by
  unfold candidates
  have row : edge 2087 = (SortedNatTree.contains FastRows.row2087) := by
    change Data.row2087.contains = (SortedNatTree.contains FastRows.row2087)
    exact FastRows.row2087_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root173Child3 : search edge 3 [1049, 2087] root173Candidates3 3136 winner = winner := by decide
theorem root173Step3 : scan edge (search edge 3) 4 [1049] 1049 root173Tail3 winner =
    scan edge (search edge 3) 4 [1049] 1049 root173Tail4 winner := by
  rw [root173Tail3, scan, if_neg (by decide), root173Candidates3_checked]
  change scan edge (search edge 3) 4 [1049] 1049 root173Tail4
    (if 3 ≤ root173Candidates3.length then search edge 3 [1049, 2087] root173Candidates3 3136 winner else winner) = _
  rw [if_pos (by decide), root173Child3]
def root173Candidates4 : List Nat := [10037, 11171, 11927]
set_option maxRecDepth 8192 in
theorem root173Candidates4_checked : candidates edge 4 1049 2297 root173Tail5 winner.bound = root173Candidates4 := by
  unfold candidates
  have row : edge 2297 = (SortedNatTree.contains FastRows.row2297) := by
    change Data.row2297.contains = (SortedNatTree.contains FastRows.row2297)
    exact FastRows.row2297_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root173Child4 : search edge 3 [1049, 2297] root173Candidates4 3346 winner = winner := by decide
theorem root173Step4 : scan edge (search edge 3) 4 [1049] 1049 root173Tail4 winner =
    scan edge (search edge 3) 4 [1049] 1049 root173Tail5 winner := by
  rw [root173Tail4, scan, if_neg (by decide), root173Candidates4_checked]
  change scan edge (search edge 3) 4 [1049] 1049 root173Tail5
    (if 3 ≤ root173Candidates4.length then search edge 3 [1049, 2297] root173Candidates4 3346 winner else winner) = _
  rw [if_pos (by decide), root173Child4]
def root173Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root173Candidates5_checked : candidates edge 4 1049 2633 root173Tail6 winner.bound = root173Candidates5 := by
  unfold candidates
  have row : edge 2633 = (SortedNatTree.contains FastRows.row2633) := by
    change Data.row2633.contains = (SortedNatTree.contains FastRows.row2633)
    exact FastRows.row2633_checked.symm
  rw [row]
  decide
theorem root173Step5 : scan edge (search edge 3) 4 [1049] 1049 root173Tail5 winner =
    scan edge (search edge 3) 4 [1049] 1049 root173Tail6 winner := by
  rw [root173Tail5, scan, if_neg (by decide), root173Candidates5_checked]
  change scan edge (search edge 3) 4 [1049] 1049 root173Tail6
    (if 3 ≤ root173Candidates5.length then search edge 3 [1049, 2633] root173Candidates5 3682 winner else winner) = _
  rw [if_neg (by decide)]
def root173Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root173Candidates6_checked : candidates edge 4 1049 3449 root173Tail7 winner.bound = root173Candidates6 := by
  unfold candidates
  have row : edge 3449 = (SortedNatTree.contains FastRows.row3449) := by
    change Data.row3449.contains = (SortedNatTree.contains FastRows.row3449)
    exact FastRows.row3449_checked.symm
  rw [row]
  decide
theorem root173Step6 : scan edge (search edge 3) 4 [1049] 1049 root173Tail6 winner =
    scan edge (search edge 3) 4 [1049] 1049 root173Tail7 winner := by
  rw [root173Tail6, scan, if_neg (by decide), root173Candidates6_checked]
  change scan edge (search edge 3) 4 [1049] 1049 root173Tail7
    (if 3 ≤ root173Candidates6.length then search edge 3 [1049, 3449] root173Candidates6 4498 winner else winner) = _
  rw [if_neg (by decide)]
def root173Candidates7 : List Nat := [7607, 9413, 15329]
set_option maxRecDepth 8192 in
theorem root173Candidates7_checked : candidates edge 4 1049 3617 root173Tail8 winner.bound = root173Candidates7 := by
  unfold candidates
  have row : edge 3617 = (SortedNatTree.contains FastRows.row3617) := by
    change Data.row3617.contains = (SortedNatTree.contains FastRows.row3617)
    exact FastRows.row3617_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root173Child7 : search edge 3 [1049, 3617] root173Candidates7 4666 winner = winner := by decide
theorem root173Step7 : scan edge (search edge 3) 4 [1049] 1049 root173Tail7 winner =
    scan edge (search edge 3) 4 [1049] 1049 root173Tail8 winner := by
  rw [root173Tail7, scan, if_neg (by decide), root173Candidates7_checked]
  change scan edge (search edge 3) 4 [1049] 1049 root173Tail8
    (if 3 ≤ root173Candidates7.length then search edge 3 [1049, 3617] root173Candidates7 4666 winner else winner) = _
  rw [if_pos (by decide), root173Child7]
def root173Candidates8 : List Nat := [4493, 9623]
set_option maxRecDepth 8192 in
theorem root173Candidates8_checked : candidates edge 4 1049 3701 root173Tail9 winner.bound = root173Candidates8 := by
  unfold candidates
  have row : edge 3701 = (SortedNatTree.contains FastRows.row3701) := by
    change Data.row3701.contains = (SortedNatTree.contains FastRows.row3701)
    exact FastRows.row3701_checked.symm
  rw [row]
  decide
theorem root173Step8 : scan edge (search edge 3) 4 [1049] 1049 root173Tail8 winner =
    scan edge (search edge 3) 4 [1049] 1049 root173Tail9 winner := by
  rw [root173Tail8, scan, if_neg (by decide), root173Candidates8_checked]
  change scan edge (search edge 3) 4 [1049] 1049 root173Tail9
    (if 3 ≤ root173Candidates8.length then search edge 3 [1049, 3701] root173Candidates8 4750 winner else winner) = _
  rw [if_neg (by decide)]
def root173Candidates9 : List Nat := [13037]
set_option maxRecDepth 8192 in
theorem root173Candidates9_checked : candidates edge 4 1049 3851 root173Tail10 winner.bound = root173Candidates9 := by
  unfold candidates
  have row : edge 3851 = (SortedNatTree.contains FastRows.row3851) := by
    change Data.row3851.contains = (SortedNatTree.contains FastRows.row3851)
    exact FastRows.row3851_checked.symm
  rw [row]
  decide
theorem root173Step9 : scan edge (search edge 3) 4 [1049] 1049 root173Tail9 winner =
    scan edge (search edge 3) 4 [1049] 1049 root173Tail10 winner := by
  rw [root173Tail9, scan, if_neg (by decide), root173Candidates9_checked]
  change scan edge (search edge 3) 4 [1049] 1049 root173Tail10
    (if 3 ≤ root173Candidates9.length then search edge 3 [1049, 3851] root173Candidates9 4900 winner else winner) = _
  rw [if_neg (by decide)]
def root173Candidates10 : List Nat := [14741, 20297]
set_option maxRecDepth 8192 in
theorem root173Candidates10_checked : candidates edge 4 1049 4493 root173Tail11 winner.bound = root173Candidates10 := by
  unfold candidates
  have row : edge 4493 = (SortedNatTree.contains FastRows.row4493) := by
    change Data.row4493.contains = (SortedNatTree.contains FastRows.row4493)
    exact FastRows.row4493_checked.symm
  rw [row]
  decide
theorem root173Step10 : scan edge (search edge 3) 4 [1049] 1049 root173Tail10 winner =
    scan edge (search edge 3) 4 [1049] 1049 root173Tail11 winner := by
  rw [root173Tail10, scan, if_neg (by decide), root173Candidates10_checked]
  change scan edge (search edge 3) 4 [1049] 1049 root173Tail11
    (if 3 ≤ root173Candidates10.length then search edge 3 [1049, 4493] root173Candidates10 5542 winner else winner) = _
  rw [if_neg (by decide)]
def root173Candidates11 : List Nat := [7013, 7229]
set_option maxRecDepth 8192 in
theorem root173Candidates11_checked : candidates edge 4 1049 4787 root173Tail12 winner.bound = root173Candidates11 := by
  unfold candidates
  have row : edge 4787 = (SortedNatTree.contains FastRows.row4787) := by
    change Data.row4787.contains = (SortedNatTree.contains FastRows.row4787)
    exact FastRows.row4787_checked.symm
  rw [row]
  decide
theorem root173Step11 : scan edge (search edge 3) 4 [1049] 1049 root173Tail11 winner =
    scan edge (search edge 3) 4 [1049] 1049 root173Tail12 winner := by
  rw [root173Tail11, scan, if_neg (by decide), root173Candidates11_checked]
  change scan edge (search edge 3) 4 [1049] 1049 root173Tail12
    (if 3 ≤ root173Candidates11.length then search edge 3 [1049, 4787] root173Candidates11 5836 winner else winner) = _
  rw [if_neg (by decide)]
def root173Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root173Candidates12_checked : candidates edge 4 1049 5297 root173Tail13 winner.bound = root173Candidates12 := by
  unfold candidates
  have row : edge 5297 = (SortedNatTree.contains FastRows.row5297) := by
    change Data.row5297.contains = (SortedNatTree.contains FastRows.row5297)
    exact FastRows.row5297_checked.symm
  rw [row]
  decide
theorem root173Step12 : scan edge (search edge 3) 4 [1049] 1049 root173Tail12 winner =
    scan edge (search edge 3) 4 [1049] 1049 root173Tail13 winner := by
  rw [root173Tail12, scan, if_neg (by decide), root173Candidates12_checked]
  change scan edge (search edge 3) 4 [1049] 1049 root173Tail13
    (if 3 ≤ root173Candidates12.length then search edge 3 [1049, 5297] root173Candidates12 6346 winner else winner) = _
  rw [if_neg (by decide)]
def root173Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root173Candidates13_checked : candidates edge 4 1049 5393 root173Tail14 winner.bound = root173Candidates13 := by
  unfold candidates
  have row : edge 5393 = (SortedNatTree.contains FastRows.row5393) := by
    change Data.row5393.contains = (SortedNatTree.contains FastRows.row5393)
    exact FastRows.row5393_checked.symm
  rw [row]
  decide
theorem root173Step13 : scan edge (search edge 3) 4 [1049] 1049 root173Tail13 winner =
    scan edge (search edge 3) 4 [1049] 1049 root173Tail14 winner := by
  rw [root173Tail13, scan, if_neg (by decide), root173Candidates13_checked]
  change scan edge (search edge 3) 4 [1049] 1049 root173Tail14
    (if 3 ≤ root173Candidates13.length then search edge 3 [1049, 5393] root173Candidates13 6442 winner else winner) = _
  rw [if_neg (by decide)]
def root173Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root173Candidates14_checked : candidates edge 4 1049 5897 root173Tail15 winner.bound = root173Candidates14 := by
  unfold candidates
  have row : edge 5897 = (SortedNatTree.contains FastRows.row5897) := by
    change Data.row5897.contains = (SortedNatTree.contains FastRows.row5897)
    exact FastRows.row5897_checked.symm
  rw [row]
  decide
theorem root173Step14 : scan edge (search edge 3) 4 [1049] 1049 root173Tail14 winner =
    scan edge (search edge 3) 4 [1049] 1049 root173Tail15 winner := by
  rw [root173Tail14, scan, if_neg (by decide), root173Candidates14_checked]
  change scan edge (search edge 3) 4 [1049] 1049 root173Tail15
    (if 3 ≤ root173Candidates14.length then search edge 3 [1049, 5897] root173Candidates14 6946 winner else winner) = _
  rw [if_neg (by decide)]
def root173Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root173Candidates15_checked : candidates edge 4 1049 6203 root173Tail16 winner.bound = root173Candidates15 := by
  unfold candidates
  have row : edge 6203 = (SortedNatTree.contains FastRows.row6203) := by
    change Data.row6203.contains = (SortedNatTree.contains FastRows.row6203)
    exact FastRows.row6203_checked.symm
  rw [row]
  decide
theorem root173Step15 : scan edge (search edge 3) 4 [1049] 1049 root173Tail15 winner =
    scan edge (search edge 3) 4 [1049] 1049 root173Tail16 winner := by
  rw [root173Tail15, scan, if_neg (by decide), root173Candidates15_checked]
  change scan edge (search edge 3) 4 [1049] 1049 root173Tail16
    (if 3 ≤ root173Candidates15.length then search edge 3 [1049, 6203] root173Candidates15 7252 winner else winner) = _
  rw [if_neg (by decide)]
theorem root173Scan16 : scan edge (search edge 3) 4 [1049] 1049 root173Tail16 winner = winner := by rw [root173Tail16, scan, if_pos (by decide)]
theorem root173Scan15 : scan edge (search edge 3) 4 [1049] 1049 root173Tail15 winner = winner :=
  (root173Step15).trans root173Scan16
theorem root173Scan14 : scan edge (search edge 3) 4 [1049] 1049 root173Tail14 winner = winner :=
  (root173Step14).trans root173Scan15
theorem root173Scan13 : scan edge (search edge 3) 4 [1049] 1049 root173Tail13 winner = winner :=
  (root173Step13).trans root173Scan14
theorem root173Scan12 : scan edge (search edge 3) 4 [1049] 1049 root173Tail12 winner = winner :=
  (root173Step12).trans root173Scan13
theorem root173Scan11 : scan edge (search edge 3) 4 [1049] 1049 root173Tail11 winner = winner :=
  (root173Step11).trans root173Scan12
theorem root173Scan10 : scan edge (search edge 3) 4 [1049] 1049 root173Tail10 winner = winner :=
  (root173Step10).trans root173Scan11
theorem root173Scan9 : scan edge (search edge 3) 4 [1049] 1049 root173Tail9 winner = winner :=
  (root173Step9).trans root173Scan10
theorem root173Scan8 : scan edge (search edge 3) 4 [1049] 1049 root173Tail8 winner = winner :=
  (root173Step8).trans root173Scan9
theorem root173Scan7 : scan edge (search edge 3) 4 [1049] 1049 root173Tail7 winner = winner :=
  (root173Step7).trans root173Scan8
theorem root173Scan6 : scan edge (search edge 3) 4 [1049] 1049 root173Tail6 winner = winner :=
  (root173Step6).trans root173Scan7
theorem root173Scan5 : scan edge (search edge 3) 4 [1049] 1049 root173Tail5 winner = winner :=
  (root173Step5).trans root173Scan6
theorem root173Scan4 : scan edge (search edge 3) 4 [1049] 1049 root173Tail4 winner = winner :=
  (root173Step4).trans root173Scan5
theorem root173Scan3 : scan edge (search edge 3) 4 [1049] 1049 root173Tail3 winner = winner :=
  (root173Step3).trans root173Scan4
theorem root173Scan2 : scan edge (search edge 3) 4 [1049] 1049 root173Tail2 winner = winner :=
  (root173Step2).trans root173Scan3
theorem root173Scan1 : scan edge (search edge 3) 4 [1049] 1049 root173Tail1 winner = winner :=
  (root173Step1).trans root173Scan2
theorem root173Scan0 : scan edge (search edge 3) 4 [1049] 1049 root173Tail0 winner = winner :=
  (root173Step0).trans root173Scan1
set_option maxRecDepth 4096 in
theorem root173Branch : search edge 4 [1049] root173Candidates 1049 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1049] 1049 root173Tail0 winner = winner
  exact root173Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
