import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root176Candidates : List Nat := Data.row1063
set_option maxRecDepth 8192 in
theorem root176Candidates_checked : candidates edge 5 0 1063 tail177 winner.bound = root176Candidates := by
  apply candidates_eq_row edge 5 0 1063 tail177 root176Candidates winner.bound tail177_ordered
  · decide
  · decide
  · funext q
    rfl
def root176Tail52 : List Nat := []
def root176Tail51 : List Nat := 24007 :: root176Tail52
def root176Tail50 : List Nat := 23197 :: root176Tail51
def root176Tail49 : List Nat := 21937 :: root176Tail50
def root176Tail48 : List Nat := 21013 :: root176Tail49
def root176Tail47 : List Nat := 20857 :: root176Tail48
def root176Tail46 : List Nat := 20353 :: root176Tail47
def root176Tail45 : List Nat := 20107 :: root176Tail46
def root176Tail44 : List Nat := 19963 :: root176Tail45
def root176Tail43 : List Nat := 19753 :: root176Tail44
def root176Tail42 : List Nat := 19699 :: root176Tail43
def root176Tail41 : List Nat := 19417 :: root176Tail42
def root176Tail40 : List Nat := 19009 :: root176Tail41
def root176Tail39 : List Nat := 18061 :: root176Tail40
def root176Tail38 : List Nat := 17737 :: root176Tail39
def root176Tail37 : List Nat := 17581 :: root176Tail38
def root176Tail36 : List Nat := 16843 :: root176Tail37
def root176Tail35 : List Nat := 16447 :: root176Tail36
def root176Tail34 : List Nat := 15271 :: root176Tail35
def root176Tail33 : List Nat := 14851 :: root176Tail34
def root176Tail32 : List Nat := 14479 :: root176Tail33
def root176Tail31 : List Nat := 14389 :: root176Tail32
def root176Tail30 : List Nat := 13093 :: root176Tail31
def root176Tail29 : List Nat := 11329 :: root176Tail30
def root176Tail28 : List Nat := 10753 :: root176Tail29
def root176Tail27 : List Nat := 9931 :: root176Tail28
def root176Tail26 : List Nat := 9631 :: root176Tail27
def root176Tail25 : List Nat := 9463 :: root176Tail26
def root176Tail24 : List Nat := 9397 :: root176Tail25
def root176Tail23 : List Nat := 9241 :: root176Tail24
def root176Tail22 : List Nat := 9157 :: root176Tail23
def root176Tail21 : List Nat := 8539 :: root176Tail22
def root176Tail20 : List Nat := 7507 :: root176Tail21
def root176Tail19 : List Nat := 7213 :: root176Tail20
def root176Tail18 : List Nat := 6793 :: root176Tail19
def root176Tail17 : List Nat := 6247 :: root176Tail18
def root176Tail16 : List Nat := 5701 :: root176Tail17
def root176Tail15 : List Nat := 5197 :: root176Tail16
def root176Tail14 : List Nat := 4339 :: root176Tail15
def root176Tail13 : List Nat := 3919 :: root176Tail14
def root176Tail12 : List Nat := 3547 :: root176Tail13
def root176Tail11 : List Nat := 3499 :: root176Tail12
def root176Tail10 : List Nat := 3463 :: root176Tail11
def root176Tail9 : List Nat := 3307 :: root176Tail10
def root176Tail8 : List Nat := 3067 :: root176Tail9
def root176Tail7 : List Nat := 3001 :: root176Tail8
def root176Tail6 : List Nat := 2953 :: root176Tail7
def root176Tail5 : List Nat := 2749 :: root176Tail6
def root176Tail4 : List Nat := 2719 :: root176Tail5
def root176Tail3 : List Nat := 2551 :: root176Tail4
def root176Tail2 : List Nat := 1609 :: root176Tail3
def root176Tail1 : List Nat := 1531 :: root176Tail2
def root176Tail0 : List Nat := 1471 :: root176Tail1
def root176Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root176Candidates0_checked : candidates edge 4 1063 1471 root176Tail1 winner.bound = root176Candidates0 := by
  unfold candidates
  have row : edge 1471 = (SortedNatTree.contains FastRows.row1471) := by
    change Data.row1471.contains = (SortedNatTree.contains FastRows.row1471)
    exact FastRows.row1471_checked.symm
  rw [row]
  decide
theorem root176Step0 : scan edge (search edge 3) 4 [1063] 1063 root176Tail0 winner =
    scan edge (search edge 3) 4 [1063] 1063 root176Tail1 winner := by
  rw [root176Tail0, scan, if_neg (by decide), root176Candidates0_checked]
  change scan edge (search edge 3) 4 [1063] 1063 root176Tail1
    (if 3 ≤ root176Candidates0.length then search edge 3 [1063, 1471] root176Candidates0 2534 winner else winner) = _
  rw [if_neg (by decide)]
def root176Candidates1 : List Nat := [7213, 9463, 19753]
set_option maxRecDepth 8192 in
theorem root176Candidates1_checked : candidates edge 4 1063 1531 root176Tail2 winner.bound = root176Candidates1 := by
  unfold candidates
  have row : edge 1531 = (SortedNatTree.contains FastRows.row1531) := by
    change Data.row1531.contains = (SortedNatTree.contains FastRows.row1531)
    exact FastRows.row1531_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root176Child1 : search edge 3 [1063, 1531] root176Candidates1 2594 winner = winner := by decide
theorem root176Step1 : scan edge (search edge 3) 4 [1063] 1063 root176Tail1 winner =
    scan edge (search edge 3) 4 [1063] 1063 root176Tail2 winner := by
  rw [root176Tail1, scan, if_neg (by decide), root176Candidates1_checked]
  change scan edge (search edge 3) 4 [1063] 1063 root176Tail2
    (if 3 ≤ root176Candidates1.length then search edge 3 [1063, 1531] root176Candidates1 2594 winner else winner) = _
  rw [if_pos (by decide), root176Child1]
def root176Candidates2 : List Nat := [16843, 19009]
set_option maxRecDepth 8192 in
theorem root176Candidates2_checked : candidates edge 4 1063 1609 root176Tail3 winner.bound = root176Candidates2 := by
  unfold candidates
  have row : edge 1609 = (SortedNatTree.contains FastRows.row1609) := by
    change Data.row1609.contains = (SortedNatTree.contains FastRows.row1609)
    exact FastRows.row1609_checked.symm
  rw [row]
  decide
theorem root176Step2 : scan edge (search edge 3) 4 [1063] 1063 root176Tail2 winner =
    scan edge (search edge 3) 4 [1063] 1063 root176Tail3 winner := by
  rw [root176Tail2, scan, if_neg (by decide), root176Candidates2_checked]
  change scan edge (search edge 3) 4 [1063] 1063 root176Tail3
    (if 3 ≤ root176Candidates2.length then search edge 3 [1063, 1609] root176Candidates2 2672 winner else winner) = _
  rw [if_neg (by decide)]
def root176Candidates3 : List Nat := [6793, 9631]
set_option maxRecDepth 8192 in
theorem root176Candidates3_checked : candidates edge 4 1063 2551 root176Tail4 winner.bound = root176Candidates3 := by
  unfold candidates
  have row : edge 2551 = (SortedNatTree.contains FastRows.row2551) := by
    change Data.row2551.contains = (SortedNatTree.contains FastRows.row2551)
    exact FastRows.row2551_checked.symm
  rw [row]
  decide
theorem root176Step3 : scan edge (search edge 3) 4 [1063] 1063 root176Tail3 winner =
    scan edge (search edge 3) 4 [1063] 1063 root176Tail4 winner := by
  rw [root176Tail3, scan, if_neg (by decide), root176Candidates3_checked]
  change scan edge (search edge 3) 4 [1063] 1063 root176Tail4
    (if 3 ≤ root176Candidates3.length then search edge 3 [1063, 2551] root176Candidates3 3614 winner else winner) = _
  rw [if_neg (by decide)]
def root176Candidates4 : List Nat := [9631, 14479]
set_option maxRecDepth 8192 in
theorem root176Candidates4_checked : candidates edge 4 1063 2719 root176Tail5 winner.bound = root176Candidates4 := by
  unfold candidates
  have row : edge 2719 = (SortedNatTree.contains FastRows.row2719) := by
    change Data.row2719.contains = (SortedNatTree.contains FastRows.row2719)
    exact FastRows.row2719_checked.symm
  rw [row]
  decide
theorem root176Step4 : scan edge (search edge 3) 4 [1063] 1063 root176Tail4 winner =
    scan edge (search edge 3) 4 [1063] 1063 root176Tail5 winner := by
  rw [root176Tail4, scan, if_neg (by decide), root176Candidates4_checked]
  change scan edge (search edge 3) 4 [1063] 1063 root176Tail5
    (if 3 ≤ root176Candidates4.length then search edge 3 [1063, 2719] root176Candidates4 3782 winner else winner) = _
  rw [if_neg (by decide)]
def root176Candidates5 : List Nat := [19963]
set_option maxRecDepth 8192 in
theorem root176Candidates5_checked : candidates edge 4 1063 2749 root176Tail6 winner.bound = root176Candidates5 := by
  unfold candidates
  have row : edge 2749 = (SortedNatTree.contains FastRows.row2749) := by
    change Data.row2749.contains = (SortedNatTree.contains FastRows.row2749)
    exact FastRows.row2749_checked.symm
  rw [row]
  decide
theorem root176Step5 : scan edge (search edge 3) 4 [1063] 1063 root176Tail5 winner =
    scan edge (search edge 3) 4 [1063] 1063 root176Tail6 winner := by
  rw [root176Tail5, scan, if_neg (by decide), root176Candidates5_checked]
  change scan edge (search edge 3) 4 [1063] 1063 root176Tail6
    (if 3 ≤ root176Candidates5.length then search edge 3 [1063, 2749] root176Candidates5 3812 winner else winner) = _
  rw [if_neg (by decide)]
def root176Candidates6 : List Nat := [7213]
set_option maxRecDepth 8192 in
theorem root176Candidates6_checked : candidates edge 4 1063 2953 root176Tail7 winner.bound = root176Candidates6 := by
  unfold candidates
  have row : edge 2953 = (SortedNatTree.contains FastRows.row2953) := by
    change Data.row2953.contains = (SortedNatTree.contains FastRows.row2953)
    exact FastRows.row2953_checked.symm
  rw [row]
  decide
theorem root176Step6 : scan edge (search edge 3) 4 [1063] 1063 root176Tail6 winner =
    scan edge (search edge 3) 4 [1063] 1063 root176Tail7 winner := by
  rw [root176Tail6, scan, if_neg (by decide), root176Candidates6_checked]
  change scan edge (search edge 3) 4 [1063] 1063 root176Tail7
    (if 3 ≤ root176Candidates6.length then search edge 3 [1063, 2953] root176Candidates6 4016 winner else winner) = _
  rw [if_neg (by decide)]
def root176Candidates7 : List Nat := [3499, 9931, 10753]
set_option maxRecDepth 8192 in
theorem root176Candidates7_checked : candidates edge 4 1063 3001 root176Tail8 winner.bound = root176Candidates7 := by
  unfold candidates
  have row : edge 3001 = (SortedNatTree.contains FastRows.row3001) := by
    change Data.row3001.contains = (SortedNatTree.contains FastRows.row3001)
    exact FastRows.row3001_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root176Child7 : search edge 3 [1063, 3001] root176Candidates7 4064 winner = winner := by decide
theorem root176Step7 : scan edge (search edge 3) 4 [1063] 1063 root176Tail7 winner =
    scan edge (search edge 3) 4 [1063] 1063 root176Tail8 winner := by
  rw [root176Tail7, scan, if_neg (by decide), root176Candidates7_checked]
  change scan edge (search edge 3) 4 [1063] 1063 root176Tail8
    (if 3 ≤ root176Candidates7.length then search edge 3 [1063, 3001] root176Candidates7 4064 winner else winner) = _
  rw [if_pos (by decide), root176Child7]
def root176Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root176Candidates8_checked : candidates edge 4 1063 3067 root176Tail9 winner.bound = root176Candidates8 := by
  unfold candidates
  have row : edge 3067 = (SortedNatTree.contains FastRows.row3067) := by
    change Data.row3067.contains = (SortedNatTree.contains FastRows.row3067)
    exact FastRows.row3067_checked.symm
  rw [row]
  decide
theorem root176Step8 : scan edge (search edge 3) 4 [1063] 1063 root176Tail8 winner =
    scan edge (search edge 3) 4 [1063] 1063 root176Tail9 winner := by
  rw [root176Tail8, scan, if_neg (by decide), root176Candidates8_checked]
  change scan edge (search edge 3) 4 [1063] 1063 root176Tail9
    (if 3 ≤ root176Candidates8.length then search edge 3 [1063, 3067] root176Candidates8 4130 winner else winner) = _
  rw [if_neg (by decide)]
def root176Candidates9 : List Nat := [14389]
set_option maxRecDepth 8192 in
theorem root176Candidates9_checked : candidates edge 4 1063 3307 root176Tail10 winner.bound = root176Candidates9 := by
  unfold candidates
  have row : edge 3307 = (SortedNatTree.contains FastRows.row3307) := by
    change Data.row3307.contains = (SortedNatTree.contains FastRows.row3307)
    exact FastRows.row3307_checked.symm
  rw [row]
  decide
theorem root176Step9 : scan edge (search edge 3) 4 [1063] 1063 root176Tail9 winner =
    scan edge (search edge 3) 4 [1063] 1063 root176Tail10 winner := by
  rw [root176Tail9, scan, if_neg (by decide), root176Candidates9_checked]
  change scan edge (search edge 3) 4 [1063] 1063 root176Tail10
    (if 3 ≤ root176Candidates9.length then search edge 3 [1063, 3307] root176Candidates9 4370 winner else winner) = _
  rw [if_neg (by decide)]
def root176Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root176Candidates10_checked : candidates edge 4 1063 3463 root176Tail11 winner.bound = root176Candidates10 := by
  unfold candidates
  have row : edge 3463 = (SortedNatTree.contains FastRows.row3463) := by
    change Data.row3463.contains = (SortedNatTree.contains FastRows.row3463)
    exact FastRows.row3463_checked.symm
  rw [row]
  decide
theorem root176Step10 : scan edge (search edge 3) 4 [1063] 1063 root176Tail10 winner =
    scan edge (search edge 3) 4 [1063] 1063 root176Tail11 winner := by
  rw [root176Tail10, scan, if_neg (by decide), root176Candidates10_checked]
  change scan edge (search edge 3) 4 [1063] 1063 root176Tail11
    (if 3 ≤ root176Candidates10.length then search edge 3 [1063, 3463] root176Candidates10 4526 winner else winner) = _
  rw [if_neg (by decide)]
def root176Candidates11 : List Nat := [9157, 9931]
set_option maxRecDepth 8192 in
theorem root176Candidates11_checked : candidates edge 4 1063 3499 root176Tail12 winner.bound = root176Candidates11 := by
  unfold candidates
  have row : edge 3499 = (SortedNatTree.contains FastRows.row3499) := by
    change Data.row3499.contains = (SortedNatTree.contains FastRows.row3499)
    exact FastRows.row3499_checked.symm
  rw [row]
  decide
theorem root176Step11 : scan edge (search edge 3) 4 [1063] 1063 root176Tail11 winner =
    scan edge (search edge 3) 4 [1063] 1063 root176Tail12 winner := by
  rw [root176Tail11, scan, if_neg (by decide), root176Candidates11_checked]
  change scan edge (search edge 3) 4 [1063] 1063 root176Tail12
    (if 3 ≤ root176Candidates11.length then search edge 3 [1063, 3499] root176Candidates11 4562 winner else winner) = _
  rw [if_neg (by decide)]
def root176Candidates12 : List Nat := [7213, 18061]
set_option maxRecDepth 8192 in
theorem root176Candidates12_checked : candidates edge 4 1063 3547 root176Tail13 winner.bound = root176Candidates12 := by
  unfold candidates
  have row : edge 3547 = (SortedNatTree.contains FastRows.row3547) := by
    change Data.row3547.contains = (SortedNatTree.contains FastRows.row3547)
    exact FastRows.row3547_checked.symm
  rw [row]
  decide
theorem root176Step12 : scan edge (search edge 3) 4 [1063] 1063 root176Tail12 winner =
    scan edge (search edge 3) 4 [1063] 1063 root176Tail13 winner := by
  rw [root176Tail12, scan, if_neg (by decide), root176Candidates12_checked]
  change scan edge (search edge 3) 4 [1063] 1063 root176Tail13
    (if 3 ≤ root176Candidates12.length then search edge 3 [1063, 3547] root176Candidates12 4610 winner else winner) = _
  rw [if_neg (by decide)]
def root176Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root176Candidates13_checked : candidates edge 4 1063 3919 root176Tail14 winner.bound = root176Candidates13 := by
  unfold candidates
  have row : edge 3919 = (SortedNatTree.contains FastRows.row3919) := by
    change Data.row3919.contains = (SortedNatTree.contains FastRows.row3919)
    exact FastRows.row3919_checked.symm
  rw [row]
  decide
theorem root176Step13 : scan edge (search edge 3) 4 [1063] 1063 root176Tail13 winner =
    scan edge (search edge 3) 4 [1063] 1063 root176Tail14 winner := by
  rw [root176Tail13, scan, if_neg (by decide), root176Candidates13_checked]
  change scan edge (search edge 3) 4 [1063] 1063 root176Tail14
    (if 3 ≤ root176Candidates13.length then search edge 3 [1063, 3919] root176Candidates13 4982 winner else winner) = _
  rw [if_neg (by decide)]
def root176Candidates14 : List Nat := [5701, 9241, 9931, 11329, 14479, 19417]
set_option maxRecDepth 8192 in
theorem root176Candidates14_checked : candidates edge 4 1063 4339 root176Tail15 winner.bound = root176Candidates14 := by
  unfold candidates
  have row : edge 4339 = (SortedNatTree.contains FastRows.row4339) := by
    change Data.row4339.contains = (SortedNatTree.contains FastRows.row4339)
    exact FastRows.row4339_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root176Child14 : search edge 3 [1063, 4339] root176Candidates14 5402 winner = winner := by decide
theorem root176Step14 : scan edge (search edge 3) 4 [1063] 1063 root176Tail14 winner =
    scan edge (search edge 3) 4 [1063] 1063 root176Tail15 winner := by
  rw [root176Tail14, scan, if_neg (by decide), root176Candidates14_checked]
  change scan edge (search edge 3) 4 [1063] 1063 root176Tail15
    (if 3 ≤ root176Candidates14.length then search edge 3 [1063, 4339] root176Candidates14 5402 winner else winner) = _
  rw [if_pos (by decide), root176Child14]
def root176Candidates15 : List Nat := [5701]
set_option maxRecDepth 8192 in
theorem root176Candidates15_checked : candidates edge 4 1063 5197 root176Tail16 winner.bound = root176Candidates15 := by
  unfold candidates
  have row : edge 5197 = (SortedNatTree.contains FastRows.row5197) := by
    change Data.row5197.contains = (SortedNatTree.contains FastRows.row5197)
    exact FastRows.row5197_checked.symm
  rw [row]
  decide
theorem root176Step15 : scan edge (search edge 3) 4 [1063] 1063 root176Tail15 winner =
    scan edge (search edge 3) 4 [1063] 1063 root176Tail16 winner := by
  rw [root176Tail15, scan, if_neg (by decide), root176Candidates15_checked]
  change scan edge (search edge 3) 4 [1063] 1063 root176Tail16
    (if 3 ≤ root176Candidates15.length then search edge 3 [1063, 5197] root176Candidates15 6260 winner else winner) = _
  rw [if_neg (by decide)]
def root176Candidates16 : List Nat := [14851]
set_option maxRecDepth 8192 in
theorem root176Candidates16_checked : candidates edge 4 1063 5701 root176Tail17 winner.bound = root176Candidates16 := by
  unfold candidates
  have row : edge 5701 = (SortedNatTree.contains FastRows.row5701) := by
    change Data.row5701.contains = (SortedNatTree.contains FastRows.row5701)
    exact FastRows.row5701_checked.symm
  rw [row]
  decide
theorem root176Step16 : scan edge (search edge 3) 4 [1063] 1063 root176Tail16 winner =
    scan edge (search edge 3) 4 [1063] 1063 root176Tail17 winner := by
  rw [root176Tail16, scan, if_neg (by decide), root176Candidates16_checked]
  change scan edge (search edge 3) 4 [1063] 1063 root176Tail17
    (if 3 ≤ root176Candidates16.length then search edge 3 [1063, 5701] root176Candidates16 6764 winner else winner) = _
  rw [if_neg (by decide)]
theorem root176Scan17 : scan edge (search edge 3) 4 [1063] 1063 root176Tail17 winner = winner := by rw [root176Tail17, scan, if_pos (by decide)]
theorem root176Scan16 : scan edge (search edge 3) 4 [1063] 1063 root176Tail16 winner = winner :=
  (root176Step16).trans root176Scan17
theorem root176Scan15 : scan edge (search edge 3) 4 [1063] 1063 root176Tail15 winner = winner :=
  (root176Step15).trans root176Scan16
theorem root176Scan14 : scan edge (search edge 3) 4 [1063] 1063 root176Tail14 winner = winner :=
  (root176Step14).trans root176Scan15
theorem root176Scan13 : scan edge (search edge 3) 4 [1063] 1063 root176Tail13 winner = winner :=
  (root176Step13).trans root176Scan14
theorem root176Scan12 : scan edge (search edge 3) 4 [1063] 1063 root176Tail12 winner = winner :=
  (root176Step12).trans root176Scan13
theorem root176Scan11 : scan edge (search edge 3) 4 [1063] 1063 root176Tail11 winner = winner :=
  (root176Step11).trans root176Scan12
theorem root176Scan10 : scan edge (search edge 3) 4 [1063] 1063 root176Tail10 winner = winner :=
  (root176Step10).trans root176Scan11
theorem root176Scan9 : scan edge (search edge 3) 4 [1063] 1063 root176Tail9 winner = winner :=
  (root176Step9).trans root176Scan10
theorem root176Scan8 : scan edge (search edge 3) 4 [1063] 1063 root176Tail8 winner = winner :=
  (root176Step8).trans root176Scan9
theorem root176Scan7 : scan edge (search edge 3) 4 [1063] 1063 root176Tail7 winner = winner :=
  (root176Step7).trans root176Scan8
theorem root176Scan6 : scan edge (search edge 3) 4 [1063] 1063 root176Tail6 winner = winner :=
  (root176Step6).trans root176Scan7
theorem root176Scan5 : scan edge (search edge 3) 4 [1063] 1063 root176Tail5 winner = winner :=
  (root176Step5).trans root176Scan6
theorem root176Scan4 : scan edge (search edge 3) 4 [1063] 1063 root176Tail4 winner = winner :=
  (root176Step4).trans root176Scan5
theorem root176Scan3 : scan edge (search edge 3) 4 [1063] 1063 root176Tail3 winner = winner :=
  (root176Step3).trans root176Scan4
theorem root176Scan2 : scan edge (search edge 3) 4 [1063] 1063 root176Tail2 winner = winner :=
  (root176Step2).trans root176Scan3
theorem root176Scan1 : scan edge (search edge 3) 4 [1063] 1063 root176Tail1 winner = winner :=
  (root176Step1).trans root176Scan2
theorem root176Scan0 : scan edge (search edge 3) 4 [1063] 1063 root176Tail0 winner = winner :=
  (root176Step0).trans root176Scan1
set_option maxRecDepth 4096 in
theorem root176Branch : search edge 4 [1063] root176Candidates 1063 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1063] 1063 root176Tail0 winner = winner
  exact root176Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
