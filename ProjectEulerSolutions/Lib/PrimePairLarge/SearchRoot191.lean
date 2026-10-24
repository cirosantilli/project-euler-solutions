import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root191Candidates : List Nat := Data.row1181
set_option maxRecDepth 8192 in
theorem root191Candidates_checked : candidates edge 5 0 1181 tail192 winner.bound = root191Candidates := by
  apply candidates_eq_row edge 5 0 1181 tail192 root191Candidates winner.bound tail192_ordered
  · decide
  · decide
  · funext q
    rfl
def root191Tail59 : List Nat := []
def root191Tail58 : List Nat := 24821 :: root191Tail59
def root191Tail57 : List Nat := 24407 :: root191Tail58
def root191Tail56 : List Nat := 23567 :: root191Tail57
def root191Tail55 : List Nat := 23333 :: root191Tail56
def root191Tail54 : List Nat := 22811 :: root191Tail55
def root191Tail53 : List Nat := 22709 :: root191Tail54
def root191Tail52 : List Nat := 21929 :: root191Tail53
def root191Tail51 : List Nat := 21881 :: root191Tail52
def root191Tail50 : List Nat := 21863 :: root191Tail51
def root191Tail49 : List Nat := 21701 :: root191Tail50
def root191Tail48 : List Nat := 21587 :: root191Tail49
def root191Tail47 : List Nat := 21407 :: root191Tail48
def root191Tail46 : List Nat := 21323 :: root191Tail47
def root191Tail45 : List Nat := 20627 :: root191Tail46
def root191Tail44 : List Nat := 18149 :: root191Tail45
def root191Tail43 : List Nat := 17327 :: root191Tail44
def root191Tail42 : List Nat := 17159 :: root191Tail43
def root191Tail41 : List Nat := 16763 :: root191Tail42
def root191Tail40 : List Nat := 16451 :: root191Tail41
def root191Tail39 : List Nat := 16421 :: root191Tail40
def root191Tail38 : List Nat := 15161 :: root191Tail39
def root191Tail37 : List Nat := 15017 :: root191Tail38
def root191Tail36 : List Nat := 14939 :: root191Tail37
def root191Tail35 : List Nat := 14831 :: root191Tail36
def root191Tail34 : List Nat := 14783 :: root191Tail35
def root191Tail33 : List Nat := 13001 :: root191Tail34
def root191Tail32 : List Nat := 12377 :: root191Tail33
def root191Tail31 : List Nat := 11933 :: root191Tail32
def root191Tail30 : List Nat := 10739 :: root191Tail31
def root191Tail29 : List Nat := 9719 :: root191Tail30
def root191Tail28 : List Nat := 9257 :: root191Tail29
def root191Tail27 : List Nat := 8669 :: root191Tail28
def root191Tail26 : List Nat := 8237 :: root191Tail27
def root191Tail25 : List Nat := 8069 :: root191Tail26
def root191Tail24 : List Nat := 7577 :: root191Tail25
def root191Tail23 : List Nat := 7433 :: root191Tail24
def root191Tail22 : List Nat := 6977 :: root191Tail23
def root191Tail21 : List Nat := 6911 :: root191Tail22
def root191Tail20 : List Nat := 6719 :: root191Tail21
def root191Tail19 : List Nat := 6449 :: root191Tail20
def root191Tail18 : List Nat := 5717 :: root191Tail19
def root191Tail17 : List Nat := 5501 :: root191Tail18
def root191Tail16 : List Nat := 5171 :: root191Tail17
def root191Tail15 : List Nat := 4931 :: root191Tail16
def root191Tail14 : List Nat := 4889 :: root191Tail15
def root191Tail13 : List Nat := 4013 :: root191Tail14
def root191Tail12 : List Nat := 3797 :: root191Tail13
def root191Tail11 : List Nat := 3251 :: root191Tail12
def root191Tail10 : List Nat := 3209 :: root191Tail11
def root191Tail9 : List Nat := 3083 :: root191Tail10
def root191Tail8 : List Nat := 2729 :: root191Tail9
def root191Tail7 : List Nat := 2687 :: root191Tail8
def root191Tail6 : List Nat := 2621 :: root191Tail7
def root191Tail5 : List Nat := 1847 :: root191Tail6
def root191Tail4 : List Nat := 1601 :: root191Tail5
def root191Tail3 : List Nat := 1451 :: root191Tail4
def root191Tail2 : List Nat := 1427 :: root191Tail3
def root191Tail1 : List Nat := 1367 :: root191Tail2
def root191Tail0 : List Nat := 1361 :: root191Tail1
def root191Candidates0 : List Nat := [13001]
set_option maxRecDepth 8192 in
theorem root191Candidates0_checked : candidates edge 4 1181 1361 root191Tail1 winner.bound = root191Candidates0 := by
  unfold candidates
  have row : edge 1361 = (SortedNatTree.contains FastRows.row1361) := by
    change Data.row1361.contains = (SortedNatTree.contains FastRows.row1361)
    exact FastRows.row1361_checked.symm
  rw [row]
  decide
theorem root191Step0 : scan edge (search edge 3) 4 [1181] 1181 root191Tail0 winner =
    scan edge (search edge 3) 4 [1181] 1181 root191Tail1 winner := by
  rw [root191Tail0, scan, if_neg (by decide), root191Candidates0_checked]
  change scan edge (search edge 3) 4 [1181] 1181 root191Tail1
    (if 3 ≤ root191Candidates0.length then search edge 3 [1181, 1361] root191Candidates0 2542 winner else winner) = _
  rw [if_neg (by decide)]
def root191Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root191Candidates1_checked : candidates edge 4 1181 1367 root191Tail2 winner.bound = root191Candidates1 := by
  unfold candidates
  have row : edge 1367 = (SortedNatTree.contains FastRows.row1367) := by
    change Data.row1367.contains = (SortedNatTree.contains FastRows.row1367)
    exact FastRows.row1367_checked.symm
  rw [row]
  decide
theorem root191Step1 : scan edge (search edge 3) 4 [1181] 1181 root191Tail1 winner =
    scan edge (search edge 3) 4 [1181] 1181 root191Tail2 winner := by
  rw [root191Tail1, scan, if_neg (by decide), root191Candidates1_checked]
  change scan edge (search edge 3) 4 [1181] 1181 root191Tail2
    (if 3 ≤ root191Candidates1.length then search edge 3 [1181, 1367] root191Candidates1 2548 winner else winner) = _
  rw [if_neg (by decide)]
def root191Candidates2 : List Nat := [5501, 21881]
set_option maxRecDepth 8192 in
theorem root191Candidates2_checked : candidates edge 4 1181 1427 root191Tail3 winner.bound = root191Candidates2 := by
  unfold candidates
  have row : edge 1427 = (SortedNatTree.contains FastRows.row1427) := by
    change Data.row1427.contains = (SortedNatTree.contains FastRows.row1427)
    exact FastRows.row1427_checked.symm
  rw [row]
  decide
theorem root191Step2 : scan edge (search edge 3) 4 [1181] 1181 root191Tail2 winner =
    scan edge (search edge 3) 4 [1181] 1181 root191Tail3 winner := by
  rw [root191Tail2, scan, if_neg (by decide), root191Candidates2_checked]
  change scan edge (search edge 3) 4 [1181] 1181 root191Tail3
    (if 3 ≤ root191Candidates2.length then search edge 3 [1181, 1427] root191Candidates2 2608 winner else winner) = _
  rw [if_neg (by decide)]
def root191Candidates3 : List Nat := [21929]
set_option maxRecDepth 8192 in
theorem root191Candidates3_checked : candidates edge 4 1181 1451 root191Tail4 winner.bound = root191Candidates3 := by
  unfold candidates
  have row : edge 1451 = (SortedNatTree.contains FastRows.row1451) := by
    change Data.row1451.contains = (SortedNatTree.contains FastRows.row1451)
    exact FastRows.row1451_checked.symm
  rw [row]
  decide
theorem root191Step3 : scan edge (search edge 3) 4 [1181] 1181 root191Tail3 winner =
    scan edge (search edge 3) 4 [1181] 1181 root191Tail4 winner := by
  rw [root191Tail3, scan, if_neg (by decide), root191Candidates3_checked]
  change scan edge (search edge 3) 4 [1181] 1181 root191Tail4
    (if 3 ≤ root191Candidates3.length then search edge 3 [1181, 1451] root191Candidates3 2632 winner else winner) = _
  rw [if_neg (by decide)]
def root191Candidates4 : List Nat := [1847, 9257, 15017, 21881]
set_option maxRecDepth 8192 in
theorem root191Candidates4_checked : candidates edge 4 1181 1601 root191Tail5 winner.bound = root191Candidates4 := by
  unfold candidates
  have row : edge 1601 = (SortedNatTree.contains FastRows.row1601) := by
    change Data.row1601.contains = (SortedNatTree.contains FastRows.row1601)
    exact FastRows.row1601_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root191Child4 : search edge 3 [1181, 1601] root191Candidates4 2782 winner = winner := by decide
theorem root191Step4 : scan edge (search edge 3) 4 [1181] 1181 root191Tail4 winner =
    scan edge (search edge 3) 4 [1181] 1181 root191Tail5 winner := by
  rw [root191Tail4, scan, if_neg (by decide), root191Candidates4_checked]
  change scan edge (search edge 3) 4 [1181] 1181 root191Tail5
    (if 3 ≤ root191Candidates4.length then search edge 3 [1181, 1601] root191Candidates4 2782 winner else winner) = _
  rw [if_pos (by decide), root191Child4]
def root191Candidates5 : List Nat := [2687, 8069]
set_option maxRecDepth 8192 in
theorem root191Candidates5_checked : candidates edge 4 1181 1847 root191Tail6 winner.bound = root191Candidates5 := by
  unfold candidates
  have row : edge 1847 = (SortedNatTree.contains FastRows.row1847) := by
    change Data.row1847.contains = (SortedNatTree.contains FastRows.row1847)
    exact FastRows.row1847_checked.symm
  rw [row]
  decide
theorem root191Step5 : scan edge (search edge 3) 4 [1181] 1181 root191Tail5 winner =
    scan edge (search edge 3) 4 [1181] 1181 root191Tail6 winner := by
  rw [root191Tail5, scan, if_neg (by decide), root191Candidates5_checked]
  change scan edge (search edge 3) 4 [1181] 1181 root191Tail6
    (if 3 ≤ root191Candidates5.length then search edge 3 [1181, 1847] root191Candidates5 3028 winner else winner) = _
  rw [if_neg (by decide)]
def root191Candidates6 : List Nat := [6977]
set_option maxRecDepth 8192 in
theorem root191Candidates6_checked : candidates edge 4 1181 2621 root191Tail7 winner.bound = root191Candidates6 := by
  unfold candidates
  have row : edge 2621 = (SortedNatTree.contains FastRows.row2621) := by
    change Data.row2621.contains = (SortedNatTree.contains FastRows.row2621)
    exact FastRows.row2621_checked.symm
  rw [row]
  decide
theorem root191Step6 : scan edge (search edge 3) 4 [1181] 1181 root191Tail6 winner =
    scan edge (search edge 3) 4 [1181] 1181 root191Tail7 winner := by
  rw [root191Tail6, scan, if_neg (by decide), root191Candidates6_checked]
  change scan edge (search edge 3) 4 [1181] 1181 root191Tail7
    (if 3 ≤ root191Candidates6.length then search edge 3 [1181, 2621] root191Candidates6 3802 winner else winner) = _
  rw [if_neg (by decide)]
def root191Candidates7 : List Nat := [8069]
set_option maxRecDepth 8192 in
theorem root191Candidates7_checked : candidates edge 4 1181 2687 root191Tail8 winner.bound = root191Candidates7 := by
  unfold candidates
  have row : edge 2687 = (SortedNatTree.contains FastRows.row2687) := by
    change Data.row2687.contains = (SortedNatTree.contains FastRows.row2687)
    exact FastRows.row2687_checked.symm
  rw [row]
  decide
theorem root191Step7 : scan edge (search edge 3) 4 [1181] 1181 root191Tail7 winner =
    scan edge (search edge 3) 4 [1181] 1181 root191Tail8 winner := by
  rw [root191Tail7, scan, if_neg (by decide), root191Candidates7_checked]
  change scan edge (search edge 3) 4 [1181] 1181 root191Tail8
    (if 3 ≤ root191Candidates7.length then search edge 3 [1181, 2687] root191Candidates7 3868 winner else winner) = _
  rw [if_neg (by decide)]
def root191Candidates8 : List Nat := [9719]
set_option maxRecDepth 8192 in
theorem root191Candidates8_checked : candidates edge 4 1181 2729 root191Tail9 winner.bound = root191Candidates8 := by
  unfold candidates
  have row : edge 2729 = (SortedNatTree.contains FastRows.row2729) := by
    change Data.row2729.contains = (SortedNatTree.contains FastRows.row2729)
    exact FastRows.row2729_checked.symm
  rw [row]
  decide
theorem root191Step8 : scan edge (search edge 3) 4 [1181] 1181 root191Tail8 winner =
    scan edge (search edge 3) 4 [1181] 1181 root191Tail9 winner := by
  rw [root191Tail8, scan, if_neg (by decide), root191Candidates8_checked]
  change scan edge (search edge 3) 4 [1181] 1181 root191Tail9
    (if 3 ≤ root191Candidates8.length then search edge 3 [1181, 2729] root191Candidates8 3910 winner else winner) = _
  rw [if_neg (by decide)]
def root191Candidates9 : List Nat := [3797, 4931]
set_option maxRecDepth 8192 in
theorem root191Candidates9_checked : candidates edge 4 1181 3083 root191Tail10 winner.bound = root191Candidates9 := by
  unfold candidates
  have row : edge 3083 = (SortedNatTree.contains FastRows.row3083) := by
    change Data.row3083.contains = (SortedNatTree.contains FastRows.row3083)
    exact FastRows.row3083_checked.symm
  rw [row]
  decide
theorem root191Step9 : scan edge (search edge 3) 4 [1181] 1181 root191Tail9 winner =
    scan edge (search edge 3) 4 [1181] 1181 root191Tail10 winner := by
  rw [root191Tail9, scan, if_neg (by decide), root191Candidates9_checked]
  change scan edge (search edge 3) 4 [1181] 1181 root191Tail10
    (if 3 ≤ root191Candidates9.length then search edge 3 [1181, 3083] root191Candidates9 4264 winner else winner) = _
  rw [if_neg (by decide)]
def root191Candidates10 : List Nat := [5171, 5717, 7433, 9257, 17159]
set_option maxRecDepth 8192 in
theorem root191Candidates10_checked : candidates edge 4 1181 3209 root191Tail11 winner.bound = root191Candidates10 := by
  unfold candidates
  have row : edge 3209 = (SortedNatTree.contains FastRows.row3209) := by
    change Data.row3209.contains = (SortedNatTree.contains FastRows.row3209)
    exact FastRows.row3209_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root191Child10 : search edge 3 [1181, 3209] root191Candidates10 4390 winner = winner := by decide
theorem root191Step10 : scan edge (search edge 3) 4 [1181] 1181 root191Tail10 winner =
    scan edge (search edge 3) 4 [1181] 1181 root191Tail11 winner := by
  rw [root191Tail10, scan, if_neg (by decide), root191Candidates10_checked]
  change scan edge (search edge 3) 4 [1181] 1181 root191Tail11
    (if 3 ≤ root191Candidates10.length then search edge 3 [1181, 3209] root191Candidates10 4390 winner else winner) = _
  rw [if_pos (by decide), root191Child10]
def root191Candidates11 : List Nat := [14831]
set_option maxRecDepth 8192 in
theorem root191Candidates11_checked : candidates edge 4 1181 3251 root191Tail12 winner.bound = root191Candidates11 := by
  unfold candidates
  have row : edge 3251 = (SortedNatTree.contains FastRows.row3251) := by
    change Data.row3251.contains = (SortedNatTree.contains FastRows.row3251)
    exact FastRows.row3251_checked.symm
  rw [row]
  decide
theorem root191Step11 : scan edge (search edge 3) 4 [1181] 1181 root191Tail11 winner =
    scan edge (search edge 3) 4 [1181] 1181 root191Tail12 winner := by
  rw [root191Tail11, scan, if_neg (by decide), root191Candidates11_checked]
  change scan edge (search edge 3) 4 [1181] 1181 root191Tail12
    (if 3 ≤ root191Candidates11.length then search edge 3 [1181, 3251] root191Candidates11 4432 winner else winner) = _
  rw [if_neg (by decide)]
def root191Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root191Candidates12_checked : candidates edge 4 1181 3797 root191Tail13 winner.bound = root191Candidates12 := by
  unfold candidates
  have row : edge 3797 = (SortedNatTree.contains FastRows.row3797) := by
    change Data.row3797.contains = (SortedNatTree.contains FastRows.row3797)
    exact FastRows.row3797_checked.symm
  rw [row]
  decide
theorem root191Step12 : scan edge (search edge 3) 4 [1181] 1181 root191Tail12 winner =
    scan edge (search edge 3) 4 [1181] 1181 root191Tail13 winner := by
  rw [root191Tail12, scan, if_neg (by decide), root191Candidates12_checked]
  change scan edge (search edge 3) 4 [1181] 1181 root191Tail13
    (if 3 ≤ root191Candidates12.length then search edge 3 [1181, 3797] root191Candidates12 4978 winner else winner) = _
  rw [if_neg (by decide)]
def root191Candidates13 : List Nat := [8069]
set_option maxRecDepth 8192 in
theorem root191Candidates13_checked : candidates edge 4 1181 4013 root191Tail14 winner.bound = root191Candidates13 := by
  unfold candidates
  have row : edge 4013 = (SortedNatTree.contains FastRows.row4013) := by
    change Data.row4013.contains = (SortedNatTree.contains FastRows.row4013)
    exact FastRows.row4013_checked.symm
  rw [row]
  decide
theorem root191Step13 : scan edge (search edge 3) 4 [1181] 1181 root191Tail13 winner =
    scan edge (search edge 3) 4 [1181] 1181 root191Tail14 winner := by
  rw [root191Tail13, scan, if_neg (by decide), root191Candidates13_checked]
  change scan edge (search edge 3) 4 [1181] 1181 root191Tail14
    (if 3 ≤ root191Candidates13.length then search edge 3 [1181, 4013] root191Candidates13 5194 winner else winner) = _
  rw [if_neg (by decide)]
def root191Candidates14 : List Nat := [8237, 18149]
set_option maxRecDepth 8192 in
theorem root191Candidates14_checked : candidates edge 4 1181 4889 root191Tail15 winner.bound = root191Candidates14 := by
  unfold candidates
  have row : edge 4889 = (SortedNatTree.contains FastRows.row4889) := by
    change Data.row4889.contains = (SortedNatTree.contains FastRows.row4889)
    exact FastRows.row4889_checked.symm
  rw [row]
  decide
theorem root191Step14 : scan edge (search edge 3) 4 [1181] 1181 root191Tail14 winner =
    scan edge (search edge 3) 4 [1181] 1181 root191Tail15 winner := by
  rw [root191Tail14, scan, if_neg (by decide), root191Candidates14_checked]
  change scan edge (search edge 3) 4 [1181] 1181 root191Tail15
    (if 3 ≤ root191Candidates14.length then search edge 3 [1181, 4889] root191Candidates14 6070 winner else winner) = _
  rw [if_neg (by decide)]
def root191Candidates15 : List Nat := [11933, 15017]
set_option maxRecDepth 8192 in
theorem root191Candidates15_checked : candidates edge 4 1181 4931 root191Tail16 winner.bound = root191Candidates15 := by
  unfold candidates
  have row : edge 4931 = (SortedNatTree.contains FastRows.row4931) := by
    change Data.row4931.contains = (SortedNatTree.contains FastRows.row4931)
    exact FastRows.row4931_checked.symm
  rw [row]
  decide
theorem root191Step15 : scan edge (search edge 3) 4 [1181] 1181 root191Tail15 winner =
    scan edge (search edge 3) 4 [1181] 1181 root191Tail16 winner := by
  rw [root191Tail15, scan, if_neg (by decide), root191Candidates15_checked]
  change scan edge (search edge 3) 4 [1181] 1181 root191Tail16
    (if 3 ≤ root191Candidates15.length then search edge 3 [1181, 4931] root191Candidates15 6112 winner else winner) = _
  rw [if_neg (by decide)]
def root191Candidates16 : List Nat := [15017]
set_option maxRecDepth 8192 in
theorem root191Candidates16_checked : candidates edge 4 1181 5171 root191Tail17 winner.bound = root191Candidates16 := by
  unfold candidates
  have row : edge 5171 = (SortedNatTree.contains FastRows.row5171) := by
    change Data.row5171.contains = (SortedNatTree.contains FastRows.row5171)
    exact FastRows.row5171_checked.symm
  rw [row]
  decide
theorem root191Step16 : scan edge (search edge 3) 4 [1181] 1181 root191Tail16 winner =
    scan edge (search edge 3) 4 [1181] 1181 root191Tail17 winner := by
  rw [root191Tail16, scan, if_neg (by decide), root191Candidates16_checked]
  change scan edge (search edge 3) 4 [1181] 1181 root191Tail17
    (if 3 ≤ root191Candidates16.length then search edge 3 [1181, 5171] root191Candidates16 6352 winner else winner) = _
  rw [if_neg (by decide)]
def root191Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root191Candidates17_checked : candidates edge 4 1181 5501 root191Tail18 winner.bound = root191Candidates17 := by
  unfold candidates
  have row : edge 5501 = (SortedNatTree.contains FastRows.row5501) := by
    change Data.row5501.contains = (SortedNatTree.contains FastRows.row5501)
    exact FastRows.row5501_checked.symm
  rw [row]
  decide
theorem root191Step17 : scan edge (search edge 3) 4 [1181] 1181 root191Tail17 winner =
    scan edge (search edge 3) 4 [1181] 1181 root191Tail18 winner := by
  rw [root191Tail17, scan, if_neg (by decide), root191Candidates17_checked]
  change scan edge (search edge 3) 4 [1181] 1181 root191Tail18
    (if 3 ≤ root191Candidates17.length then search edge 3 [1181, 5501] root191Candidates17 6682 winner else winner) = _
  rw [if_neg (by decide)]
def root191Candidates18 : List Nat := [16763]
set_option maxRecDepth 8192 in
theorem root191Candidates18_checked : candidates edge 4 1181 5717 root191Tail19 winner.bound = root191Candidates18 := by
  unfold candidates
  have row : edge 5717 = (SortedNatTree.contains FastRows.row5717) := by
    change Data.row5717.contains = (SortedNatTree.contains FastRows.row5717)
    exact FastRows.row5717_checked.symm
  rw [row]
  decide
theorem root191Step18 : scan edge (search edge 3) 4 [1181] 1181 root191Tail18 winner =
    scan edge (search edge 3) 4 [1181] 1181 root191Tail19 winner := by
  rw [root191Tail18, scan, if_neg (by decide), root191Candidates18_checked]
  change scan edge (search edge 3) 4 [1181] 1181 root191Tail19
    (if 3 ≤ root191Candidates18.length then search edge 3 [1181, 5717] root191Candidates18 6898 winner else winner) = _
  rw [if_neg (by decide)]
theorem root191Scan19 : scan edge (search edge 3) 4 [1181] 1181 root191Tail19 winner = winner := by rw [root191Tail19, scan, if_pos (by decide)]
theorem root191Scan18 : scan edge (search edge 3) 4 [1181] 1181 root191Tail18 winner = winner :=
  (root191Step18).trans root191Scan19
theorem root191Scan17 : scan edge (search edge 3) 4 [1181] 1181 root191Tail17 winner = winner :=
  (root191Step17).trans root191Scan18
theorem root191Scan16 : scan edge (search edge 3) 4 [1181] 1181 root191Tail16 winner = winner :=
  (root191Step16).trans root191Scan17
theorem root191Scan15 : scan edge (search edge 3) 4 [1181] 1181 root191Tail15 winner = winner :=
  (root191Step15).trans root191Scan16
theorem root191Scan14 : scan edge (search edge 3) 4 [1181] 1181 root191Tail14 winner = winner :=
  (root191Step14).trans root191Scan15
theorem root191Scan13 : scan edge (search edge 3) 4 [1181] 1181 root191Tail13 winner = winner :=
  (root191Step13).trans root191Scan14
theorem root191Scan12 : scan edge (search edge 3) 4 [1181] 1181 root191Tail12 winner = winner :=
  (root191Step12).trans root191Scan13
theorem root191Scan11 : scan edge (search edge 3) 4 [1181] 1181 root191Tail11 winner = winner :=
  (root191Step11).trans root191Scan12
theorem root191Scan10 : scan edge (search edge 3) 4 [1181] 1181 root191Tail10 winner = winner :=
  (root191Step10).trans root191Scan11
theorem root191Scan9 : scan edge (search edge 3) 4 [1181] 1181 root191Tail9 winner = winner :=
  (root191Step9).trans root191Scan10
theorem root191Scan8 : scan edge (search edge 3) 4 [1181] 1181 root191Tail8 winner = winner :=
  (root191Step8).trans root191Scan9
theorem root191Scan7 : scan edge (search edge 3) 4 [1181] 1181 root191Tail7 winner = winner :=
  (root191Step7).trans root191Scan8
theorem root191Scan6 : scan edge (search edge 3) 4 [1181] 1181 root191Tail6 winner = winner :=
  (root191Step6).trans root191Scan7
theorem root191Scan5 : scan edge (search edge 3) 4 [1181] 1181 root191Tail5 winner = winner :=
  (root191Step5).trans root191Scan6
theorem root191Scan4 : scan edge (search edge 3) 4 [1181] 1181 root191Tail4 winner = winner :=
  (root191Step4).trans root191Scan5
theorem root191Scan3 : scan edge (search edge 3) 4 [1181] 1181 root191Tail3 winner = winner :=
  (root191Step3).trans root191Scan4
theorem root191Scan2 : scan edge (search edge 3) 4 [1181] 1181 root191Tail2 winner = winner :=
  (root191Step2).trans root191Scan3
theorem root191Scan1 : scan edge (search edge 3) 4 [1181] 1181 root191Tail1 winner = winner :=
  (root191Step1).trans root191Scan2
theorem root191Scan0 : scan edge (search edge 3) 4 [1181] 1181 root191Tail0 winner = winner :=
  (root191Step0).trans root191Scan1
set_option maxRecDepth 4096 in
theorem root191Branch : search edge 4 [1181] root191Candidates 1181 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1181] 1181 root191Tail0 winner = winner
  exact root191Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
