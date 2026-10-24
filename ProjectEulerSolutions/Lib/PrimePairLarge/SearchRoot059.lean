import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root59Candidates : List Nat := Data.row293
set_option maxRecDepth 8192 in
theorem root59Candidates_checked : candidates edge 5 0 293 tail60 winner.bound = root59Candidates := by
  apply candidates_eq_row edge 5 0 293 tail60 root59Candidates winner.bound tail60_ordered
  · decide
  · decide
  · funext q
    rfl
def root59Tail69 : List Nat := []
def root59Tail68 : List Nat := 24767 :: root59Tail69
def root59Tail67 : List Nat := 23447 :: root59Tail68
def root59Tail66 : List Nat := 22961 :: root59Tail67
def root59Tail65 : List Nat := 21863 :: root59Tail66
def root59Tail64 : List Nat := 21647 :: root59Tail65
def root59Tail63 : List Nat := 21557 :: root59Tail64
def root59Tail62 : List Nat := 21323 :: root59Tail63
def root59Tail61 : List Nat := 21101 :: root59Tail62
def root59Tail60 : List Nat := 21011 :: root59Tail61
def root59Tail59 : List Nat := 20981 :: root59Tail60
def root59Tail58 : List Nat := 20897 :: root59Tail59
def root59Tail57 : List Nat := 20483 :: root59Tail58
def root59Tail56 : List Nat := 19973 :: root59Tail57
def root59Tail55 : List Nat := 19919 :: root59Tail56
def root59Tail54 : List Nat := 18719 :: root59Tail55
def root59Tail53 : List Nat := 18371 :: root59Tail54
def root59Tail52 : List Nat := 18257 :: root59Tail53
def root59Tail51 : List Nat := 17903 :: root59Tail52
def root59Tail50 : List Nat := 16649 :: root59Tail51
def root59Tail49 : List Nat := 16223 :: root59Tail50
def root59Tail48 : List Nat := 16193 :: root59Tail49
def root59Tail47 : List Nat := 16103 :: root59Tail48
def root59Tail46 : List Nat := 15527 :: root59Tail47
def root59Tail45 : List Nat := 15401 :: root59Tail46
def root59Tail44 : List Nat := 15101 :: root59Tail45
def root59Tail43 : List Nat := 13829 :: root59Tail44
def root59Tail42 : List Nat := 13679 :: root59Tail43
def root59Tail41 : List Nat := 13337 :: root59Tail42
def root59Tail40 : List Nat := 12923 :: root59Tail41
def root59Tail39 : List Nat := 12659 :: root59Tail40
def root59Tail38 : List Nat := 12401 :: root59Tail39
def root59Tail37 : List Nat := 12113 :: root59Tail38
def root59Tail36 : List Nat := 11831 :: root59Tail37
def root59Tail35 : List Nat := 11483 :: root59Tail36
def root59Tail34 : List Nat := 10973 :: root59Tail35
def root59Tail33 : List Nat := 10847 :: root59Tail34
def root59Tail32 : List Nat := 10529 :: root59Tail33
def root59Tail31 : List Nat := 9941 :: root59Tail32
def root59Tail30 : List Nat := 9767 :: root59Tail31
def root59Tail29 : List Nat := 9689 :: root59Tail30
def root59Tail28 : List Nat := 9473 :: root59Tail29
def root59Tail27 : List Nat := 9371 :: root59Tail28
def root59Tail26 : List Nat := 8783 :: root59Tail27
def root59Tail25 : List Nat := 8753 :: root59Tail26
def root59Tail24 : List Nat := 8669 :: root59Tail25
def root59Tail23 : List Nat := 8237 :: root59Tail24
def root59Tail22 : List Nat := 8009 :: root59Tail23
def root59Tail21 : List Nat := 7757 :: root59Tail22
def root59Tail20 : List Nat := 7523 :: root59Tail21
def root59Tail19 : List Nat := 5717 :: root59Tail20
def root59Tail18 : List Nat := 5189 :: root59Tail19
def root59Tail17 : List Nat := 5171 :: root59Tail18
def root59Tail16 : List Nat := 5147 :: root59Tail17
def root59Tail15 : List Nat := 4751 :: root59Tail16
def root59Tail14 : List Nat := 4091 :: root59Tail15
def root59Tail13 : List Nat := 3083 :: root59Tail14
def root59Tail12 : List Nat := 2777 :: root59Tail13
def root59Tail11 : List Nat := 2459 :: root59Tail12
def root59Tail10 : List Nat := 2411 :: root59Tail11
def root59Tail9 : List Nat := 1997 :: root59Tail10
def root59Tail8 : List Nat := 1823 :: root59Tail9
def root59Tail7 : List Nat := 1583 :: root59Tail8
def root59Tail6 : List Nat := 1493 :: root59Tail7
def root59Tail5 : List Nat := 1433 :: root59Tail6
def root59Tail4 : List Nat := 1319 :: root59Tail5
def root59Tail3 : List Nat := 827 :: root59Tail4
def root59Tail2 : List Nat := 617 :: root59Tail3
def root59Tail1 : List Nat := 467 :: root59Tail2
def root59Tail0 : List Nat := 311 :: root59Tail1
def root59Candidates0 : List Nat := [827, 5171, 7523, 8237, 13829, 15401, 21557]
set_option maxRecDepth 8192 in
theorem root59Candidates0_checked : candidates edge 4 293 311 root59Tail1 winner.bound = root59Candidates0 := by
  unfold candidates
  have row : edge 311 = (SortedNatTree.contains FastRows.row311) := by
    change Data.row311.contains = (SortedNatTree.contains FastRows.row311)
    exact FastRows.row311_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root59Child0 : search edge 3 [293, 311] root59Candidates0 604 winner = winner := by decide
theorem root59Step0 : scan edge (search edge 3) 4 [293] 293 root59Tail0 winner =
    scan edge (search edge 3) 4 [293] 293 root59Tail1 winner := by
  rw [root59Tail0, scan, if_neg (by decide), root59Candidates0_checked]
  change scan edge (search edge 3) 4 [293] 293 root59Tail1
    (if 3 ≤ root59Candidates0.length then search edge 3 [293, 311] root59Candidates0 604 winner else winner) = _
  rw [if_pos (by decide), root59Child0]
def root59Candidates1 : List Nat := [617, 9473, 9767, 13679, 23447]
set_option maxRecDepth 8192 in
theorem root59Candidates1_checked : candidates edge 4 293 467 root59Tail2 winner.bound = root59Candidates1 := by
  unfold candidates
  have row : edge 467 = (SortedNatTree.contains FastRows.row467) := by
    change Data.row467.contains = (SortedNatTree.contains FastRows.row467)
    exact FastRows.row467_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root59Child1 : search edge 3 [293, 467] root59Candidates1 760 winner = winner := by decide
theorem root59Step1 : scan edge (search edge 3) 4 [293] 293 root59Tail1 winner =
    scan edge (search edge 3) 4 [293] 293 root59Tail2 winner := by
  rw [root59Tail1, scan, if_neg (by decide), root59Candidates1_checked]
  change scan edge (search edge 3) 4 [293] 293 root59Tail2
    (if 3 ≤ root59Candidates1.length then search edge 3 [293, 467] root59Candidates1 760 winner else winner) = _
  rw [if_pos (by decide), root59Child1]
def root59Candidates2 : List Nat := [9473, 10973, 16649, 19919]
set_option maxRecDepth 8192 in
theorem root59Candidates2_checked : candidates edge 4 293 617 root59Tail3 winner.bound = root59Candidates2 := by
  unfold candidates
  have row : edge 617 = (SortedNatTree.contains FastRows.row617) := by
    change Data.row617.contains = (SortedNatTree.contains FastRows.row617)
    exact FastRows.row617_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root59Child2 : search edge 3 [293, 617] root59Candidates2 910 winner = winner := by decide
theorem root59Step2 : scan edge (search edge 3) 4 [293] 293 root59Tail2 winner =
    scan edge (search edge 3) 4 [293] 293 root59Tail3 winner := by
  rw [root59Tail2, scan, if_neg (by decide), root59Candidates2_checked]
  change scan edge (search edge 3) 4 [293] 293 root59Tail3
    (if 3 ≤ root59Candidates2.length then search edge 3 [293, 617] root59Candidates2 910 winner else winner) = _
  rw [if_pos (by decide), root59Child2]
def root59Candidates3 : List Nat := [5717, 11483]
set_option maxRecDepth 8192 in
theorem root59Candidates3_checked : candidates edge 4 293 827 root59Tail4 winner.bound = root59Candidates3 := by
  unfold candidates
  have row : edge 827 = (SortedNatTree.contains FastRows.row827) := by
    change Data.row827.contains = (SortedNatTree.contains FastRows.row827)
    exact FastRows.row827_checked.symm
  rw [row]
  decide
theorem root59Step3 : scan edge (search edge 3) 4 [293] 293 root59Tail3 winner =
    scan edge (search edge 3) 4 [293] 293 root59Tail4 winner := by
  rw [root59Tail3, scan, if_neg (by decide), root59Candidates3_checked]
  change scan edge (search edge 3) 4 [293] 293 root59Tail4
    (if 3 ≤ root59Candidates3.length then search edge 3 [293, 827] root59Candidates3 1120 winner else winner) = _
  rw [if_neg (by decide)]
def root59Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root59Candidates4_checked : candidates edge 4 293 1319 root59Tail5 winner.bound = root59Candidates4 := by
  unfold candidates
  have row : edge 1319 = (SortedNatTree.contains FastRows.row1319) := by
    change Data.row1319.contains = (SortedNatTree.contains FastRows.row1319)
    exact FastRows.row1319_checked.symm
  rw [row]
  decide
theorem root59Step4 : scan edge (search edge 3) 4 [293] 293 root59Tail4 winner =
    scan edge (search edge 3) 4 [293] 293 root59Tail5 winner := by
  rw [root59Tail4, scan, if_neg (by decide), root59Candidates4_checked]
  change scan edge (search edge 3) 4 [293] 293 root59Tail5
    (if 3 ≤ root59Candidates4.length then search edge 3 [293, 1319] root59Candidates4 1612 winner else winner) = _
  rw [if_neg (by decide)]
def root59Candidates5 : List Nat := [21863]
set_option maxRecDepth 8192 in
theorem root59Candidates5_checked : candidates edge 4 293 1433 root59Tail6 winner.bound = root59Candidates5 := by
  unfold candidates
  have row : edge 1433 = (SortedNatTree.contains FastRows.row1433) := by
    change Data.row1433.contains = (SortedNatTree.contains FastRows.row1433)
    exact FastRows.row1433_checked.symm
  rw [row]
  decide
theorem root59Step5 : scan edge (search edge 3) 4 [293] 293 root59Tail5 winner =
    scan edge (search edge 3) 4 [293] 293 root59Tail6 winner := by
  rw [root59Tail5, scan, if_neg (by decide), root59Candidates5_checked]
  change scan edge (search edge 3) 4 [293] 293 root59Tail6
    (if 3 ≤ root59Candidates5.length then search edge 3 [293, 1433] root59Candidates5 1726 winner else winner) = _
  rw [if_neg (by decide)]
def root59Candidates6 : List Nat := [5189, 9473]
set_option maxRecDepth 8192 in
theorem root59Candidates6_checked : candidates edge 4 293 1493 root59Tail7 winner.bound = root59Candidates6 := by
  unfold candidates
  have row : edge 1493 = (SortedNatTree.contains FastRows.row1493) := by
    change Data.row1493.contains = (SortedNatTree.contains FastRows.row1493)
    exact FastRows.row1493_checked.symm
  rw [row]
  decide
theorem root59Step6 : scan edge (search edge 3) 4 [293] 293 root59Tail6 winner =
    scan edge (search edge 3) 4 [293] 293 root59Tail7 winner := by
  rw [root59Tail6, scan, if_neg (by decide), root59Candidates6_checked]
  change scan edge (search edge 3) 4 [293] 293 root59Tail7
    (if 3 ≤ root59Candidates6.length then search edge 3 [293, 1493] root59Candidates6 1786 winner else winner) = _
  rw [if_neg (by decide)]
def root59Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root59Candidates7_checked : candidates edge 4 293 1583 root59Tail8 winner.bound = root59Candidates7 := by
  unfold candidates
  have row : edge 1583 = (SortedNatTree.contains FastRows.row1583) := by
    change Data.row1583.contains = (SortedNatTree.contains FastRows.row1583)
    exact FastRows.row1583_checked.symm
  rw [row]
  decide
theorem root59Step7 : scan edge (search edge 3) 4 [293] 293 root59Tail7 winner =
    scan edge (search edge 3) 4 [293] 293 root59Tail8 winner := by
  rw [root59Tail7, scan, if_neg (by decide), root59Candidates7_checked]
  change scan edge (search edge 3) 4 [293] 293 root59Tail8
    (if 3 ≤ root59Candidates7.length then search edge 3 [293, 1583] root59Candidates7 1876 winner else winner) = _
  rw [if_neg (by decide)]
def root59Candidates8 : List Nat := [8783, 12659]
set_option maxRecDepth 8192 in
theorem root59Candidates8_checked : candidates edge 4 293 1823 root59Tail9 winner.bound = root59Candidates8 := by
  unfold candidates
  have row : edge 1823 = (SortedNatTree.contains FastRows.row1823) := by
    change Data.row1823.contains = (SortedNatTree.contains FastRows.row1823)
    exact FastRows.row1823_checked.symm
  rw [row]
  decide
theorem root59Step8 : scan edge (search edge 3) 4 [293] 293 root59Tail8 winner =
    scan edge (search edge 3) 4 [293] 293 root59Tail9 winner := by
  rw [root59Tail8, scan, if_neg (by decide), root59Candidates8_checked]
  change scan edge (search edge 3) 4 [293] 293 root59Tail9
    (if 3 ≤ root59Candidates8.length then search edge 3 [293, 1823] root59Candidates8 2116 winner else winner) = _
  rw [if_neg (by decide)]
def root59Candidates9 : List Nat := [9473, 9941]
set_option maxRecDepth 8192 in
theorem root59Candidates9_checked : candidates edge 4 293 1997 root59Tail10 winner.bound = root59Candidates9 := by
  unfold candidates
  have row : edge 1997 = (SortedNatTree.contains FastRows.row1997) := by
    change Data.row1997.contains = (SortedNatTree.contains FastRows.row1997)
    exact FastRows.row1997_checked.symm
  rw [row]
  decide
theorem root59Step9 : scan edge (search edge 3) 4 [293] 293 root59Tail9 winner =
    scan edge (search edge 3) 4 [293] 293 root59Tail10 winner := by
  rw [root59Tail9, scan, if_neg (by decide), root59Candidates9_checked]
  change scan edge (search edge 3) 4 [293] 293 root59Tail10
    (if 3 ≤ root59Candidates9.length then search edge 3 [293, 1997] root59Candidates9 2290 winner else winner) = _
  rw [if_neg (by decide)]
def root59Candidates10 : List Nat := [5171]
set_option maxRecDepth 8192 in
theorem root59Candidates10_checked : candidates edge 4 293 2411 root59Tail11 winner.bound = root59Candidates10 := by
  unfold candidates
  have row : edge 2411 = (SortedNatTree.contains FastRows.row2411) := by
    change Data.row2411.contains = (SortedNatTree.contains FastRows.row2411)
    exact FastRows.row2411_checked.symm
  rw [row]
  decide
theorem root59Step10 : scan edge (search edge 3) 4 [293] 293 root59Tail10 winner =
    scan edge (search edge 3) 4 [293] 293 root59Tail11 winner := by
  rw [root59Tail10, scan, if_neg (by decide), root59Candidates10_checked]
  change scan edge (search edge 3) 4 [293] 293 root59Tail11
    (if 3 ≤ root59Candidates10.length then search edge 3 [293, 2411] root59Candidates10 2704 winner else winner) = _
  rw [if_neg (by decide)]
def root59Candidates11 : List Nat := [5171, 8009]
set_option maxRecDepth 8192 in
theorem root59Candidates11_checked : candidates edge 4 293 2459 root59Tail12 winner.bound = root59Candidates11 := by
  unfold candidates
  have row : edge 2459 = (SortedNatTree.contains FastRows.row2459) := by
    change Data.row2459.contains = (SortedNatTree.contains FastRows.row2459)
    exact FastRows.row2459_checked.symm
  rw [row]
  decide
theorem root59Step11 : scan edge (search edge 3) 4 [293] 293 root59Tail11 winner =
    scan edge (search edge 3) 4 [293] 293 root59Tail12 winner := by
  rw [root59Tail11, scan, if_neg (by decide), root59Candidates11_checked]
  change scan edge (search edge 3) 4 [293] 293 root59Tail12
    (if 3 ≤ root59Candidates11.length then search edge 3 [293, 2459] root59Candidates11 2752 winner else winner) = _
  rw [if_neg (by decide)]
def root59Candidates12 : List Nat := [15401, 18719]
set_option maxRecDepth 8192 in
theorem root59Candidates12_checked : candidates edge 4 293 2777 root59Tail13 winner.bound = root59Candidates12 := by
  unfold candidates
  have row : edge 2777 = (SortedNatTree.contains FastRows.row2777) := by
    change Data.row2777.contains = (SortedNatTree.contains FastRows.row2777)
    exact FastRows.row2777_checked.symm
  rw [row]
  decide
theorem root59Step12 : scan edge (search edge 3) 4 [293] 293 root59Tail12 winner =
    scan edge (search edge 3) 4 [293] 293 root59Tail13 winner := by
  rw [root59Tail12, scan, if_neg (by decide), root59Candidates12_checked]
  change scan edge (search edge 3) 4 [293] 293 root59Tail13
    (if 3 ≤ root59Candidates12.length then search edge 3 [293, 2777] root59Candidates12 3070 winner else winner) = _
  rw [if_neg (by decide)]
def root59Candidates13 : List Nat := [17903]
set_option maxRecDepth 8192 in
theorem root59Candidates13_checked : candidates edge 4 293 3083 root59Tail14 winner.bound = root59Candidates13 := by
  unfold candidates
  have row : edge 3083 = (SortedNatTree.contains FastRows.row3083) := by
    change Data.row3083.contains = (SortedNatTree.contains FastRows.row3083)
    exact FastRows.row3083_checked.symm
  rw [row]
  decide
theorem root59Step13 : scan edge (search edge 3) 4 [293] 293 root59Tail13 winner =
    scan edge (search edge 3) 4 [293] 293 root59Tail14 winner := by
  rw [root59Tail13, scan, if_neg (by decide), root59Candidates13_checked]
  change scan edge (search edge 3) 4 [293] 293 root59Tail14
    (if 3 ≤ root59Candidates13.length then search edge 3 [293, 3083] root59Candidates13 3376 winner else winner) = _
  rw [if_neg (by decide)]
def root59Candidates14 : List Nat := [8237, 10973, 19973]
set_option maxRecDepth 8192 in
theorem root59Candidates14_checked : candidates edge 4 293 4091 root59Tail15 winner.bound = root59Candidates14 := by
  unfold candidates
  have row : edge 4091 = (SortedNatTree.contains FastRows.row4091) := by
    change Data.row4091.contains = (SortedNatTree.contains FastRows.row4091)
    exact FastRows.row4091_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root59Child14 : search edge 3 [293, 4091] root59Candidates14 4384 winner = winner := by decide
theorem root59Step14 : scan edge (search edge 3) 4 [293] 293 root59Tail14 winner =
    scan edge (search edge 3) 4 [293] 293 root59Tail15 winner := by
  rw [root59Tail14, scan, if_neg (by decide), root59Candidates14_checked]
  change scan edge (search edge 3) 4 [293] 293 root59Tail15
    (if 3 ≤ root59Candidates14.length then search edge 3 [293, 4091] root59Candidates14 4384 winner else winner) = _
  rw [if_pos (by decide), root59Child14]
def root59Candidates15 : List Nat := [18257]
set_option maxRecDepth 8192 in
theorem root59Candidates15_checked : candidates edge 4 293 4751 root59Tail16 winner.bound = root59Candidates15 := by
  unfold candidates
  have row : edge 4751 = (SortedNatTree.contains FastRows.row4751) := by
    change Data.row4751.contains = (SortedNatTree.contains FastRows.row4751)
    exact FastRows.row4751_checked.symm
  rw [row]
  decide
theorem root59Step15 : scan edge (search edge 3) 4 [293] 293 root59Tail15 winner =
    scan edge (search edge 3) 4 [293] 293 root59Tail16 winner := by
  rw [root59Tail15, scan, if_neg (by decide), root59Candidates15_checked]
  change scan edge (search edge 3) 4 [293] 293 root59Tail16
    (if 3 ≤ root59Candidates15.length then search edge 3 [293, 4751] root59Candidates15 5044 winner else winner) = _
  rw [if_neg (by decide)]
def root59Candidates16 : List Nat := [15101]
set_option maxRecDepth 8192 in
theorem root59Candidates16_checked : candidates edge 4 293 5147 root59Tail17 winner.bound = root59Candidates16 := by
  unfold candidates
  have row : edge 5147 = (SortedNatTree.contains FastRows.row5147) := by
    change Data.row5147.contains = (SortedNatTree.contains FastRows.row5147)
    exact FastRows.row5147_checked.symm
  rw [row]
  decide
theorem root59Step16 : scan edge (search edge 3) 4 [293] 293 root59Tail16 winner =
    scan edge (search edge 3) 4 [293] 293 root59Tail17 winner := by
  rw [root59Tail16, scan, if_neg (by decide), root59Candidates16_checked]
  change scan edge (search edge 3) 4 [293] 293 root59Tail17
    (if 3 ≤ root59Candidates16.length then search edge 3 [293, 5147] root59Candidates16 5440 winner else winner) = _
  rw [if_neg (by decide)]
def root59Candidates17 : List Nat := [13679]
set_option maxRecDepth 8192 in
theorem root59Candidates17_checked : candidates edge 4 293 5171 root59Tail18 winner.bound = root59Candidates17 := by
  unfold candidates
  have row : edge 5171 = (SortedNatTree.contains FastRows.row5171) := by
    change Data.row5171.contains = (SortedNatTree.contains FastRows.row5171)
    exact FastRows.row5171_checked.symm
  rw [row]
  decide
theorem root59Step17 : scan edge (search edge 3) 4 [293] 293 root59Tail17 winner =
    scan edge (search edge 3) 4 [293] 293 root59Tail18 winner := by
  rw [root59Tail17, scan, if_neg (by decide), root59Candidates17_checked]
  change scan edge (search edge 3) 4 [293] 293 root59Tail18
    (if 3 ≤ root59Candidates17.length then search edge 3 [293, 5171] root59Candidates17 5464 winner else winner) = _
  rw [if_neg (by decide)]
def root59Candidates18 : List Nat := []
set_option maxRecDepth 8192 in
theorem root59Candidates18_checked : candidates edge 4 293 5189 root59Tail19 winner.bound = root59Candidates18 := by
  unfold candidates
  have row : edge 5189 = (SortedNatTree.contains FastRows.row5189) := by
    change Data.row5189.contains = (SortedNatTree.contains FastRows.row5189)
    exact FastRows.row5189_checked.symm
  rw [row]
  decide
theorem root59Step18 : scan edge (search edge 3) 4 [293] 293 root59Tail18 winner =
    scan edge (search edge 3) 4 [293] 293 root59Tail19 winner := by
  rw [root59Tail18, scan, if_neg (by decide), root59Candidates18_checked]
  change scan edge (search edge 3) 4 [293] 293 root59Tail19
    (if 3 ≤ root59Candidates18.length then search edge 3 [293, 5189] root59Candidates18 5482 winner else winner) = _
  rw [if_neg (by decide)]
def root59Candidates19 : List Nat := [9371]
set_option maxRecDepth 8192 in
theorem root59Candidates19_checked : candidates edge 4 293 5717 root59Tail20 winner.bound = root59Candidates19 := by
  unfold candidates
  have row : edge 5717 = (SortedNatTree.contains FastRows.row5717) := by
    change Data.row5717.contains = (SortedNatTree.contains FastRows.row5717)
    exact FastRows.row5717_checked.symm
  rw [row]
  decide
theorem root59Step19 : scan edge (search edge 3) 4 [293] 293 root59Tail19 winner =
    scan edge (search edge 3) 4 [293] 293 root59Tail20 winner := by
  rw [root59Tail19, scan, if_neg (by decide), root59Candidates19_checked]
  change scan edge (search edge 3) 4 [293] 293 root59Tail20
    (if 3 ≤ root59Candidates19.length then search edge 3 [293, 5717] root59Candidates19 6010 winner else winner) = _
  rw [if_neg (by decide)]
theorem root59Scan20 : scan edge (search edge 3) 4 [293] 293 root59Tail20 winner = winner := by rw [root59Tail20, scan, if_pos (by decide)]
theorem root59Scan19 : scan edge (search edge 3) 4 [293] 293 root59Tail19 winner = winner :=
  (root59Step19).trans root59Scan20
theorem root59Scan18 : scan edge (search edge 3) 4 [293] 293 root59Tail18 winner = winner :=
  (root59Step18).trans root59Scan19
theorem root59Scan17 : scan edge (search edge 3) 4 [293] 293 root59Tail17 winner = winner :=
  (root59Step17).trans root59Scan18
theorem root59Scan16 : scan edge (search edge 3) 4 [293] 293 root59Tail16 winner = winner :=
  (root59Step16).trans root59Scan17
theorem root59Scan15 : scan edge (search edge 3) 4 [293] 293 root59Tail15 winner = winner :=
  (root59Step15).trans root59Scan16
theorem root59Scan14 : scan edge (search edge 3) 4 [293] 293 root59Tail14 winner = winner :=
  (root59Step14).trans root59Scan15
theorem root59Scan13 : scan edge (search edge 3) 4 [293] 293 root59Tail13 winner = winner :=
  (root59Step13).trans root59Scan14
theorem root59Scan12 : scan edge (search edge 3) 4 [293] 293 root59Tail12 winner = winner :=
  (root59Step12).trans root59Scan13
theorem root59Scan11 : scan edge (search edge 3) 4 [293] 293 root59Tail11 winner = winner :=
  (root59Step11).trans root59Scan12
theorem root59Scan10 : scan edge (search edge 3) 4 [293] 293 root59Tail10 winner = winner :=
  (root59Step10).trans root59Scan11
theorem root59Scan9 : scan edge (search edge 3) 4 [293] 293 root59Tail9 winner = winner :=
  (root59Step9).trans root59Scan10
theorem root59Scan8 : scan edge (search edge 3) 4 [293] 293 root59Tail8 winner = winner :=
  (root59Step8).trans root59Scan9
theorem root59Scan7 : scan edge (search edge 3) 4 [293] 293 root59Tail7 winner = winner :=
  (root59Step7).trans root59Scan8
theorem root59Scan6 : scan edge (search edge 3) 4 [293] 293 root59Tail6 winner = winner :=
  (root59Step6).trans root59Scan7
theorem root59Scan5 : scan edge (search edge 3) 4 [293] 293 root59Tail5 winner = winner :=
  (root59Step5).trans root59Scan6
theorem root59Scan4 : scan edge (search edge 3) 4 [293] 293 root59Tail4 winner = winner :=
  (root59Step4).trans root59Scan5
theorem root59Scan3 : scan edge (search edge 3) 4 [293] 293 root59Tail3 winner = winner :=
  (root59Step3).trans root59Scan4
theorem root59Scan2 : scan edge (search edge 3) 4 [293] 293 root59Tail2 winner = winner :=
  (root59Step2).trans root59Scan3
theorem root59Scan1 : scan edge (search edge 3) 4 [293] 293 root59Tail1 winner = winner :=
  (root59Step1).trans root59Scan2
theorem root59Scan0 : scan edge (search edge 3) 4 [293] 293 root59Tail0 winner = winner :=
  (root59Step0).trans root59Scan1
set_option maxRecDepth 4096 in
theorem root59Branch : search edge 4 [293] root59Candidates 293 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [293] 293 root59Tail0 winner = winner
  exact root59Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
