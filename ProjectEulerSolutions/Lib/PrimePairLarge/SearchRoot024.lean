import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root24Candidates : List Nat := Data.row103
set_option maxRecDepth 8192 in
theorem root24Candidates_checked : candidates edge 5 0 103 tail25 winner.bound = root24Candidates := by
  apply candidates_eq_row edge 5 0 103 tail25 root24Candidates winner.bound tail25_ordered
  · decide
  · decide
  · funext q
    rfl
def root24Tail84 : List Nat := []
def root24Tail83 : List Nat := 25801 :: root24Tail84
def root24Tail82 : List Nat := 25423 :: root24Tail83
def root24Tail81 : List Nat := 23473 :: root24Tail82
def root24Tail80 : List Nat := 23227 :: root24Tail81
def root24Tail79 : List Nat := 23017 :: root24Tail80
def root24Tail78 : List Nat := 23011 :: root24Tail79
def root24Tail77 : List Nat := 22639 :: root24Tail78
def root24Tail76 : List Nat := 22441 :: root24Tail77
def root24Tail75 : List Nat := 22303 :: root24Tail76
def root24Tail74 : List Nat := 22189 :: root24Tail75
def root24Tail73 : List Nat := 22003 :: root24Tail74
def root24Tail72 : List Nat := 21841 :: root24Tail73
def root24Tail71 : List Nat := 21559 :: root24Tail72
def root24Tail70 : List Nat := 20341 :: root24Tail71
def root24Tail69 : List Nat := 20029 :: root24Tail70
def root24Tail68 : List Nat := 19891 :: root24Tail69
def root24Tail67 : List Nat := 19501 :: root24Tail68
def root24Tail66 : List Nat := 19333 :: root24Tail67
def root24Tail65 : List Nat := 18313 :: root24Tail66
def root24Tail64 : List Nat := 18013 :: root24Tail65
def root24Tail63 : List Nat := 17851 :: root24Tail64
def root24Tail62 : List Nat := 17467 :: root24Tail63
def root24Tail61 : List Nat := 17443 :: root24Tail62
def root24Tail60 : List Nat := 17401 :: root24Tail61
def root24Tail59 : List Nat := 17053 :: root24Tail60
def root24Tail58 : List Nat := 16993 :: root24Tail59
def root24Tail57 : List Nat := 16729 :: root24Tail58
def root24Tail56 : List Nat := 16519 :: root24Tail57
def root24Tail55 : List Nat := 16477 :: root24Tail56
def root24Tail54 : List Nat := 16363 :: root24Tail55
def root24Tail53 : List Nat := 15919 :: root24Tail54
def root24Tail52 : List Nat := 15031 :: root24Tail53
def root24Tail51 : List Nat := 14683 :: root24Tail52
def root24Tail50 : List Nat := 14431 :: root24Tail51
def root24Tail49 : List Nat := 13267 :: root24Tail50
def root24Tail48 : List Nat := 12979 :: root24Tail49
def root24Tail47 : List Nat := 12919 :: root24Tail48
def root24Tail46 : List Nat := 12301 :: root24Tail47
def root24Tail45 : List Nat := 12007 :: root24Tail46
def root24Tail44 : List Nat := 11953 :: root24Tail45
def root24Tail43 : List Nat := 11731 :: root24Tail44
def root24Tail42 : List Nat := 11593 :: root24Tail43
def root24Tail41 : List Nat := 11551 :: root24Tail42
def root24Tail40 : List Nat := 11467 :: root24Tail41
def root24Tail39 : List Nat := 11083 :: root24Tail40
def root24Tail38 : List Nat := 10957 :: root24Tail39
def root24Tail37 : List Nat := 10357 :: root24Tail38
def root24Tail36 : List Nat := 10039 :: root24Tail37
def root24Tail35 : List Nat := 9901 :: root24Tail36
def root24Tail34 : List Nat := 9817 :: root24Tail35
def root24Tail33 : List Nat := 9187 :: root24Tail34
def root24Tail32 : List Nat := 9127 :: root24Tail33
def root24Tail31 : List Nat := 8803 :: root24Tail32
def root24Tail30 : List Nat := 8731 :: root24Tail31
def root24Tail29 : List Nat := 8707 :: root24Tail30
def root24Tail28 : List Nat := 8623 :: root24Tail29
def root24Tail27 : List Nat := 8329 :: root24Tail28
def root24Tail26 : List Nat := 8269 :: root24Tail27
def root24Tail25 : List Nat := 7879 :: root24Tail26
def root24Tail24 : List Nat := 7489 :: root24Tail25
def root24Tail23 : List Nat := 6661 :: root24Tail24
def root24Tail22 : List Nat := 6247 :: root24Tail23
def root24Tail21 : List Nat := 6229 :: root24Tail22
def root24Tail20 : List Nat := 5869 :: root24Tail21
def root24Tail19 : List Nat := 5527 :: root24Tail20
def root24Tail18 : List Nat := 5413 :: root24Tail19
def root24Tail17 : List Nat := 4567 :: root24Tail18
def root24Tail16 : List Nat := 4357 :: root24Tail17
def root24Tail15 : List Nat := 4339 :: root24Tail16
def root24Tail14 : List Nat := 4027 :: root24Tail15
def root24Tail13 : List Nat := 3061 :: root24Tail14
def root24Tail12 : List Nat := 2803 :: root24Tail13
def root24Tail11 : List Nat := 2683 :: root24Tail12
def root24Tail10 : List Nat := 2617 :: root24Tail11
def root24Tail9 : List Nat := 2467 :: root24Tail10
def root24Tail8 : List Nat := 2341 :: root24Tail9
def root24Tail7 : List Nat := 1999 :: root24Tail8
def root24Tail6 : List Nat := 1831 :: root24Tail7
def root24Tail5 : List Nat := 1759 :: root24Tail6
def root24Tail4 : List Nat := 1669 :: root24Tail5
def root24Tail3 : List Nat := 1483 :: root24Tail4
def root24Tail2 : List Nat := 997 :: root24Tail3
def root24Tail1 : List Nat := 421 :: root24Tail2
def root24Tail0 : List Nat := 307 :: root24Tail1
def root24Candidates0 : List Nat := [2467, 18313]
set_option maxRecDepth 8192 in
theorem root24Candidates0_checked : candidates edge 4 103 307 root24Tail1 winner.bound = root24Candidates0 := by
  unfold candidates
  have row : edge 307 = (SortedNatTree.contains FastRows.row307) := by
    change Data.row307.contains = (SortedNatTree.contains FastRows.row307)
    exact FastRows.row307_checked.symm
  rw [row]
  decide
theorem root24Step0 : scan edge (search edge 3) 4 [103] 103 root24Tail0 winner =
    scan edge (search edge 3) 4 [103] 103 root24Tail1 winner := by
  rw [root24Tail0, scan, if_neg (by decide), root24Candidates0_checked]
  change scan edge (search edge 3) 4 [103] 103 root24Tail1
    (if 3 ≤ root24Candidates0.length then search edge 3 [103, 307] root24Candidates0 410 winner else winner) = _
  rw [if_neg (by decide)]
def root24Candidates1 : List Nat := [8707]
set_option maxRecDepth 8192 in
theorem root24Candidates1_checked : candidates edge 4 103 421 root24Tail2 winner.bound = root24Candidates1 := by
  unfold candidates
  have row : edge 421 = (SortedNatTree.contains FastRows.row421) := by
    change Data.row421.contains = (SortedNatTree.contains FastRows.row421)
    exact FastRows.row421_checked.symm
  rw [row]
  decide
theorem root24Step1 : scan edge (search edge 3) 4 [103] 103 root24Tail1 winner =
    scan edge (search edge 3) 4 [103] 103 root24Tail2 winner := by
  rw [root24Tail1, scan, if_neg (by decide), root24Candidates1_checked]
  change scan edge (search edge 3) 4 [103] 103 root24Tail2
    (if 3 ≤ root24Candidates1.length then search edge 3 [103, 421] root24Candidates1 524 winner else winner) = _
  rw [if_neg (by decide)]
def root24Candidates2 : List Nat := [1483, 2617, 11083, 17467, 18013, 23011, 23017]
set_option maxRecDepth 8192 in
theorem root24Candidates2_checked : candidates edge 4 103 997 root24Tail3 winner.bound = root24Candidates2 := by
  unfold candidates
  have row : edge 997 = (SortedNatTree.contains FastRows.row997) := by
    change Data.row997.contains = (SortedNatTree.contains FastRows.row997)
    exact FastRows.row997_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root24Child2 : search edge 3 [103, 997] root24Candidates2 1100 winner = winner := by decide
theorem root24Step2 : scan edge (search edge 3) 4 [103] 103 root24Tail2 winner =
    scan edge (search edge 3) 4 [103] 103 root24Tail3 winner := by
  rw [root24Tail2, scan, if_neg (by decide), root24Candidates2_checked]
  change scan edge (search edge 3) 4 [103] 103 root24Tail3
    (if 3 ≤ root24Candidates2.length then search edge 3 [103, 997] root24Candidates2 1100 winner else winner) = _
  rw [if_pos (by decide), root24Child2]
def root24Candidates3 : List Nat := [2341, 4357, 5869, 9127, 12979]
set_option maxRecDepth 8192 in
theorem root24Candidates3_checked : candidates edge 4 103 1483 root24Tail4 winner.bound = root24Candidates3 := by
  unfold candidates
  have row : edge 1483 = (SortedNatTree.contains FastRows.row1483) := by
    change Data.row1483.contains = (SortedNatTree.contains FastRows.row1483)
    exact FastRows.row1483_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root24Child3 : search edge 3 [103, 1483] root24Candidates3 1586 winner = winner := by decide
theorem root24Step3 : scan edge (search edge 3) 4 [103] 103 root24Tail3 winner =
    scan edge (search edge 3) 4 [103] 103 root24Tail4 winner := by
  rw [root24Tail3, scan, if_neg (by decide), root24Candidates3_checked]
  change scan edge (search edge 3) 4 [103] 103 root24Tail4
    (if 3 ≤ root24Candidates3.length then search edge 3 [103, 1483] root24Candidates3 1586 winner else winner) = _
  rw [if_pos (by decide), root24Child3]
def root24Candidates4 : List Nat := [6661, 20029]
set_option maxRecDepth 8192 in
theorem root24Candidates4_checked : candidates edge 4 103 1669 root24Tail5 winner.bound = root24Candidates4 := by
  unfold candidates
  have row : edge 1669 = (SortedNatTree.contains FastRows.row1669) := by
    change Data.row1669.contains = (SortedNatTree.contains FastRows.row1669)
    exact FastRows.row1669_checked.symm
  rw [row]
  decide
theorem root24Step4 : scan edge (search edge 3) 4 [103] 103 root24Tail4 winner =
    scan edge (search edge 3) 4 [103] 103 root24Tail5 winner := by
  rw [root24Tail4, scan, if_neg (by decide), root24Candidates4_checked]
  change scan edge (search edge 3) 4 [103] 103 root24Tail5
    (if 3 ≤ root24Candidates4.length then search edge 3 [103, 1669] root24Candidates4 1772 winner else winner) = _
  rw [if_neg (by decide)]
def root24Candidates5 : List Nat := [1831, 16729]
set_option maxRecDepth 8192 in
theorem root24Candidates5_checked : candidates edge 4 103 1759 root24Tail6 winner.bound = root24Candidates5 := by
  unfold candidates
  have row : edge 1759 = (SortedNatTree.contains FastRows.row1759) := by
    change Data.row1759.contains = (SortedNatTree.contains FastRows.row1759)
    exact FastRows.row1759_checked.symm
  rw [row]
  decide
theorem root24Step5 : scan edge (search edge 3) 4 [103] 103 root24Tail5 winner =
    scan edge (search edge 3) 4 [103] 103 root24Tail6 winner := by
  rw [root24Tail5, scan, if_neg (by decide), root24Candidates5_checked]
  change scan edge (search edge 3) 4 [103] 103 root24Tail6
    (if 3 ≤ root24Candidates5.length then search edge 3 [103, 1759] root24Candidates5 1862 winner else winner) = _
  rw [if_neg (by decide)]
def root24Candidates6 : List Nat := [2803, 20029]
set_option maxRecDepth 8192 in
theorem root24Candidates6_checked : candidates edge 4 103 1831 root24Tail7 winner.bound = root24Candidates6 := by
  unfold candidates
  have row : edge 1831 = (SortedNatTree.contains FastRows.row1831) := by
    change Data.row1831.contains = (SortedNatTree.contains FastRows.row1831)
    exact FastRows.row1831_checked.symm
  rw [row]
  decide
theorem root24Step6 : scan edge (search edge 3) 4 [103] 103 root24Tail6 winner =
    scan edge (search edge 3) 4 [103] 103 root24Tail7 winner := by
  rw [root24Tail6, scan, if_neg (by decide), root24Candidates6_checked]
  change scan edge (search edge 3) 4 [103] 103 root24Tail7
    (if 3 ≤ root24Candidates6.length then search edge 3 [103, 1831] root24Candidates6 1934 winner else winner) = _
  rw [if_neg (by decide)]
def root24Candidates7 : List Nat := [5869, 13267]
set_option maxRecDepth 8192 in
theorem root24Candidates7_checked : candidates edge 4 103 1999 root24Tail8 winner.bound = root24Candidates7 := by
  unfold candidates
  have row : edge 1999 = (SortedNatTree.contains FastRows.row1999) := by
    change Data.row1999.contains = (SortedNatTree.contains FastRows.row1999)
    exact FastRows.row1999_checked.symm
  rw [row]
  decide
theorem root24Step7 : scan edge (search edge 3) 4 [103] 103 root24Tail7 winner =
    scan edge (search edge 3) 4 [103] 103 root24Tail8 winner := by
  rw [root24Tail7, scan, if_neg (by decide), root24Candidates7_checked]
  change scan edge (search edge 3) 4 [103] 103 root24Tail8
    (if 3 ≤ root24Candidates7.length then search edge 3 [103, 1999] root24Candidates7 2102 winner else winner) = _
  rw [if_neg (by decide)]
def root24Candidates8 : List Nat := [7489, 7879, 11083, 11953]
set_option maxRecDepth 8192 in
theorem root24Candidates8_checked : candidates edge 4 103 2341 root24Tail9 winner.bound = root24Candidates8 := by
  unfold candidates
  have row : edge 2341 = (SortedNatTree.contains FastRows.row2341) := by
    change Data.row2341.contains = (SortedNatTree.contains FastRows.row2341)
    exact FastRows.row2341_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root24Child8 : search edge 3 [103, 2341] root24Candidates8 2444 winner = winner := by decide
theorem root24Step8 : scan edge (search edge 3) 4 [103] 103 root24Tail8 winner =
    scan edge (search edge 3) 4 [103] 103 root24Tail9 winner := by
  rw [root24Tail8, scan, if_neg (by decide), root24Candidates8_checked]
  change scan edge (search edge 3) 4 [103] 103 root24Tail9
    (if 3 ≤ root24Candidates8.length then search edge 3 [103, 2341] root24Candidates8 2444 winner else winner) = _
  rw [if_pos (by decide), root24Child8]
def root24Candidates9 : List Nat := [10957, 12919]
set_option maxRecDepth 8192 in
theorem root24Candidates9_checked : candidates edge 4 103 2467 root24Tail10 winner.bound = root24Candidates9 := by
  unfold candidates
  have row : edge 2467 = (SortedNatTree.contains FastRows.row2467) := by
    change Data.row2467.contains = (SortedNatTree.contains FastRows.row2467)
    exact FastRows.row2467_checked.symm
  rw [row]
  decide
theorem root24Step9 : scan edge (search edge 3) 4 [103] 103 root24Tail9 winner =
    scan edge (search edge 3) 4 [103] 103 root24Tail10 winner := by
  rw [root24Tail9, scan, if_neg (by decide), root24Candidates9_checked]
  change scan edge (search edge 3) 4 [103] 103 root24Tail10
    (if 3 ≤ root24Candidates9.length then search edge 3 [103, 2467] root24Candidates9 2570 winner else winner) = _
  rw [if_neg (by decide)]
def root24Candidates10 : List Nat := [2803, 4567, 9817, 10357, 16519, 16993, 19333]
set_option maxRecDepth 8192 in
theorem root24Candidates10_checked : candidates edge 4 103 2617 root24Tail11 winner.bound = root24Candidates10 := by
  unfold candidates
  have row : edge 2617 = (SortedNatTree.contains FastRows.row2617) := by
    change Data.row2617.contains = (SortedNatTree.contains FastRows.row2617)
    exact FastRows.row2617_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root24Child10 : search edge 3 [103, 2617] root24Candidates10 2720 winner = winner := by decide
theorem root24Step10 : scan edge (search edge 3) 4 [103] 103 root24Tail10 winner =
    scan edge (search edge 3) 4 [103] 103 root24Tail11 winner := by
  rw [root24Tail10, scan, if_neg (by decide), root24Candidates10_checked]
  change scan edge (search edge 3) 4 [103] 103 root24Tail11
    (if 3 ≤ root24Candidates10.length then search edge 3 [103, 2617] root24Candidates10 2720 winner else winner) = _
  rw [if_pos (by decide), root24Child10]
def root24Candidates11 : List Nat := [8803, 11467, 22189]
set_option maxRecDepth 8192 in
theorem root24Candidates11_checked : candidates edge 4 103 2683 root24Tail12 winner.bound = root24Candidates11 := by
  unfold candidates
  have row : edge 2683 = (SortedNatTree.contains FastRows.row2683) := by
    change Data.row2683.contains = (SortedNatTree.contains FastRows.row2683)
    exact FastRows.row2683_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root24Child11 : search edge 3 [103, 2683] root24Candidates11 2786 winner = winner := by decide
theorem root24Step11 : scan edge (search edge 3) 4 [103] 103 root24Tail11 winner =
    scan edge (search edge 3) 4 [103] 103 root24Tail12 winner := by
  rw [root24Tail11, scan, if_neg (by decide), root24Candidates11_checked]
  change scan edge (search edge 3) 4 [103] 103 root24Tail12
    (if 3 ≤ root24Candidates11.length then search edge 3 [103, 2683] root24Candidates11 2786 winner else winner) = _
  rw [if_pos (by decide), root24Child11]
def root24Candidates12 : List Nat := [9127, 15031]
set_option maxRecDepth 8192 in
theorem root24Candidates12_checked : candidates edge 4 103 2803 root24Tail13 winner.bound = root24Candidates12 := by
  unfold candidates
  have row : edge 2803 = (SortedNatTree.contains FastRows.row2803) := by
    change Data.row2803.contains = (SortedNatTree.contains FastRows.row2803)
    exact FastRows.row2803_checked.symm
  rw [row]
  decide
theorem root24Step12 : scan edge (search edge 3) 4 [103] 103 root24Tail12 winner =
    scan edge (search edge 3) 4 [103] 103 root24Tail13 winner := by
  rw [root24Tail12, scan, if_neg (by decide), root24Candidates12_checked]
  change scan edge (search edge 3) 4 [103] 103 root24Tail13
    (if 3 ≤ root24Candidates12.length then search edge 3 [103, 2803] root24Candidates12 2906 winner else winner) = _
  rw [if_neg (by decide)]
def root24Candidates13 : List Nat := [15919]
set_option maxRecDepth 8192 in
theorem root24Candidates13_checked : candidates edge 4 103 3061 root24Tail14 winner.bound = root24Candidates13 := by
  unfold candidates
  have row : edge 3061 = (SortedNatTree.contains FastRows.row3061) := by
    change Data.row3061.contains = (SortedNatTree.contains FastRows.row3061)
    exact FastRows.row3061_checked.symm
  rw [row]
  decide
theorem root24Step13 : scan edge (search edge 3) 4 [103] 103 root24Tail13 winner =
    scan edge (search edge 3) 4 [103] 103 root24Tail14 winner := by
  rw [root24Tail13, scan, if_neg (by decide), root24Candidates13_checked]
  change scan edge (search edge 3) 4 [103] 103 root24Tail14
    (if 3 ≤ root24Candidates13.length then search edge 3 [103, 3061] root24Candidates13 3164 winner else winner) = _
  rw [if_neg (by decide)]
def root24Candidates14 : List Nat := [5413]
set_option maxRecDepth 8192 in
theorem root24Candidates14_checked : candidates edge 4 103 4027 root24Tail15 winner.bound = root24Candidates14 := by
  unfold candidates
  have row : edge 4027 = (SortedNatTree.contains FastRows.row4027) := by
    change Data.row4027.contains = (SortedNatTree.contains FastRows.row4027)
    exact FastRows.row4027_checked.symm
  rw [row]
  decide
theorem root24Step14 : scan edge (search edge 3) 4 [103] 103 root24Tail14 winner =
    scan edge (search edge 3) 4 [103] 103 root24Tail15 winner := by
  rw [root24Tail14, scan, if_neg (by decide), root24Candidates14_checked]
  change scan edge (search edge 3) 4 [103] 103 root24Tail15
    (if 3 ≤ root24Candidates14.length then search edge 3 [103, 4027] root24Candidates14 4130 winner else winner) = _
  rw [if_neg (by decide)]
def root24Candidates15 : List Nat := [4567, 21559]
set_option maxRecDepth 8192 in
theorem root24Candidates15_checked : candidates edge 4 103 4339 root24Tail16 winner.bound = root24Candidates15 := by
  unfold candidates
  have row : edge 4339 = (SortedNatTree.contains FastRows.row4339) := by
    change Data.row4339.contains = (SortedNatTree.contains FastRows.row4339)
    exact FastRows.row4339_checked.symm
  rw [row]
  decide
theorem root24Step15 : scan edge (search edge 3) 4 [103] 103 root24Tail15 winner =
    scan edge (search edge 3) 4 [103] 103 root24Tail16 winner := by
  rw [root24Tail15, scan, if_neg (by decide), root24Candidates15_checked]
  change scan edge (search edge 3) 4 [103] 103 root24Tail16
    (if 3 ≤ root24Candidates15.length then search edge 3 [103, 4339] root24Candidates15 4442 winner else winner) = _
  rw [if_neg (by decide)]
def root24Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root24Candidates16_checked : candidates edge 4 103 4357 root24Tail17 winner.bound = root24Candidates16 := by
  unfold candidates
  have row : edge 4357 = (SortedNatTree.contains FastRows.row4357) := by
    change Data.row4357.contains = (SortedNatTree.contains FastRows.row4357)
    exact FastRows.row4357_checked.symm
  rw [row]
  decide
theorem root24Step16 : scan edge (search edge 3) 4 [103] 103 root24Tail16 winner =
    scan edge (search edge 3) 4 [103] 103 root24Tail17 winner := by
  rw [root24Tail16, scan, if_neg (by decide), root24Candidates16_checked]
  change scan edge (search edge 3) 4 [103] 103 root24Tail17
    (if 3 ≤ root24Candidates16.length then search edge 3 [103, 4357] root24Candidates16 4460 winner else winner) = _
  rw [if_neg (by decide)]
def root24Candidates17 : List Nat := [17467]
set_option maxRecDepth 8192 in
theorem root24Candidates17_checked : candidates edge 4 103 4567 root24Tail18 winner.bound = root24Candidates17 := by
  unfold candidates
  have row : edge 4567 = (SortedNatTree.contains FastRows.row4567) := by
    change Data.row4567.contains = (SortedNatTree.contains FastRows.row4567)
    exact FastRows.row4567_checked.symm
  rw [row]
  decide
theorem root24Step17 : scan edge (search edge 3) 4 [103] 103 root24Tail17 winner =
    scan edge (search edge 3) 4 [103] 103 root24Tail18 winner := by
  rw [root24Tail17, scan, if_neg (by decide), root24Candidates17_checked]
  change scan edge (search edge 3) 4 [103] 103 root24Tail18
    (if 3 ≤ root24Candidates17.length then search edge 3 [103, 4567] root24Candidates17 4670 winner else winner) = _
  rw [if_neg (by decide)]
def root24Candidates18 : List Nat := [5527, 18313]
set_option maxRecDepth 8192 in
theorem root24Candidates18_checked : candidates edge 4 103 5413 root24Tail19 winner.bound = root24Candidates18 := by
  unfold candidates
  have row : edge 5413 = (SortedNatTree.contains FastRows.row5413) := by
    change Data.row5413.contains = (SortedNatTree.contains FastRows.row5413)
    exact FastRows.row5413_checked.symm
  rw [row]
  decide
theorem root24Step18 : scan edge (search edge 3) 4 [103] 103 root24Tail18 winner =
    scan edge (search edge 3) 4 [103] 103 root24Tail19 winner := by
  rw [root24Tail18, scan, if_neg (by decide), root24Candidates18_checked]
  change scan edge (search edge 3) 4 [103] 103 root24Tail19
    (if 3 ≤ root24Candidates18.length then search edge 3 [103, 5413] root24Candidates18 5516 winner else winner) = _
  rw [if_neg (by decide)]
def root24Candidates19 : List Nat := []
set_option maxRecDepth 8192 in
theorem root24Candidates19_checked : candidates edge 4 103 5527 root24Tail20 winner.bound = root24Candidates19 := by
  unfold candidates
  have row : edge 5527 = (SortedNatTree.contains FastRows.row5527) := by
    change Data.row5527.contains = (SortedNatTree.contains FastRows.row5527)
    exact FastRows.row5527_checked.symm
  rw [row]
  decide
theorem root24Step19 : scan edge (search edge 3) 4 [103] 103 root24Tail19 winner =
    scan edge (search edge 3) 4 [103] 103 root24Tail20 winner := by
  rw [root24Tail19, scan, if_neg (by decide), root24Candidates19_checked]
  change scan edge (search edge 3) 4 [103] 103 root24Tail20
    (if 3 ≤ root24Candidates19.length then search edge 3 [103, 5527] root24Candidates19 5630 winner else winner) = _
  rw [if_neg (by decide)]
def root24Candidates20 : List Nat := []
set_option maxRecDepth 8192 in
theorem root24Candidates20_checked : candidates edge 4 103 5869 root24Tail21 winner.bound = root24Candidates20 := by
  unfold candidates
  have row : edge 5869 = (SortedNatTree.contains FastRows.row5869) := by
    change Data.row5869.contains = (SortedNatTree.contains FastRows.row5869)
    exact FastRows.row5869_checked.symm
  rw [row]
  decide
theorem root24Step20 : scan edge (search edge 3) 4 [103] 103 root24Tail20 winner =
    scan edge (search edge 3) 4 [103] 103 root24Tail21 winner := by
  rw [root24Tail20, scan, if_neg (by decide), root24Candidates20_checked]
  change scan edge (search edge 3) 4 [103] 103 root24Tail21
    (if 3 ≤ root24Candidates20.length then search edge 3 [103, 5869] root24Candidates20 5972 winner else winner) = _
  rw [if_neg (by decide)]
def root24Candidates21 : List Nat := [11953]
set_option maxRecDepth 8192 in
theorem root24Candidates21_checked : candidates edge 4 103 6229 root24Tail22 winner.bound = root24Candidates21 := by
  unfold candidates
  have row : edge 6229 = (SortedNatTree.contains FastRows.row6229) := by
    change Data.row6229.contains = (SortedNatTree.contains FastRows.row6229)
    exact FastRows.row6229_checked.symm
  rw [row]
  decide
theorem root24Step21 : scan edge (search edge 3) 4 [103] 103 root24Tail21 winner =
    scan edge (search edge 3) 4 [103] 103 root24Tail22 winner := by
  rw [root24Tail21, scan, if_neg (by decide), root24Candidates21_checked]
  change scan edge (search edge 3) 4 [103] 103 root24Tail22
    (if 3 ≤ root24Candidates21.length then search edge 3 [103, 6229] root24Candidates21 6332 winner else winner) = _
  rw [if_neg (by decide)]
def root24Candidates22 : List Nat := [7879, 9187]
set_option maxRecDepth 8192 in
theorem root24Candidates22_checked : candidates edge 4 103 6247 root24Tail23 winner.bound = root24Candidates22 := by
  unfold candidates
  have row : edge 6247 = (SortedNatTree.contains FastRows.row6247) := by
    change Data.row6247.contains = (SortedNatTree.contains FastRows.row6247)
    exact FastRows.row6247_checked.symm
  rw [row]
  decide
theorem root24Step22 : scan edge (search edge 3) 4 [103] 103 root24Tail22 winner =
    scan edge (search edge 3) 4 [103] 103 root24Tail23 winner := by
  rw [root24Tail22, scan, if_neg (by decide), root24Candidates22_checked]
  change scan edge (search edge 3) 4 [103] 103 root24Tail23
    (if 3 ≤ root24Candidates22.length then search edge 3 [103, 6247] root24Candidates22 6350 winner else winner) = _
  rw [if_neg (by decide)]
theorem root24Scan23 : scan edge (search edge 3) 4 [103] 103 root24Tail23 winner = winner := by rw [root24Tail23, scan, if_pos (by decide)]
theorem root24Scan22 : scan edge (search edge 3) 4 [103] 103 root24Tail22 winner = winner :=
  (root24Step22).trans root24Scan23
theorem root24Scan21 : scan edge (search edge 3) 4 [103] 103 root24Tail21 winner = winner :=
  (root24Step21).trans root24Scan22
theorem root24Scan20 : scan edge (search edge 3) 4 [103] 103 root24Tail20 winner = winner :=
  (root24Step20).trans root24Scan21
theorem root24Scan19 : scan edge (search edge 3) 4 [103] 103 root24Tail19 winner = winner :=
  (root24Step19).trans root24Scan20
theorem root24Scan18 : scan edge (search edge 3) 4 [103] 103 root24Tail18 winner = winner :=
  (root24Step18).trans root24Scan19
theorem root24Scan17 : scan edge (search edge 3) 4 [103] 103 root24Tail17 winner = winner :=
  (root24Step17).trans root24Scan18
theorem root24Scan16 : scan edge (search edge 3) 4 [103] 103 root24Tail16 winner = winner :=
  (root24Step16).trans root24Scan17
theorem root24Scan15 : scan edge (search edge 3) 4 [103] 103 root24Tail15 winner = winner :=
  (root24Step15).trans root24Scan16
theorem root24Scan14 : scan edge (search edge 3) 4 [103] 103 root24Tail14 winner = winner :=
  (root24Step14).trans root24Scan15
theorem root24Scan13 : scan edge (search edge 3) 4 [103] 103 root24Tail13 winner = winner :=
  (root24Step13).trans root24Scan14
theorem root24Scan12 : scan edge (search edge 3) 4 [103] 103 root24Tail12 winner = winner :=
  (root24Step12).trans root24Scan13
theorem root24Scan11 : scan edge (search edge 3) 4 [103] 103 root24Tail11 winner = winner :=
  (root24Step11).trans root24Scan12
theorem root24Scan10 : scan edge (search edge 3) 4 [103] 103 root24Tail10 winner = winner :=
  (root24Step10).trans root24Scan11
theorem root24Scan9 : scan edge (search edge 3) 4 [103] 103 root24Tail9 winner = winner :=
  (root24Step9).trans root24Scan10
theorem root24Scan8 : scan edge (search edge 3) 4 [103] 103 root24Tail8 winner = winner :=
  (root24Step8).trans root24Scan9
theorem root24Scan7 : scan edge (search edge 3) 4 [103] 103 root24Tail7 winner = winner :=
  (root24Step7).trans root24Scan8
theorem root24Scan6 : scan edge (search edge 3) 4 [103] 103 root24Tail6 winner = winner :=
  (root24Step6).trans root24Scan7
theorem root24Scan5 : scan edge (search edge 3) 4 [103] 103 root24Tail5 winner = winner :=
  (root24Step5).trans root24Scan6
theorem root24Scan4 : scan edge (search edge 3) 4 [103] 103 root24Tail4 winner = winner :=
  (root24Step4).trans root24Scan5
theorem root24Scan3 : scan edge (search edge 3) 4 [103] 103 root24Tail3 winner = winner :=
  (root24Step3).trans root24Scan4
theorem root24Scan2 : scan edge (search edge 3) 4 [103] 103 root24Tail2 winner = winner :=
  (root24Step2).trans root24Scan3
theorem root24Scan1 : scan edge (search edge 3) 4 [103] 103 root24Tail1 winner = winner :=
  (root24Step1).trans root24Scan2
theorem root24Scan0 : scan edge (search edge 3) 4 [103] 103 root24Tail0 winner = winner :=
  (root24Step0).trans root24Scan1
set_option maxRecDepth 4096 in
theorem root24Branch : search edge 4 [103] root24Candidates 103 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [103] 103 root24Tail0 winner = winner
  exact root24Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
