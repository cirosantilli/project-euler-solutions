import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root63Candidates : List Nat := Data.row317
set_option maxRecDepth 8192 in
theorem root63Candidates_checked : candidates edge 5 0 317 tail64 winner.bound = root63Candidates := by
  apply candidates_eq_row edge 5 0 317 tail64 root63Candidates winner.bound tail64_ordered
  · decide
  · decide
  · funext q
    rfl
def root63Tail83 : List Nat := []
def root63Tail82 : List Nat := 24137 :: root63Tail83
def root63Tail81 : List Nat := 23663 :: root63Tail82
def root63Tail80 : List Nat := 22907 :: root63Tail81
def root63Tail79 : List Nat := 22811 :: root63Tail80
def root63Tail78 : List Nat := 22739 :: root63Tail79
def root63Tail77 : List Nat := 22721 :: root63Tail78
def root63Tail76 : List Nat := 22637 :: root63Tail77
def root63Tail75 : List Nat := 22349 :: root63Tail76
def root63Tail74 : List Nat := 22247 :: root63Tail75
def root63Tail73 : List Nat := 22193 :: root63Tail74
def root63Tail72 : List Nat := 21467 :: root63Tail73
def root63Tail71 : List Nat := 21419 :: root63Tail72
def root63Tail70 : List Nat := 21407 :: root63Tail71
def root63Tail69 : List Nat := 21383 :: root63Tail70
def root63Tail68 : List Nat := 21341 :: root63Tail69
def root63Tail67 : List Nat := 21017 :: root63Tail68
def root63Tail66 : List Nat := 20873 :: root63Tail67
def root63Tail65 : List Nat := 20849 :: root63Tail66
def root63Tail64 : List Nat := 20543 :: root63Tail65
def root63Tail63 : List Nat := 19559 :: root63Tail64
def root63Tail62 : List Nat := 19319 :: root63Tail63
def root63Tail61 : List Nat := 19013 :: root63Tail62
def root63Tail60 : List Nat := 18593 :: root63Tail61
def root63Tail59 : List Nat := 18371 :: root63Tail60
def root63Tail58 : List Nat := 18191 :: root63Tail59
def root63Tail57 : List Nat := 17981 :: root63Tail58
def root63Tail56 : List Nat := 17489 :: root63Tail57
def root63Tail55 : List Nat := 16649 :: root63Tail56
def root63Tail54 : List Nat := 16427 :: root63Tail55
def root63Tail53 : List Nat := 15881 :: root63Tail54
def root63Tail52 : List Nat := 15107 :: root63Tail53
def root63Tail51 : List Nat := 14939 :: root63Tail52
def root63Tail50 : List Nat := 14153 :: root63Tail51
def root63Tail49 : List Nat := 13763 :: root63Tail50
def root63Tail48 : List Nat := 12809 :: root63Tail49
def root63Tail47 : List Nat := 12503 :: root63Tail48
def root63Tail46 : List Nat := 12347 :: root63Tail47
def root63Tail45 : List Nat := 12143 :: root63Tail46
def root63Tail44 : List Nat := 12011 :: root63Tail45
def root63Tail43 : List Nat := 11777 :: root63Tail44
def root63Tail42 : List Nat := 11717 :: root63Tail43
def root63Tail41 : List Nat := 11633 :: root63Tail42
def root63Tail40 : List Nat := 11423 :: root63Tail41
def root63Tail39 : List Nat := 10937 :: root63Tail40
def root63Tail38 : List Nat := 10391 :: root63Tail39
def root63Tail37 : List Nat := 10331 :: root63Tail38
def root63Tail36 : List Nat := 10163 :: root63Tail37
def root63Tail35 : List Nat := 10139 :: root63Tail36
def root63Tail34 : List Nat := 9941 :: root63Tail35
def root63Tail33 : List Nat := 9749 :: root63Tail34
def root63Tail32 : List Nat := 9257 :: root63Tail33
def root63Tail31 : List Nat := 9173 :: root63Tail32
def root63Tail30 : List Nat := 8999 :: root63Tail31
def root63Tail29 : List Nat := 8543 :: root63Tail30
def root63Tail28 : List Nat := 7949 :: root63Tail29
def root63Tail27 : List Nat := 7607 :: root63Tail28
def root63Tail26 : List Nat := 6221 :: root63Tail27
def root63Tail25 : List Nat := 6053 :: root63Tail26
def root63Tail24 : List Nat := 5801 :: root63Tail25
def root63Tail23 : List Nat := 5099 :: root63Tail24
def root63Tail22 : List Nat := 5009 :: root63Tail23
def root63Tail21 : List Nat := 4973 :: root63Tail22
def root63Tail20 : List Nat := 4799 :: root63Tail21
def root63Tail19 : List Nat := 4793 :: root63Tail20
def root63Tail18 : List Nat := 4091 :: root63Tail19
def root63Tail17 : List Nat := 3659 :: root63Tail18
def root63Tail16 : List Nat := 3539 :: root63Tail17
def root63Tail15 : List Nat := 3119 :: root63Tail16
def root63Tail14 : List Nat := 2801 :: root63Tail15
def root63Tail13 : List Nat := 2441 :: root63Tail14
def root63Tail12 : List Nat := 2399 :: root63Tail13
def root63Tail11 : List Nat := 2243 :: root63Tail12
def root63Tail10 : List Nat := 2237 :: root63Tail11
def root63Tail9 : List Nat := 2027 :: root63Tail10
def root63Tail8 : List Nat := 1709 :: root63Tail9
def root63Tail7 : List Nat := 1697 :: root63Tail8
def root63Tail6 : List Nat := 1601 :: root63Tail7
def root63Tail5 : List Nat := 1187 :: root63Tail6
def root63Tail4 : List Nat := 983 :: root63Tail5
def root63Tail3 : List Nat := 773 :: root63Tail4
def root63Tail2 : List Nat := 503 :: root63Tail3
def root63Tail1 : List Nat := 419 :: root63Tail2
def root63Tail0 : List Nat := 353 :: root63Tail1
def root63Candidates0 : List Nat := [12143]
set_option maxRecDepth 8192 in
theorem root63Candidates0_checked : candidates edge 4 317 353 root63Tail1 winner.bound = root63Candidates0 := by
  unfold candidates
  have row : edge 353 = (SortedNatTree.contains FastRows.row353) := by
    change Data.row353.contains = (SortedNatTree.contains FastRows.row353)
    exact FastRows.row353_checked.symm
  rw [row]
  decide
theorem root63Step0 : scan edge (search edge 3) 4 [317] 317 root63Tail0 winner =
    scan edge (search edge 3) 4 [317] 317 root63Tail1 winner := by
  rw [root63Tail0, scan, if_neg (by decide), root63Candidates0_checked]
  change scan edge (search edge 3) 4 [317] 317 root63Tail1
    (if 3 ≤ root63Candidates0.length then search edge 3 [317, 353] root63Candidates0 670 winner else winner) = _
  rw [if_neg (by decide)]
def root63Candidates1 : List Nat := [9257, 21017]
set_option maxRecDepth 8192 in
theorem root63Candidates1_checked : candidates edge 4 317 419 root63Tail2 winner.bound = root63Candidates1 := by
  unfold candidates
  have row : edge 419 = (SortedNatTree.contains FastRows.row419) := by
    change Data.row419.contains = (SortedNatTree.contains FastRows.row419)
    exact FastRows.row419_checked.symm
  rw [row]
  decide
theorem root63Step1 : scan edge (search edge 3) 4 [317] 317 root63Tail1 winner =
    scan edge (search edge 3) 4 [317] 317 root63Tail2 winner := by
  rw [root63Tail1, scan, if_neg (by decide), root63Candidates1_checked]
  change scan edge (search edge 3) 4 [317] 317 root63Tail2
    (if 3 ≤ root63Candidates1.length then search edge 3 [317, 419] root63Candidates1 736 winner else winner) = _
  rw [if_neg (by decide)]
def root63Candidates2 : List Nat := [1601, 10391]
set_option maxRecDepth 8192 in
theorem root63Candidates2_checked : candidates edge 4 317 503 root63Tail3 winner.bound = root63Candidates2 := by
  unfold candidates
  have row : edge 503 = (SortedNatTree.contains FastRows.row503) := by
    change Data.row503.contains = (SortedNatTree.contains FastRows.row503)
    exact FastRows.row503_checked.symm
  rw [row]
  decide
theorem root63Step2 : scan edge (search edge 3) 4 [317] 317 root63Tail2 winner =
    scan edge (search edge 3) 4 [317] 317 root63Tail3 winner := by
  rw [root63Tail2, scan, if_neg (by decide), root63Candidates2_checked]
  change scan edge (search edge 3) 4 [317] 317 root63Tail3
    (if 3 ≤ root63Candidates2.length then search edge 3 [317, 503] root63Candidates2 820 winner else winner) = _
  rw [if_neg (by decide)]
def root63Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root63Candidates3_checked : candidates edge 4 317 773 root63Tail4 winner.bound = root63Candidates3 := by
  unfold candidates
  have row : edge 773 = (SortedNatTree.contains FastRows.row773) := by
    change Data.row773.contains = (SortedNatTree.contains FastRows.row773)
    exact FastRows.row773_checked.symm
  rw [row]
  decide
theorem root63Step3 : scan edge (search edge 3) 4 [317] 317 root63Tail3 winner =
    scan edge (search edge 3) 4 [317] 317 root63Tail4 winner := by
  rw [root63Tail3, scan, if_neg (by decide), root63Candidates3_checked]
  change scan edge (search edge 3) 4 [317] 317 root63Tail4
    (if 3 ≤ root63Candidates3.length then search edge 3 [317, 773] root63Candidates3 1090 winner else winner) = _
  rw [if_neg (by decide)]
def root63Candidates4 : List Nat := [2399, 13763, 21467]
set_option maxRecDepth 8192 in
theorem root63Candidates4_checked : candidates edge 4 317 983 root63Tail5 winner.bound = root63Candidates4 := by
  unfold candidates
  have row : edge 983 = (SortedNatTree.contains FastRows.row983) := by
    change Data.row983.contains = (SortedNatTree.contains FastRows.row983)
    exact FastRows.row983_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root63Child4 : search edge 3 [317, 983] root63Candidates4 1300 winner = winner := by decide
theorem root63Step4 : scan edge (search edge 3) 4 [317] 317 root63Tail4 winner =
    scan edge (search edge 3) 4 [317] 317 root63Tail5 winner := by
  rw [root63Tail4, scan, if_neg (by decide), root63Candidates4_checked]
  change scan edge (search edge 3) 4 [317] 317 root63Tail5
    (if 3 ≤ root63Candidates4.length then search edge 3 [317, 983] root63Candidates4 1300 winner else winner) = _
  rw [if_pos (by decide), root63Child4]
def root63Candidates5 : List Nat := [9173, 22349, 22907]
set_option maxRecDepth 8192 in
theorem root63Candidates5_checked : candidates edge 4 317 1187 root63Tail6 winner.bound = root63Candidates5 := by
  unfold candidates
  have row : edge 1187 = (SortedNatTree.contains FastRows.row1187) := by
    change Data.row1187.contains = (SortedNatTree.contains FastRows.row1187)
    exact FastRows.row1187_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root63Child5 : search edge 3 [317, 1187] root63Candidates5 1504 winner = winner := by decide
theorem root63Step5 : scan edge (search edge 3) 4 [317] 317 root63Tail5 winner =
    scan edge (search edge 3) 4 [317] 317 root63Tail6 winner := by
  rw [root63Tail5, scan, if_neg (by decide), root63Candidates5_checked]
  change scan edge (search edge 3) 4 [317] 317 root63Tail6
    (if 3 ≤ root63Candidates5.length then search edge 3 [317, 1187] root63Candidates5 1504 winner else winner) = _
  rw [if_pos (by decide), root63Child5]
def root63Candidates6 : List Nat := [9257, 9749]
set_option maxRecDepth 8192 in
theorem root63Candidates6_checked : candidates edge 4 317 1601 root63Tail7 winner.bound = root63Candidates6 := by
  unfold candidates
  have row : edge 1601 = (SortedNatTree.contains FastRows.row1601) := by
    change Data.row1601.contains = (SortedNatTree.contains FastRows.row1601)
    exact FastRows.row1601_checked.symm
  rw [row]
  decide
theorem root63Step6 : scan edge (search edge 3) 4 [317] 317 root63Tail6 winner =
    scan edge (search edge 3) 4 [317] 317 root63Tail7 winner := by
  rw [root63Tail6, scan, if_neg (by decide), root63Candidates6_checked]
  change scan edge (search edge 3) 4 [317] 317 root63Tail7
    (if 3 ≤ root63Candidates6.length then search edge 3 [317, 1601] root63Candidates6 1918 winner else winner) = _
  rw [if_neg (by decide)]
def root63Candidates7 : List Nat := [2027, 5099, 10391]
set_option maxRecDepth 8192 in
theorem root63Candidates7_checked : candidates edge 4 317 1697 root63Tail8 winner.bound = root63Candidates7 := by
  unfold candidates
  have row : edge 1697 = (SortedNatTree.contains FastRows.row1697) := by
    change Data.row1697.contains = (SortedNatTree.contains FastRows.row1697)
    exact FastRows.row1697_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root63Child7 : search edge 3 [317, 1697] root63Candidates7 2014 winner = winner := by decide
theorem root63Step7 : scan edge (search edge 3) 4 [317] 317 root63Tail7 winner =
    scan edge (search edge 3) 4 [317] 317 root63Tail8 winner := by
  rw [root63Tail7, scan, if_neg (by decide), root63Candidates7_checked]
  change scan edge (search edge 3) 4 [317] 317 root63Tail8
    (if 3 ≤ root63Candidates7.length then search edge 3 [317, 1697] root63Candidates7 2014 winner else winner) = _
  rw [if_pos (by decide), root63Child7]
def root63Candidates8 : List Nat := [10139, 19559]
set_option maxRecDepth 8192 in
theorem root63Candidates8_checked : candidates edge 4 317 1709 root63Tail9 winner.bound = root63Candidates8 := by
  unfold candidates
  have row : edge 1709 = (SortedNatTree.contains FastRows.row1709) := by
    change Data.row1709.contains = (SortedNatTree.contains FastRows.row1709)
    exact FastRows.row1709_checked.symm
  rw [row]
  decide
theorem root63Step8 : scan edge (search edge 3) 4 [317] 317 root63Tail8 winner =
    scan edge (search edge 3) 4 [317] 317 root63Tail9 winner := by
  rw [root63Tail8, scan, if_neg (by decide), root63Candidates8_checked]
  change scan edge (search edge 3) 4 [317] 317 root63Tail9
    (if 3 ≤ root63Candidates8.length then search edge 3 [317, 1709] root63Candidates8 2026 winner else winner) = _
  rw [if_neg (by decide)]
def root63Candidates9 : List Nat := [3119]
set_option maxRecDepth 8192 in
theorem root63Candidates9_checked : candidates edge 4 317 2027 root63Tail10 winner.bound = root63Candidates9 := by
  unfold candidates
  have row : edge 2027 = (SortedNatTree.contains FastRows.row2027) := by
    change Data.row2027.contains = (SortedNatTree.contains FastRows.row2027)
    exact FastRows.row2027_checked.symm
  rw [row]
  decide
theorem root63Step9 : scan edge (search edge 3) 4 [317] 317 root63Tail9 winner =
    scan edge (search edge 3) 4 [317] 317 root63Tail10 winner := by
  rw [root63Tail9, scan, if_neg (by decide), root63Candidates9_checked]
  change scan edge (search edge 3) 4 [317] 317 root63Tail10
    (if 3 ≤ root63Candidates9.length then search edge 3 [317, 2027] root63Candidates9 2344 winner else winner) = _
  rw [if_neg (by decide)]
def root63Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root63Candidates10_checked : candidates edge 4 317 2237 root63Tail11 winner.bound = root63Candidates10 := by
  unfold candidates
  have row : edge 2237 = (SortedNatTree.contains FastRows.row2237) := by
    change Data.row2237.contains = (SortedNatTree.contains FastRows.row2237)
    exact FastRows.row2237_checked.symm
  rw [row]
  decide
theorem root63Step10 : scan edge (search edge 3) 4 [317] 317 root63Tail10 winner =
    scan edge (search edge 3) 4 [317] 317 root63Tail11 winner := by
  rw [root63Tail10, scan, if_neg (by decide), root63Candidates10_checked]
  change scan edge (search edge 3) 4 [317] 317 root63Tail11
    (if 3 ≤ root63Candidates10.length then search edge 3 [317, 2237] root63Candidates10 2554 winner else winner) = _
  rw [if_neg (by decide)]
def root63Candidates11 : List Nat := [3539, 16427, 18191]
set_option maxRecDepth 8192 in
theorem root63Candidates11_checked : candidates edge 4 317 2243 root63Tail12 winner.bound = root63Candidates11 := by
  unfold candidates
  have row : edge 2243 = (SortedNatTree.contains FastRows.row2243) := by
    change Data.row2243.contains = (SortedNatTree.contains FastRows.row2243)
    exact FastRows.row2243_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root63Child11 : search edge 3 [317, 2243] root63Candidates11 2560 winner = winner := by decide
theorem root63Step11 : scan edge (search edge 3) 4 [317] 317 root63Tail11 winner =
    scan edge (search edge 3) 4 [317] 317 root63Tail12 winner := by
  rw [root63Tail11, scan, if_neg (by decide), root63Candidates11_checked]
  change scan edge (search edge 3) 4 [317] 317 root63Tail12
    (if 3 ≤ root63Candidates11.length then search edge 3 [317, 2243] root63Candidates11 2560 winner else winner) = _
  rw [if_pos (by decide), root63Child11]
def root63Candidates12 : List Nat := [14153]
set_option maxRecDepth 8192 in
theorem root63Candidates12_checked : candidates edge 4 317 2399 root63Tail13 winner.bound = root63Candidates12 := by
  unfold candidates
  have row : edge 2399 = (SortedNatTree.contains FastRows.row2399) := by
    change Data.row2399.contains = (SortedNatTree.contains FastRows.row2399)
    exact FastRows.row2399_checked.symm
  rw [row]
  decide
theorem root63Step12 : scan edge (search edge 3) 4 [317] 317 root63Tail12 winner =
    scan edge (search edge 3) 4 [317] 317 root63Tail13 winner := by
  rw [root63Tail12, scan, if_neg (by decide), root63Candidates12_checked]
  change scan edge (search edge 3) 4 [317] 317 root63Tail13
    (if 3 ≤ root63Candidates12.length then search edge 3 [317, 2399] root63Candidates12 2716 winner else winner) = _
  rw [if_neg (by decide)]
def root63Candidates13 : List Nat := [11423]
set_option maxRecDepth 8192 in
theorem root63Candidates13_checked : candidates edge 4 317 2441 root63Tail14 winner.bound = root63Candidates13 := by
  unfold candidates
  have row : edge 2441 = (SortedNatTree.contains FastRows.row2441) := by
    change Data.row2441.contains = (SortedNatTree.contains FastRows.row2441)
    exact FastRows.row2441_checked.symm
  rw [row]
  decide
theorem root63Step13 : scan edge (search edge 3) 4 [317] 317 root63Tail13 winner =
    scan edge (search edge 3) 4 [317] 317 root63Tail14 winner := by
  rw [root63Tail13, scan, if_neg (by decide), root63Candidates13_checked]
  change scan edge (search edge 3) 4 [317] 317 root63Tail14
    (if 3 ≤ root63Candidates13.length then search edge 3 [317, 2441] root63Candidates13 2758 winner else winner) = _
  rw [if_neg (by decide)]
def root63Candidates14 : List Nat := [12011]
set_option maxRecDepth 8192 in
theorem root63Candidates14_checked : candidates edge 4 317 2801 root63Tail15 winner.bound = root63Candidates14 := by
  unfold candidates
  have row : edge 2801 = (SortedNatTree.contains FastRows.row2801) := by
    change Data.row2801.contains = (SortedNatTree.contains FastRows.row2801)
    exact FastRows.row2801_checked.symm
  rw [row]
  decide
theorem root63Step14 : scan edge (search edge 3) 4 [317] 317 root63Tail14 winner =
    scan edge (search edge 3) 4 [317] 317 root63Tail15 winner := by
  rw [root63Tail14, scan, if_neg (by decide), root63Candidates14_checked]
  change scan edge (search edge 3) 4 [317] 317 root63Tail15
    (if 3 ≤ root63Candidates14.length then search edge 3 [317, 2801] root63Candidates14 3118 winner else winner) = _
  rw [if_neg (by decide)]
def root63Candidates15 : List Nat := [10139, 21341]
set_option maxRecDepth 8192 in
theorem root63Candidates15_checked : candidates edge 4 317 3119 root63Tail16 winner.bound = root63Candidates15 := by
  unfold candidates
  have row : edge 3119 = (SortedNatTree.contains FastRows.row3119) := by
    change Data.row3119.contains = (SortedNatTree.contains FastRows.row3119)
    exact FastRows.row3119_checked.symm
  rw [row]
  decide
theorem root63Step15 : scan edge (search edge 3) 4 [317] 317 root63Tail15 winner =
    scan edge (search edge 3) 4 [317] 317 root63Tail16 winner := by
  rw [root63Tail15, scan, if_neg (by decide), root63Candidates15_checked]
  change scan edge (search edge 3) 4 [317] 317 root63Tail16
    (if 3 ≤ root63Candidates15.length then search edge 3 [317, 3119] root63Candidates15 3436 winner else winner) = _
  rw [if_neg (by decide)]
def root63Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root63Candidates16_checked : candidates edge 4 317 3539 root63Tail17 winner.bound = root63Candidates16 := by
  unfold candidates
  have row : edge 3539 = (SortedNatTree.contains FastRows.row3539) := by
    change Data.row3539.contains = (SortedNatTree.contains FastRows.row3539)
    exact FastRows.row3539_checked.symm
  rw [row]
  decide
theorem root63Step16 : scan edge (search edge 3) 4 [317] 317 root63Tail16 winner =
    scan edge (search edge 3) 4 [317] 317 root63Tail17 winner := by
  rw [root63Tail16, scan, if_neg (by decide), root63Candidates16_checked]
  change scan edge (search edge 3) 4 [317] 317 root63Tail17
    (if 3 ≤ root63Candidates16.length then search edge 3 [317, 3539] root63Candidates16 3856 winner else winner) = _
  rw [if_neg (by decide)]
def root63Candidates17 : List Nat := [4091, 10163, 21407]
set_option maxRecDepth 8192 in
theorem root63Candidates17_checked : candidates edge 4 317 3659 root63Tail18 winner.bound = root63Candidates17 := by
  unfold candidates
  have row : edge 3659 = (SortedNatTree.contains FastRows.row3659) := by
    change Data.row3659.contains = (SortedNatTree.contains FastRows.row3659)
    exact FastRows.row3659_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root63Child17 : search edge 3 [317, 3659] root63Candidates17 3976 winner = winner := by decide
theorem root63Step17 : scan edge (search edge 3) 4 [317] 317 root63Tail17 winner =
    scan edge (search edge 3) 4 [317] 317 root63Tail18 winner := by
  rw [root63Tail17, scan, if_neg (by decide), root63Candidates17_checked]
  change scan edge (search edge 3) 4 [317] 317 root63Tail18
    (if 3 ≤ root63Candidates17.length then search edge 3 [317, 3659] root63Candidates17 3976 winner else winner) = _
  rw [if_pos (by decide), root63Child17]
def root63Candidates18 : List Nat := []
set_option maxRecDepth 8192 in
theorem root63Candidates18_checked : candidates edge 4 317 4091 root63Tail19 winner.bound = root63Candidates18 := by
  unfold candidates
  have row : edge 4091 = (SortedNatTree.contains FastRows.row4091) := by
    change Data.row4091.contains = (SortedNatTree.contains FastRows.row4091)
    exact FastRows.row4091_checked.symm
  rw [row]
  decide
theorem root63Step18 : scan edge (search edge 3) 4 [317] 317 root63Tail18 winner =
    scan edge (search edge 3) 4 [317] 317 root63Tail19 winner := by
  rw [root63Tail18, scan, if_neg (by decide), root63Candidates18_checked]
  change scan edge (search edge 3) 4 [317] 317 root63Tail19
    (if 3 ≤ root63Candidates18.length then search edge 3 [317, 4091] root63Candidates18 4408 winner else winner) = _
  rw [if_neg (by decide)]
def root63Candidates19 : List Nat := [5099, 16427]
set_option maxRecDepth 8192 in
theorem root63Candidates19_checked : candidates edge 4 317 4793 root63Tail20 winner.bound = root63Candidates19 := by
  unfold candidates
  have row : edge 4793 = (SortedNatTree.contains FastRows.row4793) := by
    change Data.row4793.contains = (SortedNatTree.contains FastRows.row4793)
    exact FastRows.row4793_checked.symm
  rw [row]
  decide
theorem root63Step19 : scan edge (search edge 3) 4 [317] 317 root63Tail19 winner =
    scan edge (search edge 3) 4 [317] 317 root63Tail20 winner := by
  rw [root63Tail19, scan, if_neg (by decide), root63Candidates19_checked]
  change scan edge (search edge 3) 4 [317] 317 root63Tail20
    (if 3 ≤ root63Candidates19.length then search edge 3 [317, 4793] root63Candidates19 5110 winner else winner) = _
  rw [if_neg (by decide)]
def root63Candidates20 : List Nat := [17489]
set_option maxRecDepth 8192 in
theorem root63Candidates20_checked : candidates edge 4 317 4799 root63Tail21 winner.bound = root63Candidates20 := by
  unfold candidates
  have row : edge 4799 = (SortedNatTree.contains FastRows.row4799) := by
    change Data.row4799.contains = (SortedNatTree.contains FastRows.row4799)
    exact FastRows.row4799_checked.symm
  rw [row]
  decide
theorem root63Step20 : scan edge (search edge 3) 4 [317] 317 root63Tail20 winner =
    scan edge (search edge 3) 4 [317] 317 root63Tail21 winner := by
  rw [root63Tail20, scan, if_neg (by decide), root63Candidates20_checked]
  change scan edge (search edge 3) 4 [317] 317 root63Tail21
    (if 3 ≤ root63Candidates20.length then search edge 3 [317, 4799] root63Candidates20 5116 winner else winner) = _
  rw [if_neg (by decide)]
def root63Candidates21 : List Nat := [6221, 16427]
set_option maxRecDepth 8192 in
theorem root63Candidates21_checked : candidates edge 4 317 4973 root63Tail22 winner.bound = root63Candidates21 := by
  unfold candidates
  have row : edge 4973 = (SortedNatTree.contains FastRows.row4973) := by
    change Data.row4973.contains = (SortedNatTree.contains FastRows.row4973)
    exact FastRows.row4973_checked.symm
  rw [row]
  decide
theorem root63Step21 : scan edge (search edge 3) 4 [317] 317 root63Tail21 winner =
    scan edge (search edge 3) 4 [317] 317 root63Tail22 winner := by
  rw [root63Tail21, scan, if_neg (by decide), root63Candidates21_checked]
  change scan edge (search edge 3) 4 [317] 317 root63Tail22
    (if 3 ≤ root63Candidates21.length then search edge 3 [317, 4973] root63Candidates21 5290 winner else winner) = _
  rw [if_neg (by decide)]
def root63Candidates22 : List Nat := [15881]
set_option maxRecDepth 8192 in
theorem root63Candidates22_checked : candidates edge 4 317 5009 root63Tail23 winner.bound = root63Candidates22 := by
  unfold candidates
  have row : edge 5009 = (SortedNatTree.contains FastRows.row5009) := by
    change Data.row5009.contains = (SortedNatTree.contains FastRows.row5009)
    exact FastRows.row5009_checked.symm
  rw [row]
  decide
theorem root63Step22 : scan edge (search edge 3) 4 [317] 317 root63Tail22 winner =
    scan edge (search edge 3) 4 [317] 317 root63Tail23 winner := by
  rw [root63Tail22, scan, if_neg (by decide), root63Candidates22_checked]
  change scan edge (search edge 3) 4 [317] 317 root63Tail23
    (if 3 ≤ root63Candidates22.length then search edge 3 [317, 5009] root63Candidates22 5326 winner else winner) = _
  rw [if_neg (by decide)]
def root63Candidates23 : List Nat := [7949]
set_option maxRecDepth 8192 in
theorem root63Candidates23_checked : candidates edge 4 317 5099 root63Tail24 winner.bound = root63Candidates23 := by
  unfold candidates
  have row : edge 5099 = (SortedNatTree.contains FastRows.row5099) := by
    change Data.row5099.contains = (SortedNatTree.contains FastRows.row5099)
    exact FastRows.row5099_checked.symm
  rw [row]
  decide
theorem root63Step23 : scan edge (search edge 3) 4 [317] 317 root63Tail23 winner =
    scan edge (search edge 3) 4 [317] 317 root63Tail24 winner := by
  rw [root63Tail23, scan, if_neg (by decide), root63Candidates23_checked]
  change scan edge (search edge 3) 4 [317] 317 root63Tail24
    (if 3 ≤ root63Candidates23.length then search edge 3 [317, 5099] root63Candidates23 5416 winner else winner) = _
  rw [if_neg (by decide)]
def root63Candidates24 : List Nat := [18191, 19013]
set_option maxRecDepth 8192 in
theorem root63Candidates24_checked : candidates edge 4 317 5801 root63Tail25 winner.bound = root63Candidates24 := by
  unfold candidates
  have row : edge 5801 = (SortedNatTree.contains FastRows.row5801) := by
    change Data.row5801.contains = (SortedNatTree.contains FastRows.row5801)
    exact FastRows.row5801_checked.symm
  rw [row]
  decide
theorem root63Step24 : scan edge (search edge 3) 4 [317] 317 root63Tail24 winner =
    scan edge (search edge 3) 4 [317] 317 root63Tail25 winner := by
  rw [root63Tail24, scan, if_neg (by decide), root63Candidates24_checked]
  change scan edge (search edge 3) 4 [317] 317 root63Tail25
    (if 3 ≤ root63Candidates24.length then search edge 3 [317, 5801] root63Candidates24 6118 winner else winner) = _
  rw [if_neg (by decide)]
def root63Candidates25 : List Nat := [12143, 12503]
set_option maxRecDepth 8192 in
theorem root63Candidates25_checked : candidates edge 4 317 6053 root63Tail26 winner.bound = root63Candidates25 := by
  unfold candidates
  have row : edge 6053 = (SortedNatTree.contains FastRows.row6053) := by
    change Data.row6053.contains = (SortedNatTree.contains FastRows.row6053)
    exact FastRows.row6053_checked.symm
  rw [row]
  decide
theorem root63Step25 : scan edge (search edge 3) 4 [317] 317 root63Tail25 winner =
    scan edge (search edge 3) 4 [317] 317 root63Tail26 winner := by
  rw [root63Tail25, scan, if_neg (by decide), root63Candidates25_checked]
  change scan edge (search edge 3) 4 [317] 317 root63Tail26
    (if 3 ≤ root63Candidates25.length then search edge 3 [317, 6053] root63Candidates25 6370 winner else winner) = _
  rw [if_neg (by decide)]
def root63Candidates26 : List Nat := [9173, 15107, 18593]
set_option maxRecDepth 8192 in
theorem root63Candidates26_checked : candidates edge 4 317 6221 root63Tail27 winner.bound = root63Candidates26 := by
  unfold candidates
  have row : edge 6221 = (SortedNatTree.contains FastRows.row6221) := by
    change Data.row6221.contains = (SortedNatTree.contains FastRows.row6221)
    exact FastRows.row6221_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root63Child26 : search edge 3 [317, 6221] root63Candidates26 6538 winner = winner := by decide
theorem root63Step26 : scan edge (search edge 3) 4 [317] 317 root63Tail26 winner =
    scan edge (search edge 3) 4 [317] 317 root63Tail27 winner := by
  rw [root63Tail26, scan, if_neg (by decide), root63Candidates26_checked]
  change scan edge (search edge 3) 4 [317] 317 root63Tail27
    (if 3 ≤ root63Candidates26.length then search edge 3 [317, 6221] root63Candidates26 6538 winner else winner) = _
  rw [if_pos (by decide), root63Child26]
theorem root63Scan27 : scan edge (search edge 3) 4 [317] 317 root63Tail27 winner = winner := by rw [root63Tail27, scan, if_pos (by decide)]
theorem root63Scan26 : scan edge (search edge 3) 4 [317] 317 root63Tail26 winner = winner :=
  (root63Step26).trans root63Scan27
theorem root63Scan25 : scan edge (search edge 3) 4 [317] 317 root63Tail25 winner = winner :=
  (root63Step25).trans root63Scan26
theorem root63Scan24 : scan edge (search edge 3) 4 [317] 317 root63Tail24 winner = winner :=
  (root63Step24).trans root63Scan25
theorem root63Scan23 : scan edge (search edge 3) 4 [317] 317 root63Tail23 winner = winner :=
  (root63Step23).trans root63Scan24
theorem root63Scan22 : scan edge (search edge 3) 4 [317] 317 root63Tail22 winner = winner :=
  (root63Step22).trans root63Scan23
theorem root63Scan21 : scan edge (search edge 3) 4 [317] 317 root63Tail21 winner = winner :=
  (root63Step21).trans root63Scan22
theorem root63Scan20 : scan edge (search edge 3) 4 [317] 317 root63Tail20 winner = winner :=
  (root63Step20).trans root63Scan21
theorem root63Scan19 : scan edge (search edge 3) 4 [317] 317 root63Tail19 winner = winner :=
  (root63Step19).trans root63Scan20
theorem root63Scan18 : scan edge (search edge 3) 4 [317] 317 root63Tail18 winner = winner :=
  (root63Step18).trans root63Scan19
theorem root63Scan17 : scan edge (search edge 3) 4 [317] 317 root63Tail17 winner = winner :=
  (root63Step17).trans root63Scan18
theorem root63Scan16 : scan edge (search edge 3) 4 [317] 317 root63Tail16 winner = winner :=
  (root63Step16).trans root63Scan17
theorem root63Scan15 : scan edge (search edge 3) 4 [317] 317 root63Tail15 winner = winner :=
  (root63Step15).trans root63Scan16
theorem root63Scan14 : scan edge (search edge 3) 4 [317] 317 root63Tail14 winner = winner :=
  (root63Step14).trans root63Scan15
theorem root63Scan13 : scan edge (search edge 3) 4 [317] 317 root63Tail13 winner = winner :=
  (root63Step13).trans root63Scan14
theorem root63Scan12 : scan edge (search edge 3) 4 [317] 317 root63Tail12 winner = winner :=
  (root63Step12).trans root63Scan13
theorem root63Scan11 : scan edge (search edge 3) 4 [317] 317 root63Tail11 winner = winner :=
  (root63Step11).trans root63Scan12
theorem root63Scan10 : scan edge (search edge 3) 4 [317] 317 root63Tail10 winner = winner :=
  (root63Step10).trans root63Scan11
theorem root63Scan9 : scan edge (search edge 3) 4 [317] 317 root63Tail9 winner = winner :=
  (root63Step9).trans root63Scan10
theorem root63Scan8 : scan edge (search edge 3) 4 [317] 317 root63Tail8 winner = winner :=
  (root63Step8).trans root63Scan9
theorem root63Scan7 : scan edge (search edge 3) 4 [317] 317 root63Tail7 winner = winner :=
  (root63Step7).trans root63Scan8
theorem root63Scan6 : scan edge (search edge 3) 4 [317] 317 root63Tail6 winner = winner :=
  (root63Step6).trans root63Scan7
theorem root63Scan5 : scan edge (search edge 3) 4 [317] 317 root63Tail5 winner = winner :=
  (root63Step5).trans root63Scan6
theorem root63Scan4 : scan edge (search edge 3) 4 [317] 317 root63Tail4 winner = winner :=
  (root63Step4).trans root63Scan5
theorem root63Scan3 : scan edge (search edge 3) 4 [317] 317 root63Tail3 winner = winner :=
  (root63Step3).trans root63Scan4
theorem root63Scan2 : scan edge (search edge 3) 4 [317] 317 root63Tail2 winner = winner :=
  (root63Step2).trans root63Scan3
theorem root63Scan1 : scan edge (search edge 3) 4 [317] 317 root63Tail1 winner = winner :=
  (root63Step1).trans root63Scan2
theorem root63Scan0 : scan edge (search edge 3) 4 [317] 317 root63Tail0 winner = winner :=
  (root63Step0).trans root63Scan1
set_option maxRecDepth 4096 in
theorem root63Branch : search edge 4 [317] root63Candidates 317 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [317] 317 root63Tail0 winner = winner
  exact root63Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
