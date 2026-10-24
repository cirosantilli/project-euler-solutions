import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root28Candidates : List Nat := Data.row127
set_option maxRecDepth 8192 in
theorem root28Candidates_checked : candidates edge 5 0 127 tail29 winner.bound = root28Candidates := by
  apply candidates_eq_row edge 5 0 127 tail29 root28Candidates winner.bound tail29_ordered
  · decide
  · decide
  · funext q
    rfl
def root28Tail92 : List Nat := []
def root28Tail91 : List Nat := 25849 :: root28Tail92
def root28Tail90 : List Nat := 25771 :: root28Tail91
def root28Tail89 : List Nat := 25183 :: root28Tail90
def root28Tail88 : List Nat := 25057 :: root28Tail89
def root28Tail87 : List Nat := 24709 :: root28Tail88
def root28Tail86 : List Nat := 24421 :: root28Tail87
def root28Tail85 : List Nat := 24223 :: root28Tail86
def root28Tail84 : List Nat := 24181 :: root28Tail85
def root28Tail83 : List Nat := 24133 :: root28Tail84
def root28Tail82 : List Nat := 24097 :: root28Tail83
def root28Tail81 : List Nat := 23509 :: root28Tail82
def root28Tail80 : List Nat := 23203 :: root28Tail81
def root28Tail79 : List Nat := 22921 :: root28Tail80
def root28Tail78 : List Nat := 22447 :: root28Tail79
def root28Tail77 : List Nat := 22291 :: root28Tail78
def root28Tail76 : List Nat := 22159 :: root28Tail77
def root28Tail75 : List Nat := 21019 :: root28Tail76
def root28Tail74 : List Nat := 20047 :: root28Tail75
def root28Tail73 : List Nat := 19891 :: root28Tail74
def root28Tail72 : List Nat := 19387 :: root28Tail73
def root28Tail71 : List Nat := 19141 :: root28Tail72
def root28Tail70 : List Nat := 18253 :: root28Tail71
def root28Tail69 : List Nat := 17491 :: root28Tail70
def root28Tail68 : List Nat := 17077 :: root28Tail69
def root28Tail67 : List Nat := 16981 :: root28Tail68
def root28Tail66 : List Nat := 16747 :: root28Tail67
def root28Tail65 : List Nat := 16651 :: root28Tail66
def root28Tail64 : List Nat := 16411 :: root28Tail65
def root28Tail63 : List Nat := 15817 :: root28Tail64
def root28Tail62 : List Nat := 14983 :: root28Tail63
def root28Tail61 : List Nat := 14851 :: root28Tail62
def root28Tail60 : List Nat := 14461 :: root28Tail61
def root28Tail59 : List Nat := 14293 :: root28Tail60
def root28Tail58 : List Nat := 13873 :: root28Tail59
def root28Tail57 : List Nat := 13003 :: root28Tail58
def root28Tail56 : List Nat := 12967 :: root28Tail57
def root28Tail55 : List Nat := 12703 :: root28Tail56
def root28Tail54 : List Nat := 12619 :: root28Tail55
def root28Tail53 : List Nat := 12409 :: root28Tail54
def root28Tail52 : List Nat := 12241 :: root28Tail53
def root28Tail51 : List Nat := 11821 :: root28Tail52
def root28Tail50 : List Nat := 11701 :: root28Tail51
def root28Tail49 : List Nat := 11329 :: root28Tail50
def root28Tail48 : List Nat := 11287 :: root28Tail49
def root28Tail47 : List Nat := 11161 :: root28Tail48
def root28Tail46 : List Nat := 10987 :: root28Tail47
def root28Tail45 : List Nat := 10597 :: root28Tail46
def root28Tail44 : List Nat := 10273 :: root28Tail45
def root28Tail43 : List Nat := 10177 :: root28Tail44
def root28Tail42 : List Nat := 9547 :: root28Tail43
def root28Tail41 : List Nat := 9337 :: root28Tail42
def root28Tail40 : List Nat := 9181 :: root28Tail41
def root28Tail39 : List Nat := 9043 :: root28Tail40
def root28Tail38 : List Nat := 8713 :: root28Tail39
def root28Tail37 : List Nat := 7993 :: root28Tail38
def root28Tail36 : List Nat := 7753 :: root28Tail37
def root28Tail35 : List Nat := 7723 :: root28Tail36
def root28Tail34 : List Nat := 7369 :: root28Tail35
def root28Tail33 : List Nat := 6949 :: root28Tail34
def root28Tail32 : List Nat := 6871 :: root28Tail33
def root28Tail31 : List Nat := 6733 :: root28Tail32
def root28Tail30 : List Nat := 6571 :: root28Tail31
def root28Tail29 : List Nat := 6481 :: root28Tail30
def root28Tail28 : List Nat := 6361 :: root28Tail29
def root28Tail27 : List Nat := 6271 :: root28Tail28
def root28Tail26 : List Nat := 5779 :: root28Tail27
def root28Tail25 : List Nat := 5749 :: root28Tail26
def root28Tail24 : List Nat := 5683 :: root28Tail25
def root28Tail23 : List Nat := 5563 :: root28Tail24
def root28Tail22 : List Nat := 5179 :: root28Tail23
def root28Tail21 : List Nat := 4759 :: root28Tail22
def root28Tail20 : List Nat := 4423 :: root28Tail21
def root28Tail19 : List Nat := 4363 :: root28Tail20
def root28Tail18 : List Nat := 3673 :: root28Tail19
def root28Tail17 : List Nat := 3541 :: root28Tail18
def root28Tail16 : List Nat := 3343 :: root28Tail17
def root28Tail15 : List Nat := 3001 :: root28Tail16
def root28Tail14 : List Nat := 2749 :: root28Tail15
def root28Tail13 : List Nat := 2281 :: root28Tail14
def root28Tail12 : List Nat := 1987 :: root28Tail13
def root28Tail11 : List Nat := 1129 :: root28Tail12
def root28Tail10 : List Nat := 1069 :: root28Tail11
def root28Tail9 : List Nat := 1033 :: root28Tail10
def root28Tail8 : List Nat := 733 :: root28Tail9
def root28Tail7 : List Nat := 607 :: root28Tail8
def root28Tail6 : List Nat := 601 :: root28Tail7
def root28Tail5 : List Nat := 373 :: root28Tail6
def root28Tail4 : List Nat := 331 :: root28Tail5
def root28Tail3 : List Nat := 271 :: root28Tail4
def root28Tail2 : List Nat := 241 :: root28Tail3
def root28Tail1 : List Nat := 163 :: root28Tail2
def root28Tail0 : List Nat := 157 :: root28Tail1
def root28Candidates0 : List Nat := [733, 2749, 8713, 22159]
set_option maxRecDepth 8192 in
theorem root28Candidates0_checked : candidates edge 4 127 157 root28Tail1 winner.bound = root28Candidates0 := by
  unfold candidates
  have row : edge 157 = (SortedNatTree.contains FastRows.row157) := by
    change Data.row157.contains = (SortedNatTree.contains FastRows.row157)
    exact FastRows.row157_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root28Child0 : search edge 3 [127, 157] root28Candidates0 284 winner = winner := by decide
theorem root28Step0 : scan edge (search edge 3) 4 [127] 127 root28Tail0 winner =
    scan edge (search edge 3) 4 [127] 127 root28Tail1 winner := by
  rw [root28Tail0, scan, if_neg (by decide), root28Candidates0_checked]
  change scan edge (search edge 3) 4 [127] 127 root28Tail1
    (if 3 ≤ root28Candidates0.length then search edge 3 [127, 157] root28Candidates0 284 winner else winner) = _
  rw [if_pos (by decide), root28Child0]
def root28Candidates1 : List Nat := [5563, 11821, 16411, 16747]
set_option maxRecDepth 8192 in
theorem root28Candidates1_checked : candidates edge 4 127 163 root28Tail2 winner.bound = root28Candidates1 := by
  unfold candidates
  have row : edge 163 = (SortedNatTree.contains FastRows.row163) := by
    change Data.row163.contains = (SortedNatTree.contains FastRows.row163)
    exact FastRows.row163_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root28Child1 : search edge 3 [127, 163] root28Candidates1 290 winner = winner := by decide
theorem root28Step1 : scan edge (search edge 3) 4 [127] 127 root28Tail1 winner =
    scan edge (search edge 3) 4 [127] 127 root28Tail2 winner := by
  rw [root28Tail1, scan, if_neg (by decide), root28Candidates1_checked]
  change scan edge (search edge 3) 4 [127] 127 root28Tail2
    (if 3 ≤ root28Candidates1.length then search edge 3 [127, 163] root28Candidates1 290 winner else winner) = _
  rw [if_pos (by decide), root28Child1]
def root28Candidates2 : List Nat := [271, 601, 6361]
set_option maxRecDepth 8192 in
theorem root28Candidates2_checked : candidates edge 4 127 241 root28Tail3 winner.bound = root28Candidates2 := by
  unfold candidates
  have row : edge 241 = (SortedNatTree.contains FastRows.row241) := by
    change Data.row241.contains = (SortedNatTree.contains FastRows.row241)
    exact FastRows.row241_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root28Child2 : search edge 3 [127, 241] root28Candidates2 368 winner = winner := by decide
theorem root28Step2 : scan edge (search edge 3) 4 [127] 127 root28Tail2 winner =
    scan edge (search edge 3) 4 [127] 127 root28Tail3 winner := by
  rw [root28Tail2, scan, if_neg (by decide), root28Candidates2_checked]
  change scan edge (search edge 3) 4 [127] 127 root28Tail3
    (if 3 ≤ root28Candidates2.length then search edge 3 [127, 241] root28Candidates2 368 winner else winner) = _
  rw [if_pos (by decide), root28Child2]
def root28Candidates3 : List Nat := [3541, 6733, 9043, 13873, 14293]
set_option maxRecDepth 8192 in
theorem root28Candidates3_checked : candidates edge 4 127 271 root28Tail4 winner.bound = root28Candidates3 := by
  unfold candidates
  have row : edge 271 = (SortedNatTree.contains FastRows.row271) := by
    change Data.row271.contains = (SortedNatTree.contains FastRows.row271)
    exact FastRows.row271_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root28Child3 : search edge 3 [127, 271] root28Candidates3 398 winner = winner := by decide
theorem root28Step3 : scan edge (search edge 3) 4 [127] 127 root28Tail3 winner =
    scan edge (search edge 3) 4 [127] 127 root28Tail4 winner := by
  rw [root28Tail3, scan, if_neg (by decide), root28Candidates3_checked]
  change scan edge (search edge 3) 4 [127] 127 root28Tail4
    (if 3 ≤ root28Candidates3.length then search edge 3 [127, 271] root28Candidates3 398 winner else winner) = _
  rw [if_pos (by decide), root28Child3]
def root28Candidates4 : List Nat := [5779, 6361, 6949, 16651, 23509]
set_option maxRecDepth 8192 in
theorem root28Candidates4_checked : candidates edge 4 127 331 root28Tail5 winner.bound = root28Candidates4 := by
  unfold candidates
  have row : edge 331 = (SortedNatTree.contains FastRows.row331) := by
    change Data.row331.contains = (SortedNatTree.contains FastRows.row331)
    exact FastRows.row331_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root28Child4 : search edge 3 [127, 331] root28Candidates4 458 winner = winner := by decide
theorem root28Step4 : scan edge (search edge 3) 4 [127] 127 root28Tail4 winner =
    scan edge (search edge 3) 4 [127] 127 root28Tail5 winner := by
  rw [root28Tail4, scan, if_neg (by decide), root28Candidates4_checked]
  change scan edge (search edge 3) 4 [127] 127 root28Tail5
    (if 3 ≤ root28Candidates4.length then search edge 3 [127, 331] root28Candidates4 458 winner else winner) = _
  rw [if_pos (by decide), root28Child4]
def root28Candidates5 : List Nat := [17491]
set_option maxRecDepth 8192 in
theorem root28Candidates5_checked : candidates edge 4 127 373 root28Tail6 winner.bound = root28Candidates5 := by
  unfold candidates
  have row : edge 373 = (SortedNatTree.contains FastRows.row373) := by
    change Data.row373.contains = (SortedNatTree.contains FastRows.row373)
    exact FastRows.row373_checked.symm
  rw [row]
  decide
theorem root28Step5 : scan edge (search edge 3) 4 [127] 127 root28Tail5 winner =
    scan edge (search edge 3) 4 [127] 127 root28Tail6 winner := by
  rw [root28Tail5, scan, if_neg (by decide), root28Candidates5_checked]
  change scan edge (search edge 3) 4 [127] 127 root28Tail6
    (if 3 ≤ root28Candidates5.length then search edge 3 [127, 373] root28Candidates5 500 winner else winner) = _
  rw [if_neg (by decide)]
def root28Candidates6 : List Nat := [22159, 22291]
set_option maxRecDepth 8192 in
theorem root28Candidates6_checked : candidates edge 4 127 601 root28Tail7 winner.bound = root28Candidates6 := by
  unfold candidates
  have row : edge 601 = (SortedNatTree.contains FastRows.row601) := by
    change Data.row601.contains = (SortedNatTree.contains FastRows.row601)
    exact FastRows.row601_checked.symm
  rw [row]
  decide
theorem root28Step6 : scan edge (search edge 3) 4 [127] 127 root28Tail6 winner =
    scan edge (search edge 3) 4 [127] 127 root28Tail7 winner := by
  rw [root28Tail6, scan, if_neg (by decide), root28Candidates6_checked]
  change scan edge (search edge 3) 4 [127] 127 root28Tail7
    (if 3 ≤ root28Candidates6.length then search edge 3 [127, 601] root28Candidates6 728 winner else winner) = _
  rw [if_neg (by decide)]
def root28Candidates7 : List Nat := [4363, 9043, 9547]
set_option maxRecDepth 8192 in
theorem root28Candidates7_checked : candidates edge 4 127 607 root28Tail8 winner.bound = root28Candidates7 := by
  unfold candidates
  have row : edge 607 = (SortedNatTree.contains FastRows.row607) := by
    change Data.row607.contains = (SortedNatTree.contains FastRows.row607)
    exact FastRows.row607_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root28Child7 : search edge 3 [127, 607] root28Candidates7 734 winner = winner := by decide
theorem root28Step7 : scan edge (search edge 3) 4 [127] 127 root28Tail7 winner =
    scan edge (search edge 3) 4 [127] 127 root28Tail8 winner := by
  rw [root28Tail7, scan, if_neg (by decide), root28Candidates7_checked]
  change scan edge (search edge 3) 4 [127] 127 root28Tail8
    (if 3 ≤ root28Candidates7.length then search edge 3 [127, 607] root28Candidates7 734 winner else winner) = _
  rw [if_pos (by decide), root28Child7]
def root28Candidates8 : List Nat := [7723, 9043]
set_option maxRecDepth 8192 in
theorem root28Candidates8_checked : candidates edge 4 127 733 root28Tail9 winner.bound = root28Candidates8 := by
  unfold candidates
  have row : edge 733 = (SortedNatTree.contains FastRows.row733) := by
    change Data.row733.contains = (SortedNatTree.contains FastRows.row733)
    exact FastRows.row733_checked.symm
  rw [row]
  decide
theorem root28Step8 : scan edge (search edge 3) 4 [127] 127 root28Tail8 winner =
    scan edge (search edge 3) 4 [127] 127 root28Tail9 winner := by
  rw [root28Tail8, scan, if_neg (by decide), root28Candidates8_checked]
  change scan edge (search edge 3) 4 [127] 127 root28Tail9
    (if 3 ≤ root28Candidates8.length then search edge 3 [127, 733] root28Candidates8 860 winner else winner) = _
  rw [if_neg (by decide)]
def root28Candidates9 : List Nat := [11161, 22291]
set_option maxRecDepth 8192 in
theorem root28Candidates9_checked : candidates edge 4 127 1033 root28Tail10 winner.bound = root28Candidates9 := by
  unfold candidates
  have row : edge 1033 = (SortedNatTree.contains FastRows.row1033) := by
    change Data.row1033.contains = (SortedNatTree.contains FastRows.row1033)
    exact FastRows.row1033_checked.symm
  rw [row]
  decide
theorem root28Step9 : scan edge (search edge 3) 4 [127] 127 root28Tail9 winner =
    scan edge (search edge 3) 4 [127] 127 root28Tail10 winner := by
  rw [root28Tail9, scan, if_neg (by decide), root28Candidates9_checked]
  change scan edge (search edge 3) 4 [127] 127 root28Tail10
    (if 3 ≤ root28Candidates9.length then search edge 3 [127, 1033] root28Candidates9 1160 winner else winner) = _
  rw [if_neg (by decide)]
def root28Candidates10 : List Nat := [2281, 5683, 7993]
set_option maxRecDepth 8192 in
theorem root28Candidates10_checked : candidates edge 4 127 1069 root28Tail11 winner.bound = root28Candidates10 := by
  unfold candidates
  have row : edge 1069 = (SortedNatTree.contains FastRows.row1069) := by
    change Data.row1069.contains = (SortedNatTree.contains FastRows.row1069)
    exact FastRows.row1069_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root28Child10 : search edge 3 [127, 1069] root28Candidates10 1196 winner = winner := by decide
theorem root28Step10 : scan edge (search edge 3) 4 [127] 127 root28Tail10 winner =
    scan edge (search edge 3) 4 [127] 127 root28Tail11 winner := by
  rw [root28Tail10, scan, if_neg (by decide), root28Candidates10_checked]
  change scan edge (search edge 3) 4 [127] 127 root28Tail11
    (if 3 ≤ root28Candidates10.length then search edge 3 [127, 1069] root28Candidates10 1196 winner else winner) = _
  rw [if_pos (by decide), root28Child10]
def root28Candidates11 : List Nat := [3001, 6871, 16651]
set_option maxRecDepth 8192 in
theorem root28Candidates11_checked : candidates edge 4 127 1129 root28Tail12 winner.bound = root28Candidates11 := by
  unfold candidates
  have row : edge 1129 = (SortedNatTree.contains FastRows.row1129) := by
    change Data.row1129.contains = (SortedNatTree.contains FastRows.row1129)
    exact FastRows.row1129_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root28Child11 : search edge 3 [127, 1129] root28Candidates11 1256 winner = winner := by decide
theorem root28Step11 : scan edge (search edge 3) 4 [127] 127 root28Tail11 winner =
    scan edge (search edge 3) 4 [127] 127 root28Tail12 winner := by
  rw [root28Tail11, scan, if_neg (by decide), root28Candidates11_checked]
  change scan edge (search edge 3) 4 [127] 127 root28Tail12
    (if 3 ≤ root28Candidates11.length then search edge 3 [127, 1129] root28Candidates11 1256 winner else winner) = _
  rw [if_pos (by decide), root28Child11]
def root28Candidates12 : List Nat := [4423, 5563, 6481, 11287, 15817, 16747, 19891]
set_option maxRecDepth 8192 in
theorem root28Candidates12_checked : candidates edge 4 127 1987 root28Tail13 winner.bound = root28Candidates12 := by
  unfold candidates
  have row : edge 1987 = (SortedNatTree.contains FastRows.row1987) := by
    change Data.row1987.contains = (SortedNatTree.contains FastRows.row1987)
    exact FastRows.row1987_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root28Child12 : search edge 3 [127, 1987] root28Candidates12 2114 winner = winner := by decide
theorem root28Step12 : scan edge (search edge 3) 4 [127] 127 root28Tail12 winner =
    scan edge (search edge 3) 4 [127] 127 root28Tail13 winner := by
  rw [root28Tail12, scan, if_neg (by decide), root28Candidates12_checked]
  change scan edge (search edge 3) 4 [127] 127 root28Tail13
    (if 3 ≤ root28Candidates12.length then search edge 3 [127, 1987] root28Candidates12 2114 winner else winner) = _
  rw [if_pos (by decide), root28Child12]
def root28Candidates13 : List Nat := [6361, 6481]
set_option maxRecDepth 8192 in
theorem root28Candidates13_checked : candidates edge 4 127 2281 root28Tail14 winner.bound = root28Candidates13 := by
  unfold candidates
  have row : edge 2281 = (SortedNatTree.contains FastRows.row2281) := by
    change Data.row2281.contains = (SortedNatTree.contains FastRows.row2281)
    exact FastRows.row2281_checked.symm
  rw [row]
  decide
theorem root28Step13 : scan edge (search edge 3) 4 [127] 127 root28Tail13 winner =
    scan edge (search edge 3) 4 [127] 127 root28Tail14 winner := by
  rw [root28Tail13, scan, if_neg (by decide), root28Candidates13_checked]
  change scan edge (search edge 3) 4 [127] 127 root28Tail14
    (if 3 ≤ root28Candidates13.length then search edge 3 [127, 2281] root28Candidates13 2408 winner else winner) = _
  rw [if_neg (by decide)]
def root28Candidates14 : List Nat := [3541]
set_option maxRecDepth 8192 in
theorem root28Candidates14_checked : candidates edge 4 127 2749 root28Tail15 winner.bound = root28Candidates14 := by
  unfold candidates
  have row : edge 2749 = (SortedNatTree.contains FastRows.row2749) := by
    change Data.row2749.contains = (SortedNatTree.contains FastRows.row2749)
    exact FastRows.row2749_checked.symm
  rw [row]
  decide
theorem root28Step14 : scan edge (search edge 3) 4 [127] 127 root28Tail14 winner =
    scan edge (search edge 3) 4 [127] 127 root28Tail15 winner := by
  rw [root28Tail14, scan, if_neg (by decide), root28Candidates14_checked]
  change scan edge (search edge 3) 4 [127] 127 root28Tail15
    (if 3 ≤ root28Candidates14.length then search edge 3 [127, 2749] root28Candidates14 2876 winner else winner) = _
  rw [if_neg (by decide)]
def root28Candidates15 : List Nat := [5749, 6733, 16981]
set_option maxRecDepth 8192 in
theorem root28Candidates15_checked : candidates edge 4 127 3001 root28Tail16 winner.bound = root28Candidates15 := by
  unfold candidates
  have row : edge 3001 = (SortedNatTree.contains FastRows.row3001) := by
    change Data.row3001.contains = (SortedNatTree.contains FastRows.row3001)
    exact FastRows.row3001_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root28Child15 : search edge 3 [127, 3001] root28Candidates15 3128 winner = winner := by decide
theorem root28Step15 : scan edge (search edge 3) 4 [127] 127 root28Tail15 winner =
    scan edge (search edge 3) 4 [127] 127 root28Tail16 winner := by
  rw [root28Tail15, scan, if_neg (by decide), root28Candidates15_checked]
  change scan edge (search edge 3) 4 [127] 127 root28Tail16
    (if 3 ≤ root28Candidates15.length then search edge 3 [127, 3001] root28Candidates15 3128 winner else winner) = _
  rw [if_pos (by decide), root28Child15]
def root28Candidates16 : List Nat := [19891]
set_option maxRecDepth 8192 in
theorem root28Candidates16_checked : candidates edge 4 127 3343 root28Tail17 winner.bound = root28Candidates16 := by
  unfold candidates
  have row : edge 3343 = (SortedNatTree.contains FastRows.row3343) := by
    change Data.row3343.contains = (SortedNatTree.contains FastRows.row3343)
    exact FastRows.row3343_checked.symm
  rw [row]
  decide
theorem root28Step16 : scan edge (search edge 3) 4 [127] 127 root28Tail16 winner =
    scan edge (search edge 3) 4 [127] 127 root28Tail17 winner := by
  rw [root28Tail16, scan, if_neg (by decide), root28Candidates16_checked]
  change scan edge (search edge 3) 4 [127] 127 root28Tail17
    (if 3 ≤ root28Candidates16.length then search edge 3 [127, 3343] root28Candidates16 3470 winner else winner) = _
  rw [if_neg (by decide)]
def root28Candidates17 : List Nat := [10177, 10597, 11701]
set_option maxRecDepth 8192 in
theorem root28Candidates17_checked : candidates edge 4 127 3541 root28Tail18 winner.bound = root28Candidates17 := by
  unfold candidates
  have row : edge 3541 = (SortedNatTree.contains FastRows.row3541) := by
    change Data.row3541.contains = (SortedNatTree.contains FastRows.row3541)
    exact FastRows.row3541_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root28Child17 : search edge 3 [127, 3541] root28Candidates17 3668 winner = winner := by decide
theorem root28Step17 : scan edge (search edge 3) 4 [127] 127 root28Tail17 winner =
    scan edge (search edge 3) 4 [127] 127 root28Tail18 winner := by
  rw [root28Tail17, scan, if_neg (by decide), root28Candidates17_checked]
  change scan edge (search edge 3) 4 [127] 127 root28Tail18
    (if 3 ≤ root28Candidates17.length then search edge 3 [127, 3541] root28Candidates17 3668 winner else winner) = _
  rw [if_pos (by decide), root28Child17]
def root28Candidates18 : List Nat := [4759]
set_option maxRecDepth 8192 in
theorem root28Candidates18_checked : candidates edge 4 127 3673 root28Tail19 winner.bound = root28Candidates18 := by
  unfold candidates
  have row : edge 3673 = (SortedNatTree.contains FastRows.row3673) := by
    change Data.row3673.contains = (SortedNatTree.contains FastRows.row3673)
    exact FastRows.row3673_checked.symm
  rw [row]
  decide
theorem root28Step18 : scan edge (search edge 3) 4 [127] 127 root28Tail18 winner =
    scan edge (search edge 3) 4 [127] 127 root28Tail19 winner := by
  rw [root28Tail18, scan, if_neg (by decide), root28Candidates18_checked]
  change scan edge (search edge 3) 4 [127] 127 root28Tail19
    (if 3 ≤ root28Candidates18.length then search edge 3 [127, 3673] root28Candidates18 3800 winner else winner) = _
  rw [if_neg (by decide)]
def root28Candidates19 : List Nat := [8713, 13873]
set_option maxRecDepth 8192 in
theorem root28Candidates19_checked : candidates edge 4 127 4363 root28Tail20 winner.bound = root28Candidates19 := by
  unfold candidates
  have row : edge 4363 = (SortedNatTree.contains FastRows.row4363) := by
    change Data.row4363.contains = (SortedNatTree.contains FastRows.row4363)
    exact FastRows.row4363_checked.symm
  rw [row]
  decide
theorem root28Step19 : scan edge (search edge 3) 4 [127] 127 root28Tail19 winner =
    scan edge (search edge 3) 4 [127] 127 root28Tail20 winner := by
  rw [root28Tail19, scan, if_neg (by decide), root28Candidates19_checked]
  change scan edge (search edge 3) 4 [127] 127 root28Tail20
    (if 3 ≤ root28Candidates19.length then search edge 3 [127, 4363] root28Candidates19 4490 winner else winner) = _
  rw [if_neg (by decide)]
def root28Candidates20 : List Nat := [9337]
set_option maxRecDepth 8192 in
theorem root28Candidates20_checked : candidates edge 4 127 4423 root28Tail21 winner.bound = root28Candidates20 := by
  unfold candidates
  have row : edge 4423 = (SortedNatTree.contains FastRows.row4423) := by
    change Data.row4423.contains = (SortedNatTree.contains FastRows.row4423)
    exact FastRows.row4423_checked.symm
  rw [row]
  decide
theorem root28Step20 : scan edge (search edge 3) 4 [127] 127 root28Tail20 winner =
    scan edge (search edge 3) 4 [127] 127 root28Tail21 winner := by
  rw [root28Tail20, scan, if_neg (by decide), root28Candidates20_checked]
  change scan edge (search edge 3) 4 [127] 127 root28Tail21
    (if 3 ≤ root28Candidates20.length then search edge 3 [127, 4423] root28Candidates20 4550 winner else winner) = _
  rw [if_neg (by decide)]
def root28Candidates21 : List Nat := [20047]
set_option maxRecDepth 8192 in
theorem root28Candidates21_checked : candidates edge 4 127 4759 root28Tail22 winner.bound = root28Candidates21 := by
  unfold candidates
  have row : edge 4759 = (SortedNatTree.contains FastRows.row4759) := by
    change Data.row4759.contains = (SortedNatTree.contains FastRows.row4759)
    exact FastRows.row4759_checked.symm
  rw [row]
  decide
theorem root28Step21 : scan edge (search edge 3) 4 [127] 127 root28Tail21 winner =
    scan edge (search edge 3) 4 [127] 127 root28Tail22 winner := by
  rw [root28Tail21, scan, if_neg (by decide), root28Candidates21_checked]
  change scan edge (search edge 3) 4 [127] 127 root28Tail22
    (if 3 ≤ root28Candidates21.length then search edge 3 [127, 4759] root28Candidates21 4886 winner else winner) = _
  rw [if_neg (by decide)]
def root28Candidates22 : List Nat := [5683, 9043, 9337, 11287]
set_option maxRecDepth 8192 in
theorem root28Candidates22_checked : candidates edge 4 127 5179 root28Tail23 winner.bound = root28Candidates22 := by
  unfold candidates
  have row : edge 5179 = (SortedNatTree.contains FastRows.row5179) := by
    change Data.row5179.contains = (SortedNatTree.contains FastRows.row5179)
    exact FastRows.row5179_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root28Child22 : search edge 3 [127, 5179] root28Candidates22 5306 winner = winner := by decide
theorem root28Step22 : scan edge (search edge 3) 4 [127] 127 root28Tail22 winner =
    scan edge (search edge 3) 4 [127] 127 root28Tail23 winner := by
  rw [root28Tail22, scan, if_neg (by decide), root28Candidates22_checked]
  change scan edge (search edge 3) 4 [127] 127 root28Tail23
    (if 3 ≤ root28Candidates22.length then search edge 3 [127, 5179] root28Candidates22 5306 winner else winner) = _
  rw [if_pos (by decide), root28Child22]
def root28Candidates23 : List Nat := [12241, 12703]
set_option maxRecDepth 8192 in
theorem root28Candidates23_checked : candidates edge 4 127 5563 root28Tail24 winner.bound = root28Candidates23 := by
  unfold candidates
  have row : edge 5563 = (SortedNatTree.contains FastRows.row5563) := by
    change Data.row5563.contains = (SortedNatTree.contains FastRows.row5563)
    exact FastRows.row5563_checked.symm
  rw [row]
  decide
theorem root28Step23 : scan edge (search edge 3) 4 [127] 127 root28Tail23 winner =
    scan edge (search edge 3) 4 [127] 127 root28Tail24 winner := by
  rw [root28Tail23, scan, if_neg (by decide), root28Candidates23_checked]
  change scan edge (search edge 3) 4 [127] 127 root28Tail24
    (if 3 ≤ root28Candidates23.length then search edge 3 [127, 5563] root28Candidates23 5690 winner else winner) = _
  rw [if_neg (by decide)]
def root28Candidates24 : List Nat := [6271, 7369]
set_option maxRecDepth 8192 in
theorem root28Candidates24_checked : candidates edge 4 127 5683 root28Tail25 winner.bound = root28Candidates24 := by
  unfold candidates
  have row : edge 5683 = (SortedNatTree.contains FastRows.row5683) := by
    change Data.row5683.contains = (SortedNatTree.contains FastRows.row5683)
    exact FastRows.row5683_checked.symm
  rw [row]
  decide
theorem root28Step24 : scan edge (search edge 3) 4 [127] 127 root28Tail24 winner =
    scan edge (search edge 3) 4 [127] 127 root28Tail25 winner := by
  rw [root28Tail24, scan, if_neg (by decide), root28Candidates24_checked]
  change scan edge (search edge 3) 4 [127] 127 root28Tail25
    (if 3 ≤ root28Candidates24.length then search edge 3 [127, 5683] root28Candidates24 5810 winner else winner) = _
  rw [if_neg (by decide)]
def root28Candidates25 : List Nat := [7753, 13873]
set_option maxRecDepth 8192 in
theorem root28Candidates25_checked : candidates edge 4 127 5749 root28Tail26 winner.bound = root28Candidates25 := by
  unfold candidates
  have row : edge 5749 = (SortedNatTree.contains FastRows.row5749) := by
    change Data.row5749.contains = (SortedNatTree.contains FastRows.row5749)
    exact FastRows.row5749_checked.symm
  rw [row]
  decide
theorem root28Step25 : scan edge (search edge 3) 4 [127] 127 root28Tail25 winner =
    scan edge (search edge 3) 4 [127] 127 root28Tail26 winner := by
  rw [root28Tail25, scan, if_neg (by decide), root28Candidates25_checked]
  change scan edge (search edge 3) 4 [127] 127 root28Tail26
    (if 3 ≤ root28Candidates25.length then search edge 3 [127, 5749] root28Candidates25 5876 winner else winner) = _
  rw [if_neg (by decide)]
def root28Candidates26 : List Nat := [19387, 19891]
set_option maxRecDepth 8192 in
theorem root28Candidates26_checked : candidates edge 4 127 5779 root28Tail27 winner.bound = root28Candidates26 := by
  unfold candidates
  have row : edge 5779 = (SortedNatTree.contains FastRows.row5779) := by
    change Data.row5779.contains = (SortedNatTree.contains FastRows.row5779)
    exact FastRows.row5779_checked.symm
  rw [row]
  decide
theorem root28Step26 : scan edge (search edge 3) 4 [127] 127 root28Tail26 winner =
    scan edge (search edge 3) 4 [127] 127 root28Tail27 winner := by
  rw [root28Tail26, scan, if_neg (by decide), root28Candidates26_checked]
  change scan edge (search edge 3) 4 [127] 127 root28Tail27
    (if 3 ≤ root28Candidates26.length then search edge 3 [127, 5779] root28Candidates26 5906 winner else winner) = _
  rw [if_neg (by decide)]
def root28Candidates27 : List Nat := []
set_option maxRecDepth 8192 in
theorem root28Candidates27_checked : candidates edge 4 127 6271 root28Tail28 winner.bound = root28Candidates27 := by
  unfold candidates
  have row : edge 6271 = (SortedNatTree.contains FastRows.row6271) := by
    change Data.row6271.contains = (SortedNatTree.contains FastRows.row6271)
    exact FastRows.row6271_checked.symm
  rw [row]
  decide
theorem root28Step27 : scan edge (search edge 3) 4 [127] 127 root28Tail27 winner =
    scan edge (search edge 3) 4 [127] 127 root28Tail28 winner := by
  rw [root28Tail27, scan, if_neg (by decide), root28Candidates27_checked]
  change scan edge (search edge 3) 4 [127] 127 root28Tail28
    (if 3 ≤ root28Candidates27.length then search edge 3 [127, 6271] root28Candidates27 6398 winner else winner) = _
  rw [if_neg (by decide)]
def root28Candidates28 : List Nat := [6481, 9547]
set_option maxRecDepth 8192 in
theorem root28Candidates28_checked : candidates edge 4 127 6361 root28Tail29 winner.bound = root28Candidates28 := by
  unfold candidates
  have row : edge 6361 = (SortedNatTree.contains FastRows.row6361) := by
    change Data.row6361.contains = (SortedNatTree.contains FastRows.row6361)
    exact FastRows.row6361_checked.symm
  rw [row]
  decide
theorem root28Step28 : scan edge (search edge 3) 4 [127] 127 root28Tail28 winner =
    scan edge (search edge 3) 4 [127] 127 root28Tail29 winner := by
  rw [root28Tail28, scan, if_neg (by decide), root28Candidates28_checked]
  change scan edge (search edge 3) 4 [127] 127 root28Tail29
    (if 3 ≤ root28Candidates28.length then search edge 3 [127, 6361] root28Candidates28 6488 winner else winner) = _
  rw [if_neg (by decide)]
theorem root28Scan29 : scan edge (search edge 3) 4 [127] 127 root28Tail29 winner = winner := by rw [root28Tail29, scan, if_pos (by decide)]
theorem root28Scan28 : scan edge (search edge 3) 4 [127] 127 root28Tail28 winner = winner :=
  (root28Step28).trans root28Scan29
theorem root28Scan27 : scan edge (search edge 3) 4 [127] 127 root28Tail27 winner = winner :=
  (root28Step27).trans root28Scan28
theorem root28Scan26 : scan edge (search edge 3) 4 [127] 127 root28Tail26 winner = winner :=
  (root28Step26).trans root28Scan27
theorem root28Scan25 : scan edge (search edge 3) 4 [127] 127 root28Tail25 winner = winner :=
  (root28Step25).trans root28Scan26
theorem root28Scan24 : scan edge (search edge 3) 4 [127] 127 root28Tail24 winner = winner :=
  (root28Step24).trans root28Scan25
theorem root28Scan23 : scan edge (search edge 3) 4 [127] 127 root28Tail23 winner = winner :=
  (root28Step23).trans root28Scan24
theorem root28Scan22 : scan edge (search edge 3) 4 [127] 127 root28Tail22 winner = winner :=
  (root28Step22).trans root28Scan23
theorem root28Scan21 : scan edge (search edge 3) 4 [127] 127 root28Tail21 winner = winner :=
  (root28Step21).trans root28Scan22
theorem root28Scan20 : scan edge (search edge 3) 4 [127] 127 root28Tail20 winner = winner :=
  (root28Step20).trans root28Scan21
theorem root28Scan19 : scan edge (search edge 3) 4 [127] 127 root28Tail19 winner = winner :=
  (root28Step19).trans root28Scan20
theorem root28Scan18 : scan edge (search edge 3) 4 [127] 127 root28Tail18 winner = winner :=
  (root28Step18).trans root28Scan19
theorem root28Scan17 : scan edge (search edge 3) 4 [127] 127 root28Tail17 winner = winner :=
  (root28Step17).trans root28Scan18
theorem root28Scan16 : scan edge (search edge 3) 4 [127] 127 root28Tail16 winner = winner :=
  (root28Step16).trans root28Scan17
theorem root28Scan15 : scan edge (search edge 3) 4 [127] 127 root28Tail15 winner = winner :=
  (root28Step15).trans root28Scan16
theorem root28Scan14 : scan edge (search edge 3) 4 [127] 127 root28Tail14 winner = winner :=
  (root28Step14).trans root28Scan15
theorem root28Scan13 : scan edge (search edge 3) 4 [127] 127 root28Tail13 winner = winner :=
  (root28Step13).trans root28Scan14
theorem root28Scan12 : scan edge (search edge 3) 4 [127] 127 root28Tail12 winner = winner :=
  (root28Step12).trans root28Scan13
theorem root28Scan11 : scan edge (search edge 3) 4 [127] 127 root28Tail11 winner = winner :=
  (root28Step11).trans root28Scan12
theorem root28Scan10 : scan edge (search edge 3) 4 [127] 127 root28Tail10 winner = winner :=
  (root28Step10).trans root28Scan11
theorem root28Scan9 : scan edge (search edge 3) 4 [127] 127 root28Tail9 winner = winner :=
  (root28Step9).trans root28Scan10
theorem root28Scan8 : scan edge (search edge 3) 4 [127] 127 root28Tail8 winner = winner :=
  (root28Step8).trans root28Scan9
theorem root28Scan7 : scan edge (search edge 3) 4 [127] 127 root28Tail7 winner = winner :=
  (root28Step7).trans root28Scan8
theorem root28Scan6 : scan edge (search edge 3) 4 [127] 127 root28Tail6 winner = winner :=
  (root28Step6).trans root28Scan7
theorem root28Scan5 : scan edge (search edge 3) 4 [127] 127 root28Tail5 winner = winner :=
  (root28Step5).trans root28Scan6
theorem root28Scan4 : scan edge (search edge 3) 4 [127] 127 root28Tail4 winner = winner :=
  (root28Step4).trans root28Scan5
theorem root28Scan3 : scan edge (search edge 3) 4 [127] 127 root28Tail3 winner = winner :=
  (root28Step3).trans root28Scan4
theorem root28Scan2 : scan edge (search edge 3) 4 [127] 127 root28Tail2 winner = winner :=
  (root28Step2).trans root28Scan3
theorem root28Scan1 : scan edge (search edge 3) 4 [127] 127 root28Tail1 winner = winner :=
  (root28Step1).trans root28Scan2
theorem root28Scan0 : scan edge (search edge 3) 4 [127] 127 root28Tail0 winner = winner :=
  (root28Step0).trans root28Scan1
set_option maxRecDepth 4096 in
theorem root28Branch : search edge 4 [127] root28Candidates 127 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [127] 127 root28Tail0 winner = winner
  exact root28Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
