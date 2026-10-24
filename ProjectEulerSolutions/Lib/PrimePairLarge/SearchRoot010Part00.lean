import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot010Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root10Candidates0 : List Nat := [593, 719, 2939, 4691, 11867, 22709, 23087]
set_option maxRecDepth 8192 in
theorem root10Candidates0_checked : candidates edge 4 41 227 root10Tail1 winner.bound = root10Candidates0 := by
  unfold candidates
  have row : edge 227 = (SortedNatTree.contains FastRows.row227) := by
    change Data.row227.contains = (SortedNatTree.contains FastRows.row227)
    exact FastRows.row227_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root10Child0 : search edge 3 [41, 227] root10Candidates0 268 winner = winner := by decide
theorem root10Step0 : scan edge (search edge 3) 4 [41] 41 root10Tail0 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail1 winner := by
  rw [root10Tail0, scan, if_neg (by decide), root10Candidates0_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail1
    (if 3 ≤ root10Candidates0.length then search edge 3 [41, 227] root10Candidates0 268 winner else winner) = _
  rw [if_pos (by decide), root10Child0]
def root10Candidates1 : List Nat := [1823, 2939, 8423, 15791]
set_option maxRecDepth 8192 in
theorem root10Candidates1_checked : candidates edge 4 41 257 root10Tail2 winner.bound = root10Candidates1 := by
  unfold candidates
  have row : edge 257 = (SortedNatTree.contains FastRows.row257) := by
    change Data.row257.contains = (SortedNatTree.contains FastRows.row257)
    exact FastRows.row257_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root10Child1 : search edge 3 [41, 257] root10Candidates1 298 winner = winner := by decide
theorem root10Step1 : scan edge (search edge 3) 4 [41] 41 root10Tail1 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail2 winner := by
  rw [root10Tail1, scan, if_neg (by decide), root10Candidates1_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail2
    (if 3 ≤ root10Candidates1.length then search edge 3 [41, 257] root10Candidates1 298 winner else winner) = _
  rw [if_pos (by decide), root10Child1]
def root10Candidates2 : List Nat := [863, 2411, 3461, 12263, 12497]
set_option maxRecDepth 8192 in
theorem root10Candidates2_checked : candidates edge 4 41 593 root10Tail3 winner.bound = root10Candidates2 := by
  unfold candidates
  have row : edge 593 = (SortedNatTree.contains FastRows.row593) := by
    change Data.row593.contains = (SortedNatTree.contains FastRows.row593)
    exact FastRows.row593_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root10Child2 : search edge 3 [41, 593] root10Candidates2 634 winner = winner := by decide
theorem root10Step2 : scan edge (search edge 3) 4 [41] 41 root10Tail2 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail3 winner := by
  rw [root10Tail2, scan, if_neg (by decide), root10Candidates2_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail3
    (if 3 ≤ root10Candidates2.length then search edge 3 [41, 593] root10Candidates2 634 winner else winner) = _
  rw [if_pos (by decide), root10Child2]
def root10Candidates3 : List Nat := [911, 2333, 7793, 12413, 14009, 19697, 22973]
set_option maxRecDepth 8192 in
theorem root10Candidates3_checked : candidates edge 4 41 719 root10Tail4 winner.bound = root10Candidates3 := by
  unfold candidates
  have row : edge 719 = (SortedNatTree.contains FastRows.row719) := by
    change Data.row719.contains = (SortedNatTree.contains FastRows.row719)
    exact FastRows.row719_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root10Child3 : search edge 3 [41, 719] root10Candidates3 760 winner = winner := by decide
theorem root10Step3 : scan edge (search edge 3) 4 [41] 41 root10Tail3 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail4 winner := by
  rw [root10Tail3, scan, if_neg (by decide), root10Candidates3_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail4
    (if 3 ≤ root10Candidates3.length then search edge 3 [41, 719] root10Candidates3 760 winner else winner) = _
  rw [if_pos (by decide), root10Child3]
def root10Candidates4 : List Nat := [1823, 5717, 15263, 16187, 20231, 24443]
set_option maxRecDepth 8192 in
theorem root10Candidates4_checked : candidates edge 4 41 863 root10Tail5 winner.bound = root10Candidates4 := by
  unfold candidates
  have row : edge 863 = (SortedNatTree.contains FastRows.row863) := by
    change Data.row863.contains = (SortedNatTree.contains FastRows.row863)
    exact FastRows.row863_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root10Child4 : search edge 3 [41, 863] root10Candidates4 904 winner = winner := by decide
theorem root10Step4 : scan edge (search edge 3) 4 [41] 41 root10Tail4 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail5 winner := by
  rw [root10Tail4, scan, if_neg (by decide), root10Candidates4_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail5
    (if 3 ≤ root10Candidates4.length then search edge 3 [41, 863] root10Candidates4 904 winner else winner) = _
  rw [if_pos (by decide), root10Child4]
def root10Candidates5 : List Nat := [4049, 8423, 12413]
set_option maxRecDepth 8192 in
theorem root10Candidates5_checked : candidates edge 4 41 887 root10Tail6 winner.bound = root10Candidates5 := by
  unfold candidates
  have row : edge 887 = (SortedNatTree.contains FastRows.row887) := by
    change Data.row887.contains = (SortedNatTree.contains FastRows.row887)
    exact FastRows.row887_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root10Child5 : search edge 3 [41, 887] root10Candidates5 928 winner = winner := by decide
theorem root10Step5 : scan edge (search edge 3) 4 [41] 41 root10Tail5 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail6 winner := by
  rw [root10Tail5, scan, if_neg (by decide), root10Candidates5_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail6
    (if 3 ≤ root10Candidates5.length then search edge 3 [41, 887] root10Candidates5 928 winner else winner) = _
  rw [if_pos (by decide), root10Child5]
def root10Candidates6 : List Nat := [6089, 16187]
set_option maxRecDepth 8192 in
theorem root10Candidates6_checked : candidates edge 4 41 911 root10Tail7 winner.bound = root10Candidates6 := by
  unfold candidates
  have row : edge 911 = (SortedNatTree.contains FastRows.row911) := by
    change Data.row911.contains = (SortedNatTree.contains FastRows.row911)
    exact FastRows.row911_checked.symm
  rw [row]
  decide
theorem root10Step6 : scan edge (search edge 3) 4 [41] 41 root10Tail6 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail7 winner := by
  rw [root10Tail6, scan, if_neg (by decide), root10Candidates6_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail7
    (if 3 ≤ root10Candidates6.length then search edge 3 [41, 911] root10Candidates6 952 winner else winner) = _
  rw [if_neg (by decide)]
def root10Candidates7 : List Nat := [6719, 7187, 7331, 8867, 22709]
set_option maxRecDepth 8192 in
theorem root10Candidates7_checked : candidates edge 4 41 1013 root10Tail8 winner.bound = root10Candidates7 := by
  unfold candidates
  have row : edge 1013 = (SortedNatTree.contains FastRows.row1013) := by
    change Data.row1013.contains = (SortedNatTree.contains FastRows.row1013)
    exact FastRows.row1013_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root10Child7 : search edge 3 [41, 1013] root10Candidates7 1054 winner = winner := by decide
theorem root10Step7 : scan edge (search edge 3) 4 [41] 41 root10Tail7 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail8 winner := by
  rw [root10Tail7, scan, if_neg (by decide), root10Candidates7_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail8
    (if 3 ≤ root10Candidates7.length then search edge 3 [41, 1013] root10Candidates7 1054 winner else winner) = _
  rw [if_pos (by decide), root10Child7]
def root10Candidates8 : List Nat := [3533, 11867]
set_option maxRecDepth 8192 in
theorem root10Candidates8_checked : candidates edge 4 41 1031 root10Tail9 winner.bound = root10Candidates8 := by
  unfold candidates
  have row : edge 1031 = (SortedNatTree.contains FastRows.row1031) := by
    change Data.row1031.contains = (SortedNatTree.contains FastRows.row1031)
    exact FastRows.row1031_checked.symm
  rw [row]
  decide
theorem root10Step8 : scan edge (search edge 3) 4 [41] 41 root10Tail8 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail9 winner := by
  rw [root10Tail8, scan, if_neg (by decide), root10Candidates8_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail9
    (if 3 ≤ root10Candidates8.length then search edge 3 [41, 1031] root10Candidates8 1072 winner else winner) = _
  rw [if_neg (by decide)]
def root10Candidates9 : List Nat := [2609, 9467, 12263, 14741, 17489, 17783]
set_option maxRecDepth 8192 in
theorem root10Candidates9_checked : candidates edge 4 41 1259 root10Tail10 winner.bound = root10Candidates9 := by
  unfold candidates
  have row : edge 1259 = (SortedNatTree.contains FastRows.row1259) := by
    change Data.row1259.contains = (SortedNatTree.contains FastRows.row1259)
    exact FastRows.row1259_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root10Child9 : search edge 3 [41, 1259] root10Candidates9 1300 winner = winner := by decide
theorem root10Step9 : scan edge (search edge 3) 4 [41] 41 root10Tail9 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail10 winner := by
  rw [root10Tail9, scan, if_neg (by decide), root10Candidates9_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail10
    (if 3 ≤ root10Candidates9.length then search edge 3 [41, 1259] root10Candidates9 1300 winner else winner) = _
  rw [if_pos (by decide), root10Child9]
def root10Candidates10 : List Nat := [1637, 4451, 7643, 17657]
set_option maxRecDepth 8192 in
theorem root10Candidates10_checked : candidates edge 4 41 1583 root10Tail11 winner.bound = root10Candidates10 := by
  unfold candidates
  have row : edge 1583 = (SortedNatTree.contains FastRows.row1583) := by
    change Data.row1583.contains = (SortedNatTree.contains FastRows.row1583)
    exact FastRows.row1583_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root10Child10 : search edge 3 [41, 1583] root10Candidates10 1624 winner = winner := by decide
theorem root10Step10 : scan edge (search edge 3) 4 [41] 41 root10Tail10 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail11 winner := by
  rw [root10Tail10, scan, if_neg (by decide), root10Candidates10_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail11
    (if 3 ≤ root10Candidates10.length then search edge 3 [41, 1583] root10Candidates10 1624 winner else winner) = _
  rw [if_pos (by decide), root10Child10]
def root10Candidates11 : List Nat := [1637, 23957]
set_option maxRecDepth 8192 in
theorem root10Candidates11_checked : candidates edge 4 41 1613 root10Tail12 winner.bound = root10Candidates11 := by
  unfold candidates
  have row : edge 1613 = (SortedNatTree.contains FastRows.row1613) := by
    change Data.row1613.contains = (SortedNatTree.contains FastRows.row1613)
    exact FastRows.row1613_checked.symm
  rw [row]
  decide
theorem root10Step11 : scan edge (search edge 3) 4 [41] 41 root10Tail11 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail12 winner := by
  rw [root10Tail11, scan, if_neg (by decide), root10Candidates11_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail12
    (if 3 ≤ root10Candidates11.length then search edge 3 [41, 1613] root10Candidates11 1654 winner else winner) = _
  rw [if_neg (by decide)]
def root10Candidates12 : List Nat := [8933, 14741]
set_option maxRecDepth 8192 in
theorem root10Candidates12_checked : candidates edge 4 41 1637 root10Tail13 winner.bound = root10Candidates12 := by
  unfold candidates
  have row : edge 1637 = (SortedNatTree.contains FastRows.row1637) := by
    change Data.row1637.contains = (SortedNatTree.contains FastRows.row1637)
    exact FastRows.row1637_checked.symm
  rw [row]
  decide
theorem root10Step12 : scan edge (search edge 3) 4 [41] 41 root10Tail12 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail13 winner := by
  rw [root10Tail12, scan, if_neg (by decide), root10Candidates12_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail13
    (if 3 ≤ root10Candidates12.length then search edge 3 [41, 1637] root10Candidates12 1678 winner else winner) = _
  rw [if_neg (by decide)]
def root10Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root10Candidates13_checked : candidates edge 4 41 1667 root10Tail14 winner.bound = root10Candidates13 := by
  unfold candidates
  have row : edge 1667 = (SortedNatTree.contains FastRows.row1667) := by
    change Data.row1667.contains = (SortedNatTree.contains FastRows.row1667)
    exact FastRows.row1667_checked.symm
  rw [row]
  decide
theorem root10Step13 : scan edge (search edge 3) 4 [41] 41 root10Tail13 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail14 winner := by
  rw [root10Tail13, scan, if_neg (by decide), root10Candidates13_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail14
    (if 3 ≤ root10Candidates13.length then search edge 3 [41, 1667] root10Candidates13 1708 winner else winner) = _
  rw [if_neg (by decide)]
def root10Candidates14 : List Nat := [3461, 6719, 8069]
set_option maxRecDepth 8192 in
theorem root10Candidates14_checked : candidates edge 4 41 1823 root10Tail15 winner.bound = root10Candidates14 := by
  unfold candidates
  have row : edge 1823 = (SortedNatTree.contains FastRows.row1823) := by
    change Data.row1823.contains = (SortedNatTree.contains FastRows.row1823)
    exact FastRows.row1823_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root10Child14 : search edge 3 [41, 1823] root10Candidates14 1864 winner = winner := by decide
theorem root10Step14 : scan edge (search edge 3) 4 [41] 41 root10Tail14 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail15 winner := by
  rw [root10Tail14, scan, if_neg (by decide), root10Candidates14_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail15
    (if 3 ≤ root10Candidates14.length then search edge 3 [41, 1823] root10Candidates14 1864 winner else winner) = _
  rw [if_pos (by decide), root10Child14]
def root10Candidates15 : List Nat := [4691, 7793, 11489, 22973]
set_option maxRecDepth 8192 in
theorem root10Candidates15_checked : candidates edge 4 41 2081 root10Tail16 winner.bound = root10Candidates15 := by
  unfold candidates
  have row : edge 2081 = (SortedNatTree.contains FastRows.row2081) := by
    change Data.row2081.contains = (SortedNatTree.contains FastRows.row2081)
    exact FastRows.row2081_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root10Child15 : search edge 3 [41, 2081] root10Candidates15 2122 winner = winner := by decide
theorem root10Step15 : scan edge (search edge 3) 4 [41] 41 root10Tail15 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail16 winner := by
  rw [root10Tail15, scan, if_neg (by decide), root10Candidates15_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail16
    (if 3 ≤ root10Candidates15.length then search edge 3 [41, 2081] root10Candidates15 2122 winner else winner) = _
  rw [if_pos (by decide), root10Child15]
def root10Candidates16 : List Nat := [2339, 3533, 8867, 9473, 11489, 15101, 20231]
set_option maxRecDepth 8192 in
theorem root10Candidates16_checked : candidates edge 4 41 2333 root10Tail17 winner.bound = root10Candidates16 := by
  unfold candidates
  have row : edge 2333 = (SortedNatTree.contains FastRows.row2333) := by
    change Data.row2333.contains = (SortedNatTree.contains FastRows.row2333)
    exact FastRows.row2333_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root10Child16 : search edge 3 [41, 2333] root10Candidates16 2374 winner = winner := by decide
theorem root10Step16 : scan edge (search edge 3) 4 [41] 41 root10Tail16 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail17 winner := by
  rw [root10Tail16, scan, if_neg (by decide), root10Candidates16_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail17
    (if 3 ≤ root10Candidates16.length then search edge 3 [41, 2333] root10Candidates16 2374 winner else winner) = _
  rw [if_pos (by decide), root10Child16]
def root10Candidates17 : List Nat := [14741, 21503]
set_option maxRecDepth 8192 in
theorem root10Candidates17_checked : candidates edge 4 41 2339 root10Tail18 winner.bound = root10Candidates17 := by
  unfold candidates
  have row : edge 2339 = (SortedNatTree.contains FastRows.row2339) := by
    change Data.row2339.contains = (SortedNatTree.contains FastRows.row2339)
    exact FastRows.row2339_checked.symm
  rw [row]
  decide
theorem root10Step17 : scan edge (search edge 3) 4 [41] 41 root10Tail17 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail18 winner := by
  rw [root10Tail17, scan, if_neg (by decide), root10Candidates17_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail18
    (if 3 ≤ root10Candidates17.length then search edge 3 [41, 2339] root10Candidates17 2380 winner else winner) = _
  rw [if_neg (by decide)]
def root10Candidates18 : List Nat := [9281, 18749]
set_option maxRecDepth 8192 in
theorem root10Candidates18_checked : candidates edge 4 41 2411 root10Tail19 winner.bound = root10Candidates18 := by
  unfold candidates
  have row : edge 2411 = (SortedNatTree.contains FastRows.row2411) := by
    change Data.row2411.contains = (SortedNatTree.contains FastRows.row2411)
    exact FastRows.row2411_checked.symm
  rw [row]
  decide
theorem root10Step18 : scan edge (search edge 3) 4 [41] 41 root10Tail18 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail19 winner := by
  rw [root10Tail18, scan, if_neg (by decide), root10Candidates18_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail19
    (if 3 ≤ root10Candidates18.length then search edge 3 [41, 2411] root10Candidates18 2452 winner else winner) = _
  rw [if_neg (by decide)]
def root10Candidates19 : List Nat := [7451, 8069, 22247, 22973]
set_option maxRecDepth 8192 in
theorem root10Candidates19_checked : candidates edge 4 41 2609 root10Tail20 winner.bound = root10Candidates19 := by
  unfold candidates
  have row : edge 2609 = (SortedNatTree.contains FastRows.row2609) := by
    change Data.row2609.contains = (SortedNatTree.contains FastRows.row2609)
    exact FastRows.row2609_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root10Child19 : search edge 3 [41, 2609] root10Candidates19 2650 winner = winner := by decide
theorem root10Step19 : scan edge (search edge 3) 4 [41] 41 root10Tail19 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail20 winner := by
  rw [root10Tail19, scan, if_neg (by decide), root10Candidates19_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail20
    (if 3 ≤ root10Candidates19.length then search edge 3 [41, 2609] root10Candidates19 2650 winner else winner) = _
  rw [if_pos (by decide), root10Child19]
def root10Candidates20 : List Nat := [4217, 4679]
set_option maxRecDepth 8192 in
theorem root10Candidates20_checked : candidates edge 4 41 2939 root10Tail21 winner.bound = root10Candidates20 := by
  unfold candidates
  have row : edge 2939 = (SortedNatTree.contains FastRows.row2939) := by
    change Data.row2939.contains = (SortedNatTree.contains FastRows.row2939)
    exact FastRows.row2939_checked.symm
  rw [row]
  decide
theorem root10Step20 : scan edge (search edge 3) 4 [41] 41 root10Tail20 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail21 winner := by
  rw [root10Tail20, scan, if_neg (by decide), root10Candidates20_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail21
    (if 3 ≤ root10Candidates20.length then search edge 3 [41, 2939] root10Candidates20 2980 winner else winner) = _
  rw [if_neg (by decide)]
def root10Candidates21 : List Nat := [15101]
set_option maxRecDepth 8192 in
theorem root10Candidates21_checked : candidates edge 4 41 3299 root10Tail22 winner.bound = root10Candidates21 := by
  unfold candidates
  have row : edge 3299 = (SortedNatTree.contains FastRows.row3299) := by
    change Data.row3299.contains = (SortedNatTree.contains FastRows.row3299)
    exact FastRows.row3299_checked.symm
  rw [row]
  decide
theorem root10Step21 : scan edge (search edge 3) 4 [41] 41 root10Tail21 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail22 winner := by
  rw [root10Tail21, scan, if_neg (by decide), root10Candidates21_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail22
    (if 3 ≤ root10Candidates21.length then search edge 3 [41, 3299] root10Candidates21 3340 winner else winner) = _
  rw [if_neg (by decide)]
def root10Candidates22 : List Nat := []
set_option maxRecDepth 8192 in
theorem root10Candidates22_checked : candidates edge 4 41 3461 root10Tail23 winner.bound = root10Candidates22 := by
  unfold candidates
  have row : edge 3461 = (SortedNatTree.contains FastRows.row3461) := by
    change Data.row3461.contains = (SortedNatTree.contains FastRows.row3461)
    exact FastRows.row3461_checked.symm
  rw [row]
  decide
theorem root10Step22 : scan edge (search edge 3) 4 [41] 41 root10Tail22 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail23 winner := by
  rw [root10Tail22, scan, if_neg (by decide), root10Candidates22_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail23
    (if 3 ≤ root10Candidates22.length then search edge 3 [41, 3461] root10Candidates22 3502 winner else winner) = _
  rw [if_neg (by decide)]
def root10Candidates23 : List Nat := [3533, 4457]
set_option maxRecDepth 8192 in
theorem root10Candidates23_checked : candidates edge 4 41 3527 root10Tail24 winner.bound = root10Candidates23 := by
  unfold candidates
  have row : edge 3527 = (SortedNatTree.contains FastRows.row3527) := by
    change Data.row3527.contains = (SortedNatTree.contains FastRows.row3527)
    exact FastRows.row3527_checked.symm
  rw [row]
  decide
theorem root10Step23 : scan edge (search edge 3) 4 [41] 41 root10Tail23 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail24 winner := by
  rw [root10Tail23, scan, if_neg (by decide), root10Candidates23_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail24
    (if 3 ≤ root10Candidates23.length then search edge 3 [41, 3527] root10Candidates23 3568 winner else winner) = _
  rw [if_neg (by decide)]
def root10Candidates24 : List Nat := [4679, 10163, 22247]
set_option maxRecDepth 8192 in
theorem root10Candidates24_checked : candidates edge 4 41 3533 root10Tail25 winner.bound = root10Candidates24 := by
  unfold candidates
  have row : edge 3533 = (SortedNatTree.contains FastRows.row3533) := by
    change Data.row3533.contains = (SortedNatTree.contains FastRows.row3533)
    exact FastRows.row3533_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root10Child24 : search edge 3 [41, 3533] root10Candidates24 3574 winner = winner := by decide
theorem root10Step24 : scan edge (search edge 3) 4 [41] 41 root10Tail24 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail25 winner := by
  rw [root10Tail24, scan, if_neg (by decide), root10Candidates24_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail25
    (if 3 ≤ root10Candidates24.length then search edge 3 [41, 3533] root10Candidates24 3574 winner else winner) = _
  rw [if_pos (by decide), root10Child24]
def root10Candidates25 : List Nat := [13259]
set_option maxRecDepth 8192 in
theorem root10Candidates25_checked : candidates edge 4 41 3719 root10Tail26 winner.bound = root10Candidates25 := by
  unfold candidates
  have row : edge 3719 = (SortedNatTree.contains FastRows.row3719) := by
    change Data.row3719.contains = (SortedNatTree.contains FastRows.row3719)
    exact FastRows.row3719_checked.symm
  rw [row]
  decide
theorem root10Step25 : scan edge (search edge 3) 4 [41] 41 root10Tail25 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail26 winner := by
  rw [root10Tail25, scan, if_neg (by decide), root10Candidates25_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail26
    (if 3 ≤ root10Candidates25.length then search edge 3 [41, 3719] root10Candidates25 3760 winner else winner) = _
  rw [if_neg (by decide)]
def root10Candidates26 : List Nat := [8069, 8867]
set_option maxRecDepth 8192 in
theorem root10Candidates26_checked : candidates edge 4 41 4013 root10Tail27 winner.bound = root10Candidates26 := by
  unfold candidates
  have row : edge 4013 = (SortedNatTree.contains FastRows.row4013) := by
    change Data.row4013.contains = (SortedNatTree.contains FastRows.row4013)
    exact FastRows.row4013_checked.symm
  rw [row]
  decide
theorem root10Step26 : scan edge (search edge 3) 4 [41] 41 root10Tail26 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail27 winner := by
  rw [root10Tail26, scan, if_neg (by decide), root10Candidates26_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail27
    (if 3 ≤ root10Candidates26.length then search edge 3 [41, 4013] root10Candidates26 4054 winner else winner) = _
  rw [if_neg (by decide)]
def root10Candidates27 : List Nat := [9623, 16649]
set_option maxRecDepth 8192 in
theorem root10Candidates27_checked : candidates edge 4 41 4049 root10Tail28 winner.bound = root10Candidates27 := by
  unfold candidates
  have row : edge 4049 = (SortedNatTree.contains FastRows.row4049) := by
    change Data.row4049.contains = (SortedNatTree.contains FastRows.row4049)
    exact FastRows.row4049_checked.symm
  rw [row]
  decide
theorem root10Step27 : scan edge (search edge 3) 4 [41] 41 root10Tail27 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail28 winner := by
  rw [root10Tail27, scan, if_neg (by decide), root10Candidates27_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail28
    (if 3 ≤ root10Candidates27.length then search edge 3 [41, 4049] root10Candidates27 4090 winner else winner) = _
  rw [if_neg (by decide)]
def root10Candidates28 : List Nat := [11489, 12413]
set_option maxRecDepth 8192 in
theorem root10Candidates28_checked : candidates edge 4 41 4217 root10Tail29 winner.bound = root10Candidates28 := by
  unfold candidates
  have row : edge 4217 = (SortedNatTree.contains FastRows.row4217) := by
    change Data.row4217.contains = (SortedNatTree.contains FastRows.row4217)
    exact FastRows.row4217_checked.symm
  rw [row]
  decide
theorem root10Step28 : scan edge (search edge 3) 4 [41] 41 root10Tail28 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail29 winner := by
  rw [root10Tail28, scan, if_neg (by decide), root10Candidates28_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail29
    (if 3 ≤ root10Candidates28.length then search edge 3 [41, 4217] root10Candidates28 4258 winner else winner) = _
  rw [if_neg (by decide)]
def root10Candidates29 : List Nat := [6089, 6833, 15053, 15791, 21503]
set_option maxRecDepth 8192 in
theorem root10Candidates29_checked : candidates edge 4 41 4451 root10Tail30 winner.bound = root10Candidates29 := by
  unfold candidates
  have row : edge 4451 = (SortedNatTree.contains FastRows.row4451) := by
    change Data.row4451.contains = (SortedNatTree.contains FastRows.row4451)
    exact FastRows.row4451_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root10Child29 : search edge 3 [41, 4451] root10Candidates29 4492 winner = winner := by decide
theorem root10Step29 : scan edge (search edge 3) 4 [41] 41 root10Tail29 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail30 winner := by
  rw [root10Tail29, scan, if_neg (by decide), root10Candidates29_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail30
    (if 3 ≤ root10Candidates29.length then search edge 3 [41, 4451] root10Candidates29 4492 winner else winner) = _
  rw [if_pos (by decide), root10Child29]
def root10Candidates30 : List Nat := [4679, 6473, 6491, 7331, 9281, 10163, 12413, 15773, 15791]
set_option maxRecDepth 8192 in
theorem root10Candidates30_checked : candidates edge 4 41 4457 root10Tail31 winner.bound = root10Candidates30 := by
  unfold candidates
  have row : edge 4457 = (SortedNatTree.contains FastRows.row4457) := by
    change Data.row4457.contains = (SortedNatTree.contains FastRows.row4457)
    exact FastRows.row4457_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root10Child30 : search edge 3 [41, 4457] root10Candidates30 4498 winner = winner := by decide
theorem root10Step30 : scan edge (search edge 3) 4 [41] 41 root10Tail30 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail31 winner := by
  rw [root10Tail30, scan, if_neg (by decide), root10Candidates30_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail31
    (if 3 ≤ root10Candidates30.length then search edge 3 [41, 4457] root10Candidates30 4498 winner else winner) = _
  rw [if_pos (by decide), root10Child30]
def root10Candidates31 : List Nat := [5717, 8867]
set_option maxRecDepth 8192 in
theorem root10Candidates31_checked : candidates edge 4 41 4679 root10Tail32 winner.bound = root10Candidates31 := by
  unfold candidates
  have row : edge 4679 = (SortedNatTree.contains FastRows.row4679) := by
    change Data.row4679.contains = (SortedNatTree.contains FastRows.row4679)
    exact FastRows.row4679_checked.symm
  rw [row]
  decide
theorem root10Step31 : scan edge (search edge 3) 4 [41] 41 root10Tail31 winner =
    scan edge (search edge 3) 4 [41] 41 root10Tail32 winner := by
  rw [root10Tail31, scan, if_neg (by decide), root10Candidates31_checked]
  change scan edge (search edge 3) 4 [41] 41 root10Tail32
    (if 3 ≤ root10Candidates31.length then search edge 3 [41, 4679] root10Candidates31 4720 winner else winner) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
