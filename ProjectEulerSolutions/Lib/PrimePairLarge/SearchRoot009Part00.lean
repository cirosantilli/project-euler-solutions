import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot009Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root9Candidates0 : List Nat := [1303, 2377, 4789, 5923, 8941, 20947]
set_option maxRecDepth 8192 in
theorem root9Candidates0_checked : candidates edge 4 37 67 root9Tail1 winner.bound = root9Candidates0 := by
  unfold candidates
  have row : edge 67 = (SortedNatTree.contains FastRows.row67) := by
    change Data.row67.contains = (SortedNatTree.contains FastRows.row67)
    exact FastRows.row67_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child0 : search edge 3 [37, 67] root9Candidates0 104 winner = winner := by decide
theorem root9Step0 : scan edge (search edge 3) 4 [37] 37 root9Tail0 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail1 winner := by
  rw [root9Tail0, scan, if_neg (by decide), root9Candidates0_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail1
    (if 3 ≤ root9Candidates0.length then search edge 3 [37, 67] root9Candidates0 104 winner else winner) = _
  rw [if_pos (by decide), root9Child0]
def root9Candidates1 : List Nat := [967, 1447, 2179, 2371, 2689, 3181, 5101, 5449, 6373, 9679, 9859, 14779, 17011, 24967, 25849]
set_option maxRecDepth 8192 in
theorem root9Candidates1_checked : candidates edge 4 37 79 root9Tail2 winner.bound = root9Candidates1 := by
  unfold candidates
  have row : edge 79 = (SortedNatTree.contains FastRows.row79) := by
    change Data.row79.contains = (SortedNatTree.contains FastRows.row79)
    exact FastRows.row79_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child1 : search edge 3 [37, 79] root9Candidates1 116 winner = winner := by decide
theorem root9Step1 : scan edge (search edge 3) 4 [37] 37 root9Tail1 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail2 winner := by
  rw [root9Tail1, scan, if_neg (by decide), root9Candidates1_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail2
    (if 3 ≤ root9Candidates1.length then search edge 3 [37, 79] root9Candidates1 116 winner else winner) = _
  rw [if_pos (by decide), root9Child1]
def root9Candidates2 : List Nat := [1153, 9721]
set_option maxRecDepth 8192 in
theorem root9Candidates2_checked : candidates edge 4 37 199 root9Tail3 winner.bound = root9Candidates2 := by
  unfold candidates
  have row : edge 199 = (SortedNatTree.contains FastRows.row199) := by
    change Data.row199.contains = (SortedNatTree.contains FastRows.row199)
    exact FastRows.row199_checked.symm
  rw [row]
  decide
theorem root9Step2 : scan edge (search edge 3) 4 [37] 37 root9Tail2 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail3 winner := by
  rw [root9Tail2, scan, if_neg (by decide), root9Candidates2_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail3
    (if 3 ≤ root9Candidates2.length then search edge 3 [37, 199] root9Candidates2 236 winner else winner) = _
  rw [if_neg (by decide)]
def root9Candidates3 : List Nat := [1471, 9859, 24967]
set_option maxRecDepth 8192 in
theorem root9Candidates3_checked : candidates edge 4 37 277 root9Tail4 winner.bound = root9Candidates3 := by
  unfold candidates
  have row : edge 277 = (SortedNatTree.contains FastRows.row277) := by
    change Data.row277.contains = (SortedNatTree.contains FastRows.row277)
    exact FastRows.row277_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child3 : search edge 3 [37, 277] root9Candidates3 314 winner = winner := by decide
theorem root9Step3 : scan edge (search edge 3) 4 [37] 37 root9Tail3 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail4 winner := by
  rw [root9Tail3, scan, if_neg (by decide), root9Candidates3_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail4
    (if 3 ≤ root9Candidates3.length then search edge 3 [37, 277] root9Candidates3 314 winner else winner) = _
  rw [if_pos (by decide), root9Child3]
def root9Candidates4 : List Nat := [991, 9187, 9811, 14827]
set_option maxRecDepth 8192 in
theorem root9Candidates4_checked : candidates edge 4 37 313 root9Tail5 winner.bound = root9Candidates4 := by
  unfold candidates
  have row : edge 313 = (SortedNatTree.contains FastRows.row313) := by
    change Data.row313.contains = (SortedNatTree.contains FastRows.row313)
    exact FastRows.row313_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child4 : search edge 3 [37, 313] root9Candidates4 350 winner = winner := by decide
theorem root9Step4 : scan edge (search edge 3) 4 [37] 37 root9Tail4 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail5 winner := by
  rw [root9Tail4, scan, if_neg (by decide), root9Candidates4_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail5
    (if 3 ≤ root9Candidates4.length then search edge 3 [37, 313] root9Candidates4 350 winner else winner) = _
  rw [if_pos (by decide), root9Child4]
def root9Candidates5 : List Nat := [2473, 5623, 8761, 9931, 20023]
set_option maxRecDepth 8192 in
theorem root9Candidates5_checked : candidates edge 4 37 463 root9Tail6 winner.bound = root9Candidates5 := by
  unfold candidates
  have row : edge 463 = (SortedNatTree.contains FastRows.row463) := by
    change Data.row463.contains = (SortedNatTree.contains FastRows.row463)
    exact FastRows.row463_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child5 : search edge 3 [37, 463] root9Candidates5 500 winner = winner := by decide
theorem root9Step5 : scan edge (search edge 3) 4 [37] 37 root9Tail5 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail6 winner := by
  rw [root9Tail5, scan, if_neg (by decide), root9Candidates5_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail6
    (if 3 ≤ root9Candidates5.length then search edge 3 [37, 463] root9Candidates5 500 winner else winner) = _
  rw [if_pos (by decide), root9Child5]
def root9Candidates6 : List Nat := [967, 991, 4603, 19273]
set_option maxRecDepth 8192 in
theorem root9Candidates6_checked : candidates edge 4 37 607 root9Tail7 winner.bound = root9Candidates6 := by
  unfold candidates
  have row : edge 607 = (SortedNatTree.contains FastRows.row607) := by
    change Data.row607.contains = (SortedNatTree.contains FastRows.row607)
    exact FastRows.row607_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child6 : search edge 3 [37, 607] root9Candidates6 644 winner = winner := by decide
theorem root9Step6 : scan edge (search edge 3) 4 [37] 37 root9Tail6 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail7 winner := by
  rw [root9Tail6, scan, if_neg (by decide), root9Candidates6_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail7
    (if 3 ≤ root9Candidates6.length then search edge 3 [37, 607] root9Candidates6 644 winner else winner) = _
  rw [if_pos (by decide), root9Child6]
def root9Candidates7 : List Nat := [1951, 3049, 5623, 5779, 5857, 8761, 24967]
set_option maxRecDepth 8192 in
theorem root9Candidates7_checked : candidates edge 4 37 967 root9Tail8 winner.bound = root9Candidates7 := by
  unfold candidates
  have row : edge 967 = (SortedNatTree.contains FastRows.row967) := by
    change Data.row967.contains = (SortedNatTree.contains FastRows.row967)
    exact FastRows.row967_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child7 : search edge 3 [37, 967] root9Candidates7 1004 winner = winner := by decide
theorem root9Step7 : scan edge (search edge 3) 4 [37] 37 root9Tail7 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail8 winner := by
  rw [root9Tail7, scan, if_neg (by decide), root9Candidates7_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail8
    (if 3 ≤ root9Candidates7.length then search edge 3 [37, 967] root9Candidates7 1004 winner else winner) = _
  rw [if_pos (by decide), root9Child7]
def root9Candidates8 : List Nat := [2269, 2689, 2803, 3613, 9199, 15739, 23053, 24967]
set_option maxRecDepth 8192 in
theorem root9Candidates8_checked : candidates edge 4 37 991 root9Tail9 winner.bound = root9Candidates8 := by
  unfold candidates
  have row : edge 991 = (SortedNatTree.contains FastRows.row991) := by
    change Data.row991.contains = (SortedNatTree.contains FastRows.row991)
    exact FastRows.row991_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child8 : search edge 3 [37, 991] root9Candidates8 1028 winner = winner := by decide
theorem root9Step8 : scan edge (search edge 3) 4 [37] 37 root9Tail8 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail9 winner := by
  rw [root9Tail8, scan, if_neg (by decide), root9Candidates8_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail9
    (if 3 ≤ root9Candidates8.length then search edge 3 [37, 991] root9Candidates8 1028 winner else winner) = _
  rw [if_pos (by decide), root9Child8]
def root9Candidates9 : List Nat := [1321, 4909, 6529, 9133, 14779, 15031, 17011, 18397]
set_option maxRecDepth 8192 in
theorem root9Candidates9_checked : candidates edge 4 37 1069 root9Tail10 winner.bound = root9Candidates9 := by
  unfold candidates
  have row : edge 1069 = (SortedNatTree.contains FastRows.row1069) := by
    change Data.row1069.contains = (SortedNatTree.contains FastRows.row1069)
    exact FastRows.row1069_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child9 : search edge 3 [37, 1069] root9Candidates9 1106 winner = winner := by decide
theorem root9Step9 : scan edge (search edge 3) 4 [37] 37 root9Tail9 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail10 winner := by
  rw [root9Tail9, scan, if_neg (by decide), root9Candidates9_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail10
    (if 3 ≤ root9Candidates9.length then search edge 3 [37, 1069] root9Candidates9 1106 winner else winner) = _
  rw [if_pos (by decide), root9Child9]
def root9Candidates10 : List Nat := [1699, 2473, 2803, 4093]
set_option maxRecDepth 8192 in
theorem root9Candidates10_checked : candidates edge 4 37 1153 root9Tail11 winner.bound = root9Candidates10 := by
  unfold candidates
  have row : edge 1153 = (SortedNatTree.contains FastRows.row1153) := by
    change Data.row1153.contains = (SortedNatTree.contains FastRows.row1153)
    exact FastRows.row1153_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child10 : search edge 3 [37, 1153] root9Candidates10 1190 winner = winner := by decide
theorem root9Step10 : scan edge (search edge 3) 4 [37] 37 root9Tail10 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail11 winner := by
  rw [root9Tail10, scan, if_neg (by decide), root9Candidates10_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail11
    (if 3 ≤ root9Candidates10.length then search edge 3 [37, 1153] root9Candidates10 1190 winner else winner) = _
  rw [if_pos (by decide), root9Child10]
def root9Candidates11 : List Nat := [1549, 3187, 8713, 9931]
set_option maxRecDepth 8192 in
theorem root9Candidates11_checked : candidates edge 4 37 1237 root9Tail12 winner.bound = root9Candidates11 := by
  unfold candidates
  have row : edge 1237 = (SortedNatTree.contains FastRows.row1237) := by
    change Data.row1237.contains = (SortedNatTree.contains FastRows.row1237)
    exact FastRows.row1237_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child11 : search edge 3 [37, 1237] root9Candidates11 1274 winner = winner := by decide
theorem root9Step11 : scan edge (search edge 3) 4 [37] 37 root9Tail11 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail12 winner := by
  rw [root9Tail11, scan, if_neg (by decide), root9Candidates11_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail12
    (if 3 ≤ root9Candidates11.length then search edge 3 [37, 1237] root9Candidates11 1274 winner else winner) = _
  rw [if_pos (by decide), root9Child11]
def root9Candidates12 : List Nat := [8101, 12757]
set_option maxRecDepth 8192 in
theorem root9Candidates12_checked : candidates edge 4 37 1303 root9Tail13 winner.bound = root9Candidates12 := by
  unfold candidates
  have row : edge 1303 = (SortedNatTree.contains FastRows.row1303) := by
    change Data.row1303.contains = (SortedNatTree.contains FastRows.row1303)
    exact FastRows.row1303_checked.symm
  rw [row]
  decide
theorem root9Step12 : scan edge (search edge 3) 4 [37] 37 root9Tail12 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail13 winner := by
  rw [root9Tail12, scan, if_neg (by decide), root9Candidates12_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail13
    (if 3 ≤ root9Candidates12.length then search edge 3 [37, 1303] root9Candidates12 1340 winner else winner) = _
  rw [if_neg (by decide)]
def root9Candidates13 : List Nat := [2803, 7537, 14779, 24631]
set_option maxRecDepth 8192 in
theorem root9Candidates13_checked : candidates edge 4 37 1321 root9Tail14 winner.bound = root9Candidates13 := by
  unfold candidates
  have row : edge 1321 = (SortedNatTree.contains FastRows.row1321) := by
    change Data.row1321.contains = (SortedNatTree.contains FastRows.row1321)
    exact FastRows.row1321_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child13 : search edge 3 [37, 1321] root9Candidates13 1358 winner = winner := by decide
theorem root9Step13 : scan edge (search edge 3) 4 [37] 37 root9Tail13 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail14 winner := by
  rw [root9Tail13, scan, if_neg (by decide), root9Candidates13_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail14
    (if 3 ≤ root9Candidates13.length then search edge 3 [37, 1321] root9Candidates13 1358 winner else winner) = _
  rw [if_pos (by decide), root9Child13]
def root9Candidates14 : List Nat := [1549, 2803, 4093, 23581]
set_option maxRecDepth 8192 in
theorem root9Candidates14_checked : candidates edge 4 37 1447 root9Tail15 winner.bound = root9Candidates14 := by
  unfold candidates
  have row : edge 1447 = (SortedNatTree.contains FastRows.row1447) := by
    change Data.row1447.contains = (SortedNatTree.contains FastRows.row1447)
    exact FastRows.row1447_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child14 : search edge 3 [37, 1447] root9Candidates14 1484 winner = winner := by decide
theorem root9Step14 : scan edge (search edge 3) 4 [37] 37 root9Tail14 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail15 winner := by
  rw [root9Tail14, scan, if_neg (by decide), root9Candidates14_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail15
    (if 3 ≤ root9Candidates14.length then search edge 3 [37, 1447] root9Candidates14 1484 winner else winner) = _
  rw [if_pos (by decide), root9Child14]
def root9Candidates15 : List Nat := [3613, 4159, 6133, 8779]
set_option maxRecDepth 8192 in
theorem root9Candidates15_checked : candidates edge 4 37 1471 root9Tail16 winner.bound = root9Candidates15 := by
  unfold candidates
  have row : edge 1471 = (SortedNatTree.contains FastRows.row1471) := by
    change Data.row1471.contains = (SortedNatTree.contains FastRows.row1471)
    exact FastRows.row1471_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child15 : search edge 3 [37, 1471] root9Candidates15 1508 winner = winner := by decide
theorem root9Step15 : scan edge (search edge 3) 4 [37] 37 root9Tail15 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail16 winner := by
  rw [root9Tail15, scan, if_neg (by decide), root9Candidates15_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail16
    (if 3 ≤ root9Candidates15.length then search edge 3 [37, 1471] root9Candidates15 1508 winner else winner) = _
  rw [if_pos (by decide), root9Child15]
def root9Candidates16 : List Nat := [2707, 3463, 4789, 7393, 9007, 15031]
set_option maxRecDepth 8192 in
theorem root9Candidates16_checked : candidates edge 4 37 1549 root9Tail17 winner.bound = root9Candidates16 := by
  unfold candidates
  have row : edge 1549 = (SortedNatTree.contains FastRows.row1549) := by
    change Data.row1549.contains = (SortedNatTree.contains FastRows.row1549)
    exact FastRows.row1549_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child16 : search edge 3 [37, 1549] root9Candidates16 1586 winner = winner := by decide
theorem root9Step16 : scan edge (search edge 3) 4 [37] 37 root9Tail16 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail17 winner := by
  rw [root9Tail16, scan, if_neg (by decide), root9Candidates16_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail17
    (if 3 ≤ root9Candidates16.length then search edge 3 [37, 1549] root9Candidates16 1586 winner else winner) = _
  rw [if_pos (by decide), root9Child16]
def root9Candidates17 : List Nat := [2179, 4177, 8713]
set_option maxRecDepth 8192 in
theorem root9Candidates17_checked : candidates edge 4 37 1699 root9Tail18 winner.bound = root9Candidates17 := by
  unfold candidates
  have row : edge 1699 = (SortedNatTree.contains FastRows.row1699) := by
    change Data.row1699.contains = (SortedNatTree.contains FastRows.row1699)
    exact FastRows.row1699_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child17 : search edge 3 [37, 1699] root9Candidates17 1736 winner = winner := by decide
theorem root9Step17 : scan edge (search edge 3) 4 [37] 37 root9Tail17 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail18 winner := by
  rw [root9Tail17, scan, if_neg (by decide), root9Candidates17_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail18
    (if 3 ≤ root9Candidates17.length then search edge 3 [37, 1699] root9Candidates17 1736 winner else winner) = _
  rw [if_pos (by decide), root9Child17]
def root9Candidates18 : List Nat := [2269, 7963, 8011, 21943]
set_option maxRecDepth 8192 in
theorem root9Candidates18_checked : candidates edge 4 37 1873 root9Tail19 winner.bound = root9Candidates18 := by
  unfold candidates
  have row : edge 1873 = (SortedNatTree.contains FastRows.row1873) := by
    change Data.row1873.contains = (SortedNatTree.contains FastRows.row1873)
    exact FastRows.row1873_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child18 : search edge 3 [37, 1873] root9Candidates18 1910 winner = winner := by decide
theorem root9Step18 : scan edge (search edge 3) 4 [37] 37 root9Tail18 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail19 winner := by
  rw [root9Tail18, scan, if_neg (by decide), root9Candidates18_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail19
    (if 3 ≤ root9Candidates18.length then search edge 3 [37, 1873] root9Candidates18 1910 winner else winner) = _
  rw [if_pos (by decide), root9Child18]
def root9Candidates19 : List Nat := [4909, 5227, 6823, 11863, 15031, 17011, 19087]
set_option maxRecDepth 8192 in
theorem root9Candidates19_checked : candidates edge 4 37 1951 root9Tail20 winner.bound = root9Candidates19 := by
  unfold candidates
  have row : edge 1951 = (SortedNatTree.contains FastRows.row1951) := by
    change Data.row1951.contains = (SortedNatTree.contains FastRows.row1951)
    exact FastRows.row1951_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child19 : search edge 3 [37, 1951] root9Candidates19 1988 winner = winner := by decide
theorem root9Step19 : scan edge (search edge 3) 4 [37] 37 root9Tail19 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail20 winner := by
  rw [root9Tail19, scan, if_neg (by decide), root9Candidates19_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail20
    (if 3 ≤ root9Candidates19.length then search edge 3 [37, 1951] root9Candidates19 1988 winner else winner) = _
  rw [if_pos (by decide), root9Child19]
def root9Candidates20 : List Nat := [2539, 6133, 9157, 9283, 18397, 19777]
set_option maxRecDepth 8192 in
theorem root9Candidates20_checked : candidates edge 4 37 2137 root9Tail21 winner.bound = root9Candidates20 := by
  unfold candidates
  have row : edge 2137 = (SortedNatTree.contains FastRows.row2137) := by
    change Data.row2137.contains = (SortedNatTree.contains FastRows.row2137)
    exact FastRows.row2137_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child20 : search edge 3 [37, 2137] root9Candidates20 2174 winner = winner := by decide
theorem root9Step20 : scan edge (search edge 3) 4 [37] 37 root9Tail20 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail21 winner := by
  rw [root9Tail20, scan, if_neg (by decide), root9Candidates20_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail21
    (if 3 ≤ root9Candidates20.length then search edge 3 [37, 2137] root9Candidates20 2174 winner else winner) = _
  rw [if_pos (by decide), root9Child20]
def root9Candidates21 : List Nat := [5227, 5857, 6529, 23053]
set_option maxRecDepth 8192 in
theorem root9Candidates21_checked : candidates edge 4 37 2179 root9Tail22 winner.bound = root9Candidates21 := by
  unfold candidates
  have row : edge 2179 = (SortedNatTree.contains FastRows.row2179) := by
    change Data.row2179.contains = (SortedNatTree.contains FastRows.row2179)
    exact FastRows.row2179_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child21 : search edge 3 [37, 2179] root9Candidates21 2216 winner = winner := by decide
theorem root9Step21 : scan edge (search edge 3) 4 [37] 37 root9Tail21 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail22 winner := by
  rw [root9Tail21, scan, if_neg (by decide), root9Candidates21_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail22
    (if 3 ≤ root9Candidates21.length then search edge 3 [37, 2179] root9Candidates21 2216 winner else winner) = _
  rw [if_pos (by decide), root9Child21]
def root9Candidates22 : List Nat := [3613, 4909, 5923, 8779, 11131, 15373]
set_option maxRecDepth 8192 in
theorem root9Candidates22_checked : candidates edge 4 37 2269 root9Tail23 winner.bound = root9Candidates22 := by
  unfold candidates
  have row : edge 2269 = (SortedNatTree.contains FastRows.row2269) := by
    change Data.row2269.contains = (SortedNatTree.contains FastRows.row2269)
    exact FastRows.row2269_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child22 : search edge 3 [37, 2269] root9Candidates22 2306 winner = winner := by decide
theorem root9Step22 : scan edge (search edge 3) 4 [37] 37 root9Tail22 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail23 winner := by
  rw [root9Tail22, scan, if_neg (by decide), root9Candidates22_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail23
    (if 3 ≤ root9Candidates22.length then search edge 3 [37, 2269] root9Candidates22 2306 winner else winner) = _
  rw [if_pos (by decide), root9Child22]
def root9Candidates23 : List Nat := [19777]
set_option maxRecDepth 8192 in
theorem root9Candidates23_checked : candidates edge 4 37 2371 root9Tail24 winner.bound = root9Candidates23 := by
  unfold candidates
  have row : edge 2371 = (SortedNatTree.contains FastRows.row2371) := by
    change Data.row2371.contains = (SortedNatTree.contains FastRows.row2371)
    exact FastRows.row2371_checked.symm
  rw [row]
  decide
theorem root9Step23 : scan edge (search edge 3) 4 [37] 37 root9Tail23 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail24 winner := by
  rw [root9Tail23, scan, if_neg (by decide), root9Candidates23_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail24
    (if 3 ≤ root9Candidates23.length then search edge 3 [37, 2371] root9Candidates23 2408 winner else winner) = _
  rw [if_neg (by decide)]
def root9Candidates24 : List Nat := [4159, 10243, 21943]
set_option maxRecDepth 8192 in
theorem root9Candidates24_checked : candidates edge 4 37 2377 root9Tail25 winner.bound = root9Candidates24 := by
  unfold candidates
  have row : edge 2377 = (SortedNatTree.contains FastRows.row2377) := by
    change Data.row2377.contains = (SortedNatTree.contains FastRows.row2377)
    exact FastRows.row2377_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child24 : search edge 3 [37, 2377] root9Candidates24 2414 winner = winner := by decide
theorem root9Step24 : scan edge (search edge 3) 4 [37] 37 root9Tail24 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail25 winner := by
  rw [root9Tail24, scan, if_neg (by decide), root9Candidates24_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail25
    (if 3 ≤ root9Candidates24.length then search edge 3 [37, 2377] root9Candidates24 2414 winner else winner) = _
  rw [if_pos (by decide), root9Child24]
def root9Candidates25 : List Nat := [8941, 11131, 15373, 19777]
set_option maxRecDepth 8192 in
theorem root9Candidates25_checked : candidates edge 4 37 2473 root9Tail26 winner.bound = root9Candidates25 := by
  unfold candidates
  have row : edge 2473 = (SortedNatTree.contains FastRows.row2473) := by
    change Data.row2473.contains = (SortedNatTree.contains FastRows.row2473)
    exact FastRows.row2473_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child25 : search edge 3 [37, 2473] root9Candidates25 2510 winner = winner := by decide
theorem root9Step25 : scan edge (search edge 3) 4 [37] 37 root9Tail25 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail26 winner := by
  rw [root9Tail25, scan, if_neg (by decide), root9Candidates25_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail26
    (if 3 ≤ root9Candidates25.length then search edge 3 [37, 2473] root9Candidates25 2510 winner else winner) = _
  rw [if_pos (by decide), root9Child25]
def root9Candidates26 : List Nat := [3463, 5449, 6823]
set_option maxRecDepth 8192 in
theorem root9Candidates26_checked : candidates edge 4 37 2539 root9Tail27 winner.bound = root9Candidates26 := by
  unfold candidates
  have row : edge 2539 = (SortedNatTree.contains FastRows.row2539) := by
    change Data.row2539.contains = (SortedNatTree.contains FastRows.row2539)
    exact FastRows.row2539_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child26 : search edge 3 [37, 2539] root9Candidates26 2576 winner = winner := by decide
theorem root9Step26 : scan edge (search edge 3) 4 [37] 37 root9Tail26 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail27 winner := by
  rw [root9Tail26, scan, if_neg (by decide), root9Candidates26_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail27
    (if 3 ≤ root9Candidates26.length then search edge 3 [37, 2539] root9Candidates26 2576 winner else winner) = _
  rw [if_pos (by decide), root9Child26]
def root9Candidates27 : List Nat := [3049, 7369, 9007, 9859]
set_option maxRecDepth 8192 in
theorem root9Candidates27_checked : candidates edge 4 37 2677 root9Tail28 winner.bound = root9Candidates27 := by
  unfold candidates
  have row : edge 2677 = (SortedNatTree.contains FastRows.row2677) := by
    change Data.row2677.contains = (SortedNatTree.contains FastRows.row2677)
    exact FastRows.row2677_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child27 : search edge 3 [37, 2677] root9Candidates27 2714 winner = winner := by decide
theorem root9Step27 : scan edge (search edge 3) 4 [37] 37 root9Tail27 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail28 winner := by
  rw [root9Tail27, scan, if_neg (by decide), root9Candidates27_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail28
    (if 3 ≤ root9Candidates27.length then search edge 3 [37, 2677] root9Candidates27 2714 winner else winner) = _
  rw [if_pos (by decide), root9Child27]
def root9Candidates28 : List Nat := [7477, 8089, 8761, 17041]
set_option maxRecDepth 8192 in
theorem root9Candidates28_checked : candidates edge 4 37 2689 root9Tail29 winner.bound = root9Candidates28 := by
  unfold candidates
  have row : edge 2689 = (SortedNatTree.contains FastRows.row2689) := by
    change Data.row2689.contains = (SortedNatTree.contains FastRows.row2689)
    exact FastRows.row2689_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child28 : search edge 3 [37, 2689] root9Candidates28 2726 winner = winner := by decide
theorem root9Step28 : scan edge (search edge 3) 4 [37] 37 root9Tail28 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail29 winner := by
  rw [root9Tail28, scan, if_neg (by decide), root9Candidates28_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail29
    (if 3 ≤ root9Candidates28.length then search edge 3 [37, 2689] root9Candidates28 2726 winner else winner) = _
  rw [if_pos (by decide), root9Child28]
def root9Candidates29 : List Nat := [3463, 4909, 5857]
set_option maxRecDepth 8192 in
theorem root9Candidates29_checked : candidates edge 4 37 2707 root9Tail30 winner.bound = root9Candidates29 := by
  unfold candidates
  have row : edge 2707 = (SortedNatTree.contains FastRows.row2707) := by
    change Data.row2707.contains = (SortedNatTree.contains FastRows.row2707)
    exact FastRows.row2707_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child29 : search edge 3 [37, 2707] root9Candidates29 2744 winner = winner := by decide
theorem root9Step29 : scan edge (search edge 3) 4 [37] 37 root9Tail29 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail30 winner := by
  rw [root9Tail29, scan, if_neg (by decide), root9Candidates29_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail30
    (if 3 ≤ root9Candidates29.length then search edge 3 [37, 2707] root9Candidates29 2744 winner else winner) = _
  rw [if_pos (by decide), root9Child29]
def root9Candidates30 : List Nat := [15031]
set_option maxRecDepth 8192 in
theorem root9Candidates30_checked : candidates edge 4 37 2803 root9Tail31 winner.bound = root9Candidates30 := by
  unfold candidates
  have row : edge 2803 = (SortedNatTree.contains FastRows.row2803) := by
    change Data.row2803.contains = (SortedNatTree.contains FastRows.row2803)
    exact FastRows.row2803_checked.symm
  rw [row]
  decide
theorem root9Step30 : scan edge (search edge 3) 4 [37] 37 root9Tail30 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail31 winner := by
  rw [root9Tail30, scan, if_neg (by decide), root9Candidates30_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail31
    (if 3 ≤ root9Candidates30.length then search edge 3 [37, 2803] root9Candidates30 2840 winner else winner) = _
  rw [if_neg (by decide)]
def root9Candidates31 : List Nat := [7297, 8089, 9811, 15031, 20947]
set_option maxRecDepth 8192 in
theorem root9Candidates31_checked : candidates edge 4 37 3049 root9Tail32 winner.bound = root9Candidates31 := by
  unfold candidates
  have row : edge 3049 = (SortedNatTree.contains FastRows.row3049) := by
    change Data.row3049.contains = (SortedNatTree.contains FastRows.row3049)
    exact FastRows.row3049_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root9Child31 : search edge 3 [37, 3049] root9Candidates31 3086 winner = winner := by decide
theorem root9Step31 : scan edge (search edge 3) 4 [37] 37 root9Tail31 winner =
    scan edge (search edge 3) 4 [37] 37 root9Tail32 winner := by
  rw [root9Tail31, scan, if_neg (by decide), root9Candidates31_checked]
  change scan edge (search edge 3) 4 [37] 37 root9Tail32
    (if 3 ≤ root9Candidates31.length then search edge 3 [37, 3049] root9Candidates31 3086 winner else winner) = _
  rw [if_pos (by decide), root9Child31]
end ProjectEulerSolutions.PrimePairLarge.Search
