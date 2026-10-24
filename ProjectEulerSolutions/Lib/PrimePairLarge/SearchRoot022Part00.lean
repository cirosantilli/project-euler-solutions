import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot022Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root22Candidates0 : List Nat := [10627, 20959, 21169]
set_option maxRecDepth 8192 in
theorem root22Candidates0_checked : candidates edge 4 97 157 root22Tail1 winner.bound = root22Candidates0 := by
  unfold candidates
  have row : edge 157 = (SortedNatTree.contains FastRows.row157) := by
    change Data.row157.contains = (SortedNatTree.contains FastRows.row157)
    exact FastRows.row157_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root22Child0 : search edge 3 [97, 157] root22Candidates0 254 winner = winner := by decide
theorem root22Step0 : scan edge (search edge 3) 4 [97] 97 root22Tail0 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail1 winner := by
  rw [root22Tail0, scan, if_neg (by decide), root22Candidates0_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail1
    (if 3 ≤ root22Candidates0.length then search edge 3 [97, 157] root22Candidates0 254 winner else winner) = _
  rw [if_pos (by decide), root22Child0]
def root22Candidates1 : List Nat := [883, 2833, 17551]
set_option maxRecDepth 8192 in
theorem root22Candidates1_checked : candidates edge 4 97 241 root22Tail2 winner.bound = root22Candidates1 := by
  unfold candidates
  have row : edge 241 = (SortedNatTree.contains FastRows.row241) := by
    change Data.row241.contains = (SortedNatTree.contains FastRows.row241)
    exact FastRows.row241_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root22Child1 : search edge 3 [97, 241] root22Candidates1 338 winner = winner := by decide
theorem root22Step1 : scan edge (search edge 3) 4 [97] 97 root22Tail1 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail2 winner := by
  rw [root22Tail1, scan, if_neg (by decide), root22Candidates1_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail2
    (if 3 ≤ root22Candidates1.length then search edge 3 [97, 241] root22Candidates1 338 winner else winner) = _
  rw [if_pos (by decide), root22Child1]
def root22Candidates2 : List Nat := [4957, 11971, 16633, 17443]
set_option maxRecDepth 8192 in
theorem root22Candidates2_checked : candidates edge 4 97 373 root22Tail3 winner.bound = root22Candidates2 := by
  unfold candidates
  have row : edge 373 = (SortedNatTree.contains FastRows.row373) := by
    change Data.row373.contains = (SortedNatTree.contains FastRows.row373)
    exact FastRows.row373_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root22Child2 : search edge 3 [97, 373] root22Candidates2 470 winner = winner := by decide
theorem root22Step2 : scan edge (search edge 3) 4 [97] 97 root22Tail2 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail3 winner := by
  rw [root22Tail2, scan, if_neg (by decide), root22Candidates2_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail3
    (if 3 ≤ root22Candidates2.length then search edge 3 [97, 373] root22Candidates2 470 winner else winner) = _
  rw [if_pos (by decide), root22Child2]
def root22Candidates3 : List Nat := [1063, 2221, 5581, 17401]
set_option maxRecDepth 8192 in
theorem root22Candidates3_checked : candidates edge 4 97 379 root22Tail4 winner.bound = root22Candidates3 := by
  unfold candidates
  have row : edge 379 = (SortedNatTree.contains FastRows.row379) := by
    change Data.row379.contains = (SortedNatTree.contains FastRows.row379)
    exact FastRows.row379_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root22Child3 : search edge 3 [97, 379] root22Candidates3 476 winner = winner := by decide
theorem root22Step3 : scan edge (search edge 3) 4 [97] 97 root22Tail3 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail4 winner := by
  rw [root22Tail3, scan, if_neg (by decide), root22Candidates3_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail4
    (if 3 ≤ root22Candidates3.length then search edge 3 [97, 379] root22Candidates3 476 winner else winner) = _
  rw [if_pos (by decide), root22Child3]
def root22Candidates4 : List Nat := [2017, 4507, 16633, 20959]
set_option maxRecDepth 8192 in
theorem root22Candidates4_checked : candidates edge 4 97 787 root22Tail5 winner.bound = root22Candidates4 := by
  unfold candidates
  have row : edge 787 = (SortedNatTree.contains FastRows.row787) := by
    change Data.row787.contains = (SortedNatTree.contains FastRows.row787)
    exact FastRows.row787_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root22Child4 : search edge 3 [97, 787] root22Candidates4 884 winner = winner := by decide
theorem root22Step4 : scan edge (search edge 3) 4 [97] 97 root22Tail4 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail5 winner := by
  rw [root22Tail4, scan, if_neg (by decide), root22Candidates4_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail5
    (if 3 ≤ root22Candidates4.length then search edge 3 [97, 787] root22Candidates4 884 winner else winner) = _
  rw [if_pos (by decide), root22Child4]
def root22Candidates5 : List Nat := [8209, 11467, 23017]
set_option maxRecDepth 8192 in
theorem root22Candidates5_checked : candidates edge 4 97 829 root22Tail6 winner.bound = root22Candidates5 := by
  unfold candidates
  have row : edge 829 = (SortedNatTree.contains FastRows.row829) := by
    change Data.row829.contains = (SortedNatTree.contains FastRows.row829)
    exact FastRows.row829_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root22Child5 : search edge 3 [97, 829] root22Candidates5 926 winner = winner := by decide
theorem root22Step5 : scan edge (search edge 3) 4 [97] 97 root22Tail5 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail6 winner := by
  rw [root22Tail5, scan, if_neg (by decide), root22Candidates5_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail6
    (if 3 ≤ root22Candidates5.length then search edge 3 [97, 829] root22Candidates5 926 winner else winner) = _
  rw [if_pos (by decide), root22Child5]
def root22Candidates6 : List Nat := [1783, 6571]
set_option maxRecDepth 8192 in
theorem root22Candidates6_checked : candidates edge 4 97 883 root22Tail7 winner.bound = root22Candidates6 := by
  unfold candidates
  have row : edge 883 = (SortedNatTree.contains FastRows.row883) := by
    change Data.row883.contains = (SortedNatTree.contains FastRows.row883)
    exact FastRows.row883_checked.symm
  rw [row]
  decide
theorem root22Step6 : scan edge (search edge 3) 4 [97] 97 root22Tail6 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail7 winner := by
  rw [root22Tail6, scan, if_neg (by decide), root22Candidates6_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail7
    (if 3 ≤ root22Candidates6.length then search edge 3 [97, 883] root22Candidates6 980 winner else winner) = _
  rw [if_neg (by decide)]
def root22Candidates7 : List Nat := [1279, 2017, 11467, 14437, 19603]
set_option maxRecDepth 8192 in
theorem root22Candidates7_checked : candidates edge 4 97 919 root22Tail8 winner.bound = root22Candidates7 := by
  unfold candidates
  have row : edge 919 = (SortedNatTree.contains FastRows.row919) := by
    change Data.row919.contains = (SortedNatTree.contains FastRows.row919)
    exact FastRows.row919_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root22Child7 : search edge 3 [97, 919] root22Candidates7 1016 winner = winner := by decide
theorem root22Step7 : scan edge (search edge 3) 4 [97] 97 root22Tail7 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail8 winner := by
  rw [root22Tail7, scan, if_neg (by decide), root22Candidates7_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail8
    (if 3 ≤ root22Candidates7.length then search edge 3 [97, 919] root22Candidates7 1016 winner else winner) = _
  rw [if_pos (by decide), root22Child7]
def root22Candidates8 : List Nat := [1381, 1951, 4507, 5011, 8269, 12697, 17443]
set_option maxRecDepth 8192 in
theorem root22Candidates8_checked : candidates edge 4 97 967 root22Tail9 winner.bound = root22Candidates8 := by
  unfold candidates
  have row : edge 967 = (SortedNatTree.contains FastRows.row967) := by
    change Data.row967.contains = (SortedNatTree.contains FastRows.row967)
    exact FastRows.row967_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root22Child8 : search edge 3 [97, 967] root22Candidates8 1064 winner = winner := by decide
theorem root22Step8 : scan edge (search edge 3) 4 [97] 97 root22Tail8 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail9 winner := by
  rw [root22Tail8, scan, if_neg (by decide), root22Candidates8_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail9
    (if 3 ≤ root22Candidates8.length then search edge 3 [97, 967] root22Candidates8 1064 winner else winner) = _
  rw [if_pos (by decide), root22Child8]
def root22Candidates9 : List Nat := [5281, 9787, 21871]
set_option maxRecDepth 8192 in
theorem root22Candidates9_checked : candidates edge 4 97 1021 root22Tail10 winner.bound = root22Candidates9 := by
  unfold candidates
  have row : edge 1021 = (SortedNatTree.contains FastRows.row1021) := by
    change Data.row1021.contains = (SortedNatTree.contains FastRows.row1021)
    exact FastRows.row1021_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root22Child9 : search edge 3 [97, 1021] root22Candidates9 1118 winner = winner := by decide
theorem root22Step9 : scan edge (search edge 3) 4 [97] 97 root22Tail9 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail10 winner := by
  rw [root22Tail9, scan, if_neg (by decide), root22Candidates9_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail10
    (if 3 ≤ root22Candidates9.length then search edge 3 [97, 1021] root22Candidates9 1118 winner else winner) = _
  rw [if_pos (by decide), root22Child9]
def root22Candidates10 : List Nat := [1279, 13309]
set_option maxRecDepth 8192 in
theorem root22Candidates10_checked : candidates edge 4 97 1039 root22Tail11 winner.bound = root22Candidates10 := by
  unfold candidates
  have row : edge 1039 = (SortedNatTree.contains FastRows.row1039) := by
    change Data.row1039.contains = (SortedNatTree.contains FastRows.row1039)
    exact FastRows.row1039_checked.symm
  rw [row]
  decide
theorem root22Step10 : scan edge (search edge 3) 4 [97] 97 root22Tail10 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail11 winner := by
  rw [root22Tail10, scan, if_neg (by decide), root22Candidates10_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail11
    (if 3 ≤ root22Candidates10.length then search edge 3 [97, 1039] root22Candidates10 1136 winner else winner) = _
  rw [if_neg (by decide)]
def root22Candidates11 : List Nat := [7507]
set_option maxRecDepth 8192 in
theorem root22Candidates11_checked : candidates edge 4 97 1063 root22Tail12 winner.bound = root22Candidates11 := by
  unfold candidates
  have row : edge 1063 = (SortedNatTree.contains FastRows.row1063) := by
    change Data.row1063.contains = (SortedNatTree.contains FastRows.row1063)
    exact FastRows.row1063_checked.symm
  rw [row]
  decide
theorem root22Step11 : scan edge (search edge 3) 4 [97] 97 root22Tail11 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail12 winner := by
  rw [root22Tail11, scan, if_neg (by decide), root22Candidates11_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail12
    (if 3 ≤ root22Candidates11.length then search edge 3 [97, 1063] root22Candidates11 1160 winner else winner) = _
  rw [if_neg (by decide)]
def root22Candidates12 : List Nat := [3331, 8689]
set_option maxRecDepth 8192 in
theorem root22Candidates12_checked : candidates edge 4 97 1093 root22Tail13 winner.bound = root22Candidates12 := by
  unfold candidates
  have row : edge 1093 = (SortedNatTree.contains FastRows.row1093) := by
    change Data.row1093.contains = (SortedNatTree.contains FastRows.row1093)
    exact FastRows.row1093_checked.symm
  rw [row]
  decide
theorem root22Step12 : scan edge (search edge 3) 4 [97] 97 root22Tail12 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail13 winner := by
  rw [root22Tail12, scan, if_neg (by decide), root22Candidates12_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail13
    (if 3 ≤ root22Candidates12.length then search edge 3 [97, 1093] root22Candidates12 1190 winner else winner) = _
  rw [if_neg (by decide)]
def root22Candidates13 : List Nat := [3691, 6271]
set_option maxRecDepth 8192 in
theorem root22Candidates13_checked : candidates edge 4 97 1279 root22Tail14 winner.bound = root22Candidates13 := by
  unfold candidates
  have row : edge 1279 = (SortedNatTree.contains FastRows.row1279) := by
    change Data.row1279.contains = (SortedNatTree.contains FastRows.row1279)
    exact FastRows.row1279_checked.symm
  rw [row]
  decide
theorem root22Step13 : scan edge (search edge 3) 4 [97] 97 root22Tail13 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail14 winner := by
  rw [root22Tail13, scan, if_neg (by decide), root22Candidates13_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail14
    (if 3 ≤ root22Candidates13.length then search edge 3 [97, 1279] root22Candidates13 1376 winner else winner) = _
  rw [if_neg (by decide)]
def root22Candidates14 : List Nat := [4513, 8209]
set_option maxRecDepth 8192 in
theorem root22Candidates14_checked : candidates edge 4 97 1291 root22Tail15 winner.bound = root22Candidates14 := by
  unfold candidates
  have row : edge 1291 = (SortedNatTree.contains FastRows.row1291) := by
    change Data.row1291.contains = (SortedNatTree.contains FastRows.row1291)
    exact FastRows.row1291_checked.symm
  rw [row]
  decide
theorem root22Step14 : scan edge (search edge 3) 4 [97] 97 root22Tail14 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail15 winner := by
  rw [root22Tail14, scan, if_neg (by decide), root22Candidates14_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail15
    (if 3 ≤ root22Candidates14.length then search edge 3 [97, 1291] root22Candidates14 1388 winner else winner) = _
  rw [if_neg (by decide)]
def root22Candidates15 : List Nat := [6571, 8521, 10627, 15331, 16477]
set_option maxRecDepth 8192 in
theorem root22Candidates15_checked : candidates edge 4 97 1381 root22Tail16 winner.bound = root22Candidates15 := by
  unfold candidates
  have row : edge 1381 = (SortedNatTree.contains FastRows.row1381) := by
    change Data.row1381.contains = (SortedNatTree.contains FastRows.row1381)
    exact FastRows.row1381_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root22Child15 : search edge 3 [97, 1381] root22Candidates15 1478 winner = winner := by decide
theorem root22Step15 : scan edge (search edge 3) 4 [97] 97 root22Tail15 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail16 winner := by
  rw [root22Tail15, scan, if_neg (by decide), root22Candidates15_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail16
    (if 3 ≤ root22Candidates15.length then search edge 3 [97, 1381] root22Candidates15 1478 winner else winner) = _
  rw [if_pos (by decide), root22Child15]
def root22Candidates16 : List Nat := [5011, 8209, 17443, 20551, 24151]
set_option maxRecDepth 8192 in
theorem root22Candidates16_checked : candidates edge 4 97 1783 root22Tail17 winner.bound = root22Candidates16 := by
  unfold candidates
  have row : edge 1783 = (SortedNatTree.contains FastRows.row1783) := by
    change Data.row1783.contains = (SortedNatTree.contains FastRows.row1783)
    exact FastRows.row1783_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root22Child16 : search edge 3 [97, 1783] root22Candidates16 1880 winner = winner := by decide
theorem root22Step16 : scan edge (search edge 3) 4 [97] 97 root22Tail16 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail17 winner := by
  rw [root22Tail16, scan, if_neg (by decide), root22Candidates16_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail17
    (if 3 ≤ root22Candidates16.length then search edge 3 [97, 1783] root22Candidates16 1880 winner else winner) = _
  rw [if_pos (by decide), root22Child16]
def root22Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root22Candidates17_checked : candidates edge 4 97 1951 root22Tail18 winner.bound = root22Candidates17 := by
  unfold candidates
  have row : edge 1951 = (SortedNatTree.contains FastRows.row1951) := by
    change Data.row1951.contains = (SortedNatTree.contains FastRows.row1951)
    exact FastRows.row1951_checked.symm
  rw [row]
  decide
theorem root22Step17 : scan edge (search edge 3) 4 [97] 97 root22Tail17 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail18 winner := by
  rw [root22Tail17, scan, if_neg (by decide), root22Candidates17_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail18
    (if 3 ≤ root22Candidates17.length then search edge 3 [97, 1951] root22Candidates17 2048 winner else winner) = _
  rw [if_neg (by decide)]
def root22Candidates18 : List Nat := [2221, 9787, 10711, 12697]
set_option maxRecDepth 8192 in
theorem root22Candidates18_checked : candidates edge 4 97 2017 root22Tail19 winner.bound = root22Candidates18 := by
  unfold candidates
  have row : edge 2017 = (SortedNatTree.contains FastRows.row2017) := by
    change Data.row2017.contains = (SortedNatTree.contains FastRows.row2017)
    exact FastRows.row2017_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root22Child18 : search edge 3 [97, 2017] root22Candidates18 2114 winner = winner := by decide
theorem root22Step18 : scan edge (search edge 3) 4 [97] 97 root22Tail18 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail19 winner := by
  rw [root22Tail18, scan, if_neg (by decide), root22Candidates18_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail19
    (if 3 ≤ root22Candidates18.length then search edge 3 [97, 2017] root22Candidates18 2114 winner else winner) = _
  rw [if_pos (by decide), root22Child18]
def root22Candidates19 : List Nat := [4513, 5581, 7681, 9907]
set_option maxRecDepth 8192 in
theorem root22Candidates19_checked : candidates edge 4 97 2221 root22Tail20 winner.bound = root22Candidates19 := by
  unfold candidates
  have row : edge 2221 = (SortedNatTree.contains FastRows.row2221) := by
    change Data.row2221.contains = (SortedNatTree.contains FastRows.row2221)
    exact FastRows.row2221_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root22Child19 : search edge 3 [97, 2221] root22Candidates19 2318 winner = winner := by decide
theorem root22Step19 : scan edge (search edge 3) 4 [97] 97 root22Tail19 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail20 winner := by
  rw [root22Tail19, scan, if_neg (by decide), root22Candidates19_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail20
    (if 3 ≤ root22Candidates19.length then search edge 3 [97, 2221] root22Candidates19 2318 winner else winner) = _
  rw [if_pos (by decide), root22Child19]
def root22Candidates20 : List Nat := [19867, 20959]
set_option maxRecDepth 8192 in
theorem root22Candidates20_checked : candidates edge 4 97 2833 root22Tail21 winner.bound = root22Candidates20 := by
  unfold candidates
  have row : edge 2833 = (SortedNatTree.contains FastRows.row2833) := by
    change Data.row2833.contains = (SortedNatTree.contains FastRows.row2833)
    exact FastRows.row2833_checked.symm
  rw [row]
  decide
theorem root22Step20 : scan edge (search edge 3) 4 [97] 97 root22Tail20 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail21 winner := by
  rw [root22Tail20, scan, if_neg (by decide), root22Candidates20_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail21
    (if 3 ≤ root22Candidates20.length then search edge 3 [97, 2833] root22Candidates20 2930 winner else winner) = _
  rw [if_neg (by decide)]
def root22Candidates21 : List Nat := [4261, 7927, 16747]
set_option maxRecDepth 8192 in
theorem root22Candidates21_checked : candidates edge 4 97 3331 root22Tail22 winner.bound = root22Candidates21 := by
  unfold candidates
  have row : edge 3331 = (SortedNatTree.contains FastRows.row3331) := by
    change Data.row3331.contains = (SortedNatTree.contains FastRows.row3331)
    exact FastRows.row3331_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root22Child21 : search edge 3 [97, 3331] root22Candidates21 3428 winner = winner := by decide
theorem root22Step21 : scan edge (search edge 3) 4 [97] 97 root22Tail21 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail22 winner := by
  rw [root22Tail21, scan, if_neg (by decide), root22Candidates21_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail22
    (if 3 ≤ root22Candidates21.length then search edge 3 [97, 3331] root22Candidates21 3428 winner else winner) = _
  rw [if_pos (by decide), root22Child21]
def root22Candidates22 : List Nat := [4957, 12697, 19603]
set_option maxRecDepth 8192 in
theorem root22Candidates22_checked : candidates edge 4 97 3373 root22Tail23 winner.bound = root22Candidates22 := by
  unfold candidates
  have row : edge 3373 = (SortedNatTree.contains FastRows.row3373) := by
    change Data.row3373.contains = (SortedNatTree.contains FastRows.row3373)
    exact FastRows.row3373_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root22Child22 : search edge 3 [97, 3373] root22Candidates22 3470 winner = winner := by decide
theorem root22Step22 : scan edge (search edge 3) 4 [97] 97 root22Tail22 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail23 winner := by
  rw [root22Tail22, scan, if_neg (by decide), root22Candidates22_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail23
    (if 3 ≤ root22Candidates22.length then search edge 3 [97, 3373] root22Candidates22 3470 winner else winner) = _
  rw [if_pos (by decide), root22Child22]
def root22Candidates23 : List Nat := [8521]
set_option maxRecDepth 8192 in
theorem root22Candidates23_checked : candidates edge 4 97 3691 root22Tail24 winner.bound = root22Candidates23 := by
  unfold candidates
  have row : edge 3691 = (SortedNatTree.contains FastRows.row3691) := by
    change Data.row3691.contains = (SortedNatTree.contains FastRows.row3691)
    exact FastRows.row3691_checked.symm
  rw [row]
  decide
theorem root22Step23 : scan edge (search edge 3) 4 [97] 97 root22Tail23 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail24 winner := by
  rw [root22Tail23, scan, if_neg (by decide), root22Candidates23_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail24
    (if 3 ≤ root22Candidates23.length then search edge 3 [97, 3691] root22Candidates23 3788 winner else winner) = _
  rw [if_neg (by decide)]
def root22Candidates24 : List Nat := [5011]
set_option maxRecDepth 8192 in
theorem root22Candidates24_checked : candidates edge 4 97 3727 root22Tail25 winner.bound = root22Candidates24 := by
  unfold candidates
  have row : edge 3727 = (SortedNatTree.contains FastRows.row3727) := by
    change Data.row3727.contains = (SortedNatTree.contains FastRows.row3727)
    exact FastRows.row3727_checked.symm
  rw [row]
  decide
theorem root22Step24 : scan edge (search edge 3) 4 [97] 97 root22Tail24 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail25 winner := by
  rw [root22Tail24, scan, if_neg (by decide), root22Candidates24_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail25
    (if 3 ≤ root22Candidates24.length then search edge 3 [97, 3727] root22Candidates24 3824 winner else winner) = _
  rw [if_neg (by decide)]
def root22Candidates25 : List Nat := [6991]
set_option maxRecDepth 8192 in
theorem root22Candidates25_checked : candidates edge 4 97 3853 root22Tail26 winner.bound = root22Candidates25 := by
  unfold candidates
  have row : edge 3853 = (SortedNatTree.contains FastRows.row3853) := by
    change Data.row3853.contains = (SortedNatTree.contains FastRows.row3853)
    exact FastRows.row3853_checked.symm
  rw [row]
  decide
theorem root22Step25 : scan edge (search edge 3) 4 [97] 97 root22Tail25 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail26 winner := by
  rw [root22Tail25, scan, if_neg (by decide), root22Candidates25_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail26
    (if 3 ≤ root22Candidates25.length then search edge 3 [97, 3853] root22Candidates25 3950 winner else winner) = _
  rw [if_neg (by decide)]
def root22Candidates26 : List Nat := [5011, 5281, 11971, 19867]
set_option maxRecDepth 8192 in
theorem root22Candidates26_checked : candidates edge 4 97 4261 root22Tail27 winner.bound = root22Candidates26 := by
  unfold candidates
  have row : edge 4261 = (SortedNatTree.contains FastRows.row4261) := by
    change Data.row4261.contains = (SortedNatTree.contains FastRows.row4261)
    exact FastRows.row4261_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root22Child26 : search edge 3 [97, 4261] root22Candidates26 4358 winner = winner := by decide
theorem root22Step26 : scan edge (search edge 3) 4 [97] 97 root22Tail26 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail27 winner := by
  rw [root22Tail26, scan, if_neg (by decide), root22Candidates26_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail27
    (if 3 ≤ root22Candidates26.length then search edge 3 [97, 4261] root22Candidates26 4358 winner else winner) = _
  rw [if_pos (by decide), root22Child26]
def root22Candidates27 : List Nat := [11257]
set_option maxRecDepth 8192 in
theorem root22Candidates27_checked : candidates edge 4 97 4507 root22Tail28 winner.bound = root22Candidates27 := by
  unfold candidates
  have row : edge 4507 = (SortedNatTree.contains FastRows.row4507) := by
    change Data.row4507.contains = (SortedNatTree.contains FastRows.row4507)
    exact FastRows.row4507_checked.symm
  rw [row]
  decide
theorem root22Step27 : scan edge (search edge 3) 4 [97] 97 root22Tail27 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail28 winner := by
  rw [root22Tail27, scan, if_neg (by decide), root22Candidates27_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail28
    (if 3 ≤ root22Candidates27.length then search edge 3 [97, 4507] root22Candidates27 4604 winner else winner) = _
  rw [if_neg (by decide)]
def root22Candidates28 : List Nat := [5581, 8287, 19867]
set_option maxRecDepth 8192 in
theorem root22Candidates28_checked : candidates edge 4 97 4513 root22Tail29 winner.bound = root22Candidates28 := by
  unfold candidates
  have row : edge 4513 = (SortedNatTree.contains FastRows.row4513) := by
    change Data.row4513.contains = (SortedNatTree.contains FastRows.row4513)
    exact FastRows.row4513_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root22Child28 : search edge 3 [97, 4513] root22Candidates28 4610 winner = winner := by decide
theorem root22Step28 : scan edge (search edge 3) 4 [97] 97 root22Tail28 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail29 winner := by
  rw [root22Tail28, scan, if_neg (by decide), root22Candidates28_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail29
    (if 3 ≤ root22Candidates28.length then search edge 3 [97, 4513] root22Candidates28 4610 winner else winner) = _
  rw [if_pos (by decide), root22Child28]
def root22Candidates29 : List Nat := [8209, 19867]
set_option maxRecDepth 8192 in
theorem root22Candidates29_checked : candidates edge 4 97 4657 root22Tail30 winner.bound = root22Candidates29 := by
  unfold candidates
  have row : edge 4657 = (SortedNatTree.contains FastRows.row4657) := by
    change Data.row4657.contains = (SortedNatTree.contains FastRows.row4657)
    exact FastRows.row4657_checked.symm
  rw [row]
  decide
theorem root22Step29 : scan edge (search edge 3) 4 [97] 97 root22Tail29 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail30 winner := by
  rw [root22Tail29, scan, if_neg (by decide), root22Candidates29_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail30
    (if 3 ≤ root22Candidates29.length then search edge 3 [97, 4657] root22Candidates29 4754 winner else winner) = _
  rw [if_neg (by decide)]
def root22Candidates30 : List Nat := [9787, 12697, 19141, 20551]
set_option maxRecDepth 8192 in
theorem root22Candidates30_checked : candidates edge 4 97 4957 root22Tail31 winner.bound = root22Candidates30 := by
  unfold candidates
  have row : edge 4957 = (SortedNatTree.contains FastRows.row4957) := by
    change Data.row4957.contains = (SortedNatTree.contains FastRows.row4957)
    exact FastRows.row4957_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root22Child30 : search edge 3 [97, 4957] root22Candidates30 5054 winner = winner := by decide
theorem root22Step30 : scan edge (search edge 3) 4 [97] 97 root22Tail30 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail31 winner := by
  rw [root22Tail30, scan, if_neg (by decide), root22Candidates30_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail31
    (if 3 ≤ root22Candidates30.length then search edge 3 [97, 4957] root22Candidates30 5054 winner else winner) = _
  rw [if_pos (by decide), root22Child30]
def root22Candidates31 : List Nat := [7507, 17401, 20959]
set_option maxRecDepth 8192 in
theorem root22Candidates31_checked : candidates edge 4 97 4969 root22Tail32 winner.bound = root22Candidates31 := by
  unfold candidates
  have row : edge 4969 = (SortedNatTree.contains FastRows.row4969) := by
    change Data.row4969.contains = (SortedNatTree.contains FastRows.row4969)
    exact FastRows.row4969_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root22Child31 : search edge 3 [97, 4969] root22Candidates31 5066 winner = winner := by decide
theorem root22Step31 : scan edge (search edge 3) 4 [97] 97 root22Tail31 winner =
    scan edge (search edge 3) 4 [97] 97 root22Tail32 winner := by
  rw [root22Tail31, scan, if_neg (by decide), root22Candidates31_checked]
  change scan edge (search edge 3) 4 [97] 97 root22Tail32
    (if 3 ≤ root22Candidates31.length then search edge 3 [97, 4969] root22Candidates31 5066 winner else winner) = _
  rw [if_pos (by decide), root22Child31]
end ProjectEulerSolutions.PrimePairLarge.Search
