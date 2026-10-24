import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot008Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root8Candidates0 : List Nat := [907, 1447, 21493]
set_option maxRecDepth 8192 in
theorem root8Candidates0_checked : candidates edge 4 31 139 root8Tail1 winner.bound = root8Candidates0 := by
  unfold candidates
  have row : edge 139 = (SortedNatTree.contains FastRows.row139) := by
    change Data.row139.contains = (SortedNatTree.contains FastRows.row139)
    exact FastRows.row139_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child0 : search edge 3 [31, 139] root8Candidates0 170 winner = winner := by decide
theorem root8Step0 : scan edge (search edge 3) 4 [31] 31 root8Tail0 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail1 winner := by
  rw [root8Tail0, scan, if_neg (by decide), root8Candidates0_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail1
    (if 3 ≤ root8Candidates0.length then search edge 3 [31, 139] root8Candidates0 170 winner else winner) = _
  rw [if_pos (by decide), root8Child0]
def root8Candidates1 : List Nat := [2281, 2311, 5011, 5281, 10321]
set_option maxRecDepth 8192 in
theorem root8Candidates1_checked : candidates edge 4 31 151 root8Tail2 winner.bound = root8Candidates1 := by
  unfold candidates
  have row : edge 151 = (SortedNatTree.contains FastRows.row151) := by
    change Data.row151.contains = (SortedNatTree.contains FastRows.row151)
    exact FastRows.row151_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child1 : search edge 3 [31, 151] root8Candidates1 182 winner = winner := by decide
theorem root8Step1 : scan edge (search edge 3) 4 [31] 31 root8Tail1 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail2 winner := by
  rw [root8Tail1, scan, if_neg (by decide), root8Candidates1_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail2
    (if 3 ≤ root8Candidates1.length then search edge 3 [31, 151] root8Candidates1 182 winner else winner) = _
  rw [if_pos (by decide), root8Child1]
def root8Candidates2 : List Nat := [2311, 8419, 9679, 15187, 15667, 15991, 17923]
set_option maxRecDepth 8192 in
theorem root8Candidates2_checked : candidates edge 4 31 181 root8Tail3 winner.bound = root8Candidates2 := by
  unfold candidates
  have row : edge 181 = (SortedNatTree.contains FastRows.row181) := by
    change Data.row181.contains = (SortedNatTree.contains FastRows.row181)
    exact FastRows.row181_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child2 : search edge 3 [31, 181] root8Candidates2 212 winner = winner := by decide
theorem root8Step2 : scan edge (search edge 3) 4 [31] 31 root8Tail2 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail3 winner := by
  rw [root8Tail2, scan, if_neg (by decide), root8Candidates2_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail3
    (if 3 ≤ root8Candidates2.length then search edge 3 [31, 181] root8Candidates2 212 winner else winner) = _
  rw [if_pos (by decide), root8Child2]
def root8Candidates3 : List Nat := [3307, 4549, 24469]
set_option maxRecDepth 8192 in
theorem root8Candidates3_checked : candidates edge 4 31 859 root8Tail4 winner.bound = root8Candidates3 := by
  unfold candidates
  have row : edge 859 = (SortedNatTree.contains FastRows.row859) := by
    change Data.row859.contains = (SortedNatTree.contains FastRows.row859)
    exact FastRows.row859_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child3 : search edge 3 [31, 859] root8Candidates3 890 winner = winner := by decide
theorem root8Step3 : scan edge (search edge 3) 4 [31] 31 root8Tail3 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail4 winner := by
  rw [root8Tail3, scan, if_neg (by decide), root8Candidates3_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail4
    (if 3 ≤ root8Candidates3.length then search edge 3 [31, 859] root8Candidates3 890 winner else winner) = _
  rw [if_pos (by decide), root8Child3]
def root8Candidates4 : List Nat := [2383, 4357, 5851, 7591, 13009, 20599]
set_option maxRecDepth 8192 in
theorem root8Candidates4_checked : candidates edge 4 31 907 root8Tail5 winner.bound = root8Candidates4 := by
  unfold candidates
  have row : edge 907 = (SortedNatTree.contains FastRows.row907) := by
    change Data.row907.contains = (SortedNatTree.contains FastRows.row907)
    exact FastRows.row907_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child4 : search edge 3 [31, 907] root8Candidates4 938 winner = winner := by decide
theorem root8Step4 : scan edge (search edge 3) 4 [31] 31 root8Tail4 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail5 winner := by
  rw [root8Tail4, scan, if_neg (by decide), root8Candidates4_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail5
    (if 3 ≤ root8Candidates4.length then search edge 3 [31, 907] root8Candidates4 938 winner else winner) = _
  rw [if_pos (by decide), root8Child4]
def root8Candidates5 : List Nat := [1009, 2029, 3571, 5881, 8641, 16189, 20599]
set_option maxRecDepth 8192 in
theorem root8Candidates5_checked : candidates edge 4 31 991 root8Tail6 winner.bound = root8Candidates5 := by
  unfold candidates
  have row : edge 991 = (SortedNatTree.contains FastRows.row991) := by
    change Data.row991.contains = (SortedNatTree.contains FastRows.row991)
    exact FastRows.row991_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child5 : search edge 3 [31, 991] root8Candidates5 1022 winner = winner := by decide
theorem root8Step5 : scan edge (search edge 3) 4 [31] 31 root8Tail5 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail6 winner := by
  rw [root8Tail5, scan, if_neg (by decide), root8Candidates5_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail6
    (if 3 ≤ root8Candidates5.length then search edge 3 [31, 991] root8Candidates5 1022 winner else winner) = _
  rw [if_pos (by decide), root8Child5]
def root8Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root8Candidates6_checked : candidates edge 4 31 1009 root8Tail7 winner.bound = root8Candidates6 := by
  unfold candidates
  have row : edge 1009 = (SortedNatTree.contains FastRows.row1009) := by
    change Data.row1009.contains = (SortedNatTree.contains FastRows.row1009)
    exact FastRows.row1009_checked.symm
  rw [row]
  decide
theorem root8Step6 : scan edge (search edge 3) 4 [31] 31 root8Tail6 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail7 winner := by
  rw [root8Tail6, scan, if_neg (by decide), root8Candidates6_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail7
    (if 3 ≤ root8Candidates6.length then search edge 3 [31, 1009] root8Candidates6 1040 winner else winner) = _
  rw [if_neg (by decide)]
def root8Candidates7 : List Nat := [2029, 2281, 5281, 5527, 6037, 16033]
set_option maxRecDepth 8192 in
theorem root8Candidates7_checked : candidates edge 4 31 1123 root8Tail8 winner.bound = root8Candidates7 := by
  unfold candidates
  have row : edge 1123 = (SortedNatTree.contains FastRows.row1123) := by
    change Data.row1123.contains = (SortedNatTree.contains FastRows.row1123)
    exact FastRows.row1123_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child7 : search edge 3 [31, 1123] root8Candidates7 1154 winner = winner := by decide
theorem root8Step7 : scan edge (search edge 3) 4 [31] 31 root8Tail7 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail8 winner := by
  rw [root8Tail7, scan, if_neg (by decide), root8Candidates7_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail8
    (if 3 ≤ root8Candidates7.length then search edge 3 [31, 1123] root8Candidates7 1154 winner else winner) = _
  rw [if_pos (by decide), root8Child7]
def root8Candidates8 : List Nat := [2383, 3853, 5179, 6037, 6067, 7741, 9439]
set_option maxRecDepth 8192 in
theorem root8Candidates8_checked : candidates edge 4 31 1153 root8Tail9 winner.bound = root8Candidates8 := by
  unfold candidates
  have row : edge 1153 = (SortedNatTree.contains FastRows.row1153) := by
    change Data.row1153.contains = (SortedNatTree.contains FastRows.row1153)
    exact FastRows.row1153_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child8 : search edge 3 [31, 1153] root8Candidates8 1184 winner = winner := by decide
theorem root8Step8 : scan edge (search edge 3) 4 [31] 31 root8Tail8 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail9 winner := by
  rw [root8Tail8, scan, if_neg (by decide), root8Candidates8_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail9
    (if 3 ≤ root8Candidates8.length then search edge 3 [31, 1153] root8Candidates8 1184 winner else winner) = _
  rw [if_pos (by decide), root8Child8]
def root8Candidates9 : List Nat := [3307, 6271, 6571, 9817, 13399]
set_option maxRecDepth 8192 in
theorem root8Candidates9_checked : candidates edge 4 31 1237 root8Tail10 winner.bound = root8Candidates9 := by
  unfold candidates
  have row : edge 1237 = (SortedNatTree.contains FastRows.row1237) := by
    change Data.row1237.contains = (SortedNatTree.contains FastRows.row1237)
    exact FastRows.row1237_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child9 : search edge 3 [31, 1237] root8Candidates9 1268 winner = winner := by decide
theorem root8Step9 : scan edge (search edge 3) 4 [31] 31 root8Tail9 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail10 winner := by
  rw [root8Tail9, scan, if_neg (by decide), root8Candidates9_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail10
    (if 3 ≤ root8Candidates9.length then search edge 3 [31, 1237] root8Candidates9 1268 winner else winner) = _
  rw [if_pos (by decide), root8Child9]
def root8Candidates10 : List Nat := [5179, 6037, 6271, 8641]
set_option maxRecDepth 8192 in
theorem root8Candidates10_checked : candidates edge 4 31 1279 root8Tail11 winner.bound = root8Candidates10 := by
  unfold candidates
  have row : edge 1279 = (SortedNatTree.contains FastRows.row1279) := by
    change Data.row1279.contains = (SortedNatTree.contains FastRows.row1279)
    exact FastRows.row1279_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child10 : search edge 3 [31, 1279] root8Candidates10 1310 winner = winner := by decide
theorem root8Step10 : scan edge (search edge 3) 4 [31] 31 root8Tail10 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail11 winner := by
  rw [root8Tail10, scan, if_neg (by decide), root8Candidates10_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail11
    (if 3 ≤ root8Candidates10.length then search edge 3 [31, 1279] root8Candidates10 1310 winner else winner) = _
  rw [if_pos (by decide), root8Child10]
def root8Candidates11 : List Nat := [4441, 13399, 15667, 20599]
set_option maxRecDepth 8192 in
theorem root8Candidates11_checked : candidates edge 4 31 1447 root8Tail12 winner.bound = root8Candidates11 := by
  unfold candidates
  have row : edge 1447 = (SortedNatTree.contains FastRows.row1447) := by
    change Data.row1447.contains = (SortedNatTree.contains FastRows.row1447)
    exact FastRows.row1447_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child11 : search edge 3 [31, 1447] root8Candidates11 1478 winner = winner := by decide
theorem root8Step11 : scan edge (search edge 3) 4 [31] 31 root8Tail11 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail12 winner := by
  rw [root8Tail11, scan, if_neg (by decide), root8Candidates11_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail12
    (if 3 ≤ root8Candidates11.length then search edge 3 [31, 1447] root8Candidates11 1478 winner else winner) = _
  rw [if_pos (by decide), root8Child11]
def root8Candidates12 : List Nat := [4549, 6037, 18541]
set_option maxRecDepth 8192 in
theorem root8Candidates12_checked : candidates edge 4 31 1951 root8Tail13 winner.bound = root8Candidates12 := by
  unfold candidates
  have row : edge 1951 = (SortedNatTree.contains FastRows.row1951) := by
    change Data.row1951.contains = (SortedNatTree.contains FastRows.row1951)
    exact FastRows.row1951_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child12 : search edge 3 [31, 1951] root8Candidates12 1982 winner = winner := by decide
theorem root8Step12 : scan edge (search edge 3) 4 [31] 31 root8Tail12 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail13 winner := by
  rw [root8Tail12, scan, if_neg (by decide), root8Candidates12_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail13
    (if 3 ≤ root8Candidates12.length then search edge 3 [31, 1951] root8Candidates12 1982 winner else winner) = _
  rw [if_pos (by decide), root8Child12]
def root8Candidates13 : List Nat := [2281, 5281, 5701, 9439, 9679, 11287, 15907, 18253]
set_option maxRecDepth 8192 in
theorem root8Candidates13_checked : candidates edge 4 31 2029 root8Tail14 winner.bound = root8Candidates13 := by
  unfold candidates
  have row : edge 2029 = (SortedNatTree.contains FastRows.row2029) := by
    change Data.row2029.contains = (SortedNatTree.contains FastRows.row2029)
    exact FastRows.row2029_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child13 : search edge 3 [31, 2029] root8Candidates13 2060 winner = winner := by decide
theorem root8Step13 : scan edge (search edge 3) 4 [31] 31 root8Tail13 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail14 winner := by
  rw [root8Tail13, scan, if_neg (by decide), root8Candidates13_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail14
    (if 3 ≤ root8Candidates13.length then search edge 3 [31, 2029] root8Candidates13 2060 winner else winner) = _
  rw [if_pos (by decide), root8Child13]
def root8Candidates14 : List Nat := [7411, 9049, 9679, 13633, 16693]
set_option maxRecDepth 8192 in
theorem root8Candidates14_checked : candidates edge 4 31 2089 root8Tail15 winner.bound = root8Candidates14 := by
  unfold candidates
  have row : edge 2089 = (SortedNatTree.contains FastRows.row2089) := by
    change Data.row2089.contains = (SortedNatTree.contains FastRows.row2089)
    exact FastRows.row2089_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child14 : search edge 3 [31, 2089] root8Candidates14 2120 winner = winner := by decide
theorem root8Step14 : scan edge (search edge 3) 4 [31] 31 root8Tail14 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail15 winner := by
  rw [root8Tail14, scan, if_neg (by decide), root8Candidates14_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail15
    (if 3 ≤ root8Candidates14.length then search edge 3 [31, 2089] root8Candidates14 2120 winner else winner) = _
  rw [if_pos (by decide), root8Child14]
def root8Candidates15 : List Nat := [2887]
set_option maxRecDepth 8192 in
theorem root8Candidates15_checked : candidates edge 4 31 2281 root8Tail16 winner.bound = root8Candidates15 := by
  unfold candidates
  have row : edge 2281 = (SortedNatTree.contains FastRows.row2281) := by
    change Data.row2281.contains = (SortedNatTree.contains FastRows.row2281)
    exact FastRows.row2281_checked.symm
  rw [row]
  decide
theorem root8Step15 : scan edge (search edge 3) 4 [31] 31 root8Tail15 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail16 winner := by
  rw [root8Tail15, scan, if_neg (by decide), root8Candidates15_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail16
    (if 3 ≤ root8Candidates15.length then search edge 3 [31, 2281] root8Candidates15 2312 winner else winner) = _
  rw [if_neg (by decide)]
def root8Candidates16 : List Nat := [4549, 5179, 8287]
set_option maxRecDepth 8192 in
theorem root8Candidates16_checked : candidates edge 4 31 2311 root8Tail17 winner.bound = root8Candidates16 := by
  unfold candidates
  have row : edge 2311 = (SortedNatTree.contains FastRows.row2311) := by
    change Data.row2311.contains = (SortedNatTree.contains FastRows.row2311)
    exact FastRows.row2311_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child16 : search edge 3 [31, 2311] root8Candidates16 2342 winner = winner := by decide
theorem root8Step16 : scan edge (search edge 3) 4 [31] 31 root8Tail16 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail17 winner := by
  rw [root8Tail16, scan, if_neg (by decide), root8Candidates16_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail17
    (if 3 ≤ root8Candidates16.length then search edge 3 [31, 2311] root8Candidates16 2342 winner else winner) = _
  rw [if_pos (by decide), root8Child16]
def root8Candidates17 : List Nat := [3583, 4021, 5527, 8431, 19759, 21493]
set_option maxRecDepth 8192 in
theorem root8Candidates17_checked : candidates edge 4 31 2383 root8Tail18 winner.bound = root8Candidates17 := by
  unfold candidates
  have row : edge 2383 = (SortedNatTree.contains FastRows.row2383) := by
    change Data.row2383.contains = (SortedNatTree.contains FastRows.row2383)
    exact FastRows.row2383_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child17 : search edge 3 [31, 2383] root8Candidates17 2414 winner = winner := by decide
theorem root8Step17 : scan edge (search edge 3) 4 [31] 31 root8Tail17 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail18 winner := by
  rw [root8Tail17, scan, if_neg (by decide), root8Candidates17_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail18
    (if 3 ≤ root8Candidates17.length then search edge 3 [31, 2383] root8Candidates17 2414 winner else winner) = _
  rw [if_pos (by decide), root8Child17]
def root8Candidates18 : List Nat := [4549, 5881, 11677, 15907, 20857]
set_option maxRecDepth 8192 in
theorem root8Candidates18_checked : candidates edge 4 31 2857 root8Tail19 winner.bound = root8Candidates18 := by
  unfold candidates
  have row : edge 2857 = (SortedNatTree.contains FastRows.row2857) := by
    change Data.row2857.contains = (SortedNatTree.contains FastRows.row2857)
    exact FastRows.row2857_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child18 : search edge 3 [31, 2857] root8Candidates18 2888 winner = winner := by decide
theorem root8Step18 : scan edge (search edge 3) 4 [31] 31 root8Tail18 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail19 winner := by
  rw [root8Tail18, scan, if_neg (by decide), root8Candidates18_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail19
    (if 3 ≤ root8Candidates18.length then search edge 3 [31, 2857] root8Candidates18 2888 winner else winner) = _
  rw [if_pos (by decide), root8Child18]
def root8Candidates19 : List Nat := [9439, 9511, 14431]
set_option maxRecDepth 8192 in
theorem root8Candidates19_checked : candidates edge 4 31 2887 root8Tail20 winner.bound = root8Candidates19 := by
  unfold candidates
  have row : edge 2887 = (SortedNatTree.contains FastRows.row2887) := by
    change Data.row2887.contains = (SortedNatTree.contains FastRows.row2887)
    exact FastRows.row2887_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child19 : search edge 3 [31, 2887] root8Candidates19 2918 winner = winner := by decide
theorem root8Step19 : scan edge (search edge 3) 4 [31] 31 root8Tail19 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail20 winner := by
  rw [root8Tail19, scan, if_neg (by decide), root8Candidates19_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail20
    (if 3 ≤ root8Candidates19.length then search edge 3 [31, 2887] root8Candidates19 2918 winner else winner) = _
  rw [if_pos (by decide), root8Child19]
def root8Candidates20 : List Nat := [3331, 3517, 5701, 8431, 17209]
set_option maxRecDepth 8192 in
theorem root8Candidates20_checked : candidates edge 4 31 3037 root8Tail21 winner.bound = root8Candidates20 := by
  unfold candidates
  have row : edge 3037 = (SortedNatTree.contains FastRows.row3037) := by
    change Data.row3037.contains = (SortedNatTree.contains FastRows.row3037)
    exact FastRows.row3037_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child20 : search edge 3 [31, 3037] root8Candidates20 3068 winner = winner := by decide
theorem root8Step20 : scan edge (search edge 3) 4 [31] 31 root8Tail20 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail21 winner := by
  rw [root8Tail20, scan, if_neg (by decide), root8Candidates20_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail21
    (if 3 ≤ root8Candidates20.length then search edge 3 [31, 3037] root8Candidates20 3068 winner else winner) = _
  rw [if_pos (by decide), root8Child20]
def root8Candidates21 : List Nat := [4339, 6073]
set_option maxRecDepth 8192 in
theorem root8Candidates21_checked : candidates edge 4 31 3109 root8Tail22 winner.bound = root8Candidates21 := by
  unfold candidates
  have row : edge 3109 = (SortedNatTree.contains FastRows.row3109) := by
    change Data.row3109.contains = (SortedNatTree.contains FastRows.row3109)
    exact FastRows.row3109_checked.symm
  rw [row]
  decide
theorem root8Step21 : scan edge (search edge 3) 4 [31] 31 root8Tail21 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail22 winner := by
  rw [root8Tail21, scan, if_neg (by decide), root8Candidates21_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail22
    (if 3 ≤ root8Candidates21.length then search edge 3 [31, 3109] root8Candidates21 3140 winner else winner) = _
  rw [if_neg (by decide)]
def root8Candidates22 : List Nat := [3517, 5779, 5881, 6343, 18541]
set_option maxRecDepth 8192 in
theorem root8Candidates22_checked : candidates edge 4 31 3301 root8Tail23 winner.bound = root8Candidates22 := by
  unfold candidates
  have row : edge 3301 = (SortedNatTree.contains FastRows.row3301) := by
    change Data.row3301.contains = (SortedNatTree.contains FastRows.row3301)
    exact FastRows.row3301_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child22 : search edge 3 [31, 3301] root8Candidates22 3332 winner = winner := by decide
theorem root8Step22 : scan edge (search edge 3) 4 [31] 31 root8Tail22 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail23 winner := by
  rw [root8Tail22, scan, if_neg (by decide), root8Candidates22_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail23
    (if 3 ≤ root8Candidates22.length then search edge 3 [31, 3301] root8Candidates22 3332 winner else winner) = _
  rw [if_pos (by decide), root8Child22]
def root8Candidates23 : List Nat := [3331, 3583, 7411, 10273, 10651, 14431]
set_option maxRecDepth 8192 in
theorem root8Candidates23_checked : candidates edge 4 31 3307 root8Tail24 winner.bound = root8Candidates23 := by
  unfold candidates
  have row : edge 3307 = (SortedNatTree.contains FastRows.row3307) := by
    change Data.row3307.contains = (SortedNatTree.contains FastRows.row3307)
    exact FastRows.row3307_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child23 : search edge 3 [31, 3307] root8Candidates23 3338 winner = winner := by decide
theorem root8Step23 : scan edge (search edge 3) 4 [31] 31 root8Tail23 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail24 winner := by
  rw [root8Tail23, scan, if_neg (by decide), root8Candidates23_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail24
    (if 3 ≤ root8Candidates23.length then search edge 3 [31, 3307] root8Candidates23 3338 winner else winner) = _
  rw [if_pos (by decide), root8Child23]
def root8Candidates24 : List Nat := [4261, 5701, 9679, 16699, 20857]
set_option maxRecDepth 8192 in
theorem root8Candidates24_checked : candidates edge 4 31 3331 root8Tail25 winner.bound = root8Candidates24 := by
  unfold candidates
  have row : edge 3331 = (SortedNatTree.contains FastRows.row3331) := by
    change Data.row3331.contains = (SortedNatTree.contains FastRows.row3331)
    exact FastRows.row3331_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child24 : search edge 3 [31, 3331] root8Candidates24 3362 winner = winner := by decide
theorem root8Step24 : scan edge (search edge 3) 4 [31] 31 root8Tail24 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail25 winner := by
  rw [root8Tail24, scan, if_neg (by decide), root8Candidates24_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail25
    (if 3 ≤ root8Candidates24.length then search edge 3 [31, 3331] root8Candidates24 3362 winner else winner) = _
  rw [if_pos (by decide), root8Child24]
def root8Candidates25 : List Nat := [16693, 17791]
set_option maxRecDepth 8192 in
theorem root8Candidates25_checked : candidates edge 4 31 3343 root8Tail26 winner.bound = root8Candidates25 := by
  unfold candidates
  have row : edge 3343 = (SortedNatTree.contains FastRows.row3343) := by
    change Data.row3343.contains = (SortedNatTree.contains FastRows.row3343)
    exact FastRows.row3343_checked.symm
  rw [row]
  decide
theorem root8Step25 : scan edge (search edge 3) 4 [31] 31 root8Tail25 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail26 winner := by
  rw [root8Tail25, scan, if_neg (by decide), root8Candidates25_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail26
    (if 3 ≤ root8Candidates25.length then search edge 3 [31, 3343] root8Candidates25 3374 winner else winner) = _
  rw [if_neg (by decide)]
def root8Candidates26 : List Nat := [8161, 21817]
set_option maxRecDepth 8192 in
theorem root8Candidates26_checked : candidates edge 4 31 3517 root8Tail27 winner.bound = root8Candidates26 := by
  unfold candidates
  have row : edge 3517 = (SortedNatTree.contains FastRows.row3517) := by
    change Data.row3517.contains = (SortedNatTree.contains FastRows.row3517)
    exact FastRows.row3517_checked.symm
  rw [row]
  decide
theorem root8Step26 : scan edge (search edge 3) 4 [31] 31 root8Tail26 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail27 winner := by
  rw [root8Tail26, scan, if_neg (by decide), root8Candidates26_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail27
    (if 3 ≤ root8Candidates26.length then search edge 3 [31, 3517] root8Candidates26 3548 winner else winner) = _
  rw [if_neg (by decide)]
def root8Candidates27 : List Nat := [14143, 15907]
set_option maxRecDepth 8192 in
theorem root8Candidates27_checked : candidates edge 4 31 3571 root8Tail28 winner.bound = root8Candidates27 := by
  unfold candidates
  have row : edge 3571 = (SortedNatTree.contains FastRows.row3571) := by
    change Data.row3571.contains = (SortedNatTree.contains FastRows.row3571)
    exact FastRows.row3571_checked.symm
  rw [row]
  decide
theorem root8Step27 : scan edge (search edge 3) 4 [31] 31 root8Tail27 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail28 winner := by
  rw [root8Tail27, scan, if_neg (by decide), root8Candidates27_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail28
    (if 3 ≤ root8Candidates27.length then search edge 3 [31, 3571] root8Candidates27 3602 winner else winner) = _
  rw [if_neg (by decide)]
def root8Candidates28 : List Nat := [5179, 6067, 8191]
set_option maxRecDepth 8192 in
theorem root8Candidates28_checked : candidates edge 4 31 3583 root8Tail29 winner.bound = root8Candidates28 := by
  unfold candidates
  have row : edge 3583 = (SortedNatTree.contains FastRows.row3583) := by
    change Data.row3583.contains = (SortedNatTree.contains FastRows.row3583)
    exact FastRows.row3583_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child28 : search edge 3 [31, 3583] root8Candidates28 3614 winner = winner := by decide
theorem root8Step28 : scan edge (search edge 3) 4 [31] 31 root8Tail28 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail29 winner := by
  rw [root8Tail28, scan, if_neg (by decide), root8Candidates28_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail29
    (if 3 ≤ root8Candidates28.length then search edge 3 [31, 3583] root8Candidates28 3614 winner else winner) = _
  rw [if_pos (by decide), root8Child28]
def root8Candidates29 : List Nat := [6343]
set_option maxRecDepth 8192 in
theorem root8Candidates29_checked : candidates edge 4 31 3637 root8Tail30 winner.bound = root8Candidates29 := by
  unfold candidates
  have row : edge 3637 = (SortedNatTree.contains FastRows.row3637) := by
    change Data.row3637.contains = (SortedNatTree.contains FastRows.row3637)
    exact FastRows.row3637_checked.symm
  rw [row]
  decide
theorem root8Step29 : scan edge (search edge 3) 4 [31] 31 root8Tail29 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail30 winner := by
  rw [root8Tail29, scan, if_neg (by decide), root8Candidates29_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail30
    (if 3 ≤ root8Candidates29.length then search edge 3 [31, 3637] root8Candidates29 3668 winner else winner) = _
  rw [if_neg (by decide)]
def root8Candidates30 : List Nat := [7591, 10273, 10399]
set_option maxRecDepth 8192 in
theorem root8Candidates30_checked : candidates edge 4 31 3853 root8Tail31 winner.bound = root8Candidates30 := by
  unfold candidates
  have row : edge 3853 = (SortedNatTree.contains FastRows.row3853) := by
    change Data.row3853.contains = (SortedNatTree.contains FastRows.row3853)
    exact FastRows.row3853_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child30 : search edge 3 [31, 3853] root8Candidates30 3884 winner = winner := by decide
theorem root8Step30 : scan edge (search edge 3) 4 [31] 31 root8Tail30 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail31 winner := by
  rw [root8Tail30, scan, if_neg (by decide), root8Candidates30_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail31
    (if 3 ≤ root8Candidates30.length then search edge 3 [31, 3853] root8Candidates30 3884 winner else winner) = _
  rw [if_pos (by decide), root8Child30]
def root8Candidates31 : List Nat := [4021, 9439, 18253, 19309]
set_option maxRecDepth 8192 in
theorem root8Candidates31_checked : candidates edge 4 31 3889 root8Tail32 winner.bound = root8Candidates31 := by
  unfold candidates
  have row : edge 3889 = (SortedNatTree.contains FastRows.row3889) := by
    change Data.row3889.contains = (SortedNatTree.contains FastRows.row3889)
    exact FastRows.row3889_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root8Child31 : search edge 3 [31, 3889] root8Candidates31 3920 winner = winner := by decide
theorem root8Step31 : scan edge (search edge 3) 4 [31] 31 root8Tail31 winner =
    scan edge (search edge 3) 4 [31] 31 root8Tail32 winner := by
  rw [root8Tail31, scan, if_neg (by decide), root8Candidates31_checked]
  change scan edge (search edge 3) 4 [31] 31 root8Tail32
    (if 3 ≤ root8Candidates31.length then search edge 3 [31, 3889] root8Candidates31 3920 winner else winner) = _
  rw [if_pos (by decide), root8Child31]
end ProjectEulerSolutions.PrimePairLarge.Search
