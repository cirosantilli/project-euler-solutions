import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot012Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root12Candidates0 : List Nat := [251, 3617, 9419, 10253]
set_option maxRecDepth 8192 in
theorem root12Candidates0_checked : candidates edge 4 47 149 root12Tail1 winner.bound = root12Candidates0 := by
  unfold candidates
  have row : edge 149 = (SortedNatTree.contains FastRows.row149) := by
    change Data.row149.contains = (SortedNatTree.contains FastRows.row149)
    exact FastRows.row149_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child0 : search edge 3 [47, 149] root12Candidates0 196 winner = winner := by decide
theorem root12Step0 : scan edge (search edge 3) 4 [47] 47 root12Tail0 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail1 winner := by
  rw [root12Tail0, scan, if_neg (by decide), root12Candidates0_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail1
    (if 3 ≤ root12Candidates0.length then search edge 3 [47, 149] root12Candidates0 196 winner else winner) = _
  rw [if_pos (by decide), root12Child0]
def root12Candidates1 : List Nat := [2063, 5147, 5273, 25673]
set_option maxRecDepth 8192 in
theorem root12Candidates1_checked : candidates edge 4 47 251 root12Tail2 winner.bound = root12Candidates1 := by
  unfold candidates
  have row : edge 251 = (SortedNatTree.contains FastRows.row251) := by
    change Data.row251.contains = (SortedNatTree.contains FastRows.row251)
    exact FastRows.row251_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child1 : search edge 3 [47, 251] root12Candidates1 298 winner = winner := by decide
theorem root12Step1 : scan edge (search edge 3) 4 [47] 47 root12Tail1 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail2 winner := by
  rw [root12Tail1, scan, if_neg (by decide), root12Candidates1_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail2
    (if 3 ≤ root12Candidates1.length then search edge 3 [47, 251] root12Candidates1 298 winner else winner) = _
  rw [if_pos (by decide), root12Child1]
def root12Candidates2 : List Nat := [2741, 8573, 13697, 16139, 19031, 25667, 25673]
set_option maxRecDepth 8192 in
theorem root12Candidates2_checked : candidates edge 4 47 269 root12Tail3 winner.bound = root12Candidates2 := by
  unfold candidates
  have row : edge 269 = (SortedNatTree.contains FastRows.row269) := by
    change Data.row269.contains = (SortedNatTree.contains FastRows.row269)
    exact FastRows.row269_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child2 : search edge 3 [47, 269] root12Candidates2 316 winner = winner := by decide
theorem root12Step2 : scan edge (search edge 3) 4 [47] 47 root12Tail2 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail3 winner := by
  rw [root12Tail2, scan, if_neg (by decide), root12Candidates2_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail3
    (if 3 ≤ root12Candidates2.length then search edge 3 [47, 269] root12Candidates2 316 winner else winner) = _
  rw [if_pos (by decide), root12Child2]
def root12Candidates3 : List Nat := [4751, 5147, 9371]
set_option maxRecDepth 8192 in
theorem root12Candidates3_checked : candidates edge 4 47 293 root12Tail4 winner.bound = root12Candidates3 := by
  unfold candidates
  have row : edge 293 = (SortedNatTree.contains FastRows.row293) := by
    change Data.row293.contains = (SortedNatTree.contains FastRows.row293)
    exact FastRows.row293_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child3 : search edge 3 [47, 293] root12Candidates3 340 winner = winner := by decide
theorem root12Step3 : scan edge (search edge 3) 4 [47] 47 root12Tail3 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail4 winner := by
  rw [root12Tail3, scan, if_neg (by decide), root12Candidates3_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail4
    (if 3 ≤ root12Candidates3.length then search edge 3 [47, 293] root12Candidates3 340 winner else winner) = _
  rw [if_pos (by decide), root12Child3]
def root12Candidates4 : List Nat := [5273, 5297, 8111]
set_option maxRecDepth 8192 in
theorem root12Candidates4_checked : candidates edge 4 47 419 root12Tail5 winner.bound = root12Candidates4 := by
  unfold candidates
  have row : edge 419 = (SortedNatTree.contains FastRows.row419) := by
    change Data.row419.contains = (SortedNatTree.contains FastRows.row419)
    exact FastRows.row419_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child4 : search edge 3 [47, 419] root12Candidates4 466 winner = winner := by decide
theorem root12Step4 : scan edge (search edge 3) 4 [47] 47 root12Tail4 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail5 winner := by
  rw [root12Tail4, scan, if_neg (by decide), root12Candidates4_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail5
    (if 3 ≤ root12Candidates4.length then search edge 3 [47, 419] root12Candidates4 466 winner else winner) = _
  rw [if_pos (by decide), root12Child4]
def root12Candidates5 : List Nat := [1091, 1721, 4337, 5381, 10313]
set_option maxRecDepth 8192 in
theorem root12Candidates5_checked : candidates edge 4 47 521 root12Tail6 winner.bound = root12Candidates5 := by
  unfold candidates
  have row : edge 521 = (SortedNatTree.contains FastRows.row521) := by
    change Data.row521.contains = (SortedNatTree.contains FastRows.row521)
    exact FastRows.row521_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child5 : search edge 3 [47, 521] root12Candidates5 568 winner = winner := by decide
theorem root12Step5 : scan edge (search edge 3) 4 [47] 47 root12Tail5 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail6 winner := by
  rw [root12Tail5, scan, if_neg (by decide), root12Candidates5_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail6
    (if 3 ≤ root12Candidates5.length then search edge 3 [47, 521] root12Candidates5 568 winner else winner) = _
  rw [if_pos (by decide), root12Child5]
def root12Candidates6 : List Nat := [1439, 1901, 4931, 8111, 9377, 9419]
set_option maxRecDepth 8192 in
theorem root12Candidates6_checked : candidates edge 4 47 947 root12Tail7 winner.bound = root12Candidates6 := by
  unfold candidates
  have row : edge 947 = (SortedNatTree.contains FastRows.row947) := by
    change Data.row947.contains = (SortedNatTree.contains FastRows.row947)
    exact FastRows.row947_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child6 : search edge 3 [47, 947] root12Candidates6 994 winner = winner := by decide
theorem root12Step6 : scan edge (search edge 3) 4 [47] 47 root12Tail6 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail7 winner := by
  rw [root12Tail6, scan, if_neg (by decide), root12Candidates6_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail7
    (if 3 ≤ root12Candidates6.length then search edge 3 [47, 947] root12Candidates6 994 winner else winner) = _
  rw [if_pos (by decide), root12Child6]
def root12Candidates7 : List Nat := [4931, 8111, 9833]
set_option maxRecDepth 8192 in
theorem root12Candidates7_checked : candidates edge 4 47 1091 root12Tail8 winner.bound = root12Candidates7 := by
  unfold candidates
  have row : edge 1091 = (SortedNatTree.contains FastRows.row1091) := by
    change Data.row1091.contains = (SortedNatTree.contains FastRows.row1091)
    exact FastRows.row1091_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child7 : search edge 3 [47, 1091] root12Candidates7 1138 winner = winner := by decide
theorem root12Step7 : scan edge (search edge 3) 4 [47] 47 root12Tail7 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail8 winner := by
  rw [root12Tail7, scan, if_neg (by decide), root12Candidates7_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail8
    (if 3 ≤ root12Candidates7.length then search edge 3 [47, 1091] root12Candidates7 1138 winner else winner) = _
  rw [if_pos (by decide), root12Child7]
def root12Candidates8 : List Nat := [4937, 6701, 8963]
set_option maxRecDepth 8192 in
theorem root12Candidates8_checked : candidates edge 4 47 1187 root12Tail9 winner.bound = root12Candidates8 := by
  unfold candidates
  have row : edge 1187 = (SortedNatTree.contains FastRows.row1187) := by
    change Data.row1187.contains = (SortedNatTree.contains FastRows.row1187)
    exact FastRows.row1187_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child8 : search edge 3 [47, 1187] root12Candidates8 1234 winner = winner := by decide
theorem root12Step8 : scan edge (search edge 3) 4 [47] 47 root12Tail8 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail9 winner := by
  rw [root12Tail8, scan, if_neg (by decide), root12Candidates8_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail9
    (if 3 ≤ root12Candidates8.length then search edge 3 [47, 1187] root12Candidates8 1234 winner else winner) = _
  rw [if_pos (by decide), root12Child8]
def root12Candidates9 : List Nat := [4787, 9209, 17519]
set_option maxRecDepth 8192 in
theorem root12Candidates9_checked : candidates edge 4 47 1277 root12Tail10 winner.bound = root12Candidates9 := by
  unfold candidates
  have row : edge 1277 = (SortedNatTree.contains FastRows.row1277) := by
    change Data.row1277.contains = (SortedNatTree.contains FastRows.row1277)
    exact FastRows.row1277_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child9 : search edge 3 [47, 1277] root12Candidates9 1324 winner = winner := by decide
theorem root12Step9 : scan edge (search edge 3) 4 [47] 47 root12Tail9 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail10 winner := by
  rw [root12Tail9, scan, if_neg (by decide), root12Candidates9_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail10
    (if 3 ≤ root12Candidates9.length then search edge 3 [47, 1277] root12Candidates9 1324 winner else winner) = _
  rw [if_pos (by decide), root12Child9]
def root12Candidates10 : List Nat := [1619, 2741, 3617, 5807, 6869, 9371, 9419, 10313]
set_option maxRecDepth 8192 in
theorem root12Candidates10_checked : candidates edge 4 47 1283 root12Tail11 winner.bound = root12Candidates10 := by
  unfold candidates
  have row : edge 1283 = (SortedNatTree.contains FastRows.row1283) := by
    change Data.row1283.contains = (SortedNatTree.contains FastRows.row1283)
    exact FastRows.row1283_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child10 : search edge 3 [47, 1283] root12Candidates10 1330 winner = winner := by decide
theorem root12Step10 : scan edge (search edge 3) 4 [47] 47 root12Tail10 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail11 winner := by
  rw [root12Tail10, scan, if_neg (by decide), root12Candidates10_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail11
    (if 3 ≤ root12Candidates10.length then search edge 3 [47, 1283] root12Candidates10 1330 winner else winner) = _
  rw [if_pos (by decide), root12Child10]
def root12Candidates11 : List Nat := [2393, 5669]
set_option maxRecDepth 8192 in
theorem root12Candidates11_checked : candidates edge 4 47 1301 root12Tail12 winner.bound = root12Candidates11 := by
  unfold candidates
  have row : edge 1301 = (SortedNatTree.contains FastRows.row1301) := by
    change Data.row1301.contains = (SortedNatTree.contains FastRows.row1301)
    exact FastRows.row1301_checked.symm
  rw [row]
  decide
theorem root12Step11 : scan edge (search edge 3) 4 [47] 47 root12Tail11 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail12 winner := by
  rw [root12Tail11, scan, if_neg (by decide), root12Candidates11_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail12
    (if 3 ≤ root12Candidates11.length then search edge 3 [47, 1301] root12Candidates11 1348 winner else winner) = _
  rw [if_neg (by decide)]
def root12Candidates12 : List Nat := [3833, 4337, 4931, 5273, 6659, 6869, 9041]
set_option maxRecDepth 8192 in
theorem root12Candidates12_checked : candidates edge 4 47 1439 root12Tail13 winner.bound = root12Candidates12 := by
  unfold candidates
  have row : edge 1439 = (SortedNatTree.contains FastRows.row1439) := by
    change Data.row1439.contains = (SortedNatTree.contains FastRows.row1439)
    exact FastRows.row1439_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child12 : search edge 3 [47, 1439] root12Candidates12 1486 winner = winner := by decide
theorem root12Step12 : scan edge (search edge 3) 4 [47] 47 root12Tail12 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail13 winner := by
  rw [root12Tail12, scan, if_neg (by decide), root12Candidates12_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail13
    (if 3 ≤ root12Candidates12.length then search edge 3 [47, 1439] root12Candidates12 1486 winner else winner) = _
  rw [if_pos (by decide), root12Child12]
def root12Candidates13 : List Nat := [4211, 5639, 6869, 8963, 17393]
set_option maxRecDepth 8192 in
theorem root12Candidates13_checked : candidates edge 4 47 1481 root12Tail14 winner.bound = root12Candidates13 := by
  unfold candidates
  have row : edge 1481 = (SortedNatTree.contains FastRows.row1481) := by
    change Data.row1481.contains = (SortedNatTree.contains FastRows.row1481)
    exact FastRows.row1481_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child13 : search edge 3 [47, 1481] root12Candidates13 1528 winner = winner := by decide
theorem root12Step13 : scan edge (search edge 3) 4 [47] 47 root12Tail13 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail14 winner := by
  rw [root12Tail13, scan, if_neg (by decide), root12Candidates13_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail14
    (if 3 ≤ root12Candidates13.length then search edge 3 [47, 1481] root12Candidates13 1528 winner else winner) = _
  rw [if_pos (by decide), root12Child13]
def root12Candidates14 : List Nat := [8573]
set_option maxRecDepth 8192 in
theorem root12Candidates14_checked : candidates edge 4 47 1487 root12Tail15 winner.bound = root12Candidates14 := by
  unfold candidates
  have row : edge 1487 = (SortedNatTree.contains FastRows.row1487) := by
    change Data.row1487.contains = (SortedNatTree.contains FastRows.row1487)
    exact FastRows.row1487_checked.symm
  rw [row]
  decide
theorem root12Step14 : scan edge (search edge 3) 4 [47] 47 root12Tail14 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail15 winner := by
  rw [root12Tail14, scan, if_neg (by decide), root12Candidates14_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail15
    (if 3 ≤ root12Candidates14.length then search edge 3 [47, 1487] root12Candidates14 1534 winner else winner) = _
  rw [if_neg (by decide)]
def root12Candidates15 : List Nat := [3257, 6701, 9041, 18671, 21737]
set_option maxRecDepth 8192 in
theorem root12Candidates15_checked : candidates edge 4 47 1619 root12Tail16 winner.bound = root12Candidates15 := by
  unfold candidates
  have row : edge 1619 = (SortedNatTree.contains FastRows.row1619) := by
    change Data.row1619.contains = (SortedNatTree.contains FastRows.row1619)
    exact FastRows.row1619_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child15 : search edge 3 [47, 1619] root12Candidates15 1666 winner = winner := by decide
theorem root12Step15 : scan edge (search edge 3) 4 [47] 47 root12Tail15 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail16 winner := by
  rw [root12Tail15, scan, if_neg (by decide), root12Candidates15_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail16
    (if 3 ≤ root12Candidates15.length then search edge 3 [47, 1619] root12Candidates15 1666 winner else winner) = _
  rw [if_pos (by decide), root12Child15]
def root12Candidates16 : List Nat := [8273, 8573]
set_option maxRecDepth 8192 in
theorem root12Candidates16_checked : candidates edge 4 47 1721 root12Tail17 winner.bound = root12Candidates16 := by
  unfold candidates
  have row : edge 1721 = (SortedNatTree.contains FastRows.row1721) := by
    change Data.row1721.contains = (SortedNatTree.contains FastRows.row1721)
    exact FastRows.row1721_checked.symm
  rw [row]
  decide
theorem root12Step16 : scan edge (search edge 3) 4 [47] 47 root12Tail16 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail17 winner := by
  rw [root12Tail16, scan, if_neg (by decide), root12Candidates16_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail17
    (if 3 ≤ root12Candidates16.length then search edge 3 [47, 1721] root12Candidates16 1768 winner else winner) = _
  rw [if_neg (by decide)]
def root12Candidates17 : List Nat := [10313, 19751]
set_option maxRecDepth 8192 in
theorem root12Candidates17_checked : candidates edge 4 47 1901 root12Tail18 winner.bound = root12Candidates17 := by
  unfold candidates
  have row : edge 1901 = (SortedNatTree.contains FastRows.row1901) := by
    change Data.row1901.contains = (SortedNatTree.contains FastRows.row1901)
    exact FastRows.row1901_checked.symm
  rw [row]
  decide
theorem root12Step17 : scan edge (search edge 3) 4 [47] 47 root12Tail17 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail18 winner := by
  rw [root12Tail17, scan, if_neg (by decide), root12Candidates17_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail18
    (if 3 ≤ root12Candidates17.length then search edge 3 [47, 1901] root12Candidates17 1948 winner else winner) = _
  rw [if_neg (by decide)]
def root12Candidates18 : List Nat := [2399, 5297, 6719, 9533, 14447, 18773, 19571]
set_option maxRecDepth 8192 in
theorem root12Candidates18_checked : candidates edge 4 47 1931 root12Tail19 winner.bound = root12Candidates18 := by
  unfold candidates
  have row : edge 1931 = (SortedNatTree.contains FastRows.row1931) := by
    change Data.row1931.contains = (SortedNatTree.contains FastRows.row1931)
    exact FastRows.row1931_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child18 : search edge 3 [47, 1931] root12Candidates18 1978 winner = winner := by decide
theorem root12Step18 : scan edge (search edge 3) 4 [47] 47 root12Tail18 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail19 winner := by
  rw [root12Tail18, scan, if_neg (by decide), root12Candidates18_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail19
    (if 3 ≤ root12Candidates18.length then search edge 3 [47, 1931] root12Candidates18 1978 winner else winner) = _
  rw [if_pos (by decide), root12Child18]
def root12Candidates19 : List Nat := [3761, 5483, 6701, 6911, 10463, 19763]
set_option maxRecDepth 8192 in
theorem root12Candidates19_checked : candidates edge 4 47 2027 root12Tail20 winner.bound = root12Candidates19 := by
  unfold candidates
  have row : edge 2027 = (SortedNatTree.contains FastRows.row2027) := by
    change Data.row2027.contains = (SortedNatTree.contains FastRows.row2027)
    exact FastRows.row2027_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child19 : search edge 3 [47, 2027] root12Candidates19 2074 winner = winner := by decide
theorem root12Step19 : scan edge (search edge 3) 4 [47] 47 root12Tail19 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail20 winner := by
  rw [root12Tail19, scan, if_neg (by decide), root12Candidates19_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail20
    (if 3 ≤ root12Candidates19.length then search edge 3 [47, 2027] root12Candidates19 2074 winner else winner) = _
  rw [if_pos (by decide), root12Child19]
def root12Candidates20 : List Nat := [2333, 3617, 6599, 9239, 13697, 16529, 21737]
set_option maxRecDepth 8192 in
theorem root12Candidates20_checked : candidates edge 4 47 2063 root12Tail21 winner.bound = root12Candidates20 := by
  unfold candidates
  have row : edge 2063 = (SortedNatTree.contains FastRows.row2063) := by
    change Data.row2063.contains = (SortedNatTree.contains FastRows.row2063)
    exact FastRows.row2063_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child20 : search edge 3 [47, 2063] root12Candidates20 2110 winner = winner := by decide
theorem root12Step20 : scan edge (search edge 3) 4 [47] 47 root12Tail20 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail21 winner := by
  rw [root12Tail20, scan, if_neg (by decide), root12Candidates20_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail21
    (if 3 ≤ root12Candidates20.length then search edge 3 [47, 2063] root12Candidates20 2110 winner else winner) = _
  rw [if_pos (by decide), root12Child20]
def root12Candidates21 : List Nat := [2963, 5297, 5927, 6701, 12203, 15413, 16253]
set_option maxRecDepth 8192 in
theorem root12Candidates21_checked : candidates edge 4 47 2333 root12Tail22 winner.bound = root12Candidates21 := by
  unfold candidates
  have row : edge 2333 = (SortedNatTree.contains FastRows.row2333) := by
    change Data.row2333.contains = (SortedNatTree.contains FastRows.row2333)
    exact FastRows.row2333_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child21 : search edge 3 [47, 2333] root12Candidates21 2380 winner = winner := by decide
theorem root12Step21 : scan edge (search edge 3) 4 [47] 47 root12Tail21 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail22 winner := by
  rw [root12Tail21, scan, if_neg (by decide), root12Candidates21_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail22
    (if 3 ≤ root12Candidates21.length then search edge 3 [47, 2333] root12Candidates21 2380 winner else winner) = _
  rw [if_pos (by decide), root12Child21]
def root12Candidates22 : List Nat := [4241, 8069, 9371, 14249]
set_option maxRecDepth 8192 in
theorem root12Candidates22_checked : candidates edge 4 47 2393 root12Tail23 winner.bound = root12Candidates22 := by
  unfold candidates
  have row : edge 2393 = (SortedNatTree.contains FastRows.row2393) := by
    change Data.row2393.contains = (SortedNatTree.contains FastRows.row2393)
    exact FastRows.row2393_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child22 : search edge 3 [47, 2393] root12Candidates22 2440 winner = winner := by decide
theorem root12Step22 : scan edge (search edge 3) 4 [47] 47 root12Tail22 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail23 winner := by
  rw [root12Tail22, scan, if_neg (by decide), root12Candidates22_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail23
    (if 3 ≤ root12Candidates22.length then search edge 3 [47, 2393] root12Candidates22 2440 winner else winner) = _
  rw [if_pos (by decide), root12Child22]
def root12Candidates23 : List Nat := [6701, 6737, 8087, 14447, 14639]
set_option maxRecDepth 8192 in
theorem root12Candidates23_checked : candidates edge 4 47 2399 root12Tail24 winner.bound = root12Candidates23 := by
  unfold candidates
  have row : edge 2399 = (SortedNatTree.contains FastRows.row2399) := by
    change Data.row2399.contains = (SortedNatTree.contains FastRows.row2399)
    exact FastRows.row2399_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child23 : search edge 3 [47, 2399] root12Candidates23 2446 winner = winner := by decide
theorem root12Step23 : scan edge (search edge 3) 4 [47] 47 root12Tail23 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail24 winner := by
  rw [root12Tail23, scan, if_neg (by decide), root12Candidates23_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail24
    (if 3 ≤ root12Candidates23.length then search edge 3 [47, 2399] root12Candidates23 2446 winner else winner) = _
  rw [if_pos (by decide), root12Child23]
def root12Candidates24 : List Nat := [6599, 6737, 8069, 8573, 11087, 14447]
set_option maxRecDepth 8192 in
theorem root12Candidates24_checked : candidates edge 4 47 2687 root12Tail25 winner.bound = root12Candidates24 := by
  unfold candidates
  have row : edge 2687 = (SortedNatTree.contains FastRows.row2687) := by
    change Data.row2687.contains = (SortedNatTree.contains FastRows.row2687)
    exact FastRows.row2687_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child24 : search edge 3 [47, 2687] root12Candidates24 2734 winner = winner := by decide
theorem root12Step24 : scan edge (search edge 3) 4 [47] 47 root12Tail24 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail25 winner := by
  rw [root12Tail24, scan, if_neg (by decide), root12Candidates24_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail25
    (if 3 ≤ root12Candidates24.length then search edge 3 [47, 2687] root12Candidates24 2734 winner else winner) = _
  rw [if_pos (by decide), root12Child24]
def root12Candidates25 : List Nat := [2963, 6737, 6911]
set_option maxRecDepth 8192 in
theorem root12Candidates25_checked : candidates edge 4 47 2741 root12Tail26 winner.bound = root12Candidates25 := by
  unfold candidates
  have row : edge 2741 = (SortedNatTree.contains FastRows.row2741) := by
    change Data.row2741.contains = (SortedNatTree.contains FastRows.row2741)
    exact FastRows.row2741_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child25 : search edge 3 [47, 2741] root12Candidates25 2788 winner = winner := by decide
theorem root12Step25 : scan edge (search edge 3) 4 [47] 47 root12Tail25 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail26 winner := by
  rw [root12Tail25, scan, if_neg (by decide), root12Candidates25_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail26
    (if 3 ≤ root12Candidates25.length then search edge 3 [47, 2741] root12Candidates25 2788 winner else winner) = _
  rw [if_pos (by decide), root12Child25]
def root12Candidates26 : List Nat := [5381, 8111, 9371, 17393, 21737]
set_option maxRecDepth 8192 in
theorem root12Candidates26_checked : candidates edge 4 47 2963 root12Tail27 winner.bound = root12Candidates26 := by
  unfold candidates
  have row : edge 2963 = (SortedNatTree.contains FastRows.row2963) := by
    change Data.row2963.contains = (SortedNatTree.contains FastRows.row2963)
    exact FastRows.row2963_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child26 : search edge 3 [47, 2963] root12Candidates26 3010 winner = winner := by decide
theorem root12Step26 : scan edge (search edge 3) 4 [47] 47 root12Tail26 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail27 winner := by
  rw [root12Tail26, scan, if_neg (by decide), root12Candidates26_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail27
    (if 3 ≤ root12Candidates26.length then search edge 3 [47, 2963] root12Candidates26 3010 winner else winner) = _
  rw [if_pos (by decide), root12Child26]
def root12Candidates27 : List Nat := [8069, 8111, 15413]
set_option maxRecDepth 8192 in
theorem root12Candidates27_checked : candidates edge 4 47 3191 root12Tail28 winner.bound = root12Candidates27 := by
  unfold candidates
  have row : edge 3191 = (SortedNatTree.contains FastRows.row3191) := by
    change Data.row3191.contains = (SortedNatTree.contains FastRows.row3191)
    exact FastRows.row3191_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child27 : search edge 3 [47, 3191] root12Candidates27 3238 winner = winner := by decide
theorem root12Step27 : scan edge (search edge 3) 4 [47] 47 root12Tail27 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail28 winner := by
  rw [root12Tail27, scan, if_neg (by decide), root12Candidates27_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail28
    (if 3 ≤ root12Candidates27.length then search edge 3 [47, 3191] root12Candidates27 3238 winner else winner) = _
  rw [if_pos (by decide), root12Child27]
def root12Candidates28 : List Nat := [4937, 17519]
set_option maxRecDepth 8192 in
theorem root12Candidates28_checked : candidates edge 4 47 3257 root12Tail29 winner.bound = root12Candidates28 := by
  unfold candidates
  have row : edge 3257 = (SortedNatTree.contains FastRows.row3257) := by
    change Data.row3257.contains = (SortedNatTree.contains FastRows.row3257)
    exact FastRows.row3257_checked.symm
  rw [row]
  decide
theorem root12Step28 : scan edge (search edge 3) 4 [47] 47 root12Tail28 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail29 winner := by
  rw [root12Tail28, scan, if_neg (by decide), root12Candidates28_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail29
    (if 3 ≤ root12Candidates28.length then search edge 3 [47, 3257] root12Candidates28 3304 winner else winner) = _
  rw [if_neg (by decide)]
def root12Candidates29 : List Nat := [3923, 5669, 6089, 10313]
set_option maxRecDepth 8192 in
theorem root12Candidates29_checked : candidates edge 4 47 3617 root12Tail30 winner.bound = root12Candidates29 := by
  unfold candidates
  have row : edge 3617 = (SortedNatTree.contains FastRows.row3617) := by
    change Data.row3617.contains = (SortedNatTree.contains FastRows.row3617)
    exact FastRows.row3617_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child29 : search edge 3 [47, 3617] root12Candidates29 3664 winner = winner := by decide
theorem root12Step29 : scan edge (search edge 3) 4 [47] 47 root12Tail29 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail30 winner := by
  rw [root12Tail29, scan, if_neg (by decide), root12Candidates29_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail30
    (if 3 ≤ root12Candidates29.length then search edge 3 [47, 3617] root12Candidates29 3664 winner else winner) = _
  rw [if_pos (by decide), root12Child29]
def root12Candidates30 : List Nat := [5807, 8573, 9209]
set_option maxRecDepth 8192 in
theorem root12Candidates30_checked : candidates edge 4 47 3761 root12Tail31 winner.bound = root12Candidates30 := by
  unfold candidates
  have row : edge 3761 = (SortedNatTree.contains FastRows.row3761) := by
    change Data.row3761.contains = (SortedNatTree.contains FastRows.row3761)
    exact FastRows.row3761_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root12Child30 : search edge 3 [47, 3761] root12Candidates30 3808 winner = winner := by decide
theorem root12Step30 : scan edge (search edge 3) 4 [47] 47 root12Tail30 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail31 winner := by
  rw [root12Tail30, scan, if_neg (by decide), root12Candidates30_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail31
    (if 3 ≤ root12Candidates30.length then search edge 3 [47, 3761] root12Candidates30 3808 winner else winner) = _
  rw [if_pos (by decide), root12Child30]
def root12Candidates31 : List Nat := [6911, 13001]
set_option maxRecDepth 8192 in
theorem root12Candidates31_checked : candidates edge 4 47 3833 root12Tail32 winner.bound = root12Candidates31 := by
  unfold candidates
  have row : edge 3833 = (SortedNatTree.contains FastRows.row3833) := by
    change Data.row3833.contains = (SortedNatTree.contains FastRows.row3833)
    exact FastRows.row3833_checked.symm
  rw [row]
  decide
theorem root12Step31 : scan edge (search edge 3) 4 [47] 47 root12Tail31 winner =
    scan edge (search edge 3) 4 [47] 47 root12Tail32 winner := by
  rw [root12Tail31, scan, if_neg (by decide), root12Candidates31_checked]
  change scan edge (search edge 3) 4 [47] 47 root12Tail32
    (if 3 ≤ root12Candidates31.length then search edge 3 [47, 3833] root12Candidates31 3880 winner else winner) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
