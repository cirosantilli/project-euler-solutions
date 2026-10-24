import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot004Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root4Candidates0 : List Nat := [449, 1553, 2411, 3191, 3347, 5897, 7283, 12197, 13763, 16619, 19841, 24923]
set_option maxRecDepth 8192 in
theorem root4Candidates0_checked : candidates edge 4 17 83 root4Tail1 winner.bound = root4Candidates0 := by
  unfold candidates
  have row : edge 83 = (SortedNatTree.contains FastRows.row83) := by
    change Data.row83.contains = (SortedNatTree.contains FastRows.row83)
    exact FastRows.row83_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child0 : search edge 3 [17, 83] root4Candidates0 100 winner = winner := by decide
theorem root4Step0 : scan edge (search edge 3) 4 [17] 17 root4Tail0 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail1 winner := by
  rw [root4Tail0, scan, if_neg (by decide), root4Candidates0_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail1
    (if 3 ≤ root4Candidates0.length then search edge 3 [17, 83] root4Candidates0 100 winner else winner) = _
  rw [if_pos (by decide), root4Child0]
def root4Candidates1 : List Nat := [1049, 4157, 4241, 8537, 24413]
set_option maxRecDepth 8192 in
theorem root4Candidates1_checked : candidates edge 4 17 239 root4Tail2 winner.bound = root4Candidates1 := by
  unfold candidates
  have row : edge 239 = (SortedNatTree.contains FastRows.row239) := by
    change Data.row239.contains = (SortedNatTree.contains FastRows.row239)
    exact FastRows.row239_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child1 : search edge 3 [17, 239] root4Candidates1 256 winner = winner := by decide
theorem root4Step1 : scan edge (search edge 3) 4 [17] 17 root4Tail1 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail2 winner := by
  rw [root4Tail1, scan, if_neg (by decide), root4Candidates1_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail2
    (if 3 ≤ root4Candidates1.length then search edge 3 [17, 239] root4Candidates1 256 winner else winner) = _
  rw [if_pos (by decide), root4Child1]
def root4Candidates2 : List Nat := [2027, 2243, 2741, 3023, 7283, 25463]
set_option maxRecDepth 8192 in
theorem root4Candidates2_checked : candidates edge 4 17 257 root4Tail3 winner.bound = root4Candidates2 := by
  unfold candidates
  have row : edge 257 = (SortedNatTree.contains FastRows.row257) := by
    change Data.row257.contains = (SortedNatTree.contains FastRows.row257)
    exact FastRows.row257_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child2 : search edge 3 [17, 257] root4Candidates2 274 winner = winner := by decide
theorem root4Step2 : scan edge (search edge 3) 4 [17] 17 root4Tail2 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail3 winner := by
  rw [root4Tail2, scan, if_neg (by decide), root4Candidates2_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail3
    (if 3 ≤ root4Candidates2.length then search edge 3 [17, 257] root4Candidates2 274 winner else winner) = _
  rw [if_pos (by decide), root4Child2]
def root4Candidates3 : List Nat := [3359, 6317, 9923, 13763, 20297]
set_option maxRecDepth 8192 in
theorem root4Candidates3_checked : candidates edge 4 17 383 root4Tail4 winner.bound = root4Candidates3 := by
  unfold candidates
  have row : edge 383 = (SortedNatTree.contains FastRows.row383) := by
    change Data.row383.contains = (SortedNatTree.contains FastRows.row383)
    exact FastRows.row383_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child3 : search edge 3 [17, 383] root4Candidates3 400 winner = winner := by decide
theorem root4Step3 : scan edge (search edge 3) 4 [17] 17 root4Tail3 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail4 winner := by
  rw [root4Tail3, scan, if_neg (by decide), root4Candidates3_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail4
    (if 3 ≤ root4Candidates3.length then search edge 3 [17, 383] root4Candidates3 400 winner else winner) = _
  rw [if_pos (by decide), root4Child3]
def root4Candidates4 : List Nat := [971, 3137, 3191, 4637, 4703, 11399, 11471, 18401, 23747]
set_option maxRecDepth 8192 in
theorem root4Candidates4_checked : candidates edge 4 17 389 root4Tail5 winner.bound = root4Candidates4 := by
  unfold candidates
  have row : edge 389 = (SortedNatTree.contains FastRows.row389) := by
    change Data.row389.contains = (SortedNatTree.contains FastRows.row389)
    exact FastRows.row389_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child4 : search edge 3 [17, 389] root4Candidates4 406 winner = winner := by decide
theorem root4Step4 : scan edge (search edge 3) 4 [17] 17 root4Tail4 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail5 winner := by
  rw [root4Tail4, scan, if_neg (by decide), root4Candidates4_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail5
    (if 3 ≤ root4Candidates4.length then search edge 3 [17, 389] root4Candidates4 406 winner else winner) = _
  rw [if_pos (by decide), root4Child4]
def root4Candidates5 : List Nat := [4943, 7127, 7283, 20123]
set_option maxRecDepth 8192 in
theorem root4Candidates5_checked : candidates edge 4 17 431 root4Tail6 winner.bound = root4Candidates5 := by
  unfold candidates
  have row : edge 431 = (SortedNatTree.contains FastRows.row431) := by
    change Data.row431.contains = (SortedNatTree.contains FastRows.row431)
    exact FastRows.row431_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child5 : search edge 3 [17, 431] root4Candidates5 448 winner = winner := by decide
theorem root4Step5 : scan edge (search edge 3) 4 [17] 17 root4Tail5 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail6 winner := by
  rw [root4Tail5, scan, if_neg (by decide), root4Candidates5_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail6
    (if 3 ≤ root4Candidates5.length then search edge 3 [17, 431] root4Candidates5 448 winner else winner) = _
  rw [if_pos (by decide), root4Child5]
def root4Candidates6 : List Nat := [2069, 4157, 6353, 6599, 7109, 8609, 9923]
set_option maxRecDepth 8192 in
theorem root4Candidates6_checked : candidates edge 4 17 449 root4Tail7 winner.bound = root4Candidates6 := by
  unfold candidates
  have row : edge 449 = (SortedNatTree.contains FastRows.row449) := by
    change Data.row449.contains = (SortedNatTree.contains FastRows.row449)
    exact FastRows.row449_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child6 : search edge 3 [17, 449] root4Candidates6 466 winner = winner := by decide
theorem root4Step6 : scan edge (search edge 3) 4 [17] 17 root4Tail6 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail7 winner := by
  rw [root4Tail6, scan, if_neg (by decide), root4Candidates6_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail7
    (if 3 ≤ root4Candidates6.length then search edge 3 [17, 449] root4Candidates6 466 winner else winner) = _
  rw [if_pos (by decide), root4Child6]
def root4Candidates7 : List Nat := [3917, 12197, 20549, 22013]
set_option maxRecDepth 8192 in
theorem root4Candidates7_checked : candidates edge 4 17 491 root4Tail8 winner.bound = root4Candidates7 := by
  unfold candidates
  have row : edge 491 = (SortedNatTree.contains FastRows.row491) := by
    change Data.row491.contains = (SortedNatTree.contains FastRows.row491)
    exact FastRows.row491_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child7 : search edge 3 [17, 491] root4Candidates7 508 winner = winner := by decide
theorem root4Step7 : scan edge (search edge 3) 4 [17] 17 root4Tail7 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail8 winner := by
  rw [root4Tail7, scan, if_neg (by decide), root4Candidates7_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail8
    (if 3 ≤ root4Candidates7.length then search edge 3 [17, 491] root4Candidates7 508 winner else winner) = _
  rw [if_pos (by decide), root4Child7]
def root4Candidates8 : List Nat := [2411, 10937, 15761]
set_option maxRecDepth 8192 in
theorem root4Candidates8_checked : candidates edge 4 17 881 root4Tail9 winner.bound = root4Candidates8 := by
  unfold candidates
  have row : edge 881 = (SortedNatTree.contains FastRows.row881) := by
    change Data.row881.contains = (SortedNatTree.contains FastRows.row881)
    exact FastRows.row881_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child8 : search edge 3 [17, 881] root4Candidates8 898 winner = winner := by decide
theorem root4Step8 : scan edge (search edge 3) 4 [17] 17 root4Tail8 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail9 winner := by
  rw [root4Tail8, scan, if_neg (by decide), root4Candidates8_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail9
    (if 3 ≤ root4Candidates8.length then search edge 3 [17, 881] root4Candidates8 898 winner else winner) = _
  rw [if_pos (by decide), root4Child8]
def root4Candidates9 : List Nat := [8831, 18701]
set_option maxRecDepth 8192 in
theorem root4Candidates9_checked : candidates edge 4 17 971 root4Tail10 winner.bound = root4Candidates9 := by
  unfold candidates
  have row : edge 971 = (SortedNatTree.contains FastRows.row971) := by
    change Data.row971.contains = (SortedNatTree.contains FastRows.row971)
    exact FastRows.row971_checked.symm
  rw [row]
  decide
theorem root4Step9 : scan edge (search edge 3) 4 [17] 17 root4Tail9 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail10 winner := by
  rw [root4Tail9, scan, if_neg (by decide), root4Candidates9_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail10
    (if 3 ≤ root4Candidates9.length then search edge 3 [17, 971] root4Candidates9 988 winner else winner) = _
  rw [if_neg (by decide)]
def root4Candidates10 : List Nat := [2297, 5897, 7841, 20297, 24923]
set_option maxRecDepth 8192 in
theorem root4Candidates10_checked : candidates edge 4 17 1049 root4Tail11 winner.bound = root4Candidates10 := by
  unfold candidates
  have row : edge 1049 = (SortedNatTree.contains FastRows.row1049) := by
    change Data.row1049.contains = (SortedNatTree.contains FastRows.row1049)
    exact FastRows.row1049_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child10 : search edge 3 [17, 1049] root4Candidates10 1066 winner = winner := by decide
theorem root4Step10 : scan edge (search edge 3) 4 [17] 17 root4Tail10 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail11 winner := by
  rw [root4Tail10, scan, if_neg (by decide), root4Candidates10_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail11
    (if 3 ≤ root4Candidates10.length then search edge 3 [17, 1049] root4Candidates10 1066 winner else winner) = _
  rw [if_pos (by decide), root4Child10]
def root4Candidates11 : List Nat := [7211, 7883, 8753, 10937, 12401]
set_option maxRecDepth 8192 in
theorem root4Candidates11_checked : candidates edge 4 17 1553 root4Tail12 winner.bound = root4Candidates11 := by
  unfold candidates
  have row : edge 1553 = (SortedNatTree.contains FastRows.row1553) := by
    change Data.row1553.contains = (SortedNatTree.contains FastRows.row1553)
    exact FastRows.row1553_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child11 : search edge 3 [17, 1553] root4Candidates11 1570 winner = winner := by decide
theorem root4Step11 : scan edge (search edge 3) 4 [17] 17 root4Tail11 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail12 winner := by
  rw [root4Tail11, scan, if_neg (by decide), root4Candidates11_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail12
    (if 3 ≤ root4Candidates11.length then search edge 3 [17, 1553] root4Candidates11 1570 winner else winner) = _
  rw [if_pos (by decide), root4Child11]
def root4Candidates12 : List Nat := [18401]
set_option maxRecDepth 8192 in
theorem root4Candidates12_checked : candidates edge 4 17 2027 root4Tail13 winner.bound = root4Candidates12 := by
  unfold candidates
  have row : edge 2027 = (SortedNatTree.contains FastRows.row2027) := by
    change Data.row2027.contains = (SortedNatTree.contains FastRows.row2027)
    exact FastRows.row2027_checked.symm
  rw [row]
  decide
theorem root4Step12 : scan edge (search edge 3) 4 [17] 17 root4Tail12 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail13 winner := by
  rw [root4Tail12, scan, if_neg (by decide), root4Candidates12_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail13
    (if 3 ≤ root4Candidates12.length then search edge 3 [17, 2027] root4Candidates12 2044 winner else winner) = _
  rw [if_neg (by decide)]
def root4Candidates13 : List Nat := [2297, 4637, 7283, 14423, 14759, 22481]
set_option maxRecDepth 8192 in
theorem root4Candidates13_checked : candidates edge 4 17 2069 root4Tail14 winner.bound = root4Candidates13 := by
  unfold candidates
  have row : edge 2069 = (SortedNatTree.contains FastRows.row2069) := by
    change Data.row2069.contains = (SortedNatTree.contains FastRows.row2069)
    exact FastRows.row2069_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child13 : search edge 3 [17, 2069] root4Candidates13 2086 winner = winner := by decide
theorem root4Step13 : scan edge (search edge 3) 4 [17] 17 root4Tail13 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail14 winner := by
  rw [root4Tail13, scan, if_neg (by decide), root4Candidates13_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail14
    (if 3 ≤ root4Candidates13.length then search edge 3 [17, 2069] root4Candidates13 2086 winner else winner) = _
  rw [if_pos (by decide), root4Child13]
def root4Candidates14 : List Nat := [10853, 17321, 20123]
set_option maxRecDepth 8192 in
theorem root4Candidates14_checked : candidates edge 4 17 2153 root4Tail15 winner.bound = root4Candidates14 := by
  unfold candidates
  have row : edge 2153 = (SortedNatTree.contains FastRows.row2153) := by
    change Data.row2153.contains = (SortedNatTree.contains FastRows.row2153)
    exact FastRows.row2153_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child14 : search edge 3 [17, 2153] root4Candidates14 2170 winner = winner := by decide
theorem root4Step14 : scan edge (search edge 3) 4 [17] 17 root4Tail14 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail15 winner := by
  rw [root4Tail14, scan, if_neg (by decide), root4Candidates14_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail15
    (if 3 ≤ root4Candidates14.length then search edge 3 [17, 2153] root4Candidates14 2170 winner else winner) = _
  rw [if_pos (by decide), root4Child14]
def root4Candidates15 : List Nat := [15767]
set_option maxRecDepth 8192 in
theorem root4Candidates15_checked : candidates edge 4 17 2213 root4Tail16 winner.bound = root4Candidates15 := by
  unfold candidates
  have row : edge 2213 = (SortedNatTree.contains FastRows.row2213) := by
    change Data.row2213.contains = (SortedNatTree.contains FastRows.row2213)
    exact FastRows.row2213_checked.symm
  rw [row]
  decide
theorem root4Step15 : scan edge (search edge 3) 4 [17] 17 root4Tail15 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail16 winner := by
  rw [root4Tail15, scan, if_neg (by decide), root4Candidates15_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail16
    (if 3 ≤ root4Candidates15.length then search edge 3 [17, 2213] root4Candidates15 2230 winner else winner) = _
  rw [if_neg (by decide)]
def root4Candidates16 : List Nat := [2411, 3539, 6599, 6899, 7487, 16427, 21683]
set_option maxRecDepth 8192 in
theorem root4Candidates16_checked : candidates edge 4 17 2243 root4Tail17 winner.bound = root4Candidates16 := by
  unfold candidates
  have row : edge 2243 = (SortedNatTree.contains FastRows.row2243) := by
    change Data.row2243.contains = (SortedNatTree.contains FastRows.row2243)
    exact FastRows.row2243_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child16 : search edge 3 [17, 2243] root4Candidates16 2260 winner = winner := by decide
theorem root4Step16 : scan edge (search edge 3) 4 [17] 17 root4Tail16 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail17 winner := by
  rw [root4Tail16, scan, if_neg (by decide), root4Candidates16_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail17
    (if 3 ≤ root4Candidates16.length then search edge 3 [17, 2243] root4Candidates16 2260 winner else winner) = _
  rw [if_pos (by decide), root4Child16]
def root4Candidates17 : List Nat := [7127, 20549, 20639]
set_option maxRecDepth 8192 in
theorem root4Candidates17_checked : candidates edge 4 17 2297 root4Tail18 winner.bound = root4Candidates17 := by
  unfold candidates
  have row : edge 2297 = (SortedNatTree.contains FastRows.row2297) := by
    change Data.row2297.contains = (SortedNatTree.contains FastRows.row2297)
    exact FastRows.row2297_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child17 : search edge 3 [17, 2297] root4Candidates17 2314 winner = winner := by decide
theorem root4Step17 : scan edge (search edge 3) 4 [17] 17 root4Tail17 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail18 winner := by
  rw [root4Tail17, scan, if_neg (by decide), root4Candidates17_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail18
    (if 3 ≤ root4Candidates17.length then search edge 3 [17, 2297] root4Candidates17 2314 winner else winner) = _
  rw [if_pos (by decide), root4Child17]
def root4Candidates18 : List Nat := [5519, 7841, 8609, 12197, 14423, 19853]
set_option maxRecDepth 8192 in
theorem root4Candidates18_checked : candidates edge 4 17 2351 root4Tail19 winner.bound = root4Candidates18 := by
  unfold candidates
  have row : edge 2351 = (SortedNatTree.contains FastRows.row2351) := by
    change Data.row2351.contains = (SortedNatTree.contains FastRows.row2351)
    exact FastRows.row2351_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child18 : search edge 3 [17, 2351] root4Candidates18 2368 winner = winner := by decide
theorem root4Step18 : scan edge (search edge 3) 4 [17] 17 root4Tail18 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail19 winner := by
  rw [root4Tail18, scan, if_neg (by decide), root4Candidates18_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail19
    (if 3 ≤ root4Candidates18.length then search edge 3 [17, 2351] root4Candidates18 2368 winner else winner) = _
  rw [if_pos (by decide), root4Child18]
def root4Candidates19 : List Nat := [3917, 4241, 5519, 8447, 16217, 18869]
set_option maxRecDepth 8192 in
theorem root4Candidates19_checked : candidates edge 4 17 2411 root4Tail20 winner.bound = root4Candidates19 := by
  unfold candidates
  have row : edge 2411 = (SortedNatTree.contains FastRows.row2411) := by
    change Data.row2411.contains = (SortedNatTree.contains FastRows.row2411)
    exact FastRows.row2411_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child19 : search edge 3 [17, 2411] root4Candidates19 2428 winner = winner := by decide
theorem root4Step19 : scan edge (search edge 3) 4 [17] 17 root4Tail19 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail20 winner := by
  rw [root4Tail19, scan, if_neg (by decide), root4Candidates19_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail20
    (if 3 ≤ root4Candidates19.length then search edge 3 [17, 2411] root4Candidates19 2428 winner else winner) = _
  rw [if_pos (by decide), root4Child19]
def root4Candidates20 : List Nat := [20639, 21143]
set_option maxRecDepth 8192 in
theorem root4Candidates20_checked : candidates edge 4 17 2657 root4Tail21 winner.bound = root4Candidates20 := by
  unfold candidates
  have row : edge 2657 = (SortedNatTree.contains FastRows.row2657) := by
    change Data.row2657.contains = (SortedNatTree.contains FastRows.row2657)
    exact FastRows.row2657_checked.symm
  rw [row]
  decide
theorem root4Step20 : scan edge (search edge 3) 4 [17] 17 root4Tail20 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail21 winner := by
  rw [root4Tail20, scan, if_neg (by decide), root4Candidates20_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail21
    (if 3 ≤ root4Candidates20.length then search edge 3 [17, 2657] root4Candidates20 2674 winner else winner) = _
  rw [if_neg (by decide)]
def root4Candidates21 : List Nat := [3917, 4649, 6353, 6899, 8537, 10799, 14411, 18371, 20123]
set_option maxRecDepth 8192 in
theorem root4Candidates21_checked : candidates edge 4 17 2741 root4Tail22 winner.bound = root4Candidates21 := by
  unfold candidates
  have row : edge 2741 = (SortedNatTree.contains FastRows.row2741) := by
    change Data.row2741.contains = (SortedNatTree.contains FastRows.row2741)
    exact FastRows.row2741_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child21 : search edge 3 [17, 2741] root4Candidates21 2758 winner = winner := by decide
theorem root4Step21 : scan edge (search edge 3) 4 [17] 17 root4Tail21 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail22 winner := by
  rw [root4Tail21, scan, if_neg (by decide), root4Candidates21_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail22
    (if 3 ≤ root4Candidates21.length then search edge 3 [17, 2741] root4Candidates21 2758 winner else winner) = _
  rw [if_pos (by decide), root4Child21]
def root4Candidates22 : List Nat := [4637, 4649, 8069]
set_option maxRecDepth 8192 in
theorem root4Candidates22_checked : candidates edge 4 17 3023 root4Tail23 winner.bound = root4Candidates22 := by
  unfold candidates
  have row : edge 3023 = (SortedNatTree.contains FastRows.row3023) := by
    change Data.row3023.contains = (SortedNatTree.contains FastRows.row3023)
    exact FastRows.row3023_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child22 : search edge 3 [17, 3023] root4Candidates22 3040 winner = winner := by decide
theorem root4Step22 : scan edge (search edge 3) 4 [17] 17 root4Tail22 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail23 winner := by
  rw [root4Tail22, scan, if_neg (by decide), root4Candidates22_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail23
    (if 3 ≤ root4Candidates22.length then search edge 3 [17, 3023] root4Candidates22 3040 winner else winner) = _
  rw [if_pos (by decide), root4Child22]
def root4Candidates23 : List Nat := [9923, 10937, 18371]
set_option maxRecDepth 8192 in
theorem root4Candidates23_checked : candidates edge 4 17 3137 root4Tail24 winner.bound = root4Candidates23 := by
  unfold candidates
  have row : edge 3137 = (SortedNatTree.contains FastRows.row3137) := by
    change Data.row3137.contains = (SortedNatTree.contains FastRows.row3137)
    exact FastRows.row3137_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child23 : search edge 3 [17, 3137] root4Candidates23 3154 winner = winner := by decide
theorem root4Step23 : scan edge (search edge 3) 4 [17] 17 root4Tail23 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail24 winner := by
  rw [root4Tail23, scan, if_neg (by decide), root4Candidates23_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail24
    (if 3 ≤ root4Candidates23.length then search edge 3 [17, 3137] root4Candidates23 3154 winner else winner) = _
  rw [if_pos (by decide), root4Child23]
def root4Candidates24 : List Nat := [4583, 7883, 8069, 14423, 20297]
set_option maxRecDepth 8192 in
theorem root4Candidates24_checked : candidates edge 4 17 3191 root4Tail25 winner.bound = root4Candidates24 := by
  unfold candidates
  have row : edge 3191 = (SortedNatTree.contains FastRows.row3191) := by
    change Data.row3191.contains = (SortedNatTree.contains FastRows.row3191)
    exact FastRows.row3191_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child24 : search edge 3 [17, 3191] root4Candidates24 3208 winner = winner := by decide
theorem root4Step24 : scan edge (search edge 3) 4 [17] 17 root4Tail24 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail25 winner := by
  rw [root4Tail24, scan, if_neg (by decide), root4Candidates24_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail25
    (if 3 ≤ root4Candidates24.length then search edge 3 [17, 3191] root4Candidates24 3208 winner else winner) = _
  rw [if_pos (by decide), root4Child24]
def root4Candidates25 : List Nat := [4673, 4703, 5333, 8231, 8693, 18869]
set_option maxRecDepth 8192 in
theorem root4Candidates25_checked : candidates edge 4 17 3347 root4Tail26 winner.bound = root4Candidates25 := by
  unfold candidates
  have row : edge 3347 = (SortedNatTree.contains FastRows.row3347) := by
    change Data.row3347.contains = (SortedNatTree.contains FastRows.row3347)
    exact FastRows.row3347_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child25 : search edge 3 [17, 3347] root4Candidates25 3364 winner = winner := by decide
theorem root4Step25 : scan edge (search edge 3) 4 [17] 17 root4Tail25 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail26 winner := by
  rw [root4Tail25, scan, if_neg (by decide), root4Candidates25_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail26
    (if 3 ≤ root4Candidates25.length then search edge 3 [17, 3347] root4Candidates25 3364 winner else winner) = _
  rw [if_pos (by decide), root4Child25]
def root4Candidates26 : List Nat := [6599, 8447, 8753, 20297, 21143]
set_option maxRecDepth 8192 in
theorem root4Candidates26_checked : candidates edge 4 17 3359 root4Tail27 winner.bound = root4Candidates26 := by
  unfold candidates
  have row : edge 3359 = (SortedNatTree.contains FastRows.row3359) := by
    change Data.row3359.contains = (SortedNatTree.contains FastRows.row3359)
    exact FastRows.row3359_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child26 : search edge 3 [17, 3359] root4Candidates26 3376 winner = winner := by decide
theorem root4Step26 : scan edge (search edge 3) 4 [17] 17 root4Tail26 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail27 winner := by
  rw [root4Tail26, scan, if_neg (by decide), root4Candidates26_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail27
    (if 3 ≤ root4Candidates26.length then search edge 3 [17, 3359] root4Candidates26 3376 winner else winner) = _
  rw [if_pos (by decide), root4Child26]
def root4Candidates27 : List Nat := [4649, 6521, 16889]
set_option maxRecDepth 8192 in
theorem root4Candidates27_checked : candidates edge 4 17 3539 root4Tail28 winner.bound = root4Candidates27 := by
  unfold candidates
  have row : edge 3539 = (SortedNatTree.contains FastRows.row3539) := by
    change Data.row3539.contains = (SortedNatTree.contains FastRows.row3539)
    exact FastRows.row3539_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child27 : search edge 3 [17, 3539] root4Candidates27 3556 winner = winner := by decide
theorem root4Step27 : scan edge (search edge 3) 4 [17] 17 root4Tail27 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail28 winner := by
  rw [root4Tail27, scan, if_neg (by decide), root4Candidates27_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail28
    (if 3 ≤ root4Candidates27.length then search edge 3 [17, 3539] root4Candidates27 3556 winner else winner) = _
  rw [if_pos (by decide), root4Child27]
def root4Candidates28 : List Nat := [4019, 4649, 6899, 8753, 17609, 18869]
set_option maxRecDepth 8192 in
theorem root4Candidates28_checked : candidates edge 4 17 3917 root4Tail29 winner.bound = root4Candidates28 := by
  unfold candidates
  have row : edge 3917 = (SortedNatTree.contains FastRows.row3917) := by
    change Data.row3917.contains = (SortedNatTree.contains FastRows.row3917)
    exact FastRows.row3917_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child28 : search edge 3 [17, 3917] root4Candidates28 3934 winner = winner := by decide
theorem root4Step28 : scan edge (search edge 3) 4 [17] 17 root4Tail28 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail29 winner := by
  rw [root4Tail28, scan, if_neg (by decide), root4Candidates28_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail29
    (if 3 ≤ root4Candidates28.length then search edge 3 [17, 3917] root4Candidates28 3934 winner else winner) = _
  rw [if_pos (by decide), root4Child28]
def root4Candidates29 : List Nat := [7211, 9479, 18719, 20063, 20231]
set_option maxRecDepth 8192 in
theorem root4Candidates29_checked : candidates edge 4 17 4019 root4Tail30 winner.bound = root4Candidates29 := by
  unfold candidates
  have row : edge 4019 = (SortedNatTree.contains FastRows.row4019) := by
    change Data.row4019.contains = (SortedNatTree.contains FastRows.row4019)
    exact FastRows.row4019_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root4Child29 : search edge 3 [17, 4019] root4Candidates29 4036 winner = winner := by decide
theorem root4Step29 : scan edge (search edge 3) 4 [17] 17 root4Tail29 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail30 winner := by
  rw [root4Tail29, scan, if_neg (by decide), root4Candidates29_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail30
    (if 3 ≤ root4Candidates29.length then search edge 3 [17, 4019] root4Candidates29 4036 winner else winner) = _
  rw [if_pos (by decide), root4Child29]
def root4Candidates30 : List Nat := [5897, 9743]
set_option maxRecDepth 8192 in
theorem root4Candidates30_checked : candidates edge 4 17 4079 root4Tail31 winner.bound = root4Candidates30 := by
  unfold candidates
  have row : edge 4079 = (SortedNatTree.contains FastRows.row4079) := by
    change Data.row4079.contains = (SortedNatTree.contains FastRows.row4079)
    exact FastRows.row4079_checked.symm
  rw [row]
  decide
theorem root4Step30 : scan edge (search edge 3) 4 [17] 17 root4Tail30 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail31 winner := by
  rw [root4Tail30, scan, if_neg (by decide), root4Candidates30_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail31
    (if 3 ≤ root4Candidates30.length then search edge 3 [17, 4079] root4Candidates30 4096 winner else winner) = _
  rw [if_neg (by decide)]
def root4Candidates31 : List Nat := [5333, 7127]
set_option maxRecDepth 8192 in
theorem root4Candidates31_checked : candidates edge 4 17 4157 root4Tail32 winner.bound = root4Candidates31 := by
  unfold candidates
  have row : edge 4157 = (SortedNatTree.contains FastRows.row4157) := by
    change Data.row4157.contains = (SortedNatTree.contains FastRows.row4157)
    exact FastRows.row4157_checked.symm
  rw [row]
  decide
theorem root4Step31 : scan edge (search edge 3) 4 [17] 17 root4Tail31 winner =
    scan edge (search edge 3) 4 [17] 17 root4Tail32 winner := by
  rw [root4Tail31, scan, if_neg (by decide), root4Candidates31_checked]
  change scan edge (search edge 3) 4 [17] 17 root4Tail32
    (if 3 ≤ root4Candidates31.length then search edge 3 [17, 4157] root4Candidates31 4174 winner else winner) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
