import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot014Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root14Candidates0 : List Nat := [2087, 5981, 7253, 7673, 9857, 10559, 10973, 16943, 18371, 19559]
set_option maxRecDepth 8192 in
theorem root14Candidates0_checked : candidates edge 4 59 167 root14Tail1 winner.bound = root14Candidates0 := by
  unfold candidates
  have row : edge 167 = (SortedNatTree.contains FastRows.row167) := by
    change Data.row167.contains = (SortedNatTree.contains FastRows.row167)
    exact FastRows.row167_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root14Child0 : search edge 3 [59, 167] root14Candidates0 226 winner = winner := by decide
theorem root14Step0 : scan edge (search edge 3) 4 [59] 59 root14Tail0 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail1 winner := by
  rw [root14Tail0, scan, if_neg (by decide), root14Candidates0_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail1
    (if 3 ≤ root14Candidates0.length then search edge 3 [59, 167] root14Candidates0 226 winner else winner) = _
  rw [if_pos (by decide), root14Child0]
def root14Candidates1 : List Nat := [971, 2099, 9281, 9491, 16349, 16649, 19289]
set_option maxRecDepth 8192 in
theorem root14Candidates1_checked : candidates edge 4 59 197 root14Tail2 winner.bound = root14Candidates1 := by
  unfold candidates
  have row : edge 197 = (SortedNatTree.contains FastRows.row197) := by
    change Data.row197.contains = (SortedNatTree.contains FastRows.row197)
    exact FastRows.row197_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root14Child1 : search edge 3 [59, 197] root14Candidates1 256 winner = winner := by decide
theorem root14Step1 : scan edge (search edge 3) 4 [59] 59 root14Tail1 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail2 winner := by
  rw [root14Tail1, scan, if_neg (by decide), root14Candidates1_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail2
    (if 3 ≤ root14Candidates1.length then search edge 3 [59, 197] root14Candidates1 256 winner else winner) = _
  rw [if_pos (by decide), root14Child1]
def root14Candidates2 : List Nat := [929, 2087, 7253, 16349]
set_option maxRecDepth 8192 in
theorem root14Candidates2_checked : candidates edge 4 59 419 root14Tail3 winner.bound = root14Candidates2 := by
  unfold candidates
  have row : edge 419 = (SortedNatTree.contains FastRows.row419) := by
    change Data.row419.contains = (SortedNatTree.contains FastRows.row419)
    exact FastRows.row419_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root14Child2 : search edge 3 [59, 419] root14Candidates2 478 winner = winner := by decide
theorem root14Step2 : scan edge (search edge 3) 4 [59] 59 root14Tail2 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail3 winner := by
  rw [root14Tail2, scan, if_neg (by decide), root14Candidates2_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail3
    (if 3 ≤ root14Candidates2.length then search edge 3 [59, 419] root14Candidates2 478 winner else winner) = _
  rw [if_pos (by decide), root14Child2]
def root14Candidates3 : List Nat := [15551, 19559]
set_option maxRecDepth 8192 in
theorem root14Candidates3_checked : candidates edge 4 59 929 root14Tail4 winner.bound = root14Candidates3 := by
  unfold candidates
  have row : edge 929 = (SortedNatTree.contains FastRows.row929) := by
    change Data.row929.contains = (SortedNatTree.contains FastRows.row929)
    exact FastRows.row929_checked.symm
  rw [row]
  decide
theorem root14Step3 : scan edge (search edge 3) 4 [59] 59 root14Tail3 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail4 winner := by
  rw [root14Tail3, scan, if_neg (by decide), root14Candidates3_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail4
    (if 3 ≤ root14Candidates3.length then search edge 3 [59, 929] root14Candidates3 988 winner else winner) = _
  rw [if_neg (by decide)]
def root14Candidates4 : List Nat := [2897, 6569, 8123, 24851]
set_option maxRecDepth 8192 in
theorem root14Candidates4_checked : candidates edge 4 59 971 root14Tail5 winner.bound = root14Candidates4 := by
  unfold candidates
  have row : edge 971 = (SortedNatTree.contains FastRows.row971) := by
    change Data.row971.contains = (SortedNatTree.contains FastRows.row971)
    exact FastRows.row971_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root14Child4 : search edge 3 [59, 971] root14Candidates4 1030 winner = winner := by decide
theorem root14Step4 : scan edge (search edge 3) 4 [59] 59 root14Tail4 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail5 winner := by
  rw [root14Tail4, scan, if_neg (by decide), root14Candidates4_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail5
    (if 3 ≤ root14Candidates4.length then search edge 3 [59, 971] root14Candidates4 1030 winner else winner) = _
  rw [if_pos (by decide), root14Child4]
def root14Candidates5 : List Nat := [1319, 4931, 8933, 9479]
set_option maxRecDepth 8192 in
theorem root14Candidates5_checked : candidates edge 4 59 1091 root14Tail6 winner.bound = root14Candidates5 := by
  unfold candidates
  have row : edge 1091 = (SortedNatTree.contains FastRows.row1091) := by
    change Data.row1091.contains = (SortedNatTree.contains FastRows.row1091)
    exact FastRows.row1091_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root14Child5 : search edge 3 [59, 1091] root14Candidates5 1150 winner = winner := by decide
theorem root14Step5 : scan edge (search edge 3) 4 [59] 59 root14Tail5 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail6 winner := by
  rw [root14Tail5, scan, if_neg (by decide), root14Candidates5_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail6
    (if 3 ≤ root14Candidates5.length then search edge 3 [59, 1091] root14Candidates5 1150 winner else winner) = _
  rw [if_pos (by decide), root14Child5]
def root14Candidates6 : List Nat := [6899, 7349, 9479, 11057, 13931, 19139, 20483]
set_option maxRecDepth 8192 in
theorem root14Candidates6_checked : candidates edge 4 59 1163 root14Tail7 winner.bound = root14Candidates6 := by
  unfold candidates
  have row : edge 1163 = (SortedNatTree.contains FastRows.row1163) := by
    change Data.row1163.contains = (SortedNatTree.contains FastRows.row1163)
    exact FastRows.row1163_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root14Child6 : search edge 3 [59, 1163] root14Candidates6 1222 winner = winner := by decide
theorem root14Step6 : scan edge (search edge 3) 4 [59] 59 root14Tail6 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail7 winner := by
  rw [root14Tail6, scan, if_neg (by decide), root14Candidates6_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail7
    (if 3 ≤ root14Candidates6.length then search edge 3 [59, 1163] root14Candidates6 1222 winner else winner) = _
  rw [if_pos (by decide), root14Child6]
def root14Candidates7 : List Nat := [1901, 4931, 5519, 6779, 7559, 9341, 14627, 23189]
set_option maxRecDepth 8192 in
theorem root14Candidates7_checked : candidates edge 4 59 1193 root14Tail8 winner.bound = root14Candidates7 := by
  unfold candidates
  have row : edge 1193 = (SortedNatTree.contains FastRows.row1193) := by
    change Data.row1193.contains = (SortedNatTree.contains FastRows.row1193)
    exact FastRows.row1193_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root14Child7 : search edge 3 [59, 1193] root14Candidates7 1252 winner = winner := by decide
theorem root14Step7 : scan edge (search edge 3) 4 [59] 59 root14Tail7 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail8 winner := by
  rw [root14Tail7, scan, if_neg (by decide), root14Candidates7_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail8
    (if 3 ≤ root14Candidates7.length then search edge 3 [59, 1193] root14Candidates7 1252 winner else winner) = _
  rw [if_pos (by decide), root14Child7]
def root14Candidates8 : List Nat := [2609, 5039, 11351]
set_option maxRecDepth 8192 in
theorem root14Candidates8_checked : candidates edge 4 59 1259 root14Tail9 winner.bound = root14Candidates8 := by
  unfold candidates
  have row : edge 1259 = (SortedNatTree.contains FastRows.row1259) := by
    change Data.row1259.contains = (SortedNatTree.contains FastRows.row1259)
    exact FastRows.row1259_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root14Child8 : search edge 3 [59, 1259] root14Candidates8 1318 winner = winner := by decide
theorem root14Step8 : scan edge (search edge 3) 4 [59] 59 root14Tail8 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail9 winner := by
  rw [root14Tail8, scan, if_neg (by decide), root14Candidates8_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail9
    (if 3 ≤ root14Candidates8.length then search edge 3 [59, 1259] root14Candidates8 1318 winner else winner) = _
  rw [if_pos (by decide), root14Child8]
def root14Candidates9 : List Nat := [2237, 3767, 8933, 13241]
set_option maxRecDepth 8192 in
theorem root14Candidates9_checked : candidates edge 4 59 1289 root14Tail10 winner.bound = root14Candidates9 := by
  unfold candidates
  have row : edge 1289 = (SortedNatTree.contains FastRows.row1289) := by
    change Data.row1289.contains = (SortedNatTree.contains FastRows.row1289)
    exact FastRows.row1289_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root14Child9 : search edge 3 [59, 1289] root14Candidates9 1348 winner = winner := by decide
theorem root14Step9 : scan edge (search edge 3) 4 [59] 59 root14Tail9 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail10 winner := by
  rw [root14Tail9, scan, if_neg (by decide), root14Candidates9_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail10
    (if 3 ≤ root14Candidates9.length then search edge 3 [59, 1289] root14Candidates9 1348 winner else winner) = _
  rw [if_pos (by decide), root14Child9]
def root14Candidates10 : List Nat := [1901, 5843, 5981, 9341]
set_option maxRecDepth 8192 in
theorem root14Candidates10_checked : candidates edge 4 59 1319 root14Tail11 winner.bound = root14Candidates10 := by
  unfold candidates
  have row : edge 1319 = (SortedNatTree.contains FastRows.row1319) := by
    change Data.row1319.contains = (SortedNatTree.contains FastRows.row1319)
    exact FastRows.row1319_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root14Child10 : search edge 3 [59, 1319] root14Candidates10 1378 winner = winner := by decide
theorem root14Step10 : scan edge (search edge 3) 4 [59] 59 root14Tail10 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail11 winner := by
  rw [root14Tail10, scan, if_neg (by decide), root14Candidates10_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail11
    (if 3 ≤ root14Candidates10.length then search edge 3 [59, 1319] root14Candidates10 1378 winner else winner) = _
  rw [if_pos (by decide), root14Child10]
def root14Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root14Candidates11_checked : candidates edge 4 59 1601 root14Tail12 winner.bound = root14Candidates11 := by
  unfold candidates
  have row : edge 1601 = (SortedNatTree.contains FastRows.row1601) := by
    change Data.row1601.contains = (SortedNatTree.contains FastRows.row1601)
    exact FastRows.row1601_checked.symm
  rw [row]
  decide
theorem root14Step11 : scan edge (search edge 3) 4 [59] 59 root14Tail11 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail12 winner := by
  rw [root14Tail11, scan, if_neg (by decide), root14Candidates11_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail12
    (if 3 ≤ root14Candidates11.length then search edge 3 [59, 1601] root14Candidates11 1660 winner else winner) = _
  rw [if_neg (by decide)]
def root14Candidates12 : List Nat := [18371]
set_option maxRecDepth 8192 in
theorem root14Candidates12_checked : candidates edge 4 59 1901 root14Tail13 winner.bound = root14Candidates12 := by
  unfold candidates
  have row : edge 1901 = (SortedNatTree.contains FastRows.row1901) := by
    change Data.row1901.contains = (SortedNatTree.contains FastRows.row1901)
    exact FastRows.row1901_checked.symm
  rw [row]
  decide
theorem root14Step12 : scan edge (search edge 3) 4 [59] 59 root14Tail12 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail13 winner := by
  rw [root14Tail12, scan, if_neg (by decide), root14Candidates12_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail13
    (if 3 ≤ root14Candidates12.length then search edge 3 [59, 1901] root14Candidates12 1960 winner else winner) = _
  rw [if_neg (by decide)]
def root14Candidates13 : List Nat := [4397, 4751, 5741, 11057, 22067]
set_option maxRecDepth 8192 in
theorem root14Candidates13_checked : candidates edge 4 59 1973 root14Tail14 winner.bound = root14Candidates13 := by
  unfold candidates
  have row : edge 1973 = (SortedNatTree.contains FastRows.row1973) := by
    change Data.row1973.contains = (SortedNatTree.contains FastRows.row1973)
    exact FastRows.row1973_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root14Child13 : search edge 3 [59, 1973] root14Candidates13 2032 winner = winner := by decide
theorem root14Step13 : scan edge (search edge 3) 4 [59] 59 root14Tail13 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail14 winner := by
  rw [root14Tail13, scan, if_neg (by decide), root14Candidates13_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail14
    (if 3 ≤ root14Candidates13.length then search edge 3 [59, 1973] root14Candidates13 2032 winner else winner) = _
  rw [if_pos (by decide), root14Child13]
def root14Candidates14 : List Nat := [7559, 8537]
set_option maxRecDepth 8192 in
theorem root14Candidates14_checked : candidates edge 4 59 2087 root14Tail15 winner.bound = root14Candidates14 := by
  unfold candidates
  have row : edge 2087 = (SortedNatTree.contains FastRows.row2087) := by
    change Data.row2087.contains = (SortedNatTree.contains FastRows.row2087)
    exact FastRows.row2087_checked.symm
  rw [row]
  decide
theorem root14Step14 : scan edge (search edge 3) 4 [59] 59 root14Tail14 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail15 winner := by
  rw [root14Tail14, scan, if_neg (by decide), root14Candidates14_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail15
    (if 3 ≤ root14Candidates14.length then search edge 3 [59, 2087] root14Candidates14 2146 winner else winner) = _
  rw [if_neg (by decide)]
def root14Candidates15 : List Nat := [3083, 5717, 6863, 8093, 9479, 10559]
set_option maxRecDepth 8192 in
theorem root14Candidates15_checked : candidates edge 4 59 2099 root14Tail16 winner.bound = root14Candidates15 := by
  unfold candidates
  have row : edge 2099 = (SortedNatTree.contains FastRows.row2099) := by
    change Data.row2099.contains = (SortedNatTree.contains FastRows.row2099)
    exact FastRows.row2099_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root14Child15 : search edge 3 [59, 2099] root14Candidates15 2158 winner = winner := by decide
theorem root14Step15 : scan edge (search edge 3) 4 [59] 59 root14Tail15 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail16 winner := by
  rw [root14Tail15, scan, if_neg (by decide), root14Candidates15_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail16
    (if 3 ≤ root14Candidates15.length then search edge 3 [59, 2099] root14Candidates15 2158 winner else winner) = _
  rw [if_pos (by decide), root14Child15]
def root14Candidates16 : List Nat := [6779, 6917, 13241]
set_option maxRecDepth 8192 in
theorem root14Candidates16_checked : candidates edge 4 59 2237 root14Tail17 winner.bound = root14Candidates16 := by
  unfold candidates
  have row : edge 2237 = (SortedNatTree.contains FastRows.row2237) := by
    change Data.row2237.contains = (SortedNatTree.contains FastRows.row2237)
    exact FastRows.row2237_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root14Child16 : search edge 3 [59, 2237] root14Candidates16 2296 winner = winner := by decide
theorem root14Step16 : scan edge (search edge 3) 4 [59] 59 root14Tail16 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail17 winner := by
  rw [root14Tail16, scan, if_neg (by decide), root14Candidates16_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail17
    (if 3 ≤ root14Candidates16.length then search edge 3 [59, 2237] root14Candidates16 2296 winner else winner) = _
  rw [if_pos (by decide), root14Child16]
def root14Candidates17 : List Nat := [5081, 15551, 15809, 16943, 18119]
set_option maxRecDepth 8192 in
theorem root14Candidates17_checked : candidates edge 4 59 2309 root14Tail18 winner.bound = root14Candidates17 := by
  unfold candidates
  have row : edge 2309 = (SortedNatTree.contains FastRows.row2309) := by
    change Data.row2309.contains = (SortedNatTree.contains FastRows.row2309)
    exact FastRows.row2309_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root14Child17 : search edge 3 [59, 2309] root14Candidates17 2368 winner = winner := by decide
theorem root14Step17 : scan edge (search edge 3) 4 [59] 59 root14Tail17 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail18 winner := by
  rw [root14Tail17, scan, if_neg (by decide), root14Candidates17_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail18
    (if 3 ≤ root14Candidates17.length then search edge 3 [59, 2309] root14Candidates17 2368 winner else winner) = _
  rw [if_pos (by decide), root14Child17]
def root14Candidates18 : List Nat := [5519, 9341, 9491]
set_option maxRecDepth 8192 in
theorem root14Candidates18_checked : candidates edge 4 59 2351 root14Tail19 winner.bound = root14Candidates18 := by
  unfold candidates
  have row : edge 2351 = (SortedNatTree.contains FastRows.row2351) := by
    change Data.row2351.contains = (SortedNatTree.contains FastRows.row2351)
    exact FastRows.row2351_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root14Child18 : search edge 3 [59, 2351] root14Candidates18 2410 winner = winner := by decide
theorem root14Step18 : scan edge (search edge 3) 4 [59] 59 root14Tail18 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail19 winner := by
  rw [root14Tail18, scan, if_neg (by decide), root14Candidates18_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail19
    (if 3 ≤ root14Candidates18.length then search edge 3 [59, 2351] root14Candidates18 2410 winner else winner) = _
  rw [if_pos (by decide), root14Child18]
def root14Candidates19 : List Nat := [4889, 8123, 15809, 16649, 16943, 17099]
set_option maxRecDepth 8192 in
theorem root14Candidates19_checked : candidates edge 4 59 2531 root14Tail20 winner.bound = root14Candidates19 := by
  unfold candidates
  have row : edge 2531 = (SortedNatTree.contains FastRows.row2531) := by
    change Data.row2531.contains = (SortedNatTree.contains FastRows.row2531)
    exact FastRows.row2531_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root14Child19 : search edge 3 [59, 2531] root14Candidates19 2590 winner = winner := by decide
theorem root14Step19 : scan edge (search edge 3) 4 [59] 59 root14Tail19 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail20 winner := by
  rw [root14Tail19, scan, if_neg (by decide), root14Candidates19_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail20
    (if 3 ≤ root14Candidates19.length then search edge 3 [59, 2531] root14Candidates19 2590 winner else winner) = _
  rw [if_pos (by decide), root14Child19]
def root14Candidates20 : List Nat := [4931, 6569, 19937]
set_option maxRecDepth 8192 in
theorem root14Candidates20_checked : candidates edge 4 59 2609 root14Tail21 winner.bound = root14Candidates20 := by
  unfold candidates
  have row : edge 2609 = (SortedNatTree.contains FastRows.row2609) := by
    change Data.row2609.contains = (SortedNatTree.contains FastRows.row2609)
    exact FastRows.row2609_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root14Child20 : search edge 3 [59, 2609] root14Candidates20 2668 winner = winner := by decide
theorem root14Step20 : scan edge (search edge 3) 4 [59] 59 root14Tail20 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail21 winner := by
  rw [root14Tail20, scan, if_neg (by decide), root14Candidates20_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail21
    (if 3 ≤ root14Candidates20.length then search edge 3 [59, 2609] root14Candidates20 2668 winner else winner) = _
  rw [if_pos (by decide), root14Child20]
def root14Candidates21 : List Nat := [8783, 16943, 22853]
set_option maxRecDepth 8192 in
theorem root14Candidates21_checked : candidates edge 4 59 2663 root14Tail22 winner.bound = root14Candidates21 := by
  unfold candidates
  have row : edge 2663 = (SortedNatTree.contains FastRows.row2663) := by
    change Data.row2663.contains = (SortedNatTree.contains FastRows.row2663)
    exact FastRows.row2663_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root14Child21 : search edge 3 [59, 2663] root14Candidates21 2722 winner = winner := by decide
theorem root14Step21 : scan edge (search edge 3) 4 [59] 59 root14Tail21 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail22 winner := by
  rw [root14Tail21, scan, if_neg (by decide), root14Candidates21_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail22
    (if 3 ≤ root14Candidates21.length then search edge 3 [59, 2663] root14Candidates21 2722 winner else winner) = _
  rw [if_pos (by decide), root14Child21]
def root14Candidates22 : List Nat := [4721, 8867, 11057]
set_option maxRecDepth 8192 in
theorem root14Candidates22_checked : candidates edge 4 59 2897 root14Tail23 winner.bound = root14Candidates22 := by
  unfold candidates
  have row : edge 2897 = (SortedNatTree.contains FastRows.row2897) := by
    change Data.row2897.contains = (SortedNatTree.contains FastRows.row2897)
    exact FastRows.row2897_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root14Child22 : search edge 3 [59, 2897] root14Candidates22 2956 winner = winner := by decide
theorem root14Step22 : scan edge (search edge 3) 4 [59] 59 root14Tail22 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail23 winner := by
  rw [root14Tail22, scan, if_neg (by decide), root14Candidates22_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail23
    (if 3 ≤ root14Candidates22.length then search edge 3 [59, 2897] root14Candidates22 2956 winner else winner) = _
  rw [if_pos (by decide), root14Child22]
def root14Candidates23 : List Nat := [4721, 5927, 6779, 8123, 19937]
set_option maxRecDepth 8192 in
theorem root14Candidates23_checked : candidates edge 4 59 2903 root14Tail24 winner.bound = root14Candidates23 := by
  unfold candidates
  have row : edge 2903 = (SortedNatTree.contains FastRows.row2903) := by
    change Data.row2903.contains = (SortedNatTree.contains FastRows.row2903)
    exact FastRows.row2903_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root14Child23 : search edge 3 [59, 2903] root14Candidates23 2962 winner = winner := by decide
theorem root14Step23 : scan edge (search edge 3) 4 [59] 59 root14Tail23 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail24 winner := by
  rw [root14Tail23, scan, if_neg (by decide), root14Candidates23_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail24
    (if 3 ≤ root14Candidates23.length then search edge 3 [59, 2903] root14Candidates23 2962 winner else winner) = _
  rw [if_pos (by decide), root14Child23]
def root14Candidates24 : List Nat := [3449, 4931, 17903]
set_option maxRecDepth 8192 in
theorem root14Candidates24_checked : candidates edge 4 59 3083 root14Tail25 winner.bound = root14Candidates24 := by
  unfold candidates
  have row : edge 3083 = (SortedNatTree.contains FastRows.row3083) := by
    change Data.row3083.contains = (SortedNatTree.contains FastRows.row3083)
    exact FastRows.row3083_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root14Child24 : search edge 3 [59, 3083] root14Candidates24 3142 winner = winner := by decide
theorem root14Step24 : scan edge (search edge 3) 4 [59] 59 root14Tail24 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail25 winner := by
  rw [root14Tail24, scan, if_neg (by decide), root14Candidates24_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail25
    (if 3 ≤ root14Candidates24.length then search edge 3 [59, 3083] root14Candidates24 3142 winner else winner) = _
  rw [if_pos (by decide), root14Child24]
def root14Candidates25 : List Nat := [15359]
set_option maxRecDepth 8192 in
theorem root14Candidates25_checked : candidates edge 4 59 3449 root14Tail26 winner.bound = root14Candidates25 := by
  unfold candidates
  have row : edge 3449 = (SortedNatTree.contains FastRows.row3449) := by
    change Data.row3449.contains = (SortedNatTree.contains FastRows.row3449)
    exact FastRows.row3449_checked.symm
  rw [row]
  decide
theorem root14Step25 : scan edge (search edge 3) 4 [59] 59 root14Tail25 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail26 winner := by
  rw [root14Tail25, scan, if_neg (by decide), root14Candidates25_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail26
    (if 3 ≤ root14Candidates25.length then search edge 3 [59, 3449] root14Candidates25 3508 winner else winner) = _
  rw [if_neg (by decide)]
def root14Candidates26 : List Nat := [8933, 13763]
set_option maxRecDepth 8192 in
theorem root14Candidates26_checked : candidates edge 4 59 3767 root14Tail27 winner.bound = root14Candidates26 := by
  unfold candidates
  have row : edge 3767 = (SortedNatTree.contains FastRows.row3767) := by
    change Data.row3767.contains = (SortedNatTree.contains FastRows.row3767)
    exact FastRows.row3767_checked.symm
  rw [row]
  decide
theorem root14Step26 : scan edge (search edge 3) 4 [59] 59 root14Tail26 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail27 winner := by
  rw [root14Tail26, scan, if_neg (by decide), root14Candidates26_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail27
    (if 3 ≤ root14Candidates26.length then search edge 3 [59, 3767] root14Candidates26 3826 winner else winner) = _
  rw [if_neg (by decide)]
def root14Candidates27 : List Nat := [6899, 6983, 7673, 9491]
set_option maxRecDepth 8192 in
theorem root14Candidates27_checked : candidates edge 4 59 3851 root14Tail28 winner.bound = root14Candidates27 := by
  unfold candidates
  have row : edge 3851 = (SortedNatTree.contains FastRows.row3851) := by
    change Data.row3851.contains = (SortedNatTree.contains FastRows.row3851)
    exact FastRows.row3851_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root14Child27 : search edge 3 [59, 3851] root14Candidates27 3910 winner = winner := by decide
theorem root14Step27 : scan edge (search edge 3) 4 [59] 59 root14Tail27 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail28 winner := by
  rw [root14Tail27, scan, if_neg (by decide), root14Candidates27_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail28
    (if 3 ≤ root14Candidates27.length then search edge 3 [59, 3851] root14Candidates27 3910 winner else winner) = _
  rw [if_pos (by decide), root14Child27]
def root14Candidates28 : List Nat := [5717, 21599]
set_option maxRecDepth 8192 in
theorem root14Candidates28_checked : candidates edge 4 59 4157 root14Tail29 winner.bound = root14Candidates28 := by
  unfold candidates
  have row : edge 4157 = (SortedNatTree.contains FastRows.row4157) := by
    change Data.row4157.contains = (SortedNatTree.contains FastRows.row4157)
    exact FastRows.row4157_checked.symm
  rw [row]
  decide
theorem root14Step28 : scan edge (search edge 3) 4 [59] 59 root14Tail28 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail29 winner := by
  rw [root14Tail28, scan, if_neg (by decide), root14Candidates28_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail29
    (if 3 ≤ root14Candidates28.length then search edge 3 [59, 4157] root14Candidates28 4216 winner else winner) = _
  rw [if_neg (by decide)]
def root14Candidates29 : List Nat := [5351, 10973, 19289]
set_option maxRecDepth 8192 in
theorem root14Candidates29_checked : candidates edge 4 59 4211 root14Tail30 winner.bound = root14Candidates29 := by
  unfold candidates
  have row : edge 4211 = (SortedNatTree.contains FastRows.row4211) := by
    change Data.row4211.contains = (SortedNatTree.contains FastRows.row4211)
    exact FastRows.row4211_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root14Child29 : search edge 3 [59, 4211] root14Candidates29 4270 winner = winner := by decide
theorem root14Step29 : scan edge (search edge 3) 4 [59] 59 root14Tail29 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail30 winner := by
  rw [root14Tail29, scan, if_neg (by decide), root14Candidates29_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail30
    (if 3 ≤ root14Candidates29.length then search edge 3 [59, 4211] root14Candidates29 4270 winner else winner) = _
  rw [if_pos (by decide), root14Child29]
def root14Candidates30 : List Nat := [11057, 20219]
set_option maxRecDepth 8192 in
theorem root14Candidates30_checked : candidates edge 4 59 4397 root14Tail31 winner.bound = root14Candidates30 := by
  unfold candidates
  have row : edge 4397 = (SortedNatTree.contains FastRows.row4397) := by
    change Data.row4397.contains = (SortedNatTree.contains FastRows.row4397)
    exact FastRows.row4397_checked.symm
  rw [row]
  decide
theorem root14Step30 : scan edge (search edge 3) 4 [59] 59 root14Tail30 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail31 winner := by
  rw [root14Tail30, scan, if_neg (by decide), root14Candidates30_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail31
    (if 3 ≤ root14Candidates30.length then search edge 3 [59, 4397] root14Candidates30 4456 winner else winner) = _
  rw [if_neg (by decide)]
def root14Candidates31 : List Nat := [14627, 19031]
set_option maxRecDepth 8192 in
theorem root14Candidates31_checked : candidates edge 4 59 4721 root14Tail32 winner.bound = root14Candidates31 := by
  unfold candidates
  have row : edge 4721 = (SortedNatTree.contains FastRows.row4721) := by
    change Data.row4721.contains = (SortedNatTree.contains FastRows.row4721)
    exact FastRows.row4721_checked.symm
  rw [row]
  decide
theorem root14Step31 : scan edge (search edge 3) 4 [59] 59 root14Tail31 winner =
    scan edge (search edge 3) 4 [59] 59 root14Tail32 winner := by
  rw [root14Tail31, scan, if_neg (by decide), root14Candidates31_checked]
  change scan edge (search edge 3) 4 [59] 59 root14Tail32
    (if 3 ≤ root14Candidates31.length then search edge 3 [59, 4721] root14Candidates31 4780 winner else winner) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
