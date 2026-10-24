import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot021Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root21Candidates0 : List Nat := [1061, 4973, 5507, 23357]
set_option maxRecDepth 8192 in
theorem root21Candidates0_checked : candidates edge 4 89 107 root21Tail1 winner.bound = root21Candidates0 := by
  unfold candidates
  have row : edge 107 = (SortedNatTree.contains FastRows.row107) := by
    change Data.row107.contains = (SortedNatTree.contains FastRows.row107)
    exact FastRows.row107_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root21Child0 : search edge 3 [89, 107] root21Candidates0 196 winner = winner := by decide
theorem root21Step0 : scan edge (search edge 3) 4 [89] 89 root21Tail0 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail1 winner := by
  rw [root21Tail0, scan, if_neg (by decide), root21Candidates0_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail1
    (if 3 ≤ root21Candidates0.length then search edge 3 [89, 107] root21Candidates0 196 winner else winner) = _
  rw [if_pos (by decide), root21Child0]
def root21Candidates1 : List Nat := [3989, 4721, 8147, 12791, 13187, 14621, 18773, 22937]
set_option maxRecDepth 8192 in
theorem root21Candidates1_checked : candidates edge 4 89 137 root21Tail2 winner.bound = root21Candidates1 := by
  unfold candidates
  have row : edge 137 = (SortedNatTree.contains FastRows.row137) := by
    change Data.row137.contains = (SortedNatTree.contains FastRows.row137)
    exact FastRows.row137_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root21Child1 : search edge 3 [89, 137] root21Candidates1 226 winner = winner := by decide
theorem root21Step1 : scan edge (search edge 3) 4 [89] 89 root21Tail1 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail2 winner := by
  rw [root21Tail1, scan, if_neg (by decide), root21Candidates1_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail2
    (if 3 ≤ root21Candidates1.length then search edge 3 [89, 137] root21Candidates1 226 winner else winner) = _
  rw [if_pos (by decide), root21Child1]
def root21Candidates2 : List Nat := [1823, 2777, 5171, 5189, 8753, 9473, 21863]
set_option maxRecDepth 8192 in
theorem root21Candidates2_checked : candidates edge 4 89 293 root21Tail3 winner.bound = root21Candidates2 := by
  unfold candidates
  have row : edge 293 = (SortedNatTree.contains FastRows.row293) := by
    change Data.row293.contains = (SortedNatTree.contains FastRows.row293)
    exact FastRows.row293_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root21Child2 : search edge 3 [89, 293] root21Candidates2 382 winner = winner := by decide
theorem root21Step2 : scan edge (search edge 3) 4 [89] 89 root21Tail2 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail3 winner := by
  rw [root21Tail2, scan, if_neg (by decide), root21Candidates2_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail3
    (if 3 ≤ root21Candidates2.length then search edge 3 [89, 293] root21Candidates2 382 winner else winner) = _
  rw [if_pos (by decide), root21Child2]
def root21Candidates3 : List Nat := [983, 1823, 3671, 5903, 11489, 13043, 20771, 21467]
set_option maxRecDepth 8192 in
theorem root21Candidates3_checked : candidates edge 4 89 431 root21Tail4 winner.bound = root21Candidates3 := by
  unfold candidates
  have row : edge 431 = (SortedNatTree.contains FastRows.row431) := by
    change Data.row431.contains = (SortedNatTree.contains FastRows.row431)
    exact FastRows.row431_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root21Child3 : search edge 3 [89, 431] root21Candidates3 520 winner = winner := by decide
theorem root21Step3 : scan edge (search edge 3) 4 [89] 89 root21Tail3 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail4 winner := by
  rw [root21Tail3, scan, if_neg (by decide), root21Candidates3_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail4
    (if 3 ≤ root21Candidates3.length then search edge 3 [89, 431] root21Candidates3 520 winner else winner) = _
  rw [if_pos (by decide), root21Child3]
def root21Candidates4 : List Nat := [983, 1523, 17729]
set_option maxRecDepth 8192 in
theorem root21Candidates4_checked : candidates edge 4 89 443 root21Tail5 winner.bound = root21Candidates4 := by
  unfold candidates
  have row : edge 443 = (SortedNatTree.contains FastRows.row443) := by
    change Data.row443.contains = (SortedNatTree.contains FastRows.row443)
    exact FastRows.row443_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root21Child4 : search edge 3 [89, 443] root21Candidates4 532 winner = winner := by decide
theorem root21Step4 : scan edge (search edge 3) 4 [89] 89 root21Tail4 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail5 winner := by
  rw [root21Tail4, scan, if_neg (by decide), root21Candidates4_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail5
    (if 3 ≤ root21Candidates4.length then search edge 3 [89, 443] root21Candidates4 532 winner else winner) = _
  rw [if_pos (by decide), root21Child4]
def root21Candidates5 : List Nat := [809, 1523, 2777, 6353, 6521, 9791, 20771, 21467, 22817]
set_option maxRecDepth 8192 in
theorem root21Candidates5_checked : candidates edge 4 89 521 root21Tail6 winner.bound = root21Candidates5 := by
  unfold candidates
  have row : edge 521 = (SortedNatTree.contains FastRows.row521) := by
    change Data.row521.contains = (SortedNatTree.contains FastRows.row521)
    exact FastRows.row521_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root21Child5 : search edge 3 [89, 521] root21Candidates5 610 winner = winner := by decide
theorem root21Step5 : scan edge (search edge 3) 4 [89] 89 root21Tail5 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail6 winner := by
  rw [root21Tail5, scan, if_neg (by decide), root21Candidates5_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail6
    (if 3 ≤ root21Candidates5.length then search edge 3 [89, 521] root21Candidates5 610 winner else winner) = _
  rw [if_pos (by decide), root21Child5]
def root21Candidates6 : List Nat := [821, 983, 1427, 2141, 5393]
set_option maxRecDepth 8192 in
theorem root21Candidates6_checked : candidates edge 4 89 809 root21Tail7 winner.bound = root21Candidates6 := by
  unfold candidates
  have row : edge 809 = (SortedNatTree.contains FastRows.row809) := by
    change Data.row809.contains = (SortedNatTree.contains FastRows.row809)
    exact FastRows.row809_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root21Child6 : search edge 3 [89, 809] root21Candidates6 898 winner = winner := by decide
theorem root21Step6 : scan edge (search edge 3) 4 [89] 89 root21Tail6 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail7 winner := by
  rw [root21Tail6, scan, if_neg (by decide), root21Candidates6_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail7
    (if 3 ≤ root21Candidates6.length then search edge 3 [89, 809] root21Candidates6 898 winner else winner) = _
  rw [if_pos (by decide), root21Child6]
def root21Candidates7 : List Nat := [6521, 8663, 8753, 12347, 18521]
set_option maxRecDepth 8192 in
theorem root21Candidates7_checked : candidates edge 4 89 821 root21Tail8 winner.bound = root21Candidates7 := by
  unfold candidates
  have row : edge 821 = (SortedNatTree.contains FastRows.row821) := by
    change Data.row821.contains = (SortedNatTree.contains FastRows.row821)
    exact FastRows.row821_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root21Child7 : search edge 3 [89, 821] root21Candidates7 910 winner = winner := by decide
theorem root21Step7 : scan edge (search edge 3) 4 [89] 89 root21Tail7 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail8 winner := by
  rw [root21Tail7, scan, if_neg (by decide), root21Candidates7_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail8
    (if 3 ≤ root21Candidates7.length then search edge 3 [89, 821] root21Candidates7 910 winner else winner) = _
  rw [if_pos (by decide), root21Child7]
def root21Candidates8 : List Nat := [1409, 2141, 5039, 8753, 12347, 19913, 21467, 23063]
set_option maxRecDepth 8192 in
theorem root21Candidates8_checked : candidates edge 4 89 977 root21Tail9 winner.bound = root21Candidates8 := by
  unfold candidates
  have row : edge 977 = (SortedNatTree.contains FastRows.row977) := by
    change Data.row977.contains = (SortedNatTree.contains FastRows.row977)
    exact FastRows.row977_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root21Child8 : search edge 3 [89, 977] root21Candidates8 1066 winner = winner := by decide
theorem root21Step8 : scan edge (search edge 3) 4 [89] 89 root21Tail8 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail9 winner := by
  rw [root21Tail8, scan, if_neg (by decide), root21Candidates8_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail9
    (if 3 ≤ root21Candidates8.length then search edge 3 [89, 977] root21Candidates8 1066 winner else winner) = _
  rw [if_pos (by decide), root21Child8]
def root21Candidates9 : List Nat := [2267, 5171, 11489, 13043, 15083, 18797, 21467]
set_option maxRecDepth 8192 in
theorem root21Candidates9_checked : candidates edge 4 89 983 root21Tail10 winner.bound = root21Candidates9 := by
  unfold candidates
  have row : edge 983 = (SortedNatTree.contains FastRows.row983) := by
    change Data.row983.contains = (SortedNatTree.contains FastRows.row983)
    exact FastRows.row983_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root21Child9 : search edge 3 [89, 983] root21Candidates9 1072 winner = winner := by decide
theorem root21Step9 : scan edge (search edge 3) 4 [89] 89 root21Tail9 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail10 winner := by
  rw [root21Tail9, scan, if_neg (by decide), root21Candidates9_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail10
    (if 3 ≤ root21Candidates9.length then search edge 3 [89, 983] root21Candidates9 1072 winner else winner) = _
  rw [if_pos (by decide), root21Child9]
def root21Candidates10 : List Nat := [4973, 10463, 17021, 22811, 23669]
set_option maxRecDepth 8192 in
theorem root21Candidates10_checked : candidates edge 4 89 1061 root21Tail11 winner.bound = root21Candidates10 := by
  unfold candidates
  have row : edge 1061 = (SortedNatTree.contains FastRows.row1061) := by
    change Data.row1061.contains = (SortedNatTree.contains FastRows.row1061)
    exact FastRows.row1061_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root21Child10 : search edge 3 [89, 1061] root21Candidates10 1150 winner = winner := by decide
theorem root21Step10 : scan edge (search edge 3) 4 [89] 89 root21Tail10 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail11 winner := by
  rw [root21Tail10, scan, if_neg (by decide), root21Candidates10_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail11
    (if 3 ≤ root21Candidates10.length then search edge 3 [89, 1061] root21Candidates10 1150 winner else winner) = _
  rw [if_pos (by decide), root21Child10]
def root21Candidates11 : List Nat := [2141, 2579, 2777, 6113]
set_option maxRecDepth 8192 in
theorem root21Candidates11_checked : candidates edge 4 89 1223 root21Tail12 winner.bound = root21Candidates11 := by
  unfold candidates
  have row : edge 1223 = (SortedNatTree.contains FastRows.row1223) := by
    change Data.row1223.contains = (SortedNatTree.contains FastRows.row1223)
    exact FastRows.row1223_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root21Child11 : search edge 3 [89, 1223] root21Candidates11 1312 winner = winner := by decide
theorem root21Step11 : scan edge (search edge 3) 4 [89] 89 root21Tail11 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail12 winner := by
  rw [root21Tail11, scan, if_neg (by decide), root21Candidates11_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail12
    (if 3 ≤ root21Candidates11.length then search edge 3 [89, 1223] root21Candidates11 1312 winner else winner) = _
  rw [if_pos (by decide), root21Child11]
def root21Candidates12 : List Nat := [5387, 6521, 9473, 15083, 15461, 19319, 22811]
set_option maxRecDepth 8192 in
theorem root21Candidates12_checked : candidates edge 4 89 1409 root21Tail13 winner.bound = root21Candidates12 := by
  unfold candidates
  have row : edge 1409 = (SortedNatTree.contains FastRows.row1409) := by
    change Data.row1409.contains = (SortedNatTree.contains FastRows.row1409)
    exact FastRows.row1409_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root21Child12 : search edge 3 [89, 1409] root21Candidates12 1498 winner = winner := by decide
theorem root21Step12 : scan edge (search edge 3) 4 [89] 89 root21Tail12 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail13 winner := by
  rw [root21Tail12, scan, if_neg (by decide), root21Candidates12_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail13
    (if 3 ≤ root21Candidates12.length then search edge 3 [89, 1409] root21Candidates12 1498 winner else winner) = _
  rw [if_pos (by decide), root21Child12]
def root21Candidates13 : List Nat := [2777, 3989]
set_option maxRecDepth 8192 in
theorem root21Candidates13_checked : candidates edge 4 89 1427 root21Tail14 winner.bound = root21Candidates13 := by
  unfold candidates
  have row : edge 1427 = (SortedNatTree.contains FastRows.row1427) := by
    change Data.row1427.contains = (SortedNatTree.contains FastRows.row1427)
    exact FastRows.row1427_checked.symm
  rw [row]
  decide
theorem root21Step13 : scan edge (search edge 3) 4 [89] 89 root21Tail13 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail14 winner := by
  rw [root21Tail13, scan, if_neg (by decide), root21Candidates13_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail14
    (if 3 ≤ root21Candidates13.length then search edge 3 [89, 1427] root21Candidates13 1516 winner else winner) = _
  rw [if_neg (by decide)]
def root21Candidates14 : List Nat := [2579, 5309, 23357, 23669]
set_option maxRecDepth 8192 in
theorem root21Candidates14_checked : candidates edge 4 89 1523 root21Tail15 winner.bound = root21Candidates14 := by
  unfold candidates
  have row : edge 1523 = (SortedNatTree.contains FastRows.row1523) := by
    change Data.row1523.contains = (SortedNatTree.contains FastRows.row1523)
    exact FastRows.row1523_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root21Child14 : search edge 3 [89, 1523] root21Candidates14 1612 winner = winner := by decide
theorem root21Step14 : scan edge (search edge 3) 4 [89] 89 root21Tail14 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail15 winner := by
  rw [root21Tail14, scan, if_neg (by decide), root21Candidates14_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail15
    (if 3 ≤ root21Candidates14.length then search edge 3 [89, 1523] root21Candidates14 1612 winner else winner) = _
  rw [if_pos (by decide), root21Child14]
def root21Candidates15 : List Nat := [5333, 5879, 10271, 14879]
set_option maxRecDepth 8192 in
theorem root21Candidates15_checked : candidates edge 4 89 1571 root21Tail16 winner.bound = root21Candidates15 := by
  unfold candidates
  have row : edge 1571 = (SortedNatTree.contains FastRows.row1571) := by
    change Data.row1571.contains = (SortedNatTree.contains FastRows.row1571)
    exact FastRows.row1571_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root21Child15 : search edge 3 [89, 1571] root21Candidates15 1660 winner = winner := by decide
theorem root21Step15 : scan edge (search edge 3) 4 [89] 89 root21Tail15 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail16 winner := by
  rw [root21Tail15, scan, if_neg (by decide), root21Candidates15_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail16
    (if 3 ≤ root21Candidates15.length then search edge 3 [89, 1571] root21Candidates15 1660 winner else winner) = _
  rw [if_pos (by decide), root21Child15]
def root21Candidates16 : List Nat := [6353, 6947]
set_option maxRecDepth 8192 in
theorem root21Candidates16_checked : candidates edge 4 89 1823 root21Tail17 winner.bound = root21Candidates16 := by
  unfold candidates
  have row : edge 1823 = (SortedNatTree.contains FastRows.row1823) := by
    change Data.row1823.contains = (SortedNatTree.contains FastRows.row1823)
    exact FastRows.row1823_checked.symm
  rw [row]
  decide
theorem root21Step16 : scan edge (search edge 3) 4 [89] 89 root21Tail16 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail17 winner := by
  rw [root21Tail16, scan, if_neg (by decide), root21Candidates16_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail17
    (if 3 ≤ root21Candidates16.length then search edge 3 [89, 1823] root21Candidates16 1912 winner else winner) = _
  rw [if_neg (by decide)]
def root21Candidates17 : List Nat := [2267, 7727, 8951]
set_option maxRecDepth 8192 in
theorem root21Candidates17_checked : candidates edge 4 89 2141 root21Tail18 winner.bound = root21Candidates17 := by
  unfold candidates
  have row : edge 2141 = (SortedNatTree.contains FastRows.row2141) := by
    change Data.row2141.contains = (SortedNatTree.contains FastRows.row2141)
    exact FastRows.row2141_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root21Child17 : search edge 3 [89, 2141] root21Candidates17 2230 winner = winner := by decide
theorem root21Step17 : scan edge (search edge 3) 4 [89] 89 root21Tail17 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail18 winner := by
  rw [root21Tail17, scan, if_neg (by decide), root21Candidates17_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail18
    (if 3 ≤ root21Candidates17.length then search edge 3 [89, 2141] root21Candidates17 2230 winner else winner) = _
  rw [if_pos (by decide), root21Child17]
def root21Candidates18 : List Nat := [14621, 18911]
set_option maxRecDepth 8192 in
theorem root21Candidates18_checked : candidates edge 4 89 2153 root21Tail19 winner.bound = root21Candidates18 := by
  unfold candidates
  have row : edge 2153 = (SortedNatTree.contains FastRows.row2153) := by
    change Data.row2153.contains = (SortedNatTree.contains FastRows.row2153)
    exact FastRows.row2153_checked.symm
  rw [row]
  decide
theorem root21Step18 : scan edge (search edge 3) 4 [89] 89 root21Tail18 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail19 winner := by
  rw [root21Tail18, scan, if_neg (by decide), root21Candidates18_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail19
    (if 3 ≤ root21Candidates18.length then search edge 3 [89, 2153] root21Candidates18 2242 winner else winner) = _
  rw [if_neg (by decide)]
def root21Candidates19 : List Nat := [4721, 5879, 8147, 15551]
set_option maxRecDepth 8192 in
theorem root21Candidates19_checked : candidates edge 4 89 2267 root21Tail20 winner.bound = root21Candidates19 := by
  unfold candidates
  have row : edge 2267 = (SortedNatTree.contains FastRows.row2267) := by
    change Data.row2267.contains = (SortedNatTree.contains FastRows.row2267)
    exact FastRows.row2267_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root21Child19 : search edge 3 [89, 2267] root21Candidates19 2356 winner = winner := by decide
theorem root21Step19 : scan edge (search edge 3) 4 [89] 89 root21Tail19 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail20 winner := by
  rw [root21Tail19, scan, if_neg (by decide), root21Candidates19_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail20
    (if 3 ≤ root21Candidates19.length then search edge 3 [89, 2267] root21Candidates19 2356 winner else winner) = _
  rw [if_pos (by decide), root21Child19]
def root21Candidates20 : List Nat := [2579, 3821, 5087, 6323]
set_option maxRecDepth 8192 in
theorem root21Candidates20_checked : candidates edge 4 89 2357 root21Tail21 winner.bound = root21Candidates20 := by
  unfold candidates
  have row : edge 2357 = (SortedNatTree.contains FastRows.row2357) := by
    change Data.row2357.contains = (SortedNatTree.contains FastRows.row2357)
    exact FastRows.row2357_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root21Child20 : search edge 3 [89, 2357] root21Candidates20 2446 winner = winner := by decide
theorem root21Step20 : scan edge (search edge 3) 4 [89] 89 root21Tail20 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail21 winner := by
  rw [root21Tail20, scan, if_neg (by decide), root21Candidates20_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail21
    (if 3 ≤ root21Candidates20.length then search edge 3 [89, 2357] root21Candidates20 2446 winner else winner) = _
  rw [if_pos (by decide), root21Child20]
def root21Candidates21 : List Nat := [7727]
set_option maxRecDepth 8192 in
theorem root21Candidates21_checked : candidates edge 4 89 2579 root21Tail22 winner.bound = root21Candidates21 := by
  unfold candidates
  have row : edge 2579 = (SortedNatTree.contains FastRows.row2579) := by
    change Data.row2579.contains = (SortedNatTree.contains FastRows.row2579)
    exact FastRows.row2579_checked.symm
  rw [row]
  decide
theorem root21Step21 : scan edge (search edge 3) 4 [89] 89 root21Tail21 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail22 winner := by
  rw [root21Tail21, scan, if_neg (by decide), root21Candidates21_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail22
    (if 3 ≤ root21Candidates21.length then search edge 3 [89, 2579] root21Candidates21 2668 winner else winner) = _
  rw [if_neg (by decide)]
def root21Candidates22 : List Nat := [4973, 22811]
set_option maxRecDepth 8192 in
theorem root21Candidates22_checked : candidates edge 4 89 2777 root21Tail23 winner.bound = root21Candidates22 := by
  unfold candidates
  have row : edge 2777 = (SortedNatTree.contains FastRows.row2777) := by
    change Data.row2777.contains = (SortedNatTree.contains FastRows.row2777)
    exact FastRows.row2777_checked.symm
  rw [row]
  decide
theorem root21Step22 : scan edge (search edge 3) 4 [89] 89 root21Tail22 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail23 winner := by
  rw [root21Tail22, scan, if_neg (by decide), root21Candidates22_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail23
    (if 3 ≤ root21Candidates22.length then search edge 3 [89, 2777] root21Candidates22 2866 winner else winner) = _
  rw [if_neg (by decide)]
def root21Candidates23 : List Nat := [3671, 6299, 11471]
set_option maxRecDepth 8192 in
theorem root21Candidates23_checked : candidates edge 4 89 3407 root21Tail24 winner.bound = root21Candidates23 := by
  unfold candidates
  have row : edge 3407 = (SortedNatTree.contains FastRows.row3407) := by
    change Data.row3407.contains = (SortedNatTree.contains FastRows.row3407)
    exact FastRows.row3407_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root21Child23 : search edge 3 [89, 3407] root21Candidates23 3496 winner = winner := by decide
theorem root21Step23 : scan edge (search edge 3) 4 [89] 89 root21Tail23 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail24 winner := by
  rw [root21Tail23, scan, if_neg (by decide), root21Candidates23_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail24
    (if 3 ≤ root21Candidates23.length then search edge 3 [89, 3407] root21Candidates23 3496 winner else winner) = _
  rw [if_pos (by decide), root21Child23]
def root21Candidates24 : List Nat := [5039, 15461]
set_option maxRecDepth 8192 in
theorem root21Candidates24_checked : candidates edge 4 89 3671 root21Tail25 winner.bound = root21Candidates24 := by
  unfold candidates
  have row : edge 3671 = (SortedNatTree.contains FastRows.row3671) := by
    change Data.row3671.contains = (SortedNatTree.contains FastRows.row3671)
    exact FastRows.row3671_checked.symm
  rw [row]
  decide
theorem root21Step24 : scan edge (search edge 3) 4 [89] 89 root21Tail24 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail25 winner := by
  rw [root21Tail24, scan, if_neg (by decide), root21Candidates24_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail25
    (if 3 ≤ root21Candidates24.length then search edge 3 [89, 3671] root21Candidates24 3760 winner else winner) = _
  rw [if_neg (by decide)]
def root21Candidates25 : List Nat := [6263, 7307, 8951, 13187]
set_option maxRecDepth 8192 in
theorem root21Candidates25_checked : candidates edge 4 89 3821 root21Tail26 winner.bound = root21Candidates25 := by
  unfold candidates
  have row : edge 3821 = (SortedNatTree.contains FastRows.row3821) := by
    change Data.row3821.contains = (SortedNatTree.contains FastRows.row3821)
    exact FastRows.row3821_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root21Child25 : search edge 3 [89, 3821] root21Candidates25 3910 winner = winner := by decide
theorem root21Step25 : scan edge (search edge 3) 4 [89] 89 root21Tail25 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail26 winner := by
  rw [root21Tail25, scan, if_neg (by decide), root21Candidates25_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail26
    (if 3 ≤ root21Candidates25.length then search edge 3 [89, 3821] root21Candidates25 3910 winner else winner) = _
  rw [if_pos (by decide), root21Child25]
def root21Candidates26 : List Nat := [4211, 5189, 5987, 6323, 8753, 12791]
set_option maxRecDepth 8192 in
theorem root21Candidates26_checked : candidates edge 4 89 3917 root21Tail27 winner.bound = root21Candidates26 := by
  unfold candidates
  have row : edge 3917 = (SortedNatTree.contains FastRows.row3917) := by
    change Data.row3917.contains = (SortedNatTree.contains FastRows.row3917)
    exact FastRows.row3917_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root21Child26 : search edge 3 [89, 3917] root21Candidates26 4006 winner = winner := by decide
theorem root21Step26 : scan edge (search edge 3) 4 [89] 89 root21Tail26 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail27 winner := by
  rw [root21Tail26, scan, if_neg (by decide), root21Candidates26_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail27
    (if 3 ≤ root21Candidates26.length then search edge 3 [89, 3917] root21Candidates26 4006 winner else winner) = _
  rw [if_pos (by decide), root21Child26]
def root21Candidates27 : List Nat := []
set_option maxRecDepth 8192 in
theorem root21Candidates27_checked : candidates edge 4 89 3989 root21Tail28 winner.bound = root21Candidates27 := by
  unfold candidates
  have row : edge 3989 = (SortedNatTree.contains FastRows.row3989) := by
    change Data.row3989.contains = (SortedNatTree.contains FastRows.row3989)
    exact FastRows.row3989_checked.symm
  rw [row]
  decide
theorem root21Step27 : scan edge (search edge 3) 4 [89] 89 root21Tail27 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail28 winner := by
  rw [root21Tail27, scan, if_neg (by decide), root21Candidates27_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail28
    (if 3 ≤ root21Candidates27.length then search edge 3 [89, 3989] root21Candidates27 4078 winner else winner) = _
  rw [if_neg (by decide)]
def root21Candidates28 : List Nat := []
set_option maxRecDepth 8192 in
theorem root21Candidates28_checked : candidates edge 4 89 4211 root21Tail29 winner.bound = root21Candidates28 := by
  unfold candidates
  have row : edge 4211 = (SortedNatTree.contains FastRows.row4211) := by
    change Data.row4211.contains = (SortedNatTree.contains FastRows.row4211)
    exact FastRows.row4211_checked.symm
  rw [row]
  decide
theorem root21Step28 : scan edge (search edge 3) 4 [89] 89 root21Tail28 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail29 winner := by
  rw [root21Tail28, scan, if_neg (by decide), root21Candidates28_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail29
    (if 3 ≤ root21Candidates28.length then search edge 3 [89, 4211] root21Candidates28 4300 winner else winner) = _
  rw [if_neg (by decide)]
def root21Candidates29 : List Nat := [5087, 17729]
set_option maxRecDepth 8192 in
theorem root21Candidates29_checked : candidates edge 4 89 4259 root21Tail30 winner.bound = root21Candidates29 := by
  unfold candidates
  have row : edge 4259 = (SortedNatTree.contains FastRows.row4259) := by
    change Data.row4259.contains = (SortedNatTree.contains FastRows.row4259)
    exact FastRows.row4259_checked.symm
  rw [row]
  decide
theorem root21Step29 : scan edge (search edge 3) 4 [89] 89 root21Tail29 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail30 winner := by
  rw [root21Tail29, scan, if_neg (by decide), root21Candidates29_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail30
    (if 3 ≤ root21Candidates29.length then search edge 3 [89, 4259] root21Candidates29 4348 winner else winner) = _
  rw [if_neg (by decide)]
def root21Candidates30 : List Nat := [5087, 9791, 18521]
set_option maxRecDepth 8192 in
theorem root21Candidates30_checked : candidates edge 4 89 4409 root21Tail31 winner.bound = root21Candidates30 := by
  unfold candidates
  have row : edge 4409 = (SortedNatTree.contains FastRows.row4409) := by
    change Data.row4409.contains = (SortedNatTree.contains FastRows.row4409)
    exact FastRows.row4409_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root21Child30 : search edge 3 [89, 4409] root21Candidates30 4498 winner = winner := by decide
theorem root21Step30 : scan edge (search edge 3) 4 [89] 89 root21Tail30 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail31 winner := by
  rw [root21Tail30, scan, if_neg (by decide), root21Candidates30_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail31
    (if 3 ≤ root21Candidates30.length then search edge 3 [89, 4409] root21Candidates30 4498 winner else winner) = _
  rw [if_pos (by decide), root21Child30]
def root21Candidates31 : List Nat := [8753]
set_option maxRecDepth 8192 in
theorem root21Candidates31_checked : candidates edge 4 89 4721 root21Tail32 winner.bound = root21Candidates31 := by
  unfold candidates
  have row : edge 4721 = (SortedNatTree.contains FastRows.row4721) := by
    change Data.row4721.contains = (SortedNatTree.contains FastRows.row4721)
    exact FastRows.row4721_checked.symm
  rw [row]
  decide
theorem root21Step31 : scan edge (search edge 3) 4 [89] 89 root21Tail31 winner =
    scan edge (search edge 3) 4 [89] 89 root21Tail32 winner := by
  rw [root21Tail31, scan, if_neg (by decide), root21Candidates31_checked]
  change scan edge (search edge 3) 4 [89] 89 root21Tail32
    (if 3 ≤ root21Candidates31.length then search edge 3 [89, 4721] root21Candidates31 4810 winner else winner) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
