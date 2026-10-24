import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot013Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root13Candidates0 : List Nat := [5333, 22193, 23027, 24623]
set_option maxRecDepth 8192 in
theorem root13Candidates0_checked : candidates edge 4 53 113 root13Tail1 winner.bound = root13Candidates0 := by
  unfold candidates
  have row : edge 113 = (SortedNatTree.contains FastRows.row113) := by
    change Data.row113.contains = (SortedNatTree.contains FastRows.row113)
    exact FastRows.row113_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root13Child0 : search edge 3 [53, 113] root13Candidates0 166 winner = winner := by decide
theorem root13Step0 : scan edge (search edge 3) 4 [53] 53 root13Tail0 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail1 winner := by
  rw [root13Tail0, scan, if_neg (by decide), root13Candidates0_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail1
    (if 3 ≤ root13Candidates0.length then search edge 3 [53, 113] root13Candidates0 166 winner else winner) = _
  rw [if_pos (by decide), root13Child0]
def root13Candidates1 : List Nat := [1481, 1637, 2099, 16349, 24623]
set_option maxRecDepth 8192 in
theorem root13Candidates1_checked : candidates edge 4 53 197 root13Tail2 winner.bound = root13Candidates1 := by
  unfold candidates
  have row : edge 197 = (SortedNatTree.contains FastRows.row197) := by
    change Data.row197.contains = (SortedNatTree.contains FastRows.row197)
    exact FastRows.row197_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root13Child1 : search edge 3 [53, 197] root13Candidates1 250 winner = winner := by decide
theorem root13Step1 : scan edge (search edge 3) 4 [53] 53 root13Tail1 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail2 winner := by
  rw [root13Tail1, scan, if_neg (by decide), root13Candidates1_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail2
    (if 3 ≤ root13Candidates1.length then search edge 3 [53, 197] root13Candidates1 250 winner else winner) = _
  rw [if_pos (by decide), root13Child1]
def root13Candidates2 : List Nat := [23027]
set_option maxRecDepth 8192 in
theorem root13Candidates2_checked : candidates edge 4 53 269 root13Tail3 winner.bound = root13Candidates2 := by
  unfold candidates
  have row : edge 269 = (SortedNatTree.contains FastRows.row269) := by
    change Data.row269.contains = (SortedNatTree.contains FastRows.row269)
    exact FastRows.row269_checked.symm
  rw [row]
  decide
theorem root13Step2 : scan edge (search edge 3) 4 [53] 53 root13Tail2 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail3 winner := by
  rw [root13Tail2, scan, if_neg (by decide), root13Candidates2_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail3
    (if 3 ≤ root13Candidates2.length then search edge 3 [53, 269] root13Candidates2 322 winner else winner) = _
  rw [if_neg (by decide)]
def root13Candidates3 : List Nat := [1481, 3371, 6791, 11909, 16349]
set_option maxRecDepth 8192 in
theorem root13Candidates3_checked : candidates edge 4 53 353 root13Tail4 winner.bound = root13Candidates3 := by
  unfold candidates
  have row : edge 353 = (SortedNatTree.contains FastRows.row353) := by
    change Data.row353.contains = (SortedNatTree.contains FastRows.row353)
    exact FastRows.row353_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root13Child3 : search edge 3 [53, 353] root13Candidates3 406 winner = winner := by decide
theorem root13Step3 : scan edge (search edge 3) 4 [53] 53 root13Tail3 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail4 winner := by
  rw [root13Tail3, scan, if_neg (by decide), root13Candidates3_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail4
    (if 3 ≤ root13Candidates3.length then search edge 3 [53, 353] root13Candidates3 406 winner else winner) = _
  rw [if_pos (by decide), root13Child3]
def root13Candidates4 : List Nat := [11171, 11927, 21839]
set_option maxRecDepth 8192 in
theorem root13Candidates4_checked : candidates edge 4 53 401 root13Tail5 winner.bound = root13Candidates4 := by
  unfold candidates
  have row : edge 401 = (SortedNatTree.contains FastRows.row401) := by
    change Data.row401.contains = (SortedNatTree.contains FastRows.row401)
    exact FastRows.row401_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root13Child4 : search edge 3 [53, 401] root13Candidates4 454 winner = winner := by decide
theorem root13Step4 : scan edge (search edge 3) 4 [53] 53 root13Tail4 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail5 winner := by
  rw [root13Tail4, scan, if_neg (by decide), root13Candidates4_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail5
    (if 3 ≤ root13Candidates4.length then search edge 3 [53, 401] root13Candidates4 454 winner else winner) = _
  rw [if_pos (by decide), root13Child4]
def root13Candidates5 : List Nat := [5273, 7883, 16349, 23789]
set_option maxRecDepth 8192 in
theorem root13Candidates5_checked : candidates edge 4 53 419 root13Tail6 winner.bound = root13Candidates5 := by
  unfold candidates
  have row : edge 419 = (SortedNatTree.contains FastRows.row419) := by
    change Data.row419.contains = (SortedNatTree.contains FastRows.row419)
    exact FastRows.row419_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root13Child5 : search edge 3 [53, 419] root13Candidates5 472 winner = winner := by decide
theorem root13Step5 : scan edge (search edge 3) 4 [53] 53 root13Tail5 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail6 winner := by
  rw [root13Tail5, scan, if_neg (by decide), root13Candidates5_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail6
    (if 3 ≤ root13Candidates5.length then search edge 3 [53, 419] root13Candidates5 472 winner else winner) = _
  rw [if_pos (by decide), root13Child5]
def root13Candidates6 : List Nat := [2531, 3167, 22511]
set_option maxRecDepth 8192 in
theorem root13Candidates6_checked : candidates edge 4 53 653 root13Tail7 winner.bound = root13Candidates6 := by
  unfold candidates
  have row : edge 653 = (SortedNatTree.contains FastRows.row653) := by
    change Data.row653.contains = (SortedNatTree.contains FastRows.row653)
    exact FastRows.row653_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root13Child6 : search edge 3 [53, 653] root13Candidates6 706 winner = winner := by decide
theorem root13Step6 : scan edge (search edge 3) 4 [53] 53 root13Tail6 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail7 winner := by
  rw [root13Tail6, scan, if_neg (by decide), root13Candidates6_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail7
    (if 3 ≤ root13Candidates6.length then search edge 3 [53, 653] root13Candidates6 706 winner else winner) = _
  rw [if_pos (by decide), root13Child6]
def root13Candidates7 : List Nat := [2663, 5861, 6353]
set_option maxRecDepth 8192 in
theorem root13Candidates7_checked : candidates edge 4 53 1229 root13Tail8 winner.bound = root13Candidates7 := by
  unfold candidates
  have row : edge 1229 = (SortedNatTree.contains FastRows.row1229) := by
    change Data.row1229.contains = (SortedNatTree.contains FastRows.row1229)
    exact FastRows.row1229_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root13Child7 : search edge 3 [53, 1229] root13Candidates7 1282 winner = winner := by decide
theorem root13Step7 : scan edge (search edge 3) 4 [53] 53 root13Tail7 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail8 winner := by
  rw [root13Tail7, scan, if_neg (by decide), root13Candidates7_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail8
    (if 3 ≤ root13Candidates7.length then search edge 3 [53, 1229] root13Candidates7 1282 winner else winner) = _
  rw [if_pos (by decide), root13Child7]
def root13Candidates8 : List Nat := [2663, 3167, 7703, 16061, 19427, 21227]
set_option maxRecDepth 8192 in
theorem root13Candidates8_checked : candidates edge 4 53 1481 root13Tail9 winner.bound = root13Candidates8 := by
  unfold candidates
  have row : edge 1481 = (SortedNatTree.contains FastRows.row1481) := by
    change Data.row1481.contains = (SortedNatTree.contains FastRows.row1481)
    exact FastRows.row1481_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root13Child8 : search edge 3 [53, 1481] root13Candidates8 1534 winner = winner := by decide
theorem root13Step8 : scan edge (search edge 3) 4 [53] 53 root13Tail8 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail9 winner := by
  rw [root13Tail8, scan, if_neg (by decide), root13Candidates8_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail9
    (if 3 ≤ root13Candidates8.length then search edge 3 [53, 1481] root13Candidates8 1534 winner else winner) = _
  rw [if_pos (by decide), root13Child8]
def root13Candidates9 : List Nat := [4649, 7841, 14669]
set_option maxRecDepth 8192 in
theorem root13Candidates9_checked : candidates edge 4 53 1637 root13Tail10 winner.bound = root13Candidates9 := by
  unfold candidates
  have row : edge 1637 = (SortedNatTree.contains FastRows.row1637) := by
    change Data.row1637.contains = (SortedNatTree.contains FastRows.row1637)
    exact FastRows.row1637_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root13Child9 : search edge 3 [53, 1637] root13Candidates9 1690 winner = winner := by decide
theorem root13Step9 : scan edge (search edge 3) 4 [53] 53 root13Tail9 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail10 winner := by
  rw [root13Tail9, scan, if_neg (by decide), root13Candidates9_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail10
    (if 3 ≤ root13Candidates9.length then search edge 3 [53, 1637] root13Candidates9 1690 winner else winner) = _
  rw [if_pos (by decide), root13Child9]
def root13Candidates10 : List Nat := [2069, 6353, 14669]
set_option maxRecDepth 8192 in
theorem root13Candidates10_checked : candidates edge 4 53 1823 root13Tail11 winner.bound = root13Candidates10 := by
  unfold candidates
  have row : edge 1823 = (SortedNatTree.contains FastRows.row1823) := by
    change Data.row1823.contains = (SortedNatTree.contains FastRows.row1823)
    exact FastRows.row1823_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root13Child10 : search edge 3 [53, 1823] root13Candidates10 1876 winner = winner := by decide
theorem root13Step10 : scan edge (search edge 3) 4 [53] 53 root13Tail10 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail11 winner := by
  rw [root13Tail10, scan, if_neg (by decide), root13Candidates10_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail11
    (if 3 ≤ root13Candidates10.length then search edge 3 [53, 1823] root13Candidates10 1876 winner else winner) = _
  rw [if_pos (by decide), root13Child10]
def root13Candidates11 : List Nat := [3011, 4007, 4283, 7307, 7703, 8123, 9311]
set_option maxRecDepth 8192 in
theorem root13Candidates11_checked : candidates edge 4 53 1847 root13Tail12 winner.bound = root13Candidates11 := by
  unfold candidates
  have row : edge 1847 = (SortedNatTree.contains FastRows.row1847) := by
    change Data.row1847.contains = (SortedNatTree.contains FastRows.row1847)
    exact FastRows.row1847_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root13Child11 : search edge 3 [53, 1847] root13Candidates11 1900 winner = winner := by decide
theorem root13Step11 : scan edge (search edge 3) 4 [53] 53 root13Tail11 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail12 winner := by
  rw [root13Tail11, scan, if_neg (by decide), root13Candidates11_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail12
    (if 3 ≤ root13Candidates11.length then search edge 3 [53, 1847] root13Candidates11 1900 winner else winner) = _
  rw [if_pos (by decide), root13Child11]
def root13Candidates12 : List Nat := [16001]
set_option maxRecDepth 8192 in
theorem root13Candidates12_checked : candidates edge 4 53 1997 root13Tail13 winner.bound = root13Candidates12 := by
  unfold candidates
  have row : edge 1997 = (SortedNatTree.contains FastRows.row1997) := by
    change Data.row1997.contains = (SortedNatTree.contains FastRows.row1997)
    exact FastRows.row1997_checked.symm
  rw [row]
  decide
theorem root13Step12 : scan edge (search edge 3) 4 [53] 53 root13Tail12 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail13 winner := by
  rw [root13Tail12, scan, if_neg (by decide), root13Candidates12_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail13
    (if 3 ≤ root13Candidates12.length then search edge 3 [53, 1997] root13Candidates12 2050 winner else winner) = _
  rw [if_neg (by decide)]
def root13Candidates13 : List Nat := [7703]
set_option maxRecDepth 8192 in
theorem root13Candidates13_checked : candidates edge 4 53 2027 root13Tail14 winner.bound = root13Candidates13 := by
  unfold candidates
  have row : edge 2027 = (SortedNatTree.contains FastRows.row2027) := by
    change Data.row2027.contains = (SortedNatTree.contains FastRows.row2027)
    exact FastRows.row2027_checked.symm
  rw [row]
  decide
theorem root13Step13 : scan edge (search edge 3) 4 [53] 53 root13Tail13 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail14 winner := by
  rw [root13Tail13, scan, if_neg (by decide), root13Candidates13_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail14
    (if 3 ≤ root13Candidates13.length then search edge 3 [53, 2027] root13Candidates13 2080 winner else winner) = _
  rw [if_neg (by decide)]
def root13Candidates14 : List Nat := [4637, 14207]
set_option maxRecDepth 8192 in
theorem root13Candidates14_checked : candidates edge 4 53 2069 root13Tail15 winner.bound = root13Candidates14 := by
  unfold candidates
  have row : edge 2069 = (SortedNatTree.contains FastRows.row2069) := by
    change Data.row2069.contains = (SortedNatTree.contains FastRows.row2069)
    exact FastRows.row2069_checked.symm
  rw [row]
  decide
theorem root13Step14 : scan edge (search edge 3) 4 [53] 53 root13Tail14 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail15 winner := by
  rw [root13Tail14, scan, if_neg (by decide), root13Candidates14_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail15
    (if 3 ≤ root13Candidates14.length then search edge 3 [53, 2069] root13Candidates14 2122 winner else winner) = _
  rw [if_neg (by decide)]
def root13Candidates15 : List Nat := [7841, 10343]
set_option maxRecDepth 8192 in
theorem root13Candidates15_checked : candidates edge 4 53 2099 root13Tail16 winner.bound = root13Candidates15 := by
  unfold candidates
  have row : edge 2099 = (SortedNatTree.contains FastRows.row2099) := by
    change Data.row2099.contains = (SortedNatTree.contains FastRows.row2099)
    exact FastRows.row2099_checked.symm
  rw [row]
  decide
theorem root13Step15 : scan edge (search edge 3) 4 [53] 53 root13Tail15 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail16 winner := by
  rw [root13Tail15, scan, if_neg (by decide), root13Candidates15_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail16
    (if 3 ≤ root13Candidates15.length then search edge 3 [53, 2099] root13Candidates15 2152 winner else winner) = _
  rw [if_neg (by decide)]
def root13Candidates16 : List Nat := [2663, 5273, 14207]
set_option maxRecDepth 8192 in
theorem root13Candidates16_checked : candidates edge 4 53 2153 root13Tail17 winner.bound = root13Candidates16 := by
  unfold candidates
  have row : edge 2153 = (SortedNatTree.contains FastRows.row2153) := by
    change Data.row2153.contains = (SortedNatTree.contains FastRows.row2153)
    exact FastRows.row2153_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root13Child16 : search edge 3 [53, 2153] root13Candidates16 2206 winner = winner := by decide
theorem root13Step16 : scan edge (search edge 3) 4 [53] 53 root13Tail16 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail17 winner := by
  rw [root13Tail16, scan, if_neg (by decide), root13Candidates16_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail17
    (if 3 ≤ root13Candidates16.length then search edge 3 [53, 2153] root13Candidates16 2206 winner else winner) = _
  rw [if_pos (by decide), root13Child16]
def root13Candidates17 : List Nat := [4637, 6791, 15551, 21839]
set_option maxRecDepth 8192 in
theorem root13Candidates17_checked : candidates edge 4 53 2267 root13Tail18 winner.bound = root13Candidates17 := by
  unfold candidates
  have row : edge 2267 = (SortedNatTree.contains FastRows.row2267) := by
    change Data.row2267.contains = (SortedNatTree.contains FastRows.row2267)
    exact FastRows.row2267_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root13Child17 : search edge 3 [53, 2267] root13Candidates17 2320 winner = winner := by decide
theorem root13Step17 : scan edge (search edge 3) 4 [53] 53 root13Tail17 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail18 winner := by
  rw [root13Tail17, scan, if_neg (by decide), root13Candidates17_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail18
    (if 3 ≤ root13Candidates17.length then search edge 3 [53, 2267] root13Candidates17 2320 winner else winner) = _
  rw [if_pos (by decide), root13Child17]
def root13Candidates18 : List Nat := [6311, 19949]
set_option maxRecDepth 8192 in
theorem root13Candidates18_checked : candidates edge 4 53 2333 root13Tail19 winner.bound = root13Candidates18 := by
  unfold candidates
  have row : edge 2333 = (SortedNatTree.contains FastRows.row2333) := by
    change Data.row2333.contains = (SortedNatTree.contains FastRows.row2333)
    exact FastRows.row2333_checked.symm
  rw [row]
  decide
theorem root13Step18 : scan edge (search edge 3) 4 [53] 53 root13Tail18 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail19 winner := by
  rw [root13Tail18, scan, if_neg (by decide), root13Candidates18_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail19
    (if 3 ≤ root13Candidates18.length then search edge 3 [53, 2333] root13Candidates18 2386 winner else winner) = _
  rw [if_neg (by decide)]
def root13Candidates19 : List Nat := [3011, 4019, 6323, 8123, 12911, 13463, 21179]
set_option maxRecDepth 8192 in
theorem root13Candidates19_checked : candidates edge 4 53 2447 root13Tail20 winner.bound = root13Candidates19 := by
  unfold candidates
  have row : edge 2447 = (SortedNatTree.contains FastRows.row2447) := by
    change Data.row2447.contains = (SortedNatTree.contains FastRows.row2447)
    exact FastRows.row2447_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root13Child19 : search edge 3 [53, 2447] root13Candidates19 2500 winner = winner := by decide
theorem root13Step19 : scan edge (search edge 3) 4 [53] 53 root13Tail19 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail20 winner := by
  rw [root13Tail19, scan, if_neg (by decide), root13Candidates19_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail20
    (if 3 ≤ root13Candidates19.length then search edge 3 [53, 2447] root13Candidates19 2500 winner else winner) = _
  rw [if_pos (by decide), root13Child19]
def root13Candidates20 : List Nat := [5273, 7883, 8123]
set_option maxRecDepth 8192 in
theorem root13Candidates20_checked : candidates edge 4 53 2531 root13Tail21 winner.bound = root13Candidates20 := by
  unfold candidates
  have row : edge 2531 = (SortedNatTree.contains FastRows.row2531) := by
    change Data.row2531.contains = (SortedNatTree.contains FastRows.row2531)
    exact FastRows.row2531_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root13Child20 : search edge 3 [53, 2531] root13Candidates20 2584 winner = winner := by decide
theorem root13Step20 : scan edge (search edge 3) 4 [53] 53 root13Tail20 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail21 winner := by
  rw [root13Tail20, scan, if_neg (by decide), root13Candidates20_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail21
    (if 3 ≤ root13Candidates20.length then search edge 3 [53, 2531] root13Candidates20 2584 winner else winner) = _
  rw [if_pos (by decide), root13Child20]
def root13Candidates21 : List Nat := []
set_option maxRecDepth 8192 in
theorem root13Candidates21_checked : candidates edge 4 53 2621 root13Tail22 winner.bound = root13Candidates21 := by
  unfold candidates
  have row : edge 2621 = (SortedNatTree.contains FastRows.row2621) := by
    change Data.row2621.contains = (SortedNatTree.contains FastRows.row2621)
    exact FastRows.row2621_checked.symm
  rw [row]
  decide
theorem root13Step21 : scan edge (search edge 3) 4 [53] 53 root13Tail21 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail22 winner := by
  rw [root13Tail21, scan, if_neg (by decide), root13Candidates21_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail22
    (if 3 ≤ root13Candidates21.length then search edge 3 [53, 2621] root13Candidates21 2674 winner else winner) = _
  rw [if_neg (by decide)]
def root13Candidates22 : List Nat := [19421]
set_option maxRecDepth 8192 in
theorem root13Candidates22_checked : candidates edge 4 53 2663 root13Tail23 winner.bound = root13Candidates22 := by
  unfold candidates
  have row : edge 2663 = (SortedNatTree.contains FastRows.row2663) := by
    change Data.row2663.contains = (SortedNatTree.contains FastRows.row2663)
    exact FastRows.row2663_checked.symm
  rw [row]
  decide
theorem root13Step22 : scan edge (search edge 3) 4 [53] 53 root13Tail22 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail23 winner := by
  rw [root13Tail22, scan, if_neg (by decide), root13Candidates22_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail23
    (if 3 ≤ root13Candidates22.length then search edge 3 [53, 2663] root13Candidates22 2716 winner else winner) = _
  rw [if_neg (by decide)]
def root13Candidates23 : List Nat := [6353, 7883, 14207]
set_option maxRecDepth 8192 in
theorem root13Candidates23_checked : candidates edge 4 53 3011 root13Tail24 winner.bound = root13Candidates23 := by
  unfold candidates
  have row : edge 3011 = (SortedNatTree.contains FastRows.row3011) := by
    change Data.row3011.contains = (SortedNatTree.contains FastRows.row3011)
    exact FastRows.row3011_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root13Child23 : search edge 3 [53, 3011] root13Candidates23 3064 winner = winner := by decide
theorem root13Step23 : scan edge (search edge 3) 4 [53] 53 root13Tail23 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail24 winner := by
  rw [root13Tail23, scan, if_neg (by decide), root13Candidates23_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail24
    (if 3 ≤ root13Candidates23.length then search edge 3 [53, 3011] root13Candidates23 3064 winner else winner) = _
  rw [if_pos (by decide), root13Child23]
def root13Candidates24 : List Nat := [13967, 19421]
set_option maxRecDepth 8192 in
theorem root13Candidates24_checked : candidates edge 4 53 3167 root13Tail25 winner.bound = root13Candidates24 := by
  unfold candidates
  have row : edge 3167 = (SortedNatTree.contains FastRows.row3167) := by
    change Data.row3167.contains = (SortedNatTree.contains FastRows.row3167)
    exact FastRows.row3167_checked.symm
  rw [row]
  decide
theorem root13Step24 : scan edge (search edge 3) 4 [53] 53 root13Tail24 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail25 winner := by
  rw [root13Tail24, scan, if_neg (by decide), root13Candidates24_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail25
    (if 3 ≤ root13Candidates24.length then search edge 3 [53, 3167] root13Candidates24 3220 winner else winner) = _
  rw [if_neg (by decide)]
def root13Candidates25 : List Nat := [4649]
set_option maxRecDepth 8192 in
theorem root13Candidates25_checked : candidates edge 4 53 3257 root13Tail26 winner.bound = root13Candidates25 := by
  unfold candidates
  have row : edge 3257 = (SortedNatTree.contains FastRows.row3257) := by
    change Data.row3257.contains = (SortedNatTree.contains FastRows.row3257)
    exact FastRows.row3257_checked.symm
  rw [row]
  decide
theorem root13Step25 : scan edge (search edge 3) 4 [53] 53 root13Tail25 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail26 winner := by
  rw [root13Tail25, scan, if_neg (by decide), root13Candidates25_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail26
    (if 3 ≤ root13Candidates25.length then search edge 3 [53, 3257] root13Candidates25 3310 winner else winner) = _
  rw [if_neg (by decide)]
def root13Candidates26 : List Nat := [4649, 4799, 9677]
set_option maxRecDepth 8192 in
theorem root13Candidates26_checked : candidates edge 4 53 3371 root13Tail27 winner.bound = root13Candidates26 := by
  unfold candidates
  have row : edge 3371 = (SortedNatTree.contains FastRows.row3371) := by
    change Data.row3371.contains = (SortedNatTree.contains FastRows.row3371)
    exact FastRows.row3371_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root13Child26 : search edge 3 [53, 3371] root13Candidates26 3424 winner = winner := by decide
theorem root13Step26 : scan edge (search edge 3) 4 [53] 53 root13Tail26 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail27 winner := by
  rw [root13Tail26, scan, if_neg (by decide), root13Candidates26_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail27
    (if 3 ≤ root13Candidates26.length then search edge 3 [53, 3371] root13Candidates26 3424 winner else winner) = _
  rw [if_pos (by decide), root13Child26]
def root13Candidates27 : List Nat := [4091, 5303]
set_option maxRecDepth 8192 in
theorem root13Candidates27_checked : candidates edge 4 53 3581 root13Tail28 winner.bound = root13Candidates27 := by
  unfold candidates
  have row : edge 3581 = (SortedNatTree.contains FastRows.row3581) := by
    change Data.row3581.contains = (SortedNatTree.contains FastRows.row3581)
    exact FastRows.row3581_checked.symm
  rw [row]
  decide
theorem root13Step27 : scan edge (search edge 3) 4 [53] 53 root13Tail27 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail28 winner := by
  rw [root13Tail27, scan, if_neg (by decide), root13Candidates27_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail28
    (if 3 ≤ root13Candidates27.length then search edge 3 [53, 3581] root13Candidates27 3634 winner else winner) = _
  rw [if_neg (by decide)]
def root13Candidates28 : List Nat := [6311]
set_option maxRecDepth 8192 in
theorem root13Candidates28_checked : candidates edge 4 53 3593 root13Tail29 winner.bound = root13Candidates28 := by
  unfold candidates
  have row : edge 3593 = (SortedNatTree.contains FastRows.row3593) := by
    change Data.row3593.contains = (SortedNatTree.contains FastRows.row3593)
    exact FastRows.row3593_checked.symm
  rw [row]
  decide
theorem root13Step28 : scan edge (search edge 3) 4 [53] 53 root13Tail28 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail29 winner := by
  rw [root13Tail28, scan, if_neg (by decide), root13Candidates28_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail29
    (if 3 ≤ root13Candidates28.length then search edge 3 [53, 3593] root13Candidates28 3646 winner else winner) = _
  rw [if_neg (by decide)]
def root13Candidates29 : List Nat := [4091, 6353, 7853]
set_option maxRecDepth 8192 in
theorem root13Candidates29_checked : candidates edge 4 53 4007 root13Tail30 winner.bound = root13Candidates29 := by
  unfold candidates
  have row : edge 4007 = (SortedNatTree.contains FastRows.row4007) := by
    change Data.row4007.contains = (SortedNatTree.contains FastRows.row4007)
    exact FastRows.row4007_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root13Child29 : search edge 3 [53, 4007] root13Candidates29 4060 winner = winner := by decide
theorem root13Step29 : scan edge (search edge 3) 4 [53] 53 root13Tail29 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail30 winner := by
  rw [root13Tail29, scan, if_neg (by decide), root13Candidates29_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail30
    (if 3 ≤ root13Candidates29.length then search edge 3 [53, 4007] root13Candidates29 4060 winner else winner) = _
  rw [if_pos (by decide), root13Child29]
def root13Candidates30 : List Nat := [6791, 17903]
set_option maxRecDepth 8192 in
theorem root13Candidates30_checked : candidates edge 4 53 4019 root13Tail31 winner.bound = root13Candidates30 := by
  unfold candidates
  have row : edge 4019 = (SortedNatTree.contains FastRows.row4019) := by
    change Data.row4019.contains = (SortedNatTree.contains FastRows.row4019)
    exact FastRows.row4019_checked.symm
  rw [row]
  decide
theorem root13Step30 : scan edge (search edge 3) 4 [53] 53 root13Tail30 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail31 winner := by
  rw [root13Tail30, scan, if_neg (by decide), root13Candidates30_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail31
    (if 3 ≤ root13Candidates30.length then search edge 3 [53, 4019] root13Candidates30 4072 winner else winner) = _
  rw [if_neg (by decide)]
def root13Candidates31 : List Nat := [4283]
set_option maxRecDepth 8192 in
theorem root13Candidates31_checked : candidates edge 4 53 4091 root13Tail32 winner.bound = root13Candidates31 := by
  unfold candidates
  have row : edge 4091 = (SortedNatTree.contains FastRows.row4091) := by
    change Data.row4091.contains = (SortedNatTree.contains FastRows.row4091)
    exact FastRows.row4091_checked.symm
  rw [row]
  decide
theorem root13Step31 : scan edge (search edge 3) 4 [53] 53 root13Tail31 winner =
    scan edge (search edge 3) 4 [53] 53 root13Tail32 winner := by
  rw [root13Tail31, scan, if_neg (by decide), root13Candidates31_checked]
  change scan edge (search edge 3) 4 [53] 53 root13Tail32
    (if 3 ≤ root13Candidates31.length then search edge 3 [53, 4091] root13Candidates31 4144 winner else winner) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
