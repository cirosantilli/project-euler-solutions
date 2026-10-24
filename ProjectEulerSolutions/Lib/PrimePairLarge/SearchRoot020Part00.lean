import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot020Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root20Candidates0 : List Nat := [719, 2729, 5591, 8093]
set_option maxRecDepth 8192 in
theorem root20Candidates0_checked : candidates edge 4 83 227 root20Tail1 winner.bound = root20Candidates0 := by
  unfold candidates
  have row : edge 227 = (SortedNatTree.contains FastRows.row227) := by
    change Data.row227.contains = (SortedNatTree.contains FastRows.row227)
    exact FastRows.row227_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root20Child0 : search edge 3 [83, 227] root20Candidates0 310 winner = winner := by decide
theorem root20Step0 : scan edge (search edge 3) 4 [83] 83 root20Tail0 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail1 winner := by
  rw [root20Tail0, scan, if_neg (by decide), root20Candidates0_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail1
    (if 3 ≤ root20Candidates0.length then search edge 3 [83, 227] root20Candidates0 310 winner else winner) = _
  rw [if_pos (by decide), root20Child0]
def root20Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root20Candidates1_checked : candidates edge 4 83 311 root20Tail2 winner.bound = root20Candidates1 := by
  unfold candidates
  have row : edge 311 = (SortedNatTree.contains FastRows.row311) := by
    change Data.row311.contains = (SortedNatTree.contains FastRows.row311)
    exact FastRows.row311_checked.symm
  rw [row]
  decide
theorem root20Step1 : scan edge (search edge 3) 4 [83] 83 root20Tail1 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail2 winner := by
  rw [root20Tail1, scan, if_neg (by decide), root20Candidates1_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail2
    (if 3 ≤ root20Candidates1.length then search edge 3 [83, 311] root20Candidates1 394 winner else winner) = _
  rw [if_neg (by decide)]
def root20Candidates2 : List Nat := [701, 1811, 2693, 5897, 7283, 19559, 24923]
set_option maxRecDepth 8192 in
theorem root20Candidates2_checked : candidates edge 4 83 443 root20Tail3 winner.bound = root20Candidates2 := by
  unfold candidates
  have row : edge 443 = (SortedNatTree.contains FastRows.row443) := by
    change Data.row443.contains = (SortedNatTree.contains FastRows.row443)
    exact FastRows.row443_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root20Child2 : search edge 3 [83, 443] root20Candidates2 526 winner = winner := by decide
theorem root20Step2 : scan edge (search edge 3) 4 [83] 83 root20Tail2 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail3 winner := by
  rw [root20Tail2, scan, if_neg (by decide), root20Candidates2_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail3
    (if 3 ≤ root20Candidates2.length then search edge 3 [83, 443] root20Candidates2 526 winner else winner) = _
  rw [if_pos (by decide), root20Child2]
def root20Candidates3 : List Nat := [563, 3659, 6491, 7583]
set_option maxRecDepth 8192 in
theorem root20Candidates3_checked : candidates edge 4 83 449 root20Tail4 winner.bound = root20Candidates3 := by
  unfold candidates
  have row : edge 449 = (SortedNatTree.contains FastRows.row449) := by
    change Data.row449.contains = (SortedNatTree.contains FastRows.row449)
    exact FastRows.row449_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root20Child3 : search edge 3 [83, 449] root20Candidates3 532 winner = winner := by decide
theorem root20Step3 : scan edge (search edge 3) 4 [83] 83 root20Tail3 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail4 winner := by
  rw [root20Tail3, scan, if_neg (by decide), root20Candidates3_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail4
    (if 3 ≤ root20Candidates3.length then search edge 3 [83, 449] root20Candidates3 532 winner else winner) = _
  rw [if_pos (by decide), root20Child3]
def root20Candidates4 : List Nat := [5717, 19559, 25013]
set_option maxRecDepth 8192 in
theorem root20Candidates4_checked : candidates edge 4 83 563 root20Tail5 winner.bound = root20Candidates4 := by
  unfold candidates
  have row : edge 563 = (SortedNatTree.contains FastRows.row563) := by
    change Data.row563.contains = (SortedNatTree.contains FastRows.row563)
    exact FastRows.row563_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root20Child4 : search edge 3 [83, 563] root20Candidates4 646 winner = winner := by decide
theorem root20Step4 : scan edge (search edge 3) 4 [83] 83 root20Tail4 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail5 winner := by
  rw [root20Tail4, scan, if_neg (by decide), root20Candidates4_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail5
    (if 3 ≤ root20Candidates4.length then search edge 3 [83, 563] root20Candidates4 646 winner else winner) = _
  rw [if_pos (by decide), root20Child4]
def root20Candidates5 : List Nat := [1553, 8171, 9473, 12197]
set_option maxRecDepth 8192 in
theorem root20Candidates5_checked : candidates edge 4 83 701 root20Tail6 winner.bound = root20Candidates5 := by
  unfold candidates
  have row : edge 701 = (SortedNatTree.contains FastRows.row701) := by
    change Data.row701.contains = (SortedNatTree.contains FastRows.row701)
    exact FastRows.row701_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root20Child5 : search edge 3 [83, 701] root20Candidates5 784 winner = winner := by decide
theorem root20Step5 : scan edge (search edge 3) 4 [83] 83 root20Tail5 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail6 winner := by
  rw [root20Tail5, scan, if_neg (by decide), root20Candidates5_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail6
    (if 3 ≤ root20Candidates5.length then search edge 3 [83, 701] root20Candidates5 784 winner else winner) = _
  rw [if_pos (by decide), root20Child5]
def root20Candidates6 : List Nat := [773, 911, 5039, 7079, 7283, 12119]
set_option maxRecDepth 8192 in
theorem root20Candidates6_checked : candidates edge 4 83 719 root20Tail7 winner.bound = root20Candidates6 := by
  unfold candidates
  have row : edge 719 = (SortedNatTree.contains FastRows.row719) := by
    change Data.row719.contains = (SortedNatTree.contains FastRows.row719)
    exact FastRows.row719_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root20Child6 : search edge 3 [83, 719] root20Candidates6 802 winner = winner := by decide
theorem root20Step6 : scan edge (search edge 3) 4 [83] 83 root20Tail6 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail7 winner := by
  rw [root20Tail6, scan, if_neg (by decide), root20Candidates6_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail7
    (if 3 ≤ root20Candidates6.length then search edge 3 [83, 719] root20Candidates6 802 winner else winner) = _
  rw [if_pos (by decide), root20Child6]
def root20Candidates7 : List Nat := [2063, 2339, 2591, 11909, 12119, 16619]
set_option maxRecDepth 8192 in
theorem root20Candidates7_checked : candidates edge 4 83 773 root20Tail8 winner.bound = root20Candidates7 := by
  unfold candidates
  have row : edge 773 = (SortedNatTree.contains FastRows.row773) := by
    change Data.row773.contains = (SortedNatTree.contains FastRows.row773)
    exact FastRows.row773_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root20Child7 : search edge 3 [83, 773] root20Candidates7 856 winner = winner := by decide
theorem root20Step7 : scan edge (search edge 3) 4 [83] 83 root20Tail7 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail8 winner := by
  rw [root20Tail7, scan, if_neg (by decide), root20Candidates7_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail8
    (if 3 ≤ root20Candidates7.length then search edge 3 [83, 773] root20Candidates7 856 winner else winner) = _
  rw [if_pos (by decide), root20Child7]
def root20Candidates8 : List Nat := [4643, 22679]
set_option maxRecDepth 8192 in
theorem root20Candidates8_checked : candidates edge 4 83 911 root20Tail9 winner.bound = root20Candidates8 := by
  unfold candidates
  have row : edge 911 = (SortedNatTree.contains FastRows.row911) := by
    change Data.row911.contains = (SortedNatTree.contains FastRows.row911)
    exact FastRows.row911_checked.symm
  rw [row]
  decide
theorem root20Step8 : scan edge (search edge 3) 4 [83] 83 root20Tail8 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail9 winner := by
  rw [root20Tail8, scan, if_neg (by decide), root20Candidates8_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail9
    (if 3 ≤ root20Candidates8.length then search edge 3 [83, 911] root20Candidates8 994 winner else winner) = _
  rw [if_neg (by decide)]
def root20Candidates9 : List Nat := [1811, 4643, 10691, 10781, 22871]
set_option maxRecDepth 8192 in
theorem root20Candidates9_checked : candidates edge 4 83 1031 root20Tail10 winner.bound = root20Candidates9 := by
  unfold candidates
  have row : edge 1031 = (SortedNatTree.contains FastRows.row1031) := by
    change Data.row1031.contains = (SortedNatTree.contains FastRows.row1031)
    exact FastRows.row1031_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root20Child9 : search edge 3 [83, 1031] root20Candidates9 1114 winner = winner := by decide
theorem root20Step9 : scan edge (search edge 3) 4 [83] 83 root20Tail9 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail10 winner := by
  rw [root20Tail9, scan, if_neg (by decide), root20Candidates9_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail10
    (if 3 ≤ root20Candidates9.length then search edge 3 [83, 1031] root20Candidates9 1114 winner else winner) = _
  rw [if_pos (by decide), root20Child9]
def root20Candidates10 : List Nat := [2969, 7283, 9551]
set_option maxRecDepth 8192 in
theorem root20Candidates10_checked : candidates edge 4 83 1301 root20Tail11 winner.bound = root20Candidates10 := by
  unfold candidates
  have row : edge 1301 = (SortedNatTree.contains FastRows.row1301) := by
    change Data.row1301.contains = (SortedNatTree.contains FastRows.row1301)
    exact FastRows.row1301_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root20Child10 : search edge 3 [83, 1301] root20Candidates10 1384 winner = winner := by decide
theorem root20Step10 : scan edge (search edge 3) 4 [83] 83 root20Tail10 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail11 winner := by
  rw [root20Tail10, scan, if_neg (by decide), root20Candidates10_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail11
    (if 3 ≤ root20Candidates10.length then search edge 3 [83, 1301] root20Candidates10 1384 winner else winner) = _
  rw [if_pos (by decide), root20Child10]
def root20Candidates11 : List Nat := [2591, 6491]
set_option maxRecDepth 8192 in
theorem root20Candidates11_checked : candidates edge 4 83 1373 root20Tail12 winner.bound = root20Candidates11 := by
  unfold candidates
  have row : edge 1373 = (SortedNatTree.contains FastRows.row1373) := by
    change Data.row1373.contains = (SortedNatTree.contains FastRows.row1373)
    exact FastRows.row1373_checked.symm
  rw [row]
  decide
theorem root20Step11 : scan edge (search edge 3) 4 [83] 83 root20Tail11 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail12 winner := by
  rw [root20Tail11, scan, if_neg (by decide), root20Candidates11_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail12
    (if 3 ≤ root20Candidates11.length then search edge 3 [83, 1373] root20Candidates11 1456 winner else winner) = _
  rw [if_neg (by decide)]
def root20Candidates12 : List Nat := [4643, 9473, 12713, 19319]
set_option maxRecDepth 8192 in
theorem root20Candidates12_checked : candidates edge 4 83 1409 root20Tail13 winner.bound = root20Candidates12 := by
  unfold candidates
  have row : edge 1409 = (SortedNatTree.contains FastRows.row1409) := by
    change Data.row1409.contains = (SortedNatTree.contains FastRows.row1409)
    exact FastRows.row1409_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root20Child12 : search edge 3 [83, 1409] root20Candidates12 1492 winner = winner := by decide
theorem root20Step12 : scan edge (search edge 3) 4 [83] 83 root20Tail12 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail13 winner := by
  rw [root20Tail12, scan, if_neg (by decide), root20Candidates12_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail13
    (if 3 ≤ root20Candidates12.length then search edge 3 [83, 1409] root20Candidates12 1492 winner else winner) = _
  rw [if_pos (by decide), root20Child12]
def root20Candidates13 : List Nat := [1697, 1811, 2687, 6203]
set_option maxRecDepth 8192 in
theorem root20Candidates13_checked : candidates edge 4 83 1553 root20Tail14 winner.bound = root20Candidates13 := by
  unfold candidates
  have row : edge 1553 = (SortedNatTree.contains FastRows.row1553) := by
    change Data.row1553.contains = (SortedNatTree.contains FastRows.row1553)
    exact FastRows.row1553_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root20Child13 : search edge 3 [83, 1553] root20Candidates13 1636 winner = winner := by decide
theorem root20Step13 : scan edge (search edge 3) 4 [83] 83 root20Tail13 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail14 winner := by
  rw [root20Tail13, scan, if_neg (by decide), root20Candidates13_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail14
    (if 3 ≤ root20Candidates13.length then search edge 3 [83, 1553] root20Candidates13 1636 winner else winner) = _
  rw [if_pos (by decide), root20Child13]
def root20Candidates14 : List Nat := [2003, 2339, 5987, 11447, 12119]
set_option maxRecDepth 8192 in
theorem root20Candidates14_checked : candidates edge 4 83 1619 root20Tail15 winner.bound = root20Candidates14 := by
  unfold candidates
  have row : edge 1619 = (SortedNatTree.contains FastRows.row1619) := by
    change Data.row1619.contains = (SortedNatTree.contains FastRows.row1619)
    exact FastRows.row1619_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root20Child14 : search edge 3 [83, 1619] root20Candidates14 1702 winner = winner := by decide
theorem root20Step14 : scan edge (search edge 3) 4 [83] 83 root20Tail14 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail15 winner := by
  rw [root20Tail14, scan, if_neg (by decide), root20Candidates14_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail15
    (if 3 ≤ root20Candidates14.length then search edge 3 [83, 1619] root20Candidates14 1702 winner else winner) = _
  rw [if_pos (by decide), root20Child14]
def root20Candidates15 : List Nat := [9887]
set_option maxRecDepth 8192 in
theorem root20Candidates15_checked : candidates edge 4 83 1697 root20Tail16 winner.bound = root20Candidates15 := by
  unfold candidates
  have row : edge 1697 = (SortedNatTree.contains FastRows.row1697) := by
    change Data.row1697.contains = (SortedNatTree.contains FastRows.row1697)
    exact FastRows.row1697_checked.symm
  rw [row]
  decide
theorem root20Step15 : scan edge (search edge 3) 4 [83] 83 root20Tail15 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail16 winner := by
  rw [root20Tail15, scan, if_neg (by decide), root20Candidates15_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail16
    (if 3 ≤ root20Candidates15.length then search edge 3 [83, 1697] root20Candidates15 1780 winner else winner) = _
  rw [if_neg (by decide)]
def root20Candidates16 : List Nat := [5039, 10691, 16619]
set_option maxRecDepth 8192 in
theorem root20Candidates16_checked : candidates edge 4 83 1811 root20Tail17 winner.bound = root20Candidates16 := by
  unfold candidates
  have row : edge 1811 = (SortedNatTree.contains FastRows.row1811) := by
    change Data.row1811.contains = (SortedNatTree.contains FastRows.row1811)
    exact FastRows.row1811_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root20Child16 : search edge 3 [83, 1811] root20Candidates16 1894 winner = winner := by decide
theorem root20Step16 : scan edge (search edge 3) 4 [83] 83 root20Tail16 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail17 winner := by
  rw [root20Tail16, scan, if_neg (by decide), root20Candidates16_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail17
    (if 3 ≤ root20Candidates16.length then search edge 3 [83, 1811] root20Candidates16 1894 winner else winner) = _
  rw [if_pos (by decide), root20Child16]
def root20Candidates17 : List Nat := [2633, 3491, 9539, 10781]
set_option maxRecDepth 8192 in
theorem root20Candidates17_checked : candidates edge 4 83 1889 root20Tail18 winner.bound = root20Candidates17 := by
  unfold candidates
  have row : edge 1889 = (SortedNatTree.contains FastRows.row1889) := by
    change Data.row1889.contains = (SortedNatTree.contains FastRows.row1889)
    exact FastRows.row1889_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root20Child17 : search edge 3 [83, 1889] root20Candidates17 1972 winner = winner := by decide
theorem root20Step17 : scan edge (search edge 3) 4 [83] 83 root20Tail17 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail18 winner := by
  rw [root20Tail17, scan, if_neg (by decide), root20Candidates17_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail18
    (if 3 ≤ root20Candidates17.length then search edge 3 [83, 1889] root20Candidates17 1972 winner else winner) = _
  rw [if_pos (by decide), root20Child17]
def root20Candidates18 : List Nat := [3557, 5039, 8171, 12713, 20693]
set_option maxRecDepth 8192 in
theorem root20Candidates18_checked : candidates edge 4 83 2003 root20Tail19 winner.bound = root20Candidates18 := by
  unfold candidates
  have row : edge 2003 = (SortedNatTree.contains FastRows.row2003) := by
    change Data.row2003.contains = (SortedNatTree.contains FastRows.row2003)
    exact FastRows.row2003_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root20Child18 : search edge 3 [83, 2003] root20Candidates18 2086 winner = winner := by decide
theorem root20Step18 : scan edge (search edge 3) 4 [83] 83 root20Tail18 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail19 winner := by
  rw [root20Tail18, scan, if_neg (by decide), root20Candidates18_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail19
    (if 3 ≤ root20Candidates18.length then search edge 3 [83, 2003] root20Candidates18 2086 winner else winner) = _
  rw [if_pos (by decide), root20Child18]
def root20Candidates19 : List Nat := [2411, 19391]
set_option maxRecDepth 8192 in
theorem root20Candidates19_checked : candidates edge 4 83 2063 root20Tail20 winner.bound = root20Candidates19 := by
  unfold candidates
  have row : edge 2063 = (SortedNatTree.contains FastRows.row2063) := by
    change Data.row2063.contains = (SortedNatTree.contains FastRows.row2063)
    exact FastRows.row2063_checked.symm
  rw [row]
  decide
theorem root20Step19 : scan edge (search edge 3) 4 [83] 83 root20Tail19 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail20 winner := by
  rw [root20Tail19, scan, if_neg (by decide), root20Candidates19_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail20
    (if 3 ≤ root20Candidates19.length then search edge 3 [83, 2063] root20Candidates19 2146 winner else winner) = _
  rw [if_neg (by decide)]
def root20Candidates20 : List Nat := [2729, 3347, 5039, 7307, 19319, 21503, 22679]
set_option maxRecDepth 8192 in
theorem root20Candidates20_checked : candidates edge 4 83 2339 root20Tail21 winner.bound = root20Candidates20 := by
  unfold candidates
  have row : edge 2339 = (SortedNatTree.contains FastRows.row2339) := by
    change Data.row2339.contains = (SortedNatTree.contains FastRows.row2339)
    exact FastRows.row2339_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root20Child20 : search edge 3 [83, 2339] root20Candidates20 2422 winner = winner := by decide
theorem root20Step20 : scan edge (search edge 3) 4 [83] 83 root20Tail20 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail21 winner := by
  rw [root20Tail20, scan, if_neg (by decide), root20Candidates20_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail21
    (if 3 ≤ root20Candidates20.length then search edge 3 [83, 2339] root20Candidates20 2422 winner else winner) = _
  rw [if_pos (by decide), root20Child20]
def root20Candidates21 : List Nat := []
set_option maxRecDepth 8192 in
theorem root20Candidates21_checked : candidates edge 4 83 2411 root20Tail22 winner.bound = root20Candidates21 := by
  unfold candidates
  have row : edge 2411 = (SortedNatTree.contains FastRows.row2411) := by
    change Data.row2411.contains = (SortedNatTree.contains FastRows.row2411)
    exact FastRows.row2411_checked.symm
  rw [row]
  decide
theorem root20Step21 : scan edge (search edge 3) 4 [83] 83 root20Tail21 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail22 winner := by
  rw [root20Tail21, scan, if_neg (by decide), root20Candidates21_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail22
    (if 3 ≤ root20Candidates21.length then search edge 3 [83, 2411] root20Candidates21 2494 winner else winner) = _
  rw [if_neg (by decide)]
def root20Candidates22 : List Nat := [2633, 13457]
set_option maxRecDepth 8192 in
theorem root20Candidates22_checked : candidates edge 4 83 2591 root20Tail23 winner.bound = root20Candidates22 := by
  unfold candidates
  have row : edge 2591 = (SortedNatTree.contains FastRows.row2591) := by
    change Data.row2591.contains = (SortedNatTree.contains FastRows.row2591)
    exact FastRows.row2591_checked.symm
  rw [row]
  decide
theorem root20Step22 : scan edge (search edge 3) 4 [83] 83 root20Tail22 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail23 winner := by
  rw [root20Tail22, scan, if_neg (by decide), root20Candidates22_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail23
    (if 3 ≤ root20Candidates22.length then search edge 3 [83, 2591] root20Candidates22 2674 winner else winner) = _
  rw [if_neg (by decide)]
def root20Candidates23 : List Nat := [7079, 13763, 20051]
set_option maxRecDepth 8192 in
theorem root20Candidates23_checked : candidates edge 4 83 2633 root20Tail24 winner.bound = root20Candidates23 := by
  unfold candidates
  have row : edge 2633 = (SortedNatTree.contains FastRows.row2633) := by
    change Data.row2633.contains = (SortedNatTree.contains FastRows.row2633)
    exact FastRows.row2633_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root20Child23 : search edge 3 [83, 2633] root20Candidates23 2716 winner = winner := by decide
theorem root20Step23 : scan edge (search edge 3) 4 [83] 83 root20Tail23 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail24 winner := by
  rw [root20Tail23, scan, if_neg (by decide), root20Candidates23_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail24
    (if 3 ≤ root20Candidates23.length then search edge 3 [83, 2633] root20Candidates23 2716 winner else winner) = _
  rw [if_pos (by decide), root20Child23]
def root20Candidates24 : List Nat := [5897, 9551, 23201]
set_option maxRecDepth 8192 in
theorem root20Candidates24_checked : candidates edge 4 83 2687 root20Tail25 winner.bound = root20Candidates24 := by
  unfold candidates
  have row : edge 2687 = (SortedNatTree.contains FastRows.row2687) := by
    change Data.row2687.contains = (SortedNatTree.contains FastRows.row2687)
    exact FastRows.row2687_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root20Child24 : search edge 3 [83, 2687] root20Candidates24 2770 winner = winner := by decide
theorem root20Step24 : scan edge (search edge 3) 4 [83] 83 root20Tail24 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail25 winner := by
  rw [root20Tail24, scan, if_neg (by decide), root20Candidates24_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail25
    (if 3 ≤ root20Candidates24.length then search edge 3 [83, 2687] root20Candidates24 2770 winner else winner) = _
  rw [if_pos (by decide), root20Child24]
def root20Candidates25 : List Nat := [12713, 20393]
set_option maxRecDepth 8192 in
theorem root20Candidates25_checked : candidates edge 4 83 2693 root20Tail26 winner.bound = root20Candidates25 := by
  unfold candidates
  have row : edge 2693 = (SortedNatTree.contains FastRows.row2693) := by
    change Data.row2693.contains = (SortedNatTree.contains FastRows.row2693)
    exact FastRows.row2693_checked.symm
  rw [row]
  decide
theorem root20Step25 : scan edge (search edge 3) 4 [83] 83 root20Tail25 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail26 winner := by
  rw [root20Tail25, scan, if_neg (by decide), root20Candidates25_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail26
    (if 3 ≤ root20Candidates25.length then search edge 3 [83, 2693] root20Candidates25 2776 winner else winner) = _
  rw [if_neg (by decide)]
def root20Candidates26 : List Nat := [3191, 13457, 14897]
set_option maxRecDepth 8192 in
theorem root20Candidates26_checked : candidates edge 4 83 2729 root20Tail27 winner.bound = root20Candidates26 := by
  unfold candidates
  have row : edge 2729 = (SortedNatTree.contains FastRows.row2729) := by
    change Data.row2729.contains = (SortedNatTree.contains FastRows.row2729)
    exact FastRows.row2729_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root20Child26 : search edge 3 [83, 2729] root20Candidates26 2812 winner = winner := by decide
theorem root20Step26 : scan edge (search edge 3) 4 [83] 83 root20Tail26 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail27 winner := by
  rw [root20Tail26, scan, if_neg (by decide), root20Candidates26_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail27
    (if 3 ≤ root20Candidates26.length then search edge 3 [83, 2729] root20Candidates26 2812 winner else winner) = _
  rw [if_pos (by decide), root20Child26]
def root20Candidates27 : List Nat := [5987, 8171, 16619, 20177]
set_option maxRecDepth 8192 in
theorem root20Candidates27_checked : candidates edge 4 83 2801 root20Tail28 winner.bound = root20Candidates27 := by
  unfold candidates
  have row : edge 2801 = (SortedNatTree.contains FastRows.row2801) := by
    change Data.row2801.contains = (SortedNatTree.contains FastRows.row2801)
    exact FastRows.row2801_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root20Child27 : search edge 3 [83, 2801] root20Candidates27 2884 winner = winner := by decide
theorem root20Step27 : scan edge (search edge 3) 4 [83] 83 root20Tail27 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail28 winner := by
  rw [root20Tail27, scan, if_neg (by decide), root20Candidates27_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail28
    (if 3 ≤ root20Candidates27.length then search edge 3 [83, 2801] root20Candidates27 2884 winner else winner) = _
  rw [if_pos (by decide), root20Child27]
def root20Candidates28 : List Nat := [3557, 8597, 12197]
set_option maxRecDepth 8192 in
theorem root20Candidates28_checked : candidates edge 4 83 2969 root20Tail29 winner.bound = root20Candidates28 := by
  unfold candidates
  have row : edge 2969 = (SortedNatTree.contains FastRows.row2969) := by
    change Data.row2969.contains = (SortedNatTree.contains FastRows.row2969)
    exact FastRows.row2969_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root20Child28 : search edge 3 [83, 2969] root20Candidates28 3052 winner = winner := by decide
theorem root20Step28 : scan edge (search edge 3) 4 [83] 83 root20Tail28 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail29 winner := by
  rw [root20Tail28, scan, if_neg (by decide), root20Candidates28_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail29
    (if 3 ≤ root20Candidates28.length then search edge 3 [83, 2969] root20Candidates28 3052 winner else winner) = _
  rw [if_pos (by decide), root20Child28]
def root20Candidates29 : List Nat := [8597, 9161, 9539]
set_option maxRecDepth 8192 in
theorem root20Candidates29_checked : candidates edge 4 83 3191 root20Tail30 winner.bound = root20Candidates29 := by
  unfold candidates
  have row : edge 3191 = (SortedNatTree.contains FastRows.row3191) := by
    change Data.row3191.contains = (SortedNatTree.contains FastRows.row3191)
    exact FastRows.row3191_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root20Child29 : search edge 3 [83, 3191] root20Candidates29 3274 winner = winner := by decide
theorem root20Step29 : scan edge (search edge 3) 4 [83] 83 root20Tail29 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail30 winner := by
  rw [root20Tail29, scan, if_neg (by decide), root20Candidates29_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail30
    (if 3 ≤ root20Candidates29.length then search edge 3 [83, 3191] root20Candidates29 3274 winner else winner) = _
  rw [if_pos (by decide), root20Child29]
def root20Candidates30 : List Nat := [12377]
set_option maxRecDepth 8192 in
theorem root20Candidates30_checked : candidates edge 4 83 3347 root20Tail31 winner.bound = root20Candidates30 := by
  unfold candidates
  have row : edge 3347 = (SortedNatTree.contains FastRows.row3347) := by
    change Data.row3347.contains = (SortedNatTree.contains FastRows.row3347)
    exact FastRows.row3347_checked.symm
  rw [row]
  decide
theorem root20Step30 : scan edge (search edge 3) 4 [83] 83 root20Tail30 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail31 winner := by
  rw [root20Tail30, scan, if_neg (by decide), root20Candidates30_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail31
    (if 3 ≤ root20Candidates30.length then search edge 3 [83, 3347] root20Candidates30 3430 winner else winner) = _
  rw [if_neg (by decide)]
def root20Candidates31 : List Nat := [19559]
set_option maxRecDepth 8192 in
theorem root20Candidates31_checked : candidates edge 4 83 3491 root20Tail32 winner.bound = root20Candidates31 := by
  unfold candidates
  have row : edge 3491 = (SortedNatTree.contains FastRows.row3491) := by
    change Data.row3491.contains = (SortedNatTree.contains FastRows.row3491)
    exact FastRows.row3491_checked.symm
  rw [row]
  decide
theorem root20Step31 : scan edge (search edge 3) 4 [83] 83 root20Tail31 winner =
    scan edge (search edge 3) 4 [83] 83 root20Tail32 winner := by
  rw [root20Tail31, scan, if_neg (by decide), root20Candidates31_checked]
  change scan edge (search edge 3) 4 [83] 83 root20Tail32
    (if 3 ≤ root20Candidates31.length then search edge 3 [83, 3491] root20Candidates31 3574 winner else winner) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
