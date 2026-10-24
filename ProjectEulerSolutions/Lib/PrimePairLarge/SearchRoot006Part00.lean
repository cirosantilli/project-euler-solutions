import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot006Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root6Candidates0 : List Nat := [1481, 2333, 2741, 3617, 4211, 4931, 5483, 5927, 6659, 8171, 16529, 17519]
set_option maxRecDepth 8192 in
theorem root6Candidates0_checked : candidates edge 4 23 47 root6Tail1 winner.bound = root6Candidates0 := by
  unfold candidates
  have row : edge 47 = (SortedNatTree.contains FastRows.row47) := by
    change Data.row47.contains = (SortedNatTree.contains FastRows.row47)
    exact FastRows.row47_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child0 : search edge 3 [23, 47] root6Candidates0 70 winner = winner := by decide
theorem root6Step0 : scan edge (search edge 3) 4 [23] 23 root6Tail0 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail1 winner := by
  rw [root6Tail0, scan, if_neg (by decide), root6Candidates0_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail1
    (if 3 ≤ root6Candidates0.length then search edge 3 [23, 47] root6Candidates0 70 winner else winner) = _
  rw [if_pos (by decide), root6Child0]
def root6Candidates1 : List Nat := [1223, 2357, 3407, 4211, 4721, 6323, 16487, 25793]
set_option maxRecDepth 8192 in
theorem root6Candidates1_checked : candidates edge 4 23 89 root6Tail2 winner.bound = root6Candidates1 := by
  unfold candidates
  have row : edge 89 = (SortedNatTree.contains FastRows.row89) := by
    change Data.row89.contains = (SortedNatTree.contains FastRows.row89)
    exact FastRows.row89_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child1 : search edge 3 [23, 89] root6Candidates1 112 winner = winner := by decide
theorem root6Step1 : scan edge (search edge 3) 4 [23] 23 root6Tail1 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail2 winner := by
  rw [root6Tail1, scan, if_neg (by decide), root6Candidates1_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail2
    (if 3 ≤ root6Candidates1.length then search edge 3 [23, 89] root6Candidates1 112 winner else winner) = _
  rw [if_pos (by decide), root6Child1]
def root6Candidates2 : List Nat := [677, 827, 1223, 8237, 24851]
set_option maxRecDepth 8192 in
theorem root6Candidates2_checked : candidates edge 4 23 311 root6Tail3 winner.bound = root6Candidates2 := by
  unfold candidates
  have row : edge 311 = (SortedNatTree.contains FastRows.row311) := by
    change Data.row311.contains = (SortedNatTree.contains FastRows.row311)
    exact FastRows.row311_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child2 : search edge 3 [23, 311] root6Candidates2 334 winner = winner := by decide
theorem root6Step2 : scan edge (search edge 3) 4 [23] 23 root6Tail2 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail3 winner := by
  rw [root6Tail2, scan, if_neg (by decide), root6Candidates2_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail3
    (if 3 ≤ root6Candidates2.length then search edge 3 [23, 311] root6Candidates2 334 winner else winner) = _
  rw [if_pos (by decide), root6Child2]
def root6Candidates3 : List Nat := [1613, 5483, 6563, 11321, 16487, 17327, 23039]
set_option maxRecDepth 8192 in
theorem root6Candidates3_checked : candidates edge 4 23 509 root6Tail4 winner.bound = root6Candidates3 := by
  unfold candidates
  have row : edge 509 = (SortedNatTree.contains FastRows.row509) := by
    change Data.row509.contains = (SortedNatTree.contains FastRows.row509)
    exact FastRows.row509_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child3 : search edge 3 [23, 509] root6Candidates3 532 winner = winner := by decide
theorem root6Step3 : scan edge (search edge 3) 4 [23] 23 root6Tail3 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail4 winner := by
  rw [root6Tail3, scan, if_neg (by decide), root6Candidates3_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail4
    (if 3 ≤ root6Candidates3.length then search edge 3 [23, 509] root6Candidates3 532 winner else winner) = _
  rw [if_pos (by decide), root6Child3]
def root6Candidates4 : List Nat := [827, 1493, 1871, 16487]
set_option maxRecDepth 8192 in
theorem root6Candidates4_checked : candidates edge 4 23 677 root6Tail5 winner.bound = root6Candidates4 := by
  unfold candidates
  have row : edge 677 = (SortedNatTree.contains FastRows.row677) := by
    change Data.row677.contains = (SortedNatTree.contains FastRows.row677)
    exact FastRows.row677_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child4 : search edge 3 [23, 677] root6Candidates4 700 winner = winner := by decide
theorem root6Step4 : scan edge (search edge 3) 4 [23] 23 root6Tail4 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail5 winner := by
  rw [root6Tail4, scan, if_neg (by decide), root6Candidates4_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail5
    (if 3 ≤ root6Candidates4.length then search edge 3 [23, 677] root6Candidates4 700 winner else winner) = _
  rw [if_pos (by decide), root6Child4]
def root6Candidates5 : List Nat := [1481, 1493, 1871, 2753, 5591, 6323, 18119, 19841, 23369, 24953]
set_option maxRecDepth 8192 in
theorem root6Candidates5_checked : candidates edge 4 23 743 root6Tail6 winner.bound = root6Candidates5 := by
  unfold candidates
  have row : edge 743 = (SortedNatTree.contains FastRows.row743) := by
    change Data.row743.contains = (SortedNatTree.contains FastRows.row743)
    exact FastRows.row743_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child5 : search edge 3 [23, 743] root6Candidates5 766 winner = winner := by decide
theorem root6Step5 : scan edge (search edge 3) 4 [23] 23 root6Tail5 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail6 winner := by
  rw [root6Tail5, scan, if_neg (by decide), root6Candidates5_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail6
    (if 3 ≤ root6Candidates5.length then search edge 3 [23, 743] root6Candidates5 766 winner else winner) = _
  rw [if_pos (by decide), root6Child5]
def root6Candidates6 : List Nat := [1613, 4463, 5009, 13799, 13931, 16487]
set_option maxRecDepth 8192 in
theorem root6Candidates6_checked : candidates edge 4 23 761 root6Tail7 winner.bound = root6Candidates6 := by
  unfold candidates
  have row : edge 761 = (SortedNatTree.contains FastRows.row761) := by
    change Data.row761.contains = (SortedNatTree.contains FastRows.row761)
    exact FastRows.row761_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child6 : search edge 3 [23, 761] root6Candidates6 784 winner = winner := by decide
theorem root6Step6 : scan edge (search edge 3) 4 [23] 23 root6Tail6 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail7 winner := by
  rw [root6Tail6, scan, if_neg (by decide), root6Candidates6_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail7
    (if 3 ≤ root6Candidates6.length then search edge 3 [23, 761] root6Candidates6 784 winner else winner) = _
  rw [if_pos (by decide), root6Child6]
def root6Candidates7 : List Nat := [2129, 2591, 5927, 7547, 9461]
set_option maxRecDepth 8192 in
theorem root6Candidates7_checked : candidates edge 4 23 773 root6Tail8 winner.bound = root6Candidates7 := by
  unfold candidates
  have row : edge 773 = (SortedNatTree.contains FastRows.row773) := by
    change Data.row773.contains = (SortedNatTree.contains FastRows.row773)
    exact FastRows.row773_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child7 : search edge 3 [23, 773] root6Candidates7 796 winner = winner := by decide
theorem root6Step7 : scan edge (search edge 3) 4 [23] 23 root6Tail7 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail8 winner := by
  rw [root6Tail7, scan, if_neg (by decide), root6Candidates7_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail8
    (if 3 ≤ root6Candidates7.length then search edge 3 [23, 773] root6Candidates7 796 winner else winner) = _
  rw [if_pos (by decide), root6Child7]
def root6Candidates8 : List Nat := [1871, 2819, 4463, 9461, 16553]
set_option maxRecDepth 8192 in
theorem root6Candidates8_checked : candidates edge 4 23 827 root6Tail9 winner.bound = root6Candidates8 := by
  unfold candidates
  have row : edge 827 = (SortedNatTree.contains FastRows.row827) := by
    change Data.row827.contains = (SortedNatTree.contains FastRows.row827)
    exact FastRows.row827_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child8 : search edge 3 [23, 827] root6Candidates8 850 winner = winner := by decide
theorem root6Step8 : scan edge (search edge 3) 4 [23] 23 root6Tail8 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail9 winner := by
  rw [root6Tail8, scan, if_neg (by decide), root6Candidates8_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail9
    (if 3 ≤ root6Candidates8.length then search edge 3 [23, 827] root6Candidates8 850 winner else winner) = _
  rw [if_pos (by decide), root6Child8]
def root6Candidates9 : List Nat := [1481, 11321]
set_option maxRecDepth 8192 in
theorem root6Candidates9_checked : candidates edge 4 23 1031 root6Tail10 winner.bound = root6Candidates9 := by
  unfold candidates
  have row : edge 1031 = (SortedNatTree.contains FastRows.row1031) := by
    change Data.row1031.contains = (SortedNatTree.contains FastRows.row1031)
    exact FastRows.row1031_checked.symm
  rw [row]
  decide
theorem root6Step9 : scan edge (search edge 3) 4 [23] 23 root6Tail9 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail10 winner := by
  rw [root6Tail9, scan, if_neg (by decide), root6Candidates9_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail10
    (if 3 ≤ root6Candidates9.length then search edge 3 [23, 1031] root6Candidates9 1054 winner else winner) = _
  rw [if_neg (by decide)]
def root6Candidates10 : List Nat := [14957, 21101]
set_option maxRecDepth 8192 in
theorem root6Candidates10_checked : candidates edge 4 23 1109 root6Tail11 winner.bound = root6Candidates10 := by
  unfold candidates
  have row : edge 1109 = (SortedNatTree.contains FastRows.row1109) := by
    change Data.row1109.contains = (SortedNatTree.contains FastRows.row1109)
    exact FastRows.row1109_checked.symm
  rw [row]
  decide
theorem root6Step10 : scan edge (search edge 3) 4 [23] 23 root6Tail10 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail11 winner := by
  rw [root6Tail10, scan, if_neg (by decide), root6Candidates10_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail11
    (if 3 ≤ root6Candidates10.length then search edge 3 [23, 1109] root6Candidates10 1132 winner else winner) = _
  rw [if_neg (by decide)]
def root6Candidates11 : List Nat := [1493, 8423, 18593]
set_option maxRecDepth 8192 in
theorem root6Candidates11_checked : candidates edge 4 23 1223 root6Tail12 winner.bound = root6Candidates11 := by
  unfold candidates
  have row : edge 1223 = (SortedNatTree.contains FastRows.row1223) := by
    change Data.row1223.contains = (SortedNatTree.contains FastRows.row1223)
    exact FastRows.row1223_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child11 : search edge 3 [23, 1223] root6Candidates11 1246 winner = winner := by decide
theorem root6Step11 : scan edge (search edge 3) 4 [23] 23 root6Tail11 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail12 winner := by
  rw [root6Tail11, scan, if_neg (by decide), root6Candidates11_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail12
    (if 3 ≤ root6Candidates11.length then search edge 3 [23, 1223] root6Candidates11 1246 winner else winner) = _
  rw [if_pos (by decide), root6Child11]
def root6Candidates12 : List Nat := [4139, 6449, 22193]
set_option maxRecDepth 8192 in
theorem root6Candidates12_checked : candidates edge 4 23 1289 root6Tail13 winner.bound = root6Candidates12 := by
  unfold candidates
  have row : edge 1289 = (SortedNatTree.contains FastRows.row1289) := by
    change Data.row1289.contains = (SortedNatTree.contains FastRows.row1289)
    exact FastRows.row1289_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child12 : search edge 3 [23, 1289] root6Candidates12 1312 winner = winner := by decide
theorem root6Step12 : scan edge (search edge 3) 4 [23] 23 root6Tail12 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail13 winner := by
  rw [root6Tail12, scan, if_neg (by decide), root6Candidates12_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail13
    (if 3 ≤ root6Candidates12.length then search edge 3 [23, 1289] root6Candidates12 1312 winner else winner) = _
  rw [if_pos (by decide), root6Child12]
def root6Candidates13 : List Nat := [4211, 13799, 21023]
set_option maxRecDepth 8192 in
theorem root6Candidates13_checked : candidates edge 4 23 1481 root6Tail14 winner.bound = root6Candidates13 := by
  unfold candidates
  have row : edge 1481 = (SortedNatTree.contains FastRows.row1481) := by
    change Data.row1481.contains = (SortedNatTree.contains FastRows.row1481)
    exact FastRows.row1481_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child13 : search edge 3 [23, 1481] root6Candidates13 1504 winner = winner := by decide
theorem root6Step13 : scan edge (search edge 3) 4 [23] 23 root6Tail13 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail14 winner := by
  rw [root6Tail13, scan, if_neg (by decide), root6Candidates13_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail14
    (if 3 ≤ root6Candidates13.length then search edge 3 [23, 1481] root6Candidates13 1504 winner else winner) = _
  rw [if_pos (by decide), root6Child13]
def root6Candidates14 : List Nat := [7607, 9461, 16451, 23003]
set_option maxRecDepth 8192 in
theorem root6Candidates14_checked : candidates edge 4 23 1493 root6Tail15 winner.bound = root6Candidates14 := by
  unfold candidates
  have row : edge 1493 = (SortedNatTree.contains FastRows.row1493) := by
    change Data.row1493.contains = (SortedNatTree.contains FastRows.row1493)
    exact FastRows.row1493_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child14 : search edge 3 [23, 1493] root6Candidates14 1516 winner = winner := by decide
theorem root6Step14 : scan edge (search edge 3) 4 [23] 23 root6Tail14 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail15 winner := by
  rw [root6Tail14, scan, if_neg (by decide), root6Candidates14_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail15
    (if 3 ≤ root6Candidates14.length then search edge 3 [23, 1493] root6Candidates14 1516 winner else winner) = _
  rw [if_pos (by decide), root6Child14]
def root6Candidates15 : List Nat := [5009, 5849, 7673, 8171, 21023, 24281]
set_option maxRecDepth 8192 in
theorem root6Candidates15_checked : candidates edge 4 23 1607 root6Tail16 winner.bound = root6Candidates15 := by
  unfold candidates
  have row : edge 1607 = (SortedNatTree.contains FastRows.row1607) := by
    change Data.row1607.contains = (SortedNatTree.contains FastRows.row1607)
    exact FastRows.row1607_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child15 : search edge 3 [23, 1607] root6Candidates15 1630 winner = winner := by decide
theorem root6Step15 : scan edge (search edge 3) 4 [23] 23 root6Tail15 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail16 winner := by
  rw [root6Tail15, scan, if_neg (by decide), root6Candidates15_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail16
    (if 3 ≤ root6Candidates15.length then search edge 3 [23, 1607] root6Candidates15 1630 winner else winner) = _
  rw [if_pos (by decide), root6Child15]
def root6Candidates16 : List Nat := [5813, 7487, 14957, 19841]
set_option maxRecDepth 8192 in
theorem root6Candidates16_checked : candidates edge 4 23 1613 root6Tail17 winner.bound = root6Candidates16 := by
  unfold candidates
  have row : edge 1613 = (SortedNatTree.contains FastRows.row1613) := by
    change Data.row1613.contains = (SortedNatTree.contains FastRows.row1613)
    exact FastRows.row1613_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child16 : search edge 3 [23, 1613] root6Candidates16 1636 winner = winner := by decide
theorem root6Step16 : scan edge (search edge 3) 4 [23] 23 root6Tail16 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail17 winner := by
  rw [root6Tail16, scan, if_neg (by decide), root6Candidates16_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail17
    (if 3 ≤ root6Candidates16.length then search edge 3 [23, 1613] root6Candidates16 1636 winner else winner) = _
  rw [if_pos (by decide), root6Child16]
def root6Candidates17 : List Nat := [4139, 5051, 5441, 6323, 8171, 8681, 16451, 17747]
set_option maxRecDepth 8192 in
theorem root6Candidates17_checked : candidates edge 4 23 1871 root6Tail18 winner.bound = root6Candidates17 := by
  unfold candidates
  have row : edge 1871 = (SortedNatTree.contains FastRows.row1871) := by
    change Data.row1871.contains = (SortedNatTree.contains FastRows.row1871)
    exact FastRows.row1871_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child17 : search edge 3 [23, 1871] root6Candidates17 1894 winner = winner := by decide
theorem root6Step17 : scan edge (search edge 3) 4 [23] 23 root6Tail17 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail18 winner := by
  rw [root6Tail17, scan, if_neg (by decide), root6Candidates17_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail18
    (if 3 ≤ root6Candidates17.length then search edge 3 [23, 1871] root6Candidates17 1894 winner else winner) = _
  rw [if_pos (by decide), root6Child17]
def root6Candidates18 : List Nat := [3557, 5051, 6563, 8171, 8747, 13931]
set_option maxRecDepth 8192 in
theorem root6Candidates18_checked : candidates edge 4 23 2003 root6Tail19 winner.bound = root6Candidates18 := by
  unfold candidates
  have row : edge 2003 = (SortedNatTree.contains FastRows.row2003) := by
    change Data.row2003.contains = (SortedNatTree.contains FastRows.row2003)
    exact FastRows.row2003_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child18 : search edge 3 [23, 2003] root6Candidates18 2026 winner = winner := by decide
theorem root6Step18 : scan edge (search edge 3) 4 [23] 23 root6Tail18 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail19 winner := by
  rw [root6Tail18, scan, if_neg (by decide), root6Candidates18_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail19
    (if 3 ≤ root6Candidates18.length then search edge 3 [23, 2003] root6Candidates18 2026 winner else winner) = _
  rw [if_pos (by decide), root6Child18]
def root6Candidates19 : List Nat := [5051, 13163, 23549]
set_option maxRecDepth 8192 in
theorem root6Candidates19_checked : candidates edge 4 23 2129 root6Tail20 winner.bound = root6Candidates19 := by
  unfold candidates
  have row : edge 2129 = (SortedNatTree.contains FastRows.row2129) := by
    change Data.row2129.contains = (SortedNatTree.contains FastRows.row2129)
    exact FastRows.row2129_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child19 : search edge 3 [23, 2129] root6Candidates19 2152 winner = winner := by decide
theorem root6Step19 : scan edge (search edge 3) 4 [23] 23 root6Tail19 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail20 winner := by
  rw [root6Tail19, scan, if_neg (by decide), root6Candidates19_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail20
    (if 3 ≤ root6Candidates19.length then search edge 3 [23, 2129] root6Candidates19 2152 winner else winner) = _
  rw [if_pos (by decide), root6Child19]
def root6Candidates20 : List Nat := [5927, 16451, 16553]
set_option maxRecDepth 8192 in
theorem root6Candidates20_checked : candidates edge 4 23 2333 root6Tail21 winner.bound = root6Candidates20 := by
  unfold candidates
  have row : edge 2333 = (SortedNatTree.contains FastRows.row2333) := by
    change Data.row2333.contains = (SortedNatTree.contains FastRows.row2333)
    exact FastRows.row2333_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child20 : search edge 3 [23, 2333] root6Candidates20 2356 winner = winner := by decide
theorem root6Step20 : scan edge (search edge 3) 4 [23] 23 root6Tail20 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail21 winner := by
  rw [root6Tail20, scan, if_neg (by decide), root6Candidates20_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail21
    (if 3 ≤ root6Candidates20.length then search edge 3 [23, 2333] root6Candidates20 2356 winner else winner) = _
  rw [if_pos (by decide), root6Child20]
def root6Candidates21 : List Nat := [6323, 7331, 8237]
set_option maxRecDepth 8192 in
theorem root6Candidates21_checked : candidates edge 4 23 2357 root6Tail22 winner.bound = root6Candidates21 := by
  unfold candidates
  have row : edge 2357 = (SortedNatTree.contains FastRows.row2357) := by
    change Data.row2357.contains = (SortedNatTree.contains FastRows.row2357)
    exact FastRows.row2357_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child21 : search edge 3 [23, 2357] root6Candidates21 2380 winner = winner := by decide
theorem root6Step21 : scan edge (search edge 3) 4 [23] 23 root6Tail21 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail22 winner := by
  rw [root6Tail21, scan, if_neg (by decide), root6Candidates21_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail22
    (if 3 ≤ root6Candidates21.length then search edge 3 [23, 2357] root6Candidates21 2380 winner else winner) = _
  rw [if_pos (by decide), root6Child21]
def root6Candidates22 : List Nat := [2633, 4211, 5849, 8423, 19433]
set_option maxRecDepth 8192 in
theorem root6Candidates22_checked : candidates edge 4 23 2591 root6Tail23 winner.bound = root6Candidates22 := by
  unfold candidates
  have row : edge 2591 = (SortedNatTree.contains FastRows.row2591) := by
    change Data.row2591.contains = (SortedNatTree.contains FastRows.row2591)
    exact FastRows.row2591_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child22 : search edge 3 [23, 2591] root6Candidates22 2614 winner = winner := by decide
theorem root6Step22 : scan edge (search edge 3) 4 [23] 23 root6Tail22 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail23 winner := by
  rw [root6Tail22, scan, if_neg (by decide), root6Candidates22_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail23
    (if 3 ≤ root6Candidates22.length then search edge 3 [23, 2591] root6Candidates22 2614 winner else winner) = _
  rw [if_pos (by decide), root6Child22]
def root6Candidates23 : List Nat := [2699, 5051, 7547, 23003]
set_option maxRecDepth 8192 in
theorem root6Candidates23_checked : candidates edge 4 23 2633 root6Tail24 winner.bound = root6Candidates23 := by
  unfold candidates
  have row : edge 2633 = (SortedNatTree.contains FastRows.row2633) := by
    change Data.row2633.contains = (SortedNatTree.contains FastRows.row2633)
    exact FastRows.row2633_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child23 : search edge 3 [23, 2633] root6Candidates23 2656 winner = winner := by decide
theorem root6Step23 : scan edge (search edge 3) 4 [23] 23 root6Tail23 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail24 winner := by
  rw [root6Tail23, scan, if_neg (by decide), root6Candidates23_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail24
    (if 3 ≤ root6Candidates23.length then search edge 3 [23, 2633] root6Candidates23 2656 winner else winner) = _
  rw [if_pos (by decide), root6Child23]
def root6Candidates24 : List Nat := [6329, 17327, 22193]
set_option maxRecDepth 8192 in
theorem root6Candidates24_checked : candidates edge 4 23 2699 root6Tail25 winner.bound = root6Candidates24 := by
  unfold candidates
  have row : edge 2699 = (SortedNatTree.contains FastRows.row2699) := by
    change Data.row2699.contains = (SortedNatTree.contains FastRows.row2699)
    exact FastRows.row2699_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child24 : search edge 3 [23, 2699] root6Candidates24 2722 winner = winner := by decide
theorem root6Step24 : scan edge (search edge 3) 4 [23] 23 root6Tail24 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail25 winner := by
  rw [root6Tail24, scan, if_neg (by decide), root6Candidates24_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail25
    (if 3 ≤ root6Candidates24.length then search edge 3 [23, 2699] root6Candidates24 2722 winner else winner) = _
  rw [if_pos (by decide), root6Child24]
def root6Candidates25 : List Nat := [7331, 8747, 13799, 22193]
set_option maxRecDepth 8192 in
theorem root6Candidates25_checked : candidates edge 4 23 2741 root6Tail26 winner.bound = root6Candidates25 := by
  unfold candidates
  have row : edge 2741 = (SortedNatTree.contains FastRows.row2741) := by
    change Data.row2741.contains = (SortedNatTree.contains FastRows.row2741)
    exact FastRows.row2741_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child25 : search edge 3 [23, 2741] root6Candidates25 2764 winner = winner := by decide
theorem root6Step25 : scan edge (search edge 3) 4 [23] 23 root6Tail25 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail26 winner := by
  rw [root6Tail25, scan, if_neg (by decide), root6Candidates25_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail26
    (if 3 ≤ root6Candidates25.length then search edge 3 [23, 2741] root6Candidates25 2764 winner else winner) = _
  rw [if_pos (by decide), root6Child25]
def root6Candidates26 : List Nat := [7487, 18089]
set_option maxRecDepth 8192 in
theorem root6Candidates26_checked : candidates edge 4 23 2753 root6Tail27 winner.bound = root6Candidates26 := by
  unfold candidates
  have row : edge 2753 = (SortedNatTree.contains FastRows.row2753) := by
    change Data.row2753.contains = (SortedNatTree.contains FastRows.row2753)
    exact FastRows.row2753_checked.symm
  rw [row]
  decide
theorem root6Step26 : scan edge (search edge 3) 4 [23] 23 root6Tail26 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail27 winner := by
  rw [root6Tail26, scan, if_neg (by decide), root6Candidates26_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail27
    (if 3 ≤ root6Candidates26.length then search edge 3 [23, 2753] root6Candidates26 2776 winner else winner) = _
  rw [if_neg (by decide)]
def root6Candidates27 : List Nat := [3617, 5927, 7607, 13799, 23039]
set_option maxRecDepth 8192 in
theorem root6Candidates27_checked : candidates edge 4 23 2819 root6Tail28 winner.bound = root6Candidates27 := by
  unfold candidates
  have row : edge 2819 = (SortedNatTree.contains FastRows.row2819) := by
    change Data.row2819.contains = (SortedNatTree.contains FastRows.row2819)
    exact FastRows.row2819_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child27 : search edge 3 [23, 2819] root6Candidates27 2842 winner = winner := by decide
theorem root6Step27 : scan edge (search edge 3) 4 [23] 23 root6Tail27 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail28 winner := by
  rw [root6Tail27, scan, if_neg (by decide), root6Candidates27_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail28
    (if 3 ≤ root6Candidates27.length then search edge 3 [23, 2819] root6Candidates27 2842 winner else winner) = _
  rw [if_pos (by decide), root6Child27]
def root6Candidates28 : List Nat := [4931, 8747, 9461, 9929]
set_option maxRecDepth 8192 in
theorem root6Candidates28_checked : candidates edge 4 23 3083 root6Tail29 winner.bound = root6Candidates28 := by
  unfold candidates
  have row : edge 3083 = (SortedNatTree.contains FastRows.row3083) := by
    change Data.row3083.contains = (SortedNatTree.contains FastRows.row3083)
    exact FastRows.row3083_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child28 : search edge 3 [23, 3083] root6Candidates28 3106 winner = winner := by decide
theorem root6Step28 : scan edge (search edge 3) 4 [23] 23 root6Tail28 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail29 winner := by
  rw [root6Tail28, scan, if_neg (by decide), root6Candidates28_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail29
    (if 3 ≤ root6Candidates28.length then search edge 3 [23, 3083] root6Candidates28 3106 winner else winner) = _
  rw [if_pos (by decide), root6Child28]
def root6Candidates29 : List Nat := [7691, 11411, 19433]
set_option maxRecDepth 8192 in
theorem root6Candidates29_checked : candidates edge 4 23 3407 root6Tail30 winner.bound = root6Candidates29 := by
  unfold candidates
  have row : edge 3407 = (SortedNatTree.contains FastRows.row3407) := by
    change Data.row3407.contains = (SortedNatTree.contains FastRows.row3407)
    exact FastRows.row3407_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child29 : search edge 3 [23, 3407] root6Candidates29 3430 winner = winner := by decide
theorem root6Step29 : scan edge (search edge 3) 4 [23] 23 root6Tail29 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail30 winner := by
  rw [root6Tail29, scan, if_neg (by decide), root6Candidates29_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail30
    (if 3 ≤ root6Candidates29.length then search edge 3 [23, 3407] root6Candidates29 3430 winner else winner) = _
  rw [if_pos (by decide), root6Child29]
def root6Candidates30 : List Nat := [4139, 5051, 6329, 8681, 9857]
set_option maxRecDepth 8192 in
theorem root6Candidates30_checked : candidates edge 4 23 3557 root6Tail31 winner.bound = root6Candidates30 := by
  unfold candidates
  have row : edge 3557 = (SortedNatTree.contains FastRows.row3557) := by
    change Data.row3557.contains = (SortedNatTree.contains FastRows.row3557)
    exact FastRows.row3557_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child30 : search edge 3 [23, 3557] root6Candidates30 3580 winner = winner := by decide
theorem root6Step30 : scan edge (search edge 3) 4 [23] 23 root6Tail30 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail31 winner := by
  rw [root6Tail30, scan, if_neg (by decide), root6Candidates30_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail31
    (if 3 ≤ root6Candidates30.length then search edge 3 [23, 3557] root6Candidates30 3580 winner else winner) = _
  rw [if_pos (by decide), root6Child30]
def root6Candidates31 : List Nat := [7607, 8291, 17747]
set_option maxRecDepth 8192 in
theorem root6Candidates31_checked : candidates edge 4 23 3617 root6Tail32 winner.bound = root6Candidates31 := by
  unfold candidates
  have row : edge 3617 = (SortedNatTree.contains FastRows.row3617) := by
    change Data.row3617.contains = (SortedNatTree.contains FastRows.row3617)
    exact FastRows.row3617_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root6Child31 : search edge 3 [23, 3617] root6Candidates31 3640 winner = winner := by decide
theorem root6Step31 : scan edge (search edge 3) 4 [23] 23 root6Tail31 winner =
    scan edge (search edge 3) 4 [23] 23 root6Tail32 winner := by
  rw [root6Tail31, scan, if_neg (by decide), root6Candidates31_checked]
  change scan edge (search edge 3) 4 [23] 23 root6Tail32
    (if 3 ≤ root6Candidates31.length then search edge 3 [23, 3617] root6Candidates31 3640 winner else winner) = _
  rw [if_pos (by decide), root6Child31]
end ProjectEulerSolutions.PrimePairLarge.Search
