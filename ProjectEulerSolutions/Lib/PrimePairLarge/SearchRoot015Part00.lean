import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot015Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root15Candidates0 : List Nat := [4657, 5413, 13309, 16273]
set_option maxRecDepth 8192 in
theorem root15Candidates0_checked : candidates edge 4 61 151 root15Tail1 winner.bound = root15Candidates0 := by
  unfold candidates
  have row : edge 151 = (SortedNatTree.contains FastRows.row151) := by
    change Data.row151.contains = (SortedNatTree.contains FastRows.row151)
    exact FastRows.row151_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root15Child0 : search edge 3 [61, 151] root15Candidates0 212 winner = winner := by decide
theorem root15Step0 : scan edge (search edge 3) 4 [61] 61 root15Tail0 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail1 winner := by
  rw [root15Tail0, scan, if_neg (by decide), root15Candidates0_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail1
    (if 3 ≤ root15Candidates0.length then search edge 3 [61, 151] root15Candidates0 212 winner else winner) = _
  rw [if_pos (by decide), root15Child0]
def root15Candidates1 : List Nat := [1999, 3637, 11527, 19543]
set_option maxRecDepth 8192 in
theorem root15Candidates1_checked : candidates edge 4 61 331 root15Tail2 winner.bound = root15Candidates1 := by
  unfold candidates
  have row : edge 331 = (SortedNatTree.contains FastRows.row331) := by
    change Data.row331.contains = (SortedNatTree.contains FastRows.row331)
    exact FastRows.row331_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root15Child1 : search edge 3 [61, 331] root15Candidates1 392 winner = winner := by decide
theorem root15Step1 : scan edge (search edge 3) 4 [61] 61 root15Tail1 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail2 winner := by
  rw [root15Tail1, scan, if_neg (by decide), root15Candidates1_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail2
    (if 3 ≤ root15Candidates1.length then search edge 3 [61, 331] root15Candidates1 392 winner else winner) = _
  rw [if_pos (by decide), root15Child1]
def root15Candidates2 : List Nat := [1873, 7699, 16567, 16657, 17581]
set_option maxRecDepth 8192 in
theorem root15Candidates2_checked : candidates edge 4 61 409 root15Tail3 winner.bound = root15Candidates2 := by
  unfold candidates
  have row : edge 409 = (SortedNatTree.contains FastRows.row409) := by
    change Data.row409.contains = (SortedNatTree.contains FastRows.row409)
    exact FastRows.row409_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root15Child2 : search edge 3 [61, 409] root15Candidates2 470 winner = winner := by decide
theorem root15Step2 : scan edge (search edge 3) 4 [61] 61 root15Tail2 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail3 winner := by
  rw [root15Tail2, scan, if_neg (by decide), root15Candidates2_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail3
    (if 3 ≤ root15Candidates2.length then search edge 3 [61, 409] root15Candidates2 470 winner else winner) = _
  rw [if_pos (by decide), root15Child2]
def root15Candidates3 : List Nat := [8941, 18517, 21727]
set_option maxRecDepth 8192 in
theorem root15Candidates3_checked : candidates edge 4 61 487 root15Tail4 winner.bound = root15Candidates3 := by
  unfold candidates
  have row : edge 487 = (SortedNatTree.contains FastRows.row487) := by
    change Data.row487.contains = (SortedNatTree.contains FastRows.row487)
    exact FastRows.row487_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root15Child3 : search edge 3 [61, 487] root15Candidates3 548 winner = winner := by decide
theorem root15Step3 : scan edge (search edge 3) 4 [61] 61 root15Tail3 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail4 winner := by
  rw [root15Tail3, scan, if_neg (by decide), root15Candidates3_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail4
    (if 3 ≤ root15Candidates3.length then search edge 3 [61, 487] root15Candidates3 548 winner else winner) = _
  rw [if_pos (by decide), root15Child3]
def root15Candidates4 : List Nat := [2383, 14887, 16057, 19423, 19543]
set_option maxRecDepth 8192 in
theorem root15Candidates4_checked : candidates edge 4 61 751 root15Tail5 winner.bound = root15Candidates4 := by
  unfold candidates
  have row : edge 751 = (SortedNatTree.contains FastRows.row751) := by
    change Data.row751.contains = (SortedNatTree.contains FastRows.row751)
    exact FastRows.row751_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root15Child4 : search edge 3 [61, 751] root15Candidates4 812 winner = winner := by decide
theorem root15Step4 : scan edge (search edge 3) 4 [61] 61 root15Tail4 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail5 winner := by
  rw [root15Tail4, scan, if_neg (by decide), root15Candidates4_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail5
    (if 3 ≤ root15Candidates4.length then search edge 3 [61, 751] root15Candidates4 812 winner else winner) = _
  rw [if_pos (by decide), root15Child4]
def root15Candidates5 : List Nat := [8161, 13309]
set_option maxRecDepth 8192 in
theorem root15Candidates5_checked : candidates edge 4 61 1069 root15Tail6 winner.bound = root15Candidates5 := by
  unfold candidates
  have row : edge 1069 = (SortedNatTree.contains FastRows.row1069) := by
    change Data.row1069.contains = (SortedNatTree.contains FastRows.row1069)
    exact FastRows.row1069_checked.symm
  rw [row]
  decide
theorem root15Step5 : scan edge (search edge 3) 4 [61] 61 root15Tail5 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail6 winner := by
  rw [root15Tail5, scan, if_neg (by decide), root15Candidates5_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail6
    (if 3 ≤ root15Candidates5.length then search edge 3 [61, 1069] root15Candidates5 1130 winner else winner) = _
  rw [if_neg (by decide)]
def root15Candidates6 : List Nat := [2797, 3049, 7129, 16111, 20809]
set_option maxRecDepth 8192 in
theorem root15Candidates6_checked : candidates edge 4 61 1453 root15Tail7 winner.bound = root15Candidates6 := by
  unfold candidates
  have row : edge 1453 = (SortedNatTree.contains FastRows.row1453) := by
    change Data.row1453.contains = (SortedNatTree.contains FastRows.row1453)
    exact FastRows.row1453_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root15Child6 : search edge 3 [61, 1453] root15Candidates6 1514 winner = winner := by decide
theorem root15Step6 : scan edge (search edge 3) 4 [61] 61 root15Tail6 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail7 winner := by
  rw [root15Tail6, scan, if_neg (by decide), root15Candidates6_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail7
    (if 3 ≤ root15Candidates6.length then search edge 3 [61, 1453] root15Candidates6 1514 winner else winner) = _
  rw [if_pos (by decide), root15Child6]
def root15Candidates7 : List Nat := [2341, 5869, 7873, 9511, 10861]
set_option maxRecDepth 8192 in
theorem root15Candidates7_checked : candidates edge 4 61 1483 root15Tail8 winner.bound = root15Candidates7 := by
  unfold candidates
  have row : edge 1483 = (SortedNatTree.contains FastRows.row1483) := by
    change Data.row1483.contains = (SortedNatTree.contains FastRows.row1483)
    exact FastRows.row1483_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root15Child7 : search edge 3 [61, 1483] root15Candidates7 1544 winner = winner := by decide
theorem root15Step7 : scan edge (search edge 3) 4 [61] 61 root15Tail7 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail8 winner := by
  rw [root15Tail7, scan, if_neg (by decide), root15Candidates7_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail8
    (if 3 ≤ root15Candidates7.length then search edge 3 [61, 1483] root15Candidates7 1544 winner else winner) = _
  rw [if_pos (by decide), root15Child7]
def root15Candidates8 : List Nat := [1753, 2371, 3181]
set_option maxRecDepth 8192 in
theorem root15Candidates8_checked : candidates edge 4 61 1693 root15Tail9 winner.bound = root15Candidates8 := by
  unfold candidates
  have row : edge 1693 = (SortedNatTree.contains FastRows.row1693) := by
    change Data.row1693.contains = (SortedNatTree.contains FastRows.row1693)
    exact FastRows.row1693_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root15Child8 : search edge 3 [61, 1693] root15Candidates8 1754 winner = winner := by decide
theorem root15Step8 : scan edge (search edge 3) 4 [61] 61 root15Tail8 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail9 winner := by
  rw [root15Tail8, scan, if_neg (by decide), root15Candidates8_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail9
    (if 3 ≤ root15Candidates8.length then search edge 3 [61, 1693] root15Candidates8 1754 winner else winner) = _
  rw [if_pos (by decide), root15Child8]
def root15Candidates9 : List Nat := [1873, 1951, 5827, 7237, 8161, 9007, 11527, 11839, 13309, 21121]
set_option maxRecDepth 8192 in
theorem root15Candidates9_checked : candidates edge 4 61 1753 root15Tail10 winner.bound = root15Candidates9 := by
  unfold candidates
  have row : edge 1753 = (SortedNatTree.contains FastRows.row1753) := by
    change Data.row1753.contains = (SortedNatTree.contains FastRows.row1753)
    exact FastRows.row1753_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root15Child9 : search edge 3 [61, 1753] root15Candidates9 1814 winner = winner := by decide
theorem root15Step9 : scan edge (search edge 3) 4 [61] 61 root15Tail9 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail10 winner := by
  rw [root15Tail9, scan, if_neg (by decide), root15Candidates9_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail10
    (if 3 ≤ root15Candidates9.length then search edge 3 [61, 1753] root15Candidates9 1814 winner else winner) = _
  rw [if_pos (by decide), root15Child9]
def root15Candidates10 : List Nat := [20533]
set_option maxRecDepth 8192 in
theorem root15Candidates10_checked : candidates edge 4 61 1801 root15Tail11 winner.bound = root15Candidates10 := by
  unfold candidates
  have row : edge 1801 = (SortedNatTree.contains FastRows.row1801) := by
    change Data.row1801.contains = (SortedNatTree.contains FastRows.row1801)
    exact FastRows.row1801_checked.symm
  rw [row]
  decide
theorem root15Step10 : scan edge (search edge 3) 4 [61] 61 root15Tail10 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail11 winner := by
  rw [root15Tail10, scan, if_neg (by decide), root15Candidates10_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail11
    (if 3 ≤ root15Candidates10.length then search edge 3 [61, 1801] root15Candidates10 1862 winner else winner) = _
  rw [if_neg (by decide)]
def root15Candidates11 : List Nat := [6997, 7963]
set_option maxRecDepth 8192 in
theorem root15Candidates11_checked : candidates edge 4 61 1873 root15Tail12 winner.bound = root15Candidates11 := by
  unfold candidates
  have row : edge 1873 = (SortedNatTree.contains FastRows.row1873) := by
    change Data.row1873.contains = (SortedNatTree.contains FastRows.row1873)
    exact FastRows.row1873_checked.symm
  rw [row]
  decide
theorem root15Step11 : scan edge (search edge 3) 4 [61] 61 root15Tail11 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail12 winner := by
  rw [root15Tail11, scan, if_neg (by decide), root15Candidates11_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail12
    (if 3 ≤ root15Candidates11.length then search edge 3 [61, 1873] root15Candidates11 1934 winner else winner) = _
  rw [if_neg (by decide)]
def root15Candidates12 : List Nat := [2341, 3733, 8581]
set_option maxRecDepth 8192 in
theorem root15Candidates12_checked : candidates edge 4 61 1951 root15Tail13 winner.bound = root15Candidates12 := by
  unfold candidates
  have row : edge 1951 = (SortedNatTree.contains FastRows.row1951) := by
    change Data.row1951.contains = (SortedNatTree.contains FastRows.row1951)
    exact FastRows.row1951_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root15Child12 : search edge 3 [61, 1951] root15Candidates12 2012 winner = winner := by decide
theorem root15Step12 : scan edge (search edge 3) 4 [61] 61 root15Tail12 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail13 winner := by
  rw [root15Tail12, scan, if_neg (by decide), root15Candidates12_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail13
    (if 3 ≤ root15Candidates12.length then search edge 3 [61, 1951] root15Candidates12 2012 winner else winner) = _
  rw [if_pos (by decide), root15Child12]
def root15Candidates13 : List Nat := [3469, 3733, 5869, 12541, 16567, 17299]
set_option maxRecDepth 8192 in
theorem root15Candidates13_checked : candidates edge 4 61 1999 root15Tail14 winner.bound = root15Candidates13 := by
  unfold candidates
  have row : edge 1999 = (SortedNatTree.contains FastRows.row1999) := by
    change Data.row1999.contains = (SortedNatTree.contains FastRows.row1999)
    exact FastRows.row1999_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root15Child13 : search edge 3 [61, 1999] root15Candidates13 2060 winner = winner := by decide
theorem root15Step13 : scan edge (search edge 3) 4 [61] 61 root15Tail13 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail14 winner := by
  rw [root15Tail13, scan, if_neg (by decide), root15Candidates13_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail14
    (if 3 ≤ root15Candidates13.length then search edge 3 [61, 1999] root15Candidates13 2060 winner else winner) = _
  rw [if_pos (by decide), root15Child13]
def root15Candidates14 : List Nat := [9511, 13309]
set_option maxRecDepth 8192 in
theorem root15Candidates14_checked : candidates edge 4 61 2341 root15Tail15 winner.bound = root15Candidates14 := by
  unfold candidates
  have row : edge 2341 = (SortedNatTree.contains FastRows.row2341) := by
    change Data.row2341.contains = (SortedNatTree.contains FastRows.row2341)
    exact FastRows.row2341_checked.symm
  rw [row]
  decide
theorem root15Step14 : scan edge (search edge 3) 4 [61] 61 root15Tail14 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail15 winner := by
  rw [root15Tail14, scan, if_neg (by decide), root15Candidates14_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail15
    (if 3 ≤ root15Candidates14.length then search edge 3 [61, 2341] root15Candidates14 2402 winner else winner) = _
  rw [if_neg (by decide)]
def root15Candidates15 : List Nat := [3169, 12703, 16567]
set_option maxRecDepth 8192 in
theorem root15Candidates15_checked : candidates edge 4 61 2371 root15Tail16 winner.bound = root15Candidates15 := by
  unfold candidates
  have row : edge 2371 = (SortedNatTree.contains FastRows.row2371) := by
    change Data.row2371.contains = (SortedNatTree.contains FastRows.row2371)
    exact FastRows.row2371_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root15Child15 : search edge 3 [61, 2371] root15Candidates15 2432 winner = winner := by decide
theorem root15Step15 : scan edge (search edge 3) 4 [61] 61 root15Tail15 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail16 winner := by
  rw [root15Tail15, scan, if_neg (by decide), root15Candidates15_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail16
    (if 3 ≤ root15Candidates15.length then search edge 3 [61, 2371] root15Candidates15 2432 winner else winner) = _
  rw [if_pos (by decide), root15Child15]
def root15Candidates16 : List Nat := [5431, 7129, 7873]
set_option maxRecDepth 8192 in
theorem root15Candidates16_checked : candidates edge 4 61 2383 root15Tail17 winner.bound = root15Candidates16 := by
  unfold candidates
  have row : edge 2383 = (SortedNatTree.contains FastRows.row2383) := by
    change Data.row2383.contains = (SortedNatTree.contains FastRows.row2383)
    exact FastRows.row2383_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root15Child16 : search edge 3 [61, 2383] root15Candidates16 2444 winner = winner := by decide
theorem root15Step16 : scan edge (search edge 3) 4 [61] 61 root15Tail16 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail17 winner := by
  rw [root15Tail16, scan, if_neg (by decide), root15Candidates16_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail17
    (if 3 ≤ root15Candidates16.length then search edge 3 [61, 2383] root15Candidates16 2444 winner else winner) = _
  rw [if_pos (by decide), root15Child16]
def root15Candidates17 : List Nat := [5449, 6367, 7681, 9181, 17959, 22447]
set_option maxRecDepth 8192 in
theorem root15Candidates17_checked : candidates edge 4 61 2797 root15Tail18 winner.bound = root15Candidates17 := by
  unfold candidates
  have row : edge 2797 = (SortedNatTree.contains FastRows.row2797) := by
    change Data.row2797.contains = (SortedNatTree.contains FastRows.row2797)
    exact FastRows.row2797_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root15Child17 : search edge 3 [61, 2797] root15Candidates17 2858 winner = winner := by decide
theorem root15Step17 : scan edge (search edge 3) 4 [61] 61 root15Tail17 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail18 winner := by
  rw [root15Tail17, scan, if_neg (by decide), root15Candidates17_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail18
    (if 3 ≤ root15Candidates17.length then search edge 3 [61, 2797] root15Candidates17 2858 winner else winner) = _
  rw [if_pos (by decide), root15Child17]
def root15Candidates18 : List Nat := [5557, 6871, 9511]
set_option maxRecDepth 8192 in
theorem root15Candidates18_checked : candidates edge 4 61 2971 root15Tail19 winner.bound = root15Candidates18 := by
  unfold candidates
  have row : edge 2971 = (SortedNatTree.contains FastRows.row2971) := by
    change Data.row2971.contains = (SortedNatTree.contains FastRows.row2971)
    exact FastRows.row2971_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root15Child18 : search edge 3 [61, 2971] root15Candidates18 3032 winner = winner := by decide
theorem root15Step18 : scan edge (search edge 3) 4 [61] 61 root15Tail18 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail19 winner := by
  rw [root15Tail18, scan, if_neg (by decide), root15Candidates18_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail19
    (if 3 ≤ root15Candidates18.length then search edge 3 [61, 2971] root15Candidates18 3032 winner else winner) = _
  rw [if_pos (by decide), root15Child18]
def root15Candidates19 : List Nat := [16111, 16657]
set_option maxRecDepth 8192 in
theorem root15Candidates19_checked : candidates edge 4 61 3049 root15Tail20 winner.bound = root15Candidates19 := by
  unfold candidates
  have row : edge 3049 = (SortedNatTree.contains FastRows.row3049) := by
    change Data.row3049.contains = (SortedNatTree.contains FastRows.row3049)
    exact FastRows.row3049_checked.symm
  rw [row]
  decide
theorem root15Step19 : scan edge (search edge 3) 4 [61] 61 root15Tail19 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail20 winner := by
  rw [root15Tail19, scan, if_neg (by decide), root15Candidates19_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail20
    (if 3 ≤ root15Candidates19.length then search edge 3 [61, 3049] root15Candidates19 3110 winner else winner) = _
  rw [if_neg (by decide)]
def root15Candidates20 : List Nat := [3733]
set_option maxRecDepth 8192 in
theorem root15Candidates20_checked : candidates edge 4 61 3169 root15Tail21 winner.bound = root15Candidates20 := by
  unfold candidates
  have row : edge 3169 = (SortedNatTree.contains FastRows.row3169) := by
    change Data.row3169.contains = (SortedNatTree.contains FastRows.row3169)
    exact FastRows.row3169_checked.symm
  rw [row]
  decide
theorem root15Step20 : scan edge (search edge 3) 4 [61] 61 root15Tail20 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail21 winner := by
  rw [root15Tail20, scan, if_neg (by decide), root15Candidates20_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail21
    (if 3 ≤ root15Candidates20.length then search edge 3 [61, 3169] root15Candidates20 3230 winner else winner) = _
  rw [if_neg (by decide)]
def root15Candidates21 : List Nat := [11587, 18517]
set_option maxRecDepth 8192 in
theorem root15Candidates21_checked : candidates edge 4 61 3181 root15Tail22 winner.bound = root15Candidates21 := by
  unfold candidates
  have row : edge 3181 = (SortedNatTree.contains FastRows.row3181) := by
    change Data.row3181.contains = (SortedNatTree.contains FastRows.row3181)
    exact FastRows.row3181_checked.symm
  rw [row]
  decide
theorem root15Step21 : scan edge (search edge 3) 4 [61] 61 root15Tail21 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail22 winner := by
  rw [root15Tail21, scan, if_neg (by decide), root15Candidates21_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail22
    (if 3 ≤ root15Candidates21.length then search edge 3 [61, 3181] root15Candidates21 3242 winner else winner) = _
  rw [if_neg (by decide)]
def root15Candidates22 : List Nat := [5869, 6079]
set_option maxRecDepth 8192 in
theorem root15Candidates22_checked : candidates edge 4 61 3463 root15Tail23 winner.bound = root15Candidates22 := by
  unfold candidates
  have row : edge 3463 = (SortedNatTree.contains FastRows.row3463) := by
    change Data.row3463.contains = (SortedNatTree.contains FastRows.row3463)
    exact FastRows.row3463_checked.symm
  rw [row]
  decide
theorem root15Step22 : scan edge (search edge 3) 4 [61] 61 root15Tail22 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail23 winner := by
  rw [root15Tail22, scan, if_neg (by decide), root15Candidates22_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail23
    (if 3 ≤ root15Candidates22.length then search edge 3 [61, 3463] root15Candidates22 3524 winner else winner) = _
  rw [if_neg (by decide)]
def root15Candidates23 : List Nat := [8311, 11839, 15643]
set_option maxRecDepth 8192 in
theorem root15Candidates23_checked : candidates edge 4 61 3469 root15Tail24 winner.bound = root15Candidates23 := by
  unfold candidates
  have row : edge 3469 = (SortedNatTree.contains FastRows.row3469) := by
    change Data.row3469.contains = (SortedNatTree.contains FastRows.row3469)
    exact FastRows.row3469_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root15Child23 : search edge 3 [61, 3469] root15Candidates23 3530 winner = winner := by decide
theorem root15Step23 : scan edge (search edge 3) 4 [61] 61 root15Tail23 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail24 winner := by
  rw [root15Tail23, scan, if_neg (by decide), root15Candidates23_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail24
    (if 3 ≤ root15Candidates23.length then search edge 3 [61, 3469] root15Candidates23 3530 winner else winner) = _
  rw [if_pos (by decide), root15Child23]
def root15Candidates24 : List Nat := [6871, 13309]
set_option maxRecDepth 8192 in
theorem root15Candidates24_checked : candidates edge 4 61 3637 root15Tail25 winner.bound = root15Candidates24 := by
  unfold candidates
  have row : edge 3637 = (SortedNatTree.contains FastRows.row3637) := by
    change Data.row3637.contains = (SortedNatTree.contains FastRows.row3637)
    exact FastRows.row3637_checked.symm
  rw [row]
  decide
theorem root15Step24 : scan edge (search edge 3) 4 [61] 61 root15Tail24 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail25 winner := by
  rw [root15Tail24, scan, if_neg (by decide), root15Candidates24_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail25
    (if 3 ≤ root15Candidates24.length then search edge 3 [61, 3637] root15Candidates24 3698 winner else winner) = _
  rw [if_neg (by decide)]
def root15Candidates25 : List Nat := [4657, 13309, 15643, 15973, 21121]
set_option maxRecDepth 8192 in
theorem root15Candidates25_checked : candidates edge 4 61 3733 root15Tail26 winner.bound = root15Candidates25 := by
  unfold candidates
  have row : edge 3733 = (SortedNatTree.contains FastRows.row3733) := by
    change Data.row3733.contains = (SortedNatTree.contains FastRows.row3733)
    exact FastRows.row3733_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root15Child25 : search edge 3 [61, 3733] root15Candidates25 3794 winner = winner := by decide
theorem root15Step25 : scan edge (search edge 3) 4 [61] 61 root15Tail25 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail26 winner := by
  rw [root15Tail25, scan, if_neg (by decide), root15Candidates25_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail26
    (if 3 ≤ root15Candidates25.length then search edge 3 [61, 3733] root15Candidates25 3794 winner else winner) = _
  rw [if_pos (by decide), root15Child25]
def root15Candidates26 : List Nat := [5821, 13729, 14887]
set_option maxRecDepth 8192 in
theorem root15Candidates26_checked : candidates edge 4 61 3889 root15Tail27 winner.bound = root15Candidates26 := by
  unfold candidates
  have row : edge 3889 = (SortedNatTree.contains FastRows.row3889) := by
    change Data.row3889.contains = (SortedNatTree.contains FastRows.row3889)
    exact FastRows.row3889_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root15Child26 : search edge 3 [61, 3889] root15Candidates26 3950 winner = winner := by decide
theorem root15Step26 : scan edge (search edge 3) 4 [61] 61 root15Tail26 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail27 winner := by
  rw [root15Tail26, scan, if_neg (by decide), root15Candidates26_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail27
    (if 3 ≤ root15Candidates26.length then search edge 3 [61, 3889] root15Candidates26 3950 winner else winner) = _
  rw [if_pos (by decide), root15Child26]
def root15Candidates27 : List Nat := [16057]
set_option maxRecDepth 8192 in
theorem root15Candidates27_checked : candidates edge 4 61 4657 root15Tail28 winner.bound = root15Candidates27 := by
  unfold candidates
  have row : edge 4657 = (SortedNatTree.contains FastRows.row4657) := by
    change Data.row4657.contains = (SortedNatTree.contains FastRows.row4657)
    exact FastRows.row4657_checked.symm
  rw [row]
  decide
theorem root15Step27 : scan edge (search edge 3) 4 [61] 61 root15Tail27 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail28 winner := by
  rw [root15Tail27, scan, if_neg (by decide), root15Candidates27_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail28
    (if 3 ≤ root15Candidates27.length then search edge 3 [61, 4657] root15Candidates27 4718 winner else winner) = _
  rw [if_neg (by decide)]
def root15Candidates28 : List Nat := [5821, 19543]
set_option maxRecDepth 8192 in
theorem root15Candidates28_checked : candidates edge 4 61 5413 root15Tail29 winner.bound = root15Candidates28 := by
  unfold candidates
  have row : edge 5413 = (SortedNatTree.contains FastRows.row5413) := by
    change Data.row5413.contains = (SortedNatTree.contains FastRows.row5413)
    exact FastRows.row5413_checked.symm
  rw [row]
  decide
theorem root15Step28 : scan edge (search edge 3) 4 [61] 61 root15Tail28 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail29 winner := by
  rw [root15Tail28, scan, if_neg (by decide), root15Candidates28_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail29
    (if 3 ≤ root15Candidates28.length then search edge 3 [61, 5413] root15Candidates28 5474 winner else winner) = _
  rw [if_neg (by decide)]
def root15Candidates29 : List Nat := [12541]
set_option maxRecDepth 8192 in
theorem root15Candidates29_checked : candidates edge 4 61 5431 root15Tail30 winner.bound = root15Candidates29 := by
  unfold candidates
  have row : edge 5431 = (SortedNatTree.contains FastRows.row5431) := by
    change Data.row5431.contains = (SortedNatTree.contains FastRows.row5431)
    exact FastRows.row5431_checked.symm
  rw [row]
  decide
theorem root15Step29 : scan edge (search edge 3) 4 [61] 61 root15Tail29 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail30 winner := by
  rw [root15Tail29, scan, if_neg (by decide), root15Candidates29_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail30
    (if 3 ≤ root15Candidates29.length then search edge 3 [61, 5431] root15Candidates29 5492 winner else winner) = _
  rw [if_neg (by decide)]
def root15Candidates30 : List Nat := [8581, 15643, 17239]
set_option maxRecDepth 8192 in
theorem root15Candidates30_checked : candidates edge 4 61 5449 root15Tail31 winner.bound = root15Candidates30 := by
  unfold candidates
  have row : edge 5449 = (SortedNatTree.contains FastRows.row5449) := by
    change Data.row5449.contains = (SortedNatTree.contains FastRows.row5449)
    exact FastRows.row5449_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root15Child30 : search edge 3 [61, 5449] root15Candidates30 5510 winner = winner := by decide
theorem root15Step30 : scan edge (search edge 3) 4 [61] 61 root15Tail30 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail31 winner := by
  rw [root15Tail30, scan, if_neg (by decide), root15Candidates30_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail31
    (if 3 ≤ root15Candidates30.length then search edge 3 [61, 5449] root15Candidates30 5510 winner else winner) = _
  rw [if_pos (by decide), root15Child30]
def root15Candidates31 : List Nat := [17959]
set_option maxRecDepth 8192 in
theorem root15Candidates31_checked : candidates edge 4 61 5557 root15Tail32 winner.bound = root15Candidates31 := by
  unfold candidates
  have row : edge 5557 = (SortedNatTree.contains FastRows.row5557) := by
    change Data.row5557.contains = (SortedNatTree.contains FastRows.row5557)
    exact FastRows.row5557_checked.symm
  rw [row]
  decide
theorem root15Step31 : scan edge (search edge 3) 4 [61] 61 root15Tail31 winner =
    scan edge (search edge 3) 4 [61] 61 root15Tail32 winner := by
  rw [root15Tail31, scan, if_neg (by decide), root15Candidates31_checked]
  change scan edge (search edge 3) 4 [61] 61 root15Tail32
    (if 3 ≤ root15Candidates31.length then search edge 3 [61, 5557] root15Candidates31 5618 winner else winner) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
