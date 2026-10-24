import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot049Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root49Candidates0 : List Nat := [1049, 1091, 3467, 6197, 8087, 23357]
set_option maxRecDepth 8192 in
theorem root49Candidates0_checked : candidates edge 4 239 263 root49Tail1 winner.bound = root49Candidates0 := by
  unfold candidates
  have row : edge 263 = (SortedNatTree.contains FastRows.row263) := by
    change Data.row263.contains = (SortedNatTree.contains FastRows.row263)
    exact FastRows.row263_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root49Child0 : search edge 3 [239, 263] root49Candidates0 502 winner = winner := by decide
theorem root49Step0 : scan edge (search edge 3) 4 [239] 239 root49Tail0 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail1 winner := by
  rw [root49Tail0, scan, if_neg (by decide), root49Candidates0_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail1
    (if 3 ≤ root49Candidates0.length then search edge 3 [239, 263] root49Candidates0 502 winner else winner) = _
  rw [if_pos (by decide), root49Child0]
def root49Candidates1 : List Nat := [1847, 6257, 10433, 11717, 20177]
set_option maxRecDepth 8192 in
theorem root49Candidates1_checked : candidates edge 4 239 347 root49Tail2 winner.bound = root49Candidates1 := by
  unfold candidates
  have row : edge 347 = (SortedNatTree.contains FastRows.row347) := by
    change Data.row347.contains = (SortedNatTree.contains FastRows.row347)
    exact FastRows.row347_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root49Child1 : search edge 3 [239, 347] root49Candidates1 586 winner = winner := by decide
theorem root49Step1 : scan edge (search edge 3) 4 [239] 239 root49Tail1 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail2 winner := by
  rw [root49Tail1, scan, if_neg (by decide), root49Candidates1_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail2
    (if 3 ≤ root49Candidates1.length then search edge 3 [239, 347] root49Candidates1 586 winner else winner) = _
  rw [if_pos (by decide), root49Child1]
def root49Candidates2 : List Nat := [17189, 25127]
set_option maxRecDepth 8192 in
theorem root49Candidates2_checked : candidates edge 4 239 461 root49Tail3 winner.bound = root49Candidates2 := by
  unfold candidates
  have row : edge 461 = (SortedNatTree.contains FastRows.row461) := by
    change Data.row461.contains = (SortedNatTree.contains FastRows.row461)
    exact FastRows.row461_checked.symm
  rw [row]
  decide
theorem root49Step2 : scan edge (search edge 3) 4 [239] 239 root49Tail2 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail3 winner := by
  rw [root49Tail2, scan, if_neg (by decide), root49Candidates2_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail3
    (if 3 ≤ root49Candidates2.length then search edge 3 [239, 461] root49Candidates2 700 winner else winner) = _
  rw [if_neg (by decide)]
def root49Candidates3 : List Nat := [947, 1091, 4241, 23099]
set_option maxRecDepth 8192 in
theorem root49Candidates3_checked : candidates edge 4 239 509 root49Tail4 winner.bound = root49Candidates3 := by
  unfold candidates
  have row : edge 509 = (SortedNatTree.contains FastRows.row509) := by
    change Data.row509.contains = (SortedNatTree.contains FastRows.row509)
    exact FastRows.row509_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root49Child3 : search edge 3 [239, 509] root49Candidates3 748 winner = winner := by decide
theorem root49Step3 : scan edge (search edge 3) 4 [239] 239 root49Tail3 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail4 winner := by
  rw [root49Tail3, scan, if_neg (by decide), root49Candidates3_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail4
    (if 3 ≤ root49Candidates3.length then search edge 3 [239, 509] root49Candidates3 748 winner else winner) = _
  rw [if_pos (by decide), root49Child3]
def root49Candidates4 : List Nat := [929, 4787, 6197, 23819]
set_option maxRecDepth 8192 in
theorem root49Candidates4_checked : candidates edge 4 239 641 root49Tail5 winner.bound = root49Candidates4 := by
  unfold candidates
  have row : edge 641 = (SortedNatTree.contains FastRows.row641) := by
    change Data.row641.contains = (SortedNatTree.contains FastRows.row641)
    exact FastRows.row641_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root49Child4 : search edge 3 [239, 641] root49Candidates4 880 winner = winner := by decide
theorem root49Step4 : scan edge (search edge 3) 4 [239] 239 root49Tail4 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail5 winner := by
  rw [root49Tail4, scan, if_neg (by decide), root49Candidates4_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail5
    (if 3 ≤ root49Candidates4.length then search edge 3 [239, 641] root49Candidates4 880 winner else winner) = _
  rw [if_pos (by decide), root49Child4]
def root49Candidates5 : List Nat := [3467, 3761, 14207]
set_option maxRecDepth 8192 in
theorem root49Candidates5_checked : candidates edge 4 239 929 root49Tail6 winner.bound = root49Candidates5 := by
  unfold candidates
  have row : edge 929 = (SortedNatTree.contains FastRows.row929) := by
    change Data.row929.contains = (SortedNatTree.contains FastRows.row929)
    exact FastRows.row929_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root49Child5 : search edge 3 [239, 929] root49Candidates5 1168 winner = winner := by decide
theorem root49Step5 : scan edge (search edge 3) 4 [239] 239 root49Tail5 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail6 winner := by
  rw [root49Tail5, scan, if_neg (by decide), root49Candidates5_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail6
    (if 3 ≤ root49Candidates5.length then search edge 3 [239, 929] root49Candidates5 1168 winner else winner) = _
  rw [if_pos (by decide), root49Child5]
def root49Candidates6 : List Nat := [1439, 8111]
set_option maxRecDepth 8192 in
theorem root49Candidates6_checked : candidates edge 4 239 947 root49Tail7 winner.bound = root49Candidates6 := by
  unfold candidates
  have row : edge 947 = (SortedNatTree.contains FastRows.row947) := by
    change Data.row947.contains = (SortedNatTree.contains FastRows.row947)
    exact FastRows.row947_checked.symm
  rw [row]
  decide
theorem root49Step6 : scan edge (search edge 3) 4 [239] 239 root49Tail6 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail7 winner := by
  rw [root49Tail6, scan, if_neg (by decide), root49Candidates6_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail7
    (if 3 ≤ root49Candidates6.length then search edge 3 [239, 947] root49Candidates6 1186 winner else winner) = _
  rw [if_neg (by decide)]
def root49Candidates7 : List Nat := [5039, 10079, 11717, 17909, 20177, 23957]
set_option maxRecDepth 8192 in
theorem root49Candidates7_checked : candidates edge 4 239 977 root49Tail8 winner.bound = root49Candidates7 := by
  unfold candidates
  have row : edge 977 = (SortedNatTree.contains FastRows.row977) := by
    change Data.row977.contains = (SortedNatTree.contains FastRows.row977)
    exact FastRows.row977_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root49Child7 : search edge 3 [239, 977] root49Candidates7 1216 winner = winner := by decide
theorem root49Step7 : scan edge (search edge 3) 4 [239] 239 root49Tail7 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail8 winner := by
  rw [root49Tail7, scan, if_neg (by decide), root49Candidates7_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail8
    (if 3 ≤ root49Candidates7.length then search edge 3 [239, 977] root49Candidates7 1216 winner else winner) = _
  rw [if_pos (by decide), root49Child7]
def root49Candidates8 : List Nat := [1091, 5039, 13103, 15077, 19319]
set_option maxRecDepth 8192 in
theorem root49Candidates8_checked : candidates edge 4 239 1013 root49Tail9 winner.bound = root49Candidates8 := by
  unfold candidates
  have row : edge 1013 = (SortedNatTree.contains FastRows.row1013) := by
    change Data.row1013.contains = (SortedNatTree.contains FastRows.row1013)
    exact FastRows.row1013_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root49Child8 : search edge 3 [239, 1013] root49Candidates8 1252 winner = winner := by decide
theorem root49Step8 : scan edge (search edge 3) 4 [239] 239 root49Tail8 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail9 winner := by
  rw [root49Tail8, scan, if_neg (by decide), root49Candidates8_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail9
    (if 3 ≤ root49Candidates8.length then search edge 3 [239, 1013] root49Candidates8 1252 winner else winner) = _
  rw [if_pos (by decide), root49Child8]
def root49Candidates9 : List Nat := [1847, 4787, 7523, 11057, 14081, 15887, 23957]
set_option maxRecDepth 8192 in
theorem root49Candidates9_checked : candidates edge 4 239 1049 root49Tail10 winner.bound = root49Candidates9 := by
  unfold candidates
  have row : edge 1049 = (SortedNatTree.contains FastRows.row1049) := by
    change Data.row1049.contains = (SortedNatTree.contains FastRows.row1049)
    exact FastRows.row1049_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root49Child9 : search edge 3 [239, 1049] root49Candidates9 1288 winner = winner := by decide
theorem root49Step9 : scan edge (search edge 3) 4 [239] 239 root49Tail9 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail10 winner := by
  rw [root49Tail9, scan, if_neg (by decide), root49Candidates9_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail10
    (if 3 ≤ root49Candidates9.length then search edge 3 [239, 1049] root49Candidates9 1288 winner else winner) = _
  rw [if_pos (by decide), root49Child9]
def root49Candidates10 : List Nat := [1847, 8111, 17909, 19319, 21773, 22193]
set_option maxRecDepth 8192 in
theorem root49Candidates10_checked : candidates edge 4 239 1091 root49Tail11 winner.bound = root49Candidates10 := by
  unfold candidates
  have row : edge 1091 = (SortedNatTree.contains FastRows.row1091) := by
    change Data.row1091.contains = (SortedNatTree.contains FastRows.row1091)
    exact FastRows.row1091_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root49Child10 : search edge 3 [239, 1091] root49Candidates10 1330 winner = winner := by decide
theorem root49Step10 : scan edge (search edge 3) 4 [239] 239 root49Tail10 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail11 winner := by
  rw [root49Tail10, scan, if_neg (by decide), root49Candidates10_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail11
    (if 3 ≤ root49Candidates10.length then search edge 3 [239, 1091] root49Candidates10 1330 winner else winner) = _
  rw [if_pos (by decide), root49Child10]
def root49Candidates11 : List Nat := [6659, 21383, 21773, 23957]
set_option maxRecDepth 8192 in
theorem root49Candidates11_checked : candidates edge 4 239 1439 root49Tail12 winner.bound = root49Candidates11 := by
  unfold candidates
  have row : edge 1439 = (SortedNatTree.contains FastRows.row1439) := by
    change Data.row1439.contains = (SortedNatTree.contains FastRows.row1439)
    exact FastRows.row1439_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root49Child11 : search edge 3 [239, 1439] root49Candidates11 1678 winner = winner := by decide
theorem root49Step11 : scan edge (search edge 3) 4 [239] 239 root49Tail11 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail12 winner := by
  rw [root49Tail11, scan, if_neg (by decide), root49Candidates11_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail12
    (if 3 ≤ root49Candidates11.length then search edge 3 [239, 1439] root49Candidates11 1678 winner else winner) = _
  rw [if_pos (by decide), root49Child11]
def root49Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root49Candidates12_checked : candidates edge 4 239 1637 root49Tail13 winner.bound = root49Candidates12 := by
  unfold candidates
  have row : edge 1637 = (SortedNatTree.contains FastRows.row1637) := by
    change Data.row1637.contains = (SortedNatTree.contains FastRows.row1637)
    exact FastRows.row1637_checked.symm
  rw [row]
  decide
theorem root49Step12 : scan edge (search edge 3) 4 [239] 239 root49Tail12 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail13 winner := by
  rw [root49Tail12, scan, if_neg (by decide), root49Candidates12_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail13
    (if 3 ≤ root49Candidates12.length then search edge 3 [239, 1637] root49Candidates12 1876 winner else winner) = _
  rw [if_neg (by decide)]
def root49Candidates13 : List Nat := [5927, 7523, 15299]
set_option maxRecDepth 8192 in
theorem root49Candidates13_checked : candidates edge 4 239 1847 root49Tail14 winner.bound = root49Candidates13 := by
  unfold candidates
  have row : edge 1847 = (SortedNatTree.contains FastRows.row1847) := by
    change Data.row1847.contains = (SortedNatTree.contains FastRows.row1847)
    exact FastRows.row1847_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root49Child13 : search edge 3 [239, 1847] root49Candidates13 2086 winner = winner := by decide
theorem root49Step13 : scan edge (search edge 3) 4 [239] 239 root49Tail13 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail14 winner := by
  rw [root49Tail13, scan, if_neg (by decide), root49Candidates13_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail14
    (if 3 ≤ root49Candidates13.length then search edge 3 [239, 1847] root49Candidates13 2086 winner else winner) = _
  rw [if_pos (by decide), root49Child13]
def root49Candidates14 : List Nat := [5927]
set_option maxRecDepth 8192 in
theorem root49Candidates14_checked : candidates edge 4 239 2333 root49Tail15 winner.bound = root49Candidates14 := by
  unfold candidates
  have row : edge 2333 = (SortedNatTree.contains FastRows.row2333) := by
    change Data.row2333.contains = (SortedNatTree.contains FastRows.row2333)
    exact FastRows.row2333_checked.symm
  rw [row]
  decide
theorem root49Step14 : scan edge (search edge 3) 4 [239] 239 root49Tail14 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail15 winner := by
  rw [root49Tail14, scan, if_neg (by decide), root49Candidates14_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail15
    (if 3 ≤ root49Candidates14.length then search edge 3 [239, 2333] root49Candidates14 2572 winner else winner) = _
  rw [if_neg (by decide)]
def root49Candidates15 : List Nat := [21011]
set_option maxRecDepth 8192 in
theorem root49Candidates15_checked : candidates edge 4 239 2447 root49Tail16 winner.bound = root49Candidates15 := by
  unfold candidates
  have row : edge 2447 = (SortedNatTree.contains FastRows.row2447) := by
    change Data.row2447.contains = (SortedNatTree.contains FastRows.row2447)
    exact FastRows.row2447_checked.symm
  rw [row]
  decide
theorem root49Step15 : scan edge (search edge 3) 4 [239] 239 root49Tail15 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail16 winner := by
  rw [root49Tail15, scan, if_neg (by decide), root49Candidates15_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail16
    (if 3 ≤ root49Candidates15.length then search edge 3 [239, 2447] root49Candidates15 2686 winner else winner) = _
  rw [if_neg (by decide)]
def root49Candidates16 : List Nat := [15077]
set_option maxRecDepth 8192 in
theorem root49Candidates16_checked : candidates edge 4 239 2459 root49Tail17 winner.bound = root49Candidates16 := by
  unfold candidates
  have row : edge 2459 = (SortedNatTree.contains FastRows.row2459) := by
    change Data.row2459.contains = (SortedNatTree.contains FastRows.row2459)
    exact FastRows.row2459_checked.symm
  rw [row]
  decide
theorem root49Step16 : scan edge (search edge 3) 4 [239] 239 root49Tail16 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail17 winner := by
  rw [root49Tail16, scan, if_neg (by decide), root49Candidates16_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail17
    (if 3 ≤ root49Candidates16.length then search edge 3 [239, 2459] root49Candidates16 2698 winner else winner) = _
  rw [if_neg (by decide)]
def root49Candidates17 : List Nat := [6131, 22193]
set_option maxRecDepth 8192 in
theorem root49Candidates17_checked : candidates edge 4 239 2729 root49Tail18 winner.bound = root49Candidates17 := by
  unfold candidates
  have row : edge 2729 = (SortedNatTree.contains FastRows.row2729) := by
    change Data.row2729.contains = (SortedNatTree.contains FastRows.row2729)
    exact FastRows.row2729_checked.symm
  rw [row]
  decide
theorem root49Step17 : scan edge (search edge 3) 4 [239] 239 root49Tail17 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail18 winner := by
  rw [root49Tail17, scan, if_neg (by decide), root49Candidates17_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail18
    (if 3 ≤ root49Candidates17.length then search edge 3 [239, 2729] root49Candidates17 2968 winner else winner) = _
  rw [if_neg (by decide)]
def root49Candidates18 : List Nat := [3257]
set_option maxRecDepth 8192 in
theorem root49Candidates18_checked : candidates edge 4 239 2861 root49Tail19 winner.bound = root49Candidates18 := by
  unfold candidates
  have row : edge 2861 = (SortedNatTree.contains FastRows.row2861) := by
    change Data.row2861.contains = (SortedNatTree.contains FastRows.row2861)
    exact FastRows.row2861_checked.symm
  rw [row]
  decide
theorem root49Step18 : scan edge (search edge 3) 4 [239] 239 root49Tail18 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail19 winner := by
  rw [root49Tail18, scan, if_neg (by decide), root49Candidates18_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail19
    (if 3 ≤ root49Candidates18.length then search edge 3 [239, 2861] root49Candidates18 3100 winner else winner) = _
  rw [if_neg (by decide)]
def root49Candidates19 : List Nat := []
set_option maxRecDepth 8192 in
theorem root49Candidates19_checked : candidates edge 4 239 3257 root49Tail20 winner.bound = root49Candidates19 := by
  unfold candidates
  have row : edge 3257 = (SortedNatTree.contains FastRows.row3257) := by
    change Data.row3257.contains = (SortedNatTree.contains FastRows.row3257)
    exact FastRows.row3257_checked.symm
  rw [row]
  decide
theorem root49Step19 : scan edge (search edge 3) 4 [239] 239 root49Tail19 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail20 winner := by
  rw [root49Tail19, scan, if_neg (by decide), root49Candidates19_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail20
    (if 3 ≤ root49Candidates19.length then search edge 3 [239, 3257] root49Candidates19 3496 winner else winner) = _
  rw [if_neg (by decide)]
def root49Candidates20 : List Nat := [5807, 8573]
set_option maxRecDepth 8192 in
theorem root49Candidates20_checked : candidates edge 4 239 3467 root49Tail21 winner.bound = root49Candidates20 := by
  unfold candidates
  have row : edge 3467 = (SortedNatTree.contains FastRows.row3467) := by
    change Data.row3467.contains = (SortedNatTree.contains FastRows.row3467)
    exact FastRows.row3467_checked.symm
  rw [row]
  decide
theorem root49Step20 : scan edge (search edge 3) 4 [239] 239 root49Tail20 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail21 winner := by
  rw [root49Tail20, scan, if_neg (by decide), root49Candidates20_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail21
    (if 3 ≤ root49Candidates20.length then search edge 3 [239, 3467] root49Candidates20 3706 winner else winner) = _
  rw [if_neg (by decide)]
def root49Candidates21 : List Nat := [5807, 7523, 8573]
set_option maxRecDepth 8192 in
theorem root49Candidates21_checked : candidates edge 4 239 3761 root49Tail22 winner.bound = root49Candidates21 := by
  unfold candidates
  have row : edge 3761 = (SortedNatTree.contains FastRows.row3761) := by
    change Data.row3761.contains = (SortedNatTree.contains FastRows.row3761)
    exact FastRows.row3761_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root49Child21 : search edge 3 [239, 3761] root49Candidates21 4000 winner = winner := by decide
theorem root49Step21 : scan edge (search edge 3) 4 [239] 239 root49Tail21 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail22 winner := by
  rw [root49Tail21, scan, if_neg (by decide), root49Candidates21_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail22
    (if 3 ≤ root49Candidates21.length then search edge 3 [239, 3761] root49Candidates21 4000 winner else winner) = _
  rw [if_pos (by decide), root49Child21]
def root49Candidates22 : List Nat := []
set_option maxRecDepth 8192 in
theorem root49Candidates22_checked : candidates edge 4 239 4157 root49Tail23 winner.bound = root49Candidates22 := by
  unfold candidates
  have row : edge 4157 = (SortedNatTree.contains FastRows.row4157) := by
    change Data.row4157.contains = (SortedNatTree.contains FastRows.row4157)
    exact FastRows.row4157_checked.symm
  rw [row]
  decide
theorem root49Step22 : scan edge (search edge 3) 4 [239] 239 root49Tail22 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail23 winner := by
  rw [root49Tail22, scan, if_neg (by decide), root49Candidates22_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail23
    (if 3 ≤ root49Candidates22.length then search edge 3 [239, 4157] root49Candidates22 4396 winner else winner) = _
  rw [if_neg (by decide)]
def root49Candidates23 : List Nat := [18743, 21011]
set_option maxRecDepth 8192 in
theorem root49Candidates23_checked : candidates edge 4 239 4241 root49Tail24 winner.bound = root49Candidates23 := by
  unfold candidates
  have row : edge 4241 = (SortedNatTree.contains FastRows.row4241) := by
    change Data.row4241.contains = (SortedNatTree.contains FastRows.row4241)
    exact FastRows.row4241_checked.symm
  rw [row]
  decide
theorem root49Step23 : scan edge (search edge 3) 4 [239] 239 root49Tail23 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail24 winner := by
  rw [root49Tail23, scan, if_neg (by decide), root49Candidates23_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail24
    (if 3 ≤ root49Candidates23.length then search edge 3 [239, 4241] root49Candidates23 4480 winner else winner) = _
  rw [if_neg (by decide)]
def root49Candidates24 : List Nat := [8573]
set_option maxRecDepth 8192 in
theorem root49Candidates24_checked : candidates edge 4 239 4787 root49Tail25 winner.bound = root49Candidates24 := by
  unfold candidates
  have row : edge 4787 = (SortedNatTree.contains FastRows.row4787) := by
    change Data.row4787.contains = (SortedNatTree.contains FastRows.row4787)
    exact FastRows.row4787_checked.symm
  rw [row]
  decide
theorem root49Step24 : scan edge (search edge 3) 4 [239] 239 root49Tail24 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail25 winner := by
  rw [root49Tail24, scan, if_neg (by decide), root49Candidates24_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail25
    (if 3 ≤ root49Candidates24.length then search edge 3 [239, 4787] root49Candidates24 5026 winner else winner) = _
  rw [if_neg (by decide)]
def root49Candidates25 : List Nat := [8111, 9539, 20177, 20201]
set_option maxRecDepth 8192 in
theorem root49Candidates25_checked : candidates edge 4 239 5039 root49Tail26 winner.bound = root49Candidates25 := by
  unfold candidates
  have row : edge 5039 = (SortedNatTree.contains FastRows.row5039) := by
    change Data.row5039.contains = (SortedNatTree.contains FastRows.row5039)
    exact FastRows.row5039_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root49Child25 : search edge 3 [239, 5039] root49Candidates25 5278 winner = winner := by decide
theorem root49Step25 : scan edge (search edge 3) 4 [239] 239 root49Tail25 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail26 winner := by
  rw [root49Tail25, scan, if_neg (by decide), root49Candidates25_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail26
    (if 3 ≤ root49Candidates25.length then search edge 3 [239, 5039] root49Candidates25 5278 winner else winner) = _
  rw [if_pos (by decide), root49Child25]
def root49Candidates26 : List Nat := [7823, 8243]
set_option maxRecDepth 8192 in
theorem root49Candidates26_checked : candidates edge 4 239 5303 root49Tail27 winner.bound = root49Candidates26 := by
  unfold candidates
  have row : edge 5303 = (SortedNatTree.contains FastRows.row5303) := by
    change Data.row5303.contains = (SortedNatTree.contains FastRows.row5303)
    exact FastRows.row5303_checked.symm
  rw [row]
  decide
theorem root49Step26 : scan edge (search edge 3) 4 [239] 239 root49Tail26 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail27 winner := by
  rw [root49Tail26, scan, if_neg (by decide), root49Candidates26_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail27
    (if 3 ≤ root49Candidates26.length then search edge 3 [239, 5303] root49Candidates26 5542 winner else winner) = _
  rw [if_neg (by decide)]
def root49Candidates27 : List Nat := []
set_option maxRecDepth 8192 in
theorem root49Candidates27_checked : candidates edge 4 239 5807 root49Tail28 winner.bound = root49Candidates27 := by
  unfold candidates
  have row : edge 5807 = (SortedNatTree.contains FastRows.row5807) := by
    change Data.row5807.contains = (SortedNatTree.contains FastRows.row5807)
    exact FastRows.row5807_checked.symm
  rw [row]
  decide
theorem root49Step27 : scan edge (search edge 3) 4 [239] 239 root49Tail27 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail28 winner := by
  rw [root49Tail27, scan, if_neg (by decide), root49Candidates27_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail28
    (if 3 ≤ root49Candidates27.length then search edge 3 [239, 5807] root49Candidates27 6046 winner else winner) = _
  rw [if_neg (by decide)]
def root49Candidates28 : List Nat := [15299, 17909]
set_option maxRecDepth 8192 in
theorem root49Candidates28_checked : candidates edge 4 239 5879 root49Tail29 winner.bound = root49Candidates28 := by
  unfold candidates
  have row : edge 5879 = (SortedNatTree.contains FastRows.row5879) := by
    change Data.row5879.contains = (SortedNatTree.contains FastRows.row5879)
    exact FastRows.row5879_checked.symm
  rw [row]
  decide
theorem root49Step28 : scan edge (search edge 3) 4 [239] 239 root49Tail28 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail29 winner := by
  rw [root49Tail28, scan, if_neg (by decide), root49Candidates28_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail29
    (if 3 ≤ root49Candidates28.length then search edge 3 [239, 5879] root49Candidates28 6118 winner else winner) = _
  rw [if_neg (by decide)]
def root49Candidates29 : List Nat := [10247, 18773]
set_option maxRecDepth 8192 in
theorem root49Candidates29_checked : candidates edge 4 239 5927 root49Tail30 winner.bound = root49Candidates29 := by
  unfold candidates
  have row : edge 5927 = (SortedNatTree.contains FastRows.row5927) := by
    change Data.row5927.contains = (SortedNatTree.contains FastRows.row5927)
    exact FastRows.row5927_checked.symm
  rw [row]
  decide
theorem root49Step29 : scan edge (search edge 3) 4 [239] 239 root49Tail29 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail30 winner := by
  rw [root49Tail29, scan, if_neg (by decide), root49Candidates29_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail30
    (if 3 ≤ root49Candidates29.length then search edge 3 [239, 5927] root49Candidates29 6166 winner else winner) = _
  rw [if_neg (by decide)]
def root49Candidates30 : List Nat := [19319]
set_option maxRecDepth 8192 in
theorem root49Candidates30_checked : candidates edge 4 239 6131 root49Tail31 winner.bound = root49Candidates30 := by
  unfold candidates
  have row : edge 6131 = (SortedNatTree.contains FastRows.row6131) := by
    change Data.row6131.contains = (SortedNatTree.contains FastRows.row6131)
    exact FastRows.row6131_checked.symm
  rw [row]
  decide
theorem root49Step30 : scan edge (search edge 3) 4 [239] 239 root49Tail30 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail31 winner := by
  rw [root49Tail30, scan, if_neg (by decide), root49Candidates30_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail31
    (if 3 ≤ root49Candidates30.length then search edge 3 [239, 6131] root49Candidates30 6370 winner else winner) = _
  rw [if_neg (by decide)]
def root49Candidates31 : List Nat := []
set_option maxRecDepth 8192 in
theorem root49Candidates31_checked : candidates edge 4 239 6197 root49Tail32 winner.bound = root49Candidates31 := by
  unfold candidates
  have row : edge 6197 = (SortedNatTree.contains FastRows.row6197) := by
    change Data.row6197.contains = (SortedNatTree.contains FastRows.row6197)
    exact FastRows.row6197_checked.symm
  rw [row]
  decide
theorem root49Step31 : scan edge (search edge 3) 4 [239] 239 root49Tail31 winner =
    scan edge (search edge 3) 4 [239] 239 root49Tail32 winner := by
  rw [root49Tail31, scan, if_neg (by decide), root49Candidates31_checked]
  change scan edge (search edge 3) 4 [239] 239 root49Tail32
    (if 3 ≤ root49Candidates31.length then search edge 3 [239, 6197] root49Candidates31 6436 winner else winner) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
