import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot005Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root5Candidates0 : List Nat := [181, 991, 1237, 1447, 4723, 6073, 6271, 7741, 8641, 9049, 9601, 9679, 19843]
set_option maxRecDepth 8192 in
theorem root5Candidates0_checked : candidates edge 4 19 31 root5Tail1 winner.bound = root5Candidates0 := by
  unfold candidates
  have row : edge 31 = (SortedNatTree.contains FastRows.row31) := by
    change Data.row31.contains = (SortedNatTree.contains FastRows.row31)
    exact FastRows.row31_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child0 : search edge 3 [19, 31] root5Candidates0 50 winner = winner := by decide
theorem root5Step0 : scan edge (search edge 3) 4 [19] 19 root5Tail0 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail1 winner := by
  rw [root5Tail0, scan, if_neg (by decide), root5Candidates0_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail1
    (if 3 ≤ root5Candidates0.length then search edge 3 [19, 31] root5Candidates0 50 winner else winner) = _
  rw [if_pos (by decide), root5Child0]
def root5Candidates1 : List Nat := [1447, 1657, 5077, 5659, 5791, 8179, 9679, 10687, 18517]
set_option maxRecDepth 8192 in
theorem root5Candidates1_checked : candidates edge 4 19 79 root5Tail2 winner.bound = root5Candidates1 := by
  unfold candidates
  have row : edge 79 = (SortedNatTree.contains FastRows.row79) := by
    change Data.row79.contains = (SortedNatTree.contains FastRows.row79)
    exact FastRows.row79_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child1 : search edge 3 [19, 79] root5Candidates1 98 winner = winner := by decide
theorem root5Step1 : scan edge (search edge 3) 4 [19] 19 root5Tail1 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail2 winner := by
  rw [root5Tail1, scan, if_neg (by decide), root5Candidates1_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail2
    (if 3 ≤ root5Candidates1.length then search edge 3 [19, 79] root5Candidates1 98 winner else winner) = _
  rw [if_pos (by decide), root5Child1]
def root5Candidates2 : List Nat := [1039, 3727, 4507, 6271, 6991, 7507, 25801]
set_option maxRecDepth 8192 in
theorem root5Candidates2_checked : candidates edge 4 19 97 root5Tail3 winner.bound = root5Candidates2 := by
  unfold candidates
  have row : edge 97 = (SortedNatTree.contains FastRows.row97) := by
    change Data.row97.contains = (SortedNatTree.contains FastRows.row97)
    exact FastRows.row97_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child2 : search edge 3 [19, 97] root5Candidates2 116 winner = winner := by decide
theorem root5Step2 : scan edge (search edge 3) 4 [19] 19 root5Tail2 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail3 winner := by
  rw [root5Tail2, scan, if_neg (by decide), root5Candidates2_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail3
    (if 3 ≤ root5Candidates2.length then search edge 3 [19, 97] root5Candidates2 116 winner else winner) = _
  rw [if_pos (by decide), root5Child2]
def root5Candidates3 : List Nat := [997, 8599, 8719, 8929, 12739]
set_option maxRecDepth 8192 in
theorem root5Candidates3_checked : candidates edge 4 19 163 root5Tail4 winner.bound = root5Candidates3 := by
  unfold candidates
  have row : edge 163 = (SortedNatTree.contains FastRows.row163) := by
    change Data.row163.contains = (SortedNatTree.contains FastRows.row163)
    exact FastRows.row163_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child3 : search edge 3 [19, 163] root5Candidates3 182 winner = winner := by decide
theorem root5Step3 : scan edge (search edge 3) 4 [19] 19 root5Tail3 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail4 winner := by
  rw [root5Tail3, scan, if_neg (by decide), root5Candidates3_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail4
    (if 3 ≤ root5Candidates3.length then search edge 3 [19, 163] root5Candidates3 182 winner else winner) = _
  rw [if_pos (by decide), root5Child3]
def root5Candidates4 : List Nat := [1657, 9679]
set_option maxRecDepth 8192 in
theorem root5Candidates4_checked : candidates edge 4 19 181 root5Tail5 winner.bound = root5Candidates4 := by
  unfold candidates
  have row : edge 181 = (SortedNatTree.contains FastRows.row181) := by
    change Data.row181.contains = (SortedNatTree.contains FastRows.row181)
    exact FastRows.row181_checked.symm
  rw [row]
  decide
theorem root5Step4 : scan edge (search edge 3) 4 [19] 19 root5Tail4 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail5 winner := by
  rw [root5Tail4, scan, if_neg (by decide), root5Candidates4_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail5
    (if 3 ≤ root5Candidates4.length then search edge 3 [19, 181] root5Candidates4 200 winner else winner) = _
  rw [if_neg (by decide)]
def root5Candidates5 : List Nat := [571, 23671]
set_option maxRecDepth 8192 in
theorem root5Candidates5_checked : candidates edge 4 19 433 root5Tail6 winner.bound = root5Candidates5 := by
  unfold candidates
  have row : edge 433 = (SortedNatTree.contains FastRows.row433) := by
    change Data.row433.contains = (SortedNatTree.contains FastRows.row433)
    exact FastRows.row433_checked.symm
  rw [row]
  decide
theorem root5Step5 : scan edge (search edge 3) 4 [19] 19 root5Tail5 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail6 winner := by
  rw [root5Tail5, scan, if_neg (by decide), root5Candidates5_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail6
    (if 3 ≤ root5Candidates5.length then search edge 3 [19, 433] root5Candidates5 452 winner else winner) = _
  rw [if_neg (by decide)]
def root5Candidates6 : List Nat := [7297, 8641, 13627, 14947, 24547]
set_option maxRecDepth 8192 in
theorem root5Candidates6_checked : candidates edge 4 19 571 root5Tail7 winner.bound = root5Candidates6 := by
  unfold candidates
  have row : edge 571 = (SortedNatTree.contains FastRows.row571) := by
    change Data.row571.contains = (SortedNatTree.contains FastRows.row571)
    exact FastRows.row571_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child6 : search edge 3 [19, 571] root5Candidates6 590 winner = winner := by decide
theorem root5Step6 : scan edge (search edge 3) 4 [19] 19 root5Tail6 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail7 winner := by
  rw [root5Tail6, scan, if_neg (by decide), root5Candidates6_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail7
    (if 3 ≤ root5Candidates6.length then search edge 3 [19, 571] root5Candidates6 590 winner else winner) = _
  rw [if_pos (by decide), root5Child6]
def root5Candidates7 : List Nat := [937, 2539, 9343, 10567, 10663, 12583, 14947]
set_option maxRecDepth 8192 in
theorem root5Candidates7_checked : candidates edge 4 19 577 root5Tail8 winner.bound = root5Candidates7 := by
  unfold candidates
  have row : edge 577 = (SortedNatTree.contains FastRows.row577) := by
    change Data.row577.contains = (SortedNatTree.contains FastRows.row577)
    exact FastRows.row577_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child7 : search edge 3 [19, 577] root5Candidates7 596 winner = winner := by decide
theorem root5Step7 : scan edge (search edge 3) 4 [19] 19 root5Tail7 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail8 winner := by
  rw [root5Tail7, scan, if_neg (by decide), root5Candidates7_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail8
    (if 3 ≤ root5Candidates7.length then search edge 3 [19, 577] root5Candidates7 596 winner else winner) = _
  rw [if_pos (by decide), root5Child7]
def root5Candidates8 : List Nat := [10729, 20173]
set_option maxRecDepth 8192 in
theorem root5Candidates8_checked : candidates edge 4 19 709 root5Tail9 winner.bound = root5Candidates8 := by
  unfold candidates
  have row : edge 709 = (SortedNatTree.contains FastRows.row709) := by
    change Data.row709.contains = (SortedNatTree.contains FastRows.row709)
    exact FastRows.row709_checked.symm
  rw [row]
  decide
theorem root5Step8 : scan edge (search edge 3) 4 [19] 19 root5Tail8 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail9 winner := by
  rw [root5Tail8, scan, if_neg (by decide), root5Candidates8_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail9
    (if 3 ≤ root5Candidates8.length then search edge 3 [19, 709] root5Candidates8 728 winner else winner) = _
  rw [if_neg (by decide)]
def root5Candidates9 : List Nat := [997, 1237, 1669]
set_option maxRecDepth 8192 in
theorem root5Candidates9_checked : candidates edge 4 19 727 root5Tail10 winner.bound = root5Candidates9 := by
  unfold candidates
  have row : edge 727 = (SortedNatTree.contains FastRows.row727) := by
    change Data.row727.contains = (SortedNatTree.contains FastRows.row727)
    exact FastRows.row727_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child9 : search edge 3 [19, 727] root5Candidates9 746 winner = winner := by decide
theorem root5Step9 : scan edge (search edge 3) 4 [19] 19 root5Tail9 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail10 winner := by
  rw [root5Tail9, scan, if_neg (by decide), root5Candidates9_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail10
    (if 3 ≤ root5Candidates9.length then search edge 3 [19, 727] root5Candidates9 746 winner else winner) = _
  rw [if_pos (by decide), root5Child9]
def root5Candidates10 : List Nat := [2053, 2791, 3943, 5791, 9679, 14389, 15241, 23671]
set_option maxRecDepth 8192 in
theorem root5Candidates10_checked : candidates edge 4 19 937 root5Tail11 winner.bound = root5Candidates10 := by
  unfold candidates
  have row : edge 937 = (SortedNatTree.contains FastRows.row937) := by
    change Data.row937.contains = (SortedNatTree.contains FastRows.row937)
    exact FastRows.row937_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child10 : search edge 3 [19, 937] root5Candidates10 956 winner = winner := by decide
theorem root5Step10 : scan edge (search edge 3) 4 [19] 19 root5Tail10 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail11 winner := by
  rw [root5Tail10, scan, if_neg (by decide), root5Candidates10_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail11
    (if 3 ≤ root5Candidates10.length then search edge 3 [19, 937] root5Candidates10 956 winner else winner) = _
  rw [if_pos (by decide), root5Child10]
def root5Candidates11 : List Nat := [1861, 3727, 7759, 8641, 9697, 22027]
set_option maxRecDepth 8192 in
theorem root5Candidates11_checked : candidates edge 4 19 991 root5Tail12 winner.bound = root5Candidates11 := by
  unfold candidates
  have row : edge 991 = (SortedNatTree.contains FastRows.row991) := by
    change Data.row991.contains = (SortedNatTree.contains FastRows.row991)
    exact FastRows.row991_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child11 : search edge 3 [19, 991] root5Candidates11 1010 winner = winner := by decide
theorem root5Step11 : scan edge (search edge 3) 4 [19] 19 root5Tail11 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail12 winner := by
  rw [root5Tail11, scan, if_neg (by decide), root5Candidates11_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail12
    (if 3 ≤ root5Candidates11.length then search edge 3 [19, 991] root5Candidates11 1010 winner else winner) = _
  rw [if_pos (by decide), root5Child11]
def root5Candidates12 : List Nat := [7741, 12577, 19471]
set_option maxRecDepth 8192 in
theorem root5Candidates12_checked : candidates edge 4 19 997 root5Tail13 winner.bound = root5Candidates12 := by
  unfold candidates
  have row : edge 997 = (SortedNatTree.contains FastRows.row997) := by
    change Data.row997.contains = (SortedNatTree.contains FastRows.row997)
    exact FastRows.row997_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child12 : search edge 3 [19, 997] root5Candidates12 1016 winner = winner := by decide
theorem root5Step12 : scan edge (search edge 3) 4 [19] 19 root5Tail12 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail13 winner := by
  rw [root5Tail12, scan, if_neg (by decide), root5Candidates12_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail13
    (if 3 ≤ root5Candidates12.length then search edge 3 [19, 997] root5Candidates12 1016 winner else winner) = _
  rw [if_pos (by decide), root5Child12]
def root5Candidates13 : List Nat := [1249, 5869, 8179, 8719, 9601, 10159, 12457, 18517, 20521]
set_option maxRecDepth 8192 in
theorem root5Candidates13_checked : candidates edge 4 19 1033 root5Tail14 winner.bound = root5Candidates13 := by
  unfold candidates
  have row : edge 1033 = (SortedNatTree.contains FastRows.row1033) := by
    change Data.row1033.contains = (SortedNatTree.contains FastRows.row1033)
    exact FastRows.row1033_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child13 : search edge 3 [19, 1033] root5Candidates13 1052 winner = winner := by decide
theorem root5Step13 : scan edge (search edge 3) 4 [19] 19 root5Tail13 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail14 winner := by
  rw [root5Tail13, scan, if_neg (by decide), root5Candidates13_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail14
    (if 3 ≤ root5Candidates13.length then search edge 3 [19, 1033] root5Candidates13 1052 winner else winner) = _
  rw [if_pos (by decide), root5Child13]
def root5Candidates14 : List Nat := [4723, 6073, 6709, 12583, 18517]
set_option maxRecDepth 8192 in
theorem root5Candidates14_checked : candidates edge 4 19 1039 root5Tail15 winner.bound = root5Candidates14 := by
  unfold candidates
  have row : edge 1039 = (SortedNatTree.contains FastRows.row1039) := by
    change Data.row1039.contains = (SortedNatTree.contains FastRows.row1039)
    exact FastRows.row1039_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child14 : search edge 3 [19, 1039] root5Candidates14 1058 winner = winner := by decide
theorem root5Step14 : scan edge (search edge 3) 4 [19] 19 root5Tail14 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail15 winner := by
  rw [root5Tail14, scan, if_neg (by decide), root5Candidates14_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail15
    (if 3 ≤ root5Candidates14.length then search edge 3 [19, 1039] root5Candidates14 1058 winner else winner) = _
  rw [if_pos (by decide), root5Child14]
def root5Candidates15 : List Nat := [1657, 6271, 9343, 9739, 13999, 16183]
set_option maxRecDepth 8192 in
theorem root5Candidates15_checked : candidates edge 4 19 1237 root5Tail16 winner.bound = root5Candidates15 := by
  unfold candidates
  have row : edge 1237 = (SortedNatTree.contains FastRows.row1237) := by
    change Data.row1237.contains = (SortedNatTree.contains FastRows.row1237)
    exact FastRows.row1237_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child15 : search edge 3 [19, 1237] root5Candidates15 1256 winner = winner := by decide
theorem root5Step15 : scan edge (search edge 3) 4 [19] 19 root5Tail15 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail16 winner := by
  rw [root5Tail15, scan, if_neg (by decide), root5Candidates15_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail16
    (if 3 ≤ root5Candidates15.length then search edge 3 [19, 1237] root5Candidates15 1256 winner else winner) = _
  rw [if_pos (by decide), root5Child15]
def root5Candidates16 : List Nat := [3727, 8641]
set_option maxRecDepth 8192 in
theorem root5Candidates16_checked : candidates edge 4 19 1249 root5Tail17 winner.bound = root5Candidates16 := by
  unfold candidates
  have row : edge 1249 = (SortedNatTree.contains FastRows.row1249) := by
    change Data.row1249.contains = (SortedNatTree.contains FastRows.row1249)
    exact FastRows.row1249_checked.symm
  rw [row]
  decide
theorem root5Step16 : scan edge (search edge 3) 4 [19] 19 root5Tail16 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail17 winner := by
  rw [root5Tail16, scan, if_neg (by decide), root5Candidates16_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail17
    (if 3 ≤ root5Candidates16.length then search edge 3 [19, 1249] root5Candidates16 1268 winner else winner) = _
  rw [if_neg (by decide)]
def root5Candidates17 : List Nat := [9601, 9697, 12577, 18451]
set_option maxRecDepth 8192 in
theorem root5Candidates17_checked : candidates edge 4 19 1297 root5Tail18 winner.bound = root5Candidates17 := by
  unfold candidates
  have row : edge 1297 = (SortedNatTree.contains FastRows.row1297) := by
    change Data.row1297.contains = (SortedNatTree.contains FastRows.row1297)
    exact FastRows.row1297_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child17 : search edge 3 [19, 1297] root5Candidates17 1316 winner = winner := by decide
theorem root5Step17 : scan edge (search edge 3) 4 [19] 19 root5Tail17 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail18 winner := by
  rw [root5Tail17, scan, if_neg (by decide), root5Candidates17_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail18
    (if 3 ≤ root5Candidates17.length then search edge 3 [19, 1297] root5Candidates17 1316 winner else winner) = _
  rw [if_pos (by decide), root5Child17]
def root5Candidates18 : List Nat := [6991, 7759, 15619, 18517]
set_option maxRecDepth 8192 in
theorem root5Candidates18_checked : candidates edge 4 19 1447 root5Tail19 winner.bound = root5Candidates18 := by
  unfold candidates
  have row : edge 1447 = (SortedNatTree.contains FastRows.row1447) := by
    change Data.row1447.contains = (SortedNatTree.contains FastRows.row1447)
    exact FastRows.row1447_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child18 : search edge 3 [19, 1447] root5Candidates18 1466 winner = winner := by decide
theorem root5Step18 : scan edge (search edge 3) 4 [19] 19 root5Tail18 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail19 winner := by
  rw [root5Tail18, scan, if_neg (by decide), root5Candidates18_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail19
    (if 3 ≤ root5Candidates18.length then search edge 3 [19, 1447] root5Candidates18 1466 winner else winner) = _
  rw [if_pos (by decide), root5Child18]
def root5Candidates19 : List Nat := [3943, 6073, 8971, 9403]
set_option maxRecDepth 8192 in
theorem root5Candidates19_checked : candidates edge 4 19 1621 root5Tail20 winner.bound = root5Candidates19 := by
  unfold candidates
  have row : edge 1621 = (SortedNatTree.contains FastRows.row1621) := by
    change Data.row1621.contains = (SortedNatTree.contains FastRows.row1621)
    exact FastRows.row1621_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child19 : search edge 3 [19, 1621] root5Candidates19 1640 winner = winner := by decide
theorem root5Step19 : scan edge (search edge 3) 4 [19] 19 root5Tail19 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail20 winner := by
  rw [root5Tail19, scan, if_neg (by decide), root5Candidates19_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail20
    (if 3 ≤ root5Candidates19.length then search edge 3 [19, 1621] root5Candidates19 1640 winner else winner) = _
  rw [if_pos (by decide), root5Child19]
def root5Candidates20 : List Nat := [1699, 2791, 6907, 8623, 12739, 14947, 20173]
set_option maxRecDepth 8192 in
theorem root5Candidates20_checked : candidates edge 4 19 1657 root5Tail21 winner.bound = root5Candidates20 := by
  unfold candidates
  have row : edge 1657 = (SortedNatTree.contains FastRows.row1657) := by
    change Data.row1657.contains = (SortedNatTree.contains FastRows.row1657)
    exact FastRows.row1657_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child20 : search edge 3 [19, 1657] root5Candidates20 1676 winner = winner := by decide
theorem root5Step20 : scan edge (search edge 3) 4 [19] 19 root5Tail20 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail21 winner := by
  rw [root5Tail20, scan, if_neg (by decide), root5Candidates20_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail21
    (if 3 ≤ root5Candidates20.length then search edge 3 [19, 1657] root5Candidates20 1676 winner else winner) = _
  rw [if_pos (by decide), root5Child20]
def root5Candidates21 : List Nat := [2791, 5659, 6661, 6709, 8599, 10729, 12457, 22153]
set_option maxRecDepth 8192 in
theorem root5Candidates21_checked : candidates edge 4 19 1669 root5Tail22 winner.bound = root5Candidates21 := by
  unfold candidates
  have row : edge 1669 = (SortedNatTree.contains FastRows.row1669) := by
    change Data.row1669.contains = (SortedNatTree.contains FastRows.row1669)
    exact FastRows.row1669_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child21 : search edge 3 [19, 1669] root5Candidates21 1688 winner = winner := by decide
theorem root5Step21 : scan edge (search edge 3) 4 [19] 19 root5Tail21 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail22 winner := by
  rw [root5Tail21, scan, if_neg (by decide), root5Candidates21_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail22
    (if 3 ≤ root5Candidates21.length then search edge 3 [19, 1669] root5Candidates21 1688 winner else winner) = _
  rw [if_pos (by decide), root5Child21]
def root5Candidates22 : List Nat := [6991, 20521]
set_option maxRecDepth 8192 in
theorem root5Candidates22_checked : candidates edge 4 19 1693 root5Tail23 winner.bound = root5Candidates22 := by
  unfold candidates
  have row : edge 1693 = (SortedNatTree.contains FastRows.row1693) := by
    change Data.row1693.contains = (SortedNatTree.contains FastRows.row1693)
    exact FastRows.row1693_checked.symm
  rw [row]
  decide
theorem root5Step22 : scan edge (search edge 3) 4 [19] 19 root5Tail22 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail23 winner := by
  rw [root5Tail22, scan, if_neg (by decide), root5Candidates22_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail23
    (if 3 ≤ root5Candidates22.length then search edge 3 [19, 1693] root5Candidates22 1712 winner else winner) = _
  rw [if_neg (by decide)]
def root5Candidates23 : List Nat := [1861, 5659, 5791, 7507, 12577, 17569]
set_option maxRecDepth 8192 in
theorem root5Candidates23_checked : candidates edge 4 19 1699 root5Tail24 winner.bound = root5Candidates23 := by
  unfold candidates
  have row : edge 1699 = (SortedNatTree.contains FastRows.row1699) := by
    change Data.row1699.contains = (SortedNatTree.contains FastRows.row1699)
    exact FastRows.row1699_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child23 : search edge 3 [19, 1699] root5Candidates23 1718 winner = winner := by decide
theorem root5Step23 : scan edge (search edge 3) 4 [19] 19 root5Tail23 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail24 winner := by
  rw [root5Tail23, scan, if_neg (by decide), root5Candidates23_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail24
    (if 3 ≤ root5Candidates23.length then search edge 3 [19, 1699] root5Candidates23 1718 winner else winner) = _
  rw [if_pos (by decide), root5Child23]
def root5Candidates24 : List Nat := [8941, 10663, 10729, 13003, 14389, 15817, 22447]
set_option maxRecDepth 8192 in
theorem root5Candidates24_checked : candidates edge 4 19 1831 root5Tail25 winner.bound = root5Candidates24 := by
  unfold candidates
  have row : edge 1831 = (SortedNatTree.contains FastRows.row1831) := by
    change Data.row1831.contains = (SortedNatTree.contains FastRows.row1831)
    exact FastRows.row1831_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child24 : search edge 3 [19, 1831] root5Candidates24 1850 winner = winner := by decide
theorem root5Step24 : scan edge (search edge 3) 4 [19] 19 root5Tail24 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail25 winner := by
  rw [root5Tail24, scan, if_neg (by decide), root5Candidates24_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail25
    (if 3 ≤ root5Candidates24.length then search edge 3 [19, 1831] root5Candidates24 1850 winner else winner) = _
  rw [if_pos (by decide), root5Child24]
def root5Candidates25 : List Nat := [2791, 6907]
set_option maxRecDepth 8192 in
theorem root5Candidates25_checked : candidates edge 4 19 1861 root5Tail26 winner.bound = root5Candidates25 := by
  unfold candidates
  have row : edge 1861 = (SortedNatTree.contains FastRows.row1861) := by
    change Data.row1861.contains = (SortedNatTree.contains FastRows.row1861)
    exact FastRows.row1861_checked.symm
  rw [row]
  decide
theorem root5Step25 : scan edge (search edge 3) 4 [19] 19 root5Tail25 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail26 winner := by
  rw [root5Tail25, scan, if_neg (by decide), root5Candidates25_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail26
    (if 3 ≤ root5Candidates25.length then search edge 3 [19, 1861] root5Candidates25 1880 winner else winner) = _
  rw [if_neg (by decide)]
def root5Candidates26 : List Nat := [13627]
set_option maxRecDepth 8192 in
theorem root5Candidates26_checked : candidates edge 4 19 2053 root5Tail27 winner.bound = root5Candidates26 := by
  unfold candidates
  have row : edge 2053 = (SortedNatTree.contains FastRows.row2053) := by
    change Data.row2053.contains = (SortedNatTree.contains FastRows.row2053)
    exact FastRows.row2053_checked.symm
  rw [row]
  decide
theorem root5Step26 : scan edge (search edge 3) 4 [19] 19 root5Tail26 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail27 winner := by
  rw [root5Tail26, scan, if_neg (by decide), root5Candidates26_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail27
    (if 3 ≤ root5Candidates26.length then search edge 3 [19, 2053] root5Candidates26 2072 winner else winner) = _
  rw [if_neg (by decide)]
def root5Candidates27 : List Nat := [3727, 4507, 18451]
set_option maxRecDepth 8192 in
theorem root5Candidates27_checked : candidates edge 4 19 2113 root5Tail28 winner.bound = root5Candidates27 := by
  unfold candidates
  have row : edge 2113 = (SortedNatTree.contains FastRows.row2113) := by
    change Data.row2113.contains = (SortedNatTree.contains FastRows.row2113)
    exact FastRows.row2113_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child27 : search edge 3 [19, 2113] root5Candidates27 2132 winner = winner := by decide
theorem root5Step27 : scan edge (search edge 3) 4 [19] 19 root5Tail27 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail28 winner := by
  rw [root5Tail27, scan, if_neg (by decide), root5Candidates27_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail28
    (if 3 ≤ root5Candidates27.length then search edge 3 [19, 2113] root5Candidates27 2132 winner else winner) = _
  rw [if_pos (by decide), root5Child27]
def root5Candidates28 : List Nat := [7759, 9721, 10177, 10729]
set_option maxRecDepth 8192 in
theorem root5Candidates28_checked : candidates edge 4 19 2161 root5Tail29 winner.bound = root5Candidates28 := by
  unfold candidates
  have row : edge 2161 = (SortedNatTree.contains FastRows.row2161) := by
    change Data.row2161.contains = (SortedNatTree.contains FastRows.row2161)
    exact FastRows.row2161_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child28 : search edge 3 [19, 2161] root5Candidates28 2180 winner = winner := by decide
theorem root5Step28 : scan edge (search edge 3) 4 [19] 19 root5Tail28 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail29 winner := by
  rw [root5Tail28, scan, if_neg (by decide), root5Candidates28_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail29
    (if 3 ≤ root5Candidates28.length then search edge 3 [19, 2161] root5Candidates28 2180 winner else winner) = _
  rw [if_pos (by decide), root5Child28]
def root5Candidates29 : List Nat := []
set_option maxRecDepth 8192 in
theorem root5Candidates29_checked : candidates edge 4 19 2239 root5Tail30 winner.bound = root5Candidates29 := by
  unfold candidates
  have row : edge 2239 = (SortedNatTree.contains FastRows.row2239) := by
    change Data.row2239.contains = (SortedNatTree.contains FastRows.row2239)
    exact FastRows.row2239_checked.symm
  rw [row]
  decide
theorem root5Step29 : scan edge (search edge 3) 4 [19] 19 root5Tail29 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail30 winner := by
  rw [root5Tail29, scan, if_neg (by decide), root5Candidates29_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail30
    (if 3 ≤ root5Candidates29.length then search edge 3 [19, 2239] root5Candidates29 2258 winner else winner) = _
  rw [if_neg (by decide)]
def root5Candidates30 : List Nat := [12457, 12541, 13003]
set_option maxRecDepth 8192 in
theorem root5Candidates30_checked : candidates edge 4 19 2251 root5Tail31 winner.bound = root5Candidates30 := by
  unfold candidates
  have row : edge 2251 = (SortedNatTree.contains FastRows.row2251) := by
    change Data.row2251.contains = (SortedNatTree.contains FastRows.row2251)
    exact FastRows.row2251_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child30 : search edge 3 [19, 2251] root5Candidates30 2270 winner = winner := by decide
theorem root5Step30 : scan edge (search edge 3) 4 [19] 19 root5Tail30 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail31 winner := by
  rw [root5Tail30, scan, if_neg (by decide), root5Candidates30_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail31
    (if 3 ≤ root5Candidates30.length then search edge 3 [19, 2251] root5Candidates30 2270 winner else winner) = _
  rw [if_pos (by decide), root5Child30]
def root5Candidates31 : List Nat := [4723, 7159, 13999, 19843]
set_option maxRecDepth 8192 in
theorem root5Candidates31_checked : candidates edge 4 19 2539 root5Tail32 winner.bound = root5Candidates31 := by
  unfold candidates
  have row : edge 2539 = (SortedNatTree.contains FastRows.row2539) := by
    change Data.row2539.contains = (SortedNatTree.contains FastRows.row2539)
    exact FastRows.row2539_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child31 : search edge 3 [19, 2539] root5Candidates31 2558 winner = winner := by decide
theorem root5Step31 : scan edge (search edge 3) 4 [19] 19 root5Tail31 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail32 winner := by
  rw [root5Tail31, scan, if_neg (by decide), root5Candidates31_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail32
    (if 3 ≤ root5Candidates31.length then search edge 3 [19, 2539] root5Candidates31 2558 winner else winner) = _
  rw [if_pos (by decide), root5Child31]
end ProjectEulerSolutions.PrimePairLarge.Search
