import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot016Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root16Candidates0 : List Nat := [547, 619, 3313, 5743, 22039]
set_option maxRecDepth 8192 in
theorem root16Candidates0_checked : candidates edge 4 67 139 root16Tail1 winner.bound = root16Candidates0 := by
  unfold candidates
  have row : edge 139 = (SortedNatTree.contains FastRows.row139) := by
    change Data.row139.contains = (SortedNatTree.contains FastRows.row139)
    exact FastRows.row139_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root16Child0 : search edge 3 [67, 139] root16Candidates0 206 winner = winner := by decide
theorem root16Step0 : scan edge (search edge 3) 4 [67] 67 root16Tail0 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail1 winner := by
  rw [root16Tail0, scan, if_neg (by decide), root16Candidates0_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail1
    (if 3 ≤ root16Candidates0.length then search edge 3 [67, 139] root16Candidates0 206 winner else winner) = _
  rw [if_pos (by decide), root16Child0]
def root16Candidates1 : List Nat := [2377, 6661, 7561, 9883, 11071, 15859, 19861]
set_option maxRecDepth 8192 in
theorem root16Candidates1_checked : candidates edge 4 67 157 root16Tail2 winner.bound = root16Candidates1 := by
  unfold candidates
  have row : edge 157 = (SortedNatTree.contains FastRows.row157) := by
    change Data.row157.contains = (SortedNatTree.contains FastRows.row157)
    exact FastRows.row157_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root16Child1 : search edge 3 [67, 157] root16Candidates1 224 winner = winner := by decide
theorem root16Step1 : scan edge (search edge 3) 4 [67] 67 root16Tail1 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail2 winner := by
  rw [root16Tail1, scan, if_neg (by decide), root16Candidates1_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail2
    (if 3 ≤ root16Candidates1.length then search edge 3 [67, 157] root16Candidates1 224 winner else winner) = _
  rw [if_pos (by decide), root16Child1]
def root16Candidates2 : List Nat := [9319, 20947, 22039]
set_option maxRecDepth 8192 in
theorem root16Candidates2_checked : candidates edge 4 67 547 root16Tail3 winner.bound = root16Candidates2 := by
  unfold candidates
  have row : edge 547 = (SortedNatTree.contains FastRows.row547) := by
    change Data.row547.contains = (SortedNatTree.contains FastRows.row547)
    exact FastRows.row547_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root16Child2 : search edge 3 [67, 547] root16Candidates2 614 winner = winner := by decide
theorem root16Step2 : scan edge (search edge 3) 4 [67] 67 root16Tail2 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail3 winner := by
  rw [root16Tail2, scan, if_neg (by decide), root16Candidates2_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail3
    (if 3 ≤ root16Candidates2.length then search edge 3 [67, 547] root16Candidates2 614 winner else winner) = _
  rw [if_pos (by decide), root16Child2]
def root16Candidates3 : List Nat := [7639, 10909, 11071, 21157, 21961, 22291]
set_option maxRecDepth 8192 in
theorem root16Candidates3_checked : candidates edge 4 67 601 root16Tail4 winner.bound = root16Candidates3 := by
  unfold candidates
  have row : edge 601 = (SortedNatTree.contains FastRows.row601) := by
    change Data.row601.contains = (SortedNatTree.contains FastRows.row601)
    exact FastRows.row601_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root16Child3 : search edge 3 [67, 601] root16Candidates3 668 winner = winner := by decide
theorem root16Step3 : scan edge (search edge 3) 4 [67] 67 root16Tail3 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail4 winner := by
  rw [root16Tail3, scan, if_neg (by decide), root16Candidates3_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail4
    (if 3 ≤ root16Candidates3.length then search edge 3 [67, 601] root16Candidates3 668 winner else winner) = _
  rw [if_pos (by decide), root16Child3]
def root16Candidates4 : List Nat := [5419, 7687]
set_option maxRecDepth 8192 in
theorem root16Candidates4_checked : candidates edge 4 67 619 root16Tail5 winner.bound = root16Candidates4 := by
  unfold candidates
  have row : edge 619 = (SortedNatTree.contains FastRows.row619) := by
    change Data.row619.contains = (SortedNatTree.contains FastRows.row619)
    exact FastRows.row619_checked.symm
  rw [row]
  decide
theorem root16Step4 : scan edge (search edge 3) 4 [67] 67 root16Tail4 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail5 winner := by
  rw [root16Tail4, scan, if_neg (by decide), root16Candidates4_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail5
    (if 3 ≤ root16Candidates4.length then search edge 3 [67, 619] root16Candidates4 686 winner else winner) = _
  rw [if_neg (by decide)]
def root16Candidates5 : List Nat := [2377, 19501, 22039]
set_option maxRecDepth 8192 in
theorem root16Candidates5_checked : candidates edge 4 67 751 root16Tail6 winner.bound = root16Candidates5 := by
  unfold candidates
  have row : edge 751 = (SortedNatTree.contains FastRows.row751) := by
    change Data.row751.contains = (SortedNatTree.contains FastRows.row751)
    exact FastRows.row751_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root16Child5 : search edge 3 [67, 751] root16Candidates5 818 winner = winner := by decide
theorem root16Step5 : scan edge (search edge 3) 4 [67] 67 root16Tail5 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail6 winner := by
  rw [root16Tail5, scan, if_neg (by decide), root16Candidates5_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail6
    (if 3 ≤ root16Candidates5.length then search edge 3 [67, 751] root16Candidates5 818 winner else winner) = _
  rw [if_pos (by decide), root16Child5]
def root16Candidates6 : List Nat := [5569, 6661, 12421, 21157]
set_option maxRecDepth 8192 in
theorem root16Candidates6_checked : candidates edge 4 67 757 root16Tail7 winner.bound = root16Candidates6 := by
  unfold candidates
  have row : edge 757 = (SortedNatTree.contains FastRows.row757) := by
    change Data.row757.contains = (SortedNatTree.contains FastRows.row757)
    exact FastRows.row757_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root16Child6 : search edge 3 [67, 757] root16Candidates6 824 winner = winner := by decide
theorem root16Step6 : scan edge (search edge 3) 4 [67] 67 root16Tail6 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail7 winner := by
  rw [root16Tail6, scan, if_neg (by decide), root16Candidates6_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail7
    (if 3 ≤ root16Candidates6.length then search edge 3 [67, 757] root16Candidates6 824 winner else winner) = _
  rw [if_pos (by decide), root16Child6]
def root16Candidates7 : List Nat := [1999, 3637, 12421, 22129]
set_option maxRecDepth 8192 in
theorem root16Candidates7_checked : candidates edge 4 67 1039 root16Tail8 winner.bound = root16Candidates7 := by
  unfold candidates
  have row : edge 1039 = (SortedNatTree.contains FastRows.row1039) := by
    change Data.row1039.contains = (SortedNatTree.contains FastRows.row1039)
    exact FastRows.row1039_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root16Child7 : search edge 3 [67, 1039] root16Candidates7 1106 winner = winner := by decide
theorem root16Step7 : scan edge (search edge 3) 4 [67] 67 root16Tail7 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail8 winner := by
  rw [root16Tail7, scan, if_neg (by decide), root16Candidates7_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail8
    (if 3 ≤ root16Candidates7.length then search edge 3 [67, 1039] root16Candidates7 1106 winner else winner) = _
  rw [if_pos (by decide), root16Child7]
def root16Candidates8 : List Nat := [1609, 3499, 9463, 24007]
set_option maxRecDepth 8192 in
theorem root16Candidates8_checked : candidates edge 4 67 1063 root16Tail9 winner.bound = root16Candidates8 := by
  unfold candidates
  have row : edge 1063 = (SortedNatTree.contains FastRows.row1063) := by
    change Data.row1063.contains = (SortedNatTree.contains FastRows.row1063)
    exact FastRows.row1063_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root16Child8 : search edge 3 [67, 1063] root16Candidates8 1130 winner = winner := by decide
theorem root16Step8 : scan edge (search edge 3) 4 [67] 67 root16Tail8 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail9 winner := by
  rw [root16Tail8, scan, if_neg (by decide), root16Candidates8_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail9
    (if 3 ≤ root16Candidates8.length then search edge 3 [67, 1063] root16Candidates8 1130 winner else winner) = _
  rw [if_pos (by decide), root16Child8]
def root16Candidates9 : List Nat := [5569, 9883, 17077, 23971]
set_option maxRecDepth 8192 in
theorem root16Candidates9_checked : candidates edge 4 67 1093 root16Tail10 winner.bound = root16Candidates9 := by
  unfold candidates
  have row : edge 1093 = (SortedNatTree.contains FastRows.row1093) := by
    change Data.row1093.contains = (SortedNatTree.contains FastRows.row1093)
    exact FastRows.row1093_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root16Child9 : search edge 3 [67, 1093] root16Candidates9 1160 winner = winner := by decide
theorem root16Step9 : scan edge (search edge 3) 4 [67] 67 root16Tail9 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail10 winner := by
  rw [root16Tail9, scan, if_neg (by decide), root16Candidates9_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail10
    (if 3 ≤ root16Candidates9.length then search edge 3 [67, 1093] root16Candidates9 1160 winner else winner) = _
  rw [if_pos (by decide), root16Child9]
def root16Candidates10 : List Nat := [1303, 3637, 6337, 8641, 11071, 17077, 19861, 24007]
set_option maxRecDepth 8192 in
theorem root16Candidates10_checked : candidates edge 4 67 1201 root16Tail11 winner.bound = root16Candidates10 := by
  unfold candidates
  have row : edge 1201 = (SortedNatTree.contains FastRows.row1201) := by
    change Data.row1201.contains = (SortedNatTree.contains FastRows.row1201)
    exact FastRows.row1201_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root16Child10 : search edge 3 [67, 1201] root16Candidates10 1268 winner = winner := by decide
theorem root16Step10 : scan edge (search edge 3) 4 [67] 67 root16Tail10 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail11 winner := by
  rw [root16Tail10, scan, if_neg (by decide), root16Candidates10_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail11
    (if 3 ≤ root16Candidates10.length then search edge 3 [67, 1201] root16Candidates10 1268 winner else winner) = _
  rw [if_pos (by decide), root16Child10]
def root16Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root16Candidates11_checked : candidates edge 4 67 1303 root16Tail12 winner.bound = root16Candidates11 := by
  unfold candidates
  have row : edge 1303 = (SortedNatTree.contains FastRows.row1303) := by
    change Data.row1303.contains = (SortedNatTree.contains FastRows.row1303)
    exact FastRows.row1303_checked.symm
  rw [row]
  decide
theorem root16Step11 : scan edge (search edge 3) 4 [67] 67 root16Tail11 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail12 winner := by
  rw [root16Tail11, scan, if_neg (by decide), root16Candidates11_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail12
    (if 3 ≤ root16Candidates11.length then search edge 3 [67, 1303] root16Candidates11 1370 winner else winner) = _
  rw [if_neg (by decide)]
def root16Candidates12 : List Nat := [5179, 9883, 12421, 17077, 18211]
set_option maxRecDepth 8192 in
theorem root16Candidates12_checked : candidates edge 4 67 1609 root16Tail13 winner.bound = root16Candidates12 := by
  unfold candidates
  have row : edge 1609 = (SortedNatTree.contains FastRows.row1609) := by
    change Data.row1609.contains = (SortedNatTree.contains FastRows.row1609)
    exact FastRows.row1609_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root16Child12 : search edge 3 [67, 1609] root16Candidates12 1676 winner = winner := by decide
theorem root16Step12 : scan edge (search edge 3) 4 [67] 67 root16Tail12 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail13 winner := by
  rw [root16Tail12, scan, if_neg (by decide), root16Candidates12_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail13
    (if 3 ≤ root16Candidates12.length then search edge 3 [67, 1609] root16Candidates12 1676 winner else winner) = _
  rw [if_pos (by decide), root16Child12]
def root16Candidates13 : List Nat := [8941, 11491, 12421]
set_option maxRecDepth 8192 in
theorem root16Candidates13_checked : candidates edge 4 67 1831 root16Tail14 winner.bound = root16Candidates13 := by
  unfold candidates
  have row : edge 1831 = (SortedNatTree.contains FastRows.row1831) := by
    change Data.row1831.contains = (SortedNatTree.contains FastRows.row1831)
    exact FastRows.row1831_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root16Child13 : search edge 3 [67, 1831] root16Candidates13 1898 winner = winner := by decide
theorem root16Step13 : scan edge (search edge 3) 4 [67] 67 root16Tail13 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail14 winner := by
  rw [root16Tail13, scan, if_neg (by decide), root16Candidates13_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail14
    (if 3 ≤ root16Candidates13.length then search edge 3 [67, 1831] root16Candidates13 1898 winner else winner) = _
  rw [if_pos (by decide), root16Child13]
def root16Candidates14 : List Nat := [3313, 7639, 9907, 23071]
set_option maxRecDepth 8192 in
theorem root16Candidates14_checked : candidates edge 4 67 1933 root16Tail15 winner.bound = root16Candidates14 := by
  unfold candidates
  have row : edge 1933 = (SortedNatTree.contains FastRows.row1933) := by
    change Data.row1933.contains = (SortedNatTree.contains FastRows.row1933)
    exact FastRows.row1933_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root16Child14 : search edge 3 [67, 1933] root16Candidates14 2000 winner = winner := by decide
theorem root16Step14 : scan edge (search edge 3) 4 [67] 67 root16Tail14 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail15 winner := by
  rw [root16Tail14, scan, if_neg (by decide), root16Candidates14_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail15
    (if 3 ≤ root16Candidates14.length then search edge 3 [67, 1933] root16Candidates14 2000 winner else winner) = _
  rw [if_pos (by decide), root16Child14]
def root16Candidates15 : List Nat := [12421, 22291]
set_option maxRecDepth 8192 in
theorem root16Candidates15_checked : candidates edge 4 67 1999 root16Tail16 winner.bound = root16Candidates15 := by
  unfold candidates
  have row : edge 1999 = (SortedNatTree.contains FastRows.row1999) := by
    change Data.row1999.contains = (SortedNatTree.contains FastRows.row1999)
    exact FastRows.row1999_checked.symm
  rw [row]
  decide
theorem root16Step15 : scan edge (search edge 3) 4 [67] 67 root16Tail15 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail16 winner := by
  rw [root16Tail15, scan, if_neg (by decide), root16Candidates15_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail16
    (if 3 ≤ root16Candidates15.length then search edge 3 [67, 1999] root16Candidates15 2066 winner else winner) = _
  rw [if_neg (by decide)]
def root16Candidates16 : List Nat := [2341, 5419]
set_option maxRecDepth 8192 in
theorem root16Candidates16_checked : candidates edge 4 67 2029 root16Tail17 winner.bound = root16Candidates16 := by
  unfold candidates
  have row : edge 2029 = (SortedNatTree.contains FastRows.row2029) := by
    change Data.row2029.contains = (SortedNatTree.contains FastRows.row2029)
    exact FastRows.row2029_checked.symm
  rw [row]
  decide
theorem root16Step16 : scan edge (search edge 3) 4 [67] 67 root16Tail16 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail17 winner := by
  rw [root16Tail16, scan, if_neg (by decide), root16Candidates16_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail17
    (if 3 ≤ root16Candidates16.length then search edge 3 [67, 2029] root16Candidates16 2096 winner else winner) = _
  rw [if_neg (by decide)]
def root16Candidates17 : List Nat := [5881, 8191, 12457, 22039]
set_option maxRecDepth 8192 in
theorem root16Candidates17_checked : candidates edge 4 67 2251 root16Tail18 winner.bound = root16Candidates17 := by
  unfold candidates
  have row : edge 2251 = (SortedNatTree.contains FastRows.row2251) := by
    change Data.row2251.contains = (SortedNatTree.contains FastRows.row2251)
    exact FastRows.row2251_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root16Child17 : search edge 3 [67, 2251] root16Candidates17 2318 winner = winner := by decide
theorem root16Step17 : scan edge (search edge 3) 4 [67] 67 root16Tail17 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail18 winner := by
  rw [root16Tail17, scan, if_neg (by decide), root16Candidates17_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail18
    (if 3 ≤ root16Candidates17.length then search edge 3 [67, 2251] root16Candidates17 2318 winner else winner) = _
  rw [if_pos (by decide), root16Child17]
def root16Candidates18 : List Nat := []
set_option maxRecDepth 8192 in
theorem root16Candidates18_checked : candidates edge 4 67 2341 root16Tail19 winner.bound = root16Candidates18 := by
  unfold candidates
  have row : edge 2341 = (SortedNatTree.contains FastRows.row2341) := by
    change Data.row2341.contains = (SortedNatTree.contains FastRows.row2341)
    exact FastRows.row2341_checked.symm
  rw [row]
  decide
theorem root16Step18 : scan edge (search edge 3) 4 [67] 67 root16Tail18 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail19 winner := by
  rw [root16Tail18, scan, if_neg (by decide), root16Candidates18_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail19
    (if 3 ≤ root16Candidates18.length then search edge 3 [67, 2341] root16Candidates18 2408 winner else winner) = _
  rw [if_neg (by decide)]
def root16Candidates19 : List Nat := [2557, 5743, 9733, 10909]
set_option maxRecDepth 8192 in
theorem root16Candidates19_checked : candidates edge 4 67 2377 root16Tail20 winner.bound = root16Candidates19 := by
  unfold candidates
  have row : edge 2377 = (SortedNatTree.contains FastRows.row2377) := by
    change Data.row2377.contains = (SortedNatTree.contains FastRows.row2377)
    exact FastRows.row2377_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root16Child19 : search edge 3 [67, 2377] root16Candidates19 2444 winner = winner := by decide
theorem root16Step19 : scan edge (search edge 3) 4 [67] 67 root16Tail19 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail20 winner := by
  rw [root16Tail19, scan, if_neg (by decide), root16Candidates19_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail20
    (if 3 ≤ root16Candidates19.length then search edge 3 [67, 2377] root16Candidates19 2444 winner else winner) = _
  rw [if_pos (by decide), root16Child19]
def root16Candidates20 : List Nat := [3313, 7687, 8191]
set_option maxRecDepth 8192 in
theorem root16Candidates20_checked : candidates edge 4 67 2557 root16Tail21 winner.bound = root16Candidates20 := by
  unfold candidates
  have row : edge 2557 = (SortedNatTree.contains FastRows.row2557) := by
    change Data.row2557.contains = (SortedNatTree.contains FastRows.row2557)
    exact FastRows.row2557_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root16Child20 : search edge 3 [67, 2557] root16Candidates20 2624 winner = winner := by decide
theorem root16Step20 : scan edge (search edge 3) 4 [67] 67 root16Tail20 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail21 winner := by
  rw [root16Tail20, scan, if_neg (by decide), root16Candidates20_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail21
    (if 3 ≤ root16Candidates20.length then search edge 3 [67, 2557] root16Candidates20 2624 winner else winner) = _
  rw [if_pos (by decide), root16Child20]
def root16Candidates21 : List Nat := [4099, 7561]
set_option maxRecDepth 8192 in
theorem root16Candidates21_checked : candidates edge 4 67 2767 root16Tail22 winner.bound = root16Candidates21 := by
  unfold candidates
  have row : edge 2767 = (SortedNatTree.contains FastRows.row2767) := by
    change Data.row2767.contains = (SortedNatTree.contains FastRows.row2767)
    exact FastRows.row2767_checked.symm
  rw [row]
  decide
theorem root16Step21 : scan edge (search edge 3) 4 [67] 67 root16Tail21 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail22 winner := by
  rw [root16Tail21, scan, if_neg (by decide), root16Candidates21_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail22
    (if 3 ≤ root16Candidates21.length then search edge 3 [67, 2767] root16Candidates21 2834 winner else winner) = _
  rw [if_neg (by decide)]
def root16Candidates22 : List Nat := [4831]
set_option maxRecDepth 8192 in
theorem root16Candidates22_checked : candidates edge 4 67 3313 root16Tail23 winner.bound = root16Candidates22 := by
  unfold candidates
  have row : edge 3313 = (SortedNatTree.contains FastRows.row3313) := by
    change Data.row3313.contains = (SortedNatTree.contains FastRows.row3313)
    exact FastRows.row3313_checked.symm
  rw [row]
  decide
theorem root16Step22 : scan edge (search edge 3) 4 [67] 67 root16Tail22 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail23 winner := by
  rw [root16Tail22, scan, if_neg (by decide), root16Candidates22_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail23
    (if 3 ≤ root16Candidates22.length then search edge 3 [67, 3313] root16Candidates22 3380 winner else winner) = _
  rw [if_neg (by decide)]
def root16Candidates23 : List Nat := [5881, 8191]
set_option maxRecDepth 8192 in
theorem root16Candidates23_checked : candidates edge 4 67 3499 root16Tail24 winner.bound = root16Candidates23 := by
  unfold candidates
  have row : edge 3499 = (SortedNatTree.contains FastRows.row3499) := by
    change Data.row3499.contains = (SortedNatTree.contains FastRows.row3499)
    exact FastRows.row3499_checked.symm
  rw [row]
  decide
theorem root16Step23 : scan edge (search edge 3) 4 [67] 67 root16Tail23 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail24 winner := by
  rw [root16Tail23, scan, if_neg (by decide), root16Candidates23_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail24
    (if 3 ≤ root16Candidates23.length then search edge 3 [67, 3499] root16Candidates23 3566 winner else winner) = _
  rw [if_neg (by decide)]
def root16Candidates24 : List Nat := [7639, 18211, 22129]
set_option maxRecDepth 8192 in
theorem root16Candidates24_checked : candidates edge 4 67 3637 root16Tail25 winner.bound = root16Candidates24 := by
  unfold candidates
  have row : edge 3637 = (SortedNatTree.contains FastRows.row3637) := by
    change Data.row3637.contains = (SortedNatTree.contains FastRows.row3637)
    exact FastRows.row3637_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root16Child24 : search edge 3 [67, 3637] root16Candidates24 3704 winner = winner := by decide
theorem root16Step24 : scan edge (search edge 3) 4 [67] 67 root16Tail24 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail25 winner := by
  rw [root16Tail24, scan, if_neg (by decide), root16Candidates24_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail25
    (if 3 ≤ root16Candidates24.length then search edge 3 [67, 3637] root16Candidates24 3704 winner else winner) = _
  rw [if_pos (by decide), root16Child24]
def root16Candidates25 : List Nat := [5569, 5581, 6829]
set_option maxRecDepth 8192 in
theorem root16Candidates25_checked : candidates edge 4 67 3943 root16Tail26 winner.bound = root16Candidates25 := by
  unfold candidates
  have row : edge 3943 = (SortedNatTree.contains FastRows.row3943) := by
    change Data.row3943.contains = (SortedNatTree.contains FastRows.row3943)
    exact FastRows.row3943_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root16Child25 : search edge 3 [67, 3943] root16Candidates25 4010 winner = winner := by decide
theorem root16Step25 : scan edge (search edge 3) 4 [67] 67 root16Tail25 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail26 winner := by
  rw [root16Tail25, scan, if_neg (by decide), root16Candidates25_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail26
    (if 3 ≤ root16Candidates25.length then search edge 3 [67, 3943] root16Candidates25 4010 winner else winner) = _
  rw [if_pos (by decide), root16Child25]
def root16Candidates26 : List Nat := [4483, 5419, 9829, 14107]
set_option maxRecDepth 8192 in
theorem root16Candidates26_checked : candidates edge 4 67 4057 root16Tail27 winner.bound = root16Candidates26 := by
  unfold candidates
  have row : edge 4057 = (SortedNatTree.contains FastRows.row4057) := by
    change Data.row4057.contains = (SortedNatTree.contains FastRows.row4057)
    exact FastRows.row4057_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root16Child26 : search edge 3 [67, 4057] root16Candidates26 4124 winner = winner := by decide
theorem root16Step26 : scan edge (search edge 3) 4 [67] 67 root16Tail26 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail27 winner := by
  rw [root16Tail26, scan, if_neg (by decide), root16Candidates26_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail27
    (if 3 ≤ root16Candidates26.length then search edge 3 [67, 4057] root16Candidates26 4124 winner else winner) = _
  rw [if_pos (by decide), root16Child26]
def root16Candidates27 : List Nat := [10909, 18211]
set_option maxRecDepth 8192 in
theorem root16Candidates27_checked : candidates edge 4 67 4099 root16Tail28 winner.bound = root16Candidates27 := by
  unfold candidates
  have row : edge 4099 = (SortedNatTree.contains FastRows.row4099) := by
    change Data.row4099.contains = (SortedNatTree.contains FastRows.row4099)
    exact FastRows.row4099_checked.symm
  rw [row]
  decide
theorem root16Step27 : scan edge (search edge 3) 4 [67] 67 root16Tail27 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail28 winner := by
  rw [root16Tail27, scan, if_neg (by decide), root16Candidates27_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail28
    (if 3 ≤ root16Candidates27.length then search edge 3 [67, 4099] root16Candidates27 4166 winner else winner) = _
  rw [if_neg (by decide)]
def root16Candidates28 : List Nat := [9463, 17959]
set_option maxRecDepth 8192 in
theorem root16Candidates28_checked : candidates edge 4 67 4483 root16Tail29 winner.bound = root16Candidates28 := by
  unfold candidates
  have row : edge 4483 = (SortedNatTree.contains FastRows.row4483) := by
    change Data.row4483.contains = (SortedNatTree.contains FastRows.row4483)
    exact FastRows.row4483_checked.symm
  rw [row]
  decide
theorem root16Step28 : scan edge (search edge 3) 4 [67] 67 root16Tail28 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail29 winner := by
  rw [root16Tail28, scan, if_neg (by decide), root16Candidates28_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail29
    (if 3 ≤ root16Candidates28.length then search edge 3 [67, 4483] root16Candidates28 4550 winner else winner) = _
  rw [if_neg (by decide)]
def root16Candidates29 : List Nat := [5179, 7561, 11071]
set_option maxRecDepth 8192 in
theorem root16Candidates29_checked : candidates edge 4 67 4789 root16Tail30 winner.bound = root16Candidates29 := by
  unfold candidates
  have row : edge 4789 = (SortedNatTree.contains FastRows.row4789) := by
    change Data.row4789.contains = (SortedNatTree.contains FastRows.row4789)
    exact FastRows.row4789_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root16Child29 : search edge 3 [67, 4789] root16Candidates29 4856 winner = winner := by decide
theorem root16Step29 : scan edge (search edge 3) 4 [67] 67 root16Tail29 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail30 winner := by
  rw [root16Tail29, scan, if_neg (by decide), root16Candidates29_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail30
    (if 3 ≤ root16Candidates29.length then search edge 3 [67, 4789] root16Candidates29 4856 winner else winner) = _
  rw [if_pos (by decide), root16Child29]
def root16Candidates30 : List Nat := [8191]
set_option maxRecDepth 8192 in
theorem root16Candidates30_checked : candidates edge 4 67 4831 root16Tail31 winner.bound = root16Candidates30 := by
  unfold candidates
  have row : edge 4831 = (SortedNatTree.contains FastRows.row4831) := by
    change Data.row4831.contains = (SortedNatTree.contains FastRows.row4831)
    exact FastRows.row4831_checked.symm
  rw [row]
  decide
theorem root16Step30 : scan edge (search edge 3) 4 [67] 67 root16Tail30 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail31 winner := by
  rw [root16Tail30, scan, if_neg (by decide), root16Candidates30_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail31
    (if 3 ≤ root16Candidates30.length then search edge 3 [67, 4831] root16Candidates30 4898 winner else winner) = _
  rw [if_neg (by decide)]
def root16Candidates31 : List Nat := [17341]
set_option maxRecDepth 8192 in
theorem root16Candidates31_checked : candidates edge 4 67 4903 root16Tail32 winner.bound = root16Candidates31 := by
  unfold candidates
  have row : edge 4903 = (SortedNatTree.contains FastRows.row4903) := by
    change Data.row4903.contains = (SortedNatTree.contains FastRows.row4903)
    exact FastRows.row4903_checked.symm
  rw [row]
  decide
theorem root16Step31 : scan edge (search edge 3) 4 [67] 67 root16Tail31 winner =
    scan edge (search edge 3) 4 [67] 67 root16Tail32 winner := by
  rw [root16Tail31, scan, if_neg (by decide), root16Candidates31_checked]
  change scan edge (search edge 3) 4 [67] 67 root16Tail32
    (if 3 ≤ root16Candidates31.length then search edge 3 [67, 4903] root16Candidates31 4970 winner else winner) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
