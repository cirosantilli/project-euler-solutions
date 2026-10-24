import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot011Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root11Candidates0 : List Nat := [1381, 3373, 8287, 8521, 9787, 12697, 20959]
set_option maxRecDepth 8192 in
theorem root11Candidates0_checked : candidates edge 4 43 97 root11Tail1 winner.bound = root11Candidates0 := by
  unfold candidates
  have row : edge 97 = (SortedNatTree.contains FastRows.row97) := by
    change Data.row97.contains = (SortedNatTree.contains FastRows.row97)
    exact FastRows.row97_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root11Child0 : search edge 3 [43, 97] root11Candidates0 140 winner = winner := by decide
theorem root11Step0 : scan edge (search edge 3) 4 [43] 43 root11Tail0 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail1 winner := by
  rw [root11Tail0, scan, if_neg (by decide), root11Candidates0_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail1
    (if 3 ≤ root11Candidates0.length then search edge 3 [43, 97] root11Candidates0 140 winner else winner) = _
  rw [if_pos (by decide), root11Child0]
def root11Candidates1 : List Nat := [8329, 8623, 8707, 16519]
set_option maxRecDepth 8192 in
theorem root11Candidates1_checked : candidates edge 4 43 103 root11Tail2 winner.bound = root11Candidates1 := by
  unfold candidates
  have row : edge 103 = (SortedNatTree.contains FastRows.row103) := by
    change Data.row103.contains = (SortedNatTree.contains FastRows.row103)
    exact FastRows.row103_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root11Child1 : search edge 3 [43, 103] root11Candidates1 146 winner = winner := by decide
theorem root11Step1 : scan edge (search edge 3) 4 [43] 43 root11Tail1 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail2 winner := by
  rw [root11Tail1, scan, if_neg (by decide), root11Candidates1_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail2
    (if 3 ≤ root11Candidates1.length then search edge 3 [43, 103] root11Candidates1 146 winner else winner) = _
  rw [if_pos (by decide), root11Child1]
def root11Candidates2 : List Nat := [7159, 9007, 19993]
set_option maxRecDepth 8192 in
theorem root11Candidates2_checked : candidates edge 4 43 223 root11Tail3 winner.bound = root11Candidates2 := by
  unfold candidates
  have row : edge 223 = (SortedNatTree.contains FastRows.row223) := by
    change Data.row223.contains = (SortedNatTree.contains FastRows.row223)
    exact FastRows.row223_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root11Child2 : search edge 3 [43, 223] root11Candidates2 266 winner = winner := by decide
theorem root11Step2 : scan edge (search edge 3) 4 [43] 43 root11Tail2 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail3 winner := by
  rw [root11Tail2, scan, if_neg (by decide), root11Candidates2_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail3
    (if 3 ≤ root11Candidates2.length then search edge 3 [43, 223] root11Candidates2 266 winner else winner) = _
  rw [if_pos (by decide), root11Child2]
def root11Candidates3 : List Nat := [1213, 1399, 4111, 9421, 18661]
set_option maxRecDepth 8192 in
theorem root11Candidates3_checked : candidates edge 4 43 271 root11Tail4 winner.bound = root11Candidates3 := by
  unfold candidates
  have row : edge 271 = (SortedNatTree.contains FastRows.row271) := by
    change Data.row271.contains = (SortedNatTree.contains FastRows.row271)
    exact FastRows.row271_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root11Child3 : search edge 3 [43, 271] root11Candidates3 314 winner = winner := by decide
theorem root11Step3 : scan edge (search edge 3) 4 [43] 43 root11Tail3 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail4 winner := by
  rw [root11Tail3, scan, if_neg (by decide), root11Candidates3_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail4
    (if 3 ≤ root11Candidates3.length then search edge 3 [43, 271] root11Candidates3 314 winner else winner) = _
  rw [if_pos (by decide), root11Child3]
def root11Candidates4 : List Nat := [691, 14281, 24373, 25339]
set_option maxRecDepth 8192 in
theorem root11Candidates4_checked : candidates edge 4 43 499 root11Tail5 winner.bound = root11Candidates4 := by
  unfold candidates
  have row : edge 499 = (SortedNatTree.contains FastRows.row499) := by
    change Data.row499.contains = (SortedNatTree.contains FastRows.row499)
    exact FastRows.row499_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root11Child4 : search edge 3 [43, 499] root11Candidates4 542 winner = winner := by decide
theorem root11Step4 : scan edge (search edge 3) 4 [43] 43 root11Tail4 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail5 winner := by
  rw [root11Tail4, scan, if_neg (by decide), root11Candidates4_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail5
    (if 3 ≤ root11Candidates4.length then search edge 3 [43, 499] root11Candidates4 542 winner else winner) = _
  rw [if_pos (by decide), root11Child4]
def root11Candidates5 : List Nat := [4363, 6547, 6967, 12073, 22783]
set_option maxRecDepth 8192 in
theorem root11Candidates5_checked : candidates edge 4 43 613 root11Tail6 winner.bound = root11Candidates5 := by
  unfold candidates
  have row : edge 613 = (SortedNatTree.contains FastRows.row613) := by
    change Data.row613.contains = (SortedNatTree.contains FastRows.row613)
    exact FastRows.row613_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root11Child5 : search edge 3 [43, 613] root11Candidates5 656 winner = winner := by decide
theorem root11Step5 : scan edge (search edge 3) 4 [43] 43 root11Tail5 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail6 winner := by
  rw [root11Tail5, scan, if_neg (by decide), root11Candidates5_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail6
    (if 3 ≤ root11Candidates5.length then search edge 3 [43, 613] root11Candidates5 656 winner else winner) = _
  rw [if_pos (by decide), root11Child5]
def root11Candidates6 : List Nat := [1693, 3469, 9421, 16519, 18451]
set_option maxRecDepth 8192 in
theorem root11Candidates6_checked : candidates edge 4 43 691 root11Tail7 winner.bound = root11Candidates6 := by
  unfold candidates
  have row : edge 691 = (SortedNatTree.contains FastRows.row691) := by
    change Data.row691.contains = (SortedNatTree.contains FastRows.row691)
    exact FastRows.row691_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root11Child6 : search edge 3 [43, 691] root11Candidates6 734 winner = winner := by decide
theorem root11Step6 : scan edge (search edge 3) 4 [43] 43 root11Tail6 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail7 winner := by
  rw [root11Tail6, scan, if_neg (by decide), root11Candidates6_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail7
    (if 3 ≤ root11Candidates6.length then search edge 3 [43, 691] root11Candidates6 734 winner else winner) = _
  rw [if_pos (by decide), root11Child6]
def root11Candidates7 : List Nat := [2389, 4813, 8623, 9697]
set_option maxRecDepth 8192 in
theorem root11Candidates7_checked : candidates edge 4 43 1051 root11Tail8 winner.bound = root11Candidates7 := by
  unfold candidates
  have row : edge 1051 = (SortedNatTree.contains FastRows.row1051) := by
    change Data.row1051.contains = (SortedNatTree.contains FastRows.row1051)
    exact FastRows.row1051_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root11Child7 : search edge 3 [43, 1051] root11Candidates7 1094 winner = winner := by decide
theorem root11Step7 : scan edge (search edge 3) 4 [43] 43 root11Tail7 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail8 winner := by
  rw [root11Tail7, scan, if_neg (by decide), root11Candidates7_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail8
    (if 3 ≤ root11Candidates7.length then search edge 3 [43, 1051] root11Candidates7 1094 winner else winner) = _
  rw [if_pos (by decide), root11Child7]
def root11Candidates8 : List Nat := [3253, 14029]
set_option maxRecDepth 8192 in
theorem root11Candidates8_checked : candidates edge 4 43 1153 root11Tail9 winner.bound = root11Candidates8 := by
  unfold candidates
  have row : edge 1153 = (SortedNatTree.contains FastRows.row1153) := by
    change Data.row1153.contains = (SortedNatTree.contains FastRows.row1153)
    exact FastRows.row1153_checked.symm
  rw [row]
  decide
theorem root11Step8 : scan edge (search edge 3) 4 [43] 43 root11Tail8 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail9 winner := by
  rw [root11Tail8, scan, if_neg (by decide), root11Candidates8_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail9
    (if 3 ≤ root11Candidates8.length then search edge 3 [43, 1153] root11Candidates8 1196 winner else winner) = _
  rw [if_neg (by decide)]
def root11Candidates9 : List Nat := [1621, 6547, 8521, 9349, 13999, 17569, 17827, 23743]
set_option maxRecDepth 8192 in
theorem root11Candidates9_checked : candidates edge 4 43 1213 root11Tail10 winner.bound = root11Candidates9 := by
  unfold candidates
  have row : edge 1213 = (SortedNatTree.contains FastRows.row1213) := by
    change Data.row1213.contains = (SortedNatTree.contains FastRows.row1213)
    exact FastRows.row1213_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root11Child9 : search edge 3 [43, 1213] root11Candidates9 1256 winner = winner := by decide
theorem root11Step9 : scan edge (search edge 3) 4 [43] 43 root11Tail9 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail10 winner := by
  rw [root11Tail9, scan, if_neg (by decide), root11Candidates9_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail10
    (if 3 ≤ root11Candidates9.length then search edge 3 [43, 1213] root11Candidates9 1256 winner else winner) = _
  rw [if_pos (by decide), root11Child9]
def root11Candidates10 : List Nat := [4111, 8521, 17191]
set_option maxRecDepth 8192 in
theorem root11Candidates10_checked : candidates edge 4 43 1381 root11Tail11 winner.bound = root11Candidates10 := by
  unfold candidates
  have row : edge 1381 = (SortedNatTree.contains FastRows.row1381) := by
    change Data.row1381.contains = (SortedNatTree.contains FastRows.row1381)
    exact FastRows.row1381_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root11Child10 : search edge 3 [43, 1381] root11Candidates10 1424 winner = winner := by decide
theorem root11Step10 : scan edge (search edge 3) 4 [43] 43 root11Tail10 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail11 winner := by
  rw [root11Tail10, scan, if_neg (by decide), root11Candidates10_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail11
    (if 3 ≤ root11Candidates10.length then search edge 3 [43, 1381] root11Candidates10 1424 winner else winner) = _
  rw [if_pos (by decide), root11Child10]
def root11Candidates11 : List Nat := [6529]
set_option maxRecDepth 8192 in
theorem root11Candidates11_checked : candidates edge 4 43 1399 root11Tail12 winner.bound = root11Candidates11 := by
  unfold candidates
  have row : edge 1399 = (SortedNatTree.contains FastRows.row1399) := by
    change Data.row1399.contains = (SortedNatTree.contains FastRows.row1399)
    exact FastRows.row1399_checked.symm
  rw [row]
  decide
theorem root11Step11 : scan edge (search edge 3) 4 [43] 43 root11Tail11 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail12 winner := by
  rw [root11Tail11, scan, if_neg (by decide), root11Candidates11_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail12
    (if 3 ≤ root11Candidates11.length then search edge 3 [43, 1399] root11Candidates11 1442 winner else winner) = _
  rw [if_neg (by decide)]
def root11Candidates12 : List Nat := [3187, 16651]
set_option maxRecDepth 8192 in
theorem root11Candidates12_checked : candidates edge 4 43 1621 root11Tail13 winner.bound = root11Candidates12 := by
  unfold candidates
  have row : edge 1621 = (SortedNatTree.contains FastRows.row1621) := by
    change Data.row1621.contains = (SortedNatTree.contains FastRows.row1621)
    exact FastRows.row1621_checked.symm
  rw [row]
  decide
theorem root11Step12 : scan edge (search edge 3) 4 [43] 43 root11Tail12 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail13 winner := by
  rw [root11Tail12, scan, if_neg (by decide), root11Candidates12_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail13
    (if 3 ≤ root11Candidates12.length then search edge 3 [43, 1621] root11Candidates12 1664 winner else winner) = _
  rw [if_neg (by decide)]
def root11Candidates13 : List Nat := [3373, 12697, 15607, 18661]
set_option maxRecDepth 8192 in
theorem root11Candidates13_checked : candidates edge 4 43 1693 root11Tail14 winner.bound = root11Candidates13 := by
  unfold candidates
  have row : edge 1693 = (SortedNatTree.contains FastRows.row1693) := by
    change Data.row1693.contains = (SortedNatTree.contains FastRows.row1693)
    exact FastRows.row1693_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root11Child13 : search edge 3 [43, 1693] root11Candidates13 1736 winner = winner := by decide
theorem root11Step13 : scan edge (search edge 3) 4 [43] 43 root11Tail13 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail14 winner := by
  rw [root11Tail13, scan, if_neg (by decide), root11Candidates13_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail14
    (if 3 ≤ root11Candidates13.length then search edge 3 [43, 1693] root11Candidates13 1736 winner else winner) = _
  rw [if_pos (by decide), root11Child13]
def root11Candidates14 : List Nat := [3253, 9787, 12433, 12823, 24109]
set_option maxRecDepth 8192 in
theorem root11Candidates14_checked : candidates edge 4 43 1777 root11Tail15 winner.bound = root11Candidates14 := by
  unfold candidates
  have row : edge 1777 = (SortedNatTree.contains FastRows.row1777) := by
    change Data.row1777.contains = (SortedNatTree.contains FastRows.row1777)
    exact FastRows.row1777_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root11Child14 : search edge 3 [43, 1777] root11Candidates14 1820 winner = winner := by decide
theorem root11Step14 : scan edge (search edge 3) 4 [43] 43 root11Tail14 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail15 winner := by
  rw [root11Tail14, scan, if_neg (by decide), root11Candidates14_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail15
    (if 3 ≤ root11Candidates14.length then search edge 3 [43, 1777] root11Candidates14 1820 winner else winner) = _
  rw [if_pos (by decide), root11Child14]
def root11Candidates15 : List Nat := [4111, 12157, 14029]
set_option maxRecDepth 8192 in
theorem root11Candidates15_checked : candidates edge 4 43 1867 root11Tail16 winner.bound = root11Candidates15 := by
  unfold candidates
  have row : edge 1867 = (SortedNatTree.contains FastRows.row1867) := by
    change Data.row1867.contains = (SortedNatTree.contains FastRows.row1867)
    exact FastRows.row1867_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root11Child15 : search edge 3 [43, 1867] root11Candidates15 1910 winner = winner := by decide
theorem root11Step15 : scan edge (search edge 3) 4 [43] 43 root11Tail15 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail16 winner := by
  rw [root11Tail15, scan, if_neg (by decide), root11Candidates15_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail16
    (if 3 ≤ root11Candidates15.length then search edge 3 [43, 1867] root11Candidates15 1910 winner else winner) = _
  rw [if_pos (by decide), root11Child15]
def root11Candidates16 : List Nat := [8287, 9973, 12823, 19993]
set_option maxRecDepth 8192 in
theorem root11Candidates16_checked : candidates edge 4 43 1993 root11Tail17 winner.bound = root11Candidates16 := by
  unfold candidates
  have row : edge 1993 = (SortedNatTree.contains FastRows.row1993) := by
    change Data.row1993.contains = (SortedNatTree.contains FastRows.row1993)
    exact FastRows.row1993_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root11Child16 : search edge 3 [43, 1993] root11Candidates16 2036 winner = winner := by decide
theorem root11Step16 : scan edge (search edge 3) 4 [43] 43 root11Tail16 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail17 winner := by
  rw [root11Tail16, scan, if_neg (by decide), root11Candidates16_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail17
    (if 3 ≤ root11Candidates16.length then search edge 3 [43, 1993] root11Candidates16 2036 winner else winner) = _
  rw [if_pos (by decide), root11Child16]
def root11Candidates17 : List Nat := [4111, 9349, 9787, 12157, 16651]
set_option maxRecDepth 8192 in
theorem root11Candidates17_checked : candidates edge 4 43 2251 root11Tail18 winner.bound = root11Candidates17 := by
  unfold candidates
  have row : edge 2251 = (SortedNatTree.contains FastRows.row2251) := by
    change Data.row2251.contains = (SortedNatTree.contains FastRows.row2251)
    exact FastRows.row2251_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root11Child17 : search edge 3 [43, 2251] root11Candidates17 2294 winner = winner := by decide
theorem root11Step17 : scan edge (search edge 3) 4 [43] 43 root11Tail17 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail18 winner := by
  rw [root11Tail17, scan, if_neg (by decide), root11Candidates17_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail18
    (if 3 ≤ root11Candidates17.length then search edge 3 [43, 2251] root11Candidates17 2294 winner else winner) = _
  rw [if_pos (by decide), root11Child17]
def root11Candidates18 : List Nat := [2503, 4243, 6529, 7351, 9697, 12433, 12823]
set_option maxRecDepth 8192 in
theorem root11Candidates18_checked : candidates edge 4 43 2389 root11Tail19 winner.bound = root11Candidates18 := by
  unfold candidates
  have row : edge 2389 = (SortedNatTree.contains FastRows.row2389) := by
    change Data.row2389.contains = (SortedNatTree.contains FastRows.row2389)
    exact FastRows.row2389_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root11Child18 : search edge 3 [43, 2389] root11Candidates18 2432 winner = winner := by decide
theorem root11Step18 : scan edge (search edge 3) 4 [43] 43 root11Tail18 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail19 winner := by
  rw [root11Tail18, scan, if_neg (by decide), root11Candidates18_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail19
    (if 3 ≤ root11Candidates18.length then search edge 3 [43, 2389] root11Candidates18 2432 winner else winner) = _
  rw [if_pos (by decide), root11Child18]
def root11Candidates19 : List Nat := [6967, 23251]
set_option maxRecDepth 8192 in
theorem root11Candidates19_checked : candidates edge 4 43 2503 root11Tail20 winner.bound = root11Candidates19 := by
  unfold candidates
  have row : edge 2503 = (SortedNatTree.contains FastRows.row2503) := by
    change Data.row2503.contains = (SortedNatTree.contains FastRows.row2503)
    exact FastRows.row2503_checked.symm
  rw [row]
  decide
theorem root11Step19 : scan edge (search edge 3) 4 [43] 43 root11Tail19 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail20 winner := by
  rw [root11Tail19, scan, if_neg (by decide), root11Candidates19_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail20
    (if 3 ≤ root11Candidates19.length then search edge 3 [43, 2503] root11Candidates19 2546 winner else winner) = _
  rw [if_neg (by decide)]
def root11Candidates20 : List Nat := [4243, 8521, 12157, 19963]
set_option maxRecDepth 8192 in
theorem root11Candidates20_checked : candidates edge 4 43 2557 root11Tail21 winner.bound = root11Candidates20 := by
  unfold candidates
  have row : edge 2557 = (SortedNatTree.contains FastRows.row2557) := by
    change Data.row2557.contains = (SortedNatTree.contains FastRows.row2557)
    exact FastRows.row2557_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root11Child20 : search edge 3 [43, 2557] root11Candidates20 2600 winner = winner := by decide
theorem root11Step20 : scan edge (search edge 3) 4 [43] 43 root11Tail20 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail21 winner := by
  rw [root11Tail20, scan, if_neg (by decide), root11Candidates20_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail21
    (if 3 ≤ root11Candidates20.length then search edge 3 [43, 2557] root11Candidates20 2600 winner else winner) = _
  rw [if_pos (by decide), root11Child20]
def root11Candidates21 : List Nat := [12073, 14281, 17191, 19963, 22741]
set_option maxRecDepth 8192 in
theorem root11Candidates21_checked : candidates edge 4 43 2749 root11Tail22 winner.bound = root11Candidates21 := by
  unfold candidates
  have row : edge 2749 = (SortedNatTree.contains FastRows.row2749) := by
    change Data.row2749.contains = (SortedNatTree.contains FastRows.row2749)
    exact FastRows.row2749_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root11Child21 : search edge 3 [43, 2749] root11Candidates21 2792 winner = winner := by decide
theorem root11Step21 : scan edge (search edge 3) 4 [43] 43 root11Tail21 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail22 winner := by
  rw [root11Tail21, scan, if_neg (by decide), root11Candidates21_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail22
    (if 3 ≤ root11Candidates21.length then search edge 3 [43, 2749] root11Candidates21 2792 winner else winner) = _
  rw [if_pos (by decide), root11Child21]
def root11Candidates22 : List Nat := [3259, 9787]
set_option maxRecDepth 8192 in
theorem root11Candidates22_checked : candidates edge 4 43 3049 root11Tail23 winner.bound = root11Candidates22 := by
  unfold candidates
  have row : edge 3049 = (SortedNatTree.contains FastRows.row3049) := by
    change Data.row3049.contains = (SortedNatTree.contains FastRows.row3049)
    exact FastRows.row3049_checked.symm
  rw [row]
  decide
theorem root11Step22 : scan edge (search edge 3) 4 [43] 43 root11Tail22 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail23 winner := by
  rw [root11Tail22, scan, if_neg (by decide), root11Candidates22_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail23
    (if 3 ≤ root11Candidates22.length then search edge 3 [43, 3049] root11Candidates22 3092 winner else winner) = _
  rw [if_neg (by decide)]
def root11Candidates23 : List Nat := []
set_option maxRecDepth 8192 in
theorem root11Candidates23_checked : candidates edge 4 43 3187 root11Tail24 winner.bound = root11Candidates23 := by
  unfold candidates
  have row : edge 3187 = (SortedNatTree.contains FastRows.row3187) := by
    change Data.row3187.contains = (SortedNatTree.contains FastRows.row3187)
    exact FastRows.row3187_checked.symm
  rw [row]
  decide
theorem root11Step23 : scan edge (search edge 3) 4 [43] 43 root11Tail23 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail24 winner := by
  rw [root11Tail23, scan, if_neg (by decide), root11Candidates23_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail24
    (if 3 ≤ root11Candidates23.length then search edge 3 [43, 3187] root11Candidates23 3230 winner else winner) = _
  rw [if_neg (by decide)]
def root11Candidates24 : List Nat := [9973, 12073]
set_option maxRecDepth 8192 in
theorem root11Candidates24_checked : candidates edge 4 43 3253 root11Tail25 winner.bound = root11Candidates24 := by
  unfold candidates
  have row : edge 3253 = (SortedNatTree.contains FastRows.row3253) := by
    change Data.row3253.contains = (SortedNatTree.contains FastRows.row3253)
    exact FastRows.row3253_checked.symm
  rw [row]
  decide
theorem root11Step24 : scan edge (search edge 3) 4 [43] 43 root11Tail24 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail25 winner := by
  rw [root11Tail24, scan, if_neg (by decide), root11Candidates24_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail25
    (if 3 ≤ root11Candidates24.length then search edge 3 [43, 3253] root11Candidates24 3296 winner else winner) = _
  rw [if_neg (by decide)]
def root11Candidates25 : List Nat := [5569]
set_option maxRecDepth 8192 in
theorem root11Candidates25_checked : candidates edge 4 43 3259 root11Tail26 winner.bound = root11Candidates25 := by
  unfold candidates
  have row : edge 3259 = (SortedNatTree.contains FastRows.row3259) := by
    change Data.row3259.contains = (SortedNatTree.contains FastRows.row3259)
    exact FastRows.row3259_checked.symm
  rw [row]
  decide
theorem root11Step25 : scan edge (search edge 3) 4 [43] 43 root11Tail25 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail26 winner := by
  rw [root11Tail25, scan, if_neg (by decide), root11Candidates25_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail26
    (if 3 ≤ root11Candidates25.length then search edge 3 [43, 3259] root11Candidates25 3302 winner else winner) = _
  rw [if_neg (by decide)]
def root11Candidates26 : List Nat := [3373, 4363]
set_option maxRecDepth 8192 in
theorem root11Candidates26_checked : candidates edge 4 43 3319 root11Tail27 winner.bound = root11Candidates26 := by
  unfold candidates
  have row : edge 3319 = (SortedNatTree.contains FastRows.row3319) := by
    change Data.row3319.contains = (SortedNatTree.contains FastRows.row3319)
    exact FastRows.row3319_checked.symm
  rw [row]
  decide
theorem root11Step26 : scan edge (search edge 3) 4 [43] 43 root11Tail26 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail27 winner := by
  rw [root11Tail26, scan, if_neg (by decide), root11Candidates26_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail27
    (if 3 ≤ root11Candidates26.length then search edge 3 [43, 3319] root11Candidates26 3362 winner else winner) = _
  rw [if_neg (by decide)]
def root11Candidates27 : List Nat := [6343, 15607, 21613]
set_option maxRecDepth 8192 in
theorem root11Candidates27_checked : candidates edge 4 43 3361 root11Tail28 winner.bound = root11Candidates27 := by
  unfold candidates
  have row : edge 3361 = (SortedNatTree.contains FastRows.row3361) := by
    change Data.row3361.contains = (SortedNatTree.contains FastRows.row3361)
    exact FastRows.row3361_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root11Child27 : search edge 3 [43, 3361] root11Candidates27 3404 winner = winner := by decide
theorem root11Step27 : scan edge (search edge 3) 4 [43] 43 root11Tail27 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail28 winner := by
  rw [root11Tail27, scan, if_neg (by decide), root11Candidates27_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail28
    (if 3 ≤ root11Candidates27.length then search edge 3 [43, 3361] root11Candidates27 3404 winner else winner) = _
  rw [if_pos (by decide), root11Child27]
def root11Candidates28 : List Nat := [12697]
set_option maxRecDepth 8192 in
theorem root11Candidates28_checked : candidates edge 4 43 3373 root11Tail29 winner.bound = root11Candidates28 := by
  unfold candidates
  have row : edge 3373 = (SortedNatTree.contains FastRows.row3373) := by
    change Data.row3373.contains = (SortedNatTree.contains FastRows.row3373)
    exact FastRows.row3373_checked.symm
  rw [row]
  decide
theorem root11Step28 : scan edge (search edge 3) 4 [43] 43 root11Tail28 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail29 winner := by
  rw [root11Tail28, scan, if_neg (by decide), root11Candidates28_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail29
    (if 3 ≤ root11Candidates28.length then search edge 3 [43, 3373] root11Candidates28 3416 winner else winner) = _
  rw [if_neg (by decide)]
def root11Candidates29 : List Nat := []
set_option maxRecDepth 8192 in
theorem root11Candidates29_checked : candidates edge 4 43 3469 root11Tail30 winner.bound = root11Candidates29 := by
  unfold candidates
  have row : edge 3469 = (SortedNatTree.contains FastRows.row3469) := by
    change Data.row3469.contains = (SortedNatTree.contains FastRows.row3469)
    exact FastRows.row3469_checked.symm
  rw [row]
  decide
theorem root11Step29 : scan edge (search edge 3) 4 [43] 43 root11Tail29 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail30 winner := by
  rw [root11Tail29, scan, if_neg (by decide), root11Candidates29_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail30
    (if 3 ≤ root11Candidates29.length then search edge 3 [43, 3469] root11Candidates29 3512 winner else winner) = _
  rw [if_neg (by decide)]
def root11Candidates30 : List Nat := [7159]
set_option maxRecDepth 8192 in
theorem root11Candidates30_checked : candidates edge 4 43 3877 root11Tail31 winner.bound = root11Candidates30 := by
  unfold candidates
  have row : edge 3877 = (SortedNatTree.contains FastRows.row3877) := by
    change Data.row3877.contains = (SortedNatTree.contains FastRows.row3877)
    exact FastRows.row3877_checked.symm
  rw [row]
  decide
theorem root11Step30 : scan edge (search edge 3) 4 [43] 43 root11Tail30 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail31 winner := by
  rw [root11Tail30, scan, if_neg (by decide), root11Candidates30_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail31
    (if 3 ≤ root11Candidates30.length then search edge 3 [43, 3877] root11Candidates30 3920 winner else winner) = _
  rw [if_neg (by decide)]
def root11Candidates31 : List Nat := [21397]
set_option maxRecDepth 8192 in
theorem root11Candidates31_checked : candidates edge 4 43 3931 root11Tail32 winner.bound = root11Candidates31 := by
  unfold candidates
  have row : edge 3931 = (SortedNatTree.contains FastRows.row3931) := by
    change Data.row3931.contains = (SortedNatTree.contains FastRows.row3931)
    exact FastRows.row3931_checked.symm
  rw [row]
  decide
theorem root11Step31 : scan edge (search edge 3) 4 [43] 43 root11Tail31 winner =
    scan edge (search edge 3) 4 [43] 43 root11Tail32 winner := by
  rw [root11Tail31, scan, if_neg (by decide), root11Candidates31_checked]
  change scan edge (search edge 3) 4 [43] 43 root11Tail32
    (if 3 ≤ root11Candidates31.length then search edge 3 [43, 3931] root11Candidates31 3974 winner else winner) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
