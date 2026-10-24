import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot007Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root7Candidates0 : List Nat := [2477, 3863, 4673, 5237, 5441, 6563, 7589, 7841, 9239, 9419, 13553, 17939, 21767]
set_option maxRecDepth 8192 in
theorem root7Candidates0_checked : candidates edge 4 29 71 root7Tail1 winner.bound = root7Candidates0 := by
  unfold candidates
  have row : edge 71 = (SortedNatTree.contains FastRows.row71) := by
    change Data.row71.contains = (SortedNatTree.contains FastRows.row71)
    exact FastRows.row71_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child0 : search edge 3 [29, 71] root7Candidates0 100 winner = winner := by decide
theorem root7Step0 : scan edge (search edge 3) 4 [29] 29 root7Tail0 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail1 winner := by
  rw [root7Tail0, scan, if_neg (by decide), root7Candidates0_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail1
    (if 3 ≤ root7Candidates0.length then search edge 3 [29, 71] root7Candidates0 100 winner else winner) = _
  rw [if_pos (by decide), root7Child0]
def root7Candidates1 : List Nat := [5237, 7757, 9239, 9857, 12473, 13187, 16931, 20543, 22721, 24533]
set_option maxRecDepth 8192 in
theorem root7Candidates1_checked : candidates edge 4 29 137 root7Tail2 winner.bound = root7Candidates1 := by
  unfold candidates
  have row : edge 137 = (SortedNatTree.contains FastRows.row137) := by
    change Data.row137.contains = (SortedNatTree.contains FastRows.row137)
    exact FastRows.row137_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child1 : search edge 3 [29, 137] root7Candidates1 166 winner = winner := by decide
theorem root7Step1 : scan edge (search edge 3) 4 [29] 29 root7Tail1 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail2 winner := by
  rw [root7Tail1, scan, if_neg (by decide), root7Candidates1_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail2
    (if 3 ≤ root7Candidates1.length then search edge 3 [29, 137] root7Candidates1 166 winner else winner) = _
  rw [if_pos (by decide), root7Child1]
def root7Candidates2 : List Nat := [2441, 5441, 6827, 9521, 9857]
set_option maxRecDepth 8192 in
theorem root7Candidates2_checked : candidates edge 4 29 167 root7Tail3 winner.bound = root7Candidates2 := by
  unfold candidates
  have row : edge 167 = (SortedNatTree.contains FastRows.row167) := by
    change Data.row167.contains = (SortedNatTree.contains FastRows.row167)
    exact FastRows.row167_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child2 : search edge 3 [29, 167] root7Candidates2 196 winner = winner := by decide
theorem root7Step2 : scan edge (search edge 3) 4 [29] 29 root7Tail2 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail3 winner := by
  rw [root7Tail2, scan, if_neg (by decide), root7Candidates2_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail3
    (if 3 ≤ root7Candidates2.length then search edge 3 [29, 167] root7Candidates2 196 winner else winner) = _
  rw [if_pos (by decide), root7Child2]
def root7Candidates3 : List Nat := [383, 2309, 4787, 9311, 10463, 12641, 22343]
set_option maxRecDepth 8192 in
theorem root7Candidates3_checked : candidates edge 4 29 179 root7Tail4 winner.bound = root7Candidates3 := by
  unfold candidates
  have row : edge 179 = (SortedNatTree.contains FastRows.row179) := by
    change Data.row179.contains = (SortedNatTree.contains FastRows.row179)
    exact FastRows.row179_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child3 : search edge 3 [29, 179] root7Candidates3 208 winner = winner := by decide
theorem root7Step3 : scan edge (search edge 3) 4 [29] 29 root7Tail3 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail4 winner := by
  rw [root7Tail3, scan, if_neg (by decide), root7Candidates3_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail4
    (if 3 ≤ root7Candidates3.length then search edge 3 [29, 179] root7Candidates3 208 winner else winner) = _
  rw [if_pos (by decide), root7Child3]
def root7Candidates4 : List Nat := [401, 5861, 7841, 9857]
set_option maxRecDepth 8192 in
theorem root7Candidates4_checked : candidates edge 4 29 347 root7Tail5 winner.bound = root7Candidates4 := by
  unfold candidates
  have row : edge 347 = (SortedNatTree.contains FastRows.row347) := by
    change Data.row347.contains = (SortedNatTree.contains FastRows.row347)
    exact FastRows.row347_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child4 : search edge 3 [29, 347] root7Candidates4 376 winner = winner := by decide
theorem root7Step4 : scan edge (search edge 3) 4 [29] 29 root7Tail4 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail5 winner := by
  rw [root7Tail4, scan, if_neg (by decide), root7Candidates4_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail5
    (if 3 ≤ root7Candidates4.length then search edge 3 [29, 347] root7Candidates4 376 winner else winner) = _
  rw [if_pos (by decide), root7Child4]
def root7Candidates5 : List Nat := [3257, 7853, 9239, 11621, 22721]
set_option maxRecDepth 8192 in
theorem root7Candidates5_checked : candidates edge 4 29 383 root7Tail6 winner.bound = root7Candidates5 := by
  unfold candidates
  have row : edge 383 = (SortedNatTree.contains FastRows.row383) := by
    change Data.row383.contains = (SortedNatTree.contains FastRows.row383)
    exact FastRows.row383_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child5 : search edge 3 [29, 383] root7Candidates5 412 winner = winner := by decide
theorem root7Step5 : scan edge (search edge 3) 4 [29] 29 root7Tail5 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail6 winner := by
  rw [root7Tail5, scan, if_neg (by decide), root7Candidates5_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail6
    (if 3 ≤ root7Candidates5.length then search edge 3 [29, 383] root7Candidates5 412 winner else winner) = _
  rw [if_pos (by decide), root7Child5]
def root7Candidates6 : List Nat := [2441, 7589, 12893]
set_option maxRecDepth 8192 in
theorem root7Candidates6_checked : candidates edge 4 29 401 root7Tail7 winner.bound = root7Candidates6 := by
  unfold candidates
  have row : edge 401 = (SortedNatTree.contains FastRows.row401) := by
    change Data.row401.contains = (SortedNatTree.contains FastRows.row401)
    exact FastRows.row401_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child6 : search edge 3 [29, 401] root7Candidates6 430 winner = winner := by decide
theorem root7Step6 : scan edge (search edge 3) 4 [29] 29 root7Tail6 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail7 winner := by
  rw [root7Tail6, scan, if_neg (by decide), root7Candidates6_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail7
    (if 3 ≤ root7Candidates6.length then search edge 3 [29, 401] root7Candidates6 430 winner else winner) = _
  rw [if_pos (by decide), root7Child6]
def root7Candidates7 : List Nat := [4673, 9311, 11159, 12227, 13187, 20399]
set_option maxRecDepth 8192 in
theorem root7Candidates7_checked : candidates edge 4 29 569 root7Tail8 winner.bound = root7Candidates7 := by
  unfold candidates
  have row : edge 569 = (SortedNatTree.contains FastRows.row569) := by
    change Data.row569.contains = (SortedNatTree.contains FastRows.row569)
    exact FastRows.row569_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child7 : search edge 3 [29, 569] root7Candidates7 598 winner = winner := by decide
theorem root7Step7 : scan edge (search edge 3) 4 [29] 29 root7Tail7 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail8 winner := by
  rw [root7Tail7, scan, if_neg (by decide), root7Candidates7_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail8
    (if 3 ≤ root7Candidates7.length then search edge 3 [29, 569] root7Candidates7 598 winner else winner) = _
  rw [if_pos (by decide), root7Child7]
def root7Candidates8 : List Nat := [5081, 6047, 7841, 12227, 12473, 15923, 17477]
set_option maxRecDepth 8192 in
theorem root7Candidates8_checked : candidates edge 4 29 599 root7Tail9 winner.bound = root7Candidates8 := by
  unfold candidates
  have row : edge 599 = (SortedNatTree.contains FastRows.row599) := by
    change Data.row599.contains = (SortedNatTree.contains FastRows.row599)
    exact FastRows.row599_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child8 : search edge 3 [29, 599] root7Candidates8 628 winner = winner := by decide
theorem root7Step8 : scan edge (search edge 3) 4 [29] 29 root7Tail8 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail9 winner := by
  rw [root7Tail8, scan, if_neg (by decide), root7Candidates8_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail9
    (if 3 ≤ root7Candidates8.length then search edge 3 [29, 599] root7Candidates8 628 winner else winner) = _
  rw [if_pos (by decide), root7Child8]
def root7Candidates9 : List Nat := [6047, 8861, 9311, 12641, 21089]
set_option maxRecDepth 8192 in
theorem root7Candidates9_checked : candidates edge 4 29 683 root7Tail10 winner.bound = root7Candidates9 := by
  unfold candidates
  have row : edge 683 = (SortedNatTree.contains FastRows.row683) := by
    change Data.row683.contains = (SortedNatTree.contains FastRows.row683)
    exact FastRows.row683_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child9 : search edge 3 [29, 683] root7Candidates9 712 winner = winner := by decide
theorem root7Step9 : scan edge (search edge 3) 4 [29] 29 root7Tail9 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail10 winner := by
  rw [root7Tail9, scan, if_neg (by decide), root7Candidates9_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail10
    (if 3 ≤ root7Candidates9.length then search edge 3 [29, 683] root7Candidates9 712 winner else winner) = _
  rw [if_pos (by decide), root7Child9]
def root7Candidates10 : List Nat := [6911, 12527, 13619]
set_option maxRecDepth 8192 in
theorem root7Candidates10_checked : candidates edge 4 29 761 root7Tail11 winner.bound = root7Candidates10 := by
  unfold candidates
  have row : edge 761 = (SortedNatTree.contains FastRows.row761) := by
    change Data.row761.contains = (SortedNatTree.contains FastRows.row761)
    exact FastRows.row761_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child10 : search edge 3 [29, 761] root7Candidates10 790 winner = winner := by decide
theorem root7Step10 : scan edge (search edge 3) 4 [29] 29 root7Tail10 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail11 winner := by
  rw [root7Tail10, scan, if_neg (by decide), root7Candidates10_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail11
    (if 3 ≤ root7Candidates10.length then search edge 3 [29, 761] root7Candidates10 790 winner else winner) = _
  rw [if_pos (by decide), root7Child10]
def root7Candidates11 : List Nat := [6827, 7607, 16811]
set_option maxRecDepth 8192 in
theorem root7Candidates11_checked : candidates edge 4 29 881 root7Tail12 winner.bound = root7Candidates11 := by
  unfold candidates
  have row : edge 881 = (SortedNatTree.contains FastRows.row881) := by
    change Data.row881.contains = (SortedNatTree.contains FastRows.row881)
    exact FastRows.row881_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child11 : search edge 3 [29, 881] root7Candidates11 910 winner = winner := by decide
theorem root7Step11 : scan edge (search edge 3) 4 [29] 29 root7Tail11 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail12 winner := by
  rw [root7Tail11, scan, if_neg (by decide), root7Candidates11_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail12
    (if 3 ≤ root7Candidates11.length then search edge 3 [29, 881] root7Candidates11 910 winner else winner) = _
  rw [if_pos (by decide), root7Child11]
def root7Candidates12 : List Nat := [2801, 2969, 5861, 7589, 15923]
set_option maxRecDepth 8192 in
theorem root7Candidates12_checked : candidates edge 4 29 1901 root7Tail13 winner.bound = root7Candidates12 := by
  unfold candidates
  have row : edge 1901 = (SortedNatTree.contains FastRows.row1901) := by
    change Data.row1901.contains = (SortedNatTree.contains FastRows.row1901)
    exact FastRows.row1901_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child12 : search edge 3 [29, 1901] root7Candidates12 1930 winner = winner := by decide
theorem root7Step12 : scan edge (search edge 3) 4 [29] 29 root7Tail12 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail13 winner := by
  rw [root7Tail12, scan, if_neg (by decide), root7Candidates12_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail13
    (if 3 ≤ root7Candidates12.length then search edge 3 [29, 1901] root7Candidates12 1930 winner else winner) = _
  rw [if_pos (by decide), root7Child12]
def root7Candidates13 : List Nat := [4397]
set_option maxRecDepth 8192 in
theorem root7Candidates13_checked : candidates edge 4 29 1997 root7Tail14 winner.bound = root7Candidates13 := by
  unfold candidates
  have row : edge 1997 = (SortedNatTree.contains FastRows.row1997) := by
    change Data.row1997.contains = (SortedNatTree.contains FastRows.row1997)
    exact FastRows.row1997_checked.symm
  rw [row]
  decide
theorem root7Step13 : scan edge (search edge 3) 4 [29] 29 root7Tail13 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail14 winner := by
  rw [root7Tail13, scan, if_neg (by decide), root7Candidates13_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail14
    (if 3 ≤ root7Candidates13.length then search edge 3 [29, 1997] root7Candidates13 2026 winner else winner) = _
  rw [if_neg (by decide)]
def root7Candidates14 : List Nat := [6911, 10463, 22853]
set_option maxRecDepth 8192 in
theorem root7Candidates14_checked : candidates edge 4 29 2027 root7Tail15 winner.bound = root7Candidates14 := by
  unfold candidates
  have row : edge 2027 = (SortedNatTree.contains FastRows.row2027) := by
    change Data.row2027.contains = (SortedNatTree.contains FastRows.row2027)
    exact FastRows.row2027_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child14 : search edge 3 [29, 2027] root7Candidates14 2056 winner = winner := by decide
theorem root7Step14 : scan edge (search edge 3) 4 [29] 29 root7Tail14 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail15 winner := by
  rw [root7Tail14, scan, if_neg (by decide), root7Candidates14_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail15
    (if 3 ≤ root7Candidates14.length then search edge 3 [29, 2027] root7Candidates14 2056 winner else winner) = _
  rw [if_pos (by decide), root7Child14]
def root7Candidates15 : List Nat := [5441, 7793, 9857, 12473]
set_option maxRecDepth 8192 in
theorem root7Candidates15_checked : candidates edge 4 29 2081 root7Tail16 winner.bound = root7Candidates15 := by
  unfold candidates
  have row : edge 2081 = (SortedNatTree.contains FastRows.row2081) := by
    change Data.row2081.contains = (SortedNatTree.contains FastRows.row2081)
    exact FastRows.row2081_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child15 : search edge 3 [29, 2081] root7Candidates15 2110 winner = winner := by decide
theorem root7Step15 : scan edge (search edge 3) 4 [29] 29 root7Tail15 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail16 winner := by
  rw [root7Tail15, scan, if_neg (by decide), root7Candidates15_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail16
    (if 3 ≤ root7Candidates15.length then search edge 3 [29, 2081] root7Candidates15 2110 winner else winner) = _
  rw [if_pos (by decide), root7Child15]
def root7Candidates16 : List Nat := [2309, 19013, 20903]
set_option maxRecDepth 8192 in
theorem root7Candidates16_checked : candidates edge 4 29 2141 root7Tail17 winner.bound = root7Candidates16 := by
  unfold candidates
  have row : edge 2141 = (SortedNatTree.contains FastRows.row2141) := by
    change Data.row2141.contains = (SortedNatTree.contains FastRows.row2141)
    exact FastRows.row2141_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child16 : search edge 3 [29, 2141] root7Candidates16 2170 winner = winner := by decide
theorem root7Step16 : scan edge (search edge 3) 4 [29] 29 root7Tail16 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail17 winner := by
  rw [root7Tail16, scan, if_neg (by decide), root7Candidates16_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail17
    (if 3 ≤ root7Candidates16.length then search edge 3 [29, 2141] root7Candidates16 2170 winner else winner) = _
  rw [if_pos (by decide), root7Child16]
def root7Candidates17 : List Nat := [5081, 8237, 8861, 22721]
set_option maxRecDepth 8192 in
theorem root7Candidates17_checked : candidates edge 4 29 2309 root7Tail18 winner.bound = root7Candidates17 := by
  unfold candidates
  have row : edge 2309 = (SortedNatTree.contains FastRows.row2309) := by
    change Data.row2309.contains = (SortedNatTree.contains FastRows.row2309)
    exact FastRows.row2309_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child17 : search edge 3 [29, 2309] root7Candidates17 2338 winner = winner := by decide
theorem root7Step17 : scan edge (search edge 3) 4 [29] 29 root7Tail17 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail18 winner := by
  rw [root7Tail17, scan, if_neg (by decide), root7Candidates17_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail18
    (if 3 ≤ root7Candidates17.length then search edge 3 [29, 2309] root7Candidates17 2338 winner else winner) = _
  rw [if_pos (by decide), root7Child17]
def root7Candidates18 : List Nat := [4241]
set_option maxRecDepth 8192 in
theorem root7Candidates18_checked : candidates edge 4 29 2393 root7Tail19 winner.bound = root7Candidates18 := by
  unfold candidates
  have row : edge 2393 = (SortedNatTree.contains FastRows.row2393) := by
    change Data.row2393.contains = (SortedNatTree.contains FastRows.row2393)
    exact FastRows.row2393_checked.symm
  rw [row]
  decide
theorem root7Step18 : scan edge (search edge 3) 4 [29] 29 root7Tail18 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail19 winner := by
  rw [root7Tail18, scan, if_neg (by decide), root7Candidates18_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail19
    (if 3 ≤ root7Candidates18.length then search edge 3 [29, 2393] root7Candidates18 2422 winner else winner) = _
  rw [if_neg (by decide)]
def root7Candidates19 : List Nat := [4157, 4787, 10169, 22709]
set_option maxRecDepth 8192 in
theorem root7Candidates19_checked : candidates edge 4 29 2441 root7Tail20 winner.bound = root7Candidates19 := by
  unfold candidates
  have row : edge 2441 = (SortedNatTree.contains FastRows.row2441) := by
    change Data.row2441.contains = (SortedNatTree.contains FastRows.row2441)
    exact FastRows.row2441_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child19 : search edge 3 [29, 2441] root7Candidates19 2470 winner = winner := by decide
theorem root7Step19 : scan edge (search edge 3) 4 [29] 29 root7Tail19 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail20 winner := by
  rw [root7Tail19, scan, if_neg (by decide), root7Candidates19_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail20
    (if 3 ≤ root7Candidates19.length then search edge 3 [29, 2441] root7Candidates19 2470 winner else winner) = _
  rw [if_pos (by decide), root7Child19]
def root7Candidates20 : List Nat := [2801, 5039, 7853, 19013]
set_option maxRecDepth 8192 in
theorem root7Candidates20_checked : candidates edge 4 29 2477 root7Tail21 winner.bound = root7Candidates20 := by
  unfold candidates
  have row : edge 2477 = (SortedNatTree.contains FastRows.row2477) := by
    change Data.row2477.contains = (SortedNatTree.contains FastRows.row2477)
    exact FastRows.row2477_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child20 : search edge 3 [29, 2477] root7Candidates20 2506 winner = winner := by decide
theorem root7Step20 : scan edge (search edge 3) 4 [29] 29 root7Tail20 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail21 winner := by
  rw [root7Tail20, scan, if_neg (by decide), root7Candidates20_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail21
    (if 3 ≤ root7Candidates20.length then search edge 3 [29, 2477] root7Candidates20 2506 winner else winner) = _
  rw [if_pos (by decide), root7Child20]
def root7Candidates21 : List Nat := [4241, 7793, 9743, 13553, 15467, 19013]
set_option maxRecDepth 8192 in
theorem root7Candidates21_checked : candidates edge 4 29 2549 root7Tail22 winner.bound = root7Candidates21 := by
  unfold candidates
  have row : edge 2549 = (SortedNatTree.contains FastRows.row2549) := by
    change Data.row2549.contains = (SortedNatTree.contains FastRows.row2549)
    exact FastRows.row2549_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child21 : search edge 3 [29, 2549] root7Candidates21 2578 winner = winner := by decide
theorem root7Step21 : scan edge (search edge 3) 4 [29] 29 root7Tail21 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail22 winner := by
  rw [root7Tail21, scan, if_neg (by decide), root7Candidates21_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail22
    (if 3 ≤ root7Candidates21.length then search edge 3 [29, 2549] root7Candidates21 2578 winner else winner) = _
  rw [if_pos (by decide), root7Child21]
def root7Candidates22 : List Nat := [5441, 7589]
set_option maxRecDepth 8192 in
theorem root7Candidates22_checked : candidates edge 4 29 2711 root7Tail23 winner.bound = root7Candidates22 := by
  unfold candidates
  have row : edge 2711 = (SortedNatTree.contains FastRows.row2711) := by
    change Data.row2711.contains = (SortedNatTree.contains FastRows.row2711)
    exact FastRows.row2711_checked.symm
  rw [row]
  decide
theorem root7Step22 : scan edge (search edge 3) 4 [29] 29 root7Tail22 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail23 winner := by
  rw [root7Tail22, scan, if_neg (by decide), root7Candidates22_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail23
    (if 3 ≤ root7Candidates22.length then search edge 3 [29, 2711] root7Candidates22 2740 winner else winner) = _
  rw [if_neg (by decide)]
def root7Candidates23 : List Nat := [3803, 9743, 9857, 13187, 13451, 18719]
set_option maxRecDepth 8192 in
theorem root7Candidates23_checked : candidates edge 4 29 2801 root7Tail24 winner.bound = root7Candidates23 := by
  unfold candidates
  have row : edge 2801 = (SortedNatTree.contains FastRows.row2801) := by
    change Data.row2801.contains = (SortedNatTree.contains FastRows.row2801)
    exact FastRows.row2801_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child23 : search edge 3 [29, 2801] root7Candidates23 2830 winner = winner := by decide
theorem root7Step23 : scan edge (search edge 3) 4 [29] 29 root7Tail23 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail24 winner := by
  rw [root7Tail23, scan, if_neg (by decide), root7Candidates23_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail24
    (if 3 ≤ root7Candidates23.length then search edge 3 [29, 2801] root7Candidates23 2830 winner else winner) = _
  rw [if_pos (by decide), root7Child23]
def root7Candidates24 : List Nat := [7841, 8861, 13451]
set_option maxRecDepth 8192 in
theorem root7Candidates24_checked : candidates edge 4 29 2969 root7Tail25 winner.bound = root7Candidates24 := by
  unfold candidates
  have row : edge 2969 = (SortedNatTree.contains FastRows.row2969) := by
    change Data.row2969.contains = (SortedNatTree.contains FastRows.row2969)
    exact FastRows.row2969_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child24 : search edge 3 [29, 2969] root7Candidates24 2998 winner = winner := by decide
theorem root7Step24 : scan edge (search edge 3) 4 [29] 29 root7Tail24 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail25 winner := by
  rw [root7Tail24, scan, if_neg (by decide), root7Candidates24_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail25
    (if 3 ≤ root7Candidates24.length then search edge 3 [29, 2969] root7Candidates24 2998 winner else winner) = _
  rw [if_pos (by decide), root7Child24]
def root7Candidates25 : List Nat := [5441, 6221]
set_option maxRecDepth 8192 in
theorem root7Candidates25_checked : candidates edge 4 29 3257 root7Tail26 winner.bound = root7Candidates25 := by
  unfold candidates
  have row : edge 3257 = (SortedNatTree.contains FastRows.row3257) := by
    change Data.row3257.contains = (SortedNatTree.contains FastRows.row3257)
    exact FastRows.row3257_checked.symm
  rw [row]
  decide
theorem root7Step25 : scan edge (search edge 3) 4 [29] 29 root7Tail25 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail26 winner := by
  rw [root7Tail25, scan, if_neg (by decide), root7Candidates25_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail26
    (if 3 ≤ root7Candidates25.length then search edge 3 [29, 3257] root7Candidates25 3286 winner else winner) = _
  rw [if_neg (by decide)]
def root7Candidates26 : List Nat := [3803, 4157, 6047, 14741]
set_option maxRecDepth 8192 in
theorem root7Candidates26_checked : candidates edge 4 29 3659 root7Tail27 winner.bound = root7Candidates26 := by
  unfold candidates
  have row : edge 3659 = (SortedNatTree.contains FastRows.row3659) := by
    change Data.row3659.contains = (SortedNatTree.contains FastRows.row3659)
    exact FastRows.row3659_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child26 : search edge 3 [29, 3659] root7Candidates26 3688 winner = winner := by decide
theorem root7Step26 : scan edge (search edge 3) 4 [29] 29 root7Tail26 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail27 winner := by
  rw [root7Tail26, scan, if_neg (by decide), root7Candidates26_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail27
    (if 3 ≤ root7Candidates26.length then search edge 3 [29, 3659] root7Candidates26 3688 winner else winner) = _
  rw [if_pos (by decide), root7Child26]
def root7Candidates27 : List Nat := [4523, 6911, 7487, 7793]
set_option maxRecDepth 8192 in
theorem root7Candidates27_checked : candidates edge 4 29 3767 root7Tail28 winner.bound = root7Candidates27 := by
  unfold candidates
  have row : edge 3767 = (SortedNatTree.contains FastRows.row3767) := by
    change Data.row3767.contains = (SortedNatTree.contains FastRows.row3767)
    exact FastRows.row3767_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child27 : search edge 3 [29, 3767] root7Candidates27 3796 winner = winner := by decide
theorem root7Step27 : scan edge (search edge 3) 4 [29] 29 root7Tail27 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail28 winner := by
  rw [root7Tail27, scan, if_neg (by decide), root7Candidates27_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail28
    (if 3 ≤ root7Candidates27.length then search edge 3 [29, 3767] root7Candidates27 3796 winner else winner) = _
  rw [if_pos (by decide), root7Child27]
def root7Candidates28 : List Nat := [17939]
set_option maxRecDepth 8192 in
theorem root7Candidates28_checked : candidates edge 4 29 3803 root7Tail29 winner.bound = root7Candidates28 := by
  unfold candidates
  have row : edge 3803 = (SortedNatTree.contains FastRows.row3803) := by
    change Data.row3803.contains = (SortedNatTree.contains FastRows.row3803)
    exact FastRows.row3803_checked.symm
  rw [row]
  decide
theorem root7Step28 : scan edge (search edge 3) 4 [29] 29 root7Tail28 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail29 winner := by
  rw [root7Tail28, scan, if_neg (by decide), root7Candidates28_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail29
    (if 3 ≤ root7Candidates28.length then search edge 3 [29, 3803] root7Candidates28 3832 winner else winner) = _
  rw [if_neg (by decide)]
def root7Candidates29 : List Nat := [7757, 18719, 19013]
set_option maxRecDepth 8192 in
theorem root7Candidates29_checked : candidates edge 4 29 3863 root7Tail30 winner.bound = root7Candidates29 := by
  unfold candidates
  have row : edge 3863 = (SortedNatTree.contains FastRows.row3863) := by
    change Data.row3863.contains = (SortedNatTree.contains FastRows.row3863)
    exact FastRows.row3863_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root7Child29 : search edge 3 [29, 3863] root7Candidates29 3892 winner = winner := by decide
theorem root7Step29 : scan edge (search edge 3) 4 [29] 29 root7Tail29 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail30 winner := by
  rw [root7Tail29, scan, if_neg (by decide), root7Candidates29_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail30
    (if 3 ≤ root7Candidates29.length then search edge 3 [29, 3863] root7Candidates29 3892 winner else winner) = _
  rw [if_pos (by decide), root7Child29]
def root7Candidates30 : List Nat := [20903]
set_option maxRecDepth 8192 in
theorem root7Candidates30_checked : candidates edge 4 29 4013 root7Tail31 winner.bound = root7Candidates30 := by
  unfold candidates
  have row : edge 4013 = (SortedNatTree.contains FastRows.row4013) := by
    change Data.row4013.contains = (SortedNatTree.contains FastRows.row4013)
    exact FastRows.row4013_checked.symm
  rw [row]
  decide
theorem root7Step30 : scan edge (search edge 3) 4 [29] 29 root7Tail30 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail31 winner := by
  rw [root7Tail30, scan, if_neg (by decide), root7Candidates30_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail31
    (if 3 ≤ root7Candidates30.length then search edge 3 [29, 4013] root7Candidates30 4042 winner else winner) = _
  rw [if_neg (by decide)]
def root7Candidates31 : List Nat := [11159, 20693]
set_option maxRecDepth 8192 in
theorem root7Candidates31_checked : candidates edge 4 29 4157 root7Tail32 winner.bound = root7Candidates31 := by
  unfold candidates
  have row : edge 4157 = (SortedNatTree.contains FastRows.row4157) := by
    change Data.row4157.contains = (SortedNatTree.contains FastRows.row4157)
    exact FastRows.row4157_checked.symm
  rw [row]
  decide
theorem root7Step31 : scan edge (search edge 3) 4 [29] 29 root7Tail31 winner =
    scan edge (search edge 3) 4 [29] 29 root7Tail32 winner := by
  rw [root7Tail31, scan, if_neg (by decide), root7Candidates31_checked]
  change scan edge (search edge 3) 4 [29] 29 root7Tail32
    (if 3 ≤ root7Candidates31.length then search edge 3 [29, 4157] root7Candidates31 4186 winner else winner) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
