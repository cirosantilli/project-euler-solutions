import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot001Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root1Candidates0 : List Nat := [97, 433, 937, 1237, 1249, 1693, 1861, 2161, 3727, 4507, 5659, 8941, 9601, 10663, 13627, 22153, 23293, 24547, 25867]
set_option maxRecDepth 8192 in
theorem root1Candidates0_checked : candidates edge 4 7 19 root1Tail1 seed.bound = root1Candidates0 := by
  unfold candidates
  have row : edge 19 = (SortedNatTree.contains FastRows.row19) := by
    change Data.row19.contains = (SortedNatTree.contains FastRows.row19)
    exact FastRows.row19_checked.symm
  rw [row]
  decide
def root1Child0Tail19 : List Nat := []
def root1Child0Tail18 : List Nat := 25867 :: root1Child0Tail19
def root1Child0Tail17 : List Nat := 24547 :: root1Child0Tail18
def root1Child0Tail16 : List Nat := 23293 :: root1Child0Tail17
def root1Child0Tail15 : List Nat := 22153 :: root1Child0Tail16
def root1Child0Tail14 : List Nat := 13627 :: root1Child0Tail15
def root1Child0Tail13 : List Nat := 10663 :: root1Child0Tail14
def root1Child0Tail12 : List Nat := 9601 :: root1Child0Tail13
def root1Child0Tail11 : List Nat := 8941 :: root1Child0Tail12
def root1Child0Tail10 : List Nat := 5659 :: root1Child0Tail11
def root1Child0Tail9 : List Nat := 4507 :: root1Child0Tail10
def root1Child0Tail8 : List Nat := 3727 :: root1Child0Tail9
def root1Child0Tail7 : List Nat := 2161 :: root1Child0Tail8
def root1Child0Tail6 : List Nat := 1861 :: root1Child0Tail7
def root1Child0Tail5 : List Nat := 1693 :: root1Child0Tail6
def root1Child0Tail4 : List Nat := 1249 :: root1Child0Tail5
def root1Child0Tail3 : List Nat := 1237 :: root1Child0Tail4
def root1Child0Tail2 : List Nat := 937 :: root1Child0Tail3
def root1Child0Tail1 : List Nat := 433 :: root1Child0Tail2
def root1Child0Tail0 : List Nat := 97 :: root1Child0Tail1
def root1Child0Candidates0 : List Nat := [3727, 4507]
set_option maxRecDepth 8192 in
theorem root1Child0Candidates0_checked : candidates edge 3 26 97 root1Child0Tail1 seed.bound = root1Child0Candidates0 := by
  unfold candidates
  have row : edge 97 = (SortedNatTree.contains FastRows.row97) := by
    change Data.row97.contains = (SortedNatTree.contains FastRows.row97)
    exact FastRows.row97_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child0Child0 : search edge 2 [7, 19, 97] root1Child0Candidates0 123 seed = seed := by decide
def root1Child0Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root1Child0Candidates1_checked : candidates edge 3 26 433 root1Child0Tail2 seed.bound = root1Child0Candidates1 := by
  unfold candidates
  have row : edge 433 = (SortedNatTree.contains FastRows.row433) := by
    change Data.row433.contains = (SortedNatTree.contains FastRows.row433)
    exact FastRows.row433_checked.symm
  rw [row]
  decide
def root1Child0Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root1Child0Candidates2_checked : candidates edge 3 26 937 root1Child0Tail3 seed.bound = root1Child0Candidates2 := by
  unfold candidates
  have row : edge 937 = (SortedNatTree.contains FastRows.row937) := by
    change Data.row937.contains = (SortedNatTree.contains FastRows.row937)
    exact FastRows.row937_checked.symm
  rw [row]
  decide
def root1Child0Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root1Child0Candidates3_checked : candidates edge 3 26 1237 root1Child0Tail4 seed.bound = root1Child0Candidates3 := by
  unfold candidates
  have row : edge 1237 = (SortedNatTree.contains FastRows.row1237) := by
    change Data.row1237.contains = (SortedNatTree.contains FastRows.row1237)
    exact FastRows.row1237_checked.symm
  rw [row]
  decide
def root1Child0Candidates4 : List Nat := [3727]
set_option maxRecDepth 8192 in
theorem root1Child0Candidates4_checked : candidates edge 3 26 1249 root1Child0Tail5 seed.bound = root1Child0Candidates4 := by
  unfold candidates
  have row : edge 1249 = (SortedNatTree.contains FastRows.row1249) := by
    change Data.row1249.contains = (SortedNatTree.contains FastRows.row1249)
    exact FastRows.row1249_checked.symm
  rw [row]
  decide
def root1Child0Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root1Child0Candidates5_checked : candidates edge 3 26 1693 root1Child0Tail6 seed.bound = root1Child0Candidates5 := by
  unfold candidates
  have row : edge 1693 = (SortedNatTree.contains FastRows.row1693) := by
    change Data.row1693.contains = (SortedNatTree.contains FastRows.row1693)
    exact FastRows.row1693_checked.symm
  rw [row]
  decide
def root1Child0Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root1Child0Candidates6_checked : candidates edge 3 26 1861 root1Child0Tail7 seed.bound = root1Child0Candidates6 := by
  unfold candidates
  have row : edge 1861 = (SortedNatTree.contains FastRows.row1861) := by
    change Data.row1861.contains = (SortedNatTree.contains FastRows.row1861)
    exact FastRows.row1861_checked.symm
  rw [row]
  decide
def root1Child0Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root1Child0Candidates7_checked : candidates edge 3 26 2161 root1Child0Tail8 seed.bound = root1Child0Candidates7 := by
  unfold candidates
  have row : edge 2161 = (SortedNatTree.contains FastRows.row2161) := by
    change Data.row2161.contains = (SortedNatTree.contains FastRows.row2161)
    exact FastRows.row2161_checked.symm
  rw [row]
  decide
def root1Child0Candidates8 : List Nat := [5659]
set_option maxRecDepth 8192 in
theorem root1Child0Candidates8_checked : candidates edge 3 26 3727 root1Child0Tail9 seed.bound = root1Child0Candidates8 := by
  unfold candidates
  have row : edge 3727 = (SortedNatTree.contains FastRows.row3727) := by
    change Data.row3727.contains = (SortedNatTree.contains FastRows.row3727)
    exact FastRows.row3727_checked.symm
  rw [row]
  decide
def root1Child0Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root1Child0Candidates9_checked : candidates edge 3 26 4507 root1Child0Tail10 seed.bound = root1Child0Candidates9 := by
  unfold candidates
  have row : edge 4507 = (SortedNatTree.contains FastRows.row4507) := by
    change Data.row4507.contains = (SortedNatTree.contains FastRows.row4507)
    exact FastRows.row4507_checked.symm
  rw [row]
  decide
def root1Child0Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root1Child0Candidates10_checked : candidates edge 3 26 5659 root1Child0Tail11 seed.bound = root1Child0Candidates10 := by
  unfold candidates
  have row : edge 5659 = (SortedNatTree.contains FastRows.row5659) := by
    change Data.row5659.contains = (SortedNatTree.contains FastRows.row5659)
    exact FastRows.row5659_checked.symm
  rw [row]
  decide
theorem root1Child0Scan11 : scan edge (search edge 2) 3 [7, 19] 26 root1Child0Tail11 seed = seed := by rw [root1Child0Tail11, scan, if_pos (by decide)]
theorem root1Child0Scan10 : scan edge (search edge 2) 3 [7, 19] 26 root1Child0Tail10 seed = seed := by
  rw [root1Child0Tail10, scan, if_neg (by decide), root1Child0Candidates10_checked]
  change scan edge (search edge 2) 3 [7, 19] 26 root1Child0Tail11
    (if 2 ≤ root1Child0Candidates10.length then search edge 2 [7, 19, 5659] root1Child0Candidates10 5685 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root1Child0Scan11
theorem root1Child0Scan9 : scan edge (search edge 2) 3 [7, 19] 26 root1Child0Tail9 seed = seed := by
  rw [root1Child0Tail9, scan, if_neg (by decide), root1Child0Candidates9_checked]
  change scan edge (search edge 2) 3 [7, 19] 26 root1Child0Tail10
    (if 2 ≤ root1Child0Candidates9.length then search edge 2 [7, 19, 4507] root1Child0Candidates9 4533 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root1Child0Scan10
theorem root1Child0Scan8 : scan edge (search edge 2) 3 [7, 19] 26 root1Child0Tail8 seed = seed := by
  rw [root1Child0Tail8, scan, if_neg (by decide), root1Child0Candidates8_checked]
  change scan edge (search edge 2) 3 [7, 19] 26 root1Child0Tail9
    (if 2 ≤ root1Child0Candidates8.length then search edge 2 [7, 19, 3727] root1Child0Candidates8 3753 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root1Child0Scan9
theorem root1Child0Scan7 : scan edge (search edge 2) 3 [7, 19] 26 root1Child0Tail7 seed = seed := by
  rw [root1Child0Tail7, scan, if_neg (by decide), root1Child0Candidates7_checked]
  change scan edge (search edge 2) 3 [7, 19] 26 root1Child0Tail8
    (if 2 ≤ root1Child0Candidates7.length then search edge 2 [7, 19, 2161] root1Child0Candidates7 2187 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root1Child0Scan8
theorem root1Child0Scan6 : scan edge (search edge 2) 3 [7, 19] 26 root1Child0Tail6 seed = seed := by
  rw [root1Child0Tail6, scan, if_neg (by decide), root1Child0Candidates6_checked]
  change scan edge (search edge 2) 3 [7, 19] 26 root1Child0Tail7
    (if 2 ≤ root1Child0Candidates6.length then search edge 2 [7, 19, 1861] root1Child0Candidates6 1887 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root1Child0Scan7
theorem root1Child0Scan5 : scan edge (search edge 2) 3 [7, 19] 26 root1Child0Tail5 seed = seed := by
  rw [root1Child0Tail5, scan, if_neg (by decide), root1Child0Candidates5_checked]
  change scan edge (search edge 2) 3 [7, 19] 26 root1Child0Tail6
    (if 2 ≤ root1Child0Candidates5.length then search edge 2 [7, 19, 1693] root1Child0Candidates5 1719 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root1Child0Scan6
theorem root1Child0Scan4 : scan edge (search edge 2) 3 [7, 19] 26 root1Child0Tail4 seed = seed := by
  rw [root1Child0Tail4, scan, if_neg (by decide), root1Child0Candidates4_checked]
  change scan edge (search edge 2) 3 [7, 19] 26 root1Child0Tail5
    (if 2 ≤ root1Child0Candidates4.length then search edge 2 [7, 19, 1249] root1Child0Candidates4 1275 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root1Child0Scan5
theorem root1Child0Scan3 : scan edge (search edge 2) 3 [7, 19] 26 root1Child0Tail3 seed = seed := by
  rw [root1Child0Tail3, scan, if_neg (by decide), root1Child0Candidates3_checked]
  change scan edge (search edge 2) 3 [7, 19] 26 root1Child0Tail4
    (if 2 ≤ root1Child0Candidates3.length then search edge 2 [7, 19, 1237] root1Child0Candidates3 1263 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root1Child0Scan4
theorem root1Child0Scan2 : scan edge (search edge 2) 3 [7, 19] 26 root1Child0Tail2 seed = seed := by
  rw [root1Child0Tail2, scan, if_neg (by decide), root1Child0Candidates2_checked]
  change scan edge (search edge 2) 3 [7, 19] 26 root1Child0Tail3
    (if 2 ≤ root1Child0Candidates2.length then search edge 2 [7, 19, 937] root1Child0Candidates2 963 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root1Child0Scan3
theorem root1Child0Scan1 : scan edge (search edge 2) 3 [7, 19] 26 root1Child0Tail1 seed = seed := by
  rw [root1Child0Tail1, scan, if_neg (by decide), root1Child0Candidates1_checked]
  change scan edge (search edge 2) 3 [7, 19] 26 root1Child0Tail2
    (if 2 ≤ root1Child0Candidates1.length then search edge 2 [7, 19, 433] root1Child0Candidates1 459 seed else seed) = seed
  rw [if_neg (by decide)]
  exact root1Child0Scan2
theorem root1Child0Scan0 : scan edge (search edge 2) 3 [7, 19] 26 root1Child0Tail0 seed = seed := by
  rw [root1Child0Tail0, scan, if_neg (by decide), root1Child0Candidates0_checked]
  change scan edge (search edge 2) 3 [7, 19] 26 root1Child0Tail1
    (if 2 ≤ root1Child0Candidates0.length then search edge 2 [7, 19, 97] root1Child0Candidates0 123 seed else seed) = seed
  rw [if_pos (by decide), root1Child0Child0]
  exact root1Child0Scan1
theorem root1Child0 : search edge 3 [7, 19] root1Candidates0 26 seed = seed := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 2) 3 [7, 19] 26 root1Child0Tail0 seed = seed
  exact root1Child0Scan0
theorem root1Step0 : scan edge (search edge 3) 4 [7] 7 root1Tail0 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail1 seed := by
  rw [root1Tail0, scan, if_neg (by decide), root1Candidates0_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail1
    (if 3 ≤ root1Candidates0.length then search edge 3 [7, 19] root1Candidates0 26 seed else seed) = _
  rw [if_pos (by decide), root1Child0]
def root1Candidates1 : List Nat := [487, 1453, 1693, 1999, 2341, 3181, 5821, 8311, 8941, 11617, 13309, 15643, 15973, 25939]
set_option maxRecDepth 8192 in
theorem root1Candidates1_checked : candidates edge 4 7 61 root1Tail2 seed.bound = root1Candidates1 := by
  unfold candidates
  have row : edge 61 = (SortedNatTree.contains FastRows.row61) := by
    change Data.row61.contains = (SortedNatTree.contains FastRows.row61)
    exact FastRows.row61_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child1 : search edge 3 [7, 61] root1Candidates1 68 seed = seed := by decide
theorem root1Step1 : scan edge (search edge 3) 4 [7] 7 root1Tail1 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail2 seed := by
  rw [root1Tail1, scan, if_neg (by decide), root1Candidates1_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail2
    (if 3 ≤ root1Candidates1.length then search edge 3 [7, 61] root1Candidates1 68 seed else seed) = _
  rw [if_pos (by decide), root1Child1]
def root1Candidates2 : List Nat := [829, 883, 3331, 3727, 4507, 9337, 10513, 10627, 13309, 17989]
set_option maxRecDepth 8192 in
theorem root1Candidates2_checked : candidates edge 4 7 97 root1Tail3 seed.bound = root1Candidates2 := by
  unfold candidates
  have row : edge 97 = (SortedNatTree.contains FastRows.row97) := by
    change Data.row97.contains = (SortedNatTree.contains FastRows.row97)
    exact FastRows.row97_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child2 : search edge 3 [7, 97] root1Candidates2 104 seed = seed := by decide
theorem root1Step2 : scan edge (search edge 3) 4 [7] 7 root1Tail2 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail3 seed := by
  rw [root1Tail2, scan, if_neg (by decide), root1Candidates2_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail3
    (if 3 ≤ root1Candidates2.length then search edge 3 [7, 97] root1Candidates2 104 seed else seed) = _
  rw [if_pos (by decide), root1Child2]
def root1Candidates3 : List Nat := [673, 883, 1399, 3331, 4441, 4567, 7351, 7933, 9337, 13597, 13807, 16063, 16981, 18433, 20887, 24733]
set_option maxRecDepth 8192 in
theorem root1Candidates3_checked : candidates edge 4 7 109 root1Tail4 seed.bound = root1Candidates3 := by
  unfold candidates
  have row : edge 109 = (SortedNatTree.contains FastRows.row109) := by
    change Data.row109.contains = (SortedNatTree.contains FastRows.row109)
    exact FastRows.row109_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child3 : search edge 3 [7, 109] root1Candidates3 116 seed = seed := by decide
theorem root1Step3 : scan edge (search edge 3) 4 [7] 7 root1Tail3 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail4 seed := by
  rw [root1Tail3, scan, if_neg (by decide), root1Candidates3_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail4
    (if 3 ≤ root1Candidates3.length then search edge 3 [7, 109] root1Candidates3 116 seed else seed) = _
  rw [if_pos (by decide), root1Child3]
def root1Candidates4 : List Nat := [6481, 6949, 7723, 9043, 9337, 11287, 11701, 12409, 14851, 16411, 16981, 17491, 22921]
set_option maxRecDepth 8192 in
theorem root1Candidates4_checked : candidates edge 4 7 127 root1Tail5 seed.bound = root1Candidates4 := by
  unfold candidates
  have row : edge 127 = (SortedNatTree.contains FastRows.row127) := by
    change Data.row127.contains = (SortedNatTree.contains FastRows.row127)
    exact FastRows.row127_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child4 : search edge 3 [7, 127] root1Candidates4 134 seed = seed := by decide
theorem root1Step4 : scan edge (search edge 3) 4 [7] 7 root1Tail4 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail5 seed := by
  rw [root1Tail4, scan, if_neg (by decide), root1Candidates4_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail5
    (if 3 ≤ root1Candidates4.length then search edge 3 [7, 127] root1Candidates4 134 seed else seed) = _
  rw [if_pos (by decide), root1Child4]
def root1Candidates5 : List Nat := [433, 547, 937, 5659, 12511, 15289, 15859, 18433, 20983, 22639]
set_option maxRecDepth 8192 in
theorem root1Candidates5_checked : candidates edge 4 7 229 root1Tail6 seed.bound = root1Candidates5 := by
  unfold candidates
  have row : edge 229 = (SortedNatTree.contains FastRows.row229) := by
    change Data.row229.contains = (SortedNatTree.contains FastRows.row229)
    exact FastRows.row229_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child5 : search edge 3 [7, 229] root1Candidates5 236 seed = seed := by decide
theorem root1Step5 : scan edge (search edge 3) 4 [7] 7 root1Tail5 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail6 seed := by
  rw [root1Tail5, scan, if_neg (by decide), root1Candidates5_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail6
    (if 3 ≤ root1Candidates5.length then search edge 3 [7, 229] root1Candidates5 236 seed else seed) = _
  rw [if_pos (by decide), root1Child5]
def root1Candidates6 : List Nat := [487, 541, 1237, 2287, 2503, 9349, 10837, 15289, 16087, 20899, 22639, 24847]
set_option maxRecDepth 8192 in
theorem root1Candidates6_checked : candidates edge 4 7 283 root1Tail7 seed.bound = root1Candidates6 := by
  unfold candidates
  have row : edge 283 = (SortedNatTree.contains FastRows.row283) := by
    change Data.row283.contains = (SortedNatTree.contains FastRows.row283)
    exact FastRows.row283_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child6 : search edge 3 [7, 283] root1Candidates6 290 seed = seed := by decide
theorem root1Step6 : scan edge (search edge 3) 4 [7] 7 root1Tail6 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail7 seed := by
  rw [root1Tail6, scan, if_neg (by decide), root1Candidates6_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail7
    (if 3 ≤ root1Candidates6.length then search edge 3 [7, 283] root1Candidates6 290 seed else seed) = _
  rw [if_pos (by decide), root1Child6]
def root1Candidates7 : List Nat := [883, 1471, 2719, 3613, 4441, 4903, 6091, 8167, 9613, 9901, 13177, 17419, 18433, 19681, 21739]
set_option maxRecDepth 8192 in
theorem root1Candidates7_checked : candidates edge 4 7 433 root1Tail8 seed.bound = root1Candidates7 := by
  unfold candidates
  have row : edge 433 = (SortedNatTree.contains FastRows.row433) := by
    change Data.row433.contains = (SortedNatTree.contains FastRows.row433)
    exact FastRows.row433_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child7 : search edge 3 [7, 433] root1Candidates7 440 seed = seed := by decide
theorem root1Step7 : scan edge (search edge 3) 4 [7] 7 root1Tail7 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail8 seed := by
  rw [root1Tail7, scan, if_neg (by decide), root1Candidates7_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail8
    (if 3 ≤ root1Candidates7.length then search edge 3 [7, 433] root1Candidates7 440 seed else seed) = _
  rw [if_pos (by decide), root1Child7]
def root1Candidates8 : List Nat := [757, 3433, 8893, 8941, 11287, 15193, 21397, 21661]
set_option maxRecDepth 8192 in
theorem root1Candidates8_checked : candidates edge 4 7 487 root1Tail9 seed.bound = root1Candidates8 := by
  unfold candidates
  have row : edge 487 = (SortedNatTree.contains FastRows.row487) := by
    change Data.row487.contains = (SortedNatTree.contains FastRows.row487)
    exact FastRows.row487_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child8 : search edge 3 [7, 487] root1Candidates8 494 seed = seed := by decide
theorem root1Step8 : scan edge (search edge 3) 4 [7] 7 root1Tail8 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail9 seed := by
  rw [root1Tail8, scan, if_neg (by decide), root1Candidates8_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail9
    (if 3 ≤ root1Candidates8.length then search edge 3 [7, 487] root1Candidates8 494 seed else seed) = _
  rw [if_pos (by decide), root1Child8]
def root1Candidates9 : List Nat := [541, 2089, 2671, 10459, 10837, 13309, 25111]
set_option maxRecDepth 8192 in
theorem root1Candidates9_checked : candidates edge 4 7 523 root1Tail10 seed.bound = root1Candidates9 := by
  unfold candidates
  have row : edge 523 = (SortedNatTree.contains FastRows.row523) := by
    change Data.row523.contains = (SortedNatTree.contains FastRows.row523)
    exact FastRows.row523_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child9 : search edge 3 [7, 523] root1Candidates9 530 seed = seed := by decide
theorem root1Step9 : scan edge (search edge 3) 4 [7] 7 root1Tail9 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail10 seed := by
  rw [root1Tail9, scan, if_neg (by decide), root1Candidates9_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail10
    (if 3 ≤ root1Candidates9.length then search edge 3 [7, 523] root1Candidates9 530 seed else seed) = _
  rw [if_pos (by decide), root1Child9]
def root1Candidates10 : List Nat := [2467, 4159, 8167, 12433, 15643, 17419, 19267, 19801, 20887]
set_option maxRecDepth 8192 in
theorem root1Candidates10_checked : candidates edge 4 7 541 root1Tail11 seed.bound = root1Candidates10 := by
  unfold candidates
  have row : edge 541 = (SortedNatTree.contains FastRows.row541) := by
    change Data.row541.contains = (SortedNatTree.contains FastRows.row541)
    exact FastRows.row541_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child10 : search edge 3 [7, 541] root1Candidates10 548 seed = seed := by decide
theorem root1Step10 : scan edge (search edge 3) 4 [7] 7 root1Tail10 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail11 seed := by
  rw [root1Tail10, scan, if_neg (by decide), root1Candidates10_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail11
    (if 3 ≤ root1Candidates10.length then search edge 3 [7, 541] root1Candidates10 548 seed else seed) = _
  rw [if_pos (by decide), root1Child10]
def root1Candidates11 : List Nat := [823, 853, 1237, 8311, 10321, 13627, 14563, 18457, 19597, 20947, 22669]
set_option maxRecDepth 8192 in
theorem root1Candidates11_checked : candidates edge 4 7 547 root1Tail12 seed.bound = root1Candidates11 := by
  unfold candidates
  have row : edge 547 = (SortedNatTree.contains FastRows.row547) := by
    change Data.row547.contains = (SortedNatTree.contains FastRows.row547)
    exact FastRows.row547_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child11 : search edge 3 [7, 547] root1Candidates11 554 seed = seed := by decide
theorem root1Step11 : scan edge (search edge 3) 4 [7] 7 root1Tail11 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail12 seed := by
  rw [root1Tail11, scan, if_neg (by decide), root1Candidates11_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail12
    (if 3 ≤ root1Candidates11.length then search edge 3 [7, 547] root1Candidates11 554 seed else seed) = _
  rw [if_pos (by decide), root1Child11]
def root1Candidates12 : List Nat := [7417, 11701, 13177, 14563, 17191, 19009, 20887, 22921]
set_option maxRecDepth 8192 in
theorem root1Candidates12_checked : candidates edge 4 7 673 root1Tail13 seed.bound = root1Candidates12 := by
  unfold candidates
  have row : edge 673 = (SortedNatTree.contains FastRows.row673) := by
    change Data.row673.contains = (SortedNatTree.contains FastRows.row673)
    exact FastRows.row673_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child12 : search edge 3 [7, 673] root1Candidates12 680 seed = seed := by decide
theorem root1Step12 : scan edge (search edge 3) 4 [7] 7 root1Tail12 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail13 seed := by
  rw [root1Tail12, scan, if_neg (by decide), root1Candidates12_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail13
    (if 3 ≤ root1Candidates12.length then search edge 3 [7, 673] root1Candidates12 680 seed else seed) = _
  rw [if_pos (by decide), root1Child12]
def root1Candidates13 : List Nat := [1249, 1693, 3433, 4729, 9043, 15877, 15973, 21991, 23167, 23293]
set_option maxRecDepth 8192 in
theorem root1Candidates13_checked : candidates edge 4 7 691 root1Tail14 seed.bound = root1Candidates13 := by
  unfold candidates
  have row : edge 691 = (SortedNatTree.contains FastRows.row691) := by
    change Data.row691.contains = (SortedNatTree.contains FastRows.row691)
    exact FastRows.row691_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child13 : search edge 3 [7, 691] root1Candidates13 698 seed = seed := by decide
theorem root1Step13 : scan edge (search edge 3) 4 [7] 7 root1Tail13 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail14 seed := by
  rw [root1Tail13, scan, if_neg (by decide), root1Candidates13_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail14
    (if 3 ≤ root1Candidates13.length then search edge 3 [7, 691] root1Candidates13 698 seed else seed) = _
  rw [if_pos (by decide), root1Child13]
def root1Candidates14 : List Nat := [829, 11437, 19183, 21991, 22921, 22963]
set_option maxRecDepth 8192 in
theorem root1Candidates14_checked : candidates edge 4 7 757 root1Tail15 seed.bound = root1Candidates14 := by
  unfold candidates
  have row : edge 757 = (SortedNatTree.contains FastRows.row757) := by
    change Data.row757.contains = (SortedNatTree.contains FastRows.row757)
    exact FastRows.row757_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child14 : search edge 3 [7, 757] root1Candidates14 764 seed = seed := by decide
theorem root1Step14 : scan edge (search edge 3) 4 [7] 7 root1Tail14 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail15 seed := by
  rw [root1Tail14, scan, if_neg (by decide), root1Candidates14_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail15
    (if 3 ≤ root1Candidates14.length then search edge 3 [7, 757] root1Candidates14 764 seed else seed) = _
  rw [if_pos (by decide), root1Child14]
def root1Candidates15 : List Nat := [2953, 8101, 10663, 10909, 17659, 24547, 25147]
set_option maxRecDepth 8192 in
theorem root1Candidates15_checked : candidates edge 4 7 823 root1Tail16 seed.bound = root1Candidates15 := by
  unfold candidates
  have row : edge 823 = (SortedNatTree.contains FastRows.row823) := by
    change Data.row823.contains = (SortedNatTree.contains FastRows.row823)
    exact FastRows.row823_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child15 : search edge 3 [7, 823] root1Candidates15 830 seed = seed := by decide
theorem root1Step15 : scan edge (search edge 3) 4 [7] 7 root1Tail15 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail16 seed := by
  rw [root1Tail15, scan, if_neg (by decide), root1Candidates15_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail16
    (if 3 ≤ root1Candidates15.length then search edge 3 [7, 823] root1Candidates15 830 seed else seed) = _
  rw [if_pos (by decide), root1Child15]
def root1Candidates16 : List Nat := [1237, 2671, 3361, 7351, 8803, 10903, 11299, 13477, 15643, 20857, 24631]
set_option maxRecDepth 8192 in
theorem root1Candidates16_checked : candidates edge 4 7 829 root1Tail17 seed.bound = root1Candidates16 := by
  unfold candidates
  have row : edge 829 = (SortedNatTree.contains FastRows.row829) := by
    change Data.row829.contains = (SortedNatTree.contains FastRows.row829)
    exact FastRows.row829_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child16 : search edge 3 [7, 829] root1Candidates16 836 seed = seed := by decide
theorem root1Step16 : scan edge (search edge 3) 4 [7] 7 root1Tail16 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail17 seed := by
  rw [root1Tail16, scan, if_neg (by decide), root1Candidates16_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail17
    (if 3 ≤ root1Candidates16.length then search edge 3 [7, 829] root1Candidates16 836 seed else seed) = _
  rw [if_pos (by decide), root1Child16]
def root1Candidates17 : List Nat := [2269, 2707, 8017, 8779, 10909, 12973, 13309, 14557, 18049, 21661]
set_option maxRecDepth 8192 in
theorem root1Candidates17_checked : candidates edge 4 7 853 root1Tail18 seed.bound = root1Candidates17 := by
  unfold candidates
  have row : edge 853 = (SortedNatTree.contains FastRows.row853) := by
    change Data.row853.contains = (SortedNatTree.contains FastRows.row853)
    exact FastRows.row853_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child17 : search edge 3 [7, 853] root1Candidates17 860 seed = seed := by decide
theorem root1Step17 : scan edge (search edge 3) 4 [7] 7 root1Tail17 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail18 seed := by
  rw [root1Tail17, scan, if_neg (by decide), root1Candidates17_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail18
    (if 3 ≤ root1Candidates17.length then search edge 3 [7, 853] root1Candidates17 860 seed else seed) = _
  rw [if_pos (by decide), root1Child17]
def root1Candidates18 : List Nat := [9043, 9601, 13807, 14797, 15973, 18049, 19801]
set_option maxRecDepth 8192 in
theorem root1Candidates18_checked : candidates edge 4 7 883 root1Tail19 seed.bound = root1Candidates18 := by
  unfold candidates
  have row : edge 883 = (SortedNatTree.contains FastRows.row883) := by
    change Data.row883.contains = (SortedNatTree.contains FastRows.row883)
    exact FastRows.row883_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child18 : search edge 3 [7, 883] root1Candidates18 890 seed = seed := by decide
theorem root1Step18 : scan edge (search edge 3) 4 [7] 7 root1Tail18 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail19 seed := by
  rw [root1Tail18, scan, if_neg (by decide), root1Candidates18_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail19
    (if 3 ≤ root1Candidates18.length then search edge 3 [7, 883] root1Candidates18 890 seed else seed) = _
  rw [if_pos (by decide), root1Child18]
def root1Candidates19 : List Nat := [4159, 5437, 10321, 12973, 13177, 19183, 22807]
set_option maxRecDepth 8192 in
theorem root1Candidates19_checked : candidates edge 4 7 937 root1Tail20 seed.bound = root1Candidates19 := by
  unfold candidates
  have row : edge 937 = (SortedNatTree.contains FastRows.row937) := by
    change Data.row937.contains = (SortedNatTree.contains FastRows.row937)
    exact FastRows.row937_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child19 : search edge 3 [7, 937] root1Candidates19 944 seed = seed := by decide
theorem root1Step19 : scan edge (search edge 3) 4 [7] 7 root1Tail19 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail20 seed := by
  rw [root1Tail19, scan, if_neg (by decide), root1Candidates19_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail20
    (if 3 ≤ root1Candidates19.length then search edge 3 [7, 937] root1Candidates19 944 seed else seed) = _
  rw [if_pos (by decide), root1Child19]
def root1Candidates20 : List Nat := [2161, 10459, 12301]
set_option maxRecDepth 8192 in
theorem root1Candidates20_checked : candidates edge 4 7 1171 root1Tail21 seed.bound = root1Candidates20 := by
  unfold candidates
  have row : edge 1171 = (SortedNatTree.contains FastRows.row1171) := by
    change Data.row1171.contains = (SortedNatTree.contains FastRows.row1171)
    exact FastRows.row1171_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child20 : search edge 3 [7, 1171] root1Candidates20 1178 seed = seed := by decide
theorem root1Step20 : scan edge (search edge 3) 4 [7] 7 root1Tail20 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail21 seed := by
  rw [root1Tail20, scan, if_neg (by decide), root1Candidates20_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail21
    (if 3 ≤ root1Candidates20.length then search edge 3 [7, 1171] root1Candidates20 1178 seed else seed) = _
  rw [if_pos (by decide), root1Child20]
def root1Candidates21 : List Nat := [1549, 1741, 2341, 3019, 4027, 12409, 15877, 16981, 18433, 19801, 22639]
set_option maxRecDepth 8192 in
theorem root1Candidates21_checked : candidates edge 4 7 1237 root1Tail22 seed.bound = root1Candidates21 := by
  unfold candidates
  have row : edge 1237 = (SortedNatTree.contains FastRows.row1237) := by
    change Data.row1237.contains = (SortedNatTree.contains FastRows.row1237)
    exact FastRows.row1237_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child21 : search edge 3 [7, 1237] root1Candidates21 1244 seed = seed := by decide
theorem root1Step21 : scan edge (search edge 3) 4 [7] 7 root1Tail21 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail22 seed := by
  rw [root1Tail21, scan, if_neg (by decide), root1Candidates21_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail22
    (if 3 ≤ root1Candidates21.length then search edge 3 [7, 1237] root1Candidates21 1244 seed else seed) = _
  rw [if_pos (by decide), root1Child21]
def root1Candidates22 : List Nat := [3727, 4441, 6949, 8803, 12157, 20899, 21283, 22921]
set_option maxRecDepth 8192 in
theorem root1Candidates22_checked : candidates edge 4 7 1249 root1Tail23 seed.bound = root1Candidates22 := by
  unfold candidates
  have row : edge 1249 = (SortedNatTree.contains FastRows.row1249) := by
    change Data.row1249.contains = (SortedNatTree.contains FastRows.row1249)
    exact FastRows.row1249_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child22 : search edge 3 [7, 1249] root1Candidates22 1256 seed = seed := by decide
theorem root1Step22 : scan edge (search edge 3) 4 [7] 7 root1Tail22 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail23 seed := by
  rw [root1Tail22, scan, if_neg (by decide), root1Candidates22_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail23
    (if 3 ≤ root1Candidates22.length then search edge 3 [7, 1249] root1Candidates22 1256 seed else seed) = _
  rw [if_pos (by decide), root1Child22]
def root1Candidates23 : List Nat := [1741, 3433, 6481, 15289, 17317, 23167]
set_option maxRecDepth 8192 in
theorem root1Candidates23_checked : candidates edge 4 7 1399 root1Tail24 seed.bound = root1Candidates23 := by
  unfold candidates
  have row : edge 1399 = (SortedNatTree.contains FastRows.row1399) := by
    change Data.row1399.contains = (SortedNatTree.contains FastRows.row1399)
    exact FastRows.row1399_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child23 : search edge 3 [7, 1399] root1Candidates23 1406 seed = seed := by decide
theorem root1Step23 : scan edge (search edge 3) 4 [7] 7 root1Tail23 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail24 seed := by
  rw [root1Tail23, scan, if_neg (by decide), root1Candidates23_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail24
    (if 3 ≤ root1Candidates23.length then search edge 3 [7, 1399] root1Candidates23 1406 seed else seed) = _
  rw [if_pos (by decide), root1Child23]
def root1Candidates24 : List Nat := [1879, 3361, 7417, 7933, 14551, 20023, 24547]
set_option maxRecDepth 8192 in
theorem root1Candidates24_checked : candidates edge 4 7 1453 root1Tail25 seed.bound = root1Candidates24 := by
  unfold candidates
  have row : edge 1453 = (SortedNatTree.contains FastRows.row1453) := by
    change Data.row1453.contains = (SortedNatTree.contains FastRows.row1453)
    exact FastRows.row1453_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child24 : search edge 3 [7, 1453] root1Candidates24 1460 seed = seed := by decide
theorem root1Step24 : scan edge (search edge 3) 4 [7] 7 root1Tail24 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail25 seed := by
  rw [root1Tail24, scan, if_neg (by decide), root1Candidates24_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail25
    (if 3 ≤ root1Candidates24.length then search edge 3 [7, 1453] root1Candidates24 1460 seed else seed) = _
  rw [if_pos (by decide), root1Child24]
def root1Candidates25 : List Nat := [3613, 4159, 7933, 8779, 10399, 10867, 21529, 24517]
set_option maxRecDepth 8192 in
theorem root1Candidates25_checked : candidates edge 4 7 1471 root1Tail26 seed.bound = root1Candidates25 := by
  unfold candidates
  have row : edge 1471 = (SortedNatTree.contains FastRows.row1471) := by
    change Data.row1471.contains = (SortedNatTree.contains FastRows.row1471)
    exact FastRows.row1471_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child25 : search edge 3 [7, 1471] root1Candidates25 1478 seed = seed := by decide
theorem root1Step25 : scan edge (search edge 3) 4 [7] 7 root1Tail25 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail26 seed := by
  rw [root1Tail25, scan, if_neg (by decide), root1Candidates25_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail26
    (if 3 ≤ root1Candidates25.length then search edge 3 [7, 1471] root1Candidates25 1478 seed else seed) = _
  rw [if_pos (by decide), root1Child25]
def root1Candidates26 : List Nat := [2707, 3019, 3823, 9829, 15289, 15727]
set_option maxRecDepth 8192 in
theorem root1Candidates26_checked : candidates edge 4 7 1549 root1Tail27 seed.bound = root1Candidates26 := by
  unfold candidates
  have row : edge 1549 = (SortedNatTree.contains FastRows.row1549) := by
    change Data.row1549.contains = (SortedNatTree.contains FastRows.row1549)
    exact FastRows.row1549_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child26 : search edge 3 [7, 1549] root1Candidates26 1556 seed = seed := by decide
theorem root1Step26 : scan edge (search edge 3) 4 [7] 7 root1Tail26 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail27 seed := by
  rw [root1Tail26, scan, if_neg (by decide), root1Candidates26_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail27
    (if 3 ≤ root1Candidates26.length then search edge 3 [7, 1549] root1Candidates26 1556 seed else seed) = _
  rw [if_pos (by decide), root1Child26]
def root1Candidates27 : List Nat := [3181, 4441, 7489, 9337, 10459, 14797, 16981, 22147, 22921]
set_option maxRecDepth 8192 in
theorem root1Candidates27_checked : candidates edge 4 7 1693 root1Tail28 seed.bound = root1Candidates27 := by
  unfold candidates
  have row : edge 1693 = (SortedNatTree.contains FastRows.row1693) := by
    change Data.row1693.contains = (SortedNatTree.contains FastRows.row1693)
    exact FastRows.row1693_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child27 : search edge 3 [7, 1693] root1Candidates27 1700 seed = seed := by decide
theorem root1Step27 : scan edge (search edge 3) 4 [7] 7 root1Tail27 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail28 seed := by
  rw [root1Tail27, scan, if_neg (by decide), root1Candidates27_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail28
    (if 3 ≤ root1Candidates27.length then search edge 3 [7, 1693] root1Candidates27 1700 seed else seed) = _
  rw [if_pos (by decide), root1Child27]
def root1Candidates28 : List Nat := [2287, 2953, 7489, 8893, 10837, 12973, 13309, 14029]
set_option maxRecDepth 8192 in
theorem root1Candidates28_checked : candidates edge 4 7 1741 root1Tail29 seed.bound = root1Candidates28 := by
  unfold candidates
  have row : edge 1741 = (SortedNatTree.contains FastRows.row1741) := by
    change Data.row1741.contains = (SortedNatTree.contains FastRows.row1741)
    exact FastRows.row1741_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child28 : search edge 3 [7, 1741] root1Candidates28 1748 seed = seed := by decide
theorem root1Step28 : scan edge (search edge 3) 4 [7] 7 root1Tail28 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail29 seed := by
  rw [root1Tail28, scan, if_neg (by decide), root1Candidates28_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail29
    (if 3 ≤ root1Candidates28.length then search edge 3 [7, 1741] root1Candidates28 1748 seed else seed) = _
  rw [if_pos (by decide), root1Child28]
def root1Candidates29 : List Nat := [3433, 3709, 5521, 10627, 11437, 18043]
set_option maxRecDepth 8192 in
theorem root1Candidates29_checked : candidates edge 4 7 1861 root1Tail30 seed.bound = root1Candidates29 := by
  unfold candidates
  have row : edge 1861 = (SortedNatTree.contains FastRows.row1861) := by
    change Data.row1861.contains = (SortedNatTree.contains FastRows.row1861)
    exact FastRows.row1861_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child29 : search edge 3 [7, 1861] root1Candidates29 1868 seed = seed := by decide
theorem root1Step29 : scan edge (search edge 3) 4 [7] 7 root1Tail29 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail30 seed := by
  rw [root1Tail29, scan, if_neg (by decide), root1Candidates29_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail30
    (if 3 ≤ root1Candidates29.length then search edge 3 [7, 1861] root1Candidates29 1868 seed else seed) = _
  rw [if_pos (by decide), root1Child29]
def root1Candidates30 : List Nat := [1999, 2341, 3331, 12301, 12409, 13183, 19183, 19267]
set_option maxRecDepth 8192 in
theorem root1Candidates30_checked : candidates edge 4 7 1879 root1Tail31 seed.bound = root1Candidates30 := by
  unfold candidates
  have row : edge 1879 = (SortedNatTree.contains FastRows.row1879) := by
    change Data.row1879.contains = (SortedNatTree.contains FastRows.row1879)
    exact FastRows.row1879_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child30 : search edge 3 [7, 1879] root1Candidates30 1886 seed = seed := by decide
theorem root1Step30 : scan edge (search edge 3) 4 [7] 7 root1Tail30 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail31 seed := by
  rw [root1Tail30, scan, if_neg (by decide), root1Candidates30_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail31
    (if 3 ≤ root1Candidates30.length then search edge 3 [7, 1879] root1Candidates30 1886 seed else seed) = _
  rw [if_pos (by decide), root1Child30]
def root1Candidates31 : List Nat := [6481, 16063, 17491, 22147, 22639]
set_option maxRecDepth 8192 in
theorem root1Candidates31_checked : candidates edge 4 7 1993 root1Tail32 seed.bound = root1Candidates31 := by
  unfold candidates
  have row : edge 1993 = (SortedNatTree.contains FastRows.row1993) := by
    change Data.row1993.contains = (SortedNatTree.contains FastRows.row1993)
    exact FastRows.row1993_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root1Child31 : search edge 3 [7, 1993] root1Candidates31 2000 seed = seed := by decide
theorem root1Step31 : scan edge (search edge 3) 4 [7] 7 root1Tail31 seed =
    scan edge (search edge 3) 4 [7] 7 root1Tail32 seed := by
  rw [root1Tail31, scan, if_neg (by decide), root1Candidates31_checked]
  change scan edge (search edge 3) 4 [7] 7 root1Tail32
    (if 3 ≤ root1Candidates31.length then search edge 3 [7, 1993] root1Candidates31 2000 seed else seed) = _
  rw [if_pos (by decide), root1Child31]
end ProjectEulerSolutions.PrimePairLarge.Search
