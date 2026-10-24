import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root92Candidates : List Nat := Data.row499
set_option maxRecDepth 8192 in
theorem root92Candidates_checked : candidates edge 5 0 499 tail93 winner.bound = root92Candidates := by
  apply candidates_eq_row edge 5 0 499 tail93 root92Candidates winner.bound tail93_ordered
  · decide
  · decide
  · funext q
    rfl
def root92Tail68 : List Nat := []
def root92Tail67 : List Nat := 25453 :: root92Tail68
def root92Tail66 : List Nat := 25339 :: root92Tail67
def root92Tail65 : List Nat := 25219 :: root92Tail66
def root92Tail64 : List Nat := 25189 :: root92Tail65
def root92Tail63 : List Nat := 25087 :: root92Tail64
def root92Tail62 : List Nat := 24841 :: root92Tail63
def root92Tail61 : List Nat := 24709 :: root92Tail62
def root92Tail60 : List Nat := 24373 :: root92Tail61
def root92Tail59 : List Nat := 24181 :: root92Tail60
def root92Tail58 : List Nat := 23689 :: root92Tail59
def root92Tail57 : List Nat := 23563 :: root92Tail58
def root92Tail56 : List Nat := 23371 :: root92Tail57
def root92Tail55 : List Nat := 23059 :: root92Tail56
def root92Tail54 : List Nat := 22123 :: root92Tail55
def root92Tail53 : List Nat := 21787 :: root92Tail54
def root92Tail52 : List Nat := 21517 :: root92Tail53
def root92Tail51 : List Nat := 20947 :: root92Tail52
def root92Tail50 : List Nat := 19801 :: root92Tail51
def root92Tail49 : List Nat := 18457 :: root92Tail50
def root92Tail48 : List Nat := 18397 :: root92Tail49
def root92Tail47 : List Nat := 17851 :: root92Tail48
def root92Tail46 : List Nat := 17449 :: root92Tail47
def root92Tail45 : List Nat := 17293 :: root92Tail46
def root92Tail44 : List Nat := 16987 :: root92Tail45
def root92Tail43 : List Nat := 16843 :: root92Tail44
def root92Tail42 : List Nat := 16189 :: root92Tail43
def root92Tail41 : List Nat := 15091 :: root92Tail42
def root92Tail40 : List Nat := 14983 :: root92Tail41
def root92Tail39 : List Nat := 14827 :: root92Tail40
def root92Tail38 : List Nat := 14629 :: root92Tail39
def root92Tail37 : List Nat := 14281 :: root92Tail38
def root92Tail36 : List Nat := 13711 :: root92Tail37
def root92Tail35 : List Nat := 12547 :: root92Tail36
def root92Tail34 : List Nat := 11971 :: root92Tail35
def root92Tail33 : List Nat := 11593 :: root92Tail34
def root92Tail32 : List Nat := 11353 :: root92Tail33
def root92Tail31 : List Nat := 11317 :: root92Tail32
def root92Tail30 : List Nat := 11299 :: root92Tail31
def root92Tail29 : List Nat := 10177 :: root92Tail30
def root92Tail28 : List Nat := 9187 :: root92Tail29
def root92Tail27 : List Nat := 9103 :: root92Tail28
def root92Tail26 : List Nat := 8971 :: root92Tail27
def root92Tail25 : List Nat := 8467 :: root92Tail26
def root92Tail24 : List Nat := 7717 :: root92Tail25
def root92Tail23 : List Nat := 7687 :: root92Tail24
def root92Tail22 : List Nat := 7393 :: root92Tail23
def root92Tail21 : List Nat := 6361 :: root92Tail22
def root92Tail20 : List Nat := 5827 :: root92Tail21
def root92Tail19 : List Nat := 5503 :: root92Tail20
def root92Tail18 : List Nat := 5281 :: root92Tail19
def root92Tail17 : List Nat := 4621 :: root92Tail18
def root92Tail16 : List Nat := 4513 :: root92Tail17
def root92Tail15 : List Nat := 4357 :: root92Tail16
def root92Tail14 : List Nat := 4051 :: root92Tail15
def root92Tail13 : List Nat := 3907 :: root92Tail14
def root92Tail12 : List Nat := 3847 :: root92Tail13
def root92Tail11 : List Nat := 3343 :: root92Tail12
def root92Tail10 : List Nat := 2707 :: root92Tail11
def root92Tail9 : List Nat := 2551 :: root92Tail10
def root92Tail8 : List Nat := 1933 :: root92Tail9
def root92Tail7 : List Nat := 1453 :: root92Tail8
def root92Tail6 : List Nat := 1303 :: root92Tail7
def root92Tail5 : List Nat := 1249 :: root92Tail6
def root92Tail4 : List Nat := 1237 :: root92Tail5
def root92Tail3 : List Nat := 1033 :: root92Tail4
def root92Tail2 : List Nat := 1009 :: root92Tail3
def root92Tail1 : List Nat := 691 :: root92Tail2
def root92Tail0 : List Nat := 673 :: root92Tail1
def root92Candidates0 : List Nat := [14983]
set_option maxRecDepth 8192 in
theorem root92Candidates0_checked : candidates edge 4 499 673 root92Tail1 winner.bound = root92Candidates0 := by
  unfold candidates
  have row : edge 673 = (SortedNatTree.contains FastRows.row673) := by
    change Data.row673.contains = (SortedNatTree.contains FastRows.row673)
    exact FastRows.row673_checked.symm
  rw [row]
  decide
theorem root92Step0 : scan edge (search edge 3) 4 [499] 499 root92Tail0 winner =
    scan edge (search edge 3) 4 [499] 499 root92Tail1 winner := by
  rw [root92Tail0, scan, if_neg (by decide), root92Candidates0_checked]
  change scan edge (search edge 3) 4 [499] 499 root92Tail1
    (if 3 ≤ root92Candidates0.length then search edge 3 [499, 673] root92Candidates0 1172 winner else winner) = _
  rw [if_neg (by decide)]
def root92Candidates1 : List Nat := [1249, 5827, 24709]
set_option maxRecDepth 8192 in
theorem root92Candidates1_checked : candidates edge 4 499 691 root92Tail2 winner.bound = root92Candidates1 := by
  unfold candidates
  have row : edge 691 = (SortedNatTree.contains FastRows.row691) := by
    change Data.row691.contains = (SortedNatTree.contains FastRows.row691)
    exact FastRows.row691_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root92Child1 : search edge 3 [499, 691] root92Candidates1 1190 winner = winner := by decide
theorem root92Step1 : scan edge (search edge 3) 4 [499] 499 root92Tail1 winner =
    scan edge (search edge 3) 4 [499] 499 root92Tail2 winner := by
  rw [root92Tail1, scan, if_neg (by decide), root92Candidates1_checked]
  change scan edge (search edge 3) 4 [499] 499 root92Tail2
    (if 3 ≤ root92Candidates1.length then search edge 3 [499, 691] root92Candidates1 1190 winner else winner) = _
  rw [if_pos (by decide), root92Child1]
def root92Candidates2 : List Nat := [1933, 7393, 16843, 21517]
set_option maxRecDepth 8192 in
theorem root92Candidates2_checked : candidates edge 4 499 1009 root92Tail3 winner.bound = root92Candidates2 := by
  unfold candidates
  have row : edge 1009 = (SortedNatTree.contains FastRows.row1009) := by
    change Data.row1009.contains = (SortedNatTree.contains FastRows.row1009)
    exact FastRows.row1009_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root92Child2 : search edge 3 [499, 1009] root92Candidates2 1508 winner = winner := by decide
theorem root92Step2 : scan edge (search edge 3) 4 [499] 499 root92Tail2 winner =
    scan edge (search edge 3) 4 [499] 499 root92Tail3 winner := by
  rw [root92Tail2, scan, if_neg (by decide), root92Candidates2_checked]
  change scan edge (search edge 3) 4 [499] 499 root92Tail3
    (if 3 ≤ root92Candidates2.length then search edge 3 [499, 1009] root92Candidates2 1508 winner else winner) = _
  rw [if_pos (by decide), root92Child2]
def root92Candidates3 : List Nat := [1249, 11317]
set_option maxRecDepth 8192 in
theorem root92Candidates3_checked : candidates edge 4 499 1033 root92Tail4 winner.bound = root92Candidates3 := by
  unfold candidates
  have row : edge 1033 = (SortedNatTree.contains FastRows.row1033) := by
    change Data.row1033.contains = (SortedNatTree.contains FastRows.row1033)
    exact FastRows.row1033_checked.symm
  rw [row]
  decide
theorem root92Step3 : scan edge (search edge 3) 4 [499] 499 root92Tail3 winner =
    scan edge (search edge 3) 4 [499] 499 root92Tail4 winner := by
  rw [root92Tail3, scan, if_neg (by decide), root92Candidates3_checked]
  change scan edge (search edge 3) 4 [499] 499 root92Tail4
    (if 3 ≤ root92Candidates3.length then search edge 3 [499, 1033] root92Candidates3 1532 winner else winner) = _
  rw [if_neg (by decide)]
def root92Candidates4 : List Nat := [19801]
set_option maxRecDepth 8192 in
theorem root92Candidates4_checked : candidates edge 4 499 1237 root92Tail5 winner.bound = root92Candidates4 := by
  unfold candidates
  have row : edge 1237 = (SortedNatTree.contains FastRows.row1237) := by
    change Data.row1237.contains = (SortedNatTree.contains FastRows.row1237)
    exact FastRows.row1237_checked.symm
  rw [row]
  decide
theorem root92Step4 : scan edge (search edge 3) 4 [499] 499 root92Tail4 winner =
    scan edge (search edge 3) 4 [499] 499 root92Tail5 winner := by
  rw [root92Tail4, scan, if_neg (by decide), root92Candidates4_checked]
  change scan edge (search edge 3) 4 [499] 499 root92Tail5
    (if 3 ≤ root92Candidates4.length then search edge 3 [499, 1237] root92Candidates4 1736 winner else winner) = _
  rw [if_neg (by decide)]
def root92Candidates5 : List Nat := [2551, 9187]
set_option maxRecDepth 8192 in
theorem root92Candidates5_checked : candidates edge 4 499 1249 root92Tail6 winner.bound = root92Candidates5 := by
  unfold candidates
  have row : edge 1249 = (SortedNatTree.contains FastRows.row1249) := by
    change Data.row1249.contains = (SortedNatTree.contains FastRows.row1249)
    exact FastRows.row1249_checked.symm
  rw [row]
  decide
theorem root92Step5 : scan edge (search edge 3) 4 [499] 499 root92Tail5 winner =
    scan edge (search edge 3) 4 [499] 499 root92Tail6 winner := by
  rw [root92Tail5, scan, if_neg (by decide), root92Candidates5_checked]
  change scan edge (search edge 3) 4 [499] 499 root92Tail6
    (if 3 ≤ root92Candidates5.length then search edge 3 [499, 1249] root92Candidates5 1748 winner else winner) = _
  rw [if_neg (by decide)]
def root92Candidates6 : List Nat := [23563]
set_option maxRecDepth 8192 in
theorem root92Candidates6_checked : candidates edge 4 499 1303 root92Tail7 winner.bound = root92Candidates6 := by
  unfold candidates
  have row : edge 1303 = (SortedNatTree.contains FastRows.row1303) := by
    change Data.row1303.contains = (SortedNatTree.contains FastRows.row1303)
    exact FastRows.row1303_checked.symm
  rw [row]
  decide
theorem root92Step6 : scan edge (search edge 3) 4 [499] 499 root92Tail6 winner =
    scan edge (search edge 3) 4 [499] 499 root92Tail7 winner := by
  rw [root92Tail6, scan, if_neg (by decide), root92Candidates6_checked]
  change scan edge (search edge 3) 4 [499] 499 root92Tail7
    (if 3 ≤ root92Candidates6.length then search edge 3 [499, 1303] root92Candidates6 1802 winner else winner) = _
  rw [if_neg (by decide)]
def root92Candidates7 : List Nat := [5281, 9103]
set_option maxRecDepth 8192 in
theorem root92Candidates7_checked : candidates edge 4 499 1453 root92Tail8 winner.bound = root92Candidates7 := by
  unfold candidates
  have row : edge 1453 = (SortedNatTree.contains FastRows.row1453) := by
    change Data.row1453.contains = (SortedNatTree.contains FastRows.row1453)
    exact FastRows.row1453_checked.symm
  rw [row]
  decide
theorem root92Step7 : scan edge (search edge 3) 4 [499] 499 root92Tail7 winner =
    scan edge (search edge 3) 4 [499] 499 root92Tail8 winner := by
  rw [root92Tail7, scan, if_neg (by decide), root92Candidates7_checked]
  change scan edge (search edge 3) 4 [499] 499 root92Tail8
    (if 3 ≤ root92Candidates7.length then search edge 3 [499, 1453] root92Candidates7 1952 winner else winner) = _
  rw [if_neg (by decide)]
def root92Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root92Candidates8_checked : candidates edge 4 499 1933 root92Tail9 winner.bound = root92Candidates8 := by
  unfold candidates
  have row : edge 1933 = (SortedNatTree.contains FastRows.row1933) := by
    change Data.row1933.contains = (SortedNatTree.contains FastRows.row1933)
    exact FastRows.row1933_checked.symm
  rw [row]
  decide
theorem root92Step8 : scan edge (search edge 3) 4 [499] 499 root92Tail8 winner =
    scan edge (search edge 3) 4 [499] 499 root92Tail9 winner := by
  rw [root92Tail8, scan, if_neg (by decide), root92Candidates8_checked]
  change scan edge (search edge 3) 4 [499] 499 root92Tail9
    (if 3 ≤ root92Candidates8.length then search edge 3 [499, 1933] root92Candidates8 2432 winner else winner) = _
  rw [if_neg (by decide)]
def root92Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root92Candidates9_checked : candidates edge 4 499 2551 root92Tail10 winner.bound = root92Candidates9 := by
  unfold candidates
  have row : edge 2551 = (SortedNatTree.contains FastRows.row2551) := by
    change Data.row2551.contains = (SortedNatTree.contains FastRows.row2551)
    exact FastRows.row2551_checked.symm
  rw [row]
  decide
theorem root92Step9 : scan edge (search edge 3) 4 [499] 499 root92Tail9 winner =
    scan edge (search edge 3) 4 [499] 499 root92Tail10 winner := by
  rw [root92Tail9, scan, if_neg (by decide), root92Candidates9_checked]
  change scan edge (search edge 3) 4 [499] 499 root92Tail10
    (if 3 ≤ root92Candidates9.length then search edge 3 [499, 2551] root92Candidates9 3050 winner else winner) = _
  rw [if_neg (by decide)]
def root92Candidates10 : List Nat := [14281]
set_option maxRecDepth 8192 in
theorem root92Candidates10_checked : candidates edge 4 499 2707 root92Tail11 winner.bound = root92Candidates10 := by
  unfold candidates
  have row : edge 2707 = (SortedNatTree.contains FastRows.row2707) := by
    change Data.row2707.contains = (SortedNatTree.contains FastRows.row2707)
    exact FastRows.row2707_checked.symm
  rw [row]
  decide
theorem root92Step10 : scan edge (search edge 3) 4 [499] 499 root92Tail10 winner =
    scan edge (search edge 3) 4 [499] 499 root92Tail11 winner := by
  rw [root92Tail10, scan, if_neg (by decide), root92Candidates10_checked]
  change scan edge (search edge 3) 4 [499] 499 root92Tail11
    (if 3 ≤ root92Candidates10.length then search edge 3 [499, 2707] root92Candidates10 3206 winner else winner) = _
  rw [if_neg (by decide)]
def root92Candidates11 : List Nat := [7687, 8467, 17449]
set_option maxRecDepth 8192 in
theorem root92Candidates11_checked : candidates edge 4 499 3343 root92Tail12 winner.bound = root92Candidates11 := by
  unfold candidates
  have row : edge 3343 = (SortedNatTree.contains FastRows.row3343) := by
    change Data.row3343.contains = (SortedNatTree.contains FastRows.row3343)
    exact FastRows.row3343_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root92Child11 : search edge 3 [499, 3343] root92Candidates11 3842 winner = winner := by decide
theorem root92Step11 : scan edge (search edge 3) 4 [499] 499 root92Tail11 winner =
    scan edge (search edge 3) 4 [499] 499 root92Tail12 winner := by
  rw [root92Tail11, scan, if_neg (by decide), root92Candidates11_checked]
  change scan edge (search edge 3) 4 [499] 499 root92Tail12
    (if 3 ≤ root92Candidates11.length then search edge 3 [499, 3343] root92Candidates11 3842 winner else winner) = _
  rw [if_pos (by decide), root92Child11]
def root92Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root92Candidates12_checked : candidates edge 4 499 3847 root92Tail13 winner.bound = root92Candidates12 := by
  unfold candidates
  have row : edge 3847 = (SortedNatTree.contains FastRows.row3847) := by
    change Data.row3847.contains = (SortedNatTree.contains FastRows.row3847)
    exact FastRows.row3847_checked.symm
  rw [row]
  decide
theorem root92Step12 : scan edge (search edge 3) 4 [499] 499 root92Tail12 winner =
    scan edge (search edge 3) 4 [499] 499 root92Tail13 winner := by
  rw [root92Tail12, scan, if_neg (by decide), root92Candidates12_checked]
  change scan edge (search edge 3) 4 [499] 499 root92Tail13
    (if 3 ≤ root92Candidates12.length then search edge 3 [499, 3847] root92Candidates12 4346 winner else winner) = _
  rw [if_neg (by decide)]
def root92Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root92Candidates13_checked : candidates edge 4 499 3907 root92Tail14 winner.bound = root92Candidates13 := by
  unfold candidates
  have row : edge 3907 = (SortedNatTree.contains FastRows.row3907) := by
    change Data.row3907.contains = (SortedNatTree.contains FastRows.row3907)
    exact FastRows.row3907_checked.symm
  rw [row]
  decide
theorem root92Step13 : scan edge (search edge 3) 4 [499] 499 root92Tail13 winner =
    scan edge (search edge 3) 4 [499] 499 root92Tail14 winner := by
  rw [root92Tail13, scan, if_neg (by decide), root92Candidates13_checked]
  change scan edge (search edge 3) 4 [499] 499 root92Tail14
    (if 3 ≤ root92Candidates13.length then search edge 3 [499, 3907] root92Candidates13 4406 winner else winner) = _
  rw [if_neg (by decide)]
def root92Candidates14 : List Nat := [18457]
set_option maxRecDepth 8192 in
theorem root92Candidates14_checked : candidates edge 4 499 4051 root92Tail15 winner.bound = root92Candidates14 := by
  unfold candidates
  have row : edge 4051 = (SortedNatTree.contains FastRows.row4051) := by
    change Data.row4051.contains = (SortedNatTree.contains FastRows.row4051)
    exact FastRows.row4051_checked.symm
  rw [row]
  decide
theorem root92Step14 : scan edge (search edge 3) 4 [499] 499 root92Tail14 winner =
    scan edge (search edge 3) 4 [499] 499 root92Tail15 winner := by
  rw [root92Tail14, scan, if_neg (by decide), root92Candidates14_checked]
  change scan edge (search edge 3) 4 [499] 499 root92Tail15
    (if 3 ≤ root92Candidates14.length then search edge 3 [499, 4051] root92Candidates14 4550 winner else winner) = _
  rw [if_neg (by decide)]
def root92Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root92Candidates15_checked : candidates edge 4 499 4357 root92Tail16 winner.bound = root92Candidates15 := by
  unfold candidates
  have row : edge 4357 = (SortedNatTree.contains FastRows.row4357) := by
    change Data.row4357.contains = (SortedNatTree.contains FastRows.row4357)
    exact FastRows.row4357_checked.symm
  rw [row]
  decide
theorem root92Step15 : scan edge (search edge 3) 4 [499] 499 root92Tail15 winner =
    scan edge (search edge 3) 4 [499] 499 root92Tail16 winner := by
  rw [root92Tail15, scan, if_neg (by decide), root92Candidates15_checked]
  change scan edge (search edge 3) 4 [499] 499 root92Tail16
    (if 3 ≤ root92Candidates15.length then search edge 3 [499, 4357] root92Candidates15 4856 winner else winner) = _
  rw [if_neg (by decide)]
def root92Candidates16 : List Nat := [9187, 11353]
set_option maxRecDepth 8192 in
theorem root92Candidates16_checked : candidates edge 4 499 4513 root92Tail17 winner.bound = root92Candidates16 := by
  unfold candidates
  have row : edge 4513 = (SortedNatTree.contains FastRows.row4513) := by
    change Data.row4513.contains = (SortedNatTree.contains FastRows.row4513)
    exact FastRows.row4513_checked.symm
  rw [row]
  decide
theorem root92Step16 : scan edge (search edge 3) 4 [499] 499 root92Tail16 winner =
    scan edge (search edge 3) 4 [499] 499 root92Tail17 winner := by
  rw [root92Tail16, scan, if_neg (by decide), root92Candidates16_checked]
  change scan edge (search edge 3) 4 [499] 499 root92Tail17
    (if 3 ≤ root92Candidates16.length then search edge 3 [499, 4513] root92Candidates16 5012 winner else winner) = _
  rw [if_neg (by decide)]
def root92Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root92Candidates17_checked : candidates edge 4 499 4621 root92Tail18 winner.bound = root92Candidates17 := by
  unfold candidates
  have row : edge 4621 = (SortedNatTree.contains FastRows.row4621) := by
    change Data.row4621.contains = (SortedNatTree.contains FastRows.row4621)
    exact FastRows.row4621_checked.symm
  rw [row]
  decide
theorem root92Step17 : scan edge (search edge 3) 4 [499] 499 root92Tail17 winner =
    scan edge (search edge 3) 4 [499] 499 root92Tail18 winner := by
  rw [root92Tail17, scan, if_neg (by decide), root92Candidates17_checked]
  change scan edge (search edge 3) 4 [499] 499 root92Tail18
    (if 3 ≤ root92Candidates17.length then search edge 3 [499, 4621] root92Candidates17 5120 winner else winner) = _
  rw [if_neg (by decide)]
def root92Candidates18 : List Nat := [16987]
set_option maxRecDepth 8192 in
theorem root92Candidates18_checked : candidates edge 4 499 5281 root92Tail19 winner.bound = root92Candidates18 := by
  unfold candidates
  have row : edge 5281 = (SortedNatTree.contains FastRows.row5281) := by
    change Data.row5281.contains = (SortedNatTree.contains FastRows.row5281)
    exact FastRows.row5281_checked.symm
  rw [row]
  decide
theorem root92Step18 : scan edge (search edge 3) 4 [499] 499 root92Tail18 winner =
    scan edge (search edge 3) 4 [499] 499 root92Tail19 winner := by
  rw [root92Tail18, scan, if_neg (by decide), root92Candidates18_checked]
  change scan edge (search edge 3) 4 [499] 499 root92Tail19
    (if 3 ≤ root92Candidates18.length then search edge 3 [499, 5281] root92Candidates18 5780 winner else winner) = _
  rw [if_neg (by decide)]
def root92Candidates19 : List Nat := [7393]
set_option maxRecDepth 8192 in
theorem root92Candidates19_checked : candidates edge 4 499 5503 root92Tail20 winner.bound = root92Candidates19 := by
  unfold candidates
  have row : edge 5503 = (SortedNatTree.contains FastRows.row5503) := by
    change Data.row5503.contains = (SortedNatTree.contains FastRows.row5503)
    exact FastRows.row5503_checked.symm
  rw [row]
  decide
theorem root92Step19 : scan edge (search edge 3) 4 [499] 499 root92Tail19 winner =
    scan edge (search edge 3) 4 [499] 499 root92Tail20 winner := by
  rw [root92Tail19, scan, if_neg (by decide), root92Candidates19_checked]
  change scan edge (search edge 3) 4 [499] 499 root92Tail20
    (if 3 ≤ root92Candidates19.length then search edge 3 [499, 5503] root92Candidates19 6002 winner else winner) = _
  rw [if_neg (by decide)]
def root92Candidates20 : List Nat := [9103]
set_option maxRecDepth 8192 in
theorem root92Candidates20_checked : candidates edge 4 499 5827 root92Tail21 winner.bound = root92Candidates20 := by
  unfold candidates
  have row : edge 5827 = (SortedNatTree.contains FastRows.row5827) := by
    change Data.row5827.contains = (SortedNatTree.contains FastRows.row5827)
    exact FastRows.row5827_checked.symm
  rw [row]
  decide
theorem root92Step20 : scan edge (search edge 3) 4 [499] 499 root92Tail20 winner =
    scan edge (search edge 3) 4 [499] 499 root92Tail21 winner := by
  rw [root92Tail20, scan, if_neg (by decide), root92Candidates20_checked]
  change scan edge (search edge 3) 4 [499] 499 root92Tail21
    (if 3 ≤ root92Candidates20.length then search edge 3 [499, 5827] root92Candidates20 6326 winner else winner) = _
  rw [if_neg (by decide)]
def root92Candidates21 : List Nat := []
set_option maxRecDepth 8192 in
theorem root92Candidates21_checked : candidates edge 4 499 6361 root92Tail22 winner.bound = root92Candidates21 := by
  unfold candidates
  have row : edge 6361 = (SortedNatTree.contains FastRows.row6361) := by
    change Data.row6361.contains = (SortedNatTree.contains FastRows.row6361)
    exact FastRows.row6361_checked.symm
  rw [row]
  decide
theorem root92Step21 : scan edge (search edge 3) 4 [499] 499 root92Tail21 winner =
    scan edge (search edge 3) 4 [499] 499 root92Tail22 winner := by
  rw [root92Tail21, scan, if_neg (by decide), root92Candidates21_checked]
  change scan edge (search edge 3) 4 [499] 499 root92Tail22
    (if 3 ≤ root92Candidates21.length then search edge 3 [499, 6361] root92Candidates21 6860 winner else winner) = _
  rw [if_neg (by decide)]
theorem root92Scan22 : scan edge (search edge 3) 4 [499] 499 root92Tail22 winner = winner := by rw [root92Tail22, scan, if_pos (by decide)]
theorem root92Scan21 : scan edge (search edge 3) 4 [499] 499 root92Tail21 winner = winner :=
  (root92Step21).trans root92Scan22
theorem root92Scan20 : scan edge (search edge 3) 4 [499] 499 root92Tail20 winner = winner :=
  (root92Step20).trans root92Scan21
theorem root92Scan19 : scan edge (search edge 3) 4 [499] 499 root92Tail19 winner = winner :=
  (root92Step19).trans root92Scan20
theorem root92Scan18 : scan edge (search edge 3) 4 [499] 499 root92Tail18 winner = winner :=
  (root92Step18).trans root92Scan19
theorem root92Scan17 : scan edge (search edge 3) 4 [499] 499 root92Tail17 winner = winner :=
  (root92Step17).trans root92Scan18
theorem root92Scan16 : scan edge (search edge 3) 4 [499] 499 root92Tail16 winner = winner :=
  (root92Step16).trans root92Scan17
theorem root92Scan15 : scan edge (search edge 3) 4 [499] 499 root92Tail15 winner = winner :=
  (root92Step15).trans root92Scan16
theorem root92Scan14 : scan edge (search edge 3) 4 [499] 499 root92Tail14 winner = winner :=
  (root92Step14).trans root92Scan15
theorem root92Scan13 : scan edge (search edge 3) 4 [499] 499 root92Tail13 winner = winner :=
  (root92Step13).trans root92Scan14
theorem root92Scan12 : scan edge (search edge 3) 4 [499] 499 root92Tail12 winner = winner :=
  (root92Step12).trans root92Scan13
theorem root92Scan11 : scan edge (search edge 3) 4 [499] 499 root92Tail11 winner = winner :=
  (root92Step11).trans root92Scan12
theorem root92Scan10 : scan edge (search edge 3) 4 [499] 499 root92Tail10 winner = winner :=
  (root92Step10).trans root92Scan11
theorem root92Scan9 : scan edge (search edge 3) 4 [499] 499 root92Tail9 winner = winner :=
  (root92Step9).trans root92Scan10
theorem root92Scan8 : scan edge (search edge 3) 4 [499] 499 root92Tail8 winner = winner :=
  (root92Step8).trans root92Scan9
theorem root92Scan7 : scan edge (search edge 3) 4 [499] 499 root92Tail7 winner = winner :=
  (root92Step7).trans root92Scan8
theorem root92Scan6 : scan edge (search edge 3) 4 [499] 499 root92Tail6 winner = winner :=
  (root92Step6).trans root92Scan7
theorem root92Scan5 : scan edge (search edge 3) 4 [499] 499 root92Tail5 winner = winner :=
  (root92Step5).trans root92Scan6
theorem root92Scan4 : scan edge (search edge 3) 4 [499] 499 root92Tail4 winner = winner :=
  (root92Step4).trans root92Scan5
theorem root92Scan3 : scan edge (search edge 3) 4 [499] 499 root92Tail3 winner = winner :=
  (root92Step3).trans root92Scan4
theorem root92Scan2 : scan edge (search edge 3) 4 [499] 499 root92Tail2 winner = winner :=
  (root92Step2).trans root92Scan3
theorem root92Scan1 : scan edge (search edge 3) 4 [499] 499 root92Tail1 winner = winner :=
  (root92Step1).trans root92Scan2
theorem root92Scan0 : scan edge (search edge 3) 4 [499] 499 root92Tail0 winner = winner :=
  (root92Step0).trans root92Scan1
set_option maxRecDepth 4096 in
theorem root92Branch : search edge 4 [499] root92Candidates 499 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [499] 499 root92Tail0 winner = winner
  exact root92Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
