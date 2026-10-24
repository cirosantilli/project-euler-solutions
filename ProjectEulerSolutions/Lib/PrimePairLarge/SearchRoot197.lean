import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root197Candidates : List Nat := Data.row1223
set_option maxRecDepth 8192 in
theorem root197Candidates_checked : candidates edge 5 0 1223 tail198 winner.bound = root197Candidates := by
  apply candidates_eq_row edge 5 0 1223 tail198 root197Candidates winner.bound tail198_ordered
  · decide
  · decide
  · funext q
    rfl
def root197Tail50 : List Nat := []
def root197Tail49 : List Nat := 24179 :: root197Tail50
def root197Tail48 : List Nat := 22787 :: root197Tail49
def root197Tail47 : List Nat := 22727 :: root197Tail48
def root197Tail46 : List Nat := 22433 :: root197Tail47
def root197Tail45 : List Nat := 20981 :: root197Tail46
def root197Tail44 : List Nat := 20789 :: root197Tail45
def root197Tail43 : List Nat := 20477 :: root197Tail44
def root197Tail42 : List Nat := 19973 :: root197Tail43
def root197Tail41 : List Nat := 19577 :: root197Tail42
def root197Tail40 : List Nat := 19157 :: root197Tail41
def root197Tail39 : List Nat := 18947 :: root197Tail40
def root197Tail38 : List Nat := 18593 :: root197Tail39
def root197Tail37 : List Nat := 17033 :: root197Tail38
def root197Tail36 : List Nat := 15329 :: root197Tail37
def root197Tail35 : List Nat := 14867 :: root197Tail36
def root197Tail34 : List Nat := 14519 :: root197Tail35
def root197Tail33 : List Nat := 14033 :: root197Tail34
def root197Tail32 : List Nat := 13757 :: root197Tail33
def root197Tail31 : List Nat := 13229 :: root197Tail32
def root197Tail30 : List Nat := 12893 :: root197Tail31
def root197Tail29 : List Nat := 12479 :: root197Tail30
def root197Tail28 : List Nat := 11069 :: root197Tail29
def root197Tail27 : List Nat := 8867 :: root197Tail28
def root197Tail26 : List Nat := 8423 :: root197Tail27
def root197Tail25 : List Nat := 8297 :: root197Tail26
def root197Tail24 : List Nat := 7703 :: root197Tail25
def root197Tail23 : List Nat := 7559 :: root197Tail24
def root197Tail22 : List Nat := 7523 :: root197Tail23
def root197Tail21 : List Nat := 6761 :: root197Tail22
def root197Tail20 : List Nat := 6701 :: root197Tail21
def root197Tail19 : List Nat := 6131 :: root197Tail20
def root197Tail18 : List Nat := 6113 :: root197Tail19
def root197Tail17 : List Nat := 5939 :: root197Tail18
def root197Tail16 : List Nat := 5897 :: root197Tail17
def root197Tail15 : List Nat := 5381 :: root197Tail16
def root197Tail14 : List Nat := 4793 :: root197Tail15
def root197Tail13 : List Nat := 4517 :: root197Tail14
def root197Tail12 : List Nat := 3467 :: root197Tail13
def root197Tail11 : List Nat := 3323 :: root197Tail12
def root197Tail10 : List Nat := 2777 :: root197Tail11
def root197Tail9 : List Nat := 2687 :: root197Tail10
def root197Tail8 : List Nat := 2621 :: root197Tail9
def root197Tail7 : List Nat := 2579 :: root197Tail8
def root197Tail6 : List Nat := 2543 :: root197Tail7
def root197Tail5 : List Nat := 2141 :: root197Tail6
def root197Tail4 : List Nat := 1931 :: root197Tail5
def root197Tail3 : List Nat := 1697 :: root197Tail4
def root197Tail2 : List Nat := 1493 :: root197Tail3
def root197Tail1 : List Nat := 1433 :: root197Tail2
def root197Tail0 : List Nat := 1283 :: root197Tail1
def root197Candidates0 : List Nat := [14867]
set_option maxRecDepth 8192 in
theorem root197Candidates0_checked : candidates edge 4 1223 1283 root197Tail1 winner.bound = root197Candidates0 := by
  unfold candidates
  have row : edge 1283 = (SortedNatTree.contains FastRows.row1283) := by
    change Data.row1283.contains = (SortedNatTree.contains FastRows.row1283)
    exact FastRows.row1283_checked.symm
  rw [row]
  decide
theorem root197Step0 : scan edge (search edge 3) 4 [1223] 1223 root197Tail0 winner =
    scan edge (search edge 3) 4 [1223] 1223 root197Tail1 winner := by
  rw [root197Tail0, scan, if_neg (by decide), root197Candidates0_checked]
  change scan edge (search edge 3) 4 [1223] 1223 root197Tail1
    (if 3 ≤ root197Candidates0.length then search edge 3 [1223, 1283] root197Candidates0 2506 winner else winner) = _
  rw [if_neg (by decide)]
def root197Candidates1 : List Nat := [3323, 6761, 8297, 12893, 15329]
set_option maxRecDepth 8192 in
theorem root197Candidates1_checked : candidates edge 4 1223 1433 root197Tail2 winner.bound = root197Candidates1 := by
  unfold candidates
  have row : edge 1433 = (SortedNatTree.contains FastRows.row1433) := by
    change Data.row1433.contains = (SortedNatTree.contains FastRows.row1433)
    exact FastRows.row1433_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root197Child1 : search edge 3 [1223, 1433] root197Candidates1 2656 winner = winner := by decide
theorem root197Step1 : scan edge (search edge 3) 4 [1223] 1223 root197Tail1 winner =
    scan edge (search edge 3) 4 [1223] 1223 root197Tail2 winner := by
  rw [root197Tail1, scan, if_neg (by decide), root197Candidates1_checked]
  change scan edge (search edge 3) 4 [1223] 1223 root197Tail2
    (if 3 ≤ root197Candidates1.length then search edge 3 [1223, 1433] root197Candidates1 2656 winner else winner) = _
  rw [if_pos (by decide), root197Child1]
def root197Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root197Candidates2_checked : candidates edge 4 1223 1493 root197Tail3 winner.bound = root197Candidates2 := by
  unfold candidates
  have row : edge 1493 = (SortedNatTree.contains FastRows.row1493) := by
    change Data.row1493.contains = (SortedNatTree.contains FastRows.row1493)
    exact FastRows.row1493_checked.symm
  rw [row]
  decide
theorem root197Step2 : scan edge (search edge 3) 4 [1223] 1223 root197Tail2 winner =
    scan edge (search edge 3) 4 [1223] 1223 root197Tail3 winner := by
  rw [root197Tail2, scan, if_neg (by decide), root197Candidates2_checked]
  change scan edge (search edge 3) 4 [1223] 1223 root197Tail3
    (if 3 ≤ root197Candidates2.length then search edge 3 [1223, 1493] root197Candidates2 2716 winner else winner) = _
  rw [if_neg (by decide)]
def root197Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root197Candidates3_checked : candidates edge 4 1223 1697 root197Tail4 winner.bound = root197Candidates3 := by
  unfold candidates
  have row : edge 1697 = (SortedNatTree.contains FastRows.row1697) := by
    change Data.row1697.contains = (SortedNatTree.contains FastRows.row1697)
    exact FastRows.row1697_checked.symm
  rw [row]
  decide
theorem root197Step3 : scan edge (search edge 3) 4 [1223] 1223 root197Tail3 winner =
    scan edge (search edge 3) 4 [1223] 1223 root197Tail4 winner := by
  rw [root197Tail3, scan, if_neg (by decide), root197Candidates3_checked]
  change scan edge (search edge 3) 4 [1223] 1223 root197Tail4
    (if 3 ≤ root197Candidates3.length then search edge 3 [1223, 1697] root197Candidates3 2920 winner else winner) = _
  rw [if_neg (by decide)]
def root197Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root197Candidates4_checked : candidates edge 4 1223 1931 root197Tail5 winner.bound = root197Candidates4 := by
  unfold candidates
  have row : edge 1931 = (SortedNatTree.contains FastRows.row1931) := by
    change Data.row1931.contains = (SortedNatTree.contains FastRows.row1931)
    exact FastRows.row1931_checked.symm
  rw [row]
  decide
theorem root197Step4 : scan edge (search edge 3) 4 [1223] 1223 root197Tail4 winner =
    scan edge (search edge 3) 4 [1223] 1223 root197Tail5 winner := by
  rw [root197Tail4, scan, if_neg (by decide), root197Candidates4_checked]
  change scan edge (search edge 3) 4 [1223] 1223 root197Tail5
    (if 3 ≤ root197Candidates4.length then search edge 3 [1223, 1931] root197Candidates4 3154 winner else winner) = _
  rw [if_neg (by decide)]
def root197Candidates5 : List Nat := [2621, 20789]
set_option maxRecDepth 8192 in
theorem root197Candidates5_checked : candidates edge 4 1223 2141 root197Tail6 winner.bound = root197Candidates5 := by
  unfold candidates
  have row : edge 2141 = (SortedNatTree.contains FastRows.row2141) := by
    change Data.row2141.contains = (SortedNatTree.contains FastRows.row2141)
    exact FastRows.row2141_checked.symm
  rw [row]
  decide
theorem root197Step5 : scan edge (search edge 3) 4 [1223] 1223 root197Tail5 winner =
    scan edge (search edge 3) 4 [1223] 1223 root197Tail6 winner := by
  rw [root197Tail5, scan, if_neg (by decide), root197Candidates5_checked]
  change scan edge (search edge 3) 4 [1223] 1223 root197Tail6
    (if 3 ≤ root197Candidates5.length then search edge 3 [1223, 2141] root197Candidates5 3364 winner else winner) = _
  rw [if_neg (by decide)]
def root197Candidates6 : List Nat := [12893, 15329, 19973]
set_option maxRecDepth 8192 in
theorem root197Candidates6_checked : candidates edge 4 1223 2543 root197Tail7 winner.bound = root197Candidates6 := by
  unfold candidates
  have row : edge 2543 = (SortedNatTree.contains FastRows.row2543) := by
    change Data.row2543.contains = (SortedNatTree.contains FastRows.row2543)
    exact FastRows.row2543_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root197Child6 : search edge 3 [1223, 2543] root197Candidates6 3766 winner = winner := by decide
theorem root197Step6 : scan edge (search edge 3) 4 [1223] 1223 root197Tail6 winner =
    scan edge (search edge 3) 4 [1223] 1223 root197Tail7 winner := by
  rw [root197Tail6, scan, if_neg (by decide), root197Candidates6_checked]
  change scan edge (search edge 3) 4 [1223] 1223 root197Tail7
    (if 3 ≤ root197Candidates6.length then search edge 3 [1223, 2543] root197Candidates6 3766 winner else winner) = _
  rw [if_pos (by decide), root197Child6]
def root197Candidates7 : List Nat := [6131]
set_option maxRecDepth 8192 in
theorem root197Candidates7_checked : candidates edge 4 1223 2579 root197Tail8 winner.bound = root197Candidates7 := by
  unfold candidates
  have row : edge 2579 = (SortedNatTree.contains FastRows.row2579) := by
    change Data.row2579.contains = (SortedNatTree.contains FastRows.row2579)
    exact FastRows.row2579_checked.symm
  rw [row]
  decide
theorem root197Step7 : scan edge (search edge 3) 4 [1223] 1223 root197Tail7 winner =
    scan edge (search edge 3) 4 [1223] 1223 root197Tail8 winner := by
  rw [root197Tail7, scan, if_neg (by decide), root197Candidates7_checked]
  change scan edge (search edge 3) 4 [1223] 1223 root197Tail8
    (if 3 ≤ root197Candidates7.length then search edge 3 [1223, 2579] root197Candidates7 3802 winner else winner) = _
  rw [if_neg (by decide)]
def root197Candidates8 : List Nat := [13757]
set_option maxRecDepth 8192 in
theorem root197Candidates8_checked : candidates edge 4 1223 2621 root197Tail9 winner.bound = root197Candidates8 := by
  unfold candidates
  have row : edge 2621 = (SortedNatTree.contains FastRows.row2621) := by
    change Data.row2621.contains = (SortedNatTree.contains FastRows.row2621)
    exact FastRows.row2621_checked.symm
  rw [row]
  decide
theorem root197Step8 : scan edge (search edge 3) 4 [1223] 1223 root197Tail8 winner =
    scan edge (search edge 3) 4 [1223] 1223 root197Tail9 winner := by
  rw [root197Tail8, scan, if_neg (by decide), root197Candidates8_checked]
  change scan edge (search edge 3) 4 [1223] 1223 root197Tail9
    (if 3 ≤ root197Candidates8.length then search edge 3 [1223, 2621] root197Candidates8 3844 winner else winner) = _
  rw [if_neg (by decide)]
def root197Candidates9 : List Nat := [5897, 7703]
set_option maxRecDepth 8192 in
theorem root197Candidates9_checked : candidates edge 4 1223 2687 root197Tail10 winner.bound = root197Candidates9 := by
  unfold candidates
  have row : edge 2687 = (SortedNatTree.contains FastRows.row2687) := by
    change Data.row2687.contains = (SortedNatTree.contains FastRows.row2687)
    exact FastRows.row2687_checked.symm
  rw [row]
  decide
theorem root197Step9 : scan edge (search edge 3) 4 [1223] 1223 root197Tail9 winner =
    scan edge (search edge 3) 4 [1223] 1223 root197Tail10 winner := by
  rw [root197Tail9, scan, if_neg (by decide), root197Candidates9_checked]
  change scan edge (search edge 3) 4 [1223] 1223 root197Tail10
    (if 3 ≤ root197Candidates9.length then search edge 3 [1223, 2687] root197Candidates9 3910 winner else winner) = _
  rw [if_neg (by decide)]
def root197Candidates10 : List Nat := [4793]
set_option maxRecDepth 8192 in
theorem root197Candidates10_checked : candidates edge 4 1223 2777 root197Tail11 winner.bound = root197Candidates10 := by
  unfold candidates
  have row : edge 2777 = (SortedNatTree.contains FastRows.row2777) := by
    change Data.row2777.contains = (SortedNatTree.contains FastRows.row2777)
    exact FastRows.row2777_checked.symm
  rw [row]
  decide
theorem root197Step10 : scan edge (search edge 3) 4 [1223] 1223 root197Tail10 winner =
    scan edge (search edge 3) 4 [1223] 1223 root197Tail11 winner := by
  rw [root197Tail10, scan, if_neg (by decide), root197Candidates10_checked]
  change scan edge (search edge 3) 4 [1223] 1223 root197Tail11
    (if 3 ≤ root197Candidates10.length then search edge 3 [1223, 2777] root197Candidates10 4000 winner else winner) = _
  rw [if_neg (by decide)]
def root197Candidates11 : List Nat := [5381, 8423, 19577]
set_option maxRecDepth 8192 in
theorem root197Candidates11_checked : candidates edge 4 1223 3323 root197Tail12 winner.bound = root197Candidates11 := by
  unfold candidates
  have row : edge 3323 = (SortedNatTree.contains FastRows.row3323) := by
    change Data.row3323.contains = (SortedNatTree.contains FastRows.row3323)
    exact FastRows.row3323_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root197Child11 : search edge 3 [1223, 3323] root197Candidates11 4546 winner = winner := by decide
theorem root197Step11 : scan edge (search edge 3) 4 [1223] 1223 root197Tail11 winner =
    scan edge (search edge 3) 4 [1223] 1223 root197Tail12 winner := by
  rw [root197Tail11, scan, if_neg (by decide), root197Candidates11_checked]
  change scan edge (search edge 3) 4 [1223] 1223 root197Tail12
    (if 3 ≤ root197Candidates11.length then search edge 3 [1223, 3323] root197Candidates11 4546 winner else winner) = _
  rw [if_pos (by decide), root197Child11]
def root197Candidates12 : List Nat := [6113, 18947]
set_option maxRecDepth 8192 in
theorem root197Candidates12_checked : candidates edge 4 1223 3467 root197Tail13 winner.bound = root197Candidates12 := by
  unfold candidates
  have row : edge 3467 = (SortedNatTree.contains FastRows.row3467) := by
    change Data.row3467.contains = (SortedNatTree.contains FastRows.row3467)
    exact FastRows.row3467_checked.symm
  rw [row]
  decide
theorem root197Step12 : scan edge (search edge 3) 4 [1223] 1223 root197Tail12 winner =
    scan edge (search edge 3) 4 [1223] 1223 root197Tail13 winner := by
  rw [root197Tail12, scan, if_neg (by decide), root197Candidates12_checked]
  change scan edge (search edge 3) 4 [1223] 1223 root197Tail13
    (if 3 ≤ root197Candidates12.length then search edge 3 [1223, 3467] root197Candidates12 4690 winner else winner) = _
  rw [if_neg (by decide)]
def root197Candidates13 : List Nat := [7523]
set_option maxRecDepth 8192 in
theorem root197Candidates13_checked : candidates edge 4 1223 4517 root197Tail14 winner.bound = root197Candidates13 := by
  unfold candidates
  have row : edge 4517 = (SortedNatTree.contains FastRows.row4517) := by
    change Data.row4517.contains = (SortedNatTree.contains FastRows.row4517)
    exact FastRows.row4517_checked.symm
  rw [row]
  decide
theorem root197Step13 : scan edge (search edge 3) 4 [1223] 1223 root197Tail13 winner =
    scan edge (search edge 3) 4 [1223] 1223 root197Tail14 winner := by
  rw [root197Tail13, scan, if_neg (by decide), root197Candidates13_checked]
  change scan edge (search edge 3) 4 [1223] 1223 root197Tail14
    (if 3 ≤ root197Candidates13.length then search edge 3 [1223, 4517] root197Candidates13 5740 winner else winner) = _
  rw [if_neg (by decide)]
def root197Candidates14 : List Nat := [13229]
set_option maxRecDepth 8192 in
theorem root197Candidates14_checked : candidates edge 4 1223 4793 root197Tail15 winner.bound = root197Candidates14 := by
  unfold candidates
  have row : edge 4793 = (SortedNatTree.contains FastRows.row4793) := by
    change Data.row4793.contains = (SortedNatTree.contains FastRows.row4793)
    exact FastRows.row4793_checked.symm
  rw [row]
  decide
theorem root197Step14 : scan edge (search edge 3) 4 [1223] 1223 root197Tail14 winner =
    scan edge (search edge 3) 4 [1223] 1223 root197Tail15 winner := by
  rw [root197Tail14, scan, if_neg (by decide), root197Candidates14_checked]
  change scan edge (search edge 3) 4 [1223] 1223 root197Tail15
    (if 3 ≤ root197Candidates14.length then search edge 3 [1223, 4793] root197Candidates14 6016 winner else winner) = _
  rw [if_neg (by decide)]
def root197Candidates15 : List Nat := [13229]
set_option maxRecDepth 8192 in
theorem root197Candidates15_checked : candidates edge 4 1223 5381 root197Tail16 winner.bound = root197Candidates15 := by
  unfold candidates
  have row : edge 5381 = (SortedNatTree.contains FastRows.row5381) := by
    change Data.row5381.contains = (SortedNatTree.contains FastRows.row5381)
    exact FastRows.row5381_checked.symm
  rw [row]
  decide
theorem root197Step15 : scan edge (search edge 3) 4 [1223] 1223 root197Tail15 winner =
    scan edge (search edge 3) 4 [1223] 1223 root197Tail16 winner := by
  rw [root197Tail15, scan, if_neg (by decide), root197Candidates15_checked]
  change scan edge (search edge 3) 4 [1223] 1223 root197Tail16
    (if 3 ≤ root197Candidates15.length then search edge 3 [1223, 5381] root197Candidates15 6604 winner else winner) = _
  rw [if_neg (by decide)]
def root197Candidates16 : List Nat := [6131, 8867]
set_option maxRecDepth 8192 in
theorem root197Candidates16_checked : candidates edge 4 1223 5897 root197Tail17 winner.bound = root197Candidates16 := by
  unfold candidates
  have row : edge 5897 = (SortedNatTree.contains FastRows.row5897) := by
    change Data.row5897.contains = (SortedNatTree.contains FastRows.row5897)
    exact FastRows.row5897_checked.symm
  rw [row]
  decide
theorem root197Step16 : scan edge (search edge 3) 4 [1223] 1223 root197Tail16 winner =
    scan edge (search edge 3) 4 [1223] 1223 root197Tail17 winner := by
  rw [root197Tail16, scan, if_neg (by decide), root197Candidates16_checked]
  change scan edge (search edge 3) 4 [1223] 1223 root197Tail17
    (if 3 ≤ root197Candidates16.length then search edge 3 [1223, 5897] root197Candidates16 7120 winner else winner) = _
  rw [if_neg (by decide)]
def root197Candidates17 : List Nat := [13757, 17033]
set_option maxRecDepth 8192 in
theorem root197Candidates17_checked : candidates edge 4 1223 5939 root197Tail18 winner.bound = root197Candidates17 := by
  unfold candidates
  have row : edge 5939 = (SortedNatTree.contains FastRows.row5939) := by
    change Data.row5939.contains = (SortedNatTree.contains FastRows.row5939)
    exact FastRows.row5939_checked.symm
  rw [row]
  decide
theorem root197Step17 : scan edge (search edge 3) 4 [1223] 1223 root197Tail17 winner =
    scan edge (search edge 3) 4 [1223] 1223 root197Tail18 winner := by
  rw [root197Tail17, scan, if_neg (by decide), root197Candidates17_checked]
  change scan edge (search edge 3) 4 [1223] 1223 root197Tail18
    (if 3 ≤ root197Candidates17.length then search edge 3 [1223, 5939] root197Candidates17 7162 winner else winner) = _
  rw [if_neg (by decide)]
def root197Candidates18 : List Nat := [17033]
set_option maxRecDepth 8192 in
theorem root197Candidates18_checked : candidates edge 4 1223 6113 root197Tail19 winner.bound = root197Candidates18 := by
  unfold candidates
  have row : edge 6113 = (SortedNatTree.contains FastRows.row6113) := by
    change Data.row6113.contains = (SortedNatTree.contains FastRows.row6113)
    exact FastRows.row6113_checked.symm
  rw [row]
  decide
theorem root197Step18 : scan edge (search edge 3) 4 [1223] 1223 root197Tail18 winner =
    scan edge (search edge 3) 4 [1223] 1223 root197Tail19 winner := by
  rw [root197Tail18, scan, if_neg (by decide), root197Candidates18_checked]
  change scan edge (search edge 3) 4 [1223] 1223 root197Tail19
    (if 3 ≤ root197Candidates18.length then search edge 3 [1223, 6113] root197Candidates18 7336 winner else winner) = _
  rw [if_neg (by decide)]
def root197Candidates19 : List Nat := []
set_option maxRecDepth 8192 in
theorem root197Candidates19_checked : candidates edge 4 1223 6131 root197Tail20 winner.bound = root197Candidates19 := by
  unfold candidates
  have row : edge 6131 = (SortedNatTree.contains FastRows.row6131) := by
    change Data.row6131.contains = (SortedNatTree.contains FastRows.row6131)
    exact FastRows.row6131_checked.symm
  rw [row]
  decide
theorem root197Step19 : scan edge (search edge 3) 4 [1223] 1223 root197Tail19 winner =
    scan edge (search edge 3) 4 [1223] 1223 root197Tail20 winner := by
  rw [root197Tail19, scan, if_neg (by decide), root197Candidates19_checked]
  change scan edge (search edge 3) 4 [1223] 1223 root197Tail20
    (if 3 ≤ root197Candidates19.length then search edge 3 [1223, 6131] root197Candidates19 7354 winner else winner) = _
  rw [if_neg (by decide)]
theorem root197Scan20 : scan edge (search edge 3) 4 [1223] 1223 root197Tail20 winner = winner := by rw [root197Tail20, scan, if_pos (by decide)]
theorem root197Scan19 : scan edge (search edge 3) 4 [1223] 1223 root197Tail19 winner = winner :=
  (root197Step19).trans root197Scan20
theorem root197Scan18 : scan edge (search edge 3) 4 [1223] 1223 root197Tail18 winner = winner :=
  (root197Step18).trans root197Scan19
theorem root197Scan17 : scan edge (search edge 3) 4 [1223] 1223 root197Tail17 winner = winner :=
  (root197Step17).trans root197Scan18
theorem root197Scan16 : scan edge (search edge 3) 4 [1223] 1223 root197Tail16 winner = winner :=
  (root197Step16).trans root197Scan17
theorem root197Scan15 : scan edge (search edge 3) 4 [1223] 1223 root197Tail15 winner = winner :=
  (root197Step15).trans root197Scan16
theorem root197Scan14 : scan edge (search edge 3) 4 [1223] 1223 root197Tail14 winner = winner :=
  (root197Step14).trans root197Scan15
theorem root197Scan13 : scan edge (search edge 3) 4 [1223] 1223 root197Tail13 winner = winner :=
  (root197Step13).trans root197Scan14
theorem root197Scan12 : scan edge (search edge 3) 4 [1223] 1223 root197Tail12 winner = winner :=
  (root197Step12).trans root197Scan13
theorem root197Scan11 : scan edge (search edge 3) 4 [1223] 1223 root197Tail11 winner = winner :=
  (root197Step11).trans root197Scan12
theorem root197Scan10 : scan edge (search edge 3) 4 [1223] 1223 root197Tail10 winner = winner :=
  (root197Step10).trans root197Scan11
theorem root197Scan9 : scan edge (search edge 3) 4 [1223] 1223 root197Tail9 winner = winner :=
  (root197Step9).trans root197Scan10
theorem root197Scan8 : scan edge (search edge 3) 4 [1223] 1223 root197Tail8 winner = winner :=
  (root197Step8).trans root197Scan9
theorem root197Scan7 : scan edge (search edge 3) 4 [1223] 1223 root197Tail7 winner = winner :=
  (root197Step7).trans root197Scan8
theorem root197Scan6 : scan edge (search edge 3) 4 [1223] 1223 root197Tail6 winner = winner :=
  (root197Step6).trans root197Scan7
theorem root197Scan5 : scan edge (search edge 3) 4 [1223] 1223 root197Tail5 winner = winner :=
  (root197Step5).trans root197Scan6
theorem root197Scan4 : scan edge (search edge 3) 4 [1223] 1223 root197Tail4 winner = winner :=
  (root197Step4).trans root197Scan5
theorem root197Scan3 : scan edge (search edge 3) 4 [1223] 1223 root197Tail3 winner = winner :=
  (root197Step3).trans root197Scan4
theorem root197Scan2 : scan edge (search edge 3) 4 [1223] 1223 root197Tail2 winner = winner :=
  (root197Step2).trans root197Scan3
theorem root197Scan1 : scan edge (search edge 3) 4 [1223] 1223 root197Tail1 winner = winner :=
  (root197Step1).trans root197Scan2
theorem root197Scan0 : scan edge (search edge 3) 4 [1223] 1223 root197Tail0 winner = winner :=
  (root197Step0).trans root197Scan1
set_option maxRecDepth 4096 in
theorem root197Branch : search edge 4 [1223] root197Candidates 1223 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1223] 1223 root197Tail0 winner = winner
  exact root197Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
