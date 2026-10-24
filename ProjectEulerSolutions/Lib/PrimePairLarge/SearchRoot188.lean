import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root188Candidates : List Nat := Data.row1153
set_option maxRecDepth 8192 in
theorem root188Candidates_checked : candidates edge 5 0 1153 tail189 winner.bound = root188Candidates := by
  apply candidates_eq_row edge 5 0 1153 tail189 root188Candidates winner.bound tail189_ordered
  · decide
  · decide
  · funext q
    rfl
def root188Tail60 : List Nat := []
def root188Tail59 : List Nat := 24439 :: root188Tail60
def root188Tail58 : List Nat := 23899 :: root188Tail59
def root188Tail57 : List Nat := 23017 :: root188Tail58
def root188Tail56 : List Nat := 21649 :: root188Tail57
def root188Tail55 : List Nat := 21247 :: root188Tail56
def root188Tail54 : List Nat := 20743 :: root188Tail55
def root188Tail53 : List Nat := 20323 :: root188Tail54
def root188Tail52 : List Nat := 19423 :: root188Tail53
def root188Tail51 : List Nat := 19231 :: root188Tail52
def root188Tail50 : List Nat := 18979 :: root188Tail51
def root188Tail49 : List Nat := 18919 :: root188Tail50
def root188Tail48 : List Nat := 18793 :: root188Tail49
def root188Tail47 : List Nat := 17317 :: root188Tail48
def root188Tail46 : List Nat := 15331 :: root188Tail47
def root188Tail45 : List Nat := 14503 :: root188Tail46
def root188Tail44 : List Nat := 14029 :: root188Tail45
def root188Tail43 : List Nat := 13291 :: root188Tail44
def root188Tail42 : List Nat := 12907 :: root188Tail43
def root188Tail41 : List Nat := 12511 :: root188Tail42
def root188Tail40 : List Nat := 12391 :: root188Tail41
def root188Tail39 : List Nat := 11503 :: root188Tail40
def root188Tail38 : List Nat := 11083 :: root188Tail39
def root188Tail37 : List Nat := 10939 :: root188Tail38
def root188Tail36 : List Nat := 10687 :: root188Tail37
def root188Tail35 : List Nat := 9907 :: root188Tail36
def root188Tail34 : List Nat := 9613 :: root188Tail35
def root188Tail33 : List Nat := 9439 :: root188Tail34
def root188Tail32 : List Nat := 8731 :: root188Tail33
def root188Tail31 : List Nat := 8017 :: root188Tail32
def root188Tail30 : List Nat := 7993 :: root188Tail31
def root188Tail29 : List Nat := 7741 :: root188Tail30
def root188Tail28 : List Nat := 7681 :: root188Tail29
def root188Tail27 : List Nat := 7573 :: root188Tail28
def root188Tail26 : List Nat := 7507 :: root188Tail27
def root188Tail25 : List Nat := 7243 :: root188Tail26
def root188Tail24 : List Nat := 6691 :: root188Tail25
def root188Tail23 : List Nat := 6673 :: root188Tail24
def root188Tail22 : List Nat := 6361 :: root188Tail23
def root188Tail21 : List Nat := 6067 :: root188Tail22
def root188Tail20 : List Nat := 6037 :: root188Tail21
def root188Tail19 : List Nat := 5437 :: root188Tail20
def root188Tail18 : List Nat := 5179 :: root188Tail19
def root188Tail17 : List Nat := 4951 :: root188Tail18
def root188Tail16 : List Nat := 4597 :: root188Tail17
def root188Tail15 : List Nat := 4423 :: root188Tail16
def root188Tail14 : List Nat := 4093 :: root188Tail15
def root188Tail13 : List Nat := 3943 :: root188Tail14
def root188Tail12 : List Nat := 3919 :: root188Tail13
def root188Tail11 : List Nat := 3853 :: root188Tail12
def root188Tail10 : List Nat := 3673 :: root188Tail11
def root188Tail9 : List Nat := 3253 :: root188Tail10
def root188Tail8 : List Nat := 3163 :: root188Tail9
def root188Tail7 : List Nat := 2803 :: root188Tail8
def root188Tail6 : List Nat := 2473 :: root188Tail7
def root188Tail5 : List Nat := 2383 :: root188Tail6
def root188Tail4 : List Nat := 1801 :: root188Tail5
def root188Tail3 : List Nat := 1759 :: root188Tail4
def root188Tail2 : List Nat := 1699 :: root188Tail3
def root188Tail1 : List Nat := 1543 :: root188Tail2
def root188Tail0 : List Nat := 1231 :: root188Tail1
def root188Candidates0 : List Nat := [4597, 6037, 12907]
set_option maxRecDepth 8192 in
theorem root188Candidates0_checked : candidates edge 4 1153 1231 root188Tail1 winner.bound = root188Candidates0 := by
  unfold candidates
  have row : edge 1231 = (SortedNatTree.contains FastRows.row1231) := by
    change Data.row1231.contains = (SortedNatTree.contains FastRows.row1231)
    exact FastRows.row1231_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root188Child0 : search edge 3 [1153, 1231] root188Candidates0 2384 winner = winner := by decide
theorem root188Step0 : scan edge (search edge 3) 4 [1153] 1153 root188Tail0 winner =
    scan edge (search edge 3) 4 [1153] 1153 root188Tail1 winner := by
  rw [root188Tail0, scan, if_neg (by decide), root188Candidates0_checked]
  change scan edge (search edge 3) 4 [1153] 1153 root188Tail1
    (if 3 ≤ root188Candidates0.length then search edge 3 [1153, 1231] root188Candidates0 2384 winner else winner) = _
  rw [if_pos (by decide), root188Child0]
def root188Candidates1 : List Nat := [4093, 4423]
set_option maxRecDepth 8192 in
theorem root188Candidates1_checked : candidates edge 4 1153 1543 root188Tail2 winner.bound = root188Candidates1 := by
  unfold candidates
  have row : edge 1543 = (SortedNatTree.contains FastRows.row1543) := by
    change Data.row1543.contains = (SortedNatTree.contains FastRows.row1543)
    exact FastRows.row1543_checked.symm
  rw [row]
  decide
theorem root188Step1 : scan edge (search edge 3) 4 [1153] 1153 root188Tail1 winner =
    scan edge (search edge 3) 4 [1153] 1153 root188Tail2 winner := by
  rw [root188Tail1, scan, if_neg (by decide), root188Candidates1_checked]
  change scan edge (search edge 3) 4 [1153] 1153 root188Tail2
    (if 3 ≤ root188Candidates1.length then search edge 3 [1153, 1543] root188Candidates1 2696 winner else winner) = _
  rw [if_neg (by decide)]
def root188Candidates2 : List Nat := [1759, 3853, 6691, 7507]
set_option maxRecDepth 8192 in
theorem root188Candidates2_checked : candidates edge 4 1153 1699 root188Tail3 winner.bound = root188Candidates2 := by
  unfold candidates
  have row : edge 1699 = (SortedNatTree.contains FastRows.row1699) := by
    change Data.row1699.contains = (SortedNatTree.contains FastRows.row1699)
    exact FastRows.row1699_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root188Child2 : search edge 3 [1153, 1699] root188Candidates2 2852 winner = winner := by decide
theorem root188Step2 : scan edge (search edge 3) 4 [1153] 1153 root188Tail2 winner =
    scan edge (search edge 3) 4 [1153] 1153 root188Tail3 winner := by
  rw [root188Tail2, scan, if_neg (by decide), root188Candidates2_checked]
  change scan edge (search edge 3) 4 [1153] 1153 root188Tail3
    (if 3 ≤ root188Candidates2.length then search edge 3 [1153, 1699] root188Candidates2 2852 winner else winner) = _
  rw [if_pos (by decide), root188Child2]
def root188Candidates3 : List Nat := [6361, 9613, 10687, 15331]
set_option maxRecDepth 8192 in
theorem root188Candidates3_checked : candidates edge 4 1153 1759 root188Tail4 winner.bound = root188Candidates3 := by
  unfold candidates
  have row : edge 1759 = (SortedNatTree.contains FastRows.row1759) := by
    change Data.row1759.contains = (SortedNatTree.contains FastRows.row1759)
    exact FastRows.row1759_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root188Child3 : search edge 3 [1153, 1759] root188Candidates3 2912 winner = winner := by decide
theorem root188Step3 : scan edge (search edge 3) 4 [1153] 1153 root188Tail3 winner =
    scan edge (search edge 3) 4 [1153] 1153 root188Tail4 winner := by
  rw [root188Tail3, scan, if_neg (by decide), root188Candidates3_checked]
  change scan edge (search edge 3) 4 [1153] 1153 root188Tail4
    (if 3 ≤ root188Candidates3.length then search edge 3 [1153, 1759] root188Candidates3 2912 winner else winner) = _
  rw [if_pos (by decide), root188Child3]
def root188Candidates4 : List Nat := [3253, 18919]
set_option maxRecDepth 8192 in
theorem root188Candidates4_checked : candidates edge 4 1153 1801 root188Tail5 winner.bound = root188Candidates4 := by
  unfold candidates
  have row : edge 1801 = (SortedNatTree.contains FastRows.row1801) := by
    change Data.row1801.contains = (SortedNatTree.contains FastRows.row1801)
    exact FastRows.row1801_checked.symm
  rw [row]
  decide
theorem root188Step4 : scan edge (search edge 3) 4 [1153] 1153 root188Tail4 winner =
    scan edge (search edge 3) 4 [1153] 1153 root188Tail5 winner := by
  rw [root188Tail4, scan, if_neg (by decide), root188Candidates4_checked]
  change scan edge (search edge 3) 4 [1153] 1153 root188Tail5
    (if 3 ≤ root188Candidates4.length then search edge 3 [1153, 1801] root188Candidates4 2954 winner else winner) = _
  rw [if_neg (by decide)]
def root188Candidates5 : List Nat := [7573, 21649]
set_option maxRecDepth 8192 in
theorem root188Candidates5_checked : candidates edge 4 1153 2383 root188Tail6 winner.bound = root188Candidates5 := by
  unfold candidates
  have row : edge 2383 = (SortedNatTree.contains FastRows.row2383) := by
    change Data.row2383.contains = (SortedNatTree.contains FastRows.row2383)
    exact FastRows.row2383_checked.symm
  rw [row]
  decide
theorem root188Step5 : scan edge (search edge 3) 4 [1153] 1153 root188Tail5 winner =
    scan edge (search edge 3) 4 [1153] 1153 root188Tail6 winner := by
  rw [root188Tail5, scan, if_neg (by decide), root188Candidates5_checked]
  change scan edge (search edge 3) 4 [1153] 1153 root188Tail6
    (if 3 ≤ root188Candidates5.length then search edge 3 [1153, 2383] root188Candidates5 3536 winner else winner) = _
  rw [if_neg (by decide)]
def root188Candidates6 : List Nat := [5437]
set_option maxRecDepth 8192 in
theorem root188Candidates6_checked : candidates edge 4 1153 2473 root188Tail7 winner.bound = root188Candidates6 := by
  unfold candidates
  have row : edge 2473 = (SortedNatTree.contains FastRows.row2473) := by
    change Data.row2473.contains = (SortedNatTree.contains FastRows.row2473)
    exact FastRows.row2473_checked.symm
  rw [row]
  decide
theorem root188Step6 : scan edge (search edge 3) 4 [1153] 1153 root188Tail6 winner =
    scan edge (search edge 3) 4 [1153] 1153 root188Tail7 winner := by
  rw [root188Tail6, scan, if_neg (by decide), root188Candidates6_checked]
  change scan edge (search edge 3) 4 [1153] 1153 root188Tail7
    (if 3 ≤ root188Candidates6.length then search edge 3 [1153, 2473] root188Candidates6 3626 winner else winner) = _
  rw [if_neg (by decide)]
def root188Candidates7 : List Nat := [4423, 6361, 7993, 13291]
set_option maxRecDepth 8192 in
theorem root188Candidates7_checked : candidates edge 4 1153 2803 root188Tail8 winner.bound = root188Candidates7 := by
  unfold candidates
  have row : edge 2803 = (SortedNatTree.contains FastRows.row2803) := by
    change Data.row2803.contains = (SortedNatTree.contains FastRows.row2803)
    exact FastRows.row2803_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root188Child7 : search edge 3 [1153, 2803] root188Candidates7 3956 winner = winner := by decide
theorem root188Step7 : scan edge (search edge 3) 4 [1153] 1153 root188Tail7 winner =
    scan edge (search edge 3) 4 [1153] 1153 root188Tail8 winner := by
  rw [root188Tail7, scan, if_neg (by decide), root188Candidates7_checked]
  change scan edge (search edge 3) 4 [1153] 1153 root188Tail8
    (if 3 ≤ root188Candidates7.length then search edge 3 [1153, 2803] root188Candidates7 3956 winner else winner) = _
  rw [if_pos (by decide), root188Child7]
def root188Candidates8 : List Nat := [6037]
set_option maxRecDepth 8192 in
theorem root188Candidates8_checked : candidates edge 4 1153 3163 root188Tail9 winner.bound = root188Candidates8 := by
  unfold candidates
  have row : edge 3163 = (SortedNatTree.contains FastRows.row3163) := by
    change Data.row3163.contains = (SortedNatTree.contains FastRows.row3163)
    exact FastRows.row3163_checked.symm
  rw [row]
  decide
theorem root188Step8 : scan edge (search edge 3) 4 [1153] 1153 root188Tail8 winner =
    scan edge (search edge 3) 4 [1153] 1153 root188Tail9 winner := by
  rw [root188Tail8, scan, if_neg (by decide), root188Candidates8_checked]
  change scan edge (search edge 3) 4 [1153] 1153 root188Tail9
    (if 3 ≤ root188Candidates8.length then search edge 3 [1153, 3163] root188Candidates8 4316 winner else winner) = _
  rw [if_neg (by decide)]
def root188Candidates9 : List Nat := [6067, 14503]
set_option maxRecDepth 8192 in
theorem root188Candidates9_checked : candidates edge 4 1153 3253 root188Tail10 winner.bound = root188Candidates9 := by
  unfold candidates
  have row : edge 3253 = (SortedNatTree.contains FastRows.row3253) := by
    change Data.row3253.contains = (SortedNatTree.contains FastRows.row3253)
    exact FastRows.row3253_checked.symm
  rw [row]
  decide
theorem root188Step9 : scan edge (search edge 3) 4 [1153] 1153 root188Tail9 winner =
    scan edge (search edge 3) 4 [1153] 1153 root188Tail10 winner := by
  rw [root188Tail9, scan, if_neg (by decide), root188Candidates9_checked]
  change scan edge (search edge 3) 4 [1153] 1153 root188Tail10
    (if 3 ≤ root188Candidates9.length then search edge 3 [1153, 3253] root188Candidates9 4406 winner else winner) = _
  rw [if_neg (by decide)]
def root188Candidates10 : List Nat := [6691]
set_option maxRecDepth 8192 in
theorem root188Candidates10_checked : candidates edge 4 1153 3673 root188Tail11 winner.bound = root188Candidates10 := by
  unfold candidates
  have row : edge 3673 = (SortedNatTree.contains FastRows.row3673) := by
    change Data.row3673.contains = (SortedNatTree.contains FastRows.row3673)
    exact FastRows.row3673_checked.symm
  rw [row]
  decide
theorem root188Step10 : scan edge (search edge 3) 4 [1153] 1153 root188Tail10 winner =
    scan edge (search edge 3) 4 [1153] 1153 root188Tail11 winner := by
  rw [root188Tail10, scan, if_neg (by decide), root188Candidates10_checked]
  change scan edge (search edge 3) 4 [1153] 1153 root188Tail11
    (if 3 ≤ root188Candidates10.length then search edge 3 [1153, 3673] root188Candidates10 4826 winner else winner) = _
  rw [if_neg (by decide)]
def root188Candidates11 : List Nat := [6361, 7243, 18979]
set_option maxRecDepth 8192 in
theorem root188Candidates11_checked : candidates edge 4 1153 3853 root188Tail12 winner.bound = root188Candidates11 := by
  unfold candidates
  have row : edge 3853 = (SortedNatTree.contains FastRows.row3853) := by
    change Data.row3853.contains = (SortedNatTree.contains FastRows.row3853)
    exact FastRows.row3853_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root188Child11 : search edge 3 [1153, 3853] root188Candidates11 5006 winner = winner := by decide
theorem root188Step11 : scan edge (search edge 3) 4 [1153] 1153 root188Tail11 winner =
    scan edge (search edge 3) 4 [1153] 1153 root188Tail12 winner := by
  rw [root188Tail11, scan, if_neg (by decide), root188Candidates11_checked]
  change scan edge (search edge 3) 4 [1153] 1153 root188Tail12
    (if 3 ≤ root188Candidates11.length then search edge 3 [1153, 3853] root188Candidates11 5006 winner else winner) = _
  rw [if_pos (by decide), root188Child11]
def root188Candidates12 : List Nat := [5179]
set_option maxRecDepth 8192 in
theorem root188Candidates12_checked : candidates edge 4 1153 3919 root188Tail13 winner.bound = root188Candidates12 := by
  unfold candidates
  have row : edge 3919 = (SortedNatTree.contains FastRows.row3919) := by
    change Data.row3919.contains = (SortedNatTree.contains FastRows.row3919)
    exact FastRows.row3919_checked.symm
  rw [row]
  decide
theorem root188Step12 : scan edge (search edge 3) 4 [1153] 1153 root188Tail12 winner =
    scan edge (search edge 3) 4 [1153] 1153 root188Tail13 winner := by
  rw [root188Tail12, scan, if_neg (by decide), root188Candidates12_checked]
  change scan edge (search edge 3) 4 [1153] 1153 root188Tail13
    (if 3 ≤ root188Candidates12.length then search edge 3 [1153, 3919] root188Candidates12 5072 winner else winner) = _
  rw [if_neg (by decide)]
def root188Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root188Candidates13_checked : candidates edge 4 1153 3943 root188Tail14 winner.bound = root188Candidates13 := by
  unfold candidates
  have row : edge 3943 = (SortedNatTree.contains FastRows.row3943) := by
    change Data.row3943.contains = (SortedNatTree.contains FastRows.row3943)
    exact FastRows.row3943_checked.symm
  rw [row]
  decide
theorem root188Step13 : scan edge (search edge 3) 4 [1153] 1153 root188Tail13 winner =
    scan edge (search edge 3) 4 [1153] 1153 root188Tail14 winner := by
  rw [root188Tail13, scan, if_neg (by decide), root188Candidates13_checked]
  change scan edge (search edge 3) 4 [1153] 1153 root188Tail14
    (if 3 ≤ root188Candidates13.length then search edge 3 [1153, 3943] root188Candidates13 5096 winner else winner) = _
  rw [if_neg (by decide)]
def root188Candidates14 : List Nat := [11503, 12391]
set_option maxRecDepth 8192 in
theorem root188Candidates14_checked : candidates edge 4 1153 4093 root188Tail15 winner.bound = root188Candidates14 := by
  unfold candidates
  have row : edge 4093 = (SortedNatTree.contains FastRows.row4093) := by
    change Data.row4093.contains = (SortedNatTree.contains FastRows.row4093)
    exact FastRows.row4093_checked.symm
  rw [row]
  decide
theorem root188Step14 : scan edge (search edge 3) 4 [1153] 1153 root188Tail14 winner =
    scan edge (search edge 3) 4 [1153] 1153 root188Tail15 winner := by
  rw [root188Tail14, scan, if_neg (by decide), root188Candidates14_checked]
  change scan edge (search edge 3) 4 [1153] 1153 root188Tail15
    (if 3 ≤ root188Candidates14.length then search edge 3 [1153, 4093] root188Candidates14 5246 winner else winner) = _
  rw [if_neg (by decide)]
def root188Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root188Candidates15_checked : candidates edge 4 1153 4423 root188Tail16 winner.bound = root188Candidates15 := by
  unfold candidates
  have row : edge 4423 = (SortedNatTree.contains FastRows.row4423) := by
    change Data.row4423.contains = (SortedNatTree.contains FastRows.row4423)
    exact FastRows.row4423_checked.symm
  rw [row]
  decide
theorem root188Step15 : scan edge (search edge 3) 4 [1153] 1153 root188Tail15 winner =
    scan edge (search edge 3) 4 [1153] 1153 root188Tail16 winner := by
  rw [root188Tail15, scan, if_neg (by decide), root188Candidates15_checked]
  change scan edge (search edge 3) 4 [1153] 1153 root188Tail16
    (if 3 ≤ root188Candidates15.length then search edge 3 [1153, 4423] root188Candidates15 5576 winner else winner) = _
  rw [if_neg (by decide)]
def root188Candidates16 : List Nat := [7993, 9907, 15331]
set_option maxRecDepth 8192 in
theorem root188Candidates16_checked : candidates edge 4 1153 4597 root188Tail17 winner.bound = root188Candidates16 := by
  unfold candidates
  have row : edge 4597 = (SortedNatTree.contains FastRows.row4597) := by
    change Data.row4597.contains = (SortedNatTree.contains FastRows.row4597)
    exact FastRows.row4597_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root188Child16 : search edge 3 [1153, 4597] root188Candidates16 5750 winner = winner := by decide
theorem root188Step16 : scan edge (search edge 3) 4 [1153] 1153 root188Tail16 winner =
    scan edge (search edge 3) 4 [1153] 1153 root188Tail17 winner := by
  rw [root188Tail16, scan, if_neg (by decide), root188Candidates16_checked]
  change scan edge (search edge 3) 4 [1153] 1153 root188Tail17
    (if 3 ≤ root188Candidates16.length then search edge 3 [1153, 4597] root188Candidates16 5750 winner else winner) = _
  rw [if_pos (by decide), root188Child16]
def root188Candidates17 : List Nat := [14029]
set_option maxRecDepth 8192 in
theorem root188Candidates17_checked : candidates edge 4 1153 4951 root188Tail18 winner.bound = root188Candidates17 := by
  unfold candidates
  have row : edge 4951 = (SortedNatTree.contains FastRows.row4951) := by
    change Data.row4951.contains = (SortedNatTree.contains FastRows.row4951)
    exact FastRows.row4951_checked.symm
  rw [row]
  decide
theorem root188Step17 : scan edge (search edge 3) 4 [1153] 1153 root188Tail17 winner =
    scan edge (search edge 3) 4 [1153] 1153 root188Tail18 winner := by
  rw [root188Tail17, scan, if_neg (by decide), root188Candidates17_checked]
  change scan edge (search edge 3) 4 [1153] 1153 root188Tail18
    (if 3 ≤ root188Candidates17.length then search edge 3 [1153, 4951] root188Candidates17 6104 winner else winner) = _
  rw [if_neg (by decide)]
def root188Candidates18 : List Nat := [9439, 14029, 17317]
set_option maxRecDepth 8192 in
theorem root188Candidates18_checked : candidates edge 4 1153 5179 root188Tail19 winner.bound = root188Candidates18 := by
  unfold candidates
  have row : edge 5179 = (SortedNatTree.contains FastRows.row5179) := by
    change Data.row5179.contains = (SortedNatTree.contains FastRows.row5179)
    exact FastRows.row5179_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root188Child18 : search edge 3 [1153, 5179] root188Candidates18 6332 winner = winner := by decide
theorem root188Step18 : scan edge (search edge 3) 4 [1153] 1153 root188Tail18 winner =
    scan edge (search edge 3) 4 [1153] 1153 root188Tail19 winner := by
  rw [root188Tail18, scan, if_neg (by decide), root188Candidates18_checked]
  change scan edge (search edge 3) 4 [1153] 1153 root188Tail19
    (if 3 ≤ root188Candidates18.length then search edge 3 [1153, 5179] root188Candidates18 6332 winner else winner) = _
  rw [if_pos (by decide), root188Child18]
def root188Candidates19 : List Nat := [8017]
set_option maxRecDepth 8192 in
theorem root188Candidates19_checked : candidates edge 4 1153 5437 root188Tail20 winner.bound = root188Candidates19 := by
  unfold candidates
  have row : edge 5437 = (SortedNatTree.contains FastRows.row5437) := by
    change Data.row5437.contains = (SortedNatTree.contains FastRows.row5437)
    exact FastRows.row5437_checked.symm
  rw [row]
  decide
theorem root188Step19 : scan edge (search edge 3) 4 [1153] 1153 root188Tail19 winner =
    scan edge (search edge 3) 4 [1153] 1153 root188Tail20 winner := by
  rw [root188Tail19, scan, if_neg (by decide), root188Candidates19_checked]
  change scan edge (search edge 3) 4 [1153] 1153 root188Tail20
    (if 3 ≤ root188Candidates19.length then search edge 3 [1153, 5437] root188Candidates19 6590 winner else winner) = _
  rw [if_neg (by decide)]
def root188Candidates20 : List Nat := [13291]
set_option maxRecDepth 8192 in
theorem root188Candidates20_checked : candidates edge 4 1153 6037 root188Tail21 winner.bound = root188Candidates20 := by
  unfold candidates
  have row : edge 6037 = (SortedNatTree.contains FastRows.row6037) := by
    change Data.row6037.contains = (SortedNatTree.contains FastRows.row6037)
    exact FastRows.row6037_checked.symm
  rw [row]
  decide
theorem root188Step20 : scan edge (search edge 3) 4 [1153] 1153 root188Tail20 winner =
    scan edge (search edge 3) 4 [1153] 1153 root188Tail21 winner := by
  rw [root188Tail20, scan, if_neg (by decide), root188Candidates20_checked]
  change scan edge (search edge 3) 4 [1153] 1153 root188Tail21
    (if 3 ≤ root188Candidates20.length then search edge 3 [1153, 6037] root188Candidates20 7190 winner else winner) = _
  rw [if_neg (by decide)]
def root188Candidates21 : List Nat := [7993]
set_option maxRecDepth 8192 in
theorem root188Candidates21_checked : candidates edge 4 1153 6067 root188Tail22 winner.bound = root188Candidates21 := by
  unfold candidates
  have row : edge 6067 = (SortedNatTree.contains FastRows.row6067) := by
    change Data.row6067.contains = (SortedNatTree.contains FastRows.row6067)
    exact FastRows.row6067_checked.symm
  rw [row]
  decide
theorem root188Step21 : scan edge (search edge 3) 4 [1153] 1153 root188Tail21 winner =
    scan edge (search edge 3) 4 [1153] 1153 root188Tail22 winner := by
  rw [root188Tail21, scan, if_neg (by decide), root188Candidates21_checked]
  change scan edge (search edge 3) 4 [1153] 1153 root188Tail22
    (if 3 ≤ root188Candidates21.length then search edge 3 [1153, 6067] root188Candidates21 7220 winner else winner) = _
  rw [if_neg (by decide)]
theorem root188Scan22 : scan edge (search edge 3) 4 [1153] 1153 root188Tail22 winner = winner := by rw [root188Tail22, scan, if_pos (by decide)]
theorem root188Scan21 : scan edge (search edge 3) 4 [1153] 1153 root188Tail21 winner = winner :=
  (root188Step21).trans root188Scan22
theorem root188Scan20 : scan edge (search edge 3) 4 [1153] 1153 root188Tail20 winner = winner :=
  (root188Step20).trans root188Scan21
theorem root188Scan19 : scan edge (search edge 3) 4 [1153] 1153 root188Tail19 winner = winner :=
  (root188Step19).trans root188Scan20
theorem root188Scan18 : scan edge (search edge 3) 4 [1153] 1153 root188Tail18 winner = winner :=
  (root188Step18).trans root188Scan19
theorem root188Scan17 : scan edge (search edge 3) 4 [1153] 1153 root188Tail17 winner = winner :=
  (root188Step17).trans root188Scan18
theorem root188Scan16 : scan edge (search edge 3) 4 [1153] 1153 root188Tail16 winner = winner :=
  (root188Step16).trans root188Scan17
theorem root188Scan15 : scan edge (search edge 3) 4 [1153] 1153 root188Tail15 winner = winner :=
  (root188Step15).trans root188Scan16
theorem root188Scan14 : scan edge (search edge 3) 4 [1153] 1153 root188Tail14 winner = winner :=
  (root188Step14).trans root188Scan15
theorem root188Scan13 : scan edge (search edge 3) 4 [1153] 1153 root188Tail13 winner = winner :=
  (root188Step13).trans root188Scan14
theorem root188Scan12 : scan edge (search edge 3) 4 [1153] 1153 root188Tail12 winner = winner :=
  (root188Step12).trans root188Scan13
theorem root188Scan11 : scan edge (search edge 3) 4 [1153] 1153 root188Tail11 winner = winner :=
  (root188Step11).trans root188Scan12
theorem root188Scan10 : scan edge (search edge 3) 4 [1153] 1153 root188Tail10 winner = winner :=
  (root188Step10).trans root188Scan11
theorem root188Scan9 : scan edge (search edge 3) 4 [1153] 1153 root188Tail9 winner = winner :=
  (root188Step9).trans root188Scan10
theorem root188Scan8 : scan edge (search edge 3) 4 [1153] 1153 root188Tail8 winner = winner :=
  (root188Step8).trans root188Scan9
theorem root188Scan7 : scan edge (search edge 3) 4 [1153] 1153 root188Tail7 winner = winner :=
  (root188Step7).trans root188Scan8
theorem root188Scan6 : scan edge (search edge 3) 4 [1153] 1153 root188Tail6 winner = winner :=
  (root188Step6).trans root188Scan7
theorem root188Scan5 : scan edge (search edge 3) 4 [1153] 1153 root188Tail5 winner = winner :=
  (root188Step5).trans root188Scan6
theorem root188Scan4 : scan edge (search edge 3) 4 [1153] 1153 root188Tail4 winner = winner :=
  (root188Step4).trans root188Scan5
theorem root188Scan3 : scan edge (search edge 3) 4 [1153] 1153 root188Tail3 winner = winner :=
  (root188Step3).trans root188Scan4
theorem root188Scan2 : scan edge (search edge 3) 4 [1153] 1153 root188Tail2 winner = winner :=
  (root188Step2).trans root188Scan3
theorem root188Scan1 : scan edge (search edge 3) 4 [1153] 1153 root188Tail1 winner = winner :=
  (root188Step1).trans root188Scan2
theorem root188Scan0 : scan edge (search edge 3) 4 [1153] 1153 root188Tail0 winner = winner :=
  (root188Step0).trans root188Scan1
set_option maxRecDepth 4096 in
theorem root188Branch : search edge 4 [1153] root188Candidates 1153 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1153] 1153 root188Tail0 winner = winner
  exact root188Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
