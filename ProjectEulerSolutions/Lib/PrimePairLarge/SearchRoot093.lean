import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root93Candidates : List Nat := Data.row503
set_option maxRecDepth 8192 in
theorem root93Candidates_checked : candidates edge 5 0 503 tail94 winner.bound = root93Candidates := by
  apply candidates_eq_row edge 5 0 503 tail94 root93Candidates winner.bound tail94_ordered
  · decide
  · decide
  · funext q
    rfl
def root93Tail64 : List Nat := []
def root93Tail63 : List Nat := 25349 :: root93Tail64
def root93Tail62 : List Nat := 24413 :: root93Tail63
def root93Tail61 : List Nat := 24203 :: root93Tail62
def root93Tail60 : List Nat := 23993 :: root93Tail61
def root93Tail59 : List Nat := 23627 :: root93Tail60
def root93Tail58 : List Nat := 23087 :: root93Tail59
def root93Tail57 : List Nat := 22691 :: root93Tail58
def root93Tail56 : List Nat := 21863 :: root93Tail57
def root93Tail55 : List Nat := 21647 :: root93Tail56
def root93Tail54 : List Nat := 21617 :: root93Tail55
def root93Tail53 : List Nat := 21521 :: root93Tail54
def root93Tail52 : List Nat := 20483 :: root93Tail53
def root93Tail51 : List Nat := 20129 :: root93Tail52
def root93Tail50 : List Nat := 19793 :: root93Tail51
def root93Tail49 : List Nat := 19463 :: root93Tail50
def root93Tail48 : List Nat := 19433 :: root93Tail49
def root93Tail47 : List Nat := 19121 :: root93Tail48
def root93Tail46 : List Nat := 19079 :: root93Tail47
def root93Tail45 : List Nat := 18077 :: root93Tail46
def root93Tail44 : List Nat := 16691 :: root93Tail45
def root93Tail43 : List Nat := 16229 :: root93Tail44
def root93Tail42 : List Nat := 15773 :: root93Tail43
def root93Tail41 : List Nat := 15737 :: root93Tail42
def root93Tail40 : List Nat := 15269 :: root93Tail41
def root93Tail39 : List Nat := 15149 :: root93Tail40
def root93Tail38 : List Nat := 15101 :: root93Tail39
def root93Tail37 : List Nat := 14753 :: root93Tail38
def root93Tail36 : List Nat := 13883 :: root93Tail37
def root93Tail35 : List Nat := 13553 :: root93Tail36
def root93Tail34 : List Nat := 12959 :: root93Tail35
def root93Tail33 : List Nat := 12917 :: root93Tail34
def root93Tail32 : List Nat := 12539 :: root93Tail33
def root93Tail31 : List Nat := 11903 :: root93Tail32
def root93Tail30 : List Nat := 11621 :: root93Tail31
def root93Tail29 : List Nat := 11273 :: root93Tail30
def root93Tail28 : List Nat := 11117 :: root93Tail29
def root93Tail27 : List Nat := 10847 :: root93Tail28
def root93Tail26 : List Nat := 10607 :: root93Tail27
def root93Tail25 : List Nat := 10391 :: root93Tail26
def root93Tail24 : List Nat := 9623 :: root93Tail25
def root93Tail23 : List Nat := 9311 :: root93Tail24
def root93Tail22 : List Nat := 9059 :: root93Tail23
def root93Tail21 : List Nat := 8963 :: root93Tail22
def root93Tail20 : List Nat := 7817 :: root93Tail21
def root93Tail19 : List Nat := 7673 :: root93Tail20
def root93Tail18 : List Nat := 5903 :: root93Tail19
def root93Tail17 : List Nat := 5843 :: root93Tail18
def root93Tail16 : List Nat := 5507 :: root93Tail17
def root93Tail15 : List Nat := 5351 :: root93Tail16
def root93Tail14 : List Nat := 5309 :: root93Tail15
def root93Tail13 : List Nat := 5021 :: root93Tail14
def root93Tail12 : List Nat := 5003 :: root93Tail13
def root93Tail11 : List Nat := 4229 :: root93Tail12
def root93Tail10 : List Nat := 3593 :: root93Tail11
def root93Tail9 : List Nat := 3299 :: root93Tail10
def root93Tail8 : List Nat := 2711 :: root93Tail9
def root93Tail7 : List Nat := 2411 :: root93Tail8
def root93Tail6 : List Nat := 1997 :: root93Tail7
def root93Tail5 : List Nat := 1601 :: root93Tail6
def root93Tail4 : List Nat := 1193 :: root93Tail5
def root93Tail3 : List Nat := 911 :: root93Tail4
def root93Tail2 : List Nat := 653 :: root93Tail3
def root93Tail1 : List Nat := 647 :: root93Tail2
def root93Tail0 : List Nat := 563 :: root93Tail1
def root93Candidates0 : List Nat := [1601, 15269]
set_option maxRecDepth 8192 in
theorem root93Candidates0_checked : candidates edge 4 503 563 root93Tail1 winner.bound = root93Candidates0 := by
  unfold candidates
  have row : edge 563 = (SortedNatTree.contains FastRows.row563) := by
    change Data.row563.contains = (SortedNatTree.contains FastRows.row563)
    exact FastRows.row563_checked.symm
  rw [row]
  decide
theorem root93Step0 : scan edge (search edge 3) 4 [503] 503 root93Tail0 winner =
    scan edge (search edge 3) 4 [503] 503 root93Tail1 winner := by
  rw [root93Tail0, scan, if_neg (by decide), root93Candidates0_checked]
  change scan edge (search edge 3) 4 [503] 503 root93Tail1
    (if 3 ≤ root93Candidates0.length then search edge 3 [503, 563] root93Candidates0 1066 winner else winner) = _
  rw [if_neg (by decide)]
def root93Candidates1 : List Nat := [15149]
set_option maxRecDepth 8192 in
theorem root93Candidates1_checked : candidates edge 4 503 647 root93Tail2 winner.bound = root93Candidates1 := by
  unfold candidates
  have row : edge 647 = (SortedNatTree.contains FastRows.row647) := by
    change Data.row647.contains = (SortedNatTree.contains FastRows.row647)
    exact FastRows.row647_checked.symm
  rw [row]
  decide
theorem root93Step1 : scan edge (search edge 3) 4 [503] 503 root93Tail1 winner =
    scan edge (search edge 3) 4 [503] 503 root93Tail2 winner := by
  rw [root93Tail1, scan, if_neg (by decide), root93Candidates1_checked]
  change scan edge (search edge 3) 4 [503] 503 root93Tail2
    (if 3 ≤ root93Candidates1.length then search edge 3 [503, 647] root93Candidates1 1150 winner else winner) = _
  rw [if_neg (by decide)]
def root93Candidates2 : List Nat := [8963, 11273, 15149, 19433, 21617, 21647]
set_option maxRecDepth 8192 in
theorem root93Candidates2_checked : candidates edge 4 503 653 root93Tail3 winner.bound = root93Candidates2 := by
  unfold candidates
  have row : edge 653 = (SortedNatTree.contains FastRows.row653) := by
    change Data.row653.contains = (SortedNatTree.contains FastRows.row653)
    exact FastRows.row653_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root93Child2 : search edge 3 [503, 653] root93Candidates2 1156 winner = winner := by decide
theorem root93Step2 : scan edge (search edge 3) 4 [503] 503 root93Tail2 winner =
    scan edge (search edge 3) 4 [503] 503 root93Tail3 winner := by
  rw [root93Tail2, scan, if_neg (by decide), root93Candidates2_checked]
  change scan edge (search edge 3) 4 [503] 503 root93Tail3
    (if 3 ≤ root93Candidates2.length then search edge 3 [503, 653] root93Candidates2 1156 winner else winner) = _
  rw [if_pos (by decide), root93Child2]
def root93Candidates3 : List Nat := [11903, 21647]
set_option maxRecDepth 8192 in
theorem root93Candidates3_checked : candidates edge 4 503 911 root93Tail4 winner.bound = root93Candidates3 := by
  unfold candidates
  have row : edge 911 = (SortedNatTree.contains FastRows.row911) := by
    change Data.row911.contains = (SortedNatTree.contains FastRows.row911)
    exact FastRows.row911_checked.symm
  rw [row]
  decide
theorem root93Step3 : scan edge (search edge 3) 4 [503] 503 root93Tail3 winner =
    scan edge (search edge 3) 4 [503] 503 root93Tail4 winner := by
  rw [root93Tail3, scan, if_neg (by decide), root93Candidates3_checked]
  change scan edge (search edge 3) 4 [503] 503 root93Tail4
    (if 3 ≤ root93Candidates3.length then search edge 3 [503, 911] root93Candidates3 1414 winner else winner) = _
  rw [if_neg (by decide)]
def root93Candidates4 : List Nat := [5507, 15101]
set_option maxRecDepth 8192 in
theorem root93Candidates4_checked : candidates edge 4 503 1193 root93Tail5 winner.bound = root93Candidates4 := by
  unfold candidates
  have row : edge 1193 = (SortedNatTree.contains FastRows.row1193) := by
    change Data.row1193.contains = (SortedNatTree.contains FastRows.row1193)
    exact FastRows.row1193_checked.symm
  rw [row]
  decide
theorem root93Step4 : scan edge (search edge 3) 4 [503] 503 root93Tail4 winner =
    scan edge (search edge 3) 4 [503] 503 root93Tail5 winner := by
  rw [root93Tail4, scan, if_neg (by decide), root93Candidates4_checked]
  change scan edge (search edge 3) 4 [503] 503 root93Tail5
    (if 3 ≤ root93Candidates4.length then search edge 3 [503, 1193] root93Candidates4 1696 winner else winner) = _
  rw [if_neg (by decide)]
def root93Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root93Candidates5_checked : candidates edge 4 503 1601 root93Tail6 winner.bound = root93Candidates5 := by
  unfold candidates
  have row : edge 1601 = (SortedNatTree.contains FastRows.row1601) := by
    change Data.row1601.contains = (SortedNatTree.contains FastRows.row1601)
    exact FastRows.row1601_checked.symm
  rw [row]
  decide
theorem root93Step5 : scan edge (search edge 3) 4 [503] 503 root93Tail5 winner =
    scan edge (search edge 3) 4 [503] 503 root93Tail6 winner := by
  rw [root93Tail5, scan, if_neg (by decide), root93Candidates5_checked]
  change scan edge (search edge 3) 4 [503] 503 root93Tail6
    (if 3 ≤ root93Candidates5.length then search edge 3 [503, 1601] root93Candidates5 2104 winner else winner) = _
  rw [if_neg (by decide)]
def root93Candidates6 : List Nat := [11117, 15773]
set_option maxRecDepth 8192 in
theorem root93Candidates6_checked : candidates edge 4 503 1997 root93Tail7 winner.bound = root93Candidates6 := by
  unfold candidates
  have row : edge 1997 = (SortedNatTree.contains FastRows.row1997) := by
    change Data.row1997.contains = (SortedNatTree.contains FastRows.row1997)
    exact FastRows.row1997_checked.symm
  rw [row]
  decide
theorem root93Step6 : scan edge (search edge 3) 4 [503] 503 root93Tail6 winner =
    scan edge (search edge 3) 4 [503] 503 root93Tail7 winner := by
  rw [root93Tail6, scan, if_neg (by decide), root93Candidates6_checked]
  change scan edge (search edge 3) 4 [503] 503 root93Tail7
    (if 3 ≤ root93Candidates6.length then search edge 3 [503, 1997] root93Candidates6 2500 winner else winner) = _
  rw [if_neg (by decide)]
def root93Candidates7 : List Nat := [15269, 16691]
set_option maxRecDepth 8192 in
theorem root93Candidates7_checked : candidates edge 4 503 2411 root93Tail8 winner.bound = root93Candidates7 := by
  unfold candidates
  have row : edge 2411 = (SortedNatTree.contains FastRows.row2411) := by
    change Data.row2411.contains = (SortedNatTree.contains FastRows.row2411)
    exact FastRows.row2411_checked.symm
  rw [row]
  decide
theorem root93Step7 : scan edge (search edge 3) 4 [503] 503 root93Tail7 winner =
    scan edge (search edge 3) 4 [503] 503 root93Tail8 winner := by
  rw [root93Tail7, scan, if_neg (by decide), root93Candidates7_checked]
  change scan edge (search edge 3) 4 [503] 503 root93Tail8
    (if 3 ≤ root93Candidates7.length then search edge 3 [503, 2411] root93Candidates7 2914 winner else winner) = _
  rw [if_neg (by decide)]
def root93Candidates8 : List Nat := [7817, 12917]
set_option maxRecDepth 8192 in
theorem root93Candidates8_checked : candidates edge 4 503 2711 root93Tail9 winner.bound = root93Candidates8 := by
  unfold candidates
  have row : edge 2711 = (SortedNatTree.contains FastRows.row2711) := by
    change Data.row2711.contains = (SortedNatTree.contains FastRows.row2711)
    exact FastRows.row2711_checked.symm
  rw [row]
  decide
theorem root93Step8 : scan edge (search edge 3) 4 [503] 503 root93Tail8 winner =
    scan edge (search edge 3) 4 [503] 503 root93Tail9 winner := by
  rw [root93Tail8, scan, if_neg (by decide), root93Candidates8_checked]
  change scan edge (search edge 3) 4 [503] 503 root93Tail9
    (if 3 ≤ root93Candidates8.length then search edge 3 [503, 2711] root93Candidates8 3214 winner else winner) = _
  rw [if_neg (by decide)]
def root93Candidates9 : List Nat := [15101]
set_option maxRecDepth 8192 in
theorem root93Candidates9_checked : candidates edge 4 503 3299 root93Tail10 winner.bound = root93Candidates9 := by
  unfold candidates
  have row : edge 3299 = (SortedNatTree.contains FastRows.row3299) := by
    change Data.row3299.contains = (SortedNatTree.contains FastRows.row3299)
    exact FastRows.row3299_checked.symm
  rw [row]
  decide
theorem root93Step9 : scan edge (search edge 3) 4 [503] 503 root93Tail9 winner =
    scan edge (search edge 3) 4 [503] 503 root93Tail10 winner := by
  rw [root93Tail9, scan, if_neg (by decide), root93Candidates9_checked]
  change scan edge (search edge 3) 4 [503] 503 root93Tail10
    (if 3 ≤ root93Candidates9.length then search edge 3 [503, 3299] root93Candidates9 3802 winner else winner) = _
  rw [if_neg (by decide)]
def root93Candidates10 : List Nat := [5021, 5903, 10607, 15773]
set_option maxRecDepth 8192 in
theorem root93Candidates10_checked : candidates edge 4 503 3593 root93Tail11 winner.bound = root93Candidates10 := by
  unfold candidates
  have row : edge 3593 = (SortedNatTree.contains FastRows.row3593) := by
    change Data.row3593.contains = (SortedNatTree.contains FastRows.row3593)
    exact FastRows.row3593_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root93Child10 : search edge 3 [503, 3593] root93Candidates10 4096 winner = winner := by decide
theorem root93Step10 : scan edge (search edge 3) 4 [503] 503 root93Tail10 winner =
    scan edge (search edge 3) 4 [503] 503 root93Tail11 winner := by
  rw [root93Tail10, scan, if_neg (by decide), root93Candidates10_checked]
  change scan edge (search edge 3) 4 [503] 503 root93Tail11
    (if 3 ≤ root93Candidates10.length then search edge 3 [503, 3593] root93Candidates10 4096 winner else winner) = _
  rw [if_pos (by decide), root93Child10]
def root93Candidates11 : List Nat := [20483]
set_option maxRecDepth 8192 in
theorem root93Candidates11_checked : candidates edge 4 503 4229 root93Tail12 winner.bound = root93Candidates11 := by
  unfold candidates
  have row : edge 4229 = (SortedNatTree.contains FastRows.row4229) := by
    change Data.row4229.contains = (SortedNatTree.contains FastRows.row4229)
    exact FastRows.row4229_checked.symm
  rw [row]
  decide
theorem root93Step11 : scan edge (search edge 3) 4 [503] 503 root93Tail11 winner =
    scan edge (search edge 3) 4 [503] 503 root93Tail12 winner := by
  rw [root93Tail11, scan, if_neg (by decide), root93Candidates11_checked]
  change scan edge (search edge 3) 4 [503] 503 root93Tail12
    (if 3 ≤ root93Candidates11.length then search edge 3 [503, 4229] root93Candidates11 4732 winner else winner) = _
  rw [if_neg (by decide)]
def root93Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root93Candidates12_checked : candidates edge 4 503 5003 root93Tail13 winner.bound = root93Candidates12 := by
  unfold candidates
  have row : edge 5003 = (SortedNatTree.contains FastRows.row5003) := by
    change Data.row5003.contains = (SortedNatTree.contains FastRows.row5003)
    exact FastRows.row5003_checked.symm
  rw [row]
  decide
theorem root93Step12 : scan edge (search edge 3) 4 [503] 503 root93Tail12 winner =
    scan edge (search edge 3) 4 [503] 503 root93Tail13 winner := by
  rw [root93Tail12, scan, if_neg (by decide), root93Candidates12_checked]
  change scan edge (search edge 3) 4 [503] 503 root93Tail13
    (if 3 ≤ root93Candidates12.length then search edge 3 [503, 5003] root93Candidates12 5506 winner else winner) = _
  rw [if_neg (by decide)]
def root93Candidates13 : List Nat := [5507]
set_option maxRecDepth 8192 in
theorem root93Candidates13_checked : candidates edge 4 503 5021 root93Tail14 winner.bound = root93Candidates13 := by
  unfold candidates
  have row : edge 5021 = (SortedNatTree.contains FastRows.row5021) := by
    change Data.row5021.contains = (SortedNatTree.contains FastRows.row5021)
    exact FastRows.row5021_checked.symm
  rw [row]
  decide
theorem root93Step13 : scan edge (search edge 3) 4 [503] 503 root93Tail13 winner =
    scan edge (search edge 3) 4 [503] 503 root93Tail14 winner := by
  rw [root93Tail13, scan, if_neg (by decide), root93Candidates13_checked]
  change scan edge (search edge 3) 4 [503] 503 root93Tail14
    (if 3 ≤ root93Candidates13.length then search edge 3 [503, 5021] root93Candidates13 5524 winner else winner) = _
  rw [if_neg (by decide)]
def root93Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root93Candidates14_checked : candidates edge 4 503 5309 root93Tail15 winner.bound = root93Candidates14 := by
  unfold candidates
  have row : edge 5309 = (SortedNatTree.contains FastRows.row5309) := by
    change Data.row5309.contains = (SortedNatTree.contains FastRows.row5309)
    exact FastRows.row5309_checked.symm
  rw [row]
  decide
theorem root93Step14 : scan edge (search edge 3) 4 [503] 503 root93Tail14 winner =
    scan edge (search edge 3) 4 [503] 503 root93Tail15 winner := by
  rw [root93Tail14, scan, if_neg (by decide), root93Candidates14_checked]
  change scan edge (search edge 3) 4 [503] 503 root93Tail15
    (if 3 ≤ root93Candidates14.length then search edge 3 [503, 5309] root93Candidates14 5812 winner else winner) = _
  rw [if_neg (by decide)]
def root93Candidates15 : List Nat := [15101]
set_option maxRecDepth 8192 in
theorem root93Candidates15_checked : candidates edge 4 503 5351 root93Tail16 winner.bound = root93Candidates15 := by
  unfold candidates
  have row : edge 5351 = (SortedNatTree.contains FastRows.row5351) := by
    change Data.row5351.contains = (SortedNatTree.contains FastRows.row5351)
    exact FastRows.row5351_checked.symm
  rw [row]
  decide
theorem root93Step15 : scan edge (search edge 3) 4 [503] 503 root93Tail15 winner =
    scan edge (search edge 3) 4 [503] 503 root93Tail16 winner := by
  rw [root93Tail15, scan, if_neg (by decide), root93Candidates15_checked]
  change scan edge (search edge 3) 4 [503] 503 root93Tail16
    (if 3 ≤ root93Candidates15.length then search edge 3 [503, 5351] root93Candidates15 5854 winner else winner) = _
  rw [if_neg (by decide)]
def root93Candidates16 : List Nat := [9623, 10607, 15773]
set_option maxRecDepth 8192 in
theorem root93Candidates16_checked : candidates edge 4 503 5507 root93Tail17 winner.bound = root93Candidates16 := by
  unfold candidates
  have row : edge 5507 = (SortedNatTree.contains FastRows.row5507) := by
    change Data.row5507.contains = (SortedNatTree.contains FastRows.row5507)
    exact FastRows.row5507_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root93Child16 : search edge 3 [503, 5507] root93Candidates16 6010 winner = winner := by decide
theorem root93Step16 : scan edge (search edge 3) 4 [503] 503 root93Tail16 winner =
    scan edge (search edge 3) 4 [503] 503 root93Tail17 winner := by
  rw [root93Tail16, scan, if_neg (by decide), root93Candidates16_checked]
  change scan edge (search edge 3) 4 [503] 503 root93Tail17
    (if 3 ≤ root93Candidates16.length then search edge 3 [503, 5507] root93Candidates16 6010 winner else winner) = _
  rw [if_pos (by decide), root93Child16]
def root93Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root93Candidates17_checked : candidates edge 4 503 5843 root93Tail18 winner.bound = root93Candidates17 := by
  unfold candidates
  have row : edge 5843 = (SortedNatTree.contains FastRows.row5843) := by
    change Data.row5843.contains = (SortedNatTree.contains FastRows.row5843)
    exact FastRows.row5843_checked.symm
  rw [row]
  decide
theorem root93Step17 : scan edge (search edge 3) 4 [503] 503 root93Tail17 winner =
    scan edge (search edge 3) 4 [503] 503 root93Tail18 winner := by
  rw [root93Tail17, scan, if_neg (by decide), root93Candidates17_checked]
  change scan edge (search edge 3) 4 [503] 503 root93Tail18
    (if 3 ≤ root93Candidates17.length then search edge 3 [503, 5843] root93Candidates17 6346 winner else winner) = _
  rw [if_neg (by decide)]
def root93Candidates18 : List Nat := [9059, 13553]
set_option maxRecDepth 8192 in
theorem root93Candidates18_checked : candidates edge 4 503 5903 root93Tail19 winner.bound = root93Candidates18 := by
  unfold candidates
  have row : edge 5903 = (SortedNatTree.contains FastRows.row5903) := by
    change Data.row5903.contains = (SortedNatTree.contains FastRows.row5903)
    exact FastRows.row5903_checked.symm
  rw [row]
  decide
theorem root93Step18 : scan edge (search edge 3) 4 [503] 503 root93Tail18 winner =
    scan edge (search edge 3) 4 [503] 503 root93Tail19 winner := by
  rw [root93Tail18, scan, if_neg (by decide), root93Candidates18_checked]
  change scan edge (search edge 3) 4 [503] 503 root93Tail19
    (if 3 ≤ root93Candidates18.length then search edge 3 [503, 5903] root93Candidates18 6406 winner else winner) = _
  rw [if_neg (by decide)]
theorem root93Scan19 : scan edge (search edge 3) 4 [503] 503 root93Tail19 winner = winner := by rw [root93Tail19, scan, if_pos (by decide)]
theorem root93Scan18 : scan edge (search edge 3) 4 [503] 503 root93Tail18 winner = winner :=
  (root93Step18).trans root93Scan19
theorem root93Scan17 : scan edge (search edge 3) 4 [503] 503 root93Tail17 winner = winner :=
  (root93Step17).trans root93Scan18
theorem root93Scan16 : scan edge (search edge 3) 4 [503] 503 root93Tail16 winner = winner :=
  (root93Step16).trans root93Scan17
theorem root93Scan15 : scan edge (search edge 3) 4 [503] 503 root93Tail15 winner = winner :=
  (root93Step15).trans root93Scan16
theorem root93Scan14 : scan edge (search edge 3) 4 [503] 503 root93Tail14 winner = winner :=
  (root93Step14).trans root93Scan15
theorem root93Scan13 : scan edge (search edge 3) 4 [503] 503 root93Tail13 winner = winner :=
  (root93Step13).trans root93Scan14
theorem root93Scan12 : scan edge (search edge 3) 4 [503] 503 root93Tail12 winner = winner :=
  (root93Step12).trans root93Scan13
theorem root93Scan11 : scan edge (search edge 3) 4 [503] 503 root93Tail11 winner = winner :=
  (root93Step11).trans root93Scan12
theorem root93Scan10 : scan edge (search edge 3) 4 [503] 503 root93Tail10 winner = winner :=
  (root93Step10).trans root93Scan11
theorem root93Scan9 : scan edge (search edge 3) 4 [503] 503 root93Tail9 winner = winner :=
  (root93Step9).trans root93Scan10
theorem root93Scan8 : scan edge (search edge 3) 4 [503] 503 root93Tail8 winner = winner :=
  (root93Step8).trans root93Scan9
theorem root93Scan7 : scan edge (search edge 3) 4 [503] 503 root93Tail7 winner = winner :=
  (root93Step7).trans root93Scan8
theorem root93Scan6 : scan edge (search edge 3) 4 [503] 503 root93Tail6 winner = winner :=
  (root93Step6).trans root93Scan7
theorem root93Scan5 : scan edge (search edge 3) 4 [503] 503 root93Tail5 winner = winner :=
  (root93Step5).trans root93Scan6
theorem root93Scan4 : scan edge (search edge 3) 4 [503] 503 root93Tail4 winner = winner :=
  (root93Step4).trans root93Scan5
theorem root93Scan3 : scan edge (search edge 3) 4 [503] 503 root93Tail3 winner = winner :=
  (root93Step3).trans root93Scan4
theorem root93Scan2 : scan edge (search edge 3) 4 [503] 503 root93Tail2 winner = winner :=
  (root93Step2).trans root93Scan3
theorem root93Scan1 : scan edge (search edge 3) 4 [503] 503 root93Tail1 winner = winner :=
  (root93Step1).trans root93Scan2
theorem root93Scan0 : scan edge (search edge 3) 4 [503] 503 root93Tail0 winner = winner :=
  (root93Step0).trans root93Scan1
set_option maxRecDepth 4096 in
theorem root93Branch : search edge 4 [503] root93Candidates 503 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [503] 503 root93Tail0 winner = winner
  exact root93Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
