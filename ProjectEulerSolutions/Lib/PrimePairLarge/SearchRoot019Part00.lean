import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot019Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root19Candidates0 : List Nat := [1627, 13183, 21493]
set_option maxRecDepth 8192 in
theorem root19Candidates0_checked : candidates edge 4 79 193 root19Tail1 winner.bound = root19Candidates0 := by
  unfold candidates
  have row : edge 193 = (SortedNatTree.contains FastRows.row193) := by
    change Data.row193.contains = (SortedNatTree.contains FastRows.row193)
    exact FastRows.row193_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root19Child0 : search edge 3 [79, 193] root19Candidates0 272 winner = winner := by decide
theorem root19Step0 : scan edge (search edge 3) 4 [79] 79 root19Tail0 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail1 winner := by
  rw [root19Tail0, scan, if_neg (by decide), root19Candidates0_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail1
    (if 3 ≤ root19Candidates0.length then search edge 3 [79, 193] root19Candidates0 272 winner else winner) = _
  rw [if_pos (by decide), root19Child0]
def root19Candidates1 : List Nat := [2797, 3253, 3511, 5503, 23497, 25237]
set_option maxRecDepth 8192 in
theorem root19Candidates1_checked : candidates edge 4 79 241 root19Tail2 winner.bound = root19Candidates1 := by
  unfold candidates
  have row : edge 241 = (SortedNatTree.contains FastRows.row241) := by
    change Data.row241.contains = (SortedNatTree.contains FastRows.row241)
    exact FastRows.row241_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root19Child1 : search edge 3 [79, 241] root19Candidates1 320 winner = winner := by decide
theorem root19Step1 : scan edge (search edge 3) 4 [79] 79 root19Tail1 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail2 winner := by
  rw [root19Tail1, scan, if_neg (by decide), root19Candidates1_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail2
    (if 3 ≤ root19Candidates1.length then search edge 3 [79, 241] root19Candidates1 320 winner else winner) = _
  rw [if_pos (by decide), root19Child1]
def root19Candidates2 : List Nat := [613, 1117, 6091, 7207, 8887, 9679, 15643, 17341]
set_option maxRecDepth 8192 in
theorem root19Candidates2_checked : candidates edge 4 79 367 root19Tail3 winner.bound = root19Candidates2 := by
  unfold candidates
  have row : edge 367 = (SortedNatTree.contains FastRows.row367) := by
    change Data.row367.contains = (SortedNatTree.contains FastRows.row367)
    exact FastRows.row367_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root19Child2 : search edge 3 [79, 367] root19Candidates2 446 winner = winner := by decide
theorem root19Step2 : scan edge (search edge 3) 4 [79] 79 root19Tail2 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail3 winner := by
  rw [root19Tail2, scan, if_neg (by decide), root19Candidates2_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail3
    (if 3 ≤ root19Candidates2.length then search edge 3 [79, 367] root19Candidates2 446 winner else winner) = _
  rw [if_pos (by decide), root19Child2]
def root19Candidates3 : List Nat := [2593, 3511, 4483, 5791, 7933, 17137, 25237]
set_option maxRecDepth 8192 in
theorem root19Candidates3_checked : candidates edge 4 79 397 root19Tail4 winner.bound = root19Candidates3 := by
  unfold candidates
  have row : edge 397 = (SortedNatTree.contains FastRows.row397) := by
    change Data.row397.contains = (SortedNatTree.contains FastRows.row397)
    exact FastRows.row397_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root19Child3 : search edge 3 [79, 397] root19Candidates3 476 winner = winner := by decide
theorem root19Step3 : scan edge (search edge 3) 4 [79] 79 root19Tail3 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail4 winner := by
  rw [root19Tail3, scan, if_neg (by decide), root19Candidates3_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail4
    (if 3 ≤ root19Candidates3.length then search edge 3 [79, 397] root19Candidates3 476 winner else winner) = _
  rw [if_pos (by decide), root19Child3]
def root19Candidates4 : List Nat := [4153, 5581, 6673, 15643, 18793]
set_option maxRecDepth 8192 in
theorem root19Candidates4_checked : candidates edge 4 79 613 root19Tail5 winner.bound = root19Candidates4 := by
  unfold candidates
  have row : edge 613 = (SortedNatTree.contains FastRows.row613) := by
    change Data.row613.contains = (SortedNatTree.contains FastRows.row613)
    exact FastRows.row613_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root19Child4 : search edge 3 [79, 613] root19Candidates4 692 winner = winner := by decide
theorem root19Step4 : scan edge (search edge 3) 4 [79] 79 root19Tail4 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail5 winner := by
  rw [root19Tail4, scan, if_neg (by decide), root19Candidates4_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail5
    (if 3 ≤ root19Candidates4.length then search edge 3 [79, 613] root19Candidates4 692 winner else winner) = _
  rw [if_pos (by decide), root19Child4]
def root19Candidates5 : List Nat := [3253, 13669, 17419]
set_option maxRecDepth 8192 in
theorem root19Candidates5_checked : candidates edge 4 79 631 root19Tail6 winner.bound = root19Candidates5 := by
  unfold candidates
  have row : edge 631 = (SortedNatTree.contains FastRows.row631) := by
    change Data.row631.contains = (SortedNatTree.contains FastRows.row631)
    exact FastRows.row631_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root19Child5 : search edge 3 [79, 631] root19Candidates5 710 winner = winner := by decide
theorem root19Step5 : scan edge (search edge 3) 4 [79] 79 root19Tail5 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail6 winner := by
  rw [root19Tail5, scan, if_neg (by decide), root19Candidates5_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail6
    (if 3 ≤ root19Candidates5.length then search edge 3 [79, 631] root19Candidates5 710 winner else winner) = _
  rw [if_pos (by decide), root19Child5]
def root19Candidates6 : List Nat := [1117, 3511, 22111, 24967]
set_option maxRecDepth 8192 in
theorem root19Candidates6_checked : candidates edge 4 79 967 root19Tail7 winner.bound = root19Candidates6 := by
  unfold candidates
  have row : edge 967 = (SortedNatTree.contains FastRows.row967) := by
    change Data.row967.contains = (SortedNatTree.contains FastRows.row967)
    exact FastRows.row967_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root19Child6 : search edge 3 [79, 967] root19Candidates6 1046 winner = winner := by decide
theorem root19Step6 : scan edge (search edge 3) 4 [79] 79 root19Tail6 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail7 winner := by
  rw [root19Tail6, scan, if_neg (by decide), root19Candidates6_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail7
    (if 3 ≤ root19Candidates6.length then search edge 3 [79, 967] root19Candidates6 1046 winner else winner) = _
  rw [if_pos (by decide), root19Child6]
def root19Candidates7 : List Nat := [6553, 7057, 19993, 20611]
set_option maxRecDepth 8192 in
theorem root19Candidates7_checked : candidates edge 4 79 1093 root19Tail8 winner.bound = root19Candidates7 := by
  unfold candidates
  have row : edge 1093 = (SortedNatTree.contains FastRows.row1093) := by
    change Data.row1093.contains = (SortedNatTree.contains FastRows.row1093)
    exact FastRows.row1093_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root19Child7 : search edge 3 [79, 1093] root19Candidates7 1172 winner = winner := by decide
theorem root19Step7 : scan edge (search edge 3) 4 [79] 79 root19Tail7 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail8 winner := by
  rw [root19Tail7, scan, if_neg (by decide), root19Candidates7_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail8
    (if 3 ≤ root19Candidates7.length then search edge 3 [79, 1093] root19Candidates7 1172 winner else winner) = _
  rw [if_pos (by decide), root19Child7]
def root19Candidates8 : List Nat := [2797, 3343, 3511, 5077, 5323, 7591, 8887, 9859, 12511, 23497]
set_option maxRecDepth 8192 in
theorem root19Candidates8_checked : candidates edge 4 79 1117 root19Tail9 winner.bound = root19Candidates8 := by
  unfold candidates
  have row : edge 1117 = (SortedNatTree.contains FastRows.row1117) := by
    change Data.row1117.contains = (SortedNatTree.contains FastRows.row1117)
    exact FastRows.row1117_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root19Child8 : search edge 3 [79, 1117] root19Candidates8 1196 winner = winner := by decide
theorem root19Step8 : scan edge (search edge 3) 4 [79] 79 root19Tail8 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail9 winner := by
  rw [root19Tail8, scan, if_neg (by decide), root19Candidates8_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail9
    (if 3 ≤ root19Candidates8.length then search edge 3 [79, 1117] root19Candidates8 1196 winner else winner) = _
  rw [if_pos (by decide), root19Child8]
def root19Candidates9 : List Nat := [1993, 18517, 22531]
set_option maxRecDepth 8192 in
theorem root19Candidates9_checked : candidates edge 4 79 1447 root19Tail10 winner.bound = root19Candidates9 := by
  unfold candidates
  have row : edge 1447 = (SortedNatTree.contains FastRows.row1447) := by
    change Data.row1447.contains = (SortedNatTree.contains FastRows.row1447)
    exact FastRows.row1447_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root19Child9 : search edge 3 [79, 1447] root19Candidates9 1526 winner = winner := by decide
theorem root19Step9 : scan edge (search edge 3) 4 [79] 79 root19Tail9 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail10 winner := by
  rw [root19Tail9, scan, if_neg (by decide), root19Candidates9_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail10
    (if 3 ≤ root19Candidates9.length then search edge 3 [79, 1447] root19Candidates9 1526 winner else winner) = _
  rw [if_pos (by decide), root19Child9]
def root19Candidates10 : List Nat := [8461]
set_option maxRecDepth 8192 in
theorem root19Candidates10_checked : candidates edge 4 79 1627 root19Tail11 winner.bound = root19Candidates10 := by
  unfold candidates
  have row : edge 1627 = (SortedNatTree.contains FastRows.row1627) := by
    change Data.row1627.contains = (SortedNatTree.contains FastRows.row1627)
    exact FastRows.row1627_checked.symm
  rw [row]
  decide
theorem root19Step10 : scan edge (search edge 3) 4 [79] 79 root19Tail10 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail11 winner := by
  rw [root19Tail10, scan, if_neg (by decide), root19Candidates10_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail11
    (if 3 ≤ root19Candidates10.length then search edge 3 [79, 1627] root19Candidates10 1706 winner else winner) = _
  rw [if_neg (by decide)]
def root19Candidates11 : List Nat := [5101, 14449, 18793, 23563]
set_option maxRecDepth 8192 in
theorem root19Candidates11_checked : candidates edge 4 79 1657 root19Tail12 winner.bound = root19Candidates11 := by
  unfold candidates
  have row : edge 1657 = (SortedNatTree.contains FastRows.row1657) := by
    change Data.row1657.contains = (SortedNatTree.contains FastRows.row1657)
    exact FastRows.row1657_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root19Child11 : search edge 3 [79, 1657] root19Candidates11 1736 winner = winner := by decide
theorem root19Step11 : scan edge (search edge 3) 4 [79] 79 root19Tail11 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail12 winner := by
  rw [root19Tail11, scan, if_neg (by decide), root19Candidates11_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail12
    (if 3 ≤ root19Candidates11.length then search edge 3 [79, 1657] root19Candidates11 1736 winner else winner) = _
  rw [if_pos (by decide), root19Child11]
def root19Candidates12 : List Nat := [2647, 3253, 3547]
set_option maxRecDepth 8192 in
theorem root19Candidates12_checked : candidates edge 4 79 1801 root19Tail13 winner.bound = root19Candidates12 := by
  unfold candidates
  have row : edge 1801 = (SortedNatTree.contains FastRows.row1801) := by
    change Data.row1801.contains = (SortedNatTree.contains FastRows.row1801)
    exact FastRows.row1801_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root19Child12 : search edge 3 [79, 1801] root19Candidates12 1880 winner = winner := by decide
theorem root19Step12 : scan edge (search edge 3) 4 [79] 79 root19Tail12 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail13 winner := by
  rw [root19Tail12, scan, if_neg (by decide), root19Candidates12_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail13
    (if 3 ≤ root19Candidates12.length then search edge 3 [79, 1801] root19Candidates12 1880 winner else winner) = _
  rw [if_pos (by decide), root19Child12]
def root19Candidates13 : List Nat := [3733, 5791, 7933, 8521, 19993]
set_option maxRecDepth 8192 in
theorem root19Candidates13_checked : candidates edge 4 79 1933 root19Tail14 winner.bound = root19Candidates13 := by
  unfold candidates
  have row : edge 1933 = (SortedNatTree.contains FastRows.row1933) := by
    change Data.row1933.contains = (SortedNatTree.contains FastRows.row1933)
    exact FastRows.row1933_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root19Child13 : search edge 3 [79, 1933] root19Candidates13 2012 winner = winner := by decide
theorem root19Step13 : scan edge (search edge 3) 4 [79] 79 root19Tail13 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail14 winner := by
  rw [root19Tail13, scan, if_neg (by decide), root19Candidates13_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail14
    (if 3 ≤ root19Candidates13.length then search edge 3 [79, 1933] root19Candidates13 2012 winner else winner) = _
  rw [if_pos (by decide), root19Child13]
def root19Candidates14 : List Nat := [4483, 5581, 6781, 19993, 23563]
set_option maxRecDepth 8192 in
theorem root19Candidates14_checked : candidates edge 4 79 1993 root19Tail15 winner.bound = root19Candidates14 := by
  unfold candidates
  have row : edge 1993 = (SortedNatTree.contains FastRows.row1993) := by
    change Data.row1993.contains = (SortedNatTree.contains FastRows.row1993)
    exact FastRows.row1993_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root19Child14 : search edge 3 [79, 1993] root19Candidates14 2072 winner = winner := by decide
theorem root19Step14 : scan edge (search edge 3) 4 [79] 79 root19Tail14 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail15 winner := by
  rw [root19Tail14, scan, if_neg (by decide), root19Candidates14_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail15
    (if 3 ≤ root19Candidates14.length then search edge 3 [79, 1993] root19Candidates14 2072 winner else winner) = _
  rw [if_pos (by decide), root19Child14]
def root19Candidates15 : List Nat := [3253, 5647, 18517]
set_option maxRecDepth 8192 in
theorem root19Candidates15_checked : candidates edge 4 79 2179 root19Tail16 winner.bound = root19Candidates15 := by
  unfold candidates
  have row : edge 2179 = (SortedNatTree.contains FastRows.row2179) := by
    change Data.row2179.contains = (SortedNatTree.contains FastRows.row2179)
    exact FastRows.row2179_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root19Child15 : search edge 3 [79, 2179] root19Candidates15 2258 winner = winner := by decide
theorem root19Step15 : scan edge (search edge 3) 4 [79] 79 root19Tail15 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail16 winner := by
  rw [root19Tail15, scan, if_neg (by decide), root19Candidates15_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail16
    (if 3 ≤ root19Candidates15.length then search edge 3 [79, 2179] root19Candidates15 2258 winner else winner) = _
  rw [if_pos (by decide), root19Child15]
def root19Candidates16 : List Nat := [3607, 19609]
set_option maxRecDepth 8192 in
theorem root19Candidates16_checked : candidates edge 4 79 2371 root19Tail17 winner.bound = root19Candidates16 := by
  unfold candidates
  have row : edge 2371 = (SortedNatTree.contains FastRows.row2371) := by
    change Data.row2371.contains = (SortedNatTree.contains FastRows.row2371)
    exact FastRows.row2371_checked.symm
  rw [row]
  decide
theorem root19Step16 : scan edge (search edge 3) 4 [79] 79 root19Tail16 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail17 winner := by
  rw [root19Tail16, scan, if_neg (by decide), root19Candidates16_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail17
    (if 3 ≤ root19Candidates16.length then search edge 3 [79, 2371] root19Candidates16 2450 winner else winner) = _
  rw [if_neg (by decide)]
def root19Candidates17 : List Nat := [9859]
set_option maxRecDepth 8192 in
theorem root19Candidates17_checked : candidates edge 4 79 2551 root19Tail18 winner.bound = root19Candidates17 := by
  unfold candidates
  have row : edge 2551 = (SortedNatTree.contains FastRows.row2551) := by
    change Data.row2551.contains = (SortedNatTree.contains FastRows.row2551)
    exact FastRows.row2551_checked.symm
  rw [row]
  decide
theorem root19Step17 : scan edge (search edge 3) 4 [79] 79 root19Tail17 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail18 winner := by
  rw [root19Tail17, scan, if_neg (by decide), root19Candidates17_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail18
    (if 3 ≤ root19Candidates17.length then search edge 3 [79, 2551] root19Candidates17 2630 winner else winner) = _
  rw [if_neg (by decide)]
def root19Candidates18 : List Nat := [5101, 20389]
set_option maxRecDepth 8192 in
theorem root19Candidates18_checked : candidates edge 4 79 2593 root19Tail19 winner.bound = root19Candidates18 := by
  unfold candidates
  have row : edge 2593 = (SortedNatTree.contains FastRows.row2593) := by
    change Data.row2593.contains = (SortedNatTree.contains FastRows.row2593)
    exact FastRows.row2593_checked.symm
  rw [row]
  decide
theorem root19Step18 : scan edge (search edge 3) 4 [79] 79 root19Tail18 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail19 winner := by
  rw [root19Tail18, scan, if_neg (by decide), root19Candidates18_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail19
    (if 3 ≤ root19Candidates18.length then search edge 3 [79, 2593] root19Candidates18 2672 winner else winner) = _
  rw [if_neg (by decide)]
def root19Candidates19 : List Nat := [6091, 6553, 6673]
set_option maxRecDepth 8192 in
theorem root19Candidates19_checked : candidates edge 4 79 2647 root19Tail20 winner.bound = root19Candidates19 := by
  unfold candidates
  have row : edge 2647 = (SortedNatTree.contains FastRows.row2647) := by
    change Data.row2647.contains = (SortedNatTree.contains FastRows.row2647)
    exact FastRows.row2647_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root19Child19 : search edge 3 [79, 2647] root19Candidates19 2726 winner = winner := by decide
theorem root19Step19 : scan edge (search edge 3) 4 [79] 79 root19Tail19 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail20 winner := by
  rw [root19Tail19, scan, if_neg (by decide), root19Candidates19_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail20
    (if 3 ≤ root19Candidates19.length then search edge 3 [79, 2647] root19Candidates19 2726 winner else winner) = _
  rw [if_pos (by decide), root19Child19]
def root19Candidates20 : List Nat := [3343]
set_option maxRecDepth 8192 in
theorem root19Candidates20_checked : candidates edge 4 79 2689 root19Tail21 winner.bound = root19Candidates20 := by
  unfold candidates
  have row : edge 2689 = (SortedNatTree.contains FastRows.row2689) := by
    change Data.row2689.contains = (SortedNatTree.contains FastRows.row2689)
    exact FastRows.row2689_checked.symm
  rw [row]
  decide
theorem root19Step20 : scan edge (search edge 3) 4 [79] 79 root19Tail20 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail21 winner := by
  rw [root19Tail20, scan, if_neg (by decide), root19Candidates20_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail21
    (if 3 ≤ root19Candidates20.length then search edge 3 [79, 2689] root19Candidates20 2768 winner else winner) = _
  rw [if_neg (by decide)]
def root19Candidates21 : List Nat := [5449, 7207, 17419, 18979]
set_option maxRecDepth 8192 in
theorem root19Candidates21_checked : candidates edge 4 79 2797 root19Tail22 winner.bound = root19Candidates21 := by
  unfold candidates
  have row : edge 2797 = (SortedNatTree.contains FastRows.row2797) := by
    change Data.row2797.contains = (SortedNatTree.contains FastRows.row2797)
    exact FastRows.row2797_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root19Child21 : search edge 3 [79, 2797] root19Candidates21 2876 winner = winner := by decide
theorem root19Step21 : scan edge (search edge 3) 4 [79] 79 root19Tail21 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail22 winner := by
  rw [root19Tail21, scan, if_neg (by decide), root19Candidates21_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail22
    (if 3 ≤ root19Candidates21.length then search edge 3 [79, 2797] root19Candidates21 2876 winner else winner) = _
  rw [if_pos (by decide), root19Child21]
def root19Candidates22 : List Nat := [3343, 18517, 20611]
set_option maxRecDepth 8192 in
theorem root19Candidates22_checked : candidates edge 4 79 3181 root19Tail23 winner.bound = root19Candidates22 := by
  unfold candidates
  have row : edge 3181 = (SortedNatTree.contains FastRows.row3181) := by
    change Data.row3181.contains = (SortedNatTree.contains FastRows.row3181)
    exact FastRows.row3181_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root19Child22 : search edge 3 [79, 3181] root19Candidates22 3260 winner = winner := by decide
theorem root19Step22 : scan edge (search edge 3) 4 [79] 79 root19Tail22 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail23 winner := by
  rw [root19Tail22, scan, if_neg (by decide), root19Candidates22_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail23
    (if 3 ≤ root19Candidates22.length then search edge 3 [79, 3181] root19Candidates22 3260 winner else winner) = _
  rw [if_pos (by decide), root19Child22]
def root19Candidates23 : List Nat := [3517, 3547, 3607, 4231, 8263, 20389]
set_option maxRecDepth 8192 in
theorem root19Candidates23_checked : candidates edge 4 79 3253 root19Tail24 winner.bound = root19Candidates23 := by
  unfold candidates
  have row : edge 3253 = (SortedNatTree.contains FastRows.row3253) := by
    change Data.row3253.contains = (SortedNatTree.contains FastRows.row3253)
    exact FastRows.row3253_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root19Child23 : search edge 3 [79, 3253] root19Candidates23 3332 winner = winner := by decide
theorem root19Step23 : scan edge (search edge 3) 4 [79] 79 root19Tail23 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail24 winner := by
  rw [root19Tail23, scan, if_neg (by decide), root19Candidates23_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail24
    (if 3 ≤ root19Candidates23.length then search edge 3 [79, 3253] root19Candidates23 3332 winner else winner) = _
  rw [if_pos (by decide), root19Child23]
def root19Candidates24 : List Nat := [4483, 5449, 5791, 6373, 6451, 17341]
set_option maxRecDepth 8192 in
theorem root19Candidates24_checked : candidates edge 4 79 3343 root19Tail25 winner.bound = root19Candidates24 := by
  unfold candidates
  have row : edge 3343 = (SortedNatTree.contains FastRows.row3343) := by
    change Data.row3343.contains = (SortedNatTree.contains FastRows.row3343)
    exact FastRows.row3343_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root19Child24 : search edge 3 [79, 3343] root19Candidates24 3422 winner = winner := by decide
theorem root19Step24 : scan edge (search edge 3) 4 [79] 79 root19Tail24 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail25 winner := by
  rw [root19Tail24, scan, if_neg (by decide), root19Candidates24_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail25
    (if 3 ≤ root19Candidates24.length then search edge 3 [79, 3343] root19Candidates24 3422 winner else winner) = _
  rw [if_pos (by decide), root19Child24]
def root19Candidates25 : List Nat := [4153, 7561, 18637]
set_option maxRecDepth 8192 in
theorem root19Candidates25_checked : candidates edge 4 79 3511 root19Tail26 winner.bound = root19Candidates25 := by
  unfold candidates
  have row : edge 3511 = (SortedNatTree.contains FastRows.row3511) := by
    change Data.row3511.contains = (SortedNatTree.contains FastRows.row3511)
    exact FastRows.row3511_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root19Child25 : search edge 3 [79, 3511] root19Candidates25 3590 winner = winner := by decide
theorem root19Step25 : scan edge (search edge 3) 4 [79] 79 root19Tail25 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail26 winner := by
  rw [root19Tail25, scan, if_neg (by decide), root19Candidates25_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail26
    (if 3 ≤ root19Candidates25.length then search edge 3 [79, 3511] root19Candidates25 3590 winner else winner) = _
  rw [if_pos (by decide), root19Child25]
def root19Candidates26 : List Nat := [5659, 8263]
set_option maxRecDepth 8192 in
theorem root19Candidates26_checked : candidates edge 4 79 3517 root19Tail27 winner.bound = root19Candidates26 := by
  unfold candidates
  have row : edge 3517 = (SortedNatTree.contains FastRows.row3517) := by
    change Data.row3517.contains = (SortedNatTree.contains FastRows.row3517)
    exact FastRows.row3517_checked.symm
  rw [row]
  decide
theorem root19Step26 : scan edge (search edge 3) 4 [79] 79 root19Tail26 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail27 winner := by
  rw [root19Tail26, scan, if_neg (by decide), root19Candidates26_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail27
    (if 3 ≤ root19Candidates26.length then search edge 3 [79, 3517] root19Candidates26 3596 winner else winner) = _
  rw [if_neg (by decide)]
def root19Candidates27 : List Nat := [5449, 8461, 12343, 18979, 22129]
set_option maxRecDepth 8192 in
theorem root19Candidates27_checked : candidates edge 4 79 3547 root19Tail28 winner.bound = root19Candidates27 := by
  unfold candidates
  have row : edge 3547 = (SortedNatTree.contains FastRows.row3547) := by
    change Data.row3547.contains = (SortedNatTree.contains FastRows.row3547)
    exact FastRows.row3547_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root19Child27 : search edge 3 [79, 3547] root19Candidates27 3626 winner = winner := by decide
theorem root19Step27 : scan edge (search edge 3) 4 [79] 79 root19Tail27 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail28 winner := by
  rw [root19Tail27, scan, if_neg (by decide), root19Candidates27_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail28
    (if 3 ≤ root19Candidates27.length then search edge 3 [79, 3547] root19Candidates27 3626 winner else winner) = _
  rw [if_pos (by decide), root19Child27]
def root19Candidates28 : List Nat := [5077, 8521, 17011, 18517]
set_option maxRecDepth 8192 in
theorem root19Candidates28_checked : candidates edge 4 79 3607 root19Tail29 winner.bound = root19Candidates28 := by
  unfold candidates
  have row : edge 3607 = (SortedNatTree.contains FastRows.row3607) := by
    change Data.row3607.contains = (SortedNatTree.contains FastRows.row3607)
    exact FastRows.row3607_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root19Child28 : search edge 3 [79, 3607] root19Candidates28 3686 winner = winner := by decide
theorem root19Step28 : scan edge (search edge 3) 4 [79] 79 root19Tail28 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail29 winner := by
  rw [root19Tail28, scan, if_neg (by decide), root19Candidates28_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail29
    (if 3 ≤ root19Candidates28.length then search edge 3 [79, 3607] root19Candidates28 3686 winner else winner) = _
  rw [if_pos (by decide), root19Child28]
def root19Candidates29 : List Nat := [6451, 8461, 15643]
set_option maxRecDepth 8192 in
theorem root19Candidates29_checked : candidates edge 4 79 3733 root19Tail30 winner.bound = root19Candidates29 := by
  unfold candidates
  have row : edge 3733 = (SortedNatTree.contains FastRows.row3733) := by
    change Data.row3733.contains = (SortedNatTree.contains FastRows.row3733)
    exact FastRows.row3733_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root19Child29 : search edge 3 [79, 3733] root19Candidates29 3812 winner = winner := by decide
theorem root19Step29 : scan edge (search edge 3) 4 [79] 79 root19Tail29 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail30 winner := by
  rw [root19Tail29, scan, if_neg (by decide), root19Candidates29_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail30
    (if 3 ≤ root19Candidates29.length then search edge 3 [79, 3733] root19Candidates29 3812 winner else winner) = _
  rw [if_pos (by decide), root19Child29]
def root19Candidates30 : List Nat := [7591, 8887, 18979]
set_option maxRecDepth 8192 in
theorem root19Candidates30_checked : candidates edge 4 79 3853 root19Tail31 winner.bound = root19Candidates30 := by
  unfold candidates
  have row : edge 3853 = (SortedNatTree.contains FastRows.row3853) := by
    change Data.row3853.contains = (SortedNatTree.contains FastRows.row3853)
    exact FastRows.row3853_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root19Child30 : search edge 3 [79, 3853] root19Candidates30 3932 winner = winner := by decide
theorem root19Step30 : scan edge (search edge 3) 4 [79] 79 root19Tail30 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail31 winner := by
  rw [root19Tail30, scan, if_neg (by decide), root19Candidates30_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail31
    (if 3 ≤ root19Candidates30.length then search edge 3 [79, 3853] root19Candidates30 3932 winner else winner) = _
  rw [if_pos (by decide), root19Child30]
def root19Candidates31 : List Nat := [5077, 5101, 5503, 12343]
set_option maxRecDepth 8192 in
theorem root19Candidates31_checked : candidates edge 4 79 4153 root19Tail32 winner.bound = root19Candidates31 := by
  unfold candidates
  have row : edge 4153 = (SortedNatTree.contains FastRows.row4153) := by
    change Data.row4153.contains = (SortedNatTree.contains FastRows.row4153)
    exact FastRows.row4153_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root19Child31 : search edge 3 [79, 4153] root19Candidates31 4232 winner = winner := by decide
theorem root19Step31 : scan edge (search edge 3) 4 [79] 79 root19Tail31 winner =
    scan edge (search edge 3) 4 [79] 79 root19Tail32 winner := by
  rw [root19Tail31, scan, if_neg (by decide), root19Candidates31_checked]
  change scan edge (search edge 3) 4 [79] 79 root19Tail32
    (if 3 ≤ root19Candidates31.length then search edge 3 [79, 4153] root19Candidates31 4232 winner else winner) = _
  rw [if_pos (by decide), root19Child31]
end ProjectEulerSolutions.PrimePairLarge.Search
