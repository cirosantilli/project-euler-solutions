import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root175Candidates : List Nat := Data.row1061
set_option maxRecDepth 8192 in
theorem root175Candidates_checked : candidates edge 5 0 1061 tail176 winner.bound = root175Candidates := by
  apply candidates_eq_row edge 5 0 1061 tail176 root175Candidates winner.bound tail176_ordered
  · decide
  · decide
  · funext q
    rfl
def root175Tail53 : List Nat := []
def root175Tail52 : List Nat := 24677 :: root175Tail53
def root175Tail51 : List Nat := 24533 :: root175Tail52
def root175Tail50 : List Nat := 24203 :: root175Tail51
def root175Tail49 : List Nat := 23753 :: root175Tail50
def root175Tail48 : List Nat := 23669 :: root175Tail49
def root175Tail47 : List Nat := 23609 :: root175Tail48
def root175Tail46 : List Nat := 23291 :: root175Tail47
def root175Tail45 : List Nat := 23021 :: root175Tail46
def root175Tail44 : List Nat := 22811 :: root175Tail45
def root175Tail43 : List Nat := 20393 :: root175Tail44
def root175Tail42 : List Nat := 20249 :: root175Tail43
def root175Tail41 : List Nat := 20123 :: root175Tail42
def root175Tail40 : List Nat := 19889 :: root175Tail41
def root175Tail39 : List Nat := 19001 :: root175Tail40
def root175Tail38 : List Nat := 18587 :: root175Tail39
def root175Tail37 : List Nat := 17783 :: root175Tail38
def root175Tail36 : List Nat := 17579 :: root175Tail37
def root175Tail35 : List Nat := 17021 :: root175Tail36
def root175Tail34 : List Nat := 16889 :: root175Tail35
def root175Tail33 : List Nat := 15731 :: root175Tail34
def root175Tail32 : List Nat := 15467 :: root175Tail33
def root175Tail31 : List Nat := 14303 :: root175Tail32
def root175Tail30 : List Nat := 13967 :: root175Tail31
def root175Tail29 : List Nat := 12527 :: root175Tail30
def root175Tail28 : List Nat := 10463 :: root175Tail29
def root175Tail27 : List Nat := 9923 :: root175Tail28
def root175Tail26 : List Nat := 9209 :: root175Tail27
def root175Tail25 : List Nat := 9137 :: root175Tail26
def root175Tail24 : List Nat := 8513 :: root175Tail25
def root175Tail23 : List Nat := 8429 :: root175Tail24
def root175Tail22 : List Nat := 8117 :: root175Tail23
def root175Tail21 : List Nat := 7589 :: root175Tail22
def root175Tail20 : List Nat := 6329 :: root175Tail21
def root175Tail19 : List Nat := 6197 :: root175Tail20
def root175Tail18 : List Nat := 5477 :: root175Tail19
def root175Tail17 : List Nat := 5147 :: root175Tail18
def root175Tail16 : List Nat := 4973 :: root175Tail17
def root175Tail15 : List Nat := 4733 :: root175Tail16
def root175Tail14 : List Nat := 4649 :: root175Tail15
def root175Tail13 : List Nat := 4463 :: root175Tail14
def root175Tail12 : List Nat := 4391 :: root175Tail13
def root175Tail11 : List Nat := 4139 :: root175Tail12
def root175Tail10 : List Nat := 3767 :: root175Tail11
def root175Tail9 : List Nat := 3257 :: root175Tail10
def root175Tail8 : List Nat := 3167 :: root175Tail9
def root175Tail7 : List Nat := 3041 :: root175Tail8
def root175Tail6 : List Nat := 2711 :: root175Tail7
def root175Tail5 : List Nat := 2213 :: root175Tail6
def root175Tail4 : List Nat := 2027 :: root175Tail5
def root175Tail3 : List Nat := 1619 :: root175Tail4
def root175Tail2 : List Nat := 1493 :: root175Tail3
def root175Tail1 : List Nat := 1229 :: root175Tail2
def root175Tail0 : List Nat := 1151 :: root175Tail1
def root175Candidates0 : List Nat := [19889]
set_option maxRecDepth 8192 in
theorem root175Candidates0_checked : candidates edge 4 1061 1151 root175Tail1 winner.bound = root175Candidates0 := by
  unfold candidates
  have row : edge 1151 = (SortedNatTree.contains FastRows.row1151) := by
    change Data.row1151.contains = (SortedNatTree.contains FastRows.row1151)
    exact FastRows.row1151_checked.symm
  rw [row]
  decide
theorem root175Step0 : scan edge (search edge 3) 4 [1061] 1061 root175Tail0 winner =
    scan edge (search edge 3) 4 [1061] 1061 root175Tail1 winner := by
  rw [root175Tail0, scan, if_neg (by decide), root175Candidates0_checked]
  change scan edge (search edge 3) 4 [1061] 1061 root175Tail1
    (if 3 ≤ root175Candidates0.length then search edge 3 [1061, 1151] root175Candidates0 2212 winner else winner) = _
  rw [if_neg (by decide)]
def root175Candidates1 : List Nat := [17579, 20249]
set_option maxRecDepth 8192 in
theorem root175Candidates1_checked : candidates edge 4 1061 1229 root175Tail2 winner.bound = root175Candidates1 := by
  unfold candidates
  have row : edge 1229 = (SortedNatTree.contains FastRows.row1229) := by
    change Data.row1229.contains = (SortedNatTree.contains FastRows.row1229)
    exact FastRows.row1229_checked.symm
  rw [row]
  decide
theorem root175Step1 : scan edge (search edge 3) 4 [1061] 1061 root175Tail1 winner =
    scan edge (search edge 3) 4 [1061] 1061 root175Tail2 winner := by
  rw [root175Tail1, scan, if_neg (by decide), root175Candidates1_checked]
  change scan edge (search edge 3) 4 [1061] 1061 root175Tail2
    (if 3 ≤ root175Candidates1.length then search edge 3 [1061, 1229] root175Candidates1 2290 winner else winner) = _
  rw [if_neg (by decide)]
def root175Candidates2 : List Nat := [4649]
set_option maxRecDepth 8192 in
theorem root175Candidates2_checked : candidates edge 4 1061 1493 root175Tail3 winner.bound = root175Candidates2 := by
  unfold candidates
  have row : edge 1493 = (SortedNatTree.contains FastRows.row1493) := by
    change Data.row1493.contains = (SortedNatTree.contains FastRows.row1493)
    exact FastRows.row1493_checked.symm
  rw [row]
  decide
theorem root175Step2 : scan edge (search edge 3) 4 [1061] 1061 root175Tail2 winner =
    scan edge (search edge 3) 4 [1061] 1061 root175Tail3 winner := by
  rw [root175Tail2, scan, if_neg (by decide), root175Candidates2_checked]
  change scan edge (search edge 3) 4 [1061] 1061 root175Tail3
    (if 3 ≤ root175Candidates2.length then search edge 3 [1061, 1493] root175Candidates2 2554 winner else winner) = _
  rw [if_neg (by decide)]
def root175Candidates3 : List Nat := [2213, 3257, 6197, 8513, 14303, 23291]
set_option maxRecDepth 8192 in
theorem root175Candidates3_checked : candidates edge 4 1061 1619 root175Tail4 winner.bound = root175Candidates3 := by
  unfold candidates
  have row : edge 1619 = (SortedNatTree.contains FastRows.row1619) := by
    change Data.row1619.contains = (SortedNatTree.contains FastRows.row1619)
    exact FastRows.row1619_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root175Child3 : search edge 3 [1061, 1619] root175Candidates3 2680 winner = winner := by decide
theorem root175Step3 : scan edge (search edge 3) 4 [1061] 1061 root175Tail3 winner =
    scan edge (search edge 3) 4 [1061] 1061 root175Tail4 winner := by
  rw [root175Tail3, scan, if_neg (by decide), root175Candidates3_checked]
  change scan edge (search edge 3) 4 [1061] 1061 root175Tail4
    (if 3 ≤ root175Candidates3.length then search edge 3 [1061, 1619] root175Candidates3 2680 winner else winner) = _
  rw [if_pos (by decide), root175Child3]
def root175Candidates4 : List Nat := [10463]
set_option maxRecDepth 8192 in
theorem root175Candidates4_checked : candidates edge 4 1061 2027 root175Tail5 winner.bound = root175Candidates4 := by
  unfold candidates
  have row : edge 2027 = (SortedNatTree.contains FastRows.row2027) := by
    change Data.row2027.contains = (SortedNatTree.contains FastRows.row2027)
    exact FastRows.row2027_checked.symm
  rw [row]
  decide
theorem root175Step4 : scan edge (search edge 3) 4 [1061] 1061 root175Tail4 winner =
    scan edge (search edge 3) 4 [1061] 1061 root175Tail5 winner := by
  rw [root175Tail4, scan, if_neg (by decide), root175Candidates4_checked]
  change scan edge (search edge 3) 4 [1061] 1061 root175Tail5
    (if 3 ≤ root175Candidates4.length then search edge 3 [1061, 2027] root175Candidates4 3088 winner else winner) = _
  rw [if_neg (by decide)]
def root175Candidates5 : List Nat := [2711]
set_option maxRecDepth 8192 in
theorem root175Candidates5_checked : candidates edge 4 1061 2213 root175Tail6 winner.bound = root175Candidates5 := by
  unfold candidates
  have row : edge 2213 = (SortedNatTree.contains FastRows.row2213) := by
    change Data.row2213.contains = (SortedNatTree.contains FastRows.row2213)
    exact FastRows.row2213_checked.symm
  rw [row]
  decide
theorem root175Step5 : scan edge (search edge 3) 4 [1061] 1061 root175Tail5 winner =
    scan edge (search edge 3) 4 [1061] 1061 root175Tail6 winner := by
  rw [root175Tail5, scan, if_neg (by decide), root175Candidates5_checked]
  change scan edge (search edge 3) 4 [1061] 1061 root175Tail6
    (if 3 ≤ root175Candidates5.length then search edge 3 [1061, 2213] root175Candidates5 3274 winner else winner) = _
  rw [if_neg (by decide)]
def root175Candidates6 : List Nat := [7589, 9923]
set_option maxRecDepth 8192 in
theorem root175Candidates6_checked : candidates edge 4 1061 2711 root175Tail7 winner.bound = root175Candidates6 := by
  unfold candidates
  have row : edge 2711 = (SortedNatTree.contains FastRows.row2711) := by
    change Data.row2711.contains = (SortedNatTree.contains FastRows.row2711)
    exact FastRows.row2711_checked.symm
  rw [row]
  decide
theorem root175Step6 : scan edge (search edge 3) 4 [1061] 1061 root175Tail6 winner =
    scan edge (search edge 3) 4 [1061] 1061 root175Tail7 winner := by
  rw [root175Tail6, scan, if_neg (by decide), root175Candidates6_checked]
  change scan edge (search edge 3) 4 [1061] 1061 root175Tail7
    (if 3 ≤ root175Candidates6.length then search edge 3 [1061, 2711] root175Candidates6 3772 winner else winner) = _
  rw [if_neg (by decide)]
def root175Candidates7 : List Nat := [20249]
set_option maxRecDepth 8192 in
theorem root175Candidates7_checked : candidates edge 4 1061 3041 root175Tail8 winner.bound = root175Candidates7 := by
  unfold candidates
  have row : edge 3041 = (SortedNatTree.contains FastRows.row3041) := by
    change Data.row3041.contains = (SortedNatTree.contains FastRows.row3041)
    exact FastRows.row3041_checked.symm
  rw [row]
  decide
theorem root175Step7 : scan edge (search edge 3) 4 [1061] 1061 root175Tail7 winner =
    scan edge (search edge 3) 4 [1061] 1061 root175Tail8 winner := by
  rw [root175Tail7, scan, if_neg (by decide), root175Candidates7_checked]
  change scan edge (search edge 3) 4 [1061] 1061 root175Tail8
    (if 3 ≤ root175Candidates7.length then search edge 3 [1061, 3041] root175Candidates7 4102 winner else winner) = _
  rw [if_neg (by decide)]
def root175Candidates8 : List Nat := [10463, 13967]
set_option maxRecDepth 8192 in
theorem root175Candidates8_checked : candidates edge 4 1061 3167 root175Tail9 winner.bound = root175Candidates8 := by
  unfold candidates
  have row : edge 3167 = (SortedNatTree.contains FastRows.row3167) := by
    change Data.row3167.contains = (SortedNatTree.contains FastRows.row3167)
    exact FastRows.row3167_checked.symm
  rw [row]
  decide
theorem root175Step8 : scan edge (search edge 3) 4 [1061] 1061 root175Tail8 winner =
    scan edge (search edge 3) 4 [1061] 1061 root175Tail9 winner := by
  rw [root175Tail8, scan, if_neg (by decide), root175Candidates8_checked]
  change scan edge (search edge 3) 4 [1061] 1061 root175Tail9
    (if 3 ≤ root175Candidates8.length then search edge 3 [1061, 3167] root175Candidates8 4228 winner else winner) = _
  rw [if_neg (by decide)]
def root175Candidates9 : List Nat := [4649]
set_option maxRecDepth 8192 in
theorem root175Candidates9_checked : candidates edge 4 1061 3257 root175Tail10 winner.bound = root175Candidates9 := by
  unfold candidates
  have row : edge 3257 = (SortedNatTree.contains FastRows.row3257) := by
    change Data.row3257.contains = (SortedNatTree.contains FastRows.row3257)
    exact FastRows.row3257_checked.symm
  rw [row]
  decide
theorem root175Step9 : scan edge (search edge 3) 4 [1061] 1061 root175Tail9 winner =
    scan edge (search edge 3) 4 [1061] 1061 root175Tail10 winner := by
  rw [root175Tail9, scan, if_neg (by decide), root175Candidates9_checked]
  change scan edge (search edge 3) 4 [1061] 1061 root175Tail10
    (if 3 ≤ root175Candidates9.length then search edge 3 [1061, 3257] root175Candidates9 4318 winner else winner) = _
  rw [if_neg (by decide)]
def root175Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root175Candidates10_checked : candidates edge 4 1061 3767 root175Tail11 winner.bound = root175Candidates10 := by
  unfold candidates
  have row : edge 3767 = (SortedNatTree.contains FastRows.row3767) := by
    change Data.row3767.contains = (SortedNatTree.contains FastRows.row3767)
    exact FastRows.row3767_checked.symm
  rw [row]
  decide
theorem root175Step10 : scan edge (search edge 3) 4 [1061] 1061 root175Tail10 winner =
    scan edge (search edge 3) 4 [1061] 1061 root175Tail11 winner := by
  rw [root175Tail10, scan, if_neg (by decide), root175Candidates10_checked]
  change scan edge (search edge 3) 4 [1061] 1061 root175Tail11
    (if 3 ≤ root175Candidates10.length then search edge 3 [1061, 3767] root175Candidates10 4828 winner else winner) = _
  rw [if_neg (by decide)]
def root175Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root175Candidates11_checked : candidates edge 4 1061 4139 root175Tail12 winner.bound = root175Candidates11 := by
  unfold candidates
  have row : edge 4139 = (SortedNatTree.contains FastRows.row4139) := by
    change Data.row4139.contains = (SortedNatTree.contains FastRows.row4139)
    exact FastRows.row4139_checked.symm
  rw [row]
  decide
theorem root175Step11 : scan edge (search edge 3) 4 [1061] 1061 root175Tail11 winner =
    scan edge (search edge 3) 4 [1061] 1061 root175Tail12 winner := by
  rw [root175Tail11, scan, if_neg (by decide), root175Candidates11_checked]
  change scan edge (search edge 3) 4 [1061] 1061 root175Tail12
    (if 3 ≤ root175Candidates11.length then search edge 3 [1061, 4139] root175Candidates11 5200 winner else winner) = _
  rw [if_neg (by decide)]
def root175Candidates12 : List Nat := [18587]
set_option maxRecDepth 8192 in
theorem root175Candidates12_checked : candidates edge 4 1061 4391 root175Tail13 winner.bound = root175Candidates12 := by
  unfold candidates
  have row : edge 4391 = (SortedNatTree.contains FastRows.row4391) := by
    change Data.row4391.contains = (SortedNatTree.contains FastRows.row4391)
    exact FastRows.row4391_checked.symm
  rw [row]
  decide
theorem root175Step12 : scan edge (search edge 3) 4 [1061] 1061 root175Tail12 winner =
    scan edge (search edge 3) 4 [1061] 1061 root175Tail13 winner := by
  rw [root175Tail12, scan, if_neg (by decide), root175Candidates12_checked]
  change scan edge (search edge 3) 4 [1061] 1061 root175Tail13
    (if 3 ≤ root175Candidates12.length then search edge 3 [1061, 4391] root175Candidates12 5452 winner else winner) = _
  rw [if_neg (by decide)]
def root175Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root175Candidates13_checked : candidates edge 4 1061 4463 root175Tail14 winner.bound = root175Candidates13 := by
  unfold candidates
  have row : edge 4463 = (SortedNatTree.contains FastRows.row4463) := by
    change Data.row4463.contains = (SortedNatTree.contains FastRows.row4463)
    exact FastRows.row4463_checked.symm
  rw [row]
  decide
theorem root175Step13 : scan edge (search edge 3) 4 [1061] 1061 root175Tail13 winner =
    scan edge (search edge 3) 4 [1061] 1061 root175Tail14 winner := by
  rw [root175Tail13, scan, if_neg (by decide), root175Candidates13_checked]
  change scan edge (search edge 3) 4 [1061] 1061 root175Tail14
    (if 3 ≤ root175Candidates13.length then search edge 3 [1061, 4463] root175Candidates13 5524 winner else winner) = _
  rw [if_neg (by decide)]
def root175Candidates14 : List Nat := [8117]
set_option maxRecDepth 8192 in
theorem root175Candidates14_checked : candidates edge 4 1061 4649 root175Tail15 winner.bound = root175Candidates14 := by
  unfold candidates
  have row : edge 4649 = (SortedNatTree.contains FastRows.row4649) := by
    change Data.row4649.contains = (SortedNatTree.contains FastRows.row4649)
    exact FastRows.row4649_checked.symm
  rw [row]
  decide
theorem root175Step14 : scan edge (search edge 3) 4 [1061] 1061 root175Tail14 winner =
    scan edge (search edge 3) 4 [1061] 1061 root175Tail15 winner := by
  rw [root175Tail14, scan, if_neg (by decide), root175Candidates14_checked]
  change scan edge (search edge 3) 4 [1061] 1061 root175Tail15
    (if 3 ≤ root175Candidates14.length then search edge 3 [1061, 4649] root175Candidates14 5710 winner else winner) = _
  rw [if_neg (by decide)]
def root175Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root175Candidates15_checked : candidates edge 4 1061 4733 root175Tail16 winner.bound = root175Candidates15 := by
  unfold candidates
  have row : edge 4733 = (SortedNatTree.contains FastRows.row4733) := by
    change Data.row4733.contains = (SortedNatTree.contains FastRows.row4733)
    exact FastRows.row4733_checked.symm
  rw [row]
  decide
theorem root175Step15 : scan edge (search edge 3) 4 [1061] 1061 root175Tail15 winner =
    scan edge (search edge 3) 4 [1061] 1061 root175Tail16 winner := by
  rw [root175Tail15, scan, if_neg (by decide), root175Candidates15_checked]
  change scan edge (search edge 3) 4 [1061] 1061 root175Tail16
    (if 3 ≤ root175Candidates15.length then search edge 3 [1061, 4733] root175Candidates15 5794 winner else winner) = _
  rw [if_neg (by decide)]
def root175Candidates16 : List Nat := [17783, 18587]
set_option maxRecDepth 8192 in
theorem root175Candidates16_checked : candidates edge 4 1061 4973 root175Tail17 winner.bound = root175Candidates16 := by
  unfold candidates
  have row : edge 4973 = (SortedNatTree.contains FastRows.row4973) := by
    change Data.row4973.contains = (SortedNatTree.contains FastRows.row4973)
    exact FastRows.row4973_checked.symm
  rw [row]
  decide
theorem root175Step16 : scan edge (search edge 3) 4 [1061] 1061 root175Tail16 winner =
    scan edge (search edge 3) 4 [1061] 1061 root175Tail17 winner := by
  rw [root175Tail16, scan, if_neg (by decide), root175Candidates16_checked]
  change scan edge (search edge 3) 4 [1061] 1061 root175Tail17
    (if 3 ≤ root175Candidates16.length then search edge 3 [1061, 4973] root175Candidates16 6034 winner else winner) = _
  rw [if_neg (by decide)]
def root175Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root175Candidates17_checked : candidates edge 4 1061 5147 root175Tail18 winner.bound = root175Candidates17 := by
  unfold candidates
  have row : edge 5147 = (SortedNatTree.contains FastRows.row5147) := by
    change Data.row5147.contains = (SortedNatTree.contains FastRows.row5147)
    exact FastRows.row5147_checked.symm
  rw [row]
  decide
theorem root175Step17 : scan edge (search edge 3) 4 [1061] 1061 root175Tail17 winner =
    scan edge (search edge 3) 4 [1061] 1061 root175Tail18 winner := by
  rw [root175Tail17, scan, if_neg (by decide), root175Candidates17_checked]
  change scan edge (search edge 3) 4 [1061] 1061 root175Tail18
    (if 3 ≤ root175Candidates17.length then search edge 3 [1061, 5147] root175Candidates17 6208 winner else winner) = _
  rw [if_neg (by decide)]
def root175Candidates18 : List Nat := []
set_option maxRecDepth 8192 in
theorem root175Candidates18_checked : candidates edge 4 1061 5477 root175Tail19 winner.bound = root175Candidates18 := by
  unfold candidates
  have row : edge 5477 = (SortedNatTree.contains FastRows.row5477) := by
    change Data.row5477.contains = (SortedNatTree.contains FastRows.row5477)
    exact FastRows.row5477_checked.symm
  rw [row]
  decide
theorem root175Step18 : scan edge (search edge 3) 4 [1061] 1061 root175Tail18 winner =
    scan edge (search edge 3) 4 [1061] 1061 root175Tail19 winner := by
  rw [root175Tail18, scan, if_neg (by decide), root175Candidates18_checked]
  change scan edge (search edge 3) 4 [1061] 1061 root175Tail19
    (if 3 ≤ root175Candidates18.length then search edge 3 [1061, 5477] root175Candidates18 6538 winner else winner) = _
  rw [if_neg (by decide)]
def root175Candidates19 : List Nat := [9209]
set_option maxRecDepth 8192 in
theorem root175Candidates19_checked : candidates edge 4 1061 6197 root175Tail20 winner.bound = root175Candidates19 := by
  unfold candidates
  have row : edge 6197 = (SortedNatTree.contains FastRows.row6197) := by
    change Data.row6197.contains = (SortedNatTree.contains FastRows.row6197)
    exact FastRows.row6197_checked.symm
  rw [row]
  decide
theorem root175Step19 : scan edge (search edge 3) 4 [1061] 1061 root175Tail19 winner =
    scan edge (search edge 3) 4 [1061] 1061 root175Tail20 winner := by
  rw [root175Tail19, scan, if_neg (by decide), root175Candidates19_checked]
  change scan edge (search edge 3) 4 [1061] 1061 root175Tail20
    (if 3 ≤ root175Candidates19.length then search edge 3 [1061, 6197] root175Candidates19 7258 winner else winner) = _
  rw [if_neg (by decide)]
theorem root175Scan20 : scan edge (search edge 3) 4 [1061] 1061 root175Tail20 winner = winner := by rw [root175Tail20, scan, if_pos (by decide)]
theorem root175Scan19 : scan edge (search edge 3) 4 [1061] 1061 root175Tail19 winner = winner :=
  (root175Step19).trans root175Scan20
theorem root175Scan18 : scan edge (search edge 3) 4 [1061] 1061 root175Tail18 winner = winner :=
  (root175Step18).trans root175Scan19
theorem root175Scan17 : scan edge (search edge 3) 4 [1061] 1061 root175Tail17 winner = winner :=
  (root175Step17).trans root175Scan18
theorem root175Scan16 : scan edge (search edge 3) 4 [1061] 1061 root175Tail16 winner = winner :=
  (root175Step16).trans root175Scan17
theorem root175Scan15 : scan edge (search edge 3) 4 [1061] 1061 root175Tail15 winner = winner :=
  (root175Step15).trans root175Scan16
theorem root175Scan14 : scan edge (search edge 3) 4 [1061] 1061 root175Tail14 winner = winner :=
  (root175Step14).trans root175Scan15
theorem root175Scan13 : scan edge (search edge 3) 4 [1061] 1061 root175Tail13 winner = winner :=
  (root175Step13).trans root175Scan14
theorem root175Scan12 : scan edge (search edge 3) 4 [1061] 1061 root175Tail12 winner = winner :=
  (root175Step12).trans root175Scan13
theorem root175Scan11 : scan edge (search edge 3) 4 [1061] 1061 root175Tail11 winner = winner :=
  (root175Step11).trans root175Scan12
theorem root175Scan10 : scan edge (search edge 3) 4 [1061] 1061 root175Tail10 winner = winner :=
  (root175Step10).trans root175Scan11
theorem root175Scan9 : scan edge (search edge 3) 4 [1061] 1061 root175Tail9 winner = winner :=
  (root175Step9).trans root175Scan10
theorem root175Scan8 : scan edge (search edge 3) 4 [1061] 1061 root175Tail8 winner = winner :=
  (root175Step8).trans root175Scan9
theorem root175Scan7 : scan edge (search edge 3) 4 [1061] 1061 root175Tail7 winner = winner :=
  (root175Step7).trans root175Scan8
theorem root175Scan6 : scan edge (search edge 3) 4 [1061] 1061 root175Tail6 winner = winner :=
  (root175Step6).trans root175Scan7
theorem root175Scan5 : scan edge (search edge 3) 4 [1061] 1061 root175Tail5 winner = winner :=
  (root175Step5).trans root175Scan6
theorem root175Scan4 : scan edge (search edge 3) 4 [1061] 1061 root175Tail4 winner = winner :=
  (root175Step4).trans root175Scan5
theorem root175Scan3 : scan edge (search edge 3) 4 [1061] 1061 root175Tail3 winner = winner :=
  (root175Step3).trans root175Scan4
theorem root175Scan2 : scan edge (search edge 3) 4 [1061] 1061 root175Tail2 winner = winner :=
  (root175Step2).trans root175Scan3
theorem root175Scan1 : scan edge (search edge 3) 4 [1061] 1061 root175Tail1 winner = winner :=
  (root175Step1).trans root175Scan2
theorem root175Scan0 : scan edge (search edge 3) 4 [1061] 1061 root175Tail0 winner = winner :=
  (root175Step0).trans root175Scan1
set_option maxRecDepth 4096 in
theorem root175Branch : search edge 4 [1061] root175Candidates 1061 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1061] 1061 root175Tail0 winner = winner
  exact root175Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
