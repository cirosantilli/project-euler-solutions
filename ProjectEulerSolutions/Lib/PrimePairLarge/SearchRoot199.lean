import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root199Candidates : List Nat := Data.row1231
set_option maxRecDepth 8192 in
theorem root199Candidates_checked : candidates edge 5 0 1231 tail200 winner.bound = root199Candidates := by
  apply candidates_eq_row edge 5 0 1231 tail200 root199Candidates winner.bound tail200_ordered
  · decide
  · decide
  · funext q
    rfl
def root199Tail50 : List Nat := []
def root199Tail49 : List Nat := 24373 :: root199Tail50
def root199Tail48 : List Nat := 20521 :: root199Tail49
def root199Tail47 : List Nat := 20161 :: root199Tail48
def root199Tail46 : List Nat := 20047 :: root199Tail47
def root199Tail45 : List Nat := 19333 :: root199Tail46
def root199Tail44 : List Nat := 19309 :: root199Tail45
def root199Tail43 : List Nat := 18397 :: root199Tail44
def root199Tail42 : List Nat := 17977 :: root199Tail43
def root199Tail41 : List Nat := 17713 :: root199Tail42
def root199Tail40 : List Nat := 16573 :: root199Tail41
def root199Tail39 : List Nat := 16111 :: root199Tail40
def root199Tail38 : List Nat := 15901 :: root199Tail39
def root199Tail37 : List Nat := 14683 :: root199Tail38
def root199Tail36 : List Nat := 13999 :: root199Tail37
def root199Tail35 : List Nat := 13567 :: root199Tail36
def root199Tail34 : List Nat := 13147 :: root199Tail35
def root199Tail33 : List Nat := 13063 :: root199Tail34
def root199Tail32 : List Nat := 12907 :: root199Tail33
def root199Tail31 : List Nat := 12853 :: root199Tail32
def root199Tail30 : List Nat := 11731 :: root199Tail31
def root199Tail29 : List Nat := 11719 :: root199Tail30
def root199Tail28 : List Nat := 11257 :: root199Tail29
def root199Tail27 : List Nat := 11113 :: root199Tail28
def root199Tail26 : List Nat := 9337 :: root199Tail27
def root199Tail25 : List Nat := 8431 :: root199Tail26
def root199Tail24 : List Nat := 8221 :: root199Tail25
def root199Tail23 : List Nat := 7309 :: root199Tail24
def root199Tail22 : List Nat := 7297 :: root199Tail23
def root199Tail21 : List Nat := 6949 :: root199Tail22
def root199Tail20 : List Nat := 6529 :: root199Tail21
def root199Tail19 : List Nat := 6229 :: root199Tail20
def root199Tail18 : List Nat := 6091 :: root199Tail19
def root199Tail17 : List Nat := 6037 :: root199Tail18
def root199Tail16 : List Nat := 5881 :: root199Tail17
def root199Tail15 : List Nat := 5659 :: root199Tail16
def root199Tail14 : List Nat := 5647 :: root199Tail15
def root199Tail13 : List Nat := 5449 :: root199Tail14
def root199Tail12 : List Nat := 5323 :: root199Tail13
def root199Tail11 : List Nat := 4789 :: root199Tail12
def root199Tail10 : List Nat := 4597 :: root199Tail11
def root199Tail9 : List Nat := 4507 :: root199Tail10
def root199Tail8 : List Nat := 4483 :: root199Tail9
def root199Tail7 : List Nat := 4327 :: root199Tail8
def root199Tail6 : List Nat := 4243 :: root199Tail7
def root199Tail5 : List Nat := 4231 :: root199Tail6
def root199Tail4 : List Nat := 2917 :: root199Tail5
def root199Tail3 : List Nat := 2689 :: root199Tail4
def root199Tail2 : List Nat := 2143 :: root199Tail3
def root199Tail1 : List Nat := 1951 :: root199Tail2
def root199Tail0 : List Nat := 1777 :: root199Tail1
def root199Candidates0 : List Nat := [14683]
set_option maxRecDepth 8192 in
theorem root199Candidates0_checked : candidates edge 4 1231 1777 root199Tail1 winner.bound = root199Candidates0 := by
  unfold candidates
  have row : edge 1777 = (SortedNatTree.contains FastRows.row1777) := by
    change Data.row1777.contains = (SortedNatTree.contains FastRows.row1777)
    exact FastRows.row1777_checked.symm
  rw [row]
  decide
theorem root199Step0 : scan edge (search edge 3) 4 [1231] 1231 root199Tail0 winner =
    scan edge (search edge 3) 4 [1231] 1231 root199Tail1 winner := by
  rw [root199Tail0, scan, if_neg (by decide), root199Candidates0_checked]
  change scan edge (search edge 3) 4 [1231] 1231 root199Tail1
    (if 3 ≤ root199Candidates0.length then search edge 3 [1231, 1777] root199Candidates0 3008 winner else winner) = _
  rw [if_neg (by decide)]
def root199Candidates1 : List Nat := [6037, 20047]
set_option maxRecDepth 8192 in
theorem root199Candidates1_checked : candidates edge 4 1231 1951 root199Tail2 winner.bound = root199Candidates1 := by
  unfold candidates
  have row : edge 1951 = (SortedNatTree.contains FastRows.row1951) := by
    change Data.row1951.contains = (SortedNatTree.contains FastRows.row1951)
    exact FastRows.row1951_checked.symm
  rw [row]
  decide
theorem root199Step1 : scan edge (search edge 3) 4 [1231] 1231 root199Tail1 winner =
    scan edge (search edge 3) 4 [1231] 1231 root199Tail2 winner := by
  rw [root199Tail1, scan, if_neg (by decide), root199Candidates1_checked]
  change scan edge (search edge 3) 4 [1231] 1231 root199Tail2
    (if 3 ≤ root199Candidates1.length then search edge 3 [1231, 1951] root199Candidates1 3182 winner else winner) = _
  rw [if_neg (by decide)]
def root199Candidates2 : List Nat := [6229, 13999, 20521]
set_option maxRecDepth 8192 in
theorem root199Candidates2_checked : candidates edge 4 1231 2143 root199Tail3 winner.bound = root199Candidates2 := by
  unfold candidates
  have row : edge 2143 = (SortedNatTree.contains FastRows.row2143) := by
    change Data.row2143.contains = (SortedNatTree.contains FastRows.row2143)
    exact FastRows.row2143_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root199Child2 : search edge 3 [1231, 2143] root199Candidates2 3374 winner = winner := by decide
theorem root199Step2 : scan edge (search edge 3) 4 [1231] 1231 root199Tail2 winner =
    scan edge (search edge 3) 4 [1231] 1231 root199Tail3 winner := by
  rw [root199Tail2, scan, if_neg (by decide), root199Candidates2_checked]
  change scan edge (search edge 3) 4 [1231] 1231 root199Tail3
    (if 3 ≤ root199Candidates2.length then search edge 3 [1231, 2143] root199Candidates2 3374 winner else winner) = _
  rw [if_pos (by decide), root199Child2]
def root199Candidates3 : List Nat := [4507, 12907, 19309]
set_option maxRecDepth 8192 in
theorem root199Candidates3_checked : candidates edge 4 1231 2689 root199Tail4 winner.bound = root199Candidates3 := by
  unfold candidates
  have row : edge 2689 = (SortedNatTree.contains FastRows.row2689) := by
    change Data.row2689.contains = (SortedNatTree.contains FastRows.row2689)
    exact FastRows.row2689_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root199Child3 : search edge 3 [1231, 2689] root199Candidates3 3920 winner = winner := by decide
theorem root199Step3 : scan edge (search edge 3) 4 [1231] 1231 root199Tail3 winner =
    scan edge (search edge 3) 4 [1231] 1231 root199Tail4 winner := by
  rw [root199Tail3, scan, if_neg (by decide), root199Candidates3_checked]
  change scan edge (search edge 3) 4 [1231] 1231 root199Tail4
    (if 3 ≤ root199Candidates3.length then search edge 3 [1231, 2689] root199Candidates3 3920 winner else winner) = _
  rw [if_pos (by decide), root199Child3]
def root199Candidates4 : List Nat := [13567]
set_option maxRecDepth 8192 in
theorem root199Candidates4_checked : candidates edge 4 1231 2917 root199Tail5 winner.bound = root199Candidates4 := by
  unfold candidates
  have row : edge 2917 = (SortedNatTree.contains FastRows.row2917) := by
    change Data.row2917.contains = (SortedNatTree.contains FastRows.row2917)
    exact FastRows.row2917_checked.symm
  rw [row]
  decide
theorem root199Step4 : scan edge (search edge 3) 4 [1231] 1231 root199Tail4 winner =
    scan edge (search edge 3) 4 [1231] 1231 root199Tail5 winner := by
  rw [root199Tail4, scan, if_neg (by decide), root199Candidates4_checked]
  change scan edge (search edge 3) 4 [1231] 1231 root199Tail5
    (if 3 ≤ root199Candidates4.length then search edge 3 [1231, 2917] root199Candidates4 4148 winner else winner) = _
  rw [if_neg (by decide)]
def root199Candidates5 : List Nat := [6037, 6529]
set_option maxRecDepth 8192 in
theorem root199Candidates5_checked : candidates edge 4 1231 4231 root199Tail6 winner.bound = root199Candidates5 := by
  unfold candidates
  have row : edge 4231 = (SortedNatTree.contains FastRows.row4231) := by
    change Data.row4231.contains = (SortedNatTree.contains FastRows.row4231)
    exact FastRows.row4231_checked.symm
  rw [row]
  decide
theorem root199Step5 : scan edge (search edge 3) 4 [1231] 1231 root199Tail5 winner =
    scan edge (search edge 3) 4 [1231] 1231 root199Tail6 winner := by
  rw [root199Tail5, scan, if_neg (by decide), root199Candidates5_checked]
  change scan edge (search edge 3) 4 [1231] 1231 root199Tail6
    (if 3 ≤ root199Candidates5.length then search edge 3 [1231, 4231] root199Candidates5 5462 winner else winner) = _
  rw [if_neg (by decide)]
def root199Candidates6 : List Nat := [13063, 19309]
set_option maxRecDepth 8192 in
theorem root199Candidates6_checked : candidates edge 4 1231 4243 root199Tail7 winner.bound = root199Candidates6 := by
  unfold candidates
  have row : edge 4243 = (SortedNatTree.contains FastRows.row4243) := by
    change Data.row4243.contains = (SortedNatTree.contains FastRows.row4243)
    exact FastRows.row4243_checked.symm
  rw [row]
  decide
theorem root199Step6 : scan edge (search edge 3) 4 [1231] 1231 root199Tail6 winner =
    scan edge (search edge 3) 4 [1231] 1231 root199Tail7 winner := by
  rw [root199Tail6, scan, if_neg (by decide), root199Candidates6_checked]
  change scan edge (search edge 3) 4 [1231] 1231 root199Tail7
    (if 3 ≤ root199Candidates6.length then search edge 3 [1231, 4243] root199Candidates6 5474 winner else winner) = _
  rw [if_neg (by decide)]
def root199Candidates7 : List Nat := [9337, 13063]
set_option maxRecDepth 8192 in
theorem root199Candidates7_checked : candidates edge 4 1231 4327 root199Tail8 winner.bound = root199Candidates7 := by
  unfold candidates
  have row : edge 4327 = (SortedNatTree.contains FastRows.row4327) := by
    change Data.row4327.contains = (SortedNatTree.contains FastRows.row4327)
    exact FastRows.row4327_checked.symm
  rw [row]
  decide
theorem root199Step7 : scan edge (search edge 3) 4 [1231] 1231 root199Tail7 winner =
    scan edge (search edge 3) 4 [1231] 1231 root199Tail8 winner := by
  rw [root199Tail7, scan, if_neg (by decide), root199Candidates7_checked]
  change scan edge (search edge 3) 4 [1231] 1231 root199Tail8
    (if 3 ≤ root199Candidates7.length then search edge 3 [1231, 4327] root199Candidates7 5558 winner else winner) = _
  rw [if_neg (by decide)]
def root199Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root199Candidates8_checked : candidates edge 4 1231 4483 root199Tail9 winner.bound = root199Candidates8 := by
  unfold candidates
  have row : edge 4483 = (SortedNatTree.contains FastRows.row4483) := by
    change Data.row4483.contains = (SortedNatTree.contains FastRows.row4483)
    exact FastRows.row4483_checked.symm
  rw [row]
  decide
theorem root199Step8 : scan edge (search edge 3) 4 [1231] 1231 root199Tail8 winner =
    scan edge (search edge 3) 4 [1231] 1231 root199Tail9 winner := by
  rw [root199Tail8, scan, if_neg (by decide), root199Candidates8_checked]
  change scan edge (search edge 3) 4 [1231] 1231 root199Tail9
    (if 3 ≤ root199Candidates8.length then search edge 3 [1231, 4483] root199Candidates8 5714 winner else winner) = _
  rw [if_neg (by decide)]
def root199Candidates9 : List Nat := [4789, 5881, 6091, 11257, 12907, 19309]
set_option maxRecDepth 8192 in
theorem root199Candidates9_checked : candidates edge 4 1231 4507 root199Tail10 winner.bound = root199Candidates9 := by
  unfold candidates
  have row : edge 4507 = (SortedNatTree.contains FastRows.row4507) := by
    change Data.row4507.contains = (SortedNatTree.contains FastRows.row4507)
    exact FastRows.row4507_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root199Child9 : search edge 3 [1231, 4507] root199Candidates9 5738 winner = winner := by decide
theorem root199Step9 : scan edge (search edge 3) 4 [1231] 1231 root199Tail9 winner =
    scan edge (search edge 3) 4 [1231] 1231 root199Tail10 winner := by
  rw [root199Tail9, scan, if_neg (by decide), root199Candidates9_checked]
  change scan edge (search edge 3) 4 [1231] 1231 root199Tail10
    (if 3 ≤ root199Candidates9.length then search edge 3 [1231, 4507] root199Candidates9 5738 winner else winner) = _
  rw [if_pos (by decide), root199Child9]
def root199Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root199Candidates10_checked : candidates edge 4 1231 4597 root199Tail11 winner.bound = root199Candidates10 := by
  unfold candidates
  have row : edge 4597 = (SortedNatTree.contains FastRows.row4597) := by
    change Data.row4597.contains = (SortedNatTree.contains FastRows.row4597)
    exact FastRows.row4597_checked.symm
  rw [row]
  decide
theorem root199Step10 : scan edge (search edge 3) 4 [1231] 1231 root199Tail10 winner =
    scan edge (search edge 3) 4 [1231] 1231 root199Tail11 winner := by
  rw [root199Tail10, scan, if_neg (by decide), root199Candidates10_checked]
  change scan edge (search edge 3) 4 [1231] 1231 root199Tail11
    (if 3 ≤ root199Candidates10.length then search edge 3 [1231, 4597] root199Candidates10 5828 winner else winner) = _
  rw [if_neg (by decide)]
def root199Candidates11 : List Nat := [11113, 13147]
set_option maxRecDepth 8192 in
theorem root199Candidates11_checked : candidates edge 4 1231 4789 root199Tail12 winner.bound = root199Candidates11 := by
  unfold candidates
  have row : edge 4789 = (SortedNatTree.contains FastRows.row4789) := by
    change Data.row4789.contains = (SortedNatTree.contains FastRows.row4789)
    exact FastRows.row4789_checked.symm
  rw [row]
  decide
theorem root199Step11 : scan edge (search edge 3) 4 [1231] 1231 root199Tail11 winner =
    scan edge (search edge 3) 4 [1231] 1231 root199Tail12 winner := by
  rw [root199Tail11, scan, if_neg (by decide), root199Candidates11_checked]
  change scan edge (search edge 3) 4 [1231] 1231 root199Tail12
    (if 3 ≤ root199Candidates11.length then search edge 3 [1231, 4789] root199Candidates11 6020 winner else winner) = _
  rw [if_neg (by decide)]
def root199Candidates12 : List Nat := [5659, 6037, 16111]
set_option maxRecDepth 8192 in
theorem root199Candidates12_checked : candidates edge 4 1231 5323 root199Tail13 winner.bound = root199Candidates12 := by
  unfold candidates
  have row : edge 5323 = (SortedNatTree.contains FastRows.row5323) := by
    change Data.row5323.contains = (SortedNatTree.contains FastRows.row5323)
    exact FastRows.row5323_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root199Child12 : search edge 3 [1231, 5323] root199Candidates12 6554 winner = winner := by decide
theorem root199Step12 : scan edge (search edge 3) 4 [1231] 1231 root199Tail12 winner =
    scan edge (search edge 3) 4 [1231] 1231 root199Tail13 winner := by
  rw [root199Tail12, scan, if_neg (by decide), root199Candidates12_checked]
  change scan edge (search edge 3) 4 [1231] 1231 root199Tail13
    (if 3 ≤ root199Candidates12.length then search edge 3 [1231, 5323] root199Candidates12 6554 winner else winner) = _
  rw [if_pos (by decide), root199Child12]
def root199Candidates13 : List Nat := [15901]
set_option maxRecDepth 8192 in
theorem root199Candidates13_checked : candidates edge 4 1231 5449 root199Tail14 winner.bound = root199Candidates13 := by
  unfold candidates
  have row : edge 5449 = (SortedNatTree.contains FastRows.row5449) := by
    change Data.row5449.contains = (SortedNatTree.contains FastRows.row5449)
    exact FastRows.row5449_checked.symm
  rw [row]
  decide
theorem root199Step13 : scan edge (search edge 3) 4 [1231] 1231 root199Tail13 winner =
    scan edge (search edge 3) 4 [1231] 1231 root199Tail14 winner := by
  rw [root199Tail13, scan, if_neg (by decide), root199Candidates13_checked]
  change scan edge (search edge 3) 4 [1231] 1231 root199Tail14
    (if 3 ≤ root199Candidates13.length then search edge 3 [1231, 5449] root199Candidates13 6680 winner else winner) = _
  rw [if_neg (by decide)]
def root199Candidates14 : List Nat := [5659, 18397]
set_option maxRecDepth 8192 in
theorem root199Candidates14_checked : candidates edge 4 1231 5647 root199Tail15 winner.bound = root199Candidates14 := by
  unfold candidates
  have row : edge 5647 = (SortedNatTree.contains FastRows.row5647) := by
    change Data.row5647.contains = (SortedNatTree.contains FastRows.row5647)
    exact FastRows.row5647_checked.symm
  rw [row]
  decide
theorem root199Step14 : scan edge (search edge 3) 4 [1231] 1231 root199Tail14 winner =
    scan edge (search edge 3) 4 [1231] 1231 root199Tail15 winner := by
  rw [root199Tail14, scan, if_neg (by decide), root199Candidates14_checked]
  change scan edge (search edge 3) 4 [1231] 1231 root199Tail15
    (if 3 ≤ root199Candidates14.length then search edge 3 [1231, 5647] root199Candidates14 6878 winner else winner) = _
  rw [if_neg (by decide)]
def root199Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root199Candidates15_checked : candidates edge 4 1231 5659 root199Tail16 winner.bound = root199Candidates15 := by
  unfold candidates
  have row : edge 5659 = (SortedNatTree.contains FastRows.row5659) := by
    change Data.row5659.contains = (SortedNatTree.contains FastRows.row5659)
    exact FastRows.row5659_checked.symm
  rw [row]
  decide
theorem root199Step15 : scan edge (search edge 3) 4 [1231] 1231 root199Tail15 winner =
    scan edge (search edge 3) 4 [1231] 1231 root199Tail16 winner := by
  rw [root199Tail15, scan, if_neg (by decide), root199Candidates15_checked]
  change scan edge (search edge 3) 4 [1231] 1231 root199Tail16
    (if 3 ≤ root199Candidates15.length then search edge 3 [1231, 5659] root199Candidates15 6890 winner else winner) = _
  rw [if_neg (by decide)]
def root199Candidates16 : List Nat := []
set_option maxRecDepth 8192 in
theorem root199Candidates16_checked : candidates edge 4 1231 5881 root199Tail17 winner.bound = root199Candidates16 := by
  unfold candidates
  have row : edge 5881 = (SortedNatTree.contains FastRows.row5881) := by
    change Data.row5881.contains = (SortedNatTree.contains FastRows.row5881)
    exact FastRows.row5881_checked.symm
  rw [row]
  decide
theorem root199Step16 : scan edge (search edge 3) 4 [1231] 1231 root199Tail16 winner =
    scan edge (search edge 3) 4 [1231] 1231 root199Tail17 winner := by
  rw [root199Tail16, scan, if_neg (by decide), root199Candidates16_checked]
  change scan edge (search edge 3) 4 [1231] 1231 root199Tail17
    (if 3 ≤ root199Candidates16.length then search edge 3 [1231, 5881] root199Candidates16 7112 winner else winner) = _
  rw [if_neg (by decide)]
def root199Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root199Candidates17_checked : candidates edge 4 1231 6037 root199Tail18 winner.bound = root199Candidates17 := by
  unfold candidates
  have row : edge 6037 = (SortedNatTree.contains FastRows.row6037) := by
    change Data.row6037.contains = (SortedNatTree.contains FastRows.row6037)
    exact FastRows.row6037_checked.symm
  rw [row]
  decide
theorem root199Step17 : scan edge (search edge 3) 4 [1231] 1231 root199Tail17 winner =
    scan edge (search edge 3) 4 [1231] 1231 root199Tail18 winner := by
  rw [root199Tail17, scan, if_neg (by decide), root199Candidates17_checked]
  change scan edge (search edge 3) 4 [1231] 1231 root199Tail18
    (if 3 ≤ root199Candidates17.length then search edge 3 [1231, 6037] root199Candidates17 7268 winner else winner) = _
  rw [if_neg (by decide)]
def root199Candidates18 : List Nat := []
set_option maxRecDepth 8192 in
theorem root199Candidates18_checked : candidates edge 4 1231 6091 root199Tail19 winner.bound = root199Candidates18 := by
  unfold candidates
  have row : edge 6091 = (SortedNatTree.contains FastRows.row6091) := by
    change Data.row6091.contains = (SortedNatTree.contains FastRows.row6091)
    exact FastRows.row6091_checked.symm
  rw [row]
  decide
theorem root199Step18 : scan edge (search edge 3) 4 [1231] 1231 root199Tail18 winner =
    scan edge (search edge 3) 4 [1231] 1231 root199Tail19 winner := by
  rw [root199Tail18, scan, if_neg (by decide), root199Candidates18_checked]
  change scan edge (search edge 3) 4 [1231] 1231 root199Tail19
    (if 3 ≤ root199Candidates18.length then search edge 3 [1231, 6091] root199Candidates18 7322 winner else winner) = _
  rw [if_neg (by decide)]
theorem root199Scan19 : scan edge (search edge 3) 4 [1231] 1231 root199Tail19 winner = winner := by rw [root199Tail19, scan, if_pos (by decide)]
theorem root199Scan18 : scan edge (search edge 3) 4 [1231] 1231 root199Tail18 winner = winner :=
  (root199Step18).trans root199Scan19
theorem root199Scan17 : scan edge (search edge 3) 4 [1231] 1231 root199Tail17 winner = winner :=
  (root199Step17).trans root199Scan18
theorem root199Scan16 : scan edge (search edge 3) 4 [1231] 1231 root199Tail16 winner = winner :=
  (root199Step16).trans root199Scan17
theorem root199Scan15 : scan edge (search edge 3) 4 [1231] 1231 root199Tail15 winner = winner :=
  (root199Step15).trans root199Scan16
theorem root199Scan14 : scan edge (search edge 3) 4 [1231] 1231 root199Tail14 winner = winner :=
  (root199Step14).trans root199Scan15
theorem root199Scan13 : scan edge (search edge 3) 4 [1231] 1231 root199Tail13 winner = winner :=
  (root199Step13).trans root199Scan14
theorem root199Scan12 : scan edge (search edge 3) 4 [1231] 1231 root199Tail12 winner = winner :=
  (root199Step12).trans root199Scan13
theorem root199Scan11 : scan edge (search edge 3) 4 [1231] 1231 root199Tail11 winner = winner :=
  (root199Step11).trans root199Scan12
theorem root199Scan10 : scan edge (search edge 3) 4 [1231] 1231 root199Tail10 winner = winner :=
  (root199Step10).trans root199Scan11
theorem root199Scan9 : scan edge (search edge 3) 4 [1231] 1231 root199Tail9 winner = winner :=
  (root199Step9).trans root199Scan10
theorem root199Scan8 : scan edge (search edge 3) 4 [1231] 1231 root199Tail8 winner = winner :=
  (root199Step8).trans root199Scan9
theorem root199Scan7 : scan edge (search edge 3) 4 [1231] 1231 root199Tail7 winner = winner :=
  (root199Step7).trans root199Scan8
theorem root199Scan6 : scan edge (search edge 3) 4 [1231] 1231 root199Tail6 winner = winner :=
  (root199Step6).trans root199Scan7
theorem root199Scan5 : scan edge (search edge 3) 4 [1231] 1231 root199Tail5 winner = winner :=
  (root199Step5).trans root199Scan6
theorem root199Scan4 : scan edge (search edge 3) 4 [1231] 1231 root199Tail4 winner = winner :=
  (root199Step4).trans root199Scan5
theorem root199Scan3 : scan edge (search edge 3) 4 [1231] 1231 root199Tail3 winner = winner :=
  (root199Step3).trans root199Scan4
theorem root199Scan2 : scan edge (search edge 3) 4 [1231] 1231 root199Tail2 winner = winner :=
  (root199Step2).trans root199Scan3
theorem root199Scan1 : scan edge (search edge 3) 4 [1231] 1231 root199Tail1 winner = winner :=
  (root199Step1).trans root199Scan2
theorem root199Scan0 : scan edge (search edge 3) 4 [1231] 1231 root199Tail0 winner = winner :=
  (root199Step0).trans root199Scan1
set_option maxRecDepth 4096 in
theorem root199Branch : search edge 4 [1231] root199Candidates 1231 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1231] 1231 root199Tail0 winner = winner
  exact root199Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
