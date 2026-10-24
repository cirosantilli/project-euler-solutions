import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root189Candidates : List Nat := Data.row1163
set_option maxRecDepth 8192 in
theorem root189Candidates_checked : candidates edge 5 0 1163 tail190 winner.bound = root189Candidates := by
  apply candidates_eq_row edge 5 0 1163 tail190 root189Candidates winner.bound tail190_ordered
  · decide
  · decide
  · funext q
    rfl
def root189Tail50 : List Nat := []
def root189Tail49 : List Nat := 24851 :: root189Tail50
def root189Tail48 : List Nat := 23747 :: root189Tail49
def root189Tail47 : List Nat := 23663 :: root189Tail48
def root189Tail46 : List Nat := 23327 :: root189Tail47
def root189Tail45 : List Nat := 23087 :: root189Tail46
def root189Tail44 : List Nat := 22343 :: root189Tail45
def root189Tail43 : List Nat := 21347 :: root189Tail44
def root189Tail42 : List Nat := 20747 :: root189Tail43
def root189Tail41 : List Nat := 20483 :: root189Tail42
def root189Tail40 : List Nat := 19139 :: root189Tail41
def root189Tail39 : List Nat := 17837 :: root189Tail40
def root189Tail38 : List Nat := 17681 :: root189Tail39
def root189Tail37 : List Nat := 17027 :: root189Tail38
def root189Tail36 : List Nat := 16787 :: root189Tail37
def root189Tail35 : List Nat := 15053 :: root189Tail36
def root189Tail34 : List Nat := 14423 :: root189Tail35
def root189Tail33 : List Nat := 13931 :: root189Tail34
def root189Tail32 : List Nat := 13007 :: root189Tail33
def root189Tail31 : List Nat := 12743 :: root189Tail32
def root189Tail30 : List Nat := 12473 :: root189Tail31
def root189Tail29 : List Nat := 12347 :: root189Tail30
def root189Tail28 : List Nat := 12197 :: root189Tail29
def root189Tail27 : List Nat := 12107 :: root189Tail28
def root189Tail26 : List Nat := 11777 :: root189Tail27
def root189Tail25 : List Nat := 11057 :: root189Tail26
def root189Tail24 : List Nat := 10847 :: root189Tail25
def root189Tail23 : List Nat := 10529 :: root189Tail24
def root189Tail22 : List Nat := 9479 :: root189Tail23
def root189Tail21 : List Nat := 9437 :: root189Tail22
def root189Tail20 : List Nat := 9323 :: root189Tail21
def root189Tail19 : List Nat := 8243 :: root189Tail20
def root189Tail18 : List Nat := 7919 :: root189Tail19
def root189Tail17 : List Nat := 7853 :: root189Tail18
def root189Tail16 : List Nat := 7349 :: root189Tail17
def root189Tail15 : List Nat := 7211 :: root189Tail16
def root189Tail14 : List Nat := 6899 :: root189Tail15
def root189Tail13 : List Nat := 6803 :: root189Tail14
def root189Tail12 : List Nat := 6197 :: root189Tail13
def root189Tail11 : List Nat := 5849 :: root189Tail12
def root189Tail10 : List Nat := 5711 :: root189Tail11
def root189Tail9 : List Nat := 5639 :: root189Tail10
def root189Tail8 : List Nat := 4967 :: root189Tail9
def root189Tail7 : List Nat := 4673 :: root189Tail8
def root189Tail6 : List Nat := 4547 :: root189Tail7
def root189Tail5 : List Nat := 2927 :: root189Tail6
def root189Tail4 : List Nat := 2447 :: root189Tail5
def root189Tail3 : List Nat := 2417 :: root189Tail4
def root189Tail2 : List Nat := 2111 :: root189Tail3
def root189Tail1 : List Nat := 1877 :: root189Tail2
def root189Tail0 : List Nat := 1733 :: root189Tail1
def root189Candidates0 : List Nat := [2927]
set_option maxRecDepth 8192 in
theorem root189Candidates0_checked : candidates edge 4 1163 1733 root189Tail1 winner.bound = root189Candidates0 := by
  unfold candidates
  have row : edge 1733 = (SortedNatTree.contains FastRows.row1733) := by
    change Data.row1733.contains = (SortedNatTree.contains FastRows.row1733)
    exact FastRows.row1733_checked.symm
  rw [row]
  decide
theorem root189Step0 : scan edge (search edge 3) 4 [1163] 1163 root189Tail0 winner =
    scan edge (search edge 3) 4 [1163] 1163 root189Tail1 winner := by
  rw [root189Tail0, scan, if_neg (by decide), root189Candidates0_checked]
  change scan edge (search edge 3) 4 [1163] 1163 root189Tail1
    (if 3 ≤ root189Candidates0.length then search edge 3 [1163, 1733] root189Candidates0 2896 winner else winner) = _
  rw [if_neg (by decide)]
def root189Candidates1 : List Nat := [2447, 5711, 12197, 13931]
set_option maxRecDepth 8192 in
theorem root189Candidates1_checked : candidates edge 4 1163 1877 root189Tail2 winner.bound = root189Candidates1 := by
  unfold candidates
  have row : edge 1877 = (SortedNatTree.contains FastRows.row1877) := by
    change Data.row1877.contains = (SortedNatTree.contains FastRows.row1877)
    exact FastRows.row1877_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root189Child1 : search edge 3 [1163, 1877] root189Candidates1 3040 winner = winner := by decide
theorem root189Step1 : scan edge (search edge 3) 4 [1163] 1163 root189Tail1 winner =
    scan edge (search edge 3) 4 [1163] 1163 root189Tail2 winner := by
  rw [root189Tail1, scan, if_neg (by decide), root189Candidates1_checked]
  change scan edge (search edge 3) 4 [1163] 1163 root189Tail2
    (if 3 ≤ root189Candidates1.length then search edge 3 [1163, 1877] root189Candidates1 3040 winner else winner) = _
  rw [if_pos (by decide), root189Child1]
def root189Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root189Candidates2_checked : candidates edge 4 1163 2111 root189Tail3 winner.bound = root189Candidates2 := by
  unfold candidates
  have row : edge 2111 = (SortedNatTree.contains FastRows.row2111) := by
    change Data.row2111.contains = (SortedNatTree.contains FastRows.row2111)
    exact FastRows.row2111_checked.symm
  rw [row]
  decide
theorem root189Step2 : scan edge (search edge 3) 4 [1163] 1163 root189Tail2 winner =
    scan edge (search edge 3) 4 [1163] 1163 root189Tail3 winner := by
  rw [root189Tail2, scan, if_neg (by decide), root189Candidates2_checked]
  change scan edge (search edge 3) 4 [1163] 1163 root189Tail3
    (if 3 ≤ root189Candidates2.length then search edge 3 [1163, 2111] root189Candidates2 3274 winner else winner) = _
  rw [if_neg (by decide)]
def root189Candidates3 : List Nat := [9437]
set_option maxRecDepth 8192 in
theorem root189Candidates3_checked : candidates edge 4 1163 2417 root189Tail4 winner.bound = root189Candidates3 := by
  unfold candidates
  have row : edge 2417 = (SortedNatTree.contains FastRows.row2417) := by
    change Data.row2417.contains = (SortedNatTree.contains FastRows.row2417)
    exact FastRows.row2417_checked.symm
  rw [row]
  decide
theorem root189Step3 : scan edge (search edge 3) 4 [1163] 1163 root189Tail3 winner =
    scan edge (search edge 3) 4 [1163] 1163 root189Tail4 winner := by
  rw [root189Tail3, scan, if_neg (by decide), root189Candidates3_checked]
  change scan edge (search edge 3) 4 [1163] 1163 root189Tail4
    (if 3 ≤ root189Candidates3.length then search edge 3 [1163, 2417] root189Candidates3 3580 winner else winner) = _
  rw [if_neg (by decide)]
def root189Candidates4 : List Nat := [12197]
set_option maxRecDepth 8192 in
theorem root189Candidates4_checked : candidates edge 4 1163 2447 root189Tail5 winner.bound = root189Candidates4 := by
  unfold candidates
  have row : edge 2447 = (SortedNatTree.contains FastRows.row2447) := by
    change Data.row2447.contains = (SortedNatTree.contains FastRows.row2447)
    exact FastRows.row2447_checked.symm
  rw [row]
  decide
theorem root189Step4 : scan edge (search edge 3) 4 [1163] 1163 root189Tail4 winner =
    scan edge (search edge 3) 4 [1163] 1163 root189Tail5 winner := by
  rw [root189Tail4, scan, if_neg (by decide), root189Candidates4_checked]
  change scan edge (search edge 3) 4 [1163] 1163 root189Tail5
    (if 3 ≤ root189Candidates4.length then search edge 3 [1163, 2447] root189Candidates4 3610 winner else winner) = _
  rw [if_neg (by decide)]
def root189Candidates5 : List Nat := [8243, 11057]
set_option maxRecDepth 8192 in
theorem root189Candidates5_checked : candidates edge 4 1163 2927 root189Tail6 winner.bound = root189Candidates5 := by
  unfold candidates
  have row : edge 2927 = (SortedNatTree.contains FastRows.row2927) := by
    change Data.row2927.contains = (SortedNatTree.contains FastRows.row2927)
    exact FastRows.row2927_checked.symm
  rw [row]
  decide
theorem root189Step5 : scan edge (search edge 3) 4 [1163] 1163 root189Tail5 winner =
    scan edge (search edge 3) 4 [1163] 1163 root189Tail6 winner := by
  rw [root189Tail5, scan, if_neg (by decide), root189Candidates5_checked]
  change scan edge (search edge 3) 4 [1163] 1163 root189Tail6
    (if 3 ≤ root189Candidates5.length then search edge 3 [1163, 2927] root189Candidates5 4090 winner else winner) = _
  rw [if_neg (by decide)]
def root189Candidates6 : List Nat := [7919]
set_option maxRecDepth 8192 in
theorem root189Candidates6_checked : candidates edge 4 1163 4547 root189Tail7 winner.bound = root189Candidates6 := by
  unfold candidates
  have row : edge 4547 = (SortedNatTree.contains FastRows.row4547) := by
    change Data.row4547.contains = (SortedNatTree.contains FastRows.row4547)
    exact FastRows.row4547_checked.symm
  rw [row]
  decide
theorem root189Step6 : scan edge (search edge 3) 4 [1163] 1163 root189Tail6 winner =
    scan edge (search edge 3) 4 [1163] 1163 root189Tail7 winner := by
  rw [root189Tail6, scan, if_neg (by decide), root189Candidates6_checked]
  change scan edge (search edge 3) 4 [1163] 1163 root189Tail7
    (if 3 ≤ root189Candidates6.length then search edge 3 [1163, 4547] root189Candidates6 5710 winner else winner) = _
  rw [if_neg (by decide)]
def root189Candidates7 : List Nat := [9437]
set_option maxRecDepth 8192 in
theorem root189Candidates7_checked : candidates edge 4 1163 4673 root189Tail8 winner.bound = root189Candidates7 := by
  unfold candidates
  have row : edge 4673 = (SortedNatTree.contains FastRows.row4673) := by
    change Data.row4673.contains = (SortedNatTree.contains FastRows.row4673)
    exact FastRows.row4673_checked.symm
  rw [row]
  decide
theorem root189Step7 : scan edge (search edge 3) 4 [1163] 1163 root189Tail7 winner =
    scan edge (search edge 3) 4 [1163] 1163 root189Tail8 winner := by
  rw [root189Tail7, scan, if_neg (by decide), root189Candidates7_checked]
  change scan edge (search edge 3) 4 [1163] 1163 root189Tail8
    (if 3 ≤ root189Candidates7.length then search edge 3 [1163, 4673] root189Candidates7 5836 winner else winner) = _
  rw [if_neg (by decide)]
def root189Candidates8 : List Nat := [5849, 11057]
set_option maxRecDepth 8192 in
theorem root189Candidates8_checked : candidates edge 4 1163 4967 root189Tail9 winner.bound = root189Candidates8 := by
  unfold candidates
  have row : edge 4967 = (SortedNatTree.contains FastRows.row4967) := by
    change Data.row4967.contains = (SortedNatTree.contains FastRows.row4967)
    exact FastRows.row4967_checked.symm
  rw [row]
  decide
theorem root189Step8 : scan edge (search edge 3) 4 [1163] 1163 root189Tail8 winner =
    scan edge (search edge 3) 4 [1163] 1163 root189Tail9 winner := by
  rw [root189Tail8, scan, if_neg (by decide), root189Candidates8_checked]
  change scan edge (search edge 3) 4 [1163] 1163 root189Tail9
    (if 3 ≤ root189Candidates8.length then search edge 3 [1163, 4967] root189Candidates8 6130 winner else winner) = _
  rw [if_neg (by decide)]
def root189Candidates9 : List Nat := [14423]
set_option maxRecDepth 8192 in
theorem root189Candidates9_checked : candidates edge 4 1163 5639 root189Tail10 winner.bound = root189Candidates9 := by
  unfold candidates
  have row : edge 5639 = (SortedNatTree.contains FastRows.row5639) := by
    change Data.row5639.contains = (SortedNatTree.contains FastRows.row5639)
    exact FastRows.row5639_checked.symm
  rw [row]
  decide
theorem root189Step9 : scan edge (search edge 3) 4 [1163] 1163 root189Tail9 winner =
    scan edge (search edge 3) 4 [1163] 1163 root189Tail10 winner := by
  rw [root189Tail9, scan, if_neg (by decide), root189Candidates9_checked]
  change scan edge (search edge 3) 4 [1163] 1163 root189Tail10
    (if 3 ≤ root189Candidates9.length then search edge 3 [1163, 5639] root189Candidates9 6802 winner else winner) = _
  rw [if_neg (by decide)]
def root189Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root189Candidates10_checked : candidates edge 4 1163 5711 root189Tail11 winner.bound = root189Candidates10 := by
  unfold candidates
  have row : edge 5711 = (SortedNatTree.contains FastRows.row5711) := by
    change Data.row5711.contains = (SortedNatTree.contains FastRows.row5711)
    exact FastRows.row5711_checked.symm
  rw [row]
  decide
theorem root189Step10 : scan edge (search edge 3) 4 [1163] 1163 root189Tail10 winner =
    scan edge (search edge 3) 4 [1163] 1163 root189Tail11 winner := by
  rw [root189Tail10, scan, if_neg (by decide), root189Candidates10_checked]
  change scan edge (search edge 3) 4 [1163] 1163 root189Tail11
    (if 3 ≤ root189Candidates10.length then search edge 3 [1163, 5711] root189Candidates10 6874 winner else winner) = _
  rw [if_neg (by decide)]
def root189Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root189Candidates11_checked : candidates edge 4 1163 5849 root189Tail12 winner.bound = root189Candidates11 := by
  unfold candidates
  have row : edge 5849 = (SortedNatTree.contains FastRows.row5849) := by
    change Data.row5849.contains = (SortedNatTree.contains FastRows.row5849)
    exact FastRows.row5849_checked.symm
  rw [row]
  decide
theorem root189Step11 : scan edge (search edge 3) 4 [1163] 1163 root189Tail11 winner =
    scan edge (search edge 3) 4 [1163] 1163 root189Tail12 winner := by
  rw [root189Tail11, scan, if_neg (by decide), root189Candidates11_checked]
  change scan edge (search edge 3) 4 [1163] 1163 root189Tail12
    (if 3 ≤ root189Candidates11.length then search edge 3 [1163, 5849] root189Candidates11 7012 winner else winner) = _
  rw [if_neg (by decide)]
def root189Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root189Candidates12_checked : candidates edge 4 1163 6197 root189Tail13 winner.bound = root189Candidates12 := by
  unfold candidates
  have row : edge 6197 = (SortedNatTree.contains FastRows.row6197) := by
    change Data.row6197.contains = (SortedNatTree.contains FastRows.row6197)
    exact FastRows.row6197_checked.symm
  rw [row]
  decide
theorem root189Step12 : scan edge (search edge 3) 4 [1163] 1163 root189Tail12 winner =
    scan edge (search edge 3) 4 [1163] 1163 root189Tail13 winner := by
  rw [root189Tail12, scan, if_neg (by decide), root189Candidates12_checked]
  change scan edge (search edge 3) 4 [1163] 1163 root189Tail13
    (if 3 ≤ root189Candidates12.length then search edge 3 [1163, 6197] root189Candidates12 7360 winner else winner) = _
  rw [if_neg (by decide)]
theorem root189Scan13 : scan edge (search edge 3) 4 [1163] 1163 root189Tail13 winner = winner := by rw [root189Tail13, scan, if_pos (by decide)]
theorem root189Scan12 : scan edge (search edge 3) 4 [1163] 1163 root189Tail12 winner = winner :=
  (root189Step12).trans root189Scan13
theorem root189Scan11 : scan edge (search edge 3) 4 [1163] 1163 root189Tail11 winner = winner :=
  (root189Step11).trans root189Scan12
theorem root189Scan10 : scan edge (search edge 3) 4 [1163] 1163 root189Tail10 winner = winner :=
  (root189Step10).trans root189Scan11
theorem root189Scan9 : scan edge (search edge 3) 4 [1163] 1163 root189Tail9 winner = winner :=
  (root189Step9).trans root189Scan10
theorem root189Scan8 : scan edge (search edge 3) 4 [1163] 1163 root189Tail8 winner = winner :=
  (root189Step8).trans root189Scan9
theorem root189Scan7 : scan edge (search edge 3) 4 [1163] 1163 root189Tail7 winner = winner :=
  (root189Step7).trans root189Scan8
theorem root189Scan6 : scan edge (search edge 3) 4 [1163] 1163 root189Tail6 winner = winner :=
  (root189Step6).trans root189Scan7
theorem root189Scan5 : scan edge (search edge 3) 4 [1163] 1163 root189Tail5 winner = winner :=
  (root189Step5).trans root189Scan6
theorem root189Scan4 : scan edge (search edge 3) 4 [1163] 1163 root189Tail4 winner = winner :=
  (root189Step4).trans root189Scan5
theorem root189Scan3 : scan edge (search edge 3) 4 [1163] 1163 root189Tail3 winner = winner :=
  (root189Step3).trans root189Scan4
theorem root189Scan2 : scan edge (search edge 3) 4 [1163] 1163 root189Tail2 winner = winner :=
  (root189Step2).trans root189Scan3
theorem root189Scan1 : scan edge (search edge 3) 4 [1163] 1163 root189Tail1 winner = winner :=
  (root189Step1).trans root189Scan2
theorem root189Scan0 : scan edge (search edge 3) 4 [1163] 1163 root189Tail0 winner = winner :=
  (root189Step0).trans root189Scan1
set_option maxRecDepth 4096 in
theorem root189Branch : search edge 4 [1163] root189Candidates 1163 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1163] 1163 root189Tail0 winner = winner
  exact root189Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
