import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root161Candidates : List Nat := Data.row971
set_option maxRecDepth 8192 in
theorem root161Candidates_checked : candidates edge 5 0 971 tail162 winner.bound = root161Candidates := by
  apply candidates_eq_row edge 5 0 971 tail162 root161Candidates winner.bound tail162_ordered
  · decide
  · decide
  · funext q
    rfl
def root161Tail60 : List Nat := []
def root161Tail59 : List Nat := 24851 :: root161Tail60
def root161Tail58 : List Nat := 24329 :: root161Tail59
def root161Tail57 : List Nat := 23633 :: root161Tail58
def root161Tail56 : List Nat := 23417 :: root161Tail57
def root161Tail55 : List Nat := 21737 :: root161Tail56
def root161Tail54 : List Nat := 21503 :: root161Tail55
def root161Tail53 : List Nat := 21149 :: root161Tail54
def root161Tail52 : List Nat := 20981 :: root161Tail53
def root161Tail51 : List Nat := 19571 :: root161Tail52
def root161Tail50 : List Nat := 19433 :: root161Tail51
def root161Tail49 : List Nat := 18701 :: root161Tail50
def root161Tail48 : List Nat := 18587 :: root161Tail49
def root161Tail47 : List Nat := 17333 :: root161Tail48
def root161Tail46 : List Nat := 17093 :: root161Tail47
def root161Tail45 : List Nat := 16871 :: root161Tail46
def root161Tail44 : List Nat := 16673 :: root161Tail45
def root161Tail43 : List Nat := 16319 :: root161Tail44
def root161Tail42 : List Nat := 16139 :: root161Tail43
def root161Tail41 : List Nat := 15413 :: root161Tail42
def root161Tail40 : List Nat := 14951 :: root161Tail41
def root161Tail39 : List Nat := 14879 :: root161Tail40
def root161Tail38 : List Nat := 14669 :: root161Tail39
def root161Tail37 : List Nat := 14639 :: root161Tail38
def root161Tail36 : List Nat := 14321 :: root161Tail37
def root161Tail35 : List Nat := 14009 :: root161Tail36
def root161Tail34 : List Nat := 13463 :: root161Tail35
def root161Tail33 : List Nat := 13103 :: root161Tail34
def root161Tail32 : List Nat := 13043 :: root161Tail33
def root161Tail31 : List Nat := 13007 :: root161Tail32
def root161Tail30 : List Nat := 11369 :: root161Tail31
def root161Tail29 : List Nat := 9029 :: root161Tail30
def root161Tail28 : List Nat := 8999 :: root161Tail29
def root161Tail27 : List Nat := 8831 :: root161Tail28
def root161Tail26 : List Nat := 8171 :: root161Tail27
def root161Tail25 : List Nat := 8123 :: root161Tail26
def root161Tail24 : List Nat := 8081 :: root161Tail25
def root161Tail23 : List Nat := 8039 :: root161Tail24
def root161Tail22 : List Nat := 7919 :: root161Tail23
def root161Tail21 : List Nat := 7517 :: root161Tail22
def root161Tail20 : List Nat := 7193 :: root161Tail21
def root161Tail19 : List Nat := 6701 :: root161Tail20
def root161Tail18 : List Nat := 6689 :: root161Tail19
def root161Tail17 : List Nat := 6569 :: root161Tail18
def root161Tail16 : List Nat := 6491 :: root161Tail17
def root161Tail15 : List Nat := 6269 :: root161Tail16
def root161Tail14 : List Nat := 5501 :: root161Tail15
def root161Tail13 : List Nat := 5477 :: root161Tail14
def root161Tail12 : List Nat := 5309 :: root161Tail13
def root161Tail11 : List Nat := 5009 :: root161Tail12
def root161Tail10 : List Nat := 4421 :: root161Tail11
def root161Tail9 : List Nat := 3947 :: root161Tail10
def root161Tail8 : List Nat := 3863 :: root161Tail9
def root161Tail7 : List Nat := 2897 :: root161Tail8
def root161Tail6 : List Nat := 2699 :: root161Tail7
def root161Tail5 : List Nat := 2543 :: root161Tail6
def root161Tail4 : List Nat := 2447 :: root161Tail5
def root161Tail3 : List Nat := 2111 :: root161Tail4
def root161Tail2 : List Nat := 1511 :: root161Tail3
def root161Tail1 : List Nat := 1019 :: root161Tail2
def root161Tail0 : List Nat := 977 :: root161Tail1
def root161Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root161Candidates0_checked : candidates edge 4 971 977 root161Tail1 winner.bound = root161Candidates0 := by
  unfold candidates
  have row : edge 977 = (SortedNatTree.contains FastRows.row977) := by
    change Data.row977.contains = (SortedNatTree.contains FastRows.row977)
    exact FastRows.row977_checked.symm
  rw [row]
  decide
theorem root161Step0 : scan edge (search edge 3) 4 [971] 971 root161Tail0 winner =
    scan edge (search edge 3) 4 [971] 971 root161Tail1 winner := by
  rw [root161Tail0, scan, if_neg (by decide), root161Candidates0_checked]
  change scan edge (search edge 3) 4 [971] 971 root161Tail1
    (if 3 ≤ root161Candidates0.length then search edge 3 [971, 977] root161Candidates0 1948 winner else winner) = _
  rw [if_neg (by decide)]
def root161Candidates1 : List Nat := [5477, 7193]
set_option maxRecDepth 8192 in
theorem root161Candidates1_checked : candidates edge 4 971 1019 root161Tail2 winner.bound = root161Candidates1 := by
  unfold candidates
  have row : edge 1019 = (SortedNatTree.contains FastRows.row1019) := by
    change Data.row1019.contains = (SortedNatTree.contains FastRows.row1019)
    exact FastRows.row1019_checked.symm
  rw [row]
  decide
theorem root161Step1 : scan edge (search edge 3) 4 [971] 971 root161Tail1 winner =
    scan edge (search edge 3) 4 [971] 971 root161Tail2 winner := by
  rw [root161Tail1, scan, if_neg (by decide), root161Candidates1_checked]
  change scan edge (search edge 3) 4 [971] 971 root161Tail2
    (if 3 ≤ root161Candidates1.length then search edge 3 [971, 1019] root161Candidates1 1990 winner else winner) = _
  rw [if_neg (by decide)]
def root161Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root161Candidates2_checked : candidates edge 4 971 1511 root161Tail3 winner.bound = root161Candidates2 := by
  unfold candidates
  have row : edge 1511 = (SortedNatTree.contains FastRows.row1511) := by
    change Data.row1511.contains = (SortedNatTree.contains FastRows.row1511)
    exact FastRows.row1511_checked.symm
  rw [row]
  decide
theorem root161Step2 : scan edge (search edge 3) 4 [971] 971 root161Tail2 winner =
    scan edge (search edge 3) 4 [971] 971 root161Tail3 winner := by
  rw [root161Tail2, scan, if_neg (by decide), root161Candidates2_checked]
  change scan edge (search edge 3) 4 [971] 971 root161Tail3
    (if 3 ≤ root161Candidates2.length then search edge 3 [971, 1511] root161Candidates2 2482 winner else winner) = _
  rw [if_neg (by decide)]
def root161Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root161Candidates3_checked : candidates edge 4 971 2111 root161Tail4 winner.bound = root161Candidates3 := by
  unfold candidates
  have row : edge 2111 = (SortedNatTree.contains FastRows.row2111) := by
    change Data.row2111.contains = (SortedNatTree.contains FastRows.row2111)
    exact FastRows.row2111_checked.symm
  rw [row]
  decide
theorem root161Step3 : scan edge (search edge 3) 4 [971] 971 root161Tail3 winner =
    scan edge (search edge 3) 4 [971] 971 root161Tail4 winner := by
  rw [root161Tail3, scan, if_neg (by decide), root161Candidates3_checked]
  change scan edge (search edge 3) 4 [971] 971 root161Tail4
    (if 3 ≤ root161Candidates3.length then search edge 3 [971, 2111] root161Candidates3 3082 winner else winner) = _
  rw [if_neg (by decide)]
def root161Candidates4 : List Nat := [8123, 13463]
set_option maxRecDepth 8192 in
theorem root161Candidates4_checked : candidates edge 4 971 2447 root161Tail5 winner.bound = root161Candidates4 := by
  unfold candidates
  have row : edge 2447 = (SortedNatTree.contains FastRows.row2447) := by
    change Data.row2447.contains = (SortedNatTree.contains FastRows.row2447)
    exact FastRows.row2447_checked.symm
  rw [row]
  decide
theorem root161Step4 : scan edge (search edge 3) 4 [971] 971 root161Tail4 winner =
    scan edge (search edge 3) 4 [971] 971 root161Tail5 winner := by
  rw [root161Tail4, scan, if_neg (by decide), root161Candidates4_checked]
  change scan edge (search edge 3) 4 [971] 971 root161Tail5
    (if 3 ≤ root161Candidates4.length then search edge 3 [971, 2447] root161Candidates4 3418 winner else winner) = _
  rw [if_neg (by decide)]
def root161Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root161Candidates5_checked : candidates edge 4 971 2543 root161Tail6 winner.bound = root161Candidates5 := by
  unfold candidates
  have row : edge 2543 = (SortedNatTree.contains FastRows.row2543) := by
    change Data.row2543.contains = (SortedNatTree.contains FastRows.row2543)
    exact FastRows.row2543_checked.symm
  rw [row]
  decide
theorem root161Step5 : scan edge (search edge 3) 4 [971] 971 root161Tail5 winner =
    scan edge (search edge 3) 4 [971] 971 root161Tail6 winner := by
  rw [root161Tail5, scan, if_neg (by decide), root161Candidates5_checked]
  change scan edge (search edge 3) 4 [971] 971 root161Tail6
    (if 3 ≤ root161Candidates5.length then search edge 3 [971, 2543] root161Candidates5 3514 winner else winner) = _
  rw [if_neg (by decide)]
def root161Candidates6 : List Nat := [8039, 16319]
set_option maxRecDepth 8192 in
theorem root161Candidates6_checked : candidates edge 4 971 2699 root161Tail7 winner.bound = root161Candidates6 := by
  unfold candidates
  have row : edge 2699 = (SortedNatTree.contains FastRows.row2699) := by
    change Data.row2699.contains = (SortedNatTree.contains FastRows.row2699)
    exact FastRows.row2699_checked.symm
  rw [row]
  decide
theorem root161Step6 : scan edge (search edge 3) 4 [971] 971 root161Tail6 winner =
    scan edge (search edge 3) 4 [971] 971 root161Tail7 winner := by
  rw [root161Tail6, scan, if_neg (by decide), root161Candidates6_checked]
  change scan edge (search edge 3) 4 [971] 971 root161Tail7
    (if 3 ≤ root161Candidates6.length then search edge 3 [971, 2699] root161Candidates6 3670 winner else winner) = _
  rw [if_neg (by decide)]
def root161Candidates7 : List Nat := [3947, 5501]
set_option maxRecDepth 8192 in
theorem root161Candidates7_checked : candidates edge 4 971 2897 root161Tail8 winner.bound = root161Candidates7 := by
  unfold candidates
  have row : edge 2897 = (SortedNatTree.contains FastRows.row2897) := by
    change Data.row2897.contains = (SortedNatTree.contains FastRows.row2897)
    exact FastRows.row2897_checked.symm
  rw [row]
  decide
theorem root161Step7 : scan edge (search edge 3) 4 [971] 971 root161Tail7 winner =
    scan edge (search edge 3) 4 [971] 971 root161Tail8 winner := by
  rw [root161Tail7, scan, if_neg (by decide), root161Candidates7_checked]
  change scan edge (search edge 3) 4 [971] 971 root161Tail8
    (if 3 ≤ root161Candidates7.length then search edge 3 [971, 2897] root161Candidates7 3868 winner else winner) = _
  rw [if_neg (by decide)]
def root161Candidates8 : List Nat := [6569, 14879, 17093]
set_option maxRecDepth 8192 in
theorem root161Candidates8_checked : candidates edge 4 971 3863 root161Tail9 winner.bound = root161Candidates8 := by
  unfold candidates
  have row : edge 3863 = (SortedNatTree.contains FastRows.row3863) := by
    change Data.row3863.contains = (SortedNatTree.contains FastRows.row3863)
    exact FastRows.row3863_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root161Child8 : search edge 3 [971, 3863] root161Candidates8 4834 winner = winner := by decide
theorem root161Step8 : scan edge (search edge 3) 4 [971] 971 root161Tail8 winner =
    scan edge (search edge 3) 4 [971] 971 root161Tail9 winner := by
  rw [root161Tail8, scan, if_neg (by decide), root161Candidates8_checked]
  change scan edge (search edge 3) 4 [971] 971 root161Tail9
    (if 3 ≤ root161Candidates8.length then search edge 3 [971, 3863] root161Candidates8 4834 winner else winner) = _
  rw [if_pos (by decide), root161Child8]
def root161Candidates9 : List Nat := [17333]
set_option maxRecDepth 8192 in
theorem root161Candidates9_checked : candidates edge 4 971 3947 root161Tail10 winner.bound = root161Candidates9 := by
  unfold candidates
  have row : edge 3947 = (SortedNatTree.contains FastRows.row3947) := by
    change Data.row3947.contains = (SortedNatTree.contains FastRows.row3947)
    exact FastRows.row3947_checked.symm
  rw [row]
  decide
theorem root161Step9 : scan edge (search edge 3) 4 [971] 971 root161Tail9 winner =
    scan edge (search edge 3) 4 [971] 971 root161Tail10 winner := by
  rw [root161Tail9, scan, if_neg (by decide), root161Candidates9_checked]
  change scan edge (search edge 3) 4 [971] 971 root161Tail10
    (if 3 ≤ root161Candidates9.length then search edge 3 [971, 3947] root161Candidates9 4918 winner else winner) = _
  rw [if_neg (by decide)]
def root161Candidates10 : List Nat := [8039]
set_option maxRecDepth 8192 in
theorem root161Candidates10_checked : candidates edge 4 971 4421 root161Tail11 winner.bound = root161Candidates10 := by
  unfold candidates
  have row : edge 4421 = (SortedNatTree.contains FastRows.row4421) := by
    change Data.row4421.contains = (SortedNatTree.contains FastRows.row4421)
    exact FastRows.row4421_checked.symm
  rw [row]
  decide
theorem root161Step10 : scan edge (search edge 3) 4 [971] 971 root161Tail10 winner =
    scan edge (search edge 3) 4 [971] 971 root161Tail11 winner := by
  rw [root161Tail10, scan, if_neg (by decide), root161Candidates10_checked]
  change scan edge (search edge 3) 4 [971] 971 root161Tail11
    (if 3 ≤ root161Candidates10.length then search edge 3 [971, 4421] root161Candidates10 5392 winner else winner) = _
  rw [if_neg (by decide)]
def root161Candidates11 : List Nat := [13043, 14669, 16139]
set_option maxRecDepth 8192 in
theorem root161Candidates11_checked : candidates edge 4 971 5009 root161Tail12 winner.bound = root161Candidates11 := by
  unfold candidates
  have row : edge 5009 = (SortedNatTree.contains FastRows.row5009) := by
    change Data.row5009.contains = (SortedNatTree.contains FastRows.row5009)
    exact FastRows.row5009_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root161Child11 : search edge 3 [971, 5009] root161Candidates11 5980 winner = winner := by decide
theorem root161Step11 : scan edge (search edge 3) 4 [971] 971 root161Tail11 winner =
    scan edge (search edge 3) 4 [971] 971 root161Tail12 winner := by
  rw [root161Tail11, scan, if_neg (by decide), root161Candidates11_checked]
  change scan edge (search edge 3) 4 [971] 971 root161Tail12
    (if 3 ≤ root161Candidates11.length then search edge 3 [971, 5009] root161Candidates11 5980 winner else winner) = _
  rw [if_pos (by decide), root161Child11]
def root161Candidates12 : List Nat := [5501]
set_option maxRecDepth 8192 in
theorem root161Candidates12_checked : candidates edge 4 971 5309 root161Tail13 winner.bound = root161Candidates12 := by
  unfold candidates
  have row : edge 5309 = (SortedNatTree.contains FastRows.row5309) := by
    change Data.row5309.contains = (SortedNatTree.contains FastRows.row5309)
    exact FastRows.row5309_checked.symm
  rw [row]
  decide
theorem root161Step12 : scan edge (search edge 3) 4 [971] 971 root161Tail12 winner =
    scan edge (search edge 3) 4 [971] 971 root161Tail13 winner := by
  rw [root161Tail12, scan, if_neg (by decide), root161Candidates12_checked]
  change scan edge (search edge 3) 4 [971] 971 root161Tail13
    (if 3 ≤ root161Candidates12.length then search edge 3 [971, 5309] root161Candidates12 6280 winner else winner) = _
  rw [if_neg (by decide)]
def root161Candidates13 : List Nat := [8039, 8081, 8123, 14009]
set_option maxRecDepth 8192 in
theorem root161Candidates13_checked : candidates edge 4 971 5477 root161Tail14 winner.bound = root161Candidates13 := by
  unfold candidates
  have row : edge 5477 = (SortedNatTree.contains FastRows.row5477) := by
    change Data.row5477.contains = (SortedNatTree.contains FastRows.row5477)
    exact FastRows.row5477_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root161Child13 : search edge 3 [971, 5477] root161Candidates13 6448 winner = winner := by decide
theorem root161Step13 : scan edge (search edge 3) 4 [971] 971 root161Tail13 winner =
    scan edge (search edge 3) 4 [971] 971 root161Tail14 winner := by
  rw [root161Tail13, scan, if_neg (by decide), root161Candidates13_checked]
  change scan edge (search edge 3) 4 [971] 971 root161Tail14
    (if 3 ≤ root161Candidates13.length then search edge 3 [971, 5477] root161Candidates13 6448 winner else winner) = _
  rw [if_pos (by decide), root161Child13]
def root161Candidates14 : List Nat := [19433]
set_option maxRecDepth 8192 in
theorem root161Candidates14_checked : candidates edge 4 971 5501 root161Tail15 winner.bound = root161Candidates14 := by
  unfold candidates
  have row : edge 5501 = (SortedNatTree.contains FastRows.row5501) := by
    change Data.row5501.contains = (SortedNatTree.contains FastRows.row5501)
    exact FastRows.row5501_checked.symm
  rw [row]
  decide
theorem root161Step14 : scan edge (search edge 3) 4 [971] 971 root161Tail14 winner =
    scan edge (search edge 3) 4 [971] 971 root161Tail15 winner := by
  rw [root161Tail14, scan, if_neg (by decide), root161Candidates14_checked]
  change scan edge (search edge 3) 4 [971] 971 root161Tail15
    (if 3 ≤ root161Candidates14.length then search edge 3 [971, 5501] root161Candidates14 6472 winner else winner) = _
  rw [if_neg (by decide)]
theorem root161Scan15 : scan edge (search edge 3) 4 [971] 971 root161Tail15 winner = winner := by rw [root161Tail15, scan, if_pos (by decide)]
theorem root161Scan14 : scan edge (search edge 3) 4 [971] 971 root161Tail14 winner = winner :=
  (root161Step14).trans root161Scan15
theorem root161Scan13 : scan edge (search edge 3) 4 [971] 971 root161Tail13 winner = winner :=
  (root161Step13).trans root161Scan14
theorem root161Scan12 : scan edge (search edge 3) 4 [971] 971 root161Tail12 winner = winner :=
  (root161Step12).trans root161Scan13
theorem root161Scan11 : scan edge (search edge 3) 4 [971] 971 root161Tail11 winner = winner :=
  (root161Step11).trans root161Scan12
theorem root161Scan10 : scan edge (search edge 3) 4 [971] 971 root161Tail10 winner = winner :=
  (root161Step10).trans root161Scan11
theorem root161Scan9 : scan edge (search edge 3) 4 [971] 971 root161Tail9 winner = winner :=
  (root161Step9).trans root161Scan10
theorem root161Scan8 : scan edge (search edge 3) 4 [971] 971 root161Tail8 winner = winner :=
  (root161Step8).trans root161Scan9
theorem root161Scan7 : scan edge (search edge 3) 4 [971] 971 root161Tail7 winner = winner :=
  (root161Step7).trans root161Scan8
theorem root161Scan6 : scan edge (search edge 3) 4 [971] 971 root161Tail6 winner = winner :=
  (root161Step6).trans root161Scan7
theorem root161Scan5 : scan edge (search edge 3) 4 [971] 971 root161Tail5 winner = winner :=
  (root161Step5).trans root161Scan6
theorem root161Scan4 : scan edge (search edge 3) 4 [971] 971 root161Tail4 winner = winner :=
  (root161Step4).trans root161Scan5
theorem root161Scan3 : scan edge (search edge 3) 4 [971] 971 root161Tail3 winner = winner :=
  (root161Step3).trans root161Scan4
theorem root161Scan2 : scan edge (search edge 3) 4 [971] 971 root161Tail2 winner = winner :=
  (root161Step2).trans root161Scan3
theorem root161Scan1 : scan edge (search edge 3) 4 [971] 971 root161Tail1 winner = winner :=
  (root161Step1).trans root161Scan2
theorem root161Scan0 : scan edge (search edge 3) 4 [971] 971 root161Tail0 winner = winner :=
  (root161Step0).trans root161Scan1
set_option maxRecDepth 4096 in
theorem root161Branch : search edge 4 [971] root161Candidates 971 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [971] 971 root161Tail0 winner = winner
  exact root161Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
