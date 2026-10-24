import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root216Candidates : List Nat := Data.row1367
set_option maxRecDepth 8192 in
theorem root216Candidates_checked : candidates edge 5 0 1367 tail217 winner.bound = root216Candidates := by
  apply candidates_eq_row edge 5 0 1367 tail217 root216Candidates winner.bound tail217_ordered
  · decide
  · decide
  · funext q
    rfl
def root216Tail55 : List Nat := []
def root216Tail54 : List Nat := 24281 :: root216Tail55
def root216Tail53 : List Nat := 24113 :: root216Tail54
def root216Tail52 : List Nat := 23099 :: root216Tail53
def root216Tail51 : List Nat := 23039 :: root216Tail52
def root216Tail50 : List Nat := 22973 :: root216Tail51
def root216Tail49 : List Nat := 22343 :: root216Tail50
def root216Tail48 : List Nat := 22259 :: root216Tail49
def root216Tail47 : List Nat := 20981 :: root216Tail48
def root216Tail46 : List Nat := 20939 :: root216Tail47
def root216Tail45 : List Nat := 19553 :: root216Tail46
def root216Tail44 : List Nat := 17123 :: root216Tail45
def root216Tail43 : List Nat := 17093 :: root216Tail44
def root216Tail42 : List Nat := 16319 :: root216Tail43
def root216Tail41 : List Nat := 15269 :: root216Tail42
def root216Tail40 : List Nat := 14813 :: root216Tail41
def root216Tail39 : List Nat := 14771 :: root216Tail40
def root216Tail38 : List Nat := 14489 :: root216Tail39
def root216Tail37 : List Nat := 14447 :: root216Tail38
def root216Tail36 : List Nat := 13763 :: root216Tail37
def root216Tail35 : List Nat := 13691 :: root216Tail36
def root216Tail34 : List Nat := 12539 :: root216Tail35
def root216Tail33 : List Nat := 12473 :: root216Tail34
def root216Tail32 : List Nat := 11369 :: root216Tail33
def root216Tail31 : List Nat := 11243 :: root216Tail32
def root216Tail30 : List Nat := 11177 :: root216Tail31
def root216Tail29 : List Nat := 11117 :: root216Tail30
def root216Tail28 : List Nat := 10289 :: root216Tail29
def root216Tail27 : List Nat := 9767 :: root216Tail28
def root216Tail26 : List Nat := 9539 :: root216Tail27
def root216Tail25 : List Nat := 9431 :: root216Tail26
def root216Tail24 : List Nat := 8831 :: root216Tail25
def root216Tail23 : List Nat := 8447 :: root216Tail24
def root216Tail22 : List Nat := 8219 :: root216Tail23
def root216Tail21 : List Nat := 7949 :: root216Tail22
def root216Tail20 : List Nat := 7919 :: root216Tail21
def root216Tail19 : List Nat := 7229 :: root216Tail20
def root216Tail18 : List Nat := 7043 :: root216Tail19
def root216Tail17 : List Nat := 6563 :: root216Tail18
def root216Tail16 : List Nat := 6353 :: root216Tail17
def root216Tail15 : List Nat := 6323 :: root216Tail16
def root216Tail14 : List Nat := 5471 :: root216Tail15
def root216Tail13 : List Nat := 4877 :: root216Tail14
def root216Tail12 : List Nat := 4733 :: root216Tail13
def root216Tail11 : List Nat := 4691 :: root216Tail12
def root216Tail10 : List Nat := 4373 :: root216Tail11
def root216Tail9 : List Nat := 4079 :: root216Tail10
def root216Tail8 : List Nat := 3929 :: root216Tail9
def root216Tail7 : List Nat := 3467 :: root216Tail8
def root216Tail6 : List Nat := 3221 :: root216Tail7
def root216Tail5 : List Nat := 2963 :: root216Tail6
def root216Tail4 : List Nat := 2843 :: root216Tail5
def root216Tail3 : List Nat := 2591 :: root216Tail4
def root216Tail2 : List Nat := 2297 :: root216Tail3
def root216Tail1 : List Nat := 2027 :: root216Tail2
def root216Tail0 : List Nat := 1733 :: root216Tail1
def root216Candidates0 : List Nat := [6323, 7043, 7949]
set_option maxRecDepth 8192 in
theorem root216Candidates0_checked : candidates edge 4 1367 1733 root216Tail1 winner.bound = root216Candidates0 := by
  unfold candidates
  have row : edge 1733 = (SortedNatTree.contains FastRows.row1733) := by
    change Data.row1733.contains = (SortedNatTree.contains FastRows.row1733)
    exact FastRows.row1733_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root216Child0 : search edge 3 [1367, 1733] root216Candidates0 3100 winner = winner := by decide
theorem root216Step0 : scan edge (search edge 3) 4 [1367] 1367 root216Tail0 winner =
    scan edge (search edge 3) 4 [1367] 1367 root216Tail1 winner := by
  rw [root216Tail0, scan, if_neg (by decide), root216Candidates0_checked]
  change scan edge (search edge 3) 4 [1367] 1367 root216Tail1
    (if 3 ≤ root216Candidates0.length then search edge 3 [1367, 1733] root216Candidates0 3100 winner else winner) = _
  rw [if_pos (by decide), root216Child0]
def root216Candidates1 : List Nat := [15269]
set_option maxRecDepth 8192 in
theorem root216Candidates1_checked : candidates edge 4 1367 2027 root216Tail2 winner.bound = root216Candidates1 := by
  unfold candidates
  have row : edge 2027 = (SortedNatTree.contains FastRows.row2027) := by
    change Data.row2027.contains = (SortedNatTree.contains FastRows.row2027)
    exact FastRows.row2027_checked.symm
  rw [row]
  decide
theorem root216Step1 : scan edge (search edge 3) 4 [1367] 1367 root216Tail1 winner =
    scan edge (search edge 3) 4 [1367] 1367 root216Tail2 winner := by
  rw [root216Tail1, scan, if_neg (by decide), root216Candidates1_checked]
  change scan edge (search edge 3) 4 [1367] 1367 root216Tail2
    (if 3 ≤ root216Candidates1.length then search edge 3 [1367, 2027] root216Candidates1 3394 winner else winner) = _
  rw [if_neg (by decide)]
def root216Candidates2 : List Nat := [11243]
set_option maxRecDepth 8192 in
theorem root216Candidates2_checked : candidates edge 4 1367 2297 root216Tail3 winner.bound = root216Candidates2 := by
  unfold candidates
  have row : edge 2297 = (SortedNatTree.contains FastRows.row2297) := by
    change Data.row2297.contains = (SortedNatTree.contains FastRows.row2297)
    exact FastRows.row2297_checked.symm
  rw [row]
  decide
theorem root216Step2 : scan edge (search edge 3) 4 [1367] 1367 root216Tail2 winner =
    scan edge (search edge 3) 4 [1367] 1367 root216Tail3 winner := by
  rw [root216Tail2, scan, if_neg (by decide), root216Candidates2_checked]
  change scan edge (search edge 3) 4 [1367] 1367 root216Tail3
    (if 3 ≤ root216Candidates2.length then search edge 3 [1367, 2297] root216Candidates2 3664 winner else winner) = _
  rw [if_neg (by decide)]
def root216Candidates3 : List Nat := [9767]
set_option maxRecDepth 8192 in
theorem root216Candidates3_checked : candidates edge 4 1367 2591 root216Tail4 winner.bound = root216Candidates3 := by
  unfold candidates
  have row : edge 2591 = (SortedNatTree.contains FastRows.row2591) := by
    change Data.row2591.contains = (SortedNatTree.contains FastRows.row2591)
    exact FastRows.row2591_checked.symm
  rw [row]
  decide
theorem root216Step3 : scan edge (search edge 3) 4 [1367] 1367 root216Tail3 winner =
    scan edge (search edge 3) 4 [1367] 1367 root216Tail4 winner := by
  rw [root216Tail3, scan, if_neg (by decide), root216Candidates3_checked]
  change scan edge (search edge 3) 4 [1367] 1367 root216Tail4
    (if 3 ≤ root216Candidates3.length then search edge 3 [1367, 2591] root216Candidates3 3958 winner else winner) = _
  rw [if_neg (by decide)]
def root216Candidates4 : List Nat := [8831]
set_option maxRecDepth 8192 in
theorem root216Candidates4_checked : candidates edge 4 1367 2843 root216Tail5 winner.bound = root216Candidates4 := by
  unfold candidates
  have row : edge 2843 = (SortedNatTree.contains FastRows.row2843) := by
    change Data.row2843.contains = (SortedNatTree.contains FastRows.row2843)
    exact FastRows.row2843_checked.symm
  rw [row]
  decide
theorem root216Step4 : scan edge (search edge 3) 4 [1367] 1367 root216Tail4 winner =
    scan edge (search edge 3) 4 [1367] 1367 root216Tail5 winner := by
  rw [root216Tail4, scan, if_neg (by decide), root216Candidates4_checked]
  change scan edge (search edge 3) 4 [1367] 1367 root216Tail5
    (if 3 ≤ root216Candidates4.length then search edge 3 [1367, 2843] root216Candidates4 4210 winner else winner) = _
  rw [if_neg (by decide)]
def root216Candidates5 : List Nat := [4079, 4691]
set_option maxRecDepth 8192 in
theorem root216Candidates5_checked : candidates edge 4 1367 2963 root216Tail6 winner.bound = root216Candidates5 := by
  unfold candidates
  have row : edge 2963 = (SortedNatTree.contains FastRows.row2963) := by
    change Data.row2963.contains = (SortedNatTree.contains FastRows.row2963)
    exact FastRows.row2963_checked.symm
  rw [row]
  decide
theorem root216Step5 : scan edge (search edge 3) 4 [1367] 1367 root216Tail5 winner =
    scan edge (search edge 3) 4 [1367] 1367 root216Tail6 winner := by
  rw [root216Tail5, scan, if_neg (by decide), root216Candidates5_checked]
  change scan edge (search edge 3) 4 [1367] 1367 root216Tail6
    (if 3 ≤ root216Candidates5.length then search edge 3 [1367, 2963] root216Candidates5 4330 winner else winner) = _
  rw [if_neg (by decide)]
def root216Candidates6 : List Nat := [14447]
set_option maxRecDepth 8192 in
theorem root216Candidates6_checked : candidates edge 4 1367 3221 root216Tail7 winner.bound = root216Candidates6 := by
  unfold candidates
  have row : edge 3221 = (SortedNatTree.contains FastRows.row3221) := by
    change Data.row3221.contains = (SortedNatTree.contains FastRows.row3221)
    exact FastRows.row3221_checked.symm
  rw [row]
  decide
theorem root216Step6 : scan edge (search edge 3) 4 [1367] 1367 root216Tail6 winner =
    scan edge (search edge 3) 4 [1367] 1367 root216Tail7 winner := by
  rw [root216Tail6, scan, if_neg (by decide), root216Candidates6_checked]
  change scan edge (search edge 3) 4 [1367] 1367 root216Tail7
    (if 3 ≤ root216Candidates6.length then search edge 3 [1367, 3221] root216Candidates6 4588 winner else winner) = _
  rw [if_neg (by decide)]
def root216Candidates7 : List Nat := [5471, 7949, 17123]
set_option maxRecDepth 8192 in
theorem root216Candidates7_checked : candidates edge 4 1367 3467 root216Tail8 winner.bound = root216Candidates7 := by
  unfold candidates
  have row : edge 3467 = (SortedNatTree.contains FastRows.row3467) := by
    change Data.row3467.contains = (SortedNatTree.contains FastRows.row3467)
    exact FastRows.row3467_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root216Child7 : search edge 3 [1367, 3467] root216Candidates7 4834 winner = winner := by decide
theorem root216Step7 : scan edge (search edge 3) 4 [1367] 1367 root216Tail7 winner =
    scan edge (search edge 3) 4 [1367] 1367 root216Tail8 winner := by
  rw [root216Tail7, scan, if_neg (by decide), root216Candidates7_checked]
  change scan edge (search edge 3) 4 [1367] 1367 root216Tail8
    (if 3 ≤ root216Candidates7.length then search edge 3 [1367, 3467] root216Candidates7 4834 winner else winner) = _
  rw [if_pos (by decide), root216Child7]
def root216Candidates8 : List Nat := [7043, 16319]
set_option maxRecDepth 8192 in
theorem root216Candidates8_checked : candidates edge 4 1367 3929 root216Tail9 winner.bound = root216Candidates8 := by
  unfold candidates
  have row : edge 3929 = (SortedNatTree.contains FastRows.row3929) := by
    change Data.row3929.contains = (SortedNatTree.contains FastRows.row3929)
    exact FastRows.row3929_checked.symm
  rw [row]
  decide
theorem root216Step8 : scan edge (search edge 3) 4 [1367] 1367 root216Tail8 winner =
    scan edge (search edge 3) 4 [1367] 1367 root216Tail9 winner := by
  rw [root216Tail8, scan, if_neg (by decide), root216Candidates8_checked]
  change scan edge (search edge 3) 4 [1367] 1367 root216Tail9
    (if 3 ≤ root216Candidates8.length then search edge 3 [1367, 3929] root216Candidates8 5296 winner else winner) = _
  rw [if_neg (by decide)]
def root216Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root216Candidates9_checked : candidates edge 4 1367 4079 root216Tail10 winner.bound = root216Candidates9 := by
  unfold candidates
  have row : edge 4079 = (SortedNatTree.contains FastRows.row4079) := by
    change Data.row4079.contains = (SortedNatTree.contains FastRows.row4079)
    exact FastRows.row4079_checked.symm
  rw [row]
  decide
theorem root216Step9 : scan edge (search edge 3) 4 [1367] 1367 root216Tail9 winner =
    scan edge (search edge 3) 4 [1367] 1367 root216Tail10 winner := by
  rw [root216Tail9, scan, if_neg (by decide), root216Candidates9_checked]
  change scan edge (search edge 3) 4 [1367] 1367 root216Tail10
    (if 3 ≤ root216Candidates9.length then search edge 3 [1367, 4079] root216Candidates9 5446 winner else winner) = _
  rw [if_neg (by decide)]
def root216Candidates10 : List Nat := [7919, 9431]
set_option maxRecDepth 8192 in
theorem root216Candidates10_checked : candidates edge 4 1367 4373 root216Tail11 winner.bound = root216Candidates10 := by
  unfold candidates
  have row : edge 4373 = (SortedNatTree.contains FastRows.row4373) := by
    change Data.row4373.contains = (SortedNatTree.contains FastRows.row4373)
    exact FastRows.row4373_checked.symm
  rw [row]
  decide
theorem root216Step10 : scan edge (search edge 3) 4 [1367] 1367 root216Tail10 winner =
    scan edge (search edge 3) 4 [1367] 1367 root216Tail11 winner := by
  rw [root216Tail10, scan, if_neg (by decide), root216Candidates10_checked]
  change scan edge (search edge 3) 4 [1367] 1367 root216Tail11
    (if 3 ≤ root216Candidates10.length then search edge 3 [1367, 4373] root216Candidates10 5740 winner else winner) = _
  rw [if_neg (by decide)]
def root216Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root216Candidates11_checked : candidates edge 4 1367 4691 root216Tail12 winner.bound = root216Candidates11 := by
  unfold candidates
  have row : edge 4691 = (SortedNatTree.contains FastRows.row4691) := by
    change Data.row4691.contains = (SortedNatTree.contains FastRows.row4691)
    exact FastRows.row4691_checked.symm
  rw [row]
  decide
theorem root216Step11 : scan edge (search edge 3) 4 [1367] 1367 root216Tail11 winner =
    scan edge (search edge 3) 4 [1367] 1367 root216Tail12 winner := by
  rw [root216Tail11, scan, if_neg (by decide), root216Candidates11_checked]
  change scan edge (search edge 3) 4 [1367] 1367 root216Tail12
    (if 3 ≤ root216Candidates11.length then search edge 3 [1367, 4691] root216Candidates11 6058 winner else winner) = _
  rw [if_neg (by decide)]
def root216Candidates12 : List Nat := [7919, 10289]
set_option maxRecDepth 8192 in
theorem root216Candidates12_checked : candidates edge 4 1367 4733 root216Tail13 winner.bound = root216Candidates12 := by
  unfold candidates
  have row : edge 4733 = (SortedNatTree.contains FastRows.row4733) := by
    change Data.row4733.contains = (SortedNatTree.contains FastRows.row4733)
    exact FastRows.row4733_checked.symm
  rw [row]
  decide
theorem root216Step12 : scan edge (search edge 3) 4 [1367] 1367 root216Tail12 winner =
    scan edge (search edge 3) 4 [1367] 1367 root216Tail13 winner := by
  rw [root216Tail12, scan, if_neg (by decide), root216Candidates12_checked]
  change scan edge (search edge 3) 4 [1367] 1367 root216Tail13
    (if 3 ≤ root216Candidates12.length then search edge 3 [1367, 4733] root216Candidates12 6100 winner else winner) = _
  rw [if_neg (by decide)]
def root216Candidates13 : List Nat := [11177, 14771]
set_option maxRecDepth 8192 in
theorem root216Candidates13_checked : candidates edge 4 1367 4877 root216Tail14 winner.bound = root216Candidates13 := by
  unfold candidates
  have row : edge 4877 = (SortedNatTree.contains FastRows.row4877) := by
    change Data.row4877.contains = (SortedNatTree.contains FastRows.row4877)
    exact FastRows.row4877_checked.symm
  rw [row]
  decide
theorem root216Step13 : scan edge (search edge 3) 4 [1367] 1367 root216Tail13 winner =
    scan edge (search edge 3) 4 [1367] 1367 root216Tail14 winner := by
  rw [root216Tail13, scan, if_neg (by decide), root216Candidates13_checked]
  change scan edge (search edge 3) 4 [1367] 1367 root216Tail14
    (if 3 ≤ root216Candidates13.length then search edge 3 [1367, 4877] root216Candidates13 6244 winner else winner) = _
  rw [if_neg (by decide)]
def root216Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root216Candidates14_checked : candidates edge 4 1367 5471 root216Tail15 winner.bound = root216Candidates14 := by
  unfold candidates
  have row : edge 5471 = (SortedNatTree.contains FastRows.row5471) := by
    change Data.row5471.contains = (SortedNatTree.contains FastRows.row5471)
    exact FastRows.row5471_checked.symm
  rw [row]
  decide
theorem root216Step14 : scan edge (search edge 3) 4 [1367] 1367 root216Tail14 winner =
    scan edge (search edge 3) 4 [1367] 1367 root216Tail15 winner := by
  rw [root216Tail14, scan, if_neg (by decide), root216Candidates14_checked]
  change scan edge (search edge 3) 4 [1367] 1367 root216Tail15
    (if 3 ≤ root216Candidates14.length then search edge 3 [1367, 5471] root216Candidates14 6838 winner else winner) = _
  rw [if_neg (by decide)]
theorem root216Scan15 : scan edge (search edge 3) 4 [1367] 1367 root216Tail15 winner = winner := by rw [root216Tail15, scan, if_pos (by decide)]
theorem root216Scan14 : scan edge (search edge 3) 4 [1367] 1367 root216Tail14 winner = winner :=
  (root216Step14).trans root216Scan15
theorem root216Scan13 : scan edge (search edge 3) 4 [1367] 1367 root216Tail13 winner = winner :=
  (root216Step13).trans root216Scan14
theorem root216Scan12 : scan edge (search edge 3) 4 [1367] 1367 root216Tail12 winner = winner :=
  (root216Step12).trans root216Scan13
theorem root216Scan11 : scan edge (search edge 3) 4 [1367] 1367 root216Tail11 winner = winner :=
  (root216Step11).trans root216Scan12
theorem root216Scan10 : scan edge (search edge 3) 4 [1367] 1367 root216Tail10 winner = winner :=
  (root216Step10).trans root216Scan11
theorem root216Scan9 : scan edge (search edge 3) 4 [1367] 1367 root216Tail9 winner = winner :=
  (root216Step9).trans root216Scan10
theorem root216Scan8 : scan edge (search edge 3) 4 [1367] 1367 root216Tail8 winner = winner :=
  (root216Step8).trans root216Scan9
theorem root216Scan7 : scan edge (search edge 3) 4 [1367] 1367 root216Tail7 winner = winner :=
  (root216Step7).trans root216Scan8
theorem root216Scan6 : scan edge (search edge 3) 4 [1367] 1367 root216Tail6 winner = winner :=
  (root216Step6).trans root216Scan7
theorem root216Scan5 : scan edge (search edge 3) 4 [1367] 1367 root216Tail5 winner = winner :=
  (root216Step5).trans root216Scan6
theorem root216Scan4 : scan edge (search edge 3) 4 [1367] 1367 root216Tail4 winner = winner :=
  (root216Step4).trans root216Scan5
theorem root216Scan3 : scan edge (search edge 3) 4 [1367] 1367 root216Tail3 winner = winner :=
  (root216Step3).trans root216Scan4
theorem root216Scan2 : scan edge (search edge 3) 4 [1367] 1367 root216Tail2 winner = winner :=
  (root216Step2).trans root216Scan3
theorem root216Scan1 : scan edge (search edge 3) 4 [1367] 1367 root216Tail1 winner = winner :=
  (root216Step1).trans root216Scan2
theorem root216Scan0 : scan edge (search edge 3) 4 [1367] 1367 root216Tail0 winner = winner :=
  (root216Step0).trans root216Scan1
set_option maxRecDepth 4096 in
theorem root216Branch : search edge 4 [1367] root216Candidates 1367 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1367] 1367 root216Tail0 winner = winner
  exact root216Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
