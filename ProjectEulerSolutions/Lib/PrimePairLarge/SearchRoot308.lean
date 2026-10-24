import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root308Candidates : List Nat := Data.row2063
set_option maxRecDepth 8192 in
theorem root308Candidates_checked : candidates edge 5 0 2063 tail309 winner.bound = root308Candidates := by
  apply candidates_eq_row edge 5 0 2063 tail309 root308Candidates winner.bound tail309_ordered
  · decide
  · decide
  · funext q
    rfl
def root308Tail41 : List Nat := []
def root308Tail40 : List Nat := 23459 :: root308Tail41
def root308Tail39 : List Nat := 22157 :: root308Tail40
def root308Tail38 : List Nat := 21737 :: root308Tail39
def root308Tail37 : List Nat := 21383 :: root308Tail38
def root308Tail36 : List Nat := 19979 :: root308Tail37
def root308Tail35 : List Nat := 19697 :: root308Tail36
def root308Tail34 : List Nat := 19391 :: root308Tail35
def root308Tail33 : List Nat := 18719 :: root308Tail34
def root308Tail32 : List Nat := 18521 :: root308Tail33
def root308Tail31 : List Nat := 17597 :: root308Tail32
def root308Tail30 : List Nat := 17387 :: root308Tail31
def root308Tail29 : List Nat := 16889 :: root308Tail30
def root308Tail28 : List Nat := 16673 :: root308Tail29
def root308Tail27 : List Nat := 16631 :: root308Tail28
def root308Tail26 : List Nat := 16529 :: root308Tail27
def root308Tail25 : List Nat := 14621 :: root308Tail26
def root308Tail24 : List Nat := 14177 :: root308Tail25
def root308Tail23 : List Nat := 13697 :: root308Tail24
def root308Tail22 : List Nat := 13109 :: root308Tail23
def root308Tail21 : List Nat := 12101 :: root308Tail22
def root308Tail20 : List Nat := 11579 :: root308Tail21
def root308Tail19 : List Nat := 9791 :: root308Tail20
def root308Tail18 : List Nat := 9239 :: root308Tail19
def root308Tail17 : List Nat := 8861 :: root308Tail18
def root308Tail16 : List Nat := 8741 :: root308Tail17
def root308Tail15 : List Nat := 6599 :: root308Tail16
def root308Tail14 : List Nat := 6449 :: root308Tail15
def root308Tail13 : List Nat := 5981 :: root308Tail14
def root308Tail12 : List Nat := 5903 :: root308Tail13
def root308Tail11 : List Nat := 5711 :: root308Tail12
def root308Tail10 : List Nat := 5477 :: root308Tail11
def root308Tail9 : List Nat := 5231 :: root308Tail10
def root308Tail8 : List Nat := 4349 :: root308Tail9
def root308Tail7 : List Nat := 4217 :: root308Tail8
def root308Tail6 : List Nat := 3803 :: root308Tail7
def root308Tail5 : List Nat := 3617 :: root308Tail6
def root308Tail4 : List Nat := 3089 :: root308Tail5
def root308Tail3 : List Nat := 2411 :: root308Tail4
def root308Tail2 : List Nat := 2333 :: root308Tail3
def root308Tail1 : List Nat := 2243 :: root308Tail2
def root308Tail0 : List Nat := 2099 :: root308Tail1
def root308Candidates0 : List Nat := [17597]
set_option maxRecDepth 8192 in
theorem root308Candidates0_checked : candidates edge 4 2063 2099 root308Tail1 winner.bound = root308Candidates0 := by
  unfold candidates
  have row : edge 2099 = (SortedNatTree.contains FastRows.row2099) := by
    change Data.row2099.contains = (SortedNatTree.contains FastRows.row2099)
    exact FastRows.row2099_checked.symm
  rw [row]
  decide
theorem root308Step0 : scan edge (search edge 3) 4 [2063] 2063 root308Tail0 winner =
    scan edge (search edge 3) 4 [2063] 2063 root308Tail1 winner := by
  rw [root308Tail0, scan, if_neg (by decide), root308Candidates0_checked]
  change scan edge (search edge 3) 4 [2063] 2063 root308Tail1
    (if 3 ≤ root308Candidates0.length then search edge 3 [2063, 2099] root308Candidates0 4162 winner else winner) = _
  rw [if_neg (by decide)]
def root308Candidates1 : List Nat := [2411, 5477, 6599]
set_option maxRecDepth 8192 in
theorem root308Candidates1_checked : candidates edge 4 2063 2243 root308Tail2 winner.bound = root308Candidates1 := by
  unfold candidates
  have row : edge 2243 = (SortedNatTree.contains FastRows.row2243) := by
    change Data.row2243.contains = (SortedNatTree.contains FastRows.row2243)
    exact FastRows.row2243_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root308Child1 : search edge 3 [2063, 2243] root308Candidates1 4306 winner = winner := by decide
theorem root308Step1 : scan edge (search edge 3) 4 [2063] 2063 root308Tail1 winner =
    scan edge (search edge 3) 4 [2063] 2063 root308Tail2 winner := by
  rw [root308Tail1, scan, if_neg (by decide), root308Candidates1_checked]
  change scan edge (search edge 3) 4 [2063] 2063 root308Tail2
    (if 3 ≤ root308Candidates1.length then search edge 3 [2063, 2243] root308Candidates1 4306 winner else winner) = _
  rw [if_pos (by decide), root308Child1]
def root308Candidates2 : List Nat := [13109]
set_option maxRecDepth 8192 in
theorem root308Candidates2_checked : candidates edge 4 2063 2333 root308Tail3 winner.bound = root308Candidates2 := by
  unfold candidates
  have row : edge 2333 = (SortedNatTree.contains FastRows.row2333) := by
    change Data.row2333.contains = (SortedNatTree.contains FastRows.row2333)
    exact FastRows.row2333_checked.symm
  rw [row]
  decide
theorem root308Step2 : scan edge (search edge 3) 4 [2063] 2063 root308Tail2 winner =
    scan edge (search edge 3) 4 [2063] 2063 root308Tail3 winner := by
  rw [root308Tail2, scan, if_neg (by decide), root308Candidates2_checked]
  change scan edge (search edge 3) 4 [2063] 2063 root308Tail3
    (if 3 ≤ root308Candidates2.length then search edge 3 [2063, 2333] root308Candidates2 4396 winner else winner) = _
  rw [if_neg (by decide)]
def root308Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root308Candidates3_checked : candidates edge 4 2063 2411 root308Tail4 winner.bound = root308Candidates3 := by
  unfold candidates
  have row : edge 2411 = (SortedNatTree.contains FastRows.row2411) := by
    change Data.row2411.contains = (SortedNatTree.contains FastRows.row2411)
    exact FastRows.row2411_checked.symm
  rw [row]
  decide
theorem root308Step3 : scan edge (search edge 3) 4 [2063] 2063 root308Tail3 winner =
    scan edge (search edge 3) 4 [2063] 2063 root308Tail4 winner := by
  rw [root308Tail3, scan, if_neg (by decide), root308Candidates3_checked]
  change scan edge (search edge 3) 4 [2063] 2063 root308Tail4
    (if 3 ≤ root308Candidates3.length then search edge 3 [2063, 2411] root308Candidates3 4474 winner else winner) = _
  rw [if_neg (by decide)]
def root308Candidates4 : List Nat := [11579, 16889]
set_option maxRecDepth 8192 in
theorem root308Candidates4_checked : candidates edge 4 2063 3089 root308Tail5 winner.bound = root308Candidates4 := by
  unfold candidates
  have row : edge 3089 = (SortedNatTree.contains FastRows.row3089) := by
    change Data.row3089.contains = (SortedNatTree.contains FastRows.row3089)
    exact FastRows.row3089_checked.symm
  rw [row]
  decide
theorem root308Step4 : scan edge (search edge 3) 4 [2063] 2063 root308Tail4 winner =
    scan edge (search edge 3) 4 [2063] 2063 root308Tail5 winner := by
  rw [root308Tail4, scan, if_neg (by decide), root308Candidates4_checked]
  change scan edge (search edge 3) 4 [2063] 2063 root308Tail5
    (if 3 ≤ root308Candidates4.length then search edge 3 [2063, 3089] root308Candidates4 5152 winner else winner) = _
  rw [if_neg (by decide)]
def root308Candidates5 : List Nat := [3803, 11579, 16673]
set_option maxRecDepth 8192 in
theorem root308Candidates5_checked : candidates edge 4 2063 3617 root308Tail6 winner.bound = root308Candidates5 := by
  unfold candidates
  have row : edge 3617 = (SortedNatTree.contains FastRows.row3617) := by
    change Data.row3617.contains = (SortedNatTree.contains FastRows.row3617)
    exact FastRows.row3617_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root308Child5 : search edge 3 [2063, 3617] root308Candidates5 5680 winner = winner := by decide
theorem root308Step5 : scan edge (search edge 3) 4 [2063] 2063 root308Tail5 winner =
    scan edge (search edge 3) 4 [2063] 2063 root308Tail6 winner := by
  rw [root308Tail5, scan, if_neg (by decide), root308Candidates5_checked]
  change scan edge (search edge 3) 4 [2063] 2063 root308Tail6
    (if 3 ≤ root308Candidates5.length then search edge 3 [2063, 3617] root308Candidates5 5680 winner else winner) = _
  rw [if_pos (by decide), root308Child5]
def root308Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root308Candidates6_checked : candidates edge 4 2063 3803 root308Tail7 winner.bound = root308Candidates6 := by
  unfold candidates
  have row : edge 3803 = (SortedNatTree.contains FastRows.row3803) := by
    change Data.row3803.contains = (SortedNatTree.contains FastRows.row3803)
    exact FastRows.row3803_checked.symm
  rw [row]
  decide
theorem root308Step6 : scan edge (search edge 3) 4 [2063] 2063 root308Tail6 winner =
    scan edge (search edge 3) 4 [2063] 2063 root308Tail7 winner := by
  rw [root308Tail6, scan, if_neg (by decide), root308Candidates6_checked]
  change scan edge (search edge 3) 4 [2063] 2063 root308Tail7
    (if 3 ≤ root308Candidates6.length then search edge 3 [2063, 3803] root308Candidates6 5866 winner else winner) = _
  rw [if_neg (by decide)]
def root308Candidates7 : List Nat := [5981]
set_option maxRecDepth 8192 in
theorem root308Candidates7_checked : candidates edge 4 2063 4217 root308Tail8 winner.bound = root308Candidates7 := by
  unfold candidates
  have row : edge 4217 = (SortedNatTree.contains FastRows.row4217) := by
    change Data.row4217.contains = (SortedNatTree.contains FastRows.row4217)
    exact FastRows.row4217_checked.symm
  rw [row]
  decide
theorem root308Step7 : scan edge (search edge 3) 4 [2063] 2063 root308Tail7 winner =
    scan edge (search edge 3) 4 [2063] 2063 root308Tail8 winner := by
  rw [root308Tail7, scan, if_neg (by decide), root308Candidates7_checked]
  change scan edge (search edge 3) 4 [2063] 2063 root308Tail8
    (if 3 ≤ root308Candidates7.length then search edge 3 [2063, 4217] root308Candidates7 6280 winner else winner) = _
  rw [if_neg (by decide)]
def root308Candidates8 : List Nat := [13697]
set_option maxRecDepth 8192 in
theorem root308Candidates8_checked : candidates edge 4 2063 4349 root308Tail9 winner.bound = root308Candidates8 := by
  unfold candidates
  have row : edge 4349 = (SortedNatTree.contains FastRows.row4349) := by
    change Data.row4349.contains = (SortedNatTree.contains FastRows.row4349)
    exact FastRows.row4349_checked.symm
  rw [row]
  decide
theorem root308Step8 : scan edge (search edge 3) 4 [2063] 2063 root308Tail8 winner =
    scan edge (search edge 3) 4 [2063] 2063 root308Tail9 winner := by
  rw [root308Tail8, scan, if_neg (by decide), root308Candidates8_checked]
  change scan edge (search edge 3) 4 [2063] 2063 root308Tail9
    (if 3 ≤ root308Candidates8.length then search edge 3 [2063, 4349] root308Candidates8 6412 winner else winner) = _
  rw [if_neg (by decide)]
def root308Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root308Candidates9_checked : candidates edge 4 2063 5231 root308Tail10 winner.bound = root308Candidates9 := by
  unfold candidates
  have row : edge 5231 = (SortedNatTree.contains FastRows.row5231) := by
    change Data.row5231.contains = (SortedNatTree.contains FastRows.row5231)
    exact FastRows.row5231_checked.symm
  rw [row]
  decide
theorem root308Step9 : scan edge (search edge 3) 4 [2063] 2063 root308Tail9 winner =
    scan edge (search edge 3) 4 [2063] 2063 root308Tail10 winner := by
  rw [root308Tail9, scan, if_neg (by decide), root308Candidates9_checked]
  change scan edge (search edge 3) 4 [2063] 2063 root308Tail10
    (if 3 ≤ root308Candidates9.length then search edge 3 [2063, 5231] root308Candidates9 7294 winner else winner) = _
  rw [if_neg (by decide)]
def root308Candidates10 : List Nat := [5711, 6599, 8741]
set_option maxRecDepth 8192 in
theorem root308Candidates10_checked : candidates edge 4 2063 5477 root308Tail11 winner.bound = root308Candidates10 := by
  unfold candidates
  have row : edge 5477 = (SortedNatTree.contains FastRows.row5477) := by
    change Data.row5477.contains = (SortedNatTree.contains FastRows.row5477)
    exact FastRows.row5477_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root308Child10 : search edge 3 [2063, 5477] root308Candidates10 7540 winner = winner := by decide
theorem root308Step10 : scan edge (search edge 3) 4 [2063] 2063 root308Tail10 winner =
    scan edge (search edge 3) 4 [2063] 2063 root308Tail11 winner := by
  rw [root308Tail10, scan, if_neg (by decide), root308Candidates10_checked]
  change scan edge (search edge 3) 4 [2063] 2063 root308Tail11
    (if 3 ≤ root308Candidates10.length then search edge 3 [2063, 5477] root308Candidates10 7540 winner else winner) = _
  rw [if_pos (by decide), root308Child10]
def root308Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root308Candidates11_checked : candidates edge 4 2063 5711 root308Tail12 winner.bound = root308Candidates11 := by
  unfold candidates
  have row : edge 5711 = (SortedNatTree.contains FastRows.row5711) := by
    change Data.row5711.contains = (SortedNatTree.contains FastRows.row5711)
    exact FastRows.row5711_checked.symm
  rw [row]
  decide
theorem root308Step11 : scan edge (search edge 3) 4 [2063] 2063 root308Tail11 winner =
    scan edge (search edge 3) 4 [2063] 2063 root308Tail12 winner := by
  rw [root308Tail11, scan, if_neg (by decide), root308Candidates11_checked]
  change scan edge (search edge 3) 4 [2063] 2063 root308Tail12
    (if 3 ≤ root308Candidates11.length then search edge 3 [2063, 5711] root308Candidates11 7774 winner else winner) = _
  rw [if_neg (by decide)]
def root308Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root308Candidates12_checked : candidates edge 4 2063 5903 root308Tail13 winner.bound = root308Candidates12 := by
  unfold candidates
  have row : edge 5903 = (SortedNatTree.contains FastRows.row5903) := by
    change Data.row5903.contains = (SortedNatTree.contains FastRows.row5903)
    exact FastRows.row5903_checked.symm
  rw [row]
  decide
theorem root308Step12 : scan edge (search edge 3) 4 [2063] 2063 root308Tail12 winner =
    scan edge (search edge 3) 4 [2063] 2063 root308Tail13 winner := by
  rw [root308Tail12, scan, if_neg (by decide), root308Candidates12_checked]
  change scan edge (search edge 3) 4 [2063] 2063 root308Tail13
    (if 3 ≤ root308Candidates12.length then search edge 3 [2063, 5903] root308Candidates12 7966 winner else winner) = _
  rw [if_neg (by decide)]
def root308Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root308Candidates13_checked : candidates edge 4 2063 5981 root308Tail14 winner.bound = root308Candidates13 := by
  unfold candidates
  have row : edge 5981 = (SortedNatTree.contains FastRows.row5981) := by
    change Data.row5981.contains = (SortedNatTree.contains FastRows.row5981)
    exact FastRows.row5981_checked.symm
  rw [row]
  decide
theorem root308Step13 : scan edge (search edge 3) 4 [2063] 2063 root308Tail13 winner =
    scan edge (search edge 3) 4 [2063] 2063 root308Tail14 winner := by
  rw [root308Tail13, scan, if_neg (by decide), root308Candidates13_checked]
  change scan edge (search edge 3) 4 [2063] 2063 root308Tail14
    (if 3 ≤ root308Candidates13.length then search edge 3 [2063, 5981] root308Candidates13 8044 winner else winner) = _
  rw [if_neg (by decide)]
theorem root308Scan14 : scan edge (search edge 3) 4 [2063] 2063 root308Tail14 winner = winner := by rw [root308Tail14, scan, if_pos (by decide)]
theorem root308Scan13 : scan edge (search edge 3) 4 [2063] 2063 root308Tail13 winner = winner :=
  (root308Step13).trans root308Scan14
theorem root308Scan12 : scan edge (search edge 3) 4 [2063] 2063 root308Tail12 winner = winner :=
  (root308Step12).trans root308Scan13
theorem root308Scan11 : scan edge (search edge 3) 4 [2063] 2063 root308Tail11 winner = winner :=
  (root308Step11).trans root308Scan12
theorem root308Scan10 : scan edge (search edge 3) 4 [2063] 2063 root308Tail10 winner = winner :=
  (root308Step10).trans root308Scan11
theorem root308Scan9 : scan edge (search edge 3) 4 [2063] 2063 root308Tail9 winner = winner :=
  (root308Step9).trans root308Scan10
theorem root308Scan8 : scan edge (search edge 3) 4 [2063] 2063 root308Tail8 winner = winner :=
  (root308Step8).trans root308Scan9
theorem root308Scan7 : scan edge (search edge 3) 4 [2063] 2063 root308Tail7 winner = winner :=
  (root308Step7).trans root308Scan8
theorem root308Scan6 : scan edge (search edge 3) 4 [2063] 2063 root308Tail6 winner = winner :=
  (root308Step6).trans root308Scan7
theorem root308Scan5 : scan edge (search edge 3) 4 [2063] 2063 root308Tail5 winner = winner :=
  (root308Step5).trans root308Scan6
theorem root308Scan4 : scan edge (search edge 3) 4 [2063] 2063 root308Tail4 winner = winner :=
  (root308Step4).trans root308Scan5
theorem root308Scan3 : scan edge (search edge 3) 4 [2063] 2063 root308Tail3 winner = winner :=
  (root308Step3).trans root308Scan4
theorem root308Scan2 : scan edge (search edge 3) 4 [2063] 2063 root308Tail2 winner = winner :=
  (root308Step2).trans root308Scan3
theorem root308Scan1 : scan edge (search edge 3) 4 [2063] 2063 root308Tail1 winner = winner :=
  (root308Step1).trans root308Scan2
theorem root308Scan0 : scan edge (search edge 3) 4 [2063] 2063 root308Tail0 winner = winner :=
  (root308Step0).trans root308Scan1
set_option maxRecDepth 4096 in
theorem root308Branch : search edge 4 [2063] root308Candidates 2063 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2063] 2063 root308Tail0 winner = winner
  exact root308Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
