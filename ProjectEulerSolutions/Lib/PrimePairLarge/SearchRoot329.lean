import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root329Candidates : List Nat := Data.row2237
set_option maxRecDepth 8192 in
theorem root329Candidates_checked : candidates edge 5 0 2237 tail330 winner.bound = root329Candidates := by
  apply candidates_eq_row edge 5 0 2237 tail330 root329Candidates winner.bound tail330_ordered
  · decide
  · decide
  · funext q
    rfl
def root329Tail48 : List Nat := []
def root329Tail47 : List Nat := 23399 :: root329Tail48
def root329Tail46 : List Nat := 23333 :: root329Tail47
def root329Tail45 : List Nat := 22571 :: root329Tail46
def root329Tail44 : List Nat := 22229 :: root329Tail45
def root329Tail43 : List Nat := 21821 :: root329Tail44
def root329Tail42 : List Nat := 21317 :: root329Tail43
def root329Tail41 : List Nat := 21089 :: root329Tail42
def root329Tail40 : List Nat := 19997 :: root329Tail41
def root329Tail39 : List Nat := 19553 :: root329Tail40
def root329Tail38 : List Nat := 18959 :: root329Tail39
def root329Tail37 : List Nat := 18251 :: root329Tail38
def root329Tail36 : List Nat := 17789 :: root329Tail37
def root329Tail35 : List Nat := 16937 :: root329Tail36
def root329Tail34 : List Nat := 16901 :: root329Tail35
def root329Tail33 : List Nat := 16091 :: root329Tail34
def root329Tail32 : List Nat := 15923 :: root329Tail33
def root329Tail31 : List Nat := 14831 :: root329Tail32
def root329Tail30 : List Nat := 13613 :: root329Tail31
def root329Tail29 : List Nat := 13241 :: root329Tail30
def root329Tail28 : List Nat := 12149 :: root329Tail29
def root329Tail27 : List Nat := 11987 :: root329Tail28
def root329Tail26 : List Nat := 11981 :: root329Tail27
def root329Tail25 : List Nat := 11261 :: root329Tail26
def root329Tail24 : List Nat := 10211 :: root329Tail25
def root329Tail23 : List Nat := 9431 :: root329Tail24
def root329Tail22 : List Nat := 9419 :: root329Tail23
def root329Tail21 : List Nat := 8423 :: root329Tail22
def root329Tail20 : List Nat := 6917 :: root329Tail21
def root329Tail19 : List Nat := 6791 :: root329Tail20
def root329Tail18 : List Nat := 6779 :: root329Tail19
def root329Tail17 : List Nat := 6701 :: root329Tail18
def root329Tail16 : List Nat := 6689 :: root329Tail17
def root329Tail15 : List Nat := 6659 :: root329Tail16
def root329Tail14 : List Nat := 6581 :: root329Tail15
def root329Tail13 : List Nat := 5693 :: root329Tail14
def root329Tail12 : List Nat := 4637 :: root329Tail13
def root329Tail11 : List Nat := 4481 :: root329Tail12
def root329Tail10 : List Nat := 4421 :: root329Tail11
def root329Tail9 : List Nat := 4349 :: root329Tail10
def root329Tail8 : List Nat := 4139 :: root329Tail9
def root329Tail7 : List Nat := 4049 :: root329Tail8
def root329Tail6 : List Nat := 4019 :: root329Tail7
def root329Tail5 : List Nat := 3929 :: root329Tail6
def root329Tail4 : List Nat := 3257 :: root329Tail5
def root329Tail3 : List Nat := 3251 :: root329Tail4
def root329Tail2 : List Nat := 2549 :: root329Tail3
def root329Tail1 : List Nat := 2411 :: root329Tail2
def root329Tail0 : List Nat := 2381 :: root329Tail1
def root329Candidates0 : List Nat := [8423]
set_option maxRecDepth 8192 in
theorem root329Candidates0_checked : candidates edge 4 2237 2381 root329Tail1 winner.bound = root329Candidates0 := by
  unfold candidates
  have row : edge 2381 = (SortedNatTree.contains FastRows.row2381) := by
    change Data.row2381.contains = (SortedNatTree.contains FastRows.row2381)
    exact FastRows.row2381_checked.symm
  rw [row]
  decide
theorem root329Step0 : scan edge (search edge 3) 4 [2237] 2237 root329Tail0 winner =
    scan edge (search edge 3) 4 [2237] 2237 root329Tail1 winner := by
  rw [root329Tail0, scan, if_neg (by decide), root329Candidates0_checked]
  change scan edge (search edge 3) 4 [2237] 2237 root329Tail1
    (if 3 ≤ root329Candidates0.length then search edge 3 [2237, 2381] root329Candidates0 4618 winner else winner) = _
  rw [if_neg (by decide)]
def root329Candidates1 : List Nat := [19553]
set_option maxRecDepth 8192 in
theorem root329Candidates1_checked : candidates edge 4 2237 2411 root329Tail2 winner.bound = root329Candidates1 := by
  unfold candidates
  have row : edge 2411 = (SortedNatTree.contains FastRows.row2411) := by
    change Data.row2411.contains = (SortedNatTree.contains FastRows.row2411)
    exact FastRows.row2411_checked.symm
  rw [row]
  decide
theorem root329Step1 : scan edge (search edge 3) 4 [2237] 2237 root329Tail1 winner =
    scan edge (search edge 3) 4 [2237] 2237 root329Tail2 winner := by
  rw [root329Tail1, scan, if_neg (by decide), root329Candidates1_checked]
  change scan edge (search edge 3) 4 [2237] 2237 root329Tail2
    (if 3 ≤ root329Candidates1.length then search edge 3 [2237, 2411] root329Candidates1 4648 winner else winner) = _
  rw [if_neg (by decide)]
def root329Candidates2 : List Nat := [4019, 4349]
set_option maxRecDepth 8192 in
theorem root329Candidates2_checked : candidates edge 4 2237 2549 root329Tail3 winner.bound = root329Candidates2 := by
  unfold candidates
  have row : edge 2549 = (SortedNatTree.contains FastRows.row2549) := by
    change Data.row2549.contains = (SortedNatTree.contains FastRows.row2549)
    exact FastRows.row2549_checked.symm
  rw [row]
  decide
theorem root329Step2 : scan edge (search edge 3) 4 [2237] 2237 root329Tail2 winner =
    scan edge (search edge 3) 4 [2237] 2237 root329Tail3 winner := by
  rw [root329Tail2, scan, if_neg (by decide), root329Candidates2_checked]
  change scan edge (search edge 3) 4 [2237] 2237 root329Tail3
    (if 3 ≤ root329Candidates2.length then search edge 3 [2237, 2549] root329Candidates2 4786 winner else winner) = _
  rw [if_neg (by decide)]
def root329Candidates3 : List Nat := [14831]
set_option maxRecDepth 8192 in
theorem root329Candidates3_checked : candidates edge 4 2237 3251 root329Tail4 winner.bound = root329Candidates3 := by
  unfold candidates
  have row : edge 3251 = (SortedNatTree.contains FastRows.row3251) := by
    change Data.row3251.contains = (SortedNatTree.contains FastRows.row3251)
    exact FastRows.row3251_checked.symm
  rw [row]
  decide
theorem root329Step3 : scan edge (search edge 3) 4 [2237] 2237 root329Tail3 winner =
    scan edge (search edge 3) 4 [2237] 2237 root329Tail4 winner := by
  rw [root329Tail3, scan, if_neg (by decide), root329Candidates3_checked]
  change scan edge (search edge 3) 4 [2237] 2237 root329Tail4
    (if 3 ≤ root329Candidates3.length then search edge 3 [2237, 3251] root329Candidates3 5488 winner else winner) = _
  rw [if_neg (by decide)]
def root329Candidates4 : List Nat := [6917, 11261]
set_option maxRecDepth 8192 in
theorem root329Candidates4_checked : candidates edge 4 2237 3257 root329Tail5 winner.bound = root329Candidates4 := by
  unfold candidates
  have row : edge 3257 = (SortedNatTree.contains FastRows.row3257) := by
    change Data.row3257.contains = (SortedNatTree.contains FastRows.row3257)
    exact FastRows.row3257_checked.symm
  rw [row]
  decide
theorem root329Step4 : scan edge (search edge 3) 4 [2237] 2237 root329Tail4 winner =
    scan edge (search edge 3) 4 [2237] 2237 root329Tail5 winner := by
  rw [root329Tail4, scan, if_neg (by decide), root329Candidates4_checked]
  change scan edge (search edge 3) 4 [2237] 2237 root329Tail5
    (if 3 ≤ root329Candidates4.length then search edge 3 [2237, 3257] root329Candidates4 5494 winner else winner) = _
  rw [if_neg (by decide)]
def root329Candidates5 : List Nat := [4349, 4421]
set_option maxRecDepth 8192 in
theorem root329Candidates5_checked : candidates edge 4 2237 3929 root329Tail6 winner.bound = root329Candidates5 := by
  unfold candidates
  have row : edge 3929 = (SortedNatTree.contains FastRows.row3929) := by
    change Data.row3929.contains = (SortedNatTree.contains FastRows.row3929)
    exact FastRows.row3929_checked.symm
  rw [row]
  decide
theorem root329Step5 : scan edge (search edge 3) 4 [2237] 2237 root329Tail5 winner =
    scan edge (search edge 3) 4 [2237] 2237 root329Tail6 winner := by
  rw [root329Tail5, scan, if_neg (by decide), root329Candidates5_checked]
  change scan edge (search edge 3) 4 [2237] 2237 root329Tail6
    (if 3 ≤ root329Candidates5.length then search edge 3 [2237, 3929] root329Candidates5 6166 winner else winner) = _
  rw [if_neg (by decide)]
def root329Candidates6 : List Nat := [6791, 11987]
set_option maxRecDepth 8192 in
theorem root329Candidates6_checked : candidates edge 4 2237 4019 root329Tail7 winner.bound = root329Candidates6 := by
  unfold candidates
  have row : edge 4019 = (SortedNatTree.contains FastRows.row4019) := by
    change Data.row4019.contains = (SortedNatTree.contains FastRows.row4019)
    exact FastRows.row4019_checked.symm
  rw [row]
  decide
theorem root329Step6 : scan edge (search edge 3) 4 [2237] 2237 root329Tail6 winner =
    scan edge (search edge 3) 4 [2237] 2237 root329Tail7 winner := by
  rw [root329Tail6, scan, if_neg (by decide), root329Candidates6_checked]
  change scan edge (search edge 3) 4 [2237] 2237 root329Tail7
    (if 3 ≤ root329Candidates6.length then search edge 3 [2237, 4019] root329Candidates6 6256 winner else winner) = _
  rw [if_neg (by decide)]
def root329Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root329Candidates7_checked : candidates edge 4 2237 4049 root329Tail8 winner.bound = root329Candidates7 := by
  unfold candidates
  have row : edge 4049 = (SortedNatTree.contains FastRows.row4049) := by
    change Data.row4049.contains = (SortedNatTree.contains FastRows.row4049)
    exact FastRows.row4049_checked.symm
  rw [row]
  decide
theorem root329Step7 : scan edge (search edge 3) 4 [2237] 2237 root329Tail7 winner =
    scan edge (search edge 3) 4 [2237] 2237 root329Tail8 winner := by
  rw [root329Tail7, scan, if_neg (by decide), root329Candidates7_checked]
  change scan edge (search edge 3) 4 [2237] 2237 root329Tail8
    (if 3 ≤ root329Candidates7.length then search edge 3 [2237, 4049] root329Candidates7 6286 winner else winner) = _
  rw [if_neg (by decide)]
def root329Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root329Candidates8_checked : candidates edge 4 2237 4139 root329Tail9 winner.bound = root329Candidates8 := by
  unfold candidates
  have row : edge 4139 = (SortedNatTree.contains FastRows.row4139) := by
    change Data.row4139.contains = (SortedNatTree.contains FastRows.row4139)
    exact FastRows.row4139_checked.symm
  rw [row]
  decide
theorem root329Step8 : scan edge (search edge 3) 4 [2237] 2237 root329Tail8 winner =
    scan edge (search edge 3) 4 [2237] 2237 root329Tail9 winner := by
  rw [root329Tail8, scan, if_neg (by decide), root329Candidates8_checked]
  change scan edge (search edge 3) 4 [2237] 2237 root329Tail9
    (if 3 ≤ root329Candidates8.length then search edge 3 [2237, 4139] root329Candidates8 6376 winner else winner) = _
  rw [if_neg (by decide)]
def root329Candidates9 : List Nat := [6581]
set_option maxRecDepth 8192 in
theorem root329Candidates9_checked : candidates edge 4 2237 4349 root329Tail10 winner.bound = root329Candidates9 := by
  unfold candidates
  have row : edge 4349 = (SortedNatTree.contains FastRows.row4349) := by
    change Data.row4349.contains = (SortedNatTree.contains FastRows.row4349)
    exact FastRows.row4349_checked.symm
  rw [row]
  decide
theorem root329Step9 : scan edge (search edge 3) 4 [2237] 2237 root329Tail9 winner =
    scan edge (search edge 3) 4 [2237] 2237 root329Tail10 winner := by
  rw [root329Tail9, scan, if_neg (by decide), root329Candidates9_checked]
  change scan edge (search edge 3) 4 [2237] 2237 root329Tail10
    (if 3 ≤ root329Candidates9.length then search edge 3 [2237, 4349] root329Candidates9 6586 winner else winner) = _
  rw [if_neg (by decide)]
def root329Candidates10 : List Nat := [6791, 13241]
set_option maxRecDepth 8192 in
theorem root329Candidates10_checked : candidates edge 4 2237 4421 root329Tail11 winner.bound = root329Candidates10 := by
  unfold candidates
  have row : edge 4421 = (SortedNatTree.contains FastRows.row4421) := by
    change Data.row4421.contains = (SortedNatTree.contains FastRows.row4421)
    exact FastRows.row4421_checked.symm
  rw [row]
  decide
theorem root329Step10 : scan edge (search edge 3) 4 [2237] 2237 root329Tail10 winner =
    scan edge (search edge 3) 4 [2237] 2237 root329Tail11 winner := by
  rw [root329Tail10, scan, if_neg (by decide), root329Candidates10_checked]
  change scan edge (search edge 3) 4 [2237] 2237 root329Tail11
    (if 3 ≤ root329Candidates10.length then search edge 3 [2237, 4421] root329Candidates10 6658 winner else winner) = _
  rw [if_neg (by decide)]
def root329Candidates11 : List Nat := [6791]
set_option maxRecDepth 8192 in
theorem root329Candidates11_checked : candidates edge 4 2237 4481 root329Tail12 winner.bound = root329Candidates11 := by
  unfold candidates
  have row : edge 4481 = (SortedNatTree.contains FastRows.row4481) := by
    change Data.row4481.contains = (SortedNatTree.contains FastRows.row4481)
    exact FastRows.row4481_checked.symm
  rw [row]
  decide
theorem root329Step11 : scan edge (search edge 3) 4 [2237] 2237 root329Tail11 winner =
    scan edge (search edge 3) 4 [2237] 2237 root329Tail12 winner := by
  rw [root329Tail11, scan, if_neg (by decide), root329Candidates11_checked]
  change scan edge (search edge 3) 4 [2237] 2237 root329Tail12
    (if 3 ≤ root329Candidates11.length then search edge 3 [2237, 4481] root329Candidates11 6718 winner else winner) = _
  rw [if_neg (by decide)]
def root329Candidates12 : List Nat := []
set_option maxRecDepth 8192 in
theorem root329Candidates12_checked : candidates edge 4 2237 4637 root329Tail13 winner.bound = root329Candidates12 := by
  unfold candidates
  have row : edge 4637 = (SortedNatTree.contains FastRows.row4637) := by
    change Data.row4637.contains = (SortedNatTree.contains FastRows.row4637)
    exact FastRows.row4637_checked.symm
  rw [row]
  decide
theorem root329Step12 : scan edge (search edge 3) 4 [2237] 2237 root329Tail12 winner =
    scan edge (search edge 3) 4 [2237] 2237 root329Tail13 winner := by
  rw [root329Tail12, scan, if_neg (by decide), root329Candidates12_checked]
  change scan edge (search edge 3) 4 [2237] 2237 root329Tail13
    (if 3 ≤ root329Candidates12.length then search edge 3 [2237, 4637] root329Candidates12 6874 winner else winner) = _
  rw [if_neg (by decide)]
def root329Candidates13 : List Nat := [10211]
set_option maxRecDepth 8192 in
theorem root329Candidates13_checked : candidates edge 4 2237 5693 root329Tail14 winner.bound = root329Candidates13 := by
  unfold candidates
  have row : edge 5693 = (SortedNatTree.contains FastRows.row5693) := by
    change Data.row5693.contains = (SortedNatTree.contains FastRows.row5693)
    exact FastRows.row5693_checked.symm
  rw [row]
  decide
theorem root329Step13 : scan edge (search edge 3) 4 [2237] 2237 root329Tail13 winner =
    scan edge (search edge 3) 4 [2237] 2237 root329Tail14 winner := by
  rw [root329Tail13, scan, if_neg (by decide), root329Candidates13_checked]
  change scan edge (search edge 3) 4 [2237] 2237 root329Tail14
    (if 3 ≤ root329Candidates13.length then search edge 3 [2237, 5693] root329Candidates13 7930 winner else winner) = _
  rw [if_neg (by decide)]
theorem root329Scan14 : scan edge (search edge 3) 4 [2237] 2237 root329Tail14 winner = winner := by rw [root329Tail14, scan, if_pos (by decide)]
theorem root329Scan13 : scan edge (search edge 3) 4 [2237] 2237 root329Tail13 winner = winner :=
  (root329Step13).trans root329Scan14
theorem root329Scan12 : scan edge (search edge 3) 4 [2237] 2237 root329Tail12 winner = winner :=
  (root329Step12).trans root329Scan13
theorem root329Scan11 : scan edge (search edge 3) 4 [2237] 2237 root329Tail11 winner = winner :=
  (root329Step11).trans root329Scan12
theorem root329Scan10 : scan edge (search edge 3) 4 [2237] 2237 root329Tail10 winner = winner :=
  (root329Step10).trans root329Scan11
theorem root329Scan9 : scan edge (search edge 3) 4 [2237] 2237 root329Tail9 winner = winner :=
  (root329Step9).trans root329Scan10
theorem root329Scan8 : scan edge (search edge 3) 4 [2237] 2237 root329Tail8 winner = winner :=
  (root329Step8).trans root329Scan9
theorem root329Scan7 : scan edge (search edge 3) 4 [2237] 2237 root329Tail7 winner = winner :=
  (root329Step7).trans root329Scan8
theorem root329Scan6 : scan edge (search edge 3) 4 [2237] 2237 root329Tail6 winner = winner :=
  (root329Step6).trans root329Scan7
theorem root329Scan5 : scan edge (search edge 3) 4 [2237] 2237 root329Tail5 winner = winner :=
  (root329Step5).trans root329Scan6
theorem root329Scan4 : scan edge (search edge 3) 4 [2237] 2237 root329Tail4 winner = winner :=
  (root329Step4).trans root329Scan5
theorem root329Scan3 : scan edge (search edge 3) 4 [2237] 2237 root329Tail3 winner = winner :=
  (root329Step3).trans root329Scan4
theorem root329Scan2 : scan edge (search edge 3) 4 [2237] 2237 root329Tail2 winner = winner :=
  (root329Step2).trans root329Scan3
theorem root329Scan1 : scan edge (search edge 3) 4 [2237] 2237 root329Tail1 winner = winner :=
  (root329Step1).trans root329Scan2
theorem root329Scan0 : scan edge (search edge 3) 4 [2237] 2237 root329Tail0 winner = winner :=
  (root329Step0).trans root329Scan1
set_option maxRecDepth 4096 in
theorem root329Branch : search edge 4 [2237] root329Candidates 2237 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2237] 2237 root329Tail0 winner = winner
  exact root329Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
