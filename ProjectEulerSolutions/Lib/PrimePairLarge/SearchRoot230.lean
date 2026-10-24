import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root230Candidates : List Nat := Data.row1471
set_option maxRecDepth 8192 in
theorem root230Candidates_checked : candidates edge 5 0 1471 tail231 winner.bound = root230Candidates := by
  apply candidates_eq_row edge 5 0 1471 tail231 root230Candidates winner.bound tail231_ordered
  · decide
  · decide
  · funext q
    rfl
def root230Tail48 : List Nat := []
def root230Tail47 : List Nat := 24517 :: root230Tail48
def root230Tail46 : List Nat := 24121 :: root230Tail47
def root230Tail45 : List Nat := 21841 :: root230Tail46
def root230Tail44 : List Nat := 21577 :: root230Tail45
def root230Tail43 : List Nat := 21529 :: root230Tail44
def root230Tail42 : List Nat := 21157 :: root230Tail43
def root230Tail41 : List Nat := 21067 :: root230Tail42
def root230Tail40 : List Nat := 18913 :: root230Tail41
def root230Tail39 : List Nat := 18661 :: root230Tail40
def root230Tail38 : List Nat := 18313 :: root230Tail39
def root230Tail37 : List Nat := 17431 :: root230Tail38
def root230Tail36 : List Nat := 17107 :: root230Tail37
def root230Tail35 : List Nat := 16831 :: root230Tail36
def root230Tail34 : List Nat := 15817 :: root230Tail35
def root230Tail33 : List Nat := 15541 :: root230Tail34
def root230Tail32 : List Nat := 15187 :: root230Tail33
def root230Tail31 : List Nat := 14431 :: root230Tail32
def root230Tail30 : List Nat := 12379 :: root230Tail31
def root230Tail29 : List Nat := 11257 :: root230Tail30
def root230Tail28 : List Nat := 10867 :: root230Tail29
def root230Tail27 : List Nat := 10657 :: root230Tail28
def root230Tail26 : List Nat := 10567 :: root230Tail27
def root230Tail25 : List Nat := 10399 :: root230Tail26
def root230Tail24 : List Nat := 9787 :: root230Tail25
def root230Tail23 : List Nat := 9181 :: root230Tail24
def root230Tail22 : List Nat := 8779 :: root230Tail23
def root230Tail21 : List Nat := 8443 :: root230Tail22
def root230Tail20 : List Nat := 7933 :: root230Tail21
def root230Tail19 : List Nat := 7321 :: root230Tail20
def root230Tail18 : List Nat := 7207 :: root230Tail19
def root230Tail17 : List Nat := 7069 :: root230Tail18
def root230Tail16 : List Nat := 6301 :: root230Tail17
def root230Tail15 : List Nat := 6133 :: root230Tail16
def root230Tail14 : List Nat := 6073 :: root230Tail15
def root230Tail13 : List Nat := 5419 :: root230Tail14
def root230Tail12 : List Nat := 5179 :: root230Tail13
def root230Tail11 : List Nat := 4951 :: root230Tail12
def root230Tail10 : List Nat := 4561 :: root230Tail11
def root230Tail9 : List Nat := 4159 :: root230Tail10
def root230Tail8 : List Nat := 3931 :: root230Tail9
def root230Tail7 : List Nat := 3613 :: root230Tail8
def root230Tail6 : List Nat := 3571 :: root230Tail7
def root230Tail5 : List Nat := 3541 :: root230Tail6
def root230Tail4 : List Nat := 2521 :: root230Tail5
def root230Tail3 : List Nat := 2389 :: root230Tail4
def root230Tail2 : List Nat := 2311 :: root230Tail3
def root230Tail1 : List Nat := 2221 :: root230Tail2
def root230Tail0 : List Nat := 1597 :: root230Tail1
def root230Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root230Candidates0_checked : candidates edge 4 1471 1597 root230Tail1 winner.bound = root230Candidates0 := by
  unfold candidates
  have row : edge 1597 = (SortedNatTree.contains FastRows.row1597) := by
    change Data.row1597.contains = (SortedNatTree.contains FastRows.row1597)
    exact FastRows.row1597_checked.symm
  rw [row]
  decide
theorem root230Step0 : scan edge (search edge 3) 4 [1471] 1471 root230Tail0 winner =
    scan edge (search edge 3) 4 [1471] 1471 root230Tail1 winner := by
  rw [root230Tail0, scan, if_neg (by decide), root230Candidates0_checked]
  change scan edge (search edge 3) 4 [1471] 1471 root230Tail1
    (if 3 ≤ root230Candidates0.length then search edge 3 [1471, 1597] root230Candidates0 3068 winner else winner) = _
  rw [if_neg (by decide)]
def root230Candidates1 : List Nat := [2389]
set_option maxRecDepth 8192 in
theorem root230Candidates1_checked : candidates edge 4 1471 2221 root230Tail2 winner.bound = root230Candidates1 := by
  unfold candidates
  have row : edge 2221 = (SortedNatTree.contains FastRows.row2221) := by
    change Data.row2221.contains = (SortedNatTree.contains FastRows.row2221)
    exact FastRows.row2221_checked.symm
  rw [row]
  decide
theorem root230Step1 : scan edge (search edge 3) 4 [1471] 1471 root230Tail1 winner =
    scan edge (search edge 3) 4 [1471] 1471 root230Tail2 winner := by
  rw [root230Tail1, scan, if_neg (by decide), root230Candidates1_checked]
  change scan edge (search edge 3) 4 [1471] 1471 root230Tail2
    (if 3 ≤ root230Candidates1.length then search edge 3 [1471, 2221] root230Candidates1 3692 winner else winner) = _
  rw [if_neg (by decide)]
def root230Candidates2 : List Nat := [5179]
set_option maxRecDepth 8192 in
theorem root230Candidates2_checked : candidates edge 4 1471 2311 root230Tail3 winner.bound = root230Candidates2 := by
  unfold candidates
  have row : edge 2311 = (SortedNatTree.contains FastRows.row2311) := by
    change Data.row2311.contains = (SortedNatTree.contains FastRows.row2311)
    exact FastRows.row2311_checked.symm
  rw [row]
  decide
theorem root230Step2 : scan edge (search edge 3) 4 [1471] 1471 root230Tail2 winner =
    scan edge (search edge 3) 4 [1471] 1471 root230Tail3 winner := by
  rw [root230Tail2, scan, if_neg (by decide), root230Candidates2_checked]
  change scan edge (search edge 3) 4 [1471] 1471 root230Tail3
    (if 3 ≤ root230Candidates2.length then search edge 3 [1471, 2311] root230Candidates2 3782 winner else winner) = _
  rw [if_neg (by decide)]
def root230Candidates3 : List Nat := [2521]
set_option maxRecDepth 8192 in
theorem root230Candidates3_checked : candidates edge 4 1471 2389 root230Tail4 winner.bound = root230Candidates3 := by
  unfold candidates
  have row : edge 2389 = (SortedNatTree.contains FastRows.row2389) := by
    change Data.row2389.contains = (SortedNatTree.contains FastRows.row2389)
    exact FastRows.row2389_checked.symm
  rw [row]
  decide
theorem root230Step3 : scan edge (search edge 3) 4 [1471] 1471 root230Tail3 winner =
    scan edge (search edge 3) 4 [1471] 1471 root230Tail4 winner := by
  rw [root230Tail3, scan, if_neg (by decide), root230Candidates3_checked]
  change scan edge (search edge 3) 4 [1471] 1471 root230Tail4
    (if 3 ≤ root230Candidates3.length then search edge 3 [1471, 2389] root230Candidates3 3860 winner else winner) = _
  rw [if_neg (by decide)]
def root230Candidates4 : List Nat := [18913]
set_option maxRecDepth 8192 in
theorem root230Candidates4_checked : candidates edge 4 1471 2521 root230Tail5 winner.bound = root230Candidates4 := by
  unfold candidates
  have row : edge 2521 = (SortedNatTree.contains FastRows.row2521) := by
    change Data.row2521.contains = (SortedNatTree.contains FastRows.row2521)
    exact FastRows.row2521_checked.symm
  rw [row]
  decide
theorem root230Step4 : scan edge (search edge 3) 4 [1471] 1471 root230Tail4 winner =
    scan edge (search edge 3) 4 [1471] 1471 root230Tail5 winner := by
  rw [root230Tail4, scan, if_neg (by decide), root230Candidates4_checked]
  change scan edge (search edge 3) 4 [1471] 1471 root230Tail5
    (if 3 ≤ root230Candidates4.length then search edge 3 [1471, 2521] root230Candidates4 3992 winner else winner) = _
  rw [if_neg (by decide)]
def root230Candidates5 : List Nat := [10567]
set_option maxRecDepth 8192 in
theorem root230Candidates5_checked : candidates edge 4 1471 3541 root230Tail6 winner.bound = root230Candidates5 := by
  unfold candidates
  have row : edge 3541 = (SortedNatTree.contains FastRows.row3541) := by
    change Data.row3541.contains = (SortedNatTree.contains FastRows.row3541)
    exact FastRows.row3541_checked.symm
  rw [row]
  decide
theorem root230Step5 : scan edge (search edge 3) 4 [1471] 1471 root230Tail5 winner =
    scan edge (search edge 3) 4 [1471] 1471 root230Tail6 winner := by
  rw [root230Tail5, scan, if_neg (by decide), root230Candidates5_checked]
  change scan edge (search edge 3) 4 [1471] 1471 root230Tail6
    (if 3 ≤ root230Candidates5.length then search edge 3 [1471, 3541] root230Candidates5 5012 winner else winner) = _
  rw [if_neg (by decide)]
def root230Candidates6 : List Nat := [9181]
set_option maxRecDepth 8192 in
theorem root230Candidates6_checked : candidates edge 4 1471 3571 root230Tail7 winner.bound = root230Candidates6 := by
  unfold candidates
  have row : edge 3571 = (SortedNatTree.contains FastRows.row3571) := by
    change Data.row3571.contains = (SortedNatTree.contains FastRows.row3571)
    exact FastRows.row3571_checked.symm
  rw [row]
  decide
theorem root230Step6 : scan edge (search edge 3) 4 [1471] 1471 root230Tail6 winner =
    scan edge (search edge 3) 4 [1471] 1471 root230Tail7 winner := by
  rw [root230Tail6, scan, if_neg (by decide), root230Candidates6_checked]
  change scan edge (search edge 3) 4 [1471] 1471 root230Tail7
    (if 3 ≤ root230Candidates6.length then search edge 3 [1471, 3571] root230Candidates6 5042 winner else winner) = _
  rw [if_neg (by decide)]
def root230Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root230Candidates7_checked : candidates edge 4 1471 3613 root230Tail8 winner.bound = root230Candidates7 := by
  unfold candidates
  have row : edge 3613 = (SortedNatTree.contains FastRows.row3613) := by
    change Data.row3613.contains = (SortedNatTree.contains FastRows.row3613)
    exact FastRows.row3613_checked.symm
  rw [row]
  decide
theorem root230Step7 : scan edge (search edge 3) 4 [1471] 1471 root230Tail7 winner =
    scan edge (search edge 3) 4 [1471] 1471 root230Tail8 winner := by
  rw [root230Tail7, scan, if_neg (by decide), root230Candidates7_checked]
  change scan edge (search edge 3) 4 [1471] 1471 root230Tail8
    (if 3 ≤ root230Candidates7.length then search edge 3 [1471, 3613] root230Candidates7 5084 winner else winner) = _
  rw [if_neg (by decide)]
def root230Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root230Candidates8_checked : candidates edge 4 1471 3931 root230Tail9 winner.bound = root230Candidates8 := by
  unfold candidates
  have row : edge 3931 = (SortedNatTree.contains FastRows.row3931) := by
    change Data.row3931.contains = (SortedNatTree.contains FastRows.row3931)
    exact FastRows.row3931_checked.symm
  rw [row]
  decide
theorem root230Step8 : scan edge (search edge 3) 4 [1471] 1471 root230Tail8 winner =
    scan edge (search edge 3) 4 [1471] 1471 root230Tail9 winner := by
  rw [root230Tail8, scan, if_neg (by decide), root230Candidates8_checked]
  change scan edge (search edge 3) 4 [1471] 1471 root230Tail9
    (if 3 ≤ root230Candidates8.length then search edge 3 [1471, 3931] root230Candidates8 5402 winner else winner) = _
  rw [if_neg (by decide)]
def root230Candidates9 : List Nat := [15541]
set_option maxRecDepth 8192 in
theorem root230Candidates9_checked : candidates edge 4 1471 4159 root230Tail10 winner.bound = root230Candidates9 := by
  unfold candidates
  have row : edge 4159 = (SortedNatTree.contains FastRows.row4159) := by
    change Data.row4159.contains = (SortedNatTree.contains FastRows.row4159)
    exact FastRows.row4159_checked.symm
  rw [row]
  decide
theorem root230Step9 : scan edge (search edge 3) 4 [1471] 1471 root230Tail9 winner =
    scan edge (search edge 3) 4 [1471] 1471 root230Tail10 winner := by
  rw [root230Tail9, scan, if_neg (by decide), root230Candidates9_checked]
  change scan edge (search edge 3) 4 [1471] 1471 root230Tail10
    (if 3 ≤ root230Candidates9.length then search edge 3 [1471, 4159] root230Candidates9 5630 winner else winner) = _
  rw [if_neg (by decide)]
def root230Candidates10 : List Nat := [5419, 6073, 8779, 15817]
set_option maxRecDepth 8192 in
theorem root230Candidates10_checked : candidates edge 4 1471 4561 root230Tail11 winner.bound = root230Candidates10 := by
  unfold candidates
  have row : edge 4561 = (SortedNatTree.contains FastRows.row4561) := by
    change Data.row4561.contains = (SortedNatTree.contains FastRows.row4561)
    exact FastRows.row4561_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root230Child10 : search edge 3 [1471, 4561] root230Candidates10 6032 winner = winner := by decide
theorem root230Step10 : scan edge (search edge 3) 4 [1471] 1471 root230Tail10 winner =
    scan edge (search edge 3) 4 [1471] 1471 root230Tail11 winner := by
  rw [root230Tail10, scan, if_neg (by decide), root230Candidates10_checked]
  change scan edge (search edge 3) 4 [1471] 1471 root230Tail11
    (if 3 ≤ root230Candidates10.length then search edge 3 [1471, 4561] root230Candidates10 6032 winner else winner) = _
  rw [if_pos (by decide), root230Child10]
def root230Candidates11 : List Nat := [8779]
set_option maxRecDepth 8192 in
theorem root230Candidates11_checked : candidates edge 4 1471 4951 root230Tail12 winner.bound = root230Candidates11 := by
  unfold candidates
  have row : edge 4951 = (SortedNatTree.contains FastRows.row4951) := by
    change Data.row4951.contains = (SortedNatTree.contains FastRows.row4951)
    exact FastRows.row4951_checked.symm
  rw [row]
  decide
theorem root230Step11 : scan edge (search edge 3) 4 [1471] 1471 root230Tail11 winner =
    scan edge (search edge 3) 4 [1471] 1471 root230Tail12 winner := by
  rw [root230Tail11, scan, if_neg (by decide), root230Candidates11_checked]
  change scan edge (search edge 3) 4 [1471] 1471 root230Tail12
    (if 3 ≤ root230Candidates11.length then search edge 3 [1471, 4951] root230Candidates11 6422 winner else winner) = _
  rw [if_neg (by decide)]
def root230Candidates12 : List Nat := [6301]
set_option maxRecDepth 8192 in
theorem root230Candidates12_checked : candidates edge 4 1471 5179 root230Tail13 winner.bound = root230Candidates12 := by
  unfold candidates
  have row : edge 5179 = (SortedNatTree.contains FastRows.row5179) := by
    change Data.row5179.contains = (SortedNatTree.contains FastRows.row5179)
    exact FastRows.row5179_checked.symm
  rw [row]
  decide
theorem root230Step12 : scan edge (search edge 3) 4 [1471] 1471 root230Tail12 winner =
    scan edge (search edge 3) 4 [1471] 1471 root230Tail13 winner := by
  rw [root230Tail12, scan, if_neg (by decide), root230Candidates12_checked]
  change scan edge (search edge 3) 4 [1471] 1471 root230Tail13
    (if 3 ≤ root230Candidates12.length then search edge 3 [1471, 5179] root230Candidates12 6650 winner else winner) = _
  rw [if_neg (by decide)]
def root230Candidates13 : List Nat := [10657]
set_option maxRecDepth 8192 in
theorem root230Candidates13_checked : candidates edge 4 1471 5419 root230Tail14 winner.bound = root230Candidates13 := by
  unfold candidates
  have row : edge 5419 = (SortedNatTree.contains FastRows.row5419) := by
    change Data.row5419.contains = (SortedNatTree.contains FastRows.row5419)
    exact FastRows.row5419_checked.symm
  rw [row]
  decide
theorem root230Step13 : scan edge (search edge 3) 4 [1471] 1471 root230Tail13 winner =
    scan edge (search edge 3) 4 [1471] 1471 root230Tail14 winner := by
  rw [root230Tail13, scan, if_neg (by decide), root230Candidates13_checked]
  change scan edge (search edge 3) 4 [1471] 1471 root230Tail14
    (if 3 ≤ root230Candidates13.length then search edge 3 [1471, 5419] root230Candidates13 6890 winner else winner) = _
  rw [if_neg (by decide)]
def root230Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root230Candidates14_checked : candidates edge 4 1471 6073 root230Tail15 winner.bound = root230Candidates14 := by
  unfold candidates
  have row : edge 6073 = (SortedNatTree.contains FastRows.row6073) := by
    change Data.row6073.contains = (SortedNatTree.contains FastRows.row6073)
    exact FastRows.row6073_checked.symm
  rw [row]
  decide
theorem root230Step14 : scan edge (search edge 3) 4 [1471] 1471 root230Tail14 winner =
    scan edge (search edge 3) 4 [1471] 1471 root230Tail15 winner := by
  rw [root230Tail14, scan, if_neg (by decide), root230Candidates14_checked]
  change scan edge (search edge 3) 4 [1471] 1471 root230Tail15
    (if 3 ≤ root230Candidates14.length then search edge 3 [1471, 6073] root230Candidates14 7544 winner else winner) = _
  rw [if_neg (by decide)]
def root230Candidates15 : List Nat := []
set_option maxRecDepth 8192 in
theorem root230Candidates15_checked : candidates edge 4 1471 6133 root230Tail16 winner.bound = root230Candidates15 := by
  unfold candidates
  have row : edge 6133 = (SortedNatTree.contains FastRows.row6133) := by
    change Data.row6133.contains = (SortedNatTree.contains FastRows.row6133)
    exact FastRows.row6133_checked.symm
  rw [row]
  decide
theorem root230Step15 : scan edge (search edge 3) 4 [1471] 1471 root230Tail15 winner =
    scan edge (search edge 3) 4 [1471] 1471 root230Tail16 winner := by
  rw [root230Tail15, scan, if_neg (by decide), root230Candidates15_checked]
  change scan edge (search edge 3) 4 [1471] 1471 root230Tail16
    (if 3 ≤ root230Candidates15.length then search edge 3 [1471, 6133] root230Candidates15 7604 winner else winner) = _
  rw [if_neg (by decide)]
theorem root230Scan16 : scan edge (search edge 3) 4 [1471] 1471 root230Tail16 winner = winner := by rw [root230Tail16, scan, if_pos (by decide)]
theorem root230Scan15 : scan edge (search edge 3) 4 [1471] 1471 root230Tail15 winner = winner :=
  (root230Step15).trans root230Scan16
theorem root230Scan14 : scan edge (search edge 3) 4 [1471] 1471 root230Tail14 winner = winner :=
  (root230Step14).trans root230Scan15
theorem root230Scan13 : scan edge (search edge 3) 4 [1471] 1471 root230Tail13 winner = winner :=
  (root230Step13).trans root230Scan14
theorem root230Scan12 : scan edge (search edge 3) 4 [1471] 1471 root230Tail12 winner = winner :=
  (root230Step12).trans root230Scan13
theorem root230Scan11 : scan edge (search edge 3) 4 [1471] 1471 root230Tail11 winner = winner :=
  (root230Step11).trans root230Scan12
theorem root230Scan10 : scan edge (search edge 3) 4 [1471] 1471 root230Tail10 winner = winner :=
  (root230Step10).trans root230Scan11
theorem root230Scan9 : scan edge (search edge 3) 4 [1471] 1471 root230Tail9 winner = winner :=
  (root230Step9).trans root230Scan10
theorem root230Scan8 : scan edge (search edge 3) 4 [1471] 1471 root230Tail8 winner = winner :=
  (root230Step8).trans root230Scan9
theorem root230Scan7 : scan edge (search edge 3) 4 [1471] 1471 root230Tail7 winner = winner :=
  (root230Step7).trans root230Scan8
theorem root230Scan6 : scan edge (search edge 3) 4 [1471] 1471 root230Tail6 winner = winner :=
  (root230Step6).trans root230Scan7
theorem root230Scan5 : scan edge (search edge 3) 4 [1471] 1471 root230Tail5 winner = winner :=
  (root230Step5).trans root230Scan6
theorem root230Scan4 : scan edge (search edge 3) 4 [1471] 1471 root230Tail4 winner = winner :=
  (root230Step4).trans root230Scan5
theorem root230Scan3 : scan edge (search edge 3) 4 [1471] 1471 root230Tail3 winner = winner :=
  (root230Step3).trans root230Scan4
theorem root230Scan2 : scan edge (search edge 3) 4 [1471] 1471 root230Tail2 winner = winner :=
  (root230Step2).trans root230Scan3
theorem root230Scan1 : scan edge (search edge 3) 4 [1471] 1471 root230Tail1 winner = winner :=
  (root230Step1).trans root230Scan2
theorem root230Scan0 : scan edge (search edge 3) 4 [1471] 1471 root230Tail0 winner = winner :=
  (root230Step0).trans root230Scan1
set_option maxRecDepth 4096 in
theorem root230Branch : search edge 4 [1471] root230Candidates 1471 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1471] 1471 root230Tail0 winner = winner
  exact root230Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
