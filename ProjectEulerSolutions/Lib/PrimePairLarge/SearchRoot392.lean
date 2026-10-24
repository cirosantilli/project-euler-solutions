import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root392Candidates : List Nat := Data.row2711
set_option maxRecDepth 8192 in
theorem root392Candidates_checked : candidates edge 5 0 2711 tail393 winner.bound = root392Candidates := by
  apply candidates_eq_row edge 5 0 2711 tail393 root392Candidates winner.bound tail393_ordered
  · decide
  · decide
  · funext q
    rfl
def root392Tail46 : List Nat := []
def root392Tail45 : List Nat := 23189 :: root392Tail46
def root392Tail44 : List Nat := 22961 :: root392Tail45
def root392Tail43 : List Nat := 22811 :: root392Tail44
def root392Tail42 : List Nat := 22283 :: root392Tail43
def root392Tail41 : List Nat := 21701 :: root392Tail42
def root392Tail40 : List Nat := 20357 :: root392Tail41
def root392Tail39 : List Nat := 18671 :: root392Tail40
def root392Tail38 : List Nat := 18521 :: root392Tail39
def root392Tail37 : List Nat := 18047 :: root392Tail38
def root392Tail36 : List Nat := 17417 :: root392Tail37
def root392Tail35 : List Nat := 17387 :: root392Tail36
def root392Tail34 : List Nat := 16829 :: root392Tail35
def root392Tail33 : List Nat := 16661 :: root392Tail34
def root392Tail32 : List Nat := 16217 :: root392Tail33
def root392Tail31 : List Nat := 14411 :: root392Tail32
def root392Tail30 : List Nat := 12917 :: root392Tail31
def root392Tail29 : List Nat := 12269 :: root392Tail30
def root392Tail28 : List Nat := 12251 :: root392Tail29
def root392Tail27 : List Nat := 12203 :: root392Tail28
def root392Tail26 : List Nat := 11783 :: root392Tail27
def root392Tail25 : List Nat := 11657 :: root392Tail26
def root392Tail24 : List Nat := 11261 :: root392Tail25
def root392Tail23 : List Nat := 10667 :: root392Tail24
def root392Tail22 : List Nat := 10061 :: root392Tail23
def root392Tail21 : List Nat := 9923 :: root392Tail22
def root392Tail20 : List Nat := 9431 :: root392Tail21
def root392Tail19 : List Nat := 9413 :: root392Tail20
def root392Tail18 : List Nat := 9371 :: root392Tail19
def root392Tail17 : List Nat := 9221 :: root392Tail18
def root392Tail16 : List Nat := 8747 :: root392Tail17
def root392Tail15 : List Nat := 7877 :: root392Tail16
def root392Tail14 : List Nat := 7817 :: root392Tail15
def root392Tail13 : List Nat := 7589 :: root392Tail14
def root392Tail12 : List Nat := 7457 :: root392Tail13
def root392Tail11 : List Nat := 7151 :: root392Tail12
def root392Tail10 : List Nat := 6491 :: root392Tail11
def root392Tail9 : List Nat := 6113 :: root392Tail10
def root392Tail8 : List Nat := 5741 :: root392Tail9
def root392Tail7 : List Nat := 5651 :: root392Tail8
def root392Tail6 : List Nat := 5441 :: root392Tail7
def root392Tail5 : List Nat := 4421 :: root392Tail6
def root392Tail4 : List Nat := 4211 :: root392Tail5
def root392Tail3 : List Nat := 3851 :: root392Tail4
def root392Tail2 : List Nat := 3413 :: root392Tail3
def root392Tail1 : List Nat := 3137 :: root392Tail2
def root392Tail0 : List Nat := 2963 :: root392Tail1
def root392Candidates0 : List Nat := [4421, 7151, 9371]
set_option maxRecDepth 8192 in
theorem root392Candidates0_checked : candidates edge 4 2711 2963 root392Tail1 winner.bound = root392Candidates0 := by
  unfold candidates
  have row : edge 2963 = (SortedNatTree.contains FastRows.row2963) := by
    change Data.row2963.contains = (SortedNatTree.contains FastRows.row2963)
    exact FastRows.row2963_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root392Child0 : search edge 3 [2711, 2963] root392Candidates0 5674 winner = winner := by decide
theorem root392Step0 : scan edge (search edge 3) 4 [2711] 2711 root392Tail0 winner =
    scan edge (search edge 3) 4 [2711] 2711 root392Tail1 winner := by
  rw [root392Tail0, scan, if_neg (by decide), root392Candidates0_checked]
  change scan edge (search edge 3) 4 [2711] 2711 root392Tail1
    (if 3 ≤ root392Candidates0.length then search edge 3 [2711, 2963] root392Candidates0 5674 winner else winner) = _
  rw [if_pos (by decide), root392Child0]
def root392Candidates1 : List Nat := [9923, 10667, 17387]
set_option maxRecDepth 8192 in
theorem root392Candidates1_checked : candidates edge 4 2711 3137 root392Tail2 winner.bound = root392Candidates1 := by
  unfold candidates
  have row : edge 3137 = (SortedNatTree.contains FastRows.row3137) := by
    change Data.row3137.contains = (SortedNatTree.contains FastRows.row3137)
    exact FastRows.row3137_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root392Child1 : search edge 3 [2711, 3137] root392Candidates1 5848 winner = winner := by decide
theorem root392Step1 : scan edge (search edge 3) 4 [2711] 2711 root392Tail1 winner =
    scan edge (search edge 3) 4 [2711] 2711 root392Tail2 winner := by
  rw [root392Tail1, scan, if_neg (by decide), root392Candidates1_checked]
  change scan edge (search edge 3) 4 [2711] 2711 root392Tail2
    (if 3 ≤ root392Candidates1.length then search edge 3 [2711, 3137] root392Candidates1 5848 winner else winner) = _
  rw [if_pos (by decide), root392Child1]
def root392Candidates2 : List Nat := [12269]
set_option maxRecDepth 8192 in
theorem root392Candidates2_checked : candidates edge 4 2711 3413 root392Tail3 winner.bound = root392Candidates2 := by
  unfold candidates
  have row : edge 3413 = (SortedNatTree.contains FastRows.row3413) := by
    change Data.row3413.contains = (SortedNatTree.contains FastRows.row3413)
    exact FastRows.row3413_checked.symm
  rw [row]
  decide
theorem root392Step2 : scan edge (search edge 3) 4 [2711] 2711 root392Tail2 winner =
    scan edge (search edge 3) 4 [2711] 2711 root392Tail3 winner := by
  rw [root392Tail2, scan, if_neg (by decide), root392Candidates2_checked]
  change scan edge (search edge 3) 4 [2711] 2711 root392Tail3
    (if 3 ≤ root392Candidates2.length then search edge 3 [2711, 3413] root392Candidates2 6124 winner else winner) = _
  rw [if_neg (by decide)]
def root392Candidates3 : List Nat := [5651, 7151, 16661]
set_option maxRecDepth 8192 in
theorem root392Candidates3_checked : candidates edge 4 2711 3851 root392Tail4 winner.bound = root392Candidates3 := by
  unfold candidates
  have row : edge 3851 = (SortedNatTree.contains FastRows.row3851) := by
    change Data.row3851.contains = (SortedNatTree.contains FastRows.row3851)
    exact FastRows.row3851_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root392Child3 : search edge 3 [2711, 3851] root392Candidates3 6562 winner = winner := by decide
theorem root392Step3 : scan edge (search edge 3) 4 [2711] 2711 root392Tail3 winner =
    scan edge (search edge 3) 4 [2711] 2711 root392Tail4 winner := by
  rw [root392Tail3, scan, if_neg (by decide), root392Candidates3_checked]
  change scan edge (search edge 3) 4 [2711] 2711 root392Tail4
    (if 3 ≤ root392Candidates3.length then search edge 3 [2711, 3851] root392Candidates3 6562 winner else winner) = _
  rw [if_pos (by decide), root392Child3]
def root392Candidates4 : List Nat := [9221, 18671]
set_option maxRecDepth 8192 in
theorem root392Candidates4_checked : candidates edge 4 2711 4211 root392Tail5 winner.bound = root392Candidates4 := by
  unfold candidates
  have row : edge 4211 = (SortedNatTree.contains FastRows.row4211) := by
    change Data.row4211.contains = (SortedNatTree.contains FastRows.row4211)
    exact FastRows.row4211_checked.symm
  rw [row]
  decide
theorem root392Step4 : scan edge (search edge 3) 4 [2711] 2711 root392Tail4 winner =
    scan edge (search edge 3) 4 [2711] 2711 root392Tail5 winner := by
  rw [root392Tail4, scan, if_neg (by decide), root392Candidates4_checked]
  change scan edge (search edge 3) 4 [2711] 2711 root392Tail5
    (if 3 ≤ root392Candidates4.length then search edge 3 [2711, 4211] root392Candidates4 6922 winner else winner) = _
  rw [if_neg (by decide)]
def root392Candidates5 : List Nat := [6113, 18671]
set_option maxRecDepth 8192 in
theorem root392Candidates5_checked : candidates edge 4 2711 4421 root392Tail6 winner.bound = root392Candidates5 := by
  unfold candidates
  have row : edge 4421 = (SortedNatTree.contains FastRows.row4421) := by
    change Data.row4421.contains = (SortedNatTree.contains FastRows.row4421)
    exact FastRows.row4421_checked.symm
  rw [row]
  decide
theorem root392Step5 : scan edge (search edge 3) 4 [2711] 2711 root392Tail5 winner =
    scan edge (search edge 3) 4 [2711] 2711 root392Tail6 winner := by
  rw [root392Tail5, scan, if_neg (by decide), root392Candidates5_checked]
  change scan edge (search edge 3) 4 [2711] 2711 root392Tail6
    (if 3 ≤ root392Candidates5.length then search edge 3 [2711, 4421] root392Candidates5 7132 winner else winner) = _
  rw [if_neg (by decide)]
def root392Candidates6 : List Nat := [7457, 11657]
set_option maxRecDepth 8192 in
theorem root392Candidates6_checked : candidates edge 4 2711 5441 root392Tail7 winner.bound = root392Candidates6 := by
  unfold candidates
  have row : edge 5441 = (SortedNatTree.contains FastRows.row5441) := by
    change Data.row5441.contains = (SortedNatTree.contains FastRows.row5441)
    exact FastRows.row5441_checked.symm
  rw [row]
  decide
theorem root392Step6 : scan edge (search edge 3) 4 [2711] 2711 root392Tail6 winner =
    scan edge (search edge 3) 4 [2711] 2711 root392Tail7 winner := by
  rw [root392Tail6, scan, if_neg (by decide), root392Candidates6_checked]
  change scan edge (search edge 3) 4 [2711] 2711 root392Tail7
    (if 3 ≤ root392Candidates6.length then search edge 3 [2711, 5441] root392Candidates6 8152 winner else winner) = _
  rw [if_neg (by decide)]
def root392Candidates7 : List Nat := [9413]
set_option maxRecDepth 8192 in
theorem root392Candidates7_checked : candidates edge 4 2711 5651 root392Tail8 winner.bound = root392Candidates7 := by
  unfold candidates
  have row : edge 5651 = (SortedNatTree.contains FastRows.row5651) := by
    change Data.row5651.contains = (SortedNatTree.contains FastRows.row5651)
    exact FastRows.row5651_checked.symm
  rw [row]
  decide
theorem root392Step7 : scan edge (search edge 3) 4 [2711] 2711 root392Tail7 winner =
    scan edge (search edge 3) 4 [2711] 2711 root392Tail8 winner := by
  rw [root392Tail7, scan, if_neg (by decide), root392Candidates7_checked]
  change scan edge (search edge 3) 4 [2711] 2711 root392Tail8
    (if 3 ≤ root392Candidates7.length then search edge 3 [2711, 5651] root392Candidates7 8362 winner else winner) = _
  rw [if_neg (by decide)]
def root392Candidates8 : List Nat := []
set_option maxRecDepth 8192 in
theorem root392Candidates8_checked : candidates edge 4 2711 5741 root392Tail9 winner.bound = root392Candidates8 := by
  unfold candidates
  have row : edge 5741 = (SortedNatTree.contains FastRows.row5741) := by
    change Data.row5741.contains = (SortedNatTree.contains FastRows.row5741)
    exact FastRows.row5741_checked.symm
  rw [row]
  decide
theorem root392Step8 : scan edge (search edge 3) 4 [2711] 2711 root392Tail8 winner =
    scan edge (search edge 3) 4 [2711] 2711 root392Tail9 winner := by
  rw [root392Tail8, scan, if_neg (by decide), root392Candidates8_checked]
  change scan edge (search edge 3) 4 [2711] 2711 root392Tail9
    (if 3 ≤ root392Candidates8.length then search edge 3 [2711, 5741] root392Candidates8 8452 winner else winner) = _
  rw [if_neg (by decide)]
theorem root392Scan9 : scan edge (search edge 3) 4 [2711] 2711 root392Tail9 winner = winner := by rw [root392Tail9, scan, if_pos (by decide)]
theorem root392Scan8 : scan edge (search edge 3) 4 [2711] 2711 root392Tail8 winner = winner :=
  (root392Step8).trans root392Scan9
theorem root392Scan7 : scan edge (search edge 3) 4 [2711] 2711 root392Tail7 winner = winner :=
  (root392Step7).trans root392Scan8
theorem root392Scan6 : scan edge (search edge 3) 4 [2711] 2711 root392Tail6 winner = winner :=
  (root392Step6).trans root392Scan7
theorem root392Scan5 : scan edge (search edge 3) 4 [2711] 2711 root392Tail5 winner = winner :=
  (root392Step5).trans root392Scan6
theorem root392Scan4 : scan edge (search edge 3) 4 [2711] 2711 root392Tail4 winner = winner :=
  (root392Step4).trans root392Scan5
theorem root392Scan3 : scan edge (search edge 3) 4 [2711] 2711 root392Tail3 winner = winner :=
  (root392Step3).trans root392Scan4
theorem root392Scan2 : scan edge (search edge 3) 4 [2711] 2711 root392Tail2 winner = winner :=
  (root392Step2).trans root392Scan3
theorem root392Scan1 : scan edge (search edge 3) 4 [2711] 2711 root392Tail1 winner = winner :=
  (root392Step1).trans root392Scan2
theorem root392Scan0 : scan edge (search edge 3) 4 [2711] 2711 root392Tail0 winner = winner :=
  (root392Step0).trans root392Scan1
set_option maxRecDepth 4096 in
theorem root392Branch : search edge 4 [2711] root392Candidates 2711 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2711] 2711 root392Tail0 winner = winner
  exact root392Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
