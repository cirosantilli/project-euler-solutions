import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root190Candidates : List Nat := Data.row1171
set_option maxRecDepth 8192 in
theorem root190Candidates_checked : candidates edge 5 0 1171 tail191 winner.bound = root190Candidates := by
  apply candidates_eq_row edge 5 0 1171 tail191 root190Candidates winner.bound tail191_ordered
  · decide
  · decide
  · funext q
    rfl
def root190Tail44 : List Nat := []
def root190Tail43 : List Nat := 24043 :: root190Tail44
def root190Tail42 : List Nat := 23857 :: root190Tail43
def root190Tail41 : List Nat := 23059 :: root190Tail42
def root190Tail40 : List Nat := 21061 :: root190Tail41
def root190Tail39 : List Nat := 19963 :: root190Tail40
def root190Tail38 : List Nat := 19759 :: root190Tail39
def root190Tail37 : List Nat := 19447 :: root190Tail38
def root190Tail36 : List Nat := 18583 :: root190Tail37
def root190Tail35 : List Nat := 18493 :: root190Tail36
def root190Tail34 : List Nat := 18289 :: root190Tail35
def root190Tail33 : List Nat := 16381 :: root190Tail34
def root190Tail32 : List Nat := 15511 :: root190Tail33
def root190Tail31 : List Nat := 14293 :: root190Tail32
def root190Tail30 : List Nat := 14281 :: root190Tail31
def root190Tail29 : List Nat := 13831 :: root190Tail30
def root190Tail28 : List Nat := 12907 :: root190Tail29
def root190Tail27 : List Nat := 12301 :: root190Tail28
def root190Tail26 : List Nat := 11971 :: root190Tail27
def root190Tail25 : List Nat := 11719 :: root190Tail26
def root190Tail24 : List Nat := 11551 :: root190Tail25
def root190Tail23 : List Nat := 11443 :: root190Tail24
def root190Tail22 : List Nat := 10459 :: root190Tail23
def root190Tail21 : List Nat := 10099 :: root190Tail22
def root190Tail20 : List Nat := 9199 :: root190Tail21
def root190Tail19 : List Nat := 9133 :: root190Tail20
def root190Tail18 : List Nat := 9109 :: root190Tail19
def root190Tail17 : List Nat := 8821 :: root190Tail18
def root190Tail16 : List Nat := 8647 :: root190Tail17
def root190Tail15 : List Nat := 8461 :: root190Tail16
def root190Tail14 : List Nat := 7963 :: root190Tail15
def root190Tail13 : List Nat := 7873 :: root190Tail14
def root190Tail12 : List Nat := 6337 :: root190Tail13
def root190Tail11 : List Nat := 6007 :: root190Tail12
def root190Tail10 : List Nat := 5881 :: root190Tail11
def root190Tail9 : List Nat := 4951 :: root190Tail10
def root190Tail8 : List Nat := 4933 :: root190Tail9
def root190Tail7 : List Nat := 4663 :: root190Tail8
def root190Tail6 : List Nat := 3607 :: root190Tail7
def root190Tail5 : List Nat := 2161 :: root190Tail6
def root190Tail4 : List Nat := 2137 :: root190Tail5
def root190Tail3 : List Nat := 2053 :: root190Tail4
def root190Tail2 : List Nat := 2011 :: root190Tail3
def root190Tail1 : List Nat := 1699 :: root190Tail2
def root190Tail0 : List Nat := 1303 :: root190Tail1
def root190Candidates0 : List Nat := [8647, 9109, 15511]
set_option maxRecDepth 8192 in
theorem root190Candidates0_checked : candidates edge 4 1171 1303 root190Tail1 winner.bound = root190Candidates0 := by
  unfold candidates
  have row : edge 1303 = (SortedNatTree.contains FastRows.row1303) := by
    change Data.row1303.contains = (SortedNatTree.contains FastRows.row1303)
    exact FastRows.row1303_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root190Child0 : search edge 3 [1171, 1303] root190Candidates0 2474 winner = winner := by decide
theorem root190Step0 : scan edge (search edge 3) 4 [1171] 1171 root190Tail0 winner =
    scan edge (search edge 3) 4 [1171] 1171 root190Tail1 winner := by
  rw [root190Tail0, scan, if_neg (by decide), root190Candidates0_checked]
  change scan edge (search edge 3) 4 [1171] 1171 root190Tail1
    (if 3 ≤ root190Candidates0.length then search edge 3 [1171, 1303] root190Candidates0 2474 winner else winner) = _
  rw [if_pos (by decide), root190Child0]
def root190Candidates1 : List Nat := [3607, 15511]
set_option maxRecDepth 8192 in
theorem root190Candidates1_checked : candidates edge 4 1171 1699 root190Tail2 winner.bound = root190Candidates1 := by
  unfold candidates
  have row : edge 1699 = (SortedNatTree.contains FastRows.row1699) := by
    change Data.row1699.contains = (SortedNatTree.contains FastRows.row1699)
    exact FastRows.row1699_checked.symm
  rw [row]
  decide
theorem root190Step1 : scan edge (search edge 3) 4 [1171] 1171 root190Tail1 winner =
    scan edge (search edge 3) 4 [1171] 1171 root190Tail2 winner := by
  rw [root190Tail1, scan, if_neg (by decide), root190Candidates1_checked]
  change scan edge (search edge 3) 4 [1171] 1171 root190Tail2
    (if 3 ≤ root190Candidates1.length then search edge 3 [1171, 1699] root190Candidates1 2870 winner else winner) = _
  rw [if_neg (by decide)]
def root190Candidates2 : List Nat := [7963, 14293]
set_option maxRecDepth 8192 in
theorem root190Candidates2_checked : candidates edge 4 1171 2011 root190Tail3 winner.bound = root190Candidates2 := by
  unfold candidates
  have row : edge 2011 = (SortedNatTree.contains FastRows.row2011) := by
    change Data.row2011.contains = (SortedNatTree.contains FastRows.row2011)
    exact FastRows.row2011_checked.symm
  rw [row]
  decide
theorem root190Step2 : scan edge (search edge 3) 4 [1171] 1171 root190Tail2 winner =
    scan edge (search edge 3) 4 [1171] 1171 root190Tail3 winner := by
  rw [root190Tail2, scan, if_neg (by decide), root190Candidates2_checked]
  change scan edge (search edge 3) 4 [1171] 1171 root190Tail3
    (if 3 ≤ root190Candidates2.length then search edge 3 [1171, 2011] root190Candidates2 3182 winner else winner) = _
  rw [if_neg (by decide)]
def root190Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root190Candidates3_checked : candidates edge 4 1171 2053 root190Tail4 winner.bound = root190Candidates3 := by
  unfold candidates
  have row : edge 2053 = (SortedNatTree.contains FastRows.row2053) := by
    change Data.row2053.contains = (SortedNatTree.contains FastRows.row2053)
    exact FastRows.row2053_checked.symm
  rw [row]
  decide
theorem root190Step3 : scan edge (search edge 3) 4 [1171] 1171 root190Tail3 winner =
    scan edge (search edge 3) 4 [1171] 1171 root190Tail4 winner := by
  rw [root190Tail3, scan, if_neg (by decide), root190Candidates3_checked]
  change scan edge (search edge 3) 4 [1171] 1171 root190Tail4
    (if 3 ≤ root190Candidates3.length then search edge 3 [1171, 2053] root190Candidates3 3224 winner else winner) = _
  rw [if_neg (by decide)]
def root190Candidates4 : List Nat := [10099]
set_option maxRecDepth 8192 in
theorem root190Candidates4_checked : candidates edge 4 1171 2137 root190Tail5 winner.bound = root190Candidates4 := by
  unfold candidates
  have row : edge 2137 = (SortedNatTree.contains FastRows.row2137) := by
    change Data.row2137.contains = (SortedNatTree.contains FastRows.row2137)
    exact FastRows.row2137_checked.symm
  rw [row]
  decide
theorem root190Step4 : scan edge (search edge 3) 4 [1171] 1171 root190Tail4 winner =
    scan edge (search edge 3) 4 [1171] 1171 root190Tail5 winner := by
  rw [root190Tail4, scan, if_neg (by decide), root190Candidates4_checked]
  change scan edge (search edge 3) 4 [1171] 1171 root190Tail5
    (if 3 ≤ root190Candidates4.length then search edge 3 [1171, 2137] root190Candidates4 3308 winner else winner) = _
  rw [if_neg (by decide)]
def root190Candidates5 : List Nat := [6337, 18289]
set_option maxRecDepth 8192 in
theorem root190Candidates5_checked : candidates edge 4 1171 2161 root190Tail6 winner.bound = root190Candidates5 := by
  unfold candidates
  have row : edge 2161 = (SortedNatTree.contains FastRows.row2161) := by
    change Data.row2161.contains = (SortedNatTree.contains FastRows.row2161)
    exact FastRows.row2161_checked.symm
  rw [row]
  decide
theorem root190Step5 : scan edge (search edge 3) 4 [1171] 1171 root190Tail5 winner =
    scan edge (search edge 3) 4 [1171] 1171 root190Tail6 winner := by
  rw [root190Tail5, scan, if_neg (by decide), root190Candidates5_checked]
  change scan edge (search edge 3) 4 [1171] 1171 root190Tail6
    (if 3 ≤ root190Candidates5.length then search edge 3 [1171, 2161] root190Candidates5 3332 winner else winner) = _
  rw [if_neg (by decide)]
def root190Candidates6 : List Nat := [16381, 18493]
set_option maxRecDepth 8192 in
theorem root190Candidates6_checked : candidates edge 4 1171 3607 root190Tail7 winner.bound = root190Candidates6 := by
  unfold candidates
  have row : edge 3607 = (SortedNatTree.contains FastRows.row3607) := by
    change Data.row3607.contains = (SortedNatTree.contains FastRows.row3607)
    exact FastRows.row3607_checked.symm
  rw [row]
  decide
theorem root190Step6 : scan edge (search edge 3) 4 [1171] 1171 root190Tail6 winner =
    scan edge (search edge 3) 4 [1171] 1171 root190Tail7 winner := by
  rw [root190Tail6, scan, if_neg (by decide), root190Candidates6_checked]
  change scan edge (search edge 3) 4 [1171] 1171 root190Tail7
    (if 3 ≤ root190Candidates6.length then search edge 3 [1171, 3607] root190Candidates6 4778 winner else winner) = _
  rw [if_neg (by decide)]
def root190Candidates7 : List Nat := [9199]
set_option maxRecDepth 8192 in
theorem root190Candidates7_checked : candidates edge 4 1171 4663 root190Tail8 winner.bound = root190Candidates7 := by
  unfold candidates
  have row : edge 4663 = (SortedNatTree.contains FastRows.row4663) := by
    change Data.row4663.contains = (SortedNatTree.contains FastRows.row4663)
    exact FastRows.row4663_checked.symm
  rw [row]
  decide
theorem root190Step7 : scan edge (search edge 3) 4 [1171] 1171 root190Tail7 winner =
    scan edge (search edge 3) 4 [1171] 1171 root190Tail8 winner := by
  rw [root190Tail7, scan, if_neg (by decide), root190Candidates7_checked]
  change scan edge (search edge 3) 4 [1171] 1171 root190Tail8
    (if 3 ≤ root190Candidates7.length then search edge 3 [1171, 4663] root190Candidates7 5834 winner else winner) = _
  rw [if_neg (by decide)]
def root190Candidates8 : List Nat := [14293]
set_option maxRecDepth 8192 in
theorem root190Candidates8_checked : candidates edge 4 1171 4933 root190Tail9 winner.bound = root190Candidates8 := by
  unfold candidates
  have row : edge 4933 = (SortedNatTree.contains FastRows.row4933) := by
    change Data.row4933.contains = (SortedNatTree.contains FastRows.row4933)
    exact FastRows.row4933_checked.symm
  rw [row]
  decide
theorem root190Step8 : scan edge (search edge 3) 4 [1171] 1171 root190Tail8 winner =
    scan edge (search edge 3) 4 [1171] 1171 root190Tail9 winner := by
  rw [root190Tail8, scan, if_neg (by decide), root190Candidates8_checked]
  change scan edge (search edge 3) 4 [1171] 1171 root190Tail9
    (if 3 ≤ root190Candidates8.length then search edge 3 [1171, 4933] root190Candidates8 6104 winner else winner) = _
  rw [if_neg (by decide)]
def root190Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root190Candidates9_checked : candidates edge 4 1171 4951 root190Tail10 winner.bound = root190Candidates9 := by
  unfold candidates
  have row : edge 4951 = (SortedNatTree.contains FastRows.row4951) := by
    change Data.row4951.contains = (SortedNatTree.contains FastRows.row4951)
    exact FastRows.row4951_checked.symm
  rw [row]
  decide
theorem root190Step9 : scan edge (search edge 3) 4 [1171] 1171 root190Tail9 winner =
    scan edge (search edge 3) 4 [1171] 1171 root190Tail10 winner := by
  rw [root190Tail9, scan, if_neg (by decide), root190Candidates9_checked]
  change scan edge (search edge 3) 4 [1171] 1171 root190Tail10
    (if 3 ≤ root190Candidates9.length then search edge 3 [1171, 4951] root190Candidates9 6122 winner else winner) = _
  rw [if_neg (by decide)]
def root190Candidates10 : List Nat := [11971]
set_option maxRecDepth 8192 in
theorem root190Candidates10_checked : candidates edge 4 1171 5881 root190Tail11 winner.bound = root190Candidates10 := by
  unfold candidates
  have row : edge 5881 = (SortedNatTree.contains FastRows.row5881) := by
    change Data.row5881.contains = (SortedNatTree.contains FastRows.row5881)
    exact FastRows.row5881_checked.symm
  rw [row]
  decide
theorem root190Step10 : scan edge (search edge 3) 4 [1171] 1171 root190Tail10 winner =
    scan edge (search edge 3) 4 [1171] 1171 root190Tail11 winner := by
  rw [root190Tail10, scan, if_neg (by decide), root190Candidates10_checked]
  change scan edge (search edge 3) 4 [1171] 1171 root190Tail11
    (if 3 ≤ root190Candidates10.length then search edge 3 [1171, 5881] root190Candidates10 7052 winner else winner) = _
  rw [if_neg (by decide)]
def root190Candidates11 : List Nat := [13831, 14281]
set_option maxRecDepth 8192 in
theorem root190Candidates11_checked : candidates edge 4 1171 6007 root190Tail12 winner.bound = root190Candidates11 := by
  unfold candidates
  have row : edge 6007 = (SortedNatTree.contains FastRows.row6007) := by
    change Data.row6007.contains = (SortedNatTree.contains FastRows.row6007)
    exact FastRows.row6007_checked.symm
  rw [row]
  decide
theorem root190Step11 : scan edge (search edge 3) 4 [1171] 1171 root190Tail11 winner =
    scan edge (search edge 3) 4 [1171] 1171 root190Tail12 winner := by
  rw [root190Tail11, scan, if_neg (by decide), root190Candidates11_checked]
  change scan edge (search edge 3) 4 [1171] 1171 root190Tail12
    (if 3 ≤ root190Candidates11.length then search edge 3 [1171, 6007] root190Candidates11 7178 winner else winner) = _
  rw [if_neg (by decide)]
theorem root190Scan12 : scan edge (search edge 3) 4 [1171] 1171 root190Tail12 winner = winner := by rw [root190Tail12, scan, if_pos (by decide)]
theorem root190Scan11 : scan edge (search edge 3) 4 [1171] 1171 root190Tail11 winner = winner :=
  (root190Step11).trans root190Scan12
theorem root190Scan10 : scan edge (search edge 3) 4 [1171] 1171 root190Tail10 winner = winner :=
  (root190Step10).trans root190Scan11
theorem root190Scan9 : scan edge (search edge 3) 4 [1171] 1171 root190Tail9 winner = winner :=
  (root190Step9).trans root190Scan10
theorem root190Scan8 : scan edge (search edge 3) 4 [1171] 1171 root190Tail8 winner = winner :=
  (root190Step8).trans root190Scan9
theorem root190Scan7 : scan edge (search edge 3) 4 [1171] 1171 root190Tail7 winner = winner :=
  (root190Step7).trans root190Scan8
theorem root190Scan6 : scan edge (search edge 3) 4 [1171] 1171 root190Tail6 winner = winner :=
  (root190Step6).trans root190Scan7
theorem root190Scan5 : scan edge (search edge 3) 4 [1171] 1171 root190Tail5 winner = winner :=
  (root190Step5).trans root190Scan6
theorem root190Scan4 : scan edge (search edge 3) 4 [1171] 1171 root190Tail4 winner = winner :=
  (root190Step4).trans root190Scan5
theorem root190Scan3 : scan edge (search edge 3) 4 [1171] 1171 root190Tail3 winner = winner :=
  (root190Step3).trans root190Scan4
theorem root190Scan2 : scan edge (search edge 3) 4 [1171] 1171 root190Tail2 winner = winner :=
  (root190Step2).trans root190Scan3
theorem root190Scan1 : scan edge (search edge 3) 4 [1171] 1171 root190Tail1 winner = winner :=
  (root190Step1).trans root190Scan2
theorem root190Scan0 : scan edge (search edge 3) 4 [1171] 1171 root190Tail0 winner = winner :=
  (root190Step0).trans root190Scan1
set_option maxRecDepth 4096 in
theorem root190Branch : search edge 4 [1171] root190Candidates 1171 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1171] 1171 root190Tail0 winner = winner
  exact root190Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
