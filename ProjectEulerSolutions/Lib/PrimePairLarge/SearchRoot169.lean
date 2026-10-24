import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root169Candidates : List Nat := Data.row1021
set_option maxRecDepth 8192 in
theorem root169Candidates_checked : candidates edge 5 0 1021 tail170 winner.bound = root169Candidates := by
  apply candidates_eq_row edge 5 0 1021 tail170 root169Candidates winner.bound tail170_ordered
  · decide
  · decide
  · funext q
    rfl
def root169Tail56 : List Nat := []
def root169Tail55 : List Nat := 24223 :: root169Tail56
def root169Tail54 : List Nat := 23929 :: root169Tail55
def root169Tail53 : List Nat := 23899 :: root169Tail54
def root169Tail52 : List Nat := 23227 :: root169Tail53
def root169Tail51 : List Nat := 23143 :: root169Tail52
def root169Tail50 : List Nat := 21871 :: root169Tail51
def root169Tail49 : List Nat := 21031 :: root169Tail50
def root169Tail48 : List Nat := 20857 :: root169Tail49
def root169Tail47 : List Nat := 20521 :: root169Tail48
def root169Tail46 : List Nat := 20443 :: root169Tail47
def root169Tail45 : List Nat := 19543 :: root169Tail46
def root169Tail44 : List Nat := 19531 :: root169Tail45
def root169Tail43 : List Nat := 19309 :: root169Tail44
def root169Tail42 : List Nat := 19267 :: root169Tail43
def root169Tail41 : List Nat := 18637 :: root169Tail42
def root169Tail40 : List Nat := 18253 :: root169Tail41
def root169Tail39 : List Nat := 18127 :: root169Tail40
def root169Tail38 : List Nat := 17791 :: root169Tail39
def root169Tail37 : List Nat := 17683 :: root169Tail38
def root169Tail36 : List Nat := 17203 :: root169Tail37
def root169Tail35 : List Nat := 17053 :: root169Tail36
def root169Tail34 : List Nat := 16741 :: root169Tail35
def root169Tail33 : List Nat := 16699 :: root169Tail34
def root169Tail32 : List Nat := 16141 :: root169Tail33
def root169Tail31 : List Nat := 15733 :: root169Tail32
def root169Tail30 : List Nat := 14011 :: root169Tail31
def root169Tail29 : List Nat := 13633 :: root169Tail30
def root169Tail28 : List Nat := 12739 :: root169Tail29
def root169Tail27 : List Nat := 12541 :: root169Tail28
def root169Tail26 : List Nat := 12211 :: root169Tail27
def root169Tail25 : List Nat := 12163 :: root169Tail26
def root169Tail24 : List Nat := 11617 :: root169Tail25
def root169Tail23 : List Nat := 11047 :: root169Tail24
def root169Tail22 : List Nat := 10069 :: root169Tail23
def root169Tail21 : List Nat := 9883 :: root169Tail22
def root169Tail20 : List Nat := 9787 :: root169Tail21
def root169Tail19 : List Nat := 9103 :: root169Tail20
def root169Tail18 : List Nat := 8629 :: root169Tail19
def root169Tail17 : List Nat := 8431 :: root169Tail18
def root169Tail16 : List Nat := 7873 :: root169Tail17
def root169Tail15 : List Nat := 7213 :: root169Tail16
def root169Tail14 : List Nat := 6691 :: root169Tail15
def root169Tail13 : List Nat := 6397 :: root169Tail14
def root169Tail12 : List Nat := 6373 :: root169Tail13
def root169Tail11 : List Nat := 6247 :: root169Tail12
def root169Tail10 : List Nat := 6079 :: root169Tail11
def root169Tail9 : List Nat := 5701 :: root169Tail10
def root169Tail8 : List Nat := 5281 :: root169Tail9
def root169Tail7 : List Nat := 4597 :: root169Tail8
def root169Tail6 : List Nat := 3769 :: root169Tail7
def root169Tail5 : List Nat := 3571 :: root169Tail6
def root169Tail4 : List Nat := 3559 :: root169Tail5
def root169Tail3 : List Nat := 3529 :: root169Tail4
def root169Tail2 : List Nat := 2749 :: root169Tail3
def root169Tail1 : List Nat := 2539 :: root169Tail2
def root169Tail0 : List Nat := 1699 :: root169Tail1
def root169Candidates0 : List Nat := [6691]
set_option maxRecDepth 8192 in
theorem root169Candidates0_checked : candidates edge 4 1021 1699 root169Tail1 winner.bound = root169Candidates0 := by
  unfold candidates
  have row : edge 1699 = (SortedNatTree.contains FastRows.row1699) := by
    change Data.row1699.contains = (SortedNatTree.contains FastRows.row1699)
    exact FastRows.row1699_checked.symm
  rw [row]
  decide
theorem root169Step0 : scan edge (search edge 3) 4 [1021] 1021 root169Tail0 winner =
    scan edge (search edge 3) 4 [1021] 1021 root169Tail1 winner := by
  rw [root169Tail0, scan, if_neg (by decide), root169Candidates0_checked]
  change scan edge (search edge 3) 4 [1021] 1021 root169Tail1
    (if 3 ≤ root169Candidates0.length then search edge 3 [1021, 1699] root169Candidates0 2720 winner else winner) = _
  rw [if_neg (by decide)]
def root169Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root169Candidates1_checked : candidates edge 4 1021 2539 root169Tail2 winner.bound = root169Candidates1 := by
  unfold candidates
  have row : edge 2539 = (SortedNatTree.contains FastRows.row2539) := by
    change Data.row2539.contains = (SortedNatTree.contains FastRows.row2539)
    exact FastRows.row2539_checked.symm
  rw [row]
  decide
theorem root169Step1 : scan edge (search edge 3) 4 [1021] 1021 root169Tail1 winner =
    scan edge (search edge 3) 4 [1021] 1021 root169Tail2 winner := by
  rw [root169Tail1, scan, if_neg (by decide), root169Candidates1_checked]
  change scan edge (search edge 3) 4 [1021] 1021 root169Tail2
    (if 3 ≤ root169Candidates1.length then search edge 3 [1021, 2539] root169Candidates1 3560 winner else winner) = _
  rw [if_neg (by decide)]
def root169Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root169Candidates2_checked : candidates edge 4 1021 2749 root169Tail3 winner.bound = root169Candidates2 := by
  unfold candidates
  have row : edge 2749 = (SortedNatTree.contains FastRows.row2749) := by
    change Data.row2749.contains = (SortedNatTree.contains FastRows.row2749)
    exact FastRows.row2749_checked.symm
  rw [row]
  decide
theorem root169Step2 : scan edge (search edge 3) 4 [1021] 1021 root169Tail2 winner =
    scan edge (search edge 3) 4 [1021] 1021 root169Tail3 winner := by
  rw [root169Tail2, scan, if_neg (by decide), root169Candidates2_checked]
  change scan edge (search edge 3) 4 [1021] 1021 root169Tail3
    (if 3 ≤ root169Candidates2.length then search edge 3 [1021, 2749] root169Candidates2 3770 winner else winner) = _
  rw [if_neg (by decide)]
def root169Candidates3 : List Nat := [6397, 10069, 12211, 16699, 20521, 20857]
set_option maxRecDepth 8192 in
theorem root169Candidates3_checked : candidates edge 4 1021 3529 root169Tail4 winner.bound = root169Candidates3 := by
  unfold candidates
  have row : edge 3529 = (SortedNatTree.contains FastRows.row3529) := by
    change Data.row3529.contains = (SortedNatTree.contains FastRows.row3529)
    exact FastRows.row3529_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root169Child3 : search edge 3 [1021, 3529] root169Candidates3 4550 winner = winner := by decide
theorem root169Step3 : scan edge (search edge 3) 4 [1021] 1021 root169Tail3 winner =
    scan edge (search edge 3) 4 [1021] 1021 root169Tail4 winner := by
  rw [root169Tail3, scan, if_neg (by decide), root169Candidates3_checked]
  change scan edge (search edge 3) 4 [1021] 1021 root169Tail4
    (if 3 ≤ root169Candidates3.length then search edge 3 [1021, 3529] root169Candidates3 4550 winner else winner) = _
  rw [if_pos (by decide), root169Child3]
def root169Candidates4 : List Nat := [3769, 18637]
set_option maxRecDepth 8192 in
theorem root169Candidates4_checked : candidates edge 4 1021 3559 root169Tail5 winner.bound = root169Candidates4 := by
  unfold candidates
  have row : edge 3559 = (SortedNatTree.contains FastRows.row3559) := by
    change Data.row3559.contains = (SortedNatTree.contains FastRows.row3559)
    exact FastRows.row3559_checked.symm
  rw [row]
  decide
theorem root169Step4 : scan edge (search edge 3) 4 [1021] 1021 root169Tail4 winner =
    scan edge (search edge 3) 4 [1021] 1021 root169Tail5 winner := by
  rw [root169Tail4, scan, if_neg (by decide), root169Candidates4_checked]
  change scan edge (search edge 3) 4 [1021] 1021 root169Tail5
    (if 3 ≤ root169Candidates4.length then search edge 3 [1021, 3559] root169Candidates4 4580 winner else winner) = _
  rw [if_neg (by decide)]
def root169Candidates5 : List Nat := [17203, 18127]
set_option maxRecDepth 8192 in
theorem root169Candidates5_checked : candidates edge 4 1021 3571 root169Tail6 winner.bound = root169Candidates5 := by
  unfold candidates
  have row : edge 3571 = (SortedNatTree.contains FastRows.row3571) := by
    change Data.row3571.contains = (SortedNatTree.contains FastRows.row3571)
    exact FastRows.row3571_checked.symm
  rw [row]
  decide
theorem root169Step5 : scan edge (search edge 3) 4 [1021] 1021 root169Tail5 winner =
    scan edge (search edge 3) 4 [1021] 1021 root169Tail6 winner := by
  rw [root169Tail5, scan, if_neg (by decide), root169Candidates5_checked]
  change scan edge (search edge 3) 4 [1021] 1021 root169Tail6
    (if 3 ≤ root169Candidates5.length then search edge 3 [1021, 3571] root169Candidates5 4592 winner else winner) = _
  rw [if_neg (by decide)]
def root169Candidates6 : List Nat := [4597]
set_option maxRecDepth 8192 in
theorem root169Candidates6_checked : candidates edge 4 1021 3769 root169Tail7 winner.bound = root169Candidates6 := by
  unfold candidates
  have row : edge 3769 = (SortedNatTree.contains FastRows.row3769) := by
    change Data.row3769.contains = (SortedNatTree.contains FastRows.row3769)
    exact FastRows.row3769_checked.symm
  rw [row]
  decide
theorem root169Step6 : scan edge (search edge 3) 4 [1021] 1021 root169Tail6 winner =
    scan edge (search edge 3) 4 [1021] 1021 root169Tail7 winner := by
  rw [root169Tail6, scan, if_neg (by decide), root169Candidates6_checked]
  change scan edge (search edge 3) 4 [1021] 1021 root169Tail7
    (if 3 ≤ root169Candidates6.length then search edge 3 [1021, 3769] root169Candidates6 4790 winner else winner) = _
  rw [if_neg (by decide)]
def root169Candidates7 : List Nat := [6079, 7213]
set_option maxRecDepth 8192 in
theorem root169Candidates7_checked : candidates edge 4 1021 4597 root169Tail8 winner.bound = root169Candidates7 := by
  unfold candidates
  have row : edge 4597 = (SortedNatTree.contains FastRows.row4597) := by
    change Data.row4597.contains = (SortedNatTree.contains FastRows.row4597)
    exact FastRows.row4597_checked.symm
  rw [row]
  decide
theorem root169Step7 : scan edge (search edge 3) 4 [1021] 1021 root169Tail7 winner =
    scan edge (search edge 3) 4 [1021] 1021 root169Tail8 winner := by
  rw [root169Tail7, scan, if_neg (by decide), root169Candidates7_checked]
  change scan edge (search edge 3) 4 [1021] 1021 root169Tail8
    (if 3 ≤ root169Candidates7.length then search edge 3 [1021, 4597] root169Candidates7 5618 winner else winner) = _
  rw [if_neg (by decide)]
def root169Candidates8 : List Nat := [7873]
set_option maxRecDepth 8192 in
theorem root169Candidates8_checked : candidates edge 4 1021 5281 root169Tail9 winner.bound = root169Candidates8 := by
  unfold candidates
  have row : edge 5281 = (SortedNatTree.contains FastRows.row5281) := by
    change Data.row5281.contains = (SortedNatTree.contains FastRows.row5281)
    exact FastRows.row5281_checked.symm
  rw [row]
  decide
theorem root169Step8 : scan edge (search edge 3) 4 [1021] 1021 root169Tail8 winner =
    scan edge (search edge 3) 4 [1021] 1021 root169Tail9 winner := by
  rw [root169Tail8, scan, if_neg (by decide), root169Candidates8_checked]
  change scan edge (search edge 3) 4 [1021] 1021 root169Tail9
    (if 3 ≤ root169Candidates8.length then search edge 3 [1021, 5281] root169Candidates8 6302 winner else winner) = _
  rw [if_neg (by decide)]
def root169Candidates9 : List Nat := [17203]
set_option maxRecDepth 8192 in
theorem root169Candidates9_checked : candidates edge 4 1021 5701 root169Tail10 winner.bound = root169Candidates9 := by
  unfold candidates
  have row : edge 5701 = (SortedNatTree.contains FastRows.row5701) := by
    change Data.row5701.contains = (SortedNatTree.contains FastRows.row5701)
    exact FastRows.row5701_checked.symm
  rw [row]
  decide
theorem root169Step9 : scan edge (search edge 3) 4 [1021] 1021 root169Tail9 winner =
    scan edge (search edge 3) 4 [1021] 1021 root169Tail10 winner := by
  rw [root169Tail9, scan, if_neg (by decide), root169Candidates9_checked]
  change scan edge (search edge 3) 4 [1021] 1021 root169Tail10
    (if 3 ≤ root169Candidates9.length then search edge 3 [1021, 5701] root169Candidates9 6722 winner else winner) = _
  rw [if_neg (by decide)]
def root169Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root169Candidates10_checked : candidates edge 4 1021 6079 root169Tail11 winner.bound = root169Candidates10 := by
  unfold candidates
  have row : edge 6079 = (SortedNatTree.contains FastRows.row6079) := by
    change Data.row6079.contains = (SortedNatTree.contains FastRows.row6079)
    exact FastRows.row6079_checked.symm
  rw [row]
  decide
theorem root169Step10 : scan edge (search edge 3) 4 [1021] 1021 root169Tail10 winner =
    scan edge (search edge 3) 4 [1021] 1021 root169Tail11 winner := by
  rw [root169Tail10, scan, if_neg (by decide), root169Candidates10_checked]
  change scan edge (search edge 3) 4 [1021] 1021 root169Tail11
    (if 3 ≤ root169Candidates10.length then search edge 3 [1021, 6079] root169Candidates10 7100 winner else winner) = _
  rw [if_neg (by decide)]
def root169Candidates11 : List Nat := [6691]
set_option maxRecDepth 8192 in
theorem root169Candidates11_checked : candidates edge 4 1021 6247 root169Tail12 winner.bound = root169Candidates11 := by
  unfold candidates
  have row : edge 6247 = (SortedNatTree.contains FastRows.row6247) := by
    change Data.row6247.contains = (SortedNatTree.contains FastRows.row6247)
    exact FastRows.row6247_checked.symm
  rw [row]
  decide
theorem root169Step11 : scan edge (search edge 3) 4 [1021] 1021 root169Tail11 winner =
    scan edge (search edge 3) 4 [1021] 1021 root169Tail12 winner := by
  rw [root169Tail11, scan, if_neg (by decide), root169Candidates11_checked]
  change scan edge (search edge 3) 4 [1021] 1021 root169Tail12
    (if 3 ≤ root169Candidates11.length then search edge 3 [1021, 6247] root169Candidates11 7268 winner else winner) = _
  rw [if_neg (by decide)]
theorem root169Scan12 : scan edge (search edge 3) 4 [1021] 1021 root169Tail12 winner = winner := by rw [root169Tail12, scan, if_pos (by decide)]
theorem root169Scan11 : scan edge (search edge 3) 4 [1021] 1021 root169Tail11 winner = winner :=
  (root169Step11).trans root169Scan12
theorem root169Scan10 : scan edge (search edge 3) 4 [1021] 1021 root169Tail10 winner = winner :=
  (root169Step10).trans root169Scan11
theorem root169Scan9 : scan edge (search edge 3) 4 [1021] 1021 root169Tail9 winner = winner :=
  (root169Step9).trans root169Scan10
theorem root169Scan8 : scan edge (search edge 3) 4 [1021] 1021 root169Tail8 winner = winner :=
  (root169Step8).trans root169Scan9
theorem root169Scan7 : scan edge (search edge 3) 4 [1021] 1021 root169Tail7 winner = winner :=
  (root169Step7).trans root169Scan8
theorem root169Scan6 : scan edge (search edge 3) 4 [1021] 1021 root169Tail6 winner = winner :=
  (root169Step6).trans root169Scan7
theorem root169Scan5 : scan edge (search edge 3) 4 [1021] 1021 root169Tail5 winner = winner :=
  (root169Step5).trans root169Scan6
theorem root169Scan4 : scan edge (search edge 3) 4 [1021] 1021 root169Tail4 winner = winner :=
  (root169Step4).trans root169Scan5
theorem root169Scan3 : scan edge (search edge 3) 4 [1021] 1021 root169Tail3 winner = winner :=
  (root169Step3).trans root169Scan4
theorem root169Scan2 : scan edge (search edge 3) 4 [1021] 1021 root169Tail2 winner = winner :=
  (root169Step2).trans root169Scan3
theorem root169Scan1 : scan edge (search edge 3) 4 [1021] 1021 root169Tail1 winner = winner :=
  (root169Step1).trans root169Scan2
theorem root169Scan0 : scan edge (search edge 3) 4 [1021] 1021 root169Tail0 winner = winner :=
  (root169Step0).trans root169Scan1
set_option maxRecDepth 4096 in
theorem root169Branch : search edge 4 [1021] root169Candidates 1021 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1021] 1021 root169Tail0 winner = winner
  exact root169Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
