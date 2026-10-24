import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root440Candidates : List Nat := Data.row3109
set_option maxRecDepth 8192 in
theorem root440Candidates_checked : candidates edge 5 0 3109 tail441 winner.bound = root440Candidates := by
  apply candidates_eq_row edge 5 0 3109 tail441 root440Candidates winner.bound tail441_ordered
  · decide
  · decide
  · funext q
    rfl
def root440Tail51 : List Nat := []
def root440Tail50 : List Nat := 22093 :: root440Tail51
def root440Tail49 : List Nat := 21997 :: root440Tail50
def root440Tail48 : List Nat := 21937 :: root440Tail49
def root440Tail47 : List Nat := 21277 :: root440Tail48
def root440Tail46 : List Nat := 21001 :: root440Tail47
def root440Tail45 : List Nat := 18217 :: root440Tail46
def root440Tail44 : List Nat := 17863 :: root440Tail45
def root440Tail43 : List Nat := 17713 :: root440Tail44
def root440Tail42 : List Nat := 17707 :: root440Tail43
def root440Tail41 : List Nat := 17137 :: root440Tail42
def root440Tail40 : List Nat := 16993 :: root440Tail41
def root440Tail39 : List Nat := 15739 :: root440Tail40
def root440Tail38 : List Nat := 15511 :: root440Tail39
def root440Tail37 : List Nat := 15271 :: root440Tail38
def root440Tail36 : List Nat := 14563 :: root440Tail37
def root440Tail35 : List Nat := 14251 :: root440Tail36
def root440Tail34 : List Nat := 14083 :: root440Tail35
def root440Tail33 : List Nat := 13597 :: root440Tail34
def root440Tail32 : List Nat := 12391 :: root440Tail33
def root440Tail31 : List Nat := 12373 :: root440Tail32
def root440Tail30 : List Nat := 12211 :: root440Tail31
def root440Tail29 : List Nat := 12109 :: root440Tail30
def root440Tail28 : List Nat := 11353 :: root440Tail29
def root440Tail27 : List Nat := 10903 :: root440Tail28
def root440Tail26 : List Nat := 10459 :: root440Tail27
def root440Tail25 : List Nat := 9949 :: root440Tail26
def root440Tail24 : List Nat := 9883 :: root440Tail25
def root440Tail23 : List Nat := 9781 :: root440Tail24
def root440Tail22 : List Nat := 9697 :: root440Tail23
def root440Tail21 : List Nat := 9157 :: root440Tail22
def root440Tail20 : List Nat := 8941 :: root440Tail21
def root440Tail19 : List Nat := 8677 :: root440Tail20
def root440Tail18 : List Nat := 7993 :: root440Tail19
def root440Tail17 : List Nat := 7603 :: root440Tail18
def root440Tail16 : List Nat := 7177 :: root440Tail17
def root440Tail15 : List Nat := 7069 :: root440Tail16
def root440Tail14 : List Nat := 6997 :: root440Tail15
def root440Tail13 : List Nat := 6883 :: root440Tail14
def root440Tail12 : List Nat := 6733 :: root440Tail13
def root440Tail11 : List Nat := 6607 :: root440Tail12
def root440Tail10 : List Nat := 6073 :: root440Tail11
def root440Tail9 : List Nat := 5791 :: root440Tail10
def root440Tail8 : List Nat := 4801 :: root440Tail9
def root440Tail7 : List Nat := 4759 :: root440Tail8
def root440Tail6 : List Nat := 4447 :: root440Tail7
def root440Tail5 : List Nat := 4339 :: root440Tail6
def root440Tail4 : List Nat := 4099 :: root440Tail5
def root440Tail3 : List Nat := 4057 :: root440Tail4
def root440Tail2 : List Nat := 4027 :: root440Tail3
def root440Tail1 : List Nat := 3877 :: root440Tail2
def root440Tail0 : List Nat := 3541 :: root440Tail1
def root440Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root440Candidates0_checked : candidates edge 4 3109 3541 root440Tail1 winner.bound = root440Candidates0 := by
  unfold candidates
  have row : edge 3541 = (SortedNatTree.contains FastRows.row3541) := by
    change Data.row3541.contains = (SortedNatTree.contains FastRows.row3541)
    exact FastRows.row3541_checked.symm
  rw [row]
  decide
theorem root440Step0 : scan edge (search edge 3) 4 [3109] 3109 root440Tail0 winner =
    scan edge (search edge 3) 4 [3109] 3109 root440Tail1 winner := by
  rw [root440Tail0, scan, if_neg (by decide), root440Candidates0_checked]
  change scan edge (search edge 3) 4 [3109] 3109 root440Tail1
    (if 3 ≤ root440Candidates0.length then search edge 3 [3109, 3541] root440Candidates0 6650 winner else winner) = _
  rw [if_neg (by decide)]
def root440Candidates1 : List Nat := [6607]
set_option maxRecDepth 8192 in
theorem root440Candidates1_checked : candidates edge 4 3109 3877 root440Tail2 winner.bound = root440Candidates1 := by
  unfold candidates
  have row : edge 3877 = (SortedNatTree.contains FastRows.row3877) := by
    change Data.row3877.contains = (SortedNatTree.contains FastRows.row3877)
    exact FastRows.row3877_checked.symm
  rw [row]
  decide
theorem root440Step1 : scan edge (search edge 3) 4 [3109] 3109 root440Tail1 winner =
    scan edge (search edge 3) 4 [3109] 3109 root440Tail2 winner := by
  rw [root440Tail1, scan, if_neg (by decide), root440Candidates1_checked]
  change scan edge (search edge 3) 4 [3109] 3109 root440Tail2
    (if 3 ≤ root440Candidates1.length then search edge 3 [3109, 3877] root440Candidates1 6986 winner else winner) = _
  rw [if_neg (by decide)]
def root440Candidates2 : List Nat := [8677, 9157, 12109, 14083]
set_option maxRecDepth 8192 in
theorem root440Candidates2_checked : candidates edge 4 3109 4027 root440Tail3 winner.bound = root440Candidates2 := by
  unfold candidates
  have row : edge 4027 = (SortedNatTree.contains FastRows.row4027) := by
    change Data.row4027.contains = (SortedNatTree.contains FastRows.row4027)
    exact FastRows.row4027_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root440Child2 : search edge 3 [3109, 4027] root440Candidates2 7136 winner = winner := by decide
theorem root440Step2 : scan edge (search edge 3) 4 [3109] 3109 root440Tail2 winner =
    scan edge (search edge 3) 4 [3109] 3109 root440Tail3 winner := by
  rw [root440Tail2, scan, if_neg (by decide), root440Candidates2_checked]
  change scan edge (search edge 3) 4 [3109] 3109 root440Tail3
    (if 3 ≤ root440Candidates2.length then search edge 3 [3109, 4027] root440Candidates2 7136 winner else winner) = _
  rw [if_pos (by decide), root440Child2]
def root440Candidates3 : List Nat := [13597]
set_option maxRecDepth 8192 in
theorem root440Candidates3_checked : candidates edge 4 3109 4057 root440Tail4 winner.bound = root440Candidates3 := by
  unfold candidates
  have row : edge 4057 = (SortedNatTree.contains FastRows.row4057) := by
    change Data.row4057.contains = (SortedNatTree.contains FastRows.row4057)
    exact FastRows.row4057_checked.symm
  rw [row]
  decide
theorem root440Step3 : scan edge (search edge 3) 4 [3109] 3109 root440Tail3 winner =
    scan edge (search edge 3) 4 [3109] 3109 root440Tail4 winner := by
  rw [root440Tail3, scan, if_neg (by decide), root440Candidates3_checked]
  change scan edge (search edge 3) 4 [3109] 3109 root440Tail4
    (if 3 ≤ root440Candidates3.length then search edge 3 [3109, 4057] root440Candidates3 7166 winner else winner) = _
  rw [if_neg (by decide)]
def root440Candidates4 : List Nat := [13597, 16993]
set_option maxRecDepth 8192 in
theorem root440Candidates4_checked : candidates edge 4 3109 4099 root440Tail5 winner.bound = root440Candidates4 := by
  unfold candidates
  have row : edge 4099 = (SortedNatTree.contains FastRows.row4099) := by
    change Data.row4099.contains = (SortedNatTree.contains FastRows.row4099)
    exact FastRows.row4099_checked.symm
  rw [row]
  decide
theorem root440Step4 : scan edge (search edge 3) 4 [3109] 3109 root440Tail4 winner =
    scan edge (search edge 3) 4 [3109] 3109 root440Tail5 winner := by
  rw [root440Tail4, scan, if_neg (by decide), root440Candidates4_checked]
  change scan edge (search edge 3) 4 [3109] 3109 root440Tail5
    (if 3 ≤ root440Candidates4.length then search edge 3 [3109, 4099] root440Candidates4 7208 winner else winner) = _
  rw [if_neg (by decide)]
def root440Candidates5 : List Nat := [7603, 10903, 14083, 14563]
set_option maxRecDepth 8192 in
theorem root440Candidates5_checked : candidates edge 4 3109 4339 root440Tail6 winner.bound = root440Candidates5 := by
  unfold candidates
  have row : edge 4339 = (SortedNatTree.contains FastRows.row4339) := by
    change Data.row4339.contains = (SortedNatTree.contains FastRows.row4339)
    exact FastRows.row4339_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root440Child5 : search edge 3 [3109, 4339] root440Candidates5 7448 winner = winner := by decide
theorem root440Step5 : scan edge (search edge 3) 4 [3109] 3109 root440Tail5 winner =
    scan edge (search edge 3) 4 [3109] 3109 root440Tail6 winner := by
  rw [root440Tail5, scan, if_neg (by decide), root440Candidates5_checked]
  change scan edge (search edge 3) 4 [3109] 3109 root440Tail6
    (if 3 ≤ root440Candidates5.length then search edge 3 [3109, 4339] root440Candidates5 7448 winner else winner) = _
  rw [if_pos (by decide), root440Child5]
def root440Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root440Candidates6_checked : candidates edge 4 3109 4447 root440Tail7 winner.bound = root440Candidates6 := by
  unfold candidates
  have row : edge 4447 = (SortedNatTree.contains FastRows.row4447) := by
    change Data.row4447.contains = (SortedNatTree.contains FastRows.row4447)
    exact FastRows.row4447_checked.symm
  rw [row]
  decide
theorem root440Step6 : scan edge (search edge 3) 4 [3109] 3109 root440Tail6 winner =
    scan edge (search edge 3) 4 [3109] 3109 root440Tail7 winner := by
  rw [root440Tail6, scan, if_neg (by decide), root440Candidates6_checked]
  change scan edge (search edge 3) 4 [3109] 3109 root440Tail7
    (if 3 ≤ root440Candidates6.length then search edge 3 [3109, 4447] root440Candidates6 7556 winner else winner) = _
  rw [if_neg (by decide)]
def root440Candidates7 : List Nat := [13597]
set_option maxRecDepth 8192 in
theorem root440Candidates7_checked : candidates edge 4 3109 4759 root440Tail8 winner.bound = root440Candidates7 := by
  unfold candidates
  have row : edge 4759 = (SortedNatTree.contains FastRows.row4759) := by
    change Data.row4759.contains = (SortedNatTree.contains FastRows.row4759)
    exact FastRows.row4759_checked.symm
  rw [row]
  decide
theorem root440Step7 : scan edge (search edge 3) 4 [3109] 3109 root440Tail7 winner =
    scan edge (search edge 3) 4 [3109] 3109 root440Tail8 winner := by
  rw [root440Tail7, scan, if_neg (by decide), root440Candidates7_checked]
  change scan edge (search edge 3) 4 [3109] 3109 root440Tail8
    (if 3 ≤ root440Candidates7.length then search edge 3 [3109, 4759] root440Candidates7 7868 winner else winner) = _
  rw [if_neg (by decide)]
def root440Candidates8 : List Nat := [6733, 7993, 12109, 12211]
set_option maxRecDepth 8192 in
theorem root440Candidates8_checked : candidates edge 4 3109 4801 root440Tail9 winner.bound = root440Candidates8 := by
  unfold candidates
  have row : edge 4801 = (SortedNatTree.contains FastRows.row4801) := by
    change Data.row4801.contains = (SortedNatTree.contains FastRows.row4801)
    exact FastRows.row4801_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root440Child8 : search edge 3 [3109, 4801] root440Candidates8 7910 winner = winner := by decide
theorem root440Step8 : scan edge (search edge 3) 4 [3109] 3109 root440Tail8 winner =
    scan edge (search edge 3) 4 [3109] 3109 root440Tail9 winner := by
  rw [root440Tail8, scan, if_neg (by decide), root440Candidates8_checked]
  change scan edge (search edge 3) 4 [3109] 3109 root440Tail9
    (if 3 ≤ root440Candidates8.length then search edge 3 [3109, 4801] root440Candidates8 7910 winner else winner) = _
  rw [if_pos (by decide), root440Child8]
theorem root440Scan9 : scan edge (search edge 3) 4 [3109] 3109 root440Tail9 winner = winner := by rw [root440Tail9, scan, if_pos (by decide)]
theorem root440Scan8 : scan edge (search edge 3) 4 [3109] 3109 root440Tail8 winner = winner :=
  (root440Step8).trans root440Scan9
theorem root440Scan7 : scan edge (search edge 3) 4 [3109] 3109 root440Tail7 winner = winner :=
  (root440Step7).trans root440Scan8
theorem root440Scan6 : scan edge (search edge 3) 4 [3109] 3109 root440Tail6 winner = winner :=
  (root440Step6).trans root440Scan7
theorem root440Scan5 : scan edge (search edge 3) 4 [3109] 3109 root440Tail5 winner = winner :=
  (root440Step5).trans root440Scan6
theorem root440Scan4 : scan edge (search edge 3) 4 [3109] 3109 root440Tail4 winner = winner :=
  (root440Step4).trans root440Scan5
theorem root440Scan3 : scan edge (search edge 3) 4 [3109] 3109 root440Tail3 winner = winner :=
  (root440Step3).trans root440Scan4
theorem root440Scan2 : scan edge (search edge 3) 4 [3109] 3109 root440Tail2 winner = winner :=
  (root440Step2).trans root440Scan3
theorem root440Scan1 : scan edge (search edge 3) 4 [3109] 3109 root440Tail1 winner = winner :=
  (root440Step1).trans root440Scan2
theorem root440Scan0 : scan edge (search edge 3) 4 [3109] 3109 root440Tail0 winner = winner :=
  (root440Step0).trans root440Scan1
set_option maxRecDepth 4096 in
theorem root440Branch : search edge 4 [3109] root440Candidates 3109 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3109] 3109 root440Tail0 winner = winner
  exact root440Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
