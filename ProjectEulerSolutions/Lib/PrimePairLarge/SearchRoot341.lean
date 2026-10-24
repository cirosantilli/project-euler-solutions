import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root341Candidates : List Nat := Data.row2311
set_option maxRecDepth 8192 in
theorem root341Candidates_checked : candidates edge 5 0 2311 tail342 winner.bound = root341Candidates := by
  apply candidates_eq_row edge 5 0 2311 tail342 root341Candidates winner.bound tail342_ordered
  · decide
  · decide
  · funext q
    rfl
def root341Tail42 : List Nat := []
def root341Tail41 : List Nat := 23689 :: root341Tail42
def root341Tail40 : List Nat := 23593 :: root341Tail41
def root341Tail39 : List Nat := 22651 :: root341Tail40
def root341Tail38 : List Nat := 22273 :: root341Tail39
def root341Tail37 : List Nat := 22063 :: root341Tail38
def root341Tail36 : List Nat := 21937 :: root341Tail37
def root341Tail35 : List Nat := 21661 :: root341Tail36
def root341Tail34 : List Nat := 20023 :: root341Tail35
def root341Tail33 : List Nat := 19417 :: root341Tail34
def root341Tail32 : List Nat := 17977 :: root341Tail33
def root341Tail31 : List Nat := 17569 :: root341Tail32
def root341Tail30 : List Nat := 17011 :: root341Tail31
def root341Tail29 : List Nat := 16981 :: root341Tail30
def root341Tail28 : List Nat := 16339 :: root341Tail29
def root341Tail27 : List Nat := 15277 :: root341Tail28
def root341Tail26 : List Nat := 15271 :: root341Tail27
def root341Tail25 : List Nat := 14731 :: root341Tail26
def root341Tail24 : List Nat := 14449 :: root341Tail25
def root341Tail23 : List Nat := 14029 :: root341Tail24
def root341Tail22 : List Nat := 13327 :: root341Tail23
def root341Tail21 : List Nat := 11833 :: root341Tail22
def root341Tail20 : List Nat := 11689 :: root341Tail21
def root341Tail19 : List Nat := 10753 :: root341Tail20
def root341Tail18 : List Nat := 9739 :: root341Tail19
def root341Tail17 : List Nat := 9391 :: root341Tail18
def root341Tail16 : List Nat := 8941 :: root341Tail17
def root341Tail15 : List Nat := 8731 :: root341Tail16
def root341Tail14 : List Nat := 8287 :: root341Tail15
def root341Tail13 : List Nat := 7639 :: root341Tail14
def root341Tail12 : List Nat := 6883 :: root341Tail13
def root341Tail11 : List Nat := 6871 :: root341Tail12
def root341Tail10 : List Nat := 6619 :: root341Tail11
def root341Tail9 : List Nat := 6469 :: root341Tail10
def root341Tail8 : List Nat := 6199 :: root341Tail9
def root341Tail7 : List Nat := 5179 :: root341Tail8
def root341Tail6 : List Nat := 4549 :: root341Tail7
def root341Tail5 : List Nat := 3943 :: root341Tail6
def root341Tail4 : List Nat := 3847 :: root341Tail5
def root341Tail3 : List Nat := 3529 :: root341Tail4
def root341Tail2 : List Nat := 3313 :: root341Tail3
def root341Tail1 : List Nat := 3061 :: root341Tail2
def root341Tail0 : List Nat := 2647 :: root341Tail1
def root341Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root341Candidates0_checked : candidates edge 4 2311 2647 root341Tail1 winner.bound = root341Candidates0 := by
  unfold candidates
  have row : edge 2647 = (SortedNatTree.contains FastRows.row2647) := by
    change Data.row2647.contains = (SortedNatTree.contains FastRows.row2647)
    exact FastRows.row2647_checked.symm
  rw [row]
  decide
theorem root341Step0 : scan edge (search edge 3) 4 [2311] 2311 root341Tail0 winner =
    scan edge (search edge 3) 4 [2311] 2311 root341Tail1 winner := by
  rw [root341Tail0, scan, if_neg (by decide), root341Candidates0_checked]
  change scan edge (search edge 3) 4 [2311] 2311 root341Tail1
    (if 3 ≤ root341Candidates0.length then search edge 3 [2311, 2647] root341Candidates0 4958 winner else winner) = _
  rw [if_neg (by decide)]
def root341Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root341Candidates1_checked : candidates edge 4 2311 3061 root341Tail2 winner.bound = root341Candidates1 := by
  unfold candidates
  have row : edge 3061 = (SortedNatTree.contains FastRows.row3061) := by
    change Data.row3061.contains = (SortedNatTree.contains FastRows.row3061)
    exact FastRows.row3061_checked.symm
  rw [row]
  decide
theorem root341Step1 : scan edge (search edge 3) 4 [2311] 2311 root341Tail1 winner =
    scan edge (search edge 3) 4 [2311] 2311 root341Tail2 winner := by
  rw [root341Tail1, scan, if_neg (by decide), root341Candidates1_checked]
  change scan edge (search edge 3) 4 [2311] 2311 root341Tail2
    (if 3 ≤ root341Candidates1.length then search edge 3 [2311, 3061] root341Candidates1 5372 winner else winner) = _
  rw [if_neg (by decide)]
def root341Candidates2 : List Nat := [6871, 9739, 14449]
set_option maxRecDepth 8192 in
theorem root341Candidates2_checked : candidates edge 4 2311 3313 root341Tail3 winner.bound = root341Candidates2 := by
  unfold candidates
  have row : edge 3313 = (SortedNatTree.contains FastRows.row3313) := by
    change Data.row3313.contains = (SortedNatTree.contains FastRows.row3313)
    exact FastRows.row3313_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root341Child2 : search edge 3 [2311, 3313] root341Candidates2 5624 winner = winner := by decide
theorem root341Step2 : scan edge (search edge 3) 4 [2311] 2311 root341Tail2 winner =
    scan edge (search edge 3) 4 [2311] 2311 root341Tail3 winner := by
  rw [root341Tail2, scan, if_neg (by decide), root341Candidates2_checked]
  change scan edge (search edge 3) 4 [2311] 2311 root341Tail3
    (if 3 ≤ root341Candidates2.length then search edge 3 [2311, 3313] root341Candidates2 5624 winner else winner) = _
  rw [if_pos (by decide), root341Child2]
def root341Candidates3 : List Nat := [3943, 16981]
set_option maxRecDepth 8192 in
theorem root341Candidates3_checked : candidates edge 4 2311 3529 root341Tail4 winner.bound = root341Candidates3 := by
  unfold candidates
  have row : edge 3529 = (SortedNatTree.contains FastRows.row3529) := by
    change Data.row3529.contains = (SortedNatTree.contains FastRows.row3529)
    exact FastRows.row3529_checked.symm
  rw [row]
  decide
theorem root341Step3 : scan edge (search edge 3) 4 [2311] 2311 root341Tail3 winner =
    scan edge (search edge 3) 4 [2311] 2311 root341Tail4 winner := by
  rw [root341Tail3, scan, if_neg (by decide), root341Candidates3_checked]
  change scan edge (search edge 3) 4 [2311] 2311 root341Tail4
    (if 3 ≤ root341Candidates3.length then search edge 3 [2311, 3529] root341Candidates3 5840 winner else winner) = _
  rw [if_neg (by decide)]
def root341Candidates4 : List Nat := [6883, 15277]
set_option maxRecDepth 8192 in
theorem root341Candidates4_checked : candidates edge 4 2311 3847 root341Tail5 winner.bound = root341Candidates4 := by
  unfold candidates
  have row : edge 3847 = (SortedNatTree.contains FastRows.row3847) := by
    change Data.row3847.contains = (SortedNatTree.contains FastRows.row3847)
    exact FastRows.row3847_checked.symm
  rw [row]
  decide
theorem root341Step4 : scan edge (search edge 3) 4 [2311] 2311 root341Tail4 winner =
    scan edge (search edge 3) 4 [2311] 2311 root341Tail5 winner := by
  rw [root341Tail4, scan, if_neg (by decide), root341Candidates4_checked]
  change scan edge (search edge 3) 4 [2311] 2311 root341Tail5
    (if 3 ≤ root341Candidates4.length then search edge 3 [2311, 3847] root341Candidates4 6158 winner else winner) = _
  rw [if_neg (by decide)]
def root341Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root341Candidates5_checked : candidates edge 4 2311 3943 root341Tail6 winner.bound = root341Candidates5 := by
  unfold candidates
  have row : edge 3943 = (SortedNatTree.contains FastRows.row3943) := by
    change Data.row3943.contains = (SortedNatTree.contains FastRows.row3943)
    exact FastRows.row3943_checked.symm
  rw [row]
  decide
theorem root341Step5 : scan edge (search edge 3) 4 [2311] 2311 root341Tail5 winner =
    scan edge (search edge 3) 4 [2311] 2311 root341Tail6 winner := by
  rw [root341Tail5, scan, if_neg (by decide), root341Candidates5_checked]
  change scan edge (search edge 3) 4 [2311] 2311 root341Tail6
    (if 3 ≤ root341Candidates5.length then search edge 3 [2311, 3943] root341Candidates5 6254 winner else winner) = _
  rw [if_neg (by decide)]
def root341Candidates6 : List Nat := [8287, 15277, 16339]
set_option maxRecDepth 8192 in
theorem root341Candidates6_checked : candidates edge 4 2311 4549 root341Tail7 winner.bound = root341Candidates6 := by
  unfold candidates
  have row : edge 4549 = (SortedNatTree.contains FastRows.row4549) := by
    change Data.row4549.contains = (SortedNatTree.contains FastRows.row4549)
    exact FastRows.row4549_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root341Child6 : search edge 3 [2311, 4549] root341Candidates6 6860 winner = winner := by decide
theorem root341Step6 : scan edge (search edge 3) 4 [2311] 2311 root341Tail6 winner =
    scan edge (search edge 3) 4 [2311] 2311 root341Tail7 winner := by
  rw [root341Tail6, scan, if_neg (by decide), root341Candidates6_checked]
  change scan edge (search edge 3) 4 [2311] 2311 root341Tail7
    (if 3 ≤ root341Candidates6.length then search edge 3 [2311, 4549] root341Candidates6 6860 winner else winner) = _
  rw [if_pos (by decide), root341Child6]
def root341Candidates7 : List Nat := [14029]
set_option maxRecDepth 8192 in
theorem root341Candidates7_checked : candidates edge 4 2311 5179 root341Tail8 winner.bound = root341Candidates7 := by
  unfold candidates
  have row : edge 5179 = (SortedNatTree.contains FastRows.row5179) := by
    change Data.row5179.contains = (SortedNatTree.contains FastRows.row5179)
    exact FastRows.row5179_checked.symm
  rw [row]
  decide
theorem root341Step7 : scan edge (search edge 3) 4 [2311] 2311 root341Tail7 winner =
    scan edge (search edge 3) 4 [2311] 2311 root341Tail8 winner := by
  rw [root341Tail7, scan, if_neg (by decide), root341Candidates7_checked]
  change scan edge (search edge 3) 4 [2311] 2311 root341Tail8
    (if 3 ≤ root341Candidates7.length then search edge 3 [2311, 5179] root341Candidates7 7490 winner else winner) = _
  rw [if_neg (by decide)]
theorem root341Scan8 : scan edge (search edge 3) 4 [2311] 2311 root341Tail8 winner = winner := by rw [root341Tail8, scan, if_pos (by decide)]
theorem root341Scan7 : scan edge (search edge 3) 4 [2311] 2311 root341Tail7 winner = winner :=
  (root341Step7).trans root341Scan8
theorem root341Scan6 : scan edge (search edge 3) 4 [2311] 2311 root341Tail6 winner = winner :=
  (root341Step6).trans root341Scan7
theorem root341Scan5 : scan edge (search edge 3) 4 [2311] 2311 root341Tail5 winner = winner :=
  (root341Step5).trans root341Scan6
theorem root341Scan4 : scan edge (search edge 3) 4 [2311] 2311 root341Tail4 winner = winner :=
  (root341Step4).trans root341Scan5
theorem root341Scan3 : scan edge (search edge 3) 4 [2311] 2311 root341Tail3 winner = winner :=
  (root341Step3).trans root341Scan4
theorem root341Scan2 : scan edge (search edge 3) 4 [2311] 2311 root341Tail2 winner = winner :=
  (root341Step2).trans root341Scan3
theorem root341Scan1 : scan edge (search edge 3) 4 [2311] 2311 root341Tail1 winner = winner :=
  (root341Step1).trans root341Scan2
theorem root341Scan0 : scan edge (search edge 3) 4 [2311] 2311 root341Tail0 winner = winner :=
  (root341Step0).trans root341Scan1
set_option maxRecDepth 4096 in
theorem root341Branch : search edge 4 [2311] root341Candidates 2311 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2311] 2311 root341Tail0 winner = winner
  exact root341Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
