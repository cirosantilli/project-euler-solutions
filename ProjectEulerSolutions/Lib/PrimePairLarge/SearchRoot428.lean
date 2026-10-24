import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root428Candidates : List Nat := Data.row3001
set_option maxRecDepth 8192 in
theorem root428Candidates_checked : candidates edge 5 0 3001 tail429 winner.bound = root428Candidates := by
  apply candidates_eq_row edge 5 0 3001 tail429 root428Candidates winner.bound tail429_ordered
  · decide
  · decide
  · funext q
    rfl
def root428Tail44 : List Nat := []
def root428Tail43 : List Nat := 22741 :: root428Tail44
def root428Tail42 : List Nat := 20323 :: root428Tail43
def root428Tail41 : List Nat := 20071 :: root428Tail42
def root428Tail40 : List Nat := 19777 :: root428Tail41
def root428Tail39 : List Nat := 19447 :: root428Tail40
def root428Tail38 : List Nat := 19381 :: root428Tail39
def root428Tail37 : List Nat := 19069 :: root428Tail38
def root428Tail36 : List Nat := 18367 :: root428Tail37
def root428Tail35 : List Nat := 17047 :: root428Tail36
def root428Tail34 : List Nat := 16981 :: root428Tail35
def root428Tail33 : List Nat := 15289 :: root428Tail34
def root428Tail32 : List Nat := 14341 :: root428Tail33
def root428Tail31 : List Nat := 12517 :: root428Tail32
def root428Tail30 : List Nat := 12301 :: root428Tail31
def root428Tail29 : List Nat := 11941 :: root428Tail30
def root428Tail28 : List Nat := 11173 :: root428Tail29
def root428Tail27 : List Nat := 11059 :: root428Tail28
def root428Tail26 : List Nat := 10753 :: root428Tail27
def root428Tail25 : List Nat := 10687 :: root428Tail26
def root428Tail24 : List Nat := 10477 :: root428Tail25
def root428Tail23 : List Nat := 10267 :: root428Tail24
def root428Tail22 : List Nat := 9931 :: root428Tail23
def root428Tail21 : List Nat := 9697 :: root428Tail22
def root428Tail20 : List Nat := 9007 :: root428Tail21
def root428Tail19 : List Nat := 8647 :: root428Tail20
def root428Tail18 : List Nat := 7789 :: root428Tail19
def root428Tail17 : List Nat := 7621 :: root428Tail18
def root428Tail16 : List Nat := 7573 :: root428Tail17
def root428Tail15 : List Nat := 7069 :: root428Tail16
def root428Tail14 : List Nat := 6733 :: root428Tail15
def root428Tail13 : List Nat := 6607 :: root428Tail14
def root428Tail12 : List Nat := 6121 :: root428Tail13
def root428Tail11 : List Nat := 5749 :: root428Tail12
def root428Tail10 : List Nat := 4969 :: root428Tail11
def root428Tail9 : List Nat := 4951 :: root428Tail10
def root428Tail8 : List Nat := 4909 :: root428Tail9
def root428Tail7 : List Nat := 4591 :: root428Tail8
def root428Tail6 : List Nat := 4219 :: root428Tail7
def root428Tail5 : List Nat := 4021 :: root428Tail6
def root428Tail4 : List Nat := 3967 :: root428Tail5
def root428Tail3 : List Nat := 3727 :: root428Tail4
def root428Tail2 : List Nat := 3499 :: root428Tail3
def root428Tail1 : List Nat := 3391 :: root428Tail2
def root428Tail0 : List Nat := 3163 :: root428Tail1
def root428Candidates0 : List Nat := [3967, 6733]
set_option maxRecDepth 8192 in
theorem root428Candidates0_checked : candidates edge 4 3001 3163 root428Tail1 winner.bound = root428Candidates0 := by
  unfold candidates
  have row : edge 3163 = (SortedNatTree.contains FastRows.row3163) := by
    change Data.row3163.contains = (SortedNatTree.contains FastRows.row3163)
    exact FastRows.row3163_checked.symm
  rw [row]
  decide
theorem root428Step0 : scan edge (search edge 3) 4 [3001] 3001 root428Tail0 winner =
    scan edge (search edge 3) 4 [3001] 3001 root428Tail1 winner := by
  rw [root428Tail0, scan, if_neg (by decide), root428Candidates0_checked]
  change scan edge (search edge 3) 4 [3001] 3001 root428Tail1
    (if 3 ≤ root428Candidates0.length then search edge 3 [3001, 3163] root428Candidates0 6164 winner else winner) = _
  rw [if_neg (by decide)]
def root428Candidates1 : List Nat := [6607, 6733, 10267]
set_option maxRecDepth 8192 in
theorem root428Candidates1_checked : candidates edge 4 3001 3391 root428Tail2 winner.bound = root428Candidates1 := by
  unfold candidates
  have row : edge 3391 = (SortedNatTree.contains FastRows.row3391) := by
    change Data.row3391.contains = (SortedNatTree.contains FastRows.row3391)
    exact FastRows.row3391_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root428Child1 : search edge 3 [3001, 3391] root428Candidates1 6392 winner = winner := by decide
theorem root428Step1 : scan edge (search edge 3) 4 [3001] 3001 root428Tail1 winner =
    scan edge (search edge 3) 4 [3001] 3001 root428Tail2 winner := by
  rw [root428Tail1, scan, if_neg (by decide), root428Candidates1_checked]
  change scan edge (search edge 3) 4 [3001] 3001 root428Tail2
    (if 3 ≤ root428Candidates1.length then search edge 3 [3001, 3391] root428Candidates1 6392 winner else winner) = _
  rw [if_pos (by decide), root428Child1]
def root428Candidates2 : List Nat := [9931]
set_option maxRecDepth 8192 in
theorem root428Candidates2_checked : candidates edge 4 3001 3499 root428Tail3 winner.bound = root428Candidates2 := by
  unfold candidates
  have row : edge 3499 = (SortedNatTree.contains FastRows.row3499) := by
    change Data.row3499.contains = (SortedNatTree.contains FastRows.row3499)
    exact FastRows.row3499_checked.symm
  rw [row]
  decide
theorem root428Step2 : scan edge (search edge 3) 4 [3001] 3001 root428Tail2 winner =
    scan edge (search edge 3) 4 [3001] 3001 root428Tail3 winner := by
  rw [root428Tail2, scan, if_neg (by decide), root428Candidates2_checked]
  change scan edge (search edge 3) 4 [3001] 3001 root428Tail3
    (if 3 ≤ root428Candidates2.length then search edge 3 [3001, 3499] root428Candidates2 6500 winner else winner) = _
  rw [if_neg (by decide)]
def root428Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root428Candidates3_checked : candidates edge 4 3001 3727 root428Tail4 winner.bound = root428Candidates3 := by
  unfold candidates
  have row : edge 3727 = (SortedNatTree.contains FastRows.row3727) := by
    change Data.row3727.contains = (SortedNatTree.contains FastRows.row3727)
    exact FastRows.row3727_checked.symm
  rw [row]
  decide
theorem root428Step3 : scan edge (search edge 3) 4 [3001] 3001 root428Tail3 winner =
    scan edge (search edge 3) 4 [3001] 3001 root428Tail4 winner := by
  rw [root428Tail3, scan, if_neg (by decide), root428Candidates3_checked]
  change scan edge (search edge 3) 4 [3001] 3001 root428Tail4
    (if 3 ≤ root428Candidates3.length then search edge 3 [3001, 3727] root428Candidates3 6728 winner else winner) = _
  rw [if_neg (by decide)]
def root428Candidates4 : List Nat := [11173, 12517]
set_option maxRecDepth 8192 in
theorem root428Candidates4_checked : candidates edge 4 3001 3967 root428Tail5 winner.bound = root428Candidates4 := by
  unfold candidates
  have row : edge 3967 = (SortedNatTree.contains FastRows.row3967) := by
    change Data.row3967.contains = (SortedNatTree.contains FastRows.row3967)
    exact FastRows.row3967_checked.symm
  rw [row]
  decide
theorem root428Step4 : scan edge (search edge 3) 4 [3001] 3001 root428Tail4 winner =
    scan edge (search edge 3) 4 [3001] 3001 root428Tail5 winner := by
  rw [root428Tail4, scan, if_neg (by decide), root428Candidates4_checked]
  change scan edge (search edge 3) 4 [3001] 3001 root428Tail5
    (if 3 ≤ root428Candidates4.length then search edge 3 [3001, 3967] root428Candidates4 6968 winner else winner) = _
  rw [if_neg (by decide)]
def root428Candidates5 : List Nat := [6733]
set_option maxRecDepth 8192 in
theorem root428Candidates5_checked : candidates edge 4 3001 4021 root428Tail6 winner.bound = root428Candidates5 := by
  unfold candidates
  have row : edge 4021 = (SortedNatTree.contains FastRows.row4021) := by
    change Data.row4021.contains = (SortedNatTree.contains FastRows.row4021)
    exact FastRows.row4021_checked.symm
  rw [row]
  decide
theorem root428Step5 : scan edge (search edge 3) 4 [3001] 3001 root428Tail5 winner =
    scan edge (search edge 3) 4 [3001] 3001 root428Tail6 winner := by
  rw [root428Tail5, scan, if_neg (by decide), root428Candidates5_checked]
  change scan edge (search edge 3) 4 [3001] 3001 root428Tail6
    (if 3 ≤ root428Candidates5.length then search edge 3 [3001, 4021] root428Candidates5 7022 winner else winner) = _
  rw [if_neg (by decide)]
def root428Candidates6 : List Nat := [5749]
set_option maxRecDepth 8192 in
theorem root428Candidates6_checked : candidates edge 4 3001 4219 root428Tail7 winner.bound = root428Candidates6 := by
  unfold candidates
  have row : edge 4219 = (SortedNatTree.contains FastRows.row4219) := by
    change Data.row4219.contains = (SortedNatTree.contains FastRows.row4219)
    exact FastRows.row4219_checked.symm
  rw [row]
  decide
theorem root428Step6 : scan edge (search edge 3) 4 [3001] 3001 root428Tail6 winner =
    scan edge (search edge 3) 4 [3001] 3001 root428Tail7 winner := by
  rw [root428Tail6, scan, if_neg (by decide), root428Candidates6_checked]
  change scan edge (search edge 3) 4 [3001] 3001 root428Tail7
    (if 3 ≤ root428Candidates6.length then search edge 3 [3001, 4219] root428Candidates6 7220 winner else winner) = _
  rw [if_neg (by decide)]
def root428Candidates7 : List Nat := [6121, 7621, 9007, 9931, 18367]
set_option maxRecDepth 8192 in
theorem root428Candidates7_checked : candidates edge 4 3001 4591 root428Tail8 winner.bound = root428Candidates7 := by
  unfold candidates
  have row : edge 4591 = (SortedNatTree.contains FastRows.row4591) := by
    change Data.row4591.contains = (SortedNatTree.contains FastRows.row4591)
    exact FastRows.row4591_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root428Child7 : search edge 3 [3001, 4591] root428Candidates7 7592 winner = winner := by decide
theorem root428Step7 : scan edge (search edge 3) 4 [3001] 3001 root428Tail7 winner =
    scan edge (search edge 3) 4 [3001] 3001 root428Tail8 winner := by
  rw [root428Tail7, scan, if_neg (by decide), root428Candidates7_checked]
  change scan edge (search edge 3) 4 [3001] 3001 root428Tail8
    (if 3 ≤ root428Candidates7.length then search edge 3 [3001, 4591] root428Candidates7 7592 winner else winner) = _
  rw [if_pos (by decide), root428Child7]
def root428Candidates8 : List Nat := [10477]
set_option maxRecDepth 8192 in
theorem root428Candidates8_checked : candidates edge 4 3001 4909 root428Tail9 winner.bound = root428Candidates8 := by
  unfold candidates
  have row : edge 4909 = (SortedNatTree.contains FastRows.row4909) := by
    change Data.row4909.contains = (SortedNatTree.contains FastRows.row4909)
    exact FastRows.row4909_checked.symm
  rw [row]
  decide
theorem root428Step8 : scan edge (search edge 3) 4 [3001] 3001 root428Tail8 winner =
    scan edge (search edge 3) 4 [3001] 3001 root428Tail9 winner := by
  rw [root428Tail8, scan, if_neg (by decide), root428Candidates8_checked]
  change scan edge (search edge 3) 4 [3001] 3001 root428Tail9
    (if 3 ≤ root428Candidates8.length then search edge 3 [3001, 4909] root428Candidates8 7910 winner else winner) = _
  rw [if_neg (by decide)]
def root428Candidates9 : List Nat := [9697]
set_option maxRecDepth 8192 in
theorem root428Candidates9_checked : candidates edge 4 3001 4951 root428Tail10 winner.bound = root428Candidates9 := by
  unfold candidates
  have row : edge 4951 = (SortedNatTree.contains FastRows.row4951) := by
    change Data.row4951.contains = (SortedNatTree.contains FastRows.row4951)
    exact FastRows.row4951_checked.symm
  rw [row]
  decide
theorem root428Step9 : scan edge (search edge 3) 4 [3001] 3001 root428Tail9 winner =
    scan edge (search edge 3) 4 [3001] 3001 root428Tail10 winner := by
  rw [root428Tail9, scan, if_neg (by decide), root428Candidates9_checked]
  change scan edge (search edge 3) 4 [3001] 3001 root428Tail10
    (if 3 ≤ root428Candidates9.length then search edge 3 [3001, 4951] root428Candidates9 7952 winner else winner) = _
  rw [if_neg (by decide)]
def root428Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root428Candidates10_checked : candidates edge 4 3001 4969 root428Tail11 winner.bound = root428Candidates10 := by
  unfold candidates
  have row : edge 4969 = (SortedNatTree.contains FastRows.row4969) := by
    change Data.row4969.contains = (SortedNatTree.contains FastRows.row4969)
    exact FastRows.row4969_checked.symm
  rw [row]
  decide
theorem root428Step10 : scan edge (search edge 3) 4 [3001] 3001 root428Tail10 winner =
    scan edge (search edge 3) 4 [3001] 3001 root428Tail11 winner := by
  rw [root428Tail10, scan, if_neg (by decide), root428Candidates10_checked]
  change scan edge (search edge 3) 4 [3001] 3001 root428Tail11
    (if 3 ≤ root428Candidates10.length then search edge 3 [3001, 4969] root428Candidates10 7970 winner else winner) = _
  rw [if_neg (by decide)]
def root428Candidates11 : List Nat := []
set_option maxRecDepth 8192 in
theorem root428Candidates11_checked : candidates edge 4 3001 5749 root428Tail12 winner.bound = root428Candidates11 := by
  unfold candidates
  have row : edge 5749 = (SortedNatTree.contains FastRows.row5749) := by
    change Data.row5749.contains = (SortedNatTree.contains FastRows.row5749)
    exact FastRows.row5749_checked.symm
  rw [row]
  decide
theorem root428Step11 : scan edge (search edge 3) 4 [3001] 3001 root428Tail11 winner =
    scan edge (search edge 3) 4 [3001] 3001 root428Tail12 winner := by
  rw [root428Tail11, scan, if_neg (by decide), root428Candidates11_checked]
  change scan edge (search edge 3) 4 [3001] 3001 root428Tail12
    (if 3 ≤ root428Candidates11.length then search edge 3 [3001, 5749] root428Candidates11 8750 winner else winner) = _
  rw [if_neg (by decide)]
theorem root428Scan12 : scan edge (search edge 3) 4 [3001] 3001 root428Tail12 winner = winner := by rw [root428Tail12, scan, if_pos (by decide)]
theorem root428Scan11 : scan edge (search edge 3) 4 [3001] 3001 root428Tail11 winner = winner :=
  (root428Step11).trans root428Scan12
theorem root428Scan10 : scan edge (search edge 3) 4 [3001] 3001 root428Tail10 winner = winner :=
  (root428Step10).trans root428Scan11
theorem root428Scan9 : scan edge (search edge 3) 4 [3001] 3001 root428Tail9 winner = winner :=
  (root428Step9).trans root428Scan10
theorem root428Scan8 : scan edge (search edge 3) 4 [3001] 3001 root428Tail8 winner = winner :=
  (root428Step8).trans root428Scan9
theorem root428Scan7 : scan edge (search edge 3) 4 [3001] 3001 root428Tail7 winner = winner :=
  (root428Step7).trans root428Scan8
theorem root428Scan6 : scan edge (search edge 3) 4 [3001] 3001 root428Tail6 winner = winner :=
  (root428Step6).trans root428Scan7
theorem root428Scan5 : scan edge (search edge 3) 4 [3001] 3001 root428Tail5 winner = winner :=
  (root428Step5).trans root428Scan6
theorem root428Scan4 : scan edge (search edge 3) 4 [3001] 3001 root428Tail4 winner = winner :=
  (root428Step4).trans root428Scan5
theorem root428Scan3 : scan edge (search edge 3) 4 [3001] 3001 root428Tail3 winner = winner :=
  (root428Step3).trans root428Scan4
theorem root428Scan2 : scan edge (search edge 3) 4 [3001] 3001 root428Tail2 winner = winner :=
  (root428Step2).trans root428Scan3
theorem root428Scan1 : scan edge (search edge 3) 4 [3001] 3001 root428Tail1 winner = winner :=
  (root428Step1).trans root428Scan2
theorem root428Scan0 : scan edge (search edge 3) 4 [3001] 3001 root428Tail0 winner = winner :=
  (root428Step0).trans root428Scan1
set_option maxRecDepth 4096 in
theorem root428Branch : search edge 4 [3001] root428Candidates 3001 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3001] 3001 root428Tail0 winner = winner
  exact root428Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
