import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root349Candidates : List Nat := Data.row2377
set_option maxRecDepth 8192 in
theorem root349Candidates_checked : candidates edge 5 0 2377 tail350 winner.bound = root349Candidates := by
  apply candidates_eq_row edge 5 0 2377 tail350 root349Candidates winner.bound tail350_ordered
  · decide
  · decide
  · funext q
    rfl
def root349Tail33 : List Nat := []
def root349Tail32 : List Nat := 22171 :: root349Tail33
def root349Tail31 : List Nat := 21943 :: root349Tail32
def root349Tail30 : List Nat := 21061 :: root349Tail31
def root349Tail29 : List Nat := 20521 :: root349Tail30
def root349Tail28 : List Nat := 19477 :: root349Tail29
def root349Tail27 : List Nat := 18301 :: root349Tail28
def root349Tail26 : List Nat := 18181 :: root349Tail27
def root349Tail25 : List Nat := 17509 :: root349Tail26
def root349Tail24 : List Nat := 16693 :: root349Tail25
def root349Tail23 : List Nat := 16249 :: root349Tail24
def root349Tail22 : List Nat := 14437 :: root349Tail23
def root349Tail21 : List Nat := 13963 :: root349Tail22
def root349Tail20 : List Nat := 13627 :: root349Tail21
def root349Tail19 : List Nat := 12637 :: root349Tail20
def root349Tail18 : List Nat := 10909 :: root349Tail19
def root349Tail17 : List Nat := 10657 :: root349Tail18
def root349Tail16 : List Nat := 10243 :: root349Tail17
def root349Tail15 : List Nat := 9949 :: root349Tail16
def root349Tail14 : List Nat := 9733 :: root349Tail15
def root349Tail13 : List Nat := 9649 :: root349Tail14
def root349Tail12 : List Nat := 8719 :: root349Tail13
def root349Tail11 : List Nat := 7603 :: root349Tail12
def root349Tail10 : List Nat := 6553 :: root349Tail11
def root349Tail9 : List Nat := 6199 :: root349Tail10
def root349Tail8 : List Nat := 6073 :: root349Tail9
def root349Tail7 : List Nat := 5743 :: root349Tail8
def root349Tail6 : List Nat := 4327 :: root349Tail7
def root349Tail5 : List Nat := 4159 :: root349Tail6
def root349Tail4 : List Nat := 3433 :: root349Tail5
def root349Tail3 : List Nat := 3319 :: root349Tail4
def root349Tail2 : List Nat := 3067 :: root349Tail3
def root349Tail1 : List Nat := 2797 :: root349Tail2
def root349Tail0 : List Nat := 2557 :: root349Tail1
def root349Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root349Candidates0_checked : candidates edge 4 2377 2557 root349Tail1 winner.bound = root349Candidates0 := by
  unfold candidates
  have row : edge 2557 = (SortedNatTree.contains FastRows.row2557) := by
    change Data.row2557.contains = (SortedNatTree.contains FastRows.row2557)
    exact FastRows.row2557_checked.symm
  rw [row]
  decide
theorem root349Step0 : scan edge (search edge 3) 4 [2377] 2377 root349Tail0 winner =
    scan edge (search edge 3) 4 [2377] 2377 root349Tail1 winner := by
  rw [root349Tail0, scan, if_neg (by decide), root349Candidates0_checked]
  change scan edge (search edge 3) 4 [2377] 2377 root349Tail1
    (if 3 ≤ root349Candidates0.length then search edge 3 [2377, 2557] root349Candidates0 4934 winner else winner) = _
  rw [if_neg (by decide)]
def root349Candidates1 : List Nat := [13963]
set_option maxRecDepth 8192 in
theorem root349Candidates1_checked : candidates edge 4 2377 2797 root349Tail2 winner.bound = root349Candidates1 := by
  unfold candidates
  have row : edge 2797 = (SortedNatTree.contains FastRows.row2797) := by
    change Data.row2797.contains = (SortedNatTree.contains FastRows.row2797)
    exact FastRows.row2797_checked.symm
  rw [row]
  decide
theorem root349Step1 : scan edge (search edge 3) 4 [2377] 2377 root349Tail1 winner =
    scan edge (search edge 3) 4 [2377] 2377 root349Tail2 winner := by
  rw [root349Tail1, scan, if_neg (by decide), root349Candidates1_checked]
  change scan edge (search edge 3) 4 [2377] 2377 root349Tail2
    (if 3 ≤ root349Candidates1.length then search edge 3 [2377, 2797] root349Candidates1 5174 winner else winner) = _
  rw [if_neg (by decide)]
def root349Candidates2 : List Nat := [10909]
set_option maxRecDepth 8192 in
theorem root349Candidates2_checked : candidates edge 4 2377 3067 root349Tail3 winner.bound = root349Candidates2 := by
  unfold candidates
  have row : edge 3067 = (SortedNatTree.contains FastRows.row3067) := by
    change Data.row3067.contains = (SortedNatTree.contains FastRows.row3067)
    exact FastRows.row3067_checked.symm
  rw [row]
  decide
theorem root349Step2 : scan edge (search edge 3) 4 [2377] 2377 root349Tail2 winner =
    scan edge (search edge 3) 4 [2377] 2377 root349Tail3 winner := by
  rw [root349Tail2, scan, if_neg (by decide), root349Candidates2_checked]
  change scan edge (search edge 3) 4 [2377] 2377 root349Tail3
    (if 3 ≤ root349Candidates2.length then search edge 3 [2377, 3067] root349Candidates2 5444 winner else winner) = _
  rw [if_neg (by decide)]
def root349Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root349Candidates3_checked : candidates edge 4 2377 3319 root349Tail4 winner.bound = root349Candidates3 := by
  unfold candidates
  have row : edge 3319 = (SortedNatTree.contains FastRows.row3319) := by
    change Data.row3319.contains = (SortedNatTree.contains FastRows.row3319)
    exact FastRows.row3319_checked.symm
  rw [row]
  decide
theorem root349Step3 : scan edge (search edge 3) 4 [2377] 2377 root349Tail3 winner =
    scan edge (search edge 3) 4 [2377] 2377 root349Tail4 winner := by
  rw [root349Tail3, scan, if_neg (by decide), root349Candidates3_checked]
  change scan edge (search edge 3) 4 [2377] 2377 root349Tail4
    (if 3 ≤ root349Candidates3.length then search edge 3 [2377, 3319] root349Candidates3 5696 winner else winner) = _
  rw [if_neg (by decide)]
def root349Candidates4 : List Nat := [17509]
set_option maxRecDepth 8192 in
theorem root349Candidates4_checked : candidates edge 4 2377 3433 root349Tail5 winner.bound = root349Candidates4 := by
  unfold candidates
  have row : edge 3433 = (SortedNatTree.contains FastRows.row3433) := by
    change Data.row3433.contains = (SortedNatTree.contains FastRows.row3433)
    exact FastRows.row3433_checked.symm
  rw [row]
  decide
theorem root349Step4 : scan edge (search edge 3) 4 [2377] 2377 root349Tail4 winner =
    scan edge (search edge 3) 4 [2377] 2377 root349Tail5 winner := by
  rw [root349Tail4, scan, if_neg (by decide), root349Candidates4_checked]
  change scan edge (search edge 3) 4 [2377] 2377 root349Tail5
    (if 3 ≤ root349Candidates4.length then search edge 3 [2377, 3433] root349Candidates4 5810 winner else winner) = _
  rw [if_neg (by decide)]
def root349Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root349Candidates5_checked : candidates edge 4 2377 4159 root349Tail6 winner.bound = root349Candidates5 := by
  unfold candidates
  have row : edge 4159 = (SortedNatTree.contains FastRows.row4159) := by
    change Data.row4159.contains = (SortedNatTree.contains FastRows.row4159)
    exact FastRows.row4159_checked.symm
  rw [row]
  decide
theorem root349Step5 : scan edge (search edge 3) 4 [2377] 2377 root349Tail5 winner =
    scan edge (search edge 3) 4 [2377] 2377 root349Tail6 winner := by
  rw [root349Tail5, scan, if_neg (by decide), root349Candidates5_checked]
  change scan edge (search edge 3) 4 [2377] 2377 root349Tail6
    (if 3 ≤ root349Candidates5.length then search edge 3 [2377, 4159] root349Candidates5 6536 winner else winner) = _
  rw [if_neg (by decide)]
def root349Candidates6 : List Nat := [6553]
set_option maxRecDepth 8192 in
theorem root349Candidates6_checked : candidates edge 4 2377 4327 root349Tail7 winner.bound = root349Candidates6 := by
  unfold candidates
  have row : edge 4327 = (SortedNatTree.contains FastRows.row4327) := by
    change Data.row4327.contains = (SortedNatTree.contains FastRows.row4327)
    exact FastRows.row4327_checked.symm
  rw [row]
  decide
theorem root349Step6 : scan edge (search edge 3) 4 [2377] 2377 root349Tail6 winner =
    scan edge (search edge 3) 4 [2377] 2377 root349Tail7 winner := by
  rw [root349Tail6, scan, if_neg (by decide), root349Candidates6_checked]
  change scan edge (search edge 3) 4 [2377] 2377 root349Tail7
    (if 3 ≤ root349Candidates6.length then search edge 3 [2377, 4327] root349Candidates6 6704 winner else winner) = _
  rw [if_neg (by decide)]
def root349Candidates7 : List Nat := [10909]
set_option maxRecDepth 8192 in
theorem root349Candidates7_checked : candidates edge 4 2377 5743 root349Tail8 winner.bound = root349Candidates7 := by
  unfold candidates
  have row : edge 5743 = (SortedNatTree.contains FastRows.row5743) := by
    change Data.row5743.contains = (SortedNatTree.contains FastRows.row5743)
    exact FastRows.row5743_checked.symm
  rw [row]
  decide
theorem root349Step7 : scan edge (search edge 3) 4 [2377] 2377 root349Tail7 winner =
    scan edge (search edge 3) 4 [2377] 2377 root349Tail8 winner := by
  rw [root349Tail7, scan, if_neg (by decide), root349Candidates7_checked]
  change scan edge (search edge 3) 4 [2377] 2377 root349Tail8
    (if 3 ≤ root349Candidates7.length then search edge 3 [2377, 5743] root349Candidates7 8120 winner else winner) = _
  rw [if_neg (by decide)]
theorem root349Scan8 : scan edge (search edge 3) 4 [2377] 2377 root349Tail8 winner = winner := by rw [root349Tail8, scan, if_pos (by decide)]
theorem root349Scan7 : scan edge (search edge 3) 4 [2377] 2377 root349Tail7 winner = winner :=
  (root349Step7).trans root349Scan8
theorem root349Scan6 : scan edge (search edge 3) 4 [2377] 2377 root349Tail6 winner = winner :=
  (root349Step6).trans root349Scan7
theorem root349Scan5 : scan edge (search edge 3) 4 [2377] 2377 root349Tail5 winner = winner :=
  (root349Step5).trans root349Scan6
theorem root349Scan4 : scan edge (search edge 3) 4 [2377] 2377 root349Tail4 winner = winner :=
  (root349Step4).trans root349Scan5
theorem root349Scan3 : scan edge (search edge 3) 4 [2377] 2377 root349Tail3 winner = winner :=
  (root349Step3).trans root349Scan4
theorem root349Scan2 : scan edge (search edge 3) 4 [2377] 2377 root349Tail2 winner = winner :=
  (root349Step2).trans root349Scan3
theorem root349Scan1 : scan edge (search edge 3) 4 [2377] 2377 root349Tail1 winner = winner :=
  (root349Step1).trans root349Scan2
theorem root349Scan0 : scan edge (search edge 3) 4 [2377] 2377 root349Tail0 winner = winner :=
  (root349Step0).trans root349Scan1
set_option maxRecDepth 4096 in
theorem root349Branch : search edge 4 [2377] root349Candidates 2377 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2377] 2377 root349Tail0 winner = winner
  exact root349Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
