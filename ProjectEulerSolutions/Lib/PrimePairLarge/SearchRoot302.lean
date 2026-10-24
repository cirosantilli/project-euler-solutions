import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root302Candidates : List Nat := Data.row2011
set_option maxRecDepth 8192 in
theorem root302Candidates_checked : candidates edge 5 0 2011 tail303 winner.bound = root302Candidates := by
  apply candidates_eq_row edge 5 0 2011 tail303 root302Candidates winner.bound tail303_ordered
  · decide
  · decide
  · funext q
    rfl
def root302Tail39 : List Nat := []
def root302Tail38 : List Nat := 23269 :: root302Tail39
def root302Tail37 : List Nat := 22573 :: root302Tail38
def root302Tail36 : List Nat := 22093 :: root302Tail37
def root302Tail35 : List Nat := 21673 :: root302Tail36
def root302Tail34 : List Nat := 20749 :: root302Tail35
def root302Tail33 : List Nat := 20323 :: root302Tail34
def root302Tail32 : List Nat := 20287 :: root302Tail33
def root302Tail31 : List Nat := 20113 :: root302Tail32
def root302Tail30 : List Nat := 19423 :: root302Tail31
def root302Tail29 : List Nat := 18661 :: root302Tail30
def root302Tail28 : List Nat := 18481 :: root302Tail29
def root302Tail27 : List Nat := 17047 :: root302Tail28
def root302Tail26 : List Nat := 16693 :: root302Tail27
def root302Tail25 : List Nat := 16063 :: root302Tail26
def root302Tail24 : List Nat := 15877 :: root302Tail25
def root302Tail23 : List Nat := 15349 :: root302Tail24
def root302Tail22 : List Nat := 15331 :: root302Tail23
def root302Tail21 : List Nat := 14923 :: root302Tail22
def root302Tail20 : List Nat := 14293 :: root302Tail21
def root302Tail19 : List Nat := 13183 :: root302Tail20
def root302Tail18 : List Nat := 12409 :: root302Tail19
def root302Tail17 : List Nat := 10477 :: root302Tail18
def root302Tail16 : List Nat := 9769 :: root302Tail17
def root302Tail15 : List Nat := 9661 :: root302Tail16
def root302Tail14 : List Nat := 9433 :: root302Tail15
def root302Tail13 : List Nat := 9091 :: root302Tail14
def root302Tail12 : List Nat := 9049 :: root302Tail13
def root302Tail11 : List Nat := 8419 :: root302Tail12
def root302Tail10 : List Nat := 7963 :: root302Tail11
def root302Tail9 : List Nat := 6781 :: root302Tail10
def root302Tail8 : List Nat := 6637 :: root302Tail9
def root302Tail7 : List Nat := 6379 :: root302Tail8
def root302Tail6 : List Nat := 5521 :: root302Tail7
def root302Tail5 : List Nat := 5413 :: root302Tail6
def root302Tail4 : List Nat := 4177 :: root302Tail5
def root302Tail3 : List Nat := 4099 :: root302Tail4
def root302Tail2 : List Nat := 4093 :: root302Tail3
def root302Tail1 : List Nat := 3673 :: root302Tail2
def root302Tail0 : List Nat := 3253 :: root302Tail1
def root302Candidates0 : List Nat := [14293]
set_option maxRecDepth 8192 in
theorem root302Candidates0_checked : candidates edge 4 2011 3253 root302Tail1 winner.bound = root302Candidates0 := by
  unfold candidates
  have row : edge 3253 = (SortedNatTree.contains FastRows.row3253) := by
    change Data.row3253.contains = (SortedNatTree.contains FastRows.row3253)
    exact FastRows.row3253_checked.symm
  rw [row]
  decide
theorem root302Step0 : scan edge (search edge 3) 4 [2011] 2011 root302Tail0 winner =
    scan edge (search edge 3) 4 [2011] 2011 root302Tail1 winner := by
  rw [root302Tail0, scan, if_neg (by decide), root302Candidates0_checked]
  change scan edge (search edge 3) 4 [2011] 2011 root302Tail1
    (if 3 ≤ root302Candidates0.length then search edge 3 [2011, 3253] root302Candidates0 5264 winner else winner) = _
  rw [if_neg (by decide)]
def root302Candidates1 : List Nat := [13183]
set_option maxRecDepth 8192 in
theorem root302Candidates1_checked : candidates edge 4 2011 3673 root302Tail2 winner.bound = root302Candidates1 := by
  unfold candidates
  have row : edge 3673 = (SortedNatTree.contains FastRows.row3673) := by
    change Data.row3673.contains = (SortedNatTree.contains FastRows.row3673)
    exact FastRows.row3673_checked.symm
  rw [row]
  decide
theorem root302Step1 : scan edge (search edge 3) 4 [2011] 2011 root302Tail1 winner =
    scan edge (search edge 3) 4 [2011] 2011 root302Tail2 winner := by
  rw [root302Tail1, scan, if_neg (by decide), root302Candidates1_checked]
  change scan edge (search edge 3) 4 [2011] 2011 root302Tail2
    (if 3 ≤ root302Candidates1.length then search edge 3 [2011, 3673] root302Candidates1 5684 winner else winner) = _
  rw [if_neg (by decide)]
def root302Candidates2 : List Nat := [13183, 15877]
set_option maxRecDepth 8192 in
theorem root302Candidates2_checked : candidates edge 4 2011 4093 root302Tail3 winner.bound = root302Candidates2 := by
  unfold candidates
  have row : edge 4093 = (SortedNatTree.contains FastRows.row4093) := by
    change Data.row4093.contains = (SortedNatTree.contains FastRows.row4093)
    exact FastRows.row4093_checked.symm
  rw [row]
  decide
theorem root302Step2 : scan edge (search edge 3) 4 [2011] 2011 root302Tail2 winner =
    scan edge (search edge 3) 4 [2011] 2011 root302Tail3 winner := by
  rw [root302Tail2, scan, if_neg (by decide), root302Candidates2_checked]
  change scan edge (search edge 3) 4 [2011] 2011 root302Tail3
    (if 3 ≤ root302Candidates2.length then search edge 3 [2011, 4093] root302Candidates2 6104 winner else winner) = _
  rw [if_neg (by decide)]
def root302Candidates3 : List Nat := [16693]
set_option maxRecDepth 8192 in
theorem root302Candidates3_checked : candidates edge 4 2011 4099 root302Tail4 winner.bound = root302Candidates3 := by
  unfold candidates
  have row : edge 4099 = (SortedNatTree.contains FastRows.row4099) := by
    change Data.row4099.contains = (SortedNatTree.contains FastRows.row4099)
    exact FastRows.row4099_checked.symm
  rw [row]
  decide
theorem root302Step3 : scan edge (search edge 3) 4 [2011] 2011 root302Tail3 winner =
    scan edge (search edge 3) 4 [2011] 2011 root302Tail4 winner := by
  rw [root302Tail3, scan, if_neg (by decide), root302Candidates3_checked]
  change scan edge (search edge 3) 4 [2011] 2011 root302Tail4
    (if 3 ≤ root302Candidates3.length then search edge 3 [2011, 4099] root302Candidates3 6110 winner else winner) = _
  rw [if_neg (by decide)]
def root302Candidates4 : List Nat := [6379, 6781]
set_option maxRecDepth 8192 in
theorem root302Candidates4_checked : candidates edge 4 2011 4177 root302Tail5 winner.bound = root302Candidates4 := by
  unfold candidates
  have row : edge 4177 = (SortedNatTree.contains FastRows.row4177) := by
    change Data.row4177.contains = (SortedNatTree.contains FastRows.row4177)
    exact FastRows.row4177_checked.symm
  rw [row]
  decide
theorem root302Step4 : scan edge (search edge 3) 4 [2011] 2011 root302Tail4 winner =
    scan edge (search edge 3) 4 [2011] 2011 root302Tail5 winner := by
  rw [root302Tail4, scan, if_neg (by decide), root302Candidates4_checked]
  change scan edge (search edge 3) 4 [2011] 2011 root302Tail5
    (if 3 ≤ root302Candidates4.length then search edge 3 [2011, 4177] root302Candidates4 6188 winner else winner) = _
  rw [if_neg (by decide)]
def root302Candidates5 : List Nat := [10477]
set_option maxRecDepth 8192 in
theorem root302Candidates5_checked : candidates edge 4 2011 5413 root302Tail6 winner.bound = root302Candidates5 := by
  unfold candidates
  have row : edge 5413 = (SortedNatTree.contains FastRows.row5413) := by
    change Data.row5413.contains = (SortedNatTree.contains FastRows.row5413)
    exact FastRows.row5413_checked.symm
  rw [row]
  decide
theorem root302Step5 : scan edge (search edge 3) 4 [2011] 2011 root302Tail5 winner =
    scan edge (search edge 3) 4 [2011] 2011 root302Tail6 winner := by
  rw [root302Tail5, scan, if_neg (by decide), root302Candidates5_checked]
  change scan edge (search edge 3) 4 [2011] 2011 root302Tail6
    (if 3 ≤ root302Candidates5.length then search edge 3 [2011, 5413] root302Candidates5 7424 winner else winner) = _
  rw [if_neg (by decide)]
def root302Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root302Candidates6_checked : candidates edge 4 2011 5521 root302Tail7 winner.bound = root302Candidates6 := by
  unfold candidates
  have row : edge 5521 = (SortedNatTree.contains FastRows.row5521) := by
    change Data.row5521.contains = (SortedNatTree.contains FastRows.row5521)
    exact FastRows.row5521_checked.symm
  rw [row]
  decide
theorem root302Step6 : scan edge (search edge 3) 4 [2011] 2011 root302Tail6 winner =
    scan edge (search edge 3) 4 [2011] 2011 root302Tail7 winner := by
  rw [root302Tail6, scan, if_neg (by decide), root302Candidates6_checked]
  change scan edge (search edge 3) 4 [2011] 2011 root302Tail7
    (if 3 ≤ root302Candidates6.length then search edge 3 [2011, 5521] root302Candidates6 7532 winner else winner) = _
  rw [if_neg (by decide)]
theorem root302Scan7 : scan edge (search edge 3) 4 [2011] 2011 root302Tail7 winner = winner := by rw [root302Tail7, scan, if_pos (by decide)]
theorem root302Scan6 : scan edge (search edge 3) 4 [2011] 2011 root302Tail6 winner = winner :=
  (root302Step6).trans root302Scan7
theorem root302Scan5 : scan edge (search edge 3) 4 [2011] 2011 root302Tail5 winner = winner :=
  (root302Step5).trans root302Scan6
theorem root302Scan4 : scan edge (search edge 3) 4 [2011] 2011 root302Tail4 winner = winner :=
  (root302Step4).trans root302Scan5
theorem root302Scan3 : scan edge (search edge 3) 4 [2011] 2011 root302Tail3 winner = winner :=
  (root302Step3).trans root302Scan4
theorem root302Scan2 : scan edge (search edge 3) 4 [2011] 2011 root302Tail2 winner = winner :=
  (root302Step2).trans root302Scan3
theorem root302Scan1 : scan edge (search edge 3) 4 [2011] 2011 root302Tail1 winner = winner :=
  (root302Step1).trans root302Scan2
theorem root302Scan0 : scan edge (search edge 3) 4 [2011] 2011 root302Tail0 winner = winner :=
  (root302Step0).trans root302Scan1
set_option maxRecDepth 4096 in
theorem root302Branch : search edge 4 [2011] root302Candidates 2011 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2011] 2011 root302Tail0 winner = winner
  exact root302Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
