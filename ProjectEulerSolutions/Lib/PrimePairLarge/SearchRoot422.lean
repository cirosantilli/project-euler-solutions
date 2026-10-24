import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root422Candidates : List Nat := Data.row2953
set_option maxRecDepth 8192 in
theorem root422Candidates_checked : candidates edge 5 0 2953 tail423 winner.bound = root422Candidates := by
  apply candidates_eq_row edge 5 0 2953 tail423 root422Candidates winner.bound tail423_ordered
  · decide
  · decide
  · funext q
    rfl
def root422Tail39 : List Nat := []
def root422Tail38 : List Nat := 21001 :: root422Tail39
def root422Tail37 : List Nat := 20359 :: root422Tail38
def root422Tail36 : List Nat := 20101 :: root422Tail37
def root422Tail35 : List Nat := 19867 :: root422Tail36
def root422Tail34 : List Nat := 18553 :: root422Tail35
def root422Tail33 : List Nat := 18313 :: root422Tail34
def root422Tail32 : List Nat := 18217 :: root422Tail33
def root422Tail31 : List Nat := 16057 :: root422Tail32
def root422Tail30 : List Nat := 15919 :: root422Tail31
def root422Tail29 : List Nat := 15679 :: root422Tail30
def root422Tail28 : List Nat := 14683 :: root422Tail29
def root422Tail27 : List Nat := 14629 :: root422Tail28
def root422Tail26 : List Nat := 14281 :: root422Tail27
def root422Tail25 : List Nat := 13633 :: root422Tail26
def root422Tail24 : List Nat := 12517 :: root422Tail25
def root422Tail23 : List Nat := 10459 :: root422Tail24
def root422Tail22 : List Nat := 10069 :: root422Tail23
def root422Tail21 : List Nat := 9649 :: root422Tail22
def root422Tail20 : List Nat := 9343 :: root422Tail21
def root422Tail19 : List Nat := 8461 :: root422Tail20
def root422Tail18 : List Nat := 7867 :: root422Tail19
def root422Tail17 : List Nat := 7759 :: root422Tail18
def root422Tail16 : List Nat := 7591 :: root422Tail17
def root422Tail15 : List Nat := 7537 :: root422Tail16
def root422Tail14 : List Nat := 7369 :: root422Tail15
def root422Tail13 : List Nat := 7309 :: root422Tail14
def root422Tail12 : List Nat := 7213 :: root422Tail13
def root422Tail11 : List Nat := 6907 :: root422Tail12
def root422Tail10 : List Nat := 6733 :: root422Tail11
def root422Tail9 : List Nat := 6709 :: root422Tail10
def root422Tail8 : List Nat := 6679 :: root422Tail9
def root422Tail7 : List Nat := 6271 :: root422Tail8
def root422Tail6 : List Nat := 6217 :: root422Tail7
def root422Tail5 : List Nat := 4957 :: root422Tail6
def root422Tail4 : List Nat := 4357 :: root422Tail5
def root422Tail3 : List Nat := 4273 :: root422Tail4
def root422Tail2 : List Nat := 3643 :: root422Tail3
def root422Tail1 : List Nat := 3433 :: root422Tail2
def root422Tail0 : List Nat := 3181 :: root422Tail1
def root422Candidates0 : List Nat := [4273, 6709, 18217]
set_option maxRecDepth 8192 in
theorem root422Candidates0_checked : candidates edge 4 2953 3181 root422Tail1 winner.bound = root422Candidates0 := by
  unfold candidates
  have row : edge 3181 = (SortedNatTree.contains FastRows.row3181) := by
    change Data.row3181.contains = (SortedNatTree.contains FastRows.row3181)
    exact FastRows.row3181_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root422Child0 : search edge 3 [2953, 3181] root422Candidates0 6134 winner = winner := by decide
theorem root422Step0 : scan edge (search edge 3) 4 [2953] 2953 root422Tail0 winner =
    scan edge (search edge 3) 4 [2953] 2953 root422Tail1 winner := by
  rw [root422Tail0, scan, if_neg (by decide), root422Candidates0_checked]
  change scan edge (search edge 3) 4 [2953] 2953 root422Tail1
    (if 3 ≤ root422Candidates0.length then search edge 3 [2953, 3181] root422Candidates0 6134 winner else winner) = _
  rw [if_pos (by decide), root422Child0]
def root422Candidates1 : List Nat := [3643, 6733, 8461]
set_option maxRecDepth 8192 in
theorem root422Candidates1_checked : candidates edge 4 2953 3433 root422Tail2 winner.bound = root422Candidates1 := by
  unfold candidates
  have row : edge 3433 = (SortedNatTree.contains FastRows.row3433) := by
    change Data.row3433.contains = (SortedNatTree.contains FastRows.row3433)
    exact FastRows.row3433_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root422Child1 : search edge 3 [2953, 3433] root422Candidates1 6386 winner = winner := by decide
theorem root422Step1 : scan edge (search edge 3) 4 [2953] 2953 root422Tail1 winner =
    scan edge (search edge 3) 4 [2953] 2953 root422Tail2 winner := by
  rw [root422Tail1, scan, if_neg (by decide), root422Candidates1_checked]
  change scan edge (search edge 3) 4 [2953] 2953 root422Tail2
    (if 3 ≤ root422Candidates1.length then search edge 3 [2953, 3433] root422Candidates1 6386 winner else winner) = _
  rw [if_pos (by decide), root422Child1]
def root422Candidates2 : List Nat := [4273, 18553]
set_option maxRecDepth 8192 in
theorem root422Candidates2_checked : candidates edge 4 2953 3643 root422Tail3 winner.bound = root422Candidates2 := by
  unfold candidates
  have row : edge 3643 = (SortedNatTree.contains FastRows.row3643) := by
    change Data.row3643.contains = (SortedNatTree.contains FastRows.row3643)
    exact FastRows.row3643_checked.symm
  rw [row]
  decide
theorem root422Step2 : scan edge (search edge 3) 4 [2953] 2953 root422Tail2 winner =
    scan edge (search edge 3) 4 [2953] 2953 root422Tail3 winner := by
  rw [root422Tail2, scan, if_neg (by decide), root422Candidates2_checked]
  change scan edge (search edge 3) 4 [2953] 2953 root422Tail3
    (if 3 ≤ root422Candidates2.length then search edge 3 [2953, 3643] root422Candidates2 6596 winner else winner) = _
  rw [if_neg (by decide)]
def root422Candidates3 : List Nat := [8461, 14683]
set_option maxRecDepth 8192 in
theorem root422Candidates3_checked : candidates edge 4 2953 4273 root422Tail4 winner.bound = root422Candidates3 := by
  unfold candidates
  have row : edge 4273 = (SortedNatTree.contains FastRows.row4273) := by
    change Data.row4273.contains = (SortedNatTree.contains FastRows.row4273)
    exact FastRows.row4273_checked.symm
  rw [row]
  decide
theorem root422Step3 : scan edge (search edge 3) 4 [2953] 2953 root422Tail3 winner =
    scan edge (search edge 3) 4 [2953] 2953 root422Tail4 winner := by
  rw [root422Tail3, scan, if_neg (by decide), root422Candidates3_checked]
  change scan edge (search edge 3) 4 [2953] 2953 root422Tail4
    (if 3 ≤ root422Candidates3.length then search edge 3 [2953, 4273] root422Candidates3 7226 winner else winner) = _
  rw [if_neg (by decide)]
def root422Candidates4 : List Nat := [18553]
set_option maxRecDepth 8192 in
theorem root422Candidates4_checked : candidates edge 4 2953 4357 root422Tail5 winner.bound = root422Candidates4 := by
  unfold candidates
  have row : edge 4357 = (SortedNatTree.contains FastRows.row4357) := by
    change Data.row4357.contains = (SortedNatTree.contains FastRows.row4357)
    exact FastRows.row4357_checked.symm
  rw [row]
  decide
theorem root422Step4 : scan edge (search edge 3) 4 [2953] 2953 root422Tail4 winner =
    scan edge (search edge 3) 4 [2953] 2953 root422Tail5 winner := by
  rw [root422Tail4, scan, if_neg (by decide), root422Candidates4_checked]
  change scan edge (search edge 3) 4 [2953] 2953 root422Tail5
    (if 3 ≤ root422Candidates4.length then search edge 3 [2953, 4357] root422Candidates4 7310 winner else winner) = _
  rw [if_neg (by decide)]
def root422Candidates5 : List Nat := [15919]
set_option maxRecDepth 8192 in
theorem root422Candidates5_checked : candidates edge 4 2953 4957 root422Tail6 winner.bound = root422Candidates5 := by
  unfold candidates
  have row : edge 4957 = (SortedNatTree.contains FastRows.row4957) := by
    change Data.row4957.contains = (SortedNatTree.contains FastRows.row4957)
    exact FastRows.row4957_checked.symm
  rw [row]
  decide
theorem root422Step5 : scan edge (search edge 3) 4 [2953] 2953 root422Tail5 winner =
    scan edge (search edge 3) 4 [2953] 2953 root422Tail6 winner := by
  rw [root422Tail5, scan, if_neg (by decide), root422Candidates5_checked]
  change scan edge (search edge 3) 4 [2953] 2953 root422Tail6
    (if 3 ≤ root422Candidates5.length then search edge 3 [2953, 4957] root422Candidates5 7910 winner else winner) = _
  rw [if_neg (by decide)]
theorem root422Scan6 : scan edge (search edge 3) 4 [2953] 2953 root422Tail6 winner = winner := by rw [root422Tail6, scan, if_pos (by decide)]
theorem root422Scan5 : scan edge (search edge 3) 4 [2953] 2953 root422Tail5 winner = winner :=
  (root422Step5).trans root422Scan6
theorem root422Scan4 : scan edge (search edge 3) 4 [2953] 2953 root422Tail4 winner = winner :=
  (root422Step4).trans root422Scan5
theorem root422Scan3 : scan edge (search edge 3) 4 [2953] 2953 root422Tail3 winner = winner :=
  (root422Step3).trans root422Scan4
theorem root422Scan2 : scan edge (search edge 3) 4 [2953] 2953 root422Tail2 winner = winner :=
  (root422Step2).trans root422Scan3
theorem root422Scan1 : scan edge (search edge 3) 4 [2953] 2953 root422Tail1 winner = winner :=
  (root422Step1).trans root422Scan2
theorem root422Scan0 : scan edge (search edge 3) 4 [2953] 2953 root422Tail0 winner = winner :=
  (root422Step0).trans root422Scan1
set_option maxRecDepth 4096 in
theorem root422Branch : search edge 4 [2953] root422Candidates 2953 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2953] 2953 root422Tail0 winner = winner
  exact root422Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
