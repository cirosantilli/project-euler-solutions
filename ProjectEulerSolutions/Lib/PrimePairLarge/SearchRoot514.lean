import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root514Candidates : List Nat := Data.row3701
set_option maxRecDepth 8192 in
theorem root514Candidates_checked : candidates edge 5 0 3701 tail515 winner.bound = root514Candidates := by
  apply candidates_eq_row edge 5 0 3701 tail515 root514Candidates winner.bound tail515_ordered
  · decide
  · decide
  · funext q
    rfl
def root514Tail38 : List Nat := []
def root514Tail37 : List Nat := 22073 :: root514Tail38
def root514Tail36 : List Nat := 21821 :: root514Tail37
def root514Tail35 : List Nat := 21491 :: root514Tail36
def root514Tail34 : List Nat := 19073 :: root514Tail35
def root514Tail33 : List Nat := 17333 :: root514Tail34
def root514Tail32 : List Nat := 16301 :: root514Tail33
def root514Tail31 : List Nat := 16127 :: root514Tail32
def root514Tail30 : List Nat := 16001 :: root514Tail31
def root514Tail29 : List Nat := 14951 :: root514Tail30
def root514Tail28 : List Nat := 14639 :: root514Tail29
def root514Tail27 : List Nat := 14369 :: root514Tail28
def root514Tail26 : List Nat := 13859 :: root514Tail27
def root514Tail25 : List Nat := 13697 :: root514Tail26
def root514Tail24 : List Nat := 13463 :: root514Tail25
def root514Tail23 : List Nat := 12539 :: root514Tail24
def root514Tail22 : List Nat := 11867 :: root514Tail23
def root514Tail21 : List Nat := 10781 :: root514Tail22
def root514Tail20 : List Nat := 9791 :: root514Tail21
def root514Tail19 : List Nat := 9623 :: root514Tail20
def root514Tail18 : List Nat := 9473 :: root514Tail19
def root514Tail17 : List Nat := 9161 :: root514Tail18
def root514Tail16 : List Nat := 8609 :: root514Tail17
def root514Tail15 : List Nat := 8543 :: root514Tail16
def root514Tail14 : List Nat := 7829 :: root514Tail15
def root514Tail13 : List Nat := 6911 :: root514Tail14
def root514Tail12 : List Nat := 6869 :: root514Tail13
def root514Tail11 : List Nat := 6299 :: root514Tail12
def root514Tail10 : List Nat := 5501 :: root514Tail11
def root514Tail9 : List Nat := 5351 :: root514Tail10
def root514Tail8 : List Nat := 5309 :: root514Tail9
def root514Tail7 : List Nat := 5081 :: root514Tail8
def root514Tail6 : List Nat := 4973 :: root514Tail7
def root514Tail5 : List Nat := 4931 :: root514Tail6
def root514Tail4 : List Nat := 4919 :: root514Tail5
def root514Tail3 : List Nat := 4877 :: root514Tail4
def root514Tail2 : List Nat := 4493 :: root514Tail3
def root514Tail1 : List Nat := 4283 :: root514Tail2
def root514Tail0 : List Nat := 3797 :: root514Tail1
def root514Candidates0 : List Nat := [9791]
set_option maxRecDepth 8192 in
theorem root514Candidates0_checked : candidates edge 4 3701 3797 root514Tail1 winner.bound = root514Candidates0 := by
  unfold candidates
  have row : edge 3797 = (SortedNatTree.contains FastRows.row3797) := by
    change Data.row3797.contains = (SortedNatTree.contains FastRows.row3797)
    exact FastRows.row3797_checked.symm
  rw [row]
  decide
theorem root514Step0 : scan edge (search edge 3) 4 [3701] 3701 root514Tail0 winner =
    scan edge (search edge 3) 4 [3701] 3701 root514Tail1 winner := by
  rw [root514Tail0, scan, if_neg (by decide), root514Candidates0_checked]
  change scan edge (search edge 3) 4 [3701] 3701 root514Tail1
    (if 3 ≤ root514Candidates0.length then search edge 3 [3701, 3797] root514Candidates0 7498 winner else winner) = _
  rw [if_neg (by decide)]
def root514Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root514Candidates1_checked : candidates edge 4 3701 4283 root514Tail2 winner.bound = root514Candidates1 := by
  unfold candidates
  have row : edge 4283 = (SortedNatTree.contains FastRows.row4283) := by
    change Data.row4283.contains = (SortedNatTree.contains FastRows.row4283)
    exact FastRows.row4283_checked.symm
  rw [row]
  decide
theorem root514Step1 : scan edge (search edge 3) 4 [3701] 3701 root514Tail1 winner =
    scan edge (search edge 3) 4 [3701] 3701 root514Tail2 winner := by
  rw [root514Tail1, scan, if_neg (by decide), root514Candidates1_checked]
  change scan edge (search edge 3) 4 [3701] 3701 root514Tail2
    (if 3 ≤ root514Candidates1.length then search edge 3 [3701, 4283] root514Candidates1 7984 winner else winner) = _
  rw [if_neg (by decide)]
def root514Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root514Candidates2_checked : candidates edge 4 3701 4493 root514Tail3 winner.bound = root514Candidates2 := by
  unfold candidates
  have row : edge 4493 = (SortedNatTree.contains FastRows.row4493) := by
    change Data.row4493.contains = (SortedNatTree.contains FastRows.row4493)
    exact FastRows.row4493_checked.symm
  rw [row]
  decide
theorem root514Step2 : scan edge (search edge 3) 4 [3701] 3701 root514Tail2 winner =
    scan edge (search edge 3) 4 [3701] 3701 root514Tail3 winner := by
  rw [root514Tail2, scan, if_neg (by decide), root514Candidates2_checked]
  change scan edge (search edge 3) 4 [3701] 3701 root514Tail3
    (if 3 ≤ root514Candidates2.length then search edge 3 [3701, 4493] root514Candidates2 8194 winner else winner) = _
  rw [if_neg (by decide)]
def root514Candidates3 : List Nat := [5501, 9161]
set_option maxRecDepth 8192 in
theorem root514Candidates3_checked : candidates edge 4 3701 4877 root514Tail4 winner.bound = root514Candidates3 := by
  unfold candidates
  have row : edge 4877 = (SortedNatTree.contains FastRows.row4877) := by
    change Data.row4877.contains = (SortedNatTree.contains FastRows.row4877)
    exact FastRows.row4877_checked.symm
  rw [row]
  decide
theorem root514Step3 : scan edge (search edge 3) 4 [3701] 3701 root514Tail3 winner =
    scan edge (search edge 3) 4 [3701] 3701 root514Tail4 winner := by
  rw [root514Tail3, scan, if_neg (by decide), root514Candidates3_checked]
  change scan edge (search edge 3) 4 [3701] 3701 root514Tail4
    (if 3 ≤ root514Candidates3.length then search edge 3 [3701, 4877] root514Candidates3 8578 winner else winner) = _
  rw [if_neg (by decide)]
def root514Candidates4 : List Nat := [6869, 7829, 10781]
set_option maxRecDepth 8192 in
theorem root514Candidates4_checked : candidates edge 4 3701 4919 root514Tail5 winner.bound = root514Candidates4 := by
  unfold candidates
  have row : edge 4919 = (SortedNatTree.contains FastRows.row4919) := by
    change Data.row4919.contains = (SortedNatTree.contains FastRows.row4919)
    exact FastRows.row4919_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root514Child4 : search edge 3 [3701, 4919] root514Candidates4 8620 winner = winner := by decide
theorem root514Step4 : scan edge (search edge 3) 4 [3701] 3701 root514Tail4 winner =
    scan edge (search edge 3) 4 [3701] 3701 root514Tail5 winner := by
  rw [root514Tail4, scan, if_neg (by decide), root514Candidates4_checked]
  change scan edge (search edge 3) 4 [3701] 3701 root514Tail5
    (if 3 ≤ root514Candidates4.length then search edge 3 [3701, 4919] root514Candidates4 8620 winner else winner) = _
  rw [if_pos (by decide), root514Child4]
def root514Candidates5 : List Nat := [5309]
set_option maxRecDepth 8192 in
theorem root514Candidates5_checked : candidates edge 4 3701 4931 root514Tail6 winner.bound = root514Candidates5 := by
  unfold candidates
  have row : edge 4931 = (SortedNatTree.contains FastRows.row4931) := by
    change Data.row4931.contains = (SortedNatTree.contains FastRows.row4931)
    exact FastRows.row4931_checked.symm
  rw [row]
  decide
theorem root514Step5 : scan edge (search edge 3) 4 [3701] 3701 root514Tail5 winner =
    scan edge (search edge 3) 4 [3701] 3701 root514Tail6 winner := by
  rw [root514Tail5, scan, if_neg (by decide), root514Candidates5_checked]
  change scan edge (search edge 3) 4 [3701] 3701 root514Tail6
    (if 3 ≤ root514Candidates5.length then search edge 3 [3701, 4931] root514Candidates5 8632 winner else winner) = _
  rw [if_neg (by decide)]
def root514Candidates6 : List Nat := [7829]
set_option maxRecDepth 8192 in
theorem root514Candidates6_checked : candidates edge 4 3701 4973 root514Tail7 winner.bound = root514Candidates6 := by
  unfold candidates
  have row : edge 4973 = (SortedNatTree.contains FastRows.row4973) := by
    change Data.row4973.contains = (SortedNatTree.contains FastRows.row4973)
    exact FastRows.row4973_checked.symm
  rw [row]
  decide
theorem root514Step6 : scan edge (search edge 3) 4 [3701] 3701 root514Tail6 winner =
    scan edge (search edge 3) 4 [3701] 3701 root514Tail7 winner := by
  rw [root514Tail6, scan, if_neg (by decide), root514Candidates6_checked]
  change scan edge (search edge 3) 4 [3701] 3701 root514Tail7
    (if 3 ≤ root514Candidates6.length then search edge 3 [3701, 4973] root514Candidates6 8674 winner else winner) = _
  rw [if_neg (by decide)]
def root514Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root514Candidates7_checked : candidates edge 4 3701 5081 root514Tail8 winner.bound = root514Candidates7 := by
  unfold candidates
  have row : edge 5081 = (SortedNatTree.contains FastRows.row5081) := by
    change Data.row5081.contains = (SortedNatTree.contains FastRows.row5081)
    exact FastRows.row5081_checked.symm
  rw [row]
  decide
theorem root514Step7 : scan edge (search edge 3) 4 [3701] 3701 root514Tail7 winner =
    scan edge (search edge 3) 4 [3701] 3701 root514Tail8 winner := by
  rw [root514Tail7, scan, if_neg (by decide), root514Candidates7_checked]
  change scan edge (search edge 3) 4 [3701] 3701 root514Tail8
    (if 3 ≤ root514Candidates7.length then search edge 3 [3701, 5081] root514Candidates7 8782 winner else winner) = _
  rw [if_neg (by decide)]
def root514Candidates8 : List Nat := [5501]
set_option maxRecDepth 8192 in
theorem root514Candidates8_checked : candidates edge 4 3701 5309 root514Tail9 winner.bound = root514Candidates8 := by
  unfold candidates
  have row : edge 5309 = (SortedNatTree.contains FastRows.row5309) := by
    change Data.row5309.contains = (SortedNatTree.contains FastRows.row5309)
    exact FastRows.row5309_checked.symm
  rw [row]
  decide
theorem root514Step8 : scan edge (search edge 3) 4 [3701] 3701 root514Tail8 winner =
    scan edge (search edge 3) 4 [3701] 3701 root514Tail9 winner := by
  rw [root514Tail8, scan, if_neg (by decide), root514Candidates8_checked]
  change scan edge (search edge 3) 4 [3701] 3701 root514Tail9
    (if 3 ≤ root514Candidates8.length then search edge 3 [3701, 5309] root514Candidates8 9010 winner else winner) = _
  rw [if_neg (by decide)]
def root514Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root514Candidates9_checked : candidates edge 4 3701 5351 root514Tail10 winner.bound = root514Candidates9 := by
  unfold candidates
  have row : edge 5351 = (SortedNatTree.contains FastRows.row5351) := by
    change Data.row5351.contains = (SortedNatTree.contains FastRows.row5351)
    exact FastRows.row5351_checked.symm
  rw [row]
  decide
theorem root514Step9 : scan edge (search edge 3) 4 [3701] 3701 root514Tail9 winner =
    scan edge (search edge 3) 4 [3701] 3701 root514Tail10 winner := by
  rw [root514Tail9, scan, if_neg (by decide), root514Candidates9_checked]
  change scan edge (search edge 3) 4 [3701] 3701 root514Tail10
    (if 3 ≤ root514Candidates9.length then search edge 3 [3701, 5351] root514Candidates9 9052 winner else winner) = _
  rw [if_neg (by decide)]
def root514Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root514Candidates10_checked : candidates edge 4 3701 5501 root514Tail11 winner.bound = root514Candidates10 := by
  unfold candidates
  have row : edge 5501 = (SortedNatTree.contains FastRows.row5501) := by
    change Data.row5501.contains = (SortedNatTree.contains FastRows.row5501)
    exact FastRows.row5501_checked.symm
  rw [row]
  decide
theorem root514Step10 : scan edge (search edge 3) 4 [3701] 3701 root514Tail10 winner =
    scan edge (search edge 3) 4 [3701] 3701 root514Tail11 winner := by
  rw [root514Tail10, scan, if_neg (by decide), root514Candidates10_checked]
  change scan edge (search edge 3) 4 [3701] 3701 root514Tail11
    (if 3 ≤ root514Candidates10.length then search edge 3 [3701, 5501] root514Candidates10 9202 winner else winner) = _
  rw [if_neg (by decide)]
theorem root514Scan11 : scan edge (search edge 3) 4 [3701] 3701 root514Tail11 winner = winner := by rw [root514Tail11, scan, if_pos (by decide)]
theorem root514Scan10 : scan edge (search edge 3) 4 [3701] 3701 root514Tail10 winner = winner :=
  (root514Step10).trans root514Scan11
theorem root514Scan9 : scan edge (search edge 3) 4 [3701] 3701 root514Tail9 winner = winner :=
  (root514Step9).trans root514Scan10
theorem root514Scan8 : scan edge (search edge 3) 4 [3701] 3701 root514Tail8 winner = winner :=
  (root514Step8).trans root514Scan9
theorem root514Scan7 : scan edge (search edge 3) 4 [3701] 3701 root514Tail7 winner = winner :=
  (root514Step7).trans root514Scan8
theorem root514Scan6 : scan edge (search edge 3) 4 [3701] 3701 root514Tail6 winner = winner :=
  (root514Step6).trans root514Scan7
theorem root514Scan5 : scan edge (search edge 3) 4 [3701] 3701 root514Tail5 winner = winner :=
  (root514Step5).trans root514Scan6
theorem root514Scan4 : scan edge (search edge 3) 4 [3701] 3701 root514Tail4 winner = winner :=
  (root514Step4).trans root514Scan5
theorem root514Scan3 : scan edge (search edge 3) 4 [3701] 3701 root514Tail3 winner = winner :=
  (root514Step3).trans root514Scan4
theorem root514Scan2 : scan edge (search edge 3) 4 [3701] 3701 root514Tail2 winner = winner :=
  (root514Step2).trans root514Scan3
theorem root514Scan1 : scan edge (search edge 3) 4 [3701] 3701 root514Tail1 winner = winner :=
  (root514Step1).trans root514Scan2
theorem root514Scan0 : scan edge (search edge 3) 4 [3701] 3701 root514Tail0 winner = winner :=
  (root514Step0).trans root514Scan1
set_option maxRecDepth 4096 in
theorem root514Branch : search edge 4 [3701] root514Candidates 3701 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3701] 3701 root514Tail0 winner = winner
  exact root514Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
