import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root478Candidates : List Nat := Data.row3433
set_option maxRecDepth 8192 in
theorem root478Candidates_checked : candidates edge 5 0 3433 tail479 winner.bound = root478Candidates := by
  apply candidates_eq_row edge 5 0 3433 tail479 root478Candidates winner.bound tail479_ordered
  · decide
  · decide
  · funext q
    rfl
def root478Tail38 : List Nat := []
def root478Tail37 : List Nat := 22171 :: root478Tail38
def root478Tail36 : List Nat := 21247 :: root478Tail37
def root478Tail35 : List Nat := 20983 :: root478Tail36
def root478Tail34 : List Nat := 20641 :: root478Tail35
def root478Tail33 : List Nat := 20431 :: root478Tail34
def root478Tail32 : List Nat := 19927 :: root478Tail33
def root478Tail31 : List Nat := 19687 :: root478Tail32
def root478Tail30 : List Nat := 19423 :: root478Tail31
def root478Tail29 : List Nat := 18427 :: root478Tail30
def root478Tail28 : List Nat := 17971 :: root478Tail29
def root478Tail27 : List Nat := 17551 :: root478Tail28
def root478Tail26 : List Nat := 17509 :: root478Tail27
def root478Tail25 : List Nat := 17383 :: root478Tail26
def root478Tail24 : List Nat := 17041 :: root478Tail25
def root478Tail23 : List Nat := 16063 :: root478Tail24
def root478Tail22 : List Nat := 15073 :: root478Tail23
def root478Tail21 : List Nat := 14929 :: root478Tail22
def root478Tail20 : List Nat := 14887 :: root478Tail21
def root478Tail19 : List Nat := 11443 :: root478Tail20
def root478Tail18 : List Nat := 11161 :: root478Tail19
def root478Tail17 : List Nat := 10987 :: root478Tail18
def root478Tail16 : List Nat := 8539 :: root478Tail17
def root478Tail15 : List Nat := 8461 :: root478Tail16
def root478Tail14 : List Nat := 8221 :: root478Tail15
def root478Tail13 : List Nat := 8167 :: root478Tail14
def root478Tail12 : List Nat := 7741 :: root478Tail13
def root478Tail11 : List Nat := 7207 :: root478Tail12
def root478Tail10 : List Nat := 7129 :: root478Tail11
def root478Tail9 : List Nat := 6733 :: root478Tail10
def root478Tail8 : List Nat := 6607 :: root478Tail9
def root478Tail7 : List Nat := 6451 :: root478Tail8
def root478Tail6 : List Nat := 6373 :: root478Tail7
def root478Tail5 : List Nat := 5407 :: root478Tail6
def root478Tail4 : List Nat := 5233 :: root478Tail5
def root478Tail3 : List Nat := 4987 :: root478Tail4
def root478Tail2 : List Nat := 4129 :: root478Tail3
def root478Tail1 : List Nat := 4051 :: root478Tail2
def root478Tail0 : List Nat := 3643 :: root478Tail1
def root478Candidates0 : List Nat := [4987, 6607]
set_option maxRecDepth 8192 in
theorem root478Candidates0_checked : candidates edge 4 3433 3643 root478Tail1 winner.bound = root478Candidates0 := by
  unfold candidates
  have row : edge 3643 = (SortedNatTree.contains FastRows.row3643) := by
    change Data.row3643.contains = (SortedNatTree.contains FastRows.row3643)
    exact FastRows.row3643_checked.symm
  rw [row]
  decide
theorem root478Step0 : scan edge (search edge 3) 4 [3433] 3433 root478Tail0 winner =
    scan edge (search edge 3) 4 [3433] 3433 root478Tail1 winner := by
  rw [root478Tail0, scan, if_neg (by decide), root478Candidates0_checked]
  change scan edge (search edge 3) 4 [3433] 3433 root478Tail1
    (if 3 ≤ root478Candidates0.length then search edge 3 [3433, 3643] root478Candidates0 7076 winner else winner) = _
  rw [if_neg (by decide)]
def root478Candidates1 : List Nat := [4129]
set_option maxRecDepth 8192 in
theorem root478Candidates1_checked : candidates edge 4 3433 4051 root478Tail2 winner.bound = root478Candidates1 := by
  unfold candidates
  have row : edge 4051 = (SortedNatTree.contains FastRows.row4051) := by
    change Data.row4051.contains = (SortedNatTree.contains FastRows.row4051)
    exact FastRows.row4051_checked.symm
  rw [row]
  decide
theorem root478Step1 : scan edge (search edge 3) 4 [3433] 3433 root478Tail1 winner =
    scan edge (search edge 3) 4 [3433] 3433 root478Tail2 winner := by
  rw [root478Tail1, scan, if_neg (by decide), root478Candidates1_checked]
  change scan edge (search edge 3) 4 [3433] 3433 root478Tail2
    (if 3 ≤ root478Candidates1.length then search edge 3 [3433, 4051] root478Candidates1 7484 winner else winner) = _
  rw [if_neg (by decide)]
def root478Candidates2 : List Nat := [8167, 8221]
set_option maxRecDepth 8192 in
theorem root478Candidates2_checked : candidates edge 4 3433 4129 root478Tail3 winner.bound = root478Candidates2 := by
  unfold candidates
  have row : edge 4129 = (SortedNatTree.contains FastRows.row4129) := by
    change Data.row4129.contains = (SortedNatTree.contains FastRows.row4129)
    exact FastRows.row4129_checked.symm
  rw [row]
  decide
theorem root478Step2 : scan edge (search edge 3) 4 [3433] 3433 root478Tail2 winner =
    scan edge (search edge 3) 4 [3433] 3433 root478Tail3 winner := by
  rw [root478Tail2, scan, if_neg (by decide), root478Candidates2_checked]
  change scan edge (search edge 3) 4 [3433] 3433 root478Tail3
    (if 3 ≤ root478Candidates2.length then search edge 3 [3433, 4129] root478Candidates2 7562 winner else winner) = _
  rw [if_neg (by decide)]
def root478Candidates3 : List Nat := [8461]
set_option maxRecDepth 8192 in
theorem root478Candidates3_checked : candidates edge 4 3433 4987 root478Tail4 winner.bound = root478Candidates3 := by
  unfold candidates
  have row : edge 4987 = (SortedNatTree.contains FastRows.row4987) := by
    change Data.row4987.contains = (SortedNatTree.contains FastRows.row4987)
    exact FastRows.row4987_checked.symm
  rw [row]
  decide
theorem root478Step3 : scan edge (search edge 3) 4 [3433] 3433 root478Tail3 winner =
    scan edge (search edge 3) 4 [3433] 3433 root478Tail4 winner := by
  rw [root478Tail3, scan, if_neg (by decide), root478Candidates3_checked]
  change scan edge (search edge 3) 4 [3433] 3433 root478Tail4
    (if 3 ≤ root478Candidates3.length then search edge 3 [3433, 4987] root478Candidates3 8420 winner else winner) = _
  rw [if_neg (by decide)]
def root478Candidates4 : List Nat := [6373, 6451]
set_option maxRecDepth 8192 in
theorem root478Candidates4_checked : candidates edge 4 3433 5233 root478Tail5 winner.bound = root478Candidates4 := by
  unfold candidates
  have row : edge 5233 = (SortedNatTree.contains FastRows.row5233) := by
    change Data.row5233.contains = (SortedNatTree.contains FastRows.row5233)
    exact FastRows.row5233_checked.symm
  rw [row]
  decide
theorem root478Step4 : scan edge (search edge 3) 4 [3433] 3433 root478Tail4 winner =
    scan edge (search edge 3) 4 [3433] 3433 root478Tail5 winner := by
  rw [root478Tail4, scan, if_neg (by decide), root478Candidates4_checked]
  change scan edge (search edge 3) 4 [3433] 3433 root478Tail5
    (if 3 ≤ root478Candidates4.length then search edge 3 [3433, 5233] root478Candidates4 8666 winner else winner) = _
  rw [if_neg (by decide)]
def root478Candidates5 : List Nat := [8461]
set_option maxRecDepth 8192 in
theorem root478Candidates5_checked : candidates edge 4 3433 5407 root478Tail6 winner.bound = root478Candidates5 := by
  unfold candidates
  have row : edge 5407 = (SortedNatTree.contains FastRows.row5407) := by
    change Data.row5407.contains = (SortedNatTree.contains FastRows.row5407)
    exact FastRows.row5407_checked.symm
  rw [row]
  decide
theorem root478Step5 : scan edge (search edge 3) 4 [3433] 3433 root478Tail5 winner =
    scan edge (search edge 3) 4 [3433] 3433 root478Tail6 winner := by
  rw [root478Tail5, scan, if_neg (by decide), root478Candidates5_checked]
  change scan edge (search edge 3) 4 [3433] 3433 root478Tail6
    (if 3 ≤ root478Candidates5.length then search edge 3 [3433, 5407] root478Candidates5 8840 winner else winner) = _
  rw [if_neg (by decide)]
theorem root478Scan6 : scan edge (search edge 3) 4 [3433] 3433 root478Tail6 winner = winner := by rw [root478Tail6, scan, if_pos (by decide)]
theorem root478Scan5 : scan edge (search edge 3) 4 [3433] 3433 root478Tail5 winner = winner :=
  (root478Step5).trans root478Scan6
theorem root478Scan4 : scan edge (search edge 3) 4 [3433] 3433 root478Tail4 winner = winner :=
  (root478Step4).trans root478Scan5
theorem root478Scan3 : scan edge (search edge 3) 4 [3433] 3433 root478Tail3 winner = winner :=
  (root478Step3).trans root478Scan4
theorem root478Scan2 : scan edge (search edge 3) 4 [3433] 3433 root478Tail2 winner = winner :=
  (root478Step2).trans root478Scan3
theorem root478Scan1 : scan edge (search edge 3) 4 [3433] 3433 root478Tail1 winner = winner :=
  (root478Step1).trans root478Scan2
theorem root478Scan0 : scan edge (search edge 3) 4 [3433] 3433 root478Tail0 winner = winner :=
  (root478Step0).trans root478Scan1
set_option maxRecDepth 4096 in
theorem root478Branch : search edge 4 [3433] root478Candidates 3433 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3433] 3433 root478Tail0 winner = winner
  exact root478Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
