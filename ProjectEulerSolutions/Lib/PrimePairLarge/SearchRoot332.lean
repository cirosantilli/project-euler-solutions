import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root332Candidates : List Nat := Data.row2251
set_option maxRecDepth 8192 in
theorem root332Candidates_checked : candidates edge 5 0 2251 tail333 winner.bound = root332Candidates := by
  apply candidates_eq_row edge 5 0 2251 tail333 root332Candidates winner.bound tail333_ordered
  · decide
  · decide
  · funext q
    rfl
def root332Tail35 : List Nat := []
def root332Tail34 : List Nat := 23719 :: root332Tail35
def root332Tail33 : List Nat := 23593 :: root332Tail34
def root332Tail32 : List Nat := 22039 :: root332Tail33
def root332Tail31 : List Nat := 21313 :: root332Tail32
def root332Tail30 : List Nat := 20641 :: root332Tail31
def root332Tail29 : List Nat := 19717 :: root332Tail30
def root332Tail28 : List Nat := 16651 :: root332Tail29
def root332Tail27 : List Nat := 15271 :: root332Tail28
def root332Tail26 : List Nat := 14683 :: root332Tail27
def root332Tail25 : List Nat := 13417 :: root332Tail26
def root332Tail24 : List Nat := 13219 :: root332Tail25
def root332Tail23 : List Nat := 13003 :: root332Tail24
def root332Tail22 : List Nat := 12541 :: root332Tail23
def root332Tail21 : List Nat := 12457 :: root332Tail22
def root332Tail20 : List Nat := 12157 :: root332Tail21
def root332Tail19 : List Nat := 11839 :: root332Tail20
def root332Tail18 : List Nat := 10939 :: root332Tail19
def root332Tail17 : List Nat := 9787 :: root332Tail18
def root332Tail16 : List Nat := 9613 :: root332Tail17
def root332Tail15 : List Nat := 9349 :: root332Tail16
def root332Tail14 : List Nat := 9103 :: root332Tail15
def root332Tail13 : List Nat := 8233 :: root332Tail14
def root332Tail12 : List Nat := 8191 :: root332Tail13
def root332Tail11 : List Nat := 7927 :: root332Tail12
def root332Tail10 : List Nat := 7669 :: root332Tail11
def root332Tail9 : List Nat := 6841 :: root332Tail10
def root332Tail8 : List Nat := 6451 :: root332Tail9
def root332Tail7 : List Nat := 5881 :: root332Tail8
def root332Tail6 : List Nat := 5503 :: root332Tail7
def root332Tail5 : List Nat := 4201 :: root332Tail6
def root332Tail4 : List Nat := 4111 :: root332Tail5
def root332Tail3 : List Nat := 4021 :: root332Tail4
def root332Tail2 : List Nat := 3307 :: root332Tail3
def root332Tail1 : List Nat := 3217 :: root332Tail2
def root332Tail0 : List Nat := 2593 :: root332Tail1
def root332Candidates0 : List Nat := [3307, 7927, 12541]
set_option maxRecDepth 8192 in
theorem root332Candidates0_checked : candidates edge 4 2251 2593 root332Tail1 winner.bound = root332Candidates0 := by
  unfold candidates
  have row : edge 2593 = (SortedNatTree.contains FastRows.row2593) := by
    change Data.row2593.contains = (SortedNatTree.contains FastRows.row2593)
    exact FastRows.row2593_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root332Child0 : search edge 3 [2251, 2593] root332Candidates0 4844 winner = winner := by decide
theorem root332Step0 : scan edge (search edge 3) 4 [2251] 2251 root332Tail0 winner =
    scan edge (search edge 3) 4 [2251] 2251 root332Tail1 winner := by
  rw [root332Tail0, scan, if_neg (by decide), root332Candidates0_checked]
  change scan edge (search edge 3) 4 [2251] 2251 root332Tail1
    (if 3 ≤ root332Candidates0.length then search edge 3 [2251, 2593] root332Candidates0 4844 winner else winner) = _
  rw [if_pos (by decide), root332Child0]
def root332Candidates1 : List Nat := [13417]
set_option maxRecDepth 8192 in
theorem root332Candidates1_checked : candidates edge 4 2251 3217 root332Tail2 winner.bound = root332Candidates1 := by
  unfold candidates
  have row : edge 3217 = (SortedNatTree.contains FastRows.row3217) := by
    change Data.row3217.contains = (SortedNatTree.contains FastRows.row3217)
    exact FastRows.row3217_checked.symm
  rw [row]
  decide
theorem root332Step1 : scan edge (search edge 3) 4 [2251] 2251 root332Tail1 winner =
    scan edge (search edge 3) 4 [2251] 2251 root332Tail2 winner := by
  rw [root332Tail1, scan, if_neg (by decide), root332Candidates1_checked]
  change scan edge (search edge 3) 4 [2251] 2251 root332Tail2
    (if 3 ≤ root332Candidates1.length then search edge 3 [2251, 3217] root332Candidates1 5468 winner else winner) = _
  rw [if_neg (by decide)]
def root332Candidates2 : List Nat := [9349]
set_option maxRecDepth 8192 in
theorem root332Candidates2_checked : candidates edge 4 2251 3307 root332Tail3 winner.bound = root332Candidates2 := by
  unfold candidates
  have row : edge 3307 = (SortedNatTree.contains FastRows.row3307) := by
    change Data.row3307.contains = (SortedNatTree.contains FastRows.row3307)
    exact FastRows.row3307_checked.symm
  rw [row]
  decide
theorem root332Step2 : scan edge (search edge 3) 4 [2251] 2251 root332Tail2 winner =
    scan edge (search edge 3) 4 [2251] 2251 root332Tail3 winner := by
  rw [root332Tail2, scan, if_neg (by decide), root332Candidates2_checked]
  change scan edge (search edge 3) 4 [2251] 2251 root332Tail3
    (if 3 ≤ root332Candidates2.length then search edge 3 [2251, 3307] root332Candidates2 5558 winner else winner) = _
  rw [if_neg (by decide)]
def root332Candidates3 : List Nat := [9349, 13417, 16651]
set_option maxRecDepth 8192 in
theorem root332Candidates3_checked : candidates edge 4 2251 4021 root332Tail4 winner.bound = root332Candidates3 := by
  unfold candidates
  have row : edge 4021 = (SortedNatTree.contains FastRows.row4021) := by
    change Data.row4021.contains = (SortedNatTree.contains FastRows.row4021)
    exact FastRows.row4021_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root332Child3 : search edge 3 [2251, 4021] root332Candidates3 6272 winner = winner := by decide
theorem root332Step3 : scan edge (search edge 3) 4 [2251] 2251 root332Tail3 winner =
    scan edge (search edge 3) 4 [2251] 2251 root332Tail4 winner := by
  rw [root332Tail3, scan, if_neg (by decide), root332Candidates3_checked]
  change scan edge (search edge 3) 4 [2251] 2251 root332Tail4
    (if 3 ≤ root332Candidates3.length then search edge 3 [2251, 4021] root332Candidates3 6272 winner else winner) = _
  rw [if_pos (by decide), root332Child3]
def root332Candidates4 : List Nat := [9349, 12157]
set_option maxRecDepth 8192 in
theorem root332Candidates4_checked : candidates edge 4 2251 4111 root332Tail5 winner.bound = root332Candidates4 := by
  unfold candidates
  have row : edge 4111 = (SortedNatTree.contains FastRows.row4111) := by
    change Data.row4111.contains = (SortedNatTree.contains FastRows.row4111)
    exact FastRows.row4111_checked.symm
  rw [row]
  decide
theorem root332Step4 : scan edge (search edge 3) 4 [2251] 2251 root332Tail4 winner =
    scan edge (search edge 3) 4 [2251] 2251 root332Tail5 winner := by
  rw [root332Tail4, scan, if_neg (by decide), root332Candidates4_checked]
  change scan edge (search edge 3) 4 [2251] 2251 root332Tail5
    (if 3 ≤ root332Candidates4.length then search edge 3 [2251, 4111] root332Candidates4 6362 winner else winner) = _
  rw [if_neg (by decide)]
def root332Candidates5 : List Nat := [13219]
set_option maxRecDepth 8192 in
theorem root332Candidates5_checked : candidates edge 4 2251 4201 root332Tail6 winner.bound = root332Candidates5 := by
  unfold candidates
  have row : edge 4201 = (SortedNatTree.contains FastRows.row4201) := by
    change Data.row4201.contains = (SortedNatTree.contains FastRows.row4201)
    exact FastRows.row4201_checked.symm
  rw [row]
  decide
theorem root332Step5 : scan edge (search edge 3) 4 [2251] 2251 root332Tail5 winner =
    scan edge (search edge 3) 4 [2251] 2251 root332Tail6 winner := by
  rw [root332Tail5, scan, if_neg (by decide), root332Candidates5_checked]
  change scan edge (search edge 3) 4 [2251] 2251 root332Tail6
    (if 3 ≤ root332Candidates5.length then search edge 3 [2251, 4201] root332Candidates5 6452 winner else winner) = _
  rw [if_neg (by decide)]
def root332Candidates6 : List Nat := [12541]
set_option maxRecDepth 8192 in
theorem root332Candidates6_checked : candidates edge 4 2251 5503 root332Tail7 winner.bound = root332Candidates6 := by
  unfold candidates
  have row : edge 5503 = (SortedNatTree.contains FastRows.row5503) := by
    change Data.row5503.contains = (SortedNatTree.contains FastRows.row5503)
    exact FastRows.row5503_checked.symm
  rw [row]
  decide
theorem root332Step6 : scan edge (search edge 3) 4 [2251] 2251 root332Tail6 winner =
    scan edge (search edge 3) 4 [2251] 2251 root332Tail7 winner := by
  rw [root332Tail6, scan, if_neg (by decide), root332Candidates6_checked]
  change scan edge (search edge 3) 4 [2251] 2251 root332Tail7
    (if 3 ≤ root332Candidates6.length then search edge 3 [2251, 5503] root332Candidates6 7754 winner else winner) = _
  rw [if_neg (by decide)]
def root332Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root332Candidates7_checked : candidates edge 4 2251 5881 root332Tail8 winner.bound = root332Candidates7 := by
  unfold candidates
  have row : edge 5881 = (SortedNatTree.contains FastRows.row5881) := by
    change Data.row5881.contains = (SortedNatTree.contains FastRows.row5881)
    exact FastRows.row5881_checked.symm
  rw [row]
  decide
theorem root332Step7 : scan edge (search edge 3) 4 [2251] 2251 root332Tail7 winner =
    scan edge (search edge 3) 4 [2251] 2251 root332Tail8 winner := by
  rw [root332Tail7, scan, if_neg (by decide), root332Candidates7_checked]
  change scan edge (search edge 3) 4 [2251] 2251 root332Tail8
    (if 3 ≤ root332Candidates7.length then search edge 3 [2251, 5881] root332Candidates7 8132 winner else winner) = _
  rw [if_neg (by decide)]
theorem root332Scan8 : scan edge (search edge 3) 4 [2251] 2251 root332Tail8 winner = winner := by rw [root332Tail8, scan, if_pos (by decide)]
theorem root332Scan7 : scan edge (search edge 3) 4 [2251] 2251 root332Tail7 winner = winner :=
  (root332Step7).trans root332Scan8
theorem root332Scan6 : scan edge (search edge 3) 4 [2251] 2251 root332Tail6 winner = winner :=
  (root332Step6).trans root332Scan7
theorem root332Scan5 : scan edge (search edge 3) 4 [2251] 2251 root332Tail5 winner = winner :=
  (root332Step5).trans root332Scan6
theorem root332Scan4 : scan edge (search edge 3) 4 [2251] 2251 root332Tail4 winner = winner :=
  (root332Step4).trans root332Scan5
theorem root332Scan3 : scan edge (search edge 3) 4 [2251] 2251 root332Tail3 winner = winner :=
  (root332Step3).trans root332Scan4
theorem root332Scan2 : scan edge (search edge 3) 4 [2251] 2251 root332Tail2 winner = winner :=
  (root332Step2).trans root332Scan3
theorem root332Scan1 : scan edge (search edge 3) 4 [2251] 2251 root332Tail1 winner = winner :=
  (root332Step1).trans root332Scan2
theorem root332Scan0 : scan edge (search edge 3) 4 [2251] 2251 root332Tail0 winner = winner :=
  (root332Step0).trans root332Scan1
set_option maxRecDepth 4096 in
theorem root332Branch : search edge 4 [2251] root332Candidates 2251 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2251] 2251 root332Tail0 winner = winner
  exact root332Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
