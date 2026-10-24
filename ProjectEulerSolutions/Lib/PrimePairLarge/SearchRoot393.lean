import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root393Candidates : List Nat := Data.row2713
set_option maxRecDepth 8192 in
theorem root393Candidates_checked : candidates edge 5 0 2713 tail394 winner.bound = root393Candidates := by
  apply candidates_eq_row edge 5 0 2713 tail394 root393Candidates winner.bound tail394_ordered
  · decide
  · decide
  · funext q
    rfl
def root393Tail44 : List Nat := []
def root393Tail43 : List Nat := 23131 :: root393Tail44
def root393Tail42 : List Nat := 22783 :: root393Tail43
def root393Tail41 : List Nat := 22039 :: root393Tail42
def root393Tail40 : List Nat := 20011 :: root393Tail41
def root393Tail39 : List Nat := 19603 :: root393Tail40
def root393Tail38 : List Nat := 18211 :: root393Tail39
def root393Tail37 : List Nat := 17539 :: root393Tail38
def root393Tail36 : List Nat := 17419 :: root393Tail37
def root393Tail35 : List Nat := 17257 :: root393Tail36
def root393Tail34 : List Nat := 16189 :: root393Tail35
def root393Tail33 : List Nat := 15823 :: root393Tail34
def root393Tail32 : List Nat := 15817 :: root393Tail33
def root393Tail31 : List Nat := 15313 :: root393Tail32
def root393Tail30 : List Nat := 15013 :: root393Tail31
def root393Tail29 : List Nat := 14551 :: root393Tail30
def root393Tail28 : List Nat := 14431 :: root393Tail29
def root393Tail27 : List Nat := 13807 :: root393Tail28
def root393Tail26 : List Nat := 13477 :: root393Tail27
def root393Tail25 : List Nat := 12583 :: root393Tail26
def root393Tail24 : List Nat := 12553 :: root393Tail25
def root393Tail23 : List Nat := 10909 :: root393Tail24
def root393Tail22 : List Nat := 10321 :: root393Tail23
def root393Tail21 : List Nat := 10141 :: root393Tail22
def root393Tail20 : List Nat := 9817 :: root393Tail21
def root393Tail19 : List Nat := 9613 :: root393Tail20
def root393Tail18 : List Nat := 9403 :: root393Tail19
def root393Tail17 : List Nat := 9397 :: root393Tail18
def root393Tail16 : List Nat := 9199 :: root393Tail17
def root393Tail15 : List Nat := 9187 :: root393Tail16
def root393Tail14 : List Nat := 8737 :: root393Tail15
def root393Tail13 : List Nat := 8389 :: root393Tail14
def root393Tail12 : List Nat := 8191 :: root393Tail13
def root393Tail11 : List Nat := 7213 :: root393Tail12
def root393Tail10 : List Nat := 7207 :: root393Tail11
def root393Tail9 : List Nat := 6469 :: root393Tail10
def root393Tail8 : List Nat := 6247 :: root393Tail9
def root393Tail7 : List Nat := 5869 :: root393Tail8
def root393Tail6 : List Nat := 5749 :: root393Tail7
def root393Tail5 : List Nat := 5443 :: root393Tail6
def root393Tail4 : List Nat := 3889 :: root393Tail5
def root393Tail3 : List Nat := 3739 :: root393Tail4
def root393Tail2 : List Nat := 3517 :: root393Tail3
def root393Tail1 : List Nat := 3433 :: root393Tail2
def root393Tail0 : List Nat := 2797 :: root393Tail1
def root393Candidates0 : List Nat := [7207, 12583, 17419]
set_option maxRecDepth 8192 in
theorem root393Candidates0_checked : candidates edge 4 2713 2797 root393Tail1 winner.bound = root393Candidates0 := by
  unfold candidates
  have row : edge 2797 = (SortedNatTree.contains FastRows.row2797) := by
    change Data.row2797.contains = (SortedNatTree.contains FastRows.row2797)
    exact FastRows.row2797_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root393Child0 : search edge 3 [2713, 2797] root393Candidates0 5510 winner = winner := by decide
theorem root393Step0 : scan edge (search edge 3) 4 [2713] 2713 root393Tail0 winner =
    scan edge (search edge 3) 4 [2713] 2713 root393Tail1 winner := by
  rw [root393Tail0, scan, if_neg (by decide), root393Candidates0_checked]
  change scan edge (search edge 3) 4 [2713] 2713 root393Tail1
    (if 3 ≤ root393Candidates0.length then search edge 3 [2713, 2797] root393Candidates0 5510 winner else winner) = _
  rw [if_pos (by decide), root393Child0]
def root393Candidates1 : List Nat := [7207]
set_option maxRecDepth 8192 in
theorem root393Candidates1_checked : candidates edge 4 2713 3433 root393Tail2 winner.bound = root393Candidates1 := by
  unfold candidates
  have row : edge 3433 = (SortedNatTree.contains FastRows.row3433) := by
    change Data.row3433.contains = (SortedNatTree.contains FastRows.row3433)
    exact FastRows.row3433_checked.symm
  rw [row]
  decide
theorem root393Step1 : scan edge (search edge 3) 4 [2713] 2713 root393Tail1 winner =
    scan edge (search edge 3) 4 [2713] 2713 root393Tail2 winner := by
  rw [root393Tail1, scan, if_neg (by decide), root393Candidates1_checked]
  change scan edge (search edge 3) 4 [2713] 2713 root393Tail2
    (if 3 ≤ root393Candidates1.length then search edge 3 [2713, 3433] root393Candidates1 6146 winner else winner) = _
  rw [if_neg (by decide)]
def root393Candidates2 : List Nat := [15817]
set_option maxRecDepth 8192 in
theorem root393Candidates2_checked : candidates edge 4 2713 3517 root393Tail3 winner.bound = root393Candidates2 := by
  unfold candidates
  have row : edge 3517 = (SortedNatTree.contains FastRows.row3517) := by
    change Data.row3517.contains = (SortedNatTree.contains FastRows.row3517)
    exact FastRows.row3517_checked.symm
  rw [row]
  decide
theorem root393Step2 : scan edge (search edge 3) 4 [2713] 2713 root393Tail2 winner =
    scan edge (search edge 3) 4 [2713] 2713 root393Tail3 winner := by
  rw [root393Tail2, scan, if_neg (by decide), root393Candidates2_checked]
  change scan edge (search edge 3) 4 [2713] 2713 root393Tail3
    (if 3 ≤ root393Candidates2.length then search edge 3 [2713, 3517] root393Candidates2 6230 winner else winner) = _
  rw [if_neg (by decide)]
def root393Candidates3 : List Nat := [13807]
set_option maxRecDepth 8192 in
theorem root393Candidates3_checked : candidates edge 4 2713 3739 root393Tail4 winner.bound = root393Candidates3 := by
  unfold candidates
  have row : edge 3739 = (SortedNatTree.contains FastRows.row3739) := by
    change Data.row3739.contains = (SortedNatTree.contains FastRows.row3739)
    exact FastRows.row3739_checked.symm
  rw [row]
  decide
theorem root393Step3 : scan edge (search edge 3) 4 [2713] 2713 root393Tail3 winner =
    scan edge (search edge 3) 4 [2713] 2713 root393Tail4 winner := by
  rw [root393Tail3, scan, if_neg (by decide), root393Candidates3_checked]
  change scan edge (search edge 3) 4 [2713] 2713 root393Tail4
    (if 3 ≤ root393Candidates3.length then search edge 3 [2713, 3739] root393Candidates3 6452 winner else winner) = _
  rw [if_neg (by decide)]
def root393Candidates4 : List Nat := [5749, 13807, 17257]
set_option maxRecDepth 8192 in
theorem root393Candidates4_checked : candidates edge 4 2713 3889 root393Tail5 winner.bound = root393Candidates4 := by
  unfold candidates
  have row : edge 3889 = (SortedNatTree.contains FastRows.row3889) := by
    change Data.row3889.contains = (SortedNatTree.contains FastRows.row3889)
    exact FastRows.row3889_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root393Child4 : search edge 3 [2713, 3889] root393Candidates4 6602 winner = winner := by decide
theorem root393Step4 : scan edge (search edge 3) 4 [2713] 2713 root393Tail4 winner =
    scan edge (search edge 3) 4 [2713] 2713 root393Tail5 winner := by
  rw [root393Tail4, scan, if_neg (by decide), root393Candidates4_checked]
  change scan edge (search edge 3) 4 [2713] 2713 root393Tail5
    (if 3 ≤ root393Candidates4.length then search edge 3 [2713, 3889] root393Candidates4 6602 winner else winner) = _
  rw [if_pos (by decide), root393Child4]
def root393Candidates5 : List Nat := [14431]
set_option maxRecDepth 8192 in
theorem root393Candidates5_checked : candidates edge 4 2713 5443 root393Tail6 winner.bound = root393Candidates5 := by
  unfold candidates
  have row : edge 5443 = (SortedNatTree.contains FastRows.row5443) := by
    change Data.row5443.contains = (SortedNatTree.contains FastRows.row5443)
    exact FastRows.row5443_checked.symm
  rw [row]
  decide
theorem root393Step5 : scan edge (search edge 3) 4 [2713] 2713 root393Tail5 winner =
    scan edge (search edge 3) 4 [2713] 2713 root393Tail6 winner := by
  rw [root393Tail5, scan, if_neg (by decide), root393Candidates5_checked]
  change scan edge (search edge 3) 4 [2713] 2713 root393Tail6
    (if 3 ≤ root393Candidates5.length then search edge 3 [2713, 5443] root393Candidates5 8156 winner else winner) = _
  rw [if_neg (by decide)]
def root393Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root393Candidates6_checked : candidates edge 4 2713 5749 root393Tail7 winner.bound = root393Candidates6 := by
  unfold candidates
  have row : edge 5749 = (SortedNatTree.contains FastRows.row5749) := by
    change Data.row5749.contains = (SortedNatTree.contains FastRows.row5749)
    exact FastRows.row5749_checked.symm
  rw [row]
  decide
theorem root393Step6 : scan edge (search edge 3) 4 [2713] 2713 root393Tail6 winner =
    scan edge (search edge 3) 4 [2713] 2713 root393Tail7 winner := by
  rw [root393Tail6, scan, if_neg (by decide), root393Candidates6_checked]
  change scan edge (search edge 3) 4 [2713] 2713 root393Tail7
    (if 3 ≤ root393Candidates6.length then search edge 3 [2713, 5749] root393Candidates6 8462 winner else winner) = _
  rw [if_neg (by decide)]
theorem root393Scan7 : scan edge (search edge 3) 4 [2713] 2713 root393Tail7 winner = winner := by rw [root393Tail7, scan, if_pos (by decide)]
theorem root393Scan6 : scan edge (search edge 3) 4 [2713] 2713 root393Tail6 winner = winner :=
  (root393Step6).trans root393Scan7
theorem root393Scan5 : scan edge (search edge 3) 4 [2713] 2713 root393Tail5 winner = winner :=
  (root393Step5).trans root393Scan6
theorem root393Scan4 : scan edge (search edge 3) 4 [2713] 2713 root393Tail4 winner = winner :=
  (root393Step4).trans root393Scan5
theorem root393Scan3 : scan edge (search edge 3) 4 [2713] 2713 root393Tail3 winner = winner :=
  (root393Step3).trans root393Scan4
theorem root393Scan2 : scan edge (search edge 3) 4 [2713] 2713 root393Tail2 winner = winner :=
  (root393Step2).trans root393Scan3
theorem root393Scan1 : scan edge (search edge 3) 4 [2713] 2713 root393Tail1 winner = winner :=
  (root393Step1).trans root393Scan2
theorem root393Scan0 : scan edge (search edge 3) 4 [2713] 2713 root393Tail0 winner = winner :=
  (root393Step0).trans root393Scan1
set_option maxRecDepth 4096 in
theorem root393Branch : search edge 4 [2713] root393Candidates 2713 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2713] 2713 root393Tail0 winner = winner
  exact root393Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
