import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root426Candidates : List Nat := Data.row2971
set_option maxRecDepth 8192 in
theorem root426Candidates_checked : candidates edge 5 0 2971 tail427 winner.bound = root426Candidates := by
  apply candidates_eq_row edge 5 0 2971 tail427 root426Candidates winner.bound tail427_ordered
  · decide
  · decide
  · funext q
    rfl
def root426Tail35 : List Nat := []
def root426Tail34 : List Nat := 22963 :: root426Tail35
def root426Tail33 : List Nat := 22129 :: root426Tail34
def root426Tail32 : List Nat := 21859 :: root426Tail33
def root426Tail31 : List Nat := 21481 :: root426Tail32
def root426Tail30 : List Nat := 20353 :: root426Tail31
def root426Tail29 : List Nat := 20011 :: root426Tail30
def root426Tail28 : List Nat := 18121 :: root426Tail29
def root426Tail27 : List Nat := 18049 :: root426Tail28
def root426Tail26 : List Nat := 17713 :: root426Tail27
def root426Tail25 : List Nat := 17419 :: root426Tail26
def root426Tail24 : List Nat := 17257 :: root426Tail25
def root426Tail23 : List Nat := 15649 :: root426Tail24
def root426Tail22 : List Nat := 15271 :: root426Tail23
def root426Tail21 : List Nat := 15061 :: root426Tail22
def root426Tail20 : List Nat := 14221 :: root426Tail21
def root426Tail19 : List Nat := 13591 :: root426Tail20
def root426Tail18 : List Nat := 13477 :: root426Tail19
def root426Tail17 : List Nat := 12589 :: root426Tail18
def root426Tail16 : List Nat := 12421 :: root426Tail17
def root426Tail15 : List Nat := 9649 :: root426Tail16
def root426Tail14 : List Nat := 9511 :: root426Tail15
def root426Tail13 : List Nat := 9343 :: root426Tail14
def root426Tail12 : List Nat := 9337 :: root426Tail13
def root426Tail11 : List Nat := 7573 :: root426Tail12
def root426Tail10 : List Nat := 7459 :: root426Tail11
def root426Tail9 : List Nat := 6961 :: root426Tail10
def root426Tail8 : List Nat := 6871 :: root426Tail9
def root426Tail7 : List Nat := 6067 :: root426Tail8
def root426Tail6 : List Nat := 5953 :: root426Tail7
def root426Tail5 : List Nat := 5557 :: root426Tail6
def root426Tail4 : List Nat := 4639 :: root426Tail5
def root426Tail3 : List Nat := 4273 :: root426Tail4
def root426Tail2 : List Nat := 4057 :: root426Tail3
def root426Tail1 : List Nat := 3709 :: root426Tail2
def root426Tail0 : List Nat := 3037 :: root426Tail1
def root426Candidates0 : List Nat := [3709]
set_option maxRecDepth 8192 in
theorem root426Candidates0_checked : candidates edge 4 2971 3037 root426Tail1 winner.bound = root426Candidates0 := by
  unfold candidates
  have row : edge 3037 = (SortedNatTree.contains FastRows.row3037) := by
    change Data.row3037.contains = (SortedNatTree.contains FastRows.row3037)
    exact FastRows.row3037_checked.symm
  rw [row]
  decide
theorem root426Step0 : scan edge (search edge 3) 4 [2971] 2971 root426Tail0 winner =
    scan edge (search edge 3) 4 [2971] 2971 root426Tail1 winner := by
  rw [root426Tail0, scan, if_neg (by decide), root426Candidates0_checked]
  change scan edge (search edge 3) 4 [2971] 2971 root426Tail1
    (if 3 ≤ root426Candidates0.length then search edge 3 [2971, 3037] root426Candidates0 6008 winner else winner) = _
  rw [if_neg (by decide)]
def root426Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root426Candidates1_checked : candidates edge 4 2971 3709 root426Tail2 winner.bound = root426Candidates1 := by
  unfold candidates
  have row : edge 3709 = (SortedNatTree.contains FastRows.row3709) := by
    change Data.row3709.contains = (SortedNatTree.contains FastRows.row3709)
    exact FastRows.row3709_checked.symm
  rw [row]
  decide
theorem root426Step1 : scan edge (search edge 3) 4 [2971] 2971 root426Tail1 winner =
    scan edge (search edge 3) 4 [2971] 2971 root426Tail2 winner := by
  rw [root426Tail1, scan, if_neg (by decide), root426Candidates1_checked]
  change scan edge (search edge 3) 4 [2971] 2971 root426Tail2
    (if 3 ≤ root426Candidates1.length then search edge 3 [2971, 3709] root426Candidates1 6680 winner else winner) = _
  rw [if_neg (by decide)]
def root426Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root426Candidates2_checked : candidates edge 4 2971 4057 root426Tail3 winner.bound = root426Candidates2 := by
  unfold candidates
  have row : edge 4057 = (SortedNatTree.contains FastRows.row4057) := by
    change Data.row4057.contains = (SortedNatTree.contains FastRows.row4057)
    exact FastRows.row4057_checked.symm
  rw [row]
  decide
theorem root426Step2 : scan edge (search edge 3) 4 [2971] 2971 root426Tail2 winner =
    scan edge (search edge 3) 4 [2971] 2971 root426Tail3 winner := by
  rw [root426Tail2, scan, if_neg (by decide), root426Candidates2_checked]
  change scan edge (search edge 3) 4 [2971] 2971 root426Tail3
    (if 3 ≤ root426Candidates2.length then search edge 3 [2971, 4057] root426Candidates2 7028 winner else winner) = _
  rw [if_neg (by decide)]
def root426Candidates3 : List Nat := [7573, 12421]
set_option maxRecDepth 8192 in
theorem root426Candidates3_checked : candidates edge 4 2971 4273 root426Tail4 winner.bound = root426Candidates3 := by
  unfold candidates
  have row : edge 4273 = (SortedNatTree.contains FastRows.row4273) := by
    change Data.row4273.contains = (SortedNatTree.contains FastRows.row4273)
    exact FastRows.row4273_checked.symm
  rw [row]
  decide
theorem root426Step3 : scan edge (search edge 3) 4 [2971] 2971 root426Tail3 winner =
    scan edge (search edge 3) 4 [2971] 2971 root426Tail4 winner := by
  rw [root426Tail3, scan, if_neg (by decide), root426Candidates3_checked]
  change scan edge (search edge 3) 4 [2971] 2971 root426Tail4
    (if 3 ≤ root426Candidates3.length then search edge 3 [2971, 4273] root426Candidates3 7244 winner else winner) = _
  rw [if_neg (by decide)]
def root426Candidates4 : List Nat := [17257]
set_option maxRecDepth 8192 in
theorem root426Candidates4_checked : candidates edge 4 2971 4639 root426Tail5 winner.bound = root426Candidates4 := by
  unfold candidates
  have row : edge 4639 = (SortedNatTree.contains FastRows.row4639) := by
    change Data.row4639.contains = (SortedNatTree.contains FastRows.row4639)
    exact FastRows.row4639_checked.symm
  rw [row]
  decide
theorem root426Step4 : scan edge (search edge 3) 4 [2971] 2971 root426Tail4 winner =
    scan edge (search edge 3) 4 [2971] 2971 root426Tail5 winner := by
  rw [root426Tail4, scan, if_neg (by decide), root426Candidates4_checked]
  change scan edge (search edge 3) 4 [2971] 2971 root426Tail5
    (if 3 ≤ root426Candidates4.length then search edge 3 [2971, 4639] root426Candidates4 7610 winner else winner) = _
  rw [if_neg (by decide)]
def root426Candidates5 : List Nat := [6067]
set_option maxRecDepth 8192 in
theorem root426Candidates5_checked : candidates edge 4 2971 5557 root426Tail6 winner.bound = root426Candidates5 := by
  unfold candidates
  have row : edge 5557 = (SortedNatTree.contains FastRows.row5557) := by
    change Data.row5557.contains = (SortedNatTree.contains FastRows.row5557)
    exact FastRows.row5557_checked.symm
  rw [row]
  decide
theorem root426Step5 : scan edge (search edge 3) 4 [2971] 2971 root426Tail5 winner =
    scan edge (search edge 3) 4 [2971] 2971 root426Tail6 winner := by
  rw [root426Tail5, scan, if_neg (by decide), root426Candidates5_checked]
  change scan edge (search edge 3) 4 [2971] 2971 root426Tail6
    (if 3 ≤ root426Candidates5.length then search edge 3 [2971, 5557] root426Candidates5 8528 winner else winner) = _
  rw [if_neg (by decide)]
theorem root426Scan6 : scan edge (search edge 3) 4 [2971] 2971 root426Tail6 winner = winner := by rw [root426Tail6, scan, if_pos (by decide)]
theorem root426Scan5 : scan edge (search edge 3) 4 [2971] 2971 root426Tail5 winner = winner :=
  (root426Step5).trans root426Scan6
theorem root426Scan4 : scan edge (search edge 3) 4 [2971] 2971 root426Tail4 winner = winner :=
  (root426Step4).trans root426Scan5
theorem root426Scan3 : scan edge (search edge 3) 4 [2971] 2971 root426Tail3 winner = winner :=
  (root426Step3).trans root426Scan4
theorem root426Scan2 : scan edge (search edge 3) 4 [2971] 2971 root426Tail2 winner = winner :=
  (root426Step2).trans root426Scan3
theorem root426Scan1 : scan edge (search edge 3) 4 [2971] 2971 root426Tail1 winner = winner :=
  (root426Step1).trans root426Scan2
theorem root426Scan0 : scan edge (search edge 3) 4 [2971] 2971 root426Tail0 winner = winner :=
  (root426Step0).trans root426Scan1
set_option maxRecDepth 4096 in
theorem root426Branch : search edge 4 [2971] root426Candidates 2971 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2971] 2971 root426Tail0 winner = winner
  exact root426Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
