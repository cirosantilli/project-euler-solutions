import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root485Candidates : List Nat := Data.row3491
set_option maxRecDepth 8192 in
theorem root485Candidates_checked : candidates edge 5 0 3491 tail486 winner.bound = root485Candidates := by
  apply candidates_eq_row edge 5 0 3491 tail486 root485Candidates winner.bound tail486_ordered
  · decide
  · decide
  · funext q
    rfl
def root485Tail35 : List Nat := []
def root485Tail34 : List Nat := 21863 :: root485Tail35
def root485Tail33 : List Nat := 21317 :: root485Tail34
def root485Tail32 : List Nat := 19949 :: root485Tail33
def root485Tail31 : List Nat := 19559 :: root485Tail32
def root485Tail30 : List Nat := 18899 :: root485Tail31
def root485Tail29 : List Nat := 18593 :: root485Tail30
def root485Tail28 : List Nat := 18257 :: root485Tail29
def root485Tail27 : List Nat := 18089 :: root485Tail28
def root485Tail26 : List Nat := 17471 :: root485Tail27
def root485Tail25 : List Nat := 17333 :: root485Tail26
def root485Tail24 : List Nat := 17159 :: root485Tail25
def root485Tail23 : List Nat := 17033 :: root485Tail24
def root485Tail22 : List Nat := 16319 :: root485Tail23
def root485Tail21 : List Nat := 16073 :: root485Tail22
def root485Tail20 : List Nat := 15959 :: root485Tail21
def root485Tail19 : List Nat := 15161 :: root485Tail20
def root485Tail18 : List Nat := 14009 :: root485Tail19
def root485Tail17 : List Nat := 13781 :: root485Tail18
def root485Tail16 : List Nat := 11069 :: root485Tail17
def root485Tail15 : List Nat := 10631 :: root485Tail16
def root485Tail14 : List Nat := 9281 :: root485Tail15
def root485Tail13 : List Nat := 8123 :: root485Tail14
def root485Tail12 : List Nat := 8111 :: root485Tail13
def root485Tail11 : List Nat := 7331 :: root485Tail12
def root485Tail10 : List Nat := 7121 :: root485Tail11
def root485Tail9 : List Nat := 6701 :: root485Tail10
def root485Tail8 : List Nat := 6221 :: root485Tail9
def root485Tail7 : List Nat := 5843 :: root485Tail8
def root485Tail6 : List Nat := 5171 :: root485Tail7
def root485Tail5 : List Nat := 5003 :: root485Tail6
def root485Tail4 : List Nat := 4931 :: root485Tail5
def root485Tail3 : List Nat := 4349 :: root485Tail4
def root485Tail2 : List Nat := 4259 :: root485Tail3
def root485Tail1 : List Nat := 3947 :: root485Tail2
def root485Tail0 : List Nat := 3761 :: root485Tail1
def root485Candidates0 : List Nat := [5843]
set_option maxRecDepth 8192 in
theorem root485Candidates0_checked : candidates edge 4 3491 3761 root485Tail1 winner.bound = root485Candidates0 := by
  unfold candidates
  have row : edge 3761 = (SortedNatTree.contains FastRows.row3761) := by
    change Data.row3761.contains = (SortedNatTree.contains FastRows.row3761)
    exact FastRows.row3761_checked.symm
  rw [row]
  decide
theorem root485Step0 : scan edge (search edge 3) 4 [3491] 3491 root485Tail0 winner =
    scan edge (search edge 3) 4 [3491] 3491 root485Tail1 winner := by
  rw [root485Tail0, scan, if_neg (by decide), root485Candidates0_checked]
  change scan edge (search edge 3) 4 [3491] 3491 root485Tail1
    (if 3 ≤ root485Candidates0.length then search edge 3 [3491, 3761] root485Candidates0 7252 winner else winner) = _
  rw [if_neg (by decide)]
def root485Candidates1 : List Nat := [5171, 17333]
set_option maxRecDepth 8192 in
theorem root485Candidates1_checked : candidates edge 4 3491 3947 root485Tail2 winner.bound = root485Candidates1 := by
  unfold candidates
  have row : edge 3947 = (SortedNatTree.contains FastRows.row3947) := by
    change Data.row3947.contains = (SortedNatTree.contains FastRows.row3947)
    exact FastRows.row3947_checked.symm
  rw [row]
  decide
theorem root485Step1 : scan edge (search edge 3) 4 [3491] 3491 root485Tail1 winner =
    scan edge (search edge 3) 4 [3491] 3491 root485Tail2 winner := by
  rw [root485Tail1, scan, if_neg (by decide), root485Candidates1_checked]
  change scan edge (search edge 3) 4 [3491] 3491 root485Tail2
    (if 3 ≤ root485Candidates1.length then search edge 3 [3491, 3947] root485Candidates1 7438 winner else winner) = _
  rw [if_neg (by decide)]
def root485Candidates2 : List Nat := [6221]
set_option maxRecDepth 8192 in
theorem root485Candidates2_checked : candidates edge 4 3491 4259 root485Tail3 winner.bound = root485Candidates2 := by
  unfold candidates
  have row : edge 4259 = (SortedNatTree.contains FastRows.row4259) := by
    change Data.row4259.contains = (SortedNatTree.contains FastRows.row4259)
    exact FastRows.row4259_checked.symm
  rw [row]
  decide
theorem root485Step2 : scan edge (search edge 3) 4 [3491] 3491 root485Tail2 winner =
    scan edge (search edge 3) 4 [3491] 3491 root485Tail3 winner := by
  rw [root485Tail2, scan, if_neg (by decide), root485Candidates2_checked]
  change scan edge (search edge 3) 4 [3491] 3491 root485Tail3
    (if 3 ≤ root485Candidates2.length then search edge 3 [3491, 4259] root485Candidates2 7750 winner else winner) = _
  rw [if_neg (by decide)]
def root485Candidates3 : List Nat := [17471]
set_option maxRecDepth 8192 in
theorem root485Candidates3_checked : candidates edge 4 3491 4349 root485Tail4 winner.bound = root485Candidates3 := by
  unfold candidates
  have row : edge 4349 = (SortedNatTree.contains FastRows.row4349) := by
    change Data.row4349.contains = (SortedNatTree.contains FastRows.row4349)
    exact FastRows.row4349_checked.symm
  rw [row]
  decide
theorem root485Step3 : scan edge (search edge 3) 4 [3491] 3491 root485Tail3 winner =
    scan edge (search edge 3) 4 [3491] 3491 root485Tail4 winner := by
  rw [root485Tail3, scan, if_neg (by decide), root485Candidates3_checked]
  change scan edge (search edge 3) 4 [3491] 3491 root485Tail4
    (if 3 ≤ root485Candidates3.length then search edge 3 [3491, 4349] root485Candidates3 7840 winner else winner) = _
  rw [if_neg (by decide)]
def root485Candidates4 : List Nat := [5843]
set_option maxRecDepth 8192 in
theorem root485Candidates4_checked : candidates edge 4 3491 4931 root485Tail5 winner.bound = root485Candidates4 := by
  unfold candidates
  have row : edge 4931 = (SortedNatTree.contains FastRows.row4931) := by
    change Data.row4931.contains = (SortedNatTree.contains FastRows.row4931)
    exact FastRows.row4931_checked.symm
  rw [row]
  decide
theorem root485Step4 : scan edge (search edge 3) 4 [3491] 3491 root485Tail4 winner =
    scan edge (search edge 3) 4 [3491] 3491 root485Tail5 winner := by
  rw [root485Tail4, scan, if_neg (by decide), root485Candidates4_checked]
  change scan edge (search edge 3) 4 [3491] 3491 root485Tail5
    (if 3 ≤ root485Candidates4.length then search edge 3 [3491, 4931] root485Candidates4 8422 winner else winner) = _
  rw [if_neg (by decide)]
def root485Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root485Candidates5_checked : candidates edge 4 3491 5003 root485Tail6 winner.bound = root485Candidates5 := by
  unfold candidates
  have row : edge 5003 = (SortedNatTree.contains FastRows.row5003) := by
    change Data.row5003.contains = (SortedNatTree.contains FastRows.row5003)
    exact FastRows.row5003_checked.symm
  rw [row]
  decide
theorem root485Step5 : scan edge (search edge 3) 4 [3491] 3491 root485Tail5 winner =
    scan edge (search edge 3) 4 [3491] 3491 root485Tail6 winner := by
  rw [root485Tail5, scan, if_neg (by decide), root485Candidates5_checked]
  change scan edge (search edge 3) 4 [3491] 3491 root485Tail6
    (if 3 ≤ root485Candidates5.length then search edge 3 [3491, 5003] root485Candidates5 8494 winner else winner) = _
  rw [if_neg (by decide)]
def root485Candidates6 : List Nat := [8111]
set_option maxRecDepth 8192 in
theorem root485Candidates6_checked : candidates edge 4 3491 5171 root485Tail7 winner.bound = root485Candidates6 := by
  unfold candidates
  have row : edge 5171 = (SortedNatTree.contains FastRows.row5171) := by
    change Data.row5171.contains = (SortedNatTree.contains FastRows.row5171)
    exact FastRows.row5171_checked.symm
  rw [row]
  decide
theorem root485Step6 : scan edge (search edge 3) 4 [3491] 3491 root485Tail6 winner =
    scan edge (search edge 3) 4 [3491] 3491 root485Tail7 winner := by
  rw [root485Tail6, scan, if_neg (by decide), root485Candidates6_checked]
  change scan edge (search edge 3) 4 [3491] 3491 root485Tail7
    (if 3 ≤ root485Candidates6.length then search edge 3 [3491, 5171] root485Candidates6 8662 winner else winner) = _
  rw [if_neg (by decide)]
theorem root485Scan7 : scan edge (search edge 3) 4 [3491] 3491 root485Tail7 winner = winner := by rw [root485Tail7, scan, if_pos (by decide)]
theorem root485Scan6 : scan edge (search edge 3) 4 [3491] 3491 root485Tail6 winner = winner :=
  (root485Step6).trans root485Scan7
theorem root485Scan5 : scan edge (search edge 3) 4 [3491] 3491 root485Tail5 winner = winner :=
  (root485Step5).trans root485Scan6
theorem root485Scan4 : scan edge (search edge 3) 4 [3491] 3491 root485Tail4 winner = winner :=
  (root485Step4).trans root485Scan5
theorem root485Scan3 : scan edge (search edge 3) 4 [3491] 3491 root485Tail3 winner = winner :=
  (root485Step3).trans root485Scan4
theorem root485Scan2 : scan edge (search edge 3) 4 [3491] 3491 root485Tail2 winner = winner :=
  (root485Step2).trans root485Scan3
theorem root485Scan1 : scan edge (search edge 3) 4 [3491] 3491 root485Tail1 winner = winner :=
  (root485Step1).trans root485Scan2
theorem root485Scan0 : scan edge (search edge 3) 4 [3491] 3491 root485Tail0 winner = winner :=
  (root485Step0).trans root485Scan1
set_option maxRecDepth 4096 in
theorem root485Branch : search edge 4 [3491] root485Candidates 3491 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3491] 3491 root485Tail0 winner = winner
  exact root485Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
