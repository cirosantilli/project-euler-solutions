import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root574Candidates : List Nat := Data.row4217
set_option maxRecDepth 8192 in
theorem root574Candidates_checked : candidates edge 5 0 4217 tail575 winner.bound = root574Candidates := by
  apply candidates_eq_row edge 5 0 4217 tail575 root574Candidates winner.bound tail575_ordered
  · decide
  · decide
  · funext q
    rfl
def root574Tail31 : List Nat := []
def root574Tail30 : List Nat := 21401 :: root574Tail31
def root574Tail29 : List Nat := 20879 :: root574Tail30
def root574Tail28 : List Nat := 20771 :: root574Tail29
def root574Tail27 : List Nat := 20549 :: root574Tail28
def root574Tail26 : List Nat := 17909 :: root574Tail27
def root574Tail25 : List Nat := 15641 :: root574Tail26
def root574Tail24 : List Nat := 15467 :: root574Tail25
def root574Tail23 : List Nat := 14537 :: root574Tail24
def root574Tail22 : List Nat := 14327 :: root574Tail23
def root574Tail21 : List Nat := 12479 :: root574Tail22
def root574Tail20 : List Nat := 12413 :: root574Tail21
def root574Tail19 : List Nat := 11597 :: root574Tail20
def root574Tail18 : List Nat := 11489 :: root574Tail19
def root574Tail17 : List Nat := 11393 :: root574Tail18
def root574Tail16 : List Nat := 11261 :: root574Tail17
def root574Tail15 : List Nat := 11177 :: root574Tail16
def root574Tail14 : List Nat := 9743 :: root574Tail15
def root574Tail13 : List Nat := 9719 :: root574Tail14
def root574Tail12 : List Nat := 9629 :: root574Tail13
def root574Tail11 : List Nat := 9377 :: root574Tail12
def root574Tail10 : List Nat := 9209 :: root574Tail11
def root574Tail9 : List Nat := 8363 :: root574Tail10
def root574Tail8 : List Nat := 8111 :: root574Tail9
def root574Tail7 : List Nat := 7829 :: root574Tail8
def root574Tail6 : List Nat := 7823 :: root574Tail7
def root574Tail5 : List Nat := 5981 :: root574Tail6
def root574Tail4 : List Nat := 5171 :: root574Tail5
def root574Tail3 : List Nat := 4751 :: root574Tail4
def root574Tail2 : List Nat := 4673 :: root574Tail3
def root574Tail1 : List Nat := 4421 :: root574Tail2
def root574Tail0 : List Nat := 4337 :: root574Tail1
def root574Candidates0 : List Nat := [4421]
set_option maxRecDepth 8192 in
theorem root574Candidates0_checked : candidates edge 4 4217 4337 root574Tail1 winner.bound = root574Candidates0 := by
  unfold candidates
  have row : edge 4337 = (SortedNatTree.contains FastRows.row4337) := by
    change Data.row4337.contains = (SortedNatTree.contains FastRows.row4337)
    exact FastRows.row4337_checked.symm
  rw [row]
  decide
theorem root574Step0 : scan edge (search edge 3) 4 [4217] 4217 root574Tail0 winner =
    scan edge (search edge 3) 4 [4217] 4217 root574Tail1 winner := by
  rw [root574Tail0, scan, if_neg (by decide), root574Candidates0_checked]
  change scan edge (search edge 3) 4 [4217] 4217 root574Tail1
    (if 3 ≤ root574Candidates0.length then search edge 3 [4217, 4337] root574Candidates0 8554 winner else winner) = _
  rw [if_neg (by decide)]
def root574Candidates1 : List Nat := [11393, 11597]
set_option maxRecDepth 8192 in
theorem root574Candidates1_checked : candidates edge 4 4217 4421 root574Tail2 winner.bound = root574Candidates1 := by
  unfold candidates
  have row : edge 4421 = (SortedNatTree.contains FastRows.row4421) := by
    change Data.row4421.contains = (SortedNatTree.contains FastRows.row4421)
    exact FastRows.row4421_checked.symm
  rw [row]
  decide
theorem root574Step1 : scan edge (search edge 3) 4 [4217] 4217 root574Tail1 winner =
    scan edge (search edge 3) 4 [4217] 4217 root574Tail2 winner := by
  rw [root574Tail1, scan, if_neg (by decide), root574Candidates1_checked]
  change scan edge (search edge 3) 4 [4217] 4217 root574Tail2
    (if 3 ≤ root574Candidates1.length then search edge 3 [4217, 4421] root574Candidates1 8638 winner else winner) = _
  rw [if_neg (by decide)]
def root574Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root574Candidates2_checked : candidates edge 4 4217 4673 root574Tail3 winner.bound = root574Candidates2 := by
  unfold candidates
  have row : edge 4673 = (SortedNatTree.contains FastRows.row4673) := by
    change Data.row4673.contains = (SortedNatTree.contains FastRows.row4673)
    exact FastRows.row4673_checked.symm
  rw [row]
  decide
theorem root574Step2 : scan edge (search edge 3) 4 [4217] 4217 root574Tail2 winner =
    scan edge (search edge 3) 4 [4217] 4217 root574Tail3 winner := by
  rw [root574Tail2, scan, if_neg (by decide), root574Candidates2_checked]
  change scan edge (search edge 3) 4 [4217] 4217 root574Tail3
    (if 3 ≤ root574Candidates2.length then search edge 3 [4217, 4673] root574Candidates2 8890 winner else winner) = _
  rw [if_neg (by decide)]
def root574Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root574Candidates3_checked : candidates edge 4 4217 4751 root574Tail4 winner.bound = root574Candidates3 := by
  unfold candidates
  have row : edge 4751 = (SortedNatTree.contains FastRows.row4751) := by
    change Data.row4751.contains = (SortedNatTree.contains FastRows.row4751)
    exact FastRows.row4751_checked.symm
  rw [row]
  decide
theorem root574Step3 : scan edge (search edge 3) 4 [4217] 4217 root574Tail3 winner =
    scan edge (search edge 3) 4 [4217] 4217 root574Tail4 winner := by
  rw [root574Tail3, scan, if_neg (by decide), root574Candidates3_checked]
  change scan edge (search edge 3) 4 [4217] 4217 root574Tail4
    (if 3 ≤ root574Candidates3.length then search edge 3 [4217, 4751] root574Candidates3 8968 winner else winner) = _
  rw [if_neg (by decide)]
def root574Candidates4 : List Nat := [8111, 14537]
set_option maxRecDepth 8192 in
theorem root574Candidates4_checked : candidates edge 4 4217 5171 root574Tail5 winner.bound = root574Candidates4 := by
  unfold candidates
  have row : edge 5171 = (SortedNatTree.contains FastRows.row5171) := by
    change Data.row5171.contains = (SortedNatTree.contains FastRows.row5171)
    exact FastRows.row5171_checked.symm
  rw [row]
  decide
theorem root574Step4 : scan edge (search edge 3) 4 [4217] 4217 root574Tail4 winner =
    scan edge (search edge 3) 4 [4217] 4217 root574Tail5 winner := by
  rw [root574Tail4, scan, if_neg (by decide), root574Candidates4_checked]
  change scan edge (search edge 3) 4 [4217] 4217 root574Tail5
    (if 3 ≤ root574Candidates4.length then search edge 3 [4217, 5171] root574Candidates4 9388 winner else winner) = _
  rw [if_neg (by decide)]
theorem root574Scan5 : scan edge (search edge 3) 4 [4217] 4217 root574Tail5 winner = winner := by rw [root574Tail5, scan, if_pos (by decide)]
theorem root574Scan4 : scan edge (search edge 3) 4 [4217] 4217 root574Tail4 winner = winner :=
  (root574Step4).trans root574Scan5
theorem root574Scan3 : scan edge (search edge 3) 4 [4217] 4217 root574Tail3 winner = winner :=
  (root574Step3).trans root574Scan4
theorem root574Scan2 : scan edge (search edge 3) 4 [4217] 4217 root574Tail2 winner = winner :=
  (root574Step2).trans root574Scan3
theorem root574Scan1 : scan edge (search edge 3) 4 [4217] 4217 root574Tail1 winner = winner :=
  (root574Step1).trans root574Scan2
theorem root574Scan0 : scan edge (search edge 3) 4 [4217] 4217 root574Tail0 winner = winner :=
  (root574Step0).trans root574Scan1
set_option maxRecDepth 4096 in
theorem root574Branch : search edge 4 [4217] root574Candidates 4217 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4217] 4217 root574Tail0 winner = winner
  exact root574Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
