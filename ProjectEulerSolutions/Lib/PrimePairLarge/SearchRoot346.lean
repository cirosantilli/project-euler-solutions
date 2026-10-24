import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root346Candidates : List Nat := Data.row2351
set_option maxRecDepth 8192 in
theorem root346Candidates_checked : candidates edge 5 0 2351 tail347 winner.bound = root346Candidates := by
  apply candidates_eq_row edge 5 0 2351 tail347 root346Candidates winner.bound tail347_ordered
  · decide
  · decide
  · funext q
    rfl
def root346Tail40 : List Nat := []
def root346Tail39 : List Nat := 23117 :: root346Tail40
def root346Tail38 : List Nat := 22571 :: root346Tail39
def root346Tail37 : List Nat := 21881 :: root346Tail38
def root346Tail36 : List Nat := 20147 :: root346Tail37
def root346Tail35 : List Nat := 19853 :: root346Tail36
def root346Tail34 : List Nat := 19463 :: root346Tail35
def root346Tail33 : List Nat := 19457 :: root346Tail34
def root346Tail32 : List Nat := 18287 :: root346Tail33
def root346Tail31 : List Nat := 17027 :: root346Tail32
def root346Tail30 : List Nat := 16421 :: root346Tail31
def root346Tail29 : List Nat := 14549 :: root346Tail30
def root346Tail28 : List Nat := 14489 :: root346Tail29
def root346Tail27 : List Nat := 14423 :: root346Tail28
def root346Tail26 : List Nat := 13679 :: root346Tail27
def root346Tail25 : List Nat := 13649 :: root346Tail26
def root346Tail24 : List Nat := 13397 :: root346Tail25
def root346Tail23 : List Nat := 12503 :: root346Tail24
def root346Tail22 : List Nat := 12197 :: root346Tail23
def root346Tail21 : List Nat := 11621 :: root346Tail22
def root346Tail20 : List Nat := 11171 :: root346Tail21
def root346Tail19 : List Nat := 10529 :: root346Tail20
def root346Tail18 : List Nat := 10331 :: root346Tail19
def root346Tail17 : List Nat := 9491 :: root346Tail18
def root346Tail16 : List Nat := 9341 :: root346Tail17
def root346Tail15 : List Nat := 9059 :: root346Tail16
def root346Tail14 : List Nat := 8609 :: root346Tail15
def root346Tail13 : List Nat := 8237 :: root346Tail14
def root346Tail12 : List Nat := 7853 :: root346Tail13
def root346Tail11 : List Nat := 7841 :: root346Tail12
def root346Tail10 : List Nat := 7643 :: root346Tail11
def root346Tail9 : List Nat := 7229 :: root346Tail10
def root346Tail8 : List Nat := 7019 :: root346Tail9
def root346Tail7 : List Nat := 6959 :: root346Tail8
def root346Tail6 : List Nat := 6761 :: root346Tail7
def root346Tail5 : List Nat := 5519 :: root346Tail6
def root346Tail4 : List Nat := 5507 :: root346Tail5
def root346Tail3 : List Nat := 4973 :: root346Tail4
def root346Tail2 : List Nat := 3881 :: root346Tail3
def root346Tail1 : List Nat := 3863 :: root346Tail2
def root346Tail0 : List Nat := 3617 :: root346Tail1
def root346Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root346Candidates0_checked : candidates edge 4 2351 3617 root346Tail1 winner.bound = root346Candidates0 := by
  unfold candidates
  have row : edge 3617 = (SortedNatTree.contains FastRows.row3617) := by
    change Data.row3617.contains = (SortedNatTree.contains FastRows.row3617)
    exact FastRows.row3617_checked.symm
  rw [row]
  decide
theorem root346Step0 : scan edge (search edge 3) 4 [2351] 2351 root346Tail0 winner =
    scan edge (search edge 3) 4 [2351] 2351 root346Tail1 winner := by
  rw [root346Tail0, scan, if_neg (by decide), root346Candidates0_checked]
  change scan edge (search edge 3) 4 [2351] 2351 root346Tail1
    (if 3 ≤ root346Candidates0.length then search edge 3 [2351, 3617] root346Candidates0 5968 winner else winner) = _
  rw [if_neg (by decide)]
def root346Candidates1 : List Nat := [19463]
set_option maxRecDepth 8192 in
theorem root346Candidates1_checked : candidates edge 4 2351 3863 root346Tail2 winner.bound = root346Candidates1 := by
  unfold candidates
  have row : edge 3863 = (SortedNatTree.contains FastRows.row3863) := by
    change Data.row3863.contains = (SortedNatTree.contains FastRows.row3863)
    exact FastRows.row3863_checked.symm
  rw [row]
  decide
theorem root346Step1 : scan edge (search edge 3) 4 [2351] 2351 root346Tail1 winner =
    scan edge (search edge 3) 4 [2351] 2351 root346Tail2 winner := by
  rw [root346Tail1, scan, if_neg (by decide), root346Candidates1_checked]
  change scan edge (search edge 3) 4 [2351] 2351 root346Tail2
    (if 3 ≤ root346Candidates1.length then search edge 3 [2351, 3863] root346Candidates1 6214 winner else winner) = _
  rw [if_neg (by decide)]
def root346Candidates2 : List Nat := [5519, 8609]
set_option maxRecDepth 8192 in
theorem root346Candidates2_checked : candidates edge 4 2351 3881 root346Tail3 winner.bound = root346Candidates2 := by
  unfold candidates
  have row : edge 3881 = (SortedNatTree.contains FastRows.row3881) := by
    change Data.row3881.contains = (SortedNatTree.contains FastRows.row3881)
    exact FastRows.row3881_checked.symm
  rw [row]
  decide
theorem root346Step2 : scan edge (search edge 3) 4 [2351] 2351 root346Tail2 winner =
    scan edge (search edge 3) 4 [2351] 2351 root346Tail3 winner := by
  rw [root346Tail2, scan, if_neg (by decide), root346Candidates2_checked]
  change scan edge (search edge 3) 4 [2351] 2351 root346Tail3
    (if 3 ≤ root346Candidates2.length then search edge 3 [2351, 3881] root346Candidates2 6232 winner else winner) = _
  rw [if_neg (by decide)]
def root346Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root346Candidates3_checked : candidates edge 4 2351 4973 root346Tail4 winner.bound = root346Candidates3 := by
  unfold candidates
  have row : edge 4973 = (SortedNatTree.contains FastRows.row4973) := by
    change Data.row4973.contains = (SortedNatTree.contains FastRows.row4973)
    exact FastRows.row4973_checked.symm
  rw [row]
  decide
theorem root346Step3 : scan edge (search edge 3) 4 [2351] 2351 root346Tail3 winner =
    scan edge (search edge 3) 4 [2351] 2351 root346Tail4 winner := by
  rw [root346Tail3, scan, if_neg (by decide), root346Candidates3_checked]
  change scan edge (search edge 3) 4 [2351] 2351 root346Tail4
    (if 3 ≤ root346Candidates3.length then search edge 3 [2351, 4973] root346Candidates3 7324 winner else winner) = _
  rw [if_neg (by decide)]
def root346Candidates4 : List Nat := [7643, 7853]
set_option maxRecDepth 8192 in
theorem root346Candidates4_checked : candidates edge 4 2351 5507 root346Tail5 winner.bound = root346Candidates4 := by
  unfold candidates
  have row : edge 5507 = (SortedNatTree.contains FastRows.row5507) := by
    change Data.row5507.contains = (SortedNatTree.contains FastRows.row5507)
    exact FastRows.row5507_checked.symm
  rw [row]
  decide
theorem root346Step4 : scan edge (search edge 3) 4 [2351] 2351 root346Tail4 winner =
    scan edge (search edge 3) 4 [2351] 2351 root346Tail5 winner := by
  rw [root346Tail4, scan, if_neg (by decide), root346Candidates4_checked]
  change scan edge (search edge 3) 4 [2351] 2351 root346Tail5
    (if 3 ≤ root346Candidates4.length then search edge 3 [2351, 5507] root346Candidates4 7858 winner else winner) = _
  rw [if_neg (by decide)]
def root346Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root346Candidates5_checked : candidates edge 4 2351 5519 root346Tail6 winner.bound = root346Candidates5 := by
  unfold candidates
  have row : edge 5519 = (SortedNatTree.contains FastRows.row5519) := by
    change Data.row5519.contains = (SortedNatTree.contains FastRows.row5519)
    exact FastRows.row5519_checked.symm
  rw [row]
  decide
theorem root346Step5 : scan edge (search edge 3) 4 [2351] 2351 root346Tail5 winner =
    scan edge (search edge 3) 4 [2351] 2351 root346Tail6 winner := by
  rw [root346Tail5, scan, if_neg (by decide), root346Candidates5_checked]
  change scan edge (search edge 3) 4 [2351] 2351 root346Tail6
    (if 3 ≤ root346Candidates5.length then search edge 3 [2351, 5519] root346Candidates5 7870 winner else winner) = _
  rw [if_neg (by decide)]
theorem root346Scan6 : scan edge (search edge 3) 4 [2351] 2351 root346Tail6 winner = winner := by rw [root346Tail6, scan, if_pos (by decide)]
theorem root346Scan5 : scan edge (search edge 3) 4 [2351] 2351 root346Tail5 winner = winner :=
  (root346Step5).trans root346Scan6
theorem root346Scan4 : scan edge (search edge 3) 4 [2351] 2351 root346Tail4 winner = winner :=
  (root346Step4).trans root346Scan5
theorem root346Scan3 : scan edge (search edge 3) 4 [2351] 2351 root346Tail3 winner = winner :=
  (root346Step3).trans root346Scan4
theorem root346Scan2 : scan edge (search edge 3) 4 [2351] 2351 root346Tail2 winner = winner :=
  (root346Step2).trans root346Scan3
theorem root346Scan1 : scan edge (search edge 3) 4 [2351] 2351 root346Tail1 winner = winner :=
  (root346Step1).trans root346Scan2
theorem root346Scan0 : scan edge (search edge 3) 4 [2351] 2351 root346Tail0 winner = winner :=
  (root346Step0).trans root346Scan1
set_option maxRecDepth 4096 in
theorem root346Branch : search edge 4 [2351] root346Candidates 2351 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2351] 2351 root346Tail0 winner = winner
  exact root346Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
