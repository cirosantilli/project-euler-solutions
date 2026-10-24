import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root462Candidates : List Nat := Data.row3307
set_option maxRecDepth 8192 in
theorem root462Candidates_checked : candidates edge 5 0 3307 tail463 winner.bound = root462Candidates := by
  apply candidates_eq_row edge 5 0 3307 tail463 root462Candidates winner.bound tail463_ordered
  · decide
  · decide
  · funext q
    rfl
def root462Tail38 : List Nat := []
def root462Tail37 : List Nat := 22501 :: root462Tail38
def root462Tail36 : List Nat := 21871 :: root462Tail37
def root462Tail35 : List Nat := 21529 :: root462Tail36
def root462Tail34 : List Nat := 21499 :: root462Tail35
def root462Tail33 : List Nat := 20407 :: root462Tail34
def root462Tail32 : List Nat := 18457 :: root462Tail33
def root462Tail31 : List Nat := 18301 :: root462Tail32
def root462Tail30 : List Nat := 18133 :: root462Tail31
def root462Tail29 : List Nat := 15313 :: root462Tail30
def root462Tail28 : List Nat := 15199 :: root462Tail29
def root462Tail27 : List Nat := 15031 :: root462Tail28
def root462Tail26 : List Nat := 14431 :: root462Tail27
def root462Tail25 : List Nat := 14389 :: root462Tail26
def root462Tail24 : List Nat := 12073 :: root462Tail25
def root462Tail23 : List Nat := 12037 :: root462Tail24
def root462Tail22 : List Nat := 10651 :: root462Tail23
def root462Tail21 : List Nat := 10273 :: root462Tail22
def root462Tail20 : List Nat := 10099 :: root462Tail21
def root462Tail19 : List Nat := 9973 :: root462Tail20
def root462Tail18 : List Nat := 9643 :: root462Tail19
def root462Tail17 : List Nat := 9391 :: root462Tail18
def root462Tail16 : List Nat := 9349 :: root462Tail17
def root462Tail15 : List Nat := 9181 :: root462Tail16
def root462Tail14 : List Nat := 9091 :: root462Tail15
def root462Tail13 : List Nat := 8713 :: root462Tail14
def root462Tail12 : List Nat := 7411 :: root462Tail13
def root462Tail11 : List Nat := 7321 :: root462Tail12
def root462Tail10 : List Nat := 7039 :: root462Tail11
def root462Tail9 : List Nat := 6661 :: root462Tail10
def root462Tail8 : List Nat := 6619 :: root462Tail9
def root462Tail7 : List Nat := 6367 :: root462Tail8
def root462Tail6 : List Nat := 6151 :: root462Tail7
def root462Tail5 : List Nat := 5953 :: root462Tail6
def root462Tail4 : List Nat := 5737 :: root462Tail5
def root462Tail3 : List Nat := 5443 :: root462Tail4
def root462Tail2 : List Nat := 4519 :: root462Tail3
def root462Tail1 : List Nat := 3583 :: root462Tail2
def root462Tail0 : List Nat := 3331 :: root462Tail1
def root462Candidates0 : List Nat := [8713]
set_option maxRecDepth 8192 in
theorem root462Candidates0_checked : candidates edge 4 3307 3331 root462Tail1 winner.bound = root462Candidates0 := by
  unfold candidates
  have row : edge 3331 = (SortedNatTree.contains FastRows.row3331) := by
    change Data.row3331.contains = (SortedNatTree.contains FastRows.row3331)
    exact FastRows.row3331_checked.symm
  rw [row]
  decide
theorem root462Step0 : scan edge (search edge 3) 4 [3307] 3307 root462Tail0 winner =
    scan edge (search edge 3) 4 [3307] 3307 root462Tail1 winner := by
  rw [root462Tail0, scan, if_neg (by decide), root462Candidates0_checked]
  change scan edge (search edge 3) 4 [3307] 3307 root462Tail1
    (if 3 ≤ root462Candidates0.length then search edge 3 [3307, 3331] root462Candidates0 6638 winner else winner) = _
  rw [if_neg (by decide)]
def root462Candidates1 : List Nat := [5953]
set_option maxRecDepth 8192 in
theorem root462Candidates1_checked : candidates edge 4 3307 3583 root462Tail2 winner.bound = root462Candidates1 := by
  unfold candidates
  have row : edge 3583 = (SortedNatTree.contains FastRows.row3583) := by
    change Data.row3583.contains = (SortedNatTree.contains FastRows.row3583)
    exact FastRows.row3583_checked.symm
  rw [row]
  decide
theorem root462Step1 : scan edge (search edge 3) 4 [3307] 3307 root462Tail1 winner =
    scan edge (search edge 3) 4 [3307] 3307 root462Tail2 winner := by
  rw [root462Tail1, scan, if_neg (by decide), root462Candidates1_checked]
  change scan edge (search edge 3) 4 [3307] 3307 root462Tail2
    (if 3 ≤ root462Candidates1.length then search edge 3 [3307, 3583] root462Candidates1 6890 winner else winner) = _
  rw [if_neg (by decide)]
def root462Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root462Candidates2_checked : candidates edge 4 3307 4519 root462Tail3 winner.bound = root462Candidates2 := by
  unfold candidates
  have row : edge 4519 = (SortedNatTree.contains FastRows.row4519) := by
    change Data.row4519.contains = (SortedNatTree.contains FastRows.row4519)
    exact FastRows.row4519_checked.symm
  rw [row]
  decide
theorem root462Step2 : scan edge (search edge 3) 4 [3307] 3307 root462Tail2 winner =
    scan edge (search edge 3) 4 [3307] 3307 root462Tail3 winner := by
  rw [root462Tail2, scan, if_neg (by decide), root462Candidates2_checked]
  change scan edge (search edge 3) 4 [3307] 3307 root462Tail3
    (if 3 ≤ root462Candidates2.length then search edge 3 [3307, 4519] root462Candidates2 7826 winner else winner) = _
  rw [if_neg (by decide)]
def root462Candidates3 : List Nat := [7039, 9391, 12037, 14431]
set_option maxRecDepth 8192 in
theorem root462Candidates3_checked : candidates edge 4 3307 5443 root462Tail4 winner.bound = root462Candidates3 := by
  unfold candidates
  have row : edge 5443 = (SortedNatTree.contains FastRows.row5443) := by
    change Data.row5443.contains = (SortedNatTree.contains FastRows.row5443)
    exact FastRows.row5443_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root462Child3 : search edge 3 [3307, 5443] root462Candidates3 8750 winner = winner := by decide
theorem root462Step3 : scan edge (search edge 3) 4 [3307] 3307 root462Tail3 winner =
    scan edge (search edge 3) 4 [3307] 3307 root462Tail4 winner := by
  rw [root462Tail3, scan, if_neg (by decide), root462Candidates3_checked]
  change scan edge (search edge 3) 4 [3307] 3307 root462Tail4
    (if 3 ≤ root462Candidates3.length then search edge 3 [3307, 5443] root462Candidates3 8750 winner else winner) = _
  rw [if_pos (by decide), root462Child3]
theorem root462Scan4 : scan edge (search edge 3) 4 [3307] 3307 root462Tail4 winner = winner := by rw [root462Tail4, scan, if_pos (by decide)]
theorem root462Scan3 : scan edge (search edge 3) 4 [3307] 3307 root462Tail3 winner = winner :=
  (root462Step3).trans root462Scan4
theorem root462Scan2 : scan edge (search edge 3) 4 [3307] 3307 root462Tail2 winner = winner :=
  (root462Step2).trans root462Scan3
theorem root462Scan1 : scan edge (search edge 3) 4 [3307] 3307 root462Tail1 winner = winner :=
  (root462Step1).trans root462Scan2
theorem root462Scan0 : scan edge (search edge 3) 4 [3307] 3307 root462Tail0 winner = winner :=
  (root462Step0).trans root462Scan1
set_option maxRecDepth 4096 in
theorem root462Branch : search edge 4 [3307] root462Candidates 3307 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3307] 3307 root462Tail0 winner = winner
  exact root462Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
