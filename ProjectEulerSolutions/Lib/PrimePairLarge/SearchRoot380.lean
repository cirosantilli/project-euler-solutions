import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root380Candidates : List Nat := Data.row2647
set_option maxRecDepth 8192 in
theorem root380Candidates_checked : candidates edge 5 0 2647 tail381 winner.bound = root380Candidates := by
  apply candidates_eq_row edge 5 0 2647 tail381 root380Candidates winner.bound tail381_ordered
  · decide
  · decide
  · funext q
    rfl
def root380Tail50 : List Nat := []
def root380Tail49 : List Nat := 21727 :: root380Tail50
def root380Tail48 : List Nat := 21673 :: root380Tail49
def root380Tail47 : List Nat := 21277 :: root380Tail48
def root380Tail46 : List Nat := 20359 :: root380Tail47
def root380Tail45 : List Nat := 19963 :: root380Tail46
def root380Tail44 : List Nat := 19387 :: root380Tail45
def root380Tail43 : List Nat := 19069 :: root380Tail44
def root380Tail42 : List Nat := 18397 :: root380Tail43
def root380Tail41 : List Nat := 18301 :: root380Tail42
def root380Tail40 : List Nat := 18211 :: root380Tail41
def root380Tail39 : List Nat := 17377 :: root380Tail40
def root380Tail38 : List Nat := 17317 :: root380Tail39
def root380Tail37 : List Nat := 17293 :: root380Tail38
def root380Tail36 : List Nat := 16993 :: root380Tail37
def root380Tail35 : List Nat := 16057 :: root380Tail36
def root380Tail34 : List Nat := 15889 :: root380Tail35
def root380Tail33 : List Nat := 15511 :: root380Tail34
def root380Tail32 : List Nat := 15361 :: root380Tail33
def root380Tail31 : List Nat := 15289 :: root380Tail32
def root380Tail30 : List Nat := 15259 :: root380Tail31
def root380Tail29 : List Nat := 14071 :: root380Tail30
def root380Tail28 : List Nat := 13807 :: root380Tail29
def root380Tail27 : List Nat := 13789 :: root380Tail28
def root380Tail26 : List Nat := 13441 :: root380Tail27
def root380Tail25 : List Nat := 13411 :: root380Tail26
def root380Tail24 : List Nat := 13219 :: root380Tail25
def root380Tail23 : List Nat := 12853 :: root380Tail24
def root380Tail22 : List Nat := 12739 :: root380Tail23
def root380Tail21 : List Nat := 12697 :: root380Tail22
def root380Tail20 : List Nat := 12583 :: root380Tail21
def root380Tail19 : List Nat := 12391 :: root380Tail20
def root380Tail18 : List Nat := 11731 :: root380Tail19
def root380Tail17 : List Nat := 11197 :: root380Tail18
def root380Tail16 : List Nat := 11113 :: root380Tail17
def root380Tail15 : List Nat := 9151 :: root380Tail16
def root380Tail14 : List Nat := 8971 :: root380Tail15
def root380Tail13 : List Nat := 8737 :: root380Tail14
def root380Tail12 : List Nat := 8563 :: root380Tail13
def root380Tail11 : List Nat := 8161 :: root380Tail12
def root380Tail10 : List Nat := 7951 :: root380Tail11
def root380Tail9 : List Nat := 6673 :: root380Tail10
def root380Tail8 : List Nat := 6577 :: root380Tail9
def root380Tail7 : List Nat := 6553 :: root380Tail8
def root380Tail6 : List Nat := 6271 :: root380Tail7
def root380Tail5 : List Nat := 6091 :: root380Tail6
def root380Tail4 : List Nat := 5431 :: root380Tail5
def root380Tail3 : List Nat := 4663 :: root380Tail4
def root380Tail2 : List Nat := 4057 :: root380Tail3
def root380Tail1 : List Nat := 3793 :: root380Tail2
def root380Tail0 : List Nat := 2851 :: root380Tail1
def root380Candidates0 : List Nat := [6091, 12739]
set_option maxRecDepth 8192 in
theorem root380Candidates0_checked : candidates edge 4 2647 2851 root380Tail1 winner.bound = root380Candidates0 := by
  unfold candidates
  have row : edge 2851 = (SortedNatTree.contains FastRows.row2851) := by
    change Data.row2851.contains = (SortedNatTree.contains FastRows.row2851)
    exact FastRows.row2851_checked.symm
  rw [row]
  decide
theorem root380Step0 : scan edge (search edge 3) 4 [2647] 2647 root380Tail0 winner =
    scan edge (search edge 3) 4 [2647] 2647 root380Tail1 winner := by
  rw [root380Tail0, scan, if_neg (by decide), root380Candidates0_checked]
  change scan edge (search edge 3) 4 [2647] 2647 root380Tail1
    (if 3 ≤ root380Candidates0.length then search edge 3 [2647, 2851] root380Candidates0 5498 winner else winner) = _
  rw [if_neg (by decide)]
def root380Candidates1 : List Nat := [11731, 13219, 17293]
set_option maxRecDepth 8192 in
theorem root380Candidates1_checked : candidates edge 4 2647 3793 root380Tail2 winner.bound = root380Candidates1 := by
  unfold candidates
  have row : edge 3793 = (SortedNatTree.contains FastRows.row3793) := by
    change Data.row3793.contains = (SortedNatTree.contains FastRows.row3793)
    exact FastRows.row3793_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root380Child1 : search edge 3 [2647, 3793] root380Candidates1 6440 winner = winner := by decide
theorem root380Step1 : scan edge (search edge 3) 4 [2647] 2647 root380Tail1 winner =
    scan edge (search edge 3) 4 [2647] 2647 root380Tail2 winner := by
  rw [root380Tail1, scan, if_neg (by decide), root380Candidates1_checked]
  change scan edge (search edge 3) 4 [2647] 2647 root380Tail2
    (if 3 ≤ root380Candidates1.length then search edge 3 [2647, 3793] root380Candidates1 6440 winner else winner) = _
  rw [if_pos (by decide), root380Child1]
def root380Candidates2 : List Nat := [4663, 12583]
set_option maxRecDepth 8192 in
theorem root380Candidates2_checked : candidates edge 4 2647 4057 root380Tail3 winner.bound = root380Candidates2 := by
  unfold candidates
  have row : edge 4057 = (SortedNatTree.contains FastRows.row4057) := by
    change Data.row4057.contains = (SortedNatTree.contains FastRows.row4057)
    exact FastRows.row4057_checked.symm
  rw [row]
  decide
theorem root380Step2 : scan edge (search edge 3) 4 [2647] 2647 root380Tail2 winner =
    scan edge (search edge 3) 4 [2647] 2647 root380Tail3 winner := by
  rw [root380Tail2, scan, if_neg (by decide), root380Candidates2_checked]
  change scan edge (search edge 3) 4 [2647] 2647 root380Tail3
    (if 3 ≤ root380Candidates2.length then search edge 3 [2647, 4057] root380Candidates2 6704 winner else winner) = _
  rw [if_neg (by decide)]
def root380Candidates3 : List Nat := [11113, 16057]
set_option maxRecDepth 8192 in
theorem root380Candidates3_checked : candidates edge 4 2647 4663 root380Tail4 winner.bound = root380Candidates3 := by
  unfold candidates
  have row : edge 4663 = (SortedNatTree.contains FastRows.row4663) := by
    change Data.row4663.contains = (SortedNatTree.contains FastRows.row4663)
    exact FastRows.row4663_checked.symm
  rw [row]
  decide
theorem root380Step3 : scan edge (search edge 3) 4 [2647] 2647 root380Tail3 winner =
    scan edge (search edge 3) 4 [2647] 2647 root380Tail4 winner := by
  rw [root380Tail3, scan, if_neg (by decide), root380Candidates3_checked]
  change scan edge (search edge 3) 4 [2647] 2647 root380Tail4
    (if 3 ≤ root380Candidates3.length then search edge 3 [2647, 4663] root380Candidates3 7310 winner else winner) = _
  rw [if_neg (by decide)]
def root380Candidates4 : List Nat := [15259]
set_option maxRecDepth 8192 in
theorem root380Candidates4_checked : candidates edge 4 2647 5431 root380Tail5 winner.bound = root380Candidates4 := by
  unfold candidates
  have row : edge 5431 = (SortedNatTree.contains FastRows.row5431) := by
    change Data.row5431.contains = (SortedNatTree.contains FastRows.row5431)
    exact FastRows.row5431_checked.symm
  rw [row]
  decide
theorem root380Step4 : scan edge (search edge 3) 4 [2647] 2647 root380Tail4 winner =
    scan edge (search edge 3) 4 [2647] 2647 root380Tail5 winner := by
  rw [root380Tail4, scan, if_neg (by decide), root380Candidates4_checked]
  change scan edge (search edge 3) 4 [2647] 2647 root380Tail5
    (if 3 ≤ root380Candidates4.length then search edge 3 [2647, 5431] root380Candidates4 8078 winner else winner) = _
  rw [if_neg (by decide)]
theorem root380Scan5 : scan edge (search edge 3) 4 [2647] 2647 root380Tail5 winner = winner := by rw [root380Tail5, scan, if_pos (by decide)]
theorem root380Scan4 : scan edge (search edge 3) 4 [2647] 2647 root380Tail4 winner = winner :=
  (root380Step4).trans root380Scan5
theorem root380Scan3 : scan edge (search edge 3) 4 [2647] 2647 root380Tail3 winner = winner :=
  (root380Step3).trans root380Scan4
theorem root380Scan2 : scan edge (search edge 3) 4 [2647] 2647 root380Tail2 winner = winner :=
  (root380Step2).trans root380Scan3
theorem root380Scan1 : scan edge (search edge 3) 4 [2647] 2647 root380Tail1 winner = winner :=
  (root380Step1).trans root380Scan2
theorem root380Scan0 : scan edge (search edge 3) 4 [2647] 2647 root380Tail0 winner = winner :=
  (root380Step0).trans root380Scan1
set_option maxRecDepth 4096 in
theorem root380Branch : search edge 4 [2647] root380Candidates 2647 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2647] 2647 root380Tail0 winner = winner
  exact root380Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
