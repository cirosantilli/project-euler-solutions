import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root544Candidates : List Nat := Data.row3943
set_option maxRecDepth 8192 in
theorem root544Candidates_checked : candidates edge 5 0 3943 tail545 winner.bound = root544Candidates := by
  apply candidates_eq_row edge 5 0 3943 tail545 root544Candidates winner.bound tail545_ordered
  · decide
  · decide
  · funext q
    rfl
def root544Tail41 : List Nat := []
def root544Tail40 : List Nat := 22003 :: root544Tail41
def root544Tail39 : List Nat := 21727 :: root544Tail40
def root544Tail38 : List Nat := 20431 :: root544Tail39
def root544Tail37 : List Nat := 19753 :: root544Tail38
def root544Tail36 : List Nat := 19507 :: root544Tail37
def root544Tail35 : List Nat := 19213 :: root544Tail36
def root544Tail34 : List Nat := 18523 :: root544Tail35
def root544Tail33 : List Nat := 18367 :: root544Tail34
def root544Tail32 : List Nat := 17239 :: root544Tail33
def root544Tail31 : List Nat := 16927 :: root544Tail32
def root544Tail30 : List Nat := 16183 :: root544Tail31
def root544Tail29 : List Nat := 15661 :: root544Tail30
def root544Tail28 : List Nat := 15199 :: root544Tail29
def root544Tail27 : List Nat := 15139 :: root544Tail28
def root544Tail26 : List Nat := 14887 :: root544Tail27
def root544Tail25 : List Nat := 14593 :: root544Tail26
def root544Tail24 : List Nat := 14533 :: root544Tail25
def root544Tail23 : List Nat := 14083 :: root544Tail24
def root544Tail22 : List Nat := 13873 :: root544Tail23
def root544Tail21 : List Nat := 12979 :: root544Tail22
def root544Tail20 : List Nat := 12619 :: root544Tail21
def root544Tail19 : List Nat := 12553 :: root544Tail20
def root544Tail18 : List Nat := 12277 :: root544Tail19
def root544Tail17 : List Nat := 10627 :: root544Tail18
def root544Tail16 : List Nat := 9787 :: root544Tail17
def root544Tail15 : List Nat := 9769 :: root544Tail16
def root544Tail14 : List Nat := 9277 :: root544Tail15
def root544Tail13 : List Nat := 8923 :: root544Tail14
def root544Tail12 : List Nat := 8647 :: root544Tail13
def root544Tail11 : List Nat := 8563 :: root544Tail12
def root544Tail10 : List Nat := 8059 :: root544Tail11
def root544Tail9 : List Nat := 7789 :: root544Tail10
def root544Tail8 : List Nat := 6829 :: root544Tail9
def root544Tail7 : List Nat := 5839 :: root544Tail8
def root544Tail6 : List Nat := 5779 :: root544Tail7
def root544Tail5 : List Nat := 5581 :: root544Tail6
def root544Tail4 : List Nat := 5569 :: root544Tail5
def root544Tail3 : List Nat := 5479 :: root544Tail4
def root544Tail2 : List Nat := 5413 :: root544Tail3
def root544Tail1 : List Nat := 5167 :: root544Tail2
def root544Tail0 : List Nat := 4621 :: root544Tail1
def root544Candidates0 : List Nat := [5779]
set_option maxRecDepth 8192 in
theorem root544Candidates0_checked : candidates edge 4 3943 4621 root544Tail1 winner.bound = root544Candidates0 := by
  unfold candidates
  have row : edge 4621 = (SortedNatTree.contains FastRows.row4621) := by
    change Data.row4621.contains = (SortedNatTree.contains FastRows.row4621)
    exact FastRows.row4621_checked.symm
  rw [row]
  decide
theorem root544Step0 : scan edge (search edge 3) 4 [3943] 3943 root544Tail0 winner =
    scan edge (search edge 3) 4 [3943] 3943 root544Tail1 winner := by
  rw [root544Tail0, scan, if_neg (by decide), root544Candidates0_checked]
  change scan edge (search edge 3) 4 [3943] 3943 root544Tail1
    (if 3 ≤ root544Candidates0.length then search edge 3 [3943, 4621] root544Candidates0 8564 winner else winner) = _
  rw [if_neg (by decide)]
def root544Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root544Candidates1_checked : candidates edge 4 3943 5167 root544Tail2 winner.bound = root544Candidates1 := by
  unfold candidates
  have row : edge 5167 = (SortedNatTree.contains FastRows.row5167) := by
    change Data.row5167.contains = (SortedNatTree.contains FastRows.row5167)
    exact FastRows.row5167_checked.symm
  rw [row]
  decide
theorem root544Step1 : scan edge (search edge 3) 4 [3943] 3943 root544Tail1 winner =
    scan edge (search edge 3) 4 [3943] 3943 root544Tail2 winner := by
  rw [root544Tail1, scan, if_neg (by decide), root544Candidates1_checked]
  change scan edge (search edge 3) 4 [3943] 3943 root544Tail2
    (if 3 ≤ root544Candidates1.length then search edge 3 [3943, 5167] root544Candidates1 9110 winner else winner) = _
  rw [if_neg (by decide)]
def root544Candidates2 : List Nat := [5569]
set_option maxRecDepth 8192 in
theorem root544Candidates2_checked : candidates edge 4 3943 5413 root544Tail3 winner.bound = root544Candidates2 := by
  unfold candidates
  have row : edge 5413 = (SortedNatTree.contains FastRows.row5413) := by
    change Data.row5413.contains = (SortedNatTree.contains FastRows.row5413)
    exact FastRows.row5413_checked.symm
  rw [row]
  decide
theorem root544Step2 : scan edge (search edge 3) 4 [3943] 3943 root544Tail2 winner =
    scan edge (search edge 3) 4 [3943] 3943 root544Tail3 winner := by
  rw [root544Tail2, scan, if_neg (by decide), root544Candidates2_checked]
  change scan edge (search edge 3) 4 [3943] 3943 root544Tail3
    (if 3 ≤ root544Candidates2.length then search edge 3 [3943, 5413] root544Candidates2 9356 winner else winner) = _
  rw [if_neg (by decide)]
def root544Candidates3 : List Nat := [5581]
set_option maxRecDepth 8192 in
theorem root544Candidates3_checked : candidates edge 4 3943 5479 root544Tail4 winner.bound = root544Candidates3 := by
  unfold candidates
  have row : edge 5479 = (SortedNatTree.contains FastRows.row5479) := by
    change Data.row5479.contains = (SortedNatTree.contains FastRows.row5479)
    exact FastRows.row5479_checked.symm
  rw [row]
  decide
theorem root544Step3 : scan edge (search edge 3) 4 [3943] 3943 root544Tail3 winner =
    scan edge (search edge 3) 4 [3943] 3943 root544Tail4 winner := by
  rw [root544Tail3, scan, if_neg (by decide), root544Candidates3_checked]
  change scan edge (search edge 3) 4 [3943] 3943 root544Tail4
    (if 3 ≤ root544Candidates3.length then search edge 3 [3943, 5479] root544Candidates3 9422 winner else winner) = _
  rw [if_neg (by decide)]
theorem root544Scan4 : scan edge (search edge 3) 4 [3943] 3943 root544Tail4 winner = winner := by rw [root544Tail4, scan, if_pos (by decide)]
theorem root544Scan3 : scan edge (search edge 3) 4 [3943] 3943 root544Tail3 winner = winner :=
  (root544Step3).trans root544Scan4
theorem root544Scan2 : scan edge (search edge 3) 4 [3943] 3943 root544Tail2 winner = winner :=
  (root544Step2).trans root544Scan3
theorem root544Scan1 : scan edge (search edge 3) 4 [3943] 3943 root544Tail1 winner = winner :=
  (root544Step1).trans root544Scan2
theorem root544Scan0 : scan edge (search edge 3) 4 [3943] 3943 root544Tail0 winner = winner :=
  (root544Step0).trans root544Scan1
set_option maxRecDepth 4096 in
theorem root544Branch : search edge 4 [3943] root544Candidates 3943 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3943] 3943 root544Tail0 winner = winner
  exact root544Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
