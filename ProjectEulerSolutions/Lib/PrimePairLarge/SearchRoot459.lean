import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root459Candidates : List Nat := Data.row3271
set_option maxRecDepth 8192 in
theorem root459Candidates_checked : candidates edge 5 0 3271 tail460 winner.bound = root459Candidates := by
  apply candidates_eq_row edge 5 0 3271 tail460 root459Candidates winner.bound tail460_ordered
  · decide
  · decide
  · funext q
    rfl
def root459Tail33 : List Nat := []
def root459Tail32 : List Nat := 22573 :: root459Tail33
def root459Tail31 : List Nat := 22441 :: root459Tail32
def root459Tail30 : List Nat := 22153 :: root459Tail31
def root459Tail29 : List Nat := 21187 :: root459Tail30
def root459Tail28 : List Nat := 21169 :: root459Tail29
def root459Tail27 : List Nat := 20389 :: root459Tail28
def root459Tail26 : List Nat := 20089 :: root459Tail27
def root459Tail25 : List Nat := 19927 :: root459Tail26
def root459Tail24 : List Nat := 19231 :: root459Tail25
def root459Tail23 : List Nat := 18481 :: root459Tail24
def root459Tail22 : List Nat := 17911 :: root459Tail23
def root459Tail21 : List Nat := 17107 :: root459Tail22
def root459Tail20 : List Nat := 16759 :: root459Tail21
def root459Tail19 : List Nat := 16417 :: root459Tail20
def root459Tail18 : List Nat := 16249 :: root459Tail19
def root459Tail17 : List Nat := 14503 :: root459Tail18
def root459Tail16 : List Nat := 13921 :: root459Tail17
def root459Tail15 : List Nat := 12763 :: root459Tail16
def root459Tail14 : List Nat := 10711 :: root459Tail15
def root459Tail13 : List Nat := 9697 :: root459Tail14
def root459Tail12 : List Nat := 9343 :: root459Tail13
def root459Tail11 : List Nat := 9199 :: root459Tail12
def root459Tail10 : List Nat := 9013 :: root459Tail11
def root459Tail9 : List Nat := 8377 :: root459Tail10
def root459Tail8 : List Nat := 7621 :: root459Tail9
def root459Tail7 : List Nat := 7603 :: root459Tail8
def root459Tail6 : List Nat := 6661 :: root459Tail7
def root459Tail5 : List Nat := 5923 :: root459Tail6
def root459Tail4 : List Nat := 5653 :: root459Tail5
def root459Tail3 : List Nat := 5227 :: root459Tail4
def root459Tail2 : List Nat := 4243 :: root459Tail3
def root459Tail1 : List Nat := 3361 :: root459Tail2
def root459Tail0 : List Nat := 3319 :: root459Tail1
def root459Candidates0 : List Nat := [17107]
set_option maxRecDepth 8192 in
theorem root459Candidates0_checked : candidates edge 4 3271 3319 root459Tail1 winner.bound = root459Candidates0 := by
  unfold candidates
  have row : edge 3319 = (SortedNatTree.contains FastRows.row3319) := by
    change Data.row3319.contains = (SortedNatTree.contains FastRows.row3319)
    exact FastRows.row3319_checked.symm
  rw [row]
  decide
theorem root459Step0 : scan edge (search edge 3) 4 [3271] 3271 root459Tail0 winner =
    scan edge (search edge 3) 4 [3271] 3271 root459Tail1 winner := by
  rw [root459Tail0, scan, if_neg (by decide), root459Candidates0_checked]
  change scan edge (search edge 3) 4 [3271] 3271 root459Tail1
    (if 3 ≤ root459Candidates0.length then search edge 3 [3271, 3319] root459Candidates0 6590 winner else winner) = _
  rw [if_neg (by decide)]
def root459Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root459Candidates1_checked : candidates edge 4 3271 3361 root459Tail2 winner.bound = root459Candidates1 := by
  unfold candidates
  have row : edge 3361 = (SortedNatTree.contains FastRows.row3361) := by
    change Data.row3361.contains = (SortedNatTree.contains FastRows.row3361)
    exact FastRows.row3361_checked.symm
  rw [row]
  decide
theorem root459Step1 : scan edge (search edge 3) 4 [3271] 3271 root459Tail1 winner =
    scan edge (search edge 3) 4 [3271] 3271 root459Tail2 winner := by
  rw [root459Tail1, scan, if_neg (by decide), root459Candidates1_checked]
  change scan edge (search edge 3) 4 [3271] 3271 root459Tail2
    (if 3 ≤ root459Candidates1.length then search edge 3 [3271, 3361] root459Candidates1 6632 winner else winner) = _
  rw [if_neg (by decide)]
def root459Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root459Candidates2_checked : candidates edge 4 3271 4243 root459Tail3 winner.bound = root459Candidates2 := by
  unfold candidates
  have row : edge 4243 = (SortedNatTree.contains FastRows.row4243) := by
    change Data.row4243.contains = (SortedNatTree.contains FastRows.row4243)
    exact FastRows.row4243_checked.symm
  rw [row]
  decide
theorem root459Step2 : scan edge (search edge 3) 4 [3271] 3271 root459Tail2 winner =
    scan edge (search edge 3) 4 [3271] 3271 root459Tail3 winner := by
  rw [root459Tail2, scan, if_neg (by decide), root459Candidates2_checked]
  change scan edge (search edge 3) 4 [3271] 3271 root459Tail3
    (if 3 ≤ root459Candidates2.length then search edge 3 [3271, 4243] root459Candidates2 7514 winner else winner) = _
  rw [if_neg (by decide)]
def root459Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root459Candidates3_checked : candidates edge 4 3271 5227 root459Tail4 winner.bound = root459Candidates3 := by
  unfold candidates
  have row : edge 5227 = (SortedNatTree.contains FastRows.row5227) := by
    change Data.row5227.contains = (SortedNatTree.contains FastRows.row5227)
    exact FastRows.row5227_checked.symm
  rw [row]
  decide
theorem root459Step3 : scan edge (search edge 3) 4 [3271] 3271 root459Tail3 winner =
    scan edge (search edge 3) 4 [3271] 3271 root459Tail4 winner := by
  rw [root459Tail3, scan, if_neg (by decide), root459Candidates3_checked]
  change scan edge (search edge 3) 4 [3271] 3271 root459Tail4
    (if 3 ≤ root459Candidates3.length then search edge 3 [3271, 5227] root459Candidates3 8498 winner else winner) = _
  rw [if_neg (by decide)]
def root459Candidates4 : List Nat := [5923, 9013, 13921]
set_option maxRecDepth 8192 in
theorem root459Candidates4_checked : candidates edge 4 3271 5653 root459Tail5 winner.bound = root459Candidates4 := by
  unfold candidates
  have row : edge 5653 = (SortedNatTree.contains FastRows.row5653) := by
    change Data.row5653.contains = (SortedNatTree.contains FastRows.row5653)
    exact FastRows.row5653_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root459Child4 : search edge 3 [3271, 5653] root459Candidates4 8924 winner = winner := by decide
theorem root459Step4 : scan edge (search edge 3) 4 [3271] 3271 root459Tail4 winner =
    scan edge (search edge 3) 4 [3271] 3271 root459Tail5 winner := by
  rw [root459Tail4, scan, if_neg (by decide), root459Candidates4_checked]
  change scan edge (search edge 3) 4 [3271] 3271 root459Tail5
    (if 3 ≤ root459Candidates4.length then search edge 3 [3271, 5653] root459Candidates4 8924 winner else winner) = _
  rw [if_pos (by decide), root459Child4]
theorem root459Scan5 : scan edge (search edge 3) 4 [3271] 3271 root459Tail5 winner = winner := by rw [root459Tail5, scan, if_pos (by decide)]
theorem root459Scan4 : scan edge (search edge 3) 4 [3271] 3271 root459Tail4 winner = winner :=
  (root459Step4).trans root459Scan5
theorem root459Scan3 : scan edge (search edge 3) 4 [3271] 3271 root459Tail3 winner = winner :=
  (root459Step3).trans root459Scan4
theorem root459Scan2 : scan edge (search edge 3) 4 [3271] 3271 root459Tail2 winner = winner :=
  (root459Step2).trans root459Scan3
theorem root459Scan1 : scan edge (search edge 3) 4 [3271] 3271 root459Tail1 winner = winner :=
  (root459Step1).trans root459Scan2
theorem root459Scan0 : scan edge (search edge 3) 4 [3271] 3271 root459Tail0 winner = winner :=
  (root459Step0).trans root459Scan1
set_option maxRecDepth 4096 in
theorem root459Branch : search edge 4 [3271] root459Candidates 3271 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3271] 3271 root459Tail0 winner = winner
  exact root459Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
