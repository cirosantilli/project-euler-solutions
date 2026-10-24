import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root603Candidates : List Nat := Data.row4457
set_option maxRecDepth 8192 in
theorem root603Candidates_checked : candidates edge 5 0 4457 tail604 winner.bound = root603Candidates := by
  apply candidates_eq_row edge 5 0 4457 tail604 root603Candidates winner.bound tail604_ordered
  · decide
  · decide
  · funext q
    rfl
def root603Tail41 : List Nat := []
def root603Tail40 : List Nat := 21407 :: root603Tail41
def root603Tail39 : List Nat := 20771 :: root603Tail40
def root603Tail38 : List Nat := 19997 :: root603Tail39
def root603Tail37 : List Nat := 18149 :: root603Tail38
def root603Tail36 : List Nat := 18089 :: root603Tail37
def root603Tail35 : List Nat := 17729 :: root603Tail36
def root603Tail34 : List Nat := 17669 :: root603Tail35
def root603Tail33 : List Nat := 17579 :: root603Tail34
def root603Tail32 : List Nat := 16547 :: root603Tail33
def root603Tail31 : List Nat := 15791 :: root603Tail32
def root603Tail30 : List Nat := 15773 :: root603Tail31
def root603Tail29 : List Nat := 14561 :: root603Tail30
def root603Tail28 : List Nat := 13487 :: root603Tail29
def root603Tail27 : List Nat := 13007 :: root603Tail28
def root603Tail26 : List Nat := 12809 :: root603Tail27
def root603Tail25 : List Nat := 12413 :: root603Tail26
def root603Tail24 : List Nat := 11579 :: root603Tail25
def root603Tail23 : List Nat := 10313 :: root603Tail24
def root603Tail22 : List Nat := 10163 :: root603Tail23
def root603Tail21 : List Nat := 10061 :: root603Tail22
def root603Tail20 : List Nat := 9791 :: root603Tail21
def root603Tail19 : List Nat := 9281 :: root603Tail20
def root603Tail18 : List Nat := 9239 :: root603Tail19
def root603Tail17 : List Nat := 9203 :: root603Tail18
def root603Tail16 : List Nat := 8543 :: root603Tail17
def root603Tail15 : List Nat := 7919 :: root603Tail16
def root603Tail14 : List Nat := 7853 :: root603Tail15
def root603Tail13 : List Nat := 7607 :: root603Tail14
def root603Tail12 : List Nat := 7499 :: root603Tail13
def root603Tail11 : List Nat := 7433 :: root603Tail12
def root603Tail10 : List Nat := 7331 :: root603Tail11
def root603Tail9 : List Nat := 7019 :: root603Tail10
def root603Tail8 : List Nat := 6491 :: root603Tail9
def root603Tail7 : List Nat := 6473 :: root603Tail8
def root603Tail6 : List Nat := 5981 :: root603Tail7
def root603Tail5 : List Nat := 5591 :: root603Tail6
def root603Tail4 : List Nat := 5507 :: root603Tail5
def root603Tail3 : List Nat := 5483 :: root603Tail4
def root603Tail2 : List Nat := 5393 :: root603Tail3
def root603Tail1 : List Nat := 4793 :: root603Tail2
def root603Tail0 : List Nat := 4679 :: root603Tail1
def root603Candidates0 : List Nat := [5393]
set_option maxRecDepth 8192 in
theorem root603Candidates0_checked : candidates edge 4 4457 4679 root603Tail1 winner.bound = root603Candidates0 := by
  unfold candidates
  have row : edge 4679 = (SortedNatTree.contains FastRows.row4679) := by
    change Data.row4679.contains = (SortedNatTree.contains FastRows.row4679)
    exact FastRows.row4679_checked.symm
  rw [row]
  decide
theorem root603Step0 : scan edge (search edge 3) 4 [4457] 4457 root603Tail0 winner =
    scan edge (search edge 3) 4 [4457] 4457 root603Tail1 winner := by
  rw [root603Tail0, scan, if_neg (by decide), root603Candidates0_checked]
  change scan edge (search edge 3) 4 [4457] 4457 root603Tail1
    (if 3 ≤ root603Candidates0.length then search edge 3 [4457, 4679] root603Candidates0 9136 winner else winner) = _
  rw [if_neg (by decide)]
def root603Candidates1 : List Nat := [9791, 13007]
set_option maxRecDepth 8192 in
theorem root603Candidates1_checked : candidates edge 4 4457 4793 root603Tail2 winner.bound = root603Candidates1 := by
  unfold candidates
  have row : edge 4793 = (SortedNatTree.contains FastRows.row4793) := by
    change Data.row4793.contains = (SortedNatTree.contains FastRows.row4793)
    exact FastRows.row4793_checked.symm
  rw [row]
  decide
theorem root603Step1 : scan edge (search edge 3) 4 [4457] 4457 root603Tail1 winner =
    scan edge (search edge 3) 4 [4457] 4457 root603Tail2 winner := by
  rw [root603Tail1, scan, if_neg (by decide), root603Candidates1_checked]
  change scan edge (search edge 3) 4 [4457] 4457 root603Tail2
    (if 3 ≤ root603Candidates1.length then search edge 3 [4457, 4793] root603Candidates1 9250 winner else winner) = _
  rw [if_neg (by decide)]
def root603Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root603Candidates2_checked : candidates edge 4 4457 5393 root603Tail3 winner.bound = root603Candidates2 := by
  unfold candidates
  have row : edge 5393 = (SortedNatTree.contains FastRows.row5393) := by
    change Data.row5393.contains = (SortedNatTree.contains FastRows.row5393)
    exact FastRows.row5393_checked.symm
  rw [row]
  decide
theorem root603Step2 : scan edge (search edge 3) 4 [4457] 4457 root603Tail2 winner =
    scan edge (search edge 3) 4 [4457] 4457 root603Tail3 winner := by
  rw [root603Tail2, scan, if_neg (by decide), root603Candidates2_checked]
  change scan edge (search edge 3) 4 [4457] 4457 root603Tail3
    (if 3 ≤ root603Candidates2.length then search edge 3 [4457, 5393] root603Candidates2 9850 winner else winner) = _
  rw [if_neg (by decide)]
theorem root603Scan3 : scan edge (search edge 3) 4 [4457] 4457 root603Tail3 winner = winner := by rw [root603Tail3, scan, if_pos (by decide)]
theorem root603Scan2 : scan edge (search edge 3) 4 [4457] 4457 root603Tail2 winner = winner :=
  (root603Step2).trans root603Scan3
theorem root603Scan1 : scan edge (search edge 3) 4 [4457] 4457 root603Tail1 winner = winner :=
  (root603Step1).trans root603Scan2
theorem root603Scan0 : scan edge (search edge 3) 4 [4457] 4457 root603Tail0 winner = winner :=
  (root603Step0).trans root603Scan1
set_option maxRecDepth 4096 in
theorem root603Branch : search edge 4 [4457] root603Candidates 4457 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4457] 4457 root603Tail0 winner = winner
  exact root603Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
