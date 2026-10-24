import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root641Candidates : List Nat := Data.row4789
set_option maxRecDepth 8192 in
theorem root641Candidates_checked : candidates edge 5 0 4789 tail642 winner.bound = root641Candidates := by
  apply candidates_eq_row edge 5 0 4789 tail642 root641Candidates winner.bound tail642_ordered
  · decide
  · decide
  · funext q
    rfl
def root641Tail34 : List Nat := []
def root641Tail33 : List Nat := 20161 :: root641Tail34
def root641Tail32 : List Nat := 19489 :: root641Tail33
def root641Tail31 : List Nat := 18301 :: root641Tail32
def root641Tail30 : List Nat := 18127 :: root641Tail31
def root641Tail29 : List Nat := 18061 :: root641Tail30
def root641Tail28 : List Nat := 17683 :: root641Tail29
def root641Tail27 : List Nat := 17623 :: root641Tail28
def root641Tail26 : List Nat := 17599 :: root641Tail27
def root641Tail25 : List Nat := 17317 :: root641Tail26
def root641Tail24 : List Nat := 17209 :: root641Tail25
def root641Tail23 : List Nat := 17191 :: root641Tail24
def root641Tail22 : List Nat := 17041 :: root641Tail23
def root641Tail21 : List Nat := 16369 :: root641Tail22
def root641Tail20 : List Nat := 15991 :: root641Tail21
def root641Tail19 : List Nat := 15667 :: root641Tail20
def root641Tail18 : List Nat := 14713 :: root641Tail19
def root641Tail17 : List Nat := 14431 :: root641Tail18
def root641Tail16 : List Nat := 14221 :: root641Tail17
def root641Tail15 : List Nat := 13399 :: root641Tail16
def root641Tail14 : List Nat := 13147 :: root641Tail15
def root641Tail13 : List Nat := 12979 :: root641Tail14
def root641Tail12 : List Nat := 11617 :: root641Tail13
def root641Tail11 : List Nat := 11113 :: root641Tail12
def root641Tail10 : List Nat := 11071 :: root641Tail11
def root641Tail9 : List Nat := 10657 :: root641Tail10
def root641Tail8 : List Nat := 10177 :: root641Tail9
def root641Tail7 : List Nat := 9871 :: root641Tail8
def root641Tail6 : List Nat := 9277 :: root641Tail7
def root641Tail5 : List Nat := 8269 :: root641Tail6
def root641Tail4 : List Nat := 7561 :: root641Tail5
def root641Tail3 : List Nat := 6553 :: root641Tail4
def root641Tail2 : List Nat := 6547 :: root641Tail3
def root641Tail1 : List Nat := 5179 :: root641Tail2
def root641Tail0 : List Nat := 5119 :: root641Tail1
def root641Candidates0 : List Nat := [10177]
set_option maxRecDepth 8192 in
theorem root641Candidates0_checked : candidates edge 4 4789 5119 root641Tail1 winner.bound = root641Candidates0 := by
  unfold candidates
  have row : edge 5119 = (SortedNatTree.contains FastRows.row5119) := by
    change Data.row5119.contains = (SortedNatTree.contains FastRows.row5119)
    exact FastRows.row5119_checked.symm
  rw [row]
  decide
theorem root641Step0 : scan edge (search edge 3) 4 [4789] 4789 root641Tail0 winner =
    scan edge (search edge 3) 4 [4789] 4789 root641Tail1 winner := by
  rw [root641Tail0, scan, if_neg (by decide), root641Candidates0_checked]
  change scan edge (search edge 3) 4 [4789] 4789 root641Tail1
    (if 3 ≤ root641Candidates0.length then search edge 3 [4789, 5119] root641Candidates0 9908 winner else winner) = _
  rw [if_neg (by decide)]
def root641Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root641Candidates1_checked : candidates edge 4 4789 5179 root641Tail2 winner.bound = root641Candidates1 := by
  unfold candidates
  have row : edge 5179 = (SortedNatTree.contains FastRows.row5179) := by
    change Data.row5179.contains = (SortedNatTree.contains FastRows.row5179)
    exact FastRows.row5179_checked.symm
  rw [row]
  decide
theorem root641Step1 : scan edge (search edge 3) 4 [4789] 4789 root641Tail1 winner =
    scan edge (search edge 3) 4 [4789] 4789 root641Tail2 winner := by
  rw [root641Tail1, scan, if_neg (by decide), root641Candidates1_checked]
  change scan edge (search edge 3) 4 [4789] 4789 root641Tail2
    (if 3 ≤ root641Candidates1.length then search edge 3 [4789, 5179] root641Candidates1 9968 winner else winner) = _
  rw [if_neg (by decide)]
theorem root641Scan2 : scan edge (search edge 3) 4 [4789] 4789 root641Tail2 winner = winner := by rw [root641Tail2, scan, if_pos (by decide)]
theorem root641Scan1 : scan edge (search edge 3) 4 [4789] 4789 root641Tail1 winner = winner :=
  (root641Step1).trans root641Scan2
theorem root641Scan0 : scan edge (search edge 3) 4 [4789] 4789 root641Tail0 winner = winner :=
  (root641Step0).trans root641Scan1
set_option maxRecDepth 4096 in
theorem root641Branch : search edge 4 [4789] root641Candidates 4789 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4789] 4789 root641Tail0 winner = winner
  exact root641Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
