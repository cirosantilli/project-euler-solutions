import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root489Candidates : List Nat := Data.row3527
set_option maxRecDepth 8192 in
theorem root489Candidates_checked : candidates edge 5 0 3527 tail490 winner.bound = root489Candidates := by
  apply candidates_eq_row edge 5 0 3527 tail490 root489Candidates winner.bound tail490_ordered
  · decide
  · decide
  · funext q
    rfl
def root489Tail32 : List Nat := []
def root489Tail31 : List Nat := 22229 :: root489Tail32
def root489Tail30 : List Nat := 21587 :: root489Tail31
def root489Tail29 : List Nat := 21059 :: root489Tail30
def root489Tail28 : List Nat := 20747 :: root489Tail29
def root489Tail27 : List Nat := 20507 :: root489Tail28
def root489Tail26 : List Nat := 20327 :: root489Tail27
def root489Tail25 : List Nat := 19709 :: root489Tail26
def root489Tail24 : List Nat := 19571 :: root489Tail25
def root489Tail23 : List Nat := 19163 :: root489Tail24
def root489Tail22 : List Nat := 18617 :: root489Tail23
def root489Tail21 : List Nat := 17837 :: root489Tail22
def root489Tail20 : List Nat := 17471 :: root489Tail21
def root489Tail19 : List Nat := 16631 :: root489Tail20
def root489Tail18 : List Nat := 16421 :: root489Tail19
def root489Tail17 : List Nat := 14897 :: root489Tail18
def root489Tail16 : List Nat := 12911 :: root489Tail17
def root489Tail15 : List Nat := 12377 :: root489Tail16
def root489Tail14 : List Nat := 11519 :: root489Tail15
def root489Tail13 : List Nat := 11351 :: root489Tail14
def root489Tail12 : List Nat := 10979 :: root489Tail13
def root489Tail11 : List Nat := 9323 :: root489Tail12
def root489Tail10 : List Nat := 9239 :: root489Tail11
def root489Tail9 : List Nat := 8861 :: root489Tail10
def root489Tail8 : List Nat := 8669 :: root489Tail9
def root489Tail7 : List Nat := 8429 :: root489Tail8
def root489Tail6 : List Nat := 7901 :: root489Tail7
def root489Tail5 : List Nat := 7013 :: root489Tail6
def root489Tail4 : List Nat := 6803 :: root489Tail5
def root489Tail3 : List Nat := 6551 :: root489Tail4
def root489Tail2 : List Nat := 4649 :: root489Tail3
def root489Tail1 : List Nat := 4457 :: root489Tail2
def root489Tail0 : List Nat := 3533 :: root489Tail1
def root489Candidates0 : List Nat := [8861, 9323]
set_option maxRecDepth 8192 in
theorem root489Candidates0_checked : candidates edge 4 3527 3533 root489Tail1 winner.bound = root489Candidates0 := by
  unfold candidates
  have row : edge 3533 = (SortedNatTree.contains FastRows.row3533) := by
    change Data.row3533.contains = (SortedNatTree.contains FastRows.row3533)
    exact FastRows.row3533_checked.symm
  rw [row]
  decide
theorem root489Step0 : scan edge (search edge 3) 4 [3527] 3527 root489Tail0 winner =
    scan edge (search edge 3) 4 [3527] 3527 root489Tail1 winner := by
  rw [root489Tail0, scan, if_neg (by decide), root489Candidates0_checked]
  change scan edge (search edge 3) 4 [3527] 3527 root489Tail1
    (if 3 ≤ root489Candidates0.length then search edge 3 [3527, 3533] root489Candidates0 7060 winner else winner) = _
  rw [if_neg (by decide)]
def root489Candidates1 : List Nat := [9239]
set_option maxRecDepth 8192 in
theorem root489Candidates1_checked : candidates edge 4 3527 4457 root489Tail2 winner.bound = root489Candidates1 := by
  unfold candidates
  have row : edge 4457 = (SortedNatTree.contains FastRows.row4457) := by
    change Data.row4457.contains = (SortedNatTree.contains FastRows.row4457)
    exact FastRows.row4457_checked.symm
  rw [row]
  decide
theorem root489Step1 : scan edge (search edge 3) 4 [3527] 3527 root489Tail1 winner =
    scan edge (search edge 3) 4 [3527] 3527 root489Tail2 winner := by
  rw [root489Tail1, scan, if_neg (by decide), root489Candidates1_checked]
  change scan edge (search edge 3) 4 [3527] 3527 root489Tail2
    (if 3 ≤ root489Candidates1.length then search edge 3 [3527, 4457] root489Candidates1 7984 winner else winner) = _
  rw [if_neg (by decide)]
def root489Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root489Candidates2_checked : candidates edge 4 3527 4649 root489Tail3 winner.bound = root489Candidates2 := by
  unfold candidates
  have row : edge 4649 = (SortedNatTree.contains FastRows.row4649) := by
    change Data.row4649.contains = (SortedNatTree.contains FastRows.row4649)
    exact FastRows.row4649_checked.symm
  rw [row]
  decide
theorem root489Step2 : scan edge (search edge 3) 4 [3527] 3527 root489Tail2 winner =
    scan edge (search edge 3) 4 [3527] 3527 root489Tail3 winner := by
  rw [root489Tail2, scan, if_neg (by decide), root489Candidates2_checked]
  change scan edge (search edge 3) 4 [3527] 3527 root489Tail3
    (if 3 ≤ root489Candidates2.length then search edge 3 [3527, 4649] root489Candidates2 8176 winner else winner) = _
  rw [if_neg (by decide)]
theorem root489Scan3 : scan edge (search edge 3) 4 [3527] 3527 root489Tail3 winner = winner := by rw [root489Tail3, scan, if_pos (by decide)]
theorem root489Scan2 : scan edge (search edge 3) 4 [3527] 3527 root489Tail2 winner = winner :=
  (root489Step2).trans root489Scan3
theorem root489Scan1 : scan edge (search edge 3) 4 [3527] 3527 root489Tail1 winner = winner :=
  (root489Step1).trans root489Scan2
theorem root489Scan0 : scan edge (search edge 3) 4 [3527] 3527 root489Tail0 winner = winner :=
  (root489Step0).trans root489Scan1
set_option maxRecDepth 4096 in
theorem root489Branch : search edge 4 [3527] root489Candidates 3527 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3527] 3527 root489Tail0 winner = winner
  exact root489Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
