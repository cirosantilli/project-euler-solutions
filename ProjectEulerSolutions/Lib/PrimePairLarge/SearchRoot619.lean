import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root619Candidates : List Nat := Data.row4597
set_option maxRecDepth 8192 in
theorem root619Candidates_checked : candidates edge 5 0 4597 tail620 winner.bound = root619Candidates := by
  apply candidates_eq_row edge 5 0 4597 tail620 root619Candidates winner.bound tail620_ordered
  · decide
  · decide
  · funext q
    rfl
def root619Tail36 : List Nat := []
def root619Tail35 : List Nat := 21139 :: root619Tail36
def root619Tail34 : List Nat := 20431 :: root619Tail35
def root619Tail33 : List Nat := 20089 :: root619Tail34
def root619Tail32 : List Nat := 19417 :: root619Tail33
def root619Tail31 : List Nat := 18307 :: root619Tail32
def root619Tail30 : List Nat := 17431 :: root619Tail31
def root619Tail29 : List Nat := 16987 :: root619Tail30
def root619Tail28 : List Nat := 16189 :: root619Tail29
def root619Tail27 : List Nat := 15661 :: root619Tail28
def root619Tail26 : List Nat := 15331 :: root619Tail27
def root619Tail25 : List Nat := 15259 :: root619Tail26
def root619Tail24 : List Nat := 14197 :: root619Tail25
def root619Tail23 : List Nat := 14071 :: root619Tail24
def root619Tail22 : List Nat := 12073 :: root619Tail23
def root619Tail21 : List Nat := 11779 :: root619Tail22
def root619Tail20 : List Nat := 11437 :: root619Tail21
def root619Tail19 : List Nat := 11383 :: root619Tail20
def root619Tail18 : List Nat := 9931 :: root619Tail19
def root619Tail17 : List Nat := 9907 :: root619Tail18
def root619Tail16 : List Nat := 9403 :: root619Tail17
def root619Tail15 : List Nat := 9397 :: root619Tail16
def root619Tail14 : List Nat := 9151 :: root619Tail15
def root619Tail13 : List Nat := 8923 :: root619Tail14
def root619Tail12 : List Nat := 8389 :: root619Tail13
def root619Tail11 : List Nat := 8293 :: root619Tail12
def root619Tail10 : List Nat := 8263 :: root619Tail11
def root619Tail9 : List Nat := 8011 :: root619Tail10
def root619Tail8 : List Nat := 7993 :: root619Tail9
def root619Tail7 : List Nat := 7213 :: root619Tail8
def root619Tail6 : List Nat := 6079 :: root619Tail7
def root619Tail5 : List Nat := 6043 :: root619Tail6
def root619Tail4 : List Nat := 5749 :: root619Tail5
def root619Tail3 : List Nat := 5581 :: root619Tail4
def root619Tail2 : List Nat := 5563 :: root619Tail3
def root619Tail1 : List Nat := 5113 :: root619Tail2
def root619Tail0 : List Nat := 4969 :: root619Tail1
def root619Candidates0 : List Nat := [8923, 11383]
set_option maxRecDepth 8192 in
theorem root619Candidates0_checked : candidates edge 4 4597 4969 root619Tail1 winner.bound = root619Candidates0 := by
  unfold candidates
  have row : edge 4969 = (SortedNatTree.contains FastRows.row4969) := by
    change Data.row4969.contains = (SortedNatTree.contains FastRows.row4969)
    exact FastRows.row4969_checked.symm
  rw [row]
  decide
theorem root619Step0 : scan edge (search edge 3) 4 [4597] 4597 root619Tail0 winner =
    scan edge (search edge 3) 4 [4597] 4597 root619Tail1 winner := by
  rw [root619Tail0, scan, if_neg (by decide), root619Candidates0_checked]
  change scan edge (search edge 3) 4 [4597] 4597 root619Tail1
    (if 3 ≤ root619Candidates0.length then search edge 3 [4597, 4969] root619Candidates0 9566 winner else winner) = _
  rw [if_neg (by decide)]
def root619Candidates1 : List Nat := [8293]
set_option maxRecDepth 8192 in
theorem root619Candidates1_checked : candidates edge 4 4597 5113 root619Tail2 winner.bound = root619Candidates1 := by
  unfold candidates
  have row : edge 5113 = (SortedNatTree.contains FastRows.row5113) := by
    change Data.row5113.contains = (SortedNatTree.contains FastRows.row5113)
    exact FastRows.row5113_checked.symm
  rw [row]
  decide
theorem root619Step1 : scan edge (search edge 3) 4 [4597] 4597 root619Tail1 winner =
    scan edge (search edge 3) 4 [4597] 4597 root619Tail2 winner := by
  rw [root619Tail1, scan, if_neg (by decide), root619Candidates1_checked]
  change scan edge (search edge 3) 4 [4597] 4597 root619Tail2
    (if 3 ≤ root619Candidates1.length then search edge 3 [4597, 5113] root619Candidates1 9710 winner else winner) = _
  rw [if_neg (by decide)]
theorem root619Scan2 : scan edge (search edge 3) 4 [4597] 4597 root619Tail2 winner = winner := by rw [root619Tail2, scan, if_pos (by decide)]
theorem root619Scan1 : scan edge (search edge 3) 4 [4597] 4597 root619Tail1 winner = winner :=
  (root619Step1).trans root619Scan2
theorem root619Scan0 : scan edge (search edge 3) 4 [4597] 4597 root619Tail0 winner = winner :=
  (root619Step0).trans root619Scan1
set_option maxRecDepth 4096 in
theorem root619Branch : search edge 4 [4597] root619Candidates 4597 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4597] 4597 root619Tail0 winner = winner
  exact root619Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
