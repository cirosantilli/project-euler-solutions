import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root597Candidates : List Nat := Data.row4409
set_option maxRecDepth 8192 in
theorem root597Candidates_checked : candidates edge 5 0 4409 tail598 winner.bound = root597Candidates := by
  apply candidates_eq_row edge 5 0 4409 tail598 root597Candidates winner.bound tail598_ordered
  · decide
  · decide
  · funext q
    rfl
def root597Tail39 : List Nat := []
def root597Tail38 : List Nat := 21347 :: root597Tail39
def root597Tail37 : List Nat := 20021 :: root597Tail38
def root597Tail36 : List Nat := 19793 :: root597Tail37
def root597Tail35 : List Nat := 19709 :: root597Tail36
def root597Tail34 : List Nat := 18743 :: root597Tail35
def root597Tail33 : List Nat := 18719 :: root597Tail34
def root597Tail32 : List Nat := 18521 :: root597Tail33
def root597Tail31 : List Nat := 18311 :: root597Tail32
def root597Tail30 : List Nat := 18287 :: root597Tail31
def root597Tail29 : List Nat := 16253 :: root597Tail30
def root597Tail28 : List Nat := 16103 :: root597Tail29
def root597Tail27 : List Nat := 16061 :: root597Tail28
def root597Tail26 : List Nat := 15161 :: root597Tail27
def root597Tail25 : List Nat := 14657 :: root597Tail26
def root597Tail24 : List Nat := 14153 :: root597Tail25
def root597Tail23 : List Nat := 13997 :: root597Tail24
def root597Tail22 : List Nat := 13523 :: root597Tail23
def root597Tail21 : List Nat := 13469 :: root597Tail22
def root597Tail20 : List Nat := 13331 :: root597Tail21
def root597Tail19 : List Nat := 12611 :: root597Tail20
def root597Tail18 : List Nat := 11987 :: root597Tail19
def root597Tail17 : List Nat := 11483 :: root597Tail18
def root597Tail16 : List Nat := 10937 :: root597Tail17
def root597Tail15 : List Nat := 10259 :: root597Tail16
def root597Tail14 : List Nat := 9791 :: root597Tail15
def root597Tail13 : List Nat := 9689 :: root597Tail14
def root597Tail12 : List Nat := 9413 :: root597Tail13
def root597Tail11 : List Nat := 9371 :: root597Tail12
def root597Tail10 : List Nat := 9239 :: root597Tail11
def root597Tail9 : List Nat := 9059 :: root597Tail10
def root597Tail8 : List Nat := 8447 :: root597Tail9
def root597Tail7 : List Nat := 7253 :: root597Tail8
def root597Tail6 : List Nat := 6863 :: root597Tail7
def root597Tail5 : List Nat := 6737 :: root597Tail6
def root597Tail4 : List Nat := 6719 :: root597Tail5
def root597Tail3 : List Nat := 5867 :: root597Tail4
def root597Tail2 : List Nat := 5477 :: root597Tail3
def root597Tail1 : List Nat := 5279 :: root597Tail2
def root597Tail0 : List Nat := 5087 :: root597Tail1
def root597Candidates0 : List Nat := [6719]
set_option maxRecDepth 8192 in
theorem root597Candidates0_checked : candidates edge 4 4409 5087 root597Tail1 winner.bound = root597Candidates0 := by
  unfold candidates
  have row : edge 5087 = (SortedNatTree.contains FastRows.row5087) := by
    change Data.row5087.contains = (SortedNatTree.contains FastRows.row5087)
    exact FastRows.row5087_checked.symm
  rw [row]
  decide
theorem root597Step0 : scan edge (search edge 3) 4 [4409] 4409 root597Tail0 winner =
    scan edge (search edge 3) 4 [4409] 4409 root597Tail1 winner := by
  rw [root597Tail0, scan, if_neg (by decide), root597Candidates0_checked]
  change scan edge (search edge 3) 4 [4409] 4409 root597Tail1
    (if 3 ≤ root597Candidates0.length then search edge 3 [4409, 5087] root597Candidates0 9496 winner else winner) = _
  rw [if_neg (by decide)]
def root597Candidates1 : List Nat := [6719, 7253]
set_option maxRecDepth 8192 in
theorem root597Candidates1_checked : candidates edge 4 4409 5279 root597Tail2 winner.bound = root597Candidates1 := by
  unfold candidates
  have row : edge 5279 = (SortedNatTree.contains FastRows.row5279) := by
    change Data.row5279.contains = (SortedNatTree.contains FastRows.row5279)
    exact FastRows.row5279_checked.symm
  rw [row]
  decide
theorem root597Step1 : scan edge (search edge 3) 4 [4409] 4409 root597Tail1 winner =
    scan edge (search edge 3) 4 [4409] 4409 root597Tail2 winner := by
  rw [root597Tail1, scan, if_neg (by decide), root597Candidates1_checked]
  change scan edge (search edge 3) 4 [4409] 4409 root597Tail2
    (if 3 ≤ root597Candidates1.length then search edge 3 [4409, 5279] root597Candidates1 9688 winner else winner) = _
  rw [if_neg (by decide)]
theorem root597Scan2 : scan edge (search edge 3) 4 [4409] 4409 root597Tail2 winner = winner := by rw [root597Tail2, scan, if_pos (by decide)]
theorem root597Scan1 : scan edge (search edge 3) 4 [4409] 4409 root597Tail1 winner = winner :=
  (root597Step1).trans root597Scan2
theorem root597Scan0 : scan edge (search edge 3) 4 [4409] 4409 root597Tail0 winner = winner :=
  (root597Step0).trans root597Scan1
set_option maxRecDepth 4096 in
theorem root597Branch : search edge 4 [4409] root597Candidates 4409 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4409] 4409 root597Tail0 winner = winner
  exact root597Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
