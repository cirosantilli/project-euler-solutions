import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root518Candidates : List Nat := Data.row3733
set_option maxRecDepth 8192 in
theorem root518Candidates_checked : candidates edge 5 0 3733 tail519 winner.bound = root518Candidates := by
  apply candidates_eq_row edge 5 0 3733 tail519 root518Candidates winner.bound tail519_ordered
  · decide
  · decide
  · funext q
    rfl
def root518Tail33 : List Nat := []
def root518Tail32 : List Nat := 22273 :: root518Tail33
def root518Tail31 : List Nat := 21121 :: root518Tail32
def root518Tail30 : List Nat := 20641 :: root518Tail31
def root518Tail29 : List Nat := 17257 :: root518Tail30
def root518Tail28 : List Nat := 16981 :: root518Tail29
def root518Tail27 : List Nat := 16231 :: root518Tail28
def root518Tail26 : List Nat := 15973 :: root518Tail27
def root518Tail25 : List Nat := 15889 :: root518Tail26
def root518Tail24 : List Nat := 15643 :: root518Tail25
def root518Tail23 : List Nat := 15541 :: root518Tail24
def root518Tail22 : List Nat := 15091 :: root518Tail23
def root518Tail21 : List Nat := 14629 :: root518Tail22
def root518Tail20 : List Nat := 13399 :: root518Tail21
def root518Tail19 : List Nat := 13309 :: root518Tail20
def root518Tail18 : List Nat := 12697 :: root518Tail19
def root518Tail17 : List Nat := 12097 :: root518Tail18
def root518Tail16 : List Nat := 11971 :: root518Tail17
def root518Tail15 : List Nat := 11941 :: root518Tail16
def root518Tail14 : List Nat := 11779 :: root518Tail15
def root518Tail13 : List Nat := 10753 :: root518Tail14
def root518Tail12 : List Nat := 10627 :: root518Tail13
def root518Tail11 : List Nat := 10513 :: root518Tail12
def root518Tail10 : List Nat := 10159 :: root518Tail11
def root518Tail9 : List Nat := 9349 :: root518Tail10
def root518Tail8 : List Nat := 8461 :: root518Tail9
def root518Tail7 : List Nat := 8089 :: root518Tail8
def root518Tail6 : List Nat := 7687 :: root518Tail7
def root518Tail5 : List Nat := 6529 :: root518Tail6
def root518Tail4 : List Nat := 6451 :: root518Tail5
def root518Tail3 : List Nat := 6421 :: root518Tail4
def root518Tail2 : List Nat := 6151 :: root518Tail3
def root518Tail1 : List Nat := 5521 :: root518Tail2
def root518Tail0 : List Nat := 4657 :: root518Tail1
def root518Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root518Candidates0_checked : candidates edge 4 3733 4657 root518Tail1 winner.bound = root518Candidates0 := by
  unfold candidates
  have row : edge 4657 = (SortedNatTree.contains FastRows.row4657) := by
    change Data.row4657.contains = (SortedNatTree.contains FastRows.row4657)
    exact FastRows.row4657_checked.symm
  rw [row]
  decide
theorem root518Step0 : scan edge (search edge 3) 4 [3733] 3733 root518Tail0 winner =
    scan edge (search edge 3) 4 [3733] 3733 root518Tail1 winner := by
  rw [root518Tail0, scan, if_neg (by decide), root518Candidates0_checked]
  change scan edge (search edge 3) 4 [3733] 3733 root518Tail1
    (if 3 ≤ root518Candidates0.length then search edge 3 [3733, 4657] root518Candidates0 8390 winner else winner) = _
  rw [if_neg (by decide)]
def root518Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root518Candidates1_checked : candidates edge 4 3733 5521 root518Tail2 winner.bound = root518Candidates1 := by
  unfold candidates
  have row : edge 5521 = (SortedNatTree.contains FastRows.row5521) := by
    change Data.row5521.contains = (SortedNatTree.contains FastRows.row5521)
    exact FastRows.row5521_checked.symm
  rw [row]
  decide
theorem root518Step1 : scan edge (search edge 3) 4 [3733] 3733 root518Tail1 winner =
    scan edge (search edge 3) 4 [3733] 3733 root518Tail2 winner := by
  rw [root518Tail1, scan, if_neg (by decide), root518Candidates1_checked]
  change scan edge (search edge 3) 4 [3733] 3733 root518Tail2
    (if 3 ≤ root518Candidates1.length then search edge 3 [3733, 5521] root518Candidates1 9254 winner else winner) = _
  rw [if_neg (by decide)]
theorem root518Scan2 : scan edge (search edge 3) 4 [3733] 3733 root518Tail2 winner = winner := by rw [root518Tail2, scan, if_pos (by decide)]
theorem root518Scan1 : scan edge (search edge 3) 4 [3733] 3733 root518Tail1 winner = winner :=
  (root518Step1).trans root518Scan2
theorem root518Scan0 : scan edge (search edge 3) 4 [3733] 3733 root518Tail0 winner = winner :=
  (root518Step0).trans root518Scan1
set_option maxRecDepth 4096 in
theorem root518Branch : search edge 4 [3733] root518Candidates 3733 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3733] 3733 root518Tail0 winner = winner
  exact root518Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
