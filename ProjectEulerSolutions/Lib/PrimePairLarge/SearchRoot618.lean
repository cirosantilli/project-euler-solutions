import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root618Candidates : List Nat := Data.row4591
set_option maxRecDepth 8192 in
theorem root618Candidates_checked : candidates edge 5 0 4591 tail619 winner.bound = root618Candidates := by
  apply candidates_eq_row edge 5 0 4591 tail619 root618Candidates winner.bound tail619_ordered
  · decide
  · decide
  · funext q
    rfl
def root618Tail34 : List Nat := []
def root618Tail33 : List Nat := 20611 :: root618Tail34
def root618Tail32 : List Nat := 20443 :: root618Tail33
def root618Tail31 : List Nat := 18493 :: root618Tail32
def root618Tail30 : List Nat := 18367 :: root618Tail31
def root618Tail29 : List Nat := 17851 :: root618Tail30
def root618Tail28 : List Nat := 17707 :: root618Tail29
def root618Tail27 : List Nat := 17683 :: root618Tail28
def root618Tail26 : List Nat := 17443 :: root618Tail27
def root618Tail25 : List Nat := 17359 :: root618Tail26
def root618Tail24 : List Nat := 14797 :: root618Tail25
def root618Tail23 : List Nat := 14011 :: root618Tail24
def root618Tail22 : List Nat := 12739 :: root618Tail23
def root618Tail21 : List Nat := 12487 :: root618Tail22
def root618Tail20 : List Nat := 12109 :: root618Tail21
def root618Tail19 : List Nat := 11833 :: root618Tail20
def root618Tail18 : List Nat := 11353 :: root618Tail19
def root618Tail17 : List Nat := 11239 :: root618Tail18
def root618Tail16 : List Nat := 10903 :: root618Tail17
def root618Tail15 : List Nat := 10627 :: root618Tail16
def root618Tail14 : List Nat := 9931 :: root618Tail15
def root618Tail13 : List Nat := 9883 :: root618Tail14
def root618Tail12 : List Nat := 9007 :: root618Tail13
def root618Tail11 : List Nat := 9001 :: root618Tail12
def root618Tail10 : List Nat := 8329 :: root618Tail11
def root618Tail9 : List Nat := 8293 :: root618Tail10
def root618Tail8 : List Nat := 8287 :: root618Tail9
def root618Tail7 : List Nat := 8167 :: root618Tail8
def root618Tail6 : List Nat := 7621 :: root618Tail7
def root618Tail5 : List Nat := 7369 :: root618Tail6
def root618Tail4 : List Nat := 7213 :: root618Tail5
def root618Tail3 : List Nat := 6121 :: root618Tail4
def root618Tail2 : List Nat := 5197 :: root618Tail3
def root618Tail1 : List Nat := 5101 :: root618Tail2
def root618Tail0 : List Nat := 4651 :: root618Tail1
def root618Candidates0 : List Nat := [8293]
set_option maxRecDepth 8192 in
theorem root618Candidates0_checked : candidates edge 4 4591 4651 root618Tail1 winner.bound = root618Candidates0 := by
  unfold candidates
  have row : edge 4651 = (SortedNatTree.contains FastRows.row4651) := by
    change Data.row4651.contains = (SortedNatTree.contains FastRows.row4651)
    exact FastRows.row4651_checked.symm
  rw [row]
  decide
theorem root618Step0 : scan edge (search edge 3) 4 [4591] 4591 root618Tail0 winner =
    scan edge (search edge 3) 4 [4591] 4591 root618Tail1 winner := by
  rw [root618Tail0, scan, if_neg (by decide), root618Candidates0_checked]
  change scan edge (search edge 3) 4 [4591] 4591 root618Tail1
    (if 3 ≤ root618Candidates0.length then search edge 3 [4591, 4651] root618Candidates0 9242 winner else winner) = _
  rw [if_neg (by decide)]
def root618Candidates1 : List Nat := [8293, 9931]
set_option maxRecDepth 8192 in
theorem root618Candidates1_checked : candidates edge 4 4591 5101 root618Tail2 winner.bound = root618Candidates1 := by
  unfold candidates
  have row : edge 5101 = (SortedNatTree.contains FastRows.row5101) := by
    change Data.row5101.contains = (SortedNatTree.contains FastRows.row5101)
    exact FastRows.row5101_checked.symm
  rw [row]
  decide
theorem root618Step1 : scan edge (search edge 3) 4 [4591] 4591 root618Tail1 winner =
    scan edge (search edge 3) 4 [4591] 4591 root618Tail2 winner := by
  rw [root618Tail1, scan, if_neg (by decide), root618Candidates1_checked]
  change scan edge (search edge 3) 4 [4591] 4591 root618Tail2
    (if 3 ≤ root618Candidates1.length then search edge 3 [4591, 5101] root618Candidates1 9692 winner else winner) = _
  rw [if_neg (by decide)]
def root618Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root618Candidates2_checked : candidates edge 4 4591 5197 root618Tail3 winner.bound = root618Candidates2 := by
  unfold candidates
  have row : edge 5197 = (SortedNatTree.contains FastRows.row5197) := by
    change Data.row5197.contains = (SortedNatTree.contains FastRows.row5197)
    exact FastRows.row5197_checked.symm
  rw [row]
  decide
theorem root618Step2 : scan edge (search edge 3) 4 [4591] 4591 root618Tail2 winner =
    scan edge (search edge 3) 4 [4591] 4591 root618Tail3 winner := by
  rw [root618Tail2, scan, if_neg (by decide), root618Candidates2_checked]
  change scan edge (search edge 3) 4 [4591] 4591 root618Tail3
    (if 3 ≤ root618Candidates2.length then search edge 3 [4591, 5197] root618Candidates2 9788 winner else winner) = _
  rw [if_neg (by decide)]
theorem root618Scan3 : scan edge (search edge 3) 4 [4591] 4591 root618Tail3 winner = winner := by rw [root618Tail3, scan, if_pos (by decide)]
theorem root618Scan2 : scan edge (search edge 3) 4 [4591] 4591 root618Tail2 winner = winner :=
  (root618Step2).trans root618Scan3
theorem root618Scan1 : scan edge (search edge 3) 4 [4591] 4591 root618Tail1 winner = winner :=
  (root618Step1).trans root618Scan2
theorem root618Scan0 : scan edge (search edge 3) 4 [4591] 4591 root618Tail0 winner = winner :=
  (root618Step0).trans root618Scan1
set_option maxRecDepth 4096 in
theorem root618Branch : search edge 4 [4591] root618Candidates 4591 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4591] 4591 root618Tail0 winner = winner
  exact root618Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
