import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root633Candidates : List Nat := Data.row4721
set_option maxRecDepth 8192 in
theorem root633Candidates_checked : candidates edge 5 0 4721 tail634 winner.bound = root633Candidates := by
  apply candidates_eq_row edge 5 0 4721 tail634 root633Candidates winner.bound tail634_ordered
  · decide
  · decide
  · funext q
    rfl
def root633Tail25 : List Nat := []
def root633Tail24 : List Nat := 20921 :: root633Tail25
def root633Tail23 : List Nat := 20507 :: root633Tail24
def root633Tail22 : List Nat := 19997 :: root633Tail23
def root633Tail21 : List Nat := 19031 :: root633Tail22
def root633Tail20 : List Nat := 18287 :: root633Tail21
def root633Tail19 : List Nat := 17483 :: root633Tail20
def root633Tail18 : List Nat := 15731 :: root633Tail19
def root633Tail17 : List Nat := 14891 :: root633Tail18
def root633Tail16 : List Nat := 14843 :: root633Tail17
def root633Tail15 : List Nat := 14627 :: root633Tail16
def root633Tail14 : List Nat := 13751 :: root633Tail15
def root633Tail13 : List Nat := 12269 :: root633Tail14
def root633Tail12 : List Nat := 10883 :: root633Tail13
def root633Tail11 : List Nat := 10589 :: root633Tail12
def root633Tail10 : List Nat := 9803 :: root633Tail11
def root633Tail9 : List Nat := 9257 :: root633Tail10
def root633Tail8 : List Nat := 9209 :: root633Tail9
def root633Tail7 : List Nat := 8753 :: root633Tail8
def root633Tail6 : List Nat := 7949 :: root633Tail7
def root633Tail5 : List Nat := 6971 :: root633Tail6
def root633Tail4 : List Nat := 5813 :: root633Tail5
def root633Tail3 : List Nat := 5501 :: root633Tail4
def root633Tail2 : List Nat := 5417 :: root633Tail3
def root633Tail1 : List Nat := 4919 :: root633Tail2
def root633Tail0 : List Nat := 4877 :: root633Tail1
def root633Candidates0 : List Nat := [5501, 5813, 14891]
set_option maxRecDepth 8192 in
theorem root633Candidates0_checked : candidates edge 4 4721 4877 root633Tail1 winner.bound = root633Candidates0 := by
  unfold candidates
  have row : edge 4877 = (SortedNatTree.contains FastRows.row4877) := by
    change Data.row4877.contains = (SortedNatTree.contains FastRows.row4877)
    exact FastRows.row4877_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root633Child0 : search edge 3 [4721, 4877] root633Candidates0 9598 winner = winner := by decide
theorem root633Step0 : scan edge (search edge 3) 4 [4721] 4721 root633Tail0 winner =
    scan edge (search edge 3) 4 [4721] 4721 root633Tail1 winner := by
  rw [root633Tail0, scan, if_neg (by decide), root633Candidates0_checked]
  change scan edge (search edge 3) 4 [4721] 4721 root633Tail1
    (if 3 ≤ root633Candidates0.length then search edge 3 [4721, 4877] root633Candidates0 9598 winner else winner) = _
  rw [if_pos (by decide), root633Child0]
def root633Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root633Candidates1_checked : candidates edge 4 4721 4919 root633Tail2 winner.bound = root633Candidates1 := by
  unfold candidates
  have row : edge 4919 = (SortedNatTree.contains FastRows.row4919) := by
    change Data.row4919.contains = (SortedNatTree.contains FastRows.row4919)
    exact FastRows.row4919_checked.symm
  rw [row]
  decide
theorem root633Step1 : scan edge (search edge 3) 4 [4721] 4721 root633Tail1 winner =
    scan edge (search edge 3) 4 [4721] 4721 root633Tail2 winner := by
  rw [root633Tail1, scan, if_neg (by decide), root633Candidates1_checked]
  change scan edge (search edge 3) 4 [4721] 4721 root633Tail2
    (if 3 ≤ root633Candidates1.length then search edge 3 [4721, 4919] root633Candidates1 9640 winner else winner) = _
  rw [if_neg (by decide)]
theorem root633Scan2 : scan edge (search edge 3) 4 [4721] 4721 root633Tail2 winner = winner := by rw [root633Tail2, scan, if_pos (by decide)]
theorem root633Scan1 : scan edge (search edge 3) 4 [4721] 4721 root633Tail1 winner = winner :=
  (root633Step1).trans root633Scan2
theorem root633Scan0 : scan edge (search edge 3) 4 [4721] 4721 root633Tail0 winner = winner :=
  (root633Step0).trans root633Scan1
set_option maxRecDepth 4096 in
theorem root633Branch : search edge 4 [4721] root633Candidates 4721 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4721] 4721 root633Tail0 winner = winner
  exact root633Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
