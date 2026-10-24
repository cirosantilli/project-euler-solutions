import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root639Candidates : List Nat := Data.row4783
set_option maxRecDepth 8192 in
theorem root639Candidates_checked : candidates edge 5 0 4783 tail640 winner.bound = root639Candidates := by
  apply candidates_eq_row edge 5 0 4783 tail640 root639Candidates winner.bound tail640_ordered
  · decide
  · decide
  · funext q
    rfl
def root639Tail31 : List Nat := []
def root639Tail30 : List Nat := 20641 :: root639Tail31
def root639Tail29 : List Nat := 19843 :: root639Tail30
def root639Tail28 : List Nat := 18793 :: root639Tail29
def root639Tail27 : List Nat := 18661 :: root639Tail28
def root639Tail26 : List Nat := 18583 :: root639Tail27
def root639Tail25 : List Nat := 18439 :: root639Tail26
def root639Tail24 : List Nat := 18397 :: root639Tail25
def root639Tail23 : List Nat := 18223 :: root639Tail24
def root639Tail22 : List Nat := 17971 :: root639Tail23
def root639Tail21 : List Nat := 17911 :: root639Tail22
def root639Tail20 : List Nat := 16087 :: root639Tail21
def root639Tail19 : List Nat := 15601 :: root639Tail20
def root639Tail18 : List Nat := 15493 :: root639Tail19
def root639Tail17 : List Nat := 14293 :: root639Tail18
def root639Tail16 : List Nat := 14251 :: root639Tail17
def root639Tail15 : List Nat := 13963 :: root639Tail16
def root639Tail14 : List Nat := 11311 :: root639Tail15
def root639Tail13 : List Nat := 10303 :: root639Tail14
def root639Tail12 : List Nat := 9787 :: root639Tail13
def root639Tail11 : List Nat := 9613 :: root639Tail12
def root639Tail10 : List Nat := 9199 :: root639Tail11
def root639Tail9 : List Nat := 9049 :: root639Tail10
def root639Tail8 : List Nat := 8803 :: root639Tail9
def root639Tail7 : List Nat := 7879 :: root639Tail8
def root639Tail6 : List Nat := 7873 :: root639Tail7
def root639Tail5 : List Nat := 7219 :: root639Tail6
def root639Tail4 : List Nat := 6781 :: root639Tail5
def root639Tail3 : List Nat := 6073 :: root639Tail4
def root639Tail2 : List Nat := 5779 :: root639Tail3
def root639Tail1 : List Nat := 5167 :: root639Tail2
def root639Tail0 : List Nat := 4999 :: root639Tail1
def root639Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root639Candidates0_checked : candidates edge 4 4783 4999 root639Tail1 winner.bound = root639Candidates0 := by
  unfold candidates
  have row : edge 4999 = (SortedNatTree.contains FastRows.row4999) := by
    change Data.row4999.contains = (SortedNatTree.contains FastRows.row4999)
    exact FastRows.row4999_checked.symm
  rw [row]
  decide
theorem root639Step0 : scan edge (search edge 3) 4 [4783] 4783 root639Tail0 winner =
    scan edge (search edge 3) 4 [4783] 4783 root639Tail1 winner := by
  rw [root639Tail0, scan, if_neg (by decide), root639Candidates0_checked]
  change scan edge (search edge 3) 4 [4783] 4783 root639Tail1
    (if 3 ≤ root639Candidates0.length then search edge 3 [4783, 4999] root639Candidates0 9782 winner else winner) = _
  rw [if_neg (by decide)]
def root639Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root639Candidates1_checked : candidates edge 4 4783 5167 root639Tail2 winner.bound = root639Candidates1 := by
  unfold candidates
  have row : edge 5167 = (SortedNatTree.contains FastRows.row5167) := by
    change Data.row5167.contains = (SortedNatTree.contains FastRows.row5167)
    exact FastRows.row5167_checked.symm
  rw [row]
  decide
theorem root639Step1 : scan edge (search edge 3) 4 [4783] 4783 root639Tail1 winner =
    scan edge (search edge 3) 4 [4783] 4783 root639Tail2 winner := by
  rw [root639Tail1, scan, if_neg (by decide), root639Candidates1_checked]
  change scan edge (search edge 3) 4 [4783] 4783 root639Tail2
    (if 3 ≤ root639Candidates1.length then search edge 3 [4783, 5167] root639Candidates1 9950 winner else winner) = _
  rw [if_neg (by decide)]
theorem root639Scan2 : scan edge (search edge 3) 4 [4783] 4783 root639Tail2 winner = winner := by rw [root639Tail2, scan, if_pos (by decide)]
theorem root639Scan1 : scan edge (search edge 3) 4 [4783] 4783 root639Tail1 winner = winner :=
  (root639Step1).trans root639Scan2
theorem root639Scan0 : scan edge (search edge 3) 4 [4783] 4783 root639Tail0 winner = winner :=
  (root639Step0).trans root639Scan1
set_option maxRecDepth 4096 in
theorem root639Branch : search edge 4 [4783] root639Candidates 4783 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4783] 4783 root639Tail0 winner = winner
  exact root639Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
