import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root565Candidates : List Nat := Data.row4129
set_option maxRecDepth 8192 in
theorem root565Candidates_checked : candidates edge 5 0 4129 tail566 winner.bound = root565Candidates := by
  apply candidates_eq_row edge 5 0 4129 tail566 root565Candidates winner.bound tail566_ordered
  · decide
  · decide
  · funext q
    rfl
def root565Tail40 : List Nat := []
def root565Tail39 : List Nat := 21841 :: root565Tail40
def root565Tail38 : List Nat := 21559 :: root565Tail39
def root565Tail37 : List Nat := 21031 :: root565Tail38
def root565Tail36 : List Nat := 20857 :: root565Tail37
def root565Tail35 : List Nat := 19471 :: root565Tail36
def root565Tail34 : List Nat := 18439 :: root565Tail35
def root565Tail33 : List Nat := 18061 :: root565Tail34
def root565Tail32 : List Nat := 18043 :: root565Tail33
def root565Tail31 : List Nat := 17389 :: root565Tail32
def root565Tail30 : List Nat := 17317 :: root565Tail31
def root565Tail29 : List Nat := 17203 :: root565Tail30
def root565Tail28 : List Nat := 17107 :: root565Tail29
def root565Tail27 : List Nat := 15991 :: root565Tail28
def root565Tail26 : List Nat := 15859 :: root565Tail27
def root565Tail25 : List Nat := 14533 :: root565Tail26
def root565Tail24 : List Nat := 14431 :: root565Tail25
def root565Tail23 : List Nat := 13003 :: root565Tail24
def root565Tail22 : List Nat := 12757 :: root565Tail23
def root565Tail21 : List Nat := 12457 :: root565Tail22
def root565Tail20 : List Nat := 12373 :: root565Tail21
def root565Tail19 : List Nat := 11833 :: root565Tail20
def root565Tail18 : List Nat := 11353 :: root565Tail19
def root565Tail17 : List Nat := 11131 :: root565Tail18
def root565Tail16 : List Nat := 11071 :: root565Tail17
def root565Tail15 : List Nat := 10723 :: root565Tail16
def root565Tail14 : List Nat := 9397 :: root565Tail15
def root565Tail13 : List Nat := 9343 :: root565Tail14
def root565Tail12 : List Nat := 9091 :: root565Tail13
def root565Tail11 : List Nat := 8377 :: root565Tail12
def root565Tail10 : List Nat := 8221 :: root565Tail11
def root565Tail9 : List Nat := 8167 :: root565Tail10
def root565Tail8 : List Nat := 7717 :: root565Tail9
def root565Tail7 : List Nat := 7159 :: root565Tail8
def root565Tail6 : List Nat := 6949 :: root565Tail7
def root565Tail5 : List Nat := 6781 :: root565Tail6
def root565Tail4 : List Nat := 6361 :: root565Tail5
def root565Tail3 : List Nat := 6271 :: root565Tail4
def root565Tail2 : List Nat := 5521 :: root565Tail3
def root565Tail1 : List Nat := 4861 :: root565Tail2
def root565Tail0 : List Nat := 4651 :: root565Tail1
def root565Candidates0 : List Nat := [9091]
set_option maxRecDepth 8192 in
theorem root565Candidates0_checked : candidates edge 4 4129 4651 root565Tail1 winner.bound = root565Candidates0 := by
  unfold candidates
  have row : edge 4651 = (SortedNatTree.contains FastRows.row4651) := by
    change Data.row4651.contains = (SortedNatTree.contains FastRows.row4651)
    exact FastRows.row4651_checked.symm
  rw [row]
  decide
theorem root565Step0 : scan edge (search edge 3) 4 [4129] 4129 root565Tail0 winner =
    scan edge (search edge 3) 4 [4129] 4129 root565Tail1 winner := by
  rw [root565Tail0, scan, if_neg (by decide), root565Candidates0_checked]
  change scan edge (search edge 3) 4 [4129] 4129 root565Tail1
    (if 3 ≤ root565Candidates0.length then search edge 3 [4129, 4651] root565Candidates0 8780 winner else winner) = _
  rw [if_neg (by decide)]
def root565Candidates1 : List Nat := [6361]
set_option maxRecDepth 8192 in
theorem root565Candidates1_checked : candidates edge 4 4129 4861 root565Tail2 winner.bound = root565Candidates1 := by
  unfold candidates
  have row : edge 4861 = (SortedNatTree.contains FastRows.row4861) := by
    change Data.row4861.contains = (SortedNatTree.contains FastRows.row4861)
    exact FastRows.row4861_checked.symm
  rw [row]
  decide
theorem root565Step1 : scan edge (search edge 3) 4 [4129] 4129 root565Tail1 winner =
    scan edge (search edge 3) 4 [4129] 4129 root565Tail2 winner := by
  rw [root565Tail1, scan, if_neg (by decide), root565Candidates1_checked]
  change scan edge (search edge 3) 4 [4129] 4129 root565Tail2
    (if 3 ≤ root565Candidates1.length then search edge 3 [4129, 4861] root565Candidates1 8990 winner else winner) = _
  rw [if_neg (by decide)]
theorem root565Scan2 : scan edge (search edge 3) 4 [4129] 4129 root565Tail2 winner = winner := by rw [root565Tail2, scan, if_pos (by decide)]
theorem root565Scan1 : scan edge (search edge 3) 4 [4129] 4129 root565Tail1 winner = winner :=
  (root565Step1).trans root565Scan2
theorem root565Scan0 : scan edge (search edge 3) 4 [4129] 4129 root565Tail0 winner = winner :=
  (root565Step0).trans root565Scan1
set_option maxRecDepth 4096 in
theorem root565Branch : search edge 4 [4129] root565Candidates 4129 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4129] 4129 root565Tail0 winner = winner
  exact root565Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
