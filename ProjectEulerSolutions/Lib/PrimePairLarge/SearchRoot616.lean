import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root616Candidates : List Nat := Data.row4567
set_option maxRecDepth 8192 in
theorem root616Candidates_checked : candidates edge 5 0 4567 tail617 winner.bound = root616Candidates := by
  apply candidates_eq_row edge 5 0 4567 tail617 root616Candidates winner.bound tail617_ordered
  · decide
  · decide
  · funext q
    rfl
def root616Tail32 : List Nat := []
def root616Tail31 : List Nat := 20809 :: root616Tail32
def root616Tail30 : List Nat := 20533 :: root616Tail31
def root616Tail29 : List Nat := 19507 :: root616Tail30
def root616Tail28 : List Nat := 19429 :: root616Tail29
def root616Tail27 : List Nat := 19381 :: root616Tail28
def root616Tail26 : List Nat := 19141 :: root616Tail27
def root616Tail25 : List Nat := 18517 :: root616Tail26
def root616Tail24 : List Nat := 18433 :: root616Tail25
def root616Tail23 : List Nat := 17467 :: root616Tail24
def root616Tail22 : List Nat := 17029 :: root616Tail23
def root616Tail21 : List Nat := 16417 :: root616Tail22
def root616Tail20 : List Nat := 16369 :: root616Tail21
def root616Tail19 : List Nat := 16189 :: root616Tail20
def root616Tail18 : List Nat := 15859 :: root616Tail19
def root616Tail17 : List Nat := 15823 :: root616Tail18
def root616Tail16 : List Nat := 15073 :: root616Tail17
def root616Tail15 : List Nat := 14779 :: root616Tail16
def root616Tail14 : List Nat := 14737 :: root616Tail15
def root616Tail13 : List Nat := 14437 :: root616Tail14
def root616Tail12 : List Nat := 12829 :: root616Tail13
def root616Tail11 : List Nat := 10861 :: root616Tail12
def root616Tail10 : List Nat := 10399 :: root616Tail11
def root616Tail9 : List Nat := 9643 :: root616Tail10
def root616Tail8 : List Nat := 9091 :: root616Tail9
def root616Tail7 : List Nat := 8719 :: root616Tail8
def root616Tail6 : List Nat := 8467 :: root616Tail7
def root616Tail5 : List Nat := 7993 :: root616Tail6
def root616Tail4 : List Nat := 7789 :: root616Tail5
def root616Tail3 : List Nat := 7417 :: root616Tail4
def root616Tail2 : List Nat := 7039 :: root616Tail3
def root616Tail1 : List Nat := 5557 :: root616Tail2
def root616Tail0 : List Nat := 5323 :: root616Tail1
def root616Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root616Candidates0_checked : candidates edge 4 4567 5323 root616Tail1 winner.bound = root616Candidates0 := by
  unfold candidates
  have row : edge 5323 = (SortedNatTree.contains FastRows.row5323) := by
    change Data.row5323.contains = (SortedNatTree.contains FastRows.row5323)
    exact FastRows.row5323_checked.symm
  rw [row]
  decide
theorem root616Step0 : scan edge (search edge 3) 4 [4567] 4567 root616Tail0 winner =
    scan edge (search edge 3) 4 [4567] 4567 root616Tail1 winner := by
  rw [root616Tail0, scan, if_neg (by decide), root616Candidates0_checked]
  change scan edge (search edge 3) 4 [4567] 4567 root616Tail1
    (if 3 ≤ root616Candidates0.length then search edge 3 [4567, 5323] root616Candidates0 9890 winner else winner) = _
  rw [if_neg (by decide)]
theorem root616Scan1 : scan edge (search edge 3) 4 [4567] 4567 root616Tail1 winner = winner := by rw [root616Tail1, scan, if_pos (by decide)]
theorem root616Scan0 : scan edge (search edge 3) 4 [4567] 4567 root616Tail0 winner = winner :=
  (root616Step0).trans root616Scan1
set_option maxRecDepth 4096 in
theorem root616Branch : search edge 4 [4567] root616Candidates 4567 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4567] 4567 root616Tail0 winner = winner
  exact root616Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
