import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root509Candidates : List Nat := Data.row3671
set_option maxRecDepth 8192 in
theorem root509Candidates_checked : candidates edge 5 0 3671 tail510 winner.bound = root509Candidates := by
  apply candidates_eq_row edge 5 0 3671 tail510 root509Candidates winner.bound tail510_ordered
  · decide
  · decide
  · funext q
    rfl
def root509Tail39 : List Nat := []
def root509Tail38 : List Nat := 22307 :: root509Tail39
def root509Tail37 : List Nat := 20123 :: root509Tail38
def root509Tail36 : List Nat := 19751 :: root509Tail37
def root509Tail35 : List Nat := 19073 :: root509Tail36
def root509Tail34 : List Nat := 18287 :: root509Tail35
def root509Tail33 : List Nat := 18041 :: root509Tail34
def root509Tail32 : List Nat := 17939 :: root509Tail33
def root509Tail31 : List Nat := 17483 :: root509Tail32
def root509Tail30 : List Nat := 17327 :: root509Tail31
def root509Tail29 : List Nat := 15461 :: root509Tail30
def root509Tail28 : List Nat := 15173 :: root509Tail29
def root509Tail27 : List Nat := 14537 :: root509Tail28
def root509Tail26 : List Nat := 13883 :: root509Tail27
def root509Tail25 : List Nat := 13337 :: root509Tail26
def root509Tail24 : List Nat := 12953 :: root509Tail25
def root509Tail23 : List Nat := 12437 :: root509Tail24
def root509Tail22 : List Nat := 12413 :: root509Tail23
def root509Tail21 : List Nat := 10301 :: root509Tail22
def root509Tail20 : List Nat := 10181 :: root509Tail21
def root509Tail19 : List Nat := 9929 :: root509Tail20
def root509Tail18 : List Nat := 9923 :: root509Tail19
def root509Tail17 : List Nat := 9623 :: root509Tail18
def root509Tail16 : List Nat := 9467 :: root509Tail17
def root509Tail15 : List Nat := 9461 :: root509Tail16
def root509Tail14 : List Nat := 9323 :: root509Tail15
def root509Tail13 : List Nat := 8501 :: root509Tail14
def root509Tail12 : List Nat := 8447 :: root509Tail13
def root509Tail11 : List Nat := 7853 :: root509Tail12
def root509Tail10 : List Nat := 7691 :: root509Tail11
def root509Tail9 : List Nat := 7523 :: root509Tail10
def root509Tail8 : List Nat := 7151 :: root509Tail9
def root509Tail7 : List Nat := 7019 :: root509Tail8
def root509Tail6 : List Nat := 6653 :: root509Tail7
def root509Tail5 : List Nat := 6131 :: root509Tail6
def root509Tail4 : List Nat := 6011 :: root509Tail5
def root509Tail3 : List Nat := 5849 :: root509Tail4
def root509Tail2 : List Nat := 5381 :: root509Tail3
def root509Tail1 : List Nat := 5039 :: root509Tail2
def root509Tail0 : List Nat := 4091 :: root509Tail1
def root509Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root509Candidates0_checked : candidates edge 4 3671 4091 root509Tail1 winner.bound = root509Candidates0 := by
  unfold candidates
  have row : edge 4091 = (SortedNatTree.contains FastRows.row4091) := by
    change Data.row4091.contains = (SortedNatTree.contains FastRows.row4091)
    exact FastRows.row4091_checked.symm
  rw [row]
  decide
theorem root509Step0 : scan edge (search edge 3) 4 [3671] 3671 root509Tail0 winner =
    scan edge (search edge 3) 4 [3671] 3671 root509Tail1 winner := by
  rw [root509Tail0, scan, if_neg (by decide), root509Candidates0_checked]
  change scan edge (search edge 3) 4 [3671] 3671 root509Tail1
    (if 3 ≤ root509Candidates0.length then search edge 3 [3671, 4091] root509Candidates0 7762 winner else winner) = _
  rw [if_neg (by decide)]
def root509Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root509Candidates1_checked : candidates edge 4 3671 5039 root509Tail2 winner.bound = root509Candidates1 := by
  unfold candidates
  have row : edge 5039 = (SortedNatTree.contains FastRows.row5039) := by
    change Data.row5039.contains = (SortedNatTree.contains FastRows.row5039)
    exact FastRows.row5039_checked.symm
  rw [row]
  decide
theorem root509Step1 : scan edge (search edge 3) 4 [3671] 3671 root509Tail1 winner =
    scan edge (search edge 3) 4 [3671] 3671 root509Tail2 winner := by
  rw [root509Tail1, scan, if_neg (by decide), root509Candidates1_checked]
  change scan edge (search edge 3) 4 [3671] 3671 root509Tail2
    (if 3 ≤ root509Candidates1.length then search edge 3 [3671, 5039] root509Candidates1 8710 winner else winner) = _
  rw [if_neg (by decide)]
def root509Candidates2 : List Nat := [6011, 9929]
set_option maxRecDepth 8192 in
theorem root509Candidates2_checked : candidates edge 4 3671 5381 root509Tail3 winner.bound = root509Candidates2 := by
  unfold candidates
  have row : edge 5381 = (SortedNatTree.contains FastRows.row5381) := by
    change Data.row5381.contains = (SortedNatTree.contains FastRows.row5381)
    exact FastRows.row5381_checked.symm
  rw [row]
  decide
theorem root509Step2 : scan edge (search edge 3) 4 [3671] 3671 root509Tail2 winner =
    scan edge (search edge 3) 4 [3671] 3671 root509Tail3 winner := by
  rw [root509Tail2, scan, if_neg (by decide), root509Candidates2_checked]
  change scan edge (search edge 3) 4 [3671] 3671 root509Tail3
    (if 3 ≤ root509Candidates2.length then search edge 3 [3671, 5381] root509Candidates2 9052 winner else winner) = _
  rw [if_neg (by decide)]
theorem root509Scan3 : scan edge (search edge 3) 4 [3671] 3671 root509Tail3 winner = winner := by rw [root509Tail3, scan, if_pos (by decide)]
theorem root509Scan2 : scan edge (search edge 3) 4 [3671] 3671 root509Tail2 winner = winner :=
  (root509Step2).trans root509Scan3
theorem root509Scan1 : scan edge (search edge 3) 4 [3671] 3671 root509Tail1 winner = winner :=
  (root509Step1).trans root509Scan2
theorem root509Scan0 : scan edge (search edge 3) 4 [3671] 3671 root509Tail0 winner = winner :=
  (root509Step0).trans root509Scan1
set_option maxRecDepth 4096 in
theorem root509Branch : search edge 4 [3671] root509Candidates 3671 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3671] 3671 root509Tail0 winner = winner
  exact root509Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
