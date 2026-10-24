import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root584Candidates : List Nat := Data.row4273
set_option maxRecDepth 8192 in
theorem root584Candidates_checked : candidates edge 5 0 4273 tail585 winner.bound = root584Candidates := by
  apply candidates_eq_row edge 5 0 4273 tail585 root584Candidates winner.bound tail585_ordered
  · decide
  · decide
  · funext q
    rfl
def root584Tail36 : List Nat := []
def root584Tail35 : List Nat := 21277 :: root584Tail36
def root584Tail34 : List Nat := 21157 :: root584Tail35
def root584Tail33 : List Nat := 20809 :: root584Tail34
def root584Tail32 : List Nat := 20743 :: root584Tail33
def root584Tail31 : List Nat := 20731 :: root584Tail32
def root584Tail30 : List Nat := 19609 :: root584Tail31
def root584Tail29 : List Nat := 17497 :: root584Tail30
def root584Tail28 : List Nat := 16921 :: root584Tail29
def root584Tail27 : List Nat := 15991 :: root584Tail28
def root584Tail26 : List Nat := 15277 :: root584Tail27
def root584Tail25 : List Nat := 14683 :: root584Tail26
def root584Tail24 : List Nat := 13171 :: root584Tail25
def root584Tail23 : List Nat := 12703 :: root584Tail24
def root584Tail22 : List Nat := 12421 :: root584Tail23
def root584Tail21 : List Nat := 10867 :: root584Tail22
def root584Tail20 : List Nat := 10567 :: root584Tail21
def root584Tail19 : List Nat := 10243 :: root584Tail20
def root584Tail18 : List Nat := 9883 :: root584Tail19
def root584Tail17 : List Nat := 9397 :: root584Tail18
def root584Tail16 : List Nat := 9007 :: root584Tail17
def root584Tail15 : List Nat := 8803 :: root584Tail16
def root584Tail14 : List Nat := 8461 :: root584Tail15
def root584Tail13 : List Nat := 8287 :: root584Tail14
def root584Tail12 : List Nat := 7951 :: root584Tail13
def root584Tail11 : List Nat := 7573 :: root584Tail12
def root584Tail10 : List Nat := 7549 :: root584Tail11
def root584Tail9 : List Nat := 7489 :: root584Tail10
def root584Tail8 : List Nat := 7297 :: root584Tail9
def root584Tail7 : List Nat := 7207 :: root584Tail8
def root584Tail6 : List Nat := 6703 :: root584Tail7
def root584Tail5 : List Nat := 6373 :: root584Tail6
def root584Tail4 : List Nat := 6247 :: root584Tail5
def root584Tail3 : List Nat := 5023 :: root584Tail4
def root584Tail2 : List Nat := 4969 :: root584Tail3
def root584Tail1 : List Nat := 4723 :: root584Tail2
def root584Tail0 : List Nat := 4567 :: root584Tail1
def root584Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root584Candidates0_checked : candidates edge 4 4273 4567 root584Tail1 winner.bound = root584Candidates0 := by
  unfold candidates
  have row : edge 4567 = (SortedNatTree.contains FastRows.row4567) := by
    change Data.row4567.contains = (SortedNatTree.contains FastRows.row4567)
    exact FastRows.row4567_checked.symm
  rw [row]
  decide
theorem root584Step0 : scan edge (search edge 3) 4 [4273] 4273 root584Tail0 winner =
    scan edge (search edge 3) 4 [4273] 4273 root584Tail1 winner := by
  rw [root584Tail0, scan, if_neg (by decide), root584Candidates0_checked]
  change scan edge (search edge 3) 4 [4273] 4273 root584Tail1
    (if 3 ≤ root584Candidates0.length then search edge 3 [4273, 4567] root584Candidates0 8840 winner else winner) = _
  rw [if_neg (by decide)]
def root584Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root584Candidates1_checked : candidates edge 4 4273 4723 root584Tail2 winner.bound = root584Candidates1 := by
  unfold candidates
  have row : edge 4723 = (SortedNatTree.contains FastRows.row4723) := by
    change Data.row4723.contains = (SortedNatTree.contains FastRows.row4723)
    exact FastRows.row4723_checked.symm
  rw [row]
  decide
theorem root584Step1 : scan edge (search edge 3) 4 [4273] 4273 root584Tail1 winner =
    scan edge (search edge 3) 4 [4273] 4273 root584Tail2 winner := by
  rw [root584Tail1, scan, if_neg (by decide), root584Candidates1_checked]
  change scan edge (search edge 3) 4 [4273] 4273 root584Tail2
    (if 3 ≤ root584Candidates1.length then search edge 3 [4273, 4723] root584Candidates1 8996 winner else winner) = _
  rw [if_neg (by decide)]
def root584Candidates2 : List Nat := [8461]
set_option maxRecDepth 8192 in
theorem root584Candidates2_checked : candidates edge 4 4273 4969 root584Tail3 winner.bound = root584Candidates2 := by
  unfold candidates
  have row : edge 4969 = (SortedNatTree.contains FastRows.row4969) := by
    change Data.row4969.contains = (SortedNatTree.contains FastRows.row4969)
    exact FastRows.row4969_checked.symm
  rw [row]
  decide
theorem root584Step2 : scan edge (search edge 3) 4 [4273] 4273 root584Tail2 winner =
    scan edge (search edge 3) 4 [4273] 4273 root584Tail3 winner := by
  rw [root584Tail2, scan, if_neg (by decide), root584Candidates2_checked]
  change scan edge (search edge 3) 4 [4273] 4273 root584Tail3
    (if 3 ≤ root584Candidates2.length then search edge 3 [4273, 4969] root584Candidates2 9242 winner else winner) = _
  rw [if_neg (by decide)]
def root584Candidates3 : List Nat := [15277]
set_option maxRecDepth 8192 in
theorem root584Candidates3_checked : candidates edge 4 4273 5023 root584Tail4 winner.bound = root584Candidates3 := by
  unfold candidates
  have row : edge 5023 = (SortedNatTree.contains FastRows.row5023) := by
    change Data.row5023.contains = (SortedNatTree.contains FastRows.row5023)
    exact FastRows.row5023_checked.symm
  rw [row]
  decide
theorem root584Step3 : scan edge (search edge 3) 4 [4273] 4273 root584Tail3 winner =
    scan edge (search edge 3) 4 [4273] 4273 root584Tail4 winner := by
  rw [root584Tail3, scan, if_neg (by decide), root584Candidates3_checked]
  change scan edge (search edge 3) 4 [4273] 4273 root584Tail4
    (if 3 ≤ root584Candidates3.length then search edge 3 [4273, 5023] root584Candidates3 9296 winner else winner) = _
  rw [if_neg (by decide)]
theorem root584Scan4 : scan edge (search edge 3) 4 [4273] 4273 root584Tail4 winner = winner := by rw [root584Tail4, scan, if_pos (by decide)]
theorem root584Scan3 : scan edge (search edge 3) 4 [4273] 4273 root584Tail3 winner = winner :=
  (root584Step3).trans root584Scan4
theorem root584Scan2 : scan edge (search edge 3) 4 [4273] 4273 root584Tail2 winner = winner :=
  (root584Step2).trans root584Scan3
theorem root584Scan1 : scan edge (search edge 3) 4 [4273] 4273 root584Tail1 winner = winner :=
  (root584Step1).trans root584Scan2
theorem root584Scan0 : scan edge (search edge 3) 4 [4273] 4273 root584Tail0 winner = winner :=
  (root584Step0).trans root584Scan1
set_option maxRecDepth 4096 in
theorem root584Branch : search edge 4 [4273] root584Candidates 4273 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4273] 4273 root584Tail0 winner = winner
  exact root584Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
