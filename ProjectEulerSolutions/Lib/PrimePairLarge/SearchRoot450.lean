import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root450Candidates : List Nat := Data.row3203
set_option maxRecDepth 8192 in
theorem root450Candidates_checked : candidates edge 5 0 3203 tail451 winner.bound = root450Candidates := by
  apply candidates_eq_row edge 5 0 3203 tail451 root450Candidates winner.bound tail451_ordered
  · decide
  · decide
  · funext q
    rfl
def root450Tail40 : List Nat := []
def root450Tail39 : List Nat := 22811 :: root450Tail40
def root450Tail38 : List Nat := 21059 :: root450Tail39
def root450Tail37 : List Nat := 21011 :: root450Tail38
def root450Tail36 : List Nat := 20771 :: root450Tail37
def root450Tail35 : List Nat := 20123 :: root450Tail36
def root450Tail34 : List Nat := 19751 :: root450Tail35
def root450Tail33 : List Nat := 19463 :: root450Tail34
def root450Tail32 : List Nat := 19163 :: root450Tail33
def root450Tail31 : List Nat := 19079 :: root450Tail32
def root450Tail30 : List Nat := 19037 :: root450Tail31
def root450Tail29 : List Nat := 18839 :: root450Tail30
def root450Tail28 : List Nat := 18749 :: root450Tail29
def root450Tail27 : List Nat := 17483 :: root450Tail28
def root450Tail26 : List Nat := 17117 :: root450Tail27
def root450Tail25 : List Nat := 16103 :: root450Tail26
def root450Tail24 : List Nat := 15263 :: root450Tail25
def root450Tail23 : List Nat := 13619 :: root450Tail24
def root450Tail22 : List Nat := 13043 :: root450Tail23
def root450Tail21 : List Nat := 11903 :: root450Tail22
def root450Tail20 : List Nat := 11597 :: root450Tail21
def root450Tail19 : List Nat := 10253 :: root450Tail20
def root450Tail18 : List Nat := 10223 :: root450Tail19
def root450Tail17 : List Nat := 10061 :: root450Tail18
def root450Tail16 : List Nat := 9923 :: root450Tail17
def root450Tail15 : List Nat := 9311 :: root450Tail16
def root450Tail14 : List Nat := 9221 :: root450Tail15
def root450Tail13 : List Nat := 9209 :: root450Tail14
def root450Tail12 : List Nat := 8933 :: root450Tail13
def root450Tail11 : List Nat := 8627 :: root450Tail12
def root450Tail10 : List Nat := 8597 :: root450Tail11
def root450Tail9 : List Nat := 8219 :: root450Tail10
def root450Tail8 : List Nat := 7949 :: root450Tail9
def root450Tail7 : List Nat := 7541 :: root450Tail8
def root450Tail6 : List Nat := 7331 :: root450Tail7
def root450Tail5 : List Nat := 6827 :: root450Tail6
def root450Tail4 : List Nat := 6101 :: root450Tail5
def root450Tail3 : List Nat := 5867 :: root450Tail4
def root450Tail2 : List Nat := 5279 :: root450Tail3
def root450Tail1 : List Nat := 4481 :: root450Tail2
def root450Tail0 : List Nat := 3467 :: root450Tail1
def root450Candidates0 : List Nat := [7949]
set_option maxRecDepth 8192 in
theorem root450Candidates0_checked : candidates edge 4 3203 3467 root450Tail1 winner.bound = root450Candidates0 := by
  unfold candidates
  have row : edge 3467 = (SortedNatTree.contains FastRows.row3467) := by
    change Data.row3467.contains = (SortedNatTree.contains FastRows.row3467)
    exact FastRows.row3467_checked.symm
  rw [row]
  decide
theorem root450Step0 : scan edge (search edge 3) 4 [3203] 3203 root450Tail0 winner =
    scan edge (search edge 3) 4 [3203] 3203 root450Tail1 winner := by
  rw [root450Tail0, scan, if_neg (by decide), root450Candidates0_checked]
  change scan edge (search edge 3) 4 [3203] 3203 root450Tail1
    (if 3 ≤ root450Candidates0.length then search edge 3 [3203, 3467] root450Candidates0 6670 winner else winner) = _
  rw [if_neg (by decide)]
def root450Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root450Candidates1_checked : candidates edge 4 3203 4481 root450Tail2 winner.bound = root450Candidates1 := by
  unfold candidates
  have row : edge 4481 = (SortedNatTree.contains FastRows.row4481) := by
    change Data.row4481.contains = (SortedNatTree.contains FastRows.row4481)
    exact FastRows.row4481_checked.symm
  rw [row]
  decide
theorem root450Step1 : scan edge (search edge 3) 4 [3203] 3203 root450Tail1 winner =
    scan edge (search edge 3) 4 [3203] 3203 root450Tail2 winner := by
  rw [root450Tail1, scan, if_neg (by decide), root450Candidates1_checked]
  change scan edge (search edge 3) 4 [3203] 3203 root450Tail2
    (if 3 ≤ root450Candidates1.length then search edge 3 [3203, 4481] root450Candidates1 7684 winner else winner) = _
  rw [if_neg (by decide)]
def root450Candidates2 : List Nat := [6827, 9209, 17483]
set_option maxRecDepth 8192 in
theorem root450Candidates2_checked : candidates edge 4 3203 5279 root450Tail3 winner.bound = root450Candidates2 := by
  unfold candidates
  have row : edge 5279 = (SortedNatTree.contains FastRows.row5279) := by
    change Data.row5279.contains = (SortedNatTree.contains FastRows.row5279)
    exact FastRows.row5279_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root450Child2 : search edge 3 [3203, 5279] root450Candidates2 8482 winner = winner := by decide
theorem root450Step2 : scan edge (search edge 3) 4 [3203] 3203 root450Tail2 winner =
    scan edge (search edge 3) 4 [3203] 3203 root450Tail3 winner := by
  rw [root450Tail2, scan, if_neg (by decide), root450Candidates2_checked]
  change scan edge (search edge 3) 4 [3203] 3203 root450Tail3
    (if 3 ≤ root450Candidates2.length then search edge 3 [3203, 5279] root450Candidates2 8482 winner else winner) = _
  rw [if_pos (by decide), root450Child2]
theorem root450Scan3 : scan edge (search edge 3) 4 [3203] 3203 root450Tail3 winner = winner := by rw [root450Tail3, scan, if_pos (by decide)]
theorem root450Scan2 : scan edge (search edge 3) 4 [3203] 3203 root450Tail2 winner = winner :=
  (root450Step2).trans root450Scan3
theorem root450Scan1 : scan edge (search edge 3) 4 [3203] 3203 root450Tail1 winner = winner :=
  (root450Step1).trans root450Scan2
theorem root450Scan0 : scan edge (search edge 3) 4 [3203] 3203 root450Tail0 winner = winner :=
  (root450Step0).trans root450Scan1
set_option maxRecDepth 4096 in
theorem root450Branch : search edge 4 [3203] root450Candidates 3203 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3203] 3203 root450Tail0 winner = winner
  exact root450Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
