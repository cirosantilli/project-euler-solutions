import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root512Candidates : List Nat := Data.row3691
set_option maxRecDepth 8192 in
theorem root512Candidates_checked : candidates edge 5 0 3691 tail513 winner.bound = root512Candidates := by
  apply candidates_eq_row edge 5 0 3691 tail513 root512Candidates winner.bound tail513_ordered
  · decide
  · decide
  · funext q
    rfl
def root512Tail41 : List Nat := []
def root512Tail40 : List Nat := 22111 :: root512Tail41
def root512Tail39 : List Nat := 21673 :: root512Tail40
def root512Tail38 : List Nat := 21481 :: root512Tail39
def root512Tail37 : List Nat := 21397 :: root512Tail38
def root512Tail36 : List Nat := 21001 :: root512Tail37
def root512Tail35 : List Nat := 19381 :: root512Tail36
def root512Tail34 : List Nat := 17971 :: root512Tail35
def root512Tail33 : List Nat := 17239 :: root512Tail34
def root512Tail32 : List Nat := 16603 :: root512Tail33
def root512Tail31 : List Nat := 15919 :: root512Tail32
def root512Tail30 : List Nat := 15787 :: root512Tail31
def root512Tail29 : List Nat := 15601 :: root512Tail30
def root512Tail28 : List Nat := 15511 :: root512Tail29
def root512Tail27 : List Nat := 14071 :: root512Tail28
def root512Tail26 : List Nat := 13291 :: root512Tail27
def root512Tail25 : List Nat := 13267 :: root512Tail26
def root512Tail24 : List Nat := 12739 :: root512Tail25
def root512Tail23 : List Nat := 12343 :: root512Tail24
def root512Tail22 : List Nat := 12157 :: root512Tail23
def root512Tail21 : List Nat := 11923 :: root512Tail22
def root512Tail20 : List Nat := 11317 :: root512Tail21
def root512Tail19 : List Nat := 10369 :: root512Tail20
def root512Tail18 : List Nat := 9433 :: root512Tail19
def root512Tail17 : List Nat := 9319 :: root512Tail18
def root512Tail16 : List Nat := 9013 :: root512Tail17
def root512Tail15 : List Nat := 8863 :: root512Tail16
def root512Tail14 : List Nat := 8731 :: root512Tail15
def root512Tail13 : List Nat := 8521 :: root512Tail14
def root512Tail12 : List Nat := 8419 :: root512Tail13
def root512Tail11 : List Nat := 8377 :: root512Tail12
def root512Tail10 : List Nat := 7789 :: root512Tail11
def root512Tail9 : List Nat := 7669 :: root512Tail10
def root512Tail8 : List Nat := 7561 :: root512Tail9
def root512Tail7 : List Nat := 7477 :: root512Tail8
def root512Tail6 : List Nat := 7333 :: root512Tail7
def root512Tail5 : List Nat := 6337 :: root512Tail6
def root512Tail4 : List Nat := 5419 :: root512Tail5
def root512Tail3 : List Nat := 4561 :: root512Tail4
def root512Tail2 : List Nat := 4177 :: root512Tail3
def root512Tail1 : List Nat := 4153 :: root512Tail2
def root512Tail0 : List Nat := 4051 :: root512Tail1
def root512Candidates0 : List Nat := [12157]
set_option maxRecDepth 8192 in
theorem root512Candidates0_checked : candidates edge 4 3691 4051 root512Tail1 winner.bound = root512Candidates0 := by
  unfold candidates
  have row : edge 4051 = (SortedNatTree.contains FastRows.row4051) := by
    change Data.row4051.contains = (SortedNatTree.contains FastRows.row4051)
    exact FastRows.row4051_checked.symm
  rw [row]
  decide
theorem root512Step0 : scan edge (search edge 3) 4 [3691] 3691 root512Tail0 winner =
    scan edge (search edge 3) 4 [3691] 3691 root512Tail1 winner := by
  rw [root512Tail0, scan, if_neg (by decide), root512Candidates0_checked]
  change scan edge (search edge 3) 4 [3691] 3691 root512Tail1
    (if 3 ≤ root512Candidates0.length then search edge 3 [3691, 4051] root512Candidates0 7742 winner else winner) = _
  rw [if_neg (by decide)]
def root512Candidates1 : List Nat := [12343, 14071, 17971]
set_option maxRecDepth 8192 in
theorem root512Candidates1_checked : candidates edge 4 3691 4153 root512Tail2 winner.bound = root512Candidates1 := by
  unfold candidates
  have row : edge 4153 = (SortedNatTree.contains FastRows.row4153) := by
    change Data.row4153.contains = (SortedNatTree.contains FastRows.row4153)
    exact FastRows.row4153_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root512Child1 : search edge 3 [3691, 4153] root512Candidates1 7844 winner = winner := by decide
theorem root512Step1 : scan edge (search edge 3) 4 [3691] 3691 root512Tail1 winner =
    scan edge (search edge 3) 4 [3691] 3691 root512Tail2 winner := by
  rw [root512Tail1, scan, if_neg (by decide), root512Candidates1_checked]
  change scan edge (search edge 3) 4 [3691] 3691 root512Tail2
    (if 3 ≤ root512Candidates1.length then search edge 3 [3691, 4153] root512Candidates1 7844 winner else winner) = _
  rw [if_pos (by decide), root512Child1]
def root512Candidates2 : List Nat := [8521, 15919]
set_option maxRecDepth 8192 in
theorem root512Candidates2_checked : candidates edge 4 3691 4177 root512Tail3 winner.bound = root512Candidates2 := by
  unfold candidates
  have row : edge 4177 = (SortedNatTree.contains FastRows.row4177) := by
    change Data.row4177.contains = (SortedNatTree.contains FastRows.row4177)
    exact FastRows.row4177_checked.symm
  rw [row]
  decide
theorem root512Step2 : scan edge (search edge 3) 4 [3691] 3691 root512Tail2 winner =
    scan edge (search edge 3) 4 [3691] 3691 root512Tail3 winner := by
  rw [root512Tail2, scan, if_neg (by decide), root512Candidates2_checked]
  change scan edge (search edge 3) 4 [3691] 3691 root512Tail3
    (if 3 ≤ root512Candidates2.length then search edge 3 [3691, 4177] root512Candidates2 7868 winner else winner) = _
  rw [if_neg (by decide)]
def root512Candidates3 : List Nat := [5419, 8863]
set_option maxRecDepth 8192 in
theorem root512Candidates3_checked : candidates edge 4 3691 4561 root512Tail4 winner.bound = root512Candidates3 := by
  unfold candidates
  have row : edge 4561 = (SortedNatTree.contains FastRows.row4561) := by
    change Data.row4561.contains = (SortedNatTree.contains FastRows.row4561)
    exact FastRows.row4561_checked.symm
  rw [row]
  decide
theorem root512Step3 : scan edge (search edge 3) 4 [3691] 3691 root512Tail3 winner =
    scan edge (search edge 3) 4 [3691] 3691 root512Tail4 winner := by
  rw [root512Tail3, scan, if_neg (by decide), root512Candidates3_checked]
  change scan edge (search edge 3) 4 [3691] 3691 root512Tail4
    (if 3 ≤ root512Candidates3.length then search edge 3 [3691, 4561] root512Candidates3 8252 winner else winner) = _
  rw [if_neg (by decide)]
def root512Candidates4 : List Nat := [12739]
set_option maxRecDepth 8192 in
theorem root512Candidates4_checked : candidates edge 4 3691 5419 root512Tail5 winner.bound = root512Candidates4 := by
  unfold candidates
  have row : edge 5419 = (SortedNatTree.contains FastRows.row5419) := by
    change Data.row5419.contains = (SortedNatTree.contains FastRows.row5419)
    exact FastRows.row5419_checked.symm
  rw [row]
  decide
theorem root512Step4 : scan edge (search edge 3) 4 [3691] 3691 root512Tail4 winner =
    scan edge (search edge 3) 4 [3691] 3691 root512Tail5 winner := by
  rw [root512Tail4, scan, if_neg (by decide), root512Candidates4_checked]
  change scan edge (search edge 3) 4 [3691] 3691 root512Tail5
    (if 3 ≤ root512Candidates4.length then search edge 3 [3691, 5419] root512Candidates4 9110 winner else winner) = _
  rw [if_neg (by decide)]
theorem root512Scan5 : scan edge (search edge 3) 4 [3691] 3691 root512Tail5 winner = winner := by rw [root512Tail5, scan, if_pos (by decide)]
theorem root512Scan4 : scan edge (search edge 3) 4 [3691] 3691 root512Tail4 winner = winner :=
  (root512Step4).trans root512Scan5
theorem root512Scan3 : scan edge (search edge 3) 4 [3691] 3691 root512Tail3 winner = winner :=
  (root512Step3).trans root512Scan4
theorem root512Scan2 : scan edge (search edge 3) 4 [3691] 3691 root512Tail2 winner = winner :=
  (root512Step2).trans root512Scan3
theorem root512Scan1 : scan edge (search edge 3) 4 [3691] 3691 root512Tail1 winner = winner :=
  (root512Step1).trans root512Scan2
theorem root512Scan0 : scan edge (search edge 3) 4 [3691] 3691 root512Tail0 winner = winner :=
  (root512Step0).trans root512Scan1
set_option maxRecDepth 4096 in
theorem root512Branch : search edge 4 [3691] root512Candidates 3691 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3691] 3691 root512Tail0 winner = winner
  exact root512Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
