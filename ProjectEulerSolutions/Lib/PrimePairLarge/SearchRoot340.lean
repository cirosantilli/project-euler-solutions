import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root340Candidates : List Nat := Data.row2309
set_option maxRecDepth 8192 in
theorem root340Candidates_checked : candidates edge 5 0 2309 tail341 winner.bound = root340Candidates := by
  apply candidates_eq_row edge 5 0 2309 tail341 root340Candidates winner.bound tail341_ordered
  · decide
  · decide
  · funext q
    rfl
def root340Tail44 : List Nat := []
def root340Tail43 : List Nat := 23531 :: root340Tail44
def root340Tail42 : List Nat := 23447 :: root340Tail43
def root340Tail41 : List Nat := 22817 :: root340Tail42
def root340Tail40 : List Nat := 22721 :: root340Tail41
def root340Tail39 : List Nat := 21683 :: root340Tail40
def root340Tail38 : List Nat := 21563 :: root340Tail39
def root340Tail37 : List Nat := 21419 :: root340Tail38
def root340Tail36 : List Nat := 21227 :: root340Tail37
def root340Tail35 : List Nat := 20369 :: root340Tail36
def root340Tail34 : List Nat := 19991 :: root340Tail35
def root340Tail33 : List Nat := 19841 :: root340Tail34
def root340Tail32 : List Nat := 19697 :: root340Tail33
def root340Tail31 : List Nat := 18119 :: root340Tail32
def root340Tail30 : List Nat := 17573 :: root340Tail31
def root340Tail29 : List Nat := 16943 :: root340Tail30
def root340Tail28 : List Nat := 16547 :: root340Tail29
def root340Tail27 : List Nat := 15809 :: root340Tail28
def root340Tail26 : List Nat := 15551 :: root340Tail27
def root340Tail25 : List Nat := 14561 :: root340Tail26
def root340Tail24 : List Nat := 13313 :: root340Tail25
def root340Tail23 : List Nat := 13217 :: root340Tail24
def root340Tail22 : List Nat := 12911 :: root340Tail23
def root340Tail21 : List Nat := 12197 :: root340Tail22
def root340Tail20 : List Nat := 11003 :: root340Tail21
def root340Tail19 : List Nat := 10391 :: root340Tail20
def root340Tail18 : List Nat := 10037 :: root340Tail19
def root340Tail17 : List Nat := 9887 :: root340Tail18
def root340Tail16 : List Nat := 8861 :: root340Tail17
def root340Tail15 : List Nat := 8681 :: root340Tail16
def root340Tail14 : List Nat := 8237 :: root340Tail15
def root340Tail13 : List Nat := 7829 :: root340Tail14
def root340Tail12 : List Nat := 7817 :: root340Tail13
def root340Tail11 : List Nat := 7691 :: root340Tail12
def root340Tail10 : List Nat := 6947 :: root340Tail11
def root340Tail9 : List Nat := 6257 :: root340Tail10
def root340Tail8 : List Nat := 6131 :: root340Tail9
def root340Tail7 : List Nat := 5783 :: root340Tail8
def root340Tail6 : List Nat := 5417 :: root340Tail7
def root340Tail5 : List Nat := 5087 :: root340Tail6
def root340Tail4 : List Nat := 5081 :: root340Tail5
def root340Tail3 : List Nat := 4451 :: root340Tail4
def root340Tail2 : List Nat := 4133 :: root340Tail3
def root340Tail1 : List Nat := 3491 :: root340Tail2
def root340Tail0 : List Nat := 2687 :: root340Tail1
def root340Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root340Candidates0_checked : candidates edge 4 2309 2687 root340Tail1 winner.bound = root340Candidates0 := by
  unfold candidates
  have row : edge 2687 = (SortedNatTree.contains FastRows.row2687) := by
    change Data.row2687.contains = (SortedNatTree.contains FastRows.row2687)
    exact FastRows.row2687_checked.symm
  rw [row]
  decide
theorem root340Step0 : scan edge (search edge 3) 4 [2309] 2309 root340Tail0 winner =
    scan edge (search edge 3) 4 [2309] 2309 root340Tail1 winner := by
  rw [root340Tail0, scan, if_neg (by decide), root340Candidates0_checked]
  change scan edge (search edge 3) 4 [2309] 2309 root340Tail1
    (if 3 ≤ root340Candidates0.length then search edge 3 [2309, 2687] root340Candidates0 4996 winner else winner) = _
  rw [if_neg (by decide)]
def root340Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root340Candidates1_checked : candidates edge 4 2309 3491 root340Tail2 winner.bound = root340Candidates1 := by
  unfold candidates
  have row : edge 3491 = (SortedNatTree.contains FastRows.row3491) := by
    change Data.row3491.contains = (SortedNatTree.contains FastRows.row3491)
    exact FastRows.row3491_checked.symm
  rw [row]
  decide
theorem root340Step1 : scan edge (search edge 3) 4 [2309] 2309 root340Tail1 winner =
    scan edge (search edge 3) 4 [2309] 2309 root340Tail2 winner := by
  rw [root340Tail1, scan, if_neg (by decide), root340Candidates1_checked]
  change scan edge (search edge 3) 4 [2309] 2309 root340Tail2
    (if 3 ≤ root340Candidates1.length then search edge 3 [2309, 3491] root340Candidates1 5800 winner else winner) = _
  rw [if_neg (by decide)]
def root340Candidates2 : List Nat := [5081, 6947]
set_option maxRecDepth 8192 in
theorem root340Candidates2_checked : candidates edge 4 2309 4133 root340Tail3 winner.bound = root340Candidates2 := by
  unfold candidates
  have row : edge 4133 = (SortedNatTree.contains FastRows.row4133) := by
    change Data.row4133.contains = (SortedNatTree.contains FastRows.row4133)
    exact FastRows.row4133_checked.symm
  rw [row]
  decide
theorem root340Step2 : scan edge (search edge 3) 4 [2309] 2309 root340Tail2 winner =
    scan edge (search edge 3) 4 [2309] 2309 root340Tail3 winner := by
  rw [root340Tail2, scan, if_neg (by decide), root340Candidates2_checked]
  change scan edge (search edge 3) 4 [2309] 2309 root340Tail3
    (if 3 ≤ root340Candidates2.length then search edge 3 [2309, 4133] root340Candidates2 6442 winner else winner) = _
  rw [if_neg (by decide)]
def root340Candidates3 : List Nat := [6257, 8681, 8861, 16943]
set_option maxRecDepth 8192 in
theorem root340Candidates3_checked : candidates edge 4 2309 4451 root340Tail4 winner.bound = root340Candidates3 := by
  unfold candidates
  have row : edge 4451 = (SortedNatTree.contains FastRows.row4451) := by
    change Data.row4451.contains = (SortedNatTree.contains FastRows.row4451)
    exact FastRows.row4451_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root340Child3 : search edge 3 [2309, 4451] root340Candidates3 6760 winner = winner := by decide
theorem root340Step3 : scan edge (search edge 3) 4 [2309] 2309 root340Tail3 winner =
    scan edge (search edge 3) 4 [2309] 2309 root340Tail4 winner := by
  rw [root340Tail3, scan, if_neg (by decide), root340Candidates3_checked]
  change scan edge (search edge 3) 4 [2309] 2309 root340Tail4
    (if 3 ≤ root340Candidates3.length then search edge 3 [2309, 4451] root340Candidates3 6760 winner else winner) = _
  rw [if_pos (by decide), root340Child3]
def root340Candidates4 : List Nat := [5783, 7817, 15809]
set_option maxRecDepth 8192 in
theorem root340Candidates4_checked : candidates edge 4 2309 5081 root340Tail5 winner.bound = root340Candidates4 := by
  unfold candidates
  have row : edge 5081 = (SortedNatTree.contains FastRows.row5081) := by
    change Data.row5081.contains = (SortedNatTree.contains FastRows.row5081)
    exact FastRows.row5081_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root340Child4 : search edge 3 [2309, 5081] root340Candidates4 7390 winner = winner := by decide
theorem root340Step4 : scan edge (search edge 3) 4 [2309] 2309 root340Tail4 winner =
    scan edge (search edge 3) 4 [2309] 2309 root340Tail5 winner := by
  rw [root340Tail4, scan, if_neg (by decide), root340Candidates4_checked]
  change scan edge (search edge 3) 4 [2309] 2309 root340Tail5
    (if 3 ≤ root340Candidates4.length then search edge 3 [2309, 5081] root340Candidates4 7390 winner else winner) = _
  rw [if_pos (by decide), root340Child4]
def root340Candidates5 : List Nat := [14561]
set_option maxRecDepth 8192 in
theorem root340Candidates5_checked : candidates edge 4 2309 5087 root340Tail6 winner.bound = root340Candidates5 := by
  unfold candidates
  have row : edge 5087 = (SortedNatTree.contains FastRows.row5087) := by
    change Data.row5087.contains = (SortedNatTree.contains FastRows.row5087)
    exact FastRows.row5087_checked.symm
  rw [row]
  decide
theorem root340Step5 : scan edge (search edge 3) 4 [2309] 2309 root340Tail5 winner =
    scan edge (search edge 3) 4 [2309] 2309 root340Tail6 winner := by
  rw [root340Tail5, scan, if_neg (by decide), root340Candidates5_checked]
  change scan edge (search edge 3) 4 [2309] 2309 root340Tail6
    (if 3 ≤ root340Candidates5.length then search edge 3 [2309, 5087] root340Candidates5 7396 winner else winner) = _
  rw [if_neg (by decide)]
def root340Candidates6 : List Nat := [6131]
set_option maxRecDepth 8192 in
theorem root340Candidates6_checked : candidates edge 4 2309 5417 root340Tail7 winner.bound = root340Candidates6 := by
  unfold candidates
  have row : edge 5417 = (SortedNatTree.contains FastRows.row5417) := by
    change Data.row5417.contains = (SortedNatTree.contains FastRows.row5417)
    exact FastRows.row5417_checked.symm
  rw [row]
  decide
theorem root340Step6 : scan edge (search edge 3) 4 [2309] 2309 root340Tail6 winner =
    scan edge (search edge 3) 4 [2309] 2309 root340Tail7 winner := by
  rw [root340Tail6, scan, if_neg (by decide), root340Candidates6_checked]
  change scan edge (search edge 3) 4 [2309] 2309 root340Tail7
    (if 3 ≤ root340Candidates6.length then search edge 3 [2309, 5417] root340Candidates6 7726 winner else winner) = _
  rw [if_neg (by decide)]
def root340Candidates7 : List Nat := [13217]
set_option maxRecDepth 8192 in
theorem root340Candidates7_checked : candidates edge 4 2309 5783 root340Tail8 winner.bound = root340Candidates7 := by
  unfold candidates
  have row : edge 5783 = (SortedNatTree.contains FastRows.row5783) := by
    change Data.row5783.contains = (SortedNatTree.contains FastRows.row5783)
    exact FastRows.row5783_checked.symm
  rw [row]
  decide
theorem root340Step7 : scan edge (search edge 3) 4 [2309] 2309 root340Tail7 winner =
    scan edge (search edge 3) 4 [2309] 2309 root340Tail8 winner := by
  rw [root340Tail7, scan, if_neg (by decide), root340Candidates7_checked]
  change scan edge (search edge 3) 4 [2309] 2309 root340Tail8
    (if 3 ≤ root340Candidates7.length then search edge 3 [2309, 5783] root340Candidates7 8092 winner else winner) = _
  rw [if_neg (by decide)]
theorem root340Scan8 : scan edge (search edge 3) 4 [2309] 2309 root340Tail8 winner = winner := by rw [root340Tail8, scan, if_pos (by decide)]
theorem root340Scan7 : scan edge (search edge 3) 4 [2309] 2309 root340Tail7 winner = winner :=
  (root340Step7).trans root340Scan8
theorem root340Scan6 : scan edge (search edge 3) 4 [2309] 2309 root340Tail6 winner = winner :=
  (root340Step6).trans root340Scan7
theorem root340Scan5 : scan edge (search edge 3) 4 [2309] 2309 root340Tail5 winner = winner :=
  (root340Step5).trans root340Scan6
theorem root340Scan4 : scan edge (search edge 3) 4 [2309] 2309 root340Tail4 winner = winner :=
  (root340Step4).trans root340Scan5
theorem root340Scan3 : scan edge (search edge 3) 4 [2309] 2309 root340Tail3 winner = winner :=
  (root340Step3).trans root340Scan4
theorem root340Scan2 : scan edge (search edge 3) 4 [2309] 2309 root340Tail2 winner = winner :=
  (root340Step2).trans root340Scan3
theorem root340Scan1 : scan edge (search edge 3) 4 [2309] 2309 root340Tail1 winner = winner :=
  (root340Step1).trans root340Scan2
theorem root340Scan0 : scan edge (search edge 3) 4 [2309] 2309 root340Tail0 winner = winner :=
  (root340Step0).trans root340Scan1
set_option maxRecDepth 4096 in
theorem root340Branch : search edge 4 [2309] root340Candidates 2309 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2309] 2309 root340Tail0 winner = winner
  exact root340Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
