import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root373Candidates : List Nat := Data.row2579
set_option maxRecDepth 8192 in
theorem root373Candidates_checked : candidates edge 5 0 2579 tail374 winner.bound = root373Candidates := by
  apply candidates_eq_row edge 5 0 2579 tail374 root373Candidates winner.bound tail374_ordered
  · decide
  · decide
  · funext q
    rfl
def root373Tail34 : List Nat := []
def root373Tail33 : List Nat := 23399 :: root373Tail34
def root373Tail32 : List Nat := 23027 :: root373Tail33
def root373Tail31 : List Nat := 22859 :: root373Tail32
def root373Tail30 : List Nat := 22271 :: root373Tail31
def root373Tail29 : List Nat := 20879 :: root373Tail30
def root373Tail28 : List Nat := 20759 :: root373Tail29
def root373Tail27 : List Nat := 19919 :: root373Tail28
def root373Tail26 : List Nat := 19463 :: root373Tail27
def root373Tail25 : List Nat := 19373 :: root373Tail26
def root373Tail24 : List Nat := 17519 :: root373Tail25
def root373Tail23 : List Nat := 15173 :: root373Tail24
def root373Tail22 : List Nat := 13841 :: root373Tail23
def root373Tail21 : List Nat := 12527 :: root373Tail22
def root373Tail20 : List Nat := 12113 :: root373Tail21
def root373Tail19 : List Nat := 11987 :: root373Tail20
def root373Tail18 : List Nat := 11783 :: root373Tail19
def root373Tail17 : List Nat := 11483 :: root373Tail18
def root373Tail16 : List Nat := 11393 :: root373Tail17
def root373Tail15 : List Nat := 11273 :: root373Tail16
def root373Tail14 : List Nat := 11261 :: root373Tail15
def root373Tail13 : List Nat := 11177 :: root373Tail14
def root373Tail12 : List Nat := 10559 :: root373Tail13
def root373Tail11 : List Nat := 7727 :: root373Tail12
def root373Tail10 : List Nat := 7349 :: root373Tail11
def root373Tail9 : List Nat := 7229 :: root373Tail10
def root373Tail8 : List Nat := 6131 :: root373Tail9
def root373Tail7 : List Nat := 5843 :: root373Tail8
def root373Tail6 : List Nat := 5639 :: root373Tail7
def root373Tail5 : List Nat := 4931 :: root373Tail6
def root373Tail4 : List Nat := 4871 :: root373Tail5
def root373Tail3 : List Nat := 4421 :: root373Tail4
def root373Tail2 : List Nat := 4007 :: root373Tail3
def root373Tail1 : List Nat := 3623 :: root373Tail2
def root373Tail0 : List Nat := 3359 :: root373Tail1
def root373Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root373Candidates0_checked : candidates edge 4 2579 3359 root373Tail1 winner.bound = root373Candidates0 := by
  unfold candidates
  have row : edge 3359 = (SortedNatTree.contains FastRows.row3359) := by
    change Data.row3359.contains = (SortedNatTree.contains FastRows.row3359)
    exact FastRows.row3359_checked.symm
  rw [row]
  decide
theorem root373Step0 : scan edge (search edge 3) 4 [2579] 2579 root373Tail0 winner =
    scan edge (search edge 3) 4 [2579] 2579 root373Tail1 winner := by
  rw [root373Tail0, scan, if_neg (by decide), root373Candidates0_checked]
  change scan edge (search edge 3) 4 [2579] 2579 root373Tail1
    (if 3 ≤ root373Candidates0.length then search edge 3 [2579, 3359] root373Candidates0 5938 winner else winner) = _
  rw [if_neg (by decide)]
def root373Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root373Candidates1_checked : candidates edge 4 2579 3623 root373Tail2 winner.bound = root373Candidates1 := by
  unfold candidates
  have row : edge 3623 = (SortedNatTree.contains FastRows.row3623) := by
    change Data.row3623.contains = (SortedNatTree.contains FastRows.row3623)
    exact FastRows.row3623_checked.symm
  rw [row]
  decide
theorem root373Step1 : scan edge (search edge 3) 4 [2579] 2579 root373Tail1 winner =
    scan edge (search edge 3) 4 [2579] 2579 root373Tail2 winner := by
  rw [root373Tail1, scan, if_neg (by decide), root373Candidates1_checked]
  change scan edge (search edge 3) 4 [2579] 2579 root373Tail2
    (if 3 ≤ root373Candidates1.length then search edge 3 [2579, 3623] root373Candidates1 6202 winner else winner) = _
  rw [if_neg (by decide)]
def root373Candidates2 : List Nat := [5639, 12113, 15173, 17519]
set_option maxRecDepth 8192 in
theorem root373Candidates2_checked : candidates edge 4 2579 4007 root373Tail3 winner.bound = root373Candidates2 := by
  unfold candidates
  have row : edge 4007 = (SortedNatTree.contains FastRows.row4007) := by
    change Data.row4007.contains = (SortedNatTree.contains FastRows.row4007)
    exact FastRows.row4007_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root373Child2 : search edge 3 [2579, 4007] root373Candidates2 6586 winner = winner := by decide
theorem root373Step2 : scan edge (search edge 3) 4 [2579] 2579 root373Tail2 winner =
    scan edge (search edge 3) 4 [2579] 2579 root373Tail3 winner := by
  rw [root373Tail2, scan, if_neg (by decide), root373Candidates2_checked]
  change scan edge (search edge 3) 4 [2579] 2579 root373Tail3
    (if 3 ≤ root373Candidates2.length then search edge 3 [2579, 4007] root373Candidates2 6586 winner else winner) = _
  rw [if_pos (by decide), root373Child2]
def root373Candidates3 : List Nat := [11393]
set_option maxRecDepth 8192 in
theorem root373Candidates3_checked : candidates edge 4 2579 4421 root373Tail4 winner.bound = root373Candidates3 := by
  unfold candidates
  have row : edge 4421 = (SortedNatTree.contains FastRows.row4421) := by
    change Data.row4421.contains = (SortedNatTree.contains FastRows.row4421)
    exact FastRows.row4421_checked.symm
  rw [row]
  decide
theorem root373Step3 : scan edge (search edge 3) 4 [2579] 2579 root373Tail3 winner =
    scan edge (search edge 3) 4 [2579] 2579 root373Tail4 winner := by
  rw [root373Tail3, scan, if_neg (by decide), root373Candidates3_checked]
  change scan edge (search edge 3) 4 [2579] 2579 root373Tail4
    (if 3 ≤ root373Candidates3.length then search edge 3 [2579, 4421] root373Candidates3 7000 winner else winner) = _
  rw [if_neg (by decide)]
def root373Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root373Candidates4_checked : candidates edge 4 2579 4871 root373Tail5 winner.bound = root373Candidates4 := by
  unfold candidates
  have row : edge 4871 = (SortedNatTree.contains FastRows.row4871) := by
    change Data.row4871.contains = (SortedNatTree.contains FastRows.row4871)
    exact FastRows.row4871_checked.symm
  rw [row]
  decide
theorem root373Step4 : scan edge (search edge 3) 4 [2579] 2579 root373Tail4 winner =
    scan edge (search edge 3) 4 [2579] 2579 root373Tail5 winner := by
  rw [root373Tail4, scan, if_neg (by decide), root373Candidates4_checked]
  change scan edge (search edge 3) 4 [2579] 2579 root373Tail5
    (if 3 ≤ root373Candidates4.length then search edge 3 [2579, 4871] root373Candidates4 7450 winner else winner) = _
  rw [if_neg (by decide)]
def root373Candidates5 : List Nat := [5843, 7349]
set_option maxRecDepth 8192 in
theorem root373Candidates5_checked : candidates edge 4 2579 4931 root373Tail6 winner.bound = root373Candidates5 := by
  unfold candidates
  have row : edge 4931 = (SortedNatTree.contains FastRows.row4931) := by
    change Data.row4931.contains = (SortedNatTree.contains FastRows.row4931)
    exact FastRows.row4931_checked.symm
  rw [row]
  decide
theorem root373Step5 : scan edge (search edge 3) 4 [2579] 2579 root373Tail5 winner =
    scan edge (search edge 3) 4 [2579] 2579 root373Tail6 winner := by
  rw [root373Tail5, scan, if_neg (by decide), root373Candidates5_checked]
  change scan edge (search edge 3) 4 [2579] 2579 root373Tail6
    (if 3 ≤ root373Candidates5.length then search edge 3 [2579, 4931] root373Candidates5 7510 winner else winner) = _
  rw [if_neg (by decide)]
def root373Candidates6 : List Nat := [13841]
set_option maxRecDepth 8192 in
theorem root373Candidates6_checked : candidates edge 4 2579 5639 root373Tail7 winner.bound = root373Candidates6 := by
  unfold candidates
  have row : edge 5639 = (SortedNatTree.contains FastRows.row5639) := by
    change Data.row5639.contains = (SortedNatTree.contains FastRows.row5639)
    exact FastRows.row5639_checked.symm
  rw [row]
  decide
theorem root373Step6 : scan edge (search edge 3) 4 [2579] 2579 root373Tail6 winner =
    scan edge (search edge 3) 4 [2579] 2579 root373Tail7 winner := by
  rw [root373Tail6, scan, if_neg (by decide), root373Candidates6_checked]
  change scan edge (search edge 3) 4 [2579] 2579 root373Tail7
    (if 3 ≤ root373Candidates6.length then search edge 3 [2579, 5639] root373Candidates6 8218 winner else winner) = _
  rw [if_neg (by decide)]
def root373Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root373Candidates7_checked : candidates edge 4 2579 5843 root373Tail8 winner.bound = root373Candidates7 := by
  unfold candidates
  have row : edge 5843 = (SortedNatTree.contains FastRows.row5843) := by
    change Data.row5843.contains = (SortedNatTree.contains FastRows.row5843)
    exact FastRows.row5843_checked.symm
  rw [row]
  decide
theorem root373Step7 : scan edge (search edge 3) 4 [2579] 2579 root373Tail7 winner =
    scan edge (search edge 3) 4 [2579] 2579 root373Tail8 winner := by
  rw [root373Tail7, scan, if_neg (by decide), root373Candidates7_checked]
  change scan edge (search edge 3) 4 [2579] 2579 root373Tail8
    (if 3 ≤ root373Candidates7.length then search edge 3 [2579, 5843] root373Candidates7 8422 winner else winner) = _
  rw [if_neg (by decide)]
theorem root373Scan8 : scan edge (search edge 3) 4 [2579] 2579 root373Tail8 winner = winner := by rw [root373Tail8, scan, if_pos (by decide)]
theorem root373Scan7 : scan edge (search edge 3) 4 [2579] 2579 root373Tail7 winner = winner :=
  (root373Step7).trans root373Scan8
theorem root373Scan6 : scan edge (search edge 3) 4 [2579] 2579 root373Tail6 winner = winner :=
  (root373Step6).trans root373Scan7
theorem root373Scan5 : scan edge (search edge 3) 4 [2579] 2579 root373Tail5 winner = winner :=
  (root373Step5).trans root373Scan6
theorem root373Scan4 : scan edge (search edge 3) 4 [2579] 2579 root373Tail4 winner = winner :=
  (root373Step4).trans root373Scan5
theorem root373Scan3 : scan edge (search edge 3) 4 [2579] 2579 root373Tail3 winner = winner :=
  (root373Step3).trans root373Scan4
theorem root373Scan2 : scan edge (search edge 3) 4 [2579] 2579 root373Tail2 winner = winner :=
  (root373Step2).trans root373Scan3
theorem root373Scan1 : scan edge (search edge 3) 4 [2579] 2579 root373Tail1 winner = winner :=
  (root373Step1).trans root373Scan2
theorem root373Scan0 : scan edge (search edge 3) 4 [2579] 2579 root373Tail0 winner = winner :=
  (root373Step0).trans root373Scan1
set_option maxRecDepth 4096 in
theorem root373Branch : search edge 4 [2579] root373Candidates 2579 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2579] 2579 root373Tail0 winner = winner
  exact root373Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
