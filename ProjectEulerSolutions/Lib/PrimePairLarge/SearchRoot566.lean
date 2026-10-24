import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root566Candidates : List Nat := Data.row4133
set_option maxRecDepth 8192 in
theorem root566Candidates_checked : candidates edge 5 0 4133 tail567 winner.bound = root566Candidates := by
  apply candidates_eq_row edge 5 0 4133 tail567 root566Candidates winner.bound tail567_ordered
  · decide
  · decide
  · funext q
    rfl
def root566Tail37 : List Nat := []
def root566Tail36 : List Nat := 21863 :: root566Tail37
def root566Tail35 : List Nat := 21647 :: root566Tail36
def root566Tail34 : List Nat := 21179 :: root566Tail35
def root566Tail33 : List Nat := 21023 :: root566Tail34
def root566Tail32 : List Nat := 21011 :: root566Tail33
def root566Tail31 : List Nat := 20849 :: root566Tail32
def root566Tail30 : List Nat := 19763 :: root566Tail31
def root566Tail29 : List Nat := 19157 :: root566Tail30
def root566Tail28 : List Nat := 17747 :: root566Tail29
def root566Tail27 : List Nat := 17579 :: root566Tail28
def root566Tail26 : List Nat := 17231 :: root566Tail27
def root566Tail25 : List Nat := 16061 :: root566Tail26
def root566Tail24 : List Nat := 15803 :: root566Tail25
def root566Tail23 : List Nat := 15647 :: root566Tail24
def root566Tail22 : List Nat := 14051 :: root566Tail23
def root566Tail21 : List Nat := 14009 :: root566Tail22
def root566Tail20 : List Nat := 13001 :: root566Tail21
def root566Tail19 : List Nat := 12953 :: root566Tail20
def root566Tail18 : List Nat := 12569 :: root566Tail19
def root566Tail17 : List Nat := 12329 :: root566Tail18
def root566Tail16 : List Nat := 12269 :: root566Tail17
def root566Tail15 : List Nat := 10169 :: root566Tail16
def root566Tail14 : List Nat := 9371 :: root566Tail15
def root566Tail13 : List Nat := 9293 :: root566Tail14
def root566Tail12 : List Nat := 9173 :: root566Tail13
def root566Tail11 : List Nat := 8111 :: root566Tail12
def root566Tail10 : List Nat := 7451 :: root566Tail11
def root566Tail9 : List Nat := 7433 :: root566Tail10
def root566Tail8 : List Nat := 6947 :: root566Tail9
def root566Tail7 : List Nat := 6173 :: root566Tail8
def root566Tail6 : List Nat := 5387 :: root566Tail7
def root566Tail5 : List Nat := 5351 :: root566Tail6
def root566Tail4 : List Nat := 5171 :: root566Tail5
def root566Tail3 : List Nat := 5081 :: root566Tail4
def root566Tail2 : List Nat := 4721 :: root566Tail3
def root566Tail1 : List Nat := 4493 :: root566Tail2
def root566Tail0 : List Nat := 4409 :: root566Tail1
def root566Candidates0 : List Nat := [9371, 16061]
set_option maxRecDepth 8192 in
theorem root566Candidates0_checked : candidates edge 4 4133 4409 root566Tail1 winner.bound = root566Candidates0 := by
  unfold candidates
  have row : edge 4409 = (SortedNatTree.contains FastRows.row4409) := by
    change Data.row4409.contains = (SortedNatTree.contains FastRows.row4409)
    exact FastRows.row4409_checked.symm
  rw [row]
  decide
theorem root566Step0 : scan edge (search edge 3) 4 [4133] 4133 root566Tail0 winner =
    scan edge (search edge 3) 4 [4133] 4133 root566Tail1 winner := by
  rw [root566Tail0, scan, if_neg (by decide), root566Candidates0_checked]
  change scan edge (search edge 3) 4 [4133] 4133 root566Tail1
    (if 3 ≤ root566Candidates0.length then search edge 3 [4133, 4409] root566Candidates0 8542 winner else winner) = _
  rw [if_neg (by decide)]
def root566Candidates1 : List Nat := [6173]
set_option maxRecDepth 8192 in
theorem root566Candidates1_checked : candidates edge 4 4133 4493 root566Tail2 winner.bound = root566Candidates1 := by
  unfold candidates
  have row : edge 4493 = (SortedNatTree.contains FastRows.row4493) := by
    change Data.row4493.contains = (SortedNatTree.contains FastRows.row4493)
    exact FastRows.row4493_checked.symm
  rw [row]
  decide
theorem root566Step1 : scan edge (search edge 3) 4 [4133] 4133 root566Tail1 winner =
    scan edge (search edge 3) 4 [4133] 4133 root566Tail2 winner := by
  rw [root566Tail1, scan, if_neg (by decide), root566Candidates1_checked]
  change scan edge (search edge 3) 4 [4133] 4133 root566Tail2
    (if 3 ≤ root566Candidates1.length then search edge 3 [4133, 4493] root566Candidates1 8626 winner else winner) = _
  rw [if_neg (by decide)]
def root566Candidates2 : List Nat := [12269]
set_option maxRecDepth 8192 in
theorem root566Candidates2_checked : candidates edge 4 4133 4721 root566Tail3 winner.bound = root566Candidates2 := by
  unfold candidates
  have row : edge 4721 = (SortedNatTree.contains FastRows.row4721) := by
    change Data.row4721.contains = (SortedNatTree.contains FastRows.row4721)
    exact FastRows.row4721_checked.symm
  rw [row]
  decide
theorem root566Step2 : scan edge (search edge 3) 4 [4133] 4133 root566Tail2 winner =
    scan edge (search edge 3) 4 [4133] 4133 root566Tail3 winner := by
  rw [root566Tail2, scan, if_neg (by decide), root566Candidates2_checked]
  change scan edge (search edge 3) 4 [4133] 4133 root566Tail3
    (if 3 ≤ root566Candidates2.length then search edge 3 [4133, 4721] root566Candidates2 8854 winner else winner) = _
  rw [if_neg (by decide)]
def root566Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root566Candidates3_checked : candidates edge 4 4133 5081 root566Tail4 winner.bound = root566Candidates3 := by
  unfold candidates
  have row : edge 5081 = (SortedNatTree.contains FastRows.row5081) := by
    change Data.row5081.contains = (SortedNatTree.contains FastRows.row5081)
    exact FastRows.row5081_checked.symm
  rw [row]
  decide
theorem root566Step3 : scan edge (search edge 3) 4 [4133] 4133 root566Tail3 winner =
    scan edge (search edge 3) 4 [4133] 4133 root566Tail4 winner := by
  rw [root566Tail3, scan, if_neg (by decide), root566Candidates3_checked]
  change scan edge (search edge 3) 4 [4133] 4133 root566Tail4
    (if 3 ≤ root566Candidates3.length then search edge 3 [4133, 5081] root566Candidates3 9214 winner else winner) = _
  rw [if_neg (by decide)]
def root566Candidates4 : List Nat := [8111]
set_option maxRecDepth 8192 in
theorem root566Candidates4_checked : candidates edge 4 4133 5171 root566Tail5 winner.bound = root566Candidates4 := by
  unfold candidates
  have row : edge 5171 = (SortedNatTree.contains FastRows.row5171) := by
    change Data.row5171.contains = (SortedNatTree.contains FastRows.row5171)
    exact FastRows.row5171_checked.symm
  rw [row]
  decide
theorem root566Step4 : scan edge (search edge 3) 4 [4133] 4133 root566Tail4 winner =
    scan edge (search edge 3) 4 [4133] 4133 root566Tail5 winner := by
  rw [root566Tail4, scan, if_neg (by decide), root566Candidates4_checked]
  change scan edge (search edge 3) 4 [4133] 4133 root566Tail5
    (if 3 ≤ root566Candidates4.length then search edge 3 [4133, 5171] root566Candidates4 9304 winner else winner) = _
  rw [if_neg (by decide)]
def root566Candidates5 : List Nat := [15803]
set_option maxRecDepth 8192 in
theorem root566Candidates5_checked : candidates edge 4 4133 5351 root566Tail6 winner.bound = root566Candidates5 := by
  unfold candidates
  have row : edge 5351 = (SortedNatTree.contains FastRows.row5351) := by
    change Data.row5351.contains = (SortedNatTree.contains FastRows.row5351)
    exact FastRows.row5351_checked.symm
  rw [row]
  decide
theorem root566Step5 : scan edge (search edge 3) 4 [4133] 4133 root566Tail5 winner =
    scan edge (search edge 3) 4 [4133] 4133 root566Tail6 winner := by
  rw [root566Tail5, scan, if_neg (by decide), root566Candidates5_checked]
  change scan edge (search edge 3) 4 [4133] 4133 root566Tail6
    (if 3 ≤ root566Candidates5.length then search edge 3 [4133, 5351] root566Candidates5 9484 winner else winner) = _
  rw [if_neg (by decide)]
def root566Candidates6 : List Nat := [9173, 12569]
set_option maxRecDepth 8192 in
theorem root566Candidates6_checked : candidates edge 4 4133 5387 root566Tail7 winner.bound = root566Candidates6 := by
  unfold candidates
  have row : edge 5387 = (SortedNatTree.contains FastRows.row5387) := by
    change Data.row5387.contains = (SortedNatTree.contains FastRows.row5387)
    exact FastRows.row5387_checked.symm
  rw [row]
  decide
theorem root566Step6 : scan edge (search edge 3) 4 [4133] 4133 root566Tail6 winner =
    scan edge (search edge 3) 4 [4133] 4133 root566Tail7 winner := by
  rw [root566Tail6, scan, if_neg (by decide), root566Candidates6_checked]
  change scan edge (search edge 3) 4 [4133] 4133 root566Tail7
    (if 3 ≤ root566Candidates6.length then search edge 3 [4133, 5387] root566Candidates6 9520 winner else winner) = _
  rw [if_neg (by decide)]
theorem root566Scan7 : scan edge (search edge 3) 4 [4133] 4133 root566Tail7 winner = winner := by rw [root566Tail7, scan, if_pos (by decide)]
theorem root566Scan6 : scan edge (search edge 3) 4 [4133] 4133 root566Tail6 winner = winner :=
  (root566Step6).trans root566Scan7
theorem root566Scan5 : scan edge (search edge 3) 4 [4133] 4133 root566Tail5 winner = winner :=
  (root566Step5).trans root566Scan6
theorem root566Scan4 : scan edge (search edge 3) 4 [4133] 4133 root566Tail4 winner = winner :=
  (root566Step4).trans root566Scan5
theorem root566Scan3 : scan edge (search edge 3) 4 [4133] 4133 root566Tail3 winner = winner :=
  (root566Step3).trans root566Scan4
theorem root566Scan2 : scan edge (search edge 3) 4 [4133] 4133 root566Tail2 winner = winner :=
  (root566Step2).trans root566Scan3
theorem root566Scan1 : scan edge (search edge 3) 4 [4133] 4133 root566Tail1 winner = winner :=
  (root566Step1).trans root566Scan2
theorem root566Scan0 : scan edge (search edge 3) 4 [4133] 4133 root566Tail0 winner = winner :=
  (root566Step0).trans root566Scan1
set_option maxRecDepth 4096 in
theorem root566Branch : search edge 4 [4133] root566Candidates 4133 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4133] 4133 root566Tail0 winner = winner
  exact root566Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
