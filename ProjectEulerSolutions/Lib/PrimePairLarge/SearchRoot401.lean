import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root401Candidates : List Nat := Data.row2777
set_option maxRecDepth 8192 in
theorem root401Candidates_checked : candidates edge 5 0 2777 tail402 winner.bound = root401Candidates := by
  apply candidates_eq_row edge 5 0 2777 tail402 root401Candidates winner.bound tail402_ordered
  · decide
  · decide
  · funext q
    rfl
def root401Tail46 : List Nat := []
def root401Tail45 : List Nat := 22811 :: root401Tail46
def root401Tail44 : List Nat := 21383 :: root401Tail45
def root401Tail43 : List Nat := 20903 :: root401Tail44
def root401Tail42 : List Nat := 20627 :: root401Tail43
def root401Tail41 : List Nat := 20249 :: root401Tail42
def root401Tail40 : List Nat := 20123 :: root401Tail41
def root401Tail39 : List Nat := 19073 :: root401Tail40
def root401Tail38 : List Nat := 18719 :: root401Tail39
def root401Tail37 : List Nat := 18671 :: root401Tail38
def root401Tail36 : List Nat := 17807 :: root401Tail37
def root401Tail35 : List Nat := 17681 :: root401Tail36
def root401Tail34 : List Nat := 17519 :: root401Tail35
def root401Tail33 : List Nat := 17333 :: root401Tail34
def root401Tail32 : List Nat := 17183 :: root401Tail33
def root401Tail31 : List Nat := 16619 :: root401Tail32
def root401Tail30 : List Nat := 15797 :: root401Tail31
def root401Tail29 : List Nat := 15497 :: root401Tail30
def root401Tail28 : List Nat := 15401 :: root401Tail29
def root401Tail27 : List Nat := 15161 :: root401Tail28
def root401Tail26 : List Nat := 14561 :: root401Tail27
def root401Tail25 : List Nat := 14447 :: root401Tail26
def root401Tail24 : List Nat := 14387 :: root401Tail25
def root401Tail23 : List Nat := 14243 :: root401Tail24
def root401Tail22 : List Nat := 13859 :: root401Tail23
def root401Tail21 : List Nat := 13781 :: root401Tail22
def root401Tail20 : List Nat := 13691 :: root401Tail21
def root401Tail19 : List Nat := 13127 :: root401Tail20
def root401Tail18 : List Nat := 12377 :: root401Tail19
def root401Tail17 : List Nat := 11597 :: root401Tail18
def root401Tail16 : List Nat := 10799 :: root401Tail17
def root401Tail15 : List Nat := 10781 :: root401Tail16
def root401Tail14 : List Nat := 10331 :: root401Tail15
def root401Tail13 : List Nat := 10163 :: root401Tail14
def root401Tail12 : List Nat := 9551 :: root401Tail13
def root401Tail11 : List Nat := 9533 :: root401Tail12
def root401Tail10 : List Nat := 8819 :: root401Tail11
def root401Tail9 : List Nat := 8291 :: root401Tail10
def root401Tail8 : List Nat := 8087 :: root401Tail9
def root401Tail7 : List Nat := 8039 :: root401Tail8
def root401Tail6 : List Nat := 6971 :: root401Tail7
def root401Tail5 : List Nat := 6473 :: root401Tail6
def root401Tail4 : List Nat := 6173 :: root401Tail5
def root401Tail3 : List Nat := 5477 :: root401Tail4
def root401Tail2 : List Nat := 4973 :: root401Tail3
def root401Tail1 : List Nat := 4793 :: root401Tail2
def root401Tail0 : List Nat := 4289 :: root401Tail1
def root401Candidates0 : List Nat := [6473, 8087, 8819]
set_option maxRecDepth 8192 in
theorem root401Candidates0_checked : candidates edge 4 2777 4289 root401Tail1 winner.bound = root401Candidates0 := by
  unfold candidates
  have row : edge 4289 = (SortedNatTree.contains FastRows.row4289) := by
    change Data.row4289.contains = (SortedNatTree.contains FastRows.row4289)
    exact FastRows.row4289_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root401Child0 : search edge 3 [2777, 4289] root401Candidates0 7066 winner = winner := by decide
theorem root401Step0 : scan edge (search edge 3) 4 [2777] 2777 root401Tail0 winner =
    scan edge (search edge 3) 4 [2777] 2777 root401Tail1 winner := by
  rw [root401Tail0, scan, if_neg (by decide), root401Candidates0_checked]
  change scan edge (search edge 3) 4 [2777] 2777 root401Tail1
    (if 3 ≤ root401Candidates0.length then search edge 3 [2777, 4289] root401Candidates0 7066 winner else winner) = _
  rw [if_pos (by decide), root401Child0]
def root401Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root401Candidates1_checked : candidates edge 4 2777 4793 root401Tail2 winner.bound = root401Candidates1 := by
  unfold candidates
  have row : edge 4793 = (SortedNatTree.contains FastRows.row4793) := by
    change Data.row4793.contains = (SortedNatTree.contains FastRows.row4793)
    exact FastRows.row4793_checked.symm
  rw [row]
  decide
theorem root401Step1 : scan edge (search edge 3) 4 [2777] 2777 root401Tail1 winner =
    scan edge (search edge 3) 4 [2777] 2777 root401Tail2 winner := by
  rw [root401Tail1, scan, if_neg (by decide), root401Candidates1_checked]
  change scan edge (search edge 3) 4 [2777] 2777 root401Tail2
    (if 3 ≤ root401Candidates1.length then search edge 3 [2777, 4793] root401Candidates1 7570 winner else winner) = _
  rw [if_neg (by decide)]
def root401Candidates2 : List Nat := [17519]
set_option maxRecDepth 8192 in
theorem root401Candidates2_checked : candidates edge 4 2777 4973 root401Tail3 winner.bound = root401Candidates2 := by
  unfold candidates
  have row : edge 4973 = (SortedNatTree.contains FastRows.row4973) := by
    change Data.row4973.contains = (SortedNatTree.contains FastRows.row4973)
    exact FastRows.row4973_checked.symm
  rw [row]
  decide
theorem root401Step2 : scan edge (search edge 3) 4 [2777] 2777 root401Tail2 winner =
    scan edge (search edge 3) 4 [2777] 2777 root401Tail3 winner := by
  rw [root401Tail2, scan, if_neg (by decide), root401Candidates2_checked]
  change scan edge (search edge 3) 4 [2777] 2777 root401Tail3
    (if 3 ≤ root401Candidates2.length then search edge 3 [2777, 4973] root401Candidates2 7750 winner else winner) = _
  rw [if_neg (by decide)]
def root401Candidates3 : List Nat := [8039, 9551]
set_option maxRecDepth 8192 in
theorem root401Candidates3_checked : candidates edge 4 2777 5477 root401Tail4 winner.bound = root401Candidates3 := by
  unfold candidates
  have row : edge 5477 = (SortedNatTree.contains FastRows.row5477) := by
    change Data.row5477.contains = (SortedNatTree.contains FastRows.row5477)
    exact FastRows.row5477_checked.symm
  rw [row]
  decide
theorem root401Step3 : scan edge (search edge 3) 4 [2777] 2777 root401Tail3 winner =
    scan edge (search edge 3) 4 [2777] 2777 root401Tail4 winner := by
  rw [root401Tail3, scan, if_neg (by decide), root401Candidates3_checked]
  change scan edge (search edge 3) 4 [2777] 2777 root401Tail4
    (if 3 ≤ root401Candidates3.length then search edge 3 [2777, 5477] root401Candidates3 8254 winner else winner) = _
  rw [if_neg (by decide)]
theorem root401Scan4 : scan edge (search edge 3) 4 [2777] 2777 root401Tail4 winner = winner := by rw [root401Tail4, scan, if_pos (by decide)]
theorem root401Scan3 : scan edge (search edge 3) 4 [2777] 2777 root401Tail3 winner = winner :=
  (root401Step3).trans root401Scan4
theorem root401Scan2 : scan edge (search edge 3) 4 [2777] 2777 root401Tail2 winner = winner :=
  (root401Step2).trans root401Scan3
theorem root401Scan1 : scan edge (search edge 3) 4 [2777] 2777 root401Tail1 winner = winner :=
  (root401Step1).trans root401Scan2
theorem root401Scan0 : scan edge (search edge 3) 4 [2777] 2777 root401Tail0 winner = winner :=
  (root401Step0).trans root401Scan1
set_option maxRecDepth 4096 in
theorem root401Branch : search edge 4 [2777] root401Candidates 2777 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2777] 2777 root401Tail0 winner = winner
  exact root401Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
