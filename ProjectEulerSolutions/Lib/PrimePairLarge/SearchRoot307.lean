import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root307Candidates : List Nat := Data.row2053
set_option maxRecDepth 8192 in
theorem root307Candidates_checked : candidates edge 5 0 2053 tail308 winner.bound = root307Candidates := by
  apply candidates_eq_row edge 5 0 2053 tail308 root307Candidates winner.bound tail308_ordered
  · decide
  · decide
  · funext q
    rfl
def root307Tail35 : List Nat := []
def root307Tail34 : List Nat := 23497 :: root307Tail35
def root307Tail33 : List Nat := 23143 :: root307Tail34
def root307Tail32 : List Nat := 23029 :: root307Tail33
def root307Tail31 : List Nat := 22111 :: root307Tail32
def root307Tail30 : List Nat := 21799 :: root307Tail31
def root307Tail29 : List Nat := 19069 :: root307Tail30
def root307Tail28 : List Nat := 18787 :: root307Tail29
def root307Tail27 : List Nat := 18523 :: root307Tail28
def root307Tail26 : List Nat := 18181 :: root307Tail27
def root307Tail25 : List Nat := 17257 :: root307Tail26
def root307Tail24 : List Nat := 17191 :: root307Tail25
def root307Tail23 : List Nat := 16879 :: root307Tail24
def root307Tail22 : List Nat := 16249 :: root307Tail23
def root307Tail21 : List Nat := 16087 :: root307Tail22
def root307Tail20 : List Nat := 15937 :: root307Tail21
def root307Tail19 : List Nat := 15889 :: root307Tail20
def root307Tail18 : List Nat := 14929 :: root307Tail19
def root307Tail17 : List Nat := 14419 :: root307Tail18
def root307Tail16 : List Nat := 13921 :: root307Tail17
def root307Tail15 : List Nat := 13627 :: root307Tail16
def root307Tail14 : List Nat := 12781 :: root307Tail15
def root307Tail13 : List Nat := 12097 :: root307Tail14
def root307Tail12 : List Nat := 8923 :: root307Tail13
def root307Tail11 : List Nat := 8521 :: root307Tail12
def root307Tail10 : List Nat := 8317 :: root307Tail11
def root307Tail9 : List Nat := 6619 :: root307Tail10
def root307Tail8 : List Nat := 6421 :: root307Tail9
def root307Tail7 : List Nat := 6361 :: root307Tail8
def root307Tail6 : List Nat := 6343 :: root307Tail7
def root307Tail5 : List Nat := 5653 :: root307Tail6
def root307Tail4 : List Nat := 4783 :: root307Tail5
def root307Tail3 : List Nat := 4729 :: root307Tail4
def root307Tail2 : List Nat := 3361 :: root307Tail3
def root307Tail1 : List Nat := 3049 :: root307Tail2
def root307Tail0 : List Nat := 2797 :: root307Tail1
def root307Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root307Candidates0_checked : candidates edge 4 2053 2797 root307Tail1 winner.bound = root307Candidates0 := by
  unfold candidates
  have row : edge 2797 = (SortedNatTree.contains FastRows.row2797) := by
    change Data.row2797.contains = (SortedNatTree.contains FastRows.row2797)
    exact FastRows.row2797_checked.symm
  rw [row]
  decide
theorem root307Step0 : scan edge (search edge 3) 4 [2053] 2053 root307Tail0 winner =
    scan edge (search edge 3) 4 [2053] 2053 root307Tail1 winner := by
  rw [root307Tail0, scan, if_neg (by decide), root307Candidates0_checked]
  change scan edge (search edge 3) 4 [2053] 2053 root307Tail1
    (if 3 ≤ root307Candidates0.length then search edge 3 [2053, 2797] root307Candidates0 4850 winner else winner) = _
  rw [if_neg (by decide)]
def root307Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root307Candidates1_checked : candidates edge 4 2053 3049 root307Tail2 winner.bound = root307Candidates1 := by
  unfold candidates
  have row : edge 3049 = (SortedNatTree.contains FastRows.row3049) := by
    change Data.row3049.contains = (SortedNatTree.contains FastRows.row3049)
    exact FastRows.row3049_checked.symm
  rw [row]
  decide
theorem root307Step1 : scan edge (search edge 3) 4 [2053] 2053 root307Tail1 winner =
    scan edge (search edge 3) 4 [2053] 2053 root307Tail2 winner := by
  rw [root307Tail1, scan, if_neg (by decide), root307Candidates1_checked]
  change scan edge (search edge 3) 4 [2053] 2053 root307Tail2
    (if 3 ≤ root307Candidates1.length then search edge 3 [2053, 3049] root307Candidates1 5102 winner else winner) = _
  rw [if_neg (by decide)]
def root307Candidates2 : List Nat := [6343]
set_option maxRecDepth 8192 in
theorem root307Candidates2_checked : candidates edge 4 2053 3361 root307Tail3 winner.bound = root307Candidates2 := by
  unfold candidates
  have row : edge 3361 = (SortedNatTree.contains FastRows.row3361) := by
    change Data.row3361.contains = (SortedNatTree.contains FastRows.row3361)
    exact FastRows.row3361_checked.symm
  rw [row]
  decide
theorem root307Step2 : scan edge (search edge 3) 4 [2053] 2053 root307Tail2 winner =
    scan edge (search edge 3) 4 [2053] 2053 root307Tail3 winner := by
  rw [root307Tail2, scan, if_neg (by decide), root307Candidates2_checked]
  change scan edge (search edge 3) 4 [2053] 2053 root307Tail3
    (if 3 ≤ root307Candidates2.length then search edge 3 [2053, 3361] root307Candidates2 5414 winner else winner) = _
  rw [if_neg (by decide)]
def root307Candidates3 : List Nat := [16879, 18181]
set_option maxRecDepth 8192 in
theorem root307Candidates3_checked : candidates edge 4 2053 4729 root307Tail4 winner.bound = root307Candidates3 := by
  unfold candidates
  have row : edge 4729 = (SortedNatTree.contains FastRows.row4729) := by
    change Data.row4729.contains = (SortedNatTree.contains FastRows.row4729)
    exact FastRows.row4729_checked.symm
  rw [row]
  decide
theorem root307Step3 : scan edge (search edge 3) 4 [2053] 2053 root307Tail3 winner =
    scan edge (search edge 3) 4 [2053] 2053 root307Tail4 winner := by
  rw [root307Tail3, scan, if_neg (by decide), root307Candidates3_checked]
  change scan edge (search edge 3) 4 [2053] 2053 root307Tail4
    (if 3 ≤ root307Candidates3.length then search edge 3 [2053, 4729] root307Candidates3 6782 winner else winner) = _
  rw [if_neg (by decide)]
def root307Candidates4 : List Nat := [16087]
set_option maxRecDepth 8192 in
theorem root307Candidates4_checked : candidates edge 4 2053 4783 root307Tail5 winner.bound = root307Candidates4 := by
  unfold candidates
  have row : edge 4783 = (SortedNatTree.contains FastRows.row4783) := by
    change Data.row4783.contains = (SortedNatTree.contains FastRows.row4783)
    exact FastRows.row4783_checked.symm
  rw [row]
  decide
theorem root307Step4 : scan edge (search edge 3) 4 [2053] 2053 root307Tail4 winner =
    scan edge (search edge 3) 4 [2053] 2053 root307Tail5 winner := by
  rw [root307Tail4, scan, if_neg (by decide), root307Candidates4_checked]
  change scan edge (search edge 3) 4 [2053] 2053 root307Tail5
    (if 3 ≤ root307Candidates4.length then search edge 3 [2053, 4783] root307Candidates4 6836 winner else winner) = _
  rw [if_neg (by decide)]
def root307Candidates5 : List Nat := [6619, 13921]
set_option maxRecDepth 8192 in
theorem root307Candidates5_checked : candidates edge 4 2053 5653 root307Tail6 winner.bound = root307Candidates5 := by
  unfold candidates
  have row : edge 5653 = (SortedNatTree.contains FastRows.row5653) := by
    change Data.row5653.contains = (SortedNatTree.contains FastRows.row5653)
    exact FastRows.row5653_checked.symm
  rw [row]
  decide
theorem root307Step5 : scan edge (search edge 3) 4 [2053] 2053 root307Tail5 winner =
    scan edge (search edge 3) 4 [2053] 2053 root307Tail6 winner := by
  rw [root307Tail5, scan, if_neg (by decide), root307Candidates5_checked]
  change scan edge (search edge 3) 4 [2053] 2053 root307Tail6
    (if 3 ≤ root307Candidates5.length then search edge 3 [2053, 5653] root307Candidates5 7706 winner else winner) = _
  rw [if_neg (by decide)]
theorem root307Scan6 : scan edge (search edge 3) 4 [2053] 2053 root307Tail6 winner = winner := by rw [root307Tail6, scan, if_pos (by decide)]
theorem root307Scan5 : scan edge (search edge 3) 4 [2053] 2053 root307Tail5 winner = winner :=
  (root307Step5).trans root307Scan6
theorem root307Scan4 : scan edge (search edge 3) 4 [2053] 2053 root307Tail4 winner = winner :=
  (root307Step4).trans root307Scan5
theorem root307Scan3 : scan edge (search edge 3) 4 [2053] 2053 root307Tail3 winner = winner :=
  (root307Step3).trans root307Scan4
theorem root307Scan2 : scan edge (search edge 3) 4 [2053] 2053 root307Tail2 winner = winner :=
  (root307Step2).trans root307Scan3
theorem root307Scan1 : scan edge (search edge 3) 4 [2053] 2053 root307Tail1 winner = winner :=
  (root307Step1).trans root307Scan2
theorem root307Scan0 : scan edge (search edge 3) 4 [2053] 2053 root307Tail0 winner = winner :=
  (root307Step0).trans root307Scan1
set_option maxRecDepth 4096 in
theorem root307Branch : search edge 4 [2053] root307Candidates 2053 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2053] 2053 root307Tail0 winner = winner
  exact root307Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
