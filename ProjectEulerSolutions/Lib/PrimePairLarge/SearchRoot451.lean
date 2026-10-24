import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root451Candidates : List Nat := Data.row3209
set_option maxRecDepth 8192 in
theorem root451Candidates_checked : candidates edge 5 0 3209 tail452 winner.bound = root451Candidates := by
  apply candidates_eq_row edge 5 0 3209 tail452 root451Candidates winner.bound tail452_ordered
  · decide
  · decide
  · funext q
    rfl
def root451Tail40 : List Nat := []
def root451Tail39 : List Nat := 22739 :: root451Tail40
def root451Tail38 : List Nat := 22229 :: root451Tail39
def root451Tail37 : List Nat := 21929 :: root451Tail38
def root451Tail36 : List Nat := 21383 :: root451Tail37
def root451Tail35 : List Nat := 21011 :: root451Tail36
def root451Tail34 : List Nat := 20681 :: root451Tail35
def root451Tail33 : List Nat := 19889 :: root451Tail34
def root451Tail32 : List Nat := 19301 :: root451Tail33
def root451Tail31 : List Nat := 19163 :: root451Tail32
def root451Tail30 : List Nat := 18119 :: root451Tail31
def root451Tail29 : List Nat := 18077 :: root451Tail30
def root451Tail28 : List Nat := 17159 :: root451Tail29
def root451Tail27 : List Nat := 16487 :: root451Tail28
def root451Tail26 : List Nat := 16067 :: root451Tail27
def root451Tail25 : List Nat := 14051 :: root451Tail26
def root451Tail24 : List Nat := 13757 :: root451Tail25
def root451Tail23 : List Nat := 13613 :: root451Tail24
def root451Tail22 : List Nat := 12821 :: root451Tail23
def root451Tail21 : List Nat := 12539 :: root451Tail22
def root451Tail20 : List Nat := 11519 :: root451Tail21
def root451Tail19 : List Nat := 11447 :: root451Tail20
def root451Tail18 : List Nat := 10979 :: root451Tail19
def root451Tail17 : List Nat := 9497 :: root451Tail18
def root451Tail16 : List Nat := 9293 :: root451Tail17
def root451Tail15 : List Nat := 9257 :: root451Tail16
def root451Tail14 : List Nat := 8861 :: root451Tail15
def root451Tail13 : List Nat := 8699 :: root451Tail14
def root451Tail12 : List Nat := 7817 :: root451Tail13
def root451Tail11 : List Nat := 7523 :: root451Tail12
def root451Tail10 : List Nat := 7433 :: root451Tail11
def root451Tail9 : List Nat := 7187 :: root451Tail10
def root451Tail8 : List Nat := 6761 :: root451Tail9
def root451Tail7 : List Nat := 6653 :: root451Tail8
def root451Tail6 : List Nat := 5717 :: root451Tail7
def root451Tail5 : List Nat := 5471 :: root451Tail6
def root451Tail4 : List Nat := 5417 :: root451Tail5
def root451Tail3 : List Nat := 5171 :: root451Tail4
def root451Tail2 : List Nat := 4721 :: root451Tail3
def root451Tail1 : List Nat := 3833 :: root451Tail2
def root451Tail0 : List Nat := 3617 :: root451Tail1
def root451Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root451Candidates0_checked : candidates edge 4 3209 3617 root451Tail1 winner.bound = root451Candidates0 := by
  unfold candidates
  have row : edge 3617 = (SortedNatTree.contains FastRows.row3617) := by
    change Data.row3617.contains = (SortedNatTree.contains FastRows.row3617)
    exact FastRows.row3617_checked.symm
  rw [row]
  decide
theorem root451Step0 : scan edge (search edge 3) 4 [3209] 3209 root451Tail0 winner =
    scan edge (search edge 3) 4 [3209] 3209 root451Tail1 winner := by
  rw [root451Tail0, scan, if_neg (by decide), root451Candidates0_checked]
  change scan edge (search edge 3) 4 [3209] 3209 root451Tail1
    (if 3 ≤ root451Candidates0.length then search edge 3 [3209, 3617] root451Candidates0 6826 winner else winner) = _
  rw [if_neg (by decide)]
def root451Candidates1 : List Nat := [11447]
set_option maxRecDepth 8192 in
theorem root451Candidates1_checked : candidates edge 4 3209 3833 root451Tail2 winner.bound = root451Candidates1 := by
  unfold candidates
  have row : edge 3833 = (SortedNatTree.contains FastRows.row3833) := by
    change Data.row3833.contains = (SortedNatTree.contains FastRows.row3833)
    exact FastRows.row3833_checked.symm
  rw [row]
  decide
theorem root451Step1 : scan edge (search edge 3) 4 [3209] 3209 root451Tail1 winner =
    scan edge (search edge 3) 4 [3209] 3209 root451Tail2 winner := by
  rw [root451Tail1, scan, if_neg (by decide), root451Candidates1_checked]
  change scan edge (search edge 3) 4 [3209] 3209 root451Tail2
    (if 3 ≤ root451Candidates1.length then search edge 3 [3209, 3833] root451Candidates1 7042 winner else winner) = _
  rw [if_neg (by decide)]
def root451Candidates2 : List Nat := [5417, 9257]
set_option maxRecDepth 8192 in
theorem root451Candidates2_checked : candidates edge 4 3209 4721 root451Tail3 winner.bound = root451Candidates2 := by
  unfold candidates
  have row : edge 4721 = (SortedNatTree.contains FastRows.row4721) := by
    change Data.row4721.contains = (SortedNatTree.contains FastRows.row4721)
    exact FastRows.row4721_checked.symm
  rw [row]
  decide
theorem root451Step2 : scan edge (search edge 3) 4 [3209] 3209 root451Tail2 winner =
    scan edge (search edge 3) 4 [3209] 3209 root451Tail3 winner := by
  rw [root451Tail2, scan, if_neg (by decide), root451Candidates2_checked]
  change scan edge (search edge 3) 4 [3209] 3209 root451Tail3
    (if 3 ≤ root451Candidates2.length then search edge 3 [3209, 4721] root451Candidates2 7930 winner else winner) = _
  rw [if_neg (by decide)]
def root451Candidates3 : List Nat := [7187]
set_option maxRecDepth 8192 in
theorem root451Candidates3_checked : candidates edge 4 3209 5171 root451Tail4 winner.bound = root451Candidates3 := by
  unfold candidates
  have row : edge 5171 = (SortedNatTree.contains FastRows.row5171) := by
    change Data.row5171.contains = (SortedNatTree.contains FastRows.row5171)
    exact FastRows.row5171_checked.symm
  rw [row]
  decide
theorem root451Step3 : scan edge (search edge 3) 4 [3209] 3209 root451Tail3 winner =
    scan edge (search edge 3) 4 [3209] 3209 root451Tail4 winner := by
  rw [root451Tail3, scan, if_neg (by decide), root451Candidates3_checked]
  change scan edge (search edge 3) 4 [3209] 3209 root451Tail4
    (if 3 ≤ root451Candidates3.length then search edge 3 [3209, 5171] root451Candidates3 8380 winner else winner) = _
  rw [if_neg (by decide)]
def root451Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root451Candidates4_checked : candidates edge 4 3209 5417 root451Tail5 winner.bound = root451Candidates4 := by
  unfold candidates
  have row : edge 5417 = (SortedNatTree.contains FastRows.row5417) := by
    change Data.row5417.contains = (SortedNatTree.contains FastRows.row5417)
    exact FastRows.row5417_checked.symm
  rw [row]
  decide
theorem root451Step4 : scan edge (search edge 3) 4 [3209] 3209 root451Tail4 winner =
    scan edge (search edge 3) 4 [3209] 3209 root451Tail5 winner := by
  rw [root451Tail4, scan, if_neg (by decide), root451Candidates4_checked]
  change scan edge (search edge 3) 4 [3209] 3209 root451Tail5
    (if 3 ≤ root451Candidates4.length then search edge 3 [3209, 5417] root451Candidates4 8626 winner else winner) = _
  rw [if_neg (by decide)]
def root451Candidates5 : List Nat := [6653, 11519]
set_option maxRecDepth 8192 in
theorem root451Candidates5_checked : candidates edge 4 3209 5471 root451Tail6 winner.bound = root451Candidates5 := by
  unfold candidates
  have row : edge 5471 = (SortedNatTree.contains FastRows.row5471) := by
    change Data.row5471.contains = (SortedNatTree.contains FastRows.row5471)
    exact FastRows.row5471_checked.symm
  rw [row]
  decide
theorem root451Step5 : scan edge (search edge 3) 4 [3209] 3209 root451Tail5 winner =
    scan edge (search edge 3) 4 [3209] 3209 root451Tail6 winner := by
  rw [root451Tail5, scan, if_neg (by decide), root451Candidates5_checked]
  change scan edge (search edge 3) 4 [3209] 3209 root451Tail6
    (if 3 ≤ root451Candidates5.length then search edge 3 [3209, 5471] root451Candidates5 8680 winner else winner) = _
  rw [if_neg (by decide)]
theorem root451Scan6 : scan edge (search edge 3) 4 [3209] 3209 root451Tail6 winner = winner := by rw [root451Tail6, scan, if_pos (by decide)]
theorem root451Scan5 : scan edge (search edge 3) 4 [3209] 3209 root451Tail5 winner = winner :=
  (root451Step5).trans root451Scan6
theorem root451Scan4 : scan edge (search edge 3) 4 [3209] 3209 root451Tail4 winner = winner :=
  (root451Step4).trans root451Scan5
theorem root451Scan3 : scan edge (search edge 3) 4 [3209] 3209 root451Tail3 winner = winner :=
  (root451Step3).trans root451Scan4
theorem root451Scan2 : scan edge (search edge 3) 4 [3209] 3209 root451Tail2 winner = winner :=
  (root451Step2).trans root451Scan3
theorem root451Scan1 : scan edge (search edge 3) 4 [3209] 3209 root451Tail1 winner = winner :=
  (root451Step1).trans root451Scan2
theorem root451Scan0 : scan edge (search edge 3) 4 [3209] 3209 root451Tail0 winner = winner :=
  (root451Step0).trans root451Scan1
set_option maxRecDepth 4096 in
theorem root451Branch : search edge 4 [3209] root451Candidates 3209 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3209] 3209 root451Tail0 winner = winner
  exact root451Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
