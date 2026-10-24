import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root389Candidates : List Nat := Data.row2693
set_option maxRecDepth 8192 in
theorem root389Candidates_checked : candidates edge 5 0 2693 tail390 winner.bound = root389Candidates := by
  apply candidates_eq_row edge 5 0 2693 tail390 root389Candidates winner.bound tail390_ordered
  · decide
  · decide
  · funext q
    rfl
def root389Tail48 : List Nat := []
def root389Tail47 : List Nat := 23039 :: root389Tail48
def root389Tail46 : List Nat := 21863 :: root389Tail47
def root389Tail45 : List Nat := 20393 :: root389Tail46
def root389Tail44 : List Nat := 19991 :: root389Tail45
def root389Tail43 : List Nat := 18593 :: root389Tail44
def root389Tail42 : List Nat := 18443 :: root389Tail43
def root389Tail41 : List Nat := 18341 :: root389Tail42
def root389Tail40 : List Nat := 18089 :: root389Tail41
def root389Tail39 : List Nat := 17681 :: root389Tail40
def root389Tail38 : List Nat := 17099 :: root389Tail39
def root389Tail37 : List Nat := 16319 :: root389Tail38
def root389Tail36 : List Nat := 15797 :: root389Tail37
def root389Tail35 : List Nat := 15161 :: root389Tail36
def root389Tail34 : List Nat := 15077 :: root389Tail35
def root389Tail33 : List Nat := 14327 :: root389Tail34
def root389Tail32 : List Nat := 13907 :: root389Tail33
def root389Tail31 : List Nat := 13553 :: root389Tail32
def root389Tail30 : List Nat := 13217 :: root389Tail31
def root389Tail29 : List Nat := 12893 :: root389Tail30
def root389Tail28 : List Nat := 12809 :: root389Tail29
def root389Tail27 : List Nat := 12713 :: root389Tail28
def root389Tail26 : List Nat := 12671 :: root389Tail27
def root389Tail25 : List Nat := 12269 :: root389Tail26
def root389Tail24 : List Nat := 12161 :: root389Tail25
def root389Tail23 : List Nat := 11177 :: root389Tail24
def root389Tail22 : List Nat := 9833 :: root389Tail23
def root389Tail21 : List Nat := 9533 :: root389Tail22
def root389Tail20 : List Nat := 9431 :: root389Tail21
def root389Tail19 : List Nat := 9239 :: root389Tail20
def root389Tail18 : List Nat := 8501 :: root389Tail19
def root389Tail17 : List Nat := 8123 :: root389Tail18
def root389Tail16 : List Nat := 7901 :: root389Tail17
def root389Tail15 : List Nat := 7331 :: root389Tail16
def root389Tail14 : List Nat := 7103 :: root389Tail15
def root389Tail13 : List Nat := 7013 :: root389Tail14
def root389Tail12 : List Nat := 6779 :: root389Tail13
def root389Tail11 : List Nat := 6263 :: root389Tail12
def root389Tail10 : List Nat := 6257 :: root389Tail11
def root389Tail9 : List Nat := 6113 :: root389Tail10
def root389Tail8 : List Nat := 6011 :: root389Tail9
def root389Tail7 : List Nat := 5903 :: root389Tail8
def root389Tail6 : List Nat := 5879 :: root389Tail7
def root389Tail5 : List Nat := 5651 :: root389Tail6
def root389Tail4 : List Nat := 4931 :: root389Tail5
def root389Tail3 : List Nat := 4373 :: root389Tail4
def root389Tail2 : List Nat := 4139 :: root389Tail3
def root389Tail1 : List Nat := 4079 :: root389Tail2
def root389Tail0 : List Nat := 4049 :: root389Tail1
def root389Candidates0 : List Nat := [17099]
set_option maxRecDepth 8192 in
theorem root389Candidates0_checked : candidates edge 4 2693 4049 root389Tail1 winner.bound = root389Candidates0 := by
  unfold candidates
  have row : edge 4049 = (SortedNatTree.contains FastRows.row4049) := by
    change Data.row4049.contains = (SortedNatTree.contains FastRows.row4049)
    exact FastRows.row4049_checked.symm
  rw [row]
  decide
theorem root389Step0 : scan edge (search edge 3) 4 [2693] 2693 root389Tail0 winner =
    scan edge (search edge 3) 4 [2693] 2693 root389Tail1 winner := by
  rw [root389Tail0, scan, if_neg (by decide), root389Candidates0_checked]
  change scan edge (search edge 3) 4 [2693] 2693 root389Tail1
    (if 3 ≤ root389Candidates0.length then search edge 3 [2693, 4049] root389Candidates0 6742 winner else winner) = _
  rw [if_neg (by decide)]
def root389Candidates1 : List Nat := [5879, 13553]
set_option maxRecDepth 8192 in
theorem root389Candidates1_checked : candidates edge 4 2693 4079 root389Tail2 winner.bound = root389Candidates1 := by
  unfold candidates
  have row : edge 4079 = (SortedNatTree.contains FastRows.row4079) := by
    change Data.row4079.contains = (SortedNatTree.contains FastRows.row4079)
    exact FastRows.row4079_checked.symm
  rw [row]
  decide
theorem root389Step1 : scan edge (search edge 3) 4 [2693] 2693 root389Tail1 winner =
    scan edge (search edge 3) 4 [2693] 2693 root389Tail2 winner := by
  rw [root389Tail1, scan, if_neg (by decide), root389Candidates1_checked]
  change scan edge (search edge 3) 4 [2693] 2693 root389Tail2
    (if 3 ≤ root389Candidates1.length then search edge 3 [2693, 4079] root389Candidates1 6772 winner else winner) = _
  rw [if_neg (by decide)]
def root389Candidates2 : List Nat := [6263, 11177]
set_option maxRecDepth 8192 in
theorem root389Candidates2_checked : candidates edge 4 2693 4139 root389Tail3 winner.bound = root389Candidates2 := by
  unfold candidates
  have row : edge 4139 = (SortedNatTree.contains FastRows.row4139) := by
    change Data.row4139.contains = (SortedNatTree.contains FastRows.row4139)
    exact FastRows.row4139_checked.symm
  rw [row]
  decide
theorem root389Step2 : scan edge (search edge 3) 4 [2693] 2693 root389Tail2 winner =
    scan edge (search edge 3) 4 [2693] 2693 root389Tail3 winner := by
  rw [root389Tail2, scan, if_neg (by decide), root389Candidates2_checked]
  change scan edge (search edge 3) 4 [2693] 2693 root389Tail3
    (if 3 ≤ root389Candidates2.length then search edge 3 [2693, 4139] root389Candidates2 6832 winner else winner) = _
  rw [if_neg (by decide)]
def root389Candidates3 : List Nat := [9431]
set_option maxRecDepth 8192 in
theorem root389Candidates3_checked : candidates edge 4 2693 4373 root389Tail4 winner.bound = root389Candidates3 := by
  unfold candidates
  have row : edge 4373 = (SortedNatTree.contains FastRows.row4373) := by
    change Data.row4373.contains = (SortedNatTree.contains FastRows.row4373)
    exact FastRows.row4373_checked.symm
  rw [row]
  decide
theorem root389Step3 : scan edge (search edge 3) 4 [2693] 2693 root389Tail3 winner =
    scan edge (search edge 3) 4 [2693] 2693 root389Tail4 winner := by
  rw [root389Tail3, scan, if_neg (by decide), root389Candidates3_checked]
  change scan edge (search edge 3) 4 [2693] 2693 root389Tail4
    (if 3 ≤ root389Candidates3.length then search edge 3 [2693, 4373] root389Candidates3 7066 winner else winner) = _
  rw [if_neg (by decide)]
def root389Candidates4 : List Nat := [8501, 9833, 14327]
set_option maxRecDepth 8192 in
theorem root389Candidates4_checked : candidates edge 4 2693 4931 root389Tail5 winner.bound = root389Candidates4 := by
  unfold candidates
  have row : edge 4931 = (SortedNatTree.contains FastRows.row4931) := by
    change Data.row4931.contains = (SortedNatTree.contains FastRows.row4931)
    exact FastRows.row4931_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root389Child4 : search edge 3 [2693, 4931] root389Candidates4 7624 winner = winner := by decide
theorem root389Step4 : scan edge (search edge 3) 4 [2693] 2693 root389Tail4 winner =
    scan edge (search edge 3) 4 [2693] 2693 root389Tail5 winner := by
  rw [root389Tail4, scan, if_neg (by decide), root389Candidates4_checked]
  change scan edge (search edge 3) 4 [2693] 2693 root389Tail5
    (if 3 ≤ root389Candidates4.length then search edge 3 [2693, 4931] root389Candidates4 7624 winner else winner) = _
  rw [if_pos (by decide), root389Child4]
def root389Candidates5 : List Nat := [6011, 13553, 14327]
set_option maxRecDepth 8192 in
theorem root389Candidates5_checked : candidates edge 4 2693 5651 root389Tail6 winner.bound = root389Candidates5 := by
  unfold candidates
  have row : edge 5651 = (SortedNatTree.contains FastRows.row5651) := by
    change Data.row5651.contains = (SortedNatTree.contains FastRows.row5651)
    exact FastRows.row5651_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root389Child5 : search edge 3 [2693, 5651] root389Candidates5 8344 winner = winner := by decide
theorem root389Step5 : scan edge (search edge 3) 4 [2693] 2693 root389Tail5 winner =
    scan edge (search edge 3) 4 [2693] 2693 root389Tail6 winner := by
  rw [root389Tail5, scan, if_neg (by decide), root389Candidates5_checked]
  change scan edge (search edge 3) 4 [2693] 2693 root389Tail6
    (if 3 ≤ root389Candidates5.length then search edge 3 [2693, 5651] root389Candidates5 8344 winner else winner) = _
  rw [if_pos (by decide), root389Child5]
theorem root389Scan6 : scan edge (search edge 3) 4 [2693] 2693 root389Tail6 winner = winner := by rw [root389Tail6, scan, if_pos (by decide)]
theorem root389Scan5 : scan edge (search edge 3) 4 [2693] 2693 root389Tail5 winner = winner :=
  (root389Step5).trans root389Scan6
theorem root389Scan4 : scan edge (search edge 3) 4 [2693] 2693 root389Tail4 winner = winner :=
  (root389Step4).trans root389Scan5
theorem root389Scan3 : scan edge (search edge 3) 4 [2693] 2693 root389Tail3 winner = winner :=
  (root389Step3).trans root389Scan4
theorem root389Scan2 : scan edge (search edge 3) 4 [2693] 2693 root389Tail2 winner = winner :=
  (root389Step2).trans root389Scan3
theorem root389Scan1 : scan edge (search edge 3) 4 [2693] 2693 root389Tail1 winner = winner :=
  (root389Step1).trans root389Scan2
theorem root389Scan0 : scan edge (search edge 3) 4 [2693] 2693 root389Tail0 winner = winner :=
  (root389Step0).trans root389Scan1
set_option maxRecDepth 4096 in
theorem root389Branch : search edge 4 [2693] root389Candidates 2693 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2693] 2693 root389Tail0 winner = winner
  exact root389Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
