import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root350Candidates : List Nat := Data.row2381
set_option maxRecDepth 8192 in
theorem root350Candidates_checked : candidates edge 5 0 2381 tail351 winner.bound = root350Candidates := by
  apply candidates_eq_row edge 5 0 2381 tail351 root350Candidates winner.bound tail351_ordered
  · decide
  · decide
  · funext q
    rfl
def root350Tail36 : List Nat := []
def root350Tail35 : List Nat := 22739 :: root350Tail36
def root350Tail34 : List Nat := 21893 :: root350Tail35
def root350Tail33 : List Nat := 21347 :: root350Tail34
def root350Tail32 : List Nat := 20681 :: root350Tail33
def root350Tail31 : List Nat := 19727 :: root350Tail32
def root350Tail30 : List Nat := 15887 :: root350Tail31
def root350Tail29 : List Nat := 14753 :: root350Tail30
def root350Tail28 : List Nat := 14081 :: root350Tail29
def root350Tail27 : List Nat := 13313 :: root350Tail28
def root350Tail26 : List Nat := 13037 :: root350Tail27
def root350Tail25 : List Nat := 12071 :: root350Tail26
def root350Tail24 : List Nat := 11789 :: root350Tail25
def root350Tail23 : List Nat := 11519 :: root350Tail24
def root350Tail22 : List Nat := 11087 :: root350Tail23
def root350Tail21 : List Nat := 10979 :: root350Tail22
def root350Tail20 : List Nat := 10781 :: root350Tail21
def root350Tail19 : List Nat := 10193 :: root350Tail20
def root350Tail18 : List Nat := 9923 :: root350Tail19
def root350Tail17 : List Nat := 9491 :: root350Tail18
def root350Tail16 : List Nat := 9041 :: root350Tail17
def root350Tail15 : List Nat := 8807 :: root350Tail16
def root350Tail14 : List Nat := 8423 :: root350Tail15
def root350Tail13 : List Nat := 8369 :: root350Tail14
def root350Tail12 : List Nat := 8117 :: root350Tail13
def root350Tail11 : List Nat := 7853 :: root350Tail12
def root350Tail10 : List Nat := 7487 :: root350Tail11
def root350Tail9 : List Nat := 6857 :: root350Tail10
def root350Tail8 : List Nat := 6563 :: root350Tail9
def root350Tail7 : List Nat := 5807 :: root350Tail8
def root350Tail6 : List Nat := 5639 :: root350Tail7
def root350Tail5 : List Nat := 5501 :: root350Tail6
def root350Tail4 : List Nat := 5483 :: root350Tail5
def root350Tail3 : List Nat := 4073 :: root350Tail4
def root350Tail2 : List Nat := 3539 :: root350Tail3
def root350Tail1 : List Nat := 3221 :: root350Tail2
def root350Tail0 : List Nat := 2447 :: root350Tail1
def root350Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root350Candidates0_checked : candidates edge 4 2381 2447 root350Tail1 winner.bound = root350Candidates0 := by
  unfold candidates
  have row : edge 2447 = (SortedNatTree.contains FastRows.row2447) := by
    change Data.row2447.contains = (SortedNatTree.contains FastRows.row2447)
    exact FastRows.row2447_checked.symm
  rw [row]
  decide
theorem root350Step0 : scan edge (search edge 3) 4 [2381] 2381 root350Tail0 winner =
    scan edge (search edge 3) 4 [2381] 2381 root350Tail1 winner := by
  rw [root350Tail0, scan, if_neg (by decide), root350Candidates0_checked]
  change scan edge (search edge 3) 4 [2381] 2381 root350Tail1
    (if 3 ≤ root350Candidates0.length then search edge 3 [2381, 2447] root350Candidates0 4828 winner else winner) = _
  rw [if_neg (by decide)]
def root350Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root350Candidates1_checked : candidates edge 4 2381 3221 root350Tail2 winner.bound = root350Candidates1 := by
  unfold candidates
  have row : edge 3221 = (SortedNatTree.contains FastRows.row3221) := by
    change Data.row3221.contains = (SortedNatTree.contains FastRows.row3221)
    exact FastRows.row3221_checked.symm
  rw [row]
  decide
theorem root350Step1 : scan edge (search edge 3) 4 [2381] 2381 root350Tail1 winner =
    scan edge (search edge 3) 4 [2381] 2381 root350Tail2 winner := by
  rw [root350Tail1, scan, if_neg (by decide), root350Candidates1_checked]
  change scan edge (search edge 3) 4 [2381] 2381 root350Tail2
    (if 3 ≤ root350Candidates1.length then search edge 3 [2381, 3221] root350Candidates1 5602 winner else winner) = _
  rw [if_neg (by decide)]
def root350Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root350Candidates2_checked : candidates edge 4 2381 3539 root350Tail3 winner.bound = root350Candidates2 := by
  unfold candidates
  have row : edge 3539 = (SortedNatTree.contains FastRows.row3539) := by
    change Data.row3539.contains = (SortedNatTree.contains FastRows.row3539)
    exact FastRows.row3539_checked.symm
  rw [row]
  decide
theorem root350Step2 : scan edge (search edge 3) 4 [2381] 2381 root350Tail2 winner =
    scan edge (search edge 3) 4 [2381] 2381 root350Tail3 winner := by
  rw [root350Tail2, scan, if_neg (by decide), root350Candidates2_checked]
  change scan edge (search edge 3) 4 [2381] 2381 root350Tail3
    (if 3 ≤ root350Candidates2.length then search edge 3 [2381, 3539] root350Candidates2 5920 winner else winner) = _
  rw [if_neg (by decide)]
def root350Candidates3 : List Nat := [13037]
set_option maxRecDepth 8192 in
theorem root350Candidates3_checked : candidates edge 4 2381 4073 root350Tail4 winner.bound = root350Candidates3 := by
  unfold candidates
  have row : edge 4073 = (SortedNatTree.contains FastRows.row4073) := by
    change Data.row4073.contains = (SortedNatTree.contains FastRows.row4073)
    exact FastRows.row4073_checked.symm
  rw [row]
  decide
theorem root350Step3 : scan edge (search edge 3) 4 [2381] 2381 root350Tail3 winner =
    scan edge (search edge 3) 4 [2381] 2381 root350Tail4 winner := by
  rw [root350Tail3, scan, if_neg (by decide), root350Candidates3_checked]
  change scan edge (search edge 3) 4 [2381] 2381 root350Tail4
    (if 3 ≤ root350Candidates3.length then search edge 3 [2381, 4073] root350Candidates3 6454 winner else winner) = _
  rw [if_neg (by decide)]
def root350Candidates4 : List Nat := [7487, 8369, 8423]
set_option maxRecDepth 8192 in
theorem root350Candidates4_checked : candidates edge 4 2381 5483 root350Tail5 winner.bound = root350Candidates4 := by
  unfold candidates
  have row : edge 5483 = (SortedNatTree.contains FastRows.row5483) := by
    change Data.row5483.contains = (SortedNatTree.contains FastRows.row5483)
    exact FastRows.row5483_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root350Child4 : search edge 3 [2381, 5483] root350Candidates4 7864 winner = winner := by decide
theorem root350Step4 : scan edge (search edge 3) 4 [2381] 2381 root350Tail4 winner =
    scan edge (search edge 3) 4 [2381] 2381 root350Tail5 winner := by
  rw [root350Tail4, scan, if_neg (by decide), root350Candidates4_checked]
  change scan edge (search edge 3) 4 [2381] 2381 root350Tail5
    (if 3 ≤ root350Candidates4.length then search edge 3 [2381, 5483] root350Candidates4 7864 winner else winner) = _
  rw [if_pos (by decide), root350Child4]
def root350Candidates5 : List Nat := [9491, 11519]
set_option maxRecDepth 8192 in
theorem root350Candidates5_checked : candidates edge 4 2381 5501 root350Tail6 winner.bound = root350Candidates5 := by
  unfold candidates
  have row : edge 5501 = (SortedNatTree.contains FastRows.row5501) := by
    change Data.row5501.contains = (SortedNatTree.contains FastRows.row5501)
    exact FastRows.row5501_checked.symm
  rw [row]
  decide
theorem root350Step5 : scan edge (search edge 3) 4 [2381] 2381 root350Tail5 winner =
    scan edge (search edge 3) 4 [2381] 2381 root350Tail6 winner := by
  rw [root350Tail5, scan, if_neg (by decide), root350Candidates5_checked]
  change scan edge (search edge 3) 4 [2381] 2381 root350Tail6
    (if 3 ≤ root350Candidates5.length then search edge 3 [2381, 5501] root350Candidates5 7882 winner else winner) = _
  rw [if_neg (by decide)]
def root350Candidates6 : List Nat := [6563]
set_option maxRecDepth 8192 in
theorem root350Candidates6_checked : candidates edge 4 2381 5639 root350Tail7 winner.bound = root350Candidates6 := by
  unfold candidates
  have row : edge 5639 = (SortedNatTree.contains FastRows.row5639) := by
    change Data.row5639.contains = (SortedNatTree.contains FastRows.row5639)
    exact FastRows.row5639_checked.symm
  rw [row]
  decide
theorem root350Step6 : scan edge (search edge 3) 4 [2381] 2381 root350Tail6 winner =
    scan edge (search edge 3) 4 [2381] 2381 root350Tail7 winner := by
  rw [root350Tail6, scan, if_neg (by decide), root350Candidates6_checked]
  change scan edge (search edge 3) 4 [2381] 2381 root350Tail7
    (if 3 ≤ root350Candidates6.length then search edge 3 [2381, 5639] root350Candidates6 8020 winner else winner) = _
  rw [if_neg (by decide)]
def root350Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root350Candidates7_checked : candidates edge 4 2381 5807 root350Tail8 winner.bound = root350Candidates7 := by
  unfold candidates
  have row : edge 5807 = (SortedNatTree.contains FastRows.row5807) := by
    change Data.row5807.contains = (SortedNatTree.contains FastRows.row5807)
    exact FastRows.row5807_checked.symm
  rw [row]
  decide
theorem root350Step7 : scan edge (search edge 3) 4 [2381] 2381 root350Tail7 winner =
    scan edge (search edge 3) 4 [2381] 2381 root350Tail8 winner := by
  rw [root350Tail7, scan, if_neg (by decide), root350Candidates7_checked]
  change scan edge (search edge 3) 4 [2381] 2381 root350Tail8
    (if 3 ≤ root350Candidates7.length then search edge 3 [2381, 5807] root350Candidates7 8188 winner else winner) = _
  rw [if_neg (by decide)]
theorem root350Scan8 : scan edge (search edge 3) 4 [2381] 2381 root350Tail8 winner = winner := by rw [root350Tail8, scan, if_pos (by decide)]
theorem root350Scan7 : scan edge (search edge 3) 4 [2381] 2381 root350Tail7 winner = winner :=
  (root350Step7).trans root350Scan8
theorem root350Scan6 : scan edge (search edge 3) 4 [2381] 2381 root350Tail6 winner = winner :=
  (root350Step6).trans root350Scan7
theorem root350Scan5 : scan edge (search edge 3) 4 [2381] 2381 root350Tail5 winner = winner :=
  (root350Step5).trans root350Scan6
theorem root350Scan4 : scan edge (search edge 3) 4 [2381] 2381 root350Tail4 winner = winner :=
  (root350Step4).trans root350Scan5
theorem root350Scan3 : scan edge (search edge 3) 4 [2381] 2381 root350Tail3 winner = winner :=
  (root350Step3).trans root350Scan4
theorem root350Scan2 : scan edge (search edge 3) 4 [2381] 2381 root350Tail2 winner = winner :=
  (root350Step2).trans root350Scan3
theorem root350Scan1 : scan edge (search edge 3) 4 [2381] 2381 root350Tail1 winner = winner :=
  (root350Step1).trans root350Scan2
theorem root350Scan0 : scan edge (search edge 3) 4 [2381] 2381 root350Tail0 winner = winner :=
  (root350Step0).trans root350Scan1
set_option maxRecDepth 4096 in
theorem root350Branch : search edge 4 [2381] root350Candidates 2381 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2381] 2381 root350Tail0 winner = winner
  exact root350Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
