import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root416Candidates : List Nat := Data.row2897
set_option maxRecDepth 8192 in
theorem root416Candidates_checked : candidates edge 5 0 2897 tail417 winner.bound = root416Candidates := by
  apply candidates_eq_row edge 5 0 2897 tail417 root416Candidates winner.bound tail417_ordered
  · decide
  · decide
  · funext q
    rfl
def root416Tail36 : List Nat := []
def root416Tail35 : List Nat := 22739 :: root416Tail36
def root416Tail34 : List Nat := 22343 :: root416Tail35
def root416Tail33 : List Nat := 22193 :: root416Tail34
def root416Tail32 : List Nat := 21521 :: root416Tail33
def root416Tail31 : List Nat := 20789 :: root416Tail32
def root416Tail30 : List Nat := 19403 :: root416Tail31
def root416Tail29 : List Nat := 18911 :: root416Tail30
def root416Tail28 : List Nat := 18353 :: root416Tail29
def root416Tail27 : List Nat := 18059 :: root416Tail28
def root416Tail26 : List Nat := 15299 :: root416Tail27
def root416Tail25 : List Nat := 15053 :: root416Tail26
def root416Tail24 : List Nat := 11789 :: root416Tail25
def root416Tail23 : List Nat := 11057 :: root416Tail24
def root416Tail22 : List Nat := 10937 :: root416Tail23
def root416Tail21 : List Nat := 10163 :: root416Tail22
def root416Tail20 : List Nat := 10151 :: root416Tail21
def root416Tail19 : List Nat := 10079 :: root416Tail20
def root416Tail18 : List Nat := 9551 :: root416Tail19
def root416Tail17 : List Nat := 9539 :: root416Tail18
def root416Tail16 : List Nat := 9257 :: root416Tail17
def root416Tail15 : List Nat := 9227 :: root416Tail16
def root416Tail14 : List Nat := 8867 :: root416Tail15
def root416Tail13 : List Nat := 7547 :: root416Tail14
def root416Tail12 : List Nat := 7307 :: root416Tail13
def root416Tail11 : List Nat := 7211 :: root416Tail12
def root416Tail10 : List Nat := 6473 :: root416Tail11
def root416Tail9 : List Nat := 6329 :: root416Tail10
def root416Tail8 : List Nat := 5867 :: root416Tail9
def root416Tail7 : List Nat := 5501 :: root416Tail8
def root416Tail6 : List Nat := 4877 :: root416Tail7
def root416Tail5 : List Nat := 4721 :: root416Tail6
def root416Tail4 : List Nat := 4493 :: root416Tail5
def root416Tail3 : List Nat := 4289 :: root416Tail4
def root416Tail2 : List Nat := 3947 :: root416Tail3
def root416Tail1 : List Nat := 3209 :: root416Tail2
def root416Tail0 : List Nat := 3023 :: root416Tail1
def root416Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root416Candidates0_checked : candidates edge 4 2897 3023 root416Tail1 winner.bound = root416Candidates0 := by
  unfold candidates
  have row : edge 3023 = (SortedNatTree.contains FastRows.row3023) := by
    change Data.row3023.contains = (SortedNatTree.contains FastRows.row3023)
    exact FastRows.row3023_checked.symm
  rw [row]
  decide
theorem root416Step0 : scan edge (search edge 3) 4 [2897] 2897 root416Tail0 winner =
    scan edge (search edge 3) 4 [2897] 2897 root416Tail1 winner := by
  rw [root416Tail0, scan, if_neg (by decide), root416Candidates0_checked]
  change scan edge (search edge 3) 4 [2897] 2897 root416Tail1
    (if 3 ≤ root416Candidates0.length then search edge 3 [2897, 3023] root416Candidates0 5920 winner else winner) = _
  rw [if_neg (by decide)]
def root416Candidates1 : List Nat := [4721, 9257]
set_option maxRecDepth 8192 in
theorem root416Candidates1_checked : candidates edge 4 2897 3209 root416Tail2 winner.bound = root416Candidates1 := by
  unfold candidates
  have row : edge 3209 = (SortedNatTree.contains FastRows.row3209) := by
    change Data.row3209.contains = (SortedNatTree.contains FastRows.row3209)
    exact FastRows.row3209_checked.symm
  rw [row]
  decide
theorem root416Step1 : scan edge (search edge 3) 4 [2897] 2897 root416Tail1 winner =
    scan edge (search edge 3) 4 [2897] 2897 root416Tail2 winner := by
  rw [root416Tail1, scan, if_neg (by decide), root416Candidates1_checked]
  change scan edge (search edge 3) 4 [2897] 2897 root416Tail2
    (if 3 ≤ root416Candidates1.length then search edge 3 [2897, 3209] root416Candidates1 6106 winner else winner) = _
  rw [if_neg (by decide)]
def root416Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root416Candidates2_checked : candidates edge 4 2897 3947 root416Tail3 winner.bound = root416Candidates2 := by
  unfold candidates
  have row : edge 3947 = (SortedNatTree.contains FastRows.row3947) := by
    change Data.row3947.contains = (SortedNatTree.contains FastRows.row3947)
    exact FastRows.row3947_checked.symm
  rw [row]
  decide
theorem root416Step2 : scan edge (search edge 3) 4 [2897] 2897 root416Tail2 winner =
    scan edge (search edge 3) 4 [2897] 2897 root416Tail3 winner := by
  rw [root416Tail2, scan, if_neg (by decide), root416Candidates2_checked]
  change scan edge (search edge 3) 4 [2897] 2897 root416Tail3
    (if 3 ≤ root416Candidates2.length then search edge 3 [2897, 3947] root416Candidates2 6844 winner else winner) = _
  rw [if_neg (by decide)]
def root416Candidates3 : List Nat := [6473]
set_option maxRecDepth 8192 in
theorem root416Candidates3_checked : candidates edge 4 2897 4289 root416Tail4 winner.bound = root416Candidates3 := by
  unfold candidates
  have row : edge 4289 = (SortedNatTree.contains FastRows.row4289) := by
    change Data.row4289.contains = (SortedNatTree.contains FastRows.row4289)
    exact FastRows.row4289_checked.symm
  rw [row]
  decide
theorem root416Step3 : scan edge (search edge 3) 4 [2897] 2897 root416Tail3 winner =
    scan edge (search edge 3) 4 [2897] 2897 root416Tail4 winner := by
  rw [root416Tail3, scan, if_neg (by decide), root416Candidates3_checked]
  change scan edge (search edge 3) 4 [2897] 2897 root416Tail4
    (if 3 ≤ root416Candidates3.length then search edge 3 [2897, 4289] root416Candidates3 7186 winner else winner) = _
  rw [if_neg (by decide)]
def root416Candidates4 : List Nat := [10151]
set_option maxRecDepth 8192 in
theorem root416Candidates4_checked : candidates edge 4 2897 4493 root416Tail5 winner.bound = root416Candidates4 := by
  unfold candidates
  have row : edge 4493 = (SortedNatTree.contains FastRows.row4493) := by
    change Data.row4493.contains = (SortedNatTree.contains FastRows.row4493)
    exact FastRows.row4493_checked.symm
  rw [row]
  decide
theorem root416Step4 : scan edge (search edge 3) 4 [2897] 2897 root416Tail4 winner =
    scan edge (search edge 3) 4 [2897] 2897 root416Tail5 winner := by
  rw [root416Tail4, scan, if_neg (by decide), root416Candidates4_checked]
  change scan edge (search edge 3) 4 [2897] 2897 root416Tail5
    (if 3 ≤ root416Candidates4.length then search edge 3 [2897, 4493] root416Candidates4 7390 winner else winner) = _
  rw [if_neg (by decide)]
def root416Candidates5 : List Nat := [4877, 5501, 9257]
set_option maxRecDepth 8192 in
theorem root416Candidates5_checked : candidates edge 4 2897 4721 root416Tail6 winner.bound = root416Candidates5 := by
  unfold candidates
  have row : edge 4721 = (SortedNatTree.contains FastRows.row4721) := by
    change Data.row4721.contains = (SortedNatTree.contains FastRows.row4721)
    exact FastRows.row4721_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root416Child5 : search edge 3 [2897, 4721] root416Candidates5 7618 winner = winner := by decide
theorem root416Step5 : scan edge (search edge 3) 4 [2897] 2897 root416Tail5 winner =
    scan edge (search edge 3) 4 [2897] 2897 root416Tail6 winner := by
  rw [root416Tail5, scan, if_neg (by decide), root416Candidates5_checked]
  change scan edge (search edge 3) 4 [2897] 2897 root416Tail6
    (if 3 ≤ root416Candidates5.length then search edge 3 [2897, 4721] root416Candidates5 7618 winner else winner) = _
  rw [if_pos (by decide), root416Child5]
def root416Candidates6 : List Nat := [5501, 7307]
set_option maxRecDepth 8192 in
theorem root416Candidates6_checked : candidates edge 4 2897 4877 root416Tail7 winner.bound = root416Candidates6 := by
  unfold candidates
  have row : edge 4877 = (SortedNatTree.contains FastRows.row4877) := by
    change Data.row4877.contains = (SortedNatTree.contains FastRows.row4877)
    exact FastRows.row4877_checked.symm
  rw [row]
  decide
theorem root416Step6 : scan edge (search edge 3) 4 [2897] 2897 root416Tail6 winner =
    scan edge (search edge 3) 4 [2897] 2897 root416Tail7 winner := by
  rw [root416Tail6, scan, if_neg (by decide), root416Candidates6_checked]
  change scan edge (search edge 3) 4 [2897] 2897 root416Tail7
    (if 3 ≤ root416Candidates6.length then search edge 3 [2897, 4877] root416Candidates6 7774 winner else winner) = _
  rw [if_neg (by decide)]
def root416Candidates7 : List Nat := [5867, 7211]
set_option maxRecDepth 8192 in
theorem root416Candidates7_checked : candidates edge 4 2897 5501 root416Tail8 winner.bound = root416Candidates7 := by
  unfold candidates
  have row : edge 5501 = (SortedNatTree.contains FastRows.row5501) := by
    change Data.row5501.contains = (SortedNatTree.contains FastRows.row5501)
    exact FastRows.row5501_checked.symm
  rw [row]
  decide
theorem root416Step7 : scan edge (search edge 3) 4 [2897] 2897 root416Tail7 winner =
    scan edge (search edge 3) 4 [2897] 2897 root416Tail8 winner := by
  rw [root416Tail7, scan, if_neg (by decide), root416Candidates7_checked]
  change scan edge (search edge 3) 4 [2897] 2897 root416Tail8
    (if 3 ≤ root416Candidates7.length then search edge 3 [2897, 5501] root416Candidates7 8398 winner else winner) = _
  rw [if_neg (by decide)]
theorem root416Scan8 : scan edge (search edge 3) 4 [2897] 2897 root416Tail8 winner = winner := by rw [root416Tail8, scan, if_pos (by decide)]
theorem root416Scan7 : scan edge (search edge 3) 4 [2897] 2897 root416Tail7 winner = winner :=
  (root416Step7).trans root416Scan8
theorem root416Scan6 : scan edge (search edge 3) 4 [2897] 2897 root416Tail6 winner = winner :=
  (root416Step6).trans root416Scan7
theorem root416Scan5 : scan edge (search edge 3) 4 [2897] 2897 root416Tail5 winner = winner :=
  (root416Step5).trans root416Scan6
theorem root416Scan4 : scan edge (search edge 3) 4 [2897] 2897 root416Tail4 winner = winner :=
  (root416Step4).trans root416Scan5
theorem root416Scan3 : scan edge (search edge 3) 4 [2897] 2897 root416Tail3 winner = winner :=
  (root416Step3).trans root416Scan4
theorem root416Scan2 : scan edge (search edge 3) 4 [2897] 2897 root416Tail2 winner = winner :=
  (root416Step2).trans root416Scan3
theorem root416Scan1 : scan edge (search edge 3) 4 [2897] 2897 root416Tail1 winner = winner :=
  (root416Step1).trans root416Scan2
theorem root416Scan0 : scan edge (search edge 3) 4 [2897] 2897 root416Tail0 winner = winner :=
  (root416Step0).trans root416Scan1
set_option maxRecDepth 4096 in
theorem root416Branch : search edge 4 [2897] root416Candidates 2897 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2897] 2897 root416Tail0 winner = winner
  exact root416Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
