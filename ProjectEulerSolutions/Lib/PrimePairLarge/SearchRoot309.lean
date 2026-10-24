import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root309Candidates : List Nat := Data.row2069
set_option maxRecDepth 8192 in
theorem root309Candidates_checked : candidates edge 5 0 2069 tail310 winner.bound = root309Candidates := by
  apply candidates_eq_row edge 5 0 2069 tail310 root309Candidates winner.bound tail310_ordered
  · decide
  · decide
  · funext q
    rfl
def root309Tail46 : List Nat := []
def root309Tail45 : List Nat := 23669 :: root309Tail46
def root309Tail44 : List Nat := 22943 :: root309Tail45
def root309Tail43 : List Nat := 22739 :: root309Tail44
def root309Tail42 : List Nat := 22481 :: root309Tail43
def root309Tail41 : List Nat := 21269 :: root309Tail42
def root309Tail40 : List Nat := 21017 :: root309Tail41
def root309Tail39 : List Nat := 20807 :: root309Tail40
def root309Tail38 : List Nat := 20183 :: root309Tail39
def root309Tail37 : List Nat := 19463 :: root309Tail38
def root309Tail36 : List Nat := 18077 :: root309Tail37
def root309Tail35 : List Nat := 17573 :: root309Tail36
def root309Tail34 : List Nat := 16007 :: root309Tail35
def root309Tail33 : List Nat := 15887 :: root309Tail34
def root309Tail32 : List Nat := 15461 :: root309Tail33
def root309Tail31 : List Nat := 15017 :: root309Tail32
def root309Tail30 : List Nat := 14759 :: root309Tail31
def root309Tail29 : List Nat := 14723 :: root309Tail30
def root309Tail28 : List Nat := 14627 :: root309Tail29
def root309Tail27 : List Nat := 14423 :: root309Tail28
def root309Tail26 : List Nat := 14207 :: root309Tail27
def root309Tail25 : List Nat := 14033 :: root309Tail26
def root309Tail24 : List Nat := 13907 :: root309Tail25
def root309Tail23 : List Nat := 13799 :: root309Tail24
def root309Tail22 : List Nat := 12239 :: root309Tail23
def root309Tail21 : List Nat := 11093 :: root309Tail22
def root309Tail20 : List Nat := 11069 :: root309Tail21
def root309Tail19 : List Nat := 10433 :: root309Tail20
def root309Tail18 : List Nat := 9221 :: root309Tail19
def root309Tail17 : List Nat := 9059 :: root309Tail18
def root309Tail16 : List Nat := 8747 :: root309Tail17
def root309Tail15 : List Nat := 8369 :: root309Tail16
def root309Tail14 : List Nat := 8219 :: root309Tail15
def root309Tail13 : List Nat := 7877 :: root309Tail14
def root309Tail12 : List Nat := 7757 :: root309Tail13
def root309Tail11 : List Nat := 7283 :: root309Tail12
def root309Tail10 : List Nat := 7253 :: root309Tail11
def root309Tail9 : List Nat := 7121 :: root309Tail10
def root309Tail8 : List Nat := 6983 :: root309Tail9
def root309Tail7 : List Nat := 6959 :: root309Tail8
def root309Tail6 : List Nat := 6827 :: root309Tail7
def root309Tail5 : List Nat := 5021 :: root309Tail6
def root309Tail4 : List Nat := 4691 :: root309Tail5
def root309Tail3 : List Nat := 4637 :: root309Tail4
def root309Tail2 : List Nat := 4397 :: root309Tail3
def root309Tail1 : List Nat := 2339 :: root309Tail2
def root309Tail0 : List Nat := 2297 :: root309Tail1
def root309Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root309Candidates0_checked : candidates edge 4 2069 2297 root309Tail1 winner.bound = root309Candidates0 := by
  unfold candidates
  have row : edge 2297 = (SortedNatTree.contains FastRows.row2297) := by
    change Data.row2297.contains = (SortedNatTree.contains FastRows.row2297)
    exact FastRows.row2297_checked.symm
  rw [row]
  decide
theorem root309Step0 : scan edge (search edge 3) 4 [2069] 2069 root309Tail0 winner =
    scan edge (search edge 3) 4 [2069] 2069 root309Tail1 winner := by
  rw [root309Tail0, scan, if_neg (by decide), root309Candidates0_checked]
  change scan edge (search edge 3) 4 [2069] 2069 root309Tail1
    (if 3 ≤ root309Candidates0.length then search edge 3 [2069, 2297] root309Candidates0 4366 winner else winner) = _
  rw [if_neg (by decide)]
def root309Candidates1 : List Nat := [8369, 20807]
set_option maxRecDepth 8192 in
theorem root309Candidates1_checked : candidates edge 4 2069 2339 root309Tail2 winner.bound = root309Candidates1 := by
  unfold candidates
  have row : edge 2339 = (SortedNatTree.contains FastRows.row2339) := by
    change Data.row2339.contains = (SortedNatTree.contains FastRows.row2339)
    exact FastRows.row2339_checked.symm
  rw [row]
  decide
theorem root309Step1 : scan edge (search edge 3) 4 [2069] 2069 root309Tail1 winner =
    scan edge (search edge 3) 4 [2069] 2069 root309Tail2 winner := by
  rw [root309Tail1, scan, if_neg (by decide), root309Candidates1_checked]
  change scan edge (search edge 3) 4 [2069] 2069 root309Tail2
    (if 3 ≤ root309Candidates1.length then search edge 3 [2069, 2339] root309Candidates1 4408 winner else winner) = _
  rw [if_neg (by decide)]
def root309Candidates2 : List Nat := [4691]
set_option maxRecDepth 8192 in
theorem root309Candidates2_checked : candidates edge 4 2069 4397 root309Tail3 winner.bound = root309Candidates2 := by
  unfold candidates
  have row : edge 4397 = (SortedNatTree.contains FastRows.row4397) := by
    change Data.row4397.contains = (SortedNatTree.contains FastRows.row4397)
    exact FastRows.row4397_checked.symm
  rw [row]
  decide
theorem root309Step2 : scan edge (search edge 3) 4 [2069] 2069 root309Tail2 winner =
    scan edge (search edge 3) 4 [2069] 2069 root309Tail3 winner := by
  rw [root309Tail2, scan, if_neg (by decide), root309Candidates2_checked]
  change scan edge (search edge 3) 4 [2069] 2069 root309Tail3
    (if 3 ≤ root309Candidates2.length then search edge 3 [2069, 4397] root309Candidates2 6466 winner else winner) = _
  rw [if_neg (by decide)]
def root309Candidates3 : List Nat := [11069, 14759]
set_option maxRecDepth 8192 in
theorem root309Candidates3_checked : candidates edge 4 2069 4637 root309Tail4 winner.bound = root309Candidates3 := by
  unfold candidates
  have row : edge 4637 = (SortedNatTree.contains FastRows.row4637) := by
    change Data.row4637.contains = (SortedNatTree.contains FastRows.row4637)
    exact FastRows.row4637_checked.symm
  rw [row]
  decide
theorem root309Step3 : scan edge (search edge 3) 4 [2069] 2069 root309Tail3 winner =
    scan edge (search edge 3) 4 [2069] 2069 root309Tail4 winner := by
  rw [root309Tail3, scan, if_neg (by decide), root309Candidates3_checked]
  change scan edge (search edge 3) 4 [2069] 2069 root309Tail4
    (if 3 ≤ root309Candidates3.length then search edge 3 [2069, 4637] root309Candidates3 6706 winner else winner) = _
  rw [if_neg (by decide)]
def root309Candidates4 : List Nat := [12239, 15887]
set_option maxRecDepth 8192 in
theorem root309Candidates4_checked : candidates edge 4 2069 4691 root309Tail5 winner.bound = root309Candidates4 := by
  unfold candidates
  have row : edge 4691 = (SortedNatTree.contains FastRows.row4691) := by
    change Data.row4691.contains = (SortedNatTree.contains FastRows.row4691)
    exact FastRows.row4691_checked.symm
  rw [row]
  decide
theorem root309Step4 : scan edge (search edge 3) 4 [2069] 2069 root309Tail4 winner =
    scan edge (search edge 3) 4 [2069] 2069 root309Tail5 winner := by
  rw [root309Tail4, scan, if_neg (by decide), root309Candidates4_checked]
  change scan edge (search edge 3) 4 [2069] 2069 root309Tail5
    (if 3 ≤ root309Candidates4.length then search edge 3 [2069, 4691] root309Candidates4 6760 winner else winner) = _
  rw [if_neg (by decide)]
def root309Candidates5 : List Nat := [8219]
set_option maxRecDepth 8192 in
theorem root309Candidates5_checked : candidates edge 4 2069 5021 root309Tail6 winner.bound = root309Candidates5 := by
  unfold candidates
  have row : edge 5021 = (SortedNatTree.contains FastRows.row5021) := by
    change Data.row5021.contains = (SortedNatTree.contains FastRows.row5021)
    exact FastRows.row5021_checked.symm
  rw [row]
  decide
theorem root309Step5 : scan edge (search edge 3) 4 [2069] 2069 root309Tail5 winner =
    scan edge (search edge 3) 4 [2069] 2069 root309Tail6 winner := by
  rw [root309Tail5, scan, if_neg (by decide), root309Candidates5_checked]
  change scan edge (search edge 3) 4 [2069] 2069 root309Tail6
    (if 3 ≤ root309Candidates5.length then search edge 3 [2069, 5021] root309Candidates5 7090 winner else winner) = _
  rw [if_neg (by decide)]
theorem root309Scan6 : scan edge (search edge 3) 4 [2069] 2069 root309Tail6 winner = winner := by rw [root309Tail6, scan, if_pos (by decide)]
theorem root309Scan5 : scan edge (search edge 3) 4 [2069] 2069 root309Tail5 winner = winner :=
  (root309Step5).trans root309Scan6
theorem root309Scan4 : scan edge (search edge 3) 4 [2069] 2069 root309Tail4 winner = winner :=
  (root309Step4).trans root309Scan5
theorem root309Scan3 : scan edge (search edge 3) 4 [2069] 2069 root309Tail3 winner = winner :=
  (root309Step3).trans root309Scan4
theorem root309Scan2 : scan edge (search edge 3) 4 [2069] 2069 root309Tail2 winner = winner :=
  (root309Step2).trans root309Scan3
theorem root309Scan1 : scan edge (search edge 3) 4 [2069] 2069 root309Tail1 winner = winner :=
  (root309Step1).trans root309Scan2
theorem root309Scan0 : scan edge (search edge 3) 4 [2069] 2069 root309Tail0 winner = winner :=
  (root309Step0).trans root309Scan1
set_option maxRecDepth 4096 in
theorem root309Branch : search edge 4 [2069] root309Candidates 2069 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2069] 2069 root309Tail0 winner = winner
  exact root309Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
