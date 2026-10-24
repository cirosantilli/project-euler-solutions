import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root415Candidates : List Nat := Data.row2887
set_option maxRecDepth 8192 in
theorem root415Candidates_checked : candidates edge 5 0 2887 tail416 winner.bound = root415Candidates := by
  apply candidates_eq_row edge 5 0 2887 tail416 root415Candidates winner.bound tail416_ordered
  · decide
  · decide
  · funext q
    rfl
def root415Tail42 : List Nat := []
def root415Tail41 : List Nat := 22669 :: root415Tail42
def root415Tail40 : List Nat := 22573 :: root415Tail41
def root415Tail39 : List Nat := 22129 :: root415Tail40
def root415Tail38 : List Nat := 21943 :: root415Tail39
def root415Tail37 : List Nat := 21859 :: root415Tail38
def root415Tail36 : List Nat := 20359 :: root415Tail37
def root415Tail35 : List Nat := 20101 :: root415Tail36
def root415Tail34 : List Nat := 20071 :: root415Tail35
def root415Tail33 : List Nat := 19813 :: root415Tail34
def root415Tail32 : List Nat := 19477 :: root415Tail33
def root415Tail31 : List Nat := 18181 :: root415Tail32
def root415Tail30 : List Nat := 17929 :: root415Tail31
def root415Tail29 : List Nat := 17713 :: root415Tail30
def root415Tail28 : List Nat := 17491 :: root415Tail29
def root415Tail27 : List Nat := 17041 :: root415Tail28
def root415Tail26 : List Nat := 16651 :: root415Tail27
def root415Tail25 : List Nat := 16369 :: root415Tail26
def root415Tail24 : List Nat := 15277 :: root415Tail25
def root415Tail23 : List Nat := 14851 :: root415Tail24
def root415Tail22 : List Nat := 14821 :: root415Tail23
def root415Tail21 : List Nat := 14593 :: root415Tail22
def root415Tail20 : List Nat := 14431 :: root415Tail21
def root415Tail19 : List Nat := 14011 :: root415Tail20
def root415Tail18 : List Nat := 13219 :: root415Tail19
def root415Tail17 : List Nat := 13171 :: root415Tail18
def root415Tail16 : List Nat := 12973 :: root415Tail17
def root415Tail15 : List Nat := 12967 :: root415Tail16
def root415Tail14 : List Nat := 12829 :: root415Tail15
def root415Tail13 : List Nat := 12763 :: root415Tail14
def root415Tail12 : List Nat := 12511 :: root415Tail13
def root415Tail11 : List Nat := 9511 :: root415Tail12
def root415Tail10 : List Nat := 9439 :: root415Tail11
def root415Tail9 : List Nat := 9001 :: root415Tail10
def root415Tail8 : List Nat := 7573 :: root415Tail9
def root415Tail7 : List Nat := 6733 :: root415Tail8
def root415Tail6 : List Nat := 6709 :: root415Tail7
def root415Tail5 : List Nat := 5653 :: root415Tail6
def root415Tail4 : List Nat := 4987 :: root415Tail5
def root415Tail3 : List Nat := 4597 :: root415Tail4
def root415Tail2 : List Nat := 4273 :: root415Tail3
def root415Tail1 : List Nat := 4231 :: root415Tail2
def root415Tail0 : List Nat := 2917 :: root415Tail1
def root415Candidates0 : List Nat := [12829]
set_option maxRecDepth 8192 in
theorem root415Candidates0_checked : candidates edge 4 2887 2917 root415Tail1 winner.bound = root415Candidates0 := by
  unfold candidates
  have row : edge 2917 = (SortedNatTree.contains FastRows.row2917) := by
    change Data.row2917.contains = (SortedNatTree.contains FastRows.row2917)
    exact FastRows.row2917_checked.symm
  rw [row]
  decide
theorem root415Step0 : scan edge (search edge 3) 4 [2887] 2887 root415Tail0 winner =
    scan edge (search edge 3) 4 [2887] 2887 root415Tail1 winner := by
  rw [root415Tail0, scan, if_neg (by decide), root415Candidates0_checked]
  change scan edge (search edge 3) 4 [2887] 2887 root415Tail1
    (if 3 ≤ root415Candidates0.length then search edge 3 [2887, 2917] root415Candidates0 5804 winner else winner) = _
  rw [if_neg (by decide)]
def root415Candidates1 : List Nat := [12763, 12967, 14011]
set_option maxRecDepth 8192 in
theorem root415Candidates1_checked : candidates edge 4 2887 4231 root415Tail2 winner.bound = root415Candidates1 := by
  unfold candidates
  have row : edge 4231 = (SortedNatTree.contains FastRows.row4231) := by
    change Data.row4231.contains = (SortedNatTree.contains FastRows.row4231)
    exact FastRows.row4231_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root415Child1 : search edge 3 [2887, 4231] root415Candidates1 7118 winner = winner := by decide
theorem root415Step1 : scan edge (search edge 3) 4 [2887] 2887 root415Tail1 winner =
    scan edge (search edge 3) 4 [2887] 2887 root415Tail2 winner := by
  rw [root415Tail1, scan, if_neg (by decide), root415Candidates1_checked]
  change scan edge (search edge 3) 4 [2887] 2887 root415Tail2
    (if 3 ≤ root415Candidates1.length then search edge 3 [2887, 4231] root415Candidates1 7118 winner else winner) = _
  rw [if_pos (by decide), root415Child1]
def root415Candidates2 : List Nat := [7573, 13171, 15277]
set_option maxRecDepth 8192 in
theorem root415Candidates2_checked : candidates edge 4 2887 4273 root415Tail3 winner.bound = root415Candidates2 := by
  unfold candidates
  have row : edge 4273 = (SortedNatTree.contains FastRows.row4273) := by
    change Data.row4273.contains = (SortedNatTree.contains FastRows.row4273)
    exact FastRows.row4273_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root415Child2 : search edge 3 [2887, 4273] root415Candidates2 7160 winner = winner := by decide
theorem root415Step2 : scan edge (search edge 3) 4 [2887] 2887 root415Tail2 winner =
    scan edge (search edge 3) 4 [2887] 2887 root415Tail3 winner := by
  rw [root415Tail2, scan, if_neg (by decide), root415Candidates2_checked]
  change scan edge (search edge 3) 4 [2887] 2887 root415Tail3
    (if 3 ≤ root415Candidates2.length then search edge 3 [2887, 4273] root415Candidates2 7160 winner else winner) = _
  rw [if_pos (by decide), root415Child2]
def root415Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root415Candidates3_checked : candidates edge 4 2887 4597 root415Tail4 winner.bound = root415Candidates3 := by
  unfold candidates
  have row : edge 4597 = (SortedNatTree.contains FastRows.row4597) := by
    change Data.row4597.contains = (SortedNatTree.contains FastRows.row4597)
    exact FastRows.row4597_checked.symm
  rw [row]
  decide
theorem root415Step3 : scan edge (search edge 3) 4 [2887] 2887 root415Tail3 winner =
    scan edge (search edge 3) 4 [2887] 2887 root415Tail4 winner := by
  rw [root415Tail3, scan, if_neg (by decide), root415Candidates3_checked]
  change scan edge (search edge 3) 4 [2887] 2887 root415Tail4
    (if 3 ≤ root415Candidates3.length then search edge 3 [2887, 4597] root415Candidates3 7484 winner else winner) = _
  rw [if_neg (by decide)]
def root415Candidates4 : List Nat := [12763, 12829]
set_option maxRecDepth 8192 in
theorem root415Candidates4_checked : candidates edge 4 2887 4987 root415Tail5 winner.bound = root415Candidates4 := by
  unfold candidates
  have row : edge 4987 = (SortedNatTree.contains FastRows.row4987) := by
    change Data.row4987.contains = (SortedNatTree.contains FastRows.row4987)
    exact FastRows.row4987_checked.symm
  rw [row]
  decide
theorem root415Step4 : scan edge (search edge 3) 4 [2887] 2887 root415Tail4 winner =
    scan edge (search edge 3) 4 [2887] 2887 root415Tail5 winner := by
  rw [root415Tail4, scan, if_neg (by decide), root415Candidates4_checked]
  change scan edge (search edge 3) 4 [2887] 2887 root415Tail5
    (if 3 ≤ root415Candidates4.length then search edge 3 [2887, 4987] root415Candidates4 7874 winner else winner) = _
  rw [if_neg (by decide)]
def root415Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root415Candidates5_checked : candidates edge 4 2887 5653 root415Tail6 winner.bound = root415Candidates5 := by
  unfold candidates
  have row : edge 5653 = (SortedNatTree.contains FastRows.row5653) := by
    change Data.row5653.contains = (SortedNatTree.contains FastRows.row5653)
    exact FastRows.row5653_checked.symm
  rw [row]
  decide
theorem root415Step5 : scan edge (search edge 3) 4 [2887] 2887 root415Tail5 winner =
    scan edge (search edge 3) 4 [2887] 2887 root415Tail6 winner := by
  rw [root415Tail5, scan, if_neg (by decide), root415Candidates5_checked]
  change scan edge (search edge 3) 4 [2887] 2887 root415Tail6
    (if 3 ≤ root415Candidates5.length then search edge 3 [2887, 5653] root415Candidates5 8540 winner else winner) = _
  rw [if_neg (by decide)]
theorem root415Scan6 : scan edge (search edge 3) 4 [2887] 2887 root415Tail6 winner = winner := by rw [root415Tail6, scan, if_pos (by decide)]
theorem root415Scan5 : scan edge (search edge 3) 4 [2887] 2887 root415Tail5 winner = winner :=
  (root415Step5).trans root415Scan6
theorem root415Scan4 : scan edge (search edge 3) 4 [2887] 2887 root415Tail4 winner = winner :=
  (root415Step4).trans root415Scan5
theorem root415Scan3 : scan edge (search edge 3) 4 [2887] 2887 root415Tail3 winner = winner :=
  (root415Step3).trans root415Scan4
theorem root415Scan2 : scan edge (search edge 3) 4 [2887] 2887 root415Tail2 winner = winner :=
  (root415Step2).trans root415Scan3
theorem root415Scan1 : scan edge (search edge 3) 4 [2887] 2887 root415Tail1 winner = winner :=
  (root415Step1).trans root415Scan2
theorem root415Scan0 : scan edge (search edge 3) 4 [2887] 2887 root415Tail0 winner = winner :=
  (root415Step0).trans root415Scan1
set_option maxRecDepth 4096 in
theorem root415Branch : search edge 4 [2887] root415Candidates 2887 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2887] 2887 root415Tail0 winner = winner
  exact root415Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
