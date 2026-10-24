import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root568Candidates : List Nat := Data.row4153
set_option maxRecDepth 8192 in
theorem root568Candidates_checked : candidates edge 5 0 4153 tail569 winner.bound = root568Candidates := by
  apply candidates_eq_row edge 5 0 4153 tail569 root568Candidates winner.bound tail569_ordered
  · decide
  · decide
  · funext q
    rfl
def root568Tail28 : List Nat := []
def root568Tail27 : List Nat := 20809 :: root568Tail28
def root568Tail26 : List Nat := 19753 :: root568Tail27
def root568Tail25 : List Nat := 18481 :: root568Tail26
def root568Tail24 : List Nat := 18181 :: root568Tail25
def root568Tail23 : List Nat := 17971 :: root568Tail24
def root568Tail22 : List Nat := 15913 :: root568Tail23
def root568Tail21 : List Nat := 15289 :: root568Tail22
def root568Tail20 : List Nat := 14869 :: root568Tail21
def root568Tail19 : List Nat := 14071 :: root568Tail20
def root568Tail18 : List Nat := 12553 :: root568Tail19
def root568Tail17 : List Nat := 12409 :: root568Tail18
def root568Tail16 : List Nat := 12343 :: root568Tail17
def root568Tail15 : List Nat := 11593 :: root568Tail16
def root568Tail14 : List Nat := 11353 :: root568Tail15
def root568Tail13 : List Nat := 10837 :: root568Tail14
def root568Tail12 : List Nat := 9661 :: root568Tail13
def root568Tail11 : List Nat := 9283 :: root568Tail12
def root568Tail10 : List Nat := 8821 :: root568Tail11
def root568Tail9 : List Nat := 8311 :: root568Tail10
def root568Tail8 : List Nat := 7753 :: root568Tail9
def root568Tail7 : List Nat := 6997 :: root568Tail8
def root568Tail6 : List Nat := 5521 :: root568Tail7
def root568Tail5 : List Nat := 5503 :: root568Tail6
def root568Tail4 : List Nat := 5101 :: root568Tail5
def root568Tail3 : List Nat := 5077 :: root568Tail4
def root568Tail2 : List Nat := 5023 :: root568Tail3
def root568Tail1 : List Nat := 4789 :: root568Tail2
def root568Tail0 : List Nat := 4201 :: root568Tail1
def root568Candidates0 : List Nat := []
set_option maxRecDepth 8192 in
theorem root568Candidates0_checked : candidates edge 4 4153 4201 root568Tail1 winner.bound = root568Candidates0 := by
  unfold candidates
  have row : edge 4201 = (SortedNatTree.contains FastRows.row4201) := by
    change Data.row4201.contains = (SortedNatTree.contains FastRows.row4201)
    exact FastRows.row4201_checked.symm
  rw [row]
  decide
theorem root568Step0 : scan edge (search edge 3) 4 [4153] 4153 root568Tail0 winner =
    scan edge (search edge 3) 4 [4153] 4153 root568Tail1 winner := by
  rw [root568Tail0, scan, if_neg (by decide), root568Candidates0_checked]
  change scan edge (search edge 3) 4 [4153] 4153 root568Tail1
    (if 3 ≤ root568Candidates0.length then search edge 3 [4153, 4201] root568Candidates0 8354 winner else winner) = _
  rw [if_neg (by decide)]
def root568Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root568Candidates1_checked : candidates edge 4 4153 4789 root568Tail2 winner.bound = root568Candidates1 := by
  unfold candidates
  have row : edge 4789 = (SortedNatTree.contains FastRows.row4789) := by
    change Data.row4789.contains = (SortedNatTree.contains FastRows.row4789)
    exact FastRows.row4789_checked.symm
  rw [row]
  decide
theorem root568Step1 : scan edge (search edge 3) 4 [4153] 4153 root568Tail1 winner =
    scan edge (search edge 3) 4 [4153] 4153 root568Tail2 winner := by
  rw [root568Tail1, scan, if_neg (by decide), root568Candidates1_checked]
  change scan edge (search edge 3) 4 [4153] 4153 root568Tail2
    (if 3 ≤ root568Candidates1.length then search edge 3 [4153, 4789] root568Candidates1 8942 winner else winner) = _
  rw [if_neg (by decide)]
def root568Candidates2 : List Nat := [7753]
set_option maxRecDepth 8192 in
theorem root568Candidates2_checked : candidates edge 4 4153 5023 root568Tail3 winner.bound = root568Candidates2 := by
  unfold candidates
  have row : edge 5023 = (SortedNatTree.contains FastRows.row5023) := by
    change Data.row5023.contains = (SortedNatTree.contains FastRows.row5023)
    exact FastRows.row5023_checked.symm
  rw [row]
  decide
theorem root568Step2 : scan edge (search edge 3) 4 [4153] 4153 root568Tail2 winner =
    scan edge (search edge 3) 4 [4153] 4153 root568Tail3 winner := by
  rw [root568Tail2, scan, if_neg (by decide), root568Candidates2_checked]
  change scan edge (search edge 3) 4 [4153] 4153 root568Tail3
    (if 3 ≤ root568Candidates2.length then search edge 3 [4153, 5023] root568Candidates2 9176 winner else winner) = _
  rw [if_neg (by decide)]
def root568Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root568Candidates3_checked : candidates edge 4 4153 5077 root568Tail4 winner.bound = root568Candidates3 := by
  unfold candidates
  have row : edge 5077 = (SortedNatTree.contains FastRows.row5077) := by
    change Data.row5077.contains = (SortedNatTree.contains FastRows.row5077)
    exact FastRows.row5077_checked.symm
  rw [row]
  decide
theorem root568Step3 : scan edge (search edge 3) 4 [4153] 4153 root568Tail3 winner =
    scan edge (search edge 3) 4 [4153] 4153 root568Tail4 winner := by
  rw [root568Tail3, scan, if_neg (by decide), root568Candidates3_checked]
  change scan edge (search edge 3) 4 [4153] 4153 root568Tail4
    (if 3 ≤ root568Candidates3.length then search edge 3 [4153, 5077] root568Candidates3 9230 winner else winner) = _
  rw [if_neg (by decide)]
def root568Candidates4 : List Nat := [12553, 15913]
set_option maxRecDepth 8192 in
theorem root568Candidates4_checked : candidates edge 4 4153 5101 root568Tail5 winner.bound = root568Candidates4 := by
  unfold candidates
  have row : edge 5101 = (SortedNatTree.contains FastRows.row5101) := by
    change Data.row5101.contains = (SortedNatTree.contains FastRows.row5101)
    exact FastRows.row5101_checked.symm
  rw [row]
  decide
theorem root568Step4 : scan edge (search edge 3) 4 [4153] 4153 root568Tail4 winner =
    scan edge (search edge 3) 4 [4153] 4153 root568Tail5 winner := by
  rw [root568Tail4, scan, if_neg (by decide), root568Candidates4_checked]
  change scan edge (search edge 3) 4 [4153] 4153 root568Tail5
    (if 3 ≤ root568Candidates4.length then search edge 3 [4153, 5101] root568Candidates4 9254 winner else winner) = _
  rw [if_neg (by decide)]
theorem root568Scan5 : scan edge (search edge 3) 4 [4153] 4153 root568Tail5 winner = winner := by rw [root568Tail5, scan, if_pos (by decide)]
theorem root568Scan4 : scan edge (search edge 3) 4 [4153] 4153 root568Tail4 winner = winner :=
  (root568Step4).trans root568Scan5
theorem root568Scan3 : scan edge (search edge 3) 4 [4153] 4153 root568Tail3 winner = winner :=
  (root568Step3).trans root568Scan4
theorem root568Scan2 : scan edge (search edge 3) 4 [4153] 4153 root568Tail2 winner = winner :=
  (root568Step2).trans root568Scan3
theorem root568Scan1 : scan edge (search edge 3) 4 [4153] 4153 root568Tail1 winner = winner :=
  (root568Step1).trans root568Scan2
theorem root568Scan0 : scan edge (search edge 3) 4 [4153] 4153 root568Tail0 winner = winner :=
  (root568Step0).trans root568Scan1
set_option maxRecDepth 4096 in
theorem root568Branch : search edge 4 [4153] root568Candidates 4153 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4153] 4153 root568Tail0 winner = winner
  exact root568Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
