import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root460Candidates : List Nat := Data.row3299
set_option maxRecDepth 8192 in
theorem root460Candidates_checked : candidates edge 5 0 3299 tail461 winner.bound = root460Candidates := by
  apply candidates_eq_row edge 5 0 3299 tail461 root460Candidates winner.bound tail461_ordered
  · decide
  · decide
  · funext q
    rfl
def root460Tail34 : List Nat := []
def root460Tail33 : List Nat := 22391 :: root460Tail34
def root460Tail32 : List Nat := 20117 :: root460Tail33
def root460Tail31 : List Nat := 19973 :: root460Tail32
def root460Tail30 : List Nat := 19727 :: root460Tail31
def root460Tail29 : List Nat := 19013 :: root460Tail30
def root460Tail28 : List Nat := 17807 :: root460Tail29
def root460Tail27 : List Nat := 17417 :: root460Tail28
def root460Tail26 : List Nat := 16073 :: root460Tail27
def root460Tail25 : List Nat := 15581 :: root460Tail26
def root460Tail24 : List Nat := 15359 :: root460Tail25
def root460Tail23 : List Nat := 15227 :: root460Tail24
def root460Tail22 : List Nat := 15161 :: root460Tail23
def root460Tail21 : List Nat := 15101 :: root460Tail22
def root460Tail20 : List Nat := 13187 :: root460Tail21
def root460Tail19 : List Nat := 13109 :: root460Tail20
def root460Tail18 : List Nat := 12269 :: root460Tail19
def root460Tail17 : List Nat := 12119 :: root460Tail18
def root460Tail16 : List Nat := 12041 :: root460Tail17
def root460Tail15 : List Nat := 11927 :: root460Tail16
def root460Tail14 : List Nat := 11213 :: root460Tail15
def root460Tail13 : List Nat := 10247 :: root460Tail14
def root460Tail12 : List Nat := 9203 :: root460Tail13
def root460Tail11 : List Nat := 8447 :: root460Tail12
def root460Tail10 : List Nat := 8429 :: root460Tail11
def root460Tail9 : List Nat := 7499 :: root460Tail10
def root460Tail8 : List Nat := 7043 :: root460Tail9
def root460Tail7 : List Nat := 6959 :: root460Tail8
def root460Tail6 : List Nat := 6869 :: root460Tail7
def root460Tail5 : List Nat := 6203 :: root460Tail6
def root460Tail4 : List Nat := 6011 :: root460Tail5
def root460Tail3 : List Nat := 4817 :: root460Tail4
def root460Tail2 : List Nat := 4523 :: root460Tail3
def root460Tail1 : List Nat := 4019 :: root460Tail2
def root460Tail0 : List Nat := 3767 :: root460Tail1
def root460Candidates0 : List Nat := [4523, 12041, 17417]
set_option maxRecDepth 8192 in
theorem root460Candidates0_checked : candidates edge 4 3299 3767 root460Tail1 winner.bound = root460Candidates0 := by
  unfold candidates
  have row : edge 3767 = (SortedNatTree.contains FastRows.row3767) := by
    change Data.row3767.contains = (SortedNatTree.contains FastRows.row3767)
    exact FastRows.row3767_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root460Child0 : search edge 3 [3299, 3767] root460Candidates0 7066 winner = winner := by decide
theorem root460Step0 : scan edge (search edge 3) 4 [3299] 3299 root460Tail0 winner =
    scan edge (search edge 3) 4 [3299] 3299 root460Tail1 winner := by
  rw [root460Tail0, scan, if_neg (by decide), root460Candidates0_checked]
  change scan edge (search edge 3) 4 [3299] 3299 root460Tail1
    (if 3 ≤ root460Candidates0.length then search edge 3 [3299, 3767] root460Candidates0 7066 winner else winner) = _
  rw [if_pos (by decide), root460Child0]
def root460Candidates1 : List Nat := [15161]
set_option maxRecDepth 8192 in
theorem root460Candidates1_checked : candidates edge 4 3299 4019 root460Tail2 winner.bound = root460Candidates1 := by
  unfold candidates
  have row : edge 4019 = (SortedNatTree.contains FastRows.row4019) := by
    change Data.row4019.contains = (SortedNatTree.contains FastRows.row4019)
    exact FastRows.row4019_checked.symm
  rw [row]
  decide
theorem root460Step1 : scan edge (search edge 3) 4 [3299] 3299 root460Tail1 winner =
    scan edge (search edge 3) 4 [3299] 3299 root460Tail2 winner := by
  rw [root460Tail1, scan, if_neg (by decide), root460Candidates1_checked]
  change scan edge (search edge 3) 4 [3299] 3299 root460Tail2
    (if 3 ≤ root460Candidates1.length then search edge 3 [3299, 4019] root460Candidates1 7318 winner else winner) = _
  rw [if_neg (by decide)]
def root460Candidates2 : List Nat := [4817]
set_option maxRecDepth 8192 in
theorem root460Candidates2_checked : candidates edge 4 3299 4523 root460Tail3 winner.bound = root460Candidates2 := by
  unfold candidates
  have row : edge 4523 = (SortedNatTree.contains FastRows.row4523) := by
    change Data.row4523.contains = (SortedNatTree.contains FastRows.row4523)
    exact FastRows.row4523_checked.symm
  rw [row]
  decide
theorem root460Step2 : scan edge (search edge 3) 4 [3299] 3299 root460Tail2 winner =
    scan edge (search edge 3) 4 [3299] 3299 root460Tail3 winner := by
  rw [root460Tail2, scan, if_neg (by decide), root460Candidates2_checked]
  change scan edge (search edge 3) 4 [3299] 3299 root460Tail3
    (if 3 ≤ root460Candidates2.length then search edge 3 [3299, 4523] root460Candidates2 7822 winner else winner) = _
  rw [if_neg (by decide)]
def root460Candidates3 : List Nat := [17417]
set_option maxRecDepth 8192 in
theorem root460Candidates3_checked : candidates edge 4 3299 4817 root460Tail4 winner.bound = root460Candidates3 := by
  unfold candidates
  have row : edge 4817 = (SortedNatTree.contains FastRows.row4817) := by
    change Data.row4817.contains = (SortedNatTree.contains FastRows.row4817)
    exact FastRows.row4817_checked.symm
  rw [row]
  decide
theorem root460Step3 : scan edge (search edge 3) 4 [3299] 3299 root460Tail3 winner =
    scan edge (search edge 3) 4 [3299] 3299 root460Tail4 winner := by
  rw [root460Tail3, scan, if_neg (by decide), root460Candidates3_checked]
  change scan edge (search edge 3) 4 [3299] 3299 root460Tail4
    (if 3 ≤ root460Candidates3.length then search edge 3 [3299, 4817] root460Candidates3 8116 winner else winner) = _
  rw [if_neg (by decide)]
theorem root460Scan4 : scan edge (search edge 3) 4 [3299] 3299 root460Tail4 winner = winner := by rw [root460Tail4, scan, if_pos (by decide)]
theorem root460Scan3 : scan edge (search edge 3) 4 [3299] 3299 root460Tail3 winner = winner :=
  (root460Step3).trans root460Scan4
theorem root460Scan2 : scan edge (search edge 3) 4 [3299] 3299 root460Tail2 winner = winner :=
  (root460Step2).trans root460Scan3
theorem root460Scan1 : scan edge (search edge 3) 4 [3299] 3299 root460Tail1 winner = winner :=
  (root460Step1).trans root460Scan2
theorem root460Scan0 : scan edge (search edge 3) 4 [3299] 3299 root460Tail0 winner = winner :=
  (root460Step0).trans root460Scan1
set_option maxRecDepth 4096 in
theorem root460Branch : search edge 4 [3299] root460Candidates 3299 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3299] 3299 root460Tail0 winner = winner
  exact root460Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
