import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root553Candidates : List Nat := Data.row4021
set_option maxRecDepth 8192 in
theorem root553Candidates_checked : candidates edge 5 0 4021 tail554 winner.bound = root553Candidates := by
  apply candidates_eq_row edge 5 0 4021 tail554 root553Candidates winner.bound tail554_ordered
  · decide
  · decide
  · funext q
    rfl
def root553Tail30 : List Nat := []
def root553Tail29 : List Nat := 21523 :: root553Tail30
def root553Tail28 : List Nat := 20899 :: root553Tail29
def root553Tail27 : List Nat := 20857 :: root553Tail28
def root553Tail26 : List Nat := 18973 :: root553Tail27
def root553Tail25 : List Nat := 18097 :: root553Tail26
def root553Tail24 : List Nat := 17929 :: root553Tail25
def root553Tail23 : List Nat := 17923 :: root553Tail24
def root553Tail22 : List Nat := 16921 :: root553Tail23
def root553Tail21 : List Nat := 16831 :: root553Tail22
def root553Tail20 : List Nat := 16651 :: root553Tail21
def root553Tail19 : List Nat := 15313 :: root553Tail20
def root553Tail18 : List Nat := 14737 :: root553Tail19
def root553Tail17 : List Nat := 14143 :: root553Tail18
def root553Tail16 : List Nat := 13879 :: root553Tail17
def root553Tail15 : List Nat := 13417 :: root553Tail16
def root553Tail14 : List Nat := 12163 :: root553Tail15
def root553Tail13 : List Nat := 12043 :: root553Tail14
def root553Tail12 : List Nat := 12037 :: root553Tail13
def root553Tail11 : List Nat := 11923 :: root553Tail12
def root553Tail10 : List Nat := 10987 :: root553Tail11
def root553Tail9 : List Nat := 10453 :: root553Tail10
def root553Tail8 : List Nat := 9547 :: root553Tail9
def root553Tail7 : List Nat := 9349 :: root553Tail8
def root553Tail6 : List Nat := 7537 :: root553Tail7
def root553Tail5 : List Nat := 7207 :: root553Tail6
def root553Tail4 : List Nat := 6733 :: root553Tail5
def root553Tail3 : List Nat := 5563 :: root553Tail4
def root553Tail2 : List Nat := 5347 :: root553Tail3
def root553Tail1 : List Nat := 4903 :: root553Tail2
def root553Tail0 : List Nat := 4441 :: root553Tail1
def root553Candidates0 : List Nat := [5347, 9547]
set_option maxRecDepth 8192 in
theorem root553Candidates0_checked : candidates edge 4 4021 4441 root553Tail1 winner.bound = root553Candidates0 := by
  unfold candidates
  have row : edge 4441 = (SortedNatTree.contains FastRows.row4441) := by
    change Data.row4441.contains = (SortedNatTree.contains FastRows.row4441)
    exact FastRows.row4441_checked.symm
  rw [row]
  decide
theorem root553Step0 : scan edge (search edge 3) 4 [4021] 4021 root553Tail0 winner =
    scan edge (search edge 3) 4 [4021] 4021 root553Tail1 winner := by
  rw [root553Tail0, scan, if_neg (by decide), root553Candidates0_checked]
  change scan edge (search edge 3) 4 [4021] 4021 root553Tail1
    (if 3 ≤ root553Candidates0.length then search edge 3 [4021, 4441] root553Candidates0 8462 winner else winner) = _
  rw [if_neg (by decide)]
def root553Candidates1 : List Nat := [9349]
set_option maxRecDepth 8192 in
theorem root553Candidates1_checked : candidates edge 4 4021 4903 root553Tail2 winner.bound = root553Candidates1 := by
  unfold candidates
  have row : edge 4903 = (SortedNatTree.contains FastRows.row4903) := by
    change Data.row4903.contains = (SortedNatTree.contains FastRows.row4903)
    exact FastRows.row4903_checked.symm
  rw [row]
  decide
theorem root553Step1 : scan edge (search edge 3) 4 [4021] 4021 root553Tail1 winner =
    scan edge (search edge 3) 4 [4021] 4021 root553Tail2 winner := by
  rw [root553Tail1, scan, if_neg (by decide), root553Candidates1_checked]
  change scan edge (search edge 3) 4 [4021] 4021 root553Tail2
    (if 3 ≤ root553Candidates1.length then search edge 3 [4021, 4903] root553Candidates1 8924 winner else winner) = _
  rw [if_neg (by decide)]
def root553Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root553Candidates2_checked : candidates edge 4 4021 5347 root553Tail3 winner.bound = root553Candidates2 := by
  unfold candidates
  have row : edge 5347 = (SortedNatTree.contains FastRows.row5347) := by
    change Data.row5347.contains = (SortedNatTree.contains FastRows.row5347)
    exact FastRows.row5347_checked.symm
  rw [row]
  decide
theorem root553Step2 : scan edge (search edge 3) 4 [4021] 4021 root553Tail2 winner =
    scan edge (search edge 3) 4 [4021] 4021 root553Tail3 winner := by
  rw [root553Tail2, scan, if_neg (by decide), root553Candidates2_checked]
  change scan edge (search edge 3) 4 [4021] 4021 root553Tail3
    (if 3 ≤ root553Candidates2.length then search edge 3 [4021, 5347] root553Candidates2 9368 winner else winner) = _
  rw [if_neg (by decide)]
theorem root553Scan3 : scan edge (search edge 3) 4 [4021] 4021 root553Tail3 winner = winner := by rw [root553Tail3, scan, if_pos (by decide)]
theorem root553Scan2 : scan edge (search edge 3) 4 [4021] 4021 root553Tail2 winner = winner :=
  (root553Step2).trans root553Scan3
theorem root553Scan1 : scan edge (search edge 3) 4 [4021] 4021 root553Tail1 winner = winner :=
  (root553Step1).trans root553Scan2
theorem root553Scan0 : scan edge (search edge 3) 4 [4021] 4021 root553Tail0 winner = winner :=
  (root553Step0).trans root553Scan1
set_option maxRecDepth 4096 in
theorem root553Branch : search edge 4 [4021] root553Candidates 4021 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4021] 4021 root553Tail0 winner = winner
  exact root553Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
