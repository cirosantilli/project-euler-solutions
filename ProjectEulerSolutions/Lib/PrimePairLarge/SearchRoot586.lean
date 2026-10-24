import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root586Candidates : List Nat := Data.row4289
set_option maxRecDepth 8192 in
theorem root586Candidates_checked : candidates edge 5 0 4289 tail587 winner.bound = root586Candidates := by
  apply candidates_eq_row edge 5 0 4289 tail587 root586Candidates winner.bound tail587_ordered
  · decide
  · decide
  · funext q
    rfl
def root586Tail37 : List Nat := []
def root586Tail36 : List Nat := 21401 :: root586Tail37
def root586Tail35 : List Nat := 20903 :: root586Tail36
def root586Tail34 : List Nat := 20753 :: root586Tail35
def root586Tail33 : List Nat := 20477 :: root586Tail34
def root586Tail32 : List Nat := 19949 :: root586Tail33
def root586Tail31 : List Nat := 18233 :: root586Tail32
def root586Tail30 : List Nat := 17387 :: root586Tail31
def root586Tail29 : List Nat := 16427 :: root586Tail30
def root586Tail28 : List Nat := 16091 :: root586Tail29
def root586Tail27 : List Nat := 16073 :: root586Tail28
def root586Tail26 : List Nat := 15737 :: root586Tail27
def root586Tail25 : List Nat := 14033 :: root586Tail26
def root586Tail24 : List Nat := 13721 :: root586Tail25
def root586Tail23 : List Nat := 13151 :: root586Tail24
def root586Tail22 : List Nat := 11933 :: root586Tail23
def root586Tail21 : List Nat := 11867 :: root586Tail22
def root586Tail20 : List Nat := 11831 :: root586Tail21
def root586Tail19 : List Nat := 11489 :: root586Tail20
def root586Tail18 : List Nat := 11471 :: root586Tail19
def root586Tail17 : List Nat := 11159 :: root586Tail18
def root586Tail16 : List Nat := 10709 :: root586Tail17
def root586Tail15 : List Nat := 10211 :: root586Tail16
def root586Tail14 : List Nat := 9803 :: root586Tail15
def root586Tail13 : List Nat := 8861 :: root586Tail14
def root586Tail12 : List Nat := 8819 :: root586Tail13
def root586Tail11 : List Nat := 8171 :: root586Tail12
def root586Tail10 : List Nat := 8087 :: root586Tail11
def root586Tail9 : List Nat := 7937 :: root586Tail10
def root586Tail8 : List Nat := 7331 :: root586Tail9
def root586Tail7 : List Nat := 6869 :: root586Tail8
def root586Tail6 : List Nat := 6473 :: root586Tail7
def root586Tail5 : List Nat := 6317 :: root586Tail6
def root586Tail4 : List Nat := 5519 :: root586Tail5
def root586Tail3 : List Nat := 5507 :: root586Tail4
def root586Tail2 : List Nat := 4889 :: root586Tail3
def root586Tail1 : List Nat := 4679 :: root586Tail2
def root586Tail0 : List Nat := 4391 :: root586Tail1
def root586Candidates0 : List Nat := [6869, 10211]
set_option maxRecDepth 8192 in
theorem root586Candidates0_checked : candidates edge 4 4289 4391 root586Tail1 winner.bound = root586Candidates0 := by
  unfold candidates
  have row : edge 4391 = (SortedNatTree.contains FastRows.row4391) := by
    change Data.row4391.contains = (SortedNatTree.contains FastRows.row4391)
    exact FastRows.row4391_checked.symm
  rw [row]
  decide
theorem root586Step0 : scan edge (search edge 3) 4 [4289] 4289 root586Tail0 winner =
    scan edge (search edge 3) 4 [4289] 4289 root586Tail1 winner := by
  rw [root586Tail0, scan, if_neg (by decide), root586Candidates0_checked]
  change scan edge (search edge 3) 4 [4289] 4289 root586Tail1
    (if 3 ≤ root586Candidates0.length then search edge 3 [4289, 4391] root586Candidates0 8680 winner else winner) = _
  rw [if_neg (by decide)]
def root586Candidates1 : List Nat := [4889, 8819, 14033, 16091]
set_option maxRecDepth 8192 in
theorem root586Candidates1_checked : candidates edge 4 4289 4679 root586Tail2 winner.bound = root586Candidates1 := by
  unfold candidates
  have row : edge 4679 = (SortedNatTree.contains FastRows.row4679) := by
    change Data.row4679.contains = (SortedNatTree.contains FastRows.row4679)
    exact FastRows.row4679_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root586Child1 : search edge 3 [4289, 4679] root586Candidates1 8968 winner = winner := by decide
theorem root586Step1 : scan edge (search edge 3) 4 [4289] 4289 root586Tail1 winner =
    scan edge (search edge 3) 4 [4289] 4289 root586Tail2 winner := by
  rw [root586Tail1, scan, if_neg (by decide), root586Candidates1_checked]
  change scan edge (search edge 3) 4 [4289] 4289 root586Tail2
    (if 3 ≤ root586Candidates1.length then search edge 3 [4289, 4679] root586Candidates1 8968 winner else winner) = _
  rw [if_pos (by decide), root586Child1]
def root586Candidates2 : List Nat := []
set_option maxRecDepth 8192 in
theorem root586Candidates2_checked : candidates edge 4 4289 4889 root586Tail3 winner.bound = root586Candidates2 := by
  unfold candidates
  have row : edge 4889 = (SortedNatTree.contains FastRows.row4889) := by
    change Data.row4889.contains = (SortedNatTree.contains FastRows.row4889)
    exact FastRows.row4889_checked.symm
  rw [row]
  decide
theorem root586Step2 : scan edge (search edge 3) 4 [4289] 4289 root586Tail2 winner =
    scan edge (search edge 3) 4 [4289] 4289 root586Tail3 winner := by
  rw [root586Tail2, scan, if_neg (by decide), root586Candidates2_checked]
  change scan edge (search edge 3) 4 [4289] 4289 root586Tail3
    (if 3 ≤ root586Candidates2.length then search edge 3 [4289, 4889] root586Candidates2 9178 winner else winner) = _
  rw [if_neg (by decide)]
theorem root586Scan3 : scan edge (search edge 3) 4 [4289] 4289 root586Tail3 winner = winner := by rw [root586Tail3, scan, if_pos (by decide)]
theorem root586Scan2 : scan edge (search edge 3) 4 [4289] 4289 root586Tail2 winner = winner :=
  (root586Step2).trans root586Scan3
theorem root586Scan1 : scan edge (search edge 3) 4 [4289] 4289 root586Tail1 winner = winner :=
  (root586Step1).trans root586Scan2
theorem root586Scan0 : scan edge (search edge 3) 4 [4289] 4289 root586Tail0 winner = winner :=
  (root586Step0).trans root586Scan1
set_option maxRecDepth 4096 in
theorem root586Branch : search edge 4 [4289] root586Candidates 4289 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4289] 4289 root586Tail0 winner = winner
  exact root586Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
