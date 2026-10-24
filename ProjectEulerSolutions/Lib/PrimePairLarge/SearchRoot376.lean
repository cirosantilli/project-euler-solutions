import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root376Candidates : List Nat := Data.row2609
set_option maxRecDepth 8192 in
theorem root376Candidates_checked : candidates edge 5 0 2609 tail377 winner.bound = root376Candidates := by
  apply candidates_eq_row edge 5 0 2609 tail377 root376Candidates winner.bound tail377_ordered
  · decide
  · decide
  · funext q
    rfl
def root376Tail39 : List Nat := []
def root376Tail38 : List Nat := 23099 :: root376Tail39
def root376Tail37 : List Nat := 22973 :: root376Tail38
def root376Tail36 : List Nat := 22433 :: root376Tail37
def root376Tail35 : List Nat := 22247 :: root376Tail36
def root376Tail34 : List Nat := 22157 :: root376Tail35
def root376Tail33 : List Nat := 21803 :: root376Tail34
def root376Tail32 : List Nat := 21587 :: root376Tail33
def root376Tail31 : List Nat := 20873 :: root376Tail32
def root376Tail30 : List Nat := 19937 :: root376Tail31
def root376Tail29 : List Nat := 19073 :: root376Tail30
def root376Tail28 : List Nat := 18131 :: root376Tail29
def root376Tail27 : List Nat := 17837 :: root376Tail28
def root376Tail26 : List Nat := 16829 :: root376Tail27
def root376Tail25 : List Nat := 16127 :: root376Tail26
def root376Tail24 : List Nat := 15107 :: root376Tail25
def root376Tail23 : List Nat := 14891 :: root376Tail24
def root376Tail22 : List Nat := 14387 :: root376Tail23
def root376Tail21 : List Nat := 14033 :: root376Tail22
def root376Tail20 : List Nat := 13613 :: root376Tail21
def root376Tail19 : List Nat := 13007 :: root376Tail20
def root376Tail18 : List Nat := 11117 :: root376Tail19
def root376Tail17 : List Nat := 10313 :: root376Tail18
def root376Tail16 : List Nat := 10037 :: root376Tail17
def root376Tail15 : List Nat := 8297 :: root376Tail16
def root376Tail14 : List Nat := 8273 :: root376Tail15
def root376Tail13 : List Nat := 8069 :: root376Tail14
def root376Tail12 : List Nat := 7829 :: root376Tail13
def root376Tail11 : List Nat := 7451 :: root376Tail12
def root376Tail10 : List Nat := 6599 :: root376Tail11
def root376Tail9 : List Nat := 6569 :: root376Tail10
def root376Tail8 : List Nat := 6197 :: root376Tail9
def root376Tail7 : List Nat := 5651 :: root376Tail8
def root376Tail6 : List Nat := 4973 :: root376Tail7
def root376Tail5 : List Nat := 4931 :: root376Tail6
def root376Tail4 : List Nat := 4241 :: root376Tail5
def root376Tail3 : List Nat := 3803 :: root376Tail4
def root376Tail2 : List Nat := 3491 :: root376Tail3
def root376Tail1 : List Nat := 3257 :: root376Tail2
def root376Tail0 : List Nat := 2777 :: root376Tail1
def root376Candidates0 : List Nat := [4973, 14387, 19073]
set_option maxRecDepth 8192 in
theorem root376Candidates0_checked : candidates edge 4 2609 2777 root376Tail1 winner.bound = root376Candidates0 := by
  unfold candidates
  have row : edge 2777 = (SortedNatTree.contains FastRows.row2777) := by
    change Data.row2777.contains = (SortedNatTree.contains FastRows.row2777)
    exact FastRows.row2777_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root376Child0 : search edge 3 [2609, 2777] root376Candidates0 5386 winner = winner := by decide
theorem root376Step0 : scan edge (search edge 3) 4 [2609] 2609 root376Tail0 winner =
    scan edge (search edge 3) 4 [2609] 2609 root376Tail1 winner := by
  rw [root376Tail0, scan, if_neg (by decide), root376Candidates0_checked]
  change scan edge (search edge 3) 4 [2609] 2609 root376Tail1
    (if 3 ≤ root376Candidates0.length then search edge 3 [2609, 2777] root376Candidates0 5386 winner else winner) = _
  rw [if_pos (by decide), root376Child0]
def root376Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root376Candidates1_checked : candidates edge 4 2609 3257 root376Tail2 winner.bound = root376Candidates1 := by
  unfold candidates
  have row : edge 3257 = (SortedNatTree.contains FastRows.row3257) := by
    change Data.row3257.contains = (SortedNatTree.contains FastRows.row3257)
    exact FastRows.row3257_checked.symm
  rw [row]
  decide
theorem root376Step1 : scan edge (search edge 3) 4 [2609] 2609 root376Tail1 winner =
    scan edge (search edge 3) 4 [2609] 2609 root376Tail2 winner := by
  rw [root376Tail1, scan, if_neg (by decide), root376Candidates1_checked]
  change scan edge (search edge 3) 4 [2609] 2609 root376Tail2
    (if 3 ≤ root376Candidates1.length then search edge 3 [2609, 3257] root376Candidates1 5866 winner else winner) = _
  rw [if_neg (by decide)]
def root376Candidates2 : List Nat := [4931]
set_option maxRecDepth 8192 in
theorem root376Candidates2_checked : candidates edge 4 2609 3491 root376Tail3 winner.bound = root376Candidates2 := by
  unfold candidates
  have row : edge 3491 = (SortedNatTree.contains FastRows.row3491) := by
    change Data.row3491.contains = (SortedNatTree.contains FastRows.row3491)
    exact FastRows.row3491_checked.symm
  rw [row]
  decide
theorem root376Step2 : scan edge (search edge 3) 4 [2609] 2609 root376Tail2 winner =
    scan edge (search edge 3) 4 [2609] 2609 root376Tail3 winner := by
  rw [root376Tail2, scan, if_neg (by decide), root376Candidates2_checked]
  change scan edge (search edge 3) 4 [2609] 2609 root376Tail3
    (if 3 ≤ root376Candidates2.length then search edge 3 [2609, 3491] root376Candidates2 6100 winner else winner) = _
  rw [if_neg (by decide)]
def root376Candidates3 : List Nat := [8069, 16127]
set_option maxRecDepth 8192 in
theorem root376Candidates3_checked : candidates edge 4 2609 3803 root376Tail4 winner.bound = root376Candidates3 := by
  unfold candidates
  have row : edge 3803 = (SortedNatTree.contains FastRows.row3803) := by
    change Data.row3803.contains = (SortedNatTree.contains FastRows.row3803)
    exact FastRows.row3803_checked.symm
  rw [row]
  decide
theorem root376Step3 : scan edge (search edge 3) 4 [2609] 2609 root376Tail3 winner =
    scan edge (search edge 3) 4 [2609] 2609 root376Tail4 winner := by
  rw [root376Tail3, scan, if_neg (by decide), root376Candidates3_checked]
  change scan edge (search edge 3) 4 [2609] 2609 root376Tail4
    (if 3 ≤ root376Candidates3.length then search edge 3 [2609, 3803] root376Candidates3 6412 winner else winner) = _
  rw [if_neg (by decide)]
def root376Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root376Candidates4_checked : candidates edge 4 2609 4241 root376Tail5 winner.bound = root376Candidates4 := by
  unfold candidates
  have row : edge 4241 = (SortedNatTree.contains FastRows.row4241) := by
    change Data.row4241.contains = (SortedNatTree.contains FastRows.row4241)
    exact FastRows.row4241_checked.symm
  rw [row]
  decide
theorem root376Step4 : scan edge (search edge 3) 4 [2609] 2609 root376Tail4 winner =
    scan edge (search edge 3) 4 [2609] 2609 root376Tail5 winner := by
  rw [root376Tail4, scan, if_neg (by decide), root376Candidates4_checked]
  change scan edge (search edge 3) 4 [2609] 2609 root376Tail5
    (if 3 ≤ root376Candidates4.length then search edge 3 [2609, 4241] root376Candidates4 6850 winner else winner) = _
  rw [if_neg (by decide)]
def root376Candidates5 : List Nat := []
set_option maxRecDepth 8192 in
theorem root376Candidates5_checked : candidates edge 4 2609 4931 root376Tail6 winner.bound = root376Candidates5 := by
  unfold candidates
  have row : edge 4931 = (SortedNatTree.contains FastRows.row4931) := by
    change Data.row4931.contains = (SortedNatTree.contains FastRows.row4931)
    exact FastRows.row4931_checked.symm
  rw [row]
  decide
theorem root376Step5 : scan edge (search edge 3) 4 [2609] 2609 root376Tail5 winner =
    scan edge (search edge 3) 4 [2609] 2609 root376Tail6 winner := by
  rw [root376Tail5, scan, if_neg (by decide), root376Candidates5_checked]
  change scan edge (search edge 3) 4 [2609] 2609 root376Tail6
    (if 3 ≤ root376Candidates5.length then search edge 3 [2609, 4931] root376Candidates5 7540 winner else winner) = _
  rw [if_neg (by decide)]
def root376Candidates6 : List Nat := [6569, 7829]
set_option maxRecDepth 8192 in
theorem root376Candidates6_checked : candidates edge 4 2609 4973 root376Tail7 winner.bound = root376Candidates6 := by
  unfold candidates
  have row : edge 4973 = (SortedNatTree.contains FastRows.row4973) := by
    change Data.row4973.contains = (SortedNatTree.contains FastRows.row4973)
    exact FastRows.row4973_checked.symm
  rw [row]
  decide
theorem root376Step6 : scan edge (search edge 3) 4 [2609] 2609 root376Tail6 winner =
    scan edge (search edge 3) 4 [2609] 2609 root376Tail7 winner := by
  rw [root376Tail6, scan, if_neg (by decide), root376Candidates6_checked]
  change scan edge (search edge 3) 4 [2609] 2609 root376Tail7
    (if 3 ≤ root376Candidates6.length then search edge 3 [2609, 4973] root376Candidates6 7582 winner else winner) = _
  rw [if_neg (by decide)]
def root376Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root376Candidates7_checked : candidates edge 4 2609 5651 root376Tail8 winner.bound = root376Candidates7 := by
  unfold candidates
  have row : edge 5651 = (SortedNatTree.contains FastRows.row5651) := by
    change Data.row5651.contains = (SortedNatTree.contains FastRows.row5651)
    exact FastRows.row5651_checked.symm
  rw [row]
  decide
theorem root376Step7 : scan edge (search edge 3) 4 [2609] 2609 root376Tail7 winner =
    scan edge (search edge 3) 4 [2609] 2609 root376Tail8 winner := by
  rw [root376Tail7, scan, if_neg (by decide), root376Candidates7_checked]
  change scan edge (search edge 3) 4 [2609] 2609 root376Tail8
    (if 3 ≤ root376Candidates7.length then search edge 3 [2609, 5651] root376Candidates7 8260 winner else winner) = _
  rw [if_neg (by decide)]
theorem root376Scan8 : scan edge (search edge 3) 4 [2609] 2609 root376Tail8 winner = winner := by rw [root376Tail8, scan, if_pos (by decide)]
theorem root376Scan7 : scan edge (search edge 3) 4 [2609] 2609 root376Tail7 winner = winner :=
  (root376Step7).trans root376Scan8
theorem root376Scan6 : scan edge (search edge 3) 4 [2609] 2609 root376Tail6 winner = winner :=
  (root376Step6).trans root376Scan7
theorem root376Scan5 : scan edge (search edge 3) 4 [2609] 2609 root376Tail5 winner = winner :=
  (root376Step5).trans root376Scan6
theorem root376Scan4 : scan edge (search edge 3) 4 [2609] 2609 root376Tail4 winner = winner :=
  (root376Step4).trans root376Scan5
theorem root376Scan3 : scan edge (search edge 3) 4 [2609] 2609 root376Tail3 winner = winner :=
  (root376Step3).trans root376Scan4
theorem root376Scan2 : scan edge (search edge 3) 4 [2609] 2609 root376Tail2 winner = winner :=
  (root376Step2).trans root376Scan3
theorem root376Scan1 : scan edge (search edge 3) 4 [2609] 2609 root376Tail1 winner = winner :=
  (root376Step1).trans root376Scan2
theorem root376Scan0 : scan edge (search edge 3) 4 [2609] 2609 root376Tail0 winner = winner :=
  (root376Step0).trans root376Scan1
set_option maxRecDepth 4096 in
theorem root376Branch : search edge 4 [2609] root376Candidates 2609 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2609] 2609 root376Tail0 winner = winner
  exact root376Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
