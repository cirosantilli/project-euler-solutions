import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root453Candidates : List Nat := Data.row3221
set_option maxRecDepth 8192 in
theorem root453Candidates_checked : candidates edge 5 0 3221 tail454 winner.bound = root453Candidates := by
  apply candidates_eq_row edge 5 0 3221 tail454 root453Candidates winner.bound tail454_ordered
  · decide
  · decide
  · funext q
    rfl
def root453Tail34 : List Nat := []
def root453Tail33 : List Nat := 22247 :: root453Tail34
def root453Tail32 : List Nat := 21143 :: root453Tail33
def root453Tail31 : List Nat := 20849 :: root453Tail32
def root453Tail30 : List Nat := 19841 :: root453Tail31
def root453Tail29 : List Nat := 19697 :: root453Tail30
def root453Tail28 : List Nat := 18959 :: root453Tail29
def root453Tail27 : List Nat := 18803 :: root453Tail28
def root453Tail26 : List Nat := 18743 :: root453Tail27
def root453Tail25 : List Nat := 18521 :: root453Tail26
def root453Tail24 : List Nat := 18143 :: root453Tail25
def root453Tail23 : List Nat := 17669 :: root453Tail24
def root453Tail22 : List Nat := 16631 :: root453Tail23
def root453Tail21 : List Nat := 16547 :: root453Tail22
def root453Tail20 : List Nat := 15641 :: root453Tail21
def root453Tail19 : List Nat := 15467 :: root453Tail20
def root453Tail18 : List Nat := 14447 :: root453Tail19
def root453Tail17 : List Nat := 13931 :: root453Tail18
def root453Tail16 : List Nat := 13877 :: root453Tail17
def root453Tail15 : List Nat := 12239 :: root453Tail16
def root453Tail14 : List Nat := 10259 :: root453Tail15
def root453Tail13 : List Nat := 8609 :: root453Tail14
def root453Tail12 : List Nat := 8429 :: root453Tail13
def root453Tail11 : List Nat := 7727 :: root453Tail12
def root453Tail10 : List Nat := 7643 :: root453Tail11
def root453Tail9 : List Nat := 7247 :: root453Tail10
def root453Tail8 : List Nat := 6311 :: root453Tail9
def root453Tail7 : List Nat := 6269 :: root453Tail8
def root453Tail6 : List Nat := 4817 :: root453Tail7
def root453Tail5 : List Nat := 4547 :: root453Tail6
def root453Tail4 : List Nat := 4409 :: root453Tail5
def root453Tail3 : List Nat := 4349 :: root453Tail4
def root453Tail2 : List Nat := 3851 :: root453Tail3
def root453Tail1 : List Nat := 3767 :: root453Tail2
def root453Tail0 : List Nat := 3761 :: root453Tail1
def root453Candidates0 : List Nat := [3767]
set_option maxRecDepth 8192 in
theorem root453Candidates0_checked : candidates edge 4 3221 3761 root453Tail1 winner.bound = root453Candidates0 := by
  unfold candidates
  have row : edge 3761 = (SortedNatTree.contains FastRows.row3761) := by
    change Data.row3761.contains = (SortedNatTree.contains FastRows.row3761)
    exact FastRows.row3761_checked.symm
  rw [row]
  decide
theorem root453Step0 : scan edge (search edge 3) 4 [3221] 3221 root453Tail0 winner =
    scan edge (search edge 3) 4 [3221] 3221 root453Tail1 winner := by
  rw [root453Tail0, scan, if_neg (by decide), root453Candidates0_checked]
  change scan edge (search edge 3) 4 [3221] 3221 root453Tail1
    (if 3 ≤ root453Candidates0.length then search edge 3 [3221, 3761] root453Candidates0 6982 winner else winner) = _
  rw [if_neg (by decide)]
def root453Candidates1 : List Nat := []
set_option maxRecDepth 8192 in
theorem root453Candidates1_checked : candidates edge 4 3221 3767 root453Tail2 winner.bound = root453Candidates1 := by
  unfold candidates
  have row : edge 3767 = (SortedNatTree.contains FastRows.row3767) := by
    change Data.row3767.contains = (SortedNatTree.contains FastRows.row3767)
    exact FastRows.row3767_checked.symm
  rw [row]
  decide
theorem root453Step1 : scan edge (search edge 3) 4 [3221] 3221 root453Tail1 winner =
    scan edge (search edge 3) 4 [3221] 3221 root453Tail2 winner := by
  rw [root453Tail1, scan, if_neg (by decide), root453Candidates1_checked]
  change scan edge (search edge 3) 4 [3221] 3221 root453Tail2
    (if 3 ≤ root453Candidates1.length then search edge 3 [3221, 3767] root453Candidates1 6988 winner else winner) = _
  rw [if_neg (by decide)]
def root453Candidates2 : List Nat := [13877]
set_option maxRecDepth 8192 in
theorem root453Candidates2_checked : candidates edge 4 3221 3851 root453Tail3 winner.bound = root453Candidates2 := by
  unfold candidates
  have row : edge 3851 = (SortedNatTree.contains FastRows.row3851) := by
    change Data.row3851.contains = (SortedNatTree.contains FastRows.row3851)
    exact FastRows.row3851_checked.symm
  rw [row]
  decide
theorem root453Step2 : scan edge (search edge 3) 4 [3221] 3221 root453Tail2 winner =
    scan edge (search edge 3) 4 [3221] 3221 root453Tail3 winner := by
  rw [root453Tail2, scan, if_neg (by decide), root453Candidates2_checked]
  change scan edge (search edge 3) 4 [3221] 3221 root453Tail3
    (if 3 ≤ root453Candidates2.length then search edge 3 [3221, 3851] root453Candidates2 7072 winner else winner) = _
  rw [if_neg (by decide)]
def root453Candidates3 : List Nat := []
set_option maxRecDepth 8192 in
theorem root453Candidates3_checked : candidates edge 4 3221 4349 root453Tail4 winner.bound = root453Candidates3 := by
  unfold candidates
  have row : edge 4349 = (SortedNatTree.contains FastRows.row4349) := by
    change Data.row4349.contains = (SortedNatTree.contains FastRows.row4349)
    exact FastRows.row4349_checked.symm
  rw [row]
  decide
theorem root453Step3 : scan edge (search edge 3) 4 [3221] 3221 root453Tail3 winner =
    scan edge (search edge 3) 4 [3221] 3221 root453Tail4 winner := by
  rw [root453Tail3, scan, if_neg (by decide), root453Candidates3_checked]
  change scan edge (search edge 3) 4 [3221] 3221 root453Tail4
    (if 3 ≤ root453Candidates3.length then search edge 3 [3221, 4349] root453Candidates3 7570 winner else winner) = _
  rw [if_neg (by decide)]
def root453Candidates4 : List Nat := [10259]
set_option maxRecDepth 8192 in
theorem root453Candidates4_checked : candidates edge 4 3221 4409 root453Tail5 winner.bound = root453Candidates4 := by
  unfold candidates
  have row : edge 4409 = (SortedNatTree.contains FastRows.row4409) := by
    change Data.row4409.contains = (SortedNatTree.contains FastRows.row4409)
    exact FastRows.row4409_checked.symm
  rw [row]
  decide
theorem root453Step4 : scan edge (search edge 3) 4 [3221] 3221 root453Tail4 winner =
    scan edge (search edge 3) 4 [3221] 3221 root453Tail5 winner := by
  rw [root453Tail4, scan, if_neg (by decide), root453Candidates4_checked]
  change scan edge (search edge 3) 4 [3221] 3221 root453Tail5
    (if 3 ≤ root453Candidates4.length then search edge 3 [3221, 4409] root453Candidates4 7630 winner else winner) = _
  rw [if_neg (by decide)]
def root453Candidates5 : List Nat := [6311]
set_option maxRecDepth 8192 in
theorem root453Candidates5_checked : candidates edge 4 3221 4547 root453Tail6 winner.bound = root453Candidates5 := by
  unfold candidates
  have row : edge 4547 = (SortedNatTree.contains FastRows.row4547) := by
    change Data.row4547.contains = (SortedNatTree.contains FastRows.row4547)
    exact FastRows.row4547_checked.symm
  rw [row]
  decide
theorem root453Step5 : scan edge (search edge 3) 4 [3221] 3221 root453Tail5 winner =
    scan edge (search edge 3) 4 [3221] 3221 root453Tail6 winner := by
  rw [root453Tail5, scan, if_neg (by decide), root453Candidates5_checked]
  change scan edge (search edge 3) 4 [3221] 3221 root453Tail6
    (if 3 ≤ root453Candidates5.length then search edge 3 [3221, 4547] root453Candidates5 7768 winner else winner) = _
  rw [if_neg (by decide)]
def root453Candidates6 : List Nat := []
set_option maxRecDepth 8192 in
theorem root453Candidates6_checked : candidates edge 4 3221 4817 root453Tail7 winner.bound = root453Candidates6 := by
  unfold candidates
  have row : edge 4817 = (SortedNatTree.contains FastRows.row4817) := by
    change Data.row4817.contains = (SortedNatTree.contains FastRows.row4817)
    exact FastRows.row4817_checked.symm
  rw [row]
  decide
theorem root453Step6 : scan edge (search edge 3) 4 [3221] 3221 root453Tail6 winner =
    scan edge (search edge 3) 4 [3221] 3221 root453Tail7 winner := by
  rw [root453Tail6, scan, if_neg (by decide), root453Candidates6_checked]
  change scan edge (search edge 3) 4 [3221] 3221 root453Tail7
    (if 3 ≤ root453Candidates6.length then search edge 3 [3221, 4817] root453Candidates6 8038 winner else winner) = _
  rw [if_neg (by decide)]
theorem root453Scan7 : scan edge (search edge 3) 4 [3221] 3221 root453Tail7 winner = winner := by rw [root453Tail7, scan, if_pos (by decide)]
theorem root453Scan6 : scan edge (search edge 3) 4 [3221] 3221 root453Tail6 winner = winner :=
  (root453Step6).trans root453Scan7
theorem root453Scan5 : scan edge (search edge 3) 4 [3221] 3221 root453Tail5 winner = winner :=
  (root453Step5).trans root453Scan6
theorem root453Scan4 : scan edge (search edge 3) 4 [3221] 3221 root453Tail4 winner = winner :=
  (root453Step4).trans root453Scan5
theorem root453Scan3 : scan edge (search edge 3) 4 [3221] 3221 root453Tail3 winner = winner :=
  (root453Step3).trans root453Scan4
theorem root453Scan2 : scan edge (search edge 3) 4 [3221] 3221 root453Tail2 winner = winner :=
  (root453Step2).trans root453Scan3
theorem root453Scan1 : scan edge (search edge 3) 4 [3221] 3221 root453Tail1 winner = winner :=
  (root453Step1).trans root453Scan2
theorem root453Scan0 : scan edge (search edge 3) 4 [3221] 3221 root453Tail0 winner = winner :=
  (root453Step0).trans root453Scan1
set_option maxRecDepth 4096 in
theorem root453Branch : search edge 4 [3221] root453Candidates 3221 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3221] 3221 root453Tail0 winner = winner
  exact root453Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
